import os
import json
import time
import boto3
from typing import List, Dict, Any, Optional
from pypdf import PdfReader
from docx import Document
from botocore.exceptions import ClientError

class DocumentParser:
    """
    Handles parsing of PDF and DOCX files.
    - PDF: Uses AWS Bedrock (Mistral) for intelligent structural extraction.
    - DOCX: Standard parsing (fallback).
    """

    def __init__(self, region_name: str = "ap-south-1"):
        self.region_name = region_name
        self.max_retries = 3
        # Mistral Large
        self.model_id = "mistral.mistral-large-2402-v1:0"
        
        try:
             self.bedrock_client = boto3.client(
                 service_name='bedrock-runtime',
                 region_name=self.region_name
             )
        except Exception as e:
             print(f"Warning: Bedrock init failed in Parser: {e}")
             self.bedrock_client = None

    def parse_file(self, file_path: str) -> List[Dict]:
        """
        Parses a file and returns a list of chunks/pages.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext == '.pdf':
            # Use Bedrock if available, else standard
            if self.bedrock_client:
                return self._parse_pdf_bedrock(file_path)
            else:
                return self._parse_pdf_standard(file_path)
        elif ext == '.docx':
            return self._parse_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _parse_pdf_standard(self, file_path: str) -> List[Dict]:
        """Fallback standard parser if Bedrock fails or not init."""
        results = []
        try:
            reader = PdfReader(file_path)
            source_name = os.path.basename(file_path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    results.append({
                        'text': text.strip(),
                        'page': i + 1,
                        'source': source_name
                    })
        except Exception as e:
            print(f"Error parsing PDF (Standard) {file_path}: {e}")
        return results

    def _parse_pdf_bedrock(self, file_path: str) -> List[Dict]:
        """
        Uses AWS Bedrock (Mistral) to extract heavily structured text.
        For consistency with the pipeline, we return page-level chunks,
        but enriched with the structure the LLM detects.
        """
        results = []
        try:
            source_name = os.path.basename(file_path)
            
            # Simple approach: Extract text first, then clean/structure per page with LLM
            # Just passing full PDF text to LLM context window might be too large for multi-page docs.
            # We will iterate pages and cleanup/structure each page.
            
            reader = PdfReader(file_path)
            for i, page in enumerate(reader.pages):
                raw_text = page.extract_text()
                if not raw_text.strip():
                    continue
                
                # Intelligent Cleanup via Bedrock
                structured_text = self._invoke_bedrock_cleanup(raw_text)
                
                results.append({
                    'text': structured_text, # High quality text
                    'page': i + 1,
                    'source': source_name
                })
                
        except Exception as e:
            print(f"Error parsing PDF (Bedrock) {file_path}: {e}")
            # Fallback
            return self._parse_pdf_standard(file_path)
            
        return results

    def _invoke_bedrock_cleanup(self, raw_text: str) -> str:
        prompt = f"""<s>[INST] You are an expert document parser.
        Clean up the following raw PDF text. 
        Fix extraction errors, merge broken lines, and preserve the layout/tables as Markdown text.
        
        CRITICAL: OUTPUT ONLY THE CLEANED TEXT. NO "HERE IS THE TEXT" OR "SURE".
        DO NOT SUMMARIZE. PROHIBITED TO OMIT DATA.
        
        Raw Text:
        {raw_text[:8000]} 
        [/INST]"""
        
        try:
            body = {
                "prompt": prompt,
                "max_tokens": 4096,
                "temperature": 0.0
            }
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )
            response_body = json.loads(response['body'].read())
            
            if 'outputs' in response_body:
                 return response_body['outputs'][0]['text'].strip()
            elif 'choices' in response_body:
                 return response_body['choices'][0]['message']['content'].strip()
            else:
                 return raw_text # Fail safe
                 
        except Exception as e:
            print(f"Bedrock Parsing Error: {e}")
            return raw_text

    def _parse_docx(self, file_path: str) -> List[Dict]:
        results = []
        try:
            doc = Document(file_path)
            source_name = os.path.basename(file_path)
            
            full_text = []
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text.strip())
            
            results.append({
                'text': "\n".join(full_text),
                'page': 1, 
                'source': source_name
            })
        except Exception as e:
            print(f"Error parsing DOCX {file_path}: {e}")
            raise
        return results

if __name__ == "__main__":
    pass
