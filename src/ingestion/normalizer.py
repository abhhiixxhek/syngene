import boto3
import json
import os
from typing import List, Dict

class RequirementNormalizer:
    """
    Decomposes raw text clauses into atomic, verifiable requirements using AWS Bedrock (Mistral).
    """
    
    def __init__(self, region_name: str = "ap-south-1"):
        self.region_name = region_name
        self.max_retries = 3
        # Using the Model ID provided by user or standard Bedrock one.
        # User provided: "mistral.mistral-large-3-675b-instruct"
        # Standard Bedrock: "mistral.mistral-large-2402-v1:0"
        self.model_id = "mistral.mistral-large-2402-v1:0" 
        
        try:
            self.bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=self.region_name
            )
            print(f"✅ Bedrock Client Initialized (Region: {self.region_name}, Model: {self.model_id})")
        except Exception as e:
            print(f"❌ Failed to init Bedrock: {e}")
            raise e

    def normalize_text(self, text_chunk: str, source_meta: Dict) -> List[Dict]:
        """
        Takes a chunk of text and returns a list of atomic requirements.
        """
        prompt = self._build_prompt(text_chunk)
        
        try:
            response_text = self._invoke_bedrock(prompt)
            requirements = self._parse_response(response_text)
            
            # Enrich with metadata
            atomic_reqs = []
            for i, req in enumerate(requirements):
                # Mapping user's new schema keys to internal keys
                result_text = req.get('requirement_text', '') 
                severity = req.get('severity', 'SHOULD')
                
                if not result_text and isinstance(req, str):
                     result_text = req
                     
                atomic_reqs.append({
                    'requirement_id': f"{source_meta.get('source', 'DOC')}-{source_meta.get('page', 0)}-{i+1}",
                    'full_requirement_text': result_text,
                    'original_text': text_chunk,
                    'mandatory_level': severity,
                    'source_document': source_meta.get('source', ''),
                    'page_number': source_meta.get('page', 0)
                })
            return atomic_reqs
            
        except Exception as e:
            print(f"Error normalizing text: {e}")
            return [{
                'requirement_id': f"{source_meta.get('source', 'DOC')}-{source_meta.get('page', 0)}-FAIL",
                'full_requirement_text': f"FAILED TO NORMALIZE: {text_chunk[:50]}...",
                'original_text': text_chunk,
                'mandatory_level': 'UNKNOWN',
                'source_document': source_meta.get('source', ''),
                'page_number': source_meta.get('page', 0)
            }]

    def _build_prompt(self, text: str) -> str:
        # User Provided PROMPT TEMPLATE 1
        return f"""<s>[INST] You are a regulatory compliance analyst.
        Your task is to decompose reference text into atomic, auditable obligations.
        
        Reference Text:
        \"""
        {text}
        \"""
        
        Instructions:
        - Split the text into ATOMIC REQUIREMENTS.
        - One requirement = one independently auditable obligation.
        - Do NOT merge multiple obligations.
        - Do NOT split a single obligation unnecessarily.
        - Preserve original wording (no paraphrasing).
        - If normalization is ambiguous, return NORMALIZATION_FAILED.
        - For lists (e.g. "education, training, experience"), SPLIT into separate requirements.
        - For combined actions (e.g. "maintain documentation demonstrating..."), KEEP together.
        
        Return JSON only:
        [
          {{
            "requirement_text": "verbatim requirement",
            "severity": "MUST | SHOULD | MAY"
          }}
        ] [/INST]"""

    def _invoke_bedrock(self, prompt: str) -> str:
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                # Mistral Large on Bedrock
                body = {
                    "prompt": prompt,
                    "max_tokens": 4096,
                    "temperature": 0.1
                }
                
                response = self.bedrock_client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(body)
                )
                
                response_body = json.loads(response['body'].read())
                
                # Mistral output parsing
                # Standard Bedrock Mistral output: {"outputs": [{"text": "..."}]}
                if 'outputs' in response_body:
                     return response_body['outputs'][0]['text']
                # Or sometimes just the text or choices depending on exact model version
                elif 'choices' in response_body:
                     return response_body['choices'][0]['message']['content']
                else:
                     return str(response_body)

            except Exception as e:
                print(f"Bedrock Error (Attempt {retry_count+1}): {e}")
                import time
                if "ThrottlingException" in str(e):
                    time.sleep(5 * (retry_count + 1))
                else:
                    time.sleep(2)
                retry_count += 1
        
        raise Exception("Max retries exceeded for Bedrock invocation.")

    def _parse_response(self, response_text: str) -> List[Dict]:
        try:
            # Clean md code blocks
            clean_text = response_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()
            
            # Find JSON if wrapped in text
            start = clean_text.find('[')
            end = clean_text.rfind(']') + 1
            if start != -1 and end > start:
                 clean_text = clean_text[start:end]
            
            return json.loads(clean_text)
        except Exception as e:
            print(f"JSON Parse Error: {e}")
            return []

if __name__ == "__main__":
    # Test stub
    pass
