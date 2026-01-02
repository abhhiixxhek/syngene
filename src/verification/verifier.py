import json
import boto3
import os
import numpy as np
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from ..ingestion.parser import DocumentParser
from ..ingestion.embedder import RequirementEmbedder

# Load env
load_dotenv()

class SOPVerifier:
    """
    The Runtime Engine.
    1. Loads Reference Index from S3/Local.
    2. Ingests User SOP into ephemeral vector store.
    3. Performs Reverse-Check (Reference -> SOP).
    4. Invokes LLM (AWS Bedrock Mistral) for gaps.
    """
    
    def __init__(self, bucket_name: str, region_name: str = "ap-south-1"):
        self.bucket_name = bucket_name
        self.region_name = region_name
        
        # S3 Init
        if bucket_name != "local":
            self.s3_client = boto3.client('s3', region_name=region_name)
        else:
            self.s3_client = None
            
        # Bedrock Init
        try:
             self.bedrock_client = boto3.client(
                 service_name='bedrock-runtime',
                 region_name=self.region_name
             )
             # User specified model logic
             self.model_id = "mistral.mistral-large-2402-v1:0"
             print(f"✅ Bedrock Client Initialized (Region: {self.region_name}, Model: {self.model_id})")
        except Exception as e:
             print(f"❌ Failed to init Bedrock: {e}")
             self.bedrock_client = None

        # Re-use components
        self.parser = DocumentParser()
        self.embedder = RequirementEmbedder(region_name=region_name)
        
        # State
        self.reference_index: List[Dict] = []
        self.sop_store: List[Dict] = []
        self.sop_embeddings_matrix = None

    def load_reference_index(self, key: str = "reference_index.json"):
        """
        Loads the pre-computed reference requirements from S3 or local disk.
        """
        try:
            if self.bucket_name == "local":
                print(f"Loading reference index from local file: {key}...")
                with open(key, 'r', encoding='utf-8') as f:
                    self.reference_index = json.load(f)
            else:
                print(f"Loading reference index from s3://{self.bucket_name}/{key}...")
                response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
                content = response['Body'].read().decode('utf-8')
                self.reference_index = json.loads(content)
                
            print(f"Loaded {len(self.reference_index)} reference requirements.")
        except Exception as e:
            print(f"Error loading reference index: {e}")
            raise

    def ingest_sop(self, file_path: str):
        """
        Ingests the SOP at runtime.
        Parses -> Chunks -> Embeds -> Builds Matrix.
        """
        print(f"Ingesting SOP: {file_path}")
        raw_chunks = self.parser.parse_file(file_path)
        
        sop_items = []
        for chunk in raw_chunks:
            sop_items.append({
                'full_requirement_text': chunk['text'], 
                'text': chunk['text'],
                'page': chunk['page'],
                'source': chunk['source']
            })
            
        embedded_items = self.embedder.embed_requirements(sop_items)
        
        self.sop_store = embedded_items
        
        # Build Numpy Matrix for Vector Search
        embeddings = [item['embedding'] for item in embedded_items if item.get('embedding')]
        if embeddings:
            self.sop_embeddings_matrix = np.array(embeddings)
            # Normalize for cosine similarity
            norm = np.linalg.norm(self.sop_embeddings_matrix, axis=1, keepdims=True)
            self.sop_embeddings_matrix = self.sop_embeddings_matrix / (norm + 1e-10)
        else:
            self.sop_embeddings_matrix = np.array([])
            
        print(f"SOP Ingested. Vector Store Size: {len(self.sop_store)} chunks.")

    def verify_documents(self) -> List[Dict]:
        """
        Core Loop: Reverse Check.
        """
        if not self.reference_index or not self.sop_store:
            raise ValueError("Reference Index or SOP Store not initialized.")
            
        gaps = []
        
        for i, req in enumerate(self.reference_index):
            if i % 10 == 0:
                print(f"Verifying requirement {i+1}/{len(self.reference_index)}...")
                
            req_embedding = np.array(req['embedding'])
            req_text = req['full_requirement_text']
            
            # 1. Semantic Search (Reference -> SOP)
            relevant_chunks = self._search_sop(req_embedding, top_k=3)
            
            # 2. Similarity Check
            # Get best score
            best_score = relevant_chunks[0]['score'] if relevant_chunks else 0
            
            status = "PRESENT"
            justification = "High similarity match found."
            
            # 3. Decision Logic (Strict Traffic Light)
            if best_score >= 0.90:
                 status = "PRESENT" # Strong evidence - Auto-Pass
                 justification = "High confidence semantic match (> 90%)."
            elif best_score >= 0.70:
                 # Ambiguous - Check with LLM
                 status, justification = self._llm_validation(req_text, relevant_chunks)
            else:
                 # Likely Missing - But verify with LLM to be sure
                 status, justification = self._llm_validation(req_text, relevant_chunks)
            
            # 4. Record Gap if not PRESENT
            if status in ["MISSING", "PARTIAL"]:
                gaps.append({
                    'requirement_id': req.get('requirement_id'),
                    'status': status,
                    'reference_requirement': req_text, # Verbatim Atomic Text
                    'reference_context': req.get('original_text', ''),
                    'reference_source': f"{req.get('source_document', '')} (Page {req.get('page_number')})",
                    'severity': req.get('mandatory_level', 'UNKNOWN'),
                    'sop_evidence': self._format_evidence(relevant_chunks) if status == 'PARTIAL' else "Not Found",
                    'justification': justification
                })
                
        return gaps

    def _search_sop(self, query_vec: np.ndarray, top_k: int = 3) -> List[Dict]:
        if self.sop_embeddings_matrix.size == 0:
            return []
            
        # Normalize query
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        
        # Cosine Similarity
        scores = np.dot(self.sop_embeddings_matrix, query_vec)
        
        # Get top K indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'chunk': self.sop_store[idx],
                'score': float(scores[idx])
            })
        return results

    def _format_evidence(self, chunks: List[Dict]) -> str:
        if not chunks:
            return "No relevant text found."
        # distinct chunks
        evidence = []
        for c in chunks:
            evidence.append(f"[Page {c['chunk']['page']}]: \"{c['chunk']['text'][:200]}...\" (Score: {c['score']:.2f})")
        return "\n".join(evidence)

    def _llm_validation(self, req_text: str, sop_chunks: List[Dict]) -> Tuple[str, str]:
        """
        Asks Bedrock to decide: PRESENT / PARTIAL / WEAK / MISSING.
        """
        
        evidence_text = ""
        for c in sop_chunks:
            evidence_text += f"---\n[Page {c['chunk']['page']}]\n{c['chunk']['text']}\n"
        
        # Mistral Instruct Format [INST]
        prompt = f"""<s>[INST] You are a rigorous Compliance Auditor.
        
        TASK: Compare the "Reference Requirement" against the provided "SOP Evidence" excerpts.
        Determine if the requirement is fully satisfied.
        
        Reference Requirement:
        "{req_text}"
        
        SOP Evidence (Best potential matches found):
        {evidence_text}
        
        INSTRUCTIONS:
        1. If the evidence explicitly confirms the requirement, typically matching the intent and key details, output PRESENT.
        2. If the concept is mentioned but key specific details from the Reference are missing, output PARTIAL.
        3. If the language is too vague (e.g., "should be considered" vs "must be done"), output WEAK.
        4. If the evidence is irrelevant or contradictory, output MISSING.
        
        OUTPUT FORMAT:
        {"{"}
            "status": "PRESENT | PARTIAL | WEAK | MISSING",
            "justification": "One sentence explanation."
        {"}"}
        
        Return ONLY valid JSON. [/INST]"""
        
        # Retry Logic
        max_retries = 3
        
        for attempt in range(max_retries + 1):
            try:
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
                
                if 'outputs' in response_body:
                     text_out = response_body['outputs'][0]['text']
                elif 'choices' in response_body:
                     text_out = response_body['choices'][0]['message']['content']
                else:
                     text_out = str(response_body)

                # Extract JSON
                clean_text = text_out.strip()
                if clean_text.startswith("```json"):
                    clean_text = clean_text[7:]
                if clean_text.endswith("```"):
                    clean_text = clean_text[:-3]
                clean_text = clean_text.strip()
                
                start = clean_text.find('{')
                end = clean_text.rfind('}') + 1
                if start != -1:
                    data = json.loads(clean_text[start:end])
                    return data.get('status', 'MISSING'), data.get('justification', 'LLM parsed.')
                else:
                    return "MISSING", f"LLM output unparseable: {clean_text[:50]}"
                     
            except Exception as e:
                error_str = str(e)
                print(f"Bedrock Error (Attempt {attempt+1}): {e}")
                
                if attempt < max_retries:
                    import time
                    if "ThrottlingException" in error_str:
                         time.sleep(5 * (attempt + 1))
                    else:
                         time.sleep(2)
                    continue
                
                return "MISSING", "LLM Error."
        
        return "MISSING", "LLM Max Retries."
