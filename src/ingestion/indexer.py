import json
import boto3
import os
from typing import List
from .parser import DocumentParser
from .normalizer import RequirementNormalizer
from .embedder import RequirementEmbedder

class ReferenceIndexer:
    """
    Orchestrates the ingestion of Reference Documents:
    Parsing -> Normalization -> Embedding -> S3 Persistence.
    """
    
    def __init__(self, bucket_name: str, region_name: str = "us-east-1"):
        self.bucket_name = bucket_name
        if bucket_name != "local":
            self.s3_client = boto3.client('s3', region_name=region_name)
        else:
            self.s3_client = None
        
        # Initialize components
        self.parser = DocumentParser()
        self.normalizer = RequirementNormalizer(region_name=region_name)
        self.embedder = RequirementEmbedder(region_name=region_name)

    def process_and_index(self, file_paths: List[str], index_name: str = "reference_index.json"):
        """
        Main entry point. Processes a list of reference documents and uploads the index to S3 or saves locally.
        """
        all_requirements = []
        
        print(f"Starting ingestion for {len(file_paths)} files...")
        
        for file_path in file_paths:
            print(f"Processing: {file_path}")
            try:
                # 1. Parse
                raw_chunks = self.parser.parse_file(file_path)
                
                # 2. Normalize (Decompose into atomic requirements)
                document_requirements = []
                for chunk in raw_chunks:
                    reqs = self.normalizer.normalize_text(chunk['text'], chunk)
                    document_requirements.extend(reqs)
                
                # 3. Embed
                # We embed per document batch to keep it organized, or can do all at once.
                # Doing per document to fail fast or progress save if needed.
                embedded_reqs = self.embedder.embed_requirements(document_requirements)
                
                all_requirements.extend(embedded_reqs)
                print(f"  - Extracted {len(embedded_reqs)} atomic requirements.")
                
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
        
        # 4. Persist
        if all_requirements:
            self._save(all_requirements, index_name)
            location = f"s3://{self.bucket_name}/{index_name}" if self.bucket_name != "local" else f"./{index_name}"
            print(f"Successfully indexed {len(all_requirements)} requirements to {location}")
        else:
            print("No requirements found to index.")

    def _save(self, data: List[dict], key: str):
        json_data = json.dumps(data, indent=2)
        
        if self.bucket_name == "local":
            # Save locally
            with open(key, 'w', encoding='utf-8') as f:
                f.write(json_data)
        else:
            # Save to S3
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=key,
                    Body=json_data,
                    ContentType='application/json'
                )
            except Exception as e:
                print(f"Error saving to S3: {e}")
                raise

if __name__ == "__main__":
    # Example Usage
    # indexer = ReferenceIndexer(bucket_name="my-sop-bucket")
    # indexer.process_and_index(["./ref/GCP_Guidelines.pdf"])
    pass
