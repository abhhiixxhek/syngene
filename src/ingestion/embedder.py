import json
from typing import List
from sentence_transformers import SentenceTransformer

class RequirementEmbedder:
    """
    Generates embeddings for text using SentenceTransformers (all-MiniLM-L6-v2).
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", region_name: str = "us-east-1"):
        # region_name is kept for compatibility with existing calls but not used
        print(f"Loading local embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def embed_requirements(self, requirements: List[dict]) -> List[dict]:
        """
        Enrich the requirement dictionary with an 'embedding' field.
        """
        texts_to_embed = [req.get('full_requirement_text', '') for req in requirements]
        
        # Batch embed
        try:
            embeddings = self.model.encode(texts_to_embed, convert_to_numpy=True).tolist()
            
            for i, req in enumerate(requirements):
                req['embedding'] = embeddings[i]
                
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Fallback or error handling
            for req in requirements:
                req['embedding'] = []
                
        return requirements

    def _generate_embedding(self, text: str) -> List[float]:
        # Helper for single text if needed elsewhere
        return self.model.encode(text).tolist()
