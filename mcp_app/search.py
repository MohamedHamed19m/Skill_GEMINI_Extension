# search.py
# Search strategies & SearchManager

from typing import List, Optional, Dict, Any
from pathlib import Path

from mcp_app.models import SkillMetadata
# ============================================================================
# Search Strategy (Strategy Pattern)
# ============================================================================
from abc import ABC, abstractmethod
import threading

class SearchStrategy(ABC):
    @abstractmethod
    def search(self, query: str, skills: List[SkillMetadata], limit: int) -> List[Dict]:
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        pass

class KeywordSearchStrategy(SearchStrategy):
    """Fast, simple, always ready"""
    def search(self, query: str, skills: List[SkillMetadata], limit: int) -> List[Dict]:
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored_skills = []
        
        for skill in skills:
            score = 0
            matches = []
            
            # Exact keyword matches (highest score)
            keyword_matches = [kw for kw in skill.keywords if kw.lower() in query_lower]
            score += len(keyword_matches) * 10
            if keyword_matches:
                matches.append(f"keywords: {', '.join(keyword_matches)}")
            
            # Name match
            if skill.name.lower() in query_lower:
                score += 15
                matches.append(f"name: {skill.name}")
            
            # Description word overlap
            desc_words = set(skill.description.lower().split())
            common_words = query_words & desc_words
            score += len(common_words) * 2
            if common_words:
                matches.append(f"description: {', '.join(list(common_words)[:3])}")
            
            if score > 0:
                scored_skills.append({
                    "skill": skill,
                    "score": score,
                    "matches": matches
                })
        
        # Sort by score
        scored_skills.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top N
        results = []
        for item in scored_skills[:limit]:
            results.append({
                "name": item["skill"].name,
                "score": item["score"],
                "match_reason": "; ".join(item["matches"]),
                "search_method": "keyword"
            })
        
        return results
    
    def is_ready(self) -> bool:
        return True

import pickle

class EmbeddingSearchStrategy(SearchStrategy):
    """Better quality, takes time to load"""
    def __init__(self):
        self.model = None  # Not loaded yet
        self.embeddings_cache = {}  # skill_name -> embedding
        self.cache_path = Path(__file__).parent.parent / ".embeddings_cache.pkl"
        
    def load_cache(self) -> bool:
        """Load cached embeddings from disk"""
        if not self.cache_path.exists():
            return False
        
        try:
            with open(self.cache_path, 'rb') as f:
                self.embeddings_cache = pickle.load(f)
            print(f"[EmbeddingSearch] Loaded {len(self.embeddings_cache)} cached embeddings")
            return True
        except Exception as e:
            print(f"[EmbeddingSearch] Failed to load cache: {e}")
            return False
    
    def save_cache(self):
        """Save embeddings to disk"""
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
            print(f"[EmbeddingSearch] Saved {len(self.embeddings_cache)} embeddings to cache")
        except Exception as e:
            print(f"[EmbeddingSearch] Failed to save cache: {e}")

    def search(self, query: str, skills: List[SkillMetadata], limit: int) -> List[Dict]:
        if not self.is_ready():
            return []  # Not ready yet
        
        # Encode query
        query_embedding = self.model.encode(query, convert_to_tensor=False)
        
        # Calculate similarities
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        results = []
        for skill in skills:
            if skill.name not in self.embeddings_cache:
                continue  # Skip if embedding not precomputed
            
            skill_embedding = self.embeddings_cache[skill.name]
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                skill_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity > 0.2:  # Threshold
                results.append({
                    "skill": skill,
                    "score": float(similarity)
                })
        
        # Sort by similarity
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Format output
        formatted = []
        for item in results[:limit]:
            formatted.append({
                "name": item["skill"].name,
                "score": round(item["score"], 3),
                "match_reason": "semantic similarity",
                "search_method": "embedding"
            })
        
        return formatted
    
    def is_ready(self) -> bool:
        return self.model is not None

class SearchManager:
    def __init__(self, skills_manager: 'SkillsManager'):
        self.keyword_search = KeywordSearchStrategy()
        self.embedding_search = EmbeddingSearchStrategy()
        self.skills_manager = skills_manager
        self._loading_thread = None
        self._loading_embeddings = False
        
        # Start loading embeddings in background
        self.start_embedding_loader()

    def search(self, query: str, skills: List[SkillMetadata], limit: int):
        """Use best available strategy"""
        if self.embedding_search.is_ready():
            return self.embedding_search.search(query, skills, limit)
        else:
            # Fallback to keyword search
            return self.keyword_search.search(query, skills, limit)
    

    def start_embedding_loader(self):
        """Non-blocking: Start loading embeddings in background"""
        if self._loading_embeddings:
            return  # Already loading
        
        self._loading_embeddings = True
        self._loading_thread = threading.Thread(
            target=self._load_embeddings_background,
            daemon=True  # Dies when main thread dies
        )
        self._loading_thread.start()
    
    def _load_embeddings_background(self):
        """Runs in background thread"""
        import time
        timeout = 600  # 10 minutes max
        start_time = time.time()

        try:
            # Try loading from cache first
            if self.embedding_search.load_cache():
                # Still need to load model for new searches
                from sentence_transformers import SentenceTransformer
                self.embedding_search.model = SentenceTransformer('all-MiniLM-L6-v2')
                print("[SearchManager] Embedding model ready (from cache)!")
                return

            print("[SearchManager] Loading embedding model (max 10 min)...")
            
            # Import only when needed
            import torch  # Check if PyTorch is available
            from sentence_transformers import SentenceTransformer
            
            print("[SearchManager] Downloading model (if needed)...")
            model = SentenceTransformer('all-MiniLM-L6-v2')  # ~80MB
            
            # Pre-compute embeddings for all skills
            print("[SearchManager] Computing embeddings for skills...")
            skills = self.skills_manager.get_all_skills_metadata()
            embeddings = {}

            for i, skill in enumerate(skills):
                # Check for timeout
                if time.time() - start_time > timeout:
                    raise TimeoutError("Embedding model loading timed out")

                text = f"{skill.description} {' '.join(skill.keywords)}"
                embeddings[skill.name] = model.encode(text, convert_to_tensor=False)

                # Progress indicator every 10 skills
                if (i + 1) % 10 == 0:
                    print(f"[SearchManager] Processed {i + 1}/{len(skills)} skills...")

            # Atomic update
            self.embedding_search.model = model
            self.embedding_search.embeddings_cache = embeddings
            
            # Save to cache
            self.embedding_search.save_cache()

            print(f"[SearchManager] Embedding model ready! Pre-computed {len(embeddings)} skills.")
            
        except Exception as e:
            print(f"[SearchManager] Failed to load embeddings: {e}")
            print("[SearchManager] Falling back to keyword search permanently.")
        finally:
            self._loading_embeddings = False