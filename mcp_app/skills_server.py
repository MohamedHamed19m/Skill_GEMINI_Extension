"""
MCP Skills Server - Clean Architecture with Best Practices
Following the "Definition vs. Executor" pattern with protocol-driven design
"""
from fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass
import yaml
import json
from datetime import datetime

# ============================================================================
# Core Data Models (Pydantic for validation)
# ============================================================================

class SkillMetadata(BaseModel):
    """Lightweight skill metadata for discovery - returned by list_skills"""
    name: str = Field(description="Unique skill identifier")
    title: str = Field(description="Human-readable title")
    description: str = Field(description="What this skill provides")
    keywords: List[str] = Field(description="Keywords for relevance matching")
    version: str = Field(default="1.0.0")
    auto_activate: bool = Field(default=True)
    
    model_config = ConfigDict(json_schema_extra = {
            "example": {
                "name": "capl-arethil",
                "title": "CAPL ARETHIL Library Expert",
                "description": "Expert knowledge of Vector CAPL ARETHIL library",
                "keywords": ["arethil", "ethernet", "capl"],
                "version": "1.0.0",
                "auto_activate": True
            }
        })


class SkillLoadResult(BaseModel):
    """Result returned by load_skill tool"""
    status: str = Field(description="Status: 'loaded', 'already_loaded', or 'error'")
    skill_name: str
    content: Optional[str] = Field(None, description="Full skill content if loaded")
    message: str = Field(description="Human-readable status message")
    loaded_at: Optional[str] = Field(None, description="ISO timestamp when loaded")
    
    model_config = ConfigDict(json_schema_extra = {
            "example": {
                "status": "loaded",
                "skill_name": "capl-arethil",
                "content": "# CAPL ARETHIL Expert\n...",
                "message": "Skill loaded successfully",
                "loaded_at": "2024-12-08T10:30:00"
            }
        })


class SkillsListResult(BaseModel):
    """Result returned by list_skills tool"""
    skills: List[SkillMetadata] = Field(description="List of available skills")
    total_available: int = Field(description="Total number of skills found")
    currently_loaded: List[str] = Field(description="Names of currently loaded skills")
    
    model_config = ConfigDict(json_schema_extra = {
            "example": {
                "skills": [],
                "total_available": 5,
                "currently_loaded": ["capl-arethil"]
            }
        })


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
                "title": item["skill"].title,
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
                "title": item["skill"].title,
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
    
    def get_search_status(self) -> Dict[str, Any]:
        """For debugging/monitoring"""
        return {
            "current_strategy": "embedding" if self.embedding_search.is_ready() else "keyword",
            "embedding_ready": self.embedding_search.is_ready(),
            "loading_in_progress": self._loading_embeddings
        }

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

                text = f"{skill.title} {skill.description} {' '.join(skill.keywords)}"
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

# ============================================================================
# Skills Manager (Single Responsibility: Manage Skills State)
# ============================================================================

class SkillsManager:
    """
    Manages skill lifecycle: scanning, loading, tracking state.
    Follows Single Responsibility Principle - only manages skills, doesn't make decisions.
    """
    
    def __init__(self, default_skills_dirs: Optional[List[str]] = None):
        # Default directory
        default_dir = Path(__file__).parent.parent / "skills"
        
        if default_skills_dirs:
            self.default_skills_dirs = [Path(d) for d in default_skills_dirs]
        else:
            self.default_skills_dirs = [default_dir]
        
        # Active directories (starts with defaults)
        self.active_skills_dirs: List[Path] = self.default_skills_dirs.copy()
        
        # Ensure default directories exist
        for dir_path in self.default_skills_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # State: Available skills (metadata, path, AND source directory)
        self._available_skills: Dict[str, Dict[str, Any]] = {}
        
        # State: Loaded skills (full content + metadata)
        self._loaded_skills: Dict[str, Dict[str, Any]] = {}
        
        # Initialize by scanning directories
        self._scan_directories()

        # Add search manager
        self.search_manager = SearchManager(self)
    

    def _scan_directories(self) -> None:
        """Scan all active skills directories and parse metadata from all SKILL.md files"""
        self._available_skills.clear()
        
        for skills_dir in self.active_skills_dirs:
            if not skills_dir.exists():
                print(f"Warning: Skills directory does not exist: {skills_dir}")
                continue
                
            for md_file in skills_dir.rglob("SKILL.md"):
                try:
                    metadata = self._parse_skill_metadata(md_file)
                    skill_name = metadata.name
                    
                    # Handle duplicates: first directory in list wins
                    if skill_name in self._available_skills:
                        print(f"Warning: Duplicate skill '{skill_name}' found in {skills_dir}. "
                              f"Using version from {self._available_skills[skill_name]['source_dir']}")
                        continue
                    
                    self._available_skills[skill_name] = {
                        "metadata": metadata,
                        "path": md_file,
                        "source_dir": skills_dir  # Track which directory it came from
                    }
                except Exception as e:
                    print(f"Warning: Failed to parse {md_file.name} in folder {md_file.parent.name}: {e}")
    def _parse_skill_metadata(self, file_path: Path) -> SkillMetadata:
        """
        Parse YAML frontmatter from skill file and extract metadata only.
        Does NOT load the full content - that's done separately by load_skill.
        """
        content = file_path.read_text(encoding='utf-8')
        
        # Extract YAML frontmatter
        if not content.startswith('---'):
            raise ValueError(f"File {file_path.name} in {file_path.parent.name} missing YAML frontmatter")
        
        parts = content.split('---', 2)
        if len(parts) < 3:
            raise ValueError(f"Invalid frontmatter format in {file_path.name} in {file_path.parent.name}")
        
        frontmatter = yaml.safe_load(parts[1])
        full_content = parts[2].strip()
        
        # Create metadata object (Pydantic validation happens here)
        return SkillMetadata(
            name=frontmatter.get('name', file_path.parent.name),
            title=frontmatter.get('title', file_path.parent.name.replace('-', ' ').title()),
            description=frontmatter.get('description', ''),
            keywords=frontmatter.get('keywords', []),
            version=frontmatter.get('version', '1.0.0'),
            auto_activate=frontmatter.get('auto_activate', True)
        )
    
    
    def get_all_skills_metadata(self) -> List[SkillMetadata]:
        """Get metadata for all available skills (lightweight)"""
        return [item['metadata'] for item in self._available_skills.values()]
    
    def get_loaded_skill_names(self) -> List[str]:
        """Get names of currently loaded skills"""
        return list(self._loaded_skills.keys())
    
    def is_skill_loaded(self, skill_name: str) -> bool:
        """Check if skill is already loaded"""
        return skill_name in self._loaded_skills
    
    def skill_exists(self, skill_name: str) -> bool:
        """Check if skill exists in available skills"""
        return skill_name in self._available_skills
    
    def load_skill_content(self, skill_name: str, force_reload: bool = False) -> SkillLoadResult:
        """
        Load full skill content. Returns SkillLoadResult with appropriate status.
        
        Design Pattern: This method returns a Result object, not raw data.
        This allows the caller to handle different outcomes cleanly.
        """
        # Check if skill exists
        if not self.skill_exists(skill_name):
            return SkillLoadResult(
                status="error",
                skill_name=skill_name,
                message=f"Skill '{skill_name}' not found in available skills"
            )
        
        # Check if already loaded (unless force reload)
        if self.is_skill_loaded(skill_name) and not force_reload:
            existing = self._loaded_skills[skill_name]
            return SkillLoadResult(
                status="already_loaded",
                skill_name=skill_name,
                content=None,  # Don't return content again to save tokens
                message=f"Skill '{skill_name}' is already loaded. Use force_reload=true to reload.",
                loaded_at=existing['loaded_at']
            )
        
        # Load the skill content
        try:
            skill_info = self._available_skills[skill_name]
            metadata = skill_info['metadata']
            file_path = skill_info['path']
            
            if not file_path.exists():
                raise FileNotFoundError(f"Skill file not found at path: {file_path}")

            content = file_path.read_text(encoding='utf-8')
            
            # Remove frontmatter, keep only content
            if content.startswith('---'):
                parts = content.split('---', 2)
                content = parts[2].strip() if len(parts) >= 3 else content
            
            # Store in loaded skills
            loaded_at = datetime.now().isoformat()
            self._loaded_skills[skill_name] = {
                'content': content,
                'loaded_at': loaded_at,
                'metadata': metadata
            }
            
            return SkillLoadResult(
                status="loaded",
                skill_name=skill_name,
                content=content,
                message=f"Skill '{skill_name}' loaded successfully",
                loaded_at=loaded_at
            )
            
        except Exception as e:
            return SkillLoadResult(
                status="error",
                skill_name=skill_name,
                message=f"Failed to load skill: {str(e)}"
            )
    
    def unload_skill(self, skill_name: str) -> bool:
        """Unload a skill from memory"""
        if skill_name in self._loaded_skills:
            del self._loaded_skills[skill_name]
            return True
        return False
    
    def reload_directory(self) -> Dict[str, Any]:
        """Rescan skills directory for new or updated skills"""
        old_count = len(self._available_skills)
        self._scan_directories()
        new_count = len(self._available_skills)
        
        return {
            "previous_count": old_count,
            "current_count": new_count,
            "new_skills": new_count - old_count,
            "skill_names": [s['metadata'].name for s in self._available_skills.values()]
        }

    def add_skills_directory(self, path: str) -> Dict[str, Any]:
        """
        Add a new directory to scan for skills.
        Returns result dict with success status.
        """
        try:
            dir_path = Path(path).expanduser().resolve()
            
            # Validate directory exists
            if not dir_path.exists():
                return {
                    "success": False,
                    "error": f"Directory does not exist: {path}"
                }
            
            if not dir_path.is_dir():
                return {
                    "success": False,
                    "error": f"Path is not a directory: {path}"
                }
            
            # Check if already added (normalize paths for comparison)
            if dir_path in self.active_skills_dirs:
                return {
                    "success": True,
                    "message": "Directory already in active paths",
                    "active_directories": [str(d) for d in self.active_skills_dirs]
                }
            
            # Add to active directories
            self.active_skills_dirs.append(dir_path)
            
            # Rescan to include new directory
            old_count = len(self._available_skills)
            self._scan_directories()
            new_count = len(self._available_skills)
            
            return {
                "success": True,
                "message": f"Directory added successfully: {dir_path}",
                "active_directories": [str(d) for d in self.active_skills_dirs],
                "skills_before": old_count,
                "skills_after": new_count,
                "new_skills_found": new_count - old_count
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to add directory: {str(e)}"
            }

    def search_skills(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for relevant skills"""
        all_skills = self.get_all_skills_metadata()
        return self.search_manager.search(query, all_skills, limit)
    
    def get_search_status(self) -> Dict[str, Any]:
        """Get current search backend status"""
        return self.search_manager.get_search_status()



# ============================================================================
# MCP Server Setup
# ============================================================================

# Global instance (Singleton pattern for this use case)
skills_manager = SkillsManager()

 
mcp = FastMCP("Skills Server")

# ============================================================================
# MCP Tools (Protocol-Driven Interface)
# ============================================================================


@mcp.tool(
    name="list_skills",
    description="""
    List all available skills with their metadata (lightweight operation).
    
    This is ALWAYS the first tool the AI should call to discover available skills.
    Returns metadata only - no full content is loaded.
    
    The AI should analyze the metadata (description, keywords) to determine
    which skills are relevant to the user's query, then call load_skill
    for the specific skills needed.
    
    Returns:
        SkillsListResult with all available skills metadata
    """
)
def list_skills() -> SkillsListResult:
    """
    List all available skills with their metadata (lightweight operation).
    
    This is ALWAYS the first tool the AI should call to discover available skills.
    Returns metadata only - no full content is loaded.
    
    The AI should analyze the metadata (description, keywords) to determine
    which skills are relevant to the user's query, then call load_skill
    for the specific skills needed.
    
    Returns:
        SkillsListResult with all available skills metadata
    """
    skills = skills_manager.get_all_skills_metadata()
    loaded_names = skills_manager.get_loaded_skill_names()
    
    return SkillsListResult(
        skills=skills,
        total_available=len(skills),
        currently_loaded=loaded_names
    )


@mcp.tool(
    name="load_skill",
    description="""
    Load the full content of a specific skill into context.
    
    IMPORTANT BEHAVIOR:
    - If skill is already loaded, returns status="already_loaded" WITHOUT
      returning the content again (to avoid wasting context tokens)
    - Use force_reload=true to reload a skill that's already loaded
    - The AI should check the status field in the response
    
    Design Pattern:
    This follows the "Result Object" pattern - always returns a structured
    result with status, never throws exceptions to the LLM.
    
    Args:
        skill_name: Name of the skill to load (from list_skills)
        force_reload: If true, reload even if already loaded
    
    Returns:
        SkillLoadResult with status and content (if newly loaded)
    """
)
def load_skill(
    skill_name: str,
    force_reload: bool = False
) -> SkillLoadResult:
    """
    Load the full content of a specific skill into context.
    
    IMPORTANT BEHAVIOR:
    - If skill is already loaded, returns status="already_loaded" WITHOUT
      returning the content again (to avoid wasting context tokens)
    - Use force_reload=true to reload a skill that's already loaded
    - The AI should check the status field in the response
    
    Design Pattern:
    This follows the "Result Object" pattern - always returns a structured
    result with status, never throws exceptions to the LLM.
    
    Args:
        skill_name: Name of the skill to load (from list_skills)
        force_reload: If true, reload even if already loaded
    
    Returns:
        SkillLoadResult with status and content (if newly loaded)
    """
    return skills_manager.load_skill_content(skill_name, force_reload)




@mcp.tool(
    name="search_skills",
    description="""
    Search for relevant skills using intelligent query matching.
    
    This tool automatically uses the best available search method:
    - Keyword matching (fast, always available)
    - Semantic search (better quality, loads in background)
    
    The AI doesn't need to know which method is used - just call this tool
    and get ranked results based on relevance to your query.
    
    RECOMMENDED WORKFLOW:
    1. Call search_skills(query="user's question")
    2. Review top 3-5 suggestions
    3. Load the most relevant skills using load_skill()
    
    Args:
        query: The user's question or topic
        limit: Maximum number of suggestions (default: 5)
    
    Returns:
        List of relevant skills with scores and match reasons
    """
)
def search_skills(query: str, limit: int = 5) -> Dict[str, Any]:
    """Search for relevant skills based on query"""
    results = skills_manager.search_skills(query, limit)
    status = skills_manager.get_search_status()
    
    return {
        "results": results,
        "total_found": len(results),
        "query": query,
        "search_method": status["current_strategy"],
        "note": "Using keyword search" if status["current_strategy"] == "keyword" 
                else "Using semantic search (higher quality)"
    }

# Optional: Tool to check search status
@mcp.tool(
    name="get_search_status",
    description="Check which search backend is currently active (for debugging)"
)
def get_search_status() -> Dict[str, Any]:
    """Get current search backend status"""
    return skills_manager.get_search_status()

@mcp.tool(
    name="get_embedding_error",
    description="Get detailed error info if semantic search failed to load"
)
def get_embedding_error() -> Dict[str, Any]:
    """Get embedding loading error details"""
    status = skills_manager.get_search_status()
    
    if status["embedding_ready"]:
        return {"error": None, "message": "Embeddings loaded successfully"}
    
    # Check common issues
    issues = []
    
    try:
        import sentence_transformers
    except ImportError:
        issues.append("sentence-transformers not installed. Run: pip install sentence-transformers")
    
    try:
        import torch
    except ImportError:
        issues.append("PyTorch not installed. Run: pip install torch")
    
    return {
        "error": "Embeddings failed to load",
        "current_search": "keyword (fallback)",
        "potential_issues": issues,
        "recommendation": "Install missing dependencies or use keyword search"
    }
    
@mcp.tool(
    name="add_skills_directory",
    description="Add a new directory to scan for skills files. Returns result dict with success status."
)
def add_skills_directory(path: str) -> Dict[str, Any]:
    """
    Add a new directory to scan for skills.
    Returns result dict with success status.
    """
    return skills_manager.add_skills_directory(path)

# ============================================================================
# Server Entry Point
# ============================================================================

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()