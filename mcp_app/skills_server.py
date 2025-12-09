"""
MCP Skills Server - Clean Architecture with Best Practices
Following the "Definition vs. Executor" pattern with protocol-driven design
"""

from fastmcp import FastMCP
from pydantic import BaseModel, Field
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
    estimated_tokens: int = Field(description="Approximate token count")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "capl-arethil",
                "title": "CAPL ARETHIL Library Expert",
                "description": "Expert knowledge of Vector CAPL ARETHIL library",
                "keywords": ["arethil", "ethernet", "capl"],
                "version": "1.0.0",
                "auto_activate": True,
                "estimated_tokens": 4500
            }
        }


class SkillLoadResult(BaseModel):
    """Result returned by load_skill tool"""
    status: str = Field(description="Status: 'loaded', 'already_loaded', or 'error'")
    skill_name: str
    content: Optional[str] = Field(None, description="Full skill content if loaded")
    message: str = Field(description="Human-readable status message")
    tokens_loaded: Optional[int] = Field(None, description="Token count if newly loaded")
    loaded_at: Optional[str] = Field(None, description="ISO timestamp when loaded")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "loaded",
                "skill_name": "capl-arethil",
                "content": "# CAPL ARETHIL Expert\n...",
                "message": "Skill loaded successfully",
                "tokens_loaded": 4500,
                "loaded_at": "2024-12-08T10:30:00"
            }
        }


class SkillsListResult(BaseModel):
    """Result returned by list_skills tool"""
    skills: List[SkillMetadata] = Field(description="List of available skills")
    total_available: int = Field(description="Total number of skills found")
    currently_loaded: List[str] = Field(description="Names of currently loaded skills")
    total_tokens_loaded: int = Field(description="Total tokens in loaded skills")
    
    class Config:
        json_schema_extra = {
            "example": {
                "skills": [],
                "total_available": 5,
                "currently_loaded": ["capl-arethil"],
                "total_tokens_loaded": 4500
            }
        }


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
            auto_activate=frontmatter.get('auto_activate', True),
            estimated_tokens=self._estimate_tokens(full_content)
        )
    
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimation: 1 token ≈ 4 characters"""
        return len(text) // 4
    
    def get_all_skills_metadata(self) -> List[SkillMetadata]:
        """Get metadata for all available skills (lightweight)"""
        return [item['metadata'] for item in self._available_skills.values()]
    
    def get_loaded_skill_names(self) -> List[str]:
        """Get names of currently loaded skills"""
        return list(self._loaded_skills.keys())
    
    def get_total_loaded_tokens(self) -> int:
        """Calculate total tokens from loaded skills"""
        return sum(
            skill_data['tokens'] 
            for skill_data in self._loaded_skills.values()
        )
    
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
                tokens_loaded=existing['tokens'],
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
                'tokens': metadata.estimated_tokens,
                'loaded_at': loaded_at,
                'metadata': metadata
            }
            
            return SkillLoadResult(
                status="loaded",
                skill_name=skill_name,
                content=content,
                message=f"Skill '{skill_name}' loaded successfully",
                tokens_loaded=metadata.estimated_tokens,
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



# ============================================================================
# MCP Server Setup
# ============================================================================

mcp = FastMCP("CAPL Skills Server")

# Global instance (Singleton pattern for this use case)
skills_manager = SkillsManager()


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
    total_tokens = skills_manager.get_total_loaded_tokens()
    
    return SkillsListResult(
        skills=skills,
        total_available=len(skills),
        currently_loaded=loaded_names,
        total_tokens_loaded=total_tokens
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



from sklearn.feature_extraction.text import TfidfVectorizer  # For scoring; lightweight
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

@mcp.tool(
    name="suggest_relevant_skills",
    description="Given a user query, suggest top relevant skills with scores. Use this BEFORE loading to minimize context bloat. Returns top 3-5 with match reasons."
)
def suggest_relevant_skills(query: str, max_suggestions: int = 3) -> Dict[str, Any]:
    all_metadata = skills_manager.get_all_skills_metadata()
    if not all_metadata:
        return {"suggestions": [], "message": "No skills available"}
    
    # Quick TF-IDF scoring
    documents = [f"{m.description} {' '.join(m.keywords)} {m.name}" for m in all_metadata]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
    tfidf_matrix = vectorizer.fit_transform(documents + [query])  # Include query
    query_vec = tfidf_matrix[-1:]
    doc_vecs = tfidf_matrix[:-1]
    
    scores = cosine_similarity(query_vec, doc_vecs)[0]
    scored_skills = sorted(zip(all_metadata, scores), key=lambda x: x[1], reverse=True)
    
    suggestions = []
    for skill, score in scored_skills[:max_suggestions]:
        if score > 0.1:  # Threshold for relevance
            suggestions.append({
                "name": skill.name,
                "title": skill.title,
                "relevance_score": float(score),
                "reason": f"Matches on: {', '.join(set(skill.keywords) & set(query.lower().split()))}"
            })
    
    return {
        "suggestions": suggestions,
        "total_scored": len(scored_skills),
        "query": query,
        "advice": "Load suggested skills via load_skill for best results."
    }

@mcp.tool(
    name="add_skills_directory",
    description="Add a new directory to scan for skills. Returns result dict with success status."
)
def add_skills_directory(path: str) -> Dict[str, Any]:
    """
    Add a new directory to scan for skills.
    Returns result dict with success status.
    """
    return skills_manager.add_skills_directory(path)
# ============================================================================
# Usage Documentation
# ============================================================================

"""
USAGE PATTERN FOR AI AGENTS:

1. Discovery Phase (ALWAYS START HERE):
   ```
   result = list_skills()
   # Analyze result.skills to find relevant ones based on:
   # - description field
   # - keywords field
   # - skill_name field
   ```

2. Loading Phase (LOAD ONLY WHAT YOU NEED):
   ```
   # Load relevant skills
   skill1 = load_skill(skill_name="capl-arethil")
   if skill1.status == "loaded":
       # Use skill1.content for context
       pass
   elif skill1.status == "already_loaded":
       # Skill is already in context, no need to load again
       pass
   ```
 
EXAMPLE CONVERSATION FLOW:

User: "How do I send an ARETHIL frame in CAPL?"

AI reasoning:
1. Call list_skills() to discover available skills
2. Analyze metadata: "capl-arethil" has keywords ["arethil", "capl"]
   and description mentions "ARETHIL library"
3. Call load_skill(skill_name="capl-arethil")
4. Use the loaded content to answer the question with accurate code

User: "Now I need to work with the database"

AI reasoning:
1. Current context has "capl-arethil" loaded
2. User is switching domains
4. Call load_skill(skill_name="capl-database")
5. Answer using database skill content

ANTI-PATTERNS TO AVOID:

❌ Loading all skills at once
   - Wastes context tokens
   - May hit context limits

❌ Loading the same skill multiple times
   - Check status field in load_skill response
   - Use force_reload only when needed

❌ Not calling list_skills first
   - You won't know what skills are available
   - May request skills that don't exist

✅ BEST PRACTICE:
   Always call list_skills → analyze metadata → load specific skills
"""


# ============================================================================
# Server Entry Point
# ============================================================================

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()