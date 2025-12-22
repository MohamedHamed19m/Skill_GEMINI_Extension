# models.py
# Pydantic models (Data schemas)

from fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional

# ============================================================================
# Core Data Models (Pydantic for validation)
# ============================================================================

class SkillMetadata(BaseModel):
    """Lightweight skill metadata for discovery - returned by list_skills"""
    name: str = Field(description="Unique skill identifier")
    description: str = Field(description="What this skill provides")
    keywords: List[str] = Field(description="Keywords for relevance matching")
    
    model_config = ConfigDict(json_schema_extra = {
            "example": {
                "name": "capl-arethil",
                "description": "Expert knowledge of Vector CAPL ARETHIL library",
                "keywords": ["arethil", "ethernet", "capl"]
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

