# skills_server.py
# MCP Server & Tool Definitions (Entry Point)

from fastmcp import FastMCP
from typing import Dict, Any

from mcp_app import skills_manager
from mcp_app.models import SkillsListResult, SkillLoadResult, AddDirectoryResult
from mcp_app.version import __version__


mcp = FastMCP(name="Gemini Skills" , version=__version__)

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
    Search for relevant skills using keyword matching.
    
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
    results_with_method = skills_manager.search_skills(query, limit)
    
    return {
        "results": results_with_method["results"],
        "total_found": results_with_method["total_found"],
        "query": query,
        "search_method": results_with_method["search_method"],
        "note": results_with_method["note"]
    }


@mcp.tool(
    name="add_skills_directory",
    description="Add a new directory to scan for skills files. Returns result dict with success status."
)
def add_skills_directory(path: str) -> AddDirectoryResult:
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