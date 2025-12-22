# manager.py
# SkillsManager (Core logic)

from typing import List, Optional, Dict, Any

from pathlib import Path

from datetime import datetime

import yaml



from mcp_app.models import SkillMetadata, SkillLoadResult

from mcp_app.search import SearchManager





# ============================================================================

# Skills Manager (Single Responsibility: Manage Skills State)

# ============================================================================



class SkillsManager:

    """

    Manages skill lifecycle: scanning, loading, tracking state.

    Follows Single Responsibility Principle - only manages skills, doesn't make decisions.

    """

    

    def __init__(self, default_skills_dirs: Optional[List[str]] = None):

        

        if default_skills_dirs:

            self.default_skills_dirs = [Path(d) for d in default_skills_dirs]

        else:

            self.default_skills_dirs = [Path("skills")]
        
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
        
        # Always use folder name as unique identifier
        skill_name = file_path.parent.name

        # Create metadata object (Pydantic validation happens here)
        return SkillMetadata(
            name=skill_name,  # ← Always from folder
            description=frontmatter.get('description', ''),
            keywords=frontmatter.get('keywords', [])
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

    def search_skills(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Search for relevant skills and return formatted results."""
        all_skills = self.get_all_skills_metadata()
        
        # The search_manager.search method now returns results with search_method included
        results_from_search_manager = self.search_manager.search(query, all_skills, limit)
        
        # Determine the search method used for the results
        search_method_used = "keyword"
        if results_from_search_manager:
            # Assume all results from a single call use the same search method
            search_method_used = results_from_search_manager[0].get("search_method", "keyword")
            
        note_message = "Using keyword search" if search_method_used == "keyword" else "Using semantic search (higher quality)"

        return {
            "results": results_from_search_manager,
            "total_found": len(results_from_search_manager),
            "query": query,
            "search_method": search_method_used,
            "note": note_message
        }
