# mcp_app/__init__.py

import os
from pathlib import Path
from .manager import SkillsManager

# Define the base directory for skills, using an environment variable if available,
# otherwise defaulting to 'skills' relative to the project root.
# This assumes the project structure where 'skills' is a top-level directory.
# Adjust Path(os.getenv("SKILLS_DIR", "skills")) if the default is different.
SKILLS_DIR = Path(os.getenv("SKILLS_DIR", "skills")).resolve()

# Initialize the global SkillsManager instance as a singleton for the application.
# This manager will handle the scanning, loading, and tracking of skills.
skills_manager = SkillsManager(default_skills_dirs=[str(SKILLS_DIR)])

# You can optionally define __all__ if you want to control what's imported
# when someone does `from mcp_app import *`
# __all__ = ["skills_manager", "SKILLS_DIR"]
