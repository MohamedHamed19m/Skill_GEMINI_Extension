# Skill: Gemini Skills
An extension that contains an MCP (Model Context Protocol) server for loading skills on demand. 

## Overview
This project provides a flexible and dynamic way to provide "skills" to an AI agent. This Gemini extension includes an MCP (Model Context Protocol) server for loading skills on demand and for custom commands. 
This helps AI agents manage their context window by loading skills as needed.

## As a Gemini Extension

This project is intended to be used as a Gemini extension. The `gemini-extension.json` file defines how the extension is loaded and configured. It includes an MCP server and custom commands that are started automatically by the Gemini CLI.

The MCP server is defined in `mcp_app/skills_server.py`, and custom commands are located in the `commands` directory. The skills themselves are located in the `skills` directory.

## 1- Installation

This project uses `uv` for package and environment management.

### For Developers (Local Setup)

1.  **Create and Sync the Virtual Environment:**

    Run the following command to create a virtual environment named `.venv` and install the dependencies from `pyproject.toml`:
    ```bash
    uv venv
    uv sync
    ```

2.  **Activate the Virtual Environment:**

    Before running the server, activate the environment.

    On Windows:
    ```powershell
    .venv\Scripts\activate
    ```

4.  **Run the Server (for local testing/development):**

    Once the environment is activated, you can start the server:
    ```bash
    python mcp_app/skills_server.py
    ```

### For Users (Gemini CLI Extension Installation)

To install this repository as a Gemini CLI extension:

1.  **Ensure `uv` is installed:** Follow the instructions above to install `uv` if you haven't already. `uv` is required to run the MCP server embedded within this extension.
2.  **Install the extension:**
    ```bash
    gemini extensions install https://github.com/MohamedHamed19m/Gemini_Skills
    ```

After installation, restart your Gemini CLI session to ensure the extension is loaded.

## 2. Activate

Restart the Gemini CLI. The following commands will be available:

- `list_skills()` - Lists all available skills with metadata (lightweight).

- `search_skills(query: str, limit: int = 5)` - Search for relevant skills using keyword matching.

- `load_skill(skill_name: str, force_reload: bool = False)`- Load full skill content into context.

- `add_skills_directory(path: str)` - Add custom skills directory to scan.
  **Note:** This automatically creates local shortcuts (symlinks/junctions) in the main `skills` folder. This ensures you can always reference skill resources using a consistent path: `${extensionPath}/skills/<skill-name>/...`


## Update Extension
- gemini extensions update Gemini_Skills

- gemini extensions update --all

## Skill Structure

Skills are organized in a directory structure. The server scans the `skills` directory within the project.

Each skill must be in its own directory. The directory name is the skill name. Inside the skill directory, there must be a file named `SKILL.md`.

Example structure:
```
skills/
└── skill-name/
    ├── SKILL.md (required)
    │   ├── YAML frontmatter metadata (required)
    │   │   ├── name: (required)
    │   │   └── description: (required)
    │   └── Markdown instructions (required)
    ├── scripts/ (optional)
    │   └── *.py
    ├── templates/ (optional)
    │   └── *
    └── resources/ (optional)
        └── *
```

In this example, the skill name is `my-awesome-skill`. The content of `SKILL.md` will be loaded as the skill's content. Any other files in the skill's directory are for reference and will not be loaded by the server.

The `SKILL.md` file can have a YAML frontmatter to provide metadata, for example:

```yaml
---
description: This is a skill that does awesome things.
keywords: [awesome, skill]
---

The rest of the file is the skill content.
```
The skill name is always derived from the folder name. Any `name` field provided in the frontmatter will be ignored.

## AI Agent Workflow

### Recommended Pattern (with search)
1. **Search for relevant skills:**
```
   results = search_skills(query="How do I send ARETHIL frames?")
   # Returns: Top 3-5 relevant skills with scores
```

2. **Load the best matches:**
```
   load_skill(skill_name=results[0]["name"])
```

3. **Use the content to answer the question**

### Fallback Pattern (without search)
1. **List all skills** (only if search fails):
```
   all_skills = list_skills()
```

2. **Manually filter** based on keywords/description

3. **Load specific skills**

*This project is maintained by MohamedHamed.*
