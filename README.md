# Skill: GEMINI Extension
An extension that contains an MCP (Model Context Protocol) server for loading skills on demand. 

## Overview
This project provides a flexible and dynamic way to provide "skills" to an AI agent. This Gemini extension includes an MCP (Model Context Protocol) server for loading skills on demand and for custom commands. 
This helps AI agents manage their context window by loading skills as needed.

## As a Gemini Extension

This project is intended to be used as a Gemini extension. The `gemini-extension.json` file defines how the extension is loaded and configured. It includes an MCP server and custom commands that are started automatically by the Gemini CLI.

The MCP server is defined in `mcp_app/skills_server.py`, and custom commands are located in the `commands` directory. The skills themselves are located in the `skills` directory.

## Installation

This project uses `uv` for package and environment management.

### For Developers (Local Setup)

1.  **Install `uv`:**

    If you don't have `uv` installed, follow the official installation instructions for your OS. For example, on macOS/Linux:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    On Windows:
    ```powershell
    irm https://astral.sh/uv/install.ps1 | iex
    ```

2.  **Create and Sync the Virtual Environment:**

    Run the following command to create a virtual environment named `.venv` and install the dependencies from `pyproject.toml`:
    ```bash
    uv venv
    uv sync
    ```

3.  **Activate the Virtual Environment:**

    Before running the server, activate the environment.
    
    On macOS/Linux:
    ```bash
    source .venv/bin/activate
    ```
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
    gemini extensions install https://github.com/MohamedHamed19m/Skill_MCP_Server.git
    ```
    (Replace `https://github.com/MohamedHamed19m/Skill_MCP_Server.git` with the actual public repository URL if it changes.)

    You can specify a branch, tag, or commit using `--ref`, e.g.:
    ```bash
    gemini extensions install https://github.com/MohamedHamed19m/Skill_MCP_Server.git --ref=main
    ```

After installation, restart your Gemini CLI session to ensure the extension is loaded.

## Releasing

This extension can be easily distributed via its Git repository. Users can install it directly using the Gemini CLI:

```bash
gemini extensions install https://github.com/MohamedHamed19m/Skill_MCP_Server.git
```

For more advanced releasing options, such as GitHub Releases for pre-built archives, refer to the [Gemini CLI Extension Releasing Guide](https://geminicli.com/docs/extensions/extension-releasing).

## Skill Structure

Skills are organized in a directory structure. The server scans the `skills` directory within the project.

Each skill must be in its own directory. The directory name is the skill name. Inside the skill directory, there must be a file named `SKILL.md`.

Example structure:
```
skills/
└── my-awesome-skill/
    ├── SKILL.md
    └── some-other-file.txt
```

In this example, the skill name is `my-awesome-skill`. The content of `SKILL.md` will be loaded as the skill's content. Any other files in the skill's directory are for reference and will not be loaded by the server.

The `SKILL.md` file can have a YAML frontmatter to provide metadata, for example:

```yaml
---
name: my-awesome-skill-override
title: My Awesome Skill
description: This is a skill that does awesome things.
keywords: [awesome, skill]
---

The rest of the file is the skill content.
```
If the `name` is provided in the frontmatter, it will override the folder name.

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

## Search Quality

- **First 5-10 minutes:** Uses fast keyword matching
- **After model loads:** Automatically upgrades to semantic search
- **Transparent:** AI doesn't need to know which backend is active

*This project is maintained by MohamedHamed19m.*
