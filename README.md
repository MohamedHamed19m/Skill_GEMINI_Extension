# Skills MCP Server for Gemini

An MCP (Model Context Protocol) server for loading skills on demand, designed to be used as a Gemini extension. This helps AI agents manage their context window by loading and unloading skills as needed.

## Overview
This project provides a flexible and dynamic way to provide "skills" to an AI agent. A skill is a piece of text, usually a document, that provides the AI with specific knowledge or capabilities. The server, running as a Gemini extension, allows the agent to list available skills and load only the ones relevant to the current task, thus saving precious context window space.

## As a Gemini Extension

This project is intended to be used as a Gemini extension. The `gemini-extension.json` file defines how the extension is loaded and configured. The MCP server is started automatically by the Gemini CLI.

The server is defined in `mcp_app/skills_server.py`, and the skills are located in the `skills` directory.

## Skill Structure

Skills are organized in a directory structure. The server scans the `skills` directory within the project.

Each skill must be in its own directory. The name of the directory is the name of the skill. Inside the skill directory, there must be a file named `SKILL.md`.

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

## Usage

The primary way to interact with this extension is through the tools it provides to the Gemini agent.

### USAGE PATTERN FOR AI AGENTS:

1.  **Discovery Phase (ALWAYS START HERE):**
    ```
    result = list_skills()
    # Analyze result.skills to find relevant ones based on:
    # - description field
    # - keywords field
    # - skill_name field
    ```

2.  **Loading Phase (LOAD ONLY WHAT YOU NEED):**
    ```
    # Load relevant skills
    skill1 = load_skill(skill_name="some-skill")
    if skill1.status == "loaded":
        # Use skill1.content for context
        pass
    elif skill1.status == "already_loaded":
        # Skill is already in context, no need to load again
        pass
    ```

### ANTI-PATTERNS TO AVOID:

*   **Loading all skills at once:** This wastes context tokens and may hit context limits.
*   **Loading the same skill multiple times:** Check the `status` field in the `load_skill` response. Use `force_reload` only when needed.
*   **Not calling `list_skills` first:** You won't know what skills are available and may request skills that don't exist.

**BEST PRACTICE:** Always call `list_skills` → analyze metadata → load specific skills.

*This project is maintained by MohamedHamed19m.*
