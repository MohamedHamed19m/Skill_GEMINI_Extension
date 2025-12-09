# Vector Documentation Parser

A Python package for extracting structured function information from Vector (CAPL) documentation in Markdown format and MCP Server to utilize this parser and offering tools for AI agents.

## Overview
This tool parses Vector documentation files (such as CAPL function md docs) and extracts detailed function information into structured Python class objects. It automates the process of converting Markdown-based documentation into data that can be further processed, analyzed, or integrated into other systems.

## Setup

This project includes a setup script (`setup_mcp.py`) to manage the MCP server configuration, allowing for both local (project-specific) and global installations in gemini settings.json (in future will support other agents like claude).

**Usage:**

```bash
# Install globally (checks existing config, appends or overwrites)
python setup_mcp.py install --global

# Install locally for this project only
python setup_mcp.py install --local

# Check current status of both local and global configs
python setup_mcp.py status

# Remove from global config (keeps other servers)
python setup_mcp.py uninstall --global

# Remove local config completely
python setup_mcp.py uninstall --local
```

Run `python setup_mcp.py --help` for more details on commands and options.

### Environment Setup with `uv`

This project uses `uv` for efficient dependency management and virtual environment creation.

1.  **Create Virtual Environment:**
    ```bash
    uv venv
    ```

2.  **Activate Environment:**
    -   macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```
    -   Windows (PowerShell):
        ```bash
        .venv\Scripts\Activate.ps1
        ```

3.  **Synchronize Dependencies and Create Lock File:**
    ```bash
    uv sync
    ```
    This command reads `pyproject.toml`, creates `uv.lock` (if it doesn't exist), and installs dependencies into `.venv`. Commit `uv.lock` to Git.

5.  **Regenerate Lock File:**
    If `pyproject.toml` is manually changed or branches are switched:
    ```bash
    uv lock
    ```

6.  **Deactivate Environment:**
    ```bash
    deactivate
    ```

## Features
- **Comprehensive Markdown Parsing**: Extracts function names, multiple syntax forms, descriptions, parameters, return values, code examples, and "Valid for" information.
- **Structured Output**: Stores all extracted data in well-defined Python dataclasses (`FunctionInfo`, `Parameter`).
- **Handles Complex Docs**: Supports multiple syntax forms, preserves code examples, and robustly parses various Markdown section formats.
- **Batch Processing**: Can parse a single file or all Markdown files in a directory.
- **Pretty Printing**: Includes `__str__` methods for easy inspection of parsed data.
- **Error Handling**: Designed to handle malformed or incomplete documentation gracefully.

## Main Components
- **`FunctionInfo` class**: Stores all extracted information for a function, including:
	- Function name
	- Multiple syntax forms
	- Description
	- Parameters (with names and descriptions)
	- Return values
	- Example code
	- "Valid for" information
- **`Parameter` class**: Represents individual parameters with name and description.
- **`VectorDocParser` class**: Handles the actual parsing logic for files and directories.

## Usage

### Parse a single file
```python
parser = VectorDocParser()
func_info = parser.parse_file("path/to/file.md")
```

### Parse all markdown files in a directory
```python
functions = parser.parse_directory("./vector_docs")
```

### Command-line usage
Run the script directly to parse a file or directory:
```bash
python main.py <file_path_or_directory>
```

- To parse a single file: `python main.py path/to/file.md`
- To parse a directory: `python main.py path/to/directory/`

## Requirements
- Python 3.7+
- No external dependencies (uses only the Python standard library)

## Input Format
- Markdown files documenting Vector (CAPL) functions, with sections for syntax, parameters, return values, examples, etc.

## Output
- Structured Python objects containing all extracted information, ready for further processing or integration.

## Project Structure
- `main.py`          — Entry point for the parser (CLI and importable)
- `setup_mcp.py`     — Setup script to configure the MCP server.
- `mcp_app/`         — Contains the MCP server application.
  - `MCP_Server.py`  — Hosts the parsing logic as an MCP server for Gemini.
- `src/`             — Core Python source code for parsing and searching.
- `docs/`            — Detailed documentation files.
- `inputs/`          — Example input Markdown files.
- `README.md`        — This file.

## Documentation

This project includes an MCP (Model Context Protocol) server that exposes the documentation parsing and search functionality as a set of tools for AI agents.

For more details, see the documentation below:

- **[MCP Usage Guide (docs/MCP_Usage.md)](docs/MCP_Usage.md)**: Explains when and why to use this MCP server.
- **[MCP Server Details (docs/README_MCP.md)](docs/README_MCP.md)**: Provides detailed information on the server's tools and how to connect to it.

## Skills MCP Server

This project also includes a general-purpose MCP server for managing and loading "skills" for an AI agent. A skill is a piece of text, usually a document, that provides the AI with knowledge or capabilities.

### Skill Structure

Skills are organized in a directory structure. The server scans a `skills` directory (by default `~/.capl-skills/skills`, but this is ignored by git in the project).

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
 
 
📘 Example CAPL Documentation

If you want to explore additional CAPL function documentation examples or test Markdown files,
you can refer to the following public repository:

🔗 Sylphith/capl_docs — AUTOSAR Ethernet IL CAPL Functions
[https://github.com/Sylphith/capl_docs/blob/main/IP/AUTOSARethIL/CAPLfunctionsAREthILOverview.md].

This repo contains real-world CAPL .md documentation files that you can use to test or validate the parser.

*This project is maintained by MohamedHamed19m.*
