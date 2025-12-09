# Gemini Skills Extension Context

This extension provides a generic Skills MCP (Model Context Protocol) server to dynamically load and manage skills. This approach helps the AI agent efficiently utilize its context window by only loading relevant skills as needed, saving valuable space.

## Core Skill Workflow

- **What is a Skill?** A skill is a piece of text, typically a markdown document (`SKILL.md`), that provides the AI with specific knowledge or capabilities.


## USAGE PATTERN FOR AI AGENTS:

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


