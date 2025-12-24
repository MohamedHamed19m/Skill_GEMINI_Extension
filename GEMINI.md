# Skills MCP Server

You have access to a Skills MCP server that provides specialized knowledge through modular skill files. Skills contain domain-specific information, code examples, and best practices that augment your capabilities.

## Core Concept

**Skills are NOT pre-loaded.** When a conversation starts, your context is empty of skill content. You must explicitly discover and load skills as needed.

## Recommended Workflow

Here's how to effectively use the Skills system:

### Simple Queries (Known Domain)

```
User: "i want capl code to send cyclic someip event"

Your thought process:
1. I need CAPL and ARETHIL knowledge
2. Let me search for relevant skills
   → search_skills(query="CAPL someip")
3. Top result: "capl-someip-skill" with high score
4. Load it: load_skill(skill_name="capl-someip-skill")
5. Use the loaded content to answer accurately
```

### Complex Queries (Multiple Domains)

```
User: "i want capl code to init someip server/client and make sure it works fine"

Your thought process:
1. I need knowledge about ARETHIL, FlexRay, and diagnostics
2. Search: search_skills(query="ARETHIL FlexRay diagnostics", limit=5)
3. Load top relevant skills:
   → load_skill(skill_name="capl-arethil-lib")
   → load_skill(skill_name="capl-someip")
   → load_skill(skill_name="capl-language-rules")
4. load this releated skills and answer using all three skills
```

## Common Patterns

### Pattern 1: Direct Load (Known Skill)
```
User mentions specific skill by name
→ load_skill(skill_name="exact-name")
```

### Pattern 2: Search Then Load (Discovery)
```
User asks about unfamiliar topic
→ search_skills(query="user's question")
→ Review results
→ load_skill(skill_name="top-match")
```

### Pattern 3: Multi-Skill Synthesis
```
User needs comprehensive answer
→ search_skills(query="broad topic", limit=5)
→ load_skill() for top 3-5 results
→ Synthesize answer from multiple sources
```


## Decision Guide

**When should you search/load skills?**

✅ **DO search when:**
- User asks about specific technical topics (protocols, libraries, frameworks)
- You need code examples or syntax you don't have memorized
- User references domain-specific terminology
- Question requires specialized knowledge beyond general training
- Discovery first, then loading. Use search_skills (recommended for query-specific relevance) or list_skills (for full catalog) to find skills before loading—avoid guessing names to prevent errors.

❌ **DON'T search when:**
- Answering general programming concepts
- User asks basic questions you can answer confidently
- Explaining fundamental principles that don't need specific examples
- Don't skip discovery Call `list_skills()` or `search_skills()` first to find relevant skills

# Extension Environment setup

**System path configuration:**
The **sklill root directory** is located at:
`~/.gemini/extensions/Gemini-Skills/skills`
**Path Resolution Rule:**
whenever a prmpot or skill file refernces a script path that starts with a forward slash `/` (for example `/client-node-skill/scripts/...`), you must resolve it as a relative path from **sklill root directory**.
**Example:**
A script path `/client-node-skill/scripts/run.py` should be resolved to:
`~/.gemini/extensions/Gemini-Skills/skills/client-node-skill/scripts/run.py`

## Quality Standards

- Be Transparent: Tell users when you're loading skills
- Show Sources: Reference which skill provides specific information
- Combine Knowledge: Integrate skill content with base knowledge seamlessly
- Ask for Clarification: If a query is ambiguous, ask before searching
- Cite Examples: When possible, provide concrete examples from skills