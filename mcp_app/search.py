# search.py
# Search strategies & SearchManager

from typing import List, Optional, Dict, Any
from mcp_app.models import SkillMetadata

class KeywordSearchStrategy:
    """Fast, simple, always ready"""
    def search(self, query: str, skills: List[SkillMetadata], limit: int) -> List[Dict]:
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored_skills = []
        
        for skill in skills:
            score = 0
            matches = []
            
            # Exact keyword matches (highest score)
            keyword_matches = [kw for kw in skill.keywords if kw.lower() in query_lower]
            score += len(keyword_matches) * 10
            if keyword_matches:
                matches.append(f"keywords: {', '.join(keyword_matches)}")
            
            # Name match
            if skill.name.lower() in query_lower:
                score += 15
                matches.append(f"name: {skill.name}")
            
            # Description word overlap
            desc_words = set(skill.description.lower().split())
            common_words = query_words & desc_words
            score += len(common_words) * 2
            if common_words:
                matches.append(f"description: {', '.join(list(common_words)[:3])}")
            
            if score > 0:
                scored_skills.append({
                    "skill": skill,
                    "score": score,
                    "matches": matches
                })
        
        # Sort by score
        scored_skills.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top N
        results = []
        for item in scored_skills[:limit]:
            results.append({
                "name": item["skill"].name,
                "score": item["score"],
                "match_reason": "; ".join(item["matches"]),
                "search_method": "keyword"
            })
        
        return results

class SearchManager:
    def __init__(self, skills_manager):
        # We don't need skills_manager stored if we don't do background loading
        self.keyword_search = KeywordSearchStrategy()

    def search(self, query: str, skills: List[SkillMetadata], limit: int):
        """Use keyword search"""
        return self.keyword_search.search(query, skills, limit)
