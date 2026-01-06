# search.py
# Search strategies & SearchManager

import difflib
import re
from typing import List, Dict
from mcp_app.models import SkillMetadata

class KeywordSuggester:
    def __init__(self):
        self.stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'is', 'if', 'then', 'else', 
            'for', 'with', 'was', 'were', 'to', 'of', 'in', 'on', 'at', 'by', 
            'from', 'it', 'its', 'this', 'that', 'are', 'be', 'been', 'being', 'versatile'
        }

    def suggest(self, skill: SkillMetadata, top_n: int = 5) -> List[str]:
        text = skill.name + " " + skill.description
        words = re.sub(r'[^a-z0-9\s]', '', text.lower()).split()
        
        bi_grams = []
        for i in range(len(words)-1):
            if words[i] not in self.stop_words and words[i+1] not in self.stop_words:
                bi_grams.append(f"{words[i]} {words[i+1]}")

        candidates = [w for w in words if w not in self.stop_words and len(w) > 2]
        counts = {}
        for item in (candidates + bi_grams):
            counts[item] = counts.get(item, 0) + 1
            
        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        existing = set(k.lower() for k in skill.keywords)
        suggestions = [item for item, count in sorted_items if item not in existing]
        
        return suggestions[:top_n]

class LogicSearchStrategy:
    """Token-Based Fuzzy Search Engine with Automated N-Gram Keyword Extraction.
    """

    def _get_clean_tokens(self, text: str) -> List[str]:
        return re.sub(r'[^a-z0-9\s]', '', text.lower()).split()

    def _calculate_fuzzy_score(self, query: str, target: str) -> float:
        if query == target:
            return 1.2 
        return difflib.SequenceMatcher(None, query, target).ratio()

    def search(self, query: str, skills: List[SkillMetadata], limit: int) -> List[Dict]:
        scored_skills = []
        query_str = query.lower().strip()
        query_tokens = self._get_clean_tokens(query_str)

        if not query_tokens:
            return []

        for skill in skills:
            score = 0.0
            reasons = []
            exact_hits = 0
            
            # Use original keywords (they might be enhanced by Suggester in Manager)
            kw_pool = [kw.lower() for kw in skill.keywords]
            desc_pool = self._get_clean_tokens(skill.description)
            
            logic_match_total = 0
            for q_token in query_tokens:
                best_token_score = 0
                
                # Check keywords
                if q_token in kw_pool:
                    best_token_score = 2.0
                    exact_hits += 1
                else:
                    for kw in kw_pool:
                        sim = self._calculate_fuzzy_score(q_token, kw)
                        if sim > (0.7 if len(q_token) < 4 else 0.85):
                            best_token_score = max(best_token_score, sim * 1.2)
                
                # Check description if keyword match wasn't perfect
                if best_token_score < 1.0:
                    for d_word in desc_pool:
                        sim = self._calculate_fuzzy_score(q_token, d_word)
                        if (q_token in d_word and len(q_token) > 2) or sim > 0.8:
                            best_token_score = max(best_token_score, sim)
                
                logic_match_total += best_token_score

            # Name fuzzy match
            name_sim = self._calculate_fuzzy_score(query_str, skill.name.lower())
            if name_sim > 0.6:
                score += (min(name_sim, 1.0) * 50)
                reasons.append(f"Name({int(name_sim*100)}%)")

            # Calculate logic score
            if logic_match_total > 0:
                logic_base = (logic_match_total / len(query_tokens)) * 50
                bonus = exact_hits * 15 
                score += (logic_base + bonus)
                reasons.append(f"Logic({int(logic_base + bonus)}pts)")

            if score > 5:
                scored_skills.append({
                    "name": skill.name,
                    "score": round(score, 2),
                    "match_reasons": reasons,
                    "search_method": "logic_fuzzy"
                })

        # Sort by score
        scored_skills.sort(key=lambda x: x["score"], reverse=True)
        
        return scored_skills[:limit]

class SearchManager:
    def __init__(self, skills_manager):
        # Initialize the new Logic Strategy
        self.logic_search = LogicSearchStrategy()

    def search(self, query: str, skills: List[SkillMetadata], limit: int):
        """Use logic search"""
        return self.logic_search.search(query, skills, limit)
