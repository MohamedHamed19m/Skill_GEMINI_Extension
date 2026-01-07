"""Search strategies and SearchManager for skill discovery."""

import difflib
import re
from typing import List, Dict, Any
from mcp_app.models import SkillMetadata


class KeywordSuggester:
    """Provides automated keyword and bi-gram suggestions based on skill content."""

    MIN_TOKEN_LEN = 2

    def __init__(self):
        """Initialize KeywordSuggester with a set of stop words."""
        self.stop_words = {
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "is",
            "if",
            "then",
            "else",
            "for",
            "with",
            "was",
            "were",
            "to",
            "of",
            "in",
            "on",
            "at",
            "by",
            "from",
            "it",
            "its",
            "this",
            "that",
            "are",
            "be",
            "been",
            "being",
            "versatile",
        }

    def suggest(self, skill: SkillMetadata, top_n: int = 5) -> List[str]:
        """Suggest keywords for a skill based on its name and description.

        Args:
            skill: The skill metadata to analyze.
            top_n: The maximum number of suggestions to return.

        Returns:
            A list of suggested keyword strings.

        """
        text = skill.name + " " + skill.description
        words = re.sub(r"[^a-z0-9\s]", "", text.lower()).split()

        bi_grams = []
        for i in range(len(words) - 1):
            if words[i] not in self.stop_words and words[i + 1] not in self.stop_words:
                bi_grams.append(f"{words[i]} {words[i + 1]}")

        candidates = [
            w for w in words if w not in self.stop_words and len(w) > self.MIN_TOKEN_LEN
        ]
        counts: Dict[str, int] = {}
        for item in candidates + bi_grams:
            counts[item] = counts.get(item, 0) + 1

        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        existing = set(k.lower() for k in skill.keywords)
        suggestions = [item for item, count in sorted_items if item not in existing]

        return suggestions[:top_n]


class LogicSearchStrategy:
    """Token-Based Fuzzy Search Engine with Automated N-Gram Keyword Extraction."""

    # Constants for scoring and thresholds
    MIN_TOKEN_LEN_FOR_FUZZY = 4
    TOKEN_MATCH_THRESHOLD_SHORT = 0.7
    TOKEN_MATCH_THRESHOLD_LONG = 0.85
    DESC_MATCH_THRESHOLD = 0.8
    MIN_DESC_TOKEN_LEN = 2
    NAME_MATCH_THRESHOLD = 0.6
    MIN_TOTAL_SCORE = 5
    FRONTMATTER_PARTS_EXPECTED = 3
    EXACT_MATCH_SCORE = 2.0

    def _get_clean_tokens(self, text: str) -> List[str]:
        return re.sub(r"[^a-z0-9\s]", "", text.lower()).split()

    def _calculate_fuzzy_score(self, query: str, target: str) -> float:
        if query == target:
            return 1.2
        return difflib.SequenceMatcher(None, query, target).ratio()

    def _score_token_against_keywords(self, q_token: str, kw_pool: List[str]) -> float:
        """Score a query token against a pool of keywords."""
        if q_token in kw_pool:
            return self.EXACT_MATCH_SCORE

        best_token_score = 0.0
        for kw in kw_pool:
            sim = self._calculate_fuzzy_score(q_token, kw)
            threshold = (
                self.TOKEN_MATCH_THRESHOLD_SHORT
                if len(q_token) < self.MIN_TOKEN_LEN_FOR_FUZZY
                else self.TOKEN_MATCH_THRESHOLD_LONG
            )
            if sim > threshold:
                best_token_score = max(best_token_score, sim * 1.2)
        return best_token_score

    def _score_token_against_description(
        self, q_token: str, desc_pool: List[str]
    ) -> float:
        """Score a query token against description tokens."""
        best_token_score = 0.0
        for d_word in desc_pool:
            sim = self._calculate_fuzzy_score(q_token, d_word)
            if (
                q_token in d_word and len(q_token) > self.MIN_DESC_TOKEN_LEN
            ) or sim > self.DESC_MATCH_THRESHOLD:
                best_token_score = max(best_token_score, sim)
        return best_token_score

    def _score_skill(
        self, query_str: str, query_tokens: List[str], skill: SkillMetadata
    ) -> Dict[str, Any]:
        """Calculate the relevance score for a single skill."""
        score = 0.0
        reasons = []
        exact_hits = 0

        kw_pool = [kw.lower() for kw in skill.keywords]
        desc_pool = self._get_clean_tokens(skill.description)

        logic_match_total = 0.0
        for q_token in query_tokens:
            token_score = self._score_token_against_keywords(q_token, kw_pool)
            if token_score >= self.EXACT_MATCH_SCORE:
                exact_hits += 1

            if token_score < 1.0:
                token_score = max(
                    token_score,
                    self._score_token_against_description(q_token, desc_pool),
                )

            logic_match_total += token_score

        # Name fuzzy match
        name_sim = self._calculate_fuzzy_score(query_str, skill.name.lower())
        if name_sim > self.NAME_MATCH_THRESHOLD:
            score += min(name_sim, 1.0) * 50
            reasons.append(f"Name({int(name_sim * 100)}%)")

        # Calculate logic score
        if logic_match_total > 0:
            logic_base = (logic_match_total / len(query_tokens)) * 50
            bonus = exact_hits * 15
            score += logic_base + bonus
            reasons.append(f"Logic({int(logic_base + bonus)}pts)")

        if score > self.MIN_TOTAL_SCORE:
            return {
                "name": skill.name,
                "score": round(score, 2),
                "match_reasons": reasons,
                "search_method": "logic_fuzzy",
            }
        return {}

    def search(
        self, query: str, skills: List[SkillMetadata], limit: int
    ) -> List[Dict[str, Any]]:
        """


        Search for skills using logic-based fuzzy matching.





        Args:


            query: The search query string.


            skills: The list of skills to search through.


            limit: The maximum number of results to return.





        Returns:


            A list of result dictionaries with scores and match reasons.


        """

        query_str = query.lower().strip()

        query_tokens = self._get_clean_tokens(query_str)

        if not query_tokens:
            return []

        scored_skills: List[Dict[str, Any]] = []

        for skill in skills:
            result = self._score_skill(query_str, query_tokens, skill)

            if result:
                scored_skills.append(result)

        # Sort by score

        scored_skills.sort(key=lambda x: float(x["score"]), reverse=True)

        return scored_skills[:limit]


class SearchManager:
    """Manages search strategies for skill discovery."""

    def __init__(self, skills_manager):
        """Initialize SearchManager with a reference to SkillsManager.

        Args:
            skills_manager: The SkillsManager instance.

        """
        # Initialize the new Logic Strategy
        self.logic_search = LogicSearchStrategy()

    def search(self, query: str, skills: List[SkillMetadata], limit: int):
        """Use logic search."""
        return self.logic_search.search(query, skills, limit)
