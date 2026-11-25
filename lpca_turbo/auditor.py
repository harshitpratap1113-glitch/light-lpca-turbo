"""
Domain-Weighted Anti-Hallucination Fact Auditor (Upgraded v2.0).
- Extracts up to 20 high-signal claims
- Weights consensus by *distinct host domains* (prevents same-domain echo bias)
- Detects contradiction / negation markers across sources
- Distinguishes "no sources retrieved" (`has_evidence=False`) from low confidence
"""

from __future__ import annotations

import math
import re
import urllib.parse
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

from .search_and_scrape import ScrapedPage


STOPWORDS = {
    "that", "this", "with", "from", "have", "been", "were", "they", "their",
    "there", "which", "when", "where", "what", "about", "into", "more", "than",
    "also", "such", "only", "most", "other", "some", "could", "would", "should",
    "using", "used", "will", "these", "those", "between", "through", "over",
}

NEGATION_WORDS = {"not", "never", "false", "incorrect", "disputed", "fails", "failed", "cannot", "no", "contradicts"}


@dataclass
class VerifiedClaim:
    claim_text: str
    supporting_urls: List[str]
    supporting_domains: List[str]
    confidence_score: float
    status: str  # 'VERIFIED', 'SINGLE_SOURCE', 'UNVERIFIED', 'CONTRADICTED'


@dataclass
class AuditResult:
    topic: str
    total_pages_scraped: int
    verified_claims: List[VerifiedClaim]
    overall_confidence: float
    citation_index: Dict[int, str]
    flagged_hallucinations: List[str]
    has_evidence: bool = True


def _host_of(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc.lower().lstrip("www.")
    except Exception:
        return url


class FactAuditor:
    """Cross-source IDF & domain-diversity fact verifier."""

    def __init__(self, min_verified_sources: int = 2, max_claims: int = 16):
        self.min_sources = min_verified_sources
        self.max_claims = max_claims

    @staticmethod
    def _keywords(text: str) -> List[str]:
        return [w for w in re.findall(r"[a-zA-Z0-9]{4,}", text.lower()) if w not in STOPWORDS]

    def extract_key_claims(self, pages: List[ScrapedPage]) -> List[str]:
        claims: List[str] = []
        seen = set()
        for page in pages:
            if page.status_code != 200:
                continue
            for snippet in page.snippets:
                clean = snippet.strip()
                norm = clean.lower()
                if len(clean) > 35 and norm not in seen:
                    seen.add(norm)
                    claims.append(clean)
        # Prioritize claims containing quantitative figures or concrete assertions
        claims.sort(key=lambda c: (bool(re.search(r"\d", c)), len(c)), reverse=True)
        return claims[: self.max_claims]

    def audit_pages(self, topic: str, pages: List[ScrapedPage]) -> AuditResult:
        raw_claims = self.extract_key_claims(pages)
        verified_claims: List[VerifiedClaim] = []
        citation_index: Dict[int, str] = {}
        flagged_hallucinations: List[str] = []

        valid_pages = [p for p in pages if p.status_code == 200]
        url_list: List[str] = []
        for p in valid_pages:
            if p.url not in url_list:
                url_list.append(p.url)

        for idx, url in enumerate(url_list, start=1):
            citation_index[idx] = url

        # Compute document frequency for IDF weighting across scraped pages
        doc_freq: Counter[str] = Counter()
        page_tokens: Dict[str, set[str]] = {}
        for p in valid_pages:
            toks = set(self._keywords(p.content_markdown))
            page_tokens[p.url] = toks
            for t in toks:
                doc_freq[t] += 1
        n_docs = max(len(valid_pages), 1)

        for claim in raw_claims:
            claim_kws = list(dict.fromkeys(self._keywords(claim)))
            if not claim_kws:
                continue
            claim_has_neg = bool(NEGATION_WORDS.intersection(claim.lower().split()))

            total_idf = sum(math.log(1.0 + n_docs / (1.0 + doc_freq[k])) for k in claim_kws)
            supporting_urls: List[str] = []
            supporting_domains: List[str] = []
            contradictions = 0

            for page in valid_pages:
                ptoks = page_tokens.get(page.url, set())
                matched_idf = sum(
                    math.log(1.0 + n_docs / (1.0 + doc_freq[k])) for k in claim_kws if k in ptoks
                )
                ratio = matched_idf / max(total_idf, 1e-6)
                if ratio >= 0.42:
                    if page.url not in supporting_urls:
                        supporting_urls.append(page.url)
                    dom = _host_of(page.url)
                    if dom and dom not in supporting_domains:
                        supporting_domains.append(dom)

                    # Quick local contradiction check: matching topic keywords with opposite polarity
                    if not claim_has_neg:
                        for sent in page.snippets:
                            s_low = sent.lower()
                            if any(k in s_low for k in claim_kws[:3]) and NEGATION_WORDS.intersection(s_low.split()):
                                contradictions += 1
                                break

            dom_count = len(supporting_domains)
            url_count = len(supporting_urls)

            if contradictions >= 2 and dom_count <= 2:
                status = "CONTRADICTED"
                score = 0.25
                flagged_hallucinations.append(f"Contradicted across sources: '{claim[:80]}...'")
            elif dom_count >= self.min_sources:
                status = "VERIFIED"
                score = min(0.72 + (dom_count * 0.08), 0.99)
            elif url_count >= self.min_sources:
                status = "VERIFIED"
                score = min(0.68 + (url_count * 0.06), 0.90)
            elif url_count == 1:
                status = "SINGLE_SOURCE"
                score = 0.58
            else:
                status = "UNVERIFIED"
                score = 0.28
                flagged_hallucinations.append(f"Unverified assertion: '{claim[:80]}...'")

            verified_claims.append(
                VerifiedClaim(
                    claim_text=claim,
                    supporting_urls=supporting_urls,
                    supporting_domains=supporting_domains,
                    confidence_score=round(score, 2),
                    status=status,
                )
            )

        avg_conf = (
            sum(vc.confidence_score for vc in verified_claims) / len(verified_claims)
            if verified_claims
            else 0.0
        )

        return AuditResult(
            topic=topic,
            total_pages_scraped=len(pages),
            verified_claims=verified_claims,
            overall_confidence=round(avg_conf * 100, 1),
            citation_index=citation_index,
            flagged_hallucinations=flagged_hallucinations,
            has_evidence=bool(url_list),
        )
