"""
Metadata Filtering & Boosting for RAG Retrieval
================================================

This module adds temporal and structural metadata to document chunks,
enabling:
1. Metadata filtering (year, quarter, document type)
2. Metadata boosting (weighted score increases for matching metadata)
3. Enhanced query analysis to extract metadata hints

Usage:
    from metadata_enhancer import MetadataEnhancer, enhance_kb_with_metadata
    
    # Enhance existing KB
    enhance_kb_with_metadata("./data_marker")
    
    # Use in search with metadata filtering/boosting
    kb = KBEnv(base="./data_marker")
    results = kb.search_with_metadata(
        query="What was revenue in Q2 2024?",
        k=12,
        boost_weights={"year": 2.0, "quarter": 1.5}
    )
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class MetadataHints:
    """Extracted metadata hints from query"""
    years: List[int]
    quarters: List[str]  # e.g., ["1Q2024", "2Q2024"]
    doc_types: List[str]  # e.g., ["annual_report", "quarterly_results"]
    sections: List[str]   # e.g., ["financial_highlights", "cfo_presentation"]


class MetadataEnhancer:
    """Extract and apply metadata to document chunks"""
    
    # Document type patterns
    DOC_TYPE_PATTERNS = {
        "annual_report": r"annual[_\s-]report",
        "quarterly_results": r"(quarterly|quarter|[1-4]q\d{2,4})[_\s-](results|performance|summary)",
        "ceo_presentation": r"ceo[_\s-]presentation",
        "cfo_presentation": r"cfo[_\s-]presentation",
        "trading_update": r"trading[_\s-]update",
        "press_statement": r"press[_\s-](statement|release)",
        "performance_summary": r"performance[_\s-]summary",
    }
    
    # Section patterns (common in financial reports)
    SECTION_PATTERNS = {
        "financial_highlights": r"financial\s+highlights?",
        "income_statement": r"(income|profit.*loss)\s+statement",
        "balance_sheet": r"balance\s+sheet",
        "cash_flow": r"cash\s+flow",
        "segment_results": r"segment(al)?\s+(results|performance)",
        "risk_management": r"risk\s+management",
        "notes": r"notes?\s+to\s+(the\s+)?financial",
    }
    
    @staticmethod
    def extract_year_from_doc_name(doc_name: str) -> Optional[int]:
        """Extract 4-digit year from document name"""
        # Look for patterns like: "2024", "FY2024", "FY24"
        matches = re.findall(r'(?:fy)?(\d{4})|(?:fy)?(\d{2})(?![0-9])', doc_name.lower())
        for m in matches:
            year_str = m[0] if m[0] else m[1]
            if year_str:
                year = int(year_str)
                # Convert 2-digit to 4-digit
                if year < 100:
                    year += 2000
                # Sanity check: reasonable range for financial reports
                if 2000 <= year <= 2030:
                    return year
        return None
    
    @staticmethod
    def extract_quarter_from_doc_name(doc_name: str) -> Optional[str]:
        """Extract quarter info from document name (e.g., '1Q24', '2Q2024')"""
        # Patterns: 1Q24, 2Q2024, Q1_2024, 1Q_24, Q1FY24
        patterns = [
            r'([1-4])q(\d{2,4})',      # 1Q24, 1Q2024
            r'q([1-4])[_\s-]?(\d{2,4})',  # Q1_2024, Q1-24
            r'([1-4])q[_\s-]?fy(\d{2,4})',  # 1QFY24
        ]
        
        doc_lower = doc_name.lower()
        for pattern in patterns:
            match = re.search(pattern, doc_lower)
            if match:
                quarter = int(match.group(1))
                year_str = match.group(2)
                year = int(year_str)
                if year < 100:
                    year += 2000
                if 2000 <= year <= 2030 and 1 <= quarter <= 4:
                    return f"{quarter}Q{year}"
        return None
    
    @staticmethod
    def extract_doc_type(doc_name: str) -> Optional[str]:
        """Classify document type based on name"""
        doc_lower = doc_name.lower()
        for doc_type, pattern in MetadataEnhancer.DOC_TYPE_PATTERNS.items():
            if re.search(pattern, doc_lower):
                return doc_type
        return "other"
    
    @staticmethod
    def extract_section_from_text(text: str, max_chars: int = 500) -> Optional[str]:
        """Detect section type from chunk text (check first max_chars)"""
        text_sample = text[:max_chars].lower()
        for section, pattern in MetadataEnhancer.SECTION_PATTERNS.items():
            if re.search(pattern, text_sample):
                return section
        return None
    
    @staticmethod
    def enhance_chunk_metadata(chunk_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add metadata columns to existing chunk DataFrame
        
        Adds:
        - year (int): Extracted from doc name
        - quarter (str): e.g., "1Q2024" or None
        - doc_type (str): Document classification
        - section (str): Section hint from text content
        """
        df = chunk_df.copy()
        
        # Extract temporal metadata from document names
        df['year'] = df['doc'].apply(MetadataEnhancer.extract_year_from_doc_name)
        df['quarter'] = df['doc'].apply(MetadataEnhancer.extract_quarter_from_doc_name)
        df['doc_type'] = df['doc'].apply(MetadataEnhancer.extract_doc_type)
        
        # Extract section hints from text (if 'text' column exists)
        if 'text' in df.columns:
            df['section'] = df['text'].apply(
                lambda t: MetadataEnhancer.extract_section_from_text(str(t))
            )
        else:
            df['section'] = None
        
        return df


class MetadataQueryAnalyzer:
    """Extract metadata hints from user queries"""
    
    @staticmethod
    def extract_years(query: str) -> List[int]:
        """Extract all years mentioned in query"""
        years = []
        # 4-digit years
        for match in re.finditer(r'\b(20\d{2})\b', query):
            years.append(int(match.group(1)))
        # 2-digit years with context (e.g., "FY24", "'24")
        for match in re.finditer(r"(?:fy|')?(\d{2})\b", query.lower()):
            year = int(match.group(1))
            if 20 <= year <= 30:  # Reasonable range for 2020-2030
                years.append(2000 + year)
        return sorted(list(set(years)))
    
    @staticmethod
    def extract_quarters(query: str, kb_path: Optional[str] = None) -> List[str]:
        """
        Extract quarter references (e.g., 'Q2 2024', '1Q24', 'last 5 quarters')
        
        Args:
            query: User query string
            kb_path: Optional path to kb_chunks.parquet to auto-detect latest quarter
                     If None, falls back to current date
        """
        quarters = []
        patterns = [
            r'([1-4])q\s*(\d{2,4})',     # 1Q24, 1Q 2024
            r'q([1-4])\s*(\d{2,4})',     # Q1 24, Q1 2024
            r'([1-4])q\s*fy\s*(\d{2,4})', # 1Q FY24
        ]
        
        query_lower = query.lower()
        
        # Check for relative quarter expressions: "last N quarters", "past N quarters", "previous N quarters"
        # IMPROVED: More flexible patterns
        relative_patterns = [
            r'\b(?:over|for|in)\s+(?:the\s+)?(?:last|past|previous|recent)\s+(\d+)\s+quarters?\b',
            r'\b(?:last|past|previous|recent)\s+(\d+)\s+quarters?\b',
            r'\b(\d+)\s+(?:last|past|previous|recent)\s+quarters?\b',
        ]
        
        # Determine reference quarter (most recent quarter)
        current_year = None
        current_quarter = None
        
        # Try to auto-detect from KB data
        if kb_path:
            try:
                kb_df = pd.read_parquet(kb_path)
                if 'quarter' in kb_df.columns and 'year' in kb_df.columns:
                    # Extract all valid quarters and find the latest
                    valid_quarters = kb_df.dropna(subset=['quarter', 'year'])
                    if not valid_quarters.empty:
                        # Parse quarter strings like "1Q2024" to compare
                        def parse_quarter(q_str, year):
                            if isinstance(q_str, str) and 'Q' in q_str:
                                q_num = int(q_str[0])
                                return (year, q_num)
                            return None
                        
                        quarter_tuples = [
                            parse_quarter(row['quarter'], row['year']) 
                            for _, row in valid_quarters.iterrows()
                        ]
                        quarter_tuples = [qt for qt in quarter_tuples if qt is not None]
                        
                        if quarter_tuples:
                            current_year, current_quarter = max(quarter_tuples)
            except Exception:
                pass  # Fallback to datetime
        
        # Fallback to current date if KB detection failed
        if current_year is None or current_quarter is None:
            from datetime import datetime
            current_year = datetime.now().year
            current_quarter = (datetime.now().month - 1) // 3 + 1
        
        for rel_pattern in relative_patterns:
            match = re.search(rel_pattern, query_lower)
            if match:
                n_quarters = int(match.group(1))
                # Generate last N quarters from reference point
                q, y = current_quarter, current_year
                for _ in range(n_quarters):
                    quarters.append(f"{q}Q{y}")
                    q -= 1
                    if q < 1:
                        q = 4
                        y -= 1
                break  # Only process first match
        
        # Extract specific quarters
        for pattern in patterns:
            for match in re.finditer(pattern, query_lower):
                q = int(match.group(1))
                y_str = match.group(2)
                year = int(y_str)
                if year < 100:
                    year += 2000
                if 2000 <= year <= 2030 and 1 <= q <= 4:
                    quarters.append(f"{q}Q{year}")
        
        # Also check for written forms: "second quarter 2024"
        quarter_words = {"first": 1, "second": 2, "third": 3, "fourth": 4, "1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
        for word, num in quarter_words.items():
            pattern = rf'\b{word}\s+quarter\s+(?:of\s+)?(\d{{4}})\b'
            for match in re.finditer(pattern, query_lower):
                year = int(match.group(1))
                if 2000 <= year <= 2030:
                    quarters.append(f"{num}Q{year}")
        
        return sorted(list(set(quarters)))
    
    @staticmethod
    def extract_doc_types(query: str) -> List[str]:
        """Detect document type preferences in query"""
        doc_types = []
        query_lower = query.lower()
        
        if re.search(r'\bannual\s+report', query_lower):
            doc_types.append("annual_report")
        if re.search(r'\bquarterly\s+(results|report|summary)', query_lower):
            doc_types.append("quarterly_results")
        if re.search(r'\bceo\s+presentation', query_lower):
            doc_types.append("ceo_presentation")
        if re.search(r'\bcfo\s+presentation', query_lower):
            doc_types.append("cfo_presentation")
        if re.search(r'\btrading\s+update', query_lower):
            doc_types.append("trading_update")
        if re.search(r'\bpress\s+(statement|release)', query_lower):
            doc_types.append("press_statement")
        
        # Boost quarterly docs if query mentions quarters/quarterly data
        # Match both "Q1" and "1Q" formats (with or without year), plus "quarter(s)" and "quarterly"
        if re.search(r'\b(quarter|quarters|quarterly|[1-4]q|q[1-4])', query_lower):
            # Add all quarterly document types if not already present
            quarterly_docs = ["quarterly_results", "cfo_presentation", "trading_update"]
            for doc in quarterly_docs:
                if doc not in doc_types:
                    doc_types.append(doc)
        
        return doc_types
    
    @staticmethod
    def extract_section_hints(query: str) -> List[str]:
        """Detect section preferences in query"""
        sections = []
        query_lower = query.lower()
        
        if re.search(r'\b(income|profit|loss)\s+statement', query_lower):
            sections.append("income_statement")
        if re.search(r'\bbalance\s+sheet', query_lower):
            sections.append("balance_sheet")
        if re.search(r'\bcash\s+flow', query_lower):
            sections.append("cash_flow")
        if re.search(r'\bsegment(al)?\s+(results|performance)', query_lower):
            sections.append("segment_results")
        if re.search(r'\brisk', query_lower):
            sections.append("risk_management")
        
        return sections
    
    @staticmethod
    def analyze(query: str, kb_path: Optional[str] = None) -> MetadataHints:
        """
        Extract all metadata hints from query
        
        Args:
            query: User query string
            kb_path: Optional path to kb_chunks.parquet for auto-detecting latest quarter
        """
        return MetadataHints(
            years=MetadataQueryAnalyzer.extract_years(query),
            quarters=MetadataQueryAnalyzer.extract_quarters(query, kb_path),
            doc_types=MetadataQueryAnalyzer.extract_doc_types(query),
            sections=MetadataQueryAnalyzer.extract_section_hints(query)
        )


class MetadataBooster:
    """Apply metadata-based score boosting to search results"""
    
    # Document type priority for different query intents
    DOC_TYPE_PRIORITY = {
        "metrics": ["quarterly_results", "performance_summary", "cfo_presentation", "annual_report"],
        "narrative": ["ceo_presentation", "annual_report", "trading_update", "press_statement"],
        "financial": ["annual_report", "quarterly_results", "performance_summary"],
        "comparison": ["quarterly_results", "performance_summary", "annual_report"],
    }
    
    @staticmethod
    def get_adaptive_boost_weights(query: str, hints: MetadataHints) -> Dict[str, float]:
        """
        Adjust boost weights based on query type and context
        
        Args:
            query: User query string
            hints: Extracted metadata hints
        
        Returns:
            Adaptive boost weight dictionary
        """
        query_lower = query.lower()
        
        # Default weights (more conservative than before)
        weights = {
            "quarter": 4.0,    # Reduced from 5.0
            "doc_type": 1.8,   # Reduced from 2.0
            "year": 1.6,       # Reduced from 1.8
            "section": 1.2     # Reduced from 1.3
        }
        
        # If query asks for specific quarter, boost quarter heavily
        if hints.quarters:
            weights["quarter"] = 6.0   # Reduced from 8.0
            weights["doc_type"] = 1.4  # Reduced from 1.5
        
        # If query is YoY comparison, boost year more
        yoy_terms = ["year-on-year", "yoy", "y-o-y", "versus", "vs", "compared to", "comparison"]
        if len(hints.years) > 1 or any(term in query_lower for term in yoy_terms):
            weights["year"] = 3.0      # Reduced from 3.5
            weights["quarter"] = 1.8   # Reduced from 2.0
            weights["doc_type"] = 2.2  # Reduced from 2.5
        
        # If query asks for annual data, prefer annual reports
        if "annual" in query_lower or "fiscal year" in query_lower or "fy" in query_lower:
            weights["doc_type"] = 3.5  # Reduced from 4.0
            weights["quarter"] = 1.0
            weights["year"] = 2.2      # Reduced from 2.5
        
        # If query mentions "latest" or "recent", boost recency more
        recency_terms = ["latest", "recent", "current", "last", "most recent"]
        if any(term in query_lower for term in recency_terms):
            weights["quarter"] = 5.5   # Reduced from 7.0
            weights["year"] = 2.2      # Reduced from 2.5
        
        # If query is about trends or multiple periods
        if any(term in query_lower for term in ["trend", "over time", "last n", "past"]):
            weights["quarter"] = 5.0   # Reduced from 6.0
            weights["year"] = 1.8      # Reduced from 2.0
        
        return weights
    
    @staticmethod
    def apply_recency_decay(
        results_df: pd.DataFrame,
        hints: MetadataHints,
        decay_factor: float = 0.95
    ) -> pd.DataFrame:
        """
        Apply exponential decay based on how old the document is
        relative to the most recent quarter/year in hints.
        
        Args:
            results_df: Search results with metadata
            hints: Extracted metadata hints
            decay_factor: Multiplier per quarter/year of age (0.95 = 5% decay per period)
                         Changed from 0.93 to 0.95 for less aggressive decay
        
        Returns:
            DataFrame with recency_boost applied to scores
        """
        if results_df.empty or 'year' not in results_df.columns:
            return results_df
        
        df = results_df.copy()
        
        # Find the most recent target from hints
        target_year = max(hints.years) if hints.years else int(df['year'].max())
        
        # Calculate quarter distance (if applicable)
        if hints.quarters and 'quarter' in df.columns:
            # Parse target quarter (e.g., "2Q2024" -> (2024, 2))
            target_q = hints.quarters[0]
            match = re.search(r'([1-4])Q(\d{4})', target_q)
            if match:
                target_year = int(match.group(2))
                target_qtr = int(match.group(1))
                
                def calc_quarter_distance(row):
                    if pd.isna(row['quarter']):
                        return 8  # ~2 years away if no quarter
                    m = re.search(r'([1-4])Q(\d{4})', str(row['quarter']))
                    if not m:
                        return 8
                    qtr = int(m.group(1))
                    year = int(m.group(2))
                    # Distance in quarters
                    return abs((target_year - year) * 4 + (target_qtr - qtr))
                
                df['_quarter_distance'] = df.apply(calc_quarter_distance, axis=1)
                df['recency_boost'] = decay_factor ** df['_quarter_distance']
                df.drop(columns=['_quarter_distance'], inplace=True)
            else:
                # No quarter parsing, use year only
                df['recency_boost'] = decay_factor ** (4 * (target_year - df['year'].fillna(target_year)))
        else:
            # Year-only decay (multiply by 4 to simulate quarters)
            df['recency_boost'] = decay_factor ** (4 * (target_year - df['year'].fillna(target_year)))
        
        # Apply recency boost to score
        if 'score' in df.columns:
            df['score'] = df['score'] * df['recency_boost']
        
        return df.sort_values('score', ascending=False)
    
    @staticmethod
    def score_doc_type_relevance(doc_type: str, query: str) -> float:
        """
        Score how relevant a doc type is for this query based on intent
        
        Args:
            doc_type: Document type string
            query: User query
        
        Returns:
            Relevance score (higher = better)
        """
        query_lower = query.lower()
        
        # Detect query intent
        if any(word in query_lower for word in ["calculate", "ratio", "margin", "revenue", "profit", "expense", "income"]):
            intent = "metrics"
        elif any(word in query_lower for word in ["strategy", "outlook", "plan", "initiative", "vision"]):
            intent = "narrative"
        elif any(word in query_lower for word in ["balance sheet", "cash flow", "statement", "financial position"]):
            intent = "financial"
        elif any(word in query_lower for word in ["versus", "vs", "compared", "comparison", "yoy", "year-on-year"]):
            intent = "comparison"
        else:
            intent = "metrics"  # Default
        
        # Return priority score based on position in priority list
        priority_list = MetadataBooster.DOC_TYPE_PRIORITY.get(intent, [])
        try:
            position = priority_list.index(doc_type)
            return 1.0 / (position + 1)  # 1.0, 0.5, 0.33, 0.25...
        except (KeyError, ValueError):
            return 0.5  # Unknown doc type gets mid-priority
    
    @staticmethod
    def apply_boost(
        results_df: pd.DataFrame,
        hints: MetadataHints,
        boost_weights: Optional[Dict[str, float]] = None,
        query: str = "",
        apply_recency: bool = True
    ) -> pd.DataFrame:
        """
        Apply metadata boosting to search results with adaptive weights
        
        Args:
            results_df: Search results with score and metadata columns
            hints: Extracted metadata from query
            boost_weights: Multiplicative boosts for each metadata type
                          If None, will use adaptive weights based on query
            query: Original query (needed for adaptive weights)
            apply_recency: Whether to apply recency decay
        
        Returns:
            DataFrame with adjusted 'score' column and 'boost_applied' flag
        """
        # Use adaptive weights if not provided
        if boost_weights is None and query:
            boost_weights = MetadataBooster.get_adaptive_boost_weights(query, hints)
        elif boost_weights is None:
            boost_weights = {
                "year": 1.6,
                "quarter": 4.0,
                "doc_type": 1.8,
                "section": 1.2
            }
        
        df = results_df.copy()
        df['boost_factor'] = 1.0
        df['boost_applied'] = ""
        
        # Year boosting
        if hints.years and 'year' in df.columns:
            year_mask = df['year'].isin(hints.years)
            df.loc[year_mask, 'boost_factor'] *= boost_weights.get("year", 1.8)
            df.loc[year_mask, 'boost_applied'] += "year "
        
        # Quarter boosting
        if hints.quarters and 'quarter' in df.columns:
            quarter_mask = df['quarter'].isin(hints.quarters)
            df.loc[quarter_mask, 'boost_factor'] *= boost_weights.get("quarter", 5.0)
            df.loc[quarter_mask, 'boost_applied'] += "quarter "
        
        # Document type boosting (enhanced with relevance scoring)
        if 'doc_type' in df.columns:
            # Base boost for matching doc types
            if hints.doc_types:
                doctype_mask = df['doc_type'].isin(hints.doc_types)
                df.loc[doctype_mask, 'boost_factor'] *= boost_weights.get("doc_type", 2.0)
                df.loc[doctype_mask, 'boost_applied'] += "doc_type "
            
            # Additional relevance-based boost (subtle)
            if query:
                df['_doc_relevance'] = df['doc_type'].apply(
                    lambda dt: MetadataBooster.score_doc_type_relevance(dt, query)
                )
                df['boost_factor'] *= (1.0 + df['_doc_relevance'] * 0.3)  # Up to 1.3x additional boost
                df.drop(columns=['_doc_relevance'], inplace=True)
        
        # Section boosting
        if hints.sections and 'section' in df.columns:
            section_mask = df['section'].isin(hints.sections)
            df.loc[section_mask, 'boost_factor'] *= boost_weights.get("section", 1.3)
            df.loc[section_mask, 'boost_applied'] += "section "
        
        # Apply boost to scores
        df['score'] = df['score'] * df['boost_factor']
        df['boost_applied'] = df['boost_applied'].str.strip()
        
        # Apply recency decay if requested
        if apply_recency:
            df = MetadataBooster.apply_recency_decay(df, hints, decay_factor=0.95)
        
        # Re-sort and re-rank
        df = df.sort_values('score', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    @staticmethod
    def filter_by_metadata(
        results_df: pd.DataFrame,
        hints: MetadataHints,
        strict: bool = False
    ) -> pd.DataFrame:
        """
        Filter results by metadata
        
        Args:
            results_df: Search results with metadata columns
            hints: Extracted metadata from query
            strict: If True, require ALL hints to match; if False, use OR logic
        
        Returns:
            Filtered DataFrame
        """
        df = results_df.copy()
        masks = []
        
        # Year filter
        if hints.years and 'year' in df.columns:
            masks.append(df['year'].isin(hints.years))
        
        # Quarter filter
        if hints.quarters and 'quarter' in df.columns:
            masks.append(df['quarter'].isin(hints.quarters))
        
        # Document type filter
        if hints.doc_types and 'doc_type' in df.columns:
            masks.append(df['doc_type'].isin(hints.doc_types))
        
        # Section filter
        if hints.sections and 'section' in df.columns:
            masks.append(df['section'].isin(hints.sections))
        
        if not masks:
            return df
        
        if strict:
            # AND logic: all conditions must match
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask &= mask
        else:
            # OR logic: any condition matches
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask |= mask
        
        filtered = df[combined_mask].reset_index(drop=True)
        filtered['rank'] = range(1, len(filtered) + 1)
        
        return filtered


# ============================================================================
# Standalone enhancement function
# ============================================================================

def enhance_kb_with_metadata(kb_base_path: str = "./data_marker"):
    """
    Add metadata columns to existing kb_chunks.parquet
    
    This function:
    1. Loads kb_chunks.parquet and kb_texts.npy
    2. Adds metadata columns (year, quarter, doc_type, section)
    3. Saves enhanced version back to kb_chunks.parquet
    """
    kb_path = Path(kb_base_path)
    chunks_path = kb_path / "kb_chunks.parquet"
    texts_path = kb_path / "kb_texts.npy"
    
    if not chunks_path.exists():
        raise FileNotFoundError(f"kb_chunks.parquet not found at {chunks_path}")
    if not texts_path.exists():
        raise FileNotFoundError(f"kb_texts.npy not found at {texts_path}")
    
    print(f"Loading chunks from {chunks_path}...")
    df = pd.read_parquet(chunks_path)
    
    print(f"Loading texts from {texts_path}...")
    texts = np.load(texts_path, allow_pickle=True).tolist()
    
    # Add text column temporarily for section extraction
    if len(texts) == len(df):
        df['text'] = texts
    else:
        print(f"Text count mismatch: {len(texts)} texts vs {len(df)} chunks")
        df['text'] = [texts[i] if i < len(texts) else "" for i in range(len(df))]
    
    print(f" Enhancing metadata...")
    df_enhanced = MetadataEnhancer.enhance_chunk_metadata(df)
    
    # Remove text column before saving (it's stored separately in kb_texts.npy)
    df_enhanced = df_enhanced.drop(columns=['text'], errors='ignore')
    
    # Show summary
    print(f"\nMetadata Enhancement Summary:")
    print(f"   Total chunks: {len(df_enhanced)}")
    if 'year' in df_enhanced.columns:
        year_counts = df_enhanced['year'].value_counts().sort_index()
        print(f"   Years found: {dict(year_counts.head(10))}")
    if 'quarter' in df_enhanced.columns:
        quarter_counts = df_enhanced['quarter'].value_counts()
        print(f"   Quarters found: {len(quarter_counts)} unique quarters")
    if 'doc_type' in df_enhanced.columns:
        doctype_counts = df_enhanced['doc_type'].value_counts()
        print(f"   Doc types: {dict(doctype_counts)}")
    
    # Save back
    print(f"\nSaving enhanced chunks to {chunks_path}...")
    df_enhanced.to_parquet(chunks_path, engine="pyarrow", index=False)
    


# if __name__ == "__main__":
#     # Example usage
#     print("=" * 60)
#     print("Metadata Enhancement for RAG System")
#     print("=" * 60)
    
#     # Enhance KB
#     enhance_kb_with_metadata("./data_marker")
    
#     # Test query analysis
#     print("\n" + "=" * 60)
#     print("Query Analysis Examples")
#     print("=" * 60)
    
#     test_queries = [
#         "What was revenue in Q2 2024?",
#         "Compare operating expenses in 2023 vs 2024 annual reports",
#         "Show net interest margin for 1Q24 and 2Q24",
#         "What's in the CEO presentation for FY24?",
#     ]
    
#     for query in test_queries:
#         hints = MetadataQueryAnalyzer.analyze(query)
#         print(f"\nQuery: {query}")
#         print(f"  Years: {hints.years}")
#         print(f"  Quarters: {hints.quarters}")
#         print(f"  Doc types: {hints.doc_types}")
#         print(f"  Sections: {hints.sections}")
