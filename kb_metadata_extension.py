"""
KBEnv Extension: Metadata-Enhanced Search
==========================================

This module extends KBEnv with metadata filtering and boosting capabilities.

Add this cell to your notebook AFTER the KBEnv class definition to enable metadata search.
"""

from typing import Optional, Dict, List
import pandas as pd
from metadata_enhancer import MetadataQueryAnalyzer, MetadataBooster, MetadataHints


# ============================================================================
# Extension: Add metadata-aware search to KBEnv
# ============================================================================

def search_with_metadata(
    self,
    query: str,
    k: int = 12,
    alpha: float = 0.6,
    rerank_top_k: Optional[int] = None,
    enable_metadata_boost: bool = True,
    enable_metadata_filter: bool = False,
    boost_weights: Optional[Dict[str, float]] = None,
    filter_strict: bool = False
) -> pd.DataFrame:
    """
    Enhanced search with metadata filtering and boosting
    
    Args:
        query: Search query
        k: Number of final results to return
        alpha: Weight for vector vs BM25 (used in base search)
        rerank_top_k: Number of candidates for reranking
        enable_metadata_boost: Apply metadata-based score boosting
        enable_metadata_filter: Pre-filter results by metadata
        boost_weights: Custom boost multipliers (default: year=1.5, quarter=2.0, etc.)
        filter_strict: If True, require ALL metadata hints to match (AND logic)
    
    Returns:
        DataFrame with search results, metadata columns, and boost info
    """
    # Step 1: Extract metadata hints from query (auto-detect latest quarter from KB)
    kb_path = str(self.chunks_path) if hasattr(self, 'chunks_path') else None
    hints = MetadataQueryAnalyzer.analyze(query, kb_path)
    
    has_metadata_hints = any([
        hints.years,
        hints.quarters,
        hints.doc_types,
        hints.sections
    ])
    
    if not has_metadata_hints:
        # No metadata hints in query, use standard search
        return self.search(query, k=k, alpha=alpha, rerank_top_k=rerank_top_k)
    
    # Log detected metadata (if verbose mode exists in self)
    if hasattr(self, 'verbose') and getattr(self, 'verbose', False):
        print(f"[Metadata] Detected hints:")
        if hints.years:
            print(f"  Years: {hints.years}")
        if hints.quarters:
            print(f"  Quarters: {hints.quarters}")
        if hints.doc_types:
            print(f"  Doc types: {hints.doc_types}")
        if hints.sections:
            print(f"  Sections: {hints.sections}")
    
    # Step 2: Get initial results (LARGER pool for filtering/boosting)
    # Use k*10 to give enough room for low-ranked but metadata-matching docs to get boosted to top
    initial_k = k * 10 if enable_metadata_filter or enable_metadata_boost else k
    results = self.search(query, k=initial_k, alpha=alpha, rerank_top_k=rerank_top_k)

    # If the base KBEnv.search() doesn't include metadata columns, merge them
    # from self.meta_df using (doc, chunk) as the join key. This avoids needing
    # a notebook-level monkey-patch for metadata to be present.
    try:
        if results is not None and not results.empty and hasattr(self, 'meta_df'):
            # Check if metadata columns are missing
            needs_merge = not all(col in results.columns for col in ['year', 'quarter', 'doc_type', 'section'])
            
            if needs_merge:
                meta = self.meta_df.loc[:, ['doc', 'chunk', 'year', 'quarter', 'doc_type', 'section']].copy()
                # Normalize chunk types for reliable join
                if 'chunk' in meta.columns:
                    try:
                        meta['chunk'] = meta['chunk'].fillna(-1).astype(int)
                    except Exception:
                        pass
                if 'chunk' in results.columns:
                    try:
                        results['chunk'] = results['chunk'].fillna(-1).astype(int)
                    except Exception:
                        pass
                # Merge metadata
                results = results.merge(meta, on=['doc', 'chunk'], how='left', suffixes=('', '_meta'))
                # If suffixes were added (duplicate columns), drop the originals and rename
                for col in ['year', 'quarter', 'doc_type', 'section']:
                    if f'{col}_meta' in results.columns:
                        results[col] = results[f'{col}_meta']
                        results.drop(columns=[f'{col}_meta'], inplace=True)
    except Exception as e:
        # If merging fails, continue and allow downstream logic to handle it
        if hasattr(self, 'verbose') and getattr(self, 'verbose', False):
            print(f"[Metadata] Warning: merge failed: {e}")
        pass

    if results is None or results.empty:
        return results

    # Step 3: Apply metadata filtering (optional)
    if enable_metadata_filter:
        filtered = MetadataBooster.filter_by_metadata(results, hints, strict=filter_strict)

        if hasattr(self, 'verbose') and getattr(self, 'verbose', False):
            print(f"[Metadata] Filtered: {len(results)} → {len(filtered)} results")

        # If filtering removed too many results, fall back to unfiltered
        if len(filtered) < k // 2:
            if hasattr(self, 'verbose') and getattr(self, 'verbose', False):
                print(f"[Metadata] Too few filtered results, using unfiltered")
            filtered = results

        results = filtered

    # Step 4: Apply metadata boosting (optional)
    if enable_metadata_boost:
        results = MetadataBooster.apply_boost(results, hints, boost_weights)

        if hasattr(self, 'verbose') and getattr(self, 'verbose', False):
            boosted_count = (results.get('boost_factor', 1.0) > 1.0).sum()
            print(f"[Metadata] Boosted {boosted_count}/{len(results)} results")

    # Step 5: Return top-k
    final_results = results.head(k).reset_index(drop=True)
    final_results['rank'] = range(1, len(final_results) + 1)

    return final_results


# Monkey-patch the method onto KBEnv
def add_metadata_search_to_kbenv():
    """
    Add search_with_metadata method to KBEnv class
    
    Call this after KBEnv is defined in your notebook.
    """
    # Import KBEnv from the global namespace
    import sys
    if 'KBEnv' in globals():
        KBEnv = globals()['KBEnv']
    elif '__main__' in sys.modules and hasattr(sys.modules['__main__'], 'KBEnv'):
        KBEnv = sys.modules['__main__'].KBEnv
    else:
        raise RuntimeError("KBEnv class not found. Make sure it's defined before calling this function.")
    
    # Add the method
    KBEnv.search_with_metadata = search_with_metadata
    print("Added 'search_with_metadata' method to KBEnv class")


# ============================================================================
# Convenience wrapper for Agent integration
# ============================================================================

class MetadataAwareAgent:
    """
    Wrapper to make Agent use metadata-enhanced search automatically
    
    Usage:
        agent = MetadataAwareAgent(kb, enable_metadata_boost=True)
        result = agent.run("What was revenue in Q2 2024?")
    """
    
    def __init__(
        self,
        kb,
        enable_metadata_boost: bool = True,
        enable_metadata_filter: bool = False,
        boost_weights: Optional[Dict[str, float]] = None,
        use_parallel_subqueries: bool = False,
        verbose: bool = True
    ):
        """Initialize metadata-aware agent"""
        # Store original kb.search method
        self._original_search = kb.search
        self.kb = kb
        self.enable_metadata_boost = enable_metadata_boost
        self.enable_metadata_filter = enable_metadata_filter
        self.boost_weights = boost_weights
        
        # Add metadata search capability if not exists
        if not hasattr(kb, 'search_with_metadata'):
            kb.search_with_metadata = lambda *args, **kwargs: search_with_metadata(kb, *args, **kwargs)
        
        # Temporarily replace kb.search with metadata version
        def _wrapped_search(query: str, k: int = 12, **kwargs):
            return kb.search_with_metadata(
                query=query,
                k=k,
                enable_metadata_boost=self.enable_metadata_boost,
                enable_metadata_filter=self.enable_metadata_filter,
                boost_weights=self.boost_weights,
                **kwargs
            )
        
        kb.search = _wrapped_search
        
        # Import and initialize the base Agent
        import sys
        if 'Agent' in globals():
            AgentClass = globals()['Agent']
        elif '__main__' in sys.modules and hasattr(sys.modules['__main__'], 'Agent'):
            AgentClass = sys.modules['__main__'].Agent
        else:
            raise RuntimeError("Agent class not found")
        
        self.agent = AgentClass(kb, use_parallel_subqueries=use_parallel_subqueries, verbose=verbose)
    
    def run(self, query: str, k_ctx: int = 12):
        """Run agent with metadata-enhanced search"""
        return self.agent.run(query, k_ctx=k_ctx)
    
    def __del__(self):
        """Restore original search method"""
        if hasattr(self, 'kb') and hasattr(self, '_original_search'):
            self.kb.search = self._original_search
