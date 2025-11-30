"""
ReAct Agent CFO - Enhanced with Parallel Query Decomposition

Key Enhancements:
1. Parallel tool execution for independent operations
2. Query decomposition for complex multi-metric queries  
3. Async/sync hybrid execution compatible with Jupyter
4. Read/write operation classification for safe concurrency
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import json
import time
import numpy as np
import g2x
import re
import asyncio
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================================
# QUERY ANALYZER
# ============================================================================

class QueryAnalyzer:
    """Lightweight query understanding"""
    
    @staticmethod
    def needs_calculation(query: str) -> bool:
        calc_keywords = ['ratio', 'efficiency', 'calculate', 'compute', '÷', '/', 'divide']
        return any(kw in query.lower() for kw in calc_keywords)
    
    @staticmethod
    def extract_metric(query: str) -> Optional[str]:
        """Extract primary metric from query"""
        patterns = [
            r'(Operating Expenses?|Opex)',
            r'(Operating Income|Total Income)',
            r'(Net Interest Margin|NIM|Gross Margin)',
            r'(Efficiency Ratio)',
            r'(Revenue|Profit|Expenses?)'
        ]
        for pattern in patterns:
            match = re.search(pattern, query, re.I)
            if match:
                return match.group(1)
        return None
    
    @staticmethod
    def extract_years(query: str) -> List[int]:
        """Extract year mentions"""
        years = re.findall(r'\b(20\d{2})\b', query)
        return sorted(set(int(y) for y in years))
    
    @staticmethod
    def extract_num_periods(query: str) -> Optional[int]:
        """Extract number of periods (e.g., 'last 3 years')"""
        match = re.search(r'(?:last|past)\s+(\d+)\s+(?:year|quarter|period)', query, re.I)
        return int(match.group(1)) if match else None
    
    @staticmethod
    def want_compare(query: str) -> bool:
        """Check if query wants comparison"""
        compare_keywords = ['compare', 'comparison', 'versus', 'vs', 'year-over-year', 'yoy']
        return any(kw in query.lower() for kw in compare_keywords)


# ============================================================================
# PARALLEL QUERY DECOMPOSER
# ============================================================================

class ParallelQueryDecomposer:
    """Decomposes complex queries for parallel execution"""
    
    @staticmethod
    def decompose(query: str) -> List[str]:
        """
        Intelligent query decomposition using query analysis
        Returns list of independent sub-queries that can run in parallel
        """
        analyzer = QueryAnalyzer
        
        needs_calc = analyzer.needs_calculation(query)
        metric = analyzer.extract_metric(query)
        years = analyzer.extract_years(query)
        num_periods = analyzer.extract_num_periods(query)
        
        # Q3: Efficiency Ratio (Opex ÷ Income) - decompose into 2 parallel extractions
        if needs_calc and metric and "efficiency" in metric.lower():
            return [
                f"Extract Operating Expenses for the last {num_periods or 3} fiscal years",
                f"Extract Total Income for the last {num_periods or 3} fiscal years"
            ]
        
        # General ratio calculation (A ÷ B)
        if needs_calc and ("ratio" in query.lower() or "÷" in query or "/" in query):
            parts = re.split(r'[÷/]|\bdivided by\b', query, flags=re.I)
            if len(parts) == 2:
                metric_a = analyzer.extract_metric(parts[0])
                metric_b = analyzer.extract_metric(parts[1])
                if metric_a and metric_b:
                    return [
                        f"Extract {metric_a} for the last {num_periods or 3} fiscal years",
                        f"Extract {metric_b} for the last {num_periods or 3} fiscal years"
                    ]
        
        # Multi-metric comparison across years
        if analyzer.want_compare(query) and metric and len(years) > 2:
            return [f"Extract {metric} for FY{y}" for y in years]
        
        # Single metric query (no decomposition needed)
        return [query]


# ============================================================================
# AUTO-DETECTION UTILITIES
# ============================================================================

class DataIntrospector:
    """Automatically detect available metrics and time periods from KB"""
    
    def __init__(self, tables_df: pd.DataFrame):
        self.df = tables_df
        self._cache = {}
    
    def detect_quarters(self, n: int = 5) -> List[str]:
        """Auto-detect last N quarters from data"""
        if 'quarters' in self._cache:
            return self._cache['quarters']
        
        quarter_pattern = r'\b([1-4]Q\d{2})\b'
        all_quarters = set()
        
        for col in self.df['column'].dropna():
            matches = re.findall(quarter_pattern, str(col))
            all_quarters.update(matches)
        
        def sort_key(q):
            match = re.match(r'([1-4])Q(\d{2})', q)
            if match:
                return (int(match.group(2)), int(match.group(1)))
            return (0, 0)
        
        sorted_quarters = sorted(all_quarters, key=sort_key)[-n:]
        self._cache['quarters'] = sorted_quarters
        return sorted_quarters
    
    def detect_years(self, n: int = 3) -> List[int]:
        """Auto-detect last N years from annual reports"""
        if 'years' in self._cache:
            return self._cache['years']
        
        year_pattern = r'annual-report-(\d{4})'
        all_years = set()
        
        for doc in self.df['doc_name'].unique():
            match = re.search(year_pattern, str(doc))
            if match:
                all_years.add(int(match.group(1)))
        
        sorted_years = sorted(all_years)[-n:]
        self._cache['years'] = sorted_years
        return sorted_years
    
    def detect_document_patterns(self) -> Dict[str, str]:
        """Detect document naming patterns"""
        if 'doc_patterns' in self._cache:
            return self._cache['doc_patterns']
        
        patterns = {
            'cfo_quarterly': None,
            'annual_report': None,
            'company_name': None
        }
        
        sample_docs = self.df['doc_name'].unique()[:50]
        
        for doc in sample_docs:
            if 'CFO' in doc and 'Q' in doc:
                patterns['cfo_quarterly'] = '{period}_CFO_presentation'
            
            if 'annual-report' in doc:
                match = re.match(r'([a-z]+)-annual-report-\d{4}', doc)
                if match:
                    patterns['company_name'] = match.group(1)
                    patterns['annual_report'] = f'{match.group(1)}-annual-report-' + '{year}'
        
        self._cache['doc_patterns'] = patterns
        return patterns
    
    def suggest_metric_keywords(self, metric_name: str) -> List[str]:
        """Suggest keywords for a metric based on data"""
        metric_name_lower = metric_name.lower()
        
        keyword_map = {
            'nim': ['Group NIM (%)', 'Commercial NIM (%)', 'Net Interest Margin', 'NIM'],
            'net interest margin': ['Group NIM (%)', 'Commercial NIM (%)', 'Net Interest Margin', 'NIM'],
            'gross margin': ['Group NIM (%)', 'Gross Margin'],
            'income': ['Total income', 'Operating income', 'Net income'],
            'expense': ['Total expenses', 'Operating expenses', 'Opex'],
            'revenue': ['Total revenue', 'Revenue', 'Total income'],
            'profit': ['Profit', 'Net profit', 'Profit before tax']
        }
        
        for key, keywords in keyword_map.items():
            if key in metric_name_lower:
                return keywords
        
        return [metric_name]


# ============================================================================
# TOOLS WITH AUTO-DETECTION
# ============================================================================

@dataclass
class ToolCall:
    """Records a tool call"""
    tool_name: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    latency_ms: float
    success: bool = True
    error: Optional[str] = None
    parallel_group: Optional[int] = None  # NEW: Track parallel execution groups


class SmartTableParser:
    """Intelligent table parser with proven extraction logic"""
    
    def __init__(self, tables_df: pd.DataFrame, introspector: DataIntrospector):
        self.df = tables_df
        self.introspector = introspector
        self.name = "SmartTableParser"
        self.is_read_only = True  # NEW: Classification for parallel execution
    
    def parse(self, metric: str, periods: Optional[List[str]] = None, 
              doc_pattern: Optional[str] = None) -> Dict[str, Any]:
        """Parse financial metric using PROVEN extraction logic from Agent CFO"""
        start_time = time.time()
        
        try:
            # Auto-detect if not provided
            if periods is None:
                if any(q in metric.lower() for q in ['nim', 'margin', 'quarterly']):
                    periods = self.introspector.detect_quarters()
                else:
                    periods = [str(y) for y in self.introspector.detect_years()]
            
            keywords = self.introspector.suggest_metric_keywords(metric)
            
            if doc_pattern is None:
                doc_pattern = 'dbs-annual-report'
            
            # Extract data using PROVEN Agent CFO logic
            results = {}
            sources = []
            
            for period in periods:
                # Quarterly data from CFO presentations
                if 'Q' in period and len(period) <= 4:
                    nim_rows = self.df[
                        (self.df['doc_name'].str.contains(f"{period}_CFO_presentation", na=False)) &
                        (self.df['column'].str.contains('Group NIM', case=False, na=False))
                    ]
                    
                    if not nim_rows.empty:
                        tid = nim_rows['table_id'].iloc[0]
                        table_data = self.df[
                            (self.df['doc_name'].str.contains(f"{period}_CFO_presentation", na=False)) &
                            (self.df['table_id'] == tid)
                        ]
                        
                        for row_id in table_data['row_id'].unique():
                            row = table_data[table_data['row_id'] == row_id]
                            quarter_cells = row[row['column'].str.contains('Quarter', case=False, na=False)]
                            if not quarter_cells.empty:
                                quarter_val = quarter_cells.iloc[0]['value_str']
                                if period in str(quarter_val):
                                    nim_cells = row[row['column'].str.contains('Group NIM', case=False, na=False)]
                                    if not nim_cells.empty and pd.notna(nim_cells.iloc[0]['value_num']):
                                        results[period] = float(nim_cells.iloc[0]['value_num'])
                                        sources.append({
                                            'file': nim_cells.iloc[0]['doc_name'],
                                            'page': int(nim_cells.iloc[0]['page']) if pd.notna(nim_cells.iloc[0]['page']) else None,
                                            'table_id': int(tid)
                                        })
                                        break
                
                # Annual data
                else:
                    metric_rows = self.df[
                        (self.df['doc_name'].str.contains(f'{doc_pattern}-{period}', na=False)) &
                        (self.df['value_str'].str.contains('|'.join(keywords), case=False, na=False, regex=True))
                    ]
                    
                    if not metric_rows.empty:
                        for _, row in metric_rows.iterrows():
                            table_data = self.df[
                                (self.df['doc_name'] == row['doc_name']) &
                                (self.df['table_id'] == row['table_id']) &
                                (self.df['row_id'] == row['row_id'])
                            ]
                            
                            # For income: prioritize columns with year or "Total"
                            if 'income' in '|'.join(keywords).lower():
                                candidates = []
                                for _, cell in table_data.iterrows():
                                    col_name = str(cell['column']).lower()
                                    if pd.notna(cell['value_num']) and cell['value_num'] > 10000:
                                        if period in col_name or 'total' in col_name:
                                            candidates.append((3, cell['value_num'], cell))
                                        else:
                                            candidates.append((1, cell['value_num'], cell))
                                
                                if candidates:
                                    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
                                    results[period] = float(candidates[0][1])
                                    sources.append({
                                        'file': candidates[0][2]['doc_name'],
                                        'page': int(candidates[0][2]['page']) if pd.notna(candidates[0][2]['page']) else None,
                                        'table_id': int(row['table_id'])
                                    })
                                    break
                            
                            # For expenses: first numeric value > 1000
                            else:
                                nums = table_data[table_data['value_num'].notna() & (table_data['value_num'] > 1000)]
                                if not nums.empty:
                                    results[period] = float(nums.iloc[0]['value_num'])
                                    sources.append({
                                        'file': nums.iloc[0]['doc_name'],
                                        'page': int(nums.iloc[0]['page']) if pd.notna(nums.iloc[0]['page']) else None,
                                        'table_id': int(row['table_id'])
                                    })
                                    break
            
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                'data': results,
                'sources': sources,
                'latency_ms': round(latency_ms, 2),
                'periods': periods,
                'keywords_used': keywords
            }
        
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return {
                'data': {},
                'sources': [],
                'latency_ms': round(latency_ms, 2),
                'error': str(e)
            }


class AdvancedCalculator:
    """Calculator with common financial computations"""
    
    def __init__(self):
        self.name = "AdvancedCalculator"
        self.is_read_only = True  # NEW: Calculations don't modify data
    
    def compute(self, operation: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform calculation"""
        start_time = time.time()
        
        try:
            if operation == 'ratio':
                result = self._compute_ratio(data['numerator'], data['denominator'])
            elif operation == 'yoy_change':
                result = self._compute_yoy(data['values'])
            elif operation == 'average':
                result = self._compute_average(data['values'])
            elif operation == 'growth_rate':
                result = self._compute_growth_rate(data['values'])
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            latency_ms = (time.time() - start_time) * 1000
            return {
                'result': result,
                'latency_ms': round(latency_ms, 2)
            }
        
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return {
                'result': None,
                'latency_ms': round(latency_ms, 2),
                'error': str(e)
            }
    
    def _compute_ratio(self, numerator: Dict[str, float], 
                       denominator: Dict[str, float]) -> Dict[str, float]:
        """Compute ratio for each period"""
        result = {}
        for period in numerator.keys():
            if period in denominator and denominator[period] != 0:
                result[period] = round((numerator[period] / denominator[period]) * 100, 2)
        return result
    
    def _compute_yoy(self, values: Dict[str, float]) -> Dict[str, float]:
        """Compute year-over-year changes"""
        sorted_periods = sorted(values.keys())
        result = {}
        
        for i in range(1, len(sorted_periods)):
            prev = values[sorted_periods[i-1]]
            curr = values[sorted_periods[i]]
            change = ((curr - prev) / prev) * 100
            result[f"{sorted_periods[i-1]}→{sorted_periods[i]}"] = round(change, 2)
        
        return result
    
    def _compute_average(self, values: Dict[str, float]) -> float:
        """Compute average"""
        return round(sum(values.values()) / len(values), 2)
    
    def _compute_growth_rate(self, values: Dict[str, float]) -> float:
        """Compute CAGR"""
        sorted_periods = sorted(values.keys())
        start_val = values[sorted_periods[0]]
        end_val = values[sorted_periods[-1]]
        n = len(sorted_periods) - 1
        
        if n > 0 and start_val > 0:
            cagr = ((end_val / start_val) ** (1/n) - 1) * 100
            return round(cagr, 2)
        return 0.0


class SmartTrendAnalyzer:
    """Analyze patterns in financial data"""
    
    def __init__(self):
        self.name = "SmartTrendAnalyzer"
        self.is_read_only = True  # NEW: Analysis doesn't modify data
    
    def analyze(self, values: Dict[str, float]) -> Dict[str, Any]:
        """Analyze trend pattern"""
        start_time = time.time()
        
        if len(values) < 2:
            return {
                'pattern': 'Insufficient Data',
                'latency_ms': round((time.time() - start_time) * 1000, 2)
            }
        
        sorted_periods = sorted(values.keys())
        sorted_values = [values[p] for p in sorted_periods]
        
        # Detect pattern
        increasing = all(sorted_values[i] <= sorted_values[i+1] for i in range(len(sorted_values)-1))
        decreasing = all(sorted_values[i] >= sorted_values[i+1] for i in range(len(sorted_values)-1))
        
        if increasing:
            pattern = "Consistently Increasing"
        elif decreasing:
            pattern = "Consistently Decreasing"
        else:
            pattern = "Fluctuating"
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            'pattern': pattern,
            'min': round(min(sorted_values), 2),
            'max': round(max(sorted_values), 2),
            'avg': round(sum(sorted_values) / len(sorted_values), 2),
            'range': round(max(sorted_values) - min(sorted_values), 2),
            'latency_ms': round(latency_ms, 2)
        }


# ============================================================================
# PARALLEL EXECUTION ENGINE
# ============================================================================

class ParallelExecutor:
    """Manages parallel tool execution with dependency resolution"""
    
    @staticmethod
    def identify_parallel_groups(steps: List[Dict[str, Any]]) -> List[int]:
        """
        Identify which steps can run in parallel
        Returns group IDs for each step (same ID = can run in parallel)
        """
        groups = []
        current_group = 0
        
        for i, step in enumerate(steps):
            inputs = step.get('inputs', {})
            
            # Check if this step depends on previous steps
            has_dependency = any(
                isinstance(v, str) and v.startswith('$step')
                for v in inputs.values()
            )
            
            if has_dependency:
                # Start new sequential group
                current_group += 1
                groups.append(current_group)
                current_group += 1
            else:
                # Can run in parallel with other non-dependent steps
                groups.append(current_group)
        
        return groups
    
    @staticmethod
    async def execute_parallel_async(tool_calls: List[Tuple[str, Dict, Any]]) -> List[Dict[str, Any]]:
        """
        Execute multiple tool calls in parallel using asyncio
        
        Args:
            tool_calls: List of (tool_name, inputs, tool_instance) tuples
        """
        loop = asyncio.get_event_loop()
        
        def execute_sync(tool_name, inputs, tool_instance):
            """Wrapper for synchronous tool execution"""
            if tool_name == 'SmartTableParser':
                return tool_instance.parse(**inputs)
            elif tool_name == 'AdvancedCalculator':
                return tool_instance.compute(**inputs)
            elif tool_name == 'SmartTrendAnalyzer':
                return tool_instance.analyze(**inputs)
            else:
                return {'error': f'Unknown tool: {tool_name}'}
        
        # Execute all tools in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(tool_calls), 4)) as executor:
            tasks = [
                loop.run_in_executor(executor, execute_sync, name, inputs, tool)
                for name, inputs, tool in tool_calls
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'error': str(result),
                    'latency_ms': 0
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    @staticmethod
    def execute_parallel(tool_calls: List[Tuple[str, Dict, Any]]) -> List[Dict[str, Any]]:
        """
        Blocking wrapper for parallel execution
        Compatible with both regular Python and Jupyter notebooks
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in event loop (Jupyter), use nest_asyncio
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                except ImportError:
                    print("[Warning] nest_asyncio not available, falling back to sequential execution")
                    return ParallelExecutor._execute_sequential(tool_calls)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            ParallelExecutor.execute_parallel_async(tool_calls)
        )
    
    @staticmethod
    def _execute_sequential(tool_calls: List[Tuple[str, Dict, Any]]) -> List[Dict[str, Any]]:
        """Fallback sequential execution"""
        results = []
        for tool_name, inputs, tool_instance in tool_calls:
            if tool_name == 'SmartTableParser':
                result = tool_instance.parse(**inputs)
            elif tool_name == 'AdvancedCalculator':
                result = tool_instance.compute(**inputs)
            elif tool_name == 'SmartTrendAnalyzer':
                result = tool_instance.analyze(**inputs)
            else:
                result = {'error': f'Unknown tool: {tool_name}'}
            results.append(result)
        return results


# ============================================================================
# REACT AGENT WITH PARALLEL EXECUTION
# ============================================================================

class ReActAgentCFO:
    """
    ReAct-based Agent with parallel tool execution
    
    Features:
    - LLM-driven planning with few-shot examples
    - Automatic identification of parallel execution opportunities
    - Read-only operation batching for concurrent execution
    - Query decomposition for complex multi-metric queries
    """
    
    def __init__(self, tables_df: pd.DataFrame, llm_client_tuple: Tuple):
        self.provider, self.client, self.model = llm_client_tuple
        
        # Initialize introspection
        self.introspector = DataIntrospector(tables_df)
        
        # Initialize tools
        self.parser = SmartTableParser(tables_df, self.introspector)
        self.calculator = AdvancedCalculator()
        self.analyzer = SmartTrendAnalyzer()
        
        self.tools = {
            'SmartTableParser': self.parser,
            'AdvancedCalculator': self.calculator,
            'SmartTrendAnalyzer': self.analyzer
        }
        
        self.tool_calls = []
        self.parallel_executor = ParallelExecutor()
        self.token_usage = {'total_tokens': 0}  
    
    def _track_usage(self, response):
        """Helper to capture tokens from response objects"""
        try:
            if self.provider == 'groq':
                if hasattr(response, 'usage') and response.usage:
                    self.token_usage['total_tokens'] += response.usage.total_tokens
            else: # Gemini
                if hasattr(response, 'usage_metadata'):
                    self.token_usage['total_tokens'] += response.usage_metadata.total_token_count
        except Exception:
            pass
        
    def run(self, query: str, enable_parallel: bool = True) -> Dict[str, Any]:
        self.token_usage = {'total_tokens': 0} # Reset
        self.tool_calls = [] # Reset
        
        t_total_start = time.time()
        t_ingest = getattr(self.introspector.df, 't_ingest', 0)
        
        t_reason_start = time.time()
        plan = self._generate_plan(query)
        t_reason = (time.time() - t_reason_start) * 1000

        if 'error' in plan:
            return {
                'answer': f"Error: {plan['error']}",
                'timings': {'T_ingest': t_ingest, 'T_retrieve': 0, 'T_reason': t_reason, 'T_generate': 0, 'T_total': t_reason},
                'tokens': self.token_usage,
                'tools': []
            }
        
        t_retrieve_start = time.time()
        if enable_parallel: execution_results = self._execute_plan_parallel(plan)
        else: execution_results = self._execute_plan_sequential(plan)
        t_retrieve = (time.time() - t_retrieve_start) * 1000
        
        t_generate_start = time.time()
        answer = self._generate_answer(query, plan, execution_results)
        t_generate = (time.time() - t_generate_start) * 1000
        
        t_total = (time.time() - t_total_start) * 1000
        
        # Extract tool names for logging
        tool_names = [tc.tool_name for tc in self.tool_calls]

        return {
            'answer': answer,
            'plan': plan,
            'tool_calls': self.tool_calls,
            'latency_ms': round(t_total, 2),
            'timings': {
                'T_ingest': t_ingest,
                'T_retrieve': round(t_retrieve, 2),
                'T_reason': round(t_reason, 2),
                'T_generate': round(t_generate, 2),
                'T_total': round(t_total, 2)
            },
            'tokens': self.token_usage,
            'tools': tool_names,
            'parallel_groups': len(set(tc.parallel_group for tc in self.tool_calls if tc.parallel_group is not None))
        }
        
    def _generate_plan(self, query: str) -> Dict[str, Any]:
        """LLM generates execution plan using few-shot examples"""
        
        # Get available data context
        quarters = self.introspector.detect_quarters()
        years = self.introspector.detect_years()
        doc_patterns = self.introspector.detect_document_patterns()
        
        system_prompt = f"""You are a financial analysis planning agent. Analyze the query and create an execution plan.

        Available Tools:
        1. SmartTableParser: Extract metrics from financial documents (READ-ONLY, parallelizable)
        - Input: {{"metric": "metric name"}}
        - Returns: {{"data": {{"period": value}}, "sources": [...]}}

        2. AdvancedCalculator: Perform calculations (READ-ONLY, parallelizable)
        - Operations: ratio, yoy_change, average, growth_rate
        - Input: {{"operation": "type", "data": {{...}}}}
        - Returns: {{"result": {{...}}}}

        3. SmartTrendAnalyzer: Analyze patterns (READ-ONLY, parallelizable)
        - Input: {{"values": {{"period": value}}}}
        - Returns: {{"pattern": "...", "min": x, "max": y, "avg": z}}

        Context:
        - Available quarters: {quarters}
        - Available years: {years}
        - Company: {doc_patterns.get('company_name', 'unknown')}

        IMPORTANT: Multiple SmartTableParser calls with no dependencies CAN RUN IN PARALLEL.
        For ratio calculations, extract numerator and denominator as separate parallel steps.

        Output JSON:
        {{
        "reasoning": "step-by-step thought process",
        "steps": [
            {{"tool": "ToolName", "inputs": {{...}}, "purpose": "why this step"}},
            ...
        ]
        }}
        """

        few_shot_examples = """
        Examples:

        Query: "What is the Net Interest Margin for the last 5 quarters?"
        Plan:
        {
        "reasoning": "Query asks for NIM over 5 quarters. I need to: (1) Extract NIM data using SmartTableParser, (2) Analyze the trend.",
        "steps": [
            {"tool": "SmartTableParser", "inputs": {"metric": "Net Interest Margin"}, "purpose": "Extract NIM values for recent quarters"},
            {"tool": "SmartTrendAnalyzer", "inputs": {"values": "$step1.data"}, "purpose": "Identify pattern in NIM"}
        ]
        }

        Query: "Calculate Operating Efficiency Ratio for the last 3 years"
        Plan:
        {
        "reasoning": "Efficiency ratio = Opex ÷ Income. These are INDEPENDENT extractions that can run in PARALLEL. Steps: (1) Get expenses in parallel with (2) Get income, then (3) Calculate ratio, (4) Analyze trend.",
        "steps": [
            {"tool": "SmartTableParser", "inputs": {"metric": "Operating Expenses"}, "purpose": "Extract expenses (can run in parallel)"},
            {"tool": "SmartTableParser", "inputs": {"metric": "Operating Income"}, "purpose": "Extract income (can run in parallel)"},
            {"tool": "AdvancedCalculator", "inputs": {"operation": "ratio", "data": {"numerator": "$step1.data", "denominator": "$step2.data"}}, "purpose": "Compute efficiency ratio"},
            {"tool": "SmartTrendAnalyzer", "inputs": {"values": "$step3.result"}, "purpose": "Analyze ratio trend"}
        ]
        }

        Query: "Show Operating Expenses year-over-year for 3 years"
        Plan:
        {
        "reasoning": "Need expenses and YoY changes. Steps: (1) Extract expenses, (2) Calculate YoY.",
        "steps": [
            {"tool": "SmartTableParser", "inputs": {"metric": "Operating Expenses"}, "purpose": "Get expenses for 3 years"},
            {"tool": "AdvancedCalculator", "inputs": {"operation": "yoy_change", "data": {"values": "$step1.data"}}, "purpose": "Calculate YoY changes"}
        ]
        }
        """

        user_message = f"{few_shot_examples}\n\nNow plan for this query:\nQuery: \"{query}\"\nPlan:"
        
        try:
            if self.provider == 'groq':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                self._track_usage(response) 
                plan_text = response.choices[0].message.content.strip()
            else:  # gemini
                chat = self.client.start_chat(history=[])
                response = chat.send_message(f"{system_prompt}\n\n{user_message}")
                self._track_usage(response) 
                plan_text = response.text.strip()
            
            # Parse JSON
            if '```json' in plan_text:
                plan_text = plan_text.split('```json').split('```')
            elif '```' in plan_text:
                plan_text = plan_text.split('``````')[0].strip()
            
            plan = json.loads(plan_text)
            return plan
        
        except Exception as e:
            return {'error': f"Plan generation failed: {str(e)}"}
    
    def _execute_plan_parallel(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute plan with parallel optimization
        Identifies independent steps and runs them concurrently
        """
        steps = plan.get('steps', [])
        if not steps:
            return []
        
        # Identify parallel groups
        parallel_groups = self.parallel_executor.identify_parallel_groups(steps)
        
        results = []
        self.tool_calls = []
        current_results_map = {}  # Map step index to result
        
        # Execute by groups
        unique_groups = sorted(set(parallel_groups))
        
        for group_id in unique_groups:
            # Get all steps in this group
            group_steps = [(i, steps[i]) for i, g in enumerate(parallel_groups) if g == group_id]
            
            if len(group_steps) == 1:
                # Single step - execute normally
                step_idx, step = group_steps[0]
                tool_name = step['tool']
                inputs = self._resolve_references(step['inputs'], current_results_map)
                
                tool_result = self._execute_tool(tool_name, inputs)
                current_results_map[step_idx] = tool_result
                results.append(tool_result)
                
                self.tool_calls.append(ToolCall(
                    tool_name=tool_name,
                    inputs=inputs,
                    outputs=tool_result,
                    latency_ms=tool_result.get('latency_ms', 0),
                    parallel_group=group_id
                ))
            
            else:
                # Multiple steps - execute in parallel
                tool_calls_batch = []
                for step_idx, step in group_steps:
                    tool_name = step['tool']
                    inputs = self._resolve_references(step['inputs'], current_results_map)
                    tool_instance = self.tools.get(tool_name)
                    tool_calls_batch.append((tool_name, inputs, tool_instance))
                
                # Execute in parallel
                parallel_start = time.time()
                parallel_results = self.parallel_executor.execute_parallel(tool_calls_batch)
                parallel_latency = (time.time() - parallel_start) * 1000
                
                # Record results
                for (step_idx, step), tool_result, (tool_name, inputs, _) in zip(group_steps, parallel_results, tool_calls_batch):
                    current_results_map[step_idx] = tool_result
                    results.append(tool_result)
                    
                    self.tool_calls.append(ToolCall(
                        tool_name=tool_name,
                        inputs=inputs,
                        outputs=tool_result,
                        latency_ms=tool_result.get('latency_ms', 0),
                        parallel_group=group_id
                    ))
                
                print(f"[Parallel Execution] Group {group_id}: {len(group_steps)} tools in {parallel_latency:.2f}ms")
        
        return results
    
    def _execute_plan_sequential(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute plan sequentially (original behavior)"""
        results = []
        self.tool_calls = []
        
        for i, step in enumerate(plan.get('steps', [])):
            tool_name = step['tool']
            inputs = step['inputs']
            
            # Resolve references
            inputs = self._resolve_references(inputs, results)
            
            # Execute tool
            tool_result = self._execute_tool(tool_name, inputs)
            
            # Record
            self.tool_calls.append(ToolCall(
                tool_name=tool_name,
                inputs=inputs,
                outputs=tool_result,
                latency_ms=tool_result.get('latency_ms', 0)
            ))
            
            results.append(tool_result)
        
        return results
    
    def _resolve_references(self, inputs: Dict[str, Any], 
                           results: Union[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Resolve references like $step1.data to actual values
        Supports both list (sequential) and dict (parallel) result storage
        """
        resolved = {}
        
        for key, value in inputs.items():
            if isinstance(value, str) and value.startswith('$step'):
                # Parse reference: $step1.data
                match = re.match(r'\$step(\d+)\.(\w+)', value)
                if match:
                    step_idx = int(match.group(1)) - 1
                    field = match.group(2)
                    
                    # Handle both list and dict result storage
                    if isinstance(results, dict):
                        if step_idx in results:
                            resolved[key] = results[step_idx].get(field, {})
                        else:
                            resolved[key] = {}
                    else:
                        if 0 <= step_idx < len(results):
                            resolved[key] = results[step_idx].get(field, {})
                        else:
                            resolved[key] = {}
                else:
                    resolved[key] = value
            elif isinstance(value, dict):
                resolved[key] = self._resolve_references(value, results)
            else:
                resolved[key] = value
        
        return resolved
    
    def _execute_tool(self, tool_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single tool"""
        try:
            if tool_name == 'SmartTableParser':
                return self.parser.parse(**inputs)
            elif tool_name == 'AdvancedCalculator':
                return self.calculator.compute(**inputs)
            elif tool_name == 'SmartTrendAnalyzer':
                return self.analyzer.analyze(**inputs)
            else:
                return {'error': f'Unknown tool: {tool_name}'}
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_answer(self, query: str, plan: Dict[str, Any], 
                        results: List[Dict[str, Any]]) -> str:
        """LLM generates final answer from execution results"""
        
        system_prompt = """You are a financial analyst. Generate a clear, professional answer using execution results.

        Include:
        1. Direct answer
        2. Data in table format if applicable
        3. Key insights/trends
        4. Citations with page numbers

        Format: [doc_name p.X]
        """

        # Compile results
        results_summary = []
        for i, (step, result) in enumerate(zip(plan['steps'], results)):
            summary = f"Step {i+1} ({step['tool']}): "
            if 'data' in result:
                summary += f"Extracted {len(result['data'])} values"
            elif 'result' in result:
                summary += f"Computed {len(result['result'])} values" if isinstance(result['result'], dict) else "Computed result"
            elif 'pattern' in result:
                summary += f"Pattern: {result['pattern']}"
            
            results_summary.append(summary)
            results_summary.append(f"  Output: {json.dumps(result, indent=2)}")
        
        user_message = f"""Query: {query}

        Plan: {plan['reasoning']}

        Execution Results:
        {chr(10).join(results_summary)}

        Generate professional answer with tables and citations."""

        try:
            if self.provider == 'groq':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.3,
                    max_tokens=2000
                )
                self._track_usage(response)
                return response.choices[0].message.content.strip()
            else:  # gemini
                chat = self.client.start_chat(history=[])
                response = chat.send_message(f"{system_prompt}\n\n{user_message}")
                self._track_usage(response)
                return response.text.strip()
        
        except Exception as e:
            return self._fallback_answer(query, results)
    
    def _fallback_answer(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Fallback answer generation"""
        lines = [f"Query: {query}\n"]
        
        for i, result in enumerate(results):
            if 'data' in result and result['data']:
                lines.append(f"\nData (Step {i+1}):")
                for period, value in result['data'].items():
                    lines.append(f"  {period}: {value}")
                
                if 'sources' in result:
                    lines.append("\nCitations:")
                    for src in result['sources'][:3]:
                        lines.append(f"  [{src['file']} p.{src['page']}]")
        
        return "\n".join(lines)

# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

# 1. Initialize Knowledge Base
kb = g2x.KBEnv()

# --- CRITICAL FIX: Pass T_ingest from KB to the DataFrame ---
# This ensures the Agent can "see" the ingestion time
if kb.tables_df is not None:
    kb.tables_df.t_ingest = kb.t_ingest 
# ------------------------------------------------------------

llm_tuple = g2x._make_llm_client()

print("[ReAct Agent CFO with Parallel Execution] Initializing...")
react_agent = ReActAgentCFO(kb.tables_df, llm_tuple)
print(f"[ReAct Agent CFO] Ready")
print(f"  - LLM: {llm_tuple[0]}")
print(f"  - Auto-detected: {react_agent.introspector.detect_quarters(5)} quarters")
print(f"  - Auto-detected: {react_agent.introspector.detect_years(3)} years")

# 2. Initialize the Scorecard (Logs Schema)
# We do this RIGHT BEFORE the loop starts
logs = pd.DataFrame(columns=['Query','T_ingest','T_retrieve','T_reason','T_generate','T_total','Tokens','Tools'])

# Define Queries
queries = [
    "Report the Gross Margin (or Net Interest Margin, if a bank) over the last 5 quarters, with values.",
    "Show Operating Expenses for the last 3 fiscal years, year-on-year comparison.",
    "Calculate the Operating Efficiency Ratio (Opex ÷ Operating Income) for the last 3 fiscal years, showing the working."
]

print("\n" + "=" * 60)
print("  REACT AGENT CFO BENCHMARK (WITH PARALLEL EXECUTION)")
print("=" * 60)

results_json = []
latencies_parallel = []
latencies_sequential = []

for i, query in enumerate(queries, 1):
    print(f"\n{'=' * 60}")
    print(f"Q{i}. {query}")
    print("=" * 60)
    
    # Run with parallel execution
    print("\n[Mode: PARALLEL]")
    result_parallel = react_agent.run(query, enable_parallel=True)
    latencies_parallel.append(result_parallel['latency_ms'])
    
    # --- 3. FILL THE LOGS (The part you were missing) ---
    # Safely get timings (defaults to 0 if missing to prevent KeyError)
    timings = result_parallel.get('timings', {})
    
    # Safely get token count
    token_data = result_parallel.get('tokens', {})
    total_tokens = token_data.get('total_tokens', 0) if isinstance(token_data, dict) else 0

    new_log = pd.DataFrame([{
        'Query': query,
        'T_ingest': timings.get('T_ingest', 0),
        'T_retrieve': timings.get('T_retrieve', 0),
        'T_reason': timings.get('T_reason', 0),
        'T_generate': timings.get('T_generate', 0),
        'T_total': timings.get('T_total', 0),
        'Tokens': result_parallel.get('tokens', {}).get('total_tokens', 0),
        'Tools': len(result_parallel.get('tools', []))
    }])
    
    logs = pd.concat([logs, new_log], ignore_index=True)
    # ----------------------------------------------------
    
    # Display reasoning
    if 'plan' in result_parallel and 'reasoning' in result_parallel['plan']:
        print(f"\n[LLM Reasoning] {result_parallel['plan']['reasoning']}\n")
    
    # Display tool execution
    if result_parallel['tool_calls']:
        print(f"[Tool Execution] {len(result_parallel['tool_calls'])} tools called:")
        for tc in result_parallel['tool_calls']:
            group_info = f" (Group {tc.parallel_group})" if tc.parallel_group is not None else ""
            print(f"  - {tc.tool_name}{group_info}: {tc.latency_ms:.2f} ms")
        print(f"[Parallel Groups] {result_parallel.get('parallel_groups', 0)} groups")
    
    # Display answer
    print(f"\n{result_parallel['answer']}")
    print(f"\n(Parallel Latency: {result_parallel['latency_ms']:.2f} ms)")
    
    # Run sequential for comparison
    print(f"\n[Mode: SEQUENTIAL - for comparison]")
    result_sequential = react_agent.run(query, enable_parallel=False)
    latencies_sequential.append(result_sequential['latency_ms'])
    print(f"(Sequential Latency: {result_sequential['latency_ms']:.2f} ms)")
    
    speedup = ((result_sequential['latency_ms'] - result_parallel['latency_ms']) / result_sequential['latency_ms']) * 100
    print(f"[Speedup] {speedup:.1f}% faster with parallel execution")
    
    # Record for JSON
    results_json.append({
        'query_id': f'Q{i}',
        'query': query,
        'answer': result_parallel['answer'],
        'plan_reasoning': result_parallel.get('plan', {}).get('reasoning', ''),
        'tool_calls': [
            {
                'tool': tc.tool_name,
                'latency_ms': tc.latency_ms,
                'parallel_group': tc.parallel_group
            }
            for tc in result_parallel['tool_calls']
        ],
        'latency_parallel_ms': result_parallel['latency_ms'],
        'latency_sequential_ms': result_sequential['latency_ms'],
        'speedup_percent': round(speedup, 1)
    })

# Summary
p50_parallel = np.percentile(latencies_parallel, 50)
p95_parallel = np.percentile(latencies_parallel, 95)
p50_sequential = np.percentile(latencies_sequential, 50)
p95_sequential = np.percentile(latencies_sequential, 95)

print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"PARALLEL MODE:")
print(f"  P50: {p50_parallel:.1f} ms")
print(f"  P95: {p95_parallel:.1f} ms")
print(f"\nSEQUENTIAL MODE:")
print(f"  P50: {p50_sequential:.1f} ms")
print(f"  P95: {p95_sequential:.1f} ms")
print(f"\nOVERALL SPEEDUP:")
print(f"  P50: {((p50_sequential - p50_parallel) / p50_sequential * 100):.1f}%")
print(f"  P95: {((p95_sequential - p95_parallel) / p95_sequential * 100):.1f}%")

# Save results
output_path = "./data_marker/bench_react_agent_cfo_parallel.json"
with open(output_path, 'w') as f:
    json.dump({
        "system": "ReAct Agent CFO (Parallel)",
        "approach": "LLM-driven planning with parallel tool execution",
        "latency_parallel": {
            "p50_ms": round(p50_parallel, 2),
            "p95_ms": round(p95_parallel, 2)
        },
        "latency_sequential": {
            "p50_ms": round(p50_sequential, 2),
            "p95_ms": round(p95_sequential, 2)
        },
        "speedup": {
            "p50_percent": round(((p50_sequential - p50_parallel) / p50_sequential * 100), 2),
            "p95_percent": round(((p95_sequential - p95_parallel) / p95_sequential * 100), 2)
        },
        "results": results_json
    }, f, indent=2)

print("\n" + "=" * 60)
print("  COMPLETE")
print("=" * 60)
print(f"Results: {output_path}")
print("=" * 60)

# Display the final populated logs
print("\nFinal Instrumentation Logs:")
print(logs)
