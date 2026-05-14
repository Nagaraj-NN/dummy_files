#!/usr/bin/env python3
"""
Snowflake SQL to dbt Model Converter
=====================================
This version includes $TYPE=STG support AND optional $DESC_TABLE for INSERT/UPDATE/MERGE types.

Parameters (non-STG types - all mandatory except $DESC_TABLE which is optional):
  $TYPE, $TARGET_TABLE, $TARGET_TABLE_COLUMNS, $PRIMARY_KEY, $SPLIT_CTE,
  $UPDATE_COLUMNS, $SOURCE_CODE (or $INSERT_CODE), $AUTOSYS_JOB_NAME,
  $PRE_HOOKn, $POST_HOOKn, $DESC_TABLE (optional)

Parameters (STG type - only these are mandatory):
  $TYPE: STG;
  $SOURCE_CODE:  -- INSERT INTO target (cols) WITH ctes... SELECT ... FROM cte;
  $DESC_TABLE:   -- Snowflake DESC TABLE output (tab/space separated)

STG type behavior:
  - Pre-hook always adds: "TRUNCATE TABLE IF EXISTS {{this}}" along with log_model_start
  - Final SELECT builds columns in DESC_TABLE order
  - Columns present in INSERT INTO: CAST(source_expr AS type) AS target_col
  - With -trim flag: CAST(TRIM(source_expr) AS type) AS target_col
  - VARCHAR columns always wrapped with LEFT(..., N) where N is the VARCHAR length
  - Columns missing from INSERT INTO: default AS col (or CAST(NULL AS type) AS col)
  - Trailing 'AS alias' in source expressions is stripped to avoid double-wrapping

INSERT/UPDATE/MERGE with $DESC_TABLE (optional):
  - If DESC_TABLE is provided:
    * Column order = DESC_TABLE order (canonical)
    * Each column wrapped with CAST([LEFT(][TRIM(]expr[)][, N)] AS type)
      (LEFT only for VARCHAR, TRIM only with -trim)
    * MERGE: wraps the whole COALESCE: CAST(COALESCE(SRC.x, TGT.y) AS type) AS y
    * PK: CAST(SRC.pk AS type) AS pk (no COALESCE)
    * DECODE: both tgt and src sides get CAST/LEFT/TRIM for consistent comparison
    * Columns in DESC_TABLE but not in $TARGET_TABLE_COLUMNS: CAST(NULL AS type) or default
  - If DESC_TABLE is NOT provided: existing behavior, no casting

Usage:
  python sql_to_dbt_converter.py input.sql sources.yml [-o output.sql]
  python sql_to_dbt_converter.py input.sql sources.yml -o output.sql -cc
  python sql_to_dbt_converter.py input.sql sources.yml -o output.sql -trim
  python sql_to_dbt_converter.py --batch input/ sources.yml -d output/ [--report]
"""

import re, sys, os, argparse, yaml
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SourceMapping:
    source_name: str
    table_name: str
    database: str
    schema: str

@dataclass
class ParsedParams:
    model_type: str = ""
    target_table: str = ""
    target_table_columns: List[str] = field(default_factory=list)
    primary_key: str = ""
    split_cte: str = ""
    update_columns: List[Tuple[str, str]] = field(default_factory=list)
    pre_hooks: Dict[str, str] = field(default_factory=dict)
    post_hooks: Dict[str, str] = field(default_factory=dict)
    source_code: str = ""
    autosys_job_name: str = ""
    target_db: str = ""
    target_schema: str = ""
    target_table_name: str = ""
    # STG type additions
    desc_table_raw: str = ""
    desc_table: List[Tuple[str, str, str]] = field(default_factory=list)  # (name, type, default)
    model_name: str = ""  # optional, for dbt run --select in test cases

    @property
    def core_type(self) -> str:
        parts = [p.strip() for p in self.model_type.upper().split(',')]
        core = [p for p in parts if p not in ('PRE_HOOK', 'POST_HOOK', '')]
        has_i, has_u, has_t = 'INSERT' in core, 'UPDATE' in core, 'TRUNCATE' in core
        has_stg = 'STG' in core
        has_merge = 'MERGE' in core
        if has_merge: return 'MERGE'
        if has_stg: return 'STG'
        if has_t and has_i: return 'TRUNCATE_INSERT'
        elif has_i and has_u: return 'INCREMENTAL'
        elif has_u: return 'UPDATE'
        elif has_i: return 'INSERT'
        return self.model_type.upper()

    @property
    def is_stg(self) -> bool:
        return self.core_type == 'STG'

    @property
    def is_merge_type(self) -> bool:
        return self.core_type == 'MERGE'

    @property
    def needs_merge(self) -> bool:
        return len(self.update_columns) > 0


# =============================================================================
# SOURCES REGISTRY
# =============================================================================

class SourcesRegistry:
    def __init__(self, yml_path: str):
        self.mappings: Dict[str, List[SourceMapping]] = {}
        self.schema_map: Dict[str, SourceMapping] = {}
        with open(yml_path) as f:
            data = yaml.safe_load(f)
        for src in data.get('sources', []):
            sn, db, sc = src['name'], src.get('database', ''), src.get('schema', '')
            schema_key = f"{db}.{sc}".lower()
            for t in src.get('tables', []):
                fqn = f"{db}.{sc}.{t['name']}".lower()
                m = SourceMapping(sn, t['name'], db, sc)
                self.mappings.setdefault(fqn, []).append(m)
            if schema_key not in self.schema_map:
                self.schema_map[schema_key] = SourceMapping(sn, '', db, sc)

    def resolve(self, fq_table: str) -> Tuple[Optional[str], bool]:
        key = fq_table.strip().lower()
        parts = key.split('.')
        if len(parts) != 3:
            return None, False
        db, schema, table = parts
        if key in self.mappings:
            m = self.mappings[key][0]
            return f"{{{{source('{m.source_name}','{m.table_name}')}}}}", True
        schema_key = f"{db}.{schema}".lower()
        if schema_key in self.schema_map:
            m = self.schema_map[schema_key]
            return f"{{{{source('{m.source_name}','{table.upper()}')}}}}", False
        return None, False

    def resolve_metadata(self, target_db: str, target_schema: str) -> Tuple[Optional[str], str, str]:
        schema_prefix = target_schema.upper()
        param_table = f"{schema_prefix}_JOB_PARAMETERS"
        control_table = f"{schema_prefix}_JOB_CONTROL"
        param_lower = param_table.lower()
        for fqn, ml in self.mappings.items():
            if param_lower in fqn:
                return ml[0].source_name, param_table, control_table
        for fqn, ml in self.mappings.items():
            for m in ml:
                if 'metadata' in m.schema.lower() and param_lower == m.table_name.lower():
                    return m.source_name, param_table, control_table
        return None, param_table, control_table


# =============================================================================
# PARAMETER PARSER
# =============================================================================

class ParameterParser:
    @staticmethod
    def parse(content: str) -> ParsedParams:
        p = ParsedParams()
        blocks = ParameterParser._extract_blocks(content)
        for param_name, raw_value in blocks.items():
            name = param_name.upper()
            if name == 'TYPE':
                p.model_type = ParameterParser._strip_semi(raw_value)
            elif name == 'TARGET_TABLE':
                p.target_table = ParameterParser._strip_semi(raw_value)
                if p.target_table and len(p.target_table.split('.')) == 3:
                    p.target_db, p.target_schema, p.target_table_name = p.target_table.split('.')
            elif name == 'TARGET_TABLE_COLUMNS':
                raw = ParameterParser._strip_semi(raw_value)
                p.target_table_columns = [c.strip().upper() for c in raw.split(',') if c.strip()]
            elif name == 'PRIMARY_KEY':
                p.primary_key = ParameterParser._strip_semi(raw_value).strip().upper()
            elif name == 'SPLIT_CTE':
                p.split_cte = ParameterParser._strip_semi(raw_value).strip()
                if p.split_cte.upper() == 'NONE':
                    p.split_cte = ''
            elif name == 'AUTOSYS_JOB_NAME':
                p.autosys_job_name = ParameterParser._strip_semi(raw_value).strip()
            elif name == 'UPDATE_COLUMNS':
                raw = raw_value.strip().rstrip(';').strip()
                if raw and raw.upper() != 'NONE':
                    # Split by newlines first, then by comma within each line
                    # This handles both multi-line and single-line (comma+tab separated) formats
                    for line in raw.split('\n'):
                        line = line.strip()
                        if not line or line.startswith('--'):
                            continue
                        # Split by comma to handle: TGT.A=SRC.A, TGT.B=SRC.B, ...
                        for part in re.split(r',', line):
                            part = part.strip().rstrip(',').strip()
                            if not part:
                                continue
                            m = re.match(r'tgt\.(\w+)\s*=\s*src\.(\w+)', part, re.I)
                            if m:
                                p.update_columns.append((m.group(1).upper(), m.group(2).upper()))
            elif name in ('SOURCE_CODE', 'INSERT_CODE'):
                p.source_code = raw_value.strip()
            elif name == 'DESC_TABLE':
                p.desc_table_raw = raw_value.strip().rstrip(';').strip()
                p.desc_table = ParameterParser._parse_desc_table(p.desc_table_raw)
            elif name.startswith('PRE_HOOK'):
                sql = ParameterParser._strip_semi(raw_value).strip()
                if sql and sql.upper() != 'NONE' and len(sql) > 1:
                    p.pre_hooks[name] = sql
            elif name.startswith('POST_HOOK'):
                sql = ParameterParser._strip_semi(raw_value).strip()
                if sql and sql.upper() != 'NONE' and len(sql) > 1:
                    p.post_hooks[name] = sql
            elif name == 'MODEL_NAME':
                p.model_name = ParameterParser._strip_semi(raw_value).strip()
        return p

    @staticmethod
    def _extract_blocks(content: str) -> Dict[str, str]:
        param_positions = []
        for m in re.finditer(r'^\s*\$(\w+)\s*:\s*', content, re.MULTILINE):
            # Issue #6: Skip if preceded by -- on the same line
            line_start = content.rfind('\n', 0, m.start()) + 1
            prefix = content[line_start:m.start()].strip()
            if prefix.startswith('--'):
                continue
            param_positions.append((m.group(1), m.start(), m.end()))
        blocks = {}
        for i, (name, block_start, value_start) in enumerate(param_positions):
            value_end = param_positions[i + 1][1] if i + 1 < len(param_positions) else len(content)
            blocks[name] = content[value_start:value_end].strip()
        return blocks

    @staticmethod
    def _strip_semi(raw: str) -> str:
        in_single, in_double = False, False
        for i, ch in enumerate(raw):
            if ch == "'" and not in_double: in_single = not in_single
            elif ch == '"' and not in_single: in_double = not in_double
            elif ch == ';' and not in_single and not in_double:
                return raw[:i].strip()
        return raw.strip()

    @staticmethod
    def _parse_desc_table(raw: str) -> List[Tuple[str, str, str]]:
        """Parse $DESC_TABLE parameter value (Snowflake DESC TABLE output).
        Expected format (tab-separated, or whitespace-separated):
            name<SEP>type<SEP>kind<SEP>null?<SEP>default
            c1<SEP>NUMBER(20,0)<SEP>COLUMN<SEP>Y<SEP>
            c2<SEP>VARCHAR(20)<SEP>COLUMN<SEP>Y<SEP>some_default

        Returns list of (name_upper, type_upper, default) in the DESC TABLE order.
        Skips header row and non-COLUMN rows.

        Handles:
         - Tab separators
         - Multi-space separators
         - Single-space separators (when tokens don't contain spaces)
         - Defaults that may contain spaces (e.g., CAST(CURRENT_TIMESTAMP() AS TIMESTAMP_NTZ(6)))
        """
        result = []
        if not raw:
            return result
        lines = raw.split('\n')
        for line in lines:
            line = line.rstrip()
            if not line.strip():
                continue

            # Try tab-separated first
            if '\t' in line:
                fields = line.split('\t')
            else:
                # For space-separated: name, type, kind, null?, default
                # type may contain parens but no spaces in identifier names or type decls
                # Use a smart split: first 4 whitespace-delimited tokens, then rest = default
                tokens = line.split(None, 4)  # max 5 parts
                fields = tokens

            if len(fields) < 2:
                continue

            name = fields[0].strip()
            col_type = fields[1].strip()
            kind = fields[2].strip().upper() if len(fields) > 2 else 'COLUMN'
            # null = fields[3] if present - not used
            default = fields[4].strip() if len(fields) > 4 else ''

            # Skip header row
            if name.lower() == 'name' and col_type.lower() == 'type':
                continue
            # Skip non-COLUMN rows
            if kind and kind != 'COLUMN':
                continue
            if not name:
                continue
            result.append((name.upper(), col_type.upper(), default))
        return result


# =============================================================================
# SQL TRANSFORMER
# =============================================================================

class SQLTransformer:
    def __init__(self, params: ParsedParams, sources: SourcesRegistry, use_trim: bool = False, skip_numeric: bool = False):
        self.params = params
        self.sources = sources
        self.use_trim = use_trim
        self.skip_numeric = skip_numeric
        self.warnings: List[str] = []
        self.missing_columns: List[str] = []
        self.unresolved_tables: List[str] = []
        self.schema_only_tables: List[str] = []
        self.join_columns: Set[str] = set()
        self.found_params: List[Tuple[str, str]] = []
        self.insert_into_columns: List[str] = []       # columns from INSERT INTO (target side)
        self.column_mappings: List[Tuple[str, str]] = []  # (target_col, source_expr) where names differ
        self.func_columns: Dict[str, str] = {}           # target_col -> function expression (e.g. substr(...))
        self.defaulted_columns: List[str] = []           # in TARGET_TABLE_COLUMNS but not in INSERT INTO
        # STG-type tracking
        self.stg_defaulted: List[Tuple[str, str, str]] = []  # (col_name, type, default_or_NULL) for STG missing

    def transform(self) -> str:
        is_stg = self.params.is_stg
        is_truncate = self.params.core_type == 'TRUNCATE_INSERT'
        is_merge = self.params.needs_merge

        # Scan source code for hardcoded parameters
        self._scan_for_params()

        # For STG type, we analyze insert columns differently (no merge, no update cols)
        if is_stg:
            self._analyze_insert_columns()
        else:
            # Parse INSERT INTO columns and compare
            self._analyze_insert_columns()

        if is_stg:
            result = self._build_stg_model()
        else:
            result = self._build_model(is_merge=is_merge, is_truncate=is_truncate)
        header = self._warning_header()
        return (header + '\n' + result) if header else result

    def _analyze_insert_columns(self):
        """Parse INSERT INTO column list and pair with final SELECT expressions.
        Detects: renamed columns, function-wrapped columns, defaulted columns."""
        sql = self.params.source_code
        if not sql:
            return

        # Extract INSERT INTO ... (col1, col2, ...) column list
        m = re.search(r'INSERT\s+INTO\s+\S+\s*\(\s*(.*?)\s*\)', sql, re.I | re.DOTALL)
        if not m:
            return
        raw_cols = m.group(1)
        self.insert_into_columns = [c.strip().upper() for c in raw_cols.split(',') if c.strip()]

        # Get the final SELECT raw expressions (position-paired with INSERT INTO)
        ctes, final_select = self._extract_ctes(sql)
        if not final_select:
            return

        fm = re.search(r'SELECT\s+(.*?)\s*FROM\s+', final_select, re.I | re.DOTALL)
        if not fm:
            return
        select_exprs = self._split_cols(fm.group(1).strip())

        # Warn if counts differ - position-based pairing may be unreliable
        if len(self.insert_into_columns) != len(select_exprs):
            self.warnings.append(
                f"INSERT INTO column count ({len(self.insert_into_columns)}) differs from "
                f"final SELECT expression count ({len(select_exprs)}). "
                f"Position-based column mapping may be inaccurate. "
                f"$UPDATE_COLUMNS mappings will be used as override where available."
            )

        # Build mappings from INSERT INTO + final SELECT (position-based)
        # This is safe because INSERT INTO (col1,col2) SELECT (expr1,expr2) are paired by position
        upd_map = {t: s for t, s in self.params.update_columns}

        for ins_col, sel_expr in zip(self.insert_into_columns, select_exprs):
            sel_expr = sel_expr.strip()
            if not sel_expr:
                continue

            # Remove inline comments first (may have multi-line comment text from _split_cols)
            sel_clean = re.sub(r'\s*--.*$', '', sel_expr, flags=re.MULTILINE).strip()
            if not sel_clean:
                continue

            # Skip if the entire cleaned expression is a comment or empty
            if sel_clean.lstrip().startswith('--'):
                continue

            # Check if it's a function call (contains parentheses that aren't just casting)
            has_func = bool(re.search(r'\w+\s*\(', sel_clean)) and \
                       not re.match(r"^'[^']*'\s+AS\s+\w+$", sel_clean, re.I) and \
                       not re.match(r'^-?\d+\s+AS\s+\w+$', sel_clean, re.I) and \
                       not re.match(r'^NULL\s+AS\s+\w+$', sel_clean, re.I)

            if has_func:
                # Function-wrapped column: substr(X1INTNOTES,1,300) -> INTERNAL_NOTES
                self.func_columns[ins_col] = sel_clean
                # Also add to mappings for header display
                self.column_mappings.append((ins_col, sel_clean))
                continue

            # Extract the source column name from the expression
            # Handle: "col AS alias", "table.col alias", "col", "-1 as COL"
            src_name = None
            as_match = re.search(r'\bAS\s+(\w+)\s*$', sel_clean, re.I)
            if as_match:
                # Expression has AS alias - the source is everything before AS
                src_part = sel_clean[:as_match.start()].strip()
                src_name = src_part.split('.')[-1].strip().upper()
            else:
                pts = sel_clean.split()
                if len(pts) >= 2 and re.match(r'^\w+$', pts[-1]):
                    src_name = pts[-1].upper()
                else:
                    src_name = sel_clean.split('.')[-1].strip().upper()

            # Skip if already handled by $UPDATE_COLUMNS
            if ins_col in upd_map:
                continue

            # Record mapping if names differ
            if src_name and src_name != ins_col:
                self.column_mappings.append((ins_col, src_name))

        # Find defaulted columns: in TARGET_TABLE_COLUMNS but NOT in INSERT INTO
        if self.insert_into_columns:
            insert_set = set(self.insert_into_columns)
            for col in self.params.target_table_columns:
                if col not in insert_set:
                    self.defaulted_columns.append(col)

    def _scan_for_params(self):
        """Scan source code for hardcoded values that need JOB_PARAM/JOB_CONTROL entries."""
        sql = self.params.source_code
        if not sql:
            return
        for line in sql.split('\n'):
            stripped = line.strip()
            if stripped.startswith('--'):
                continue
            # ROWSTAMP - detect TO_NUMBER, TRY_TO_NUMBER, or direct comparisons
            if re.search(r'ROWSTAMP', line, re.I) and \
               (re.search(r'(?:TO_NUMBER|TRY_TO_NUMBER)\s*\(\s*\d+\s*\)', line, re.I) or
                re.search(r'(?:TO_NUMBER|TRY_TO_NUMBER)\s*\(\s*\w*\.?ROWSTAMP', line, re.I) or
                re.search(r'ROWSTAMP\s*<=\s*\d+', line, re.I)):
                self.found_params.append(('ROWSTAMP', stripped))
            # LOAD_ID and variants (OUT_LOAD_ID, V_LOAD_ID, SRC_LOAD_ID, etc.)
            if re.search(r'\b\d+\s+AS\s+\w*LOAD_ID\b', line, re.I):
                self.found_params.append(('LOAD_ID', stripped))
            # LAST_UPDATE_LOAD_ID and variants
            if re.search(r'-?\d+\s+AS\s+\w*LAST_UPDATE_LOAD_ID\b', line, re.I):
                self.found_params.append(('LAST_UPDATE_LOAD_ID', stripped))
            # Also catch assignments like: variable := LOAD_ID value patterns
            if re.search(r'\bLOAD_ID\b', line, re.I) and re.search(r'\b\d+\b', line) and \
               not re.search(r'ROWSTAMP|LAST_UPDATE', line, re.I) and \
               'AS LOAD_ID' not in line.upper() and 'PARAM_NAME' not in line.upper():
                # Only if it looks like a hardcoded assignment, not a column reference
                if re.search(r"'?\d+'?\s+AS\s+\w*LOAD_ID", line, re.I):
                    if not any(p[1] == stripped for p in self.found_params):
                        self.found_params.append(('LOAD_ID', stripped))
            # Date ranges with between
            if re.search(r"between\s+'[\d\-]+\s+[\d:]+'\s+and\s+'[\d\-]+\s+[\d:]+'", line, re.I):
                self.found_params.append(('DATE_RANGE', stripped))
            # TO_DATE with hardcoded dates
            elif re.search(r"(?:TO_DATE|TRY_TO_DATE)\s*\(\s*'\d{4}-\d{2}-\d{2}", line, re.I):
                self.found_params.append(('DATE_RANGE', stripped))

    # =========================================================================
    # MAIN BUILDER
    # =========================================================================

    def _build_model(self, is_merge=False, is_truncate=False) -> str:
        parts = []
        parts.append(self._gen_config(is_truncate))
        parts.append('')
        parts.append('WITH')
        parts.append(self._gen_job_ctes())

        source_sql = self.params.source_code
        if not source_sql:
            self.warnings.append("No $SOURCE_CODE found")
            return '\n'.join(parts)

        # Extract JOIN columns before table replacement
        if is_merge:
            self.join_columns = self._extract_join_columns(source_sql)

        # Replace tables, highlight hardcoded values
        transformed = self._replace_tables(source_sql)
        transformed = self._highlight_hardcoded(transformed)

        # Extract CTEs
        ctes, final_select = self._extract_ctes(transformed)

        # Truncate CTE list at split CTE (remove CTEs that come after it)
        split_name = self.params.split_cte.upper() if self.params.split_cte else ''
        if split_name:
            truncated_ctes = []
            for name, body in ctes:
                truncated_ctes.append((name, body))
                if name.upper() == split_name:
                    break
            ctes = truncated_ctes

        # Output CTEs
        num_ctes = len(ctes)
        for i, (name, body) in enumerate(ctes):
            sep = '' if (i == num_ctes - 1) else ','
            parts.append(f"{name} AS (\n{self._indent(body, '  ')}\n){sep}")

        # Final SELECT - always build from split CTE / TARGET_TABLE_COLUMNS
        # Discard the original INSERT INTO's final SELECT
        if is_merge:
            parts.append(self._gen_merge_select())
        else:
            # Insert-only: build simple SELECT from split CTE using TARGET_TABLE_COLUMNS
            parts.append(self._gen_insert_select())

        return '\n'.join(parts)

    # =========================================================================
    # CONFIG BLOCK
    # =========================================================================

    def _gen_config(self, is_truncate=False, is_stg=False) -> str:
        pk = self.params.primary_key
        autosys = self.params.autosys_job_name or '/* TODO: set $AUTOSYS_JOB_NAME */'
        is_insert_only = not self.params.needs_merge and not is_stg and not is_truncate

        # Pre-hooks: log_model_start first, then truncate if needed, then custom
        pre = []
        pre.append(f"log_model_start(this, '{autosys}')")
        if is_stg:
            # STG: always truncate-and-recreate
            pre.append('"TRUNCATE TABLE IF EXISTS {{this}}"')
        elif is_truncate:
            pre.append('truncate_model(this)')
        for k in sorted(self.params.pre_hooks):
            pre.append(f'"{self._transform_hook(self.params.pre_hooks[k])}"')

        # Post-hooks: custom first, then log_model_end last
        post = []
        for k in sorted(self.params.post_hooks):
            post.append(f'"{self._transform_hook(self.params.post_hooks[k])}"')
        post.append(f"log_model_end(this, '{autosys}')")

        lines = ['{{ config(']
        if is_stg:
            lines.append(f"    materialized='incremental',")
            lines.append(f"    incremental_strategy='append',")
            lines.append(f"    full_refresh=false,")
        elif is_insert_only:
            # INSERT only (no UPDATE_COLUMNS) - use append strategy
            lines.append(f"    materialized='incremental',")
            lines.append(f"    incremental_strategy='append',")
            lines.append(f"    full_refresh=false,")
        else:
            lines.append(f"    materialized='incremental',")
            lines.append(f"    unique_key='{pk}',")
            lines.append(f"    incremental_strategy='merge',")
            lines.append(f"    full_refresh=false,")
        lines.append('    pre_hook=[')
        lines.append('        ' + ',\n        '.join(pre))
        lines.append('    ],')
        lines.append('    post_hook=[')
        lines.append('        ' + ',\n        '.join(post))
        lines.append('    ]')
        lines.append(') }}')
        return '\n'.join(lines)

    # =========================================================================
    # JOB CTEs (with AUTOSYS_JOB_NAME)
    # =========================================================================

    def _gen_job_ctes(self) -> str:
        ms, param_tbl, ctrl_tbl = self.sources.resolve_metadata(
            self.params.target_db, self.params.target_schema)
        jn = self.params.target_table_name
        autosys = self.params.autosys_job_name

        if ms:
            param_ref = f"{{{{source('{ms}','{param_tbl}')}}}}"
            ctrl_ref = f"{{{{source('{ms}','{ctrl_tbl}')}}}}"
        else:
            self.warnings.append(f"Metadata source for {param_tbl}/{ctrl_tbl} not found in sources.yml")
            param_ref = f"/* TODO: resolve source */ METADATA.{param_tbl}"
            ctrl_ref = f"/* TODO: resolve source */ METADATA.{ctrl_tbl}"

        # Build WHERE clause - always include AUTOSYS_JOB_NAME
        autosys_val = autosys if autosys else '/* TODO: set $AUTOSYS_JOB_NAME */'
        param_where = f"WHERE JOB_NAME= '{jn}' AND AUTOSYS_JOB_NAME = '{autosys_val}' AND ACTIVE_IND = 'Y'"
        ctrl_where = f"WHERE JOB_NAME= '{jn}' AND AUTOSYS_JOB_NAME = '{autosys_val}'"

        return (
            f"JOB_PARAM AS (\n"
            f"    SELECT PARAM_ID, PARAM_NAME, PARAM_VALUE FROM {param_ref}\n"
            f"    {param_where}\n"
            f"),\n\n"
            f"JOB_CONTROL AS (\n"
            f"    SELECT JOB_ID, START_DATE, END_DATE FROM {ctrl_ref}\n"
            f"    {ctrl_where}\n"
            f"),"
        )

    # =========================================================================
    # TABLE REPLACEMENT (scan SQL, skip comments)
    # =========================================================================

    def _replace_tables(self, sql: str) -> str:
        tgt_fq = self.params.target_table.lower() if self.params.target_table else ''

        # Find all db.schema.table patterns, skipping commented lines and UDFs
        fq_pattern = re.compile(r'\b(\w+\.\w+\.\w+)\b')
        found_tables = set()
        for line in sql.split('\n'):
            stripped = line.lstrip()
            if stripped.startswith('--'):
                continue  # Issue #5: skip commented lines
            for m in fq_pattern.finditer(line):
                candidate = m.group(1)
                parts = candidate.split('.')
                if any(p.isdigit() for p in parts):
                    continue
                # Skip UDFs (followed by parenthesis)
                rest = line[m.end():].lstrip()
                if rest and rest[0] == '(':
                    continue
                found_tables.add(candidate)

        # Build replacements
        reps = []
        already = set()
        if self.params.target_table:
            reps.append((self.params.target_table, '{{this}}'))
            already.add(tgt_fq)

        for table_fq in found_tables:
            if table_fq.lower() in already:
                continue
            already.add(table_fq.lower())
            parts = table_fq.split('.')
            tname = parts[-1].upper()

            if any(tname.startswith(px) for px in ('STG_', 'DM_', 'FACT_', 'DIM_')):
                reps.append((table_fq, f"{{{{ref('{tname}')}}}}"))
                continue
            resolved, is_exact = self.sources.resolve(table_fq)
            if resolved:
                reps.append((table_fq, resolved))
                if not is_exact:
                    self.schema_only_tables.append(f"{table_fq} -> {resolved}")
            else:
                self.unresolved_tables.append(table_fq)
                # Keep original name, comment will be added at end of line during replacement
                reps.append((table_fq, table_fq))

        reps.sort(key=lambda x: len(x[0]), reverse=True)

        # Apply line-by-line, comment at END of line, skip commented lines
        for old, new in reps:
            # Use word boundary to prevent partial matches (e.g., DM_WORK_ORDER inside DM_WORK_ORDER_TSK)
            pattern = re.compile(r'\b' + re.escape(old) + r'\b', re.I)
            lines = sql.split('\n')
            new_lines = []
            for line in lines:
                stripped = line.lstrip()
                if stripped.startswith('--'):
                    new_lines.append(line)
                    continue
                if pattern.search(line):
                    replaced = pattern.sub(new, line)
                    replaced = replaced.rstrip() + f"  --{old}"
                    new_lines.append(replaced)
                else:
                    new_lines.append(line)
            sql = '\n'.join(new_lines)
        return sql

    # =========================================================================
    # HIGHLIGHT HARDCODED VALUES
    # =========================================================================

    def _highlight_hardcoded(self, sql: str) -> str:
        lines = sql.split('\n')
        new_lines = []
        for line in lines:
            if line.lstrip().startswith('--'):
                new_lines.append(line)
                continue
            if '-- TODO' in line:
                new_lines.append(line)
                continue
            commented = line
            if re.search(r'TO_NUMBER\s*\(\s*\d+\s*\)', line, re.I) and re.search(r'ROWSTAMP', line, re.I):
                commented = commented.rstrip() + '  -- TODO: parameterize ROWSTAMP via JOB_PARAM'
            elif re.search(r"between\s+'[\d\-]+\s+[\d:]+'\s+and\s+'[\d\-]+\s+[\d:]+'", line, re.I):
                commented = commented.rstrip() + '  -- TODO: parameterize dates via JOB_CONTROL'
            elif re.search(r"(?:TO_DATE|TRY_TO_DATE)\s*\(\s*'\d{4}-\d{2}-\d{2}", line, re.I):
                commented = commented.rstrip() + '  -- TODO: parameterize dates via JOB_CONTROL'
            elif re.search(r'\b\d+\s+AS\s+LOAD_ID\b', line, re.I):
                commented = commented.rstrip() + '  -- TODO: parameterize LOAD_ID via JOB_PARAM'
            elif re.search(r'-?\d+\s+AS\s+LAST_UPDATE_LOAD_ID\b', line, re.I):
                commented = commented.rstrip() + '  -- TODO: parameterize LAST_UPDATE_LOAD_ID via JOB_PARAM'
            new_lines.append(commented)
        return '\n'.join(new_lines)

    # =========================================================================
    # EXTRACT JOIN COLUMNS (only when target is RIGHT side of LEFT/INNER JOIN)
    # =========================================================================

    def _extract_join_columns(self, sql: str) -> Set[str]:
        join_cols = set()
        target_fq = self.params.target_table.lower()
        ctes, _ = self._extract_ctes(sql)
        for cte_name, cte_body in ctes:
            if not re.search(re.escape(target_fq), cte_body, re.I):
                continue
            # Issue #13: Only when target is on RIGHT side of JOIN
            # Pattern: LEFT [OUTER] JOIN target_table alias  OR  INNER JOIN target_table alias
            join_match = re.search(
                r'(?:LEFT\s+(?:OUTER\s+)?JOIN|INNER\s+JOIN)\s+' + re.escape(target_fq) + r'\s+(\w+)',
                cte_body, re.I
            )
            if not join_match:
                continue
            alias = join_match.group(1)
            on_section = cte_body[join_match.end():]
            on_match = re.search(r'\bON\b\s+(.*?)(?:\bWHERE\b|\bQUALIFY\b|\bGROUP\b|\bORDER\b|$)',
                                 on_section, re.I | re.DOTALL)
            if on_match:
                for col_match in re.finditer(rf'\b{re.escape(alias)}\.(\w+)', on_match.group(1), re.I):
                    join_cols.add(col_match.group(1).upper())
        return join_cols

    # =========================================================================
    # CTE EXTRACTION (handles )--comment patterns)
    # =========================================================================

    def _extract_ctes(self, sql: str) -> Tuple[List[Tuple[str, str]], Optional[str]]:
        ctes = []
        # Find the first 'WITH name AS (' pattern - this is robust against any wrapper
        # (INSERT INTO, SELECT cols FROM (, UPDATE ... FROM (, including ones with {{ref(...)}})
        with_match = re.search(r'\bWITH\s+\w+\s+AS\s*\(', sql, re.I)
        if not with_match:
            return ctes, sql.strip() or None

        # Position right after 'WITH '
        wm = re.match(r'WITH\s+', sql[with_match.start():], re.I)
        rem = sql[with_match.start() + wm.end():]
        final = None

        while rem.strip():
            # Match CTE name: 'NAME AS (' OR 'NAME (' (tolerant of missing AS - common in legacy SQL)
            # Try standard 'NAME AS (' first
            nm = re.match(r'\s*(\w+)\s+AS\s*\(?\s*', rem, re.I)
            missing_as = False
            if not nm:
                # Fallback: 'NAME (' without AS
                nm = re.match(r'\s*(\w+)\s*\(\s*', rem)
                if nm:
                    missing_as = True
                else:
                    break

            cte_name = nm.group(1)

            if missing_as:
                # Find the opening paren directly (no AS keyword)
                paren_match = re.search(r'\(\s*', rem[nm.start():])
            else:
                # Find the opening paren (might be on same line or next line)
                after_as = rem[nm.start():]
                paren_match = re.search(r'\bAS\s*\(\s*', after_as, re.I)
                if not paren_match:
                    # Try: AS \n(
                    paren_match = re.search(r'\bAS\s*\n\s*\(\s*', after_as, re.I)
            if not paren_match:
                break

            pos = nm.start() + paren_match.end()

            # Find matching closing paren (quote-aware AND comment-aware)
            depth = 1
            i = pos
            in_single, in_double = False, False
            while i < len(rem) and depth > 0:
                ch = rem[i]
                # Skip -- single-line comments (don't count parens inside them)
                if not in_single and not in_double and i + 1 < len(rem) and rem[i:i+2] == '--':
                    # Skip to end of line
                    eol = rem.find('\n', i)
                    if eol == -1:
                        i = len(rem)
                    else:
                        i = eol + 1
                    continue
                if ch == "'" and not in_double:
                    in_single = not in_single
                elif ch == '"' and not in_single:
                    in_double = not in_double
                elif not in_single and not in_double:
                    if ch == '(':
                        depth += 1
                    elif ch == ')':
                        depth -= 1
                i += 1

            cte_body = rem[pos:i - 1].strip()
            ctes.append((cte_name, cte_body))

            # Move past closing paren
            rem = rem[i:]

            # Helper: skip comments and whitespace
            def _skip_comments(s):
                s = s.lstrip()
                while s.startswith('--'):
                    eol = s.find('\n')
                    if eol == -1:
                        return ''
                    s = s[eol + 1:].lstrip()
                return s

            # Skip comments/whitespace/semicolons after )
            rem = _skip_comments(rem)
            rem = rem.lstrip(';')
            rem = _skip_comments(rem)

            # Check what comes next
            if rem.startswith(','):
                rem = rem[1:]
                rem = _skip_comments(rem)  # skip comments after comma too
            elif rem.upper().lstrip().startswith('SELECT'):
                final = rem.strip()
                break
            elif re.match(r'\s*\w+\s*(?:AS\s*)?\(', rem, re.I):
                # Tolerant: looks like another CTE name even though comma is missing
                # (common in legacy SQL - just continue)
                pass
            elif not rem.strip():
                break

        return ctes, final

    # =========================================================================
    # MERGE SELECT
    # =========================================================================

    def _gen_merge_select(self) -> str:
        pk = self.params.primary_key
        upd_cols_map = {t: s for t, s in self.params.update_columns}  # tgt -> src
        upd_cols = set(upd_cols_map.keys())
        tgt_cols = self.params.target_table_columns
        has_desc = bool(self.params.desc_table)

        # Build column name mapping: target_col -> source_col or source_expression
        # Priority: 1) func_columns (function-wrapped), 2) $UPDATE_COLUMNS, 3) INSERT rename mapping, 4) same name
        col_name_map = {}  # target_col -> source_col_name
        col_func_map = {}  # target_col -> full function expression (for SELECT/DECODE)
        insert_rename_map = {t: s for t, s in self.column_mappings if t not in self.func_columns}

        # First: populate from $TARGET_TABLE_COLUMNS (explicit list)
        for col in tgt_cols:
            if col in self.func_columns:
                # Function-wrapped: substr(X1INTNOTES,1,300) -> INTERNAL_NOTES
                col_func_map[col] = self.func_columns[col]
                col_name_map[col] = col  # placeholder
            elif col in upd_cols_map:
                col_name_map[col] = upd_cols_map[col]
            elif col in insert_rename_map:
                col_name_map[col] = insert_rename_map[col]
            else:
                col_name_map[col] = col  # same name

        # Second: populate from UPDATE_COLUMNS and INSERT INTO for columns NOT in $TARGET_TABLE_COLUMNS
        # (important when DESC_TABLE is used and $TARGET_TABLE_COLUMNS may be incomplete)
        for tgt_col, src_col in self.params.update_columns:
            if tgt_col not in col_name_map:
                col_name_map[tgt_col] = src_col
        for tgt_col, src_expr in self.column_mappings:
            if tgt_col not in col_name_map:
                col_name_map[tgt_col] = src_expr
        # Also populate from INSERT INTO columns (same-name mapping)
        for ins_col in self.insert_into_columns:
            if ins_col not in col_name_map:
                col_name_map[ins_col] = ins_col

        # Get split CTE columns
        split_cols = {c.upper() for c in self._get_split_cte_columns()}
        missing = []
        for c in tgt_cols:
            src = col_name_map.get(c, c)
            if c in col_func_map:
                continue  # function columns are always available
            if src not in split_cols and c not in split_cols:
                missing.append(c)
        self.missing_columns = missing
        missing_set = set(missing)

        join_cols = self.join_columns
        exclude_from_decode = {pk} | join_cols
        audit_timestamp_cols = {'LOAD_DT', 'LAST_UPDATE_LOAD_DT', 'EDW_LAST_UPDT_DTM', 'LAST_UPDT_TS',
                                'EDW_LAST_UPDT_TS', 'OUT_LAST_UPDT_TS', 'LAST_UPDATE_LOAD_ID'}

        # Determine output column order
        if has_desc:
            output_cols = [name for name, _typ, _dft in self.params.desc_table]
        else:
            output_cols = tgt_cols
        tgt_cols_set = {c.upper() for c in tgt_cols}
        # Also build set of ALL known columns from all sources (for DESC_TABLE mode)
        insert_into_set = {c.upper() for c in self.insert_into_columns}
        upd_tgt_set = upd_cols  # already a set of uppercase update target cols

        # SELECT lines -- use function expressions where applicable
        sel = []
        num_cols = len(output_cols)
        for idx, col in enumerate(output_cols):
            is_last = (idx == num_cols - 1)
            comma = '' if is_last else ','
            col_type = self._desc_type_for(col) if has_desc else None

            # If DESC_TABLE provided, check ALL possible sources for this column:
            # 1) $TARGET_TABLE_COLUMNS
            # 2) INSERT INTO column list
            # 3) $UPDATE_COLUMNS target side
            # 4) col_name_map (already built from all the above)
            is_in_source = (not has_desc) or (
                col.upper() in tgt_cols_set or
                col.upper() in insert_into_set or
                col.upper() in upd_tgt_set or
                col.upper() in col_name_map
            )

            if has_desc and not is_in_source:
                # DESC_TABLE column missing from $TARGET_TABLE_COLUMNS -> default or NULL
                default = self._desc_default_for(col)
                if default:
                    sel.append(f'    {default} AS {col}{comma}  -- defaulted from DESC_TABLE')
                else:
                    sel.append(f'    CAST(NULL AS {col_type}) AS {col}{comma}  -- missing, no default')
                continue

            src_col = col_name_map.get(col, col)

            # Validate: if mapped source col doesn't exist in split CTE,
            # check if the target col name itself exists in split CTE (common when
            # position-based pairing from INSERT INTO doesn't match split CTE layout)
            if src_col.upper() not in split_cols and col.upper() in split_cols:
                src_col = col  # use target name directly (it exists in split CTE)
            elif src_col.upper() not in split_cols:
                # Neither mapped source nor target name in split CTE - try finding
                # a case-insensitive match in split CTE for the target name
                for sc in split_cols:
                    if sc == col.upper():
                        src_col = sc
                        break

            func_expr = col_func_map.get(col)
            needs_alias = (src_col != col) or (func_expr is not None)
            is_missing = col in missing_set

            # If DESC_TABLE is provided and the column is truly missing from the split CTE
            # (not in INSERT INTO, not in update columns, source name not in split CTE),
            # then default it instead of generating a COALESCE/SRC reference that won't exist
            if has_desc and is_missing and col_type:
                # Check if the source column name is actually in the split CTE
                actual_src = src_col.upper()
                if actual_src not in split_cols and col.upper() not in split_cols:
                    default = self._desc_default_for(col)
                    if default:
                        sel.append(f'    {default} AS {col}{comma}  -- defaulted from DESC_TABLE')
                    else:
                        sel.append(f'    CAST(NULL AS {col_type}) AS {col}{comma}  -- not in split CTE, no default')
                    continue

            cmt = '  -- missing in split CTE' if is_missing else ''

            # Build the inner "value" expression (before CAST wrapping if applicable)
            if func_expr:
                src_func = self._prefix_func_cols(func_expr, 'SRC.')
                if col == pk or col in upd_cols:
                    value_expr = src_func
                else:
                    value_expr = f'COALESCE(TGT.{col}, {src_func})'
            elif col == pk or col in upd_cols:
                value_expr = f'SRC.{src_col}' if needs_alias else f'SRC.{col}'
            else:
                value_expr = f'COALESCE(TGT.{col}, SRC.{src_col})' if needs_alias else f'COALESCE(TGT.{col}, SRC.{col})'

            # For UPDATE columns, prefer SRC over TGT: COALESCE(SRC.x, TGT.y)
            # (current code uses COALESCE(TGT.y, SRC.x) - keep existing for update cols: ORIG PREF)
            # NOTE: keeping the existing COALESCE direction - not changing behavior here

            # Apply CAST wrapping if DESC_TABLE provided and type is known
            if has_desc and col_type:
                wrapped = self._cast_wrap(value_expr, col_type)
                sel.append(f'    {wrapped} AS {col}{comma}{cmt}')
            else:
                sel.append(f'    {value_expr} AS {col}{comma}{cmt}')

        # DECODE -- each DECODE has its own = 0, joined by OR
        # Categorize: active, audit (commented), join-key (commented)
        # Use function expressions for func_columns
        active_dec = []
        commented_dec = []

        for tc, sc in self.params.update_columns:
            if tc == pk:
                continue
            cmt_parts = []
            if tc in missing_set:
                cmt_parts.append('missing in split CTE')

            # Determine the source expression for DECODE
            if tc in col_func_map:
                src_expr = self._prefix_func_cols(col_func_map[tc], 'src.')
            else:
                src_expr = f'src.{sc}'

            # Target side always just 'tgt.col'
            tgt_expr = f'tgt.{tc}'

            # If DESC_TABLE is provided, wrap both sides with CAST/LEFT/TRIM for consistency
            dec_type = self._desc_type_for(tc) if has_desc else None
            if dec_type:
                src_wrapped = self._cast_wrap(src_expr, dec_type)
                tgt_wrapped = self._cast_wrap(tgt_expr, dec_type)
            else:
                src_wrapped = src_expr
                tgt_wrapped = tgt_expr

            if tc in join_cols:
                commented_dec.append((tc, tgt_wrapped, src_wrapped, 'JOIN KEY COLUMN, EXCLUDED FROM CHANGE DETECTION'))
            elif tc in audit_timestamp_cols:
                commented_dec.append((tc, tgt_wrapped, src_wrapped, 'AUDIT COLUMN, MAY NOT BE NEEDED - PLEASE VERIFY'))
            else:
                active_dec.append((tc, tgt_wrapped, src_wrapped, '  -- ' + ', '.join(cmt_parts) if cmt_parts else ''))

        # Build DECODE lines with correct logic: DECODE() = 0 OR DECODE() = 0
        dec_lines = []
        total_active = len(active_dec)

        for idx, (tc, tgt_expr, src_expr, cmt) in enumerate(active_dec):
            is_last_active = (idx == total_active - 1)
            or_part = '' if is_last_active else ' OR'
            dec_lines.append(f'   DECODE({tgt_expr}, {src_expr}, 1, 0) = 0{or_part}{cmt}')

        # Add commented-out lines (join keys and audit columns)
        for idx, (tc, tgt_expr, src_expr, reason) in enumerate(commented_dec):
            is_last = (idx == len(commented_dec) - 1)
            or_part = '' if is_last else ' OR'
            dec_lines.append(f'   --DECODE({tgt_expr}, {src_expr}, 1, 0) = 0{or_part}  -- {reason}')

        # Build WHERE clause
        if dec_lines:
            where = "WHERE\n  (\n" + chr(10).join(dec_lines) + "\n  )"
        else:
            where = ''

        split = self.params.split_cte

        # Use source column name for PK in JOIN (may differ from target PK name)
        pk_src = col_name_map.get(pk, pk)
        # Validate: if mapped PK source col doesn't exist in split CTE, check alternatives
        if pk_src.upper() not in split_cols and pk.upper() in split_cols:
            pk_src = pk
        elif pk_src.upper() not in split_cols:
            # Search split CTE for a column that matches the INSERT INTO mapping
            for sc in split_cols:
                if sc == pk.upper():
                    pk_src = sc
                    break

        return '\n'.join([
            '', 'SELECT', '\n'.join(sel), '',
            'FROM', f'    {split} SRC',
            'LEFT JOIN', '    {{this}} TGT',
            'ON', f'    SRC.{pk_src} = TGT.{pk}',
        ] + ([where] if where else []))

    # =========================================================================
    # INSERT-ONLY SELECT (no merge, all from split CTE)
    # =========================================================================

    def _gen_insert_select(self) -> str:
        """Generate SELECT for insert-only models.
        All columns from split CTE, using $TARGET_TABLE_COLUMNS order.

        If $DESC_TABLE is provided (optional for INSERT type):
          - Column order = DESC_TABLE order (canonical)
          - Each column wrapped with CAST (+ optional TRIM, + LEFT for VARCHAR)
          - Columns in DESC_TABLE but not in $TARGET_TABLE_COLUMNS get:
            - default value AS col  (if DESC_TABLE has default)
            - CAST(NULL AS type) AS col  (otherwise)
        """
        tgt_cols = self.params.target_table_columns
        split = self.params.split_cte
        has_desc = bool(self.params.desc_table)

        if not split:
            self.warnings.append("No $SPLIT_CTE defined for insert-only model")
            return ''

        # Build column name mapping from INSERT INTO analysis
        insert_map = {t: s for t, s in self.column_mappings}
        # Also include UPDATE_COLUMNS mappings for completeness
        for tgt_col, src_col in self.params.update_columns:
            if tgt_col not in insert_map:
                insert_map[tgt_col] = src_col

        # Get split CTE columns for validation
        split_cols = {c.upper() for c in self._get_split_cte_columns()}

        # Determine output column order
        if has_desc:
            # DESC_TABLE order (canonical physical order)
            output_cols = [name for name, _typ, _dft in self.params.desc_table]
        else:
            output_cols = tgt_cols

        tgt_cols_set = {c.upper() for c in tgt_cols}
        insert_into_set = {c.upper() for c in self.insert_into_columns}

        # Missing columns: present in $TARGET_TABLE_COLUMNS but not in split CTE
        missing = []
        for c in tgt_cols:
            src_name = insert_map.get(c, c)
            if src_name not in split_cols and c not in split_cols:
                missing.append(c)
        self.missing_columns = missing
        missing_set = set(missing)

        if not output_cols:
            return f"\nSELECT\n    *\nFROM\n    {split}"

        # Build SELECT
        sel = []
        num_cols = len(output_cols)
        for idx, col in enumerate(output_cols):
            is_last = (idx == num_cols - 1)
            comma = '' if is_last else ','
            col_type = self._desc_type_for(col) if has_desc else None

            # Check ALL possible sources for this column
            is_in_source = (not has_desc) or (
                col.upper() in tgt_cols_set or
                col.upper() in insert_into_set or
                col.upper() in insert_map
            )

            if has_desc and not is_in_source:
                # DESC_TABLE column not in $TARGET_TABLE_COLUMNS -> use default or NULL
                default = self._desc_default_for(col)
                if default:
                    sel.append(f'    {default} AS {col}{comma}  -- defaulted from DESC_TABLE')
                else:
                    sel.append(f'    CAST(NULL AS {col_type}) AS {col}{comma}  -- missing, no default')
                continue

            src_col = insert_map.get(col, col)
            needs_alias = (src_col != col)
            cmt = '  -- missing in split CTE' if col in missing_set else ''

            if has_desc and col_type:
                # Cast-wrap (with optional TRIM, LEFT for VARCHAR)
                src_expr = f'SRC.{src_col}'
                wrapped = self._cast_wrap(src_expr, col_type)
                sel.append(f'    {wrapped} AS {col}{comma}{cmt}')
            else:
                # Original behavior when DESC_TABLE is not provided
                if needs_alias:
                    sel.append(f'    SRC.{src_col} AS {col}{comma}{cmt}')
                else:
                    sel.append(f'    SRC.{col}{comma}{cmt}')

        return '\n'.join([
            '', 'SELECT', '\n'.join(sel), '',
            'FROM', f'    {split} SRC',
        ])

    # =========================================================================
    # STG MODEL (truncate + recreate pattern)
    # =========================================================================

    def _build_stg_model(self) -> str:
        """Build dbt model for STG type.
        - truncate+recreate via pre-hook (added in _gen_config)
        - Source: INSERT INTO TABLE (cols) WITH ctes... SELECT cols FROM cte
        - Output columns in DESC_TABLE order (canonical physical table order)
        - Each column: CAST([TRIM(]source_col[)], TYPE) AS target_col, or default, or NULL
        """
        parts = []
        parts.append(self._gen_config(is_stg=True))
        parts.append('')

        source_sql = self.params.source_code
        if not source_sql:
            self.warnings.append("No $SOURCE_CODE found")
            return '\n'.join(parts)

        # Replace tables, highlight hardcoded values (ref/source lookups still apply)
        transformed = self._replace_tables(source_sql)
        transformed = self._highlight_hardcoded(transformed)

        # Extract CTEs
        ctes, final_select = self._extract_ctes(transformed)

        has_ctes = len(ctes) > 0
        if has_ctes:
            parts.append('WITH')
            num_ctes = len(ctes)
            for i, (name, body) in enumerate(ctes):
                sep = '' if (i == num_ctes - 1) else ','
                parts.append(f"{name} AS (\n{self._indent(body, '  ')}\n){sep}")

        # Build final SELECT
        parts.append(self._gen_stg_select(final_select))
        return '\n'.join(parts)

    def _gen_stg_select(self, final_select_sql: Optional[str]) -> str:
        """Generate final SELECT for STG model.
        Uses DESC_TABLE as the canonical column list (in DESC order).
        Each target column either:
          - Matches an INSERT INTO column -> take source expression from original SELECT
            (paired by INSERT INTO position), wrap in CAST([TRIM(]...)], TYPE)
          - Has a default in DESC_TABLE -> use default AS target
          - Otherwise -> NULL AS target
        """
        # Determine the FROM source
        from_clause = 'CTE  -- TODO: replace with actual final source'

        if final_select_sql:
            # Extract FROM ... clause from the original final SELECT
            fm_from = re.search(r'\bFROM\s+(.+?)(?:\s*;\s*)?$', final_select_sql, re.I | re.DOTALL)
            if fm_from:
                from_src = fm_from.group(1).strip().rstrip(';').strip()
                from_clause = from_src

            # Extract SELECT expression list from original final SELECT
            fm_sel = re.search(r'SELECT\s+(.*?)\s*FROM\s+', final_select_sql, re.I | re.DOTALL)
            select_exprs = self._split_cols(fm_sel.group(1).strip()) if fm_sel else []
        else:
            select_exprs = []

        # Pair INSERT INTO position N <-> original SELECT position N
        # insert_into_columns was populated by _analyze_insert_columns
        ins_cols = self.insert_into_columns
        # Build a map: target_col_upper -> source_expression (raw as appeared in final SELECT)
        insert_src_map: Dict[str, str] = {}
        for ic, se in zip(ins_cols, select_exprs):
            expr = se.strip().rstrip(',').strip()
            # Strip trailing inline comments
            expr = re.sub(r'\s*--.*$', '', expr).strip()
            # Strip trailing 'AS alias' so we don't double-wrap things like
            # 'LEFT(LTRIM(RTRIM(col)), 50) AS TRIMODIFIEDBYTX' into garbage CAST
            expr = self._strip_alias(expr)
            if not expr:
                continue
            insert_src_map[ic.upper()] = expr

        # Build CAST SELECT list in DESC_TABLE order
        sel_lines = []
        stg_defaulted = []
        desc_cols = self.params.desc_table

        for idx, (col_name, col_type, default) in enumerate(desc_cols):
            is_last = (idx == len(desc_cols) - 1)
            comma = '' if is_last else ','
            col_upper = col_name.upper()

            if col_upper in insert_src_map:
                src_expr = insert_src_map[col_upper]

                # Check if it's a VARCHAR type - extract the length if so
                varchar_match = re.match(r'VARCHAR\s*\(\s*(\d+)\s*\)', col_type, re.I)
                is_varchar = bool(varchar_match)
                varchar_len = varchar_match.group(1) if varchar_match else None

                # Build the inner expression: optionally TRIM, then optionally LEFT for VARCHAR
                if self.use_trim:
                    inner = f"TRIM({src_expr})"
                else:
                    inner = src_expr

                if is_varchar:
                    # Wrap with LEFT(..., N) to protect against source values exceeding column length
                    inner = f"LEFT({inner}, {varchar_len})"

                sel_lines.append(f"    CAST({inner} AS {col_type}) AS {col_name}{comma}")
            else:
                # Not in INSERT INTO -> default or NULL
                if default and default.strip():
                    default_val = default.strip()
                    sel_lines.append(f"    {default_val} AS {col_name}{comma}  -- defaulted from DESC_TABLE")
                    stg_defaulted.append((col_name, col_type, default_val))
                else:
                    # Cast NULL to the target type so Snowflake doesn't treat it as VARCHAR(16777216)
                    sel_lines.append(f"    CAST(NULL AS {col_type}) AS {col_name}{comma}  -- missing, no default")
                    stg_defaulted.append((col_name, col_type, f'CAST(NULL AS {col_type})'))

        self.stg_defaulted = stg_defaulted

        return '\n'.join([
            '', 'SELECT',
            '\n'.join(sel_lines),
            'FROM',
            f'    {from_clause}',
        ])

    @staticmethod
    def _strip_alias(expr: str) -> str:
        """Strip a trailing 'AS alias' (or just 'alias') from a SELECT expression.
        Used in STG mode so we don't double-wrap expressions like
        'LEFT(LTRIM(RTRIM(col)), 50) AS TRIMODIFIEDBYTX' inside another CAST/TRIM/LEFT.

        Only strips at TOP LEVEL (depth 0 of parens, not inside quotes).
        Returns the raw expression without the alias.

        Examples:
          'LEFT(LTRIM(RTRIM(col)), 50) AS TRIMODIFIEDBYTX' -> 'LEFT(LTRIM(RTRIM(col)), 50)'
          'col_name AS alias_name'                          -> 'col_name'
          'CAST(x AS VARCHAR(10))'                          -> 'CAST(x AS VARCHAR(10))'  (no top-level AS)
          'src.col'                                         -> 'src.col'
        """
        if not expr:
            return expr
        # Walk the string, tracking paren depth and quote state.
        # Find the LAST top-level whitespace (not inside parens/quotes) that
        # could be a separator between expression and alias.
        depth = 0
        in_single = False
        in_double = False
        i = 0
        # Find positions of all top-level 'AS' keywords (case-insensitive, word-bounded)
        last_as_pos = -1
        while i < len(expr):
            ch = expr[i]
            if ch == "'" and not in_double:
                in_single = not in_single
                i += 1
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                i += 1
                continue
            if not in_single and not in_double:
                if ch == '(':
                    depth += 1
                    i += 1
                    continue
                if ch == ')':
                    depth -= 1
                    i += 1
                    continue
                # Look for 'AS' at depth 0 (case-insensitive, word boundary)
                if depth == 0 and i + 1 < len(expr):
                    if expr[i:i+2].upper() == 'AS':
                        # Check word boundary on both sides
                        prev_ok = (i == 0 or not (expr[i-1].isalnum() or expr[i-1] == '_'))
                        end_idx = i + 2
                        next_ok = (end_idx >= len(expr) or not (expr[end_idx].isalnum() or expr[end_idx] == '_'))
                        if prev_ok and next_ok:
                            last_as_pos = i
            i += 1

        if last_as_pos == -1:
            # No top-level AS found - return as-is
            return expr.strip()

        # Strip the AS alias portion
        return expr[:last_as_pos].strip()

    def _desc_type_for(self, col_name: str) -> Optional[str]:
        """Look up column type from DESC_TABLE by (case-insensitive) name.
        Returns the type string (e.g. 'VARCHAR(100)') or None if not found.
        """
        if not self.params.desc_table:
            return None
        col_upper = col_name.upper()
        for name, typ, _default in self.params.desc_table:
            if name.upper() == col_upper:
                return typ
        return None

    def _desc_default_for(self, col_name: str) -> Optional[str]:
        """Look up default value from DESC_TABLE by (case-insensitive) name.
        Returns the default string or None if not found / empty.
        """
        if not self.params.desc_table:
            return None
        col_upper = col_name.upper()
        for name, _typ, default in self.params.desc_table:
            if name.upper() == col_upper:
                return default.strip() if default and default.strip() else None
        return None

    def _cast_wrap(self, expr: str, col_type: str) -> str:
        """Wrap an expression with CAST (and optionally TRIM / LEFT for VARCHAR).
        Used for both SELECT-side wrapping and DECODE-side wrapping when DESC_TABLE provided.
        If skip_numeric (-non_num flag): skip CAST for NUMBER/INT types, only cast VARCHAR.
        """
        if not col_type:
            return expr
        # Skip numeric types if -non_num flag
        if self.skip_numeric:
            is_numeric = any(col_type.upper().startswith(t) for t in ('NUMBER', 'NUMERIC', 'DECIMAL', 'FLOAT', 'DOUBLE', 'INT', 'BIGINT', 'SMALLINT', 'TINYINT'))
            if is_numeric:
                return expr
        inner = expr
        if self.use_trim:
            inner = f"TRIM({inner})"
        varchar_match = re.match(r'VARCHAR\s*\(\s*(\d+)\s*\)', col_type, re.I)
        if varchar_match:
            inner = f"LEFT({inner}, {varchar_match.group(1)})"
        return f"CAST({inner} AS {col_type})"

    # =========================================================================
    # COLUMN HELPERS
    # =========================================================================

    def _prefix_func_cols(self, func_expr: str, prefix: str) -> str:
        """Add table prefix to column references inside a function expression.
        e.g. substr(X1INTNOTES,1,300) -> substr(SRC.X1INTNOTES,1,300)
        Only prefixes identifiers that look like column names (not numbers, strings, keywords).
        """
        # Find the first argument that looks like a column name (not a number or string)
        # Strategy: find the function name and opening paren, then prefix the first identifier
        m = re.match(r'(\w+)\s*\(', func_expr)
        if not m:
            return prefix + func_expr
        func_name = m.group(1)
        inner = func_expr[m.end():-1] if func_expr.endswith(')') else func_expr[m.end():]

        # Split args, prefix column-like args
        args = self._split_cols(inner)
        new_args = []
        for arg in args:
            arg = arg.strip()
            # If it's a plain identifier (column name), prefix it
            if re.match(r'^[A-Za-z_]\w*$', arg) and arg.upper() not in (
                'TRUE', 'FALSE', 'NULL', 'AND', 'OR', 'NOT', 'AS', 'IS',
                'THEN', 'WHEN', 'ELSE', 'END', 'CASE', 'IN', 'LIKE'):
                new_args.append(prefix + arg)
            else:
                new_args.append(arg)

        return f"{func_name}({', '.join(new_args)})"

    def _get_split_cte_columns(self) -> List[str]:
        sql = self.params.source_code
        if not sql:
            return []
        # Use raw SQL (before table replacement) for CTE extraction
        ctes, _ = self._extract_ctes(sql)
        split_name = (self.params.split_cte or '').upper()
        cte_names = [name.upper() for name, body in ctes]

        # Fix 28: Validate split CTE name exists
        if split_name and split_name not in cte_names:
            self.warnings.append(
                f"$SPLIT_CTE '{self.params.split_cte}' not found in extracted CTEs. "
                f"Available CTEs: {', '.join(name for name, _ in ctes)}"
            )

        for name, body in ctes:
            if name.upper() == split_name:
                cols = self._parse_select_columns(body)
                if cols:
                    return cols
        # Fallback: last CTE
        if ctes:
            return self._parse_select_columns(ctes[-1][1])
        return []

    def _parse_select_columns(self, sql: str) -> List[str]:
        # Find the outer SELECT and the top-level FROM (at paren depth 0)
        # This correctly skips nested subqueries like COALESCE((SELECT MAX(x) FROM t), 0)

        # First find SELECT keyword
        sel_match = re.match(r'\s*SELECT\s+', sql, re.I)
        if sel_match:
            after_select = sel_match.end()
        else:
            # Tolerant: no SELECT keyword (legacy SQL)
            after_select = 0

        # Walk through and find the top-level FROM at depth 0
        depth = 0
        in_single, in_double = False, False
        i = after_select
        from_pos = -1
        while i < len(sql):
            ch = sql[i]
            # Skip -- comments
            if not in_single and not in_double and ch == '-' and i + 1 < len(sql) and sql[i+1] == '-':
                eol = sql.find('\n', i)
                if eol == -1:
                    i = len(sql)
                else:
                    i = eol + 1
                continue
            if ch == "'" and not in_double:
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double
            elif not in_single and not in_double:
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                # Check for FROM at depth 0
                elif depth == 0 and sql[i:i+4].upper() == 'FROM':
                    # Verify word boundary
                    before_ok = (i == 0 or not (sql[i-1].isalnum() or sql[i-1] == '_'))
                    after_ok = (i + 4 >= len(sql) or not (sql[i+4].isalnum() or sql[i+4] == '_'))
                    if before_ok and after_ok:
                        from_pos = i
                        break
            i += 1

        if from_pos == -1:
            return []

        col_text = sql[after_select:from_pos].strip()
        if not col_text:
            return []
        # Check it's not garbage
        if col_text.upper().startswith(('UPDATE', 'DELETE', 'INSERT', 'WITH')):
            return []

        sp = re.sub(r'^\s*DISTINCT\s+', '', col_text, flags=re.I)
        if sp.strip() == '*':
            return []
        cols = []
        for expr in self._split_cols(sp):
            expr = expr.strip()
            if not expr or expr == '*':
                continue
            if expr.lstrip().startswith('--'):
                continue
            expr_clean = re.sub(r'\s*--.*$', '', expr, flags=re.MULTILINE).strip()
            if not expr_clean:
                continue
            am = re.search(r'\bAS\s+(\w+)\s*$', expr_clean, re.I)
            if am:
                cols.append(am.group(1).upper())
            else:
                pts = expr_clean.split()
                if len(pts) >= 2 and re.match(r'^\w+$', pts[-1]):
                    cols.append(pts[-1].upper())
                else:
                    col_name = expr_clean.split('.')[-1].strip()
                    if col_name and re.match(r'^\w+$', col_name):
                        cols.append(col_name.upper())
        return cols

    def _split_cols(self, s: str) -> List[str]:
        out, cur, d = [], [], 0
        in_single, in_double = False, False
        i = 0
        while i < len(s):
            ch = s[i]
            # Skip -- comments to end of line (don't split on commas inside them)
            if not in_single and not in_double and ch == '-' and i + 1 < len(s) and s[i + 1] == '-':
                # Append everything from -- to end of line into current token
                eol = s.find('\n', i)
                if eol == -1:
                    cur.append(s[i:])
                    i = len(s)
                else:
                    cur.append(s[i:eol])
                    i = eol  # let the next iteration handle \n
                continue
            if ch == "'" and not in_double:
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double
            if not in_single and not in_double:
                if ch == '(':
                    d += 1
                elif ch == ')':
                    d -= 1
            if ch == ',' and d == 0 and not in_single and not in_double:
                out.append(''.join(cur))
                cur = []
            else:
                cur.append(ch)
            i += 1
        if cur:
            out.append(''.join(cur))
        return out

    def _detect_missing(self, sql_cols: List[str]):
        sql_up = {c.upper() for c in sql_cols}
        self.missing_columns = [c for c in self.params.target_table_columns if c not in sql_up]

    # =========================================================================
    # HOOKS
    # =========================================================================

    def _transform_hook(self, sql: str) -> str:
        if self.params.target_table:
            sql = re.compile(re.escape(self.params.target_table), re.I).sub('{{this}}', sql)
        sql = self._replace_tables(sql)
        sql = self._highlight_hardcoded(sql)
        sql = self._inject_job_ctes_into_hook(sql)
        return sql

    def _inject_job_ctes_into_hook(self, sql: str) -> str:
        with_match = re.search(r'\bWITH\s+', sql, re.I)
        if not with_match:
            return sql
        ms, param_tbl, ctrl_tbl = self.sources.resolve_metadata(
            self.params.target_db, self.params.target_schema)
        jn = self.params.target_table_name
        autosys = self.params.autosys_job_name
        if ms:
            param_ref = f"{{{{source('{ms}','{param_tbl}')}}}}"
            ctrl_ref = f"{{{{source('{ms}','{ctrl_tbl}')}}}}"
        else:
            param_ref = f"/* TODO */ METADATA.{param_tbl}"
            ctrl_ref = f"/* TODO */ METADATA.{ctrl_tbl}"
        autosys_val = autosys if autosys else '/* TODO: set $AUTOSYS_JOB_NAME */'
        pw = f"WHERE JOB_NAME= '{jn}' AND AUTOSYS_JOB_NAME = '{autosys_val}' AND ACTIVE_IND = 'Y'"
        cw = f"WHERE JOB_NAME= '{jn}' AND AUTOSYS_JOB_NAME = '{autosys_val}'"
        job_ctes = (
            f"JOB_PARAM AS (\n"
            f"    SELECT PARAM_ID, PARAM_NAME, PARAM_VALUE FROM {param_ref}\n"
            f"    {pw}\n"
            f"),\n"
            f"JOB_CONTROL AS (\n"
            f"    SELECT JOB_ID, START_DATE, END_DATE FROM {ctrl_ref}\n"
            f"    {cw}\n"
            f"),\n"
        )
        insert_pos = with_match.end()
        return sql[:insert_pos] + '\n' + job_ctes + sql[insert_pos:]

    # =========================================================================
    # WARNING HEADER
    # =========================================================================

    def _warning_header(self) -> str:
        secs = []

        # AUTOSYS JOB NAME info
        autosys = self.params.autosys_job_name
        if autosys:
            secs.append(f"AUTOSYS_JOB_NAME: {autosys}")
        else:
            secs.append("TODO: $AUTOSYS_JOB_NAME not provided - add it to input file")

        # Parameters that need JOB_PARAM/JOB_CONTROL entries
        if self.found_params:
            secs.append("")
            secs.append("PARAMETERS TO ADD IN JOB_PARAM / JOB_CONTROL:")
            seen_types = set()
            for ptype, line_content in self.found_params:
                if ptype not in seen_types:
                    seen_types.add(ptype)
                    if ptype == 'ROWSTAMP':
                        secs.append(f"  - ROWSTAMP -> add entry in JOB_PARAM (PARAM_NAME='ROWSTAMP')")
                    elif ptype == 'LOAD_ID':
                        secs.append(f"  - LOAD_ID -> add entry in JOB_PARAM (PARAM_NAME='LOAD_ID')")
                    elif ptype == 'LAST_UPDATE_LOAD_ID':
                        secs.append(f"  - LAST_UPDATE_LOAD_ID -> add entry in JOB_PARAM (PARAM_NAME='LAST_UPDATE_LOAD_ID')")
                    elif ptype == 'DATE_RANGE':
                        secs.append(f"  - DATE_RANGE -> parameterize via JOB_CONTROL (START_DATE / END_DATE)")
            secs.append("  Found at:")
            for ptype, line_content in self.found_params:
                display = line_content[:120] + '...' if len(line_content) > 120 else line_content
                secs.append(f"    [{ptype}] {display}")

        # Function-wrapped columns (e.g. substr, IFF, CASE)
        if self.func_columns:
            secs.append("")
            secs.append("FUNCTION-WRAPPED COLUMNS (source uses function, included in SELECT and DECODE):")
            for tgt_col, func_expr in self.func_columns.items():
                secs.append(f"  - {tgt_col} <- {func_expr}")

        # Column name differences between INSERT INTO (target) and SELECT (source)
        # Exclude func_columns from this section (already shown above)
        rename_mappings = [(t, s) for t, s in self.column_mappings if t not in self.func_columns]
        if rename_mappings:
            secs.append("")
            secs.append("COLUMN NAME MAPPINGS (source -> target, names differ during INSERT):")
            for tgt_col, src_col in rename_mappings:
                secs.append(f"  - {src_col} -> {tgt_col}")

        # Defaulted columns (in TARGET_TABLE_COLUMNS but not in INSERT INTO)
        if self.defaulted_columns:
            secs.append("")
            secs.append("POSSIBLY DEFAULTED COLUMNS (in $TARGET_TABLE_COLUMNS but not in INSERT INTO list):")
            secs.extend(f"  - {c}" for c in self.defaulted_columns)

        # STG defaulted columns (in DESC_TABLE but not in INSERT INTO)
        if self.stg_defaulted:
            secs.append("")
            secs.append("STG DEFAULTED COLUMNS (in $DESC_TABLE but not in INSERT INTO - defaulted in final SELECT):")
            for name, col_type, default in self.stg_defaulted:
                secs.append(f"  - {name} ({col_type}) -> {default}")

        # Missing columns in split CTE
        if self.missing_columns:
            secs.append("")
            secs.append("MISSING COLUMNS (in $TARGET_TABLE_COLUMNS but not in split CTE):")
            secs.extend(f"  - {c}" for c in self.missing_columns)

        # Unresolved tables
        if self.unresolved_tables:
            secs.append("")
            secs.append("UNRESOLVED TABLES (not in sources.yml):")
            secs.extend(f"  - {t}" for t in self.unresolved_tables)

        # Schema-only resolved tables
        if self.schema_only_tables:
            secs.append("")
            secs.append("TABLES RESOLVED BY SCHEMA (may need to add to sources.yml):")
            secs.extend(f"  - {t}" for t in self.schema_only_tables)

        # Other warnings
        if self.warnings:
            secs.append("")
            secs.append("WARNINGS:")
            secs.extend(f"  - {w}" for w in self.warnings)

        if not secs:
            return ''
        return '\n'.join(
            ['-- ' + '=' * 60, '-- CONVERSION SUMMARY', '-- ' + '=' * 60] +
            [f'-- {s}' for s in secs] +
            ['-- ' + '=' * 60, '']
        )

    def _indent(self, t, p):
        return '\n'.join(p + l if l.strip() else l for l in t.split('\n'))


# =============================================================================
# MERGE TYPE: Data classes, Parser, and Transformer
# (Handles $TYPE: MERGE - Snowflake MERGE INTO statements)
# Auto-extracts target table, PKs, update cols, insert cols from MERGE body.
# Supports optional $DESC_TABLE for CAST/TRIM/LEFT wrapping.
# =============================================================================

@dataclass
class MergeStatement:
    """Parsed MERGE statement components."""
    target_table: str = ""
    target_alias: str = "tgt"
    source_expression: str = ""
    source_alias: str = "src"
    on_clause: str = ""
    primary_keys: List[str] = field(default_factory=list)
    update_columns: List[Tuple[str, str]] = field(default_factory=list)
    insert_columns: List[str] = field(default_factory=list)
    insert_values: List[str] = field(default_factory=list)
    ctes: List[Tuple[str, str]] = field(default_factory=list)


class MergeStatementParser:
    """Parses a Snowflake MERGE statement (WITH CTEs + MERGE INTO ...)."""

    @staticmethod
    def parse(sql: str) -> MergeStatement:
        m = MergeStatement()
        sql = MergeStatementParser._strip_leading_comments(sql)
        m.ctes, remaining = MergeStatementParser._extract_ctes(sql)
        MergeStatementParser._parse_merge(remaining, m)
        return m

    @staticmethod
    def _strip_leading_comments(sql: str) -> str:
        lines = sql.split('\n')
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if not stripped or stripped.startswith('--'):
                i += 1
            else:
                break
        return '\n'.join(lines[i:])

    @staticmethod
    def _extract_ctes(sql: str) -> Tuple[List[Tuple[str, str]], str]:
        ctes = []
        wm = re.match(r'\s*WITH\s+', sql, re.I)
        if not wm:
            return ctes, sql
        rem = sql[wm.end():]
        while rem.strip():
            nm = re.match(r'\s*(\w+)\s+AS\s*\(\s*', rem, re.I)
            if not nm:
                # Tolerant: name ( without AS
                nm = re.match(r'\s*(\w+)\s*\(\s*', rem)
                if not nm:
                    break
            cte_name = nm.group(1)
            # Find opening paren
            paren_start = rem.find('(', nm.start())
            if paren_start == -1:
                break
            pos = paren_start + 1
            depth = 1
            i = pos
            in_single, in_double = False, False
            while i < len(rem) and depth > 0:
                ch = rem[i]
                if not in_single and not in_double and i + 1 < len(rem) and rem[i:i+2] == '--':
                    eol = rem.find('\n', i)
                    if eol == -1:
                        i = len(rem)
                    else:
                        i = eol + 1
                    continue
                if ch == "'" and not in_double:
                    in_single = not in_single
                elif ch == '"' and not in_single:
                    in_double = not in_double
                elif not in_single and not in_double:
                    if ch == '(':
                        depth += 1
                    elif ch == ')':
                        depth -= 1
                i += 1
            cte_body = rem[pos:i - 1].strip()
            ctes.append((cte_name, cte_body))
            rem = rem[i:].lstrip()
            # Skip comments
            while rem.startswith('--'):
                eol = rem.find('\n')
                if eol == -1:
                    rem = ''
                    break
                rem = rem[eol + 1:].lstrip()
            if rem.startswith(','):
                rem = rem[1:].lstrip()
                while rem.startswith('--'):
                    eol = rem.find('\n')
                    if eol == -1:
                        rem = ''
                        break
                    rem = rem[eol + 1:].lstrip()
            elif re.match(r'\s*MERGE\s+', rem, re.I):
                break
            elif re.match(r'\s*\w+\s*(?:AS\s*)?\(', rem, re.I):
                pass  # tolerant: missing comma
            elif not rem.strip():
                break
        return ctes, rem

    @staticmethod
    def _parse_merge(sql: str, m: MergeStatement):
        merge_match = re.match(r'\s*MERGE\s+INTO\s+(\S+)(?:\s+(\w+))?\s+', sql, re.I)
        if not merge_match:
            raise ValueError("MERGE INTO statement not found")
        m.target_table = merge_match.group(1)
        if merge_match.group(2):
            m.target_alias = merge_match.group(2)
        rem = sql[merge_match.end():]

        # Handle two USING patterns:
        # 1) USING cte_name alias  (simple)
        # 2) USING (WITH ... SELECT ... FROM ...) alias  (subquery with CTEs)
        using_kw = re.match(r'USING\s+', rem, re.I)
        if not using_kw:
            raise ValueError("USING clause not found")
        rem = rem[using_kw.end():]

        if rem.lstrip().startswith('('):
            # Subquery pattern: USING (...) alias
            # Find matching closing paren
            pos = rem.index('(') + 1
            depth = 1
            i = pos
            in_single, in_double = False, False
            while i < len(rem) and depth > 0:
                ch = rem[i]
                if not in_single and not in_double and i + 1 < len(rem) and rem[i:i+2] == '--':
                    eol = rem.find('\n', i)
                    if eol == -1:
                        i = len(rem)
                    else:
                        i = eol + 1
                    continue
                if ch == "'" and not in_double:
                    in_single = not in_single
                elif ch == '"' and not in_single:
                    in_double = not in_double
                elif not in_single and not in_double:
                    if ch == '(':
                        depth += 1
                    elif ch == ')':
                        depth -= 1
                i += 1
            subquery = rem[pos:i - 1].strip()
            rem = rem[i:].lstrip()

            # Extract alias after closing paren
            alias_match = re.match(r'(\w+)\s+', rem)
            if alias_match:
                m.source_alias = alias_match.group(1)
                rem = rem[alias_match.end():]

            # Parse CTEs from inside the subquery and find the final SELECT
            sub_ctes, sub_final = MergeStatementParser._extract_ctes(subquery)
            if sub_ctes:
                m.ctes.extend(sub_ctes)

            # The source expression is the last CTE or the subquery's FROM table
            if sub_final:
                # Find what the final SELECT reads FROM
                from_match = re.search(r'\bFROM\s+(\w+)', sub_final, re.I)
                if from_match:
                    m.source_expression = from_match.group(1)
                else:
                    m.source_expression = sub_ctes[-1][0] if sub_ctes else 'SUBQUERY'
            elif sub_ctes:
                m.source_expression = sub_ctes[-1][0]
            else:
                m.source_expression = 'SUBQUERY'
        else:
            # Simple pattern: USING cte_name alias
            simple_match = re.match(r'(\S+)(?:\s+(\w+))?\s+', rem)
            if not simple_match:
                raise ValueError("Could not parse USING clause")
            m.source_expression = simple_match.group(1)
            if simple_match.group(2):
                m.source_alias = simple_match.group(2)
            rem = rem[simple_match.end():]

        on_match = re.match(r'ON\s+(.*?)(?=\bWHEN\b)', rem, re.I | re.DOTALL)
        if not on_match:
            raise ValueError("ON clause not found")
        m.on_clause = on_match.group(1).strip()
        m.primary_keys = [match.group(1).upper()
                          for match in re.finditer(rf'\b{re.escape(m.target_alias)}\.(\w+)\s*=', m.on_clause, re.I)]
        rem = rem[on_match.end():]

        update_match = re.search(
            r'WHEN\s+MATCHED\s+THEN\s+(?:UPDATE\s+SET|UPDATE)\s+(.*?)(?=\bWHEN\s+NOT\s+MATCHED\b|$)',
            rem, re.I | re.DOTALL)
        if update_match:
            body = re.sub(r'--.*?(?:\n|$)', '\n', update_match.group(1).strip())
            parts = MergeStatementParser._split_args(body)
            for part in parts:
                part = part.strip().rstrip(',').strip()
                if not part or part.startswith('--'):
                    continue
                pm = re.match(rf'{re.escape(m.target_alias)}\.(\w+)\s*=\s*(.+)$', part, re.I | re.DOTALL)
                if pm:
                    m.update_columns.append((pm.group(1).upper(), pm.group(2).strip()))

        insert_match = re.search(
            r'WHEN\s+NOT\s+MATCHED\s+THEN\s+INSERT\s*\((.*?)\)\s*VALUES\s*\((.*?)\)\s*;?\s*$',
            rem, re.I | re.DOTALL)
        if insert_match:
            m.insert_columns = [c.upper() for c in MergeStatementParser._split_args(insert_match.group(1))]
            m.insert_values = MergeStatementParser._split_args(insert_match.group(2))

    @staticmethod
    def _split_args(s: str) -> List[str]:
        out, cur = [], []
        depth, in_single, in_double = 0, False, False
        i = 0
        while i < len(s):
            ch = s[i]
            if not in_single and not in_double and i + 1 < len(s) and s[i:i+2] == '--':
                eol = s.find('\n', i)
                if eol == -1:
                    break
                i = eol + 1
                continue
            if ch == "'" and not in_double:
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double
            if not in_single and not in_double:
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
            if ch == ',' and depth == 0 and not in_single and not in_double:
                out.append(''.join(cur).strip())
                cur = []
            else:
                cur.append(ch)
            i += 1
        if cur:
            out.append(''.join(cur).strip())
        return [x for x in out if x]


class MergeTransformer:
    """Transformer for $TYPE: MERGE - Snowflake MERGE INTO statements.
    Completely separate from SQLTransformer to avoid touching existing logic.
    Supports: $DESC_TABLE (optional), -trim, -cc, $PRE_HOOK/$POST_HOOK,
    $AUTOSYS_JOB_NAME, $TARGET_TABLE_COLUMNS (optional).
    """

    def __init__(self, params: ParsedParams, sources: SourcesRegistry, use_trim: bool = False, skip_numeric: bool = False):
        self.params = params
        self.sources = sources
        self.use_trim = use_trim
        self.skip_numeric = skip_numeric
        self.warnings: List[str] = []
        self.missing_columns: List[str] = []
        self.unresolved_tables: List[str] = []
        self.audit_columns = {
            'LOAD_DT', 'LAST_UPDATE_LOAD_DT', 'EDW_LAST_UPDT_DTM',
            'LAST_UPDT_TS', 'EDW_LAST_UPDT_TS', 'OUT_LAST_UPDT_TS',
            'LAST_UPDATE_LOAD_ID', 'LOAD_ID',
        }
        # Also detect audit columns by pattern (any column ending with _TS or _DTM
        # that maps to CURRENT_TIMESTAMP or similar)
        self._audit_patterns = ['LAST_UPDT', 'LAST_UPDATE', 'EDW_LAST', 'LOAD_DT']

    def transform(self) -> str:
        try:
            merge = MergeStatementParser.parse(self.params.source_code)
        except Exception as e:
            self.warnings.append(f"MERGE parse error: {e}")
            return f"-- ERROR: Could not parse MERGE statement: {e}"

        has_desc = bool(self.params.desc_table)
        desc_map = {n: (t, d) for n, t, d in self.params.desc_table} if has_desc else {}

        # Determine column ordering
        if has_desc:
            output_cols = [n for n, _t, _d in self.params.desc_table]
        elif merge.insert_columns:
            output_cols = list(merge.insert_columns)
        else:
            output_cols = [t for t, _ in merge.update_columns]

        pk_set = set(merge.primary_keys)
        upd_map = {t: s for t, s in merge.update_columns}
        upd_set = set(upd_map.keys())

        # Build source expression map from INSERT VALUES (position-paired with INSERT columns)
        insert_src_map: Dict[str, str] = {}
        if merge.insert_columns and merge.insert_values:
            for tc, sv in zip(merge.insert_columns, merge.insert_values):
                insert_src_map[tc] = sv.strip()

        # All known target cols (from INSERT + UPDATE)
        all_known = set(merge.insert_columns) | upd_set | pk_set

        # Track defaulted columns for header
        defaulted_cols: List[Tuple[str, str, str]] = []  # (name, type, default_or_NULL)

        # Build parts
        parts = []
        # Header placeholder - will be prepended at the end
        # Config
        parts.append(self._gen_config(merge))
        parts.append('')
        # CTEs
        if merge.ctes:
            parts.append('WITH')
            ctes_out = []
            for name, body in merge.ctes:
                transformed = self._transform_cte_body(body, merge)
                ctes_out.append(f"{name} AS (\n{self._indent(transformed, '  ')}\n)")
            parts.append(',\n\n'.join(ctes_out))
        parts.append('')

        # SELECT
        sel_lines = []
        num_cols = len(output_cols)
        for idx, col in enumerate(output_cols):
            is_last = (idx == num_cols - 1)
            comma = '' if is_last else ','
            col_type = desc_map[col][0] if col in desc_map else None
            default = desc_map[col][1] if col in desc_map else ''

            is_known = col in all_known

            if not is_known:
                # Column in DESC_TABLE but not in MERGE at all -> default
                if default and default.strip():
                    sel_lines.append(f'    {default.strip()} AS {col}{comma}  -- not in original MERGE, defaulted from DESC_TABLE - PLEASE VERIFY')
                    defaulted_cols.append((col, col_type or '?', default.strip()))
                elif col_type:
                    sel_lines.append(f'    CAST(NULL AS {col_type}) AS {col}{comma}  -- not in original MERGE, defaulted to NULL - PLEASE VERIFY')
                    defaulted_cols.append((col, col_type, 'CAST(NULL AS ' + col_type + ')'))
                else:
                    sel_lines.append(f'    NULL AS {col}{comma}  -- not in original MERGE, defaulted - PLEASE VERIFY')
                    defaulted_cols.append((col, '?', 'NULL'))
                continue

            # Get source expression
            if col in upd_map:
                src_raw = upd_map[col]
            elif col in insert_src_map:
                src_raw = insert_src_map[col]
            else:
                src_raw = f'src.{col}'

            src_expr = self._normalize_src_expr(src_raw, 'SRC.')

            # Build value expression
            if col in pk_set:
                value_expr = src_expr
            elif col in upd_set:
                value_expr = f'COALESCE({src_expr}, TGT.{col})'
            else:
                value_expr = f'COALESCE(TGT.{col}, {src_expr})'

            # CAST wrap if DESC_TABLE provided
            if has_desc and col_type:
                wrapped = self._cast_wrap(value_expr, col_type)
                sel_lines.append(f'    {wrapped} AS {col}{comma}')
            else:
                sel_lines.append(f'    {value_expr} AS {col}{comma}')

        # DECODE
        active_dec, commented_dec = [], []
        for tc, src_raw in merge.update_columns:
            if tc in pk_set:
                src_n = self._normalize_src_expr(src_raw, 'src.')
                commented_dec.append((tc, src_n, 'PRIMARY KEY, EXCLUDED'))
                continue
            if tc in self.audit_columns or any(p in tc for p in self._audit_patterns):
                src_n = self._normalize_src_expr(src_raw, 'src.')
                commented_dec.append((tc, src_n, 'AUDIT COLUMN, MAY NOT BE NEEDED'))
                continue
            src_n = self._normalize_src_expr(src_raw, 'src.')
            active_dec.append((tc, src_n))

        dec_lines = []
        for idx, (tc, src_expr) in enumerate(active_dec):
            is_last = (idx == len(active_dec) - 1)
            or_part = '' if is_last else ' OR'
            col_type = desc_map[tc][0] if tc in desc_map else None
            if has_desc and col_type:
                src_w = self._cast_wrap(src_expr, col_type)
                tgt_w = self._cast_wrap(f'tgt.{tc}', col_type)
                dec_lines.append(f'   DECODE({tgt_w}, {src_w}, 1, 0) = 0{or_part}')
            else:
                dec_lines.append(f'   DECODE(tgt.{tc}, {src_expr}, 1, 0) = 0{or_part}')

        for idx, (tc, src_expr, reason) in enumerate(commented_dec):
            is_last = (idx == len(commented_dec) - 1)
            or_part = '' if is_last else ' OR'
            dec_lines.append(f'   --DECODE(tgt.{tc}, {src_expr}, 1, 0) = 0{or_part}  -- {reason}')

        where = ''
        if dec_lines:
            where = "WHERE\n  (\n" + '\n'.join(dec_lines) + "\n  )"

        # Build JOIN ON (multi-key)
        pks = merge.primary_keys
        if pks:
            on_parts = [f'SRC.{k} = TGT.{k}' for k in pks]
            on_clause = '\n    AND '.join(on_parts)
        else:
            on_clause = '/* TODO: no primary key detected */'

        parts.extend(['SELECT', '\n'.join(sel_lines), '',
                      'FROM', f'    {merge.source_expression} SRC',
                      'LEFT JOIN', '    {{ this }} TGT',
                      'ON', f'    {on_clause}'])
        if where:
            parts.append(where)

        # Generate header now (after we know defaulted_cols)
        header = self._gen_header(merge, has_desc, output_cols, all_known, desc_map, defaulted_cols)
        return header + '\n\n' + '\n'.join(parts)

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _gen_header(self, merge, has_desc, output_cols, all_known, desc_map, defaulted_cols=None) -> str:
        secs = []
        secs.append(f"Source: Snowflake MERGE -> dbt incremental model")
        secs.append(f"Target: {merge.target_table}")
        secs.append(f"Primary Key(s): {', '.join(merge.primary_keys) if merge.primary_keys else '(none)'}")
        secs.append(f"Source CTE: {merge.source_expression}")
        secs.append(f"CTE count: {len(merge.ctes)}")
        secs.append(f"Update columns: {len(merge.update_columns)}")
        secs.append(f"Insert columns: {len(merge.insert_columns)}")
        if self.params.autosys_job_name:
            secs.append(f"AUTOSYS_JOB_NAME: {self.params.autosys_job_name}")
        else:
            secs.append("TODO: $AUTOSYS_JOB_NAME not provided")

        # Audit columns that were commented out in DECODE
        audit_in_update = []
        for tc, _ in merge.update_columns:
            if tc in self.audit_columns or any(p in tc for p in self._audit_patterns):
                audit_in_update.append(tc)
        if audit_in_update:
            secs.append("")
            secs.append("AUDIT COLUMNS (commented out in DECODE - please verify):")
            for c in audit_in_update:
                secs.append(f"  - {c}")

        # Defaulted columns (not present in original MERGE)
        if defaulted_cols:
            secs.append("")
            secs.append("DEFAULTED COLUMNS (not in original MERGE statement, defaulted in SELECT - PLEASE VERIFY):")
            for name, col_type, default_val in defaulted_cols:
                secs.append(f"  - {name} ({col_type}) -> {default_val}")

        if self.unresolved_tables:
            secs.append("")
            secs.append("UNRESOLVED TABLES:")
            for t in self.unresolved_tables:
                secs.append(f"  - {t}")
        if self.warnings:
            secs.append("")
            secs.append("WARNINGS:")
            for w in self.warnings:
                secs.append(f"  - {w}")
        return '\n'.join(
            ['-- ' + '=' * 60, '-- MERGE SQL TO DBT CONVERSION SUMMARY', '-- ' + '=' * 60] +
            [f'-- {s}' for s in secs] +
            ['-- ' + '=' * 60]
        )

    def _gen_config(self, merge) -> str:
        autosys = self.params.autosys_job_name or '/* TODO: set $AUTOSYS_JOB_NAME */'
        pks = merge.primary_keys
        if len(pks) == 1:
            unique_key = f"'{pks[0]}'"
        elif len(pks) > 1:
            unique_key = '[' + ', '.join(f"'{k}'" for k in pks) + ']'
        else:
            unique_key = "'TODO_PRIMARY_KEY'"
        pre = [f"log_model_start(this, '{autosys}')"]
        for k in sorted(self.params.pre_hooks):
            pre.append(f'"{self._transform_hook(self.params.pre_hooks[k], merge)}"')
        post = []
        for k in sorted(self.params.post_hooks):
            post.append(f'"{self._transform_hook(self.params.post_hooks[k], merge)}"')
        post.append(f"log_model_end(this, '{autosys}')")
        lines = ['{{ config(']
        lines.append("    materialized='incremental',")
        lines.append(f"    unique_key={unique_key},")
        lines.append("    incremental_strategy='merge',")
        lines.append("    full_refresh=false,")
        lines.append("    pre_hook=[")
        lines.append('        ' + ',\n        '.join(pre))
        lines.append("    ],")
        lines.append("    post_hook=[")
        lines.append('        ' + ',\n        '.join(post))
        lines.append("    ]")
        lines.append(") }}")
        return '\n'.join(lines)

    def _transform_cte_body(self, body: str, merge: MergeStatement) -> str:
        cte_names = {n.upper() for n, _ in merge.ctes}
        def replace_table(match):
            keyword, table_name = match.group(1), match.group(2)
            alias = match.group(3) if match.group(3) else ''
            if table_name.upper() in cte_names:
                return match.group(0)
            # Try target table -> {{this}}
            tgt_parts = merge.target_table.split('.')
            tgt_short = tgt_parts[-1] if tgt_parts else merge.target_table
            if table_name.lower() == merge.target_table.lower() or table_name.lower() == tgt_short.lower():
                return f"{keyword} {{{{this}}}}{(' ' + alias) if alias else ''}  --{table_name}"
            # Try sources.yml (supports fully-qualified db.schema.table)
            resolved, _ = self.sources.resolve(table_name)
            if resolved:
                return f"{keyword} {resolved}{(' ' + alias) if alias else ''}  --{table_name}"
            # Common dbt prefixes -> ref()
            tbl_upper = table_name.split('.')[-1].upper()
            if any(tbl_upper.startswith(p) for p in ('STG_', 'DM_', 'FACT_', 'DIM_', 'ODS_')):
                return f"{keyword} {{{{ref('{table_name.split('.')[-1]}')}}}}{(' ' + alias) if alias else ''}  --{table_name}"
            if table_name not in self.unresolved_tables:
                self.unresolved_tables.append(table_name)
            return match.group(0)
        pattern = re.compile(r'\b(FROM|JOIN)\s+(\w+(?:\.\w+){0,2})(?:\s+(\w+))?', re.I)
        return pattern.sub(replace_table, body)

    def _normalize_src_expr(self, expr: str, prefix: str = 'SRC.') -> str:
        expr = expr.strip().rstrip(',').strip()
        m = re.match(r'^src\.(\w+)$', expr, re.I)
        if m:
            return f'{prefix}{m.group(1)}'
        if re.search(r'\bsrc\.', expr, re.I):
            return re.sub(r'\bsrc\.', prefix, expr, flags=re.I)
        if re.match(r'^\w+$', expr):
            return f'{prefix}{expr}'
        return expr

    def _cast_wrap(self, expr: str, col_type: str) -> str:
        if not col_type:
            return expr
        if self.skip_numeric:
            is_numeric = any(col_type.upper().startswith(t) for t in ('NUMBER', 'NUMERIC', 'DECIMAL', 'FLOAT', 'DOUBLE', 'INT', 'BIGINT', 'SMALLINT', 'TINYINT'))
            if is_numeric:
                return expr
        inner = expr
        if self.use_trim:
            inner = f"TRIM({inner})"
        varchar_match = re.match(r'VARCHAR\s*\(\s*(\d+)\s*\)', col_type, re.I)
        if varchar_match:
            inner = f"LEFT({inner}, {varchar_match.group(1)})"
        return f"CAST({inner} AS {col_type})"

    def _transform_hook(self, sql: str, merge: MergeStatement) -> str:
        if merge.target_table:
            sql = re.compile(re.escape(merge.target_table), re.I).sub('{{this}}', sql)
        return sql

    @staticmethod
    def _indent(text: str, prefix: str) -> str:
        return '\n'.join(prefix + line if line.strip() else line for line in text.split('\n'))


# =============================================================================
# VALIDATION
# =============================================================================

class Validator:
    @staticmethod
    def validate(p: ParsedParams) -> List[str]:
        errors = []
        if not p.model_type: errors.append("$TYPE is required")
        if p.is_stg:
            if not p.source_code: errors.append("$SOURCE_CODE is required for STG type")
            if not p.desc_table: errors.append("$DESC_TABLE is required for STG type")
            return errors
        if p.is_merge_type:
            # MERGE type - only $TYPE and $SOURCE_CODE mandatory
            # Everything else (target, PKs, update cols) auto-extracted from MERGE statement
            if not p.source_code: errors.append("$SOURCE_CODE is required for MERGE type")
            return errors
        # Non-STG/MERGE types keep existing mandatory validations
        if not p.target_table: errors.append("$TARGET_TABLE is required")
        if not p.source_code: errors.append("$SOURCE_CODE (or $INSERT_CODE) is required")
        if not p.primary_key: errors.append("$PRIMARY_KEY is required")
        if p.needs_merge and not p.split_cte:
            errors.append("$SPLIT_CTE required when $UPDATE_COLUMNS provided")
        return errors


# =============================================================================
# TEST CASE GENERATOR
# Generates test case SQL files for validating dbt models against BKP tables.
# Only for TYPE: MERGE and INSERT,UPDATE (INCREMENTAL).
# Requires $DESC_TABLE for column lists.
# =============================================================================

class TestCaseGenerator:
    """Generates test case SQL file for dbt model validation.
    Compares MAIN table (dbt target) vs MAIN_BKP (source of truth).
    Single DBT run after both DELETE + UPDATE tests.
    """

    def __init__(self, params, merge_stmt=None):
        self.params = params
        self.merge_stmt = merge_stmt
        self.errors = []

    def _derive_dbt_execute(self, db, schema, model_name):
        """Build Snowflake EXECUTE DBT PROJECT command."""
        mn_upper = model_name.upper()
        if mn_upper.startswith('STG_'):
            folder = 'STAGING'
        elif mn_upper.startswith('DM_'):
            folder = 'DIMENSION'
        else:
            parts = model_name.split('_')
            folder = parts[0] if parts else model_name
        ws = f'RePlatform_{db}_{schema}'
        pr = f'/dbt_{db}_{schema}'
        return (
            f"execute dbt project from workspace \"{db}\".\"{schema}\".\"{ws}\" "
            f"project_root='{pr}' "
            f"args='run --target dev "
            f"--select ''models/{schema}/{folder}/{model_name}.sql'' "
            f"--select {model_name}';"
        )

    def generate(self):
        if self.params.is_merge_type and self.merge_stmt:
            target_table = self.merge_stmt.target_table
            primary_keys = self.merge_stmt.primary_keys
            update_cols = [(t, s) for t, s in self.merge_stmt.update_columns]
        else:
            target_table = self.params.target_table
            pk_raw = self.params.primary_key
            primary_keys = [p.strip().upper() for p in pk_raw.split(',') if p.strip()]
            update_cols = list(self.params.update_columns)

        audit_set = {'LOAD_DT', 'LAST_UPDATE_LOAD_DT', 'EDW_LAST_UPDT_DTM',
                     'LAST_UPDT_TS', 'EDW_LAST_UPDT_TS', 'OUT_LAST_UPDT_TS'}
        audit_patterns = ['LAST_UPDT', 'LAST_UPDATE', 'EDW_LAST', 'LOAD_DT']

        bkp_table = target_table + '_BKP'
        desc = self.params.desc_table
        model_name = self.params.model_name or '/* TODO: set $MODEL_NAME */'

        if not desc:
            self.errors.append("$DESC_TABLE is required for test case generation")
            return ''
        if not primary_keys:
            self.errors.append("Primary key(s) required for test case generation")
            return ''

        # Derive DB and SCHEMA from target table
        tgt_parts = target_table.split('.')
        db = tgt_parts[0] if len(tgt_parts) >= 1 else 'DB'
        schema = tgt_parts[1] if len(tgt_parts) >= 2 else 'SCHEMA'

        # DBT execute command
        if self.params.model_name:
            dbt_cmd = self._derive_dbt_execute(db, schema, model_name)
        else:
            dbt_cmd = f"/* TODO: set $MODEL_NAME to generate dbt execute command */"

        pk_set = set(primary_keys)

        def is_audit(col_name):
            return col_name in audit_set or any(p in col_name for p in audit_patterns)

        all_cols = [(n, t) for n, t, _ in desc]
        non_audit_cols = [(n, t) for n, t in all_cols if not is_audit(n)]
        upd_tgt_cols = [(t, s) for t, s in update_cols if not is_audit(t) and t not in pk_set]

        pk_list = ', '.join(primary_keys)
        pk_list_main = ', '.join(f'MAIN.{k}' for k in primary_keys)
        pk_join = ' AND '.join(f'MAIN.{k} = BKP.{k}' for k in primary_keys)

        def select_cols_line(alias=''):
            parts = []
            for n, t in non_audit_cols:
                prefix = f'{alias}.' if alias else ''
                if any(t.upper().startswith(tp) for tp in ('NUMBER', 'NUMERIC', 'DECIMAL', 'FLOAT', 'DOUBLE', 'INT')):
                    parts.append(f'ROUND({prefix}{n}, 2) AS {n}')
                else:
                    parts.append(f'{prefix}{n}')
            return ', '.join(parts)

        audit_col = None
        for n, t, _ in desc:
            if is_audit(n) and 'TIMESTAMP' in t.upper():
                audit_col = n
                break
        if not audit_col:
            for n, t, _ in desc:
                if is_audit(n):
                    audit_col = n
                    break

        L = []
        L.append('-- ' + '=' * 70)
        L.append(f'-- TEST CASES FOR: {target_table}')
        L.append(f'-- BKP TABLE: {bkp_table}')
        L.append(f'-- PRIMARY KEY(S): {pk_list}')
        L.append(f'-- MODEL NAME: {model_name}')
        L.append('-- GENERATED BY: sql_to_dbt_converter -test_cases')
        L.append('-- ' + '=' * 70)
        L.append('')

        # TC0: Audit column range
        L.append('--#TEST_CASE0#--')
        L.append('--#VERIFY AUDIT COLUMN RANGE - CONFIRM TABLES ARE DIFFERENT#--')
        if audit_col:
            L.append(f'SELECT MIN({audit_col}) AS MIN_AUDIT, MAX({audit_col}) AS MAX_AUDIT FROM {target_table};')
            L.append(f'SELECT MIN({audit_col}) AS MIN_AUDIT, MAX({audit_col}) AS MAX_AUDIT FROM {bkp_table};')
        else:
            L.append('-- WARNING: No audit timestamp column detected in DESC_TABLE')
        L.append('')

        # TC1: Counts
        L.append('--#TEST_CASE1#--')
        L.append('--#COMPARE RECORD COUNTS BETWEEN MAIN AND BKP#--')
        L.append(f'SELECT COUNT(*) AS MAIN_COUNT FROM {target_table};')
        L.append(f'SELECT COUNT(*) AS BKP_COUNT FROM {bkp_table};')
        L.append('')

        # TC2: MAIN MINUS BKP
        L.append('--#TEST_CASE2#--')
        L.append('--#MAIN MINUS BKP - CHECK ALL RECORDS IN MAIN EXIST IN BKP#--')
        L.append(f'SELECT COUNT(*) AS MAIN_MINUS_BKP_COUNT FROM (SELECT {select_cols_line()} FROM {target_table} MINUS SELECT {select_cols_line()} FROM {bkp_table});')
        L.append(f'SELECT * FROM (SELECT {select_cols_line()} FROM {target_table} MINUS SELECT {select_cols_line()} FROM {bkp_table}) LIMIT 10;')
        L.append('')

        # TC3: BKP MINUS MAIN
        L.append('--#TEST_CASE3#--')
        L.append('--#BKP MINUS MAIN - CHECK ALL RECORDS IN BKP EXIST IN MAIN#--')
        L.append(f'SELECT COUNT(*) AS BKP_MINUS_MAIN_COUNT FROM (SELECT {select_cols_line()} FROM {bkp_table} MINUS SELECT {select_cols_line()} FROM {target_table});')
        L.append(f'SELECT * FROM (SELECT {select_cols_line()} FROM {bkp_table} MINUS SELECT {select_cols_line()} FROM {target_table}) LIMIT 10;')
        L.append('')

        # TC4: DECODE column-by-column
        L.append('--#TEST_CASE4#--')
        L.append('--#DECODE COLUMN-BY-COLUMN COMPARISON (EXCLUDING AUDIT AND PK COLUMNS)#--')
        decode_parts = [f"DECODE(MAIN.{n}, BKP.{n}, 'PASS', 'FAIL') AS {n}" for n, t in non_audit_cols if n not in pk_set]
        L.append(f'SELECT DISTINCT {", ".join(decode_parts)} FROM {target_table} MAIN FULL OUTER JOIN {bkp_table} BKP ON {pk_join};')
        L.append('')

        # TC5: DELETE test (part 1 - delete records)
        L.append('--#TEST_CASE5#--')
        L.append('--#INCREMENTAL LOAD TEST - DELETE RECORDS, UPDATE A RECORD, THEN VERIFY AFTER SINGLE DBT RUN#--')
        L.append('')
        L.append('--#STEP1: GET PRIMARY KEYS TO DELETE (RUN THIS FIRST)#--')
        L.append(f'/* TODO: paste PK values from the query below */')
        L.append(f'SELECT {pk_list} FROM {target_table} LIMIT 10;')
        L.append('')
        L.append('--#STEP2: DELETE 10 RECORDS FROM MAIN TABLE#--')
        L.append(f'/* TODO: paste the PK values from STEP1 into the IN clause below */')
        if len(primary_keys) == 1:
            L.append(f'DELETE FROM {target_table} WHERE {primary_keys[0]} IN (/* paste PKs here */);')
        else:
            L.append(f'DELETE FROM {target_table} WHERE ({pk_list}) IN (/* paste PK tuples here */);')
        L.append('')
        L.append('--#STEP3: VERIFY RECORDS ARE DELETED#--')
        L.append(f'/* Expected: 0 rows */')
        if len(primary_keys) == 1:
            L.append(f'SELECT {pk_list} FROM {target_table} WHERE {primary_keys[0]} IN (/* paste deleted PKs from STEP2 */);')
        else:
            L.append(f'SELECT {pk_list} FROM {target_table} WHERE ({pk_list}) IN (/* paste deleted PK tuples from STEP2 */);')
        L.append('')

        # TC6: UPDATE test (part 1 - update a record)
        L.append('--#TEST_CASE6#--')
        L.append('--#UPDATE TEST - MODIFY A RECORD BEFORE DBT RUN#--')
        L.append('')
        L.append('--#STEP1: GET A PRIMARY KEY VALUE TO UPDATE (RUN THIS FIRST)#--')
        L.append(f'/* TODO: paste 1 PK value from the query below */')
        L.append(f'SELECT {pk_list} FROM {target_table} LIMIT 1;')
        L.append('')
        if upd_tgt_cols:
            set_parts = []
            for tgt_col, _ in upd_tgt_cols:
                col_type = None
                for n, t, _ in desc:
                    if n == tgt_col:
                        col_type = t
                        break
                if col_type and any(col_type.upper().startswith(t) for t in ('VARCHAR', 'CHAR', 'STRING', 'TEXT')):
                    set_parts.append(f"{tgt_col} = 'TEST_{tgt_col}'")
                elif col_type and any(col_type.upper().startswith(t) for t in ('NUMBER', 'NUMERIC', 'DECIMAL', 'FLOAT', 'DOUBLE', 'INT')):
                    set_parts.append(f"{tgt_col} = 0")
                elif col_type and 'TIMESTAMP' in col_type.upper():
                    set_parts.append(f"{tgt_col} = '1900-01-01'")
                elif col_type and 'DATE' in col_type.upper():
                    set_parts.append(f"{tgt_col} = '1900-01-01'")
                else:
                    set_parts.append(f"{tgt_col} = NULL")
            if len(primary_keys) == 1:
                pk_where = f'{primary_keys[0]} = /* PK_VALUE */'
            else:
                pk_where = ' AND '.join(f'{k} = /* {k}_VALUE */' for k in primary_keys)
            L.append('--#STEP2: UPDATE THE RECORD IN MAIN TABLE#--')
            L.append(f'/* TODO: replace PK_VALUE with the actual value from TC6 STEP1 */')
            L.append(f'UPDATE {target_table} SET {", ".join(set_parts)} WHERE {pk_where};')
            L.append('')
            upd_col_list = ', '.join(t for t, _ in upd_tgt_cols)
            L.append('--#STEP3: VERIFY THE UPDATE WAS APPLIED#--')
            L.append(f'/* Expected: updated values should show TEST_xxx for VARCHAR, 0 for NUMBER */')
            L.append(f'SELECT {pk_list}, {upd_col_list} FROM {target_table} WHERE {pk_where};')
            L.append('')
        else:
            L.append('-- No update columns defined - skipping UPDATE test steps')
            L.append('')

        # TC7: SINGLE DBT RUN (after both delete + update)
        L.append('--#TEST_CASE7#--')
        L.append('--#RUN DBT MODEL (SINGLE RUN TO VERIFY BOTH DELETE RE-INSERT AND UPDATE REVERT)#--')
        L.append('')
        L.append('--#STEP1: RUN THE DBT MODEL#--')
        L.append(f'/* {dbt_cmd} */')
        L.append('')

        # TC8: Verify everything after DBT run
        L.append('--#TEST_CASE8#--')
        L.append('--#VERIFY ALL CHANGES AFTER DBT RUN#--')
        L.append('')

        # Verify deleted records are back
        L.append('--#STEP1: VERIFY DELETED RECORDS ARE RE-INSERTED#--')
        L.append(f'/* Expected: 10 rows (records should be back) */')
        if len(primary_keys) == 1:
            L.append(f'SELECT {pk_list} FROM {target_table} WHERE {primary_keys[0]} IN (/* paste same PKs from TC5 */);')
        else:
            L.append(f'SELECT {pk_list} FROM {target_table} WHERE ({pk_list}) IN (/* paste same PK tuples from TC5 */);')
        L.append('')

        # Verify update was reverted
        if upd_tgt_cols:
            if len(primary_keys) == 1:
                pk_where_main = f'MAIN.{primary_keys[0]} = /* PK_VALUE from TC6 */'
            else:
                pk_where_main = ' AND '.join(f'MAIN.{k} = /* {k}_VALUE from TC6 */' for k in primary_keys)
            L.append('--#STEP2: VERIFY UPDATE WAS REVERTED (DECODE CHECK)#--')
            decode_upd = [f"DECODE(MAIN.{t}, BKP.{t}, 'PASS', 'FAIL') AS {t}" for t, _ in upd_tgt_cols]
            L.append(f'SELECT {pk_list_main}, {", ".join(decode_upd)} FROM {target_table} MAIN FULL OUTER JOIN {bkp_table} BKP ON {pk_join} WHERE {pk_where_main};')
            L.append('')

        # Count match
        L.append('--#STEP3: VERIFY COUNTS MATCH#--')
        L.append(f"WITH counts AS (SELECT (SELECT COUNT(*) FROM {target_table}) AS main_cnt, (SELECT COUNT(*) FROM {bkp_table}) AS bkp_cnt) SELECT main_cnt, bkp_cnt, CASE WHEN main_cnt = bkp_cnt THEN 'PASS' ELSE 'FAIL' END AS result FROM counts;")
        L.append('')

        # Final MINUS both ways
        L.append('--#STEP4: FINAL MINUS CHECK (BOTH WAYS)#--')
        L.append(f'SELECT COUNT(*) AS MAIN_MINUS_BKP_COUNT FROM (SELECT {select_cols_line()} FROM {target_table} MINUS SELECT {select_cols_line()} FROM {bkp_table});')
        L.append(f'SELECT COUNT(*) AS BKP_MINUS_MAIN_COUNT FROM (SELECT {select_cols_line()} FROM {bkp_table} MINUS SELECT {select_cols_line()} FROM {target_table});')
        L.append('')

        L.append('-- ' + '=' * 70)
        L.append('-- END OF TEST CASES')
        L.append('-- ' + '=' * 70)

        # Append $DESC_TABLE as plain text (not commented) for downstream tool
        L.append('$DESC_TABLE:')
        if self.params.desc_table_raw:
            # Add header if not already present
            raw_stripped = self.params.desc_table_raw.strip()
            if not raw_stripped.lower().startswith('name'):
                L.append('name\ttype\tkind\tnull?\tdefault')
            for line in raw_stripped.split('\n'):
                L.append(line.rstrip())
        else:
            L.append('name\ttype\tkind\tnull?\tdefault')
            for name, typ, dft in desc:
                L.append(f'{name}\t{typ}\tCOLUMN\tY\t{dft}')

        return '\n'.join(L)


class SqlToDbtConverter:
    def __init__(self, inp, yml, out=None, clean_comments=False, use_trim=False, gen_test_cases=False, skip_numeric=False):
        self.inp, self.yml = inp, yml
        self.out = out or (os.path.splitext(inp)[0] + '_dbt.sql')
        self.clean_comments = clean_comments
        self.use_trim = use_trim
        self.gen_test_cases = gen_test_cases
        self.skip_numeric = skip_numeric
    def convert(self):
        with open(self.inp, encoding='utf-8') as f: content = f.read()
        # Optional: strip commented lines from source code before parsing
        if self.clean_comments:
            content = self._strip_comments_from_source(content)
        sources = SourcesRegistry(self.yml)
        params = ParameterParser.parse(content)
        errs = Validator.validate(params)
        if errs: return ('', [], errs)

        # Route to appropriate transformer based on type
        merge_stmt = None
        if params.is_merge_type:
            try:
                merge_stmt = MergeStatementParser.parse(params.source_code)
            except Exception as e:
                return ('', [], [f"MERGE parse error: {e}"])
            t = MergeTransformer(params, sources, use_trim=self.use_trim, skip_numeric=self.skip_numeric)
        else:
            t = SQLTransformer(params, sources, use_trim=self.use_trim, skip_numeric=self.skip_numeric)
        result = t.transform()
        os.makedirs(os.path.dirname(self.out) or '.', exist_ok=True)
        with open(self.out, 'w', encoding='utf-8') as f: f.write(result)

        # Generate test cases if requested
        if self.gen_test_cases:
            if params.is_merge_type or params.core_type == 'INCREMENTAL':
                tc_gen = TestCaseGenerator(params, merge_stmt)
                tc_content = tc_gen.generate()
                if tc_gen.errors:
                    for e in tc_gen.errors:
                        t.warnings.append(f"Test case: {e}")
                if tc_content:
                    # Output file: same name with _TEST_CASE before extension
                    base, ext = os.path.splitext(self.out)
                    tc_out = base + '_TEST_CASE' + ext
                    with open(tc_out, 'w', encoding='utf-8') as f:
                        f.write(tc_content)
                    print(f"    TEST CASES: {os.path.basename(tc_out)}")
            else:
                t.warnings.append(f"Test cases only supported for TYPE: MERGE and INSERT,UPDATE (current: {params.core_type})")

        return (result, t.warnings, [])

    @staticmethod
    def _strip_comments_from_source(content: str) -> str:
        """Remove lines starting with -- from $SOURCE_CODE, $INSERT_CODE, and hook sections."""
        blocks = ParameterParser._extract_blocks(content)
        for name, raw in blocks.items():
            name_upper = name.upper()
            # Strip from SOURCE_CODE, INSERT_CODE, and all hooks
            if name_upper in ('SOURCE_CODE', 'INSERT_CODE') or \
               name_upper.startswith('PRE_HOOK') or name_upper.startswith('POST_HOOK'):
                lines = raw.split('\n')
                cleaned = [l for l in lines if not l.lstrip().startswith('--')]
                cleaned_raw = '\n'.join(cleaned)
                if raw != cleaned_raw:
                    content = content.replace(raw, cleaned_raw)
        return content

def convert_single(inp, yml, out=None, clean_comments=False, use_trim=False, gen_test_cases=False, skip_numeric=False):
    if not os.path.exists(inp):
        print(f"  ERROR: not found: {inp}"); return False
    c = SqlToDbtConverter(inp, yml, out, clean_comments, use_trim, gen_test_cases, skip_numeric)
    _, warns, errs = c.convert()
    if errs:
        print(f"  FAILED: {os.path.basename(inp)}")
        for e in errs: print(f"    - {e}")
        return False
    for w in warns: print(f"    WARN: {w}")
    print(f"  OK: {os.path.basename(inp)} -> {c.out}")
    return True

def main():
    ap = argparse.ArgumentParser(description='Snowflake SQL to dbt converter (with STG/MERGE type support + test cases)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python sql_to_dbt_converter.py input.sql sources.yml -o output.sql\n"
               "  python sql_to_dbt_converter.py input.sql sources.yml -o output.sql -cc\n"
               "  python sql_to_dbt_converter.py input.sql sources.yml -o output.sql -trim\n"
               "  python sql_to_dbt_converter.py input.sql sources.yml -o output.sql -test_cases\n"
               "  python sql_to_dbt_converter.py --batch input/ sources.yml -d output/ --report")
    mg = ap.add_mutually_exclusive_group(required=True)
    mg.add_argument('input_sql', nargs='?', default=None)
    mg.add_argument('--batch', '-b', metavar='DIR')
    ap.add_argument('sources_yml')
    ap.add_argument('--output', '-o')
    ap.add_argument('--output-dir', '-d', default='output')
    ap.add_argument('--report', '-r', action='store_true')
    ap.add_argument('-cc', '--clean-comments', action='store_true',
                    help='Remove -- commented lines from source code (makes output leaner)')
    ap.add_argument('-trim', '--trim', action='store_true',
                    help='Wrap source columns with TRIM() before CAST in final SELECT')
    ap.add_argument('-test_cases', '--test-cases', action='store_true',
                    help='Generate test case SQL file (_TEST_CASE.sql) for MERGE and INSERT,UPDATE types')
    ap.add_argument('-non_num', '--non-num', action='store_true',
                    help='Skip CAST for NUMBER/INT types - only cast VARCHAR columns (useful when IDs may change type)')
    a = ap.parse_args()
    if not os.path.exists(a.sources_yml):
        print(f"Error: {a.sources_yml} not found"); sys.exit(1)
    cc = a.clean_comments
    tr = a.trim
    tc = a.test_cases
    nn = a.non_num
    if a.input_sql:
        sys.exit(0 if convert_single(a.input_sql, a.sources_yml, a.output, cc, tr, tc, nn) else 1)
    d, od = a.batch, a.output_dir
    if not os.path.isdir(d): print(f"Error: {d} not found"); sys.exit(1)
    os.makedirs(od, exist_ok=True)
    fs = sorted(f for f in os.listdir(d) if f.lower().endswith('.sql'))
    if not fs: print(f"No .sql files in {d}"); sys.exit(1)
    print(f"Converting {len(fs)} file(s): {d}/ -> {od}/")
    print('-' * 60)
    ok, fail, res = 0, 0, []
    for f in fs:
        try:
            s = convert_single(os.path.join(d, f), a.sources_yml, os.path.join(od, f), cc, tr, tc, nn)
            if s: ok += 1; res.append((f, 'OK', ''))
            else: fail += 1; res.append((f, 'FAILED', 'validation'))
        except Exception as e:
            fail += 1; print(f"  ERROR: {f} -> {e}"); res.append((f, 'ERROR', str(e)))
    print('-' * 60)
    print(f"Done: {ok} passed, {fail} failed, {len(fs)} total")
    if a.report:
        print(f"\n{'='*60}\n{'File':<40} Status\n{'-'*60}")
        for f, s, dd in res: print(f"{f:<40} {s}" + (f" ({dd})" if dd else ''))
        print('='*60)
    sys.exit(1 if fail else 0)

if __name__ == '__main__': main()
