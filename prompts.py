import polars as pl

SYSTEM_PROMPT = """\
You are a Polars (Python) expert. Output ONLY valid Python code. No explanation, no markdown fences.

STRICT RULES:
- import polars as pl must be the first line
- Use the exact variable names provided — never use "df" unless the table is named "df"
- Assign final answer to: result = ...
- NEVER use pandas, numpy, or any non-Polars library
- NEVER use groupby() — always group_by()
- NEVER use pl.count() — always pl.len()
- NEVER use .loc, .iloc, or index-based selection
- NEVER assign columns like df["col"] = ... — use .with_columns()
- Use the EXACT alias names mentioned in the question — do NOT invent aliases from column names

POLARS API (use exactly as shown):
# Filter & Select
  tbl.filter(pl.col("x") > 1)
  tbl.filter((pl.col("a") > 1) & (pl.col("b") == "x"))
  tbl.select("a", "b")
  tbl.select(pl.col("a"), pl.col("b").alias("renamed"))

# Mutate
  tbl.with_columns((pl.col("a") * 2).alias("b"))
  tbl.with_columns([expr1, expr2])

# Aggregate
  tbl.group_by("col").agg(pl.col("x").sum(), pl.len().alias("count"))
  tbl.group_by("col").agg(pl.col("x").mean().round(2).alias("avg_x"))
  tbl.group_by(["a", "b"]).agg(pl.col("x").n_unique().alias("distinct"))

# Sort & Limit
  tbl.sort("col", descending=True).head(5)
  tbl.sort(["a", "b"], descending=[True, False])

# Join
  tbl.join(tbl2, on="id", how="inner"|"left"|"anti")
  tbl.join(tbl2, left_on="a_id", right_on="b_id")
  # COLLISION: if both tables have "name", right becomes "name_right"
  # Fix: tbl2.rename({"name": "name2"}) before join, OR .select(pl.col("name_right").alias("name2"))

# Window functions (sort first if order matters)
  tbl.sort("date").with_columns(pl.col("x").cum_sum().over("group").alias("running"))
  tbl.with_columns(pl.col("x").rank(method="dense", descending=True).over("group").alias("rnk"))
  tbl.with_columns(pl.col("x").mean().over("group").alias("group_avg"))

# Conditional
  pl.when(cond).then(val).otherwise(other).alias("col")

# Dates (columns often stored as strings)
  pl.col("date").str.slice(0, 4)          # year as string "YYYY"
  pl.col("date").str.slice(0, 7)          # "YYYY-MM"
  pl.col("date").str.to_date()            # parse to Date
  pl.col("date").dt.year()                # only after str.to_date()

# Strings
  pl.col("s").str.contains("pat")
  pl.col("s").str.to_lowercase()
  pl.col("s").is_in(["a", "b"])

# Nulls & Types
  tbl.drop_nulls(subset=["col"])
  pl.col("x").fill_null(0)
  pl.col("x").is_null() / .is_not_null()
  pl.col("x").cast(pl.Int64) / .cast(pl.Float64) / .cast(pl.Utf8)

# Misc
  tbl.rename({"old": "new"})
  tbl.unique(subset=["col"])
  pl.concat([tbl1, tbl2], how="vertical")
  tbl.pivot(on="col", index="row", values="val", aggregate_function="sum")
"""


def _extract_datasets(tables: dict) -> dict:
    """
    Handles multiple payload shapes:
      1. Flat:   {"sk_films": {"columns": [...], "file_name": "..."}}
      2. Nested: {"prop": {"datasets": {"sk_films": {"file_name": "...", "format": "..."}}}}
    Returns {table_name: {"file_name": ..., "columns": [...]} or {}}
    """
    # Shape 2: any top-level value has a "datasets" key
    for v in tables.values():
        if isinstance(v, dict) and "datasets" in v:
            return v["datasets"]
    # Shape 1: top-level keys are table names
    return tables


def _get_columns(info: dict) -> list:
    """Try to read column names from the file if not already in info."""
    if isinstance(info, dict) and info.get("columns"):
        return info["columns"]

    file_name = info.get("file_name", "") if isinstance(info, dict) else ""
    if not file_name:
        return []
    try:
        fmt = (info.get("format", "") or "").lower() if isinstance(info, dict) else ""
        if fmt == "parquet" or file_name.endswith(".parquet"):
            return pl.read_parquet(file_name, n_rows=0).columns  # type: ignore[return-value]
        if fmt == "csv" or file_name.endswith(".csv"):
            return pl.read_csv(file_name, n_rows=0).columns  # type: ignore[return-value]
    except Exception:
        pass
    return []


def build_messages(question: str, tables: dict) -> list:
    datasets = _extract_datasets(tables)

    schema_lines = []
    table_names = []
    for name, info in datasets.items():
        table_names.append(name)
        cols = _get_columns(info)
        if cols:
            schema_lines.append(f"  {name}: columns={cols}")
        else:
            schema_lines.append(f"  {name}: (columns unknown)")

    schema_str = "\n".join(schema_lines)
    system = SYSTEM_PROMPT + f"\nTables (use these exact variable names):\n{schema_str}"


    messages = [{"role": "system", "content": system}]
    # messages.extend(few_shots)
    messages.append({"role": "user", "content": f"Tables: {table_names}\nQuestion: {question}"})
    return messages
