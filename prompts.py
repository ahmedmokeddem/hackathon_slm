import polars as pl

SYSTEM_PROMPT = """\
You are a Polars (Python) expert. Output ONLY valid Python code. No explanation, no markdown fences.
Every response MUST start with: import polars as pl

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
- NEVER use a column name that is not listed in the schema above
- If a column seems missing, use the closest available one from the schema

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
  # Fix: tbl2.rename({"name": "name2"}) before join

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

FEW_SHOTS = [
    {
        "role": "user",
        "content": "Tables: ['df']\nQuestion: Count rows per category."
    },
    {
        "role": "assistant",
        "content": "import polars as pl\nresult = df.group_by(\"category\").agg(pl.len().alias(\"count\"))"
    },
    {
        "role": "user",
        "content": "Tables: ['nw_orders', 'nw_customers']\nQuestion: Join orders with customers on customer_id."
    },
    {
        "role": "assistant",
        "content": "import polars as pl\nresult = nw_orders.join(nw_customers, on=\"customer_id\")"
    },
    {
        "role": "user",
        "content": "Tables: ['nw_customers', 'nw_orders']\nQuestion: Find customers who have never placed an order."
    },
    {
        "role": "assistant",
        "content": "import polars as pl\nresult = nw_customers.join(nw_orders, on=\"customer_id\", how=\"anti\")"
    },
    {
        "role": "user",
        "content": "Tables: ['sk_payments']\nQuestion: Select payment_id, customer_id, amount, and add amount_category: 'large' if above 5, 'medium' if above 2, else 'small'."
    },
    {
        "role": "assistant",
        "content": (
            "import polars as pl\n"
            "result = sk_payments.select(\n"
            "    \"payment_id\", \"customer_id\", \"amount\",\n"
            "    pl.when(pl.col(\"amount\") > 5).then(pl.lit(\"large\"))\n"
            "      .when(pl.col(\"amount\") > 2).then(pl.lit(\"medium\"))\n"
            "      .otherwise(pl.lit(\"small\")).alias(\"amount_category\")\n"
            ")"
        )
    },
    {
        "role": "user",
        "content": "Tables: ['tpch_lineitem']\nQuestion: TPC-H Q1: group by returnflag and linestatus, compute sum_qty, sum_price, avg_discount, count."
    },
    {
        "role": "assistant",
        "content": (
            "import polars as pl\n"
            "result = tpch_lineitem.group_by(\"l_returnflag\", \"l_linestatus\").agg(\n"
            "    pl.col(\"l_quantity\").sum().alias(\"sum_qty\"),\n"
            "    pl.col(\"l_extendedprice\").sum().alias(\"sum_price\"),\n"
            "    pl.col(\"l_discount\").mean().round(4).alias(\"avg_disc\"),\n"
            "    pl.len().alias(\"count_order\")\n"
            ").sort(\"l_returnflag\", \"l_linestatus\")"
        )
    },
    {
        "role": "user",
        "content": "Tables: ['nw_products', 'nw_categories']\nQuestion: Pivot: for each category, show count of discontinued vs active products as columns."
    },
    {
        "role": "assistant",
        "content": (
            "import polars as pl\n"
            "result = nw_products.join(nw_categories, on=\"category_id\").group_by(\n"
            "    \"category_name\", \"discontinued\"\n"
            ").agg(pl.len().alias(\"count\")).pivot(\n"
            "    on=\"discontinued\", index=\"category_name\", values=\"count\"\n"
            ").fill_null(0)"
        )
    },
    {
        "role": "user",
        "content": "Tables: ['nw_order_details', 'nw_orders', 'nw_customers']\nQuestion: Compute average order value per country."
    },
    {
        "role": "assistant",
        "content": (
            "import polars as pl\n"
            "result = nw_order_details.with_columns(\n"
            "    (pl.col(\"unit_price\") * pl.col(\"quantity\") * (1 - pl.col(\"discount\"))).alias(\"revenue\")\n"
            ").group_by(\"order_id\").agg(\n"
            "    pl.col(\"revenue\").sum().alias(\"order_value\")\n"
            ").join(nw_orders, on=\"order_id\").join(nw_customers, on=\"customer_id\").group_by(\"country\").agg(\n"
            "    pl.col(\"order_value\").mean().round(2).alias(\"avg_order_value\"),\n"
            "    pl.len().alias(\"order_count\")\n"
            ").sort(\"avg_order_value\", descending=True)"
        )
    },
]


def _extract_datasets(tables: dict) -> dict:
    for v in tables.values():
        if isinstance(v, dict) and "datasets" in v:
            return v["datasets"]
    return tables


def _get_columns(info: dict) -> list:
    if isinstance(info, dict) and info.get("columns"):
        return info["columns"]

    file_name = info.get("file_name", "") if isinstance(info, dict) else ""
    if not file_name:
        return []
    try:
        fmt = (info.get("format", "") or "").lower() if isinstance(info, dict) else ""
        if fmt == "parquet" or file_name.endswith(".parquet"):
            return pl.read_parquet(file_name, n_rows=0).columns
        if fmt == "csv" or file_name.endswith(".csv"):
            return pl.read_csv(file_name, n_rows=0).columns
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
    messages.extend(FEW_SHOTS)
    messages.append({"role": "user", "content": f"Tables: {table_names}\nQuestion: {question}"})
    return messages