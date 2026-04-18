# hackathon_slm

A FastAPI service that converts natural language questions into valid [Polars](https://pola.rs/) Python code using a fine-tuned SLM (Small Language Model).

## Model

`Sokheng/qwen2.5-coder-7b-polars`

## Setup

```bash
uv venv
uv sync
```

## Run

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API

### `POST /chat`

```json
{
  "message": "Find top 5 customers by revenue in the Seafood category",
  "tables": {
    "nw_order_details": { "columns": ["order_id", "product_id", "unit_price", "quantity", "discount"] },
    "nw_products": { "columns": ["product_id", "product_name", "category_id"] },
    "nw_categories": { "columns": ["category_id", "category_name"] },
    "nw_orders": { "columns": ["order_id", "customer_id"] },
    "nw_customers": { "columns": ["customer_id", "company_name"] }
  }
}
```

**Response:**

```json
{
  "response": "import polars as pl\nresult = ..."
}
```