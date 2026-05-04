"""
Build catalog.duckdb from filtered_catalog.json.
Creates an indexed DuckDB database for fast product queries.

Usage:
    python scripts/build_catalog_db.py --input ../filtered_catalog.json --output catalog.duckdb
"""
import argparse
import json
import sys

import duckdb


def main():
    parser = argparse.ArgumentParser(description="Build catalog DuckDB from filtered_catalog.json")
    parser.add_argument("--input", default="../filtered_catalog.json", help="NDJSON input file")
    parser.add_argument("--output", default="catalog.duckdb", help="Output DuckDB path")
    args = parser.parse_args()

    print(f"Reading {args.input}...")
    conn = None
    try:
        conn = duckdb.connect(args.output)
        conn.execute("DROP TABLE IF EXISTS products")
        conn.execute("""
            CREATE TABLE products (
                asin VARCHAR PRIMARY KEY,
                title VARCHAR,
                description VARCHAR,
                price DOUBLE,
                imUrl VARCHAR,
                categories_json VARCHAR,
                also_bought_json VARCHAR,
                also_viewed_json VARCHAR,
                target_class VARCHAR,
                in_graph BOOLEAN,
                graph_idx INTEGER
            )
        """)

        # Insert data line by line (streaming)
        inserted = 0
        errors = 0
        with open(args.input, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    p = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  WARNING: Line {line_num} is not valid JSON: {e}", file=sys.stderr)
                    errors += 1
                    continue

                asin = p.get("asin")
                if not asin:
                    print(f"  WARNING: Line {line_num} missing 'asin' field", file=sys.stderr)
                    errors += 1
                    continue

                conn.execute(
                    """INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        asin,
                        p.get("title", ""),
                        p.get("description", ""),
                        p.get("price"),
                        p.get("imUrl"),
                        json.dumps(p.get("categories", [])),
                        json.dumps(p.get("also_bought", [])),
                        json.dumps(p.get("also_viewed", [])),
                        p.get("target_class", "Unknown"),
                        p.get("graph_idx") is not None,
                        p.get("graph_idx"),
                    ],
                )
                inserted += 1

        # Create indexes
        print("Creating indexes...")
        conn.execute("CREATE INDEX idx_target_class ON products(target_class)")
        conn.execute("CREATE INDEX idx_in_graph ON products(in_graph)")

        count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        print(f"Done! {count} products in {args.output}")
        if errors:
            print(f"  ({errors} lines skipped due to errors)")

    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    main()
