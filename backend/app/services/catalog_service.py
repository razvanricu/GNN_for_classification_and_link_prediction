import logging
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import duckdb

from app.schemas.catalog import CatalogQuery

logger = logging.getLogger(__name__)


class CatalogService:
    """DuckDB-backed catalog queries for fast product browsing."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._local = threading.local()
        path = Path(db_path)
        if path.exists():
            self._valid = True
            logger.info(f"Catalog DB found: {path}")
        else:
            self._valid = False
            logger.warning(f"Catalog DB not found: {path} — catalog endpoints will fail")

    @property
    def _conn(self):
        if not self._valid:
            return None
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = duckdb.connect(str(self._db_path), read_only=True)
            self._local.conn = conn
        return conn

    def search(self, query: CatalogQuery) -> Tuple[List[dict], int]:
        """Search and paginate products."""
        if not self._conn:
            return [], 0

        conditions = []
        params = []

        if query.q:
            conditions.append("title ILIKE ?")
            params.append(f"%{query.q}%")
        if query.category:
            conditions.append("target_class = ?")
            params.append(query.category)
        if query.min_price is not None:
            conditions.append("price >= ?")
            params.append(query.min_price)
        if query.max_price is not None:
            conditions.append("price <= ?")
            params.append(query.max_price)
        if query.in_graph is not None:
            conditions.append("in_graph = ?")
            params.append(query.in_graph)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Count
        count_sql = f"SELECT COUNT(*) FROM products WHERE {where_clause}"
        total = self._conn.execute(count_sql, params).fetchone()[0]

        # Fetch page
        offset = (query.page - 1) * query.size
        data_sql = (
            f"SELECT asin, title, price, imUrl, target_class, in_graph, graph_idx "
            f"FROM products WHERE {where_clause} "
            f"ORDER BY asin LIMIT ? OFFSET ?"
        )
        rows = self._conn.execute(data_sql, params + [query.size, offset]).fetchall()

        items = [
            {
                "asin": r[0],
                "title": r[1],
                "price": r[2],
                "imUrl": r[3],
                "target_class": r[4],
                "in_graph": r[5],
                "graph_idx": r[6],
            }
            for r in rows
        ]
        return items, total

    def get_by_asin(self, asin: str) -> Optional[dict]:
        """Get full product detail by ASIN."""
        if not self._conn:
            return None

        sql = "SELECT * FROM products WHERE asin = ?"
        result = self._conn.execute(sql, [asin]).fetchone()
        if not result:
            return None

        columns = [desc[0] for desc in self._conn.description]
        return dict(zip(columns, result))

    def get_categories(self) -> List[str]:
        """Get distinct class names."""
        if not self._conn:
            return []

        rows = self._conn.execute(
            "SELECT DISTINCT target_class FROM products ORDER BY target_class"
        ).fetchall()
        return [r[0] for r in rows]

    def get_stats(self) -> dict:
        """Get catalog-level statistics."""
        if not self._conn:
            return {}

        result = self._conn.execute("""
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE in_graph) as in_graph_count,
                AVG(price) FILTER (WHERE price IS NOT NULL) as avg_price
            FROM products
        """).fetchone()

        return {
            "total_products": result[0],
            "in_graph_count": result[1],
            "avg_price": round(result[2], 2) if result[2] else None,
        }

    def close(self):
        conn = getattr(self._local, "conn", None)
        if conn:
            conn.close()
            self._local.conn = None
