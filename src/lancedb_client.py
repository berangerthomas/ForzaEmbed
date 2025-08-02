import lancedb
import pyarrow as pa
from typing import List, Dict, Any

class LanceDBClient:
    def __init__(self, db_path: str = "data/lancedb"):
        self.db = lancedb.connect(db_path)

    def get_or_create_table(self, table_name: str, schema: pa.Schema):
        try:
            table = self.db.open_table(table_name)
        except FileNotFoundError:
            table = self.db.create_table(table_name, schema=schema)
        return table

    def add_embeddings(self, table_name: str, data: List[Dict[str, Any]]):
        if not data:
            return
        
        # Infer schema from the first item if not provided
        first_item = data[0]
        fields = []
        for key, value in first_item.items():
            if key == "vector":
                fields.append(pa.field(key, pa.list_(pa.float32())))
            elif isinstance(value, str):
                fields.append(pa.field(key, pa.string()))
            elif isinstance(value, int):
                fields.append(pa.field(key, pa.int64()))
            elif isinstance(value, float):
                fields.append(pa.field(key, pa.float64()))
        
        schema = pa.schema(fields)
        table = self.get_or_create_table(table_name, schema)
        table.add(data)

    def search(self, table_name: str, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        table = self.db.open_table(table_name)
        results = table.search(query_vector).limit(limit).to_df()
        return results.to_dict("records")

    def create_table_if_not_exists(self, table_name: str, schema: pa.Schema):
        if table_name not in self.db.table_names():
            self.db.create_table(table_name, schema=schema)
