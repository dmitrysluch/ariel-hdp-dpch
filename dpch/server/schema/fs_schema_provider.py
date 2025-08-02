from pydantic import ValidationError

from dpch.common.schema import Schema
from dpch.server.schema.interface import SchemaProviderMixin, SchemaValueError


# Schema is a pydantic model. Please implement the following mixin for importing from json config.
# The constructor receives a single argument file
class FSSchemaProvider(SchemaProviderMixin):
    def __init__(self, file: str):
        self.file = file

    async def get_schema(self) -> Schema:
        try:
            with open(self.file, "r") as f:
                schema = f.read()
        except FileNotFoundError:
            raise SchemaValueError("Schema file not found.")
        try:
            return Schema.model_validate_json(schema)
        except ValidationError as e:
            raise SchemaValueError("Failed validating schema") from e
