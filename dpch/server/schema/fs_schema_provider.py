from dpch.common.schema import Schema, SchemaProviderMixin


# Schema is a pydantic model. Please implement the following mixin for importing from json config.
# The constructor receives a single argument file
class FSSchemaProvider(SchemaProviderMixin):
    def __init__(self, file: str):
        self.file = file

    async def get_schema(self) -> Schema:
        with open(self.file, "r") as f:
            schema = f.read()
        return Schema.model_validate_json(schema)
