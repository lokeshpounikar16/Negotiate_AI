"""PostgreSQL persistence placeholder for the negotiation workflow."""


class PostgresStorage:
    """Placeholder for future database-backed negotiation storage."""

    def __init__(self, connection_string: str | None = None):
        self.connection_string = connection_string

    async def connect(self):
        return None

    async def disconnect(self):
        return None
