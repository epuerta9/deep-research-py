from rich.console import Console


class ModelConfig:
    console = Console()
    service = "openai"
    model = "o3-mini"

    @classmethod
    def get_service(cls) -> str:
        return cls.service

    @classmethod
    def set_service(cls, new_service: str) -> None:
        cls.service = new_service

    @classmethod
    def get_model(cls) -> str:
        return cls.model

    @classmethod
    def set_model(cls, new_model: str) -> None:
        cls.model = new_model
