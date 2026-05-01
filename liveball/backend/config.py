from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    s3_endpoint: str = "minio.stateball.svc.cluster.local:9000"
    s3_bucket: str = "dazoo"
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""
    s3_use_ssl: bool = False
    ducklake_db_host: str = "ducklake-db-rw.stateball.svc.cluster.local"
    ducklake_db_port: int = 5432
    ducklake_db_user: str = "ducklake"
    ducklake_db_password: str = ""
    ducklake_db_name: str = "ducklake"
    jwt_secret: str = ""
    jwt_ttl_hours: int = 24
    jwt_cookie_secure: bool = True

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @model_validator(mode="after")
    def validate_secrets(self) -> "Settings":
        if not self.jwt_secret:
            raise ValueError("JWT_SECRET must be set")
        return self


settings = Settings()
