from dataclasses import dataclass
from enum import Enum


class Role(str, Enum):
    admin = "admin"
    user = "user"


@dataclass
class User:
    id: str
    username: str
    role: Role
