import os
import dataclasses
from typing import Dict, Type, Any
from dotenv import dotenv_values


@dataclasses.dataclass(frozen=True)
class Env:
    BE_AUTH_TOKEN: str
    FE_BASE_URL: str
    BE_BASE_URL: str
    INPUT_DIR: str
    OUTPUT_DIR: str


def load_env(cls: Type[Any], path: str) -> Any:
    arg_types = {field.name: field for field in dataclasses.fields(cls)}
    settings: Dict[str, Any] = dict()
    for arg, val in dotenv_values(path).items():
        if arg not in arg_types:
            continue
        typ = arg_types[arg].type
        assert type(typ) is type  # FIXME: this works for now but isn't rigorous
        settings[arg] = typ(val)
    return cls(**settings)


ENV: Env = load_env(Env, ".env")

os.makedirs(ENV.INPUT_DIR, exist_ok=True)
os.makedirs(ENV.OUTPUT_DIR, exist_ok=True)
