#!/usr/bin/env python3
""" """
from typing import Any


def printc(header: str, *args: Any, column_width: int = 40) -> None:
    print(f"{header:<{column_width}}", *args)
