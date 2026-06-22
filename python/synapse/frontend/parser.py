from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class ParsedFunction:
    name: str
    source: str
    tree: ast.Module
    file_name: str | None

def parse_function(fn: Callable) -> ParsedFunction:
    file_name = inspect.getsourcefile(fn)
    source = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(source)
    
    return ParsedFunction(
        name=fn.__name__,
        source=source,
        tree=tree,
        file_name=file_name 
    )

def dump_ast(fn: Callable) -> str:
    parsed = parse_function(fn)
    return ast.dump(parsed.tree, indent=2)
    