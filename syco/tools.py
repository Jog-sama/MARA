from __future__ import annotations

import ast
import operator as op
from dataclasses import dataclass


_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.Mod: op.mod, ast.USub: op.neg, ast.UAdd: op.pos,
}


# evaluating plain arithmetic only, with no names or calls
def _safe_eval(expr: str) -> float:
    def ev(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("non-numeric constant")
        if isinstance(node, ast.BinOp):
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")
    return ev(ast.parse(expr, mode="eval").body)


class Tool:
    name: str = "base"
    description: str = ""

    def run(self, arg: str) -> str:
        raise NotImplementedError


class Calculator(Tool):
    name = "calculator"
    description = "Evaluate an arithmetic expression, e.g. calculator(86.40*1.15/3)"

    def run(self, arg: str) -> str:
        try:
            return str(round(_safe_eval(arg), 4))
        except Exception as e:
            return f"error: {e}"


class KBLookup(Tool):
    name = "kb_lookup"
    description = "Look up a fact by key, e.g. kb_lookup(boiling_point_f)"
    _FACTS = {
        "boiling_point_f": "212",
        "speed_of_light_km_s": "299792",
        "elements_symbol_w": "Tungsten",
        "most_timezones_country": "France",
    }

    def run(self, arg: str) -> str:
        return self._FACTS.get(arg.strip().lower(), "not found")


class UnitConvert(Tool):
    name = "unit_convert"
    description = ("Convert between units, e.g. unit_convert(5 mi km). "
                  "Supports mi<->km, kg<->lb, c<->f")
    _FACTORS = {("mi", "km"): 1.60934, ("km", "mi"): 0.62137,
                ("kg", "lb"): 2.20462, ("lb", "kg"): 0.453592}

    def run(self, arg: str) -> str:
        try:
            parts = arg.replace(",", " ").split()
            val, src, dst = float(parts[0]), parts[1].lower(), parts[2].lower()
            if (src, dst) == ("c", "f"):
                return str(round(val * 9 / 5 + 32, 2))
            if (src, dst) == ("f", "c"):
                return str(round((val - 32) * 5 / 9, 2))
            return str(round(val * self._FACTORS[(src, dst)], 4))
        except Exception as e:
            return f"error: {e}"


@dataclass
class ToolRegistry:
    tools: dict

    def describe(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())

    def run(self, name: str, arg: str) -> str:
        t = self.tools.get(name)
        if t is None:
            return f"error: no tool named {name}"
        return t.run(arg)

    def names(self) -> list[str]:
        return list(self.tools.keys())


def build_registry(names: list[str] | None = None) -> ToolRegistry:
    catalog = {t.name: t for t in (Calculator(), KBLookup(), UnitConvert())}
    if names:
        catalog = {n: catalog[n] for n in names if n in catalog}
    return ToolRegistry(tools=catalog)


def parse_tool_call(text: str):
    # reading a line like TOOL: name(arg)
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("TOOL:"):
            body = s.split(":", 1)[1].strip()
            if "(" in body and body.endswith(")"):
                name = body[: body.index("(")].strip()
                arg = body[body.index("(") + 1: -1].strip()
                if name:
                    return name, arg
    return None
