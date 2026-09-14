#!/usr/bin/env python3
import os
import random

from syco.env import load_env
load_env()

from syco.models import OpenAIClient
from syco.tools import build_registry
from syco.tasks import TrapTask


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("no OPENAI_API_KEY found. put it in .env or export it.")
        return
    try:
        import openai  # noqa: F401
    except ImportError:
        print("openai package not installed. run: pip install openai")
        return

    q = TrapTask().questions()[0]
    client = OpenAIClient(model=os.environ.get("SYCO_MODEL", "gpt-4o-mini"))
    registry = build_registry(["calculator", "kb_lookup", "unit_convert"])

    print(f"question: {q.prompt}")
    print(f"correct:  {q.answer}\n")
    turn = client.respond(q.prompt, q.options, [], {}, registry,
                          random.Random(0), "independent")
    print("raw reply (truncated):")
    print("  " + (turn.thought or "").replace("\n", "\n  ")[:400])
    print(f"\nparsed answer: {turn.answer}")
    print(f"parsed confidence: {turn.confidence}")
    print(f"used tool: {turn.tool_name}")
    print(f"tokens: in={turn.tokens_in} out={turn.tokens_out}")
    print("\nmatch: " + ("yes, parsing works" if turn.answer == q.answer
                          else "answer differs from correct (could be a real miss or a parse issue; check the raw reply)"))


if __name__ == "__main__":
    main()
