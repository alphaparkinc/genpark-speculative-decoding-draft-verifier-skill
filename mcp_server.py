"""
MCP Server for Speculative Decoding Draft Verifier Skill.
"""

import json
import sys
from client import SpeculativeDecoder

DECODER = SpeculativeDecoder(gamma=4)


def handle_request(req: dict) -> dict:
    method = req.get("method")
    params = req.get("params", {})

    if method == "tools/list":
        return {
            "tools": [
                {
                    "name": "verify_speculative_tokens",
                    "description": "Verify draft tokens using rejection sampling against target distribution",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "draft_tokens": {"type": "array", "items": {"type": "string"}},
                            "draft_probs": {"type": "array", "items": {"type": "number"}},
                            "target_probs": {"type": "array", "items": {"type": "number"}},
                            "uniform_randoms": {"type": "array", "items": {"type": "number"}}
                        },
                        "required": ["draft_tokens", "draft_probs", "target_probs"]
                    }
                },
                {
                    "name": "get_aggregate_stats",
                    "description": "Get overall acceptance rate and speculative speedup ratio",
                    "inputSchema": {
                        "type": "object"
                    }
                }
            ]
        }
    elif method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})

        if tool_name == "verify_speculative_tokens":
            randoms = args.get("uniform_randoms", [0.5] * len(args["draft_tokens"]))
            res = DECODER.verify_tokens(
                args["draft_tokens"],
                args["draft_probs"],
                args["target_probs"],
                randoms
            )
            return {"content": [{"type": "text", "text": json.dumps(res)}]}

        elif tool_name == "get_aggregate_stats":
            res = DECODER.get_aggregate_stats()
            return {"content": [{"type": "text", "text": json.dumps(res)}]}

        return {"error": f"Unknown tool: {tool_name}"}

    return {"error": f"Unknown method: {method}"}


def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            resp = handle_request(req)
            resp["id"] = req.get("id")
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"error": str(e)}) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
