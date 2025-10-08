# utils/prompter.py
"""
A dedicated helper to manage templates and prompt building.

Enhancements:
- Backward compatible with the original call sites.
- Robust JSON-string input parsing (auto-detect and format as structured input).
- Intentionally ignore 'input.extra' to avoid duplicated/ noisy prompt content.
- Rich diagnostics: template validation, placeholder checks, input-shape checks.
- Strict mode (raise) vs lenient mode (warn) for easier integration debugging.
- CLI debug utility: preview the composed prompt from the first sample of a JSON/JSONL file.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from typing import Union, Any, Dict, List, Optional


# -------------------------
# Custom exceptions
# -------------------------

class PrompterError(Exception):
    """Base class for Prompter-related errors."""


class TemplateLoadError(PrompterError):
    pass


class TemplateFormatError(PrompterError):
    pass


class JSONParseError(PrompterError):
    pass


class DataShapeError(PrompterError):
    pass


# -------------------------
# Utility helpers
# -------------------------

def _warn(msg: str) -> None:
    sys.stderr.write(f"WARNING: {msg}\n")


def _err(msg: str) -> None:
    sys.stderr.write(f"ERROR: {msg}\n")


def _json_dumps_safe(obj: Any, **kwargs) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, **kwargs)
    except Exception:
        return str(obj)


# -------------------------
# Prompter
# -------------------------

class Prompter(object):
    __slots__ = ("template", "_verbose", "_strict", "_use_template", "_custom_response_split", "template_name")

    def __init__(self, template_name: str = "", verbose: bool = False, strict: bool = False,
                 use_template: bool = True):
        """
        Args:
            template_name: name under templates/<name>.json (default "alpaca")
            verbose: print detailed debug logs to stdout/stderr
            strict: raise on suspicious states instead of warning
            use_template: when False, do not use JSON template; compose prompt in a minimal custom format.
                If None, read from env PROMPTER_USE_TEMPLATE (default True)
        """
        self._verbose = verbose
        self._strict = strict
        self._custom_response_split = "### Response:"

        # Internal boolean flag only
        self._use_template = bool(use_template)

        if not template_name:
            template_name = "alpaca"
        self.template_name = template_name

        if self._use_template:
            file_name = osp.join("templates", f"{template_name}.json")
            # 1) Load template file (only when using template)
            if not osp.exists(file_name):
                msg = f"Can't read template file: {file_name}"
                if strict:
                    raise TemplateLoadError(msg)
                _warn(msg + " (falling back to error later)")
            try:
                with open(file_name, "r", encoding="utf-8") as fp:
                    self.template = json.load(fp)
            except Exception as e:
                msg = f"Failed to load template JSON {file_name}: {e}"
                if strict:
                    raise TemplateLoadError(msg)
                _err(msg)
                # Minimal fallback to avoid crash later
                self.template = {"prompt_input": "{instruction}\n{input}", "prompt_no_input": "{instruction}", "response_split": "### Response:"}

            # 2) Validate template structure (only if using template)
            self._validate_template(template_name)
        else:
            # Custom mode: no template needed
            self.template = {}

        if self._verbose:
            print(
                f"Using {'TEMPLATE' if self._use_template else 'CUSTOM'} prompter [{template_name}]: {self.template.get('description', '') if self._use_template else 'no preamble; minimal format'}"
            )

    # -------------------------
    # Template validation
    # -------------------------
    def _validate_template(self, template_name: str) -> None:
        required_keys = ["prompt_input", "prompt_no_input", "response_split"]
        missing = [k for k in required_keys if k not in self.template]
        if missing:
            msg = f"Template '{template_name}' missing keys: {missing}"
            if self._strict:
                raise TemplateFormatError(msg)
            _warn(msg)

        # Placeholder checks
        pi = self.template.get("prompt_input", "")
        pni = self.template.get("prompt_no_input", "")
        if "{instruction}" not in pi or "{input}" not in pi:
            msg = "Template 'prompt_input' must contain both '{instruction}' and '{input}'."
            if self._strict:
                raise TemplateFormatError(msg)
            _warn(msg)
        if "{instruction}" not in pni:
            msg = "Template 'prompt_no_input' must contain '{instruction}'."
            if self._strict:
                raise TemplateFormatError(msg)
            _warn(msg)

        rs = self.template.get("response_split", "")
        if not isinstance(rs, str) or not rs.strip():
            msg = "Template 'response_split' should be a non-empty string."
            if self._strict:
                raise TemplateFormatError(msg)
            _warn(msg)

    # -------------------------
    # Format helpers
    # -------------------------
    def _format_structured_input(self, obj: Dict[str, Any]) -> str:
        """
        Canonical fields (all optional):
          - head: str
          - tail: str
          - candidates: List[str]
          - paths_text: List[str]   # preformatted natural language paths
          - paths: List[Dict]       # items may contain 'path_text'
          - addition: Any           # handled separately, not printed here
          - extra: Dict|str         # IGNORED on purpose in this version
        Fallback: JSON dump (without 'extra'/'addition').
        """
        if not isinstance(obj, dict):
            if self._strict:
                raise DataShapeError(f"Structured input expected dict, got {type(obj).__name__}")
            _warn(f"Structured input expected dict, got {type(obj).__name__}; dumping raw.")
            return _json_dumps_safe(obj)

        # Drop noisy keys from printed block; 'addition' is handled outside
        safe_obj: Dict[str, Any] = {k: v for k, v in obj.items() if k not in ("extra", "addition")}
        parts: List[str] = []

        # Head / Tail
        head = safe_obj.get("head")
        tail = safe_obj.get("tail")
        if head is not None and not isinstance(head, (str, int, float)):
            msg = f"'head' should be a string-like, got {type(head).__name__}"
            if self._strict:
                raise DataShapeError(msg)
            _warn(msg)
        if tail is not None and not isinstance(tail, (str, int, float)):
            msg = f"'tail' should be a string-like, got {type(tail).__name__}"
            if self._strict:
                raise DataShapeError(msg)
            _warn(msg)

        if head is not None:
            parts.append(f"Head: {head}")
        if tail is not None:
            parts.append(f"Tail: {tail}")

        # Candidates
        candidates: Any = safe_obj.get("candidates") or []
        if candidates and not isinstance(candidates, list):
            msg = f"'candidates' should be a list, got {type(candidates).__name__}"
            if self._strict:
                raise DataShapeError(msg)
            _warn(msg)
        if isinstance(candidates, list) and candidates:
            parts.append("candidate: " + _json_dumps_safe(candidates))

        # Multi-hop evidence
        paths_text: Any = safe_obj.get("paths_text") or []
        evidence_lines: List[str] = []
        if paths_text:
            if not isinstance(paths_text, list):
                msg = f"'paths_text' should be a list, got {type(paths_text).__name__}"
                if self._strict:
                    raise DataShapeError(msg)
                _warn(msg)
                paths_text = [str(paths_text)]
            for p in paths_text:
                if p is None:
                    continue
                evidence_lines.append(str(p))

        if not evidence_lines:
            raw_paths = safe_obj.get("paths") or []
            if raw_paths:
                if not isinstance(raw_paths, list):
                    msg = f"'paths' should be a list, got {type(raw_paths).__name__}"
                    if self._strict:
                        raise DataShapeError(msg)
                    _warn(msg)
                    raw_paths = []
                for idx, p in enumerate(raw_paths):
                    if isinstance(p, dict) and p.get("path_text"):
                        evidence_lines.append(str(p["path_text"]))
                    elif isinstance(p, dict):
                        _warn(f"'paths[{idx}]' has no 'path_text' field; skipping.")
                    else:
                        _warn(f"'paths[{idx}]' is not an object; skipping.")

        if evidence_lines:
            parts.append("Evidence:\n" + "\n".join(f"- {ln}" for ln in evidence_lines))
        else:
            # 显式占位，便于后续替换或展示
            parts.append("Evidence:\nNone")

        if parts:
            return "\n".join(parts)

        # Fallback to a JSON dump of sanitized object
        return _json_dumps_safe(safe_obj)

    def _format_addition(self, addition: Any) -> str:
        """Format the optional 'addition' block appended at the END of the prompt."""
        if addition is None:
            return ""
        if isinstance(addition, str):
            text = addition.strip()
            return text if text else ""
        if isinstance(addition, list):
            lines = []
            for it in addition:
                s = str(it).strip()
                if s:
                    lines.append(f"- {s}")
            return "\n".join(lines)
        if isinstance(addition, dict):
            return _json_dumps_safe(addition, indent=2, sort_keys=True)
        return str(addition)

    # -------------------------
    # Main API
    # -------------------------
    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str, Dict[str, Any]] = None,
        label: Union[None, str] = None,
        addition: Any = None,
    ) -> str:
        """
        Build the full prompt from instruction and optional input/label/addition.

        Backward compatibility:
        - If `input` is a JSON string, we'll try `json.loads(input)`; when it becomes a dict,
          we format structurally (ignoring 'extra') instead of dumping raw JSON.
        - If `addition` is None and `input` (after possible parsing) contains key 'addition',
          we auto-use it. 'extra' is never printed.
        """
        input_text: Optional[str] = None
        parsed_input_obj: Optional[Dict[str, Any]] = None

        # Normalize 'instruction'
        if instruction is None:
            if self._strict:
                raise DataShapeError("'instruction' must not be None")
            _warn("'instruction' is None; using empty string.")
            instruction = ""
        elif not isinstance(instruction, str):
            _warn(f"'instruction' expected str, got {type(instruction).__name__}; coercing to str.")
            instruction = str(instruction)

        # Normalize input: dict OR JSON string OR raw string
        if input:
            if isinstance(input, dict):
                parsed_input_obj = input
                if self._verbose:
                    print("[DEBUG] Input is dict; will format structurally and ignore 'extra'.")
            else:
                s = str(input)
                # Try to parse JSON
                try:
                    maybe = json.loads(s)
                    if isinstance(maybe, dict):
                        parsed_input_obj = maybe
                        if self._verbose:
                            print("[DEBUG] Input is JSON string -> parsed to dict; will format structurally and ignore 'extra'.")
                    else:
                        input_text = s  # keep as raw string
                        if self._verbose:
                            print("[DEBUG] Input is string but not a JSON object; keeping raw.")
                except Exception as e:
                    msg = f"Failed to parse input JSON string: {e}"
                    if self._strict:
                        raise JSONParseError(msg)
                    _warn(msg)
                    input_text = s

        # If we got a dict (from caller or parsed), format structurally and ignore 'extra'
        if parsed_input_obj is not None:
            if addition is None and "addition" in parsed_input_obj:
                addition = parsed_input_obj.get("addition")
                if self._verbose and addition is not None:
                    print("[DEBUG] Auto-picked 'addition' from input dict.")
            input_text = self._format_structured_input(parsed_input_obj)

        # Build prompt
        if self._use_template:
            try:
                if input_text is not None:
                    res = self.template["prompt_input"].format(instruction=instruction, input=input_text)
                else:
                    res = self.template["prompt_no_input"].format(instruction=instruction)
            except KeyError as e:
                msg = f"Missing template key while formatting: {e}"
                if self._strict:
                    raise TemplateFormatError(msg)
                _err(msg)
                # Last-resort fallback
                res = f"{instruction}\n{input_text or ''}"
            except Exception as e:
                msg = f"Unexpected error while formatting prompt: {e}"
                if self._strict:
                    raise TemplateFormatError(msg)
                _err(msg)
                res = f"{instruction}\n{input_text or ''}"
        else:
            # Custom minimal format: start directly with instruction value; no extra headers, no response marker
            if input_text is not None and input_text.strip():
                res = f"{instruction}\n\n{input_text}"
            else:
                res = f"{instruction}"

        # Append addition at the very end (before label) with explicit header
        addition_text = self._format_addition(addition)
        if addition_text:
            # 若已有 "Evidence:\nNone"，则替换为具体内容；否则追加新的 Evidence 块
            marker = "Evidence:\nNone"
            if marker in res:
                res = res.replace(marker, f"Evidence:\n{addition_text}")
            else:
                res = f"{res}\n\nEvidence:\n{addition_text}"
        else:
            # 若当前没有 Evidence 块（例如 input 非结构化且无 paths），补充占位
            if "Evidence:" not in res:
                res = f"{res}\n\nEvidence:\nNone"

        # Append label if provided (training-time)
        if label is not None and not isinstance(label, str):
            _warn(f"'label' expected str or None, got {type(label).__name__}; coercing to str.")
            label = str(label)
        if label:
            # 确保label前有换行符分隔
            res = f"{res}\n{label}"

        if self._verbose:
            print("=== [DEBUG] Composed Prompt Begin ===")
            print(res)
            print("=== [DEBUG] Composed Prompt End ===")
        return res

    def get_response(self, output: str) -> str:
        """Extract the model response after the response_split marker. Be robust if marker missing."""
        if output is None:
            if self._strict:
                raise DataShapeError("'output' is None in get_response")
            _warn("'output' is None in get_response; returning empty string.")
            return ""

        if not isinstance(output, str):
            _warn(f"'output' expected str, got {type(output).__name__}; coercing to str.")
            output = str(output)

        split_token = (self._custom_response_split if not self._use_template else self.template.get("response_split", ""))
        if not split_token:
            msg = "Empty 'response_split' in template; cannot split reliably."
            if self._strict:
                raise TemplateFormatError(msg)
            _warn(msg)
            return output.strip()

        parts = output.split(split_token, 1)
        if len(parts) == 2:
            return parts[1].strip()
        # 静默回退：默认不告警以避免评测阶段大量刷屏；仅在 verbose/strict 下提示
        if self._verbose or self._strict:
            _warn(f"'response_split' token not found in output; returning full output. token={_json_dumps_safe(split_token)}")
        return output.strip()


# -------------------------
# CLI / Debug utilities
# -------------------------

def _read_first_record(path: str) -> Dict[str, Any]:
    """
    Read the first record from a JSON or JSONL file.
    - JSONL: first non-empty line parsed as an object.
    - JSON (list): first element.
    - JSON (object): the object itself.
    """
    if not osp.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    # Try JSON object/array first
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            data = json.loads(text)
            if isinstance(data, list):
                if not data:
                    raise ValueError("Empty JSON list.")
                if not isinstance(data[0], dict):
                    raise ValueError("First JSON list item is not an object.")
                return data[0]
            if isinstance(data, dict):
                return data
    except json.JSONDecodeError:
        # Fall back to JSONL
        pass
    except Exception as e:
        raise ValueError(f"Failed to read JSON file '{path}': {e}")

    # JSONL fallback
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"First JSONL record (line {lineno}) is not an object.")
                return obj
            except Exception as e:
                raise ValueError(f"Failed to parse JSONL at line {lineno}: {e}")
    raise ValueError("No records found.")

def _guess_fields(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Map common field names to instruction/input/label/addition for debug preview."""
    instr = (
        rec.get("instruction")
        or rec.get("prompt")
        or rec.get("question")
        or rec.get("task")
        or ""
    )
    inp = rec.get("input") or rec.get("inputs") or rec.get("context") or None
    add = rec.get("addition", None)
    label = (
        rec.get("label")
        or rec.get("output")
        or rec.get("outputs")
        or rec.get("response")
        or rec.get("answer")
        or None
    )
    return {"instruction": instr, "input": inp, "label": label, "addition": add}

def _print_debug_summary(rec: Dict[str, Any], prompt_text: str, sanitized_input: Optional[Dict[str, Any]], dump_template: bool, template: Dict[str, Any]) -> None:
    print("\n=== Debug Summary ===")
    print(f"- Keys in record: {', '.join(rec.keys())}")
    if isinstance(rec.get("input"), dict) and "extra" in rec["input"]:
        print("- Note: 'input.extra' detected (it will be ignored).")
    if sanitized_input is not None:
        print("- Sanitized input keys (printed): " + ", ".join(sanitized_input.keys()))
    print("\n=== Composed Prompt (BEGIN) ===")
    print(prompt_text)
    print("=== Composed Prompt (END) ===\n")
    if dump_template:
        print("=== Template Dump ===")
        print(_json_dumps_safe(template, indent=2, sort_keys=True))
        print("=====================\n")

def main():
    parser = argparse.ArgumentParser(
        description="Prompter debug utility: build and print a composed prompt from the first record of a JSON/JSONL file."
    )
    parser.add_argument("--template", type=str, default="alpaca", help="Template name under templates/<name>.json")
    parser.add_argument("--file", type=str, required=True, help="Path to a JSON or JSONL dataset file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging in Prompter")
    parser.add_argument("--strict", action="store_true", help="Enable strict mode (raise on issues instead of warn)")
    parser.add_argument("--override-addition", type=str, default=None, help="Optional raw text to override 'addition'.")
    parser.add_argument("--dump-template", action="store_true", help="Print the loaded prompt template JSON")
    parser.add_argument("--print-sanitized-input", action="store_true", help="Print the sanitized structured input (with 'extra' removed) if available")
    parser.add_argument("--use-template", type=int, default=1, help="1=use JSON template; 0=custom minimal format")
    args = parser.parse_args()

    try:
        rec = _read_first_record(args.file)
    except Exception as e:
        _err(str(e))
        sys.exit(2)

    fields = _guess_fields(rec)
    instruction = fields["instruction"] or ""
    input_block = fields["input"]
    label = fields["label"]
    addition = args.override_addition if args.override_addition is not None else fields["addition"]

    try:
        prompter = Prompter(template_name=args.template, verbose=args.verbose, strict=args.strict,
                            use_template=bool(args.use_template))
    except Exception as e:
        _err(f"Failed to initialize Prompter: {e}")
        sys.exit(2)

    # If input is a dict, show the sanitized structure (without 'extra'/'addition') for debugging
    sanitized_input: Optional[Dict[str, Any]] = None
    if args.print_sanitized_input and isinstance(input_block, dict):
        sanitized_input = {k: v for k, v in input_block.items() if k not in ("extra", "addition")}

    try:
        prompt_text = prompter.generate_prompt(
            instruction=instruction, input=input_block, label=label, addition=addition
        )
    except PrompterError as e:
        _err(f"Prompter error while generating prompt: {e}")
        sys.exit(3)
    except Exception as e:
        _err(f"Unexpected error while generating prompt: {e}")
        sys.exit(3)

    _print_debug_summary(rec, prompt_text, sanitized_input if args.print_sanitized_input else None, args.dump_template, prompter.template)


if __name__ == "__main__":
    main()
