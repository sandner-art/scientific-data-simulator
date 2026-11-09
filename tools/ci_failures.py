"""Failed tests, short, for the GitHub Actions run summary (CI's "Failures" step).

    python tools/ci_failures.py pytest-report.xml "pytest (windows-latest, 3.12)" >> "$GITHUB_STEP_SUMMARY"

Reads pytest's JUnit XML (``pytest --junitxml=pytest-report.xml``) and writes Markdown: the job,
the counts, and per failed test its name, message and the end of its traceback - everything
needed to fix it, with none of the passing tests. Every failing job adds its part, so the run's
Summary page holds all failures on one page, ready to copy.
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET

KEEP_LINES = 45
MAX_MESSAGE = 500


def _test_id(case: ET.Element) -> str:
    """``tests/test_x.py::test_name``, or ``tests/test_x.py::TestClass::test_name``."""
    parts = [p for p in case.get("classname", "").split(".") if p]
    name = case.get("name", "?")
    if not parts:
        return name
    if len(parts) > 1 and parts[-1][:1].isupper():  # a test class inside the module
        return "/".join(parts[:-1]) + f".py::{parts[-1]}::{name}"
    return "/".join(parts) + f".py::{name}"


def _fence(lines: list) -> str:
    """A code fence longer than any backtick run in the text, so a traceback cannot close it."""
    longest = max((len(m) for line in lines for m in re.findall(r"`+", line)), default=0)
    return "`" * max(3, longest + 1)


def main(path: str, job: str) -> str:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as e:
        return f"### ❌ {job}\nNo test report ({e}) - the tests did not run; see the log of the Install step.\n"
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    n = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    bad_count = n["failures"] + n["errors"]
    out = [f"### ❌ {job} - {bad_count} failed, "
           f"{n['tests'] - bad_count - n['skipped']} passed, {n['skipped']} skipped", ""]
    for case in root.iter("testcase"):
        for bad in [*case.findall("failure"), *case.findall("error")]:
            kind = "error" if bad.tag == "error" else "failed"
            lines = (bad.text or "").rstrip().splitlines()
            if len(lines) > KEEP_LINES:
                lines = ["…"] + lines[-KEEP_LINES:]
            message = (bad.get("message") or "").strip()[:MAX_MESSAGE]
            quoted = ["> " + line for line in message.splitlines()] or [">"]
            fence = _fence(lines)
            out += [f"**{_test_id(case)}** ({kind})", "", *quoted, "",
                    f"{fence}text", *lines, fence, ""]
    if bad_count == 0:
        out += ["No failed tests in the report: the job failed in another step; see its log.", ""]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    # The summary is UTF-8 Markdown; a Windows runner's redirected stdout would otherwise be cp1252.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdout.write(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "tests"))
