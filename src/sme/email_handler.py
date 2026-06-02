from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional

from config.settings import settings

# Regex to pull structured answers out of an SME reply body
# Matches: [REF:uuid] answer text  (one per line or until next [REF:])
_REF_ANSWER_RE = re.compile(
    r"\[REF:([a-f0-9\-]{36})\]\s*:?\s*(.+?)(?=\[REF:|$)",
    re.IGNORECASE | re.DOTALL,
)

# Regex to extract run_id from email subject
_SUBJECT_RUN_RE = re.compile(r"\[Lineage Agent\] Run ([a-f0-9\-]{36})", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Sending
# ---------------------------------------------------------------------------

def send_sme_questions(questions: list[dict]) -> list[dict]:
    """
    Send gap questions to SMEs via Azure Communication Services.

    Questions are grouped by SME email so each SME receives one email
    containing all their questions.  Each question gets a [REF:gap_id]
    marker so replies can be matched back deterministically.

    Returns the questions list with message_id and send_status populated.
    """
    # Group by SME email first — no Azure connection needed for this step
    by_sme: dict[str, list[dict]] = defaultdict(list)
    no_email: list[dict] = []
    for q in questions:
        if q.get("sme_email"):
            by_sme[q["sme_email"]].append(q)
        else:
            q["send_status"] = "skipped_no_email"
            no_email.append(q)

    sent_all: list[dict] = list(no_email)

    if not by_sme:
        return sent_all  # nothing to send via email

    if not settings.azure_communication_connection_string:
        for sme_questions in by_sme.values():
            for q in sme_questions:
                q.setdefault("send_status", "skipped_no_connection")
                sent_all.append(q)
        return sent_all

    try:
        from azure.communication.email import EmailClient
        client = EmailClient.from_connection_string(
            settings.azure_communication_connection_string
        )
    except Exception as exc:
        for sme_questions in by_sme.values():
            for q in sme_questions:
                q["send_status"] = f"error: {exc}"
                sent_all.append(q)
        return sent_all

    for sme_email, sme_questions in by_sme.items():
        run_id = sme_questions[0].get("run_id", "unknown")
        subject = f"[Lineage Agent] Run {run_id} — Input needed for data lineage review"
        plain, html = _format_email(sme_questions, run_id)

        message = {
            "senderAddress": settings.email_sender_address,
            "recipients": {"to": [{"address": sme_email}]},
            "content": {
                "subject": subject,
                "plainText": plain,
                "html": html,
            },
        }
        try:
            poller = client.begin_send(message)
            result = poller.result()
            msg_id = result.get("id") or result.get("messageId")
            for q in sme_questions:
                q["message_id"] = msg_id
                q["send_status"] = "sent"
        except Exception as exc:
            for q in sme_questions:
                q["send_status"] = f"error: {exc}"

        sent_all.extend(sme_questions)

    return sent_all


# ---------------------------------------------------------------------------
# Receiving / parsing
# ---------------------------------------------------------------------------

def parse_email_responses(raw_responses: list[dict]) -> list[dict]:
    """
    Parse inbound SME email replies into structured response objects.

    raw_responses: list of email payloads.  Expected keys:
      - from / sender   — SME email address
      - subject         — email subject (used to extract run_id)
      - body / text     — reply body text

    Returns a list of response dicts with:
      gap_id, sme_email, run_id, raw_answer, raw_text
    """
    parsed: list[dict] = []

    for email in raw_responses:
        body = email.get("body") or email.get("text", "")
        sme_email = email.get("from") or email.get("sender", "")
        subject = email.get("subject", "")

        run_id = _extract_run_id(subject, body)

        # Extract structured answers via [REF:gap_id]: answer pattern
        ref_answers = _extract_ref_answers(body)

        if ref_answers:
            for gap_id, raw_answer in ref_answers.items():
                parsed.append({
                    "gap_id":     gap_id,
                    "sme_email":  sme_email,
                    "run_id":     run_id,
                    "raw_answer": raw_answer.strip(),
                    "raw_text":   body,
                })
        else:
            # No structured markers — return the full body against no specific gap_id
            # The response processor will attempt to match by context
            parsed.append({
                "gap_id":     None,
                "sme_email":  sme_email,
                "run_id":     run_id,
                "raw_answer": body.strip(),
                "raw_text":   body,
            })

    return parsed


# ---------------------------------------------------------------------------
# Email formatting
# ---------------------------------------------------------------------------

def _format_email(questions: list[dict], run_id: str) -> tuple[str, str]:
    """Return (plain_text, html) for a batch of questions to one SME."""
    plain = _format_plain(questions, run_id)
    html = _format_html(questions, run_id)
    return plain, html


def _format_plain(questions: list[dict], run_id: str) -> str:
    lines = [
        "Hello,",
        "",
        "The Data Lineage Agent has identified gaps in the lineage documentation",
        "and needs your input to complete the data catalog.",
        "",
        "Please answer the questions below by replying to this email.",
        "Include the [REF:...] marker before each answer so your reply can be",
        "processed automatically.",
        "",
        "─" * 60,
    ]

    for i, q in enumerate(questions, 1):
        lines += [
            "",
            f"Question {i}  [REF:{q['gap_id']}]",
            f"Type: {q.get('gap_type', 'unknown')}",
            "",
            q["question"],
            "",
            f"  Answer:  [REF:{q['gap_id']}]: <your answer here>",
        ]

    lines += [
        "",
        "─" * 60,
        "",
        "Example reply format:",
    ]
    for q in questions:
        lines.append(f"  [REF:{q['gap_id']}]: <your answer>")

    lines += [
        "",
        "Thank you for helping improve our data documentation.",
        "",
        "Data Governance Team",
        "",
        f"Run ID: {run_id}",
    ]

    return "\n".join(lines)


def _format_html(questions: list[dict], run_id: str) -> str:
    rows = ""
    for i, q in enumerate(questions, 1):
        rows += f"""
      <tr>
        <td style="padding:8px;border-top:1px solid #ddd;vertical-align:top;
                   font-weight:bold;white-space:nowrap">Q{i}</td>
        <td style="padding:8px;border-top:1px solid #ddd;color:#666;
                   white-space:nowrap">{q.get('gap_type','')}</td>
        <td style="padding:8px;border-top:1px solid #ddd">{q['question']}</td>
        <td style="padding:8px;border-top:1px solid #ddd;font-family:monospace;
                   font-size:12px;color:#555">[REF:{q['gap_id']}]</td>
      </tr>"""

    example_replies = "\n".join(
        f"  [REF:{q['gap_id']}]: &lt;your answer&gt;" for q in questions
    )

    return f"""<!DOCTYPE html>
<html>
<body style="font-family:Helvetica,Arial,sans-serif;font-size:14px;color:#333;max-width:700px">
  <h2 style="color:#4472C4">Data Lineage — SME Input Required</h2>
  <p>The Data Lineage Agent needs your expertise to complete the lineage documentation.
     Please answer the questions below and reply to this email.</p>
  <p>Include the <code>[REF:...]</code> marker before each answer.</p>

  <table style="width:100%;border-collapse:collapse;margin:16px 0">
    <thead>
      <tr style="background:#f0f4fa">
        <th style="padding:8px;text-align:left">#</th>
        <th style="padding:8px;text-align:left">Type</th>
        <th style="padding:8px;text-align:left">Question</th>
        <th style="padding:8px;text-align:left">Reference</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>

  <h3>How to reply</h3>
  <p>Reply to this email with your answers using the reference markers:</p>
  <pre style="background:#f5f5f5;padding:12px;border-radius:4px;font-size:13px">{example_replies}</pre>

  <p style="color:#888;font-size:12px;margin-top:24px">
    Data Governance Team &nbsp;|&nbsp; Run ID: {run_id}
  </p>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_run_id(subject: str, body: str) -> Optional[str]:
    for text in (subject, body):
        m = _SUBJECT_RUN_RE.search(text)
        if m:
            return m.group(1)
    return None


def _extract_ref_answers(body: str) -> dict[str, str]:
    """Extract {gap_id: answer_text} from an SME reply body."""
    results: dict[str, str] = {}
    for m in _REF_ANSWER_RE.finditer(body):
        gap_id = m.group(1).lower().strip()
        answer = m.group(2).strip()
        if gap_id and answer:
            results[gap_id] = answer
    return results
