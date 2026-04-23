"""
core/collector.py

Pulls email metadata via IMAP. Reads only headers — no message body is ever fetched.
Extracts: timestamp, sender domain, recipient count, thread id, subject length.
"""

import imaplib
import email
import email.utils
import pandas as pd
from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


def connect(address: str, password: str, server: str, port: int) -> imaplib.IMAP4_SSL:
    conn = imaplib.IMAP4_SSL(server, port)
    conn.login(address, password)
    return conn


def fetch_metadata(conn: imaplib.IMAP4_SSL, max_emails: int = config.MAX_EMAILS) -> pd.DataFrame:
    """
    Fetch header-only metadata from INBOX.
    Returns a DataFrame with one row per email.
    """
    conn.select("INBOX", readonly=True)
    _, message_ids = conn.search(None, "ALL")
    ids = message_ids[0].split()
    ids = ids[-max_emails:]  # most recent N

    records = []
    print(f"Fetching metadata for {len(ids)} emails...")

    for i, msg_id in enumerate(ids):
        if i % 200 == 0:
            print(f"  {i}/{len(ids)}")
        try:
            _, data = conn.fetch(msg_id, "(BODY.PEEK[HEADER.FIELDS (FROM TO DATE SUBJECT MESSAGE-ID REFERENCES)])")
            raw = data[0][1]
            msg = email.message_from_bytes(raw)

            date_str = msg.get("Date", "")
            try:
                parsed_date = email.utils.parsedate_to_datetime(date_str)
                timestamp = parsed_date.timestamp()
                hour = parsed_date.hour
                weekday = parsed_date.weekday()
            except Exception:
                continue

            sender = msg.get("From", "")
            sender_domain = _extract_domain(sender)

            recipients = msg.get("To", "") or ""
            recipient_count = recipients.count("@")

            subject = msg.get("Subject", "") or ""
            subject_length = len(subject)

            thread_id = msg.get("Message-ID", "") or ""
            references = msg.get("References", "") or ""
            thread_depth = len(references.split()) if references else 0

            records.append({
                "timestamp": timestamp,
                "hour": hour,
                "weekday": weekday,
                "sender_domain": sender_domain,
                "recipient_count": max(1, recipient_count),
                "subject_length": subject_length,
                "thread_depth": thread_depth,
                "has_references": int(bool(references)),
            })
        except Exception:
            continue

    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _extract_domain(sender: str) -> str:
    try:
        addr = email.utils.parseaddr(sender)[1]
        return addr.split("@")[-1].lower() if "@" in addr else "unknown"
    except Exception:
        return "unknown"


def collect_and_save():
    print("Connecting to IMAP...")
    conn = connect(config.EMAIL_ADDRESS, config.EMAIL_PASSWORD, config.IMAP_SERVER, config.IMAP_PORT)
    df = fetch_metadata(conn)
    conn.logout()
    os.makedirs(config.DATA_DIR, exist_ok=True)
    df.to_csv(config.METADATA_PATH, index=False)
    print(f"Saved {len(df)} records to {config.METADATA_PATH}")
    return df


if __name__ == "__main__":
    collect_and_save()
