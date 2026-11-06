"""Create a Buttondown DRAFT from the newest report_week<N>.md.

It never sends anything. You open Buttondown, read the draft, and press send.
Skips quietly if BUTTONDOWN_API_KEY isn't set, and only drafts each week once
(tracked in drafted_weeks.txt).
"""
import glob
import json
import os
import re
import sys
import urllib.error
import urllib.request

key = os.environ.get("BUTTONDOWN_API_KEY", "").strip()
if not key:
    print("No BUTTONDOWN_API_KEY set, skipping newsletter draft.")
    sys.exit(0)

files = glob.glob("report_week*.md")
weeks = {}
for f in files:
    m = re.search(r"report_week(\d+)\.md$", f)
    if m:
        weeks[int(m.group(1))] = f
if not weeks:
    print("No report_week*.md found, skipping.")
    sys.exit(0)

week = max(weeks)
done = set()
if os.path.exists("drafted_weeks.txt"):
    done = set(open("drafted_weeks.txt").read().split())
tag = os.environ.get("SEASON", "") + "-" + str(week)
if tag in done:
    print("Week %d already drafted, skipping." % week)
    sys.exit(0)

body = open(weeks[week], encoding="utf-8").read()
payload = {
    "subject": "StartSignal Week %d: waiver wire and trends" % week,
    "body": body,
    "status": "draft",
}
req = urllib.request.Request(
    "https://api.buttondown.com/v1/emails",
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": "Token " + key,
        "Content-Type": "application/json",
    },
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        print("Draft created for week %d (HTTP %s)." % (week, r.status))
except urllib.error.HTTPError as e:
    print("Buttondown refused the draft: HTTP %s %s" % (e.code, e.read()[:300]))
    sys.exit(1)

with open("drafted_weeks.txt", "a") as fh:
    fh.write(tag + "\n")
