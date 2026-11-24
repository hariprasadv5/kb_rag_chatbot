"""
make_sample_corpus.py — generates a realistic demo corpus for "Northwind Systems"
(plus a second tenant "Contoso") so every Module-2 technique has something to prove:

  * PDFs ........ HR leave policy (2024 + 2026 versions → time filtering), travel policy
  * DOCX ........ DB-failover runbook (numbered steps), Acme MSA contract (clauses)
  * HTML ........ incident-response runbook (web page)
  * PPTX ........ security-awareness deck
  * XLSX ........ approved-software catalogue
  * CSV ......... support tickets with exact IDs (INC-2041 …) → BM25 exact-ID wins
  * Contoso PDF . another tenant's leave policy → tenant-isolation test

Run:  python make_sample_corpus.py      (writes data/raw/** and data/manifest.yaml)
Replace data/raw with your real documents later — just update data/manifest.yaml.
"""
from __future__ import annotations

import csv
from pathlib import Path

import yaml

RAW = Path(__file__).resolve().parent / "data" / "raw"


# ----------------------------------------------------------------------------- writers
def write_pdf(path: Path, title: str, sections: list[tuple[str, str]]) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    for i, (head, body) in enumerate(sections):
        story.append(Paragraph(head, styles["Heading2"]))
        for para in body.strip().split("\n\n"):
            story.append(Paragraph(para.replace("\n", " "), styles["BodyText"]))
            story.append(Spacer(1, 6))
        if i % 2 == 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(path), pagesize=A4).build(story)


def write_docx(path: Path, title: str, blocks: list[tuple[str, object]]) -> None:
    import docx

    path.parent.mkdir(parents=True, exist_ok=True)
    d = docx.Document()
    d.add_heading(title, level=0)
    for kind, content in blocks:
        if kind == "h1":
            d.add_heading(content, level=1)
        elif kind == "p":
            d.add_paragraph(content)
        elif kind == "steps":
            for s in content:
                d.add_paragraph(s, style="List Number")
        elif kind == "table":
            rows = content
            t = d.add_table(rows=len(rows), cols=len(rows[0]))
            for r, row in enumerate(rows):
                for c, val in enumerate(row):
                    t.cell(r, c).text = str(val)
    d.save(path)


def write_pptx(path: Path, slides: list[tuple[str, list[str], str]]) -> None:
    from pptx import Presentation

    path.parent.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    for title, bullets, notes in slides:
        s = prs.slides.add_slide(prs.slide_layouts[1])
        s.shapes.title.text = title
        tf = s.placeholders[1].text_frame
        tf.text = bullets[0]
        for b in bullets[1:]:
            tf.add_paragraph().text = b
        s.notes_slide.notes_text_frame.text = notes
    prs.save(path)


def write_xlsx(path: Path, header: list[str], rows: list[list]) -> None:
    from openpyxl import Workbook

    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "Approved Software"
    ws.append(header)
    for r in rows:
        ws.append(r)
    wb.save(path)


# ----------------------------------------------------------------------------- content
LEAVE_2024 = [
    ("1. Purpose and scope", """This policy (HR-POL-002, version 3.1) describes paid time off (PTO), sick leave and parental leave for all full-time Northwind Systems employees. It is effective from 15 January 2024 and replaces version 2.4.

Part-time employees receive leave pro-rated to their contracted hours."""),
    ("2. Annual paid time off", """Full-time employees accrue 18 days of paid time off per calendar year, accrued monthly at 1.5 days per month.

Up to 3 unused PTO days may be carried over into the next calendar year. Carried-over days expire on 31 March. PTO requests longer than 5 consecutive working days must be submitted at least 2 weeks in advance in Workday."""),
    ("3. Sick leave", """Employees receive 8 days of paid sick leave per year. A medical certificate is required for absences longer than 3 consecutive days."""),
    ("4. Parental leave", """Primary caregivers receive 16 weeks of fully paid parental leave. Secondary caregivers receive 4 weeks of fully paid leave, to be taken within 12 months of the birth or adoption."""),
]

LEAVE_2026 = [
    ("1. Purpose and scope", """This policy (HR-POL-002, version 4.0) describes paid time off (PTO), sick leave, parental leave and volunteering leave for all full-time Northwind Systems employees. It is effective from 1 March 2026 and supersedes version 3.1 dated 15 January 2024.

Part-time employees receive leave pro-rated to their contracted hours. Contractors are not covered by this policy."""),
    ("2. Annual paid time off", """Full-time employees accrue 22 days of paid time off per calendar year, accrued monthly at 1.83 days per month.

Up to 5 unused PTO days may be carried over into the next calendar year. Carried-over days expire on 30 June. PTO requests longer than 5 consecutive working days must be submitted at least 2 weeks in advance in Workday and approved by the line manager."""),
    ("3. Sick leave", """Employees receive 10 days of paid sick leave per year. A medical certificate is required for absences longer than 3 consecutive days. Sick leave does not carry over."""),
    ("4. Parental leave", """Primary caregivers receive 26 weeks of fully paid parental leave. Secondary caregivers receive 8 weeks of fully paid leave, to be taken within 12 months of the birth or adoption. Parental leave can be split into at most two blocks."""),
    ("5. Volunteering leave", """Every employee may take 2 paid volunteering days per year for registered charities. Volunteering days are booked in Workday under the leave type 'VOL'."""),
    ("6. Questions", """Questions about this policy go to the People Team at people@northwind.example or the #ask-hr Slack channel."""),
]

TRAVEL = [
    ("1. Scope", """The Travel and Expense Policy (FIN-POL-007, effective 1 July 2025) applies to all business travel and reimbursable expenses at Northwind Systems. Expenses are submitted in Expensify."""),
    ("2. Approval thresholds", """Trips with an estimated total cost up to USD 5,000 require line-manager approval. Trips above USD 5,000 require approval from the department Vice President. International travel always requires VP approval regardless of cost."""),
    ("3. Flights and hotels", """Economy class is the default for flights under 6 hours. Premium economy is allowed for flights of 6 hours or more; business class requires CFO approval.

The nightly hotel cap is USD 250 in standard cities and USD 350 in high-cost cities (New York, London, San Francisco, Zurich, Tokyo)."""),
    ("4. Meals and per diem", """The daily meal per diem is USD 75 for domestic travel and USD 95 for international travel. Alcohol is not reimbursable except for approved client entertainment."""),
    ("5. Receipts and deadlines", """Itemised receipts are required for every expense above USD 25. Expense reports must be submitted within 30 days of the trip end date; reports older than 90 days will not be reimbursed. Corporate card statements must be reconciled monthly."""),
]

CONTOSO_LEAVE = [
    ("1. Annual leave", """Contoso Ltd employees receive 15 days of paid annual leave per year. No carry-over is allowed. This policy CT-HR-11 is effective from 1 January 2025."""),
    ("2. Parental leave", """Contoso primary caregivers receive 12 weeks of paid parental leave."""),
]

DB_RUNBOOK = [
    ("p", "Runbook ID: RB-DB-004. Owner: Platform/SRE team. Last reviewed: 10 February 2026. Applies to the primary PostgreSQL 16 cluster 'orders-db' running in eu-west-1."),
    ("h1", "When to use this runbook"),
    ("p", "Use this runbook when the primary database 'orders-db-primary' is unhealthy for more than 5 minutes, when replication lag on the standby exceeds 30 seconds for 10 minutes, or when the alert PGPrimaryDown fires in PagerDuty."),
    ("h1", "Pre-checks"),
    ("steps", [
        "Acknowledge the PagerDuty incident and announce the failover in #inc-orders.",
        "Confirm the primary is really down: run 'pg_isready -h orders-db-primary -p 5432'.",
        "Check standby replication lag in Grafana dashboard 'Postgres / Replication'; lag must be below 5 seconds before promoting.",
        "Freeze deployments by setting the feature flag DEPLOY_FREEZE=true in LaunchDarkly.",
    ]),
    ("h1", "Failover procedure"),
    ("steps", [
        "Promote the standby: run 'patronictl -c /etc/patroni.yml failover orders-db --candidate orders-db-standby-1 --force'.",
        "Verify the new primary with 'patronictl list' — the role column must show Leader for orders-db-standby-1.",
        "Update the PgBouncer target by running the Ansible playbook 'pgbouncer-repoint.yml' with -e target=orders-db-standby-1.",
        "Run the smoke test 'make smoke-orders' and confirm checkout succeeds in staging and production.",
        "Rebuild the failed node as a new standby using 'patronictl reinit orders-db orders-db-primary'.",
    ]),
    ("h1", "Rollback and communication"),
    ("p", "If the smoke test fails, roll back by re-pointing PgBouncer to the previous primary (only if it is healthy) and escalate to the Database on-call (PagerDuty schedule 'DB-Primary'). Expected recovery time objective (RTO) is 15 minutes and recovery point objective (RPO) is 1 minute. Post the status update to status.northwind.example every 30 minutes until resolved."),
    ("table", [["Contact", "Role", "Channel"], ["Priya Nair", "DB on-call lead", "PagerDuty DB-Primary"], ["Marco Silva", "SRE manager", "#sre-leads"]]),
]

CONTRACT = [
    ("p", "Master Services Agreement — Contract ID CTR-2025-0412 between Northwind Systems Inc. ('Provider') and Acme Retail Group ('Customer'). Effective date: 12 May 2025. Initial term: 36 months."),
    ("h1", "Section 4 — Service levels"),
    ("p", "4.1 Provider shall make the Platform available 99.9% of the time in each calendar month, excluding scheduled maintenance announced at least 72 hours in advance. 4.2 If availability falls below 99.9%, Customer is entitled to a service credit of 10% of the monthly fee; below 99.0%, the credit is 25%. Service credits are Customer's sole remedy for availability failures."),
    ("h1", "Section 7 — Fees and payment"),
    ("p", "7.1 The annual subscription fee is USD 480,000, invoiced quarterly in advance. 7.2 Invoices are payable within 45 days of the invoice date. 7.3 Late payments accrue interest at 1% per month. 7.4 Fees may increase by at most 5% at each renewal."),
    ("h1", "Section 11 — Limitation of liability"),
    ("p", "11.1 Each party's total aggregate liability is capped at the fees paid or payable in the 12 months preceding the claim. 11.2 The cap does not apply to breaches of confidentiality, data-protection obligations or indemnification obligations, which are capped at 3 times the annual fees."),
    ("h1", "Section 14 — Term and termination"),
    ("p", "14.1 The agreement renews automatically for successive 12-month periods unless either party gives written notice of non-renewal at least 90 days before the end of the current term. 14.2 Either party may terminate for material breach with 60 days' written notice if the breach is not cured within that period. 14.3 On termination, Provider will return or delete Customer Data within 30 days."),
    ("h1", "Section 16 — Contacts"),
    ("table", [["Party", "Contract owner", "Email"], ["Northwind", "Legal — Hannah Okafor", "legal@northwind.example"], ["Acme Retail", "Procurement — Tom Reyes", "procurement@acme.example"]]),
]

INCIDENT_HTML = """<!doctype html><html><head><title>Incident Response Runbook — Northwind Systems</title></head>
<body><nav>Home | Engineering | Runbooks</nav>
<h1>Incident Response Runbook (RB-OPS-001)</h1>
<p>Last updated 20 November 2025. Owner: SRE. This page defines severity levels, paging and communication for production incidents.</p>
<h2>Severity levels</h2>
<ul>
<li><b>SEV1</b> — full outage or data loss affecting all customers. Page the Incident Commander immediately; status page update within 15 minutes; executive update every 30 minutes.</li>
<li><b>SEV2</b> — major feature degraded for many customers. Page the on-call engineer; status page update within 30 minutes.</li>
<li><b>SEV3</b> — minor degradation with a workaround. Create a Jira ticket in project OPS; no paging outside business hours.</li>
</ul>
<h2>Roles</h2>
<p>The Incident Commander (IC) owns the incident, the Communications Lead owns updates to status.northwind.example and customers, and the Ops Lead drives the technical fix. The IC is paged through the PagerDuty schedule 'IC-Primary'.</p>
<h2>Declaring an incident</h2>
<p>Type <code>/incident declare</code> in Slack. This creates a channel named #inc-YYYYMMDD-shortname, opens a Jira incident and starts the timeline. Anyone can declare an incident — when in doubt, declare.</p>
<h2>Post-incident review</h2>
<p>A blameless post-incident review (PIR) is mandatory for SEV1 and SEV2. The draft is due within 5 business days and action items are tracked in Jira with the label 'pir-action'.</p>
<footer>© Northwind Systems — internal use only</footer></body></html>"""

SECURITY_SLIDES = [
    ("Security Awareness 2025", ["Mandatory annual training for all staff", "Policy SEC-POL-010, effective 1 September 2025"], "Completion deadline is 30 November 2025."),
    ("Passwords and MFA", ["Minimum password length: 14 characters", "Use the company password manager (1Password)", "MFA is mandatory on all SaaS apps; use hardware keys for admin accounts"], "Passwords must never be shared, even with IT."),
    ("Reporting phishing", ["Do not click links in unexpected emails", "Use the 'Report Phish' button in Outlook or forward to security@northwind.example", "Security responds to reports within 1 business hour"], "Reporting quickly matters more than being certain."),
    ("Laptops and data", ["Full-disk encryption must stay enabled", "Customer data must never be stored on USB drives", "Lost or stolen devices must be reported to IT within 1 hour"], "Report to the IT service desk at ext. 4357 (HELP)."),
]

SOFTWARE = [
    ["Slack", "Communication", "IT", "Approved", "All staff", "2026-01-10"],
    ["Zoom", "Communication", "IT", "Approved", "All staff", "2026-01-10"],
    ["1Password", "Security", "Security", "Approved", "All staff", "2025-09-01"],
    ["Figma", "Design", "Product", "Approved", "Design, Product", "2025-06-15"],
    ["Notion", "Docs", "IT", "Restricted", "Engineering only; no customer data", "2025-04-02"],
    ["Dropbox", "Storage", "IT", "Not approved", "Use Google Drive instead", "2025-04-02"],
    ["DataGrip", "Developer", "Engineering", "Approved", "Engineering", "2025-11-20"],
    ["ChatGPT (consumer)", "AI", "Security", "Not approved", "Use the internal AI gateway instead", "2026-02-01"],
]

TICKETS = [
    ["INC-2041", "2025-12-03", "SEV2", "Resolved", "orders-db", "Checkout failing with 'connection pool exhausted' errors",
     "PgBouncer max_client_conn was 500 and saturated during the holiday sale. Root cause: connection leak in the orders service after release 3.14.2. Fix: rolled back to 3.14.1 and raised max_client_conn to 1000. Follow-up PIR action: add pool-saturation alert."],
    ["INC-2057", "2026-01-14", "SEV1", "Resolved", "orders-db", "Primary database down in eu-west-1",
     "orders-db-primary lost its disk. Failover executed using runbook RB-DB-004; standby-1 promoted in 11 minutes. RPO met (no data loss)."],
    ["INC-2063", "2026-02-02", "SEV3", "Resolved", "sso", "Some users cannot log in with Okta SSO",
     "Clock skew on two Okta agent hosts. Fixed by re-enabling NTP sync. Workaround was password + MFA login."],
    ["INC-2078", "2026-03-19", "SEV2", "Resolved", "search", "Product search returns stale prices",
     "Elasticsearch index refresh job stuck after a mapping change. Reindexed products-v7 and added a staleness monitor."],
    ["INC-2090", "2026-04-07", "SEV3", "Open", "email", "Password reset emails delayed by 20+ minutes",
     "SendGrid rate limiting due to marketing campaign sharing the same API key. Proposed fix: separate transactional sub-user."],
    ["INC-2102", "2026-05-22", "SEV2", "Resolved", "payments", "Card payments declined for Amex",
     "Payment gateway rotated its TLS certificate; our pinning config was outdated. Updated pinned certificates and removed pinning for the gateway domain."],
    ["REQ-3310", "2026-06-11", "-", "Closed", "it", "Request: access to DataGrip for analytics team",
     "Approved per software catalogue; licences assigned by IT."],
    ["REQ-3324", "2026-07-02", "-", "Closed", "hr", "Question: does carry-over PTO expire?",
     "Answered by People Team: under HR-POL-002 v4.0 carried-over days expire on 30 June."],
]


# ----------------------------------------------------------------------------- main
def main() -> None:
    files = []

    def add(rel, department, doc_type, dt, tenant="northwind", title=None, access="all"):
        files.append(dict(path=rel, department=department, doc_type=doc_type, date=dt,
                          tenant=tenant, title=title or Path(rel).stem.replace("_", " ").title(), access=access))

    write_pdf(RAW / "northwind/hr/leave_policy_2024.pdf", "Leave Policy HR-POL-002 v3.1 (2024)", LEAVE_2024)
    add("northwind/hr/leave_policy_2024.pdf", "hr", "policy", "2024-01-15", title="Leave Policy v3.1 (2024)")
    write_pdf(RAW / "northwind/hr/leave_policy_2026.pdf", "Leave Policy HR-POL-002 v4.0 (2026)", LEAVE_2026)
    add("northwind/hr/leave_policy_2026.pdf", "hr", "policy", "2026-03-01", title="Leave Policy v4.0 (2026)")
    write_pdf(RAW / "northwind/finance/travel_expense_policy.pdf", "Travel and Expense Policy FIN-POL-007", TRAVEL)
    add("northwind/finance/travel_expense_policy.pdf", "finance", "policy", "2025-07-01", title="Travel & Expense Policy")

    write_docx(RAW / "northwind/engineering/runbook_db_failover.docx", "Runbook: orders-db Failover", DB_RUNBOOK)
    add("northwind/engineering/runbook_db_failover.docx", "engineering", "runbook", "2026-02-10", title="Runbook RB-DB-004 DB Failover")
    write_docx(RAW / "northwind/legal/msa_acme_retail.docx", "Master Services Agreement — Acme Retail", CONTRACT)
    add("northwind/legal/msa_acme_retail.docx", "legal", "contract", "2025-05-12", title="MSA Acme Retail CTR-2025-0412", access="legal")

    p = RAW / "northwind/engineering/incident_response.html"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(INCIDENT_HTML, encoding="utf-8")
    add("northwind/engineering/incident_response.html", "engineering", "runbook", "2025-11-20", title="Incident Response Runbook RB-OPS-001")

    write_pptx(RAW / "northwind/it/security_awareness_2025.pptx", SECURITY_SLIDES)
    add("northwind/it/security_awareness_2025.pptx", "it", "policy", "2025-09-01", title="Security Awareness 2025 (SEC-POL-010)")
    write_xlsx(RAW / "northwind/it/approved_software.xlsx",
               ["Software", "Category", "Owner team", "Status", "Allowed for", "Last reviewed"], SOFTWARE)
    add("northwind/it/approved_software.xlsx", "it", "catalog", "2026-02-01", title="Approved Software Catalogue")

    tp = RAW / "northwind/support/tickets.csv"
    tp.parent.mkdir(parents=True, exist_ok=True)
    with tp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ticket_id", "date", "severity", "status", "service", "title", "resolution"])
        w.writerows(TICKETS)
    add("northwind/support/tickets.csv", "support", "ticket", "2026-07-02", title="Support Tickets")

    write_pdf(RAW / "contoso/hr/contoso_leave_policy.pdf", "Contoso Leave Policy CT-HR-11", CONTOSO_LEAVE)
    add("contoso/hr/contoso_leave_policy.pdf", "hr", "policy", "2025-01-01", tenant="contoso", title="Contoso Leave Policy")

    manifest = {
        "documents": files,
        # Web pages: uncomment / add your own intranet or public URLs.
        "urls": [
            # {"url": "https://example.com/handbook", "department": "hr", "doc_type": "web",
            #  "date": "2026-01-01", "tenant": "northwind", "title": "Public handbook"},
        ],
    }
    (RAW.parent / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(f"Wrote {len(files)} documents under {RAW} and data/manifest.yaml")


if __name__ == "__main__":
    main()
