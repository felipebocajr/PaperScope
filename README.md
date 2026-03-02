# PaperScope

PaperScope is an automated pipeline and static web application that curates, evaluates, and summarizes the top AI research papers from arXiv on a weekly basis. It uses Large Language Models to rank papers across predefined research topics and publishes the results as a lightweight, serverless web digest.

---

## How It Works

The pipeline runs once per week on a scheduled EC2 instance and produces a static JSON output consumed directly by the frontend.

1. **Fetch** — Targeted arXiv queries retrieve recent papers filtered by Computer Science categories (`cs.AI`, `cs.LG`, `cs.CV`, `cs.CL`) across six research topics.
2. **Score (Qualifiers)** — Papers are evaluated in batches. Each batch is scored on four criteria: innovation, impact, methodology, and topic fit. The top 10 candidates per topic advance.
3. **Score (Finals)** — The 10 finalists are ranked directly against each other to calibrate a global baseline and select the definitive winner.
4. **Summarize** — The winning paper per topic receives a 220-260 word plain-text summary and a technical glossary, generated from the full paper HTML.
5. **Publish** — Results are written as JSON files to an S3 bucket. The EC2 instance shuts itself down on completion.

---

## Research Topics

- Agentic AI
- Reinforcement Learning
- LLMs & Applications
- Computer Vision
- Multimodal
- ML for Industrial & Physical Systems

Topics and their search queries are configured in the `TOPIC_QUERIES` dictionary in `main.py`. The frontend adapts dynamically to whatever topics are present in the JSON output.

---

## Architecture

| Component | Service | Role |
|---|---|---|
| Scheduler | Amazon EventBridge | Starts EC2 on a cron schedule |
| Worker | Amazon EC2 (t3.micro) | Runs the pipeline on boot |
| LLM | Amazon Bedrock | Scoring and summarization |
| Storage | Amazon S3 | Hosts JSON data and `index.html` |
| Frontend | Static HTML/JS | Reads JSON from S3, renders the digest |

---

## Project Structure

```
.
├── main.py              # Pipeline: fetch, score, summarize, upload, shutdown
├── index.html           # Frontend: single-file static web app
├── requirements.txt     # Python dependencies
├── weeks.json           # Index of available weekly digests
└── cache/               # Auto-generated temporary cache
```

---

## Setup

### Prerequisites

- Python 3.9+
- AWS account with access to S3, EC2, and Amazon Bedrock

### Local

```bash
git clone https://github.com/YOUR-USERNAME/PaperScope.git
cd PaperScope
pip install -r requirements.txt
```

Create a `.env` file:

```
AWS_REGION=us-east-1
OUTPUT_BUCKET=your-s3-bucket-name
OUTPUT_PREFIX=weeks
```

### AWS

**S3**
- Create a bucket and disable "Block all public access"
- Add a bucket policy granting public `s3:GetObject`
- Configure CORS to allow `GET` from your domain
- Upload `index.html` to the bucket root

**EC2**
- Launch a `t3.micro` instance (Amazon Linux 2023)
- Attach an IAM role with `AmazonS3FullAccess` and `AmazonBedrockFullAccess`
- Add a startup cron job:

```
@reboot sleep 60 && cd /home/ec2-user/PaperScope && python3 main.py > pipeline.log 2>&1
```

**EventBridge**
- Create a scheduler targeting the EC2 `StartInstances` API with your instance ID

---

## Stack

Python · AWS S3 · AWS EC2 · AWS Bedrock · Amazon EventBridge · arXiv API
