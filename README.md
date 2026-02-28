AI Research Digest 🧠📚

An automated pipeline and serverless web application that extracts, evaluates, and summarizes the best Artificial Intelligence research papers from arXiv every week.

This project addresses the "information overload" problem in AI research by utilizing Large Language Models (LLMs) to act as an expert panel of judges. The pipeline scores papers, selects winners across various categories, and publishes a sleek, static web page to an AWS S3 bucket.

✨ Key Features

Targeted arXiv Extraction: Searches for recent papers using complex queries filtered strictly by Computer Science categories (cs.AI, cs.LG, cs.CV, cs.CL) to eliminate noise from unrelated fields.

Two-Pass Tournament Scoring: Solves the context window fragmentation problem common in LLMs.

Phase 1 (Qualifiers): Evaluates papers in batches of 5 to filter noise and identify the top 10 candidates.

Phase 2 (Finals): Directly compares the top 10 finalists against each other to calibrate a global baseline and determine the definitive winner.

Accessible Summaries: Extracts full HTML text from arXiv to generate 220-260 word summaries tailored for practitioners (ML Engineers, Data Scientists).

Single-File Frontend: A modern, responsive (Dark Mode) static web app consolidated into a single index.html file for extreme cost optimization and fast load times on S3.

AWS Cost Efficiency: Runs on a low-cost Amazon EC2 instance that is triggered by a schedule and automatically shuts down as soon as the pipeline completes, reducing compute costs to pennies per month.

🏗️ Architecture

Scheduling: Amazon EventBridge starts the EC2 instance (e.g., every Monday at 02:00).

Execution (CRON): A Linux cron job triggers main.py upon startup.

Processing:

Fetches arXiv metadata.

Interacts with Amazon Bedrock (e.g., openai.gpt-oss-120b-1:0) to evaluate and summarize papers.

Storage: Processed JSON files (weeks.json and YYYY-MM-DD.json) are uploaded to a public Amazon S3 bucket.

Shutdown: The Python script invokes the OS shutdown command to power off the EC2 instance.

Presentation: Users access index.html on S3, which dynamically consumes the JSON data.

🚀 Setup & Installation

Prerequisites

Python 3.9+

AWS Account with permissions for S3, EC2, and Amazon Bedrock.

Local Installation

Clone the repository:

git clone [https://github.com/YOUR-USERNAME/PaperScope.git](https://github.com/YOUR-USERNAME/PaperScope.git)
cd PaperScope


Install dependencies:

pip install -r requirements.txt


Configure environment variables in a .env file:

AWS_REGION=us-east-1
BEDROCK_MODEL_ID=openai.gpt-oss-120b-1:0
OUTPUT_BUCKET=your-s3-bucket-name
OUTPUT_PREFIX=weeks


AWS Configuration

1. Amazon S3 (Frontend & Data)

Create an S3 bucket and disable "Block all public access."

Add a Bucket Policy to allow public read access (s3:GetObject).

Configure CORS rules to allow GET methods from your domain.

Upload index.html to the root of the bucket.

2. Amazon EC2 (Pipeline Worker)

Launch a t3.micro instance with Amazon Linux 2023.

Create and attach an IAM Role with AmazonS3FullAccess and AmazonBedrockFullAccess policies.

Transfer your code to the instance and install dependencies.

Add a cron job (crontab -e):

@reboot sleep 60 && cd /home/ec2-user/PaperScope && python3 main.py > pipeline.log 2>&1


3. Amazon EventBridge (Scheduler)

Create a Scheduler in EventBridge to trigger the EC2: StartInstances API at your desired time, setting your EC2 Instance ID as the target.

📂 Project Structure

.
├── main.py              # Core pipeline (Scraping, LLM, S3 Upload, and Shutdown)
├── requirements.txt     # Python dependencies
├── index.html           # Unified static frontend (HTML, CSS, and JS)
└── cache/               # (Auto-generated) Temporary file cache


🛠️ Modifying Topics

You can add or modify research categories by editing the TOPIC_QUERIES dictionary in main.py. The index.html frontend dynamically adapts its dropdown menus based on the generated JSON—no HTML code changes required!

Built with Python, AWS, and advanced LLMs.