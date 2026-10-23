# Experiment 7: Preparing an LLM Application for Production-Ready Deployment

## 1. Aim
To prepare the RAG application from Experiment 5 for API-based use by adding input validation, error handling, basic logging, response-time measurement, and endpoint testing using FastAPI, LangChain, and GroqCloud.

---

## 2. Project Architecture & Setup
- **Framework**: FastAPI with Uvicorn ASGI server
- **LLM Backend**: GroqCloud API with `qwen/qwen3.8-27b` (selected for lowest token consumption, fastest inference ~160-350ms, and free tier quota efficiency)
- **Embeddings & Vector Store**: `BAAI/bge-small-en-v1.5` with Chroma vector store
- **Knowledge Base**: `data/wellness.txt`, `data/medication_safety.pdf`, and `data/faq.docx`
- **Orchestration**: LangChain LCEL (`ChatPromptTemplate | ChatGroq | StrOutputParser`)

### Directory Structure
```text
Exp7/
├── data/                               # Isolated dataset folder reducing clutter
│   ├── faq.docx                        # Paramount TPA FAQ document
│   ├── medication_safety.pdf           # Medication Safety Guide document
│   └── wellness.txt                    # Health & Wellness Notes
├── app.py                              # FastAPI REST API endpoints, validation, lifespan warmup & logging
├── support_rag.py                      # Decoupled RAG retrieval and GroqCloud generation pipeline
├── test_app.py                         # Pytest test suite for endpoint and error handling tests
├── client_test.py                      # Comprehensive HTTP test client executing all manual test cases
├── requirements.txt                    # Python package dependencies
├── .env.example                        # Template for environment variables
├── .env                                # Local secrets (excluded from git)
├── .gitignore                          # Excludes secrets, caches, and virtual environments
├── DISCUSSION_ANSWERS.md               # Complete answers to viva and discussion questions
└── README.md                           # Documentation and verified observation table
```

---

## 3. How to Run the Application

### Step 1: Activate Environment & Dependencies
```bash
cd /home/aryantikam/Desktop/SPIT/Exp7
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables
Set your GroqCloud credentials in `.env`:
```ini
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=qwen/qwen3.8-27b
MAX_OUTPUT_TOKENS=250
```

### Step 3: Start the FastAPI Server
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger Documentation: `http://127.0.0.1:8000/docs`

### Step 4: Run the Complete Test Client
In another terminal, execute the test client covering all edge cases and Exp5 questions:
```bash
python client_test.py
```

### Step 5: Run Automated Unit Tests
```bash
pytest test_app.py -v
```

---

## 4. Verified Observation Table

### A. Manual Section 9 Required Observation Table
This table records the exact test cases and outcomes as prescribed in Section 9 of the lab manual:

| Test Case | Input / Condition | Expected Result | Actual Result Obtained | Latency (ms) | Pass / Fail |
|---|---|---|---|:---:|:---:|
| **Health endpoint** | `GET /health` | API reports running and initialized | `{"status": "healthy", "version": "1.0.0", "retriever_initialized": true, "backend": "groqcloud"}` | 2.10 ms | **PASS** |
| **Supported question** | *What does the wellness note say about hydration?* | Grounded answer is returned with cited sources | Detailed factual answer on water, fruits, and broths extracted from `wellness.txt` | 288.73 ms | **PASS** |
| **Unsupported question** | *How do you build a quantum computer?* | Insufficient information / fallback triggered | *"The provided documents do not contain sufficient information to answer this question."* | 168.45 ms | **PASS** |
| **Empty question** | `{"question": ""}` | Client validation error (HTTP 422) | Returned HTTP 422 with detail: `String should have at least 1 character` | 1.97 ms | **PASS** |
| **Backend unavailable** | Simulated network / authentication failure | Controlled error; no secret details or traces leaked | Returned HTTP 500: *"Failed to generate answer from the backend LLM service. Please check API key, quota, or network connectivity."* | 1.10 ms | **PASS** |
| **Response-time measurement** | Latency recorded in milliseconds | Latency is recorded and returned in response | Every valid response includes `latency_ms` (ranging from 157.88 ms to 425.43 ms) | Recorded | **PASS** |

---

### B. Extended Observation Table: Input Validation & Edge Cases

| Edge Case Test | Test Input | HTTP Status | Actual Detail / Response | Client Latency | Pass / Fail |
|---|---|:---:|---|:---:|:---:|
| **Empty String** | `""` | 422 | `string_too_short: min_length 1` | 1.97 ms | **PASS** |
| **Whitespace Only** | `"          "` | 422 | `Value error: Question cannot be empty or contain only whitespace` | 1.43 ms | **PASS** |
| **Oversized Input** | 1,090 chars (> 1000 limit) | 422 | `string_too_long: max_length 1000` | 1.13 ms | **PASS** |
| **Script / SQL Injection** | `<script>alert('xss')</script> OR '1'='1' --` | 200 | Safely sanitized and treated as text; returned insufficient info fallback | 427.37 ms | **PASS** |
| **Malformed JSON** | `{"query": "..."}` (missing `question`) | 422 | `missing: Field required` | 1.54 ms | **PASS** |

---

### C. Extended Observation Table: Experiment 5 Test Questions

#### 1. Relevant Questions (Supported Knowledge Base)
All 10 relevant questions from Experiment 5 successfully retrieved grounded answers without hallucinations:

| # | Question | Retrieved Source(s) | Actual Extracted Answer Summary | Backend Latency | Pass / Fail |
|:---:|---|---|---|:---:|:---:|
| 1 | *What does the wellness note say about hydration?* | `wellness.txt` | Explains hydration importance, drinking water, fruits (watermelon, oranges), soups. | 286.79 ms | **PASS** |
| 2 | *What are the signs of under-nutrition?* | `wellness.txt` | Lists thin appearance, fatigue, frequent illness, delayed growth in children. | 322.19 ms | **PASS** |
| 3 | *What are some simple tips for healthy living regarding proteins?* | `wellness.txt` | Tips: include protein in every meal, variety of sources, protein-rich snacks, grilling/steaming. | 389.27 ms | **PASS** |
| 4 | *How many hours of sleep should one get each night for healthy living?* | `wellness.txt` | Correctly identified **8-10 hours** of sleep each night. | 177.29 ms | **PASS** |
| 5 | *What is the Turnaround Time (TAT) for cashless file submission to the TPA?* | `faq.docx` | Correctly identified **15 days** from date of patient discharge. | 373.91 ms | **PASS** |
| 6 | *What is the Provider Module?* | `faq.docx`, `medication_safety.pdf` | Portal for network hospitals to track cashless files, download approval/query letters. | 296.52 ms | **PASS** |
| 7 | *Where should the hospital submit the claim documents physically?* | `faq.docx` | Physically at the local Paramount TPA office. | 178.62 ms | **PASS** |
| 8 | *What is the National Health Claim Exchange (NHCX)?* | `faq.docx` | Common repository by NHA and IRDAI under Ayushman Bharat Digital Mission. | 275.40 ms | **PASS** |
| 9 | *How does NCC MERP define a medication error?* | `medication_safety.pdf` | *"Any preventable event that may cause or lead to harm..."* | 214.47 ms | **PASS** |
| 10 | *What is the most common therapeutic intervention to improve outcomes?* | `medication_safety.pdf` | Identifies **pharmacotherapy** as the most common intervention. | 190.63 ms | **PASS** |

#### 2. Irrelevant Questions (Out-of-Domain Refusal)
All 5 out-of-domain questions from Experiment 5 triggered strict refusal without hallucination:

| # | Question | Expected Behavior | Actual Response | Backend Latency | Pass / Fail |
|:---:|---|---|---|:---:|:---:|
| 1 | *How do you build a quantum computer?* | Strict refusal | *"The provided documents do not contain sufficient information to answer this question."* | 166.47 ms | **PASS** |
| 2 | *What is the capital of France?* | Strict refusal | *"The provided documents do not contain sufficient information to answer this question."* | 159.70 ms | **PASS** |
| 3 | *Who won the FIFA World Cup in 2022?* | Strict refusal | *"The provided documents do not contain sufficient information to answer this question."* | 256.03 ms | **PASS** |
| 4 | *How many planets are in the solar system?* | Strict refusal | *"The provided documents do not contain sufficient information to answer this question."* | 166.88 ms | **PASS** |
| 5 | *What is the recipe for baking chocolate chip cookies?* | Strict refusal | *"The provided documents do not contain sufficient information to answer this question."* | 155.83 ms | **PASS** |

---

## 5. Result
The RAG application from Experiment 5 was prepared for production-ready deployment by:
1. Exposing it via modular **FastAPI** endpoints (`GET /health` and `POST /ask`).
2. Selecting **Option A (GroqCloud API)** with `qwen/qwen3.8-27b`, achieving average inference latencies of ~160 ms to 350 ms.
3. Implementing robust **Pydantic input validation** preventing empty, whitespace-only, and oversized queries.
4. Implementing controlled **error handling** preventing credential or stack trace leakage on backend failure.
5. Verifying zero multiline comments, clean single-line section headers, and modular project organization.

Selected backend: **[X] Cloud-hosted API (GroqCloud)** &nbsp;&nbsp;&nbsp;&nbsp; [ ] Ollama local LLM

| Test Case | Input / Condition | Expected Result | Actual Result Obtained | Latency (ms) | Pass / Fail |
|---|---|---|---|:---:|:---:|
| **Health endpoint** | `GET /health` | API reports running and initialized | `{"status": "healthy", "version": "1.0.0", "retriever_initialized": true, "backend": "groqcloud"}` | 2.10 ms | **PASS** |
| **Supported question** | *What does the wellness note say about hydration?* | Grounded answer is returned with cited sources | Detailed factual answer on water, fruits, and broths extracted from `wellness.txt` | 288.73 ms | **PASS** |
| **Unsupported question** | *How do you build a quantum computer?* | Insufficient information / fallback triggered | *"The provided documents do not contain sufficient information to answer this question."* | 168.45 ms | **PASS** |
| **Empty question** | `{"question": ""}` | Client validation error (HTTP 422) | Returned HTTP 422 with detail: `String should have at least 1 character` | 1.97 ms | **PASS** |
| **Backend unavailable** | Simulated network / authentication failure | Controlled error; no secret details or traces leaked | Returned HTTP 500: *"Failed to generate answer from the backend LLM service. Please check API key, quota, or network connectivity."* | 1.10 ms | **PASS** |
| **Response-time measurement** | Latency recorded in milliseconds | Latency is recorded and returned in response | Every valid response includes `latency_ms` (ranging from 157.88 ms to 425.43 ms) | Recorded | **PASS** |

---

### B. Extended Observation Table: Input Validation & Edge Cases

| Edge Case Test | Test Input | HTTP Status | Actual Detail / Response | Client Latency | Pass / Fail |
|---|---|:---:|---|:---:|:---:|
| **Empty String** | `""` | 422 | `string_too_short: min_length 1` | 1.97 ms | **PASS** |
| **Whitespace Only** | `"          "` | 422 | `Value error: Question cannot be empty or contain only whitespace` | 1.43 ms | **PASS** |
| **Oversized Input** | 1,090 chars (> 1000 limit) | 422 | `string_too_long: max_length 1000` | 1.13 ms | **PASS** |
| **Script / SQL Injection** | `<script>alert('xss')</script> OR '1'='1' --` | 200 | Safely sanitized and treated as text; returned insufficient info fallback | 427.37 ms | **PASS** |
| **Malformed JSON** | `{"query": "..."}` (missing `question`) | 422 | `missing: Field required` | 1.54 ms | **PASS** |

---

### C. Extended Observation Table: Experiment 5 Test Questions

#### 1. Relevant Questions (Supported Knowledge Base)
All 10 relevant questions from Experiment 5 successfully retrieved grounded answers without hallucinations:

| # | Question | Retrieved Source(s) | Actual Extracted Answer Summary | Backend Latency | Pass / Fail |
|:---:|---|---|---|:---:|:---:|
| 1 | *What does the wellness note say about hydration?* | `wellness.txt` | Explains hydration importance, drinking water, fruits (watermelon, oranges), soups. | 286.79 ms | **PASS** |
| 2 | *What are the signs of under-nutrition?* | `wellness.txt` | Lists thin appearance, fatigue, frequent illness, delayed growth in children. | 322.19 ms | **PASS** |
| 3 | *What are some simple tips for healthy living regarding proteins?* | `wellness.txt` | Tips: include protein in every meal, variety of sources, protein-rich snacks, grilling/steaming. | 389.27 ms | **PASS** |
| 4 | *How many hours of sleep should one get each night for healthy living?* | `wellness.txt` | Correctly identified **8-10 hours** of sleep each night. | 177.29 ms | **PASS** |
| 5 | *What is the Turnaround Time (TAT) for cashless file submission to the TPA?* | `faq.docx` | Correctly identified **15 days** from date of patient discharge. | 373.91 ms | **PASS** |
| 6 | *What is the Provider Module?* | `faq.docx`, `medication_safety.pdf` | Portal for network hospitals to track cashless files, download approval/query letters. | 296.52 ms | **PASS** |
| 7 | *Where should the hospital submit the claim documents physically?* | `faq.docx` | Physically at the local Paramount TPA office. | 178.62 ms | **PASS** |
| 8 | *What is the National Health Claim Exchange (NHCX)?* | `faq.docx` | Common repository by NHA and IRDAI under Ayushman Bharat Digital Mission. | 275.40 ms | **PASS** |
| 9 | *How does NCC MERP define a medication error?* | `medication_safety.pdf` | *"Any preventable event that may cause or lead to harm..."* | 214.47 ms | **PASS** |
| 10 | *What is the most common therapeutic intervention to improve outcomes?* | `medication_safety.pdf` | Identifies **pharmacotherapy** as the most common intervention. | 190.63 ms | **PASS** |

#### 2. Irrelevant Questions (Out-of-Domain Refusal)
All 5 out-of-domain questions from Experiment 5 triggered strict refusal without hallucination:

| # | Question | Expected Behavior | Actual Response | Backend Latency | Pass / Fail |
|:---:|---|---|---|:---:|:---:|
| 1 | *How do you build a quantum computer?* | Strict refusal | *"The provided documents do not contain sufficient information to answer this question."* | 166.47 ms | **PASS** |
| 2 | *What is the capital of France?* | Strict refusal | *"The provided documents do not contain sufficient information to answer this question."* | 159.70 ms | **PASS** |
| 3 | *Who won the FIFA World Cup in 2022?* | Strict refusal | *"The provided documents do not contain sufficient information to answer this question."* | 256.03 ms | **PASS** |
| 4 | *How many planets are in the solar system?* | Strict refusal | *"The provided documents do not contain sufficient information to answer this question."* | 166.88 ms | **PASS** |
| 5 | *What is the recipe for baking chocolate chip cookies?* | Strict refusal | *"The provided documents do not contain sufficient information to answer this question."* | 155.83 ms | **PASS** |

---

## 5. Result
The RAG application from Experiment 5 was prepared for production-ready deployment by:
1. Exposing it via modular **FastAPI** endpoints (`GET /health` and `POST /ask`).
2. Selecting **Option A (GroqCloud API)** with `qwen/qwen3.8-27b`, achieving average inference latencies of ~160 ms to 350 ms.
3. Implementing robust **Pydantic input validation** preventing empty, whitespace-only, and oversized queries.
4. Implementing controlled **error handling** preventing credential or stack trace leakage on backend failure.
5. Verifying zero multiline comments, clean single-line section headers, and modular project organization.

Selected backend: **[X] Cloud-hosted API (GroqCloud)** &nbsp;&nbsp;&nbsp;&nbsp; [ ] Ollama local LLM

