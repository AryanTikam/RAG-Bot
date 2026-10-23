# Comprehensive HTTP client testing all manual edge cases and Exp5 questions
import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"


# Helper function to print formatted test case results
def print_section(title: str):
    print("\n" + "=" * 80)
    print(f" {title.upper()} ")
    print("=" * 80)


def execute_test(category: str, test_name: str, payload: dict, expected_status: int = 200):
    print(f"\n[{category}] -> {test_name}")
    if "question" in payload:
        display_q = payload["question"]
        if len(display_q) > 80:
            display_q = display_q[:77] + "..."
        print(f"Question : {display_q}")

    try:
        start = time.perf_counter()
        response = requests.post(f"{BASE_URL}/ask", json=payload, timeout=20)
        client_latency = round((time.perf_counter() - start) * 1000, 2)

        print(f"Status   : HTTP {response.status_code} (Expected: {expected_status})")
        print(f"Latency  : {client_latency} ms")

        try:
            body = response.json()
            if response.status_code == 200:
                print(f"Answer   : {body.get('answer', '')}")
                sources = list({s['source'] for s in body.get('sources', [])})
                print(f"Sources  : {sources}")
                print(f"Backend  : {body.get('latency_ms', '')} ms")
            else:
                print(f"Detail   : {body.get('detail', '')}")
        except Exception:
            print(f"Raw Body : {response.text}")

        verdict = "PASS" if response.status_code == expected_status else "FAIL"
        print(f"Verdict  : {verdict}")

    except Exception as exc:
        print(f"Execution Error: {exc}")
        print("Verdict  : FAIL (Connection or Timeout)")


# Section 1: Health Check Endpoint Test
def run_health_check():
    print_section("1. Health Endpoint Verification")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Status   : HTTP {response.status_code}")
        print(f"Response : {json.dumps(response.json(), indent=2)}")
        verdict = "PASS" if response.status_code == 200 else "FAIL"
        print(f"Verdict  : {verdict}")
    except Exception as exc:
        print(f"Error reaching /health: {exc}")


# Section 2: Input Validation Edge Cases from Manual
def run_manual_edge_cases():
    print_section("2. Manual Edge Cases & Input Validation")

    # Edge Case 1: Empty String
    execute_test(
        category="VALIDATION",
        test_name="Empty Question String",
        payload={"question": ""},
        expected_status=422
    )

    # Edge Case 2: Whitespace Only
    execute_test(
        category="VALIDATION",
        test_name="Whitespace Only Question",
        payload={"question": "          "},
        expected_status=422
    )

    # Edge Case 3: Oversized Question > 1000 Characters
    oversized = "What are the safe medication practices? " + ("detail " * 150)
    execute_test(
        category="VALIDATION",
        test_name=f"Oversized Question ({len(oversized)} chars > 1000 limit)",
        payload={"question": oversized},
        expected_status=422
    )

    # Edge Case 4: Special Characters and SQL/HTML Injection Attempt
    execute_test(
        category="VALIDATION",
        test_name="Special Characters & Script Injection Payload",
        payload={"question": "<script>alert('xss')</script> OR '1'='1' --"},
        expected_status=200
    )

    # Edge Case 5: Missing Required 'question' Field
    execute_test(
        category="VALIDATION",
        test_name="Malformed JSON (Missing question key)",
        payload={"query": "What is hydration?"},
        expected_status=422
    )


# Section 3: Exp5 Relevant Questions (Knowledge Base Supported)
def run_exp5_relevant_questions():
    print_section("3. Exp5 Supported Knowledge Base Questions (Relevant)")

    relevant_questions = [
        "What does the wellness note say about hydration?",
        "What are the signs of under-nutrition?",
        "What are some simple tips for healthy living regarding proteins?",
        "How many hours of sleep should one get each night for healthy living?",
        "What is the Turnaround Time (TAT) for cashless file submission to the TPA?",
        "What is the Provider Module?",
        "Where should the hospital submit the claim documents physically?",
        "What is the National Health Claim Exchange (NHCX)?",
        "How does the United States National Coordinating Council for Medication Error Reporting and Prevention define a medication error?",
        "What is the most common therapeutic intervention in healthcare to improve health outcomes?"
    ]

    for idx, q in enumerate(relevant_questions, 1):
        execute_test(
            category=f"RELEVANT {idx}/10",
            test_name=q,
            payload={"question": q},
            expected_status=200
        )


# Section 4: Exp5 Irrelevant Questions (Out of Context / Fallback)
def run_exp5_irrelevant_questions():
    print_section("4. Exp5 Out-Of-Domain Questions (Irrelevant / Fallback)")

    irrelevant_questions = [
        "How do you build a quantum computer?",
        "What is the capital of France?",
        "Who won the FIFA World Cup in 2022?",
        "How many planets are in the solar system?",
        "What is the recipe for baking chocolate chip cookies?"
    ]

    for idx, q in enumerate(irrelevant_questions, 1):
        execute_test(
            category=f"IRRELEVANT {idx}/5",
            test_name=q,
            payload={"question": q},
            expected_status=200
        )


# Runner for all test suites
if __name__ == "__main__":
    run_health_check()
    run_manual_edge_cases()
    run_exp5_relevant_questions()
    run_exp5_irrelevant_questions()
    print("\n" + "=" * 80)
    print(" ALL TESTS COMPLETED ")
    print("=" * 80)
