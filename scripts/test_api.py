
import httpx

API_URL = "http://localhost:8000/generate"

# Generate a schema exceeding the permitted limit
large_schema = "A" * 7000

payload = {
    "schema": large_schema,
    "question": "How many singers are older than 30?"
}

response = httpx.post(
    API_URL,
    json=payload,
    timeout=30
)

print("Status:", response.status_code)
print("Response:", response.text)

assert response.status_code == 422, (
    f"Expected 422, got {response.status_code}"
)

print("PASS: Oversized schema correctly rejected.")