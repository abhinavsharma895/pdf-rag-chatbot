from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.models import GeminiModel
from dotenv import load_dotenv
load_dotenv()


model = GeminiModel(
    model="gemini-3.5-flash-lite",
    temperature=0,
)

metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=model,
    include_reason=True
)
test_case = LLMTestCase(
    input="what is abhinave phone number?",
    # Replace this with the output from your LLM app
    actual_output="harsh phone number is 1234567890",
)

# To run metric as a standalone
# metric.measure(test_case)
# print(metric.score, metric.reason)

evaluate(test_cases=[test_case], metrics=[metric])