import pytest
import os
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ToxicityMetric, SummarizationMetric, HallucinationMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

os.environ["OPENAI_API_KEY"] = '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'

def pull_dataset_from_deepeval():
    dataset = EvaluationDataset()

    dataset.pull(alias="Eval Dataset v2")

    return dataset

dataset = pull_dataset_from_deepeval()

@pytest.mark.parametrize(
    "test_case",
    dataset,
)
def test_customer_chatbot(test_case: LLMTestCase):
    faithfulness_metric = FaithfulnessMetric(threshold=0.3)
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    toxicity_metric = ToxicityMetric(threshold=0.5)
    hallucination_metric = HallucinationMetric(threshold=0.5)
    summarization_metric = SummarizationMetric(threshold=0.5, model="gpt-4",
    assessment_questions=[
        "Is the coverage score based on how accurately the summarization covered all the aspects?",
        "Does the score ensure the summary's accuracy with the source?",
        "Does a higher score mean a more comprehensive summary?"
    ])
    assert_test(test_case, [faithfulness_metric, answer_relevancy_metric, toxicity_metric, summarization_metric, hallucination_metric])
