import pandas as pd
import json
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

def create_dataset_from_local():
    df = pd.read_csv('/Users/fahadpatel/Downloads/EvalDataset2.csv')
    print(df.columns)
    df = df.rename(columns={'Retrieval_text_1': 'input', 'response': 'actual_output', 'reference_summary': 'expected_output'})
    return df

def row_to_dict(row):
    result = {
        'input': row['input'],
        'actual_output': row['actual_output'],
        'expected_output': row['expected_output'],
        'retrieval_context': list(row['prompt'])
    }

    return result

def push_dataset_to_deepeval():
    df = create_dataset_from_local()
    original_dataset = df.apply(row_to_dict, axis=1).tolist()
    # json_dataset = json.dumps(original_dataset, indent=4)

    test_cases = []
    for datapoint in original_dataset:
        input = datapoint.get("input", None)
        actual_output = datapoint.get("actual_output", None)
        expected_output = datapoint.get("expected_output", None)

        test_case = LLMTestCase(
            input=input,
            actual_output=actual_output,
            expected_output=expected_output,
        )
        test_cases.append(test_case)

    dataset = EvaluationDataset(test_cases=test_cases)

    ## Push dataset to deep eval
    dataset.push(alias="Eval Dataset v2")

def pull_dataset_from_deepeval():
    dataset = EvaluationDataset()

    dataset.pull(alias="Eval Dataset v2")

push_dataset_to_deepeval()