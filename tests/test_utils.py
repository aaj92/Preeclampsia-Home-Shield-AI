import numpy as np
from utils import parse_urine_score, extract_positive_probability


def test_parse_urine_score_valid():
    assert parse_urine_score("2: Medium Green") == 2.0


def test_parse_urine_score_invalid():
    assert parse_urine_score("no colon here") == 0.0


def test_extract_positive_probability_array():
    arr = np.array([[0.7, 0.3]])
    assert extract_positive_probability(arr) == 0.3


def test_extract_positive_probability_list():
    arr = [[0.6, 0.4]]
    assert extract_positive_probability(arr) == 0.4


def test_extract_positive_probability_fallback():
    arr = [[0.9]]
    assert extract_positive_probability(arr) == 0.9
