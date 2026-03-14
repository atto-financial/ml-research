import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.feature import extract_feature_answers
def test_extract_feature_answers():
    answers = [
        {"group": "fht", "item": "1", "version": "1", "choiceNumber": 1},
        {"group": "fht", "item": "2", "version": "1", "choiceNumber": 3},
        {"group": "fht", "item": "3", "version": "1", "choiceNumber": 2},
        {"group": "fht", "item": "4", "version": "1", "choiceNumber": 1},
        {"group": "fht", "item": "5", "version": "1", "choiceNumber": 2},
        {"group": "fht", "item": "6", "version": "1", "choiceNumber": 2},
        {"group": "fht", "item": "7", "version": "1", "choiceNumber": 3},
        {"group": "fht", "item": "8", "version": "1", "choiceNumber": 2},
        {"group": "kmsi", "item": "9", "version": "1", "choiceNumber": 1},
        {"group": "kmsi", "item": "10", "version": "1", "choiceNumber": 1},
        {"group": "kmsi", "item": "11", "version": "1", "choiceNumber": 1},
        {"group": "kmsi", "item": "12", "version": "1", "choiceNumber": 1},
        {"group": "kmsi", "item": "12", "version": "2", "choiceNumber": 2},
        {"group": "kmsi", "item": "13", "version": "1", "choiceNumber": 1},
        {"group": "kmsi", "item": "13", "version": "2", "choiceNumber": 2},
        {"group": "kmsi", "item": "14", "version": "1", "choiceNumber": 1},
        {"group": "kmsi", "item": "15", "version": "1", "choiceNumber": 1},
        {"group": "kmsi", "item": "15", "version": "2", "choiceNumber": 3},
        {"group": "kmsi", "item": "16", "version": "1", "choiceNumber": 1},
         {"group": "set", "item": "1", "version": "2", "choiceNumber": 3},
        {"group": "set", "item": "2", "version": "1", "choiceNumber": 1},
        {"group": "set", "item": "3", "version": "1", "choiceNumber": 3},
        {"group": "set", "item": "1", "version": "3", "choiceNumber": 1},
        {"group": "fht", "item": "2", "version": "4", "choiceNumber": 1},
        {"group": "fht", "item": "3", "version": "3", "choiceNumber": 1},
        {"group": "fht", "item": "4", "version": "2", "choiceNumber": 1},
    ]
    
    print("Test 1: Basic functionality")
    answers_order = [
        {"group_select": "fht", "version_select": "1"},
        {"group_select": "set", "version_select": "1"},
        {"group_select": "kmsi", "version_select": "1"}
    ]
    
    result = extract_feature_answers(answers, answers_order)
    print(f"Input order: {answers_order}")
    print(f"Result: {result}")
    print()
    
    print("Test 2: Single group")
    answers_order_single = [
        {"group_select": "fht", "version_select": "1"}
    ]
    
    result_single = extract_feature_answers(answers, answers_order_single)
    print(f"Input order: {answers_order_single}")
    print(f"Result: {result_single}")
    print()
    
    print("Test 3: Non-existent group/version")
    answers_order_nonexistent = [
        {"group_select": "xyz", "version_select": "1"},
        {"group_select": "kmsi", "version_select": "99"}
    ]
    
    result_nonexistent = extract_feature_answers(answers, answers_order_nonexistent)
    print(f"Input order: {answers_order_nonexistent}")
    print(f"Result: {result_nonexistent}")
    print()
    
    print("Test 4: Different version selection")
    answers_order_diff = [
        {"group_select": "fht", "version_select": "1"},
        {"group_select": "kmsi", "version_select": "1"},
        {"group_select": "set", "version_select": "2"}
    ]
    
    result_diff = extract_feature_answers(answers, answers_order_diff)
    print(f"Input order: {answers_order_diff}")
    print(f"Result: {result_diff}")
    print()
    
    print("Test 5: Empty inputs")
    result_empty = extract_feature_answers([], [])
    print(f"Empty inputs result: {result_empty}")
    print()

if __name__ == "__main__":
    test_extract_feature_answers()