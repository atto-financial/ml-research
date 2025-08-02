def extract_feature_answers(answers, answers_order):
    result = {}
    for order_spec in answers_order:
        group_name = order_spec['group']
        target_version = order_spec['version']
        filtered_answers = [
            answer for answer in answers 
            if answer['group'] == group_name and answer['version'] == target_version
        ]
        filtered_answers.sort(key=lambda x: int(x['item']))
        choice_numbers = [answer['choiceNumber'] for answer in filtered_answers]
        if choice_numbers:
            result[group_name] = choice_numbers
    
    return result