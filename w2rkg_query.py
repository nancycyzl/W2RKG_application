'''
A simple query tool for the W2RKG.
You can input waste and / or resource, and the tool will return the triples that match the query.
'''


import json
from rapidfuzz import fuzz
import re

w2rkg_db = json.load(open('data_utils/fused_triples_aggregated.json', 'r', encoding='utf-8'))

continue_query = True

def match_two_strings(str1, str2, exact_match):
    # exact match
    if exact_match.lower() == 'y' or exact_match.lower() == 'yes':
        if str1 == str2:
            return True
        else:
            return False
        
    # fuzzy match
    else:
        if compute_material_similarity(str1, str2) > 0.7:
            return True
        else:   
            return False
        

def preprocess_material_name(s):
    # Lowercase, remove punctuation, and normalize whitespace
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def compute_material_similarity(str1, str2):
    # If str1 is empty, return 1
    if not str1:
        return 1.0
    # If str1 is in str2 or str2 is in str1, return 1
    if str1 in str2 or str2 in str1:
        return 1.0
    s1 = preprocess_material_name(str1)
    s2 = preprocess_material_name(str2)
    return fuzz.token_set_ratio(s1, s2) / 100.0
    
def parse_dict_to_string(dict_item):
    return "\n".join([f"{key}: {value}" for key, value in dict_item.items()])

while continue_query:
    print("----------------------------------------------")
    waste = input("Enter a waste (q to quit): ")
    resource = input("Enter a resource (q to quit):")
    exact_match = input("Exact match? (y/n): ")
    number = int(input("Return how many results, -1 for all: "))

    if resource == 'q' or waste == 'q':
        continue_query = False
        break
    
    if (not waste) and resource:
        for one_w2r in w2rkg_db:
            if match_two_strings(one_w2r['transformed_resource'], resource, exact_match):
                print(parse_dict_to_string(one_w2r))
                if number != -1:
                    number -= 1
                    if number == 0:
                        break
    
    if (not resource) and waste:
        for one_w2r in w2rkg_db:
            if match_two_strings(one_w2r['waste'], waste, exact_match):
                print(parse_dict_to_string(one_w2r))
                if number != -1:
                    number -= 1
                    if number == 0:
                        break
    
    if resource and waste:
        for one_w2r in w2rkg_db:
            if match_two_strings(one_w2r['transformed_resource'], resource, exact_match) and match_two_strings(one_w2r['waste'], waste, exact_match):
                print(parse_dict_to_string(one_w2r))
                if number != -1:
                    number -= 1
                    if number == 0:
                        break






