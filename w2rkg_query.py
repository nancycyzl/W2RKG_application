import json

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
        if str1 in str2 or str2 in str1:
            return True
        else:   
            return False
    
def parse_dict_to_string(dict_item):
    return "\n".join([f"{key}: {value}" for key, value in dict_item.items()])

while continue_query:
    print("----------------------------------------------")
    waste = input("Enter a waste (q to quit): ")
    resource = input("Enter a resource (q to quit):")
    exact_match = input("Exact match? (y/n): ")

    if resource == 'q' or waste == 'q':
        continue_query = False
        break
    
    if (not waste) and resource:
        for one_w2r in w2rkg_db:
            if match_two_strings(one_w2r['transformed_resource'], resource, exact_match):
                print(parse_dict_to_string(one_w2r))
                break
    
    if (not resource) and waste:
        for one_w2r in w2rkg_db:
            if match_two_strings(one_w2r['waste'], waste, exact_match):
                print(parse_dict_to_string(one_w2r))
                break
    
    if resource and waste:
        for one_w2r in w2rkg_db:
            if match_two_strings(one_w2r['transformed_resource'], resource, exact_match) and match_two_strings(one_w2r['waste'], waste, exact_match):
                print(parse_dict_to_string(one_w2r))
                break






