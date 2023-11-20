from os import path
import json
from file_indexer import get_file_idx

json_file_name = 'twist_json_file.json'


# Depreciated
def export_twist(cur_twist):

    # File naming index
    file_index = get_file_idx()

    # Convert our Twist message to a serializable dictionary object
    twist_d = twist_2_dict(cur_twist)

    # Dict to populate JSON file with
    entries = {}

    # If the file already exists we will add to it
    if path.exists(json_file_name):

        # load existing entries
        with open(json_file_name, 'r') as export_file:
            entries = json.load(export_file)
    
    # Update by appending new entry
    entries[str(file_index)] = twist_d

    with open(json_file_name, 'w') as export_file:
        json.dump(entries, export_file, indent=4)


# Converts a Twist message to a dictionary
def twist_2_dict(twist):

    t_dict = {
        "linear": {
            "x": twist.linear.x,
            "y": twist.linear.y,
            "z": twist.linear.z
        },
        "angular": {
            "x": twist.angular.x,
            "y": twist.angular.y,
            "z": twist.angular.z
        }
    }

    return t_dict