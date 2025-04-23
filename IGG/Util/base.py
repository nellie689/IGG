import json

def load_config(file_path):
    with open(file_path) as f:
        return json.load(f)

def merge_configs(basic_config, advanced_config):
    merged_config = basic_config.copy()

    for section, settings in advanced_config.items():
        if section not in merged_config:
            merged_config[section] = {}
        merged_config[section].update(settings)

    return merged_config

def save_config(config, output_file):
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=4)

# def main():
#     basic_config = load_config('basic_config.json')
#     advanced_config = load_config('advanced_config.json')
#     final_config = merge_configs(basic_config, advanced_config)

#     save_config(final_config, 'final_config.json')

