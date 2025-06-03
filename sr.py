import argparse
import utils.parse_json as parse_json

if __name__ == "__main__":

    ## parse configuration as .json (or .jsonc) file
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.jsonc',
                        help='JSON file for configuration')

    # parse configs
    args = parser.parse_args()
    opt = parse_json.parse(args)
    
    # Convert to NoneDict, which return None for missing key.
    opt = parse_json.dict_to_nonedict(opt)
    
    
    
    
    
    