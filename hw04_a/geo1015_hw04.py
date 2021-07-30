
import json, sys
from my_code_hw04 import lasToPlanes #detect_planes

def main():
    try:
        jparams = json.load(open('params.json'))
    except:
        print("ERROR: something is wrong with the params.json file.")
        sys.exit()

    #detect_planes(jparams)
    
    lasToPlanes(jparams)
     #-- runtime for smaller dataset: 0:00:15.799140
     #-- runtime for larger dataset: 0:01:28.193313

if __name__ == "__main__":
    main()