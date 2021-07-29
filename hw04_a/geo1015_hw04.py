
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
     #-- runtime for smaller dataset: runtime: 0:00:15.231516
     #-- runtime for larger dataset: runtime: 0:01:13.752119

if __name__ == "__main__":
    main()