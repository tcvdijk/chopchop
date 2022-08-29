import argparse
parser = argparse.ArgumentParser(description="Looks at the 'pos' and 'neg' subdirectory and !!DELETE!! from the one that has more.")
parser.add_argument('dir', type=str,
                    help="Directory with a 'pos' and a 'neg' subdirectory.")
parser.add_argument('-f','--factor', type=float, default=1,
                    help="Allows a factor of imbalance. (Default: 1)")
args = vars(parser.parse_args())

### settings

from_dir = args['dir']
factor = args['factor']

### imports
from glob import glob
from random import shuffle
from os import unlink

###

pos = glob(f"{from_dir}/pos/*.png")
print( "Number of pos:", len(pos) )

neg = glob(f"{from_dir}/neg/*.png")
print( "Number of neg:", len(neg) )

many = None
if len(pos)>factor*len(neg):
    print("Pos has more than",factor,"times more files than neg; going to delete files from pos.")
    many = pos
    target = int(factor*len(neg))
elif len(neg)>factor*len(pos):
    print("Neg has more than",factor,"times more files than pos; going to delete files from neg.")
    many = neg
    target = int(factor*len(pos))
else:
    print("Pos and neg are close enough; doing nothing.")
    exit()

if many:
    print("Going to reduce number of files to",target)
    shuffle(many)
    to_delete = many[:-target]
    print("About to delete files like",to_delete[0])
    confirm = input("Is that ok? Enter to continue...")
    if confirm!="":
        print("Aborting.")
        exit()
    for file in to_delete:
        unlink(file)

print("done :)")