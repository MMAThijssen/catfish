import helper_functions
from sys import argv
#from ExampleDb import check_lengths

db = helper_functions.load_db(argv[1])
number = int(argv[2]) * 2

db.set_ranges(83)
size = 250

[db.check_lengths(size, i) for i in range(number // size)]

rest = number - (number // size)
db.check_lengths(rest, 1)


#labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#print(len(labels))