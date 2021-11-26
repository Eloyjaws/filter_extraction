import argparse

parser = argparse.ArgumentParser(description='Run filter extraction')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
print(args)

if __name__ == "__main__":
    print(f"Extracting Filters from ")
