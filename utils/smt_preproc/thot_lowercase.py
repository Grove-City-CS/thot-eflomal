# Author: Daniel Ortiz Mart\'inez
# *- python -*

# import modules
import io, sys, getopt
import thot_smt_preproc as smtpr

##################################################
def print_help():
    print >> sys.stderr, "thot_lowercase -f <string> [--help]"
    print >> sys.stderr, ""
    print >> sys.stderr, "-f <string>    File with text to be processed (can be read from stdin)"
    print >> sys.stderr, "--help         Print this help message"

##################################################
def main(argv):
    # take parameters
    f_given=False
    filename = ""
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hf:",["help","filename="])
    except getopt.GetoptError:
        print_help()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_help()
            sys.exit()
        elif opt in ("-f", "--filename"):
            filename = arg
            f_given=True

    # print parameters
    if(f_given==True):
        print >> sys.stderr, "f is %s" % (filename)

    # open file
    if(f_given==True):
        # open file
#        file = open(filename, 'r')
        file = io.open(filename, 'r', encoding="utf-8")
    else:
        # fallback to stdin
#        file = sys.stdin
        file = io.open(sys.stdin.fileno(), 'r', encoding='utf8')

    # read file line by line
    for line in file:
        line=line.strip("\n")
        line=smtpr.lowercase(line)
        print line.encode("utf-8")

if __name__ == "__main__":
    main(sys.argv)
