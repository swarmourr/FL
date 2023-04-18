import argparse
from FL import main
# Construct the argument parser and parse the arguments
arg_desc = '''\
        Let's run a some FL rounds with MNIST Datasets!
        --------------------------------  
        '''
parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter,
                                    description= arg_desc)
 
parser.add_argument("-p", "--participants", metavar="Participants", help = "Numbre of participants" , type=int)
parser.add_argument("-e", "--epochs", metavar="Epochs", help = "Number of epochs", type=int )
parser.add_argument("-b", "--batch", metavar="Batch", help = "Size of training batch", type=int )
parser.add_argument("-r", "--rounds", metavar="Rounds", help = "Number of rounds",type=int)
parser.add_argument("-d", "--data_path", metavar="data", help = "Path to training data if local and global same machine", type=str)
args = vars(parser.parse_args())
main(args["participants"],args["rounds"],args["epochs"],args["batch"])