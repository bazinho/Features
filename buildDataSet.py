#!/usr/bin/env python
"""
SYNOPSIS

    %prog -c classfile [-h,--help] [-v,--verbose] [--version]

DESCRIPTION

    This program ...

EXAMPLES

    %prog -c classfile

EXIT STATUS

    TODO: List exit codes

AUTHOR

    Edmar Rezende <bazinho@gmail.com>

LICENSE

    This script is in the public domain, free from copyrights or restrictions.

VERSION

    0.1
"""

import sys, os, traceback, optparse
import time
import struct
from dataset import *

def main ():
    global options
    ds = dataset(options.classfile,options.featuredir,options.outputfile,options.verbose)
    
if __name__ == '__main__':
    try:
        start_time = time.time()
        usage = 'Usage: %prog -c classfile -d featuredir -o outputfile [-h,--help] [-v,--verbose] [--version]'
        parser = optparse.OptionParser(usage=usage, version='0.1')
        parser.add_option ('-v', '--verbose', action='store_true', default=False, help='verbose output')
        parser.add_option ('-c', '--classfile', action="store", type="string", dest="classfile", help='set intput csv file containing filename <-> class map')
        parser.add_option ('-d', '--featuredir', action="store", type="string", dest="featuredir", help='directory containing one feature file for each filename')
        parser.add_option ('-o', '--outputfile', action="store", type="string", dest="outputfile", help='set output file to write the dataset')
        (options, args) = parser.parse_args()
        #if len(args) < 1:
        #    parser.error ('missing argument')
        if not options.classfile:
	  parser.error ('missing option: -c classfile')
        if options.verbose: print 'Start time:', time.asctime()
        main()
        if options.verbose: print 'End time:', time.asctime()
        if options.verbose: print 'TOTAL TIME:', (time.time() - start_time), 's'
        sys.exit(0)
    except KeyboardInterrupt, e: # Ctrl-C
        raise e
    except SystemExit, e: # sys.exit()
        raise e
    except Exception, e:
        print 'ERROR, UNEXPECTED EXCEPTION'
        print str(e)
        traceback.print_exc()
        os._exit(1)
  