## \file
## \ingroup tutorial_pyroot
## This function provides an example of how one might
## massage a csv data file to read into a ROOT TTree
## via TTree::ReadStream.  This could be useful if the
## data read out from some DAQ program doesn't 'quite'
## match the formatting expected by ROOT (e.g. comma-
## separated, tab-separated with white-space strings,
## headers not matching the expected format, etc.)
##
## This example is shipped with a data
## file that looks like:
##
## ~~~{.cpp}
## Date/Time   Synchro   Capacity   Temp.Cold Head   Temp. Electrode   HV Supply Voltage   Electrode 1   Electrode 2   Electrode 3   Electrode 4
## # Example data to read out.  Some data have oddities that might need to
## # dealt with, including the 'NaN' in Electrode 4 and the empty string in Date/Time (last row)
## 08112010.160622   7   5.719000E-10   8.790500   24.237700   -0.008332   0   0   0   0
## 8112010.160626   7   5.710000E-10   8.828400   24.237500   -0.008818   0   0   0   0
## 08112010.160626   7   5.719000E-10   8.828400   24.237500   -0.008818   0   0   0   0
## 08112010.160627   7   5.719000E-10   9.014300   24.237400   -0.028564   0   0   0   NaN
## 08112010.160627   7   5.711000E-10   8.786000   24.237400   -0.008818   0   0   0   0
## 08112010.160628   7   5.702000E-10   8.786000   24.237400   -0.009141   0   0   0   0
## 08112010.160633   7   5.710000E-10   9.016200   24.237200   -0.008818   0   0   0   0
##    7   5.710000E-10   8.903400   24.237200   -0.008818   0   0   0   0
## ~~~
##
## These data require some massaging, including:
##
##  - Date/Time has a blank ('') entry that must be handled
##  - The headers are not in the correct format
##  - Tab-separated entries with additional white space
##  - NaN entries
##
## \macro_code
##
## \author Michael Marino
from __future__ import print_function

import ROOT
import sys
import os

def parse_CSV_file_with_TTree_ReadStream(tree_name, afile):


    ROOT.gROOT.SetBatch()
    # The mapping dictionary defines the proper branch names and types given a header name.
    header_mapping_dictionary = {
               'Date/Time'         : ('Datetime'       , str) ,
               'Synchro'           : ('Synchro'        , int) ,
               'Capacity'          : ('Capacitance'    , float) ,
               'Temp.Cold Head'    : ('TempColdHead'   , float) ,
               'Temp. Electrode'   : ('TempElectrode'  , float) ,
               'HV Supply Voltage' : ('HVSupplyVoltage', float) ,
               'Electrode 1'       : ('Electrode1'     , int) ,
               'Electrode 2'       : ('Electrode2'     , int) ,
               'Electrode 3'       : ('Electrode3'     , int) ,
               'Electrode 4'       : ('Electrode4'     , int) ,
                         }

    type_mapping_dictionary = {
               str   : 'C',
               int   : 'I',
               float : 'F'
                              }



    # Grab the header row of the file.  In this particular example,
    # the data are separated using tabs, but some of the header names
    # include spaces and are not generally in the ROOT expected format, e.g.
    #
    # FloatData/F:StringData/C:IntData/I
    #
    # etc.  Therefore, we grab the header_row of the file, and use
    # a python dictionary to set up the appropriate branch descriptor
    # line.

    # Open a file, grab the first line, strip the new lines
    # and split it into a list along 'tab' boundaries
    header_row        = open(afile).readline().strip().split('\t')
    # Create the branch descriptor
    branch_descriptor = ':'.join([header_mapping_dictionary[row][0]+'/'+
                           type_mapping_dictionary[header_mapping_dictionary[row][1]]
                           for row in header_row])
    #print(branch_descriptor)

    # Handling the input and output names.  Using the same
    # base name for the ROOT output file.
    output_ROOT_file_name  = os.path.splitext(afile)[0] + '.root'
    output_file            = ROOT.TFile(output_ROOT_file_name, 'recreate')
    print("Outputting %s -> %s" % (afile, output_ROOT_file_name))

    output_tree            = ROOT.TTree(tree_name, tree_name)
    file_lines             = open(afile).readlines()

    # Clean the data entries: remove the first (header) row.
    # Ensure empty strings are tagged as such since
    # ROOT doesn't differentiate between different types
    # of white space.  Therefore, we change all of these
    # entries to 'empty'.  Also, avoiding any lines that begin
    # with '#'
    file_lines     = ['\t'.join([val if (val.find(' ') == -1 and val != '')
                                else 'empty' for val in line.split('\t')])
                             for line in file_lines[1:] if line[0] != '#' ]

    # Removing NaN, setting these entries to 0.0.
    # Also joining the list of strings into one large string.
    file_as_string = ('\n'.join(file_lines)).replace('NaN', str(0.0))
    #print(file_as_string)

    # creating an istringstream to pass into ReadStream
    istring        = ROOT.istringstream(file_as_string)

    # Now read the stream
    output_tree.ReadStream(istring, branch_descriptor)

    output_file.cd()
    output_tree.Write()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: %s file_to_parse.dat" % sys.argv[0])
        sys.exit(1)
    parse_CSV_file_with_TTree_ReadStream("example_tree", sys.argv[1])

