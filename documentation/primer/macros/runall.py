#! /usr/bin/env python

macros = [\
"slit.C",
"write_ntuple_to_file_advanced.C",
"write_ntuple_to_file.C",
"write_to_file.C",
"ExampleMacro.C",
"ExampleMacro_GUI.C",
"makeMySelector.C",
"RunMySelector.C",
"macro1.C",
"macro2.C",
"macro3.C",
"macro4.C",
"macro5.C",
"macro6.C",
"macro7.C",
"macro8.C",
"macro9.C",
"read_from_file.C",
"read_ntuple_from_file.C",
"read_ntuple_with_chain.C"]

import os
import sys

for mName in macros:
    command = "root -b -l -q %s" %mName
    if mName == "slits.C": command = 'echo "2 4" | %s' %command
    print "\n ******* Running %s" %mName
    if 0 !=os.system(command):
       print "Macro %s" %mName
       sys.exit(1)
print "\n"+"-"*80+"\nAll macros ran successfully"
sys.exit(0)
 
