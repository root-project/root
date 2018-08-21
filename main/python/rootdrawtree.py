#!/usr/bin/env @python@

# ROOT command line tools: rootdrawtree
# Author: Luca Giommi
# Mail: luca.giommi2@studio.unibo.it
# Date: 15/09/16

import sys
import argparse
import ROOT
from sys import stderr
import textwrap

def get_argparse():
	parser = argparse.ArgumentParser(
		formatter_class=argparse.RawDescriptionHelpFormatter, prog='rootdrawtree',
		description=textwrap.dedent('''\
Python script that loops over a chain to create histograms.
There are two ways to do it: one is parsing the name of the configuration file as argument,
that must have a proper syntax as shown in the class documentation of TSimpleAnalysis.

Example:
user@users-desktop:~$ rootdrawtree input_file.txt   # input_file.txt is the configuration file

The other way is to pass as arguments the name of the output file, the name of the .root input
files, the expressions (that will be shown in the histograms) and the name of the tree (that
is optional if there is only one tree inside the first .root input file).

Examples:
user@users-desktop:~$ rootdrawtree --output output.root --input hsimple.root --tree ntuple --histo 'hpxpy=px:py if px>2'
user@users-desktop:~$ rootdrawtree --output output.root --input hsimple.root hsimple2.root --histo 'hpx=px' 'hpxpy=px:py if px>2'

		'''))
	
	parser.add_argument('configFile', nargs='?', default='', help = "Configuration file")
	parser.add_argument('-o', '--output', default='', action='store', dest='output', help='Name of the output file in which will be stored the histograms')
	parser.add_argument('-i', '--input', default=[], nargs='*', dest='inputFiles', help='.root input files')
	parser.add_argument('-t', '--tree', default='', action='store', dest='tree', help='Name of the tree')
	parser.add_argument('-hs', '--histo', default=[], nargs='*', dest='histoExpr', help='Expressions to build the histograms in the form "NAME = EXPRESSION if CUT"')
	return parser

if __name__ == "__main__":
	parser = get_argparse()

	args = parser.parse_args()
	sys.argv = []

	if (args.configFile != '' and (args.output != '' or args.inputFiles != [] or args.histoExpr != [] or args.tree != '')):
		stderr.write("Error: both configuration file and options are provided \n")
		exit(1)
	if (args.configFile != ''):
		a = ROOT.TSimpleAnalysis(args.configFile)
		if a.Configure():
			a.Run()
	else:
		inputfile = ROOT.vector("string")(len(args.inputFiles))
		for i,s in enumerate(args.inputFiles):
			inputfile[i] = s
		expr = ROOT.vector("string")(len(args.histoExpr))
		for k,l in enumerate(args.histoExpr):
			expr[k]=l
		a = ROOT.TSimpleAnalysis(args.output, inputfile, expr, args.tree)
		a.Run()
