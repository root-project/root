#!/usr/bin/env @python@
import ROOT
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('inputFile', nargs='?', default='', help = "inputFile is the input file")
parser.add_argument('-o', '--output', default='', action='store', dest='output', help='Output name')
parser.add_argument('-r', '--root', default=[], nargs='*', dest='root', help='Input root files')
parser.add_argument('-nt', '--ntuple', default='', action='store', dest='ntupla', help='Name of the ntuple')
parser.add_argument('-hs', '--histo', default=[], nargs='*', dest='histoName', help='Expression to build histograms in the form "histoName=histo if histoCut"')
args = parser.parse_args()
if (args.inputFile != '' and args.output=='' or args.root==[] or args.ntupla=='' or args.histoName==[]):
	ROOT.gInterpreter.ProcessLine("#include \"TSimpleAnalysis.h\"")
	ROOT.RunSimpleAnalysis(args.inputFile)
elif (args.inputFile=='' and args.output!='' and args.root!=[] and args.ntupla!='' and args.histoName!=[]):
	ROOT.gInterpreter.ProcessLine("#include \"TSimpleAnalysis.h\"")
	inputfile=ROOT.vector("string")(len(args.root))
	for i,s in enumerate(args.root):
		inputfile[i]=s
	expr=ROOT.vector("string")(len(args.histoName))
	for k,l in enumerate(args.histoName):
		expr[k]=l
	a = ROOT.TSimpleAnalysis(args.output, inputfile, args.ntupla, expr)
	a.Run()
else:
	print "Invalid argument set"
