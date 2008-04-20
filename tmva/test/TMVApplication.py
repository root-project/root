#!/usr/bin/env python
# @(#)root/tmva $Id$
# ------------------------------------------------------------------------------ #
# Project      : TMVA - a Root-integrated toolkit for multivariate data analysis #
# Package      : TMVA                                                            #
# Python script: TMVApplication.py                                               #
#                                                                                #
# This macro provides a simple example on how to use the trained classifiers     #
# within an analysis module                                                      #
#                                                                                #
# for help type "python TMVApplication.py --help"                                #
# ------------------------------------------------------------------------------ #

# --------------------------------------------
# Standard python imports
import sys              # exit
import time             # time accounting
import getopt           # command line parser
import math             # some math...
from array import array # used to reference variables in TMVA and TTrees

# --------------------------------------------

# Default settings for command line arguments
DEFAULT_OUTFNAME = "TMVA.root"
DEFAULT_INFNAME  = "tmva_example.root"
DEFAULT_TREESIG  = "TreeS"
DEFAULT_TREEBKG  = "TreeB"
DEFAULT_METHODS  = "CutsGA,Likelihood,LikelihoodPCA,PDERS,KNN,HMatrix,Fisher,FDA_MT,MLP,SVM_Gauss,BDT,BDTD,RuleFit"

# Print usage help
def usage():
    print " "
    print "Usage: python %s [options]" % sys.argv[0]
    print "  -m | --methods    : gives methods to be run (default: all methods)"
    print "  -i | --inputfile  : name of input ROOT file (default: '%s')" % DEFAULT_INFNAME
    print "  -o | --outputfile : name of output ROOT file containing results (default: '%s')" % DEFAULT_OUTFNAME
    print "  -t | --inputtrees : input ROOT Trees for signal and background (default: '%s %s')" \
        % (DEFAULT_TREESIG, DEFAULT_TREEBKG)
    print "  -v | --verbose"
    print "  -? | --usage      : print this help message"
    print "  -h | --help       : print this help message"
    print " "

# Main routine
def main():

    try:
        # Retrive command line options
        shortopts  = "m:i:t:o:vh?"
        longopts   = ["methods=", "inputfile=", "inputtrees=", "outputfile=", "verbose", "help", "usage"]
        opts, args = getopt.getopt( sys.argv[1:], shortopts, longopts )

    except getopt.GetoptError:
        # Print help information and exit:
        print "ERROR: unknown options in argument %s" % sys.argv[1:]
        usage()
        sys.exit(1)

    infname     = DEFAULT_INFNAME
    treeNameSig = DEFAULT_TREESIG
    treeNameBkg = DEFAULT_TREEBKG
    outfname    = DEFAULT_OUTFNAME
    methods     = DEFAULT_METHODS
    verbose     = False
    for o, a in opts:
        if o in ("-?", "-h", "--help", "--usage"):
            usage()
            sys.exit(0)
        elif o in ("-m", "--methods"):
            methods = a
        elif o in ("-i", "--inputfile"):
            infname = a
        elif o in ("-o", "--outputfile"):
            outfname = a
        elif o in ("-t", "--inputtrees"):
            a.strip()
            trees = a.rsplit( ' ' )
            trees.sort()
            trees.reverse()
            if len(trees)-trees.count('') != 2:
                print "ERROR: need to give two trees (each one for signal and background)"
                print trees
                sys.exit(1)

            treeNameSig = trees[0]
            treeNameBkg = trees[1]
        elif o in ("-v", "--verbose"):
            verbose = True

    # Print methods
    mlist = methods.replace(' ',',').split(',')
    print "=== TMVApplication: use method(s)..."
    for m in mlist:
        if m.strip() != '':
            print "=== - <%s>" % m.strip()

    # Import ROOT classes
    from ROOT import gSystem, gROOT, gApplication, TFile, TTree, TCut, TH1F, TStopwatch
    
    # check ROOT version, give alarm if 5.18
    if gROOT.GetVersionCode() >= 332288 and gROOT.GetVersionCode() < 332544:
        print "*** You are running ROOT version 5.18, which has problems in PyROOT such that TMVA"
        print "*** does not run properly (function calls with enums in the argument are ignored)."
        print "*** Solution: either use CINT or a C++ compiled version (see TMVA/macros or TMVA/examples),"
        print "*** or use another ROOT version (e.g., ROOT 5.19)."
        sys.exit(1)
    
    # Logon not automatically loaded through PyROOT (logon loads TMVA library) load also GUI
    gROOT.SetMacroPath( "../macros/" )
    gROOT.Macro       ( "../macros/TMVAlogon.C" )    

    # Import TMVA classes from ROOT
    from ROOT import TMVA

    # Create the Reader object
    reader = TMVA.Reader("!Color")
    
    # Create a set of variables and declare them to the reader
    # - the variable names must corresponds in name and type to 
    # those given in the weight file(s) that you use
    
    # what to do ???
    var1 = array( 'f', [ 0 ] )
    var2 = array( 'f', [ 0 ] )
    var3 = array( 'f', [ 0 ] )
    var4 = array( 'f', [ 0 ] )
    reader.AddVariable( "var1+var2", var1 )
    reader.AddVariable( "var1-var2", var2 )
    reader.AddVariable( "var3",      var3 )
    reader.AddVariable( "var4",      var4 )
    
    # book the MVA methods
    dir    = "weights/"
    prefix = "TMVAnalysis_"
    
    for m in mlist:
        reader.BookMVA( m + " method", dir + prefix + m + ".weights.txt" )
    
    #######################################################################
    # For an example how to apply your own plugin method, please see
    # TMVA/macros/TMVApplication.C
    #######################################################################

    # Book output histograms    
    nbin = 80

    histList = []
    for m in mlist:
        histList.append( TH1F( m, m, nbin, -3, 3 ) )

    # Book example histogram for probability (the other methods would be done similarly)
    if "Fisher" in mlist:
        probHistFi   = TH1F( "PROBA_MVA_Fisher",  "PROBA_MVA_Fisher",  nbin, 0, 1 )
        rarityHistFi = TH1F( "RARITY_MVA_Fisher", "RARITY_MVA_Fisher", nbin, 0, 1 )

    # Prepare input tree (this must be replaced by your data source)
    # in this example, there is a toy tree with signal and one with background events
    # we'll later on use only the "signal" events for the test in this example.
    #   
    fname = "./tmva_example.root"   
    print "--- Accessing data file: %s" % fname 
    input = TFile.Open( fname )
    if not input:
        print "ERROR: could not open data file: %s" % fname
        sys.exit(1)

    #
    # Prepare the analysis tree
    # - here the variable names have to corresponds to your tree
    # - you can use the same variables as above which is slightly faster,
    #   but of course you can use different ones and copy the values inside the event loop
    #
    print "--- Select signal sample"
    theTree = input.Get("TreeS")
    userVar1 = array( 'f', [0] )
    userVar2 = array( 'f', [0] )
    theTree.SetBranchAddress( "var1", userVar1 )
    theTree.SetBranchAddress( "var2", userVar2 )
    theTree.SetBranchAddress( "var3", var3 )
    theTree.SetBranchAddress( "var4", var4 )

    # Efficiency calculator for cut method
    nSelCuts   = 0
    effS       = 0.7

    # Process the events
    print "--- Processing: %i events" % theTree.GetEntries()
    sw = TStopwatch()
    sw.Start()
    for ievt in range(theTree.GetEntries()):

      if ievt%1000 == 0:
         print "--- ... Processing event: %i" % ievt 

      # Fill event in memory
      theTree.GetEntry(ievt)

      # Compute MVA input variables
      var1[0] = userVar1[0] + userVar2[0]
      var2[0] = userVar1[0] - userVar2[0]
    
      # Return the MVAs and fill to histograms
      if "CutsGA" in mlist:
          passed = reader.EvaluateMVA( "CutsGA method", effS )
          if passed:
              nSelCuts = nSelCuts + 1

      # Fill histograms with MVA outputs
      for h in histList:
          h.Fill( reader.EvaluateMVA( h.GetName() + " method" ) )

      # Retrieve probability instead of MVA output
      if "Fisher" in mlist:
          probHistFi  .Fill( reader.GetProba ( "Fisher method" ) )
          rarityHistFi.Fill( reader.GetRarity( "Fisher method" ) )

    # Get elapsed time
    sw.Stop()
    print "--- End of event loop: %s" % sw.Print()

    # Return computed efficeincies
    if "CutsGA" in mlist:   
        eff  = float(nSelCuts)/theTree.GetEntries()
        deff = math.sqrt(eff*(1.0-eff)/theTree.GetEntries())
        print "--- Signal efficiency for Cuts method : %.5g +- %.5g (required was: %.5g)" % (eff, deff, effS)

        # Test: retrieve cuts for particular signal efficiency
        mcuts = reader.FindMVA( "CutsGA method" )        
        cutsMin = array( 'd', [0, 0, 0, 0] )        
        cutsMax = array( 'd', [0, 0, 0, 0] )
        mcuts.GetCuts( 0.7, cutsMin, cutsMax )
        print "--- -------------------------------------------------------------" 
        print "--- Retrieve cut values for signal efficiency of 0.7 from Reader" 
        for ivar in range(4):
            print "... Cut: %.5g < %s <= %.5g" % (cutsMin[ivar], reader.GetVarName(ivar), cutsMax[ivar])

        print "--- -------------------------------------------------------------" 

    #
    # write histograms
    #
    target  = TFile( "TMVApp.root","RECREATE" )
    for h in histList:
        h.Write()

    # Write also probability hists
    if "Fisher" in mlist: 
        probHistFi.Write() 
        rarityHistFi.Write() 

    target.Close()

    print "--- Created root file: \"TMVApp.root\" containing the MVA output histograms"   
    print "==> TMVApplication is done!"  
 
if __name__ == '__main__':
    main()
