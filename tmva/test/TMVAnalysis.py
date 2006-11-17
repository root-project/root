#!/usr/bin/env python
# @(#)root/tmva $Id: TMVAnalysis.py,v 1.3 2006/11/15 11:00:51 andreas.hoecker Exp $
# ------------------------------------------------------------------------------ #
# Project      : TMVA - a Root-integrated toolkit for multivariate data analysis #
# Package      : TMVA                                                            #
# Python script: TMVAnalysis.py                                                  #
#                                                                                #
# This python script gives an example on training and testing of several         #
# Multivariate Analyser (MVA) methods through PyROOT. Note that PyROOT requires  #
# that you have a python version > 2.2 installed on your computer.               #
#                                                                                #
# As input file we use a toy MC sample (you find it in TMVA/examples/data)       #
#                                                                                #
# The methods to be used can be switched on and off by means of booleans.        #
#                                                                                #
# The output file "TMVA.root" can be analysed with the use of dedicated          #
# macros (simply say: root -l <../macros/macro.C>), which can be conveniently    #
# invoked through a GUI that will appear at the end of the run of this macro.    #
#                                                                                #
# for help type "python TMVAnalysis.py --help"                                   #
# ------------------------------------------------------------------------------ #

# --------------------------------------------
# standard python import
import sys    # exit
import time   # time accounting
import getopt # command line parser

# --------------------------------------------

# default settings for command line arguments
DEFAULT_OUTFNAME = "TMVA.root"
DEFAULT_INFNAME  = "../examples/data/toy_sigbkg.root"
DEFAULT_TREESIG  = "TreeS"
DEFAULT_TREEBKG  = "TreeB"
DEFAULT_METHODS  = "Cuts CutsD Likelihood LikelihoodD PDERS HMatrix Fisher MLP BDT"

# print help
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

# main routine
def main():

    try:
        # retrive command line options
        shortopts  = "m:i:t:o:vh?"
        longopts   = ["methods=", "inputfile=", "inputtrees=", "outputfile=", "verbose", "help", "usage"]
        opts, args = getopt.getopt( sys.argv[1:], shortopts, longopts )

    except getopt.GetoptError:
        # print help information and exit:
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

    # print methods
    mlist = methods.split(' ')
    print "=== TMVAnalysis: use methods..."
    for m in mlist:
        if m != '':
            print "=== ... <%s>" % m

    # import ROOT classes
    from ROOT import gSystem, gROOT, gApplication, TFile, TTree, TCut
    
    # load TMVA library and GUI
    gSystem.Load( 'libTMVA.1.so' )
    gROOT.LoadMacro( '../macros/TMVAGui.C' )
    
    # import TMVA classes from ROOT
    from ROOT import TMVA

    # output file
    outputFile = TFile( outfname, 'RECREATE' )
    
    # create einstance of factory
    factory = TMVA.Factory( "MVAnalysis", outputFile, "" )

    # set verbosity
    factory.SetVerbose( verbose )
    
    # read input data
    if not gSystem.AccessPathName( infname ):
        input = TFile( infname )
    else:
        print "ERROR: could not access data file %s\n" % infname

    signal      = input.Get( treeNameSig )
    background  = input.Get( treeNameBkg )
    
    # global event weights (see below for setting event-wise weights)
    signalWeight     = 1.0
    backgroundWeight = 1.0
    
    if not factory.SetInputTrees( signal, background, signalWeight, backgroundWeight ):
        print "ERROR: could not set input trees\n"
        sys.exit(1)
        
    # Define the input variables that shall be used for the MVA training
    # note that you may also use variable expressions, such as: "3*var1/var2*abs(var3)"
    # [all types of expressions that can also be parsed by TTree::Draw( "expression" )]
    factory.AddVariable("var1", 'F')
    factory.AddVariable("var2", 'F')
    factory.AddVariable("var3", 'F')
    factory.AddVariable("var4", 'F')
    
    # This would set individual event weights (the variables defined in the 
    # expression need to exist in the original TTree)
    # factory->SetWeightExpression("weight1*weight2")
    #
    # Apply additional cuts on the signal and background sample. 
    # Assumptions on size of training and testing sample:
    #    a) equal number of signal and background events is used for training
    #    b) any numbers of signal and background events are used for testing
    #    c) an explicit syntax can violate a)
    # more Documentation with the Factory class
    # example for cut: mycut = TCut( "abs(var1)<0.5 && abs(var2-0.5)<1" )
    mycut = TCut( "" ) 
    
    # here, the relevant variables are copied over in new, slim trees that are
    # used for TMVA training and testing
    factory.PrepareTrainingAndTestTree( mycut, 2000, 4000 )  
    
    # Cut optimisation
    if methods.find( "Cuts" ) != -1:
        factory.BookMethod( TMVA.Types.Cuts, "Cuts", "!V:MC:EffSel:MC_NRandCuts=100000:AllFSmart" )
        
    # Cut optimisation with a Genetic Algorithm
    if methods.find( "Cuts" ) != -1:
        factory.BookMethod( TMVA.Types.Cuts, "CutsGA",
                            "!V:GA:EffSel:GA_nsteps=40:GA_cycles=30:GA_popSize=100:GA_SC_steps=10:GA_SC_offsteps=5:GA_SC_factor=0.95" )
        
    # Cut optimisation using decorrelated input variables
    if methods.find( "CutsD" ) != -1:
        factory.BookMethod( TMVA.Types.Cuts, "CutsD", "!V:MC:EffSel:MC_NRandCuts=200000:AllFSmart:Preprocess=Decorrelate" )
            
    # Likelihood
    if methods.find( "Likelihood" ) != -1:
        factory.BookMethod( TMVA.Types.Likelihood, "Likelihood", "!V:!TransformOutput:Spline=2:NSmooth=5" ) 
        
    # test the decorrelated likelihood
    if methods.find( "LikelihoodD" ) != -1:
        factory.BookMethod( TMVA.Types.Likelihood, "LikelihoodD", "!V:!TransformOutput:Spline=2:NSmooth=5:Preprocess=Decorrelate") 

    # Fisher:
    if methods.find( "Fisher" ) != -1:
        factory.BookMethod( TMVA.Types.Fisher, "Fisher", "!V:Fisher" )    

    # the new TMVA ANN: MLP (recommended ANN)
    if methods.find( "MLP" ) != -1:
        factory.BookMethod( TMVA.Types.MLP, "MLP", "!V:NCycles=200:HiddenLayers=N+1,N:TestRate=5" )

    # CF(Clermont-Ferrand)ANN
    if methods.find( "CFMlpANN" ) != -1:
        factory.BookMethod( TMVA.Types.CFMlpANN, "CFMlpANN", "!V:H:NCycles=5000:HiddenLayers=N,N"  ) # n_cycles:#nodes:#nodes:...  

    # Tmlp(Root)ANN
    if methods.find( "TMlpANN" ) != -1:
        factory.BookMethod( TMVA.Types.TMlpANN, "TMlpANN", "!V:NCycles=200:HiddenLayers=N+1,N"  ) # n_cycles:#nodes:#nodes:...

    # HMatrix
    if methods.find( "HMatrix" ) != -1:
        factory.BookMethod( TMVA.Types.HMatrix, "HMatrix", "!V" ) # H-Matrix (chi2-squared) method

    # PDE - RS method
    if methods.find( "PDERS" ) != -1:
        factory.BookMethod( TMVA.Types.PDERS, "PDERS", 
                            "!V:VolumeRangeMode=RMS:KernelEstimator=Teepee:MaxVIterations=50:InitialScale=0.99" ) 
    if methods.find( "PDERSD" ) != -1:
        factory.BookMethod( TMVA.Types.PDERS, "PDERSD", 
                            "!V:VolumeRangeMode=RMS:KernelEstimator=Teepee:MaxVIterations=50:InitialScale=0.99:Preprocess=Decorrelate" ) 

    # Boosted Decision Trees
    if methods.find( "BDT" ) != -1:
        factory.BookMethod( TMVA.Types.BDT, "BDT", 
                            "!V:NTrees=400:BoostType=AdaBoost:SeparationType=GiniIndex:nEventsMin=20:SignalFraction=0.:nCuts=20:PruneMethod=CostComplexity:PruneStrength=3.5" );
   
    # Friedman's RuleFit method
    if methods.find( "RuleFit" ) != -1:
        factory.BookMethod( TMVA.Types.RuleFit, "RuleFit", 
                            "!V:NTrees=20:SampleFraction=-1:nEventsMin=60:nCuts=20:MinImp=0.001:Model=ModLinear:GDTau=0.6:GDStep=0.01:GDNSteps=100000:SeparationType=GiniIndex:RuleMaxDist=0.00001" )

    # Bayesian classifier
    if methods.find( "BayesClassifier" ) != -1:
        factory.BookMethod( TMVA.Types.BayesClassifier, "BayesClassifier", "!V:myOptions" )

    # ---- Now you can tell the factory to train, test, and evaluate the MVAs. 

    # Train MVAs
    factory.TrainAllMethods()
    
    # Test MVAs
    factory.TestAllMethods()
    
    # Evaluate MVAs
    factory.EvaluateAllMethods()    
    
    # Save the output.
    outputFile.Close()
    
    # clean up
    factory.IsA().Destructor( factory )
    
    print "=== wrote root file %s\n" % outfname
    print "=== TMVAnalysis is done!\n"
    
    # open the GUI for the result macros    
    gROOT.ProcessLine( "TMVAGui(\"%s\")" % outfname );
    
    # keep the ROOT thread running
    gApplication.Run() 

# ----------------------------------------------------------

if __name__ == "__main__":
    main()
