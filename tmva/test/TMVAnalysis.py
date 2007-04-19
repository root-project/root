#!/usr/bin/env python
# @(#)root/tmva $Id: TMVAnalysis.py,v 1.30 2007/04/17 22:04:23 andreas.hoecker Exp $
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
DEFAULT_METHODS  = "Cuts CutsD Likelihood LikelihoodD PDERS HMatrix Fisher MLP BDT SVM_Gauss SVM_Poly SVM_Lin"

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
    
    # logon not automatically loaded through PyROOT (logon loads TMVA library) load also GUI
    gROOT.Macro( '../macros/TMVAlogon.C' )    
    gROOT.LoadMacro( '../macros/TMVAGui.C' )
    
    # import TMVA classes from ROOT
    from ROOT import TMVA

    # output file
    outputFile = TFile( outfname, 'RECREATE' )
    
    # create einstance of factory
    factory = TMVA.Factory( "MVAnalysis", outputFile, "" )
    print "**************** here "

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
        
    # Define the input variables that shall be used for the classifier training
    # note that you may also use variable expressions, such as: "3*var1/var2*abs(var3)"
    # [all types of expressions that can also be parsed by TTree::Draw( "expression" )]
    factory.AddVariable("var1+var2", 'F')
    factory.AddVariable("var1-var2", 'F')
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
    # "SplitMode=Random" means that the input events are randomly shuffled before
    # splitting them into training and test samples
    factory.PrepareTrainingAndTestTree( mycut, "NSigTrain=3000:NBkgTrain=3000:SplitMode=Random:!V" )

    # and alternative call to use a different number of signal and background training/test event is:
    # factory.PrepareTrainingAndTestTree( mycut, "NSigTrain=3000:NBkgTrain=3000:NSigTest=3000:NBkgTest=3000:SplitMode=Random:!V" )
    
    # Cut optimisation
    if "Cuts" in mlist:
        factory.BookMethod( TMVA.Types.kCuts, "Cuts", "!V:MC:EffSel:MC_NRandCuts=100000:MC_VarProp=FSmart" )

    # Cut optimisation using decorrelated input variables
    if "CutsD" in mlist:
        factory.BookMethod( TMVA.Types.kCuts, "CutsD",
                            "!V:MC:EffSel:MC_NRandCuts=200000:MC_VarProp=FSmart:VarTransform=Decorrelate" )

    # Cut optimisation with a Genetic Algorithm
    if "CutsGA" in mlist:
        factory.BookMethod( TMVA.Types.kCuts, "CutsGA",
                            "!V:GA:EffSel:GA_nsteps=40:GA_cycles=3:GA_popSize=300:GA_SC_steps=10:GA_SC_rate=5:GA_SC_factor=0.95" )

    # Likelihood
    if "Likelihood" in mlist:
        factory.BookMethod( TMVA.Types.kLikelihood, "Likelihood",
                            "!V:!TransformOutput:Spline=2:NSmooth=5:NAvEvtPerBin=50" )
        
    # test the decorrelated likelihood
    if "LikelihoodD" in mlist:
        factory.BookMethod( TMVA.Types.kLikelihood, "LikelihoodD",
                            "!V:!TransformOutput:Spline=2:NSmooth=5:NAvEvtPerBin=50:VarTransform=Decorrelate" )

    if "LikelihoodPCA" in mlist:
        factory.BookMethod( TMVA.Types.kLikelihood, "LikelihoodPCA",
                            "!V:!TransformOutput:Spline=2:NSmooth=5:NAvEvtPerBin=50:VarTransform=PCA" )

    # likelihood method with unbinned kernel estimator
    if "LikelihoodKDE" in mlist:
        factory.BookMethod( TMVA.Types.kLikelihood, "LikelihoodKDE",
                            "!V:!TransformOutput:UseKDE:KDEtype=Gauss:KDEiter=Adaptive:NAvEvtPerBin=50" )

    # Fisher - also creates PDF for MVA output (here as an example, can be used for any other classifier)
    if "Fisher" in mlist:
        factory.BookMethod( TMVA.Types.kFisher, "Fisher", "!V:Fisher:CreateMVAPdfs:NbinsMVAPdf=50:NsmoothMVAPdf=1" )

    # the new TMVA ANN: MLP (recommended ANN)
    if "MLP" in mlist:
        factory.BookMethod( TMVA.Types.kMLP, "MLP", "!V:NCycles=200:HiddenLayers=N+1,N:TestRate=5" )

    # CF(Clermont-Ferrand)ANN
    if "CFMlpANN" in mlist:
        factory.BookMethod( TMVA.Types.kCFMlpANN, "CFMlpANN", "!V:H:NCycles=500:HiddenLayers=N,N" ) 

    # Tmlp(Root)ANN
    if "TMlpANN" in mlist:
        factory.BookMethod( TMVA.Types.kTMlpANN, "TMlpANN", "!V:NCycles=200:HiddenLayers=N+1,N" )

    # HMatrix (chi2-squared) method
    if "HMatrix" in mlist:
        factory.BookMethod( TMVA.Types.kHMatrix, "HMatrix", "!V" ) 

    # PDE - RS method
    if "PDERS" in mlist:
        factory.BookMethod( TMVA.Types.kPDERS, "PDERS", 
                            "!V:VolumeRangeMode=Adaptive:KernelEstimator=Gauss:GaussSigma=0.3:NEventsMin=400:NEventsMax=600:InitialScale=0.99" )

    if "PDERSD" in mlist:
        factory.BookMethod( TMVA.Types.kPDERS, "PDERSD", 
                            "!V:VolumeRangeMode=Adaptive:KernelEstimator=Gauss:GaussSigma=0.3:NEventsMin=400:NEventsMax=600:InitialScale=0.99:VarTransform=Decorrelate" )

    if "PDERSPCA" in mlist:
        factory.BookMethod( TMVA.Types.kPDERS, "PDERSPCA", 
                            "!V:VolumeRangeMode=Adaptive:KernelEstimator=Gauss:GaussSigma=0.3:NEventsMin=400:NEventsMax=600:InitialScale=0.99:VarTransform=PCA" )

    # Boosted Decision Trees
    if "BDT" in mlist:
        factory.BookMethod( TMVA.Types.kBDT, "BDT", 
                            "!V:NTrees=400:BoostType=AdaBoost:SeparationType=GiniIndex:nEventsMin=20:nCuts=20:PruneMethod=CostComplexity:PruneStrength=4.5")

    # Decorrelated Boosted Decision Trees
    if "BDTD" in mlist:
        factory.BookMethod( TMVA.Types.kBDT, "BDTD", 
                            "!V:NTrees=400:BoostType=AdaBoost:SeparationType=GiniIndex:nEventsMin=20:nCuts=20:PruneMethod=CostComplexity:PruneStrength=4.5:VarTransform=Decorrelate")

    # Friedman's RuleFit method
    if "RuleFit" in mlist:
        factory.BookMethod( TMVA.Types.kRuleFit, "RuleFit",
                            "!V:NTrees=20:SampleFraction=-1:fEventsMin=0.1:nCuts=20:SeparationType=GiniIndex:Model=ModRuleLinear:GDTau=0.6:GDTauMin=0.0:GDTauMax=1.0:GDNTau=20:GDStep=0.01:GDNSteps=5000:GDErrScale=1.1:RuleMinDist=0.0001:MinImp=0.001" )

    # Support Vector Machine with varying kernel functions
    if "SVM_Gauss" in mlist:
      factory.BookMethod( TMVA::Types::kSVM, "SVM_Gauss",
                          "Sigma=2:C=1:Tol=0.001:Kernel=Gauss" )        
    if "SVM_Poly" in mlist:
        factory.BookMethod( TMVA::Types::kSVM, "SVM_Poly",
                            "Order=4:Theta=1:C=0.1:Tol=0.001:Kernel=Polynomial" );
    if "SVM_Lin" in mlist:
        factory.BookMethod( TMVA::Types::kSVM, "SVM_Lin",
                            "!V:Kernel=Linear:C=1:Tol=0.001" );
        
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
