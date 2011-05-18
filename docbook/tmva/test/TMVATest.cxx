/**********************************************************************************
 * Project   : TMVA - a Root-integrated toolkit for multivariate data analysis    *
 * Package   : TMVA                                                               *
 * Exectuable: TMVRead                                                            *
 *                                                                                *
 * This exectutable provides an example on JUST testing of several                   *
 * Multivariate Analyser (MVA) methods using previously written weight files      *
 * on a given test tree  (set of test events), which must of course contain the   *
 * variables named as in the weight file                                          *
 *                                                                                   *
 *                                                                                  *
 * The methods to be used can be switched on and off by means of the boolians          *
 * below                                                                          *
 *                                                                                  *
 * The output file "TMVA.root" can be analysed with the use of dedicated          *
 * macros (simply say: root -l <macro.C>) :                                          *
 *                                                                                  *
 * ../macros/variables.C  ==> shows the MVA input variables for signal and backgr  *
 * ../macros/correlations.C  ==> shows the correlations between the MVA input vars *
 * ../macros/mvas.C          ==> shows the trained MVAs for the test events          *
 * ../macros/efficiencies.C  ==> shows the background rejections versus signal effs*
 *                         for all MVAs used                                          *
 *                                                                                  *
 *                                                                                *
 **********************************************************************************/

#include <iostream> // Stream declarations
#include "TMVA/Factory.h"
#include "TMVA/MethodBDT.h"
#include "TMVA/MethodCuts.h"
#include "TMVA/MethodLikelihood.h"
#include "TMVA/MethodCFMlpANN.h"
#include "TMVA/MethodFisher.h"
#include "TMVA/MethodPDERS.h"
#include "TMVA/MethodHMatrix.h"
#include "TMVA/MethodTMlpANN.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TObjString.h"
#include "TMVA/Tools.h"

using namespace std;

// ---------------------------------------------------------------
// choose MVA methods to be tested
Bool_t  Use_Cuts             = 0;
TString weightFileCuts       = "weights/TMVAnalysis_Cuts.weights";
Bool_t  Use_Likelihood       = 0;
TString weightFileLikelihood = "weights/TMVAnalysis_Likelihood.weights";
Bool_t  Use_LikelihoodD      = 0;
TString weightFileLikelihoodD= "weights/TMVAnalysis_LikelihoodD.weights";
Bool_t  Use_PDERS            = 0;
TString weightFilePDERS      = "weights/TMVAnalysis_PDERS.weights.root";
Bool_t  Use_HMatrix          = 0;
TString weightFileHMatrix    = "weights/TMVAnalysis_HMatrix.weights";
Bool_t  Use_Fisher           = 0;
TString weightFileFisher     = "weights/TMVAnalysis_Fisher.weights";
Bool_t  Use_CFMlpANN         = 0;
TString weightFileCFMlpANN   = "weights/TMVAnalysis_CFMlpANN.weights";
Bool_t  Use_TMlpANN          = 0;
TString weightFileTMlpANN    = "weights/TMVAnalysis_TMlpANN.weights";
Bool_t  Use_BDTGini          = 1; 
TString weightFileBDTGini    = "weights/TMVAnalysis_BDTGini.weights";
Bool_t  Use_BDTCD            = 0; 
TString weightFileBDTCE      = "weights/TMVAnalysis_BDTCE.weights";
Bool_t  Use_BDTStatSig       = 0; 
TString weightFileBDTStatSig = "weights/TMVAnalysis_BDTStatSig.weights";
Bool_t  Use_BDTMisCL         = 0; 
TString weightFileBDTMisCl   = "weights/TMVAnalysis_BDTMisCl.weights";
// ---------------------------------------------------------------
Bool_t EvaluateVariables  = 0; // perform evaluation for each input variable
// ---------------------------------------------------------------

int main( int argc, char** argv ) 
{
  cout << endl;
  cout << "==> start TMVRead" << endl;

  // ---- define the root output file
  char *outFile = new char[160];
  char *inFile = new char[160];
  outFile="TMVATest.root";
  inFile="dummy.root";

  if (argc<2) {
    cout << "ERROR!!!!!!"<<endl;
    cout << "    You need to provide the input file (with the test tree) and optionally "<<endl;
    cout << "    the name of the output file (defaut TMVRead.root) " << endl;
    cout << "ERROR!!!!!!"<<endl;
    exit(1);
  }
  else{
    inFile = argv[1];
  }

  if (argc>2) outFile = argv[2];
  // you want the TestTree in the output file... as it seems IMPOSSIBLE to 
  // copy (clone) a tree without knowing what is inside, I simply copy the 
  // whole file
  char syscall[160];
  sprintf (syscall,"cp %s %s ",inFile,outFile);
  system(syscall);
  TFile* target = TFile::Open( outFile, "UPDATE" );

  //attention, the first STRING given here defined the TAG of the
  //weight files, that will be looked at. Have a look at in the directory
  // weights/ and you'll understand what I mean.
  TMVA::Factory *factory = new TMVA::Factory( target ) ;

  vector<TString>* inputVars = new vector<TString>;
  inputVars->push_back("var1");
  inputVars->push_back("var2");
  inputVars->push_back("var3");
  inputVars->push_back("var4");

  if (Use_BDTGini)
    factory->BookMethod(new TMVA_MethodBDT(inputVars, weightFileBDTGini));

  TFile *testTreeFile = new TFile(inFile);
  TTree *testTree = (TTree*) testTreeFile->Get("TestTree");



  factory->SetTestTree(testTree);


  // test MVAs
  factory->TestAllMethods();

  // evaluate MVAs
  factory->EvaluateAllMethods();    

  // ---- terminate job

  // close output file
  target->Close();

  // clean up
  delete factory;

  cout << "==> wrote root file " <<   TString(outFile) <<  endl;
  cout << "==> TMVTest is done!" << endl;      
}
