// $Id$
// Author: Joerg Stelzer   11/2007 
///////////////////////////////////////////////////////////////////////////////////
//
//  TMVA functionality test suite
//  =============================
//
//  This program performs tests of TMVA
//
//   To run the program do: 
//   stressTMVA          : run standard test
//
///////////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "TString.h"
#include "TFile.h"
#include "TMath.h"

#include "TMVA/Factory.h"

class TMVATest {
public:
   
   enum tmvatests { CutsGA = 0x001, Likelihood = 0x002, LikelihoodPCA = 0x004, PDERS   = 0x008,
                    KNN    = 0x010, HMatrix    = 0x020, Fisher        = 0x040, FDA_MT  = 0x080,
                    MLP    = 0x100, SVM_Gauss  = 0x200, BDT           = 0x400, RuleFit = 0x800 };

   TMVATest() :
      fFactory(0),
      fTestDataFileName( "http://root.cern.ch/files/tmva_example.root" ),
      fReferenceDataFileName( "http://root.cern.ch/files/TMVAref.root" ),
      //fReferenceDataFileName( "TMVAref.root" ),
      fOutputFileName( "TMVAtest.root" ),
      fInputfile(0),
      fOutputfile(0),
      fReffile(0),
      fTests(0),
      fNEvents(500)
   {
      fTests |= CutsGA;
      fTests |= Likelihood;
      fTests |= LikelihoodPCA;
      fTests |= PDERS;
      fTests |= KNN;
      fTests |= HMatrix;
      fTests |= Fisher;
      fTests |= FDA_MT;
      fTests |= MLP;
      fTests |= SVM_Gauss;
      fTests |= BDT;
      fTests |= RuleFit;

      fOutputfile = TFile::Open( fOutputFileName, "RECREATE" );
      fFactory = new TMVA::Factory( "TMVAtest", fOutputfile, "S" );
   }

   ~TMVATest() {
      fOutputfile->Close();
      delete fFactory;
   }

   bool Initialize() {
      // accessing the data file via the web from
      fInputfile = TFile::Open( fTestDataFileName );

      if (!fInputfile) {
         fErrorMsg.push_back(TString(Form("ERROR: could not open data file %s", fTestDataFileName.Data() )));
         return false;
      }

      TTree *signal     = (TTree*)fInputfile->Get("TreeS");
      TTree *background = (TTree*)fInputfile->Get("TreeB");

      // global event weights (see below for setting event-wise weights)
      Double_t signalWeight     = 1.0;
      Double_t backgroundWeight = 1.0;

      fFactory->AddSignalTree    ( signal,     signalWeight );
      fFactory->AddBackgroundTree( background, backgroundWeight );

      fFactory->AddVariable("var1+var2", 'F');
      fFactory->AddVariable("var1-var2", 'F');
      fFactory->AddVariable("var3", 'F');
      fFactory->AddVariable("var4", 'F');

      // Apply additional cuts on the signal and background samples (can be different)
      TCut mycuts = ""; // for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
      TCut mycutb = ""; // for example: TCut mycutb = "abs(var1)<0.5";

      // tell the factory to use all remaining events in the trees after training for testing:
      fFactory->PrepareTrainingAndTestTree( mycuts, mycutb, 
                                            Form("NSigTrain=%i:NBkgTrain=%i:NSigTest=%i:NBkgTest=%i:SplitMode=Random:NormMode=NumEvents:V",
                                                 fNEvents,fNEvents,fNEvents,fNEvents));

      fInputfile->Close();

      // Cut optimisation with genetic algorithm

      if( fTests & CutsGA )
         fFactory->BookMethod( TMVA::Types::kCuts, "CutsGA",
                               "!H:!V:FitMethod=GA:EffSel:Steps=30:Cycles=3:PopSize=100:SC_steps=10:SC_rate=5:SC_factor=0.95:VarProp=FSmart" );

      // Likelihood
      if( fTests & Likelihood )
         fFactory->BookMethod( TMVA::Types::kLikelihood, "Likelihood", 
                               "!H:!V:!TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=10:NSmoothBkg[0]=10:NSmoothBkg[1]=10:NSmooth=10:NAvEvtPerBin=50" ); 

      // test the principal component analysis
      if( fTests & LikelihoodPCA )
         fFactory->BookMethod( TMVA::Types::kLikelihood, "LikelihoodPCA", 
                               "!H:!V:!TransformOutput:PDFInterpol=Spline2:NSmoothSig[0]=10:NSmoothBkg[0]=10:NSmooth=5:NAvEvtPerBin=50:VarTransform=PCA" ); 

      // PDE - RS method
      if( fTests & PDERS )
         fFactory->BookMethod( TMVA::Types::kPDERS, "PDERS", 
                               "!H:!V:NormTree=T:VolumeRangeMode=Adaptive:KernelEstimator=Gauss:GaussSigma=0.3:NEventsMin=400:NEventsMax=600" );

      // K-Nearest Neighbour classifier (KNN)
      if( fTests & KNN )
         fFactory->BookMethod( TMVA::Types::kKNN, "KNN", 
                               "!H:!V:nkNN=400:TreeOptDepth=6:ScaleFrac=0.8:!UseKernel:!Trim" );

      // H-Matrix (chi2-squared) method
      if( fTests & HMatrix )
         fFactory->BookMethod( TMVA::Types::kHMatrix, "HMatrix",
                               "!H:!V" ); 

      // Fisher discriminant
      if( fTests & Fisher )
         fFactory->BookMethod( TMVA::Types::kFisher, "Fisher", 
                               "!H:!V:Normalise:CreateMVAPdfs:Fisher:NbinsMVAPdf=50:NsmoothMVAPdf=1" );    

      // Function discrimination analysis (FDA) -- test of various fitters - the recommended one is Minuit or GA
      if( fTests & FDA_MT )
         fFactory->BookMethod( TMVA::Types::kFDA, "FDA_MT",
                               "!H:!V:Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1);(-10,10);(-10,10);(-10,10);(-10,10):FitMethod=MINUIT:ErrorLevel=1:PrintLevel=-1:FitStrategy=2:UseImprove:UseMinos:SetBatch" );

      // TMVA ANN: MLP (recommended ANN) -- all ANNs in TMVA are Multilayer Perceptrons
      if( fTests & MLP )
         fFactory->BookMethod( TMVA::Types::kMLP, "MLP",
                               "!H:!V:Normalise:NCycles=200:HiddenLayers=N+1,N:TestRate=5" );

      // Support Vector Machines using three different Kernel types (Gauss, polynomial and linear)
      if( fTests & SVM_Gauss )
         fFactory->BookMethod( TMVA::Types::kSVM, "SVM_Gauss",
                               "!H:!V:Sigma=2:C=1:Tol=0.001:Kernel=Gauss" );

      // Boosted Decision Trees (second one with decorrelation)
      if( fTests & BDT )
         fFactory->BookMethod( TMVA::Types::kBDT, "BDT", 
                               "!H:!V:NTrees=400:BoostType=AdaBoost:SeparationType=GiniIndex:nCuts=20:PruneMethod=CostComplexity:PruneStrength=4.5" );

      // RuleFit -- TMVA implementation of Friedman's method
      if( fTests & RuleFit )
         fFactory->BookMethod( TMVA::Types::kRuleFit, "RuleFit",
                               "!H:!V:RuleFitModule=RFTMVA:Model=ModRuleLinear:MinImp=0.001:RuleMinDist=0.001:NTrees=20:fEventsMin=0.01:fEventsMax=0.5:GDTau=-1.0:GDTauPrec=0.01:GDStep=0.01:GDNSteps=10000:GDErrScale=1.02" );

      return kTRUE;
   }

   bool TrainAndEval() {
      // Train MVAs using the set of training events
      fFactory->TrainAllMethods();
			
      // ---- Evaluate all MVAs using the set of test events
      fFactory->TestAllMethods();
			
      // ----- Evaluate and compare performance of all configured MVAs
      fFactory->EvaluateAllMethods();

      return kTRUE;
   }

   void PrintError() {
      for(std::vector<TString>::iterator it = fErrorMsg.begin();
          it != fErrorMsg.end(); it++ )
         std::cout << *it << std::endl;
      fErrorMsg.clear();
   }

   bool CompareResult() {
      fReffile = TFile::Open(fReferenceDataFileName);

      if (!fReffile) {
         fErrorMsg.push_back(TString(Form("ERROR: could not find reference file %s", fReferenceDataFileName.Data() )));
         return false;
      }

      bool success = kTRUE;

      if( fTests & CutsGA        ) success &= Check("Cuts"         , "CutsGA");
      if( fTests & Likelihood    ) success &= Check("Likelihood"   , "");
      if( fTests & LikelihoodPCA ) success &= Check("Likelihood"   , "LikelihoodPCA");
      if( fTests & PDERS         ) success &= Check("PDERS"        , "");
      if( fTests & KNN           ) success &= Check("KNN"          , "");
      if( fTests & HMatrix       ) success &= Check("HMatrix"      , "");
      if( fTests & Fisher        ) success &= Check("Fisher"       , "");
      if( fTests & FDA_MT        ) success &= Check("FDA"          , "FDA_MT");
      if( fTests & MLP           ) success &= Check("MLP"          , "");
      if( fTests & SVM_Gauss     ) success &= Check("SVM"          , "SVM_Gauss");
      if( fTests & BDT           ) success &= Check("BDT"          , "");
      if( fTests & RuleFit       ) success &= Check("RuleFit"      , "");       

      fReffile->Close();
      return success;
   }

   bool Check(const TString& classname, TString instancename="") {
      if(instancename=="") instancename = classname;
      TString hist(Form("Method_%s/%s/MVA_%s_effBvsS",classname.Data(),instancename.Data(),instancename.Data()));
      TH1 * effBvsS[2];
      effBvsS[0] = (TH1*)fOutputfile->Get(hist);
      effBvsS[1] = (TH1*)fReffile->Get(hist);
      if(!effBvsS[0]) {
         fErrorMsg.push_back(TString("Can not find test histogram ") + hist);
      }
      if(!effBvsS[1]) {
         fErrorMsg.push_back(TString("Can not find reference histogram ") + hist);
      }
      if(!effBvsS[0] || !effBvsS[1]) return kFALSE;
         
      Float_t beff[2][2], beffErr[2][2];
      Float_t distance[2]; // difference between test and reference
      Int_t nEvts[2]; nEvts[0] = fNEvents; nEvts[1] = 3000;  
      Int_t signalEff[2] = {50,80};
      for( Int_t y=0; y<2; y++) { // y=0: 50, y=1:80
         for(Int_t x=0; x<2; x++ ) { // x=0: test, x=1: reference
            beff[y][x] = effBvsS[x]->GetBinContent(signalEff[y]);
            beffErr[y][x] = TMath::Sqrt(beff[y][x]*(1-beff[y][x])/nEvts[x]);
         }
         distance[y] = TMath::Abs(beff[y][0]-beff[y][1])/TMath::Sqrt(beffErr[y][0]*beffErr[y][0]+beffErr[y][1]*beffErr[y][1]);
      }


      bool success = kTRUE;
      if(distance[0]>5) {
         fErrorMsg.push_back(TString(Form("Method %s shows disagreement of %g sigma (background efficiency at 50%% signal efficiency)",
                                          classname.Data(),distance[0])));
         std::cout << beff[0][0] << " +/- " << beffErr[0][0] << " disagrees with reference " << beff[0][1] << " +/- " << beffErr[0][1] << " (" << distance[0] << ")" << std::endl;
         success = kFALSE;
      } else {
         //std::cout << beff[0][0] << " +/- " << beffErr[0][0] << " agrees with reference " << beff[0][1] << " +/- " << beffErr[0][1] << " (" << distance[0] << ")" << std::endl;
      }
      if(distance[1]>5) {
         fErrorMsg.push_back(TString(Form("Method %s shows disagreement of %g sigma (background efficiency at 80%% signal efficiency)",
                                          classname.Data(),distance[1])));
         std::cout << beff[1][0] << " +/- " << beffErr[1][0] << " disagrees with reference " << beff[1][1] << " +/- " << beffErr[1][1] << " (" << distance[1] << ")" << std::endl;
         success = kFALSE;
      } else {
         //std::cout << beff[1][0] << " +/- " << beffErr[1][0] << " agrees with reference " << beff[1][1] << " +/- " << beffErr[1][1] << " (" << distance[1] << ")" << std::endl;
      }
      return success;
   }

private:
   TMVA::Factory        *fFactory;
   TString               fTestDataFileName;
   TString               fReferenceDataFileName;
   TString               fOutputFileName;
   TFile                *fInputfile;
   TFile                *fOutputfile;
   TFile                *fReffile;
   Int_t                 fTests;
   Int_t                 fNEvents;
   std::vector<TString>  fErrorMsg;
};

int stressTMVA() {
   TMVATest testsuite;

   std::cout << "******************************************************************" << std::endl;
   std::cout << "* TMVA - S T R E S S suite *" << std::endl;
   std::cout << "******************************************************************" << std::endl;

   // Prepare the dataset
   if( ! testsuite.Initialize() ) {
      std::cout << "Test 1: Data Preparation ............... FAILED" << std::endl;
      testsuite.PrintError();
      return 1;
   }

   std::cout << "Test 1: Data Preparation ............... OK" << std::endl;

   // Train MVAs using the set of training events
   if( ! testsuite.TrainAndEval() ) {
      std::cout << "Test 2: Training Methods ............... FAILED" << std::endl;
      testsuite.PrintError();
      return 1;
   }

   std::cout << "Test 2: Training Methods ............... OK" << std::endl;

   // Compare output
   if( ! testsuite.CompareResult() ) {
      std::cout << "Test 3: Compare to Reference ........... FAILED" << std::endl;
      testsuite.PrintError();
      return 1;
   }

   std::cout << "Test 3: Compare to Reference ........... OK" << std::endl;

   return 0;
}

int main() {
   return stressTMVA();
}
