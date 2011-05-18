// @(#)root/tmva $Id$
/**********************************************************************************
 * Project   : TMVA - a Root-integrated toolkit for multivariate data analysis    *
 * Package   : TMVA                                                               *
 * Exectuable: TMVAGAexample                                                        *
 *                                                                                *
 * This exectutable gives an example of a very simple use of the genetic algorithm*
 * of TMVA                                                                        *
 *                                                                                *
 **********************************************************************************/

#include <iostream> // Stream declarations
#include <vector>
#include <limits>

#include "TChain.h"
#include "TCut.h"
#include "TDirectory.h"
#include "TH1F.h"
#include "TH1.h"
#include "TMath.h"
#include "TFile.h"
#include "TStopwatch.h"
#include "TROOT.h"

#include "TMVA/GeneticAlgorithm.h"
#include "TMVA/GeneticFitter.h"
#include "TMVA/IFitterTarget.h"
#include "TMVA/Factory.h"
#include "TMVA/Reader.h"

using namespace std;

namespace TMVA {

// =========== Description =============
// This example shows the training of signal with three different backgrounds
// Then in the application a tree is created with all signal and background events where the true class ID and the three classifier outputs are added
// finally with the application tree, the significance is maximized with the help of the TMVA genetic algrorithm.
//


// ------------------------------ Training ----------------------------------------------------------------
void Training(){
   std::string factoryOptions( "!V:!Silent:Transformations=I;D;P;G,D:AnalysisType=Classification" );
   TString fname = "./tmva_example_multiple_background.root";

   TFile *input(0);
   input = TFile::Open( fname );

   TTree *signal      = (TTree*)input->Get("TreeS");
   TTree *background0 = (TTree*)input->Get("TreeB0");
   TTree *background1 = (TTree*)input->Get("TreeB1");
   TTree *background2 = (TTree*)input->Get("TreeB2");

   /// global event weights per tree (see below for setting event-wise weights)
   Double_t signalWeight      = 1.0;
   Double_t background0Weight = 1.0;
   Double_t background1Weight = 1.0;
   Double_t background2Weight = 1.0;

   // Create a new root output file.
   TString outfileName( "TMVASignalBackground0.root" );
   TFile* outputFile = TFile::Open( outfileName, "RECREATE" );



   // ===================== background 0
   TMVA::Factory *factory = new TMVA::Factory( "TMVAMultiBkg0", outputFile, factoryOptions );
   factory->AddVariable( "var1", "Variable 1", "", 'F' );
   factory->AddVariable( "var2", "Variable 2", "", 'F' );
   factory->AddVariable( "var3", "Variable 3", "units", 'F' );
   factory->AddVariable( "var4", "Variable 4", "units", 'F' );

   factory->AddSignalTree    ( signal,     signalWeight       );
   factory->AddBackgroundTree( background0, background0Weight );

//   factory->SetBackgroundWeightExpression("weight");
   TCut mycuts = ""; // for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
   TCut mycutb = ""; // for example: TCut mycutb = "abs(var1)<0.5";

   // tell the factory to use all remaining events in the trees after training for testing:
   factory->PrepareTrainingAndTestTree( mycuts, mycutb,
                                        "nTrain_Signal=0:nTrain_Background=0:SplitMode=Random:NormMode=NumEvents:!V" );

   // Boosted Decision Trees
   factory->BookMethod( TMVA::Types::kBDT, "BDTG",
			"!H:!V:NTrees=1000:BoostType=Grad:Shrinkage=0.30:UseBaggedGrad:GradBaggingFraction=0.6:SeparationType=GiniIndex:nCuts=20:NNodesMax=5" );
   factory->TrainAllMethods();
   factory->TestAllMethods();
   factory->EvaluateAllMethods();

   outputFile->Close();

   delete factory;



   //  ===================== background 1

   outfileName = "TMVASignalBackground1.root";
   outputFile = TFile::Open( outfileName, "RECREATE" );

   factory = new TMVA::Factory( "TMVAMultiBkg1", outputFile, factoryOptions );
   factory->AddVariable( "var1", "Variable 1", "", 'F' );
   factory->AddVariable( "var2", "Variable 2", "", 'F' );
   factory->AddVariable( "var3", "Variable 3", "units", 'F' );
   factory->AddVariable( "var4", "Variable 4", "units", 'F' );

   factory->AddSignalTree    ( signal,     signalWeight       );
   factory->AddBackgroundTree( background1, background1Weight );

//   factory->SetBackgroundWeightExpression("weight");

   // tell the factory to use all remaining events in the trees after training for testing:
   factory->PrepareTrainingAndTestTree( mycuts, mycutb,
                                        "nTrain_Signal=0:nTrain_Background=0:SplitMode=Random:NormMode=NumEvents:!V" );

   // Boosted Decision Trees
   factory->BookMethod( TMVA::Types::kBDT, "BDTG",
			"!H:!V:NTrees=1000:BoostType=Grad:Shrinkage=0.30:UseBaggedGrad:GradBaggingFraction=0.6:SeparationType=GiniIndex:nCuts=20:NNodesMax=5" );
   factory->TrainAllMethods();
   factory->TestAllMethods();
   factory->EvaluateAllMethods();

   outputFile->Close();

   delete factory;




   //  ===================== background 2
   outfileName = "TMVASignalBackground2.root";
   outputFile = TFile::Open( outfileName, "RECREATE" );

   factory = new TMVA::Factory( "TMVAMultiBkg2", outputFile, factoryOptions );
   factory->AddVariable( "var1", "Variable 1", "", 'F' );
   factory->AddVariable( "var2", "Variable 2", "", 'F' );
   factory->AddVariable( "var3", "Variable 3", "units", 'F' );
   factory->AddVariable( "var4", "Variable 4", "units", 'F' );

   factory->AddSignalTree    ( signal,     signalWeight       );
   factory->AddBackgroundTree( background2, background2Weight );

//   factory->SetBackgroundWeightExpression("weight");

   // tell the factory to use all remaining events in the trees after training for testing:
   factory->PrepareTrainingAndTestTree( mycuts, mycutb,
                                        "nTrain_Signal=0:nTrain_Background=0:SplitMode=Random:NormMode=NumEvents:!V" );

   // Boosted Decision Trees
   factory->BookMethod( TMVA::Types::kBDT, "BDTG",
			"!H:!V:NTrees=1000:BoostType=Grad:Shrinkage=0.30:UseBaggedGrad:GradBaggingFraction=0.5:SeparationType=GiniIndex:nCuts=20:NNodesMax=5" );
   factory->TrainAllMethods();
   factory->TestAllMethods();
   factory->EvaluateAllMethods();

   outputFile->Close();

   delete factory;
   
}





// ------------------------------ Application ----------------------------------------------------------------
// create a summary tree with all signal and background events and for each event the three classifier values and the true classID
void ApplicationCreateCombinedTree(){

   // Create a new root output file.
   TString outfileName( "tmva_example_multiple_backgrounds__applied.root" );
   TFile* outputFile = TFile::Open( outfileName, "RECREATE" );
   TTree* outputTree = new TTree("multiBkg","multiple backgrounds tree");

   Float_t var1, var2;
   Float_t var3, var4;
   Int_t   classID = 0;
   Float_t weight = 1.f;
   
   Float_t classifier0, classifier1, classifier2;

   outputTree->Branch("classID", &classID, "classID/I");
   outputTree->Branch("var1", &var1, "var1/F");
   outputTree->Branch("var2", &var2, "var2/F");
   outputTree->Branch("var3", &var3, "var3/F");
   outputTree->Branch("var4", &var4, "var4/F");
   outputTree->Branch("weight", &weight, "weight/F");
   outputTree->Branch("cls0", &classifier0, "cls0/F");
   outputTree->Branch("cls1", &classifier1, "cls1/F");
   outputTree->Branch("cls2", &classifier2, "cls2/F");


   // ===== create three readers for the three different signal/background classifications, .. one for each background
   TMVA::Reader *reader0 = new TMVA::Reader( "!Color:!Silent" );
   TMVA::Reader *reader1 = new TMVA::Reader( "!Color:!Silent" );
   TMVA::Reader *reader2 = new TMVA::Reader( "!Color:!Silent" );

   reader0->AddVariable( "var1", &var1 );
   reader0->AddVariable( "var2", &var2 );
   reader0->AddVariable( "var3", &var3 );
   reader0->AddVariable( "var4", &var4 );

   reader1->AddVariable( "var1", &var1 );
   reader1->AddVariable( "var2", &var2 );
   reader1->AddVariable( "var3", &var3 );
   reader1->AddVariable( "var4", &var4 );

   reader2->AddVariable( "var1", &var1 );
   reader2->AddVariable( "var2", &var2 );
   reader2->AddVariable( "var3", &var3 );
   reader2->AddVariable( "var4", &var4 );

   // ====== load the weight files for the readers
   TString method =  "BDT method";
   reader0->BookMVA( "BDT method", "weights/TMVAMultiBkg0_BDTG.weights.xml" );
   reader1->BookMVA( "BDT method", "weights/TMVAMultiBkg1_BDTG.weights.xml" );
   reader2->BookMVA( "BDT method", "weights/TMVAMultiBkg2_BDTG.weights.xml" );

   // ===== load the input file
   TFile *input(0);
   TString fname = "./tmva_example_multiple_background.root";
   input = TFile::Open( fname );

   TTree* theTree = NULL;

   // ===== loop through signal and all background trees
   for( int treeNumber = 0; treeNumber < 4; ++treeNumber ) {
      if( treeNumber == 0 ){
	 theTree = (TTree*)input->Get("TreeS");
	 std::cout << "--- Select signal sample" << std::endl;
//	 theTree->SetBranchAddress( "weight", &weight );
	 weight = 1;
	 classID = 0;
      }else if( treeNumber == 1 ){
	 theTree = (TTree*)input->Get("TreeB0");
	 std::cout << "--- Select background 0 sample" << std::endl;
//	 theTree->SetBranchAddress( "weight", &weight );
	 weight = 1;
	 classID = 1;
      }else if( treeNumber == 2 ){
	 theTree = (TTree*)input->Get("TreeB1");
	 std::cout << "--- Select background 1 sample" << std::endl;
//	 theTree->SetBranchAddress( "weight", &weight );
	 weight = 1;
	 classID = 2;
      }else if( treeNumber == 3 ){
	 theTree = (TTree*)input->Get("TreeB2");
	 std::cout << "--- Select background 2 sample" << std::endl;
//	 theTree->SetBranchAddress( "weight", &weight );
	 weight = 1;
	 classID = 3;
      }


      theTree->SetBranchAddress( "var1", &var1 );
      theTree->SetBranchAddress( "var2", &var2 );
      theTree->SetBranchAddress( "var3", &var3 );
      theTree->SetBranchAddress( "var4", &var4 );


      std::cout << "--- Processing: " << theTree->GetEntries() << " events" << std::endl;
      TStopwatch sw;
      sw.Start();
      Int_t nEvent = theTree->GetEntries();
//      Int_t nEvent = 100;
      for (Long64_t ievt=0; ievt<nEvent; ievt++) {

	 if (ievt%1000 == 0){
	    std::cout << "--- ... Processing event: " << ievt << std::endl;
	 }

	 theTree->GetEntry(ievt);
      
	 // ==== get the classifiers for each of the signal/background classifications
	 classifier0 = reader0->EvaluateMVA( method );
	 classifier1 = reader1->EvaluateMVA( method );
	 classifier2 = reader2->EvaluateMVA( method );

	 outputTree->Fill();
      }


      // get elapsed time
      sw.Stop();
      std::cout << "--- End of event loop: "; sw.Print();
   }
   input->Close();


   // write output tree
//   outputTree->SetDirectory(outputFile);
//   outputTree->Write();
   outputFile->Write();

   outputFile->Close();

   std::cout << "--- Created root file: \"" << outfileName.Data() << "\" containing the MVA output histograms" << std::endl;
  
   delete reader0;
   delete reader1;
   delete reader2;
    
   std::cout << "==> Application of readers is done! combined tree created" << std::endl << std::endl;
   
}




// ------------------------------ Genetic Algorithm Fitness definition ----------------------------------------------------------------
class MyFitness : public IFitterTarget {
public:
   // constructor
   MyFitness( TChain* _chain ) : IFitterTarget() {
      chain = _chain;

      hSignal = new TH1F("hsignal","hsignal",100,-1,1);
      hFP = new TH1F("hfp","hfp",100,-1,1);
      hTP = new TH1F("htp","htp",100,-1,1);

      TString cutsAndWeightSignal  = "weight*(classID==0)";
      nSignal = chain->Draw("Entry$/Entries$>>hsignal",cutsAndWeightSignal,"goff");
      weightsSignal = hSignal->Integral();

   }
       
   // the output of this function will be minimized
   Double_t EstimatorFunction( std::vector<Double_t> & factors ){

      TString cutsAndWeightTruePositive  = Form("weight*((classID==0) && cls0>%f && cls1>%f && cls2>%f )",factors.at(0), factors.at(1), factors.at(2));
      TString cutsAndWeightFalsePositive = Form("weight*((classID >0) && cls0>%f && cls1>%f && cls2>%f )",factors.at(0), factors.at(1), factors.at(2));
	  
      // Entry$/Entries$ just draws something reasonable. Could in principle anything
      Float_t nTP = chain->Draw("Entry$/Entries$>>htp",cutsAndWeightTruePositive,"goff");
      Float_t nFP = chain->Draw("Entry$/Entries$>>hfp",cutsAndWeightFalsePositive,"goff");

      weightsTruePositive = hTP->Integral();
      weightsFalsePositive = hFP->Integral();

      efficiency = 0;
      if( weightsSignal > 0 )
	 efficiency = weightsTruePositive/weightsSignal;
	  
      purity = 0;
      if( weightsTruePositive+weightsFalsePositive > 0 )
	 purity = weightsTruePositive/(weightsTruePositive+weightsFalsePositive);

      Float_t effTimesPur = efficiency*purity;

      Float_t toMinimize = std::numeric_limits<float>::max(); // set to the highest existing number 
      if( effTimesPur > 0 ) // if larger than 0, take 1/x. This is the value to minimize
	 toMinimize = 1./(effTimesPur); // we want to minimize 1/efficiency*purity

      // Print();

      return toMinimize;
   }


   void Print(){
      std::cout << std::endl;
      std::cout << "======================" << std::endl
		<< "Efficiency : " << efficiency << std::endl
		<< "Purity     : " << purity << std::endl << std::endl
		<< "True positive weights : " << weightsTruePositive << std::endl
		<< "False positive weights: " << weightsFalsePositive << std::endl
		<< "Signal weights        : " << weightsSignal << std::endl;
   }

   Float_t nSignal;

   Float_t efficiency;
   Float_t purity;
   Float_t weightsTruePositive;
   Float_t weightsFalsePositive;
   Float_t weightsSignal;


private:
   TChain* chain;
   TH1F* hSignal;
   TH1F* hFP;
   TH1F* hTP;

};








// ------------------------------ call of Genetic algorithm  ----------------------------------------------------------------
void MaximizeSignificance(){
       
        // define all the parameters by their minimum and maximum value
        // in this example 3 parameters (=cuts on the classifiers) are defined. 
        vector<Interval*> ranges;
        ranges.push_back( new Interval(-1,1) ); // for some classifiers (especially LD) the ranges have to be taken larger
        ranges.push_back( new Interval(-1,1) );
        ranges.push_back( new Interval(-1,1) );

	std::cout << "Classifier ranges (defined by the user)" << std::endl;
        for( std::vector<Interval*>::iterator it = ranges.begin(); it != ranges.end(); it++ ){
           std::cout << " range: " << (*it)->GetMin() << "   " << (*it)->GetMax() << std::endl;
        }

	TChain* chain = new TChain("multiBkg");
	chain->Add("tmva_example_multiple_backgrounds__applied.root");

        IFitterTarget* myFitness = new MyFitness( chain );

        // prepare the genetic algorithm with an initial population size of 20
        // mind: big population sizes will help in searching the domain space of the solution
        // but you have to weight this out to the number of generations
        // the extreme case of 1 generation and populationsize n is equal to 
        // a Monte Carlo calculation with n tries

        const TString name( "multipleBackgroundGA" );
        const TString opts( "PopSize=100:Steps=30" );

        GeneticFitter mg( *myFitness, name, ranges, opts);
	// mg.SetParameters( 4, 30, 200, 10,5, 0.95, 0.001 );

        std::vector<Double_t> result;
        Double_t estimator = mg.Run(result);

	dynamic_cast<MyFitness*>(myFitness)->Print();
	std::cout << std::endl;

	int n = 0;
	for( std::vector<Double_t>::iterator it = result.begin(); it<result.end(); it++ ){
	   std::cout << "  cutValue[" << n << "] = " << (*it) << ";"<< std::endl;
	   n++;
	}

	
}


} // namespace TMVA


// ------------------------------ Run all ----------------------------------------------------------------
int main( int argc, char** argv ) 
{
   cout << "Start Test TMVAGAexample" << endl
        << "========================" << endl
        << endl;

   gROOT->ProcessLine(".L createData.C+");
   gROOT->ProcessLine("create_MultipleBackground(2000)");


   cout << endl;
   cout << "========================" << endl;
   cout << "--- Training" << endl;
   TMVA::Training();

   cout << endl;
   cout << "========================" << endl;
   cout << "--- Application & create combined tree" << endl;
   TMVA::ApplicationCreateCombinedTree();

   cout << endl;
   cout << "========================" << endl;
   cout << "--- maximize significance" << endl;
   TMVA::MaximizeSignificance();
}
