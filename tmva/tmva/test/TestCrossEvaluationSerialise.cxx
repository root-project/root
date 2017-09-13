// @(#)root/tmva $Id$
// Author: Kim Albertsson

/*************************************************************************
 * Copyright (C) 2017, Kim Albertsson
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TMVA/CrossEvaluation.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Reader.h"
#include "TMVA/Types.h"

#include "TFile.h"
#include "TMath.h"
#include "TRandom.h"
#include "TTree.h"

#include <iostream>
#include <memory>
#include <stdexcept>


// 
// Sets up a method for cross evaluation and generates output through the 
// Factory::EvaluationAllMethods. It then recreates the method using the TMVA
// Reader and verifies that the application phase produces the same output
// using the factory produced TMVA.root as input.
// 
// TODO: Make a proper google test of it
// 

class TestContex {
public:
   virtual ~TestContex() {
      if (fDataFile  ) { fDataFile->Close();   }
      if (fOutputFile) { fOutputFile->Close(); }
   }

   void genData(UInt_t nPoints, UInt_t seed = 100);
   void setUpCe();
   void runCe();
   void setUpApplicationPhase(TString methodName);
   void runApplicationPhase(TString methodName);
   void verifyApplicationPhase();

private:
   const TString fDataFileName   = ( "data.root" );
   const TString fOutputFileName = ( "TMVA.root" );

   std::unique_ptr<TMVA::CrossEvaluation> fCrossEvaluate;
   std::shared_ptr<TFile>                 fDataFile;
   std::shared_ptr<TFile>                 fOutputFile;
   std::unique_ptr<TMVA::Reader>          fReader;

   Float_t fX;
   Float_t fY;
   Float_t fevNum;
   Float_t fMvaEval;

   std::vector<Double_t> fEvaluationResults;
   std::vector<Double_t> fApplicationResults;
};





// =============================================================================
// === DATAGEN ===
// =============================================================================

std::pair<Double_t, Double_t> generateCirclePoint (
   Float_t r,
   Float_t rsig  ,
   Float_t phimin,
   Float_t phimax,
   Float_t phisig,
   TRandom & rng
) {
   Double_t phi  = rng.Rndm() * (phimax - phimin) + phimin;
            phi += rng.Gaus() * phisig;

   r += rng.Gaus()*rsig;

   Double_t x = r * cos( TMath::DegToRad() * phi );
   Double_t y = r * sin( TMath::DegToRad() * phi );

   return std::pair<Double_t, Double_t> (x, y);
}

void createCircTree(Int_t nPoints, Double_t radius, Double_t rsig, TString name, UInt_t seed = 100)
{
   TRandom rng(seed);
   std::pair<Double_t, Double_t> p;
   static UInt_t id = 0;
   Double_t x = 0;
   Double_t y = 0;

   TTree * data = new TTree(name, name);
   data->Branch("x", &x, "x/D");
   data->Branch("y", &y, "y/D");
   data->Branch("EventNumber", &id, "EventNumber/I");

   for (Int_t n = 0; n < nPoints; ++n) {
      p = generateCirclePoint( radius, rsig, 0, 160, 0.1, rng );
      x = std::get<0>(p);
      y = std::get<1>(p);
      data->Fill();

      id++;
   }
   
   data->Write();
}

void TestContex::genData(UInt_t nPoints, UInt_t seed)
{
   TFile* dataFile = TFile::Open( fDataFileName, "RECREATE" );
   createCircTree(nPoints/2, 3.0, 0.20, "Signal"    , seed);
   createCircTree(nPoints/2, 3.5, 0.20, "Background", seed+1);
   dataFile->Close();
}

// === END DATAGEN ===
// =============================================================================






// =============================================================================
// === RUNNING CROSS EVALUATION ===
// =============================================================================

void TestContex::setUpCe()
{
   fDataFile = std::shared_ptr<TFile>(new TFile( fDataFileName ));
   TTree * treeClass0 = (TTree *)fDataFile->Get("Signal");
   TTree * treeClass1 = (TTree *)fDataFile->Get("Background");

   // Dataloader managed by CrossEvaluation
   TMVA::DataLoader * dataloader = new TMVA::DataLoader("dataset");

   dataloader->AddVariable( "x", 'D' );
   dataloader->AddVariable( "y", 'D' );
   dataloader->AddSpectator( "EventNumber", "EventNumber", "" );
   
   dataloader->AddSignalTree( treeClass0 );
   dataloader->AddBackgroundTree( treeClass1 );

   fOutputFile    = std::shared_ptr<TFile>(new TFile( fOutputFileName, "RECREATE" ));
   fCrossEvaluate = std::unique_ptr<TMVA::CrossEvaluation>(
      new TMVA::CrossEvaluation(dataloader, fOutputFile.get(), "AnalysisType=Classification:SplitExpr=int([EventNumber])%int([NumFolds])")
   );
   fCrossEvaluate->BookMethod(TMVA::Types::kBDT, "BDT", "!H:!V:NTrees=10");
}

void TestContex::runCe()
{
   fCrossEvaluate->Evaluate();
   fOutputFile->Flush();
}

// === END CROSS EVALUATION ===
// =============================================================================






// =============================================================================
// === APPLICATION PHASE ===
// =============================================================================

void TestContex::setUpApplicationPhase(TString methodName)
{
   fReader = std::unique_ptr<TMVA::Reader>(new TMVA::Reader( "!Color:!Silent" ));

   fReader->AddVariable(  "x" , &fX  );
   fReader->AddVariable(  "y" , &fY  );
   fReader->AddSpectator( "EventNumber", &fevNum );

   TString weightfile = TString("dataset/weights/") + methodName + TString(".weights.xml");
   fReader->BookMVA( methodName, weightfile );
}

void TestContex::runApplicationPhase(TString methodName)
{
   TTree* tree = (TTree *)fOutputFile->Get("dataset/TestTree");
   tree->SetBranchAddress( "x" , &fX  );
   tree->SetBranchAddress( "y" , &fY  );
   tree->SetBranchAddress( "EventNumber", &fevNum );
   tree->SetBranchAddress( "BDT", &fMvaEval );

   fApplicationResults.reserve( fApplicationResults.size() + tree->GetEntries() );
   for (Long64_t ievt=0; ievt<tree->GetEntries(); ievt++) {
      tree->GetEntry(ievt);
      
      Double_t val = fReader->EvaluateMVA( methodName );
      fApplicationResults.push_back(val);
      
      fEvaluationResults.push_back(fMvaEval);
   }
}

void TestContex::verifyApplicationPhase()
{
   if (fEvaluationResults.size() != fApplicationResults.size()) {
      throw std::runtime_error("Size not equal!");
   }

   for (UInt_t iEvent = 0; iEvent < fEvaluationResults.size(); ++iEvent) {

      std::cout << "eval:appl -- " << fEvaluationResults[iEvent] << ":" << fApplicationResults[iEvent] << std::endl;

      if (not TMath::AreEqualAbs(fEvaluationResults[iEvent], fApplicationResults[iEvent], 1e-5)) {
         throw std::runtime_error("Output not equal!");
      }
   }
}

// =============================================================================
// === END APPLICATION PHASE ===
// =============================================================================






void TestCeSerialise()
{
   TestContex tc;
   tc.genData(100);
   tc.setUpCe();
   tc.runCe(); // serialises CE method if modelPersitance
   tc.setUpApplicationPhase("_CrossEvaluation_BDT");
   tc.runApplicationPhase("_CrossEvaluation_BDT");
   tc.verifyApplicationPhase();
}


int main()
{
   TestCeSerialise();
   return 0;
}