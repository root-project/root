
#include "gtest/gtest.h"

#include <TFile.h>
#include <TMath.h>
#include <TTree.h>
#include <TString.h>
#include <TSystem.h>

#include "TMVA/CrossValidation.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Reader.h"
#include "TMVA/Tools.h"

#include <chrono>
#include <vector>

constexpr UInt_t NUM_FOLDS = 2;
constexpr UInt_t NUM_EVENTS = 100;
constexpr UInt_t NUM_EVENTS_SIG = NUM_EVENTS / 2;
constexpr UInt_t NUM_EVENTS_BKG = NUM_EVENTS - NUM_EVENTS_SIG;

/**
 * Generates two slightly offset gaussians and returns a non-owning pointer
 * to a TTree.
 */
TTree *genTree(Int_t nPoints, Double_t offset, Double_t scale = 0.3, UInt_t seed = 100)
{
   TRandom3 rng(seed);
   Float_t x = 0;
   Float_t y = 0;
   UInt_t id = 0;

   auto data = new TTree();
   data->Branch("x", &x, "x/F");
   data->Branch("y", &y, "y/F");
   data->Branch("EventNumber", &id, "EventNumber/i");

   for (Int_t n = 0; n < nPoints; ++n) {
      x = rng.Gaus(offset, scale);
      y = rng.Gaus(offset, scale);
      data->Fill();
      ++id;
   }

   // Important: Disconnects the tree from the memory locations of x and y.
   data->ResetBranchAddresses();
   return data;
}

/**
 * Trains a TMVA method using the specified number of worker processes.
 * @param  numWorkers Number of processes to use for parallel fold execution.
 * @return            The path to the weight file where the method was stored
 *                    and the time it took to execute the evaluation, exluding
 *                    data generation, in seconds.
 */
std::pair<std::string, double> runCrossValidation(UInt_t numWorkers)
{
   using cv_clock_t = std::chrono::high_resolution_clock;
   using cv_milli_t = std::chrono::duration<double, std::milli>;

   TTree *sigTree = genTree(NUM_EVENTS_SIG, 0.3, 0.3, 100);
   TTree *bkgTree = genTree(NUM_EVENTS_BKG, 0.3, 0.3, 101);

   auto *dataloader = new TMVA::DataLoader("cv-multiproc");
   dataloader->AddSignalTree(sigTree);
   dataloader->AddBackgroundTree(bkgTree);

   dataloader->AddVariable("x", 'D');
   dataloader->AddVariable("y", 'D');
   dataloader->AddSpectator("EventNumber", 'I');

   TString dataloaderOptions = Form("SplitMode=Block:nTrain_Signal=%i"
                                    ":nTrain_Background=%i:!V",
                                    NUM_EVENTS_SIG, NUM_EVENTS_BKG);
   dataloader->PrepareTrainingAndTestTree("", dataloaderOptions);

   // TMVA::CrossValidation takes ownership of dataloader
   std::string splitExpr = "UInt_t([EventNumber])%UInt_t([NumFolds])";
   TMVA::CrossValidation cv{Form("%i-proc", numWorkers), dataloader,
                            Form("!Silent:AnalysisType=Classification"
                                 ":NumWorkerProcs=%i:NumFolds=%i"
                                 ":SplitType=Deterministic:SplitExpr=%s",
                                 numWorkers, NUM_FOLDS, splitExpr.c_str())};

   cv.BookMethod(TMVA::Types::kBDT, "BDT", "!H:!V:NTrees=100:MaxDepth=3");

   auto StartTime = cv_clock_t::now();
   cv.Evaluate();
   auto EndTime = cv_clock_t::now();
   double duration = cv_milli_t(EndTime - StartTime).count() / 1000.0;

   delete sigTree;
   delete bkgTree;

   std::string dsname{dataloader->GetName()};
   std::string cvname{cv.GetName()};
   std::string weightPath = dsname + "/weights/" + cvname + "_BDT.weights.xml";

   return std::make_pair(weightPath, duration);
}

/**
 * Verify that the two methods generate the same output when presented the same
 * input.
 * @param methodA Path to a method weight file.
 * @param methodB Another path to method weight file.
 */
void verify(std::string methodA, std::string methodB)
{
   TMVA::Reader reader;
   Float_t x;
   Float_t y;
   UInt_t uid;
   Float_t fid;

   reader.AddVariable("x", &x);
   reader.AddVariable("y", &y);
   reader.AddSpectator("EventNumber", &fid);

   reader.BookMVA("BDT1", methodA.c_str());
   reader.BookMVA("BDT2", methodB.c_str());

   TTree *tree = genTree(NUM_EVENTS, 0.3, 0.3, 200);

   tree->SetBranchAddress("x", &x);
   tree->SetBranchAddress("y", &y);
   tree->SetBranchAddress("EventNumber", &uid);

   for (Long64_t ievt = 0; ievt < NUM_EVENTS; ievt++) {
      tree->GetEntry(ievt);

      // Convert TTree UInt_t to Float_t of TMVA
      fid = uid;

      Float_t valA = reader.EvaluateMVA("BDT1");
      Float_t valB = reader.EvaluateMVA("BDT2");

      ASSERT_EQ(valA, valB) << "Output when using a single process and multiple"
                               " processes differ! >:(";
   }
}

TEST(CrossValidationMultiprocess, EqualOutputTo)
{
   // Generate two methods, one created with a single process, the other in
   // parallel using TProcessExecutor.
   auto p1 = runCrossValidation(1);
   auto p2 = runCrossValidation(2);

   // Print duration
   double duration1 = std::get<1>(p1);
   double duration2 = std::get<1>(p2);
   std::cout << "CV with 1 process took  : " << duration1 << std::endl;
   std::cout << "CV with 2 processes took: " << duration2 << std::endl;

   // Verify that the two models generate the same output given the same input.
   std::string weightPath1 = std::get<0>(p1);
   std::string weightPath2 = std::get<0>(p2);

   // AccessPathName() == kFALSE means file exits
   ASSERT_FALSE(gSystem->AccessPathName(weightPath1.c_str())) << "Method was"
      << " not serialised correctly. Path: '" << weightPath1 << "' does not"
      << " exist.";
   ASSERT_FALSE(gSystem->AccessPathName(weightPath2.c_str())) << "Method was"
      << " not serialised correctly. Path: '" << weightPath2 << "' does not"
      << " exist.";

   verify(weightPath1, weightPath2);
}
