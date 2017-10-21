/// \file
/// - Project   : TMVA - a Root-integrated toolkit for multivariate data analysis
/// - Package   : TMVA
/// - Exectuable: TMVACrossEvaluation
/// 

#include "gtest/gtest.h"

#include "TFile.h"
#include "TMath.h"
#include "TTree.h"
#include "TString.h"
#include "TSystem.h"

#include "TMVA/DataLoader.h"
#include "TMVA/DataSet.h"
#include "TMVA/DataSetInfo.h"
#include "TMVA/CrossEvaluation.h"
#include "TMVA/Tools.h"

#include <algorithm>
#include <vector>

using id_vec_t      = std::vector<UInt_t>;
using data_t        = std::tuple<id_vec_t, TTree *>;
using fold_id_vec_t = std::vector<id_vec_t>;

namespace TMVA {

/*
 * Creates data for use by the test. Returns a tuple of a TTree and an id
 * vector. The TTree can be used by a dataloader to create a DataSet. The id 
 * vector can be used to verify that the splitting was done correctly.
 *
 * \param nPoints Number of data points to generate
 * \param start   Start value for the id, can be used to differentiate trees.
 */
data_t createData(Int_t nPoints, UInt_t start)
{
   std::vector<UInt_t> ids;

   UInt_t id = start;
   Double_t x = 0.;

   TTree * data = new TTree();
   data->Branch("x" , &x, "x/D");
   data->Branch("id", &id, "id/I");

   for (Int_t n = 0; n < nPoints; ++n) {
      data->Fill();
      ids.push_back(id);
      x += 0.1;
      id++;
   }
   
   return std::make_tuple(ids, data);
}

/*
 * Checks whether all points in a set (train/test etc) can be found in the original
 * generated data.
 *
 * \param ds  Input DataSet to verify
 * \param ids Original data to verify against
 * \param tt  Populated set to check (kTraining or kTesting). A populated set is
 *            what is stored in the DataSet after PrepareFoldDataSet has been called. 
 */
void testSetInOrig(DataSet * ds, id_vec_t ids, Types::ETreeType tt) {
   for (auto & ev : ds->GetEventCollection(tt)) {
      UInt_t spectator_id = ev->GetSpectators().at(0);

      auto idx_it = std::find(std::begin(ids), std::end(ids), spectator_id);
      EXPECT_NE(idx_it, std::end(ids)) << "Event with id: " << spectator_id << " not found in original data.";
   }
}

/*
 * Checks whether the datapoints contained in DataSet are the same as in original data.
 *
 * The folds are split into equal sized parts, that means that if the number of events
 * is not evenly divisible with the number of folds some events will be skipped.
 * 
 * \param ds  Input DataSet to verify
 * \param ids Original data to verify against
 */
void testSetsEqOrig(DataSet * ds, id_vec_t ids, UInt_t numFolds) {
   std::vector<UInt_t> fold_ids;
   for (auto & ev : ds->GetEventCollection(Types::kTraining)) {
      fold_ids.push_back(ev->GetSpectators().at(0));
   }
   for (auto & ev : ds->GetEventCollection(Types::kTesting)) {
      fold_ids.push_back(ev->GetSpectators().at(0));
   }

   std::sort(fold_ids.begin(), fold_ids.end());
   std::sort(ids.begin()     , ids.end());

   // Check that the number of elements in dataset is equal to number of elements in orig data.
   EXPECT_EQ(fold_ids.size(), ids.size()) << "Number of events in original dataset and the split data set differ.";

   // And that they contain the same data
   for (UInt_t i = 0; i < ids.size(); ++i) {
      ASSERT_EQ(ids[i], fold_ids[i]) << "Original dataset and split dataset contains different data.";
   }
}

/*
 * Checks wether a fold has been prepared successfully. Only does the split on the
 * training set, (d->PrepareFoldDataSet(iFold, Types::kTraining);)
 */
bool testFold(DataLoader * d, UInt_t iFold, UInt_t numFolds, id_vec_t ids) {
   DataSet * ds = d->GetDataSetInfo().GetDataSet();
   d->PrepareFoldDataSet(iFold, Types::kTraining);

   // Are folds of correct sizes?
   // TODO: This assumes that all folds are of the same size. This is not neccearily true,
   // if we input 21 events in two folds one fold will be 11 points and the other 10 points.
   // The test case should take this into account
   const UInt_t nPoints           = ids.size();
   // const UInt_t nPointsPerFold    = nPoints/numFolds;
   // const UInt_t nPointsInTrainSet = nPointsPerFold*(numFolds-1);
   // // std::cout << "Points in input dataset: " << ids.size() << std::endl;
   // // std::cout << "Points used for CE     : " << ids.size()-numSkippedEvents << std::endl;
   // // EXPECT_EQ(nPointsInTrainSet, ds->GetEventCollection(Types::kTraining).size());
   // // EXPECT_EQ(nPointsPerFold   , ds->GetEventCollection(Types::kTesting).size());

   testSetInOrig(ds, ids, Types::kTraining);
   testSetInOrig(ds, ids, Types::kTesting );

   testSetsEqOrig(ds, ids, numFolds);

   return true;
}
} // End namespace TMVA

TEST(CrossEvaluationSplitting, TrainingSetSplitOnSpectator)
{
   // TODO: In case of nPointsSig = 11; nPointsBkg = 10; the folds will be
   // of uneven size. The test suite should take this into account.

   TMVA::Tools::Instance();

   const UInt_t NUM_FOLDS = 2;
   const UInt_t nPointsSig = 11;
   const UInt_t nPointsBkg = 10;
   
   // Create DataSet
   TMVA::MsgLogger::InhibitOutput();
   data_t data_class0 = TMVA::createData(nPointsSig, 0);
   data_t data_class1 = TMVA::createData(nPointsBkg, 100);

   id_vec_t ids;
   ids.insert(ids.end(), std::get<0>(data_class0).begin(), std::get<0>(data_class0).end());
   ids.insert(ids.end(), std::get<0>(data_class1).begin(), std::get<0>(data_class1).end());

   TMVA::DataLoader * d = new TMVA::DataLoader("dataset");
   
   d->AddSignalTree    ( std::get<1>(data_class0) );
   d->AddBackgroundTree( std::get<1>(data_class1) );

   d->AddVariable(  "x" , 'D' );
   d->AddSpectator( "id", "id", "" );
   d->PrepareTrainingAndTestTree("", Form("SplitMode=Block:nTrain_Signal=%i:nTrain_Background=%i:!V", nPointsSig, nPointsBkg));
   
   d->GetDataSetInfo().GetDataSet(); // Force creation of dataset.
   TMVA::MsgLogger::EnableOutput();

   d->MakeKFoldDataSetCE(NUM_FOLDS, "id");

   // Actual test
   testFold(d, 0, NUM_FOLDS, ids);
   testFold(d, 1, NUM_FOLDS, ids);
}
