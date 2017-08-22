#include "TMVA/CvSplit.h"

#include "TMVA/DataSet.h"
#include "TMVA/DataSetInfo.h"
#include "TMVA/Event.h"
#include "TMVA/MsgLogger.h"

#include "TString.h"

/* =============================================================================
      TMVA::CvSplit
============================================================================= */

ClassImp(TMVA::CvSplit);
ClassImp(TMVA::CvSplitBootstrappedStratified);
ClassImp(TMVA::CvSplitCrossEvaluation);

TMVA::CvSplit::CvSplit (UInt_t numFolds)
: fNumFolds(numFolds), fMakeFoldDataSet(kFALSE)
{}

/* =============================================================================
      TMVA::CvSplitBootstrappedStratified
============================================================================= */

TMVA::CvSplitBootstrappedStratified::CvSplitBootstrappedStratified (UInt_t numFolds, UInt_t seed, Bool_t validationSet)
: CvSplit(numFolds), fSeed(seed), fValidationSet(validationSet)
{}

// Post-condition: Size of vector equals numElems in ETreeType
void TMVA::CvSplitBootstrappedStratified::MakeKFoldDataSet (DataSetInfo & dsi)
{
   // Remove validation set?
   // Take DataSetSplit s as arg + numFolds
   // std::vector<EventCollection> folds = s.Split();

   // No need to do it again if the sets have already been split.
   if(fMakeFoldDataSet){
      Log() << kINFO << "Splitting in k-folds has been already done" << Endl;
      return;
   }

   fMakeFoldDataSet = kTRUE;

   // Get the original event vectors for testing and training from the dataset.
   const std::vector<TMVA::Event*> TrainingData = dsi.GetDataSet()->GetEventCollection(Types::kTraining);
   const std::vector<TMVA::Event*> TestingData = dsi.GetDataSet()->GetEventCollection(Types::kTesting);

   std::vector<TMVA::Event*> TrainSigData;
   std::vector<TMVA::Event*> TrainBkgData;
   std::vector<TMVA::Event*> TestSigData;
   std::vector<TMVA::Event*> TestBkgData;

   // Split the testing and training sets into signal and background classes.
   for(UInt_t i=0; i<TrainingData.size(); ++i){
      if( strncmp( dsi.GetClassInfo( TrainingData.at(i)->GetClass() )->GetName(), "Signal", 6) == 0){ TrainSigData.push_back(TrainingData.at(i)); }
      else if( strncmp( dsi.GetClassInfo( TrainingData.at(i)->GetClass() )->GetName(), "Background", 10) == 0){ TrainBkgData.push_back(TrainingData.at(i)); }
      else{
         Log() << kFATAL << "DataSets should only contain Signal and Background classes for classification, " << dsi.GetClassInfo( TrainingData.at(i)->GetClass() )->GetName() << " is not a recognised class" << Endl;
      }
   }

   for(UInt_t i=0; i<TestingData.size(); ++i){
      if( strncmp( dsi.GetClassInfo( TestingData.at(i)->GetClass() )->GetName(), "Signal", 6) == 0){ TestSigData.push_back(TestingData.at(i)); }
      else if( strncmp( dsi.GetClassInfo( TestingData.at(i)->GetClass() )->GetName(), "Background", 10) == 0){ TestBkgData.push_back(TestingData.at(i)); }
      else{
         Log() << kFATAL << "DataSets should only contain Signal and Background classes for classification, " << dsi.GetClassInfo( TestingData.at(i)->GetClass() )->GetName() << " is not a recognised class" << Endl;
      }
   }


   // Split the sets into the number of folds.
   if(fValidationSet){
      std::vector<std::vector<TMVA::Event*>> tempSigEvents = SplitSets(TrainSigData, 2);
      std::vector<std::vector<TMVA::Event*>> tempBkgEvents = SplitSets(TrainBkgData, 2);
      fTrainSigEvents = SplitSets(tempSigEvents.at(0), fNumFolds);
      fTrainBkgEvents = SplitSets(tempBkgEvents.at(0), fNumFolds);
      fValidSigEvents = SplitSets(tempSigEvents.at(1), fNumFolds);
      fValidBkgEvents = SplitSets(tempBkgEvents.at(1), fNumFolds);
   }
   else{
      fTrainSigEvents = SplitSets(TrainSigData, fNumFolds);
      fTrainBkgEvents = SplitSets(TrainBkgData, fNumFolds);
   }

   fTestSigEvents = SplitSets(TestSigData, fNumFolds);
   fTestBkgEvents = SplitSets(TestBkgData, fNumFolds);
}

// Returns training and testing sets
// Post-condition: Size of vector = 2
void TMVA::CvSplitBootstrappedStratified::PrepareFoldDataSet (DataSetInfo & dsi, UInt_t foldNumber, Types::ETreeType tt)
{
   if (foldNumber >= fNumFolds) {
      Log() << kFATAL << "DataSet prepared for \"" << fNumFolds << "\" folds, requested fold \"" << foldNumber
            << "\" is outside of range." << Endl;
      return;
   }

   UInt_t numFolds = fTrainSigEvents.size();

   std::vector<TMVA::Event*>* tempTrain = new std::vector<TMVA::Event*>;
   std::vector<TMVA::Event*>* tempTest = new std::vector<TMVA::Event*>;

   UInt_t nTrain = 0;
   UInt_t nTest = 0;

   // Get the number of events so the memory can be reserved.
   for(UInt_t i=0; i<numFolds; ++i){
      if(tt == Types::kTraining){
         if(i!=foldNumber){
            nTrain += fTrainSigEvents.at(i).size();
            nTrain += fTrainBkgEvents.at(i).size();
         }
         else{
            nTest += fTrainSigEvents.at(i).size();
            nTest += fTrainSigEvents.at(i).size();
         }
      }
      else if(tt == Types::kValidation){
         if(i!=foldNumber){
            nTrain += fValidSigEvents.at(i).size();
            nTrain += fValidBkgEvents.at(i).size();
         }
         else{
            nTest += fValidSigEvents.at(i).size();
            nTest += fValidSigEvents.at(i).size();
         }
      }
      else if(tt == Types::kTesting){
         if(i!=foldNumber){
            nTrain += fTestSigEvents.at(i).size();
            nTrain += fTestBkgEvents.at(i).size();
         }
         else{
            nTest += fTestSigEvents.at(i).size();
            nTest += fTestSigEvents.at(i).size();
         }
      }
   }

   // Reserve memory before filling vectors
   tempTrain->reserve(nTrain);
   tempTest->reserve(nTest);

   // Fill vectors with correct folds for testing and training.
   for(UInt_t j=0; j<numFolds; ++j){
      if(tt == Types::kTraining){
         if(j!=foldNumber){
            tempTrain->insert(tempTrain->end(), fTrainSigEvents.at(j).begin(), fTrainSigEvents.at(j).end());
            tempTrain->insert(tempTrain->end(), fTrainBkgEvents.at(j).begin(), fTrainBkgEvents.at(j).end());
         }
         else{
            tempTest->insert(tempTest->end(), fTrainSigEvents.at(j).begin(), fTrainSigEvents.at(j).end());
            tempTest->insert(tempTest->end(), fTrainBkgEvents.at(j).begin(), fTrainBkgEvents.at(j).end());
         }
      }
      else if(tt == Types::kValidation){
         if(j!=foldNumber){
            tempTrain->insert(tempTrain->end(), fValidSigEvents.at(j).begin(), fValidSigEvents.at(j).end());
            tempTrain->insert(tempTrain->end(), fValidBkgEvents.at(j).begin(), fValidBkgEvents.at(j).end());
         }
         else{
            tempTest->insert(tempTest->end(), fValidSigEvents.at(j).begin(), fValidSigEvents.at(j).end());
            tempTest->insert(tempTest->end(), fValidBkgEvents.at(j).begin(), fValidBkgEvents.at(j).end());
         }
      }
      else if(tt == Types::kTesting){
         if(j!=foldNumber){
            tempTrain->insert(tempTrain->end(), fTestSigEvents.at(j).begin(), fTestSigEvents.at(j).end());
            tempTrain->insert(tempTrain->end(), fTestBkgEvents.at(j).begin(), fTestBkgEvents.at(j).end());
         }
         else{
            tempTest->insert(tempTest->end(), fTestSigEvents.at(j).begin(), fTestSigEvents.at(j).end());
            tempTest->insert(tempTest->end(), fTestBkgEvents.at(j).begin(), fTestBkgEvents.at(j).end());
         }
      }
   }

   // Assign the vectors of the events to rebuild the dataset
   dsi.GetDataSet()->SetEventCollection(tempTrain,Types::kTraining,false);
   dsi.GetDataSet()->SetEventCollection(tempTest,Types::kTesting,false);
   delete tempTest;
   delete tempTrain;
}

// Inverse of MakeKFoldsDataSet
// Post-condition: Size of vector = 2
void TMVA::CvSplitBootstrappedStratified::RecombineKFoldDataSet (DataSetInfo &, Types::ETreeType)
{
   Log() << kFATAL << "Recombination not implemented for CvSplitBootstrappedStratified" << Endl;
   return;
}

////////////////////////////////////////////////////////////////////////////////
/// Splits the input vector in to equally sized randomly sampled folds.
/// Randomly sampled!? This is bootstrapping then?
/// 

std::vector<std::vector<TMVA::Event*>>
TMVA::CvSplitBootstrappedStratified::SplitSets (std::vector<TMVA::Event*>& oldSet, UInt_t numFolds)
{
   ULong64_t nEntries = oldSet.size();
   ULong64_t foldSize = nEntries/numFolds;

   std::vector<std::vector<TMVA::Event*>> tempSets;
   tempSets.resize(numFolds);

   TRandom3 r(fSeed);

   ULong64_t inSet = 0;

   for(ULong64_t i=0; i<nEntries; i++){
      bool inTree = false;
      if(inSet == foldSize*numFolds){
         break;
      }
      else{
         while(!inTree){
            int s = r.Integer(numFolds);
            if(tempSets.at(s).size()<foldSize){
               tempSets.at(s).push_back(oldSet.at(i));
               inSet++;
               inTree=true;
            }
         }
      }
   }

   return tempSets;
}

/* =============================================================================
      TMVA::CvSplitCrossEvaluation
============================================================================= */

TMVA::CvSplitCrossEvaluation::CvSplitCrossEvaluation (UInt_t numFolds, TString spectatorName)
: CvSplit(numFolds), fSpectatorName(spectatorName)
{}

// Post-condition: Size of vector equals numElems in ETreeType
void TMVA::CvSplitCrossEvaluation::MakeKFoldDataSet (DataSetInfo & dsi)
{
   // Find index of spectator (or variable?)
   // Search on title, fallback on label/expression
   std::vector<VariableInfo> spectatorInfos = dsi.GetSpectatorInfos();
   fSpectatorIdx = -1;
   for (UInt_t iSpectator = 0; iSpectator < spectatorInfos.size(); ++iSpectator) {
      VariableInfo vi = spectatorInfos[iSpectator];
      if (vi.GetName() == fSpectatorName) {
         fSpectatorIdx = iSpectator;
         break;
      } else if (vi.GetLabel() == fSpectatorName) {
         fSpectatorIdx = iSpectator;
         break;
      } else if (vi.GetExpression() == fSpectatorName) {
         fSpectatorIdx = iSpectator;
         break;
      }
   };

   if (fSpectatorIdx == -1) {
      Log() << kFATAL << "Spectator variable\"" << fSpectatorName << "\" not found." << Endl;
      return;
   }

   // One vector for each fold (stratified? not for starters)
   // split on the var (SplitSets is really Bootstrapping *sigh*)
   //

   // No need to do it again if the sets have already been split.
   if (fMakeFoldDataSet) {
      Log() << kINFO << "Splitting in k-folds has been already done" << Endl;
      return;
   }

   fMakeFoldDataSet = kTRUE;

   // Get the original event vectors for testing and training from the dataset.
   const std::vector<Event *> trainingDataOrig =
      dsi.GetDataSet()->GetEventCollection(Types::kTraining);
   const std::vector<Event *> testingDataOrig = dsi.GetDataSet()->GetEventCollection(Types::kTesting);

   std::vector<Event *> trainData;
   std::vector<Event *> testData;

   // Split the testing and training sets into signal and background classes.
   for (UInt_t i = 0; i < trainingDataOrig.size(); ++i) {
      trainData.push_back(trainingDataOrig.at(i));
   }

   for (UInt_t i = 0; i < testingDataOrig.size(); ++i) {
      // trainData.push_back( trainingDataOrig.at(i) );
      testData.push_back(testingDataOrig.at(i));
   }

   // Split the sets into the number of folds.
   fTrainSigEvents = SplitSets(trainData, fNumFolds);
   fTestSigEvents = SplitSets(testData, fNumFolds);
   // fTrainBkgEvents = SplitSets(trainData, fNumFolds);
   // fTestBkgEvents  = SplitSets(testData,, fNumFolds);

   fTrainBkgEvents.clear();
   fTestBkgEvents.clear();
   for (UInt_t iFold = 0; iFold < fNumFolds; ++iFold) {
      fTrainBkgEvents.emplace_back();
      fTestBkgEvents.emplace_back();
   };
}

void TMVA::CvSplitCrossEvaluation::PrepareFoldDataSet (DataSetInfo & dsi, UInt_t foldNumber, Types::ETreeType tt)
{
   if (foldNumber >= fNumFolds) {
      Log() << kFATAL << "DataSet prepared for \"" << fNumFolds << "\" folds, requested fold \"" << foldNumber
            << "\" is outside of range." << Endl;
      return;
   }

   UInt_t numFolds = fTrainSigEvents.size();

   std::vector<TMVA::Event*>* tempTrain = new std::vector<TMVA::Event*>;
   std::vector<TMVA::Event*>* tempTest = new std::vector<TMVA::Event*>;

   UInt_t nTrain = 0;
   UInt_t nTest = 0;

   // Get the number of events so the memory can be reserved.
   for(UInt_t i=0; i<numFolds; ++i){
      if(tt == Types::kTraining){
         if(i!=foldNumber){
            nTrain += fTrainSigEvents.at(i).size();
            nTrain += fTrainBkgEvents.at(i).size();
         }
         else{
            nTest += fTrainSigEvents.at(i).size();
            nTest += fTrainSigEvents.at(i).size();
         }
      }
      else if(tt == Types::kValidation){
         if(i!=foldNumber){
            nTrain += fValidSigEvents.at(i).size();
            nTrain += fValidBkgEvents.at(i).size();
         }
         else{
            nTest += fValidSigEvents.at(i).size();
            nTest += fValidSigEvents.at(i).size();
         }
      }
      else if(tt == Types::kTesting){
         if(i!=foldNumber){
            nTrain += fTestSigEvents.at(i).size();
            nTrain += fTestBkgEvents.at(i).size();
         }
         else{
            nTest += fTestSigEvents.at(i).size();
            nTest += fTestSigEvents.at(i).size();
         }
      }
   }

   // Reserve memory before filling vectors
   tempTrain->reserve(nTrain);
   tempTest->reserve(nTest);

   // Fill vectors with correct folds for testing and training.
   for(UInt_t j=0; j<numFolds; ++j){
      if(tt == Types::kTraining){
         if(j!=foldNumber){
            tempTrain->insert(tempTrain->end(), fTrainSigEvents.at(j).begin(), fTrainSigEvents.at(j).end());
            tempTrain->insert(tempTrain->end(), fTrainBkgEvents.at(j).begin(), fTrainBkgEvents.at(j).end());
         }
         else{
            tempTest->insert(tempTest->end(), fTrainSigEvents.at(j).begin(), fTrainSigEvents.at(j).end());
            tempTest->insert(tempTest->end(), fTrainBkgEvents.at(j).begin(), fTrainBkgEvents.at(j).end());
         }
      }
      else if(tt == Types::kValidation){
         if(j!=foldNumber){
            tempTrain->insert(tempTrain->end(), fValidSigEvents.at(j).begin(), fValidSigEvents.at(j).end());
            tempTrain->insert(tempTrain->end(), fValidBkgEvents.at(j).begin(), fValidBkgEvents.at(j).end());
         }
         else{
            tempTest->insert(tempTest->end(), fValidSigEvents.at(j).begin(), fValidSigEvents.at(j).end());
            tempTest->insert(tempTest->end(), fValidBkgEvents.at(j).begin(), fValidBkgEvents.at(j).end());
         }
      }
      else if(tt == Types::kTesting){
         if(j!=foldNumber){
            tempTrain->insert(tempTrain->end(), fTestSigEvents.at(j).begin(), fTestSigEvents.at(j).end());
            tempTrain->insert(tempTrain->end(), fTestBkgEvents.at(j).begin(), fTestBkgEvents.at(j).end());
         }
         else{
            tempTest->insert(tempTest->end(), fTestSigEvents.at(j).begin(), fTestSigEvents.at(j).end());
            tempTest->insert(tempTest->end(), fTestBkgEvents.at(j).begin(), fTestBkgEvents.at(j).end());
         }
      }
   }

   // Assign the vectors of the events to rebuild the dataset
   dsi.GetDataSet()->SetEventCollection(tempTrain,Types::kTraining,false);
   dsi.GetDataSet()->SetEventCollection(tempTest,Types::kTesting,false);
   delete tempTest;
   delete tempTrain;
}

void TMVA::CvSplitCrossEvaluation::RecombineKFoldDataSet (DataSetInfo & dsi, Types::ETreeType tt)
{
   if (tt != Types::kTraining) {
      Log() << kFATAL << "Only kTraining is supported for CvSplitCrossEvaluation::RecombineKFoldDataSet currently." << std::endl;
   }

   std::vector<Event *> *tempVec = new std::vector<Event *>;

   for (UInt_t i = 0; i < fNumFolds; ++i) {
      tempVec->insert(tempVec->end(), fTrainSigEvents.at(i).begin(), fTrainSigEvents.at(i).end());
      tempVec->insert(tempVec->end(), fTrainBkgEvents.at(i).begin(), fTrainBkgEvents.at(i).end());
   }

   dsi.GetDataSet()->SetEventCollection(tempVec, Types::kTraining, false);
   dsi.GetDataSet()->SetEventCollection(tempVec, Types::kTesting, false);

   delete tempVec;
}

std::vector<std::vector<TMVA::Event*>>
TMVA::CvSplitCrossEvaluation::SplitSets (std::vector<TMVA::Event*>& oldSet, UInt_t numFolds)
{
   ULong64_t nEntries = oldSet.size();
   ULong64_t foldSize = nEntries / numFolds;

   std::vector<std::vector<Event *>> tempSets;
   tempSets.reserve(fNumFolds);
   for (UInt_t iFold = 0; iFold < numFolds; ++iFold) {
      tempSets.emplace_back();
      tempSets.at(iFold).reserve(foldSize);
   }

   for (ULong64_t i = 0; i < nEntries; i++) {
      TMVA::Event *ev = oldSet[i];
      auto val = ev->GetSpectator(fSpectatorIdx);

      tempSets.at((UInt_t)val % (UInt_t)numFolds).push_back(ev);
   }

   return tempSets;
}

UInt_t TMVA::CvSplitCrossEvaluation::GetSpectatorIndexForName ( TString name )
{

}
