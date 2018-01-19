// @(#)root/tmva $Id$
// Author: Kim Albertsson

/*************************************************************************
 * Copyright (C) 2018, Rene Brun and Fons Rademakers.                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMVA_CvSplit
#define ROOT_TMVA_CvSplit

#include "TMVA/Configurable.h"
#include "TMVA/Types.h"

#include <Rtypes.h>
#include <TFormula.h>

#include <memory>

class TString;

namespace TMVA {

class CrossValidation;
class DataSetInfo;
class Event;

/* =============================================================================
      TMVA::CvSplit
============================================================================= */

class CvSplit : public Configurable {
public:
   CvSplit(UInt_t numFolds);
   virtual ~CvSplit() {}

   virtual void MakeKFoldDataSet(DataSetInfo &dsi) = 0;
   virtual void PrepareFoldDataSet(DataSetInfo &dsi, UInt_t foldNumber, Types::ETreeType tt) = 0;
   virtual void RecombineKFoldDataSet(DataSetInfo &dsi, Types::ETreeType tt = Types::kTraining) = 0;

   UInt_t GetNumFolds() { return fNumFolds; }
   Bool_t NeedsRebuild() { return fMakeFoldDataSet; }

protected:
   virtual std::vector<std::vector<Event *>> SplitSets(std::vector<TMVA::Event *> &oldSet, UInt_t numFolds) = 0;

   UInt_t fNumFolds;
   Bool_t fMakeFoldDataSet;

protected:
   ClassDef(CvSplit, 0);
};

/* =============================================================================
      TMVA::CvSplitBootstrappedStratified
============================================================================= */

class CvSplitBootstrappedStratified : public CvSplit {
public:
   CvSplitBootstrappedStratified(UInt_t numFolds, UInt_t seed = 0, Bool_t validationSet = kFALSE);
   ~CvSplitBootstrappedStratified() override {}

   void MakeKFoldDataSet(DataSetInfo &dsi) override;
   void PrepareFoldDataSet(DataSetInfo &dsi, UInt_t foldNumber, Types::ETreeType tt) override;
   void RecombineKFoldDataSet(DataSetInfo &dsi, Types::ETreeType tt = Types::kTraining) override;

private:
   std::vector<std::vector<Event *>> SplitSets(std::vector<TMVA::Event *> &oldSet, UInt_t numFolds) override;

   UInt_t fSeed;
   Bool_t fValidationSet;

   std::vector<std::vector<TMVA::Event *>> fTrainSigEvents;
   std::vector<std::vector<TMVA::Event *>> fTrainBkgEvents;
   std::vector<std::vector<TMVA::Event *>> fValidSigEvents;
   std::vector<std::vector<TMVA::Event *>> fValidBkgEvents;
   std::vector<std::vector<TMVA::Event *>> fTestSigEvents;
   std::vector<std::vector<TMVA::Event *>> fTestBkgEvents;

private:
   ClassDefOverride(CvSplitBootstrappedStratified, 0);
};

/* =============================================================================
      TMVA::CvSplitCrossValidationExpr
============================================================================= */

class CvSplitCrossValidationExpr {
public:
   CvSplitCrossValidationExpr(DataSetInfo &dsi, TString expr);
   ~CvSplitCrossValidationExpr() {}

   UInt_t Eval(UInt_t numFolds, const Event *ev);

   static Bool_t Validate(TString expr);

private:
   UInt_t GetSpectatorIndexForName(DataSetInfo &dsi, TString name);

private:
   DataSetInfo &fDsi;

   std::vector<std::pair<Int_t, Int_t>>
      fFormulaParIdxToDsiSpecIdx; //! Maps parameter indicies in splitExpr to their spectator index in the datasetinfo.
   Int_t fIdxFormulaParNumFolds;  //! Keeps track of the index of reserved par "NumFolds" in splitExpr.
   TString fSplitExpr;     //! Expression used to split data into folds. Should output values between 0 and numFolds.
   TFormula fSplitFormula; //! TFormula for splitExpr.

   std::vector<Double_t> fParValues;
};

/* =============================================================================
      TMVA::CvSplitCrossValidation
============================================================================= */

class CvSplitCrossValidation : public CvSplit {

   friend CrossValidation;

public:
   CvSplitCrossValidation(UInt_t numFolds, TString splitExpr, UInt_t seed = 100);
   ~CvSplitCrossValidation() override {}

   void MakeKFoldDataSet(DataSetInfo &dsi) override;
   void PrepareFoldDataSet(DataSetInfo &dsi, UInt_t foldNumber, Types::ETreeType tt) override;
   void RecombineKFoldDataSet(DataSetInfo &dsi, Types::ETreeType tt = Types::kTraining) override;

private:
   std::vector<std::vector<Event *>> SplitSets(std::vector<TMVA::Event *> &oldSet, UInt_t numFolds) override;
   std::vector<UInt_t> GetEventIndexToFoldMapping(UInt_t nEntries, UInt_t numFolds, UInt_t seed = 100);

private:
   UInt_t fSeed;
   TString fSplitExprString; //! Expression used to split data into folds. Should output values between 0 and numFolds.
   std::unique_ptr<CvSplitCrossValidationExpr> fSplitExpr;

   std::vector<std::vector<TMVA::Event *>> fTrainEvents;
   std::vector<std::vector<TMVA::Event *>> fValidEvents;
   std::vector<std::vector<TMVA::Event *>> fTestEvents;

   // Used for CrossValidation with random splits (not using the
   // CVSplitCrossValisationExpr functionality) to communicate Event to fold
   // mapping.
   std::map<const TMVA::Event *, UInt_t> fEventToFoldMapping;

private:
   ClassDefOverride(CvSplitCrossValidation, 0);
};

} // end namespace TMVA

#endif