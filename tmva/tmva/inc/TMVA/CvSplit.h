// @(#)root/tmva $Id$
// Author: Kim Albertsson

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : CvSplit                                                               *
 * Web    : http://root.cern.ch                                                   *
 *                                                                                *
 * Description:                                                                   *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Kim Albertsson <kim.albertsson@cern.ch> - CERN/LTU                        *
 *      Sergei Gleyzer <sergei.gleyzer@cern.ch> - CERN, Switzerland               *
 *      Lorenzo Moneta <Lorenzo.Moneta@cern.ch> - CERN, Switzerland               *
 *                                                                                *
 * Copyright (c) 2005-2017:                                                       *
 *      CERN, Switzerland                                                         *
 *      ITM/UdeA, Colombia                                                        *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_CvSplit
#define ROOT_TMVA_CvSplit

#include "TMVA/Configurable.h"
#include "TMVA/Types.h"

#include "Rtypes.h"

class TString;

namespace TMVA
{

class DataSetInfo;
class Event;




/* =============================================================================
      TMVA::CvSplit
============================================================================= */

class CvSplit : public Configurable
{
public:

   CvSplit(UInt_t numFolds);
   virtual ~CvSplit() {}

   virtual void MakeKFoldDataSet (DataSetInfo & dsi) = 0;
   virtual void PrepareFoldDataSet (DataSetInfo & dsi, UInt_t foldNumber, Types::ETreeType tt) = 0;
   virtual void RecombineKFoldDataSet (DataSetInfo & dsi, Types::ETreeType tt = Types::kTraining) = 0;

   UInt_t GetNumFolds() {return fNumFolds;}
   Bool_t NeedsRebuild() {return fMakeFoldDataSet;}

protected:
   virtual std::vector<std::vector<Event*>> SplitSets (std::vector<TMVA::Event*>& oldSet, UInt_t numFolds) = 0;

   UInt_t fNumFolds;
   Bool_t fMakeFoldDataSet;

protected:
   ClassDef(CvSplit, 0);
};




/* =============================================================================
      TMVA::CvSplitBootstrappedStratified
============================================================================= */

class CvSplitBootstrappedStratified : public CvSplit
{
public:
   CvSplitBootstrappedStratified(UInt_t numFolds, UInt_t seed = 0, Bool_t validationSet = kFALSE);
   ~CvSplitBootstrappedStratified() override {}

   void MakeKFoldDataSet (DataSetInfo & dsi) override;
   void PrepareFoldDataSet (DataSetInfo & dsi, UInt_t foldNumber, Types::ETreeType tt) override;
   void RecombineKFoldDataSet (DataSetInfo & dsi, Types::ETreeType tt = Types::kTraining) override;

private:
   std::vector<std::vector<Event*>> SplitSets (std::vector<TMVA::Event*>& oldSet, UInt_t numFolds) override;

   UInt_t fSeed;
   Bool_t fValidationSet;

   std::vector<std::vector<TMVA::Event*>> fTrainSigEvents;
   std::vector<std::vector<TMVA::Event*>> fTrainBkgEvents;
   std::vector<std::vector<TMVA::Event*>> fValidSigEvents;
   std::vector<std::vector<TMVA::Event*>> fValidBkgEvents;
   std::vector<std::vector<TMVA::Event*>> fTestSigEvents;
   std::vector<std::vector<TMVA::Event*>> fTestBkgEvents;

private:
   ClassDefOverride(CvSplitBootstrappedStratified, 0);
};




/* =============================================================================
      TMVA::CvSplitCrossEvaluation
============================================================================= */

class CvSplitCrossEvaluation : public CvSplit
{
public:
   CvSplitCrossEvaluation (UInt_t numFolds, TString spectatorName);
   ~CvSplitCrossEvaluation() override {}

   void MakeKFoldDataSet (DataSetInfo & dsi) override;
   void PrepareFoldDataSet (DataSetInfo & dsi, UInt_t foldNumber, Types::ETreeType tt) override;
   void RecombineKFoldDataSet (DataSetInfo & dsi, Types::ETreeType tt = Types::kTraining) override;

private:
   std::vector<std::vector<Event*>> SplitSets (std::vector<TMVA::Event*>& oldSet, UInt_t numFolds) override;

   UInt_t GetSpectatorIndexForName(TString name);

private:
   TString fSpectatorName;
   UInt_t fSpectatorIdx;

   std::vector<std::vector<TMVA::Event*>> fTrainSigEvents;
   std::vector<std::vector<TMVA::Event*>> fTrainBkgEvents;
   std::vector<std::vector<TMVA::Event*>> fValidSigEvents;
   std::vector<std::vector<TMVA::Event*>> fValidBkgEvents;
   std::vector<std::vector<TMVA::Event*>> fTestSigEvents;
   std::vector<std::vector<TMVA::Event*>> fTestBkgEvents;

private:
   ClassDefOverride(CvSplitCrossEvaluation, 0);
};

}

#endif