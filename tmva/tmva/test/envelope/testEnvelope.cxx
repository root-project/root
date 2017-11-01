// @(#)root/tmva $Id$
// Author: Omar Zapata

/*************************************************************************
 * Copyright (C) 2017, Omar Zapata                                       *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <gtest/gtest.h>

#include <TMVA/Envelope.h>
#include <TMVA/DataLoader.h>

#include <TFile.h>
#include <TSystem.h>
#include <TString.h>

using namespace TMVA;

// Envelope is a pure base class,
// I will to create a basic class to create an object
class BasicEnvelope : public TMVA::Envelope {
public:
   BasicEnvelope(const TString &name, DataLoader *dalaloader, TFile *file, const TString options)
      : Envelope(name, dalaloader, file, options)
   {
   }
   TString GetTransformations() { return fTransformations; }

   Int_t GetJobs() { return fJobs; }

   virtual void Evaluate(){};

   std::vector<OptionMap> &GetMethods() { return fMethods; }
};

// test without DataLoader or output file
class EnvelopeTest1 : public ::testing::Test {
public:
   BasicEnvelope *envelope;
   EnvelopeTest1()
   {
      TMVA::Tools::Instance();
      envelope = new BasicEnvelope("envelope", nullptr, nullptr, "ModelPersistence=1:V:Transformations=I:Jobs=3");
   }
   ~EnvelopeTest1() { delete envelope; }
   void Booking()
   {
      envelope->BookMethod("BDT", "BDTG", "NTrees=100");
      envelope->BookMethod(TMVA::Types::kBDT, "BDTB", "BoostType=Bagging");
   }
};

TEST_F(EnvelopeTest1, ParseOptions)
{
   envelope->ParseOptions();

   EXPECT_EQ(envelope->IsModelPersistence(), kTRUE);
   envelope->SetModelPersistence(kFALSE);
   EXPECT_EQ(envelope->IsModelPersistence(), kFALSE);

   EXPECT_EQ(envelope->IsVerbose(), kTRUE);
   envelope->SetVerbose(kFALSE);
   EXPECT_EQ(envelope->IsVerbose(), kFALSE);

   EXPECT_EQ(envelope->GetTransformations(), "I");
   EXPECT_EQ(envelope->GetJobs(), 3);

   EXPECT_EQ(envelope->IsSilentFile(), kTRUE);
   EXPECT_EQ(envelope->GetFile(), nullptr);

   EXPECT_EQ(envelope->GetDataLoader(), nullptr);
}

TEST_F(EnvelopeTest1, Booking)
{
   Booking();
   EXPECT_EQ(envelope->HasMethod("BDT", "BDTG"), kTRUE);
   EXPECT_EQ(envelope->HasMethod("SVM", "SVM"), kFALSE);
   EXPECT_EQ(envelope->HasMethod("BDT", "BDTB"), kTRUE);

   envelope->BookMethod(TMVA::Types::kSVM, "SVM", "Gamma=0.25:Tol=0.001:VarTransform=Norm");
   EXPECT_EQ(envelope->HasMethod("SVM", "SVM"), kTRUE);

   auto &meths = envelope->GetMethods();
   for (auto &m : meths) {
      auto mname = m.GetValue<TString>("MethodName");
      auto mtitle = m.GetValue<TString>("MethodTitle");
      auto mopts = m.GetValue<TString>("MethodOptions");
      EXPECT_EQ(envelope->HasMethod(mname.Data(), mtitle.Data()), kTRUE);
      if (mname == "BDT" && mtitle == "BDTG")
         EXPECT_EQ(mopts, "NTrees=100");
      if (mname == "BDT" && mtitle == "BDTB")
         EXPECT_EQ(mopts, "BoostType=Bagging");
      if (mname == "SVM" && mtitle == "SVM")
         EXPECT_EQ(mopts, "Gamma=0.25:Tol=0.001:VarTransform=Norm");
   }
}
