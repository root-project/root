
// ROOT
#include "TRandom3.h"

// TMVA
#include "TMVA/DataLoader.h"
#include "TMVA/Factory.h"
#include "TMVA/Interval.h"
#include "TMVA/OptimizeConfigParameters.h"
#include "TMVA/Config.h"

// Stdlib
#include <iostream>

// External
#include "gtest/gtest.h"

class TestOptimizeConfigParametersFixture : public ::testing::Test {
public:
   TestOptimizeConfigParametersFixture()
   : fDataloader(GetDataloader()),
     fFactory(GetFactory()),
     fMethod(GetMethod(fFactory, fDataloader))
   {
      TMVA::MsgLogger::InhibitOutput();
   }

   TMVA::MethodBase * GetMethod() {return fMethod;}

protected:
   void SetUp() override {}
   void TearDown () override {}

   TMVA::DataLoader * GetDataloader()
   {
      TRandom3 rng(100);

      auto dl = new TMVA::DataLoader{"Dataloader"};
      dl->AddVariable("x");
      for (Int_t n = 0; n < 100; ++n) {
         dl->AddSignalTrainingEvent({rng.Gaus()+0.1});
         dl->AddSignalTestEvent({rng.Gaus()+0.1});
         dl->AddBackgroundTrainingEvent({rng.Gaus()-0.1});
         dl->AddBackgroundTestEvent({rng.Gaus()-0.1});
      }

      dl->PrepareTrainingAndTestTree("", "!V");

      return dl;
   }

   TMVA::Factory * GetFactory()
   {
      TMVA::gConfig().SetSilent(true);
      return new TMVA::Factory{"", "!V:Silent:!DrawProgressBar"};
   }

   TMVA::MethodBase * GetMethod(TMVA::Factory * f,
                                TMVA::DataLoader * dl)
   {
      f->BookMethod(dl, TMVA::Types::kBDT, "BDT",
                     "!V:NTrees=1:BoostType=Grad:NCuts=10");
      f->TrainAllMethods();
      auto * m = f->GetMethod(dl->GetName(), "BDT");
      return dynamic_cast<TMVA::MethodBase *>(m);
   }

   TMVA::DataLoader * fDataloader;
   TMVA::Factory * fFactory;
   TMVA::MethodBase * fMethod;
};

/*
 * This class is a friend class to the unit under test and its main purpose is
 * to forward calls to private functions.
 *
 * It also provides a static test environment so that the creation of dataset
 * and any potetial shared operations (e.g. training) is executed only once
 * for all tests.
 */
class TestOptimizeConfigParameters {
public:
   TestOptimizeConfigParameters(TString fomType, TMVA::MethodBase * method)
   :  fOpt(GetOptimiser(fomType, method))
   {}

   Double_t GetFOM() {
      return fOpt.GetFOM();
   }

   Double_t GetSigEffAtBkgEff(double x) {
      return fOpt.GetSigEffAtBkgEff(x);
   }

   Double_t GetBkgRejAtSigEff(double x) {
      return fOpt.GetBkgRejAtSigEff(x);
   }

   Double_t GetBkgEffAtSigEff(double x) {
      return fOpt.GetBkgEffAtSigEff(x);
   }

private:
   TMVA::OptimizeConfigParameters GetOptimiser(TString fomType, TMVA::MethodBase * method)
   {
      std::map<TString, TMVA::Interval *> params{};
      params.insert(std::pair<TString, TMVA::Interval *>("NTrees", new TMVA::Interval(10, 1000, 5)));
      return TMVA::OptimizeConfigParameters(method, params, fomType);
   }

   TMVA::OptimizeConfigParameters fOpt;
};


TEST_F(TestOptimizeConfigParametersFixture, FOM_SigEffAtBkgEff)
{
   {   
      auto opt0 = TestOptimizeConfigParameters("SigEffAtBkgEff0.001", GetMethod());
      EXPECT_EQ(opt0.GetFOM(), opt0.GetSigEffAtBkgEff(0.001));

      auto opt1 = TestOptimizeConfigParameters("SigEffAtBkgEff0001", GetMethod());
      EXPECT_EQ(opt1.GetFOM(), opt1.GetSigEffAtBkgEff(0.001));

      std::cout << "opt0.GetSigEffAtBkgEff(0.001): "
                << opt0.GetSigEffAtBkgEff(0.001) << std::endl;
   }

   {
      auto opt0 = TestOptimizeConfigParameters("SigEffAtBkgEff0.01", GetMethod());
      EXPECT_EQ(opt0.GetFOM(), opt0.GetSigEffAtBkgEff(0.01));

      auto opt1 = TestOptimizeConfigParameters("SigEffAtBkgEff001", GetMethod());
      EXPECT_EQ(opt1.GetFOM(), opt1.GetSigEffAtBkgEff(0.01));

      std::cout << "opt0.GetSigEffAtBkgEff(0.01): "
                << opt0.GetSigEffAtBkgEff(0.01) << std::endl;
   }

   {
      auto opt0 = TestOptimizeConfigParameters("SigEffAtBkgEff0.1", GetMethod());
      EXPECT_EQ(opt0.GetFOM(), opt0.GetSigEffAtBkgEff(0.1));

      auto opt1 = TestOptimizeConfigParameters("SigEffAtBkgEff01", GetMethod());
      EXPECT_EQ(opt1.GetFOM(), opt1.GetSigEffAtBkgEff(0.1));

      std::cout << "opt0.GetSigEffAtBkgEff(0.1): "
                << opt0.GetSigEffAtBkgEff(0.1) << std::endl;
   }

   {
      auto opt0 = TestOptimizeConfigParameters("SigEffAtBkgEff0.5", GetMethod());
      EXPECT_EQ(opt0.GetFOM(), opt0.GetSigEffAtBkgEff(0.5));

      auto opt1 = TestOptimizeConfigParameters("SigEffAtBkgEff05", GetMethod());
      EXPECT_EQ(opt1.GetFOM(), opt1.GetSigEffAtBkgEff(0.5));

      std::cout << "opt0.GetSigEffAtBkgEff(0.5): "
                << opt0.GetSigEffAtBkgEff(0.5) << std::endl;
   }


   // auto opt = TestOptimizeConfigParameters("SigEffAtBkg1", GetMethod());
   // auto opt = TestOptimizeConfigParameters("SigEffAtBkg10", GetMethod());
   // Expect crash
}


TEST_F(TestOptimizeConfigParametersFixture, FOM_BkgRejAtSigEff)
{
   {   
      auto opt0 = TestOptimizeConfigParameters("BkgRejAtSigEff0.001", GetMethod());
      EXPECT_EQ(opt0.GetFOM(), opt0.GetBkgRejAtSigEff(0.001));

      auto opt1 = TestOptimizeConfigParameters("BkgRejAtSigEff0001", GetMethod());
      EXPECT_EQ(opt1.GetFOM(), opt1.GetBkgRejAtSigEff(0.001));

      std::cout << "opt0.GetBkgRejAtSigEff(0.001): "
                << opt0.GetBkgRejAtSigEff(0.001) << std::endl;
   }

   {
      auto opt0 = TestOptimizeConfigParameters("BkgRejAtSigEff0.01", GetMethod());
      EXPECT_EQ(opt0.GetFOM(), opt0.GetBkgRejAtSigEff(0.01));

      auto opt1 = TestOptimizeConfigParameters("BkgRejAtSigEff001", GetMethod());
      EXPECT_EQ(opt1.GetFOM(), opt1.GetBkgRejAtSigEff(0.01));

      std::cout << "opt0.GetBkgRejAtSigEff(0.01): "
                << opt0.GetBkgRejAtSigEff(0.01) << std::endl;
   }

   {
      auto opt0 = TestOptimizeConfigParameters("BkgRejAtSigEff0.1", GetMethod());
      EXPECT_EQ(opt0.GetFOM(), opt0.GetBkgRejAtSigEff(0.1));

      auto opt1 = TestOptimizeConfigParameters("BkgRejAtSigEff01", GetMethod());
      EXPECT_EQ(opt1.GetFOM(), opt1.GetBkgRejAtSigEff(0.1));

      std::cout << "opt0.GetBkgRejAtSigEff(0.1): "
                << opt0.GetBkgRejAtSigEff(0.1) << std::endl;
   }

   {
      auto opt0 = TestOptimizeConfigParameters("BkgRejAtSigEff0.5", GetMethod());
      EXPECT_EQ(opt0.GetFOM(), opt0.GetBkgRejAtSigEff(0.5));

      auto opt1 = TestOptimizeConfigParameters("BkgRejAtSigEff05", GetMethod());
      EXPECT_EQ(opt1.GetFOM(), opt1.GetBkgRejAtSigEff(0.5));

      std::cout << "opt0.GetBkgRejAtSigEff(0.5): "
                << opt0.GetBkgRejAtSigEff(0.5) << std::endl;
   }


   // auto opt = TestOptimizeConfigParameters("BkgRejAtSigEff1", GetMethod());
   // auto opt = TestOptimizeConfigParameters("BkgRejAtSigEff10", GetMethod());
   // Expect crash
}


TEST_F(TestOptimizeConfigParametersFixture, FOM_BkgEffAtSigEff)
{
   {   
      auto opt0 = TestOptimizeConfigParameters("BkgEffAtSigEff0.001", GetMethod());
      EXPECT_EQ(opt0.GetFOM(), opt0.GetBkgEffAtSigEff(0.001));

      auto opt1 = TestOptimizeConfigParameters("BkgEffAtSigEff0001", GetMethod());
      EXPECT_EQ(opt1.GetFOM(), opt1.GetBkgEffAtSigEff(0.001));

      std::cout << "opt0.GetBkgEffAtSigEff(0.001): "
                << opt0.GetBkgEffAtSigEff(0.001) << std::endl;
   }

   {
      auto opt0 = TestOptimizeConfigParameters("BkgEffAtSigEff0.01", GetMethod());
      EXPECT_EQ(opt0.GetFOM(), opt0.GetBkgEffAtSigEff(0.01));

      auto opt1 = TestOptimizeConfigParameters("BkgEffAtSigEff001", GetMethod());
      EXPECT_EQ(opt1.GetFOM(), opt1.GetBkgEffAtSigEff(0.01));

      std::cout << "opt0.GetBkgEffAtSigEff(0.01): "
                << opt0.GetBkgEffAtSigEff(0.01) << std::endl;
   }

   {
      auto opt0 = TestOptimizeConfigParameters("BkgEffAtSigEff0.1", GetMethod());
      EXPECT_EQ(opt0.GetFOM(), opt0.GetBkgEffAtSigEff(0.1));

      auto opt1 = TestOptimizeConfigParameters("BkgEffAtSigEff01", GetMethod());
      EXPECT_EQ(opt1.GetFOM(), opt1.GetBkgEffAtSigEff(0.1));

      std::cout << "opt0.GetBkgEffAtSigEff(0.1): "
                << opt0.GetBkgEffAtSigEff(0.1) << std::endl;
   }

   {
      auto opt0 = TestOptimizeConfigParameters("BkgEffAtSigEff0.5", GetMethod());
      EXPECT_EQ(opt0.GetFOM(), opt0.GetBkgEffAtSigEff(0.5));

      auto opt1 = TestOptimizeConfigParameters("BkgEffAtSigEff05", GetMethod());
      EXPECT_EQ(opt1.GetFOM(), opt1.GetBkgEffAtSigEff(0.5));

      std::cout << "opt0.GetBkgEffAtSigEff(0.5): "
                << opt0.GetBkgEffAtSigEff(0.5) << std::endl;
   }


   // auto opt = TestOptimizeConfigParameters("BkgEffAtSigEff1", GetMethod());
   // auto opt = TestOptimizeConfigParameters("BkgEffAtSigEff10", GetMethod());
   // Expect crash
}
