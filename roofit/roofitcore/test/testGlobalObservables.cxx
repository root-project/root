// Tests for global observables
// Authors: Jonas Rembser, CERN  08/2021

#include "RooRealVar.h"
#include "RooMsgService.h"
#include "RooGaussian.h"
#include "RooPolynomial.h"
#include "RooAddPdf.h"
#include "RooDataSet.h"
#include "RooProdPdf.h"
#include "RooFitResult.h"
#include "RooConstVar.h"
#include "RooPoisson.h"
#include "RooProduct.h"
#include "RooWorkspace.h"

#include "gtest/gtest.h"

#include <memory>
#include <functional>

namespace {

// Helper function to check if two RooFitResults are not identical.
// We can't use RooFitResult::isIdentical() here, because it will print
// something when the comparison fails even with verbose set to false.
bool isNotIdentical(RooFitResult const &res1, RooFitResult const &res2)
{
   std::size_t n = res1.floatParsFinal().size();
   if (n != res2.floatParsFinal().size())
      return true;
   for (std::size_t i = 0; i < n; ++i) {
      if (static_cast<RooAbsRealLValue &>(res1.floatParsFinal()[i]).getVal() !=
          static_cast<RooAbsRealLValue &>(res2.floatParsFinal()[i]).getVal())
         return true;
   }
   return false;
}

} // namespace

// Test environment to verify that if we use the feature of storing global
// observables in a RooDataSet, we can reproduce the same fit results as when
// we track the global observables separately.
class TestGlobalObservables : public ::testing::Test {
public:
   void SetUp() override
   {
      // silence log output
      RooMsgService::instance().setGlobalKillBelow(RooFit::WARNING);

      // observables
      RooRealVar x("x", "x", 0.0, 0.0, 20.0);

      // global observables, always constant in fits
      RooRealVar gm("gm", "gm", 11.0, 0.0, 20.0);
      RooRealVar gs("gs", "gs", 1.0, 0.1, 10.0);
      gm.setConstant(true);
      gs.setConstant(true);

      // constrained parameters
      RooRealVar f("f", "f", 0.5, 0.0, 1.0);

      // other parameters
      RooRealVar m("m", "m", 10.0, 0.0, 20.0);
      RooRealVar s("s", "s", 2.0, 0.1, 10.0);

      // We use the global observable also in the model for the event
      // observables. It's unusual, but let's better do this to also cover the
      // corner case where the global observable is not only part of the
      // constraint term.
      RooProduct sigma{"sigma", "sigma", {s, gs}};

      // build the unconstrained model
      RooGaussian model("model", "model", x, m, sigma);

      // the constraint pdfs, they are RooPoisson so we can't have tests that accidentally
      // pass because of the symmetry of normalizing over x or mu
      RooPoisson mconstraint("mconstraint", "mconstraint", gm, m);
      RooPoisson sconstraint("sconstraint", "sconstraint", gs, s);

      // the model multiplied with the constraint term
      RooProdPdf modelc("modelc", "modelc", RooArgSet(model, mconstraint, sconstraint));

      // generate small dataset for use in fitting below, also cloned versions
      // with one or two global observables attached
      _data.reset(model.generate(x, 50));

      _dataWithMeanSigmaGlobs.reset(
         static_cast<RooDataSet *>(_data->Clone((std::string(_data->GetName()) + "_gm_gs").c_str())));
      _dataWithMeanSigmaGlobs->setGlobalObservables({gm, gs});

      _dataWithMeanGlob.reset(static_cast<RooDataSet *>(_data->Clone((std::string(_data->GetName()) + "_gm").c_str())));
      _dataWithMeanGlob->setGlobalObservables(gm);

      _workspace = std::make_unique<RooWorkspace>("workspace", "workspace");
      _workspace->import(modelc);
   }

   // reset the parameter values to initial values before fits
   void resetParameters()
   {
      std::vector<std::string> names{"x", "m", "s", "gm", "gs"};
      std::vector<double> values{0.0, 10.0, 2.0, 11.0, 1.0};
      for (std::size_t i = 0; i < names.size(); ++i) {
         auto *var = _workspace->var(names[i]);
         var->setVal(values[i]);
         var->setError(0.0);
      }
   }

   RooWorkspace &ws() { return *_workspace; }
   RooDataSet &data() { return *_data; }
   RooDataSet &dataWithMeanSigmaGlobs() { return *_dataWithMeanSigmaGlobs; }
   RooDataSet &dataWithMeanGlob() { return *_dataWithMeanGlob; }
   RooAbsPdf &model() { return *ws().pdf("model"); }
   RooAbsPdf &modelc() { return *ws().pdf("modelc"); }

   std::unique_ptr<RooFitResult> doFit(RooAbsPdf &model, RooAbsData &data, RooCmdArg const &arg1 = RooCmdArg::none(),
                                       RooCmdArg const &arg2 = RooCmdArg::none(),
                                       RooCmdArg const &arg3 = RooCmdArg::none(),
                                       RooCmdArg const &arg4 = RooCmdArg::none())
   {
      using namespace RooFit;
      return std::unique_ptr<RooFitResult>(
         model.fitTo(data, Save(), Verbose(false), PrintLevel(-1), arg1, arg2, arg3, arg4));
   }

   void TearDown() override
   {
      _workspace.reset();
      _data.reset();
      _dataWithMeanSigmaGlobs.reset();
      _data.reset();
   }

private:
   std::unique_ptr<RooWorkspace> _workspace;
   std::unique_ptr<RooDataSet> _data;
   std::unique_ptr<RooDataSet> _dataWithMeanSigmaGlobs;
   std::unique_ptr<RooDataSet> _dataWithMeanGlob;
};

TEST_F(TestGlobalObservables, NoConstraints)
{
   using namespace RooFit;

   auto &gm = *ws().var("gm");
   auto &gs = *ws().var("gs");

   // fit with no constraints
   resetParameters();
   auto res1 = doFit(model(), data());
   resetParameters();
   // vary global observable to verify true value is picked up from the dataset
   gm.setVal(gm.getVal() + 0.5);
   gs.setVal(gs.getVal() + 2.5);
   double gmVaryVal = gm.getVal();
   double gsVaryVal = gs.getVal();
   auto res2 = doFit(model(), dataWithMeanSigmaGlobs());
   EXPECT_TRUE(res1->isIdentical(*res2)) << "fitting an unconstrained model "
                                            "gave a different result when unrelated global observables were stored in "
                                            "the dataset";

   // verify that taking the global observable values from data has not changed
   // the values in the model
   {
      const auto message = "taking the global observable values from data has changed the values in the model";
      EXPECT_EQ(gmVaryVal, gm.getVal()) << message;
      EXPECT_EQ(gsVaryVal, gs.getVal()) << message;
   }
}

TEST_F(TestGlobalObservables, InternalConstrains)
{
   using namespace RooFit;

   auto &gm = *ws().var("gm");
   auto &gs = *ws().var("gs");

   // constrained fit with RooProdPdf
   resetParameters();
   auto res1 = doFit(modelc(), data(), GlobalObservables(gm, gs));
   resetParameters();
   // vary global observable to verify true value is picked up from the dataset
   gm.setVal(gm.getVal() + 0.5);
   gs.setVal(gs.getVal() + 2.5);
   double gmVaryVal = gm.getVal();
   double gsVaryVal = gs.getVal();
   auto res2 = doFit(modelc(), dataWithMeanSigmaGlobs());
   EXPECT_TRUE(res1->isIdentical(*res2)) << "fitting an model with internal "
                                            "constraints in a RooPrdPdf gave a different result when global "
                                            "observables were stored in the dataset";

   // verify that taking the global observable values from data has not changed
   // the values in the model
   {
      const auto message = "taking the global observable values from data has changed the values in the model";
      EXPECT_EQ(gmVaryVal, gm.getVal()) << message;
      EXPECT_EQ(gsVaryVal, gs.getVal()) << message;
   }
}

TEST_F(TestGlobalObservables, ExternalConstraints)
{
   using namespace RooFit;

   auto &gm = *ws().var("gm");
   auto &gs = *ws().var("gs");
   auto &mconstraint = *ws().pdf("mconstraint");
   auto &sconstraint = *ws().pdf("sconstraint");

   // constrained fit with external constraints
   resetParameters();
   auto res1 = doFit(model(), data(), ExternalConstraints({mconstraint, sconstraint}), GlobalObservables(gm, gs));
   resetParameters();
   // vary global observable to verify true value is picked up from the dataset
   gm.setVal(gm.getVal() + 0.5);
   gs.setVal(gs.getVal() + 2.5);
   double gmVaryVal = gm.getVal();
   double gsVaryVal = gs.getVal();
   auto res2 = doFit(model(), dataWithMeanSigmaGlobs(), ExternalConstraints({mconstraint, sconstraint}));
   EXPECT_TRUE(res1->isIdentical(*res2))
      << "fitting an model with external "
         "constraints passed via ExternalConstraints() gave a different result when global "
         "observables were stored in the dataset";

   // verify that taking the global observable values from data has not changed
   // the values in the model
   {
      const auto message = "taking the global observable values from data has changed the values in the model";
      EXPECT_EQ(gmVaryVal, gm.getVal()) << message;
      EXPECT_EQ(gsVaryVal, gs.getVal()) << message;
   }
}

TEST_F(TestGlobalObservables, SubsetOfConstraintsFromData)
{
   using namespace RooFit;

   auto &gm = *ws().var("gm");
   auto &gs = *ws().var("gs");

   // check if only a subset of constraints it taken from data
   resetParameters();
   auto res1 = doFit(modelc(), data(), GlobalObservables(gm, gs));
   resetParameters();
   // Vary global observable to verify true value is picked up from the dataset.
   // This time we only get gm from the dataset, so we don't vary gs for now.
   gm.setVal(gm.getVal() + 0.5);
   double gmVaryVal = gm.getVal();
   double gsVaryVal = gs.getVal();
   // if we take the global observables from the model, they have to be constant:
   auto res2 = doFit(modelc(), dataWithMeanGlob(), GlobalObservables(gm, gs));
   EXPECT_TRUE(res1->isIdentical(*res2)) << "fitting a constrained model "
                                            "to a dataset that only stores a subset of the defined global observables "
                                            "gave the wrong result";

   // verify that taking the global observable values from data has not changed
   // the values in the model
   {
      const auto message = "taking the global observable values from data has changed the values in the model";
      EXPECT_EQ(gmVaryVal, gm.getVal()) << message;
      EXPECT_EQ(gsVaryVal, gs.getVal()) << message;
   }

   resetParameters();
   // Now that we also vary gs, the fit results should not be identical.
   gm.setVal(gm.getVal() + 0.5);
   gs.setVal(gs.getVal() + 2.5);
   gmVaryVal = gm.getVal();
   gsVaryVal = gs.getVal();
   auto res3 = doFit(modelc(), dataWithMeanGlob(), GlobalObservables(gm, gs));
   EXPECT_TRUE(isNotIdentical(*res1, *res3))
      << "fitting a constrained model "
         "to a dataset that only stores a subset of the defined global observables "
         "gave the wrong result";

   // verify that taking the global observable values from data has not changed
   // the values in the model
   {
      const auto message = "taking the global observable values from data has changed the values in the model";
      EXPECT_EQ(gmVaryVal, gm.getVal()) << message;
      EXPECT_EQ(gsVaryVal, gs.getVal()) << message;
   }
}

TEST_F(TestGlobalObservables, ResetDataToWrongData)
{
   using namespace RooFit;

   auto &gm = *ws().var("gm");
   auto &gs = *ws().var("gs");
   auto &model = modelc();

   // constrained fit with RooProdPdf
   resetParameters();
   auto res1 = doFit(model, data(), GlobalObservables(gm, gs));

   resetParameters();
   // vary global observable to deliberately store "wrong" values in a cloned dataset
   std::unique_ptr<RooDataSet> wrongData{static_cast<RooDataSet *>(data().Clone())};
   gm.setVal(gm.getVal() + 0.5);
   gs.setVal(gs.getVal() + 2.5);
   wrongData->setGlobalObservables({gm, gs});

   // check that the fit works when using the dataset with the correct values
   std::unique_ptr<RooAbsReal> nll{model.createNLL(dataWithMeanSigmaGlobs())};
   RooAbsPdf::MinimizerConfig minimizerCfg;
   minimizerCfg.doSave = true;
   minimizerCfg.printLevel = -1;
   auto res2 = model.minimizeNLL(*nll, dataWithMeanSigmaGlobs(), minimizerCfg);
   EXPECT_TRUE(res1->isIdentical(*res2)) << "fitting an model with internal "
                                            "constraints in a RooPrdPdf gave a different result when global "
                                            "observables were stored in the dataset";

   nll->setData(*wrongData);
   resetParameters();
   auto res3 = model.minimizeNLL(*nll, *wrongData, minimizerCfg);

   // If resetting the dataset used for the nll worked correctly also for
   // global observables, the fit will now give the wrong result.
   EXPECT_TRUE(isNotIdentical(*res1, *res3))
      << "resetting the dataset "
         "underlying a RooNLLVar didn't change the global observable value, but it "
         "should have";
}

TEST_F(TestGlobalObservables, ResetDataToCorrectData)
{
   using namespace RooFit;

   auto &gm = *ws().var("gm");
   auto &gs = *ws().var("gs");
   auto &model = modelc();

   // constrained fit with RooProdPdf
   resetParameters();
   auto res1 = doFit(model, data(), GlobalObservables(gm, gs));

   resetParameters();
   // vary global observable to deliberately store "wrong" values in a cloned dataset
   std::unique_ptr<RooDataSet> wrongData{static_cast<RooDataSet *>(data().Clone())};
   gm.setVal(gm.getVal() + 0.5);
   gs.setVal(gs.getVal() + 2.5);
   wrongData->setGlobalObservables({gm, gs});
   resetParameters();

   // check that the fit doesn't work when using the dataset with the wrong values
   std::unique_ptr<RooAbsReal> nll{model.createNLL(*wrongData)};
   RooAbsPdf::MinimizerConfig minimizerCfg;
   minimizerCfg.doSave = true;
   minimizerCfg.printLevel = -1;
   auto res2 = model.minimizeNLL(*nll, *wrongData, minimizerCfg);
   EXPECT_TRUE(isNotIdentical(*res1, *res2)) << "fitting an model with internal "
                                                "constraints in a RooPrdPdf ignored the global "
                                                "observables stored in the dataset";

   nll->setData(dataWithMeanSigmaGlobs());
   resetParameters();
   auto res3 = model.minimizeNLL(*nll, dataWithMeanSigmaGlobs(), minimizerCfg);
   EXPECT_TRUE(res1->isIdentical(*res3)) << "resetting the dataset "
                                            "underlying a RooNLLVar didn't change the global observable value, but it "
                                            "should have";
}

TEST_F(TestGlobalObservables, GlobalObservablesSourceFromModel)
{
   using namespace RooFit;

   auto &gm = *ws().var("gm");
   auto &gs = *ws().var("gs");

   // constrained fit with RooProdPdf
   resetParameters();
   auto res1 = doFit(modelc(), data(), GlobalObservables(gm, gs));
   resetParameters();
   // vary global observable to verify true value is picked up from the dataset
   gm.setVal(gm.getVal() + 0.5);
   gs.setVal(gs.getVal() + 2.5);

   // verify that fit results are identical when global observable values are
   // taken from data
   auto res2 = doFit(modelc(), dataWithMeanSigmaGlobs());
   EXPECT_TRUE(res1->isIdentical(*res2));

   auto res3 = doFit(modelc(), dataWithMeanSigmaGlobs(), GlobalObservablesSource("model"));

   // If the global observable values are indeed taken from the model and not
   // from data, the comparison will fail now because we have changed the
   // global observable values of the model after the first fit.
   EXPECT_TRUE(isNotIdentical(*res2, *res3));
}

TEST_F(TestGlobalObservables, ResetDataButSourceFromModel)
{
   using namespace RooFit;

   auto &gm = *ws().var("gm");
   auto &gs = *ws().var("gs");
   auto &model = modelc();

   // constrained fit with RooProdPdf
   resetParameters();
   auto res1 = doFit(model, data(), GlobalObservables(gm, gs));

   resetParameters();
   // vary global observable to deliberately store "wrong" values in a cloned dataset
   std::unique_ptr<RooDataSet> wrongData{static_cast<RooDataSet *>(data().Clone())};
   gm.setVal(gm.getVal() + 0.5);
   gs.setVal(gs.getVal() + 2.5);
   wrongData->setGlobalObservables({gm, gs});

   resetParameters();

   // check that the fit works when using the dataset with the correct values
   std::unique_ptr<RooAbsReal> nll{
      model.createNLL(dataWithMeanSigmaGlobs(), GlobalObservablesSource("model"), GlobalObservables(gm, gs))};
   RooAbsPdf::MinimizerConfig minimizerCfg;
   minimizerCfg.doSave = true;
   minimizerCfg.printLevel = -1;
   auto res2 = model.minimizeNLL(*nll, dataWithMeanSigmaGlobs(), minimizerCfg);
   EXPECT_TRUE(res1->isIdentical(*res2));

   nll->setData(*wrongData);
   resetParameters();
   auto res3 = model.minimizeNLL(*nll, *wrongData, minimizerCfg);

   // this time it should still be identical because even though we reset to
   // the wrong data, we set the global observables source to "model"
   EXPECT_TRUE(res1->isIdentical(*res3));
}
