// Tests for the RooSimultaneous
// Authors: Jonas Rembser, CERN  06/2021

#include <Roo1DTable.h>
#include <RooAddPdf.h>
#include <RooAddition.h>
#include <RooCategory.h>
#include <RooChebychev.h>
#include <RooConstVar.h>
#include <RooDataSet.h>
#include <RooExponential.h>
#include <RooFitResult.h>
#include <RooGaussian.h>
#include <RooGenericPdf.h>
#include <RooHelpers.h>
#include <RooMinimizer.h>
#include <RooPlot.h>
#include <RooProdPdf.h>
#include <RooRandom.h>
#include <RooRealVar.h>
#include <RooSimultaneous.h>
#include <RooThresholdCategory.h>
#include <RooUniform.h>
#include <RooWorkspace.h>

#include "gtest_wrapper.h"

#include <memory>

/// Forum issue
/// https://root-forum.cern.ch/t/roofit-failed-to-create-nll-for-simultaneous-pdfs-with-multiple-range-names/49363.
/// Multi-range likelihoods should also work with RooSimultaneous, where one
/// of the observables is a category.
TEST(RooSimultaneous, MultiRangeNLL)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   using namespace RooFit;

   RooWorkspace ws{};
   ws.factory("Gaussian::pdfCat1(x[0,10],mu1[4,0,10],sigma[1.0,0.1,10.0])");
   ws.factory("Gaussian::pdfCat2(x,mu2[6,0,10],sigma)");

   auto &x = *ws.var("x");
   auto &pdfCat1 = *ws.pdf("pdfCat1");
   auto &pdfCat2 = *ws.pdf("pdfCat2");

   // set the ranges
   x.setRange("range1", 0.0, 4.0);
   x.setRange("range2", 6.0, 10.0);

   // Create combined pdf
   RooCategory indexCat("cat", "cat");
   indexCat.defineType("cat1");
   indexCat.defineType("cat2");
   RooSimultaneous simPdf("simPdf", "", indexCat);
   simPdf.addPdf(pdfCat1, "cat1");
   simPdf.addPdf(pdfCat2, "cat2");

   // Generate datasets
   std::map<std::string, std::unique_ptr<RooDataSet>> datasetMap{};
   datasetMap["cat1"] = std::unique_ptr<RooDataSet>{pdfCat1.generate(RooArgSet(x), 11000)};
   datasetMap["cat2"] = std::unique_ptr<RooDataSet>{pdfCat2.generate(RooArgSet(x), 11000)};
   RooDataSet combData("combData", "", RooArgSet(x), Index(indexCat), Import(datasetMap));

   std::unique_ptr<RooAbsReal> nll{simPdf.createNLL(combData, Range("range1,range2"))};
}

/// GitHub issue #10473.
/// Crash when RooSimultaneous does not contain a pdf for each value of the index category.
TEST(RooSimultaneous, CategoriesWithNoPdf)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooRealVar x("x", "", 0, 1);
   RooRealVar rnd("rnd", "", 0, 1);
   RooThresholdCategory catThreshold("cat", "", rnd, "v2", 2);
   catThreshold.addThreshold(1. / 3, "v0", 0);
   catThreshold.addThreshold(2. / 3, "v1", 1);

   RooRealVar m0("m0", "", 0.5, 0, 1);
   RooRealVar m1("m1", "", 0.5, 0, 1);
   RooGenericPdf g0("g0", "", "std::exp(-0.5*(x - m0)^2/0.01)", {x, m0});
   RooGenericPdf g1("g1", "", "std::exp(-0.5*(x - m1)^2/0.01)", {x, m1});
   RooGenericPdf rndPdf("rndPdf", "", "1", {});
   RooProdPdf pdf("pdf", "", RooArgSet(g0, rndPdf));

   std::unique_ptr<RooDataSet> ds{pdf.generate(RooArgSet(x, rnd), RooFit::Name("ds"), RooFit::NumEvents(100))};
   auto cat = dynamic_cast<RooCategory *>(ds->addColumn(catThreshold));

   RooSimultaneous sim("sim", "", *cat);
   sim.addPdf(g0, "v0");
   sim.addPdf(g1, "v1");

   // We don't care about the fit result, just that it doesn't crash.
   using namespace RooFit;
#ifdef ROOFIT_LEGACY_EVAL_BACKEND
   sim.fitTo(*ds, EvalBackend::Legacy(), PrintLevel(-1));
   m0.setVal(0.5);
   m0.setError(0.0);
   m1.setVal(0.5);
   m1.setError(0.0);
#endif
   sim.fitTo(*ds, EvalBackend::Cpu(), PrintLevel(-1));
}

/// GitHub issue #11396.
/// Test whether the RooFit::SplitRange() command argument for simultaneous
/// fits is correctly considered in multi-range fits.
TEST(RooSimultaneous, MultiRangeFitWithSplitRange)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   using namespace RooFit;

   int nEventsInCat = 11000;

   RooWorkspace wsCat1{"wsCat1"};
   wsCat1.factory("Gaussian::pdf_cat1(x_cat1[0,10],mu_cat1[4,0,10],sigma_cat1[1.0,0.1,10.0])");
   RooAbsPdf &pdfCat1 = *wsCat1.pdf("pdf_cat1");
   RooRealVar &xCat1 = *wsCat1.var("x_cat1");
   xCat1.setRange("SideBandLo_cat1", 0, 2);
   xCat1.setRange("SideBandHi_cat1", 6, 10);
   std::unique_ptr<RooDataSet> dsCat1{pdfCat1.generate(xCat1, nEventsInCat)};

   RooWorkspace wsCat2{"wsCat2"};
   wsCat2.factory("Gaussian::pdf_cat2(x_cat2[0,10],mu_cat2[6,0,10],sigma_cat2[1.0,0.1,10.0])");
   RooAbsPdf &pdfCat2 = *wsCat2.pdf("pdf_cat2");
   RooRealVar &xCat2 = *wsCat2.var("x_cat2");
   xCat2.setRange("SideBandLo_cat2", 0, 4);
   xCat2.setRange("SideBandHi_cat2", 8, 10);
   std::unique_ptr<RooDataSet> dsCat2{pdfCat2.generate(xCat2, nEventsInCat)};

   RooCategory indexCat{"cat", "cat"};
   indexCat.defineType("cat1");
   indexCat.defineType("cat2");

   RooSimultaneous simPdf{"simPdf", "", indexCat};
   simPdf.addPdf(pdfCat1, "cat1");
   simPdf.addPdf(pdfCat2, "cat2");

   std::map<std::string, RooDataSet *> dsmap{{"cat1", dsCat1.get()}, {"cat2", dsCat2.get()}};

   RooDataSet combData{"combData", "", {xCat1, xCat2}, Index(indexCat), Import(dsmap)};

   const char *cutRange1 = "SideBandLo_cat1,SideBandHi_cat1";
   const char *cutRange2 = "SideBandLo_cat2,SideBandHi_cat2";
   using RealPtr = std::unique_ptr<RooAbsReal>;
#ifdef ROOFIT_LEGACY_EVAL_BACKEND
   RealPtr nllSim{simPdf.createNLL(combData, Range("SideBandLo,SideBandHi"), SplitRange(), EvalBackend::Legacy())};
#endif
   RealPtr nllSimBatch{simPdf.createNLL(combData, Range("SideBandLo,SideBandHi"), SplitRange(), EvalBackend::Cpu())};

   // In simultaneous PDFs, the probability is normalized over the categories,
   // so we have to do that as well when computing the reference value. Since
   // we do a ranged fit, we have to consider the ranges when calculating the
   // number of events in data.
   double n1 = std::unique_ptr<RooAbsData>(dsCat1->reduce(CutRange(cutRange1)))->sumEntries();
   double n2 = std::unique_ptr<RooAbsData>(dsCat2->reduce(CutRange(cutRange2)))->sumEntries();
   const double normTerm = (n1 + n2) * std::log(2);

   std::unique_ptr<RooAbsReal> nll1{pdfCat1.createNLL(*dsCat1, Range(cutRange1))};
   std::unique_ptr<RooAbsReal> nll2{pdfCat2.createNLL(*dsCat2, Range(cutRange2))};
   RooAddition nllSimRef{"nllSimRef", "nllSimRef", {*nll1, *nll2, RooConst(normTerm)}};

   const double nllSimRefVal = nllSimRef.getVal();
#ifdef ROOFIT_LEGACY_EVAL_BACKEND
   const double nllSimVal = nllSim->getVal();
   EXPECT_FLOAT_EQ(nllSimVal, nllSimRefVal);
#endif
   const double nllSimBatchVal = nllSimBatch->getVal();
   EXPECT_FLOAT_EQ(nllSimBatchVal, nllSimRefVal) << "BatchMode and old RooFit don't agree!";
}

class TestStatisticTest : public testing::TestWithParam<std::tuple<RooFit::EvalBackend>> {
public:
   TestStatisticTest() : _evalBackend{RooFit::EvalBackend::Legacy()} {}

private:
   void SetUp() override
   {
      RooRandom::randomGenerator()->SetSeed(1337ul);
      _evalBackend = std::get<0>(GetParam());
      _changeMsgLvl = std::make_unique<RooHelpers::LocalChangeMsgLevel>(RooFit::WARNING);
   }

   void TearDown() override { _changeMsgLvl.reset(); }

protected:
   RooFit::EvalBackend _evalBackend;

private:
   std::unique_ptr<RooHelpers::LocalChangeMsgLevel> _changeMsgLvl;
};

/// GitHub issue #8307.
/// A likelihood with a model wrapped in a RooSimultaneous in one category
/// should give the same results as the likelihood with the model directly. We
/// also test that things go well if you wrap the simultaneous NLL again in
/// another class, which can happen in user frameworks.
TEST_P(TestStatisticTest, RooSimultaneousSingleChannelCrossCheck)
{
   using namespace RooFit;

   // silence log output
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooWorkspace ws;
   ws.factory("Gaussian::gauss1(x[0, 10], mean[1., 0., 10.], width[1, 0.1, 10])");
   ws.factory("AddPdf::model({gauss1}, {nsig[500, 100, 1000]})");
   ws.factory("Gaussian::fconstraint(2.0, mean, 0.2)");
   ws.factory("ProdPdf::modelConstrained({model, fconstraint})");

   RooRealVar &x = *ws.var("x");
   RooAbsPdf &model = *ws.pdf("model");
   RooAbsPdf &modelConstrained = *ws.pdf("modelConstrained");

   RooCategory cat("cat", "cat");
   cat.defineType("physics");

   RooSimultaneous modelSim("modelSim", "modelSim", RooArgList{modelConstrained}, cat);

   std::unique_ptr<RooDataSet> data{model.generate(x)};

   RooArgSet params;
   RooArgSet initialParams;
   modelConstrained.getParameters(data->get(), params);
   params.snapshot(initialParams);

   RooDataSet combData("combData", "combData", x, Index(cat), Import("physics", *data));

   using AbsRealPtr = std::unique_ptr<RooAbsReal>;

   AbsRealPtr nllDirect{modelConstrained.createNLL(combData, _evalBackend)};
   AbsRealPtr nllSimWrapped{modelSim.createNLL(combData, _evalBackend)};
   RooAddition nllAdditionWrapped{"nll_wrapped_cpu", "nll_wrapped_cpu", {*nllSimWrapped}};

   auto minimize = [&](RooAbsReal &nll) {
      params.assign(initialParams);
      RooMinimizer minim{nll};
      minim.setPrintLevel(-1);
      minim.minimize("", "");
      return std::unique_ptr<RooFitResult>{minim.save()};
   };

   std::unique_ptr<RooFitResult> resDirect{minimize(*nllDirect)};
   std::unique_ptr<RooFitResult> resSimWrapped{minimize(*nllSimWrapped)};
   std::unique_ptr<RooFitResult> resAdditionWrapped{minimize(nllAdditionWrapped)};

   EXPECT_TRUE(resSimWrapped->isIdentical(*resDirect)) << "Inconsistency in RooSimultaneous wrapping";
   EXPECT_TRUE(resAdditionWrapped->isIdentical(*resDirect))
      << "Inconsistency in RooSimultaneous + RooAddition wrapping";
}

/// Checks that the Range() command argument for fitTo can be used to select
/// specific components from a RooSimultaneous. Covers GitHub issue #8231.
TEST_P(TestStatisticTest, RangedCategory)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   using namespace RooFit;

   // Create the model with the RooWorkspace to not explicitly depend on
   // non-RooFitCore classes like RooGaussian
   RooWorkspace ws;
   ws.factory("Gaussian::pdfA(x[-10, 10], mu[0, -10, 10], sigma[2.0, 0.1, 10.0])");
   ws.factory("Gaussian::pdfB(x,          mu[0, -10, 10], sigma[2.0, 0.1, 10.0])");
   ws.factory("SIMUL::simPdf( cat[A=0,B=1], A=pdfA, B=pdfB)");

   // Get the objects from the workspace
   RooRealVar &x = *ws.var("x");
   RooRealVar &mu = *ws.var("mu");
   RooRealVar &sigma = *ws.var("sigma");
   RooAbsPdf &pdfA = *ws.pdf("pdfA");
   RooAbsPdf &pdfB = *ws.pdf("pdfB");
   RooAbsPdf &simPdf = *ws.pdf("simPdf");
   RooCategory &cat = *ws.cat("cat");

   // Generate combined toy dataset. The toy dataset is designed such that we
   // can easily check in the fit result which category states were used in
   // the fit:
   //   * Only state A: mu has to be compatible with -1.0
   //   * Only state B: mu has to be compatible with +1.0
   //   * Both states: mu has to be compatible with 0.0, as merging both
   //                  states results roughly in a Gaussian with mean 0.0.
   mu.setVal(-1.0);
   std::unique_ptr<RooDataSet> dataA{pdfA.generate(x, 5000)};
   mu.setVal(+1.0);
   std::unique_ptr<RooDataSet> dataB{pdfB.generate(x, 5000)};
   mu.setVal(0.0);
   RooDataSet data{"data", "data", x, Index(cat), Import({{"A", dataA.get()}, {"B", dataB.get()}})};

   // Define the category ranges
   cat.setRange("rA", "A");
   cat.setRange("rB", "B");
   cat.setRange("rAB", "A,B");

   // Function to reset parameters after fitting
   auto resetParameters = [&]() {
      mu.setVal(0.0);
      mu.setError(0.0);
      sigma.setVal(2.0);
      sigma.setError(0.0);
   };

   // Function to do the fit
   auto doFit = [&](RooAbsPdf &pdf, RooAbsData &dataset, const char *range = nullptr) {
      resetParameters();
      std::unique_ptr<RooFitResult> res{pdf.fitTo(dataset, Range(range), Save(), PrintLevel(-1), _evalBackend)};
      resetParameters();
      return res;
   };

   // Do fits in different configurations
   auto res = doFit(simPdf, data);
   auto resA = doFit(simPdf, data, "rA");
   auto resB = doFit(simPdf, data, "rB");
   auto resAB = doFit(simPdf, data, "rAB");
   auto resAref = doFit(pdfA, *dataA);
   auto resBref = doFit(pdfB, *dataB);

   // Validate the results
   EXPECT_TRUE(resA->isIdentical(*resAref)) << "Selecting only state A didn't work!";
   EXPECT_TRUE(resB->isIdentical(*resBref)) << "Selecting only state B didn't work!";
   EXPECT_TRUE(resAB->isIdentical(*res)) << "Result when selecting all states inconsistent with default fit!";
}

INSTANTIATE_TEST_SUITE_P(RooSimultaneous, TestStatisticTest, testing::Values(ROOFIT_EVAL_BACKENDS),
                         [](testing::TestParamInfo<TestStatisticTest::ParamType> const &paramInfo) {
                            std::stringstream ss;
                            ss << "EvalBackend" << std::get<0>(paramInfo.param).name();
                            return ss.str();
                         });

/// Check that the dataset generation from a nested RooSimultaneous with
/// protodata containing the category values works.
/// Covers GitHub issue #12020.
TEST(RooSimultaneous, NestedSimPdfGenContext)
{
   RooHelpers::LocalChangeMsgLevel locmsg(RooFit::WARNING, 0u, RooFit::InputArguments, false);

   RooRealVar x{"x", "", 0, 1};

   // It's important that there are different values for the first (inner)
   // category such that we can test that the different values are correctly
   // picked up from the proto dataset.
   RooCategory c1{"c1", ""};
   c1.defineType("c11", 1);
   c1.defineType("c12", 2);

   RooCategory c2{"c2", ""};
   c2.defineType("c21", 1);

   RooGenericPdf u11{"u11", "1.0", {}};
   RooGenericPdf u12{"u12", "1.0", {}};

   RooSimultaneous s1("s1", "", {{"c11", &u11}, {"c12", &u12}}, c1);
   RooSimultaneous s2("s2", "", {{"c21", &s1}}, c2);

   RooArgSet categories{c1, c2};
   RooDataSet proto{"proto", "", categories};

   c1.setIndex(1);
   proto.add(categories);

   c1.setIndex(2);
   proto.add(categories);

   std::unique_ptr<RooDataSet> data2{s2.generate(x, RooFit::ProtoData(proto))};

   auto catIndex = [](RooArgSet const *vars, const char *name) {
      return static_cast<RooAbsCategory *>(vars->find(name))->getCurrentIndex();
   };

   // If all went well, the category values are taken from the proto dataset
   EXPECT_EQ(catIndex(data2->get(0), "c1"), catIndex(proto.get(0), "c1"));
   EXPECT_EQ(catIndex(data2->get(0), "c2"), catIndex(proto.get(0), "c2"));
   EXPECT_EQ(catIndex(data2->get(1), "c1"), catIndex(proto.get(1), "c1"));
   EXPECT_EQ(catIndex(data2->get(1), "c2"), catIndex(proto.get(1), "c2"));
}

/// Make sure that putting a conditional RooProdPdf in a RooSimultaneous
/// doesn't result in a messed up computation graph with unnecessary integrals.
/// Covers GitHub issue #15751.
TEST(RooSimultaneous, ConditionalProdPdf)
{
   RooRealVar x{"x", "x", 0, 1};
   RooRealVar y{"y", "y", 0, 1};

   RooGenericPdf pdfx{"pdfx", "1.0 + x - x", {x}};
   RooGenericPdf pdfxy{"pdfxy", "1.0 + x - x + y - y", {x, y}};

   RooProdPdf pdf{"pdf", "pdf", pdfx, RooFit::Conditional(pdfxy, y)};

   RooArgSet normSet{x, y};

   RooCategory cat{"cat", "cat", {{"0", 0}}};
   RooSimultaneous simPdf{"simPdf", "simPdf", {{"0", &pdf}}, cat};

   auto countGraphNodes = [](RooAbsArg &arg) {
      RooArgList nodes;
      arg.treeNodeServerList(&nodes);
      return nodes.size();
   };

   RooFit::Detail::CompileContext ctx{normSet};
   RooFit::Detail::CompileContext ctxSim{normSet};

   std::unique_ptr<RooAbsPdf> compiled{static_cast<RooAbsPdf *>(pdf.compileForNormSet(normSet, ctx).release())};
   std::unique_ptr<RooAbsPdf> compiledSim{
      static_cast<RooAbsPdf *>(simPdf.compileForNormSet(normSet, ctxSim).release())};

   // We expect only two more nodes in the computation graph: one for the
   // RooSimultaneous, and one for the RooCategory.
   EXPECT_EQ(countGraphNodes(*compiledSim), countGraphNodes(*compiled) + 2);
}

// Test that we can evaluate a RooSimultaneous also if only a fraction of the
// channels can be extended. Also check if the likelihood can be created.
TEST(RooSimultaneous, PartiallyExtendedPdfs)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooWorkspace ws;
   ws.factory("Gaussian::pdfA(x_a[-10, 10], mu_a[0, -10, 10], sigma_a[2.0, 0.1, 10.0])");
   ws.factory("Gaussian::pdfB(x_b[-10, 10], mu_b[0, -10, 10], sigma_b[2.0, 0.1, 10.0])");
   ws.factory("PROD::pdfAprod(pdfA)");
   ws.factory("ExtendPdf::pdfBext(pdfB, n_b[1000., 100., 10000.])");
   ws.factory("SIMUL::simPdf( cat[A=0,B=1], A=pdfAprod, B=pdfBext)");

   RooArgSet observables{*ws.var("x_a"), *ws.var("x_b"), *ws.cat("cat")};

   auto &simPdf = *ws.pdf("simPdf");

   // A completely extended pdf, just to easily create a toy dataset
   ws.factory("ExtendPdf::pdfAext(pdfA, n_b[1000., 100., 10000.])");
   ws.factory("SIMUL::simPdfExtBoth( cat[A=0,B=1], A=pdfAext, B=pdfBext)");
   std::unique_ptr<RooDataSet> data{ws.pdf("simPdfExtBoth")->generate(observables)};

   // Check if likelihood can be instantiated
   std::unique_ptr<RooAbsReal> nll{simPdf.createNLL(*data)};
}

// Make sure that one can use the same extended pdf instance for different
// channels, and the RooSimultaneous will still evaluate correctly.
TEST(RooSimultaneous, DuplicateExtendedPdfs)
{
   RooWorkspace ws;

   ws.factory("Uniform::u_a(x[0, 10])");
   ws.factory("Uniform::u_b(x)");
   ws.factory("ExtendPdf::pdf_a(u_a, n[1000, 100, 10000])");
   ws.factory("ExtendPdf::pdf_b(u_b, n)");

   ws.factory("SIMUL::simPdf( c[A=0,B=1], A=pdf_a, B=pdf_a)");
   ws.factory("SIMUL::simPdfRef( c, A=pdf_a, B=pdf_b)");

   RooArgSet normSet{*ws.var("x")};

   RooAbsPdf &simPdf = *ws.pdf("simPdf");
   RooAbsPdf &simPdfRef = *ws.pdf("simPdfRef");
   double simPdfVal = simPdf.getVal(normSet);

   EXPECT_FLOAT_EQ(simPdfVal, 0.05);
   EXPECT_DOUBLE_EQ(simPdfVal, simPdfRef.getVal(normSet));
}

/// GitHub issue #19166.
/// RooSimultaneous::fitTo() should work well with ConditionalOvservables.
/// We test this by performing the similar test to a test case in
/// RooSimultaneousSingleChannelCrossCheck.
TEST_P(TestStatisticTest, RooSimultaneousSingleChannelCrossCheckWithCondVar)
{
   using namespace RooFit;

   // silence log output
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooWorkspace ws;
   ws.factory("Uniform::uniform(width[1, 3])");
   ws.factory("expr::sigma(\"@0*@1\", width, width_scale[0.8, 0.5, 1.5])");
   ws.factory("Gaussian::model(x[-20, 20], mean[3, -20., 20.], sigma)");

   RooRealVar &width = *ws.var("width");
   RooAbsPdf &widthModel = *ws.pdf("uniform");
   std::unique_ptr<RooDataSet> protoData{widthModel.generate(width, 100)};

   RooRealVar &x = *ws.var("x");
   RooAbsPdf &model = *ws.pdf("model");

   std::unique_ptr<RooDataSet> data{model.generate(x, *protoData)};

   RooCategory cat("cat", "cat");
   cat.defineType("physics");

   RooArgSet params;
   RooArgSet initialParams;
   model.getParameters(data->get(), params);
   params.snapshot(initialParams);

   RooSimultaneous modelSim("modelSim", "modelSim", {{"physics", &model}}, cat);

   RooDataSet combData("combData", "combData", {x, width}, Index(cat), Import({{"physics", data.get()}}));

   using namespace RooFit;

   params.assign(initialParams);
   std::unique_ptr<RooFitResult> resDirect{
      model.fitTo(*data, ConditionalObservables(width), Save(), PrintLevel(-1), _evalBackend)};

   params.assign(initialParams);
   std::unique_ptr<RooFitResult> resSimWrapped{
      modelSim.fitTo(combData, ConditionalObservables(width), Save(), PrintLevel(-1), _evalBackend)};

   EXPECT_TRUE(resSimWrapped->isIdentical(*resDirect))
      << "Inconsistency in RooSimultaneous wrapping with ConditionalObservables";
}

/// GitHub issue #18718.
/// Make sure that we can do a ranged fit on an extended RooAddPdf in a
/// RooSimultaneous with the new CPU backend.
TEST(RooSimultaneous, RangedExtendedRooAddPdf)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLevel{RooFit::WARNING};

   const double nBkgA_nom = 9000;
   const double nBkgB_nom = 10000;

   RooRealVar x("x", "Observable", 100, 150);
   x.setRange("fitRange", 100, 130);

   RooRealVar nBkgA("nBkgA", "", nBkgA_nom, 0.8 * nBkgA_nom, 1.2 * nBkgA_nom);
   RooRealVar nBkgB("nBkgB", "", nBkgB_nom, 0.8 * nBkgB_nom, 1.2 * nBkgB_nom);

   RooExponential expA("expA", "", x, RooFit::RooConst(-0.06));
   RooAddPdf modelA("modelA", "", {expA}, {nBkgA});

   RooExponential expB("expB", "", x, RooFit::RooConst(-0.09));
   RooAddPdf modelB("modelB", "", {expB}, {nBkgB});

   RooCategory runCat("runCat", "", {{"RunA", 0}, {"RunB", 1}});

   RooSimultaneous simPdf("simPdf", "", {{"RunA", &modelA}, {"RunB", &modelB}}, runCat);

   using namespace RooFit;

   std::unique_ptr<RooDataSet> combData{simPdf.generate(RooArgSet(x, runCat), Extended())};

   using Res = std::unique_ptr<RooFitResult>;

   RooArgSet params;
   RooArgSet paramsSnap;
   simPdf.getParameters(combData->get(), params);
   params.snapshot(paramsSnap);

   Res fitResult{simPdf.fitTo(*combData, Save(), Range("fitRange"), EvalBackend(EvalBackend::Cpu()), PrintLevel(-1))};

   params.assign(paramsSnap);

   Res fitResultRef{
      simPdf.fitTo(*combData, Save(), Range("fitRange"), EvalBackend(EvalBackend::Legacy()), PrintLevel(-1))};

   EXPECT_TRUE(fitResult->isIdentical(*fitResultRef));
}

/// GitHub issue #20383.
/// Check that the the simultaneous pdf is normalized correctly when plotting
/// with a projection dataset.
TEST(RooSimultaneous, PlotProjWData)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooRealVar x("x", "x", -8, 8);
   x.setBins(1);

   RooUniform model{"model", "", x};
   RooUniform model_ctl{"model_ctl", "", x};

   RooCategory sample("sample", "sample", {{"physics", 0}, {"control", 1}});

   RooArgSet vars{x, sample};
   RooDataHist combData{"combData", "", vars};
   sample.setLabel("physics");
   combData.add(vars, 1000);
   sample.setLabel("control");
   combData.add(vars, 2000);

   RooSimultaneous simPdf("simPdf", "simultaneous pdf", {{"physics", &model}, {"control", &model_ctl}}, sample);

   RooPlot *frame = x.frame();
   combData.plotOn(frame);
   simPdf.plotOn(frame, RooFit::ProjWData(sample, combData));

   // The pdf should be normalized to match the data. In this test, we plot a
   // single bin and the model is uniform, to the curve should be equal to the
   // sum of data entries in the center.
   EXPECT_DOUBLE_EQ(frame->getCurve()->interpolate(0.), combData.sumEntries());
}

/// Second part of GitHub issue #20383.
/// Check that the the simultaneous pdf is normalized correctly to the data
/// when plotting with a projection dataset, in the extended and non-extended
/// case, based on the reproducer provided by the user who opened the issue.
TEST(RooSimultaneous, PlotProjWDataExtended)
{
   using namespace RooFit;

   RooHelpers::LocalChangeMsgLevel changeMsgLevel{RooFit::WARNING};

   RooRealVar xvar1("x1", "", 0.0, 10.0);
   RooRealVar xvar2("x2", "", 0.0, 10.0);

   RooRealVar mean1("mean1", "", 3.0, 1.0, 4.0);
   RooRealVar mean2("mean2", "", 5.0, 4.0, 6.0);
   RooRealVar sigma("sigma", "", 1.0, 0.01, 2.0);

   RooGaussian gauss1("gauss1", "", xvar1, mean1, sigma);
   RooGaussian gauss2("gauss2", "", xvar2, mean2, sigma);

   RooRealVar a0("a0", "a0", -0.1, -1.0, 1.0);
   RooChebychev bkg1("bkg1", "b1", xvar1, {a0});

   RooRealVar c0("c0", "c0", -0.1, -1.0, 1.0);
   RooChebychev bkg2("bkg2", "b2", xvar2, {c0});

   RooRealVar s1("s1", "s1", 75.0, 0.0, 100000.0);
   RooRealVar b1("b1", "b1", 25.0, 0.0, 100000.0);

   // Extended models
   RooAddPdf model1e("model1e", "", {gauss1, bkg1}, {s1, b1});

   RooRealVar s2("s2", "s2", 50.0, 0.0, 100000.0);
   RooRealVar b2("b2", "b2", 50.0, 0.0, 100000.0);

   RooAddPdf model2e("model2e", "model2e", {gauss2, bkg2}, {s2, b2});

   // Non-extended models
   RooRealVar f1("f1", "f1", 0.75, 0.0, 1.0);
   RooAddPdf model1n("model1n", "model1n", {gauss1, bkg1}, {f1});

   RooRealVar f2("f2", "f2", 0.50, 0.0, 1.0);
   RooAddPdf model2n("model2n", "model2n", RooArgList(gauss2, bkg2), RooArgList(f2));

   // Case handling
   enum class Case {
      Gaussian,
      NonExtended,
      Extended
   };

   auto caseName = [](Case c) {
      switch (c) {
      case Case::Gaussian: return "Gaussian";
      case Case::NonExtended: return "NonExtended";
      case Case::Extended: return "Extended";
      }
      return "Unknown";
   };

   const std::vector<Case> cases = {Case::Gaussian, Case::NonExtended, Case::Extended};

   // Helpers
   auto integrateLastCurve = [](RooPlot *plot) {
      const double xmin = plot->getPlotVar()->getMin();
      const double xmax = plot->getPlotVar()->getMax();
      return plot->getCurve()->average(xmin, xmax) * plot->getPlotVar()->numBins();
   };

   constexpr double tol = 0.01; // tolerate 1 % sampling error

   // Test body
   auto runCase = [&](Case c) {
      SCOPED_TRACE(std::string("Case = ") + caseName(c));

      RooAbsPdf *model1 = nullptr;
      RooAbsPdf *model2 = nullptr;

      switch (c) {
      case Case::Gaussian:
         model1 = &gauss1;
         model2 = &gauss2;
         break;
      case Case::NonExtended:
         model1 = &model1n;
         model2 = &model2n;
         break;
      case Case::Extended:
         model1 = &model1e;
         model2 = &model2e;
         break;
      }

      ASSERT_NE(model1, nullptr);
      ASSERT_NE(model2, nullptr);

      // Generate data
      std::unique_ptr<RooDataSet> data1{model1->generate(xvar1, 10000)};
      std::unique_ptr<RooDataSet> data2{model2->generate(xvar2, 1000)};

      // Category
      RooCategory sample("sample", "");
      sample.defineType("Fit1");
      sample.defineType("Fit2");

      RooDataSet data("combinedData", "", {xvar1, xvar2}, Index(sample),
                      Import({{"Fit1", data1.get()}, {"Fit2", data2.get()}}));

      // Simultaneous PDF
      RooSimultaneous sim_pdf("sim_pdf", "", sample);
      sim_pdf.addPdf(*model1, "Fit1");
      sim_pdf.addPdf(*model2, "Fit2");

      std::unique_ptr<RooFitResult> result{sim_pdf.fitTo(data, Save(true), PrintLevel(-1))};

      // Plot + checks
      RooPlot *frame1 = xvar1.frame();
      data.plotOn(frame1, Cut("sample==sample::Fit1"));
      sim_pdf.plotOn(frame1, Slice(sample, "Fit1"), ProjWData(sample, data));

      EXPECT_THAT(integrateLastCurve(frame1), RelativeNear(data1->sumEntries(), tol));

      RooPlot *frame2 = xvar2.frame();
      data.plotOn(frame2, Cut("sample==sample::Fit2"));
      sim_pdf.plotOn(frame2, Slice(sample, "Fit2"), ProjWData(sample, data));

      EXPECT_THAT(integrateLastCurve(frame2), RelativeNear(data2->sumEntries(), tol));
   };

   // Execute
   for (Case c : cases) {
      runCase(c);
   }
}

/// JIRA ticket https://its.cern.ch/jira/browse/ROOT-7499
/// Check that we can also generate Asimov datasets with non-integer weights
/// via RooSimultaneous.
TEST(RooSimultaneous, ExpectedDataWithNonIntegerWeights)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLevel{RooFit::WARNING};

   RooWorkspace ws{"ws"};
   ws.factory("dummy_obs_a[0,1]");
   ws.factory("dummy_obs_b[0,1]");
   ws.factory("Uniform::uniform_a(dummy_obs_a)");
   ws.factory("Uniform::uniform_b(dummy_obs_b)");
   ws.factory("SUM::model_a(coeff_a[3.5]*uniform_a)");
   ws.factory("SUM::model_b(coeff_b[6.5]*uniform_b)");

   RooRealVar &dummy_obs_a = *ws.var("dummy_obs_a");
   RooRealVar &dummy_obs_b = *ws.var("dummy_obs_b");

   ws.factory("dummy_cat[a]");
   ws.factory("SIMUL::sim_model(dummy_cat, a = model_a, b = model_b)");
   RooAbsCategory &dummy_cat = *ws.cat("dummy_cat");

   // std::cout << "simultaneous expected = " << ws.pdf("sim_model")->expectedEvents(dummy_obs) << std::endl;
   RooDataSet *data = ws.pdf("sim_model")->generate({dummy_obs_a, dummy_obs_b, dummy_cat}, RooFit::ExpectedData());

   std::unique_ptr<Roo1DTable> tab{data->table(dummy_cat)};

   // Check that the sum of entries for each category is as expected, matching
   // the coefficients from the RooAddPdf.
   EXPECT_FLOAT_EQ(tab->get("a"), ws.var("coeff_a")->getVal());
   EXPECT_FLOAT_EQ(tab->get("b"), ws.var("coeff_b")->getVal());
}
