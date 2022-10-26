// Tests for the RooSimultaneous
// Authors: Jonas Rembser, CERN  06/2021

#include <RooAddition.h>
#include <RooAddPdf.h>
#include <RooConstVar.h>
#include <RooCategory.h>
#include <RooDataSet.h>
#include <RooGenericPdf.h>
#include <RooRealVar.h>
#include <RooSimultaneous.h>
#include <RooProdPdf.h>
#include <RooWorkspace.h>
#include <RooThresholdCategory.h>

#include <gtest/gtest.h>

#include <memory>

/// GitHub issue #8307.
/// A likelihood with a model wrapped in a RooSimultaneous in one category
/// should give the same results as the likelihood with the model directly.
TEST(RooSimultaneous, SingleChannelCrossCheck)
{
   using namespace RooFit;

   // silence log output
   RooMsgService::instance().setGlobalKillBelow(RooFit::WARNING);

   RooRealVar x("x", "x", 0, 10);
   RooRealVar mean("mean", "mean", 1., 0, 10);
   RooRealVar width("width", "width", 1, 0.1, 10);
   RooRealVar nsig("nsig", "nsig", 500, 100, 1000);

   RooGenericPdf gauss1("guass1", "gauss1", "std::exp(-0.5*(x - mean)^2/width^2)", {x, mean, width});
   RooGenericPdf fconstraint("fconstraint", "fconstraint", "std::exp(-0.5*(mean - 2.0)^2/0.2^2)", {mean});

   RooAddPdf model("model", "model", RooArgList(gauss1), RooArgList(nsig));
   RooProdPdf modelConstrained("modelConstrained", "modelConstrained", RooArgSet(model, fconstraint));

   RooCategory cat("cat", "cat");
   cat.defineType("physics");

   RooSimultaneous modelSim("modelSim", "modelSim", RooArgList{modelConstrained}, cat);

   std::unique_ptr<RooDataSet> data{model.generate(x)};
   RooDataSet combData("combData", "combData", x, Index(cat), Import("physics", *data));

   using AbsRealPtr = std::unique_ptr<RooAbsReal>;

   AbsRealPtr nllDirect{modelConstrained.createNLL(combData)};
   AbsRealPtr nllSimWrapped{modelSim.createNLL(combData)};
   AbsRealPtr nllDirectBatch{modelConstrained.createNLL(combData, BatchMode("cpu"))};
   AbsRealPtr nllSimWrappedBatch{modelSim.createNLL(combData, BatchMode("cpu"))};

   EXPECT_FLOAT_EQ(nllDirect->getVal(), nllSimWrapped->getVal()) << "Inconsistency in old RooFit";
   EXPECT_FLOAT_EQ(nllDirect->getVal(), nllDirectBatch->getVal()) << "Old RooFit and BatchMode don't agree";
   EXPECT_FLOAT_EQ(nllDirectBatch->getVal(), nllSimWrappedBatch->getVal()) << "Inconsistency in BatchMode";
}

/// Forum issue
/// https://root-forum.cern.ch/t/roofit-failed-to-create-nll-for-simultaneous-pdfs-with-multiple-range-names/49363.
/// Multi-range likelihoods should also work with RooSimultaneous, where one
/// of the observables is a category.
TEST(RooSimultaneous, MultiRangeNLL)
{
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
   std::map<std::string, RooDataSet *> datasetMap{};
   datasetMap["cat1"] = pdfCat1.generate(RooArgSet(x), 11000);
   datasetMap["cat2"] = pdfCat2.generate(RooArgSet(x), 11000);
   RooDataSet combData("combData", "", RooArgSet(x), Index(indexCat), Import(datasetMap));

   std::unique_ptr<RooAbsReal> nll{simPdf.createNLL(combData, Range("range1,range2"))};
}

/// GitHub issue #10473.
/// Crash when RooSimultaneous does not contain a pdf for each value of the index category.
TEST(RooSimultaneous, CategoriesWithNoPdf)
{
   RooRealVar x("x", "", 0, 1);
   RooRealVar rnd("rnd", "", 0, 1);
   RooThresholdCategory catThr("cat", "", rnd, "v2", 2);
   catThr.addThreshold(1. / 3, "v0", 0);
   catThr.addThreshold(2. / 3, "v1", 1);

   RooRealVar m0("m0", "", 0.5, 0, 1);
   RooRealVar m1("m1", "", 0.5, 0, 1);
   RooGenericPdf g0("g0", "", "std::exp(-0.5*(x - m0)^2/0.01)", {x, m0});
   RooGenericPdf g1("g1", "", "std::exp(-0.5*(x - m1)^2/0.01)", {x, m1});
   RooGenericPdf rndPdf("rndPdf", "", "1", {});
   RooProdPdf pdf("pdf", "", RooArgSet(g0, rndPdf));

   auto ds = pdf.generate(RooArgSet(x, rnd), RooFit::Name("ds"), RooFit::NumEvents(100));
   auto cat = dynamic_cast<RooCategory *>(ds->addColumn(catThr));

   RooSimultaneous sim("sim", "", *cat);
   sim.addPdf(g0, "v0");
   sim.addPdf(g1, "v1");

   // We don't care about the fit result, just that it doesn't crash.
   using namespace RooFit;
   sim.fitTo(*ds, BatchMode(false), PrintLevel(-1));
   m0.setVal(0.5);
   m0.setError(0.0);
   m1.setVal(0.5);
   m1.setError(0.0);
   sim.fitTo(*ds, BatchMode(true), PrintLevel(-1));
}

/// GitHub issue #11396.
/// Test whether the RooFit::SplitRange() command argument for simultaneous
/// fits is correctly considered in multi-range fits.
TEST(RooSimultaneous, MultiRangeFitWithSplitRange)
{
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
   RealPtr nllSim{simPdf.createNLL(combData, Range("SideBandLo,SideBandHi"), SplitRange(), BatchMode("off"))};
   RealPtr nllSimBatch{simPdf.createNLL(combData, Range("SideBandLo,SideBandHi"), SplitRange(), BatchMode("cpu"))};

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
   const double nllSimVal = nllSim->getVal();
   const double nllSimBatchVal = nllSimBatch->getVal();

   EXPECT_FLOAT_EQ(nllSimVal, nllSimRefVal);
   EXPECT_FLOAT_EQ(nllSimBatchVal, nllSimVal) << "BatchMode and old RooFit don't agree!";
}
