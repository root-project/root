// Tests for the RooSimultaneous
// Authors: Jonas Rembser, CERN  06/2021

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
TEST(RooSimultaneous, ImportFromTreeWithCut)
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

   RooArgSet constraints{fconstraint};

   std::unique_ptr<RooAbsReal> nllDirect{modelConstrained.createNLL(combData, Constrain(constraints))};
   std::unique_ptr<RooAbsReal> nllSimWrapped{modelSim.createNLL(combData, Constrain(constraints))};

   EXPECT_FLOAT_EQ(nllDirect->getVal(), nllSimWrapped->getVal());
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

   std::unique_ptr<RooAbsReal> nll{simPdf.createNLL(combData, Range("range1,range2"), SplitRange())};
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
