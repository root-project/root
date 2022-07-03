// Tests for RooAddPdf
// Authors: Jonas Rembser, CERN 07/2022

#include <RooAddPdf.h>
#include <RooConstVar.h>
#include <RooExponential.h>
#include <RooGaussian.h>
#include <RooMsgService.h>
#include <RooProdPdf.h>

#include <RooStats/SPlot.h>

#include <gtest/gtest.h>

#include <memory>


/// Verify that sPlot does work with a RooAddPdf. This reproduces GitHub issue
/// #10869, where creating an SPlot from a RooAdPdf unreasonably changed the
/// parameter values because of RooAddPdf normalization issue. The reproducer
/// is taken from the GitHub issue thread, with the plotting part removed.
TEST(RooAddPdf, TestSPlot)
{
   auto& msg = RooMsgService::instance();
   msg.setGlobalKillBelow(RooFit::WARNING);

   double lowRange = 0.;
   double highRange = 200.;
   RooRealVar invMass("invMass", "M_{inv}", lowRange, highRange, "GeV");
   RooRealVar isolation("isolation", "isolation", 0., 20., "GeV");

   // mass model: linear combination of two Gaussians of different widths
   RooRealVar mZ("mZ", "Z Mass", 91.2, lowRange, highRange);
   RooRealVar sigmaZ("sigmaZ", "Width of Gaussian", 2, 0.1, 10, "GeV");
   RooGaussian mZModel1("mZModel1", "Z+jets Model", invMass, mZ, sigmaZ);

   RooRealVar sigmaZ2("sigmaZ2", "Width of Gaussian", 1.5, 0.01, 10, "GeV");
   RooGaussian mZModel2("mZModel2", "Z+jets Model", invMass, mZ, sigmaZ2);

   RooRealVar frac_1a("frac_1a", "frac_1a", 0.5); // fraction of the first Gaussian

   RooArgList shapes_DCB;
   shapes_DCB.add(mZModel1);
   shapes_DCB.add(mZModel2);

   RooArgList yields_DCB;
   yields_DCB.add(frac_1a);

   RooArgSet normSet(invMass);

   RooAddPdf mZModel("mZModel", "mZModel", shapes_DCB, yields_DCB);

   // isolation model for Z-boson.  Only used to generate toy MC.

   RooConstVar zIsolDecayConst("zIsolDecayConst", "z isolation decay  constant", -1);
   RooExponential zIsolationModel("zIsolationModel", "z isolation model", isolation, zIsolDecayConst);

   // make the combined Z model
   RooProdPdf zModel("zModel", "2-d model for Z", {mZModel, zIsolationModel});

   // make the QCD model

   // mass model for QCD.

   RooRealVar qcdMassDecayConst("qcdMassDecayConst", "Decay const for QCD mass spectrum", -0.01, -100, 100, "1/GeV");
   RooExponential qcdMassModel("qcdMassModel", "qcd Mass Model", invMass, qcdMassDecayConst);

   // isolation model for QCD.
   RooConstVar qcdIsolDecayConst("qcdIsolDecayConst", "Et resolution constant", -.1);
   RooExponential qcdIsolationModel("qcdIsolationModel", "QCD isolation model", isolation, qcdIsolDecayConst);

   // make the 2D model
   RooProdPdf qcdModel("qcdModel", "2-d model for QCD", {qcdMassModel, qcdIsolationModel});

   // --------------------------------------
   // combined model

   // These variables represent the number of Z or QCD events
   // They will be fitted.
   RooRealVar zYield("zYield", "fitted yield for Z", 300, 0., 10000);
   RooRealVar qcdYield("qcdYield", "fitted yield for QCD", 2700, 0., 10000);

   // now make the combined model
   // this is the 2D model for generation only
   RooAddPdf model_gen("model_gen", "mygreatmodel_gen", {zModel, qcdModel}, {zYield, qcdYield});
   // this is the mass model for the fit only
   RooAddPdf model("model", "mygreatmodel", {mZModel, qcdMassModel}, {zYield, qcdYield});

   int nEvents = zYield.getVal() + qcdYield.getVal();

   // make the toy data
   std::unique_ptr<RooDataSet> data{model_gen.generate({invMass, isolation}, nEvents, RooFit::Name("mygendataset"))};

   // fit the model to the data.
   model.fitTo(*data, RooFit::Extended(), RooFit::PrintLevel(-1));

   double valYieldBefore = zYield.getVal();

   // fix the parameters that are not yields before doing the sPlot
   qcdMassDecayConst.setConstant(true);
   sigmaZ.setConstant(true);
   sigmaZ2.setConstant(true);
   mZ.setConstant(true);

   RooStats::SPlot("sData", "An SPlot", *data, &model, {zYield, qcdYield});

   double valYieldAfter = zYield.getVal();

   EXPECT_NEAR(valYieldAfter, valYieldBefore, 1e-1)
      << "Doing the SPlot should not change parameter values by orders of magnitudes!";
}
