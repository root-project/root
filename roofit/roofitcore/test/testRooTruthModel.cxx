// Tests for the RooTruthModel
// Authors: Jonas Rembser, CERN 11/2023

#include <RooAbsAnaConvPdf.h>
#include <RooBDecay.h>
#include <RooConstVar.h>
#include <RooDecay.h>
#include <RooRealVar.h>
#include <RooTruthModel.h>
#include <RooWorkspace.h>

#include <TFile.h>

#include <gtest/gtest.h>

#include <iostream>
#include <memory>

/// Check that the integration over a subrange works when using an analytical
/// convolution with the RooTruthModel.
TEST(RooTruthModel, IntegrateSubrange)
{
   using namespace RooFit;

   RooRealVar dt{"dt", "dt", 0, 10};

   RooTruthModel truthModel{"tm", "truth model", dt};

   RooBDecay bcpg{"bcpg0",
                  "bcpg0",
                  dt,
                  RooConst(1.547),
                  RooConst(0.323206),
                  RooConst(1),
                  RooConst(1),
                  RooConst(1.547),
                  RooConst(1.547),
                  RooConst(0.472),
                  truthModel,
                  RooBDecay::DecayType::SingleSided};
   dt.setRange("integral", 2, 2);

   std::unique_ptr<RooAbsReal> integ{bcpg.createIntegral({dt}, "integral")};
   EXPECT_NEAR(integ->getVal(), 0.0, 1e-16);
}

/// The original resolution model passed to a RooAbsAnaConvPdf is only used to
/// build the internal resModel-times-basis convolutions; it must not become a
/// (non-value, non-shape) server of the pdf itself. Otherwise it pollutes the
/// computation graph and gets dragged into RooWorkspace and JSON exports.
TEST(RooAbsAnaConvPdf, ResolutionModelIsNotASpuriousServer)
{
   RooRealVar dt{"dt", "dt", -10, 10};
   RooRealVar tau{"tau", "tau", 1.548};
   RooTruthModel tm{"tm", "truth model", dt};
   RooDecay decay{"decay_tm", "decay", dt, tau, tm, RooDecay::SingleSided};

   for (RooAbsArg *server : decay.servers()) {
      // The original resolution model should not appear in the graph at all.
      EXPECT_STRNE(server->GetName(), tm.GetName())
         << "The original resolution model should not be a server of the RooDecay!";
      // Every remaining server must be a genuine value or shape server.
      EXPECT_TRUE(server->isValueServer(decay) || server->isShapeServer(decay))
         << "Server '" << server->GetName() << "' is neither a value nor a shape server of the RooDecay!";
   }

   // The resolution model must still be reachable (e.g. for JSON serialization).
   EXPECT_STREQ(decay.getModel().GetName(), tm.GetName());
}

/// Schema evolution test for RooAbsAnaConvPdf from class version 3 to 4.
///
/// In version 3, the original resolution model was stored in a RooRealProxy and
/// was therefore a (non-value, non-shape) server of the pdf. In version 4 it is
/// an owned, non-server object. Reading back a RooDecay from a version-3
/// workspace must give the same clean server structure as a freshly constructed
/// one, while staying numerically identical.
///
/// The fixture file was created with ROOT 6.40 (RooAbsAnaConvPdf v3) via:
/// \code
///   RooRealVar dt{"dt", "dt", -10, 10};
///   RooRealVar tau{"tau", "tau", 1.548};
///   RooTruthModel tm{"tm", "truth model", dt};
///   RooDecay decay_tm{"decay_tm", "decay", dt, tau, tm, RooDecay::SingleSided};
///   RooRealVar bias{"bias", "bias", 0.2};
///   RooRealVar sigma{"sigma", "sigma", 0.9};
///   RooGaussModel gm{"gm", "gauss model", dt, bias, sigma};
///   RooDecay decay_gm{"decay_gm", "decay", dt, tau, gm, RooDecay::SingleSided};
///   RooWorkspace ws{"ws"};
///   ws.import(decay_tm);
///   ws.import(decay_gm);
///   ws.writeToFile("rooAbsAnaConvPdf_classV3.root");
/// \endcode
TEST(RooAbsAnaConvPdf, SchemaEvolutionV3)
{
   std::unique_ptr<TFile> file{TFile::Open("rooAbsAnaConvPdf_classV3.root", "READ")};
   ASSERT_TRUE(file && !file->IsZombie());

   auto *ws = file->Get<RooWorkspace>("ws");
   ASSERT_NE(ws, nullptr);

   RooRealVar &dt = *ws->var("dt");

   struct Reference {
      const char *pdfName;
      const char *modelName;
      double value; // unnormalized value at dt = 1.5
   };

   // Reference values were recorded with the version-3 code that wrote the file.
   for (auto const &ref :
        {Reference{"decay_tm", "tm", 0.37946525232591943}, Reference{"decay_gm", "gm", 0.824172112571901}}) {
      auto *decay = dynamic_cast<RooAbsAnaConvPdf *>(ws->pdf(ref.pdfName));
      ASSERT_NE(decay, nullptr) << ref.pdfName;

      // The original resolution model must not have survived as a spurious
      // (non-value, non-shape) server after the schema evolution.
      for (RooAbsArg *server : decay->servers()) {
         EXPECT_TRUE(server->isValueServer(*decay) || server->isShapeServer(*decay))
            << "Server '" << server->GetName() << "' of '" << ref.pdfName
            << "' is neither a value nor a shape server after schema evolution!";
         EXPECT_STRNE(server->GetName(), ref.modelName)
            << "The original resolution model must not be a server of '" << ref.pdfName << "'!";
      }

      // The model must still be reachable and identical to what was stored.
      EXPECT_STREQ(decay->getModel().GetName(), ref.modelName);

      // The pdf must evaluate to exactly the same value as before.
      dt.setVal(1.5);
      EXPECT_DOUBLE_EQ(decay->getVal(), ref.value) << ref.pdfName;
   }
}
