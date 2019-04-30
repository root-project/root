/// \file
/// \ingroup tutorial_forest
/// \notebook
/// Write and read STL vectors with RForest.  Adapted from the hvector tree tutorial.
/// For reading, the tutorial shows the use of a Forest View, which selectively accesses specific fields.
/// If a view is used for reading, there is no need to define the data model as an RForestModel first.
///
/// \macro_image
/// \macro_code
///
/// \date April 2019
/// \author The ROOT Team

// NOTE: The RForest classes are experimental at this point.
// Functionality, interface, and data format is still subject to changes.
// Do not use for real data!

R__LOAD_LIBRARY(libROOTForest)

#include <ROOT/RForest.hxx>
#include <ROOT/RForestModel.hxx>

#include <TCanvas.h>
#include <TH1F.h>
#include <TRandom.h>
#include <TSystem.h>

#include <cstdio>
#include <iostream>
#include <memory>
#include <vector>
#include <utility>

// Import classes from experimental namespace for the time being
using RForestModel = ROOT::Experimental::RForestModel;
using RInputForest = ROOT::Experimental::RInputForest;
using ROutputForest = ROOT::Experimental::ROutputForest;

// Where to store the forest of this example
constexpr char const* kForestFileName = "fst002_vector.root";

// Update the histogram GUI every so many fills
constexpr int kUpdateGuiFreq = 1000;

// Number of events to generate
constexpr int kNEvents = 25000;

// Generate kNEvents with vectors in kForestFileName
void Write()
{
   // We create a unique pointer to an empty data model
   auto model = RForestModel::Create();

   // Creating fields of std::vector is the same as creating fields of simple types.  As a result, we get
   // shared pointers of the given type
   //std::shared_ptr<std::vector<float>> fldVpx
   auto fldVpx = model->MakeField<std::vector<float>>("vpx");
   auto fldVpy = model->MakeField<std::vector<float>>("vpy");
   auto fldVpz = model->MakeField<std::vector<float>>("vpz");
   auto fldVrand = model->MakeField<std::vector<float>>("vrand");

   // We hand-over the data model to a newly created forest of name "F", stored in kForestFileName
   // In return, we get a unique pointer to a forest that we can fill
   auto forest = ROutputForest::Recreate(std::move(model), "F", kForestFileName);

   TH1F *hpx = new TH1F("hpx", "This is the px distribution", 100, -4, 4);
   hpx->SetFillColor(48);

   TCanvas *c1 = new TCanvas("c1", "Dynamic Filling Example", 200, 10, 700, 500);

   gRandom->SetSeed();
   for (int i = 0; i < kNEvents; i++) {
      int npx = static_cast<int>(gRandom->Rndm(1) * 15);

      fldVpx->clear();
      fldVpy->clear();
      fldVpz->clear();
      fldVrand->clear();

      // Set the field data for the current event
      for (int j = 0; j < npx; ++j) {
         float px, py, pz;
         gRandom->Rannor(px, py);
         pz = px*px + py*py;
         float random = gRandom->Rndm(1);

         hpx->Fill(px);

         fldVpx->emplace_back(px);
         fldVpy->emplace_back(py);
         fldVpz->emplace_back(pz);
         fldVrand->emplace_back(random);
      }

      // Gui updates
      if (i && (i % kUpdateGuiFreq) == 0) {
         if (i == kUpdateGuiFreq) hpx->Draw();
         c1->Modified();
         c1->Update();
         if (gSystem->ProcessEvents())
            break;
      }

      forest->Fill();
   }

   hpx->DrawCopy();

   // The forest unique pointer goes out of scope here.  On destruction, the forest flushes unwritten data to disk
   // and closes the attached ROOT file.
}


// For all of the events, histogram only one of the written vectors
void Read()
{
   // Create a forest without imposing a specific data model.  We could generate the data model from the forest
   // but here we prefer the view because we only want to access a single field
   auto forest = RInputForest::Open("F", kForestFileName);

   // Quick overview of the forest's key meta-data
   std::cout << forest->GetInfo();
   // In a future version of RForest, there will be support for forest->Show() and forest->Scan()

   // Generate a handle to a specific field.  If the field type does not match the field in the forest, a runtime
   // error is thrown.
   auto viewVpx = forest->GetView<std::vector<float>>("vpx");

   // --> Move to sub models

   TCanvas *c2 = new TCanvas("c2", "Dynamic Filling Example", 200, 10, 700, 500);
   TH1F *h = new TH1F("h", "This is the px distribution", 100, -4, 4);
   h->SetFillColor(48);

   // Iterate through all the events using i as event number and as an index for accessing the view
   for (auto i : forest->GetViewRange()) {
      // We want to lookup and load the vector px of event i only once, hence we keep the result in the
      // const reference vpx
      auto vpx = viewVpx(i);

      for (unsigned int j = 0; j < vpx.size(); ++j) {
         h->Fill(vpx.at(j));
      }

      if (i && (i % kUpdateGuiFreq) == 0) {
         if (i == kUpdateGuiFreq) h->Draw();
         c2->Modified();
         c2->Update();
         if (gSystem->ProcessEvents())
            break;
      }
   }

   // Prevent the histogram from disappearing
   h->DrawCopy();
}


void fst002_vector()
{
   Write();
   Read();
}
