#include "gtest/gtest.h"
#include "TSystem.h"
#include "TCanvas.h"
#include "TLatex.h"
#include "TString.h"
#include <memory>

// Generate a simple canvas with TLatex
std::unique_ptr<TCanvas> CreateCanvas(const char *name, const char *lbl)
{
   auto c = std::make_unique<TCanvas>(name, "Test canvas", 500, 500);
   auto l = new TLatex(0.5, 0.5, lbl);
   l->SetTextAlign(22);
   l->SetTextSize(0.1);
   c->Add(l);
   return c;
}

TEST(TCanvas, SaveHTML)
{
   auto c1 = CreateCanvas("c1", "simple text");

   TString htmlFile = "canvas.html";

   c1->SaveAs(htmlFile);

   // Check if the file was created successfully
   FileStat_t fileStat;
   int statCode = gSystem->GetPathInfo(htmlFile, fileStat);
   ASSERT_EQ(statCode, 0) << "HTML file was not created.";

   // Get the actual size of the generated file
   Long64_t actualSize = fileStat.fSize;

   // Reference file size in bytes
   const Long64_t referenceSize = 17573;
   const double tolerance = 0.02; // Allow 2% deviation, also because of newline on Windows

   // Compute acceptable size range
   Long64_t minSize = referenceSize * (1.0 - tolerance);
   Long64_t maxSize = referenceSize * (1.0 + tolerance);

   // Assert that the actual size is within acceptable range
   EXPECT_GE(actualSize, minSize) << "HTML file is smaller than expected.";
   EXPECT_LE(actualSize, maxSize) << "HTML file is larger than expected.";

   // Cleanup: delete the test file
   gSystem->Unlink(htmlFile);
}


TEST(TCanvas, SaveAllHTML)
{
   auto c1 = CreateCanvas("c1", "simple text 1");
   auto c2 = CreateCanvas("c2", "simple text 2");
   auto c3 = CreateCanvas("c3", "simple text 3");

   TString htmlFile = "canvas_all.html";

   TCanvas::SaveAll({ c1.get(), c2.get(), c3.get()}, htmlFile);

   // Check if the file was created successfully
   FileStat_t fileStat;
   int statCode = gSystem->GetPathInfo(htmlFile, fileStat);
   ASSERT_EQ(statCode, 0) << "HTML file was not created.";

   // Get the actual size of the generated file
   Long64_t actualSize = fileStat.fSize;

   // Reference file size in bytes
   const Long64_t referenceSize = 51265;
   const double tolerance = 0.02; // Allow 2% deviation, also because of newline on Windows

   // Compute acceptable size range
   Long64_t minSize = referenceSize * (1.0 - tolerance);
   Long64_t maxSize = referenceSize * (1.0 + tolerance);

   // Assert that the actual size is within acceptable range
   EXPECT_GE(actualSize, minSize) << "HTML file is smaller than expected.";
   EXPECT_LE(actualSize, maxSize) << "HTML file is larger than expected.";

   // Cleanup: delete the test file
   gSystem->Unlink(htmlFile);
}

TEST(TCanvas, SaveJSON)
{
   auto c1 = CreateCanvas("c1", "simple text");

   TString jsonFile = "canvas.json";

   c1->SaveAs(jsonFile);

   // Check if the file was created successfully
   FileStat_t fileStat;
   int statCode = gSystem->GetPathInfo(jsonFile, fileStat);
   ASSERT_EQ(statCode, 0) << "JSON file was not created.";

   // Get the actual size of the generated file
   Long64_t actualSize = fileStat.fSize;

   // Reference file size in bytes
   const Long64_t referenceSize = 2917;
   const double tolerance = 0.05; // Allow 5% deviation, mainly because of newline on Windows

   // Compute acceptable size range
   Long64_t minSize = referenceSize * (1.0 - tolerance);
   Long64_t maxSize = referenceSize * (1.0 + tolerance);

   // Assert that the actual size is within acceptable range
   EXPECT_GE(actualSize, minSize) << "JSON file is smaller than expected.";
   EXPECT_LE(actualSize, maxSize) << "JSON file is larger than expected.";

   // Cleanup: delete the test file
   gSystem->Unlink(jsonFile);
}

TEST(TCanvas, SaveAllJSON)
{
   auto c1 = CreateCanvas("c1", "simple text");
   auto c2 = CreateCanvas("c2", "simple text 2");
   auto c3 = CreateCanvas("c3", "simple text 3");

   // when store JSON via SaveAll, N times JSON should be created
   TCanvas::SaveAll({c1.get(), c2.get(), c3.get()}, "canvas.json");

   // Check if the files were created successfully
   for (int n = 0; n < 3; ++n) {
      TString jsonFile = TString::Format("canvas%d.json", n);

      FileStat_t fileStat;
      int statCode = gSystem->GetPathInfo(jsonFile, fileStat);
      ASSERT_EQ(statCode, 0) << "JSON file was not created.";

      // Get the actual size of the generated file
      Long64_t actualSize = fileStat.fSize;

      // Reference file size in bytes
      const Long64_t referenceSize = 2917;
      const double tolerance = 0.05; // Allow 5% deviation, mainly because of newline on Windows

      // Compute acceptable size range
      Long64_t minSize = referenceSize * (1.0 - tolerance);
      Long64_t maxSize = referenceSize * (1.0 + tolerance);

      // Assert that the actual size is within acceptable range
      EXPECT_GE(actualSize, minSize) << "JSON file is smaller than expected.";
      EXPECT_LE(actualSize, maxSize) << "JSON file is larger than expected.";

      // Cleanup: delete the test file
      gSystem->Unlink(jsonFile);
   }
}

