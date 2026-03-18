#include "gtest/gtest.h"
#include "TSystem.h"
#include "TCanvas.h"
#include "TString.h"
#include "TLatex.h"

TEST(TPad, PDFUrl)
{
   const TString pdfFile = "output.pdf";

   // Generate a multi-page PDF with page titles and #url in TLatex
   TCanvas c1;
   TCanvas c2;

   TLatex l1(.1, .4, "Link on #color[4]{#url[https://root.cern]{root.cern}} web site");
   TLatex l2(.1, .5, "Link on #color[2]{#url[https://cern.ch]{CERN}} web site");

   c1.cd();
   l1.Draw();
   l2.Draw();

   c2.cd();
   l2.Draw();

   c1.Print(pdfFile + "["); // Start multi-page PDF
   c2.Print(pdfFile, "Title:Page 1");
   c1.Print(pdfFile, "Title:Page 2");
   c1.Print(pdfFile + "]"); // Close multi-page PDF

   // Check if the file was created successfully
   FileStat_t fileStat;
   int statCode = gSystem->GetPathInfo(pdfFile, fileStat);
   ASSERT_EQ(statCode, 0) << "PDF file was not created.";

   // Get the actual size of the generated file
   Long64_t actualSize = fileStat.fSize;

   // Reference file size in bytes (adjust to match your expected output)
   const Long64_t referenceSize = 13927;
   const double tolerance = 0.01; // Allow 1% deviation

   // Compute acceptable size range
   Long64_t minSize = referenceSize * (1.0 - tolerance);
   Long64_t maxSize = referenceSize * (1.0 + tolerance);

   // Assert that the actual size is within acceptable range
   EXPECT_GE(actualSize, minSize) << "PDF file is smaller than expected.";
   EXPECT_LE(actualSize, maxSize) << "PDF file is larger than expected.";

   // Cleanup: delete the test file
   gSystem->Unlink(pdfFile);
}
