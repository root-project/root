#include "gtest/gtest.h"
#include "TSystem.h"
#include "TCanvas.h"
#include "TString.h"

TEST(TPad, PDFTitle)
{
    const TString pdfFile = "output.pdf";

    // Generate a multi-page PDF with a title
    TCanvas c;
    c.Print(pdfFile + "(");                  // Start multi-page PDF
    c.Print(pdfFile, "Title:Vertex");        // Add page with title
    c.Print(pdfFile + ")");                  // Close multi-page PDF

    // Check if the file was created successfully
    FileStat_t fileStat;
    int statCode = gSystem->GetPathInfo(pdfFile, fileStat);
    ASSERT_EQ(statCode, 0) << "PDF file was not created.";

    // Get the actual size of the generated file
    Long64_t actualSize = fileStat.fSize;

    // Reference file size in bytes (adjust to match your expected output)
    const Long64_t referenceSize = 13601;
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
