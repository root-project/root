// Test the C++ validity of saved objects' names

#include "gtest/gtest.h"

#include "TROOT.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TH2Poly.h"
#include "TProfile.h"
#include "TSystem.h"

TEST(TH1, MapCppNameTest)
{
   gROOT->SetBatch();
   auto C = new TCanvas("C", "C", 700, 500);
   TString n1 = "<[ !@$%&^*().,[]/+=-&:|'~]>";
   TString n2 = "<[!@$%&^*().,[]/+=-&:|'~ ]>";
   TString n3 = "<[! @$%&^*().,[]/+=-&:|'~]>";
   TString n4 = "<[!@$%&^*()., []/+=-&:|'~]>";

   auto g1 = new TGraph();
   g1->SetName(n1.Data());
   for (int i=0; i < 10; i++)
      g1->AddPoint(i * 0.123, 10 * sin(i * 0.123 + 0.2));

   auto h1 = new TH1F(n1.Data(), n1.Data(), 10 , 0, 1);
   auto h2 = new TH2F(n2.Data(), n2.Data(), 10 , 0, 1, 10, 0, 1);
   auto h3 = new TH2Poly(n3.Data(), n3.Data(), 10 , 0, 1, 10, 0, 1);
   h3->AddBin(0., 0., 1., 1.);
   auto h4 = new TProfile(n4.Data(), n4.Data(), 10, 0, 1);

   g1->Draw();
   h1->Draw("same");
   h2->Draw("same");
   h3->Draw("same");
   h4->Draw("same");

   TString CFile = "C_Output.C";
   C->SaveAs(CFile.Data());

   int FileSize = 0;
   FileStat_t fs;
   if (!gSystem->GetPathInfo(CFile.Data(), fs))
      FileSize = (Int_t)fs.fSize;

   EXPECT_NEAR(FileSize, 5867, 200);

   gSystem->Unlink(CFile.Data());
}
