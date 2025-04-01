#include "gtest/gtest.h"

#include "TString.h"
#include "TH1.h"
#include "TSystem.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

struct TestSaveAs {
   static constexpr Int_t Nbins = 5;
   static constexpr Int_t N = Nbins + 2; // 0 and N are the under/overflow bins, resp.
   TString fnam = "H1dump.";

   void SaveHist(const TString &myext, const TString &myOption = "")
   {
      // Bin contents and bin errors
      Double_t binc[N] = {5.2, 0, 10.8, 12.3, 9.5, 7.3, 15.2};
      Double_t bine[N] = {2.1, 0, 2.5, 2.1, 3.5, 2.7, 4.7};

      TH1D h("h", "h_title", Nbins, 0, Nbins);

      for (int i = 0; i < N; ++i) {
         h.SetBinContent(i, binc[i]);
         h.SetBinError(i, bine[i]);
      }

      TString filename{fnam + myext};

      if (myOption.IsNull()) {
         h.SaveAs(filename.Data());
      } else {
         h.SaveAs(filename.Data(), myOption.Data());
      }
   }

   bool IsGood_csv()
   {
      TString filename{fnam + "csv"};
      std::ifstream infile(filename.Data(), std::ios::in);
      if (!infile) {
         return false;
      }
      Int_t idx = 0;
      TString ref[N + 1] = {"# BinLowEdge,BinUpEdge,BinContent,ey",
                            "-1,0,5.2,2.1",
                            "0,1,0,0",
                            "1,2,10.8,2.5",
                            "2,3,12.3,2.1",
                            "3,4,9.5,3.5",
                            "4,5,7.3,2.7",
                            "5,6,15.2,4.7"};
      std::string line;
      while (std::getline(infile, line)) {
         idx++;
         if (idx > N + 1) {
            infile.close();
            return false;
         }
         if (line != ref[idx - 1]) {
            infile.close();
            return false;
         }
      }
      infile.close();
      return true;
   }

   bool IsGood_tsv()
   {
      TString filename{fnam + "tsv"};
      std::ifstream infile(filename.Data(), std::ios::in);
      if (!infile) {
         return false;
      }
      Int_t idx = 0;
      TString ref[N] = {"-1	0	5.2	2.1", "0	1	0	0",      "1	2	10.8	2.5", "2	3	12.3	2.1",
                        "3	4	9.5	3.5",    "4	5	7.3	2.7", "5	6	15.2	4.7"};
      std::string line;
      while (std::getline(infile, line)) {
         idx++;
         if (idx > N) {
            infile.close();
            return false;
         }
         if (line != ref[idx - 1]) {
            infile.close();
            return false;
         }
      }
      infile.close();
      return true;
   }

   bool IsGood_txt()
   {
      TString filename{fnam + "txt"};
      std::ifstream infile(filename.Data(), std::ios::in);
      if (!infile) {
         return false;
      }
      Int_t idx = 0;
      TString ref[N] = {"-1 0 5.2 2.1", "0 1 0 0",     "1 2 10.8 2.5", "2 3 12.3 2.1",
                        "3 4 9.5 3.5",  "4 5 7.3 2.7", "5 6 15.2 4.7"};
      std::string line;
      while (std::getline(infile, line)) {
         idx++;
         if (idx > N) {
            infile.close();
            return false;
         }
         if (line != ref[idx - 1]) {
            infile.close();
            return false;
         }
      }
      infile.close();
      return true;
   }

   bool IsGood_C()
   {
      TString filename{fnam + "C"};
      std::ifstream infile(filename.Data(), std::ios::in);
      if (!infile) {
         return false;
      }
      constexpr Int_t NC = 29; // lines in C file (excl. empty and commented out lines)
      Int_t idx = 0;
      TString ref[NC] = {"{",
                         "   TH1D *h__1 = new TH1D(\"h__1\", \"h_title\", 5, 0, 5);",
                         "   h__1->SetBinContent(0,5.2);",
                         "   h__1->SetBinContent(2,10.8);",
                         "   h__1->SetBinContent(3,12.3);",
                         "   h__1->SetBinContent(4,9.5);",
                         "   h__1->SetBinContent(5,7.3);",
                         "   h__1->SetBinContent(6,15.2);",
                         "   h__1->SetBinError(0,2.1);",
                         "   h__1->SetBinError(2,2.5);",
                         "   h__1->SetBinError(3,2.1);",
                         "   h__1->SetBinError(4,3.5);",
                         "   h__1->SetBinError(5,2.7);",
                         "   h__1->SetBinError(6,4.7);",
                         "   h__1->SetEntries(7);",
                         "   h__1->SetLineColor(TColor::GetColor(\"#000099\"));",
                         "   h__1->GetXaxis()->SetLabelFont(42);",
                         "   h__1->GetXaxis()->SetTitleOffset(1);",
                         "   h__1->GetXaxis()->SetTitleFont(42);",
                         "   h__1->GetYaxis()->SetLabelFont(42);",
                         "   h__1->GetYaxis()->SetTitleFont(42);",
                         "   h__1->GetZaxis()->SetLabelFont(42);",
                         "   h__1->GetZaxis()->SetTitleOffset(1);",
                         "   h__1->GetZaxis()->SetTitleFont(42);",
                         "   h__1->Draw();",
                         "}"};
      std::string line;
      while (std::getline(infile, line)) {
         // skip lines starting with '//' and short lines (empty, indentation spaces only)
         if (line != "{" && line != "}" && (line.rfind("//", 0) == 0 || line.length() < 6)) {
            continue;
         }
         idx++;
         if (idx > NC) {
            infile.close();
            return false;
         }
         if (line != ref[idx - 1]) {
            infile.close();
            return false;
         }
      }
      infile.close();
      return true;
   }
};

/// Tests for TH1::SaveAs
/// In this test we export a TH1 to 4 files of types csv, tsv, txt and C,
/// and then read those files checking whether the contents are as expected
/// In the csv file, we include the header line

TEST(TH1sa, SaveAsCSV)
{
   TestSaveAs t;
   t.SaveHist("csv", "title");
   EXPECT_TRUE(t.IsGood_csv()) << "TH1::SaveAs test: Exported .csv file failed test";
   gSystem->Unlink("H1dump.csv");
}
TEST(TH1sa, SaveAsTSV)
{
   TestSaveAs t;
   t.SaveHist("tsv");
   EXPECT_TRUE(t.IsGood_tsv()) << "TH1::SaveAs test: Exported .tsv file failed test";
   gSystem->Unlink("H1dump.tsv");
}
TEST(TH1sa, SaveAsTXT)
{
   TestSaveAs t;
   t.SaveHist("txt");
   EXPECT_TRUE(t.IsGood_txt()) << "TH1::SaveAs test: Exported .txt file failed test";
   gSystem->Unlink("H1dump.txt");
}
TEST(TH1sa, SaveAsC)
{
   TestSaveAs t;
   t.SaveHist("C");
   EXPECT_TRUE(t.IsGood_C()) << "TH1::SaveAs test: Exported .C file failed test";
   gSystem->Unlink("H1dump.C");
}
