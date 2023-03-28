#include <stdlib.h>
#include "gtest/gtest.h"

#include "ROOT/RDataFrame.hxx"
#include "RHnCUDA.h"
#include "TH1.h"

auto numRows = 10;
ROOT::RDataFrame rdf(numRows);
char env[] = "CUDA_HIST";

/**
 * Helper functions for toggling ON/OFF CUDA histogramming.
 */
void EnableCUDA()
{
   setenv(env, "1", 1);
}

void DisableCUDA()
{
   unsetenv(env);
}

/**
 * Helper functions for element-wise comparison of histogram arrays and bin edges.
 */

// Element-wise comparison between two arrays.
template <typename AType = Double_t *>
void CheckArrays(AType a, AType b, Int_t m, Int_t n)
{
   EXPECT_EQ(m, n);
   for (auto i : ROOT::TSeqI(n)) {
      EXPECT_EQ(a[i], b[i]) << "  i = " << i;
   }
}

// Comparison with labelled messages on failure.
template <typename AType = Double_t *>
void CheckArrays(AType a, AType b, Int_t m, Int_t n, const char *labelA, const char *labelB)
{
   EXPECT_EQ(m, n);
   for (auto i : ROOT::TSeqI(n)) {
      EXPECT_EQ(a[i], b[i]) << "  a = " << labelA << " b = " << labelB << " i = " << i;
   }
}

// Comparison with labelled messages per element on failure.
template <typename AType = Double_t *>
void CheckArrays(AType a, AType b, Int_t m, Int_t n, const char **labelA, const char **labelB)
{
   EXPECT_EQ(m, n);
   for (auto i : ROOT::TSeqI(n)) {
      EXPECT_EQ(a[i], b[i]) << "  a = " << labelA[i] << " b = " << labelB[i] << "i = " << i;
   }
}

void CheckBins(const TAxis *axis1, const TAxis *axis2)
{
   auto nBins1 = axis1->GetXbins()->GetSize();
   auto nBins2 = axis2->GetXbins()->GetSize();

   CheckArrays(axis1->GetXbins()->GetArray(), axis2->GetXbins()->GetArray(), nBins1, nBins2,
               "binEdges", "CUDA binEdges");
}

// Test filling 1-dimensional histogram with doubles
TEST(Hist1DTest, FillFixedbins)
{
   Double_t x = 0;
   auto d = rdf.Define("x", [&x]() { return x++; }).Define("w", [&x]() { return x + 1.; });

   DisableCUDA();
   auto h1 = d.Histo1D(::TH1D("h1", "h1", numRows, 0, 100), "x");
   auto h1Axis = h1->GetXaxis();
   auto h1Ncells = h1->GetNcells();
   auto h1Array = h1->GetArray();
   Double_t h1Stats[4];
   h1->GetStats(h1Stats);

   EnableCUDA();
   x = 0;
   auto h2 = d.Histo1D(::TH1D("h2", "h2", numRows, 0, 100), "x");
   auto h2Axis = h2->GetXaxis();
   auto h2Ncells = h2->GetNcells();
   auto h2Array = h2->GetArray();
   Double_t h2Stats[4];
   h2->GetStats(h2Stats);

   CheckBins(h1Axis, h2Axis);                                        // Compare bin edges.
   CheckArrays(h1Array, h2Array, h1Ncells, h2Ncells, "fArray", "CUDA fArray"); // Compare bin values.
   CheckArrays(h1Stats, h2Stats, 4, 4, new const char*[4]{"fTsumw", "fTsumw2", "fTsumwx", "fTsumwx2"},
               new const char*[4]{"CUDA fTsumw", "CUDA fTsumw2", "CUDA fTsumwx",
                                 "CUDA fTsumwx2"}); // Compare histogram statistics.
}


TEST(Hist1DTest, FillVarbins)
{
   Double_t x = 0;
   auto d = rdf.Define("x", [&x]() { return x++; }).Define("w", [&x]() { return x + 1.; });
   std::vector<double> edges{1, 2, 3, 4, 5, 6, 10};

   DisableCUDA();
   auto h1 = d.Histo1D(::TH1D("h1", "h1", (int)edges.size() - 1, edges.data()), "x");
   auto h1Axis = h1->GetXaxis();
   auto h1Ncells = h1->GetNcells();
   auto h1Array = h1->GetArray();
   Double_t h1Stats[4];
   h1->GetStats(h1Stats);

   EnableCUDA();
   x = 0;
   auto h2 = d.Histo1D(::TH1D("h2", "h2", (int)edges.size() - 1, edges.data()), "x");
   auto h2Axis = h2->GetXaxis();
   auto h2Ncells = h2->GetNcells();
   auto h2Array = h2->GetArray();
   Double_t h2Stats[4];
   h2->GetStats(h2Stats);

   CheckBins(h1Axis, h2Axis);                                        // Compare bin edges.
   CheckArrays(h1Array, h2Array, h1Ncells, h2Ncells, "fArray", "CUDA_fArray"); // Compare bin values.
   CheckArrays(h1Stats, h2Stats, 4, 4, new const char*[4]{"fTsumw", "fTsumw2", "fTsumwx", "fTsumwx2"},
               new const char*[4]{"CUDA_fTsumw", "CUDA_fTsumw2", "CUDA_fTsumwx",
                                 "CUDA_fTsumwx2"}); // Compare histogram statistics.
}

// Test filling 1-dimensional histogram with doubles
TEST(Hist1DTest, FillFixedbins2D)
{
   Double_t x = 0;
   auto d = rdf.Define("x", [&x]() { return x++; }).Define("y", [&x]() { return x + 1.; });

   DisableCUDA();
   auto h1 = d.Histo2D(::TH2D("h1", "h1", numRows, 0, 100, numRows, 200, 300), "x", "y");
   auto h1XAxis = h1->GetXaxis();
   auto h1YAxis = h1->GetYaxis();
   auto h1Ncells = h1->GetNcells();
   auto h1Array = h1->GetArray();
   Double_t h1Stats[7];
   h1->GetStats(h1Stats);

   EnableCUDA();
   x = 0;
   auto h2 = d.Histo2D(::TH2D("h2", "h2", numRows, 0, 100, numRows, 200, 300), "x", "y");
   auto h2XAxis = h2->GetXaxis();
   auto h2YAxis = h2->GetYaxis();
   auto h2Ncells = h2->GetNcells();
   auto h2Array = h2->GetArray();
   Double_t h2Stats[7];
   h2->GetStats(h2Stats);

   CheckBins(h1XAxis, h2XAxis);                                        // Compare bin edges.
   CheckBins(h1YAxis, h2YAxis);                                        // Compare bin edges.
   CheckArrays(h1Array, h2Array, h1Ncells, h2Ncells, "fArray", "CUDA fArray"); // Compare bin values.
   CheckArrays(h1Stats, h2Stats, 4, 4, new const char*[4]{"fTsumw", "fTsumw2", "fTsumwx", "fTsumwx2"},
               new const char*[4]{"CUDA fTsumw", "CUDA fTsumw2", "CUDA fTsumwx",
                                 "CUDA fTsumwx2"}); // Compare histogram statistics.
}
