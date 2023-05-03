////////////////////////////////////////////////////////////////////////////////////
/// Compares results of RHnCUDA and TH* with Histo*D
/// Note that these histograms are only of type double
///

#include <stdlib.h>
#include "gtest/gtest.h"

#include "ROOT/RDataFrame.hxx"
// #include "RHnCUDA.h"
#include "TH1.h"
#include "TAxis.h"

// TODO: put these in some kind of gtest environment class?
auto numRows = 42;
auto numBins = numRows - 2; // -2 to also test filling u/overflow.
auto startBin = 0;
auto startFill = startBin - 1;
auto endBin = numBins;

ROOT::RDataFrame rdf1D(numRows);
ROOT::RDataFrame rdf2D(numRows *numRows);
ROOT::RDataFrame rdf3D(numRows *numRows *numRows);
char env[] = "CUDA_HIST";

template <typename T = double, typename HIST = TH1D>
struct HistProperties {
   T *array;
   int dim, nCells;
   double *stats;
   int nStats;

   HistProperties(ROOT::RDF::RResultPtr<HIST> &h)
   {
      dim = h->GetDimension();
      nStats = 2 + 2 * dim;
      if (dim > 1)
         nStats += TMath::Binomial(dim, 2);
      stats = (double *)malloc((nStats) * sizeof(double));

      array = h->GetArray();
      nCells = h->GetNcells();
      h->GetStats(stats);
   }
};

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
 * Helper functions for element-wise comparison of histogram arrays and bin edges->
 */

// Element-wise comparison between two arrays.
#define CHECK_ARRAY(a, b, n, m)                                  \
   {                                                             \
      EXPECT_EQ(n, n);                                           \
      for (auto i : ROOT::TSeqI(n)) {                            \
         EXPECT_DOUBLE_EQ(a[i], b[i]) << "  at index i = " << i; \
      }                                                          \
   }

template <typename T = double, typename HIST = TH1D>
void CompareHistograms(const HistProperties<T, HIST> &TH, const HistProperties<T, HIST> &CUDA)
{
   CHECK_ARRAY(TH.array, CUDA.array, TH.nCells, CUDA.nCells); // Compare bin values.
   CHECK_ARRAY(TH.stats, CUDA.stats, TH.nStats, CUDA.nStats); // Compare histogram statistics
}

std::vector<double> *GetVariableBinEdges()
{
   int e = startBin;
   auto edges = new std::vector<double>(numBins + 1);
   std::generate(edges->begin(), edges->end(), [&]() { return e++; });
   (*edges)[numBins] += 10;

   return edges;
}

// Test filling 1-dimensional histogram with doubles
TEST(HistoTest, Fill1D)
{
   double x = startFill;
   auto d = rdf1D.Define("x", [&x]() { return x++; }).Define("w", [&x]() { return x; });

   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   {
      SCOPED_TRACE("Fill 1D histograms with fixed bins");

      DisableCUDA();
      auto h1ptr = d.Histo1D(::TH1D("h1", "h1", numBins, startBin, endBin), "x");
      auto h1 = HistProperties<double>(h1ptr);

      EnableCUDA();
      x = startFill; // need to reset x, because the second Histo1D redefines "x" again.
      auto h2ptr = d.Histo1D(::TH1D("h2", "h2", numBins, startBin, endBin), "x");
      auto h2 = HistProperties<double>(h2ptr);

      CompareHistograms(h1, h2);
   }

   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   {
      SCOPED_TRACE("Fill 1D histograms with weighted fixed bins");

      DisableCUDA();
      x = startFill;
      auto h1ptr = d.Histo1D(::TH1D("h1", "h1", numBins, startBin, endBin), "x", "w");
      auto h1 = HistProperties<double>(h1ptr);

      EnableCUDA();
      x = startFill; // need to reset x, because the second Histo1D redefines "x" again.
      auto h2ptr = d.Histo1D(::TH1D("h2", "h2", numBins, startBin, endBin), "x", "w");
      auto h2 = HistProperties<double>(h2ptr);

      CompareHistograms(h1, h2);
   }

   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   auto edges = GetVariableBinEdges();

   {
      SCOPED_TRACE("Fill 1D histograms with variable bins");

      DisableCUDA();
      x = startFill;
      auto h1ptr = d.Histo1D(::TH1D("h1", "h1", numBins, edges->data()), "x");
      auto h1 = HistProperties<double>(h1ptr);

      EnableCUDA();
      x = startFill; // need to reset x, because the second Histo1D redefines "x" again.
      auto h2ptr = d.Histo1D(::TH1D("h2", "h2", numBins, edges->data()), "x");
      auto h2 = HistProperties<double>(h2ptr);

      CompareHistograms<double>(h1, h2);
   }

   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   {
      SCOPED_TRACE("Fill 1D histograms with weighted variable bins");

      DisableCUDA();
      x = startFill;
      auto h1ptr = d.Histo1D(::TH1D("h1", "h1", numBins, edges->data()), "x", "w");
      auto h1 = HistProperties<double>(h1ptr);

      EnableCUDA();
      x = startFill; // need to reset x, because the second Histo1D redefines "x" again.
      auto h2ptr = d.Histo1D(::TH1D("h2", "h2", numBins, edges->data()), "x", "w");
      auto h2 = HistProperties<double>(h2ptr);

      CompareHistograms<double>(h1, h2);
   }
}

// Test filling 2-dimensional histogram with doubles
TEST(HistoTest, Fill2D)
{
   double x = startFill, y = startFill;
   int r1 = 0, r2 = 0;

   // fill every cell in the histogram once, including u/overflow.
   auto fillX = [&x, &r1]() { return ++r1 % numRows == 0 ? x++ : x; };
   auto fillY = [&y, &r2]() {
      if (r2++ % numRows == 0)
         y = startFill;
      return y++;
   };

   auto d = rdf2D.Define("x", fillX).Define("y", fillY).Define("w", [&x, &y]() { return x + y; });

   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   {
      SCOPED_TRACE("Fill 2D histograms with fixed bins");

      DisableCUDA();
      auto h1ptr = d.Histo2D(::TH2D("h1", "h1", numBins, startBin, endBin, numBins, startBin, endBin), "x", "y");
      auto h1 = HistProperties<double, TH2D>(h1ptr);

      EnableCUDA();
      x = startFill, y = startFill, r1 = 0, r2 = 0; // need to reset, because the second Histo1D redefines "x" again.
      auto h2ptr = d.Histo2D(::TH2D("h2", "h2", numBins, startBin, endBin, numBins, startBin, endBin), "x", "y");
      auto h2 = HistProperties<double, TH2D>(h2ptr);

      CompareHistograms<double, TH2D>(h1, h2);
   }

   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   {
      SCOPED_TRACE("Fill 2D histograms with weighted fixed bins");

      DisableCUDA();
      x = startFill, y = startFill, r1 = 0, r2 = 0;
      auto h1ptr = d.Histo2D(::TH2D("h1", "h1", numBins, startBin, endBin, numBins, startBin, endBin), "x", "y", "w");
      auto h1 = HistProperties<double, TH2D>(h1ptr);

      EnableCUDA();
      x = startFill, y = startFill, r1 = 0, r2 = 0;
      auto h2ptr = d.Histo2D(::TH2D("h2", "h2", numBins, startBin, endBin, numBins, startBin, endBin), "x", "y", "w");
      auto h2 = HistProperties<double, TH2D>(h2ptr);

      CompareHistograms<double, TH2D>(h1, h2);
   }

   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   auto edges = GetVariableBinEdges();

   {
      SCOPED_TRACE("Fill 2D histograms with variable bins");

      DisableCUDA();
      x = startFill, y = startFill, r1 = 0, r2 = 0;
      auto h1ptr = d.Histo2D(::TH2D("h1", "h1", numBins, edges->data(), numBins, edges->data()), "x", "y");
      auto h1 = HistProperties<double, TH2D>(h1ptr);

      EnableCUDA();
      x = startFill, y = startFill, r1 = 0, r2 = 0;
      auto h2ptr = d.Histo2D(::TH2D("h2", "h2", numBins, edges->data(), numBins, edges->data()), "x", "y");
      auto h2 = HistProperties<double, TH2D>(h2ptr);

      CompareHistograms<double, TH2D>(h1, h2);
   }

   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

   {
      SCOPED_TRACE("Fill 2D histograms with weighted variable bins");

      DisableCUDA();
      x = startFill, y = startFill, r1 = 0, r2 = 0;
      auto h1ptr = d.Histo2D(::TH2D("h1", "h1", numBins, edges->data(), numBins, edges->data()), "x", "y", "w");
      auto h1 = HistProperties<double, TH2D>(h1ptr);

      EnableCUDA();
      x = startFill, y = startFill, r1 = 0, r2 = 0;
      auto h2ptr = d.Histo2D(::TH2D("h2", "h2", numBins, edges->data(), numBins, edges->data()), "x", "y", "w");
      auto h2 = HistProperties<double, TH2D>(h2ptr);

      CompareHistograms<double, TH2D>(h1, h2);
   }

   delete edges;
}

// Test filling 3-dimensional histogram with doubles
TEST(HistoTest, Fill3D)
{
   double x = startFill, y = startFill, z = startFill;
   int r1 = 0, r2 = 0, r3 = 0;

   // fill every cell in the histogram once, including u/overflow.
   auto fillX = [&x, &r1]() {
      if (++r1 % (numRows * numRows) == 0)
         return x++;
      return x;
   };
   auto fillY = [&y, &r2]() {
      if (r2 % (numRows * numRows) == 0)
         y = startFill;
      return ++r2 % numRows == 0 ? y++ : y;
   };

   auto fillZ = [&z, &r3]() {
      if (r3++ % numRows == 0)
         z = startFill;
      return z++;
   };

   auto d =
      rdf3D.Define("x", fillX).Define("y", fillY).Define("z", fillZ).Define("w", [&x, &y, &z]() { return x + y + z; });

   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   {
      SCOPED_TRACE("Fill 3D histograms with fixed bins");

      DisableCUDA();
      auto h1ptr =
         d.Histo3D(::TH3D("h1", "h1", numBins, startBin, endBin, numBins, startBin, endBin, numBins, startBin, endBin),
                   "x", "y", "z");
      auto h1 = HistProperties<double, TH3D>(h1ptr);

      EnableCUDA();
      x = startFill, y = startFill, z = startFill, r1 = 0, r2 = 0, r3 = 0;
      auto h2ptr =
         d.Histo3D(::TH3D("h2", "h2", numBins, startBin, endBin, numBins, startBin, endBin, numBins, startBin, endBin),
                   "x", "y", "z");
      auto h2 = HistProperties<double, TH3D>(h2ptr);

      CompareHistograms<double, TH3D>(h1, h2);
   }

   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   {
      SCOPED_TRACE("Fill 3D histograms with weighted fixed bins");

      DisableCUDA();
      x = startFill, y = startFill, z = startFill, r1 = 0, r2 = 0, r3 = 0;
      auto h1ptr =
         d.Histo3D(::TH3D("h1", "h1", numBins, startBin, endBin, numBins, startBin, endBin, numBins, startBin, endBin),
                   "x", "y", "z", "w");
      auto h1 = HistProperties<double, TH3D>(h1ptr);

      EnableCUDA();
      x = startFill, y = startFill, z = startFill, r1 = 0, r2 = 0, r3 = 0;
      auto h2ptr =
         d.Histo3D(::TH3D("h2", "h2", numBins, startBin, endBin, numBins, startBin, endBin, numBins, startBin, endBin),
                   "x", "y", "z", "w");
      auto h2 = HistProperties<double, TH3D>(h2ptr);

      CompareHistograms<double, TH3D>(h1, h2);
   }

   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   auto edges = GetVariableBinEdges();

   {
      SCOPED_TRACE("Fill 3D histograms with variable bins");

      DisableCUDA();
      x = startFill, y = startFill, z = startFill, r1 = 0, r2 = 0, r3 = 0;
      auto h1ptr = d.Histo3D(::TH3D("h1", "h1", numBins, edges->data(), numBins, edges->data(), numBins, edges->data()),
                             "x", "y", "z");
      auto h1 = HistProperties<double, TH3D>(h1ptr);

      EnableCUDA();
      x = startFill, y = startFill, z = startFill, r1 = 0, r2 = 0, r3 = 0;
      auto h2ptr = d.Histo3D(::TH3D("h2", "h2", numBins, edges->data(), numBins, edges->data(), numBins, edges->data()),
                             "x", "y", "z");
      auto h2 = HistProperties<double, TH3D>(h2ptr);

      CompareHistograms<double, TH3D>(h1, h2);
   }

   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

   {
      SCOPED_TRACE("Fill 3D histograms with weighted variable bins");

      DisableCUDA();
      x = startFill, y = startFill, z = startFill, r1 = 0, r2 = 0, r3 = 0;
      auto h1ptr = d.Histo3D(::TH3D("h1", "h1", numBins, edges->data(), numBins, edges->data(), numBins, edges->data()),
                             "x", "y", "z", "w");
      auto h1 = HistProperties<double, TH3D>(h1ptr);

      EnableCUDA();
      x = startFill, y = startFill, z = startFill, r1 = 0, r2 = 0, r3 = 0;
      auto h2ptr = d.Histo3D(::TH3D("h2", "h2", numBins, edges->data(), numBins, edges->data(), numBins, edges->data()),
                             "x", "y", "z", "w");
      auto h2 = HistProperties<double, TH3D>(h2ptr);

      CompareHistograms<double, TH3D>(h1, h2);
   }

   delete edges;
}
