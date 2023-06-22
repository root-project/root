////////////////////////////////////////////////////////////////////////////////////
/// Compares results of RHnCUDA and TH* with Histo*D
/// Note that these histograms are only of type double
///

#include <stdlib.h>
#include "gtest/gtest.h"

#include "ROOT/RDataFrame.hxx"
#include "TH1.h"
#include "TAxis.h"

std::vector<const char *> test_environments = {"CUDA_HIST"};

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
      nCells = h->GetNcells();

      // Create a copy in case the array gets cleaned up by RDataframe before checking the results
      array = (T *)malloc(nCells * sizeof(T));
      auto histogram = h->GetArray();
      std::copy(histogram, histogram + nCells, array);

      stats = (double *)calloc(nStats, sizeof(double));
      h->GetStats(stats);
   }

   ~HistProperties()
   {
      free(array);
      free(stats);
   }
};

class HistoTestFixture1D : public testing::TestWithParam<const char *> {
protected:
   int numRows, numBins; // -2 to also test filling u/overflow.
   double startBin, startFill, endBin;

   const char *env;

   HistoTestFixture1D()
   {
      numRows = 42;
      numBins = numRows - 2; // -2 to also test filling u/overflow.
      startBin = 0;
      startFill = startBin - 1;
      endBin = numBins;
      env = GetParam();
   }

   std::vector<double> *GetVariableBinEdges()
   {
      int e = startBin;
      auto edges = new std::vector<double>(numBins + 1);
      std::generate(edges->begin(), edges->end(), [&]() { return e++; });
      (*edges)[numBins] += 10;

      return edges;
   }

   template <typename Hist, typename... Cols>
   auto GetHisto1D(Hist histMdl, Cols... cols)
   {
      double x = startFill;
      auto df = ROOT::RDataFrame(numRows).Define("x", [&]() { return x++; }).Define("w", [&]() { return x; });
      auto hptr = df.Histo1D(histMdl, cols...);
      auto h = HistProperties<double>(hptr);
      return h;
   }

   /**
    * Helper functions for toggling ON/OFF GPU histogramming.
    */
   void EnableGPU()
   {
      DisableGPU();
      setenv(env, "1", 1);
   }

   void DisableGPU()
   {
      for (unsigned int i = 0; i < test_environments.size(); i++)
         unsetenv(test_environments[i]);
   }
};

class HistoTestFixture2D : public HistoTestFixture1D {
protected:
   HistoTestFixture2D() : HistoTestFixture1D() {}

   template <typename Hist, typename... Cols>
   auto GetHisto2D(Hist histMdl, Cols... cols)
   {
      double x = startFill, y = startFill;
      int r1 = 0, r2 = 0;

      // fill every cell in the histogram once, including u/overflow.
      auto fillX = [&]() { return ++r1 % numRows == 0 ? x++ : x; };
      auto fillY = [&]() {
         if (r2++ % numRows == 0)
            y = startFill;
         return y++;
      };

      auto df =
         ROOT::RDataFrame(numRows * numRows).Define("x", fillX).Define("y", fillY).Define("w", [&]() { return x + y; });
      auto hptr = df.Histo2D(histMdl, cols...);
      auto h = HistProperties<double, TH2D>(hptr);
      return h;
   }
};

class HistoTestFixture3D : public HistoTestFixture1D {
protected:
   HistoTestFixture3D() : HistoTestFixture1D() {}

   template <typename Hist, typename... Cols>
   auto GetHisto3D(Hist histMdl, Cols... cols)
   {
      double x = startFill, y = startFill, z = startFill;
      int r1 = 0, r2 = 0, r3 = 0;

      // fill every cell in the histogram once, including u/overflow.
      auto fillX = [&]() {
         if (++r1 % (numRows * numRows) == 0)
            return x++;
         return x;
      };
      auto fillY = [&]() {
         if (r2 % (numRows * numRows) == 0)
            y = startFill;
         return ++r2 % numRows == 0 ? y++ : y;
      };

      auto fillZ = [&]() {
         if (r3++ % numRows == 0)
            z = startFill;
         return z++;
      };

      auto df = ROOT::RDataFrame(numRows * numRows * numRows)
                   .Define("x", fillX)
                   .Define("y", fillY)
                   .Define("z", fillZ)
                   .Define("w", [&]() { return x + y + z; });
      auto hptr = df.Histo3D(histMdl, cols...);
      auto h = HistProperties<double, TH3D>(hptr);
      return h;
   }
};

INSTANTIATE_TEST_SUITE_P(HistoTest1D, HistoTestFixture1D, testing::ValuesIn(test_environments));
INSTANTIATE_TEST_SUITE_P(HistoTest2D, HistoTestFixture2D, testing::ValuesIn(test_environments));
INSTANTIATE_TEST_SUITE_P(HistoTest3D, HistoTestFixture3D, testing::ValuesIn(test_environments));

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
void CompareHistograms(const HistProperties<T, HIST> &TH, const HistProperties<T, HIST> &GPU)
{
   CHECK_ARRAY(TH.array, GPU.array, TH.nCells, GPU.nCells); // Compare bin values.
   CHECK_ARRAY(TH.stats, GPU.stats, TH.nStats, GPU.nStats); // Compare histogram statistics
}

/***
 * Test 1D Histograms
 */

TEST_P(HistoTestFixture1D, Fill1DFixedBins)
{
   auto mdl = ::TH1D("h", "h", this->numBins, this->startBin, this->endBin);

   DisableGPU();
   auto h1 = GetHisto1D(mdl, "x");

   EnableGPU();
   auto h2 = GetHisto1D(mdl, "x");

   CompareHistograms(h1, h2);
}

TEST_P(HistoTestFixture1D, Fill1DWeightedFixedBins)
{
   auto mdl = ::TH1D("h", "h", this->numBins, this->startBin, this->endBin);

   DisableGPU();
   auto h1 = GetHisto1D(mdl, "x", "w");

   EnableGPU();
   auto h2 = GetHisto1D(mdl, "x", "w");

   CompareHistograms(h1, h2);
}

TEST_P(HistoTestFixture1D, Fill1DVariableBins)
{
   auto edges = GetVariableBinEdges();
   auto mdl = ::TH1D("h", "h", this->numBins, edges->data());

   DisableGPU();
   auto h1 = GetHisto1D(mdl, "x");

   EnableGPU();
   auto h2 = GetHisto1D(mdl, "x");

   CompareHistograms<double>(h1, h2);
   delete edges;
}

TEST_P(HistoTestFixture1D, Fill1DWeightedVariableBins)
{
   auto edges = GetVariableBinEdges();
   auto mdl = ::TH1D("h", "h", this->numBins, edges->data());

   DisableGPU();
   auto h1 = GetHisto1D(mdl, "x", "w");

   EnableGPU();
   auto h2 = GetHisto1D(mdl, "x", "w");

   CompareHistograms<double>(h1, h2);
   delete edges;
}

/***
 * Test 2D Histograms
 */

TEST_P(HistoTestFixture2D, Fill2DFixedBins)
{
   auto mdl =
      ::TH2D("h", "h", this->numBins, this->startBin, this->endBin, this->numBins, this->startBin, this->endBin);

   DisableGPU();
   auto h1 = GetHisto2D(mdl, "x", "y");

   EnableGPU();
   auto h2 = GetHisto2D(mdl, "x", "y");

   CompareHistograms<double, TH2D>(h1, h2);
}

TEST_P(HistoTestFixture2D, Fill2DWeightedFixedBins)
{
   auto mdl =
      ::TH2D("h", "h", this->numBins, this->startBin, this->endBin, this->numBins, this->startBin, this->endBin);

   DisableGPU();
   auto h1 = GetHisto2D(mdl, "x", "y", "w");

   EnableGPU();
   auto h2 = GetHisto2D(mdl, "x", "y", "w");

   CompareHistograms<double, TH2D>(h1, h2);
}

TEST_P(HistoTestFixture2D, Fill2DVariableBins)
{
   auto edges = GetVariableBinEdges();
   auto mdl = ::TH2D("h", "h", this->numBins, edges->data(), this->numBins, edges->data());

   DisableGPU();
   auto h1 = GetHisto2D(mdl, "x", "y");

   EnableGPU();
   auto h2 = GetHisto2D(mdl, "x", "y");

   CompareHistograms<double, TH2D>(h1, h2);
   delete edges;
}

TEST_P(HistoTestFixture2D, Fill2DWeightedVariableBins)
{
   auto edges = GetVariableBinEdges();
   auto mdl = ::TH2D("h", "h", this->numBins, edges->data(), this->numBins, edges->data());

   DisableGPU();
   auto h1 = GetHisto2D(mdl, "x", "y", "w");

   EnableGPU();
   auto h2 = GetHisto2D(mdl, "x", "y", "w");

   CompareHistograms<double, TH2D>(h1, h2);
   delete edges;
}

/***
 * Test 3D Histograms
 */

TEST_P(HistoTestFixture3D, Fill3DFixedBins)
{
   auto mdl = ::TH3D("h", "h", this->numBins, this->startBin, this->endBin, this->numBins, this->startBin, this->endBin,
                     this->numBins, this->startBin, this->endBin);

   DisableGPU();
   auto h1 = GetHisto3D(mdl, "x", "y", "z");

   EnableGPU();
   auto h2 = GetHisto3D(mdl, "x", "y", "z");

   CompareHistograms<double, TH3D>(h1, h2);
}

TEST_P(HistoTestFixture3D, Fill3DWeightedFixedBins)
{
   auto mdl = ::TH3D("h", "h", this->numBins, this->startBin, this->endBin, this->numBins, this->startBin, this->endBin,
                     this->numBins, this->startBin, this->endBin);

   DisableGPU();
   auto h1 = GetHisto3D(mdl, "x", "y", "z", "w");

   EnableGPU();
   auto h2 = GetHisto3D(mdl, "x", "y", "z", "w");

   CompareHistograms<double, TH3D>(h1, h2);
}

TEST_P(HistoTestFixture3D, Fill3DVariableBins)
{
   auto edges = GetVariableBinEdges();
   auto mdl =
      ::TH3D("h", "h", this->numBins, edges->data(), this->numBins, edges->data(), this->numBins, edges->data());

   DisableGPU();
   auto h1 = GetHisto3D(mdl, "x", "y", "z");

   EnableGPU();
   auto h2 = GetHisto3D(mdl, "x", "y", "z");

   CompareHistograms<double, TH3D>(h1, h2);
   delete edges;
}

TEST_P(HistoTestFixture3D, Fill3DWeightedVariableBins)
{
   auto edges = GetVariableBinEdges();
   auto mdl =
      ::TH3D("h", "h", this->numBins, edges->data(), this->numBins, edges->data(), this->numBins, edges->data());

   DisableGPU();
   auto h1 = GetHisto3D(mdl, "x", "y", "z", "w");

   EnableGPU();
   auto h2 = GetHisto3D(mdl, "x", "y", "z", "w");

   CompareHistograms<double, TH3D>(h1, h2);
   delete edges;
}
