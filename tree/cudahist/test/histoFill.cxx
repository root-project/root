////////////////////////////////////////////////////////////////////////////////////
/// Tests for filling RHnCUDA histograms with different data types and dimensions.
///
#include <climits>
#include <stdlib.h>
#include "gtest/gtest.h"

#include "ROOT/RDataFrame.hxx"
#include "RHnCUDA.h"
#include "TH1.h"
#include "TAxis.h"

// Helper function for toggling ON CUDA histogramming.
char env[] = "CUDA_HIST";
void EnableCUDA()
{
   setenv(env, "1", 1);
}

// Returns an array with the given value repeated n times.
template <typename T, int n>
std::array<T, n> Repeat(T val)
{
   std::array<T, n> result;
   result.fill(val);
   return result;
}

// Helper functions for element-wise comparison of histogram arrays.
#define CHECK_ARRAY(a, b, n)                              \
   {                                                      \
      for (auto i : ROOT::TSeqI(n)) {                     \
         EXPECT_EQ(a[i], b[i]) << "  at index i = " << i; \
      }                                                   \
   }

#define CHECK_ARRAY_FLOAT(a, b, n)                              \
   {                                                            \
      for (auto i : ROOT::TSeqI(n)) {                           \
         EXPECT_FLOAT_EQ(a[i], b[i]) << "  at index i = " << i; \
      }                                                         \
   }

#define CHECK_ARRAY_DOUBLE(a, b, n)                              \
   {                                                             \
      for (auto i : ROOT::TSeqI(n)) {                            \
         EXPECT_DOUBLE_EQ(a[i], b[i]) << "  at index i = " << i; \
      }                                                          \
   }

template <typename T>
void CompareArrays(T *result, T *expected, int n)
{
   CHECK_ARRAY(result, expected, n)
}

template <>
void CompareArrays(float *result, float *expected, int n)
{
   CHECK_ARRAY_FLOAT(result, expected, n)
}

template <>
void CompareArrays(double *result, double *expected, int n)
{
   CHECK_ARRAY_DOUBLE(result, expected, n)
}

// Create all combinations of datatype and dimension
// Partially taken from https://stackoverflow.com/questions/56115790/gtest-parametrized-tests-for-different-types

// Test "parameters"
struct OneDim {
   static constexpr int dim = 1;
};
struct TwoDim {
   static constexpr int dim = 2;
};
struct ThreeDim {
   static constexpr int dim = 3;
};

template <typename T, class P>
struct Case {
   using type = T;
   static constexpr int GetDim() { return P::dim; }
};

template <class TupleType, class TupleParam, std::size_t I>
struct make_case {
   static constexpr std::size_t N = std::tuple_size<TupleParam>::value;
   using type =
      Case<typename std::tuple_element<I / N, TupleType>::type, typename std::tuple_element<I % N, TupleParam>::type>;
};

template <class T1, class T2, class Is>
struct make_combinations;

template <class TupleType, class TupleParam, std::size_t... Is>
struct make_combinations<TupleType, TupleParam, std::index_sequence<Is...>> {
   using tuples = std::tuple<typename make_case<TupleType, TupleParam, Is>::type...>;
};

template <class TupleTypes, class... Params>
using Combinations_t = typename make_combinations<
   TupleTypes, std::tuple<Params...>,
   std::make_index_sequence<(std::tuple_size<TupleTypes>::value) * (sizeof...(Params))>>::tuples;

template <typename T>
class HistoTestFixture : public ::testing::Test {
protected:
   // Includes u/overflow bins. Uneven number chosen to have a center bin.
   const static int numBins = 5;

   // Variables for defining fixed bins.
   const double startBin = 1;
   const double endBin = 4;

   // int, double, float
   using histType = typename T::type;

   // 1, 2, or 3
   static constexpr int dim = T::GetDim();

   // Total number of cells
   const static int nCells = pow(numBins, dim);
   histType result[nCells], expectedHist[nCells];

   double *stats, *expectedStats;
   int nStats;

   CUDAhist::RHnCUDA<histType, dim> histogram;

   HistoTestFixture() : histogram(Repeat<int, dim>(numBins), Repeat<double, dim>(startBin), Repeat<double, dim>(endBin))
   {
   }

   void SetUp() override
   {
      EnableCUDA();
      nStats = 2 + 2 * dim;
      if (dim > 1)
         nStats += TMath::Binomial(dim, 2);

      stats = new double[nStats];
      expectedStats = new double[nStats];

      memset(stats, 0, nStats * sizeof(double));
      memset(expectedStats, 0, nStats * sizeof(double));
      memset(expectedHist, 0, nCells * sizeof(histType));
   }

   void TearDown() override { delete[] stats; }

   bool UOverflow(std::array<double, dim> coord)
   {
      for (auto d = 0; d < dim; d++) {
         if (coord[d] < startBin || coord[d] > endBin)
            return true;
      }
      return false;
   }

   void GetExpectedStats(std::vector<std::array<double, dim>> coords, histType weight)
   {
      for (auto i = 0; i < (int)coords.size(); i++) {
         if (UOverflow(coords[i]))
            continue;

         // Tsumw
         expectedStats[0] += weight;
         // Tsumw2
         expectedStats[1] += weight * weight;

         auto offset = 2;
         for (auto d = 0; d < dim; d++) {
            // e.g. Tsumwx
            expectedStats[offset++] += weight * coords[i][d];
            // e.g. Tsumwx2
            expectedStats[offset++] += weight * pow(coords[i][d], 2);

            for (auto prev_d = 0; prev_d < d; prev_d++) {
               // e.g. Tsumwxy
               this->expectedStats[offset++] += weight * coords[i][d] * coords[i][prev_d];
            }
         }
      }
   }
};

template <typename T>
struct Test;

template <typename... T>
struct Test<std::tuple<T...>> {
   using Types = ::testing::Types<T...>;
};

using TestTypes = Test<Combinations_t<std::tuple<double, float, int, short>, OneDim, TwoDim, ThreeDim>>::Types;
TYPED_TEST_SUITE(HistoTestFixture, TestTypes);

/////////////////////////////////////
/// Test Cases

TYPED_TEST(HistoTestFixture, FillFixedBins)
{
   // int, double, or float
   using t = typename TypeParam::type;
   auto h = this->histogram;

   std::vector<std::array<double, this->dim>> coords = {
      Repeat<double, this->dim>(this->startBin - 1),                   // Underflow
      Repeat<double, this->dim>((this->startBin + this->endBin) / 2.), // Center
      Repeat<double, this->dim>(this->endBin + 1)                      // OVerflow
   };
   auto weight = (t)1;

   std::vector<int> expectedHistBins = {0, this->nCells / 2, this->nCells - 1};

   for (auto i = 0; i < (int)coords.size(); i++) {
      h.Fill(coords[i]);
      this->expectedHist[expectedHistBins[i]] = weight;
   }

   h.RetrieveResults(this->result, this->stats);

   {
      SCOPED_TRACE("Check fill result");
      CompareArrays(this->result, this->expectedHist, this->nCells);
   }

   {
      SCOPED_TRACE("Check statistics");
      this->GetExpectedStats(coords, weight);
      CompareArrays(this->stats, this->expectedStats, this->nStats);
   }
}

TYPED_TEST(HistoTestFixture, FillFixedBinsWeighted)
{
   // int, double, or float
   using t = typename TypeParam::type;
   auto h = this->histogram;

   std::vector<std::array<double, this->dim>> coords = {
      Repeat<double, this->dim>(this->startBin - 1),                   // Underflow
      Repeat<double, this->dim>((this->startBin + this->endBin) / 2.), // Center
      Repeat<double, this->dim>(this->endBin + 1)                      // OVerflow
   };
   auto weight = (t)7;

   std::vector<int> expectedHistBins = {0, this->nCells / 2, this->nCells - 1};

   for (auto i = 0; i < (int)coords.size(); i++) {
      h.Fill(coords[i], weight);
      this->expectedHist[expectedHistBins[i]] = weight;
   }

   h.RetrieveResults(this->result, this->stats);

   {
      SCOPED_TRACE("Check fill result");
      CompareArrays(this->result, this->expectedHist, this->nCells);
   }

   {
      SCOPED_TRACE("Check statistics");
      this->GetExpectedStats(coords, weight);
      CompareArrays(this->stats, this->expectedStats, this->nStats);
   }
}

TEST(HistoTestFixture, FillIntClamp)
{
   auto h = CUDAhist::RHnCUDA<int, 1>({6}, {0}, {4});
   h.Fill({0}, INT_MAX);
   h.Fill({3}, -INT_MAX);

   for (int i = 0; i < 100; i++) {     // Repeat to test for race conditions
      h.Fill({0});                     // Should keep max value
      h.Fill({1}, long(INT_MAX) + 1);  // Clamp positive overflow
      h.Fill({2}, -long(INT_MAX) - 1); // Clamp negative overflow
      h.Fill({3}, -1);                 // Should keep min value
   }

   int r[6];
   double s[4];
   h.RetrieveResults(r, s);

   EXPECT_EQ(r[0], 0);
   EXPECT_EQ(r[1], INT_MAX);
   EXPECT_EQ(r[2], INT_MAX);
   EXPECT_EQ(r[3], -INT_MAX);
   EXPECT_EQ(r[4], -INT_MAX);
   EXPECT_EQ(r[5], 0);
}

TEST(HistoTestFixture, FillShortClamp)
{
   auto h = CUDAhist::RHnCUDA<short, 1>({10}, {0}, {8});

   // Filling short histograms is implemented using atomic operations on integers so we test each case
   // twice to test the for correct filling of the lower and upper bits.
   for (int offset = 0; offset < 2; offset++) {
      h.Fill({0. + offset}, 32767);
      h.Fill({2. + offset}, -32767);

      for (int i = 0; i < 100; i++) {   // Repeat to test for race conditions
         h.Fill({0. + offset});         // Keep max value
         h.Fill({2. + offset}, -1);     // Keep min value
         h.Fill({4. + offset}, 32769);  // Clamp positive overflow
         h.Fill({6. + offset}, -32769); // Clamp negative overflow
      }
   }

   short r[10];
   double s[4];
   h.RetrieveResults(r, s);

   int expected[10] = {0, 32767, 32767, -32767, -32767, 32767, 32767, -32767, -32767, 0};
   CHECK_ARRAY(r, expected, 10);
}
