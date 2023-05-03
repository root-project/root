#include <stdlib.h>
#include "gtest/gtest.h"

#include "ROOT/RDataFrame.hxx"
#include "RHnCUDA.h"
#include "TH1.h"
#include "TAxis.h"

/**
 * Helper function for toggling ON CUDA histogramming.
 */
char env[] = "CUDA_HIST";
void EnableCUDA()
{
   setenv(env, "1", 1);
}

template <typename T, int n>
std::array<T, n> Repeat(T val)
{
   std::array<T, n> result;
   result.fill(val);
   return result;
}

/**
 * Helper functions for element-wise comparison of histogram arrays
 */

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
void CompareArrays(T *result, T* expected, int n)
{
   CHECK_ARRAY(result, expected, n)
}

template<>
void CompareArrays(float *result, float* expected, int n)
{
   CHECK_ARRAY_FLOAT(result, expected, n)
}

template<>
void CompareArrays(double *result, double* expected, int n)
{
   CHECK_ARRAY_DOUBLE(result, expected, n)
}


// std::vector<double> *GetVariableBinEdges(int startBin, int numBins)
// {
//    int e = startBin;
//    auto edges = new std::vector<double>(numBins + 1);
//    std::generate(edges->begin(), edges->end(), [&]() { return e++; });
//    (*edges)[numBins] += 10;

//    return edges;
// }

// Create all combinations of datatype and dimension
// Partially taken from https://stackoverflow.com/questions/56115790/gtest-parametrized-tests-for-different-types

// "parameters"
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
   const static int numBins = 40;
   const double startBin = 0;
   const double endBin = 38;

   // int, double, float
   using histType = typename T::type;

   // 1, 2, or 3
   static constexpr int dim = T::GetDim();

   const static int nCells = pow(numBins, dim);
   int nStats;

   histType result[nCells], expected[nCells];
   double *stats;

   void SetUp() override
   {
      EnableCUDA();
      nStats = 2 + 2 * dim;
      if (dim > 1)
         nStats += TMath::Binomial(dim, 2);

      stats = new double[nStats];
   }

   void TearDown() override { delete[] stats; }
};

template <typename T>
struct Test;

template <typename... T>
struct Test<std::tuple<T...>> {
   using Types = ::testing::Types<T...>;
};

using TestTypes = Test<Combinations_t<std::tuple<double, float, int>, OneDim, TwoDim, ThreeDim>>::Types;
TYPED_TEST_SUITE(HistoTestFixture, TestTypes);

TYPED_TEST(HistoTestFixture, FillFixedBins)
{
   // int, double, or float
   using t = typename TypeParam::type;
   auto h =
      CUDAhist::RHnCUDA<t, this->dim>(Repeat<int, this->dim>(this->numBins), Repeat<double, this->dim>(this->startBin),
                                      Repeat<double, this->dim>(this->endBin));

   // Underflow
   h.Fill(Repeat<double, this->dim>(this->startBin - 1));
   this->expected[0] = (t)1;

   // Center
   h.Fill(Repeat<double, this->dim>((this->startBin + this->endBin) / 2.));
   this->expected[(int)round(this->nCells / 2)] = (t)1;

   // // Overflow
   h.Fill(Repeat<double, this->dim>(this->endBin + 1));
   this->expected[this->nCells - 1] = (t)1;

   h.RetrieveResults(this->result, this->stats);
   CompareArrays(this->result, this->expected, this->nCells);
}

TYPED_TEST(HistoTestFixture, FillFixedBinsWeighted)
{
   // int, double, or float
   using t = typename TypeParam::type;
   auto h =
      CUDAhist::RHnCUDA<t, this->dim>(Repeat<int, this->dim>(this->numBins), Repeat<double, this->dim>(this->startBin),
                                      Repeat<double, this->dim>(this->endBin));

   // Underflow
   h.Fill(Repeat<double, this->dim>(this->startBin - 1), (t)7);
   this->expected[0] = (t)7;

   // Center
   h.Fill(Repeat<double, this->dim>((this->startBin + this->endBin) / 2.), (t)7);
   this->expected[(int)round(this->nCells / 2)] = (t)7;

   // Overflow
   h.Fill(Repeat<double, this->dim>(this->endBin + 1), (t)7);
   this->expected[this->nCells - 1] = (t)7;

   h.RetrieveResults(this->result, this->stats);
   CompareArrays(this->result, this->expected, this->nCells);
}

// TYPED_TEST(HistoTestFixture, FillVariableBins)
// {
//    // int, double, or float
//    using t = typename TypeParam::type;
//    std::vector<const double *> binEdges;
//    for (auto i = 0; i < this->dim; i++)
//       binEdges.push_back(GetVariableBinEdges(this->startBin, this->numBins));
//    auto h = CUDAhist::RHnCUDA<t, this->dim>({this->numBins}, {this->startBin}, {this->endBin}, binEdges.data());

//    // Underflow
//    std::array<double, this->dim> uCoords;
//    uCoords.fill(this->startBin - 1);
//    h.Fill(uCoords);
//    this->expected[0] = (t)1;

//    // Center
//    std::array<double, this->dim> cCoords;
//    uCoords.fill((this->startBin + this->endBin) / 2);
//    h.Fill(cCoords);
//    this->expected[(int)pow(this->numBins, this->dim) / this->dim] = (t)1;

//    // Overflow
//    std::array<double, this->dim> oCoords;
//    uCoords.fill(this->endBin + 1);
//    h.Fill(oCoords);
//    this->expected[this->numBins - 1] = (t)1;

//    h.RetrieveResults(this->result, this->stats);
//    CHECK_ARRAY(this->result, this->expected, this->numBins);
// }
