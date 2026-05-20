#include "gtest/gtest.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TGraphAsymmErrors.h"
#include "TGraphMultiErrors.h"
#include "TGraphBentErrors.h"

#include <algorithm>
#include <vector>

TEST(TGraphSortTest, TGraphSortingTest)
{
   const int numEntries = 1000000;
   std::vector<Double_t> x(numEntries);
   std::vector<Double_t> y(numEntries);

   // Initialize the graph data with unsorted data
   for (int i = 0; i < numEntries; i++) {
      x[i] = numEntries - i;
      y[i] = x[i] * x[i];
   }

   TGraph graph(numEntries, x.data(), y.data());

   graph.Sort();

   // Check if the graph x values are sorted
   bool isSorted = std::is_sorted(graph.GetX(), graph.GetX() + numEntries);

   ASSERT_TRUE(isSorted);
}

TEST(TGraphSortTest, TGraphErrorsSortingTest)
{
    const int numEntries = 1000000;
    std::vector<Double_t> x(numEntries);
    std::vector<Double_t> y(numEntries);
    std::vector<Double_t> errors(numEntries);

    // Initialize the graph data with unsorted data
    for (int i = 0; i < numEntries; i++) {
        x[i] = numEntries - i;
        y[i] = x[i] * x[i];
        errors[i] = 1.0 / (i + 1);
    }

    TGraphErrors graphErr(numEntries, x.data(), y.data(), errors.data(), errors.data());

    graphErr.Sort();

    // Check if the graph x values are sorted
    bool isValSorted = std::is_sorted(graphErr.GetX(), graphErr.GetX() + numEntries);

    // Check if the graph errors are sorted based on the sorted values
    bool isErrSorted = std::is_sorted(graphErr.GetEX(), graphErr.GetEX() + numEntries);

    ASSERT_TRUE(isValSorted);
    ASSERT_TRUE(isErrSorted);
}

TEST(TGraphSortTest, TGraphAsymmErrorsSortingTest)
{
    const int numEntries = 1000000;
    std::vector<Double_t> x(numEntries);
    std::vector<Double_t> y(numEntries);
    std::vector<Double_t> elow(numEntries);
    std::vector<Double_t> ehigh(numEntries);

    // Initialize the graph data with unsorted data
    for (int i = 0; i < numEntries; i++) {
        x[i] = numEntries - i;
        y[i] = x[i] * x[i];
        elow[i] = 1.0 / (i + 1);
        ehigh[i] = 2.0 / (i + 1);
        
    }

    TGraphAsymmErrors graphAsymmErr(numEntries, x.data(), y.data(), elow.data(), ehigh.data(), elow.data(), ehigh.data());

    graphAsymmErr.Sort();

    // Check if the graph x values are sorted
    bool isValSorted = std::is_sorted(graphAsymmErr.GetX(), graphAsymmErr.GetX() + numEntries);

    // Check if the graph errors are sorted based on the sorted values
    bool isErrSorted = std::is_sorted(graphAsymmErr.GetEYlow(), graphAsymmErr.GetEYlow() + numEntries);

    ASSERT_TRUE(isValSorted);
    ASSERT_TRUE(isErrSorted);
}

TEST(TGraphSortTest, TGraphBentErrorsSortingTest)
{
    const int numEntries = 1000000;
    std::vector<Double_t> x(numEntries);
    std::vector<Double_t> y(numEntries);
    std::vector<Double_t> elow(numEntries);
    std::vector<Double_t> ehigh(numEntries);
    std::vector<Double_t> elowd(numEntries);
    std::vector<Double_t> ehighd(numEntries);

    // Initialize the graph data with unsorted data
    for (int i = 0; i < numEntries; i++) {
        x[i] = numEntries - i;
        y[i] = x[i] * x[i];
        elow[i] = 1.0 / (i + 1);
        ehigh[i] = 2.0 / (i + 1);
        elowd[i] = 3.0 / (i + 1);
        ehighd[i] = 0.5 * i;
    }

    TGraphBentErrors graphBentErr(numEntries, x.data(), y.data(), elow.data(), ehigh.data(), elowd.data(), ehighd.data(), elow.data(), ehigh.data(), elowd.data(), ehighd.data());

    graphBentErr.Sort();

    // Check if the graph x values are sorted
    bool isValSorted = std::is_sorted(graphBentErr.GetX(), graphBentErr.GetX() + numEntries);

    // Check if the graph errors are sorted based on the sorted values
    bool isErrSorted = std::is_sorted(graphBentErr.GetEXlow(), graphBentErr.GetEXlow() + numEntries);
    bool isErrdSorted = std::is_sorted(graphBentErr.GetEYhighd(), graphBentErr.GetEYhighd() + numEntries);

    ASSERT_TRUE(isValSorted);
    ASSERT_TRUE(isErrSorted);
    ASSERT_FALSE(isErrdSorted);
}

TEST(TGraphSortTest, TGraphMultiErrorsSortingTest)
{
    const int numEntries = 1000000;
    std::vector<Double_t> x(numEntries);
    std::vector<Double_t> y(numEntries);
    std::vector<Double_t> elow(numEntries);
    std::vector<Double_t> ehigh(numEntries);

    // Initialize the graph data with unsorted data
    for (int i = 0; i < numEntries; i++) {
        x[i] = numEntries - i;
        y[i] = x[i] * x[i];
        elow[i] = 1.0 / (i + 1);
        ehigh[i] = 2.0 / (i + 1);
    }

    TGraphMultiErrors graphMultiErr(numEntries, x.data(), y.data(), elow.data(), ehigh.data(), elow.data(), ehigh.data());

    graphMultiErr.Sort();

    // Check if the graph x values are sorted
    bool isValSorted = std::is_sorted(graphMultiErr.GetX(), graphMultiErr.GetX() + numEntries);

    // Check if the graph errors are sorted based on the sorted values
    bool isErrSorted = std::is_sorted(graphMultiErr.GetEXlow(), graphMultiErr.GetEXlow() + numEntries);

    ASSERT_TRUE(isValSorted);
    ASSERT_TRUE(isErrSorted);
}
