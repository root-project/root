#include "TMath.h"

#include "gtest/gtest.h"

TEST(DNNDensity, BasicTests)
{
   auto check = [](std::vector<double> values, int k, double dist, int k_actual) {
      double query = 0.;
      std::span<const double> queryspan{&query, 1};

      double output = 0.;
      std::span<double> outspan{&output, 1};

      // To compute the reference value
      std::size_t n = values.size();
      const double factor = n * 0.5 / (n + 1.0);

      if (!values.empty())
         dist = std::min(dist, values.back() - values.front());

      double dmin = 1.;
      TMath::KNNDensity(values, queryspan, outspan, k, k == 0 ? dmin : 0.);
      double ref = factor == 0. ? 0. : factor * k_actual * (1. / std::abs(dist));

      EXPECT_DOUBLE_EQ(output, ref);
   };

   check({}, /*k*/ 0, /*dist*/ 0., /*k_actual*/ 0);
   check({}, /*k*/ 1, /*dist*/ 0., /*k_actual*/ 0);
   check({}, /*k*/ 2, /*dist*/ 0., /*k_actual*/ 0);

   std::vector<double> values{-3.1, -2.4, -1.5, 0., 0.9, 1.9, 2.9, 4.4};
   std::vector<double> valuesabs = values;
   std::sort(valuesabs.begin(), valuesabs.end(), [](double a, double b) { return std::abs(a) < std::abs(b); });

   for (std::size_t i = 0; i < values.size() - 1; ++i) {
      double dist = valuesabs[std::max(i, std::size_t(1))];
      int k_actual = std::max(i + 1, std::size_t(2));
      check(values, /*k=*/i + 1, dist, k_actual);
   }

   check({0., 0.1}, 0, 1, 2);

   check({3.5, 2.5, 1.5, 0., 1., 2., 3., 4.}, 0, 1, 1);

   check({0., 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4}, 0, 1.2, 7);
}
