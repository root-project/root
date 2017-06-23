#include <gtest/gtest.h>
#include <cstdlib>

#include "TMVA/RegressionVariance.h"

#include "Utility.h"

TEST(MultiVariance, Simple) {
	std::vector<UInt_t> dimensions {1, 2, 4, 8};
	auto variance = TMVA::RegressionVariance();
	for (UInt_t test_iter = 0; test_iter < 100; ++test_iter) {
		for (auto dim : dimensions) {
			std::vector<Double_t> means_vector (dim, 0.0);
			std::vector<Double_t> sum_squares (dim, 0.0);
			Double_t sum = 0.0;
			for (UInt_t index = 0; index < dim; ++index) {
				means_vector[index] = static_cast <Double_t> ((1.0 - rand() % 3) * (rand()) / 
									  static_cast <Double_t> (RAND_MAX));
				sum_squares[index] += dim * means_vector[index] +
							   		  static_cast <Double_t> (rand()) / static_cast <Double_t> (RAND_MAX);
				sum += sum_squares[index];
			}
			Double_t var_truth = 0.0;
			Double_t var_check = 0.0;
			for (UInt_t index = 0; index < dim; ++index) {
				var_truth += variance.GetSeparationIndex(100, means_vector[index], sum_squares[index]);
			}
			var_check = variance.GetSeparationIndexMulti(100, &means_vector[0], sum, dim);
			ASSERT_TRUE(almost_equal_trivial(var_truth, var_check));
		}
	}
}
