#include <iostream>
#include <memory>
#include <string>
#include <gtest/gtest.h>

#include "TMVA/DecisionTree.h"
#include "TMVA/Types.h"
#include "TMVA/Event.h"
#include "TMVA/DecisionTreeNode.h"
#include "TMVA/DataSetInfo.h"

#include "Utility.h"

TEST(Multiregression, DifferentDimensions) {
	UInt_t test_size = 1000;
	for (UInt_t target_dim = 2; target_dim < 10; ++target_dim) {
		std::vector<std::pair<double, double>> target_limits;
		for (UInt_t limit_index = 0; limit_index < target_dim; ++limit_index) {
			UInt_t lower_limit = rand() % 50;
			UInt_t upper_limit = lower_limit + 1 + rand() % 50;
			target_limits.push_back(std::make_pair(lower_limit, upper_limit));
		}
		auto dataset = create_regression_dataset (test_size, 1, target_limits);
		TMVA::DecisionTreeNode::fgIsTraining = true;
		TMVA::DecisionTree tree (nullptr, 1, test_size, dataset->info.get(), 0, false, 1, false, 500, 42, 0.01, 1);
		tree.SetAnalysisType(TMVA::Types::kRegression);
		tree.SetNVars(1);
		tree.SetRoot(new TMVA::DecisionTreeNode);
		tree.BuildTree(*(dataset->data.get()), tree.GetRoot());
		for (UInt_t test_index = 0; test_index < test_size; ++test_index) {
			auto variable = static_cast<Float_t> (rand() % test_size);
			auto response = tree.GetMultiResponse(new TMVA::Event {std::vector<Float_t> {variable},
															std::vector<Float_t> (target_dim, 0)});
			for (UInt_t value_index = 0; value_index < target_dim; ++value_index) {
				ASSERT_TRUE(almost_equal_trivial(response[value_index], target_limits[value_index].first +
												variable/test_size * (target_limits[value_index].second - 
												target_limits[value_index].first)));
			}
		}
		TMVA::DecisionTreeNode::fgIsTraining = false;
	}
}
