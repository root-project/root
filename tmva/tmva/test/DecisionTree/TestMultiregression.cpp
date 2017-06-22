#include <iostream>
#include <memory>
#include <string>
#include <gtest/gtest.h>

#include "TMVA/DecisionTree.h"
#include "TMVA/Types.h"
#include "TMVA/Event.h"
#include "TMVA/DecisionTreeNode.h"
#include "TMVA/DataSetInfo.h"

struct TestDataset {
	std::unique_ptr<TMVA::DataSetInfo> info;
	std::unique_ptr<std::vector<const TMVA::Event*>> data;
	~TestDataset() {
		for (auto ev : *data.get())
			delete ev;
	}
};

bool almost_equal_double(double x, double y) {
   // the machine epsilon has to be scaled to the magnitude of the values used
   // and multiplied by the desired precision in ULPs (units in the last place)
   return std::abs(x-y) < 1E-4;
}

std::unique_ptr<TestDataset> create_regression_dataset(uint32_t events_number,
													   uint32_t features_number,
														   const std::vector<std::pair<double,
													   			         double>>& target_limits) {
	std::unique_ptr<TestDataset> dataset (new TestDataset);
	dataset->info = std::unique_ptr<TMVA::DataSetInfo> (new TMVA::DataSetInfo);
	dataset->data = std::unique_ptr<std::vector<const TMVA::Event*>> (new std::vector<const TMVA::Event*>);
	for (uint32_t feature_index = 0; feature_index < features_number; ++feature_index)
		dataset->info->AddVariable(std::to_string(feature_index));
	for (uint32_t target_index = 0; target_index < target_limits.size(); ++target_index)
		dataset->info->AddTarget("_", "_", "_", target_limits[target_index].first,
												target_limits[target_index].second);
	for (uint32_t event_index = 0; event_index < events_number; ++event_index) {
		std::vector<Float_t> features (features_number, event_index);
		std::vector<Float_t> targets;
		for (auto& limit : target_limits) {
			targets.push_back(limit.first +
							  (limit.second - limit.first) / events_number * event_index);
		}
		dataset->data->emplace_back(new TMVA::Event {std::vector<Float_t> (features_number, event_index),
														 targets});
	}
	return dataset;
}

void TestSingleTargetRegression() {
	auto dataset = create_regression_dataset(100, 2, std::vector<std::pair<double, double>> (1,
													 std::make_pair(0, 50)));
	TMVA::DecisionTreeNode::fgIsTraining = true;
	TMVA::DecisionTree tree (nullptr, 2, 0, dataset->info.get(), 0, false, 1, false, 500, 42, 0.01, 1);
	tree.SetAnalysisType(TMVA::Types::kRegression);
	tree.SetNVars(2);
	tree.SetRoot(new TMVA::DecisionTreeNode);
	tree.BuildTree(*(dataset->data.get()), tree.GetRoot());
	auto a = tree.CheckEvent(new TMVA::Event {std::vector<Float_t> (2, 10),
									 		  std::vector<Float_t> (2, 0)});
	std::cout << a << std::endl;
	TMVA::DecisionTreeNode::fgIsTraining = false;
}

void TestMultiTargetRegression() {
	auto dataset = create_regression_dataset(100, 2, std::vector<std::pair<double, double>> {
													 std::make_pair(0, 50), std::make_pair(50, 100)});
	TMVA::DecisionTreeNode::fgIsTraining = true;
	TMVA::DecisionTree tree (nullptr, 1, 200, dataset->info.get(), 0, false, 1, false, 500, 42, 0.01, 1);
	tree.SetAnalysisType(TMVA::Types::kMultiRegression);
	tree.SetNVars(2);
	tree.SetRoot(new TMVA::DecisionTreeNode);
	tree.BuildTree(*(dataset->data.get()), tree.GetRoot());
	auto a = tree.GetMultiResponse(new TMVA::Event {std::vector<Float_t> {101, 101},
									 		 	    std::vector<Float_t> (2, 0)});
	for (auto kek : a) {
		std::cout << kek << std::endl;
	}
	TMVA::DecisionTreeNode::fgIsTraining = false;
}

TEST(Multiregression, SingleVariable) {
	UInt_t test_size = 1000;
	for (UInt_t target_dim = 1; target_dim < 10; ++target_dim) {
		std::vector<std::pair<double, double>> target_limits;
		for (UInt_t limit_index = 0; limit_index < target_dim; ++limit_index) {
			UInt_t lower_limit = rand() % 50;
			UInt_t upper_limit = lower_limit + 1 + rand() % 50;
			target_limits.push_back(std::make_pair(lower_limit, upper_limit));
		}
		auto dataset = create_regression_dataset (test_size, 1, target_limits);
		TMVA::DecisionTreeNode::fgIsTraining = true;
		TMVA::DecisionTree tree (nullptr, 1, test_size, dataset->info.get(), 0, false, 1, false, 500, 42, 0.01, 1);
		tree.SetAnalysisType(TMVA::Types::kMultiRegression);
		tree.SetNVars(1);
		tree.SetRoot(new TMVA::DecisionTreeNode);
		tree.BuildTree(*(dataset->data.get()), tree.GetRoot());
		for (UInt_t test_index = 0; test_index < test_size; ++test_index) {
			auto variable = static_cast<Float_t> (rand() % test_size);
			auto response = tree.GetMultiResponse(new TMVA::Event {std::vector<Float_t> {variable},
															std::vector<Float_t> (target_dim, 0)});
			for (UInt_t value_index = 0; value_index < target_dim; ++value_index) {
				ASSERT_TRUE(almost_equal_double(response[value_index], target_limits[value_index].first +
												variable/test_size * (target_limits[value_index].second - 
												target_limits[value_index].first)));
			}
		}
		TMVA::DecisionTreeNode::fgIsTraining = false;
	}
}
