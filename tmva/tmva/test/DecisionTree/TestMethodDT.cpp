#include <gtest/gtest.h>

#include "TMVA/MethodDT.h"
#include "TMVA/MethodBDT.h"
#include "TMVA/Types.h"
#include "TMVA/Event.h"
#include "TMVA/DataSetInfo.h"
#include "TMVA/DataSet.h"
#include "TMVA/DataSetManager.h"

struct TestDataset {
	std::unique_ptr<TMVA::DataSetInfo> info;
	std::unique_ptr<std::vector<TMVA::Event*>> data;
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
	dataset->data = std::unique_ptr<std::vector<TMVA::Event*>> (new std::vector<TMVA::Event*>);
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

TEST(MethodDT, Multiregression) {
	UInt_t test_size = 100;
	std::vector<std::pair<double, double>> target_limits;
	for (UInt_t limit_index = 0; limit_index < 10; ++limit_index) {
		UInt_t lower_limit = rand() % 50;
		UInt_t upper_limit = lower_limit + 1 + rand() % 50;
		target_limits.push_back(std::make_pair(lower_limit, upper_limit));
	}
	auto data_wrapper = create_regression_dataset(1, test_size, target_limits);
}

TEST(MethodBDT, Multiregression) { //multitarget regression currenly supported in Bagging
	UInt_t test_size = 100;
	std::vector<std::pair<double, double>> target_limits;
	for (UInt_t limit_index = 0; limit_index < 10; ++limit_index) {
		UInt_t lower_limit = rand() % 50;
		UInt_t upper_limit = lower_limit + 1 + rand() % 50;
		target_limits.push_back(std::make_pair(lower_limit, upper_limit));
	}
	auto data_wrapper = create_regression_dataset(1, test_size, target_limits);
}