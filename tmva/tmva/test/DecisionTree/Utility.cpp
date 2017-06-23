#include "Utility.h"

bool almost_equal_trivial(double x, double y) {
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