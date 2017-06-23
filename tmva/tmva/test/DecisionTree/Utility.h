#pragma once

#include "TMVA/DataSetInfo.h"
#include "TMVA/Event.h"
#include <memory>

struct TestDataset {
	std::unique_ptr<TMVA::DataSetInfo> info;
	std::unique_ptr<std::vector<const TMVA::Event*>> data;
	~TestDataset() {
		for (auto ev : *data.get())
			delete ev;
	}
};

bool almost_equal_trivial(double x, double y);

std::unique_ptr<TestDataset> create_regression_dataset(uint32_t events_number,
													   uint32_t features_number,
													   const std::vector<std::pair<double,
												   			         double>>& target_limits);
