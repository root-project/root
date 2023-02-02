/*
 * Project: RooFit
 * Authors:
 *   ZW, Zef Wolffs, Nikhef, zefwolffs@gmail.com
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef ROOT_ROOFIT_MultiProcess_HeatmapAnalyzer
#define ROOT_ROOFIT_MultiProcess_HeatmapAnalyzer

#include "TString.h"
#include "TH2I.h"

#include <memory>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace RooFit {
namespace MultiProcess {

class HeatmapAnalyzer {
public:
   HeatmapAnalyzer(std::string const& logs_dir);

   // main method of this class, produces heatmap
   std::unique_ptr<TH2I> analyze(int analyzed_gradient);

   // getters to inspect logfiles
   std::vector<std::string> const getPartitionNames();
   std::vector<std::string> const getTaskNames();
   json const getMetadata();

private:
   // internal helper functions
   std::string findTaskForDuration(json durations, int start_t, int end_t);
   void sortTaskNames(std::vector<std::string> &task_names);

   TH2I matrix_;

   json gradients_;
   json metadata_;
   std::vector<json> durations_;

   std::vector<std::string> tasks_names_;
   std::vector<std::string> eval_partitions_names_;
};

} // namespace MultiProcess
} // namespace RooFit

#endif // ROOT_ROOFIT_MultiProcess_HeatmapAnalyzer