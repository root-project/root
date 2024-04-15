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

namespace RooFit {
namespace MultiProcess {

namespace Detail {
// To avoid unnecessary dependence on nlohman json in the interface. Note that
// we should not forward declare nlohmann::json directly, since its declaration
// might change (it is currently a typedef). With this wrapper type, we are
// completely decoupled on nlohmann::json in the RMetaData interface.
struct HeatmapAnalyzerJsonData;
} // namespace Detail

class HeatmapAnalyzer {
public:
   HeatmapAnalyzer(std::string const &logs_dir);
   ~HeatmapAnalyzer();

   // main method of this class, produces heatmap
   std::unique_ptr<TH2I> analyze(int analyzed_gradient);

   // getters to inspect logfiles
   std::vector<std::string> const getPartitionNames();
   std::vector<std::string> const getTaskNames();
   std::vector<std::string> const getMetadata();

private:
   TH2I matrix_;
   std::unique_ptr<Detail::HeatmapAnalyzerJsonData> jsonData_;
   std::vector<std::string> tasks_names_;
   std::vector<std::string> eval_partitions_names_;
};

} // namespace MultiProcess
} // namespace RooFit

#endif // ROOT_ROOFIT_MultiProcess_HeatmapAnalyzer
