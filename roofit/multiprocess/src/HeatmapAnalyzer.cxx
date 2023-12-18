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

#include <RooFit/MultiProcess/HeatmapAnalyzer.h>

#include <TSystemDirectory.h>
#include <TList.h>

#include <nlohmann/json.hpp>

#include <fstream>

namespace RooFit {
namespace MultiProcess {

namespace Detail {

struct HeatmapAnalyzerJsonData {
   nlohmann::json gradients;
   nlohmann::json metadata;
   std::vector<nlohmann::json> durations;
};

} // namespace Detail

namespace {

void sortTaskNames(std::vector<std::string> &task_names)
{
   char const *digits = "0123456789";
   std::vector<int> digit_vec;
   std::vector<std::pair<int, std::string>> pair_vec;
   for (auto &&el : task_names) {
      std::size_t const n = el.find_first_of(digits);
      pair_vec.push_back(std::make_pair(stoi(el.substr(n)), el));
   }

   std::sort(pair_vec.begin(), pair_vec.end());

   for (size_t i = 0; i < task_names.size(); i++) {
      task_names[i] = pair_vec[i].second;
   }
}

std::string findTaskForDuration(nlohmann::json durations, int start_t, int end_t)
{
   for (auto &&el : durations.items()) {
      if (el.key().find("eval_partition") != std::string::npos)
         continue;

      for (size_t idx = 0; idx < durations[el.key()].size(); idx += 2) {
         if (durations[el.key()][idx] <= start_t && durations[el.key()][idx + 1] >= end_t) {
            return el.key();
         }
      }
   }
   return "";
}

} // namespace

/** \class HeatmapAnalyzer
 *
 * \brief Reads and processes logfiles produced by RooFit::MultiProcess::ProcessTimer
 *
 * RooFit::MultiProcess::ProcessTimer records timings of multiple processes simultaneously
 * and allows for these timings to be written out in json format, one for each process.
 * This class, the HeatmapAnalyzer, can read these json files and produce a heatmap from
 * them with partial derivatives on the y-axis, likelihood evaluations on the x-axis, and
 * time expenditures on the z-axis. This class also contains some convenience functions
 * for inspecting these log files.
 *
 * Note that this class requires the logfiles to contain three specific keys in the json:
 *      - `master:gradient` containing an array of gradient timestamps
 *      - `*eval_task*<task_number>` containing an array of task evaluation timestamps.
 *      - `*eval_partition*` containing an array of partition evaluation timestamps
 */

////////////////////////////////////////////////////////////////////////////////
/// HeatmapAnalyzer Constructor. This method reads the input files in the folder
/// specified by the user and creates internal attributes used by the other
/// methods in this class.
/// \param[in] logs_dir Directory where log files are stored in the format
///                     outputted by RooFit::MultiProcess::ProcessTimer.
///                     There can be other files in this directory as well.
HeatmapAnalyzer::HeatmapAnalyzer(std::string const &logs_dir)
   : jsonData_{std::make_unique<Detail::HeatmapAnalyzerJsonData>()}
{
   TSystemDirectory dir(logs_dir.c_str(), logs_dir.c_str());
   std::unique_ptr<TList> durationFiles{dir.GetListOfFiles()};

   for (TObject *file : *durationFiles) {
      if (std::string(file->GetName()).find("p_") == std::string::npos)
         continue;

      std::ifstream f(logs_dir + "/" + std::string(file->GetName()));

      if (std::string(file->GetName()).find("999") != std::string::npos) {
         jsonData_->gradients = nlohmann::json::parse(f);
      } else {
         jsonData_->durations.push_back(nlohmann::json::parse(f));
      }
   }

   for (nlohmann::json &durations_json : jsonData_->durations) {
      for (auto &&el : durations_json.items()) {
         if (el.key().find("eval_task") != std::string::npos &&
             std::find(tasks_names_.begin(), tasks_names_.end(), el.key()) == tasks_names_.end()) {
            tasks_names_.push_back(el.key());
         } else if (el.key().find("eval_partition") != std::string::npos &&
                    std::find(eval_partitions_names_.begin(), eval_partitions_names_.end(), el.key()) ==
                       eval_partitions_names_.end()) {
            eval_partitions_names_.push_back(el.key());
         } else if (el.key().find("metadata") != std::string::npos) {
            jsonData_->metadata = durations_json[el.key()];
         }
      }
   }

   for (nlohmann::json &durations_json : jsonData_->durations) {
      durations_json.erase("metadata");
   }

   sortTaskNames(tasks_names_);
}

HeatmapAnalyzer::~HeatmapAnalyzer() = default;

////////////////////////////////////////////////////////////////////////////////
/// This method is the main functionality in this class. It does the heavy
/// lifting of matching duration timestamps to tasks and partition evaluations.
/// \param[in] analyzed_gradient Gradient to analyze. For example, setting to 1
///                              analyzes the first gradient (ordered by time)
///                              in the logs.
std::unique_ptr<TH2I> HeatmapAnalyzer::analyze(int analyzed_gradient)
{
   int gradient_start_t = jsonData_->gradients["master:gradient"][analyzed_gradient * 2 - 2];
   int gradient_end_t = jsonData_->gradients["master:gradient"][analyzed_gradient * 2 - 1];

   auto total_matrix =
      std::make_unique<TH2I>("heatmap", "", eval_partitions_names_.size(), 0, 1, tasks_names_.size(), 0, 1);

   // loop over all logfiles stored in durations_
   for (nlohmann::json &durations_json : jsonData_->durations) {
      // partial heatmap is the heatmap that will be filled in for the current durations logfile
      auto partial_matrix =
         std::make_unique<TH2I>("partial_heatmap", "", eval_partitions_names_.size(), 0, 1, tasks_names_.size(), 0, 1);

      // remove unnecessary components (those that are out of range)
      for (auto &&el : durations_json.items()) {
         auto beg_interval =
            std::upper_bound(durations_json[el.key()].begin(), durations_json[el.key()].end(), gradient_start_t);
         auto end_interval =
            std::upper_bound(durations_json[el.key()].begin(), durations_json[el.key()].end(), gradient_end_t);
         durations_json[el.key()].erase(end_interval, durations_json[el.key()].end());
         durations_json[el.key()].erase(durations_json[el.key()].begin(), beg_interval);
      }

      // loops over all evaluated partitions in logfile
      for (std::string &eval_partition_name : eval_partitions_names_) {

         // for this partition, loops over all durations, i.e. start and end times for partition evaluations, and for
         // each tries to find the corresponding task
         for (size_t idx = 0; idx < durations_json[eval_partition_name].size(); idx += 2) {
            if (durations_json[eval_partition_name][idx + 1] > gradient_end_t ||
                durations_json[eval_partition_name][idx] < gradient_start_t)
               continue;
            std::string task_name = findTaskForDuration(durations_json, durations_json[eval_partition_name][idx],
                                                        durations_json[eval_partition_name][idx + 1]);

            if (task_name.empty())
               continue;

            // add found combination of task, partition evaluation, and duration to partial matrix
            int tasks_idx = find(tasks_names_.begin(), tasks_names_.end(), task_name) - tasks_names_.begin() + 1;
            int eval_partitions_idx =
               find(eval_partitions_names_.begin(), eval_partitions_names_.end(), eval_partition_name) -
               eval_partitions_names_.begin() + 1;
            partial_matrix->SetBinContent(eval_partitions_idx, tasks_idx,
                                          durations_json[eval_partition_name][idx + 1].get<int>() -
                                             durations_json[eval_partition_name][idx].get<int>());
         }
      }
      // add all partial matrices to form one matrix with entire gradient evaluation information
      total_matrix->Add(partial_matrix.get());
   }

   // do not need the legend in case heatmap is plotted
   total_matrix->SetStats(false);

   // set the axes labels on the heatmap matrix
   TAxis *y = total_matrix->GetYaxis();
   TAxis *x = total_matrix->GetXaxis();
   for (std::size_t i = 0; i != tasks_names_.size(); ++i) {
      y->SetBinLabel(i + 1, jsonData_->metadata[0][i].get<std::string>().c_str());
      y->ChangeLabel(i + 1, 30, 0.01, -1, -1, -1, "");
   }
   for (std::size_t i = 0; i != eval_partitions_names_.size(); ++i) {
      x->SetBinLabel(i + 1, eval_partitions_names_[i].c_str());
      x->ChangeLabel(i + 1, 30, -1, -1, -1, -1, "");
   }
   x->LabelsOption("v");

   return total_matrix;
}

std::vector<std::string> const HeatmapAnalyzer::getTaskNames()
{
   return tasks_names_;
}

std::vector<std::string> const HeatmapAnalyzer::getPartitionNames()
{
   return eval_partitions_names_;
}

std::vector<std::string> const HeatmapAnalyzer::getMetadata()
{
   std::vector<std::string> out;
   for (auto const &item : jsonData_->metadata[0]) {
      out.emplace_back(item.get<std::string>());
   }
   return out;
}

} // namespace MultiProcess
} // namespace RooFit
