/// \file benchmark_rntuple_imt.C
/// \brief Standalone, reproducible benchmark for RNTupleDS IMT scheduling (Issue #23813 / PR #23327).
///
/// Compares scheduling performance and sequential cluster prefetching across thread slot counts (1, 2, 4, 8)
/// on both balanced datasets and datasets with tail stragglers.
///
/// Can be executed with ROOT directly:
///   root -l -b -q tree/dataframe/test/benchmark_rntuple_imt.C
///
/// Or with a remote / EOS / XRootD file list by passing arguments:
///   root -l -b -q 'tree/dataframe/test/benchmark_rntuple_imt.C("remote_file_list.txt")'

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RNTupleDS.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <TROOT.h>
#include <TSystem.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct BenchResult {
   unsigned nSlots = 0;
   double minMs = 0.0;
   double meanMs = 0.0;
   double stddevMs = 0.0;
   std::uint64_t totalEntries = 0;
};

std::vector<std::string> CreateBenchmarkDataset(const std::string &workDir, unsigned nFiles,
                                               unsigned baseClusters, unsigned stragglerClusters,
                                               unsigned entriesPerCluster)
{
   namespace fs = std::filesystem;
   fs::create_directories(workDir);

   std::vector<std::string> filePaths;
   filePaths.reserve(nFiles);

   std::cout << "Generating " << nFiles << " files in " << workDir << " ("
             << baseClusters << " base clusters, " << stragglerClusters << " straggler clusters, "
             << entriesPerCluster << " entries/cluster)..." << std::endl;

   for (unsigned i = 0; i < nFiles; ++i) {
      std::string path = workDir + "/bench_rntuple_" + std::to_string(i) + ".root";
      filePaths.push_back(path);

      // Tail straggler: every 4th file has significantly more clusters
      const unsigned nClusters = (i % 4 == 0) ? stragglerClusters : baseClusters;

      auto model = ROOT::RNTupleModel::Create();
      auto ptrX = model->MakeField<float>("x");
      auto ptrY = model->MakeField<double>("y");
      auto ptrId = model->MakeField<std::int32_t>("id");

      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "events", path);
      for (unsigned c = 0; c < nClusters; ++c) {
         for (unsigned e = 0; e < entriesPerCluster; ++e) {
            *ptrX = static_cast<float>(e % 100) * 0.01f;
            *ptrY = static_cast<double>(e + c * entriesPerCluster);
            *ptrId = static_cast<std::int32_t>(i * 100000 + c * entriesPerCluster + e);
            writer->Fill();
         }
         writer->CommitCluster();
      }
   }

   std::cout << "Dataset generation complete.\n" << std::endl;
   return filePaths;
}

BenchResult RunBenchmarkForSlots(const std::vector<std::string> &filePaths, unsigned nSlots, unsigned nReps)
{
   ROOT::EnableImplicitMT(nSlots);

   std::vector<double> timingsMs;
   timingsMs.reserve(nReps);
   std::uint64_t entriesCounted = 0;

   // Warmup run
   {
      auto df = ROOT::RDataFrame(std::make_unique<ROOT::RDF::RNTupleDS>("events", filePaths));
      auto count = df.Count();
      (void)count.GetValue();
   }

   // Timed runs
   for (unsigned rep = 0; rep < nReps; ++rep) {
      auto df = ROOT::RDataFrame(std::make_unique<ROOT::RDF::RNTupleDS>("events", filePaths));
      auto filtered = df.Filter([](float x) { return x > 0.25f; }, {"x"});
      auto sumY = filtered.Sum<double>("y");
      auto count = filtered.Count();

      const auto start = std::chrono::high_resolution_clock::now();
      entriesCounted = count.GetValue();
      (void)sumY.GetValue();
      const auto finish = std::chrono::high_resolution_clock::now();

      const double elapsedMs = std::chrono::duration<double, std::milli>(finish - start).count();
      timingsMs.push_back(elapsedMs);
   }

   ROOT::DisableImplicitMT();

   double minMs = timingsMs.front();
   double sum = 0.0;
   for (double t : timingsMs) {
      minMs = std::min(minMs, t);
      sum += t;
   }
   double meanMs = sum / timingsMs.size();

   double variance = 0.0;
   for (double t : timingsMs) {
      variance += (t - meanMs) * (t - meanMs);
   }
   double stddevMs = std::sqrt(variance / timingsMs.size());

   return BenchResult{nSlots, minMs, meanMs, stddevMs, entriesCounted};
}

void PrintResults(const std::vector<BenchResult> &results)
{
   std::cout << std::string(65, '=') << "\n";
   std::cout << std::setw(8) << "Slots"
             << std::setw(15) << "Min (ms)"
             << std::setw(15) << "Mean (ms)"
             << std::setw(15) << "StdDev (ms)"
             << std::setw(12) << "Speedup" << "\n";
   std::cout << std::string(65, '-') << "\n";

   double baselineMin = results.empty() ? 1.0 : results.front().minMs;
   for (const auto &r : results) {
      const double speedup = baselineMin / (r.minMs > 0.0 ? r.minMs : 1.0);
      std::cout << std::setw(8) << r.nSlots
                << std::setw(15) << std::fixed << std::setprecision(2) << r.minMs
                << std::setw(15) << std::fixed << std::setprecision(2) << r.meanMs
                << std::setw(15) << std::fixed << std::setprecision(2) << r.stddevMs
                << std::setw(11) << std::fixed << std::setprecision(2) << speedup << "x\n";
   }
   std::cout << std::string(65, '=') << "\n" << std::endl;
}

} // namespace

void benchmark_rntuple_imt(const char *fileListPath = nullptr)
{
   std::vector<std::string> filePaths;
   bool deleteTempFiles = false;
   std::string workDir = "/tmp/bench_rntuple_imt_data";

   if (fileListPath && strlen(fileListPath) > 0) {
      std::cout << "Loading file list from: " << fileListPath << std::endl;
      std::ifstream infile(fileListPath);
      std::string line;
      while (std::getline(infile, line)) {
         if (!line.empty() && line[0] != '#')
            filePaths.push_back(line);
      }
      if (filePaths.empty()) {
         std::cerr << "Error: No files found in " << fileListPath << std::endl;
         return;
      }
   } else {
      constexpr unsigned nFiles = 24;
      constexpr unsigned baseClusters = 8;
      constexpr unsigned stragglerClusters = 32;
      constexpr unsigned entriesPerCluster = 10000;
      filePaths = CreateBenchmarkDataset(workDir, nFiles, baseClusters, stragglerClusters, entriesPerCluster);
      deleteTempFiles = true;
   }

   const std::vector<unsigned> slotConfigurations = {1, 2, 4, 8};
   constexpr unsigned nReps = 3;

   std::cout << "Running benchmark on " << filePaths.size() << " files with "
             << nReps << " repetitions per slot configuration..." << std::endl;

   std::vector<BenchResult> results;
   for (unsigned nSlots : slotConfigurations) {
      std::cout << "Testing nSlots = " << nSlots << "..." << std::flush;
      auto res = RunBenchmarkForSlots(filePaths, nSlots, nReps);
      std::cout << " done (" << std::fixed << std::setprecision(1) << res.minMs << " ms)" << std::endl;
      results.push_back(res);
   }

   PrintResults(results);

   if (deleteTempFiles) {
      std::cout << "Cleaning up temporary benchmark files..." << std::endl;
      std::filesystem::remove_all(workDir);
      std::cout << "Done." << std::endl;
   }
}
