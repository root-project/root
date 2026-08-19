/// \file RNTupleMetrics.cxx
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2019-08-27
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RNTupleMetrics.hxx>

#include <ROOT/RConfig.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RLogger.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleUtils.hxx>
#include <ROOT/RNTupleWriter.hxx>

#include <TFile.h>
#include <TFileMerger.h>
#include <TMemFile.h>
#include <TSystem.h>

#include <cstdint>
#include <mutex>
#include <ostream>
#include <vector>

namespace {

std::mutex &GetMetricsExportMutex()
{
   static std::mutex mutex;
   return mutex;
}

thread_local bool gSuppressMetricsExport = false;

struct RSuppressMetaMetrics {
   RSuppressMetaMetrics() { gSuppressMetricsExport = true; }
   ~RSuppressMetaMetrics() { gSuppressMetricsExport = false; }
};

} // anonymous namespace

ROOT::Experimental::Detail::RNTuplePerfCounter::~RNTuplePerfCounter()
{
}

std::string ROOT::Experimental::Detail::RNTuplePerfCounter::ToString() const
{
   return fName + kFieldSeperator + fUnit + kFieldSeperator + fDescription + kFieldSeperator + GetValueAsString();
}

bool ROOT::Experimental::Detail::RNTupleMetrics::Contains(const std::string &name) const
{
  return GetLocalCounter(name) != nullptr;
}

const ROOT::Experimental::Detail::RNTuplePerfCounter*
ROOT::Experimental::Detail::RNTupleMetrics::GetLocalCounter(std::string_view name) const
{
   for (const auto &c : fCounters) {
      if (c->GetName() == name)
         return c.get();
   }
   return nullptr;
}

const ROOT::Experimental::Detail::RNTuplePerfCounter*
ROOT::Experimental::Detail::RNTupleMetrics::GetCounter(std::string_view name) const
{
   std::string prefix = fName + ".";
   if (name.compare(0, prefix.length(), std::string_view(prefix)) != 0)
      return nullptr;

   auto innerName = name.substr(prefix.length());
   if (auto counter = GetLocalCounter(innerName))
      return counter;

   for (auto m : fObservedMetrics) {
      auto counter = m->GetCounter(innerName);
      if (counter != nullptr)
         return counter;
   }

   return nullptr;
}

void ROOT::Experimental::Detail::RNTupleMetrics::Print(std::ostream &output, const std::string &prefix) const
{
   if (!fIsEnabled) {
      output << fName << " metrics disabled!\n";
      return;
   }

   for (const auto &c : fCounters) {
      output << prefix << fName << kNamespaceSeperator << c->ToString() << '\n';
   }
   for (const auto c : fObservedMetrics) {
      c->Print(output, prefix + fName + ".");
   }
}

void ROOT::Experimental::Detail::RNTupleMetrics::Enable()
{
   for (auto &c: fCounters)
      c->Enable();
   fIsEnabled = true;
   for (auto m: fObservedMetrics)
      m->Enable();
}

void ROOT::Experimental::Detail::RNTupleMetrics::ObserveMetrics(RNTupleMetrics &observee)
{
   fObservedMetrics.push_back(&observee);
}

std::string ROOT::Experimental::Detail::RNTupleMetrics::GetMetricsExportPath()
{
   if (gSuppressMetricsExport)
      return {};

   if (const char *env = gSystem->Getenv("ROOT_EXPERIMENTAL_EXPORT_RNTUPLE_METRICS"); env && *env)
      return env;

   return {};
}

ROOT::Experimental::Detail::RNTupleMetrics::~RNTupleMetrics()
{
   if (R__unlikely(fHasAttemptedToExport)) {
      R__LOG_INFO(ROOT::Internal::NTupleLog()) << "metrics export was already attempted: not retrying";
      return;
   }

   if (!fIsEnabled) {
      R__LOG_INFO(ROOT::Internal::NTupleLog()) << "metrics disabled";
      return;
   }

   if (fNTupleName.empty()) {
      R__LOG_INFO(ROOT::Internal::NTupleLog()) << "no ntuple name set";
      return;
   }

   ExportToRootFile();

   fHasAttemptedToExport = true;
}

void ROOT::Experimental::Detail::RNTupleMetrics::CollectCounters(
   std::vector<std::pair<std::string, const RNTuplePerfCounter *>> &counters) const
{
   for (const auto &counter : fCounters)
      counters.emplace_back(fName, counter.get());
   for (const auto *observed : fObservedMetrics)
      observed->CollectCounters(counters);
}

void ROOT::Experimental::Detail::RNTupleMetrics::ExportToRootFile()
{
   if (fExportPath.empty()) {
      R__LOG_INFO(ROOT::Internal::NTupleLog()) << "no export path set";
      return;
   }

   std::vector<std::pair<std::string, const RNTuplePerfCounter *>> counters;
   CollectCounters(counters);

   if (R__unlikely(counters.empty())) {
      R__LOG_INFO(ROOT::Internal::NTupleLog()) << "no counters to export for '" << fNTupleName << "'";
      return;
   }

   // Avoid inifinite recursion (an RNTupleWriter is created to print the metrics)
   RSuppressMetaMetrics suppressGuard;

   auto model = ROOT::RNTupleModel::Create();
   for (const auto &[componentName, counter] : counters) {
      const std::string fieldName = componentName + "_" + counter->GetName();
      if (const auto *calc = dynamic_cast<const RNTupleCalcPerf *>(counter))
         *model->MakeField<double>(fieldName) = calc->GetValue();
      else
         *model->MakeField<std::int64_t>(fieldName) = counter->GetValueAsInt();
   }

   TMemFile memoryFile(fNTupleName.c_str(), "RECREATE");
   {
      auto writer = ROOT::RNTupleWriter::Append(std::move(model), fNTupleName, memoryFile);
      writer->Fill();
      writer->CommitDataset();
   }

   std::lock_guard<std::mutex> lock(GetMetricsExportMutex());

   TFileMerger merger;
   merger.SetMergeOptions(std::string_view("rntuple.MergingMode=Union"));

   if (!merger.OutputFile(fExportPath.c_str(), "UPDATE")) {
      R__LOG_ERROR(ROOT::Internal::NTupleLog()) << "cannot open metrics export file '" << fExportPath << "'";
      return;
   }

   if (!merger.AddFile(&memoryFile) || !merger.PartialMerge(TFileMerger::kAll | TFileMerger::kIncremental)) {
      R__LOG_ERROR(ROOT::Internal::NTupleLog())
         << "cannot merge metrics for '" << fNTupleName << "' into '" << fExportPath << "'";
      return;
   }
}
