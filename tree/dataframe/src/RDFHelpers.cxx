// Author: Stefan Wunsch, Enrico Guiraud CERN  09/2020

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDFHelpers.hxx"

#include "ROOT/RDF/RActionImpl.hxx"  // for RActionImpl
#include "ROOT/RDF/RFilterBase.hxx"  // for RDFInternal
#include "ROOT/RDF/RLoopManager.hxx" // for RLoopManager
#include "ROOT/RDF/Utils.hxx"
#include "ROOT/RResultHandle.hxx" // for RResultHandle, RunGraphs

#include "TROOT.h"      // IsImplicitMTEnabled
#include "TError.h"     // Warning
#include "TStopwatch.h"
#include "RConfigure.h" // R__USE_IMT
#include "ROOT/RLogger.hxx"
#include "ROOT/RSlotStack.hxx"
#ifdef R__USE_IMT
#include "ROOT/TThreadExecutor.hxx"
#endif // R__USE_IMT

#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <set>

#include <numeric>

#if defined(_WIN32) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#include <io.h>
#include <Windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

class TTreeReader;

namespace {
// Get terminal size for progress bar
// TODO: Put this in core libraries?
int get_tty_size()
{
#if defined(_WIN32) || defined(_WIN64)
   if (!_isatty(_fileno(stdout)))
      return 0;
   int width = 0;
   CONSOLE_SCREEN_BUFFER_INFO csbi;
   if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi))
      width = (int)(csbi.srWindow.Right - csbi.srWindow.Left + 1);
   return width;
#else
   int width = 0;
   struct winsize w;
   ioctl(fileno(stdout), TIOCGWINSZ, &w);
   width = (int)(w.ws_col);
   return width;
#endif
}

/// Restore an output stream to its previous state using RAII.
struct RestoreStreamState {
   RestoreStreamState(std::ostream &stream) : fStream(stream) { fPreservedState.copyfmt(stream); }
   ~RestoreStreamState() { fStream.copyfmt(fPreservedState); }

   std::ostream &fStream;
   std::ios fPreservedState{nullptr};
};
} // namespace

/// Format std::chrono::seconds as `1:30m`.
std::ostream &operator<<(std::ostream &stream, std::chrono::seconds elapsedSeconds)
{
   RestoreStreamState restore(stream);
   auto h = std::chrono::duration_cast<std::chrono::hours>(elapsedSeconds);
   auto m = std::chrono::duration_cast<std::chrono::minutes>(elapsedSeconds - h);
   auto s = (elapsedSeconds - h - m).count();

   if (h.count() > 0)
      stream << h.count() << ':' << std::setw(2) << std::right << std::setfill('0');
   stream << m.count() << ':' << std::setw(2) << std::right << std::setfill('0') << s;
   return stream << (h.count() > 0 ? 'h' : 'm');
}

using ROOT::RDF::RResultHandle;

unsigned int ROOT::RDF::RunGraphs(std::vector<RResultHandle> handles)
{
   if (handles.empty()) {
      Warning("RunGraphs", "Got an empty list of handles, now quitting.");
      return 0u;
   }

   // Check that there are results which have not yet been run
   const unsigned int nToRun =
      std::count_if(handles.begin(), handles.end(), [](const auto &h) { return !h.IsReady(); });
   if (nToRun < handles.size()) {
      Warning("RunGraphs", "Got %zu handles from which %zu link to results which are already ready.", handles.size(),
              handles.size() - nToRun);
   }
   if (nToRun == 0u)
      return 0u;

   // Find the unique event loops
   auto sameGraph = [](const RResultHandle &a, const RResultHandle &b) { return a.fLoopManager < b.fLoopManager; };
   std::set<RResultHandle, decltype(sameGraph)> s(handles.begin(), handles.end(), sameGraph);
   std::vector<RResultHandle> uniqueLoops(s.begin(), s.end());

   // Trigger jitting. One call is enough to jit the code required by all computation graphs.
   TStopwatch sw;
   sw.Start();
   {
      const auto effectiveVerbosity = ROOT::Internal::GetChannelOrManager(ROOT::Detail::RDF::RDFLogChannel())
                                         .GetEffectiveVerbosity(ROOT::RLogManager::Get());
      if (effectiveVerbosity >= ROOT::ELogLevel::kDebug + 10) {
         // a very high verbosity was requested, let's not silence anything
         uniqueLoops[0].fLoopManager->Jit();
      } else {
         // silence logs from RLoopManager::Jit: RunGraphs does its own logging
         auto silenceRDFLogs = ROOT::RLogScopedVerbosity(ROOT::Detail::RDF::RDFLogChannel(), ROOT::ELogLevel::kError);
         uniqueLoops[0].fLoopManager->Jit();
      }
   }
   sw.Stop();
   R__LOG_INFO(ROOT::Detail::RDF::RDFLogChannel())
      << "Just-in-time compilation phase for RunGraphs (" << uniqueLoops.size()
      << " unique computation graphs) completed"
      << (sw.RealTime() > 1e-3 ? " in " + std::to_string(sw.RealTime()) + " seconds." : " in less than 1ms.");

   // Trigger the unique event loops
   auto slotStack = std::make_shared<ROOT::Internal::RSlotStack>(ROOT::GetThreadPoolSize());
   auto run = [&slotStack](RResultHandle &h) {
      if (h.fLoopManager) {
         h.fLoopManager->SetSlotStack(slotStack);
         h.fLoopManager->Run(/*jit=*/false);
      }
   };

   sw.Start();
#ifdef R__USE_IMT
   if (ROOT::IsImplicitMTEnabled()) {
      ROOT::TThreadExecutor{}.Foreach(run, uniqueLoops);
   } else {
#endif
      std::for_each(uniqueLoops.begin(), uniqueLoops.end(), run);
#ifdef R__USE_IMT
   }
#endif
   sw.Stop();
   R__LOG_INFO(ROOT::Detail::RDF::RDFLogChannel())
      << "Finished RunGraphs run (" << uniqueLoops.size() << " unique computation graphs, " << sw.CpuTime() << "s CPU, "
      << sw.RealTime() << "s elapsed).";

   return uniqueLoops.size();
}

namespace ROOT::RDF::Experimental {

void ThreadsPerTH3(unsigned int N)
{
   ROOT::Internal::RDF::NThreadPerTH3() = N;
}

ProgressHelper::ProgressHelper(std::size_t increment, unsigned int totalFiles, unsigned int printInterval,
                               bool useColors)
   :
#if defined(_WIN32) || defined(_WIN64)
     fIsTTY{_isatty(_fileno(stdout)) != 0},
     fUseShellColours{false && useColors},
#else
     fIsTTY{isatty(fileno(stdout)) == 1},
     fUseShellColours{useColors && fIsTTY}, // Control characters only with terminals.
#endif
     fIncrement{increment},
     fNColumns(fIsTTY ? (get_tty_size() == 0 ? 60 : get_tty_size()) : 50),
     fTotalFiles{totalFiles},
     fPrintInterval{printInterval == 0 ? (fIsTTY ? 1 : 10) : printInterval}
{
   std::fill(fEventsPerSecondStatistics.begin(), fEventsPerSecondStatistics.end(), -1.);
}

/// Register a new sample for completion statistics.
/// \see ROOT::RDF::RInterface::DefinePerSample().
/// The *id.AsString()* refers to the name of the currently processed file.
/// The idea is to populate the  event entries in the *fSampleNameToEventEntries* map
/// by selecting the greater of the two values:
/// *id.EntryRange().second* which is the upper event entry range of the processed sample
/// and the current value of the event entries in the *fSampleNameToEventEntries* map.
/// In the single threaded case, the two numbers are the same as the entry range corresponds
/// to the number of events in an individual file (each sample is simply a single file).
/// In the multithreaded case, the idea is to accumulate the higher event entry value until
/// the total number of events in a given file is reached.
void ProgressHelper::RegisterNewSample(unsigned int /*slot*/, const ROOT::RDF::RSampleInfo &id)
{
   std::scoped_lock lock(fSampleNameToEventEntriesMutex);
   fSampleNameToEventEntries[id.AsString()] = std::max(id.NEntriesTotal(), fSampleNameToEventEntries[id.AsString()]);
}

/// Compute a running mean of events/s.
double ProgressHelper::EvtPerSec() const
{
   double sum = 0;
   unsigned int n = 0;
   for (auto item : fEventsPerSecondStatistics) {
      if (item >= 0.) {
         sum += item;
         ++n;
      }
   }

   return n > 0 ? sum / n : 0.;
}

/// Compute total events in all open files.
std::size_t ProgressHelper::ComputeTotalEvents() const
{
   std::scoped_lock lock(fSampleNameToEventEntriesMutex);
   std::size_t result = 0;
   for (const auto &item : fSampleNameToEventEntries)
      result += item.second;
   return result;
}

/// Record current event counts and time stamp, populate evts/s statistics array.
/// The function assumes that a lock on the update mutex is held.
std::pair<std::size_t, std::chrono::seconds> ProgressHelper::RecordEvtCountAndTime()
{
   using namespace std::chrono;

   auto currentEventCount = fProcessedEvents.load(std::memory_order_acquire);
   auto eventsPerTimeInterval = currentEventCount - fLastProcessedEvents;
   fLastProcessedEvents = currentEventCount;

   auto oldPrintTime = fLastPrintTime;
   auto newPrintTime = system_clock::now();
   fLastPrintTime = newPrintTime;

   duration<double> secondsCurrentInterval = newPrintTime - oldPrintTime;
   fEventsPerSecondStatisticsCounter = (fEventsPerSecondStatisticsCounter + 1) % fEventsPerSecondStatistics.size();
   fEventsPerSecondStatistics[fEventsPerSecondStatisticsCounter] =
      eventsPerTimeInterval / secondsCurrentInterval.count();

   return {currentEventCount, duration_cast<seconds>(newPrintTime - fBeginTime)};
}

/// Print event and time statistics.
void ProgressHelper::PrintProgressAndStats(std::ostream &stream, std::size_t currentEventCount,
                                           std::chrono::seconds elapsedSeconds) const
{
   std::ostringstream buffer;
   auto evtpersec = EvtPerSec();
   auto totalEventsInOpenFiles = ComputeTotalEvents();
   std::size_t currentFileIdx;
   {
      std::scoped_lock lock(fSampleNameToEventEntriesMutex);
      currentFileIdx = fSampleNameToEventEntries.size();
   }

   if (totalEventsInOpenFiles == 0)
      return;

   double completion = 0.;
   if (currentFileIdx < fTotalFiles) {
      const double fractionSeenFiles = double(currentFileIdx) / fTotalFiles;
      completion = fractionSeenFiles * (double(currentEventCount) / totalEventsInOpenFiles);
   } else {
      completion = double(currentEventCount) / totalEventsInOpenFiles;
   }

   // Print the bar
   {
      const auto barWidth = fNColumns / 4;
      unsigned int nBar = std::min(completion, 1.) * barWidth;
      std::string bars(std::max(nBar, 1u), '=');
      bars.back() = (nBar == barWidth) ? '=' : '>';

      if (fUseShellColours)
         buffer << "\033[33m";
      buffer << '|' << std::setfill(' ') << std::setw(barWidth) << std::left << bars << "|   ";
      if (fUseShellColours)
         buffer << "\033[0m";
   }

   // Elapsed time
   buffer << "[";
   if (fUseShellColours)
      buffer << "\033[35m";
   buffer << "Elapsed: " << elapsedSeconds.count() << "  ";
   if (fUseShellColours)
      buffer << "\033[0m";
   buffer << "files: " << currentFileIdx << " / " << fTotalFiles << "  ";

   // Event counts:
   if (fUseShellColours)
      buffer << "\033[32m";

   buffer << "events: " << currentEventCount;
   if (totalEventsInOpenFiles != 0) {
      buffer << " / " << std::scientific << std::setprecision(2);
      if (currentFileIdx == fTotalFiles)
         buffer << totalEventsInOpenFiles;
      else
         buffer << "(" << totalEventsInOpenFiles << " + x)";
   }
   buffer << "  ";

   if (fUseShellColours)
      buffer << "\033[0m";

   // events/s
   buffer << std::scientific << std::setprecision(2) << evtpersec << " evt/s";

   // Time statistics:
   // As long as not all files have been opened, estimate when 100% completion will be reached
   // based on current completion elapsed time. (This assumes that unopened files have about the
   // same size as the files that have been seen.)
   // Once the total event count is known, use "missing events / evt/s".
   if (totalEventsInOpenFiles != 0) {
      if (fUseShellColours)
         buffer << "\033[35m";

      std::chrono::seconds remainingSeconds;
      if (currentFileIdx == fTotalFiles) {
         remainingSeconds =
            std::chrono::seconds{static_cast<long long>((ComputeTotalEvents() - currentEventCount) / evtpersec)};
      } else {
         remainingSeconds =
            std::chrono::seconds{static_cast<long long>(elapsedSeconds.count() / completion - elapsedSeconds.count())};
      }
      buffer << "  remaining ca.: " << remainingSeconds.count();

      if (fUseShellColours)
         buffer << "\033[0m";
   }

   buffer << "]";

   RestoreStreamState restore(stream);
   stream << std::left << std::setw(fNColumns - 1) << buffer.str();
}

void ProgressHelper::PrintStatsFinal() const
{
   auto &stream = std::cout;
   RestoreStreamState restore(stream);
   const auto elapsedSeconds =
      std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - fBeginTime);
   const auto totalEvents = ComputeTotalEvents();

   // The next line resets the current line output in the terminal.
   // Brings the cursor at the beginning ('\r'), prints whitespace with the
   // same length as the terminal size, then resets the cursor again so we
   // can print the final stats on a clean line.
   if (fIsTTY)
      stream << '\r' << std::string(fNColumns, ' ') << '\r';

   if (fUseShellColours)
      stream << "\033[35m";
   stream << "["
          << "Total elapsed time: " << elapsedSeconds.count() << "  ";
   if (fUseShellColours)
      stream << "\033[0m";
   stream << "processed files: " << fTotalFiles << "  ";

   // Event counts:
   if (fUseShellColours)
      stream << "\033[32m";

   stream << "processed events: " << totalEvents;

   if (fUseShellColours)
      stream << "\033[0m";

   stream << "  " << std::scientific << std::setprecision(2) << (double)totalEvents / elapsedSeconds.count()
          << " evt/s";

   stream << "]\n";
}

/// Record number of events processed and update progress bar.
/// This function will atomically record elapsed times and event statistics, and one thread will udpate the progress bar
/// every n seconds (set by the fPrintInterval).
void ProgressHelper::Update()
{
   using namespace std::chrono;
   // ***************************************************
   // Warning: Here, everything needs to be thread safe:
   // ***************************************************
   fProcessedEvents.fetch_add(fIncrement, std::memory_order_relaxed);

   // We only print every n seconds.
   if (duration_cast<seconds>(system_clock::now() - fLastPrintTime) < fPrintInterval) {
      return;
   }

   // ******************************************************
   // Update the progress bar. Only one thread can proceed.
   // ******************************************************
   std::unique_lock<std::mutex> lockGuard(fUpdateMutex, std::try_to_lock);
   if (!lockGuard)
      return;

   auto const [eventCount, elapsedSeconds] = RecordEvtCountAndTime();

   if (fIsTTY)
      std::cout << "\r";

   PrintProgressAndStats(std::cout, eventCount, elapsedSeconds);

   if (fIsTTY)
      std::cout << std::flush;
   else
      std::cout << std::endl;
}

class ProgressBarAction final : public ROOT::Detail::RDF::RActionImpl<ProgressBarAction> {
public:
   using Result_t = int;

private:
   std::shared_ptr<ProgressHelper> fHelper;
   std::shared_ptr<int> fDummyResult = std::make_shared<int>();

public:
   ProgressBarAction(std::shared_ptr<ProgressHelper> r) : fHelper(std::move(r)) {}

   std::shared_ptr<Result_t> GetResultPtr() const { return fDummyResult; }

   void Initialize() {}
   void InitTask(TTreeReader *, unsigned int) {}

   void Exec(unsigned int) {}

   void Finalize() { fHelper->PrintStatsFinal(); }

   std::string GetActionName() { return "ProgressBar"; }
   // dummy implementation of PartialUpdate
   int &PartialUpdate(unsigned int) { return *fDummyResult; }

   ROOT::RDF::SampleCallback_t GetSampleCallback() final
   {
      return
         [this](unsigned int slot, const ROOT::RDF::RSampleInfo &id) { this->fHelper->RegisterNewSample(slot, id); };
   }
};

void AddProgressBar(ROOT::RDF::RNode node)
{
   constexpr std::size_t callbackEveryNEvents = 1000;
   auto total_files = node.GetNFiles();
   auto progress = std::make_shared<ProgressHelper>(callbackEveryNEvents, total_files);
   ProgressBarAction c(progress);
   auto r = node.Book<>(c);
   r.OnPartialResultSlot(callbackEveryNEvents, [progress](unsigned int slot, auto &&arg) { (*progress)(slot, arg); });
}

void AddProgressBar(ROOT::RDataFrame dataframe)
{
   auto node = ROOT::RDF::AsRNode(dataframe);
   ROOT::RDF::Experimental::AddProgressBar(node);
}

} // namespace ROOT::RDF::Experimental
