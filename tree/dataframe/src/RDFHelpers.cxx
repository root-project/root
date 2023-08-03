// Author: Stefan Wunsch, Enrico Guiraud CERN  09/2020

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDFHelpers.hxx"
#include "TROOT.h"      // IsImplicitMTEnabled
#include "TError.h"     // Warning
#include "TStopwatch.h"
#include "RConfigure.h" // R__USE_IMT
#include "ROOT/RLogger.hxx"
#include "ROOT/RDF/RLoopManager.hxx" // for RLoopManager
#include "ROOT/RDF/Utils.hxx"
#include "ROOT/RResultHandle.hxx"    // for RResultHandle, RunGraphs
#ifdef R__USE_IMT
#include "ROOT/TThreadExecutor.hxx"
#endif // R__USE_IMT

#include <algorithm>
#include <set>

// TODO, this function should be part of core libraries
#include <numeric>
#if (!defined(_WIN32)) && (!defined(_WIN64))
#include <unistd.h>
#endif

#include <stdio.h>
#include <iostream>

#if defined(_WIN32) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#include <io.h>
#include <Windows.h>
#elif defined(__linux__)
#include <sys/ioctl.h>
#endif // Windows/Linux

// Get terminal size for progress bar
int get_tty_size()
{
#if defined(_WIN32) || defined(_WIN64)
   int width = 0;
   CONSOLE_SCREEN_BUFFER_INFO csbi;
   GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
   width = (int)(csbi.srWindow.Right - csbi.srWindow.Left + 1);
   return width;
#elif defined(__linux__)
   int width = 0;
   struct winsize w;
   ioctl(fileno(stdout), TIOCGWINSZ, &w);
   width = (int)(w.ws_col);
   return width;
#else
   return 0;
#endif // Windows/Linux
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
      Warning("RunGraphs", "Got %lu handles from which %lu link to results which are already ready.", handles.size(),
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
      const auto effectiveVerbosity =
         ROOT::Experimental::Internal::GetChannelOrManager(ROOT::Detail::RDF::RDFLogChannel())
            .GetEffectiveVerbosity(ROOT::Experimental::RLogManager::Get());
      if (effectiveVerbosity >= ROOT::Experimental::ELogLevel::kDebug + 10) {
         // a very high verbosity was requested, let's not silence anything
         uniqueLoops[0].fLoopManager->Jit();
      } else {
         // silence logs from RLoopManager::Jit: RunGraphs does its own logging
         auto silenceRDFLogs = ROOT::Experimental::RLogScopedVerbosity(ROOT::Detail::RDF::RDFLogChannel(),
                                                                       ROOT::Experimental::ELogLevel::kError);
         uniqueLoops[0].fLoopManager->Jit();
      }
   }
   sw.Stop();
   R__LOG_INFO(ROOT::Detail::RDF::RDFLogChannel())
      << "Just-in-time compilation phase for RunGraphs (" << uniqueLoops.size()
      << " unique computation graphs) completed"
      << (sw.RealTime() > 1e-3 ? " in " + std::to_string(sw.RealTime()) + " seconds." : " in less than 1ms.");

   // Trigger the unique event loops
   auto run = [](RResultHandle &h) {
      if (h.fLoopManager)
         h.fLoopManager->Run(/*jit=*/false);
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

ROOT::RDF::Experimental::SnapshotPtr_t ROOT::RDF::Experimental::VariationsFor(ROOT::RDF::Experimental::SnapshotPtr_t)
{
   throw std::logic_error("Varying a Snapshot result is not implemented yet.");
}

namespace ROOT {
namespace RDF {

ProgressHelper::ProgressHelper(std::size_t increment, unsigned int totalFiles, unsigned int progressBarWidth,
                               unsigned int printInterval, bool useColors)
   : fPrintInterval(printInterval),
     fIncrement{increment},
     fBarWidth{progressBarWidth = int(get_tty_size() / 4)},
     fTotalFiles{totalFiles},
#if defined(_WIN32) || defined(_WIN64)
     fIsTTY{_isatty(_fileno(stdout)) == 1},
     fUseShellColours{false && useColors}
#else
     fIsTTY{isatty(fileno(stdout)) == 1},
     fUseShellColours{useColors && fIsTTY} // Control characters only with terminals.
#endif
{
}

/// Compute a running mean of events/s.
double ProgressHelper::EvtPerSec() const
{
   if (fEventsPerSecondStatisticsIndex < fEventsPerSecondStatistics.size())
      return std::accumulate(fEventsPerSecondStatistics.begin(),
                             fEventsPerSecondStatistics.begin() + fEventsPerSecondStatisticsIndex, 0.) /
             fEventsPerSecondStatisticsIndex;
   else
      return std::accumulate(fEventsPerSecondStatistics.begin(), fEventsPerSecondStatistics.end(), 0.) /
             fEventsPerSecondStatistics.size();
}

/// Record current event counts and time stamp, populate evts/s statistics array.
std::pair<std::size_t, std::chrono::seconds> ProgressHelper::RecordEvtCountAndTime()
{
   using namespace std::chrono;

   auto currentEventCount = fProcessedEvents.load();
   auto eventsPerTimeInterval = currentEventCount - fLastProcessedEvents;
   fLastProcessedEvents = currentEventCount;

   auto oldPrintTime = fLastPrintTime;
   auto newPrintTime = system_clock::now();
   fLastPrintTime = newPrintTime;

   duration<double> secondsCurrentInterval = newPrintTime - oldPrintTime;
   fEventsPerSecondStatistics[fEventsPerSecondStatisticsIndex++ % fEventsPerSecondStatistics.size()] =
      eventsPerTimeInterval / secondsCurrentInterval.count();

   return {currentEventCount, duration_cast<seconds>(newPrintTime - fBeginTime)};
}

namespace {
/// Format std::chrono::seconds as `1:30m`.
std::ostream &operator<<(std::ostream &stream, std::chrono::seconds elapsedSeconds)
{
   auto h = std::chrono::duration_cast<std::chrono::hours>(elapsedSeconds);
   auto m = std::chrono::duration_cast<std::chrono::minutes>(elapsedSeconds - h);
   auto s = (elapsedSeconds - h - m).count();
   if (h.count() > 0)
      stream << h.count() << ':' << std::setw(2) << std::right << std::setfill('0');
   stream << m.count() << ':' << std::setw(2) << std::right << std::setfill('0') << s;
   return stream << (h.count() > 0 ? 'h' : 'm');
}

struct RestoreStreamState {
   RestoreStreamState(std::ostream &stream) : fStream(stream), fFlags(stream.flags()), fFillChar(stream.fill()) {}
   ~RestoreStreamState()
   {
      fStream.setf(fFlags);
      fStream.fill(fFillChar);
   }

   std::ostream &fStream;
   std::ios_base::fmtflags fFlags;
   std::ostream::char_type fFillChar;
};
} // namespace

/// Print event and time statistics.
void ProgressHelper::PrintStats(std::ostream &stream, std::size_t currentEventCount,
                                std::chrono::seconds elapsedSeconds) const
{
   auto evtpersec = EvtPerSec();
   auto GetNEventsOfCurrentFile = ComputeNEventsSoFar();
   auto currentFileIdx = ComputeCurrentFileIdx();
   auto totalFiles = fTotalFiles;

   if (fUseShellColours)
      stream << "\e[35m";
   stream << "["
          << "Elapsed time: " << elapsedSeconds << "  ";
   if (fUseShellColours)
      stream << "\e[0m";
   stream << "processing file: " << currentFileIdx << " / " << totalFiles << "  ";

   // Event counts:
   if (fUseShellColours)
      stream << "\e[32m";

   stream << "processed evts: " << currentEventCount;
   if (GetNEventsOfCurrentFile != 0) {
      stream << " / " << std::scientific << std::setprecision(2) << GetNEventsOfCurrentFile;
   }
   stream << "  ";

   if (fUseShellColours)
      stream << "\e[0m";

   // events/s
   stream << std::scientific << std::setprecision(2) << evtpersec << " evt/s";

   // Time statistics:
   if (GetNEventsOfCurrentFile != 0) {
      if (fUseShellColours)
         stream << "\e[35m";
      std::chrono::seconds remainingSeconds(
         static_cast<long long>((ComputeNEventsSoFar() - currentEventCount) / evtpersec));
      stream << " " << remainingSeconds << " "
             << " remaining time (per file)";
      if (fUseShellColours)
         stream << "\e[0m";
   }

   stream << "]   ";
}

/// Print a progress bar of width `ProgressHelper::fBarWidth` if `fGetNEventsOfCurrentFile` is known.
void ProgressHelper::PrintProgressbar(std::ostream &stream, std::size_t currentEventCount) const
{
   auto GetNEventsOfCurrentFile = ComputeNEventsSoFar();
   if (GetNEventsOfCurrentFile == 0)
      return;

   RestoreStreamState restore(stream);

   double completion = double(currentEventCount) / GetNEventsOfCurrentFile;
   unsigned int nBar = std::min(completion, 1.) * fBarWidth;

   std::string bars(std::max(nBar, 1u), '=');
   bars.back() = (nBar == fBarWidth) ? '=' : '>';

   if (fUseShellColours)
      stream << "\e[33m";
   stream << '|' << std::setfill(' ') << std::setw(fBarWidth) << std::left << bars << "|   ";
   if (fUseShellColours)
      stream << "\e[0m";
}
//*/
namespace Experimental {

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

   void Finalize() { std::cout << '\n'; }

   std::string GetActionName() { return "ProgressBar"; }
   // dummy implementation of PartialUpdate
   int &PartialUpdate(unsigned int) { return *fDummyResult; }

   ROOT::RDF::SampleCallback_t GetSampleCallback()
   {
      return [this](unsigned int slot, const ROOT::RDF::RSampleInfo &id) {
         this->fHelper->registerNewSample(slot, id);
         return this->fHelper->ComputeNEventsSoFar();
      };
   }
};

void AddProgressbar(ROOT::RDF::RNode node)
{
   auto total_files = node.GetNFiles();
   auto progress = std::make_shared<ProgressHelper>(1000, total_files);
   ProgressBarAction c(progress);
   auto r = node.Book<>(c);
   r.OnPartialResultSlot(1000, [progress](unsigned int slot, auto &&arg) { (*progress)(slot, arg); });
}

void AddProgressbar(ROOT::RDataFrame dataframe)
{
   auto node = ROOT::RDF::AsRNode(dataframe);
   ROOT::RDF::Experimental::AddProgressbar(node);
}
} // namespace Experimental
} // namespace RDF
} // namespace ROOT
