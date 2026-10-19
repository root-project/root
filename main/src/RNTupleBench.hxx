// Author: Andreas Joachim Peters 2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RNTUPLE_BENCH_HXX
#define ROOT_RNTUPLE_BENCH_HXX

#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleMetrics.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleZip.hxx>
#include <ROOT/RPageStorage.hxx>

#include <TFile.h>
#include <TKey.h>
#include <TROOT.h>
#include <TSystem.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace RNTupleBench {

struct PageDecompResult {
   bool fValid = false;
   std::string fError;
   std::size_t fNumPages = 0;
   std::size_t fTotalCompBytes = 0;
   std::size_t fTotalUncompBytes = 0;
   double fTimePerPassSec = 0.0;
   double fDecompMBs = 0.0;
};

struct EventReaderResult {
   bool fValid = false;
   std::string fError;
   int fThreads = 1;
   std::uint64_t fNEntries = 0;
   double fReadWallSec = 0.0;
   double fTotalMBs = 0.0;
   double fEntriesSec = 0.0;
   double fDecompWallSec = 0.0;
   double fDecompCpuSec = 0.0;
   double fDecompMBs = 0.0;
   long long fPages = 0;
   double fUncompMB = 0.0;
};

struct BenchmarkResult {
   std::string fLabel;
   std::string fPath;
   std::uint64_t fFileSizeBytes = 0;
   PageDecompResult fPageDecomp;
   EventReaderResult fSingleCore;
   EventReaderResult fMultiCore;
};

inline std::string Basename(std::string path)
{
   const auto slash = path.find_last_of("/\\");
   if (slash != std::string::npos)
      path.erase(0, slash + 1);
   const auto dot = path.rfind('.');
   if (dot != std::string::npos)
      path.erase(dot);
   return path;
}

inline std::string FindFirstNTupleName(const std::string &path, const std::string &preferred = "")
{
   auto file = std::unique_ptr<TFile>(TFile::Open(path.c_str(), "READ"));
   if (!file || file->IsZombie())
      return "";

   if (!preferred.empty()) {
      if (file->Get(preferred.c_str()))
         return preferred;
   }

   TIter next(file->GetListOfKeys());
   while (auto key = static_cast<TKey *>(next())) {
      if (std::strcmp(key->GetClassName(), "ROOT::RNTuple") == 0)
         return key->GetName();
   }
   return "";
}

inline const ROOT::Experimental::Detail::RNTuplePerfCounter *FindCounter(
   const ROOT::Experimental::Detail::RNTupleMetrics &metrics, const std::string &name)
{
   if (auto *c = metrics.GetCounter("RNTupleReader.RPageStorage." + name))
      return c;
   if (auto *c = metrics.GetCounter("RNTupleReader.RPageSourceFile." + name))
      return c;
   if (auto *c = metrics.GetCounter("RPageStorage." + name))
      return c;
   if (auto *c = metrics.GetCounter("RPageSourceFile." + name))
      return c;
   if (auto *c = metrics.GetCounter(name))
      return c;
   if (auto *c = metrics.GetLocalCounter(name))
      return c;
   return nullptr;
}

// `percent` in (0, 100] keeps a prefix of compressed page bytes, in cluster order.
inline std::uint64_t ScaledCount(std::uint64_t total, double percent)
{
   if (total == 0 || !(percent < 100.0))
      return total;
   const auto n = static_cast<std::uint64_t>(std::ceil(static_cast<double>(total) * (percent / 100.0)));
   if (n == 0)
      return 1;
   return std::min(n, total);
}

inline PageDecompResult RunPageDecompBenchmark(const std::string &path, const std::string &ntupleName, int iters = 5,
                                              double percent = 100.0)
{
   PageDecompResult res;
   if (iters < 1)
      iters = 1;

   // 1. Open reader to attach and register compression profiles (e.g. LHC4 Crystal profile) in lhc4codec
   std::unique_ptr<ROOT::RNTupleReader> reader;
   try {
      reader = ROOT::RNTupleReader::Open(ntupleName, path);
   } catch (const std::exception &e) {
      res.fError = e.what();
      return res;
   }
   const auto &desc = reader->GetDescriptor();

   // 2. Open file directly to extract sealed pages into RAM (fast POSIX pread or TFile fallback)
   int fd = open(path.c_str(), O_RDONLY);
   std::unique_ptr<TFile> file;
   if (fd < 0) {
      file = std::unique_ptr<TFile>(TFile::Open(path.c_str(), "READ"));
      if (!file || file->IsZombie()) {
         res.fError = "Failed to open file: " + path;
         return res;
      }
   }

   struct PageRef {
      std::uint64_t fOffset = 0;
      std::size_t fComp = 0;
      std::size_t fUncomp = 0;
   };
   std::vector<PageRef> refs;
   std::uint64_t totalComp = 0;

   for (const auto &clusterDesc : desc.GetClusterIterable()) {
      for (const auto &colRange : clusterDesc.GetColumnRangeIterable()) {
         if (colRange.IsSuppressed())
            continue;
         auto colId = colRange.GetPhysicalColumnId();
         const auto &colDesc = desc.GetColumnDescriptor(colId);
         auto nBits = colDesc.GetBitsOnStorage();
         if (nBits == 0)
            continue;
         const auto &pageRange = clusterDesc.GetPageRange(colId);

         for (const auto &pageInfo : pageRange.GetPageInfos()) {
            auto loc = pageInfo.GetLocator();
            if (loc.GetType() == ROOT::RNTupleLocator::kTypePageZero)
               continue;
            auto compBytes = loc.GetNBytesOnStorage();
            auto uncompBytes = (static_cast<std::size_t>(nBits) * pageInfo.GetNElements() + 7) / 8;
            if (compBytes == 0 || uncompBytes == 0)
               continue;

            PageRef ref;
            ref.fOffset = loc.GetPosition<uint64_t>();
            ref.fComp = compBytes;
            ref.fUncomp = uncompBytes;
            totalComp += compBytes;
            refs.push_back(ref);
         }
      }
   }

   if (percent < 100.0) {
      const auto budget = ScaledCount(totalComp, percent);
      std::uint64_t acc = 0;
      std::size_t keep = 0;
      for (const auto &ref : refs) {
         acc += ref.fComp;
         ++keep;
         if (acc >= budget)
            break;
      }
      refs.resize(keep);
   }

   struct PageData {
      std::vector<char> fCompBuffer;
      std::size_t fUncompSize = 0;
   };
   std::vector<PageData> pages;
   std::size_t maxUncomp = 0;
   pages.reserve(refs.size());
   for (const auto &ref : refs) {
      PageData pd;
      pd.fCompBuffer.resize(ref.fComp);
      pd.fUncompSize = ref.fUncomp;
      if (fd >= 0) {
         ssize_t n = pread(fd, pd.fCompBuffer.data(), ref.fComp, ref.fOffset);
         if (n != static_cast<ssize_t>(ref.fComp)) {
            // Ignore read error; buffer will remain uncompressed or zeroed
         }
      } else if (file) {
         file->Seek(ref.fOffset);
         file->ReadBuffer(pd.fCompBuffer.data(), ref.fComp);
      }
      res.fTotalCompBytes += ref.fComp;
      res.fTotalUncompBytes += ref.fUncomp;
      if (ref.fUncomp > maxUncomp)
         maxUncomp = ref.fUncomp;
      pages.push_back(std::move(pd));
   }

   if (fd >= 0) {
      close(fd);
      fd = -1;
   }

   res.fNumPages = pages.size();
   if (pages.empty() || res.fTotalUncompBytes == 0) {
      res.fError = "No sealed pages found to decompress";
      return res;
   }

   std::vector<char> outBuf(maxUncomp * 2 + 65536);

   // Warm-up pass
   for (const auto &p : pages) {
      if (p.fCompBuffer.size() < p.fUncompSize) {
         try {
            ROOT::Internal::RNTupleDecompressor::Unzip(
               p.fCompBuffer.data(), p.fCompBuffer.size(), p.fUncompSize, outBuf.data());
         } catch (...) {
            // Ignore corrupted or uncompressed pages during raw decomp
         }
      } else {
         std::memcpy(outBuf.data(), p.fCompBuffer.data(), p.fUncompSize);
      }
   }

   // Timed passes
   auto t0 = std::chrono::high_resolution_clock::now();
   for (int iter = 0; iter < iters; ++iter) {
      for (const auto &p : pages) {
         if (p.fCompBuffer.size() < p.fUncompSize) {
            try {
               ROOT::Internal::RNTupleDecompressor::Unzip(
                  p.fCompBuffer.data(), p.fCompBuffer.size(), p.fUncompSize, outBuf.data());
            } catch (...) {
            }
         } else {
            std::memcpy(outBuf.data(), p.fCompBuffer.data(), p.fUncompSize);
         }
      }
   }
   auto t1 = std::chrono::high_resolution_clock::now();

   double totalSec = std::chrono::duration<double>(t1 - t0).count();
   res.fTimePerPassSec = totalSec / iters;
   double uncompMB = static_cast<double>(res.fTotalUncompBytes) / (1024.0 * 1024.0);
   res.fDecompMBs = res.fTimePerPassSec > 0.0 ? uncompMB / res.fTimePerPassSec : 0.0;
   res.fValid = true;
   return res;
}

inline EventReaderResult RunEventReaderBenchmark(const std::string &path, const std::string &ntupleName, int threads,
                                                 std::uint64_t maxEntries = 0, double percent = 100.0)
{
   EventReaderResult res;
   res.fThreads = threads;

   if (threads > 1) {
      ROOT::EnableImplicitMT(threads);
   } else if (ROOT::IsImplicitMTEnabled()) {
      ROOT::DisableImplicitMT();
   }

   try {
      ROOT::RNTupleReadOptions readOpts;
      readOpts.SetEnableMetrics(true);
      auto reader = ROOT::RNTupleReader::Open(ntupleName, path, readOpts);
      std::uint64_t totalEntries = reader->GetNEntries();
      std::uint64_t nEntriesToRead = totalEntries;
      if (maxEntries > 0 && maxEntries < totalEntries)
         nEntriesToRead = maxEntries;
      else if (percent < 100.0)
         nEntriesToRead = ScaledCount(totalEntries, percent);
      res.fNEntries = nEntriesToRead;

      auto t0 = std::chrono::high_resolution_clock::now();
      for (std::uint64_t entryId = 0; entryId < nEntriesToRead; ++entryId) {
         reader->LoadEntry(entryId);
      }
      auto t1 = std::chrono::high_resolution_clock::now();
      res.fReadWallSec = std::chrono::duration<double>(t1 - t0).count();

      const auto &metrics = reader->GetMetrics();
      auto *szUnzip = FindCounter(metrics, "szUnzip");
      auto *timeWallUnzip = FindCounter(metrics, "timeWallUnzip");
      auto *timeCpuUnzip = FindCounter(metrics, "timeCpuUnzip");
      auto *nPageUnsealed = FindCounter(metrics, "nPageUnsealed");

      res.fUncompMB = szUnzip ? szUnzip->GetValueAsInt() / (1024.0 * 1024.0) : 0.0;
      res.fTotalMBs = res.fReadWallSec > 0.0 ? res.fUncompMB / res.fReadWallSec : 0.0;
      res.fEntriesSec = res.fReadWallSec > 0.0 ? static_cast<double>(res.fNEntries) / res.fReadWallSec : 0.0;
      res.fDecompWallSec = timeWallUnzip ? timeWallUnzip->GetValueAsInt() / 1e9 : 0.0;
      res.fDecompCpuSec = timeCpuUnzip ? timeCpuUnzip->GetValueAsInt() / 1e9 : 0.0;
      res.fDecompMBs = res.fDecompWallSec > 0.0 ? res.fUncompMB / res.fDecompWallSec : 0.0;
      res.fPages = nPageUnsealed ? nPageUnsealed->GetValueAsInt() : 0;
      res.fValid = true;
   } catch (const std::exception &e) {
      res.fError = e.what();
   }

   if (ROOT::IsImplicitMTEnabled()) {
      ROOT::DisableImplicitMT();
   }
   return res;
}

inline BenchmarkResult RunBenchmarkForFile(const std::string &label, const std::string &path,
                                          const std::string &ntupleName, int multiThreads = 0, int iters = 5,
                                          std::uint64_t maxEntries = 0, double benchPercent = 100.0)
{
   if (multiThreads <= 0) {
      multiThreads = static_cast<int>(std::thread::hardware_concurrency());
      if (multiThreads <= 0)
         multiThreads = 8;
   }

   BenchmarkResult res;
   res.fLabel = label;
   res.fPath = path;

   FileStat_t st{};
   if (gSystem->GetPathInfo(path.c_str(), st) == 0) {
      res.fFileSizeBytes = static_cast<std::uint64_t>(st.fSize);
   }

   std::string actualName = ntupleName;
   if (actualName.empty()) {
      actualName = FindFirstNTupleName(path);
      if (actualName.empty()) {
         res.fPageDecomp.fError = "No RNTuple found in " + path;
         res.fSingleCore.fError = res.fPageDecomp.fError;
         res.fMultiCore.fError = res.fPageDecomp.fError;
         return res;
      }
   }

   std::cout << "Benchmarking " << label << " (" << path << ")";
   if (benchPercent < 100.0)
      std::cout << "  [first " << benchPercent << "% of pages and entries]";
   std::cout << "..." << std::endl;
   res.fPageDecomp = RunPageDecompBenchmark(path, actualName, iters, benchPercent);
   res.fSingleCore = RunEventReaderBenchmark(path, actualName, 1, maxEntries, benchPercent);
   res.fMultiCore = RunEventReaderBenchmark(path, actualName, multiThreads, maxEntries, benchPercent);
   return res;
}

inline std::string FormatMB(double mb)
{
   std::ostringstream os;
   os << std::fixed << std::setprecision(1) << mb << " MB";
   return os.str();
}

inline std::string FormatSpeed(double mbs)
{
   std::ostringstream os;
   os << std::fixed << std::setprecision(1) << mbs << " MB/s";
   return os.str();
}

inline std::string FormatTime(double sec)
{
   std::ostringstream os;
   os << std::fixed << std::setprecision(sec < 10.0 ? 3 : 2) << sec << " s";
   return os.str();
}

inline std::string FormatEntriesPerSec(double eps)
{
   std::uint64_t v = static_cast<std::uint64_t>(eps + 0.5);
   std::string s = std::to_string(v);
   std::string formatted;
   int n = 0;
   for (auto it = s.rbegin(); it != s.rend(); ++it) {
      if (n > 0 && n % 3 == 0)
         formatted.push_back(',');
      formatted.push_back(*it);
      ++n;
   }
   std::reverse(formatted.begin(), formatted.end());
   return formatted + " ent/s";
}

inline void PrintBenchmarkSummary(const std::vector<BenchmarkResult> &results)
{
   if (results.empty())
      return;

   constexpr int kLabelWidth = 26;

   // -------------------------------------------------------------------------
   // Table 1: Pure In-Memory Page Decompression (Algorithm Benchmark)
   // -------------------------------------------------------------------------
   std::cout << "\n=== Pure In-Memory Page Decompression Benchmark (Algorithm) ===\n";
   std::cout << std::left << std::setw(kLabelWidth) << "Variant / File"
             << std::right << std::setw(14) << "Comp Size"
             << std::setw(15) << "Uncomp Size"
             << std::setw(10) << "Pages"
             << std::setw(14) << "Pass Time"
             << std::setw(18) << "Decomp Throughput"
             << std::setw(14) << "vs Baseline"
             << '\n';
   std::cout << std::string(kLabelWidth + 14 + 15 + 10 + 14 + 18 + 14, '-') << '\n';

   double baseDecompMBs = 0.0;
   for (std::size_t i = 0; i < results.size(); ++i) {
      const auto &r = results[i];
      std::cout << std::left << std::setw(kLabelWidth) << r.fLabel << std::right;
      if (r.fPageDecomp.fValid) {
         double compMB = static_cast<double>(r.fPageDecomp.fTotalCompBytes) / (1024.0 * 1024.0);
         double uncompMB = static_cast<double>(r.fPageDecomp.fTotalUncompBytes) / (1024.0 * 1024.0);
         if (i == 0)
            baseDecompMBs = r.fPageDecomp.fDecompMBs;

         std::ostringstream ratioStr;
         if (baseDecompMBs > 0.0) {
            double ratio = r.fPageDecomp.fDecompMBs / baseDecompMBs;
            ratioStr << std::fixed << std::setprecision(2) << ratio << "x";
         } else {
            ratioStr << "baseline";
         }

         std::cout << std::setw(14) << FormatMB(compMB)
                   << std::setw(15) << FormatMB(uncompMB)
                   << std::setw(10) << r.fPageDecomp.fNumPages
                   << std::setw(14) << FormatTime(r.fPageDecomp.fTimePerPassSec)
                   << std::setw(18) << FormatSpeed(r.fPageDecomp.fDecompMBs)
                   << std::setw(14) << ratioStr.str();
      } else {
         std::cout << " [Error: " << r.fPageDecomp.fError << "]";
      }
      std::cout << '\n';
   }

   // -------------------------------------------------------------------------
   // Table 2: Single-Threaded Event Reader Benchmark (LoadEntry, 1 Thread)
   // -------------------------------------------------------------------------
   std::cout << "\n=== Single-Threaded RNTuple Event Reader (LoadEntry, 1 Thread) ===\n";
   std::cout << std::left << std::setw(kLabelWidth) << "Variant / File"
             << std::right << std::setw(13) << "Read Wall"
             << std::setw(15) << "Total MB/s"
             << std::setw(18) << "Entries / sec"
             << std::setw(14) << "Decomp Wall"
             << std::setw(18) << "Decomp MB/s"
             << std::setw(10) << "Pages"
             << '\n';
   std::cout << std::string(kLabelWidth + 13 + 15 + 18 + 14 + 18 + 10, '-') << '\n';

   for (const auto &r : results) {
      std::cout << std::left << std::setw(kLabelWidth) << r.fLabel << std::right;
      if (r.fSingleCore.fValid) {
         std::cout << std::setw(13) << FormatTime(r.fSingleCore.fReadWallSec)
                   << std::setw(15) << FormatSpeed(r.fSingleCore.fTotalMBs)
                   << std::setw(18) << FormatEntriesPerSec(r.fSingleCore.fEntriesSec)
                   << std::setw(14) << FormatTime(r.fSingleCore.fDecompWallSec)
                   << std::setw(18) << FormatSpeed(r.fSingleCore.fDecompMBs)
                   << std::setw(10) << r.fSingleCore.fPages;
      } else {
         std::cout << " [Error: " << r.fSingleCore.fError << "]";
      }
      std::cout << '\n';
   }

   // -------------------------------------------------------------------------
   // Table 3: Multi-Threaded Event Reader Benchmark (LoadEntry, N Threads)
   // -------------------------------------------------------------------------
   int mtThreads = results[0].fMultiCore.fThreads;
   std::cout << "\n=== Multi-Threaded RNTuple Event Reader (LoadEntry, " << mtThreads << " Threads IMT) ===\n";
   std::cout << std::left << std::setw(kLabelWidth) << "Variant / File"
             << std::right << std::setw(9) << "Threads"
             << std::setw(13) << "Read Wall"
             << std::setw(15) << "Total MB/s"
             << std::setw(18) << "Entries / sec"
             << std::setw(14) << "Decomp Wall"
             << std::setw(18) << "Decomp MB/s"
             << std::setw(12) << "Speedup"
             << '\n';
   std::cout << std::string(kLabelWidth + 9 + 13 + 15 + 18 + 14 + 18 + 12, '-') << '\n';

   for (const auto &r : results) {
      std::cout << std::left << std::setw(kLabelWidth) << r.fLabel << std::right;
      if (r.fMultiCore.fValid) {
         std::ostringstream speedupStr;
         if (r.fSingleCore.fValid && r.fMultiCore.fReadWallSec > 0.0) {
            double speedup = r.fSingleCore.fReadWallSec / r.fMultiCore.fReadWallSec;
            speedupStr << std::fixed << std::setprecision(2) << speedup << "x";
         } else {
            speedupStr << "n/a";
         }

         std::cout << std::setw(9) << r.fMultiCore.fThreads
                   << std::setw(13) << FormatTime(r.fMultiCore.fReadWallSec)
                   << std::setw(15) << FormatSpeed(r.fMultiCore.fTotalMBs)
                   << std::setw(18) << FormatEntriesPerSec(r.fMultiCore.fEntriesSec)
                   << std::setw(14) << FormatTime(r.fMultiCore.fDecompWallSec)
                   << std::setw(18) << FormatSpeed(r.fMultiCore.fDecompMBs)
                   << std::setw(12) << speedupStr.str();
      } else {
         std::cout << " [Error: " << r.fMultiCore.fError << "]";
      }
      std::cout << '\n';
   }

   std::cout << "\nNotes:\n";
   std::cout << "  - Algorithm benchmark measures pure decompression of the included sealed pages in RAM"
                " (timed passes do no I/O).\n";
   std::cout << "  - Single/Multi-core benchmarks measure RNTupleReader::LoadEntry over the included entries.\n";
   std::cout << "  - Speedup = Single-Core Read Wall / Multi-Core Read Wall (higher is better scaling).\n\n";
}

} // namespace RNTupleBench

#endif // ROOT_RNTUPLE_BENCH_HXX
