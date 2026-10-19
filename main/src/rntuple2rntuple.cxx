/// \file rntuple2rntuple.cxx
/// \brief Re-encode an RNTuple into several output files for compression studies.

#include <ROOT/RCluster.hxx>
#include <ROOT/RClusterPool.hxx>
#include <ROOT/RColumnElementBase.hxx>
#include <ROOT/RConfig.hxx>
#include <ROOT/RNTupleMerger.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReadOptions.hxx>
#include <ROOT/RNTupleUtils.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>
#include <ROOT/RPageAllocator.hxx>
#include <ROOT/RPageStorage.hxx>
#include <ROOT/RPageStorageFile.hxx>

#include "RNTupleBench.hxx"

#include <Compression.h>
#include <TFile.h>
#include <TKey.h>
#include <TROOT.h>
#include <TSystem.h>

#ifdef R__HAS_LHC4CODEC
#include <ZipLHC4.h>
#else
// Worker-result keys compile even when ROOT was built without lhc4codec.
enum {
   kLHC4CodecBeam = 0,
   kLHC4CodecLz = 0,
   kLHC4CodecBwt = 1,
   kLHC4CodecZstd = 2,
   kLHC4CodecBzip3 = 3,
   kLHC4CodecLzma = 4,
   kLHC4CodecAuto = 5,
   kLHC4CodecMosaic = 6,
   kLHC4CodecOracle = 7,
   kLHC4CodecCrystal = 8,
   kLHC4CodecFlash = 9,
   kLHC4CodecLaser = 10,
   kLHC4CodecGrating = 11,
   kLHC4CodecCount = 12,
};
#define kLHC4CodecBit(c) (1u << (unsigned)(c))
#define kLHC4AutoCodecsRoot \
   (kLHC4CodecBit(kLHC4CodecZstd) | kLHC4CodecBit(kLHC4CodecBeam) | kLHC4CodecBit(kLHC4CodecCrystal))
#define kLHC4AutoCodecsAll                                                                 \
   (kLHC4AutoCodecsRoot | kLHC4CodecBit(kLHC4CodecMosaic) | kLHC4CodecBit(kLHC4CodecBwt) | \
    kLHC4CodecBit(kLHC4CodecLaser) | kLHC4CodecBit(kLHC4CodecLzma) | kLHC4CodecBit(kLHC4CodecBzip3))
enum {
   kLHC4ProfileNone = 0,
   kLHC4ProfileFloatKinematics = 1,
   kLHC4ProfilePoissonCounts = 2,
   kLHC4ProfileIntMonotonic = 3,
   kLHC4ProfileText = 4,
   kLHC4ProfileCustom = 5,
};
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

using ROOT::Internal::MakeUninitArray;
using ROOT::Internal::RCluster;
using ROOT::Internal::RColumnElementBase;
using ROOT::Internal::RPageAllocatorHeap;
using ROOT::Internal::RPagePersistentSink;
using ROOT::Internal::RPageSink;
using ROOT::Internal::RPageSinkFile;
using ROOT::Internal::RPageSource;
using ROOT::Internal::RPageStorage;
using ROOT::Internal::RNTupleSerializer;
using ROOT::Experimental::Internal::RNTupleMerger;
using ROOT::Experimental::Internal::RNTupleMergeOptions;
using ROOT::RNTupleReadOptions;
using ROOT::RNTupleWriteOptions;

constexpr const char *kUsage = R"(Usage: rntuple2rntuple [options] input.root [ntupleName] [outputBase]

Options:
  --keep-column-encoding   Do not unpack ROOT RNTuple split/delta/zigzag encodings; recompress
                           the already-packed pages (same as rntupletryzip --keep-column-encoding).
                           Default for LHC4 variants is to unpack, then write plain columns.
  --page-size N            Set max unzipped page size in bytes (RNTupleWriteOptions default: 1 MiB).
                           Columns start with small pages and grow up to this limit.
  --initial-page-size N    Advanced: set the initial unzipped page size in bytes (default: 256 B).
                           Only needed for micro-benchmarks; large values require much more memory.
  --level N                Compression level for LHC4 variants (1-11, default 5). Native ZSTD/LZMA
                           stay at 5/7. Grating has no levels. lhc4_laser9 is the same as
                           --level 9 --only-variant lhc4_laser5.
  --auto-min-gain N        Min % size gain for lhc4codec Auto to prefer a slower decoder
                           (default: 1). 0 = always pick the smallest payload.
  --auto-max-level         For lhc4codec Auto, race each codec at its maximum effort
                           instead of the RNTuple compression level (slower encode).
  --auto-codecs LIST       Codecs Auto may race: root (default: zstd,beam,crystal),
                           all, or a comma list (zstd,beam,crystal,mosaic,prism,laser,lzma,bzip3).
  --profile NAME           On lhc4_crystal_profile5, force one profile on every column:
                           float, poisson, monotonic, text, or custom.
                           Default: that variant picks per column. lhc4_crystal5 stays dynamic.
  --profile-id N           Profile id (0 uses the --profile enum value; custom ids are >= 5).
  --bench                  Run decompression benchmarks (algorithm + single/multi-core event
                           reader) on output files after re-encoding.
  --bench-only             Do not re-encode; run the benchmark suite directly on the given
                           RNTuple file(s). Can take multiple files.
  --bench-threads N        Number of threads for multi-core event reader benchmark
                           (default: hardware concurrency).
  --bench-iters N          Number of passes for pure page decompression benchmark (default: 5).
  --bench-entries N        Number of entries to read in event reader benchmark (default: 0 = all).
  --bench-percent N        Probe the first N percent only (default: 100). Stops the page
                           unzip after that fraction of compressed bytes, in cluster
                           order, and LoadEntry after the same fraction of entries.
                           Example: --bench-percent 5. --bench-entries overrides the
                           entry count.
  --jobs N                 Run up to N variants in parallel (default: 1). Use 0 for auto
                           (number of CPU threads, capped by variant count). Requires fork().
  --quiet                  Suppress per-variant progress (used for parallel workers).
  --no-out                 Run conversions without keeping output files; print summary only.
  --only-variant LABEL     Run only the given format (see labels below).
  --worker-result PATH     Internal: write one-line key/value result metrics to PATH.
  -h, --help               Show this help.

Re-encode an RNTuple with different page compressors:

  <outputBase>_native.root       native RNTuple column encodings + ZSTD (level 5)
  <outputBase>_native_lzma.root  native RNTuple column encodings + LZMA (level 7)
  LHC4 variants use --level (default 5) in the file suffix (lhc4_beam5, lhc4_laser9, ...).
  <outputBase>_lhc4_zstd.root    lhc4codec filters + zstd
  <outputBase>_lhc4_bzip3.root   lhc4codec filters + bzip3
  <outputBase>_lhc4_beamN.root   lhc4codec filters + Beam (lhc4_lz5 is a legacy alias)
  <outputBase>_lhc4_mosaicN.root lhc4codec filters + Mosaic
  <outputBase>_lhc4_prismN.root  lhc4codec filters + Prism
  <outputBase>_lhc4_crystalN.root lhc4codec filters + Crystal (dynamic model and byte
                                 filters, same kind of compressor as Beam)
  <outputBase>_lhc4_crystal_profileN.root lhc4codec Crystal with pretrained per-column
                                 profiles (Real32=float, Index32/UInt32=poisson, Char=text)
  <outputBase>_lhc4_oracleN.root lhc4codec filters + Oracle (not raced by Auto)
  <outputBase>_lhc4_flashN.root  lhc4codec filters + Flash (byte-aligned LZ without an
                                 entropy stage, fastest decoder; not raced by Auto)
  <outputBase>_lhc4_laserN.root  lhc4codec filters + Laser (LZMA2-class range coder
                                 with a decoupled decoder, raced by --auto-codecs all)
  <outputBase>_lhc4_grating.root lhc4codec filters + Grating (order-0 integer rANS; has no
                                 levels; not raced by Auto)
  <outputBase>_lhc4_auto.root    lhc4codec filters + Auto codec race (see --auto-min-gain,
                                 --auto-max-level, --auto-codecs)

By default, LHC4 variants unpack ROOT column encodings (split/delta/zigzag) to
plain in-memory values, then re-encode with lhc4codec byte filters
(shuffle/delta/zigzag/dict/RLE as applicable). That matches tree2ntuple's
lhc4_* write path. Use --keep-column-encoding to recompress already-packed
pages without unpacking.

Parallel mode launches one process per format so lhc4codec global settings and
compression statistics stay isolated.

If ntupleName is omitted, the first RNTuple in the input file is used.
If outputBase is omitted, it defaults to "<input-stem>_recode".
)";

struct RunOptions {
   bool fKeepColumnEncoding = false;
   bool fNoOut = false;
   bool fQuiet = false;
   int fJobs = 1;
   std::optional<std::size_t> fPageSize;
   std::optional<std::size_t> fInitialPageSize;
   int fLevel = 5;
   bool fLevelExplicit = false;
   std::optional<int> fAutoMinGainPct;
   bool fAutoMaxLevel = false;
   std::optional<unsigned> fAutoCodecs;
   int fProfile = kLHC4ProfileNone;
   unsigned fProfileId = 0;
   std::string fOnlyVariant;
   std::string fWorkerResult;
   std::string fInputFile;
   std::string fNTupleName;
   std::string fOutputBase;

   bool fBench = false;
   bool fBenchOnly = false;
   int fBenchThreads = 0;
   int fBenchIters = 5;
   std::uint64_t fBenchEntries = 0;
   double fBenchPercent = 100.0;
   std::vector<std::string> fPositionalArgs;
};

struct FilterStatsSummary {
   bool fValid = false;
   bool fPageRatioValid = false;
   double fPageRatio = 0.0;
   double fFilterOnPct = 0.0;
   double fFilterOffPct = 0.0;
   double fRawWonPct = 0.0;
   double fNoTransformPct = 0.0;
};

struct AutoCodecStatsSummary {
   bool fValid = false;
   std::uint64_t fSelections = 0;
   std::uint64_t fCodecHits[kLHC4CodecCount] = {};
};

struct VariantResult {
   std::string fLabel;
   std::uint64_t fOutputBytes = 0;
   double fElapsedSec = 0.0;
   FilterStatsSummary fFilters;
   AutoCodecStatsSummary fAutoCodec;
};

std::string BasenameNoExt(std::string path)
{
   const auto slash = path.find_last_of("/\\");
   if (slash != std::string::npos)
      path.erase(0, slash + 1);
   const auto dot = path.rfind('.');
   if (dot != std::string::npos)
      path.erase(dot);
   return path;
}

std::string JoinPath(const std::string &base, const std::string &suffix)
{
   return base + suffix + ".root";
}

void RemoveIfExists(const std::string &path)
{
   if (!gSystem->AccessPathName(path.c_str()))
      gSystem->Unlink(path.c_str());
}

std::string MakeTempRootPath()
{
   TString tmp;
   FILE *f = gSystem->TempFileName(tmp, nullptr, ".root");
   if (!f)
      throw std::runtime_error("failed to create temporary output file");
   std::fclose(f);
   return tmp.Data();
}

std::string ResolveVariantOutputPath(const RunOptions &opts, const std::string &suffix)
{
   if (opts.fNoOut)
      return MakeTempRootPath();
   return JoinPath(opts.fOutputBase, suffix);
}

std::optional<std::uint64_t> FileSizeBytes(const std::string &path)
{
   FileStat_t st{};
   if (gSystem->GetPathInfo(path.c_str(), st) != 0)
      return std::nullopt;
   return static_cast<std::uint64_t>(st.fSize);
}

std::string FormatBytes(std::uint64_t nbytes)
{
   const char *suffix = "B";
   double v = static_cast<double>(nbytes);
   if (v >= 1024.0 * 1024.0 * 1024.0) {
      v /= 1024.0 * 1024.0 * 1024.0;
      suffix = "GiB";
   } else if (v >= 1024.0 * 1024.0) {
      v /= 1024.0 * 1024.0;
      suffix = "MiB";
   } else if (v >= 1024.0) {
      v /= 1024.0;
      suffix = "KiB";
   }
   std::ostringstream os;
   os.setf(std::ios::fixed);
   os.precision(v >= 100.0 ? 0 : (v >= 10.0 ? 1 : 2));
   os << v << ' ' << suffix;
   return os.str();
}

std::string FormatSeconds(double sec)
{
   std::ostringstream os;
   os.setf(std::ios::fixed);
   os.precision(sec >= 100.0 ? 0 : 1);
   os << sec << 's';
   return os.str();
}

std::string FormatRatio(std::uint64_t outputBytes, std::uint64_t inputBytes)
{
   if (inputBytes == 0)
      return "n/a";
   std::ostringstream os;
   os.setf(std::ios::fixed);
   os.precision(3);
   os << static_cast<double>(outputBytes) / static_cast<double>(inputBytes);
   return os.str();
}

std::string FormatPageRatio(const FilterStatsSummary &filters)
{
   if (!filters.fPageRatioValid)
      return "n/a";
   std::ostringstream os;
   os.setf(std::ios::fixed);
   os.precision(3);
   os << filters.fPageRatio;
   return os.str();
}

std::string FormatPct(double pct)
{
   std::ostringstream os;
   os.setf(std::ios::fixed);
   os.precision(1);
   os << pct << '%';
   return os.str();
}

bool ParseAutoCodecName(const std::string &name, unsigned &bit)
{
   if (name == "beam" || name == "lz") {
      bit = kLHC4CodecBit(kLHC4CodecBeam);
      return true;
   }
   if (name == "bwt" || name == "prism") {
      bit = kLHC4CodecBit(kLHC4CodecBwt);
      return true;
   }
   if (name == "zstd") {
      bit = kLHC4CodecBit(kLHC4CodecZstd);
      return true;
   }
   if (name == "bzip3") {
      bit = kLHC4CodecBit(kLHC4CodecBzip3);
      return true;
   }
   if (name == "lzma" || name == "xz") {
      bit = kLHC4CodecBit(kLHC4CodecLzma);
      return true;
   }
   if (name == "mosaic") {
      bit = kLHC4CodecBit(kLHC4CodecMosaic);
      return true;
   }
   if (name == "crystal") {
      bit = kLHC4CodecBit(kLHC4CodecCrystal);
      return true;
   }
   if (name == "laser") {
      bit = kLHC4CodecBit(kLHC4CodecLaser);
      return true;
   }
   // Flash and Grating are forced-only backends; Auto never races them.
   return false;
}

unsigned ParseAutoCodecMask(const std::string &list)
{
   if (list == "all")
      return kLHC4AutoCodecsAll;
   if (list == "root" || list.empty())
      return kLHC4AutoCodecsRoot;

   unsigned mask = 0;
   std::size_t pos = 0;
   while (pos <= list.size()) {
      const auto end = list.find(',', pos);
      const auto tok = list.substr(pos, (end == std::string::npos ? list.size() : end) - pos);
      if (!tok.empty()) {
         unsigned bit = 0;
         if (!ParseAutoCodecName(tok, bit))
            throw std::runtime_error("unknown or non-raceable --auto-codecs token: " + tok);
         mask |= bit;
      }
      if (end == std::string::npos)
         break;
      pos = end + 1;
   }
   if (mask == 0)
      throw std::runtime_error("--auto-codecs list is empty");
   return mask;
}

int ParseProfileName(const std::string &name)
{
   if (name == "none" || name == "off" || name == "dynamic")
      return kLHC4ProfileNone;
   if (name == "float" || name == "float-kinematics" || name == "kinematics")
      return kLHC4ProfileFloatKinematics;
   if (name == "poisson" || name == "poisson-counts" || name == "counts")
      return kLHC4ProfilePoissonCounts;
   if (name == "monotonic" || name == "int-monotonic" || name == "int")
      return kLHC4ProfileIntMonotonic;
   if (name == "text" || name == "json")
      return kLHC4ProfileText;
   if (name == "custom")
      return kLHC4ProfileCustom;
   throw std::runtime_error("unknown --profile: " + name + " (none, float, poisson, monotonic, text, custom)");
}

const char *FormatProfileName(int profile)
{
   switch (profile) {
   case kLHC4ProfileFloatKinematics: return "float";
   case kLHC4ProfilePoissonCounts: return "poisson";
   case kLHC4ProfileIntMonotonic: return "monotonic";
   case kLHC4ProfileText: return "text";
   case kLHC4ProfileCustom: return "custom";
   default: return "none";
   }
}

std::string FormatCrystalProfileStatus(const RunOptions &opts)
{
   std::string profileFlavour = "per column";
   if (opts.fProfile != kLHC4ProfileNone) {
      profileFlavour = FormatProfileName(opts.fProfile);
      if (opts.fProfileId != 0)
         profileFlavour += " (id " + std::to_string(opts.fProfileId) + ")";
      profileFlavour += " (all columns)";
   }
   return "lhc4_crystal5 dynamic; lhc4_crystal_profile5 " + profileFlavour;
}

std::string FormatAutoCodecMask(unsigned mask)
{
   if (mask == kLHC4AutoCodecsRoot)
      return "root (zstd,beam,crystal)";
   if (mask == kLHC4AutoCodecsAll)
      return "all";

   struct Row {
      const char *fName;
      unsigned fBit;
   };
   static constexpr Row kRows[] = {{"zstd", kLHC4CodecBit(kLHC4CodecZstd)},
                                   {"beam", kLHC4CodecBit(kLHC4CodecBeam)},
                                   {"crystal", kLHC4CodecBit(kLHC4CodecCrystal)},
                                   {"mosaic", kLHC4CodecBit(kLHC4CodecMosaic)},
                                   {"prism", kLHC4CodecBit(kLHC4CodecBwt)},
                                   {"laser", kLHC4CodecBit(kLHC4CodecLaser)},
                                   {"lzma", kLHC4CodecBit(kLHC4CodecLzma)},
                                   {"bzip3", kLHC4CodecBit(kLHC4CodecBzip3)}};
   std::string out;
   for (const auto &row : kRows) {
      if (!(mask & row.fBit))
         continue;
      if (!out.empty())
         out += ',';
      out += row.fName;
   }
   return out.empty() ? "none" : out;
}

std::string FindFirstRNTupleName(const std::string &fileName)
{
   auto file = std::unique_ptr<TFile>(TFile::Open(fileName.c_str()));
   if (!file || file->IsZombie())
      throw std::runtime_error("cannot open input file: " + fileName);

   TIter next(file->GetListOfKeys());
   while (auto key = static_cast<TKey *>(next())) {
      if (std::strcmp(key->GetClassName(), "ROOT::RNTuple") == 0)
         return key->GetName();
   }
   throw std::runtime_error("no RNTuple found in " + fileName);
}

bool ParseBoolValue(const std::string &value)
{
   return value == "1" || value == "true" || value == "yes";
}

void WriteWorkerResult(const std::string &path, const VariantResult &result)
{
   std::ofstream out(path);
   if (!out)
      throw std::runtime_error("cannot write worker result: " + path);
   out << "label=" << result.fLabel << '\n';
   out << "output_bytes=" << result.fOutputBytes << '\n';
   out << "elapsed_sec=" << std::setprecision(10) << result.fElapsedSec << '\n';
   out << "filter_valid=" << (result.fFilters.fValid ? 1 : 0) << '\n';
   out << "filter_on_pct=" << result.fFilters.fFilterOnPct << '\n';
   out << "filter_off_pct=" << result.fFilters.fFilterOffPct << '\n';
   out << "raw_won_pct=" << result.fFilters.fRawWonPct << '\n';
   out << "no_transform_pct=" << result.fFilters.fNoTransformPct << '\n';
   out << "page_ratio_valid=" << (result.fFilters.fPageRatioValid ? 1 : 0) << '\n';
   out << "page_ratio=" << std::setprecision(10) << result.fFilters.fPageRatio << '\n';
   out << "auto_valid=" << (result.fAutoCodec.fValid ? 1 : 0) << '\n';
   out << "auto_selections=" << result.fAutoCodec.fSelections << '\n';
   out << "auto_zstd=" << result.fAutoCodec.fCodecHits[kLHC4CodecZstd] << '\n';
   out << "auto_beam=" << result.fAutoCodec.fCodecHits[kLHC4CodecBeam] << '\n';
   out << "auto_lz=" << result.fAutoCodec.fCodecHits[kLHC4CodecBeam] << '\n';
   out << "auto_mosaic=" << result.fAutoCodec.fCodecHits[kLHC4CodecMosaic] << '\n';
   out << "auto_bwt=" << result.fAutoCodec.fCodecHits[kLHC4CodecBwt] << '\n';
   out << "auto_lzma=" << result.fAutoCodec.fCodecHits[kLHC4CodecLzma] << '\n';
   out << "auto_bzip3=" << result.fAutoCodec.fCodecHits[kLHC4CodecBzip3] << '\n';
   out << "auto_crystal=" << result.fAutoCodec.fCodecHits[kLHC4CodecCrystal] << '\n';
   out << "auto_laser=" << result.fAutoCodec.fCodecHits[kLHC4CodecLaser] << '\n';
}

VariantResult ReadWorkerResult(const std::string &path)
{
   std::ifstream in(path);
   if (!in)
      throw std::runtime_error("cannot read worker result: " + path);

   VariantResult result;
   std::string line;
   while (std::getline(in, line)) {
      const auto eq = line.find('=');
      if (eq == std::string::npos)
         continue;
      const std::string key = line.substr(0, eq);
      const std::string value = line.substr(eq + 1);
      if (key == "label")
         result.fLabel = value;
      else if (key == "output_bytes")
         result.fOutputBytes = std::stoull(value);
      else if (key == "elapsed_sec")
         result.fElapsedSec = std::stod(value);
      else if (key == "filter_valid")
         result.fFilters.fValid = ParseBoolValue(value);
      else if (key == "filter_on_pct")
         result.fFilters.fFilterOnPct = std::stod(value);
      else if (key == "filter_off_pct")
         result.fFilters.fFilterOffPct = std::stod(value);
      else if (key == "raw_won_pct")
         result.fFilters.fRawWonPct = std::stod(value);
      else if (key == "no_transform_pct")
         result.fFilters.fNoTransformPct = std::stod(value);
      else if (key == "page_ratio_valid")
         result.fFilters.fPageRatioValid = ParseBoolValue(value);
      else if (key == "page_ratio")
         result.fFilters.fPageRatio = std::stod(value);
      else if (key == "auto_valid")
         result.fAutoCodec.fValid = ParseBoolValue(value);
      else if (key == "auto_selections")
         result.fAutoCodec.fSelections = std::stoull(value);
      else if (key == "auto_zstd")
         result.fAutoCodec.fCodecHits[kLHC4CodecZstd] = std::stoull(value);
      else if (key == "auto_beam" || key == "auto_lz")
         result.fAutoCodec.fCodecHits[kLHC4CodecBeam] = std::stoull(value);
      else if (key == "auto_mosaic")
         result.fAutoCodec.fCodecHits[kLHC4CodecMosaic] = std::stoull(value);
      else if (key == "auto_bwt")
         result.fAutoCodec.fCodecHits[kLHC4CodecBwt] = std::stoull(value);
      else if (key == "auto_lzma")
         result.fAutoCodec.fCodecHits[kLHC4CodecLzma] = std::stoull(value);
      else if (key == "auto_bzip3")
         result.fAutoCodec.fCodecHits[kLHC4CodecBzip3] = std::stoull(value);
      else if (key == "auto_crystal")
         result.fAutoCodec.fCodecHits[kLHC4CodecCrystal] = std::stoull(value);
      else if (key == "auto_laser")
         result.fAutoCodec.fCodecHits[kLHC4CodecLaser] = std::stoull(value);
   }
   return result;
}

constexpr int kDefaultLHC4Level = 5;
constexpr int kMaxLHC4Level = 11; // lhc4codec::kMaxLevel

int ParseLHC4Level(const std::string &text)
{
   int level = 0;
   try {
      std::size_t idx = 0;
      level = std::stoi(text, &idx);
      if (idx != text.size())
         throw std::invalid_argument("trailing characters");
   } catch (const std::exception &) {
      throw std::runtime_error("--level expects an integer from 1 to " + std::to_string(kMaxLHC4Level));
   }
   if (level < 1 || level > kMaxLHC4Level)
      throw std::runtime_error("--level must be between 1 and " + std::to_string(kMaxLHC4Level));
   return level;
}

int TrailingVariantLevel(const std::string &label)
{
   if (label.empty() || !std::isdigit(static_cast<unsigned char>(label.back())))
      return 0;
   std::size_t i = label.size();
   while (i > 0 && std::isdigit(static_cast<unsigned char>(label[i - 1])))
      --i;
   if (i == 0 || i == label.size())
      return 0;
   try {
      std::size_t idx = 0;
      const std::string digits = label.substr(i);
      const int n = std::stoi(digits, &idx);
      if (idx != digits.size() || n < 1 || n > kMaxLHC4Level)
         return 0;
      return n;
   } catch (const std::exception &) {
      return 0;
   }
}

std::string TrimCopy(const std::string &text)
{
   const auto start = text.find_first_not_of(" \t");
   const auto end = text.find_last_not_of(" \t");
   if (start == std::string::npos || end == std::string::npos)
      return {};
   return text.substr(start, end - start + 1);
}

void InferLevelFromVariantLabels(RunOptions &opts)
{
   if (opts.fLevelExplicit || opts.fOnlyVariant.empty())
      return;
   std::optional<int> inferred;
   std::stringstream ss(opts.fOnlyVariant);
   std::string item;
   while (std::getline(ss, item, ',')) {
      item = TrimCopy(item);
      if (item.empty())
         continue;
      const int n = TrailingVariantLevel(item);
      if (n == 0)
         continue;
      if (!inferred)
         inferred = n;
      else if (*inferred != n)
         throw std::runtime_error("conflicting levels in --only-variant; pass --level N");
   }
   if (inferred)
      opts.fLevel = *inferred;
}

RunOptions ParseArgs(int argc, char **argv)
{
   RunOptions opts;
   std::vector<std::string> positional;

   for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "-h" || arg == "--help") {
         std::cout << kUsage;
         std::exit(0);
      }
      if (arg == "--keep-column-encoding") {
         opts.fKeepColumnEncoding = true;
         continue;
      }
      if (arg == "--page-size") {
         if (i + 1 >= argc)
            throw std::runtime_error("--page-size requires an argument");
         opts.fPageSize = static_cast<std::size_t>(std::stoull(argv[++i]));
         continue;
      }
      if (arg.rfind("--page-size=", 0) == 0) {
         opts.fPageSize = static_cast<std::size_t>(std::stoull(arg.substr(12)));
         continue;
      }
      if (arg == "--initial-page-size") {
         if (i + 1 >= argc)
            throw std::runtime_error("--initial-page-size requires an argument");
         opts.fInitialPageSize = static_cast<std::size_t>(std::stoull(argv[++i]));
         continue;
      }
      if (arg.rfind("--initial-page-size=", 0) == 0) {
         opts.fInitialPageSize = static_cast<std::size_t>(std::stoull(arg.substr(20)));
         continue;
      }
      if (arg == "--level") {
         if (i + 1 >= argc)
            throw std::runtime_error("--level requires an argument");
         opts.fLevel = ParseLHC4Level(argv[++i]);
         opts.fLevelExplicit = true;
         continue;
      }
      if (arg.rfind("--level=", 0) == 0) {
         opts.fLevel = ParseLHC4Level(arg.substr(8));
         opts.fLevelExplicit = true;
         continue;
      }
      if (arg == "--auto-min-gain") {
         if (i + 1 >= argc)
            throw std::runtime_error("--auto-min-gain requires an argument");
         opts.fAutoMinGainPct = std::stoi(argv[++i]);
         continue;
      }
      if (arg.rfind("--auto-min-gain=", 0) == 0) {
         opts.fAutoMinGainPct = std::stoi(arg.substr(16));
         continue;
      }
      if (arg == "--auto-max-level") {
         opts.fAutoMaxLevel = true;
         continue;
      }
      if (arg == "--auto-codecs") {
         if (i + 1 >= argc)
            throw std::runtime_error("--auto-codecs requires an argument");
         opts.fAutoCodecs = ParseAutoCodecMask(argv[++i]);
         continue;
      }
      if (arg.rfind("--auto-codecs=", 0) == 0) {
         opts.fAutoCodecs = ParseAutoCodecMask(arg.substr(14));
         continue;
      }
      if (arg == "--profile") {
         if (i + 1 >= argc)
            throw std::runtime_error("--profile requires an argument");
         opts.fProfile = ParseProfileName(argv[++i]);
         continue;
      }
      if (arg.rfind("--profile=", 0) == 0) {
         opts.fProfile = ParseProfileName(arg.substr(10));
         continue;
      }
      if (arg == "--profile-id") {
         if (i + 1 >= argc)
            throw std::runtime_error("--profile-id requires an argument");
         opts.fProfileId = static_cast<unsigned>(std::stoul(argv[++i]));
         continue;
      }
      if (arg.rfind("--profile-id=", 0) == 0) {
         opts.fProfileId = static_cast<unsigned>(std::stoul(arg.substr(13)));
         continue;
      }
      if (arg == "--bench") {
         opts.fBench = true;
         continue;
      }
      if (arg == "--bench-only") {
         opts.fBench = true;
         opts.fBenchOnly = true;
         continue;
      }
      if (arg == "--bench-threads") {
         if (i + 1 >= argc)
            throw std::runtime_error("--bench-threads requires an argument");
         opts.fBenchThreads = std::stoi(argv[++i]);
         continue;
      }
      if (arg.rfind("--bench-threads=", 0) == 0) {
         opts.fBenchThreads = std::stoi(arg.substr(16));
         continue;
      }
      if (arg == "--bench-iters") {
         if (i + 1 >= argc)
            throw std::runtime_error("--bench-iters requires an argument");
         opts.fBenchIters = std::stoi(argv[++i]);
         continue;
      }
      if (arg.rfind("--bench-iters=", 0) == 0) {
         opts.fBenchIters = std::stoi(arg.substr(14));
         continue;
      }
      if (arg == "--bench-entries") {
         if (i + 1 >= argc)
            throw std::runtime_error("--bench-entries requires an argument");
         opts.fBenchEntries = std::stoull(argv[++i]);
         continue;
      }
      if (arg.rfind("--bench-entries=", 0) == 0) {
         opts.fBenchEntries = std::stoull(arg.substr(16));
         continue;
      }
      if (arg == "--bench-percent") {
         if (i + 1 >= argc)
            throw std::runtime_error("--bench-percent requires an argument");
         opts.fBenchPercent = std::stod(argv[++i]);
         if (!(opts.fBenchPercent > 0.0 && opts.fBenchPercent <= 100.0))
            throw std::runtime_error("--bench-percent must be in (0, 100]");
         continue;
      }
      if (arg.rfind("--bench-percent=", 0) == 0) {
         opts.fBenchPercent = std::stod(arg.substr(16));
         if (!(opts.fBenchPercent > 0.0 && opts.fBenchPercent <= 100.0))
            throw std::runtime_error("--bench-percent must be in (0, 100]");
         continue;
      }
      if (arg == "--max-page-size") {
         if (i + 1 >= argc)
            throw std::runtime_error("--max-page-size requires an argument");
         opts.fPageSize = static_cast<std::size_t>(std::stoull(argv[++i]));
         continue;
      }
      if (arg.rfind("--max-page-size=", 0) == 0) {
         opts.fPageSize = static_cast<std::size_t>(std::stoull(arg.substr(16)));
         continue;
      }
      if (arg == "--quiet") {
         opts.fQuiet = true;
         continue;
      }
      if (arg == "--no-out") {
         opts.fNoOut = true;
         continue;
      }
      if (arg == "--jobs") {
         if (i + 1 >= argc)
            throw std::runtime_error("--jobs requires an argument");
         opts.fJobs = std::stoi(argv[++i]);
         continue;
      }
      if (arg.rfind("--jobs=", 0) == 0) {
         opts.fJobs = std::stoi(arg.substr(7));
         continue;
      }
      if (arg == "--only-variant") {
         if (i + 1 >= argc)
            throw std::runtime_error("--only-variant requires an argument");
         opts.fOnlyVariant = argv[++i];
         continue;
      }
      if (arg.rfind("--only-variant=", 0) == 0) {
         opts.fOnlyVariant = arg.substr(15);
         continue;
      }
      if (arg == "--worker-result") {
         if (i + 1 >= argc)
            throw std::runtime_error("--worker-result requires an argument");
         opts.fWorkerResult = argv[++i];
         continue;
      }
      if (arg.rfind("--worker-result=", 0) == 0) {
         opts.fWorkerResult = arg.substr(16);
         continue;
      }
      if (!arg.empty() && arg[0] == '-') {
         throw std::runtime_error("unknown option: " + arg);
      }
      positional.push_back(arg);
   }

   if (positional.empty())
      throw std::runtime_error("missing input file (see --help)");

   opts.fPositionalArgs = positional;

   if (opts.fBenchOnly) {
      opts.fInputFile = positional[0];
      return opts;
   }

   opts.fInputFile = positional[0];
   if (positional.size() >= 2)
      opts.fNTupleName = positional[1];
   else
      opts.fNTupleName = FindFirstRNTupleName(opts.fInputFile);

   if (positional.size() >= 3)
      opts.fOutputBase = positional[2];
   else
      opts.fOutputBase = BasenameNoExt(opts.fInputFile) + "_recode";

   if (positional.size() > 3)
      throw std::runtime_error("too many positional arguments (see --help)");

   if (opts.fPageSize && *opts.fPageSize == 0)
      throw std::runtime_error("--page-size must be > 0");
   if (opts.fInitialPageSize && *opts.fInitialPageSize == 0)
      throw std::runtime_error("--initial-page-size must be > 0");
   if (opts.fPageSize && opts.fInitialPageSize && *opts.fInitialPageSize > *opts.fPageSize)
      throw std::runtime_error("--initial-page-size must not exceed --page-size");
   if (opts.fAutoMinGainPct && *opts.fAutoMinGainPct < 0)
      throw std::runtime_error("--auto-min-gain must be >= 0");

   InferLevelFromVariantLabels(opts);

   return opts;
}

int ResolveAutoMinGainPct(const RunOptions &opts)
{
   return opts.fAutoMinGainPct ? *opts.fAutoMinGainPct : 1;
}

bool ResolveAutoMaxLevel(const RunOptions &opts)
{
   return opts.fAutoMaxLevel;
}

unsigned ResolveAutoCodecs(const RunOptions &opts)
{
   return opts.fAutoCodecs ? *opts.fAutoCodecs : kLHC4AutoCodecsRoot;
}

void ApplyPageSizeOptions(RNTupleWriteOptions &writeOpts, const RunOptions &runOpts)
{
   if (runOpts.fPageSize)
      writeOpts.SetMaxUnzippedPageSize(*runOpts.fPageSize);

   if (runOpts.fInitialPageSize) {
      writeOpts.SetInitialUnzippedPageSize(*runOpts.fInitialPageSize);

      constexpr std::size_t kDefaultInitialPageSize = 256;
      if (*runOpts.fInitialPageSize > kDefaultInitialPageSize) {
         std::size_t autoBudget = writeOpts.GetApproxZippedClusterSize();
         if (writeOpts.GetCompression() != 0)
            autoBudget += writeOpts.GetApproxZippedClusterSize();
         writeOpts.SetPageBufferBudget(autoBudget * (*runOpts.fInitialPageSize) / kDefaultInitialPageSize);
      }
   }
}

std::string FormatPageSizeSettings(const RunOptions &opts)
{
   if (!opts.fPageSize && !opts.fInitialPageSize)
      return "default (initial=256 B, max=1 MiB)";

   const auto initial =
      opts.fInitialPageSize ? FormatBytes(*opts.fInitialPageSize) : std::string("256 B (default)");
   const auto max = opts.fPageSize ? FormatBytes(*opts.fPageSize) : std::string("1 MiB (default)");
   return "initial=" + initial + ", max=" + max;
}

void AppendPageSizeArgs(std::vector<std::string> &args, const RunOptions &opts)
{
   if (opts.fPageSize) {
      args.emplace_back("--page-size");
      args.push_back(std::to_string(*opts.fPageSize));
   }
   if (opts.fInitialPageSize) {
      args.emplace_back("--initial-page-size");
      args.push_back(std::to_string(*opts.fInitialPageSize));
   }
}

void AppendLevelArgs(std::vector<std::string> &args, const RunOptions &opts)
{
   args.emplace_back("--level");
   args.push_back(std::to_string(opts.fLevel));
}

void AppendAutoMinGainArgs(std::vector<std::string> &args, const RunOptions &opts)
{
   if (opts.fAutoMinGainPct) {
      args.emplace_back("--auto-min-gain");
      args.push_back(std::to_string(*opts.fAutoMinGainPct));
   }
}

void AppendAutoMaxLevelArgs(std::vector<std::string> &args, const RunOptions &opts)
{
   if (opts.fAutoMaxLevel)
      args.emplace_back("--auto-max-level");
}

void AppendAutoCodecsArgs(std::vector<std::string> &args, const RunOptions &opts)
{
   if (!opts.fAutoCodecs)
      return;
   args.emplace_back("--auto-codecs");
   if (*opts.fAutoCodecs == kLHC4AutoCodecsAll)
      args.emplace_back("all");
   else if (*opts.fAutoCodecs == kLHC4AutoCodecsRoot)
      args.emplace_back("root");
   else
      args.push_back(FormatAutoCodecMask(*opts.fAutoCodecs));
}

void AppendProfileArgs(std::vector<std::string> &args, const RunOptions &opts)
{
   if (opts.fProfile == kLHC4ProfileNone && opts.fProfileId == 0)
      return;
   args.emplace_back("--profile");
   args.emplace_back(FormatProfileName(opts.fProfile));
   if (opts.fProfileId != 0) {
      args.emplace_back("--profile-id");
      args.push_back(std::to_string(opts.fProfileId));
   }
}

void AppendNoOutArgs(std::vector<std::string> &args, const RunOptions &opts)
{
   if (opts.fNoOut)
      args.emplace_back("--no-out");
}

#ifdef R__HAS_LHC4CODEC
void ResetLHC4Globals(int autoMinGainPct, bool autoMaxLevel, unsigned autoCodecs)
{
   R__SetLHC4Codec(kLHC4CodecBeam);
   R__SetLHC4Filters(0);
   R__SetLHC4FilterFallback(1);
   R__SetLHC4FilterRle(1);
   R__SetLHC4FilterDict(1);
   R__SetLHC4AutoMinGainPct(autoMinGainPct);
   R__SetLHC4AutoMaxLevel(autoMaxLevel ? 1 : 0);
   R__SetLHC4AutoCodecs(autoCodecs);
   R__SetLHC4Profile(kLHC4ProfileNone);
   R__SetLHC4ProfileId(0);
   R__SetLHC4ProfilePerColumn(0);
}

RNTupleWriteOptions MakeLHC4WriteOptions(int level, bool keepColumnEncoding)
{
   RNTupleWriteOptions opts;
   opts.SetEnableColumnEncoding(keepColumnEncoding);
   opts.SetCompression(ROOT::RCompressionSetting::EAlgorithm::kLHC4, level);
   return opts;
}

void ConfigureLHC4FilterGlobals(int codec, bool enableByteFilters, int autoMinGainPct, bool autoMaxLevel,
                                unsigned autoCodecs, int profile, unsigned profileId, bool profilePerColumn)
{
   ResetLHC4Globals(autoMinGainPct, autoMaxLevel, autoCodecs);
   R__SetLHC4Codec(codec);
   if (enableByteFilters)
      R__SetLHC4Filters(1);
   // Standard Crystal stays dynamic. The profile flavour opts into static models.
   if (codec == kLHC4CodecCrystal && profilePerColumn) {
      if (profile != kLHC4ProfileNone) {
         R__SetLHC4Profile(profile);
         R__SetLHC4ProfileId(profileId);
      } else {
         R__SetLHC4ProfilePerColumn(1);
      }
   }
}

FilterStatsSummary ReadFilterStatsSummary()
{
   R__LHC4CompressStatsSummary stats{};
   R__GetLHC4CompressStatsSummary(&stats);
   FilterStatsSummary out;
   if (stats.num_pages == 0)
      return out;
   out.fValid = true;
   const double denom = static_cast<double>(stats.num_pages);
   out.fFilterOnPct = 100.0 * static_cast<double>(stats.filtered_won_pages) / denom;
   const auto filterOff = stats.raw_won_pages + stats.no_transform_pages + stats.stored_fallback_pages +
                          stats.plain_pages;
   out.fFilterOffPct = 100.0 * static_cast<double>(filterOff) / denom;
   out.fRawWonPct = 100.0 * static_cast<double>(stats.raw_won_pages) / denom;
   out.fNoTransformPct = 100.0 * static_cast<double>(stats.no_transform_pages) / denom;
   if (stats.uncompressed_bytes > 0) {
      out.fPageRatioValid = true;
      out.fPageRatio = static_cast<double>(stats.compressed_bytes) /
                       static_cast<double>(stats.uncompressed_bytes);
   }
   return out;
}

AutoCodecStatsSummary ReadAutoCodecStatsSummary()
{
   R__LHC4AutoCodecStatsSummary stats{};
   R__GetLHC4AutoCodecStatsSummary(&stats);
   AutoCodecStatsSummary out;
   if (stats.auto_selections == 0)
      return out;
   out.fValid = true;
   out.fSelections = stats.auto_selections;
   for (int i = 0; i < kLHC4CodecCount; ++i)
      out.fCodecHits[i] = stats.codec_hits[i];
   return out;
}
#endif

struct Variant {
   const char *fLabel = nullptr;
   std::string fSuffix;
   RNTupleWriteOptions fOptions;
   bool fUsesLHC4Globals = false;
   bool fEnableLHC4ByteFilters = false;
   int fLHC4Codec = 0;
   bool fRequireCodecAvailable = false;
   bool fLHC4ProfilePerColumn = false;
};

RNTupleWriteOptions MakeNativeWriteOptions(ROOT::RCompressionSetting::EAlgorithm::EValues algorithm, int level)
{
   RNTupleWriteOptions opts;
   opts.SetCompression(algorithm, level);
   return opts;
}

std::vector<Variant> MakeVariants(const RunOptions &runOpts)
{
   std::vector<Variant> variants;

   variants.push_back({"native_zstd", "_native",
                       MakeNativeWriteOptions(ROOT::RCompressionSetting::EAlgorithm::kZSTD,
                                              ROOT::RCompressionSetting::ELevel::kDefaultZSTD),
                       false, false, 0, false});
   variants.push_back({"native_lzma", "_native_lzma",
                       MakeNativeWriteOptions(ROOT::RCompressionSetting::EAlgorithm::kLZMA,
                                              ROOT::RCompressionSetting::ELevel::kDefaultLZMA),
                       false, false, 0, false});

#ifdef R__HAS_LHC4CODEC
   const int kLevel = runOpts.fLevel;
   const auto lhc4Opts = MakeLHC4WriteOptions(kLevel, runOpts.fKeepColumnEncoding);
   const std::string lv = std::to_string(kLevel);

   variants.push_back({"lhc4_zstd", "_lhc4_zstd", lhc4Opts, true, true, kLHC4CodecZstd, true});
   variants.push_back({"lhc4_bzip3", "_lhc4_bzip3", lhc4Opts, true, true, kLHC4CodecBzip3, true});
   variants.push_back({"lhc4_beam5", "_lhc4_beam" + lv, lhc4Opts, true, true, kLHC4CodecBeam, false});
   variants.push_back({"lhc4_mosaic5", "_lhc4_mosaic" + lv, lhc4Opts, true, true, kLHC4CodecMosaic, false});
   variants.push_back({"lhc4_prism5", "_lhc4_prism" + lv, lhc4Opts, true, true, kLHC4CodecBwt, false});
   variants.push_back({"lhc4_crystal5", "_lhc4_crystal" + lv, lhc4Opts, true, true, kLHC4CodecCrystal, false});
   variants.push_back({"lhc4_crystal_profile5", "_lhc4_crystal_profile" + lv, lhc4Opts, true, true, kLHC4CodecCrystal,
                       false, true});
   variants.push_back({"lhc4_oracle5", "_lhc4_oracle" + lv, lhc4Opts, true, true, kLHC4CodecOracle, false});
   variants.push_back({"lhc4_flash5", "_lhc4_flash" + lv, lhc4Opts, true, true, kLHC4CodecFlash, false});
   variants.push_back({"lhc4_laser5", "_lhc4_laser" + lv, lhc4Opts, true, true, kLHC4CodecLaser, false});
   // Grating has no levels (the model does not search); the RNTuple level is ignored.
   variants.push_back({"lhc4_grating", "_lhc4_grating", lhc4Opts, true, true, kLHC4CodecGrating, false});
   variants.push_back({"lhc4_auto", "_lhc4_auto", lhc4Opts, true, true, kLHC4CodecAuto, false});
#endif

   for (auto &variant : variants)
      ApplyPageSizeOptions(variant.fOptions, runOpts);

   return variants;
}

const Variant *FindVariant(const std::vector<Variant> &variants, const std::string &label)
{
   std::string resolved = label;
   if (resolved == "lhc4_lz5" || resolved == "lhc4_lz" || resolved == "lhc4_beam")
      resolved = "lhc4_beam5";
   for (const auto &variant : variants) {
      if (resolved == variant.fLabel)
         return &variant;
   }
   const int n = TrailingVariantLevel(resolved);
   std::string family = resolved;
   if (n > 0)
      family = resolved.substr(0, resolved.size() - std::to_string(n).size());
   if (family.empty())
      return nullptr;
   const std::string fallback = family + std::to_string(kDefaultLHC4Level);
   for (const auto &variant : variants) {
      if (fallback == variant.fLabel)
         return &variant;
   }
   for (const auto &variant : variants) {
      if (family == variant.fLabel)
         return &variant;
   }
   return nullptr;
}

std::unique_ptr<RColumnElementBase> MakeColumnElement(const ROOT::RColumnDescriptor &col)
{
   auto elem = RColumnElementBase::Generate(col.GetType());
   elem->SetBitsOnStorage(col.GetBitsOnStorage());
   if (const auto range = col.GetValueRange())
      elem->SetValueRange(range->fMin, range->fMax);
   return elem;
}

/// Unzip + unpack RNTuple column encodings, then pack as plain columns and zip with the destination compressor.
/// Does not reconstruct a C++ model, so streamer fields do not need dictionaries.
void RecodeUnpackedPages(RPageSource &source, RPagePersistentSink &destination)
{
   source.Attach(RNTupleSerializer::EDescriptorDeserializeMode::kForWriting);

   std::vector<std::unique_ptr<RColumnElementBase>> srcElems;
   std::vector<std::unique_ptr<RColumnElementBase>> dstElems;
   std::size_t nPhysical = 0;
   {
      auto guard = source.GetSharedDescriptorGuard();
      (void)destination.InitFromDescriptor(guard.GetRef(), false /* copyClusters */, true /* drop encodings */);
      for (const auto &extra : guard->GetExtraTypeInfoIterable())
         destination.UpdateExtraTypeInfo(extra);

      nPhysical = guard->GetNPhysicalColumns();
      srcElems.resize(nPhysical);
      dstElems.resize(nPhysical);
      const auto &dstDesc = destination.GetDescriptor();
      for (std::size_t i = 0; i < nPhysical; ++i) {
         srcElems[i] = MakeColumnElement(guard->GetColumnDescriptor(i));
         dstElems[i] = MakeColumnElement(dstDesc.GetColumnDescriptor(i));
      }
   }

   RPageAllocatorHeap pageAlloc;
   ROOT::Internal::RClusterPool clusterPool{source};
   const auto writeChecksum = destination.GetWriteOptions().GetEnablePageChecksums();
   const auto compression = destination.GetWriteOptions().GetCompression();
   const auto checksumSize = writeChecksum * RPageStorage::kNBytesPageChecksum;

   ROOT::DescriptorId_t clusterId;
   {
      auto guard = source.GetSharedDescriptorGuard();
      clusterId = guard->FindClusterId(0, 0);
   }

   struct ActiveColumn {
      ROOT::DescriptorId_t fId = 0;
      std::vector<ROOT::RClusterDescriptor::RPageInfo> fPages;
   };

   while (clusterId != ROOT::kInvalidDescriptorId) {
      ROOT::NTupleSize_t nEntries = 0;
      RCluster::ColumnSet_t loadSet;
      std::vector<ROOT::DescriptorId_t> suppressed;
      std::vector<ActiveColumn> active;
      ROOT::DescriptorId_t nextClusterId = ROOT::kInvalidDescriptorId;
      {
         auto guard = source.GetSharedDescriptorGuard();
         const auto &clusterDesc = guard->GetClusterDescriptor(clusterId);
         nEntries = clusterDesc.GetNEntries();
         nextClusterId = guard->FindNextClusterId(clusterId);
         for (std::size_t colId = 0; colId < nPhysical; ++colId) {
            if (!clusterDesc.ContainsColumn(colId))
               continue;
            if (clusterDesc.GetColumnRange(colId).IsSuppressed()) {
               suppressed.push_back(colId);
               continue;
            }
            ActiveColumn col;
            col.fId = colId;
            col.fPages = clusterDesc.GetPageRange(colId).GetPageInfos();
            loadSet.emplace(colId);
            active.push_back(std::move(col));
         }
      }

      for (auto colId : suppressed)
         destination.CommitSuppressedColumn({colId, nullptr});

      const RCluster *cluster = loadSet.empty() ? nullptr : clusterPool.GetCluster(clusterId, loadSet);

      std::vector<std::unique_ptr<std::byte[]>> buffers;
      std::vector<RPageStorage::SealedPageSequence_t> pagesV;
      std::vector<RPageStorage::RSealedPageGroup> groups;
      pagesV.reserve(active.size());
      groups.reserve(active.size());

      for (const auto &col : active) {
         RPageStorage::SealedPageSequence_t sealedPages;
         sealedPages.resize(col.fPages.size());
         for (std::size_t pageIdx = 0; pageIdx < col.fPages.size(); ++pageIdx) {
            const auto &pageInfo = col.fPages[pageIdx];
            const auto nElements = pageInfo.GetNElements();
            ROOT::Internal::RPage unpacked;

            if (pageInfo.GetLocator().GetType() == ROOT::RNTupleLocator::kTypePageZero) {
               unpacked = pageAlloc.NewPage(dstElems[col.fId]->GetSize(), nElements);
               unpacked.GrowUnchecked(nElements);
               std::memset(unpacked.GetBuffer(), 0, unpacked.GetNBytes());
            } else {
               if (!cluster)
                  throw std::runtime_error("missing cluster pages while unpacking column " +
                                           std::to_string(col.fId));
               ROOT::Internal::ROnDiskPage::Key key{col.fId, pageIdx};
               auto onDiskPage = cluster->GetOnDiskPage(key);
               const auto pageChecksumSize = pageInfo.HasChecksum() * RPageStorage::kNBytesPageChecksum;
               RPageStorage::RSealedPage inPage;
               inPage.SetNElements(nElements);
               inPage.SetHasChecksum(pageInfo.HasChecksum());
               inPage.SetBufferSize(pageInfo.GetLocator().GetNBytesOnStorage() + pageChecksumSize);
               if (!onDiskPage || onDiskPage->GetSize() != inPage.GetBufferSize())
                  throw std::runtime_error("on-disk page size mismatch while unpacking column " +
                                           std::to_string(col.fId));
               inPage.SetBuffer(onDiskPage->GetAddress());
               inPage.VerifyChecksumIfEnabled().ThrowOnError();
               unpacked = RPageSource::UnsealPage(inPage, *srcElems[col.fId], pageAlloc).Unwrap();
            }

            const auto bufSize = unpacked.GetNBytes() + checksumSize;
            auto buffer = MakeUninitArray<std::byte>(bufSize);
            const auto profileId = destination.EnsureColumnProfile(col.fId, unpacked, *dstElems[col.fId]);
            RPageSink::RSealPageConfig sealConf;
            sealConf.fPage = &unpacked;
            sealConf.fElement = dstElems[col.fId].get();
            sealConf.fCompressionSettings = compression;
            sealConf.fCompressionProfileId = profileId;
            sealConf.fWriteChecksum = writeChecksum;
            sealConf.fBuffer = buffer.get();
            sealedPages[pageIdx] = RPageSink::SealPage(sealConf);
            buffers.emplace_back(std::move(buffer));
         }
         pagesV.push_back(std::move(sealedPages));
         groups.emplace_back(col.fId, pagesV.back().cbegin(), pagesV.back().cend());
      }

      if (!groups.empty())
         destination.CommitSealedPageV(groups);
      destination.CommitCluster(nEntries);
      clusterId = nextClusterId;
   }

   destination.CommitClusterGroup();
   destination.CommitDataset();
}

std::optional<VariantResult> ReencodeVariant(const std::string &inputFile, const std::string &ntupleName,
                                             const std::string &outputFile, const Variant &variant, bool quiet,
                                             int autoMinGainPct, bool autoMaxLevel, unsigned autoCodecs, int profile,
                                             unsigned profileId, bool discardOutput)
{
#ifdef R__HAS_LHC4CODEC
   if (variant.fUsesLHC4Globals) {
      if (variant.fRequireCodecAvailable && !R__LHC4CodecAvailable(variant.fLHC4Codec)) {
         if (!quiet)
            std::cerr << "Skipping " << variant.fLabel << ": lhc4codec backend is not available in this build\n";
         return std::nullopt;
      }
      ConfigureLHC4FilterGlobals(variant.fLHC4Codec, variant.fEnableLHC4ByteFilters, autoMinGainPct, autoMaxLevel,
                                 autoCodecs, profile, profileId, variant.fLHC4ProfilePerColumn);
   } else {
      ResetLHC4Globals(autoMinGainPct, autoMaxLevel, autoCodecs);
   }
#else
   if (variant.fUsesLHC4Globals) {
      if (!quiet)
         std::cerr << "Skipping " << variant.fLabel << ": ROOT was built without lhc4codec support\n";
      return std::nullopt;
   }
#endif

   RemoveIfExists(outputFile);
   if (!quiet) {
      if (discardOutput)
         std::cout << "Converting " << variant.fLabel << " (--no-out)" << std::endl;
      else
         std::cout << "Writing " << variant.fLabel << " -> " << outputFile << std::endl;
   }

#ifdef R__HAS_LHC4CODEC
   if (variant.fUsesLHC4Globals)
      R__ResetLHC4CompressStats();
#endif

   const auto t0 = std::chrono::steady_clock::now();

   auto source = RPageSource::Create(ntupleName, inputFile, RNTupleReadOptions{});

   const auto writeOpts = variant.fOptions;
   auto destination = std::make_unique<RPageSinkFile>(ntupleName, outputFile, writeOpts);

   const bool unpackEncodings = !writeOpts.GetEnableColumnEncoding();
   if (unpackEncodings) {
      if (!quiet)
         std::cout << "Unpacking RNTuple column encodings (split/delta/zigzag) before re-encode\n";
      RecodeUnpackedPages(*source, *destination);
   } else {
      std::vector<RPageSource *> sources{source.get()};
      RNTupleMergeOptions mergeOpts;
      mergeOpts.fCompressionSettings = writeOpts.GetCompression();
      RNTupleMerger merger{std::move(destination)};
      merger.Merge(sources, mergeOpts).ThrowOnError();
   }

   const auto t1 = std::chrono::steady_clock::now();
   const std::chrono::duration<double> elapsed = t1 - t0;

#ifdef R__HAS_LHC4CODEC
   if (variant.fUsesLHC4Globals && !quiet) {
      std::cout << "Compression statistics (" << variant.fLabel << "):\n";
      R__PrintLHC4CompressStats();
   }
#endif

   if (!quiet)
      std::cout << "  done\n";

   VariantResult result;
   result.fLabel = variant.fLabel;
   result.fElapsedSec = elapsed.count();
   if (const auto size = FileSizeBytes(outputFile))
      result.fOutputBytes = *size;
#ifdef R__HAS_LHC4CODEC
   if (variant.fUsesLHC4Globals)
      result.fFilters = ReadFilterStatsSummary();
   if (std::strcmp(variant.fLabel, "lhc4_auto") == 0)
      result.fAutoCodec = ReadAutoCodecStatsSummary();
#endif
   if (discardOutput)
      RemoveIfExists(outputFile);
   return result;
}

std::string FormatGainVsNativeZstd(const VariantResult &row, std::uint64_t nativeZstdBytes)
{
   if (row.fLabel == "native_zstd")
      return "0.0%";
   if (nativeZstdBytes == 0)
      return "n/a";
   const double gain = 100.0 *
                       (static_cast<double>(nativeZstdBytes) - static_cast<double>(row.fOutputBytes)) /
                       static_cast<double>(nativeZstdBytes);
   return FormatPct(gain);
}

std::uint64_t FindNativeZstdOutputBytes(const std::vector<VariantResult> &results)
{
   for (const auto &row : results) {
      if (row.fLabel == "native_zstd")
         return row.fOutputBytes;
   }
   return 0;
}

void PrintAutoCodecSummary(const AutoCodecStatsSummary &stats)
{
   if (!stats.fValid)
      return;

   std::cout << "\nlhc4_auto backend mix (" << stats.fSelections << " page"
             << (stats.fSelections == 1 ? "" : "s") << "):\n";

   struct Row {
      const char *fName;
      int fCodec;
   };
   static constexpr Row kOrder[] = {{"zstd", kLHC4CodecZstd},     {"beam", kLHC4CodecBeam},
                                    {"mosaic", kLHC4CodecMosaic}, {"prism", kLHC4CodecBwt},
                                    {"laser", kLHC4CodecLaser},   {"lzma", kLHC4CodecLzma},
                                    {"bzip3", kLHC4CodecBzip3},   {"crystal", kLHC4CodecCrystal}};

   for (const auto &row : kOrder) {
      const auto n = stats.fCodecHits[row.fCodec];
      if (n == 0)
         continue;
      const double pct = 100.0 * static_cast<double>(n) / static_cast<double>(stats.fSelections);
      std::cout << "  " << std::left << std::setw(8) << row.fName << std::right << std::setw(8) << n << " ("
                << std::fixed << std::setprecision(1) << pct << "%)\n";
   }
}

void PrintSummaryTable(const RunOptions &opts, std::uint64_t inputBytes,
                       const std::vector<VariantResult> &results)
{
   std::cout << "\n=== rntuple2rntuple summary ===\n";
   std::cout << "Input:  " << opts.fInputFile << " [" << opts.fNTupleName << "]\n";
   std::cout << "Input size: " << FormatBytes(inputBytes) << '\n';
   if (opts.fNoOut)
      std::cout << "Output: (none, --no-out)\n";
   else
      std::cout << "Output base: " << opts.fOutputBase << '\n';
   std::cout << "Page size: " << FormatPageSizeSettings(opts) << '\n';
   std::cout << "LHC4 level: " << opts.fLevel << '\n';
   std::cout << "Auto min gain: " << ResolveAutoMinGainPct(opts) << "%\n";
   if (ResolveAutoMaxLevel(opts))
      std::cout << "Auto max level: on\n";
   std::cout << "Auto codecs: " << FormatAutoCodecMask(ResolveAutoCodecs(opts)) << '\n';
   std::cout << "Crystal: " << FormatCrystalProfileStatus(opts) << "\n\n";

   const std::uint64_t nativeZstdBytes = FindNativeZstdOutputBytes(results);

   constexpr int kLabelWidth = 24;
   std::cout << std::left << std::setw(kLabelWidth) << "Format" << std::right << std::setw(12) << "Output"
             << std::setw(10) << "Ratio" << std::setw(11) << "Page ratio" << std::setw(10) << "Gain"
             << std::setw(10) << "Time" << std::setw(12) << "Filter on" << std::setw(13) << "Filter off"
             << std::setw(12) << "Raw won" << std::setw(14) << "No transform" << '\n';
   std::cout << std::string(kLabelWidth + 12 + 10 + 11 + 10 + 10 + 12 + 13 + 12 + 14, '-') << '\n';

   for (const auto &row : results) {
      std::cout << std::left << std::setw(kLabelWidth) << row.fLabel << std::right << std::setw(12)
                << FormatBytes(row.fOutputBytes) << std::setw(10) << FormatRatio(row.fOutputBytes, inputBytes)
                << std::setw(11) << FormatPageRatio(row.fFilters) << std::setw(10)
                << FormatGainVsNativeZstd(row, nativeZstdBytes) << std::setw(10)
                << FormatSeconds(row.fElapsedSec);

      if (row.fFilters.fValid) {
         std::cout << std::setw(12) << FormatPct(row.fFilters.fFilterOnPct) << std::setw(13)
                   << FormatPct(row.fFilters.fFilterOffPct) << std::setw(12)
                   << FormatPct(row.fFilters.fRawWonPct) << std::setw(14)
                   << FormatPct(row.fFilters.fNoTransformPct);
      } else {
         std::cout << std::setw(12) << "n/a" << std::setw(13) << "n/a" << std::setw(12) << "n/a" << std::setw(14)
                   << "n/a";
      }
      std::cout << '\n';
   }

   std::cout << "\nRatio = RNTuple output size / input file size (lower is smaller).\n";
   std::cout << "Page ratio = lhc4codec compressed payload / uncompressed page bytes (lhc4_* only).\n";
   std::cout << "Gain  = space saved vs native_zstd output (positive = smaller file, native_zstd is 0%).\n";
   std::cout << "Filter on  = lhc4codec byte transform kept on wire (filtered won).\n";
   std::cout << "Filter off = raw won + no transform + stored fallback + plain pages.\n";

   for (const auto &row : results) {
      if (row.fLabel == "lhc4_auto")
         PrintAutoCodecSummary(row.fAutoCodec);
   }
}

std::vector<VariantResult> RunSequential(const RunOptions &opts)
{
   std::vector<VariantResult> results;
   for (const auto &variant : MakeVariants(opts)) {
      if (auto row = ReencodeVariant(opts.fInputFile, opts.fNTupleName, ResolveVariantOutputPath(opts, variant.fSuffix),
                                     variant, opts.fQuiet, ResolveAutoMinGainPct(opts), ResolveAutoMaxLevel(opts),
                                     ResolveAutoCodecs(opts), opts.fProfile, opts.fProfileId, opts.fNoOut)) {
         results.push_back(*row);
      }
   }
   return results;
}

#if !defined(_WIN32)
std::vector<std::string> BuildWorkerArgv(const char *executable, const RunOptions &opts, const Variant &variant,
                                         const std::string &resultPath)
{
   std::vector<std::string> args;
   args.emplace_back(executable);
   args.emplace_back("--only-variant");
   args.push_back(variant.fLabel);
   args.emplace_back("--worker-result");
   args.push_back(resultPath);
   args.emplace_back("--quiet");
   if (opts.fKeepColumnEncoding)
      args.emplace_back("--keep-column-encoding");
   AppendLevelArgs(args, opts);
   AppendPageSizeArgs(args, opts);
   AppendAutoMinGainArgs(args, opts);
   AppendAutoMaxLevelArgs(args, opts);
   AppendAutoCodecsArgs(args, opts);
   AppendProfileArgs(args, opts);
   AppendNoOutArgs(args, opts);
   args.push_back(opts.fInputFile);
   args.push_back(opts.fNTupleName);
   args.push_back(opts.fOutputBase);
   return args;
}

[[noreturn]] void ExecWorker(const std::vector<std::string> &args)
{
   std::vector<char *> argv;
   argv.reserve(args.size() + 1);
   for (const auto &arg : args)
      argv.push_back(const_cast<char *>(arg.c_str()));
   argv.push_back(nullptr);
   execvp(argv[0], argv.data());
   std::cerr << "rntuple2rntuple: failed to exec worker: " << args[0] << " (" << std::strerror(errno) << ")\n";
   _exit(127);
}

unsigned ResolveJobCount(int jobs, std::size_t numVariants)
{
   if (jobs == 0) {
      const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
      return static_cast<unsigned>(std::min<std::size_t>(hw, numVariants));
   }
   if (jobs < 0)
      throw std::runtime_error("--jobs must be >= 0");
   return static_cast<unsigned>(std::min<std::size_t>(static_cast<std::size_t>(jobs), numVariants));
}

std::vector<VariantResult> RunParallel(const RunOptions &opts, const char *executable)
{
   const auto variants = MakeVariants(opts);
   const unsigned jobCount = ResolveJobCount(opts.fJobs, variants.size());
   if (jobCount <= 1)
      return RunSequential(opts);

   std::cout << "Running " << variants.size() << " variants with up to " << jobCount
             << " parallel jobs (one process per format)\n";

   const std::string resultDir =
      std::string("/tmp/rntuple2rntuple-") + std::to_string(static_cast<long long>(getpid()));
   gSystem->mkdir(resultDir.c_str(), true);

   struct RunningJob {
      pid_t fPid = -1;
      std::string fResultPath;
      std::string fLabel;
   };

   std::vector<RunningJob> running;
   running.reserve(jobCount);
   std::vector<VariantResult> results;
   results.reserve(variants.size());
   std::size_t nextVariant = 0;
   int failedWorkers = 0;

   auto reapOne = [&]() {
      int status = 0;
      const pid_t done = wait(&status);
      if (done <= 0)
         throw std::runtime_error("wait() failed while collecting parallel workers");

      auto it = std::find_if(running.begin(), running.end(),
                             [done](const RunningJob &job) { return job.fPid == done; });
      if (it == running.end())
         throw std::runtime_error("unexpected worker pid " + std::to_string(done));

      if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
         std::cerr << "Worker failed for variant " << it->fLabel << '\n';
         ++failedWorkers;
      } else {
         results.push_back(ReadWorkerResult(it->fResultPath));
      }
      gSystem->Unlink(it->fResultPath.c_str());
      running.erase(it);
   };

   auto launchOne = [&](const Variant &variant) {
      RunningJob job;
      job.fLabel = variant.fLabel;
      job.fResultPath = resultDir + "/" + variant.fLabel + ".result";
      RemoveIfExists(job.fResultPath);

      const pid_t pid = fork();
      if (pid < 0)
         throw std::runtime_error(std::string("fork() failed for variant ") + variant.fLabel);

      if (pid == 0) {
         const auto workerArgs = BuildWorkerArgv(executable, opts, variant, job.fResultPath);
         ExecWorker(workerArgs);
      }

      job.fPid = pid;
      running.push_back(std::move(job));
   };

   while (nextVariant < variants.size() || !running.empty()) {
      while (running.size() < jobCount && nextVariant < variants.size()) {
         launchOne(variants[nextVariant]);
         ++nextVariant;
      }
      if (!running.empty())
         reapOne();
   }

   gSystem->Unlink(resultDir.c_str());

   if (failedWorkers > 0)
      throw std::runtime_error(std::to_string(failedWorkers) + " parallel worker(s) failed");

   std::sort(results.begin(), results.end(), [&variants](const VariantResult &a, const VariantResult &b) {
      auto rank = [&variants](const std::string &label) {
         for (std::size_t i = 0; i < variants.size(); ++i) {
            if (variants[i].fLabel == label)
               return i;
         }
         return variants.size();
      };
      return rank(a.fLabel) < rank(b.fLabel);
   });

   return results;
}
#endif

int RunWorkerMode(const RunOptions &opts)
{
   const auto variants = MakeVariants(opts);
   const Variant *variant = FindVariant(variants, opts.fOnlyVariant);
   if (!variant)
      throw std::runtime_error("unknown variant label: " + opts.fOnlyVariant);

   ROOT::EnableImplicitMT();

   const bool quiet = !opts.fWorkerResult.empty();
   auto result = ReencodeVariant(opts.fInputFile, opts.fNTupleName, ResolveVariantOutputPath(opts, variant->fSuffix),
                                 *variant, quiet, ResolveAutoMinGainPct(opts), ResolveAutoMaxLevel(opts),
                                 ResolveAutoCodecs(opts), opts.fProfile, opts.fProfileId, opts.fNoOut);
   if (!result)
      return 1;

   if (!opts.fWorkerResult.empty()) {
      WriteWorkerResult(opts.fWorkerResult, *result);
      return 0;
   }

   const auto inputSize = FileSizeBytes(opts.fInputFile);
   if (inputSize)
      PrintSummaryTable(opts, *inputSize, {*result});
   return 0;
}

int RunMain(const RunOptions &opts, const char *executable)
{
   if (opts.fBenchOnly) {
      std::vector<std::pair<std::string, std::string>> filesToBench;
      std::string explicitNTupleName;
      if (opts.fPositionalArgs.size() == 2 && opts.fPositionalArgs[1].find(".root") == std::string::npos) {
         filesToBench.push_back({RNTupleBench::Basename(opts.fPositionalArgs[0]), opts.fPositionalArgs[0]});
         explicitNTupleName = opts.fPositionalArgs[1];
      } else {
         for (const auto &p : opts.fPositionalArgs) {
            filesToBench.push_back({RNTupleBench::Basename(p), p});
         }
      }

      std::vector<RNTupleBench::BenchmarkResult> benchResults;
      for (const auto &[label, path] : filesToBench) {
         benchResults.push_back(RNTupleBench::RunBenchmarkForFile(
            label, path, explicitNTupleName, opts.fBenchThreads, opts.fBenchIters, opts.fBenchEntries,
            opts.fBenchPercent));
      }
      RNTupleBench::PrintBenchmarkSummary(benchResults);
      return 0;
   }

   if (!opts.fOnlyVariant.empty())
      return RunWorkerMode(opts);

   const auto inputSize = FileSizeBytes(opts.fInputFile);
   if (!inputSize)
      throw std::runtime_error("cannot stat input file: " + opts.fInputFile);

   if (opts.fKeepColumnEncoding && !opts.fQuiet)
      std::cout << "Using ROOT column encodings + lhc4codec backend for LHC4 variants\n";
   if ((opts.fPageSize || opts.fInitialPageSize) && !opts.fQuiet)
      std::cout << "Page size: " << FormatPageSizeSettings(opts) << '\n';
   if (!opts.fQuiet)
      std::cout << "LHC4 level: " << opts.fLevel << '\n';
   if (!opts.fQuiet)
      std::cout << "Auto min gain: " << ResolveAutoMinGainPct(opts) << "%\n";
   if (ResolveAutoMaxLevel(opts) && !opts.fQuiet)
      std::cout << "Auto max level: on\n";
   if (!opts.fQuiet)
      std::cout << "Auto codecs: " << FormatAutoCodecMask(ResolveAutoCodecs(opts)) << '\n';
   if (!opts.fQuiet)
      std::cout << "Crystal: " << FormatCrystalProfileStatus(opts) << '\n';

   std::vector<VariantResult> results;
#if defined(_WIN32)
   if (opts.fJobs != 1 && !opts.fQuiet)
      std::cerr << "rntuple2rntuple: parallel mode is not supported on Windows; running sequentially\n";
   ROOT::EnableImplicitMT();
   results = RunSequential(opts);
#else
   if (opts.fJobs == 1) {
      ROOT::EnableImplicitMT();
      results = RunSequential(opts);
   } else {
      results = RunParallel(opts, executable);
   }
#endif

   if (!opts.fQuiet)
      PrintSummaryTable(opts, *inputSize, results);

   if (opts.fBench && !opts.fNoOut) {
      const auto variants = MakeVariants(opts);
      std::vector<std::pair<std::string, std::string>> filesToBench;
      for (const auto &row : results) {
         const auto *v = FindVariant(variants, row.fLabel);
         if (v) {
            std::string outPath = ResolveVariantOutputPath(opts, v->fSuffix);
            filesToBench.push_back({row.fLabel, outPath});
         }
      }
      std::vector<RNTupleBench::BenchmarkResult> benchResults;
      for (const auto &[label, path] : filesToBench) {
         benchResults.push_back(RNTupleBench::RunBenchmarkForFile(
            label, path, opts.fNTupleName, opts.fBenchThreads, opts.fBenchIters, opts.fBenchEntries,
            opts.fBenchPercent));
      }
      RNTupleBench::PrintBenchmarkSummary(benchResults);
   }

   return 0;
}

} // anonymous namespace

int main(int argc, char **argv)
{
   try {
      const auto opts = ParseArgs(argc, argv);
      return RunMain(opts, argv[0]);
   } catch (const std::exception &e) {
      std::cerr << "rntuple2rntuple: " << e.what() << '\n';
      std::cerr << kUsage;
      return 1;
   }
}
