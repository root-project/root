/// \file tree2ntuple.cxx
/// \brief Convert a TTree into several RNTuple output files for compression studies.

#include <ROOT/RConfig.hxx>
#include <ROOT/RNTupleImporter.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>
#include <ROOT/RNTupleZip.hxx>

#include <Compression.h>
#include <TFile.h>
#include <TKey.h>
#include <TROOT.h>
#include <TSystem.h>
#include <TTree.h>

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
   kLHC4CodecCount = 9,
};
#define kLHC4CodecBit(c) (1u << (unsigned)(c))
#define kLHC4AutoCodecsRoot \
   (kLHC4CodecBit(kLHC4CodecZstd) | kLHC4CodecBit(kLHC4CodecBeam) | kLHC4CodecBit(kLHC4CodecCrystal))
#define kLHC4AutoCodecsAll                                                                 \
   (kLHC4AutoCodecsRoot | kLHC4CodecBit(kLHC4CodecMosaic) | kLHC4CodecBit(kLHC4CodecBwt) | \
    kLHC4CodecBit(kLHC4CodecLzma) | kLHC4CodecBit(kLHC4CodecBzip3))
#endif

#include <algorithm>
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

using ROOT::Experimental::RNTupleImporter;
using ROOT::RNTupleWriteOptions;

constexpr const char *kUsage = R"(Usage: tree2ntuple [options] input.root [treeName] [outputBase]

Options:
  --keep-column-encoding   Keep ROOT RNTuple split/delta/zigzag column encodings when using
                           lhc4codec as the page compressor (default: plain columns only).
  --page-size N            Set max unzipped page size in bytes (RNTupleWriteOptions default: 1 MiB).
                           Columns start with small pages and grow up to this limit.
  --initial-page-size N    Advanced: set the initial unzipped page size in bytes (default: 256 B).
                           Only needed for micro-benchmarks; large values require much more memory.
  --auto-min-gain N        Min % size gain for lhc4codec Auto to prefer a slower decoder
                           (default: 1). 0 = always pick the smallest payload.
  --auto-max-level         For lhc4codec Auto, race each codec at its maximum effort
                           instead of the RNTuple compression level (slower encode).
  --auto-codecs LIST       Codecs Auto may race: root (default: zstd,beam,crystal),
                           all, or a comma list (zstd,beam,crystal,mosaic,bwt,lzma,bzip3).
  --jobs N                 Run up to N variants in parallel (default: 1). Use 0 for auto
                           (number of CPU threads, capped by variant count). Requires fork().
  --quiet                  Suppress per-variant progress (used for parallel workers).
  --no-out                 Run conversions without keeping output files; print summary only.
  --verify                 After each page is compressed, decompress it and compare with the
                           raw input buffer (catches codec round-trip corruption).
  --only-variant LABEL     Run only the given format (see labels below).
  --worker-result PATH     Internal: write one-line key/value result metrics to PATH.
  -h, --help               Show this help.

Convert a TTree into RNTuple files:

  <outputBase>_native.root       native RNTuple column encodings + ZSTD (level 5)
  <outputBase>_native_lzma.root  native RNTuple column encodings + LZMA (level 7)
  <outputBase>_lhc4_zstd.root    lhc4codec filters + zstd (level 5)
  <outputBase>_lhc4_bzip3.root   lhc4codec filters + bzip3 (level 5)
  <outputBase>_lhc4_beam5.root   lhc4codec filters + Beam (level 5; lhc4_lz5 is a legacy alias)
  <outputBase>_lhc4_mosaic5.root lhc4codec filters + Mosaic (level 5)
  <outputBase>_lhc4_bwt5.root    lhc4codec filters + BWT (level 5)
  <outputBase>_lhc4_crystal5.root lhc4codec filters + Crystal (level 5)
  <outputBase>_lhc4_oracle5.root lhc4codec filters + Oracle (level 5; not raced by Auto)
  <outputBase>_lhc4_auto.root    lhc4codec filters + Auto codec race (level 5; see --auto-min-gain,
                                 --auto-max-level, --auto-codecs)

By default, LHC4 variants disable ROOT column encodings (plain columns) and enable
lhc4codec byte filters (shuffle/delta/zigzag/dict/RLE as applicable). Use
--keep-column-encoding to stack ROOT encodings on top of the lhc4codec backend.

Parallel mode launches one process per format so lhc4codec global settings and
compression statistics stay isolated.

If treeName is omitted, the first TTree in the input file is used.
If outputBase is omitted, it defaults to "<input-stem>_ntuple".
)";

struct RunOptions {
   bool fKeepColumnEncoding = false;
   bool fNoOut = false;
   bool fVerify = false;
   bool fQuiet = false;
   int fJobs = 1;
   std::optional<std::size_t> fPageSize;
   std::optional<std::size_t> fInitialPageSize;
   std::optional<int> fAutoMinGainPct;
   bool fAutoMaxLevel = false;
   std::optional<unsigned> fAutoCodecs;
   std::string fOnlyVariant;
   std::string fWorkerResult;
   std::string fInputFile;
   std::string fTreeName;
   std::string fOutputBase;
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
                                   {"bwt", kLHC4CodecBit(kLHC4CodecBwt)},
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

std::string FindFirstTreeName(const std::string &fileName)
{
   auto file = std::unique_ptr<TFile>(TFile::Open(fileName.c_str()));
   if (!file || file->IsZombie())
      throw std::runtime_error("cannot open input file: " + fileName);

   TIter next(file->GetListOfKeys());
   while (auto key = static_cast<TKey *>(next())) {
      if (std::strcmp(key->GetClassName(), "TTree") == 0)
         return key->GetName();
   }
   throw std::runtime_error("no TTree found in " + fileName);
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
   }
   return result;
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
      // Deprecated alias kept for compatibility with the first implementation.
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
      if (arg == "--verify") {
         opts.fVerify = true;
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

   opts.fInputFile = positional[0];
   if (positional.size() >= 2)
      opts.fTreeName = positional[1];
   else
      opts.fTreeName = FindFirstTreeName(opts.fInputFile);

   if (positional.size() >= 3)
      opts.fOutputBase = positional[2];
   else
      opts.fOutputBase = BasenameNoExt(opts.fInputFile) + "_ntuple";

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

      // ReservePage() allocates one initial buffer per column. Scale the auto budget only when
      // the user deliberately raises the initial page size above the ROOT default (256 B).
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

void AppendNoOutArgs(std::vector<std::string> &args, const RunOptions &opts)
{
   if (opts.fNoOut)
      args.emplace_back("--no-out");
}

void AppendVerifyArgs(std::vector<std::string> &args, const RunOptions &opts)
{
   if (opts.fVerify)
      args.emplace_back("--verify");
}

struct ZipVerifyGuard {
   explicit ZipVerifyGuard(bool enable) { ROOT::Internal::RNTupleCompressor::SetVerifyRoundtrip(enable); }
   ~ZipVerifyGuard() { ROOT::Internal::RNTupleCompressor::SetVerifyRoundtrip(false); }
   ZipVerifyGuard(const ZipVerifyGuard &) = delete;
   ZipVerifyGuard &operator=(const ZipVerifyGuard &) = delete;
};

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
}

RNTupleWriteOptions MakeLHC4WriteOptions(int level, bool keepColumnEncoding)
{
   RNTupleWriteOptions opts;
   opts.SetEnableColumnEncoding(keepColumnEncoding);
   opts.SetCompression(ROOT::RCompressionSetting::EAlgorithm::kLHC4, level);
   return opts;
}

void ConfigureLHC4FilterGlobals(int codec, bool enableByteFilters, int autoMinGainPct, bool autoMaxLevel,
                                unsigned autoCodecs)
{
   ResetLHC4Globals(autoMinGainPct, autoMaxLevel, autoCodecs);
   R__SetLHC4Codec(codec);
   if (enableByteFilters)
      R__SetLHC4Filters(1);
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
   const char *fSuffix = nullptr;
   RNTupleWriteOptions fOptions;
   bool fUsesLHC4Globals = false;
   bool fEnableLHC4ByteFilters = false;
   int fLHC4Codec = 0;
   bool fRequireCodecAvailable = false;
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
   constexpr int kLevel = 5;
   const auto lhc4Opts = MakeLHC4WriteOptions(kLevel, runOpts.fKeepColumnEncoding);

   variants.push_back({"lhc4_zstd", "_lhc4_zstd", lhc4Opts, true, true, kLHC4CodecZstd, true});
   variants.push_back({"lhc4_bzip3", "_lhc4_bzip3", lhc4Opts, true, true, kLHC4CodecBzip3, true});
   variants.push_back({"lhc4_beam5", "_lhc4_beam5", lhc4Opts, true, true, kLHC4CodecBeam, false});
   variants.push_back({"lhc4_mosaic5", "_lhc4_mosaic5", lhc4Opts, true, true, kLHC4CodecMosaic, false});
   variants.push_back({"lhc4_bwt5", "_lhc4_bwt5", lhc4Opts, true, true, kLHC4CodecBwt, false});
   variants.push_back({"lhc4_crystal5", "_lhc4_crystal5", lhc4Opts, true, true, kLHC4CodecCrystal, false});
   variants.push_back({"lhc4_oracle5", "_lhc4_oracle5", lhc4Opts, true, true, kLHC4CodecOracle, false});
   variants.push_back({"lhc4_auto", "_lhc4_auto", lhc4Opts, true, true, kLHC4CodecAuto, false});
#endif

   for (auto &variant : variants)
      ApplyPageSizeOptions(variant.fOptions, runOpts);

   return variants;
}

const Variant *FindVariant(const std::vector<Variant> &variants, const std::string &label)
{
   const std::string resolved = (label == "lhc4_lz5") ? "lhc4_beam5" : label;
   for (const auto &variant : variants) {
      if (resolved == variant.fLabel)
         return &variant;
   }
   return nullptr;
}

std::optional<VariantResult> ImportVariant(const std::string &inputFile, const std::string &treeName,
                                           const std::string &outputFile, const Variant &variant, bool quiet,
                                           int autoMinGainPct, bool autoMaxLevel, unsigned autoCodecs,
                                           bool discardOutput, bool verify)
{
#ifdef R__HAS_LHC4CODEC
   if (variant.fUsesLHC4Globals) {
      if (variant.fRequireCodecAvailable && !R__LHC4CodecAvailable(variant.fLHC4Codec)) {
         if (!quiet)
            std::cerr << "Skipping " << variant.fLabel << ": lhc4codec backend is not available in this build\n";
         return std::nullopt;
      }
      ConfigureLHC4FilterGlobals(variant.fLHC4Codec, variant.fEnableLHC4ByteFilters, autoMinGainPct, autoMaxLevel,
                                autoCodecs);
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

   ZipVerifyGuard verifyGuard(verify);
   auto importer = RNTupleImporter::Create(inputFile, treeName, outputFile);
   importer->SetWriteOptions(variant.fOptions);
   if (quiet)
      importer->SetIsQuiet(true);
   importer->Import();

   const auto t1 = std::chrono::steady_clock::now();
   const std::chrono::duration<double> elapsed = t1 - t0;

   const auto report = importer->GetLastImportReport();

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
   result.fOutputBytes = report.fFileBytesOnDisk;
   if (result.fOutputBytes == 0) {
      if (const auto size = FileSizeBytes(outputFile))
         result.fOutputBytes = *size;
   }
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
                                    {"mosaic", kLHC4CodecMosaic}, {"bwt", kLHC4CodecBwt},
                                    {"lzma", kLHC4CodecLzma},     {"bzip3", kLHC4CodecBzip3},
                                    {"crystal", kLHC4CodecCrystal}};

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
   std::cout << "\n=== tree2ntuple summary ===\n";
   std::cout << "Input:  " << opts.fInputFile << " [" << opts.fTreeName << "]\n";
   std::cout << "Input size: " << FormatBytes(inputBytes) << '\n';
   if (opts.fNoOut)
      std::cout << "Output: (none, --no-out)\n";
   else
      std::cout << "Output base: " << opts.fOutputBase << '\n';
   std::cout << "Page size: " << FormatPageSizeSettings(opts) << '\n';
   std::cout << "Auto min gain: " << ResolveAutoMinGainPct(opts) << "%\n";
   if (ResolveAutoMaxLevel(opts))
      std::cout << "Auto max level: on\n";
   std::cout << "Auto codecs: " << FormatAutoCodecMask(ResolveAutoCodecs(opts)) << '\n';
   std::cout << '\n';

   const std::uint64_t nativeZstdBytes = FindNativeZstdOutputBytes(results);

   std::cout << std::left << std::setw(14) << "Format" << std::right << std::setw(12) << "Output"
             << std::setw(10) << "Ratio" << std::setw(11) << "Page ratio" << std::setw(10) << "Gain"
             << std::setw(10) << "Time" << std::setw(12) << "Filter on" << std::setw(13) << "Filter off"
             << std::setw(12) << "Raw won" << std::setw(14) << "No transform" << '\n';
   std::cout << std::string(14 + 12 + 10 + 11 + 10 + 10 + 12 + 13 + 12 + 14, '-') << '\n';

   for (const auto &row : results) {
      std::cout << std::left << std::setw(14) << row.fLabel << std::right << std::setw(12)
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

   std::cout << "\nRatio = RNTuple output size / TTree input file size (lower is smaller).\n";
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
      if (auto row = ImportVariant(opts.fInputFile, opts.fTreeName, ResolveVariantOutputPath(opts, variant.fSuffix),
                                   variant, opts.fQuiet, ResolveAutoMinGainPct(opts), ResolveAutoMaxLevel(opts),
                                   ResolveAutoCodecs(opts), opts.fNoOut, opts.fVerify)) {
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
   AppendPageSizeArgs(args, opts);
   AppendAutoMinGainArgs(args, opts);
   AppendAutoMaxLevelArgs(args, opts);
   AppendAutoCodecsArgs(args, opts);
   AppendNoOutArgs(args, opts);
   AppendVerifyArgs(args, opts);
   args.push_back(opts.fInputFile);
   args.push_back(opts.fTreeName);
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
   std::cerr << "tree2ntuple: failed to exec worker: " << args[0] << " (" << std::strerror(errno) << ")\n";
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
      std::string("/tmp/tree2ntuple-") + std::to_string(static_cast<long long>(getpid()));
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
   auto result = ImportVariant(opts.fInputFile, opts.fTreeName, ResolveVariantOutputPath(opts, variant->fSuffix),
                               *variant, quiet, ResolveAutoMinGainPct(opts), ResolveAutoMaxLevel(opts),
                               ResolveAutoCodecs(opts), opts.fNoOut, opts.fVerify);
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
      std::cout << "Auto min gain: " << ResolveAutoMinGainPct(opts) << "%\n";
   if (ResolveAutoMaxLevel(opts) && !opts.fQuiet)
      std::cout << "Auto max level: on\n";
   if (!opts.fQuiet)
      std::cout << "Auto codecs: " << FormatAutoCodecMask(ResolveAutoCodecs(opts)) << '\n';
   if (opts.fVerify && !opts.fQuiet)
      std::cout << "Verify: enabled (compress/decompress round-trip per page)\n";

   std::vector<VariantResult> results;
#if defined(_WIN32)
   if (opts.fJobs != 1 && !opts.fQuiet)
      std::cerr << "tree2ntuple: parallel mode is not supported on Windows; running sequentially\n";
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
   return 0;
}

} // anonymous namespace

int main(int argc, char **argv)
{
   try {
      const auto opts = ParseArgs(argc, argv);
      return RunMain(opts, argv[0]);
   } catch (const std::exception &e) {
      std::cerr << "tree2ntuple: " << e.what() << '\n';
      std::cerr << kUsage;
      return 1;
   }
}
