/// \file rntupletryzip.cxx
/// \brief Re-encode RNTuple pages in memory and report sizes, without C++ dictionaries.

#include <ROOT/RColumnElementBase.hxx>
#include <ROOT/RConfig.hxx>
#include <ROOT/RNTupleInspector.hxx>
#include <ROOT/RNTupleZip.hxx>

#include <Compression.h>
#include <TError.h>
#include <TFile.h>
#include <TKey.h>
#include <TSystem.h>

#ifdef R__HAS_LHC4CODEC
#include <ZipLHC4.h>
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

using ROOT::Experimental::RNTupleInspector;
using ROOT::Internal::GenerateColumnElement;
using ROOT::Internal::RColumnElementBase;
using ROOT::Internal::RNTupleCompressor;
using ROOT::Internal::RNTupleDecompressor;

constexpr int kDefaultLHC4Level = 5;
constexpr int kDefaultZstdLevel = 5;
constexpr int kDefaultLzmaLevel = 7;

constexpr const char *kUsage = R"(Usage: rntupletryzip [options] input.root [ntupleName]

Unzip RNTuple pages from the on-disk descriptor, unpack split/delta/zigzag
encodings to logical values, and recompress them in memory. No C++ dictionaries
are loaded and no output file is written. This is the path that still works on
CMSSW MiniAOD files whose streamer fields used to block rntuple2rntuple.

By default this matches tree2ntuple / rntuple2rntuple LHC4: compress plain
column bytes, not already-split pages. Native variants (zstd/lzma) re-apply
ROOT split/delta/zigzag encodings before compressing, matching a native
RNTuple write. Remaining, unselected columns are counted at their current
on-disk size in the projected file total.

Options:
  --top N              Recode the N largest on-disk columns (default: 20).
  --all-columns        Recode every physical column (slow on MiniAOD-scale files).
  --pages N            Pages per column (default: 4, spread across clusters).
                       0 = all pages of the selected columns.
  --level N            LHC4 compression level (1-11, default: 5). Grating ignores it.
  --only-variant LIST  Comma-separated codecs (default:
                       native_zstd,native_lzma,lhc4_crystal,lhc4_laser,lhc4_grating).
  --keep-column-encoding  Compress already-packed split pages (old tryzip behaviour).
  --no-filters         LHC4: compress bytes with byte filters off.
  --brute-filters      LHC4: for each page, try none, auto-detect, and every
                       width-matched numeric filter; keep the smallest lossless
                       encoding (oracle). none and auto are also reported as
                       whole-set baselines. A singleton filter_mask forces the
                       transform (Crystal will not entropy-screen it back to raw).
  --brute-filters=all  Also try text/BCJ filters on mismatched widths. bitround
                       is lossy and is not raced.
  --threads N          Recode threads for codecs/pages and filter trials
                       (default: all hardware cores).
  --list-variants      Print available codec names and exit.
  --verify             Round-trip unzip after each recompress (slow).
  --csv                Also print a CSV of per-column scaled sizes.
  --verbose            Keep TClass dictionary warnings; print per-column progress.
  -h, --help           Show this help.

If ntupleName is omitted, Events is used when present, otherwise the RNTuple
with the largest on-disk payload.
)";

enum class Kind { kFloat, kInt, kIndex, kBit, kText, kOther };

struct RunOptions {
   std::string fInputFile;
   std::string fNTupleName;
   int fTop = 20;
   int fPages = 4;
   int fLevel = kDefaultLHC4Level;
   bool fAllColumns = false;
   bool fCsv = false;
   bool fVerbose = false;
   bool fVerify = false;
   bool fListVariants = false;
   bool fKeepColumnEncoding = false;
   bool fNoFilters = false;
   bool fBruteFilters = false;
   bool fBruteAll = false;
   int fThreads = 0; // 0 = hardware_concurrency
   std::vector<std::string> fOnly;
};

struct NTupleInfo {
   std::string fName;
   ROOT::NTupleSize_t fNEntries = 0;
   std::uint64_t fComp = 0;
   std::uint64_t fUncomp = 0;
   std::size_t fNColumns = 0;
};

struct ColRow {
   ROOT::DescriptorId_t fId = 0;
   std::string fField;
   std::string fTypeName;
   Kind fKind = Kind::kOther;
   ROOT::ENTupleColumnType fType = ROOT::ENTupleColumnType::kUnknown;
   std::uint16_t fBits = 0;
   std::uint64_t fNPages = 0;
   std::uint64_t fComp = 0;
   std::uint64_t fUncomp = 0;
};

struct PageLoc {
   std::uint64_t fOffset = 0;
   std::uint64_t fSealed = 0;
   std::size_t fPacked = 0;
   std::uint32_t fNElements = 0;
};

struct ZipVariant {
   std::string fLabel;
   int fCompression = 0;
   bool fLHC4 = false;
   int fLHC4Codec = 0;
   bool fFilters = true;
   bool fProfilePerColumn = false;
};

struct FilterSpec {
   const char *fName = nullptr;
   int fId = 0;    // 0 = none, -1 = auto, >0 = FilterId
   int fWidth = 0; // 0 = any width
};

const FilterSpec kFilterSpecs[] = {
   {"none", 0, 0},
   {"auto", -1, 0},
   {"shuffle16", 1, 16},
   {"shuffle32", 2, 32},
   {"shuffle64", 3, 64},
   {"delta16", 4, 16},
   {"delta32", 5, 32},
   {"delta64", 6, 64},
   {"zigzag_delta16", 7, 16},
   {"zigzag_delta32", 8, 32},
   {"zigzag_delta64", 9, 64},
   {"delta_shuffle32", 10, 32},
   {"zigzag_delta_shuffle32", 11, 32},
   {"bitshuffle32", 12, 32},
   {"dict32", 13, 32},
   {"dict64", 14, 64},
   {"delta_shuffle64", 15, 64},
   {"zigzag_shuffle32", 16, 32},
   {"zigzag_shuffle64", 17, 64},
   {"bitshuffle64", 18, 64},
   {"delta_bitshuffle32", 19, 32},
   {"dict16", 20, 16},
   {"for16", 21, 16},
   {"for32", 22, 32},
   {"for64", 23, 64},
   {"alp32", 24, 32},
   {"alp64", 25, 64},
   {"xor_shuffle32", 26, 32},
   {"xor_shuffle64", 27, 64},
   {"delta2_shuffle32", 28, 32},
   {"zigzag_delta2_shuffle32", 29, 32},
   {"delta2_shuffle64", 30, 64},
   {"bcj_x86", 31, 0},
   {"digit_norm", 32, 0},
   {"word_norm", 33, 0},
   {"base64", 34, 0},
   {"bitround32", 35, 32},
   {"bitround64", 36, 64},
   {"lag2_shuffle32", 37, 32},
   {"lag2_shuffle64", 38, 64},
   {"xor_lag2_shuffle32", 39, 32},
   {"bcj_arm64", 40, 0},
};

unsigned ChooseThreads(int requested, std::size_t nJobs)
{
   unsigned hw = std::thread::hardware_concurrency();
   if (hw == 0)
      hw = 1;
   unsigned n = requested > 0 ? static_cast<unsigned>(requested) : hw;
   if (nJobs == 0)
      return 1;
   if (n > nJobs)
      n = static_cast<unsigned>(nJobs);
   if (n < 1)
      n = 1;
   return n;
}

bool FilterMatchesColumn(const FilterSpec &f, int bits, bool bruteAll)
{
   if (f.fId <= 0)
      return true;
   if (bruteAll)
      return true;
   if (f.fId == 35 || f.fId == 36)
      return false; // bitround is lossy; only with --brute-filters=all
   if (f.fWidth == 0)
      return false;
   return f.fWidth == bits;
}

class IgnoreDictWarnings {
public:
   IgnoreDictWarnings() : fOld(gErrorIgnoreLevel) { gErrorIgnoreLevel = kError; }
   ~IgnoreDictWarnings() { gErrorIgnoreLevel = fOld; }
   IgnoreDictWarnings(const IgnoreDictWarnings &) = delete;
   IgnoreDictWarnings &operator=(const IgnoreDictWarnings &) = delete;

private:
   Int_t fOld = 0;
};

class PageReader {
public:
   explicit PageReader(const std::string &path)
   {
#if !defined(_WIN32)
      fFd = ::open(path.c_str(), O_RDONLY);
#endif
      if (fFd < 0) {
         fFile.reset(TFile::Open(path.c_str(), "READ"));
         if (!fFile || fFile->IsZombie())
            throw std::runtime_error("cannot open " + path + " for page reads");
      }
   }

   ~PageReader()
   {
#if !defined(_WIN32)
      if (fFd >= 0)
         ::close(fFd);
#endif
   }

   bool Read(std::uint64_t offset, void *buf, std::size_t n)
   {
#if !defined(_WIN32)
      if (fFd >= 0)
         return ::pread(fFd, buf, n, static_cast<off_t>(offset)) == static_cast<ssize_t>(n);
#endif
      if (!fFile)
         return false;
      fFile->Seek(offset);
      return !fFile->ReadBuffer(static_cast<char *>(buf), static_cast<int>(n));
   }

private:
   int fFd = -1;
   std::unique_ptr<TFile> fFile;
};

Kind Classify(ROOT::ENTupleColumnType t)
{
   using T = ROOT::ENTupleColumnType;
   switch (t) {
   case T::kReal64:
   case T::kReal32:
   case T::kReal16:
   case T::kSplitReal64:
   case T::kSplitReal32:
   case T::kReal32Trunc:
   case T::kReal32Quant: return Kind::kFloat;
   case T::kIndex64:
   case T::kIndex32:
   case T::kSplitIndex64:
   case T::kSplitIndex32: return Kind::kIndex;
   case T::kInt64:
   case T::kUInt64:
   case T::kInt32:
   case T::kUInt32:
   case T::kInt16:
   case T::kUInt16:
   case T::kInt8:
   case T::kUInt8:
   case T::kSplitInt64:
   case T::kSplitUInt64:
   case T::kSplitInt32:
   case T::kSplitUInt32:
   case T::kSplitInt16:
   case T::kSplitUInt16: return Kind::kInt;
   case T::kBit: return Kind::kBit;
   case T::kChar:
   case T::kByte: return Kind::kText;
   default: return Kind::kOther;
   }
}

bool IsSplitColumnType(ROOT::ENTupleColumnType t)
{
   using T = ROOT::ENTupleColumnType;
   switch (t) {
   case T::kSplitIndex64:
   case T::kSplitIndex32:
   case T::kSplitReal64:
   case T::kSplitReal32:
   case T::kSplitInt64:
   case T::kSplitInt32:
   case T::kSplitInt16:
   case T::kSplitUInt64:
   case T::kSplitUInt32:
   case T::kSplitUInt16: return true;
   default: return false;
   }
}

/// On-disk type a native RNTuple write would use: split floats, zigzag-split
/// integers, delta-split offsets. Types without a split form stay as they are.
ROOT::ENTupleColumnType NativeSplitType(ROOT::ENTupleColumnType t)
{
   using T = ROOT::ENTupleColumnType;
   switch (t) {
   case T::kReal32:
   case T::kSplitReal32: return T::kSplitReal32;
   case T::kReal64:
   case T::kSplitReal64: return T::kSplitReal64;
   case T::kInt16:
   case T::kSplitInt16: return T::kSplitInt16;
   case T::kInt32:
   case T::kSplitInt32: return T::kSplitInt32;
   case T::kInt64:
   case T::kSplitInt64: return T::kSplitInt64;
   case T::kUInt16:
   case T::kSplitUInt16: return T::kSplitUInt16;
   case T::kUInt32:
   case T::kSplitUInt32: return T::kSplitUInt32;
   case T::kUInt64:
   case T::kSplitUInt64: return T::kSplitUInt64;
   case T::kIndex32:
   case T::kSplitIndex32: return T::kSplitIndex32;
   case T::kIndex64:
   case T::kSplitIndex64: return T::kSplitIndex64;
   default: return t;
   }
}

std::unique_ptr<RColumnElementBase> MakeNativePacker(const RColumnElementBase &src)
{
   const auto nativeType = NativeSplitType(src.GetIdentifier().fOnDiskType);
   auto dst = GenerateColumnElement(src.GetIdentifier().fInMemoryType, nativeType);
   if (src.GetBitsOnStorage() != dst->GetBitsOnStorage()) {
      try {
         dst->SetBitsOnStorage(src.GetBitsOnStorage());
      } catch (const std::exception &) {
      }
   }
   if (const auto range = src.GetValueRange())
      dst->SetValueRange(range->first, range->second);
   return dst;
}

const char *KindName(Kind k)
{
   switch (k) {
   case Kind::kFloat: return "float";
   case Kind::kInt: return "int";
   case Kind::kIndex: return "index";
   case Kind::kBit: return "bit";
   case Kind::kText: return "text";
   default: return "other";
   }
}

std::string FormatBytes(std::uint64_t n)
{
   std::ostringstream os;
   os << std::fixed << std::setprecision(2);
   if (n >= (1ull << 30))
      os << (n / 1073741824.0) << " GiB";
   else if (n >= (1ull << 20))
      os << (n / 1048576.0) << " MiB";
   else if (n >= 1024)
      os << (n / 1024.0) << " KiB";
   else
      os << n << " B";
   return os.str();
}

std::string ShortField(const std::string &name)
{
   if (name.size() <= 40)
      return name;
   return "..." + name.substr(name.size() - 37);
}

std::vector<std::string> SplitComma(const std::string &s)
{
   std::vector<std::string> out;
   std::string cur;
   for (char c : s) {
      if (c == ',') {
         if (!cur.empty())
            out.push_back(cur);
         cur.clear();
      } else if (!std::isspace(static_cast<unsigned char>(c))) {
         cur.push_back(c);
      }
   }
   if (!cur.empty())
      out.push_back(cur);
   return out;
}

std::vector<std::string> ListRNTupleNames(const std::string &path)
{
   auto file = std::unique_ptr<TFile>(TFile::Open(path.c_str(), "READ"));
   if (!file || file->IsZombie())
      throw std::runtime_error("cannot open " + path);
   std::vector<std::string> names;
   TIter next(file->GetListOfKeys());
   while (auto key = static_cast<TKey *>(next())) {
      if (std::strcmp(key->GetClassName(), "ROOT::RNTuple") != 0)
         continue;
      const std::string name = key->GetName();
      if (std::find(names.begin(), names.end(), name) == names.end())
         names.push_back(name);
   }
   return names;
}

NTupleInfo DescribeRNTuple(const std::string &path, const std::string &name)
{
   auto inspector = RNTupleInspector::Create(name, path);
   const auto &desc = inspector->GetDescriptor();
   NTupleInfo info;
   info.fName = name;
   info.fNEntries = desc.GetNEntries();
   info.fComp = inspector->GetCompressedSize();
   info.fUncomp = inspector->GetUncompressedSize();
   info.fNColumns = desc.GetNPhysicalColumns();
   return info;
}

std::vector<NTupleInfo> DescribeAllRNTuples(const std::string &path)
{
   const auto names = ListRNTupleNames(path);
   if (names.empty())
      throw std::runtime_error("no RNTuple in " + path);
   std::vector<NTupleInfo> infos;
   infos.reserve(names.size());
   for (const auto &name : names)
      infos.push_back(DescribeRNTuple(path, name));
   return infos;
}

std::string SelectRNTuple(const std::vector<NTupleInfo> &infos, const std::string &requested)
{
   if (infos.empty())
      throw std::runtime_error("no RNTuple in file");
   if (!requested.empty()) {
      for (const auto &info : infos) {
         if (info.fName == requested)
            return requested;
      }
      std::ostringstream os;
      os << "unknown RNTuple '" << requested << "'; available:";
      for (const auto &info : infos)
         os << ' ' << info.fName;
      throw std::runtime_error(os.str());
   }
   for (const auto &info : infos) {
      if (info.fName == "Events")
         return info.fName;
   }
   auto best = std::max_element(infos.begin(), infos.end(),
                                [](const NTupleInfo &a, const NTupleInfo &b) { return a.fComp < b.fComp; });
   return best->fName;
}

void PrintRNTupleCatalog(const std::vector<NTupleInfo> &infos, const std::string &selected)
{
   if (infos.empty())
      return;
   std::cout << "RNTuples in file:\n";
   for (const auto &info : infos) {
      const char *mark = (info.fName == selected) ? "  * " : "    ";
      std::cout << mark << std::left << std::setw(22) << info.fName << std::right << std::setw(10) << info.fNEntries
                << " entries  " << std::setw(12) << FormatBytes(info.fComp) << " payload\n";
   }
   if (infos.size() > 1 && !selected.empty())
      std::cout << "  (* selected)\n";
   std::cout << '\n';
}

int AlgoLevel(ROOT::RCompressionSetting::EAlgorithm::EValues algo, int level)
{
   return ROOT::CompressionSettings(algo, level);
}

#ifdef R__HAS_LHC4CODEC
int ProfileForKind(Kind k)
{
   switch (k) {
   case Kind::kFloat: return kLHC4ProfileFloatKinematics;
   case Kind::kInt: return kLHC4ProfilePoissonCounts;
   case Kind::kIndex: return kLHC4ProfileIntMonotonic;
   case Kind::kText: return kLHC4ProfileText;
   default: return kLHC4ProfileNone;
   }
}

void ResetLHC4()
{
   R__SetLHC4Codec(kLHC4CodecBeam);
   R__SetLHC4Filters(0);
   R__SetLHC4FilterFallback(1);
   R__SetLHC4FilterRle(1);
   R__SetLHC4FilterDict(1);
   R__SetLHC4FilterMask(0);
   R__SetLHC4Profile(kLHC4ProfileNone);
   R__SetLHC4ProfileId(0);
   R__SetLHC4ProfilePerColumn(0);
   R__SetLHC4CallProfile(-1);
   R__SetLHC4CallProfileId(0);
   R__SetLHC4CallFilterTrial(-1, 0, -1);
   R__SetLHC4CallCodec(-1);
   R__SetLHC4AutoCodecs(0);
   R__SetLHC4AutoMaxLevel(0);
}

void ClearCallOverrides()
{
   R__SetLHC4CallCodec(-1);
   R__SetLHC4CallFilterTrial(-1, 0, -1);
   R__SetLHC4CallProfile(-1);
}

// Per-thread filter trial. Does not touch process-wide codec/filter knobs.
void ApplyFilterTrialOnThread(int filterId)
{
   if (filterId == 0) {
      R__SetLHC4CallFilterTrial(0, 0, 1);
      return;
   }
   if (filterId < 0) {
      R__SetLHC4CallFilterTrial(1, ~0ull, 1);
      return;
   }
   R__SetLHC4CallFilterTrial(1, 1ull << static_cast<unsigned>(filterId), 0);
}

void ApplyVariantOnThread(const ZipVariant &v, int filterId, Kind kind)
{
   if (!v.fLHC4) {
      ClearCallOverrides();
      return;
   }
   R__SetLHC4CallCodec(v.fLHC4Codec);
   ApplyFilterTrialOnThread(filterId);
   if (v.fProfilePerColumn)
      R__SetLHC4CallProfile(ProfileForKind(kind));
   else
      R__SetLHC4CallProfile(-1);
}

#endif

std::vector<ZipVariant> AllVariants(int lhc4Level)
{
   std::vector<ZipVariant> v;
   const int zstd = AlgoLevel(ROOT::RCompressionSetting::EAlgorithm::kZSTD, kDefaultZstdLevel);
   const int lzma7 = AlgoLevel(ROOT::RCompressionSetting::EAlgorithm::kLZMA, kDefaultLzmaLevel);
   const int lzma4 = AlgoLevel(ROOT::RCompressionSetting::EAlgorithm::kLZMA, 4);
   v.push_back({"native_zstd", zstd, false, 0, false, false});
   v.push_back({"native_lzma", lzma7, false, 0, false, false});
   v.push_back({"native_lzma4", lzma4, false, 0, false, false});
#ifdef R__HAS_LHC4CODEC
   const int lhc4 = AlgoLevel(ROOT::RCompressionSetting::EAlgorithm::kLHC4, lhc4Level);
   v.push_back({"lhc4_zstd", lhc4, true, kLHC4CodecZstd, true, false});
   v.push_back({"lhc4_beam", lhc4, true, kLHC4CodecBeam, true, false});
   v.push_back({"lhc4_crystal", lhc4, true, kLHC4CodecCrystal, true, false});
   v.push_back({"lhc4_crystal_profile", lhc4, true, kLHC4CodecCrystal, true, true});
   v.push_back({"lhc4_laser", lhc4, true, kLHC4CodecLaser, true, false});
   v.push_back({"lhc4_grating", lhc4, true, kLHC4CodecGrating, true, false});
   v.push_back({"lhc4_flash", lhc4, true, kLHC4CodecFlash, true, false});
   v.push_back({"lhc4_mosaic", lhc4, true, kLHC4CodecMosaic, true, false});
   v.push_back({"lhc4_prism", lhc4, true, kLHC4CodecBwt, true, false});
   v.push_back({"lhc4_auto", lhc4, true, kLHC4CodecAuto, true, false});
#else
   (void)lhc4Level;
#endif
   return v;
}

const ZipVariant *FindVariant(const std::vector<ZipVariant> &all, const std::string &label)
{
   for (const auto &v : all) {
      if (v.fLabel == label)
         return &v;
   }
   // lhc4_crystal5 / lhc4_laser9 -> strip trailing digits
   std::string family = label;
   while (!family.empty() && std::isdigit(static_cast<unsigned char>(family.back())))
      family.pop_back();
   if (family.empty() || family == label)
      return nullptr;
   for (const auto &v : all) {
      if (v.fLabel == family)
         return &v;
   }
   return nullptr;
}

std::vector<ZipVariant> SelectVariants(const RunOptions &opts)
{
   const auto all = AllVariants(opts.fLevel);
   std::vector<std::string> want = opts.fOnly;
   if (want.empty())
      want = {"native_zstd", "native_lzma", "lhc4_crystal", "lhc4_laser", "lhc4_grating"};
   std::vector<ZipVariant> out;
   for (const auto &name : want) {
      const auto *v = FindVariant(all, name);
      if (!v) {
         std::ostringstream os;
         os << "unknown variant '" << name << "'; available:";
         for (const auto &a : all)
            os << ' ' << a.fLabel;
         throw std::runtime_error(os.str());
      }
#ifdef R__HAS_LHC4CODEC
      if (v->fLHC4 && v->fLHC4Codec != kLHC4CodecAuto && !R__LHC4CodecAvailable(v->fLHC4Codec)) {
         std::cerr << "Skipping " << v->fLabel << ": backend not linked\n";
         continue;
      }
#endif
      out.push_back(*v);
   }
   if (out.empty())
      throw std::runtime_error("no usable variants");
   if (opts.fNoFilters) {
      for (auto &v : out)
         v.fFilters = false;
   }
   return out;
}

void PrintVariantList(int level)
{
   std::cout << "Available variants (--level " << level << " for LHC4):\n";
   for (const auto &v : AllVariants(level))
      std::cout << "  " << v.fLabel << '\n';
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
      if (arg == "--list-variants") {
         opts.fListVariants = true;
         continue;
      }
      if (arg == "--csv") {
         opts.fCsv = true;
         continue;
      }
      if (arg == "--verbose" || arg == "-v") {
         opts.fVerbose = true;
         continue;
      }
      if (arg == "--verify") {
         opts.fVerify = true;
         continue;
      }
      if (arg == "--all-columns") {
         opts.fAllColumns = true;
         continue;
      }
      if (arg == "--top") {
         if (i + 1 >= argc)
            throw std::runtime_error("--top requires an argument");
         opts.fTop = std::stoi(argv[++i]);
         continue;
      }
      if (arg.rfind("--top=", 0) == 0) {
         opts.fTop = std::stoi(arg.substr(6));
         continue;
      }
      if (arg == "--pages") {
         if (i + 1 >= argc)
            throw std::runtime_error("--pages requires an argument");
         opts.fPages = std::stoi(argv[++i]);
         continue;
      }
      if (arg.rfind("--pages=", 0) == 0) {
         opts.fPages = std::stoi(arg.substr(8));
         continue;
      }
      if (arg == "--level") {
         if (i + 1 >= argc)
            throw std::runtime_error("--level requires an argument");
         opts.fLevel = std::stoi(argv[++i]);
         continue;
      }
      if (arg.rfind("--level=", 0) == 0) {
         opts.fLevel = std::stoi(arg.substr(8));
         continue;
      }
      if (arg == "--only-variant") {
         if (i + 1 >= argc)
            throw std::runtime_error("--only-variant requires an argument");
         opts.fOnly = SplitComma(argv[++i]);
         continue;
      }
      if (arg.rfind("--only-variant=", 0) == 0) {
         opts.fOnly = SplitComma(arg.substr(15));
         continue;
      }
      if (arg == "--keep-column-encoding") {
         opts.fKeepColumnEncoding = true;
         continue;
      }
      if (arg == "--no-filters") {
         opts.fNoFilters = true;
         continue;
      }
      if (arg == "--brute-filters") {
         opts.fBruteFilters = true;
         continue;
      }
      if (arg == "--brute-filters=all") {
         opts.fBruteFilters = true;
         opts.fBruteAll = true;
         continue;
      }
      if (arg == "--threads") {
         if (i + 1 >= argc)
            throw std::runtime_error("--threads requires an argument");
         opts.fThreads = std::stoi(argv[++i]);
         continue;
      }
      if (arg.rfind("--threads=", 0) == 0) {
         opts.fThreads = std::stoi(arg.substr(10));
         continue;
      }
      if (!arg.empty() && arg[0] == '-')
         throw std::runtime_error("unknown option: " + arg);
      positional.push_back(arg);
   }
   if (opts.fListVariants)
      return opts;
   if (positional.empty())
      throw std::runtime_error("missing input file (see --help)");
   if (positional.size() > 2)
      throw std::runtime_error("too many positional arguments (see --help)");
   opts.fInputFile = positional[0];
   if (positional.size() == 2)
      opts.fNTupleName = positional[1];
   if (opts.fTop < 1)
      throw std::runtime_error("--top must be >= 1");
   if (opts.fPages < 0)
      throw std::runtime_error("--pages must be >= 0");
   if (opts.fLevel < 1 || opts.fLevel > 11)
      throw std::runtime_error("--level must be 1..11");
   if (opts.fThreads < 0)
      throw std::runtime_error("--threads must be >= 0");
   return opts;
}

std::vector<PageLoc> CollectPages(const ROOT::RNTupleDescriptor &desc, const ColRow &col, int maxPages)
{
   std::vector<PageLoc> all;
   all.reserve(static_cast<std::size_t>(std::min<std::uint64_t>(col.fNPages, 1024)));
   for (const auto &clusterDesc : desc.GetClusterIterable()) {
      bool present = false;
      for (const auto &colRange : clusterDesc.GetColumnRangeIterable()) {
         if (colRange.GetPhysicalColumnId() == col.fId && !colRange.IsSuppressed()) {
            present = true;
            break;
         }
      }
      if (!present)
         continue;
      const auto &pageRange = clusterDesc.GetPageRange(col.fId);
      for (const auto &pageInfo : pageRange.GetPageInfos()) {
         const auto &loc = pageInfo.GetLocator();
         if (loc.GetType() == ROOT::RNTupleLocator::kTypePageZero)
            continue;
         const auto sealed = loc.GetNBytesOnStorage();
         const auto packed = (static_cast<std::size_t>(col.fBits) * pageInfo.GetNElements() + 7) / 8;
         if (sealed == 0 || packed == 0)
            continue;
         PageLoc p;
         p.fOffset = loc.GetPosition<std::uint64_t>();
         p.fSealed = sealed;
         p.fPacked = packed;
         p.fNElements = pageInfo.GetNElements();
         all.push_back(p);
      }
   }
   if (maxPages <= 0 || static_cast<int>(all.size()) <= maxPages)
      return all;
   std::vector<PageLoc> sample;
   sample.reserve(static_cast<std::size_t>(maxPages));
   for (int i = 0; i < maxPages; ++i) {
      const std::size_t idx = (static_cast<std::size_t>(i) * all.size()) / static_cast<std::size_t>(maxPages);
      sample.push_back(all[idx]);
   }
   return sample;
}

struct InflatedPage {
   std::vector<unsigned char> fRaw;
   /// ROOT split/delta/zigzag page, used when probing native zstd/lzma.
   std::vector<unsigned char> fPackedNative;
   std::uint64_t fSealed = 0;
};

std::vector<InflatedPage> InflatePages(PageReader &reader, const std::vector<PageLoc> &pages,
                                       const RColumnElementBase &elem, bool unpackEncodings,
                                       const RColumnElementBase *nativeElem, int *nFail)
{
   std::vector<InflatedPage> out;
   out.reserve(pages.size());
   std::vector<unsigned char> sealed;
   std::vector<unsigned char> packed;
   std::vector<unsigned char> logical;
   const bool unpack = unpackEncodings && IsSplitColumnType(elem.GetIdentifier().fOnDiskType);
   for (const auto &p : pages) {
      if (sealed.size() < p.fSealed)
         sealed.resize(p.fSealed);
      if (!reader.Read(p.fOffset, sealed.data(), static_cast<std::size_t>(p.fSealed))) {
         ++(*nFail);
         continue;
      }
      InflatedPage inf;
      inf.fSealed = p.fSealed;
      try {
         packed.resize(p.fPacked);
         RNTupleDecompressor::Unzip(sealed.data(), static_cast<std::size_t>(p.fSealed), p.fPacked, packed.data());
         if (unpack) {
            inf.fRaw.resize(elem.GetSize() * p.fNElements);
            elem.Unpack(inf.fRaw.data(), packed.data(), p.fNElements);
         } else {
            inf.fRaw = packed;
         }
         if (nativeElem) {
            const unsigned char *src = packed.data();
            if (unpack) {
               src = inf.fRaw.data();
            } else if (!elem.IsMappable()) {
               logical.resize(elem.GetSize() * p.fNElements);
               elem.Unpack(logical.data(), packed.data(), p.fNElements);
               src = logical.data();
            }
            inf.fPackedNative.resize(nativeElem->GetPackedSize(p.fNElements));
            nativeElem->Pack(inf.fPackedNative.data(), src, p.fNElements);
         }
      } catch (const std::exception &) {
         ++(*nFail);
         continue;
      }
      out.push_back(std::move(inf));
   }
   return out;
}

struct RecodeJob {
   std::size_t fVariant = 0;
   std::size_t fCol = 0;
   std::size_t fPage = 0;
   int fFilterId = -1;
};

void ZipRecodeJobs(const std::vector<ZipVariant> &variants, const std::vector<ColRow> &cols,
                   const std::vector<std::vector<InflatedPage>> &inflated, const std::vector<RecodeJob> &jobs,
                   std::vector<std::uint64_t> &jobZip, std::atomic<std::uint64_t> *cpuNs, unsigned nThreads)
{
   std::atomic<std::size_t> next{0};
   auto worker = [&] {
      std::vector<unsigned char> zipped;
      for (;;) {
         const std::size_t j = next.fetch_add(1);
         if (j >= jobs.size())
            break;
         const auto &job = jobs[j];
         const auto &variant = variants[job.fVariant];
         const auto &page = inflated[job.fCol][job.fPage];
#ifdef R__HAS_LHC4CODEC
         ApplyVariantOnThread(variant, job.fFilterId, cols[job.fCol].fKind);
#endif
         const auto &src = (!variant.fLHC4 && !page.fPackedNative.empty()) ? page.fPackedNative : page.fRaw;
         if (zipped.size() < src.size())
            zipped.resize(src.size());
         const auto t0 = std::chrono::steady_clock::now();
         jobZip[j] = RNTupleCompressor::Zip(src.data(), src.size(), variant.fCompression, zipped.data());
         if (cpuNs) {
            const auto ns =
               std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
            cpuNs[job.fVariant].fetch_add(static_cast<std::uint64_t>(ns), std::memory_order_relaxed);
         }
      }
#ifdef R__HAS_LHC4CODEC
      ClearCallOverrides();
#endif
   };
   if (nThreads <= 1) {
      worker();
      return;
   }
   std::vector<std::thread> pool;
   pool.reserve(nThreads);
   for (unsigned t = 0; t < nThreads; ++t)
      pool.emplace_back(worker);
   for (auto &th : pool)
      th.join();
}

} // namespace

int main(int argc, char **argv)
{
   try {
      auto opts = ParseArgs(argc, argv);
      if (opts.fListVariants) {
         PrintVariantList(opts.fLevel);
         return 0;
      }

      FileStat_t st{};
      if (gSystem->GetPathInfo(opts.fInputFile.c_str(), st) != 0)
         throw std::runtime_error("cannot stat " + opts.fInputFile);

      std::unique_ptr<IgnoreDictWarnings> quiet;
      if (!opts.fVerbose)
         quiet = std::make_unique<IgnoreDictWarnings>();

      const auto catalog = DescribeAllRNTuples(opts.fInputFile);
      opts.fNTupleName = SelectRNTuple(catalog, opts.fNTupleName);
      auto inspector = RNTupleInspector::Create(opts.fNTupleName, opts.fInputFile);
      const auto &desc = inspector->GetDescriptor();
      const auto variants = SelectVariants(opts);

      std::cout << "File:     " << opts.fInputFile << " (" << FormatBytes(st.fSize) << ")\n";
      PrintRNTupleCatalog(catalog, opts.fNTupleName);
      std::cout << "RNTuple:  " << opts.fNTupleName << '\n';
      std::cout << "Entries:  " << desc.GetNEntries() << "   columns: " << desc.GetNPhysicalColumns() << '\n';
      std::cout << "On disk:  " << inspector->GetCompressionSettingsAsString() << "   payload "
                << FormatBytes(inspector->GetCompressedSize()) << " / "
                << FormatBytes(inspector->GetUncompressedSize()) << " uncompressed  (" << std::fixed
                << std::setprecision(2) << inspector->GetCompressionFactor() << "x)\n";
      std::cout << "Mode:     page recode in memory (no dictionaries, "
                << (opts.fKeepColumnEncoding ? "encodings kept" : "unpack split/delta/zigzag") << ")\n\n";

      std::vector<ColRow> cols;
      for (const auto &colDesc : desc.GetColumnIterable()) {
         if (colDesc.IsAliasColumn())
            continue;
         const auto id = colDesc.GetPhysicalId();
         const auto &ci = inspector->GetColumnInspector(id);
         ColRow row;
         row.fId = id;
         row.fField = desc.GetQualifiedFieldName(colDesc.GetFieldId());
         if (colDesc.GetIndex() != 0)
            row.fField += "[" + std::to_string(colDesc.GetIndex()) + "]";
         row.fTypeName = RColumnElementBase::GetColumnTypeName(colDesc.GetType());
         row.fKind = Classify(colDesc.GetType());
         row.fType = colDesc.GetType();
         row.fBits = colDesc.GetBitsOnStorage();
         row.fNPages = ci.GetNPages();
         row.fComp = ci.GetCompressedSize();
         row.fUncomp = ci.GetUncompressedSize();
         cols.push_back(std::move(row));
      }
      std::sort(cols.begin(), cols.end(), [](const ColRow &a, const ColRow &b) { return a.fComp > b.fComp; });

      int nSel = opts.fAllColumns ? static_cast<int>(cols.size())
                                  : std::min(opts.fTop, static_cast<int>(cols.size()));
      cols.resize(static_cast<std::size_t>(nSel));

      std::uint64_t selectedOrig = 0;
      for (const auto &c : cols)
         selectedOrig += c.fComp;
      const auto totalOrig = inspector->GetCompressedSize();
      const auto restOrig = totalOrig > selectedOrig ? totalOrig - selectedOrig : 0;

      std::cout << "Selected " << cols.size() << " column" << (cols.size() == 1 ? "" : "s") << " covering "
                << FormatBytes(selectedOrig) << " / " << FormatBytes(totalOrig) << " payload ("
                << std::setprecision(1) << (totalOrig ? 100.0 * selectedOrig / totalOrig : 0) << "%).\n";
      if (opts.fPages > 0)
         std::cout << "Sampling " << opts.fPages << " page(s) per column and scaling to that column's on-disk size.\n";
      else
         std::cout << "Recoding every page of the selected columns.\n";
      std::cout << "Untouched remainder stays at " << FormatBytes(restOrig) << " in the projected file size.\n\n";

      RNTupleCompressor::SetVerifyRoundtrip(opts.fVerify);
      PageReader reader(opts.fInputFile);

      int nInflateFail = 0;
      std::vector<std::vector<InflatedPage>> inflated;
      inflated.reserve(cols.size());
      int nSamplePages = 0;
      for (const auto &c : cols) {
         auto pages = CollectPages(desc, c, opts.fPages);
         auto elem = RColumnElementBase::Generate(c.fType);
         elem->SetBitsOnStorage(c.fBits);
         std::unique_ptr<RColumnElementBase> nativeElem;
         if (!opts.fKeepColumnEncoding)
            nativeElem = MakeNativePacker(*elem);
         auto inf = InflatePages(reader, pages, *elem, !opts.fKeepColumnEncoding, nativeElem.get(), &nInflateFail);
         nSamplePages += static_cast<int>(inf.size());
         inflated.push_back(std::move(inf));
      }
      std::cout << "Pages queued: " << nSamplePages;
      if (nInflateFail)
         std::cout << "  (" << nInflateFail << " unzip failures)";
      std::cout << '\n';
      if (!opts.fKeepColumnEncoding) {
         std::cout << "Unpacking RNTuple column encodings (split/delta/zigzag) before recode.\n";
         std::cout << "Native variants re-apply ROOT split/delta/zigzag encodings before zip.\n";
      }
      if (opts.fBruteFilters)
         std::cout << "Brute filters: per page, keep the best of none/auto/"
                   << (opts.fBruteAll ? "every id except bitround" : "width-matched numeric")
                   << "; unzipped pages are reused; "
                   << ChooseThreads(opts.fThreads, static_cast<std::size_t>(-1) / 2) << " threads.\n";
      if (opts.fNoFilters)
         std::cout << "LHC4 byte filters are off.\n";

      struct VariantRow {
         std::string fLabel;
         std::uint64_t fSampleNew = 0;
         std::uint64_t fSampleOrig = 0;
         std::uint64_t fSampleUncomp = 0;
         std::uint64_t fScaledSel = 0;
         std::uint64_t fProjected = 0;
         double fSec = 0;
         int fOk = 0;
         int fFail = 0;
         std::vector<std::uint64_t> fColScaled;
      };
      std::vector<VariantRow> rows;
      struct FilterWin {
         std::uint64_t fPages = 0;
         std::uint64_t fSample = 0;
      };
      struct CodecWins {
         std::string fCodec;
         std::vector<FilterWin> fWins;
         std::uint64_t fPages = 0;
      };
      std::vector<CodecWins> bruteWins;

      auto makeRowFromZips = [&](const std::string &label,
                                 const std::vector<std::vector<std::uint64_t>> &zipByCol) {
         VariantRow row;
         row.fLabel = label;
         row.fColScaled.resize(cols.size(), 0);
         for (std::size_t i = 0; i < cols.size(); ++i) {
            if (inflated[i].empty()) {
               row.fColScaled[i] = cols[i].fComp;
               row.fScaledSel += cols[i].fComp;
               continue;
            }
            std::uint64_t sampleNew = 0;
            std::uint64_t sampleOrig = 0;
            std::uint64_t sampleUncomp = 0;
            for (std::size_t p = 0; p < inflated[i].size(); ++p) {
               sampleNew += zipByCol[i][p];
               sampleOrig += inflated[i][p].fSealed;
               sampleUncomp += inflated[i][p].fRaw.size();
               ++row.fOk;
            }
            row.fSampleNew += sampleNew;
            row.fSampleOrig += sampleOrig;
            row.fSampleUncomp += sampleUncomp;
            if (sampleOrig > 0) {
               const double scale = static_cast<double>(cols[i].fComp) / static_cast<double>(sampleOrig);
               row.fColScaled[i] = static_cast<std::uint64_t>(sampleNew * scale + 0.5);
            } else {
               row.fColScaled[i] = cols[i].fComp;
            }
            row.fScaledSel += row.fColScaled[i];
         }
         row.fProjected = row.fScaledSel + restOrig;
         return row;
      };

      auto recodeOne = [&](const ZipVariant &variant, const std::string &label, int filterId, unsigned nThreads) {
         std::vector<RecodeJob> jobs;
         for (std::size_t i = 0; i < cols.size(); ++i) {
            for (std::size_t p = 0; p < inflated[i].size(); ++p)
               jobs.push_back({0, i, p, filterId});
         }
         std::vector<std::uint64_t> jobZip(jobs.size(), 0);
         std::vector<ZipVariant> one{variant};
         const auto t0 = std::chrono::steady_clock::now();
         ZipRecodeJobs(one, cols, inflated, jobs, jobZip, nullptr, nThreads);
         std::vector<std::vector<std::uint64_t>> zipByCol(cols.size());
         for (std::size_t i = 0; i < cols.size(); ++i)
            zipByCol[i].assign(inflated[i].size(), 0);
         for (std::size_t j = 0; j < jobs.size(); ++j)
            zipByCol[jobs[j].fCol][jobs[j].fPage] = jobZip[j];
         auto row = makeRowFromZips(label, zipByCol);
         row.fSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
         std::cout << "Recoding " << label << " ... " << FormatBytes(row.fScaledSel) << " selected  ("
                   << std::setprecision(1) << std::fixed << row.fSec << " s)\n";
         rows.push_back(std::move(row));
      };

#ifdef R__HAS_LHC4CODEC
      ResetLHC4();
#endif

      if (opts.fBruteFilters) {
#ifndef R__HAS_LHC4CODEC
         throw std::runtime_error("--brute-filters requires lhc4codec");
#endif
         for (const auto &variant : variants) {
            const unsigned nPageThreads = ChooseThreads(opts.fThreads, static_cast<std::size_t>(nSamplePages));
            if (!variant.fLHC4) {
               recodeOne(variant, variant.fLabel, variant.fFilters ? -1 : 0, nPageThreads);
               continue;
            }
            recodeOne(variant, variant.fLabel + "/none", 0, nPageThreads);
            recodeOne(variant, variant.fLabel + "/auto", -1, nPageThreads);

            CodecWins wins;
            wins.fCodec = variant.fLabel;
            wins.fWins.resize(std::size(kFilterSpecs));
            struct FilterJob {
               std::size_t fCol = 0;
               std::size_t fPage = 0;
               std::size_t fFilter = 0;
            };
            std::vector<FilterJob> jobs;
            for (std::size_t i = 0; i < cols.size(); ++i) {
               for (std::size_t p = 0; p < inflated[i].size(); ++p) {
                  for (std::size_t fi = 0; fi < std::size(kFilterSpecs); ++fi) {
                     const auto &fs = kFilterSpecs[fi];
                     if (fs.fId == 35 || fs.fId == 36)
                        continue;
                     if (!FilterMatchesColumn(fs, cols[i].fBits, opts.fBruteAll))
                        continue;
                     jobs.push_back({i, p, fi});
                  }
               }
            }
            std::vector<std::uint64_t> jobZip(jobs.size(), static_cast<std::uint64_t>(-1));
            const unsigned nThreads = ChooseThreads(opts.fThreads, jobs.size());
            const auto t0 = std::chrono::steady_clock::now();
            VariantRow oracle;
            oracle.fLabel = variant.fLabel + "/oracle";
            oracle.fColScaled.resize(cols.size(), 0);
            std::cout << "Recoding " << oracle.fLabel << " (best lossless filter per page, " << nThreads
                      << " thread" << (nThreads == 1 ? "" : "s") << ") ..." << std::flush;
            std::atomic<std::size_t> next{0};
            auto worker = [&] {
               std::vector<unsigned char> zipped;
               for (;;) {
                  const std::size_t j = next.fetch_add(1);
                  if (j >= jobs.size())
                     break;
                  const auto &job = jobs[j];
                  const auto &page = inflated[job.fCol][job.fPage];
#ifdef R__HAS_LHC4CODEC
                  ApplyVariantOnThread(variant, kFilterSpecs[job.fFilter].fId, cols[job.fCol].fKind);
#endif
                  if (zipped.size() < page.fRaw.size())
                     zipped.resize(page.fRaw.size());
                  jobZip[j] = RNTupleCompressor::Zip(page.fRaw.data(), page.fRaw.size(), variant.fCompression,
                                                     zipped.data());
               }
#ifdef R__HAS_LHC4CODEC
               ClearCallOverrides();
#endif
            };
            if (nThreads <= 1) {
               worker();
            } else {
               std::vector<std::thread> pool;
               pool.reserve(nThreads);
               for (unsigned t = 0; t < nThreads; ++t)
                  pool.emplace_back(worker);
               for (auto &th : pool)
                  th.join();
            }
            oracle.fOk = static_cast<int>(jobs.size());
            std::vector<std::vector<std::uint64_t>> bestZip(cols.size());
            std::vector<std::vector<std::size_t>> bestIdx(cols.size());
            for (std::size_t i = 0; i < cols.size(); ++i) {
               bestZip[i].assign(inflated[i].size(), static_cast<std::uint64_t>(-1));
               bestIdx[i].assign(inflated[i].size(), 0);
            }
            for (std::size_t j = 0; j < jobs.size(); ++j) {
               const auto &job = jobs[j];
               if (jobZip[j] < bestZip[job.fCol][job.fPage]) {
                  bestZip[job.fCol][job.fPage] = jobZip[j];
                  bestIdx[job.fCol][job.fPage] = job.fFilter;
               }
            }
            for (std::size_t i = 0; i < cols.size(); ++i) {
               if (inflated[i].empty()) {
                  oracle.fColScaled[i] = cols[i].fComp;
                  oracle.fScaledSel += cols[i].fComp;
                  continue;
               }
               std::uint64_t sampleNew = 0;
               std::uint64_t sampleOrig = 0;
               std::uint64_t sampleUncomp = 0;
               for (std::size_t p = 0; p < inflated[i].size(); ++p) {
                  sampleNew += bestZip[i][p];
                  sampleOrig += inflated[i][p].fSealed;
                  sampleUncomp += inflated[i][p].fRaw.size();
                  wins.fWins[bestIdx[i][p]].fPages++;
                  wins.fWins[bestIdx[i][p]].fSample += bestZip[i][p];
                  wins.fPages++;
               }
               oracle.fSampleNew += sampleNew;
               oracle.fSampleOrig += sampleOrig;
               oracle.fSampleUncomp += sampleUncomp;
               if (sampleOrig > 0) {
                  const double scale = static_cast<double>(cols[i].fComp) / static_cast<double>(sampleOrig);
                  oracle.fColScaled[i] = static_cast<std::uint64_t>(sampleNew * scale + 0.5);
               } else {
                  oracle.fColScaled[i] = cols[i].fComp;
               }
               oracle.fScaledSel += oracle.fColScaled[i];
            }
            oracle.fProjected = oracle.fScaledSel + restOrig;
            oracle.fSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::cout << ' ' << FormatBytes(oracle.fScaledSel) << " selected  (" << std::setprecision(1) << std::fixed
                      << oracle.fSec << " s)\n";
            rows.push_back(std::move(oracle));
            bruteWins.push_back(std::move(wins));
         }
      } else {
         std::vector<RecodeJob> jobs;
         jobs.reserve(static_cast<std::size_t>(nSamplePages) * variants.size());
         std::vector<std::vector<std::vector<std::uint64_t>>> zipByVar(variants.size());
         for (std::size_t vi = 0; vi < variants.size(); ++vi) {
            const int filterId = variants[vi].fFilters ? -1 : 0;
            zipByVar[vi].resize(cols.size());
            for (std::size_t i = 0; i < cols.size(); ++i) {
               zipByVar[vi][i].assign(inflated[i].size(), 0);
               for (std::size_t p = 0; p < inflated[i].size(); ++p)
                  jobs.push_back({vi, i, p, filterId});
            }
         }
         const unsigned nThreads = ChooseThreads(opts.fThreads, jobs.size());
         auto cpuNs = std::make_unique<std::atomic<std::uint64_t>[]>(variants.size());
         for (std::size_t vi = 0; vi < variants.size(); ++vi)
            cpuNs[vi].store(0);
         std::cout << "Recoding " << variants.size() << " codec" << (variants.size() == 1 ? "" : "s") << " × "
                   << nSamplePages << " page" << (nSamplePages == 1 ? "" : "s") << " on " << nThreads << " thread"
                   << (nThreads == 1 ? "" : "s") << " ..." << std::flush;
         std::vector<std::uint64_t> jobZip(jobs.size(), 0);
         const auto t0 = std::chrono::steady_clock::now();
         ZipRecodeJobs(variants, cols, inflated, jobs, jobZip, cpuNs.get(), nThreads);
         const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
         std::cout << ' ' << std::setprecision(1) << std::fixed << wall << " s wall\n";
         for (std::size_t j = 0; j < jobs.size(); ++j) {
            const auto &job = jobs[j];
            zipByVar[job.fVariant][job.fCol][job.fPage] = jobZip[j];
         }
         for (std::size_t vi = 0; vi < variants.size(); ++vi) {
            auto row = makeRowFromZips(variants[vi].fLabel, zipByVar[vi]);
            row.fSec = static_cast<double>(cpuNs[vi].load()) / 1e9;
            std::cout << "  " << row.fLabel << " ... " << FormatBytes(row.fScaledSel) << " selected  ("
                      << std::setprecision(1) << std::fixed << row.fSec << " s CPU)\n";
            rows.push_back(std::move(row));
         }
      }
#ifdef R__HAS_LHC4CODEC
      ResetLHC4();
#endif

      std::cout << "\n=== projected Events payload ===\n";
      const int nameW = opts.fBruteFilters ? 36 : 22;
      std::cout << std::left << std::setw(nameW) << "codec" << std::right << std::setw(14) << "selected"
                << std::setw(10) << "vs orig" << std::setw(14) << "file" << std::setw(10) << "vs file"
                << std::setw(10) << "sec" << '\n';
      auto printRow = [&](const std::string &label, std::uint64_t selected, std::uint64_t file, double sec, bool timed) {
         const double vsSel = selectedOrig ? 100.0 * (static_cast<double>(selected) / selectedOrig - 1.0) : 0;
         const double vsFile = totalOrig ? 100.0 * (static_cast<double>(file) / totalOrig - 1.0) : 0;
         std::cout << std::left << std::setw(nameW) << label << std::right << std::setw(14) << FormatBytes(selected)
                   << std::setw(9) << std::fixed << std::setprecision(1) << vsSel << "%" << std::setw(14)
                   << FormatBytes(file) << std::setw(9) << vsFile << "%";
         if (timed)
            std::cout << std::setw(10) << std::setprecision(1) << sec;
         else
            std::cout << std::setw(10) << "-";
         std::cout << '\n';
      };
      printRow("on-disk", selectedOrig, totalOrig, 0, false);
      for (const auto &r : rows)
         printRow(r.fLabel, r.fScaledSel, r.fProjected, r.fSec, true);

      for (const auto &cw : bruteWins) {
         std::vector<std::size_t> order;
         for (std::size_t fi = 0; fi < cw.fWins.size(); ++fi) {
            if (cw.fWins[fi].fPages)
               order.push_back(fi);
         }
         std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return cw.fWins[a].fSample > cw.fWins[b].fSample;
         });
         std::cout << "\n=== " << cw.fCodec << ": winning lossless filter per page ===\n";
         std::cout << std::left << std::setw(28) << "filter" << std::right << std::setw(10) << "pages"
                   << std::setw(14) << "sample" << '\n';
         for (std::size_t fi : order) {
            std::cout << std::left << std::setw(28) << kFilterSpecs[fi].fName << std::right << std::setw(10)
                      << cw.fWins[fi].fPages << std::setw(14) << FormatBytes(cw.fWins[fi].fSample) << '\n';
         }
         std::cout << std::left << std::setw(28) << "sampled pages" << std::right << std::setw(10) << cw.fPages
                   << '\n';
      }

      int nFail = 0, nOk = 0;
      for (const auto &r : rows) {
         nFail += r.fFail;
         nOk += r.fOk;
      }
      if (nFail)
         std::cout << "\nSkipped " << nFail << " page(s) that failed to unzip (" << nOk << " ok).\n";

      std::vector<const VariantRow *> colTable;
      for (const auto &r : rows) {
         if (!opts.fBruteFilters) {
            colTable.push_back(&r);
            continue;
         }
         if (r.fLabel.find("/none") != std::string::npos || r.fLabel.find("/auto") != std::string::npos ||
             r.fLabel.find("/oracle") != std::string::npos)
            colTable.push_back(&r);
      }
      std::cout << "\n=== selected columns (scaled) ===\n";
      std::cout << std::left << std::setw(42) << "field" << std::setw(14) << "type" << std::right << std::setw(12)
                << "on disk";
      for (const auto *r : colTable)
         std::cout << std::setw(16) << r->fLabel;
      std::cout << '\n';
      for (std::size_t i = 0; i < cols.size(); ++i) {
         std::cout << std::left << std::setw(42) << ShortField(cols[i].fField) << std::setw(14) << cols[i].fTypeName
                   << std::right << std::setw(12) << FormatBytes(cols[i].fComp);
         for (const auto *r : colTable)
            std::cout << std::setw(16) << FormatBytes(r->fColScaled[i]);
         std::cout << '\n';
      }

      if (opts.fCsv) {
         std::cout << "\nfield,type,kind,on_disk";
         for (const auto &r : rows)
            std::cout << ',' << r.fLabel;
         std::cout << '\n';
         for (std::size_t i = 0; i < cols.size(); ++i) {
            std::cout << '"' << cols[i].fField << '"' << ',' << cols[i].fTypeName << ',' << KindName(cols[i].fKind)
                      << ',' << cols[i].fComp;
            for (const auto &r : rows)
               std::cout << ',' << r.fColScaled[i];
            std::cout << '\n';
         }
      }
      return 0;
   } catch (const std::exception &e) {
      std::cerr << "rntupletryzip: " << e.what() << '\n';
      std::cerr << kUsage;
      return 1;
   }
}
