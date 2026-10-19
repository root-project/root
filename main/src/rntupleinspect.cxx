/// \file rntupleinspect.cxx
/// \brief Judge RNTuple column formats and remaining compressor headroom.

#include <ROOT/RColumnElementBase.hxx>
#include <ROOT/RNTupleInspector.hxx>
#include <ROOT/RNTupleZip.hxx>

#include <TError.h>
#include <TFile.h>
#include <TKey.h>
#include <TSystem.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

using ROOT::Experimental::RNTupleInspector;
using ROOT::Internal::RColumnElementBase;
using ROOT::Internal::RNTupleDecompressor;

constexpr const char *kUsage = R"(Usage: rntupleinspect [options] input.root [ntupleName]

Inspect an RNTuple's on-disk column types and estimate whether a stronger
page compressor (LHC4 Crystal/Laser/Grating, or a higher zstd level) can
still shrink the file.

The tool samples packed pages, measures byte-plane entropy, and compares
that Shannon limit with the size already stored by the file's compressor.
High-entropy SplitReal32 mantissa planes are a floor: new codecs will not
move them. Integer and offset columns are often already spent.

Options:
  --top N         Rank the N largest on-disk columns (default: 20).
  --pages N       Sample up to N pages per ranked column (default: 2).
  --list          List RNTuples in the file and exit.
  --csv           Also print a CSV of per-column verdicts.
  --verbose       Keep TClass dictionary warnings (CMSSW MiniAOD files emit many).
  -h, --help      Show this help.

If ntupleName is omitted, Events is used when present, otherwise the RNTuple
with the largest on-disk payload. CMSSW files often also contain MetaData.
Dictionary warnings are suppressed unless --verbose is passed; they do not
block page inspection. To actually recompress pages without CMSSW dictionaries,
use rntupletryzip.
)";

enum class Kind { kFloat, kInt, kIndex, kBit, kText, kOther };

enum class Headroom { kNone, kLow, kMedium, kHigh, kUnknown };

struct RunOptions {
   std::string fInputFile;
   std::string fNTupleName;
   int fTop = 20;
   int fPages = 2;
   bool fCsv = false;
   bool fList = false;
   bool fVerbose = false;
};

struct ColRow {
   ROOT::DescriptorId_t fId = 0;
   std::string fField;
   std::string fTypeName;
   Kind fKind = Kind::kOther;
   bool fSplit = false;
   std::uint16_t fBits = 0;
   std::uint64_t fNElem = 0;
   std::uint64_t fNPages = 0;
   std::uint64_t fComp = 0;
   std::uint64_t fUncomp = 0;
   int fSampledPages = 0;
   double fEntropy = -1;     ///< mean packed-byte entropy, bits/byte
   double fLsbEntropy = -1;  ///< least-significant byte plane
   double fMsbEntropy = -1;  ///< most-significant byte plane
   double fHeadroom = -1;    ///< residual vs 0th-order plane Shannon, as fraction of packed bytes
   Headroom fVerdict = Headroom::kUnknown;
   std::string fAdvice;
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

const char *HeadroomName(Headroom h)
{
   switch (h) {
   case Headroom::kNone: return "none";
   case Headroom::kLow: return "low";
   case Headroom::kMedium: return "medium";
   case Headroom::kHigh: return "high";
   default: return "n/a";
   }
}

bool IsSplitType(ROOT::ENTupleColumnType t)
{
   using T = ROOT::ENTupleColumnType;
   switch (t) {
   case T::kSplitIndex64:
   case T::kSplitIndex32:
   case T::kSplitReal64:
   case T::kSplitReal32:
   case T::kSplitInt64:
   case T::kSplitUInt64:
   case T::kSplitInt32:
   case T::kSplitUInt32:
   case T::kSplitInt16:
   case T::kSplitUInt16: return true;
   default: return false;
   }
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

struct NTupleInfo {
   std::string fName;
   ROOT::NTupleSize_t fNEntries = 0;
   std::uint64_t fComp = 0;
   std::uint64_t fUncomp = 0;
   std::size_t fNColumns = 0;
};

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

std::string FormatBytes(std::uint64_t n);

void PrintRNTupleCatalog(const std::vector<NTupleInfo> &infos, const std::string &selected)
{
   if (infos.empty())
      return;
   std::cout << "RNTuples in file:\n";
   for (const auto &info : infos) {
      const char *mark = (info.fName == selected) ? "  * " : "    ";
      std::cout << mark << std::left << std::setw(22) << info.fName << std::right << std::setw(10) << info.fNEntries
                << " entries  " << std::setw(12) << FormatBytes(info.fComp) << " payload  " << info.fNColumns
                << " columns\n";
   }
   if (infos.size() > 1 && !selected.empty())
      std::cout << "  (* selected; pass a name to inspect another, e.g. MetaData)\n";
   std::cout << '\n';
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

double Entropy(const unsigned char *p, std::size_t n)
{
   if (n == 0)
      return 0;
   std::size_t hist[256] = {};
   for (std::size_t i = 0; i < n; ++i)
      hist[p[i]]++;
   double e = 0;
   for (int i = 0; i < 256; ++i) {
      if (!hist[i])
         continue;
      const double q = static_cast<double>(hist[i]) / static_cast<double>(n);
      e -= q * std::log2(q);
   }
   return e;
}

double StridedEntropy(const unsigned char *p, std::size_t n, int plane, int width)
{
   if (width <= 0 || n < static_cast<std::size_t>(width))
      return 0;
   const std::size_t nElem = n / static_cast<std::size_t>(width);
   std::vector<unsigned char> planeBytes(nElem);
   for (std::size_t i = 0; i < nElem; ++i)
      planeBytes[i] = p[i * static_cast<std::size_t>(width) + static_cast<std::size_t>(plane)];
   return Entropy(planeBytes.data(), planeBytes.size());
}

Headroom HeadroomFromFrac(double frac)
{
   if (frac < 0.03)
      return Headroom::kNone;
   if (frac < 0.10)
      return Headroom::kLow;
   if (frac < 0.25)
      return Headroom::kMedium;
   return Headroom::kHigh;
}

std::string Advise(const ColRow &c)
{
   const double ratio = (c.fComp > 0) ? static_cast<double>(c.fUncomp) / static_cast<double>(c.fComp) : 0;

   if (c.fKind == Kind::kFloat) {
      if (c.fVerdict == Headroom::kNone && c.fLsbEntropy > 7.5)
         return "mantissa floor; LHC4 will not shrink this";
      if (c.fVerdict == Headroom::kNone)
         return "already at split+zstd floor";
      if (c.fVerdict == Headroom::kLow)
         return "Crystal float profile, maybe a few %";
      return "Crystal / Laser on float pages";
   }
   if (c.fKind == Kind::kInt) {
      if (c.fVerdict == Headroom::kNone && c.fEntropy > 7.0)
         return "unique integers; not compressible";
      if (c.fComp < 4096 || ratio > 8 || c.fVerdict == Headroom::kNone)
         return "already packed; ignore for file size";
      if (c.fVerdict == Headroom::kHigh || (c.fMsbEntropy >= 0 && c.fMsbEntropy < 1.0))
         return "Grating or Crystal poisson";
      return "Grating / Crystal on integer pages";
   }
   if (c.fKind == Kind::kIndex) {
      if (ratio > 20)
         return "offset columns already spent";
      return "keep split index; zstd/delta is enough";
   }
   if (c.fKind == Kind::kText)
      return "Prism / Mosaic / BWT";
   if (c.fKind == Kind::kBit)
      return "bit columns already packed";
   return "measure with --auto-codecs all";
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
      if (arg == "--csv") {
         opts.fCsv = true;
         continue;
      }
      if (arg == "--list") {
         opts.fList = true;
         continue;
      }
      if (arg == "--verbose" || arg == "-v") {
         opts.fVerbose = true;
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
      if (!arg.empty() && arg[0] == '-')
         throw std::runtime_error("unknown option: " + arg);
      positional.push_back(arg);
   }
   if (positional.empty())
      throw std::runtime_error("missing input file (see --help)");
   if (positional.size() > 2)
      throw std::runtime_error("too many positional arguments (see --help)");
   opts.fInputFile = positional[0];
   if (positional.size() == 2)
      opts.fNTupleName = positional[1];
   if (opts.fTop < 1)
      throw std::runtime_error("--top must be >= 1");
   if (opts.fPages < 1)
      throw std::runtime_error("--pages must be >= 1");
   return opts;
}

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

void SampleColumn(ColRow &col, const ROOT::RNTupleDescriptor &desc, PageReader &reader, int maxPages)
{
   const int width = (col.fBits >= 8 && (col.fBits % 8) == 0) ? col.fBits / 8 : 0;
   double hSum = 0;
   double lsbSum = 0;
   double msbSum = 0;
   double packedSum = 0;
   double sealedSum = 0;
   int nPages = 0;
   int nPlane = 0;

   for (const auto &clusterDesc : desc.GetClusterIterable()) {
      if (nPages >= maxPages)
         break;
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
         if (nPages >= maxPages)
            break;
         const auto &loc = pageInfo.GetLocator();
         if (loc.GetType() == ROOT::RNTupleLocator::kTypePageZero)
            continue;
         const auto sealedBytes = loc.GetNBytesOnStorage();
         const auto packedBytes =
            (static_cast<std::size_t>(col.fBits) * pageInfo.GetNElements() + 7) / 8;
         if (sealedBytes == 0 || packedBytes == 0)
            continue;

         std::vector<unsigned char> sealed(sealedBytes);
         if (!reader.Read(loc.GetPosition<std::uint64_t>(), sealed.data(), sealedBytes))
            continue;
         std::vector<unsigned char> raw(packedBytes);
         try {
            RNTupleDecompressor::Unzip(sealed.data(), sealed.size(), packedBytes, raw.data());
         } catch (const std::exception &) {
            continue;
         }

         packedSum += static_cast<double>(packedBytes);
         sealedSum += static_cast<double>(sealedBytes);
         nPages++;

         double h = Entropy(raw.data(), raw.size());
         if (width >= 2 && width <= 8) {
            const auto nElem = pageInfo.GetNElements();
            if (col.fSplit && nElem > 0 && packedBytes >= nElem * static_cast<std::size_t>(width)) {
               double planeH = 0;
               for (int p = 0; p < width; ++p)
                  planeH += Entropy(raw.data() + static_cast<std::size_t>(p) * nElem, nElem);
               h = planeH / width;
               lsbSum += Entropy(raw.data(), nElem);
               msbSum += Entropy(raw.data() + static_cast<std::size_t>(width - 1) * nElem, nElem);
            } else {
               lsbSum += StridedEntropy(raw.data(), raw.size(), 0, width);
               msbSum += StridedEntropy(raw.data(), raw.size(), width - 1, width);
            }
            nPlane++;
         }
         hSum += h;
      }
   }

   col.fSampledPages = nPages;
   if (nPages == 0)
      return;

   col.fEntropy = hSum / nPages;
   if (nPlane > 0) {
      col.fLsbEntropy = lsbSum / nPlane;
      col.fMsbEntropy = msbSum / nPlane;
   }

   const double shannon = packedSum * col.fEntropy / 8.0;
   const double residual = std::max(0.0, sealedSum - shannon);
   col.fHeadroom = (packedSum > 0) ? residual / packedSum : 0;

   // Mantissa-limited floats: LSB plane is noise. Treat as no residual even if
   // 0th-order Shannon is slightly below the sealed size.
   if (col.fKind == Kind::kFloat && col.fLsbEntropy > 7.5 && col.fHeadroom < 0.08)
      col.fHeadroom = 0;

   col.fVerdict = HeadroomFromFrac(col.fHeadroom);
   if (col.fComp < 2048)
      col.fVerdict = Headroom::kNone;
}

} // namespace

int main(int argc, char **argv)
{
   try {
      auto opts = ParseArgs(argc, argv);
      FileStat_t st{};
      if (gSystem->GetPathInfo(opts.fInputFile.c_str(), st) != 0)
         throw std::runtime_error("cannot stat " + opts.fInputFile);

      std::unique_ptr<IgnoreDictWarnings> quiet;
      if (!opts.fVerbose)
         quiet = std::make_unique<IgnoreDictWarnings>();

      const auto catalog = DescribeAllRNTuples(opts.fInputFile);
      if (opts.fList) {
         std::cout << "File:     " << opts.fInputFile << " (" << FormatBytes(st.fSize) << ")\n";
         PrintRNTupleCatalog(catalog, "");
         return 0;
      }
      opts.fNTupleName = SelectRNTuple(catalog, opts.fNTupleName);

      auto inspector = RNTupleInspector::Create(opts.fNTupleName, opts.fInputFile);
      const auto &desc = inspector->GetDescriptor();

      std::cout << "File:     " << opts.fInputFile << " (" << FormatBytes(st.fSize) << ")\n";
      PrintRNTupleCatalog(catalog, opts.fNTupleName);
      std::cout << "RNTuple:  " << opts.fNTupleName << '\n';
      std::cout << "Entries:  " << desc.GetNEntries() << "   clusters: " << desc.GetNClusters()
                << "   columns: " << desc.GetNPhysicalColumns() << '\n';
      std::cout << "Pages:    " << inspector->GetCompressionSettingsAsString() << "   payload "
                << FormatBytes(inspector->GetCompressedSize()) << " / "
                << FormatBytes(inspector->GetUncompressedSize()) << " uncompressed  (" << std::fixed
                << std::setprecision(2) << inspector->GetCompressionFactor() << "x)\n\n";

      std::cout << "=== on-disk column types ===\n";
      inspector->PrintColumnTypeInfo();
      std::cout << '\n';

      std::vector<ColRow> cols;
      struct KindAcc {
         int count = 0;
         std::uint64_t nPages = 0;
         std::uint64_t comp = 0;
         std::uint64_t uncomp = 0;
      };
      KindAcc byKind[6]{};

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
         row.fSplit = IsSplitType(colDesc.GetType());
         row.fBits = colDesc.GetBitsOnStorage();
         row.fNElem = ci.GetNElements();
         row.fNPages = ci.GetNPages();
         row.fComp = ci.GetCompressedSize();
         row.fUncomp = ci.GetUncompressedSize();
         cols.push_back(std::move(row));
         auto &acc = byKind[static_cast<int>(cols.back().fKind)];
         acc.count++;
         acc.nPages += cols.back().fNPages;
         acc.comp += cols.back().fComp;
         acc.uncomp += cols.back().fUncomp;
      }

      const auto totComp = inspector->GetCompressedSize();
      const auto totUncomp = inspector->GetUncompressedSize();
      std::cout << "=== payload by page kind ===\n";
      std::cout << std::left << std::setw(8) << "kind" << std::right << std::setw(6) << "cols" << std::setw(12)
                << "uncomp" << std::setw(10) << "%raw" << std::setw(12) << "on disk" << std::setw(10) << "%file"
                << std::setw(9) << "ratio" << '\n';
      for (auto k : {Kind::kFloat, Kind::kInt, Kind::kIndex, Kind::kBit, Kind::kText, Kind::kOther}) {
         const auto &a = byKind[static_cast<int>(k)];
         if (a.count == 0)
            continue;
         const double pu = totUncomp ? 100.0 * a.uncomp / totUncomp : 0;
         const double pc = totComp ? 100.0 * a.comp / totComp : 0;
         const double ratio = a.comp ? static_cast<double>(a.uncomp) / a.comp : 0;
         std::cout << std::left << std::setw(8) << KindName(k) << std::right << std::setw(6) << a.count << std::setw(12)
                   << FormatBytes(a.uncomp) << std::setw(9) << std::fixed << std::setprecision(1) << pu << "%"
                   << std::setw(12) << FormatBytes(a.comp) << std::setw(9) << pc << "%" << std::setw(9)
                   << std::setprecision(2) << ratio << '\n';
      }
      std::cout << '\n';

      std::sort(cols.begin(), cols.end(), [](const ColRow &a, const ColRow &b) { return a.fComp > b.fComp; });

      PageReader pages(opts.fInputFile);
      const int nRank = std::min(opts.fTop, static_cast<int>(cols.size()));
      std::unordered_set<ROOT::DescriptorId_t> sampled;
      for (int i = 0; i < nRank; ++i) {
         SampleColumn(cols[i], desc, pages, opts.fPages);
         sampled.insert(cols[i].fId);
      }
      // Always sample the heaviest column of each kind so the verdict is not
      // blinded when the top-N are all the same type.
      for (auto k : {Kind::kFloat, Kind::kInt, Kind::kIndex, Kind::kBit, Kind::kText}) {
         for (auto &c : cols) {
            if (c.fKind != k || sampled.count(c.fId))
               continue;
            SampleColumn(c, desc, pages, opts.fPages);
            sampled.insert(c.fId);
            break;
         }
      }

      for (auto &c : cols)
         c.fAdvice = Advise(c);

      std::uint64_t exhaustedComp = 0;
      std::uint64_t judgedComp = 0;
      int nNone = 0, nLow = 0, nMed = 0, nHigh = 0;
      for (const auto &c : cols) {
         if (c.fVerdict == Headroom::kUnknown)
            continue;
         judgedComp += c.fComp;
         if (c.fVerdict == Headroom::kNone) {
            exhaustedComp += c.fComp;
            nNone++;
         } else if (c.fVerdict == Headroom::kLow)
            nLow++;
         else if (c.fVerdict == Headroom::kMedium)
            nMed++;
         else
            nHigh++;
      }

      std::cout << "=== compressor potential ===\n";
      if (judgedComp == 0) {
         std::cout << "No pages could be sampled.\n";
      } else {
         const double spentPct = totComp ? 100.0 * exhaustedComp / totComp : 0;
         std::cout << "Sampled " << sampled.size() << " columns (" << nNone << " none, " << nLow << " low, " << nMed
                   << " medium, " << nHigh << " high headroom).\n";
         std::cout << "On-disk bytes already at the entropy floor: " << FormatBytes(exhaustedComp) << " ("
                   << std::setprecision(1) << std::fixed << spentPct << "% of payload).\n";
         if (spentPct >= 75 && byKind[static_cast<int>(Kind::kFloat)].comp > totComp / 2) {
            std::cout << "Verdict: little headroom. File size is dominated by high-entropy "
                         "SplitReal32 kinematics; Laser/Crystal/Grating will not move it.\n";
         } else if (nHigh > 0 && byKind[static_cast<int>(Kind::kInt)].comp > totComp * 0.15) {
            std::cout << "Verdict: integer pages still have room. Try Grating or Crystal poisson "
                         "on the int columns; leave the float kinematics alone.\n";
         } else if (nMed + nHigh > 0) {
            std::cout << "Verdict: mixed. Measure with rntuple2rntuple --auto-codecs all; "
                         "expect gains only on the medium/high columns below.\n";
         } else {
            std::cout << "Verdict: remaining columns are already well packed. A new codec is "
                         "unlikely to beat the current page compressor.\n";
         }
      }
      std::cout << '\n';

      std::cout << "=== largest on-disk columns ===\n";
      std::cout << std::left << std::setw(36) << "field" << std::setw(14) << "type" << std::right << std::setw(11)
                << "on disk" << std::setw(7) << "ratio" << std::setw(6) << "H" << std::setw(8) << "room"
                << std::left << "  advice" << '\n';
      for (int i = 0; i < nRank; ++i) {
         const auto &c = cols[i];
         const double ratio = c.fComp ? static_cast<double>(c.fUncomp) / c.fComp : 0;
         std::string name = c.fField;
         if (name.size() > 35)
            name = name.substr(0, 32) + "...";
         std::cout << std::left << std::setw(36) << name << std::setw(14) << c.fTypeName << std::right << std::setw(11)
                   << FormatBytes(c.fComp) << std::setw(7) << std::setprecision(2) << std::fixed << ratio
                   << std::setw(6);
         if (c.fEntropy >= 0)
            std::cout << std::setprecision(2) << c.fEntropy;
         else
            std::cout << "  -";
         std::cout << std::setw(8) << HeadroomName(c.fVerdict) << "  " << std::left << c.fAdvice << '\n';
      }

      if (opts.fCsv) {
         std::cout << "\nfield,colType,kind,bits,nPages,uncomp,comp,ratio,entropy,lsb,msb,headroom,verdict,advice\n";
         for (int i = 0; i < nRank; ++i) {
            const auto &c = cols[i];
            const double ratio = c.fComp ? static_cast<double>(c.fUncomp) / c.fComp : 0;
            std::cout << c.fField << ',' << c.fTypeName << ',' << KindName(c.fKind) << ',' << c.fBits << ','
                      << c.fNPages << ',' << c.fUncomp << ',' << c.fComp << ',' << std::setprecision(4) << ratio << ','
                      << c.fEntropy << ',' << c.fLsbEntropy << ',' << c.fMsbEntropy << ',' << c.fHeadroom << ','
                      << HeadroomName(c.fVerdict) << ',' << '"' << c.fAdvice << '"' << '\n';
         }
      }
      return 0;
   } catch (const std::exception &e) {
      std::cerr << "rntupleinspect: " << e.what() << '\n';
      std::cerr << kUsage;
      return 1;
   }
}
