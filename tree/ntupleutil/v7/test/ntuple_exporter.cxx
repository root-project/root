#include <ROOT/RNTupleExporter.hxx>

#include <ROOT/RNTupleModel.hxx>

#include "ntupleutil_test.hxx"
#include <filesystem>

using ROOT::Internal::RPageSource;
using namespace ROOT::Experimental;
using ROOT::Experimental::Internal::RNTupleExporter;

namespace {

std::string ReadFileToString(const char *fname)
{
   FILE *f = fopen(fname, "rb");
   if (!f) {
      R__LOG_ERROR() << "failed to open file " << fname;
      return "";
   }
   fseek(f, 0, SEEK_END);
   size_t size = ftell(f);
   fseek(f, 0, SEEK_SET);
   std::string str;
   str.resize(size);
   size_t bytesRead = fread(str.data(), 1, size, f);
   R__ASSERT(bytesRead == size);
   fclose(f);
   return str;
}

enum : bool {
   kWithoutChecksums = false,
   kWithChecksums = true,
};

void CreateExportRNTuple(std::string_view fileName, bool checksums)
{
   auto model = ROOT::RNTupleModel::Create();
   auto pFlt = model->MakeField<float>("flt");
   auto pVec = model->MakeField<std::vector<int>>("vec");

   auto opts = ROOT::RNTupleWriteOptions();
   opts.SetCompression(0);
   opts.SetEnablePageChecksums(checksums);
   auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntuple", fileName, opts);
   for (int i = 0; i < 10; ++i) {
      *pFlt = i;
      pVec->clear();
      for (int j = -3; j < 3; ++j)
         pVec->emplace_back(i - j);
      writer->Fill();
   }
}

// These are the expected contents of the pages of the RNTuple created by CreateExportRNTuple (including the checksums)
const char kVecBytes[] =
   "\x03\x00\x00\x00\x02\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\xff\xfe\xff\xff\xff\x04\x00\x00\x00"
   "\x03\x00\x00\x00\x02\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\xff\x05\x00\x00\x00\x04\x00\x00\x00"
   "\x03\x00\x00\x00\x02\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x05\x00\x00\x00\x04\x00\x00\x00"
   "\x03\x00\x00\x00\x02\x00\x00\x00\x01\x00\x00\x00\x07\x00\x00\x00\x06\x00\x00\x00\x05\x00\x00\x00\x04\x00\x00\x00"
   "\x03\x00\x00\x00\x02\x00\x00\x00\x08\x00\x00\x00\x07\x00\x00\x00\x06\x00\x00\x00\x05\x00\x00\x00\x04\x00\x00\x00"
   "\x03\x00\x00\x00\x09\x00\x00\x00\x08\x00\x00\x00\x07\x00\x00\x00\x06\x00\x00\x00\x05\x00\x00\x00\x04\x00\x00\x00"
   "\x0a\x00\x00\x00\x09\x00\x00\x00\x08\x00\x00\x00\x07\x00\x00\x00\x06\x00\x00\x00\x05\x00\x00\x00\x0b\x00\x00\x00"
   "\x0a\x00\x00\x00\x09\x00\x00\x00\x08\x00\x00\x00\x07\x00\x00\x00\x06\x00\x00\x00\x0c\x00\x00\x00\x0b\x00\x00\x00"
   "\x0a\x00\x00\x00\x09\x00\x00\x00\x08\x00\x00\x00\x07\x00\x00\x00\x56\x4a\xe0\x51\xf5\x0b\xfc\x61";
const char kVecIdxBytes[] =
   "\x06\x00\x00\x00\x00\x00\x00\x00\x0c\x00\x00\x00\x00\x00\x00\x00\x12\x00\x00\x00\x00\x00\x00\x00\x18\x00\x00\x00"
   "\x00\x00\x00\x00\x1e\x00\x00\x00\x00\x00\x00\x00\x24\x00\x00\x00\x00\x00\x00\x00\x2a\x00\x00\x00\x00\x00\x00\x00"
   "\x30\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00\x00\x00\x00\x00\x3c\x00\x00\x00\x00\x00\x00\x00\xaa\xde\x1b\xde"
   "\xb5\xf3\xbc\x1a";
const char kFltBytes[] =
   "\x00\x00\x00\x00\x00\x00\x80\x3f\x00\x00\x00\x40\x00\x00\x40\x40\x00\x00\x80\x40\x00\x00\xa0\x40\x00\x00\xc0"
   "\x40"
   "\x00\x00\xe0\x40\x00\x00\x00\x41\x00\x00\x10\x41\x1b\xad\x67\xa6\xe6\x61\x56\x9d";

constexpr auto kVecBytesLenChecksums = std::size(kVecBytes) - 1;
constexpr auto kVecIdxBytesLenChecksums = std::size(kVecIdxBytes) - 1;
constexpr auto kFltBytesLenChecksums = std::size(kFltBytes) - 1;
constexpr auto kVecBytesLenNoChecksums = kVecBytesLenChecksums - 8;
constexpr auto kVecIdxBytesLenNoChecksums = kVecIdxBytesLenChecksums - 8;
constexpr auto kFltBytesLenNoChecksums = kFltBytesLenChecksums - 8;

} // namespace

TEST(RNTupleExporter, ExportToFiles)
{
   // Dump pages of a regular RNTuple with checksums. Should not include checksums in the output.
   FileRaii fileGuard("ntuple_exporter.root");

   // Create RNTuple to export
   CreateExportRNTuple(fileGuard.GetPath(), kWithChecksums);

   // Now export the pages
   auto source = RPageSource::Create("ntuple", fileGuard.GetPath());
   auto res = RNTupleExporter::ExportPages(*source);

   EXPECT_EQ(res.fExportedFileNames.size(), 3);

   FileRaii pageVecIdx("./cluster_0_vec-0_page_0_elems_10_comp_0.page");
   FileRaii pageVec("./cluster_0_vec._0-0_page_0_elems_60_comp_0.page");
   FileRaii pageFlt("./cluster_0_flt-0_page_0_elems_10_comp_0.page");

   EXPECT_TRUE(std::find(res.fExportedFileNames.begin(), res.fExportedFileNames.end(), pageFlt.GetPath()) !=
               res.fExportedFileNames.end());
   EXPECT_TRUE(std::find(res.fExportedFileNames.begin(), res.fExportedFileNames.end(), pageVec.GetPath()) !=
               res.fExportedFileNames.end());
   EXPECT_TRUE(std::find(res.fExportedFileNames.begin(), res.fExportedFileNames.end(), pageVecIdx.GetPath()) !=
               res.fExportedFileNames.end());

   // check the file contents
   auto fltBytes = ReadFileToString(pageFlt.GetPath().c_str());
   EXPECT_EQ(fltBytes.length(), kFltBytesLenNoChecksums);
   EXPECT_EQ(memcmp(fltBytes.data(), kFltBytes, fltBytes.length()), 0);

   auto vecIdxBytes = ReadFileToString(pageVecIdx.GetPath().c_str());
   EXPECT_EQ(vecIdxBytes.length(), kVecIdxBytesLenNoChecksums);
   EXPECT_EQ(memcmp(vecIdxBytes.data(), kVecIdxBytes, vecIdxBytes.length()), 0);

   auto vecBytes = ReadFileToString(pageVec.GetPath().c_str());
   EXPECT_EQ(vecBytes.length(), kVecBytesLenNoChecksums);
   EXPECT_EQ(memcmp(vecBytes.data(), kVecBytes, vecBytes.length()), 0);
}

TEST(RNTupleExporter, ExportToFilesWithChecksum)
{
   // Dump pages of a regular RNTuple with checksums. Should include checksums in the output.
   FileRaii fileGuard("ntuple_exporter_chk.root");

   // Create RNTuple to export
   CreateExportRNTuple(fileGuard.GetPath(), kWithChecksums);

   // Now export the pages
   auto source = RPageSource::Create("ntuple", fileGuard.GetPath());
   auto opts = RNTupleExporter::RPagesOptions();
   opts.fFlags |= RNTupleExporter::RPagesOptions::kIncludeChecksums;
   auto res = RNTupleExporter::ExportPages(*source, opts);

   EXPECT_EQ(res.fExportedFileNames.size(), 3);

   FileRaii pageVecIdx("./cluster_0_vec-0_page_0_elems_10_comp_0.page");
   FileRaii pageVec("./cluster_0_vec._0-0_page_0_elems_60_comp_0.page");
   FileRaii pageFlt("./cluster_0_flt-0_page_0_elems_10_comp_0.page");

   EXPECT_TRUE(std::find(res.fExportedFileNames.begin(), res.fExportedFileNames.end(), pageFlt.GetPath()) !=
               res.fExportedFileNames.end());
   EXPECT_TRUE(std::find(res.fExportedFileNames.begin(), res.fExportedFileNames.end(), pageVec.GetPath()) !=
               res.fExportedFileNames.end());
   EXPECT_TRUE(std::find(res.fExportedFileNames.begin(), res.fExportedFileNames.end(), pageVecIdx.GetPath()) !=
               res.fExportedFileNames.end());

   // check the file contents
   auto fltBytes = ReadFileToString(pageFlt.GetPath().c_str());
   EXPECT_EQ(fltBytes.length(), kFltBytesLenChecksums);
   EXPECT_EQ(memcmp(fltBytes.data(), kFltBytes, fltBytes.length()), 0);

   auto vecIdxBytes = ReadFileToString(pageVecIdx.GetPath().c_str());
   EXPECT_EQ(vecIdxBytes.length(), kVecIdxBytesLenChecksums);
   EXPECT_EQ(memcmp(vecIdxBytes.data(), kVecIdxBytes, vecIdxBytes.length()), 0);

   auto vecBytes = ReadFileToString(pageVec.GetPath().c_str());
   EXPECT_EQ(vecBytes.length(), kVecBytesLenChecksums);
   EXPECT_EQ(memcmp(vecBytes.data(), kVecBytes, vecBytes.length()), 0);
}

TEST(RNTupleExporter, ExportToFilesWithNoChecksum)
{
   // Try to dump pages+checksum of a RNTuple that has no checksums (should succeed)
   FileRaii fileGuard("ntuple_exporter_no_chk.root");

   // Create RNTuple to export
   CreateExportRNTuple(fileGuard.GetPath(), kWithoutChecksums);

   // Now export the pages
   auto source = RPageSource::Create("ntuple", fileGuard.GetPath());
   auto opts = RNTupleExporter::RPagesOptions();
   opts.fFlags |= RNTupleExporter::RPagesOptions::kIncludeChecksums;
   auto res = RNTupleExporter::ExportPages(*source, opts);

   EXPECT_EQ(res.fExportedFileNames.size(), 3);

   FileRaii pageVecIdx("./cluster_0_vec-0_page_0_elems_10_comp_0.page");
   FileRaii pageVec("./cluster_0_vec._0-0_page_0_elems_60_comp_0.page");
   FileRaii pageFlt("./cluster_0_flt-0_page_0_elems_10_comp_0.page");

   EXPECT_TRUE(std::find(res.fExportedFileNames.begin(), res.fExportedFileNames.end(), pageFlt.GetPath()) !=
               res.fExportedFileNames.end());
   EXPECT_TRUE(std::find(res.fExportedFileNames.begin(), res.fExportedFileNames.end(), pageVec.GetPath()) !=
               res.fExportedFileNames.end());
   EXPECT_TRUE(std::find(res.fExportedFileNames.begin(), res.fExportedFileNames.end(), pageVecIdx.GetPath()) !=
               res.fExportedFileNames.end());

   // check the file contents
   auto fltBytes = ReadFileToString(pageFlt.GetPath().c_str());
   EXPECT_EQ(fltBytes.length(), kFltBytesLenNoChecksums);
   EXPECT_EQ(memcmp(fltBytes.data(), kFltBytes, fltBytes.length()), 0);

   auto vecIdxBytes = ReadFileToString(pageVecIdx.GetPath().c_str());
   EXPECT_EQ(vecIdxBytes.length(), kVecIdxBytesLenNoChecksums);
   EXPECT_EQ(memcmp(vecIdxBytes.data(), kVecIdxBytes, vecIdxBytes.length()), 0);

   auto vecBytes = ReadFileToString(pageVec.GetPath().c_str());
   EXPECT_EQ(vecBytes.length(), kVecBytesLenNoChecksums);
   EXPECT_EQ(memcmp(vecBytes.data(), kVecBytes, vecBytes.length()), 0);
}

TEST(RNTupleExporter, ExportToFilesManyPages)
{
   FileRaii fileGuard("ntuple_exporter_manypages.root");

   // Create RNTuple to export
   {
      auto model = ROOT::RNTupleModel::Create();
      auto pFlt = model->MakeField<float>("flt");
      auto pVec = model->MakeField<std::vector<int>>("vec");

      auto opts = ROOT::RNTupleWriteOptions();
      opts.SetCompression(0);
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), opts);
      for (int i = 0; i < 1000; ++i) {
         *pFlt = i;
         for (int j = -3; j < 3; ++j)
            pVec->emplace_back(i - j);
         writer->Fill();
      }
   }

   // Now export the pages
   auto source = RPageSource::Create("ntuple", fileGuard.GetPath());
   auto res = RNTupleExporter::ExportPages(*source);

   EXPECT_EQ(res.fExportedFileNames.size(), 14);

   for (const auto &file : res.fExportedFileNames)
      std::remove(file.c_str());
}

TEST(RNTupleExporter, EmptySource)
{
   FileRaii fileGuard("ntuple_exporter_empty.root");
   {
      auto model = ROOT::RNTupleModel::Create();
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
   }

   auto source = RPageSource::Create("ntuple", fileGuard.GetPath());
   auto res = RNTupleExporter::ExportPages(*source);

   EXPECT_EQ(res.fExportedFileNames.size(), 0);
}

TEST(RNTupleExporter, ExportToFilesCustomPath)
{
   FileRaii fileGuard("ntuple_exporter_custom_path.root");

   // Create RNTuple to export
   {
      auto model = ROOT::RNTupleModel::Create();
      auto pFlt = model->MakeField<float>("flt");
      auto pVec = model->MakeField<std::vector<int>>("vec");

      auto opts = ROOT::RNTupleWriteOptions();
      opts.SetCompression(505);
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), opts);
      for (int i = 0; i < 100; ++i) {
         *pFlt = i;
         pVec->clear();
         for (int j = -10; j < 10; ++j)
            pVec->emplace_back(i - j);
         writer->Fill();
      }
   }

   // create tmp directory
   static const std::string kDirName = "rntuple_exporter_custom_path";
   bool ok = std::filesystem::create_directory(kDirName);
   if (!ok) {
      FAIL() << "failed to create directory " << kDirName;
      return;
   }
   struct Defer {
      ~Defer() { std::filesystem::remove_all(kDirName); }
   } const defer;

   // Now export the pages
   auto source = RPageSource::Create("ntuple", fileGuard.GetPath());
   auto opts = RNTupleExporter::RPagesOptions();
   opts.fOutputPath = kDirName;
   auto res = RNTupleExporter::ExportPages(*source, opts);

   EXPECT_EQ(res.fExportedFileNames.size(), 3);

   FileRaii pageVecIdx(kDirName + "/cluster_0_vec-0_page_0_elems_100_comp_505.page");
   FileRaii pageVec(kDirName + "/cluster_0_vec._0-0_page_0_elems_2000_comp_505.page");
   FileRaii pageFlt(kDirName + "/cluster_0_flt-0_page_0_elems_100_comp_505.page");

   EXPECT_TRUE(std::find(res.fExportedFileNames.begin(), res.fExportedFileNames.end(), pageFlt.GetPath()) !=
               res.fExportedFileNames.end());
   EXPECT_TRUE(std::find(res.fExportedFileNames.begin(), res.fExportedFileNames.end(), pageVec.GetPath()) !=
               res.fExportedFileNames.end());
   EXPECT_TRUE(std::find(res.fExportedFileNames.begin(), res.fExportedFileNames.end(), pageVecIdx.GetPath()) !=
               res.fExportedFileNames.end());

   for (const auto &fname : res.fExportedFileNames)
      EXPECT_TRUE(std::filesystem::exists(fname));
}

TEST(RNTupleExporter, ExportToFilesWhitelist)
{
   // Dump pages of a regular RNTuple with checksums. Should not include checksums in the output.
   // Also keep only columns of type `kIndex64`.
   FileRaii fileGuard("ntuple_exporter_whitelist.root");

   // Create RNTuple to export
   CreateExportRNTuple(fileGuard.GetPath(), kWithChecksums);

   // Now export the pages
   auto source = RPageSource::Create("ntuple", fileGuard.GetPath());
   auto opts = RNTupleExporter::RPagesOptions();
   opts.fColumnTypeFilter.fType = RNTupleExporter::EFilterType::kWhitelist;
   opts.fColumnTypeFilter.fSet.insert(ROOT::ENTupleColumnType::kIndex64);
   auto res = RNTupleExporter::ExportPages(*source, opts);

   // Should only have exported the page for the index column
   EXPECT_EQ(res.fExportedFileNames.size(), 1);

   FileRaii pageVecIdx("./cluster_0_vec-0_page_0_elems_10_comp_0.page");
   FileRaii pageVec("./cluster_0_vec._0-0_page_0_elems_60_comp_0.page");
   FileRaii pageFlt("./cluster_0_flt-0_page_0_elems_10_comp_0.page");

   EXPECT_FALSE(std::find(res.fExportedFileNames.begin(), res.fExportedFileNames.end(), pageFlt.GetPath()) !=
                res.fExportedFileNames.end());
   EXPECT_FALSE(std::find(res.fExportedFileNames.begin(), res.fExportedFileNames.end(), pageVec.GetPath()) !=
                res.fExportedFileNames.end());
   EXPECT_TRUE(std::find(res.fExportedFileNames.begin(), res.fExportedFileNames.end(), pageVecIdx.GetPath()) !=
               res.fExportedFileNames.end());

   // check the file contents
   auto vecIdxBytes = ReadFileToString(pageVecIdx.GetPath().c_str());
   EXPECT_EQ(vecIdxBytes.length(), kVecIdxBytesLenNoChecksums);
   EXPECT_EQ(memcmp(vecIdxBytes.data(), kVecIdxBytes, vecIdxBytes.length()), 0);
}

TEST(RNTupleExporter, ExportToFilesBlacklist)
{
   // Dump pages of a regular RNTuple with checksums. Should not include checksums in the output.
   // Also discard columns of type `kIndex64`.
   FileRaii fileGuard("ntuple_exporter_blacklist.root");

   // Create RNTuple to export
   CreateExportRNTuple(fileGuard.GetPath(), kWithChecksums);

   // Now export the pages
   auto source = RPageSource::Create("ntuple", fileGuard.GetPath());
   auto opts = RNTupleExporter::RPagesOptions();
   opts.fColumnTypeFilter.fType = RNTupleExporter::EFilterType::kBlacklist;
   opts.fColumnTypeFilter.fSet.insert(ROOT::ENTupleColumnType::kIndex64);
   auto res = RNTupleExporter::ExportPages(*source, opts);

   // Should not have exported the page for the index column
   EXPECT_EQ(res.fExportedFileNames.size(), 2);

   FileRaii pageVecIdx("./cluster_0_vec-0_page_0_elems_10_comp_0.page");
   FileRaii pageVec("./cluster_0_vec._0-0_page_0_elems_60_comp_0.page");
   FileRaii pageFlt("./cluster_0_flt-0_page_0_elems_10_comp_0.page");

   EXPECT_TRUE(std::find(res.fExportedFileNames.begin(), res.fExportedFileNames.end(), pageFlt.GetPath()) !=
               res.fExportedFileNames.end());
   EXPECT_TRUE(std::find(res.fExportedFileNames.begin(), res.fExportedFileNames.end(), pageVec.GetPath()) !=
               res.fExportedFileNames.end());
   EXPECT_FALSE(std::find(res.fExportedFileNames.begin(), res.fExportedFileNames.end(), pageVecIdx.GetPath()) !=
                res.fExportedFileNames.end());

   // check the file contents
   auto fltBytes = ReadFileToString(pageFlt.GetPath().c_str());
   EXPECT_EQ(fltBytes.length(), kFltBytesLenNoChecksums);
   EXPECT_EQ(memcmp(fltBytes.data(), kFltBytes, fltBytes.length()), 0);

   auto vecBytes = ReadFileToString(pageVec.GetPath().c_str());
   EXPECT_EQ(vecBytes.length(), kVecBytesLenNoChecksums);
   EXPECT_EQ(memcmp(vecBytes.data(), kVecBytes, vecBytes.length()), 0);
}
