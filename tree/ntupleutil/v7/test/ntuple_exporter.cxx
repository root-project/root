#include <ROOT/RNTupleExporter.hxx>

#include <ROOT/RNTupleModel.hxx>

#include "ntupleutil_test.hxx"

using namespace ROOT::Experimental;

static std::string ReadFileToString(const char *fname)
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
   fread(str.data(), 1, size, f);
   fclose(f);
   return str;
}

TEST(RNTupleExporter, ExportToFiles)
{
   static const char kVecBytes[] =
      "\x5a\x53\x01\x9e\x00\x00\x40\x1f\x00\x28\xb5\x2f\xfd\x60\x40\x1e\xa5\x04\x00\x84\x07\x14\x12\x10\x0e\x0c\x0a\x08"
      "\x06\x04\x02\x00\x01\x03\x05\x07\x09\x0b\x0d\x0f\x11\x16\x18\x1a\x1c\x1e\x20\x22\x24\x26\x28\x2a\x2c\x2e\x30\x32"
      "\x34\x36\x38\x3a\x3c\x3e\x40\x42\x44\x46\x48\x4a\x4c\x4e\x50\x52\x54\x56\x58\x5a\x5c\x5e\x60\x62\x64\x66\x68\x6a"
      "\x6c\x6e\x70\x72\x74\x76\x78\x7a\x7c\x7e\x80\x82\x84\x86\x88\x8a\x8c\x8e\x90\x92\x94\x96\x98\x9a\x9c\x9e\xa0\xa2"
      "\xa4\xa6\xa8\xaa\xac\xae\xb0\xb2\xb4\xb6\xb8\xba\xbc\xbe\xc0\xc2\xc4\xc6\xc8\xca\xcc\xce\xd0\xd2\xd4\xd6\xd8\xda"
      "\x00\x64\xa8\x11\xe0\xef\x7f\x06\xf0\x85\x31\x12\xf8\x1f\xff\xfe\xff\x9f\x01\x6c\x07\xf7\x46\x27\x1a\x53\x15";
   static const char kVecIdxBytes[] = "\x5a\x53\x01\x16\x00\x00\x20\x03\x00\x28\xb5\x2f\xfd\x60\x20\x02\x65\x00"
                                      "\x00\x18\x14\x14\x00\x02\x00\xb8\x40\x25\x7e\x02\x58";
   static const char kFltBytes[] =
      "\x5a\x53\x01\x81\x00\x00\x90\x01\x00\x28\xb5\x2f\xfd\x60\x90\x00\xbd\x03\x00\xa4\x06\x00\x00\x80\x00\x40\x80\xa0"
      "\xc0\xe0\x00\x10\x20\x30\x40\x50\x60\x70\x80\x88\x90\x98\xa0\xa8\xb0\xb8\xc0\xc8\xd0\xd8\xe0\xe8\xf0\xf8\x00\x04"
      "\x08\x0c\x10\x14\x18\x1c\x20\x24\x28\x2c\x30\x34\x38\x3c\x40\x44\x48\x4c\x50\x54\x58\x5c\x60\x64\x68\x6c\x70\x74"
      "\x78\x7c\x80\x82\x84\x86\x88\x8a\x8c\x8e\x90\x92\x94\x96\x98\x9a\x9c\x9e\xa0\xa2\xa4\xa6\xa8\xaa\xac\xae\xb0\xb2"
      "\xb4\xb6\xb8\xba\xbc\xbe\xc0\xc2\xc4\xc6\x00\x3f\x40\x41\x42\x04\x10\x00\x80\x4a\x3d\x53\x38\x44\xaa\x0d";

   FileRaii fileGuard("ntuple_exporter.root");

   // Create RNTuple to export
   {
      auto model = RNTupleModel::Create();
      auto pFlt = model->MakeField<float>("flt");
      auto pVec = model->MakeField<std::vector<int>>("vec");

      auto opts = RNTupleWriteOptions();
      opts.SetCompression(505);
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), opts);
      for (int i = 0; i < 100; ++i) {
         *pFlt = i;
         pVec->clear();
         for (int j = -10; j < 10; ++j)
            pVec->emplace_back(i - j);
         writer->Fill();
      }
   }

   // Now export the pages
   auto source = Internal::RPageSource::Create("ntuple", fileGuard.GetPath());
   auto res = Internal::RNTupleExporter::ExportPages(*source);

   EXPECT_EQ(res.fNPagesExported, 3);

   FileRaii pageVecIdx("./cluster_0_vec-0_page_0_elems_100_comp_505.page");
   FileRaii pageVec("./cluster_0_vec._0-0_page_0_elems_2000_comp_505.page");
   FileRaii pageFlt("./cluster_0_flt-0_page_0_elems_100_comp_505.page");

   EXPECT_TRUE(std::find(res.fExportedFileNames.begin(), res.fExportedFileNames.end(), pageFlt.GetPath()) !=
               res.fExportedFileNames.end());
   EXPECT_TRUE(std::find(res.fExportedFileNames.begin(), res.fExportedFileNames.end(), pageVec.GetPath()) !=
               res.fExportedFileNames.end());
   EXPECT_TRUE(std::find(res.fExportedFileNames.begin(), res.fExportedFileNames.end(), pageVecIdx.GetPath()) !=
               res.fExportedFileNames.end());

   // check the file contents
   auto fltBytes = ReadFileToString(pageFlt.GetPath().c_str());
   EXPECT_EQ(fltBytes.length(), std::size(kFltBytes) - 1);
   EXPECT_EQ(memcmp(fltBytes.data(), kFltBytes, fltBytes.length()), 0);

   auto vecIdxBytes = ReadFileToString(pageVecIdx.GetPath().c_str());
   EXPECT_EQ(vecIdxBytes.length(), std::size(kVecIdxBytes) - 1);
   EXPECT_EQ(memcmp(vecIdxBytes.data(), kVecIdxBytes, vecIdxBytes.length()), 0);

   auto vecBytes = ReadFileToString(pageVec.GetPath().c_str());
   EXPECT_EQ(vecBytes.length(), std::size(kVecBytes) - 1);
   EXPECT_EQ(memcmp(vecBytes.data(), kVecBytes, vecBytes.length()), 0);
}

TEST(RNTupleExporter, ExportToFilesWithChecksum)
{
   static const char kVecBytes[] =
      "\x03\x00\x00\x00\x02\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\xff\xfe\xff\xff\xff\x04\x00\x00\x00"
      "\x03\x00\x00\x00\x02\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\xff\x05\x00\x00\x00\x04\x00\x00\x00"
      "\x03\x00\x00\x00\x02\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x05\x00\x00\x00\x04\x00\x00\x00"
      "\x03\x00\x00\x00\x02\x00\x00\x00\x01\x00\x00\x00\x07\x00\x00\x00\x06\x00\x00\x00\x05\x00\x00\x00\x04\x00\x00\x00"
      "\x03\x00\x00\x00\x02\x00\x00\x00\x08\x00\x00\x00\x07\x00\x00\x00\x06\x00\x00\x00\x05\x00\x00\x00\x04\x00\x00\x00"
      "\x03\x00\x00\x00\x09\x00\x00\x00\x08\x00\x00\x00\x07\x00\x00\x00\x06\x00\x00\x00\x05\x00\x00\x00\x04\x00\x00\x00"
      "\x0a\x00\x00\x00\x09\x00\x00\x00\x08\x00\x00\x00\x07\x00\x00\x00\x06\x00\x00\x00\x05\x00\x00\x00\x0b\x00\x00\x00"
      "\x0a\x00\x00\x00\x09\x00\x00\x00\x08\x00\x00\x00\x07\x00\x00\x00\x06\x00\x00\x00\x0c\x00\x00\x00\x0b\x00\x00\x00"
      "\x0a\x00\x00\x00\x09\x00\x00\x00\x08\x00\x00\x00\x07\x00\x00\x00\x56\x4a\xe0\x51\xf5\x0b\xfc\x61";
   static const char kVecIdxBytes[] =
      "\x06\x00\x00\x00\x00\x00\x00\x00\x0c\x00\x00\x00\x00\x00\x00\x00\x12\x00\x00\x00\x00\x00\x00\x00\x18\x00\x00\x00"
      "\x00\x00\x00\x00\x1e\x00\x00\x00\x00\x00\x00\x00\x24\x00\x00\x00\x00\x00\x00\x00\x2a\x00\x00\x00\x00\x00\x00\x00"
      "\x30\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00\x00\x00\x00\x00\x3c\x00\x00\x00\x00\x00\x00\x00\xaa\xde\x1b\xde"
      "\xb5\xf3\xbc\x1a";
   static const char kFltBytes[] =
      "\x00\x00\x00\x00\x00\x00\x80\x3f\x00\x00\x00\x40\x00\x00\x40\x40\x00\x00\x80\x40\x00\x00\xa0\x40\x00\x00\xc0"
      "\x40"
      "\x00\x00\xe0\x40\x00\x00\x00\x41\x00\x00\x10\x41\x1b\xad\x67\xa6\xe6\x61\x56\x9d";

   FileRaii fileGuard("ntuple_exporter_chk.root");

   // Create RNTuple to export
   {
      auto model = RNTupleModel::Create();
      auto pFlt = model->MakeField<float>("flt");
      auto pVec = model->MakeField<std::vector<int>>("vec");

      auto opts = RNTupleWriteOptions();
      opts.SetCompression(0);
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), opts);
      for (int i = 0; i < 10; ++i) {
         *pFlt = i;
         pVec->clear();
         for (int j = -3; j < 3; ++j)
            pVec->emplace_back(i - j);
         writer->Fill();
      }
   }

   // Now export the pages
   auto source = Internal::RPageSource::Create("ntuple", fileGuard.GetPath());
   auto opts = Internal::RExportPagesOptions();
   opts.fFlags |= Internal::RExportPagesOptions::kIncludeChecksums;
   auto res = Internal::RNTupleExporter::ExportPages(*source, opts);

   EXPECT_EQ(res.fNPagesExported, 3);

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
   EXPECT_EQ(fltBytes.length(), std::size(kFltBytes) - 1);
   EXPECT_EQ(memcmp(fltBytes.data(), kFltBytes, fltBytes.length()), 0);

   auto vecIdxBytes = ReadFileToString(pageVecIdx.GetPath().c_str());
   EXPECT_EQ(vecIdxBytes.length(), std::size(kVecIdxBytes) - 1);
   EXPECT_EQ(memcmp(vecIdxBytes.data(), kVecIdxBytes, vecIdxBytes.length()), 0);

   auto vecBytes = ReadFileToString(pageVec.GetPath().c_str());
   EXPECT_EQ(vecBytes.length(), std::size(kVecBytes) - 1);
   EXPECT_EQ(memcmp(vecBytes.data(), kVecBytes, vecBytes.length()), 0);
}

TEST(RNTupleExporter, ExportToFilesManyPages)
{
   FileRaii fileGuard("ntuple_exporter_manypages.root");

   // Create RNTuple to export
   {
      auto model = RNTupleModel::Create();
      auto pFlt = model->MakeField<float>("flt");
      auto pVec = model->MakeField<std::vector<int>>("vec");

      auto opts = RNTupleWriteOptions();
      opts.SetCompression(0);
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), opts);
      for (int i = 0; i < 1000; ++i) {
         *pFlt = i;
         for (int j = -3; j < 3; ++j)
            pVec->emplace_back(i - j);
         writer->Fill();
      }
   }

   // Now export the pages
   auto source = Internal::RPageSource::Create("ntuple", fileGuard.GetPath());
   auto res = Internal::RNTupleExporter::ExportPages(*source);

   EXPECT_EQ(res.fNPagesExported, 14);

   for (const auto &file : res.fExportedFileNames)
      std::remove(file.c_str());
}

TEST(RNTupleExporter, EmptySource)
{
   FileRaii fileGuard("ntuple_exporter_empty.root");
   {
      auto model = RNTupleModel::Create();
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
   }

   auto source = Internal::RPageSource::Create("ntuple", fileGuard.GetPath());
   auto res = Internal::RNTupleExporter::ExportPages(*source);

   EXPECT_EQ(res.fNPagesExported, 0);
   EXPECT_EQ(res.fExportedFileNames.size(), 0);
}

TEST(RNTupleExporter, ExportToFilesCustomPath)
{
   FileRaii fileGuard("ntuple_exporter_custom_path.root");

   // Create RNTuple to export
   {
      auto model = RNTupleModel::Create();
      auto pFlt = model->MakeField<float>("flt");
      auto pVec = model->MakeField<std::vector<int>>("vec");

      auto opts = RNTupleWriteOptions();
      opts.SetCompression(505);
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), opts);
      for (int i = 0; i < 100; ++i) {
         *pFlt = i;
         pVec->clear();
         for (int j = -10; j < 10; ++j)
            pVec->emplace_back(i - j);
         writer->Fill();
      }
   }

   // create tmp directory
   static const std::filesystem::path kDirName = "rntuple_exporter_custom_path";
   bool ok = std::filesystem::create_directory(kDirName);
   if (!ok) {
      FAIL() << "failed to create directory " << kDirName;
      return;
   }
   struct Defer {
      ~Defer() { std::filesystem::remove_all(kDirName); }
   } const defer;

   // Now export the pages
   auto source = Internal::RPageSource::Create("ntuple", fileGuard.GetPath());
   auto opts = Internal::RExportPagesOptions();
   opts.fOutputPath = kDirName;
   auto res = Internal::RNTupleExporter::ExportPages(*source, opts);

   EXPECT_EQ(res.fNPagesExported, 3);

   FileRaii pageVecIdx(kDirName / "cluster_0_vec-0_page_0_elems_100_comp_505.page");
   FileRaii pageVec(kDirName / "cluster_0_vec._0-0_page_0_elems_2000_comp_505.page");
   FileRaii pageFlt(kDirName / "cluster_0_flt-0_page_0_elems_100_comp_505.page");

   EXPECT_TRUE(std::find(res.fExportedFileNames.begin(), res.fExportedFileNames.end(), pageFlt.GetPath()) !=
               res.fExportedFileNames.end());
   EXPECT_TRUE(std::find(res.fExportedFileNames.begin(), res.fExportedFileNames.end(), pageVec.GetPath()) !=
               res.fExportedFileNames.end());
   EXPECT_TRUE(std::find(res.fExportedFileNames.begin(), res.fExportedFileNames.end(), pageVecIdx.GetPath()) !=
               res.fExportedFileNames.end());

   for (const auto &fname : res.fExportedFileNames)
      EXPECT_TRUE(std::filesystem::exists(fname));
}
