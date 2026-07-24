#include <ROOT/RNTupleInspector.hxx>
#include <ROOT/RMiniFile.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>

#include <TFile.h>
#include <ROOT/TestSupport.hxx>

#include "gmock/gmock.h"

#include "CustomStructUtil.hxx"
#include "ntupleutil_test.hxx"

using ROOT::ENTupleColumnType;
using ROOT::RField;
using ROOT::RFieldBase;
using ROOT::RNTuple;
using ROOT::RNTupleDescriptor;
using ROOT::RNTupleLocator;
using ROOT::RNTupleModel;
using ROOT::RNTupleWriteOptions;
using ROOT::RNTupleWriter;
using ROOT::Experimental::RNTupleInspector;
using ROOT::Internal::MakeUninitArray;
using ROOT::Internal::RClusterDescriptorBuilder;
using ROOT::Internal::RClusterGroupDescriptorBuilder;
using ROOT::Internal::RColumnDescriptorBuilder;
using ROOT::Internal::RFieldDescriptorBuilder;
using ROOT::Internal::RNTupleDescriptorBuilder;
using ROOT::Internal::RNTupleFileWriter;
using ROOT::Internal::RNTupleSerializer;

TEST(RNTupleInspector, CreateFromPointer)
{
   FileRaii fileGuard("test_ntuple_inspector_create_from_pointer.root");
   {
      auto model = RNTupleModel::Create();
      RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
   }

   std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str()));
   auto ntuple = std::unique_ptr<RNTuple>(file->Get<RNTuple>("ntuple"));
   auto inspector = RNTupleInspector::Create(*ntuple);
   EXPECT_EQ(inspector->GetDescriptor().GetName(), "ntuple");
}

TEST(RNTupleInspector, CreateFromString)
{
   FileRaii fileGuard("test_ntuple_inspector_create_from_string.root");
   {
      RNTupleWriter::Recreate(RNTupleModel::Create(), "ntuple", fileGuard.GetPath());
   }

   auto inspector = RNTupleInspector::Create("ntuple", fileGuard.GetPath());
   EXPECT_EQ(inspector->GetDescriptor().GetName(), "ntuple");

   EXPECT_THROW(RNTupleInspector::Create("nonexistent", fileGuard.GetPath()), ROOT::RException);
}

TEST(RNTupleInspector, CompressionSettings)
{
   FileRaii fileGuard("test_ntuple_inspector_compression_settings.root");
   {
      auto model = RNTupleModel::Create();
      auto nFldInt = model->MakeField<std::int32_t>("int");

      auto writeOptions = RNTupleWriteOptions();
      writeOptions.SetCompression(207);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), writeOptions);

      *nFldInt = 42;
      ntuple->Fill();
   }

   auto inspector = RNTupleInspector::Create("ntuple", fileGuard.GetPath());

   EXPECT_EQ(207, *inspector->GetCompressionSettings());
   EXPECT_EQ("LZMA (level 7)", inspector->GetCompressionSettingsAsString());
}

// Relevant for RNTuples created with late model extension, see https://github.com/root-project/root/issues/15661 for
// background.
TEST(RNTupleInspector, UnknownCompression)
{
   FileRaii fileGuard("test_ntuple_inspector_unknown_compression.root");
   std::vector<float> refVec{1., 2., 3.};
   {
      auto model = RNTupleModel::Create();

      *model->MakeField<std::vector<float>>("vecFld") = refVec;

      RNTupleWriteOptions opts;
      opts.SetCompression(505);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), opts);

      ntuple->Fill();
      ntuple->CommitCluster();

      auto modelUpdater = ntuple->CreateModelUpdater();

      modelUpdater->BeginUpdate();
      *modelUpdater->MakeField<std::vector<float>>("extVecFld") = refVec;
      modelUpdater->CommitUpdate();

      ntuple->Fill();
   }

   auto inspector = RNTupleInspector::Create("ntuple", fileGuard.GetPath());
   EXPECT_EQ(505, *inspector->GetCompressionSettings());
}

TEST(RNTupleInspector, Empty)
{
   FileRaii fileGuard("test_ntuple_inspector_empty.root");
   {
      auto writer = RNTupleWriter::Recreate(RNTupleModel::Create(), "ntuple", fileGuard.GetPath());
   }

   auto inspector = RNTupleInspector::Create("ntuple", fileGuard.GetPath());
   EXPECT_FALSE(inspector->GetCompressionSettings());
   EXPECT_EQ("unknown", inspector->GetCompressionSettingsAsString());
}

TEST(RNTupleInspector, SizeUncompressedSimple)
{
   FileRaii fileGuard("test_ntuple_inspector_size_uncompressed_complex.root");
   {
      auto model = RNTupleModel::Create();
      auto nFldInt = model->MakeField<std::int32_t>("int");

      auto writeOptions = RNTupleWriteOptions();
      writeOptions.SetCompression(0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), writeOptions);

      for (int i = 0; i < 5; i++) {
         *nFldInt = 1;
         ntuple->Fill();
      }
   }

   auto inspector = RNTupleInspector::Create("ntuple", fileGuard.GetPath());

   EXPECT_EQ(sizeof(int32_t) * 5, inspector->GetUncompressedSize());
   EXPECT_EQ(inspector->GetCompressedSize(), inspector->GetUncompressedSize());
}

TEST(RNTupleInspector, SizeUncompressedComplex)
{
   FileRaii fileGuard("test_ntuple_inspector_size_uncompressed_complex.root");
   {
      auto model = RNTupleModel::Create();
      auto nFldObject = model->MakeField<ComplexStructUtil>("object");

      auto writeOptions = RNTupleWriteOptions();
      writeOptions.SetCompression(0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), writeOptions);

      nFldObject->Init1();
      ntuple->Fill();
      nFldObject->Init2();
      ntuple->Fill();
      nFldObject->Init3();
      ntuple->Fill();
   }

   auto inspector = RNTupleInspector::Create("ntuple", fileGuard.GetPath());

   int nIndexCols = inspector->GetColumnCountByType(ENTupleColumnType::kIndex64);
   int nEntries = inspector->GetDescriptor().GetNEntries();

   EXPECT_EQ(2, nIndexCols);
   EXPECT_EQ(3, nEntries);

   EXPECT_EQ(inspector->GetCompressedSize(), inspector->GetUncompressedSize());
}

TEST(RNTupleInspector, SizeCompressedInt)
{
   FileRaii fileGuard("test_ntuple_inspector_size_compressed_int.root");
   {
      auto model = RNTupleModel::Create();
      auto nFldInt = model->MakeField<std::int32_t>("int");

      auto writeOptions = RNTupleWriteOptions();
      writeOptions.SetCompression(505);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), writeOptions);

      for (int32_t i = 0; i < 500; ++i) {
         *nFldInt = i;
         ntuple->Fill();

         // Store the data in ten clusters to be able to test that the size is correctly computed in this way.
         if (i % 50 == 49)
            ntuple->CommitCluster();
      }
   }

   auto inspector = RNTupleInspector::Create("ntuple", fileGuard.GetPath());

   EXPECT_EQ(sizeof(int32_t) * 500, inspector->GetUncompressedSize());
   EXPECT_LT(inspector->GetCompressedSize(), inspector->GetUncompressedSize());
   // Check the target size with a 5% tolerance to account for small fluctuations across different platforms.
   std::uint64_t targetSize = 800;
   EXPECT_NEAR(inspector->GetCompressedSize(), targetSize, targetSize * .05f);
   EXPECT_GT(inspector->GetCompressionFactor(), 1);
}

TEST(RNTupleInspector, SizeCompressedComplex)
{
   FileRaii fileGuard("test_ntuple_inspector_size_compressed_complex.root");
   {
      auto model = RNTupleModel::Create();
      auto nFldObject = model->MakeField<ComplexStructUtil>("object");

      auto writeOptions = RNTupleWriteOptions();
      writeOptions.SetCompression(505);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), writeOptions);

      for (int i = 0; i < 100; ++i) {
         nFldObject->Init1();
         ntuple->Fill();
         nFldObject->Init2();
         ntuple->Fill();
         nFldObject->Init3();
         ntuple->Fill();

         // Store the data in ten clusters to be able to test that the size is correctly computed in this way.
         if (i % 10 == 9)
            ntuple->CommitCluster();
      }
   }

   auto inspector = RNTupleInspector::Create("ntuple", fileGuard.GetPath());

   EXPECT_LT(inspector->GetCompressedSize(), inspector->GetUncompressedSize());
   // Check the target size with a 5% tolerance to account for small fluctuations across different platforms.
   std::uint64_t targetSize = 3210;
   EXPECT_NEAR(inspector->GetCompressedSize(), targetSize, targetSize * .05f);
   EXPECT_GT(inspector->GetCompressionFactor(), 1);
}

TEST(RNTupleInspector, SizeEmpty)
{
   FileRaii fileGuard("test_ntuple_inspector_size_empty.root");
   {
      auto model = RNTupleModel::Create();
      RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
   }

   auto inspector = RNTupleInspector::Create("ntuple", fileGuard.GetPath());

   EXPECT_EQ(0, inspector->GetCompressedSize());
   EXPECT_EQ(0, inspector->GetUncompressedSize());
}

TEST(RNTupleInspector, SizeProjectedFields)
{
   FileRaii fileGuard("test_ntuple_inspector_size_projected_fields.root");
   {
      auto model = RNTupleModel::Create();
      auto muonPt = model->MakeField<std::vector<float>>("muonPt");
      muonPt->emplace_back(1.0);
      muonPt->emplace_back(2.0);

      auto nMuons = RFieldBase::Create("nMuons", "ROOT::RNTupleCardinality<std::uint64_t>").Unwrap();
      model->AddProjectedField(std::move(nMuons), [](const std::string &) { return "muonPt"; });

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
      writer->Fill();
   }

   auto inspector = RNTupleInspector::Create("ntuple", fileGuard.GetPath());

   EXPECT_EQ(inspector->GetFieldTreeInspector("muonPt").GetUncompressedSize(), inspector->GetUncompressedSize());
   EXPECT_EQ(inspector->GetFieldTreeInspector("muonPt").GetCompressedSize(), inspector->GetCompressedSize());
}

TEST(RNTupleInspector, ColumnInfoCompressed)
{
   FileRaii fileGuard("test_ntuple_inspector_column_info_compressed.root");
   {
      auto model = RNTupleModel::Create();
      auto nFldObject = model->MakeField<ComplexStructUtil>("object");

      auto writeOptions = RNTupleWriteOptions();
      writeOptions.SetCompression(505);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), writeOptions);

      for (int i = 0; i < 25; ++i) {
         nFldObject->Init1();
         ntuple->Fill();
         nFldObject->Init2();
         ntuple->Fill();
         nFldObject->Init3();
         ntuple->Fill();
      }
   }

   auto inspector = RNTupleInspector::Create("ntuple", fileGuard.GetPath());

   std::uint64_t totalOnDiskSize = 0;

   for (std::size_t i = 0; i < inspector->GetDescriptor().GetNLogicalColumns(); ++i) {
      auto colInfo = inspector->GetColumnInspector(i);
      totalOnDiskSize += colInfo.GetCompressedSize();

      EXPECT_GT(colInfo.GetCompressedSize(), 0);
      EXPECT_GT(colInfo.GetUncompressedSize(), 0);
      EXPECT_LT(colInfo.GetCompressedSize(), colInfo.GetUncompressedSize());
   }

   EXPECT_EQ(totalOnDiskSize, inspector->GetCompressedSize());

   EXPECT_THROW(inspector->GetColumnInspector(42), ROOT::RException);
}

TEST(RNTupleInspector, ColumnInfoUncompressed)
{
   FileRaii fileGuard("test_ntuple_inspector_column_info_uncompressed.root");
   {
      auto model = RNTupleModel::Create();

      auto int32fld = std::make_unique<RField<std::int32_t>>("int32");
      int32fld->SetColumnRepresentatives({{ENTupleColumnType::kInt32}});
      model->AddField(std::move(int32fld));

      auto splitReal64fld = std::make_unique<RField<double>>("splitReal64");
      splitReal64fld->SetColumnRepresentatives({{ENTupleColumnType::kSplitReal64}});
      model->AddField(std::move(splitReal64fld));

      auto writeOptions = RNTupleWriteOptions();
      writeOptions.SetCompression(0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), writeOptions);

      for (int i = 0; i < 5; ++i) {
         auto e = ntuple->CreateEntry();
         *e->GetPtr<std::int32_t>("int32") = i;
         *e->GetPtr<double>("splitReal64") = i;
         ntuple->Fill(*e);
      }
   }

   auto inspector = RNTupleInspector::Create("ntuple", fileGuard.GetPath());

   std::uint64_t colTypeSizes[] = {sizeof(std::int32_t), sizeof(double)};

   for (std::size_t i = 0; i < inspector->GetDescriptor().GetNLogicalColumns(); ++i) {
      auto colInfo = inspector->GetColumnInspector(i);
      EXPECT_EQ(colInfo.GetCompressedSize(), colInfo.GetUncompressedSize());
      EXPECT_EQ(colInfo.GetCompressedSize(), colTypeSizes[i] * 5);
   }
}

TEST(RNTupleInspector, ColumnTypeCount)
{
   FileRaii fileGuard("test_ntuple_inspector_column_type_count.root");
   {
      auto model = RNTupleModel::Create();
      auto nFldObject = model->MakeField<ComplexStructUtil>("object");

      auto writeOptions = RNTupleWriteOptions();
      writeOptions.SetCompression(505);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), writeOptions);
   }

   auto inspector = RNTupleInspector::Create("ntuple", fileGuard.GetPath());

   EXPECT_EQ(2, inspector->GetColumnCountByType(ENTupleColumnType::kSplitIndex64));
   EXPECT_EQ(4, inspector->GetColumnCountByType(ENTupleColumnType::kSplitReal32));
   EXPECT_EQ(3, inspector->GetColumnCountByType(ENTupleColumnType::kSplitInt32));
}

TEST(RNTupleInspector, ColumnsByType)
{
   FileRaii fileGuard("test_ntuple_inspector_columns_by_type.root");
   {
      auto model = RNTupleModel::Create();
      auto nFldInt1 = model->MakeField<std::int64_t>("int1");
      auto nFldInt2 = model->MakeField<std::int64_t>("int2");
      auto nFldFloat = model->MakeField<float>("float");
      auto nFldFloatVec = model->MakeField<std::vector<float>>("floatVec");

      auto writeOptions = RNTupleWriteOptions();
      writeOptions.SetCompression(505);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), writeOptions);
   }

   auto inspector = RNTupleInspector::Create("ntuple", fileGuard.GetPath());
   EXPECT_EQ(2U, inspector->GetColumnsByType(ENTupleColumnType::kSplitInt64).size());
   for (const auto colId : inspector->GetColumnsByType(ENTupleColumnType::kSplitInt64)) {
      EXPECT_EQ(ENTupleColumnType::kSplitInt64, inspector->GetColumnInspector(colId).GetType());
   }

   EXPECT_EQ(2U, inspector->GetColumnsByType(ENTupleColumnType::kSplitReal32).size());
   for (const auto colId : inspector->GetColumnsByType(ENTupleColumnType::kSplitReal32)) {
      EXPECT_EQ(ENTupleColumnType::kSplitReal32, inspector->GetColumnInspector(colId).GetType());
   }

   EXPECT_EQ(1U, inspector->GetColumnsByType(ENTupleColumnType::kSplitIndex64).size());
   for (const auto colId : inspector->GetColumnsByType(ENTupleColumnType::kSplitIndex64)) {
      EXPECT_EQ(ENTupleColumnType::kSplitIndex64, inspector->GetColumnInspector(colId).GetType());
   }

   EXPECT_EQ(0U, inspector->GetColumnsByType(ENTupleColumnType::kSplitReal64).size());
}

TEST(RNTupleInspector, AllColumnsOfField)
{
   FileRaii fileGuard("test_ntuple_inspector_all_columns_of_field.root");
   {
      auto model = RNTupleModel::Create();
      auto nFldInt1 = model->MakeField<std::int64_t>("int1");
      auto nFldInt2 = model->MakeField<std::int64_t>("int2");
      auto nFldFloat = model->MakeField<float>("float");
      auto nFldFloatVec = model->MakeField<std::vector<float>>("floatVec");

      auto writeOptions = RNTupleWriteOptions();
      writeOptions.SetCompression(505);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), writeOptions);
   }

   auto inspector = RNTupleInspector::Create("ntuple", fileGuard.GetPath());

   EXPECT_EQ(1U, inspector->GetAllColumnsOfField(0).size());
   EXPECT_EQ(1U, inspector->GetAllColumnsOfField(1).size());
   for (const auto colId : inspector->GetAllColumnsOfField(0)) {
      EXPECT_EQ(ENTupleColumnType::kSplitInt64, inspector->GetColumnInspector(colId).GetType());
   }

   EXPECT_EQ(1U, inspector->GetAllColumnsOfField(2).size());
   for (const auto colId : inspector->GetAllColumnsOfField(2)) {
      EXPECT_EQ(ENTupleColumnType::kSplitReal32, inspector->GetColumnInspector(colId).GetType());
   }

   EXPECT_EQ(2U, inspector->GetAllColumnsOfField(3).size());
}

TEST(RNTupleInspector, ColumnTypes)
{
   FileRaii fileGuard("test_ntuple_inspector_column_types.root");
   {
      auto model = RNTupleModel::Create();
      auto nFldInt1 = model->MakeField<std::int64_t>("int1");
      auto nFldInt2 = model->MakeField<std::int64_t>("int2");
      auto nFldFloat = model->MakeField<float>("float");
      auto nFldFloatVec = model->MakeField<std::vector<float>>("floatVec");

      auto writeOptions = RNTupleWriteOptions();
      writeOptions.SetCompression(505);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), writeOptions);
   }

   auto inspector = RNTupleInspector::Create("ntuple", fileGuard.GetPath());
   auto types = inspector->GetColumnTypes();
   EXPECT_THAT(types, testing::UnorderedElementsAre(ENTupleColumnType::kSplitInt64, ENTupleColumnType::kSplitReal32,
                                                    ENTupleColumnType::kSplitIndex64));
}

TEST(RNTupleInspector, PrintColumnTypeInfo)
{
   FileRaii fileGuard("test_ntuple_inspector_print_column_type_info.root");
   {
      auto model = RNTupleModel::Create();
      auto nFldInt1 = model->MakeField<std::int64_t>("int1");
      auto nFldInt2 = model->MakeField<std::int64_t>("int2");
      auto nFldFloat = model->MakeField<float>("float");
      auto nFldFloatVec = model->MakeField<std::vector<float>>("floatVec");

      auto writeOptions = RNTupleWriteOptions();
      writeOptions.SetCompression(505);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), writeOptions);

      for (unsigned i = 0; i < 10; ++i) {
         *nFldInt1 = static_cast<std::int64_t>(i);
         *nFldInt2 = static_cast<std::int64_t>(i) * 2;
         *nFldFloat = static_cast<float>(i) * .1f;
         *nFldFloatVec = {static_cast<float>(i), 3.14f, static_cast<float>(i) * *nFldFloat};
         ntuple->Fill();
      }
   }

   auto inspector = RNTupleInspector::Create("ntuple", fileGuard.GetPath());

   std::stringstream csvOutput;
   inspector->PrintColumnTypeInfo(ROOT::Experimental::ENTupleInspectorPrintFormat::kCSV, csvOutput);

   std::string line;
   std::getline(csvOutput, line);
   EXPECT_EQ("columnType,count,nElements,compressedSize,uncompressedSize,compressionFactor,nPages", line);

   size_t nLines = 0;
   std::string colTypeStr;
   while (std::getline(csvOutput, line)) {
      ++nLines;
      colTypeStr = line.substr(0, line.find(','));

      if (colTypeStr != "SplitIndex64" && colTypeStr != "SplitInt64" && colTypeStr != "SplitReal32")
         FAIL() << "Unexpected column type: " << colTypeStr;
   }
   EXPECT_EQ(nLines, 3U);

   std::stringstream tableOutput;
   inspector->PrintColumnTypeInfo(ROOT::Experimental::ENTupleInspectorPrintFormat::kTable, tableOutput);

   std::getline(tableOutput, line);
   EXPECT_EQ(
      " column type    | count   | # elements  | compressed bytes | uncompressed bytes | compression ratio | # pages ",
      line);

   std::getline(tableOutput, line);
   EXPECT_EQ(
      "----------------|---------|-------------|------------------|--------------------|-------------------|-------",
      line);

   nLines = 0;
   while (std::getline(tableOutput, line)) {
      ++nLines;
      colTypeStr = line.substr(0, line.find('|'));
      colTypeStr.erase(remove_if(colTypeStr.begin(), colTypeStr.end(), isspace), colTypeStr.end());

      if (colTypeStr != "SplitIndex64" && colTypeStr != "SplitInt64" && colTypeStr != "SplitReal32")
         FAIL() << "Unexpected column type: " << colTypeStr;
   }
   EXPECT_EQ(nLines, 3U);
}

TEST(RNTupleInspector, ColumnTypeInfoHist)
{
   FileRaii fileGuard("test_ntuple_inspector_column_type_info_hist.root");
   {
      auto model = RNTupleModel::Create();
      auto nFldInt1 = model->MakeField<std::int64_t>("int1");
      auto nFldInt2 = model->MakeField<std::int64_t>("int2");
      auto nFldFloat = model->MakeField<float>("float");
      auto nFldFloatVec = model->MakeField<std::vector<float>>("floatVec");

      auto writeOptions = RNTupleWriteOptions();
      writeOptions.SetCompression(505);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), writeOptions);

      for (unsigned i = 0; i < 10; ++i) {
         *nFldInt1 = static_cast<std::int64_t>(i);
         *nFldInt2 = static_cast<std::int64_t>(i) * 2;
         *nFldFloat = static_cast<float>(i) * .1f;
         *nFldFloatVec = {static_cast<float>(i), 3.14f, static_cast<float>(i) * *nFldFloat};
         ntuple->Fill();
      }
   }

   auto inspector = RNTupleInspector::Create("ntuple", fileGuard.GetPath());

   auto countHist = inspector->GetColumnTypeInfoAsHist(ROOT::Experimental::ENTupleInspectorHist::kCount);
   EXPECT_STREQ("colTypeCountHist", countHist->GetName());
   EXPECT_STREQ("Column count by type", countHist->GetTitle());
   EXPECT_EQ(4U, countHist->GetNbinsX());
   EXPECT_EQ(inspector->GetDescriptor().GetNPhysicalColumns(), countHist->Integral());

   auto nElemsHist = inspector->GetColumnTypeInfoAsHist(ROOT::Experimental::ENTupleInspectorHist::kNElems, "elemsHist");
   EXPECT_STREQ("elemsHist", nElemsHist->GetName());
   EXPECT_STREQ("Number of elements by column type", nElemsHist->GetTitle());
   EXPECT_EQ(4U, nElemsHist->GetNbinsX());
   std::uint64_t nTotalElems = 0;
   for (const auto &col : inspector->GetDescriptor().GetColumnIterable()) {
      nTotalElems += inspector->GetDescriptor().GetNElements(col.GetPhysicalId());
   }
   EXPECT_EQ(nTotalElems, nElemsHist->Integral());

   auto compressedSizeHist = inspector->GetColumnTypeInfoAsHist(
      ROOT::Experimental::ENTupleInspectorHist::kCompressedSize, "compressedHist", "Compressed bytes per column type");
   EXPECT_STREQ("compressedHist", compressedSizeHist->GetName());
   EXPECT_STREQ("Compressed bytes per column type", compressedSizeHist->GetTitle());
   EXPECT_EQ(4U, compressedSizeHist->GetNbinsX());
   EXPECT_EQ(inspector->GetCompressedSize(), compressedSizeHist->Integral());

   auto uncompressedSizeHist = inspector->GetColumnTypeInfoAsHist(
      ROOT::Experimental::ENTupleInspectorHist::kUncompressedSize, "", "Uncompressed bytes per column type");
   EXPECT_STREQ("colTypeUncompSizeHist", uncompressedSizeHist->GetName());
   EXPECT_STREQ("Uncompressed bytes per column type", uncompressedSizeHist->GetTitle());
   EXPECT_EQ(4U, uncompressedSizeHist->GetNbinsX());
   EXPECT_EQ(inspector->GetUncompressedSize(), uncompressedSizeHist->Integral());
}

TEST(RNTupleInspector, PageSizeDistribution)
{
   FileRaii fileGuard("test_ntuple_inspector_page_size_distribution.root");
   {
      auto model = RNTupleModel::Create();
      auto nFldInt = model->MakeField<std::int64_t>("int");
      auto nFldFloat = model->MakeField<float>("float");
      auto nFldFloatVec = model->MakeField<std::vector<float>>("floatVec");
      auto nFldDoubleVec = model->MakeField<std::vector<double>>("doubleVec");

      auto writeOptions = RNTupleWriteOptions();
      writeOptions.SetCompression(505);
      writeOptions.SetInitialUnzippedPageSize(8);
      writeOptions.SetMaxUnzippedPageSize(64);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), writeOptions);

      for (unsigned i = 0; i < 100; ++i) {
         *nFldInt = static_cast<std::int64_t>(i);
         *nFldFloat = static_cast<float>(i) * .1f;
         *nFldFloatVec = {static_cast<float>(i), 3.14f, static_cast<float>(i) * *nFldFloat};
         *nFldDoubleVec = {};
         ntuple->Fill();
      }
   }

   auto inspector = RNTupleInspector::Create("ntuple", fileGuard.GetPath());

   int intColId = inspector->GetColumnsByType(ENTupleColumnType::kSplitInt64)[0];
   auto intPageSizeHisto = inspector->GetPageSizeDistribution(intColId);
   EXPECT_STREQ("pageSizeHist", intPageSizeHisto->GetName());
   EXPECT_STREQ(Form("Page size distribution for column with ID %d", intColId), intPageSizeHisto->GetTitle());
   EXPECT_STREQ("Page size (B)", intPageSizeHisto->GetXaxis()->GetTitle());
   EXPECT_STREQ("N_{pages}", intPageSizeHisto->GetYaxis()->GetTitle());
   EXPECT_EQ(64, intPageSizeHisto->GetNbinsX());
   // Make sure that all page sizes are included in the histogram
   int nIntPages = inspector->GetColumnInspector(intColId).GetNPages();
   EXPECT_EQ(nIntPages, intPageSizeHisto->Integral());

   auto floatPageSizeHisto = inspector->GetPageSizeDistribution(ENTupleColumnType::kSplitReal32, "floatPageSize",
                                                                "Float page size distribution", 100);
   EXPECT_STREQ("floatPageSize", floatPageSizeHisto->GetName());
   EXPECT_STREQ("Float page size distribution", floatPageSizeHisto->GetTitle());
   EXPECT_STREQ("Page size (B)", floatPageSizeHisto->GetXaxis()->GetTitle());
   EXPECT_STREQ("N_{pages}", floatPageSizeHisto->GetYaxis()->GetTitle());
   EXPECT_EQ(100, floatPageSizeHisto->GetNbinsX());
   // Make sure that all page sizes are included in the histogram
   int nFloatPages = 0;
   for (const auto colId : inspector->GetColumnsByType(ENTupleColumnType::kSplitReal32)) {
      nFloatPages += inspector->GetColumnInspector(colId).GetNPages();
   }
   EXPECT_EQ(nFloatPages, floatPageSizeHisto->Integral());

   auto multipleColsSizeHisto = inspector->GetPageSizeDistribution({0, 1, 2});
   EXPECT_STREQ("pageSizeHist", multipleColsSizeHisto->GetName());
   EXPECT_STREQ("Page size distribution", multipleColsSizeHisto->GetTitle());
   int nPages = inspector->GetColumnInspector(0).GetNPages() + inspector->GetColumnInspector(1).GetNPages() +
                inspector->GetColumnInspector(2).GetNPages();
   EXPECT_EQ(nPages, multipleColsSizeHisto->Integral());

   auto intFloatPageSizeHisto = inspector->GetPageSizeDistribution(
      {ENTupleColumnType::kSplitInt64, ENTupleColumnType::kSplitReal32}, "intFloatPageSize");
   EXPECT_STREQ("intFloatPageSize", intFloatPageSizeHisto->GetName());
   EXPECT_STREQ("Per-column type page size distribution", intFloatPageSizeHisto->GetTitle());
   EXPECT_EQ(2, intFloatPageSizeHisto->GetNhists());

   int intFloatIntegral = 0;
   for (auto hist : TRangeDynCast<TH1D>(intFloatPageSizeHisto->GetHists())) {
      intFloatIntegral += hist->Integral();
   }
   EXPECT_EQ(nIntPages + nFloatPages, intFloatIntegral);

   auto allColsSizeHisto = inspector->GetPageSizeDistribution();
   nPages = 0;
   for (const auto &col : inspector->GetDescriptor().GetColumnIterable()) {
      nPages += inspector->GetColumnInspector(col.GetPhysicalId()).GetNPages();
   }
   int allColsIntegral = 0;
   for (auto hist : TRangeDynCast<TH1D>(allColsSizeHisto->GetHists())) {
      allColsIntegral += hist->Integral();
   }
   EXPECT_EQ(nPages, allColsIntegral);

   // Requesting a histogram for a column with a physical ID not present in the given RNTuple should throw
   EXPECT_THROW(inspector->GetPageSizeDistribution(inspector->GetDescriptor().GetNPhysicalColumns() + 1),
                ROOT::RException);

   // Requesting a histogram for a column type not present in the given RNTuple should give an empty histogram
   auto nonExistingTypeHisto = inspector->GetPageSizeDistribution(ENTupleColumnType::kReal32);
   EXPECT_EQ(0, nonExistingTypeHisto->Integral());

   // Requesting a histogram for a column type without pages in the given RNTuple should give an empty histogram
   auto emptyTypeHisto = inspector->GetPageSizeDistribution(ENTupleColumnType::kSplitReal64);
   EXPECT_EQ(0, emptyTypeHisto->Integral());

   // Requesting a histogram for a column  without pages in the given RNTuple should give an empty histogram
   auto doubleColumns = inspector->GetColumnsByType(ENTupleColumnType::kSplitReal64);
   ASSERT_EQ(1, doubleColumns.size());
   auto emptyColumnHisto = inspector->GetPageSizeDistribution({doubleColumns[0]});
   EXPECT_EQ(0, emptyColumnHisto->Integral());
}

TEST(RNTupleInspector, FieldInfoCompressed)
{
   FileRaii fileGuard("test_ntuple_inspector_field_info_compressed.root");
   {
      auto model = RNTupleModel::Create();
      auto nFldObject = model->MakeField<ComplexStructUtil>("object");

      auto writeOptions = RNTupleWriteOptions();
      writeOptions.SetCompression(505);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), writeOptions);

      for (int i = 0; i < 25; ++i) {
         nFldObject->Init1();
         ntuple->Fill();
         nFldObject->Init2();
         ntuple->Fill();
         nFldObject->Init3();
         ntuple->Fill();
      }
   }

   auto inspector = RNTupleInspector::Create("ntuple", fileGuard.GetPath());

   auto topFieldInfo = inspector->GetFieldTreeInspector("object");

   EXPECT_GT(topFieldInfo.GetCompressedSize(), 0);
   EXPECT_EQ(topFieldInfo.GetUncompressedSize(), inspector->GetUncompressedSize());
   EXPECT_LT(topFieldInfo.GetCompressedSize(), topFieldInfo.GetUncompressedSize());

   std::uint64_t subFieldOnDiskSize = 0;
   std::uint64_t subFieldInMemorySize = 0;

   for (const auto &subField : inspector->GetDescriptor().GetFieldIterable(topFieldInfo.GetDescriptor().GetId())) {
      auto subFieldInfo = inspector->GetFieldTreeInspector(subField.GetId());
      subFieldOnDiskSize += subFieldInfo.GetCompressedSize();
      subFieldInMemorySize += subFieldInfo.GetUncompressedSize();
   }

   EXPECT_EQ(topFieldInfo.GetCompressedSize(), subFieldOnDiskSize);
   EXPECT_EQ(topFieldInfo.GetUncompressedSize(), subFieldInMemorySize);

   EXPECT_THROW(inspector->GetFieldTreeInspector("invalid_field"), ROOT::RException);
   EXPECT_THROW(inspector->GetFieldTreeInspector(inspector->GetDescriptor().GetNFields()), ROOT::RException);
}

TEST(RNTupleInspector, FieldInfoUncompressed)
{
   FileRaii fileGuard("test_ntuple_inspector_field_info_uncompressed.root");
   {
      auto model = RNTupleModel::Create();
      auto nFldObject = model->MakeField<ComplexStructUtil>("object");

      auto writeOptions = RNTupleWriteOptions();
      writeOptions.SetCompression(0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), writeOptions);

      for (int i = 0; i < 25; ++i) {
         nFldObject->Init1();
         ntuple->Fill();
         nFldObject->Init2();
         ntuple->Fill();
         nFldObject->Init3();
         ntuple->Fill();
      }
   }

   auto inspector = RNTupleInspector::Create("ntuple", fileGuard.GetPath());

   auto topFieldInfo = inspector->GetFieldTreeInspector("object");

   EXPECT_EQ(topFieldInfo.GetCompressedSize(), topFieldInfo.GetUncompressedSize());

   std::uint64_t subFieldOnDiskSize = 0;
   std::uint64_t subFieldInMemorySize = 0;

   for (const auto &subField : inspector->GetDescriptor().GetFieldIterable(topFieldInfo.GetDescriptor().GetId())) {
      auto subFieldInfo = inspector->GetFieldTreeInspector(subField.GetId());
      subFieldOnDiskSize += subFieldInfo.GetCompressedSize();
      subFieldInMemorySize += subFieldInfo.GetUncompressedSize();
   }

   EXPECT_EQ(topFieldInfo.GetCompressedSize(), subFieldOnDiskSize);
   EXPECT_EQ(topFieldInfo.GetUncompressedSize(), subFieldInMemorySize);
}

TEST(RNTupleInspector, FieldTypeCount)
{
   FileRaii fileGuard("test_ntuple_inspector_field_type_count.root");
   {
      auto model = RNTupleModel::Create();
      auto nFldObject = model->MakeField<ComplexStructUtil>("object");
      auto nFldInt1 = model->MakeField<std::int32_t>("int1");
      auto nFldInt2 = model->MakeField<std::int32_t>("int2");
      auto nFldInt3 = model->MakeField<std::int32_t>("int3");
      auto nFldString1 = model->MakeField<std::string>("string1");
      auto nFldString2 = model->MakeField<std::string>("string2");

      auto writeOptions = RNTupleWriteOptions();
      writeOptions.SetCompression(505);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), writeOptions);
   }

   auto inspector = RNTupleInspector::Create("ntuple", fileGuard.GetPath());

   EXPECT_EQ(1, inspector->GetFieldCountByType("ComplexStructUtil"));
   EXPECT_EQ(1, inspector->GetFieldCountByType("ComplexStructUtil", false));

   EXPECT_EQ(1, inspector->GetFieldCountByType("std::vector<HitUtil>"));
   EXPECT_EQ(0, inspector->GetFieldCountByType("std::vector<HitUtil>", false));

   EXPECT_EQ(2, inspector->GetFieldCountByType("std::vector<.*>"));
   EXPECT_EQ(0, inspector->GetFieldCountByType("std::vector<.*>", false));

   EXPECT_EQ(3, inspector->GetFieldCountByType("BaseUtil"));
   EXPECT_EQ(0, inspector->GetFieldCountByType("BaseUtil", false));

   EXPECT_EQ(6, inspector->GetFieldCountByType("std::int32_t"));
   EXPECT_EQ(3, inspector->GetFieldCountByType("std::int32_t", false));

   EXPECT_EQ(4, inspector->GetFieldCountByType("float"));
   EXPECT_EQ(0, inspector->GetFieldCountByType("float", false));
}

TEST(RNTupleInspector, FieldsByName)
{
   FileRaii fileGuard("test_ntuple_inspector_fields_by_name.root");
   {
      auto model = RNTupleModel::Create();
      auto nFldInt1 = model->MakeField<std::int32_t>("int1");
      auto nFldInt2 = model->MakeField<std::int32_t>("int2");
      auto nFldInt3 = model->MakeField<std::int32_t>("int3");
      auto nFldFloat1 = model->MakeField<float>("float1");
      auto nFldFloat2 = model->MakeField<float>("float2");

      auto writeOptions = RNTupleWriteOptions();
      writeOptions.SetCompression(505);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), writeOptions);
   }

   auto inspector = RNTupleInspector::Create("ntuple", fileGuard.GetPath());

   auto intFieldIds = inspector->GetFieldsByName("int.");

   EXPECT_EQ(3, intFieldIds.size());
   for (const auto fieldId : intFieldIds) {
      EXPECT_EQ("std::int32_t", inspector->GetFieldTreeInspector(fieldId).GetDescriptor().GetTypeName());
   }
}

TEST(RNTupleInspector, MultiColumnRepresentations)
{
   FileRaii fileGuard("test_ntuple_inspector_multi_column_representations.root");
   {
      auto model = RNTupleModel::Create();
      auto fldPx = RFieldBase::Create("px", "float").Unwrap();
      fldPx->SetColumnRepresentatives({{ENTupleColumnType::kReal32}, {ENTupleColumnType::kReal16}});
      model->AddField(std::move(fldPx));
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();
      writer->CommitCluster();
      ROOT::Internal::RFieldRepresentationModifier::SetPrimaryColumnRepresentation(
         const_cast<RFieldBase &>(writer->GetModel().GetConstField("px")), 1);
      writer->Fill();
   }

   auto inspector = RNTupleInspector::Create("ntpl", fileGuard.GetPath());
   auto px0Inspector = inspector->GetColumnInspector(0);
   auto px1Inspector = inspector->GetColumnInspector(1);
   EXPECT_EQ(ENTupleColumnType::kReal32, px0Inspector.GetType());
   EXPECT_EQ(1u, px0Inspector.GetNElements());
   EXPECT_EQ(ENTupleColumnType::kReal16, px1Inspector.GetType());
   EXPECT_EQ(1u, px1Inspector.GetNElements());
}

TEST(RNTupleInspector, FieldTreeAsDot)
{
   FileRaii fileGuard("test_ntuple_inspector_fields_tree_as_dot.root");
   {
      auto model = RNTupleModel::Create();
      auto fldFloat1 = model->MakeField<float>("float1");
      auto fldInt = model->MakeField<std::int32_t>("int");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
   }
   auto inspector = RNTupleInspector::Create("ntuple", fileGuard.GetPath());
   std::ostringstream dotStream;
   inspector->PrintFieldTreeAsDot(dotStream);
   const std::string dot = dotStream.str();
   const std::string &expected =
      "digraph D {\nnode[shape=box]\n0[label=<<b>RFieldZero</b>>]\n0->1\n1[label=<<b>Name: "
      "</b>float1<br></br><b>Type: </b>float<br></br><b>ID: </b>0<br></br>>]\n0->2\n2[label=<<b>Name: "
      "</b>int<br></br><b>Type: </b>std::int32_t<br></br><b>ID: </b>1<br></br>>]\n}";
   EXPECT_EQ(dot, expected);
}

TEST(RNTupleInspector, SchemaProfile)
{
   FileRaii fileGuard("test_schema_profile.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldFloat1 = model->MakeField<float>("float1");
      auto fieldInt = model->MakeField<std::int32_t>("int");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());

      for (int i = 0; i < 10; ++i) {
         *fieldFloat1 = 3.14f * i;
         *fieldInt = 42 * i;
         writer->Fill();
      }
   }
   auto inspector = RNTupleInspector::Create("ntuple", fileGuard.GetPath());
   std::ostringstream schemaProfileStream;
   inspector->PrintSchemaProfile(ROOT::Experimental::ESchemaProfileFormat::kSpeedscopeJSON, schemaProfileStream);
   const std::string schemaProfile = schemaProfileStream.str();
   const std::string expected = R"foo({
   "$schema":"https://www.speedscope.app/file-format-schema.json",
   "shared":{
      "frames":[
         { "name":"float1 (float)" },
         { "name":"[col#0] float1 (SplitReal32)" },
         { "name":"int (std::int32_t)" },
         { "name":"[col#1] int (SplitInt32)" }
      ]
   },
   "profiles":[
      {
         "type":"evented",
         "name":"Flattened Timeline",
         "unit":"bytes",
         "startValue":0,
         "endValue":80,
         "events":[
            {"type":"O","frame":0,"at":0},
            {"type":"O","frame":1,"at":0},
            {"type":"C","frame":1,"at":40},
            {"type":"C","frame":0,"at":40},
            {"type":"O","frame":2,"at":40},
            {"type":"O","frame":3,"at":40},
            {"type":"C","frame":3,"at":80},
            {"type":"C","frame":2,"at":80}
         ]
      }
   ]
}
)foo";
   EXPECT_EQ(schemaProfile, expected);
}

namespace {

void WriteShuffledNTuple(const std::string &ntupleName, const std::string &path)
{
   // Specify logical schema
   RNTupleDescriptorBuilder nTupleDescriptorBuilder;

   nTupleDescriptorBuilder.SetVersionForWriting();
   nTupleDescriptorBuilder.SetNTuple(ntupleName, "Non-contiguous cluster group, cluster and column range");

   nTupleDescriptorBuilder.AddField(RFieldDescriptorBuilder()
                                       .FieldId(0)
                                       .FieldName("")
                                       .Structure(ROOT::ENTupleStructure::kRecord)
                                       .MakeDescriptor()
                                       .Unwrap());

   for (std::uint32_t i = 0; i < 6; ++i) {
      const ROOT::DescriptorId_t fieldId = 1 + i;
      const ROOT::DescriptorId_t columnId = i;

      nTupleDescriptorBuilder.AddField(RFieldDescriptorBuilder()
                                          .FieldId(fieldId)
                                          .FieldName("tag" + std::to_string(i))
                                          .Structure(ROOT::ENTupleStructure::kPlain)
                                          .MakeDescriptor()
                                          .Unwrap());

      nTupleDescriptorBuilder.AddFieldLink(0, fieldId).ThrowOnError();

      nTupleDescriptorBuilder.AddColumn(RColumnDescriptorBuilder()
                                           .LogicalColumnId(columnId)
                                           .PhysicalColumnId(columnId)
                                           .FieldId(fieldId)
                                           .BitsOnStorage(32)
                                           .Type(ROOT::ENTupleColumnType::kIndex32)
                                           .Index(0)
                                           .MakeDescriptor()
                                           .Unwrap());
   }

   RNTupleWriteOptions options;
   auto writer =
      RNTupleFileWriter::Recreate("shuffled_ntuple", path, RNTupleFileWriter::EContainerFormat::kTFile, options);

   const RNTupleDescriptor &schemaDescriptor = nTupleDescriptorBuilder.GetDescriptor();

   // Serialize and write header
   auto context = RNTupleSerializer::SerializeHeader(nullptr, schemaDescriptor).Unwrap();
   auto headerBuffer = MakeUninitArray<unsigned char>(context.GetHeaderSize());
   context = RNTupleSerializer::SerializeHeader(headerBuffer.get(), schemaDescriptor).Unwrap();
   writer->WriteNTupleHeader(headerBuffer.get(), context.GetHeaderSize(), context.GetHeaderSize());

   // Serialize and write pages
   auto serializePage = [&](std::uint32_t firstValue, std::uint32_t numberOfElements, bool addChecksum) {
      const std::size_t payloadBytes = std::size_t(numberOfElements) * 4; // kIndex32 -> 4 bytes/element
      const std::size_t blobBytes = payloadBytes + (addChecksum ? 8 : 0);
      auto blob = MakeUninitArray<unsigned char>(blobBytes);

      for (std::uint32_t i = 0; i < numberOfElements; ++i)
         RNTupleSerializer::SerializeUInt32(firstValue + i, blob.get() + i * 4);

      if (addChecksum) {
         std::uint64_t xxhash3 = 0;
         RNTupleSerializer::SerializeXxHash3(blob.get(), payloadBytes, xxhash3, blob.get() + payloadBytes);
      }

      const std::uint64_t offset = writer->WriteBlob(blob.get(), blobBytes, payloadBytes);

      ROOT::RClusterDescriptor::RPageInfo pageInfo;
      pageInfo.SetNElements(numberOfElements);
      pageInfo.SetHasChecksum(addChecksum);
      pageInfo.GetLocator().SetPosition(offset);
      pageInfo.GetLocator().SetNBytesOnStorage(payloadBytes); // excludes the checksum
      return pageInfo;
   };

   auto page1 = serializePage(0, 50, true);
   auto page2 = serializePage(50, 25, false);
   auto page3 = serializePage(0, 100, true);
   auto page4 = serializePage(75, 25, false);
   auto page5 = serializePage(0, 100, true);
   auto page6 = serializePage(0, 100, false);
   auto page7 = serializePage(0, 100, true);
   auto page8 = serializePage(0, 100, false);

   auto makePageRange = [](ROOT::DescriptorId_t physicalColumnID,
                           std::initializer_list<ROOT::RClusterDescriptor::RPageInfo> pages) {
      ROOT::RClusterDescriptor::RPageRange pageRange;
      pageRange.SetPhysicalColumnId(physicalColumnID);
      for (const auto &p : pages)
         pageRange.GetPageInfos().emplace_back(p);
      return pageRange;
   };

   // Specify clusters and column ranges
   {
      RClusterDescriptorBuilder builder;
      builder.ClusterId(0).FirstEntryIndex(0).NEntries(100);
      builder.CommitColumnRange(0, 0, 0, makePageRange(0, {page1, page2, page4})).ThrowOnError();
      builder.CommitColumnRange(1, 0, 0, makePageRange(1, {page3})).ThrowOnError();
      builder.CommitColumnRange(3, 0, 0, makePageRange(3, {page6})).ThrowOnError();
      nTupleDescriptorBuilder.AddCluster(builder.MoveDescriptor().Unwrap()).ThrowOnError();
   }
   {
      RClusterDescriptorBuilder builder;
      builder.ClusterId(1).FirstEntryIndex(100).NEntries(100);
      builder.CommitColumnRange(2, 0, 0, makePageRange(2, {page5})).ThrowOnError();
      nTupleDescriptorBuilder.AddCluster(builder.MoveDescriptor().Unwrap()).ThrowOnError();
   }
   {
      RClusterDescriptorBuilder builder;
      builder.ClusterId(2).FirstEntryIndex(200).NEntries(100);
      builder.CommitColumnRange(4, 0, 0, makePageRange(4, {page7})).ThrowOnError();
      nTupleDescriptorBuilder.AddCluster(builder.MoveDescriptor().Unwrap()).ThrowOnError();
   }
   {
      RClusterDescriptorBuilder builder;
      builder.ClusterId(3).FirstEntryIndex(300).NEntries(100);
      builder.CommitColumnRange(5, 0, 0, makePageRange(5, {page8})).ThrowOnError();
      nTupleDescriptorBuilder.AddCluster(builder.MoveDescriptor().Unwrap()).ThrowOnError();
   }

   std::vector<ROOT::DescriptorId_t> clusterGroup0Clusters{0, 1, 3};
   std::vector<ROOT::DescriptorId_t> clusterGroup1Clusters{2};
   std::vector<ROOT::DescriptorId_t> clusterGroup0PhysicalID, clusterGroup1PhysicalID;

   for (auto id : clusterGroup0Clusters)
      clusterGroup0PhysicalID.emplace_back(context.MapClusterId(id));
   for (auto id : clusterGroup1Clusters)
      clusterGroup1PhysicalID.emplace_back(context.MapClusterId(id));

   // Serialize and write page lists
   auto writePageList = [&](std::vector<ROOT::DescriptorId_t> &physicalClusterIds, RNTupleLocator &locator) {
      const auto size = RNTupleSerializer::SerializePageList(nullptr, nTupleDescriptorBuilder.GetDescriptor(),
                                                             physicalClusterIds, context)
                           .Unwrap();
      auto buf = MakeUninitArray<unsigned char>(size);
      RNTupleSerializer::SerializePageList(buf.get(), nTupleDescriptorBuilder.GetDescriptor(), physicalClusterIds,
                                           context)
         .Unwrap();
      const std::uint64_t pageListOffset = writer->WriteBlob(buf.get(), size, size);
      locator.SetPosition(pageListOffset);
      locator.SetNBytesOnStorage(size);
      return size;
   };

   // Specify cluster groups
   RNTupleLocator clusterGroup0Location, clusterGroup1Location;
   const auto clusterGroup0Size = writePageList(clusterGroup0PhysicalID, clusterGroup0Location);
   const auto clusterGroup1Size = writePageList(clusterGroup1PhysicalID, clusterGroup1Location);

   {
      RClusterGroupDescriptorBuilder builder;
      builder.ClusterGroupId(0)
         .PageListLength(clusterGroup0Size)
         .PageListLocator(clusterGroup0Location)
         .MinEntry(0)
         .EntrySpan(400)
         .NClusters(3);
      builder.AddSortedClusters(clusterGroup0Clusters);
      nTupleDescriptorBuilder.AddClusterGroup(builder.MoveDescriptor().Unwrap()).ThrowOnError();
      context.MapClusterGroupId(0);
   }
   {
      RClusterGroupDescriptorBuilder builder;
      builder.ClusterGroupId(1)
         .PageListLength(clusterGroup1Size)
         .PageListLocator(clusterGroup1Location)
         .MinEntry(200)
         .EntrySpan(100)
         .NClusters(1);
      builder.AddSortedClusters(clusterGroup1Clusters);
      nTupleDescriptorBuilder.AddClusterGroup(builder.MoveDescriptor().Unwrap()).ThrowOnError();
      context.MapClusterGroupId(1);
   }

   auto nTupleDescriptor = nTupleDescriptorBuilder.MoveDescriptor();

   // Serialize and write footer
   const auto footerSize = RNTupleSerializer::SerializeFooter(nullptr, nTupleDescriptor, context).Unwrap();
   auto footerBuffer = MakeUninitArray<unsigned char>(footerSize);
   RNTupleSerializer::SerializeFooter(footerBuffer.get(), nTupleDescriptor, context).Unwrap();
   writer->WriteNTupleFooter(footerBuffer.get(), footerSize, footerSize);

   // Commit writes and call writer destructor to flush it
   writer->Commit();
   writer = nullptr;
}
} // namespace

TEST(RNTupleInspector, DiskProfile)
{
   FileRaii fileGuard("test_disk_profile.root");
   WriteShuffledNTuple("shuffled_ntuple", fileGuard.GetPath());

   auto inspector = RNTupleInspector::Create("shuffled_ntuple", fileGuard.GetPath());
   std::ostringstream diskProfileStream;
   inspector->PrintDiskProfile(ROOT::Experimental::ESchemaProfileFormat::kSpeedscopeJSON, diskProfileStream);
   const std::string diskProfile = diskProfileStream.str();
   const std::string expected = R"foo({
   "$schema":"https://www.speedscope.app/file-format-schema.json",
   "shared":{
      "frames":[
         { "name":"ntuple header" },
         { "name":"[cluster group 0]" },
         { "name":"[cluster 0]" },
         { "name":"[column range 0]" },
         { "name":"[page @882]" },
         { "name":"[page @1132]" },
         { "name":"[column range 1]" },
         { "name":"[page @1274]" },
         { "name":"[column range 0]" },
         { "name":"[page @1724]" },
         { "name":"[cluster 1]" },
         { "name":"[column range 0]" },
         { "name":"[page @1866]" },
         { "name":"[cluster 0]" },
         { "name":"[column range 2]" },
         { "name":"[page @2316]" },
         { "name":"[cluster group 1]" },
         { "name":"[cluster 3]" },
         { "name":"[column range 0]" },
         { "name":"[page @2758]" },
         { "name":"[cluster group 0]" },
         { "name":"[cluster 2]" },
         { "name":"[column range 0]" },
         { "name":"[page @3208]" },
         { "name":"[page list 0]" },
         { "name":"[page list 1]" },
         { "name":"ntuple footer" }
      ]
   },
   "profiles":[
      {
         "type":"evented",
         "name":"Flattened Timeline",
         "unit":"bytes",
         "startValue":0,
         "endValue":4454,
         "events":[
            {"type":"O","frame":0,"at":290},
            {"type":"C","frame":0,"at":840},
            {"type":"O","frame":1,"at":882},
            {"type":"O","frame":2,"at":882},
            {"type":"O","frame":3,"at":882},
            {"type":"O","frame":4,"at":882},
            {"type":"C","frame":4,"at":1082},
            {"type":"O","frame":5,"at":1132},
            {"type":"C","frame":5,"at":1232},
            {"type":"C","frame":3,"at":1232},
            {"type":"O","frame":6,"at":1274},
            {"type":"O","frame":7,"at":1274},
            {"type":"C","frame":7,"at":1674},
            {"type":"C","frame":6,"at":1674},
            {"type":"O","frame":8,"at":1724},
            {"type":"O","frame":9,"at":1724},
            {"type":"C","frame":9,"at":1824},
            {"type":"C","frame":8,"at":1824},
            {"type":"C","frame":2,"at":1824},
            {"type":"O","frame":10,"at":1866},
            {"type":"O","frame":11,"at":1866},
            {"type":"O","frame":12,"at":1866},
            {"type":"C","frame":12,"at":2266},
            {"type":"C","frame":11,"at":2266},
            {"type":"C","frame":10,"at":2266},
            {"type":"O","frame":13,"at":2316},
            {"type":"O","frame":14,"at":2316},
            {"type":"O","frame":15,"at":2316},
            {"type":"C","frame":15,"at":2716},
            {"type":"C","frame":14,"at":2716},
            {"type":"C","frame":13,"at":2716},
            {"type":"C","frame":1,"at":2716},
            {"type":"O","frame":16,"at":2758},
            {"type":"O","frame":17,"at":2758},
            {"type":"O","frame":18,"at":2758},
            {"type":"O","frame":19,"at":2758},
            {"type":"C","frame":19,"at":3158},
            {"type":"C","frame":18,"at":3158},
            {"type":"C","frame":17,"at":3158},
            {"type":"C","frame":16,"at":3158},
            {"type":"O","frame":20,"at":3208},
            {"type":"O","frame":21,"at":3208},
            {"type":"O","frame":22,"at":3208},
            {"type":"O","frame":23,"at":3208},
            {"type":"C","frame":23,"at":3608},
            {"type":"C","frame":22,"at":3608},
            {"type":"C","frame":21,"at":3608},
            {"type":"C","frame":20,"at":3608},
            {"type":"O","frame":24,"at":3650},
            {"type":"C","frame":24,"at":4038},
            {"type":"O","frame":25,"at":4080},
            {"type":"C","frame":25,"at":4204},
            {"type":"O","frame":26,"at":4246},
            {"type":"C","frame":26,"at":4454}
         ]
      }
   ]
}
)foo";
   EXPECT_EQ(diskProfile, expected);
}
