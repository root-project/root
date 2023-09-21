#include <ROOT/RNTupleInspector.hxx>
#include <ROOT/RNTupleOptions.hxx>

#include <TFile.h>
#include <ROOT/TestSupport.hxx>

#include "CustomStructUtil.hxx"
#include "ntupleutil_test.hxx"

using ROOT::Experimental::RField;
using ROOT::Experimental::RNTuple;
using ROOT::Experimental::RNTupleInspector;
using ROOT::Experimental::RNTupleModel;
using ROOT::Experimental::RNTupleWriteOptions;
using ROOT::Experimental::RNTupleWriter;

TEST(RNTupleInspector, CreateFromPointer)
{
   FileRaii fileGuard("test_ntuple_inspector_create_from_pointer.root");
   {
      auto model = RNTupleModel::Create();
      RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
   }

   std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str()));
   auto ntuple = file->Get<RNTuple>("ntuple");
   auto inspector = RNTupleInspector::Create(ntuple);
   EXPECT_EQ(inspector->GetDescriptor()->GetName(), "ntuple");

   auto nullNTuple = file->Get<RNTuple>("null");
   EXPECT_THROW(RNTupleInspector::Create(nullNTuple), ROOT::Experimental::RException);
}

TEST(RNTupleInspector, CreateFromString)
{
   FileRaii fileGuard("test_ntuple_inspector_create_from_string.root");
   {
      RNTupleWriter::Recreate(RNTupleModel::Create(), "ntuple", fileGuard.GetPath());
   }

   auto inspector = RNTupleInspector::Create("ntuple", fileGuard.GetPath());
   EXPECT_EQ(inspector->GetDescriptor()->GetName(), "ntuple");

   EXPECT_THROW(RNTupleInspector::Create("nonexistent", fileGuard.GetPath()), ROOT::Experimental::RException);

   ROOT::TestSupport::CheckDiagsRAII diag;
   diag.requiredDiag(kError, "TFile::TFile", "nonexistent.root does not exist",
                     /*matchFullMessage=*/false);
   EXPECT_THROW(RNTupleInspector::Create("ntuple", "nonexistent.root"), ROOT::Experimental::RException);
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

   EXPECT_EQ(207, inspector->GetCompressionSettings());
   EXPECT_EQ("LZMA (level 7)", inspector->GetCompressionSettingsAsString());
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

   int nIndexCols = inspector->GetColumnCountByType(ROOT::Experimental::EColumnType::kIndex64);
   int nEntries = inspector->GetDescriptor()->GetNEntries();

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

      for (int32_t i = 0; i < 50; ++i) {
         *nFldInt = i;
         ntuple->Fill();
      }
   }

   auto inspector = RNTupleInspector::Create("ntuple", fileGuard.GetPath());

   EXPECT_EQ(sizeof(int32_t) * 50, inspector->GetUncompressedSize());
   EXPECT_GT(inspector->GetCompressedSize(), 0);
   EXPECT_LT(inspector->GetCompressedSize(), inspector->GetUncompressedSize());
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

   EXPECT_GT(inspector->GetCompressedSize(), 0);
   EXPECT_LT(inspector->GetCompressedSize(), inspector->GetUncompressedSize());
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

   for (std::size_t i = 0; i < inspector->GetDescriptor()->GetNLogicalColumns(); ++i) {
      auto colInfo = inspector->GetColumnInfo(i);
      totalOnDiskSize += colInfo.GetCompressedSize();

      EXPECT_GT(colInfo.GetCompressedSize(), 0);
      EXPECT_GT(colInfo.GetUncompressedSize(), 0);
      EXPECT_LT(colInfo.GetCompressedSize(), colInfo.GetUncompressedSize());
   }

   EXPECT_EQ(totalOnDiskSize, inspector->GetCompressedSize());

   EXPECT_THROW(inspector->GetColumnInfo(42), ROOT::Experimental::RException);
}

TEST(RNTupleInspector, ColumnInfoUncompressed)
{
   FileRaii fileGuard("test_ntuple_inspector_column_info_uncompressed.root");
   {
      auto model = RNTupleModel::Create();

      auto int32fld = std::make_unique<RField<std::int32_t>>("int32");
      int32fld->SetColumnRepresentative({ROOT::Experimental::EColumnType::kInt32});
      model->AddField(std::move(int32fld));

      auto splitReal64fld = std::make_unique<RField<double>>("splitReal64");
      splitReal64fld->SetColumnRepresentative({ROOT::Experimental::EColumnType::kSplitReal64});
      model->AddField(std::move(splitReal64fld));

      auto writeOptions = RNTupleWriteOptions();
      writeOptions.SetCompression(0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), writeOptions);

      for (int i = 0; i < 5; ++i) {
         auto e = ntuple->CreateEntry();
         *e->Get<std::int32_t>("int32") = i;
         *e->Get<double>("splitReal64") = i;
         ntuple->Fill(*e);
      }
   }

   auto inspector = RNTupleInspector::Create("ntuple", fileGuard.GetPath());

   std::uint64_t colTypeSizes[] = {sizeof(std::int32_t), sizeof(double)};

   for (std::size_t i = 0; i < inspector->GetDescriptor()->GetNLogicalColumns(); ++i) {
      auto colInfo = inspector->GetColumnInfo(i);
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

   EXPECT_EQ(2, inspector->GetColumnCountByType(ROOT::Experimental::EColumnType::kSplitIndex64));
   EXPECT_EQ(4, inspector->GetColumnCountByType(ROOT::Experimental::EColumnType::kSplitReal32));
   EXPECT_EQ(3, inspector->GetColumnCountByType(ROOT::Experimental::EColumnType::kSplitInt32));
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

   EXPECT_EQ(2U, inspector->GetColumnsByType(ROOT::Experimental::EColumnType::kSplitInt64).size());
   for (const auto colId : inspector->GetColumnsByType(ROOT::Experimental::EColumnType::kSplitInt64)) {
      EXPECT_EQ(ROOT::Experimental::EColumnType::kSplitInt64, inspector->GetColumnInfo(colId).GetType());
   }

   EXPECT_EQ(2U, inspector->GetColumnsByType(ROOT::Experimental::EColumnType::kSplitReal32).size());
   for (const auto colId : inspector->GetColumnsByType(ROOT::Experimental::EColumnType::kSplitReal32)) {
      EXPECT_EQ(ROOT::Experimental::EColumnType::kSplitReal32, inspector->GetColumnInfo(colId).GetType());
   }

   EXPECT_EQ(1U, inspector->GetColumnsByType(ROOT::Experimental::EColumnType::kSplitIndex64).size());
   EXPECT_EQ(1U, inspector->GetColumnsByType(ROOT::Experimental::EColumnType::kSplitIndex64).size());
   for (const auto colId : inspector->GetColumnsByType(ROOT::Experimental::EColumnType::kSplitIndex64)) {
      EXPECT_EQ(ROOT::Experimental::EColumnType::kSplitIndex64, inspector->GetColumnInfo(colId).GetType());
   }

   EXPECT_EQ(0U, inspector->GetColumnsByType(ROOT::Experimental::EColumnType::kSplitReal64).size());
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
   EXPECT_EQ("columnType,count,nElements,compressedSize,uncompressedSize", line);

   size_t nLines = 0;
   std::string colTypeStr;
   while (std::getline(csvOutput, line)) {
      ++nLines;
      colTypeStr = line.substr(0, line.find(","));

      if (colTypeStr != "SplitIndex64" && colTypeStr != "SplitInt64" && colTypeStr != "SplitReal32")
         FAIL() << "Unexpected column type: " << colTypeStr;
   }
   EXPECT_EQ(nLines, 3U);

   std::stringstream tableOutput;
   inspector->PrintColumnTypeInfo(ROOT::Experimental::ENTupleInspectorPrintFormat::kTable, tableOutput);

   std::getline(tableOutput, line);
   EXPECT_EQ(" column type    | count   | # elements      | compressed bytes  | uncompressed bytes", line);

   std::getline(tableOutput, line);
   EXPECT_EQ("----------------|---------|-----------------|-------------------|--------------------", line);

   nLines = 0;
   while (std::getline(tableOutput, line)) {
      ++nLines;
      colTypeStr = line.substr(0, line.find("|"));
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
   EXPECT_EQ(inspector->GetDescriptor()->GetNPhysicalColumns(), countHist->Integral());

   auto nElemsHist = inspector->GetColumnTypeInfoAsHist(ROOT::Experimental::ENTupleInspectorHist::kNElems, "elemsHist");
   EXPECT_STREQ("elemsHist", nElemsHist->GetName());
   EXPECT_STREQ("Number of elements by column type", nElemsHist->GetTitle());
   EXPECT_EQ(4U, nElemsHist->GetNbinsX());
   std::uint64_t nTotalElems = 0;
   for (const auto &col : inspector->GetDescriptor()->GetColumnIterable()) {
      nTotalElems += inspector->GetDescriptor()->GetNElements(col.GetPhysicalId());
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

   auto topFieldInfo = inspector->GetFieldTreeInfo("object");

   EXPECT_GT(topFieldInfo.GetCompressedSize(), 0);
   EXPECT_EQ(topFieldInfo.GetUncompressedSize(), inspector->GetUncompressedSize());
   EXPECT_LT(topFieldInfo.GetCompressedSize(), topFieldInfo.GetUncompressedSize());

   std::uint64_t subFieldOnDiskSize = 0;
   std::uint64_t subFieldInMemorySize = 0;

   for (const auto &subField : inspector->GetDescriptor()->GetFieldIterable(topFieldInfo.GetDescriptor().GetId())) {
      auto subFieldInfo = inspector->GetFieldTreeInfo(subField.GetId());
      subFieldOnDiskSize += subFieldInfo.GetCompressedSize();
      subFieldInMemorySize += subFieldInfo.GetUncompressedSize();
   }

   EXPECT_EQ(topFieldInfo.GetCompressedSize(), subFieldOnDiskSize);
   EXPECT_EQ(topFieldInfo.GetUncompressedSize(), subFieldInMemorySize);

   EXPECT_THROW(inspector->GetFieldTreeInfo("invalid_field"), ROOT::Experimental::RException);
   EXPECT_THROW(inspector->GetFieldTreeInfo(inspector->GetDescriptor()->GetNFields()), ROOT::Experimental::RException);
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

   auto topFieldInfo = inspector->GetFieldTreeInfo("object");

   EXPECT_EQ(topFieldInfo.GetCompressedSize(), topFieldInfo.GetUncompressedSize());

   std::uint64_t subFieldOnDiskSize = 0;
   std::uint64_t subFieldInMemorySize = 0;

   for (const auto &subField : inspector->GetDescriptor()->GetFieldIterable(topFieldInfo.GetDescriptor().GetId())) {
      auto subFieldInfo = inspector->GetFieldTreeInfo(subField.GetId());
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
      EXPECT_EQ("std::int32_t", inspector->GetFieldTreeInfo(fieldId).GetDescriptor().GetTypeName());
   }
}
