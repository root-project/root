#include <ROOT/RNTupleInspector.hxx>
#include <ROOT/RNTupleOptions.hxx>

#include <TFile.h>

#include "CustomStructUtil.hxx"
#include "ntupleutil_test.hxx"

using ROOT::Experimental::RNTuple;
using ROOT::Experimental::RNTupleInspector;
using ROOT::Experimental::RNTupleModel;
using ROOT::Experimental::RNTupleReader;
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
   EXPECT_EQ("ntuple", inspector->GetDescriptor()->GetName());

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
   EXPECT_EQ("ntuple", inspector->GetDescriptor()->GetName());

   EXPECT_THROW(RNTupleInspector::Create("nonexistent", fileGuard.GetPath()), ROOT::Experimental::RException);
   EXPECT_THROW(RNTupleInspector::Create("ntuple", "nonexistent.root"), ROOT::Experimental::RException);
}

TEST(RNTupleInspector, NEntries)
{
   FileRaii fileGuard("test_ntuple_inspector_n_entries.root");
   {
      auto model = RNTupleModel::Create();
      auto nFldInt = model->MakeField<std::int32_t>("i");

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());

      for (int32_t i = 0; i < 50; ++i) {
         *nFldInt = i;
         ntuple->Fill();
      }
   }

   std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str()));
   auto ntuple = file->Get<RNTuple>("ntuple");
   auto inspector = RNTupleInspector::Create(ntuple);

   auto descriptor = inspector->GetDescriptor();

   EXPECT_EQ(descriptor->GetNEntries(), 50);
   EXPECT_EQ(descriptor->GetName(), "ntuple");
}

TEST(RNTupleInspector, CompressionSettings)
{
   FileRaii fileGuard("test_ntuple_inspector_compression_settings.root");
   {
      auto model = RNTupleModel::Create();
      auto nFldInt = model->MakeField<std::int32_t>("int");

      auto writeOptions = RNTupleWriteOptions();
      writeOptions.SetCompression(505);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), writeOptions);

      *nFldInt = 42;
      ntuple->Fill();
   }

   std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str()));
   auto ntuple = file->Get<RNTuple>("ntuple");
   auto inspector = RNTupleInspector::Create(ntuple);

   EXPECT_EQ(505, inspector->GetCompressionSettings());
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

   std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str()));
   auto ntuple = file->Get<RNTuple>("ntuple");
   auto inspector = RNTupleInspector::Create(ntuple);

   EXPECT_EQ(sizeof(int32_t) * 5, inspector->GetInMemorySize());

   // N.B. This property only holds for ntuples without Index fields, due to
   // the 64-bit in-memory vs. 32-bit on-disk optimization.
   EXPECT_EQ(inspector->GetOnDiskSize(), inspector->GetInMemorySize());
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

   std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str()));
   auto ntuple = file->Get<RNTuple>("ntuple");
   auto inspector = RNTupleInspector::Create(ntuple);

   int nIndexCols = inspector->GetColumnTypeCount(ROOT::Experimental::EColumnType::kIndex32);
   int nEntries = inspector->GetDescriptor()->GetNEntries();

   EXPECT_EQ(2, nIndexCols);
   EXPECT_EQ(3, nEntries);

   // We need to add another 4 bytes per index column per event, due to the 64-bit
   // in-memory vs. 32-bit on-disk optimization.
   EXPECT_EQ(inspector->GetOnDiskSize() + 4 * nIndexCols * nEntries, inspector->GetInMemorySize());
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

   std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str()));
   auto ntuple = file->Get<RNTuple>("ntuple");
   auto inspector = RNTupleInspector::Create(ntuple);

   EXPECT_EQ(sizeof(int32_t) * 50, inspector->GetInMemorySize());
   EXPECT_GT(inspector->GetOnDiskSize(), 0);
   EXPECT_LT(inspector->GetOnDiskSize(), inspector->GetInMemorySize());
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

      nFldObject->Init1();
      ntuple->Fill();
      nFldObject->Init2();
      ntuple->Fill();
      nFldObject->Init3();
      ntuple->Fill();
   }

   std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str()));
   auto ntuple = file->Get<RNTuple>("ntuple");
   auto inspector = RNTupleInspector::Create(ntuple);

   EXPECT_GT(inspector->GetOnDiskSize(), 0);
   EXPECT_LT(inspector->GetOnDiskSize(), inspector->GetInMemorySize());
   EXPECT_GT(inspector->GetCompressionFactor(), 1);
}

TEST(RNTupleInspector, SizeEmpty)
{
   FileRaii fileGuard("test_ntuple_inspector_size_empty.root");
   {
      auto model = RNTupleModel::Create();
      RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
   }

   std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str()));
   auto ntuple = file->Get<RNTuple>("ntuple");
   auto inspector = RNTupleInspector::Create(ntuple);

   EXPECT_EQ(0, inspector->GetOnDiskSize());
   EXPECT_EQ(0, inspector->GetInMemorySize());
}

TEST(RNTupleInspector, FieldTypeCount)
{
   FileRaii fileGuard("test_ntuple_inspector_field_type_frequency.root");
   {
      auto model = RNTupleModel::Create();
      auto nFldObject = model->MakeField<ComplexStructUtil>("object");

      auto writeOptions = RNTupleWriteOptions();
      writeOptions.SetCompression(505);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), writeOptions);
   }

   std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str()));
   auto ntuple = file->Get<RNTuple>("ntuple");
   auto inspector = RNTupleInspector::Create(ntuple);

   EXPECT_EQ(1, inspector->GetFieldTypeCount("ComplexStructUtil"));
   EXPECT_EQ(1, inspector->GetFieldTypeCount("ComplexStructUtil", false));

   EXPECT_EQ(1, inspector->GetFieldTypeCount("std::vector<HitUtil>"));
   EXPECT_EQ(0, inspector->GetFieldTypeCount("std::vector<HitUtil>", false));

   EXPECT_EQ(3, inspector->GetFieldTypeCount("BaseUtil"));
   EXPECT_EQ(0, inspector->GetFieldTypeCount("BaseUtil", false));

   EXPECT_EQ(3, inspector->GetFieldTypeCount("std::int32_t"));
   EXPECT_EQ(0, inspector->GetFieldTypeCount("std::int32_t", false));

   EXPECT_EQ(4, inspector->GetFieldTypeCount("float"));
   EXPECT_EQ(0, inspector->GetFieldTypeCount("float", false));
}

TEST(RNTupleInspector, ColumnTypeCount)
{
   FileRaii fileGuard("test_ntuple_inspector_column_type_frequency.root");
   {
      auto model = RNTupleModel::Create();
      auto nFldObject = model->MakeField<ComplexStructUtil>("object");

      auto writeOptions = RNTupleWriteOptions();
      writeOptions.SetCompression(505);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), writeOptions);
   }

   std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str()));
   auto ntuple = file->Get<RNTuple>("ntuple");
   auto inspector = RNTupleInspector::Create(ntuple);

   EXPECT_EQ(2, inspector->GetColumnTypeCount(ROOT::Experimental::EColumnType::kIndex32));
   EXPECT_EQ(4, inspector->GetColumnTypeCount(ROOT::Experimental::EColumnType::kSplitReal32));
   EXPECT_EQ(3, inspector->GetColumnTypeCount(ROOT::Experimental::EColumnType::kSplitInt32));
}

TEST(RNTupleInspector, ColumnInfo)
{
   FileRaii fileGuard("test_ntuple_inspector_column_info.root");
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

   std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str()));
   auto ntuple = file->Get<RNTuple>("ntuple");
   auto inspector = RNTupleInspector::Create(ntuple);

   std::uint64_t totalOnDiskSize = 0;

   for (std::size_t i = 0; i < inspector->GetDescriptor()->GetNLogicalColumns(); ++i) {
      auto colInfo = inspector->GetColumnInfo(i);
      totalOnDiskSize += colInfo.GetOnDiskSize();

      EXPECT_GT(colInfo.GetOnDiskSize(), 0);
      EXPECT_GT(colInfo.GetInMemorySize(), 0);
      EXPECT_LT(colInfo.GetOnDiskSize(), colInfo.GetInMemorySize());
   }

   EXPECT_EQ(totalOnDiskSize, inspector->GetOnDiskSize());

   EXPECT_THROW(inspector->GetColumnInfo(42), ROOT::Experimental::RException);
}

TEST(RNTupleInspector, FieldInfo)
{
   FileRaii fileGuard("test_ntuple_inspector_field_info.root");
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

   std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str()));
   auto ntuple = file->Get<RNTuple>("ntuple");
   auto inspector = RNTupleInspector::Create(ntuple);

   auto topFieldInfo = inspector->GetFieldInfo("object");

   EXPECT_GT(topFieldInfo.GetOnDiskSize(), 0);
   EXPECT_EQ(topFieldInfo.GetInMemorySize(), inspector->GetInMemorySize());
   EXPECT_LT(topFieldInfo.GetOnDiskSize(), topFieldInfo.GetInMemorySize());

   std::uint64_t subFieldOnDiskSize = 0;
   std::uint64_t subFieldInMemorySize = 0;

   for (const auto &subField : inspector->GetDescriptor()->GetFieldIterable(topFieldInfo.GetDescriptor()->GetId())) {
      auto subFieldInfo = inspector->GetFieldInfo(subField.GetId());
      subFieldOnDiskSize += subFieldInfo.GetOnDiskSize();
      subFieldInMemorySize += subFieldInfo.GetInMemorySize();
   }

   EXPECT_EQ(topFieldInfo.GetOnDiskSize(), subFieldOnDiskSize);
   EXPECT_EQ(topFieldInfo.GetInMemorySize(), subFieldInMemorySize);

   EXPECT_THROW(inspector->GetFieldInfo("invalid_field"), ROOT::Experimental::RException);
}
