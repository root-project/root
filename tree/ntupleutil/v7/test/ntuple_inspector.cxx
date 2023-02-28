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
   auto inspector = RNTupleInspector::Create(ntuple).Unwrap();
   EXPECT_EQ("ntuple", inspector->GetName());

   auto nullNTuple = file->Get<RNTuple>("null");
   EXPECT_THROW(RNTupleInspector::Create(nullNTuple), ROOT::Experimental::RException);
}

TEST(RNTupleInspector, CreateFromString)
{
   FileRaii fileGuard("test_ntuple_inspector_create_from_string.root");
   {
      RNTupleWriter::Recreate(RNTupleModel::Create(), "ntuple", fileGuard.GetPath());
   }

   auto inspector = RNTupleInspector::Create("ntuple", fileGuard.GetPath()).Unwrap();
   EXPECT_EQ("ntuple", inspector->GetName());

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

   EXPECT_EQ(inspector->GetNEntries(), 50);
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

   int nIndexCols = inspector->GetColumnTypeFrequency(ROOT::Experimental::EColumnType::kIndex32);
   int nEntries = inspector->GetNEntries();

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

TEST(RNTupleInspector, FieldTypeFrequency)
{
   FileRaii fileGuard("test_ntuple_inspector_field_type_frequency.root");
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

   std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str()));
   auto ntuple = file->Get<RNTuple>("ntuple");
   auto inspector = RNTupleInspector::Create(ntuple);

   EXPECT_EQ(3, inspector->GetFieldTypeFrequency("std::int32_t"));
   EXPECT_EQ(2, inspector->GetFieldTypeFrequency("std::string"));
   EXPECT_EQ(1, inspector->GetFieldTypeFrequency("ComplexStructUtil"));
   EXPECT_EQ(0, inspector->GetFieldTypeFrequency("std::vector<float>"));
}

TEST(RNTupleInspector, ColumnTypeFrequency)
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

   EXPECT_EQ(2, inspector->GetColumnTypeFrequency(ROOT::Experimental::EColumnType::kIndex32));
   EXPECT_EQ(4, inspector->GetColumnTypeFrequency(ROOT::Experimental::EColumnType::kSplitReal32));
   EXPECT_EQ(3, inspector->GetColumnTypeFrequency(ROOT::Experimental::EColumnType::kSplitInt32));
}

TEST(RNTupleInspector, ColumnSize)
{
   FileRaii fileGuard("test_ntuple_inspector_column_size.root");
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
   auto reader = RNTupleReader::Open(ntuple);

   for (std::size_t i = 0; i < reader->GetDescriptor()->GetNLogicalColumns(); ++i) {
      EXPECT_GT(inspector->GetOnDiskColumnSize(i), 0);
      EXPECT_GT(inspector->GetInMemoryColumnSize(i), 0);
      EXPECT_LT(inspector->GetOnDiskColumnSize(i), inspector->GetInMemoryColumnSize(i));
   }

   EXPECT_EQ(inspector->GetOnDiskColumnSize(-42), -1);
   EXPECT_EQ(inspector->GetInMemoryColumnSize(-42), -1);
}

TEST(RNTupleInspector, FieldSize)
{
   FileRaii fileGuard("test_ntuple_inspector_field_size.root");
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

   EXPECT_GT(inspector->GetOnDiskFieldSize("object"), 0);
   EXPECT_EQ(inspector->GetInMemoryFieldSize("object"), inspector->GetInMemorySize());
   EXPECT_LT(inspector->GetOnDiskFieldSize("object"), inspector->GetInMemoryFieldSize("object"));

   EXPECT_EQ(inspector->GetOnDiskFieldSize("invalid_field"), -1);
   EXPECT_EQ(inspector->GetInMemoryFieldSize("invalid_field"), -1);
}

TEST(RNTupleInspector, ColumnType)
{
   FileRaii fileGuard("test_ntuple_inspector_field_size.root");
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

   EXPECT_EQ(inspector->GetColumnType(0), ROOT::Experimental::EColumnType::kSplitReal32);
   EXPECT_EQ(inspector->GetColumnType(1), ROOT::Experimental::EColumnType::kIndex32);
   EXPECT_EQ(inspector->GetColumnType(2), ROOT::Experimental::EColumnType::kSplitInt32);
}
