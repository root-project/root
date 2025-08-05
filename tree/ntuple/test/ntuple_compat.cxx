#include "Compression.h"
#include "Rtypes.h"
#include "ntuple_test.hxx"
#include "TKey.h"
#include "ROOT/EExecutionPolicy.hxx"
#include "RXTuple.hxx"
#include <gtest/gtest.h>
#include <memory>
#include <cstdio>

TEST(RNTupleCompat, Epoch)
{
   FileRaii fileGuard("test_ntuple_compat_epoch.root");

   ROOT::RNTuple ntpl;
   // The first 16 bit integer in the struct is the epoch
   std::uint16_t *versionEpoch = reinterpret_cast<uint16_t *>(&ntpl);
   *versionEpoch = *versionEpoch + 1;
   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   file->WriteObject(&ntpl, "ntpl");
   file->Close();

   auto pageSource = RPageSource::Create("ntpl", fileGuard.GetPath());
   try {
      pageSource->Attach();
      FAIL() << "opening an RNTuple with different epoch version should fail";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("unsupported RNTuple epoch version"));
   }
}

TEST(RNTupleCompat, FeatureFlag)
{
   FileRaii fileGuard("test_ntuple_compat_feature_flag.root");

   RNTupleDescriptorBuilder descBuilder;
   descBuilder.SetNTuple("ntpl", "");
   descBuilder.SetFeature(RNTupleDescriptor::kFeatureFlagTest);
   descBuilder.AddField(RFieldDescriptorBuilder::FromField(ROOT::RFieldZero()).FieldId(0).MakeDescriptor().Unwrap());
   ASSERT_TRUE(static_cast<bool>(descBuilder.EnsureValidDescriptor()));

   RNTupleWriteOptions options;
   auto writer = RNTupleFileWriter::Recreate("ntpl", fileGuard.GetPath(), EContainerFormat::kTFile, options);
   RNTupleSerializer serializer;

   auto ctx = serializer.SerializeHeader(nullptr, descBuilder.GetDescriptor()).Unwrap();
   auto buffer = std::make_unique<unsigned char[]>(ctx.GetHeaderSize());
   ctx = serializer.SerializeHeader(buffer.get(), descBuilder.GetDescriptor()).Unwrap();
   writer->WriteNTupleHeader(buffer.get(), ctx.GetHeaderSize(), ctx.GetHeaderSize());

   auto szFooter = serializer.SerializeFooter(nullptr, descBuilder.GetDescriptor(), ctx).Unwrap();
   buffer = std::make_unique<unsigned char[]>(szFooter);
   serializer.SerializeFooter(buffer.get(), descBuilder.GetDescriptor(), ctx);
   writer->WriteNTupleFooter(buffer.get(), szFooter, szFooter);

   writer->Commit();
   // Call destructor to flush data to disk
   writer = nullptr;

   auto pageSource = RPageSource::Create("ntpl", fileGuard.GetPath());
   try {
      pageSource->Attach();
      FAIL() << "opening an RNTuple that uses an unsupported feature should fail";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("unsupported format feature: 137"));
   }
}

TEST(RNTupleCompat, FwdCompat_FutureNTupleAnchor)
{
   using ROOT::RXTuple;

   constexpr static const char *kNtupleObjName = "ntpl";

   FileRaii fileGuard("test_ntuple_compat_fwd_compat_future_ntuple.root");

   // Write an RXTuple to disk. It is a simulacrum of a future version of RNTuple, with additional fields and a higher
   // class version.
   {
      auto file = std::unique_ptr<TFile>(
         TFile::Open(fileGuard.GetPath().c_str(), "RECREATE", "", ROOT::RCompressionSetting::ELevel::kUncompressed));
      auto xtuple = RXTuple{};
      file->WriteObject(&xtuple, kNtupleObjName);

      // The file is supposed to be small enough to allow for quick scanning by the patching done later.
      // Let's put 4KB as a safe limit.
      EXPECT_LE(file->GetEND(), 4096);
   }

   // Patch all instances of 'RXTuple' -> 'RNTuple'.
   // We do this by just scanning the whole file and replacing all occurrences.
   // This is not the optimal way to go about it, but since the file is small (~1KB)
   // it is fast enough to not matter.
   {
      FILE *f = fopen(fileGuard.GetPath().c_str(), "r+b");

      fseek(f, 0, SEEK_END);
      size_t fsize = ftell(f);

      char *filebuf = new char[fsize];
      fseek(f, 0, SEEK_SET);
      size_t itemsRead = fread(filebuf, fsize, 1, f);
      EXPECT_EQ(itemsRead, 1);

      std::string_view file_view{filebuf, fsize};
      size_t pos = 0;
      while ((pos = file_view.find("XTuple"), pos) != std::string_view::npos) {
         filebuf[pos] = 'N';
         pos += 6; // skip "XTuple"
      }

      fseek(f, 0, SEEK_SET);
      size_t itemsWritten = fwrite(filebuf, fsize, 1, f);
      EXPECT_EQ(itemsWritten, 1);

      fclose(f);
      delete[] filebuf;
   }

   // Read back the RNTuple from the future with TFile
   {
      auto tfile = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "READ"));
      assert(!tfile->IsZombie());
      auto ntuple = std::unique_ptr<ROOT::RNTuple>(tfile->Get<ROOT::RNTuple>(kNtupleObjName));
      EXPECT_EQ(ntuple->GetVersionEpoch(), RXTuple{}.fVersionEpoch);
      EXPECT_EQ(ntuple->GetVersionMajor(), RXTuple{}.fVersionMajor);
      EXPECT_EQ(ntuple->GetVersionMinor(), RXTuple{}.fVersionMinor);
      EXPECT_EQ(ntuple->GetVersionPatch(), RXTuple{}.fVersionPatch);
      EXPECT_EQ(ntuple->GetSeekHeader(), RXTuple{}.fSeekHeader);
      EXPECT_EQ(ntuple->GetNBytesHeader(), RXTuple{}.fNBytesHeader);
      EXPECT_EQ(ntuple->GetLenHeader(), RXTuple{}.fLenHeader);
      EXPECT_EQ(ntuple->GetSeekFooter(), RXTuple{}.fSeekFooter);
      EXPECT_EQ(ntuple->GetNBytesFooter(), RXTuple{}.fNBytesFooter);
      EXPECT_EQ(ntuple->GetLenFooter(), RXTuple{}.fLenFooter);
   }

   // Then read it back with RMiniFile
   {
      auto rawFile = RRawFile::Create(fileGuard.GetPath());
      auto reader = RMiniFileReader{rawFile.get()};
      auto ntuple = reader.GetNTuple(kNtupleObjName).Unwrap();
      EXPECT_EQ(ntuple.GetVersionEpoch(), RXTuple{}.fVersionEpoch);
      EXPECT_EQ(ntuple.GetVersionMajor(), RXTuple{}.fVersionMajor);
      EXPECT_EQ(ntuple.GetVersionMinor(), RXTuple{}.fVersionMinor);
      EXPECT_EQ(ntuple.GetVersionPatch(), RXTuple{}.fVersionPatch);
      EXPECT_EQ(ntuple.GetSeekHeader(), RXTuple{}.fSeekHeader);
      EXPECT_EQ(ntuple.GetNBytesHeader(), RXTuple{}.fNBytesHeader);
      EXPECT_EQ(ntuple.GetLenHeader(), RXTuple{}.fLenHeader);
      EXPECT_EQ(ntuple.GetSeekFooter(), RXTuple{}.fSeekFooter);
      EXPECT_EQ(ntuple.GetNBytesFooter(), RXTuple{}.fNBytesFooter);
      EXPECT_EQ(ntuple.GetLenFooter(), RXTuple{}.fLenFooter);
   }
}

template <>
class ROOT::RField<ROOT::Internal::RTestFutureColumn> final : public RSimpleField<ROOT::Internal::RTestFutureColumn> {
protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final
   {
      return std::make_unique<RField>(newName);
   }
   const RColumnRepresentations &GetColumnRepresentations() const final
   {
      static const RColumnRepresentations representations{{{ROOT::Internal::kTestFutureColumnType}}, {}};
      return representations;
   }

public:
   static std::string TypeName() { return "ROOT::Internal::RTestFutureColumn"; }
   explicit RField(std::string_view name) : RSimpleField(name, TypeName()) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() final = default;
};

TEST(RNTupleCompat, FutureColumnType)
{
   // Write a RNTuple containing a field with an unknown column type and verify we can
   // read back the ntuple and its descriptor.

   FileRaii fileGuard("test_ntuple_compat_future_col_type.root");
   {
      auto model = RNTupleModel::Create();
      auto col = model->MakeField<ROOT::Internal::RTestFutureColumn>("futureColumn");
      auto colValid = model->MakeField<float>("float");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      col->dummy = 0x42424242;
      *colValid = 69.f;
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   const auto &desc = reader->GetDescriptor();
   const auto &fdesc = desc.GetFieldDescriptor(desc.FindFieldId("futureColumn"));
   GTEST_ASSERT_EQ(fdesc.GetLogicalColumnIds().size(), 1);
   const auto &cdesc = desc.GetColumnDescriptor(fdesc.GetLogicalColumnIds()[0]);
   EXPECT_EQ(cdesc.GetType(), ROOT::ENTupleColumnType::kUnknown);

   try {
      desc.CreateModel();
      FAIL() << "Creating a model not in fwd-compatible mode should fail";
   } catch (const ROOT::RException &ex) {
      EXPECT_THAT(ex.what(), testing::HasSubstr("unknown column types"));
   }

   {
      auto modelOpts = RNTupleDescriptor::RCreateModelOptions();
      modelOpts.SetForwardCompatible(true);
      auto model = desc.CreateModel(modelOpts);

      try {
         model->GetConstField("futureColumn");
         FAIL() << "the future column should not show up in the model";
      } catch (const ROOT::RException &ex) {
         EXPECT_THAT(ex.what(), testing::HasSubstr("invalid field"));
      }

      const auto &floatFld = model->GetConstField("float");
      EXPECT_EQ(floatFld.GetTypeName(), "float");

      reader.reset();
      reader = RNTupleReader::Open(std::move(model), "ntpl", fileGuard.GetPath());

      auto floatId = reader->GetDescriptor().FindFieldId("float");
      auto floatPtr = reader->GetView<float>(floatId);
      EXPECT_FLOAT_EQ(floatPtr(0), 69.f);
   }
}

TEST(RNTupleCompat, FutureColumnType_Nested)
{
   // Write a RNTuple containing a field with an unknown column type and verify we can
   // read back the ntuple and its descriptor.

   FileRaii fileGuard("test_ntuple_compat_future_col_type_nested.root");

   {
      auto model = RNTupleModel::Create();
      std::vector<std::unique_ptr<RFieldBase>> itemFields;
      itemFields.emplace_back(new RField<std::vector<ROOT::Internal::RTestFutureColumn>>("vec"));
      auto field = std::make_unique<ROOT::RRecordField>("future", std::move(itemFields));
      model->AddField(std::move(field));
      auto floatP = model->MakeField<float>("float");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      *floatP = 33.f;
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   const auto &desc = reader->GetDescriptor();
   const auto futureId = desc.FindFieldId("future");
   const auto &fdesc = desc.GetFieldDescriptor(desc.FindFieldId("vec._0", futureId));
   ASSERT_EQ(fdesc.GetLogicalColumnIds().size(), 1);
   const auto &cdesc = desc.GetColumnDescriptor(fdesc.GetLogicalColumnIds()[0]);
   EXPECT_EQ(cdesc.GetType(), ROOT::ENTupleColumnType::kUnknown);

   try {
      desc.CreateModel();
      FAIL() << "Creating a model not in fwd-compatible mode should fail";
   } catch (const ROOT::RException &ex) {
      EXPECT_THAT(ex.what(), testing::HasSubstr("unknown column types"));
   }

   {
      auto modelOpts = RNTupleDescriptor::RCreateModelOptions();
      modelOpts.SetForwardCompatible(true);
      auto model = desc.CreateModel(modelOpts);

      try {
         model->GetConstField("futureColumn");
         FAIL() << "the future column should not show up in the model";
      } catch (const ROOT::RException &ex) {
         EXPECT_THAT(ex.what(), testing::HasSubstr("invalid field"));
      }

      const auto &floatFld = model->GetConstField("float");
      EXPECT_EQ(floatFld.GetTypeName(), "float");

      reader.reset();
      reader = RNTupleReader::Open(std::move(model), "ntpl", fileGuard.GetPath());

      auto floatId = reader->GetDescriptor().FindFieldId("float");
      auto floatPtr = reader->GetView<float>(floatId);
      EXPECT_FLOAT_EQ(floatPtr(0), 33.f);
   }
}

class RFutureField : public RFieldBase {
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final
   {
      return std::make_unique<RFutureField>(newName);
   };
   void ConstructValue(void *) const final {}

   std::size_t AppendImpl(const void *) final { return 0; }

public:
   RFutureField(std::string_view name) : RFieldBase(name, "Future", ROOT::Internal::kTestFutureFieldStructure, false) {}

   std::size_t GetValueSize() const final { return 0; }
   std::size_t GetAlignment() const final { return 0; }
};

TEST(RNTupleCompat, FutureFieldStructuralRole)
{
   // Write a RNTuple containing a field with an unknown structural role and verify we can
   // read back the ntuple, its descriptor and reconstruct the model.

   FileRaii fileGuard("test_ntuple_compat_future_field_struct.root");
   {
      auto model = RNTupleModel::Create();
      auto field = std::make_unique<RFutureField>("future");
      model->AddField(std::move(field));
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   const auto &desc = reader->GetDescriptor();
   const auto &fdesc = desc.GetFieldDescriptor(desc.FindFieldId("future"));
   EXPECT_EQ(fdesc.GetLogicalColumnIds().size(), 0);

   try {
      desc.CreateModel();
      FAIL() << "Attempting to create a model with default options should fail";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("unexpected on-disk field structure"));
   }

   auto modelOpts = RNTupleDescriptor::RCreateModelOptions();
   modelOpts.SetForwardCompatible(true);
   auto model = desc.CreateModel(modelOpts);
   try {
      model->GetConstField("future");
      FAIL() << "trying to get a field with unknown role should fail";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("invalid field"));
   }
}

TEST(RNTupleCompat, FutureFieldStructuralRole_Nested)
{
   // Write a RNTuple containing a field with an unknown structural role and verify we can
   // read back the ntuple, its descriptor and reconstruct the model.

   FileRaii fileGuard("test_ntuple_compat_future_field_struct_nested.root");
   {
      auto model = RNTupleModel::Create();
      std::vector<std::unique_ptr<RFieldBase>> itemFields;
      itemFields.emplace_back(new RField<int>("int"));
      itemFields.emplace_back(new RFutureField("future"));
      auto field = std::make_unique<ROOT::RRecordField>("record", std::move(itemFields));
      model->AddField(std::move(field));
      model->MakeField<float>("float");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   const auto &desc = reader->GetDescriptor();
   const auto &fdesc = desc.GetFieldDescriptor(desc.FindFieldId("record"));
   EXPECT_EQ(fdesc.GetLogicalColumnIds().size(), 0);

   try {
      desc.CreateModel();
      FAIL() << "attempting to create a model with default options should fail";
   } catch (const ROOT::RException &ex) {
      EXPECT_THAT(ex.what(), testing::HasSubstr("unexpected on-disk field structure"));
   }

   auto modelOpts = RNTupleDescriptor::RCreateModelOptions();
   modelOpts.SetForwardCompatible(true);
   auto model = desc.CreateModel(modelOpts);
   const auto &floatFld = model->GetConstField("float");
   EXPECT_EQ(floatFld.GetTypeName(), "float");
   try {
      model->GetConstField("record");
      FAIL() << "trying to get a field with unknown role should fail";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("invalid field"));
   }
}

class RPageSinkTestLocator : public RPageSinkFile {
   ROOT::RNTupleLocator WriteSealedPage(const RPageStorage::RSealedPage &sealedPage, std::size_t)
   {
      auto payload = ROOT::RNTupleLocatorObject64{0x420};
      RNTupleLocator result;
      result.SetPosition(payload);
      result.SetType(ROOT::Internal::kTestLocatorType);
      result.SetNBytesOnStorage(sealedPage.GetDataSize());
      return result;
   }

   RNTupleLocator CommitPageImpl(ColumnHandle_t columnHandle, const RPage &page) override
   {
      auto element = columnHandle.fColumn->GetElement();
      RPageStorage::RSealedPage sealedPage = SealPage(page, *element);
      return WriteSealedPage(sealedPage, element->GetPackedSize(page.GetNElements()));
   }

public:
   using RPageSinkFile::RPageSinkFile;
};

TEST(RNTupleCompat, UnknownLocatorType)
{
   // Write a RNTuple containing a page with an unknown locator type and verify we can
   // read back the ntuple, its descriptor and reconstruct the model (but not read pages)

   FileRaii fileGuard("test_ntuple_compat_future_locator.root");

   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<float>("pt");
      auto wopts = RNTupleWriteOptions();
      auto sink = std::make_unique<RPageSinkTestLocator>("ntpl", fileGuard.GetPath(), wopts);
      auto writer = ROOT::Internal::CreateRNTupleWriter(std::move(model), std::move(sink));
      *fieldPt = 33.f;
      writer->Fill();
   }

   auto readOpts = RNTupleReadOptions();
   // disable the cluster cache so we can catch the exception that happens on LoadEntry
   readOpts.SetClusterCache(RNTupleReadOptions::EClusterCache::kOff);
   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath(), readOpts);
   const auto &desc = reader->GetDescriptor();
   const auto &fdesc = desc.GetFieldDescriptor(desc.FindFieldId("pt"));
   EXPECT_EQ(fdesc.GetLogicalColumnIds().size(), 1);

   // Creating a model should succeed
   auto model = desc.CreateModel();
   (void)model;

   try {
      reader->LoadEntry(0);
      FAIL() << "trying to read a field with an unknown locator should fail";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("tried to read a page with an unknown locator"));
   }
}
