#include "Compression.h"
#include "Rtypes.h"
#include "ntuple_test.hxx"
#include "TKey.h"
#include "ROOT/EExecutionPolicy.hxx"
#include "RXTuple.hxx"
#include <gtest/gtest.h>

TEST(RNTupleCompat, Epoch)
{
   FileRaii fileGuard("test_ntuple_compat_epoch.root");

   RNTuple ntpl;
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
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("unsupported RNTuple epoch version"));
   }
}

TEST(RNTupleCompat, FeatureFlag)
{
   FileRaii fileGuard("test_ntuple_compat_feature_flag.root");

   RNTupleDescriptorBuilder descBuilder;
   descBuilder.SetNTuple("ntpl", "");
   descBuilder.SetFeature(RNTupleDescriptor::kFeatureFlagTest);
   descBuilder.AddField(
      RFieldDescriptorBuilder::FromField(ROOT::Experimental::RFieldZero()).FieldId(0).MakeDescriptor().Unwrap());
   ASSERT_TRUE(static_cast<bool>(descBuilder.EnsureValidDescriptor()));

   auto writer = RNTupleFileWriter::Recreate("ntpl", fileGuard.GetPath(), 0, EContainerFormat::kTFile,
                                             RNTupleWriteOptions::kDefaultMaxKeySize);
   RNTupleSerializer serializer;

   auto ctx = serializer.SerializeHeader(nullptr, descBuilder.GetDescriptor());
   auto buffer = std::make_unique<unsigned char[]>(ctx.GetHeaderSize());
   ctx = serializer.SerializeHeader(buffer.get(), descBuilder.GetDescriptor());
   writer->WriteNTupleHeader(buffer.get(), ctx.GetHeaderSize(), ctx.GetHeaderSize());

   auto szFooter = serializer.SerializeFooter(nullptr, descBuilder.GetDescriptor(), ctx);
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
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("unsupported format feature: 137"));
   }
}

TEST(RNTupleCompat, FwdCompat_FutureNTuple)
{
   using ROOT::Experimental::RXTuple;

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
      auto *ntuple = tfile->Get<RNTuple>(kNtupleObjName);
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
