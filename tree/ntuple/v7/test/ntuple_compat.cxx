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

   auto writer =
      RNTupleFileWriter::Recreate("ntpl", fileGuard.GetPath(), 0, RNTupleFileWriter::EContainerFormat::kTFile);
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

namespace fwd_compat {

class RHandcraftedKeyRNTuple : public TKey {
public:
   RHandcraftedKeyRNTuple() = default;

   RHandcraftedKeyRNTuple(TFile *file, std::size_t keyLen, std::size_t objLen) : TKey(file)
   {
      fClassName = RNTuple::Class_Name();
      fKeylen = keyLen;
      fObjlen = objLen;
   }

   /// Register a new key for a data record of size nbytes
   void Reserve(size_t nbytes, std::uint64_t *seekKey)
   {
      Create(nbytes);
      *seekKey = fSeekKey;
   }

   ClassDefInlineOverride(RHandcraftedKeyRNTuple, 0)
};

constexpr static const char *kNtupleObjName = "ntpl";

#define EXPECT_CORRECT_NTUPLE(ntuple, epoch, major, minor, patch) \
   EXPECT_EQ((ntuple).GetVersionEpoch(), epoch);                  \
   EXPECT_EQ((ntuple).GetVersionMajor(), major);                  \
   EXPECT_EQ((ntuple).GetVersionMinor(), minor);                  \
   EXPECT_EQ((ntuple).GetVersionPatch(), patch);                  \
   EXPECT_EQ((ntuple).GetSeekHeader(), 282);                      \
   EXPECT_EQ((ntuple).GetNBytesHeader(), 393);                    \
   EXPECT_EQ((ntuple).GetLenHeader(), 1470);                      \
   EXPECT_EQ((ntuple).GetSeekFooter(), 1073);                     \
   EXPECT_EQ((ntuple).GetNBytesFooter(), 82);                     \
   EXPECT_EQ((ntuple).GetLenFooter(), 172)

} // namespace fwd_compat

using ROOT::Experimental::RXTuple;

TEST(RNTupleCompat, FwdCompat_ValidNTuple)
{
   using namespace fwd_compat;

   // Write a valid hand-crafted RNTuple to disk and verify we can read it.
   FileRaii fileGuard("test_ntuple_compat_fwd_compat_good.root");
   fileGuard.PreserveFile();

   {
      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));

      auto xtuple = RXTuple{};
      auto key = TKey(&xtuple, RXTuple::Class(), kNtupleObjName, sizeof(RXTuple), file.get());
      key.WriteFile();
      file->Close();
   }
   
   // Patch all instances of 'RXTuple' -> 'RNTuple'.
   // We do this by just scanning the whole file and replacing all occurrences.
   // This is not the optimal way to go about it, but since the file is small (~1KB)
   // it is fast enough to not matter.
   {
      FILE *f = fopen(fileGuard.GetPath().c_str(), "r+b");

      fseek(f, 0, SEEK_END);
      std::size_t fsize = ftell(f);

      char *filebuf = new char[fsize];
      fseek(f, 0, SEEK_SET);
      fread(filebuf, fsize, 1, f);

      std::string_view file_view { filebuf, fsize };
      std::size_t pos = 0;
      while ((pos = file_view.find("XTuple"), pos) != std::string_view::npos) {
         filebuf[pos] = 'N';
         pos += 6; // skip 'XTuple'
      }
      
      fseek(f, 0, SEEK_SET);
      fwrite(filebuf, fsize, 1, f);
      
      fclose(f);
      delete [] filebuf;
   }

   {
      auto tfile = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "READ"));
      assert(!tfile->IsZombie());
      auto *ntuple = tfile->Get<RNTuple>(kNtupleObjName);
      EXPECT_CORRECT_NTUPLE(*ntuple, 9, 9, 9, 9);
   }
}

