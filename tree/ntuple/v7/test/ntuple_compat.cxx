#include "ntuple_test.hxx"
#include "TKey.h"
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

class RKeyBlob : public TKey {
public:
   RKeyBlob(TFile *file, std::size_t keyLen) : TKey(file)
   {
      fClassName = RNTuple::Class_Name();
      fKeylen = keyLen;
   }

   /// Register a new key for a data record of size nbytes
   void Reserve(size_t nbytes, std::uint64_t *seekKey)
   {
      Create(nbytes);
      *seekKey = fSeekKey;
   }
};

TEST(RNTupleCompat, FwdCompat)
{
#ifndef R__BYTESWAP
   constexpr static const char kKeyHeader[27] = "\x82\x00\x00\x00"  // nbytes
                                                "\x04\x00"          // version
                                                "\x46\x00\x00\x00"  // objlen
                                                "\x61\x03\x6D\x75"  // datetime
                                                "\x3c\x00"          // keylen
                                                "\x01\x00"          // cycle
                                                "\x12\x01\x00\x00"  // seek key
                                                "\x64\x00\x00\x00"; // seek Pdir
#else
   constexpr static const char kKeyHeader[27] = "\x00\x00\x00\x82"  // nbytes
                                                "\x00\x04"          // version
                                                "\x00\x00\x00\x46"  // objlen
                                                "\x75\x6D\x03\x61"  // datetime
                                                "\x00\x3c"          // keylen
                                                "\x00\x01"          // cycle
                                                "\x00\x00\x01\x12"  // seek key
                                                "\x00\x00\x00\x64"; // seek Pdir
#endif

   constexpr static std::size_t kKeyHeaderSize = sizeof(kKeyHeader) - 1; // exclude trailing zero

   // A valid RNTuple anchor on-disk representation (anchor version 0.2.0.0, version class 5)
#ifndef R__BYTESWAP
   constexpr static const char kAnchorBin[71] = "\x3a\x00\x00\x00"                  // byte count
                                                "\x05\x00"                          // version class
                                                "\x00\x00\x02\x00\x00\x00\x00\x00"  // version epoch|major|minor|patch
                                                "\x1A\x01\x00\x00\x00\x00\x00\x00"  // seek header
                                                "\x89\x01\x00\x00\x00\x00\x00\x00"  // nbytes header
                                                "\xBE\x05\x00\x00\x00\x00\x00\x00"  // len header
                                                "\x31\x04\x00\x00\x00\x00\x00\x00"  // seek footer
                                                "\x52\x00\x00\x00\x00\x00\x00\x00"  // nbytes footer
                                                "\xAC\x00\x00\x00\x00\x00\x00\x00"  // len footer
                                                "\x8D\x6A\x65\x49\x28\xA5\x5C\xAA"; // checksum
#else
   constexpr static const char kAnchorBin[71] = "\x00\x00\x00\x3a"                  // byte count
                                                "\x00\x05"                          // version class
                                                "\x00\x00\x00\x02\x00\x00\x00\x00"  // version epoch|major|minor|patch
                                                "\x00\x00\x00\x00\x00\x00\x01\x1A"  // seek header
                                                "\x00\x00\x00\x00\x00\x00\x01\x89"  // nbytes header
                                                "\x00\x00\x00\x00\x00\x00\x05\xBE"  // len header
                                                "\x00\x00\x00\x00\x00\x00\x04\x31"  // seek footer
                                                "\x00\x00\x00\x00\x00\x00\x00\x52"  // nbytes footer
                                                "\x00\x00\x00\x00\x00\x00\x00\xAC"  // len footer
                                                "\x5B\x84\x10\x4B\x0F\xA4\x3D\x32"; // checksum
#endif

   constexpr static std::size_t kAnchorSize = sizeof(kAnchorBin) - 1; // exclude trailing zero
   constexpr static const char *kNtupleObjName = "ntpl";

   const std::size_t kNtupleClassNameLen = strlen(RNTuple::Class_Name());
   const std::size_t kNtupleObjNameLen = strlen(kNtupleObjName);

   // clang-format off
   const std::size_t kKeyLen = kKeyHeaderSize + kNtupleClassNameLen + kNtupleObjNameLen 
         + 0 // obj title is empty
         + 3 // +1 byte per serialized string (containing the str length)
         ;
   // clang-format on

   EXPECT_EQ(kKeyLen, 0x3c);

   {
      FileRaii fileGuard("test_ntuple_compat_fwd_compat_good.root");

      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      RKeyBlob key{file.get(), kKeyLen};
      std::uint64_t offset;
      key.Reserve(kAnchorSize, &offset);
      key.SetName(kNtupleObjName);
      file->AppendKey(&key);

      auto Write = [&offset, file = file.get()](const void *payload, std::size_t size) {
         file->Seek(offset);
         file->WriteBuffer(reinterpret_cast<const char *>(payload), size);
         offset += size;
      };

      // Write key header
      Write(kKeyHeader, kKeyHeaderSize);
      assert(kNtupleClassNameLen < 255);
      unsigned char strLen = static_cast<unsigned char>(kNtupleClassNameLen);
      Write(&strLen, 1);
      Write(RNTuple::Class_Name(), kNtupleClassNameLen);
      strLen = 4;
      Write(&strLen, 1);
      Write(kNtupleObjName, kNtupleObjNameLen);
      strLen = 0;
      Write(&strLen, 1);
      // Write anchor
      Write(kAnchorBin, kAnchorSize);
      file->Close();

      auto rawFile = RRawFile::Create(fileGuard.GetPath());
      auto reader = RMiniFileReader{rawFile.get()};
      auto ntuple = reader.GetNTuple(kNtupleObjName).Unwrap();
      EXPECT_EQ(ntuple.GetVersionEpoch(), 0);
      EXPECT_EQ(ntuple.GetVersionMajor(), 2);
      EXPECT_EQ(ntuple.GetVersionMinor(), 0);
      EXPECT_EQ(ntuple.GetVersionPatch(), 0);
      EXPECT_EQ(ntuple.GetSeekHeader(), 282);
      EXPECT_EQ(ntuple.GetNBytesHeader(), 393);
      EXPECT_EQ(ntuple.GetLenHeader(), 1470);
      EXPECT_EQ(ntuple.GetSeekFooter(), 1073);
      EXPECT_EQ(ntuple.GetNBytesFooter(), 82);
      EXPECT_EQ(ntuple.GetLenFooter(), 172);
   }

   // Now simulate a corrupted anchor by chopping off some bytes
   // {
   //    constexpr static std::size_t kCorruptedAnchorSize = kAnchorSize - 10;

   //    FileRaii fileGuard("test_ntuple_compat_fwd_compat_trnc.root");

   //    auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   //    // file->WriteObjectAny(kAnchorBin, RNTuple::Class(), "ntpl", "", kCorruptedAnchorSize);
   //    // file->Close();

   //    auto rawFile = RRawFile::Create(fileGuard.GetPath());
   //    auto reader = RMiniFileReader{rawFile.get()};
   //    auto ntuple = reader.GetNTuple("ntpl");
   //    EXPECT_FALSE(static_cast<bool>(ntuple));
   // }
}
