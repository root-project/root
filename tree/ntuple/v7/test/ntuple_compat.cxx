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

   if (false)
   {
      auto tfile = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "READ"));
      assert(!tfile->IsZombie());
      auto *xtuple = tfile->Get<RXTuple>(kNtupleObjName);
      EXPECT_EQ(xtuple->fVersionEpoch, 9);
   }
}

#if 0
// The header for the TKey that will contain the handcrafted RNTuple
constexpr static char kKeyHeader[27] = "\x00\x00\x00\x82"             // nbytes
                                       "\x00\x04"                     // version
                                       "\x00\x00\x00\x46"             // objlen
                                       "\x75\x6D\x03\x61"             // datetime
                                       "\x00\x3c"                     // keylen
                                       "\x00\x01"                     // cycle
                                       "\x00\x00\x01\x12"             // seek key
                                       "\x00\x00\x00\x64";            // seek Pdir
constexpr static std::size_t kKeyHeaderSize = sizeof(kKeyHeader) - 1; // exclude trailing zero

// The header for the TKey that will contain the handcrafted RNTuple from the future
constexpr static char kFutureKeyHeader[27] = "\x00\x00\x00\xA2"  // nbytes
                                             "\x00\x04"          // version
                                             "\x00\x00\x00\x66"  // objlen
                                             "\x75\x6D\x03\x61"  // datetime
                                             "\x00\x3c"          // keylen
                                             "\x00\x01"          // cycle
                                             "\x00\x00\x01\x12"  // seek key
                                             "\x00\x00\x00\x64"; // seek Pdir
static_assert(sizeof(kKeyHeader) == sizeof(kFutureKeyHeader));

// A valid RNTuple anchor on-disk representation (anchor version 0.2.0.0, version class 5)
constexpr static char kAnchorBin[71] = "\x40\x00\x00\x3a"                  // byte count
                                       "\x00\x05"                          // version class
                                       "\x00\x00\x00\x02\x00\x00\x00\x00"  // version epoch|major|minor|patch
                                       "\x00\x00\x00\x00\x00\x00\x01\x1A"  // seek header
                                       "\x00\x00\x00\x00\x00\x00\x01\x89"  // nbytes header
                                       "\x00\x00\x00\x00\x00\x00\x05\xBE"  // len header
                                       "\x00\x00\x00\x00\x00\x00\x04\x31"  // seek footer
                                       "\x00\x00\x00\x00\x00\x00\x00\x52"  // nbytes footer
                                       "\x00\x00\x00\x00\x00\x00\x00\xAC"  // len footer
                                       "\x5B\x84\x10\x4B\x0F\xA4\x3D\x32"; // checksum
constexpr static std::size_t kAnchorSize = sizeof(kAnchorBin) - 1;         // exclude trailing zero

// A valid RNTuple from the future (v 9.9.9.9)
constexpr static char kFutureAnchorBin[103] = "\x40\x00\x00\x5a"                  // byte count
                                              "\x03\xE7"                          // version class (v 999)
                                              "\x00\x09\x00\x09\x00\x09\x00\x09"  // version epoch|major|minor|patch
                                              "\x00\x00\x00\x00\x00\x00\x01\x1A"  // seek header
                                              "\x00\x00\x00\x00\x00\x00\x01\x89"  // nbytes header
                                              "\x00\x00\x00\x00\x00\x00\x05\xBE"  // len header
                                              "\x00\x00\x00\x00\x00\x00\x04\x31"  // seek footer
                                              "\x00\x00\x00\x00\x00\x00\x00\x52"  // nbytes footer
                                              "\x00\x00\x00\x00\x00\x00\x00\xAC"  // len footer
                                              "\xDE\xAD\xC0\xDE\xDE\xAD\xC0\xDE"  // hypotethical future field 0
                                              "\xDE\xAD\xC0\xDE\xDE\xAD\xC0\xDE"  // hypotethical future field 1
                                              "\xDE\xAD\xC0\xDE\xDE\xAD\xC0\xDE"  // hypotethical future field 2
                                              "\xDE\xAD\xC0\xDE\xDE\xAD\xC0\xDE"  // hypotethical future field 3
                                              "\x9B\xD3\x59\xBD\x40\x29\x50\x0D"; // checksum
constexpr static std::size_t kFutureAnchorSize = sizeof(kFutureAnchorBin) - 1;    // exclude trailing zero

constexpr static std::size_t kNtupleClassNameLen = std::char_traits<char>::length("ROOT::Experimental::RNTuple");

// clang-format off
constexpr static std::size_t kKeyLen = kKeyHeaderSize + kNtupleClassNameLen + kNtupleObjNameLen 
                                       + 0 // obj title is empty
                                       + 3 // +1 byte per serialized string (containing the str length)
                                       ;
// clang-format on

// kNtupleObjName should be "ntpl" and the title should be empty.
static_assert(kKeyLen == 0x3c);

static void Write(TFile &file, std::uint64_t &offset, const void *payload, std::size_t size)
{
   file.Seek(offset);
   file.WriteBuffer(reinterpret_cast<const char *>(payload), size);
   offset += size;
}

static std::uint64_t WriteKeyHeader(TFile *file, RHandcraftedKeyRNTuple &key, const void *keyHeader = kKeyHeader,
                                    std::size_t anchorSize = kAnchorSize)
{
   std::uint64_t offset;
   key.Reserve(anchorSize, &offset);
   key.SetName(kNtupleObjName);
   file->AppendKey(&key);

   Write(*file, offset, keyHeader, kKeyHeaderSize);
   // class name
   assert(kNtupleClassNameLen < 255);
   unsigned char strLen = static_cast<unsigned char>(kNtupleClassNameLen);
   Write(*file, offset, &strLen, 1);
   Write(*file, offset, RNTuple::Class_Name(), kNtupleClassNameLen);
   // obj name
   strLen = 4;
   Write(*file, offset, &strLen, 1);
   Write(*file, offset, kNtupleObjName, kNtupleObjNameLen);
   // obj title
   strLen = 0;
   Write(*file, offset, &strLen, 1);

   return offset;
}

} // end namespace fwd_compat

TEST(RNTupleCompat, FwdCompat_ValidNTuple)
{
   using namespace fwd_compat;

   // Write a valid hand-crafted RNTuple to disk and verify we can read it.
   FileRaii fileGuard("test_ntuple_compat_fwd_compat_good.root");
   fileGuard.PreserveFile();

   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));

   RHandcraftedKeyRNTuple key{file.get(), kKeyLen, kAnchorSize};
   auto offset = WriteKeyHeader(file.get(), key);

   // Write anchor
   Write(*file, offset, kAnchorBin, kAnchorSize);
   file->Close();

   {
      auto rawFile = RRawFile::Create(fileGuard.GetPath());
      auto reader = RMiniFileReader{rawFile.get()};
      auto ntuple = reader.GetNTuple(kNtupleObjName).Unwrap();
      EXPECT_CORRECT_NTUPLE(ntuple, 0, 2, 0, 0);
   }

   {
      auto tfile = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "READ"));
      assert(!tfile->IsZombie());
      auto *ntuple = tfile->Get<RNTuple>(kNtupleObjName);
      EXPECT_CORRECT_NTUPLE(*ntuple, 0, 2, 0, 0);
   }
}

TEST(RNTupleCompat, FwdCompat_ValidNTuple_Future)
{
   using namespace fwd_compat;

   // Write to disk a valid hand-crafted RNTuple from the future, containing some new hypothetical fields
   // that the current version doesn't know, and verify that it can be read.
   FileRaii fileGuard("test_ntuple_compat_fwd_compat_futr.root");

   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));

   RHandcraftedKeyRNTuple key{file.get(), kKeyLen, kFutureAnchorSize};
   auto offset = WriteKeyHeader(file.get(), key, kFutureKeyHeader, kFutureAnchorSize);

   // Write anchor
   Write(*file, offset, kFutureAnchorBin, kFutureAnchorSize);
   file->Close();

   {
      auto rawFile = RRawFile::Create(fileGuard.GetPath());
      auto reader = RMiniFileReader{rawFile.get()};
      auto ntuple = reader.GetNTuple(kNtupleObjName).Unwrap();
      EXPECT_CORRECT_NTUPLE(ntuple, 9, 9, 9, 9);
   }

   {
      auto tfile = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "READ"));
      auto *ntuple = tfile->Get<RNTuple>(kNtupleObjName);
      EXPECT_CORRECT_NTUPLE(*ntuple, 9, 9, 9, 9);
   }
}

TEST(RNTupleCompat, FwdCompat_Invalid_Chopped)
{
   using namespace fwd_compat;

   // Simulate a corrupted anchor by chopping off some bytes from the otherwise-valid anchor.
   constexpr static std::size_t kCorruptedAnchorSize = kAnchorSize - 10;

   FileRaii fileGuard("test_ntuple_compat_fwd_compat_trnc.root");

   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));

   RHandcraftedKeyRNTuple key{file.get(), kKeyLen, kAnchorSize};
   auto offset = WriteKeyHeader(file.get(), key);

   // Write anchor
   Write(*file, offset, kAnchorBin, kCorruptedAnchorSize);
   file->Close();

   {
      auto rawFile = RRawFile::Create(fileGuard.GetPath());
      auto reader = RMiniFileReader{rawFile.get()};
      auto ntuple = reader.GetNTuple(kNtupleObjName);
      EXPECT_FALSE(static_cast<bool>(ntuple));
   }

   {
      auto tfile = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "READ"));
      auto *ntuple = tfile->Get<RNTuple>(kNtupleObjName);
      EXPECT_EQ(ntuple, nullptr);
   }
}

TEST(RNTupleCompat, FwdCompat_Invalid_ChoppedOne)
{
   using namespace fwd_compat;

   // Simulate a corrupted anchor by chopping off a single byte from the otherwise-valid anchor
   constexpr static std::size_t kCorruptedAnchorSize = kAnchorSize - 1;

   FileRaii fileGuard("test_ntuple_compat_fwd_compat_trc2.root");

   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));

   RHandcraftedKeyRNTuple key{file.get(), kKeyLen, kAnchorSize};
   auto offset = WriteKeyHeader(file.get(), key);

   // Write anchor
   Write(*file, offset, kAnchorBin, kCorruptedAnchorSize);
   file->Close();

   {
      auto rawFile = RRawFile::Create(fileGuard.GetPath());
      auto reader = RMiniFileReader{rawFile.get()};
      auto ntuple = reader.GetNTuple(kNtupleObjName);
      EXPECT_FALSE(static_cast<bool>(ntuple));
   }

   {
      auto tfile = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "READ"));
      auto *ntuple = tfile->Get<RNTuple>(kNtupleObjName);
      EXPECT_EQ(ntuple, nullptr);
   }
}

TEST(RNTupleCompat, FwdCompat_Invalid_Flipped)
{
   using namespace fwd_compat;

   // simulate a corrupted anchor by flipping a random bit from the otherwise-valid anchor.
   char corruptedAnchor[sizeof(kAnchorBin)];
   memcpy(corruptedAnchor, kAnchorBin, sizeof(kAnchorBin));
   corruptedAnchor[kAnchorSize / 2] ^= (1u << 3);

   FileRaii fileGuard("test_ntuple_compat_fwd_compat_flip.root");

   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));

   RHandcraftedKeyRNTuple key{file.get(), kKeyLen, kAnchorSize};
   auto offset = WriteKeyHeader(file.get(), key);

   // Write anchor
   Write(*file, offset, corruptedAnchor, kAnchorSize);
   file->Close();

   {
      auto rawFile = RRawFile::Create(fileGuard.GetPath());
      auto reader = RMiniFileReader{rawFile.get()};
      auto ntuple = reader.GetNTuple(kNtupleObjName);
      EXPECT_FALSE(static_cast<bool>(ntuple));
   }

   {
      auto tfile = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "READ"));
      auto *ntuple = tfile->Get<RNTuple>(kNtupleObjName);
      EXPECT_EQ(ntuple, nullptr);
   }
}
#endif
