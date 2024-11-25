#include "ntuple_test.hxx"
#include <TKey.h>
#include <TTree.h>
#include <TVector2.h>
#include <TVector3.h>
#include <TVirtualStreamerInfo.h>

#include <cstring>

using ROOT::Experimental::Internal::RNTupleWriteOptionsManip;

namespace {
bool IsEqual(const ROOT::RNTuple &a, const ROOT::RNTuple &b)
{
   return a.GetVersionEpoch() == b.GetVersionEpoch() && a.GetVersionMajor() == b.GetVersionMajor() &&
          a.GetVersionMinor() == b.GetVersionMinor() && a.GetVersionPatch() == b.GetVersionPatch() &&
          a.GetSeekHeader() == b.GetSeekHeader() && a.GetNBytesHeader() == b.GetNBytesHeader() &&
          a.GetLenHeader() == b.GetLenHeader() && a.GetSeekFooter() == b.GetSeekFooter() &&
          a.GetNBytesFooter() == b.GetNBytesFooter() && a.GetLenFooter() == b.GetLenFooter() &&
          a.GetMaxKeySize() == b.GetMaxKeySize();
}

struct RNTupleTester {
   ROOT::RNTuple fNtpl;

   explicit RNTupleTester(const ROOT::RNTuple &ntpl) : fNtpl(ntpl) {}
   ROOT::RNTuple GetAnchor() const { return fNtpl; }
};
} // namespace

TEST(MiniFile, Raw)
{
   FileRaii fileGuard("test_ntuple_minifile_raw.ntuple");

   RNTupleWriteOptions options;
   auto writer = RNTupleFileWriter::Recreate("MyNTuple", fileGuard.GetPath(), EContainerFormat::kBare, options);
   char header = 'h';
   char footer = 'f';
   char blob = 'b';
   auto offHeader = writer->WriteNTupleHeader(&header, 1, 1);
   auto offBlob = writer->WriteBlob(&blob, 1, 1);
   auto offFooter = writer->WriteNTupleFooter(&footer, 1, 1);
   writer->Commit();

   auto rawFile = RRawFile::Create(fileGuard.GetPath());
   RMiniFileReader reader(rawFile.get());
   auto ntuple = reader.GetNTuple("MyNTuple").Inspect();
   EXPECT_EQ(offHeader, ntuple.GetSeekHeader());
   EXPECT_EQ(offFooter, ntuple.GetSeekFooter());

   char buf;
   reader.ReadBuffer(&buf, 1, offBlob);
   EXPECT_EQ(blob, buf);
   reader.ReadBuffer(&buf, 1, offHeader);
   EXPECT_EQ(header, buf);
   reader.ReadBuffer(&buf, 1, offFooter);
   EXPECT_EQ(footer, buf);
}

TEST(MiniFile, Stream)
{
   FileRaii fileGuard("test_ntuple_minifile_stream.root");

   RNTupleWriteOptions options;
   auto writer = RNTupleFileWriter::Recreate("MyNTuple", fileGuard.GetPath(), EContainerFormat::kTFile, options);
   char header = 'h';
   char footer = 'f';
   char blob = 'b';
   auto offHeader = writer->WriteNTupleHeader(&header, 1, 1);
   auto offBlob = writer->WriteBlob(&blob, 1, 1);
   auto offFooter = writer->WriteNTupleFooter(&footer, 1, 1);
   writer->Commit();

   auto rawFile = RRawFile::Create(fileGuard.GetPath());
   RMiniFileReader reader(rawFile.get());
   auto ntuple = reader.GetNTuple("MyNTuple").Inspect();
   EXPECT_EQ(offHeader, ntuple.GetSeekHeader());
   EXPECT_EQ(offFooter, ntuple.GetSeekFooter());

   char buf;
   reader.ReadBuffer(&buf, 1, offBlob);
   EXPECT_EQ(blob, buf);
   reader.ReadBuffer(&buf, 1, offHeader);
   EXPECT_EQ(header, buf);
   reader.ReadBuffer(&buf, 1, offFooter);
   EXPECT_EQ(footer, buf);

   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "READ"));
   ASSERT_TRUE(file);
   auto k = std::unique_ptr<ROOT::RNTuple>(file->Get<ROOT::RNTuple>("MyNTuple"));
   EXPECT_TRUE(IsEqual(ntuple, RNTupleTester(*k).GetAnchor()));
}

TEST(MiniFile, Proper)
{
   FileRaii fileGuard("test_ntuple_minifile_proper.root");

   std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   auto writer = RNTupleFileWriter::Append("MyNTuple", *file, RNTupleWriteOptions::kDefaultMaxKeySize);

   char header = 'h';
   char footer = 'f';
   char blob = 'b';
   auto offHeader = writer->WriteNTupleHeader(&header, 1, 1);
   auto offBlob = writer->WriteBlob(&blob, 1, 1);
   auto offFooter = writer->WriteNTupleFooter(&footer, 1, 1);
   writer->Commit();

   auto rawFile = RRawFile::Create(fileGuard.GetPath());
   RMiniFileReader reader(rawFile.get());
   auto ntuple = reader.GetNTuple("MyNTuple").Inspect();
   EXPECT_EQ(offHeader, ntuple.GetSeekHeader());
   EXPECT_EQ(offFooter, ntuple.GetSeekFooter());

   char buf;
   reader.ReadBuffer(&buf, 1, offBlob);
   EXPECT_EQ(blob, buf);
   reader.ReadBuffer(&buf, 1, offHeader);
   EXPECT_EQ(header, buf);
   reader.ReadBuffer(&buf, 1, offFooter);
   EXPECT_EQ(footer, buf);
}

TEST(MiniFile, SimpleKeys)
{
   FileRaii fileGuard("test_ntuple_minifile_simple_keys.root");

   RNTupleWriteOptions options;
   // We check the file size at the end, so Direct I/O alignment requirements must not introduce padding.
   options.SetUseDirectIO(false);
   auto writer = RNTupleFileWriter::Recreate("MyNTuple", fileGuard.GetPath(), EContainerFormat::kTFile, options);

   char blob1 = '1';
   auto offBlob1 = writer->WriteBlob(&blob1, 1, 1);

   // Reserve a blob and fully write it.
   char blob2 = '2';
   auto offBlob2 = writer->ReserveBlob(1, 1);
   writer->WriteIntoReservedBlob(&blob2, 1, offBlob2);

   // Reserve a blob, but only write at the beginning.
   char blob3 = '3';
   auto offBlob3 = writer->ReserveBlob(2, 2);
   writer->WriteIntoReservedBlob(&blob3, 1, offBlob3);

   // Reserve a blob, but only write somewhere in the middle.
   char blob4 = '4';
   auto offBlob4 = writer->ReserveBlob(3, 3);
   auto offBlob4Write = offBlob4 + 1;
   writer->WriteIntoReservedBlob(&blob4, 1, offBlob4Write);

   // Reserve a blob, but don't write it at all.
   auto offBlob5 = writer->ReserveBlob(2, 2);

   // For good measure, write a final blob to make sure all indices match up.
   char blob6 = '6';
   auto offBlob6 = writer->WriteBlob(&blob6, 1, 1);

   writer->Commit();

   // Manually check the written keys.
   FILE *f = fopen(fileGuard.GetPath().c_str(), "rb");
   fseek(f, 0, SEEK_END);
   long size = ftell(f);
   rewind(f);

   std::unique_ptr<char[]> buffer(new char[size]);
   ASSERT_EQ(fread(buffer.get(), 1, size, f), size);

   Long64_t offset = 100;
   std::unique_ptr<TKey> key;
   auto readNextKey = [&]() {
      if (offset >= size) {
         return false;
      }

      char *keyBuffer = buffer.get() + offset;
      key.reset(new TKey(offset, /*size=*/0, nullptr));
      key->ReadKeyBuffer(keyBuffer);
      offset = key->GetSeekKey() + key->GetNbytes();
      return true;
   };

   ASSERT_TRUE(readNextKey());
   EXPECT_STREQ(key->GetClassName(), "TFile");

   ASSERT_TRUE(readNextKey());
   EXPECT_STREQ(key->GetClassName(), "RBlob");
   EXPECT_EQ(key->GetSeekKey() + key->GetKeylen(), offBlob1);
   EXPECT_EQ(buffer[offBlob1], blob1);

   ASSERT_TRUE(readNextKey());
   EXPECT_STREQ(key->GetClassName(), "RBlob");
   EXPECT_EQ(key->GetSeekKey() + key->GetKeylen(), offBlob2);
   EXPECT_EQ(buffer[offBlob2], blob2);

   ASSERT_TRUE(readNextKey());
   EXPECT_STREQ(key->GetClassName(), "RBlob");
   EXPECT_EQ(key->GetSeekKey() + key->GetKeylen(), offBlob3);
   EXPECT_EQ(buffer[offBlob3], blob3);

   ASSERT_TRUE(readNextKey());
   EXPECT_STREQ(key->GetClassName(), "RBlob");
   EXPECT_EQ(key->GetSeekKey() + key->GetKeylen(), offBlob4);
   EXPECT_EQ(buffer[offBlob4Write], blob4);

   ASSERT_TRUE(readNextKey());
   EXPECT_STREQ(key->GetClassName(), "RBlob");
   EXPECT_EQ(key->GetSeekKey() + key->GetKeylen(), offBlob5);

   ASSERT_TRUE(readNextKey());
   EXPECT_STREQ(key->GetClassName(), "RBlob");
   EXPECT_EQ(key->GetSeekKey() + key->GetKeylen(), offBlob6);
   EXPECT_EQ(buffer[offBlob6], blob6);

   ASSERT_TRUE(readNextKey());
   EXPECT_STREQ(key->GetClassName(), "ROOT::RNTuple");

   ASSERT_TRUE(readNextKey());
   // KeysList

   ASSERT_TRUE(readNextKey());
   EXPECT_STREQ(key->GetName(), "StreamerInfo");

   ASSERT_TRUE(readNextKey());
   // FreeSegments

   EXPECT_EQ(offset, size);
}

TEST(MiniFile, ProperKeys)
{
   FileRaii fileGuard("test_ntuple_minifile_proper_keys.root");

   std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   auto writer = RNTupleFileWriter::Append("MyNTuple", *file, RNTupleWriteOptions::kDefaultMaxKeySize);

   char blob1 = '1';
   auto offBlob1 = writer->WriteBlob(&blob1, 1, 1);

   // Reserve a blob and fully write it.
   char blob2 = '2';
   auto offBlob2 = writer->ReserveBlob(1, 1);
   writer->WriteIntoReservedBlob(&blob2, 1, offBlob2);

   // Reserve a blob, but only write at the beginning.
   char blob3 = '3';
   auto offBlob3 = writer->ReserveBlob(2, 2);
   writer->WriteIntoReservedBlob(&blob3, 1, offBlob3);

   // Reserve a blob, but only write somewhere in the middle.
   char blob4 = '4';
   auto offBlob4 = writer->ReserveBlob(3, 3);
   auto offBlob4Write = offBlob4 + 1;
   writer->WriteIntoReservedBlob(&blob4, 1, offBlob4Write);

   // Reserve a blob, but don't write it at all.
   auto offBlob5 = writer->ReserveBlob(2, 2);

   // For good measure, write a final blob to make sure all indices match up.
   char blob6 = '6';
   auto offBlob6 = writer->WriteBlob(&blob6, 1, 1);

   writer->Commit();

   // Manually check the written keys.
   FILE *f = fopen(fileGuard.GetPath().c_str(), "rb");
   fseek(f, 0, SEEK_END);
   long size = ftell(f);
   rewind(f);

   std::unique_ptr<char[]> buffer(new char[size]);
   ASSERT_EQ(fread(buffer.get(), 1, size, f), size);

   Long64_t offset = 100;
   std::unique_ptr<TKey> key;
   auto readNextKey = [&]() {
      if (offset >= size) {
         return false;
      }

      char *keyBuffer = buffer.get() + offset;
      key.reset(new TKey(offset, /*size=*/0, nullptr));
      key->ReadKeyBuffer(keyBuffer);
      offset = key->GetSeekKey() + key->GetNbytes();
      return true;
   };

   ASSERT_TRUE(readNextKey());
   EXPECT_STREQ(key->GetClassName(), "TFile");

   ASSERT_TRUE(readNextKey());
   EXPECT_STREQ(key->GetClassName(), "RBlob");
   EXPECT_EQ(key->GetSeekKey() + key->GetKeylen(), offBlob1);
   EXPECT_EQ(buffer[offBlob1], blob1);

   ASSERT_TRUE(readNextKey());
   EXPECT_STREQ(key->GetClassName(), "RBlob");
   EXPECT_EQ(key->GetSeekKey() + key->GetKeylen(), offBlob2);
   EXPECT_EQ(buffer[offBlob2], blob2);

   ASSERT_TRUE(readNextKey());
   EXPECT_STREQ(key->GetClassName(), "RBlob");
   EXPECT_EQ(key->GetSeekKey() + key->GetKeylen(), offBlob3);
   EXPECT_EQ(buffer[offBlob3], blob3);

   ASSERT_TRUE(readNextKey());
   EXPECT_STREQ(key->GetClassName(), "RBlob");
   EXPECT_EQ(key->GetSeekKey() + key->GetKeylen(), offBlob4);
   EXPECT_EQ(buffer[offBlob4Write], blob4);

   ASSERT_TRUE(readNextKey());
   EXPECT_STREQ(key->GetClassName(), "RBlob");
   EXPECT_EQ(key->GetSeekKey() + key->GetKeylen(), offBlob5);

   ASSERT_TRUE(readNextKey());
   EXPECT_STREQ(key->GetClassName(), "RBlob");
   EXPECT_EQ(key->GetSeekKey() + key->GetKeylen(), offBlob6);
   EXPECT_EQ(buffer[offBlob6], blob6);

   ASSERT_TRUE(readNextKey());
   EXPECT_STREQ(key->GetClassName(), "ROOT::RNTuple");

   ASSERT_TRUE(readNextKey());
   // KeysList

   ASSERT_TRUE(readNextKey());
   EXPECT_STREQ(key->GetName(), "StreamerInfo");

   ASSERT_TRUE(readNextKey());
   // FreeSegments

   EXPECT_EQ(offset, size);
}

TEST(MiniFile, LongString)
{
   FileRaii fileGuard("test_ntuple_minifile_long_string.root");

   static constexpr const char *LongString =
      "This is a very long text with exactly 254 characters, which is the maximum that the RNTupleWriter can currently "
      "store in a TFile header. For longer strings, a length of 255 is special and means that the first length byte is "
      "followed by an integer length.";
   std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE", LongString));
   auto writer = RNTupleFileWriter::Append("ntuple", *file, RNTupleWriteOptions::kDefaultMaxKeySize);

   char header = 'h';
   char footer = 'f';
   auto offHeader = writer->WriteNTupleHeader(&header, 1, 1);
   auto offFooter = writer->WriteNTupleFooter(&footer, 1, 1);
   writer->Commit();

   auto rawFile = RRawFile::Create(fileGuard.GetPath());
   RMiniFileReader reader(rawFile.get());
   auto ntuple1 = reader.GetNTuple("ntuple").Inspect();
   EXPECT_EQ(offHeader, ntuple1.GetSeekHeader());
   EXPECT_EQ(offFooter, ntuple1.GetSeekFooter());
}

TEST(MiniFile, MultiKeyBlob)
{
   FileRaii fileGuard("test_ntuple_minifile_multi_key_blob.root");

   const auto kMaxKeySize = 10 * 1024 * 1024; // 10 MiB
   const auto dataSize = kMaxKeySize * 2;
   auto data = std::make_unique<unsigned char[]>(dataSize);
   std::uint64_t blobOffset;

   {
      RNTupleWriteOptions options;
      RNTupleWriteOptionsManip::SetMaxKeySize(options, kMaxKeySize);
      auto writer = RNTupleFileWriter::Recreate("ntpl", fileGuard.GetPath(), EContainerFormat::kTFile, options);
      memset(data.get(), 0x99, dataSize);
      data[42] = 0x42;
      data[dataSize - 42] = 0x11;
      blobOffset = writer->WriteBlob(data.get(), dataSize, dataSize);
      writer->Commit();
   }
   {
      memset(data.get(), 0, dataSize);

      auto rawFile = RRawFile::Create(fileGuard.GetPath());
      auto reader = RMiniFileReader{rawFile.get()};
      auto ntuple = reader.GetNTuple("ntpl").Unwrap();
      reader.SetMaxKeySize(ntuple.GetMaxKeySize());
      reader.ReadBuffer(data.get(), dataSize, blobOffset);

      EXPECT_EQ(data[0], 0x99);
      EXPECT_EQ(data[dataSize / 2], 0x99);
      EXPECT_EQ(data[2 * dataSize / 3], 0x99);
      EXPECT_EQ(data[dataSize - 1], 0x99);
      EXPECT_EQ(data[42], 0x42);
      EXPECT_EQ(data[dataSize - 42], 0x11);
   }
}

TEST(MiniFile, MultiKeyBlob_ExactlyMax)
{
   // Write a payload that's exactly `maxKeySize` long and verify it doesn't split the key.

   FileRaii fileGuard("test_ntuple_minifile_multi_key_exact.root");

   const auto kMaxKeySize = 100 * 1024; // 100 KiB
   const auto dataSize = kMaxKeySize;
   auto data = std::make_unique<unsigned char[]>(dataSize);
   std::uint64_t blobOffset;

   {
      RNTupleWriteOptions options;
      RNTupleWriteOptionsManip::SetMaxKeySize(options, kMaxKeySize);
      auto writer = RNTupleFileWriter::Recreate("ntpl", fileGuard.GetPath(), EContainerFormat::kTFile, options);
      memset(data.get(), 0, dataSize);
      blobOffset = writer->WriteBlob(data.get(), dataSize, dataSize);
      writer->Commit();
   }
   {
      // Fill read buffer with sentinel data (they should be overwritten by zeroes)
      memset(data.get(), 0x99, dataSize);

      auto rawFile = RRawFile::Create(fileGuard.GetPath());
      auto reader = RMiniFileReader{rawFile.get()};
      auto ntuple = reader.GetNTuple("ntpl").Unwrap();
      reader.SetMaxKeySize(ntuple.GetMaxKeySize());
      rawFile->ReadAt(data.get(), dataSize, blobOffset);

      // If we didn't split the key, we expect to find all zeroes at the end of `data`.
      // Otherwise we will have some non-zero bytes, since it will host the next chunk offset.
      uint64_t lastU64 = *reinterpret_cast<uint64_t *>(&data[dataSize - sizeof(uint64_t)]);
      EXPECT_EQ(lastU64, 0);
   }
}

TEST(MiniFile, MultiKeyBlob_ExactlyTwo)
{
   // Write a payload that fits into two keys.

   FileRaii fileGuard("test_ntuple_minifile_multi_key_two.root");

   const auto kMaxKeySize = 100 * 1024; // 100 KiB
   const auto dataSize = 2 * kMaxKeySize - 8;
   auto data = std::make_unique<unsigned char[]>(dataSize);
   std::uint64_t blobOffset;

   {
      RNTupleWriteOptions options;
      RNTupleWriteOptionsManip::SetMaxKeySize(options, kMaxKeySize);
      auto writer = RNTupleFileWriter::Recreate("ntpl", fileGuard.GetPath(), EContainerFormat::kTFile, options);
      memset(data.get(), 0, dataSize / 2);
      memset(data.get() + dataSize / 2, 0x99, dataSize / 2);
      data[42] = 0x42;
      data[dataSize - 42] = 0x84;
      blobOffset = writer->WriteBlob(data.get(), dataSize, dataSize);
      writer->Commit();
   }
   {
      // Fill read buffer with sentinel data (they should be overwritten by zeroes)
      memset(data.get(), 0x99, dataSize);

      auto rawFile = RRawFile::Create(fileGuard.GetPath());
      auto reader = RMiniFileReader{rawFile.get()};
      auto ntuple = reader.GetNTuple("ntpl").Unwrap();
      reader.SetMaxKeySize(ntuple.GetMaxKeySize());

      rawFile->ReadAt(data.get(), dataSize, blobOffset);
      // If the blob was split into exactly two keys, there should be only one pointer to the next chunk.
      uint64_t secondLastU64 = *reinterpret_cast<uint64_t *>(&data[kMaxKeySize - 2 * sizeof(uint64_t)]);
      EXPECT_EQ(secondLastU64, 0);

      memset(data.get(), 0, dataSize);
      reader.ReadBuffer(data.get(), dataSize, blobOffset);

      EXPECT_EQ(data[0], 0);
      EXPECT_EQ(data[dataSize / 2 - 1], 0);
      EXPECT_EQ(data[2 * dataSize / 3], 0x99);
      EXPECT_EQ(data[dataSize - 1], 0x99);
      EXPECT_EQ(data[42], 0x42);
      EXPECT_EQ(data[dataSize - 42], 0x84);
   }
}

TEST(MiniFile, MultiKeyBlob_SmallKey)
{
   FileRaii fileGuard("test_ntuple_minifile_multi_key_blob_small_key.root");

   const auto kMaxKeySize = 50 * 1024; // 50 KiB
   const auto dataSize = kMaxKeySize * 1000;
   auto data = std::make_unique<unsigned char[]>(dataSize);
   std::uint64_t blobOffset;

   {
      RNTupleWriteOptions options;
      RNTupleWriteOptionsManip::SetMaxKeySize(options, kMaxKeySize);
      auto writer = RNTupleFileWriter::Recreate("ntpl", fileGuard.GetPath(), EContainerFormat::kTFile, options);
      memset(data.get(), 0x99, dataSize);
      data[42] = 0x42;
      data[dataSize - 42] = 0x84;
      blobOffset = writer->WriteBlob(data.get(), dataSize, dataSize);
      writer->Commit();
   }
   {
      memset(data.get(), 0, dataSize);

      auto rawFile = RRawFile::Create(fileGuard.GetPath());
      auto reader = RMiniFileReader{rawFile.get()};
      auto ntuple = reader.GetNTuple("ntpl").Unwrap();
      reader.SetMaxKeySize(ntuple.GetMaxKeySize());
      reader.ReadBuffer(data.get(), dataSize, blobOffset);

      EXPECT_EQ(data[0], 0x99);
      EXPECT_EQ(data[dataSize / 2], 0x99);
      EXPECT_EQ(data[2 * dataSize / 3], 0x99);
      EXPECT_EQ(data[dataSize - 1], 0x99);
      EXPECT_EQ(data[42], 0x42);
      EXPECT_EQ(data[dataSize - 42], 0x84);
   }
}

TEST(MiniFile, MultiKeyBlob_TooManyChunks)
{
   // Try writing more than the max possible number of chunks for a split key and verify it fails

#ifdef GTEST_FLAG_SET
   // Death tests must run single-threaded:
   // https://github.com/google/googletest/blob/main/docs/advanced.md#death-tests-and-threads
   GTEST_FLAG_SET(death_test_style, "threadsafe");
#endif

   FileRaii fileGuard("test_ntuple_minifile_multi_key_blob_small_key.root");

   const auto kMaxKeySize = 128;
   RNTupleWriteOptions options;
   RNTupleWriteOptionsManip::SetMaxKeySize(options, kMaxKeySize);

   {
      const auto kOkayDataSize = 1024;
      const auto data = std::make_unique<unsigned char[]>(kOkayDataSize);
      auto writer = RNTupleFileWriter::Recreate("ntpl", fileGuard.GetPath(), EContainerFormat::kTFile, options);
      memset(data.get(), 0x99, kOkayDataSize);
      writer->WriteBlob(data.get(), kOkayDataSize, kOkayDataSize);
      writer->Commit();
   }

   {
      const auto kTooBigDataSize = 5000;
      const auto data = std::make_unique<unsigned char[]>(kTooBigDataSize);
      auto writer = RNTupleFileWriter::Recreate("ntpl", fileGuard.GetPath(), EContainerFormat::kTFile, options);
      memset(data.get(), 0x99, kTooBigDataSize);
      EXPECT_DEATH(writer->WriteBlob(data.get(), kTooBigDataSize, kTooBigDataSize), "");
      writer->Commit();
   }
}

TEST(MiniFile, Multi)
{
   FileRaii fileGuard("test_ntuple_minifile_multi.root");

   std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   auto writer1 = RNTupleFileWriter::Append("FirstNTuple", *file, RNTupleWriteOptions::kDefaultMaxKeySize);
   auto writer2 = RNTupleFileWriter::Append("SecondNTuple", *file, RNTupleWriteOptions::kDefaultMaxKeySize);

   char header1 = 'h';
   char footer1 = 'f';
   char blob1 = 'b';
   char header2 = 'H';
   char footer2 = 'F';
   char blob2 = 'B';
   auto offHeader1 = writer1->WriteNTupleHeader(&header1, 1, 1);
   auto offHeader2 = writer2->WriteNTupleHeader(&header2, 1, 1);
   auto offBlob1 = writer1->WriteBlob(&blob1, 1, 1);
   auto offBlob2 = writer2->WriteBlob(&blob2, 1, 1);
   auto offFooter1 = writer1->WriteNTupleFooter(&footer1, 1, 1);
   auto offFooter2 = writer2->WriteNTupleFooter(&footer2, 1, 1);
   writer1->Commit();
   writer2->Commit();

   auto rawFile = RRawFile::Create(fileGuard.GetPath());
   RMiniFileReader reader(rawFile.get());
   auto ntuple1 = reader.GetNTuple("FirstNTuple").Inspect();
   EXPECT_EQ(offHeader1, ntuple1.GetSeekHeader());
   EXPECT_EQ(offFooter1, ntuple1.GetSeekFooter());
   auto ntuple2 = reader.GetNTuple("SecondNTuple").Inspect();
   EXPECT_EQ(offHeader2, ntuple2.GetSeekHeader());
   EXPECT_EQ(offFooter2, ntuple2.GetSeekFooter());

   char buf;
   reader.ReadBuffer(&buf, 1, offBlob1);
   EXPECT_EQ(blob1, buf);
   reader.ReadBuffer(&buf, 1, offHeader1);
   EXPECT_EQ(header1, buf);
   reader.ReadBuffer(&buf, 1, offFooter1);
   EXPECT_EQ(footer1, buf);
   reader.ReadBuffer(&buf, 1, offBlob2);
   EXPECT_EQ(blob2, buf);
   reader.ReadBuffer(&buf, 1, offHeader2);
   EXPECT_EQ(header2, buf);
   reader.ReadBuffer(&buf, 1, offFooter2);
   EXPECT_EQ(footer2, buf);
}

TEST(MiniFile, Failures)
{
   RNTupleWriteOptions options;
   // TODO(jblomer): failures should be exceptions
   EXPECT_DEATH(RNTupleFileWriter::Recreate("MyNTuple", "/can/not/open", EContainerFormat::kTFile, options), ".*");

   FileRaii fileGuard("test_ntuple_minifile_failures.root");

   auto writer = RNTupleFileWriter::Recreate("MyNTuple", fileGuard.GetPath(), EContainerFormat::kTFile, options);
   char header = 'h';
   char footer = 'f';
   char blob = 'b';
   writer->WriteNTupleHeader(&header, 1, 1);
   writer->WriteBlob(&blob, 1, 1);
   writer->WriteNTupleFooter(&footer, 1, 1);
   writer->Commit();

   auto rawFile = RRawFile::Create(fileGuard.GetPath());
   RMiniFileReader reader(rawFile.get());
   ROOT::RNTuple anchor;
   try {
      anchor = reader.GetNTuple("No such RNTuple").Inspect();
      FAIL() << "bad RNTuple names should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("no RNTuple named 'No such RNTuple' in file '" + fileGuard.GetPath()));
   }
}

TEST(MiniFile, KeyClassName)
{
   FileRaii fileGuard("test_ntuple_minifile_key_class_name.root");
   auto file = std::make_unique<TFile>(fileGuard.GetPath().c_str(), "RECREATE", "", 209);
   {
      auto tree = std::make_unique<TTree>("Events", "");
      file->Write();
   }
   file->Close();

   try {
      auto readerFail = RNTupleReader::Open("Events", fileGuard.GetPath());
      FAIL() << "RNTuple should only open Events key of type `RNTuple`";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("no RNTuple named 'Events' in file"));
   }
}

TEST(MiniFile, DifferentTKeys)
{
   FileRaii fileGuard("test_ntuple_minifile_different_tkeys.root");
   auto file = std::make_unique<TFile>(fileGuard.GetPath().c_str(), "RECREATE", "", 209);
   {
      auto tree = std::make_unique<TTree>("SomeTTree", "");
      tree->Fill();
      auto ntuple = RNTupleWriter::Append(RNTupleModel::Create(), "Events", *file);
      ntuple->Fill();
      file->Write();
   }

   file->Close();
   auto ntuple = RNTupleReader::Open("Events", fileGuard.GetPath());
   EXPECT_EQ(1, ntuple->GetNEntries());
}

TEST(MiniFile, StreamerInfo)
{
   FileRaii fileGuardProper("test_ntuple_minifile_streamer_info_proper.root");
   FileRaii fileGuardSimple("test_ntuple_minifile_streamer_info_simple.root");

   RNTupleSerializer::StreamerInfoMap_t streamerInfos;
   auto infoTVector2 = TClass::GetClass("TVector2")->GetStreamerInfo();
   auto infoTVector3 = TClass::GetClass("TVector3")->GetStreamerInfo();
   streamerInfos[infoTVector2->GetNumber()] = infoTVector2;
   streamerInfos[infoTVector3->GetNumber()] = infoTVector3;

   {
      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuardProper.GetPath().c_str(), "RECREATE"));
      auto writerProper = RNTupleFileWriter::Append("MyNTuple", *file, RNTupleWriteOptions::kDefaultMaxKeySize);
      writerProper->UpdateStreamerInfos(streamerInfos);
      writerProper->Commit();
   }

   {
      auto writerSimple = RNTupleFileWriter::Recreate(
         "ntpl", fileGuardSimple.GetPath(), RNTupleFileWriter::EContainerFormat::kTFile, RNTupleWriteOptions());
      writerSimple->UpdateStreamerInfos(streamerInfos);
      writerSimple->Commit();
   }

   std::vector<TVirtualStreamerInfo *> vecInfos;
   for (const auto &path : {fileGuardProper.GetPath(), fileGuardSimple.GetPath()}) {
      auto file = std::make_unique<TFile>(path.c_str());

      vecInfos.clear();
      for (auto info : TRangeDynCast<TVirtualStreamerInfo>(*file->GetStreamerInfoList())) {
         vecInfos.emplace_back(info);
      }

      auto fnComp = [](TVirtualStreamerInfo *a, TVirtualStreamerInfo *b) {
         return strcmp(a->GetName(), b->GetName()) < 0;
      };
      std::sort(vecInfos.begin(), vecInfos.end(), fnComp);
      ASSERT_EQ(3u, vecInfos.size());
      EXPECT_STREQ("ROOT::RNTuple", vecInfos[0]->GetName());
      EXPECT_STREQ("TVector2", vecInfos[1]->GetName());
      EXPECT_STREQ("TVector3", vecInfos[2]->GetName());
   }
}
