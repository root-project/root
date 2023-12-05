#include "ntuple_test.hxx"
#include <TKey.h>
#include <TTree.h>

namespace {
bool IsEqual(const ROOT::Experimental::RNTuple &a, const ROOT::Experimental::RNTuple &b)
{
   return a.fVersionEpoch == b.fVersionEpoch && a.fVersionMajor == b.fVersionMajor &&
          a.fVersionMinor == b.fVersionMinor && a.fVersionPatch == b.fVersionPatch && a.fSeekHeader == b.fSeekHeader &&
          a.fNBytesHeader == b.fNBytesHeader && a.fLenHeader == b.fLenHeader && a.fSeekFooter == b.fSeekFooter &&
          a.fNBytesFooter == b.fNBytesFooter && a.fLenFooter == b.fLenFooter && a.fChecksum == b.fChecksum;
}

struct RNTupleTester {
   ROOT::Experimental::RNTuple fNtpl;

   explicit RNTupleTester(const ROOT::Experimental::RNTuple &ntpl) : fNtpl(ntpl) {}
   RNTuple GetAnchor() const { return fNtpl; }
};
} // namespace

TEST(MiniFile, Raw)
{
   FileRaii fileGuard("test_ntuple_minifile_raw.ntuple");

   auto writer = std::unique_ptr<RNTupleFileWriter>(
      RNTupleFileWriter::Recreate("MyNTuple", fileGuard.GetPath(), 0, ENTupleContainerFormat::kBare));
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
   EXPECT_EQ(offHeader, ntuple.fSeekHeader);
   EXPECT_EQ(offFooter, ntuple.fSeekFooter);

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

   auto writer = std::unique_ptr<RNTupleFileWriter>(
      RNTupleFileWriter::Recreate("MyNTuple", fileGuard.GetPath(), 0, ENTupleContainerFormat::kTFile));
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
   EXPECT_EQ(offHeader, ntuple.fSeekHeader);
   EXPECT_EQ(offFooter, ntuple.fSeekFooter);

   char buf;
   reader.ReadBuffer(&buf, 1, offBlob);
   EXPECT_EQ(blob, buf);
   reader.ReadBuffer(&buf, 1, offHeader);
   EXPECT_EQ(header, buf);
   reader.ReadBuffer(&buf, 1, offFooter);
   EXPECT_EQ(footer, buf);

   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "READ"));
   ASSERT_TRUE(file);
   auto k = std::unique_ptr<ROOT::Experimental::RNTuple>(file->Get<ROOT::Experimental::RNTuple>("MyNTuple"));
   EXPECT_TRUE(IsEqual(ntuple, RNTupleTester(*k).GetAnchor()));
}


TEST(MiniFile, Proper)
{
   FileRaii fileGuard("test_ntuple_minifile_proper.root");

   std::unique_ptr<TFile> file;
   auto writer = std::unique_ptr<RNTupleFileWriter>(RNTupleFileWriter::Recreate("MyNTuple", fileGuard.GetPath(), file));

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
   EXPECT_EQ(offHeader, ntuple.fSeekHeader);
   EXPECT_EQ(offFooter, ntuple.fSeekFooter);

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

   auto writer = std::unique_ptr<RNTupleFileWriter>(
      RNTupleFileWriter::Recreate("MyNTuple", fileGuard.GetPath(), 0, ENTupleContainerFormat::kTFile));

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
   EXPECT_STREQ(key->GetClassName(), "ROOT::Experimental::RNTuple");

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

   std::unique_ptr<TFile> file;
   auto writer = std::unique_ptr<RNTupleFileWriter>(RNTupleFileWriter::Recreate("MyNTuple", fileGuard.GetPath(), file));

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
   EXPECT_STREQ(key->GetClassName(), "ROOT::Experimental::RNTuple");

   ASSERT_TRUE(readNextKey());
   // KeysList

   ASSERT_TRUE(readNextKey());
   EXPECT_STREQ(key->GetName(), "StreamerInfo");

   ASSERT_TRUE(readNextKey());
   // FreeSegments

   EXPECT_EQ(offset, size);
}

TEST(MiniFile, Multi)
{
   FileRaii fileGuard("test_ntuple_minifile_multi.root");

   std::unique_ptr<TFile> file;
   auto writer1 =
      std::unique_ptr<RNTupleFileWriter>(RNTupleFileWriter::Recreate("FirstNTuple", fileGuard.GetPath(), file));
   auto writer2 = std::unique_ptr<RNTupleFileWriter>(RNTupleFileWriter::Append("SecondNTuple", *file));

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
   EXPECT_EQ(offHeader1, ntuple1.fSeekHeader);
   EXPECT_EQ(offFooter1, ntuple1.fSeekFooter);
   auto ntuple2 = reader.GetNTuple("SecondNTuple").Inspect();
   EXPECT_EQ(offHeader2, ntuple2.fSeekHeader);
   EXPECT_EQ(offFooter2, ntuple2.fSeekFooter);

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
   // TODO(jblomer): failures should be exceptions
   EXPECT_DEATH(RNTupleFileWriter::Recreate("MyNTuple", "/can/not/open", 0, ENTupleContainerFormat::kTFile), ".*");

   FileRaii fileGuard("test_ntuple_minifile_failures.root");

   auto writer = std::unique_ptr<RNTupleFileWriter>(
      RNTupleFileWriter::Recreate("MyNTuple", fileGuard.GetPath(), 0, ENTupleContainerFormat::kTFile));
   char header = 'h';
   char footer = 'f';
   char blob = 'b';
   writer->WriteNTupleHeader(&header, 1, 1);
   writer->WriteBlob(&blob, 1, 1);
   writer->WriteNTupleFooter(&footer, 1, 1);
   writer->Commit();

   auto rawFile = RRawFile::Create(fileGuard.GetPath());
   RMiniFileReader reader(rawFile.get());
   ROOT::Experimental::RNTuple anchor;
   try {
      anchor = reader.GetNTuple("No such RNTuple").Inspect();
      FAIL() << "bad RNTuple names should throw";
   } catch (const RException& err) {
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
