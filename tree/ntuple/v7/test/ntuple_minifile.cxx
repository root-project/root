#include "ntuple_test.hxx"

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
   auto k = std::unique_ptr<RNTuple>(file->Get<RNTuple>("MyNTuple"));
   EXPECT_EQ(ntuple, *k);
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
   RNTuple ntuple;
   try {
      ntuple = reader.GetNTuple("No such RNTuple").Inspect();
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

TEST(MiniFile, FailOnForwardIncompatibility)
{
   FileRaii fileGuard("test_ntuple_minifile_forward_incompat.root");

   // First create a regular RNTuple
   auto model = RNTupleModel::Create();
   auto fldPt = model->MakeField<float>("pt", 42.0);
   {
      RNTupleWriteOptions options;
      options.SetCompression(0);
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), options);
      writer->Fill();
   }
   {
      auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
      ASSERT_EQ(1U, reader->GetNEntries());
      reader->LoadEntry(0);
      EXPECT_EQ(42.0, *(reader->GetModel()->GetDefaultEntry()->Get<float>("pt")));
   }

   // Fix the version numbers in the header

   // Figure out the header offset
   auto rawFile = RRawFile::Create(fileGuard.GetPath());
   RMiniFileReader reader(rawFile.get());
   auto ntuple = reader.GetNTuple("ntuple").Inspect();
   // Construct incompatible version numbers in little-endian binary format
   std::uint16_t futureVersion = RNTupleSerializer::kEnvelopeCurrentVersion + 1;
   unsigned char futureVersionLE[2];
   futureVersionLE[0] = (futureVersion & 0x00FF);
   futureVersionLE[1] = (futureVersion & 0xFF00) >> 8;
   // Write out twice (min version and writer version)
   FILE *f = fopen(fileGuard.GetPath().c_str(), "rb+");
   ASSERT_TRUE(f != nullptr);
   int posHeader = ntuple.fSeekHeader;
   EXPECT_EQ(0, fseek(f, posHeader, SEEK_SET));
   EXPECT_EQ(2u, fwrite(futureVersionLE, 1, 2, f));
   EXPECT_EQ(2u, fwrite(futureVersionLE, 1, 2, f));
   fclose(f);

   try {
      auto readerFail = RNTupleReader::Open("ntuple", fileGuard.GetPath());
      FAIL() << "unsupported minimum version number should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("RNTuple format is too new"));
   }
}
