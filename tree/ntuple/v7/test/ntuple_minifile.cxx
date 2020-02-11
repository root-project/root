#include "gtest/gtest.h"

#include <ROOT/RMiniFile.hxx>
#include <ROOT/RRawFile.hxx>

#include <TFile.h>

#include <iostream>
#include <memory>

using ENTupleContainerFormat = ROOT::Experimental::ENTupleContainerFormat;
using RMiniFileReader = ROOT::Experimental::Internal::RMiniFileReader;
using RNTupleFileWriter = ROOT::Experimental::Internal::RNTupleFileWriter;
using RNTuple = ROOT::Experimental::RNTuple;
using RRawFile = ROOT::Internal::RRawFile;

namespace {

/**
 * An RAII wrapper around an open temporary file on disk. It cleans up the guarded file when the wrapper object
 * goes out of scope.
 */
class FileRaii {
private:
   std::string fPath;
public:
   explicit FileRaii(const std::string &path) : fPath(path) { }
   FileRaii(const FileRaii&) = delete;
   FileRaii& operator=(const FileRaii&) = delete;
   ~FileRaii() { std::remove(fPath.c_str()); }
   std::string GetPath() const { return fPath; }
};

} // anonymous namespace


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

   auto rawFile = std::unique_ptr<RRawFile>(RRawFile::Create(fileGuard.GetPath()));
   RMiniFileReader reader(rawFile.get());
   auto ntuple = reader.GetNTuple("MyNTuple");
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

   auto rawFile = std::unique_ptr<RRawFile>(RRawFile::Create(fileGuard.GetPath()));
   RMiniFileReader reader(rawFile.get());
   auto ntuple = reader.GetNTuple("MyNTuple");
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

   auto rawFile = std::unique_ptr<RRawFile>(RRawFile::Create(fileGuard.GetPath()));
   RMiniFileReader reader(rawFile.get());
   auto ntuple = reader.GetNTuple("MyNTuple");
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

   auto rawFile = std::unique_ptr<RRawFile>(RRawFile::Create(fileGuard.GetPath()));
   RMiniFileReader reader(rawFile.get());
   auto ntuple1 = reader.GetNTuple("FirstNTuple");
   EXPECT_EQ(offHeader1, ntuple1.fSeekHeader);
   EXPECT_EQ(offFooter1, ntuple1.fSeekFooter);
   auto ntuple2 = reader.GetNTuple("SecondNTuple");
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

   auto rawFile = std::unique_ptr<RRawFile>(RRawFile::Create(fileGuard.GetPath()));
   RMiniFileReader reader(rawFile.get());
   EXPECT_DEATH(reader.GetNTuple("No such NTiple"), ".*");
}
