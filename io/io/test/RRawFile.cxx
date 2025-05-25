#include "io_test.hxx"

#include "TFile.h"

#include "ROOT/RRawFileTFile.hxx"
using ROOT::Internal::RRawFileTFile;

namespace {

/**
 * A minimal RRawFile implementation that serves data from a string. It keeps a counter of the number of read calls
 * to help veryfing the buffer logic in the base class.
 */
class RRawFileMock : public RRawFile {
public:
   std::string fContent;
   unsigned fNumReadAt;

   RRawFileMock(const std::string &content, RRawFile::ROptions options)
     : RRawFile("", options), fContent(content), fNumReadAt(0) { }

   std::unique_ptr<RRawFile> Clone() const final {
      return std::make_unique<RRawFileMock>(fContent, fOptions);
   }

   void OpenImpl() final
   {
   }

   size_t ReadAtImpl(void *buffer, size_t nbytes, std::uint64_t offset) final
   {
      fNumReadAt++;
      if (offset > fContent.length())
         return 0;

      auto slice = fContent.substr(offset, nbytes);
      memcpy(buffer, slice.data(), slice.length());
      return slice.length();
   }

   std::uint64_t GetSizeImpl() final { return fContent.size(); }
};

} // anonymous namespace


TEST(RRawFile, Empty)
{
   FileRaii emptyGuard("test_rrawfile_empty", "");
   auto f = RRawFile::Create(emptyGuard.GetPath());
   EXPECT_FALSE(f->IsOpen());
   EXPECT_EQ(0u, f->GetSize());
   EXPECT_EQ(0u, f->GetFilePos());
   EXPECT_EQ(0u, f->Read(nullptr, 0));
   EXPECT_EQ(0u, f->ReadAt(nullptr, 0, 1));
   std::string line;
   EXPECT_FALSE(f->Readln(line));
   EXPECT_TRUE(f->IsOpen());

   RRawFile::ROptions options;
   options.fBlockSize = 0;
   f = RRawFile::Create(emptyGuard.GetPath(), options);
   EXPECT_EQ(0u, f->Read(nullptr, 0));
   EXPECT_EQ(0u, f->ReadAt(nullptr, 0, 1));
}


TEST(RRawFile, Basic)
{
   FileRaii basicGuard("test_rrawfile_basic", "foo\nbar");
   auto f = RRawFile::Create(basicGuard.GetPath());
   EXPECT_EQ(7u, f->GetSize());
   std::string line;
   EXPECT_TRUE(f->Readln(line));
   EXPECT_STREQ("foo", line.c_str());
   EXPECT_TRUE(f->Readln(line));
   EXPECT_STREQ("bar", line.c_str());
   EXPECT_FALSE(f->Readln(line));
   auto clone = f->Clone();
   // file pointer is reset by clone
   EXPECT_TRUE(clone->Readln(line));
   EXPECT_STREQ("foo", line.c_str());
   // Rinse and repeat
   EXPECT_EQ(4U, clone->GetFilePos());
   clone->Seek(0);
   EXPECT_TRUE(clone->Readln(line));
   EXPECT_STREQ("foo", line.c_str());

   auto f2 = RRawFile::Create("NoSuchFile");
   EXPECT_THROW(f2->Readln(line), std::runtime_error);

   auto f3 = RRawFile::Create(std::string("FiLE://") + basicGuard.GetPath());
   EXPECT_EQ(7u, f3->GetSize());

   EXPECT_THROW(RRawFile::Create(std::string("://") + basicGuard.GetPath()), std::runtime_error);
   EXPECT_THROW(RRawFile::Create("Communicator://Kirk"), std::runtime_error);
}


TEST(RRawFile, Remote)
{
#ifdef R__HAS_DAVIX
   auto f = RRawFile::Create("http://root.cern/files/davix.test");
   std::string line;
   EXPECT_TRUE(f->Readln(line));
   EXPECT_STREQ("Hello, World", line.c_str());
#else
   EXPECT_THROW(RRawFile::Create("http://root.cern/files/davix.test"), std::runtime_error);
#endif
}


TEST(RRawFile, Readln)
{
   FileRaii linebreakGuard("test_rrawfile_linebreak", "foo\r\none\nline\r\n\r\n");
   auto f = RRawFile::Create(linebreakGuard.GetPath());
   std::string line;
   EXPECT_TRUE(f->Readln(line));
   EXPECT_STREQ("foo", line.c_str());
   EXPECT_TRUE(f->Readln(line));
   EXPECT_STREQ("one\nline", line.c_str());
   EXPECT_TRUE(f->Readln(line));
   EXPECT_TRUE(line.empty());
   EXPECT_FALSE(f->Readln(line));
}


TEST(RRawFile, ReadV)
{
   FileRaii readvGuard("test_rawfile_readv", "Hello, World");
   auto f = RRawFile::Create(readvGuard.GetPath());

   char buffer[2];
   buffer[0] = buffer[1] = 0;
   RRawFile::RIOVec iovec[2];
   iovec[0].fBuffer = &buffer[0];
   iovec[0].fOffset = 0;
   iovec[0].fSize = 1;
   iovec[1].fBuffer = &buffer[1];
   iovec[1].fOffset = 11;
   iovec[1].fSize = 2;
   f->ReadV(iovec, 2);

   EXPECT_EQ(1U, iovec[0].fOutBytes);
   EXPECT_EQ(1U, iovec[1].fOutBytes);
   EXPECT_EQ('H', buffer[0]);
   EXPECT_EQ('d', buffer[1]);
}


TEST(RRawFile, SplitUrl)
{
   EXPECT_STREQ("C:\\Data\\events.root", RRawFile::GetLocation("C:\\Data\\events.root").c_str());
   EXPECT_STREQ("///many/slashes", RRawFile::GetLocation("///many/slashes").c_str());
   EXPECT_STREQ("/many/slashes", RRawFile::GetLocation(":///many/slashes").c_str());
   EXPECT_STREQ("file", RRawFile::GetTransport("/foo").c_str());
   EXPECT_STREQ("http", RRawFile::GetTransport("http://").c_str());
   EXPECT_STREQ("", RRawFile::GetLocation("http://").c_str());
   EXPECT_STREQ("http", RRawFile::GetTransport("http://file:///bar").c_str());
}


TEST(RRawFile, ReadDirect)
{
   FileRaii directGuard("test_rrawfile_direct", "abc");
   char buffer;
   RRawFile::ROptions options;
   options.fBlockSize = 0;
   auto f = RRawFile::Create(directGuard.GetPath());
   EXPECT_EQ(0u, f->Read(&buffer, 0));
   EXPECT_EQ(1u, f->Read(&buffer, 1));
   EXPECT_EQ('a', buffer);
   EXPECT_EQ(1u, f->ReadAt(&buffer, 1, 2));
   EXPECT_EQ('c', buffer);

}


TEST(RRawFile, ReadBuffered)
{
   char buffer[8];
   RRawFile::ROptions options;
   options.fBlockSize = 2;
   std::unique_ptr<RRawFileMock> f(new RRawFileMock("abcdef", options));

   buffer[3] = '\0';
   EXPECT_EQ(3u, f->ReadAt(buffer, 3, 1));
   EXPECT_STREQ("bcd", buffer);
   EXPECT_EQ(1u, f->fNumReadAt); f->fNumReadAt = 0;

   buffer[2] = '\0';
   EXPECT_EQ(2u, f->ReadAt(buffer, 2, 2));
   EXPECT_STREQ("cd", buffer);
   EXPECT_EQ(2u, f->ReadAt(buffer, 2, 0));
   EXPECT_STREQ("ab", buffer);
   EXPECT_EQ(2u, f->ReadAt(buffer, 2, 2));
   EXPECT_STREQ("cd", buffer);
   EXPECT_EQ(2u, f->ReadAt(buffer, 2, 1));
   EXPECT_STREQ("bc", buffer);
   EXPECT_EQ(2u, f->fNumReadAt); f->fNumReadAt = 0;

   EXPECT_EQ(2u, f->ReadAt(buffer, 2, 0));
   EXPECT_STREQ("ab", buffer);
   EXPECT_EQ(1u, f->ReadAt(buffer, 1, 1));
   EXPECT_STREQ("bb", buffer);
   EXPECT_EQ(2u, f->ReadAt(buffer, 2, 1));
   EXPECT_STREQ("bc", buffer);
   EXPECT_EQ(0u, f->fNumReadAt); f->fNumReadAt = 0;
   EXPECT_EQ(2u, f->ReadAt(buffer, 2, 3));
   EXPECT_STREQ("de", buffer);
   EXPECT_EQ(1u, f->fNumReadAt); f->fNumReadAt = 0;
   EXPECT_EQ(1u, f->ReadAt(buffer, 1, 2));
   EXPECT_STREQ("ce", buffer);
   EXPECT_EQ(0u, f->fNumReadAt); f->fNumReadAt = 0;
   EXPECT_EQ(1u, f->ReadAt(buffer, 1, 1));
   EXPECT_STREQ("be", buffer);
   EXPECT_EQ(1u, f->fNumReadAt); f->fNumReadAt = 0;
}

TEST(RRawFile, SetBuffering)
{
   char buffer[3];
   RRawFile::ROptions options;
   options.fBlockSize = 2;
   std::unique_ptr<RRawFileMock> f(new RRawFileMock("abcd", options));

   buffer[2] = '\0';
   EXPECT_EQ(1u, f->ReadAt(buffer, 1, 0));
   EXPECT_EQ(1u, f->ReadAt(buffer + 1, 1, 1));
   EXPECT_STREQ("ab", buffer);
   EXPECT_EQ(1u, f->fNumReadAt);
   f->fNumReadAt = 0;

   f->SetBuffering(false);
   // idempotent
   f->SetBuffering(false);
   EXPECT_EQ(1u, f->ReadAt(buffer, 1, 0));
   EXPECT_EQ(1u, f->ReadAt(buffer + 1, 1, 1));
   EXPECT_STREQ("ab", buffer);
   EXPECT_EQ(2u, f->fNumReadAt);
   f->fNumReadAt = 0;

   f->SetBuffering(true);
   // idempotent
   f->SetBuffering(true);
   EXPECT_EQ(1u, f->ReadAt(buffer, 1, 2));
   EXPECT_EQ(1u, f->ReadAt(buffer + 1, 1, 3));
   EXPECT_STREQ("cd", buffer);
   EXPECT_EQ(1u, f->fNumReadAt);
   f->fNumReadAt = 0;
}

TEST(RRawFileTFile, TFile)
{
   FileRaii tfileGuard("test_rawfile_tfile.root", "");

   std::unique_ptr<TFile> file(TFile::Open(tfileGuard.GetPath().c_str(), "RECREATE"));
   file->Write();

   auto rawFile = std::make_unique<RRawFileTFile>(file.get());

   // The first four bytes should be 'root'.
   char root[5] = {};
   rawFile->ReadAt(root, 4, 0);
   EXPECT_STREQ(root, "root");

   // fBEGIN = 100, and its seek key should be 100.
   unsigned char seek[4] = {};
   rawFile->ReadAt(seek, 4, 100 + 18);
   EXPECT_EQ(seek[0], 0);
   EXPECT_EQ(seek[1], 0);
   EXPECT_EQ(seek[2], 0);
   EXPECT_EQ(seek[3], 100);
}
