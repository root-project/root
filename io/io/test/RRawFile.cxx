#include "RConfigure.h"
#include "ROOT/RRawFile.hxx"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "gtest/gtest.h"

using namespace ROOT::Experimental::Detail;

namespace {

/**
 * An RAII wrapper around an open temporary file on disk. It cleans up the guarded file when the wrapper object
 * goes out of scope.
 */
class FileRaii {
private:
   std::string fPath;
public:
   FileRaii(const std::string &path, const std::string &content) : fPath(path)
   {
      std::ofstream ostrm(path, std::ios::binary | std::ios::out | std::ios::trunc);
      ostrm << content;
   }
   FileRaii(const FileRaii&) = delete;
   FileRaii& operator=(const FileRaii&) = delete;
   ~FileRaii() {
      std::remove(fPath.c_str());
   }
};


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

   void DoOpen() final
   {
   }

   size_t DoReadAt(void *buffer, size_t nbytes, std::uint64_t offset) final
   {
      fNumReadAt++;
      if (offset > fContent.length())
         return 0;

      auto slice = fContent.substr(offset, nbytes);
      memcpy(buffer, slice.data(), slice.length());
      return slice.length();
   }

   std::uint64_t DoGetSize() final { return fContent.size(); }
};

} // anonymous namespace


TEST(RRawFile, Empty)
{
   FileRaii emptyGuard("testEmpty", "");
   std::unique_ptr<RRawFile> f(RRawFile::Create("testEmpty"));
   EXPECT_EQ(0u, f->GetSize());
   EXPECT_EQ(0u, f->Read(nullptr, 0));
   EXPECT_EQ(0u, f->ReadAt(nullptr, 0, 1));
   std::string line;
   EXPECT_FALSE(f->Readln(line));
}


TEST(RRawFile, Basic)
{
   FileRaii basicGuard("testBasic", "foo\nbar");
   std::unique_ptr<RRawFile> f(RRawFile::Create("testBasic"));
   EXPECT_EQ(7u, f->GetSize());
   std::string line;
   EXPECT_TRUE(f->Readln(line));
   EXPECT_STREQ("foo", line.c_str());
   EXPECT_TRUE(f->Readln(line));
   EXPECT_STREQ("bar", line.c_str());
   EXPECT_FALSE(f->Readln(line));
   auto clone = f->Clone();
   /// file pointer is reset by clone
   EXPECT_TRUE(clone->Readln(line));
   EXPECT_STREQ("foo", line.c_str());

   std::unique_ptr<RRawFile> f2(RRawFile::Create("NoSuchFile"));
   EXPECT_THROW(f2->Readln(line), std::runtime_error);

   std::unique_ptr<RRawFile> f3(RRawFile::Create("FiLE://testBasic"));
   EXPECT_EQ(7u, f3->GetSize());

   EXPECT_THROW(RRawFile::Create("://testBasic"), std::runtime_error);
   EXPECT_THROW(RRawFile::Create("Communicator://Kirk"), std::runtime_error);
}


TEST(RRawFile, Remote)
{
#ifdef R__HAS_DAVIX
   std::unique_ptr<RRawFile> f(RRawFile::Create("http://root.cern.ch/files/davix.test"));
   std::string line;
   EXPECT_TRUE(f->Readln(line));
   EXPECT_STREQ("Hello, World", line.c_str());
#else
   EXPECT_THROW(RRawFile::Create("http://root.cern.ch/files/davix.test"), std::runtime_error);
#endif
}


TEST(RRawFile, Readln)
{
   FileRaii linebreakGuard("testLinebreak", "foo\r\none\nline\r\n\r\n");
   std::unique_ptr<RRawFile> f(RRawFile::Create("testLinebreak"));
   std::string line;
   EXPECT_TRUE(f->Readln(line));
   EXPECT_STREQ("foo", line.c_str());
   EXPECT_TRUE(f->Readln(line));
   EXPECT_STREQ("one\nline", line.c_str());
   EXPECT_TRUE(f->Readln(line));
   EXPECT_TRUE(line.empty());
   EXPECT_FALSE(f->Readln(line));
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
   FileRaii directGuard("testDirect", "abc");
   char buffer;
   RRawFile::ROptions options;
   options.fBlockSize = 0;
   std::unique_ptr<RRawFile> f(RRawFile::Create("testDirect"));
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
