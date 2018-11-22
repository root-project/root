#include "RConfigure.h"
#include "ROOT/RRawFile.hxx"

#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "gtest/gtest.h"

using namespace ROOT::Detail;

namespace {

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

} // anonymous namespace


TEST(RRawFile, Empty)
{
   FileRaii emptyGuard("testEmpty", "");
   std::unique_ptr<RRawFile> f(RRawFile::Create("testEmpty"));
   EXPECT_EQ(0u, f->GetSize());
   EXPECT_EQ(0u, f->Read(nullptr, 0));
   EXPECT_EQ(0u, f->Pread(nullptr, 0, 1));
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

   std::unique_ptr<RRawFile> f2(RRawFile::Create("NoSuchFile"));
   EXPECT_THROW(f2->Readln(line), std::runtime_error);
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
   EXPECT_EQ(1u, f->Pread(&buffer, 1, 2));
   EXPECT_EQ('c', buffer);

}


TEST(RRawFile, ReadBufferd)
{
   FileRaii bufferedGuard("testBuffered", "abcdef");
   char buffer[8];
   RRawFile::ROptions options;
   options.fBlockSize = 2;
   std::unique_ptr<RRawFile> f(RRawFile::Create("testBuffered"));

   buffer[3] = '\0';
   EXPECT_EQ(3u, f->Pread(buffer, 3, 1));
   EXPECT_STREQ("bcd", buffer);

   buffer[2] = '\0';
   EXPECT_EQ(2u, f->Pread(buffer, 2, 2));
   EXPECT_STREQ("cd", buffer);
   EXPECT_EQ(2u, f->Pread(buffer, 2, 0));
   EXPECT_STREQ("ab", buffer);

   EXPECT_EQ(2u, f->Pread(buffer, 2, 0));
   EXPECT_STREQ("ab", buffer);
   EXPECT_EQ(1u, f->Pread(buffer, 1, 1));
   EXPECT_STREQ("bb", buffer);
   EXPECT_EQ(2u, f->Pread(buffer, 2, 1));
   EXPECT_STREQ("bc", buffer);
}
