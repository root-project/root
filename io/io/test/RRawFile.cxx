#include "RConfigure.h"
#include "ROOT/RRawFile.hxx"

#include <cstdio>
#include <fstream>
#include <memory>
#include <string>

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
   FileRaii empty("test_empty", "");
   std::unique_ptr<RRawFile> f(RRawFile::Create("test_empty"));
   EXPECT_EQ(0u, f->GetSize());
   EXPECT_EQ(0u, f->Read(nullptr, 0));
   EXPECT_EQ(0u, f->Pread(nullptr, 0, 1));
   std::string line;
   EXPECT_FALSE(f->Readln(line));
}


TEST(RRawFile, Basic)
{
   FileRaii empty("test_basic", "foo\nbar");
   std::unique_ptr<RRawFile> f(RRawFile::Create("test_basic"));
   EXPECT_EQ(7u, f->GetSize());
   std::string line;
   EXPECT_TRUE(f->Readln(line));
   EXPECT_STREQ("foo", line.c_str());
   EXPECT_TRUE(f->Readln(line));
   EXPECT_STREQ("bar", line.c_str());
   EXPECT_FALSE(f->Readln(line));
}
