#include "gtest/gtest.h"

#include "TSystem.h"
#include "TString.h"
#include "TROOT.h"

#include <ROOT/FoundationUtils.hxx>

#include <string>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <streambuf>

static const char kPathSep = ROOT::FoundationUtils::GetEnvPathSeparator();

TEST(TSystem, TempFile)
{
   TString fname = "root_test_";
   auto ftmp = gSystem->TempFileName(fname);

   EXPECT_TRUE(fname.Length() > 10);
   EXPECT_TRUE(ftmp != nullptr);

   std::string content = "test_temp_file_content";
   auto res_write = fwrite(content.data(), 1, content.length(), ftmp);
   EXPECT_EQ(res_write, content.length());

   auto res_close = fclose(ftmp);
   EXPECT_EQ(res_close, 0);

   std::ifstream fread(fname.Data());
   std::string str((std::istreambuf_iterator<char>(fread)), std::istreambuf_iterator<char>());
   EXPECT_STREQ(content.c_str(), str.c_str());

   gSystem->Unlink(fname);
}

// Count occurrences of `dir` as a full path component of `path`.
static int CountPathComponent(const TString &path, const TString &dir)
{
   int count = 0;
   TString token;
   Ssiz_t from = 0;
   while (path.Tokenize(token, from, TString::Format("%c", kPathSep)))
      if (token == dir)
         ++count;
   return count;
}

// Exercise the interplay of Get/Add/SetDynamicPath as a single scenario:
// the dynamic path is process-global state, so the steps are order dependent.
TEST(TSystem, DynamicPath)
{
   const TString defaultPath = gSystem->GetDynamicPath();
   // The ROOT library directory is always part of the default path.
   EXPECT_TRUE(defaultPath.Contains(TROOT::GetLibDir()))
      << "default path: " << defaultPath;

   // AddDynamicPath appends the directory (at the end) ...
   // (the directories do not need to exist; use names that cannot already be
   // part of the default path)
   const TString extraDir1 = TString::Format("%s/root-dynpath-gtest-1", gSystem->TempDirectory());
   gSystem->AddDynamicPath(extraDir1);
   TString path = gSystem->GetDynamicPath();
   EXPECT_TRUE(path.EndsWith(extraDir1)) << "path: " << path;
   EXPECT_EQ(1, CountPathComponent(path, extraDir1)) << "path: " << path;
   // ... and keeps the rest of the path intact.
   EXPECT_TRUE(path.BeginsWith(defaultPath)) << "path: " << path;

   // Appended directories accumulate in order.
   const TString extraDir2 = TString::Format("%s/root-dynpath-gtest-2", gSystem->TempDirectory());
   gSystem->AddDynamicPath(extraDir2);
   path = gSystem->GetDynamicPath();
   EXPECT_TRUE(path.EndsWith(TString::Format("%s%c%s", extraDir1.Data(), kPathSep, extraDir2.Data())))
      << "path: " << path;

   // AddDynamicPath(nullptr) is a no-op.
   gSystem->AddDynamicPath(nullptr);
   EXPECT_STREQ(path, gSystem->GetDynamicPath());

   // SetDynamicPath freezes the path to exactly the given value.
   const TString userPath = TString::Format("%s%c%s", extraDir2.Data(), kPathSep, extraDir1.Data());
   gSystem->SetDynamicPath(userPath);
   EXPECT_STREQ(userPath, gSystem->GetDynamicPath());

   // SetDynamicPath(nullptr) resets to the default: the explicitly set value
   // and the previously appended directories are gone, the ROOT library
   // directory is back.
   gSystem->SetDynamicPath(nullptr);
   path = gSystem->GetDynamicPath();
   EXPECT_TRUE(path.Contains(TROOT::GetLibDir())) << "path: " << path;
   EXPECT_EQ(0, CountPathComponent(path, extraDir2)) << "path: " << path;
}

TEST(TSystem, TempFileSuffix)
{
   TString fname = "root_suffix_test_";
   const char *suffix = ".txt";
   auto ftmp = gSystem->TempFileName(fname, nullptr, suffix);

   EXPECT_TRUE(fname.Length() > 16);
   EXPECT_TRUE(ftmp != nullptr);

   // check that suffix really at the end of the file name
   EXPECT_STREQ(fname(fname.Length() - strlen(suffix), strlen(suffix)).Data(), suffix);

   std::string content = "test_temp_file_content_suffix";
   auto res_write = fwrite(content.data(), 1, content.length(), ftmp);
   EXPECT_EQ(res_write, content.length());

   auto res_close = fclose(ftmp);
   EXPECT_EQ(res_close, 0);

   std::ifstream fread(fname.Data());
   std::string str((std::istreambuf_iterator<char>(fread)), std::istreambuf_iterator<char>());
   EXPECT_STREQ(content.c_str(), str.c_str());

   gSystem->Unlink(fname);
}
