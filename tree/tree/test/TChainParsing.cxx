#include <string>
#include <vector>

#include "TChain.h"
#include "TFile.h"
#include "TSystem.h"
#include "TTree.h"
#include "TROOT.h"
#include "TPluginManager.h"

#include "ROOT/InternalTreeUtils.hxx"

#include "gtest/gtest.h"

void EXPECT_VEC_EQ(const std::vector<std::string> &v1, const std::vector<std::string> &v2)
{
   ASSERT_EQ(v1.size(), v2.size());
   for (std::size_t i = 0ul; i < v1.size(); ++i) {
      EXPECT_EQ(v1[i], v2[i]);
   }
}

std::string ConcatUnixFileName(const char *dir, const char *name)
{
   std::unique_ptr<char[]> fileName{gSystem->ConcatFileName(dir, name)};
   return gSystem->UnixPathName(fileName.get());
}

TEST(TChainParsing, RemoteAdd)
{
   TChain c("defaultname");
   c.Add("root://some.domain/path/to/file.root/treename");
   c.Add("root://some.domain//path/to/file.root/treename");
   c.Add("root://some.domain/path/to/foo.something/file.root/treename");
   c.Add("root://some.domain/path/to/foo.root/file.root/treename"); // ROOT-9344
   c.Add("root://some.domain/path//to/file.root/treename"); // ROOT-10494
   c.Add("root://some.domain//path//to//file.root//treename");
   c.Add("https://some.domain:8443/path/to/file.root.1?#treename");
   c.Add("https://some.domain:8443/path/to/file.root.1#anchor");
   c.Add("https://some.domain:8443/path/to/file.root.1?a=b&x=y&");
   c.Add("https://some.domain:8443/path/to/file.root.1?a=b&x=y&#treename");
   c.Add("https://some.domain:8443/path/to/file.root.1?a=b&x=y&#X=Y");
   const auto files = c.GetListOfFiles();

   EXPECT_STREQ(files->At(0)->GetTitle(), "root://some.domain/path/to/file.root");
   EXPECT_STREQ(files->At(0)->GetName(), "treename");

   EXPECT_STREQ(files->At(1)->GetTitle(), "root://some.domain//path/to/file.root");
   EXPECT_STREQ(files->At(1)->GetName(), "treename");

   EXPECT_STREQ(files->At(2)->GetTitle(), "root://some.domain/path/to/foo.something/file.root");
   EXPECT_STREQ(files->At(2)->GetName(), "treename");

   EXPECT_STREQ(files->At(3)->GetTitle(), "root://some.domain/path/to/foo.root/file.root");
   EXPECT_STREQ(files->At(3)->GetName(), "treename");

   EXPECT_STREQ(files->At(4)->GetTitle(), "root://some.domain/path/to/file.root");
   EXPECT_STREQ(files->At(4)->GetName(), "treename");

   EXPECT_STREQ(files->At(5)->GetTitle(), "root://some.domain//path/to/file.root");
   EXPECT_STREQ(files->At(5)->GetName(), "treename");

   EXPECT_STREQ(files->At(6)->GetTitle(), "https://some.domain:8443/path/to/file.root.1");
   EXPECT_STREQ(files->At(6)->GetName(), "treename");

   EXPECT_STREQ(files->At(7)->GetTitle(), "https://some.domain:8443/path/to/file.root.1#anchor");
   EXPECT_STREQ(files->At(7)->GetName(), "defaultname");

   EXPECT_STREQ(files->At(8)->GetTitle(), "https://some.domain:8443/path/to/file.root.1?a=b&x=y&");
   EXPECT_STREQ(files->At(8)->GetName(), "defaultname");

   EXPECT_STREQ(files->At(9)->GetTitle(), "https://some.domain:8443/path/to/file.root.1?a=b&x=y&");
   EXPECT_STREQ(files->At(9)->GetName(), "treename");

   EXPECT_STREQ(files->At(10)->GetTitle(), "https://some.domain:8443/path/to/file.root.1?a=b&x=y&#X=Y");
   EXPECT_STREQ(files->At(10)->GetName(), "defaultname");
}

TEST(TChainParsing, LocalAdd)
{
   TChain c;
   c.Add("/path/to/file.root");
   c.Add("/path/to/file.root/foo");
   c.Add("/path/to/file.root/foo/bar");
   c.Add("/path/to/file.root/foo.bar/treename");
   c.Add("/path/to/file.root/foo.root/treename");
   c.Add("/path/to/file.root/root/treename");
   c.Add("path/to/file.root/treename");
   c.Add("/path/to/file.root//treename");
   const auto files = c.GetListOfFiles();

   EXPECT_STREQ(files->At(0)->GetTitle(), "/path/to/file.root");
   EXPECT_STREQ(files->At(0)->GetName(), "");

   EXPECT_STREQ(files->At(1)->GetTitle(), "/path/to/file.root");
   EXPECT_STREQ(files->At(1)->GetName(), "foo");

   EXPECT_STREQ(files->At(2)->GetTitle(), "/path/to/file.root");
   EXPECT_STREQ(files->At(2)->GetName(), "foo/bar");

   EXPECT_STREQ(files->At(3)->GetTitle(), "/path/to/file.root");
   EXPECT_STREQ(files->At(3)->GetName(), "foo.bar/treename");

   EXPECT_STREQ(files->At(4)->GetTitle(), "/path/to/file.root/foo.root");
   EXPECT_STREQ(files->At(4)->GetName(), "treename");

   EXPECT_STREQ(files->At(5)->GetTitle(), "/path/to/file.root");
   EXPECT_STREQ(files->At(5)->GetName(), "root/treename");

   EXPECT_STREQ(files->At(6)->GetTitle(), "path/to/file.root");
   EXPECT_STREQ(files->At(6)->GetName(), "treename");

   EXPECT_STREQ(files->At(7)->GetTitle(), "/path/to/file.root");
   EXPECT_STREQ(files->At(7)->GetName(), "/treename");
}

void FillTree(const char *filename, const char *treeName)
{
   TFile f{filename, "RECREATE"};
   if (f.IsZombie()) {
      throw std::runtime_error("Could not create file for the test!");
   }
   TTree t{treeName, treeName};

   int b;
   t.Branch("b1", &b);

   for (int i = 0; i < 5; ++i) {
      b = i;
      t.Fill();
   }

   const auto writtenBytes{t.Write()};
   if (writtenBytes == 0) {
      throw std::runtime_error("Could not write a tree for the test!");
   }
   f.Close();
}

TEST(TChainParsing, GlobbingWithTreenameToken)
{
   // Mix files with/without .root extension
   // to cover both cases.
   std::vector<std::string> fileNames{"tchain_globbingwithtreenametoken_0", "tchain_globbingwithtreenametoken_00",
                                      "tchain_globbingwithtreenametoken_000.root",
                                      "tchain_globbingwithtreenametoken_0000",
                                      "tchain_globbingwithtreenametoken_00000.root"};
   for (const auto &fileName : fileNames) {
      FillTree(fileName.c_str(), "events");
   }
   TChain c;
   c.Add("tchain_globbingwithtreenametoken_0*?#events");

   const auto *chainFiles = c.GetListOfFiles();
   ASSERT_TRUE(chainFiles);
   EXPECT_EQ(chainFiles->GetEntries(), 5);

   const auto *cwd = gSystem->WorkingDirectory();
   for (std::size_t i = 0; i < 5; i++) {
      const std::string fileName = chainFiles->At(i)->GetTitle();
      const std::string treeName = chainFiles->At(i)->GetName();
      EXPECT_EQ(treeName, "events");
      const auto normalizedPath = ConcatUnixFileName(cwd, fileNames[i].c_str());
      EXPECT_EQ(fileName, normalizedPath);
   }

   for (const auto &fileName : fileNames) {
      gSystem->Unlink(fileName.c_str());
   }
}

TEST(TChainParsing, GlobbingWithNonExistingDir)
{
   // Check that TChain::Add doesn't throw and an empty list of files is created
   TChain c;
   c.Add("nonexistingpath/nonexistingfile*");

   const auto *chainFiles = c.GetListOfFiles();
   ASSERT_TRUE(chainFiles);
   EXPECT_EQ(chainFiles->GetEntries(), 0);

   // Check that the equivalent call to the ExpandGlob function throws
   try {
      const auto expanded_glob = ROOT::Internal::TreeUtils::ExpandGlob("nonexistingpath/nonexistingfile*");
   } catch (const std::runtime_error &err) {
      std::string msg{"ExpandGlob: could not open directory 'nonexistingpath'."};
      EXPECT_EQ(msg, err.what());
   }
}

std::vector<std::string> GetFileNamesVec(const TObjArray *chainFiles)
{
   std::vector<std::string> chainFileNames;
   chainFileNames.reserve(chainFiles->GetEntries());
   for (const auto *obj : *chainFiles) {
      chainFileNames.push_back(obj->GetTitle());
   }
   return chainFileNames;
}

void MakeDirIfNotExist(const char *dir)
{
   // Note that AccessPathName returns FALSE if the dir DOES exist
   if (gSystem->AccessPathName(dir, kFileExists)) {
      ASSERT_EQ(gSystem->mkdir(dir), 0);
   }
}

TEST(TChainParsing, RecursiveGlob)
{
   // Need to change working directory to ensure "*" only adds files we create here.
   const auto *testDir = "testdir";
   MakeDirIfNotExist(testDir);
   gSystem->ChangeDirectory(testDir);

   const auto *cwd = gSystem->WorkingDirectory();

   MakeDirIfNotExist("testglob");
   MakeDirIfNotExist("testglob/subdir1");
   MakeDirIfNotExist("testglob/subdir1/subdir11");
   MakeDirIfNotExist("testglob/subdir2");
   MakeDirIfNotExist("testglob/subdir2/subdir21");
   MakeDirIfNotExist("testglob/subdir3");

   // Unsorted to also test sorting the final list of globbed files.
   std::vector<std::string> fileNames{"0a.root",
                                      "testglob/subdir1/1a.root",
                                      "testglob/subdir1/1b.root",
                                      "testglob/subdir3/3a.root",
                                      "testglob/subdir2/subdir21/21a.root",
                                      "testglob/subdir2/subdir21/21b.root",
                                      "testglob/subdir1/subdir11/11b.root",
                                      "testglob/subdir1/subdir11/11a.root"};
   for (const auto &fileName : fileNames) {
      FillTree(fileName.c_str(), "events");
   }

   TChain nodir;
   TChain none;
   TChain nested;
   TChain globDir;
   TChain regex;

   nodir.Add("*");
   const auto *nodirFiles = nodir.GetListOfFiles();
   ASSERT_TRUE(nodirFiles);
   EXPECT_EQ(nodirFiles->GetEntries(), 1);
   auto expectedFileNameNodir = ConcatUnixFileName(cwd, "0a.root");
   EXPECT_EQ(std::string{nodirFiles->At(0)->GetTitle()}, expectedFileNameNodir);

   none.Add("testglob/*");
   const auto *noneChainFiles = none.GetListOfFiles();
   ASSERT_TRUE(noneChainFiles);
   EXPECT_EQ(noneChainFiles->GetEntries(), 0);

   nested.Add("testglob/*/*/*.root");
   const auto *nestedChainFiles = nested.GetListOfFiles();
   ASSERT_TRUE(nestedChainFiles);
   EXPECT_EQ(nestedChainFiles->GetEntries(), 4);
   std::vector<std::string> expectedFileNamesNested{
      "testglob/subdir1/subdir11/11a.root", "testglob/subdir1/subdir11/11b.root", "testglob/subdir2/subdir21/21a.root",
      "testglob/subdir2/subdir21/21b.root"};
   auto nestedChainFileNames = GetFileNamesVec(nestedChainFiles);
   EXPECT_VEC_EQ(nestedChainFileNames, expectedFileNamesNested);

   globDir.Add("*/subdir[0-9]/*");
   const auto *globDirChainFiles = globDir.GetListOfFiles();
   ASSERT_TRUE(globDirChainFiles);
   EXPECT_EQ(globDirChainFiles->GetEntries(), 3);
   std::vector<std::string> expectedFileNamesGlobDir{ConcatUnixFileName(cwd, "testglob/subdir1/1a.root"),
                                                     ConcatUnixFileName(cwd, "testglob/subdir1/1b.root"),
                                                     ConcatUnixFileName(cwd, "testglob/subdir3/3a.root")};
   auto globDirChainFileNames = GetFileNamesVec(globDirChainFiles);
   for (std::size_t i = 0; i < expectedFileNamesGlobDir.size(); i++) {
      EXPECT_EQ(globDirChainFileNames[i], expectedFileNamesGlobDir[i]);
   }

   regex.Add("test*/subdir?/[0-9]?.root?#events");
   const auto *regexChainFiles = regex.GetListOfFiles();
   ASSERT_TRUE(regexChainFiles);
   EXPECT_EQ(regexChainFiles->GetEntries(), 3);
   std::vector<std::string> expectedFileNamesRegex{ConcatUnixFileName(cwd, "testglob/subdir1/1a.root"),
                                                   ConcatUnixFileName(cwd, "testglob/subdir1/1b.root"),
                                                   ConcatUnixFileName(cwd, "testglob/subdir3/3a.root")};
   auto regexChainFileNames = GetFileNamesVec(regexChainFiles);
   EXPECT_VEC_EQ(regexChainFileNames, expectedFileNamesRegex);
}

#if !defined(_MSC_VER) || defined(R__ENABLE_BROKEN_WIN_TESTS)
// No XRootD support on Windows
TEST(TChainParsing, RemoteGlob)
{

   // Check if xrootd is enabled.
   if (nullptr == TClass::GetClass(gROOT->GetPluginManager()->FindHandler("TFile", "root://")->GetClass()))
   {
      GTEST_SKIP();
   }

   TChain c;
   c.Add("root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run*");
   const auto *chainFiles = c.GetListOfFiles();

   ASSERT_TRUE(chainFiles);
   EXPECT_EQ(chainFiles->GetEntries(), 4);

   std::vector<std::string> expectedFileNames{
      "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleElectron.root",
      "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012B_DoubleMuParked.root",
      "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012C_DoubleElectron.root",
      "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/Run2012C_DoubleMuParked.root"};

   auto chainFileNames = GetFileNamesVec(chainFiles);
   EXPECT_VEC_EQ(chainFileNames, expectedFileNames);
}

TEST(TChainParsing, DoubleSlash)
{
   // Check if xrootd is enabled.
   if (nullptr == TClass::GetClass(gROOT->GetPluginManager()->FindHandler("TFile", "root://")->GetClass()))
   {
      GTEST_SKIP();
   }
   // Tests #7159
   TChain c("Events");
   c.Add("root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod//ZZTo2e2mu.root");
   EXPECT_EQ(c.GetEntries(), 1497445);
}

#endif
