#include <string>
#include <vector>

#include "TChain.h"
#include "TFile.h"
#include "TSystem.h"
#include "TTree.h"

#include "gtest/gtest.h"

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
      const auto *fileName = chainFiles->At(i)->GetTitle();
      const auto *treeName = chainFiles->At(i)->GetName();
      EXPECT_STREQ(treeName, "events");
      const auto *fullPathToFile = gSystem->ConcatFileName(cwd, fileNames[i].c_str());
      const auto *normalizedPath = gSystem->UnixPathName(fullPathToFile);
      EXPECT_STREQ(fileName, normalizedPath);
      // The docs of `ConcatFileName` tell us we should delete the string
      delete[] fullPathToFile;
   }

   for (const auto &fileName : fileNames) {
      gSystem->Unlink(fileName.c_str());
   }
}
