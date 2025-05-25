#include <memory>
#include <vector>
#include <string>

#include "gtest/gtest.h"

#include "TFile.h"
#include "TMemFile.h"
#include "TDirectory.h"
#include "TKey.h"
#include "TNamed.h"
#include "TPluginManager.h"
#include "TROOT.h" // gROOT
#include "TSystem.h"
#include "TEnv.h" // gEnv

TEST(TFile, WriteObjectTObject)
{
    auto filename{"tfile_writeobject_tobject.root"};
    auto tnamed_name{"mytnamed_name"};
    auto tnamed_title{"mytnamed_title"};

    {
        TNamed mytnamed{tnamed_name, tnamed_title};
        TFile f{filename, "recreate"};
        f.WriteObject(&mytnamed, mytnamed.GetName());
        f.Close();
    }

    TFile input{filename};
    auto named = input.Get<TNamed>(tnamed_name);
    auto keyptr = static_cast<TKey *>(input.GetListOfKeys()->At(0));

    EXPECT_STREQ(named->GetName(), tnamed_name);
    EXPECT_STREQ(named->GetTitle(), tnamed_title);
    EXPECT_STREQ(keyptr->GetName(), tnamed_name);
    EXPECT_STREQ(keyptr->GetTitle(), tnamed_title);

    input.Close();
    gSystem->Unlink(filename);
}

TEST(TFile, WriteObjectVector)
{
    auto filename{"tfile_writeobject_vector.root"};
    auto vec_name{"object name"}; // Decided arbitrarily

    {
        std::vector<int> myvec{1,2,3,4,5};
        TFile f{filename, "recreate"};
        f.WriteObject(&myvec, vec_name);
        f.Close();
    }

    TFile input{filename};
    auto retvecptr = input.Get<std::vector<int>>(vec_name);
    const auto &retvec = *retvecptr;
    auto retkey = static_cast<TKey *>(input.GetListOfKeys()->At(0));

    std::vector<int> expected{1,2,3,4,5};

    ASSERT_EQ(retvec.size(), expected.size());
    for (std::size_t i = 0; i < retvec.size(); ++i) {
        EXPECT_EQ(retvec[i], expected[i]);
    }

    EXPECT_STREQ(retkey->GetName(), vec_name);
    EXPECT_STREQ(retkey->GetTitle(), ""); // Objects that don't derive from TObject have no title

    input.Close();
    gSystem->Unlink(filename);
}

// Tests ROOT-9857
TEST(TFile, ReadFromSameFile)
{
   const auto filename = "ReadFromSameFile.root";
   const auto objname = "foo";
   const auto objpath = "./ReadFromSameFile.root/foo";
   {
      TFile f(filename, "RECREATE");
      TObject obj;
      f.WriteObject(&obj, objname);
   }

   TFile f1(filename);
   auto o1 = f1.Get(objname);

   TFile f2(filename);
   auto o2 = f2.Get(objpath);

   EXPECT_TRUE(o1 != o2) << "Same objects read from two different files have the same pointer!";
}

TEST(TFile, ReadWithoutGlobalRegistrationLocal)
{
   const auto localFile = "TFileTestReadWithoutGlobalRegistrationLocal.root";

   // create local input file
   {
      std::unique_ptr<TFile> input{TFile::Open(localFile, "RECREATE")};
      ASSERT_TRUE(input != nullptr);
      ASSERT_FALSE(input->IsZombie());
   }

   // test that with READ_WITHOUT_GLOBALREGISTRATION the file does not end up in the global list of files
   std::unique_ptr<TFile> f{TFile::Open(localFile, "READ_WITHOUT_GLOBALREGISTRATION")};
   EXPECT_TRUE(f != nullptr);
   EXPECT_FALSE(f->IsZombie());
   EXPECT_TRUE(gROOT->GetListOfFiles()->FindObject(localFile) == nullptr);

   gSystem->Unlink(localFile);
}

void TestReadWithoutGlobalRegistrationIfPossible(const char *fname)
{
   TPluginHandler *h;
   if ((h = gROOT->GetPluginManager()->FindHandler("TFile", fname))) {
      if (h->LoadPlugin() == -1)
         return;
   }

   // test that with READ_WITHOUT_GLOBALREGISTRATION the file does not end up in the global list of files
   std::unique_ptr<TFile> f{TFile::Open(fname, "READ_WITHOUT_GLOBALREGISTRATION")};
   EXPECT_TRUE(f != nullptr);
   EXPECT_FALSE(f->IsZombie());
   EXPECT_TRUE(gROOT->GetListOfFiles()->FindObject(fname) == nullptr);
}

// https://github.com/root-project/root/issues/10742
#ifndef R__WIN32
// We prefer not to read remotely files from Windows, if possible
TEST(TFile, ReadWithoutGlobalRegistrationWeb)
{
   const auto webFile = "http://root.cern/files/h1/dstarmb.root";
   TestReadWithoutGlobalRegistrationIfPossible(webFile);
}
TEST(TFile, ReadWithoutGlobalRegistrationNet)
{
   const auto netFile = "root://eospublic.cern.ch//eos/root-eos/h1/dstarmb.root";
   TestReadWithoutGlobalRegistrationIfPossible(netFile);
}
#endif 

// https://github.com/root-project/root/issues/16189
TEST(TFile, k630forwardCompatibility)
{
   gEnv->SetValue("TFile.v630forwardCompatibility", 1);
   const std::string filename{"filek30.root"};
   // Testing that the flag is also set when creating the file from scratch (as opposed to "UPDATE")
   TFile filec{filename.c_str(),"RECREATE"};
   ASSERT_EQ(filec.TestBit(TFile::k630forwardCompatibility), true);  
   filec.Close();
   TFile filer{filename.c_str(),"READ"};
   ASSERT_EQ(filer.TestBit(TFile::k630forwardCompatibility), true);  
   filer.Close();
   TFile fileu{filename.c_str(),"UPDATE"};
   ASSERT_EQ(fileu.TestBit(TFile::k630forwardCompatibility), true);  
   fileu.Close();
   gSystem->Unlink(filename.c_str());
}

// https://github.com/root-project/root/issues/17824
TEST(TFile, MakeSubDirectory)
{
   // create test file
   TMemFile outFile("dirTest17824.root", "RECREATE");
   // create test dir
   auto d = outFile.mkdir("test");
   // check if returned pointer points to test dir
   EXPECT_EQ(std::string(d->GetName()), "test");
   // move to dir and check
   d->cd();
   EXPECT_EQ(std::string(gDirectory->GetPath()), "dirTest17824.root:/test");
   EXPECT_EQ(std::string(gDirectory->GetName()), "test");

   // make test2 subdir
   auto d2 = outFile.mkdir("test/test2");
   // check if returned pointer points to test2 subdir
   EXPECT_NE(d2, d);
   EXPECT_EQ(std::string(d2->GetName()), "test2");
   // move to test2 subdir
   d2->cd();
   EXPECT_EQ(d2, gDirectory);
   EXPECT_EQ(std::string(gDirectory->GetPath()), "dirTest17824.root:/test/test2");
   EXPECT_EQ(std::string(gDirectory->GetName()), "test2");
   // rebase (because paths in cd() are relative) and move to test2 subdir via gDirectory and explicit path
   outFile.cd();
   gDirectory->cd("test/test2");
   // check location again
   EXPECT_EQ(d2, gDirectory);
   EXPECT_EQ(std::string(gDirectory->GetPath()), "dirTest17824.root:/test/test2");
   EXPECT_EQ(std::string(gDirectory->GetName()), "test2");
   // test now three-level as in the doxygen docu
   outFile.cd();
   auto c = outFile.mkdir("a/b/c");
   EXPECT_EQ(std::string(c->GetPath()), "dirTest17824.root:/a/b/c");
   EXPECT_EQ(std::string(c->GetName()), "c");
   gDirectory->cd("a/b/c");
   EXPECT_EQ(c, gDirectory);
   EXPECT_EQ(std::string(gDirectory->GetPath()), "dirTest17824.root:/a/b/c");
   EXPECT_EQ(std::string(gDirectory->GetName()), "c");
}
