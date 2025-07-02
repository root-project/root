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

#include <cstdio>

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
      std::vector<int> myvec{1, 2, 3, 4, 5};
      TFile f{filename, "recreate"};
      f.WriteObject(&myvec, vec_name);
      f.Close();
   }

   TFile input{filename};
   auto retvecptr = input.Get<std::vector<int>>(vec_name);
   const auto &retvec = *retvecptr;
   auto retkey = static_cast<TKey *>(input.GetListOfKeys()->At(0));

   std::vector<int> expected{1, 2, 3, 4, 5};

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
#ifdef R__HAS_DAVIX
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
#endif

// https://github.com/root-project/root/issues/16189
TEST(TFile, k630forwardCompatibility)
{
   gEnv->SetValue("TFile.v630forwardCompatibility", 1);
   const std::string filename{"filek30.root"};
   // Testing that the flag is also set when creating the file from scratch (as opposed to "UPDATE")
   TFile filec{filename.c_str(), "RECREATE"};
   ASSERT_EQ(filec.TestBit(TFile::k630forwardCompatibility), true);
   filec.Close();
   TFile filer{filename.c_str(), "READ"};
   ASSERT_EQ(filer.TestBit(TFile::k630forwardCompatibility), true);
   filer.Close();
   TFile fileu{filename.c_str(), "UPDATE"};
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

TEST(TFile, WalkTKeys)
{
   struct FileRaii {
      std::string fFilename;
      FileRaii(std::string_view fname) : fFilename(fname) {}
      ~FileRaii() { gSystem->Unlink(fFilename.c_str()); }
   } fileGuard("tfile_walk_tkeys.root");

   TFile outFile(fileGuard.fFilename.c_str(), "RECREATE");

   std::string foo = "foo";
   outFile.WriteObject(&foo, "foo");

   // Write an object with an extremely long name (> 128 chars but < 256)
   static const char kLongKey[] = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                                  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                                  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
   static_assert(std::size(kLongKey) > 128);
   static_assert(std::size(kLongKey) < 256);
   outFile.WriteObject(&foo, kLongKey);

   // Write an object with an even longer name (> 256 chars)
   static const char kLongerKey[] = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                                    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                                    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                                    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                                    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                                    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
   static_assert(std::size(kLongerKey) > 256);
   outFile.WriteObject(&foo, kLongerKey);
   outFile.Close();

   TFile inFile(fileGuard.fFilename.c_str(), "READ");
   auto keys = inFile.WalkTKeys();
   auto it = keys.begin();
   EXPECT_EQ(it->fKeyName, "tfile_walk_tkeys.root");
   EXPECT_EQ(it->fClassName, "TFile");
   ++it;
   EXPECT_EQ(it->fKeyName, "foo");
   EXPECT_EQ(it->fClassName, "string");
   ++it;
   EXPECT_EQ(it->fKeyName, kLongKey);
   EXPECT_EQ(it->fClassName, "string");
   ++it;
   EXPECT_EQ(it->fKeyName, kLongerKey);
   EXPECT_EQ(it->fClassName, "string");
}

TEST(TFile, DeleteKey)
{
   struct FileRaii {
      std::string fFilename;
      FileRaii(std::string_view fname) : fFilename(fname) {}
      ~FileRaii() { gSystem->Unlink(fFilename.c_str()); }
   } fileGuard("tfile_test_delete_keys.root");

   auto fnCountGaps = [](const std::string &fileName) {
      auto f = std::unique_ptr<TFile>(TFile::Open(fileName.c_str()));
      std::uint64_t nGaps = 0;
      for (const auto &k : f->WalkTKeys()) {
         if (k.fType == ROOT::Detail::TKeyMapNode::kGap)
            nGaps++;
      }
      return nGaps;
   };

   auto f = std::unique_ptr<TFile>(TFile::Open(fileGuard.fFilename.c_str(), "RECREATE"));
   f->SetCompressionSettings(0);
   f->Write();
   f->Close();

   // The empty file should have no gaps. Note that gaps are created temporarily when certain keys are overwritten.
   EXPECT_EQ(0, fnCountGaps(fileGuard.fFilename));

   f = std::unique_ptr<TFile>(TFile::Open(fileGuard.fFilename.c_str(), "UPDATE"));
   std::vector<char> v;
   f->WriteObject(&v, "va0");
   f->WriteObject(&v, "va1");
   f->WriteObject(&v, "va2");
   f->Write();
   f->Close();
   // 2 gaps: new (larger) keys list and free list are written
   EXPECT_EQ(2, fnCountGaps(fileGuard.fFilename));

   f = std::unique_ptr<TFile>(TFile::Open(fileGuard.fFilename.c_str(), "UPDATE"));
   f->Delete("va1;*"); // should create small gap that cannot be merged, trapped between v0 and v2
   f->Write();
   f->Close();

   EXPECT_EQ(3, fnCountGaps(fileGuard.fFilename));

   f = std::unique_ptr<TFile>(TFile::Open(fileGuard.fFilename.c_str(), "UPDATE"));
   f->Delete("va2;*"); // gaps at the tail should merge
   f->Write();
   f->Close();

   EXPECT_EQ(2, fnCountGaps(fileGuard.fFilename));

   // The following tests run out of memory on 32bit platforms
   if (sizeof(std::size_t) == 4) {
      printf("Skipping test partially on 32bit platform.\n");
      return;
   }

   v.resize(1000 * 1000 * 1000 - 100, 'x'); // almost 1GB
   f = std::unique_ptr<TFile>(TFile::Open(fileGuard.fFilename.c_str(), "UPDATE"));
   f->WriteObject(&v, "vb0");
   f->WriteObject(&v, "vb1");
   v.resize(1000 * 1000); // truncate next objects to 1MB
   f->WriteObject(&v, "vc0");
   f->WriteObject(&v, "vc1");
   f->WriteObject(&v, "vc2");
   f->WriteObject(&v, "vc3");
   f->Write();
   EXPECT_GT(f->GetEND(), TFile::kStartBigFile);
   f->Close();

   // New keys list, hence 3 gaps
   EXPECT_EQ(3, fnCountGaps(fileGuard.fFilename));

   f = std::unique_ptr<TFile>(TFile::Open(fileGuard.fFilename.c_str(), "UPDATE"));
   f->Delete("vb0;*"); //  |
   f->Delete("vb1;*"); //  |---> First merged gap
   f->Delete("vc0;*"); //  |
   f->Delete("vc1;*"); //  |---> Second merged gap
   f->Write();
   f->Close();

   // Two merged gaps, the new keys list fits into either of them
   EXPECT_EQ(4, fnCountGaps(fileGuard.fFilename));

   f = std::unique_ptr<TFile>(TFile::Open(fileGuard.fFilename.c_str(), "UPDATE"));
   // Delete the remaining data at the tail of the file in reverse order
   f->Delete("vc2;*");
   f->Delete("vc3;*");
   f->Write();
   // Back to small file
   EXPECT_LT(f->GetEND(), TFile::kStartBigFile);
   f->Close();

   // Only the original 2 gaps from the first keys list and free list overwrite
   EXPECT_EQ(2, fnCountGaps(fileGuard.fFilename));

   f = std::unique_ptr<TFile>(TFile::Open(fileGuard.fFilename.c_str(), "UPDATE"));
   v.resize(700 * 1000 * 1000); // construct objects such that 3 consecutive gaps surpass 2GB (but not 2)
   f->WriteObject(&v, "vd0");
   f->WriteObject(&v, "vd1");
   f->WriteObject(&v, "vd2");
   f->WriteObject(&v, "vd3");
   f->WriteObject(&v, "vd4");
   f->Write();
   f->Close();

   // New keys list --> 3 gaps
   EXPECT_EQ(3, fnCountGaps(fileGuard.fFilename));

   f = std::unique_ptr<TFile>(TFile::Open(fileGuard.fFilename.c_str(), "UPDATE"));
   f->Delete("vd1;*");
   f->Delete("vd3;*");
   f->Write();
   f->Close();

   // Nothing mergable, 2 more gaps
   EXPECT_EQ(5, fnCountGaps(fileGuard.fFilename));

   f = std::unique_ptr<TFile>(TFile::Open(fileGuard.fFilename.c_str(), "UPDATE"));
   auto theEnd = f->GetEND();
   f->Delete("vd2;*");
   f->Write();
   f->Close();

   // We can only merge the gaps of v1 and v2, not all three (vd1, vd2, vd3) due to the gap size
   EXPECT_EQ(5, fnCountGaps(fileGuard.fFilename));

   // Ensure that the file's tail is still intact
   f = std::unique_ptr<TFile>(TFile::Open(fileGuard.fFilename.c_str()));
   EXPECT_EQ(f->GetEND(), theEnd);
}
