#include <memory>
#include <vector>
#include <string>
#include <array>
#include <stdexcept>

#include "gtest/gtest.h"

#include <ROOT/TestSupport.hxx>

#include "TDirectory.h"
#include "TEnv.h"
#include "TFile.h"
#include "TFree.h"
#include "TKey.h"
#include "TMemFile.h"
#include "TNamed.h"
#include "TPluginManager.h"
#include "TROOT.h"
#include "TSystem.h"

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
#if defined(R__HAS_DAVIX) || defined(R__HAS_CURL)
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
TEST(TFile, ReadWithCacheWithoutGlobalRegistration)
{
   const auto webFile = "http://root.cern/files/h1/dstarmb.root";
   TFile::SetCacheFileDir(".");
   delete TFile::Open(webFile, "READ_WITHOUT_GLOBALREGISTRATION");
   EXPECT_TRUE(gSystem->AccessPathName("./files/h1/dstarmb.root"));
   TFile::SetCacheFileDir("");
   gSystem->Unlink("./files");
}
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

// https://its.cern.ch/jira/browse/ROOT-10352
TEST(TDirectoryFile, SeekParent)
{
   // create test file
   TMemFile f("subdirTest10352.root", "RECREATE");
   auto dir1 = f.mkdir("dir-1");
   dir1->cd();
   auto dir11 = dir1->mkdir("dir-11");
   dir11->cd();
   f.Write();
   dir1 = static_cast<TDirectory*>(f.Get("dir-1"));
   EXPECT_EQ(dir1->GetSeekDir(), 239);
   EXPECT_EQ(dir1->GetSeekParent(), 100);
   dir11 = static_cast<TDirectory*>(dir1->Get("dir-11"));
   EXPECT_EQ(dir11->GetSeekDir(), 348);
   EXPECT_EQ(dir11->GetSeekParent(), 239);
}

TEST(TDirectoryFile, RecursiveMkdir)
{
   TMemFile f("mkdirtest.root", "RECREATE");
   auto dir1 = f.mkdir("a/b/c", "my dir");
   EXPECT_NE(dir1, nullptr);
   {
      ROOT::TestSupport::CheckDiagsRAII diags;
      diags.requiredDiag(kError, "TDirectoryFile::mkdir","An object with name c exists already");
      auto dir2 = f.mkdir("a/b/c", "", /* returnExisting = */ false);
      EXPECT_EQ(dir2, nullptr);
   }
   auto dir3 = f.mkdir("a/b/c", "foobar", /* returnExisting = */ true);
   EXPECT_EQ(dir3, dir1);
   EXPECT_STREQ(dir3->GetTitle(), "my dir");
   auto dirB = dir3->GetMotherDir();
   ASSERT_NE(dirB, nullptr);
   EXPECT_STREQ(dirB->GetTitle(), "b");
   auto dirA = dirB->GetMotherDir();
   ASSERT_NE(dirA, nullptr);
   EXPECT_STREQ(dirA->GetTitle(), "a");
}

// https://its.cern.ch/jira/browse/ROOT-10581
TEST(TFile, PersistTObjectStdArray)
{
   auto filename = "foo10581.root";
   {
      std::array<TObject *, 2> arr;
      arr[0] = new TObject();
      arr[0]->SetUniqueID(123);
      arr[1] = new TObject();
      arr[1]->SetUniqueID(456);
      TFile f(filename, "RECREATE");
      f.WriteObject(&arr, "array");
      f.Close();
      delete arr[0];
      delete arr[1];
   }
   {
      TFile ff(filename, "READ");
      std::array<TObject *, 2> *arr2 = nullptr;
      ff.GetObject("array", arr2);
      EXPECT_EQ((*arr2)[0]->GetUniqueID(), 123);
      EXPECT_EQ((*arr2)[1]->GetUniqueID(), 456);
   }
   gSystem->Unlink(filename);
}

TEST(TFile, UUID)
{
   TMemFile f("uuidtest.root", "RECREATE");
   EXPECT_EQ('4', f.GetUUID().AsString()[14]);
}

TEST(TFile, DeleteKey)
{
   ROOT::TestSupport::FileRaii fileGuard("tfile_test_delete_keys.root");

   auto fnCountGaps = [](const std::string &fileName) -> std::uint64_t {
      auto f = std::unique_ptr<TFile>(TFile::Open(fileName.c_str()));
      std::uint64_t nGaps = 0;
      for (const auto &k : f->WalkTKeys()) {
         if (k.fLen == TFile::kMaxGapSize) {
            // this used to indicate a truncated free segment (corrupt segment list). Gaps could still exactly
            // be 2GB in size but not for the files in this unit test.
            throw std::runtime_error("truncated free segment");
         }
         if (k.fType == ROOT::Detail::TKeyMapNode::kGap)
            nGaps++;
      }
      return nGaps;
   };

   auto f = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   f->SetCompressionSettings(0);
   f->Write();
   f->Close();

   // The empty file should have no gaps. Note that gaps are created temporarily when certain keys are overwritten.
   EXPECT_EQ(0, fnCountGaps(fileGuard.GetPath()));

   f = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "UPDATE"));
   std::vector<char> v;
   f->WriteObject(&v, "va0");
   f->WriteObject(&v, "va1");
   f->WriteObject(&v, "va2");
   f->Write();
   f->Close();
   // 2 gaps: new (larger) keys list and free list are written
   EXPECT_EQ(2, fnCountGaps(fileGuard.GetPath()));

   f = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "UPDATE"));
   f->Delete("va1;*"); // should create small gap that cannot be merged, trapped between v0 and v2
   f->Write();
   f->Close();

   EXPECT_EQ(3, fnCountGaps(fileGuard.GetPath()));

   f = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "UPDATE"));
   f->Delete("va2;*"); // gaps at the tail should merge
   f->Write();
   f->Close();

   EXPECT_EQ(2, fnCountGaps(fileGuard.GetPath()));

   // The following tests run out of memory on 32bit platforms
   if (sizeof(std::size_t) == 4) {
      printf("Skipping test partially on 32bit platform.\n");
      return;
   }

   v.resize(1000 * 1000, 'x'); // next few objects are 1MB in size
   f = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   f->SetCompressionSettings(0);
   f->WriteObject(&v, "vb0");
   f->WriteObject(&v, "vb1");
   f->WriteObject(&v, "vb2");
   f->WriteObject(&v, "vb3");
   v.resize(1000 * 1000 * 1000 - 100, 'x'); // almost 1GB
   f->WriteObject(&v, "vc0");
   f->WriteObject(&v, "vc1");
   f->WriteObject(&v, "vc2");
   f->Write();
   EXPECT_GT(f->GetEND(), TFile::kStartBigFile);
   f->Close();

   // New file, no gaps
   EXPECT_EQ(0, fnCountGaps(fileGuard.GetPath()));

   f = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "UPDATE"));
   f->Delete("vb1;*"); // Make a medium sized gap into which smaller objects fit, e.g. the free list
   f->Delete("vb3;*"); //  |
   f->Delete("vc0;*"); //  |
   f->Delete("vc1;*"); //  |---> Single merged gap in free list, multi-hop free segment on disk
   f->Write();
   f->Close();

   // Free list in gap created by vb1, one gap at the end because we have a smaller keys list. Two consecutive
   // gaps for removed vb3, vc0, vc1.
   EXPECT_EQ(4, fnCountGaps(fileGuard.GetPath()));
   f = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "UPDATE"));
   EXPECT_FALSE(f->TestBit(TFile::kRecovered));
   // Only 3 real gaps plus one virtual gap at the end of the file
   EXPECT_EQ(4, f->GetNfree());
   // Force the next open to recover the file
   f->GetListOfKeys()->Clear();
   f->Write();
   f->Close();

   {
      ROOT::TestSupport::CheckDiagsRAII diagsRaii;
      diagsRaii.requiredDiag(kInfo, "TFile::Recover", "recovered key vector<char>", false);
      f = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "UPDATE"));
      EXPECT_TRUE(f->TestBit(TFile::kRecovered));
      // We got one more free gap due to the replacement of the empty keys list.  Otherwise, we still want to see
      // that the large gap was merged from the smaller segments.
      EXPECT_EQ(5, f->GetNfree());
      bool foundLargeGap = false;
      for (const auto gap : ROOT::Detail::TRangeStaticCast<TFree>(f->GetListOfFree())) {
         if (gap->GetLast() - gap->GetFirst() >= TFile::kMaxGapSize) {
            foundLargeGap = true;
            break;
         }
      }
      EXPECT_TRUE(foundLargeGap);
      f->Write();
      f->Close();
   }
   // Same as before the recovery
   EXPECT_EQ(4, fnCountGaps(fileGuard.GetPath()));
}

TEST(TFile, KeySizeLimit)
{
   // The following tests run out of memory on 32bit platforms
   if (sizeof(std::size_t) == 4) {
      GTEST_SKIP() << "Skipping test on 32bit platform.";
   }

   ROOT::TestSupport::FileRaii fileGuard("tfile_test_key_size_limit.root");

   auto f = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   f->SetCompressionSettings(0);

   // Check that we can add keys >1GB (but smaller than 1GiB, obviously) in small and large files.
   // This does work even though the last, virtual free segment is 1GB (and not 1GiB). The reason it works is
   // that when the last free segment is not large enough, the code path that supports upgrading from a small file
   // to a large file is activated and extends the last free segment as needed.

   std::vector<char> v;
   v.resize(1000 * 1000 * 1000 + 100, 'x'); // more than 1GB but less the 1GiB
   f->WriteObject(&v, "v0");
   EXPECT_LT(f->GetEND(), TFile::kStartBigFile);
   f->WriteObject(&v, "v1");
   EXPECT_GT(f->GetEND(), TFile::kStartBigFile);
   f->Write();
   f->Close();

   f = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "UPDATE"));
   EXPECT_GT(f->GetEND(), TFile::kStartBigFile);
   f->WriteObject(&v, "v2");
   f->Write();
   f->Close();
}
