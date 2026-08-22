#include <memory>
#include <vector>
#include <string>
#include <array>
#include <algorithm>
#include <fstream>

#include "gtest/gtest.h"

#include <ROOT/TestSupport.hxx>

#include "TFile.h"
#include "TMemFile.h"
#include "TDirectory.h"
#include "TKey.h"
#include "TNamed.h"
#include "TPluginManager.h"
#include "TROOT.h" // gROOT
#include "TSystem.h"
#include "TEnv.h" // gEnv
#include "TFree.h"
#include "TError.h"

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

namespace {
std::string gCollectedDiags;
void CollectDiags(int /*level*/, Bool_t /*abort*/, const char *location, const char *msg)
{
   gCollectedDiags += location;
   gCollectedDiags += ": ";
   gCollectedDiags += msg;
   gCollectedDiags += '\n';
}
} // namespace

TEST(TFile, ReadKeysValid)
{
   ROOT::TestSupport::FileRaii fileGuard("tfile_readkeys_valid.root");
   {
      TFile f(fileGuard.GetPath().c_str(), "RECREATE");
      TNamed named("short", "t");
      named.Write();
   }
   TFile in(fileGuard.GetPath().c_str());
   ASSERT_FALSE(in.IsZombie());
   EXPECT_EQ(in.GetNkeys(), 1);
   auto *named = in.Get<TNamed>("short");
   ASSERT_NE(named, nullptr);
   EXPECT_STREQ(named->GetTitle(), "t");
}

TEST(TFile, ReadKeysOversizedString)
{
   ROOT::TestSupport::FileRaii fileGuard("tfile_readkeys_oversize.root");
   Long64_t seekKeys = 0;
   Int_t nbytesKeys = 0;
   {
      TFile f(fileGuard.GetPath().c_str(), "RECREATE");
      TNamed named("short", "t");
      named.Write();
      f.Write();
      seekKeys = f.GetSeekKeys();
      nbytesKeys = f.GetNbytesKeys();
   }
   ASSERT_GT(seekKeys, 0);
   ASSERT_GT(nbytesKeys, 0);

   {
      std::fstream fs(fileGuard.GetPath(), std::ios::in | std::ios::out | std::ios::binary);
      ASSERT_TRUE(fs.good());
      std::vector<char> rec(static_cast<std::size_t>(nbytesKeys));
      fs.seekg(seekKeys);
      fs.read(rec.data(), nbytesKeys);
      ASSERT_EQ(fs.gcount(), nbytesKeys);

      const char needle[] = {'\x05', 's', 'h', 'o', 'r', 't'};
      auto it = std::search(rec.begin(), rec.end(), std::begin(needle), std::end(needle));
      ASSERT_NE(it, rec.end());
      *it = static_cast<char>(255);
      fs.seekp(seekKeys);
      fs.write(rec.data(), nbytesKeys);
      ASSERT_TRUE(fs.good());
   }

   gCollectedDiags.clear();
   {
      ROOT::TestSupport::FilterDiagsRAII capture(CollectDiags);
      TFile in(fileGuard.GetPath().c_str());
      // Opening must return; do not walk off the keys buffer.
      EXPECT_TRUE(in.IsZombie() || in.GetNkeys() >= 0);
   }
   EXPECT_NE(gCollectedDiags.find("given buffer is too small"), std::string::npos);
}

TEST(TFile, ReadFreeValid)
{
   ROOT::TestSupport::FileRaii fileGuard("tfile_readfree_valid.root");
   {
      TFile f(fileGuard.GetPath().c_str(), "RECREATE");
      TNamed named("n", "t");
      named.Write();
   }
   {
      TFile f(fileGuard.GetPath().c_str(), "UPDATE");
      ASSERT_FALSE(f.IsZombie());
      TNamed named2("n2", "t");
      named2.Write();
   }
   TFile in(fileGuard.GetPath().c_str());
   ASSERT_FALSE(in.IsZombie());
   EXPECT_NE(in.Get<TNamed>("n"), nullptr);
   EXPECT_NE(in.Get<TNamed>("n2"), nullptr);
}

TEST(TFree, ReadBufferBounds)
{
   char packed[10] = {};
   char *p = packed;
   TFree out;
   out.SetFirst(100);
   out.SetLast(200);
   out.FillBuffer(p);
   ASSERT_EQ(p - packed, 10);

   p = packed;
   TFree in;
   EXPECT_TRUE(in.ReadBuffer(p, sizeof(packed)));
   EXPECT_EQ(in.GetFirst(), 100);
   EXPECT_EQ(in.GetLast(), 200);

   char tooSmall[3] = {};
   p = tooSmall;
   TFree truncated;
   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.requiredDiag(kError, "TFree::ReadBuffer", "The given buffer is too small", false);
   EXPECT_FALSE(truncated.ReadBuffer(p, sizeof(tooSmall)));
}
