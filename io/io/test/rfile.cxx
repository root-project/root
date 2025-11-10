#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <TFile.h>
#include <TH1D.h>
#include <TRandom3.h>
#include <TROOT.h>
#include <TSystem.h>
#include <RZip.h>
#include <ROOT/RError.hxx>
#include <ROOT/RFile.hxx>
#include <ROOT/TestSupport.hxx>
#include <ROOT/RLogger.hxx>

using ROOT::Experimental::RFile;

namespace {

/**
 * An RAII wrapper around an open temporary file on disk. It cleans up the guarded file when the wrapper object
 * goes out of scope.
 */
class FileRaii {
private:
   std::string fPath;
   bool fPreserveFile = false;

public:
   explicit FileRaii(const std::string &path) : fPath(path) {}
   FileRaii(FileRaii &&) = default;
   FileRaii(const FileRaii &) = delete;
   FileRaii &operator=(FileRaii &&) = default;
   FileRaii &operator=(const FileRaii &) = delete;
   ~FileRaii()
   {
      if (!fPreserveFile)
         std::remove(fPath.c_str());
   }
   const std::string &GetPath() const { return fPath; }

   // Useful if you want to keep a test file after the test has finished running
   // for debugging purposes. Should only be used locally and never pushed.
   void PreserveFile() { fPreserveFile = true; }
};

} // anonymous namespace

static std::string JoinKeyNames(const ROOT::Experimental::RFileKeyIterable &iterable)
{
   auto beg = iterable.begin();
   if (beg == iterable.end())
      return std::string("");
   return std::accumulate(std::next(beg), iterable.end(), (*beg).GetPath(),
                          [](const auto &a, const auto &b) { return a + ", " + b.GetPath(); });
};

TEST(RFile, DecomposePath)
{
   using ROOT::Experimental::Detail::DecomposePath;

   auto Pair = [](std::string_view a, std::string_view b) { return std::make_pair(a, b); };

   EXPECT_EQ(DecomposePath("/foo/bar/baz"), Pair("/foo/bar/", "baz"));
   EXPECT_EQ(DecomposePath("/foo/bar/baz/"), Pair("/foo/bar/baz/", ""));
   EXPECT_EQ(DecomposePath("foo/bar/baz"), Pair("foo/bar/", "baz"));
   EXPECT_EQ(DecomposePath("foo"), Pair("", "foo"));
   EXPECT_EQ(DecomposePath("/"), Pair("/", ""));
   EXPECT_EQ(DecomposePath("////"), Pair("////", ""));
   EXPECT_EQ(DecomposePath(""), Pair("", ""));
   EXPECT_EQ(DecomposePath("asd/"), Pair("asd/", ""));
   EXPECT_EQ(DecomposePath("  "), Pair("", "  "));
   EXPECT_EQ(DecomposePath("/  "), Pair("/", "  "));
   EXPECT_EQ(DecomposePath("  /"), Pair("  /", ""));
}

TEST(RFile, Open)
{
   FileRaii fileGuard("test_rfile_read.root");

   // Create a root file to open
   {
      auto tfile = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      TH1D hist("hist", "", 100, -10, 10);
      hist.FillRandom("gaus", 1000);
      tfile->WriteObject(&hist, "hist");
   }

   auto file = RFile::Open(fileGuard.GetPath());
   auto hist = file->Get<TH1D>("hist");
   EXPECT_TRUE(hist);

   EXPECT_FALSE(file->Get<TH1D>("inexistent"));
   EXPECT_FALSE(file->Get<TH1F>("hist"));
   EXPECT_TRUE(file->Get<TH1>("hist"));

   // We do NOT want to globally register RFiles ever.
   EXPECT_EQ(ROOT::GetROOT()->GetListOfFiles()->GetSize(), 0);

   std::string foo = "foo";
   EXPECT_THROW(file->Put("foo", foo), ROOT::RException);
}

TEST(RFile, OpenInexistent)
{
   FileRaii fileGuard("does_not_exist.root");

   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.optionalDiag(kSysError, "TFile::TFile", "", false);
   diags.optionalDiag(kError, "TFile::TFile", "", false);
   
   try {
      auto f = RFile::Open("does_not_exist.root");
      FAIL() << "trying to open an inexistent file should throw";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("failed to open file"));
   }
   try {
      auto f = RFile::Update("/a/random/directory/that/definitely/does_not_exist.root");
      FAIL() << "trying to update a file under an inexistent directory should throw";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("failed to open file"));
   }
   try {
      auto f = RFile::Recreate("/a/random/directory/that/definitely/does_not_exist.root");
      FAIL() << "trying to create a file under an inexistent directory should throw";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("failed to open file"));
   }

   // This succeeds because Update creates the file if it doesn't exist.
   FileRaii fileGuard2("created_by_update.root");
   // in case a previous run of the test failed to clean up, make sure the file doesn't exist:
   gSystem->Unlink(fileGuard2.GetPath().c_str());
   EXPECT_NO_THROW(RFile::Update(fileGuard2.GetPath()));
}

TEST(RFile, OpenForWriting)
{
   FileRaii fileGuard("test_rfile_write.root");

   auto hist = std::make_unique<TH1D>("hist", "", 100, -10, 10);
   hist->FillRandom("gaus", 1000);

   auto file = RFile::Recreate(fileGuard.GetPath());
   file->Put("hist", *hist);
   EXPECT_TRUE(file->Get<TH1D>("hist"));

   EXPECT_EQ(ROOT::GetROOT()->GetListOfFiles()->GetSize(), 0);
}

TEST(RFile, CheckNoAutoRegistrationWrite)
{
   FileRaii fileGuard("test_rfile_noautoreg_write.root");

   auto file = RFile::Recreate(fileGuard.GetPath());
   EXPECT_EQ(gDirectory, gROOT);
   auto hist = std::make_unique<TH1D>("hist", "", 100, -10, 10);
   file->Put("hist", *hist);
   EXPECT_EQ(hist->GetDirectory(), gROOT);
   file->Close();
   EXPECT_EQ(hist->GetDirectory(), gROOT);
   hist.reset();
   // no double free should happen when ROOT exits
}

TEST(RFile, CheckNoAutoRegistrationRead)
{
   FileRaii fileGuard("test_rfile_noautoreg_read.root");

   {
      auto file = RFile::Recreate(fileGuard.GetPath());
      auto hist = std::make_unique<TH1D>("hist", "", 100, -10, 10);
      hist->Fill(4);
      file->Put("hist", *hist);
   }

   {
      auto file = RFile::Open(fileGuard.GetPath());
      EXPECT_EQ(gDirectory, gROOT);
      auto hist = file->Get<TH1D>("hist");
      ASSERT_NE(hist, nullptr);
      EXPECT_EQ(hist->GetDirectory(), nullptr);
      EXPECT_FLOAT_EQ(hist->GetEntries(), 1);
   }
   // no double free should happen when ROOT exits
}

TEST(RFile, CheckNoAutoRegistrationUpdate)
{
   FileRaii fileGuard("test_rfile_noautoreg_update.root");

   {
      auto file = RFile::Recreate(fileGuard.GetPath());
      auto hist = std::make_unique<TH1D>("hist", "", 100, -10, 10);
      hist->Fill(4);
      file->Put("hist", *hist);
   }

   {
      auto file = RFile::Update(fileGuard.GetPath());
      EXPECT_EQ(gDirectory, gROOT);
      auto hist = file->Get<TH1D>("hist");
      ASSERT_NE(hist, nullptr);
      EXPECT_EQ(hist->GetDirectory(), nullptr);
      EXPECT_FLOAT_EQ(hist->GetEntries(), 1);
   }
   // no double free should happen when ROOT exits
}

TEST(RFile, WriteInvalidPaths)
{
   FileRaii fileGuard("test_rfile_write_invalid.root");

   auto file = RFile::Recreate(fileGuard.GetPath());
   std::string a;
   EXPECT_THROW(file->Put("", a), ROOT::RException);
   EXPECT_THROW(file->Put("..", a), ROOT::RException);
   EXPECT_THROW(file->Put(" a", a), ROOT::RException);
   EXPECT_THROW(file->Put("a\n", a), ROOT::RException);
   EXPECT_THROW(file->Put(".", a), ROOT::RException);
   EXPECT_THROW(file->Put("\0", a), ROOT::RException);
   EXPECT_THROW(file->Put(".a", a), ROOT::RException);
   EXPECT_THROW(file->Put("a..", a), ROOT::RException);
}

TEST(RFile, OpenForUpdating)
{
   FileRaii fileGuard("test_rfile_update.root");

   {
      TH1D hist("hist", "", 100, -10, 10);
      hist.FillRandom("gaus", 1000);
      auto file = RFile::Recreate(fileGuard.GetPath());
      file->Put("hist", hist);
   }

   auto file = RFile::Update(fileGuard.GetPath());
   EXPECT_TRUE(file->Get<TH1D>("hist"));
   {
      auto hist2 = std::make_unique<TH1D>("hist2", "a different hist", 10, -1, 1);
      file->Put("hist2", *hist2);
   }
   EXPECT_TRUE(file->Get<TH1D>("hist2"));

   EXPECT_EQ(ROOT::GetROOT()->GetListOfFiles()->GetSize(), 0);
}

TEST(RFile, PutOverwrite)
{
   FileRaii fileGuard("test_rfile_putoverwrite.root");

   auto file = RFile::Recreate(fileGuard.GetPath());

   {
      TH1D hist("hist", "", 100, -10, 10);
      hist.FillRandom("gaus", 1000);
      file->Put("hist", hist);
   }

   {
      auto hist = file->Get<TH1D>("hist");
      ASSERT_TRUE(hist);
      EXPECT_EQ(static_cast<int>(hist->GetEntries()), 1000);
   }

   // Try putting another object at the same path, should fail
   TH1D hist2("hist2", "a different hist", 10, -1, 1);
   hist2.FillRandom("gaus", 10);
   EXPECT_THROW(file->Put("hist", hist2), ROOT::RException);

   // Try with Overwrite, should work (and preserve the old object)
   file->Overwrite("hist", hist2);
   {
      auto hist = file->Get<TH1D>("hist");
      ASSERT_TRUE(hist);
      EXPECT_EQ(static_cast<int>(hist->GetEntries()), 10);

      hist = file->Get<TH1D>("hist;1");
      ASSERT_TRUE(hist);
      EXPECT_EQ(static_cast<int>(hist->GetEntries()), 1000);
   }

   // Now try overwriting without preserving the existing object
   std::string s;
   file->Overwrite("hist", s, false);
   {
      // the previous cycle should be gone...
      auto hist = file->Get<TH1D>("hist;2");
      EXPECT_EQ(hist, nullptr);
      // ...but any cycle before the latest should still be there!
      hist = file->Get<TH1D>("hist;1");
      EXPECT_NE(hist, nullptr);
   }
}

TEST(RFile, WrongExtension)
{
   ROOT::RLogScopedVerbosity logVerb(ROOT::ELogLevel::kInfo);
   // Root files with unconventional extensions are supported.
   {
      FileRaii fileGuard("test_rfile_wrong.root.1");
      RFile::Recreate(fileGuard.GetPath());
   }

   // XML files are not supported.
   FileRaii fileGuardXml("test_rfile_wrong.xml");
   {
      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuardXml.GetPath().c_str(), "RECREATE"));
      TH1D h("h", "h", 10, 0, 1);
      file->WriteObject(&h, "h");
   }
   {
      EXPECT_THROW(RFile::Open(fileGuardXml.GetPath()), ROOT::RException);
      EXPECT_THROW(RFile::Update(fileGuardXml.GetPath()), ROOT::RException);
      EXPECT_THROW(RFile::Recreate(fileGuardXml.GetPath()), ROOT::RException);
   }
}

TEST(RFile, WriteReadInDir)
{
   FileRaii fileGuard("test_rfile_dir.root");

   {
      auto hist = std::make_unique<TH1D>("hist", "", 100, -10, 10);
      hist->FillRandom("gaus", 1000);
      auto file = RFile::Recreate(fileGuard.GetPath());
      file->Put("a/b/hist", *hist);
   }

   {
      auto file = RFile::Open(fileGuard.GetPath());
      EXPECT_TRUE(file->Get<TH1D>("a/b/hist"));
   }
}

TEST(RFile, WriteReadInTFileDir)
{
   FileRaii fileGuard("test_rfile_tfile_dir.root");

   {
      auto hist = std::make_unique<TH1D>("hist", "", 100, -10, 10);
      hist->FillRandom("gaus", 1000);
      TFile file(fileGuard.GetPath().c_str(), "RECREATE");
      auto *d = file.mkdir("a/b");
      d->WriteObject(hist.get(), "hist");
   }

   {
      auto file = RFile::Open(fileGuard.GetPath());
      EXPECT_TRUE(file->Get<TH1D>("a/b/hist"));
      // We won't find any object with a '/' in its name through RFile.
      EXPECT_FALSE(file->Get<TH1D>("a/b/c/d"));
   }
}

TEST(RFile, IterateKeys)
{
   FileRaii fileGuard("test_rfile_iteratekeys.root");

   {
      auto file = RFile::Recreate(fileGuard.GetPath());
      TH1D a;
      auto b = std::make_unique<std::string>();
      std::string c = "0";
      file->Put("a", a);
      file->Put("b", *b);
      file->Put("c", c);
      file->Put("/foo/bar/c", c);
   }

   {
      auto file = RFile::Open(fileGuard.GetPath());
      const auto expected = "a,b,c,foo/bar/c,";
      std::string s = "";
      for (const auto &key : file->ListKeys()) {
         s += key.GetPath() + ",";
      }
      EXPECT_EQ(expected, s);

      // verify the expected iterator operations work
      const auto expected2 = "b,c,foo/bar/c,";
      s = "";
      auto iterable = file->ListKeys();
      auto it = iterable.begin();
      std::advance(it, 1);
      for (; it != iterable.end(); ++it) {
         s += (*it).GetPath() + ",";
      }
      EXPECT_EQ(expected2, s);
   }
}

TEST(RFile, SaneHierarchy)
{
   // verify that we can't create weird hierarchies like:
   //
   // (root)
   //   `--- "a/b": object
   //   |
   //   `--- "a": dir
   //         |
   //         `--- "b": object
   //
   // (who should "a/b" be in this case??)
   //

   FileRaii fileGuard("test_rfile_sane_hierarchy.root");

   {
      auto file = RFile::Recreate(fileGuard.GetPath());
      std::string s;
      file->Put("a", s);
      EXPECT_THROW(file->Put("a/b", s), ROOT::RException);
      file->Put("b/c", s);
      file->Put("b/d", s);
      EXPECT_THROW(file->Put("b/c/d", s), ROOT::RException);
      EXPECT_THROW(file->Put("b", s), ROOT::RException);

      EXPECT_NE(file->Get<std::string>("a"), nullptr);
      EXPECT_EQ(file->Get<std::string>("a/b"), nullptr);
      EXPECT_NE(file->Get<std::string>("b/c"), nullptr);
      EXPECT_NE(file->Get<std::string>("b/d"), nullptr);
      EXPECT_EQ(file->Get<std::string>("b/c/d"), nullptr);
      EXPECT_EQ(file->Get<std::string>("b"), nullptr);
   }
}

TEST(RFile, RefuseToCreateDirOverLeaf)
{
   FileRaii fileGuard("test_rfile_dir_over_leaf.root");
   auto file = RFile::Recreate(fileGuard.GetPath());
   std::string s;
   file->Put("a/b", s);
   try {
      file->Put("a/b/c", s);
      FAIL() << "creating a directory over a leaf path should fail.";
   } catch (const ROOT::RException &ex) {
      EXPECT_THAT(ex.what(), testing::HasSubstr("'a/b'"));
      EXPECT_THAT(ex.what(), testing::HasSubstr("name already taken"));
   }
}

TEST(RFile, IterateKeysRecursive)
{
   FileRaii fileGuard("test_rfile_iteratekeys_recursive.root");

   {
      auto file = RFile::Recreate(fileGuard.GetPath());
      std::string s;
      file->Put("a/c", s);
      file->Put("a/b/d", s);
      file->Put("e/f", s);
      file->Put("e/c/g", s);
   }

   {
      auto file = RFile::Open(fileGuard.GetPath());
      EXPECT_EQ(JoinKeyNames(file->ListKeys()), "a/c, a/b/d, e/f, e/c/g");
      EXPECT_EQ(JoinKeyNames(file->ListKeys("a")), "a/c, a/b/d");
      EXPECT_EQ(JoinKeyNames(file->ListKeys("a/b")), "a/b/d");
      EXPECT_EQ(JoinKeyNames(file->ListKeys("a/b/c")), "");
      EXPECT_EQ(JoinKeyNames(file->ListKeys("e/c")), "e/c/g");
      EXPECT_EQ(JoinKeyNames(file->ListKeys("e/f")), "e/f");
   }
}

TEST(RFile, IterateKeysNonRecursive)
{
   FileRaii fileGuard("test_rfile_iteratekeys_nonrecursive.root");

   {
      auto file = RFile::Recreate(fileGuard.GetPath());
      std::string s;
      file->Put("h", s);
      file->Put("a/c", s);
      file->Put("a/b/d", s);
      file->Put("e/f", s);
      file->Put("e/c/g", s);
   }

   {
      auto file = RFile::Open(fileGuard.GetPath());
      EXPECT_EQ(JoinKeyNames(file->ListKeys("", RFile::kListObjects)), "h");
      EXPECT_EQ(JoinKeyNames(file->ListKeys("a", RFile::kListObjects)), "a/c");
      EXPECT_EQ(JoinKeyNames(file->ListKeys("a/b", RFile::kListObjects)), "a/b/d");
      EXPECT_EQ(JoinKeyNames(file->ListKeys("a/b/c", RFile::kListObjects)), "");
      EXPECT_EQ(JoinKeyNames(file->ListKeys("e", RFile::kListObjects)), "e/f");
   }
}

TEST(RFile, IterateKeysOnlyDirs)
{
   FileRaii fileGuard("test_rfile_iteratekeys_onlydirs.root");

   {
      auto file = RFile::Recreate(fileGuard.GetPath());
      std::string s;
      file->Put("h", s);
      file->Put("a/c", s);
      file->Put("a/b/d", s);
      file->Put("e/f", s);
      file->Put("e/c/g", s);
   }

   {
      auto file = RFile::Open(fileGuard.GetPath());
      EXPECT_EQ(JoinKeyNames(file->ListKeys("", RFile::kListDirs | RFile::kListRecursive)), "a, a/b, e, e/c");
      EXPECT_EQ(JoinKeyNames(file->ListKeys("a", RFile::kListDirs | RFile::kListRecursive)), "a, a/b");
      EXPECT_EQ(JoinKeyNames(file->ListKeys("a/b", RFile::kListDirs | RFile::kListRecursive)), "a/b");
      EXPECT_EQ(JoinKeyNames(file->ListKeys("a/b/c", RFile::kListDirs | RFile::kListRecursive)), "");
      EXPECT_EQ(JoinKeyNames(file->ListKeys("e", RFile::kListDirs | RFile::kListRecursive)), "e, e/c");
   }
}

TEST(RFile, IterateKeysOnlyDirsNonRecursive)
{
   FileRaii fileGuard("test_rfile_iteratekeys_onlydirs_nonrec.root");

   {
      auto file = RFile::Recreate(fileGuard.GetPath());
      std::string s;
      file->Put("h", s);
      file->Put("a/c", s);
      file->Put("a/b/d", s);
      file->Put("e/f", s);
      file->Put("e/c/g", s);
   }

   {
      auto file = RFile::Open(fileGuard.GetPath());
      EXPECT_EQ(JoinKeyNames(file->ListKeys("", RFile::kListDirs)), "a, e");
      EXPECT_EQ(JoinKeyNames(file->ListKeys("a", RFile::kListDirs)), "a, a/b");
      EXPECT_EQ(JoinKeyNames(file->ListKeys("a/b", RFile::kListDirs)), "a/b");
      EXPECT_EQ(JoinKeyNames(file->ListKeys("a/b/c", RFile::kListDirs)), "");
      EXPECT_EQ(JoinKeyNames(file->ListKeys("e", RFile::kListDirs)), "e, e/c");
   }
}

// TODO: this test could in principle also run without davix: need to figure out a way to detect if we have
// remote access capabilities.
#ifdef R__HAS_DAVIX
TEST(RFile, RemoteRead)
{
   constexpr const char *kFileName = "https://root.cern/files/rootcode.root";

   auto file = RFile::Open(kFileName);
   auto content = file->Get<TDirectoryFile>("root");
   ASSERT_NE(content, nullptr);
}
#endif

TEST(RFile, ComplexExample)
{
   FileRaii fileGuard("test_rfile_complex.root");

   auto file = RFile::Recreate(fileGuard.GetPath());

   const std::string topLevelDirs[] = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"};
   for (const auto &dir : topLevelDirs) {
      const auto kNRuns = 5;
      for (int runIdx = 0; runIdx < kNRuns; ++runIdx) {
         const auto runDir = dir + "/run" + (runIdx + 1);

         const auto kNHist = 10;
         for (int i = 0; i < kNHist; ++i) {
            const auto histName = std::string("h") + (i + 1);
            const auto histPath = runDir + "/hists/" + histName;
            const auto histTitle = std::string("Histogram #") + (i + 1);
            TH1D hist(histName, histTitle, 100, -10 * (i + 1), 10 * (i + 1));
            file->Put(histPath, hist);
         }
      }
   }
}

TEST(RFile, Closing)
{
   FileRaii fileGuard("test_rfile_closing.root");

   {
      auto file = RFile::Recreate(fileGuard.GetPath());
      std::string s;
      file->Put("s", s);
      // Explicitly close the file
      file->Close();
      EXPECT_THROW(file->Put("ss", s), ROOT::RException);
   }

   {
      auto file = RFile::Open(fileGuard.GetPath());
      EXPECT_NE(file->Get<std::string>("s"), nullptr);
      file->Close();
      EXPECT_THROW(file->Get<std::string>("s"), ROOT::RException);
   }
}

TEST(RFile, GetAfterOverwriteNoBackup)
{
   FileRaii fileGuard("test_rfile_getafternobackup.root");

   auto file = RFile::Recreate(fileGuard.GetPath());
   std::string s = "foo";
   file->Put("s", s);
   s = "bar";
   file->Overwrite("s", s, false);
   auto ss = file->Get<std::string>("s");
   EXPECT_EQ(*ss, s);

   std::vector<ROOT::Experimental::RKeyInfo> keys;
   for (const auto &key : file->ListKeys())
      keys.push_back(key);

   ASSERT_EQ(keys.size(), 1);
}

TEST(RFile, InvalidPaths)
{
   FileRaii fileGuard("test_rfile_invalidpaths.root");

   auto file = RFile::Recreate(fileGuard.GetPath());
   std::string obj = "obj";

   static const char *const kKeyWhitespaces = "my path with spaces/foo";
   EXPECT_THROW(file->Put(kKeyWhitespaces, obj), ROOT::RException);

   static const char *const kKeyCtrlChars = "my\tpath\nwith\bcontrolcharacters";
   EXPECT_THROW(file->Put(kKeyCtrlChars, obj), ROOT::RException);

   static const char *const kKeyDot = "my/./path";
   EXPECT_THROW(file->Put(kKeyDot, obj), ROOT::RException);
   static const char *const kKeyDot2 = "my/.path";
   EXPECT_THROW(file->Put(kKeyDot2, obj), ROOT::RException);
   static const char *const kKeyDot3 = "../my/path";
   EXPECT_THROW(file->Put(kKeyDot3, obj), ROOT::RException);

   EXPECT_THROW(file->Put("", obj), ROOT::RException);

   // ';' is banned while writing
   EXPECT_THROW(file->Put("myobj;2", obj), ROOT::RException);

   static const char *const kKeyBackslash = "this\\actually\\works!";
   EXPECT_NO_THROW(file->Put(kKeyBackslash, obj));
}

TEST(RFile, LongKeyName)
{
   FileRaii fileGuard("test_rfile_longkey.root");

   auto file = RFile::Recreate(fileGuard.GetPath());

   static const char kKeyLong[] = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                                  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                                  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                                  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
   static_assert(std::size(kKeyLong) > 256);
   std::string obj = "obj";
   EXPECT_NO_THROW(file->Put(kKeyLong, obj));

   auto keys = file->ListKeys();
   auto it = keys.begin();
   EXPECT_EQ((*it).GetPath(), kKeyLong);

   static const char *const kKeyFragmentLong =
      "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/"
      "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
   EXPECT_NO_THROW(file->Put(kKeyFragmentLong, obj));

   static const char *const kKeyFragmentOk =
      "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/"
      "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/AAAAAAAAAAAAAAAAAAAAAAAAAAAA";
   EXPECT_NO_THROW(file->Put(kKeyFragmentOk, obj));
}

TEST(RFile, NormalizedPaths)
{
   FileRaii fileGuard("test_rfile_normalizedpaths.root");

   auto file = RFile::Recreate(fileGuard.GetPath());
   std::string obj = "obj";
   file->Put("/s", obj);
   // "a" and "/a" are equivalent paths, so we cannot overwrite it using Put()...
   EXPECT_THROW(file->Put("s", obj), ROOT::RException);
   // ...and this is true no matter how many leading slashes we have.
   EXPECT_THROW(file->Put("////s", obj), ROOT::RException);
   EXPECT_EQ(*file->Get<std::string>("s"), obj);
   EXPECT_EQ(*file->Get<std::string>("//s"), obj);

   TH1D h("h", "h", 10, -10, 10);
   // Cannot write directory 's': already taken by `obj`.
   EXPECT_THROW(file->Put("s/b//c", h), ROOT::RException);
   file->Put("a/b//c", h);
   EXPECT_THROW(file->Put("a/b/c", h), ROOT::RException);
   EXPECT_NE(file->Get<TH1D>("a/b/c"), nullptr);
   EXPECT_NE(file->Get<TH1D>("//a////b/c"), nullptr);
   EXPECT_THROW(file->Get<TH1D>("a/b/c/"), ROOT::RException);
}

TEST(RFile, GetKeyInfo)
{
   FileRaii fileGuard("test_rfile_getkeyinfo.root");

   auto file = RFile::Recreate(fileGuard.GetPath());
   std::string obj = "obj";
   file->Put("/s", obj);
   file->Put("a/b/c", obj);
   file->Put("b", obj);
   file->Put("/a/d", obj);

   EXPECT_EQ(file->GetKeyInfo("foo"), std::nullopt);

   for (const std::string_view path : { "/s", "a/b/c", "b", "/a/d" }) {
      auto key = file->GetKeyInfo(path);
      ASSERT_NE(key, std::nullopt);
      EXPECT_EQ(key->GetPath(), path[0] == '/' ? path.substr(1) : path);
      EXPECT_EQ(key->GetClassName(), "string");
      EXPECT_EQ(key->GetTitle(), "");
      EXPECT_EQ(key->GetCycle(), 1);
   }
}
