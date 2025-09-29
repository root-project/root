#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <TFile.h>
#include <TH1D.h>
#include <TROOT.h>
#include <ROOT/RError.hxx>
#include <ROOT/RFile.hxx>
#include <ROOT/TestSupport.hxx>
#include <numeric>

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
   std::string GetPath() const { return fPath; }

   // Useful if you want to keep a test file after the test has finished running
   // for debugging purposes. Should only be used locally and never pushed.
   void PreserveFile() { fPreserveFile = true; }
};

} // anonymous namespace

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
      EXPECT_EQ(hist->GetDirectory(), nullptr);
      ASSERT_NE(hist, nullptr);
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
   {
      FileRaii fileGuard("test_rfile_wrong.root.1");
      ROOT::TestSupport::CheckDiagsRAII diagsRaii;
      diagsRaii.requiredDiag(kWarning, "ROOT.File", "preferred file extension is \".root\"", false);
      RFile::Recreate(fileGuard.GetPath());
   }
   {
      FileRaii fileGuard("test_rfile_wrong.xml");
      ROOT::TestSupport::CheckDiagsRAII diagsRaii;
      EXPECT_THROW(RFile::Recreate(fileGuard.GetPath()), ROOT::RException);
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

TEST(RFile, InvalidPaths)
{
   FileRaii fileGuard("test_rfile_invalidpaths.root");

   auto file = RFile::Recreate(fileGuard.GetPath());

   static const char *const kKeyLong = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                                       "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                                       "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
                                       "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
   std::string obj = "obj";
   EXPECT_NO_THROW(file->Put(kKeyLong, obj));

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
