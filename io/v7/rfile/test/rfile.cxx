#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <TFile.h>
#include <TH1D.h>
#include <TROOT.h>
#include <TTree.h>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RError.hxx>
#include <ROOT/RFile.hxx>
#include <ROOT/RNTupleWriter.hxx>
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

static std::string JoinKeyNames(const ROOT::Experimental::RFileKeyIterable &iterable)
{
   auto beg = iterable.begin();
   if (beg == iterable.end())
      return std::string("");
   return std::accumulate(std::next(beg), iterable.end(), beg->fName,
                          [](const auto &a, const auto &b) { return a + ", " + b.fName; });
};

TEST(RFile, DecomposePath)
{
   using ROOT::Experimental::DecomposePath;

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

TEST(RFile, OpenForReading)
{
   FileRaii fileGuard("test_rfile_read.root");

   // Create a root file to open
   {
      auto tfile = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      TH1D hist("hist", "", 100, -10, 10);
      hist.FillRandom("gaus", 1000);
      tfile->WriteObject(&hist, "hist");
   }

   auto file = RFile::OpenForReading(fileGuard.GetPath());
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

   auto file = RFile::OpenForUpdate(fileGuard.GetPath());
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
   FileRaii fileGuard("test_rfile_wrong.xml");
   EXPECT_THROW(RFile::Recreate(fileGuard.GetPath()), ROOT::RException);
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
      auto file = RFile::OpenForReading(fileGuard.GetPath());
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
      d->WriteObject(hist.get(), "c/d");
   }

   {
      auto file = RFile::OpenForReading(fileGuard.GetPath());
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
      auto b = std::make_unique<TTree>();
      std::string c = "0";
      file->Put("a", a);
      file->Put("b", *b);
      file->Put("c", c);
   }

   {
      auto file = RFile::OpenForReading(fileGuard.GetPath());
      const auto expected = "a,b,c,";
      std::string s = "";
      for (const auto &key : file->GetKeys()) {
         s += key.fName + ",";
      }
      EXPECT_EQ(expected, s);

      // verify the expected iterator operations work
      const auto expected2 = "b,c,";
      s = "";
      auto iterable = file->GetKeys();
      auto it = iterable.begin();
      std::advance(it, 1);
      for (; it != iterable.end(); ++it) {
         s += it->fName + ",";
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
      auto file = RFile::OpenForReading(fileGuard.GetPath());
      EXPECT_EQ(JoinKeyNames(file->GetKeys()), "a/c, a/b/d, e/f, e/c/g");
      EXPECT_EQ(JoinKeyNames(file->GetKeys("a")), "a/c, a/b/d");
      EXPECT_EQ(JoinKeyNames(file->GetKeys("a/b")), "a/b/d");
      EXPECT_EQ(JoinKeyNames(file->GetKeys("a/b/c")), "");
      EXPECT_EQ(JoinKeyNames(file->GetKeys("e/c")), "e/c/g");
      EXPECT_EQ(JoinKeyNames(file->GetKeys("e/f")), "e/f");
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
      auto file = RFile::OpenForReading(fileGuard.GetPath());
      EXPECT_EQ(JoinKeyNames(file->GetKeysNonRecursive()), "h");
      EXPECT_EQ(JoinKeyNames(file->GetKeysNonRecursive("a")), "a/c");
      EXPECT_EQ(JoinKeyNames(file->GetKeysNonRecursive("a/b")), "a/b/d");
      EXPECT_EQ(JoinKeyNames(file->GetKeysNonRecursive("a/b/c")), "");
      EXPECT_EQ(JoinKeyNames(file->GetKeysNonRecursive("e")), "e/f");
   }
}

TEST(RFile, IterateKeysRegex)
{
   FileRaii fileGuard("test_rfile_iteratekeys_regex.root");

   {
      auto file = RFile::Recreate(fileGuard.GetPath());
      std::string s;
      file->Put("a/c", s);
      file->Put("a/b/j/j/j/j", s);
      file->Put("a/b/d", s);
      file->Put("e/f", s);
      file->Put("e/c/g", s);
      file->Put("e/a/b/d", s);
   }

   {
      auto file = RFile::OpenForReading(fileGuard.GetPath());
      EXPECT_EQ(JoinKeyNames(file->GetKeysRegex("")), "");
      EXPECT_EQ(JoinKeyNames(file->GetKeysRegex(".*")), "a/c, a/b/j/j/j/j, a/b/d, e/f, e/c/g, e/a/b/d");
      EXPECT_EQ(JoinKeyNames(file->GetKeysRegex("a")), "");
      EXPECT_EQ(JoinKeyNames(file->GetKeysRegex("a.*")), "a/c, a/b/j/j/j/j, a/b/d");
      EXPECT_EQ(JoinKeyNames(file->GetKeysRegex(".*a/.*")), "a/c, a/b/j/j/j/j, a/b/d, e/a/b/d");
      EXPECT_EQ(JoinKeyNames(file->GetKeysRegex(".*/a/.*")), "e/a/b/d");
      EXPECT_EQ(JoinKeyNames(file->GetKeysRegex("a/b/.*")), "a/b/j/j/j/j, a/b/d");
      EXPECT_EQ(JoinKeyNames(file->GetKeysRegex("a/b/c")), "");
      EXPECT_EQ(JoinKeyNames(file->GetKeysRegex("e/c/.")), "e/c/g");
      EXPECT_EQ(JoinKeyNames(file->GetKeysRegex("e/f")), "e/f");
      EXPECT_EQ(JoinKeyNames(file->GetKeysRegex("./?a/b/.")), "e/a/b/d");
   }
}

#ifdef R__HAS_DAVIX
TEST(RFile, RemoteRead)
{
   constexpr const char *kFileName = "http://root.cern/files/RNTuple.root";

   auto file = RFile::OpenForReading(kFileName);
   auto ntuple = file->Get<ROOT::RNTuple>("Contributors");
   ASSERT_NE(ntuple, nullptr);
}
#endif

TEST(RFile, ComplexExample)
{
   FileRaii fileGuard("test_rfile_complex.root");

   auto file = RFile::Recreate(fileGuard.GetPath());

   auto model = ROOT::RNTupleModel::Create();
   model->MakeField<float>("x");
   model->MakeField<std::vector<float>>("v");

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

         // TODO: add RFile impl in RNTupleFileWriter
         const auto kNDatasets = 10;
         for (int i = 0; i < kNDatasets; ++i) {
            const auto datasetName = std::string("data_") + (i + 1);
            const auto datasetPath = runDir + "/data/" + datasetName;
            const auto dataset = ROOT::RNTupleWriter::Append(model->Clone(), datasetPath, *file);
            for (int j = 0; j < 100; ++j)
               dataset->Fill();
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
      auto file = RFile::OpenForReading(fileGuard.GetPath());
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
   file->Overwrite("s", s, false);
   auto ss = file->Get<std::string>("s");
   EXPECT_EQ(*ss, s);

   std::vector<ROOT::Experimental::RFileKeyInfo> keys;
   for (const auto &key : file->GetKeys())
      keys.push_back(key);

   EXPECT_EQ(keys.size(), 1);
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
   file->Put(kKeyLong, obj);

   auto keys = file->GetKeys();
   auto it = keys.begin();
   EXPECT_EQ(it->fName, kKeyLong);
}
