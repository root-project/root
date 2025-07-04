#include "gtest/gtest.h"

#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleClassicBrowse.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/TestSupport.hxx>

#include <TBrowser.h>
#include <TBrowserImp.h>
#include <TClass.h>
#include <TFile.h>

#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

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

class TBrowserTestImp : public TBrowserImp {
public:
   std::vector<std::string> fAdded;

   void Add(TObject *, const char *name, Int_t) final { fAdded.push_back(name); }
};

} // anonymous namespace

TEST(RNTupleBrowse, Simple)
{
   FileRaii fileGuard("test_ntuple_browse_simple.root");

   {
      auto model = ROOT::RNTupleModel::Create();
      model->MakeField<float>("f");
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
   }

   auto imp = std::make_unique<TBrowserTestImp>();
   ROOT::TestSupport::CheckDiagsRAII diagRAII;
   diagRAII.optionalDiag(kWarning, "TBrowser::TBrowser", "The ROOT browser cannot run in batch mode",
                         /*wholeStringNeedsToMatch=*/false);
   auto b = new TBrowser("", "", imp.get());

   std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str()));
   std::unique_ptr<ROOT::RNTuple> ntpl(file->Get<ROOT::RNTuple>("ntpl"));

   ROOT::Internal::BrowseRNTuple(ntpl.get(), b);

   ASSERT_EQ(1u, imp->fAdded.size());
   EXPECT_EQ("f", imp->fAdded[0]);
}
