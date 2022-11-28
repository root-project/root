#include "gtest/gtest.h"

#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleImporter.hxx>

#include <TFile.h>
#include <TTree.h>

#include <cstdio>
#include <string>

using ROOT::Experimental::RNTupleImporter;
using ROOT::Experimental::RNTupleReader;

/**
 * An RAII wrapper around an open temporary file on disk. It cleans up the guarded file when the wrapper object
 * goes out of scope.
 */
class FileRaii {
private:
   std::string fPath;

public:
   explicit FileRaii(const std::string &path) : fPath(path) {}
   FileRaii(const FileRaii &) = delete;
   FileRaii &operator=(const FileRaii &) = delete;
   ~FileRaii() { std::remove(fPath.c_str()); }
   std::string GetPath() const { return fPath; }
};

TEST(RNTupleImporter, Empty)
{
   FileRaii fileGuard("test_ntuple_importer_empty.root");
   {
      std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto tree = std::make_unique<TTree>("tree", "");
      tree->Write();
   }

   auto importer = RNTupleImporter::Create(fileGuard.GetPath(), "tree", fileGuard.GetPath()).Unwrap();
   importer->SetIsQuiet(true);
   EXPECT_THROW(importer->Import(), ROOT::Experimental::RException);
   importer->SetNTupleName("ntuple");
   importer->Import();
   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(0U, reader->GetNEntries());
}
