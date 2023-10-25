#include "gtest/gtest.h"

#include "ROOT/RNTupleImporterCLI.hxx"
#include "ROOT/RNTupleInspector.hxx"

#include "ROOT/TestSupport.hxx"
#include "ntupleutil_test.hxx"

#include "TFile.h"
#include "TSystem.h"
#include "TTree.h"

using namespace ROOT::Experimental::RNTupleImporterCLI;
using ROOT::Experimental::RNTupleInspector;
using ROOT::Experimental::RNTupleReader;

TEST(RNTupleImporterCLI, Basic)
{
   FileRaii inputFileGuard("test_ntuple_importer_cli_basic_in.root");
   {
      std::unique_ptr<TFile> file(TFile::Open(inputFileGuard.GetPath().c_str(), "RECREATE"));
      auto tree = std::make_unique<TTree>("tree", "");
      tree->Write();
   }
   FileRaii outputFileGuard("test_ntuple_importer_cli_basic_out.root");

   const std::vector<std::string> args{"ttree2rntuple",          "-t", "tree", "-i", inputFileGuard.GetPath(), "-o",
                                       outputFileGuard.GetPath()};

   const auto importerCfg = ParseArgs(args);

   EXPECT_TRUE(importerCfg.fShouldRun) << "Arguments are valid and should have been correctly parsed.";

   RunImporter(importerCfg);

   EXPECT_NO_THROW(RNTupleReader::Open("tree", outputFileGuard.GetPath()))
      << "RNTuple should exist in the provided output file";
}

TEST(RNTupleImporterCLI, LongArgs)
{
   const std::vector<std::string> args{"ttree2rntuple", "--ttree",          "tree", "--infile", "my_tree_file.root",
                                       "--outfile",     "my_tree_file.root"};

   const auto importerCfg = ParseArgs(args);

   EXPECT_EQ(importerCfg.fTreeName, "tree") << "Provided TTree name does not match expected name.";
   EXPECT_EQ(importerCfg.fTreePath, "my_tree_file.root") << "Provided input filename does not match expected name.";
   EXPECT_EQ(importerCfg.fNTuplePath, "my_tree_file.root") << "Provided output filename does not match expected name.";
}

TEST(RNTupleImporterCLI, NoArgs)
{
   const std::vector<std::string> args{"ttree2rntuple"};

   const auto importerCfg = ParseArgs(args);

   EXPECT_TRUE(!importerCfg.fShouldRun) << "Program running when not provided any argument.";
}

TEST(RNTupleImporterCLI, HelpArg)
{
   const std::vector<std::string> args{"ttree2rntuple", "--help"};

   const auto importerCfg = ParseArgs(args);

   EXPECT_TRUE(!importerCfg.fShouldRun) << "Program running when provided --help argument.";
}

TEST(RNTupleImporterCLI, MissingTreeName)
{
   const std::vector<std::string> args{"ttree2rntuple", "--infile", "my_tree_file.root", "--outfile",
                                       "my_tree_file.root"};

   const auto importerCfg = ParseArgs(args);

   EXPECT_TRUE(!importerCfg.fShouldRun) << "Program running when missing --ttree argument.";
}

TEST(RNTupleImporterCLI, MissingTreePath)
{
   const std::vector<std::string> args{"ttree2rntuple", "--ttree", "tree", "--outfile", "my_tree_file.root"};

   const auto importerCfg = ParseArgs(args);

   EXPECT_TRUE(!importerCfg.fShouldRun) << "Program running when missing --infile argument.";
}

TEST(RNTupleImporterCLI, MissingNTuplePath)
{
   const std::vector<std::string> args{"ttree2rntuple", "--ttree", "tree", "--infile", "my_tree_file.root"};

   const auto importerCfg = ParseArgs(args);

   EXPECT_TRUE(!importerCfg.fShouldRun) << "Program running when missing --outfile argument.";
}

TEST(RNTupleImporterCLI, InvalidArgs)
{
   const std::vector<std::string> args{"ttree2rntuple", "--wrong"};

   const auto importerCfg = ParseArgs(args);

   EXPECT_TRUE(!importerCfg.fShouldRun) << "Program running when not using any arguments.";
}

TEST(RNTupleImporterCLI, Name)
{
   FileRaii inputFileGuard("test_ntuple_importer_cli_name_in.root");
   {
      std::unique_ptr<TFile> file(TFile::Open(inputFileGuard.GetPath().c_str(), "RECREATE"));
      auto tree = std::make_unique<TTree>("tree", "");
      tree->Write();
   }
   FileRaii outputFileGuard("test_ntuple_importer_cli_name_out.root");

   const std::vector<std::string> args{
      "ttree2rntuple", "-t", "tree", "-i", inputFileGuard.GetPath(), "-r", "ntuple", "-o", outputFileGuard.GetPath()};

   const auto importerCfg = ParseArgs(args);

   EXPECT_EQ("ntuple", importerCfg.fNTupleName) << "RNTuple name should be correctly set";
   EXPECT_TRUE(importerCfg.fShouldRun) << "Arguments are valid and should have been correctly parsed.";

   RunImporter(importerCfg);

   EXPECT_NO_THROW(RNTupleReader::Open("ntuple", outputFileGuard.GetPath()))
      << "RNTuple should exist in the provided output file";
}

TEST(RNTupleImporterCLI, WriteOptions)
{
   FileRaii inputFileGuard("test_ntuple_importer_cli_writeopts_in.root");
   {
      std::unique_ptr<TFile> file(TFile::Open(inputFileGuard.GetPath().c_str(), "RECREATE"));
      auto tree = std::make_unique<TTree>("tree", "");
      Int_t a = 42;
      tree->Branch("a", &a);
      tree->Fill();
      tree->Write();
   }
   FileRaii outputFileGuard("test_ntuple_importer_cli_writeopts_out.root");

   const std::vector<std::string> args{"ttree2rntuple",
                                       "-t",
                                       "tree",
                                       "-i",
                                       inputFileGuard.GetPath(),
                                       "-o",
                                       outputFileGuard.GetPath(),
                                       "-c",
                                       "207",
                                       "--unzipped-page-size",
                                       std::to_string(32 * 1024),
                                       "--zipped-cluster-size",
                                       std::to_string(25 * 1000 * 1000),
                                       "--max-unzipped-cluster-size",
                                       std::to_string(256 * 1024 * 1024)};

   const auto importerCfg = ParseArgs(args);

   EXPECT_EQ(207, importerCfg.fNTupleOpts.GetCompression()) << "Compression should be correctly set";
   EXPECT_EQ(32 * 1024, importerCfg.fNTupleOpts.GetApproxUnzippedPageSize()) << "Page size should be correctly set";
   EXPECT_EQ(25 * 1000 * 1000, importerCfg.fNTupleOpts.GetApproxZippedClusterSize())
      << "Cluster size should be correctly set";
   EXPECT_EQ(256 * 1024 * 1024, importerCfg.fNTupleOpts.GetMaxUnzippedClusterSize())
      << "Max cluster size should be correctly set";
   EXPECT_TRUE(importerCfg.fShouldRun) << "Arguments are valid and should have been correctly parsed.";

   RunImporter(importerCfg);

   auto inspector = RNTupleInspector::Create("tree", outputFileGuard.GetPath());

   EXPECT_EQ(207, inspector->GetCompressionSettings()) << "Compression should be correctly set";
}

TEST(RNTupleImporterCLI, ConvertDots)
{
   FileRaii inputFileGuard("test_ntuple_importer_cli_dots_in.root");
   {
      std::unique_ptr<TFile> file(TFile::Open(inputFileGuard.GetPath().c_str(), "RECREATE"));
      auto tree = std::make_unique<TTree>("tree", "");
      Int_t pt = 42;
      tree->Branch("a.pt", &pt);
      tree->Fill();
      tree->Write();
   }
   FileRaii outputFileGuard("test_ntuple_importer_cli_dots_out.root");

   const std::vector<std::string> args{
      "ttree2rntuple", "-t", "tree", "-i", inputFileGuard.GetPath(), "-o", outputFileGuard.GetPath(), "--convert-dots"};

   const auto importerCfg = ParseArgs(args);

   EXPECT_TRUE(importerCfg.fShouldRun) << "Arguments are valid and should have been correctly parsed.";

   RunImporter(importerCfg);

   auto ntuple = RNTupleReader::Open("tree", outputFileGuard.GetPath());

   auto pt = ntuple->GetView<Int_t>("a_pt");
   EXPECT_EQ(42, pt(0)) << "RNTuple entry value should match that of TTree";
}
