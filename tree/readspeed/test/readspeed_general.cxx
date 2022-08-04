#include "gtest/gtest.h"

#include "ReadSpeed.hxx"
#include "ReadSpeedCLI.hxx"

#include "ROOT/TTreeProcessorMT.hxx" // for TTreeProcessorMT::GetTasksPerWorkerHint
#include "TFile.h"
#include "TSystem.h"
#include "TTree.h"

using namespace ReadSpeed;

// Helper function to generate a .root file with some dummy data in it.
void RequireFile(const std::string &fname, const std::vector<std::string> &branchNames = {"x"})
{
   if (gSystem->AccessPathName(fname.c_str()) == false) // then the file already exists: weird return value convention
      return;                                           // nothing to do

   TFile f(fname.c_str(), "recreate");
   TTree t("t", "t");

   int var = 42;
   for (const auto &b : branchNames) {
      t.Branch(b.c_str(), &var);
   }

   for (int i = 0; i < 10000000; ++i)
      t.Fill();
   t.Write();
}

// Helper function to concatenate two vectors of strings.
std::vector<std::string> ConcatVectors(const std::vector<std::string> &first, const std::vector<std::string> &second)
{
   std::vector<std::string> all;

   all.insert(all.end(), first.begin(), first.end());
   all.insert(all.end(), second.begin(), second.end());

   return all;
}

// Creates all of our needed .root files and deletes them once the testing is over.
class Environment : public ::testing::Environment {
public:
   ~Environment() override {}

   void SetUp() override
   {
      RequireFile("test1.root");
      RequireFile("test2.root");
      RequireFile("test3.root", {"x", "x_branch", "y_brunch", "mismatched"});
   }

   void TearDown() override
   {
      gSystem->Unlink("test1.root");
      gSystem->Unlink("test2.root");
      gSystem->Unlink("test3.root");
   }
};

int main(int argc, char **argv)
{
   testing::InitGoogleTest(&argc, argv);
   testing::AddGlobalTestEnvironment(new Environment);
   return RUN_ALL_TESTS();
}

TEST(ReadSpeedIntegration, SingleThread)
{
   const auto result = EvalThroughput({{"t"}, {"test1.root", "test2.root"}, {"x"}}, 0);

   EXPECT_EQ(result.fUncompressedBytesRead, 80000000) << "Wrong number of uncompressed bytes read";
   EXPECT_EQ(result.fCompressedBytesRead, 643934) << "Wrong number of compressed bytes read";
}

TEST(ReadSpeedIntegration, MultiThread)
{
   const auto result = EvalThroughput({{"t"}, {"test1.root", "test2.root"}, {"x"}}, 2);

   EXPECT_EQ(result.fUncompressedBytesRead, 80000000) << "Wrong number of uncompressed bytes read";
   EXPECT_EQ(result.fCompressedBytesRead, 643934) << "Wrong number of compressed bytes read";
}

TEST(ReadSpeedIntegration, NonExistentFile)
{
   EXPECT_THROW(EvalThroughput({{"t"}, {"test_fake.root"}, {"x"}}, 0), std::runtime_error)
      << "Should throw for non-existent file";
}

TEST(ReadSpeedIntegration, NonExistentTree)
{
   EXPECT_THROW(EvalThroughput({{"t_fake"}, {"test1.root"}, {"x"}}, 0), std::runtime_error)
      << "Should throw for non-existent tree";
}

TEST(ReadSpeedIntegration, NonExistentBranch)
{
   EXPECT_THROW(EvalThroughput({{"t"}, {"test1.root"}, {"z"}}, 0), std::runtime_error)
      << "Should throw for non-existent branch";
}

TEST(ReadSpeedBranches, SingleBranch)
{
   const auto result = EvalThroughput({{"t"}, {"test3.root"}, {"x"}}, 0);

   EXPECT_EQ(result.fUncompressedBytesRead, 40000000) << "Wrong number of uncompressed bytes read";
   EXPECT_EQ(result.fCompressedBytesRead, 321967) << "Wrong number of compressed bytes read";
}

TEST(ReadSpeedBranches, PatternBranch)
{
   const auto result = EvalThroughput({{"t"}, {"test3.root"}, {"(x|y)_.*nch"}, true}, 0);

   EXPECT_EQ(result.fUncompressedBytesRead, 80000000) << "Wrong number of uncompressed bytes read";
   EXPECT_EQ(result.fCompressedBytesRead, 661576) << "Wrong number of compressed bytes read";
}

TEST(ReadSpeedBranches, NoMatches)
{
   EXPECT_THROW(EvalThroughput({{"t"}, {"test3.root"}, {"x_.*"}, false}, 0), std::runtime_error)
      << "Should throw for no matching branch";
   EXPECT_THROW(EvalThroughput({{"t"}, {"test3.root"}, {"z_.*"}, true}, 0), std::runtime_error)
      << "Should throw for no matching branch";
}

TEST(ReadSpeedBranches, AllBranches)
{
   const auto result = EvalThroughput({{"t"}, {"test3.root"}, {".*"}, true}, 0);

   EXPECT_EQ(result.fUncompressedBytesRead, 160000000) << "Wrong number of uncompressed bytes read";
   EXPECT_EQ(result.fCompressedBytesRead, 1316837) << "Wrong number of compressed bytes read";
}

TEST(ReadSpeedCLI, CheckFilenames)
{
   const std::vector<std::string> baseArgs{"root-readspeed", "--trees", "t", "--branches", "x", "--files"};
   const std::vector<std::string> inFiles{"file-a.root", "file-b.root", "file-c.root"};

   const auto allArgs = ConcatVectors(baseArgs, inFiles);

   const auto parsedArgs = ParseArgs(allArgs);
   const auto outFiles = parsedArgs.fData.fFileNames;

   EXPECT_EQ(outFiles.size(), inFiles.size()) << "Number of parsed files does not match number of provided files.";
   EXPECT_EQ(outFiles, inFiles) << "List of parsed files does not match list of provided files.";
}

TEST(ReadSpeedCLI, CheckTrees)
{
   const std::vector<std::string> baseArgs{"root-readspeed", "--files", "file.root", "--branches", "x", "--trees"};
   const std::vector<std::string> inTrees{"t1", "t2", "tree3"};

   const auto allArgs = ConcatVectors(baseArgs, inTrees);

   const auto parsedArgs = ParseArgs(allArgs);
   const auto outTrees = parsedArgs.fData.fTreeNames;

   EXPECT_EQ(outTrees.size(), inTrees.size()) << "Number of parsed trees does not match number of provided trees.";
   EXPECT_EQ(outTrees, inTrees) << "List of parsed trees does not match list of provided trees.";
}

TEST(ReadSpeedCLI, CheckBranches)
{
   const std::vector<std::string> baseArgs{
      "root-readspeed", "--files", "file.root", "--trees", "t", "--branches",
   };
   const std::vector<std::string> inBranches{"x", "x_branch", "long_branch_name"};

   const auto allArgs = ConcatVectors(baseArgs, inBranches);

   const auto parsedArgs = ParseArgs(allArgs);
   const auto outBranches = parsedArgs.fData.fBranchNames;

   EXPECT_EQ(outBranches.size(), inBranches.size())
      << "Number of parsed trees does not match number of provided trees.";
   EXPECT_EQ(outBranches, inBranches) << "List of parsed trees does not match list of provided trees.";
}

TEST(ReadSpeedCLI, HelpArg)
{
   const std::vector<std::string> allArgs{"root-readspeed", "--help"};

   const auto parsedArgs = ParseArgs(allArgs);

   EXPECT_TRUE(!parsedArgs.fShouldRun) << "Program running when using help argument";
}

TEST(ReadSpeedCLI, NoArgs)
{
   const std::vector<std::string> allArgs{"root-readspeed"};

   const auto parsedArgs = ParseArgs(allArgs);

   EXPECT_TRUE(!parsedArgs.fShouldRun) << "Program running when not using any arguments";
}

TEST(ReadSpeedCLI, InvalidArgs)
{
   const std::vector<std::string> allArgs{
      "root-readspeed", "--files", "file.root", "--trees", "t", "--branches", "x", "--fake-flag",
   };

   const auto parsedArgs = ParseArgs(allArgs);

   EXPECT_TRUE(!parsedArgs.fShouldRun) << "Program running when using invalid flags";
}

TEST(ReadSpeedCLI, RegularArgs)
{
   const std::vector<std::string> allArgs{
      "root-readspeed", "--files", "file.root", "--trees", "t", "--branches", "x",
   };

   const auto parsedArgs = ParseArgs(allArgs);

   EXPECT_TRUE(parsedArgs.fShouldRun) << "Program not running when given valid arguments";
   EXPECT_TRUE(!parsedArgs.fData.fUseRegex) << "Program using regex when it should not";
   EXPECT_EQ(parsedArgs.fNThreads, 0) << "Program not set to single thread mode";
}

TEST(ReadSpeedCLI, RegexArgs)
{
   const std::vector<std::string> allArgs{
      "root-readspeed", "--files", "file.root", "--trees", "t", "--branches-regex", "x.*",
   };

   const auto parsedArgs = ParseArgs(allArgs);

   EXPECT_TRUE(parsedArgs.fShouldRun) << "Program not running when given valid arguments";
   EXPECT_TRUE(parsedArgs.fData.fUseRegex) << "Program not using regex when it should";
}

TEST(ReadSpeedCLI, AllBranches)
{
   const std::vector<std::string> allArgs{
      "root-readspeed", "--files", "file.root", "--trees", "t", "--all-branches",
   };
   const std::vector<std::string> allBranches = {".*"};

   const auto parsedArgs = ParseArgs(allArgs);

   EXPECT_TRUE(parsedArgs.fShouldRun) << "Program not running when given valid arguments";
   EXPECT_TRUE(parsedArgs.fData.fUseRegex) << "Program not using regex when it should";
   EXPECT_TRUE(parsedArgs.fAllBranches) << "Program not checking for all branches when it should";
   EXPECT_EQ(parsedArgs.fData.fBranchNames, allBranches) << "All branch regex not correct";
}

TEST(ReadSpeedCLI, MultipleThreads)
{
   const std::vector<std::string> allArgs{
      "root-readspeed", "--files", "file.root", "--trees", "t", "--branches", "x", "--threads", "16",
   };
   const unsigned int threads = 16;

   const auto parsedArgs = ParseArgs(allArgs);

   EXPECT_TRUE(parsedArgs.fShouldRun) << "Program not running when given valid arguments";
   EXPECT_EQ(parsedArgs.fNThreads, threads) << "Program not using the correct amount of threads";
}

TEST(ReadSpeedCLI, WorkerThreadsHint)
{
   const unsigned int oldTasksPerWorker = ROOT::TTreeProcessorMT::GetTasksPerWorkerHint();
   const std::vector<std::string> allArgs{
      "root-readspeed",
      "--files",
      "file.root",
      "--trees",
      "t",
      "--branches",
      "x",
      "--tasks-per-worker",
      std::to_string(oldTasksPerWorker + 10),
   };

   const auto parsedArgs = ParseArgs(allArgs);
   const auto newTasksPerWorker = ROOT::TTreeProcessorMT::GetTasksPerWorkerHint();

   EXPECT_TRUE(parsedArgs.fShouldRun) << "Program not running when given valid arguments";
   EXPECT_EQ(newTasksPerWorker, oldTasksPerWorker + 10) << "Tasks per worker hint not updated correctly";
}
