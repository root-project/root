#include "TFileMerger.h"

#include "TMemFile.h"
#include "TTree.h"

#include "gtest/gtest.h"

namespace {
using testing::internal::GetCapturedStderr;
using testing::internal::CaptureStderr;
using testing::internal::RE;
class ExpectedErrorRAII {
   std::string ExpectedRegex;
   void pop()
   {
      std::string Seen = GetCapturedStderr();
      bool match = RE::FullMatch(Seen, RE(ExpectedRegex));
      EXPECT_TRUE(match);
      if (!match) {
         std::string msg = "Match failed!\nSeen: '" + Seen + "'\nRegex: '" + ExpectedRegex + "'\n";
         GTEST_NONFATAL_FAILURE_(msg.c_str());
      }
   }

public:
   ExpectedErrorRAII(std::string E) : ExpectedRegex(E) { CaptureStderr(); }
   ~ExpectedErrorRAII() { pop(); }
};
}

#define EXPECT_ROOT_ERROR(expression, expected_error) \
   {                                                  \
      ExpectedErrorRAII EE(expected_error);           \
      expression;                                     \
   }

static void CreateATuple(TMemFile &file, const char *name, double value)
{
   auto mytree = new TTree(name, "A tree");
   // FIXME: We inherit EnableImplicitIMT from TBufferMerger tests (we are sharing the same executable) where we call
   // EnableThreadSafety(). Here, we hit a race condition in TBranch::FlushBaskets. Once we get that fixed we probably
   // should re-enable implicit MT.
   //
   // In general, we should probably have a way to conditionally enable/disable thread safety.
   mytree->SetImplicitMT(false);

   mytree->SetDirectory(&file);
   mytree->Branch(name, &value);
   mytree->Fill();
   file.Write();
}

static void CheckTree(TMemFile &file, const char *name, double expectedValue)
{
   auto t = static_cast<TTree *>(file.Get(name));
   ASSERT_TRUE(t != nullptr);

   double d;
   t->SetBranchAddress(name, &d);
   t->GetEntry(0);
   EXPECT_EQ(expectedValue, d);
   // Setting branch address to a stack address requires to either call ResetBranchAddresses or delete the tree.
   t->ResetBranchAddresses();
}

TEST(TFileMerger, CreateWithTFilePointer)
{
   TMemFile a("a.root", "RECREATE");
   CreateATuple(a, "a_tree", 1.);

   TMemFile b("b.root", "RECREATE");
   CreateATuple(b, "b_tree", 2.);

   TFileMerger merger;
   auto output = std::unique_ptr<TMemFile>(new TMemFile("output.root", "CREATE"));
   bool success = merger.OutputFile(std::move(output));

   ASSERT_TRUE(success);

   merger.AddFile(&a, false);
   merger.AddFile(&b, false);
   // FIXME: Calling merger.Merge() will call Close() and *delete* output.
   merger.PartialMerge();

   auto &result = *static_cast<TMemFile *>(merger.GetOutputFile());
   CheckTree(result, "a_tree", 1);
   CheckTree(result, "b_tree", 2);
}

TEST(TFileMerger, CreateWithUnwritableTFilePointer)
{
   TFileMerger merger;
   auto output = std::unique_ptr<TMemFile>(new TMemFile("output.root", "RECREATE"));
   // FIXME: The ctor of TMemFile sets the 'zombie' flag to all TMemFiles whose options are different than CREATE and
   // RECREATE. We should probably fix the API but until then work around it.
   output->SetWritable(false);
   EXPECT_ROOT_ERROR(merger.OutputFile(std::move(output)), "Error in .* output file output.root is not writable\n");
}
