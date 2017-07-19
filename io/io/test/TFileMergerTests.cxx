#include "TFileMerger.h"

#include "TMemFile.h"
#include "TTree.h"

#include "gtest/gtest.h"

static void CreateATuple(TMemFile &file, const char *name, double value)
{
   auto mytree = new TTree(name, "A tree");
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
}

TEST(TFileMerger, CreateWithTFilePointer)
{
   TMemFile a("a.root", "CREATE");
   CreateATuple(a, "a_tree", 1.);

   // FIXME: Calling this out of order causes two values to be written to the second file.
   TMemFile b("b.root", "CREATE");
   CreateATuple(b, "b_tree", 2.);

   TFileMerger merger;
   auto output = new TMemFile("output.root", "CREATE");
   output->ResetBit(kMustCleanup);
   merger.OutputFile(std::unique_ptr<TFile>(output));

   merger.AddFile(&a, false);
   merger.AddFile(&b, false);
   // FIXME: Calling merger.Merge() will call Close() and *delete* output.
   merger.PartialMerge();

   CheckTree(*output, "a_tree", 1);
   CheckTree(*output, "b_tree", 2);
}
