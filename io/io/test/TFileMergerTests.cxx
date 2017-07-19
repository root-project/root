#include "TFileMerger.h"

#include "TMemFile.h"
#include "TTree.h"

#include "gtest/gtest.h"

static void CreateATuple(TMemFile &file, const char *name, double value)
{
   auto mytree = new TTree(name, "A tree");
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
   TMemFile a("a.root", "CREATE");
   CreateATuple(a, "a_tree", 1.);

   // FIXME: Calling this out of order causes two values to be written to the second file.
   TMemFile b("b.root", "CREATE");
   CreateATuple(b, "b_tree", 2.);

   TFileMerger merger;
   auto output = std::unique_ptr<TMemFile>(new TMemFile("output.root", "CREATE"));
   output->ResetBit(kMustCleanup);
   merger.OutputFile(std::move(output));

   merger.AddFile(&a, false);
   merger.AddFile(&b, false);
   // FIXME: Calling merger.Merge() will call Close() and *delete* output.
   merger.PartialMerge();

   auto &result = *static_cast<TMemFile *>(merger.GetOutputFile());
   CheckTree(result, "a_tree", 1);
   CheckTree(result, "b_tree", 2);
}
