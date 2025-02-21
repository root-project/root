#include "ROOT/TestSupport.hxx"

#include "TFileMerger.h"
#include "TFileMergeInfo.h"

#include "TMemFile.h"
#include "TTree.h"
#include "TH1.h"

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
   success = merger.PartialMerge();
   ASSERT_TRUE(success);

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
   ROOT_EXPECT_ERROR(merger.OutputFile(std::move(output)), "TFileMerger::OutputFile",
                     "output file output.root is not writable");
}

TEST(TFileMerger, MergeSingleOnlyListed)
{
   TMemFile a("hist4.root", "CREATE");

   auto hist1 = new TH1F("hist1", "hist1", 1 , 0 , 2);
   auto hist2 = new TH1F("hist2", "hist2", 1 , 0 , 2);
   auto hist3 = new TH1F("hist3", "hist3", 1 , 0 , 2);
   auto hist4 = new TH1F("hist4", "hist4", 1 , 0 , 2);
   hist1->Fill(1);
   hist2->Fill(1);   hist2->Fill(2);
   hist3->Fill(1);   hist3->Fill(1);   hist3->Fill(1);
   hist4->Fill(1);   hist4->Fill(1);   hist4->Fill(1);   hist4->Fill(1);
   a.Write();
   
   TFileMerger merger;
   auto output = std::unique_ptr<TFile>(new TFile("SingleOnlyListed.root", "RECREATE"));
   bool success = merger.OutputFile(std::move(output));
   ASSERT_TRUE(success);
   
   merger.AddObjectNames("hist1");
   merger.AddObjectNames("hist2");
   merger.AddFile(&a, false);
   const Int_t mode = (TFileMerger::kAll | TFileMerger::kRegular | TFileMerger::kOnlyListed);
   success = merger.PartialMerge(mode); // This will delete fOutputFile as we are not using kIncremental, so merger.GetOutputFile() will return a nullptr
   ASSERT_TRUE(success);

   output = std::unique_ptr<TFile>(TFile::Open("SingleOnlyListed.root"));
   ASSERT_TRUE(output.get() && output->GetListOfKeys());
   EXPECT_EQ(output->GetListOfKeys()->GetSize(), 2);
}

// https://github.com/root-project/root/issues/14558 aka https://its.cern.ch/jira/browse/ROOT-4716
TEST(TFileMerger, ImportBranches)
{
   TTree atree("atree", "atitle");
   int value;
   atree.Branch("a", &value);
   value = 11;
   atree.Fill();
   TTree abtree("abtree", "abtitle");
   abtree.Branch("a", &value);
   abtree.Branch("b", &value);
   value = 42;
   abtree.Fill();
  
   TTree dummy("ztree", "zeroBranches");
   TList treelist;

   // Case 1 - Static: ZeroBranches + 1 entry (1 branch) + 1 entry (2 branch)
   treelist.Add(&dummy);
   treelist.Add(&atree);
   treelist.Add(&abtree);
   std::unique_ptr<TFile> file1(TFile::Open("b4716.root", "RECREATE"));
   auto rtree = TTree::MergeTrees(&treelist, "ImportBranches");
   file1->Write();
   ASSERT_TRUE(rtree->FindBranch("a") != nullptr);
   EXPECT_EQ(rtree->FindBranch("a")->GetEntries(),2);
   ASSERT_TRUE(rtree->FindBranch("b") != nullptr);
   EXPECT_EQ(rtree->FindBranch("b")->GetEntries(),2);

   // Case 2 - this (ZeroBranches) + 1 entry (1 branch) + 1 entry (2 branch)
   treelist.Clear();
   treelist.Add(&atree);
   treelist.Add(&abtree);
   std::unique_ptr<TFile> file2(TFile::Open("c4716.root", "RECREATE"));
   TFileMergeInfo info2(file2.get());
   info2.fOptions += " ImportBranches";
   dummy.Merge(&treelist, &info2);
   file2->Write();
   ASSERT_TRUE(dummy.FindBranch("a") != nullptr);
   EXPECT_EQ(dummy.FindBranch("a")->GetEntries(),2);
   ASSERT_TRUE(dummy.FindBranch("b") != nullptr);
   EXPECT_EQ(dummy.FindBranch("b")->GetEntries(),2);

   // Case 3 - this (0 entry / 1 branch) + 1 entry (1 branch) + 1 entry (2 branch)
   treelist.Clear();
   treelist.Add(&atree);
   treelist.Add(&abtree);
   TTree a0tree("a0tree", "a0title");
   a0tree.Branch("a", &value);
   std::unique_ptr<TFile> file3(TFile::Open("d4716.root", "RECREATE"));
   TFileMergeInfo info3(file3.get());
   info3.fOptions += " ImportBranches";
   a0tree.Merge(&treelist, &info3);
   file3->Write();
   ASSERT_TRUE(a0tree.FindBranch("a") != nullptr);
   EXPECT_EQ(a0tree.FindBranch("a")->GetEntries(),2);
   ASSERT_TRUE(a0tree.FindBranch("b") != nullptr);
   EXPECT_EQ(a0tree.FindBranch("b")->GetEntries(),2);

   // Case 4 - this 1 entry (3 branch) + 1 entry (1 branch) + (0 entry / 1 branch)
   TTree abctree("abctree", "abctitle");
   abctree.Branch("a", &value);
   abctree.Branch("b", &value);
   abctree.Branch("c", &value);
   value = 11;
   abctree.Fill();
   TTree ctree("ctree", "ctitle");
   ctree.Branch("c", &value);
   value = 42;
   ctree.Fill();
   TTree c0tree("c0tree", "c0title");
   c0tree.Branch("c", &value);
   std::unique_ptr<TFile> file4(TFile::Open("e4716.root", "RECREATE"));
   TFileMergeInfo info4(file4.get());
   info4.fOptions += " ImportBranches";
   treelist.Clear();
   treelist.Add(&ctree);
   treelist.Add(&c0tree);
   abctree.Merge(&treelist, &info4);
   file4->Write();
   ASSERT_TRUE(abctree.FindBranch("a") != nullptr);
   ASSERT_TRUE(abctree.FindBranch("b") != nullptr);
   ASSERT_TRUE(abctree.FindBranch("c") != nullptr);
   EXPECT_EQ(abctree.FindBranch("a")->GetEntries(),2);
   EXPECT_EQ(abctree.FindBranch("b")->GetEntries(),2);
   EXPECT_EQ(abctree.FindBranch("c")->GetEntries(),2);
}
