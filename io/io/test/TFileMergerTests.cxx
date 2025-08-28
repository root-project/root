#include "ROOT/TestSupport.hxx"

#include "TFileMerger.h"
#include "TFileMergeInfo.h"
#include "TBranch.h"

#include "TMemFile.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include "TNtuple.h"
#include "TProfile.h"
#include "TROOT.h"
#include "TSystem.h"

#include <filesystem>
#include <memory>

#include "gtest/gtest.h"

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
TEST(TFileMerger, MergeBranches)
{
   TTree atree("atree", "atitle");
   int value;
   atree.Branch("a", &value);

   TTree abtree("abtree", "abtitle");
   abtree.Branch("a", &value);
   abtree.Branch("b", &value);
   value = 11;
   abtree.Fill();
   value = 42;
   abtree.Fill();

   TTree dummy("emptytree", "emptytitle");
   TList treelist;

   // Case 1 - Static: NoBranch + NoEntries + 2 entries
   treelist.Add(&dummy);
   treelist.Add(&atree);
   treelist.Add(&abtree);
   std::unique_ptr<TFile> file1(TFile::Open("b_4716.root", "RECREATE"));
   auto rtree = TTree::MergeTrees(&treelist);
   ASSERT_TRUE(rtree->FindBranch("a") != nullptr);
   EXPECT_EQ(rtree->FindBranch("a")->GetEntries(), 2);
   ASSERT_TRUE(rtree->FindBranch("b") == nullptr);
   file1->Write();

   // Case 2 - This (NoBranch) + NoEntries + 2 entries
   treelist.Clear();
   treelist.Add(&atree);
   treelist.Add(&abtree);
   std::unique_ptr<TFile> file2(TFile::Open("c_4716.root", "RECREATE"));
   TFileMergeInfo info2(file2.get());
   dummy.Merge(&treelist, &info2);
   ASSERT_TRUE(dummy.FindBranch("a") != nullptr);
   ASSERT_TRUE(dummy.FindBranch("b") == nullptr);
   EXPECT_EQ(dummy.FindBranch("a")->GetEntries(), 2);
   EXPECT_EQ(atree.FindBranch("a")->GetEntries(), 2);
   // atree has now 2 entries instead of zero since it was used as skeleton for dummy
   file2->Write();

   // Case 3 - This (NoEntries) + 2 entries
   TTree a0tree("a0tree", "a0title"); // We cannot reuse atree since it was cannibalized by dummy
   a0tree.Branch("a", &value);
   treelist.Clear();
   treelist.Add(&abtree);
   std::unique_ptr<TFile> file3(TFile::Open("d_4716.root", "RECREATE"));
   TFileMergeInfo info3(file3.get());
   a0tree.Merge(&treelist, &info3);
   ASSERT_TRUE(a0tree.FindBranch("a") != nullptr);
   ASSERT_TRUE(a0tree.FindBranch("b") == nullptr);
   EXPECT_EQ(a0tree.FindBranch("a")->GetEntries(), 2);
   file3->Write();
}

// https://github.com/root-project/root/issues/6640
TEST(TFileMerger, ChangeFile)
{
   {
      TFile f{"file6640mergerinput.root", "RECREATE"};

      TTree t{"T", "SetMaxTreeSize(1000)", 99, &f};
      int x;
      auto nentries = 20000;

      t.Branch("x", &x, "x/I");

      // Call function to forcedly trigger TTree::ChangeFile.
      // This will produce in total 3 files:
      // * file6640mergerinput.root
      // * file6640mergerinput_1.root
      // * file6640mergerinput_2.root
      TTree::SetMaxTreeSize(1000);

      for (auto i = 0; i < nentries; i++) {
         x = i;
         t.Fill();
      }

      // Write last file to disk
      auto cf = t.GetCurrentFile();
      cf->Write();
      cf->Close();
   }
   {
      TFileMerger filemerger{false, false};
      filemerger.OutputFile(std::unique_ptr<TFile>{TFile::Open("file6640mergeroutput.root", "RECREATE")});

      TFile fin{"file6640mergerinput.root", "READ"};
      TFile fin1{"file6640mergerinput_1.root", "READ"};
      TFile fin2{"file6640mergerinput_2.root", "READ"};

      filemerger.AddAdoptFile(&fin);
      filemerger.AddAdoptFile(&fin1);
      filemerger.AddAdoptFile(&fin2);

      // Before the fix, TTree::ChangeFile was called during Merge
      // in the end deleting the TFileMerger's output file and leading to a crash.
      filemerger.Merge();
   }
   {
      TFile fout{"file6640mergeroutput.root", "READ"};
      EXPECT_EQ(fout.Get<TTree>("T")->GetEntries(), 20000);
   }
   gSystem->Unlink("file6640mergerinput.root");
   gSystem->Unlink("file6640mergerinput_1.root");
   gSystem->Unlink("file6640mergerinput_2.root");
   gSystem->Unlink("file6640mergeroutput.root");
}

TEST(TFileMerger, SelectiveMergeWithDirectories)
{
   constexpr auto input1 = "selectiveMerge_input_1.root";
   constexpr auto input2 = "selectiveMerge_input_2.root";
   constexpr auto output = "selectiveMerge_output.root";
   for (auto const &filename : {input1, input2}) {
      TH1F histo("histo", "Histo", 2, 0, 1);
      TFile infile(filename, "recreate");
      auto dir = infile.mkdir("A");
      dir->WriteObject(&histo, "Histo_A1");
      if (filename == input1)
         dir->WriteObject(&histo, "Histo_A2");
      if (filename == input2)
         dir->WriteObject(&histo, "Histo_A3");

      dir = infile.mkdir("B");
      dir->WriteObject(&histo, "Histo_B");

      dir = infile.mkdir("C")->mkdir("D");
      dir->WriteObject(&histo, "Histo_D");
      dir = infile.mkdir("E")->mkdir("F");
      dir->WriteObject(&histo, "Histo_F1");
      if (filename == input1)
         dir->WriteObject(&histo, "Histo_F2");
      if (filename == input2)
         dir->WriteObject(&histo, "Histo_F3");
   }

   {
      TFileMerger fileMerger(false);
      fileMerger.AddFile(input1);
      fileMerger.AddFile(input2);
      fileMerger.AddObjectNames("A");
      fileMerger.AddObjectNames("Histo_F1 Histo_F2 Histo_F3");
      fileMerger.OutputFile(output);
      fileMerger.PartialMerge(TFileMerger::kOnlyListed | TFileMerger::kAll | TFileMerger::kRegular);
   }

   TFile outfile(output);
   auto dir = outfile.Get<TDirectory>("A");
   ASSERT_NE(dir, nullptr);
   for (auto name : {"Histo_A1", "Histo_A2", "Histo_A3"})
      EXPECT_NE(dir->Get<TH1F>(name), nullptr) << name;

   EXPECT_EQ(outfile.Get("B"), nullptr);
   EXPECT_EQ(outfile.Get("C"), nullptr);

   dir = outfile.Get<TDirectory>("E");
   ASSERT_NE(dir, nullptr);
   dir = dir->Get<TDirectory>("F");
   ASSERT_NE(dir, nullptr);
   for (auto name : {"Histo_F1", "Histo_F2", "Histo_F3"})
      EXPECT_NE(dir->Get<TH1F>(name), nullptr) << name;

   for (auto name : {input1, input2, output})
      gSystem->Unlink(name);
}

// https://github.com/root-project/root/issues/9022
TEST(TFileMerger, SingleHistFile)
{
   auto filename1 = "f1_9022.root";
   auto filename2 = "f2_9022.root";
   auto outname = "file9022mergeroutput.root";
   {
      TFile f1(filename1, "RECREATE");
      TH1F h("h1", "h1", 1, 0, 1);
      h.Write();
      f1.Close();
      TFile f2(filename2, "RECREATE");
      TH1F h2("h2", "h2", 1, 0, 1);
      h2.Write();
      f2.Close();
   }
   {
      TFileMerger filemerger{false, false};
      filemerger.SetMaxOpenedFiles(2);
      filemerger.OutputFile(std::unique_ptr<TFile>{TFile::Open(outname, "RECREATE")});

      filemerger.AddFile(filename1);
      filemerger.AddFile(filename2);

      filemerger.Merge();
   }
   {
      TFile file(outname, "READ");
      EXPECT_NE(file.Get<TH1>("h1"), nullptr);
      EXPECT_NE(file.Get<TH1>("h2"), nullptr);
   }
   gSystem->Unlink(filename1);
   gSystem->Unlink(filename2);
   gSystem->Unlink(outname);
}

// The merging demonstrated in tutorials/io/mergeSelective.C
TEST(TFileMerger, MergeSelectiveTutorial)
{
   using namespace std::filesystem;
   struct CleanupRAII {
      std::vector<path> items;
      ~CleanupRAII()
      {
         for (auto const &item : items)
            remove(item);
      }
   } cleanup;

   // Create the files to be merged
   const auto baseDir = path{gROOT->GetTutorialsDir()};
   std::cout << "BaseDir: " << baseDir << std::endl;
   const auto file0 = baseDir / "tomerge00.root";
   const auto file1 = baseDir / "tomerge01.root";
   try {
      copy(baseDir / "hsimple.root", file0);
      copy(baseDir / "hsimple.root", file1);
      cleanup.items.push_back(file0);
      cleanup.items.push_back(file1);
   } catch (filesystem_error &e) {
      std::cerr << e.what() << "\n";
   }

   //------------------------------------
   // Merge only the listed objects
   //------------------------------------
   {
      TFileMerger fm{false};
      fm.OutputFile("exclusive.root");
      cleanup.items.push_back("exclusive.root");
      fm.AddObjectNames("hprof ntuple");
      fm.AddFile(file0.string().c_str());
      fm.AddFile(file1.string().c_str());
      // Must add new merging flag on top of the default ones
      Int_t default_mode = TFileMerger::kAll | TFileMerger::kIncremental;
      Int_t mode = default_mode | TFileMerger::kOnlyListed;
      fm.PartialMerge(mode);
   }
   {
      TFile file("exclusive.root");
      EXPECT_NE(file.Get<TProfile>("hprof"), nullptr);
      EXPECT_EQ(file.Get("hpx"), nullptr);
      EXPECT_EQ(file.Get("hpxpy"), nullptr);
      EXPECT_NE(file.Get<TNtuple>("ntuple"), nullptr);
   }

   //------------------------------------
   // Skip merging of the listed objects
   //------------------------------------
   {
      TFileMerger fm{true};
      fm.OutputFile("skipped.root");
      cleanup.items.push_back("skipped.root");
      fm.AddObjectNames("hprof folder");
      fm.AddFile(file0.string().c_str());
      fm.AddFile(file1.string().c_str());
      // Must add new merging flag on top of the default ones
      Int_t default_mode = TFileMerger::kAll | TFileMerger::kIncremental;
      auto mode = default_mode | TFileMerger::kSkipListed;
      fm.PartialMerge(mode);
      fm.Reset();
   }
   {
      TFile file("skipped.root");
      EXPECT_EQ(file.Get<TProfile>("hprof"), nullptr);
      EXPECT_NE(file.Get("hpx"), nullptr);
      EXPECT_NE(file.Get("hpxpy"), nullptr);
      EXPECT_NE(file.Get<TNtuple>("ntuple"), nullptr);
   }
}
