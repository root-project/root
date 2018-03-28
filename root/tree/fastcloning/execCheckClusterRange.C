#include "TTree.h"
#include "TBranch.h"
#include "TMemFile.h"
#include "TTreeCloner.h"

void testIterator(TTree *tree)
{
   TTree::TClusterIterator clusterIter = tree->GetClusterIterator(0);
   Long64_t clusterStart;
   Int_t c = 0;
   while( (clusterStart = clusterIter()) < tree->GetEntries()) {
      ++c;
      printf("Cluster #%2d starts at %4lld and ends at %4lld\n",c,clusterStart,clusterIter.GetNextEntry()-1);
   }
}

int testBranchClustering(TBranch *br)
{
   TTree *tree = br->GetTree();
   TTree::TClusterIterator clusterIter = tree->GetClusterIterator(0);
   Long64_t clusterStart;
   int missing = 0;
   while ((clusterStart = clusterIter()) < tree->GetEntries()) {
      bool found = false;
      for(int i = 0; i < br->GetMaxBaskets(); ++i) {
         Long64_t basketStart = br->GetBasketEntry()[i];
         if (basketStart == clusterStart) {
            found = true;
            break;
         }
      }
      if (!found) {
         printf("ERROR: The cluster starting at %lld is not found in branch \"%s\"\n",clusterStart,br->GetName());
         ++missing;
      }
   }
#if 1
   if (missing) {
      br->Print();
      for(int i = 0; i < br->GetMaxBaskets(); ++i) {
         Long64_t basketStart = br->GetBasketEntry()[i];
         printf("%s : %d %lld\n",br->GetName(), i, basketStart);
      }
   }
#endif
   return missing;
}

void AppendToTree(TTree *source, TTree *target)
{
   TTreeCloner cloner(source, target, "");
   if (cloner.IsValid()) {
      target->SetEntries(target->GetEntries() + source->GetEntries());
      cloner.Exec();
   }
}

int execCheckClusterRange()
{
   int v1 = 1;
   int v2 = 2;
   int v3 = 3;

   auto f1 = new TMemFile("f1.root", "RECREATE", "", 0);
   auto t1 = new TTree("t1", "");
   t1->Branch("v1", &v1);
   t1->SetAutoFlush(50);
   for(int i = 0; i < 300; ++i) {
      t1->Fill();
   }
   f1->Write();

   auto f2 = new TMemFile("f2.root", "RECREATE", "", 0);
   auto t2 = new TTree("t2", "");
   t2->Branch("v1", &v1);
   t2->SetAutoFlush(30);
   for(int i = 0; i < 300; ++i) {
      t2->Fill();
   }
   f2->Write();

   auto m1 = new TMemFile("m1.root", "RECREATE", "", 0);
   TTree *tm1 = t1->CloneTree(-1, "fast");
   AppendToTree(t2, tm1);

#ifdef BRANCH_FILL_AND_CLUSTER_FUTURE_FEATURE
   vector<TBranch*> newBranches;
   newBranches.push_back( tm1->Branch("v2",&v2) );
   newBranches.push_back( tm1->Branch("v3",&v2) );

   // What CMS **currently** does.
   // The new Branches are not clustered.
   for(auto b : newBranches) {
      for(Long64_t e = 0; e < tm1->GetEntries(); ++e) {
         b->Fill();
      }
   }
#endif

   m1->Write();

   testIterator(tm1);
   tm1->Print("clusters");

   // Now verify that the cluster information and the basket layout
   // are consistent.
   int errorCount = 0;

   for(auto b : TRangeDynCast<TBranch>(*tm1->GetListOfBranches())) {
      errorCount += testBranchClustering(b);
   }

   auto m2 = new TMemFile("m1.root","RECREATE");

   // Cloning the merged file a second time.
   TTree *tm2 = tm1->CloneTree(-1,"fast");
   // Appending another TTree with cluster size equal to 50.
   AppendToTree(t1, tm2);
   // Appending another TTree with variable cluster size (50 then 30)
   AppendToTree(tm1, tm2);

   m2->Write();

   testIterator(tm2);
   tm2->Print("clusters");

   // Now verify that the cluster information and the basket layout
   // are consistent.

   for(auto b : TRangeDynCast<TBranch>(*tm2->GetListOfBranches())) {
      errorCount += testBranchClustering(b);
   }

   return errorCount;
}
