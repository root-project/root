#include "TTree.h"
#include "TChain.h"
#include "TFile.h"
#include "TFileCacheRead.h"
#include "TTreeCache.h"
#include <TSelector.h>
#include <vector>

class MySelector : public TSelector {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain

   MySelector(TTree * /*tree*/ =0) : fChain(0), pfs1(0), pf1(0), first(kTRUE) { }
   ~MySelector() override { }
   Int_t   Version() const override { return 2; }
   void    Begin(TTree *tree) override { }
   void    SlaveBegin(TTree *tree) override { }
   void    Init(TTree *tree) override { fChain = tree; }
   Bool_t  Notify() override { return kTRUE; }
   Bool_t  Process(Long64_t entry) override {
      GetEntry(entry);
      printf("MySelector::Process called, entry=%lld\n", entry);
      printf("Reported cache size from fChain: %lld\n", fChain->GetCacheSize());
      printf("Reported AutoFlush value for the TTree %lld\n", fChain->GetTree()->GetAutoFlush());
      printf("Reported cache size from TTree: %lld\n", fChain->GetTree()->GetCacheSize());
      TTreeCache *pf = dynamic_cast<TTreeCache*>(fChain->GetCurrentFile()->GetCacheRead(fChain->GetTree()));
      Long64_t pfs = 0;
      if (pf) {
         pfs = pf->GetBufferSize();
         printf("Buffer size of TTreeCache: %lld\n", pfs);
         if (first) {
            pf1 = pf;
            pfs1 = pfs;
         }
      } else {
         printf("No TTreeCache after Process()\n");
      }
      if (!first) {
         if (pfs1 == pfs) {
            if (pf1 != pf) {
               printf("TTreeCache has changed memory location\n");
            } else {
               printf("TTreeCache object pointer stayed the same\n");
            }
         } else {
            printf("TTreeCache buffer size changed, object location not checked\n");
         }
      }
      first = kFALSE;
      return kTRUE;
   }
   Int_t   GetEntry(Long64_t entry, Int_t getall = 0) override { return fChain ? fChain->GetTree()->GetEntry(entry, getall) : 0; }
   void    SetOption(const char *option) override {  }
   void    SetObject(TObject *obj) override { }
   void    SetInputList(TList *input) override { }
   TList  *GetOutputList() const override { return 0; }
   void    SlaveTerminate() override { }
   void    Terminate() override { }

   Long64_t pfs1;
   TTreeCache *pf1;
   Bool_t first;

   ClassDefOverride(MySelector,0);
};

int gNum=0;

TTree *fillTree(Long64_t autof)
{
   TString name("autotree");
   TTree *tree = new TTree(name,name);
   tree->SetAutoFlush(autof);
   std::vector<Int_t> myvec;
   for(int i=0;i<1000000;i++) myvec.push_back(0x55aa55aa);
   tree->Branch("myvec", &myvec);
   for(int i=0;i<100;i++) {
     tree->Fill();
   }
   tree->ResetBranchAddresses();
   return tree;
}

void writefiles(Double_t f) {
   TFile *output = new TFile("autocache1.root","RECREATE");
   fillTree(50);
   output->Write();
   delete output;
   output = new TFile("autocache2.root","RECREATE");
   fillTree(50*f);
   output->Write();
   delete output;
}

int checkfree() {
   printf("--- Start of subtest %d\n", ++gNum);
   printf("Check if deleting a tree can affect another tree's cache\n");

   TFile *file = TFile::Open( "AliESDs-0.root" );
   TTree *T = 0;

   file->GetObject("esdTree", T);
   T->SetCacheSize(-1);

   TTree *T2 = new TTree("mytree","mytree");
   delete T2;

   TTreeCache *pf = dynamic_cast<TTreeCache*>(file->GetCacheRead(T));
   delete file;

   if (pf == 0) {
      printf("cache of first tree got deleted\n");
   } else {
      printf("cache did not get deleted\n");
   }

   printf("\n");

   return 0;
}

int runone(const Long64_t *cSizep=0) {
   printf("--- Start of subtest %d\n", ++gNum);

   TFile *file = TFile::Open( "AliESDs-0.root" );
   TTree *T = 0;
   Long64_t n;

   file->GetObject("esdTree", T);
   if (T == 0) {
      printf("ERROR: No tree found\n");
      return 1;
   }
   printf("Read TTree from AliESDs-0.root, will read events and print cache info\n");
   printf("Reported AutoFlush value for the TTree %lld\n", T->GetAutoFlush());
   if (cSizep) {
     printf("-> Calling SetCacheSize(%lld) <-\n", *cSizep);
     T->SetCacheSize(*cSizep);
   } else {
     printf("No call to SetCacheSize()\n");
   }
   Long64_t nentries = T->GetEntries();
   printf("Reported number of entries %lld. Getting all entries..\n", nentries);
   for(n=0;n<nentries;n++) {
      T->LoadTree(n);
      T->GetEntry(n);
   }
   printf("Reported cache size from TTree: %lld\n", T->GetCacheSize());
   TTreeCache *pf = dynamic_cast<TTreeCache*>(file->GetCacheRead(T));
   if (pf) {
     printf("Reported buffer size from TTreeCache: %d\n", pf->GetBufferSize());
     printf("PrintCacheStats output:\n");
     T->PrintCacheStats("");
     printf("-- end of PrintCacheStats output --\n");
   } else {
     printf("Reading %lld bytes in %d transactions\n",file->GetBytesRead(),  file->GetReadCalls());
     printf("No TTreeCache for other stats\n");
   }
   delete file;

   printf("\n");
   return 0;
}

int runtwo(const Long64_t *cSizep=0) {
   printf("--- Start of subtest %d\n", ++gNum);

   Long64_t n,pfs=0,pfs2=0;
   TTreeCache *pf,*pf2;

   TChain chain("autotree");
   chain.AddFile("autocache1.root");
   chain.AddFile("autocache2.root");
   printf("New TChain with two files, different cluster size, read across boundary\n");

   if (cSizep) {
     printf("-> Calling chain.SetCacheSize(%lld) <-\n", *cSizep);
     chain.SetCacheSize(*cSizep);
   } else {
     printf("No call to chain.SetCacheSize()\n");
   }

   printf("Reported cache size from chain: %lld\n", chain.GetCacheSize());

   Long64_t nentries = chain.GetEntries();
   printf("Chain reports %lld entries\n", nentries);

   printf("Getting last entry from first file in chain\n");
   chain.GetEntry(nentries/2-1);

   printf("Reported AutoFlush from current tree: %lld\n", chain.GetTree()->GetAutoFlush());
   printf("Reported cache size from chain: %lld\n", chain.GetCacheSize());
   printf("Reported cache size from current tree: %lld\n", chain.GetTree()->GetCacheSize());
   pf = dynamic_cast<TTreeCache*>(chain.GetCurrentFile()->GetCacheRead(chain.GetTree()));
   if (pf) {
     pfs = pf->GetBufferSize();
     printf("Reported buffer size from TTreeCache: %lld\n", pfs);
   } else {
     printf("No TTreeCache\n");
   }

   printf("Getting first entry from second file in chain\n");
   chain.GetEntry(nentries/2);

   printf("Reported AutoFlush from current tree: %lld\n", chain.GetTree()->GetAutoFlush());
   printf("Reported cache size from chain: %lld\n", chain.GetCacheSize());
   printf("Reported cache size from current tree: %lld\n", chain.GetTree()->GetCacheSize());
   pf2 = dynamic_cast<TTreeCache*>(chain.GetCurrentFile()->GetCacheRead(chain.GetTree()));
   if (pf2) {
     pfs2 = pf2->GetBufferSize();
     printf("Reported buffer size from TTreeCache: %lld\n", pfs2);
   } else {
     printf("No TTreeCache\n");
   }

   if (pfs == pfs2) {
     if (pf != pf2) {
        printf("TTreeCache has changed memory location\n");
     } else {
        printf("TTreeCache object pointer stayed the same\n");
     }
   } else {
     printf("TTreeCache buffer size changed, object location not checked\n");
   }

   printf("\n");
   return 0;
}

int runthree(Long64_t *cSizep=0) {
   printf("--- Start of subtest %d\n", ++gNum);

   TChain chain("autotree");
   chain.AddFile("autocache1.root");
   chain.AddFile("autocache2.root");

   printf("New TChain with two files, different cluster size, use a selector\n");

   MySelector mys;

   if (cSizep) {
     printf("-> Calling chain.SetCacheSize(%lld) <-\n", *cSizep);
     chain.SetCacheSize(*cSizep);
   } else {
     printf("No call to chain.SetCacheSize()\n");
   }

   Long64_t nentries = chain.GetEntries();

   chain.Process(&mys,"",2,nentries/2-1);

   printf("\n");
   return 0;
}

int runfour() {
   printf("--- Start of subtest %d\n", ++gNum);

   TFile *file=0;
   TTree *T = new TTree;
   printf("Created a new TTree with default constructor\n");
   printf("-> Calling SetCacheSize(1000000) <-\n");
   T->SetCacheSize(1000000);

   TTreeCache *pf = 0;
   file = T->GetCurrentFile();
   if (file) { pf = dynamic_cast<TTreeCache*>(file->GetCacheRead(T)); }

   if (pf) {
     printf("cache exists, tree reports size=%lld\n",T->GetCacheSize());
   } else {
     printf("cache does not exist\n");
   }
   file = TFile::Open( "AliESDs-0.root" );
   if (!file) {
      printf("ERROR: Could not open file\n");
      return 1;
   }
   gROOT->cd();
   printf("Read esdTree into existing TTree object; current working diretory gROOT\n");
   file->ReadTObject(T,"esdTree");
   file->cd();

   printf("Changed working directory back to file, load entry 0\n");
   T->LoadTree(0);
   T->GetEntry(0);

   pf = dynamic_cast<TTreeCache*>(file->GetCacheRead(T));

   if (pf) {
     printf("cache exists, tree reports size=%lld\n",T->GetCacheSize());
   } else {
     printf("cache does not exist\n");
   }

   delete T;
   delete file;
   return 0;
}

int runautocache() {
   printf("Starting runautocache() test\n\n");

   TFile::SetReadaheadSize(256000); 
   TTreeCache::SetLearnEntries(20);
   printf("* Learn entries set to 20, readahead to 256000\n");

   gSystem->Unsetenv("ROOT_TTREECACHE_SIZE");
   gSystem->Unsetenv("ROOT_TTREECACHE_PREFILL");
   printf("* Cleared ROOT_TTREECACHE_SIZE and ROOT_TTREECACHE_PREFILL env variables\n");
   gEnv->SetValue("TTreeCache.Size", 0.0);
   gEnv->SetValue("TTreeCache.Prefill", 0);
   printf("* Set resource variables TTreeCache Size and Prefill to 0.0, 0\n");

   if (checkfree()) return 1;

   if (runone()) return 1;

   Long64_t cacheSize = -1;
   if (runone(&cacheSize)) return 1;

   printf("* Resource variable TTreeCache.Size=0.0\n");
   gEnv->SetValue("TTreeCache.Size", 0.0);
   if (runone()) return 1;

   printf("* Resource variable TTreeCache.Size=0.1\n");
   gEnv->SetValue("TTreeCache.Size", 0.1);
   if (runone()) return 1;

   printf("* Resource variable TTreeCache.Prefill=1\n");
   gEnv->SetValue("TTreeCache.Prefill", 1);
   if (runone()) return 1;

   printf("* Resource variable TTreeCache.Prefill=0 and env variable ROOT_TTREECACHE_PREFILL=1\n");
   gEnv->SetValue("TTreeCache.Prefill", 0);
   gSystem->Setenv("ROOT_TTREECACHE_PREFILL", "1");
   if (runone()) return 1;

   printf("* Cleared env variable ROOT_TTREECACHE_PREFILL\n");
   gSystem->Unsetenv("ROOT_TTREECACHE_PREFILL");
   printf("* Resource variable TTreeCache.Size=0.0, env variable ROOT_TTREECACHE_SIZE=0.2\n");
   gEnv->SetValue("TTreeCache.Size", 0.0);
   gSystem->Setenv("ROOT_TTREECACHE_SIZE","0.2");
   if (runone()) return 1;

   cacheSize = -1;
   if (runone(&cacheSize)) return 1;

   cacheSize=0;
   if (runone(&cacheSize)) return 1;

   cacheSize = 950000;
   if (runone(&cacheSize)) return 1;

   printf("* env variable ROOT_TTREECACHE_SIZE=1.1\n");
   gSystem->Setenv("ROOT_TTREECACHE_SIZE","1.1");

   printf("* Writing two root files, second with AutoFlush 1.15 times the first\n");
   writefiles(1.15);

   if (runtwo()) return 1;

   cacheSize = -1;
   if (runtwo(&cacheSize)) return 1;

   printf("* Writing two root files, second with AutoFlush 1.3 times the first\n");
   writefiles(1.3);

   if (runtwo()) return 1;

   cacheSize = -1;
   if (runtwo(&cacheSize)) return 1;

   cacheSize = 0;
   if (runtwo(&cacheSize)) return 1;

   cacheSize = 1500000;
   if (runtwo(&cacheSize)) return 1;

   printf("* Cleared env variable ROOT_TTREECACHE_SIZE\n");
   gSystem->Unsetenv("ROOT_TTREECACHE_SIZE");
   if (runthree()) return 1;

   printf("* Resource variable TTreeCache.Size=1.5 ROOT_TTREECACHE_SIZE not set\n");
   gEnv->SetValue("TTreeCache.Size", 1.0);
   if (runthree()) return 1;

   printf("* env variable ROOT_TTREECACHE_SIZE=1.1\n");
   gSystem->Setenv("ROOT_TTREECACHE_SIZE","1.1");
   if (runthree()) return 1;

   cacheSize = -1;
   if (runthree(&cacheSize)) return 1;

   cacheSize = 0;
   if (runthree(&cacheSize)) return 1;

   cacheSize = 1500000;
   if (runthree(&cacheSize)) return 1;

   if (runfour()) return 1;

   return 0;
}
