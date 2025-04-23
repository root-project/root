#include "ROOT/TThreadExecutor.hxx"
#include "TBranch.h"
#include "TFile.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TTree.h"
#include "TTreeCache.h"

#include <iostream>


const std::string kDefaultFileName("./ttree_read_imt.root");

void printHelp(const char* iName, int nThreads, int nEntries)
{
  std::cout << iName << " [number of threads] [number of entries] [filename] [CacheReuse]\n\n"
            << "[number of threads] number of threads to use, default " << nThreads << "\n"
            << "[number of entries] number of entries to read, default " << nEntries << "\n"
            << "[filename] name of the tree file, default \"" << kDefaultFileName << "\"\n"
            << "[CacheReuse] Reuse (1) or reload (0) the TTreeCache at each iteration, default reuse"
            << "If no arguments are given " << nThreads << " threads, " << nEntries << " entries will be used." << std::endl;
}

std::tuple<int,int,std::string,bool> parseOptions(int argc, char** argv)
{
  constexpr int kDefaultNThreads = 4;
  constexpr int kDefaultNEntries = 1000;
  constexpr bool kDefaultReuse = true;

  int nThreads = kDefaultNThreads;
  int nEntries = kDefaultNEntries;
  std::string fileName(kDefaultFileName);
  bool reuse = kDefaultReuse;

  if (argc >= 2) {
    if (strcmp("-h", argv[1]) == 0) {
      printHelp(argv[0], kDefaultNThreads, kDefaultNEntries);
      exit(0);
    }
    nThreads = atoi(argv[1]);
  }
  if (argc >= 3) {
    nEntries = atoi(argv[2]);
  }
  if (argc >= 4) {
    fileName = argv[3];
  }
  if (argc >= 5) {
    reuse = atoi(argv[4]);
  }
  if (argc > 5) {
    printHelp(argv[0], kDefaultNThreads, kDefaultNEntries);
    exit(1);
  }

  return std::make_tuple(nThreads, nEntries, fileName, reuse);
}


Int_t ReadTree(TTree *tree, Int_t nentries, bool reuse)
{
  if (!reuse) {
     auto cache = tree->GetReadCache(tree->GetDirectory()->GetFile(), kTRUE);
     if (cache)
        cache->ResetCache();
     // If the '2nd' (or later) basket is in memory, the redo of the cache
     // will skip but when the '1st' basket is requested, the '2nd' will get
     // evicted and later when the '2nd' is request, it wont be in the cache ...
     tree->DropBaskets();
  }
  tree->SetCacheEntryRange(0, nentries);
  tree->AddBranchToCache("*", kTRUE);

  Int_t nb = 0;
  for (Long64_t i = 0; i < nentries; ++i) {
     tree->LoadTree(i);
     nb += tree->GetEntry(i);
  }

  return nb;
}


int main(int argc, char** argv) {

  auto options = parseOptions(argc, argv);

  const int nthreads  = std::get<0>(options);
  const int nentries  = std::get<1>(options);
  auto const filename = std::get<2>(options);
  const bool reuse = std::get<3>(options);

  TFile *file = TFile::Open(filename.c_str());
  if (!file || file->IsZombie())
    return 1;
  
  Int_t nbreadseq, nbreadseq2, nbreadpar, nbreadpar2;

  // First enable implicit multi-threading globally, specifying the number of threads to use 
  ROOT::EnableImplicitMT(nthreads);

  // Dummy pool which prevents the counter of subscribers to the pool to go to zero.
  ROOT::TThreadExecutor pool(nthreads);

  // Create the tree (local IMT is initialised to global, i.e. true)
  TTree *tree = (TTree*)file->GetObjectChecked("TreeIMT", "TTree");
  // ...and read in parallel
  nbreadpar = ReadTree(tree, nentries, reuse);

  // Disable IMT only for this tree
  tree->SetImplicitMT(false);
  // ...and read again, in sequential mode
  nbreadseq = ReadTree(tree, nentries, reuse);

  // Check that the number of bytes read is the same
  if (nbreadseq == nbreadpar) std::cout << "SUCCESS";
  else                        std::cout << "ERROR";
  std::cout << " [IMT] - Bytes sequential1: " << nbreadseq << " - parallel1: " << nbreadpar << std::endl;

  // Now disable IMT globally
  ROOT::DisableImplicitMT();
  //...and check the value of the flag
  if (!ROOT::IsImplicitMTEnabled()) std::cout << "SUCCESS";
  else                              std::cout << "ERROR";
  std::cout << " [IMT] - Checked state of global IMT" << std::endl;
 
  // Re-enable IMT for this tree
  tree->SetImplicitMT(true);
  // ...and read again sequentially, since the global setting dominates
  nbreadseq2 = ReadTree(tree, nentries, reuse);

  // Re-enable the global IMT
  ROOT::EnableImplicitMT();
  // ...and now we can read in parallel
  nbreadpar2 = ReadTree(tree, nentries, reuse);
 
  // Check that the number of bytes read is the same
  if (nbreadseq2 == nbreadpar2) std::cout << "SUCCESS";
  else                          std::cout << "ERROR";
  std::cout << " [IMT] - Bytes sequential2: " << nbreadseq2 << " - parallel2: " << nbreadpar2 << std::endl;

  return 0;
}

