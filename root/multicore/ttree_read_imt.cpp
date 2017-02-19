#include "ROOT/TThreadExecutor.hxx"
#include "TBranch.h"
#include "TFile.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TTree.h"

#include <iostream>


void printHelp(const char* iName, int nThreads, int nEntries)
{
  std::cout << iName << " [number of threads] [filename] [gDebug value]\n\n"
            << "[number of threads] number of threads to use\n"
            << "[filename] name of the tree file\n"
            << "[number of entries] number of entries to read\n"
            << "If no arguments are given " << nThreads << " threads, " << nEntries << " entries will be used." << std::endl;
}

const std::string kDefaultFileName("ttree_read_imt.root");

std::tuple<int,int,std::string> parseOptions(int argc, char** argv)
{
  constexpr int kDefaultNThreads = 4;
  constexpr int kDefaultNEntries = 1000;

  int nThreads = kDefaultNThreads;
  int nEntries = kDefaultNEntries;
  std::string fileName(kDefaultFileName);

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
  if (argc == 4) {
    fileName = argv[3];
  }
  if (argc > 4) {
    printHelp(argv[0], kDefaultNThreads, kDefaultNEntries);
    exit(1);
  }

  return std::make_tuple(nThreads, nEntries, fileName);
}


Int_t ReadTree(TTree *tree, Int_t nentries)
{
  Int_t nb = 0;
  for (Long64_t i = 0; i < nentries; ++i) {
    nb += tree->GetEntry(i);
  }

  return nb;
}


int main(int argc, char** argv) {

  auto options = parseOptions(argc, argv);

  const int nthreads  = std::get<0>(options);
  const int nentries  = std::get<1>(options);
  auto const filename = std::get<2>(options);

  gSystem->Load("generate_imt_tree_C.so");

  TFile *file = TFile::Open(filename.c_str());
  
  Int_t nbreadseq, nbreadseq2, nbreadpar, nbreadpar2;

  // First enable implicit multi-threading globally, specifying the number of threads to use 
  ROOT::EnableImplicitMT(nthreads);

  // Dummy pool which prevents the counter of subscribers to the pool to go to zero.
  ROOT::TThreadExecutor pool(nthreads);

  // Create the tree (local IMT is initialised to global, i.e. true)
  TTree *tree = (TTree*)file->GetObjectChecked("TreeIMT", "TTree");
  // ...and read in parallel
  nbreadpar = ReadTree(tree, nentries);

  // Disable IMT only for this tree
  tree->SetImplicitMT(false);
  // ...and read again, in sequential mode
  nbreadseq = ReadTree(tree, nentries);

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
  nbreadseq2 = ReadTree(tree, nentries);

  // Re-enable the global IMT
  ROOT::EnableImplicitMT();
  // ...and now we can read in parallel
  nbreadpar2 = ReadTree(tree, nentries);
 
  // Check that the number of bytes read is the same
  if (nbreadseq2 == nbreadpar2) std::cout << "SUCCESS";
  else                          std::cout << "ERROR";
  std::cout << " [IMT] - Bytes sequential2: " << nbreadseq2 << " - parallel2: " << nbreadpar2 << std::endl;

  return 0;
}

