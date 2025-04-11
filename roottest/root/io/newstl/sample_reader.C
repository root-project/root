#if defined(__CLING__) && !defined(__MAKECLING__) && !defined(ClingWorkAroundMissingSmartInclude)
#include "sample_bx_classes.C+"
#else
#include "sample_bx_classes.C"
#endif

#include "TFile.h"
#include "TTree.h"
#include <iostream>
#include <vector>

void sample_reader() 
{
  TFile f("samplerootfile.root");
  TTree* bxtree = 0;
  f.GetObject("bxtree", bxtree);
  BxEvent* e = 0;
  bxtree->SetBranchAddress("events", &e);
  for (int i = 0; i < bxtree->GetEntries(); ++i) {
     std::cout << "\nIn event: " << i << std::endl;
     bxtree->GetEntry(i);
     const std::vector<BxLabenCluster>& clusters(e->GetLaben().GetClusters());
     std::cout << " Size of clusters vector: " << clusters.size() << std::endl;
     if (clusters.size() > 0) {
        const std::vector<Float_t>& timesXXX(clusters[0].GetTimes());
        std::cout << " Size of \"times\" vector in 1st cluster: " << timesXXX.size() << std::endl;
     }
  }
  // If we do not break the branch's connection
  // with "e" here, then by the time the branch
  // destructor is called, "e" will no longer
  // exist and the branch will be pointing at
  // deallocated memory.
  bxtree->SetBranchAddress("events", 0);
}

