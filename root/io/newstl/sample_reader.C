#include <vector>
#if defined(__CINT__) && !defined(__MAKECINT__)
#include "sample_bx_classes.C+"
#else
#include "sample_bx_classes.C"
#endif
#include <iostream>
#include "TFile.h"
#include "TTree.h"

void sample_reader() 
{
#ifdef __CINT__
  gROOT->ProcessLine(".L sample_bx_classes.C+");
#endif

  TFile f("samplerootfile.root");
  TTree *bxtree; f.GetObject("bxtree",bxtree);

  BxEvent *e;
  bxtree->SetBranchAddress("events", &e);
  for (int i = 0; i < bxtree->GetEntries(); i++) {
     std::cout << "\nIn event: " << i << std::endl;
     bxtree->GetEntry(i);
     
     const std::vector<BxLabenCluster> & clusters( e->GetLaben().GetClusters() );
     std::cout << " Size of clusters vector: " << clusters.size() << std::endl;
     if (clusters.size()> 0) {
        const std::vector<Float_t>& timesXXX ( clusters[0].GetTimes() );
        std::cout << " Size of \"times\" vector in 1st cluster: " << timesXXX.size() << std::endl;
     }
  }
}
