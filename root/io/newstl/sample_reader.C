{
#include <vector>
gROOT->ProcessLine(".L sample_bx_classes.C+");

TFile f("samplerootfile.root");

BxEvent *e;
bxtree->SetBranchAddress("events", &e);
for (int i = 0; i < bxtree->GetEntries(); i++) {
  std::cout << "\nIn event: " << i << std::endl;
  bxtree->GetEntry(i);
  
  std::vector<BxLabenCluster>& clusters = e->GetLaben().GetClusters();
  std::cout << " Size of clusters vector: " << clusters.size() << std::endl;
  if (clusters.size()> 0) {
    std::vector<Float_t>& times = clusters[0].GetTimes();
    std::cout << " Size of \"times\" vector in 1st cluster: " << times.size() << std::endl;
  }
}

}
