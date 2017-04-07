#include "TFile.h"
#include "TH1F.h"
#include "TRandom.h"
#include "TSystem.h"
#include <sstream>
#include <vector>

#define NFILES 8

void generateHistoFile(const char *fileName) {
  TH1F *h1 = new TH1F("h1f", "histogram", 100, -4, 4);
  gRandom->SetSeed(0);
  h1->FillRandom("gaus", 1000000);
  TFile *f = new TFile(fileName, "RECREATE");
  h1->Write();
  f->Close();
}

int testHadd() {
  std::vector<const char *> mergedFiles = {"mergedSeq.root", "mergedMP.root"};
  std::vector<std::string> sourceFiles;
  for (auto i = 0; i < NFILES; i++) {
    std::stringstream buffer;
    buffer << "histo" << i << ".root";
    sourceFiles.emplace_back(buffer.str());
  }

  for (auto sf : sourceFiles)
    generateHistoFile(sf.c_str());

  system("hadd -ff -f mergedSeq.root histo*.root");
  system("hadd -j 2 -ff -f mergedMP.root histo*.root");

  TFile *f1 = new TFile(mergedFiles[0]);
  TFile *f2 = new TFile(mergedFiles[1]);
  TH1F *h1 = (TH1F *)f1->Get("h1f");
  TH1F *h2 = (TH1F *)f2->Get("h1f");

  auto h1Nbins = h1->GetXaxis()->GetNbins();
  if (h1Nbins != h2->GetXaxis()->GetNbins())
    return 1;

  for (auto i = 0; i < h1->GetXaxis()->GetNbins(); i++) {
    if (h1->GetBinContent(i) != h2->GetBinContent(i))
      return 2;
  }

  for (auto sf : sourceFiles)
    gSystem->Unlink(sf.c_str());

  for (auto mf : mergedFiles)
    gSystem->Unlink(mf);

  return 0;
}
