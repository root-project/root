#include <iostream>
#include <ctime>
#include <sys/time.h>

#include "TFile.h"
#include "TTree.h"
#include "TTreeReader.h"
#include "ROOT/TTreeReaderFast.hxx"
#include "ROOT/TTreeReaderValueFast.hxx"

const int WARM_UP = 5;
const int REPS = 100;

double diff(struct timeval endTime, struct timeval startTime) {
  return (1000L * 1000L * (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_usec - startTime.tv_usec)) / 1000.0 / 1000.0;
}

double setBranchAddress_momentum(TTree *tree, int reps) {
  double total = 0.0;

  Float_t px;
  Float_t py;
  Float_t pz;
  tree->SetBranchAddress("px", &px);
  tree->SetBranchAddress("py", &py);
  tree->SetBranchAddress("pz", &pz);

  Long64_t numEntries = tree->GetEntries();

  for (int i = 0;  i < reps;  i++) {
    for (Long64_t entry = 0;  entry < numEntries;  entry++) {
      tree->GetEntry(entry);
      total += sqrt(px*px + py*py + pz*pz);
    }
  }

  return total;
}

double ttreeReaderFast_momentum(TTree *tree, int reps) {
  double total = 0.0;

  for (int i = 0;  i < 1;  i++) {
    ROOT::Experimental::TTreeReaderFast reader(tree);
    ROOT::Experimental::TTreeReaderValueFast<float> px(reader, "px");
    ROOT::Experimental::TTreeReaderValueFast<float> py(reader, "py");
    ROOT::Experimental::TTreeReaderValueFast<float> pz(reader, "pz");

    for (auto it = reader.begin();  it != reader.end();  ++it) {
        total += sqrt((*px)*(*px) + (*py)*(*py) + (*pz)*(*pz));
    }
  }

  return total;
}

double ttreeReader_momentum(TTree *tree, int reps) {
  double total = 0.0;

  TTreeReader reader(tree);
  TTreeReaderValue<float> px(reader, "px");
  TTreeReaderValue<float> py(reader, "py");
  TTreeReaderValue<float> pz(reader, "pz");

  for (int i = 0;  i < reps;  i++) {
    reader.Restart();
    while (reader.Next()) {
      total += sqrt((*px)*(*px) + (*py)*(*py) + (*pz)*(*pz));
    }
  }

  return total;
}

void test_performance() {
  {
    TFile* file = TFile::Open("TrackResonanceNtuple_uncompressed.root");
    TTree* tree;
    file->GetObject("twoMuon", tree);
    tree->SetBranchStatus("*", 0);
    tree->SetBranchStatus("px", 1);
    tree->SetBranchStatus("py", 1);
    tree->SetBranchStatus("pz", 1);

    ttreeReader_momentum(tree, WARM_UP);

    struct timeval startTime, endTime;
    gettimeofday(&startTime, 0);
    ttreeReader_momentum(tree, REPS);
    gettimeofday(&endTime, 0);

    std::cout << "TrackResonanceNtuple_uncompressed.root TTreeReader " << diff(endTime, startTime) << " sec" << std::endl;
  }

  {
    TFile* file = TFile::Open("TrackResonanceNtuple_compressed.root");
    TTree* tree;
    file->GetObject("twoMuon", tree);
    tree->SetBranchStatus("*", 0);
    tree->SetBranchStatus("px", 1);
    tree->SetBranchStatus("py", 1);
    tree->SetBranchStatus("pz", 1);

    ttreeReader_momentum(tree, WARM_UP);

    struct timeval startTime, endTime;
    gettimeofday(&startTime, 0);
    ttreeReader_momentum(tree, REPS);
    gettimeofday(&endTime, 0);

    std::cout << "TrackResonanceNtuple_compressed.root TTreeReader " << diff(endTime, startTime) << " sec" << std::endl;
  }

  {
    TFile* file = TFile::Open("TrackResonanceNtuple_uncompressed.root");
    TTree* tree;
    file->GetObject("twoMuon", tree);
    tree->SetBranchStatus("*", 0);
    tree->SetBranchStatus("px", 1);
    tree->SetBranchStatus("py", 1);
    tree->SetBranchStatus("pz", 1);

    setBranchAddress_momentum(tree, WARM_UP);

    struct timeval startTime, endTime;
    gettimeofday(&startTime, 0);
    setBranchAddress_momentum(tree, REPS);
    gettimeofday(&endTime, 0);

    std::cout << "TrackResonanceNtuple_uncompressed.root SetBranchAddress " << diff(endTime, startTime) << " sec" << std::endl;
  }

  {
    TFile* file = TFile::Open("TrackResonanceNtuple_compressed.root");
    TTree* tree;
    file->GetObject("twoMuon", tree);
    tree->SetBranchStatus("*", 0);
    tree->SetBranchStatus("px", 1);
    tree->SetBranchStatus("py", 1);
    tree->SetBranchStatus("pz", 1);

    setBranchAddress_momentum(tree, WARM_UP);

    struct timeval startTime, endTime;
    gettimeofday(&startTime, 0);
    setBranchAddress_momentum(tree, REPS);
    gettimeofday(&endTime, 0);

    std::cout << "TrackResonanceNtuple_compressed.root SetBranchAddress " << diff(endTime, startTime) << " sec" << std::endl;
  }

  // {
  //   TFile* file = TFile::Open("TrackResonanceNtuple_uncompressed.root");
  //   TTree* tree;
  //   file->GetObject("twoMuon", tree);
  //   tree->SetBranchStatus("*", 0);
  //   tree->SetBranchStatus("px", 1);
  //   tree->SetBranchStatus("py", 1);
  //   tree->SetBranchStatus("pz", 1);

  //   ttreeReaderFast_momentum(tree, WARM_UP);

  //   struct timeval startTime, endTime;
  //   gettimeofday(&startTime, 0);
  //   ttreeReaderFast_momentum(tree, REPS);
  //   gettimeofday(&endTime, 0);

  //   std::cout << "TrackResonanceNtuple_uncompressed.root TTreeReader " << diff(endTime, startTime) << " sec" << std::endl;
  // }

  // {
  //   TFile* file = TFile::Open("TrackResonanceNtuple_compressed.root");
  //   TTree* tree;
  //   file->GetObject("twoMuon", tree);
  //   tree->SetBranchStatus("*", 0);
  //   tree->SetBranchStatus("px", 1);
  //   tree->SetBranchStatus("py", 1);
  //   tree->SetBranchStatus("pz", 1);

  //   ttreeReaderFast_momentum(tree, WARM_UP);

  //   struct timeval startTime, endTime;
  //   gettimeofday(&startTime, 0);
  //   ttreeReaderFast_momentum(tree, REPS);
  //   gettimeofday(&endTime, 0);

  //   std::cout << "TrackResonanceNtuple_compressed.root TTreeReader " << diff(endTime, startTime) << " sec" << std::endl;
  // }

}
