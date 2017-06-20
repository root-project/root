#include <iostream>
#include <ctime>
#include <sys/time.h>

#include "TFile.h"
#include "TTree.h"
#include "TTreeReader.h"

double diff(struct timeval endTime, struct timeval startTime) {
  return (1000L * 1000L * (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_usec - startTime.tv_usec)) / 1000.0 / 1000.0;
}

double ttreeReader_momentum(TTree *tree, int reps) {
  double total = 0.0;

  for (int i = 0;  i < reps;  i++) {
    TTreeReader reader(tree);
    TTreeReaderValue<float> px(reader, "px");
    TTreeReaderValue<float> py(reader, "py");
    TTreeReaderValue<float> pz(reader, "pz");

    while (reader.Next()) {
      total += sqrt((*px)*(*px) + (*py)*(*py) + (*pz)*(*pz));
    }
  }

  return total;
}

void test_performance() {
  TFile* file = TFile::Open("TrackResonanceNtuple.root");
  TTree* tree;
  file->GetObject("TrackResonanceNtuple/twoMuon", tree);

  ttreeReader_momentum(tree, 5);

  struct timeval startTime, endTime;
  gettimeofday(&startTime, 0);
  ttreeReader_momentum(tree, 100);
  gettimeofday(&endTime, 0);

  std::cout << diff(endTime, startTime) << " sec" << std::endl;
}
