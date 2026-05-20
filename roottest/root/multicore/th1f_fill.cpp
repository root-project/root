#include <cmath>
#include <vector>
#include <thread>
#include <map>

#include "TH1F.h"
#include "TFile.h"
#include "TNtuple.h"
#include "TInterpreter.h"
#include "TROOT.h"

void fillHisto(const char* filename, TH1F& histo){
  // do not print name from thread - it may appear in random order
  // printf("Reading file %s\n",filename);
  TFile f(filename);
  TNtuple* ntuple;
  f.GetObject("ntuple", ntuple);
  if (!ntuple) printf ("Cannot read ntuple from file %s\n", filename);
  //printf("Num entries: %d\n", ntuple->GetEntries());
  for (int i = 0; i<ntuple->GetEntries(); ++i){
    ntuple->GetEvent(i);
    auto args = ntuple->GetArgs(); // px py pz random i
    auto px = args[0];
    auto py = args[1];
    histo.Fill(sqrt(px*px + py*py));
    //printf("Filled %d\n", i);
  }
}

int main() {

  // Initialize ROOT threading support
   ROOT::EnableThreadSafety();

  // Don't link histos to a particular TDirectory
  TH1::AddDirectory(false);

  // Suppose to have 3 files
  std::vector<std::pair<const char*, TH1F>> filenamesHistoPairs  {{"file1.root",TH1F("pt","pt",100,0,10)},
                                                                  {"file2.root",TH1F("pt","pt",100,0,10)},
                                                                  {"file3.root",TH1F("pt","pt",100,0,10)}};
  // Fire the threads and store them
  std::vector<std::thread> threads;

  for (auto&& filenameHistoPair : filenamesHistoPairs) {
    auto filename = filenameHistoPair.first;
    auto& histo = filenameHistoPair.second;
    threads.emplace_back(std::thread(fillHisto,filename,std::ref(histo)));
  }

  // Collect them
  for (auto&& thr : threads){
    thr.join();
  }

  // Print Stats
  for (auto&& filenameHistoPair : filenamesHistoPairs) {
    printf("\n--------Histogram %s\n",filenameHistoPair.first);
    //filenameHistoPair.second.Dump();
    printf("- Num entries: %f\n",filenameHistoPair.second.GetEntries());
    printf("- Mean: %f\n",filenameHistoPair.second.GetMean());
    printf("- RMS: %f\n",filenameHistoPair.second.GetRMS());
  }

}

