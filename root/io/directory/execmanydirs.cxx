#include "TFile.h"
#include <vector>
#include <string>
#include "TROOT.h"
#include "TH1.h"

void execmanydirs(int n = 200, int mode = 16) {

  std::vector<std::string> samples;
  std::vector<std::string> systematics;

  for (int i = 0; i < n; ++i) {
    samples.emplace_back("sample_" + std::to_string(i));
    systematics.emplace_back("syst_" + std::to_string(i));
  }
  // systematics.emplace_back("syst_" + std::to_string(10));

  std::unique_ptr<TFile> out(TFile::Open("example.root", "RECREATE"));

  for (const auto& isample : samples) {
     for (const auto& isystematic : systematics) {
        if (mode == 0) 
          out->mkdir((isample+"/"+isystematic).c_str());
        else if (mode & 8) 
          out->mkdir((isample+"-"+isystematic).c_str());
        else if (mode&1) {
          TDirectory *mid = out->mkdir(isample.c_str());
          TDirectory *low = mid->mkdir(isystematic.c_str());
        }
        if (mode & 16) {
          static int count = 0;
          new TH1F(TString::Format("h%d", count), "h", 100, 0, 100);
          ++count;
        }
        if (mode&2)
          out->rmdir((isample+"/"+isystematic).c_str());
     }
  }

  out->Write();
  if (mode & 4) {
    gROOT->GetListOfFiles()->Remove(out.get());
    out.release();
  }
}
