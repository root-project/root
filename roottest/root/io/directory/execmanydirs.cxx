#include "TFile.h"
#include <vector>
#include <string>
#include "TROOT.h"
#include "TH1.h"

// Mode ( bits )
//   0 : i.e with 1 nor 8: create nested directory in a single operation
//   1 : create nested directory per sample and systematic (if not 8) in 2 operations
//   2 : delete the directory from the file
//   8 ; create one directory per sample and systematic
//  16 : Add an histogram to the directory
//  32 : if 16, see below
//  64 : (if 16 & 32) reset the kMustCleanup bit of the histo
// 128 : (if 16 & 32) write and delete the histogram
// 256 : Remove the file from the list of files (guaranteed fast)
//
// At the moment without the histo (i.e without 16), the process is fast
// with 16 and without 64 or 128, the process is very slow (scale with the
// square of number of histo)
// With 128 it is still a bit slow.
//
// 64 is necessary to avoid scanning all the directories when deleting this histogram
void execmanydirs(int n = 200, int mode = 16 | 32 | 64) {

  std::vector<std::string> samples;
  std::vector<std::string> systematics;

  for (int i = 0; i < n; ++i) {
    samples.emplace_back("sample_" + std::to_string(i));
    systematics.emplace_back("syst_" + std::to_string(i));
  }
  // systematics.emplace_back("syst_" + std::to_string(10));

  std::unique_ptr<TFile> out(TFile::Open("example_manydirs.root", "RECREATE"));
  if (mode & 256)
    gROOT->GetListOfFiles()->Remove(out.get());

  for (const auto& isample : samples) {
     for (const auto& isystematic : systematics) {
        TDirectory *target = nullptr;
        if (mode & 8)
          target = out->mkdir((isample+"-"+isystematic).c_str());
        else if (mode&1) {
          TDirectory *mid = out->mkdir(isample.c_str());
          TDirectory *low = mid->mkdir(isystematic.c_str());
          target = low;
        } else {
          target = out->mkdir((isample+"/"+isystematic).c_str());
        }
        if (mode & 16) {
          target->cd();
          static int count = 0;
          auto h = new TH1F(TString::Format("h%d", count), "h", 100, 0, 100);
          ++count;
          if (mode & 32) {
            if (mode & 128)
              h->Write();
            if (mode & 64)
              h->ResetBit(kMustCleanup); // This is necessary to avoid scanning all the directories when deleting this histogram
            if (mode & 128)
              delete h;
          }
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
