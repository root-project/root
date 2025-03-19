#include "ROOT/TTreeProcessorMP.hxx"

#include "TString.h"
#include "TROOT.h"
#include "TTree.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TEntryList.h"
#include "TTreeReader.h"
#include "TTreeReaderArray.h"
#include "TTreeReaderValue.h"
#include "TSystem.h"
#include "TMath.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TF1.h"
#include "TLine.h"
#include "TPaveStats.h"

#include <memory>

static std::string logfile = "tProcessExecutorH1Test-pool.log";
static RedirectHandle_t gRH;

const char *fh1[] = {"root://eospublic.cern.ch//eos/root-eos/h1/dstarmb.root",
                     "root://eospublic.cern.ch//eos/root-eos/h1/dstarp1a.root",
                     "root://eospublic.cern.ch//eos/root-eos/h1/dstarp1b.root",
                     "root://eospublic.cern.ch//eos/root-eos/h1/dstarp2.root"};

int main() {
   
   std::cout << "+++ Executing tProcessExecutorH1Test ......................................... +++\n";

   // MacOSX may generate connection to WindowServer errors
   gROOT->SetBatch(kTRUE);

   // Prepare dataset: vector of files
   std::vector<std::string> files;
   for (int i = 0; i < 4; i++) {
      files.push_back(fh1[i]);
   }

   // Check and fit lambdas
#include "mp_H1_lambdas.C"

   ROOT::TTreeProcessorMP pool(3);

   //
   // Processing the H1 dataset with a lambda
   gSystem->RedirectOutput(logfile.c_str(), "w", &gRH);
   auto hListFun = pool.Process(files, doH1, "h42");
   gSystem->RedirectOutput(0, 0, &gRH);
   // Check the output
   if (checkH1(hListFun) < 0) {
      std::cout << "+++ Processing the H1 dataset with a lambda .................................. failed \n";
      return -1;
   }
   // Do the fit
   if (doFit(hListFun, logfile.c_str()) < 0) {
      std::cout << "+++ Processing the H1 dataset with a lambda .................................. failed \n";
      return -1;
   }
   std::cout << "+++ Processing the H1 dataset with a lambda .................................. OK \n";

   //
   // Processing the H1 dataset with a selector 
   TString selectorPath = gROOT->GetTutorialDir();
   selectorPath += "/analysis/tree/h1analysisTreeReader.C+";
   gSystem->RedirectOutput(logfile.c_str(), "a", &gRH);
   auto *sel = TSelector::GetSelector(selectorPath);
   auto hListSel = pool.Process(files, *sel, "h42");
   gSystem->RedirectOutput(0, 0, &gRH);
   // Check the output
   if (checkH1(hListSel) < 0) {
      std::cout << "+++ Processing the H1 dataset with h1analysisTreeReader ...................... failed \n";
      return -1;
   }
   // Do the fit
   if (doFit(hListSel, logfile.c_str()) < 0) {
      std::cout << "+++ Processing the H1 dataset with h1analysisTreeReader ...................... failed \n";
      return -1;
   }
   std::cout << "+++ Processing the H1 dataset with h1analysisTreeReader ...................... OK \n";

   //
   // Processing the H1 dataset with a lambda to create an entry list
   gSystem->RedirectOutput(logfile.c_str(), "a", &gRH);
   auto sumElist = pool.Process(files, doH1fillList, "h42");
   gSystem->RedirectOutput(0, 0, &gRH);
   if (!sumElist || sumElist->GetN() != 7525) {
      std::cout << "+++ Processing the H1 dataset to create list ................................. failed \n";
      return -1;
   }
   std::cout << "+++ Processing the H1 dataset to create list ................................. OK \n";

   //
   // Processing the H1 dataset with a lambda using an entrylist
   gSystem->RedirectOutput(logfile.c_str(), "a", &gRH);
   auto hListFunEList = pool.Process(files, doH1useList, *sumElist, "h42");
   gSystem->RedirectOutput(0, 0, &gRH);
   // Check the output
   if (checkH1(hListFunEList) < 0) {
      std::cout << "+++ Processing the H1 dataset with a lambda from entry list .................. failed \n";
      return -1;
   }
   // Do the fit
   if (doFit(hListFunEList, logfile.c_str()) < 0) {
      std::cout << "+++ Processing the H1 dataset with a lambda from entry list .................. failed \n";
      return -1;
   }
   std::cout << "+++ Processing the H1 dataset with a lambda from entry list .................. OK \n";

   //
   // Processing the H1 dataset with a selector using an entrylist
   gSystem->RedirectOutput(logfile.c_str(), "a", &gRH);
   auto *selEl = TSelector::GetSelector(selectorPath);
   selEl->SetOption("useList");
   auto hListSelEList = pool.Process(files, *selEl, *sumElist, "h42");
   gSystem->RedirectOutput(0, 0, &gRH);
   // Check the output
   if (checkH1(hListSelEList) < 0) {
      std::cout << "+++ Processing the H1 dataset with h1analysisTreeReader from entry list ...... failed \n";
      return -1;
   }
   // Do the fit
   if (doFit(hListSelEList, logfile.c_str()) < 0) {
      std::cout << "+++ Processing the H1 dataset with h1analysisTreeReader from entry list ...... failed \n";
      return -1;
   }
   std::cout << "+++ Processing the H1 dataset with h1analysisTreeReader from entry list ...... OK \n";

   return 0;
}

