#include "TString.h"
#include "TROOT.h"
#include "TTree.h"
#include "TH1F.h"
#include "TTreeReader.h"
#include "TProcPool.h"

const char *fh1[] = {"http://root.cern.ch/files/h1/dstarmb.root",
                     "http://root.cern.ch/files/h1/dstarp1a.root",
                     "http://root.cern.ch/files/h1/dstarp1b.root",
                     "http://root.cern.ch/files/h1/dstarp2.root"};

int processSelector(){
  // MacOSX may generate connection to WindowServer errors
  gROOT->SetBatch(kTRUE);

  TString selectorPath = gROOT->GetTutorialsDir();
  selectorPath += "/tree/h1analysis.C+";
  std::cout<<"selector used is: "<< selectorPath<<"\n";
  TSelector *sel = TSelector::GetSelector(selectorPath);

  TFile *fp = TFile::Open(fh1[0]);
	TTree *tree = (TTree *) fp->Get("h42");

  TProcPool pool(3);

  //TProcPool::Process with a single tree
  TList* out = pool.ProcTree(*tree, *sel);;

  //TProcPool::Process with single file name and tree name
  //Note: we have less files than workers here
  out = pool.ProcTree(fh1[0], *sel, "h1");
  sel->GetOutputList()->Delete();
  //TProcPool::Process with vector of files and tree name
  //Note: we have more files than workers here (different behaviour)
  std::vector<std::string> files;
  for (int i = 0; i < 4; i++) {
     files.push_back(fh1[i]);
  }
  out = pool.ProcTree(files, *sel, "h1");
  sel->GetOutputList()->Delete();

  //TProcPool::Process with single file name, tree name and entries limit
  out = pool.ProcTree(fh1[0], *sel, "h1", 42);
  sel->GetOutputList()->Delete();

  //TProcPool::Process with vector of files, no tree name and no entries limit
  out = pool.ProcTree(files, *sel);
  sel->GetOutputList()->Delete();

  //TProcPool::Process with TFileCollection, no tree name and entries limit
  TFileCollection fc;
  fc.Add(new TFileInfo(fh1[0]));
  out = pool.ProcTree(fc, *sel, "", 42);
  sel->GetOutputList()->Delete();

  //TProcPool::Process with TChain, no tree name and no entries limit
  TChain c;
  c.Add(fh1[0]);
  c.Add(fh1[1]);
  out = pool.ProcTree(c, *sel);
  sel->GetOutputList()->Delete();

  return 0;
}
