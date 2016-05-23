#include "TFile.h"
#include <stdio.h>
#include "TList.h"
#include "TTree.h"
#include "TStopwatch.h"

int writetime(int alg, int level) {
  TStopwatch watch;
  unsigned int entries(1000u);
  TFile* orgfile = TFile::Open("org.root");
  TList* thelist = orgfile->GetListOfKeys();
  TObjLink* thelink = thelist->FirstLink();
  while (thelink) {
    if (orgfile->Get(thelink->GetObject()->GetName())->InheritsFrom("TTree")) {
      TTree* t;
      orgfile->GetObject(thelink->GetObject()->GetName(),t);
      // warm up the cache
      TFile* uncompressed = new TFile("/dev/null","recreate","",0);
      uncompressed->WriteTObject(t->CloneTree(entries));
      uncompressed->Close();
      delete uncompressed;
      uncompressed=NULL;
      {
        TFile* pointer = new TFile(Form("/dev/null"),"recreate","",alg*100+level);
        watch.Start();
        pointer->WriteTObject(t->CloneTree(entries));
        pointer->Close();
        watch.Stop();
        printf("wall clock time: %f\n",watch.RealTime());
        delete pointer;
      }
      return 0;
      break;
    }
    thelink = thelink->Next();
  }
  return 1;
}

int main(int argc, char** argv) {
  return writetime(atoi(argv[1]),atoi(argv[2]));
}
