#include "TFile.h"
#include "TList.h"
#include "TTree.h"

int todevnull(int alg, int level) {
  unsigned int entries(100u);
  TFile* orgfile = TFile::Open("org.root");
  TList* thelist = orgfile->GetListOfKeys();
  TObjLink* thelink = thelist->FirstLink();
  while (thelink) {
    if (orgfile->Get(thelink->GetObject()->GetName())->InheritsFrom("TTree")) {
      TTree* t;
      orgfile->GetObject(thelink->GetObject()->GetName(),t);
      // NB: not suited for wall clock time measurements due to cache not warmed up!
      {
        TFile* pointer = new TFile(Form("/dev/null"),"recreate","",alg*100+level);
        pointer->WriteTObject(t->CloneTree(entries));
        pointer->Close();
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
  return todevnull(atoi(argv[1]),atoi(argv[2]));
}
