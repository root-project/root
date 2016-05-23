#include "TFile.h"
#include "TList.h"
#include "TTree.h"

int generate(int alg, int level) {
  unsigned int entries(1000u);
  TFile* orgfile = TFile::Open("org.root");
  TList* thelist = orgfile->GetListOfKeys();
  TObjLink* thelink = thelist->FirstLink();
  while (thelink) {
    if (orgfile->Get(thelink->GetObject()->GetName())->InheritsFrom("TTree")) {
      TTree* t;
      orgfile->GetObject(thelink->GetObject()->GetName(),t);
      //t->MakeClass("parent");
      //TFile* uncompressed = new TFile("uncompressed.root","recreate","",ROOT::CompressionSettings(ROOT::kZLIB,0));
      //uncompressed->WriteTObject(t->CloneTree(entries));
      //uncompressed->Close();
      //delete uncompressed;
      //uncompressed=NULL;
      //for (int i = 1 ; i < 10 ; ++i) {
      //  TFile* pointer = new TFile(Form("zlib%d.root",i),"recreate","",ROOT::CompressionSettings(ROOT::kZLIB,i));
      //  pointer->WriteTObject(t->CloneTree(entries));
      //  pointer->Close();
      //  delete pointer;
      //}
      //for (int i = 1 ; i < 10 ; ++i) {
      //  TFile* pointer = new TFile(Form("lzma%d.root",i),"recreate","",ROOT::CompressionSettings(ROOT::kLZMA,i));
      //  pointer->WriteTObject(t->CloneTree(entries));
      //  pointer->Close();
      //  delete pointer;
      //}
      //for (int i = 1 ; i < 10 ; ++i) {
      //  TFile* pointer = new TFile(Form("lzo%d.root",i),"recreate","",4*100+i);
      //  pointer->WriteTObject(t->CloneTree(entries));
      //  pointer->Close();
      //  delete pointer;
      //}
      //for (int i = 1 ; i < 10 ; ++i) {
      //  TFile* pointer = new TFile(Form("lz4%d.root",i),"recreate","",5*100+i);
      //  pointer->WriteTObject(t->CloneTree(entries));
      //  pointer->Close();
      //  delete pointer;
      //}
      //for (int i = 1 ; i < 10 ; ++i) {
      //  TFile* pointer = new TFile(Form("zpf%d.root",i),"recreate","",6*100+i);
      //  pointer->WriteTObject(t->CloneTree(entries));
      //  pointer->Close();
      //  delete pointer;
      //}
      //for (int i = 5 ; i < 5+1 ; ++i) {
      //  TFile* pointer = new TFile(Form("bro%d.root",i),"recreate","",7*100+i);
      //  pointer->WriteTObject(t->CloneTree(entries));
      //  pointer->Close();
      //  delete pointer;
      //}
      {
        TFile* pointer = new TFile(Form("/dev/null"),"recreate","",alg*100+level);
        //TFile* pointer = new TFile(Form("delme.root"),"recreate","",alg*100+level);
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
  return generate(atoi(argv[1]),atoi(argv[2]));
}
