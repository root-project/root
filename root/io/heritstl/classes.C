#include "TObject.h"
#include "TFile.h"
#include "TTree.h"
#include "TBufferFile.h"

#include <vector>

class PlexItem {
public:
   int fB;
   PlexItem() {};
   virtual ~PlexItem() {};   
   ClassDef(PlexItem,1)
};


class PlexSTL : public std::vector<PlexItem> {
public:
   int a;
   PlexSTL() {};
   virtual ~PlexSTL() {};   
   ClassDef(PlexSTL,1)
};

class Object : public TNamed {
public:
   PlexSTL vect;
   ClassDef(Object,1)
};

void testing_old() {
   TBufferFile b(TBuffer::kWrite);
   b.SetWriteMode();
   PlexSTL p;
   //b << *p;
   p.Streamer(b);
   PlexSTL * np = 0;
   b.SetReadMode();
   b.Reset();
   //b >> np;
   np = new PlexSTL; np->Streamer(b);
}

void testing_direct() {
   TBufferFile b(TBuffer::kWrite);
   b.SetWriteMode();
   Object p;
   //b << *p;
   p.Streamer(b);
   Object * np = 0;
   b.SetReadMode();
   b.Reset();
   //b >> np;
   np = new Object; np->Streamer(b);
}

void testing() {
   TFile * f = new TFile("test01.root","RECREATE");
   TTree * tree = new TTree("T","T");
   Object * o = new Object;
   tree->Branch("event","Object",&o);
   tree->Fill();
   f->Write();
   f->Close();
}

#ifndef __CINT__
ClassImp(PlexSTL)
ClassImp(PlexItem)
ClassImp(Object)
#endif

