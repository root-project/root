#include "TObjArray.h"
#include "TClonesArray.h"

class Collector : public TObject {
   TClonesArray   fMyClArr; 
   TObjArray      fMyObjArr;
   Bool_t         fValid;

public:
   const TClonesArray& getArr() { return fMyClArr; }
   const TObjArray* getObjArr() { if (fValid) return &fMyObjArr; else return 0; } 
   Collector() : fMyClArr("TNamed") {}
   ClassDefOverride(Collector,1);
};

class Real : public Collector {
public:
   Float_t fValue;

   ClassDefOverride(Real,1);
};

class Event {

   TClonesArray fArr;
   Collector fCol;
   Int_t fEventNum;

public:
   Event() : fArr("Real") {};
   int getNum() { return fEventNum; }
};

#include "TTree.h"

void tobj(bool debug=false) {
   Event * e = new Event;
   TTree *t = new TTree("T","T");
   if (debug) gDebug = 9;
   t->Branch("event","Event",&e,32000,99);
   if (debug) gDebug = 0;
   t->Print();
}
