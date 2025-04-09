// begin of code - STreeEvent.h
#include <TClonesArray.h>

class TopTrack : public TObject {
public:
   int topval;
   TopTrack(int val = -1) : topval(val) {}
   ClassDefOverride(TopTrack,1);
};

class BottomTrack : public TopTrack {
public:
   int bottomval;
   BottomTrack(int val=-2) : TopTrack(-val),bottomval(val) {}
   ClassDefOverride(BottomTrack,1);
};


class PhotonsList: public TObject {
public:
   int val;
   TClonesArray* pPhotons; //photons data
   
   PhotonsList() : val(-1), pPhotons(0) { }
   ~PhotonsList() override {
      if (!pPhotons) return;
      pPhotons->Delete();
      delete pPhotons;
   }

   void Init() { pPhotons = new TClonesArray(BottomTrack::Class(), 8); }
   
   ClassDefOverride(PhotonsList,1); //list of photons for an event
}; // PhotonsList


class FittedList: public PhotonsList {
public:
   FittedList(): PhotonsList() {}
   
   ClassDefOverride(FittedList,1); //fitted event
}; // FittedList


class STreeEvent: public TObject {
public:
   FittedList Fit;                 //fitted data
   
   void Init() { Fit.Init(); }
   
   ClassDefOverride(STreeEvent,1); //event encapsulated in a tree
}; // STreeEvent

// end of code - STreeEvent.h

// begin of code - STreeEvent.cpp

ClassImp(PhotonsList)
ClassImp(FittedList)
ClassImp(STreeEvent)
// end of code - STreeEvent.cpp


// begin of code - STreeEventLinkDef.h
#ifdef __CINT__

#pragma link C++ class TopTrack+;
#pragma link C++ class BottomTrack+;
#pragma link C++ class PhotonsList+;
#pragma link C++ class FittedList+;
#pragma link C++ class STreeEvent+;

#endif
// end of code - STreeEventLinkDef.h
