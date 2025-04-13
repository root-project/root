#include "TClonesArray.h"

class Track : public TObject {
  public:
  Track() : a(0) {};
  explicit Track(int val) : a(val) {};
  int a;
  ClassDefOverride(Track,1);
};
class Header {
  public:
  int number;
};
class PhotonsList: public TObject {
    public:
  int val;
  int val2;
  TClonesArray* pPhotons; //photons data

  PhotonsList() { pPhotons = NULL; }
  ~PhotonsList() override {
    if (!pPhotons) return;
    pPhotons->Delete();
    delete pPhotons;
    }

  void Init() { pPhotons = new TClonesArray(Track::Class(), 8); }

  ClassDefOverride(PhotonsList,1) //list of photons for an event
}; // PhotonsList

class SubPhotonsList : public PhotonsList {

  ClassDefOverride(SubPhotonsList,1);
};

class STreeEvent: public TObject {
    public:
  Header header;
  PhotonsList Clusters;           //original photons data

  void Init() { Clusters.Init(); }

  ClassDefOverride(STreeEvent,1) //event encapsulated in a tree
}; // STreeEvent

class SubSTreeEvent: public STreeEvent {

  ClassDefOverride(SubSTreeEvent,1) //event encapsulated in a tree
}; // STreeEvent

#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class PhotonsList+;
#pragma link C++ class STreeEvent+;

#endif

