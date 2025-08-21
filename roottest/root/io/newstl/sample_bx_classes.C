#ifndef SAMPLE_BX_CLASS
#define SAMPLE_BX_CLASS
#include "TClonesArray.h"
#include <vector>

/*********** BxLaben *************/
class BxLabenCluster: public TObject {
  public:
    BxLabenCluster () {}
    BxLabenCluster (int n);
    virtual ~BxLabenCluster ();

    Int_t GetNHits              () const { return nhits; }
    const std::vector<Float_t>& GetTimes () const { return times; }

  private:
    Int_t nhits;
    std::vector<Float_t> times; //->

  ClassDef(BxLabenCluster, 1)

};

#ifdef __MAKECINT__
#pragma link C++ class std::vector<BxLabenCluster>+;
#pragma link C++ class std::vector<float>;
#endif 

class BxLaben: public TObject {
  public:
    BxLaben();
    virtual ~BxLaben ();

    void Assign(int i);

    const std::vector<BxLabenCluster>& GetClusters () const { return clusters; }

    Double_t GetTriggerTime () const { return trigger_time; }

  private:
    Double_t trigger_time;
    std::vector<BxLabenCluster> clusters; //->

  ClassDef(BxLaben, 1)
};


/*********** BxEvent *************/
class BxEvent : public TObject {
  public:
    BxEvent();
    virtual ~BxEvent ();

    void Assign(int i);

    // event header getters
    Int_t GetEvNum () const { return evnum; }

    // subclass getters
    const BxLaben&   GetLaben   () const { return laben; }

  private:
    Int_t evnum;
    BxLaben   laben;

  ClassDef(BxEvent, 1)
};



/********** implementation ************/
BxLabenCluster::BxLabenCluster(int n) {
  nhits = n;
  times.clear();
  times.push_back(7.3);
  times.push_back(7.5);
  times.push_back(7.7);
}

BxLaben::BxLaben() {
}

void BxLaben::Assign(int i) {
  trigger_time = 3.1;

  clusters.clear();
  if (i % 2) {
    for (int j = 0 ; j < 3; j++)
      clusters.push_back(j);
  }
}

BxEvent::BxEvent() : laben() {
}

void BxEvent::Assign(int i) { 
  evnum = i; 
  laben.Assign(i);
}

BxLabenCluster::~BxLabenCluster() {
}
BxLaben::~BxLaben() {
}

BxEvent::~BxEvent() {}
#endif
