// --------------------------------------------------------------
// History:
//
// Created by Antonino Sergi (Antonino.Sergi@cern.ch) 2009-01-08
//
// --------------------------------------------------------------
#ifndef TDetectorVEvent_H
#define TDetectorVEvent_H

#include "TClass.h"

#include "TDetectorVHit.hh"
#include "TVEvent.hh"
#include "TVDigi.hh"
#include "TVHit.hh"
#include "TClonesArray.h"

class TDetectorVEvent : public TVEvent {

public:

  TDetectorVEvent();
  explicit TDetectorVEvent(const TDetectorVEvent &);
  explicit TDetectorVEvent(TClass * Class, Int_t NMaxHits=1000);
  ~TDetectorVEvent();
  TVHit * AddHit();
  TVHit * AddHit(Int_t iCh);
  TDetectorVHit * AddHit(TDetectorVHit *);
  virtual TVHit * GetLastHitOnChannel(Int_t iCh);
  virtual TVHit * GetHit(Int_t iHit);
  virtual TVHit * GetLastHit();
  void RemoveHit(Int_t iHit);
  void Clear(Option_t* = "") override;

  virtual Int_t         GetNHits() { return fNHits; }
  virtual TClonesArray* GetHits()  { return fHits;  }

private:

  Int_t         fNHits;
  TClonesArray* fHits;
  ClassDefOverride(TDetectorVEvent,1);
};

#endif
