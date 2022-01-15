// --------------------------------------------------------------
// History:
//
// Created by Antonino Sergi (Antonino.Sergi@cern.ch) 2009-01-08
//
// --------------------------------------------------------------
#include "TDetectorVEvent.hh"

#include "Riostream.h"

TDetectorVEvent::TDetectorVEvent() : TVEvent(), fNHits(0), fHits(nullptr) {}

TDetectorVEvent::TDetectorVEvent(const TDetectorVEvent & event) :
  TVEvent((TVEvent&)event),
  fNHits(event.fNHits),
  fHits(nullptr)
{
  if (event.fHits) fHits = new TClonesArray(*event.fHits);
}

TDetectorVEvent::TDetectorVEvent(TClass * Class, Int_t NMaxHits) :
  TVEvent(),
  fNHits(0)
{
  fHits = new TClonesArray(Class,NMaxHits);
}

TDetectorVEvent::~TDetectorVEvent() {
  fNHits = 0;
  if (fHits) {
    delete fHits;
    fHits = 0;
  }
}

void TDetectorVEvent::Clear(Option_t * option){
  fNHits = 0;
  if(fHits) fHits->Clear(option);
}

TVHit* TDetectorVEvent::AddHit(){
  return static_cast<TVHit *>((fHits->ConstructedAt(fNHits++,"C")));
}

TVHit* TDetectorVEvent::AddHit(Int_t iCh){
  TVHit * hit = static_cast<TVHit *>((fHits->ConstructedAt(fNHits++,"C")));
  hit->SetChannelID(iCh);
  return hit;
}

TDetectorVHit * TDetectorVEvent::AddHit(TDetectorVHit * MCHit){
  TDetectorVHit * hit = static_cast<TDetectorVHit *>((fHits->ConstructedAt(fNHits++,"C")));
  *hit = *MCHit;
  return hit;
}

TVHit* TDetectorVEvent::GetLastHitOnChannel(Int_t iCh) {
    TVHit * hit = 0;
    for(Int_t iHit = 0; iHit < fNHits; iHit++){
        if( static_cast<TVHit*>(fHits->At(iHit))->GetChannelID() == iCh)
            hit = static_cast<TVHit*>(fHits->At(iHit));
    }
    return hit;
}

TVHit* TDetectorVEvent::GetHit(Int_t iHit) {
    if(iHit<0 || iHit>=fNHits) return 0;
    return static_cast<TVHit*>(fHits->At(iHit));
}

TVHit* TDetectorVEvent::GetLastHit() {
    if(!fNHits) return 0;
    return static_cast<TVHit*>(fHits->At(fNHits-1));
}

void TDetectorVEvent::RemoveHit(Int_t iHit){
  if(iHit>=0 && iHit<fNHits){
    fHits->RemoveAt(iHit);
    fHits->Compress();
    fNHits--;
  }
}
