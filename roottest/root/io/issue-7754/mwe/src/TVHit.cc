// --------------------------------------------------------------
// History:
//
// Created by Antonino Sergi (Antonino.Sergi@cern.ch) 2009-01-08
//
// --------------------------------------------------------------
#include "TVHit.hh"
#include "NA62Global.hh"
#include "Riostream.h"

TVHit::TVHit() : TObject(), fChannelID(-1),fMCTrackID(-1),
  fDirectInteraction(kFALSE), fKinePartIndex(-1) {}


TVHit::TVHit(const TVHit & hit) :
  TObject(hit), fChannelID(hit.fChannelID),
  fMCTrackID(hit.fMCTrackID), fDirectInteraction(hit.fDirectInteraction),
  fKinePartIndex(hit.fKinePartIndex) {}

TVHit::TVHit(Int_t iCh) :
  fChannelID(iCh), fMCTrackID(-1),
  fDirectInteraction(kFALSE), fKinePartIndex(-1) {}

TVHit& TVHit::operator=(const TVHit &right){
  fChannelID = right.fChannelID;
  fMCTrackID = right.fMCTrackID;
  fDirectInteraction = right.fDirectInteraction;
  fKinePartIndex = right.fKinePartIndex;
  return *this;
}

void TVHit::Print(Option_t *) const {
    std::cout << "ChannelID = " << fChannelID << std::endl
              << "MCTrackID = " << fMCTrackID << std::endl
	      << "KinePartIndex = " << fKinePartIndex << std::endl
              << "DirectInteraction = " << fDirectInteraction << std::endl;
}

void TVHit::Clear(Option_t* /*option*/) {
  fChannelID = -1;
  fMCTrackID = -1;
  fKinePartIndex = -1;
  fDirectInteraction = kFALSE;
}

Int_t TVHit::Compare(const TObject *obj) const {
  if(fChannelID < static_cast<const TVHit*>(obj)->GetChannelID()) return -1;
  else if(fChannelID > static_cast<const TVHit*>(obj)->GetChannelID()) return 1;
  else return 0;
}
