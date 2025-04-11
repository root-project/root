// --------------------------------------------------------------
// History:
//
// Created by Antonino Sergi (Antonino.Sergi@cern.ch) 2009-01-08
//
// --------------------------------------------------------------
#include "TDetectorVHit.hh"

#include "Riostream.h"


TDetectorVHit::TDetectorVHit() :
  TVHit(),
  fPosition(0.,0.,0.),
  fEnergy(0.),
  fTime(0.)
{
}

TDetectorVHit::TDetectorVHit(Int_t iCh) :
  TVHit(iCh),
  fPosition(0.,0.,0.),
  fEnergy(0.),
  fTime(0.)
{
}

void TDetectorVHit::Print(Option_t *) const {
    TVHit::Print();
    std::cout << "HitPosition = (" << fPosition.X() << "," << fPosition.Y() << "," << fPosition.Z() << ")" << std::endl
        << "Energy = " << fEnergy << std::endl
        << "Time = " << fTime << std::endl << std::endl;
}

void TDetectorVHit::Clear(Option_t* option) {
  TVHit::Clear(option);
  fPosition = TVector3(0.,0.,0.);
  fEnergy = 0.;
  fTime = 0.;
}

Int_t TDetectorVHit::Compare(const TObject *obj) const {
  if(TVHit::Compare(obj)==0) {
    if(fTime < static_cast<const TDetectorVHit*>(obj)->GetTime()) return -1;
    else if(fTime > static_cast<const TDetectorVHit*>(obj)->GetTime()) return 1;
    else return 0;
  }
  return TVHit::Compare(obj);
}
