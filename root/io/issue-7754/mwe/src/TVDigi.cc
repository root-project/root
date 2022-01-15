// --------------------------------------------------------------
// History:
//
// Created by Antonino Sergi (Antonino.Sergi@cern.ch) 2009-09-20
//
// --------------------------------------------------------------
#include "TVDigi.hh"
#include "NA62Global.hh"
#include <cmath>

TVDigi::TVDigi() : TVHit() {
  fMCHit = 0;
}

TVDigi::TVDigi(Int_t iCh) : TVHit(iCh) {
  fMCHit = 0;
}

TVDigi::TVDigi(TVHit* MCHit) {
  fMCHit = MCHit;
  (*static_cast<TVHit*>(this)) = (*static_cast<TVHit*>(MCHit));
}

void TVDigi::Clear(Option_t* option) {
  TVHit::Clear(option);
  fMCHit = 0;
}

Int_t TVDigi::GetDigiFineTime(){
  return ((Int_t)((Int_t) (GetTime()/TdcCalib+0.5)-floor(GetTime()/ClockPeriod)*256))%256; // defined from 0 to 255
}
