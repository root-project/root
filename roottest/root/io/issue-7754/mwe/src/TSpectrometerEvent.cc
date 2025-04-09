// --------------------------------------------------------------
// History:
//
// Created by Antonino Sergi (Antonino.Sergi@cern.ch) 2008-04-23
//
// --------------------------------------------------------------
#include "TSpectrometerEvent.hh"

#include "TSpectrometerHit.hh"

TSpectrometerEvent::TSpectrometerEvent() : TDetectorVEvent(TSpectrometerHit::Class()){
}

TSpectrometerEvent::~TSpectrometerEvent() {
}

void TSpectrometerEvent::Clear(Option_t* option) {
  TDetectorVEvent::Clear(option);
}
