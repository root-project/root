// --------------------------------------------------------------
// History:
//
// Created by Antonino Sergi (Antonino.Sergi@cern.ch) 2008-04-24
//
// --------------------------------------------------------------
#include "TSpectrometerHit.hh"

TSpectrometerHit::TSpectrometerHit() : TDetectorVHit(), SpectrometerChannelID() {
  fLocalPosition = TVector3(0,0,0);
  fDirection = TVector3(0,0,0);
  fWireDistance = 0;
}

void TSpectrometerHit::Clear(Option_t* option){
  TDetectorVHit::Clear(option);
  SpectrometerChannelID::Clear(option);
  fLocalPosition = TVector3(0,0,0);
  fDirection = TVector3(0,0,0);
  fWireDistance = 0;
}

Int_t TSpectrometerHit::EncodeChannelID(){
  fChannelID =  SpectrometerChannelID::EncodeChannelID();
  return fChannelID;
}

void TSpectrometerHit::DecodeChannelID(){
    SpectrometerChannelID::DecodeChannelID(fChannelID);
}
