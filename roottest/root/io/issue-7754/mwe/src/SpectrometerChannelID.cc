// --------------------------------------------------------------
// History:
//
// Created by Antonino Sergi (Antonino.Sergi@cern.ch) 2011-01-31
//
// --------------------------------------------------------------
#include "SpectrometerChannelID.hh"
#include "NA62Global.hh"

SpectrometerChannelID::SpectrometerChannelID() {
  fStrawID = -1;
  fPlaneID = -1;
  fHalfViewID = -1;
  fViewID = -1;
  fChamberID = -1;
}
SpectrometerChannelID& SpectrometerChannelID::operator=(const SpectrometerChannelID &right)
{
  fStrawID = right.fStrawID;
  fPlaneID = right.fPlaneID;
  fHalfViewID = right.fHalfViewID;
  fViewID = right.fViewID;
  fChamberID = right.fChamberID;
  return *this;
}
void SpectrometerChannelID::Clear(Option_t * /*option*/){
  fStrawID = -1;
  fPlaneID = -1;
  fHalfViewID = -1;
  fViewID = -1;
  fChamberID = -1;
}

Int_t SpectrometerChannelID::EncodeChannelID(){
  return (fChamberID*16 + fViewID*4 + fHalfViewID*2 + fPlaneID)*122 + fStrawID;
}

void SpectrometerChannelID::DecodeChannelID(Int_t ChannelID){
  fChamberID = ChannelID/1952;
  fViewID = (ChannelID%1952)/488;
  fHalfViewID = (ChannelID%488)/244;
  fPlaneID = (ChannelID%244)/122;
  fStrawID = ChannelID%122;
}
