// --------------------------------------------------------------
// History:
//
// Created by Antonino Sergi (Antonino.Sergi@cern.ch) 2009-10-04
//
// --------------------------------------------------------------
#include "TVEvent.hh"

TVEvent::TVEvent() : TObject(), fStartByte(0), fID(0), fBurstID(0), fRunID(0),
  fIsMC(kFALSE), fTriggerType(0), fL0TriggerType(0), fTimeStamp(0), fFineTime(0.0), fLatency(0.0) {}

TVEvent::TVEvent(TVEvent & event) :
  TObject(event),
  fStartByte(event.fStartByte),
  fID(event.fID),
  fBurstID(event.fBurstID),
  fRunID(event.fRunID),
  fIsMC(event.fIsMC),
  fTriggerType(event.fTriggerType),
  fL0TriggerType(event.fL0TriggerType),
  fTimeStamp(event.fTimeStamp),
  fFineTime(event.fFineTime),
  fLatency(event.fLatency)
{
}

TVEvent::~TVEvent() {}

Int_t TVEvent::Compare(const TObject *obj) const {
  // Compare two TVEvents objects
  if (this == obj) return 0;
  if(fTimeStamp > static_cast<const TVEvent*>(obj)->GetTimeStamp()) return 1;
  else if(fTimeStamp < static_cast<const TVEvent*>(obj)->GetTimeStamp()) return -1;
  else return 0;
}

void TVEvent::Clear(Option_t * /*option*/){
    fID = 0;
    fBurstID = 0;
    fRunID = 0;
    fIsMC = kFALSE;
    fTimeStamp = 0;
    fTriggerType = 0;
    fL0TriggerType = 0;
}
