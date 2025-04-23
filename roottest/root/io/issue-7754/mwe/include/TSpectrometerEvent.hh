// --------------------------------------------------------------
// History:
//
// Created by Antonino Sergi (Antonino.Sergi@cern.ch) 2009-01-14
//
// --------------------------------------------------------------
#ifndef TSpectrometerEvent_H
#define TSpectrometerEvent_H

#include "TDetectorVEvent.hh"

class TSpectrometerEvent : public TDetectorVEvent {

    public:

        TSpectrometerEvent();
        ~TSpectrometerEvent();
        void Clear(Option_t* = "");

    private:

        ClassDef(TSpectrometerEvent,1);
};
#endif
