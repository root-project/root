// --------------------------------------------------------------
// History:
//
// Created by Antonino Sergi (Antonino.Sergi@cern.ch) 2008-03-30
//
// --------------------------------------------------------------
#ifndef TSpectrometerHit_H
#define TSpectrometerHit_H

#include "SpectrometerChannelID.hh"
#include "TDetectorVHit.hh"

class TSpectrometerHit : public TDetectorVHit, public SpectrometerChannelID {

    public:

        TSpectrometerHit();
        virtual ~TSpectrometerHit(){};
        void Clear(Option_t* = "") override;
        Int_t EncodeChannelID();
        void DecodeChannelID();

        //Int_t GetStationID() { return GetChamberID(); }
        Int_t GetStationID() { return 0; }

    public:

        TVector3             GetDirection()                                     { return fDirection;                    };
        void                 SetDirection(TVector3 value)                       { fDirection = value;                   };
        TVector3             GetLocalPosition()                                 { return fLocalPosition;                };
        void                 SetLocalPosition(TVector3 value)                   { fLocalPosition = value;               };

        Double_t             GetWireDistance()                                  { return fWireDistance;                 };
        void                 SetWireDistance(Double_t value)                    { fWireDistance = value;                };

    protected:

        TVector3   fDirection;
        TVector3   fLocalPosition;

        Double_t   fWireDistance;

        ClassDefOverride(TSpectrometerHit,1);
};
#endif
