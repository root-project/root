// --------------------------------------------------------------
// History:
//
// Created by Antonino Sergi (Antonino.Sergi@cern.ch) 2009-01-08
//
// --------------------------------------------------------------
#ifndef TDetectorVHit_H
#define TDetectorVHit_H

#include "TVHit.hh"
#include "TVector3.h"

class TDetectorVHit : public TVHit {

    public:

        TDetectorVHit();
        TDetectorVHit(const TDetectorVHit &) = default;
        explicit TDetectorVHit(Int_t);
        virtual ~TDetectorVHit(){};
        void Clear(Option_t* = "") override;
        virtual void UpdateReferenceTime(Double_t value) override { fTime -= value; };
        void Print(Option_t* option="") const override;
        Int_t Compare(const TObject *obj) const override;

    public:

        TVector3             GetPosition() const                                { return fPosition;                     };
        void                 SetPosition(TVector3 value)                        { fPosition = value;                    };
        virtual Double_t     GetEnergy() const                                  { return fEnergy;                       };
        virtual void         SetEnergy(Double_t value)                          { fEnergy = value;                      };
        void                 AddEnergy(Double_t value)                          { fEnergy += value;                     };
        Double_t             GetTime() const                                    { return fTime;                         };
        void                 SetTime(Double_t value)                            { fTime = value;                        };

    private:

        TVector3   fPosition;
        Double_t   fEnergy;
        Double_t   fTime;

        ClassDefOverride(TDetectorVHit,1);
};
#endif
