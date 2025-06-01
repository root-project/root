// --------------------------------------------------------------
// History:
//
// Created by Antonino Sergi (Antonino.Sergi@cern.ch) 2009-09-20
//
// --------------------------------------------------------------
#ifndef TVDigi_H
#define TVDigi_H

#include "TVHit.hh"

class TVDigi : public TVHit {

    public:

        TVDigi();
        TVDigi(Int_t);
        TVDigi(TVHit*);
        virtual ~TVDigi(){};
        void Clear(Option_t* = "") override;
        virtual Double_t GetTime() = 0;
        virtual Int_t GetStationID() = 0;
        virtual Int_t EncodeChannelID() = 0;
        virtual void  DecodeChannelID() = 0;

    public:

        TVHit *              GetMCHit()                                         { return fMCHit;                        };
        void                 SetMCHit(TVHit * value)                            { fMCHit = value;                       };
        Int_t                GetDigiFineTime();  // defined from 0 to 255

    protected:

        TVHit*      fMCHit; //!  Transient data member for MCTruth Association

        ClassDefOverride(TVDigi,1);
};
#endif
