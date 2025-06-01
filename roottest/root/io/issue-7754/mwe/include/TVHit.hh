// --------------------------------------------------------------
// History:
//
// Created by Antonino Sergi (Antonino.Sergi@cern.ch) 2009-01-08
//
// --------------------------------------------------------------
#ifndef TVHit_H
#define TVHit_H

#include "TObject.h"


class TVHit : public TObject {

    public:

        TVHit();
        TVHit(const TVHit &);
        explicit TVHit(Int_t);
        virtual ~TVHit(){};
        TVHit& operator=(const TVHit &right);

        void Clear(Option_t* = "") override;
        virtual void UpdateReferenceTime(Double_t) = 0;
        void ShiftMCTrackID(Int_t value){ fMCTrackID += value; };
        Bool_t IsSortable() const override { return kTRUE; }
        Int_t Compare(const TObject *obj) const override;
        void Print(Option_t* option="") const override;

        Int_t  GetChannelID() const      { return fChannelID;         }
        Int_t  SetChannelID(Int_t value) { return fChannelID = value; }

        Int_t  GetMCTrackID()                     { return fMCTrackID;          }
        void   SetMCTrackID(Int_t value)          { fMCTrackID = value;         }
        Int_t  GetKinePartIndex()                    { return fKinePartIndex;         }
        void   SetKinePartIndex(Int_t value)         { fKinePartIndex = value;        }
        Bool_t GetDirectInteraction()             { return fDirectInteraction;  }
        void   SetDirectInteraction(Bool_t value) { fDirectInteraction = value; }

    protected:
        Int_t  fChannelID;  ///< ID of the detector channel
        Int_t  fMCTrackID;  ///< ID of the associated Geant4 track (>=1)
        Bool_t fDirectInteraction; ///< Is the Geant4 track which produced the hit stored as a KinePart?
        Int_t  fKinePartIndex; ///< Index of the associated KinePart (>=0), added in v2

  ClassDefOverride(TVHit,2);
};
#endif
