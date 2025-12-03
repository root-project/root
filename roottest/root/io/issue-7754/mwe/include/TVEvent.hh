// --------------------------------------------------------------
// History:
//
// Created by Antonino Sergi (Antonino.Sergi@cern.ch) 2009-10-04
//
// --------------------------------------------------------------
#ifndef TVEvent_H
#define TVEvent_H

#include "TObject.h"
#include "NA62Global.hh"

class TVEvent : public TObject {

    public:

        TVEvent();
        TVEvent(TVEvent &);
        virtual ~TVEvent();
        void Clear(Option_t* = "") override;
        Int_t Compare(const TObject *obj) const override;
        Bool_t IsSortable() const override { return kTRUE; }

        ULong64_t        GetStartByte() const                               { return fStartByte;                    }
        void             SetStartByte(ULong64_t value)                      { fStartByte = value;                   }
        Int_t            GetID() const                                      { return fID;                           }
        void             SetID(Int_t value)                                 { fID = value;                          }
        Int_t            GetBurstID() const                                 { return fBurstID;                      }
        void             SetBurstID(Int_t value)                            { fBurstID = value;                     }
        Int_t            GetRunID() const                                   { return fRunID;                        }
        void             SetRunID(Int_t value)                              { fRunID = value;                       }
        ULong64_t        GetTriggerType() const                             { return fTriggerType;                  }
        void             SetTriggerType(ULong64_t value)                    { fTriggerType = value;                 }
        Int_t            GetL0TriggerType() const                           { return fL0TriggerType;                }
        void             SetL0TriggerType(Int_t value)                      { fL0TriggerType = value;               }
        ULong_t          GetTimeStamp() const                               { return fTimeStamp;                    }
        void             SetTimeStamp(ULong_t value)                        { fTimeStamp = value;                   }

    private:

        ULong64_t  fStartByte;
        Int_t      fID;
        Int_t      fBurstID;
        Int_t      fRunID;
        Bool_t     fIsMC;
        ULong64_t  fTriggerType;
        Int_t      fL0TriggerType;
        ULong_t fTimeStamp;
        Float_t fFineTime;     // useless? should be removed if true..
        Float_t fLatency;      // useless? should be removed if true..

        ClassDefOverride(TVEvent,1);
};
#endif
