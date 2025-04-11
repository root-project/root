// --------------------------------------------------------------
// History:
//
// Created by Antonino Sergi (Antonino.Sergi@cern.ch) 2011-01-31
//
// --------------------------------------------------------------
#ifndef SpectrometerChannelID_H
#define SpectrometerChannelID_H
#include "Rtypes.h"

class SpectrometerChannelID {

    public:

      SpectrometerChannelID();
      virtual ~SpectrometerChannelID() {}
      SpectrometerChannelID& operator=(const SpectrometerChannelID &right);
      void Clear(Option_t* = "");

      Int_t EncodeChannelID();
      void DecodeChannelID(Int_t);

    public:

        Int_t                GetStrawID()                                       { return fStrawID;                      };
        void                 SetStrawID(Int_t value)                            { fStrawID = value;                     };
        Int_t                GetPlaneID()                                       { return fPlaneID;                      };
        void                 SetPlaneID(Int_t value)                            { fPlaneID = value;                     };
        Int_t                GetHalfViewID()                                    { return fHalfViewID;                   };
        void                 SetHalfViewID(Int_t value)                         { fHalfViewID = value;                  };
        Int_t                GetViewID()                                        { return fViewID;                       };
        void                 SetViewID(Int_t value)                             { fViewID = value;                      };
        Int_t                GetChamberID()                                     { return fChamberID;                    };
        void                 SetChamberID(Int_t value)                          { fChamberID = value;                   };

    protected:

        Int_t      fStrawID;
        Int_t      fPlaneID;
        Int_t      fHalfViewID;
        Int_t      fViewID;
        Int_t      fChamberID;

        ClassDef(SpectrometerChannelID,1);
};
#endif
