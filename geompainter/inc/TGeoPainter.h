// Author: Andrei Gheata   05/03/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TGeoPainter
#define ROOT_TGeoPainter

#include "X3DBuffer.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGeoPainter                                                          //
//                                                                      //
// Painter for TGeo geometries                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TVirtualGeoPainter
#include "TVirtualGeoPainter.h"
#endif

#ifndef ROOT_TGeoManager
#include "TGeoManager.h"
#endif


class TGeoPainter : public TVirtualGeoPainter {
private:
    TGeoManager       *fGeo;       // geometry to which applies
    Int_t              fNsegments; // number of segments approximating circles
public:
    TGeoPainter();
    virtual ~TGeoPainter();
    virtual Int_t      DistanceToPrimitive(Int_t px, Int_t py);
    virtual void       Draw(Option_t *option="");
    virtual void       DrawOnly(Option_t *option="");
    virtual void       DrawPanel();
    virtual void       ExecuteEvent(Int_t event, Int_t px, Int_t py);
    virtual char      *GetObjectInfo(Int_t px, Int_t py) const;
    Int_t              GetNsegments() const {return fNsegments;}
    virtual void       Paint(Option_t *option="");
    void               PaintShape(X3DBuffer *buff, Bool_t rangeView);
    void               PaintBox(TGeoVolume *vol, Option_t *option="");
    void               PaintTube(TGeoVolume *vol, Option_t *option="");
    void               PaintTubs(TGeoVolume *vol, Option_t *option="");
    void               PaintSphere(TGeoVolume *vol, Option_t *option="");
    void               PaintPcon(TGeoVolume *vol, Option_t *option="");
    virtual void       PaintStat(Int_t dostat, Option_t *option="");
    void               SetNsegments(Int_t nseg) {fNsegments=nseg;}
    virtual void       SetGeoManager(TGeoManager *geom) {fGeo=geom;}

    ClassDef(TGeoPainter,0)  //geometry painter
};

#endif
