// @(#)root/geom:$Name:$:$Id:$
// Author: Andrei Gheata   11/01/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TVirtualGeoPainter
#define ROOT_TVirtualGeoPainter


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualGeoPainter                                                   //
//                                                                      //
// Abstract base class for geometry painters                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TGeoVolume;
class TGeoManager;
class TObjArray;

class TVirtualGeoPainter : public TObject {


protected:
   static TVirtualGeoPainter   *fgGeoPainter; //Pointer to class painter

public:
   TVirtualGeoPainter();
   virtual ~TVirtualGeoPainter();

   virtual Int_t      DistanceToPrimitive(Int_t px, Int_t py) = 0;
   virtual void       Draw(Option_t *option="") = 0;
   virtual void       DrawOnly(Option_t *option="") = 0;
   virtual void       DrawPanel() = 0;
   virtual void       ExecuteEvent(Int_t event, Int_t px, Int_t py) = 0;
   virtual Int_t      GetNsegments() const = 0; 
   virtual char      *GetObjectInfo(Int_t px, Int_t py) const = 0;
   virtual void       Paint(Option_t *option="") = 0;
   virtual void       PaintStat(Int_t dostat, Option_t *option="") = 0;
   virtual void       PaintBox(TGeoVolume *vol, Option_t *option="") = 0;
   virtual void       PaintTube(TGeoVolume *vol, Option_t *option="") = 0;
   virtual void       PaintTubs(TGeoVolume *vol, Option_t *option="") = 0;
   virtual void       PaintSphere(TGeoVolume *vol, Option_t *option="") = 0;
   virtual void       PaintPcon(TGeoVolume *vol, Option_t *option="") = 0;
   virtual void       SetNsegments(Int_t nseg) = 0;    
   static  TVirtualGeoPainter *GeoPainter();
   static void      SetPainter(TVirtualGeoPainter *painter);

    ClassDef(TVirtualGeoPainter,0)  //Abstract interface for geometry painters
};

#endif
