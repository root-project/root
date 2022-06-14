// @(#)root/g3d:$Id$
// Author: "Valery fine"   31/10/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TMarker3DBox
#define ROOT_TMarker3DBox


////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TMarker3DBox                                                           //
//                                                                        //
// Marker3DBox is a special 3-D marker designed for event display.        //
// It has the following parameters:                                       //
//    fDx;              half length in X                                  //
//    fDy;              half length in Y                                  //
//    fDz;              half length in Z                                  //
//    fTranslation[3];  the coordinates of the center of the box          //
//    fDirCos[3];       the direction cosinus defining the orientation    //
//    fRefObject;       A reference to an object                          //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#include "TObject.h"
#include "TAttLine.h"
#include "TAttFill.h"
#include "TAtt3D.h"

class TH1;

class TMarker3DBox : public TObject, public TAttLine, public TAttFill,
                     public TAtt3D {
protected:
   Float_t  fX;               // X coordinate of center of box
   Float_t  fY;               // Y coordinate of center of box
   Float_t  fZ;               // Z coordinate of center of box
   Float_t  fDx;              // half length in x
   Float_t  fDy;              // half length in y
   Float_t  fDz;              // half length in z

   Float_t  fTheta;           // Angle of box z axis with respect to main Z axis
   Float_t  fPhi;             // Angle of box x axis with respect to main Xaxis
   TObject *fRefObject;       // Pointer to an object

   TMarker3DBox(const TMarker3DBox&);
   TMarker3DBox& operator=(const TMarker3DBox&);

   enum { kTemporary = BIT(23) }; // Use TObject::fBits to record if we are temporary

public:
   TMarker3DBox();
   TMarker3DBox(Float_t x, Float_t y, Float_t z,
                Float_t dx, Float_t dy, Float_t dz,
                Float_t theta, Float_t phi);
   virtual        ~TMarker3DBox();

   virtual Int_t   DistancetoPrimitive(Int_t px, Int_t py);
   virtual void    ExecuteEvent(Int_t event, Int_t px, Int_t py);
   TObject        *GetRefObject() const {return fRefObject;}
   virtual void    GetDirection(Float_t &theta, Float_t &phi) const {theta = fTheta; phi = fPhi;}
   virtual void    GetPosition(Float_t &x, Float_t &y, Float_t &z) const {x=fX; y=fY, z=fZ;}
   virtual void    GetSize(Float_t &dx, Float_t &dy, Float_t &dz) const {dx=fDx; dy=fDy; dz=fDz;}

   virtual void    Paint(Option_t *option);
   static  void    PaintH3(TH1 *h, Option_t *option);
   virtual void    SavePrimitive(std::ostream &out, Option_t *option = "");
   virtual void    SetPoints(Double_t *buff) const;
   virtual void    SetDirection(Float_t theta, Float_t phi);
   virtual void    SetPosition(Float_t x, Float_t y, Float_t z);
   virtual void    SetSize(Float_t dx, Float_t dy, Float_t dz);
   virtual void    SetRefObject(TObject *obj=0) {fRefObject = obj;}

   ClassDef(TMarker3DBox,2)  //A special 3-D marker designed for event display
};

#endif
