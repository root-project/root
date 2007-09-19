// @(#)root/table:$Id$
// Author:

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPolyLineShape
#define ROOT_TPolyLineShape

#ifndef ROOT_TShape
#include "TShape.h"
#endif
#ifndef ROOT_TAttMarker
#include "TAttMarker.h"
#endif

class TPoints3DABC;
class TVirtualPad;

enum EShapeTypes { kNULL=0, kSphere, kBrik};

class TPolyLineShape : public TShape, public TAttMarker {
 protected:
   Bool_t        fPointFlag;   // Flag whether we should paint "points" (option "P")
   Bool_t        fLineFlag;    // Flag whether we should connect the points with "line" (option "L")
   EShapeTypes   fShapeType;   // shape of the segment connections
   TShape       *fShape;       // shape for draw each segment of the polylins
   TShape       *fConnection;  // shape to represent the each "end" of the polyline
   TPoints3DABC   *fPoints;    // PolyLine itself
   Float_t       fWidthFactor; // factor to calculate the the tube diameters
   Bool_t        fHasDrawn;    // flag to avoid multiply plots
   Bool_t        fSmooth;      // Make smooth connections
   Size3D       *fSizeX3D;     //! the X3D buffer sizes


protected:
   virtual void  Create();
   virtual Size3D *CreateX3DSize(Bool_t marker=kFALSE);
   virtual void  SetConnection(TShape *connection){ fConnection = connection;}
   virtual Int_t PointDistancetoPrimitive(Int_t px, Int_t py);

public:
   TPolyLineShape();
   TPolyLineShape(TPoints3DABC *points,Option_t* option="P");
   virtual ~TPolyLineShape();
   virtual Int_t        DistancetoPrimitive(Int_t px, Int_t py);
   virtual void         Draw(Option_t *opt="");
   virtual void         ExecuteEvent(Int_t event, Int_t px, Int_t py);
   virtual TShape      *GetConnection() const { return fConnection;}
   virtual Color_t      GetColorAttribute() const;
   virtual const char  *GetName()  const;
   virtual char        *GetObjectInfo(Int_t px, Int_t py) const;
   virtual Size_t       GetSizeAttribute()  const;
   virtual Style_t      GetStyleAttribute() const;
   virtual const char  *GetTitle() const;
   virtual TPoints3DABC *GetMarker() const { return fPoints;}
   virtual TPoints3DABC *GetPoints() const { return fPoints;}
   virtual TShape      *GetShape() const { return fShape;}
   virtual Bool_t       GetSmooth() const { return fSmooth;}
   virtual Float_t      GetWidthFactor() const { return fWidthFactor;}
   virtual void         PaintNode(Float_t *start,Float_t *end,Option_t *option);
   virtual void         Paint(Option_t *opt);
   virtual void         Paint3d(Option_t *opt);
   virtual void         PaintX3DLine(Option_t *opt="");
   virtual void         PaintX3DMarker(Option_t *opt="");
   static Double_t     *Gyrot(Double_t *dirc, Double_t cosang,Double_t sinang, Double_t trans[3][3]);
   virtual void         PaintPoints(Int_t n, Float_t *p=0, Option_t *opt="");
   virtual void         PaintPolyMarker(Int_t n, Float_t *p=0, Marker_t m=0, Option_t *opt="");
   static Float_t       Product(Float_t *v1, Float_t *v2,Int_t ndim=3);
   static Double_t      Product(Double_t *v1, Double_t *v2,Int_t ndim=3);
   virtual Color_t      SetColorAttribute(Color_t color);
   virtual Size_t       SetSizeAttribute(Size_t size);
   virtual Int_t        SetConnection(EShapeTypes connection=kBrik);
   virtual void         SetShape(TShape *shape);
   virtual void         SetSmooth(Bool_t smooth=kTRUE){ fSmooth=smooth;}
   virtual Style_t      SetStyleAttribute(Style_t style);
   virtual void         SetWidthFactor(Float_t fact=1.0){fWidthFactor = fact;} //*MENU
   virtual Int_t        Size() const;
   virtual void         Sizeof3D() const;
   ClassDef(TPolyLineShape,0) // The base class to define an abstract 3D shape of STAR "event" geometry
};


#endif
