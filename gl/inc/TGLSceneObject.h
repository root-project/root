// @(#)root/gl:$Name:  $:$Id: TGLSceneObject.h,v 1.9 2004/11/02 16:55:20 brun Exp $
// Author:  Timur Pocheptsov  03/08/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLSceneObject
#define ROOT_TGLSceneObject

#include <utility>

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_Gtypes
#include "Gtypes.h"
#endif

class TBuffer3D;

/////////////////////////////////////////////////////////////
class TGLSelection {
private:
   typedef std::pair<Double_t, Double_t>PDD_t;
   PDD_t fRangeX;
   PDD_t fRangeY;
   PDD_t fRangeZ;

public:
   TGLSelection();
   TGLSelection(const PDD_t &x, const PDD_t &y, const PDD_t &z);
   void DrawBox()const;

   void SetBox(const PDD_t &x, const PDD_t &y, const PDD_t &z);
   void Shift(Double_t x, Double_t y, Double_t z);
   void Stretch(Double_t xs, Double_t ys, Double_t zs);

   const PDD_t &GetRangeX()const
   {
      return fRangeX;
   }
   const PDD_t &GetRangeY()const
   {
      return fRangeY;
   }
   const PDD_t &GetRangeZ()const
   {
      return fRangeZ;
   }
};

/////////////////////////////////////////////////////////////
class TGLSceneObject : public TObject {
protected:
   std::vector<Double_t>fVertices;
   Float_t fColor[17];
   TGLSelection fSelectionBox;
   Double_t fCenter[3];

private:
   UInt_t fGLName;
   TGLSceneObject *fNextT;
   TObject *fRealObject;

public:
   TGLSceneObject(const Double_t *vertStart, const Double_t *vertEnd, 
                  const Float_t *color = 0, UInt_t glName = 0, TObject *realObj = 0);

   virtual Bool_t IsTransparent()const;
   virtual void ResetTransparency(char newval);

   virtual void GLDraw()const = 0;
   virtual void Shift(Double_t x, Double_t y, Double_t z);
   virtual void Stretch(Double_t xs, Double_t ys, Double_t zs);

   TGLSelection *GetBox()
   {
      return &fSelectionBox;
   }
   const TGLSelection *GetBox()const
   {
      return &fSelectionBox;
   }
   void SetNextT(TGLSceneObject *next)
   {
      fNextT = next;
   }
   TGLSceneObject *GetNextT()const
   {
      return fNextT;
   }
   UInt_t GetGLName()const
   {
      return fGLName;
   }
   TObject *GetRealObject()const
   {
      return fRealObject;
   }
   const Float_t *GetColor()const
   {
      return fColor;
   }

   void SetColor(const Float_t *newColor);

   const Double_t *GetObjectCenter()const
   {
      return fCenter;
   }

protected:
   void SetBox();

private:
   TGLSceneObject(const TGLSceneObject &);
   TGLSceneObject & operator = (const TGLSceneObject &);
};

///////////////////////////////////////////////////////////////////////
class TGLFaceSet : public TGLSceneObject {
private:
   std::vector<Double_t> fNormals;
   std::vector<Int_t>    fPolyDesc;
   UInt_t                fNbPols;

public:
   TGLFaceSet(const TBuffer3D &buff, const Float_t *color,
              UInt_t glName, TObject *realObj);

   Bool_t IsTransparent()const;
   void ResetTransparency(char newVal);
   void GLDraw()const;
   void Shift(Double_t x, Double_t y, Double_t z);
   void Stretch(Double_t xs, Double_t ys, Double_t zs);

private:
   Int_t CheckPoints(const Int_t *source, Int_t *dest)const;
   static Bool_t Eq(const Double_t *p1, const Double_t *p2);
   void CalculateNormals();
};
////////////////////////////////////////////////////////////////////////
class TGLPolyMarker : public TGLSceneObject {
private:
   UInt_t fStyle;

public:
   TGLPolyMarker(const TBuffer3D &buff, const Float_t *color, UInt_t glName, TObject *realObject);
   void GLDraw()const;

private:
   void DrawStars()const;
};


class TGLPolyLine : public TGLSceneObject {
public:
   TGLPolyLine(const TBuffer3D &buff, const Float_t *color, UInt_t glName, TObject *realObject);
   void GLDraw()const;
};

// Utility class to draw a Sphere using OpenGL Sphere primitive
class TGLSphere : public TGLSceneObject {
private:
   Float_t fX;      // Sphere X center position
   Float_t fY;      // Sphere Y center position
   Float_t fZ;      // Sphere Z center position
   Float_t fRadius; // Sphere radius
   Int_t   fNdiv;   // Number of divisions

public:
   TGLSphere(const TBuffer3D &buff, const Float_t *color, UInt_t glName, TObject *realObject);
   void GLDraw()const;
   void Shift(Double_t x, Double_t y, Double_t z);
   Bool_t IsTransparent()const;
};

class TGLSimpleLight : public TGLSceneObject {
private:
   Float_t  fBulbRad;
   UInt_t   fLightName;

public:
   TGLSimpleLight(UInt_t glName, UInt_t lightName, const Float_t *color, const Double_t *position);
   void GLDraw()const;
   void Shift(Double_t x, Double_t y, Double_t z);
   void SetBulbRad(Float_t newRad);
};

#endif
