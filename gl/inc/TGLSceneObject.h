// @(#)root/gl:$Name:  $:$Id: TGLSceneObject.h,v 1.6 2004/09/15 14:26:58 brun Exp $
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
class TGLSceneObject : public TObject {
protected:
   //Float_t fColor[4];
   Float_t fColor[17];

private:
   UInt_t fGLName;
   TGLSceneObject *fNextT;
   TObject *fRealObject;

public:
   TGLSceneObject(const Float_t *color = 0, UInt_t glname = 0, TObject *realobj = 0);

   virtual Bool_t IsTransparent()const;
   virtual void ResetTransparency(char newval);

   virtual void GLDraw()const = 0;
   virtual void Shift(Double_t x, Double_t y, Double_t z);

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
//   void GetColor(Color_t &r, Color_t &g, Color_t &b, Color_t &a)const;
   const Float_t *GetColor()const
   {
      return fColor;
   }
   void SetColor(const Float_t *newColor);
//   void SetColor(Color_t r, Color_t g, Color_t b, Color_t a);
private:
   TGLSceneObject(const TGLSceneObject &);
   TGLSceneObject & operator = (const TGLSceneObject &);
};

///////////////////////////////////////////////////////////////////////
class TGLFaceSet : public TGLSceneObject {
private:
   std::vector<Double_t> fVertices;
   std::vector<Double_t> fNormals;
   std::vector<Int_t> fPolyDesc;

   Bool_t fIsTransparent;
   UInt_t fNbPols;

public:
   TGLFaceSet(const TBuffer3D &buff, const Float_t *color,
              UInt_t glName, TObject *realObj);

   Bool_t IsTransparent()const;
   void ResetTransparency(char newVal);
   void GLDraw()const;
   void Shift(Double_t x, Double_t y, Double_t z);

private:
   Int_t CheckPoints(const Int_t *source, Int_t *dest)const;
   static Bool_t Eq(const Double_t *p1, const Double_t *p2)
   {
      return *p1 == *p2 && p1[1] == p2[1] && p1[2] == p2[2];
   }
};
////////////////////////////////////////////////////////////////////////
class TGLPolyMarker : public TGLSceneObject {
private:
   std::vector<Double_t> fVertices;
   UInt_t fStyle;

public:
   TGLPolyMarker(const TBuffer3D &buff, const Float_t *color, UInt_t glName, TObject *realObject);
   void GLDraw()const;

private:
   void DrawStars()const;
};


class TGLPolyLine : public TGLSceneObject {
private:
   std::vector<Double_t> fVertices;

public:
   TGLPolyLine(const TBuffer3D &buff, const Float_t *color, UInt_t glName, TObject *realObject);
   void GLDraw()const;
};


class TGLSimpleLight : public TGLSceneObject {
private:
   Float_t  fPosition[4];
   Float_t  fBulbRad;
   UInt_t   fLightName;

public:
   TGLSimpleLight(UInt_t glName, UInt_t lightName, const Float_t *color, const Float_t *position = 0);
   void GLDraw()const;
   void Shift(Double_t x, Double_t y, Double_t z);
   void SetBulbRad(Float_t newRad);
};

/////////////////////////////////////////////////////////////
class TGLSelection: public TGLSceneObject {
private:
   typedef std::pair<Double_t, Double_t>PDD_t;
   PDD_t fXRange;
   PDD_t fYRange;
   PDD_t fZRange;

public:
   TGLSelection();
   TGLSelection(const PDD_t &x, const PDD_t &y, const PDD_t &z);
   void GLDraw()const;
   void SetBox(const PDD_t &x, const PDD_t &y, const PDD_t &z);
   void Shift(Double_t x, Double_t y, Double_t z);
};

#endif
