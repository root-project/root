// @(#)root/gl:$Name:  $:$Id: TGLSceneObject.h,v 1.5 2004/09/14 15:37:34 rdm Exp $
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
   Float_t fColor[4];

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
   void GetColor(Color_t &r, Color_t &g, Color_t &b, Color_t &a)const;
   void SetColor(Color_t r, Color_t g, Color_t b, Color_t a);
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
              UInt_t glname, TObject *realObj);

   Bool_t IsTransparent()const;
   void ResetTransparency(char newval);

   void GLDraw()const;

   void SetColor(const Float_t *newcolor = 0);
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
   TGLPolyMarker(const TBuffer3D &buff, const Float_t *color, UInt_t glname, TObject *realobject);
   void GLDraw()const;

private:
   void DrawStars()const;
};


class TGLPolyLine : public TGLSceneObject {
private:
   std::vector<Double_t> fVertices;

public:
   TGLPolyLine(const TBuffer3D &buff, const Float_t *color, UInt_t glname, TObject *realobject);
   void GLDraw()const;
};


class TGLSimpleLight : public TGLSceneObject {
private:
   Float_t fPosition[4];
   UInt_t fLightName;

public:
   TGLSimpleLight(UInt_t glname, UInt_t lightname, const Float_t *position, Bool_t dir = kTRUE);
   void GLDraw()const;
   void Shift(Double_t x, Double_t y, Double_t z);
};

/////////////////////////////////////////////////////////////
class TGLSelection: public TGLSceneObject {
private:
   typedef std::pair<Double_t, Double_t>PDD_t;
   PDD_t fXRange;
   PDD_t fYRange;
   PDD_t fZRange;

public:
   TGLSelection(const PDD_t &x, const PDD_t &y, const PDD_t &z);
   void GLDraw()const;
   void Shift(Double_t x, Double_t y, Double_t z);
};

#endif
