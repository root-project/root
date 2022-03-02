// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveFrameBox
#define ROOT_TEveFrameBox

#include "TEveUtil.h"
#include "TObject.h"

class TEveFrameBox : public TObject, public TEveRefBackPtr
{
   friend class TEveFrameBoxGL;

public:
   enum EFrameType_e  { kFT_None, kFT_Quad, kFT_Box };

private:
   TEveFrameBox(const TEveFrameBox&);            // Not implemented
   TEveFrameBox& operator=(const TEveFrameBox&); // Not implemented

protected:
   EFrameType_e fFrameType;
   Int_t        fFrameSize;
   Float_t     *fFramePoints;  //[fFrameSize]

   Float_t      fFrameWidth;
   Color_t      fFrameColor;
   Color_t      fBackColor;
   UChar_t      fFrameRGBA[4];
   UChar_t      fBackRGBA[4];
   Bool_t       fFrameFill;
   Bool_t       fDrawBack;

public:
   TEveFrameBox();
   virtual ~TEveFrameBox();

   void SetAAQuadXY(Float_t x, Float_t y, Float_t z, Float_t dx, Float_t dy);
   void SetAAQuadXZ(Float_t x, Float_t y, Float_t z, Float_t dx, Float_t dz);

   void SetQuadByPoints(const Float_t* pointArr, Int_t nPoints);

   void SetAABox(Float_t x,  Float_t y,  Float_t z,
                 Float_t dx, Float_t dy, Float_t dz);

   void SetAABoxCenterHalfSize(Float_t x,  Float_t y,  Float_t z,
                               Float_t dx, Float_t dy, Float_t dz);

   // ----------------------------------------------------------------

   EFrameType_e GetFrameType()   const { return fFrameType; }
   Int_t        GetFrameSize()   const { return fFrameSize; }
   Float_t*     GetFramePoints() const { return fFramePoints; }

   Float_t GetFrameWidth() const    { return fFrameWidth; }
   void    SetFrameWidth(Float_t f) { fFrameWidth = f;    }

   Color_t  GetFrameColor() const { return fFrameColor; }
   Color_t* PtrFrameColor() { return &fFrameColor; }
   UChar_t* GetFrameRGBA()  { return fFrameRGBA;  }

   void SetFrameColor(Color_t ci);
   void SetFrameColorPixel(Pixel_t pix);
   void SetFrameColorRGBA(UChar_t r, UChar_t g, UChar_t b, UChar_t a=255);

   Color_t  GetBackColor() const { return fBackColor; }
   Color_t* PtrBackColor() { return &fBackColor; }
   UChar_t* GetBackRGBA()  { return fBackRGBA;  }

   void SetBackColor(Color_t ci);
   void SetBackColorPixel(Pixel_t pix);
   void SetBackColorRGBA(UChar_t r, UChar_t g, UChar_t b, UChar_t a=255);

   Bool_t GetFrameFill() const   { return fFrameFill; }
   void   SetFrameFill(Bool_t f) { fFrameFill = f;    }

   Bool_t GetDrawBack() const   { return fDrawBack; }
   void   SetDrawBack(Bool_t f) { fDrawBack = f;    }

   virtual void OnZeroRefCount() { delete this; }

   ClassDef(TEveFrameBox, 0); // Description of a 2D or 3D frame that can be used to visually group a set of objects.
};

#endif
