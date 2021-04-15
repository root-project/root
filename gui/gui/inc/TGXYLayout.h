// @(#)root/gui:$Id$
// Author: Reiner Rohlfs   24/03/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun, Fons Rademakers and Reiner Rohlfs *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TGXYLayout
#define ROOT_TGXYLayout

#include "TGLayout.h"


class TGXYLayoutHints : public TGLayoutHints {

protected:
   Double_t   fX;    ///< x - position of widget
   Double_t   fY;    ///< y - position of widget
   Double_t   fW;    ///< width of widget
   Double_t   fH;    ///< height of widget
   UInt_t     fFlag; ///< rubber flag

public:

   enum ERubberFlag {
      kLRubberX   = BIT(0),
      kLRubberY   = BIT(1),
      kLRubberW   = BIT(2),
      kLRubberH   = BIT(3)
   };

   TGXYLayoutHints(Double_t x, Double_t y, Double_t w, Double_t h,
                   UInt_t rubberFlag = kLRubberX | kLRubberY);

   Double_t  GetX() const { return fX; };
   Double_t  GetY() const { return fY; };
   Double_t  GetW() const { return fW; };
   Double_t  GetH() const { return fH; };
   UInt_t    GetFlag() const { return fFlag; };

   void      SetX(Double_t x) { fX = x; }
   void      SetY(Double_t y) { fY = y; }
   void      SetW(Double_t w) { fW = w; }
   void      SetH(Double_t h) { fH = h; }
   void      SetFlag(UInt_t flag) { fFlag = flag; }

   virtual void SavePrimitive(std::ostream &out, Option_t * = "");

   ClassDef(TGXYLayoutHints,0)  // Hits for the X / Y - layout manager
};


class TGXYLayout : public TGLayoutManager {

protected:
   TList            *fList;           ///< list of frames to arrange
   TGCompositeFrame *fMain;           ///< container frame

   Bool_t            fFirst;          ///< flag to determine the first call of Layout()
   UInt_t            fFirstWidth;     ///< original width of the frame fMain
   UInt_t            fFirstHeight;    ///< original height of the fram fMain

   Int_t             fTWidth;         ///< text width of a default character "1234567890" / 10
   Int_t             fTHeight;        ///< text height

   TGXYLayout(const TGXYLayout&);
   TGXYLayout& operator=(const TGXYLayout&);

public:
   TGXYLayout(TGCompositeFrame *main);

   virtual void Layout();
   virtual TGDimension GetDefaultSize() const;
   virtual void SavePrimitive(std::ostream &out, Option_t * = "");

   void NewSize() { fFirst = kTRUE; }

   ClassDef(TGXYLayout,0)  // X / Y - layout manager
};

#endif
