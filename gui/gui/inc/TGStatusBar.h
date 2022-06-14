// @(#)root/gui:$Id$
// Author: Fons Rademakers   23/01/98

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGStatusBar
#define ROOT_TGStatusBar


#include "TGFrame.h"

class TGStatusBarPart;


class TGStatusBar : public TGHorizontalFrame {

friend class TGStatusBarPart;

private:
   TGStatusBar(const TGStatusBar&) = delete;
   TGStatusBar& operator=(const TGStatusBar&) = delete;

protected:
   TGStatusBarPart **fStatusPart; ///< frames containing statusbar text
   Int_t            *fParts;      ///< size of parts (in percent of total width)
   Int_t             fNpart;      ///< number of parts
   Int_t             fYt;         ///< y drawing position (depending on font)
   Int_t            *fXt;         ///< x position for each part
   Bool_t            f3DCorner;   ///< draw 3D corner (drawn by default)

   static const TGFont *fgDefaultFont;
   static TGGC         *fgDefaultGC;

   void DoRedraw() override;

   static FontStruct_t  GetDefaultFontStruct();
   static const TGGC   &GetDefaultGC();

public:
   TGStatusBar(const TGWindow *p = nullptr, UInt_t w = 4, UInt_t h = 2,
               UInt_t options = kSunkenFrame | kHorizontalFrame,
               Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGStatusBar();

   void         DrawBorder() override;
   virtual void SetText(TGString *text, Int_t partidx = 0);
   virtual void SetText(const char *text, Int_t partidx = 0);
           void AddText(const char *text, Int_t partidx = 0)
                  { SetText(text, partidx); }                  //*MENU*
   const char  *GetText(Int_t partidx = 0) const;
   virtual void SetParts(Int_t npart);                         //*MENU*
   virtual void SetParts(Int_t *parts, Int_t npart);
   void         Draw3DCorner(Bool_t corner) { f3DCorner = corner; }
   TGCompositeFrame *GetBarPart(Int_t npart) const;
   TGDimension GetDefaultSize() const override;

   void SavePrimitive(std::ostream &out, Option_t *option = "") override;

   ClassDefOverride(TGStatusBar,0)  // Status bar widget
};

#endif
