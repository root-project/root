// @(#)root/base:$Id$
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAttFill
#define ROOT_TAttFill


#include "Rtypes.h"


class TAttFill {

protected:
   Color_t    fFillColor;           ///< Fill area color
   Style_t    fFillStyle;           ///< Fill area style

public:
   TAttFill();
   TAttFill(Color_t fcolor,Style_t fstyle);
   virtual ~TAttFill();
   void             Copy(TAttFill &attfill) const;
   virtual Color_t  GetFillColor() const { return fFillColor; } ///< Return the fill area color
   virtual Style_t  GetFillStyle() const { return fFillStyle; } ///< Return the fill area style
   virtual Bool_t   IsTransparent() const;
   virtual void     Modify();
   virtual void     ResetAttFill(Option_t *option="");
   virtual void     SaveFillAttributes(std::ostream &out, const char *name, Int_t coldef=1, Int_t stydef=1001);
   virtual void     SetFillAttributes(); // *MENU*
   virtual void     SetFillColor(Color_t fcolor) { fFillColor = fcolor; } ///< Set the fill area color
   virtual void     SetFillColorAlpha(Color_t fcolor, Float_t falpha);
   virtual void     SetFillStyle(Style_t fstyle) { fFillStyle = fstyle; } ///< Set the fill area style

   ClassDef(TAttFill,2)  //Fill area attributes
};

inline Bool_t TAttFill::IsTransparent() const
{ return fFillStyle >= 4000 && fFillStyle <= 4100 ? kTRUE : kFALSE; }

   enum EFillStyle {kFDotted1  = 3001, kFDotted2    = 3002, kFDotted3  = 3003,
                    kFHatched1 = 3004, kHatched2    = 3005, kFHatched3 = 3006,
                    kFHatched4 = 3007, kFWicker     = 3008, kFScales   = 3009,
                    kFBricks   = 3010, kFSnowflakes = 3011, kFCircles  = 3012,
                    kFTiles    = 3013, kFMondrian   = 3014, kFDiamonds = 3015,
                    kFWaves1   = 3016, kFDashed1    = 3017, kFDashed2  = 3018,
                    kFAlhambra = 3019, kFWaves2     = 3020, kFStars1   = 3021,
                    kFStars2   = 3022, kFPyramids   = 3023, kFFrieze   = 3024,
                    kFMetopes  = 3025, kFEmpty      = 0   , kFSolid    = 1};

#endif
