// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine   26/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//______________________________________________________________________________
//*-*-*-*-*-*-*-*-*-The  g W I N 3 2 B r u s h  class-*-*-*-*-*-*-*-*-*-*-*
//*-*               =================================
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

#include "TGWin32Brush.h"

//______________________________________________________________________________
TGWin32Brush::TGWin32Brush(){

   fFillBitMap.bmType     =  0;   // Specifies the bitmap type. This member must be zero
   fFillBitMap.bmWidth    = 16;   // Specifies the width, in pixel, of the bitmap
   fFillBitMap.bmHeight   = 16;   // Specifies the height, in pixel, of the bitmap
   fFillBitMap.bmWidthBytes = 2;  // Specifies the number of bytes in each scan line.
   fFillBitMap.bmPlanes   =  1;   // Specifies the count of color planes.
   fFillBitMap.bmBitsPixel=  1;   // Specifies the number of bits required to ind. th color
   fFillBitMap.bmBits = &p2_bits; // points to the location of the bit values for the bitmap;

   fBrush = (HBRUSH) GetStockObject(WHITE_BRUSH);     // Create a default brush

}

//______________________________________________________________________________
TGWin32Brush::~TGWin32Brush(){
  Delete();
}

//______________________________________________________________________________
void TGWin32Brush::Delete(){
   if (fBrush!= NULL) DeleteObject(fBrush);
   fBrush = NULL;
}
//______________________________________________________________________________
void TGWin32Brush::SetStyle(int style, int fasi){

//*-*-*-*-*-*-*-*-*-*-*Set fill area style index*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =========================
//*-*  style   : fill area interior style hollow or solid
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

//*-*  Delete old brush if any

if (fBrush!= NULL) DeleteObject(fBrush);

//*-*  Create the new brush

  switch( style ) {

  case 0:                                           // solid
    fLOGFillBrush.lbStyle = BS_HOLLOW;
    fBrush = (HBRUSH)GetStockObject (HOLLOW_BRUSH);
    break;
  case 1:                                           // solid
    fLOGFillBrush.lbStyle = BS_SOLID;
//    if (fLOGFillBrush.lbColor)
       fBrush = CreateSolidBrush(fLOGFillBrush.lbColor);
//    else
//       fBrush = GetStockObject(WHITE_BRUSH);
    break;

  case 3:                                           // pattern

    fLOGFillBrush.lbStyle = BS_PATTERN;
    fLOGFillBrush.lbHatch = (LONG) (&fFillBitMap);

    if (fasi > 0 && fasi < 26 )
       fFillBitMap.bmBits = patter_bits[fasi-1];
    else
       fFillBitMap.bmBits = &p2_bits;

    fBrush =
          CreatePatternBrush(CreateBitmapIndirect((LPBITMAP)fLOGFillBrush.lbHatch));
    break;

  case 2:                                          // hatch
      fLOGFillBrush.lbStyle = BS_HATCHED;
      switch (fasi)
        {
          case 1: fLOGFillBrush.lbHatch = HS_BDIAGONAL;
                  break;
          case 2: fLOGFillBrush.lbHatch = HS_CROSS;
                  break;
          case 3: fLOGFillBrush.lbHatch = HS_DIAGCROSS;
                  break;
          case 4: fLOGFillBrush.lbHatch = HS_FDIAGONAL;
                  break;
          case 5: fLOGFillBrush.lbHatch = HS_HORIZONTAL;
                  break;
          case 6: fLOGFillBrush.lbHatch = HS_VERTICAL;
                  break;
         default: fLOGFillBrush.lbHatch = HS_FDIAGONAL;
                  break;
        }
      fBrush =
           CreateHatchBrush(fLOGFillBrush.lbHatch,fLOGFillBrush.lbColor);
      break;

 default:                                          // solid  - default
      fLOGFillBrush.lbStyle = BS_NULL;
//      if (fLOGFillBrush.lbColor)
         fBrush = CreateSolidBrush(fLOGFillBrush.lbColor);
//      else
//         fBrush = GetStockObject(WHITE_BRUSH);
      break;

 }
}

//______________________________________________________________________________
void TGWin32Brush::SetColor(COLORREF ci){
  if (fLOGFillBrush.lbColor != ci) {
   fLOGFillBrush.lbColor = ci;

   switch (fLOGFillBrush.lbStyle)
     {
       case BS_HOLLOW  :
       case BS_PATTERN :
/*                 Hollow and Patten styles needn't a change of brush */
           break;
       case BS_HATCHED :
          if (fBrush) DeleteObject(fBrush);
          fBrush =
             CreateHatchBrush(fLOGFillBrush.lbHatch,fLOGFillBrush.lbColor);
          break;
       case BS_SOLID   :
       default         :
          if (fBrush) DeleteObject(fBrush);
//          if (fLOGFillBrush.lbColor)
            fBrush = CreateSolidBrush(fLOGFillBrush.lbColor);
//          else
//            fBrush = GetStockObject(WHITE_BRUSH);
          break;
     }
   }

}
