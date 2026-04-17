// @(#)root/gpad:$Id$
// Author:  Sergey Linev  17/04/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TPadPainterBase.h"
#include "TColor.h"


/** \class TPadPainterBase
\ingroup gpad

Extends TVirtualPadPainter interface to simplify work with graphical attributes
*/

////////////////////////////////////////////////////////////////////////////////
/// Returns fill attributes after modification
/// Checks for special fill styles 4000 .. 4100

TAttFill TPadPainterBase::GetAttFillInternal(Bool_t with_transparency)
{
   Style_t style = GetAttFill().GetFillStyle();
   Color_t color = GetAttFill().GetFillColor();

   fFullyTransparent = (style == 4000) || (style == 0);
   if (fFullyTransparent) {
      style = 0;
   } else if ((style > 4000) && (style <= 4100)) {
      if ((style < 4100) && with_transparency)
         color = TColor::GetColorTransparent(color, (style - 4000) / 100.);
      style = 1001;
   }

   return { color, style };
}
