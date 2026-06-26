// Author:  Sergey Linev, GSI  26/06/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TQt6PadPainter.h"
#include "TError.h"
#include "TImage.h"
#include "TPad.h"


/** \class TQt6PadPainter
    \ingroup qt6canvas
    \brief Implement TVirtualPadPainter for Qt6 graphics
*/

//////////////////////////////////////////////////////////////////////////
/// Set opacity - similar to TVirtualPS usecase

void TQt6PadPainter::SetOpacity(Int_t percent)
{
   fAttFill.SetFillStyle(4000 + percent);
}

////////////////////////////////////////////////////////////////////////////////
///Noop, for non-gl pad TASImage calls gVirtualX->CopyArea.

void TQt6PadPainter::DrawPixels(const unsigned char * /*pixelData*/, UInt_t /*width*/, UInt_t /*height*/,
                             Int_t /*dstX*/, Int_t /*dstY*/, Bool_t /*enableAlphaBlending*/)
{

}


////////////////////////////////////////////////////////////////////////////////
/// Paint a simple line.

void TQt6PadPainter::DrawLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2)
{
   if (GetAttLine().GetLineWidth() <= 0)
      return;

}


////////////////////////////////////////////////////////////////////////////////
/// Paint a simple line in normalized coordinates.

void TQt6PadPainter::DrawLineNDC(Double_t u1, Double_t v1, Double_t u2, Double_t v2)
{
   if (GetAttLine().GetLineWidth() <= 0)
      return;
}


////////////////////////////////////////////////////////////////////////////////
/// Paint a simple box.

void TQt6PadPainter::DrawBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2, EBoxMode mode)
{
   if (GetAttLine().GetLineWidth() <= 0 && mode == TVirtualPadPainter::kHollow)
      return;

   // if (mode == TVirtualPadPainter::kHollow)
   //    draw only border
   // else
   //    draw only fill

}

////////////////////////////////////////////////////////////////////////////////
/// Paint filled area.

void TQt6PadPainter::DrawFillArea(Int_t nPoints, const Double_t *xs, const Double_t *ys)
{
   if ((GetAttFill().GetFillStyle() <= 0) || (nPoints < 3))
      return;
}

////////////////////////////////////////////////////////////////////////////////
/// Paint filled area.

void TQt6PadPainter::DrawFillArea(Int_t nPoints, const Float_t *xs, const Float_t *ys)
{
   if ((GetAttFill().GetFillStyle() <= 0) || (nPoints < 3))
      return;

}

////////////////////////////////////////////////////////////////////////////////
/// Paint Polyline.

void TQt6PadPainter::DrawPolyLine(Int_t nPoints, const Double_t *xs, const Double_t *ys)
{
   if ((GetAttLine().GetLineWidth() <= 0) || (nPoints < 2))
      return;
}

////////////////////////////////////////////////////////////////////////////////
/// Paint polyline.

void TQt6PadPainter::DrawPolyLine(Int_t nPoints, const Float_t *xs, const Float_t *ys)
{
   if ((GetAttLine().GetLineWidth() <= 0) || (nPoints < 2))
      return;
}

////////////////////////////////////////////////////////////////////////////////
/// Paint polyline in normalized coordinates.

void TQt6PadPainter::DrawPolyLineNDC(Int_t nPoints, const Double_t *u, const Double_t *v)
{
   if ((GetAttLine().GetLineWidth() <= 0) || (nPoints < 2))
      return;
}

////////////////////////////////////////////////////////////////////////////////
/// Paint polymarker.

void TQt6PadPainter::DrawPolyMarker(Int_t nPoints, const Double_t *x, const Double_t *y)
{
   if (nPoints < 1)
      return;
}

////////////////////////////////////////////////////////////////////////////////
/// Paint polymarker.

void TQt6PadPainter::DrawPolyMarker(Int_t nPoints, const Float_t *x, const Float_t *y)
{
   if (nPoints < 1)
      return;
}

////////////////////////////////////////////////////////////////////////////////
/// Paint text.

void TQt6PadPainter::DrawText(Double_t x, Double_t y, const char *text, ETextMode /*mode*/)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Paint text with url

void TQt6PadPainter::DrawTextUrl(Double_t x, Double_t y, const char *text, const char * /* url */)
{
}


////////////////////////////////////////////////////////////////////////////////
/// Special version working with wchar_t and required by TMathText.

void TQt6PadPainter::DrawText(Double_t x, Double_t y, const wchar_t * /*text*/, ETextMode /*mode*/)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Paint text in normalized coordinates.

void TQt6PadPainter::DrawTextNDC(Double_t u, Double_t v, const char *text, ETextMode /*mode*/)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Paint text in normalized coordinates.

void TQt6PadPainter::DrawTextNDC(Double_t  u , Double_t v, const wchar_t * /*text*/, ETextMode /*mode*/)
{
}


////////////////////////////////////////////////////////////////////////////////
/// Produce image

void TQt6PadPainter::SaveImage(TVirtualPad *pad, const char *fileName, Int_t /* gtype */) const
{
}

