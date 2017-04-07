// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov, 19/10/2011

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <iostream>

#include "IOSPad.h"
#include "TGIOS.h"


namespace ROOT {
namespace iOS {

////////////////////////////////////////////////////////////////////////////////
/// Default ctor.

TGIOS::TGIOS()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Ctor.

TGIOS::TGIOS(const char *name, const char *title)
         : TVirtualX(name, title)
{
}

//TAttLine.
////////////////////////////////////////////////////////////////////////////////
/// Sets color index "cindex" for drawing lines.

void TGIOS::SetLineColor(Color_t cindex)
{
   TAttLine::SetLineColor(cindex);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the line style.
///
/// linestyle <= 1 solid
/// linestyle  = 2 dashed
/// linestyle  = 3 dotted
/// linestyle  = 4 dashed-dotted

void TGIOS::SetLineStyle(Style_t linestyle)
{
   TAttLine::SetLineStyle(linestyle);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the line width.
///
/// width - the line width in pixels

void TGIOS::SetLineWidth(Width_t width)
{
   TAttLine::SetLineWidth(width);
}

//TAttFill.
////////////////////////////////////////////////////////////////////////////////
/// Sets color index "cindex" for fill areas.

void TGIOS::SetFillColor(Color_t cindex)
{
   TAttFill::SetFillColor(cindex);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets fill area style.
///
/// style - compound fill area interior style
///         style = 1000 * interiorstyle + styleindex

void TGIOS::SetFillStyle(Style_t style)
{
   TAttFill::SetFillStyle(style);
}

//TAttMarker.
////////////////////////////////////////////////////////////////////////////////
/// Sets color index "cindex" for markers.

void TGIOS::SetMarkerColor(Color_t cindex)
{
   TAttMarker::SetMarkerColor(cindex);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets marker size index.
///
/// markersize - the marker scale factor

void TGIOS::SetMarkerSize(Float_t markersize)
{
   TAttMarker::SetMarkerSize(markersize);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets marker style.

void TGIOS::SetMarkerStyle(Style_t markerstyle)
{
   TAttMarker::SetMarkerStyle(markerstyle);
}

//TAttText.
////////////////////////////////////////////////////////////////////////////////
/// Sets the text alignment.
///
/// talign = txalh horizontal text alignment
/// talign = txalv vertical text alignment

void TGIOS::SetTextAlign(Short_t talign)
{
   TAttText::SetTextAlign(talign);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the color index "cindex" for text.

void TGIOS::SetTextColor(Color_t cindex)
{
   TAttText::SetTextColor(cindex);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the current text font number.

void TGIOS::SetTextFont(Font_t fontnumber)
{
   TAttText::SetTextFont(fontnumber);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the current text size to "textsize"

void TGIOS::SetTextSize(Float_t textsize)
{
   TAttText::SetTextSize(textsize);
}

////////////////////////////////////////////////////////////////////////////////
///With all these global variables like gVirtualX and gPad, I have to use this trick.

void TGIOS::GetTextExtent(UInt_t &w, UInt_t &h, char *textLine)
{
   Pad *p = static_cast<Pad *>(gPad);
   p->GetTextExtent(w, h, textLine);
}

}
}
