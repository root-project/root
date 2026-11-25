// @(#)root/cocoa:$Id$
// Author: Sergey Linev   25/11/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "QuartzDrawable.h"

#include "TGCocoa.h"

QuartzDrawable::QuartzDrawable(Int_t id, TGCocoa &cocoa, NSObject<X11Drawable> *drawable)
   : XDrawable(id), fCocoa(cocoa), fDrawable(drawable)
{
}

QuartzDrawable::~QuartzDrawable()
{
}

////////////////////////////////////////////////////////////////////////////////
/// test if drawable is empty

Bool_t QuartzDrawable::IsNull() const
{
   return !fDrawable;
}

////////////////////////////////////////////////////////////////////////////////
/// Set fill attributes

void QuartzDrawable::SetAttFill(const TAttFill &att)
{
   fAttFill = att;
}

////////////////////////////////////////////////////////////////////////////////
/// Set line attributes

void QuartzDrawable::SetAttLine(const TAttLine &att)
{
   fAttLine = att;
}

////////////////////////////////////////////////////////////////////////////////
/// Set marker attributes for specified window

void QuartzDrawable::SetAttMarker(const TAttMarker &att)
{
   fAttMarker = att;
}

////////////////////////////////////////////////////////////////////////////////
/// Set text attributes for specified window

void QuartzDrawable::SetAttText(const TAttText &att)
{
   fAttText = att;
}

////////////////////////////////////////////////////////////////////////////////
/// Set window draw mode

void QuartzDrawable::SetDrawMode(TVirtualX::EDrawMode mode)
{
   fDrawMode = mode;
}

////////////////////////////////////////////////////////////////////////////////
/// Clear window

void QuartzDrawable::ClearWindow()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Close window

void QuartzDrawable::CloseWindow()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Update display for specified window.
///
///  \param [in] mode : (1) update (0) sync
///
/// Synchronise client and server once (not permanent).
/// Copy the pixmap gCws->fDrawing on the window gCws->fWindow
/// if the double buffer is on.

void QuartzDrawable::UpdateWindow(Int_t mode)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Set opacity. This image manipulation routine works
/// by adding to a percent amount of neutral to each pixels RGB.
/// Since it requires quite some additional color map entries is it
/// only supported on displays with more than > 8 color planes (> 256
/// colors).

void QuartzDrawable::SetOpacity(Int_t percent)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Copy the pixmap wid at the position xpos, ypos in the specified window.

void QuartzDrawable::CopyPixmap(Int_t wid, Int_t xpos, Int_t ypos)
{
}

void QuartzDrawable::DrawBox(Int_t x1, Int_t y1, Int_t x2, Int_t y2, TVirtualX::EBoxMode mode)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Fill area described by polygon on specified window
///
///  \param [in] n     number of points
///  \param [in] xy   list of points

void QuartzDrawable::DrawFillArea(Int_t n, TPoint *xy)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Draw a line
///
///  \param [in] x1,y1        : begin of line
///  \param [in] x2,y2        : end of line

void QuartzDrawable::DrawLine(Int_t x1, Int_t y1, Int_t x2, Int_t y2)
{
}


////////////////////////////////////////////////////////////////////////////////
/// Draw a line through all points
///
///  \param [in] n     number of points
///  \param [in] xy   list of points

void QuartzDrawable::DrawPolyLine(Int_t n, TPoint *xy)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Draws N segments between provided points
///
/// \param [in] n    number of segements
/// \param [in] xy   list of points, size 2*n

void QuartzDrawable::DrawLinesSegments(Int_t n, TPoint *xy)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Draw n markers with the current attributes at position x, y
///
///  \param [in] n     number of markers to draw
///  \param [in] xy   x,y coordinates of markers

void QuartzDrawable::DrawPolyMarker(Int_t n, TPoint *xy)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Draw a text string using current font
///
///  \param [in] mode       : drawing mode
///              - mode=0     : the background is not drawn (kClear)
///              - mode=1     : the background is drawn (kOpaque)
///  \param [in] x,y        : text position
///  \param [in] angle      : text angle
///  \param [in] mgn        : magnification factor
///  \param [in] text       : text string

void QuartzDrawable::DrawText(Int_t x, Int_t y, Float_t angle, Float_t mgn,
                         const char *text, TVirtualX::ETextMode mode)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Draw a text string using current font
///
///  \param [in] mode       : drawing mode
///              - mode=0     : the background is not drawn (kClear)
///              - mode=1     : the background is drawn (kOpaque)
///  \param [in] x,y        : text position
///  \param [in] angle      : text angle
///  \param [in] mgn        : magnification factor
///  \param [in] text       : text string

void QuartzDrawable::DrawText(Int_t x, Int_t y, Float_t angle, Float_t mgn,
                         const wchar_t *text, TVirtualX::ETextMode mode)
{
}

void QuartzDrawable::DrawTTFglyphs(Int_t, Int_t, TTFhandle &, TVirtualX::ETextMode)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Writes the specified window into GIF file. Returns 1 in case of success,
/// 0 otherwise.

Int_t QuartzDrawable::WriteGIF(const char *name)
{
}
