// @(#)root/base:$Name$:$Id$
// Author: Fons Rademakers   3/12/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualX                                                            //
//                                                                      //
// Semi-Abstract base class defining a generic interface to the         //
// underlying, low level, graphics system (X11, Win32, MacOS).          //
// An instance of TVirtualX itself defines a batch interface to the     //
// graphics system.                                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TVirtualX.h"
#include "TString.h"


Atom_t    gWM_DELETE_WINDOW;
Atom_t    gMOTIF_WM_HINTS;
Atom_t    gROOT_MESSAGE;

TVirtualX     *gVirtualX;      //Global pointer to the current graphics interface
TVirtualX     *gGXBatch;  //Global pointer to batch graphics interface

ClassImp(TVirtualX)

//______________________________________________________________________________
TVirtualX::TVirtualX(const char *name, const char *title) : TNamed(name, title),
      TAttLine(1,1,1),TAttFill(1,1),TAttText(11,0,1,62,0.01), TAttMarker(1,1,1)
{
   // Ctor of ABC
}

//______________________________________________________________________________
void TVirtualX::GetWindowAttributes(Window_t, WindowAttributes_t &attr)
{
   // Set WindowAttributes_t structure to defaults.

   attr.fX = attr.fY = 0;
   attr.fWidth = attr.fHeight = 0;
   attr.fVisual   = 0;
   attr.fMapState = kIsUnmapped;
   attr.fScreen   = 0;
}

//______________________________________________________________________________
Bool_t TVirtualX::ParseColor(Colormap_t, const char *, ColorStruct_t &color)
{
   // Set ColorStruct_t structure to default. Let system think we could
   // parse color.

   color.fPixel = 0;
   color.fRed   = 0;
   color.fGreen = 0;
   color.fBlue  = 0;
   color.fMask  = kDoRed | kDoGreen | kDoBlue;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TVirtualX::AllocColor(Colormap_t, ColorStruct_t &color)
{
   // Set pixel value. Let system think we could alocate color.

   color.fPixel = 0;
   return kTRUE;
}

//______________________________________________________________________________
void TVirtualX::QueryColor(Colormap_t, ColorStruct_t &color)
{
   // Set color components to default.

   color.fRed = color.fGreen = color.fBlue = 0;
}

//______________________________________________________________________________
void TVirtualX::NextEvent(Event_t &event)
{
   // Set to default event. This method however, should never be called.

   event.fType   = kButtonPress;
   event.fWindow = 0;
   event.fTime   = 0;
   event.fX      = 0;
   event.fY      = 0;
   event.fXRoot  = 0;
   event.fYRoot  = 0;
   event.fState  = 0;
   event.fCode   = 0;
   event.fWidth  = 0;
   event.fHeight = 0;
   event.fCount  = 0;
}

//______________________________________________________________________________
void TVirtualX::GetPasteBuffer(Window_t, Atom_t, TString &text, Int_t &nchar, Bool_t)
{
   // Get paste buffer. By default always empty.

   text = "";
   nchar = 0;
}
