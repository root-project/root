// @(#)root/gui:$Name:  $:$Id: TRootEmbeddedCanvas.h,v 1.11 2006/05/23 04:47:38 brun Exp $
// Author: Fons Rademakers   15/07/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TRootEmbeddedCanvas
#define ROOT_TRootEmbeddedCanvas

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootEmbeddedCanvas                                                  //
//                                                                      //
// This class creates a TGCanvas in which a TCanvas is created. Use     //
// GetCanvas() to get a pointer to the TCanvas.                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGCanvas
#include "TGCanvas.h"
#endif


class TCanvas;
class TRootEmbeddedContainer;


class TRootEmbeddedCanvas : public TGCanvas {

friend class TRootEmbeddedContainer;

protected:
   Int_t                   fCWinId;           // window id used by embedded TCanvas
   TRootEmbeddedContainer *fCanvasContainer;  // container in canvas widget
   TCanvas                *fCanvas;           // pointer to TCanvas
   Bool_t                  fAutoFit;          // canvas container keeps same size as canvas
   Int_t                   fButton;           // currently pressed button

   TRootEmbeddedCanvas(const TRootEmbeddedCanvas&);
   TRootEmbeddedCanvas& operator=(const TRootEmbeddedCanvas&);

   virtual Bool_t HandleContainerButton(Event_t *ev);
   virtual Bool_t HandleContainerDoubleClick(Event_t *ev);
   virtual Bool_t HandleContainerConfigure(Event_t *ev);
   virtual Bool_t HandleContainerKey(Event_t *ev);
   virtual Bool_t HandleContainerMotion(Event_t *ev);
   virtual Bool_t HandleContainerExpose(Event_t *ev);
   virtual Bool_t HandleContainerCrossing(Event_t *ev);

public:
   TRootEmbeddedCanvas(const char *name = 0, const TGWindow *p = 0, UInt_t w = 10,
            UInt_t h = 10, UInt_t options = kSunkenFrame | kDoubleBorder,
            Pixel_t back = GetDefaultFrameBackground());
   virtual ~TRootEmbeddedCanvas();

   void       AdoptCanvas(TCanvas *c);
   TCanvas   *GetCanvas() const { return fCanvas; }
   Int_t      GetCanvasWindowId() const { return fCWinId; }
   Bool_t     GetAutoFit() const { return fAutoFit; }
   void       SetAutoFit(Bool_t fit = kTRUE) { fAutoFit = fit; }
   virtual void SavePrimitive(ostream &out, Option_t *option = "");

   ClassDef(TRootEmbeddedCanvas,0)  //A ROOT TCanvas that can be embedded in a TGFrame
};

#endif
