// @(#)root/gui:$Id$
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
class TDNDData;

class TRootEmbeddedCanvas : public TGCanvas {

friend class TRootEmbeddedContainer;

protected:
   Int_t                   fCWinId;           // window id used by embedded TCanvas
   TRootEmbeddedContainer *fCanvasContainer;  // container in canvas widget
   TCanvas                *fCanvas;           // pointer to TCanvas
   Bool_t                  fAutoFit;          // canvas container keeps same size as canvas
   Int_t                   fButton;           // currently pressed button
   Atom_t                 *fDNDTypeList;      // handles DND types
   
   virtual Bool_t HandleContainerButton(Event_t *ev);
   virtual Bool_t HandleContainerDoubleClick(Event_t *ev);
   virtual Bool_t HandleContainerConfigure(Event_t *ev);
   virtual Bool_t HandleContainerKey(Event_t *ev);
   virtual Bool_t HandleContainerMotion(Event_t *ev);
   virtual Bool_t HandleContainerExpose(Event_t *ev);
   virtual Bool_t HandleContainerCrossing(Event_t *ev);

private:
   TRootEmbeddedCanvas(const TRootEmbeddedCanvas&);              // not implemented
   TRootEmbeddedCanvas& operator=(const TRootEmbeddedCanvas&);   // not implemented

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

   virtual Bool_t HandleDNDDrop(TDNDData *data);
   virtual Atom_t HandleDNDPosition(Int_t /*x*/, Int_t /*y*/, Atom_t action,
                                    Int_t /*xroot*/, Int_t /*yroot*/);
   virtual Atom_t HandleDNDEnter(Atom_t * typelist);
   virtual Bool_t HandleDNDLeave();

   ClassDef(TRootEmbeddedCanvas,0)  //A ROOT TCanvas that can be embedded in a TGFrame
};

#endif
