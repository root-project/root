// @(#)root/gui:$Id$
// Author: Bertrand Bellenot 23/01/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGSplitFrame
#define ROOT_TGSplitFrame

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

class TGSplitter;

class TGSplitFrame : public TGCompositeFrame {

private:
   TGSplitFrame(const TGSplitFrame&); // Not implemented
   TGSplitFrame& operator=(const TGSplitFrame&); // Not implemented

protected:
   TGFrame        *fFrame;       // Pointer to the embedded frame (if any)
   TGSplitter     *fSplitter;    // Pointer to the (H/V) Splitter (if any)
   TGSplitFrame   *fFirst;       // Pointer to the first child (if any)
   TGSplitFrame   *fSecond;      // Pointer to the second child (if any)
   Float_t         fWRatio;      // Width ratio between the first child and this
   Float_t         fHRatio;      // Height ratio between the first child and this

public:
   TGSplitFrame(const TGWindow *p = 0, UInt_t w = 1, UInt_t h = 1,
                UInt_t options = 0);
   virtual ~TGSplitFrame();

   virtual void   AddFrame(TGFrame *f, TGLayoutHints *l = 0);
   virtual void   HSplit(UInt_t h = 0);
   virtual void   VSplit(UInt_t w = 0);
   virtual void   Cleanup();
   virtual Bool_t HandleConfigureNotify(Event_t *);
   TGSplitFrame  *GetFirst() const { return fFirst; }
   TGFrame       *GetFrame() const { return fFrame; }
   TGSplitFrame  *GetSecond() const { return fSecond; }
   TGSplitter    *GetSplitter() const { return fSplitter; }
   Float_t        GetHRatio() const { return fHRatio; }
   Float_t        GetWRatio() const { return fWRatio; }
   void           SetHRatio(Float_t r) { fHRatio = r; }
   void           SetWRatio(Float_t r) { fWRatio = r; }

   static  void   SwitchFrames(TGFrame *frame, TGCompositeFrame *dest,
                               TGFrame *prev);
   virtual void   SavePrimitive(ostream &out, Option_t *option = "");

   ClassDef(TGSplitFrame, 0) // Splittable composite frame
};

#endif
