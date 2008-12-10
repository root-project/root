// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGPack
#define ROOT_TGPack

#include "TGFrame.h"

class TGPack : public TGCompositeFrame
{
private:
   TGPack(const TGPack&);            // Not implemented
   TGPack& operator=(const TGPack&); // Not implemented

protected:
   Bool_t         fVertical;
   Bool_t         fUseSplitters;
   Int_t          fSplitterLen;

   Int_t          fDragOverflow;  //!

   Int_t          GetFrameLength(const TGFrame* f) const { return fVertical ? f->GetHeight() : f->GetWidth(); }
   Int_t          GetLength()                      const { return GetFrameLength(this); }
   Int_t          GetAvailableLength()             const;

   void           SetFrameLength  (TGFrame* f, Int_t len);
   void           SetFramePosition(TGFrame* f, Int_t pos);

   Int_t          NumberOfRealFrames() const;
   Int_t          LengthOfRealFrames() const;

   void           ResizeExistingFrames(Int_t amount);
   void           ExpandExistingFrames(Int_t amount);
   void           ShrinkExistingFrames(Int_t amount);
   void           RefitFramesToPack();

   void           FindFrames(TGFrame* splitter, TGFrame*& f0, TGFrame*& f1);

   void           AddFrameInternal(TGFrame *f, TGLayoutHints* l = 0);
   Int_t          RemoveFrameInternal(TGFrame *f);

public:
   TGPack(const TGWindow *p = 0, UInt_t w = 1, UInt_t h = 1, UInt_t options = 0,
          Pixel_t back = GetDefaultFrameBackground());
   TGPack(TGClient *c, Window_t id, const TGWindow *parent = 0);
   virtual ~TGPack();

   virtual void   AddFrame(TGFrame *f, TGLayoutHints *l = 0);
   virtual void   DeleteFrame(TGFrame *f);
   virtual void   RemoveFrame(TGFrame *f);
   virtual void   ShowFrame(TGFrame *f);
   virtual void   HideFrame(TGFrame *f);

   using          TGCompositeFrame::Resize;
   virtual void   Resize(UInt_t w = 0, UInt_t h = 0);
   virtual void   MoveResize(Int_t x, Int_t y, UInt_t w = 0, UInt_t h = 0);
   virtual void   Layout();

   void           EqualizeFrames();

   void HandleSplitterStart();
   void HandleSplitterResize(Int_t delta);

   // ----------------------------------------------------------------

   Bool_t GetVertical() const { return fVertical; }
   void   SetVertical(Bool_t x);

   // For now assume this is always true. Lenght of splitter = 4 pixels.
   // Bool_t GetUseSplitters() const { return fUseSplitters; }
   // void SetUseSplitters(Bool_t x) { fUseSplitters = x; }

   ClassDef(TGPack, 0); // Horizontal or vertical stack of frames.
};

#endif
