// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGPack
#define ROOT_TGPack

#include "TGFrame.h"
#include "TGLayout.h"

class TGSplitter;


class  TGFrameElementPack : public TGFrameElement
{
private:
   TGFrameElementPack(const TGFrameElementPack&) = delete;
   TGFrameElementPack& operator=(const TGFrameElementPack&) = delete;

public:
   Float_t fWeight;              ///< relative weight
   TGFrameElementPack* fSplitFE; ///<! cached variable for optimisation

   TGFrameElementPack(TGFrame *frame, TGLayoutHints* lh = nullptr, Float_t weight = 1):
      TGFrameElement(frame, lh), fWeight(weight), fSplitFE(nullptr) { }

   ClassDefOverride(TGFrameElementPack, 0); // Class used in TGPack.
};

//==============================================================================

class TGPack : public TGCompositeFrame
{
private:
   TGPack(const TGPack&) = delete;
   TGPack& operator=(const TGPack&) = delete;

protected:
   Bool_t         fVertical;
   Bool_t         fUseSplitters;
   Int_t          fSplitterLen;

   Int_t          fDragOverflow;  ///<!

   Float_t        fWeightSum;     ///< total sum of sub  frame weights
   Int_t          fNVisible;      ///<  number of visible frames

   Int_t          GetFrameLength(const TGFrame* f) const { return fVertical ? f->GetHeight() : f->GetWidth(); }
   Int_t          GetLength()                      const { return GetFrameLength(this); }
   Int_t          GetAvailableLength()             const;

   void           SetFrameLength  (TGFrame* f, Int_t len);
   void           SetFramePosition(TGFrame* f, Int_t pos);

   void           FindFrames(TGFrame* splitter, TGFrameElementPack*& f0, TGFrameElementPack*& f1) const;

   void           CheckSplitterVisibility();
   void           ResizeExistingFrames();
   void           RefitFramesToPack();

   void           AddFrameInternal(TGFrame *f, TGLayoutHints* l = nullptr, Float_t weight = 1);
   void           RemoveFrameInternal(TGFrame *f);

public:
   TGPack(const TGWindow *p = nullptr, UInt_t w = 1, UInt_t h = 1, UInt_t options = 0,
          Pixel_t back = GetDefaultFrameBackground());
   TGPack(TGClient *c, Window_t id, const TGWindow *parent = nullptr);
   virtual ~TGPack();

   virtual void AddFrameWithWeight(TGFrame *f, TGLayoutHints* l, Float_t w);
   void AddFrame(TGFrame *f, TGLayoutHints* l = nullptr) override;

   virtual void DeleteFrame(TGFrame *f);
   void RemoveFrame(TGFrame *f) override;
   void ShowFrame(TGFrame *f) override;
   void HideFrame(TGFrame *f) override;

   using TGCompositeFrame::Resize;
   void Resize(UInt_t w = 0, UInt_t h = 0) override;

   using TGCompositeFrame::MapSubwindows;
   void MapSubwindows() override;

   void MoveResize(Int_t x, Int_t y, UInt_t w = 0, UInt_t h = 0) override;
   void Layout() override;

   void Dump() const override;

   void EqualizeFrames();
   void HandleSplitterStart();
   void HandleSplitterResize(Int_t delta);

   // ----------------------------------------------------------------

   Bool_t GetVertical() const { return fVertical; }
   void   SetVertical(Bool_t x);

   // For now assume this is always true. Length of splitter = 4 pixels.
   Bool_t GetUseSplitters() const { return fUseSplitters; }
   void SetUseSplitters(Bool_t x) { fUseSplitters = x; }

   ClassDefOverride(TGPack, 0); // Horizontal or vertical stack of frames.
};

#endif
