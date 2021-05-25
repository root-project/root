// @(#)root/gui:$Id$
// Author: Valeriy Onuchin   12/08/04

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVirtualDragManager
#define ROOT_TVirtualDragManager


#include "TGFrame.h"

enum EDragType { kDragNone, kDragMove, kDragCopy,
                 kDragLink, kDragLasso, kDragResize };


class TVirtualDragManager  {

protected:
   Bool_t         fMoveWaiting;  ///< kTRUE if source is clicked but not moved
   Bool_t         fDragging;     ///< in dragging mode?
   Bool_t         fDropping;     ///< drop is in progress
   Bool_t         fPasting;      ///< paste action is in progress
   EDragType      fDragType;     ///< dragging type
   TGFrame       *fSource;       ///< frame being dragged
   TGFrame       *fFrameUnder;   ///< frame under drag
   TGFrame       *fTarget;       ///< drop target
   TGFrame       *fPasteFrame;   ///<

protected:
   virtual void  Init();

public:
   TVirtualDragManager();
   virtual          ~TVirtualDragManager() {}

   EDragType         GetEDragType() const { return fDragType; }
   Bool_t            IsMoveWaiting() const { return fMoveWaiting; }
   Bool_t            IsDragging() const { return fDragging; }
   Bool_t            IsDropping() const { return fDropping; }
   Bool_t            IsPasting() const { return fPasting; }
   TGFrame          *GetTarget() const { return fTarget; }
   TGFrame          *GetSource() const { return fSource; }
   TGFrame          *GetFrameUnder() const { return fFrameUnder; }
   TGFrame          *GetPasteFrame() const { return fPasteFrame; }

   virtual void      SetTarget(TGFrame *f) { fTarget = f; }
   virtual void      SetSource(TGFrame *f) { fSource = f; }
   virtual void      SetPasteFrame(TGFrame *f) { fPasteFrame = f; }

   virtual Bool_t    StartDrag(TGFrame * = nullptr, Int_t = 0, Int_t = 0) { return kFALSE; }
   virtual Bool_t    EndDrag() { return kFALSE; }
   virtual Bool_t    Drop() { return kFALSE; }
   virtual Bool_t    Cancel(Bool_t = kTRUE) { return kFALSE; }

   virtual Bool_t    HandleEvent(Event_t *) { return kFALSE; }
   virtual Bool_t    HandleTimerEvent(Event_t *, TTimer *) { return kFALSE; }
   virtual Bool_t    IgnoreEvent(Event_t *) { return kTRUE; }
   virtual void      SetEditable(Bool_t) {}

   virtual Int_t     GetStrartDragX() const { return 0; }
   virtual Int_t     GetStrartDragY() const { return 0; }
   virtual Int_t     GetEndDragX() const { return 0; }
   virtual Int_t     GetEndDragY() const { return 0; }

   static  TVirtualDragManager  *Instance();

   ClassDef(TVirtualDragManager,0)  // drag and drop manager
};

R__EXTERN TVirtualDragManager *gDragManager; // global drag manager

#endif
