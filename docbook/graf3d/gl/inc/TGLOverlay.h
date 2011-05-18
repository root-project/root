// @(#)root/gl:$Id$
// Author:  Matevz Tadel, Feb 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLOverlay_H
#define ROOT_TGLOverlay_H

#include <GuiTypes.h>

class TGLRnrCtx;
class TGLOvlSelectRecord;

#include <list>

class TGLOverlayElement
{
public:
   enum ERole  { kUser, kViewer, kAnnotation, kAll };

   enum EState { kInvisible = 1, kDisabled = 2, kActive = 4,
                 kAllVisible = kDisabled | kActive };

private:
   TGLOverlayElement(const TGLOverlayElement&);            // Not implemented
   TGLOverlayElement& operator=(const TGLOverlayElement&); // Not implemented

protected:
   ERole   fRole;
   EState  fState;

   void ProjectionMatrixPushIdentity();

public:
   TGLOverlayElement(ERole r=kUser, EState s=kActive) :
      fRole(r), fState(s) {}
   virtual ~TGLOverlayElement() {}

   virtual Bool_t MouseEnter(TGLOvlSelectRecord& selRec);
   virtual Bool_t MouseStillInside(TGLOvlSelectRecord& selRec);
   virtual Bool_t Handle(TGLRnrCtx& rnrCtx, TGLOvlSelectRecord& selRec,
                         Event_t* event);
   virtual void   MouseLeave();

   virtual void Render(TGLRnrCtx& rnrCtx) = 0;

   ERole   GetRole() const  { return fRole; }
   void    SetRole(ERole r) { fRole = r; }

   EState  GetState() const   { return fState; }
   void    SetState(EState s) { fState = s; }

   void    SetBinaryState(Bool_t s) { SetState(s ? kActive : kInvisible); }

   ClassDef(TGLOverlayElement, 0) // Base class for GL overlay elements.
};


class TGLOverlayList
{
private:
   TGLOverlayList(const TGLOverlayList&);            // Not implemented
   TGLOverlayList& operator=(const TGLOverlayList&); // Not implemented

protected:
   std::list<TGLOverlayElement*> fElements;

public:
   TGLOverlayList() : fElements() {}
   virtual ~TGLOverlayList() {}

   // void AddElement(TGLOverlayElement* element);
   // void RemoveElement(TGLOverlayElement* element);

   // TGLOverlayElement* SelectElement(TGLSelectRecord& selRec, Int_t nameOff);

   ClassDef(TGLOverlayList, 0) // Collection of overlay elements to draw/select together.
};


#endif
