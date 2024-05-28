// @(#)root/gl:$Id$
// Author:  Matevz Tadel, Feb 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLManipSet
#define ROOT_TGLManipSet

#include "TGLOverlay.h"
#include "TGLPShapeRef.h"

class TGLManip;

class TGLManipSet : public TGLOverlayElement,
                    public TGLPShapeRef
{
public:
   enum EManip { kTrans, kScale, kRotate, kEndType };

private:
   TGLManipSet(const TGLManipSet&) = delete;
   TGLManipSet& operator=(const TGLManipSet&) = delete;

protected:
   TGLManip  *fManip[3]; //! manipulator store
   EManip     fType;     //! current manipulator

   Bool_t     fDrawBBox; //! also draw bounding-box around physical

public:
   TGLManipSet();
   ~TGLManipSet() override;

   void SetPShape(TGLPhysicalShape* shape) override;

   Bool_t MouseEnter(TGLOvlSelectRecord& selRec) override;
   Bool_t Handle(TGLRnrCtx& rnrCtx, TGLOvlSelectRecord& selRec,
                         Event_t* event) override;
   void   MouseLeave() override;

   void Render(TGLRnrCtx& rnrCtx) override;

   TGLManip* GetCurrentManip() const { return fManip[fType]; }

   Int_t  GetManipType()   const { return fType; }
   void   SetManipType(Int_t type);
   Bool_t GetDrawBBox()    const { return fDrawBBox; }
   void   SetDrawBBox(Bool_t bb) { fDrawBBox = bb; }

   ClassDefOverride(TGLManipSet, 0); // A collection of available manipulators.
}; // endclass TGLManipSet


#endif
