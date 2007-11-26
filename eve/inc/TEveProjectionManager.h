// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveProjectionManager
#define ROOT_TEveProjectionManager

#include <TAtt3D.h>
#include <TAttBBox.h>

#include <TEveElement.h>
#include <TEveProjections.h>
#include <TEveVSDStructs.h>

class TEveProjectionManager : public TEveElementList,
                              public TAttBBox,
                              public TAtt3D
{
private:
   TEveProjectionManager(const TEveProjectionManager&);            // Not implemented
   TEveProjectionManager& operator=(const TEveProjectionManager&); // Not implemented

   TEveProjection* fProjection;  // projection

   Bool_t          fDrawCenter;  // draw center of distortion
   Bool_t          fDrawOrigin;  // draw origin
   TEveVector      fCenter;      // center of distortion

   Int_t           fSplitInfoMode;  // tick-mark position
   Int_t           fSplitInfoLevel; // tick-mark density
   Color_t         fAxisColor;      // color of axis

   Float_t         fCurrentDepth;   // z depth of object being projected

   virtual Bool_t  ShouldImport(TEveElement* rnr_el);

public:
   TEveProjectionManager();
   virtual ~TEveProjectionManager();

   void            SetProjection(TEveProjection::PType_e type, Float_t distort=0);
   TEveProjection* GetProjection() { return fProjection; }

   virtual void    UpdateName();

   void            SetAxisColor(Color_t col)  { fAxisColor = col;       }
   Color_t         GetAxisColor()       const { return fAxisColor;      }
   void            SetSplitInfoMode(Int_t x)  { fSplitInfoMode = x;     }
   Int_t           GetSplitInfoMode()   const { return fSplitInfoMode;  }
   void            SetSplitInfoLevel(Int_t x) { fSplitInfoLevel = x;    }
   Int_t           GetSplitInfoLevel()  const { return fSplitInfoLevel; }

   void            SetDrawCenter(Bool_t x){ fDrawCenter = x; }
   Bool_t          GetDrawCenter(){ return fDrawCenter; }
   void            SetDrawOrigin(Bool_t x){ fDrawOrigin = x; }
   Bool_t          GetDrawOrigin(){ return fDrawOrigin; }

   void            SetCenter(Float_t x, Float_t y, Float_t z);
   TEveVector&     GetCenter(){return fCenter;}

   void            SetCurrentDepth(Float_t d) { fCurrentDepth = d;      }
   Float_t         GetCurrentDepth()    const { return fCurrentDepth;   }

   virtual Bool_t  HandleElementPaste(TEveElement* el);
   virtual void    ImportElementsRecurse(TEveElement* rnr_el, TEveElement* parent);
   virtual void    ImportElements(TEveElement* rnr_el);
   virtual void    ProjectChildren();
   virtual void    ProjectChildrenRecurse(TEveElement* rnr_el);

   virtual void    ComputeBBox();
   virtual void    Paint(Option_t* option = "");

   ClassDef(TEveProjectionManager, 0); // Manager class for steering of projections and managing projected objects.
};

#endif
