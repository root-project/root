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

#include "TEveElement.h"
#include "TEveProjections.h"
#include "TEveVSDStructs.h"

class TEveProjectionManager : public TEveElementList
{
private:
   TEveProjectionManager(const TEveProjectionManager&);            // Not implemented
   TEveProjectionManager& operator=(const TEveProjectionManager&); // Not implemented

protected:
   TEveProjection* fProjection;     // projection
   TEveVector      fCenter;         // center of distortion
   Float_t         fCurrentDepth;   // z depth of object being projected

   Float_t         fBBox[6];        // projected children bounding box

   List_t          fDependentEls;   // elements that depend on manager and need to be destroyed with it

   virtual Bool_t  ShouldImport(TEveElement* rnr_el);

public:
   TEveProjectionManager();
   virtual ~TEveProjectionManager();

   void AddDependent(TEveElement* el);
   void RemoveDependent(TEveElement* el);

   void            SetProjection(TEveProjection::EPType_e type, Float_t distort=0);
   TEveProjection* GetProjection() { return fProjection; }

   virtual void    UpdateName();

   void            SetCenter(Float_t x, Float_t y, Float_t z);
   TEveVector&     GetCenter(){return fCenter;}

   void            SetCurrentDepth(Float_t d) { fCurrentDepth = d;      }
   Float_t         GetCurrentDepth()    const { return fCurrentDepth;   }

   virtual Bool_t  HandleElementPaste(TEveElement* el);
   virtual void    ImportElementsRecurse(TEveElement* rnr_el, TEveElement* parent);
   virtual void    ImportElements(TEveElement* rnr_el);
   virtual void    ProjectChildren();
   virtual void    ProjectChildrenRecurse(TEveElement* rnr_el);

   Float_t*        GetBBox() { return &fBBox[0]; }

   ClassDef(TEveProjectionManager, 0); // Manager class for steering of projections and managing projected objects.
};

#endif
