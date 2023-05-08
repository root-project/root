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
#include "TAttBBox.h"
#include "TEveProjections.h"


class TEveProjectionManager : public TEveElementList,
                              public TAttBBox
{
private:
   TEveProjectionManager(const TEveProjectionManager&);            // Not implemented
   TEveProjectionManager& operator=(const TEveProjectionManager&); // Not implemented

protected:
   TEveProjection* fProjections[TEveProjection::kPT_End];

   TEveProjection* fProjection;     // current projection
   TEveVector      fCenter;         // center of distortion
   Float_t         fCurrentDepth;   // z depth of object being projected

   List_t          fDependentEls;   // elements that depend on manager and need to be destroyed with it

   Bool_t          fImportEmpty;    // import sub-trees with no projectable elements

   virtual Bool_t  ShouldImport(TEveElement* el);
   virtual void    UpdateDependentElsAndScenes(TEveElement* root);

public:
   TEveProjectionManager(TEveProjection::EPType_e type=TEveProjection::kPT_Unknown);
   virtual ~TEveProjectionManager();

   void AddDependent(TEveElement* el);
   void RemoveDependent(TEveElement* el);

   void            SetProjection(TEveProjection::EPType_e type);
   TEveProjection* GetProjection() { return fProjection; }

   virtual void    UpdateName();

   void            SetCenter(Float_t x, Float_t y, Float_t z);
   TEveVector&     GetCenter() { return fCenter; }

   void            SetCurrentDepth(Float_t d) { fCurrentDepth = d;      }
   Float_t         GetCurrentDepth()    const { return fCurrentDepth;   }

   void            SetImportEmpty(Bool_t ie)  { fImportEmpty = ie;   }
   Bool_t          GetImportEmpty()     const { return fImportEmpty; }

   virtual Bool_t  HandleElementPaste(TEveElement* el);

   virtual TEveElement* ImportElementsRecurse(TEveElement* el,
                                              TEveElement* parent);
   virtual TEveElement* ImportElements(TEveElement* el,
                                       TEveElement* ext_list = nullptr);

   virtual TEveElement* SubImportElements(TEveElement* el, TEveElement* proj_parent);
   virtual Int_t        SubImportChildren(TEveElement* el, TEveElement* proj_parent);

   virtual void    ProjectChildren();
   virtual void    ProjectChildrenRecurse(TEveElement* el);

   virtual void    ComputeBBox();

   ClassDef(TEveProjectionManager, 0); // Manager class for steering of projections and managing projected objects.
};

#endif
