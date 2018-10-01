// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_REveProjectionManager_hxx
#define ROOT_REveProjectionManager_hxx

#include "ROOT/REveElement.hxx"
#include "TAttBBox.h"
#include "ROOT/REveProjections.hxx"

namespace ROOT {
namespace Experimental {

class REveProjectionManager : public REveElementList, public TAttBBox {
private:
   REveProjectionManager(const REveProjectionManager &);            // Not implemented
   REveProjectionManager &operator=(const REveProjectionManager &); // Not implemented

protected:
   REveProjection *fProjections[REveProjection::kPT_End];

   REveProjection *fProjection; // current projection
   REveVector fCenter;          // center of distortion
   Float_t fCurrentDepth;       // z depth of object being projected

   List_t fDependentEls; // elements that depend on manager and need to be destroyed with it

   Bool_t fImportEmpty; // import sub-trees with no projectable elements

   virtual Bool_t ShouldImport(REveElement *el);
   virtual void UpdateDependentElsAndScenes(REveElement *root);

public:
   REveProjectionManager(REveProjection::EPType_e type = REveProjection::kPT_Unknown);
   virtual ~REveProjectionManager();

   void AddDependent(REveElement *el);
   void RemoveDependent(REveElement *el);

   void SetProjection(REveProjection::EPType_e type);
   REveProjection *GetProjection() { return fProjection; }

   virtual void UpdateName();

   void SetCenter(Float_t x, Float_t y, Float_t z);
   REveVector &GetCenter() { return fCenter; }

   void SetCurrentDepth(Float_t d) { fCurrentDepth = d; }
   Float_t GetCurrentDepth() const { return fCurrentDepth; }

   void SetImportEmpty(Bool_t ie) { fImportEmpty = ie; }
   Bool_t GetImportEmpty() const { return fImportEmpty; }

   virtual Bool_t HandleElementPaste(REveElement *el);

   virtual REveElement *ImportElementsRecurse(REveElement *el, REveElement *parent);
   virtual REveElement *ImportElements(REveElement *el, REveElement *ext_list = nullptr);

   virtual REveElement *SubImportElements(REveElement *el, REveElement *proj_parent);
   virtual Int_t SubImportChildren(REveElement *el, REveElement *proj_parent);

   virtual void ProjectChildren();
   virtual void ProjectChildrenRecurse(REveElement *el);

   virtual void ComputeBBox();

   ClassDef(REveProjectionManager, 0); // Manager class for steering of projections and managing projected objects.
};

} // namespace Experimental
} // namespace ROOT

#endif
