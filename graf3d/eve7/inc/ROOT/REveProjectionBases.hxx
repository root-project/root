// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_REveProjectionBases
#define ROOT7_REveProjectionBases

#include "Rtypes.h"
#include <list>
#include <set>

class TClass;

namespace ROOT {
namespace Experimental {

class REveElement;
class REveProjection;
class REveProjected;
class REveProjectionManager;

////////////////////////////////////////////////////////////////
//                                                            //
// REveProjectable                                            //
//                                                            //
// Abstract base class for non-linear projectable objects.    //
//                                                            //
////////////////////////////////////////////////////////////////

class REveProjectable
{
private:
   REveProjectable &operator=(const REveProjectable &); // Not implemented

public:
   typedef std::list<REveProjected *> ProjList_t;
   typedef ProjList_t::iterator       ProjList_i;

protected:
   ProjList_t fProjectedList; // references to projected instances.

public:
   REveProjectable();
   REveProjectable(const REveProjectable &);
   virtual ~REveProjectable();

   virtual TClass *ProjectedClass(const REveProjection *p) const = 0;

   virtual Bool_t HasProjecteds() const { return !fProjectedList.empty(); }

   ProjList_t &RefProjecteds()   { return fProjectedList;         }
   ProjList_i  BeginProjecteds() { return fProjectedList.begin(); }
   ProjList_i  EndProjecteds()   { return fProjectedList.end();   }

   virtual void AddProjected(REveProjected *p) { fProjectedList.push_back(p); }
   virtual void RemoveProjected(REveProjected *p) { fProjectedList.remove(p); }

   virtual void AnnihilateProjecteds();
   virtual void ClearProjectedList();

   virtual void AddProjectedsToSet(std::set<REveElement *> &set);

   virtual void PropagateVizParams(REveElement *el = nullptr);
   virtual void PropagateRenderState(Bool_t rnr_self, Bool_t rnr_children);
   virtual void PropagateMainColor(Color_t color, Color_t old_color);
   virtual void PropagateMainTransparency(Char_t t, Char_t old_t);

   ClassDef(REveProjectable, 0); // Abstract base class for classes that can be transformed with non-linear projections.
};

////////////////////////////////////////////////////////////////
//                                                            //
// REveProjected                                              //
//                                                            //
// Abstract base class for non-linear projected objects.      //
//                                                            //
////////////////////////////////////////////////////////////////

class REveProjected {
private:
   REveProjected(const REveProjected &);            // Not implemented
   REveProjected &operator=(const REveProjected &); // Not implemented

protected:
   REveProjectionManager *fManager{nullptr}; // manager
   REveProjectable *fProjectable{nullptr};   // link to original object
   Float_t fDepth{0.};                       // z coordinate

   void SetDepthCommon(Float_t d, REveElement *el, Float_t *bbox);
   virtual void SetDepthLocal(Float_t d);

public:
   REveProjected() = default;
   virtual ~REveProjected();

   REveProjectionManager *GetManager() const { return fManager; }
   REveProjectable *GetProjectable() const { return fProjectable; }
   Float_t GetDepth() const { return fDepth; }

   virtual void SetProjection(REveProjectionManager *mng, REveProjectable *model);
   virtual void UnRefProjectable(REveProjectable *assumed_parent, bool notifyParent = true);

   virtual void UpdateProjection() = 0;
   virtual REveElement *GetProjectedAsElement();

   virtual void SetDepth(Float_t d);

   ClassDef(REveProjected, 0); // Abstract base class for classes that hold results of a non-linear projection transformation.
};

} // namespace Experimental
} // namespace ROOT

#endif
