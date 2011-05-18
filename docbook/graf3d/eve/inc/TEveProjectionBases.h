// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveProjectionBases
#define ROOT_TEveProjectionBases

#include "Rtypes.h"
#include <list>
#include <set>

class TEveElement;
class TEveProjection;
class TEveProjected;
class TEveProjectionManager;

class TClass;

////////////////////////////////////////////////////////////////
//                                                            //
// TEveProjectable                                            //
//                                                            //
// Abstract base class for non-linear projectable objects.    //
//                                                            //
////////////////////////////////////////////////////////////////

class TEveProjectable
{
private:
   TEveProjectable(const TEveProjectable&);            // Not implemented
   TEveProjectable& operator=(const TEveProjectable&); // Not implemented

public:
   typedef std::list<TEveProjected*>            ProjList_t;
   typedef std::list<TEveProjected*>::iterator  ProjList_i;

protected:
   ProjList_t       fProjectedList; // references to projected instances.

public:
   TEveProjectable();
   virtual ~TEveProjectable();

   virtual TClass* ProjectedClass(const TEveProjection* p) const = 0;

   virtual Bool_t HasProjecteds() const { return ! fProjectedList.empty(); }

   ProjList_i   BeginProjecteds()       { return  fProjectedList.begin(); }
   ProjList_i   EndProjecteds()         { return  fProjectedList.end();   }

   virtual void AddProjected(TEveProjected* p)    { fProjectedList.push_back(p); }
   virtual void RemoveProjected(TEveProjected* p) { fProjectedList.remove(p);   }

   virtual void AnnihilateProjecteds();
   virtual void ClearProjectedList();

   virtual void AddProjectedsToSet(std::set<TEveElement*>& set);

   virtual void PropagateVizParams(TEveElement* el=0);
   virtual void PropagateRenderState(Bool_t rnr_self, Bool_t rnr_children);
   virtual void PropagateMainColor(Color_t color, Color_t old_color);
   virtual void PropagateMainTransparency(Char_t t, Char_t old_t);

   ClassDef(TEveProjectable, 0); // Abstract base class for classes that can be transformed with non-linear projections.
};


////////////////////////////////////////////////////////////////
//                                                            //
// TEveProjected                                              //
//                                                            //
// Abstract base class for non-linear projected objects.      //
//                                                            //
////////////////////////////////////////////////////////////////

class TEveProjected
{
private:
   TEveProjected(const TEveProjected&);            // Not implemented
   TEveProjected& operator=(const TEveProjected&); // Not implemented

protected:
   TEveProjectionManager *fManager;       // manager
   TEveProjectable       *fProjectable;   // link to original object
   Float_t                fDepth;         // z coordinate

   void         SetDepthCommon(Float_t d, TEveElement* el, Float_t* bbox);
   virtual void SetDepthLocal(Float_t d);

public:
   TEveProjected();
   virtual ~TEveProjected();

   TEveProjectionManager* GetManager()     const { return fManager; }
   TEveProjectable*       GetProjectable() const { return fProjectable; }
   Float_t                GetDepth()       const { return fDepth; }

   virtual void SetProjection(TEveProjectionManager* mng, TEveProjectable* model);
   virtual void UnRefProjectable(TEveProjectable* assumed_parent, bool notifyParent = true);

   virtual void UpdateProjection() = 0;   
   virtual TEveElement* GetProjectedAsElement();

   virtual void SetDepth(Float_t d);

   ClassDef(TEveProjected, 0); // Abstract base class for classes that hold results of a non-linear projection transformation.
};

#endif
