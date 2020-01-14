/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/Browsable/RGroup.hxx>

#include <ROOT/Browsable/RLevelIter.hxx>

#include <ROOT/RLogger.hxx>

using namespace ROOT::Experimental;
using namespace ROOT::Experimental::Browsable;


class RGroupIter : public RLevelIter {
   int fIndx{-1};
   RGroup &fComp;

public:

   explicit RGroupIter(RGroup &comp) : fComp(comp) {}
   virtual ~RGroupIter() = default;

   /** Shift to next element */
   bool Next() override { fIndx++; return HasItem(); }

   /** Is there current element  */
   bool HasItem() const override { return (fIndx >= 0) &&  (fIndx < (int) fComp.GetChilds().size()); }

   /** Returns current element name  */
   std::string GetName() const override { return fComp.GetChilds()[fIndx]->GetName(); }

   /** If element may have childs: 0 - no, >0 - yes, -1 - maybe */
   int CanHaveChilds() const override { return fComp.GetChilds().size(); }

   /** Returns full information for current element */
   std::shared_ptr<RElement> GetElement() override { return fComp.GetChilds()[fIndx]; }

   /** Reset iterator to the first element, returns false if not supported */
   bool Reset() override { fIndx = -1; return true; }

   /** Find item with specified name, use item MatchName() functionality */
   bool Find(const std::string &name) override
   {
      if (!Reset()) return false;

      while (Next()) {
         if (fComp.GetChilds()[fIndx]->MatchName(name))
            return true;
      }

      return false;
   }

};

/////////////////////////////////////////////////////////////////////
/// Create iterator for childs of composite

std::unique_ptr<RLevelIter> RGroup::GetChildsIter()
{
   return std::make_unique<RGroupIter>(*this);
}

