/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/Browsable/RProvider.hxx>

#include "TLeaf.h"
#include "TBranch.h"
#include "TTree.h"
#include "TH1.h"
#include "TDirectory.h"

using namespace ROOT::Experimental::Browsable;

/** Provider for drawing of ROOT6 classes */

class TLeafProvider : public RProvider {
public:

   virtual ~TLeafProvider() = default;

   static TH1 *DrawLeaf(std::unique_ptr<RHolder> &obj)
   {
      auto tleaf = obj->get_object<TLeaf>();
      if (!tleaf)
         return nullptr;

      auto ttree = tleaf->GetBranch()->GetTree();
      if (!ttree)
         return nullptr;

      std::string expr = std::string(tleaf->GetName()) + ">>htemp_tree_draw";

      ttree->Draw(expr.c_str(),"","goff");

      if (!gDirectory)
         return nullptr;

      auto htemp = dynamic_cast<TH1*>(gDirectory->FindObject("htemp_tree_draw"));

      if (!htemp)
         return nullptr;

      htemp->SetDirectory(nullptr);
      htemp->SetName(tleaf->GetName());

      return htemp;
   }

};
