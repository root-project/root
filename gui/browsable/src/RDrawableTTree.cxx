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

#include "TVirtualPad.h"

#include <ROOT/RCanvas.hxx>
#include <ROOT/RObjectDrawable.hxx>


using namespace ROOT::Experimental;

/** Provider for drawing of ROOT6 classes */

class RV6DrawProviderTTree : public Browsable::RProvider {
public:
   RV6DrawProviderTTree()
   {
      RegisterDraw6(TLeaf::Class(), [](TVirtualPad *pad, std::unique_ptr<Browsable::RHolder> &obj, const std::string &opt) -> bool {

         // try take object without ownership
         auto tleaf = obj->get_object<TLeaf>();
         if (!tleaf)
            return false;

         auto ttree = tleaf->GetBranch()->GetTree();
         if (!ttree)
            return false;


         std::string expr = std::string(tleaf->GetName()) + ">>htemp_tree_draw";

         ttree->Draw(expr.c_str(),"","goff");

         if (!gDirectory)
            return false;

         auto htemp = dynamic_cast<TH1*>(gDirectory->FindObject("htemp_tree_draw"));

         if (!htemp)
            return false;

         htemp->SetDirectory(nullptr);
         htemp->SetName(tleaf->GetName());

         pad->GetListOfPrimitives()->Clear();

         pad->GetListOfPrimitives()->Add(htemp, opt.c_str());

         return true;
      });

   }

} newRV6DrawProviderTTree;


/** Provider for drawing of ROOT7 classes */

class RV7DrawProviderTTree : public Browsable::RProvider {
public:
   RV7DrawProviderTTree()
   {
      RegisterDraw7(TLeaf::Class(), [](std::shared_ptr<RPadBase> &subpad, std::unique_ptr<Browsable::RHolder> &obj, const std::string &opt) -> bool {

         // try take object without ownership
         auto tleaf = obj->get_object<TLeaf>();
         if (!tleaf)
            return false;

         auto ttree = tleaf->GetBranch()->GetTree();
         if (!ttree)
            return false;


         std::string expr = std::string(tleaf->GetName()) + ">>htemp_tree_draw";

         ttree->Draw(expr.c_str(),"","goff");

         if (!gDirectory)
            return false;

         auto htemp = dynamic_cast<TH1*>(gDirectory->FindObject("htemp_tree_draw"));

         if (!htemp)
            return false;

         htemp->SetDirectory(nullptr);
         htemp->SetName(tleaf->GetName());

         if (subpad->NumPrimitives() > 0) {
            subpad->Wipe();
            subpad->GetCanvas()->Modified();
            subpad->GetCanvas()->Update(true);
         }

         std::shared_ptr<TH1> shared;
         shared.reset(htemp);

         subpad->Draw<RObjectDrawable>(shared, opt);


         return true;
      });

   }

} newRV7DrawProviderTTree;

