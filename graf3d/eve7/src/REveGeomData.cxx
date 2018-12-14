// @(#)root/eve7:$Id$
// Author: Sergey Linev, 14.12.2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include <ROOT/REveGeomData.hxx>

#include "TGeoNode.h"
#include "TGeoVolume.h"
#include "TGeoBBox.h"
#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TObjArray.h"

/////////////////////////////////////////////////////////////////////
/// Add node and all its childs to the flat list, exclude duplication

void ROOT::Experimental::REveGeomDescription::ScanNode(TGeoNode *node)
{
   if (!node)
      return;

   if (std::find(fNodes.begin(), fNodes.end(), node) != fNodes.end())
      return;

   fNodes.push_back(node);

   auto chlds = node->GetNodes();
   if (chlds) {
      for (int n = 0; n <= chlds->GetLast(); ++n)
         ScanNode(dynamic_cast<TGeoNode *>(chlds->At(n)));
   }
}

/////////////////////////////////////////////////////////////////////
/// Collect information about geometry hierarchy into flat list
/// like it done JSROOT.GEO.ClonedNodes.prototype.CreateClones

void ROOT::Experimental::REveGeomDescription::Build(TGeoManager *mgr)
{
   fNodes.clear();

   ScanNode(mgr->GetTopNode());

   // vector to remember number
   std::vector<Int_t> numbers;

   fDesc.clear();
   fDesc.reserve(fNodes.size());
   numbers.reserve(fNodes.size());

   // first create nodes vector and assign id as node number
   int cnt = 0;
   for (auto &&node: fNodes) {
      numbers.emplace_back(node->GetNumber());
      node->SetNumber(cnt++);
      fDesc.emplace_back(node->GetNumber());
   }

   // now create list of childs as just vector with ids
   cnt = 0;
   for (auto &&node: fNodes) {
      auto &desc = fDesc[cnt++];

      desc.name = node->GetName();

      auto shape = dynamic_cast<TGeoBBox *>(node->GetVolume()->GetShape());
      if (shape) desc.vol = shape->GetDX()*shape->GetDY()*shape->GetDZ();

      auto chlds = node->GetNodes();

      if (chlds)
         for (int n = 0; n <= chlds->GetLast(); ++n) {
            auto chld = dynamic_cast<TGeoNode *> (chlds->At(n));
            desc.chlds.emplace_back(chld->GetNumber());
         }
   }

   // recover numbers
   cnt = 0;
   for (auto &&node: fNodes)
      node->SetNumber(numbers[cnt++]);

   printf("Build description size %u\n", fDesc.size());
}
