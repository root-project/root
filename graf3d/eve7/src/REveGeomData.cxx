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

void ROOT::Experimental::REveGeomDescription::ScanNode(TGeoNode *node, std::vector<int> &numbers, int offset)
{
   if (!node)
      return;

   // artificial offset, used as identifier
   if (node->GetNumber() >= offset) return;

   numbers.emplace_back(node->GetNumber());

   node->SetNumber(offset + fNodes.size()); // use id with shift 1e9
   fNodes.emplace_back(node);

   auto chlds = node->GetNodes();
   if (chlds) {
      for (int n = 0; n <= chlds->GetLast(); ++n)
         ScanNode(dynamic_cast<TGeoNode *>(chlds->At(n)), numbers, offset);
   }
}

/////////////////////////////////////////////////////////////////////
/// Collect information about geometry hierarchy into flat list
/// like it done JSROOT.GEO.ClonedNodes.prototype.CreateClones

void ROOT::Experimental::REveGeomDescription::Build(TGeoManager *mgr)
{
   fNodes.clear();

   // vector to remember numbers
   std::vector<int> numbers;
   int offset = 1000000000;

   ScanNode(mgr->GetTopNode(), numbers, offset);

   fDesc.clear();
   fDesc.reserve(fNodes.size());
   numbers.reserve(fNodes.size());

   // create vector of desc and childs
   int cnt = 0;
   for (auto &&node: fNodes) {

      fDesc.emplace_back(node->GetNumber()-offset);
      auto &desc = fDesc[cnt++];

      desc.name = node->GetName();

      auto shape = dynamic_cast<TGeoBBox *>(node->GetVolume()->GetShape());
      if (shape) desc.vol = shape->GetDX()*shape->GetDY()*shape->GetDZ();

      auto chlds = node->GetNodes();

      if (chlds)
         for (int n = 0; n <= chlds->GetLast(); ++n) {
            auto chld = dynamic_cast<TGeoNode *> (chlds->At(n));
            desc.chlds.emplace_back(chld->GetNumber()-offset);
         }
   }

   // recover numbers
   cnt = 0;
   for (auto &&node: fNodes)
      node->SetNumber(numbers[cnt++]);

   printf("Build description size %d\n", (int) fDesc.size());
}
