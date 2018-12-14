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
   for (auto &node: fNodes) {

      fDesc.emplace_back(node->GetNumber()-offset);
      auto &desc = fDesc[cnt++];

      desc.name = node->GetName();

      auto shape = dynamic_cast<TGeoBBox *>(node->GetVolume()->GetShape());
      if (shape) {
         desc.vol = shape->GetDX()*shape->GetDY()*shape->GetDZ();
         desc.nfaces = 12; // TODO: get better value for each shape - excluding composite
      }

      auto chlds = node->GetNodes();

      if (chlds)
         for (int n = 0; n <= chlds->GetLast(); ++n) {
            auto chld = dynamic_cast<TGeoNode *> (chlds->At(n));
            desc.chlds.emplace_back(chld->GetNumber()-offset);
         }
   }

   // recover numbers
   cnt = 0;
   for (auto &node: fNodes)
      node->SetNumber(numbers[cnt++]);

   printf("Build description size %d\n", (int) fDesc.size());

   MarkVisible(); // set visibility flags
}

/////////////////////////////////////////////////////////////////////
/// Set visibility flag for each nodes

int ROOT::Experimental::REveGeomDescription::MarkVisible(bool on_screen)
{
   int res = 0, cnt = 0;
   for (auto &node: fNodes) {
      auto &desc = fDesc[cnt++];

      desc.vis = false;
      desc.visdepth = 9999999;
      desc.numvischld = 1;
      desc.idshift = 0;

      if (on_screen) {
         desc.vis = node->IsOnScreen();
      } else {
         auto vol = node->GetVolume();

         desc.vis = vol->IsVisible() && !vol->TestAttBit(TGeoAtt::kVisNone);
         if (!vol->IsVisDaughters())
            desc.visdepth = vol->TestAttBit(TGeoAtt::kVisOneLevel) ? 1 : 0;
      }

      if ((desc.vol <= 0) || (desc.nfaces <= 0)) desc.vis = false;

      if (desc.vis) res++;
   }

   return res;
}

/////////////////////////////////////////////////////////////////////
/// Iterate over all visible nodes and call function

void ROOT::Experimental::REveGeomDescription::ScanVisible(REveGeomScanFunc_t func)
{
   std::vector<int> stack;
   stack.reserve(200);
   int seqid{0};

   using ScanFunc_t = std::function<int(int, int)>;

   ScanFunc_t scan_func = [&, this](int nodeid, int lvl) {
      auto &desc = fDesc[nodeid];
      int res = 0;
      if (desc.vis && (lvl>=0))
        if (func(desc, stack)) res++;

      seqid++; // count sequence id of current position in scan

      // limit depth to which it scans
      if (lvl > desc.visdepth) lvl = desc.visdepth;

      if ((desc.chlds.size() > 0) && (desc.numvischld>0)) {
         auto pos = stack.size();
         int numvischld = 0, previd = seqid;
         stack.push_back(0);
         for (unsigned k=0; k<desc.chlds.size(); ++k) {
            stack[pos] = k;
            numvischld += scan_func(desc.chlds[k], lvl-1);
         }
         stack.pop_back();

         // if no child is visible, skip it again and correctly calculate seqid
         if (numvischld == 0) {
            desc.numvischld = 0;
            desc.idshift = seqid - previd;
         }

         res += numvischld;
      } else {
         seqid += desc.idshift;
      }

      return res;

   };

   scan_func(0, 999999);
}

