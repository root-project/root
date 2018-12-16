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

#include <ROOT/REveGeoPolyShape.hxx>
#include <ROOT/REveUtil.hxx>

#include "TGeoNode.h"
#include "TGeoVolume.h"
#include "TGeoBBox.h"
#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TGeoCompositeShape.h"
#include "TObjArray.h"
#include "TBuffer3D.h"

#include <algorithm>

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

   // build flat list of all nodes
   ScanNode(mgr->GetTopNode(), numbers, offset);

   fDesc.clear();
   fSortMap.clear();
   fDesc.reserve(fNodes.size());
   numbers.reserve(fNodes.size());
   fSortMap.reserve(fNodes.size());

   // array for sorting
   std::vector<REveGeomNode *> sortarr;
   sortarr.reserve(fNodes.size());

   // create vector of desc and childs
   int cnt = 0;
   for (auto &node: fNodes) {

      fDesc.emplace_back(node->GetNumber()-offset);
      auto &desc = fDesc[cnt++];

      sortarr.emplace_back(&desc);

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

   // sort in volume descent order
   std::sort(sortarr.begin(), sortarr.end(), [](REveGeomNode *a, REveGeomNode * b) { return a->vol > b->vol; });

   for (auto &elem: sortarr)
      fSortMap.emplace_back(elem->id);

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

/////////////////////////////////////////////////////////////////////
/// Find description object for requested shape
/// If not exists - will be created

ROOT::Experimental::REveGeomDescription::ShapeDescr &ROOT::Experimental::REveGeomDescription::FindShapeDescr(TGeoShape *s)
{
   for (auto &descr: fShapes)
      if (descr.fShape == s) return descr;

   fShapes.emplace_back(s);

   auto &elem = fShapes.back();
   elem.id = fShapes.size() - 1;
   return elem;
}

/////////////////////////////////////////////////////////////////////
/// Collect all information required to draw geometry on the client
/// This includes list of each visible nodes, meshes and matrixes

void ROOT::Experimental::REveGeomDescription::CollectVisibles(int maxnumfaces)
{
   std::vector<int> viscnt(fDesc.size(), 0);

   // first count how many times each individual node appears
   ScanVisible([&viscnt](REveGeomNode& node, std::vector<int>&) {
      viscnt[node.id]++;
      return true;
   });

   // now one can start build all shapes in volume decreasing order
   for (auto &sid: fSortMap) {
      auto &desc = fDesc[sid];
      if ((viscnt[sid] <= 0) && (desc.vol <= 0)) continue;

      auto shape = fNodes[sid]->GetVolume()->GetShape();
      if (!shape) continue;

      // now we need to create TEveGeoPolyShape, which can provide all rendering data

      auto &shape_descr = FindShapeDescr(shape);

      if (!shape_descr.fRenderData) {
         TGeoCompositeShape *comp = dynamic_cast<TGeoCompositeShape *>(shape);

         std::unique_ptr<REveGeoPolyShape> poly;

         if (comp) {
            poly = std::make_unique<REveGeoPolyShape>(comp, 20);
         } else {
            poly = std::make_unique<REveGeoPolyShape>();
            REveGeoManagerHolder gmgr(gGeoManager, 20);
            std::unique_ptr<TBuffer3D> b3d(shape->MakeBuffer3D());
            poly->SetFromBuff3D(*b3d.get());
         }

         shape_descr.fRenderData = std::make_unique<REveRenderData>();

         poly->FillRenderData(*shape_descr.fRenderData);
      }
   }
}

