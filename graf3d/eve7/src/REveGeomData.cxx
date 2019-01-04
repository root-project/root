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

#include "TMath.h"
#include "TColor.h"
#include "TROOT.h"
#include "TGeoNode.h"
#include "TGeoVolume.h"
#include "TGeoBBox.h"
#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TGeoMedium.h"
#include "TGeoMaterial.h"
#include "TGeoCompositeShape.h"
#include "TObjArray.h"
#include "TBuffer3D.h"
#include "TBufferJSON.h"

#include <algorithm>

static int dummy_cnt = 0, rotation_cnt = 0, scale_cnt = 0, trans_cnt = 0;

/////////////////////////////////////////////////////////////////////
/// Pack matrix into vector, which can be send to client
/// Following sizes can be used for vector:
///   0 - Identity matrix
///   3 - Translation
///   4 - Scale (last element always 1)
///   9 - Rotation
///  16 - Full size

void ROOT::Experimental::REveGeomDescription::PackMatrix(std::vector<float> &vect, TGeoMatrix *matr)
{
   vect.clear();

   if (!matr || matr->IsIdentity()) {
      dummy_cnt++;
      return;
   }

   auto trans = matr->GetTranslation();
   auto scale = matr->GetScale();
   auto rotate = matr->GetRotationMatrix();

   bool is_translate = matr->IsA() == TGeoTranslation::Class(),
        is_scale = matr->IsA() == TGeoScale::Class(),
        is_rotate = matr->IsA() == TGeoRotation::Class();

   if (!is_translate && !is_scale && !is_rotate) {
      // check if trivial matrix

      auto test = [](double val, double chk) { return (val==chk) || (TMath::Abs(val-chk) < 1e-20); };

      bool no_scale = test(scale[0],1) && test(scale[1],1) && test(scale[2],1);
      bool no_trans = test(trans[0],0) && test(trans[1],0) && test(trans[2],0);
      bool no_rotate = test(rotate[0],1) && test(rotate[1],0) && test(rotate[2],0) &&
                       test(rotate[3],0) && test(rotate[4],1) && test(rotate[5],0) &&
                       test(rotate[6],0) && test(rotate[7],0) && test(rotate[8],1);

      if (no_scale && no_trans && no_rotate) {
         printf("Detect extra dummy\n");
         dummy_cnt++;
         return;
      }

      if (no_scale && no_trans && !no_rotate) {
         is_rotate = true;
      } else if (no_scale && !no_trans && no_rotate) {
         is_translate = true;
      } else if (!no_scale && no_trans && no_rotate) {
         is_scale = true;
      }
   }

   if (is_translate) {
      vect.resize(3);
      vect[0] = trans[0];
      vect[1] = trans[1];
      vect[2] = trans[2];
      trans_cnt++;
      return;
   }

   if (is_scale) {
      vect.resize(4);
      vect[0] = scale[0];
      vect[1] = scale[1];
      vect[2] = scale[2];
      vect[3] = 1;
      scale_cnt++;
      return;
   }

   if (is_rotate) {
      vect.resize(9);
      for (int n=0;n<9;++n)
         vect[n] = rotate[n];
      rotation_cnt++;
      return;
   }

   vect.resize(16);
   vect[0] = rotate[0]; vect[4] = rotate[1]; vect[8]  = rotate[2]; vect[12] = trans[0];
   vect[1] = rotate[3]; vect[5] = rotate[4]; vect[9]  = rotate[5]; vect[13] = trans[1];
   vect[2] = rotate[6]; vect[6] = rotate[7]; vect[10] = rotate[8]; vect[14] = trans[2];
   vect[3] = 0;         vect[7] = 0;         vect[11] = 0;         vect[15] = 1;
}


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

   // by top node visibility always enabled and harm logic
   // later visibility can be controlled by other means
   mgr->GetTopNode()->GetVolume()->SetVisibility(kFALSE);

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

      PackMatrix(desc.matr, node->GetMatrix());

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

   cnt = 0;
   for (auto &elem: sortarr) {
      fSortMap.emplace_back(elem->id);
      elem->sortid = cnt++; // keep place in sorted array to correctly apply cut
   }

   printf("Build description size %d dummymatrix %d rotation %d scale %d translation %d\n", (int) fDesc.size(), dummy_cnt, rotation_cnt, scale_cnt, trans_cnt);

   MarkVisible(); // set visibility flags
}

/////////////////////////////////////////////////////////////////////
/// Select top visible volume, other volumes will not be shown

void ROOT::Experimental::REveGeomDescription::SelectVolume(TGeoVolume *vol)
{
   fTopDrawNode = 0;
   if (!vol) return;

   for (auto &desc: fDesc)
      if (fNodes[desc.id]->GetVolume() == vol) {
         fTopDrawNode = desc.id;
         break;
      }
}

/////////////////////////////////////////////////////////////////////
/// Select top visible node, other nodes will not be shown

void ROOT::Experimental::REveGeomDescription::SelectNode(TGeoNode *node)
{
   fTopDrawNode = 0;
   if (!node) return;

   for (auto &desc: fDesc)
      if (fNodes[desc.id] == node) {
         fTopDrawNode = desc.id;
         break;
      }
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
   int seqid{0}, inside_visisble_branch{0};

   using ScanFunc_t = std::function<int(int, int)>;

   ScanFunc_t scan_func = [&, this](int nodeid, int lvl) {
      if (nodeid == fTopDrawNode)
         inside_visisble_branch++;

      auto &desc = fDesc[nodeid];
      int res = 0;
      if (desc.vis && (lvl >= 0) && (inside_visisble_branch > 0))
         if (func(desc, stack))
            res++;

      seqid++; // count sequence id of current position in scan, will be used later for merging drawing lists

      // limit depth to which it scans
      if (lvl > desc.visdepth)
         lvl = desc.visdepth;

      if ((desc.chlds.size() > 0) && (desc.numvischld > 0)) {
         auto pos = stack.size();
         int numvischld = 0, previd = seqid;
         stack.push_back(0);
         for (unsigned k = 0; k < desc.chlds.size(); ++k) {
            stack[pos] = k; // stack provides index in list of chdils
            numvischld += scan_func(desc.chlds[k], lvl - 1);
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

      if (nodeid == fTopDrawNode)
         inside_visisble_branch--;

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

bool ROOT::Experimental::REveGeomDescription::CollectVisibles(int maxnumfaces)
{
   std::vector<int> viscnt(fDesc.size(), 0);

   // first count how many times each individual node appears
   ScanVisible([&viscnt](REveGeomNode &node, std::vector<int> &) {
      viscnt[node.id]++;
      return true;
   });

   int totalnumfaces{0}, totalnumnodes{0};

   fDrawIdCut = 0;

   // build all shapes in volume decreasing order
   for (auto &sid: fSortMap) {
      fDrawIdCut++; //
      auto &desc = fDesc[sid];
      if ((viscnt[sid] <= 0) && (desc.vol <= 0)) continue;

      auto shape = fNodes[sid]->GetVolume()->GetShape();
      if (!shape) continue;

      // now we need to create TEveGeoPolyShape, which can provide all rendering data

      auto &shape_descr = FindShapeDescr(shape);

      if (!shape_descr.fRenderData) {
         TGeoCompositeShape *comp = dynamic_cast<TGeoCompositeShape *>(shape);

         auto poly = std::make_unique<REveGeoPolyShape>();

         if (comp) {
            poly->BuildFromComposite(comp, GetNSegments());
         } else {
            poly->BuildFromShape(shape, GetNSegments());
         }

         shape_descr.fRenderData = std::make_unique<REveRenderData>();

         poly->FillRenderData(*shape_descr.fRenderData);

         shape_descr.nfaces = poly->GetNumFaces();
      }

      // check how many faces are created
      totalnumfaces += shape_descr.nfaces * viscnt[sid];
      if (totalnumfaces > maxnumfaces) break;

      // also avoid too many nodes
      totalnumnodes += viscnt[sid];
      if (totalnumnodes > maxnumfaces/12) break;
   }

   // finally we should create data for streaming to the client
   // it includes list of visible nodes and rawdata

   for (auto &s: fShapes)
      s.render_offest = -1;

   std::vector<REveGeomVisisble> visibles;
   std::vector<REveRenderData*> render_data; // data which should be send as binary
   int render_offset{0}; /// current offset

   ScanVisible([&, this](REveGeomNode &node, std::vector<int> &stack) {
      if (node.sortid < fDrawIdCut) {
         visibles.emplace_back(node.id, stack);

         auto &item = visibles.back();

         auto volume = fNodes[node.id]->GetVolume();

         TColor *col{nullptr};

         if ((volume->GetFillColor() > 1) && (volume->GetLineColor() == 1))
            col = gROOT->GetColor(volume->GetFillColor());
         else if (volume->GetLineColor() >= 0)
            col = gROOT->GetColor(volume->GetLineColor());


         if (volume->GetMedium() && (volume->GetMedium() != TGeoVolume::DummyMedium()) && volume->GetMedium()->GetMaterial()) {
            auto material = volume->GetMedium()->GetMaterial();

            auto fillstyle = material->GetFillStyle();
            if ((fillstyle>=3000) && (fillstyle<=3100)) item.opacity = (3100 - fillstyle) / 100.;
            if (!col) col = gROOT->GetColor(material->GetFillColor());
         }

         if (col) {
            item.color = std::to_string((int)(col->GetRed()*255)) + "," +
                         std::to_string((int)(col->GetGreen()*255)) + "," +
                         std::to_string((int)(col->GetBlue()*255));
            if (item.opacity == 1.)
              item.opacity = col->GetAlpha();
         } else {
            item.color = "200,200,200";
         }

         auto &sd = FindShapeDescr(volume->GetShape());
         auto *rd = sd.fRenderData.get();

         if (sd.render_offest < 0) {
            sd.render_offest = render_offset;
            render_offset += rd->GetBinarySize();
            render_data.emplace_back(rd);
         }

         item.rnr_offset = sd.render_offest;

         item.rnr_func = rd->GetRnrFunc();
         item.vert_size = rd->SizeV();
         item.norm_size = rd->SizeN();
         item.index_size = rd->SizeI();
         // item.trans_size = rd->SizeT();
      }
      return true;
   });

   // finally, create binary data with all produced shapes

   auto res = TBufferJSON::ToJSON(&visibles, 103);
   fDrawJson = "DRAW:";
   fDrawJson.append(res.Data());

   fDrawBinary.resize(render_offset);
   int off{0};

   for (auto rd : render_data) {
      auto sz = rd->Write( &fDrawBinary[off], fDrawBinary.size() - off );
      off += sz;
   }
   assert(render_offset == off);

   return true;
}

