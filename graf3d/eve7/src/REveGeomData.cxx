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
#include <ROOT/TLogger.hxx>

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

      if (no_scale && no_trans && no_rotate)
         return;

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
      return;
   }

   if (is_scale) {
      vect.resize(4);
      vect[0] = scale[0];
      vect[1] = scale[1];
      vect[2] = scale[2];
      vect[3] = 1;
      return;
   }

   if (is_rotate) {
      vect.resize(9);
      for (int n=0;n<9;++n)
         vect[n] = rotate[n];
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
   fDesc.clear();
   fNodes.clear();
   fSortMap.clear();
   ClearRawData();
   fDrawIdCut = 0;

   if (!mgr) return;

   // vector to remember numbers
   std::vector<int> numbers;
   int offset = 1000000000;

   // by top node visibility always enabled and harm logic
   // later visibility can be controlled by other means
   mgr->GetTopNode()->GetVolume()->SetVisibility(kFALSE);

   // build flat list of all nodes
   ScanNode(mgr->GetTopNode(), numbers, offset);

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

      // ignore shapes where childs are exists
      // FIXME: seems to be, in some situations shape has to be drawn
      if ((desc.chlds.size() > 0) && shape && (shape->IsA() == TGeoBBox::Class())) {
         desc.vol = 0;
         desc.nfaces = 0;
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

      desc.vis = REveGeomNode::vis_off;
      desc.numvischld = 1;
      desc.idshift = 0;

      if (on_screen) {
         if (node->IsOnScreen())
            desc.vis = REveGeomNode::vis_this;
      } else {
         auto vol = node->GetVolume();

         if (desc.chlds.size() > 0) {
            if (vol->IsVisDaughters()) {
               desc.vis = REveGeomNode::vis_chlds;
            } else if (vol->TestAttBit(TGeoAtt::kVisOneLevel)) {
               desc.vis = REveGeomNode::vis_lvl1;
            }
         } else {
            if (vol->IsVisible() && !vol->TestAttBit(TGeoAtt::kVisNone) && !node->GetFinder())
               desc.vis = REveGeomNode::vis_this;
         }
      }

      if (desc.IsVisible() && desc.CanDisplay()) res++;
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
      if (desc.IsVisible() && desc.CanDisplay() && (lvl >= 0) && (inside_visisble_branch > 0))
         if (func(desc, stack))
            res++;

      seqid++; // count sequence id of current position in scan, will be used later for merging drawing lists

      // limit depth to which it scans
      if (lvl > desc.GetVisDepth())
         lvl = desc.GetVisDepth();

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

ROOT::Experimental::REveGeomDescription::ShapeDescr &ROOT::Experimental::REveGeomDescription::FindShapeDescr(TGeoShape *shape)
{
   for (auto &&descr: fShapes)
      if (descr.fShape == shape)
         return descr;

   fShapes.emplace_back(shape);
   auto &elem = fShapes.back();
   elem.id = fShapes.size() - 1;
   return elem;
}


/////////////////////////////////////////////////////////////////////
/// Find description object and create render information

ROOT::Experimental::REveGeomDescription::ShapeDescr &
ROOT::Experimental::REveGeomDescription::MakeShapeDescr(TGeoShape *shape, bool acc_rndr)
{
   auto &elem = FindShapeDescr(shape);

   if (!elem.fRenderData) {
      TGeoCompositeShape *comp = dynamic_cast<TGeoCompositeShape *>(shape);

      auto poly = std::make_unique<REveGeoPolyShape>();

      if (comp) {
         poly->BuildFromComposite(comp, GetNSegments());
      } else {
         poly->BuildFromShape(shape, GetNSegments());
      }

      elem.fRenderData = std::make_unique<REveRenderData>();

      poly->FillRenderData(*elem.fRenderData);

      elem.nfaces = poly->GetNumFaces();
   }

   if (acc_rndr && (elem.nfaces > 0)) {
      auto &rd = elem.fRenderData;
      auto &ri = elem.fRenderInfo;

      if (ri.rnr_offset < 0) {
         ri.rnr_offset = fRndrOffest;
         fRndrOffest += rd->GetBinarySize();
         fRndrShapes.emplace_back(rd.get());

         ri.rnr_func = rd->GetRnrFunc();
         ri.vert_size = rd->SizeV();
         ri.norm_size = rd->SizeN();
         ri.index_size = rd->SizeI();
      }
   }

   return elem;
}

/////////////////////////////////////////////////////////////////////
/// Copy material properties

void ROOT::Experimental::REveGeomDescription::CopyMaterialProperties(TGeoVolume *volume, REveGeomVisisble &item)
{
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
}

/////////////////////////////////////////////////////////////////////
/// Reset shape info, which used to pack binary data

void ROOT::Experimental::REveGeomDescription::ResetRndrInfos()
{
   for (auto &s: fShapes)
      s.fRenderInfo.rnr_offset = -1;

   fRndrShapes.clear();

   fRndrOffest = 0;
}

/////////////////////////////////////////////////////////////////////
/// Fill binary buffer

void ROOT::Experimental::REveGeomDescription::BuildRndrBinary(std::vector<char> &buf)
{
   buf.resize(fRndrOffest);
   int off{0};

   for (auto rd : fRndrShapes) {
      auto sz = rd->Write( &buf[off], buf.size() - off );
      off += sz;
   }
   assert(fRndrOffest == off);

   fRndrShapes.clear();
   fRndrOffest = 0;
}

/////////////////////////////////////////////////////////////////////
/// Collect all information required to draw geometry on the client
/// This includes list of each visible nodes, meshes and matrixes

bool ROOT::Experimental::REveGeomDescription::CollectVisibles()
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

      if ((viscnt[sid] <= 0) || (desc.vol <= 0)) continue;

      auto shape = fNodes[sid]->GetVolume()->GetShape();
      if (!shape) continue;

      // now we need to create TEveGeoPolyShape, which can provide all rendering data
      auto &shape_descr = MakeShapeDescr(shape);

      // should not happen, but just in case
      if (shape_descr.nfaces <= 0) {
         R__ERROR_HERE("webeve") << "No faces for the shape " << shape->GetName() << " class " << shape->ClassName();
         continue;
      }

      // check how many faces are created
      totalnumfaces += shape_descr.nfaces * viscnt[sid];
      if ((GetMaxVisFaces() > 0) && (totalnumfaces > GetMaxVisFaces())) break;

      // also avoid too many nodes
      totalnumnodes += viscnt[sid];
      if ((GetMaxVisNodes() > 0) && (totalnumnodes > GetMaxVisNodes())) break;
   }

   // finally we should create data for streaming to the client
   // it includes list of visible nodes and rawdata

   std::vector<REveGeomVisisble> visibles;
   ResetRndrInfos();

   ScanVisible([&, this](REveGeomNode &node, std::vector<int> &stack) {
      if (node.sortid < fDrawIdCut) {
         visibles.emplace_back(node.id, stack);

         auto &item = visibles.back();
         auto volume = fNodes[node.id]->GetVolume();
         CopyMaterialProperties(volume, item);

         auto &sd = MakeShapeDescr(volume->GetShape(), true);

         item.ri = sd.rndr_info();
      }
      return true;
   });

   // finally, create binary data with all produced shapes

   fDrawJson = "GDRAW:";
   fDrawJson.append(TBufferJSON::ToJSON(&visibles, 103).Data());

   BuildRndrBinary(fDrawBinary);

   return true;
}

/////////////////////////////////////////////////////////////////////
/// Clear raw data. Will be rebuild when next connection will be established

void ROOT::Experimental::REveGeomDescription::ClearRawData()
{
   fDrawJson.clear();
   fDrawBinary.clear();
}

/////////////////////////////////////////////////////////////////////
/// return true when node used in main geometry drawing

bool ROOT::Experimental::REveGeomDescription::IsPrincipalNode(int nodeid)
{
   if ((nodeid < 0) || (nodeid >= (int)fDesc.size()))
      return false;

   auto &desc = fDesc[nodeid];

   return desc.sortid < fDrawIdCut;
}

/////////////////////////////////////////////////////////////////////
/// Search visible nodes for provided name
/// If number of found elements less than 100, create description and shapes for them
/// Returns number of match elements

int ROOT::Experimental::REveGeomDescription::SearchVisibles(const std::string &find, std::string &json, std::vector<char> &binary)
{
   std::vector<int> viscnt(fDesc.size(), 0);

   int nmatches{0};

   auto match_func = [&find](REveGeomNode &node) {
      return (node.vol > 0) && (node.name.compare(0, find.length(), find) == 0);
   };

   // first count how many times each individual node appears
   ScanVisible([&viscnt,&match_func,&nmatches](REveGeomNode &node, std::vector<int> &) {
      if (match_func(node)) {
         nmatches++;
         viscnt[node.id]++;
      };
      return true;
   });


   json.clear();
   binary.clear();

   // do not send too much data, limit could be made configurable later
   if (nmatches==0) {
      json = "FOUND:NO";
      return nmatches;
   }

   if (nmatches > 10 * GetMaxVisNodes()) {
      json = "FOUND:TOOMANY:" + std::to_string(nmatches);
      return nmatches;
   }

   // now build all necessary shapes and check number of faces - not too many

   int totalnumfaces{0}, totalnumnodes{0}, scnt{0};
   bool send_rawdata{true};

   // build all shapes in volume decreasing order
   for (auto &sid: fSortMap) {
      if (scnt++ < fDrawIdCut) continue; // no need to send most significant shapes

      if (viscnt[sid] == 0) continue; // this node is not used at all

      auto &desc = fDesc[sid];
      if ((viscnt[sid] <= 0) && (desc.vol <= 0)) continue;

      auto shape = fNodes[sid]->GetVolume()->GetShape();
      if (!shape) continue;

      // create shape raw data
      auto &shape_descr = MakeShapeDescr(shape);

      // should not happen, but just in case
      if (shape_descr.nfaces <= 0) {
         R__ERROR_HERE("webeve") << "No faces for the shape " << shape->GetName() << " class " << shape->ClassName();
         continue;
      }

      // check how many faces are created
      totalnumfaces += shape_descr.nfaces * viscnt[sid];
      if ((GetMaxVisFaces() > 0) && (totalnumfaces > GetMaxVisFaces())) { send_rawdata = false; break; }

      // also avoid too many nodes
      totalnumnodes += viscnt[sid];
      if ((GetMaxVisNodes() > 0) && (totalnumnodes > GetMaxVisNodes()))  { send_rawdata = false; break; }
   }

   // only for debug purposes - remove later
   // send_rawdata = false;

   // finally we should create data for streaming to the client
   // it includes list of visible nodes and rawdata (if there is enough space)

   ResetRndrInfos();
   std::vector<REveGeomVisisble> visibles;

   ScanVisible([&, this](REveGeomNode &node, std::vector<int> &stack) {
      // select only nodes which should match
      if (!match_func(node))
         return true;

      visibles.emplace_back(node.id, stack);

      // no need to transfer shape if it provided with main drawing list
      // also no binary will be transported when too many matches are there
      if (!send_rawdata || (node.sortid < fDrawIdCut)) {
         // do not include render data
         return true;
      }

      auto &item = visibles.back();
      auto volume = fNodes[node.id]->GetVolume();

      CopyMaterialProperties(volume, item);

      auto &sd = MakeShapeDescr(volume->GetShape(), true);

      item.ri = sd.rndr_info();
      return true;
   });

   json = "FOUND:";
   json.append(TBufferJSON::ToJSON(&visibles, 103).Data());

   BuildRndrBinary(binary);

   return nmatches;
}

/////////////////////////////////////////////////////////////////////////////////
/// Returns nodeid for given stack array, returns -1 in case of failure

int ROOT::Experimental::REveGeomDescription::FindNodeId(const std::vector<int> &stack)
{
   int nodeid{0};

   for (auto &chindx: stack) {
      auto &node = fDesc[nodeid];
      if (chindx >= (int) node.chlds.size()) return -1;
      nodeid = node.chlds[chindx];
   }

   return nodeid;
}

/////////////////////////////////////////////////////////////////////////////////
/// Return string with only part of nodes description which were modified
/// Checks also volume

std::string ROOT::Experimental::REveGeomDescription::ProduceModifyReply(int nodeid)
{
   std::vector<REveGeomNode *> nodes;
   auto vol = fNodes[nodeid]->GetVolume();

   // we take not only single node, but all there same volume is referenced
   // nodes.push_back(&fDesc[nodeid]);

   int id{0};
   for (auto &desc : fDesc)
      if (fNodes[id++]->GetVolume() == vol)
         nodes.emplace_back(&desc);

   std::string res = "MODIF:";
   res.append(TBufferJSON::ToJSON(&nodes,103).Data());
   return res;
}


/////////////////////////////////////////////////////////////////////////////////
/// Produce shape rendering data for given stack
/// All nodes, which are referencing same shape will be transferred

bool ROOT::Experimental::REveGeomDescription::ProduceDrawingFor(int nodeid, std::string &json, std::vector<char> &binary, bool check_volume)
{
   // only this shape is interesting

   TGeoVolume *vol = (nodeid < 0) ? nullptr : fNodes[nodeid]->GetVolume();

   if (!vol || !vol->GetShape()) {
      json.append("NO");
      return true;
   }

   std::vector<REveGeomVisisble> visibles;

   ScanVisible([&, this](REveGeomNode &node, std::vector<int> &stack) {
      // select only nodes which reference same shape

      if (check_volume) {
         if (fNodes[node.id]->GetVolume() != vol) return true;
      } else {
         if (node.id != nodeid) return true;
      }

      visibles.emplace_back(node.id, stack);
      auto &item = visibles.back();
      CopyMaterialProperties(vol, item);
      return true;
   });

   // no any visible nodes were done
   if (visibles.size()==0) {
      json.append("NO");
      return true;
   }

   ResetRndrInfos();

   auto &sd = MakeShapeDescr(vol->GetShape(), true);

   // assign shape data
   for (auto &item : visibles)
      item.ri = sd.rndr_info();

   json.append(TBufferJSON::ToJSON(&visibles, 103).Data());

   BuildRndrBinary(binary);

   return true;
}

/////////////////////////////////////////////////////////////////////////////////
/// Change visibility for specified element
/// Returns true if changes was performed

bool ROOT::Experimental::REveGeomDescription::ChangeNodeVisibility(int nodeid, bool selected)
{
   auto &dnode = fDesc[nodeid];

   // nothing changed
   if ((dnode.vis && selected) || (!dnode.vis && !selected))
      return false;

   auto vol = fNodes[nodeid]->GetVolume();

   if (dnode.chlds.size() > 0) {
      vol->SetVisDaughters(selected);
      vol->SetAttBit(TGeoAtt::kVisOneLevel, kFALSE); // disable one level when toggling visibility
      dnode.vis = selected ? REveGeomNode::vis_chlds : REveGeomNode::vis_off;
   } else {
      vol->SetVisibility(selected);
      dnode.vis = selected ? REveGeomNode::vis_this : REveGeomNode::vis_off;
   }

   int id{0};
   for (auto &desc: fDesc)
      if (fNodes[id++]->GetVolume() == vol)
         desc.vis = dnode.vis;

   ClearRawData(); // after change raw data is no longer valid

   return true;
}
