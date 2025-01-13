// Author: Sergey Linev, 14.12.2018

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RGeomData.hxx>

#include <ROOT/RBrowserRequest.hxx>
#include <ROOT/RBrowserReply.hxx>
#include <ROOT/RLogger.hxx>
#include "CsgOps.h"

#include "TMath.h"
#include "TColor.h"
#include "TROOT.h"
#include "TGeoNode.h"
#include "TGeoVolume.h"
#include "TGeoBBox.h"
#include "TGeoSphere.h"
#include "TGeoCone.h"
#include "TGeoTube.h"
#include "TGeoEltu.h"
#include "TGeoTorus.h"
#include "TGeoPcon.h"
#include "TGeoPgon.h"
#include "TGeoXtru.h"
#include "TGeoParaboloid.h"
#include "TGeoHype.h"
#include "TGeoTessellated.h"
#include "TGeoScaledShape.h"
#include "TGeoCompositeShape.h"
#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TGeoMedium.h"
#include "TGeoMaterial.h"
#include "TGeoBoolNode.h"
#include "TBuffer3D.h"
#include "TBufferJSON.h"
#include "TRegexp.h"

#include <algorithm>
#include <array>

ROOT::RLogChannel &ROOT::RGeomLog()
{
   static ROOT::RLogChannel sLog("ROOT.Geom");
   return sLog;
}


namespace ROOT {

/** Iterator of hierarchical geometry structures */

class RGeomBrowserIter {

   RGeomDescription &fDesc;
   int fParentId{-1};
   unsigned fChild{0};
   int fNodeId{0};

   std::vector<int> fStackParents;
   std::vector<int> fStackChilds;

public:
   RGeomBrowserIter(RGeomDescription &desc) : fDesc(desc) {}

   const std::string &GetName() const { return fDesc.fDesc[fNodeId].name; }

   const std::string &GetColor() const { return fDesc.fDesc[fNodeId].color; }

   const std::string &GetMaterial() const { return fDesc.fDesc[fNodeId].material; }

   int GetVisible() const { return fDesc.fDesc[fNodeId].vis; }

   bool IsValid() const { return fNodeId >= 0; }

   int GetNodeId() const { return fNodeId; }

   bool HasChilds() const { return (fNodeId < 0) ? true : !fDesc.fDesc[fNodeId].chlds.empty(); }

   int NumChilds() const { return (fNodeId < 0) ? 1 : fDesc.fDesc[fNodeId].chlds.size(); }

   bool Enter()
   {
      if (fNodeId < 0) {
         Reset();
         fNodeId = 0;
         return true;
      }

      if (fNodeId >= (int)fDesc.fDesc.size())
         return false;

      auto &node = fDesc.fDesc[fNodeId];
      if (node.chlds.empty())
         return false;
      fStackParents.emplace_back(fParentId);
      fStackChilds.emplace_back(fChild);
      fParentId = fNodeId;
      fChild = 0;
      fNodeId = node.chlds[fChild];
      return true;
   }

   bool Leave()
   {
      if (fStackParents.empty()) {
         fNodeId = -1;
         return false;
      }
      fParentId = fStackParents.back();
      fChild = fStackChilds.back();

      fStackParents.pop_back();
      fStackChilds.pop_back();

      if (fParentId < 0) {
         fNodeId = 0;
      } else {
         fNodeId = fDesc.fDesc[fParentId].chlds[fChild];
      }
      return true;
   }

   bool Next()
   {
      // does not have parents
      if ((fNodeId <= 0) || (fParentId < 0)) {
         Reset();
         return false;
      }

      auto &prnt = fDesc.fDesc[fParentId];
      if (++fChild >= prnt.chlds.size()) {
         fNodeId = -1; // not valid node, only Leave can be called
         return false;
      }

      fNodeId = prnt.chlds[fChild];
      return true;
   }

   bool Reset()
   {
      fParentId = -1;
      fNodeId = -1;
      fChild = 0;
      fStackParents.clear();
      fStackChilds.clear();

      return true;
   }

   bool NextNode()
   {
      if (Enter())
         return true;

      if (Next())
         return true;

      while (Leave()) {
         if (Next())
            return true;
      }

      return false;
   }

   /** Navigate to specified path - path specified as string and should start with "/" */
   bool Navigate(const std::string &path)
   {
      size_t pos = path.find('/');
      if (pos != 0)
         return false;

      Reset(); // set to the top of element

      while (++pos < path.length()) {
         auto last = pos;

         pos = path.find('/', last);

         if (pos == std::string::npos)
            pos = path.length();

         std::string folder = path.substr(last, pos - last);

         if (!Enter())
            return false;

         bool find = false;

         do {
            find = (folder.compare(GetName()) == 0);
         } while (!find && Next());

         if (!find)
            return false;
      }

      return true;
   }

   /** Navigate to specified path  */
   bool Navigate(const std::vector<std::string> &path)
   {
      Reset(); // set to the top of element

      for (auto &folder : path) {

         if (!Enter())
            return false;

         bool find = false;

         do {
            find = (folder.compare(GetName()) == 0);
         } while (!find && Next());

         if (!find)
            return false;
      }

      return true;
   }

   /** Navigate to specified volume - find first occurrence */
   bool Navigate(TGeoVolume *vol)
   {
      Reset();

      while (NextNode()) {
         if (vol == fDesc.GetVolume(GetNodeId()))
            return true;
      }

      return false;
   }

   /// Returns array of ids to currently selected node
   std::vector<int> CurrentIds() const
   {
      std::vector<int> res;
      if (IsValid()) {
         for (unsigned n = 1; n < fStackParents.size(); ++n)
            res.emplace_back(fStackParents[n]);
         if (fParentId >= 0)
            res.emplace_back(fParentId);
         res.emplace_back(fNodeId);
      }
      return res;
   }
};

} // namespace ROOT

using namespace ROOT;

using namespace std::string_literals;

namespace {

int compare_stacks(const std::vector<int> &stack1, const std::vector<int> &stack2)
{
   unsigned len1 = stack1.size(), len2 = stack2.size(), len = (len1 < len2) ? len1 : len2, indx = 0;
   while (indx < len) {
      if (stack1[indx] < stack2[indx])
         return -1;
      if (stack1[indx] > stack2[indx])
         return 1;
      ++indx;
   }

   if (len1 < len2)
      return -1;
   if (len1 > len2)
      return 1;

   return 0;
}
} // namespace

/////////////////////////////////////////////////////////////////////
/// Issue signal, which distributed on all handlers - excluding source handler

void RGeomDescription::IssueSignal(const void *handler, const std::string &kind)
{
   std::vector<RGeomSignalFunc_t> funcs;

   {
      TLockGuard lock(fMutex);
      for (auto &pair : fSignals)
         if (!handler || (pair.first != handler))
            funcs.emplace_back(pair.second);
   }

   // invoke signal outside locked mutex to avoid any locking
   for (auto func : funcs)
      func(kind);
}

/////////////////////////////////////////////////////////////////////
/// Add signal handler

void RGeomDescription::AddSignalHandler(const void *handler, RGeomSignalFunc_t func)
{
   TLockGuard lock(fMutex);
   fSignals.emplace_back(handler, func);
}

/////////////////////////////////////////////////////////////////////
/// Remove signal handler

void RGeomDescription::RemoveSignalHandler(const void *handler)
{
   TLockGuard lock(fMutex);

   for (auto iter = fSignals.begin(); iter != fSignals.end(); ++iter)
      if (handler == iter->first) {
         fSignals.erase(iter);
         return;
      }
}

/////////////////////////////////////////////////////////////////////
/// Pack matrix into vector, which can be send to client
/// Following sizes can be used for vector:
///   0 - Identity matrix
///   3 - Translation
///   4 - Scale (last element always 1)
///   9 - Rotation
///  16 - Full size

void RGeomDescription::PackMatrix(std::vector<float> &vect, TGeoMatrix *matr)
{
   vect.clear();

   if (!matr || matr->IsIdentity())
      return;

   auto trans = matr->GetTranslation();
   auto scale = matr->GetScale();
   auto rotate = matr->GetRotationMatrix();

   bool is_translate = matr->IsA() == TGeoTranslation::Class(), is_scale = matr->IsA() == TGeoScale::Class(),
        is_rotate = matr->IsA() == TGeoRotation::Class();

   if (!is_translate && !is_scale && !is_rotate) {
      // check if trivial matrix

      auto test = [](double val, double chk) { return (val == chk) || (TMath::Abs(val - chk) < 1e-20); };

      bool no_scale = test(scale[0], 1) && test(scale[1], 1) && test(scale[2], 1);
      bool no_trans = test(trans[0], 0) && test(trans[1], 0) && test(trans[2], 0);
      bool no_rotate = test(rotate[0], 1) && test(rotate[1], 0) && test(rotate[2], 0) && test(rotate[3], 0) &&
                       test(rotate[4], 1) && test(rotate[5], 0) && test(rotate[6], 0) && test(rotate[7], 0) &&
                       test(rotate[8], 1);

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
      for (int n = 0; n < 9; ++n)
         vect[n] = rotate[n];
      return;
   }

   vect.resize(16);
   vect[0] = rotate[0];
   vect[4] = rotate[1];
   vect[8] = rotate[2];
   vect[12] = trans[0];
   vect[1] = rotate[3];
   vect[5] = rotate[4];
   vect[9] = rotate[5];
   vect[13] = trans[1];
   vect[2] = rotate[6];
   vect[6] = rotate[7];
   vect[10] = rotate[8];
   vect[14] = trans[2];
   vect[3] = 0;
   vect[7] = 0;
   vect[11] = 0;
   vect[15] = 1;
}

/////////////////////////////////////////////////////////////////////
/// Collect information about geometry hierarchy into flat list
/// like it done in JSROOT ClonedNodes.createClones

void RGeomDescription::Build(TGeoManager *mgr, const std::string &volname)
{
   ClearDescription();
   if (!mgr)
      return;

   TLockGuard lock(fMutex);

   // by top node visibility always enabled and harm logic
   // later visibility can be controlled by other means
   // mgr->GetTopNode()->GetVolume()->SetVisibility(kFALSE);

   int maxnodes = mgr->GetMaxVisNodes();

   SetNSegments(mgr->GetNsegments());
   SetVisLevel(mgr->GetVisLevel());
   SetMaxVisNodes(maxnodes);
   SetMaxVisFaces((maxnodes > 5000 ? 5000 : (maxnodes < 1000 ? 1000 : maxnodes)) * 100);

   auto topnode = mgr->GetTopNode();

   BuildDescription(topnode, topnode->GetVolume());

   if (!volname.empty()) {
      auto vol = mgr->GetVolume(volname.c_str());
      RGeomBrowserIter iter(*this);
      if (vol && (vol != topnode->GetVolume()) && iter.Navigate(vol))
         fSelectedStack = MakeStackByIds(iter.CurrentIds());
   }
}

/////////////////////////////////////////////////////////////////////
/// Collect information about geometry from single volume
/// like it done in JSROOT ClonedNodes.createClones

void RGeomDescription::Build(TGeoVolume *vol)
{
   ClearDescription();
   if (!vol)
      return;

   TLockGuard lock(fMutex);

   fDrawVolume = vol;

   fSelectedStack.clear();

   BuildDescription(nullptr, fDrawVolume);
}

/////////////////////////////////////////////////////////////////////
/// Clear geometry description

void RGeomDescription::ClearDescription()
{
   TLockGuard lock(fMutex);

   fDesc.clear();
   fNodes.clear();
   fSortMap.clear();
   ClearDrawData();
   fDrawIdCut = 0;
   fDrawVolume = nullptr;
   fSelectedStack.clear();
}

/////////////////////////////////////////////////////////////////////
/// Build geometry description

void RGeomDescription::BuildDescription(TGeoNode *topnode, TGeoVolume *topvolume)
{
   // vector to remember numbers
   std::vector<int> numbers;
   int offset = 1000000000;

   // try to build flat list of all nodes
   TGeoNode *snode = topnode;
   TGeoIterator iter(topvolume);
   do {
      if (!snode) {
         numbers.emplace_back(offset);
         fNodes.emplace_back(nullptr);
      } else if (snode->GetNumber() >= offset) {
         // artificial offset already applied, used as identifier
         iter.Skip(); // no need to look inside
      } else {
         numbers.emplace_back(snode->GetNumber());
         snode->SetNumber(offset + fNodes.size()); // use id with shift 1e9
         fNodes.emplace_back(snode);
      }
   } while ((snode = iter()) != nullptr);

   fDesc.reserve(fNodes.size());
   fSortMap.reserve(fNodes.size());

   // array for sorting
   std::vector<RGeomNode *> sortarr;
   sortarr.reserve(fNodes.size());

   // create vector of desc and childs
   int cnt = 0;
   for (auto node : fNodes) {

      fDesc.emplace_back(node ? node->GetNumber() - offset : 0);
      TGeoVolume *vol = node ? node->GetVolume() : topvolume;

      auto &desc = fDesc[cnt++];

      sortarr.emplace_back(&desc);

      desc.name = node ? node->GetName() : vol->GetName();

      auto shape = dynamic_cast<TGeoBBox *>(vol->GetShape());
      if (shape) {
         desc.vol = TMath::Sqrt(shape->GetDX() * shape->GetDX() + shape->GetDY() * shape->GetDY() +
                                shape->GetDZ() * shape->GetDZ());
         desc.nfaces = CountShapeFaces(shape);
      }

      CopyMaterialProperties(vol, desc);

      auto chlds = node ? node->GetNodes() : vol->GetNodes();

      PackMatrix(desc.matr, node ? node->GetMatrix() : nullptr);

      if (chlds)
         for (int n = 0; n <= chlds->GetLast(); ++n) {
            auto chld = dynamic_cast<TGeoNode *>(chlds->At(n));
            desc.chlds.emplace_back(chld->GetNumber() - offset);
         }
   }

   // recover numbers
   cnt = 0;
   for (auto node : fNodes) {
      auto number = numbers[cnt++];
      if (node)
         node->SetNumber(number);
   }

   // sort in volume descent order
   std::sort(sortarr.begin(), sortarr.end(), [](RGeomNode *a, RGeomNode *b) { return a->vol > b->vol; });

   cnt = 0;
   for (auto &elem : sortarr) {
      fSortMap.emplace_back(elem->id);
      elem->sortid = cnt++; // keep place in sorted array to correctly apply cut
   }

   MarkVisible(); // set visibility flags

   ProduceIdShifts();
}

/////////////////////////////////////////////////////////////////////
/// Get volume for specified nodeid
/// If specific volume was configured, it will be returned for nodeid==0

TGeoVolume *RGeomDescription::GetVolume(int nodeid)
{
   auto node = fNodes[nodeid];
   if (node)
      return node->GetVolume();
   return nodeid == 0 ? fDrawVolume : nullptr;
}

/////////////////////////////////////////////////////////////////////
/// Set visibility flag for each nodes

int RGeomDescription::MarkVisible(bool on_screen)
{
   int res = 0;
   for (int nodeid = 0; nodeid < (int)fNodes.size(); nodeid++) {

      auto node = fNodes[nodeid];
      auto vol = GetVolume(nodeid);
      auto &desc = fDesc[nodeid];
      desc.vis = 0;
      desc.nochlds = false;

      if (on_screen) {
         if (!node || node->IsOnScreen())
            desc.vis = 99;
      } else {
         if (vol->IsVisible() && !vol->TestAttBit(TGeoAtt::kVisNone))
            desc.vis = 99;

         if (node && !node->IsVisDaughters())
            desc.nochlds = true;

         if ((desc.vis > 0) && (!desc.chlds.empty()) && !desc.nochlds)
            desc.vis = 1;
      }

      if (desc.IsVisible() && desc.CanDisplay())
         res++;
   }

   return res;
}

/////////////////////////////////////////////////////////////////////
/// Count total number of visible childs under each node

void RGeomDescription::ProduceIdShifts()
{
   for (auto &node : fDesc)
      node.idshift = -1;

   using ScanFunc_t = std::function<int(RGeomNode &)>;

   ScanFunc_t scan_func = [&, this](RGeomNode &node) {
      if (node.idshift < 0) {
         node.idshift = 0;
         for (auto id : node.chlds)
            node.idshift += scan_func(fDesc[id]);
      }

      return node.idshift + 1;
   };

   if (!fDesc.empty())
      scan_func(fDesc[0]);
}

/////////////////////////////////////////////////////////////////////
/// Iterate over all nodes and call function for visible

int RGeomDescription::ScanNodes(bool only_visible, int maxlvl, RGeomScanFunc_t func)
{
   if (fDesc.empty())
      return 0;

   std::vector<int> stack;
   stack.reserve(25); // reserve enough space for most use-cases
   int counter = 0;
   auto viter = fVisibility.begin();

   using ScanFunc_t = std::function<int(int, int, bool)>;

   ScanFunc_t scan_func = [&, this](int nodeid, int lvl, bool is_inside) {
      if (!is_inside && (fSelectedStack == stack))
         is_inside = true;

      auto &desc = fDesc[nodeid];
      auto desc_vis = desc.vis;
      int res = 0;

      if (desc.nochlds && (lvl > 0))
         lvl = 0;

      bool can_display = desc.CanDisplay(), scan_childs = true;

      if ((viter != fVisibility.end()) && (compare_stacks(viter->stack, stack) == 0)) {
         can_display = scan_childs = viter->visible;
         desc_vis = !viter->visible ? 0 : (!desc.chlds.empty() ? 1 : 99);
         viter++;
      }

      // same logic as in JSROOT ClonedNodes.scanVisible
      bool is_visible = (lvl >= 0) && (desc_vis > lvl) && can_display && is_inside;

      if (is_visible || !only_visible)
         if (func(desc, stack, is_visible, counter))
            res++;

      counter++; // count sequence id of current position in scan, will be used later for merging drawing lists

      if ((!desc.chlds.empty()) && (((lvl > 0) && scan_childs) || !only_visible)) {
         auto pos = stack.size();
         stack.emplace_back(0);
         for (unsigned k = 0; k < desc.chlds.size(); ++k) {
            stack[pos] = k; // stack provides index in list of childs
            res += scan_func(desc.chlds[k], is_inside ? lvl - 1 : lvl, is_inside);
         }
         stack.pop_back();
      } else {
         counter += desc.idshift;
      }

      return res;
   };

   if (!maxlvl && (GetVisLevel() > 0))
      maxlvl = GetVisLevel();
   if (!maxlvl)
      maxlvl = 4;
   if (maxlvl > 97)
      maxlvl = 97; // check while vis property of node is 99 normally

   return scan_func(0, maxlvl, false);
}

/////////////////////////////////////////////////////////////////////
/// Collect nodes which are used in visibles

void RGeomDescription::CollectNodes(RGeomDrawing &drawing, bool all_nodes)
{
   drawing.cfg = &fCfg;

   drawing.numnodes = fDesc.size();

   if (all_nodes) {
      for (auto &node : fDesc)
         drawing.nodes.emplace_back(&node);
      return;
   }

   // TODO: for now reset all flags, later can be kept longer
   for (auto &node : fDesc)
      node.useflag = false;

   for (auto &item : drawing.visibles) {
      int nodeid = 0;
      for (auto &chindx : item.stack) {
         auto &node = fDesc[nodeid];
         if (!node.useflag) {
            node.useflag = true;
            drawing.nodes.emplace_back(&node);
         }
         if (chindx >= (int)node.chlds.size())
            break;
         nodeid = node.chlds[chindx];
      }

      if (nodeid != item.nodeid)
         printf("Nodeid mismatch %d != %d when extracting nodes for visibles\n", nodeid, item.nodeid);

      auto &node = fDesc[nodeid];
      if (!node.useflag) {
         node.useflag = true;
         drawing.nodes.emplace_back(&node);
      }
   }

   // printf("SELECT NODES %d\n", (int) drawing.nodes.size());
}

/////////////////////////////////////////////////////////////////////
/// Find description object for requested shape
/// If not exists - will be created

std::string RGeomDescription::ProcessBrowserRequest(const std::string &msg)
{
   TLockGuard lock(fMutex);

   std::string res;

   auto request = TBufferJSON::FromJSON<RBrowserRequest>(msg);

   if (msg.empty()) {
      request = std::make_unique<RBrowserRequest>();
      request->first = 0;
      request->number = 100;
   }

   if (!request)
      return res;

   if (request->path.empty() && (request->first == 0) && (GetNumNodes() < (IsPreferredOffline() ? 1000000 : 1000))) {

      std::vector<RGeomNodeBase *> vect(fDesc.size(), nullptr);

      int cnt = 0;
      for (auto &item : fDesc)
         vect[cnt++] = &item;

      res = "DESCR:"s + TBufferJSON::ToJSON(&vect, GetJsonComp()).Data();

      if (!fVisibility.empty()) {
         res += ":__PHYSICAL_VISIBILITY__:";
         res += TBufferJSON::ToJSON(&fVisibility, GetJsonComp()).Data();
      }

      res += ":__SELECTED_STACK__:";
      res += TBufferJSON::ToJSON(&fSelectedStack, GetJsonComp()).Data();

   } else {
      std::vector<RGeoItem> temp_nodes;
      bool toplevel = request->path.empty();

      // create temporary object for the short time
      RBrowserReply reply;
      reply.path = request->path;
      reply.first = request->first;

      RGeomBrowserIter iter(*this);
      if (iter.Navigate(request->path)) {

         reply.nchilds = iter.NumChilds();
         // scan childs of selected nodes
         if (iter.Enter()) {

            while ((request->first > 0) && iter.Next()) {
               request->first--;
            }

            // first element
            auto stack = MakeStackByIds(iter.CurrentIds());

            while (iter.IsValid() && (request->number > 0)) {
               int pvis = IsPhysNodeVisible(stack);
               temp_nodes.emplace_back(iter.GetName(), iter.NumChilds(), iter.GetNodeId(), iter.GetColor(),
                                       iter.GetMaterial(), iter.GetVisible(), pvis < 0 ? iter.GetVisible() : pvis);
               if (toplevel)
                  temp_nodes.back().SetExpanded(true);
               if (stack == fSelectedStack)
                  temp_nodes.back().SetTop(true);
               request->number--;

               if (!stack.empty())
                  stack[stack.size() - 1]++;

               if (!iter.Next())
                  break;
            }
         }
      }

      for (auto &n : temp_nodes)
         reply.nodes.emplace_back(&n);

      res = "BREPL:"s + TBufferJSON::ToJSON(&reply, GetJsonComp()).Data();
   }

   return res;
}

/////////////////////////////////////////////////////////////////////
/// Find description object for requested shape
/// If not exists - will be created

RGeomDescription::ShapeDescr &RGeomDescription::FindShapeDescr(TGeoShape *shape)
{
   for (auto &descr : fShapes)
      if (descr.fShape == shape)
         return descr;

   fShapes.emplace_back(shape);
   auto &elem = fShapes.back();
   elem.id = fShapes.size() - 1;
   return elem;
}

////////////////////////////////////////////////////////////////////////
/// Function produces mesh for provided shape, applying matrix to the result

std::unique_ptr<RootCsg::TBaseMesh> MakeGeoMesh(TGeoMatrix *matr, TGeoShape *shape)
{
   TGeoCompositeShape *comp = dynamic_cast<TGeoCompositeShape *>(shape);

   std::unique_ptr<RootCsg::TBaseMesh> res;

   if (!comp) {
      std::unique_ptr<TBuffer3D> b3d(shape->MakeBuffer3D());

      if (matr) {
         Double_t *v = b3d->fPnts;
         Double_t buf[3];
         for (UInt_t i = 0; i < b3d->NbPnts(); ++i) {
            buf[0] = v[i * 3];
            buf[1] = v[i * 3 + 1];
            buf[2] = v[i * 3 + 2];
            matr->LocalToMaster(buf, &v[i * 3]);
         }
      }

      res.reset(RootCsg::ConvertToMesh(*b3d.get()));
   } else {
      auto node = comp->GetBoolNode();

      TGeoHMatrix mleft, mright;
      if (matr) {
         mleft = *matr;
         mright = *matr;
      }

      mleft.Multiply(node->GetLeftMatrix());
      auto left = MakeGeoMesh(&mleft, node->GetLeftShape());

      mright.Multiply(node->GetRightMatrix());
      auto right = MakeGeoMesh(&mright, node->GetRightShape());

      if (node->IsA() == TGeoUnion::Class())
         res.reset(RootCsg::BuildUnion(left.get(), right.get()));
      if (node->IsA() == TGeoIntersection::Class())
         res.reset(RootCsg::BuildIntersection(left.get(), right.get()));
      if (node->IsA() == TGeoSubtraction::Class())
         res.reset(RootCsg::BuildDifference(left.get(), right.get()));
   }

   return res;
}

/////////////////////////////////////////////////////////////////////
/// Returns really used number of cylindrical segments

int RGeomDescription::GetUsedNSegments(int min)
{
   int nsegm = 0;

   if (GetNSegments() > 0)
      nsegm = GetNSegments();
   else if (gGeoManager && (gGeoManager->GetNsegments() > 0))
      nsegm = gGeoManager->GetNsegments();

   return nsegm > min ? nsegm : min;
}

/////////////////////////////////////////////////////////////////////
/// Count number of faces for the shape

int RGeomDescription::CountShapeFaces(TGeoShape *shape)
{
   if (!shape)
      return 0;

   auto countTubeFaces = [this](const std::array<Double_t, 2> &outerR, const std::array<Double_t, 2> &innerR,
                                Double_t thetaLength = 360.) -> int {
      auto hasrmin = (innerR[0] > 0) || (innerR[1] > 0);

      int radiusSegments = TMath::Max(4, TMath::Nint(thetaLength / 360. * GetUsedNSegments()));

      // external surface
      int numfaces = radiusSegments * (((outerR[0] <= 0) || (outerR[1] <= 0)) ? 1 : 2);

      // internal surface
      if (hasrmin)
         numfaces += radiusSegments * (((innerR[0] <= 0) || (innerR[1] <= 0)) ? 1 : 2);

      // upper cap
      if (outerR[0] > 0)
         numfaces += radiusSegments * ((innerR[0] > 0) ? 2 : 1);
      // bottom cup
      if (outerR[1] > 0)
         numfaces += radiusSegments * ((innerR[1] > 0) ? 2 : 1);

      if (thetaLength < 360)
         numfaces += ((outerR[0] > innerR[0]) ? 2 : 0) + ((outerR[1] > innerR[1]) ? 2 : 0);

      return numfaces;
   };

   if (shape->IsA() == TGeoSphere::Class()) {
      TGeoSphere *sphere = (TGeoSphere *)shape;
      auto widthSegments = sphere->GetNumberOfDivisions();
      auto heightSegments = sphere->GetNz();
      auto phiLength = sphere->GetPhi2() - sphere->GetPhi1();
      auto noInside = sphere->GetRmin() <= 0;

      auto numoutside = widthSegments * heightSegments * 2;
      auto numtop = widthSegments * (noInside ? 1 : 2);
      auto numbottom = widthSegments * (noInside ? 1 : 2);
      auto numcut = (phiLength == 360.) ? 0 : heightSegments * (noInside ? 2 : 4);

      return numoutside * (noInside ? 1 : 2) + numtop + numbottom + numcut;
   } else if (shape->IsA() == TGeoCone::Class()) {
      auto cone = (TGeoCone *)shape;
      return countTubeFaces({cone->GetRmax2(), cone->GetRmax1()}, {cone->GetRmin2(), cone->GetRmin1()});
   } else if (shape->IsA() == TGeoConeSeg::Class()) {
      auto cone = (TGeoConeSeg *)shape;
      return countTubeFaces({cone->GetRmax2(), cone->GetRmax1()}, {cone->GetRmin2(), cone->GetRmin1()},
                            cone->GetPhi2() - cone->GetPhi1());
   } else if (shape->IsA() == TGeoTube::Class()) {
      auto tube = (TGeoTube *)shape;
      return countTubeFaces({tube->GetRmax(), tube->GetRmax()}, {tube->GetRmin(), tube->GetRmin()});
   } else if (shape->IsA() == TGeoTubeSeg::Class()) {
      auto tube = (TGeoTubeSeg *)shape;
      return countTubeFaces({tube->GetRmax(), tube->GetRmax()}, {tube->GetRmin(), tube->GetRmin()},
                            tube->GetPhi2() - tube->GetPhi1());
   } else if (shape->IsA() == TGeoCtub::Class()) {
      auto tube = (TGeoCtub *)shape;
      return countTubeFaces({tube->GetRmax(), tube->GetRmax()}, {tube->GetRmin(), tube->GetRmin()},
                            tube->GetPhi2() - tube->GetPhi1());
   } else if (shape->IsA() == TGeoEltu::Class()) {
      return GetUsedNSegments(4) * 4;
   } else if (shape->IsA() == TGeoTorus::Class()) {
      auto torus = (TGeoTorus *)shape;
      auto radialSegments = GetUsedNSegments(6);
      auto tubularSegments = TMath::Max(8, TMath::Nint(torus->GetDphi() / 360. * GetUsedNSegments()));
      return (torus->GetRmin() > 0 ? 4 : 2) * radialSegments * (tubularSegments + (torus->GetDphi() != 360. ? 1 : 0));
   } else if (shape->IsA() == TGeoPcon::Class()) {
      auto pcon = (TGeoPcon *)shape;

      bool hasrmin = false;
      int radiusSegments = TMath::Max(5, TMath::Nint(pcon->GetDphi() / 360 * GetUsedNSegments()));
      for (int layer = 0; layer < pcon->GetNz(); ++layer)
         if (pcon->GetRmin(layer) > 0.)
            hasrmin = true;
      return (hasrmin ? 4 : 2) * radiusSegments * (pcon->GetNz() - 1);
   } else if (shape->IsA() == TGeoPgon::Class()) {
      auto pgon = (TGeoPgon *)shape;

      bool hasrmin = false;
      int radiusSegments = TMath::Max(5, TMath::Nint(pgon->GetDphi() / 360 * GetUsedNSegments()));
      for (int layer = 0; layer < pgon->GetNz(); ++layer)
         if (pgon->GetRmin(layer) > 0.)
            hasrmin = true;
      return (hasrmin ? 4 : 2) * radiusSegments * (pgon->GetNz() - 1);
   } else if (shape->IsA() == TGeoXtru::Class()) {
      auto xtru = (TGeoXtru *)shape;
      return (xtru->GetNz() - 1) * xtru->GetNvert() * 2 + xtru->GetNvert() * 3;
   } else if (shape->IsA() == TGeoParaboloid::Class()) {
      auto para = (TGeoParaboloid *)shape;
      int radiusSegments = GetUsedNSegments(4), heightSegments = 30;
      int numfaces = (heightSegments + 1) * radiusSegments * 2;
      if (para->GetRlo() == 0.)
         numfaces -= radiusSegments * 2; // complete layer
      if (para->GetRhi() == 0.)
         numfaces -= radiusSegments * 2; // complete layer
      return numfaces;
   } else if (shape->IsA() == TGeoHype::Class()) {
      TGeoHype *hype = (TGeoHype *)shape;
      if ((hype->GetStIn() == 0) && (hype->GetStOut() == 0))
         return countTubeFaces({hype->GetRmax(), hype->GetRmax()}, {hype->GetRmin(), hype->GetRmin()});
      int radiusSegments = GetUsedNSegments(4), heightSegments = 30;
      return radiusSegments * (heightSegments + 1) * ((hype->GetRmin() > 0.) ? 4 : 2);
   } else if (shape->IsA() == TGeoTessellated::Class()) {
      auto tess = (TGeoTessellated *)shape;
      int numfaces = 0;
      for (int i = 0; i < tess->GetNfacets(); ++i) {
         if (tess->GetFacet(i).GetNvert() == 4)
            numfaces += 2;
         else
            numfaces += 1;
      }
      return numfaces;
   } else if (shape->IsA() == TGeoScaledShape::Class()) {
      auto scaled = (TGeoScaledShape *)shape;
      return CountShapeFaces(scaled->GetShape());
   } else if (shape->IsA() == TGeoCompositeShape::Class()) {
      auto comp = (TGeoCompositeShape *)shape;
      if (!comp->GetBoolNode())
         return 0;
      return CountShapeFaces(comp->GetBoolNode()->GetLeftShape()) +
             CountShapeFaces(comp->GetBoolNode()->GetRightShape());
   }

   // many of simple shapes have 12 faces
   return 12;
}

/////////////////////////////////////////////////////////////////////
/// Find description object and create render information

RGeomDescription::ShapeDescr &RGeomDescription::MakeShapeDescr(TGeoShape *shape)
{
   auto &elem = FindShapeDescr(shape);

   if (elem.nfaces == 0) {

      int boundary = 3; //
      if (shape->IsComposite()) {
         // composite is most complex for client, therefore by default build on server
         boundary = 1;
      } else if (!shape->IsCylType()) {
         // simple box geometry is compact and can be delivered as raw
         boundary = 2;
      }

      if (IsBuildShapes() < boundary) {
         elem.nfaces = 1;
         elem.fShapeInfo.shape = shape;
      } else {

         int old_nsegm = -1;
         if (fCfg.nsegm > 0 && gGeoManager) {
            old_nsegm = gGeoManager->GetNsegments();
            gGeoManager->SetNsegments(fCfg.nsegm);
         }

         auto mesh = MakeGeoMesh(nullptr, shape);

         if (old_nsegm > 0 && gGeoManager)
            gGeoManager->SetNsegments(old_nsegm);

         Int_t num_vertices = mesh->NumberOfVertices(), num_polynoms = 0;

         for (unsigned polyIndex = 0; polyIndex < mesh->NumberOfPolys(); ++polyIndex) {

            auto size_of_polygon = mesh->SizeOfPoly(polyIndex);

            if (size_of_polygon >= 3)
               num_polynoms += (size_of_polygon - 2);
         }

         Int_t index_buffer_size = num_polynoms * 3, // triangle indexes
            vertex_buffer_size = num_vertices * 3;   // X,Y,Z array

         elem.nfaces = num_polynoms;

         std::vector<float> vertices(vertex_buffer_size);

         for (Int_t i = 0; i < num_vertices; ++i) {
            auto v = mesh->GetVertex(i);
            vertices[i * 3] = v[0];
            vertices[i * 3 + 1] = v[1];
            vertices[i * 3 + 2] = v[2];
         }

         elem.fRawInfo.raw.resize(vertices.size() * sizeof(float));

         memcpy(reinterpret_cast<char *>(elem.fRawInfo.raw.data()), vertices.data(), vertices.size() * sizeof(float));

         auto &indexes = elem.fRawInfo.idx;

         indexes.resize(index_buffer_size);
         int pos = 0;

         for (unsigned polyIndex = 0; polyIndex < mesh->NumberOfPolys(); ++polyIndex) {
            auto size_of_polygon = mesh->SizeOfPoly(polyIndex);

            // add first triangle
            if (size_of_polygon >= 3)
               for (int i = 0; i < 3; ++i)
                  indexes[pos++] = mesh->GetVertexIndex(polyIndex, i);

            // add following triangles
            if (size_of_polygon > 3)
               for (unsigned vertex = 3; vertex < size_of_polygon; vertex++) {
                  indexes[pos++] = mesh->GetVertexIndex(polyIndex, 0);
                  indexes[pos++] = mesh->GetVertexIndex(polyIndex, vertex - 1);
                  indexes[pos++] = mesh->GetVertexIndex(polyIndex, vertex);
               }
         }
      }
   }

   return elem;
}

/////////////////////////////////////////////////////////////////////
/// Copy material properties

void RGeomDescription::CopyMaterialProperties(TGeoVolume *volume, RGeomNode &node)
{
   if (!volume)
      return;

   TColor *col = nullptr;

   if ((volume->GetFillColor() > 1) && (volume->GetLineColor() == 1))
      col = gROOT->GetColor(volume->GetFillColor());
   else if (volume->GetLineColor() >= 0)
      col = gROOT->GetColor(volume->GetLineColor());

   if (volume->GetMedium() && (volume->GetMedium() != TGeoVolume::DummyMedium()) &&
       volume->GetMedium()->GetMaterial()) {
      auto material = volume->GetMedium()->GetMaterial();

      node.material = material->GetName();

      auto fillstyle = material->GetFillStyle();
      if ((fillstyle >= 3000) && (fillstyle <= 3100))
         node.opacity = (3100 - fillstyle) / 100.;
      if (!col)
         col = gROOT->GetColor(material->GetFillColor());
   } else {
      node.material.clear();
   }

   if (col) {
      TString colbuf;
      colbuf.Form("#%02x%02x%02x", (int)(col->GetRed() * 255), (int)(col->GetGreen() * 255),
                  (int)(col->GetBlue() * 255));
      node.color = colbuf.Data();
      if (node.opacity == 1.)
         node.opacity = col->GetAlpha();
   } else {
      node.color.clear();
   }
}

/////////////////////////////////////////////////////////////////////
/// Reset shape info, which used to pack binary data

void RGeomDescription::ResetRndrInfos()
{
   for (auto &s : fShapes)
      s.reset();
}

/////////////////////////////////////////////////////////////////////
/// Produce JSON string which can be directly used with `build`
/// function from JSROOT to create three.js model of configured geometry
///
/// Collect all information required to draw geometry on the client
/// This includes list of each visible nodes, meshes and matrixes
/// If @param all_nodes is true, all existing nodes will be provided,
/// which allows to create complete nodes hierarchy on client side
///
/// Example of usage:
///
/// void geom() {
///    auto f = TFile::Open("file_name.root");
///    auto vol = f->Get<TGeoVolume>("object_name");
///    ROOT::RGeomDescription desc;
///    desc.Build(vol);
///    std::ofstream fout("geom.json");
///    fout << desc.ProduceJson();
///  }
///
///  In JSROOT one loads data from JSON file and call `build` function to
///  produce three.js model. Also see example in tutorials/visualisation/webgui/geom/ folder

std::string RGeomDescription::ProduceJson(bool all_nodes)
{
   TLockGuard lock(fMutex);

   std::vector<int> viscnt(fDesc.size(), 0);

   int level = GetVisLevel();

   // first count how many times each individual node appears
   int numnodes = ScanNodes(true, level, [&viscnt](RGeomNode &node, std::vector<int> &, bool, int) {
      viscnt[node.id]++;
      return true;
   });

   if (GetMaxVisNodes() > 0) {
      while ((numnodes > GetMaxVisNodes()) && (level > 1)) {
         level--;
         viscnt.assign(viscnt.size(), 0);
         numnodes = ScanNodes(true, level, [&viscnt](RGeomNode &node, std::vector<int> &, bool, int) {
            viscnt[node.id]++;
            return true;
         });
      }
   }

   fActualLevel = level;
   fDrawIdCut = 0;

   int totalnumfaces = 0, totalnumnodes = 0;

   // for (auto &node : fDesc)
   //   node.SetDisplayed(false);

   // build all shapes in volume decreasing order
   for (auto &sid : fSortMap) {
      fDrawIdCut++; //
      auto &desc = fDesc[sid];

      if ((viscnt[sid] <= 0) || (desc.vol <= 0))
         continue;

      auto shape = GetVolume(sid)->GetShape();
      if (!shape)
         continue;

      // now we need to create TEveGeoPolyShape, which can provide all rendering data
      auto &shape_descr = MakeShapeDescr(shape);

      // should not happen, but just in case
      if (shape_descr.nfaces <= 0) {
         R__LOG_ERROR(RGeomLog()) << "No faces for the shape " << shape->GetName() << " class " << shape->ClassName();
         continue;
      }

      // check how many faces are created
      totalnumfaces += shape_descr.nfaces * viscnt[sid];
      if ((GetMaxVisFaces() > 0) && (totalnumfaces > GetMaxVisFaces()))
         break;

      // also avoid too many nodes
      totalnumnodes += viscnt[sid];
      if ((GetMaxVisNodes() > 0) && (totalnumnodes > GetMaxVisNodes()))
         break;

      // desc.SetDisplayed(true);
   }

   // finally we should create data for streaming to the client
   // it includes list of visible nodes and rawdata

   RGeomDrawing drawing;
   ResetRndrInfos();
   bool has_shape = false;

   ScanNodes(true, level, [&, this](RGeomNode &node, std::vector<int> &stack, bool, int seqid) {
      if ((node.sortid < fDrawIdCut) && (viscnt[node.id] > 0)) {
         drawing.visibles.emplace_back(node.id, seqid, stack);

         auto &item = drawing.visibles.back();
         item.color = node.color;
         item.opacity = node.opacity;

         auto volume = GetVolume(node.id);

         auto &sd = MakeShapeDescr(volume->GetShape());

         item.ri = sd.rndr_info();
         if (sd.has_shape())
            has_shape = true;
      }
      return true;
   });

   CollectNodes(drawing, all_nodes);

   return MakeDrawingJson(drawing, has_shape);
}

/////////////////////////////////////////////////////////////////////
/// Check if there is draw data available

bool RGeomDescription::HasDrawData() const
{
   TLockGuard lock(fMutex);
   return (fDrawJson.length() > 0) && (fDrawIdCut > 0);
}

/////////////////////////////////////////////////////////////////////
/// Produces search data if necessary

void RGeomDescription::ProduceSearchData()
{
   TLockGuard lock(fMutex);

   if (fSearch.empty() || !fSearchJson.empty())
      return;

   std::string hjson;

   SearchVisibles(fSearch, hjson, fSearchJson);

   (void)hjson; // not used here
}

/////////////////////////////////////////////////////////////////////
/// Collect all information required to draw geometry on the client
/// This includes list of each visible nodes, meshes and matrixes

void RGeomDescription::ProduceDrawData()
{
   auto json = ProduceJson();

   TLockGuard lock(fMutex);

   fDrawJson = "GDRAW:"s + json;
}

/////////////////////////////////////////////////////////////////////
/// Clear raw data. Will be rebuild when next connection will be established

void RGeomDescription::ClearDrawData()
{
   TLockGuard lock(fMutex);

   fDrawJson.clear();
   fSearchJson.clear();
}

/////////////////////////////////////////////////////////////////////
/// Clear cached data, need to be clear when connection broken

void RGeomDescription::ClearCache()
{
   ClearDrawData();

   TLockGuard lock(fMutex);
   fShapes.clear();
   fSearch.clear();
}

/////////////////////////////////////////////////////////////////////
/// return true when node used in main geometry drawing and does not have childs
/// for such nodes one could provide optimize toggling of visibility flags

bool RGeomDescription::IsPrincipalEndNode(int nodeid)
{
   TLockGuard lock(fMutex);

   if ((nodeid < 0) || (nodeid >= (int)fDesc.size()))
      return false;

   auto &desc = fDesc[nodeid];

   return (desc.sortid < fDrawIdCut) && desc.IsVisible() && desc.CanDisplay() && (desc.chlds.empty());
}

/////////////////////////////////////////////////////////////////////
/// Search visible nodes for provided name
/// If number of found elements less than 100, create description and shapes for them
/// Returns number of match elements

int RGeomDescription::SearchVisibles(const std::string &find, std::string &hjson, std::string &json)
{
   TLockGuard lock(fMutex);

   hjson.clear();
   json.clear();

   if (find.empty()) {
      hjson = "FOUND:RESET";
      return 0;
   }

   std::vector<int> nodescnt(fDesc.size(), 0), viscnt(fDesc.size(), 0);

   int nmatches = 0;
   std::string test = find;
   int kind = 0;
   if (test.compare(0, 2, "c:") == 0) {
      test.erase(0, 2);
      kind = 1;
   } else if (test.compare(0, 2, "m:") == 0) {
      test.erase(0, 2);
      kind = 2;
   }

   TRegexp regexp(test.c_str());

   auto match_func = [&regexp, kind](RGeomNode &node) {
      return (node.vol > 0) && (TString(node.GetArg(kind)).Index(regexp) >= 0);
   };

   // first count how many times each individual node appears
   ScanNodes(false, 0,
             [&nodescnt, &viscnt, &match_func, &nmatches](RGeomNode &node, std::vector<int> &, bool is_vis, int) {
                if (match_func(node)) {
                   nmatches++;
                   nodescnt[node.id]++;
                   if (is_vis)
                      viscnt[node.id]++;
                };
                return true;
             });

   // do not send too much data, limit could be made configurable later
   if (nmatches == 0) {
      hjson = "FOUND:NO";
      return nmatches;
   }

   if ((GetMaxVisNodes() > 0) && (nmatches > 10 * GetMaxVisNodes())) {
      hjson = "FOUND:Too many " + std::to_string(nmatches);
      return nmatches;
   }

   // now build all necessary shapes and check number of faces - not too many

   int totalnumfaces = 0, totalnumnodes = 0, scnt = 0;
   bool send_rawdata = true;

   // build all shapes in volume decreasing order
   for (auto &sid : fSortMap) {
      if (scnt++ < fDrawIdCut)
         continue; // no need to send most significant shapes

      if (viscnt[sid] == 0)
         continue; // this node is not used at all

      auto &desc = fDesc[sid];
      if ((viscnt[sid] <= 0) && (desc.vol <= 0))
         continue;

      auto shape = GetVolume(sid)->GetShape();
      if (!shape)
         continue;

      // create shape raw data
      auto &shape_descr = MakeShapeDescr(shape);

      // should not happen, but just in case
      if (shape_descr.nfaces <= 0) {
         R__LOG_ERROR(RGeomLog()) << "No faces for the shape " << shape->GetName() << " class " << shape->ClassName();
         continue;
      }

      // check how many faces are created
      totalnumfaces += shape_descr.nfaces * viscnt[sid];
      if ((GetMaxVisFaces() > 0) && (totalnumfaces > GetMaxVisFaces())) {
         send_rawdata = false;
         break;
      }

      // also avoid too many nodes
      totalnumnodes += viscnt[sid];
      if ((GetMaxVisNodes() > 0) && (totalnumnodes > GetMaxVisNodes())) {
         send_rawdata = false;
         break;
      }
   }

   // only for debug purposes - remove later
   // send_rawdata = false;

   // finally we should create data for streaming to the client
   // it includes list of visible nodes and rawdata (if there is enough space)

   std::vector<RGeomNodeBase> found_desc;        ///<! hierarchy of nodes, used for search
   std::vector<int> found_map(fDesc.size(), -1); ///<! mapping between nodeid - > foundid

   // these are only selected nodes to produce hierarchy

   found_desc.emplace_back(0);
   found_desc[0].vis = fDesc[0].vis;
   found_desc[0].name = fDesc[0].name;
   found_desc[0].color = fDesc[0].color;
   found_map[0] = 0;

   ResetRndrInfos();

   RGeomDrawing drawing;
   bool has_shape = true;

   ScanNodes(false, 0, [&, this](RGeomNode &node, std::vector<int> &stack, bool is_vis, int seqid) {
      // select only nodes which should match
      if (!match_func(node))
         return true;

      // add entries into hierarchy of found elements
      int prntid = 0;
      for (auto &s : stack) {
         int chldid = fDesc[prntid].chlds[s];
         if (found_map[chldid] <= 0) {
            int newid = found_desc.size();
            found_desc.emplace_back(newid); // potentially original id can be used here
            found_map[chldid] = newid;      // re-map into reduced hierarchy

            found_desc.back().vis = fDesc[chldid].vis;
            found_desc.back().name = fDesc[chldid].name;
            found_desc.back().color = fDesc[chldid].color;
            found_desc.back().material = fDesc[chldid].material;
         }

         auto pid = found_map[prntid];
         auto cid = found_map[chldid];

         // now add entry into childs lists
         auto &pchlds = found_desc[pid].chlds;
         if (std::find(pchlds.begin(), pchlds.end(), cid) == pchlds.end())
            pchlds.emplace_back(cid);

         prntid = chldid;
      }

      // no need to add visibles
      if (!is_vis)
         return true;

      drawing.visibles.emplace_back(node.id, seqid, stack);

      // no need to transfer shape if it provided with main drawing list
      // also no binary will be transported when too many matches are there
      if (!send_rawdata || (node.sortid < fDrawIdCut)) {
         // do not include render data
         return true;
      }

      auto &item = drawing.visibles.back();
      auto volume = GetVolume(node.id);

      item.color = node.color;
      item.opacity = node.opacity;

      auto &sd = MakeShapeDescr(volume->GetShape());

      item.ri = sd.rndr_info();
      if (sd.has_shape())
         has_shape = true;
      return true;
   });

   hjson = "FESCR:"s + TBufferJSON::ToJSON(&found_desc, GetJsonComp()).Data();

   CollectNodes(drawing);

   json = "FDRAW:"s + MakeDrawingJson(drawing, has_shape);

   return nmatches;
}

/////////////////////////////////////////////////////////////////////////////////
/// Returns nodeid for given stack array, returns -1 in case of failure

int RGeomDescription::FindNodeId(const std::vector<int> &stack)
{
   TLockGuard lock(fMutex);

   int nodeid = 0;

   for (auto &chindx : stack) {
      auto &node = fDesc[nodeid];
      if (chindx >= (int)node.chlds.size())
         return -1;
      nodeid = node.chlds[chindx];
   }

   return nodeid;
}

/////////////////////////////////////////////////////////////////////////////////
/// Creates stack for given array of ids, first element always should be 0

std::vector<int> RGeomDescription::MakeStackByIds(const std::vector<int> &ids)
{
   TLockGuard lock(fMutex);

   std::vector<int> stack;

   if (ids.empty())
      return stack;

   if (ids[0] != 0) {
      printf("Wrong first id\n");
      return stack;
   }

   int nodeid = 0;

   for (unsigned k = 1; k < ids.size(); ++k) {

      int prntid = nodeid;
      nodeid = ids[k];

      if (nodeid >= (int)fDesc.size()) {
         printf("Wrong node id %d\n", nodeid);
         stack.clear();
         return stack;
      }
      auto &chlds = fDesc[prntid].chlds;
      auto pos = std::find(chlds.begin(), chlds.end(), nodeid);
      if (pos == chlds.end()) {
         printf("Wrong id %d not a child of %d - fail to find stack num %d\n", nodeid, prntid, (int)chlds.size());
         stack.clear();
         return stack;
      }

      stack.emplace_back(std::distance(chlds.begin(), pos));
   }

   return stack;
}

/////////////////////////////////////////////////////////////////////////////////
/// Produce stack based on string path
/// Used to highlight geo volumes by browser hover event

std::vector<int> RGeomDescription::MakeStackByPath(const std::vector<std::string> &path)
{
   TLockGuard lock(fMutex);

   std::vector<int> res;

   RGeomBrowserIter iter(*this);

   if (iter.Navigate(path))
      res = MakeStackByIds(iter.CurrentIds());

   return res;
}

/////////////////////////////////////////////////////////////////////////////////
/// Produce list of node ids for given stack
/// If found nodes preselected - use their ids

std::vector<int> RGeomDescription::MakeIdsByStack(const std::vector<int> &stack)
{
   TLockGuard lock(fMutex);

   std::vector<int> ids;

   ids.emplace_back(0);
   int nodeid = 0;
   bool failure = false;

   for (auto s : stack) {
      auto &chlds = fDesc[nodeid].chlds;
      if (s >= (int)chlds.size()) {
         failure = true;
         break;
      }

      ids.emplace_back(chlds[s]);

      nodeid = chlds[s];
   }

   if (failure) {
      printf("Fail to convert stack into list of nodes\n");
      ids.clear();
   }

   return ids;
}

/////////////////////////////////////////////////////////////////////////////////
/// Returns path string for provided stack

std::vector<std::string> RGeomDescription::MakePathByStack(const std::vector<int> &stack)
{
   TLockGuard lock(fMutex);

   std::vector<std::string> path;

   auto ids = MakeIdsByStack(stack);
   path.reserve(ids.size());
for (auto &id : ids)
      path.emplace_back(fDesc[id].name);

   return path;
}

/////////////////////////////////////////////////////////////////////////////////
/// Return string with only part of nodes description which were modified
/// Checks also volume

std::string RGeomDescription::ProduceModifyReply(int nodeid)
{
   TLockGuard lock(fMutex);

   std::vector<RGeomNodeBase *> nodes;
   auto vol = GetVolume(nodeid);

   // we take not only single node, but all there same volume is referenced
   // nodes.push_back(&fDesc[nodeid]);

   int id = 0;
   for (auto &desc : fDesc)
      if (GetVolume(id++) == vol)
         nodes.emplace_back(&desc);

   return "MODIF:"s + TBufferJSON::ToJSON(&nodes, GetJsonComp()).Data();
}

/////////////////////////////////////////////////////////////////////////////////
/// Produce shape rendering data for given stack
/// All nodes, which are referencing same shape will be transferred
/// Returns true if new render information provided

bool RGeomDescription::ProduceDrawingFor(int nodeid, std::string &json, bool check_volume)
{
   TLockGuard lock(fMutex);

   // only this shape is interesting

   TGeoVolume *vol = (nodeid < 0) ? nullptr : GetVolume(nodeid);

   if (!vol || !vol->GetShape()) {
      json.append("NO");
      return false;
   }

   RGeomDrawing drawing;

   ScanNodes(true, 0, [&, this](RGeomNode &node, std::vector<int> &stack, bool, int seq_id) {
      // select only nodes which reference same shape

      if (check_volume) {
         if (GetVolume(node.id) != vol)
            return true;
      } else {
         if (node.id != nodeid)
            return true;
      }

      drawing.visibles.emplace_back(node.id, seq_id, stack);

      auto &item = drawing.visibles.back();

      item.color = node.color;
      item.opacity = node.opacity;
      return true;
   });

   // no any visible nodes were done
   if (drawing.visibles.empty()) {
      json.append("NO");
      return false;
   }

   ResetRndrInfos();

   bool has_shape = false, has_raw = false;

   auto &sd = MakeShapeDescr(vol->GetShape());

   // assign shape data
   for (auto &item : drawing.visibles) {
      item.ri = sd.rndr_info();
      if (sd.has_shape())
         has_shape = true;
      if (sd.has_raw())
         has_raw = true;
   }

   CollectNodes(drawing);

   json.append(MakeDrawingJson(drawing, has_shape));

   return has_raw || has_shape;
}

/////////////////////////////////////////////////////////////////////////////////
/// Produce JSON for the drawing
/// If TGeoShape appears in the drawing, one has to keep typeinfo
/// But in this case one can exclude several classes which are not interesting,
/// but appears very often

std::string RGeomDescription::MakeDrawingJson(RGeomDrawing &drawing, bool has_shapes)
{
   int comp = GetJsonComp();

   if (!has_shapes || (comp < TBufferJSON::kSkipTypeInfo))
      return TBufferJSON::ToJSON(&drawing, comp).Data();

   comp = comp % TBufferJSON::kSkipTypeInfo; // no typeinfo skipping

   TBufferJSON json;
   json.SetCompact(comp);
   json.SetSkipClassInfo(TClass::GetClass<RGeomDrawing>());
   json.SetSkipClassInfo(TClass::GetClass<RGeomNode>());
   json.SetSkipClassInfo(TClass::GetClass<RGeomVisible>());
   json.SetSkipClassInfo(TClass::GetClass<RGeomShapeRenderInfo>());
   json.SetSkipClassInfo(TClass::GetClass<RGeomRawRenderInfo>());

   return json.StoreObject(&drawing, TClass::GetClass<RGeomDrawing>()).Data();
}

/////////////////////////////////////////////////////////////////////////////////
/// Change visibility for specified element
/// Returns true if changes was performed

bool RGeomDescription::ChangeNodeVisibility(const std::vector<std::string> &path, bool selected)
{
   TLockGuard lock(fMutex);

   RGeomBrowserIter giter(*this);
   if (!giter.Navigate(path))
      return false;

   auto nodeid = giter.GetNodeId();

   auto &dnode = fDesc[nodeid];

   auto vol = GetVolume(nodeid);

   // nothing changed
   if (vol->IsVisible() == selected)
      return false;

   dnode.vis = selected ? 99 : 0;
   vol->SetVisibility(selected);
   if (!dnode.chlds.empty()) {
      if (selected)
         dnode.vis = 1; // visibility disabled when any child
      vol->SetVisDaughters(selected);
   }

   int id = 0;
   for (auto &desc : fDesc)
      if (GetVolume(id++) == vol)
         desc.vis = dnode.vis;

   auto stack = MakeStackByIds(giter.CurrentIds());

   // any change in logical node visibility erase individual physical node settings
   for (auto iter = fVisibility.begin(); iter != fVisibility.end(); iter++)
      if (compare_stacks(iter->stack, stack) == 0) {
         fVisibility.erase(iter);
         break;
      }

   ClearDrawData(); // after change raw data is no longer valid

   return true;
}

/////////////////////////////////////////////////////////////////////////////////
/// Change visibility for specified element
/// Returns true if changes was performed

std::unique_ptr<RGeomNodeInfo> RGeomDescription::MakeNodeInfo(const std::vector<int> &stack)
{
   auto path = MakePathByStack(stack);

   TLockGuard lock(fMutex);

   std::unique_ptr<RGeomNodeInfo> res;

   RGeomBrowserIter iter(*this);

   if (iter.Navigate(path)) {

      auto node = fNodes[iter.GetNodeId()];

      auto &desc = fDesc[iter.GetNodeId()];

      res = std::make_unique<RGeomNodeInfo>();

      res->path = path;
      res->node_name = node ? node->GetName() : "node_name";
      res->node_type = node ? node->ClassName() : "no class";

      auto vol = GetVolume(iter.GetNodeId());

      TGeoShape *shape = vol ? vol->GetShape() : nullptr;

      if (shape) {
         res->shape_name = shape->GetName();
         res->shape_type = shape->ClassName();
      }

      if (shape && desc.CanDisplay()) {

         auto &shape_descr = MakeShapeDescr(shape);

         res->ri = shape_descr.rndr_info(); // temporary pointer, can be used preserved for short time
      }
   }

   return res;
}

/////////////////////////////////////////////////////////////////////////////////
/// Select top node by path
/// Used by the client to change active node
/// Returns true if selected node was changed

bool RGeomDescription::SelectTop(const std::vector<std::string> &path)
{
   TLockGuard lock(fMutex);

   RGeomBrowserIter iter(*this);

   if (!iter.Navigate(path))
      return false;

   auto stack = MakeStackByIds(iter.CurrentIds());
   if (stack == fSelectedStack)
      return false;

   fSelectedStack = stack;

   ClearDrawData();

   return true;
}

/////////////////////////////////////////////////////////////////////////////////
/// Set visibility of physical node by path
/// It overrules TGeo visibility flags - but only for specific physical node

bool RGeomDescription::SetPhysNodeVisibility(const std::vector<std::string> &path, bool on)
{
   TLockGuard lock(fMutex);

   RGeomBrowserIter giter(*this);

   if (!giter.Navigate(path))
      return false;

   auto stack = MakeStackByIds(giter.CurrentIds());

   auto nodeid = giter.GetNodeId();

   for (auto iter = fVisibility.begin(); iter != fVisibility.end(); iter++) {
      auto res = compare_stacks(iter->stack, stack);

      if (res == 0) {
         bool changed = iter->visible != on;
         if (changed) {
            iter->visible = on;
            ClearDrawData();

            // no need for custom settings if match with description
            if ((fDesc[nodeid].vis > 0) == on)
               fVisibility.erase(iter);
         }

         return changed;
      }

      if (res > 0) {
         fVisibility.emplace(iter, stack, on);
         ClearDrawData();
         return true;
      }
   }

   fVisibility.emplace_back(stack, on);
   ClearDrawData();
   return true;
}

/////////////////////////////////////////////////////////////////////////////////
/// Set visibility of physical node by itemname
/// itemname in string with path like "/TOP_1/SUB_2/NODE_3"

bool RGeomDescription::SetPhysNodeVisibility(const std::string &itemname, bool on)
{
   std::vector<std::string> path;
   std::string::size_type p1 = 0;

   while (p1 < itemname.length()) {
      if (itemname[p1] == '/') {
         p1++;
         continue;
      }
      auto p = itemname.find('/', p1);
      if (p == std::string::npos) {
         path.emplace_back(itemname.substr(p1));
         p1 = itemname.length();
      } else {
         path.emplace_back(itemname.substr(p1, p - p1));
         p1 = p + 1;
      }
   }

   return SetPhysNodeVisibility(path, on);
}

/////////////////////////////////////////////////////////////////////////////////
/// Check if there special settings for specified physical node
/// returns -1 if nothing is found

int RGeomDescription::IsPhysNodeVisible(const std::vector<int> &stack)
{
   for (auto &item : fVisibility) {
      unsigned sz = item.stack.size();
      if (stack.size() < sz)
         continue;
      bool match = true;
      for (unsigned n = 0; n < sz; ++n)
         if (stack[n] != item.stack[n]) {
            match = false;
            break;
         }

      if (match)
         return item.visible ? 1 : 0;
   }
   return -1;
}

/////////////////////////////////////////////////////////////////////////////////
/// Reset custom visibility of physical node by path

bool RGeomDescription::ClearPhysNodeVisibility(const std::vector<std::string> &path)
{
   TLockGuard lock(fMutex);

   RGeomBrowserIter giter(*this);

   if (!giter.Navigate(path))
      return false;

   auto stack = MakeStackByIds(giter.CurrentIds());

   for (auto iter = fVisibility.begin(); iter != fVisibility.end(); iter++)
      if (compare_stacks(iter->stack, stack) == 0) {
         fVisibility.erase(iter);
         ClearDrawData();
         return true;
      }

   return false;
}

/////////////////////////////////////////////////////////////////////////////////
/// Reset all custom visibility settings

bool RGeomDescription::ClearAllPhysVisibility()
{
   TLockGuard lock(fMutex);

   if (fVisibility.empty())
      return false;

   fVisibility.clear();
   ClearDrawData();
   return true;
}

/////////////////////////////////////////////////////////////////////////////////
/// Change configuration by client
/// Returns true if any parameter was really changed

bool RGeomDescription::ChangeConfiguration(const std::string &json)
{
   auto cfg = TBufferJSON::FromJSON<RGeomConfig>(json);
   if (!cfg)
      return false;

   TLockGuard lock(fMutex);

   auto json1 = TBufferJSON::ToJSON(cfg.get());
   auto json2 = TBufferJSON::ToJSON(&fCfg);

   if (json1 == json2)
      return false;

   fCfg = *cfg; // use assign

   ClearDrawData();

   return true;
}

/////////////////////////////////////////////////////////////////////////////////
/// Change search query and belongs to it json string
/// Returns true if any parameter was really changed

bool RGeomDescription::SetSearch(const std::string &query, const std::string &json)
{
   TLockGuard lock(fMutex);

   bool changed = (fSearch != query) || (fSearchJson != json);
   fSearch = query;
   fSearchJson = json;
   return changed;
}

/////////////////////////////////////////////////////////////////////////////////
/// Save geometry configuration as C++ macro

void RGeomDescription::SavePrimitive(std::ostream &fs, const std::string &name)
{
   std::string prefix = "   ";

   if (fCfg.vislevel != 0)
      fs << prefix << name << "SetVisLevel(" << fCfg.vislevel << ");" << std::endl;
   if (fCfg.maxnumnodes != 0)
      fs << prefix << name << "SetMaxVisNodes(" << fCfg.maxnumnodes << ");" << std::endl;
   if (fCfg.maxnumfaces != 0)
      fs << prefix << name << "SetMaxVisFaces(" << fCfg.maxnumfaces << ");" << std::endl;
   if (fCfg.showtop)
      fs << prefix << name << "SetTopVisible(true);" << std::endl;
   if (fCfg.build_shapes != 1)
      fs << prefix << name << "SetBuildShapes(" << fCfg.build_shapes << ");" << std::endl;
   if (fCfg.nsegm != 0)
      fs << prefix << name << "SetNSegments(" << fCfg.nsegm << ");" << std::endl;
   if (!fCfg.drawopt.empty())
      fs << prefix << name << "SetDrawOptions(\"" << fCfg.drawopt << "\");" << std::endl;
   if (fJsonComp != 0)
      fs << prefix << name << "SetJsonComp(" << fJsonComp << ");" << std::endl;

   // store custom visibility flags
   for (auto &item : fVisibility) {
      auto path = MakePathByStack(item.stack);
      fs << prefix << name << "SetPhysNodeVisibility(";
      for (int i = 0; i < (int)path.size(); ++i)
         fs << (i == 0 ? "{\"" : ", \"") << path[i] << "\"";
      fs << "}, " << (item.visible ? "true" : "false") << ");" << std::endl;
   }
}
