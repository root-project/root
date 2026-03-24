
#include <ROOT/REveGeoTopNode.hxx>

#include <ROOT/RBrowserRequest.hxx>
#include <ROOT/RBrowserReply.hxx>

#include <ROOT/REveRenderData.hxx>
#include <ROOT/RGeomData.hxx>
#include <ROOT/RWebWindow.hxx>
#include <ROOT/REveManager.hxx>
#include <ROOT/REveGeoPolyShape.hxx>

#include <ROOT/REveSelection.hxx>

#include <ROOT/REveUtil.hxx>
#include <ROOT/RLogger.hxx>
#include <ROOT/REveUtil.hxx>
#include "TBufferJSON.h"
#include "TMath.h"

#include "TGeoCompositeShape.h"
#include "TGeoManager.h"
#include "TClass.h"
#include "TGeoNode.h"
#include "TBase64.h"
#include "TStopwatch.h"


#include <cassert>
#include <iostream>
#include <regex>
#include <nlohmann/json.hpp>


using namespace ROOT::Experimental;

thread_local ElementId_t gSelId;

#define REVEGEO_DEBUG
#ifdef REVEGEO_DEBUG
#define REVEGEO_DEBUG_PRINT(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
#define REVEGEO_DEBUG_PRINT(fmt, ...)
#endif


TGeoManager* REveGeomDescription::s_geoManager = nullptr;

/*
namespace {
void PrintStackPath(const std::vector<int>& stack)
{
   printf("Path: ");

   for (auto idx : stack)
      printf("/%d", idx);

   printf("\n");
}
}*/


bool REveGeomDescription::ChangeEveVisibility(const std::vector<int> &stack, ERnrFlags flags, bool on)
{
   std::vector<RGeomNodeVisibility> &visVec = (flags == kRnrSelf) ? fVisibilitySelf : fVisibilityRec;

   for (auto iter = visVec.begin(); iter != visVec.end(); iter++) {
      if (iter->stack == stack) {
         // AMT TODO remove  path fom the vsibilirt vector if it is true
         iter->visible = on;
         return true;
      }
   }

   visVec.emplace_back(stack, on);
   return true;
}

void REveGeomDescription::RefineGeoItem(ROOT::RGeoItem &item, const std::vector<int> &iStack)
{
   std::vector<int> stack = fApex.GetIndexStack();
   stack.insert(stack.end(), iStack.begin(), iStack.end());

   auto isVisible = [&stack](std::vector<RGeomNodeVisibility> &visVec) -> bool {
      for (auto &visVecEl : visVec) {
         /*
         printf("compare ======\n");
         PrintStackPath(stack);
         PrintStackPath(visVecEl.stack);
*/
         if (stack == visVecEl.stack)
            return visVecEl.visible ? 1 : 0;
      }
      return true;
   };

   int visSelf = isVisible(fVisibilitySelf);
   int visRec = isVisible(fVisibilityRec);

    item.SetLogicalVisibility(visRec);
    item.SetPhysicalVisibility(visSelf);

   //return RGeoItem(node.name, node.chlds.size(), node.id, node.color, node.material,
   //                visRec, vis);
}

void REveGeomDescription::SetTopNodeWithPath(const std::vector<std::string>& path)
{
   fApex.SetFromPath(path);
   Build(fApex.GetNode()->GetVolume()); // rebuild geo-webviewer
}

bool REveGeomDescription::GetVisiblityForStack(const std::vector<int> &nodeStack)
{
   // visibility self
   for (auto &visVecEl : fVisibilitySelf) {
      if (nodeStack == visVecEl.stack) {
         return false;
      }
   }

   // visibility recurse/children
   for (auto &visVecEl : fVisibilityRec) {
      bool inside =
         nodeStack.size() >= visVecEl.stack.size() && std::equal(visVecEl.stack.begin(), visVecEl.stack.end(), nodeStack.begin());
      if (inside)
          return false;
   }

   return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Table signal handling

void REveGeomHierarchy::WebWindowCallback(unsigned connid, const std::string &arg)
{
   using namespace std::string_literals;
   REveGeomDescription &eveDesc = dynamic_cast<REveGeomDescription &>(fDesc);

  REveGeoManagerHolder gmgr(REveGeomDescription::GetGeoManager());
   if (arg.compare(0, 6, "CDTOP:") == 0)
   {
      std::vector<std::string> ep;
      eveDesc.SetTopNodeWithPath(ep);
      fDesc.IssueSignal(this, "CdTop");
      fWebWindow->Send(connid, "RELOAD"s);
   }
   else if (arg.compare(0, 5, "CDUP:") == 0)
   {
       std::vector<std::string> result = eveDesc.GetApexPath();
       result.pop_back();
       eveDesc.SetTopNodeWithPath(result);
      fDesc.IssueSignal(this, "CdUp");
      fWebWindow->Send(connid, "RELOAD"s);
   }
   else if (arg.compare(0, 8, "SETAPEX:") == 0) {
      auto path = TBufferJSON::FromJSON<std::vector<std::string>>(arg.substr(8));

      //const std::vector<int> &sstack = fDesc.GetSelectedStack();
    //  std::vector<std::string> sspath = fDesc.MakePathByStack(sstack);
      std::vector<std::string> result = eveDesc.GetApexPath();
      if (path->size() > 1) {
         result.insert(result.end(), path->begin() + 1, path->end());
         eveDesc.SetTopNodeWithPath(result);
         fDesc.IssueSignal(this, "SelectTop");
         fWebWindow->Send(connid, "RELOAD"s);
      }
   }
   else if ((arg.compare(0, 7, "SETVI0:") == 0) || (arg.compare(0, 7, "SETVI1:") == 0)) {
      {
         REveManager::ChangeGuard ch;
         bool on = (arg[5] == '1');
         auto path = TBufferJSON::FromJSON<std::vector<std::string>>(arg.substr(7));
         // Get integer stack from string stack
         std::vector<int> base = eveDesc.GetIndexStack();
         std::vector<int> stack = fDesc.MakeStackByPath(*path);
         stack.insert(stack.begin(), base.begin(), base.end());

         if (eveDesc.ChangeEveVisibility(stack, REveGeomDescription::kRnrChildren , on)) {
            std::cout << "Set visibilty rnr CHIDLREN \n";
            fReceiver->VisibilityChanged(on, REveGeomDescription::kRnrChildren, stack);
         }
      }
   }
   else if ((arg.compare(0, 5, "SHOW:") == 0) || (arg.compare(0, 5, "HIDE:") == 0)) {
      {
         auto path = TBufferJSON::FromJSON<std::vector<std::string>>(arg.substr(5));
         bool on = (arg.compare(0, 5, "SHOW:") == 0);
         // Get integer stack from string stack

         std::vector<int> base = eveDesc.GetIndexStack();
         std::vector<int> stack = fDesc.MakeStackByPath(*path);
         stack.insert(stack.begin(), base.begin(), base.end());

         if (path && eveDesc.ChangeEveVisibility(stack, REveGeomDescription::kRnrSelf, on)) {
            std::cout << "Set visibilty rnr PHY \n";
            REveManager::ChangeGuard ch;
            fReceiver->VisibilityChanged(on, REveGeomDescription::kRnrSelf, stack);
         }
      }
   }

   else {
      RGeomHierarchy::WebWindowCallback(connid, arg);
   }
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

TGeoManager* REveGeomDescription::GetGeoManager()
{
  assert(s_geoManager);
  return s_geoManager;
}

void REveGeomDescription::Apex::SetFromPath(std::vector<std::string> absPath)
{
   fPath = absPath;
   fNode = LocateNodeWithPath(absPath);
}

TGeoNode *REveGeomDescription::Apex::LocateNodeWithPath(const std::vector<std::string> &path) const
{
   TGeoNode *top = REveGeomDescription::GetGeoManager()->GetTopNode();
   // printf("Top node name from geoData name (%s)\n", top->GetName());
   for (size_t t = 0; t < path.size(); t++) {
      std::string s = path[t];
      std::cout << s << std::endl;
      TGeoNode *ntop = top->GetVolume()->FindNode(s.c_str());
      if (!ntop)
         throw std::runtime_error("Apex::LocateNodeWithPath(), can't locate node with path " + s);
      top = ntop;
   }
   return top;
}

std::string REveGeomDescription::Apex::GetFlatPath() const
{
   if (fPath.empty())
      return "";

   std::ostringstream oss;

   oss << fPath[0];

   for (size_t i = 1; i < fPath.size(); ++i)
      oss << "/" << fPath[i];

   return oss.str();
}

std::vector<int> REveGeomDescription::Apex::GetIndexStack() const
{
    std::vector<int> indexStack;

    TGeoNode* current = REveGeomDescription::GetGeoManager()->GetTopNode();

    // optional: skip first if it is top itself
    size_t start = 0;
    std::vector<std::string> nameStack = fPath;
   if (!nameStack.empty() && nameStack[0] == current->GetName())
        start = 1;

    for (size_t i = start; i < nameStack.size(); ++i)
    {
        const std::string& targetName = nameStack[i];

        TGeoVolume* vol = current->GetVolume();

        int nd = vol->GetNdaughters();

        int foundIndex = -1;

        for (int j = 0; j < nd; ++j)
        {
            TGeoNode* daughter = vol->GetNode(j);

            if (targetName == daughter->GetName())
            {
                foundIndex = j;
                current = daughter;
                break;
            }
        }

        if (foundIndex == -1)
        {
            std::cerr << "Node not found: " << targetName << std::endl;
            return {};
        }

        indexStack.push_back(foundIndex);
    }

    // PrintStackPath(indexStack);
    return indexStack;
}

void REveGeomDescription::ImportFile(const char* filename)
{
   s_geoManager = TGeoManager::Import(filename);
}

////////////////////////////////////////////////////////////////////////////////
///
/// Constructor.

REveGeoTopNodeData::REveGeoTopNodeData(const char* filename)
{
   // this below will be obsolete
   fDesc.AddSignalHandler(this, [this](const std::string &kind) { ProcessSignal(kind); });
   fDesc.ImportFile(filename);


   fWebHierarchy = std::make_shared<REveGeomHierarchy>(fDesc, true);
   fWebHierarchy->SetReceiver(this);
}

void REveGeoTopNodeData::SetTopNodeWithPath(const std::string &path)
{
   std::regex re(R"([/\\]+)"); // split on one or more slashes
   std::sregex_token_iterator it(path.begin(), path.end(), re, -1);
   std::sregex_token_iterator end;
   std::vector<std::string> result;

   for (; it != end; ++it) {
      if (!it->str().empty()) { // skip empty parts
         result.push_back(*it);
      }
   }

   fDesc.SetTopNodeWithPath(result);

   for (auto &el : fNieces) {
      REveGeoTopNodeViz *etn = dynamic_cast<REveGeoTopNodeViz *>(el);
      etn->BuildDesc();
   }
}

void REveGeoTopNodeData::VisibilityChanged(bool on, REveGeomDescription::ERnrFlags flag, const std::vector<int>& path)
{

   for (auto &el : fNieces) {
      REveGeoTopNodeViz *etn = dynamic_cast<REveGeoTopNodeViz *>(el);
      etn->VisibilityChanged(on, flag, path);
   }
}

////////////////////////////////////////////////////////////////////////////////

void REveGeoTopNodeData::SetChannel(unsigned connid, int chid)
{
   fWebHierarchy->Show({gEve->GetWebWindow(), connid, chid});
}

////////////////////////////////////////////////////////////////////////////////
void REveGeoTopNodeData::ProcessSignal(const std::string &kind)
{
   REveManager::ChangeGuard ch;
   if ((kind == "SelectTop") || (kind == "CdTop") || (kind == "CdUp"))
   {
      for (auto &el : fNieces) {
         REveGeoTopNodeViz *etn = dynamic_cast<REveGeoTopNodeViz *>(el);
         etn->BuildDesc();
      }
   }
   else if (kind == "HighlightItem") {
      /*
      printf("REveGeoTopNodeData element highlighted --------------------------------"\n);
      */

   } else if (kind == "ClickItem") {
      printf("REveGeoTopNodeData element CLICKED selected --------------------------------\n");
      auto sstack = fDesc.GetClickedItem();
      std::set<int> ss;

      for (auto &n : fNieces) {
         REveGeoTopNodeViz* viz = dynamic_cast<REveGeoTopNodeViz*>(n);
         viz->GetIndicesFromBrowserStack(sstack, ss);
         bool multi = false;
         bool secondary = true;
         gEve->GetSelection()->NewElementPicked(n->GetElementId(), multi, secondary, ss);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Fill core part of JSON representation.

Int_t REveGeoTopNodeData::WriteCoreJson(nlohmann::json &j, Int_t rnr_offset)
{
   Int_t ret = REveElement::WriteCoreJson(j, rnr_offset);

   return ret;
}

////////////////////////////////////////////////////////////////////////////////
// REveGeoTopNodeViz
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

REveGeoTopNodeViz::REveGeoTopNodeViz(const Text_t *n, const Text_t *t) : REveElement(n, t)
{
   SetAlwaysSecSelect(true);
}

void REveGeoTopNodeViz::SetVizMode(EMode mode)
 {
   fMode = mode;
}

bool REveGeoTopNodeViz::AcceptNode(TGeoIterator &it, bool skip) const
{
   if (fMode == EMode::kModeVisLevel)
   {
      if (it.GetLevel() > fGeoData->fDesc.GetVisLevel()) {
         it.Skip();
         return false;
      }
   }
   else if (fMode == EMode::kModeLeafOnly)
   {
      // printf("accep mkod eleaf node ptr %p \n", (void*)it.GetNode(it.GetLevel()));
      if (it.GetNode(it.GetLevel())->GetNdaughters())
         return false;
   }
   else if (fMode == EMode::kModeMixed)
   {
      if (it.GetLevel() > fGeoData->fDesc.GetVisLevel()) {
         if (skip) it.Skip();
         return false;
      }
      // printf("accep mkod eleaf node ptr %p \n", (void*)it.GetNode(it.GetLevel()));
      if (it.GetNode(it.GetLevel())->GetNdaughters())
         return false;
   }

   return true;
}

std::string REveGeoTopNodeViz::GetHighlightTooltip(const std::set<int> & set) const
{
  REveGeoManagerHolder gmgr(REveGeomDescription::GetGeoManager());
   if (set.empty()) {
      return "";
   } else {
      auto it = set.begin();
      int pos = *it;
      //const BNode &bn = fNodes[pos];
      std::cout << "highlight node with ID " <<  pos << "\n";

      std::string res = "GeoNode name";

      TGeoNode *top = fGeoData->fDesc.GetApexNode();
      TGeoIterator git(top->GetVolume());
      TGeoNode *node;
      int i = 0;
      TString path;
      while ((node = git.Next()))
      {
         if (!AcceptNode(git))
            continue;
         if (i == pos) {
            git.GetPath(path);
            res = path;
            break;
         }
         i++;
      }
      return res;
   }
}

void REveGeoTopNodeViz::BuildDesc()
{
   // locate top node
   TGeoNode* top  = fGeoData->fDesc.GetApexNode();

   fNodes.clear();
   fShapes.clear();
   // shape array
   std::set<TGeoShape *> shapes;
   TStopwatch timer;
   timer.Start();
   CollectShapes(top, shapes, fShapes);
   std::cout << "Shape size " << shapes.size() << "\n";

   timer.Stop();

   printf("Real time: %.3f s\n", timer.RealTime());
   printf("CPU  time: %.3f s\n", timer.CpuTime());

   // node array
   timer.Start();
   CollectNodes(top->GetVolume(), fNodes, fShapes);
   std::cout << "Node size " << fNodes.size() << "\n";

   timer.Stop();

   printf("NODES Real time: %.3f s\n", timer.RealTime());
   printf("NODES CPU  time: %.3f s\n", timer.CpuTime());

   StampObjProps();
}

void REveGeoTopNodeViz::CollectNodes(TGeoVolume *volume, std::vector<BNode> &bnl, std::vector<BShape> &browsables)
{
   printf("collect nodes \n");
   TGeoIterator it(volume);
   TGeoNode *node;
   int nodeId = 0;

   std::vector<int> apexStack = fGeoData->RefDescription().GetIndexStack();

   // get top node transformation
   TGeoHMatrix global;
   {
      TGeoNode *inode = REveGeomDescription::GetGeoManager()->GetTopNode();
      for (int idx : apexStack) {
         inode = inode->GetDaughter(idx);
         global.Multiply(inode->GetMatrix());
      }
   }

   while ((node = it.Next()))
   {
      if (!AcceptNode(it))
      continue;

      TGeoHMatrix full = global; // identity if global is identity
      full.Multiply(it.GetCurrentMatrix());
      const TGeoMatrix *mat = &full;

      // const TGeoMatrix *mat = it.GetCurrentMatrix();
      const Double_t *t = mat->GetTranslation();    // size 3
      const Double_t *r = mat->GetRotationMatrix(); // size 9 (3x3)

      Double_t m[16];
      if (mat->IsScale()) {
         const Double_t *s = mat->GetScale();
         m[0] = r[0] * s[0];
         m[1] = r[3] * s[0];
         m[2] = r[6] * s[0];
         m[3] = 0;
         m[4] = r[1] * s[1];
         m[5] = r[4] * s[1];
         m[6] = r[7] * s[1];
         m[7] = 0;
         m[8] = r[2] * s[2];
         m[9] = r[5] * s[2];
         m[10] = r[8] * s[2];
         m[11] = 0;
         m[12] = t[0];
         m[13] = t[1];
         m[14] = t[2];
         m[15] = 1;
      } else {
         m[0] = r[0];
         m[1] = r[3];
         m[2] = r[6];
         m[3] = 0;
         m[4] = r[1];
         m[5] = r[4];
         m[6] = r[7];
         m[7] = 0;
         m[8] = r[2];
         m[9] = r[5];
         m[10] = r[8];
         m[11] = 0;
         m[12] = t[0];
         m[13] = t[1];
         m[14] = t[2];
         m[15] = 1;
      }

      BNode b;
      b.node = node;
      b.nodeId = nodeId;
      b.color = node->GetVolume()->GetLineColor();

     // TString path; it.GetPath(path);
     //  printf("[%d] %d %s \n", node->GetNdaughters(), it.GetLevel(), path.Data());


      // set BNode transformation matrix
      for (int i = 0; i < 16; ++i)
         b.trans[i] = m[i];

      // find shape
      TGeoShape *shape = node->GetVolume()->GetShape();
      b.shapeId = -1; // mark invalid at start
      for (size_t i = 0; i < browsables.size(); i++) {
         if (shape == browsables[i].shape) {
            b.shapeId = i;
            break;
         }
      }
      assert(b.shapeId >= 0);


      // set visibility flag
      std::vector<int> visStack = apexStack;
      for (int i = 1; i <= it.GetLevel(); ++i)
         visStack.push_back(it.GetIndex(i));
      // PrintStackPath(visStack);
      b.visible = fGeoData->RefDescription().GetVisiblityForStack(visStack);

      // printf("Node %d shape id %d \n", (int)bnl.size(), b.shapeId);
      bnl.push_back(b);
      nodeId++;

      if (nodeId > 300000) {
         R__LOG_ERROR(REveLog()) << "Max number of nodes reached ... breaking the loop \n";
         printf("num nodes locked !!! \n");
         break;
      }
   }
}

void REveGeoTopNodeViz::CollectShapes(TGeoNode *tnode, std::set<TGeoShape *> &shapes, std::vector<BShape> &browsables)
{
   printf("collect shapes \n");
   TGeoIterator geoit(tnode->GetVolume());
   TGeoNode *node = nullptr;
   while ((node = geoit.Next()))
   {
      if (!AcceptNode(geoit))
         continue;

      TGeoVolume *vol = node->GetVolume();
      if (vol) {
         TGeoShape *shape = vol->GetShape();
         if (shape) {
            auto it = shapes.find(shape);
            if (it == shapes.end()) {
               shapes.insert(shape); // use set to avoid duplicates
               REveGeoPolyShape polyShape;
               TGeoCompositeShape *compositeShape = dynamic_cast<TGeoCompositeShape *>(shape);
               int n_seg = 60; // default value in the geo manager and poly shape
               if (compositeShape)
                  polyShape.BuildFromComposite(compositeShape, n_seg);
               else
                  polyShape.BuildFromShape(shape, n_seg);

               // printf("[%d] Shape name %s %s \n",(int)browsables.size(), shape->GetName(), shape->ClassName());

               //   printf("vertices %lu: \n", polyShape.fVertices.size());

               // create browser shape
               BShape browserShape;
               browserShape.shape = shape;
               browsables.push_back(browserShape);

               // copy vertices transform vec double to float
               browsables.back().vertices.reserve(polyShape.fVertices.size());
               for (size_t i = 0; i < polyShape.fVertices.size(); i++)
                  browsables.back().vertices.push_back(polyShape.fVertices[i]);

               // copy indices kip the first integer in the sequence of 4
               for (size_t i = 0; i < polyShape.fPolyDesc.size(); i += 4) {
                  browsables.back().indices.push_back(polyShape.fPolyDesc[i + 1]);
                  browsables.back().indices.push_back(polyShape.fPolyDesc[i + 2]);
                  browsables.back().indices.push_back(polyShape.fPolyDesc[i + 3]);
               }
               // printf("last browsable size indices size %lu \n",  browsables.back().indices.size());
            }
         }
      }
   }
}

void REveGeoTopNodeViz::BuildRenderData()
{
   fRenderData = std::make_unique<REveRenderData>("makeGeoTopNode");
   for (size_t i = 0; i < fNodes.size(); ++i) {

      UChar_t c[4] = {1, 2, 3, 4};
      REveUtil::ColorFromIdx(fNodes[i].color, c);
      // if (i < 400) printf("%d > %d %d %d %d \n",fNodes[i].color, c[0], c[1], c[2], c[3]);
      uint32_t v = (c[0] << 16) + (c[1] << 8) + c[2];
      float pc;
      std::memcpy(&pc, &v, sizeof(pc));
      GetRenderData()->PushV(pc);
   }
}
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
void REveGeoTopNodeViz::SetGeoData(REveGeoTopNodeData *d, bool rebuild)
{
   fGeoData = d;
   if (rebuild)
      BuildDesc();
}

//------------------------------------------------------------------------------

int REveGeoTopNodeViz::WriteCoreJson(nlohmann::json &j, Int_t rnr_offset)
{
   REveGeoManagerHolder gmgr(REveGeomDescription::GetGeoManager());
   Int_t ret = REveElement::WriteCoreJson(j, rnr_offset);

   if (!fGeoData) {
      j["dataId"] = -1;
   } else {
      // std::string json = fGeoData->fDesc.ProduceJson();
      // j["geomDescription"] = TBase64::Encode(json.c_str());
      j["dataId"] = fGeoData->GetElementId();
   }
   j["visLevel"] = fGeoData ? fGeoData->fDesc.GetVisLevel() : 0;

   // put shapes vector in json array
   using namespace nlohmann;

   json shapeVertexArr = json::array();
   int vertexOff = 0;

   json shapeIndexArr = json::array();
   json shapePolySizeArr = json::array();
   json shapePolyOffArr = json::array();

   json nodeVisibility = json::array();

   int polyOff = 0;

   // need four integers for
   for (size_t i = 0; i < fShapes.size(); ++i) {
      // vertices

      std::copy(fShapes[i].vertices.begin(), fShapes[i].vertices.end(), std::back_inserter(shapeVertexArr));

      int numVertices = int(fShapes[i].vertices.size());
      // indices
      // write shape indices with the vertexOff
      for (size_t p = 0; p < fShapes[i].indices.size(); ++p)
         shapeIndexArr.push_back(fShapes[i].indices[p] + vertexOff);

      int numIndices = int(fShapes[i].indices.size());
      shapePolySizeArr.push_back(numIndices);
      shapePolyOffArr.push_back(polyOff);

      // printf("shape [%d] numIndices %d \n", i, numIndices);

      polyOff += numIndices;
      vertexOff += numVertices / 3;
   }

   // write vector of shape ids for visible nodes
   json nodeShapeIds = json::array();
   json nodeTrans = json::array();
   json nodeColors = json::array();

   for (size_t i = 0; i < fNodes.size(); ++i) {
      nodeShapeIds.push_back(fNodes[i].shapeId);
      nodeVisibility.push_back(fNodes[i].visible);
      for (int t = 0; t < 16; t++)
         nodeTrans.push_back(fNodes[i].trans[t]);
   }
   // shape basic array

   j["shapeVertices"] = shapeVertexArr;

   // shape basic indices array
   j["shapeIndices"] = shapeIndexArr;

   // shape poly offset array
   j["shapeIndicesOff"] = shapePolyOffArr;
   j["shapeIndicesSize"] = shapePolySizeArr;

   j["nodeShapeIds"] = nodeShapeIds;
   j["nodeTrans"] = nodeTrans;
   j["nodeVisibility"] = nodeVisibility;
   j["fSecondarySelect"] = fAlwaysSecSelect;




   // ship bounding box info
   TGeoNode* top  = fGeoData->fDesc.GetApexNode();
   TGeoVolume *vol = top->GetVolume();
   TGeoShape *shape = vol->GetShape();
   shape->ComputeBBox();
   TGeoBBox *box = dynamic_cast<TGeoBBox *>(shape);
   if (box) {
      const Double_t *origin = box->GetOrigin();

      printf("BBox center: (%f, %f, %f)\n", origin[0], origin[1], origin[2]);
      //printf("origin lengths: (%f, %f, %f)\n", origin[0], origin[1], origin[2]);

      auto jbb = json::array();
      jbb.push_back(origin[0] - box->GetDX());
      jbb.push_back(origin[0] + box->GetDX());
      jbb.push_back(origin[1] - box->GetDY());
      jbb.push_back(origin[1] + box->GetDY());
      jbb.push_back(origin[2] - box->GetDZ());
      jbb.push_back(origin[2] + box->GetDZ());
      j["bbox"] = jbb;
   }
   // std::cout << "Write Core json " << j.dump(1) << "\n";
   return ret;
}

void REveGeoTopNodeViz::SetVisLevel(int vl)
{
   if (fGeoData) {
      fGeoData->fDesc.SetVisLevel(vl);
      StampObjProps();
   }
}

void REveGeoTopNodeViz::GetIndicesFromBrowserStack(const std::vector<int> &stack, std::set<int> &res)
{
   TGeoNode *top = fGeoData->fDesc.GetApexNode();
   TGeoIterator it(top->GetVolume());
   std::vector<int> nodeStack;
   int cnt = 0;
   TGeoNode *node;

   while ((node = it.Next())) {
      int level = it.GetLevel();

      bool accept = AcceptNode(it, false);

      nodeStack.resize(level);
      if (level > 0)
          nodeStack[level - 1] = it.GetIndex(level);

         bool inside = nodeStack.size() >= stack.size() && std::equal(stack.begin(), stack.end(), nodeStack.begin());
         if (inside) {
            res.insert(cnt);
      } // rnr flags
      if (accept) cnt++;
   } // while it

   printf("GetIndicesFromBrowserStack stack size %zu res size %zu\n", stack.size(), res.size());
}

void REveGeoTopNodeViz::VisibilityChanged(bool on, REveGeomDescription::ERnrFlags flag, const std::vector<int> &iStack)
{
   // function argument is full stack, we remove the apex path
   size_t apexDepth = fGeoData->RefDescription().GetApexPath().size();
   std::vector<int> stack(iStack.begin() + apexDepth, iStack.end());

   // PrintStackPath(stack);

   TGeoNode *top = fGeoData->fDesc.GetApexNode();
   TGeoIterator it(top->GetVolume());
   std::vector<int> nodeStack;
   int cnt = 0;
   TGeoNode *node;
   while ((node = it.Next())) {

      int level = it.GetLevel();
      if (!AcceptNode(it))
         continue;

      nodeStack.resize(level);
      if (level > 0)
          nodeStack[level - 1] = it.GetIndex(level);

      if (flag == REveGeomDescription::kRnrSelf) {
         /*
         printf("nODEcompare ======\n");
         PrintStackPath(stack);
         PrintStackPath(nodeStack);
         */
         if (nodeStack == stack) {
            fNodes[cnt].visible = on;

            break;
         }
      } else {
         bool inside = nodeStack.size() >= stack.size() && std::equal(stack.begin(), stack.end(), nodeStack.begin());
         if (inside) {
            fNodes[cnt].visible = on;
         }
      } // rnr flags
      cnt++;
   } // while it
   StampObjProps();
}