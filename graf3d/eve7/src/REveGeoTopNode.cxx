
#include <ROOT/REveGeoTopNode.hxx>
#include <ROOT/REveRenderData.hxx>
#include <ROOT/RGeomData.hxx>
#include <ROOT/REveManager.hxx>

#include <ROOT/REveSelection.hxx>


#include "TMath.h"
#include "TGeoManager.h"
#include "TClass.h"
#include "TGeoNode.h"
#include "TGeoManager.h"
#include "TBase64.h"

#include <cassert>
#include <iostream>

#include <nlohmann/json.hpp>


using namespace ROOT::Experimental;

thread_local ElementId_t gSelId;

#define REVEGEO_DEBUG
#ifdef REVEGEO_DEBUG
#define REVEGEO_DEBUG_PRINT(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
#define REVEGEO_DEBUG_PRINT(fmt, ...)
#endif

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

REveGeoTopNodeData::REveGeoTopNodeData(const Text_t *n, const Text_t *t) : REveElement(n, t)
{
   fWebHierarchy = std::make_shared<RGeomHierarchy>(fDesc, true);
}

void REveGeoTopNodeData::SetTNode(TGeoNode *n)
{
   fGeoNode = n;
   fDesc.Build(fGeoNode->GetVolume());
   fDesc.AddSignalHandler(this, [this](const std::string &kind) { ProcessSignal(kind); });
}
////////////////////////////////////////////////////////////////////////////////

void REveGeoTopNodeData::SetChannel(unsigned connid, int chid)
{
   fWebHierarchy->Show({gEve->GetWebWindow(), connid, chid});
}

////////////////////////////////////////////////////////////////////////////////
namespace {
std::size_t getHash(std::vector<int> &vec)
{
   std::size_t seed = vec.size();
   for (auto &x : vec) {
      uint32_t i = (uint32_t)x;
      seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
   }
   return seed;
}
} // namespace

void REveGeoTopNodeData::ProcessSignal(const std::string &kind)
{
   REveManager::ChangeGuard ch;
   if ((kind == "SelectTop") || (kind == "NodeVisibility")) {
      StampObjProps();
      for (auto &n : fNieces) {
         n->StampObjProps();
      }
   } else if (kind == "HighlightItem") {
      // printf("REveGeoTopNodeData element highlighted --------------------------------");
      auto sstack = fDesc.GetHighlightedItem();
      std::set<int> ss;
      ss.insert((int)getHash(sstack));
      for (auto &n : fNieces) {
         gEve->GetHighlight()->NewElementPicked(n->GetElementId(), false, true, ss);
      }
      gSelId = gEve->GetHighlight()->GetElementId();

   } else if (kind == "ClickItem") {
      // printf("REveGeoTopNodeData element selected --------------------------------");
      auto sstack = fDesc.GetClickedItem();
      std::set<int> ss;
      ss.insert((int)getHash(sstack));

      for (auto &n : fNieces) {
         gEve->GetSelection()->NewElementPicked(n->GetElementId(), false, true, ss);
      }
      gSelId = gEve->GetSelection()->GetElementId();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Fill core part of JSON representation.

Int_t REveGeoTopNodeData::WriteCoreJson(nlohmann::json &j, Int_t rnr_offset)
{
   Int_t ret = REveElement::WriteCoreJson(j, rnr_offset);

   if (!fGeoNode){ return ret;}
   return ret;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
REveGeoTopNodeViz::REveGeoTopNodeViz(const Text_t *n, const Text_t *t) : REveElement(n, t) {}

std::string REveGeoTopNodeViz::GetHighlightTooltip(const std::set<int> &) const
{
   auto stack = fGeoData->fDesc.GetHighlightedItem();
   auto sa = fGeoData->fDesc.MakePathByStack(stack);
   if (sa.empty())
      return "";
   else {
      std::string res;
      size_t n = sa.size();
      for (size_t i = 0; i < n; ++i) {
         res += sa[i];
         if (i < (n - 1))
            res += "/";
      }
      return res;
   }
}

void REveGeoTopNodeViz::BuildRenderData()
{
   fRenderData = std::make_unique<REveRenderData>("makeGeoTopNode");
}

int REveGeoTopNodeViz::WriteCoreJson(nlohmann::json &j, Int_t rnr_offset)
{
   Int_t ret = REveElement::WriteCoreJson(j, rnr_offset);
   if (!fGeoData) {
      j["dataId"] = -1;
   } else {
      std::string json = fGeoData->fDesc.ProduceJson();
      j["geomDescription"] = TBase64::Encode(json.c_str());
      printf("REveGeoTopNodeViz::WriteCoreJson stream geomDescription json size = %lu\n", json.size());
      j["dataId"] = fGeoData->GetElementId();
   }
   return ret;
}

void REveGeoTopNodeViz::FillExtraSelectionData(nlohmann::json &j, const std::set<int> &) const
{
   j["stack"] = nlohmann::json::array();
   std::vector<int> stack;
   if (gSelId == gEve->GetHighlight()->GetElementId())
      stack = fGeoData->fDesc.GetHighlightedItem();
   else if (gSelId == gEve->GetSelection()->GetElementId())
      stack = fGeoData->fDesc.GetClickedItem();

   if (stack.empty())
      return;

#ifdef REVEGEO_DEBUG
   printf("cicked stack: ");
   for (auto i : stack)
      printf(" %d, ", i);
   printf("\n");
#endif

   for (auto i : stack)
      j["stack"].push_back(i);


#ifdef REVEGEO_DEBUG
   printf("extra stack: ");
   int ss = j["stack"].size();
   for (int i = 0; i < ss; ++i) {
      int d = j["stack"][i];
      printf(" %d,", d);
   }
   printf("----\n");
   auto ids = fGeoData->fDesc.MakeIdsByStack(stack);
   printf("node ids from stack: ");
   for (auto i : ids)
      printf(" %d, ", i);
   printf("\n");

   int id = fGeoData->fDesc.FindNodeId(stack);
   printf("NODE ID %d\n", id);
#endif
}
