
#ifndef ROOT7_REveGeoTopNode
#define ROOT7_REveGeoTopNode

#include <ROOT/REveElement.hxx>
#include <ROOT/RGeomData.hxx>
#include <ROOT/RGeomHierarchy.hxx>

class TGeoNode;

namespace ROOT {
namespace Experimental {


class REveGeoTopNodeData : public REveElement,
                           public REveAuntAsList
{
  friend class REveGeoTopNodeViz;
protected:
   REveGeoTopNodeData(const REveGeoTopNodeData &) = delete;
   REveGeoTopNodeData &operator=(const REveGeoTopNodeData &) = delete;

   TGeoNode* fGeoNode{nullptr};
   RGeomDescription fDesc;                        ///<! geometry description, send to the client as first message
   std::shared_ptr<RGeomHierarchy> fWebHierarchy; ///<! web handle for hierarchy part

public:
   REveGeoTopNodeData(const Text_t *n = "REveGeoTopNodeData", const Text_t *t = "");
   virtual ~REveGeoTopNodeData() {}

   Int_t WriteCoreJson(nlohmann::json &j, Int_t rnr_offset) override;
   void SetTNode(TGeoNode* n);
   void ProcessSignal(const std::string &);
   RGeomDescription& RefDescription() {return fDesc;}

   void SetChannel(unsigned connid, int chid);
};
//-------------------------------------------------------------------
class REveGeoTopNodeViz : public REveElement
{
 private:
   REveGeoTopNodeViz(const REveGeoTopNodeViz &) = delete;
   REveGeoTopNodeViz &operator=(const REveGeoTopNodeViz &) = delete;

   REveGeoTopNodeData* fGeoData{nullptr};

 public:
   REveGeoTopNodeViz(const Text_t *n = "REveGeoTopNodeViz", const Text_t *t = "");
   void SetGeoData(REveGeoTopNodeData* d) {fGeoData = d;}
   Int_t WriteCoreJson(nlohmann::json &j, Int_t rnr_offset) override;
   void BuildRenderData() override;

   bool    RequiresExtraSelectionData() const override { return true; };
   void FillExtraSelectionData(nlohmann::json& j, const std::set<int>& secondary_idcs) const override;

   using REveElement::GetHighlightTooltip;
   std::string GetHighlightTooltip(const std::set<int>& secondary_idcs) const override;
};

} // namespace Experimental
} // namespace ROOT

#endif

