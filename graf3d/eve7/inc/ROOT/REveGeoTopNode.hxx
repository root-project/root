
#ifndef ROOT7_REveGeoTopNode
#define ROOT7_REveGeoTopNode

#include <ROOT/REveElement.hxx>
#include <ROOT/RGeomData.hxx>
#include <ROOT/RGeomHierarchy.hxx>
#include "ROOT/REveSecondarySelectable.hxx"

class TGeoNode;
class TGeoIterator;


namespace ROOT {
namespace Experimental {

class REveGeoTopNodeData;
/////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
//    REveGeomDescription
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

class REveGeomDescription : public RGeomDescription {
  static TGeoManager* s_geoManager;
protected:
   std::vector<RGeomNodeVisibility> fVisibilitySelf;
   std::vector<RGeomNodeVisibility> fVisibilityRec;

   virtual void RefineGeoItem(ROOT::RGeoItem &item, const std::vector<int> &stack) override;
   virtual bool IsFullModelStreamedAtOnce() const override { return false; }

   class Apex {
      std::vector<std::string> fPath;
      TGeoNode *fNode{nullptr};

   public:
      void SetFromPath(std::vector<std::string> absPath);
      TGeoNode *LocateNodeWithPath(const std::vector<std::string> &path) const;

      TGeoNode *GetNode() { return fNode; }
      std::string GetFlatPath() const;
      const std::vector<std::string>& GetPath() const { return fPath; }
      std::vector<int> GetIndexStack() const;
   };

   Apex fApex;

public:
   REveGeomDescription() : RGeomDescription() {};
   virtual ~REveGeomDescription() {};

   enum ERnrFlags {
      kRnrNone = 0,
      kRnrSelf = 1,
      kRnrChildren = 2
   };

   bool ChangeEveVisibility(const std::vector<int> &stack, ERnrFlags rnrFlag, bool on);
   std::vector<int> GetIndexStack() { return fApex.GetIndexStack(); }
   const std::vector<std::string>& GetApexPath() const { return fApex.GetPath();}
   void InitPath(const std::vector<std::string>& path);
   TGeoNode* GetApexNode() { return fApex.GetNode(); }
   TGeoNode* LocateNodeWithPath(const std::vector<std::string> &path) { return fApex.LocateNodeWithPath(path); }

   bool GetVisiblityForStack(const std::vector<int>& stack);

   void ImportFile(const char* filePath);
   static TGeoManager* GetGeoManager();
};

/////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// REveGeomHierarchy
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

class REveGeomHierarchy : public RGeomHierarchy
{
   REveGeoTopNodeData* fReceiver{nullptr};
protected:
   virtual void WebWindowCallback(unsigned connid, const std::string &kind) override;

public:
   REveGeomHierarchy(REveGeomDescription &desc, bool th) :
   RGeomHierarchy(desc, th){};

   void SetReceiver(REveGeoTopNodeData* data) { fReceiver = data; }
   virtual ~REveGeomHierarchy(){};
};

/////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// REveGeoTopNodeData
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

class REveGeoTopNodeData : public REveElement, public REveAuntAsList {
   friend class REveGeoTopNodeViz;
private:
   std::shared_ptr<REveGeomHierarchy> fWebHierarchy; ///<! web handle for hierarchy part

protected:
   REveGeoTopNodeData(const REveGeoTopNodeData &) = delete;
   REveGeoTopNodeData &operator=(const REveGeoTopNodeData &) = delete;

   REveGeomDescription fDesc;

public:
   REveGeoTopNodeData(const char* fileName);
   virtual ~REveGeoTopNodeData() {}

   Int_t WriteCoreJson(nlohmann::json &j, Int_t rnr_offset) override;
   void ProcessSignal(const std::string &);
   REveGeomDescription& RefDescription() {return fDesc;}

   void SetChannel(unsigned connid, int chid);
   void VisibilityChanged(bool on, REveGeomDescription::ERnrFlags flag, const std::vector<int>& path);
   void InitPath(const std::string& path);
};

/////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// REveGeoTopNodeViz
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
class REveGeoTopNodeViz : public REveElement,
                          public REveSecondarySelectable
{
public:
   enum EMode {
      kModeNone,
      kModeVisLevel,
      kModeLeafOnly,
      kModeMixed
   };

private:
  struct BShape {
      TGeoShape *shape;
      std::vector<int> indices;
      std::vector<float> vertices;
   };

   struct BNode {
      TGeoNode *node;
      int shapeId;
      int nodeId;
      int color;
      float trans[16];
      bool visible{true};
   };

   REveGeoTopNodeViz(const REveGeoTopNodeViz &) = delete;
   REveGeoTopNodeViz &operator=(const REveGeoTopNodeViz &) = delete;

   REveGeoTopNodeData *fGeoData{nullptr};
   std::vector<BNode> fNodes;
   std::vector<BShape> fShapes;
   EMode fMode{kModeVisLevel};

   void CollectNodes(TGeoVolume *volume, std::vector<BNode> &bnl, std::vector<BShape> &browsables);
   void CollectShapes(TGeoNode *node, std::set<TGeoShape *> &shapes, std::vector<BShape> &browsables);
   bool AcceptNode(TGeoIterator& it, bool skip = true) const;

public:
   REveGeoTopNodeViz(const Text_t *n = "REveGeoTopNodeViz", const Text_t *t = "");
   void SetGeoData(REveGeoTopNodeData *d, bool rebuild = true);
   Int_t WriteCoreJson(nlohmann::json &j, Int_t rnr_offset) override;
   void BuildRenderData() override;
   void GetIndicesFromBrowserStack(const std::vector<int> &stack, std::set<int>& outStack);

   void SetVisLevel(int);
   void VisibilityChanged(bool on,  REveGeomDescription::ERnrFlags flag, const std::vector<int>& path);
   void BuildDesc();

   EMode GetVizMode() const { return fMode; }
   void SetVizMode(EMode mode);

   using REveElement::GetHighlightTooltip;
   std::string GetHighlightTooltip(const std::set<int>& secondary_idcs) const override;
};

} // namespace Experimental
} // namespace ROOT

#endif

