// @(#)root/eve7:$Id$
// Author: Sergey Linev, 14.12.2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_REveGeomData
#define ROOT7_REveGeomData

#include <ROOT/REveRenderData.hxx>

#include <vector>
#include <string>
#include <functional>
#include <memory>

class TGeoNode;
class TGeoManager;
class TGeoShape;
class TGeoMatrix;
class TGeoVolume;

// do not use namespace to avoid too long JSON

namespace ROOT {
namespace Experimental {

class REveRenderData;

class REveGeomNode {
public:

   enum EVis { vis_off = 0, vis_this = 1, vis_chlds = 2, vis_lvl1 = 4 };

   int id{0};               ///< node id, index in array
   int sortid{0};           ///< place in sorted array, to check cuts
   std::vector<int> chlds;  ///< list of childs id
   std::string name;        ///< node name
   std::vector<float> matr; ///< matrix for the node, can have reduced number of elements
   int vis{vis_off};        ///< visibility flag, 0 - off, 1 - volume, 2 - daughters, 4 - single lvl
   double vol{0};           ///<! volume estimation
   int nfaces{0};           ///<! number of shape faces
   int numvischld{0};       ///<! number of visible childs, if all can be jump over
   int idshift{0};          ///<! used to jump over then scan all geom hierarchy

   REveGeomNode() = default;
   REveGeomNode(int _id) : id(_id) {}

   /** True when there is shape and it can be displayed */
   bool CanDisplay() const { return (vol > 0.) && (nfaces > 0); }

   bool IsVisible() const { return vis & vis_this; }

   int GetVisDepth() const { return (vis & vis_chlds) ? 999999 : ((vis & vis_lvl1) ? 1 : 0); }
};

class REveShapeRenderInfo {
public:
   // render data, equivalent of REveElement::WriteCoreJson
   int rnr_offset{-1};     ///< rnr_offset;
   std::string rnr_func;  ///< fRenderData->GetRnrFunc();
   int vert_size{0};      ///< fRenderData->SizeV();
   int norm_size{0};      ///< fRenderData->SizeN();
   int index_size{0};     ///< fRenderData->SizeI();
   // int trans_size{0};     ///< fRenderData->SizeT(); not used in GeomViewer
};

/** REveGeomVisisble contains description of visible node
 * It is path to the node plus reference to shape rendering data
 */

class REveGeomVisisble {
public:
   int nodeid{0};                    ///< selected node id,
   std::vector<int> stack;           ///< path to the node, index in list of childs
   std::string color;                ///< color in rgb format
   double opacity{1};                ///< opacity
   REveShapeRenderInfo *ri{nullptr}; ///< render information for the shape, can be same for different nodes

   REveGeomVisisble() = default;
   REveGeomVisisble(int id, const std::vector<int> &_stack) : nodeid(id), stack(_stack) {}
};

using REveGeomScanFunc_t = std::function<bool(REveGeomNode&, std::vector<int>&)>;

class REveGeomDescription {

   class ShapeDescr {
   public:
      int id{0};                                   ///<! sequential id
      TGeoShape *fShape{nullptr};                  ///<! original shape
      int nfaces{0};                               ///<! number of faces in render data
      std::unique_ptr<REveRenderData> fRenderData; ///<! binary render data
      REveShapeRenderInfo fRenderInfo;             ///<! render information for client
      ShapeDescr(TGeoShape *s) : fShape(s) {}

      /// Provide render info for visible item
      REveShapeRenderInfo *rndr_info() { return (nfaces>0) && (fRenderInfo.rnr_offset>=0) ? &fRenderInfo : nullptr; }
   };

   std::vector<TGeoNode *> fNodes;  ///<! flat list of all nodes
   std::string fDrawOptions;        ///< default draw options for client
   std::vector<REveGeomNode> fDesc; ///< converted description, send to client
   int fTopDrawNode{0};             ///<! selected top node
   std::vector<int> fSortMap;       ///<! nodes in order large -> smaller volume
   int fNSegments{0};               ///<! number of segments for cylindrical shapes
   std::vector<ShapeDescr> fShapes; ///<! shapes with created descriptions
   std::vector<REveRenderData*> fRndrShapes; ///<! list of shapes which should be packet into binary
   int fRndrOffest{0};              ///<! current render offset

   std::string fDrawJson;           ///<! JSON with main nodes drawn by client
   std::vector<char> fDrawBinary;   ///<! binary data for main draw nodes
   int fDrawIdCut{0};               ///<! sortid used for selection of most-significant nodes
   int fFacesLimit{0};              ///<! maximal number of faces to be selected for drawing
   int fNodesLimit{0};              ///<! maximal number of nodes to be selected for drawing

   void PackMatrix(std::vector<float> &arr, TGeoMatrix *matr);

   void ScanNode(TGeoNode *node, std::vector<int> &numbers, int offset);

   int MarkVisible(bool on_screen = false);

   void ScanVisible(REveGeomScanFunc_t func);

   void ResetRndrInfos();

   ShapeDescr &FindShapeDescr(TGeoShape *shape);

   ShapeDescr &MakeShapeDescr(TGeoShape *shape, bool acc_rndr = false);

   void BuildRndrBinary(std::vector<char> &buf);

   void CopyMaterialProperties(TGeoVolume *col, REveGeomVisisble &item);

public:
   REveGeomDescription() = default;

   void Build(TGeoManager *mgr);

   /** Number of unique nodes in the geometry */
   int GetNumNodes() const { return fDesc.size(); }

   /** Set maximal number of nodes which should be selected for drawing */
   void SetMaxVisNodes(int cnt) { fNodesLimit = cnt; }

   /** Returns maximal visible number of nodes, ignored when non-positive */
   int GetMaxVisNodes() const { return fNodesLimit; }

   /** Set maximal number of faces which should be selected for drawing */
   void SetMaxVisFaces(int cnt) { fFacesLimit = cnt; }

   /** Returns maximal visible number of faces, ignored when non-positive */
   int GetMaxVisFaces() const { return fFacesLimit; }

   bool CollectVisibles();

   bool IsPrincipalEndNode(int nodeid);

   bool HasDrawData() const { return (fDrawJson.length() > 0) && (fDrawBinary.size() > 0) && (fDrawIdCut > 0); }
   const std::string &GetDrawJson() const { return fDrawJson; }
   const std::vector<char> &GetDrawBinary() const { return fDrawBinary; }
   void ClearRawData();

   int SearchVisibles(const std::string &find, std::string &json, std::vector<char> &binary);

   int FindNodeId(const std::vector<int> &stack);

   std::string ProduceModifyReply(int nodeid);

   bool ProduceDrawingFor(int nodeid, std::string &json, std::vector<char> &binary, bool check_volume = false);

   bool ChangeNodeVisibility(int nodeid, bool selected);

   void SelectVolume(TGeoVolume *);

   void SelectNode(TGeoNode *);

   void SetNSegments(int n = 0) { fNSegments = n; }
   int GetNSegments() const { return fNSegments; }

   void SetDrawOptions(const std::string &opt = "") { fDrawOptions = opt; }

};


} // namespace Experimental
} // namespace ROOT

#endif
