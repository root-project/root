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
   int id{0};               ///< node id, index in array
   int sortid{0};           ///< place in sorted array, to check cuts
   std::vector<int> chlds;  ///< list of childs id
   std::string name;        ///< node name
   std::vector<float> matr; ///< matrix for the node, can have reduced number of elements
   double vol{0};           ///<! volume estimation
   int nfaces{0};           ///<! number of shape faces
   bool vis{false};         ///<! visibility flags used in selection
   int visdepth{0};         ///<! how far to check daughters visibility
   int numvischld{0};       ///<! number of visible childs, if all can be jump over
   int idshift{0};          ///<! used to jump over then scan all geom hierarchy

   REveGeomNode() = default;
   REveGeomNode(int _id) : id(_id) {}
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
   };

   std::vector<TGeoNode *> fNodes;  ///<! flat list of all nodes
   std::string fDrawOptions;        ///< default draw options for client
   std::vector<REveGeomNode> fDesc; ///< converted description, send to client
   int fTopDrawNode{0};             ///<! selected top node
   std::vector<int> fSortMap;       ///<! nodes in order large -> smaller volume
   int fNSegments{0};               ///<! number of segments for cylindrical shapes
   std::vector<ShapeDescr> fShapes; ///<! shapes with created descriptions

   std::string fDrawJson;           ///<! JSON with main nodes drawn by client
   std::vector<char> fDrawBinary;   ///<! binary data for main draw nodes
   int fDrawIdCut{0};               ///<! sortid used for selection of most-significant nodes

   void PackMatrix(std::vector<float> &arr, TGeoMatrix *matr);

   void ScanNode(TGeoNode *node, std::vector<int> &numbers, int offset);

   int MarkVisible(bool on_screen = false);

   void ScanVisible(REveGeomScanFunc_t func);

   ShapeDescr &FindShapeDescr(TGeoShape *shape);

   ShapeDescr &MakeShapeDescr(TGeoShape *shape);

   void CopyMaterialProperties(TGeoVolume *col, REveGeomVisisble &item);

public:
   REveGeomDescription() = default;

   void Build(TGeoManager *mgr);

   bool CollectVisibles(int maxnumfaces);

   bool HasDrawData() const { return (fDrawJson.length() > 0) && (fDrawBinary.size() > 0) && (fDrawIdCut > 0); }
   const std::string &GetDrawJson() const { return fDrawJson; }
   const std::vector<char> &GetDrawBinary() const { return fDrawBinary; }

   int SearchVisibles(const std::string &find, std::string &json, std::vector<char> &binary);

   void SelectVolume(TGeoVolume *);

   void SelectNode(TGeoNode *);

   void SetNSegments(int n = 0) { fNSegments = n; }
   int GetNSegments() const { return fNSegments; }

   void SetDrawOptions(const std::string &opt = "") { fDrawOptions = opt; }

};


} // namespace Experimental
} // namespace ROOT

#endif
