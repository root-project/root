// @(#)root/geom/webviewer:$Id$
// Author: Sergey Linev, 14.12.2018

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RGeomData
#define ROOT7_RGeomData

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

class RGeomBrowserIter;
class RLogChannel;

/** Base description of geometry node, required only to build hierarchy */

class RGeomNodeBase {
public:
   int id{0};               ///< node id, index in array
   std::string name;        ///< node name
   std::vector<int> chlds;  ///< list of childs id
   int vis{0};              ///< visibility flag, 0 - off, 1 - only when level==0, 99 - always
   bool nochlds{false};     ///< how far in hierarchy depth should be scanned

   std::string color;       ///< rgb code without rgb() prefix
   int sortid{0};           ///<! place in sorted array, to check cuts, or id of original node when used search structures

   RGeomNodeBase(int _id = 0) : id(_id) {}

   bool IsVisible() const { return vis > 0; }
};

/** Full node description including matrices and other attributes */

class RGeomNode : public RGeomNodeBase  {
public:
   std::vector<float> matr; ///< matrix for the node, can have reduced number of elements
   double vol{0};           ///<! volume estimation
   int nfaces{0};           ///<! number of shape faces
   int idshift{-1};         ///<! used to jump over then scan all geom hierarchy
   bool useflag{false};     ///<! extra flag, used for selection
   float opacity{1.};       ///<! opacity of the color

   RGeomNode(int _id = 0) : RGeomNodeBase(_id) {}

   /** True when there is shape and it can be displayed */
   bool CanDisplay() const { return (vol > 0.) && (nfaces > 0); }
};

/** Base class for render info block */
class RGeomRenderInfo {
public:
   /// virtual destructor required for the I/O
   virtual ~RGeomRenderInfo() = default;
};

/** Render info with raw data */
class RGeomRawRenderInfo : public RGeomRenderInfo  {
public:
   std::vector<unsigned char> raw;  ///< float vertices as raw data, JSON_base64
   std::vector<int> idx;            ///< vertex indexes, always triangles
   virtual ~RGeomRawRenderInfo() = default;
};

/** Render info with shape itself - client can produce shape better */
class RGeomShapeRenderInfo : public RGeomRenderInfo  {
public:
   TGeoShape *shape{nullptr}; ///< original shape - can be much less than binary data
   virtual ~RGeomShapeRenderInfo() = default;
};


/** RGeomVisible contains description of visible node
 * It is path to the node plus reference to shape rendering data */

class RGeomVisible {
public:
   int nodeid{0};                    ///< selected node id,
   int seqid{0};                     ///< sequence id, used for merging later
   std::vector<int> stack;           ///< path to the node, index in list of childs
   std::string color;                ///< color in rgb format
   double opacity{1};                ///< opacity
   RGeomRenderInfo *ri{nullptr};     ///< render information for the shape, can be same for different nodes

   RGeomVisible() = default;
   RGeomVisible(int _nodeid, int _seqid, const std::vector<int> &_stack) : nodeid(_nodeid), seqid(_seqid), stack(_stack) {}
};


/** Configuration parameters which can be configured on the client
 * Send as is to-from client */

class RGeomConfig {
public:
   int vislevel{0};                         ///< visible level
   int maxnumnodes{0};                      ///< maximal number of nodes
   int maxnumfaces{0};                      ///< maximal number of faces
   bool showtop{false};                     ///< show geometry top volume, off by default
   int build_shapes{1};                     ///< when shapes build on server  0 - never, 1 - TGeoComposite, 2 - plus non-cylindrical, 3 - all
   int nsegm{0};                            ///< number of segments for cylindrical shapes
   std::string drawopt;                     ///< draw options for TGeoPainter
};


/** Object with full description for drawing geometry
 * It includes list of visible items and list of nodes required to build them */

class RGeomDrawing {
public:
   RGeomConfig *cfg{nullptr};            ///< current configurations
   int numnodes{0};                      ///< total number of nodes in description
   std::vector<RGeomNode*> nodes;        ///< all used nodes to display visible items and not known for client
   std::vector<RGeomVisible> visibles;   ///< all visible items
};


/** Request object send from client for different operations */
class RGeomRequest {
public:
   std::string oper;  ///< operation like HIGHL or HOVER
   std::vector<std::string> path;  ///< path parameter, used with HOVER
   std::vector<int> stack; ///< stack parameter, used with HIGHL
};

class RGeomNodeInfo {
public:
   std::vector<std::string> path;  ///< full path to node
   std::string node_type;  ///< node class name
   std::string node_name;  ///< node name
   std::string shape_type; ///< shape type (if any)
   std::string shape_name; ///< shape class name (if any)

   RGeomRenderInfo *ri{nullptr};  ///< rendering information (if applicable)
};

using RGeomScanFunc_t = std::function<bool(RGeomNode &, std::vector<int> &, bool, int)>;


class RGeomDescription {

   friend class RGeomBrowserIter;

   class ShapeDescr {
   public:
      int id{0};                                   ///<! sequential id
      TGeoShape *fShape{nullptr};                  ///<! original shape
      int nfaces{0};                               ///<! number of faces in render data
      RGeomRawRenderInfo fRawInfo;                 ///<! raw render info
      RGeomShapeRenderInfo fShapeInfo;             ///<! shape itself as info
      ShapeDescr(TGeoShape *s) : fShape(s) {}

      bool has_shape() const { return nfaces == 1; }
      bool has_raw() const { return nfaces > 1; }

      /// Provide render info for visible item
      RGeomRenderInfo *rndr_info()
      {
         if (has_shape()) return &fShapeInfo;
         if (has_raw()) return &fRawInfo;
         return nullptr;
      }

      void reset()
      {
         nfaces = 0;
         fShapeInfo.shape = nullptr;
         fRawInfo.raw.clear();
      }
   };

   std::vector<TGeoNode *> fNodes;  ///<! flat list of all nodes
   std::vector<RGeomNode> fDesc; ///< converted description, send to client
   TGeoVolume *fDrawVolume{nullptr}; ///<! select volume independent from TGeoMaanger

   std::vector<int> fSortMap;       ///<! nodes in order large -> smaller volume
   std::vector<ShapeDescr> fShapes; ///<! shapes with created descriptions

   std::string fDrawJson;           ///<! JSON with main nodes drawn by client
   int fDrawIdCut{0};               ///<! sortid used for selection of most-significant nodes
   int fActualLevel{0};             ///<! level can be reduced when selecting nodes
   bool fPreferredOffline{false};   ///<! indicates that full description should be provided to client
   int fJsonComp{0};                ///<! default JSON compression

   RGeomConfig fCfg;                ///<! configuration parameter editable from GUI

   void PackMatrix(std::vector<float> &arr, TGeoMatrix *matr);

   int MarkVisible(bool on_screen = false);

   void ProduceIdShifts();

   int ScanNodes(bool only_visible, int maxlvl, RGeomScanFunc_t func);

   void ResetRndrInfos();

   ShapeDescr &FindShapeDescr(TGeoShape *shape);

   ShapeDescr &MakeShapeDescr(TGeoShape *shape);

   void CopyMaterialProperties(TGeoVolume *vol, RGeomNode &node);

   void CollectNodes(RGeomDrawing &drawing);

   std::string MakeDrawingJson(RGeomDrawing &drawing, bool has_shapes = false);

   void ClearDescription();

   void BuildDescription(TGeoNode *topnode, TGeoVolume *topvolume);

   TGeoVolume *GetVolume(int nodeid);

public:
   RGeomDescription() = default;

   void Build(TGeoManager *mgr, const std::string &volname = "");

   void Build(TGeoVolume *vol);

   /** Number of unique nodes in the geometry */
   int GetNumNodes() const { return fDesc.size(); }

   bool IsBuild() const { return GetNumNodes() > 0; }

   /** Set maximal number of nodes which should be selected for drawing */
   void SetMaxVisNodes(int cnt) { fCfg.maxnumnodes = cnt; }

   /** Returns maximal visible number of nodes, ignored when non-positive */
   int GetMaxVisNodes() const { return fCfg.maxnumnodes; }

   /** Set maximal number of faces which should be selected for drawing */
   void SetMaxVisFaces(int cnt) { fCfg.maxnumfaces = cnt; }

   /** Returns maximal visible number of faces, ignored when non-positive */
   int GetMaxVisFaces() const { return fCfg.maxnumfaces; }

   /** Set maximal visible level */
   void SetVisLevel(int lvl = 3) { fCfg.vislevel = lvl; }

   /** Returns maximal visible level */
   int GetVisLevel() const { return fCfg.vislevel; }

   /** Set preference of offline operations.
    * Server provides more info to client from the begin on to avoid communication */
   void SetPreferredOffline(bool on) { fPreferredOffline = on; }

   /** Is offline operations preferred.
    * After get full description, client can do most operations without extra requests */
   bool IsPreferredOffline() const { return fPreferredOffline; }

   std::string ProduceJson();

   bool IsPrincipalEndNode(int nodeid);

   std::string ProcessBrowserRequest(const std::string &req = "");

   bool HasDrawData() const { return (fDrawJson.length() > 0) && (fDrawIdCut > 0); }
   void ProduceDrawData();
   const std::string &GetDrawJson() const { return fDrawJson; }
   void ClearDrawData();

   int SearchVisibles(const std::string &find, std::string &hjson, std::string &json);

   int FindNodeId(const std::vector<int> &stack);

   std::string ProduceModifyReply(int nodeid);

   std::vector<int> MakeStackByIds(const std::vector<int> &ids);

   std::vector<int> MakeIdsByStack(const std::vector<int> &stack);

   std::vector<int> MakeStackByPath(const std::vector<std::string> &path);

   std::vector<std::string> MakePathByStack(const std::vector<int> &stack);

   bool ProduceDrawingFor(int nodeid, std::string &json, bool check_volume = false);

   bool ChangeNodeVisibility(int nodeid, bool selected);

   /** Set number of segments for cylindrical shapes, if 0 - default value will be used */
   void SetNSegments(int n = 0) { fCfg.nsegm = n; }
   /** Return of segments for cylindrical shapes, if 0 - default value will be used */
   int GetNSegments() const { return fCfg.nsegm; }

   /** Set JSON compression level for data transfer */
   void SetJsonComp(int comp = 0) { fJsonComp = comp; }
   /** Returns JSON compression level for data transfer */
   int GetJsonComp() const  { return fJsonComp; }

   /** Set draw options as string for JSROOT TGeoPainter */
   void SetDrawOptions(const std::string &opt = "") { fCfg.drawopt = opt; }
   /** Returns draw options, used for JSROOT TGeoPainter */
   std::string GetDrawOptions() const { return fCfg.drawopt; }

   /** Set draw options as string for JSROOT TGeoPainter */
   void SetTopVisible(bool on = true) { fCfg.showtop = on; }
   /** Returns draw options, used for JSROOT TGeoPainter */
   bool GetTopVisible() const { return fCfg.showtop; }

   /** Instruct to build binary 3D model already on the server (true) or send TGeoShape as is to client, which can build model itself */
   void SetBuildShapes(int lvl = 1) { fCfg.build_shapes = lvl; }
   /** Returns true if binary 3D model build already by C++ server (default) */
   int IsBuildShapes() const { return fCfg.build_shapes; }

   bool ChangeConfiguration(const std::string &json);

   std::unique_ptr<RGeomNodeInfo> MakeNodeInfo(const std::vector<std::string> &path);
};


/// Log channel for Eve diagnostics.
RLogChannel &RGeomLog();


} // namespace Experimental
} // namespace ROOT

#endif
