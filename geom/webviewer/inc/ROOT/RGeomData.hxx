// Author: Sergey Linev, 14.12.2018

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
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

#include <ROOT/Browsable/RItem.hxx>

#include "TVirtualMutex.h"

class TGeoNode;
class TGeoManager;
class TGeoShape;
class TGeoMatrix;
class TGeoVolume;

// do not use namespace to avoid too long JSON

namespace ROOT {

class RGeomBrowserIter;

namespace Experimental {
class RLogChannel;
} // namespace Experimental

/// Log channel for Geomviewer diagnostics.
Experimental::RLogChannel &RGeomLog();

/** Base description of geometry node, required only to build hierarchy */

class RGeomNodeBase {
public:
   int id{0};               ///< node id, index in array
   std::string name;        ///< node name
   std::vector<int> chlds;  ///< list of childs id
   int vis{0};              ///< visibility flag, 0 - off, 1 - only when level==0, 99 - always
   bool nochlds{false};     ///< how far in hierarchy depth should be scanned

   std::string color;       ///< rgb code in hex format
   std::string material;    ///< name of the material
   int sortid{0};           ///<! place in sorted array, to check cuts, or id of original node when used search structures

   RGeomNodeBase(int _id = 0) : id(_id) {}

   bool IsVisible() const { return vis > 0; }

   /** Returns argument for regexp */
   const char *GetArg(int kind)
   {
      if (kind == 1) return color.c_str();
      if (kind == 2) return material.c_str();
      return name.c_str();
   }
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

/** \class RGeoItem
\ingroup rbrowser
\brief Representation of single item in the geometry browser
*/

class RGeoItem : public Browsable::RItem {

protected:
   // this is part for browser, visible for I/O
   int id{0};              ///< node id
   std::string color;      ///< color
   std::string material;   ///< material
   int vis{0};             ///< visibility of logical node
   int pvis{0};            ///< visibility of physical node
   bool top{false};        ///< indicates if node selected as top

public:

   /** Default constructor */
   RGeoItem() = default;

   RGeoItem(const std::string &_name, int _nchilds, int _nodeid, const std::string &_color,
         const std::string &_material = "", int _vis = 0, int _pvis = 0) :
         Browsable::RItem(_name, _nchilds), id(_nodeid), color(_color), material(_material), vis(_vis), pvis(_pvis) {
   }

   // should be here, one needs virtual table for correct streaming of RRootBrowserReply
   ~RGeoItem() override = default;

   void SetTop(bool on = true) { top = on; }
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
   ~RGeomRawRenderInfo() override = default;
};

/** Render info with shape itself - client can produce shape better */
class RGeomShapeRenderInfo : public RGeomRenderInfo  {
public:
   TGeoShape *shape{nullptr}; ///< original shape - can be much less than binary data
   ~RGeomShapeRenderInfo() override = default;
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


/** Node information including rendering data */
class RGeomNodeInfo {
public:
   std::vector<std::string> path;  ///< full path to node
   std::string node_type;  ///< node class name
   std::string node_name;  ///< node name
   std::string shape_type; ///< shape type (if any)
   std::string shape_name; ///< shape class name (if any)

   RGeomRenderInfo *ri{nullptr};  ///< rendering information (if applicable)
};

/** Custom settings for physical Node visibility */
class RGeomNodeVisibility {
public:
   std::vector<int> stack;        ///< path to the node
   bool visible{false};           ///< visible flag
   RGeomNodeVisibility(const std::vector<int> &_stack, bool _visible) : stack(_stack), visible(_visible) {}
};

using RGeomScanFunc_t = std::function<bool(RGeomNode &, std::vector<int> &, bool, int)>;

using RGeomSignalFunc_t = std::function<void(const std::string &)>;

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
   std::vector<RGeomNode> fDesc;    ///<! converted description, send to client
   std::vector<RGeomNodeVisibility> fVisibility; ///<! custom visibility flags for physical nodes

   TGeoVolume *fDrawVolume{nullptr};///<! select volume independent from TGeoManager
   std::vector<int> fSelectedStack; ///<! selected branch of geometry by stack

   std::vector<int> fHighlightedStack; ///<! highlighted element by stack
   std::vector<int> fClickedStack;     ///<! clicked element by stack

   std::vector<int> fSortMap;       ///<! nodes in order large -> smaller volume
   std::vector<ShapeDescr> fShapes; ///<! shapes with created descriptions

   std::string fSearch;             ///<! search string in hierarchy
   std::string fSearchJson;         ///<! drawing json for search
   std::string fDrawJson;           ///<! JSON with main nodes drawn by client
   int fDrawIdCut{0};               ///<! sortid used for selection of most-significant nodes
   int fActualLevel{0};             ///<! level can be reduced when selecting nodes
   bool fPreferredOffline{false};   ///<! indicates that full description should be provided to client
   int fJsonComp{0};                ///<! default JSON compression
   std::string fActiveItemName;     ///<! name of item which should be activated in hierarchy

   RGeomConfig fCfg;                ///<! configuration parameter editable from GUI

   TVirtualMutex *fMutex{nullptr};  ///<! external mutex used to protect all data

   std::vector<std::pair<const void *, RGeomSignalFunc_t>> fSignals; ///<! registered signals

   void PackMatrix(std::vector<float> &arr, TGeoMatrix *matr);

   int MarkVisible(bool on_screen = false);

   void ProduceIdShifts();

   int ScanNodes(bool only_visible, int maxlvl, RGeomScanFunc_t func);

   void ResetRndrInfos();

   ShapeDescr &FindShapeDescr(TGeoShape *shape);

   ShapeDescr &MakeShapeDescr(TGeoShape *shape);

   int GetUsedNSegments(int min = 20);

   int CountShapeFaces(TGeoShape *shape);

   void CopyMaterialProperties(TGeoVolume *vol, RGeomNode &node);

   void CollectNodes(RGeomDrawing &drawing, bool all_nodes = false);

   std::string MakeDrawingJson(RGeomDrawing &drawing, bool has_shapes = false);

   void ClearDescription();

   void BuildDescription(TGeoNode *topnode, TGeoVolume *topvolume);

   TGeoVolume *GetVolume(int nodeid);

   int IsPhysNodeVisible(const std::vector<int> &stack);

public:
   RGeomDescription() = default;

   void AddSignalHandler(const void *handler, RGeomSignalFunc_t func);

   void RemoveSignalHandler(const void *handler);

   void IssueSignal(const void *handler, const std::string &kind);

   /** Set mutex, it must be recursive one */
   void SetMutex(TVirtualMutex *mutex) { fMutex = mutex; }
   /** Return currently used mutex */
   TVirtualMutex *GetMutex() const { return fMutex; }

   /** Set maximal number of nodes which should be selected for drawing */
   void SetMaxVisNodes(int cnt) { TLockGuard lock(fMutex); fCfg.maxnumnodes = cnt; }
   /** Returns maximal visible number of nodes, ignored when non-positive */
   int GetMaxVisNodes() const { TLockGuard lock(fMutex); return fCfg.maxnumnodes; }

   /** Set maximal number of faces which should be selected for drawing */
   void SetMaxVisFaces(int cnt) { TLockGuard lock(fMutex); fCfg.maxnumfaces = cnt; }
   /** Returns maximal visible number of faces, ignored when non-positive */
   int GetMaxVisFaces() const { TLockGuard lock(fMutex); return fCfg.maxnumfaces; }

   /** Set maximal visible level */
   void SetVisLevel(int lvl = 3) { TLockGuard lock(fMutex); fCfg.vislevel = lvl; }
   /** Returns maximal visible level */
   int GetVisLevel() const { TLockGuard lock(fMutex); return fCfg.vislevel; }

   /** Set draw options as string for JSROOT TGeoPainter */
   void SetTopVisible(bool on = true) { TLockGuard lock(fMutex); fCfg.showtop = on; }
   /** Returns draw options, used for JSROOT TGeoPainter */
   bool GetTopVisible() const { TLockGuard lock(fMutex); return fCfg.showtop; }

   /** Instruct to build binary 3D model already on the server (true) or send TGeoShape as is to client, which can build model itself */
   void SetBuildShapes(int lvl = 1) { TLockGuard lock(fMutex); fCfg.build_shapes = lvl; }
   /** Returns true if binary 3D model build already by C++ server (default) */
   int IsBuildShapes() const { TLockGuard lock(fMutex); return fCfg.build_shapes; }

   /** Set number of segments for cylindrical shapes, if 0 - default value will be used */
   void SetNSegments(int n = 0) { TLockGuard lock(fMutex); fCfg.nsegm = n; }
   /** Return of segments for cylindrical shapes, if 0 - default value will be used */
   int GetNSegments() const { TLockGuard lock(fMutex); return fCfg.nsegm; }

   /** Set draw options as string for JSROOT TGeoPainter */
   void SetDrawOptions(const std::string &opt = "") { TLockGuard lock(fMutex); fCfg.drawopt = opt; }
   /** Returns draw options, used for JSROOT TGeoPainter */
   std::string GetDrawOptions() const { TLockGuard lock(fMutex); return fCfg.drawopt; }

   /** Set JSON compression level for data transfer */
   void SetJsonComp(int comp = 0) { TLockGuard lock(fMutex); fJsonComp = comp; }
   /** Returns JSON compression level for data transfer */
   int GetJsonComp() const  { TLockGuard lock(fMutex); return fJsonComp; }

   /** Set preference of offline operations.
    * Server provides more info to client from the begin on to avoid communication */
   void SetPreferredOffline(bool on) { TLockGuard lock(fMutex); fPreferredOffline = on; }
   /** Is offline operations preferred.
    * After get full description, client can do most operations without extra requests */
   bool IsPreferredOffline() const { TLockGuard lock(fMutex); return fPreferredOffline; }

   /** Get top node path */
   const std::vector<int>& GetSelectedStack() const { return fSelectedStack; }

   void Build(TGeoManager *mgr, const std::string &volname = "");

   void Build(TGeoVolume *vol);

   /** Number of unique nodes in the geometry */
   int GetNumNodes() const { TLockGuard lock(fMutex); return fDesc.size(); }

   bool IsBuild() const { return GetNumNodes() > 0; }

   std::string ProduceJson(bool all_nodes = false);

   bool IsPrincipalEndNode(int nodeid);

   std::string ProcessBrowserRequest(const std::string &req = "");

   bool HasDrawData() const;
   void ProduceDrawData();
   void ProduceSearchData();
   std::string GetDrawJson() const { TLockGuard lock(fMutex); return fDrawJson; }
   std::string GetSearch() const { TLockGuard lock(fMutex); return fSearch; }
   std::string GetSearchJson() const { TLockGuard lock(fMutex); return fSearchJson; }
   void ClearDrawData();

   void ClearCache();

   int SearchVisibles(const std::string &find, std::string &hjson, std::string &json);

   int FindNodeId(const std::vector<int> &stack);

   std::string ProduceModifyReply(int nodeid);

   std::vector<int> MakeStackByIds(const std::vector<int> &ids);

   std::vector<int> MakeIdsByStack(const std::vector<int> &stack);

   std::vector<int> MakeStackByPath(const std::vector<std::string> &path);

   std::vector<std::string> MakePathByStack(const std::vector<int> &stack);

   bool ProduceDrawingFor(int nodeid, std::string &json, bool check_volume = false);

   bool SetHighlightedItem(const std::vector<int> &stack)
   {
      TLockGuard lock(fMutex);
      bool changed = fHighlightedStack != stack;
      fHighlightedStack = stack;
      return changed;
   }

   std::vector<int> GetHighlightedItem() const
   {
      TLockGuard lock(fMutex);
      return fHighlightedStack;
   }

   bool SetClickedItem(const std::vector<int> &stack)
   {
      TLockGuard lock(fMutex);
      bool changed = fClickedStack != stack;
      fClickedStack = stack;
      return changed;
   }

   std::vector<int> GetClickedItem() const
   {
      TLockGuard lock(fMutex);
      return fClickedStack;
   }

   bool SetActiveItem(const std::string &itemname)
   {
      TLockGuard lock(fMutex);
      bool changed = (fActiveItemName != itemname);
      fActiveItemName = itemname;
      return changed;
   }

   std::string GetActiveItem() const
   {
      TLockGuard lock(fMutex);
      return fActiveItemName;
   }

   bool ChangeConfiguration(const std::string &json);

   std::unique_ptr<RGeomNodeInfo> MakeNodeInfo(const std::vector<int> &stack);

   bool ChangeNodeVisibility(const std::vector<std::string> &path, bool on);

   bool SelectTop(const std::vector<std::string> &path);

   bool SetPhysNodeVisibility(const std::vector<std::string> &path, bool on = true);

   bool SetPhysNodeVisibility(const std::string &path, bool on = true);

   bool ClearPhysNodeVisibility(const std::vector<std::string> &path);

   bool ClearAllPhysVisibility();

   bool SetSearch(const std::string &query, const std::string &json);

   void SavePrimitive(std::ostream &fs, const std::string &name);
};

} // namespace ROOT

#endif
