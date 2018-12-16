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

// do not use namespace to avoid too long JSON

namespace ROOT {
namespace Experimental {

class REveRenderData;

class REveGeomNode {
public:
   int id{0};               ///< node id, index in array
   std::vector<int> chlds;  ///< list of childs id
   std::string name;        ///< node name
   double vol{0};           ///<! volume estimation
   int nfaces{0};           ///<! number of shape faces
   bool vis{false};         ///<! visibility flags used in selection
   int visdepth{0};         ///<! how far to check daughters visibility
   int numvischld{0};       ///<! number of visible childs, if all can be jump over
   int idshift{0};          ///<! used to jump over then scan all geom hierarchy

   REveGeomNode() = default;
   REveGeomNode(int _id) : id(_id) {}
};

using REveGeomScanFunc_t = std::function<bool(REveGeomNode&, std::vector<int>&)>;

class REveGeomDescription {

   class ShapeDescr {
   public:
      int id{0};
      TGeoShape *fShape{nullptr};
      std::unique_ptr<REveRenderData> fRenderData;
      ShapeDescr(TGeoShape *s) : fShape(s) {}
   };


   std::vector<TGeoNode *> fNodes;   ///<! flat list of all nodes
   std::vector<REveGeomNode> fDesc;  ///< converted description, send to client
   std::vector<int> fSortMap;        ///<! nodes in order large -> smaller volume
   std::vector<ShapeDescr> fShapes;  ///<! shapes with created descriptions

   void ScanNode(TGeoNode *node, std::vector<int> &numbers, int offset);

   int MarkVisible(bool on_screen = false);

   void ScanVisible(REveGeomScanFunc_t func);

   void CollectVisibles(int maxnumfaces);

   ShapeDescr &FindShapeDescr(TGeoShape *s);

public:
   REveGeomDescription() = default;

   void Build(TGeoManager *mgr);
};


} // namespace Experimental
} // namespace ROOT

#endif
