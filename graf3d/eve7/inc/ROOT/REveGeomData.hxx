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

#include "Rtypes.h"

#include <vector>
#include <string>
#include <functional>

class TGeoNode;
class TGeoManager;

// do not use namespace to avoid too long JSON

namespace ROOT {
namespace Experimental {

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

   std::vector<TGeoNode *> fNodes;   ///<! flat list of all nodes
   std::vector<REveGeomNode> fDesc;  ///< converted description, send to client

   void ScanNode(TGeoNode *node, std::vector<int> &numbers, int offset);

   int MarkVisible(bool on_screen = false);

   void ScanVisible(REveGeomScanFunc_t func);

public:
   REveGeomDescription() = default;

   void Build(TGeoManager *mgr);
};


} // namespace Experimental
} // namespace ROOT

#endif
