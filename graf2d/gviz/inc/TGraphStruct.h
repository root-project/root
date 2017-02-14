// @(#)root/hist:$Id$
// Author: Olivier Couet 13/07/09

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGraphStruct
#define ROOT_TGraphStruct

#include "TObject.h"
#include "TGraphEdge.h"
#include "TGraphNode.h"
#include "TList.h"

struct GVizAgraph_t;
struct GVC_s;


class TGraphStruct : public TObject {

protected:

   GVizAgraph_t *fGVGraph; ///< Graphviz graph
   GVC_s    *fGVC;         ///< Graphviz context
   TList    *fNodes;       ///< List of nodes in this TGraphStruct
   TList    *fEdges;       ///< List of edges in this TGraphStruct
   Double_t  fMargin;      ///< Margin around the graph (in dots)

public:

   TGraphStruct();
   virtual ~TGraphStruct();

   void         AddEdge(TGraphEdge *edge);
   void         AddNode(TGraphNode *node);
   TGraphEdge  *AddEdge(TGraphNode *n1, TGraphNode *n2);
   TGraphNode  *AddNode(const char *name, const char *title="");
   void         Draw(Option_t *option="");
   void         DumpAsDotFile(const char *filename);
   TList       *GetListOfNodes() const { return fNodes; }
   TList       *GetListOfEdges() const { return fEdges; }
   Int_t        Layout();
   virtual void SavePrimitive(std::ostream &out, Option_t *option = "");
   void         SetMargin(Double_t m=10) {fMargin = m;}

   ClassDef(TGraphStruct,2)  //Graph structure class
};

#endif
