// @(#)root/hist:$Id$
// Author: Olivier Couet 13/07/09

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGraphNode
#define ROOT_TGraphNode

#include "TNamed.h"

#include "TAttText.h"

#include "TAttFill.h"

#include "TAttLine.h"

struct GVizAgraph_t;
struct GVizAgnode_t;


class TGraphNode : public TNamed, public TAttText, public TAttFill, public TAttLine  {

protected:



   GVizAgnode_t *fGVNode; ///< Graphviz node
   Double_t fX;           ///< Node's center X coordinate
   Double_t fY;           ///< Node's center Y coordinate
   Double_t fH;           ///< Node height
   Double_t fW;           ///< Node width

public:

   TGraphNode();
   TGraphNode(const char *name, const char *title="");
   ~TGraphNode() override;

   void           CreateGVNode(GVizAgraph_t *gv);
   Int_t  DistancetoPrimitive(Int_t px, Int_t py) override;
   void   ExecuteEvent(Int_t event, Int_t px, Int_t py) override;
   void           SetGVNode(GVizAgnode_t *gvn) {fGVNode = gvn;}
   void   SetTextAngle(Float_t) override {}
   GVizAgnode_t  *GetGVNode() {return fGVNode;}
   void           Layout();
   void   Paint(Option_t *option="") override;
   void   SavePrimitive(std::ostream &, Option_t *) override;
   void           SaveAttributes(std::ostream &);

   ClassDefOverride(TGraphNode,2)  //Graph node class
};

#endif
