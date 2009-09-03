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

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

#ifndef ROOT_TAttText
#include "TAttText.h"
#endif

struct Agraph_t;
struct Agnode_t;


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGraphNode                                                           //
//                                                                      //
// Interface to the graphviz package.                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


class TGraphNode : public TNamed, public TAttText  {

protected:

   Agnode_t *fGVNode; // Graphviz node
   Double_t fX;       // Node's center X coordinate
   Double_t fY;       // Node's center Y coordinate
   Double_t fH;       // Node height
   Double_t fW;       // Node width

public:

   TGraphNode();
   TGraphNode(const char *name, const char *title="");
   virtual ~TGraphNode();

   void           CreateGVNode(Agraph_t *gv);
   virtual Int_t  DistancetoPrimitive(Int_t px, Int_t py);   
   void           SetGVNode(Agnode_t *gvn) {fGVNode = gvn;}  
   Agnode_t      *GetGVNode() {return fGVNode;}
   void           Layout();
   virtual void   Paint(Option_t *option="");   

   ClassDef(TGraphNode,1)  //Graph node class
};

#endif
