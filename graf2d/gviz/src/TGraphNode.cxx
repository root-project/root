// @(#)root/hist:$Id$
// Author: Olivier Couet 13/07/09

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TLatex.h"
#include "TEllipse.h"
#include "TGraphNode.h"

#include <gvc.h>

ClassImp(TGraphNode)

//______________________________________________________________________________
/* Begin_Html
<center><h2>Graph Node class</h2></center>
End_Html */


//______________________________________________________________________________
TGraphNode::TGraphNode(): TNamed(), TAttText()
{
   // Graph node default constructor.

   fGVNode = 0;
   fX      = 0;
   fY      = 0;
   fW      = 0;
   fH      = 0;
}


//______________________________________________________________________________
TGraphNode::TGraphNode(const char *name,const char *title)
           :TNamed(name,title), TAttText()
{
   // Graph node normal constructor.

   fGVNode = 0;
   fX      = 0;
   fY      = 0;
   fW      = 0;
   fH      = 0;
}


//______________________________________________________________________________
TGraphNode::~TGraphNode()
{
   // Graph Node default destructor.
   
}


//______________________________________________________________________________
void TGraphNode::CreateGVNode(Agraph_t *gv)
{
   // Create the GraphViz node into the GraphViz data structure gv.

   if (gv) {
      fGVNode = agnode(gv, (char *)GetName());
   } else {
      Error("CreateGVNode","Invalid graphviz graph");
   }
}


//______________________________________________________________________________
Int_t TGraphNode::DistancetoPrimitive(Int_t /*px*/, Int_t /*py*/)
{
   // Compute distance from point px,py to a node.

   return 999;
}


//______________________________________________________________________________
void TGraphNode::Layout()
{
   // Layout this node in the GraphViz space. This is done after gvLayout
   // has been performed.

#ifdef ND_coord
   fX = ND_coord(fGVNode).x;
   fY = ND_coord(fGVNode).y;
#endif
#ifdef ND_coord_i
   fX = ND_coord_i(fGVNode).x;
   fY = ND_coord_i(fGVNode).y;
#endif
   fW = ND_width(fGVNode)*36;
   fH = ND_height(fGVNode)*36;
}


//______________________________________________________________________________
void TGraphNode::Paint(Option_t *)
{
   // Paint this node with its current attributes.

   TEllipse ellipse;
   TLatex text;
   text.SetTextAlign(22);

   // Draw the node shape
   // ND_shape(fGVNode)->name gives the type of shape.
   ellipse.PaintEllipse(fX, fY, fW, fH, 0., 360., 0., "");
   
   // Draw the node name
   text.PaintLatex(fX, fY, 0., GetTextSize(), (char*)GetName());
}


//______________________________________________________________________________
void TGraphNode::Streamer(TBuffer &/*b*/)
{
}
