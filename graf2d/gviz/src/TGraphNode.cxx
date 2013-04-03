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
TGraphNode is a graph node object which can be added in a TGraphStruct.
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
#ifdef WITH_CGRAPH
      fGVNode = agnode(gv, (char *)GetName(), 1);
#else
      fGVNode = agnode(gv, (char *)GetName());
#endif
   } else {
      Error("CreateGVNode","Invalid graphviz graph");
   }
}


//______________________________________________________________________________
Int_t TGraphNode::DistancetoPrimitive(Int_t px, Int_t py)
{
   // Compute distance from point px,py to a node.

   Int_t dist;

   // The node is drawn as an ellipse
   TEllipse ellipse(fX, fY, fW, fH, 0., 360., 0.);
   ellipse.SetFillColor(1); // in order to pick the ellipse "inside"
   dist =  ellipse.DistancetoPrimitive(px, py);

   return dist;
}


//______________________________________________________________________________
void TGraphNode::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   // Execute action corresponding to one event.
   
   TEllipse ellipse(fX, fY, fW, fH, 0., 360., 0.);
   ellipse.ExecuteEvent(event,px, py);
   fX = ellipse.GetX1();
   fY = ellipse.GetY1();
   fW = ellipse.GetR1();
   fH = ellipse.GetR2();
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

   // Draw the node shape as an ellipse
   // ND_shape(fGVNode)->name gives the type of shape.
   ellipse.SetFillStyle(GetFillStyle());
   ellipse.SetFillColor(GetFillColor());
   ellipse.SetLineColor(GetLineColor());
   ellipse.SetLineStyle(GetLineStyle());
   ellipse.SetLineWidth(GetLineWidth());
   ellipse.PaintEllipse(fX, fY, fW, fH, 0., 360., 0., "");
   
   // Draw the node title
   text.SetTextColor(GetTextColor());
   text.SetTextFont(GetTextFont());
   text.PaintLatex(fX, fY, 0., GetTextSize(), (char*)GetTitle());
}


//______________________________________________________________________________
void TGraphNode::SavePrimitive(ostream &, Option_t *)
{
   // Save primitive as a C++ statement(s) on output stream out
}


//______________________________________________________________________________
void TGraphNode::SaveAttributes(ostream &out)
{
   // Save attributes as a C++ statement(s) on output stream out
   // called by TGraphStruct::SavePrimitive.

   SaveFillAttributes(out,GetName(),0,1001);
   SaveLineAttributes(out,GetName(),1,1,1);
   SaveTextAttributes(out,GetName(),0,0,0,0,0);
}

//______________________________________________________________________________
void TGraphNode::Streamer(TBuffer &/*b*/)
{
}
