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

ClassImp(TGraphNode);

/** \class TGraphNode
\ingroup gviz

A graph node object which can be added in a TGraphStruct.
*/

////////////////////////////////////////////////////////////////////////////////
/// Graph node default constructor.

TGraphNode::TGraphNode(): TNamed(), TAttText()
{
   fGVNode = 0;
   fX      = 0;
   fY      = 0;
   fW      = 0;
   fH      = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Graph node normal constructor.

TGraphNode::TGraphNode(const char *name,const char *title)
           :TNamed(name,title), TAttText()
{
   fGVNode = 0;
   fX      = 0;
   fY      = 0;
   fW      = 0;
   fH      = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Graph Node default destructor.

TGraphNode::~TGraphNode()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Create the GraphViz node into the GraphViz data structure gv.

void TGraphNode::CreateGVNode(GVizAgraph_t *gv)
{
   if (gv) {
#ifdef WITH_CGRAPH
      fGVNode = (GVizAgnode_t*)agnode((Agraph_t*)gv, (char *)GetName(), 1);
#else
      fGVNode = (GVizAgnode_t*)agnode((Agraph_t*)gv, (char *)GetName());
#endif
   } else {
      Error("CreateGVNode","Invalid graphviz graph");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Compute distance from point px,py to a node.

Int_t TGraphNode::DistancetoPrimitive(Int_t px, Int_t py)
{
   Int_t dist;

   // The node is drawn as an ellipse
   TEllipse ellipse(fX, fY, fW, fH, 0., 360., 0.);
   ellipse.SetFillColor(1); // in order to pick the ellipse "inside"
   dist =  ellipse.DistancetoPrimitive(px, py);

   return dist;
}

////////////////////////////////////////////////////////////////////////////////
/// Execute action corresponding to one event.

void TGraphNode::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   TEllipse ellipse(fX, fY, fW, fH, 0., 360., 0.);
   ellipse.ExecuteEvent(event,px, py);
   fX = ellipse.GetX1();
   fY = ellipse.GetY1();
   fW = ellipse.GetR1();
   fH = ellipse.GetR2();
}

////////////////////////////////////////////////////////////////////////////////
/// Layout this node in the GraphViz space. This is done after gvLayout
/// has been performed.

void TGraphNode::Layout()
{
#ifdef ND_coord
   fX = ND_coord((Agnode_t*)fGVNode).x;
   fY = ND_coord((Agnode_t*)fGVNode).y;
#endif
#ifdef ND_coord_i
   fX = ND_coord_i((Agnode_t*)fGVNode).x;
   fY = ND_coord_i((Agnode_t*)fGVNode).y;
#endif
   fW = ND_width((Agnode_t*)fGVNode)*36;
   fH = ND_height((Agnode_t*)fGVNode)*36;
}

////////////////////////////////////////////////////////////////////////////////
/// Paint this node with its current attributes.

void TGraphNode::Paint(Option_t *)
{
   TEllipse ellipse;
   TLatex text;
   text.SetTextAlign(22);

   // Draw the node shape as an ellipse
   // ND_shape((Agnode_t*)fGVNode)->name gives the type of shape.
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

////////////////////////////////////////////////////////////////////////////////
/// Save primitive as a C++ statement(s) on output stream out

void TGraphNode::SavePrimitive(std::ostream &, Option_t *)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Save attributes as a C++ statement(s) on output stream out
/// called by TGraphStruct::SavePrimitive.

void TGraphNode::SaveAttributes(std::ostream &out)
{
   SaveFillAttributes(out,GetName(),0,1001);
   SaveLineAttributes(out,GetName(),1,1,1);
   SaveTextAttributes(out,GetName(),0,0,0,0,0);
}

////////////////////////////////////////////////////////////////////////////////

void TGraphNode::Streamer(TBuffer &/*b*/)
{
}
