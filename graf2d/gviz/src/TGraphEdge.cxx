// @(#)root/hist:$Id$
// Author: Olivier Couet 13/07/09

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGraph.h"
#include "TArrow.h"
#include "TPolyLine.h"
#include "TGraphEdge.h"
#include "TGraphNode.h"

#include <gvc.h>

ClassImp(TGraphEdge);

/** \class TGraphEdge
\ingroup gviz

An edge object connecting two nodes which can be added in a
TGraphStruct.
*/

////////////////////////////////////////////////////////////////////////////////
/// Graph Edge default constructor.

TGraphEdge::TGraphEdge(): TObject(), TAttLine()
{
   fNode1  = 0;
   fNode2  = 0;
   fGVEdge = 0;
   fX      = 0;
   fY      = 0;
   fN      = 0;
   fArrX   = 0;
   fArrY   = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Graph Edge normal constructor.

TGraphEdge::TGraphEdge(TGraphNode *n1, TGraphNode *n2)
           :TObject(), TAttLine()
{
   fNode1  = n1;
   fNode2  = n2;
   fGVEdge = 0;
   fX      = 0;
   fY      = 0;
   fN      = 0;
   fArrX   = 0;
   fArrY   = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Graph Edge default destructor.

TGraphEdge::~TGraphEdge()
{
   if (fNode1) delete fNode1;
   if (fNode2) delete fNode2;
   if (fX) { delete [] fX; fX = 0; }
   if (fY) { delete [] fY; fY = 0; }
   if (fN) { delete [] fN; fN = 0; }
}

////////////////////////////////////////////////////////////////////////////////
/// Create the GraphViz edge into the GraphViz data structure gv.

void TGraphEdge::CreateGVEdge(GVizAgraph_t *gv)
{
   if (gv) {
      Agnode_t *n1 = (Agnode_t*)fNode1->GetGVNode();
      Agnode_t *n2 = (Agnode_t*)fNode2->GetGVNode();
#ifdef WITH_CGRAPH
      fGVEdge = (GVizAgedge_t*)agedge((Agraph_t *)gv, n1, n2, NULL, 1);
#else
      fGVEdge = (GVizAgedge_t*)agedge((Agraph_t *)gv, n1, n2);
#endif
   } else {
      Error("CreateGVEdge","Invalid graphviz graph");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Compute distance from point px,py to an edge.

Int_t TGraphEdge::DistancetoPrimitive(Int_t px, Int_t py)
{
   Int_t a = 0, dist = 999;

   for (Int_t i = 1; i <= fN[0]; i++) {
      Int_t n = fN[i];
      TPolyLine polyline(n, &fX[a], &fY[a], "L");
      auto d = polyline.DistancetoPrimitive(px, py);
      if (d < dist) dist = d;
      a += n;
   }

   return dist;
}

////////////////////////////////////////////////////////////////////////////////
/// Execute action corresponding to one event.

void TGraphEdge::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   Int_t a = 0;

   for (Int_t i = 1; i <= fN[0]; i++) {
      Int_t n = fN[i];
      TPolyLine polyline(n, &fX[a], &fY[a], "L");
      polyline.ExecuteEvent(event, px, py);
      a += n;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Layout this edge in the GraphViz space. This is done after gvLayout
/// has been performed.

void TGraphEdge::Layout()
{
   bezier bz;
   Int_t i,j;

   if (fX) { delete [] fX; fX = 0; }
   if (fY) { delete [] fY; fY = 0; }
   if (fN) { delete [] fN; fN = 0; }

   Int_t np = ED_spl((Agedge_t*)fGVEdge)->size;
   fN       = new Int_t[np+1];
   fN[0]    = np;
   Int_t nb = 0;

   // Compute the total size of the splines arrays
   for (i=0; i<np; i++) {
      bz      = ED_spl((Agedge_t*)fGVEdge)->list[i];
      fN[i+1] = bz.size;
      nb      = nb+fN[i+1];
   }

   // Create the vectors holding all the splines' points.
   fX = new Double_t[nb];
   fY = new Double_t[nb];

   // Fill the vectors with the splines' points.
   Int_t k=0;
   for (i=0; i<np; i++) {
      bz    = ED_spl((Agedge_t*)fGVEdge)->list[i];
      fArrX =  bz.ep.x;
      fArrY =  bz.ep.y;
      for (j=0; j<fN[i+1]; j++) {
         fX[k] = bz.list[j].x;
         fY[k] = bz.list[j].y;
         k++;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Paint this edge with its current attributes.

void TGraphEdge::Paint(Option_t *)
{
   Int_t i,n,a;

   TArrow arrow;
   TGraph graph;

   graph.SetLineColor(GetLineColor());
   graph.SetLineStyle(GetLineStyle());
   graph.SetLineWidth(GetLineWidth());
   arrow.SetAngle(38);
   arrow.SetFillColor(GetLineColor());
   arrow.SetLineColor(GetLineColor());

   a = 0;

   for (i=1; i<=fN[0]; i++) {

      // Draw the edge body
      n = fN[i];
      graph.PaintGraph(n, &fX[a], &fY[a], "L");

      // Draw the edge arrow
      arrow.PaintArrow(fX[a+n-1], fY[a+n-1], fArrX, fArrY, 0.03, "|>");

      a = a+n;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Save primitive as a C++ statement(s) on output stream out

void TGraphEdge::SavePrimitive(std::ostream &, Option_t *)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Save attributes as a C++ statement(s) on output stream out
/// called by TGraphStruct::SavePrimitive.

void TGraphEdge::SaveAttributes(std::ostream &out, const char* name)
{
   SaveLineAttributes(out,name,1,1,1);
}

////////////////////////////////////////////////////////////////////////////////

void TGraphEdge::Streamer(TBuffer &/*b*/)
{
}
