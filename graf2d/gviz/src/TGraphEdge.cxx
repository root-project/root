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

ClassImp(TGraphEdge)

//______________________________________________________________________________
/* Begin_Html
<center><h2>Graph Edge class</h2></center>
TGraphEdge is an edge object connecting two nodes which can be added in a
TGraphStruct.
End_Html */


//______________________________________________________________________________
TGraphEdge::TGraphEdge(): TObject(), TAttLine()
{
   // Graph Edge default constructor.

   fNode1  = 0;
   fNode2  = 0;
   fGVEdge = 0;
   fX      = 0;
   fY      = 0;
   fN      = 0;
   fArrX   = 0;
   fArrY   = 0;
}


//______________________________________________________________________________
TGraphEdge::TGraphEdge(TGraphNode *n1, TGraphNode *n2)
           :TObject(), TAttLine()
{
   // Graph Edge normal constructor.

   fNode1  = n1;
   fNode2  = n2;
   fGVEdge = 0;
   fX      = 0;
   fY      = 0;
   fN      = 0;
   fArrX   = 0;
   fArrY   = 0;
}


//______________________________________________________________________________
TGraphEdge::~TGraphEdge()
{
   // Graph Edge default destructor.

   if (fNode1) delete fNode1;
   if (fNode2) delete fNode2;
   if (fX) delete [] fX; fX = 0;
   if (fY) delete [] fY; fY = 0;
   if (fN) delete [] fN; fN = 0;
}


//______________________________________________________________________________
void TGraphEdge::CreateGVEdge(GVizAgraph_t *gv)
{
   // Create the GraphViz edge into the GraphViz data structure gv.

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


//______________________________________________________________________________
Int_t TGraphEdge::DistancetoPrimitive(Int_t px, Int_t py)
{
   // Compute distance from point px,py to an edge.

   Int_t i,n,a,dist=999;

   TPolyLine *polyline;
   a = 0;

   for (i=1; i<=fN[0]; i++) {
      n = fN[i];
      polyline = new TPolyLine(n, &fX[a], &fY[a], "L");
      dist = polyline->DistancetoPrimitive(px, py);
      a = a+n;
   }

   return dist;
}


//______________________________________________________________________________
void TGraphEdge::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   // Execute action corresponding to one event.

   Int_t i,n,a;

   TPolyLine *polyline;
   a = 0;

   for (i=1; i<=fN[0]; i++) {
      n = fN[i];
      polyline = new TPolyLine(n, &fX[a], &fY[a], "L");
      polyline->ExecuteEvent(event, px, py);
      a = a+n;
   }
}


//______________________________________________________________________________
void TGraphEdge::Layout()
{
   // Layout this edge in the GraphViz space. This is done after gvLayout
   // has been performed.

   bezier bz;
   Int_t i,j;

   if (fX) delete [] fX; fX = 0;
   if (fY) delete [] fY; fY = 0;
   if (fN) delete [] fN; fN = 0;

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


//______________________________________________________________________________
void TGraphEdge::Paint(Option_t *)
{
   // Paint this edge with its current attributes.

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


//______________________________________________________________________________
void TGraphEdge::SavePrimitive(std::ostream &, Option_t *)
{
   // Save primitive as a C++ statement(s) on output stream out
}

//______________________________________________________________________________
void TGraphEdge::SaveAttributes(std::ostream &out, const char* name)
{
   // Save attributes as a C++ statement(s) on output stream out
   // called by TGraphStruct::SavePrimitive.

   SaveLineAttributes(out,name,1,1,1);
}


//______________________________________________________________________________
void TGraphEdge::Streamer(TBuffer &/*b*/)
{
}
