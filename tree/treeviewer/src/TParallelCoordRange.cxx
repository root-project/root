// @(#)root/treeviewer:$Id$
// Author: Bastien Dalla Piazza  02/08/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TParallelCoordRange.h"
#include "TParallelCoord.h"
#include "TParallelCoordVar.h"

#include "TPolyLine.h"
#include "TList.h"
#include "TVirtualPad.h"
#include "TPoint.h"
#include "TFrame.h"
#include "TCanvas.h"
#include "TString.h"

#include <iostream>


/** \class TParallelCoordRange
A TParallelCoordRange is a range used for parallel coordinates plots.
*/

////////////////////////////////////////////////////////////////////////////////
/// Default constructor.

TParallelCoordRange::TParallelCoordRange()
   :TNamed("Range","Range"), TAttLine(), fSize(0.01)
{
   fMin = 0;
   fMax = 0;
   fVar = nullptr;
   fSelect = nullptr;
   SetBit(kShowOnPad,true);
   SetBit(kLiveUpdate,false);
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TParallelCoordRange::~TParallelCoordRange()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Normal constructor.

TParallelCoordRange::TParallelCoordRange(TParallelCoordVar *var, Double_t min, Double_t max, TParallelCoordSelect *sel)
   :TNamed("Range","Range"), TAttLine(1,1,1), fSize(0.01)
{
   if(min == max) {
      min = var->GetCurrentMin();
      max = var->GetCurrentMax();
   }
   fMin = min;
   fMax = max;

   fVar = var;
   fSelect = nullptr;

   if (!sel) {
      TParallelCoordSelect* s = var->GetParallel()->GetCurrentSelection();
      if (s) fSelect = s;
      else return;
   } else {
      fSelect = sel;
   }

   SetLineColor(fSelect->GetLineColor());

   SetBit(kShowOnPad,true);
   SetBit(kLiveUpdate,var->GetParallel()->TestBit(TParallelCoord::kLiveUpdate));
}

////////////////////////////////////////////////////////////////////////////////
/// Make the selection which owns the range to be drawn on top of the others.

void TParallelCoordRange::BringOnTop()
{
   TList *list = fVar->GetParallel()->GetSelectList();
   list->Remove(fSelect);
   list->AddLast(fSelect);
   gPad->Update();
}

////////////////////////////////////////////////////////////////////////////////
/// Delete the range.

void TParallelCoordRange::Delete(const Option_t* /*options*/)
{
   fVar->GetRanges()->Remove(this);
   fVar->GetParallel()->CleanUpSelections(this);
   delete this;
}

////////////////////////////////////////////////////////////////////////////////
/// Compute the distance to the primitive.

Int_t TParallelCoordRange::DistancetoPrimitive(Int_t px, Int_t py)
{
   if(TestBit(kShowOnPad)){
      Double_t xx,yy,thisx=0,thisy=0;
      xx = gPad->AbsPixeltoX(px);
      yy = gPad->AbsPixeltoY(py);
      fVar->GetXYfromValue(fMin,thisx,thisy);
      Int_t dist = 9999;
      if(fVar->GetVert()){
         if(xx > thisx-2*fSize && xx < thisx && yy > thisy-fSize && yy<thisy+fSize) dist = 0;
         fVar->GetXYfromValue(fMax,thisx,thisy);
         if(xx > thisx-2*fSize && xx < thisx && yy > thisy-fSize && yy<thisy+fSize) dist = 0;
      } else {
         if(yy > thisy-2*fSize && yy < thisy && xx > thisx-fSize && xx<thisx+fSize) dist = 0;
         fVar->GetXYfromValue(fMax,thisx,thisy);
         if(yy > thisy-2*fSize && yy < thisy && xx > thisx-fSize && xx<thisx+fSize) dist = 0;
      }
      return dist;
   } else return 9999;
}

////////////////////////////////////////////////////////////////////////////////
/// Draw a TParallelCoordRange.

void TParallelCoordRange::Draw(Option_t* options)
{
   AppendPad(options);
}

class TParallelCoordRangeInteractive : public TVirtualPad::TInteractive {
   public:
      Int_t dragpoint = -1;
      Double_t value;

      Bool_t GetValue(TParallelCoordVar *var, TVirtualPad &parent, Int_t px, Int_t py)
      {
         TFrame *frame = parent.GetFrame();

         Double_t xx = parent.AbsPixeltoX(px);
         Double_t yy = parent.AbsPixeltoY(py);

         value = var->GetValuefromXY(xx, yy);

         return var->GetVert() ? (yy > frame->GetY1() && yy < frame->GetY2())
                               : (xx > frame->GetX1() && xx < frame->GetX2());
      }

      void Paint(TParallelCoordVar *var, TVirtualPad &parent, Double_t size, Double_t currentMin, Double_t currentMax)
      {
         std::vector<Double_t> tx(5), ty(5);
         Double_t txx = 0, tyy = 0;
         var->GetXYfromValue(value, txx, tyy);
         if (var->GetVert()) {
            tx[0] = txx;
            tx[1] = tx[4] = txx - size;
            ty[0] = ty[1] = ty[4] = tyy;
            tx[2] = tx[3] = txx - 2 * size;
            ty[2] = tyy + size;
            ty[3] = tyy - size;
         } else {
            ty[0] = tyy;
            ty[1] = ty[4] = tyy - size;
            tx[0] = tx[1] = tx[4] = txx;
            ty[2] = ty[3] = tyy - 2 * size;
            tx[2] = txx - size;
            tx[3] = txx + size;
         }

         // paint marker
         TAttLine{kBlack, 1, 1}.ModifyOn(parent);
         parent.PaintPolyLine(5, tx.data(), ty.data(), "iparallelrangemarker");

         Double_t txx2, tyy2;
         var->GetXYfromValue(dragpoint == 1 ? currentMax : currentMin, txx2, tyy2);
         if (var->GetVert()) {
            tx[0] = tx[1] = txx - 2*size;
            ty[0] = tyy;
            ty[1] = tyy2;
         } else {
            tx[0] = txx;
            tx[1] = txx2;
            ty[0] = ty[1] = tyy - 2*size;
         }

         // paint binding line between markers
         TAttLine{kBlack, 1, 2}.ModifyOn(parent);
         parent.PaintPolyLine(2, tx.data(), ty.data(), "iparallelrangebind");

      }
};

////////////////////////////////////////////////////////////////////////////////
/// Execute the entry.

void TParallelCoordRange::ExecuteEvent(Int_t entry, Int_t px, Int_t py)
{
   if (!gPad) return;

   auto &parent = *gPad;

   if (!parent.IsEditable() && entry!=kMouseEnter) return;

   parent.SetCursor(kPointer);

   auto inter = parent.GetInteractive<TParallelCoordRangeInteractive>(this);

   switch (entry) {
      case kButton1Down:
         inter = parent.MakeInteractive<TParallelCoordRangeInteractive>(this);
         parent.GetCanvas()->Selected(&parent, fVar->GetParallel(), 1);
         if (inter->GetValue(fVar, parent, px, py)) {
            if (inter->value < (fMin + fMax) /2)
               inter->dragpoint = 1;
            else
               inter->dragpoint = 2;
         } else {
            // not found reasonable value
            parent.FreeInteractive(this);
            break;
         }
         // no break
      case kButton1Motion:
         if (inter && inter->GetValue(fVar, parent, px, py)) {
            inter->Paint(fVar, parent, fSize, fMin, fMax);
            parent.UpdateAsync();
         }
         break;
      case kButton1Up:
         if (inter) {
            Double_t min = inter->dragpoint == 1 ? inter->value : fMin;
            Double_t max = inter->dragpoint == 2 ? inter->value : fMax;
            if (min > max)
               std::swap(min, max);
            fMin = min;
            fMax = max;
            parent.Modified();
         }
         parent.FreeInteractive(this);
         break;
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Evaluate if the given value is within the range or not.

bool TParallelCoordRange::IsIn(Double_t evtval)
{
   return evtval>=fMin && evtval<=fMax;
}

////////////////////////////////////////////////////////////////////////////////
/// Paint a TParallelCoordRange.

void TParallelCoordRange::Paint(Option_t* /*options*/)
{
   if(TestBit(kShowOnPad)){
      PaintSlider(fMin,true);
      PaintSlider(fMax,true);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Paint a slider.

void TParallelCoordRange::PaintSlider(Double_t value, bool fill)
{
   SetLineColor(fSelect->GetLineColor());

   TPolyLine *p= new TPolyLine();
   p->SetLineStyle(1);
   p->SetLineColor(1);
   p->SetLineWidth(1);

   Double_t *x = new Double_t[5];
   Double_t *y = new Double_t[5];

   Double_t xx,yy;

   fVar->GetXYfromValue(value,xx,yy);
   if(fVar->GetVert()){
      x[0] = xx; x[1]=x[4]=xx-fSize; x[2]=x[3]=xx-2*fSize;
      y[0]=y[1]=y[4]=yy; y[2] = yy+fSize; y[3] = yy-fSize;
   } else {
      y[0] = yy; y[1]=y[4]=yy-fSize; y[2]=y[3]= yy-2*fSize;
      x[0]=x[1]=x[4]=xx; x[2]=xx-fSize; x[3] = xx+fSize;
   }
   if (fill) {
      p->SetFillStyle(1001);
      p->SetFillColor(0);
      p->PaintPolyLine(4,&x[1],&y[1],"f");
      p->SetFillColor(GetLineColor());
      p->SetFillStyle(3001);
      p->PaintPolyLine(4,&x[1],&y[1],"f");
   }
   p->PaintPolyLine(5,x,y);

   delete p;
   delete [] x;
   delete [] y;
}

////////////////////////////////////////////////////////////////////////////////
/// Print info about the range.

void TParallelCoordRange::Print(Option_t* /*options*/) const
{
   printf("On \"%s\" : min = %f, max = %f\n", fVar->GetTitle(), fMin, fMax);
}

////////////////////////////////////////////////////////////////////////////////
/// Make the selection which owns the range to be drawn under all the others.

void TParallelCoordRange::SendToBack()
{
   TList *list = fVar->GetParallel()->GetSelectList();
   list->Remove(fSelect);
   list->AddFirst(fSelect);
   gPad->Update();
}

////////////////////////////////////////////////////////////////////////////////
/// Set the selection line color.

void  TParallelCoordRange::SetLineColor(Color_t col)
{
   fSelect->SetLineColor(col);
   TAttLine::SetLineColor(col);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the selection line width.

void  TParallelCoordRange::SetLineWidth(Width_t wid)
{
   fSelect->SetLineWidth(wid);
}



/** \class TParallelCoordSelect
A TParallelCoordSelect is a specialised TList to hold TParallelCoordRanges used
by TParallelCoord.

Selections of specific entries can be defined over the data se using parallel
coordinates. With that representation, a selection is an ensemble of ranges
defined on the axes. Ranges defined on the same axis are conjugated with OR
(an entry must be in one or the other ranges to be selected). Ranges on
different axes are are conjugated with AND (an entry must be in all the ranges
to be selected). Several selections can be defined with different colors. It is
possible to generate an entry list from a given selection and apply it to the
tree using the editor ("Apply to tree" button).
*/

////////////////////////////////////////////////////////////////////////////////
/// Default constructor.

TParallelCoordSelect::TParallelCoordSelect()
   : TList(), TAttLine(kBlue,1,1)
{
   fTitle = "Selection";
   SetBit(kActivated,true);
   SetBit(kShowRanges,true);
}

////////////////////////////////////////////////////////////////////////////////
/// Normal constructor.

TParallelCoordSelect::TParallelCoordSelect(const char* title)
   : TList(), TAttLine(kBlue,1,1)
{
   fTitle = title;
   SetBit(kActivated,true);
   SetBit(kShowRanges,true);
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TParallelCoordSelect::~TParallelCoordSelect()
{
   TIter next(this);
   TParallelCoordRange* range;
   while ((range = (TParallelCoordRange*)next())) range->GetVar()->GetRanges()->Remove(range);
   TList::Delete();
}

////////////////////////////////////////////////////////////////////////////////
/// Activate the selection.

void TParallelCoordSelect::SetActivated(bool on)
{
   TIter next(this);
   TParallelCoordRange* range;
   while ((range = (TParallelCoordRange*)next())) range->SetBit(TParallelCoordRange::kShowOnPad,on);
   SetBit(kActivated,on);
}

////////////////////////////////////////////////////////////////////////////////
/// Show the ranges needles.

void TParallelCoordSelect::SetShowRanges(bool s)
{
   TIter next(this);
   TParallelCoordRange* range;
   while ((range = (TParallelCoordRange*)next())) range->SetBit(TParallelCoordRange::kShowOnPad,s);
   SetBit(kShowRanges,s);
}
