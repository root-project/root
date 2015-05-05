// @(#)root/hist:$Id$
// TH2Poly v2.1
// Author: Olivier Couet, Deniz Gunceler

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TClass.h"
#include "TH2Poly.h"
#include "TCutG.h"
#include "TList.h"
#include "TMath.h"
#include "TMultiGraph.h"
#include "TGraph.h"
#include "TStyle.h"
#include "TCanvas.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include "Riostream.h"

ClassImp(TH2Poly)

//______________________________________________________________________________
/* Begin_Html
<center><h2>TH2Poly: 2D Histogram with Polygonal Bins</h2></center>

<h3>Overview</h3>
<tt>TH2Poly</tt> is a 2D Histogram class (TH2) allowing to define polygonal
bins of arbitrary shape.
<p>
Each bin in the <tt>TH2Poly</tt> histogram is a <tt>TH2PolyBin</tt> object.
<tt>TH2PolyBin</tt> is a very simple class containing the vertices (stored
as <tt>TGraph</tt>s or <tt>TMultiGraph</tt>s ) and contents of the polygonal
bin as well as several related functions.
<p>
Essentially, a <tt>TH2Poly</tt> is a TList of <tt>TH2PolyBin</tt> objects
with methods to manipulate them.
<p>
Bins are defined using one of the <tt>AddBin()</tt> methods. The bin definition
should be done before filling.
<p>
The histogram can be filled with <tt>Fill(Double_t x, Double_t y, Double_t w)
</tt>. <tt>w</tt> is the weight.
If no weight is specified, it is assumed to be 1.
<p>
Not all histogram's area need to be binned. Filling an area without bins,
will falls into the overflows. Adding a bin is not retroactive; it doesn't
affect previous fillings. A <tt>Fill()</tt> call, that
was previously ignored due to the lack of a bin at the specified location, is
not reconsidered when that location is binned later.
<p>
If there are two overlapping bins, the first one in the list will be incremented
by <tt>Fill()</tt>.
<p>
The histogram may automatically extends its limits if a bin outside the
histogram limits is added. This is done when the default constructor (with no
arguments) is used. It generates a histogram with no limits along the X and Y
axis. Adding bins to it will extend it up to a proper size.
<p>
<tt>TH2Poly</tt> implements a partitioning algorithm to speed up bins' filling.
The partitioning algorithm divides the histogram into regions called cells.
The bins that each cell intersects are recorded in an array of <tt>TList</tt>s.
When a coordinate in the histogram is to be filled; the method (quickly) finds
which cell the coordinate belongs.  It then only loops over the bins
intersecting that cell to find the bin the input coordinate corresponds to.
The partitioning of the histogram is updated continuously as each bin is added.
The default number of cells on each axis is 25. This number could be set to
another value in the constructor or adjusted later by calling the
<tt>ChangePartition(Int_t, Int_t)</tt> method. The partitioning algorithm is
considerably faster than the brute force algorithm (i.e. checking if each bin
contains the input coordinates), especially if the histogram is to be filled
many times.
<p>
The following very simple macro shows how to build and fill a <tt>TH2Poly</tt>:
<pre>
{
    TH2Poly *h2p = new TH2Poly();

    Double_t x1[] = {0, 5, 6};
    Double_t y1[] = {0, 0, 5};
    Double_t x2[] = {0, -1, -1, 0};
    Double_t y2[] = {0, 0, -1, 3};
    Double_t x3[] = {4, 3, 0, 1, 2.4};
    Double_t y3[] = {4, 3.7, 1, 3.7, 2.5};

    h2p->AddBin(3, x1, y1);
    h2p->AddBin(4, x2, y2);
    h2p->AddBin(5, x3, y3);

    h2p->Fill(0.1, 0.01, 3);
    h2p->Fill(-0.5, -0.5, 7);
    h2p->Fill(-0.7, -0.5, 1);
    h2p->Fill(1, 3, 1.5);
}
</pre>
<p>
More examples can bin found in <tt>$ROOTSYS/tutorials/hist/th2poly*.C</tt>

<h3>Partitioning Algorithm</h3>
The partitioning algorithm forms an essential part of the <tt>TH2Poly</tt>
class. It is implemented to speed up the filling of bins.
<p>
With the brute force approach, the filling is done in the following way:  An
iterator loops over all bins in the <tt>TH2Poly</tt> and invokes the
method <tt>IsInside()</tt> for each of them.
This method checks if the input location is in that bin. If the filling
coordinate is inside, the bin is filled. Looping over all the bin is
very slow.
<p>
The alternative is to divide the histogram into virtual rectangular regions
called "cells". Each cell stores the pointers of the bins intersecting it.
When a coordinate is to be filled, the method finds which cell the coordinate
falls into. Since the cells are rectangular, this can be done very quickly.
It then only loops over the bins associated with that cell.
<p>
The addition of bins to the appropriate cells is done when the bin is added
to the histogram. To do this, <tt>AddBin()</tt> calls the
<tt>AddBinToPartition()</tt> method.
This method adds the input bin to the partitioning matrix.
<p>
The number of partition cells per axis can be specified in the constructor.
If it is not specified, the default value of 25 along each axis will be
assigned. This value was chosen because it is small enough to avoid slowing
down AddBin(), while being large enough to enhance Fill() by a considerable
amount. Regardless of how it is initialized at construction time, it can be
changed later with the <tt>ChangePartition()</tt> method.
<tt>ChangePartition()</tt> deletes the
old partition matrix and generates a new one with the specified number of cells
on each axis.
<p>
The optimum number of partition cells per axis changes with the number of
times <tt>Fill()</tt> will be called.  Although partitioning greatly speeds up
filling, it also adds a constant time delay into the code. When <tt>Fill()</tt>
is to be called many times, it is more efficient to divide the histogram into
a large number cells. However, if the histogram is to be filled only a few
times, it is better to divide into a small number of cells.
End_Html */



//______________________________________________________________________________
TH2Poly::TH2Poly()
{
   // Default Constructor. No boundaries specified.

   Initialize(0., 0., 0., 0., 25, 25);
   SetName("NoName");
   SetTitle("NoTitle");
   SetFloat();
}


//______________________________________________________________________________
TH2Poly::TH2Poly(const char *name,const char *title, Double_t xlow,Double_t xup
                                             , Double_t ylow,Double_t yup)
{
   // Constructor with specified name and boundaries,
   // but no partition cell number.

   Initialize(xlow, xup, ylow, yup, 25, 25);
   SetName(name);
   SetTitle(title);
   SetFloat(kFALSE);
}


//______________________________________________________________________________
TH2Poly::TH2Poly(const char *name,const char *title,
           Int_t nX, Double_t xlow, Double_t xup,
           Int_t nY, Double_t ylow, Double_t yup)
{
   // Constructor with specified name and boundaries and partition cell number.

   Initialize(xlow, xup, ylow, yup, nX, nY);
   SetName(name);
   SetTitle(title);
   SetFloat(kFALSE);
}


//______________________________________________________________________________
TH2Poly::~TH2Poly()
{
   // Destructor.

   delete[] fCells;
   delete[] fIsEmpty;
   delete[] fCompletelyInside;
   // delete at the end the bin List since it owns the objects
   delete fBins;
}


//______________________________________________________________________________
Int_t TH2Poly::AddBin(TObject *poly)
{
   // Adds a new bin to the histogram. It can be any object having the method
   // IsInside(). It returns the bin number in the histogram. It returns 0 if
   // it failed to add. To allow the histogram limits to expand when a bin
   // outside the limits is added, call SetFloat() before adding the bin.

   if (!poly) return 0;

   if (fBins == 0) {
      fBins = new TList();
      fBins->SetOwner();
   }

   fNcells++;
   TH2PolyBin *bin = new TH2PolyBin(poly, fNcells);

   // If the bin lies outside histogram boundaries, then extends the boundaries.
   // Also changes the partition information accordingly
   Bool_t flag = kFALSE;
   if (fFloat) {
      if (fXaxis.GetXmin() > bin->GetXMin()) {
         fXaxis.Set(100, bin->GetXMin(), fXaxis.GetXmax());
         flag = kTRUE;
      }
      if (fXaxis.GetXmax() < bin->GetXMax()) {
         fXaxis.Set(100, fXaxis.GetXmin(), bin->GetXMax());
         flag = kTRUE;
      }
      if (fYaxis.GetXmin() > bin->GetYMin()) {
         fYaxis.Set(100, bin->GetYMin(), fYaxis.GetXmax());
         flag = kTRUE;
      }
      if (fYaxis.GetXmax() < bin->GetYMax()) {
         fYaxis.Set(100, fYaxis.GetXmin(), bin->GetYMax());
         flag = kTRUE;
      }
      if (flag) ChangePartition(fCellX, fCellY);
   } else {
      /*Implement polygon clipping code here*/
   }

   fBins->Add((TObject*) bin);
   SetNewBinAdded(kTRUE);

   // Adds the bin to the partition matrix
   AddBinToPartition(bin);

   return fNcells;
}


//______________________________________________________________________________
Int_t TH2Poly::AddBin(Int_t n, const Double_t *x, const Double_t *y)
{
   // Adds a new bin to the histogram. The number of vertices and their (x,y)
   // coordinates are required as input. It returns the bin number in the
   // histogram.

   TGraph *g = new TGraph(n, x, y);
   Int_t bin = AddBin(g);
   return bin;
}


//______________________________________________________________________________
Int_t TH2Poly::AddBin(Double_t x1, Double_t y1, Double_t x2, Double_t  y2)
{
   // Add a new bin to the histogram. The bin shape is a rectangle.
   // It returns the bin number of the bin in the histogram.

   Double_t x[] = {x1, x1, x2, x2, x1};
   Double_t y[] = {y1, y2, y2, y1, y1};
   TGraph *g = new TGraph(5, x, y);
   Int_t bin = AddBin(g);
   return bin;
}


//______________________________________________________________________________
Bool_t TH2Poly::Add(const TH1 *h1, Double_t c1)
{
   // Performs the operation: this = this + c1*h1.

   Int_t bin;

   TH2Poly *h1p = (TH2Poly *)h1;

   // Check if number of bins is the same.
   if (h1p->GetNumberOfBins() != fNcells) {
      Error("Add","Attempt to add histograms with different number of bins");
      return kFALSE;
   }

   // Check if the bins are the same.
   TList *h1pBins = h1p->GetBins();
   TH2PolyBin *thisBin, *h1pBin;
   for (bin=1;bin<=fNcells;bin++) {
      thisBin = (TH2PolyBin*)fBins->At(bin-1);
      h1pBin  = (TH2PolyBin*)h1pBins->At(bin-1);
      if(thisBin->GetXMin() != h1pBin->GetXMin() ||
         thisBin->GetXMax() != h1pBin->GetXMax() ||
         thisBin->GetYMin() != h1pBin->GetYMin() ||
         thisBin->GetYMax() != h1pBin->GetYMax()) {
         Error("Add","Attempt to add histograms with different bin limits");
         return kFALSE;
      }
   }

   // Create Sumw2 if h1p has Sumw2 set
   if (fSumw2.fN == 0 && h1p->GetSumw2N() != 0) Sumw2();

   // Perform the Add.
   Double_t factor =1;
   if (h1p->GetNormFactor() != 0)
      factor = h1p->GetNormFactor()/h1p->GetSumOfWeights();
   for (bin=1;bin<=fNcells;bin++) {
      thisBin = (TH2PolyBin*)fBins->At(bin-1);
      h1pBin  = (TH2PolyBin*)h1pBins->At(bin-1);
      thisBin->SetContent(thisBin->GetContent()+c1*h1pBin->GetContent());
      if (fSumw2.fN) {
         Double_t e1 = factor*h1p->GetBinError(bin);
         fSumw2.fArray[bin] += c1*c1*e1*e1;
      }
   }
   return kTRUE;
}


//______________________________________________________________________________
Bool_t TH2Poly::Add(TF1 *, Double_t, Option_t *)
{
   // Performs the operation: this = this + c1*f1.

   Warning("Add","Not implement for TH2Poly");
   return kFALSE;
}


//______________________________________________________________________________
Bool_t TH2Poly::Add(const TH1 *, const TH1 *, Double_t, Double_t)
{
   // Replace contents of this histogram by the addition of h1 and h2.

   Warning("Add","Not implement for TH2Poly");
   return kFALSE;
}


//______________________________________________________________________________
void TH2Poly::AddBinToPartition(TH2PolyBin *bin)
{
   // Adds the input bin into the partition cell matrix. This method is called
   // in AddBin() and ChangePartition().

   // Cell Info
   Int_t nl, nr, mb, mt; // Max/min indices of the cells that contain the bin
   Double_t xclipl, xclipr, yclipb, yclipt; // x and y coordinates of a cell
   Double_t binXmax, binXmin, binYmax, binYmin; // The max/min bin coordinates

   binXmax = bin->GetXMax();
   binXmin = bin->GetXMin();
   binYmax = bin->GetYMax();
   binYmin = bin->GetYMin();
   nl = (Int_t)(floor((binXmin - fXaxis.GetXmin())/fStepX));
   nr = (Int_t)(floor((binXmax - fXaxis.GetXmin())/fStepX)); 
   mb = (Int_t)(floor((binYmin - fYaxis.GetXmin())/fStepY));
   mt = (Int_t)(floor((binYmax - fYaxis.GetXmin())/fStepY));

   // Make sure the array indices are correct.
   if (nr>=fCellX) nr = fCellX-1;
   if (mt>=fCellY) mt = fCellY-1;
   if (nl<0)       nl = 0;
   if (mb<0)       mb = 0;

   // number of cells in the grid
   //N.B. not to be confused with fNcells (the number of bins) ! 
   fNCells = fCellX*fCellY;

   // Loop over all cells
   for (int i = nl; i <= nr; i++) {
      xclipl = fXaxis.GetXmin() + i*fStepX;
      xclipr = xclipl + fStepX;
      for (int j = mb; j <= mt; j++) {
         yclipb = fYaxis.GetXmin() + j*fStepY;
         yclipt = yclipb + fStepY;

         // If the bin is completely inside the cell,
         // add that bin to the cell then return
         if ((binXmin >= xclipl) && (binXmax <= xclipr) &&
             (binYmax <= yclipt) && (binYmin >= yclipb)){
            fCells[i + j*fCellX].Add((TObject*) bin);
            fIsEmpty[i + j*fCellX] = kFALSE;  // Makes the cell non-empty
            return;
         }

         // If any of the sides of the cell intersect with any side of the bin,
         // add that bin then continue
         if (IsIntersecting(bin, xclipl, xclipr, yclipb, yclipt)) {
            fCells[i + j*fCellX].Add((TObject*) bin);
            fIsEmpty[i + j*fCellX] = kFALSE;  // Makes the cell non-empty
            continue;
         }
         // If a corner of the cell is inside the bin and since there is no
         // intersection, then that cell completely inside the bin.
         if((bin->IsInside(xclipl,yclipb)) || (bin->IsInside(xclipl,yclipt))){
            fCells[i + j*fCellX].Add((TObject*) bin);
            fIsEmpty[i + j*fCellX] = kFALSE;  // Makes the cell non-empty
            fCompletelyInside[i + fCellX*j] = kTRUE;
            continue;
         }
         if((bin->IsInside(xclipr,yclipb)) || (bin->IsInside(xclipr,yclipt))){
            fCells[i + j*fCellX].Add((TObject*) bin);
            fIsEmpty[i + j*fCellX] = kFALSE;  // Makes the cell non-empty
            fCompletelyInside[i + fCellX*j] = kTRUE;
            continue;
         }
      }
   }
}


//______________________________________________________________________________
void TH2Poly::ChangePartition(Int_t n, Int_t m)
{
   // Changes the number of partition cells in the histogram.
   // Deletes the old partition and constructs a new one.

   fCellX = n;                          // Set the number of cells
   fCellY = m;                          // Set the number of cells

   delete [] fCells;                    // Deletes the old partition

   // number of cells in the grid
   //N.B. not to be confused with fNcells (the number of bins) ! 
   fNCells = fCellX*fCellY;
   fCells  = new TList [fNCells];  // Sets an empty partition

   fStepX = (fXaxis.GetXmax() - fXaxis.GetXmin())/fCellX;
   fStepY = (fYaxis.GetXmax() - fYaxis.GetXmin())/fCellY;

   delete [] fIsEmpty;
   delete [] fCompletelyInside;
   fIsEmpty = new Bool_t [fNCells];
   fCompletelyInside = new Bool_t [fNCells];

   // Initializes the flags
   for (int i = 0; i<fNCells; i++) {
      fIsEmpty[i]          = kTRUE;
      fCompletelyInside[i] = kFALSE;
   }

   // TList iterator
   TIter    next(fBins);
   TObject  *obj;

   while((obj = next())){   // Loop over bins and add them to the partition
      AddBinToPartition((TH2PolyBin*) obj);
   }
}

//______________________________________________________________________________
void TH2Poly::ClearBinContents()
{
   // Clears the contents of all bins in the histogram.

   TIter next(fBins);
   TObject *obj;
   TH2PolyBin *bin;

   // Clears the bin contents
   while ((obj = next())) {
      bin = (TH2PolyBin*) obj;
      bin->ClearContent();
   }

   // Clears the statistics
   fTsumw   = 0;
   fTsumwx  = 0;
   fTsumwx2 = 0;
   fTsumwy  = 0;
   fTsumwy2 = 0;
   fEntries = 0;
}


//______________________________________________________________________________
void TH2Poly::Reset(Option_t *opt)
{
   // Reset this histogram: contents, errors, etc.

   TIter next(fBins);
   TObject *obj;
   TH2PolyBin *bin;

   // Clears the bin contents
   while ((obj = next())) {
      bin = (TH2PolyBin*) obj;
      bin->ClearContent();
   }

   TH2::Reset(opt);
}


//______________________________________________________________________________
TH1 *TH2Poly::DrawCopy(Option_t *option) const
{
   // Draw copy.

   TString opt = option;
   opt.ToLower();
   if (gPad && !opt.Contains("same")) gPad->Clear();
   TH2Poly *newth2 = (TH2Poly*)Clone();
   newth2->SetDirectory(0);
   newth2->SetBit(kCanDelete);
   newth2->AppendPad(option);
   return newth2;
}


//______________________________________________________________________________
Int_t TH2Poly::FindBin(Double_t x, Double_t y, Double_t)
{
   // Returns the bin number of the bin at the given coordinate. -1 to -9 are
   // the overflow and underflow bins.  overflow bin -5 is the unbinned areas in
   // the histogram (also called "the sea"). The third parameter can be left
   // blank.
   // The overflow/underflow bins are:
   //
   // -1 | -2 | -3
   // -------------
   // -4 | -5 | -6
   // -------------
   // -7 | -8 | -9
   //
   // where -5 means is the "sea" bin (i.e. unbinned areas)


   // Checks for overflow/underflow
   Int_t overflow = 0;
   if      (y > fYaxis.GetXmax()) overflow += -1;
   else if (y > fYaxis.GetXmin()) overflow += -4;
   else                           overflow += -7;
   if      (x > fXaxis.GetXmax()) overflow += -2;
   else if (x > fXaxis.GetXmin()) overflow += -1;
   if (overflow != -5) return overflow;

   // Finds the cell (x,y) coordinates belong to
   Int_t n = (Int_t)(floor((x-fXaxis.GetXmin())/fStepX));
   Int_t m = (Int_t)(floor((y-fYaxis.GetXmin())/fStepY));

   // Make sure the array indices are correct.
   if (n>=fCellX) n = fCellX-1;
   if (m>=fCellY) m = fCellY-1;
   if (n<0)       n = 0;
   if (m<0)       m = 0;

   if (fIsEmpty[n+fCellX*m]) return -5;

   TH2PolyBin *bin;

   TIter next(&fCells[n+fCellX*m]);
   TObject *obj;

   // Search for the bin in the cell
   while ((obj=next())) {
      bin  = (TH2PolyBin*)obj;
      if (bin->IsInside(x,y)) return bin->GetBinNumber();
   }

   // If the search has not returned a bin, the point must be on "the sea"
   return -5;
}


//______________________________________________________________________________
Int_t TH2Poly::Fill(Double_t x, Double_t y)
{
   // Increment the bin containing (x,y) by 1.
   // Uses the partitioning algorithm.

   return Fill(x, y, 1.0);
}


//______________________________________________________________________________
Int_t TH2Poly::Fill(Double_t x, Double_t y, Double_t w)
{
   // Increment the bin containing (x,y) by w.
   // Uses the partitioning algorithm.

   if (fNcells==0) return 0;
   Int_t overflow = 0;
   if      (y > fYaxis.GetXmax()) overflow += -1;
   else if (y > fYaxis.GetXmin()) overflow += -4;
   else                           overflow += -7;
   if      (x > fXaxis.GetXmax()) overflow += -2;
   else if(x > fXaxis.GetXmin())  overflow += -1;
   if (overflow != -5) {
      fOverflow[-overflow - 1]++;
      return overflow;
   }

   // Finds the cell (x,y) coordinates belong to
   Int_t n = (Int_t)(floor((x-fXaxis.GetXmin())/fStepX));
   Int_t m = (Int_t)(floor((y-fYaxis.GetXmin())/fStepY));

   // Make sure the array indices are correct.
   if (n>=fCellX) n = fCellX-1;
   if (m>=fCellY) m = fCellY-1;
   if (n<0)       n = 0;
   if (m<0)       m = 0;

   if (fIsEmpty[n+fCellX*m]) {
      fOverflow[4]++;
      return -5;
   }

   TH2PolyBin *bin;
   Int_t bi;

   TIter next(&fCells[n+fCellX*m]);
   TObject *obj;

   while ((obj=next())) {
      bin  = (TH2PolyBin*)obj;
      bi = bin->GetBinNumber()-1;
      if (bin->IsInside(x,y)) {
         bin->Fill(w);

         // Statistics
         fTsumw   = fTsumw + w;
         fTsumwx  = fTsumwx + w*x;
         fTsumwx2 = fTsumwx2 + w*x*x;
         fTsumwy  = fTsumwy + w*y;
         fTsumwy2 = fTsumwy2 + w*y*y;
         if (fSumw2.fN) fSumw2.fArray[bi] += w*w;
         fEntries++;

         SetBinContentChanged(kTRUE);

         return bin->GetBinNumber();
      }
   }

   fOverflow[4]++;
   return -5;
}


//______________________________________________________________________________
Int_t TH2Poly::Fill(const char* name, Double_t w)
{
   // Increment the bin named "name" by w.

   TString sname(name);

   TIter    next(fBins);
   TObject  *obj;
   TH2PolyBin *bin;

   while ((obj = next())) {
      bin = (TH2PolyBin*) obj;
      if (!sname.CompareTo(bin->GetPolygon()->GetName())) {
         bin->Fill(w);
         fEntries++;
         SetBinContentChanged(kTRUE);
         return bin->GetBinNumber();
      }
   }

   return 0;
}


//______________________________________________________________________________
void TH2Poly::FillN(Int_t ntimes, const Double_t* x, const Double_t* y,
                               const Double_t* w, Int_t stride)
{
   // Fills a 2-D histogram with an array of values and weights.
   //
   // ntimes:  number of entries in arrays x and w
   //          (array size must be ntimes*stride)
   // x:       array of x values to be histogrammed
   // y:       array of y values to be histogrammed
   // w:       array of weights
   // stride:  step size through arrays x, y and w

   for (int i = 0; i < ntimes; i += stride) {
      Fill(x[i], y[i], w[i]);
   }
}


//______________________________________________________________________________
Double_t TH2Poly::Integral(Option_t* option) const
{
   // Returns the integral of bin contents.
   // By default the integral is computed as the sum of bin contents.
   // If option "width" or "area" is specified, the integral is the sum of
   // the bin contents multiplied by the area of the bin.

   TString opt = option;
   opt.ToLower();

   if ((opt.Contains("width")) || (opt.Contains("area"))) {
      Double_t w;
      Double_t integral = 0.;

      TIter    next(fBins);
      TObject *obj;
      TH2PolyBin *bin;
      while ((obj=next())) {
         bin       = (TH2PolyBin*) obj;
         w         = bin->GetArea();
         integral += w*(bin->GetContent());
      }

      return integral;
   } else {
      return fTsumw;
   }
}


//______________________________________________________________________________
Double_t TH2Poly::GetBinContent(Int_t bin) const
{
   // Returns the content of the input bin
   // For the overflow/underflow/sea bins:
   //
   // -1 | -2 | -3
   // ---+----+----
   // -4 | -5 | -6
   // ---+----+----
   // -7 | -8 | -9
   //
   // where -5 is the "sea" bin (i.e. unbinned areas)

   if (bin > fNcells || bin == 0 || bin < -9) return 0; 
   if (bin<0) return fOverflow[-bin - 1];
   return ((TH2PolyBin*) fBins->At(bin-1))->GetContent();
}

//______________________________________________________________________________
Double_t TH2Poly::GetBinError(Int_t bin) const
{
   // Returns the value of error associated to bin number bin.
   // If the sum of squares of weights has been defined (via Sumw2),
   // this function returns the sqrt(sum of w2).
   // otherwise it returns the sqrt(contents) for this bin.

   if (bin < 0) bin = 0;
   if (bin > (fNcells)) return 0;
   if (fBuffer) ((TH1*)this)->BufferEmpty();
   if (fSumw2.fN) {
      Double_t err2 = fSumw2.fArray[bin-1];
      return TMath::Sqrt(err2);
   }
   Double_t error2 = TMath::Abs(GetBinContent(bin));
   return TMath::Sqrt(error2);
}


//______________________________________________________________________________
const char *TH2Poly::GetBinName(Int_t bin) const
{
   // Returns the bin name.

   if (bin > (fNcells))  return "";
   if (bin < 0)          return "";
   return ((TH2PolyBin*) fBins->At(bin-1))->GetPolygon()->GetName();
}


//______________________________________________________________________________
const char *TH2Poly::GetBinTitle(Int_t bin) const
{
   // Returns the bin title.

   if (bin > (fNcells))  return "";
   if (bin < 0)          return "";
   return ((TH2PolyBin*) fBins->At(bin-1))->GetPolygon()->GetTitle();
}


//______________________________________________________________________________
Double_t TH2Poly::GetMaximum() const
{
   // Returns the maximum value of the histogram.

   if (fNcells==0) return 0;
   if (fMaximum != -1111) return fMaximum;

   TH2PolyBin  *b;

   TIter next(fBins);
   TObject *obj;
   Double_t max,c;

   max = ((TH2PolyBin*) next())->GetContent();

   while ((obj=next())) {
      b = (TH2PolyBin*)obj;
      c = b->GetContent();
      if (c>max) max = c;
   }
   return max;
}


//______________________________________________________________________________
Double_t TH2Poly::GetMaximum(Double_t maxval) const
{
   // Returns the maximum value of the histogram that is less than maxval.

   if (fNcells==0) return 0;
   if (fMaximum != -1111) return fMaximum;

   TH2PolyBin  *b;

   TIter next(fBins);
   TObject *obj;
   Double_t max,c;

   max = ((TH2PolyBin*) next())->GetContent();

   while ((obj=next())) {
      b = (TH2PolyBin*)obj;
      c = b->GetContent();
      if (c>max && c<maxval) max=c;
   }
   return max;
}


//______________________________________________________________________________
Double_t TH2Poly::GetMinimum() const
{
   // Returns the minimum value of the histogram.

   if (fNcells==0) return 0;
   if (fMinimum != -1111) return fMinimum;

   TH2PolyBin  *b;

   TIter next(fBins);
   TObject *obj;
   Double_t min,c;

   min = ((TH2PolyBin*) next())->GetContent();

   while ((obj=next())) {
      b = (TH2PolyBin*)obj;
      c = b->GetContent();
      if (c<min) min=c;
   }
   return min;
}


//______________________________________________________________________________
Double_t TH2Poly::GetMinimum(Double_t minval) const
{
   // Returns the minimum value of the histogram that is greater than minval.

   if (fNcells==0) return 0;
   if (fMinimum != -1111) return fMinimum;

   TH2PolyBin  *b;

   TIter next(fBins);
   TObject *obj;
   Double_t min,c;

   min = ((TH2PolyBin*) next())->GetContent();

   while ((obj=next())) {
      b = (TH2PolyBin*)obj;
      c = b->GetContent();
      if (c<min && c>minval) min=c;
   }
   return min;
}


//______________________________________________________________________________
void TH2Poly::Honeycomb(Double_t xstart, Double_t ystart, Double_t a,
                     Int_t k, Int_t s)
{
   // Bins the histogram using a honeycomb structure

   // Add the bins
   Double_t numberOfHexagonsInTheRow;
   Double_t x[6], y[6];
   Double_t xloop, yloop, xtemp;
   xloop = xstart; yloop = ystart + a/2.0;
   for (int sCounter = 0; sCounter < s; sCounter++) {

      xtemp = xloop; // Resets the temp variable

      // Determine the number of hexagons in that row
      if(sCounter%2 == 0){numberOfHexagonsInTheRow = k;}
      else{numberOfHexagonsInTheRow = k - 1;}

      for (int kCounter = 0; kCounter <  numberOfHexagonsInTheRow; kCounter++) {

         // Go around the hexagon
         x[0] = xtemp;
         y[0] = yloop;
         x[1] = x[0];
         y[1] = y[0] + a;
         x[2] = x[1] + a*TMath::Sqrt(3)/2.0;
         y[2] = y[1] + a/2.0;
         x[3] = x[2] + a*TMath::Sqrt(3)/2.0;
         y[3] = y[1];
         x[4] = x[3];
         y[4] = y[0];
         x[5] = x[2];
         y[5] = y[4] - a/2.0;

         this->AddBin(6, x, y);

         // Go right
         xtemp += a*TMath::Sqrt(3);
      }

      // Increment the starting position
      if (sCounter%2 == 0) xloop += a*TMath::Sqrt(3)/2.0;
      else                 xloop -= a*TMath::Sqrt(3)/2.0;
      yloop += 1.5*a;
   }
}


//______________________________________________________________________________
void TH2Poly::Initialize(Double_t xlow, Double_t xup,
                      Double_t ylow, Double_t yup, Int_t n, Int_t m)
{
   // Initializes the TH2Poly object.  This method is called by the constructor.

   Int_t i;
   fDimension = 2;  //The dimesion of the histogram

   fBins   = 0;
   fNcells = 0;

   // Sets the boundaries of the histogram
   fXaxis.Set(100, xlow, xup);
   fYaxis.Set(100, ylow, yup);

   for (i=0; i<9; i++) fOverflow[i] = 0.;

   // Statistics
   fEntries = 0;   // The total number of entries
   fTsumw   = 0.;  // Total amount of content in the histogram
   fTsumwx  = 0.;  // Weighted sum of x coordinates
   fTsumwx2 = 0.;  // Weighted sum of the squares of x coordinates
   fTsumwy2 = 0.;  // Weighted sum of the squares of y coordinates
   fTsumwy  = 0.;  // Weighted sum of y coordinates

   fCellX = n; // Set the number of cells to default
   fCellY = m; // Set the number of cells to default

   // number of cells in the grid
   //N.B. not to be confused with fNcells (the number of bins) ! 
   fNCells = fCellX*fCellY;
   fCells  = new TList [fNCells];  // Sets an empty partition
   fStepX  = (fXaxis.GetXmax() - fXaxis.GetXmin())/fCellX; // Cell width
   fStepY  = (fYaxis.GetXmax() - fYaxis.GetXmin())/fCellY; // Cell height

   fIsEmpty = new Bool_t [fNCells]; // Empty partition
   fCompletelyInside = new Bool_t [fNCells]; // Cell is completely inside bin

   for (i = 0; i<fNCells; i++) {   // Initializes the flags
      fIsEmpty[i] = kTRUE;
      fCompletelyInside[i] = kFALSE;
   }

   // 3D Painter flags
   SetNewBinAdded(kFALSE);
   SetBinContentChanged(kFALSE);
}


//______________________________________________________________________________
Bool_t TH2Poly::IsIntersecting(TH2PolyBin *bin,
                               Double_t xclipl, Double_t xclipr,
                               Double_t yclipb, Double_t yclipt)
{
   // Returns kTRUE if the input bin is intersecting with the
   // input rectangle (xclipl, xclipr, yclipb, yclipt)

   Int_t     gn;
   Double_t *gx;
   Double_t *gy;
   Bool_t inter = kFALSE;
   TObject *poly = bin->GetPolygon();

   if (poly->IsA() == TGraph::Class()) {
      TGraph *g = (TGraph*)poly;
      gx = g->GetX();
      gy = g->GetY();
      gn = g->GetN();
      inter = IsIntersectingPolygon(gn, gx, gy, xclipl, xclipr, yclipb, yclipt);
   }

   if (poly->IsA() == TMultiGraph::Class()) {
      TMultiGraph *mg = (TMultiGraph*)poly;
      TList *gl = mg->GetListOfGraphs();
      if (!gl) return inter;
      TGraph *g;
      TIter next(gl);
      while ((g = (TGraph*) next())) {
         gx = g->GetX();
         gy = g->GetY();
         gn = g->GetN();
         inter = IsIntersectingPolygon(gn, gx, gy, xclipl, xclipr,
                                                   yclipb, yclipt);
         if (inter) return inter;
      }
   }

   return inter;
}

//______________________________________________________________________________
Bool_t TH2Poly::IsIntersectingPolygon(Int_t bn, Double_t *x, Double_t *y,
                                      Double_t xclipl, Double_t xclipr,
                                      Double_t yclipb, Double_t yclipt)
{
   // Returns kTRUE if the input polygon (bn, x, y) is intersecting with the
   // input rectangle (xclipl, xclipr, yclipb, yclipt)

   Bool_t p0R, p0L, p0T, p0B, p0xM, p0yM, p1R, p1L, p1T;
   Bool_t p1B, p1xM, p1yM, p0In, p1In;

   for (int counter = 0; counter < (bn-1); counter++) {
      // If both are on the same side, return kFALSE
      p0L = x[counter]     <= xclipl; // Point 0 is on the left
      p1L = x[counter + 1] <= xclipl; // Point 1 is on the left
      if (p0L && p1L) continue;
      p0R = x[counter]     >= xclipr; // Point 0 is on the right
      p1R = x[counter + 1] >= xclipr; // Point 1 is on the right
      if (p0R && p1R) continue;
      p0T = y[counter]     >= yclipt; // Point 0 is at the top
      p1T = y[counter + 1] >= yclipt; // Point 1 is at the top
      if (p0T && p1T) continue;
      p0B = y[counter]     <= yclipb; // Point 0 is at the bottom
      p1B = y[counter + 1] <= yclipb; // Point 1 is at the bottom
      if (p0B && p1B) continue;

      // Checks to see if any are inside
      p0xM = !p0R && !p0L; // Point 0 is inside along x
      p0yM = !p0T && !p0B; // Point 1 is inside along x
      p1xM = !p1R && !p1L; // Point 0 is inside along y
      p1yM = !p1T && !p1B; // Point 1 is inside along y
      p0In = p0xM && p0yM; // Point 0 is inside
      p1In = p1xM && p1yM; // Point 1 is inside
      if (p0In) {
         if (p1In) continue;
         return kTRUE;
      } else {
         if (p1In) return kTRUE;
      }

      // We know by now that the points are not in the same side and not inside.

      // Checks to see if they are opposite

      if (p0xM && p1xM) return kTRUE;
      if (p0yM && p1yM) return kTRUE;

      // We now know that the points are in different x and y indices

      Double_t xcoord[3], ycoord[3];
      xcoord[0] = x[counter];
      xcoord[1] = x[counter + 1];
      ycoord[0] = y[counter];
      ycoord[1] = y[counter + 1];

      if (p0L) {
         if(p1T){
            xcoord[2] = xclipl;
            ycoord[2] = yclipb;
            if((TMath::IsInside(xclipl, yclipt, 3, xcoord, ycoord)) ||
               (TMath::IsInside(xclipr, yclipb, 3, xcoord, ycoord))) continue;
            else return kTRUE;
         } else if (p1B) {
            xcoord[2] = xclipl;
            ycoord[2] = yclipt;
            if((TMath::IsInside(xclipl, yclipb, 3, xcoord, ycoord)) ||
               (TMath::IsInside(xclipr, yclipt, 3, xcoord, ycoord))) continue;
            else return kTRUE;
         } else { // p1yM
            if (p0T) {
               xcoord[2] = xclipl;
               ycoord[2] = yclipb;
               if (TMath::IsInside(xclipr, yclipt, 3, xcoord, ycoord)) continue;
               else return kTRUE;
            }
            if (p0B) {
               xcoord[2] = xclipl;
               ycoord[2] = yclipt;
               if (TMath::IsInside(xclipr, yclipb, 3, xcoord, ycoord)) continue;
               else return kTRUE;
            }
         }
      } else if (p0R) {
         if (p1T) {
            xcoord[2] = xclipl;
            ycoord[2] = yclipb;
            if ((TMath::IsInside(xclipr, yclipb, 3, xcoord, ycoord)) ||
                (TMath::IsInside(xclipl, yclipt, 3, xcoord, ycoord))) continue;
            else return kTRUE;
         } else if (p1B) {
            xcoord[2] = xclipl;
            ycoord[2] = yclipt;
            if ((TMath::IsInside(xclipl, yclipb, 3, xcoord, ycoord)) ||
                (TMath::IsInside(xclipr, yclipt, 3, xcoord, ycoord))) continue;
            else return kTRUE;
         } else{ // p1yM
            if (p0T) {
               xcoord[2] = xclipr;
               ycoord[2] = yclipb;
               if (TMath::IsInside(xclipl, yclipt, 3, xcoord, ycoord)) continue;
               else return kTRUE;
            }
            if (p0B) {
               xcoord[2] = xclipr;
               ycoord[2] = yclipt;
               if (TMath::IsInside(xclipl, yclipb, 3, xcoord, ycoord)) continue;
               else return kTRUE;
            }
         }
      }
   }
   return kFALSE;
}


//______________________________________________________________________________
Long64_t TH2Poly::Merge(TCollection *)
{
   Error("Merge","Cannot merge TH2Poly");
   return 0;
}


//______________________________________________________________________________
void TH2Poly::SavePrimitive(ostream &out, Option_t *option)
{
   // Save primitive as a C++ statement(s) on output stream out

   out <<"   "<<endl;
   out <<"   "<< ClassName() <<" *";

   //histogram pointer has by default the histogram name.
   //however, in case histogram has no directory, it is safer to add a
   //incremental suffix
   static Int_t hcounter = 0;
   TString histName = GetName();
   if (!fDirectory && !histName.Contains("Graph")) {
      hcounter++;
      histName += "__";
      histName += hcounter;
   }
   const char *hname = histName.Data();

   //Construct the class initialization
   out << hname << " = new " << ClassName() << "(\"" << hname << "\", \""
       << GetTitle() << "\", " << fCellX << ", " << fXaxis.GetXmin()
       << ", " << fXaxis.GetXmax()
       << ", " << fCellY << ", " << fYaxis.GetXmin() << ", "
       << fYaxis.GetXmax() << ");" << endl;

   // Save Bins
   TIter       next(fBins);
   TObject    *obj;
   TH2PolyBin *th2pBin;

   while((obj = next())){
      th2pBin = (TH2PolyBin*) obj;
      th2pBin->GetPolygon()->SavePrimitive(out,
                                           Form("th2poly%s",histName.Data()));
   }

   // save bin contents
   out<<"   "<<endl;
   Int_t bin;
   for (bin=1;bin<=fNcells;bin++) {
      Double_t bc = GetBinContent(bin);
      if (bc) {
         out<<"   "<<hname<<"->SetBinContent("<<bin<<","<<bc<<");"<<endl;
      }
   }

   // save bin errors
   if (fSumw2.fN) {
      for (bin=1;bin<=fNcells;bin++) {
         Double_t be = GetBinError(bin);
         if (be) {
            out<<"   "<<hname<<"->SetBinError("<<bin<<","<<be<<");"<<endl;
         }
      }
   }
   TH1::SavePrimitiveHelp(out, hname, option);
}


//______________________________________________________________________________
void TH2Poly::Scale(Double_t c1, Option_t*)
{
   // Multiply this histogram by a constant c1.

   for( int i = 0; i < this->GetNumberOfBins(); i++ ) {
      this->SetBinContent(i+1, c1*this->GetBinContent(i+1));
   }
}


//______________________________________________________________________________
void TH2Poly::SetBinContent(Int_t bin, Double_t content)
{
   // Sets the contents of the input bin to the input content
   // Negative values between -1 and -9 are for the overflows and the sea 

   if (bin > (fNcells) || bin == 0 || bin < -9 ) return;
   if (bin > 0) 
      ((TH2PolyBin*) fBins->At(bin-1))->SetContent(content);
   else
      fOverflow[-bin - 1] += content;
   SetBinContentChanged(kTRUE);
}

//______________________________________________________________________________
void TH2Poly::SetFloat(Bool_t flag)
{
   // When set to kTRUE, allows the histogram to expand if a bin outside the
   // limits is added.

   fFloat = flag;
}


//______________________________________________________________________________
TH2PolyBin::TH2PolyBin()
{
   // Default constructor.

   fPoly    = 0;
   fContent = 0.;
   fNumber  = 0;
   fXmax    = -1111;
   fXmin    = -1111;
   fYmax    = -1111;
   fYmin    = -1111;
   fArea    = 0;
   SetChanged(kTRUE);
}


//______________________________________________________________________________
TH2PolyBin::TH2PolyBin(TObject *poly, Int_t bin_number)
{
   // Normal constructor.

   fContent = 0.;
   fNumber  = bin_number;
   fArea    = 0.;
   fPoly    = poly;
   fXmax    = -1111;
   fXmin    = -1111;
   fYmax    = -1111;
   fYmin    = -1111;
   SetChanged(kTRUE);
}


//______________________________________________________________________________
TH2PolyBin::~TH2PolyBin()
{
   // Destructor.

   if (fPoly) delete fPoly;
}


//______________________________________________________________________________
Double_t TH2PolyBin::GetArea()
{
   // Returns the area of the bin.

   Int_t     bn;

   if (fArea == 0) {
      if (fPoly->IsA() == TGraph::Class()) {
         TGraph *g = (TGraph*)fPoly;
         bn    = g->GetN();
         fArea = g->Integral(0,bn-1);
      }

      if (fPoly->IsA() == TMultiGraph::Class()) {
         TMultiGraph *mg = (TMultiGraph*)fPoly;
         TList *gl = mg->GetListOfGraphs();
         if (!gl) return fArea;
         TGraph *g;
         TIter next(gl);
         while ((g = (TGraph*) next())) {
            bn    = g->GetN();
            fArea = fArea + g->Integral(0,bn-1);
         }
      }
   }

   return fArea;
}


//______________________________________________________________________________
Double_t TH2PolyBin::GetXMax()
{
   // Returns the maximum value for the x coordinates of the bin.

   if (fXmax != -1111) return fXmax;

   Int_t     bn,i;
   Double_t *bx;

   if (fPoly->IsA() == TGraph::Class()) {
      TGraph *g = (TGraph*)fPoly;
      bx    = g->GetX();
      bn    = g->GetN();
      fXmax = bx[0];
      for (i=1; i<bn; i++) {if (fXmax < bx[i]) fXmax = bx[i];}
   }

   if (fPoly->IsA() == TMultiGraph::Class()) {
      TMultiGraph *mg = (TMultiGraph*)fPoly;
      TList *gl = mg->GetListOfGraphs();
      if (!gl) return fXmax;
      TGraph *g;
      TIter next(gl);
      Bool_t first = kTRUE;
      while ((g = (TGraph*) next())) {
         bx = g->GetX();
         bn = g->GetN();
         if (first) {fXmax = bx[0]; first = kFALSE;}
         for (i=0; i<bn; i++) {if (fXmax < bx[i]) fXmax = bx[i];}
      }
   }

   return fXmax;
}


//______________________________________________________________________________
Double_t TH2PolyBin::GetXMin()
{
   // Returns the minimum value for the x coordinates of the bin.

   if (fXmin != -1111) return fXmin;

   Int_t     bn,i;
   Double_t *bx;

   if (fPoly->IsA() == TGraph::Class()) {
      TGraph *g = (TGraph*)fPoly;
      bx    = g->GetX();
      bn    = g->GetN();
      fXmin = bx[0];
      for (i=1; i<bn; i++) {if (fXmin > bx[i]) fXmin = bx[i];}
   }

   if (fPoly->IsA() == TMultiGraph::Class()) {
      TMultiGraph *mg = (TMultiGraph*)fPoly;
      TList *gl = mg->GetListOfGraphs();
      if (!gl) return fXmin;
      TGraph *g;
      TIter next(gl);
      Bool_t first = kTRUE;
      while ((g = (TGraph*) next())) {
         bx = g->GetX();
         bn = g->GetN();
         if (first) {fXmin = bx[0]; first = kFALSE;}
         for (i=0; i<bn; i++) {if (fXmin > bx[i]) fXmin = bx[i];}
      }
   }

   return fXmin;
}


//______________________________________________________________________________
Double_t TH2PolyBin::GetYMax()
{
   // Returns the maximum value for the y coordinates of the bin.

   if (fYmax != -1111) return fYmax;

   Int_t     bn,i;
   Double_t *by;

   if (fPoly->IsA() == TGraph::Class()) {
      TGraph *g = (TGraph*)fPoly;
      by    = g->GetY();
      bn    = g->GetN();
      fYmax = by[0];
      for (i=1; i<bn; i++) {if (fYmax < by[i]) fYmax = by[i];}
   }

   if (fPoly->IsA() == TMultiGraph::Class()) {
      TMultiGraph *mg = (TMultiGraph*)fPoly;
      TList *gl = mg->GetListOfGraphs();
      if (!gl) return fYmax;
      TGraph *g;
      TIter next(gl);
      Bool_t first = kTRUE;
      while ((g = (TGraph*) next())) {
         by = g->GetY();
         bn = g->GetN();
         if (first) {fYmax = by[0]; first = kFALSE;}
         for (i=0; i<bn; i++) {if (fYmax < by[i]) fYmax = by[i];}
      }
   }

   return fYmax;
}


//______________________________________________________________________________
Double_t TH2PolyBin::GetYMin()
{
   // Returns the minimum value for the y coordinates of the bin.

   if (fYmin != -1111) return fYmin;

   Int_t     bn,i;
   Double_t *by;

   if (fPoly->IsA() == TGraph::Class()) {
      TGraph *g = (TGraph*)fPoly;
      by    = g->GetY();
      bn    = g->GetN();
      fYmin = by[0];
      for (i=1; i<bn; i++) {if (fYmin > by[i]) fYmin = by[i];}
   }

   if (fPoly->IsA() == TMultiGraph::Class()) {
      TMultiGraph *mg = (TMultiGraph*)fPoly;
      TList *gl = mg->GetListOfGraphs();
      if (!gl) return fYmin;
      TGraph *g;
      TIter next(gl);
      Bool_t first = kTRUE;
      while ((g = (TGraph*) next())) {
         by = g->GetY();
         bn = g->GetN();
         if (first) {fYmin = by[0]; first = kFALSE;}
         for (i=0; i<bn; i++) {if (fYmin > by[i]) fYmin = by[i];}
      }
   }

   return fYmin;
}


//______________________________________________________________________________
Bool_t TH2PolyBin::IsInside(Double_t x, Double_t y) const
{
   // Return "true" if the point (x,y) is inside the bin.

   Int_t in=0;

   if (fPoly->IsA() == TGraph::Class()) {
      TGraph *g = (TGraph*)fPoly;
      in = g->IsInside(x, y);
   }

   if (fPoly->IsA() == TMultiGraph::Class()) {
      TMultiGraph *mg = (TMultiGraph*)fPoly;
      in = mg->IsInside(x, y);
   }

   return in;
}
