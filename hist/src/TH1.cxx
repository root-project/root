// @(#)root/hist:$Name:  $:$Id: TH1.cxx,v 1.61 2001/10/18 10:03:08 brun Exp $
// Author: Rene Brun   26/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <fstream.h>
#include <iostream.h>
#include <float.h>

#include "TROOT.h"
#include "TH1.h"
#include "TH2.h"
#include "TF2.h"
#include "TF3.h"
#include "TVirtualPad.h"
#include "Foption.h"
#include "TMath.h"
#include "TRandom.h"
#include "TVirtualFitter.h"
#include "TProfile.h"
#include "TStyle.h"
#include "TVector.h"
#include "TVectorD.h"

//______________________________________________________________________________
//*-*-*-*-*-*-*-*-*-*-*The H I S T O G R A M   Classes*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===============================
//*-*
//*-*  ROOT supports the following histogram types:
//*-*
//*-*   1-D histograms:
//*-*      TH1C : histograms with one byte per channel.   Maximum bin content = 255
//*-*      TH1S : histograms with one short per channel.  Maximum bin content = 65535
//*-*      TH1F : histograms with one float per channel.  Maximum precision 7 digits
//*-*      TH1D : histograms with one double per channel. Maximum precision 14 digits
//*-*
//*-*   2-D histograms:
//*-*      TH2C : histograms with one byte per channel.   Maximum bin content = 255
//*-*      TH2S : histograms with one short per channel.  Maximum bin content = 65535
//*-*      TH2F : histograms with one float per channel.  Maximum precision 7 digits
//*-*      TH2D : histograms with one double per channel. Maximum precision 14 digits
//*-*
//*-*   3-D histograms:
//*-*      TH3C : histograms with one byte per channel.   Maximum bin content = 255
//*-*      TH3S : histograms with one short per channel.  Maximum bin content = 65535
//*-*      TH3F : histograms with one float per channel.  Maximum precision 7 digits
//*-*      TH3D : histograms with one double per channel. Maximum precision 14 digits
//*-*
//*-*   Profile histograms: See classes  TProfile and TProfile2D
//*-*   Profile histograms are used to display the mean value of Y and its RMS
//*-*   for each bin in X. Profile histograms are in many cases an elegant
//*-*   replacement of two-dimensional histograms : the inter-relation of two
//*-*   measured quantities X and Y can always be visualized by a two-dimensional
//*-*   histogram or scatter-plot; If Y is an unknown (but single-valued)
//*-*   approximate function of X, this function is displayed by a profile
//*-*   histogram with much better precision than by a scatter-plot.
///*-*
//*-*- All histogram classes are derived from the base class TH1
//*-*
//*-*                             TH1
//*-*                              ^
//*-*                              |
//*-*                              |
//*-*                              |
//*-*      -----------------------------------------------------
//*-*             |                |       |      |      |     |
//*-*             |                |      TH1C   TH1S   TH1F  TH1D
//*-*             |                |                           |
//*-*             |                |                           |
//*-*             |               TH2                      TProfile
//*-*             |                |
//*-*             |                |
//*-*             |                -----------------------------
//*-*             |                        |      |      |     |
//*-*             |                       TH2C   TH2S   TH2F  TH2D
//*-*             |                                            |
//*-*            TH3                                           |
//*-*             |                                        TProfile2D
//*-*             |
//*-*             ------------------------------
//*-*                     |      |      |      |
//*-*                    TH3C   TH3S   TH3F   TH3D
//*-*
//*-*   The TH*C classes also inherit from the array class TArrayC.
//*-*   The TH*S classes also inherit from the array class TArrayS.
//*-*   The TH*F classes also inherit from the array class TArrayF.
//*-*   The TH*D classes also inherit from the array class TArrayD.
//*-*
//*-*  Creating histograms
//*-*  ===================
//*-*  Histograms are created by invoking one of the constructors, eg
//*-*    TH1F *h1 = new TH1F("h1","h1 title",100,0,4.4);
//*-*    TH2F *h2 = new TH2F("h2","h2 title",40,0,4,30,-3,3);
//*-*  histograms may also be created by:
//*-*    - calling the Clone function, see below
//*-*    - making a projection from a 2-D or 3-D histogram, see below
//*-*    - reading an histogram from a file
//*-*  When an histogram is created, a reference to it is automatically added
//*-*  to the list of in-memory objects for the current file or directory.
//*-*  This default behaviour can be changed by:
//*-*    h->SetDirectory(0);         // for the current histogram h
//*-*    TH1::AddDirectory(kFALSE);  // sets a global switch disabling the reference
//*-*  When the histogram is deleted, the reference to it is removed from
//*-*  the list of objects in memory.
//*-*  When a file is closed, all histograms in memory associated with this file
//*-*  are automatically deleted.
//*-*
//*-*   Fix or variable bin size
//*-*   ========================
//*-*
//*-*  All histogram types support either fix or variable bin sizes.
//*-*  2-D histograms may have fix size bins along X and variable size bins
//*-*  along Y or vice-versa. The functions to fill, manipulate, draw or access
//*-*  histograms are identical in both cases.
//*-*  Each histogram always contains 3 objects TAxis: fXaxis, fYaxis and fZaxis
//*-*  To access the axis parameters, do:
//*-*     TAxis *xaxis = h->GetXaxis(); etc.
//*-*     Double_t binCenter = xaxis->GetBinCenter(bin), etc.
//*-*  See class TAxis for a description of all the access functions.
//*-*  The axis range is always stored internally in double precision.
//*-*
//*-*   Convention for numbering bins
//*-*   =============================
//*-*   For all histogram types: nbins, xlow, xup
//*-*     bin = 0;       underflow bin
//*-*     bin = 1;       first bin with low-edge xlow INCLUDED
//*-*     bin = nbins;   last bin with upper-edge xup EXCLUDED
//*-*     bin = nbins+1; overflow bin
//*-*   In case of 2-D or 3-D histograms, a "global bin" number is defined.
//*-*   For example, assuming a 3-D histogram with binx,biny,binz, the function
//*-*     Int_t bin = h->GetBin(binx,biny,binz);
//*-*   returns a global/linearized bin number. This global bin is useful
//*-*   to access the bin information independently of the dimension.
//*-*
//*-*  Filling histograms
//*-*  ==================
//*-*  An histogram is typically filled with statements like:
//*-*    h1->Fill(x);
//*-*    h1->Fill(x,w); //fill with weight
//*-*    h2->Fill(x,y)
//*-*    h2->Fill(x,y,w)
//*-*    h3->Fill(x,y,z)
//*-*    h3->Fill(x,y,z,w)
//*-*  The Fill functions compute the bin number corresponding to the given
//*-*  x,y or z argument and increment this bin by the given weight.
//*-*  The Fill functions return the bin number for 1-D histograms or global
//*-*  bin number for 2-D and 3-D histograms.
//*-*  If TH1::Sumw2 has been called before filling, the sum of squares of
//*-*  weights is also stored.
//*-*  One can also increment directly a bin number via TH1::AddBinContent
//*-*  or replace the existing content via TH1::SetBinContent.
//*-*  To access the bin content of a given bin, do:
//*-*    Double_t binContent = h->GetBinContent(bin);
//*-*
//*-*  By default, the bin number is computed using the current axis ranges.
//*-*  If the automatic binning option has been set via
//*-*         h->SetBit(TH1::kCanRebin);
//*-*  then, the Fill Function will automatically extend the axis range to
//*-*  accomodate the new value specified in the Fill argument. The method
//*-*  used is to double the bin size until the new value fits in the range,
//*-*  merging bins two by two. This automatic binning options is extensively
//*-*  used by the TTree::Draw function when histogramming Tree variables
//*-*  with an unknown range.
//*-*  This automatic binning option is supported for 1-d, 2-D and 3-D histograms.
//*-*
//*-*  During filling, some statistics parameters are incremented to compute
//*-*  the mean value and Root Mean Square with the maximum precision.
//*-*
//*-*  In case of histograms of type TH1C, TH1S, TH2C, TH2S, TH3C, TH3S
//*-*  a check is made that the bin contents do not exceed the maximum positive
//*-*  capacity (127 or 65535). Histograms of all types may have positive
//*-*  or/and negative bin contents.
//*-*
//*-*  Rebinning
//*-*  =========
//*-*  At any time, an histogram can be rebinned via TH1::Rebin. This function
//*-*  returns a new histogram with the rebinned contents.
//*-*  If bin errors were stored, they are recomputed during the rebinning.
//*-*
//*-*  Associated errors
//*-*  =================
//*-*  By default, for each bin, the sum of weights is computed at fill time.
//*-*  One can also call TH1::Sumw2 to force the storage and computation
//*-*  of the sum of the square of weights per bin.
//*-*  If Sumw2 has been called, the error per bin is computed as the
//*-*  sqrt(sum of squares of weights), otherwise the error is set equal
//*-*  to the sqrt(bin content).
//*-*  To return the error for a given bin number, do:
//*-*     Double_t error = h->GetBinError(bin);
//*-*
//*-*  Associated functions
//*-*  ====================
//*-*  One or more object (typically a TF1*) can be added to the list
//*-*  of functions (fFunctions) associated to each histogram.
//*-*  When TH1::Fit is invoked, the fitted function is added to this list.
//*-*  Given an histogram h, one can retrieve an associated function
//*-*  with:  TF1 *myfunc = h->GetFunction("myfunc");
//*-*
//*-*  Operations on histograms
//*-*  ========================
//*-*
//*-*  Many types of operations are supported on histograms or between histograms
//*-*  - Addition of an histogram to the current histogram
//*-*  - Additions of two histograms with coefficients and storage into the current
//*-*    histogram
//*-*  - Multiplications and Divisions are supported in the same way as additions.
//*-*  - The Add, Divide and Multiply functions also exist to add,divide or multiply
//*-*    an histogram by a function.
//*-*  If an histogram has associated error bars (TH1::Sumw2 has been called),
//*-*  the resulting error bars are also computed assuming independent histograms.
//*-*  In case of divisions, Binomial errors are also supported.
//*-*
//*-*
//*-*  Fitting histograms
//*-*  ==================
//*-*
//*-*  Histograms (1-D,2-D,3-D and Profiles) can be fitted with a user
//*-*  specified function via TH1::Fit. When an histogram is fitted, the
//*-*  resulting function with its parameters is added to the list of functions
//*-*  of this histogram. If the histogram is made persistent, the list of
//*-*  associated functions is also persistent. Given a pointer (see above)
//*-*  to an associated function myfunc, one can retrieve the function/fit
//*-*  parameters with calls such as:
//*-*    Double_t chi2 = myfunc->GetChisquare();
//*-*    Double_t par0 = myfunc->GetParameter(0); //value of 1st parameter
//*-*    Double_t err0 = myfunc->GetParError(0);  //error on first parameter
//*-*
//*-*
//*-*  Projections of histograms
//*-*  ========================
//*-*  One can:
//*-*   - make a 1-D projection of a 2-D histogram or Profile
//*-*     see functions TH2::ProjectionX,Y, TH2::ProfileX,Y, TProfile::ProjectionX
//*-*   - make a 1-D, 2-D or profile out of a 3-D histogram
//*-*     see functions TH3::ProjectionZ, TH3::Project3D.
//*-*
//*-*  One can fit these projections via:
//*-*   TH2::FitSlicesX,Y, TH3::FitSlicesZ.
//*-*
//*-*  Random Numbers and histograms
//*-*  =============================
//*-*  TH1::FillRandom can be used to randomly fill an histogram using
//*-*                 the contents of an existing TF1 function or another
//*-*                 TH1 histogram (for all dimensions).
//*-*  For example the following two statements create and fill an histogram
//*-*  10000 times with a default gaussian distribution of mean 0 and sigma 1:
//*-*    TH1F h1("h1","histo from a gaussian",100,-3,3);
//*-*    h1.FillRandom("gaus",10000);
//*-*  TH1::GetRandom can be used to return a random number distributed
//*-*                 according the contents of an histogram.
//*-*
//*-*  Making a copy of an histogram
//*-*  =============================
//*-*  Like for any other Root object derived from TObject, one can use
//*-*  the Clone function. This makes an identical copy of the original histogram
//*-*  including all associated errors and functions, eg:
//*-*    TH1F *hnew = (TH1F*)h->Clone();
//*-*    hnew->SetName("hnew"); //recommended, otherwise you get 2 histograms
//*-*                           //with the same name
//*-*
//*-*  Normalizing histograms
//*-*  ======================
//*-*  One can scale an histogram such that the bins integral is equal to
//*-*  to the normalization parameter via TH1::Scale(Double_t norm);
//*-*
//*-*  Drawing histograms
//*-*  ==================
//*-*  Histograms are drawn via the THistPainter class. Each histogram has
//*-*  a pointer to its own painter (to be usable in a multithreaded program).
//*-*  Many drawing options are supported.
//*-*  See THistPainter::Paint for more details.
//*-*  The same histogram can be drawn with different options in different pads.
//*-*  When an histogram drawn in a pad is deleted, the histogram is
//*-*  automatically removed from the pad or pads where it was drawn.
//*-*  If an histogram is drawn in a pad, then filled again, the new status
//*-*  of the histogram will be automatically shown in the pad next time
//*-*  the pad is updated. One does not need to redraw the histogram.
//*-*  To draw the current version of an histogram in a pad, one can use
//*-*     h->DrawCopy();
//*-*  This makes a clone (see Clone below) of the histogram. Once the clone
//*-*  is drawn, the original histogram may be modified or deleted without
//*-*  affecting the aspect of the clone.
//*-*
//*-*  One can use TH1::SetMaximum and TH1::SetMinimum to force a particular
//*-*  value for the maximum or the minimum scale on the plot.
//*-*
//*-*  TH1::UseCurrentStyle can be used to change all histogram graphics
//*-*  attributes to correspond to the current selected style.
//*-*  This function must be called for each histogram.
//*-*  In case one reads and draws many histograms from a file, one can force
//*-*  the histograms to inherit automatically the current graphics style
//*-*  by calling before gROOT->ForceStyle();
//*-*
//*-*
//*-*  Setting Drawing histogram contour levels (2-D hists only)
//*-*  =========================================================
//*-*  By default contours are automatically generated at equidistant
//*-*  intervals. A default value of 20 levels is used. This can be modified
//*-*  via TH1::SetContour or TH1::SetContourLevel.
//*-*  the contours level info is used by the drawing options "cont", "surf",
//*-*  and "lego".
//*-*
//*-*  Setting histogram graphics attributes
//*-*  =====================================
//*-*  The histogram classes inherit from the attribute classes:
//*-*    TAttLine, TAttFill, TAttMarker and TAttText.
//*-*  See the member functions of these classes for the list of options.
//*-*
//*-*  Giving titles to the X, Y and Z axis
//*-*  =================================
//*-*    h->GetXaxis()->SetTitle("X axis title");
//*-*    h->GetYaxis()->SetTitle("Y axis title");
//*-*  The histogram title and the axis titles can be any TLatex string.
//*-*  The titles are part of the persistent histogram.
//*-*
//*-*  Saving/Reading histograms to/from a Root file
//*-*  ================================
//*-*  The following statements create a Root file and store an histogram
//*-*  on the file. Because TH1 derives from TNamed, the key identifier on
//*-*  the file is the histogram name:
//*-*     TFile f("histos.root","new");
//*-*     TH1F h1("hgaus","histo from a gaussian",100,-3,3);
//*-*     h1.FillRandom("gaus",10000);
//*-*     h1->Write();
//*-*  To Read this histogram in another Root session, do:
//*-*     TFile f("histos.root");
//*-*     TH1F *h = (TH1F*)f.Get("hgaus");
//*-*  One can save all histograms in memory to the file by:
//*-*  file->Write();
//*-*
//*-*  Miscelaneous operations
//*-*  =======================
//*-*
//*-*   TH1::KolmogorovTest: Statistical test of compatibility in shape between
//*-*                        two histograms.
//*-*   TH1::Smooth smooths the bin contents of a 1-d histogram
//*-*   TH1::Integral returns the integral of bin contents in a given bin range
//*-*   TH1::GetMean(int axis) returns the mean value along axis
//*-*   TH1::GetRMS(int axis)  returns the Root Mean Square along axis
//*-*   TH1::GetEntries returns the number of entries
//*-*   TH1::Reset() resets the bin contents and errors of an histogram.
//*-*
//Begin_Html
/*
<img src="gif/th1_classtree.gif">
*/
//End_Html
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

Bool_t TH1::fgAddDirectory = kTRUE;

Foption_t Foption;

TAxis *xaxis=0;
TAxis *yaxis=0;
TAxis *zaxis=0;

TF1 *gF1=0;

TVirtualFitter *hFitter=0;

static Int_t xfirst,xlast,yfirst,ylast,zfirst,zlast;
static Axis_t xmin, xmax, ymin, ymax, zmin, zmax, binwidx, binwidy, binwidz;

extern void H1InitGaus();
extern void H1InitExpo();
extern void H1InitPolynom();
extern void H1FitChisquare(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
extern void H1FitLikelihood(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
extern void H1LeastSquareFit(Int_t n, Int_t m, Double_t *a);
extern void H1LeastSquareLinearFit(Int_t ndata, Double_t &a0, Double_t &a1, Int_t &ifail);
extern void H1LeastSquareSeqnd(Int_t n, Double_t *a, Int_t idim, Int_t &ifail, Int_t k, Double_t *b);


ClassImp(TH1)

//______________________________________________________________________________
TH1::TH1(): TNamed(), TAttLine(), TAttFill(), TAttMarker()
{
//*-*-*-*-*-*-*-*-*-*-*Histogram default constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =============================
   fDirectory     = 0;
   fFunctions     = new TList;
   fNcells        = 0;
   fIntegral      = 0;
   fPainter       = 0;
   fEntries       = 0;
   fNormFactor    = 0;
   fTsumw         = fTsumw2=fTsumwx=fTsumwx2=0;
   fMaximum       = -1111;
   fMinimum       = -1111;
   fXaxis.SetName("xaxis");
   fYaxis.SetName("yaxis");
   fZaxis.SetName("zaxis");
   fXaxis.SetParent(this);
   fYaxis.SetParent(this);
   fZaxis.SetParent(this);
}

//______________________________________________________________________________
TH1::~TH1()
{
//*-*-*-*-*-*-*-*-*-*-*Histogram default destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ============================

   if (!TestBit(kNotDeleted)) return;
   if (fIntegral) {delete [] fIntegral; fIntegral = 0;}
   if (fFunctions) { fFunctions->Delete(); delete fFunctions; }
   if (fDirectory) {
      if (!fDirectory->TestBit(TDirectory::kCloseDirectory))
         fDirectory->GetList()->Remove(this);
   }
   delete fPainter;
   fDirectory = 0;
   fFunctions = 0;
}

//______________________________________________________________________________
TH1::TH1(const char *name,const char *title,Int_t nbins,Axis_t xlow,Axis_t xup)
    :TNamed(name,title), TAttLine(), TAttFill(), TAttMarker()
{
//*-*-*-*-*-*-*-*-*Normal constructor for fix bin size histograms*-*-*-*-*-*-*
//*-*              ==============================================
//
//     Creates the main histogram structure:
//        name   : name of histogram (avoid blanks)
//        title  : histogram title
//        nbins  : number of bins
//        xlow   : low edge of first bin
//        xup    : upper edge of last bin (not included in last bin)
//
//      When an histogram is created, it is automatically added to the list
//      of special objects in the current directory.
//      To find the pointer to this histogram in the current directory
//      by its name, do:
//      TH1F *h1 = (TH1F*)gDirectory->FindObject(name);
//
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   Build();
   if (nbins <= 0) nbins = 1;
   fXaxis.Set(nbins,xlow,xup);
   fNcells = fXaxis.GetNbins()+2;
}

//______________________________________________________________________________
TH1::TH1(const char *name,const char *title,Int_t nbins,const Float_t *xbins)
    :TNamed(name,title), TAttLine(), TAttFill(), TAttMarker()
{
//*-*-*-*-*-*-*Normal constructor for variable bin size histograms*-*-*-*-*-*-*
//*-*          ===================================================
//
//  Creates the main histogram structure:
//     name   : name of histogram (avoid blanks)
//     title  : histogram title
//     nbins  : number of bins
//     xbins  : array of low-edges for each bin
//              This is an array of size nbins+1
//
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   Build();
   if (nbins <= 0) nbins = 1;
   if (xbins) fXaxis.Set(nbins,xbins);
   else       fXaxis.Set(nbins,0,1);
   fNcells = fXaxis.GetNbins()+2;
}

//______________________________________________________________________________
TH1::TH1(const char *name,const char *title,Int_t nbins,const Double_t *xbins)
    :TNamed(name,title), TAttLine(), TAttFill(), TAttMarker()
{
//*-*-*-*-*-*-*Normal constructor for variable bin size histograms*-*-*-*-*-*-*
//*-*          ===================================================
//
//  Creates the main histogram structure:
//     name   : name of histogram (avoid blanks)
//     title  : histogram title
//     nbins  : number of bins
//     xbins  : array of low-edges for each bin
//              This is an array of size nbins+1
//
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   Build();
   if (nbins <= 0) nbins = 1;
   if (xbins) fXaxis.Set(nbins,xbins);
   else       fXaxis.Set(nbins,0,1);
   fNcells = fXaxis.GetNbins()+2;
}

//______________________________________________________________________________
Bool_t TH1::AddDirectoryStatus()
{
   //static function: cannot be inlined on Windows/NT
   return fgAddDirectory;
}

//______________________________________________________________________________
void TH1::Browse(TBrowser *)
{
    Draw();
    gPad->Update();
}


//______________________________________________________________________________
void TH1::Build()
{
//*-*-*-*-*-*-*-*-*-*Creates histogram basic data structure*-*-*-*-*-*-*-*-*-*
//*-*                ======================================

   fDirectory     = 0;
   fPainter       = 0;
   fIntegral      = 0;
   fEntries       = 0;
   fNormFactor    = 0;
   fTsumw         = fTsumw2=fTsumwx=fTsumwx2=0;
   fMaximum       = -1111;
   fMinimum       = -1111;
   fXaxis.SetName("xaxis");
   fYaxis.SetName("yaxis");
   fZaxis.SetName("zaxis");
   fYaxis.Set(1,0.,1.);
   fZaxis.Set(1,0.,1.);
   fXaxis.SetParent(this);
   fYaxis.SetParent(this);
   fZaxis.SetParent(this);

   UseCurrentStyle();

   if (fgAddDirectory && gDirectory) {
      TH1 *hold = (TH1*)gDirectory->GetList()->FindObject(GetName());
      if (hold) {
         Warning("Build","Replacing existing histogram: %s",GetName());
         gDirectory->GetList()->Remove(hold);
       //  delete hold;
      }
      gDirectory->Append(this);
      fDirectory = gDirectory;
   }
   fFunctions = new TList;
}

//______________________________________________________________________________
void TH1::Add(TF1 *f1, Double_t c1)
{
   // Performs the operation: this = this + c1*f1
   // if errors are defined (see TH1::Sumw2), errors are also recalculated.

   if (!f1) {
      Error("Add","Attempt to add a non-existing function");
      return;
   }

   Int_t nbinsx = GetNbinsX();
   Int_t nbinsy = GetNbinsY();
   Int_t nbinsz = GetNbinsZ();
   if (fDimension < 2) nbinsy = -1;
   if (fDimension < 3) nbinsz = -1;

//*-*- Add statistics
   Stat_t s1[10];
   Int_t i;
   for (i=0;i<10;i++) {s1[i] = 0;}
   PutStats(s1);

//*-*- Loop on bins (including underflows/overflows)
   Int_t bin, binx, biny, binz;
   Double_t cu;
   Double_t xx[3];
   Double_t *params = 0;
   f1->InitArgs(xx,params);
   for (binz=0;binz<=nbinsz+1;binz++) {
      xx[2] = fZaxis.GetBinCenter(binz);
      for (biny=0;biny<=nbinsy+1;biny++) {
         xx[1] = fYaxis.GetBinCenter(biny);
         for (binx=0;binx<=nbinsx+1;binx++) {
            xx[0] = fXaxis.GetBinCenter(binx);
            bin = binx +(nbinsx+2)*(biny + (nbinsy+2)*binz);
            cu  = c1*f1->EvalPar(xx);
            Double_t error1 = GetBinError(bin);
            AddBinContent(bin,cu);
            if (fSumw2.fN) {
               fSumw2.fArray[bin] = c1*c1*error1*error1;
            }
         }
      }
   }
}

//______________________________________________________________________________
void TH1::Add(TH1 *h1, Double_t c1)
{
   // Performs the operation: this = this + c1*h1
   // if errors are defined (see TH1::Sumw2), errors are also recalculated.
   // Note that if h1 has Sumw2 set, Sumw2 is automatically called for this
   // if not already set.

   if (!h1) {
      Error("Add","Attempt to add a non-existing histogram");
      return;
   }

   Int_t nbinsx = GetNbinsX();
   Int_t nbinsy = GetNbinsY();
   Int_t nbinsz = GetNbinsZ();
//*-*- Check histogram compatibility
   if (nbinsx != h1->GetNbinsX() || nbinsy != h1->GetNbinsY() || nbinsz != h1->GetNbinsZ()) {
      Error("Add","Attempt to add histograms with different number of bins");
      return;
   }
//*-*- Issue a Warning if histogram limits are different
   if (GetXaxis()->GetXmin() != h1->GetXaxis()->GetXmin() ||
       GetXaxis()->GetXmax() != h1->GetXaxis()->GetXmax() ||
       GetYaxis()->GetXmin() != h1->GetYaxis()->GetXmin() ||
       GetYaxis()->GetXmax() != h1->GetYaxis()->GetXmax() ||
       GetZaxis()->GetXmin() != h1->GetZaxis()->GetXmin() ||
       GetZaxis()->GetXmax() != h1->GetZaxis()->GetXmax()) {
       Warning("Add","Attempt to add histograms with different axis limits");
   }
   if (fDimension < 2) nbinsy = -1;
   if (fDimension < 3) nbinsz = -1;

//*-* Create Sumw2 if h1 has Sumw2 set
   if (fSumw2.fN == 0 && h1->GetSumw2N() != 0) Sumw2();

//*-*- Add statistics
   fEntries += c1*h1->GetEntries();
   Stat_t s1[10], s2[10];
   Int_t i;
   for (i=0;i<10;i++) {s1[i] = s2[i] = 0;}
   GetStats(s1);
   h1->GetStats(s2);
   for (i=0;i<10;i++) s1[i] += c1*s2[i];
   PutStats(s1);

//*-*- Loop on bins (including underflows/overflows)
   Int_t bin, binx, biny, binz;
   Double_t cu;
   for (binz=0;binz<=nbinsz+1;binz++) {
      for (biny=0;biny<=nbinsy+1;biny++) {
         for (binx=0;binx<=nbinsx+1;binx++) {
            bin = binx +(nbinsx+2)*(biny + (nbinsy+2)*binz);
            cu  = c1*h1->GetBinContent(bin);
            AddBinContent(bin,cu);
            if (fSumw2.fN) {
               Double_t error1 = h1->GetBinError(bin);
               fSumw2.fArray[bin] += c1*c1*error1*error1;
            }
         }
      }
   }
}

//______________________________________________________________________________
void TH1::Add(TH1 *h1, TH1 *h2, Double_t c1, Double_t c2)
{
//*-*-*-*-*Replace contents of this histogram by the addition of h1 and h2*-*-*
//*-*      ===============================================================
//
//   this = c1*h1 + c2*h2
//   if errors are defined (see TH1::Sumw2), errors are also recalculated
//   Note that if h1 or h2 have Sumw2 set, Sumw2 is automatically called for this
//   if not already set.
//

   if (!h1 || !h2) {
      Error("Add","Attempt to add a non-existing histogram");
      return;
   }

   Int_t nbinsx = GetNbinsX();
   Int_t nbinsy = GetNbinsY();
   Int_t nbinsz = GetNbinsZ();
//*-*- Check histogram compatibility
   if (nbinsx != h1->GetNbinsX() || nbinsy != h1->GetNbinsY() || nbinsz != h1->GetNbinsZ()
    || nbinsx != h2->GetNbinsX() || nbinsy != h2->GetNbinsY() || nbinsz != h2->GetNbinsZ()) {
      Error("Add","Attempt to add histograms with different number of bins");
      return;
   }
//*-*- Issue a Warning if histogram limits are different
   if (GetXaxis()->GetXmin() != h1->GetXaxis()->GetXmin() ||
       GetXaxis()->GetXmax() != h1->GetXaxis()->GetXmax() ||
       GetYaxis()->GetXmin() != h1->GetYaxis()->GetXmin() ||
       GetYaxis()->GetXmax() != h1->GetYaxis()->GetXmax() ||
       GetZaxis()->GetXmin() != h1->GetZaxis()->GetXmin() ||
       GetZaxis()->GetXmax() != h1->GetZaxis()->GetXmax()) {
       Warning("Add","Attempt to add histograms with different axis limits");
   }
   if (GetXaxis()->GetXmin() != h2->GetXaxis()->GetXmin() ||
       GetXaxis()->GetXmax() != h2->GetXaxis()->GetXmax() ||
       GetYaxis()->GetXmin() != h2->GetYaxis()->GetXmin() ||
       GetYaxis()->GetXmax() != h2->GetYaxis()->GetXmax() ||
       GetZaxis()->GetXmin() != h2->GetZaxis()->GetXmin() ||
       GetZaxis()->GetXmax() != h2->GetZaxis()->GetXmax()) {
       Warning("Add","Attempt to add histograms with different axis limits");
   }
   if (fDimension < 2) nbinsy = -1;
   if (fDimension < 3) nbinsz = -1;
   if (fDimension < 3) nbinsz = -1;

//*-* Create Sumw2 if h1 or h2 have Sumw2 set
   if (fSumw2.fN == 0 && (h1->GetSumw2N() != 0 || h2->GetSumw2N() != 0)) Sumw2();

//*-*- Add statistics
   fEntries = c1*h1->GetEntries() + c2*h2->GetEntries();
   Stat_t s1[10], s2[10], s3[10];
   Int_t i;
   for (i=0;i<10;i++) {s1[i] = s2[i] = s3[i] = 0;}
   h1->GetStats(s1);
   h2->GetStats(s2);
   for (i=0;i<10;i++) s3[i] = c1*s1[i] + c2*s2[i];
   PutStats(s3);

//*-*- Loop on bins (including underflows/overflows)
   Int_t bin, binx, biny, binz;
   Double_t cu;
   for (binz=0;binz<=nbinsz+1;binz++) {
      for (biny=0;biny<=nbinsy+1;biny++) {
         for (binx=0;binx<=nbinsx+1;binx++) {
            bin = binx +(nbinsx+2)*(biny + (nbinsy+2)*binz);
            cu  = c1*h1->GetBinContent(bin)+ c2*h2->GetBinContent(bin);
            SetBinContent(bin,cu);
            if (fSumw2.fN) {
               Double_t error1 = h1->GetBinError(bin);
               Double_t error2 = h2->GetBinError(bin);
               fSumw2.fArray[bin] = c1*c1*error1*error1 + c2*c2*error2*error2;
            }
         }
      }
   }
}


//______________________________________________________________________________
void TH1::AddBinContent(Int_t)
{
//*-*-*-*-*-*-*-*-*-*Increment bin content by 1*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                ==========================
   AbstractMethod("AddBinContent");
}

//______________________________________________________________________________
void TH1::AddBinContent(Int_t, Stat_t)
{
//*-*-*-*-*-*-*-*-*-*Increment bin content by a weight w*-*-*-*-*-*-*-*-*-*-*
//*-*                ===================================
   AbstractMethod("AddBinContent");
}

//______________________________________________________________________________
void TH1::AddDirectory(Bool_t add)
{
// Sets the flag controlling the automatic add of histograms in memory
//
// By default (fAddDirectory = kTRUE), histograms are automatically added
// to the list of objects in memory.
// Note that one histogram can be removed from its support directory
// by calling h->SetDirectory(0) or h->SetDirectory(dir) to add it
// to the list of objects in the directory dir.
//
//  NOTE that this is a static function. To call it, use;
//     TH1::AddDirectory

   fgAddDirectory = add;
}

//______________________________________________________________________________
Double_t TH1::ComputeIntegral()
{
//  Compute integral (cumulative sum of bins)
//  The result stored in fIntegral is used by the GetRandom functions.
//  This function is automatically called by GetRandom when the fIntegral
//  array does not exist or when the number of entries in the histogram
//  has changed since the previous call to GetRandom.
//  The resulting integral is normalized to 1

   Int_t bin, binx, biny, binz, ibin;

// delete previously computed integral (if any)
   if (fIntegral) delete [] fIntegral;

//*-*- Allocate space to store the integral and compute integral
   Int_t nbinsx = GetNbinsX();
   Int_t nbinsy = GetNbinsY();
   Int_t nbinsz = GetNbinsZ();
   Int_t nxy    = nbinsx*nbinsy;
   Int_t nbins  = nxy*nbinsz;

   fIntegral = new Double_t[nbins+2];
   ibin = 0;
   fIntegral[ibin] = 0;
   for (binz=1;binz<=nbinsz;binz++) {
      for (biny=1;biny<=nbinsy;biny++) {
         for (binx=1;binx<=nbinsx;binx++) {
            ibin++;
            bin  = GetBin(binx, biny, binz);
            fIntegral[ibin] = fIntegral[ibin-1] + GetBinContent(bin);
         }
      }
   }

//*-*- Normalize integral to 1
   if (fIntegral[nbins] == 0 ) {
      Error("ComputeIntegral", "Integral = zero"); return 0;
   }
   for (bin=1;bin<=nbins;bin++)  fIntegral[bin] /= fIntegral[nbins];
   fIntegral[nbins+1] = fEntries;
   return fIntegral[nbins];
}

//______________________________________________________________________________
void TH1::Copy(TObject &obj)
{
//*-*-*-*-*-*-*Copy this histogram structure to newth1*-*-*-*-*-*-*-*-*-*-*-*
//*-*          =======================================
//
// Note that this function does not copy the list of associated functions.
// Use TObJect::Clone to make a full copy of an histogram.

   TNamed::Copy(obj);
   ((TH1&)obj).fDimension = fDimension;
   ((TH1&)obj).fNormFactor= fNormFactor;
   ((TH1&)obj).fEntries   = fEntries;
   ((TH1&)obj).fNcells    = fNcells;
   ((TH1&)obj).fBarOffset = fBarOffset;
   ((TH1&)obj).fBarWidth  = fBarWidth;
   ((TH1&)obj).fTsumw     = fTsumw;
   ((TH1&)obj).fTsumw2    = fTsumw2;
   ((TH1&)obj).fTsumwx    = fTsumwx;
   ((TH1&)obj).fTsumwx2   = fTsumwx2;
   ((TH1&)obj).fMaximum   = fMaximum;
   ((TH1&)obj).fMinimum   = fMinimum;
   ((TH1&)obj).fOption    = fOption;
   TAttLine::Copy(((TH1&)obj));
   TAttFill::Copy(((TH1&)obj));
   TAttMarker::Copy(((TH1&)obj));
   fXaxis.Copy(((TH1&)obj).fXaxis);
   fYaxis.Copy(((TH1&)obj).fYaxis);
   fZaxis.Copy(((TH1&)obj).fZaxis);
   ((TH1&)obj).fXaxis.SetParent(&obj);
   ((TH1&)obj).fYaxis.SetParent(&obj);
   ((TH1&)obj).fZaxis.SetParent(&obj);
   fContour.Copy(((TH1&)obj).fContour);
   fSumw2.Copy(((TH1&)obj).fSumw2);
//   fFunctions->Copy(((TH1&)obj).fFunctions);
   gDirectory->Append(&obj);
//   ((TH1&)obj).AppendDirectory();
   ((TH1&)obj).fDirectory = gDirectory;
}

//______________________________________________________________________________
Int_t TH1::DistancetoPrimitive(Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*-*-*Compute distance from point px,py to a line*-*-*-*-*-*
//*-*                  ===========================================
//*-*  Compute the closest distance of approach from point px,py to elements
//*-*  of an histogram.
//*-*  The distance is computed in pixels units.
//*-*
//*-*  Algorithm:
//*-*  Currently, this simple model computes the distance from the mouse
//*-*  to the histogram contour only.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   if (!fPainter) return 9999;
   return fPainter->DistancetoPrimitive(px,py);
}

//______________________________________________________________________________
void TH1::Divide(TF1 *f1, Double_t c1)
{
   // Performs the operation: this = this/(c1*f1)
   // if errors are defined (see TH1::Sumw2), errors are also recalculated.

   if (!f1) {
      Error("Add","Attempt to divide by a non-existing function");
      return;
   }

   Int_t nbinsx = GetNbinsX();
   Int_t nbinsy = GetNbinsY();
   Int_t nbinsz = GetNbinsZ();
   if (fDimension < 2) nbinsy = -1;
   if (fDimension < 3) nbinsz = -1;

//*-*- Add statistics
   Stat_t s1[10];
   Int_t i;
   for (i=0;i<10;i++) {s1[i] = 0;}
   PutStats(s1);

//*-*- Loop on bins (including underflows/overflows)
   Int_t bin, binx, biny, binz;
   Double_t cu,w;
   Double_t xx[3];
   Double_t *params = 0;
   f1->InitArgs(xx,params);
   for (binz=0;binz<=nbinsz+1;binz++) {
      xx[2] = fZaxis.GetBinCenter(binz);
      for (biny=0;biny<=nbinsy+1;biny++) {
         xx[1] = fYaxis.GetBinCenter(biny);
         for (binx=0;binx<=nbinsx+1;binx++) {
            xx[0] = fXaxis.GetBinCenter(binx);
            bin = binx +(nbinsx+2)*(biny + (nbinsy+2)*binz);
            Double_t error1 = GetBinError(bin);
            cu  = c1*f1->EvalPar(xx);
            if (cu) w = GetBinContent(bin)/cu;
            else    w = 0;
            SetBinContent(bin,w);
            if (fSumw2.fN) {
               fSumw2.fArray[bin] = c1*c1*error1*error1;
            }
         }
      }
   }
}

//______________________________________________________________________________
void TH1::Divide(TH1 *h1)
{
//*-*-*-*-*-*-*-*-*-*-*Divide this histogram by h1*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===========================
//
//   this = this/h1
//   if errors are defined (see TH1::Sumw2), errors are also recalculated.
//   Note that if h1 has Sumw2 set, Sumw2 is automatically called for this
//   if not already set.
//   The resulting errors are calculated assuming uncorrelated histograms.
//   See the other TH1::Divide that gives the possibility to optionaly
//   compute Binomial errors.

   if (!h1) {
      Error("Divide","Attempt to divide by a non-existing histogram");
      return;
   }

   Int_t nbinsx = GetNbinsX();
   Int_t nbinsy = GetNbinsY();
   Int_t nbinsz = GetNbinsZ();
//*-*- Check histogram compatibility
   if (nbinsx != h1->GetNbinsX() || nbinsy != h1->GetNbinsY() || nbinsz != h1->GetNbinsZ()) {
      Error("Divide","Attempt to divide histograms with different number of bins");
      return;
   }
//*-*- Issue a Warning if histogram limits are different
   if (GetXaxis()->GetXmin() != h1->GetXaxis()->GetXmin() ||
       GetXaxis()->GetXmax() != h1->GetXaxis()->GetXmax() ||
       GetYaxis()->GetXmin() != h1->GetYaxis()->GetXmin() ||
       GetYaxis()->GetXmax() != h1->GetYaxis()->GetXmax() ||
       GetZaxis()->GetXmin() != h1->GetZaxis()->GetXmin() ||
       GetZaxis()->GetXmax() != h1->GetZaxis()->GetXmax()) {
       Warning("Divide","Attempt to divide histograms with different axis limits");
   }
   if (fDimension < 2) nbinsy = -1;
   if (fDimension < 3) nbinsz = -1;
   if (fDimension < 3) nbinsz = -1;

//*-* Create Sumw2 if h1 has Sumw2 set
   if (fSumw2.fN == 0 && h1->GetSumw2N() != 0) Sumw2();

//*-*- Reset statistics
   fEntries = fTsumw = 0;

//*-*- Loop on bins (including underflows/overflows)
   Int_t bin, binx, biny, binz;
   Double_t c0,c1,w;
   for (binz=0;binz<=nbinsz+1;binz++) {
      for (biny=0;biny<=nbinsy+1;biny++) {
         for (binx=0;binx<=nbinsx+1;binx++) {
            bin = GetBin(binx,biny,binz);
            c0  = GetBinContent(bin);
            c1  = h1->GetBinContent(bin);
            if (c1) w = c0/c1;
            else    w = 0;
            SetBinContent(bin,w);
            fEntries++;
            if (fSumw2.fN) {
               Double_t e0 = GetBinError(bin);
               Double_t e1 = h1->GetBinError(bin);
               Double_t c12= c1*c1;
               if (!c1) { fSumw2.fArray[bin] = 0; continue;}
               fSumw2.fArray[bin] = (e0*e0*c1*c1 + e1*e1*c0*c0)/(c12*c12);
            }
         }
      }
   }
   Stat_t s[10];
   GetStats(s);
   PutStats(s);
}


//______________________________________________________________________________
void TH1::Divide(TH1 *h1, TH1 *h2, Double_t c1, Double_t c2, Option_t *option)
{
//*-*-*-*-*Replace contents of this histogram by the division of h1 by h2*-*-*
//*-*      ==============================================================
//
//   this = c1*h1/(c2*h2)
//
//   if errors are defined (see TH1::Sumw2), errors are also recalculated
//   Note that if h1 or h2 have Sumw2 set, Sumw2 is automatically called for this
//   if not already set.
//   The resulting errors are calculated assuming uncorrelated histograms.
//   However, if option ="B" is specified, Binomial errors are computed.
//

   TString opt = option;
   opt.ToLower();
   Bool_t binomial = kFALSE;
   if (opt.Contains("b")) binomial = kTRUE;
   if (!h1 || !h2) {
      Error("Divide","Attempt to divide by a non-existing histogram");
      return;
   }

   Int_t nbinsx = GetNbinsX();
   Int_t nbinsy = GetNbinsY();
   Int_t nbinsz = GetNbinsZ();
//*-*- Check histogram compatibility
   if (nbinsx != h1->GetNbinsX() || nbinsy != h1->GetNbinsY() || nbinsz != h1->GetNbinsZ()
    || nbinsx != h2->GetNbinsX() || nbinsy != h2->GetNbinsY() || nbinsz != h2->GetNbinsZ()) {
      Error("Divide","Attempt to divide histograms with different number of bins");
      return;
   }
   if (!c2) {
      Error("Divide","Coefficient of dividing histogram cannot be zero");
      return;
   }
//*-*- Issue a Warning if histogram limits are different
   if (GetXaxis()->GetXmin() != h1->GetXaxis()->GetXmin() ||
       GetXaxis()->GetXmax() != h1->GetXaxis()->GetXmax() ||
       GetYaxis()->GetXmin() != h1->GetYaxis()->GetXmin() ||
       GetYaxis()->GetXmax() != h1->GetYaxis()->GetXmax() ||
       GetZaxis()->GetXmin() != h1->GetZaxis()->GetXmin() ||
       GetZaxis()->GetXmax() != h1->GetZaxis()->GetXmax()) {
       Warning("Divide","Attempt to divide histograms with different axis limits");
   }
   if (GetXaxis()->GetXmin() != h2->GetXaxis()->GetXmin() ||
       GetXaxis()->GetXmax() != h2->GetXaxis()->GetXmax() ||
       GetYaxis()->GetXmin() != h2->GetYaxis()->GetXmin() ||
       GetYaxis()->GetXmax() != h2->GetYaxis()->GetXmax() ||
       GetZaxis()->GetXmin() != h2->GetZaxis()->GetXmin() ||
       GetZaxis()->GetXmax() != h2->GetZaxis()->GetXmax()) {
       Warning("Divide","Attempt to divide histograms with different axis limits");
   }
   if (fDimension < 2) nbinsy = -1;
   if (fDimension < 3) nbinsz = -1;

//*-* Create Sumw2 if h1 or h2 have Sumw2 set
   if (fSumw2.fN == 0 && (h1->GetSumw2N() != 0 || h2->GetSumw2N() != 0)) Sumw2();

//*-*- Reset statistics
   fEntries = fTsumw = 0;

//*-*- Loop on bins (including underflows/overflows)
   Int_t bin, binx, biny, binz;
   Double_t b1,b2,w,d1,d2;
   d1 = c1*c1;
   d2 = c2*c2;
   for (binz=0;binz<=nbinsz+1;binz++) {
      for (biny=0;biny<=nbinsy+1;biny++) {
         for (binx=0;binx<=nbinsx+1;binx++) {
            bin = binx +(nbinsx+2)*(biny + (nbinsy+2)*binz);
            b1  = h1->GetBinContent(bin);
            b2  = h2->GetBinContent(bin);
            if (b2) w = c1*b1/(c2*b2);
            else    w = 0;
            SetBinContent(bin,w);
            fEntries++;
            if (fSumw2.fN) {
               Double_t e1 = h1->GetBinError(bin);
               Double_t e2 = h2->GetBinError(bin);
               Double_t b22= b2*b2*d2;
               if (!b2) { fSumw2.fArray[bin] = 0; continue;}
               if (binomial) {
                  fSumw2.fArray[bin] = TMath::Abs(w*(1-w)/(c2*b2));
               } else {
                  fSumw2.fArray[bin] = d1*d2*(e1*e1*b2*b2 + e2*e2*b1*b1)/(b22*b22);
               }
            }
         }
      }
   }
   Stat_t s[10];
   GetStats(s);
   PutStats(s);
}

//______________________________________________________________________________
void TH1::Draw(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*Draw this histogram with options*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ================================
//*-*
//*-*  Histograms are drawn via the THistPainter class. Each histogram has
//*-*  a pointer to its own painter (to be usable in a multithreaded program).
//*-*  The same histogram can be drawn with different options in different pads.
//*-*  When an histogram drawn in a pad is deleted, the histogram is
//*-*  automatically removed from the pad or pads where it was drawn.
//*-*  If an histogram is drawn in a pad, then filled again, the new status
//*-*  of the histogram will be automatically shown in the pad next time
//*-*  the pad is updated. One does not need to redraw the histogram.
//*-*  To draw the current version of an histogram in a pad, one can use
//*-*     h->DrawCopy();
//*-*  This makes a clone of the histogram. Once the clone is drawn, the original
//*-*  histogram may be modified or deleted without affecting the aspect of the
//*-*  clone.
//*-*  By default, TH1::Draw clears the current pad.
//*-*
//*-*  One can use TH1::SetMaximum and TH1::SetMinimum to force a particular
//*-*  value for the maximum or the minimum scale on the plot.
//*-*
//*-*  TH1::UseCurrentStyle can be used to change all histogram graphics
//*-*  attributes to correspond to the current selected style.
//*-*  This function must be called for each histogram.
//*-*  In case one reads and draws many histograms from a file, one can force
//*-*  the histograms to inherit automatically the current graphics style
//*-*  by calling before gROOT->ForceStyle();
//*-*
//*-*  See THistPainter::Paint for a description of all the drawing options
//*-*  =======================
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   TString opt = option;
   opt.ToLower();
   if (gPad) {
      if (!gPad->IsEditable()) (gROOT->GetMakeDefCanvas())();
      if (!opt.Contains("same")) {
         //the following statement is necessary in case one attempts to draw
         //a temporary histogram already in the current pad
         if (TestBit(kCanDelete)) gPad->GetListOfPrimitives()->Remove(this);
         gPad->Clear();
      }
   }
   AppendPad(opt.Data());
}

//______________________________________________________________________________
TH1 *TH1::DrawCopy(Option_t *)
{
//*-*-*-*-*-*-*Copy this histogram and Draw in the current pad*-*-*-*-*-*-*-*
//*-*          ===============================================
//*-*
//*-*  Once the histogram is drawn into the pad, any further modification
//*-*  using graphics input will be made on the copy of the histogram,
//*-*  and not to the original object.
//*-*
//*-*  See Draw for the list of options
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   AbstractMethod("DrawCopy");
   return 0;
}

//______________________________________________________________________________
void TH1::DrawPanel()
{
//*-*-*-*-*-*-*Display a panel with all histogram drawing options*-*-*-*-*-*
//*-*          ==================================================
//*-*
//*-*   See class TDrawPanelHist for example

   if (fPainter) fPainter->DrawPanel();
}

//______________________________________________________________________________
void TH1::Eval(TF1 *f1, Option_t *option)
{
//*-*-*-*-*Evaluate function f1 at the center of bins of this histogram-*-*-*-*
//*-*      ============================================================
//*-*
//*-*  If option "R" is specified, the function is evaluated only
//*-*  for the bins included in the function range.
//*-*  If option "A" is specified, the value of the function is added to the
//*-*  existing bin contents
//*-*  If option "S" is specified, the value of the function is used to
//*-*  generate an integer value, distributed according to the Poisson
//*-*  distribution, with f1 as the mean.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   Double_t x[3];
   Int_t stat,add,bin,binx,biny,binz,nbinsx, nbinsy, nbinsz;
   if (!f1) return;
   Double_t fu;
   Double_t e=0;
   TString opt = option;
   opt.ToLower();
   if (opt.Contains("a")) add  = 1;
   else                   add  = 0;
   if (opt.Contains("s")) stat = 1;
   else                   stat = 0;
   nbinsx  = fXaxis.GetNbins();
   nbinsy  = fYaxis.GetNbins();
   nbinsz  = fZaxis.GetNbins();
   if (!add) Reset();

   for (binz=1;binz<=nbinsz;binz++) {
      x[2]  = fZaxis.GetBinCenter(binz);
      for (biny=1;biny<=nbinsy;biny++) {
         x[1]  = fYaxis.GetBinCenter(biny);
         for (binx=1;binx<=nbinsx;binx++) {
            bin = GetBin(binx,biny,binz);
            x[0]  = fXaxis.GetBinCenter(binx);
            fu = f1->Eval(x[0],x[1],x[2]);
            if (stat) fu = gRandom->Poisson(fu);
            if (fSumw2.fN) e = fSumw2.fArray[bin];
            AddBinContent(bin,fu);
            if (fSumw2.fN) fSumw2.fArray[bin] = e+ fu*fu;
         }
      }
   }
}

//______________________________________________________________________________
void TH1::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*-*-*Execute action corresponding to one event*-*-*-*
//*-*                  =========================================
//*-*  This member function is called when a histogram is clicked with the locator
//*-*
//*-*  If Left button clicked on the bin top value, then the content of this bin
//*-*  is modified according to the new position of the mouse when it is released.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   if (fPainter) fPainter->ExecuteEvent(event, px, py);
}

//______________________________________________________________________________
Int_t TH1::Fill(Axis_t x)
{
//*-*-*-*-*-*-*-*-*-*Increment bin with abscissa X by 1*-*-*-*-*-*-*-*-*-*-*
//*-*                ==================================
//*-*
//*-* if x is less than the low-edge of the first bin, the Underflow bin is incremented
//*-* if x is greater than the upper edge of last bin, the Overflow bin is incremented
//*-*
//*-* If the storage of the sum of squares of weights has been triggered,
//*-* via the function Sumw2, then the sum of the squares of weights is incremented
//*-* by 1 in the bin corresponding to x.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   Int_t bin;
   fEntries++;
   bin =fXaxis.FindBin(x);
   AddBinContent(bin);
   if (fSumw2.fN) ++fSumw2.fArray[bin];
   if (bin == 0 || bin > fXaxis.GetNbins()) return -1;
   ++fTsumw;
   ++fTsumw2;
   fTsumwx  += x;
   fTsumwx2 += x*x;
   return bin;
}

//______________________________________________________________________________
Int_t TH1::Fill(Axis_t x, Stat_t w)
{
//*-*-*-*-*-*-*-*Increment bin with abscissa X with a weight w*-*-*-*-*-*-*-*
//*-*            =============================================
//*-*
//*-* if x is less than the low-edge of the first bin, the Underflow bin is incremented
//*-* if x is greater than the upper edge of last bin, the Overflow bin is incremented
//*-*
//*-* If the storage of the sum of squares of weights has been triggered,
//*-* via the function Sumw2, then the sum of the squares of weights is incremented
//*-* by w^2 in the bin corresponding to x.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   Int_t bin;
   fEntries++;
   bin =fXaxis.FindBin(x);
   AddBinContent(bin, w);
   if (fSumw2.fN) fSumw2.fArray[bin] += w*w;
   if (bin == 0 || bin > fXaxis.GetNbins()) return -1;
   Stat_t z= (w > 0 ? w : -w);
   fTsumw   += z;
   fTsumw2  += z*z;
   fTsumwx  += z*x;
   fTsumwx2 += z*x*x;
   return bin;
}

//______________________________________________________________________________
void TH1::FillN(Int_t ntimes, const Axis_t *x, const Double_t *w, Int_t stride)
{
//*-*-*-*-*-*-*-*Fill this histogram with an array x and weights w*-*-*-*-*
//*-*            =================================================
//*-*
//*-* ntimes:  number of entries in arrays x and w (array size must be ntimes*stride)
//*-* x:       array of values to be histogrammed
//*-* w:       array of weighs
//*-* stride:  step size through arrays x and w
//*-*
//*-* If the storage of the sum of squares of weights has been triggered,
//*-* via the function Sumw2, then the sum of the squares of weights is incremented
//*-* by w[i]^2 in the bin corresponding to x[i].
//*-* if w is NULL each entry is assumed a weight=1
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   Int_t bin,i;
   fEntries += ntimes;
   Double_t ww = 1;
   Int_t nbins   = fXaxis.GetNbins();
   ntimes *= stride;
   for (i=0;i<ntimes;i+=stride) {
      bin =fXaxis.FindBin(x[i]);
      if (w) ww = w[i];
      AddBinContent(bin, ww);
      if (fSumw2.fN) fSumw2.fArray[bin] += ww*ww;
      if (bin == 0 || bin > nbins) continue;
      Stat_t z= (ww > 0 ? ww : -ww);
      fTsumw   += z;
      fTsumw2  += z*z;
      fTsumwx  += z*x[i];
      fTsumwx2 += z*x[i]*x[i];
   }
}

//______________________________________________________________________________
void TH1::FillRandom(const char *fname, Int_t ntimes)
{
//*-*-*-*-*-*-*Fill histogram following distribution in function fname*-*-*-*
//*-*          =======================================================
//*-*
//*-*   The distribution contained in the function fname (TF1) is integrated
//*-*   over the channel contents.
//*-*   It is normalized to 1.
//*-*   Getting one random number implies:
//*-*     - Generating a random number between 0 and 1 (say r1)
//*-*     - Look in which bin in the normalized integral r1 corresponds to
//*-*     - Fill histogram channel
//*-*   ntimes random numbers are generated
//*-*
//*-*  One can also call TF1::GetRandom to get a random variate from a function.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*

   Int_t bin, binx, ibin, loop;
   Double_t r1, x, xv[1];
//*-*- Search for fname in the list of ROOT defined functions
   TF1 *f1 = (TF1*)gROOT->GetFunction(fname);
   if (!f1) { Error("FillRandom", "Unknown function: %s",fname); return; }

//*-*- Allocate temporary space to store the integral and compute integral
   Int_t nbinsx = GetNbinsX();

   Double_t *integral = new Double_t[nbinsx+1];
   ibin = 0;
   integral[ibin] = 0;
   for (binx=1;binx<=nbinsx;binx++) {
      xv[0] = fXaxis.GetBinCenter(binx);
//      xv[0] = fXaxis.GetBinUpEdge(binx);
      ibin++;
      integral[ibin] = integral[ibin-1] + f1->Eval(xv[0])*fXaxis.GetBinWidth(binx);
   }

//*-*- Normalize integral to 1
   if (integral[nbinsx] == 0 ) {
      Error("FillRandom", "Integral = zero"); return;
   }
   for (bin=1;bin<=nbinsx;bin++)  integral[bin] /= integral[nbinsx];

//*-*--------------Start main loop ntimes
   for (loop=0;loop<ntimes;loop++) {
      r1 = gRandom->Rndm(loop);
      ibin = TMath::BinarySearch(nbinsx,&integral[0],r1);
      binx = 1 + ibin;
      x    = fXaxis.GetBinCenter(binx);
      Fill(x, 1.);
  }
  delete [] integral;
}

//______________________________________________________________________________
void TH1::FillRandom(TH1 *h, Int_t ntimes)
{
//*-*-*-*-*-*-*Fill histogram following distribution in histogram h*-*-*-*
//*-*          ====================================================
//*-*
//*-*   The distribution contained in the histogram h (TH1) is integrated
//*-*   over the channel contents.
//*-*   It is normalized to 1.
//*-*   Getting one random number implies:
//*-*     - Generating a random number between 0 and 1 (say r1)
//*-*     - Look in which bin in the normalized integral r1 corresponds to
//*-*     - Fill histogram channel
//*-*   ntimes random numbers are generated
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*

   if (!h) { Error("FillRandom", "Null histogram"); return; }
   if (fDimension != h->GetDimension()) {
      Error("FillRandom", "Histograms with different dimensions"); return;
   }

   if (h->ComputeIntegral() == 0) return;

   Int_t loop;
   Axis_t x;
   for (loop=0;loop<ntimes;loop++) {
      x = h->GetRandom();
      Fill(x);
   }
}


//______________________________________________________________________________
Int_t TH1::FindBin(Axis_t x, Axis_t y, Axis_t z)
{
//*-*-*-*-*-*Return Global bin number corresponding to x,y,z*-*-*-*-*-*-*
//*-*        ===============================================
//*-*
//*-*   2-D and 3-D histograms are represented with a one dimensional
//*-*   structure.
//*-*   This has the advantage that all existing functions, such as
//*-*     GetBinContent, GetBinError, GetBinFunction work for all dimensions.
//*-*  See also TH1::GetBin
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   if (GetDimension() < 2) {
      return fXaxis.FindBin(x);
   }
   if (GetDimension() < 3) {
      Int_t nx   = fXaxis.GetNbins()+2;
      Int_t binx = fXaxis.FindBin(x);
      Int_t biny = fYaxis.FindBin(y);
      return  binx + nx*biny;
   }
   if (GetDimension() < 4) {
      Int_t nx   = fXaxis.GetNbins()+2;
      Int_t ny   = fYaxis.GetNbins()+2;
      Int_t binx = fXaxis.FindBin(x);
      Int_t biny = fYaxis.FindBin(y);
      Int_t binz = fZaxis.FindBin(z);
      return  binx + nx*(biny +ny*binz);
   }
   return -1;
}

//______________________________________________________________________________
void TH1::Fit(const char *fname ,Option_t *option ,Option_t *goption, Axis_t xxmin, Axis_t xxmax)
{
//*-*-*-*-*-*-*-*-*-*-*Fit histogram with function fname*-*-*-*-*-*-*-*-*-*-*
//*-*                  =================================
//*-*   fname is the name of an already predefined function created by TF1 or TF2
//*-*   Predefined functions such as gaus, expo and poln are automatically
//*-*   created by ROOT.
//*-*
//  This function finds a pointer to the TF1 object with name fname
//  and calls TH1::Fit(TF1 *f1,...)

   TF1 *f1 = (TF1*)gROOT->GetFunction(fname);
   if (!f1) { Error("Fit", "Unknown function: %s",fname); return; }
   Fit(f1,option,goption,xxmin,xxmax);
}

//______________________________________________________________________________
void TH1::Fit(TF1 *f1 ,Option_t *option ,Option_t *goption, Axis_t xxmin, Axis_t xxmax)
{
//*-*-*-*-*-*-*-*-*-*-*Fit histogram with function f1*-*-*-*-*-*-*-*-*-*-*
//*-*                  ==============================
//*-*
//*-*   Fit this histogram with function f1.
//*-*
//*-*   The list of fit options is given in parameter option.
//*-*      option = "W"  Set all errors to 1
//*-*             = "I" Use integral of function in bin instead of value at bin center
//*-*             = "L" Use Loglikelihood method (default is chisquare method)
//*-*             = "U" Use a User specified fitting algorithm (via SetFCN)
//*-*             = "Q" Quiet mode (minimum printing)
//*-*             = "V" Verbose mode (default is between Q and V)
//*-*             = "E" Perform better Errors estimation using Minos technique
//*-*             = "M" More. Improve fit results
//*-*             = "R" Use the Range specified in the function range
//*-*             = "N" Do not store the graphics function, do not draw
//*-*             = "0" Do not plot the result of the fit. By default the fitted function
//*-*                   is drawn unless the option"N" above is specified.
//*-*             = "+" Add this new fitted function to the list of fitted functions
//*-*                   (by default, any previous function is deleted)
//*-*
//*-*   When the fit is drawn (by default), the parameter goption may be used
//*-*   to specify a list of graphics options. See TH1::Draw for a complete
//*-*   list of these options.
//*-*
//*-*   In order to use the Range option, one must first create a function
//*-*   with the expression to be fitted. For example, if your histogram
//*-*   has a defined range between -4 and 4 and you want to fit a gaussian
//*-*   only in the interval 1 to 3, you can do:
//*-*        TF1 *f1 = new TF1("f1","gaus",1,3);
//*-*        histo->Fit("f1","R");
//*-*
//*-*   Setting initial conditions
//*-*   ==========================
//*-*   Parameters must be initialized before invoking the Fit function.
//*-*   The setting of the parameter initial values is automatic for the
//*-*   predefined functions : poln, expo, gaus. One can however disable
//*-*   this automatic computation by specifying the option "B".
//*-*   You can specify boundary limits for some or all parameters via
//*-*        f1->SetParLimits(p_number, parmin, parmax);
//*-*   if parmin>=parmax, the parameter is fixed
//*-*   Note that you are not forced to fix the limits for all parameters.
//*-*   For example, if you fit a function with 6 parameters, you can do:
//*-*     func->SetParameters(0,3.1,1.e-6,-1.5,0,100);
//*-*     func->SetParLimits(3,-10,-4);
//*-*     func->FixParameter(4,0);
//*-*     func->SetParLimits(5, 1,1);
//*-*   With this setup, parameters 0->2 can vary freely
//*-*   Parameter 3 has boundaries [-10,-4] with initial value -8
//*-*   Parameter 4 is fixed to 0
//*-*   Parameter 5 is fixed to 100.
//*-*   When the lower limit and upper limit are equal, teh parameter is fixed.
//*-*   However to fix a parameter to 0, one must call the FixParameter function.
//*-*
//*-*   Note that option "I" gives better results but is slower.
//*-*
//*-*
//*-*   Changing the fitting function
//*-*   =============================
//*-*  By default the fitting function H1FitChisquare is used.
//*-*  To specify a User defined fitting function, specify option "U" and
//*-*  call the following functions:
//*-*    TVirtualFitter::Fitter(myhist)->SetFCN(MyFittingFunction)
//*-*  where MyFittingFunction is of type:
//*-*  extern void MyFittingFunction(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
//*-*
//*-*  Associated functions
//*-*  ====================
//*-*  One or more object (typically a TF1*) can be added to the list
//*-*  of functions (fFunctions) associated to each histogram.
//*-*  When TH1::Fit is invoked, the fitted function is added to this list.
//*-*  Given an histogram h, one can retrieve an associated function
//*-*  with:  TF1 *myfunc = h->GetFunction("myfunc");
//*-*
//*-*   Access to the fit results
//*-*   =========================
//*-*  If the histogram is made persistent, the list of
//*-*  associated functions is also persistent. Given a pointer (see above)
//*-*  to an associated function myfunc, one can retrieve the function/fit
//*-*  parameters with calls such as:
//*-*    Double_t chi2 = myfunc->GetChisquare();
//*-*    Double_t par0 = myfunc->GetParameter(0); //value of 1st parameter
//*-*    Double_t err0 = myfunc->GetParError(0);  //error on first parameter
//*-*
//*-*   Changing the maximum number of parameters
//*-*   =========================================
//*-*  By default, the fitter TMinuit is initialized with a maximum of 25 parameters.
//*-*  You can redefine this default value by calling :
//*-*    TVirtualFitter::Fitter(0,150); //to get a maximum of 150 parameters
//*-*
//*-*   Warning when using the option "0"
//*-*   =================================
//*-*  When selecting the option "0", the fitted function is added to 
//*-*  the list of functions of the histogram, but it is not drawn. 
//*-*  You can undo what you disabled in the following way:
//*-*    h.Fit("myFunction","0"); // fit, store function but do not draw
//*-*    h.Draw(); function is not drawn
//*-*    const Int_t kNotDraw = 1<<9;
//*-*    h.GetFunction("myFunction")->ResetBit(kNotDraw);
//*-*    h.Draw();  // function is visible again
//*-*  
//*-*  By default, the fitter TMinuit is initialized with a maximum of 25 parameters.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   Int_t i, npar,nvpar,nparx;
   Double_t par, we, al, bl;
   Double_t eplus,eminus,eparab,globcc,amin,edm,errdef,werr;
   Double_t params[100], arglist[100];
   TF1 *fnew1;
   TF2 *fnew2;
   TF3 *fnew3;

   xfirst  = fXaxis.GetFirst();
   xlast   = fXaxis.GetLast();
   binwidx = fXaxis.GetBinWidth(xlast);
   xmin    = fXaxis.GetBinLowEdge(xfirst);
   xmax    = fXaxis.GetBinLowEdge(xlast) +binwidx;
   yfirst  = fYaxis.GetFirst();
   ylast   = fYaxis.GetLast();
   binwidy = fYaxis.GetBinWidth(ylast);
   ymin    = fYaxis.GetBinLowEdge(yfirst);
   ymax    = fYaxis.GetBinLowEdge(ylast) +binwidy;
   zfirst  = fZaxis.GetFirst();
   zlast   = fZaxis.GetLast();
   binwidz = fZaxis.GetBinWidth(zlast);
   zmin    = fZaxis.GetBinLowEdge(zfirst);
   zmax    = fZaxis.GetBinLowEdge(zlast) +binwidz;
   xaxis   = &fXaxis;
   yaxis   = &fYaxis;
   zaxis   = &fZaxis;

//*-*- Check if Minuit is initialized and create special functions
   hFitter = TVirtualFitter::Fitter(this);
   hFitter->Clear();

//*-*- Get pointer to the function by searching in the list of functions in ROOT
   gF1 = f1;
   if (!gF1) { Error("Fit", "Pointer to function is null"); return; }
   npar = gF1->GetNpar();
   if (npar <=0) { Error("Fit", "Illegal number of parameters = %d",npar); return; }

//*-*- Check that function has same dimension as histogram
   if (gF1->GetNdim() == 1 && GetDimension() > 1) {
      Error("Fit", "Function %s is not 2-D",f1->GetName()); return; }
   if (gF1->GetNdim() == 2 && GetDimension() < 2) {
      Error("Fit", "Function %s is not 1-D",f1->GetName()); return; }
   if (xxmin != xxmax) gF1->SetRange(xxmin,ymin,zmin,xxmax,ymax,zmax);

//*-*- Decode list of options into Foption
   if (!FitOptionsMake(option)) return;
   if (xxmin != xxmax) {
      gF1->SetRange(xxmin,ymin,zmin,xxmax,ymax,zmax);
      Foption.Range = 1;
   }
//*-*- Is a Fit range specified?
   Double_t fxmin, fymin, fzmin, fxmax, fymax, fzmax;
   if (Foption.Range) {
      gF1->GetRange(fxmin, fymin, fzmin, fxmax, fymax, fzmax);
      if (fxmin > xmin) xmin = fxmin;
      if (fymin > ymin) ymin = fymin;
      if (fzmin > zmin) zmin = fzmin;
      if (fxmax < xmax) xmax = fxmax;
      if (fymax < ymax) ymax = fymax;
      if (fzmax < zmax) zmax = fzmax;
      xfirst = fXaxis.FindFixBin(xmin); if (xfirst < 1) xfirst = 1;
      xlast  = fXaxis.FindFixBin(xmax); if (xlast > fXaxis.GetLast()) xlast = fXaxis.GetLast();
      yfirst = fYaxis.FindFixBin(ymin); if (yfirst < 1) yfirst = 1;
      ylast  = fYaxis.FindFixBin(ymax); if (ylast > fYaxis.GetLast()) ylast = fYaxis.GetLast();
      zfirst = fZaxis.FindFixBin(zmin); if (zfirst < 1) zfirst = 1;
      zlast  = fZaxis.FindFixBin(zmax); if (zlast > fZaxis.GetLast()) zlast = fZaxis.GetLast();
   } else {
      gF1->SetRange(xmin,ymin,zmin,xmax,ymax,zmax);
   }

//*-*- If case of a predefined function, then compute initial values of parameters
   Int_t special = gF1->GetNumber();
   if (Foption.Bound) special = 0;
   if      (special == 100)      H1InitGaus();
   else if (special == 400)      H1InitGaus();
   else if (special == 200)      H1InitExpo();
   else if (special == 299+npar) H1InitPolynom();

//*-*- Some initialisations
   if (!Foption.Verbose) {
      arglist[0] = -1;
      hFitter->ExecuteCommand("SET PRINT", arglist,1);
      arglist[0] = 0;
      hFitter->ExecuteCommand("SET NOW",   arglist,0);
   }

//*-*- Set error criterion for chisquare or likelihood methods
//*-*-  MINUIT ERRDEF should not be set to 0.5 in case of loglikelihood fit.
//*-*-  because the FCN is already multiplied by 2 in H1FitLikelihood
//*-*-  if Hoption.User is specified, assume that the user has already set
//*-*-  his minimization function via SetFCN.
   arglist[0] = 1;
   if (Foption.Like) {
      hFitter->SetFCN(H1FitLikelihood);
   } else {
      if (!Foption.User) hFitter->SetFCN(H1FitChisquare);
   }
   hFitter->ExecuteCommand("SET ERR",arglist,1);

//*-*- Transfer names and initial values of parameters to Minuit
   Int_t nfixed = 0;
   for (i=0;i<npar;i++) {
      par = gF1->GetParameter(i);
      gF1->GetParLimits(i,al,bl);
      if (al*bl != 0 && al >= bl) {
         al = bl = 0;
         arglist[nfixed] = i+1;
         nfixed++;
      }
      we = 0.1*TMath::Abs(bl-al);
      if (we == 0) we = 0.3*TMath::Abs(par);
//      if (we == 0) we = 1000*TMath::Abs(par);
      if (we == 0) we = binwidx;
      hFitter->SetParameter(i,gF1->GetParName(i),par,we,al,bl);
   }
   if(nfixed > 0)hFitter->ExecuteCommand("FIX",arglist,nfixed); // Otto

//*-*- Set Gradient
   if (Foption.Gradient) {
      if (Foption.Gradient == 1) arglist[0] = 1;
      else                       arglist[0] = 0;
      hFitter->ExecuteCommand("SET GRAD",arglist,1);
   }

//*-*- Reset Print level
   if (Foption.Verbose) {
      arglist[0] = 0; hFitter->ExecuteCommand("SET PRINT", arglist,1);
   }

//*-*- Compute sum of squares of errors in the bin range
   Double_t ey, sumw2=0;
   for (i=xfirst;i<=xlast;i++) {
      ey = GetBinError(i);
      sumw2 += ey*ey;
   }

//*-*- Perform minimization
   arglist[0] = TVirtualFitter::GetMaxIterations();
   arglist[1] = sumw2*TVirtualFitter::GetPrecision();
   hFitter->ExecuteCommand("MIGRAD",arglist,2);
   if (Foption.More) {
      hFitter->ExecuteCommand("IMPROVE",arglist,0);
   }
   if (Foption.Errors) {
      hFitter->ExecuteCommand("HESSE",arglist,0);
      hFitter->ExecuteCommand("MINOS",arglist,0);
   }

//*-*- Get return status
   char parName[50];
   for (i=0;i<npar;i++) {
      hFitter->GetParameter(i,parName, par,we,al,bl);
      if (Foption.Errors) werr = we;
      else {
         hFitter->GetErrors(i,eplus,eminus,eparab,globcc);
         if (eplus > 0 && eminus < 0) werr = 0.5*(eplus-eminus);
         else                         werr = we;
      }
      params[i] = par;
      gF1->SetParameter(i,par);
      gF1->SetParError(i,werr);
   }
   hFitter->GetStats(amin,edm,errdef,nvpar,nparx);

//*-*- Print final values of parameters.
   if (!Foption.Quiet) {
      if (Foption.Errors) hFitter->PrintResults(4,amin);
      else                hFitter->PrintResults(3,amin);
   }

//*-*  If Log Likelihood, compute an equivalent chisquare
   if (Foption.Like) H1FitChisquare(npar, params, amin, params, 1);

   gF1->SetChisquare(amin);
   gF1->SetNDF(gF1->GetNumberFitPoints()-npar+nfixed);
   
//*-*- Store fitted function in histogram functions list and draw
   if (!Foption.Nostore) {
      if (!Foption.Plus) {
         TIter next(fFunctions, kIterBackward);
         TObject *obj;
         while ((obj = next())) {
            if (obj->InheritsFrom(TF1::Class())) {
               fFunctions->Remove(obj);
               delete obj;
            }
         }
      }
      if (GetDimension() < 2) {
         fnew1 = new TF1();
         gF1->Copy(*fnew1);
         fFunctions->Add(fnew1);
         fnew1->SetParent(this);
         fnew1->Save(xmin,xmax,0,0,0,0);
         if (Foption.Nograph) fnew1->SetBit(TF1::kNotDraw);
         fnew1->SetBit(TFormula::kNotGlobal);
      } else if (GetDimension() < 3) {
         fnew2 = new TF2();
         gF1->Copy(*fnew2);
         fFunctions->Add(fnew2);
         fnew2->SetParent(this);
         fnew2->Save(xmin,xmax,ymin,ymax,0,0);
         if (Foption.Nograph) fnew2->SetBit(TF1::kNotDraw);
         fnew2->SetBit(TFormula::kNotGlobal);
      } else {
         fnew3 = new TF3();
         gF1->Copy(*fnew3);
         fFunctions->Add(fnew3);
         fnew3->SetParent(this);
         fnew3->SetBit(TFormula::kNotGlobal);
      }
      if (TestBit(kCanDelete)) return;
      if (!Foption.Nograph && GetDimension() < 3) Draw(goption);
  }
}

//______________________________________________________________________________
void TH1::FitPanel()
{
//*-*-*-*-*-*-*Display a panel with all histogram fit options*-*-*-*-*-*
//*-*          ==============================================
//*-*
//*-*   See class TFitPanel for example

   if (fPainter) fPainter->FitPanel();
}

//______________________________________________________________________________
char *TH1::GetObjectInfo(Int_t px, Int_t py) const
{
//   Redefines TObject::GetObjectInfo.
//   Displays the histogram info (bin number, contents, integral up to bin
//   corresponding to cursor position px,py
//
   return fPainter->GetObjectInfo(px,py);
}

//______________________________________________________________________________
Int_t TH1::FitOptionsMake(Option_t *choptin)
{
//*-*-*-*-*-*-*-*-*Decode string choptin and fill Foption structure*-*-*-*-*-*
//*-*              ================================================

   Foption.Quiet   = 0;
   Foption.Verbose = 0;
   Foption.Bound   = 0;
   Foption.Like    = 0;
   Foption.User    = 0;
   Foption.W1      = 0;
   Foption.Errors  = 0;
   Foption.More    = 0;
   Foption.Range   = 0;
   Foption.Gradient= 0;
   Foption.Nograph = 0;
   Foption.Nostore = 0;
   Foption.Plus    = 0;
   Foption.Integral= 0;

   Int_t nch = strlen(choptin);
   if (!nch) return 1;

   char chopt[32];
   strcpy(chopt,choptin);

   for (Int_t i=0;i<nch;i++) chopt[i] = toupper(choptin[i]);

   if (strstr(chopt,"Q")) Foption.Quiet   = 1;
   if (strstr(chopt,"V")){Foption.Verbose = 1; Foption.Quiet = 0;}
   if (strstr(chopt,"L")) Foption.Like    = 1;
   if (strstr(chopt,"W")) Foption.W1      = 1;
   if (strstr(chopt,"E")) Foption.Errors  = 1;
   if (strstr(chopt,"M")) Foption.More    = 1;
   if (strstr(chopt,"R")) Foption.Range   = 1;
   if (strstr(chopt,"G")) Foption.Gradient= 1;
   if (strstr(chopt,"N")) Foption.Nostore = 1;
   if (strstr(chopt,"0")) Foption.Nograph = 1;
   if (strstr(chopt,"+")) Foption.Plus    = 1;
   if (strstr(chopt,"I")) Foption.Integral= 1;
   if (strstr(chopt,"B")) Foption.Bound   = 1;
   if (strstr(chopt,"U")){Foption.User    = 1; Foption.Like = 0;}
   return 1;
}

//______________________________________________________________________________
void H1FitChisquare(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag)
{
//*-*-*-*-*-*Minimization function for H1s using a Chisquare method*-*-*-*-*-*
//*-*        ======================================================

   Double_t cu,eu,fu,fsum;
   Double_t dersum[100], grad[100];
   Double_t x[3];
   Int_t bin,binx,biny,binz,k;
   Axis_t binlow, binup, binsize;

   Int_t npfits = 0;

   npar = gF1->GetNpar();
   if (flag == 2) for (k=0;k<npar;k++) dersum[k] = gin[k] = 0;

   TH1 *hfit = (TH1*)hFitter->GetObjectFit();
   gF1->InitArgs(x,u);
   f = 0;
   for (binz=zfirst;binz<=zlast;binz++) {
      x[2]  = zaxis->GetBinCenter(binz);
      for (biny=yfirst;biny<=ylast;biny++) {
         x[1]  = yaxis->GetBinCenter(biny);
         for (binx=xfirst;binx<=xlast;binx++) {
            bin = hfit->GetBin(binx,biny,binz);
            cu  = hfit->GetBinContent(bin);
            x[0]  = xaxis->GetBinCenter(binx);
            if (Foption.Integral) {
               binlow  = xaxis->GetBinLowEdge(binx);
               binsize = xaxis->GetBinWidth(binx);
               binup   = binlow + binsize;
               fu      = gF1->Integral(binlow,binup,u)/binsize;
            } else {
               fu = gF1->EvalPar(x,u);
            }
            if (Foption.W1) {
               if (cu == 0) continue;
               eu = 1;
            } else {
               eu  = hfit->GetBinError(bin);
               if (eu <= 0) continue;
            }
            if (flag == 2) {
               for (k=0;k<npar;k++) dersum[k] += 1; //should be the derivative
            }
            npfits++;
            if (flag == 2) {
               for (k=0;k<npar;k++) grad[k] += dersum[k]*(fu-cu)/eu; dersum[k] = 0;
            }
            fsum = (cu-fu)/eu;
            f += fsum*fsum;
         }
      }
   }
//printf("f=%g, npfits=%d\n",f,npfits);
   gF1->SetNumberFitPoints(npfits);
}

//______________________________________________________________________________
void H1FitLikelihood(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag)
{
//*-*-*-*-*-*Minimization function for H1s using a Likelihood method*-*-*-*-*-*
//*-*        =======================================================
//     Basically, it forms the likelihood by determining the Poisson
//     probability that given a number of entries in a particualar bin,
//     the fit would predict it's value.  This is then done for each bin,
//     and the sum of the logs is taken as the likelihood.

   Double_t cu,fu,fobs,fsub;
   Double_t dersum[100];
   Double_t x[3];
   Int_t bin,binx,biny,binz,k,icu;
   Axis_t binlow, binup, binsize;

   Int_t npfits = 0;

   npar = gF1->GetNpar();
   if (flag == 2) for (k=0;k<npar;k++) dersum[k] = gin[k] = 0;

   TH1 *hfit = (TH1*)hFitter->GetObjectFit();
   gF1->InitArgs(x,u);
   f = 0;
   for (binz=zfirst;binz<=zlast;binz++) {
      x[2]  = zaxis->GetBinCenter(binz);
      for (biny=yfirst;biny<=ylast;biny++) {
         x[1]  = yaxis->GetBinCenter(biny);
         for (binx=xfirst;binx<=xlast;binx++) {
            bin = hfit->GetBin(binx,biny,binz);
            cu  = hfit->GetBinContent(bin);
            x[0]  = xaxis->GetBinCenter(binx);
            if (Foption.Integral) {
               binlow  = xaxis->GetBinLowEdge(binx);
               binsize = xaxis->GetBinWidth(binx);
               binup   = binlow + binsize;
               fu      = gF1->Integral(binlow,binup,u)/binsize;
            } else {
               fu = gF1->EvalPar(x,u);
            }
            npfits++;
            if (flag == 2) {
               for (k=0;k<npar;k++) {
                  dersum[k] += 1; //should be the derivative
                  //grad[k] += dersum[k]*(fu-cu)/eu; dersum[k] = 0;
               }
            }
            if (fu < 1.e-9) fu = 1.e-9;
            icu  = Int_t(cu);
            fsub = -fu +icu*TMath::Log(fu);
            fobs = hFitter->GetSumLog(icu);

            fsub -= fobs;
            f -= fsub;
         }
      }
   }
   f *= 2;
   gF1->SetNumberFitPoints(npfits);
}

//______________________________________________________________________________
void H1InitGaus()
{
//*-*-*-*-*-*Compute Initial values of parameters for a gaussian*-*-*-*-*-*-*
//*-*        ===================================================

   Double_t allcha, sumx, sumx2, x, val, rms, mean;
   Int_t bin;
   static Double_t sqrtpi = 2.506628;

//*-*- Compute mean value and RMS of the histogram in the given range
   TH1 *curHist = (TH1*)hFitter->GetObjectFit();
   Double_t valmax = curHist->GetBinContent(xfirst);
   allcha = sumx = sumx2 = 0;
   for (bin=xfirst;bin<=xlast;bin++) {
      x       = curHist->GetBinCenter(bin);
      val     = TMath::Abs(curHist->GetBinContent(bin));
      if (val > valmax) valmax = val;
      sumx   += val*x;
      sumx2  += val*x*x;
      allcha += val;
   }
   if (allcha == 0) return;
   mean = sumx/allcha;
   rms  = TMath::Sqrt(sumx2/allcha - mean*mean);
   if (rms == 0) rms = binwidx*(xlast-xfirst+1)/4;
   //if the distribution is really gaussian, the best approximation
   //is binwidx*allcha/(sqrtpi*rms)
   //However, in case of non-gaussian tails, this underestimates
   //the normalisation constant. In this case the maximum value
   //is a better approximation.
   //We take the average of both quantities
   Double_t constant = 0.5*(valmax+binwidx*allcha/(sqrtpi*rms));

   //In case the mean value is outside the histo limits and
   //the RMS is bigger than the range, we take
   //  mean = center of bins
   //  rms  = half range
   Axis_t xmin = curHist->GetXaxis()->GetXmin();
   Axis_t xmax = curHist->GetXaxis()->GetXmax();
   if ((mean < xmin || mean > xmax) && rms > (xmax-xmin)) {
      mean = 0.5*(xmax+xmin);
      rms  = 0.5*(xmax-xmin);
   }
   gF1->SetParameter(0,constant);
   gF1->SetParameter(1,mean);
   gF1->SetParameter(2,rms);
   gF1->SetParLimits(2,0,10*rms);
}

//______________________________________________________________________________
void H1InitExpo()
{
//*-*-*-*-*-*Compute Initial values of parameters for an exponential*-*-*-*-*
//*-*        =======================================================

   Double_t constant, slope;
   Int_t ifail;
   Int_t nchanx = xlast - xfirst + 1;

   H1LeastSquareLinearFit(-nchanx, constant, slope, ifail);

   gF1->SetParameter(0,constant);
   gF1->SetParameter(1,slope);

}

//______________________________________________________________________________
void H1InitPolynom()
{
//*-*-*-*-*-*Compute Initial values of parameters for a polynom*-*-*-*-*-*-*
//*-*        ===================================================

   Double_t fitpar[25];

   Int_t nchanx = xlast - xfirst + 1;
   Int_t npar   = gF1->GetNpar();

   if (nchanx <=1 || npar == 1) {
      TH1 *curHist = (TH1*)hFitter->GetObjectFit();
      fitpar[0] = curHist->GetSumOfWeights()/Double_t(nchanx);
   } else {
      H1LeastSquareFit( nchanx, npar, fitpar);
   }
   for (Int_t i=0;i<npar;i++) gF1->SetParameter(i, fitpar[i]);
}

//______________________________________________________________________________
void H1LeastSquareFit(Int_t n, Int_t m, Double_t *a)
{
//*-*-*-*-*-*-*-*Least squares lpolynomial fitting without weights*-*-*-*-*-*-*
//*-*            =================================================
//*-*
//*-*  n   number of points to fit
//*-*  m   number of parameters
//*-*  a   array of parameters
//*-*
//*-*   based on CERNLIB routine LSQ: Translated to C++ by Rene Brun
//*-*   (E.Keil.  revised by B.Schorr, 23.10.1981.)
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    const Double_t zero = 0.;
    const Double_t one = 1.;
    const Int_t idim = 20;

    Double_t  b[400]	/* was [20][20] */;
    Int_t i, k, l, ifail;
    Double_t power;
    Double_t da[20], xk, yk;

    if (m <= 2) {
       H1LeastSquareLinearFit(n, a[0], a[1], ifail);
       return;
    }
    if (m > idim || m > n) return;
    b[0]  = Double_t(n);
    da[0] = zero;
    for (l = 2; l <= m; ++l) {
	b[l-1]           = zero;
	b[m + l*20 - 21] = zero;
	da[l-1]          = zero;
    }
    TH1 *curHist = (TH1*)hFitter->GetObjectFit();
    for (k = xfirst; k <= xlast; ++k) {
	xk     = curHist->GetBinCenter(k);
	yk     = curHist->GetBinContent(k);
	power  = one;
	da[0] += yk;
	for (l = 2; l <= m; ++l) {
	    power   *= xk;
	    b[l-1]  += power;
	    da[l-1] += power*yk;
	}
	for (l = 2; l <= m; ++l) {
	    power            *= xk;
	    b[m + l*20 - 21] += power;
	}
    }
    for (i = 3; i <= m; ++i) {
	for (k = i; k <= m; ++k) {
	    b[k - 1 + (i-1)*20 - 21] = b[k + (i-2)*20 - 21];
	}
    }
    H1LeastSquareSeqnd(m, b, idim, ifail, 1, da);

    for (i=0; i<m; ++i) a[i] = da[i];

}

//______________________________________________________________________________
void H1LeastSquareLinearFit(Int_t ndata, Double_t &a0, Double_t &a1, Int_t &ifail)
{
//*-*-*-*-*-*-*-*-*-*Least square linear fit without weights*-*-*-*-*-*-*-*-*
//*-*                =======================================
//*-*
//*-*   extracted from CERNLIB LLSQ: Translated to C++ by Rene Brun
//*-*   (added to LSQ by B. Schorr, 15.02.1982.)
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    Double_t xbar, ybar, x2bar;
    Int_t i, n;
    Double_t xybar;
    Double_t fn, xk, yk;
    Double_t det;

    n     = TMath::Abs(ndata);
    ifail = -2;
    xbar  = ybar = x2bar = xybar = 0;
    TH1 *curHist = (TH1*)hFitter->GetObjectFit();
    for (i = xfirst; i <= xlast; ++i) {
	xk = curHist->GetBinCenter(i);
	yk = curHist->GetBinContent(i);
	if (ndata < 0) {
	    if (yk <= 0) yk = 1e-9;
	    yk = TMath::Log(yk);
	}
	xbar  += xk;
	ybar  += yk;
	x2bar += xk*xk;
	xybar += xk*yk;
    }
    fn    = Double_t(n);
    det   = fn*x2bar - xbar*xbar;
    ifail = -1;
    if (det <= 0) {
       a0 = ybar/fn;
       a1 = 0;
       return;
    }
    ifail = 0;
    a0 = (x2bar*ybar - xbar*xybar) / det;
    a1 = (fn*xybar - xbar*ybar) / det;

}

//______________________________________________________________________________
void H1LeastSquareSeqnd(Int_t n, Double_t *a, Int_t idim, Int_t &ifail, Int_t k, Double_t *b)
{
//*-*-*-*-*-*-*-*Extracted from CERN Program library routine DSEQN*-*-*-*-*-*
//*-*            =================================================
//*-*
//*-*        : Translated to C++ by Rene Brun
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    Int_t a_dim1, a_offset, b_dim1, b_offset;
    Int_t nmjp1, i, j, l;
    Int_t im1, jp1, nm1, nmi;
    Double_t s1, s21, s22;
    const Double_t one = 1.;

    /* Parameter adjustments */
    b_dim1 = idim;
    b_offset = b_dim1 + 1;
    b -= b_offset;
    a_dim1 = idim;
    a_offset = a_dim1 + 1;
    a -= a_offset;

    if (idim < n) return;

    ifail = 0;
    for (j = 1; j <= n; ++j) {
	if (a[j + j*a_dim1] <= 0) { ifail = -1; return; }
	a[j + j*a_dim1] = one / a[j + j*a_dim1];
	if (j == n) continue;
	jp1 = j + 1;
	for (l = jp1; l <= n; ++l) {
	    a[j + l*a_dim1] = a[j + j*a_dim1] * a[l + j*a_dim1];
	    s1 = -a[l + (j+1)*a_dim1];
	    for (i = 1; i <= j; ++i) { s1 = a[l + i*a_dim1] * a[i + (j+1)*a_dim1] + s1; }
	    a[l + (j+1)*a_dim1] = -s1;
	}
    }
    if (k <= 0) return;

    for (l = 1; l <= k; ++l) {
	b[l*b_dim1 + 1] = a[a_dim1 + 1]*b[l*b_dim1 + 1];
    }
    if (n == 1) return;
    for (l = 1; l <= k; ++l) {
	for (i = 2; i <= n; ++i) {
	    im1 = i - 1;
	    s21 = -b[i + l*b_dim1];
	    for (j = 1; j <= im1; ++j) {
		s21 = a[i + j*a_dim1]*b[j + l*b_dim1] + s21;
	    }
	    b[i + l*b_dim1] = -a[i + i*a_dim1]*s21;
	}
	nm1 = n - 1;
	for (i = 1; i <= nm1; ++i) {
	    nmi = n - i;
	    s22 = -b[nmi + l*b_dim1];
	    for (j = 1; j <= i; ++j) {
		nmjp1 = n - j + 1;
		s22 = a[nmi + nmjp1*a_dim1]*b[nmjp1 + l*b_dim1] + s22;
	    }
	    b[nmi + l*b_dim1] = -s22;
	}
    }
}


//______________________________________________________________________________
Int_t TH1::GetBin(Int_t binx, Int_t biny, Int_t binz) const
{
//*-*-*-*-*-*Return Global bin number corresponding to binx,y,z*-*-*-*-*-*-*
//*-*        ==================================================
//*-*
//*-*   2-D and 3-D histograms are represented with a one dimensional
//*-*   structure.
//*-*   This has the advantage that all existing functions, such as
//*-*     GetBinContent, GetBinError, GetBinFunction work for all dimensions.
//*-*
//*-*  In case of a TH1x, returns binx directly.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   Int_t nx, ny, nz;
   if (GetDimension() < 2) {
      nx  = fXaxis.GetNbins()+2;
      if (binx < 0)   binx = 0;
      if (binx >= nx) binx = nx-1;
      return binx;
   }
   if (GetDimension() < 3) {
      nx  = fXaxis.GetNbins()+2;
      if (binx < 0)   binx = 0;
      if (binx >= nx) binx = nx-1;
      ny  = fYaxis.GetNbins()+2;
      if (biny < 0)   biny = 0;
      if (biny >= ny) biny = ny-1;
      return  binx + nx*biny;
   }
   if (GetDimension() < 4) {
      nx  = fXaxis.GetNbins()+2;
      if (binx < 0)   binx = 0;
      if (binx >= nx) binx = nx-1;
      ny  = fYaxis.GetNbins()+2;
      if (biny < 0)   biny = 0;
      if (biny >= ny) biny = ny-1;
      nz  = fZaxis.GetNbins()+2;
      if (binz < 0)   binz = 0;
      if (binz >= nz) binz = nz-1;
      return  binx + nx*(biny +ny*binz);
   }
   return -1;
}

//______________________________________________________________________________
Axis_t TH1::GetRandom()
{
// return a random number distributed according the histogram bin contents.
// This function checks if the bins integral exists. If not, the integral
// is evaluated, normalized to one.
// The integral is automatically recomputed if the number of entries
// is not the same then when the integral was computed.
// NB Only valid for 1-d histograms. Use GetRandom2 or 3 otherwise.

   if (fDimension > 1) {
      Error("GetRandom","Function only valid for 1-d histograms");
      return 0;
   }
   Int_t nbinsx = GetNbinsX();
   Double_t integral;
   if (fIntegral) {
      if (fIntegral[nbinsx+1] != fEntries) integral = ComputeIntegral();
   } else {
      integral = ComputeIntegral();
      if (integral == 0 || fIntegral == 0) return 0;
   }
   Double_t r1 = gRandom->Rndm();
   Int_t ibin = TMath::BinarySearch(nbinsx,&fIntegral[0],r1);
   return GetBinLowEdge(ibin+1)
      +GetBinWidth(ibin+1)*(fIntegral[ibin+1]-r1)/(fIntegral[ibin+1] - fIntegral[ibin]);
}

//______________________________________________________________________________
Stat_t TH1::GetBinContent(Int_t) const
{
//*-*-*-*-*-*-*Return content of bin number bin
//*-*          ================================
// Implemented in TH1C,S,F,D

   AbstractMethod("GetBinContent");
   return 0;
}

//______________________________________________________________________________
Stat_t TH1::GetBinContent(Int_t binx, Int_t biny) const
{
//*-*-*-*-*-*-*Return content of bin number binx, biny
//*-*          =======================================
// NB: Function to be called for 2-d histograms only

   Int_t bin = GetBin(binx,biny);
   return GetBinContent(bin);
}

//______________________________________________________________________________
Stat_t TH1::GetBinContent(Int_t binx, Int_t biny, Int_t binz) const
{
//*-*-*-*-*-*-*Return content of bin number binx,biny,binz
//*-*          ===========================================
// NB: Function to be called for 3-d histograms only

   Int_t bin = GetBin(binx,biny,binz);
   return GetBinContent(bin);
}

//______________________________________________________________________________
void TH1::Multiply(TF1 *f1, Double_t c1)
{
   // Performs the operation: this = this*c1*f1
   // if errors are defined (see TH1::Sumw2), errors are also recalculated.

   if (!f1) {
      Error("Add","Attempt to multiply by a non-existing function");
      return;
   }

   Int_t nbinsx = GetNbinsX();
   Int_t nbinsy = GetNbinsY();
   Int_t nbinsz = GetNbinsZ();
   if (fDimension < 2) nbinsy = -1;
   if (fDimension < 3) nbinsz = -1;

//*-*- Add statistics
   Stat_t s1[10];
   Int_t i;
   for (i=0;i<10;i++) {s1[i] = 0;}
   PutStats(s1);

//*-*- Loop on bins (including underflows/overflows)
   Int_t bin, binx, biny, binz;
   Double_t cu,w;
   Double_t xx[3];
   Double_t *params = 0;
   f1->InitArgs(xx,params);
   for (binz=0;binz<=nbinsz+1;binz++) {
      xx[2] = fZaxis.GetBinCenter(binz);
      for (biny=0;biny<=nbinsy+1;biny++) {
         xx[1] = fYaxis.GetBinCenter(biny);
         for (binx=0;binx<=nbinsx+1;binx++) {
            xx[0] = fXaxis.GetBinCenter(binx);
            bin = binx +(nbinsx+2)*(biny + (nbinsy+2)*binz);
            Double_t error1 = GetBinError(bin);
            cu  = f1->EvalPar(xx);
            w = GetBinContent(bin)*c1*cu;
            SetBinContent(bin,w);
            if (fSumw2.fN) {
               fSumw2.fArray[bin] = c1*c1*error1*error1;
            }
         }
      }
   }
}

//______________________________________________________________________________
void TH1::Multiply(TH1 *h1)
{
//*-*-*-*-*-*-*-*-*-*-*Multiply this histogram by h1*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =============================
//
//   this = this*h1
//
//   If errors of this are available (TH1::Sumw2), errors are recalculated.
//   Note that if h1 has Sumw2 set, Sumw2 is automatically called for this
//   if not already set.

   if (!h1) {
      Error("Multiply","Attempt to multiply by a non-existing histogram");
      return;
   }

   Int_t nbinsx = GetNbinsX();
   Int_t nbinsy = GetNbinsY();
   Int_t nbinsz = GetNbinsZ();
//*-*- Check histogram compatibility
   if (nbinsx != h1->GetNbinsX() || nbinsy != h1->GetNbinsY() || nbinsz != h1->GetNbinsZ()) {
      Error("Multiply","Attempt to multiply histograms with different number of bins");
      return;
   }
//*-*- Issue a Warning if histogram limits are different
   if (GetXaxis()->GetXmin() != h1->GetXaxis()->GetXmin() ||
       GetXaxis()->GetXmax() != h1->GetXaxis()->GetXmax() ||
       GetYaxis()->GetXmin() != h1->GetYaxis()->GetXmin() ||
       GetYaxis()->GetXmax() != h1->GetYaxis()->GetXmax() ||
       GetZaxis()->GetXmin() != h1->GetZaxis()->GetXmin() ||
       GetZaxis()->GetXmax() != h1->GetZaxis()->GetXmax()) {
       Warning("Multiply","Attempt to multiply histograms with different axis limits");
   }
   if (fDimension < 2) nbinsy = -1;
   if (fDimension < 3) nbinsz = -1;
   if (fDimension < 3) nbinsz = -1;

//*-* Create Sumw2 if h1 has Sumw2 set
   if (fSumw2.fN == 0 && h1->GetSumw2N() != 0) Sumw2();

//*-*- Reset statistics
   fEntries = fTsumw = 0;

//*-*- Loop on bins (including underflows/overflows)
   Int_t bin, binx, biny, binz;
   Double_t c0,c1,w;
   for (binz=0;binz<=nbinsz+1;binz++) {
      for (biny=0;biny<=nbinsy+1;biny++) {
         for (binx=0;binx<=nbinsx+1;binx++) {
            bin = GetBin(binx,biny,binz);
            c0  = GetBinContent(bin);
            c1  = h1->GetBinContent(bin);
            w   = c0*c1;
            SetBinContent(bin,w);
            fEntries++;
            if (fSumw2.fN) {
               Double_t e0 = GetBinError(bin);
               Double_t e1 = h1->GetBinError(bin);
               fSumw2.fArray[bin] = (e0*e0*c1*c1 + e1*e1*c0*c0);
            }
         }
      }
   }
   Stat_t s[10];
   GetStats(s);
   PutStats(s);
}


//______________________________________________________________________________
void TH1::Multiply(TH1 *h1, TH1 *h2, Double_t c1, Double_t c2, Option_t *option)
{
//*-*-*-*-*Replace contents of this histogram by multiplication of h1 by h2*-*
//*-*      ================================================================
//
//   this = (c1*h1)*(c2*h2)
//
//   If errors of this are available (TH1::Sumw2), errors are recalculated.
//   Note that if h1 or h2 have Sumw2 set, Sumw2 is automatically called for this
//   if not already set.
//

   TString opt = option;
   opt.ToLower();
//   Bool_t binomial = kFALSE;
//   if (opt.Contains("b")) binomial = kTRUE;
   if (!h1 || !h2) {
      Error("Multiply","Attempt to multiply by a non-existing histogram");
      return;
   }

   Int_t nbinsx = GetNbinsX();
   Int_t nbinsy = GetNbinsY();
   Int_t nbinsz = GetNbinsZ();
//*-*- Check histogram compatibility
   if (nbinsx != h1->GetNbinsX() || nbinsy != h1->GetNbinsY() || nbinsz != h1->GetNbinsZ()
    || nbinsx != h2->GetNbinsX() || nbinsy != h2->GetNbinsY() || nbinsz != h2->GetNbinsZ()) {
      Error("Multiply","Attempt to multiply histograms with different number of bins");
      return;
   }
//*-*- Issue a Warning if histogram limits are different
   if (GetXaxis()->GetXmin() != h1->GetXaxis()->GetXmin() ||
       GetXaxis()->GetXmax() != h1->GetXaxis()->GetXmax() ||
       GetYaxis()->GetXmin() != h1->GetYaxis()->GetXmin() ||
       GetYaxis()->GetXmax() != h1->GetYaxis()->GetXmax() ||
       GetZaxis()->GetXmin() != h1->GetZaxis()->GetXmin() ||
       GetZaxis()->GetXmax() != h1->GetZaxis()->GetXmax()) {
       Warning("Multiply","Attempt to multiply histograms with different axis limits");
   }
   if (GetXaxis()->GetXmin() != h2->GetXaxis()->GetXmin() ||
       GetXaxis()->GetXmax() != h2->GetXaxis()->GetXmax() ||
       GetYaxis()->GetXmin() != h2->GetYaxis()->GetXmin() ||
       GetYaxis()->GetXmax() != h2->GetYaxis()->GetXmax() ||
       GetZaxis()->GetXmin() != h2->GetZaxis()->GetXmin() ||
       GetZaxis()->GetXmax() != h2->GetZaxis()->GetXmax()) {
       Warning("Multiply","Attempt to multiply histograms with different axis limits");
   }
   if (fDimension < 2) nbinsy = -1;
   if (fDimension < 3) nbinsz = -1;

//*-* Create Sumw2 if h1 or h2 have Sumw2 set
   if (fSumw2.fN == 0 && (h1->GetSumw2N() != 0 || h2->GetSumw2N() != 0)) Sumw2();

//*-*- Reset statistics
   fEntries = fTsumw   = fTsumw2 = fTsumwx = fTsumwx2 = 0;

//*-*- Loop on bins (including underflows/overflows)
   Int_t bin, binx, biny, binz;
   Double_t b1,b2,w,d1,d2;
   d1 = c1*c1;
   d2 = c2*c2;
   for (binz=0;binz<=nbinsz+1;binz++) {
      for (biny=0;biny<=nbinsy+1;biny++) {
         for (binx=0;binx<=nbinsx+1;binx++) {
            bin = binx +(nbinsx+2)*(biny + (nbinsy+2)*binz);
            b1  = h1->GetBinContent(bin);
            b2  = h2->GetBinContent(bin);
            w   = (c1*b1)*(c2*b2);
            SetBinContent(bin,w);
            fEntries++;
            if (fSumw2.fN) {
               Double_t e1 = h1->GetBinError(bin);
               Double_t e2 = h2->GetBinError(bin);
               fSumw2.fArray[bin] = d1*d2*(e1*e1*b2*b2 + e2*e2*b1*b1);
            }
         }
      }
   }
   Stat_t s[10];
   GetStats(s);
   PutStats(s);
}

//______________________________________________________________________________
void TH1::Paint(Option_t *option)
{
//*-*-*-*-*-*-*-*-*Control routine to paint any kind of histograms*-*-*-*-*-*-*
//*-*              ===============================================
//
//  This function is automatically called by TCanvas::Update.
//  (see TH1::Draw for the list of options)

   if (!fPainter) fPainter = TVirtualHistPainter::HistPainter(this);
   if (fPainter) {
      if (strlen(option) > 0) fPainter->Paint(option);
      else                    fPainter->Paint(fOption.Data());
   }
}

//______________________________________________________________________________
TH1 *TH1::Rebin(Int_t ngroup, const char*newname)
{
//*-*-*-*-*Rebin this histogram grouping ngroup bins together*-*-*-*-*-*-*-*-*
//*-*      ==================================================
//   if newname is not blank a new temporary histogram hnew is created.
//   else the current histogram is modified (default)
//   The parameter ngroup indicates how many bins of this have to me merged
//   into one bin of hnew
//   If the original histogram has errors stored (via Sumw2), the resulting
//   histograms has new errors correctly calculated.
//
//   examples: if h1 is an existing TH1F histogram with 100 bins
//     h1->Rebin();  //merges two bins in one in h1: previous contents of h1 are lost
//     h1->Rebin(5); //merges five bins in one in h1
//     TH1F *hnew = h1->Rebin(5,"hnew"); // creates a new histogram hnew
//                                       //merging 5 bins of h1 in one bin
//   NOTE1: This function is currently implemented only for 1-D histograms
//
//   NOTE2: If ngroup is not an exact divider of the number of bins,
//          the top limit of the rebinned histogram is changed
//          to the upper edge of the bin=newbins*ngroup and the corresponding
//          bins are added to the overflow bin.
//          Statistics will be recomputed from the new bin contents.

   Int_t nbins   = fXaxis.GetNbins();
   Axis_t xmin  = fXaxis.GetXmin();
   Axis_t xmax  = fXaxis.GetXmax();
   if ((ngroup <= 0) || (ngroup > nbins)) {
      Error("Rebin", "Illegal value of ngroup=%d",ngroup);
      return 0;
   }
   if (fDimension > 1 || InheritsFrom("TProfile")) {
      Error("Rebin", "Operation valid on 1-D histograms only");
      return 0;
   }
   Int_t newbins = nbins/ngroup;

   // Save old bin contents into a new array
   Double_t *oldBins = new Double_t[nbins];
   Int_t bin, i;
   for (bin=0;bin<nbins;bin++) oldBins[bin] = GetBinContent(bin+1);
   Double_t *oldErrors = 0;
   if (fSumw2.fN != 0) {
      oldErrors = new Double_t[nbins];
      for (bin=0;bin<nbins;bin++) oldErrors[bin] = GetBinError(bin+1);
   }

   // create a clone of the old histogram if newname is specified
   TH1 *hnew = this;
   if (strlen(newname) > 0) {
      hnew = (TH1*)Clone();
      hnew->SetName(newname);
   }

   // change axis specs and rebuild bin contents array
   if(newbins*ngroup != nbins) {
      xmax = fXaxis.GetBinUpEdge(newbins*ngroup);
      hnew->fTsumw = 0; //stats must be reset because top bins will be moved to overflow bin
   }
   hnew->SetBins(newbins,xmin,xmax); //this also changes errors array (if any)

   // copy merged bin contents (ignore under/overflows)
   Int_t oldbin = 0;
   Double_t binContent, binError;
   for (bin = 0;bin<=newbins;bin++) {
      binContent = 0;
      binError   = 0;
      for (i=0;i<ngroup;i++) {
         if (oldbin+i >= nbins) break;
         binContent += oldBins[oldbin+i];
         if (oldErrors) binError += oldErrors[oldbin+i]*oldErrors[oldbin+i];
      }
      hnew->SetBinContent(bin+1,binContent);
      if (oldErrors) hnew->SetBinError(bin+1,TMath::Sqrt(binError));
      oldbin += ngroup;
   }

   delete [] oldBins;
   if (oldErrors) delete [] oldErrors;
   return hnew;
}

//______________________________________________________________________________
void TH1::RebinAxis(Axis_t x, const char *ax)
{
// Histogram is resized along ax such that x is in the axis range.
// The new axis limits are recomputed by doubling iteratively
// the current axis range until the specified value x is within the limits.
// The algorithm makes a copy of the histogram, then loops on all bins
// of the old histogram to fill the rebinned histogram.
// Takes into account errors (Sumw2) if any.
// The algorithm works for 1-d, 2-d and 3-d histograms.
// The bit kCanRebin must be set before invoking this function.
//  Ex:  h->SetBit(TH1::kCanRebin);

   if (!TestBit(kCanRebin)) return;
   char achoice = toupper(ax[0]);
   TAxis *axis = &fXaxis;
   if (achoice == 'Y') axis = &fYaxis;
   if (achoice == 'Z') axis = &fZaxis;
   Axis_t cxmin = axis->GetXmin();
   Axis_t cxmax = axis->GetXmax();
   if (cxmin > cxmax) return;
   Int_t  nbinsx = fXaxis.GetNbins();
   Int_t  nbinsy = fYaxis.GetNbins();
   Int_t  nbinsz = fZaxis.GetNbins();
   Axis_t range = cxmax-cxmin;

    //recompute new axis limits by doubling the current range
   Int_t bin;
   if (x < cxmin) {
      while (1) {
         range *= 2;
         if (x < cxmax-range) continue;
         xmin = cxmin - range/4;
         xmax = xmin + range;
         if (x < xmin) {
            xmin = cxmax - range;
            xmax = cxmax;
         }
         if ( xmin < 0 && x >= 0) {
            xmin = 0;
            xmax = range;
         }
         if (xmax >= 0 && cxmax <= 0) {
            xmax = 0;
            xmin = xmax - range;
         }
         break;
      } 
   } else {
      while (1) {
         range *= 2;
         if (x >= cxmin+range) continue;
         xmax = cxmax + range/4;
         xmin = xmax - range;
         if (x >= xmax) {
            xmax = cxmin + range;
            xmin = cxmin;
         }
         if ( xmax > 0 && x < 0) {
            xmax = 0;
            xmin = -range;
         }
         if (xmin < 0 && cxmin >= 0) {
            xmin = 0;
            xmax = xmin + range;
         }
         break;
      }
   }

   //save a copy of this histogram
   TH1 *hold = (TH1*)Clone();
   hold->SetDirectory(0);

   //set new axis limits
   axis->SetLimits(xmin,xmax);

   //now loop on all bins and refill
   Double_t err,cu;
   Axis_t bx,by,bz;
   Int_t errors = GetSumw2N();
   Int_t ix,iy,iz,ibin,binx,biny,binz;
   Reset("ICE"); //reset only Integral, contents and Errors
   for (binz=1;binz<=nbinsz;binz++) {
      bz  = hold->GetZaxis()->GetBinCenter(binz);
      iz  = fZaxis.FindFixBin(bz);
      for (biny=1;biny<=nbinsy;biny++) {
         by  = hold->GetYaxis()->GetBinCenter(biny);
         iy  = fYaxis.FindFixBin(by);
         for (binx=1;binx<=nbinsx;binx++) {
            bx = hold->GetXaxis()->GetBinCenter(binx);
            ix  = fXaxis.FindFixBin(bx);
            bin = hold->GetBin(binx,biny,binz);
            ibin= GetBin(ix,iy,iz);
            cu  = hold->GetBinContent(bin);
            AddBinContent(ibin,cu);
            if (errors) {
               err = hold->GetBinError(bin);
               fSumw2.fArray[ibin] += err*err;
            }
         }
      }
   }
   delete hold;
}

//______________________________________________________________________________
void TH1::Scale(Double_t c1)
{
//*-*-*-*-*Multiply this histogram by a constant c1*-*-*-*-*-*-*-*-*
//*-*      ========================================
//
//   this = c1*this
//
// Note that both contents and errors(if any) are scaled.
// This function uses the services of TH1::Add
//

   Double_t ent = fEntries;
   Add(this,this,c1,0);
   fEntries = ent;

   //if contours set, must also scale contours
   Int_t ncontours = GetContour();
   if (ncontours == 0) return;
   Double_t *levels = fContour.GetArray();
   for (Int_t i=0;i<ncontours;i++) {
      levels[i] *= c1;
   }
}
// -------------------------------------------------------------------------
void  TH1::SmoothArray(Int_t NN, Double_t *XX, Int_t ntimes)
{
// smooth array XX, translation of Hbook routine hsmoof.F
// based on algorithm 353QH twice presented by J. Friedman
// in Proc.of the 1974 CERN School of Computing, Norway, 11-24 August, 1974.

   Int_t ii, jj, ik, jk, kk, nn1, nn2;
   Double_t hh[6] = {0,0,0,0,0,0};
   Double_t *YY = new Double_t[NN];
   Double_t *ZZ = new Double_t[NN];
   Double_t *RR = new Double_t[NN];

   for (Int_t pass=0;pass<ntimes;pass++) {
      // first copy original data into temp array

      for (ii = 0; ii < NN; ii++) {
         YY[ii] = XX[ii];
      }

//  do 353 i.e. running median 3, 5, and 3 in a single loop
      for  (kk = 0; kk < 3; kk++)  {
         ik = 0;
         if  (kk == 1)  ik = 1;
         nn1 = ik + 1;
         nn2 = NN - ik - 1;
       // do all elements beside the first and last point for median 3
       //  and first two and last 2 for median 5
         for  (ii = nn1; ii < nn2; ii++)  {
            for  (jj = 0; jj < 3; jj++)   {
               hh[jj] = YY[ii + jj - 1];
            }
            ZZ[ii] = TH1::SmoothMedian(3 + 2*ik, hh);
         }

         if  (kk == 0)  {   // first median 3
// first point
            hh[0] = 3*YY[1] - 2*YY[2];
            hh[1] = YY[0];
            hh[2] = YY[2];
            ZZ[0] = TH1::SmoothMedian(3, hh);
// last point
            hh[0] = YY[NN - 2];
            hh[1] = YY[NN - 1];
            hh[2] = 3*YY[NN - 2] - 2*YY[NN - 3];
            ZZ[NN - 1] = TH1::SmoothMedian(3, hh);
         }
         if  (kk == 1)  {   //  median 5
  //  first point remains the same
            ZZ[0] = YY[0];
            for  (ii = 0; ii < 3; ii++) {
               hh[ii] = YY[ii];
            }
            ZZ[1] = TH1::SmoothMedian(3, hh);
// last two points
            for  (ii = 0; ii < 3; ii++) {
               hh[ii] = YY[NN - 3 + ii];
            }
            ZZ[NN - 2] = TH1::SmoothMedian(3, hh);
            ZZ[NN - 1] = YY[NN - 1];
         }
      }

// quadratic interpolation for flat segments
      nn2 = nn2 - 1;
      for (ii = nn1 + 1; ii < nn2; ii++) {
         if  (ZZ[ii - 1] != ZZ[ii]) continue;
         if  (ZZ[ii] != ZZ[ii + 1]) continue;
         hh[0] = ZZ[ii - 2] - ZZ[ii];
         hh[1] = ZZ[ii + 2] - ZZ[ii];
         if  (hh[0] * hh[1] < 0) continue;
         jk = 0;
         if  ( TMath::Abs(hh[1]) > TMath::Abs(hh[0]) ) jk = -1;
         YY[ii] = 0.5*ZZ[ii - 2*jk] + ZZ[ii - jk]/0.75 + ZZ[ii + 2*jk] /6.;
         YY[ii + jk] = 0.5*(ZZ[ii + 2*jk] - ZZ[ii - 2*jk]) + ZZ[ii - jk];
      }

// running means
      for  (ii = 1; ii < NN - 1; ii++) {
         RR[ii] = 0.25*YY[ii - 1] + 0.5*YY[ii] + 0.25*YY[ii + 1];
      }
      RR[0] = YY[0];
      RR[NN - 1] = YY[NN - 1];

// now do the same for residuals

      for  (ii = 0; ii < NN; ii++)  {
         YY[ii] = XX[ii] - RR[ii];
      }

//  do 353 i.e. running median 3, 5, and 3 in a single loop
      for  (kk = 0; kk < 3; kk++)  {
         ik = 0;
         if  (kk == 1)  ik = 1;
         nn1 = ik + 1;
         nn2 = NN - ik - 1;
       // do all elements beside the first and last point for median 3
       //  and first two and last 2 for median 5
         for  (ii = nn1; ii < nn2; ii++)  {
            for  (jj = 0; jj < 3; jj++) {
               hh[jj] = YY[ii + jj - 1];
            }
            ZZ[ii] = TH1::SmoothMedian(3 + 2*ik, hh);
         }

         if  (kk == 0)  {   // first median 3
// first point
            hh[0] = 3*YY[1] - 2*YY[2];
            hh[1] = YY[0];
            hh[2] = YY[2];
            ZZ[0] = TH1::SmoothMedian(3, hh);
// last point
            hh[0] = YY[NN - 2];
            hh[1] = YY[NN - 1];
            hh[2] = 3*YY[NN - 2] - 2*YY[NN - 3];
            ZZ[NN - 1] = TH1::SmoothMedian(3, hh);
         }
         if  (kk == 1)  {   //  median 5
//  first point remains the same
            ZZ[0] = YY[0];
            for  (ii = 0; ii < 3; ii++) {
               hh[ii] = YY[ii];
            }
            ZZ[1] = TH1::SmoothMedian(3, hh);
// last two points
            for  (ii = 0; ii < 3; ii++) {
               hh[ii] = YY[NN - 3 + ii];
            }
            ZZ[NN - 2] = TH1::SmoothMedian(3, hh);
            ZZ[NN - 1] = YY[NN - 1];
         }
      }

// quadratic interpolation for flat segments
      nn2 = nn2 - 1;
      for (ii = nn1 + 1; ii < nn2; ii++) {
         if  (ZZ[ii - 1] != ZZ[ii]) continue;
         if  (ZZ[ii] != ZZ[ii + 1]) continue;
         hh[0] = ZZ[ii - 2] - ZZ[ii];
         hh[1] = ZZ[ii + 2] - ZZ[ii];
         if  (hh[0] * hh[1] < 0) continue;
         jk = 0;
         if  ( TMath::Abs(hh[1]) > TMath::Abs(hh[0]) ) jk = -1;
         YY[ii] = 0.5*ZZ[ii - 2*jk] + ZZ[ii - jk]/0.75 + ZZ[ii + 2*jk]/6.;
         YY[ii + jk] = 0.5*(ZZ[ii + 2*jk] - ZZ[ii - 2*jk]) + ZZ[ii - jk];
      }

// running means
      for  (ii = 1; ii < NN - 1; ii++) {
         ZZ[ii] = 0.25*YY[ii - 1] + 0.5*YY[ii] + 0.25*YY[ii + 1];
      }
      ZZ[0] = YY[0];
      ZZ[NN - 1] = YY[NN - 1];

//  add smoothed XX and smoothed residuals

      for  (ii = 0; ii < NN; ii++) {
         if (XX[ii] < 0) XX[ii] = RR[ii] + ZZ[ii];
         else            XX[ii] = TMath::Abs(RR[ii] + ZZ[ii]);
      }
   }
   delete [] YY;
   delete [] ZZ;
   delete [] RR;
}

// ------------------------------------------------------------------------
Double_t  TH1::SmoothMedian(Int_t n, Double_t *a)
{
// return the median of a vector a in monotonic order with length n
// where median is a number which divides sequence of n numbers
// into 2 halves. When n is odd, the median is kth element k = (n + 1) / 2.
// when n is even the median is a mean of the elements k = n/2 and k = n/2 + 1.

  Int_t in, imin, imax;
  Double_t  xm;

  if  (n%2 == 0)  in = n / 2;
  else            in = n / 2 + 1;

  // find array element with maximum content
  imax = TMath::LocMax(n,a);
  xm = a[imax];

  while (in < n) {
     imin = TMath::LocMin(n,a);  // find array element with minimum content
     a[imin] = xm;
     in++;
  }
  imin = TMath::LocMin(n,a);
  return a[imin];
}


// ------------------------------------------------------------------------
void  TH1::Smooth(Int_t ntimes)
{
// Smooth bin contents of this histogram.
// bin contents are replaced by their smooth values.
// Errors (if any) are not modified.
// algorithm can only be applied to 1-d histograms

   if (fDimension != 1) {
      Error("Smooth","Smooth only supported for 1-d histograms");
      return;
   }
   Int_t nbins = fXaxis.GetNbins();
   Double_t *XX = new Double_t[nbins];
   Int_t i;
   for (i=0;i<nbins;i++) {
      XX[i] = GetBinContent(i+1);
   }

   TH1::SmoothArray(nbins,XX,ntimes);

   for (i=0;i<nbins;i++) {
      SetBinContent(i+1,XX[i]);
   }
   if (gPad) gPad->Modified();
}


//_______________________________________________________________________
void TH1::Streamer(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*Stream a class object*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*              =====================
   if (b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = b.ReadVersion(&R__s, &R__c);
      if (R__v > 2) {
         TH1::Class()->ReadBuffer(b, this, R__v, R__s, R__c);

         fXaxis.SetParent(this);
         fYaxis.SetParent(this);
         fZaxis.SetParent(this);
         if (fgAddDirectory && !gROOT->ReadingObject()) {
            fDirectory = gDirectory;
            if (!gDirectory->GetList()->FindObject(this)) gDirectory->Append(this);
         }
         return;
      }
      //process old versions before automatic schema evolution
      TNamed::Streamer(b);
      TAttLine::Streamer(b);
      TAttFill::Streamer(b);
      TAttMarker::Streamer(b);
      b >> fNcells;
      fXaxis.Streamer(b);
      fYaxis.Streamer(b);
      fZaxis.Streamer(b);
      fXaxis.SetParent(this);
      fYaxis.SetParent(this);
      fZaxis.SetParent(this);
      b >> fBarOffset;
      b >> fBarWidth;
      b >> fEntries;
      b >> fTsumw;
      b >> fTsumw2;
      b >> fTsumwx;
      b >> fTsumwx2;
      if (R__v < 2) {
         Float_t maximum, minimum, norm;
         Float_t *contour=0;
         b >> maximum; fMaximum = maximum;
         b >> minimum; fMinimum = minimum;
         b >> norm;    fNormFactor = norm;
         Int_t n = b.ReadArray(contour);
         fContour.Set(n);
         for (Int_t i=0;i<n;i++) fContour.fArray[i] = contour[i];
         delete [] contour;
      } else {
         b >> fMaximum;
         b >> fMinimum;
         b >> fNormFactor;
         fContour.Streamer(b);
      }
      fSumw2.Streamer(b);
      fOption.Streamer(b);
      fFunctions->Delete();
      fFunctions->Streamer(b);
      if (!gROOT->ReadingObject()) {
         fDirectory = gDirectory;
         if (!gDirectory->GetList()->FindObject(this)) gDirectory->Append(this);
      }
      b.CheckByteCount(R__s, R__c, TH1::IsA());

   } else {
      TH1::Class()->WriteBuffer(b,this);
   }
}

//______________________________________________________________________________
void TH1::Print(Option_t *option) const
{
//*-*-*-*-*-*-*Print some global quantities for this histogram*-*-*-*-*-*-*-*
//*-*          ===============================================
//
//  If option "range" is given, bin contents and errors are also printed
//                     for all bins in the current range (default 1-->nbins)
//  If option "all" is given, bin contents and errors are also printed
//                     for all bins including under and overflows.
//
   printf( "TH1.Print Name= %s, Entries= %d, Total sum= %g\n",GetName(),Int_t(fEntries),GetSumOfWeights());
   TString opt = option;
   opt.ToLower();
   Int_t all;
   if      (opt.Contains("all"))   all = 0;
   else if (opt.Contains("range")) all = 1;
   else                            return;

   Int_t bin, binx, biny, binz;
   Int_t firstx=0,lastx=0,firsty=0,lasty=0,firstz=0,lastz=0;
   if (all == 0) {
      lastx  = fXaxis.GetNbins()+1;
      if (fDimension > 1) lasty = fYaxis.GetNbins()+1;
      if (fDimension > 2) lastz = fZaxis.GetNbins()+1;
   } else {
      firstx = fXaxis.GetFirst(); lastx  = fXaxis.GetLast();
      if (fDimension > 1) {firsty = fYaxis.GetFirst(); lasty = fYaxis.GetLast();}
      if (fDimension > 2) {firstz = fZaxis.GetFirst(); lastz = fZaxis.GetLast();}
   }
   Stat_t w,e;
   Axis_t x,y,z;
   if (fDimension == 1) {
      for (binx=firstx;binx<=lastx;binx++) {
         x = fXaxis.GetBinCenter(binx);
         w = GetBinContent(binx);
         e = GetBinError(binx);
         if(fSumw2.fN) printf(" fSumw[%d]=%g, x=%g, error=%g\n",binx,w,x,e);
         else          printf(" fSumw[%d]=%g, x=%g\n",binx,w,x);
      }
   }
   if (fDimension == 2) {
      for (biny=firsty;biny<=lasty;biny++) {
         y = fYaxis.GetBinCenter(biny);
         for (binx=firstx;binx<=lastx;binx++) {
            bin = GetBin(binx,biny);
            x = fXaxis.GetBinCenter(binx);
            w = GetBinContent(bin);
            e = GetBinError(bin);
            if(fSumw2.fN) printf(" fSumw[%d][%d]=%g, x=%g, y=%g, error=%g\n",binx,biny,w,x,y,e);
            else          printf(" fSumw[%d][%d]=%g, x=%g, y=%g\n",binx,biny,w,x,y);
         }
      }
   }
   if (fDimension == 3) {
      for (binz=firstz;binz<=lastz;binz++) {
         z = fZaxis.GetBinCenter(binz);
         for (biny=firsty;biny<=lasty;biny++) {
            y = fYaxis.GetBinCenter(biny);
            for (binx=firstx;binx<=lastx;binx++) {
               bin = GetBin(binx,biny,binz);
               x = fXaxis.GetBinCenter(binx);
               w = GetBinContent(bin);
               e = GetBinError(bin);
               if(fSumw2.fN) printf(" fSumw[%d][%d][%d]=%g, x=%g, y=%g, z=%g, error=%g\n",binx,biny,binz,w,x,y,z,e);
               else          printf(" fSumw[%d][%d][%d]=%g, x=%g, y=%g, z=%g\n",binx,biny,binz,w,x,y,z);
            }
         }
      }
   }
}

//______________________________________________________________________________
void TH1::Reset(Option_t *option)
{
//*-*-*-*-*-*-*-*Reset this histogram: contents, errors, etc*-*-*-*-*-*-*-*
//*-*            ===========================================
//
// if option "ICE" is specified, resets only Integral, Contents and Errors

   TString opt = option;
   opt.ToUpper();
   fSumw2.Reset();
   if (fIntegral) {delete [] fIntegral; fIntegral = 0;}

   if (opt.Contains("ICE")) return;
   fTsumw       = 0;
   fTsumw2      = 0;
   fTsumwx      = 0;
   fTsumwx2     = 0;
   fEntries     = 0;

   fFunctions->Delete();
   fContour.Set(0);
}

//______________________________________________________________________________
void TH1::SavePrimitive(ofstream &out, Option_t *option)
{
    // Save primitive as a C++ statement(s) on output stream out

   //Note the following restrictions in the code generated:
   // - variable bin size not implemented
   // - Objects in list of functions not saved (fits)

   char quote = '"';
   out<<"   "<<endl;
   out<<"   "<<"TH1"<<" *";

   out<<GetName()<<" = new "<<ClassName()<<"("<<quote<<GetName()<<quote<<","<<quote<<GetTitle()<<quote
                 <<","<<GetXaxis()->GetNbins()
                 <<","<<GetXaxis()->GetXmin()
                 <<","<<GetXaxis()->GetXmax();
   if (fDimension > 1) {
              out<<","<<GetYaxis()->GetNbins()
                 <<","<<GetYaxis()->GetXmin()
                 <<","<<GetYaxis()->GetXmax();
   }
   if (fDimension > 2) {
              out<<","<<GetZaxis()->GetNbins()
                 <<","<<GetZaxis()->GetXmin()
                 <<","<<GetZaxis()->GetXmax();
   }
              out<<");"<<endl;
   if (TMath::Abs(GetBarOffset()) > 1e-5) {
      out<<"   "<<GetName()<<"->SetBarOffset("<<GetBarOffset()<<");"<<endl;
   }
   if (TMath::Abs(GetBarWidth()-1) > 1e-5) {
      out<<"   "<<GetName()<<"->SetBarWidth("<<GetBarWidth()<<");"<<endl;
   }
    if (fMinimum != -1111) {
      out<<"   "<<GetName()<<"->SetMinimum("<<fMinimum<<");"<<endl;
   }
  if (fMaximum != -1111) {
      out<<"   "<<GetName()<<"->SetMaximum("<<fMaximum<<");"<<endl;
   }
   if (fNormFactor != 0) {
      out<<"   "<<GetName()<<"->SetNormFactor("<<fNormFactor<<");"<<endl;
   }
   if (fEntries != 0) {
      out<<"   "<<GetName()<<"->SetEntries("<<fEntries<<");"<<endl;
   }
   if (fDirectory == 0) {
      out<<"   "<<GetName()<<"->SetDirectory(0);"<<endl;
   }
   if (TestBit(kNoStats)) {
      out<<"   "<<GetName()<<"->SetStats(0);"<<endl;
   }
   if (fOption.Length() != 0) {
      out<<"   "<<GetName()<<"->SetOption("<<quote<<fOption.Data()<<quote<<");"<<endl;
   }
   Int_t bin;
   for (bin=0;bin<fNcells;bin++) {
      Double_t bc = GetBinContent(bin);
      if (bc) {
         out<<"   "<<GetName()<<"->SetBinContent("<<bin<<","<<bc<<");"<<endl;
      }
   }
   if (fSumw2.fN) {
      for (bin=0;bin<fNcells;bin++) {
         Double_t be = GetBinError(bin);
         if (be) {
            out<<"   "<<GetName()<<"->SetBinError("<<bin<<","<<be<<");"<<endl;
         }
      }
   }
   Int_t ncontours = GetContour();
   if (ncontours > 0) {
      out<<"   "<<GetName()<<"->SetContour("<<ncontours<<");"<<endl;
      for (bin=0;bin<ncontours;bin++) {
         out<<"   "<<GetName()<<"->SetContourLevel("<<bin<<","<<GetContourLevel(bin)<<");"<<endl;
      }
   }

   SaveFillAttributes(out,GetName(),0,1001);
   SaveLineAttributes(out,GetName(),1,1,1);
   SaveMarkerAttributes(out,GetName(),1,1,1);
   fXaxis.SaveAttributes(out,GetName(),"->GetXaxis()");
   fYaxis.SaveAttributes(out,GetName(),"->GetYaxis()");
   fZaxis.SaveAttributes(out,GetName(),"->GetZaxis()");
   TString opt = option;
   opt.ToLower();
   if (!opt.Contains("nodraw")) {
      out<<"   "<<GetName()<<"->Draw("
      <<quote<<option<<quote<<");"<<endl;
   }
}

//______________________________________________________________________________
void TH1::UseCurrentStyle()
{
//*-*-*-*-*-*Replace current attributes by current style*-*-*-*-*
//*-*        ===========================================

   fXaxis.ResetAttAxis("X");
   fYaxis.ResetAttAxis("Y");
   SetBarOffset(gStyle->GetBarOffset());
   SetBarWidth(gStyle->GetBarWidth());
   SetFillColor(gStyle->GetHistFillColor());
   SetFillStyle(gStyle->GetHistFillStyle());
   SetLineColor(gStyle->GetHistLineColor());
   SetLineStyle(gStyle->GetHistLineStyle());
   SetLineWidth(gStyle->GetHistLineWidth());
   SetMarkerColor(gStyle->GetMarkerColor());
   SetMarkerStyle(gStyle->GetMarkerStyle());
   SetMarkerSize(gStyle->GetMarkerSize());
}

//______________________________________________________________________________
Stat_t TH1::GetMean(Int_t axis) const
{
//*-*-*-*-*-*-*-*Return mean value of this histogram along the X axis*-*-*-*-*
//*-*            ====================================================

  if (axis <1 || axis > 3) return 0;
  Stat_t stats[10];
  for (Int_t i=4;i<10;i++) stats[i] = 0;
  GetStats(stats);
  if (stats[0] == 0) return 0;
  Int_t ax[3] = {2,4,7};
  return stats[ax[axis-1]]/stats[0];
}

//______________________________________________________________________________
Stat_t TH1::GetRMS(Int_t axis) const
{
//*-*-*-*-*-*-*-*Return the Root Mean Square value of this histogram*-*-*-*-*
//*-*            ===================================================

  if (axis <1 || axis > 3) return 0;
  Stat_t x, rms2, stats[10];
  for (Int_t i=4;i<10;i++) stats[i] = 0;
  GetStats(stats);
  if (stats[0] == 0) return 0;
  Int_t ax[3] = {2,4,7};
  Int_t axm = ax[axis-1];
  x    = stats[axm]/stats[0];
  rms2 = TMath::Abs(stats[axm+1]/stats[0] -x*x);
  return TMath::Sqrt(rms2);
}

//______________________________________________________________________________
void TH1::GetStats(Stat_t *stats) const
{
   // fill the array stats from the contents of this histogram
   // The array stats must be correctly dimensionned in the calling program.
   // stats[0] = sumw
   // stats[1] = sumw2
   // stats[2] = sumwx
   // stats[3] = sumwx2
   //
   // If no axis-subrange is specified (via TAxis::SetRange), the array stats
   // is simply a copy of the statistics quantities computed at filling time.
   // If a sub-range is specified, the function recomputes these quantities
   // from the bin contents in the current axis range.

   // Loop on bins (possibly including underflows/overflows)
   Int_t bin, binx;
   Stat_t w;
   Axis_t x;
   if (fTsumw == 0 || fXaxis.TestBit(TAxis::kAxisRange)) {
      for (bin=0;bin<4;bin++) stats[bin] = 0;
      for (binx=fXaxis.GetFirst();binx<=fXaxis.GetLast();binx++) {
         x = fXaxis.GetBinCenter(binx);
         w = TMath::Abs(GetBinContent(binx));
         stats[0] += w;
         stats[1] += w*w;
         stats[2] += w*x;
         stats[3] += w*x*x;
      }
   } else {
      stats[0] = fTsumw;
      stats[1] = fTsumw2;
      stats[2] = fTsumwx;
      stats[3] = fTsumwx2;
   }
}

//______________________________________________________________________________
void TH1::PutStats(Stat_t *stats)
{
   // Replace current statistics with the values in array stats

   fTsumw   = stats[0];
   fTsumw2  = stats[1];
   fTsumwx  = stats[2];
   fTsumwx2 = stats[3];
}

//______________________________________________________________________________
Stat_t TH1::GetSumOfWeights() const
{
//*-*-*-*-*-*-*-*Return the sum of weights excluding under/overflows*-*-*-*-*
//*-*            ===================================================
  Int_t bin,binx,biny,binz;
  Stat_t sum =0;
  for(binz=1; binz<=fZaxis.GetNbins(); binz++) {
     for(biny=1; biny<=fYaxis.GetNbins(); biny++) {
        for(binx=1; binx<=fXaxis.GetNbins(); binx++) {
           bin = GetBin(binx,biny,binz);
           sum += GetBinContent(bin);
        }
     }
  }
  return sum;
}


//______________________________________________________________________________
Stat_t TH1::Integral(Option_t *option)
{
//Return integral of bin contents. Only bins in the bins range are considered.
// By default the integral is computed as the sum of bin contents in the range.
// if option "width" is specified, the integral is the sum of
// the bin contents multiplied by the bin width in x.

   return Integral(fXaxis.GetFirst(),fXaxis.GetLast(),option);
}

//______________________________________________________________________________
Stat_t TH1::Integral(Int_t binx1, Int_t binx2, Option_t *option)
{
//Return integral of bin contents between binx1 and binx2 for a 1-D histogram
// By default the integral is computed as the sum of bin contents in the range.
// if option "width" is specified, the integral is the sum of
// the bin contents multiplied by the bin width in x.

   if (binx1 < 0) binx1 = 0;
   Stat_t integral = 0;

//*-*- Loop on bins in specified range
   TString opt = option;
   opt.ToLower();
   Bool_t width = kFALSE;
   if (opt.Contains("width")) width = kTRUE;
   Int_t binx;
   for (binx=binx1;binx<=binx2;binx++) {
      if (width) integral += GetBinContent(binx)*fXaxis.GetBinWidth(binx);
      else       integral += GetBinContent(binx);
   }
   return integral;
}

//______________________________________________________________________________
Double_t TH1::KolmogorovTest(TH1 *h2, Option_t *option)
{
//  Statistical test of compatibility in shape between
//  THIS histogram and h2, using Kolmogorov test.
//     Default: Ignore under- and overflow bins in comparison
//
//     option is a character string to specify options
//         "U" include Underflows in test  (also for 2-dim)
//         "O" include Overflows     (also valid for 2-dim)
//         "N" include comparison of normalizations
//         "D" Put out a line of "Debug" printout
//         "M" Return the Maximum Kolmogorov distance instead of prob
//
//   The returned function value is the probability of test
//       (much less than one means NOT compatible)
//
//  Code adapted by Rene Brun from original HBOOK routine HDIFF

   TString opt = option;
   opt.ToUpper();

   Double_t prb = 0;
   TH1 *h1 = this;
   if (h2 == 0) return 0;
   TAxis *axis1 = h1->GetXaxis();
   TAxis *axis2 = h2->GetXaxis();
   Int_t ncx1   = axis1->GetNbins();
   Int_t ncx2   = axis2->GetNbins();

     // Check consistency of dimensions
   if (h1->GetDimension() != 1 || h2->GetDimension() != 1) {
      Error("KolmogorovTest","Histograms must be 1-D\n");
      return 0;
   }

     // Check consistency in number of channels
   if (ncx1 != ncx2) {
      Error("KolmogorovTest","Number of channels is different, %d and %d\n",ncx1,ncx2);
      return 0;
   }

     // Check consistency in channel edges
   Double_t difprec = 1e-5;
   Double_t diff1 = TMath::Abs(axis1->GetXmin() - axis2->GetXmin());
   Double_t diff2 = TMath::Abs(axis1->GetXmax() - axis2->GetXmax());
   if (diff1 > difprec || diff2 > difprec) {
      Error("KolmogorovTest","histograms with different binning");
      return 0;
   }

   Bool_t afunc1 = kFALSE;
   Bool_t afunc2 = kFALSE;
   Double_t sum1 = 0, sum2 = 0;
   Int_t bin;
   for (bin=1;bin<=ncx1;bin++) {
      sum1 += h1->GetBinContent(bin);
      sum2 += h2->GetBinContent(bin);
   }
   if (sum1 == 0) {
      Error("KolmogorovTest","Histogram1 %s integral is zero\n",h1->GetName());
      return 0;
   }
   if (sum2 == 0) {
      Error("KolmogorovTest","Histogram2 %s integral is zero\n",h2->GetName());
      return 0;
   }
   Double_t tsum1 = sum1;
   Double_t tsum2 = sum2;
   if (opt.Contains("U")) {
      tsum1 += h1->GetBinContent(0);
      tsum2 += h2->GetBinContent(0);
   }
   if (opt.Contains("O")) {
      tsum1 += h1->GetBinContent(ncx1+1);
      tsum2 += h2->GetBinContent(ncx1+1);
   }

     // Check if histograms are weighted.
     // If number of entries = number of channels, probably histograms were
     // not filled via Fill(), but via SetBinContent()
   Double_t ne1 = h1->GetEntries();
   Double_t ne2 = h2->GetEntries();
     // look at first histogram
   Double_t difsum1 = (ne1-tsum1)/tsum1;
   Double_t esum1 = sum1;
   if (difsum1 > difprec && Int_t(ne1) != ncx1) {
      if (opt.Contains("U") || opt.Contains("O")) {
         Warning("KolmogorovTest","U/O option with weighted events for hist:%s\n",h1->GetName());
      }
      if (h1->GetSumw2N() == 0) {
         Warning("KolmogorovTest","Weighted events and no Sumw2, hist:%s\n",h1->GetName());
      } else {
         esum1 = h1->GetSumOfWeights();
      }
   }
     // look at second histogram
   Double_t difsum2 = (ne2-tsum2)/tsum2;
   Double_t esum2   = sum2;
   if (difsum2 > difprec && Int_t(ne2) != ncx1) {
      if (opt.Contains("U") || opt.Contains("O")) {
         Warning("KolmogorovTest","U/O option with weighted events for hist:%s\n",h2->GetName());
      }
      if (h2->GetSumw2N() == 0) {
         Warning("KolmogorovTest","Weighted events and no Sumw2, hist:%s\n",h2->GetName());
      } else {
         esum2 = h2->GetSumOfWeights();
      }
   }

   Double_t s1 = 1/sum1;
   Double_t s2 = 1/sum2;

      // Find largest difference for Kolmogorov Test
   Double_t dfmax =0, rsum1 = 0, rsum2 = 0;

   Int_t first = 1;
   Int_t last  = ncx1;
   if (opt.Contains("U")) first = 0;
   if (opt.Contains("O")) last  = ncx1+1;
   for (bin=first;bin<=last;bin++) {
      rsum1 += s1*h1->GetBinContent(bin);
      rsum2 += s2*h2->GetBinContent(bin);
      dfmax = TMath::Max(dfmax,TMath::Abs(rsum1-rsum2));
   }

      // Get Kolmogorov probability
   Double_t z, prb1=0, prb2=0;
   if (afunc1)      z = dfmax*TMath::Sqrt(esum2);
   else if (afunc2) z = dfmax*TMath::Sqrt(esum1);
   else             z = dfmax*TMath::Sqrt(esum1*esum2/(esum1+esum2));

   prb = TMath::KolmogorovProb(z);

   if (opt.Contains("N")) {
      // Combine probabilities for shape and normalization,
      prb1 = prb;
      Double_t resum1 = esum1;  if (afunc1) resum1 = 0;
      Double_t resum2 = esum2;  if (afunc2) resum2 = 0;
      Double_t d12    = esum1-esum2;
      Double_t chi2   = d12*d12/(resum1+resum2);
      prb2 = TMath::Prob(chi2,1);
      // see Eadie et al., section 11.6.2
      if (prb > 0 && prb2 > 0) prb *= prb2*(1-TMath::Log(prb*prb2));
      else                     prb = 0;
   }
      // debug printout
   if (opt.Contains("D")) {
      printf(" Kolmo Prob  h1 = %s, sum1=%g\n",h1->GetName(),sum1);
      printf(" Kolmo Prob  h2 = %s, sum2=%g\n",h2->GetName(),sum2);
      printf(" Kolmo Probabil = %g, Max Dist = %g\n",prb,dfmax);
      if (opt.Contains("N"))
      printf(" Kolmo Probabil = %f for shape alone, =%f for normalisation alone\n",prb1,prb2);
   }
      // This numerical error condition should never occur:
   if (TMath::Abs(rsum1-1) > 0.002) Warning("KolmogorovTest","Numerical problems with h1=%s\n",h1->GetName());
   if (TMath::Abs(rsum2-1) > 0.002) Warning("KolmogorovTest","Numerical problems with h2=%s\n",h2->GetName());

   if(opt.Contains("M"))  return dfmax;
   else                   return prb;
}

//______________________________________________________________________________
void TH1::SetContent(const Stat_t *content)
{
//*-*-*-*-*-*-*-*Replace bin contents by the contents of array content*-*-*-*
//*-*            =====================================================
   Int_t bin;
   Stat_t bincontent;
   for (bin=0; bin<fNcells; bin++) {
      bincontent = *(content + bin);
      SetBinContent(bin, bincontent);
   }
}

//______________________________________________________________________________
Int_t TH1::GetContour(Double_t *levels)
{
//  Return contour values into array levels if pointer levels is non zero
//
//  The function returns the number of contour levels.
//  see GetContourLevel to return one contour only
//

  Int_t nlevels = fContour.fN;
  if (levels) {
     if (nlevels == 0) {
        nlevels = 20;
        SetContour(nlevels);
     } else {
        if (TestBit(kUserContour) == 0) SetContour(nlevels);
     }
     for (Int_t level=0; level<nlevels; level++) levels[level] = fContour.fArray[level];
  }
  return nlevels;
}

//______________________________________________________________________________
Double_t TH1::GetContourLevel(Int_t level) const
{
// Return value of contour number level
// see GetContour to return the array of all contour levels

  if (level <0 || level >= fContour.fN) return 0;
  Double_t zlevel = fContour.fArray[level];
  return zlevel;
}


//______________________________________________________________________________
void TH1::SetContour(Int_t  nlevels, const Double_t *levels)
{
//*-*-*-*-*-*-*-*Set the number and values of contour levels*-*-*-*-*-*-*-*-*
//*-*            ===========================================
//
//  By default the number of contour levels is set to 20.
//
//  if argument levels = 0 or issing, equidistant contours are computed
//

  Int_t level;
  ResetBit(kUserContour);
  if (nlevels <=0 ) {
     fContour.Set(0);
     return;
  }
  fContour.Set(nlevels);

//*-*-  Contour levels are specified
  if (levels) {
     SetBit(kUserContour);
     for (level=0; level<nlevels; level++) fContour.fArray[level] = levels[level];
  } else {
//*-*- contour levels are computed automatically as equidistant contours
     Double_t zmin = GetMinimum();
     Double_t zmax = GetMaximum();
     Double_t dz   = (zmax-zmin)/Double_t(nlevels);
     if (gPad && gPad->GetLogz()) {
        if (zmax <= 0) return;
        if (zmin <= 0) zmin = 0.001*zmax;
        zmin = TMath::Log10(zmin);
        zmax = TMath::Log10(zmax);
        dz   = (zmax-zmin)/Double_t(nlevels);
     }
     for (level=0; level<nlevels; level++) {
        fContour.fArray[level] = zmin + dz*Double_t(level);
     }
  }
}

//______________________________________________________________________________
void TH1::SetContourLevel(Int_t level, Double_t value)
{
//*-*-*-*-*-*-*-*-*-*-*Set value for one contour level*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===============================
  if (level <0 || level >= fContour.fN) return;
  SetBit(kUserContour);
  fContour.fArray[level] = value;
}

//______________________________________________________________________________
Double_t TH1::GetMaximum() const
{
//*-*-*-*-*-*-*-*-*-*-*Return maximum value of bins in the range*-*-*-*-*-*
//*-*                  =========================================
  if (fMaximum != -1111) return fMaximum;
  Int_t bin, binx, biny, binz;
  Int_t xfirst  = fXaxis.GetFirst();
  Int_t xlast   = fXaxis.GetLast();
  Int_t yfirst  = fYaxis.GetFirst();
  Int_t ylast   = fYaxis.GetLast();
  Int_t zfirst  = fZaxis.GetFirst();
  Int_t zlast   = fZaxis.GetLast();
  Double_t maximum = -FLT_MAX, value;
  for (binz=zfirst;binz<=zlast;binz++) {
     for (biny=yfirst;biny<=ylast;biny++) {
        for (binx=xfirst;binx<=xlast;binx++) {
           bin = GetBin(binx,biny,binz);
           value = GetBinContent(bin);
           if (value > maximum) maximum = value;
        }
     }
  }
  return maximum;
}

//______________________________________________________________________________
Int_t TH1::GetMaximumBin() const
{
//*-*-*-*-*-*-*Return location of bin with maximum value in the range*-*
//*-*          ======================================================
  Int_t locmax, locmay, locmaz;
  return GetMaximumBin(locmax, locmay, locmaz);
}

//______________________________________________________________________________
Int_t TH1::GetMaximumBin(Int_t &locmax, Int_t &locmay, Int_t &locmaz) const
{
//*-*-*-*-*-*-*Return location of bin with maximum value in the range*-*
//*-*          ======================================================
  Int_t bin, binx, biny, binz;
  Int_t locm;
  Int_t xfirst  = fXaxis.GetFirst();
  Int_t xlast   = fXaxis.GetLast();
  Int_t yfirst  = fYaxis.GetFirst();
  Int_t ylast   = fYaxis.GetLast();
  Int_t zfirst  = fZaxis.GetFirst();
  Int_t zlast   = fZaxis.GetLast();
  Double_t maximum = -FLT_MAX, value;
  locm = locmax = locmay = locmaz = 0;
  for (binz=zfirst;binz<=zlast;binz++) {
     for (biny=yfirst;biny<=ylast;biny++) {
        for (binx=xfirst;binx<=xlast;binx++) {
           bin = GetBin(binx,biny,binz);
           value = GetBinContent(bin);
           if (value > maximum) {
              maximum = value;
              locm    = bin;
              locmax  = binx;
              locmay  = biny;
              locmaz  = binz;
           }
        }
     }
  }
  return locm;
}

//______________________________________________________________________________
Double_t TH1::GetMinimum() const
{
//*-*-*-*-*-*-*-*-*-*-*Return minimum value of bins in the range-*-*-*-*-*
//*-*                  =========================================
  if (fMinimum != -1111) return fMinimum;
  Int_t bin, binx, biny, binz;
  Int_t xfirst  = fXaxis.GetFirst();
  Int_t xlast   = fXaxis.GetLast();
  Int_t yfirst  = fYaxis.GetFirst();
  Int_t ylast   = fYaxis.GetLast();
  Int_t zfirst  = fZaxis.GetFirst();
  Int_t zlast   = fZaxis.GetLast();
  Double_t minimum=FLT_MAX, value;
  for (binz=zfirst;binz<=zlast;binz++) {
     for (biny=yfirst;biny<=ylast;biny++) {
        for (binx=xfirst;binx<=xlast;binx++) {
           bin = GetBin(binx,biny,binz);
           value = GetBinContent(bin);
           if (value < minimum) minimum = value;
        }
     }
  }
  return minimum;
}

//______________________________________________________________________________
Int_t TH1::GetMinimumBin() const
{
//*-*-*-*-*-*-*Return location of bin with minimum value in the range*-*
//*-*          ======================================================
  Int_t locmix, locmiy, locmiz;
  return GetMinimumBin(locmix, locmiy, locmiz);
}

//______________________________________________________________________________
Int_t TH1::GetMinimumBin(Int_t &locmix, Int_t &locmiy, Int_t &locmiz) const
{
//*-*-*-*-*-*-*Return location of bin with minimum value in the range*-*
//*-*          ======================================================
  Int_t bin, binx, biny, binz;
  Int_t locm;
  Int_t xfirst  = fXaxis.GetFirst();
  Int_t xlast   = fXaxis.GetLast();
  Int_t yfirst  = fYaxis.GetFirst();
  Int_t ylast   = fYaxis.GetLast();
  Int_t zfirst  = fZaxis.GetFirst();
  Int_t zlast   = fZaxis.GetLast();
  Double_t minimum = FLT_MAX, value;
  locm = locmix = locmiy = locmiz = 0;
  for (binz=zfirst;binz<=zlast;binz++) {
     for (biny=yfirst;biny<=ylast;biny++) {
        for (binx=xfirst;binx<=xlast;binx++) {
           bin = GetBin(binx,biny,binz);
           value = GetBinContent(bin);
           if (value < minimum) {
              minimum = value;
              locm    = bin;
              locmix  = binx;
              locmiy  = biny;
              locmiz  = binz;
           }
        }
     }
  }
  return locm;
}

//______________________________________________________________________________
void TH1::SetBins(Int_t nx, Axis_t xmin, Axis_t xmax)
{
//*-*-*-*-*-*-*-*-*Redefine  x axis parameters*-*-*-*-*-*-*-*-*-*-*-*
//*-*              ===========================
// The X axis parameters are modified.
// The bins content array is resized
// if errors (Sumw2) the errors array is resized
// The previous bin contents are lost
// To change only the axis limits, see TAxis::SetRange

   if (GetDimension() != 1) {
      Error("SetBins","Operation only valid for 1-d histograms");
      return;
   }
   fXaxis.SetRange(0,0);
   fXaxis.Set(nx,xmin,xmax);
   fNcells = nx+2;
   SetBinsLength(fNcells);
   if (fSumw2.fN) {
      fSumw2.Set(fNcells);
   }
}

//______________________________________________________________________________
void TH1::SetBins(Int_t nx, Axis_t xmin, Axis_t xmax, Int_t ny, Axis_t ymin, Axis_t ymax)
{
//*-*-*-*-*-*-*-*-*Redefine  x and y axis parameters*-*-*-*-*-*-*-*-*-*-*-*
//*-*              =================================
// The X and Y axis parameters are modified.
// The bins content array is resized
// if errors (Sumw2) the errors array is resized
// The previous bin contents are lost
// To change only the axis limits, see TAxis::SetRange

   if (GetDimension() != 2) {
      Error("SetBins","Operation only valid for 2-d histograms");
      return;
   }
   fXaxis.SetRange(0,0);
   fYaxis.SetRange(0,0);
   fXaxis.Set(nx,xmin,xmax);
   fYaxis.Set(ny,ymin,ymax);
   fNcells = (nx+2)*(ny+2);
   SetBinsLength(fNcells);
   if (fSumw2.fN) {
      fSumw2.Set(fNcells);
   }
}

//______________________________________________________________________________
void TH1::SetBins(Int_t nx, Axis_t xmin, Axis_t xmax, Int_t ny, Axis_t ymin, Axis_t ymax, Int_t nz, Axis_t zmin, Axis_t zmax)
{
//*-*-*-*-*-*-*-*-*Redefine  x, y and z axis parameters*-*-*-*-*-*-*-*-*-*-*-*
//*-*              ====================================
// The X, Y and Z axis parameters are modified.
// The bins content array is resized
// if errors (Sumw2) the errors array is resized
// The previous bin contents are lost
// To change only the axis limits, see TAxis::SetRange

   if (GetDimension() != 3) {
      Error("SetBins","Operation only valid for 3-d histograms");
      return;
   }
   fXaxis.SetRange(0,0);
   fYaxis.SetRange(0,0);
   fZaxis.SetRange(0,0);
   fXaxis.Set(nx,xmin,xmax);
   fYaxis.Set(ny,ymin,ymax);
   fZaxis.Set(nz,zmin,zmax);
   fNcells = (nx+2)*(ny+2)*(nz+2);
   SetBinsLength(fNcells);
   if (fSumw2.fN) {
      fSumw2.Set(fNcells);
   }
}

//______________________________________________________________________________
void TH1::SetMaximum(Double_t maximum)
{
//*-*-*-*-*-*-*-*-*Set the maximum value for the Y axis*-*-*-*-*-*-*-*-*-*-*-*
//*-*              ====================================
// By default the maximum value is automatically set to the maximum
// bin content plus a margin of 10 per cent.
// Use TH1::GetMaximum to find the maximum value of an histogram
// Use TH1::GetMaximumBin to find the bin with the maximum value of an histogram
//
   fMaximum = maximum;
}


//______________________________________________________________________________
void TH1::SetMinimum(Double_t minimum)
{
//*-*-*-*-*-*-*-*-*Set the minimum value for the Y axis*-*-*-*-*-*-*-*-*-*-*-*
//*-*              ====================================
// By default the minimum value is automatically set to zero if all bin contents
// are positive or the minimum - 10 per cent otherwise.
// Use TH1::GetMinimum to find the minimum value of an histogram
// Use TH1::GetMinimumBin to find the bin with the minimum value of an histogram
//
   fMinimum = minimum;
}

//______________________________________________________________________________
void TH1::SetDirectory(TDirectory *dir)
{
   // By default when an histogram is created, it is added to the list
   // of histogram objects in the current directory in memory.
   // Remove reference to this histogram from current directory and add
   // reference to new directory dir. dir can be 0 in which case the
   // histogram does not belong to any directory.

   if (fDirectory == dir) return;
   if (fDirectory) fDirectory->GetList()->Remove(this);
   fDirectory = dir;
   if (fDirectory) fDirectory->GetList()->Add(this);
}


//______________________________________________________________________________
void TH1::SetError(const Stat_t *error)
{
//*-*-*-*-*-*-*-*-*Replace bin errors by values in array error*-*-*-*-*-*-*-*-*
//*-*              ===========================================
  Int_t bin;
  Stat_t binerror;
  for (bin=0; bin<fNcells; bin++) {
     binerror = error[bin];
     SetBinError(bin, binerror);
  }
}

//______________________________________________________________________________
void TH1::SetName(const char *name)
{
// Change the name of this histogram
//

//  Histograms are named objects in a THashList.
//  We must update the hashlist if we change the name
   if (fDirectory) fDirectory->GetList()->Remove(this);
   fName = name;
   if (fDirectory) fDirectory->GetList()->Add(this);
}

//______________________________________________________________________________
void TH1::SetNameTitle(const char *name, const char *title)
{
// Change the name and title of this histogram
//

//  Histograms are named objects in a THashList.
//  We must update the hashlist if we change the name
   if (fDirectory) fDirectory->GetList()->Remove(this);
   fName  = name;
   fTitle = title;
   if (fDirectory) fDirectory->GetList()->Add(this);
}

//______________________________________________________________________________
void TH1::SetStats(Bool_t stats)
{
//*-*-*-*-*-*-*-*-*Set statistics option on/off
//*-*              ============================
//  By default, the statistics box is drawn.
//  The paint options can be selected via gStyle->SetOptStats.
//  This function sets/resets the kNoStats bin in the histogram object.
//  It has priority over the Style option.

   ResetBit(kNoStats);
   if (!stats) SetBit(kNoStats);
}

//______________________________________________________________________________
void TH1::Sumw2()
{
//*-*-*-*-*Create structure to store sum of squares of weights*-*-*-*-*-*-*-*
//*-*      ===================================================
//*-*
//*-*  if histogram is already filled, the sum of squares of weights
//*-*  is filled with the existing bin contents
//*-*
//*-*  The error per bin will be computed as sqrt(sum of squares of weight)
//*-*  for each bin.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

  if (fSumw2.fN) {
     Warning("Sumw2","Sum of squares of weights structure already created");
     return;
  }

  fSumw2.Set(fNcells);

  for (Int_t bin=0; bin<fNcells; bin++) {
     fSumw2.fArray[bin] = GetBinContent(bin);
  }
}

//______________________________________________________________________________
TF1 *TH1::GetFunction(const char *name) const
{
//*-*-*-*-*Return pointer to function with name*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*      ===================================
//
// Functions such as TH1::Fit store the fitted function in the list of
// functions of this histogram.

   return (TF1*)fFunctions->FindObject(name);
}

//______________________________________________________________________________
Stat_t TH1::GetBinError(Int_t bin) const
{
//*-*-*-*-*-*-*Return value of error associated to bin number bin*-*-*-*-*
//*-*          ==================================================
//*-*
//*-* if the sum of squares of weights has been defined (via Sumw2),
//*-* this function returns the sqrt(sum of w2).
//*-* otherwise it returns the sqrt(contents) for this bin.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

  if (bin < 0) bin = 0;
  if (bin >= fNcells) bin = fNcells-1;
  if (fSumw2.fN) return TMath::Sqrt(fSumw2.fArray[bin]);
  Stat_t error2 = TMath::Abs(GetBinContent(bin));
  return TMath::Sqrt(error2);
}

//______________________________________________________________________________
Stat_t TH1::GetBinError(Int_t binx, Int_t biny) const
{
//*-*-*-*-*-*-*Return error of bin number binx, biny
//*-*          =====================================
// NB: Function to be called for 2-d histograms only

   Int_t bin = GetBin(binx,biny);
   return GetBinError(bin);
}

//______________________________________________________________________________
Stat_t TH1::GetBinError(Int_t binx, Int_t biny, Int_t binz) const
{
//*-*-*-*-*-*-*Return error of bin number binx,biny,binz
//*-*          =========================================
// NB: Function to be called for 3-d histograms only

   Int_t bin = GetBin(binx,biny,binz);
   return GetBinError(bin);
}

//______________________________________________________________________________
Stat_t TH1::GetCellContent(Int_t binx, Int_t biny) const
{
//*-*-*-*-*-*-*Return content of bin number binx, biny
//*-*          =====================================
// NB: Function to be called for 2-d histograms only

   Int_t bin = GetBin(binx,biny);
   return GetBinContent(bin);
}

//______________________________________________________________________________
Stat_t TH1::GetCellError(Int_t binx, Int_t biny) const
{
//*-*-*-*-*-*-*Return error of bin number binx, biny
//*-*          =====================================
// NB: Function to be called for 2-d histograms only

   Int_t bin = GetBin(binx,biny);
   return GetBinError(bin);
}

//______________________________________________________________________________
void TH1::SetBinError(Int_t bin, Stat_t error)
{
  if (!fSumw2.fN) Sumw2();
  if (bin <0 || bin>= fSumw2.fN) return;
  fSumw2.fArray[bin] = error*error;
}

//______________________________________________________________________________
void TH1::SetBinContent(Int_t binx, Int_t biny, Stat_t content)
{
  if (binx <0 || binx>fXaxis.GetNbins()+1) return;
  if (biny <0 || biny>fYaxis.GetNbins()+1) return;
  SetBinContent(biny*(fXaxis.GetNbins()+2) + binx,content);
}

//______________________________________________________________________________
void TH1::SetBinContent(Int_t binx, Int_t biny, Int_t binz, Stat_t content)
{
  if (binx <0 || binx>fXaxis.GetNbins()+1) return;
  if (biny <0 || biny>fYaxis.GetNbins()+1) return;
  if (binz <0 || binz>fZaxis.GetNbins()+1) return;
  Int_t bin = GetBin(binx,biny,binz);
  SetBinContent(bin,content);
}

//______________________________________________________________________________
void TH1::SetCellContent(Int_t binx, Int_t biny, Stat_t content)
{
  if (binx <0 || binx>fXaxis.GetNbins()+1) return;
  if (biny <0 || biny>fYaxis.GetNbins()+1) return;
  SetBinContent(biny*(fXaxis.GetNbins()+2) + binx,content);
}

//______________________________________________________________________________
void TH1::SetBinError(Int_t binx, Int_t biny, Stat_t error)
{
  if (binx <0 || binx>fXaxis.GetNbins()+1) return;
  if (biny <0 || biny>fYaxis.GetNbins()+1) return;
  SetBinError(biny*(fXaxis.GetNbins()+2) + binx,error);
}

//______________________________________________________________________________
void TH1::SetBinError(Int_t binx, Int_t biny, Int_t binz, Stat_t error)
{
  if (binx <0 || binx>fXaxis.GetNbins()+1) return;
  if (biny <0 || biny>fYaxis.GetNbins()+1) return;
  if (binz <0 || binz>fZaxis.GetNbins()+1) return;
  Int_t bin = GetBin(binx,biny,binz);
  SetBinError(bin,error);
}

//______________________________________________________________________________
void TH1::SetCellError(Int_t binx, Int_t biny, Stat_t error)
{
  if (binx <0 || binx>fXaxis.GetNbins()+1) return;
  if (biny <0 || biny>fYaxis.GetNbins()+1) return;
  if (!fSumw2.fN) Sumw2();
  Int_t bin = biny*(fXaxis.GetNbins()+2) + binx;
  fSumw2.fArray[bin] = error*error;
}

//______________________________________________________________________________
void TH1::SetBinContent(Int_t, Stat_t)
{
   AbstractMethod("SetBinContent");
}

ClassImp(TH1C)

//______________________________________________________________________________
//                     TH1C methods
//______________________________________________________________________________
TH1C::TH1C(): TH1(), TArrayC()
{
   fDimension = 1;
}

//______________________________________________________________________________
TH1C::TH1C(const char *name,const char *title,Int_t nbins,Axis_t xlow,Axis_t xup)
     : TH1(name,title,nbins,xlow,xup), TArrayC(nbins+2)
{
//
//    Create a 1-Dim histogram with fix bins of type char (one byte per channel)
//    ==========================================================================
//                    (see TH1::TH1 for explanation of parameters)
//
   fDimension = 1;
}

//______________________________________________________________________________
TH1C::TH1C(const char *name,const char *title,Int_t nbins,const Float_t *xbins)
     : TH1(name,title,nbins,xbins), TArrayC(nbins+2)
{
//
//    Create a 1-Dim histogram with variable bins of type char (one byte per channel)
//    ==========================================================================
//                    (see TH1::TH1 for explanation of parameters)
//
   fDimension = 1;
}

//______________________________________________________________________________
TH1C::TH1C(const char *name,const char *title,Int_t nbins,const Double_t *xbins)
     : TH1(name,title,nbins,xbins), TArrayC(nbins+2)
{
//
//    Create a 1-Dim histogram with variable bins of type char (one byte per channel)
//    ==========================================================================
//                    (see TH1::TH1 for explanation of parameters)
//
   fDimension = 1;
}

//______________________________________________________________________________
TH1C::~TH1C()
{

}

//______________________________________________________________________________
TH1C::TH1C(const TH1C &h1c)
{
   ((TH1C&)h1c).Copy(*this);
}

//______________________________________________________________________________
void TH1C::AddBinContent(Int_t bin)
{
//*-*-*-*-*-*-*-*-*-*Increment bin content by 1*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                ==========================

   if (fArray[bin] < 127) fArray[bin]++;
}

//______________________________________________________________________________
void TH1C::AddBinContent(Int_t bin, Stat_t w)
{
//*-*-*-*-*-*-*-*-*-*Increment bin content by w*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                ==========================

   Int_t newval = fArray[bin] + Int_t(w);
   if (newval > -128 && newval < 128) {fArray[bin] = Char_t(newval); return;}
   if (newval < -127) fArray[bin] = -127;
   if (newval >  127) fArray[bin] =  127;
}

//______________________________________________________________________________
void TH1C::Copy(TObject &newth1)
{
   TH1::Copy(newth1);
   TArrayC::Copy((TH1C&)newth1);
}

//______________________________________________________________________________
TH1 *TH1C::DrawCopy(Option_t *option)
{

   TString opt = option;
   opt.ToLower();
   if (gPad && !opt.Contains("same")) gPad->Clear();
   TH1C *newth1 = (TH1C*)Clone();
   newth1->SetDirectory(0);
   newth1->SetBit(kCanDelete);
   newth1->AppendPad(opt.Data());
   return newth1;
}

//______________________________________________________________________________
Stat_t TH1C::GetBinContent(Int_t bin) const
{
   if (bin < 0) bin = 0;
   if (bin >= fNcells) bin = fNcells-1;
   if (!fArray) return 0;
   return Stat_t (fArray[bin]);
}

//______________________________________________________________________________
void TH1C::Reset(Option_t *option)
{
   TH1::Reset(option);
   TArrayC::Reset();
}

//______________________________________________________________________________
TH1C& TH1C::operator=(const TH1C &h1)
{
   if (this != &h1)  ((TH1C&)h1).Copy(*this);
   return *this;
}


//______________________________________________________________________________
TH1C operator*(Double_t c1, TH1C &h1)
{
   TH1C hnew = h1;
   hnew.Scale(c1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH1C operator+(TH1C &h1, TH1C &h2)
{
   TH1C hnew = h1;
   hnew.Add(&h2,1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH1C operator-(TH1C &h1, TH1C &h2)
{
   TH1C hnew = h1;
   hnew.Add(&h2,-1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH1C operator*(TH1C &h1, TH1C &h2)
{
   TH1C hnew = h1;
   hnew.Multiply(&h2);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH1C operator/(TH1C &h1, TH1C &h2)
{
   TH1C hnew = h1;
   hnew.Divide(&h2);
   hnew.SetDirectory(0);
   return hnew;
}

ClassImp(TH1S)

//______________________________________________________________________________
//                     TH1S methods
//______________________________________________________________________________
TH1S::TH1S(): TH1(), TArrayS()
{
   fDimension = 1;
}

//______________________________________________________________________________
TH1S::TH1S(const char *name,const char *title,Int_t nbins,Axis_t xlow,Axis_t xup)
     : TH1(name,title,nbins,xlow,xup), TArrayS(nbins+2)
{
//
//    Create a 1-Dim histogram with fix bins of type short
//    ====================================================
//           (see TH1::TH1 for explanation of parameters)
//
   fDimension = 1;
}

//______________________________________________________________________________
TH1S::TH1S(const char *name,const char *title,Int_t nbins,const Float_t *xbins)
     : TH1(name,title,nbins,xbins), TArrayS(nbins+2)
{
//
//    Create a 1-Dim histogram with variable bins of type short
//    =========================================================
//           (see TH1::TH1 for explanation of parameters)
//
   fDimension = 1;
}

//______________________________________________________________________________
TH1S::TH1S(const char *name,const char *title,Int_t nbins,const Double_t *xbins)
     : TH1(name,title,nbins,xbins), TArrayS(nbins+2)
{
//
//    Create a 1-Dim histogram with variable bins of type short
//    =========================================================
//           (see TH1::TH1 for explanation of parameters)
//
   fDimension = 1;
}

//______________________________________________________________________________
TH1S::~TH1S()
{

}

//______________________________________________________________________________
TH1S::TH1S(const TH1S &h1s)
{
   ((TH1S&)h1s).Copy(*this);
}

//______________________________________________________________________________
void TH1S::AddBinContent(Int_t bin)
{
//*-*-*-*-*-*-*-*-*-*Increment bin content by 1*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                ==========================

   if (fArray[bin] < 32767) fArray[bin]++;
}

//______________________________________________________________________________
void TH1S::AddBinContent(Int_t bin, Stat_t w)
{
//*-*-*-*-*-*-*-*-*-*Increment bin content by w*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                ==========================

   Int_t newval = fArray[bin] + Int_t(w);
   if (newval > -32768 && newval < 32768) {fArray[bin] = Short_t(newval); return;}
   if (newval < -32767) fArray[bin] = -32767;
   if (newval >  32767) fArray[bin] =  32767;
}

//______________________________________________________________________________
void TH1S::Copy(TObject &newth1)
{
   TH1::Copy(newth1);
   TArrayS::Copy((TH1S&)newth1);
}

//______________________________________________________________________________
TH1 *TH1S::DrawCopy(Option_t *option)
{
   TString opt = option;
   opt.ToLower();
   if (gPad && !opt.Contains("same")) gPad->Clear();
   TH1S *newth1 = (TH1S*)Clone();
   newth1->SetDirectory(0);
   newth1->SetBit(kCanDelete);
   newth1->AppendPad(opt.Data());
   return newth1;
}

//______________________________________________________________________________
Stat_t TH1S::GetBinContent(Int_t bin) const
{
   if (bin < 0) bin = 0;
   if (bin >= fNcells) bin = fNcells-1;
   if (!fArray) return 0;
   return Stat_t (fArray[bin]);
}

//______________________________________________________________________________
void TH1S::Reset(Option_t *option)
{
   TH1::Reset(option);
   TArrayS::Reset();
}

//______________________________________________________________________________
TH1S& TH1S::operator=(const TH1S &h1)
{
   if (this != &h1)  ((TH1S&)h1).Copy(*this);
   return *this;
}


//______________________________________________________________________________
TH1S operator*(Double_t c1, TH1S &h1)
{
   TH1S hnew = h1;
   hnew.Scale(c1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH1S operator+(TH1S &h1, TH1S &h2)
{
   TH1S hnew = h1;
   hnew.Add(&h2,1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH1S operator-(TH1S &h1, TH1S &h2)
{
   TH1S hnew = h1;
   hnew.Add(&h2,-1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH1S operator*(TH1S &h1, TH1S &h2)
{
   TH1S hnew = h1;
   hnew.Multiply(&h2);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH1S operator/(TH1S &h1, TH1S &h2)
{
   TH1S hnew = h1;
   hnew.Divide(&h2);
   hnew.SetDirectory(0);
   return hnew;
}

ClassImp(TH1F)

//______________________________________________________________________________
//                     TH1F methods
//______________________________________________________________________________
TH1F::TH1F(): TH1(), TArrayF()
{
   fDimension = 1;
}

//______________________________________________________________________________
TH1F::TH1F(const char *name,const char *title,Int_t nbins,Axis_t xlow,Axis_t xup)
     : TH1(name,title,nbins,xlow,xup), TArrayF(nbins+2)
{
//
//    Create a 1-Dim histogram with fix bins of type float
//    ====================================================
//           (see TH1::TH1 for explanation of parameters)
//
   fDimension = 1;
}

//______________________________________________________________________________
TH1F::TH1F(const char *name,const char *title,Int_t nbins,const Float_t *xbins)
     : TH1(name,title,nbins,xbins), TArrayF(nbins+2)
{
//
//    Create a 1-Dim histogram with variable bins of type float
//    =========================================================
//           (see TH1::TH1 for explanation of parameters)
//
   fDimension = 1;
}

//______________________________________________________________________________
TH1F::TH1F(const char *name,const char *title,Int_t nbins,const Double_t *xbins)
     : TH1(name,title,nbins,xbins), TArrayF(nbins+2)
{
//
//    Create a 1-Dim histogram with variable bins of type float
//    =========================================================
//           (see TH1::TH1 for explanation of parameters)
//
   fDimension = 1;
}

//______________________________________________________________________________
TH1F::TH1F(const TVector &v)
     : TH1("TVector","",v.GetNrows(),0,v.GetNrows()), TArrayF(v.GetNrows()+2)
{
// Create a histogram from a TVector
// by default the histogram name is "TVector" and title = ""

   fDimension = 1;
   for (Int_t i=0;i<v.GetNrows();i++) {
      SetBinContent(i+1,v(i));
   }
}

//______________________________________________________________________________
TH1F::TH1F(const TH1F &h)
{
   ((TH1F&)h).Copy(*this);
}

//______________________________________________________________________________
TH1F::~TH1F()
{

}

//______________________________________________________________________________
void TH1F::Copy(TObject &newth1)
{
   TH1::Copy(newth1);
   TArrayF::Copy((TH1F&)newth1);
}

//______________________________________________________________________________
TH1 *TH1F::DrawCopy(Option_t *option)
{
   TString opt = option;
   opt.ToLower();
   if (gPad && !opt.Contains("same")) gPad->Clear();
   TH1F *newth1 = (TH1F*)Clone();
   newth1->SetDirectory(0);
   newth1->SetBit(kCanDelete);
   newth1->AppendPad(opt.Data());
   return newth1;
}

//______________________________________________________________________________
Stat_t TH1F::GetBinContent(Int_t bin) const
{
   if (bin < 0) bin = 0;
   if (bin >= fNcells) bin = fNcells-1;
   if (!fArray) return 0;
   return Stat_t (fArray[bin]);
}

//______________________________________________________________________________
void TH1F::Reset(Option_t *option)
{
   TH1::Reset(option);
   TArrayF::Reset();
}

//______________________________________________________________________________
TH1F& TH1F::operator=(const TH1F &h1)
{
   if (this != &h1)  ((TH1F&)h1).Copy(*this);
   return *this;
}


//______________________________________________________________________________
TH1F operator*(Double_t c1, TH1F &h1)
{
   TH1F hnew = h1;
   hnew.Scale(c1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH1F operator+(TH1F &h1, TH1F &h2)
{
   TH1F hnew = h1;
   hnew.Add(&h2,1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH1F operator-(TH1F &h1, TH1F &h2)
{
   TH1F hnew = h1;
   hnew.Add(&h2,-1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH1F operator*(TH1F &h1, TH1F &h2)
{
   TH1F hnew = h1;
   hnew.Multiply(&h2);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH1F operator/(TH1F &h1, TH1F &h2)
{
   TH1F hnew = h1;
   hnew.Divide(&h2);
   hnew.SetDirectory(0);
   return hnew;
}


ClassImp(TH1D)

//______________________________________________________________________________
//                     TH1D methods
//______________________________________________________________________________
TH1D::TH1D(): TH1(), TArrayD()
{
   fDimension = 1;
}

//______________________________________________________________________________
TH1D::TH1D(const char *name,const char *title,Int_t nbins,Axis_t xlow,Axis_t xup)
     : TH1(name,title,nbins,xlow,xup), TArrayD(nbins+2)
{
//
//    Create a 1-Dim histogram with fix bins of type double
//    =====================================================
//           (see TH1::TH1 for explanation of parameters)
//
   fDimension = 1;
}

//______________________________________________________________________________
TH1D::TH1D(const char *name,const char *title,Int_t nbins,const Float_t *xbins)
     : TH1(name,title,nbins,xbins), TArrayD(nbins+2)
{
//
//    Create a 1-Dim histogram with variable bins of type double
//    =====================================================
//           (see TH1::TH1 for explanation of parameters)
//
   fDimension = 1;
}

//______________________________________________________________________________
TH1D::TH1D(const char *name,const char *title,Int_t nbins,const Double_t *xbins)
     : TH1(name,title,nbins,xbins), TArrayD(nbins+2)
{
//
//    Create a 1-Dim histogram with variable bins of type double
//    =====================================================
//           (see TH1::TH1 for explanation of parameters)
//
   fDimension = 1;
}

//______________________________________________________________________________
TH1D::TH1D(const TVectorD &v)
     : TH1("TVectorD","",v.GetNrows(),0,v.GetNrows()), TArrayD(v.GetNrows()+2)
{
// Create a histogram from a TVector
// by default the histogram name is "TVector" and title = ""

   fDimension = 1;
   for (Int_t i=0;i<v.GetNrows();i++) {
      SetBinContent(i+1,v(i));
   }
}

//______________________________________________________________________________
TH1D::~TH1D()
{

}

//______________________________________________________________________________
TH1D::TH1D(const TH1D &h1d)
{
   ((TH1D&)h1d).Copy(*this);
}

//______________________________________________________________________________
void TH1D::Copy(TObject &newth1)
{
   TH1::Copy(newth1);
   TArrayD::Copy((TH1D&)newth1);
}

//______________________________________________________________________________
TH1 *TH1D::DrawCopy(Option_t *option)
{
   TString opt = option;
   opt.ToLower();
   if (gPad && !opt.Contains("same")) gPad->Clear();
   TH1D *newth1 = (TH1D*)Clone();
   newth1->SetDirectory(0);
   newth1->SetBit(kCanDelete);
   newth1->AppendPad(opt.Data());
   return newth1;
}

//______________________________________________________________________________
Stat_t TH1D::GetBinContent(Int_t bin) const
{
   if (bin < 0) bin = 0;
   if (bin >= fNcells) bin = fNcells-1;
   if (!fArray) return 0;
   return Stat_t (fArray[bin]);
}

//______________________________________________________________________________
void TH1D::Reset(Option_t *option)
{
   TH1::Reset(option);
   TArrayD::Reset();
}

//______________________________________________________________________________
TH1D& TH1D::operator=(const TH1D &h1)
{
   if (this != &h1)  ((TH1D&)h1).Copy(*this);
   return *this;
}

//______________________________________________________________________________
TH1D operator*(Double_t c1, TH1D &h1)
{
   TH1D hnew = h1;
   hnew.Scale(c1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH1D operator+(TH1D &h1, TH1D &h2)
{
   TH1D hnew = h1;
   hnew.Add(&h2,1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH1D operator-(TH1D &h1, TH1D &h2)
{
   TH1D hnew = h1;
   hnew.Add(&h2,-1);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH1D operator*(TH1D &h1, TH1D &h2)
{
   TH1D hnew = h1;
   hnew.Multiply(&h2);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH1D operator/(TH1D &h1, TH1D &h2)
{
   TH1D hnew = h1;
   hnew.Divide(&h2);
   hnew.SetDirectory(0);
   return hnew;
}

//______________________________________________________________________________
TH1 *R__H(Int_t hid)
{
   //return pointer to histogram with name
   //   hid if id >=0
   //   h_id if id <0

   char hname[20];
   if(hid >= 0) sprintf(hname,"h%d",hid);
   else         sprintf(hname,"h_%d",hid);
   return (TH1*)gDirectory->Get(hname);
}

//______________________________________________________________________________
TH1 *R__H(const char * hname)
{
   //return pointer to histogram with name hname

   return (TH1*)gDirectory->Get(hname);
}


