// @(#)root/spectrum:$Id$
// Author: Miroslav Morhac   27/05/99

#include "TSpectrum.h"
#include "TPolyMarker.h"
#include "TVirtualPad.h"
#include "TList.h"
#include "TH1.h"
#include "TMath.h"


//______________________________________________________________________________
/* Begin_Html
<center><h2>Advanced Spectra Processing</h2></center>
This class contains advanced spectra processing functions for:
<ul>
<li> One-dimensional background estimation
<li> One-dimensional smoothing
<li> One-dimensional deconvolution
<li> One-dimensional peak search
</ul>
<p>
Author:
<br>
<br> Miroslav Morhac
<br> Institute of Physics
<br> Slovak Academy of Sciences
<br> Dubravska cesta 9, 842 28 BRATISLAVA
<br> SLOVAKIA
<br> email:fyzimiro@savba.sk,    fax:+421 7 54772479
<br>
<br>

The original code in C has been repackaged as a C++ class by R.Brun.
<p>
The algorithms in this class have been published in the following references:
<ol>
<li> M.Morhac et al.: Background elimination methods for
multidimensional coincidence gamma-ray spectra. Nuclear
Instruments and Methods in Physics Research A 401 (1997) 113-132.
<li> M.Morhac et al.: Efficient one- and two-dimensional Gold
deconvolution and its application to gamma-ray spectra
decomposition. Nuclear Instruments and Methods in Physics
Research A 401 (1997) 385-408.
<li> M.Morhac et al.: Identification of peaks in multidimensional
coincidence gamma-ray spectra. Nuclear Instruments and Methods in
Research Physics A  443(2000), 108-125.
</ol>
These NIM papers are also available as doc or ps files from:
<ul>
<li> <A href="ftp://root.cern.ch/root/Spectrum.doc">Spectrum.doc</A><br>
<li> <A href="ftp://root.cern.ch/root/SpectrumDec.ps.gz">SpectrumDec.ps.gz</A><br>
<li> <A href="ftp://root.cern.ch/root/SpectrumSrc.ps.gz">SpectrumSrc.ps.gz</A><br>
<li> <A href="ftp://root.cern.ch/root/SpectrumBck.ps.gz">SpectrumBck.ps.gz</A><br>
</ul>
End_Html */

Int_t TSpectrum::fgIterations    = 3;
Int_t TSpectrum::fgAverageWindow = 3;

#define PEAK_WINDOW 1024
ClassImp(TSpectrum)


//______________________________________________________________________________
TSpectrum::TSpectrum() :TNamed("Spectrum", "Miroslav Morhac peak finder")
{
   /* Begin_Html
   Constructor.
   End_Html */

   Int_t n = 100;
   fMaxPeaks  = n;
   fPosition   = new Float_t[n];
   fPositionX  = new Float_t[n];
   fPositionY  = new Float_t[n];
   fResolution = 1;
   fHistogram  = 0;
   fNPeaks     = 0;
}


//______________________________________________________________________________
TSpectrum::TSpectrum(Int_t maxpositions, Float_t resolution)
          :TNamed("Spectrum", "Miroslav Morhac peak finder")
{
   /* Begin_Html
   <ul>
   <li> maxpositions: maximum number of peaks
   <li> resolution: determines resolution of the neighboring peaks
                   default value is 1 correspond to 3 sigma distance
                   between peaks. Higher values allow higher resolution
                   (smaller distance between peaks.
                   May be set later through SetResolution.
   </ul>
   End_Html */

   Int_t n = maxpositions;
   if (n <= 0) n = 1;
   fMaxPeaks  = n;
   fPosition  = new Float_t[n];
   fPositionX = new Float_t[n];
   fPositionY = new Float_t[n];
   fHistogram = 0;
   fNPeaks    = 0;
   SetResolution(resolution);
}


//______________________________________________________________________________
TSpectrum::~TSpectrum()
{
   /* Begin_Html
   Destructor.
   End_Html */

   delete [] fPosition;
   delete [] fPositionX;
   delete [] fPositionY;
   delete    fHistogram;
}


//______________________________________________________________________________
void TSpectrum::SetAverageWindow(Int_t w)
{
   /* Begin_Html
   Static function: Set average window of searched peaks
   (see TSpectrum::SearchHighRes).
   End_Html */

   fgAverageWindow = w;
}


//______________________________________________________________________________
void TSpectrum::SetDeconIterations(Int_t n)
{
   /* Begin_Html
   Static function: Set max number of decon iterations in deconvolution
   operation (see TSpectrum::SearchHighRes).
   End_Html */

   fgIterations = n;
}


//______________________________________________________________________________
TH1 *TSpectrum::Background(const TH1 * h, int numberIterations,
                           Option_t * option)
{
   /* Begin_Html
   <b>One-dimensional background estimation function.</b>
   <p>
   This function calculates the background spectrum in the input histogram h.
   The background is returned as a histogram.
   <p>
   Function parameters:
   <ul>
   <li> h: input 1-d histogram
   <li> numberIterations, (default value = 20)
      Increasing numberIterations make the result smoother and lower.
   <li> option: may contain one of the following options:
      <ul>
      <li> to set the direction parameter
      "BackIncreasingWindow". By default the direction is BackDecreasingWindow
      <li> filterOrder-order of clipping filter,  (default "BackOrder2")
                  -possible values= "BackOrder4"
                                    "BackOrder6"
                                    "BackOrder8"
      <li> "nosmoothing"- if selected, the background is not smoothed
           By default the background is smoothed.
      <li> smoothWindow-width of smoothing window, (default is "BackSmoothing3")
                  -possible values= "BackSmoothing5"
                                    "BackSmoothing7"
                                    "BackSmoothing9"
                                    "BackSmoothing11"
                                    "BackSmoothing13"
                                    "BackSmoothing15"
      <li> "Compton" if selected the estimation of Compton edge
                  will be included.
      <li> "same" : if this option is specified, the resulting background
                 histogram is superimposed on the picture in the current pad.
      </ul>
   </ul>
   NOTE that the background is only evaluated in the current range of h.
   ie, if h has a bin range (set via h->GetXaxis()->SetRange(binmin,binmax),
   the returned histogram will be created with the same number of bins
   as the input histogram h, but only bins from binmin to binmax will be filled
   with the estimated background.
   End_Html */

   if (h == 0) return 0;
   Int_t dimension = h->GetDimension();
   if (dimension > 1) {
      Error("Search", "Only implemented for 1-d histograms");
      return 0;
   }
   TString opt = option;
   opt.ToLower();

   //set options
   Int_t direction = kBackDecreasingWindow;
   if (opt.Contains("backincreasingwindow")) direction = kBackIncreasingWindow;
   Int_t filterOrder = kBackOrder2;
   if (opt.Contains("backorder4")) filterOrder = kBackOrder4;
   if (opt.Contains("backorder6")) filterOrder = kBackOrder6;
   if (opt.Contains("backorder8")) filterOrder = kBackOrder8;
   Bool_t smoothing = kTRUE;
   if (opt.Contains("nosmoothing")) smoothing = kFALSE;
   Int_t smoothWindow = kBackSmoothing3;
   if (opt.Contains("backsmoothing5"))  smoothWindow = kBackSmoothing5;
   if (opt.Contains("backsmoothing7"))  smoothWindow = kBackSmoothing7;
   if (opt.Contains("backsmoothing9"))  smoothWindow = kBackSmoothing9;
   if (opt.Contains("backsmoothing11")) smoothWindow = kBackSmoothing11;
   if (opt.Contains("backsmoothing13")) smoothWindow = kBackSmoothing13;
   if (opt.Contains("backsmoothing15")) smoothWindow = kBackSmoothing15;
   Bool_t compton = kFALSE;
   if (opt.Contains("compton")) compton = kTRUE;

   Int_t first = h->GetXaxis()->GetFirst();
   Int_t last  = h->GetXaxis()->GetLast();
   Int_t size = last-first+1;
   Int_t i;
   Float_t * source = new float[size];
   for (i = 0; i < size; i++) source[i] = h->GetBinContent(i + first);

   //find background (source is input and in output contains the background
   Background(source,size,numberIterations, direction, filterOrder,smoothing,
              smoothWindow,compton);

   //create output histogram containing backgound
   //only bins in the range of the input histogram are filled
   Int_t nch = strlen(h->GetName());
   char *hbname = new char[nch+20];
   snprintf(hbname,nch+20,"%s_background",h->GetName());
   TH1 *hb = (TH1*)h->Clone(hbname);
   hb->Reset();
   hb->GetListOfFunctions()->Delete();
   hb->SetLineColor(2);
   for (i=0; i< size; i++) hb->SetBinContent(i+first,source[i]);
   hb->SetEntries(size);

   //if option "same is specified, draw the result in the pad
   if (opt.Contains("same")) {
      if (gPad) delete gPad->GetPrimitive(hbname);
      hb->Draw("same");
   }
   delete [] source;
   delete [] hbname;
   return hb;
}


//______________________________________________________________________________
void TSpectrum::Print(Option_t *) const
{
   /* Begin_Html
   Print the array of positions.
   End_Html */

   printf("\nNumber of positions = %d\n",fNPeaks);
   for (Int_t i=0;i<fNPeaks;i++) {
      printf(" x[%d] = %g, y[%d] = %g\n",i,fPositionX[i],i,fPositionY[i]);
   }
}


//______________________________________________________________________________
Int_t TSpectrum::Search(const TH1 * hin, Double_t sigma, Option_t * option,
                        Double_t threshold)
{
   /* Begin_Html
   <b>One-dimensional peak search function</b>
   <p>
   This function searches for peaks in source spectrum in hin
   The number of found peaks and their positions are written into
   the members fNpeaks and fPositionX.
   The search is performed in the current histogram range.
   <p>
   Function parameters:
   <ul>
   <li> hin:       pointer to the histogram of source spectrum
   <li> sigma:   sigma of searched peaks, for details we refer to manual
   <li> threshold: (default=0.05)  peaks with amplitude less than
       threshold*highest_peak are discarded.  0<threshold<1
   </ul>
   By default, the background is removed before deconvolution.
   Specify the option "nobackground" to not remove the background.
   <p>
   By default the "Markov" chain algorithm is used.
   Specify the option "noMarkov" to disable this algorithm
   Note that by default the source spectrum is replaced by a new spectrum
   <p>
   By default a polymarker object is created and added to the list of
   functions of the histogram. The histogram is drawn with the specified
   option and the polymarker object drawn on top of the histogram.
   The polymarker coordinates correspond to the npeaks peaks found in
   the histogram.
   <p>
   A pointer to the polymarker object can be retrieved later via:
   <pre>
    TList *functions = hin->GetListOfFunctions();
    TPolyMarker *pm = (TPolyMarker*)functions->FindObject("TPolyMarker");
   </pre>
   Specify the option "goff" to disable the storage and drawing of the
   polymarker.
   End_Html */

   if (hin == 0) return 0;
   Int_t dimension = hin->GetDimension();
   if (dimension > 2) {
      Error("Search", "Only implemented for 1-d and 2-d histograms");
      return 0;
   }
   if (threshold <=0 || threshold >= 1) {
      Warning("Search","threshold must 0<threshold<1, threshol=0.05 assumed");
      threshold = 0.05;
   }
   TString opt = option;
   opt.ToLower();
   Bool_t background = kTRUE;
   if (opt.Contains("nobackground")) {
      background = kFALSE;
      opt.ReplaceAll("nobackground","");
   }
   Bool_t markov = kTRUE;
   if (opt.Contains("nomarkov")) {
      markov = kFALSE;
      opt.ReplaceAll("nomarkov","");
   }
   if (dimension == 1) {
      Int_t first = hin->GetXaxis()->GetFirst();
      Int_t last  = hin->GetXaxis()->GetLast();
      Int_t size = last-first+1;
      Int_t i, bin, npeaks;
      Float_t * source = new float[size];
      Float_t * dest   = new float[size];
      for (i = 0; i < size; i++) source[i] = hin->GetBinContent(i + first);
      if (sigma < 1) {
         sigma = size/fMaxPeaks;
         if (sigma < 1) sigma = 1;
         if (sigma > 8) sigma = 8;
      }
      npeaks = SearchHighRes(source, dest, size, sigma, 100*threshold,
                             background, fgIterations, markov, fgAverageWindow);

      for (i = 0; i < npeaks; i++) {
         bin = first + Int_t(fPositionX[i] + 0.5);
         fPositionX[i] = hin->GetBinCenter(bin);
         fPositionY[i] = hin->GetBinContent(bin);
      }
      delete [] source;
      delete [] dest;

      if (opt.Contains("goff"))
         return npeaks;
      if (!npeaks) return 0;
      TPolyMarker * pm =
         (TPolyMarker*)hin->GetListOfFunctions()->FindObject("TPolyMarker");
      if (pm) {
         hin->GetListOfFunctions()->Remove(pm);
         delete pm;
      }
      pm = new TPolyMarker(npeaks, fPositionX, fPositionY);
      hin->GetListOfFunctions()->Add(pm);
      pm->SetMarkerStyle(23);
      pm->SetMarkerColor(kRed);
      pm->SetMarkerSize(1.3);
      opt.ReplaceAll(" ","");
      opt.ReplaceAll(",","");
      ((TH1*)hin)->Draw(opt.Data());
      return npeaks;
   }
   return 0;
}


//______________________________________________________________________________
void TSpectrum::SetResolution(Float_t resolution)
{
   /* Begin_Html
  resolution: determines resolution of the neighboring peaks
              default value is 1 correspond to 3 sigma distance
              between peaks. Higher values allow higher resolution
              (smaller distance between peaks.
              May be set later through SetResolution.
   End_Html */

   if (resolution > 1)
      fResolution = resolution;
   else
      fResolution = 1;
}


//______________________________________________________________________________
const char *TSpectrum::Background(float *spectrum, int ssize,
                                          int numberIterations,
                                          int direction, int filterOrder,
                                          bool smoothing,int smoothWindow,
                                          bool compton)
{
/* Begin_Html
This function calculates background spectrum from source spectrum.
The result is placed in the vector pointed by spe1945ctrum pointer.
The goal is to separate the useful information (peaks) from useless
information (background).

<ul>
<li> method is based on Sensitive Nonlinear Iterative Peak (SNIP) clipping
     algorithm.
<li> new value in the channel "i" is calculated
</ul>

<img width=486 height=72 src="gif/TSpectrum_Background.gif">

where p = 1, 2, ..., numberIterations. In fact it represents second order
difference filter (-1,2,-1).

One can also change the
direction of the change of the clipping window, the order of the clipping
filter, to include smoothing, to set width of smoothing window and to include
the estimation of Compton edges. On successful completion it returns 0. On
error it returns pointer to the string describing error.

<h4>Parameters:</h4>
<ul>
<li> spectrum: pointer to the vector of source spectrum
<li> ssize: length of the spectrum vector
<li> numberIterations: maximal width of clipping window,
<li> direction:  direction of change of clipping window.
     Possible values: kBackIncreasingWindow, kBackDecreasingWindow
<li> filterOrder: order of clipping filter.
     Possible values: kBackOrder2, kBackOrder4, kBackOrder6, kBackOrder8
<li> smoothing: logical variable whether the smoothing operation in the
     estimation of background will be included.
     Possible values: kFALSE, kTRUE
<li> smoothWindow: width of smoothing window.
     Possible values: kBackSmoothing3, kBackSmoothing5, kBackSmoothing7,
     kBackSmoothing9, kBackSmoothing11, kBackSmoothing13, kBackSmoothing15.
<li> compton: logical variable whether the estimation of Compton edge will be
     included. Possible values: kFALSE, kTRUE.
</ul>


<h4>References:</h4>
<ol>
<li> C. G Ryan et al.: SNIP, a statistics-sensitive background treatment for the
quantitative analysis of PIXE spectra in geoscience applications. NIM, B34
(1988), 396-402.

<li> M. Morhá&#269;, J. Kliman, V. Matoušek, M. Veselský, I. Turzo:
Background elimination methods for multidimensional gamma-ray spectra. NIM,
A401 (1997) 113-132.

<li> D. D. Burgess, R. J. Tervo: Background estimation for gamma-ray
spectroscopy. NIM 214 (1983), 431-434.
</ol>

Example 1 script Background_incr.c:
<p>
<img width=601 height=407 src="gif/TSpectrum_Background_incr.jpg">
<p>
Figure 1 Example of the estimation of background for number of iterations=6.
Original spectrum is shown in black color, estimated background in red color.
<p>
Script:
<pre>
// Example to illustrate the background estimator (class TSpectrum).
// To execute this example, do
// root > .x Background_incr.C

#include <TSpectrum>

void Background_incr() {
   Int_t i;
   Double_t nbins = 256;
   Double_t xmin  = 0;
   Double_t xmax  = (Double_t)nbins;
   Float_t * source = new float[nbins];
   TH1F *back = new TH1F("back","",nbins,xmin,xmax);
   TH1F *d = new TH1F("d","",nbins,xmin,xmax);
   TFile *f = new TFile("spectra\\TSpectrum.root");
   back=(TH1F*) f->Get("back1;1");
   TCanvas *Background = gROOT->GetListOfCanvases()->FindObject("Background");
   if (!Background) Background =
     new TCanvas("Background",
                 "Estimation of background with increasing window",
                 10,10,1000,700);
   back->Draw("L");
   TSpectrum *s = new TSpectrum();
   for (i = 0; i < nbins; i++) source[i]=back->GetBinContent(i + 1);
   s->Background(source,nbins,6,kBackIncreasingWindow,kBackOrder2,kFALSE,
                 kBackSmoothing3,kFALSE);
   for (i = 0; i < nbins; i++) d->SetBinContent(i + 1,source[i]);
   d->SetLineColor(kRed);
   d->Draw("SAME L");
}
</pre>

Example 2 script Background_decr.c:
<p>
In Figure 1. one can notice that at the edges of the peaks the estimated
background goes under the peaks. An alternative approach is to decrease the
clipping window from a given value numberIterations to the value of one, which
is presented in this example.
<p>
<img width=601 height=407 src="gif/TSpectrum_Background_decr.jpg">
<p>
Figure 2 Example of the estimation of background for numberIterations=6 using
decreasing clipping window algorithm. Original spectrum is shown in black
color, estimated background in red color.
<p>
Script:

<pre>
// Example to illustrate the background estimator (class TSpectrum).
// To execute this example, do
// root > .x Background_decr.C

#include <TSpectrum>

void Background_decr() {
   Int_t i;
   Double_t nbins = 256;
   Double_t xmin  = 0;
   Double_t xmax  = (Double_t)nbins;
   Float_t * source = new float[nbins];
   TH1F *back = new TH1F("back","",nbins,xmin,xmax);
   TH1F *d = new TH1F("d","",nbins,xmin,xmax);
   TFile *f = new TFile("spectra\\TSpectrum.root");
   back=(TH1F*) f->Get("back1;1");
   TCanvas *Background = gROOT->GetListOfCanvases()->FindObject("Background");
   if (!Background) Background =
     new TCanvas("Background","Estimation of background with decreasing window",
                 10,10,1000,700);
   back->Draw("L");
   TSpectrum *s = new TSpectrum();
   for (i = 0; i < nbins; i++) source[i]=back->GetBinContent(i + 1);
   s->Background(source,nbins,6,kBackDecreasingWindow,kBackOrder2,kFALSE,
                 kBackSmoothing3,kFALSE);
   for (i = 0; i < nbins; i++) d->SetBinContent(i + 1,source[i]);
   d->SetLineColor(kRed);
   d->Draw("SAME L");
}
</pre>

Example 3 script Background_width.c:
<p>
The question is how to choose the width of the clipping window, i.e.,
numberIterations parameter. The influence of this parameter on the estimated
background is illustrated in Figure 3.
<p>
<img width=601 height=407 src="gif/TSpectrum_Background_width.jpg">
<p>
Figure 3 Example of the influence of clipping window width on the estimated
background for numberIterations=4 (red line), 6 (blue line) 8 (green line) using
decreasing clipping window algorithm.

<p>
in general one should set this parameter so that the value
2*numberIterations+1 was greater than the widths of preserved objects (peaks).

<p>
Script:

<pre>
// Example to illustrate the influence of the clipping window width on the
// estimated background. To execute this example, do:
// root > .x Background_width.C

#include <TSpectrum>

void Background_width() {
   Int_t i;
   Double_t nbins = 256;
   Double_t xmin  = 0;
   Double_t xmax  = (Double_t)nbins;
   Float_t * source = new float[nbins];
   TH1F *h = new TH1F("h","",nbins,xmin,xmax);
   TH1F *d1 = new TH1F("d1","",nbins,xmin,xmax);
   TH1F *d2 = new TH1F("d2","",nbins,xmin,xmax);
   TH1F *d3 = new TH1F("d3","",nbins,xmin,xmax);
   TFile *f = new TFile("spectra\\TSpectrum.root");
   h=(TH1F*) f->Get("back1;1");
   TCanvas *background = gROOT->GetListOfCanvases()->FindObject("background");
   if (!background) background = new TCanvas("background",
   "Influence of clipping window width on the estimated background",
   10,10,1000,700);
   h->Draw("L");
   TSpectrum *s = new TSpectrum();
   for (i = 0; i < nbins; i++) source[i]=h->GetBinContent(i + 1);
   s->Background(source,nbins,4,kBackDecreasingWindow,kBackOrder2,kFALSE,
   kBackSmoothing3,kFALSE);
   for (i = 0; i < nbins; i++) d1->SetBinContent(i + 1,source[i]);
   d1->SetLineColor(kRed);
   d1->Draw("SAME L");
   for (i = 0; i < nbins; i++) source[i]=h->GetBinContent(i + 1);
   s->Background(source,nbins,6,kBackDecreasingWindow,kBackOrder2,kFALSE,
   kBackSmoothing3,kFALSE);
   for (i = 0; i < nbins; i++) d2->SetBinContent(i + 1,source[i]);
   d2->SetLineColor(kBlue);
   d2->Draw("SAME L");
   for (i = 0; i < nbins; i++) source[i]=h->GetBinContent(i + 1);
   s->Background(source,nbins,8,kBackDecreasingWindow,kBackOrder2,kFALSE,
   kBackSmoothing3,kFALSE);
   for (i = 0; i < nbins; i++) d3->SetBinContent(i + 1,source[i]);
   d3->SetLineColor(kGreen);
   d3->Draw("SAME L");
}
</pre>

Example 4 script Background_width2.c:
<p>
another example for very complex spectrum is given in Figure 4.
<p>
<img width=601 height=407 src="gif/TSpectrum_Background_width2.jpg">
<p>
Figure 4 Example of the influence of clipping window width on the estimated
background for numberIterations=10 (red line), 20 (blue line), 30 (green line)
and 40 (magenta line) using decreasing clipping window algorithm.

<p>
Script:

<pre>
// Example to illustrate the influence of the clipping window width on the
// estimated background. To execute this example, do:
// root > .x Background_width2.C

#include <TSpectrum>

void Background_width2() {
   Int_t i;
   Double_t nbins = 4096;
   Double_t xmin  = 0;
   Double_t xmax  = (Double_t)4096;
   Float_t * source = new float[nbins];
   TH1F *h = new TH1F("h","",nbins,xmin,xmax);
   TH1F *d1 = new TH1F("d1","",nbins,xmin,xmax);
   TH1F *d2 = new TH1F("d2","",nbins,xmin,xmax);
   TH1F *d3 = new TH1F("d3","",nbins,xmin,xmax);
   TH1F *d4 = new TH1F("d4","",nbins,xmin,xmax);
   TFile *f = new TFile("spectra\\TSpectrum.root");
   h=(TH1F*) f->Get("back2;1");
   TCanvas *background = gROOT->GetListOfCanvases()->FindObject("background");
   if (!background) background = new TCanvas("background",
   "Influence of clipping window width on the estimated background",
   10,10,1000,700);
   h->SetAxisRange(0,1000);
   h->SetMaximum(20000);
   h->Draw("L");
   TSpectrum *s = new TSpectrum();
   for (i = 0; i < nbins; i++) source[i]=h->GetBinContent(i + 1);
   s->Background(source,nbins,10,kBackDecreasingWindow,kBackOrder2,kFALSE,
   kBackSmoothing3,kFALSE);
   for (i = 0; i < nbins; i++) d1->SetBinContent(i + 1,source[i]);
   d1->SetLineColor(kRed);
   d1->Draw("SAME L");
   for (i = 0; i < nbins; i++) source[i]=h->GetBinContent(i + 1);
   s->Background(source,nbins,20,kBackDecreasingWindow,kBackOrder2,kFALSE,
   kBackSmoothing3,kFALSE);
   for (i = 0; i < nbins; i++) d2->SetBinContent(i + 1,source[i]);
   d2->SetLineColor(kBlue);
   d2->Draw("SAME L");
   for (i = 0; i < nbins; i++) source[i]=h->GetBinContent(i + 1);
   s->Background(source,nbins,30,kBackDecreasingWindow,kBackOrder2,kFALSE,
   kBackSmoothing3,kFALSE);
   for (i = 0; i < nbins; i++) d3->SetBinContent(i + 1,source[i]);
   d3->SetLineColor(kGreen);
   d3->Draw("SAME L");
   for (i = 0; i < nbins; i++) source[i]=h->GetBinContent(i + 1);
   s->Background(source,nbins,10,kBackDecreasingWindow,kBackOrder2,kFALSE,
   kBackSmoothing3,kFALSE);
   for (i = 0; i < nbins; i++) d4->SetBinContent(i + 1,source[i]);
   d4->SetLineColor(kMagenta);
   d4->Draw("SAME L");
}
</pre>

Example 5 script Background_order.c:
<p>
Second order difference filter removes linear (quasi-linear) background and
preserves symmetrical peaks. However if the shape of the background is more
complex one can employ higher-order clipping filters (see example in Figure 5)
<p>
<img width=601 height=407 src="gif/TSpectrum_Background_order.jpg">
<p>
Figure 5 Example of the influence of clipping filter difference order on the
estimated background for fNnumberIterations=40, 2-nd order red line, 4-th order
blue line, 6-th order green line and 8-th order magenta line, and using
decreasing clipping window algorithm.
<p>
Script:
<pre>
// Example to illustrate the influence of the clipping filter difference order
// on the estimated background. To execute this example, do
// root > .x Background_order.C

#include <TSpectrum>

void Background_order() {
   Int_t i;
   Double_t nbins = 4096;
   Double_t xmin  = 0;
   Double_t xmax  = (Double_t)4096;
   Float_t * source = new float[nbins];
   TH1F *h = new TH1F("h","",nbins,xmin,xmax);
   TH1F *d1 = new TH1F("d1","",nbins,xmin,xmax);
   TH1F *d2 = new TH1F("d2","",nbins,xmin,xmax);
   TH1F *d3 = new TH1F("d3","",nbins,xmin,xmax);
   TH1F *d4 = new TH1F("d4","",nbins,xmin,xmax);
   TFile *f = new TFile("spectra\\TSpectrum.root");
   h=(TH1F*) f->Get("back2;1");
   TCanvas *background = gROOT->GetListOfCanvases()->FindObject("background");
   if (!background) background = new TCanvas("background",
   "Influence of clipping filter difference order on the estimated background",
   10,10,1000,700);
   h->SetAxisRange(1220,1460);
   h->SetMaximum(11000);
   h->Draw("L");
   TSpectrum *s = new TSpectrum();
   for (i = 0; i < nbins; i++) source[i]=h->GetBinContent(i + 1);
   s->Background(source,nbins,40,kBackDecreasingWindow,kBackOrder2,kFALSE,
   kBackSmoothing3,kFALSE);
   for (i = 0; i < nbins; i++) d1->SetBinContent(i + 1,source[i]);
   d1->SetLineColor(kRed);
   d1->Draw("SAME L");
   for (i = 0; i < nbins; i++) source[i]=h->GetBinContent(i + 1);
   s->Background(source,nbins,40,kBackDecreasingWindow,kBackOrder4,kFALSE,
   kBackSmoothing3,kFALSE);
   for (i = 0; i < nbins; i++) d2->SetBinContent(i + 1,source[i]);
   d2->SetLineColor(kBlue);
   d2->Draw("SAME L");
   for (i = 0; i < nbins; i++) source[i]=h->GetBinContent(i + 1);
   s->Background(source,nbins,40,kBackDecreasingWindow,kBackOrder6,kFALSE,
   kBackSmoothing3,kFALSE);
   for (i = 0; i < nbins; i++) d3->SetBinContent(i + 1,source[i]);
   d3->SetLineColor(kGreen);
   d3->Draw("SAME L");
   for (i = 0; i < nbins; i++) source[i]=h->GetBinContent(i + 1);
   s->Background(source,nbins,40,kBackDecreasingWindow,kBackOrder8,kFALSE,
   kBackSmoothing3,kFALSE);
   for (i = 0; i < nbins; i++) d4->SetBinContent(i + 1,source[i]);
   d4->SetLineColor(kMagenta);
   d4->Draw("SAME L");
}
</pre>

Example 6 script Background_smooth.c:
<p>
The estimate of the background can be influenced by noise present in the
spectrum.  We proposed the algorithm of the background estimate with
simultaneous smoothing.  In the original algorithm without smoothing, the
estimated background snatches the lower spikes in the noise. Consequently,
the areas of peaks are biased by this error.
<p>
<img width=554 height=104 src="gif/TSpectrum_Background_smooth1.jpg">
<p>
Figure 7 Principle of background estimation algorithm with simultaneous
smoothing.
<p>
<img width=601 height=407 src="gif/TSpectrum_Background_smooth2.jpg">
<p>
Figure 8 Illustration of non-smoothing (red line) and smoothing algorithm of
background estimation (blue line).

<p>

Script:

<pre>
// Example to illustrate the background estimator (class TSpectrum) including
// Compton edges. To execute this example, do:
// root > .x Background_smooth.C

#include <TSpectrum>

void Background_smooth() {
   Int_t i;
   Double_t nbins = 4096;
   Double_t xmin  = 0;
   Double_t xmax  = (Double_t)nbins;
   Float_t * source = new float[nbins];
   TH1F *h = new TH1F("h","",nbins,xmin,xmax);
   TH1F *d1 = new TH1F("d1","",nbins,xmin,xmax);
   TH1F *d2 = new TH1F("d2","",nbins,xmin,xmax);
   TFile *f = new TFile("spectra\\TSpectrum.root");
   h=(TH1F*) f->Get("back4;1");
   TCanvas *background = gROOT->GetListOfCanvases()->FindObject("background");
   if (!background) background = new TCanvas("background",
   "Estimation of background with noise",10,10,1000,700);
   h->SetAxisRange(3460,3830);
   h->Draw("L");
   TSpectrum *s = new TSpectrum();
   for (i = 0; i < nbins; i++) source[i]=h->GetBinContent(i + 1);
   s->Background(source,nbins,6,kBackDecreasingWindow,kBackOrder2,kFALSE,
   kBackSmoothing3,kFALSE);
   for (i = 0; i < nbins; i++) d1->SetBinContent(i + 1,source[i]);
   d1->SetLineColor(kRed);
   d1->Draw("SAME L");
   for (i = 0; i < nbins; i++) source[i]=h->GetBinContent(i + 1);
   s->Background(source,nbins,6,kBackDecreasingWindow,kBackOrder2,kTRUE,
   kBackSmoothing3,kFALSE);
   for (i = 0; i < nbins; i++) d2->SetBinContent(i + 1,source[i]);
   d2->SetLineColor(kBlue);
   d2->Draw("SAME L");
}
</pre>

Example 8 script Background_compton.c:
<p>
Sometimes it is necessary to include also the Compton edges into the estimate of
the background. In Figure 8 we present the example of the synthetic spectrum
with Compton edges. The background was estimated using the 8-th order filter
with the estimation of the Compton edges using decreasing
clipping window algorithm (numberIterations=10) with smoothing
(smoothingWindow=5).
<p>
<img width=601 height=407 src="gif/TSpectrum_Background_compton.jpg">
<p>
Figure 8 Example of the estimate of the background with Compton edges (red
line) for numberIterations=10, 8-th order difference filter, using decreasing
clipping window algorithm and smoothing (smoothingWindow=5).
<p>
Script:

<pre>
// Example to illustrate the background estimator (class TSpectrum) including
// Compton edges. To execute this example, do:
// root > .x Background_compton.C

#include <TSpectrum>

void Background_compton() {
   Int_t i;
   Double_t nbins = 512;
   Double_t xmin  = 0;
   Double_t xmax  = (Double_t)nbins;
   Float_t * source = new float[nbins];
   TH1F *h = new TH1F("h","",nbins,xmin,xmax);
   TH1F *d1 = new TH1F("d1","",nbins,xmin,xmax);
   TFile *f = new TFile("spectra\\TSpectrum.root");
   h=(TH1F*) f->Get("back3;1");
   TCanvas *background = gROOT->GetListOfCanvases()->FindObject("background");
   if (!background) background = new TCanvas("background",
   "Estimation of background with Compton edges under peaks",10,10,1000,700);
   h->Draw("L");
   TSpectrum *s = new TSpectrum();
   for (i = 0; i < nbins; i++) source[i]=h->GetBinContent(i + 1);
   s->Background(source,nbins,10,kBackDecreasingWindow,kBackOrder8,kTRUE,
   kBackSmoothing5,,kTRUE);
   for (i = 0; i < nbins; i++) d1->SetBinContent(i + 1,source[i]);
   d1->SetLineColor(kRed);
   d1->Draw("SAME L");
}
</pre>

End_Html */

   int i, j, w, bw, b1, b2, priz;
   float a, b, c, d, e, yb1, yb2, ai, av, men, b4, c4, d4, e4, b6, c6, d6, e6, f6, g6, b8, c8, d8, e8, f8, g8, h8, i8;
   if (ssize <= 0)
      return "Wrong Parameters";
   if (numberIterations < 1)
      return "Width of Clipping Window Must Be Positive";
   if (ssize < 2 * numberIterations + 1)
      return "Too Large Clipping Window";
   if (smoothing == kTRUE && smoothWindow != kBackSmoothing3 && smoothWindow != kBackSmoothing5 && smoothWindow != kBackSmoothing7 && smoothWindow != kBackSmoothing9 && smoothWindow != kBackSmoothing11 && smoothWindow != kBackSmoothing13 && smoothWindow != kBackSmoothing15)
      return "Incorrect width of smoothing window";
   float *working_space = new float[2 * ssize];
   for (i = 0; i < ssize; i++){
      working_space[i] = spectrum[i];
      working_space[i + ssize] = spectrum[i];
   }
   bw=(smoothWindow-1)/2;
   if (direction == kBackIncreasingWindow)
      i = 1;
   else if(direction == kBackDecreasingWindow)
      i = numberIterations;
   if (filterOrder == kBackOrder2) {
      do{
         for (j = i; j < ssize - i; j++) {
            if (smoothing == kFALSE){
               a = working_space[ssize + j];
               b = (working_space[ssize + j - i] + working_space[ssize + j + i]) / 2.0;
               if (b < a)
                  a = b;
               working_space[j] = a;
            }

            else if (smoothing == kTRUE){
               a = working_space[ssize + j];
               av = 0;
               men = 0;
               for (w = j - bw; w <= j + bw; w++){
                  if ( w >= 0 && w < ssize){
                     av += working_space[ssize + w];
                     men +=1;
                  }
               }
               av = av / men;
               b = 0;
               men = 0;
               for (w = j - i - bw; w <= j - i + bw; w++){
                  if ( w >= 0 && w < ssize){
                     b += working_space[ssize + w];
                     men +=1;
                  }
               }
               b = b / men;
               c = 0;
               men = 0;
               for (w = j + i - bw; w <= j + i + bw; w++){
                  if ( w >= 0 && w < ssize){
                     c += working_space[ssize + w];
                     men +=1;
                  }
               }
               c = c / men;
               b = (b + c) / 2;
               if (b < a)
                  av = b;
               working_space[j]=av;
            }
         }
         for (j = i; j < ssize - i; j++)
            working_space[ssize + j] = working_space[j];
         if (direction == kBackIncreasingWindow)
            i+=1;
         else if(direction == kBackDecreasingWindow)
            i-=1;
      }while((direction == kBackIncreasingWindow && i <= numberIterations) || (direction == kBackDecreasingWindow && i >= 1));
   }

   else if (filterOrder == kBackOrder4) {
      do{
         for (j = i; j < ssize - i; j++) {
            if (smoothing == kFALSE){
               a = working_space[ssize + j];
               b = (working_space[ssize + j - i] + working_space[ssize + j + i]) / 2.0;
               c = 0;
               ai = i / 2;
               c -= working_space[ssize + j - (int) (2 * ai)] / 6;
               c += 4 * working_space[ssize + j - (int) ai] / 6;
               c += 4 * working_space[ssize + j + (int) ai] / 6;
               c -= working_space[ssize + j + (int) (2 * ai)] / 6;
               if (b < c)
                  b = c;
               if (b < a)
                  a = b;
               working_space[j] = a;
            }

            else if (smoothing == kTRUE){
               a = working_space[ssize + j];
               av = 0;
               men = 0;
               for (w = j - bw; w <= j + bw; w++){
                  if ( w >= 0 && w < ssize){
                     av += working_space[ssize + w];
                     men +=1;
                  }
               }
               av = av / men;
               b = 0;
               men = 0;
               for (w = j - i - bw; w <= j - i + bw; w++){
                  if ( w >= 0 && w < ssize){
                     b += working_space[ssize + w];
                     men +=1;
                  }
               }
               b = b / men;
               c = 0;
               men = 0;
               for (w = j + i - bw; w <= j + i + bw; w++){
                  if ( w >= 0 && w < ssize){
                     c += working_space[ssize + w];
                     men +=1;
                  }
               }
               c = c / men;
               b = (b + c) / 2;
               ai = i / 2;
               b4 = 0, men = 0;
               for (w = j - (int)(2 * ai) - bw; w <= j - (int)(2 * ai) + bw; w++){
                  if (w >= 0 && w < ssize){
                     b4 += working_space[ssize + w];
                     men +=1;
                  }
               }
               b4 = b4 / men;
               c4 = 0, men = 0;
               for (w = j - (int)ai - bw; w <= j - (int)ai + bw; w++){
                  if (w >= 0 && w < ssize){
                     c4 += working_space[ssize + w];
                     men +=1;
                  }
               }
               c4 = c4 / men;
               d4 = 0, men = 0;
               for (w = j + (int)ai - bw; w <= j + (int)ai + bw; w++){
                  if (w >= 0 && w < ssize){
                     d4 += working_space[ssize + w];
                     men +=1;
                  }
               }
               d4 = d4 / men;
               e4 = 0, men = 0;
               for (w = j + (int)(2 * ai) - bw; w <= j + (int)(2 * ai) + bw; w++){
                  if (w >= 0 && w < ssize){
                     e4 += working_space[ssize + w];
                     men +=1;
                  }
               }
               e4 = e4 / men;
               b4 = (-b4 + 4 * c4 + 4 * d4 - e4) / 6;
               if (b < b4)
                  b = b4;
               if (b < a)
                  av = b;
               working_space[j]=av;
            }
         }
         for (j = i; j < ssize - i; j++)
            working_space[ssize + j] = working_space[j];
         if (direction == kBackIncreasingWindow)
            i+=1;
         else if(direction == kBackDecreasingWindow)
            i-=1;
      }while((direction == kBackIncreasingWindow && i <= numberIterations) || (direction == kBackDecreasingWindow && i >= 1));
   }

   else if (filterOrder == kBackOrder6) {
      do{
         for (j = i; j < ssize - i; j++) {
            if (smoothing == kFALSE){
               a = working_space[ssize + j];
               b = (working_space[ssize + j - i] + working_space[ssize + j + i]) / 2.0;
               c = 0;
               ai = i / 2;
               c -= working_space[ssize + j - (int) (2 * ai)] / 6;
               c += 4 * working_space[ssize + j - (int) ai] / 6;
               c += 4 * working_space[ssize + j + (int) ai] / 6;
               c -= working_space[ssize + j + (int) (2 * ai)] / 6;
               d = 0;
               ai = i / 3;
               d += working_space[ssize + j - (int) (3 * ai)] / 20;
               d -= 6 * working_space[ssize + j - (int) (2 * ai)] / 20;
               d += 15 * working_space[ssize + j - (int) ai] / 20;
               d += 15 * working_space[ssize + j + (int) ai] / 20;
               d -= 6 * working_space[ssize + j + (int) (2 * ai)] / 20;
               d += working_space[ssize + j + (int) (3 * ai)] / 20;
               if (b < d)
                  b = d;
               if (b < c)
                  b = c;
               if (b < a)
                  a = b;
               working_space[j] = a;
            }

            else if (smoothing == kTRUE){
               a = working_space[ssize + j];
               av = 0;
               men = 0;
               for (w = j - bw; w <= j + bw; w++){
                  if ( w >= 0 && w < ssize){
                     av += working_space[ssize + w];
                     men +=1;
                  }
               }
               av = av / men;
               b = 0;
               men = 0;
               for (w = j - i - bw; w <= j - i + bw; w++){
                  if ( w >= 0 && w < ssize){
                     b += working_space[ssize + w];
                     men +=1;
                  }
               }
               b = b / men;
               c = 0;
               men = 0;
               for (w = j + i - bw; w <= j + i + bw; w++){
                  if ( w >= 0 && w < ssize){
                     c += working_space[ssize + w];
                     men +=1;
                  }
               }
               c = c / men;
               b = (b + c) / 2;
               ai = i / 2;
               b4 = 0, men = 0;
               for (w = j - (int)(2 * ai) - bw; w <= j - (int)(2 * ai) + bw; w++){
                  if (w >= 0 && w < ssize){
                     b4 += working_space[ssize + w];
                     men +=1;
                  }
               }
               b4 = b4 / men;
               c4 = 0, men = 0;
               for (w = j - (int)ai - bw; w <= j - (int)ai + bw; w++){
                  if (w >= 0 && w < ssize){
                     c4 += working_space[ssize + w];
                     men +=1;
                  }
               }
               c4 = c4 / men;
               d4 = 0, men = 0;
               for (w = j + (int)ai - bw; w <= j + (int)ai + bw; w++){
                  if (w >= 0 && w < ssize){
                     d4 += working_space[ssize + w];
                     men +=1;
                  }
               }
               d4 = d4 / men;
               e4 = 0, men = 0;
               for (w = j + (int)(2 * ai) - bw; w <= j + (int)(2 * ai) + bw; w++){
                  if (w >= 0 && w < ssize){
                     e4 += working_space[ssize + w];
                     men +=1;
                  }
               }
               e4 = e4 / men;
               b4 = (-b4 + 4 * c4 + 4 * d4 - e4) / 6;
               ai = i / 3;
               b6 = 0, men = 0;
               for (w = j - (int)(3 * ai) - bw; w <= j - (int)(3 * ai) + bw; w++){
                  if (w >= 0 && w < ssize){
                     b6 += working_space[ssize + w];
                     men +=1;
                  }
               }
               b6 = b6 / men;
               c6 = 0, men = 0;
               for (w = j - (int)(2 * ai) - bw; w <= j - (int)(2 * ai) + bw; w++){
                  if (w >= 0 && w < ssize){
                     c6 += working_space[ssize + w];
                     men +=1;
                  }
               }
               c6 = c6 / men;
               d6 = 0, men = 0;
               for (w = j - (int)ai - bw; w <= j - (int)ai + bw; w++){
                  if (w >= 0 && w < ssize){
                     d6 += working_space[ssize + w];
                     men +=1;
                  }
               }
               d6 = d6 / men;
               e6 = 0, men = 0;
               for (w = j + (int)ai - bw; w <= j + (int)ai + bw; w++){
                  if (w >= 0 && w < ssize){
                     e6 += working_space[ssize + w];
                     men +=1;
                  }
               }
               e6 = e6 / men;
               f6 = 0, men = 0;
               for (w = j + (int)(2 * ai) - bw; w <= j + (int)(2 * ai) + bw; w++){
                  if (w >= 0 && w < ssize){
                     f6 += working_space[ssize + w];
                     men +=1;
                  }
               }
               f6 = f6 / men;
               g6 = 0, men = 0;
               for (w = j + (int)(3 * ai) - bw; w <= j + (int)(3 * ai) + bw; w++){
                  if (w >= 0 && w < ssize){
                     g6 += working_space[ssize + w];
                     men +=1;
                  }
               }
               g6 = g6 / men;
               b6 = (b6 - 6 * c6 + 15 * d6 + 15 * e6 - 6 * f6 + g6) / 20;
               if (b < b6)
                  b = b6;
               if (b < b4)
                  b = b4;
               if (b < a)
                  av = b;
               working_space[j]=av;
            }
         }
         for (j = i; j < ssize - i; j++)
            working_space[ssize + j] = working_space[j];
         if (direction == kBackIncreasingWindow)
            i+=1;
         else if(direction == kBackDecreasingWindow)
            i-=1;
      }while((direction == kBackIncreasingWindow && i <= numberIterations) || (direction == kBackDecreasingWindow && i >= 1));
   }

   else if (filterOrder == kBackOrder8) {
      do{
         for (j = i; j < ssize - i; j++) {
            if (smoothing == kFALSE){
               a = working_space[ssize + j];
               b = (working_space[ssize + j - i] + working_space[ssize + j + i]) / 2.0;
               c = 0;
               ai = i / 2;
               c -= working_space[ssize + j - (int) (2 * ai)] / 6;
               c += 4 * working_space[ssize + j - (int) ai] / 6;
               c += 4 * working_space[ssize + j + (int) ai] / 6;
               c -= working_space[ssize + j + (int) (2 * ai)] / 6;
               d = 0;
               ai = i / 3;
               d += working_space[ssize + j - (int) (3 * ai)] / 20;
               d -= 6 * working_space[ssize + j - (int) (2 * ai)] / 20;
               d += 15 * working_space[ssize + j - (int) ai] / 20;
               d += 15 * working_space[ssize + j + (int) ai] / 20;
               d -= 6 * working_space[ssize + j + (int) (2 * ai)] / 20;
               d += working_space[ssize + j + (int) (3 * ai)] / 20;
               e = 0;
               ai = i / 4;
               e -= working_space[ssize + j - (int) (4 * ai)] / 70;
               e += 8 * working_space[ssize + j - (int) (3 * ai)] / 70;
               e -= 28 * working_space[ssize + j - (int) (2 * ai)] / 70;
               e += 56 * working_space[ssize + j - (int) ai] / 70;
               e += 56 * working_space[ssize + j + (int) ai] / 70;
               e -= 28 * working_space[ssize + j + (int) (2 * ai)] / 70;
               e += 8 * working_space[ssize + j + (int) (3 * ai)] / 70;
               e -= working_space[ssize + j + (int) (4 * ai)] / 70;
               if (b < e)
                  b = e;
               if (b < d)
                  b = d;
               if (b < c)
                  b = c;
               if (b < a)
                  a = b;
               working_space[j] = a;
            }

            else if (smoothing == kTRUE){
               a = working_space[ssize + j];
               av = 0;
               men = 0;
               for (w = j - bw; w <= j + bw; w++){
                  if ( w >= 0 && w < ssize){
                     av += working_space[ssize + w];
                     men +=1;
                  }
               }
               av = av / men;
               b = 0;
               men = 0;
               for (w = j - i - bw; w <= j - i + bw; w++){
                  if ( w >= 0 && w < ssize){
                     b += working_space[ssize + w];
                     men +=1;
                  }
               }
               b = b / men;
               c = 0;
               men = 0;
               for (w = j + i - bw; w <= j + i + bw; w++){
                  if ( w >= 0 && w < ssize){
                     c += working_space[ssize + w];
                     men +=1;
                  }
               }
               c = c / men;
               b = (b + c) / 2;
               ai = i / 2;
               b4 = 0, men = 0;
               for (w = j - (int)(2 * ai) - bw; w <= j - (int)(2 * ai) + bw; w++){
                  if (w >= 0 && w < ssize){
                     b4 += working_space[ssize + w];
                     men +=1;
                  }
               }
               b4 = b4 / men;
               c4 = 0, men = 0;
               for (w = j - (int)ai - bw; w <= j - (int)ai + bw; w++){
                  if (w >= 0 && w < ssize){
                     c4 += working_space[ssize + w];
                     men +=1;
                  }
               }
               c4 = c4 / men;
               d4 = 0, men = 0;
               for (w = j + (int)ai - bw; w <= j + (int)ai + bw; w++){
                  if (w >= 0 && w < ssize){
                     d4 += working_space[ssize + w];
                     men +=1;
                  }
               }
               d4 = d4 / men;
               e4 = 0, men = 0;
               for (w = j + (int)(2 * ai) - bw; w <= j + (int)(2 * ai) + bw; w++){
                  if (w >= 0 && w < ssize){
                     e4 += working_space[ssize + w];
                     men +=1;
                  }
               }
               e4 = e4 / men;
               b4 = (-b4 + 4 * c4 + 4 * d4 - e4) / 6;
               ai = i / 3;
               b6 = 0, men = 0;
               for (w = j - (int)(3 * ai) - bw; w <= j - (int)(3 * ai) + bw; w++){
                  if (w >= 0 && w < ssize){
                     b6 += working_space[ssize + w];
                     men +=1;
                  }
               }
               b6 = b6 / men;
               c6 = 0, men = 0;
               for (w = j - (int)(2 * ai) - bw; w <= j - (int)(2 * ai) + bw; w++){
                  if (w >= 0 && w < ssize){
                     c6 += working_space[ssize + w];
                     men +=1;
                  }
               }
               c6 = c6 / men;
               d6 = 0, men = 0;
               for (w = j - (int)ai - bw; w <= j - (int)ai + bw; w++){
                  if (w >= 0 && w < ssize){
                     d6 += working_space[ssize + w];
                     men +=1;
                  }
               }
               d6 = d6 / men;
               e6 = 0, men = 0;
               for (w = j + (int)ai - bw; w <= j + (int)ai + bw; w++){
                  if (w >= 0 && w < ssize){
                     e6 += working_space[ssize + w];
                     men +=1;
                  }
               }
               e6 = e6 / men;
               f6 = 0, men = 0;
               for (w = j + (int)(2 * ai) - bw; w <= j + (int)(2 * ai) + bw; w++){
                  if (w >= 0 && w < ssize){
                     f6 += working_space[ssize + w];
                     men +=1;
                  }
               }
               f6 = f6 / men;
               g6 = 0, men = 0;
               for (w = j + (int)(3 * ai) - bw; w <= j + (int)(3 * ai) + bw; w++){
                  if (w >= 0 && w < ssize){
                     g6 += working_space[ssize + w];
                     men +=1;
                  }
               }
               g6 = g6 / men;
               b6 = (b6 - 6 * c6 + 15 * d6 + 15 * e6 - 6 * f6 + g6) / 20;
               ai = i / 4;
               b8 = 0, men = 0;
               for (w = j - (int)(4 * ai) - bw; w <= j - (int)(4 * ai) + bw; w++){
                  if (w >= 0 && w < ssize){
                     b8 += working_space[ssize + w];
                     men +=1;
                  }
               }
               b8 = b8 / men;
               c8 = 0, men = 0;
               for (w = j - (int)(3 * ai) - bw; w <= j - (int)(3 * ai) + bw; w++){
                  if (w >= 0 && w < ssize){
                     c8 += working_space[ssize + w];
                     men +=1;
                  }
               }
               c8 = c8 / men;
               d8 = 0, men = 0;
               for (w = j - (int)(2 * ai) - bw; w <= j - (int)(2 * ai) + bw; w++){
                  if (w >= 0 && w < ssize){
                     d8 += working_space[ssize + w];
                     men +=1;
                  }
               }
               d8 = d8 / men;
               e8 = 0, men = 0;
               for (w = j - (int)ai - bw; w <= j - (int)ai + bw; w++){
                  if (w >= 0 && w < ssize){
                     e8 += working_space[ssize + w];
                     men +=1;
                  }
               }
               e8 = e8 / men;
               f8 = 0, men = 0;
               for (w = j + (int)ai - bw; w <= j + (int)ai + bw; w++){
                  if (w >= 0 && w < ssize){
                     f8 += working_space[ssize + w];
                     men +=1;
                  }
               }
               f8 = f8 / men;
               g8 = 0, men = 0;
               for (w = j + (int)(2 * ai) - bw; w <= j + (int)(2 * ai) + bw; w++){
                  if (w >= 0 && w < ssize){
                     g8 += working_space[ssize + w];
                     men +=1;
                  }
               }
               g8 = g8 / men;
               h8 = 0, men = 0;
               for (w = j + (int)(3 * ai) - bw; w <= j + (int)(3 * ai) + bw; w++){
                  if (w >= 0 && w < ssize){
                     h8 += working_space[ssize + w];
                     men +=1;
                  }
               }
               h8 = h8 / men;
               i8 = 0, men = 0;
               for (w = j + (int)(4 * ai) - bw; w <= j + (int)(4 * ai) + bw; w++){
                  if (w >= 0 && w < ssize){
                     i8 += working_space[ssize + w];
                     men +=1;
                  }
               }
               i8 = i8 / men;
               b8 = ( -b8 + 8 * c8 - 28 * d8 + 56 * e8 - 56 * f8 - 28 * g8 + 8 * h8 - i8)/70;
               if (b < b8)
                  b = b8;
               if (b < b6)
                  b = b6;
               if (b < b4)
                  b = b4;
               if (b < a)
                  av = b;
               working_space[j]=av;
            }
         }
         for (j = i; j < ssize - i; j++)
            working_space[ssize + j] = working_space[j];
         if (direction == kBackIncreasingWindow)
            i += 1;
         else if(direction == kBackDecreasingWindow)
            i -= 1;
      }while((direction == kBackIncreasingWindow && i <= numberIterations) || (direction == kBackDecreasingWindow && i >= 1));
   }

   if (compton == kTRUE) {
      for (i = 0, b2 = 0; i < ssize; i++){
         b1 = b2;
         a = working_space[i], b = spectrum[i];
         j = i;
         if (TMath::Abs(a - b) >= 1) {
            b1 = i - 1;
            if (b1 < 0)
               b1 = 0;
            yb1 = working_space[b1];
            for (b2 = b1 + 1, c = 0, priz = 0; priz == 0 && b2 < ssize; b2++){
               a = working_space[b2], b = spectrum[b2];
               c = c + b - yb1;
               if (TMath::Abs(a - b) < 1) {
                  priz = 1;
                  yb2 = b;
               }
            }
            if (b2 == ssize)
               b2 -= 1;
            yb2 = working_space[b2];
            if (yb1 <= yb2){
               for (j = b1, c = 0; j <= b2; j++){
                  b = spectrum[j];
                  c = c + b - yb1;
               }
               if (c > 1){
                  c = (yb2 - yb1) / c;
                  for (j = b1, d = 0; j <= b2 && j < ssize; j++){
                     b = spectrum[j];
                     d = d + b - yb1;
                     a = c * d + yb1;
                     working_space[ssize + j] = a;
                  }
               }
            }

            else{
               for (j = b2, c = 0; j >= b1; j--){
                  b = spectrum[j];
                  c = c + b - yb2;
               }
               if (c > 1){
                  c = (yb1 - yb2) / c;
                  for (j = b2, d = 0;j >= b1 && j >= 0; j--){
                     b = spectrum[j];
                     d = d + b - yb2;
                     a = c * d + yb2;
                     working_space[ssize + j] = a;
                  }
               }
            }
            i=b2;
         }
      }
   }

   for (j = 0; j < ssize; j++)
      spectrum[j] = working_space[ssize + j];
   delete[]working_space;
   return 0;
}


//______________________________________________________________________________
const char* TSpectrum::SmoothMarkov(float *source, int ssize, int averWindow)
{
   /* Begin_Html
   <b>One-dimensional markov spectrum smoothing function</b>
   <p>
   This function calculates smoothed spectrum from source spectrum based on
   Markov chain method. The result is placed in the array pointed by source
   pointer. On successful completion it returns 0. On error it returns pointer
   to the string describing error.
   <p>
   Function parameters:
   <ul>
   <li> source: pointer to the array of source spectrum
   <li> ssize: length of source array
   <li> averWindow: width of averaging smoothing window
   </ul>
   The goal of this function is the suppression of the statistical fluctuations.
   The algorithm is based on discrete Markov chain, which has very simple
   invariant distribution:
   <img width=551 height=63 src="gif/TSpectrum_Smoothing1.gif">
   <p>
   <img width=28 height=36 src="gif/TSpectrum_Smoothing2.gif"> being defined
   from the normalization condition
   <img width=70 height=52 src="gif/TSpectrum_Smoothing3.gif">.
   n is the length of the smoothed spectrum and
   <img width=258 height=60 src="gif/TSpectrum_Smoothing4.gif">
   <p>
   Reference:
   <ol>
   <li> Z.K. Silagadze, A new algorithm for automatic photopeak searches.
   NIM A 376 (1996), 451.
   </ol>
   <p>
   Example 14 - script Smoothing.c
   <p>
   <img width=296 height=182 src="gif/TSpectrum_Smoothing1.jpg">
   Fig. 23 Original noisy spectrum
   <p>
   <img width=296 height=182 src="gif/TSpectrum_Smoothing2.jpg">
   Fig. 24 Smoothed spectrum m=3
   <p>
   <img width=299 height=184 src="gif/TSpectrum_Smoothing3.jpg">
   Fig. 25 Smoothed spectrum
   <p>
   <img width=299 height=184 src="gif/TSpectrum_Smoothing4.jpg">
   Fig.26 Smoothed spectrum m=10
   <p>
   Script:
   <pre>
   // Example to illustrate smoothing using Markov algorithm (class TSpectrum).
   // To execute this example, do
   // root > .x Smoothing.C

   void Smoothing() {
      Int_t i;
      Double_t nbins = 1024;
      Double_t xmin  = 0;
      Double_t xmax  = (Double_t)nbins;
      Float_t * source = new float[nbins];
      TH1F *h = new TH1F("h","Smoothed spectrum for m=3",nbins,xmin,xmax);
      TFile *f = new TFile("spectra\\TSpectrum.root");
      h=(TH1F*) f->Get("smooth1;1");
      for (i = 0; i < nbins; i++) source[i]=h->GetBinContent(i + 1);
      TCanvas *Smooth1 = gROOT->GetListOfCanvases()->FindObject("Smooth1");
      if (!Smooth1) Smooth1 = new TCanvas("Smooth1","Smooth1",10,10,1000,700);
      TSpectrum *s = new TSpectrum();
      s->SmoothMarkov(source,1024,3);  //3, 7, 10
      for (i = 0; i < nbins; i++) h->SetBinContent(i + 1,source[i]);
      h->SetAxisRange(330,880);
      h->Draw("L");
   }
   </pre>
   End_Html */

   int xmin, xmax, i, l;
   float a, b, maxch;
   float nom, nip, nim, sp, sm, area = 0;
   if(averWindow <= 0)
      return "Averaging Window must be positive";
   float *working_space = new float[ssize];
   xmin = 0,xmax = ssize - 1;
   for(i = 0, maxch = 0; i < ssize; i++){
      working_space[i]=0;
      if(maxch < source[i])
         maxch = source[i];

      area += source[i];
   }
   if(maxch == 0) {
      delete [] working_space;
      return 0 ;
   }

   nom = 1;
   working_space[xmin] = 1;
   for(i = xmin; i < xmax; i++){
      nip = source[i] / maxch;
      nim = source[i + 1] / maxch;
      sp = 0,sm = 0;
      for(l = 1; l <= averWindow; l++){
         if((i + l) > xmax)
            a = source[xmax] / maxch;

         else
            a = source[i + l] / maxch;
         b = a - nip;
         if(a + nip <= 0)
            a = 1;

         else
            a = TMath::Sqrt(a + nip);
         b = b / a;
         b = TMath::Exp(b);
         sp = sp + b;
         if((i - l + 1) < xmin)
            a = source[xmin] / maxch;

         else
            a = source[i - l + 1] / maxch;
         b = a - nim;
         if(a + nim <= 0)
            a = 1;
         else
            a = TMath::Sqrt(a + nim);
         b = b / a;
         b = TMath::Exp(b);
         sm = sm + b;
      }
      a = sp / sm;
      a = working_space[i + 1] = working_space[i] * a;
      nom = nom + a;
   }
   for(i = xmin; i <= xmax; i++){
      working_space[i] = working_space[i] / nom;
   }
   for(i = 0; i < ssize; i++)
      source[i] = working_space[i] * area;
   delete[]working_space;
   return 0;
}


//______________________________________________________________________________
const char *TSpectrum::Deconvolution(float *source, const float *response,
                                      int ssize, int numberIterations,
                                      int numberRepetitions, double boost )
{
   /* Begin_Html
   <b>One-dimensional deconvolution function</b>
   <p>
   This function calculates deconvolution from source spectrum according to
   response spectrum using Gold deconvolution algorithm. The result is placed
   in the vector pointed by source pointer. On successful completion it
   returns 0. On error it returns pointer to the string describing error. If
   desired after every numberIterations one can apply boosting operation
   (exponential function with exponent given by boost coefficient) and repeat
   it numberRepetitions times.
   <p>
   Function parameters:
   <ul>
   <li>source:  pointer to the vector of source spectrum
   <li>response:     pointer to the vector of response spectrum
   <li>ssize:    length of source and response spectra
   numberIterations, for details we refer to the reference given below
   numberRepetitions, for repeated boosted deconvolution
   boost, boosting coefficient
   </ul>
   The goal of this function is the improvement of the resolution in spectra,
   decomposition of multiplets. The mathematical formulation of
   the convolution system is:
   <p>
   <img width=585 height=84 src="gif/TSpectrum_Deconvolution1.gif">
   <p>
   where h(i) is the impulse response function, x, y are input and output
   vectors, respectively, N is the length of x and h vectors. In matrix form
   we have:
   <p>
   <img width=597 height=360 src="gif/TSpectrum_Deconvolution2.gif">
   <p>
   Let us assume that we know the response and the output vector (spectrum) of
   the above given system. The deconvolution represents solution of the
   overdetermined system of linear equations, i.e., the calculation of the
   vector <b>x</b>. From numerical stability point of view the operation of
   deconvolution is extremely critical (ill-posed problem) as well as time
   consuming operation. The Gold deconvolution algorithm proves to work very
   well, other methods (Fourier, VanCittert etc) oscillate. It is suitable to
   process positive definite data (e.g. histograms).
   <p>
   <b>Gold deconvolution algorithm:</b>
   <p>
   <img width=551 height=233 src="gif/TSpectrum_Deconvolution3.gif">
   <p>
   Where L is given number of iterations (numberIterations parameter).
   <p>
   <b>Boosted deconvolution:</b>
   <ol>
   <li> Set the initial solution:
        End_Html Begin_Latex x^{(0)} = [1,1,...,1]^{T} End_Latex Begin_Html
   <li> Set required number of repetitions R and iterations L.
   <li> Set r = 1.
   <li>Using Gold deconvolution algorithm for k=1,2,...,L find
       End_Html Begin_Latex x^{(L)} End_Latex Begin_Html
   <li> If r = R stop calculation, else
      <ol>
      <li> Apply boosting operation, i.e., set
           End_Html Begin_Latex x^{(0)}(i) = [x^{(L)}(i)]^{p} End_Latex Begin_Html
           i=0,1,...N-1 and p is boosting coefficient &gt;0.
      <li> r = r + 1
      <li> continue in 4.
      </ol>
   </ol>
   <p>
   <b>References:</b>
   <ol>
   <li> Gold R., ANL-6984, Argonne National Laboratories, Argonne Ill, 1964.
   <li> Coote G.E., Iterative smoothing and deconvolution of one- and two-dimensional
        elemental distribution data, NIM B 130 (1997) 118.
   <li> M. Morhá&#269;, J. Kliman, V.  Matoušek, M. Veselský,
        I. Turzo: Efficient one- and two-dimensional Gold deconvolution and
        its application to gamma-ray spectra decomposition. NIM, A401 (1997) 385-408.
   <li> Morhá&#269; M., Matoušek V., Kliman J., Efficient algorithm of multidimensional
        deconvolution and its application to nuclear data processing, Digital Signal
        Processing 13 (2003) 144.
   </ol>
   <p>
   <i>Example 8 - script Deconvolution.c :</i>
   <p>
   response function (usually peak) should be shifted left to the first
   non-zero channel (bin) (see Figure 9)
   <p>
   <img width=600 height=340 src="gif/TSpectrum_Deconvolution1.jpg">
   <p>
   Figure 9 Response spectrum.
   <p>
   <img width=946 height=407 src="gif/TSpectrum_Deconvolution2.jpg">
   <p>
   Figure 10 Principle how the response matrix is composed inside of the
   Deconvolution function.
   <img width=601 height=407 src="gif/TSpectrum_Deconvolution3.jpg">
   <p>
   Figure 11 Example of Gold deconvolution. The original source spectrum is
   drawn with black color, the spectrum after the deconvolution (10000
   iterations) with red color.
   <p>
   Script:
   <p>
   <pre>
   // Example to illustrate deconvolution function (class TSpectrum).
   // To execute this example, do
   // root > .x Deconvolution.C

   #include <TSpectrum>

   void Deconvolution() {
      Int_t i;
      Double_t nbins = 256;
      Double_t xmin  = 0;
      Double_t xmax  = (Double_t)nbins;
      Float_t * source = new float[nbins];
      Float_t * response = new float[nbins];
      TH1F *h = new TH1F("h","Deconvolution",nbins,xmin,xmax);
      TH1F *d = new TH1F("d","",nbins,xmin,xmax);
      TFile *f = new TFile("spectra\\TSpectrum.root");
      h=(TH1F*) f->Get("decon1;1");
      TFile *fr = new TFile("spectra\\TSpectrum.root");
      d=(TH1F*) fr->Get("decon_response;1");
      for (i = 0; i < nbins; i++) source[i]=h->GetBinContent(i + 1);
      for (i = 0; i < nbins; i++) response[i]=d->GetBinContent(i + 1);
      TCanvas *Decon1 = gROOT->GetListOfCanvases()->FindObject("Decon1");
      if (!Decon1) Decon1 = new TCanvas("Decon1","Decon1",10,10,1000,700);
      h->Draw("L");
      TSpectrum *s = new TSpectrum();
      s->Deconvolution(source,response,256,1000,1,1);
      for (i = 0; i < nbins; i++) d->SetBinContent(i + 1,source[i]);
      d->SetLineColor(kRed);
      d->Draw("SAME L");
   }
   </pre>
   <p>
   <b>Examples of Gold deconvolution method:</b>
   <p>
   First let us study the influence of the number of iterations on the
   deconvolved spectrum (Figure 12).
   <p>
   <img width=602 height=409 src="gif/TSpectrum_Deconvolution_wide1.jpg">
   <p>
   Figure 12 Study of Gold deconvolution algorithm.The original source spectrum
   is drawn with black color, spectrum after 100 iterations with red color,
   spectrum after 1000 iterations with blue color, spectrum after 10000
   iterations with green color and spectrum after 100000 iterations with
   magenta color.
   <p>
   For relatively narrow peaks in the above given example the Gold
   deconvolution method is able to decompose overlapping peaks practically to
   delta - functions. In the next example we have chosen a synthetic data
   (spectrum, 256 channels) consisting of 5 very closely positioned, relatively
   wide peaks (sigma =5), with added noise (Figure 13). Thin lines represent
   pure Gaussians (see Table 1); thick line is a resulting spectrum with
   additive noise (10% of the amplitude of small peaks).
   <p>
   <img width=600 height=367 src="gif/TSpectrum_Deconvolution_wide2.jpg">
   <p>
   Figure 13 Testing example of synthetic spectrum composed of 5 Gaussians with
   added noise.
   <p>
   <table border=solid><tr>
   <td> Peak # </td><td> Position </td><td> Height </td><td> Area   </td>
   </tr><tr>
   <td> 1      </td><td> 50       </td><td> 500    </td><td> 10159  </td>
   </tr><tr>
   <td> 2      </td><td> 70       </td><td> 3000   </td><td> 60957  </td>
   </tr><tr>
   <td> 3      </td><td> 80       </td><td> 1000   </td><td> 20319  </td>
   </tr><tr>
   <td> 4      </td><td> 100      </td><td> 5000   </td><td> 101596 </td>
   </tr><tr>
   <td> 5      </td><td> 110      </td><td> 500    </td><td> 10159  </td>
   </tr></table>
   <p>
   Table 1 Positions, heights and areas of peaks in the spectrum shown in
   Figure 13.
   <p>
   In ideal case, we should obtain the result given in Figure 14. The areas of
   the Gaussian components of the spectrum are concentrated completely to
   delta-functions. When solving the overdetermined system of linear equations
   with data from Figure 13 in the sense of minimum least squares criterion
   without any regularization we obtain the result with large oscillations
   (Figure 15). From mathematical point of view, it is the optimal solution in
   the unconstrained space of independent variables. From physical point of
   view we are interested only in a meaningful solution. Therefore, we have to
   employ regularization techniques (e.g. Gold deconvolution) and/or to
   confine the space of allowed solutions to subspace of positive solutions.
   <p>
   <img width=589 height=189 src="gif/TSpectrum_Deconvolution_wide3.jpg">
   <p>
   Figure 14 The same spectrum like in Figure 13, outlined bars show the
   contents of present components (peaks).
   <img width=585 height=183 src="gif/TSpectrum_Deconvolution_wide4.jpg">
   <p>
   Figure 15 Least squares solution of the system of linear equations without
   regularization.
   <p>
   <i>Example 9 - script Deconvolution_wide.c</i>
   <p>
   When we employ Gold deconvolution algorithm we obtain the result given in
   Fig. 16. One can observe that the resulting spectrum is smooth. On the
   other hand the method is not able to decompose completely the peaks in the
   spectrum.
   <p>
   <img width=601 height=407 src="gif/TSpectrum_Deconvolution_wide5.jpg">
   Figure 16 Example of Gold deconvolution for closely positioned wide peaks.
   The original source spectrum is drawn with black color, the spectrum after
   the deconvolution (10000 iterations) with red color.
   <p>
   Script:
   <p>
   <pre>
   // Example to illustrate deconvolution function (class TSpectrum).
   // To execute this example, do
   // root > .x Deconvolution_wide.C

   #include <TSpectrum>

   void Deconvolution_wide() {
      Int_t i;
      Double_t nbins = 256;
      Double_t xmin  = 0;
      Double_t xmax  = (Double_t)nbins;
      Float_t * source = new float[nbins];
      Float_t * response = new float[nbins];
      TH1F *h = new TH1F("h","Deconvolution",nbins,xmin,xmax);
      TH1F *d = new TH1F("d","",nbins,xmin,xmax);
      TFile *f = new TFile("spectra\\TSpectrum.root");
      h=(TH1F*) f->Get("decon3;1");
      TFile *fr = new TFile("spectra\\TSpectrum.root");
      d=(TH1F*) fr->Get("decon_response_wide;1");
      for (i = 0; i < nbins; i++) source[i]=h->GetBinContent(i + 1);
      for (i = 0; i < nbins; i++) response[i]=d->GetBinContent(i + 1);
      TCanvas *Decon1 = gROOT->GetListOfCanvases()->FindObject("Decon1");
      if (!Decon1) Decon1 = new TCanvas("Decon1",
      "Deconvolution of closely positioned overlapping peaks using Gold deconvolution method",10,10,1000,700);
      h->SetMaximum(30000);
      h->Draw("L");
      TSpectrum *s = new TSpectrum();
      s->Deconvolution(source,response,256,10000,1,1);
      for (i = 0; i < nbins; i++) d->SetBinContent(i + 1,source[i]);
      d->SetLineColor(kRed);
      d->Draw("SAME L");
   }
   </pre>
   <p>
   <i>Example 10 - script Deconvolution_wide_boost.c :</i>
   <p>
   Further let us employ boosting operation into deconvolution (Fig. 17).
   <p>
   <img width=601 height=407 src="gif/TSpectrum_Deconvolution_wide6.jpg">
   <p>
   Figure 17 The original source spectrum is drawn with black color, the
   spectrum after the deconvolution with red color. Number of iterations = 200,
   number of repetitions = 50 and boosting coefficient = 1.2.
   <p>
   <table border=solid><tr>
   <td> Peak # </td> <td> Original/Estimated (max) position </td> <td> Original/Estimated area </td>
   </tr> <tr>
   <td> 1 </td> <td> 50/49 </td> <td> 10159/10419 </td>
   </tr> <tr>
   <td> 2 </td> <td> 70/70 </td> <td> 60957/58933 </td>
   </tr> <tr>
   <td> 3 </td> <td> 80/79 </td> <td> 20319/19935 </td>
   </tr> <tr>
   <td> 4 </td> <td> 100/100 </td> <td> 101596/105413 </td>
   </tr> <tr>
   <td> 5 </td> <td> 110/117 </td> <td> 10159/6676 </td>
   </tr> </table>
   <p>
   Table 2 Results of the estimation of peaks in spectrum shown in Figure 17.
   <p>
   One can observe that peaks are decomposed practically to delta functions.
   Number of peaks is correct, positions of big peaks as well as their areas
   are relatively well estimated. However there is a considerable error in
   the estimation of the position of small right hand peak.
   <p>
   Script:
   <p>
   <pre>
   // Example to illustrate deconvolution function (class TSpectrum).
   // To execute this example, do
   // root > .x Deconvolution_wide_boost.C

   #include <TSpectrum>

   void Deconvolution_wide_boost() {
      Int_t i;
      Double_t nbins = 256;
      Double_t xmin  = 0;
      Double_t xmax  = (Double_t)nbins;
      Float_t * source = new float[nbins];
      Float_t * response = new float[nbins];
      TH1F *h = new TH1F("h","Deconvolution",nbins,xmin,xmax);
      TH1F *d = new TH1F("d","",nbins,xmin,xmax);
      TFile *f = new TFile("spectra\\TSpectrum.root");
      h=(TH1F*) f->Get("decon3;1");
      TFile *fr = new TFile("spectra\\TSpectrum.root");
      d=(TH1F*) fr->Get("decon_response_wide;1");
      for (i = 0; i < nbins; i++) source[i]=h->GetBinContent(i + 1);
      for (i = 0; i < nbins; i++) response[i]=d->GetBinContent(i + 1);
      TCanvas *Decon1 = gROOT->GetListOfCanvases()->FindObject("Decon1");
      if (!Decon1) Decon1 = new TCanvas("Decon1",
      "Deconvolution of closely positioned overlapping peaks using boosted Gold deconvolution method",10,10,1000,700);
      h->SetMaximum(110000);
      h->Draw("L");
      TSpectrum *s = new TSpectrum();
      s->Deconvolution(source,response,256,200,50,1.2);
      for (i = 0; i < nbins; i++) d->SetBinContent(i + 1,source[i]);
      d->SetLineColor(kRed);
      d->Draw("SAME L");
   }
   </pre>
   End_Html */

   if (ssize <= 0)
      return "Wrong Parameters";

   if (numberRepetitions <= 0)
      return "Wrong Parameters";

       //   working_space-pointer to the working vector
       //   (its size must be 4*ssize of source spectrum)
   double *working_space = new double[4 * ssize];
   int i, j, k, lindex, posit, lh_gold, l, repet;
   double lda, ldb, ldc, area, maximum;
   area = 0;
   lh_gold = -1;
   posit = 0;
   maximum = 0;

//read response vector
   for (i = 0; i < ssize; i++) {
      lda = response[i];
      if (lda != 0)
         lh_gold = i + 1;
      working_space[i] = lda;
      area += lda;
      if (lda > maximum) {
         maximum = lda;
         posit = i;
      }
   }
   if (lh_gold == -1) {
      delete [] working_space;
      return "ZERO RESPONSE VECTOR";
   }

//read source vector
   for (i = 0; i < ssize; i++)
      working_space[2 * ssize + i] = source[i];

// create matrix at*a and vector at*y
   for (i = 0; i < ssize; i++){
      lda = 0;
      for (j = 0; j < ssize; j++){
         ldb = working_space[j];
         k = i + j;
         if (k < ssize){
            ldc = working_space[k];
            lda = lda + ldb * ldc;
         }
      }
      working_space[ssize + i] = lda;
      lda = 0;
      for (k = 0; k < ssize; k++){
         l = k - i;
         if (l >= 0){
            ldb = working_space[l];
            ldc = working_space[2 * ssize + k];
            lda = lda + ldb * ldc;
         }
      }
      working_space[3 * ssize + i]=lda;
   }

// move vector at*y
   for (i = 0; i < ssize; i++){
      working_space[2 * ssize + i] = working_space[3 * ssize + i];
   }

//initialization of resulting vector
   for (i = 0; i < ssize; i++)
      working_space[i] = 1;

       //**START OF ITERATIONS**
   for (repet = 0; repet < numberRepetitions; repet++) {
      if (repet != 0) {
         for (i = 0; i < ssize; i++)
            working_space[i] = TMath::Power(working_space[i], boost);
      }
      for (lindex = 0; lindex < numberIterations; lindex++) {
         for (i = 0; i < ssize; i++) {
            if (working_space[2 * ssize + i] > 0.000001
                 && working_space[i] > 0.000001) {
               lda = 0;
               for (j = 0; j < lh_gold; j++) {
                  ldb = working_space[j + ssize];
                  if (j != 0){
                     k = i + j;
                     ldc = 0;
                     if (k < ssize)
                        ldc = working_space[k];
                     k = i - j;
                     if (k >= 0)
                        ldc += working_space[k];
                  }

                  else
                     ldc = working_space[i];
                  lda = lda + ldb * ldc;
               }
               ldb = working_space[2 * ssize + i];
               if (lda != 0)
                  lda = ldb / lda;

               else
                  lda = 0;
               ldb = working_space[i];
               lda = lda * ldb;
               working_space[3 * ssize + i] = lda;
            }
         }
         for (i = 0; i < ssize; i++)
            working_space[i] = working_space[3 * ssize + i];
      }
   }

//shift resulting spectrum
   for (i = 0; i < ssize; i++) {
      lda = working_space[i];
      j = i + posit;
      j = j % ssize;
      working_space[ssize + j] = lda;
   }

//write back resulting spectrum
   for (i = 0; i < ssize; i++)
      source[i] = area * working_space[ssize + i];
   delete[]working_space;
   return 0;
}


//______________________________________________________________________________
const char *TSpectrum::DeconvolutionRL(float *source, const float *response,
                                      int ssize, int numberIterations,
                                      int numberRepetitions, double boost )
{
   /* Begin_Html
   <b>One-dimensional deconvolution function.</b>
   <p>
   This function calculates deconvolution from source spectrum according to
   response spectrum using Richardson-Lucy deconvolution algorithm. The result
   is placed in the vector pointed by source pointer. On successful completion
   it returns 0. On error it returns pointer to the string describing error.
   If desired after every numberIterations one can apply boosting operation
   (exponential function with exponent given by boost coefficient) and repeat
   it numberRepetitions times (see Gold deconvolution).
   <p>
   Function parameters:
   <ul>
   <li> source:  pointer to the vector of source spectrum
   <li> response:     pointer to the vector of response spectrum
   <li> ssize:    length of source and response spectra
   numberIterations, for details we refer to the reference given above
   numberRepetitions, for repeated boosted deconvolution
   boost, boosting coefficient
   </ul>
   <p>
   <b>Richardson-Lucy deconvolution algorithm:</b>
   <p>
   For discrete systems it has the form:
   <p>
   <img width=438 height=98 src="gif/TSpectrum_DeconvolutionRL1.gif">
   <p>
   <img width=124 height=39 src="gif/TSpectrum_DeconvolutionRL2.gif">
   <p>
   for positive input data and response matrix this iterative method forces
   the deconvoluted spectra to be non-negative. The Richardson-Lucy
   iteration converges to the maximum likelihood solution for Poisson statistics
   in the data.
   <p>
   <b>References:</b>
   <ol>
   <li> Abreu M.C. et al., A four-dimensional deconvolution method to correct NA38
   experimental data, NIM A 405 (1998) 139.
   <li> Lucy L.B., A.J. 79 (1974) 745.
   <li> Richardson W.H., J. Opt. Soc. Am. 62 (1972) 55.
   </ol>
   <p>
   <b>Examples of Richardson-Lucy deconvolution method:</b>
   <p>
   <i>Example 11 - script DeconvolutionRL_wide.c :</i>
   <p>
   When we employ Richardson-Lucy deconvolution algorithm to our data from
   Fig. 13 we obtain the result given in Fig. 18. One can observe improvements
   as compared to the result achieved by Gold deconvolution. Neverthless it is
   unable to decompose the multiplet.
   <p>
   <img width=601 height=407 src="gif/TSpectrum_DeconvolutionRL_wide1.jpg">
   Figure 18 Example of Richardson-Lucy deconvolution for closely positioned
   wide peaks. The original source spectrum is drawn with black color, the
   spectrum after the deconvolution (10000 iterations) with red color.
   <p>
   Script:
   <p>
   <pre>
   // Example to illustrate deconvolution function (class TSpectrum).
   // To execute this example, do
   // root > .x DeconvolutionRL_wide.C

   #include <TSpectrum>

   void DeconvolutionRL_wide() {
      Int_t i;
      Double_t nbins = 256;
      Double_t xmin  = 0;
      Double_t xmax  = (Double_t)nbins;
      Float_t * source = new float[nbins];
      Float_t * response = new float[nbins];
      TH1F *h = new TH1F("h","Deconvolution",nbins,xmin,xmax);
      TH1F *d = new TH1F("d","",nbins,xmin,xmax);
      TFile *f = new TFile("spectra\\TSpectrum.root");
      h=(TH1F*) f->Get("decon3;1");
      TFile *fr = new TFile("spectra\\TSpectrum.root");
      d=(TH1F*) fr->Get("decon_response_wide;1");
      for (i = 0; i < nbins; i++) source[i]=h->GetBinContent(i + 1);
      for (i = 0; i < nbins; i++) response[i]=d->GetBinContent(i + 1);
      TCanvas *Decon1 = gROOT->GetListOfCanvases()->FindObject("Decon1");
      if (!Decon1) Decon1 = new TCanvas("Decon1",
      "Deconvolution of closely positioned overlapping peaks using Richardson-Lucy deconvolution method",
      10,10,1000,700);
      h->SetMaximum(30000);
      h->Draw("L");
      TSpectrum *s = new TSpectrum();
      s->DeconvolutionRL(source,response,256,10000,1,1);
      for (i = 0; i < nbins; i++) d->SetBinContent(i + 1,source[i]);
      d->SetLineColor(kRed);
      d->Draw("SAME L");
   }
   </pre>
   <p>
   <i>Example 12 - script DeconvolutionRL_wide_boost.c :</i>
   <p>
   Further let us employ boosting operation into deconvolution (Fig. 19).
   <img width=601 height=407 src="gif/TSpectrum_DeconvolutionRL_wide2.jpg">
   <p>
   Figure 19 The original source spectrum is drawn with black color, the
   spectrum after the deconvolution with red color. Number of iterations = 200,
   number of repetitions = 50 and boosting coefficient = 1.2.
   <p>
   <table border=solid>
   <tr><td> Peak # </td><td> Original/Estimated (max) position </td><td> Original/Estimated area </td></tr>
   <tr><td> 1 </td><td> 50/51 </td><td> 10159/11426 </td></tr>
   <tr><td> 2 </td><td> 70/71 </td><td> 60957/65003 </td></tr>
   <tr><td> 3 </td><td> 80/81 </td><td> 20319/12813 </td></tr>
   <tr><td> 4 </td><td> 100/100 </td><td> 101596/101851 </td></tr>
   <tr><td> 5 </td><td> 110/111 </td><td> 10159/8920 </td></tr>
   </table>
   <p>
   Table 3 Results of the estimation of peaks in spectrum shown in Figure 19.
   <p>
   One can observe improvements in the estimation of peak positions as compared
   to the results achieved by Gold deconvolution.
   <p>
   Script:
   <pre>
   // Example to illustrate deconvolution function (class TSpectrum).
   // To execute this example, do
   // root > .x DeconvolutionRL_wide_boost.C

   #include <TSpectrum>

   void DeconvolutionRL_wide_boost() {
      Int_t i;
      Double_t nbins = 256;
      Double_t xmin  = 0;
      Double_t xmax  = (Double_t)nbins;
      Float_t * source = new float[nbins];
      Float_t * response = new float[nbins];
      TH1F *h = new TH1F("h","Deconvolution",nbins,xmin,xmax);
      TH1F *d = new TH1F("d","",nbins,xmin,xmax);
      TFile *f = new TFile("spectra\\TSpectrum.root");
      h=(TH1F*) f->Get("decon3;1");
      TFile *fr = new TFile("spectra\\TSpectrum.root");
      d=(TH1F*) fr->Get("decon_response_wide;1");
      for (i = 0; i < nbins; i++) source[i]=h->GetBinContent(i + 1);
      for (i = 0; i < nbins; i++) response[i]=d->GetBinContent(i + 1);
      TCanvas *Decon1 = gROOT->GetListOfCanvases()->FindObject("Decon1");
      if (!Decon1) Decon1 = new TCanvas("Decon1",
      "Deconvolution of closely positioned overlapping peaks using boosted Richardson-Lucy deconvolution method",
      10,10,1000,700);
      h->SetMaximum(110000);
      h->Draw("L");
      TSpectrum *s = new TSpectrum();
      s->DeconvolutionRL(source,response,256,200,50,1.2);
      for (i = 0; i < nbins; i++) d->SetBinContent(i + 1,source[i]);
      d->SetLineColor(kRed);
      d->Draw("SAME L");
   }
   </pre>
   End_Html */

   if (ssize <= 0)
      return "Wrong Parameters";

   if (numberRepetitions <= 0)
      return "Wrong Parameters";

       //   working_space-pointer to the working vector
       //   (its size must be 4*ssize of source spectrum)
   double *working_space = new double[4 * ssize];
   int i, j, k, lindex, posit, lh_gold, repet, kmin, kmax;
   double lda, ldb, ldc, maximum;
   lh_gold = -1;
   posit = 0;
   maximum = 0;

//read response vector
   for (i = 0; i < ssize; i++) {
      lda = response[i];
      if (lda != 0)
         lh_gold = i + 1;
      working_space[ssize + i] = lda;
      if (lda > maximum) {
         maximum = lda;
         posit = i;
      }
   }
   if (lh_gold == -1) {
      delete [] working_space;
      return "ZERO RESPONSE VECTOR";
   }

//read source vector
   for (i = 0; i < ssize; i++)
      working_space[2 * ssize + i] = source[i];

//initialization of resulting vector
   for (i = 0; i < ssize; i++){
      if (i <= ssize - lh_gold)
         working_space[i] = 1;

      else
         working_space[i] = 0;

   }
       //**START OF ITERATIONS**
   for (repet = 0; repet < numberRepetitions; repet++) {
      if (repet != 0) {
         for (i = 0; i < ssize; i++)
            working_space[i] = TMath::Power(working_space[i], boost);
      }
      for (lindex = 0; lindex < numberIterations; lindex++) {
         for (i = 0; i <= ssize - lh_gold; i++){
            lda = 0;
            if (working_space[i] > 0){//x[i]
               for (j = i; j < i + lh_gold; j++){
                  ldb = working_space[2 * ssize + j];//y[j]
                  if (j < ssize){
                     if (ldb > 0){//y[j]
                        kmax = j;
                        if (kmax > lh_gold - 1)
                           kmax = lh_gold - 1;
                        kmin = j + lh_gold - ssize;
                        if (kmin < 0)
                           kmin = 0;
                        ldc = 0;
                        for (k = kmax; k >= kmin; k--){
                           ldc += working_space[ssize + k] * working_space[j - k];//h[k]*x[j-k]
                        }
                        if (ldc > 0)
                           ldb = ldb / ldc;

                        else
                           ldb = 0;
                     }
                     ldb = ldb * working_space[ssize + j - i];//y[j]*h[j-i]/suma(h[j][k]x[k])
                  }
                  lda += ldb;
               }
               lda = lda * working_space[i];
            }
            working_space[3 * ssize + i] = lda;
         }
         for (i = 0; i < ssize; i++)
            working_space[i] = working_space[3 * ssize + i];
      }
   }

//shift resulting spectrum
   for (i = 0; i < ssize; i++) {
      lda = working_space[i];
      j = i + posit;
      j = j % ssize;
      working_space[ssize + j] = lda;
   }

//write back resulting spectrum
   for (i = 0; i < ssize; i++)
      source[i] = working_space[ssize + i];
   delete[]working_space;
   return 0;
}


//______________________________________________________________________________
const char *TSpectrum::Unfolding(float *source,
                                 const float **respMatrix,
                                 int ssizex, int ssizey,
                                 int numberIterations,
                                 int numberRepetitions, double boost)
{
   /* Begin_Html
   <b>One-dimensional unfolding function</b>
   <p>
   This function unfolds source spectrum according to response matrix columns.
   The result is placed in the vector pointed by source pointer.
   The coefficients of the resulting vector represent contents of the columns
   (weights) in the input vector. On successful completion it returns 0. On
   error it returns pointer to the string describing error. If desired after
   every numberIterations one can apply boosting operation (exponential
   function with exponent given by boost coefficient) and repeat it
   numberRepetitions times. For details we refer to [1].
   <p>
   Function parameters:
   <ul>
   <li> source: pointer to the vector of source spectrum
   <li> respMatrix: pointer to the matrix of response spectra
   <li> ssizex: length of source spectrum and # of columns of the response
        matrix. ssizex must be >= ssizey.
   <li> ssizey: length of destination spectrum and # of rows of the response
        matrix.
   <li> numberIterations: number of iterations
   <li> numberRepetitions: number of repetitions for boosted deconvolution.
        It must be greater or equal to one.
   <li> boost: boosting coefficient, applies only if numberRepetitions is
        greater than one.
   </ul>
   <p>
   <b>Unfolding:</b>
   <p>
   The goal is the decomposition of spectrum to a given set of component
   spectra.
   <p>
   The mathematical formulation of the discrete linear system is:
   <p>
   <img width=588 height=89 src="gif/TSpectrum_Unfolding1.gif">
   <p>
   <img width=597 height=228 src="gif/TSpectrum_Unfolding2.gif">
   <p>
   <b>References:</b>
   <ol>
   <li> Jandel M., Morhá&#269; M., Kliman J., Krupa L., Matoušek
   V., Hamilton J. H., Ramaya A. V.:
   Decomposition of continuum gamma-ray spectra using synthetized response matrix.
   NIM A 516 (2004), 172-183.
   </ol>
   <p>
   <b>Example of unfolding:</b>
   <p>
   <i>Example 13 - script Unfolding.c:</i>
   <p>
   <img width=442 height=648 src="gif/TSpectrum_Unfolding3.gif">
   <p>
   Fig. 20 Response matrix composed of neutron spectra of pure
   chemical elements.
   <img width=604 height=372 src="gif/TSpectrum_Unfolding2.jpg">
   <p>
   Fig. 21 Source neutron spectrum to be decomposed
   <P>
   <img width=600 height=360 src="gif/TSpectrum_Unfolding3.jpg">
   <p>
   Fig. 22 Spectrum after decomposition, contains 10 coefficients, which
   correspond to contents of chemical components (dominant 8-th and 10-th
   components, i.e. O, Si)
   <p>
   Script:
   <pre>
   // Example to illustrate unfolding function (class TSpectrum).
   // To execute this example, do
   // root > .x Unfolding.C

   #include <TSpectrum>

   void Unfolding() {
      Int_t i, j;
      Int_t nbinsx = 2048;
      Int_t nbinsy = 10;
      Double_t xmin  = 0;
      Double_t xmax  = (Double_t)nbinsx;
      Double_t ymin  = 0;
      Double_t ymax  = (Double_t)nbinsy;
      Float_t * source = new float[nbinsx];
      Float_t ** response = new float *[nbinsy];
      for (i=0;i<nbinsy;i++) response[i]=new float[nbinsx];
      TH1F *h = new TH1F("h","",nbinsx,xmin,xmax);
      TH1F *d = new TH1F("d","Decomposition - unfolding",nbinsx,xmin,xmax);
      TH2F *decon_unf_resp = new TH2F("decon_unf_resp","Root File",nbinsy,ymin,ymax,nbinsx,xmin,xmax);
      TFile *f = new TFile("spectra\\TSpectrum.root");
      h=(TH1F*) f->Get("decon_unf_in;1");
      TFile *fr = new TFile("spectra\\TSpectrum.root");
      decon_unf_resp = (TH2F*) fr->Get("decon_unf_resp;1");
      for (i = 0; i < nbinsx; i++) source[i] = h->GetBinContent(i + 1);
      for (i = 0; i < nbinsy; i++){
         for (j = 0; j< nbinsx; j++){
            response[i][j] = decon_unf_resp->GetBinContent(i + 1, j + 1);
         }
      }
      TCanvas *Decon1 = gROOT->GetListOfCanvases()->FindObject("Decon1");
      if (!Decon1) Decon1 = new TCanvas("Decon1","Decon1",10,10,1000,700);
      h->Draw("L");
      TSpectrum *s = new TSpectrum();
      s->Unfolding(source,response,nbinsx,nbinsy,1000,1,1);
      for (i = 0; i < nbinsy; i++) d->SetBinContent(i + 1,source[i]);
      d->SetLineColor(kRed);
      d->SetAxisRange(0,nbinsy);
      d->Draw("");
   }
   </pre>
   End_Html */

   int i, j, k, lindex, lhx = 0, repet;
   double lda, ldb, ldc, area;
   if (ssizex <= 0 || ssizey <= 0)
      return "Wrong Parameters";
   if (ssizex < ssizey)
      return "Sizex must be greater than sizey)";
   if (numberIterations <= 0)
      return "Number of iterations must be positive";
   double *working_space =
       new double[ssizex * ssizey + 2 * ssizey * ssizey + 4 * ssizex];

/*read response matrix*/
   for (j = 0; j < ssizey && lhx != -1; j++) {
      area = 0;
      lhx = -1;
      for (i = 0; i < ssizex; i++) {
         lda = respMatrix[j][i];
         if (lda != 0) {
            lhx = i + 1;
         }
         working_space[j * ssizex + i] = lda;
         area = area + lda;
      }
      if (lhx != -1) {
         for (i = 0; i < ssizex; i++)
            working_space[j * ssizex + i] /= area;
      }
   }
   if (lhx == -1) {
      delete [] working_space;
      return ("ZERO COLUMN IN RESPONSE MATRIX");
   }

/*read source vector*/
   for (i = 0; i < ssizex; i++)
      working_space[ssizex * ssizey + 2 * ssizey * ssizey + 2 * ssizex + i] =
          source[i];

/*create matrix at*a + at*y */
   for (i = 0; i < ssizey; i++) {
      for (j = 0; j < ssizey; j++) {
         lda = 0;
         for (k = 0; k < ssizex; k++) {
            ldb = working_space[ssizex * i + k];
            ldc = working_space[ssizex * j + k];
            lda = lda + ldb * ldc;
         }
         working_space[ssizex * ssizey + ssizey * i + j] = lda;
      }
      lda = 0;
      for (k = 0; k < ssizex; k++) {
         ldb = working_space[ssizex * i + k];
         ldc =
             working_space[ssizex * ssizey + 2 * ssizey * ssizey + 2 * ssizex +
                           k];
         lda = lda + ldb * ldc;
      }
      working_space[ssizex * ssizey + 2 * ssizey * ssizey + 3 * ssizex + i] =
          lda;
   }

/*move vector at*y*/
   for (i = 0; i < ssizey; i++)
      working_space[ssizex * ssizey + 2 * ssizey * ssizey + 2 * ssizex + i] =
          working_space[ssizex * ssizey + 2 * ssizey * ssizey + 3 * ssizex + i];

/*create matrix at*a*at*a + vector at*a*at*y */
   for (i = 0; i < ssizey; i++) {
      for (j = 0; j < ssizey; j++) {
         lda = 0;
         for (k = 0; k < ssizey; k++) {
            ldb = working_space[ssizex * ssizey + ssizey * i + k];
            ldc = working_space[ssizex * ssizey + ssizey * j + k];
            lda = lda + ldb * ldc;
         }
         working_space[ssizex * ssizey + ssizey * ssizey + ssizey * i + j] =
             lda;
      }
      lda = 0;
      for (k = 0; k < ssizey; k++) {
         ldb = working_space[ssizex * ssizey + ssizey * i + k];
         ldc =
             working_space[ssizex * ssizey + 2 * ssizey * ssizey + 2 * ssizex +
                           k];
         lda = lda + ldb * ldc;
      }
      working_space[ssizex * ssizey + 2 * ssizey * ssizey + 3 * ssizex + i] =
          lda;
   }

/*move at*a*at*y*/
   for (i = 0; i < ssizey; i++)
      working_space[ssizex * ssizey + 2 * ssizey * ssizey + 2 * ssizex + i] =
          working_space[ssizex * ssizey + 2 * ssizey * ssizey + 3 * ssizex + i];

/*initialization in resulting vector */
   for (i = 0; i < ssizey; i++)
      working_space[ssizex * ssizey + 2 * ssizey * ssizey + i] = 1;

        /***START OF ITERATIONS***/
   for (repet = 0; repet < numberRepetitions; repet++) {
      if (repet != 0) {
         for (i = 0; i < ssizey; i++)
            working_space[ssizex * ssizey + 2 * ssizey * ssizey + i] = TMath::Power(working_space[ssizex * ssizey + 2 * ssizey * ssizey + i], boost);
      }
      for (lindex = 0; lindex < numberIterations; lindex++) {
         for (i = 0; i < ssizey; i++) {
            lda = 0;
            for (j = 0; j < ssizey; j++) {
               ldb =
                   working_space[ssizex * ssizey + ssizey * ssizey + ssizey * i + j];
               ldc = working_space[ssizex * ssizey + 2 * ssizey * ssizey + j];
               lda = lda + ldb * ldc;
            }
            ldb =
                working_space[ssizex * ssizey + 2 * ssizey * ssizey + 2 * ssizex + i];
            if (lda != 0) {
               lda = ldb / lda;
            }

            else
               lda = 0;
            ldb = working_space[ssizex * ssizey + 2 * ssizey * ssizey + i];
            lda = lda * ldb;
            working_space[ssizex * ssizey + 2 * ssizey * ssizey + 3 * ssizex + i] = lda;
         }
         for (i = 0; i < ssizey; i++)
            working_space[ssizex * ssizey + 2 * ssizey * ssizey + i] =
                working_space[ssizex * ssizey + 2 * ssizey * ssizey + 3 * ssizex + i];
      }
   }

/*write back resulting spectrum*/
   for (i = 0; i < ssizex; i++) {
      if (i < ssizey)
         source[i] = working_space[ssizex * ssizey + 2 * ssizey * ssizey + i];

      else
         source[i] = 0;
   }
   delete[]working_space;
   return 0;
}


//______________________________________________________________________________
Int_t TSpectrum::SearchHighRes(float *source,float *destVector, int ssize,
                                     float sigma, double threshold,
                                     bool backgroundRemove,int deconIterations,
                                     bool markov, int averWindow)
{
   /* Begin_Html
   <b>One-dimensional high-resolution peak search function</b>
   <p>
   This function searches for peaks in source spectrum. It is based on
   deconvolution method. First the background is removed (if desired), then
   Markov smoothed spectrum is calculated (if desired), then the response
   function is generated according to given sigma and deconvolution is
   carried out. The order of peaks is arranged according to their heights in
   the spectrum after background elimination. The highest peak is the first in
   the list. On success it returns number of found peaks.
   <p>
   <b>Function parameters:</b>
   <ul>
   <li> source: pointer to the vector of source spectrum.
   <li> destVector: pointer to the vector of resulting deconvolved spectrum.
   <li> ssize: length of source spectrum.
   <li> sigma: sigma of searched peaks, for details we refer to manual.
   <li> threshold: threshold value in % for selected peaks, peaks with
        amplitude less than threshold*highest_peak/100
        are ignored, see manual.
   <li> backgroundRemove: logical variable, set if the removal of
        background before deconvolution is desired.
   <li> deconIterations-number of iterations in deconvolution operation.
   <li> markov: logical variable, if it is true, first the source spectrum
        is replaced by new spectrum calculated using Markov
        chains method.
   <li> averWindow: averanging window of searched peaks, for details
        we refer to manual (applies only for Markov method).
   </ul>
   <p>
   <b>Peaks searching:</b>
   <p>
   The goal of this function is to identify automatically the peaks in spectrum
   with the presence of the continuous background and statistical
   fluctuations - noise.
   <p>
   The common problems connected with correct peak identification are:
   <ul>
   <li> non-sensitivity to noise, i.e., only statistically
     relevant peaks should be identified.
   <li> non-sensitivity of the algorithm to continuous
     background.
   <li> ability to identify peaks close to the edges of the
     spectrum region. Usually peak finders fail to detect them.
   <li> resolution, decomposition of doublets and multiplets.
     The algorithm should be able to recognize close positioned peaks.
   <li> ability to identify peaks with different sigma.
   </ul>
   <img width=600 height=375 src="gif/TSpectrum_Searching1.jpg">
   <p>
   Fig. 27 An example of one-dimensional synthetic spectrum with found peaks
   denoted by markers.
   <p>
   <b>References:</b>
   <ol>
   <li> M.A. Mariscotti: A method for identification of peaks in the presence of
   background and its application to spectrum analysis. NIM 50 (1967),
   309-320.
   <li> M. Morhá&#269;, J. Kliman, V.  Matoušek, M. Veselský,
   I. Turzo.:Identification of peaks in
   multidimensional coincidence gamma-ray spectra. NIM, A443 (2000) 108-125.
   <li> Z.K. Silagadze, A new algorithm for automatic photopeak searches. NIM
   A 376 (1996), 451.
   </ol>
   <p>
   <b>Examples of peak searching method:</b>
   <p>
   The SearchHighRes function provides users with the possibility to vary the
   input parameters and with the access to the output deconvolved data in the
   destination spectrum. Based on the output data one can tune the parameters.
   <p>
   Example 15 - script SearchHR1.c:
   <img width=600 height=321 src="gif/TSpectrum_Searching1.jpg">
   <p>
   Fig. 28 One-dimensional spectrum with found peaks denoted by markers, 3
   iterations steps in the deconvolution.
   <p>
   <img width=600 height=323 src="gif/TSpectrum_Searching2.jpg">
   Fig. 29 One-dimensional spectrum with found peaks denoted by markers, 8
   iterations steps in the deconvolution.
   <p>
   Script:
   <pre>
   // Example to illustrate high resolution peak searching function (class TSpectrum).
   // To execute this example, do
   // root > .x SearchHR1.C

   #include <TSpectrum>

   void SearchHR1() {
      Float_t fPositionX[100];
      Float_t fPositionY[100];
      Int_t fNPeaks = 0;
      Int_t i,nfound,bin;
      Double_t nbins = 1024,a;
      Double_t xmin  = 0;
      Double_t xmax  = (Double_t)nbins;
      Float_t * source = new float[nbins];
      Float_t * dest = new float[nbins];
      TH1F *h = new TH1F("h","High resolution peak searching, number of iterations = 3",nbins,xmin,xmax);
      TH1F *d = new TH1F("d","",nbins,xmin,xmax);
      TFile *f = new TFile("spectra\\TSpectrum.root");
      h=(TH1F*) f->Get("search2;1");
      for (i = 0; i < nbins; i++) source[i]=h->GetBinContent(i + 1);
      TCanvas *Search = gROOT->GetListOfCanvases()->FindObject("Search");
      if (!Search) Search = new TCanvas("Search","Search",10,10,1000,700);
      h->SetMaximum(4000);
      h->Draw("L");
      TSpectrum *s = new TSpectrum();
      nfound = s->SearchHighRes(source, dest, nbins, 8, 2, kTRUE, 3, kTRUE, 3);
      Float_t *xpeaks = s->GetPositionX();
      for (i = 0; i < nfound; i++) {
         a=xpeaks[i];
         bin = 1 + Int_t(a + 0.5);
         fPositionX[i] = h->GetBinCenter(bin);
         fPositionY[i] = h->GetBinContent(bin);
      }
      TPolyMarker * pm = (TPolyMarker*)h->GetListOfFunctions()->FindObject("TPolyMarker");
      if (pm) {
         h->GetListOfFunctions()->Remove(pm);
         delete pm;
      }
      pm = new TPolyMarker(nfound, fPositionX, fPositionY);
      h->GetListOfFunctions()->Add(pm);
      pm->SetMarkerStyle(23);
      pm->SetMarkerColor(kRed);
      pm->SetMarkerSize(1.3);
      for (i = 0; i < nbins; i++) d->SetBinContent(i + 1,dest[i]);
      d->SetLineColor(kRed);
      d->Draw("SAME");
      printf("Found %d candidate peaks\n",nfound);
      for(i=0;i<nfound;i++)
         printf("posx= %d, posy= %d\n",fPositionX[i], fPositionY[i]);
      }
   </pre>
   <p>
   Example 16 - script SearchHR3.c:
   <p>
   <table border=solid>
   <tr><td> Peak # </td><td> Position </td><td> Sigma </td></tr>
   <tr><td> 1      </td><td> 118      </td><td> 26    </td></tr>
   <tr><td> 2      </td><td> 162      </td><td> 41    </td></tr>
   <tr><td> 3      </td><td> 310      </td><td> 4     </td></tr>
   <tr><td> 4      </td><td> 330      </td><td> 8     </td></tr>
   <tr><td> 5      </td><td> 482      </td><td> 22    </td></tr>
   <tr><td> 6      </td><td> 491      </td><td> 26    </td></tr>
   <tr><td> 7      </td><td> 740      </td><td> 21    </td></tr>
   <tr><td> 8      </td><td> 852      </td><td> 15    </td></tr>
   <tr><td> 9      </td><td> 954      </td><td> 12    </td></tr>
   <tr><td> 10     </td><td> 989      </td><td> 13    </td></tr>
   </table>
   <p>
   Table 4 Positions and sigma of peaks in the following examples.
   <p>
   <img width=600 height=328 src="gif/TSpectrum_Searching3.jpg">
   <p>
   Fig. 30 Influence of number of iterations (3-red, 10-blue, 100- green,
   1000-magenta), sigma=8, smoothing width=3.
   <p>
   <img width=600 height=321 src="gif/TSpectrum_Searching4.jpg">
   <p>
   Fig. 31 Influence of sigma (3-red, 8-blue, 20- green, 43-magenta),
   num. iter.=10, sm. width=3.
   <p>
   <img width=600 height=323 src="gif/TSpectrum_Searching5.jpg"></p>
   <p>
   Fig. 32 Influence smoothing width (0-red, 3-blue, 7- green, 20-magenta), num.
   iter.=10, sigma=8.
   <p>
   Script:
   <pre>
   // Example to illustrate the influence of number of iterations in deconvolution in high resolution peak searching function (class TSpectrum).
   // To execute this example, do
   // root > .x SearchHR3.C

   #include <TSpectrum>

   void SearchHR3() {
      Float_t fPositionX[100];
      Float_t fPositionY[100];
      Int_t fNPeaks = 0;
      Int_t i,nfound,bin;
      Double_t nbins = 1024,a;
      Double_t xmin  = 0;
      Double_t xmax  = (Double_t)nbins;
      Float_t * source = new float[nbins];
      Float_t * dest = new float[nbins];
      TH1F *h = new TH1F("h","Influence of # of iterations in deconvolution in peak searching",nbins,xmin,xmax);
      TH1F *d1 = new TH1F("d1","",nbins,xmin,xmax);
      TH1F *d2 = new TH1F("d2","",nbins,xmin,xmax);
      TH1F *d3 = new TH1F("d3","",nbins,xmin,xmax);
      TH1F *d4 = new TH1F("d4","",nbins,xmin,xmax);
      TFile *f = new TFile("spectra\\TSpectrum.root");
      h=(TH1F*) f->Get("search3;1");
      for (i = 0; i < nbins; i++) source[i]=h->GetBinContent(i + 1);
      TCanvas *Search = gROOT->GetListOfCanvases()->FindObject("Search");
      if (!Search) Search = new TCanvas("Search","Search",10,10,1000,700);
      h->SetMaximum(1300);
      h->Draw("L");
      TSpectrum *s = new TSpectrum();
      nfound = s->SearchHighRes(source, dest, nbins, 8, 2, kTRUE, 3, kTRUE, 3);
      Float_t *xpeaks = s->GetPositionX();
      for (i = 0; i < nfound; i++) {
         a=xpeaks[i];
         bin = 1 + Int_t(a + 0.5);
         fPositionX[i] = h->GetBinCenter(bin);
         fPositionY[i] = h->GetBinContent(bin);
      }
      TPolyMarker * pm = (TPolyMarker*)h->GetListOfFunctions()->FindObject("TPolyMarker");
      if (pm) {
         h->GetListOfFunctions()->Remove(pm);
         delete pm;
      }
      pm = new TPolyMarker(nfound, fPositionX, fPositionY);
      h->GetListOfFunctions()->Add(pm);
      pm->SetMarkerStyle(23);
      pm->SetMarkerColor(kRed);
      pm->SetMarkerSize(1.3);
      for (i = 0; i < nbins; i++) d1->SetBinContent(i + 1,dest[i]);
      h->Draw("");
      d1->SetLineColor(kRed);
      d1->Draw("SAME");
      for (i = 0; i < nbins; i++) source[i]=h->GetBinContent(i + 1);
      s->SearchHighRes(source, dest, nbins, 8, 2, kTRUE, 10, kTRUE, 3);
      for (i = 0; i < nbins; i++) d2->SetBinContent(i + 1,dest[i]);
      d2->SetLineColor(kBlue);
      d2->Draw("SAME");
      for (i = 0; i < nbins; i++) source[i]=h->GetBinContent(i + 1);
      s->SearchHighRes(source, dest, nbins, 8, 2, kTRUE, 100, kTRUE, 3);
      for (i = 0; i < nbins; i++) d3->SetBinContent(i + 1,dest[i]);
      d3->SetLineColor(kGreen);
      d3->Draw("SAME");
      for (i = 0; i < nbins; i++) source[i]=h->GetBinContent(i + 1);
      s->SearchHighRes(source, dest, nbins, 8, 2, kTRUE, 1000, kTRUE, 3);
      for (i = 0; i < nbins; i++) d4->SetBinContent(i + 1,dest[i]);
      d4->SetLineColor(kMagenta);
      d4->Draw("SAME");
      printf("Found %d candidate peaks\n",nfound);
   }
   </pre>
   End_Html */

   int i, j, numberIterations = (int)(7 * sigma + 0.5);
   double a, b, c;
   int k, lindex, posit, imin, imax, jmin, jmax, lh_gold, priz;
   double lda, ldb, ldc, area, maximum, maximum_decon;
   int xmin, xmax, l, peak_index = 0, size_ext = ssize + 2 * numberIterations, shift = numberIterations, bw = 2, w;
   double maxch;
   double nom, nip, nim, sp, sm, plocha = 0;
   double m0low=0,m1low=0,m2low=0,l0low=0,l1low=0,detlow,av,men;
   if (sigma < 1) {
      Error("SearchHighRes", "Invalid sigma, must be greater than or equal to 1");
      return 0;
   }

   if(threshold<=0 || threshold>=100){
      Error("SearchHighRes", "Invalid threshold, must be positive and less than 100");
      return 0;
   }

   j = (int) (5.0 * sigma + 0.5);
   if (j >= PEAK_WINDOW / 2) {
      Error("SearchHighRes", "Too large sigma");
      return 0;
   }

   if (markov == true) {
      if (averWindow <= 0) {
         Error("SearchHighRes", "Averanging window must be positive");
         return 0;
      }
   }

   if(backgroundRemove == true){
      if(ssize < 2 * numberIterations + 1){
         Error("SearchHighRes", "Too large clipping window");
         return 0;
      }
   }

   k = int(2 * sigma+0.5);
   if(k >= 2){
      for(i = 0;i < k;i++){
         a = i,b = source[i];
         m0low += 1,m1low += a,m2low += a * a,l0low += b,l1low += a * b;
      }
      detlow = m0low * m2low - m1low * m1low;
      if(detlow != 0)
         l1low = (-l0low * m1low + l1low * m0low) / detlow;

      else
         l1low = 0;
      if(l1low > 0)
         l1low=0;
   }

   else{
      l1low = 0;
   }

   i = (int)(7 * sigma + 0.5);
   i = 2 * i;
   double *working_space = new double [7 * (ssize + i)];
   for (j=0;j<7 * (ssize + i);j++) working_space[j] = 0;
   for(i = 0; i < size_ext; i++){
      if(i < shift){
         a = i - shift;
         working_space[i + size_ext] = source[0] + l1low * a;
         if(working_space[i + size_ext] < 0)
            working_space[i + size_ext]=0;
      }

      else if(i >= ssize + shift){
         a = i - (ssize - 1 + shift);
         working_space[i + size_ext] = source[ssize - 1];
         if(working_space[i + size_ext] < 0)
            working_space[i + size_ext]=0;
      }

      else
         working_space[i + size_ext] = source[i - shift];
   }

   if(backgroundRemove == true){
      for(i = 1; i <= numberIterations; i++){
         for(j = i; j < size_ext - i; j++){
            if(markov == false){
               a = working_space[size_ext + j];
               b = (working_space[size_ext + j - i] + working_space[size_ext + j + i]) / 2.0;
               if(b < a)
                  a = b;

               working_space[j]=a;
            }

            else{
               a = working_space[size_ext + j];
               av = 0;
               men = 0;
               for (w = j - bw; w <= j + bw; w++){
                  if ( w >= 0 && w < size_ext){
                     av += working_space[size_ext + w];
                     men +=1;
                  }
               }
               av = av / men;
               b = 0;
               men = 0;
               for (w = j - i - bw; w <= j - i + bw; w++){
                  if ( w >= 0 && w < size_ext){
                     b += working_space[size_ext + w];
                     men +=1;
                  }
               }
               b = b / men;
               c = 0;
               men = 0;
               for (w = j + i - bw; w <= j + i + bw; w++){
                  if ( w >= 0 && w < size_ext){
                     c += working_space[size_ext + w];
                     men +=1;
                  }
               }
               c = c / men;
               b = (b + c) / 2;
               if (b < a)
                  av = b;
               working_space[j]=av;
            }
         }
         for(j = i; j < size_ext - i; j++)
            working_space[size_ext + j] = working_space[j];
      }
      for(j = 0;j < size_ext; j++){
         if(j < shift){
                  a = j - shift;
                  b = source[0] + l1low * a;
                  if (b < 0) b = 0;
            working_space[size_ext + j] = b - working_space[size_ext + j];
         }

         else if(j >= ssize + shift){
                  a = j - (ssize - 1 + shift);
                  b = source[ssize - 1];
                  if (b < 0) b = 0;
            working_space[size_ext + j] = b - working_space[size_ext + j];
         }

         else{
            working_space[size_ext + j] = source[j - shift] - working_space[size_ext + j];
         }
      }
      for(j = 0;j < size_ext; j++){
         if(working_space[size_ext + j] < 0) working_space[size_ext + j] = 0;
      }
   }

   for(i = 0; i < size_ext; i++){
      working_space[i + 6*size_ext] = working_space[i + size_ext];
   }

   if(markov == true){
      for(j = 0; j < size_ext; j++)
         working_space[2 * size_ext + j] = working_space[size_ext + j];
      xmin = 0,xmax = size_ext - 1;
      for(i = 0, maxch = 0; i < size_ext; i++){
         working_space[i] = 0;
         if(maxch < working_space[2 * size_ext + i])
            maxch = working_space[2 * size_ext + i];
         plocha += working_space[2 * size_ext + i];
      }
      if(maxch == 0) {
         delete [] working_space;
         return 0;
      }

      nom = 1;
      working_space[xmin] = 1;
      for(i = xmin; i < xmax; i++){
         nip = working_space[2 * size_ext + i] / maxch;
         nim = working_space[2 * size_ext + i + 1] / maxch;
         sp = 0,sm = 0;
         for(l = 1; l <= averWindow; l++){
            if((i + l) > xmax)
               a = working_space[2 * size_ext + xmax] / maxch;

            else
               a = working_space[2 * size_ext + i + l] / maxch;

            b = a - nip;
            if(a + nip <= 0)
               a=1;

            else
               a = TMath::Sqrt(a + nip);

            b = b / a;
            b = TMath::Exp(b);
            sp = sp + b;
            if((i - l + 1) < xmin)
               a = working_space[2 * size_ext + xmin] / maxch;

            else
               a = working_space[2 * size_ext + i - l + 1] / maxch;

            b = a - nim;
            if(a + nim <= 0)
               a = 1;

            else
               a = TMath::Sqrt(a + nim);

            b = b / a;
            b = TMath::Exp(b);
            sm = sm + b;
         }
         a = sp / sm;
         a = working_space[i + 1] = working_space[i] * a;
         nom = nom + a;
      }
      for(i = xmin; i <= xmax; i++){
         working_space[i] = working_space[i] / nom;
      }
      for(j = 0; j < size_ext; j++)
         working_space[size_ext + j] = working_space[j] * plocha;
      for(j = 0; j < size_ext; j++){
         working_space[2 * size_ext + j] = working_space[size_ext + j];
      }
      if(backgroundRemove == true){
         for(i = 1; i <= numberIterations; i++){
            for(j = i; j < size_ext - i; j++){
               a = working_space[size_ext + j];
               b = (working_space[size_ext + j - i] + working_space[size_ext + j + i]) / 2.0;
               if(b < a)
                  a = b;
               working_space[j] = a;
            }
            for(j = i; j < size_ext - i; j++)
               working_space[size_ext + j] = working_space[j];
         }
         for(j = 0; j < size_ext; j++){
            working_space[size_ext + j] = working_space[2 * size_ext + j] - working_space[size_ext + j];
         }
      }
   }
//deconvolution starts
   area = 0;
   lh_gold = -1;
   posit = 0;
   maximum = 0;
//generate response vector
   for(i = 0; i < size_ext; i++){
      lda = (double)i - 3 * sigma;
      lda = lda * lda / (2 * sigma * sigma);
      j = (int)(1000 * TMath::Exp(-lda));
      lda = j;
      if(lda != 0)
         lh_gold = i + 1;

      working_space[i] = lda;
      area = area + lda;
      if(lda > maximum){
         maximum = lda;
         posit = i;
      }
   }
//read source vector
   for(i = 0; i < size_ext; i++)
      working_space[2 * size_ext + i] = TMath::Abs(working_space[size_ext + i]);
//create matrix at*a(vector b)
   i = lh_gold - 1;
   if(i > size_ext)
      i = size_ext;

   imin = -i,imax = i;
   for(i = imin; i <= imax; i++){
      lda = 0;
      jmin = 0;
      if(i < 0)
         jmin = -i;
      jmax = lh_gold - 1 - i;
      if(jmax > (lh_gold - 1))
         jmax = lh_gold - 1;

      for(j = jmin;j <= jmax; j++){
         ldb = working_space[j];
         ldc = working_space[i + j];
         lda = lda + ldb * ldc;
      }
      working_space[size_ext + i - imin] = lda;
   }
//create vector p
   i = lh_gold - 1;
   imin = -i,imax = size_ext + i - 1;
   for(i = imin; i <= imax; i++){
      lda = 0;
      for(j = 0; j <= (lh_gold - 1); j++){
         ldb = working_space[j];
         k = i + j;
         if(k >= 0 && k < size_ext){
            ldc = working_space[2 * size_ext + k];
            lda = lda + ldb * ldc;
         }

      }
      working_space[4 * size_ext + i - imin] = lda;
   }
//move vector p
   for(i = imin; i <= imax; i++)
      working_space[2 * size_ext + i - imin] = working_space[4 * size_ext + i - imin];
//initialization of resulting vector
   for(i = 0; i < size_ext; i++)
      working_space[i] = 1;
//START OF ITERATIONS
   for(lindex = 0; lindex < deconIterations; lindex++){
      for(i = 0; i < size_ext; i++){
         if(TMath::Abs(working_space[2 * size_ext + i]) > 0.00001 && TMath::Abs(working_space[i]) > 0.00001){
            lda=0;
            jmin = lh_gold - 1;
            if(jmin > i)
               jmin = i;

            jmin = -jmin;
            jmax = lh_gold - 1;
            if(jmax > (size_ext - 1 - i))
               jmax=size_ext-1-i;

            for(j = jmin; j <= jmax; j++){
               ldb = working_space[j + lh_gold - 1 + size_ext];
               ldc = working_space[i + j];
               lda = lda + ldb * ldc;
            }
            ldb = working_space[2 * size_ext + i];
            if(lda != 0)
               lda = ldb / lda;

            else
               lda = 0;

            ldb = working_space[i];
            lda = lda * ldb;
            working_space[3 * size_ext + i] = lda;
         }
      }
      for(i = 0; i < size_ext; i++){
         working_space[i] = working_space[3 * size_ext + i];
      }
   }
//shift resulting spectrum
   for(i=0;i<size_ext;i++){
      lda = working_space[i];
      j = i + posit;
      j = j % size_ext;
      working_space[size_ext + j] = lda;
   }
//write back resulting spectrum
   maximum = 0, maximum_decon = 0;
   j = lh_gold - 1;
   for(i = 0; i < size_ext - j; i++){
      if(i >= shift && i < ssize + shift){
         working_space[i] = area * working_space[size_ext + i + j];
         if(maximum_decon < working_space[i])
            maximum_decon = working_space[i];
         if(maximum < working_space[6 * size_ext + i])
            maximum = working_space[6 * size_ext + i];
      }

      else
         working_space[i] = 0;
   }
   lda=1;
   if(lda>threshold)
      lda=threshold;
   lda=lda/100;

//searching for peaks in deconvolved spectrum
   for(i = 1; i < size_ext - 1; i++){
      if(working_space[i] > working_space[i - 1] && working_space[i] > working_space[i + 1]){
         if(i >= shift && i < ssize + shift){
            if(working_space[i] > lda*maximum_decon && working_space[6 * size_ext + i] > threshold * maximum / 100.0){
               for(j = i - 1, a = 0, b = 0; j <= i + 1; j++){
                  a += (double)(j - shift) * working_space[j];
                  b += working_space[j];
               }
               a = a / b;
               if(a < 0)
                  a = 0;

               if(a >= ssize)
                  a = ssize - 1;
               if(peak_index == 0){
                  fPositionX[0] = a;
                  peak_index = 1;
               }

               else{
                  for(j = 0, priz = 0; j < peak_index && priz == 0; j++){
                     if(working_space[6 * size_ext + shift + (int)a] > working_space[6 * size_ext + shift + (int)fPositionX[j]])
                        priz = 1;
                  }
                  if(priz == 0){
                     if(j < fMaxPeaks){
                        fPositionX[j] = a;
                     }
                  }

                  else{
                     for(k = peak_index; k >= j; k--){
                        if(k < fMaxPeaks){
                           fPositionX[k] = fPositionX[k - 1];
                        }
                     }
                     fPositionX[j - 1] = a;
                  }
                  if(peak_index < fMaxPeaks)
                     peak_index += 1;
               }
            }
         }
      }
   }

   for(i = 0; i < ssize; i++) destVector[i] = working_space[i + shift];
   delete [] working_space;
   fNPeaks = peak_index;
   if(peak_index == fMaxPeaks)
      Warning("SearchHighRes", "Peak buffer full");
   return fNPeaks;
}


//______________________________________________________________________________
Int_t TSpectrum::Search1HighRes(float *source,float *destVector, int ssize,
                                     float sigma, double threshold,
                                     bool backgroundRemove,int deconIterations,
                                     bool markov, int averWindow)
{
   /* Begin_Html
   Old name of SearcHighRes introduced for back compatibility.
   This function will be removed after the June 2006 release
   End_Html */

   return SearchHighRes(source,destVector,ssize,sigma,threshold,backgroundRemove,
                        deconIterations,markov,averWindow);
}


//______________________________________________________________________________
Int_t TSpectrum::StaticSearch(const TH1 *hist, Double_t sigma, Option_t *option, Double_t threshold)
{
   /* Begin_Html
   Static function, interface to TSpectrum::Search.
   End_Html */

   TSpectrum s;
   return s.Search(hist,sigma,option,threshold);
}


//______________________________________________________________________________
TH1 *TSpectrum::StaticBackground(const TH1 *hist,Int_t niter, Option_t *option)
{
   /* Begin_Html
   Static function, interface to TSpectrum::Background.
   End_Html */

   TSpectrum s;
   return s.Background(hist,niter,option);
}
