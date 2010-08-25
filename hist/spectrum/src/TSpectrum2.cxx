// @(#)root/spectrum:$Id$
// Author: Miroslav Morhac   17/01/2006

/////////////////////////////////////////////////////////////////////////////
//   THIS CLASS CONTAINS ADVANCED SPECTRA PROCESSING FUNCTIONS.            //
//                                                                         //
//   ONE-DIMENSIONAL BACKGROUND ESTIMATION FUNCTIONS                       //
//   TWO-DIMENSIONAL BACKGROUND ESTIMATION FUNCTIONS                       //
//   ONE-DIMENSIONAL SMOOTHING FUNCTIONS                                   //
//   TWO-DIMENSIONAL SMOOTHING FUNCTIONS                                   //
//   ONE-DIMENSIONAL DECONVOLUTION FUNCTIONS                               //
//   TWO-DIMENSIONAL DECONVOLUTION FUNCTIONS                               //
//   ONE-DIMENSIONAL PEAK SEARCH FUNCTIONS                                 //
//   TWO-DIMENSIONAL PEAK SEARCH FUNCTIONS                                 //
//                                                                         //
//   These functions were written by:                                      //
//   Miroslav Morhac                                                       //
//   Institute of Physics                                                  //
//   Slovak Academy of Sciences                                            //
//   Dubravska cesta 9, 842 28 BRATISLAVA                                  //
//   SLOVAKIA                                                              //
//                                                                         //
//   email:fyzimiro@savba.sk,    fax:+421 7 54772479                       //
//                                                                         //
//  The original code in C has been repackaged as a C++ class by R.Brun    //
//                                                                         //
//  The algorithms in this class have been published in the following      //
//  references:                                                            //
//   [1]  M.Morhac et al.: Background elimination methods for              //
//   multidimensional coincidence gamma-ray spectra. Nuclear               //
//   Instruments and Methods in Physics Research A 401 (1997) 113-         //
//   132.                                                                  //
//                                                                         //
//   [2]  M.Morhac et al.: Efficient one- and two-dimensional Gold         //
//   deconvolution and its application to gamma-ray spectra                //
//   decomposition. Nuclear Instruments and Methods in Physics             //
//   Research A 401 (1997) 385-408.                                        //
//                                                                         //
//   [3]  M.Morhac et al.: Identification of peaks in multidimensional     //
//   coincidence gamma-ray spectra. Nuclear Instruments and Methods in     //
//   Research Physics A  443(2000), 108-125.                               //
//                                                                         //
//   These NIM papers are also available as Postscript files from:         //
//
/*
   ftp://root.cern.ch/root/SpectrumDec.ps.gz
   ftp://root.cern.ch/root/SpectrumSrc.ps.gz
   ftp://root.cern.ch/root/SpectrumBck.ps.gz
*/ 
//
/////////////////////////////////////////////////////////////////////////////
//
/////////////////////NEW FUNCTIONS  January 2006
//Begin_Html <!--
/* -->
<div class=Section1>

<p class=MsoNormal style='text-align:justify'><i><span lang=EN-US
style='font-size:16.0pt'>All figures in this page were prepared using DaqProVis
system, Data Acquisition, Processing and Visualization system, which is being
developed at the Institute of Physics, Slovak Academy of Sciences, Bratislava,
Slovakia:  </span></i></p>

<p class=MsoNormal style='text-align:justify'><i><span lang=EN-US
style='font-size:16.0pt'><a href="http://www.fu.sav.sk/nph/projects/DaqProVis/">http://www.fu.sav.sk/nph/projects/DaqProVis/</a>
under construction</span></i></p>

<p class=MsoNormal style='text-align:justify'><i><span lang=EN-US
style='font-size:16.0pt'><a href="http://www.fu.sav.sk/nph/projects/ProcFunc/">http://www.fu.sav.sk/nph/projects/ProcFunc/</a>
.</span></i></p>

</div>

<!-- */
// --> End_Html

//______________________________________________________________________________


    
#include "TSpectrum2.h"
#include "TPolyMarker.h"
#include "TList.h"
#include "TH1.h"
#include "TMath.h"
#define PEAK_WINDOW 1024

Int_t TSpectrum2::fgIterations    = 3;
Int_t TSpectrum2::fgAverageWindow = 3;

ClassImp(TSpectrum2)  

//______________________________________________________________________________
TSpectrum2::TSpectrum2() :TNamed("Spectrum", "Miroslav Morhac peak finder") 
{
   // Constructor.

   Int_t n = 100;
   fMaxPeaks   = n;
   fPosition   = new Float_t[n];
   fPositionX  = new Float_t[n];
   fPositionY  = new Float_t[n];
   fResolution = 1;
   fHistogram  = 0;
   fNPeaks     = 0;
}


//______________________________________________________________________________
TSpectrum2::TSpectrum2(Int_t maxpositions, Float_t resolution) :TNamed("Spectrum", "Miroslav Morhac peak finder") 
{   
//  maxpositions:  maximum number of peaks
//  resolution:    determines resolution of the neighboring peaks
//                 default value is 1 correspond to 3 sigma distance
//                 between peaks. Higher values allow higher resolution
//                 (smaller distance between peaks.
//                 May be set later through SetResolution.
   
   Int_t n = maxpositions;
   fMaxPeaks  = n;
   fPosition  = new Float_t[n];
   fPositionX = new Float_t[n];
   fPositionY = new Float_t[n];
   fHistogram = 0;
   fNPeaks    = 0;
   SetResolution(resolution);
}


//______________________________________________________________________________
TSpectrum2::~TSpectrum2() 
{
   // Destructor.

   delete [] fPosition;
   delete [] fPositionX;
   delete [] fPositionY;
   delete    fHistogram;
}


//______________________________________________________________________________
void TSpectrum2::SetAverageWindow(Int_t w)
{
  // static function: Set average window of searched peaks
  // see TSpectrum2::SearchHighRes
   
   fgAverageWindow = w;
}

//______________________________________________________________________________
void TSpectrum2::SetDeconIterations(Int_t n)
{
  // static function: Set max number of decon iterations in deconvolution operation
  // see TSpectrum2::SearchHighRes
   
   fgIterations = n;
}


//______________________________________________________________________________
TH1 *TSpectrum2::Background(const TH1 * h, int number_of_iterations,
                                   Option_t * option) 
{
/////////////////////////////////////////////////////////////////////////////
//   TWO-DIMENSIONAL BACKGROUND ESTIMATION FUNCTION                        //
//   This function calculates the background spectrum in the input histogram h.
//   The background is returned as a histogram. 
//                
//   Function parameters:
//   -h: input 2-d histogram
//   -numberIterations, (default value = 20)
//      Increasing numberIterations make the result smoother and lower.
//   -option: may contain one of the following options
//      - to set the direction parameter
//        "BackIncreasingWindow". By default the direction is BackDecreasingWindow
//      - filterOrder-order of clipping filter,  (default "BackOrder2"                         
//                  -possible values= "BackOrder4"                          
//                                    "BackOrder6"                          
//                                    "BackOrder8"                           
//      - "nosmoothing"- if selected, the background is not smoothed
//           By default the background is smoothed.
//      - smoothWindow-width of smoothing window, (default is "BackSmoothing3")         
//                  -possible values= "BackSmoothing5"                        
//                                    "BackSmoothing7"                       
//                                    "BackSmoothing9"                        
//                                    "BackSmoothing11"                       
//                                    "BackSmoothing13"                       
//                                    "BackSmoothing15"                        
//      - "Compton" if selected the estimation of Compton edge
//                  will be included.
//      - "same" : if this option is specified, the resulting background
//                 histogram is superimposed on the picture in the current pad.
//
//  NOTE that the background is only evaluated in the current range of h.
//  ie, if h has a bin range (set via h->GetXaxis()->SetRange(binmin,binmax),
//  the returned histogram will be created with the same number of bins
//  as the input histogram h, but only bins from binmin to binmax will be filled
//  with the estimated background.
//                                                                         //
/////////////////////////////////////////////////////////////////////////////
   Error("Background","function not yet implemented: h=%s, iter=%d, option=%sn"
        , h->GetName(), number_of_iterations, option);
   return 0;
}

//______________________________________________________________________________
void TSpectrum2::Print(Option_t *) const
{
   // Print the array of positions

   printf("\nNumber of positions = %d\n",fNPeaks);
   for (Int_t i=0;i<fNPeaks;i++) {
      printf(" x[%d] = %g, y[%d] = %g\n",i,fPositionX[i],i,fPositionY[i]);
   }
}



//______________________________________________________________________________
Int_t TSpectrum2::Search(const TH1 * hin, Double_t sigma,
                             Option_t * option, Double_t threshold) 
{   
/////////////////////////////////////////////////////////////////////////////
//   TWO-DIMENSIONAL PEAK SEARCH FUNCTION                                  //
//   This function searches for peaks in source spectrum in hin            //
//   The number of found peaks and their positions are written into        //
//   the members fNpeaks and fPositionX.                                   //
//   The search is performed in the current histogram range.               //
//                                                                         //
//   Function parameters:                                                  //
//   hin:       pointer to the histogram of source spectrum                //
//   sigma:   sigma of searched peaks, for details we refer to manual      //
//   threshold: (default=0.05)  peaks with amplitude less than             //
//       threshold*highest_peak are discarded.  0<threshold<1              //
//                                                                         //
//   By default, the background is removed before deconvolution.           //
//   Specify the option "nobackground" to not remove the background.       //                //
//                                                                         //
//   By default the "Markov" chain algorithm is used.                      //
//   Specify the option "noMarkov" to disable this algorithm               //
//   Note that by default the source spectrum is replaced by a new spectrum//          //
//                                                                         //
//   By default a polymarker object is created and added to the list of    //
//   functions of the histogram. The histogram is drawn with the specified //
//   option and the polymarker object drawn on top of the histogram.       //
//   The polymarker coordinates correspond to the npeaks peaks found in    //
//   the histogram.                                                        //
//   A pointer to the polymarker object can be retrieved later via:        //
//    TList *functions = hin->GetListOfFunctions();                        //
//    TPolyMarker *pm = (TPolyMarker*)functions->FindObject("TPolyMarker") //
//   Specify the option "goff" to disable the storage and drawing of the   //
//   polymarker.                                                           //
//                                                                         //
/////////////////////////////////////////////////////////////////////////////

   if (hin == 0)
      return 0;
   Int_t dimension = hin->GetDimension();
   if (dimension != 2) {
      Error("Search", "Must be a 2-d histogram");
      return 0;
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
   
   Int_t sizex = hin->GetXaxis()->GetNbins();
   Int_t sizey = hin->GetYaxis()->GetNbins();
   Int_t i, j, binx,biny, npeaks;
   Float_t ** source = new float *[sizex];
   Float_t ** dest   = new float *[sizex];
   for (i = 0; i < sizex; i++) {
      source[i] = new float[sizey];
      dest[i]   = new float[sizey];
      for (j = 0; j < sizey; j++) {
         source[i][j] = (Float_t) hin->GetBinContent(i + 1, j + 1);
      }
   }
   //npeaks = SearchHighRes(source, dest, sizex, sizey, sigma, 100*threshold, kTRUE, 3, kTRUE, 10);
   //the smoothing option is used for 1-d but not for 2-d histograms
   npeaks = SearchHighRes(source, dest, sizex, sizey, sigma, 100*threshold,  background, fgIterations, markov, fgAverageWindow);

   //The logic in the loop should be improved to use the fact
   //that fPositionX,Y give a precise position inside a bin.
   //The current algorithm takes the center of the bin only.
   for (i = 0; i < npeaks; i++) {
      binx = 1 + Int_t(fPositionX[i] + 0.5);
      biny = 1 + Int_t(fPositionY[i] + 0.5);
      fPositionX[i] = hin->GetXaxis()->GetBinCenter(binx);
      fPositionY[i] = hin->GetYaxis()->GetBinCenter(biny);
   }
   for (i = 0; i < sizex; i++) {
      delete [] source[i];
      delete [] dest[i];
   }
   delete [] source;
   delete [] dest;
      
   if (opt.Contains("goff"))
      return npeaks;
   if (!npeaks) return 0;
   TPolyMarker * pm = (TPolyMarker*)hin->GetListOfFunctions()->FindObject("TPolyMarker");
   if (pm) {
      hin->GetListOfFunctions()->Remove(pm);
      delete pm;
   }
   pm = new TPolyMarker(npeaks, fPositionX, fPositionY);
   hin->GetListOfFunctions()->Add(pm);
   pm->SetMarkerStyle(23);
   pm->SetMarkerColor(kRed);
   pm->SetMarkerSize(1.3);
   ((TH1*)hin)->Draw(option);
   return npeaks;
}


//______________________________________________________________________________
void TSpectrum2::SetResolution(Float_t resolution) 
{   
//  resolution: determines resolution of the neighboring peaks
//              default value is 1 correspond to 3 sigma distance
//              between peaks. Higher values allow higher resolution
//              (smaller distance between peaks.
//              May be set later through SetResolution.
   if (resolution > 1)
      fResolution = resolution;   
   else
      fResolution = 1;
}


//_____________________________________________________________________________
//_____________________________________________________________________________
    
/////////////////////NEW FUNCTIONS  JANUARY 2006
//______________________________________________________________________________
const char *TSpectrum2::Background(float **spectrum,
                       Int_t ssizex, Int_t ssizey,
                       Int_t numberIterationsX,
                       Int_t numberIterationsY,
                       Int_t direction,
                       Int_t filterType) 
{   
/////////////////////////////////////////////////////////////////////////////
//   TWO-DIMENSIONAL BACKGROUND ESTIMATION FUNCTION - RECTANGULAR RIDGES   // 
//   This function calculates background spectrum from source spectrum.    // 
//   The result is placed to the array pointed by spectrum pointer.        // 
//                                                                         // 
//   Function parameters:                                                  // 
//   spectrum-pointer to the array of source spectrum                      // 
//   ssizex-x length of spectrum                                           // 
//   ssizey-y length of spectrum                                           // 
//   numberIterationsX-maximal x width of clipping window                  // 
//   numberIterationsY-maximal y width of clipping window                  // 
//                           for details we refer to manual                // 
//   direction- direction of change of clipping window                     // 
//               - possible values=kBackIncreasingWindow                   // 
//                                 kBackDecreasingWindow                   // 
//   filterType-determines the algorithm of the filtering                  // 
//                  -possible values=kBackSuccessiveFiltering              // 
//                                   kBackOneStepFiltering                 // 
//                                                                         // 
//                                                                         // 
/////////////////////////////////////////////////////////////////////////////
//
//Begin_Html <!--
/* -->
<div class=Section1>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:20.0pt'>Background
estimation</span></b></p>

<p class=MsoNormal style='text-align:justify'><i><span style='font-size:18.0pt'>&nbsp;</span></i></p>

<p class=MsoNormal style='text-align:justify'><i><span style='font-size:18.0pt'>Goal:
Separation of useful information (peaks) from useless information (background)</span></i><span
style='font-size:18.0pt'> </span></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>method is based on Sensitive
Nonlinear Iterative Peak (SNIP) clipping algorithm [1]</span></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>there exist two algorithms for the
estimation of new value in the channel “<sub><img width=28 height=24
src="gif/TSpectrum2_Background1.gif"></sub>”</span></p>

<p class=MsoNormal style='margin-left:18.0pt;text-align:justify'><span
style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><i><span style='font-size:16.0pt'>Algorithm
based on Successive Comparisons</span></i></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>It
is an extension of one-dimensional SNIP algorithm to another dimension. For
details we refer to [2].</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><i><span style='font-size:16.0pt'>Algorithm
based on One Step Filtering</span></i></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>New
value in the estimated channel is calculated as</span></p>

<p class=MsoNormal style='text-align:justify'><sub><img width=133 height=39
src="gif/TSpectrum2_Background2.gif"></sub></p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><sub><img width=600 height=128
src="gif/TSpectrum2_Background3.gif"></sub></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal><sub><img width=190 height=38
src="gif/TSpectrum2_Background4.gif"></sub>.</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>where
p = 1, 2, …, number_of_iterations. </span></p>

<p class=MsoNormal><i><span style='font-size:18.0pt'>&nbsp;</span></i></p>

<p class=MsoNormal><i><span style='font-size:18.0pt'>Function:</span></i></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:18.0pt'>const
<a href="http://root.cern.ch/root/html/ListOfTypes.html#char" target="_parent">char</a>*
</span></b><span style='font-size:18.0pt'><a
href="http://root.cern.ch/root/html/TSpectrum.html#TSpectrum:Fit1Awmi"><b>TSpectrum2::Background</b></a><b>
(<a href="http://root.cern.ch/root/html/ListOfTypes.html#float" target="_parent">float</a>
**spectrum, <a href="http://root.cern.ch/root/html/ListOfTypes.html#int"
target="_parent">int</a> ssizex, <a
href="http://root.cern.ch/root/html/ListOfTypes.html#int" target="_parent">int</a>
ssizey, <a href="http://root.cern.ch/root/html/ListOfTypes.html#int"
target="_parent">int</a> numberIterationsX, <a
href="http://root.cern.ch/root/html/ListOfTypes.html#int" target="_parent">int</a>
numberIterationsY, <a href="http://root.cern.ch/root/html/ListOfTypes.html#int"
target="_parent">int</a> direction, <a
href="http://root.cern.ch/root/html/ListOfTypes.html#int" target="_parent">int</a>
filterType)  </b></span></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>This
function calculates background spectrum from the source spectrum.  The result
is placed in the matrix pointed by spectrum pointer.  One can also switch the
direction of the change of the clipping window and to select one of the two
above given algorithms.</span><span style='font-size:18.0pt'> </span><span
style='font-size:16.0pt'>On successful completion it returns 0. On error it
returns pointer to the string describing error.</span></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><i><span style='font-size:16.0pt;color:red'>Parameters:</span></i></p>

<p class=MsoNormal>        <b><span style='font-size:14.0pt'>spectrum</span></b>-pointer
to the matrix of source spectrum                  </p>

<p class=MsoNormal>        <b><span style='font-size:14.0pt'>ssizex, ssizey</span></b>-lengths
of the spectrum matrix                                 </p>

<p class=MsoNormal style='text-align:justify'>        <b><span
style='font-size:14.0pt'>numberIterationsX, numberIterationsY</span></b>maximal
widths of clipping</p>

<p class=MsoNormal style='text-align:justify'>        window,                                
</p>

<p class=MsoNormal>        <b><span style='font-size:14.0pt'>direction</span></b>-
direction of change of clipping window                  </p>

<p class=MsoNormal>               - possible
values=kBackIncreasingWindow                      </p>

<p class=MsoNormal>                                           
kBackDecreasingWindow                      </p>

<p class=MsoNormal>        <b><span style='font-size:14.0pt'>filterType</span></b>-type
of the clipping algorithm,                              </p>

<p class=MsoNormal>                  -possible values=kBack SuccessiveFiltering</p>

<p class=MsoNormal>                                             
kBackOneStepFiltering                              </p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><b><i><span style='font-size:18.0pt'>References:</span></i></b></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>[1] 
C. G Ryan et al.: SNIP, a statistics-sensitive background treatment for the
quantitative analysis of PIXE spectra in geoscience applications. NIM, B34
(1988), 396-402.</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>[2]
</span><span lang=SK style='font-size:16.0pt'> M. Morhá&#269;, J. Kliman, V.
Matoušek, M. Veselský, I. Turzo</span><span style='font-size:16.0pt'>.:
Background elimination methods for multidimensional gamma-ray spectra. NIM,
A401 (1997) 113-132.</span></p>

</div>

<!-- */
// --> End_Html
//Begin_Html <!--
/* -->
<div class=Section1>

<p class=MsoNormal><i><span style='font-size:18.0pt'>Example 1– script Back_gamma64.c
:</span></i></p>

<p class=MsoNormal><img width=602 height=418
src="gif/TSpectrum2_Background1.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Fig.
1 Original two-dimensional gamma-gamma-ray spectrum</span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'><img
width=602 height=418 src="gif/TSpectrum2_Background2.jpg"></span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Fig.
2 Background estimated from data from Fig. 1 using decreasing clipping window with
widths 4, 4 and algorithm based on successive comparisons. The estimate
includes not only continuously changing background but also one-dimensional
ridges.</span></b></p>

<p class=MsoNormal><b><span style='font-size:14.0pt;color:green'><img
width=602 height=418 src="gif/TSpectrum2_Background3.jpg"></span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Fig.
3 Resulting peaks after subtraction of the estimated background (Fig. 2) from original
two-dimensional gamma-gamma-ray spectrum (Fig. 1).</span></b></p>

<p class=MsoNormal><b><span style='font-size:14.0pt;color:green'>&nbsp;</span></b></p>

<p class=MsoNormal><b><span style='font-size:14.0pt;color:green'>&nbsp;</span></b></p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:green'>Script:</span></b></p>

<p class=MsoNormal>// Example to illustrate the background estimator (class
TSpectrum).</p>

<p class=MsoNormal>// To execute this example, do</p>

<p class=MsoNormal>// root &gt; .x Back_gamma64.C</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>#include &lt;TSpectrum&gt; </p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>void Back_gamma64() {</p>

<p class=MsoNormal>   Int_t i, j;</p>

<p class=MsoNormal>   Double_t nbinsx = 64;</p>

<p class=MsoNormal>   Double_t nbinsy = 64;   </p>

<p class=MsoNormal>   Double_t xmin  = 0;</p>

<p class=MsoNormal>   Double_t xmax  = (Double_t)nbinsx;</p>

<p class=MsoNormal>   Double_t ymin  = 0;</p>

<p class=MsoNormal>   Double_t ymax  = (Double_t)nbinsy;   </p>

<p class=MsoNormal>   Float_t ** source = new float *[nbinsx];   </p>

<p class=MsoNormal>   for (i=0;i&lt;nbinsx;i++)</p>

<p class=MsoNormal>      source[i]=new float[nbinsy];     </p>

<p class=MsoNormal>   TH2F *back = new TH2F(&quot;back&quot;,&quot;Background estimation&quot;,nbinsx,xmin,xmax,nbinsy,ymin,ymax);</p>

<p class=MsoNormal>   TFile *f = new
TFile(&quot;spectra2\\TSpectrum2.root&quot;);</p>

<p class=MsoNormal>   back=(TH2F*) f-&gt;Get(&quot;back1;1&quot;);</p>

<p class=MsoNormal>   TCanvas *Background = new
TCanvas(&quot;Background&quot;,&quot;Estimation of background with increasing
window&quot;,10,10,1000,700);</p>

<p class=MsoNormal>   TSpectrum *s = new TSpectrum();</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbinsx; i++){</p>

<p class=MsoNormal>     for (j = 0; j &lt; nbinsy; j++){</p>

<p class=MsoNormal>                source[i][j] = back-&gt;GetBinContent(i +
1,j + 1); </p>

<p class=MsoNormal>             }</p>

<p class=MsoNormal>   }     </p>

<p class=MsoNormal> s-&gt;Background(source,nbinsx,nbinsy,4,4,kBackDecreasingWindow,kBackSuccessiveFiltering);</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbinsx; i++){</p>

<p class=MsoNormal>     for (j = 0; j &lt; nbinsy; j++)</p>

<p class=MsoNormal>       back-&gt;SetBinContent(i + 1,j + 1, source[i][j]);   </p>

<p class=MsoNormal>   }</p>

<p class=MsoNormal>   back-&gt;Draw(&quot;SURF&quot;);  </p>

<p class=MsoNormal>   }</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><i><span style='font-size:18.0pt'>Example 2– script Back_gamma256.c
:</span></i></p>

<p class=MsoNormal><img width=602 height=418
src="gif/TSpectrum2_Background4.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Fig.
4 Original two-dimensional gamma-gamma-ray spectrum 256x256 channels</span></b></p>

<p class=MsoNormal><img width=602 height=418
src="gif/TSpectrum2_Background5.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Fig.
5 Peaks after subtraction of the estimated background (increasing clipping
window with widths 8, 8 and algorithm based on successive comparisons) from original
two-dimensional gamma-gamma-ray spectrum (Fig. 4).</span></b></p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:green'>&nbsp;</span></b></p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:green'>Script:</span></b></p>

<p class=MsoNormal>// Example to illustrate the background estimator (class
TSpectrum).</p>

<p class=MsoNormal>// To execute this example, do</p>

<p class=MsoNormal>// root &gt; .x Back_gamma256.C</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>#include &lt;TSpectrum&gt; </p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>void Back_gamma256() {</p>

<p class=MsoNormal>   Int_t i, j;</p>

<p class=MsoNormal>   Double_t nbinsx = 64;</p>

<p class=MsoNormal>   Double_t nbinsy = 64;   </p>

<p class=MsoNormal>   Double_t xmin  = 0;</p>

<p class=MsoNormal>   Double_t xmax  = (Double_t)nbinsx;</p>

<p class=MsoNormal>   Double_t ymin  = 0;</p>

<p class=MsoNormal>   Double_t ymax  = (Double_t)nbinsy;   </p>

<p class=MsoNormal>   Float_t ** source = new float *[nbinsx];   </p>

<p class=MsoNormal>   for (i=0;i&lt;nbinsx;i++)</p>

<p class=MsoNormal>      source[i]=new float[nbinsy];     </p>

<p class=MsoNormal>   TH2F *back = new TH2F(&quot;back&quot;,&quot;Background estimation&quot;,nbinsx,xmin,xmax,nbinsy,ymin,ymax);</p>

<p class=MsoNormal>   TFile *f = new
TFile(&quot;spectra2\\TSpectrum2.root&quot;);</p>

<p class=MsoNormal>   back=(TH2F*) f-&gt;Get(&quot;back2;1&quot;);</p>

<p class=MsoNormal>   TCanvas *Background = new
TCanvas(&quot;Background&quot;,&quot;Estimation of background with increasing
window&quot;,10,10,1000,700);</p>

<p class=MsoNormal>   TSpectrum *s = new TSpectrum();</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbinsx; i++){</p>

<p class=MsoNormal>     for (j = 0; j &lt; nbinsy; j++){</p>

<p class=MsoNormal>                source[i][j] = back-&gt;GetBinContent(i +
1,j + 1); </p>

<p class=MsoNormal>             }</p>

<p class=MsoNormal>   }     </p>

<p class=MsoNormal> s-&gt;Background(source,nbinsx,nbinsy,8,8,kBackIncreasingWindow,kBackSuccessiveFiltering);</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbinsx; i++){</p>

<p class=MsoNormal>     for (j = 0; j &lt; nbinsy; j++)</p>

<p class=MsoNormal>       back-&gt;SetBinContent(i + 1,j + 1, source[i][j]);   </p>

<p class=MsoNormal>   }</p>

<p class=MsoNormal>   back-&gt;Draw(&quot;SURF&quot;);  </p>

<p class=MsoNormal>   }</p>

<p class=MsoNormal><i><span style='font-size:18.0pt'>Example 3– script Back_synt256.c
:</span></i></p>

<p class=MsoNormal><img width=602 height=418
src="gif/TSpectrum2_Background6.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Fig.
6 Original two-dimensional synthetic spectrum 256x256 channels</span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'><img
width=602 height=418 src="gif/TSpectrum2_Background7.jpg"></span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Fig.
7 Peaks after subtraction of the estimated background (increasing clipping
window with widths 8, 8 and algorithm based on successive comparisons) from original
two-dimensional gamma-gamma-ray spectrum (Fig. 6). One can observe artifacts
(ridges) around the peaks due to the employed algorithm.</span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'><img
width=602 height=418 src="gif/TSpectrum2_Background8.jpg"></span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Fig.
8 Peaks after subtraction of the estimated background (increasing clipping
window with widths 8, 8 and algorithm based on one step filtering) from original
two-dimensional gamma-gamma-ray spectrum (Fig. 6).  The artifacts from the
above given Fig. 7 disappeared.</span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>&nbsp;</span></b></p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:green'>Script:</span></b></p>

<p class=MsoNormal>// Example to illustrate the background estimator (class
TSpectrum).</p>

<p class=MsoNormal>// To execute this example, do</p>

<p class=MsoNormal>// root &gt; .x Back_synt256.C</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>#include &lt;TSpectrum&gt; </p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>void Back_synt256() {</p>

<p class=MsoNormal>   Int_t i, j;</p>

<p class=MsoNormal>   Double_t nbinsx = 64;</p>

<p class=MsoNormal>   Double_t nbinsy = 64;   </p>

<p class=MsoNormal>   Double_t xmin  = 0;</p>

<p class=MsoNormal>   Double_t xmax  = (Double_t)nbinsx;</p>

<p class=MsoNormal>   Double_t ymin  = 0;</p>

<p class=MsoNormal>   Double_t ymax  = (Double_t)nbinsy;   </p>

<p class=MsoNormal>   Float_t ** source = new float *[nbinsx];   </p>

<p class=MsoNormal>   for (i=0;i&lt;nbinsx;i++)</p>

<p class=MsoNormal>      source[i]=new float[nbinsy];     </p>

<p class=MsoNormal>   TH2F *back = new TH2F(&quot;back&quot;,&quot;Background estimation&quot;,nbinsx,xmin,xmax,nbinsy,ymin,ymax);</p>

<p class=MsoNormal>   TFile *f = new
TFile(&quot;spectra2\\TSpectrum2.root&quot;);</p>

<p class=MsoNormal>   back=(TH2F*) f-&gt;Get(&quot;back3;1&quot;);</p>

<p class=MsoNormal>   TCanvas *Background = new
TCanvas(&quot;Background&quot;,&quot;Estimation of background with increasing
window&quot;,10,10,1000,700);</p>

<p class=MsoNormal>   TSpectrum *s = new TSpectrum();</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbinsx; i++){</p>

<p class=MsoNormal>     for (j = 0; j &lt; nbinsy; j++){</p>

<p class=MsoNormal>                source[i][j] = back-&gt;GetBinContent(i +
1,j + 1); </p>

<p class=MsoNormal>             }</p>

<p class=MsoNormal>   }     </p>

<p class=MsoNormal> s-&gt;Background(source,nbinsx,nbinsy,8,8,kBackIncreasingWindow,kBackSuccessiveFiltering);//kBackOneStepFiltering</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbinsx; i++){</p>

<p class=MsoNormal>     for (j = 0; j &lt; nbinsy; j++)</p>

<p class=MsoNormal>       back-&gt;SetBinContent(i + 1,j + 1, source[i][j]);   </p>

<p class=MsoNormal>   }</p>

<p class=MsoNormal>   back-&gt;Draw(&quot;SURF&quot;);  </p>

<p class=MsoNormal>   }</p>

</div>

<!-- */
// --> End_Html

   int i, x, y, sampling, r1, r2;
   float a, b, p1, p2, p3, p4, s1, s2, s3, s4;
   if (ssizex <= 0 || ssizey <= 0)
      return "Wrong parameters";
   if (numberIterationsX < 1 || numberIterationsY < 1)
      return "Width of Clipping Window Must Be Positive";
   if (ssizex < 2 * numberIterationsX + 1
        || ssizey < 2 * numberIterationsY + 1)
      return ("Too Large Clipping Window");
   float **working_space = new float *[ssizex];
   for (i = 0; i < ssizex; i++)
      working_space[i] = new float[ssizey];
   sampling =
       (int) TMath::Max(numberIterationsX, numberIterationsY);
   if (direction == kBackIncreasingWindow) {
      if (filterType == kBackSuccessiveFiltering) {
         for (i = 1; i <= sampling; i++) {
            r1 = (int) TMath::Min(i, numberIterationsX), r2 =
                (int) TMath::Min(i, numberIterationsY);
            for (y = r2; y < ssizey - r2; y++) {
               for (x = r1; x < ssizex - r1; x++) {
                  a = spectrum[x][y];
                  p1 = spectrum[x - r1][y - r2];
                  p2 = spectrum[x - r1][y + r2];
                  p3 = spectrum[x + r1][y - r2];
                  p4 = spectrum[x + r1][y + r2];
                  s1 = spectrum[x][y - r2];
                  s2 = spectrum[x - r1][y];
                  s3 = spectrum[x + r1][y];
                  s4 = spectrum[x][y + r2];
                  b = (p1 + p2) / 2.0;
                  if (b > s2)
                     s2 = b;
                  b = (p1 + p3) / 2.0;
                  if (b > s1)
                     s1 = b;
                  b = (p2 + p4) / 2.0;
                  if (b > s4)
                     s4 = b;
                  b = (p3 + p4) / 2.0;
                  if (b > s3)
                     s3 = b;
                  s1 = s1 - (p1 + p3) / 2.0;
                  s2 = s2 - (p1 + p2) / 2.0;
                  s3 = s3 - (p3 + p4) / 2.0;
                  s4 = s4 - (p2 + p4) / 2.0;
                  b = (s1 + s4) / 2.0 + (s2 + s3) / 2.0 + (p1 + p2 +
                                                            p3 +
                                                            p4) / 4.0;
                  if (b < a && b > 0)
                     a = b;
                  working_space[x][y] = a;
               }
            }
            for (y = r2; y < ssizey - r2; y++) {
               for (x = r1; x < ssizex - r1; x++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }                 
      }
      
      else if (filterType == kBackOneStepFiltering) {
         for (i = 1; i <= sampling; i++) {
            r1 = (int) TMath::Min(i, numberIterationsX), r2 =
                (int) TMath::Min(i, numberIterationsY);
            for (y = r2; y < ssizey - r2; y++) {
               for (x = r1; x < ssizex - r1; x++) {
                  a = spectrum[x][y];
                  b = -(spectrum[x - r1][y - r2] +
                         spectrum[x - r1][y + r2] + spectrum[x + r1][y -
                                                                     r2]
                         + spectrum[x + r1][y + r2]) / 4 +
                      (spectrum[x][y - r2] + spectrum[x - r1][y] +
                       spectrum[x + r1][y] + spectrum[x][y + r2]) / 2;
                  if (b < a && b > 0)
                     a = b;
                  working_space[x][y] = a;
               }
            }
            for (y = i; y < ssizey - i; y++) {
               for (x = i; x < ssizex - i; x++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
   }
   
   else if (direction == kBackDecreasingWindow) {
      if (filterType == kBackSuccessiveFiltering) {
         for (i = sampling; i >= 1; i--) {
            r1 = (int) TMath::Min(i, numberIterationsX), r2 =
                (int) TMath::Min(i, numberIterationsY);
            for (y = r2; y < ssizey - r2; y++) {
               for (x = r1; x < ssizex - r1; x++) {
                  a = spectrum[x][y];
                  p1 = spectrum[x - r1][y - r2];
                  p2 = spectrum[x - r1][y + r2];
                  p3 = spectrum[x + r1][y - r2];
                  p4 = spectrum[x + r1][y + r2];
                  s1 = spectrum[x][y - r2];
                  s2 = spectrum[x - r1][y];
                  s3 = spectrum[x + r1][y];
                  s4 = spectrum[x][y + r2];
                  b = (p1 + p2) / 2.0;
                  if (b > s2)
                     s2 = b;
                  b = (p1 + p3) / 2.0;
                  if (b > s1)
                     s1 = b;
                  b = (p2 + p4) / 2.0;
                  if (b > s4)
                     s4 = b;
                  b = (p3 + p4) / 2.0;
                  if (b > s3)
                     s3 = b;
                  s1 = s1 - (p1 + p3) / 2.0;
                  s2 = s2 - (p1 + p2) / 2.0;
                  s3 = s3 - (p3 + p4) / 2.0;
                  s4 = s4 - (p2 + p4) / 2.0;
                  b = (s1 + s4) / 2.0 + (s2 + s3) / 2.0 + (p1 + p2 +
                                                            p3 +
                                                            p4) / 4.0;
                  if (b < a && b > 0)
                     a = b;
                  working_space[x][y] = a;
               }
            }
            for (y = r2; y < ssizey - r2; y++) {
               for (x = r1; x < ssizex - r1; x++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }        
      
      else if (filterType == kBackOneStepFiltering) {
         for (i = sampling; i >= 1; i--) {
            r1 = (int) TMath::Min(i, numberIterationsX), r2 =
                (int) TMath::Min(i, numberIterationsY);
            for (y = r2; y < ssizey - r2; y++) {
               for (x = r1; x < ssizex - r1; x++) {
                  a = spectrum[x][y];
                  b = -(spectrum[x - r1][y - r2] +
                         spectrum[x - r1][y + r2] + spectrum[x + r1][y -
                                                                     r2]
                         + spectrum[x + r1][y + r2]) / 4 +
                      (spectrum[x][y - r2] + spectrum[x - r1][y] +
                       spectrum[x + r1][y] + spectrum[x][y + r2]) / 2;
                  if (b < a && b > 0)
                     a = b;
                  working_space[x][y] = a;
               }
            }
            for (y = i; y < ssizey - i; y++) {
               for (x = i; x < ssizex - i; x++) {
                  spectrum[x][y] = working_space[x][y];
               }
            }
         }
      }
   }
   for (i = 0; i < ssizex; i++)
      delete[]working_space[i];
   delete[]working_space;
   return 0;
}

//_____________________________________________________________________________
const char* TSpectrum2::SmoothMarkov(float **source, Int_t ssizex, Int_t ssizey, Int_t averWindow)
{
/////////////////////////////////////////////////////////////////////////////
//   TWO-DIMENSIONAL MARKOV SPECTRUM SMOOTHING FUNCTION               
//
//   This function calculates smoothed spectrum from source spectrum    
//      based on Markov chain method.                                     
//   The result is placed in the array pointed by source pointer.      
//
//   Function parameters:
//   source-pointer to the array of source spectrum
//   ssizex-x length of source
//   ssizey-y length of source
//   averWindow-width of averaging smoothing window
//
/////////////////////////////////////////////////////////////////////////////
//Begin_Html <!--
/* -->
<div class=Section1>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:20.0pt'>Smoothing</span></b></p>

<p class=MsoNormal style='text-align:justify'><i><span style='font-size:18.0pt'>&nbsp;</span></i></p>

<p class=MsoNormal><i><span style='font-size:18.0pt'>Goal: Suppression of
statistical fluctuations</span></i></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>the algorithm is based on discrete
Markov chain, which has very simple invariant distribution</span></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>            <sub><img width=551 height=63
src="gif/TSpectrum2_Smoothing1.gif"></sub><span style='font-size:16.0pt;
font-family:Arial'>     </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt;
font-family:Arial'>        </span><sub><img width=28 height=36
src="gif/TSpectrum2_Smoothing2.gif"></sub><span style='font-size:16.0pt;
font-family:Arial'>  being defined from the normalization condition </span><sub><img
width=70 height=52 src="gif/TSpectrum2_Smoothing3.gif"></sub></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><span style='font-size:16.0pt;font-family:Arial'>         n
is the length of the smoothed spectrum and </span></p>

<p class=MsoNormal>

<table cellpadding=0 cellspacing=0 align=left>
 <tr>
  <td width=57 height=15></td>
 </tr>
 <tr>
  <td></td>
  <td><img width=258 height=60 src="gif/TSpectrum2_Smoothing4.gif"></td>
 </tr>
</table>

<span style='font-size:16.0pt;font-family:Arial'>&nbsp;</span></p>

<p class=MsoNormal><span style='font-size:18.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<br clear=ALL>

<p class=MsoNormal style='margin-left:34.2pt;text-align:justify'><span
style='font-size:16.0pt;font-family:Arial'>is the probability of the change of
the peak position from channel i to the channel i+1. </span> <sub><img
width=28 height=36 src="gif/TSpectrum2_Smoothing5.gif"></sub><span
style='font-size:16.0pt;font-family:Arial'>is the normalization constant so
that </span><span style='font-family:Arial'><sub><img width=133 height=34
src="gif/TSpectrum2_Smoothing6.gif"></sub> </span><span style='font-size:16.0pt;
font-family:Arial'>and m is a width of smoothing window. We have extended this
algortihm to two dimensions. </span></p>

<p class=MsoNormal><i><span style='font-size:18.0pt'>&nbsp;</span></i></p>

<p class=MsoNormal><i><span style='font-size:18.0pt'>Function:</span></i></p>

<p class=MsoNormal><b><span style='font-size:18.0pt'>const <a
href="http://root.cern.ch/root/html/ListOfTypes.html#char" target="_parent">char</a>*
</span></b><span style='font-size:18.0pt'><a
href="http://root.cern.ch/root/html/TSpectrum.html#TSpectrum:Fit1Awmi"><b>TSpectrum2::SmoothMarkov</b></a><b>(<a
href="http://root.cern.ch/root/html/ListOfTypes.html#float" target="_parent">float</a>
**fSpectrum, <a href="http://root.cern.ch/root/html/ListOfTypes.html#int"
target="_parent">int</a> ssizex, <a
href="http://root.cern.ch/root/html/ListOfTypes.html#int" target="_parent">int</a>
ssizey,  <a href="http://root.cern.ch/root/html/ListOfTypes.html#int"
target="_parent">int</a> averWindow)  </b></span></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>This
function calculates smoothed spectrum from the source spectrum based on Markov
chain method. The result is placed in the vector pointed by source pointer. On
successful completion it returns 0. On error it returns pointer to the string
describing error.</span></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><i><span style='font-size:16.0pt;color:red'>Parameters:</span></i></p>

<p class=MsoNormal>        <b><span style='font-size:14.0pt'>fSpectrum</span></b>-pointer
to the matrix of source spectrum                  </p>

<p class=MsoNormal>        <b><span style='font-size:14.0pt'>ssizex, ssizey</span></b>
-lengths of the spectrum matrix                                 </p>

<p class=MsoNormal>        <b><span style='font-size:14.0pt'>averWindow</span></b>-width
of averaging smoothing window </p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><b><i><span style='font-size:18.0pt'>Reference:</span></i></b></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>[1]
Z.K. Silagadze, A new algorithm for automatic photopeak searches. NIM A 376
(1996), 451<b>.</b>  </span></p>

</div>

<!-- */
// --> End_Html
//Begin_Html <!--
/* -->
<div class=Section1>

<p class=MsoNormal><i><span style='font-size:16.0pt'>Example 4 – script Smooth.c
:</span></i></p>

<p class=MsoNormal><span style='font-size:16.0pt'><img width=300 height=209
src="gif/TSpectrum2_Smoothing1.jpg"><img width=297 height=207
src="gif/TSpectrum2_Smoothing2.jpg"></span></p>

<p class=MsoNormal><b><span style='font-size:16.0pt'>Fig. 9 Original noisy
spectrum.</span></b><b><span style='font-size:14.0pt'>    </span></b><b><span
style='font-size:16.0pt'>Fig. 10 Smoothed spectrum m=3</span></b></p>

<p class=MsoNormal><b><span style='font-size:16.0pt'>Peaks can hardly be
observed.     Peaks become apparent.</span></b></p>

<p class=MsoNormal><b><span style='font-size:14.0pt'><img width=293 height=203
src="gif/TSpectrum2_Smoothing3.jpg"><img width=297 height=205
src="gif/TSpectrum2_Smoothing4.jpg"></span></b></p>

<p class=MsoNormal><b><span style='font-size:16.0pt'>Fig. 11 Smoothed spectrum
m=5 Fig.12 Smoothed spectrum m=7</span></b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:#339966'>Script:</span></b></p>

<p class=MsoNormal>// Example to illustrate the Markov smoothing (class
TSpectrum).</p>

<p class=MsoNormal>// To execute this example, do</p>

<p class=MsoNormal>// root &gt; .x Smooth.C</p>

<p class=MsoNormal>#include &lt;TSpectrum&gt; </p>

<p class=MsoNormal>void Smooth() {</p>

<p class=MsoNormal>   Int_t i, j;</p>

<p class=MsoNormal>   Double_t nbinsx = 256;</p>

<p class=MsoNormal>   Double_t nbinsy = 256;   </p>

<p class=MsoNormal>   Double_t xmin  = 0;</p>

<p class=MsoNormal>   Double_t xmax  = (Double_t)nbinsx;</p>

<p class=MsoNormal>   Double_t ymin  = 0;</p>

<p class=MsoNormal>   Double_t ymax  = (Double_t)nbinsy;   </p>

<p class=MsoNormal>   Float_t ** source = new float *[nbinsx];   </p>

<p class=MsoNormal>   for (i=0;i&lt;nbinsx;i++)</p>

<p class=MsoNormal>                                    source[i]=new
float[nbinsy];     </p>

<p class=MsoNormal>   TH2F *smooth = new
TH2F(&quot;smooth&quot;,&quot;Background
estimation&quot;,nbinsx,xmin,xmax,nbinsy,ymin,ymax);</p>

<p class=MsoNormal>   TFile *f = new
TFile(&quot;spectra2\\TSpectrum2.root&quot;);</p>

<p class=MsoNormal>   smooth=(TH2F*) f-&gt;Get(&quot;smooth1;1&quot;);</p>

<p class=MsoNormal>   TCanvas *Smoothing = new
TCanvas(&quot;Smoothing&quot;,&quot;Markov smoothing&quot;,10,10,1000,700);</p>

<p class=MsoNormal>   TSpectrum *s = new TSpectrum();</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbinsx; i++){</p>

<p class=MsoNormal>     for (j = 0; j &lt; nbinsy; j++){</p>

<p class=MsoNormal>                source[i][j] = smooth-&gt;GetBinContent(i +
1,j + 1); </p>

<p class=MsoNormal>             }</p>

<p class=MsoNormal>   }</p>

<p class=MsoNormal>   s-&gt;SmoothMarkov(source,nbinsx,nbinsx,3);//5,7</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbinsx; i++){</p>

<p class=MsoNormal>     for (j = 0; j &lt; nbinsy; j++)</p>

<p class=MsoNormal>       smooth-&gt;SetBinContent(i + 1,j + 1,
source[i][j]);   </p>

<p class=MsoNormal>   }</p>

<p class=MsoNormal>   smooth-&gt;Draw(&quot;SURF&quot;);  </p>

<p class=MsoNormal>   }</p>

</div>

<!-- */
// --> End_Html
   int xmin, xmax, ymin, ymax, i, j, l;
   double a, b, maxch;
   double nom, nip, nim, sp, sm, spx, spy, smx, smy, plocha = 0;
   if(averWindow <= 0)
      return "Averaging Window must be positive";      
   float **working_space = new float* [ssizex];
   for(i = 0; i < ssizex; i++)
      working_space[i] = new float[ssizey];      
   xmin = 0;
   xmax = ssizex - 1;
   ymin = 0;
   ymax = ssizey - 1;
   for(i = 0, maxch = 0; i < ssizex; i++){
      for(j = 0; j < ssizey; j++){
         working_space[i][j] = 0;
         if(maxch < source[i][j])
            maxch = source[i][j];

         plocha += source[i][j];
      }
   }
   if(maxch == 0) {
      delete [] working_space;
      return 0;
   }

   nom = 0;
   working_space[xmin][ymin] = 1;
   for(i = xmin; i < xmax; i++){
      nip = source[i][ymin] / maxch;
      nim = source[i + 1][ymin] / maxch;
      sp = 0,sm = 0;
      for(l = 1; l <= averWindow; l++){
         if((i + l) > xmax)
            a = source[xmax][ymin] / maxch;

         else
            a = source[i + l][ymin] / maxch;
         b = a - nip;
         if(a + nip <= 0)
            a = 1;

         else
            a = TMath::Sqrt(a + nip);            
         b = b / a;
         b = TMath::Exp(b);
         sp = sp + b;
         if(i - l + 1 < xmin)
            a = source[xmin][ymin] / maxch;

         else
            a = source[i - l + 1][ymin] / maxch;
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
      a = working_space[i + 1][ymin] = a * working_space[i][ymin];
      nom = nom + a;
   }
   for(i = ymin; i < ymax; i++){
      nip = source[xmin][i] / maxch;
      nim = source[xmin][i + 1] / maxch;
      sp = 0,sm = 0;
      for(l = 1; l <= averWindow; l++){
         if((i + l) > ymax)
            a = source[xmin][ymax] / maxch;

         else
            a = source[xmin][i + l] / maxch;
         b = a - nip;
         if(a + nip <= 0)
            a = 1;

         else
            a = TMath::Sqrt(a + nip);            
         b = b / a;
         b = TMath::Exp(b);
         sp = sp + b;
         if(i - l + 1 < ymin)
            a = source[xmin][ymin] / maxch;

         else
            a = source[xmin][i - l + 1] / maxch;
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
      a = working_space[xmin][i + 1] = a * working_space[xmin][i];
      nom = nom + a;
   }
   for(i = xmin; i < xmax; i++){
      for(j = ymin; j < ymax; j++){
         nip = source[i][j + 1] / maxch;
         nim = source[i + 1][j + 1] / maxch;
         spx = 0,smx = 0;
         for(l = 1; l <= averWindow; l++){
            if(i + l > xmax)
               a = source[xmax][j] / maxch;

            else
               a = source[i + l][j] / maxch;
            b = a - nip;
            if(a + nip <= 0)
               a = 1;

            else
               a = TMath::Sqrt(a + nip);            
            b = b / a;
            b = TMath::Exp(b);
            spx = spx + b;
            if(i - l + 1 < xmin)
               a = source[xmin][j] / maxch;

            else
               a = source[i - l + 1][j] / maxch;
            b = a - nim;
            if(a + nim <= 0)
               a = 1;

            else
               a = TMath::Sqrt(a + nim);            
            b = b / a;
            b = TMath::Exp(b);
            smx = smx + b;
         }
         spy = 0,smy = 0;
         nip = source[i + 1][j] / maxch;
         nim = source[i + 1][j + 1] / maxch;
         for (l = 1; l <= averWindow; l++) {
            if (j + l > ymax) a = source[i][ymax]/maxch;
            else              a = source[i][j + l] / maxch;
            b = a - nip;
            if (a + nip <= 0) a = 1;
            else              a = TMath::Sqrt(a + nip);            
            b = b / a;
            b = TMath::Exp(b);
            spy = spy + b;
            if (j - l + 1 < ymin) a = source[i][ymin] / maxch;
            else                  a = source[i][j - l + 1] / maxch;
            b = a - nim;
            if (a + nim <= 0) a = 1;
            else              a = TMath::Sqrt(a + nim);            
            b = b / a;
            b = TMath::Exp(b);
            smy = smy + b;
         }
         a = (spx * working_space[i][j + 1] + spy * working_space[i + 1][j]) / (smx +smy);
         working_space[i + 1][j + 1] = a;
         nom = nom + a;
      }
   }
   for(i = xmin; i <= xmax; i++){
      for(j = ymin; j <= ymax; j++){
         working_space[i][j] = working_space[i][j] / nom;
      }
   }
   for(i = 0;i < ssizex; i++){
      for(j = 0; j < ssizey; j++){
         source[i][j] = plocha * working_space[i][j];
      }
   }
   for (i = 0; i < ssizex; i++)
      delete[]working_space[i];
   delete[]working_space;   
   return 0;
}

//______________________________________________________________________________________________________________________________
const char *TSpectrum2::Deconvolution(float **source, float **resp,
                                       Int_t ssizex, Int_t ssizey,
                                       Int_t numberIterations, 
                                       Int_t numberRepetitions,
                                       Double_t boost)                                        
{   
/////////////////////////////////////////////////////////////////////////////
//   TWO-DIMENSIONAL DECONVOLUTION FUNCTION
//   This function calculates deconvolution from source spectrum
//   according to response spectrum
//   The result is placed in the matrix pointed by source pointer.
//
//   Function parameters:
//   source-pointer to the matrix of source spectrum
//   resp-pointer to the matrix of response spectrum
//   ssizex-x length of source and response spectra
//   ssizey-y length of source and response spectra
//   numberIterations, for details we refer to manual
//   numberRepetitions, for details we refer to manual
//   boost, boosting factor, for details we refer to manual
//
/////////////////////////////////////////////////////////////////////////////
//Begin_Html <!--
/* -->
<div class=Section1>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:20.0pt'>Deconvolution</span></b></p>

<p class=MsoNormal style='text-align:justify'><i><span style='font-size:18.0pt'>&nbsp;</span></i></p>

<p class=MsoNormal style='text-align:justify'><i><span style='font-size:18.0pt'>Goal:
Improvement of the resolution in spectra, decomposition of multiplets</span></i></p>

<p class=MsoNormal><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal><span style='font-size:16.0pt'>Mathematical formulation of
the 2-dimensional convolution system is</span></p>

<p class=MsoNormal style='margin-left:18.0pt'>

<table cellpadding=0 cellspacing=0 align=left>
 <tr>
  <td width=0 height=18></td>
 </tr>
 <tr>
  <td></td>
  <td><img width=577 height=138 src="gif/TSpectrum2_Deconvolution1.gif"></td>
 </tr>
</table>

<span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<br clear=ALL>

<p class=MsoNormal><span style='font-size:16.0pt'>where h(i,j) is the impulse
response function, x, y are input and output matrices, respectively, <sub><img
width=45 height=24 src="gif/TSpectrum2_Deconvolution2.gif"></sub> are the lengths
of x and h matrices</span><i><span style='font-size:18.0pt'> </span></i></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>let us assume that we know the
response and the output matrices (spectra) of the above given system. </span></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>the deconvolution represents
solution of the overdetermined system of linear equations, i.e.,  the
calculation of the matrix <b>x.</b></span></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>from numerical stability point of
view the operation of deconvolution is extremely critical (ill-posed  problem)
as well as time consuming operation. </span></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>the Gold deconvolution algorithm
proves to work very well even for 2-dimensional systems. Generalization of the
algorithm for 2-dimensional systems was presented in [1], [2].</span></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>for Gold deconvolution algorithm
as well as for boosted deconvolution algorithm we refer also to TSpectrum </span></p>

<p class=MsoNormal><i><span style='font-size:18.0pt'>&nbsp;</span></i></p>

<p class=MsoNormal><i><span style='font-size:18.0pt'>Function:</span></i></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:18.0pt'>const
<a href="http://root.cern.ch/root/html/ListOfTypes.html#char">char</a>* </span></b><a
name="TSpectrum:Deconvolution1"></a><a
href="http://root.cern.ch/root/html/TSpectrum.html#TSpectrum:Fit1Awmi"><b><span
style='font-size:18.0pt'>TSpectrum2::Deconvolution</span></b></a><b><span
style='font-size:18.0pt'>(<a
href="http://root.cern.ch/root/html/ListOfTypes.html#float">float</a> **source,
const <a href="http://root.cern.ch/root/html/ListOfTypes.html#float">float</a>
**resp, <a href="http://root.cern.ch/root/html/ListOfTypes.html#int">int</a> ssizex,
<a href="http://root.cern.ch/root/html/ListOfTypes.html#int">int</a> ssizey, <a
href="http://root.cern.ch/root/html/ListOfTypes.html#int">int</a> numberIterations,
<a href="http://root.cern.ch/root/html/ListOfTypes.html#int">int</a> numberRepetitions,
<a href="http://root.cern.ch/root/html/ListOfTypes.html#double">double</a></span></b><b><span
style='font-size:16.0pt'> </span></b><b><span style='font-size:18.0pt'>boost)</span></b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>This
function calculates deconvolution from source spectrum according to response
spectrum using Gold deconvolution algorithm. The result is placed in the matrix
pointed by source pointer. On successful completion it returns 0. On error it
returns pointer to the string describing error. If desired after every
numberIterations one can apply boosting operation (exponential function with
exponent given by boost coefficient) and repeat it numberRepetitions times.</span></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><i><span style='font-size:16.0pt;color:red'>Parameters:</span></i></p>

<p class=MsoNormal>        <b><span style='font-size:14.0pt'>source</span></b>-pointer
to the matrix of source spectrum                  </p>

<p class=MsoNormal>        <b><span style='font-size:14.0pt'>resp</span></b>-pointer
to the matrix of response spectrum                  </p>

<p class=MsoNormal>        <b><span style='font-size:14.0pt'>ssizex, ssizey</span></b>-lengths
of the spectrum matrix                                 </p>

<p class=MsoNormal style='text-align:justify'>        <b><span
style='font-size:14.0pt'>numberIterations</span></b>-number of iterations </p>

<p class=MsoNormal style='text-align:justify'>        <b><span
style='font-size:14.0pt'>numberRepetitions</span></b>-number of repetitions
for boosted deconvolution. It must be </p>

<p class=MsoNormal style='text-align:justify'>        greater or equal to one.</p>

<p class=MsoNormal style='text-align:justify'>        <b><span
style='font-size:14.0pt'>boost</span></b>-boosting coefficient, applies only
if numberRepetitions is greater than one.  </p>

<p class=MsoNormal style='text-align:justify'>        <span style='font-size:
14.0pt;color:fuchsia'>Recommended range &lt;1,2&gt;.</span></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><b><i><span style='font-size:18.0pt'>References:</span></i></b></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'> [1]
</span><span lang=SK style='font-size:16.0pt'>M. Morhá&#269;, J. Kliman, V.
Matoušek, M. Veselský, I. Turzo</span><span style='font-size:16.0pt'>.:
Efficient one- and two-dimensional Gold deconvolution and its application to
gamma-ray spectra decomposition. NIM, A401 (1997) 385-408.</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>[2]
Morhá&#269; M., Matoušek V., Kliman J., Efficient algorithm of multidimensional
deconvolution and its application to nuclear data processing, Digital Signal
Processing 13 (2003) 144. </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

</div>

<!-- */
// --> End_Html
//Begin_Html <!--
/* -->
<div class=Section1>

<p class=MsoNormal><i><span style='font-size:16.0pt'>Example 5 – script Decon.c
:</span></i></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>response function (usually peak)
should be shifted to the beginning of the coordinate system (see Fig. 13)</span></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><img width=602 height=418
src="gif/TSpectrum2_Deconvolution1.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Fig.
13 2-dimensional response spectrum</span></b></p>

<p class=MsoNormal><img width=602 height=418
src="gif/TSpectrum2_Deconvolution2.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Fig.
14 2-dimensional gamma-gamma-ray input spectrum (before deconvolution)</span></b></p>

<p class=MsoNormal><img width=602 height=418
src="gif/TSpectrum2_Deconvolution3.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Fig.
15 Spectrum from Fig. 14 after deconvolution (1000 iterations)</span></b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:#339966'>Script:</span></b></p>

<p class=MsoNormal>// Example to illustrate the Gold deconvolution (class
TSpectrum2).</p>

<p class=MsoNormal>// To execute this example, do</p>

<p class=MsoNormal>// root &gt; .x Decon.C</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>#include &lt;TSpectrum2&gt; </p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>void Decon() {</p>

<p class=MsoNormal>   Int_t i, j;</p>

<p class=MsoNormal>   Double_t nbinsx = 256;</p>

<p class=MsoNormal>   Double_t nbinsy = 256;   </p>

<p class=MsoNormal>   Double_t xmin  = 0;</p>

<p class=MsoNormal>   Double_t xmax  = (Double_t)nbinsx;</p>

<p class=MsoNormal>   Double_t ymin  = 0;</p>

<p class=MsoNormal>   Double_t ymax  = (Double_t)nbinsy;   </p>

<p class=MsoNormal>   Float_t ** source = new float *[nbinsx];   </p>

<p class=MsoNormal>   for (i=0;i&lt;nbinsx;i++)</p>

<p class=MsoNormal>                                    source[i]=new
float[nbinsy];     </p>

<p class=MsoNormal>   TH2F *decon = new TH2F(&quot;decon&quot;,&quot;Gold
deconvolution&quot;,nbinsx,xmin,xmax,nbinsy,ymin,ymax);</p>

<p class=MsoNormal>   TFile *f = new
TFile(&quot;spectra2\\TSpectrum2.root&quot;);</p>

<p class=MsoNormal>   decon=(TH2F*) f-&gt;Get(&quot;decon1;1&quot;);</p>

<p class=MsoNormal>   Float_t ** response = new float *[nbinsx];   </p>

<p class=MsoNormal>   for (i=0;i&lt;nbinsx;i++)</p>

<p class=MsoNormal>                                    response[i]=new
float[nbinsy];     </p>

<p class=MsoNormal>   TH2F *resp = new TH2F(&quot;resp&quot;,&quot;Response
matrix&quot;,nbinsx,xmin,xmax,nbinsy,ymin,ymax);</p>

<p class=MsoNormal>   resp=(TH2F*) f-&gt;Get(&quot;resp1;1&quot;);   </p>

<p class=MsoNormal>   TCanvas *Deconvol = new
TCanvas(&quot;Deconvolution&quot;,&quot;Gold deconvolution&quot;,10,10,1000,700);</p>

<p class=MsoNormal>   TSpectrum *s = new TSpectrum();</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbinsx; i++){</p>

<p class=MsoNormal>     for (j = 0; j &lt; nbinsy; j++){</p>

<p class=MsoNormal>                source[i][j] = decon-&gt;GetBinContent(i +
1,j + 1); </p>

<p class=MsoNormal>             }</p>

<p class=MsoNormal>   }</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbinsx; i++){</p>

<p class=MsoNormal>     for (j = 0; j &lt; nbinsy; j++){</p>

<p class=MsoNormal>                response[i][j] = resp-&gt;GetBinContent(i +
1,j + 1); </p>

<p class=MsoNormal>             }</p>

<p class=MsoNormal>   }   </p>

<p class=MsoNormal>   s-&gt;Deconvolution(source,response,nbinsx,nbinsy,1000,1,1);  
</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbinsx; i++){</p>

<p class=MsoNormal>     for (j = 0; j &lt; nbinsy; j++)</p>

<p class=MsoNormal>       decon-&gt;SetBinContent(i + 1,j + 1, source[i][j]);  
</p>

<p class=MsoNormal>   }</p>

<p class=MsoNormal>   </p>

<p class=MsoNormal>   decon-&gt;Draw(&quot;SURF&quot;);  </p>

<p class=MsoNormal>   }</p>

<p class=MsoNormal><i><span style='font-size:16.0pt'>Example 6 – script
Decon2.c :</span></i></p>

<p class=MsoNormal><img width=602 height=418
src="gif/TSpectrum2_Deconvolution4.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Fig.
16 Response spectrum</span></b></p>

<p class=MsoNormal><img width=602 height=418
src="gif/TSpectrum2_Deconvolution5.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Fig.
17 Original synthetic input spectrum (before deconvolution). It is composed of
17 peaks. 5 peaks are overlapping in the outlined multiplet and two peaks in
doublet.</span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'><img
width=602 height=418 src="gif/TSpectrum2_Deconvolution6.jpg"></span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Fig.
18 Spectrum from Fig. 17 after deconvolution (1000 iterations). Resolution is
improved but the peaks in multiplet remained unresolved.</span></b></p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:#339966'>Script:</span></b></p>

<p class=MsoNormal>// Example to illustrate the Gold deconvolution (class
TSpectrum2).</p>

<p class=MsoNormal>// To execute this example, do</p>

<p class=MsoNormal>// root &gt; .x Decon2.C</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>#include &lt;TSpectrum2&gt; </p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>void Decon2() {</p>

<p class=MsoNormal>   Int_t i, j;</p>

<p class=MsoNormal>   Double_t nbinsx = 64;</p>

<p class=MsoNormal>   Double_t nbinsy = 64;   </p>

<p class=MsoNormal>   Double_t xmin  = 0;</p>

<p class=MsoNormal>   Double_t xmax  = (Double_t)nbinsx;</p>

<p class=MsoNormal>   Double_t ymin  = 0;</p>

<p class=MsoNormal>   Double_t ymax  = (Double_t)nbinsy;   </p>

<p class=MsoNormal>   Float_t ** source = new float *[nbinsx];   </p>

<p class=MsoNormal>   for (i=0;i&lt;nbinsx;i++)</p>

<p class=MsoNormal>                                    source[i]=new
float[nbinsy];     </p>

<p class=MsoNormal>   TH2F *decon = new TH2F(&quot;decon&quot;,&quot;Gold
deconvolution&quot;,nbinsx,xmin,xmax,nbinsy,ymin,ymax);</p>

<p class=MsoNormal>   TFile *f = new TFile(&quot;spectra2\\TSpectrum2.root&quot;);</p>

<p class=MsoNormal>   decon=(TH2F*) f-&gt;Get(&quot;decon2;1&quot;);</p>

<p class=MsoNormal>   Float_t ** response = new float *[nbinsx];   </p>

<p class=MsoNormal>   for (i=0;i&lt;nbinsx;i++)</p>

<p class=MsoNormal>                                    response[i]=new
float[nbinsy];     </p>

<p class=MsoNormal>   TH2F *resp = new TH2F(&quot;resp&quot;,&quot;Response
matrix&quot;,nbinsx,xmin,xmax,nbinsy,ymin,ymax);</p>

<p class=MsoNormal>   resp=(TH2F*) f-&gt;Get(&quot;resp2;1&quot;);   </p>

<p class=MsoNormal>   TCanvas *Deconvol = new
TCanvas(&quot;Deconvolution&quot;,&quot;Gold
deconvolution&quot;,10,10,1000,700);</p>

<p class=MsoNormal>   TSpectrum *s = new TSpectrum();</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbinsx; i++){</p>

<p class=MsoNormal>     for (j = 0; j &lt; nbinsy; j++){</p>

<p class=MsoNormal>                source[i][j] = decon-&gt;GetBinContent(i +
1,j + 1); </p>

<p class=MsoNormal>             }</p>

<p class=MsoNormal>   }</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbinsx; i++){</p>

<p class=MsoNormal>     for (j = 0; j &lt; nbinsy; j++){</p>

<p class=MsoNormal>                response[i][j] = resp-&gt;GetBinContent(i +
1,j + 1); </p>

<p class=MsoNormal>             }</p>

<p class=MsoNormal>   }   </p>

<p class=MsoNormal>   s-&gt;Deconvolution(source,response,nbinsx,nbinsy,1000,1,1);  
</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbinsx; i++){</p>

<p class=MsoNormal>     for (j = 0; j &lt; nbinsy; j++)</p>

<p class=MsoNormal>       decon-&gt;SetBinContent(i + 1,j + 1, source[i][j]);  
</p>

<p class=MsoNormal>   }</p>

<p class=MsoNormal>   decon-&gt;Draw(&quot;SURF&quot;);  </p>

<p class=MsoNormal>   }</p>

<p class=MsoNormal><i><span style='font-size:16.0pt'>Example 7 – script
Decon2HR.c :</span></i></p>

<p class=MsoNormal><img width=602 height=418
src="gif/TSpectrum2_Deconvolution7.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Fig.
19 Spectrum from Fig. 17 after boosted deconvolution (50 iterations repeated 20
times, boosting coefficient was 1.2). All the peaks in multiplet as well as in
doublet are completely decomposed.</span></b></p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:#339966'>Script:</span></b></p>

<p class=MsoNormal>// Example to illustrate boosted Gold deconvolution (class
TSpectrum2).</p>

<p class=MsoNormal>// To execute this example, do</p>

<p class=MsoNormal>// root &gt; .x Decon2HR.C</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>//#include &lt;TSpectrum2&gt; </p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>void Decon2HR() {</p>

<p class=MsoNormal>   Int_t i, j;</p>

<p class=MsoNormal>   Double_t nbinsx = 64;</p>

<p class=MsoNormal>   Double_t nbinsy = 64;   </p>

<p class=MsoNormal>   Double_t xmin  = 0;</p>

<p class=MsoNormal>   Double_t xmax  = (Double_t)nbinsx;</p>

<p class=MsoNormal>   Double_t ymin  = 0;</p>

<p class=MsoNormal>   Double_t ymax  = (Double_t)nbinsy;   </p>

<p class=MsoNormal>   Float_t ** source = new float *[nbinsx];   </p>

<p class=MsoNormal>   for (i=0;i&lt;nbinsx;i++)</p>

<p class=MsoNormal>                                    source[i]=new
float[nbinsy];     </p>

<p class=MsoNormal>   TH2F *decon = new TH2F(&quot;decon&quot;,&quot;Boosted
Gold deconvolution&quot;,nbinsx,xmin,xmax,nbinsy,ymin,ymax);</p>

<p class=MsoNormal>   TFile *f = new TFile(&quot;spectra2\\TSpectrum2.root&quot;);</p>

<p class=MsoNormal>   decon=(TH2F*) f-&gt;Get(&quot;decon2;1&quot;);</p>

<p class=MsoNormal>   Float_t ** response = new float *[nbinsx];   </p>

<p class=MsoNormal>   for (i=0;i&lt;nbinsx;i++)</p>

<p class=MsoNormal>                                    response[i]=new
float[nbinsy];     </p>

<p class=MsoNormal>   TH2F *resp = new TH2F(&quot;resp&quot;,&quot;Response
matrix&quot;,nbinsx,xmin,xmax,nbinsy,ymin,ymax);</p>

<p class=MsoNormal>   resp=(TH2F*) f-&gt;Get(&quot;resp2;1&quot;);   </p>

<p class=MsoNormal>   TCanvas *Deconvol = new
TCanvas(&quot;Deconvolution&quot;,&quot;Gold
deconvolution&quot;,10,10,1000,700);</p>

<p class=MsoNormal>   TSpectrum *s = new TSpectrum();</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbinsx; i++){</p>

<p class=MsoNormal>     for (j = 0; j &lt; nbinsy; j++){</p>

<p class=MsoNormal>                source[i][j] = decon-&gt;GetBinContent(i +
1,j + 1); </p>

<p class=MsoNormal>             }</p>

<p class=MsoNormal>   }</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbinsx; i++){</p>

<p class=MsoNormal>     for (j = 0; j &lt; nbinsy; j++){</p>

<p class=MsoNormal>                response[i][j] = resp-&gt;GetBinContent(i +
1,j + 1); </p>

<p class=MsoNormal>             }</p>

<p class=MsoNormal>   }   </p>

<p class=MsoNormal>   s-&gt;Deconvolution(source,response,nbinsx,nbinsy,1000,1,1);  
</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbinsx; i++){</p>

<p class=MsoNormal>     for (j = 0; j &lt; nbinsy; j++)</p>

<p class=MsoNormal>       decon-&gt;SetBinContent(i + 1,j + 1, source[i][j]);  
</p>

<p class=MsoNormal>   }</p>

<p class=MsoNormal>   decon-&gt;Draw(&quot;SURF&quot;);  </p>

<p class=MsoNormal>   }</p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

</div>

<!-- */
// --> End_Html
   int i, j, lhx, lhy, i1, i2, j1, j2, k1, k2, lindex, i1min, i1max,
       i2min, i2max, j1min, j1max, j2min, j2max, positx = 0, posity = 0, repet;
   double lda, ldb, ldc, area, maximum = 0;
   if (ssizex <= 0 || ssizey <= 0)
      return "Wrong parameters";
   if (numberIterations <= 0)
      return "Number of iterations must be positive";
   if (numberRepetitions <= 0)
      return "Number of repetitions must be positive";   
   double **working_space = new double *[ssizex];
   for (i = 0; i < ssizex; i++)
      working_space[i] = new double[5 * ssizey];
   area = 0;
   lhx = -1, lhy = -1;
   for (i = 0; i < ssizex; i++) {
      for (j = 0; j < ssizey; j++) {
         lda = resp[i][j];
         if (lda != 0) {
            if ((i + 1) > lhx)
               lhx = i + 1;
            if ((j + 1) > lhy)
               lhy = j + 1;
         }
         working_space[i][j] = lda;
         area = area + lda;
         if (lda > maximum) {
            maximum = lda;
            positx = i, posity = j;
         }
      }
   }
   if (lhx == -1 || lhy == -1) {
      delete [] working_space;
      return ("Zero response data");
   }
   
//calculate ht*y and write into p 
   for (i2 = 0; i2 < ssizey; i2++) {
      for (i1 = 0; i1 < ssizex; i1++) {
         ldc = 0;
         for (j2 = 0; j2 <= (lhy - 1); j2++) {
            for (j1 = 0; j1 <= (lhx - 1); j1++) {
               k2 = i2 + j2, k1 = i1 + j1;
               if (k2 >= 0 && k2 < ssizey && k1 >= 0 && k1 < ssizex) {
                  lda = working_space[j1][j2];
                  ldb = source[k1][k2];
                  ldc = ldc + lda * ldb;
               }
            }
         }
         working_space[i1][i2 + ssizey] = ldc;
      }
   }
   
//calculate matrix b=ht*h 
   i1min = -(lhx - 1), i1max = lhx - 1;
   i2min = -(lhy - 1), i2max = lhy - 1;
   for (i2 = i2min; i2 <= i2max; i2++) {
      for (i1 = i1min; i1 <= i1max; i1++) {
         ldc = 0;
         j2min = -i2;
         if (j2min < 0)
            j2min = 0;
         j2max = lhy - 1 - i2;
         if (j2max > lhy - 1)
            j2max = lhy - 1;
         for (j2 = j2min; j2 <= j2max; j2++) {
            j1min = -i1;
            if (j1min < 0)
               j1min = 0;
            j1max = lhx - 1 - i1;
            if (j1max > lhx - 1)
               j1max = lhx - 1;
            for (j1 = j1min; j1 <= j1max; j1++) {
               lda = working_space[j1][j2];
               if (i1 + j1 < ssizex && i2 + j2 < ssizey)
                  ldb = working_space[i1 + j1][i2 + j2];
               else
                  ldb = 0;
               ldc = ldc + lda * ldb;
            }
         }
         working_space[i1 - i1min][i2 - i2min + 2 * ssizey ] = ldc;
      }
   }
   
//initialization in x1 matrix 
   for (i2 = 0; i2 < ssizey; i2++) {
      for (i1 = 0; i1 < ssizex; i1++) {
         working_space[i1][i2 + 3 * ssizey] = 1;
         working_space[i1][i2 + 4 * ssizey] = 0;
      }
   }
   
   //START OF ITERATIONS 
   for (repet = 0; repet < numberRepetitions; repet++) {
      if (repet != 0) {
         for (i = 0; i < ssizex; i++) {
            for (j = 0; j < ssizey; j++) {
               working_space[i][j + 3 * ssizey] =
                   TMath::Power(working_space[i][j + 3 * ssizey], boost);
            }
         }
      }
      for (lindex = 0; lindex < numberIterations; lindex++) {
         for (i2 = 0; i2 < ssizey; i2++) {
            for (i1 = 0; i1 < ssizex; i1++) {
               ldb = 0;                 
               j2min = i2;
               if (j2min > lhy - 1)
                  j2min = lhy - 1;
               j2min = -j2min;
               j2max = ssizey - i2 - 1;
               if (j2max > lhy - 1)
                  j2max = lhy - 1;
               j1min = i1;
               if (j1min > lhx - 1)
                  j1min = lhx - 1;
               j1min = -j1min;
               j1max = ssizex - i1 - 1;
               if (j1max > lhx - 1)
                  j1max = lhx - 1;
               for (j2 = j2min; j2 <= j2max; j2++) {
                  for (j1 = j1min; j1 <= j1max; j1++) {
                     ldc =  working_space[j1 - i1min][j2 - i2min + 2 * ssizey];
                     lda = working_space[i1 + j1][i2 + j2 + 3 * ssizey];
                     ldb = ldb + lda * ldc;
                  }
               }
               lda = working_space[i1][i2 + 3 * ssizey];
               ldc = working_space[i1][i2 + 1 * ssizey];
               if (ldc * lda != 0 && ldb != 0) {
                  lda = lda * ldc / ldb;
               }

               else
                  lda = 0;
               working_space[i1][i2 + 4 * ssizey] = lda;
            }
         }
         for (i2 = 0; i2 < ssizey; i2++) {
            for (i1 = 0; i1 < ssizex; i1++)
               working_space[i1][i2 + 3 * ssizey] =
                   working_space[i1][i2 + 4 * ssizey];
         }
      }
   }
   for (i = 0; i < ssizex; i++) {
      for (j = 0; j < ssizey; j++)
         source[(i + positx) % ssizex][(j + posity) % ssizey] =
             area * working_space[i][j + 3 * ssizey];
   }
   for (i = 0; i < ssizex; i++)
      delete[]working_space[i];
   delete[]working_space;
   return 0;
}

//____________________________________________________________________________
Int_t TSpectrum2::SearchHighRes(float **source,float **dest, Int_t ssizex, Int_t ssizey,
                                 Double_t sigma, Double_t threshold,
                                 Bool_t backgroundRemove,Int_t deconIterations,
                                 Bool_t markov, Int_t averWindow)
                                     
{
/////////////////////////////////////////////////////////////////////////////
//   TWO-DIMENSIONAL HIGH-RESOLUTION PEAK SEARCH FUNCTION                  //
//   This function searches for peaks in source spectrum                   //
//      It is based on deconvolution method. First the background is       //
//      removed (if desired), then Markov spectrum is calculated           //
//      (if desired), then the response function is generated              //
//      according to given sigma and deconvolution is carried out.         //
//                                                                         //
//   Function parameters:                                                  //
//   source-pointer to the matrix of source spectrum                       //
//   dest-pointer to the matrix of resulting deconvolved spectrum          //
//   ssizex-x length of source spectrum                                    //
//   ssizey-y length of source spectrum                                    //
//   sigma-sigma of searched peaks, for details we refer to manual         //
//   threshold-threshold value in % for selected peaks, peaks with         //
//                amplitude less than threshold*highest_peak/100           //
//                are ignored, see manual                                  //
//      backgroundRemove-logical variable, set if the removal of           //
//                background before deconvolution is desired               //
//      deconIterations-number of iterations in deconvolution operation    //
//      markov-logical variable, if it is true, first the source spectrum  //
//             is replaced by new spectrum calculated using Markov         //
//             chains method.                                              //
//   averWindow-averanging window of searched peaks, for details           //
//                  we refer to manual (applies only for Markov method)    //
//                                                                         //
/////////////////////////////////////////////////////////////////////////////
//Begin_Html <!--
/* -->
<div class=Section1>

<p class=MsoNormal><b><span style='font-size:20.0pt'>Peaks searching</span></b></p>

<p class=MsoNormal style='text-align:justify'><i><span style='font-size:18.0pt'>&nbsp;</span></i></p>

<p class=MsoNormal style='text-align:justify'><i><span style='font-size:18.0pt'>Goal:
to identify automatically the peaks in spectrum with the presence of the
continuous background, one-fold coincidences (ridges) and statistical
fluctuations - noise.</span></i><span style='font-size:18.0pt'> </span></p>

<p class=MsoNormal><span style='font-size:16.0pt;font-family:Arial'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt;
font-family:Arial'>The common problems connected with correct peak
identification in two-dimensional coincidence spectra are</span></p>

<ul style='margin-top:0mm' type=disc>
 <li class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt;
     font-family:Arial'>non-sensitivity to noise, i.e., only statistically
     relevant peaks should be identified</span></li>
 <li class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt;
     font-family:Arial'>non-sensitivity of the algorithm to continuous
     background</span></li>
 <li class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt;
     font-family:Arial'>non-sensitivity to one-fold coincidences (coincidences
     peak – background in both dimensions) and their crossings</span></li>
 <li class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt;
     font-family:Arial'>ability to identify peaks close to the edges of the
     spectrum region. Usually peak finders fail to detect them</span></li>
 <li class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt;
     font-family:Arial'>resolution, decomposition of doublets and multiplets.
     The algorithm should be able to recognize close positioned peaks.</span><span
     style='font-size:18.0pt'> </span></li>
 <li class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt;
     font-family:Arial'>ability to identify peaks with different sigma</span></li>
</ul>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal><i><span style='font-size:18.0pt'>Function:</span></i></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:18.0pt'><a
href="http://root.cern.ch/root/html/ListOfTypes.html#Int_t">Int_t</a> </span></b><a
name="TSpectrum:Search1HighRes"></a><a
href="http://root.cern.ch/root/html/TSpectrum.html#TSpectrum:Fit1Awmi"><b><span
style='font-size:18.0pt'>TSpectrum2::SearchHighRes</span></b></a><b><span
style='font-size:18.0pt'> (<a
href="http://root.cern.ch/root/html/ListOfTypes.html#float">float</a> **source,<a
href="http://root.cern.ch/root/html/ListOfTypes.html#float">float</a> **dest, <a
href="http://root.cern.ch/root/html/ListOfTypes.html#int">int</a> ssizex, <a
href="http://root.cern.ch/root/html/ListOfTypes.html#int">int</a> ssizey, <a
href="http://root.cern.ch/root/html/ListOfTypes.html#float">float</a> sigma, <a
href="http://root.cern.ch/root/html/ListOfTypes.html#double">double</a> threshold,
<a href="http://root.cern.ch/root/html/ListOfTypes.html#bool">bool</a> backgroundRemove,<a
href="http://root.cern.ch/root/html/ListOfTypes.html#int">int</a> deconIterations,
<a href="http://root.cern.ch/root/html/ListOfTypes.html#bool">bool</a> markov,
<a href="http://root.cern.ch/root/html/ListOfTypes.html#int">int</a> averWindow)
  </span></b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>This
function searches for peaks in source spectrum. It is based on deconvolution
method. First the background is removed (if desired), then Markov smoothed
spectrum is calculated (if desired), then the response function is generated
according to given sigma and deconvolution is carried out. The order of peaks
is arranged according to their heights in the spectrum after background
elimination. The highest peak is the first in the list. On success it returns
number of found peaks.</span></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><i><span style='font-size:16.0pt;color:red'>Parameters:</span></i></p>

<p class=MsoNormal style='text-align:justify'>        <b><span
style='font-size:14.0pt'>source</span></b>-pointer to the matrix of source
spectrum                  </p>

<p class=MsoNormal style='text-align:justify'>        <b><span
style='font-size:14.0pt'>dest</span></b>-resulting spectrum after deconvolution</p>

<p class=MsoNormal style='text-align:justify'>        <b><span
style='font-size:14.0pt'>ssizex, ssizey</span></b>-lengths of the source and
destination spectra                </p>

<p class=MsoNormal style='text-align:justify'>        <b><span
style='font-size:14.0pt'>sigma</span></b>-sigma of searched peaks</p>

<p class=MsoNormal style='margin-left:22.8pt;text-align:justify'><b><span
style='font-size:14.0pt'>threshold</span></b>-<span style='font-size:16.0pt'> </span>threshold
value in % for selected peaks, peaks with amplitude less than
threshold*highest_peak/100 are ignored</p>

<p class=MsoNormal style='margin-left:22.8pt;text-align:justify'><b><span
style='font-size:14.0pt'>backgroundRemove</span></b>-<span style='font-size:
16.0pt'> </span>background_remove-logical variable, true if the removal of
background before deconvolution is desired  </p>

<p class=MsoNormal style='margin-left:22.8pt;text-align:justify'><b><span
style='font-size:14.0pt'>deconIterations</span></b>-number of iterations in
deconvolution operation</p>

<p class=MsoNormal style='margin-left:22.8pt;text-align:justify'><b><span
style='font-size:14.0pt'>markov</span></b>-logical variable, if it is true,
first the source spectrum is replaced by new spectrum calculated using Markov
chains method </p>

<p class=MsoNormal style='margin-left:19.95pt;text-align:justify;text-indent:
2.85pt'><b><span style='font-size:14.0pt'>averWindow</span></b>-width of
averaging smoothing window </p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><b><i><span style='font-size:18.0pt'>References:</span></i></b></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>[1]
M.A. Mariscotti: A method for identification of peaks in the presence of
background and its application to spectrum analysis. NIM 50 (1967), 309-320.</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>[2]
</span><span lang=SK style='font-size:16.0pt'> M. Morhá&#269;, J. Kliman, V.
Matoušek, M. Veselský, I. Turzo</span><span style='font-size:16.0pt'>.:Identification
of peaks in multidimensional coincidence gamma-ray spectra. NIM, A443 (2000)
108-125.</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>[3]
Z.K. Silagadze, A new algorithm for automatic photopeak searches. NIM A 376
(1996), 451.</span></p>

</div>

<!-- */
// --> End_Html
//Begin_Html <!--
/* -->
<div class=Section1>

<p class=MsoNormal><b><span style='font-size:18.0pt'>Examples of peak searching
method</span></b></p>

<p class=MsoNormal><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'><a
href="http://root.cern.ch/root/html/src/TSpectrum.cxx.html#TSpectrum:Search1HighRes"
target="_parent">SearchHighRes</a> function provides users with the possibility
to vary the input parameters and with the access to the output deconvolved data
in the destination spectrum. Based on the output data one can tune the
parameters. </span></p>

<p class=MsoNormal><i><span style='font-size:16.0pt'>Example 8 – script Src.c:</span></i></p>

<p class=MsoNormal><span style='font-size:16.0pt'><img border=0 width=602
height=455 src="gif/TSpectrum2_Searching1.jpg"></span></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Fig.
20 Two-dimensional spectrum with found peaks denoted by markers (<sub><img
border=0 width=40 height=19 src="gif/TSpectrum2_Searching2.gif"></sub>,
threshold=5%, 3 iterations steps in the deconvolution)</span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'><img
border=0 width=602 height=455 src="gif/TSpectrum2_Searching3.jpg"></span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Fig.
21 Spectrum from Fig. 20 after background elimination and deconvolution</span></b></p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:#339966'>Script:</span></b></p>

<p class=MsoNormal>// Example to illustrate high resolution peak searching
function (class TSpectrum).</p>

<p class=MsoNormal>// To execute this example, do</p>

<p class=MsoNormal>// root &gt; .x Src.C</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>#include &lt;TSpectrum2&gt;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>void Src() {</p>

<p class=MsoNormal>   Int_t i, j, nfound;</p>

<p class=MsoNormal>   Double_t nbinsx = 64;</p>

<p class=MsoNormal>   Double_t nbinsy = 64;   </p>

<p class=MsoNormal>   Double_t xmin  = 0;</p>

<p class=MsoNormal>   Double_t xmax  = (Double_t)nbinsx;</p>

<p class=MsoNormal>   Double_t ymin  = 0;</p>

<p class=MsoNormal>   Double_t ymax  = (Double_t)nbinsy;   </p>

<p class=MsoNormal>   Float_t ** source = new float *[nbinsx];   </p>

<p class=MsoNormal>   for (i=0;i&lt;nbinsx;i++)</p>

<p class=MsoNormal>                                    source[i]=new
float[nbinsy];</p>

<p class=MsoNormal>   Float_t ** dest = new float *[nbinsx];   </p>

<p class=MsoNormal>   for (i=0;i&lt;nbinsx;i++)</p>

<p class=MsoNormal>                                    dest[i]=new
float[nbinsy];</p>

<p class=MsoNormal>   TH2F *search = new TH2F(&quot;search&quot;,&quot;High
resolution peak searching&quot;,nbinsx,xmin,xmax,nbinsy,ymin,ymax);</p>

<p class=MsoNormal>   TFile *f = new TFile(&quot;spectra2\\TSpectrum2.root&quot;);</p>

<p class=MsoNormal>   search=(TH2F*) f-&gt;Get(&quot;search4;1&quot;);</p>

<p class=MsoNormal>   TCanvas *Searching = new
TCanvas(&quot;Searching&quot;,&quot;High resolution peak
searching&quot;,10,10,1000,700);</p>

<p class=MsoNormal>   TSpectrum2 *s = new TSpectrum2();</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbinsx; i++){</p>

<p class=MsoNormal>     for (j = 0; j &lt; nbinsy; j++){</p>

<p class=MsoNormal>                source[i][j] = search-&gt;GetBinContent(i +
1,j + 1); </p>

<p class=MsoNormal>             }</p>

<p class=MsoNormal>   }   </p>

<p class=MsoNormal>   nfound = s-&gt;SearchHighRes(source, dest, nbinsx,
nbinsy, 2, 5, kTRUE, 3, kFALSE, 3);   </p>

<p class=MsoNormal>   printf(&quot;Found %d candidate peaks\n&quot;,nfound);</p>

<p class=MsoNormal>   for(i=0;i&lt;nfound;i++)</p>

<p class=MsoNormal>             printf(&quot;posx= %d, posy= %d, value=
%d\n&quot;,(int)(fPositionX[i]+0.5), (int)(fPositionY[i]+0.5),
(int)source[(int)(fPositionX[i]+0.5)][(int)(fPositionY[i]+0.5)]);        </p>

<p class=MsoNormal>}</p>

<p class=MsoNormal><i><span style='font-size:16.0pt'>Example 9 – script Src2.c:</span></i></p>

<p class=MsoNormal><img border=0 width=602 height=455
src="gif/TSpectrum2_Searching4.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Fig.
22 Two-dimensional noisy spectrum with found peaks denoted by markers (<sub><img
border=0 width=40 height=19 src="gif/TSpectrum2_Searching2.gif"></sub>,
threshold=10%, 10 iterations steps in the deconvolution). One can observe that
the algorithm is insensitive to the crossings of one-dimensional ridges. It
identifies only two-cooincidence peaks.</span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'><img
border=0 width=602 height=455 src="gif/TSpectrum2_Searching5.jpg"></span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Fig.
23 Spectrum from Fig. 22 after background elimination and deconvolution</span></b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:#339966'>Script:</span></b></p>

<p class=MsoNormal>// Example to illustrate high resolution peak searching
function (class TSpectrum).</p>

<p class=MsoNormal>// To execute this example, do</p>

<p class=MsoNormal>// root &gt; .x Src2.C</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>#include &lt;TSpectrum2&gt;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>void Src2() {</p>

<p class=MsoNormal>   Int_t i, j, nfound;</p>

<p class=MsoNormal>   Double_t nbinsx = 256;</p>

<p class=MsoNormal>   Double_t nbinsy = 256;   </p>

<p class=MsoNormal>   Double_t xmin  = 0;</p>

<p class=MsoNormal>   Double_t xmax  = (Double_t)nbinsx;</p>

<p class=MsoNormal>   Double_t ymin  = 0;</p>

<p class=MsoNormal>   Double_t ymax  = (Double_t)nbinsy;   </p>

<p class=MsoNormal>   Float_t ** source = new float *[nbinsx];   </p>

<p class=MsoNormal>   for (i=0;i&lt;nbinsx;i++)</p>

<p class=MsoNormal>                                    source[i]=new
float[nbinsy];</p>

<p class=MsoNormal>   Float_t ** dest = new float *[nbinsx];   </p>

<p class=MsoNormal>   for (i=0;i&lt;nbinsx;i++)</p>

<p class=MsoNormal>                                    dest[i]=new
float[nbinsy];</p>

<p class=MsoNormal>   TH2F *search = new TH2F(&quot;search&quot;,&quot;High
resolution peak searching&quot;,nbinsx,xmin,xmax,nbinsy,ymin,ymax);</p>

<p class=MsoNormal>   TFile *f = new
TFile(&quot;spectra2\\TSpectrum2.root&quot;);</p>

<p class=MsoNormal>   search=(TH2F*) f-&gt;Get(&quot;back3;1&quot;);</p>

<p class=MsoNormal>   TCanvas *Searching = new
TCanvas(&quot;Searching&quot;,&quot;High resolution peak
searching&quot;,10,10,1000,700);</p>

<p class=MsoNormal>   TSpectrum2 *s = new TSpectrum2();</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbinsx; i++){</p>

<p class=MsoNormal>     for (j = 0; j &lt; nbinsy; j++){</p>

<p class=MsoNormal>                source[i][j] = search-&gt;GetBinContent(i +
1,j + 1); </p>

<p class=MsoNormal>             }</p>

<p class=MsoNormal>   }   </p>

<p class=MsoNormal>   nfound = s-&gt;SearchHighRes(source, dest, nbinsx,
nbinsy, 2, 10, kTRUE, 10, kFALSE, 3);   </p>

<p class=MsoNormal>   printf(&quot;Found %d candidate peaks\n&quot;,nfound);</p>

<p class=MsoNormal>   for(i=0;i&lt;nfound;i++)</p>

<p class=MsoNormal>             printf(&quot;posx= %d, posy= %d, value=
%d\n&quot;,(int)(fPositionX[i]+0.5), (int)(fPositionY[i]+0.5),
(int)source[(int)(fPositionX[i]+0.5)][(int)(fPositionY[i]+0.5)]);        </p>

<p class=MsoNormal>}</p>

<p class=MsoNormal><i><span style='font-size:16.0pt'>Example 10 – script Src3.c:</span></i></p>

<p class=MsoNormal><img border=0 width=602 height=455
src="gif/TSpectrum2_Searching6.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Fig.
24 Two-dimensional spectrum with 15 found peaks denoted by markers. Some peaks
are positioned close to each other. It is necessary to increase number of
iterations in the deconvolution. In next 3 Figs. we shall study the influence
of this parameter.</span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'><img
border=0 width=602 height=455 src="gif/TSpectrum2_Searching7.jpg"></span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Fig.
25 Spectrum from Fig. 24 after deconvolution (# of iterations = 3). Number of
identified peaks = 13.</span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'><img
border=0 width=602 height=455 src="gif/TSpectrum2_Searching8.jpg"></span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Fig.
26 Spectrum from Fig. 24 after deconvolution (# of iterations = 10). Number of
identified peaks = 13.</span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'><img
border=0 width=602 height=455 src="gif/TSpectrum2_Searching9.jpg"></span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Fig.
27 Spectrum from Fig. 24 after deconvolution (# of iterations = 100). Number of
identified peaks = 15. Now the algorithm is able to decompose two doublets in
the spectrum.</span></b></p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:#339966'>&nbsp;</span></b></p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:#339966'>Script:</span></b></p>

<p class=MsoNormal>// Example to illustrate high resolution peak searching
function (class TSpectrum).</p>

<p class=MsoNormal>// To execute this example, do</p>

<p class=MsoNormal>// root &gt; .x Src3.C</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>#include &lt;TSpectrum2&gt;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>void Src3() {</p>

<p class=MsoNormal>   Int_t i, j, nfound;</p>

<p class=MsoNormal>   Double_t nbinsx = 64;</p>

<p class=MsoNormal>   Double_t nbinsy = 64;   </p>

<p class=MsoNormal>   Double_t xmin  = 0;</p>

<p class=MsoNormal>   Double_t xmax  = (Double_t)nbinsx;</p>

<p class=MsoNormal>   Double_t ymin  = 0;</p>

<p class=MsoNormal>   Double_t ymax  = (Double_t)nbinsy;   </p>

<p class=MsoNormal>   Float_t ** source = new float *[nbinsx];   </p>

<p class=MsoNormal>   for (i=0;i&lt;nbinsx;i++)</p>

<p class=MsoNormal>                                    source[i]=new
float[nbinsy];</p>

<p class=MsoNormal>   Float_t ** dest = new float *[nbinsx];   </p>

<p class=MsoNormal>   for (i=0;i&lt;nbinsx;i++)</p>

<p class=MsoNormal>                                    dest[i]=new
float[nbinsy];</p>

<p class=MsoNormal>   TH2F *search = new TH2F(&quot;search&quot;,&quot;High
resolution peak searching&quot;,nbinsx,xmin,xmax,nbinsy,ymin,ymax);</p>

<p class=MsoNormal>   TFile *f = new
TFile(&quot;spectra2\\TSpectrum2.root&quot;);</p>

<p class=MsoNormal>   search=(TH2F*) f-&gt;Get(&quot;search1;1&quot;);</p>

<p class=MsoNormal>   TCanvas *Searching = new
TCanvas(&quot;Searching&quot;,&quot;High resolution peak
searching&quot;,10,10,1000,700);</p>

<p class=MsoNormal>   TSpectrum2 *s = new TSpectrum2();</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbinsx; i++){</p>

<p class=MsoNormal>     for (j = 0; j &lt; nbinsy; j++){</p>

<p class=MsoNormal>                source[i][j] = search-&gt;GetBinContent(i +
1,j + 1); </p>

<p class=MsoNormal>             }</p>

<p class=MsoNormal>   }   </p>

<p class=MsoNormal>   nfound = s-&gt;SearchHighRes(source, dest, nbinsx,
nbinsy, 2, 2, kFALSE, 3, kFALSE, 1);//3, 10, 100   </p>

<p class=MsoNormal>   printf(&quot;Found %d candidate peaks\n&quot;,nfound);</p>

<p class=MsoNormal> </p>

<p class=MsoNormal>   for(i=0;i&lt;nfound;i++)</p>

<p class=MsoNormal>             printf(&quot;posx= %d, posy= %d, value=
%d\n&quot;,(int)(fPositionX[i]+0.5), (int)(fPositionY[i]+0.5),
(int)source[(int)(fPositionX[i]+0.5)][(int)(fPositionY[i]+0.5)]);        </p>

<p class=MsoNormal>}</p>

<p class=MsoNormal><i><span style='font-size:16.0pt'>Example 11 – script Src4.c:</span></i></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'><img
border=0 width=602 height=455 src="gif/TSpectrum2_Searching10.jpg"></span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Fig.
28 Two-dimensional spectrum with peaks with different sigma denoted by markers (<sub><img
border=0 width=39 height=19 src="gif/TSpectrum2_Searching11.gif"></sub>,
threshold=5%, 10 iterations steps in the deconvolution, Markov smoothing with
window=3)</span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'><img
border=0 width=602 height=455 src="gif/TSpectrum2_Searching12.jpg"></span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Fig.
29 Spectrum from Fig. 28 after smoothing and deconvolution.</span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>&nbsp;</span></b></p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:#339966'>Script:</span></b></p>

<p class=MsoNormal>// Example to illustrate high resolution peak searching
function (class TSpectrum).</p>

<p class=MsoNormal>// To execute this example, do</p>

<p class=MsoNormal>// root &gt; .x Src4.C</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>#include &lt;TSpectrum2&gt;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>void Src4() {</p>

<p class=MsoNormal>   Int_t i, j, nfound;</p>

<p class=MsoNormal>   Double_t nbinsx = 64;</p>

<p class=MsoNormal>   Double_t nbinsy = 64;   </p>

<p class=MsoNormal>   Double_t xmin  = 0;</p>

<p class=MsoNormal>   Double_t xmax  = (Double_t)nbinsx;</p>

<p class=MsoNormal>   Double_t ymin  = 0;</p>

<p class=MsoNormal>   Double_t ymax  = (Double_t)nbinsy;   </p>

<p class=MsoNormal>   Float_t ** source = new float *[nbinsx];   </p>

<p class=MsoNormal>   for (i=0;i&lt;nbinsx;i++)</p>

<p class=MsoNormal>                                    source[i]=new
float[nbinsy];</p>

<p class=MsoNormal>   Float_t ** dest = new float *[nbinsx];   </p>

<p class=MsoNormal>   for (i=0;i&lt;nbinsx;i++)</p>

<p class=MsoNormal>                                    dest[i]=new
float[nbinsy];</p>

<p class=MsoNormal>   TH2F *search = new TH2F(&quot;search&quot;,&quot;High
resolution peak searching&quot;,nbinsx,xmin,xmax,nbinsy,ymin,ymax);</p>

<p class=MsoNormal>   TFile *f = new
TFile(&quot;spectra2\\TSpectrum2.root&quot;);</p>

<p class=MsoNormal>   search=(TH2F*) f-&gt;Get(&quot;search2;1&quot;);</p>

<p class=MsoNormal>   TCanvas *Searching = new
TCanvas(&quot;Searching&quot;,&quot;High resolution peak
searching&quot;,10,10,1000,700);</p>

<p class=MsoNormal>   TSpectrum2 *s = new TSpectrum2();</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbinsx; i++){</p>

<p class=MsoNormal>     for (j = 0; j &lt; nbinsy; j++){</p>

<p class=MsoNormal>                source[i][j] = search-&gt;GetBinContent(i +
1,j + 1); </p>

<p class=MsoNormal>             }</p>

<p class=MsoNormal>   }   </p>

<p class=MsoNormal>   nfound = s-&gt;SearchHighRes(source, dest, nbinsx,
nbinsy, 3, 5, kFALSE, 10, kTRUE, 3);   </p>

<p class=MsoNormal>   printf(&quot;Found %d candidate peaks\n&quot;,nfound);</p>

<p class=MsoNormal>   for(i=0;i&lt;nfound;i++)</p>

<p class=MsoNormal>             printf(&quot;posx= %d, posy= %d, value=
%d\n&quot;,(int)(fPositionX[i]+0.5), (int)(fPositionY[i]+0.5),
(int)source[(int)(fPositionX[i]+0.5)][(int)(fPositionY[i]+0.5)]);        </p>

<p class=MsoNormal>}</p>

<p class=MsoNormal><i><span style='font-size:16.0pt'>Example 12 – script Src5.c:</span></i></p>

<p class=MsoNormal><img border=0 width=602 height=455
src="gif/TSpectrum2_Searching13.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Fig.
30 Two-dimensional spectrum with peaks positioned close to the edges denoted by
markers (<sub><img border=0 width=40 height=19
src="gif/TSpectrum2_Searching2.gif"></sub>, threshold=5%, 10 iterations
steps in the deconvolution)</span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'><img
border=0 width=602 height=455 src="gif/TSpectrum2_Searching14.jpg"></span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Fig.
31 Spectrum from Fig. 30 after deconvolution.</span></b></p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:#339966'>&nbsp;</span></b></p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:#339966'>Script:</span></b></p>

<p class=MsoNormal>// Example to illustrate high resolution peak searching
function (class TSpectrum).</p>

<p class=MsoNormal>// To execute this example, do</p>

<p class=MsoNormal>// root &gt; .x Src5.C</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>#include &lt;TSpectrum2&gt;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>void Src5() {</p>

<p class=MsoNormal>   Int_t i, j, nfound;</p>

<p class=MsoNormal>   Double_t nbinsx = 64;</p>

<p class=MsoNormal>   Double_t nbinsy = 64;   </p>

<p class=MsoNormal>   Double_t xmin  = 0;</p>

<p class=MsoNormal>   Double_t xmax  = (Double_t)nbinsx;</p>

<p class=MsoNormal>   Double_t ymin  = 0;</p>

<p class=MsoNormal>   Double_t ymax  = (Double_t)nbinsy;   </p>

<p class=MsoNormal>   Float_t ** source = new float *[nbinsx];   </p>

<p class=MsoNormal>   for (i=0;i&lt;nbinsx;i++)</p>

<p class=MsoNormal>                                    source[i]=new
float[nbinsy];</p>

<p class=MsoNormal>   Float_t ** dest = new float *[nbinsx];   </p>

<p class=MsoNormal>   for (i=0;i&lt;nbinsx;i++)</p>

<p class=MsoNormal>                                    dest[i]=new
float[nbinsy];</p>

<p class=MsoNormal>   TH2F *search = new TH2F(&quot;search&quot;,&quot;High
resolution peak searching&quot;,nbinsx,xmin,xmax,nbinsy,ymin,ymax);</p>

<p class=MsoNormal>   TFile *f = new
TFile(&quot;spectra2\\TSpectrum2.root&quot;);</p>

<p class=MsoNormal>   search=(TH2F*) f-&gt;Get(&quot;search3;1&quot;);</p>

<p class=MsoNormal>   TCanvas *Searching = new
TCanvas(&quot;Searching&quot;,&quot;High resolution peak
searching&quot;,10,10,1000,700);</p>

<p class=MsoNormal>   TSpectrum2 *s = new TSpectrum2();</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbinsx; i++){</p>

<p class=MsoNormal>     for (j = 0; j &lt; nbinsy; j++){</p>

<p class=MsoNormal>                source[i][j] = search-&gt;GetBinContent(i +
1,j + 1); </p>

<p class=MsoNormal>             }</p>

<p class=MsoNormal>   }   </p>

<p class=MsoNormal>   nfound = s-&gt;SearchHighRes(source, dest, nbinsx,
nbinsy, 2, 5, kFALSE, 10, kFALSE, 1);   </p>

<p class=MsoNormal>   printf(&quot;Found %d candidate peaks\n&quot;,nfound);</p>

<p class=MsoNormal>   for(i=0;i&lt;nfound;i++)</p>

<p class=MsoNormal>             printf(&quot;posx= %d, posy= %d, value=
%d\n&quot;,(int)(fPositionX[i]+0.5), (int)(fPositionY[i]+0.5),
(int)source[(int)(fPositionX[i]+0.5)][(int)(fPositionY[i]+0.5)]);        </p>

<p class=MsoNormal>}</p>

</div>

<!-- */
// --> End_Html
   int number_of_iterations = (int)(4 * sigma + 0.5);
   int k, lindex, priz;
   double lda, ldb, ldc, area, maximum;
   int xmin, xmax, l, peak_index = 0, ssizex_ext = ssizex + 4 * number_of_iterations, ssizey_ext = ssizey + 4 * number_of_iterations, shift = 2 * number_of_iterations;
   int ymin, ymax, i, j;
   double a, b, ax, ay, maxch, plocha = 0;
   double nom, nip, nim, sp, sm, spx, spy, smx, smy;
   double p1, p2, p3, p4, s1, s2, s3, s4;
   int x, y;
   int lhx, lhy, i1, i2, j1, j2, k1, k2, i1min, i1max, i2min, i2max, j1min, j1max, j2min, j2max, positx, posity;
   if (sigma < 1) {
      Error("SearchHighRes", "Invalid sigma, must be greater than or equal to 1");
      return 0;
   }
 
   if(threshold<=0||threshold>=100){
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
      if(ssizex_ext < 2 * number_of_iterations + 1 || ssizey_ext < 2 * number_of_iterations + 1){
         Error("SearchHighRes", "Too large clipping window");
         return 0;
      }
   }   
   i = (int)(4 * sigma + 0.5);
   i = 4 * i;
   double **working_space = new double *[ssizex + i];
   for (j = 0; j < ssizex + i; j++) {
      Double_t *wsk = working_space[j] = new double[16 * (ssizey + i)];
      for (k=0;k<16 * (ssizey + i);k++) wsk[k] = 0;
   }   
   for(j = 0; j < ssizey_ext; j++){
      for(i = 0; i < ssizex_ext; i++){
         if(i < shift){
            if(j < shift)
                  working_space[i][j + ssizey_ext] = source[0][0];
                  
            else if(j >= ssizey + shift)
                  working_space[i][j + ssizey_ext] = source[0][ssizey - 1];
                  
            else
                  working_space[i][j + ssizey_ext] = source[0][j - shift];
         }
         
         else if(i >= ssizex + shift){
            if(j < shift)
               working_space[i][j + ssizey_ext] = source[ssizex - 1][0];
               
            else if(j >= ssizey + shift)
               working_space[i][j + ssizey_ext] = source[ssizex - 1][ssizey - 1];
               
            else
               working_space[i][j + ssizey_ext] = source[ssizex - 1][j - shift];
         }
         
         else{
            if(j < shift)
               working_space[i][j + ssizey_ext] = source[i - shift][0];
               
            else if(j >= ssizey + shift)
               working_space[i][j + ssizey_ext] = source[i - shift][ssizey - 1];
               
            else
               working_space[i][j + ssizey_ext] = source[i - shift][j - shift];
         }
      }
   }
   if(backgroundRemove == true){
      for(i = 1; i <= number_of_iterations; i++){
         for(y = i; y < ssizey_ext - i; y++){
            for(x = i; x < ssizex_ext - i; x++){
               a = working_space[x][y + ssizey_ext];
               p1 = working_space[x - i][y + ssizey_ext - i];
               p2 = working_space[x - i][y + ssizey_ext + i];
               p3 = working_space[x + i][y + ssizey_ext - i];
               p4 = working_space[x + i][y + ssizey_ext + i];
               s1 = working_space[x][y + ssizey_ext - i];
               s2 = working_space[x - i][y + ssizey_ext];
               s3 = working_space[x + i][y + ssizey_ext];
               s4 = working_space[x][y + ssizey_ext + i];
               b = (p1 + p2) / 2.0;
               if(b > s2)
                  s2 = b;
               b = (p1 + p3) / 2.0;
               if(b > s1)
                  s1 = b;
               b = (p2 + p4) / 2.0;
               if(b > s4)
                  s4 = b;
               b = (p3 + p4) / 2.0;
               if(b > s3)
                  s3 = b;
               s1 = s1 - (p1 + p3) / 2.0;
               s2 = s2 - (p1 + p2) / 2.0;
               s3 = s3 - (p3 + p4) / 2.0;
               s4 = s4 - (p2 + p4) / 2.0;
               b = (s1 + s4) / 2.0 + (s2 + s3) / 2.0 + (p1 + p2 + p3 + p4) / 4.0;
               if(b < a)
                  a = b;
               working_space[x][y] = a;
            }
         }
         for(y = i;y < ssizey_ext - i; y++){
            for(x = i; x < ssizex_ext - i; x++){
               working_space[x][y + ssizey_ext] = working_space[x][y];
            }
         }
      }
      for(j = 0;j < ssizey_ext; j++){
         for(i = 0; i < ssizex_ext; i++){
            if(i < shift){
               if(j < shift)
                  working_space[i][j + ssizey_ext] = source[0][0] - working_space[i][j + ssizey_ext];

               else if(j >= ssizey + shift)
                  working_space[i][j + ssizey_ext] = source[0][ssizey - 1] - working_space[i][j + ssizey_ext];

               else
                  working_space[i][j + ssizey_ext] = source[0][j - shift] - working_space[i][j + ssizey_ext];
            }

            else if(i >= ssizex + shift){
               if(j < shift)
                  working_space[i][j + ssizey_ext] = source[ssizex - 1][0] - working_space[i][j + ssizey_ext];

               else if(j >= ssizey + shift)
                  working_space[i][j + ssizey_ext] = source[ssizex - 1][ssizey - 1] - working_space[i][j + ssizey_ext];

               else
                  working_space[i][j + ssizey_ext] = source[ssizex - 1][j - shift] - working_space[i][j + ssizey_ext];
            }

            else{
               if(j < shift)
                  working_space[i][j + ssizey_ext] = source[i - shift][0] - working_space[i][j + ssizey_ext];

               else if(j >= ssizey + shift)
                  working_space[i][j + ssizey_ext] = source[i - shift][ssizey - 1] - working_space[i][j + ssizey_ext];

               else
                  working_space[i][j + ssizey_ext] = source[i - shift][j - shift] - working_space[i][j + ssizey_ext];
            }
         }
      }
   }
   for(j = 0; j < ssizey_ext; j++){
      for(i = 0; i < ssizex_ext; i++){
         working_space[i][j + 15*ssizey_ext] = working_space[i][j + ssizey_ext];
      }
   }
   if(markov == true){
      for(i = 0;i < ssizex_ext; i++){
         for(j = 0; j < ssizey_ext; j++)
         working_space[i][j + 2 * ssizex_ext] = working_space[i][ssizey_ext + j];
      }
      xmin = 0;
      xmax = ssizex_ext - 1;
      ymin = 0;
      ymax = ssizey_ext - 1;
      for(i = 0, maxch = 0; i < ssizex_ext; i++){
         for(j = 0; j < ssizey_ext; j++){
            working_space[i][j] = 0;
            if(maxch < working_space[i][j + 2 * ssizey_ext])
               maxch = working_space[i][j + 2 * ssizey_ext];
            plocha += working_space[i][j + 2 * ssizey_ext];
         }
      }
      if(maxch == 0) {
         delete [] working_space;
         return 0;
      }

      nom=0;
      working_space[xmin][ymin] = 1;
      for(i = xmin; i < xmax; i++){
         nip = working_space[i][ymin + 2 * ssizey_ext] / maxch;
         nim = working_space[i + 1][ymin + 2 * ssizey_ext] / maxch;
         sp = 0,sm = 0;
         for(l = 1;l <= averWindow; l++){
            if((i + l) > xmax)
               a = working_space[xmax][ymin + 2 * ssizey_ext] / maxch;

            else
               a = working_space[i + l][ymin + 2 * ssizey_ext] / maxch;

            b = a - nip;
            if(a + nip <= 0)
               a = 1;

            else
               a=TMath::Sqrt(a + nip);            
            b = b / a;
            b = TMath::Exp(b);
            sp = sp + b;
            if(i - l + 1 < xmin)
               a = working_space[xmin][ymin + 2 * ssizey_ext] / maxch;

            else
               a = working_space[i - l + 1][ymin + 2 * ssizey_ext] / maxch;
            b = a - nim;
            if(a + nim <= 0)
               a = 1;

            else
               a=TMath::Sqrt(a + nim);                        
            b = b / a;
            b = TMath::Exp(b);
            sm = sm + b;
         }
         a = sp / sm;
         a = working_space[i + 1][ymin] = a * working_space[i][ymin];
         nom = nom + a;
      }
      for(i = ymin; i < ymax; i++){
         nip = working_space[xmin][i + 2 * ssizey_ext] / maxch;
         nim = working_space[xmin][i + 1 + 2 * ssizey_ext] / maxch;
         sp = 0,sm = 0;
         for(l = 1; l <= averWindow; l++){
            if((i + l) > ymax)
               a = working_space[xmin][ymax + 2 * ssizey_ext] / maxch;

            else
               a = working_space[xmin][i + l + 2 * ssizey_ext] / maxch;
            b = a - nip;
            if(a + nip <= 0)
               a=1;

            else
               a=TMath::Sqrt(a + nip);            
            b = b / a;
            b = TMath::Exp(b);
            sp = sp + b;
            if(i - l + 1 < ymin)
               a = working_space[xmin][ymin + 2 * ssizey_ext] / maxch;

            else
               a = working_space[xmin][i - l + 1 + 2 * ssizey_ext] / maxch;
            b = a - nim;
            if(a + nim <= 0)
               a = 1;

            else
               a=TMath::Sqrt(a + nim);            
            b = b / a;
            b = TMath::Exp(b);
            sm = sm + b;
         }
         a = sp / sm;
         a = working_space[xmin][i + 1] = a * working_space[xmin][i];
         nom = nom + a;
      }
      for(i = xmin; i < xmax; i++){
         for(j = ymin; j < ymax; j++){
            nip = working_space[i][j + 1 + 2 * ssizey_ext] / maxch;
            nim = working_space[i + 1][j + 1 + 2 * ssizey_ext] / maxch;
            spx = 0,smx = 0;
            for(l = 1; l <= averWindow; l++){
               if(i + l > xmax)
                  a = working_space[xmax][j + 2 * ssizey_ext] / maxch;

               else
                  a = working_space[i + l][j + 2 * ssizey_ext] / maxch;
               b = a - nip;
               if(a + nip <= 0)
                  a = 1;

               else
                  a=TMath::Sqrt(a + nip);            
               b = b / a;
               b = TMath::Exp(b);
               spx = spx + b;
               if(i - l + 1 < xmin)
                  a = working_space[xmin][j + 2 * ssizey_ext] / maxch;

               else
                  a = working_space[i - l + 1][j + 2 * ssizey_ext] / maxch;
               b = a - nim;
               if(a + nim <= 0)
                  a=1;

               else
                  a=TMath::Sqrt(a + nim);            
               b = b / a;
               b = TMath::Exp(b);
               smx = smx + b;
            }
            spy = 0,smy = 0;
            nip = working_space[i + 1][j + 2 * ssizey_ext] / maxch;
            nim = working_space[i + 1][j + 1 + 2 * ssizey_ext] / maxch;
            for(l = 1; l <= averWindow; l++){
               if(j + l > ymax)
                  a = working_space[i][ymax + 2 * ssizey_ext] / maxch;

               else
                  a = working_space[i][j + l + 2 * ssizey_ext] / maxch;
               b = a - nip;
               if(a + nip <= 0)
                  a = 1;

               else
                  a=TMath::Sqrt(a + nip);            
               b = b / a;
               b = TMath::Exp(b);
               spy = spy + b;
               if(j - l + 1 < ymin)
                  a = working_space[i][ymin + 2 * ssizey_ext] / maxch;

               else
                  a = working_space[i][j - l + 1 + 2 * ssizey_ext] / maxch;
               b=a-nim;
               if(a + nim <= 0)
                  a = 1;
               else
                  a=TMath::Sqrt(a + nim);            
               b = b / a;
               b = TMath::Exp(b);
               smy = smy + b;
            }
            a = (spx * working_space[i][j + 1] + spy * working_space[i + 1][j]) / (smx + smy);
            working_space[i + 1][j + 1] = a;
            nom = nom + a;
         }
      }
      for(i = xmin; i <= xmax; i++){
         for(j = ymin; j <= ymax; j++){
            working_space[i][j] = working_space[i][j] / nom;
         }
      }
      for(i = 0; i < ssizex_ext; i++){
         for(j = 0; j < ssizey_ext; j++){
            working_space[i][j + ssizey_ext] = working_space[i][j] * plocha;
            working_space[i][2 * ssizey_ext + j] = working_space[i][ssizey_ext + j];
         }
      }
   }
   //deconvolution starts
   area = 0;
   lhx = -1,lhy = -1;
   positx = 0,posity = 0;
   maximum = 0;
   //generate response matrix
   for(i = 0; i < ssizex_ext; i++){
      for(j = 0; j < ssizey_ext; j++){
         lda = (double)i - 3 * sigma;
         ldb = (double)j - 3 * sigma;
         lda = (lda * lda + ldb * ldb) / (2 * sigma * sigma);
         k=(int)(1000 * TMath::Exp(-lda));
         lda = k;
         if(lda != 0){
            if((i + 1) > lhx)
               lhx = i + 1;

            if((j + 1) > lhy)
               lhy = j + 1;
         }
         working_space[i][j] = lda;
         area = area + lda;
         if(lda > maximum){
            maximum = lda;
            positx = i,posity = j;
         }
      }
   }
   //read source matrix
   for(i = 0;i < ssizex_ext; i++){
      for(j = 0;j < ssizey_ext; j++){
         working_space[i][j + 14 * ssizey_ext] = TMath::Abs(working_space[i][j + ssizey_ext]);
      }
   }
   //calculate matrix b=ht*h
   i = lhx - 1;
   if(i > ssizex_ext)
      i = ssizex_ext;

   j = lhy - 1;
   if(j>ssizey_ext)
      j = ssizey_ext;

   i1min = -i,i1max = i;
   i2min = -j,i2max = j;
   for(i2 = i2min; i2 <= i2max; i2++){
      for(i1 = i1min; i1 <= i1max; i1++){
         ldc = 0;
         j2min = -i2;
         if(j2min < 0)
            j2min = 0;

         j2max = lhy - 1 - i2;
         if(j2max > lhy - 1)
            j2max = lhy - 1;

         for(j2 = j2min; j2 <= j2max; j2++){
            j1min = -i1;
            if(j1min < 0)
               j1min = 0;

            j1max = lhx - 1 - i1;
            if(j1max > lhx - 1)
               j1max = lhx - 1;

            for(j1 = j1min; j1 <= j1max; j1++){
               lda = working_space[j1][j2];
               ldb = working_space[i1 + j1][i2 + j2];
               ldc = ldc + lda * ldb;
            }
         }
         k = (i1 + ssizex_ext) / ssizex_ext;
         working_space[(i1 + ssizex_ext) % ssizex_ext][i2 + ssizey_ext + 10 * ssizey_ext + k * 2 * ssizey_ext] = ldc;
      }
   }
   //calculate at*y and write into p
   i = lhx - 1;
   if(i > ssizex_ext)
      i = ssizex_ext;

   j = lhy - 1;
   if(j > ssizey_ext)
      j = ssizey_ext;

   i2min = -j,i2max = ssizey_ext + j - 1;
   i1min = -i,i1max = ssizex_ext + i - 1;
   for(i2 = i2min; i2 <= i2max; i2++){
      for(i1=i1min;i1<=i1max;i1++){
         ldc=0;
         for(j2 = 0; j2 <= (lhy - 1); j2++){
            for(j1 = 0; j1 <= (lhx - 1); j1++){
               k2 = i2 + j2,k1 = i1 + j1;
               if(k2 >= 0 && k2 < ssizey_ext && k1 >= 0 && k1 < ssizex_ext){
                  lda = working_space[j1][j2];
                  ldb = working_space[k1][k2 + 14 * ssizey_ext];
                  ldc = ldc + lda * ldb;
               }
            }
         }
         k = (i1 + ssizex_ext) / ssizex_ext;
         working_space[(i1 + ssizex_ext) % ssizex_ext][i2 + ssizey_ext + ssizey_ext + k * 3 * ssizey_ext] = ldc;
      }
   }
   //move matrix p
   for(i2 = 0; i2 < ssizey_ext; i2++){
      for(i1 = 0; i1 < ssizex_ext; i1++){
         k = (i1 + ssizex_ext) / ssizex_ext;
         ldb = working_space[(i1 + ssizex_ext) % ssizex_ext][i2 + ssizey_ext + ssizey_ext + k * 3 * ssizey_ext];
         working_space[i1][i2 + 14 * ssizey_ext] = ldb;
      }
   }
   //initialization in x1 matrix
   for(i2 = 0; i2 < ssizey_ext; i2++){
      for(i1 = 0; i1 < ssizex_ext; i1++){
         working_space[i1][i2 + ssizey_ext] = 1;
         working_space[i1][i2 + 2 * ssizey_ext] = 0;
      }
   }
   //START OF ITERATIONS
   for(lindex = 0; lindex < deconIterations; lindex++){
      for(i2 = 0; i2 < ssizey_ext; i2++){
         for(i1 = 0; i1 < ssizex_ext; i1++){
            lda = working_space[i1][i2 + ssizey_ext];
            ldc = working_space[i1][i2 + 14 * ssizey_ext];
            if(lda > 0.000001 && ldc > 0.000001){
               ldb=0;
               j2min=i2;
               if(j2min > lhy - 1)
                  j2min = lhy - 1;

               j2min = -j2min;
               j2max = ssizey_ext - i2 - 1;
               if(j2max > lhy - 1)
                  j2max = lhy - 1;

               j1min = i1;
               if(j1min > lhx - 1)
                  j1min = lhx - 1;

               j1min = -j1min;
               j1max = ssizex_ext - i1 - 1;
               if(j1max > lhx - 1)
                  j1max = lhx - 1;

               for(j2 = j2min; j2 <= j2max; j2++){
                  for(j1 = j1min; j1 <= j1max; j1++){
                     k = (j1 + ssizex_ext) / ssizex_ext;
                     ldc = working_space[(j1 + ssizex_ext) % ssizex_ext][j2 + ssizey_ext + 10 * ssizey_ext + k * 2 * ssizey_ext];
                     lda = working_space[i1 + j1][i2 + j2 + ssizey_ext];
                     ldb = ldb + lda * ldc;
                  }
               }
               lda = working_space[i1][i2 + ssizey_ext];
               ldc = working_space[i1][i2 + 14 * ssizey_ext];
               if(ldc * lda != 0 && ldb != 0){
                  lda =lda * ldc / ldb;
               }

               else
                  lda=0;
               working_space[i1][i2 + 2 * ssizey_ext] = lda;
            }
         }
      }
      for(i2 = 0; i2 < ssizey_ext; i2++){
         for(i1 = 0; i1 < ssizex_ext; i1++)
            working_space[i1][i2 + ssizey_ext] = working_space[i1][i2 + 2 * ssizey_ext];
      }
   }
   //looking for maximum
   maximum=0;
   for(i = 0; i < ssizex_ext; i++){
      for(j = 0; j < ssizey_ext; j++){
         working_space[(i + positx) % ssizex_ext][(j + posity) % ssizey_ext] = area * working_space[i][j + ssizey_ext];
         if(maximum < working_space[(i + positx) % ssizex_ext][(j + posity) % ssizey_ext])
            maximum = working_space[(i + positx) % ssizex_ext][(j + posity) % ssizey_ext];

      }
   }
   //searching for peaks in deconvolved spectrum
   for(i = 1; i < ssizex_ext - 1; i++){
      for(j = 1; j < ssizey_ext - 1; j++){
         if(working_space[i][j] > working_space[i - 1][j] && working_space[i][j] > working_space[i - 1][j - 1] && working_space[i][j] > working_space[i][j - 1] && working_space[i][j] > working_space[i + 1][j - 1] && working_space[i][j] > working_space[i + 1][j] && working_space[i][j] > working_space[i + 1][j + 1] && working_space[i][j] > working_space[i][j + 1] && working_space[i][j] > working_space[i - 1][j + 1]){
            if(i >= shift && i < ssizex + shift && j >= shift && j < ssizey + shift){
               if(working_space[i][j] > threshold * maximum / 100.0){
                  if(peak_index < fMaxPeaks){
                     for(k = i - 1,a = 0,b = 0; k <= i + 1; k++){
                        a += (double)(k - shift) * working_space[k][j];
                        b += working_space[k][j];
                     }
                     ax=a/b;
                     if(ax < 0)
                        ax = 0;

                     if(ax >= ssizex)
                        ax = ssizex - 1;

                     for(k = j - 1,a = 0,b = 0; k <= j + 1; k++){
                        a += (double)(k - shift) * working_space[i][k];
                        b += working_space[i][k];
                     }
                     ay=a/b;
                     if(ay < 0)
                        ay = 0;

                     if(ay >= ssizey)
                        ay = ssizey - 1;

                     if(peak_index == 0){
                        fPositionX[0] = ax;
                        fPositionY[0] = ay;                        
                        peak_index = 1; 
                     }

                     else{
                        for(k = 0, priz = 0; k < peak_index && priz == 0; k++){
                           if(working_space[shift+(int)(ax+0.5)][15 * ssizey_ext + shift + (int)(ay+0.5)] > working_space[shift+(int)(fPositionX[k]+0.5)][15 * ssizey_ext + shift + (int)(fPositionY[k]+0.5)])
                              priz = 1;
                        }
                        if(priz == 0){
                           if(k < fMaxPeaks){
                              fPositionX[k] = ax;
                              fPositionY[k] = ay;
                           }
                        }

                        else{
                           for(l = peak_index; l >= k; l--){
                              if(l < fMaxPeaks){
                                 fPositionX[l] = fPositionX[l - 1];
                                 fPositionY[l] = fPositionY[l - 1];
                              }
                           }
                           fPositionX[k - 1] = ax;
                           fPositionY[k - 1] = ay;
                        }
                        if(peak_index < fMaxPeaks)
                           peak_index += 1;
                     }
                  }
               }
            }
         }
      }
   }
   //writing back deconvolved spectrum 
   for(i = 0; i < ssizex; i++){
      for(j = 0; j < ssizey; j++){
         dest[i][j] = working_space[i + shift][j + shift];
      }
   }
   i = (int)(4 * sigma + 0.5);
   i = 4 * i;   
   for (j = 0; j < ssizex + i; j++)
      delete[]working_space[j];
   delete[]working_space;
   fNPeaks = peak_index;
   return fNPeaks;
}


// STATIC functions (called by TH1)

//_______________________________________________________________________________
Int_t TSpectrum2::StaticSearch(const TH1 *hist, Double_t sigma, Option_t *option, Double_t threshold)
{
   //static function, interface to TSpectrum2::Search
   
   TSpectrum2 s;
   return s.Search(hist,sigma,option,threshold);
}

//_______________________________________________________________________________
TH1 *TSpectrum2::StaticBackground(const TH1 *hist,Int_t niter, Option_t *option)
{
   //static function, interface to TSpectrum2::Background
   
   TSpectrum2 s;
   return s.Background(hist,niter,option);
}
