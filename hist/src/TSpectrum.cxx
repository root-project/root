// @(#)root/hist:$Name:  $:$Id: TSpectrum.cxx,v 1.47 2006/06/26 10:51:00 brun Exp $
// Author: Miroslav Morhac   27/05/99

//__________________________________________________________________________
//   THIS CLASS CONTAINS ADVANCED SPECTRA PROCESSING FUNCTIONS.            //
//                                                                         //
//   ONE-DIMENSIONAL BACKGROUND ESTIMATION FUNCTIONS                       //
//   ONE-DIMENSIONAL SMOOTHING FUNCTIONS                                   //
//   ONE-DIMENSIONAL DECONVOLUTION FUNCTIONS                               //
//   ONE-DIMENSIONAL PEAK SEARCH FUNCTIONS                                 //
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
//   These NIM papers are also available as doc or ps files from:          //
//BEGIN_HTML <!--
/* -->
   <A href="ftp://root.cern.ch/root/Spectrum.doc">Spectrum.doc</A><br>
   <A href="ftp://root.cern.ch/root/SpectrumDec.ps.gz">SpectrumDec.ps.gz</A><br>
   <A href="ftp://root.cern.ch/root/SpectrumSrc.ps.gz">SpectrumSrc.ps.gz</A><br>
   <A href="ftp://root.cern.ch/root/SpectrumBck.ps.gz">SpectrumBck.ps.gz</A><br>
<!--*/
// -->END_HTML
//
//____________________________________________________________________________
    
#include "TSpectrum.h"
#include "TPolyMarker.h"
#include "TVirtualPad.h"
#include "TMath.h"

Int_t TSpectrum::fgIterations    = 3;
Int_t TSpectrum::fgAverageWindow = 3;

#define PEAK_WINDOW 1024
ClassImp(TSpectrum)  

//______________________________________________________________________________
TSpectrum::TSpectrum() :TNamed("Spectrum", "Miroslav Morhac peak finder") 
{
   // Constructor.

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
TSpectrum::TSpectrum(Int_t maxpositions, Float_t resolution) :TNamed("Spectrum", "Miroslav Morhac peak finder") 
{   
//  maxpositions:  maximum number of peaks
//  resolution:    determines resolution of the neighboring peaks
//                 default value is 1 correspond to 3 sigma distance
//                 between peaks. Higher values allow higher resolution
//                 (smaller distance between peaks.
//                 May be set later through SetResolution.
   
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
   // Destructor.

   delete [] fPosition;
   delete [] fPositionX;
   delete [] fPositionY;
   delete    fHistogram;
}


//______________________________________________________________________________
void TSpectrum::SetAverageWindow(Int_t w)
{
  // static function: Set average window of searched peaks
  // see TSpectrum::SearchHighRes
   
   fgAverageWindow = w;
}

//______________________________________________________________________________
void TSpectrum::SetDeconIterations(Int_t n)
{
  // static function: Set max number of decon iterations in deconvolution operation
  // see TSpectrum::SearchHighRes
   
   fgIterations = n;
}

//______________________________________________________________________________
TH1 *TSpectrum::Background(const TH1 * h, int numberIterations, Option_t * option) 
{   
//   ONE-DIMENSIONAL BACKGROUND ESTIMATION FUNCTION 
//   This function calculates the background spectrum in the input histogram h.
//   The background is returned as a histogram. 
//                
//   Function parameters:
//   -h: input 1-d histogram
//   -numberIterations, (default value = 2)
//      Increasing numberIterations make the result smoother and lower.
//   -option: may contain one of the following options
//      - to set the direction parameter
//        "BackDecreasingWindow". By default the direction is BackIncreasingWindow
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
//      - "nocompton"- if selected the estimation of Compton edge
//                  will be not be included   (by default the compton estimation is set)
//      - "same" : if this option is specified, the resulting background
//                 histogram is superimposed on the picture in the current pad.
//
//  NOTE that the background is only evaluated in the current range of h.
//  ie, if h has a bin range (set via h->GetXaxis()->SetRange(binmin,binmax),
//  the returned histogram will be created with the same number of bins
//  as the input histogram h, but only bins from binmin to binmax will be filled
//  with the estimated background.

   if (h == 0) return 0;
   Int_t dimension = h->GetDimension();
   if (dimension > 1) {
      Error("Search", "Only implemented for 1-d histograms");
      return 0;
   }
   TString opt = option;
   opt.ToLower();
   
   //set options
   Int_t direction = kBackIncreasingWindow;
   if (opt.Contains("backdecreasingwindow")) direction = kBackDecreasingWindow;
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
   Bool_t compton = kTRUE;
   if (opt.Contains("nocompton")) compton = kFALSE;

   Int_t first = h->GetXaxis()->GetFirst();
   Int_t last  = h->GetXaxis()->GetLast();
   Int_t size = last-first+1;
   Int_t i;
   Float_t * source = new float[size];
   for (i = 0; i < size; i++) source[i] = h->GetBinContent(i + first);
   
   //find background (source is input and in output contains the background
   Background(source,size,numberIterations, direction, filterOrder,smoothing,smoothWindow,compton);
   
   //create output histogram containing backgound
   //only bins in the range of the input histogram are filled
   Int_t nch = strlen(h->GetName());
   char *hbname = new char[nch+20];
   sprintf(hbname,"%s_background",h->GetName());
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
   // Print the array of positions

   printf("\nNumber of positions = %d\n",fNPeaks);
   for (Int_t i=0;i<fNPeaks;i++) {
      printf(" x[%d] = %g, y[%d] = %g\n",i,fPositionX[i],i,fPositionY[i]);
   }
}


//______________________________________________________________________________
Int_t TSpectrum::Search(const TH1 * hin, Double_t sigma, Option_t * option, Double_t threshold) 
{   
/////////////////////////////////////////////////////////////////////////////
//   ONE-DIMENSIONAL PEAK SEARCH FUNCTION                                  //
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
//   if option is not equal to "goff" (goff is the default), then          //
//   a polymarker object is created and added to the list of functions of  //
//   the histogram. The histogram is drawn with the specified option and   //
//   the polymarker object drawn on top of the histogram.                  //
//   The polymarker coordinates correspond to the npeaks peaks found in    //
//   the histogram.                                                        //
//   A pointer to the polymarker object can be retrieved later via:        //
//    TList *functions = hin->GetListOfFunctions();                        //
//    TPolyMarker *pm = (TPolyMarker*)functions->FindObject("TPolyMarker") //
//                                                                         //
/////////////////////////////////////////////////////////////////////////////
       
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
   if (dimension == 1) {
      Int_t first = hin->GetXaxis()->GetFirst();
      Int_t last  = hin->GetXaxis()->GetLast();
      Int_t size = last-first+1;
      Int_t i, bin, npeaks;
      Float_t * source = new float[size];
      Float_t * dest   = new float[size];
      for (i = 0; i < size; i++) source[i] = hin->GetBinContent(i + first);
      if (sigma <= 1) {
         sigma = size/fMaxPeaks;
         if (sigma < 1) sigma = 1;
         if (sigma > 8) sigma = 8;
      }
      npeaks = SearchHighRes(source, dest, size, sigma, 100*threshold, kTRUE, fgIterations, kTRUE, fgAverageWindow);

      //TH1 * hnew = (TH1 *) hin->Clone("markov");
      //for (i = 0; i < size; i++)
      //   hnew->SetBinContent(i + 1, source[i]);
      for (i = 0; i < npeaks; i++) {
         bin = first + Int_t(fPositionX[i] + 0.5);
         fPositionX[i] = hin->GetBinCenter(bin);
         fPositionY[i] = hin->GetBinContent(bin);
      }
      delete [] source;
      delete [] dest;
      
      if (strstr(option, "goff"))
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
   return 0;
}


//______________________________________________________________________________
void TSpectrum::SetResolution(Float_t resolution) 
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
const char *TSpectrum::Background(float *spectrum, int ssize,
                                          int numberIterations,
                                          int direction, int filterOrder,
                                          bool smoothing,int smoothWindow,
                                          bool compton)
{   
/////////////////////////////////////////////////////////////////////////////
//        ONE-DIMENSIONAL BACKGROUND ESTIMATION FUNCTION - GENERAL FUNCTION 
//                                                                          
//        This function calculates background spectrum from source spectrum.
//        The result is placed in the vector pointed by spe1945ctrum pointer.   
//                                                                         
//        Function parameters:                                             
//        spectrum-pointer to the vector of source spectrum                
//        ssize-length of the spectrum vector                              
//        numberIterations-maximal width of clipping window,               
//        direction- direction of change of clipping window               
//               - possible values=kBackIncreasingWindow                  
//                                 kBackDecreasingWindow                  
//        filterOrder-order of clipping filter,                           
//                  -possible values=kBackOrder2                          
//                                   kBackOrder4                          
//                                   kBackOrder6                           
//                                   kBackOrder8                           
//        smoothing- logical variable whether the smoothing operation      
//               in the estimation of background will be included           
//             - possible values=kFALSE                      
//                               kTRUE                      
//        smoothWindow-width of smoothing window,          
//                  -possible values=kBackSmoothing3                        
//                                   kBackSmoothing5                       
//                                   kBackSmoothing7                       
//                                   kBackSmoothing9                        
//                                   kBackSmoothing11                       
//                                   kBackSmoothing13                       
//                                   kBackSmoothing15                        
//         compton- logical variable whether the estimation of Compton edge
//                  will be included                                         
//             - possible values=kFALSE                        
//                               kTRUE                        
//                                                                        
///////////////////////////////////////////////////////////////////////////////
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

<p class=MsoNormal style='margin-left:36.0pt;text-indent:-18.0pt'><span
style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>method is based on Sensitive
Nonlinear Iterative Peak (SNIP) clipping algorithm</span></p>

<p class=MsoNormal style='margin-left:36.0pt;text-indent:-18.0pt'><span
style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'> new value in the channel “i” is
calculated</span></p>

<p class=MsoNormal>

<table cellpadding=0 cellspacing=0 align=left>
 <tr>
  <td width=53 height=23></td>
 </tr>
 <tr>
  <td></td>
  <td><img width=486 height=72 src="gif/TSpectrum_Background.gif"></td>
 </tr>
</table>

<span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>&nbsp;</p>

<br clear=ALL>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>where
p = 1, 2, …, numberIterations. In fact it represents second order difference
filter (-1,2,-1).</span></p>

<p class=MsoNormal><i><span style='font-size:18.0pt'>&nbsp;</span></i></p>

<p class=MsoNormal><i><span style='font-size:18.0pt'>Function:</span></i></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:18.0pt'>const
<a href="http://root.cern.ch/root/html/ListOtransTypes.html#char" target="_parent">char</a>*
<a
href="http://root.cern.ch/root/html/src/TSpectrum.cxx.html#TSpectrum:Background1General"
target="_parent">Background</a>(<a
href="http://root.cern.ch/root/html/ListOtransTypes.html#float" target="_parent">float</a>
*spectrum, <a href="http://root.cern.ch/root/html/ListOtransTypes.html#int"
target="_parent">int</a> ssize, <a
href="http://root.cern.ch/root/html/ListOtransTypes.html#int" target="_parent">int</a>
numberIterations, <a href="http://root.cern.ch/root/html/ListOtransTypes.html#int"
target="_parent">int</a> direction, <a
href="http://root.cern.ch/root/html/ListOtransTypes.html#int" target="_parent">int</a>
filterOrder,  <a href="http://root.cern.ch/root/html/ListOtransTypes.html#bool"
target="_parent">bool</a> smoothing,  <a
href="http://root.cern.ch/root/html/ListOtransTypes.html#int" target="_parent">int</a>
smoothingWindow, <a href="http://root.cern.ch/root/html/ListOtransTypes.html#bool"
target="_parent">bool</a> compton)  </span></b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>This
function calculates background spectrum from the source spectrum.  The result
is placed in the vector pointed by spectrum pointer.  One can also change the
direction of the change of the clipping window, the order of the clipping
filter, to include smoothing, to set width of smoothing window and to include
the estimation of Compton edges.</span><span style='font-size:18.0pt'> </span><span
style='font-size:16.0pt'>On successful completion it returns 0. On error it
returns pointer to the string describing error.</span></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><i><span style='font-size:16.0pt;color:red'>Parameters:</span></i></p>

<p class=MsoNormal>        <b><span style='font-size:14.0pt'>spectrum</span></b>-pointer
to the vector of source spectrum                  </p>

<p class=MsoNormal>        <b><span style='font-size:14.0pt'>ssize</span></b>-length
of the spectrum vector                                 </p>

<p class=MsoNormal>        <b><span style='font-size:14.0pt'>numberIterations</span></b>-maximal
width of clipping window,                                 </p>

<p class=MsoNormal>        <b><span style='font-size:14.0pt'>direction</span></b>-
direction of change of clipping window                  </p>

<p class=MsoNormal>               - possible
values=kBackIncreasingWindow                      </p>

<p class=MsoNormal>                                            kBackDecreasingWindow                     
</p>

<p class=MsoNormal>        <b><span style='font-size:14.0pt'>filterOrder</span></b>-order
of clipping filter,                              </p>

<p class=MsoNormal>                  -possible
values=kBackOrder2                              </p>

<p class=MsoNormal>                                              kBackOrder4                             
</p>

<p class=MsoNormal>                                              kBackOrder6                             
</p>

<p class=MsoNormal>                                              kBackOrder8                            
</p>

<p class=MsoNormal>        <b><span style='font-size:14.0pt'>smoothing</span></b>-
logical variable whether the smoothing operation in the estimation of </p>

<p class=MsoNormal>               background will be included              </p>

<p class=MsoNormal>             - possible
values=kFALSE                        </p>

<p class=MsoNormal>                                          kTRUE          
             </p>

<p class=MsoNormal>        <b><span style='font-size:14.0pt'>smoothWindow</span></b>-width
of smoothing window,                            </p>

<p class=MsoNormal>                  -possible
values=kBackSmoothing3                          </p>

<p class=MsoNormal>                                             kBackSmoothing5                         
</p>

<p class=MsoNormal>                                             kBackSmoothing7                         
</p>

<p class=MsoNormal>                                             kBackSmoothing9                         
</p>

<p class=MsoNormal>                                             kBackSmoothing11                     
   </p>

<p class=MsoNormal>                                             kBackSmoothing13                        
</p>

<p class=MsoNormal>                                             kBackSmoothing15                        
</p>

<p class=MsoNormal>        <b><span style='font-size:14.0pt'>compton</span></b>-
logical variable whether the estimation of Compton edge   will be
included                                           </p>

<p class=MsoNormal>             - possible
values=kFALSE                          </p>

<p class=MsoNormal>                                          kTRUE                          
</p>

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

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>[3]
D. D. Burgess, R. J. Tervo: Background estimation for gamma-ray spectroscopy.
NIM 214 (1983), 431-434. </span></p>

</div>

<!-- */
// --> End_Html

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
      }while(direction == kBackIncreasingWindow && i <= numberIterations || direction == kBackDecreasingWindow && i >= 1);
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
      }while(direction == kBackIncreasingWindow && i <= numberIterations || direction == kBackDecreasingWindow && i >= 1);
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
      }while(direction == kBackIncreasingWindow && i <= numberIterations || direction == kBackDecreasingWindow && i >= 1);
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
      }while(direction == kBackIncreasingWindow && i <= numberIterations || direction == kBackDecreasingWindow && i >= 1);               
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

//Begin_Html <!--
/* -->
<div class=Section2>

<p class=MsoNormal><i><span style='font-size:18.0pt'>Example 1– script
Background_incr.c :</span></i></p>

<p class=MsoNormal><img width=601 height=407
src="gif/TSpectrum_Background_incr.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Figure
1 Example of the estimation of background for number of iterations=6. Original
spectrum is shown in black color, estimated background in red color.</span></b></p>

<p class=MsoNormal><b><span style='font-size:14.0pt;color:green'>&nbsp;</span></b></p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:green'>Script:</span></b></p>

<p class=MsoNormal>// Example to illustrate the background estimator (class
TSpectrum).</p>

<p class=MsoNormal>// To execute this example, do</p>

<p class=MsoNormal>// root &gt; .x Background_incr.C</p>

<p class=MsoNormal>#include &lt;TSpectrum&gt; </p>

<p class=MsoNormal>void Background_incr() {</p>

<p class=MsoNormal>   Int_t i;</p>

<p class=MsoNormal>   Double_t nbins = 256;</p>

<p class=MsoNormal>   Double_t xmin  = 0;</p>

<p class=MsoNormal>   Double_t xmax  = (Double_t)nbins;</p>

<p class=MsoNormal>   Float_t * source = new float[nbins];</p>

<p class=MsoNormal>   TH1F *back = new
TH1F(&quot;back&quot;,&quot;&quot;,nbins,xmin,xmax);</p>

<p class=MsoNormal>   TH1F *d = new
TH1F(&quot;d&quot;,&quot;&quot;,nbins,xmin,xmax);      </p>

<p class=MsoNormal>   TFile *f = new
TFile(&quot;spectra\\TSpectrum.root&quot;);</p>

<p class=MsoNormal>   back=(TH1F*) f-&gt;Get(&quot;back1;1&quot;);</p>

<p class=MsoNormal>   TCanvas *Background = gROOT-&gt;GetListOfCanvases()-&gt;FindObject(&quot;Background&quot;);</p>

<p class=MsoNormal>   if (!Background) Background = new
TCanvas(&quot;Background&quot;,&quot;Estimation of background with increasing
window&quot;,10,10,1000,700);</p>

<p class=MsoNormal>   back-&gt;Draw(&quot;L&quot;);</p>

<p class=MsoNormal>   TSpectrum *s = new TSpectrum();</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++) source[i]=back-&gt;GetBinContent(i
+ 1); </p>

<p class=MsoNormal>   s-&gt;Background(source,nbins,6,kBackIncreasingWindow,kBackOrder2,kFALSE,kBackSmoothing3,kFALSE);</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++) d-&gt;SetBinContent(i +
1,source[i]);   </p>

<p class=MsoNormal>   d-&gt;SetLineColor(kRed);</p>

<p class=MsoNormal>   d-&gt;Draw(&quot;SAME L&quot;);   </p>

<p class=MsoNormal>   }</p>

</div>

<!-- */
// --> End_Html



//Begin_Html <!--
/* -->
<div class=Section3>

<p class=MsoNormal><i><span style='font-size:18.0pt'>Example 2 – script
Background_decr.c :</span></i></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>In
Figure 1. one can notice that at the edges of the peaks the estimated
background goes under the peaks. An alternative approach is to decrease the
clipping window from a given value numberIterations to the value of one, which
is presented in this example. </span></p>

<p class=MsoNormal><img width=601 height=407
src="gif/TSpectrum_Background_decr.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Figure
2 Example of the estimation of background for numberIterations=6 using
decreasing clipping window algorithm. Original spectrum is shown in black
color, estimated background in red color.</span></b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:#339966'>Script:</span></b></p>

<p class=MsoNormal>// Example to illustrate the background estimator (class
TSpectrum).</p>

<p class=MsoNormal>// To execute this example, do</p>

<p class=MsoNormal>// root &gt; .x Background_decr.C</p>

<p class=MsoNormal>#include &lt;TSpectrum&gt;   </p>

<p class=MsoNormal>void Background_decr() {</p>

<p class=MsoNormal>   Int_t i;</p>

<p class=MsoNormal>   Double_t nbins = 256;</p>

<p class=MsoNormal>   Double_t xmin  = 0;</p>

<p class=MsoNormal>   Double_t xmax  = (Double_t)nbins;</p>

<p class=MsoNormal>   Float_t * source = new float[nbins];</p>

<p class=MsoNormal>   TH1F *back = new
TH1F(&quot;back&quot;,&quot;&quot;,nbins,xmin,xmax);</p>

<p class=MsoNormal>   TH1F *d = new
TH1F(&quot;d&quot;,&quot;&quot;,nbins,xmin,xmax);      </p>

<p class=MsoNormal>   TFile *f = new TFile(&quot;spectra\\TSpectrum.root&quot;);</p>

<p class=MsoNormal>   back=(TH1F*) f-&gt;Get(&quot;back1;1&quot;);</p>

<p class=MsoNormal>   TCanvas *Background =
gROOT-&gt;GetListOfCanvases()-&gt;FindObject(&quot;Background&quot;);</p>

<p class=MsoNormal>   if (!Background) Background = new
TCanvas(&quot;Background&quot;,&quot;Estimation of background with decreasing
window&quot;,10,10,1000,700);</p>

<p class=MsoNormal>   back-&gt;Draw(&quot;L&quot;);</p>

<p class=MsoNormal>   TSpectrum *s = new TSpectrum();</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++)
source[i]=back-&gt;GetBinContent(i + 1); </p>

<p class=MsoNormal>   s-&gt;Background(source,nbins,6,kBackDecreasingWindow,kBackOrder2,kFALSE,kBackSmoothing3,kFALSE);</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++) d-&gt;SetBinContent(i +
1,source[i]);   </p>

<p class=MsoNormal>   d-&gt;SetLineColor(kRed);</p>

<p class=MsoNormal>   d-&gt;Draw(&quot;SAME L&quot;);   </p>

<p class=MsoNormal>   }</p>

</div>

<!-- */
// --> End_Html

//Begin_Html <!--
/* -->
<div class=Section4>

<p class=MsoNormal><i><span style='font-size:18.0pt'>Example 3 – script
Background_width.c :</span></i></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>the question is how to choose the
width of the clipping window, i.e.,  numberIterations   parameter. The
influence of this parameter on the estimated background is illustrated in
Figure 3.</span></p>

<p class=MsoNormal><img width=601 height=407
src="gif/TSpectrum_Background_width.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Figure
3 Example of the influence of clipping window width on the estimated background
for numberIterations=4 (red line), 6 (blue line) 8 (green line) using
decreasing clipping window algorithm.</span></b></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:18.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><i><span style='font-size:18.0pt'>in general one should set this
parameter so that the value 2*numberIterations+1 was greater than the widths
of preserved objects (peaks).</span></i></p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt;
color:#339966'>Script:</span></b></p>

<p class=MsoNormal>// Example to illustrate the influence of the clipping
window width on the estimated background</p>

<p class=MsoNormal>// To execute this example, do</p>

<p class=MsoNormal>// root &gt; .x Background_width.C</p>

<p class=MsoNormal>#include &lt;TSpectrum&gt;</p>

<p class=MsoNormal>void Background_width() {</p>

<p class=MsoNormal>   Int_t i;</p>

<p class=MsoNormal>   Double_t nbins = 256;</p>

<p class=MsoNormal>   Double_t xmin  = 0;</p>

<p class=MsoNormal>   Double_t xmax  = (Double_t)nbins;</p>

<p class=MsoNormal>   Float_t * source = new float[nbins];</p>

<p class=MsoNormal>   TH1F *h = new
TH1F(&quot;h&quot;,&quot;&quot;,nbins,xmin,xmax);</p>

<p class=MsoNormal>   TH1F *d1 = new
TH1F(&quot;d1&quot;,&quot;&quot;,nbins,xmin,xmax);      </p>

<p class=MsoNormal>   TH1F *d2 = new
TH1F(&quot;d2&quot;,&quot;&quot;,nbins,xmin,xmax);         </p>

<p class=MsoNormal>   TH1F *d3 = new
TH1F(&quot;d3&quot;,&quot;&quot;,nbins,xmin,xmax);         </p>

<p class=MsoNormal>   TFile *f = new
TFile(&quot;spectra\\TSpectrum.root&quot;);</p>

<p class=MsoNormal>   h=(TH1F*) f-&gt;Get(&quot;back1;1&quot;);      </p>

<p class=MsoNormal>   TCanvas *background = gROOT-&gt;GetListOfCanvases()-&gt;FindObject(&quot;background&quot;);</p>

<p class=MsoNormal>   if (!background) background = new
TCanvas(&quot;background&quot;,&quot;Influence of clipping window width on the
estimated background&quot;,10,10,1000,700);</p>

<p class=MsoNormal>   h-&gt;Draw(&quot;L&quot;);</p>

<p class=MsoNormal>   TSpectrum *s = new TSpectrum();</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++) source[i]=h-&gt;GetBinContent(i
+ 1);   </p>

<p class=MsoNormal>  
s-&gt;Background(source,nbins,4,kBackDecreasingWindow,kBackOrder2,kFALSE,kBackSmoothing3,kFALSE);</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++) d1-&gt;SetBinContent(i +
1,source[i]);   </p>

<p class=MsoNormal>   d1-&gt;SetLineColor(kRed);</p>

<p class=MsoNormal>   d1-&gt;Draw(&quot;SAME L&quot;);   </p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++)
source[i]=h-&gt;GetBinContent(i + 1);</p>

<p class=MsoNormal>  
s-&gt;Background(source,nbins,6,kBackDecreasingWindow,kBackOrder2,kFALSE,kBackSmoothing3,kFALSE);  
</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++) d2-&gt;SetBinContent(i +
1,source[i]);   </p>

<p class=MsoNormal>   d2-&gt;SetLineColor(kBlue);</p>

<p class=MsoNormal>   d2-&gt;Draw(&quot;SAME L&quot;);   </p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++)
source[i]=h-&gt;GetBinContent(i + 1);</p>

<p class=MsoNormal>  
s-&gt;Background(source,nbins,8,kBackDecreasingWindow,kBackOrder2,kFALSE,kBackSmoothing3,kFALSE);  
</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++) d3-&gt;SetBinContent(i +
1,source[i]);   </p>

<p class=MsoNormal>   d3-&gt;SetLineColor(kGreen);</p>

<p class=MsoNormal>   d3-&gt;Draw(&quot;SAME L&quot;);         </p>

<p class=MsoNormal>}</p>

</div>

<!-- */
// --> End_Html


//Begin_Html <!--
/* -->
<div class=Section5>

<p class=MsoNormal><i><span style='font-size:18.0pt'>Example 4 – script
Background_width2.c :</span></i></p>

<p class=MsoNormal style='margin-left:36.0pt;text-indent:-18.0pt'><span
style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>another example for very complex
spectrum is given in Figure 4.</span></p>

<p class=MsoNormal><img width=601 height=407
src="gif/TSpectrum_Background_width2.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Figure
4 Example of the influence of clipping window width on the estimated background
for numberIterations=10 (red line), 20 (blue line), 30 (green line) and 40
(magenta line) using decreasing clipping window algorithm.</span></b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:#339966'>Script:</span></b></p>

<p class=MsoNormal>// Example to illustrate the influence of the clipping
window width on the estimated background</p>

<p class=MsoNormal>// To execute this example, do</p>

<p class=MsoNormal>// root &gt; .x Background_width2.C</p>

<p class=MsoNormal>#include &lt;TSpectrum&gt;  </p>

<p class=MsoNormal>void Background_width2() {</p>

<p class=MsoNormal>   Int_t i;</p>

<p class=MsoNormal>   Double_t nbins = 4096;</p>

<p class=MsoNormal>   Double_t xmin  = 0;</p>

<p class=MsoNormal>   Double_t xmax  = (Double_t)4096;</p>

<p class=MsoNormal>   Float_t * source = new float[nbins];</p>

<p class=MsoNormal>   TH1F *h = new
TH1F(&quot;h&quot;,&quot;&quot;,nbins,xmin,xmax);</p>

<p class=MsoNormal>   TH1F *d1 = new TH1F(&quot;d1&quot;,&quot;&quot;,nbins,xmin,xmax);     
</p>

<p class=MsoNormal>   TH1F *d2 = new
TH1F(&quot;d2&quot;,&quot;&quot;,nbins,xmin,xmax);         </p>

<p class=MsoNormal>   TH1F *d3 = new
TH1F(&quot;d3&quot;,&quot;&quot;,nbins,xmin,xmax);         </p>

<p class=MsoNormal>   TH1F *d4 = new
TH1F(&quot;d4&quot;,&quot;&quot;,nbins,xmin,xmax);            </p>

<p class=MsoNormal>   TFile *f = new
TFile(&quot;spectra\\TSpectrum.root&quot;);</p>

<p class=MsoNormal>   h=(TH1F*) f-&gt;Get(&quot;back2;1&quot;);   </p>

<p class=MsoNormal>   TCanvas *background =
gROOT-&gt;GetListOfCanvases()-&gt;FindObject(&quot;background&quot;);</p>

<p class=MsoNormal>   if (!background) background = new
TCanvas(&quot;background&quot;,&quot;Influence of clipping window width on the
estimated background&quot;,10,10,1000,700);</p>

<p class=MsoNormal>   h-&gt;SetAxisRange(0,1000);</p>

<p class=MsoNormal>   h-&gt;SetMaximum(20000);</p>

<p class=MsoNormal>   h-&gt;Draw(&quot;L&quot;);</p>

<p class=MsoNormal>   TSpectrum *s = new TSpectrum();</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++)
source[i]=h-&gt;GetBinContent(i + 1);   </p>

<p class=MsoNormal>  
s-&gt;Background(source,nbins,10,kBackDecreasingWindow,kBackOrder2,kFALSE,kBackSmoothing3,kFALSE);  
    </p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++) d1-&gt;SetBinContent(i +
1,source[i]);   </p>

<p class=MsoNormal>   d1-&gt;SetLineColor(kRed);</p>

<p class=MsoNormal>   d1-&gt;Draw(&quot;SAME L&quot;);   </p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++)
source[i]=h-&gt;GetBinContent(i + 1);</p>

<p class=MsoNormal>   s-&gt;Background(source,nbins,20,kBackDecreasingWindow,kBackOrder2,kFALSE,kBackSmoothing3,kFALSE);  
</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++) d2-&gt;SetBinContent(i +
1,source[i]);   </p>

<p class=MsoNormal>   d2-&gt;SetLineColor(kBlue);</p>

<p class=MsoNormal>   d2-&gt;Draw(&quot;SAME L&quot;);   </p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++)
source[i]=h-&gt;GetBinContent(i + 1);</p>

<p class=MsoNormal>  
s-&gt;Background(source,nbins,30,kBackDecreasingWindow,kBackOrder2,kFALSE,kBackSmoothing3,kFALSE);  
</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++) d3-&gt;SetBinContent(i +
1,source[i]);   </p>

<p class=MsoNormal>   d3-&gt;SetLineColor(kGreen);</p>

<p class=MsoNormal>   d3-&gt;Draw(&quot;SAME L&quot;);         </p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++)
source[i]=h-&gt;GetBinContent(i + 1);</p>

<p class=MsoNormal>  
s-&gt;Background(source,nbins,10,kBackDecreasingWindow,kBackOrder2,kFALSE,kBackSmoothing3,kFALSE);  
</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++) d4-&gt;SetBinContent(i +
1,source[i]);   </p>

<p class=MsoNormal>   d4-&gt;SetLineColor(kMagenta);</p>

<p class=MsoNormal>   d4-&gt;Draw(&quot;SAME L&quot;);            </p>

<p class=MsoNormal>}</p>

</div>

<!-- */
// --> End_Html

//Begin_Html <!--
/* -->
<div class=Section6>

<p class=MsoNormal><i><span style='font-size:18.0pt'>Example 5 – script
Background_order.c :</span></i></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>second order difference filter
removes linear (quasi-linear) background and preserves symmetrical peaks.</span></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>however if the shape of the
background is more complex one can employ higher-order clipping filters (see
example in Figure 5)</span></p>

<p class=MsoNormal><img width=601 height=407
src="gif/TSpectrum_Background_order.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Figure
5 Example of the influence of clipping filter difference order on the estimated
background for fNnumberIterations=40, 2-nd order red line, 4-th order blue
line, 6-th order green line and 8-th order magenta line, and using decreasing
clipping window algorithm.</span></b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:#339966'>Script:</span></b></p>

<p class=MsoNormal>// Example to illustrate the influence of the clipping
filter difference order on the estimated background</p>

<p class=MsoNormal>// To execute this example, do</p>

<p class=MsoNormal>// root &gt; .x Background_order.C</p>

<p class=MsoNormal>#include &lt;TSpectrum&gt;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>void Background_order() {</p>

<p class=MsoNormal>   Int_t i;</p>

<p class=MsoNormal>   Double_t nbins = 4096;</p>

<p class=MsoNormal>   Double_t xmin  = 0;</p>

<p class=MsoNormal>   Double_t xmax  = (Double_t)4096;</p>

<p class=MsoNormal>   Float_t * source = new float[nbins];</p>

<p class=MsoNormal>   TH1F *h = new
TH1F(&quot;h&quot;,&quot;&quot;,nbins,xmin,xmax);</p>

<p class=MsoNormal>   TH1F *d1 = new
TH1F(&quot;d1&quot;,&quot;&quot;,nbins,xmin,xmax);      </p>

<p class=MsoNormal>   TH1F *d2 = new
TH1F(&quot;d2&quot;,&quot;&quot;,nbins,xmin,xmax);         </p>

<p class=MsoNormal>   TH1F *d3 = new
TH1F(&quot;d3&quot;,&quot;&quot;,nbins,xmin,xmax);         </p>

<p class=MsoNormal>   TH1F *d4 = new
TH1F(&quot;d4&quot;,&quot;&quot;,nbins,xmin,xmax);            </p>

<p class=MsoNormal>   TFile *f = new
TFile(&quot;spectra\\TSpectrum.root&quot;);</p>

<p class=MsoNormal>   h=(TH1F*) f-&gt;Get(&quot;back2;1&quot;);</p>

<p class=MsoNormal>   TCanvas *background =
gROOT-&gt;GetListOfCanvases()-&gt;FindObject(&quot;background&quot;);</p>

<p class=MsoNormal>   if (!background) background = new
TCanvas(&quot;background&quot;,&quot;Influence of clipping filter difference
order on the estimated background&quot;,10,10,1000,700);</p>

<p class=MsoNormal>   h-&gt;SetAxisRange(1220,1460);</p>

<p class=MsoNormal>   h-&gt;SetMaximum(11000);</p>

<p class=MsoNormal>   h-&gt;Draw(&quot;L&quot;);</p>

<p class=MsoNormal>   TSpectrum *s = new TSpectrum();</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++)
source[i]=h-&gt;GetBinContent(i + 1);   </p>

<p class=MsoNormal>   s-&gt;Background(source,nbins,40,kBackDecreasingWindow,kBackOrder2,kFALSE,kBackSmoothing3,kFALSE);  
    </p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++) d1-&gt;SetBinContent(i +
1,source[i]);   </p>

<p class=MsoNormal>   d1-&gt;SetLineColor(kRed);</p>

<p class=MsoNormal>   d1-&gt;Draw(&quot;SAME L&quot;);   </p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++)
source[i]=h-&gt;GetBinContent(i + 1);</p>

<p class=MsoNormal>   s-&gt;Background(source,nbins,40,kBackDecreasingWindow,kBackOrder4,kFALSE,kBackSmoothing3,kFALSE);  
</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++) d2-&gt;SetBinContent(i +
1,source[i]);   </p>

<p class=MsoNormal>   d2-&gt;SetLineColor(kBlue);</p>

<p class=MsoNormal>   d2-&gt;Draw(&quot;SAME L&quot;);   </p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++)
source[i]=h-&gt;GetBinContent(i + 1);</p>

<p class=MsoNormal>  
s-&gt;Background(source,nbins,40,kBackDecreasingWindow,kBackOrder6,kFALSE,kBackSmoothing3,kFALSE);     
</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++) d3-&gt;SetBinContent(i +
1,source[i]);   </p>

<p class=MsoNormal>   d3-&gt;SetLineColor(kGreen);</p>

<p class=MsoNormal>   d3-&gt;Draw(&quot;SAME L&quot;);         </p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++)
source[i]=h-&gt;GetBinContent(i + 1);</p>

<p class=MsoNormal>   s-&gt;Background(source,nbins,40,kBackDecreasingWindow,kBackOrder8,kFALSE,kBackSmoothing3,kFALSE);     
</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++) d4-&gt;SetBinContent(i +
1,source[i]);   </p>

<p class=MsoNormal>   d4-&gt;SetLineColor(kMagenta);</p>

<p class=MsoNormal>   d4-&gt;Draw(&quot;SAME L&quot;);            </p>

<p class=MsoNormal>}</p>

</div>

</div>
<!-- */
// --> End_Html

//Begin_Html <!--
/* -->
<div class=Section7>

<p class=MsoNormal><i><span style='font-size:16.0pt'>Example 6 – script
Background_smooth.c :</span></i></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>the estimate of the background can
be influenced by noise present in the spectrum. We proposed  the algorithm of
the background estimate with simultaneous smoothing</span></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>in the original algorithm without
smoothing, the estimated background snatches the lower spikes in the noise.
Consequently, the areas of peaks are biased by this error. </span></p>

<p class=MsoNormal style='text-indent:5.7pt'><img width=554 height=104
src="gif/TSpectrum_Background_smooth1.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Figure
7 Principle of background estimation algorithm with  simultaneous smoothing</span></b></p>

<p class=MsoNormal><img width=601 height=407
src="gif/TSpectrum_Background_smooth2.jpg"></p>

<p class=MsoNormal><b><span style='font-size:16.0pt'>Figure 8 Illustration of
non-smoothing (red line) and smoothing algorithm of background estimation (blue
line).</span></b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:#339966'>Script:</span></b></p>

<p class=MsoNormal>// Example to illustrate the background estimator (class
TSpectrum) including Compton edges.</p>

<p class=MsoNormal>// To execute this example, do</p>

<p class=MsoNormal>// root &gt; .x Background_smooth.C</p>

<p class=MsoNormal>#include &lt;TSpectrum&gt;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>void Background_smooth() {</p>

<p class=MsoNormal>   Int_t i;</p>

<p class=MsoNormal>   Double_t nbins = 4096;</p>

<p class=MsoNormal>   Double_t xmin  = 0;</p>

<p class=MsoNormal>   Double_t xmax  = (Double_t)nbins;</p>

<p class=MsoNormal>   Float_t * source = new float[nbins];</p>

<p class=MsoNormal>   TH1F *h = new
TH1F(&quot;h&quot;,&quot;&quot;,nbins,xmin,xmax);</p>

<p class=MsoNormal>   TH1F *d1 = new TH1F(&quot;d1&quot;,&quot;&quot;,nbins,xmin,xmax);</p>

<p class=MsoNormal>   TH1F *d2 = new
TH1F(&quot;d2&quot;,&quot;&quot;,nbins,xmin,xmax);                  </p>

<p class=MsoNormal>   TFile *f = new
TFile(&quot;spectra\\TSpectrum.root&quot;);</p>

<p class=MsoNormal>   h=(TH1F*) f-&gt;Get(&quot;back4;1&quot;);</p>

<p class=MsoNormal>   TCanvas *background =
gROOT-&gt;GetListOfCanvases()-&gt;FindObject(&quot;background&quot;);</p>

<p class=MsoNormal>   if (!background) background = new
TCanvas(&quot;background&quot;,&quot;Estimation of background with
noise&quot;,10,10,1000,700);</p>

<p class=MsoNormal>   h-&gt;SetAxisRange(3460,3830);   </p>

<p class=MsoNormal>   h-&gt;Draw(&quot;L&quot;);</p>

<p class=MsoNormal>   TSpectrum *s = new TSpectrum();</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++)
source[i]=h-&gt;GetBinContent(i + 1);   </p>

<p class=MsoNormal>  
s-&gt;Background(source,nbins,6,kBackDecreasingWindow,kBackOrder2,kFALSE,kBackSmoothing3,kFALSE);     
</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++) d1-&gt;SetBinContent(i +
1,source[i]);   </p>

<p class=MsoNormal>   d1-&gt;SetLineColor(kRed);</p>

<p class=MsoNormal>   d1-&gt;Draw(&quot;SAME L&quot;);</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++)
source[i]=h-&gt;GetBinContent(i + 1);   </p>

<p class=MsoNormal>  
s-&gt;Background(source,nbins,6,kBackDecreasingWindow,kBackOrder2,kTRUE,kBackSmoothing3,kFALSE);     
</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++) d2-&gt;SetBinContent(i +
1,source[i]);   </p>

<p class=MsoNormal>   d2-&gt;SetLineColor(kBlue);</p>

<p class=MsoNormal>   d2-&gt;Draw(&quot;SAME L&quot;);      </p>

<p class=MsoNormal>}</p>

</div>

<!-- */
// --> End_Html

//Begin_Html <!--
/* -->
<div class=Section8>

<p class=MsoNormal><i><span style='font-size:16.0pt'>Example 8 – script
Background_compton.c :</span></i></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>sometimes it is necessary to
include also the Compton edges into the estimate of the background. In Figure 8
we present the example of the synthetic spectrum with Compton edges. </span></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>the background was estimated using
the 8-th order filter with the estimation of the Compton edges using decreasing
clipping window algorithm (numberIterations=10) with smoothing (
smoothingWindow=5).</span></p>

<p class=MsoNormal><img width=601 height=407
src="gif/TSpectrum_Background_compton.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Figure
8 Example of the estimate of the background with Compton edges (red line) for
numberIterations=10, 8-th order difference filter, using decreasing clipping
window algorithm and smoothing (smoothingWindow=5).</span></b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:#339966'>Script:</span></b></p>

<p class=MsoNormal>// Example to illustrate the background estimator (class
TSpectrum) including Compton edges.</p>

<p class=MsoNormal>// To execute this example, do</p>

<p class=MsoNormal>// root &gt; .x Background_compton.C</p>

<p class=MsoNormal>#include &lt;TSpectrum&gt;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>void Background_compton() {</p>

<p class=MsoNormal>   Int_t i;</p>

<p class=MsoNormal>   Double_t nbins = 512;</p>

<p class=MsoNormal>   Double_t xmin  = 0;</p>

<p class=MsoNormal>   Double_t xmax  = (Double_t)nbins;</p>

<p class=MsoNormal>   Float_t * source = new float[nbins];</p>

<p class=MsoNormal>   TH1F *h = new
TH1F(&quot;h&quot;,&quot;&quot;,nbins,xmin,xmax);</p>

<p class=MsoNormal>   TH1F *d1 = new
TH1F(&quot;d1&quot;,&quot;&quot;,nbins,xmin,xmax);      </p>

<p class=MsoNormal>   TFile *f = new
TFile(&quot;spectra\\TSpectrum.root&quot;);</p>

<p class=MsoNormal>   h=(TH1F*) f-&gt;Get(&quot;back3;1&quot;);</p>

<p class=MsoNormal>   TCanvas *background =
gROOT-&gt;GetListOfCanvases()-&gt;FindObject(&quot;background&quot;);</p>

<p class=MsoNormal>   if (!background) background = new
TCanvas(&quot;background&quot;,&quot;Estimation of background with Compton edges under peaks&quot;,10,10,1000,700);</p>

<p class=MsoNormal>   h-&gt;Draw(&quot;L&quot;);</p>

<p class=MsoNormal>   TSpectrum *s = new TSpectrum();</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++)
source[i]=h-&gt;GetBinContent(i + 1);   </p>

<p class=MsoNormal>   s-&gt;Background(source,nbins,10,kBackDecreasingWindow,kBackOrder8,kTRUE,kBackSmoothing5,,kTRUE);     
</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++) d1-&gt;SetBinContent(i +
1,source[i]);   </p>

<p class=MsoNormal>   d1-&gt;SetLineColor(kRed);</p>

<p class=MsoNormal>   d1-&gt;Draw(&quot;SAME L&quot;);   </p>

<p class=MsoNormal>}</p>

</div>

<!-- */
// --> End_Html

//_______________________________________________________________________________
const char* TSpectrum::SmoothMarkov(float *source, int ssize, int averWindow)
{
/////////////////////////////////////////////////////////////////////////////
//        ONE-DIMENSIONAL MARKOV SPECTRUM SMOOTHING FUNCTION 
//                                                           
//        This function calculates smoothed spectrum from source spectrum 
//        based on Markov chain method.                                   
//        The result is placed in the array pointed by source pointer.  
//                                                                      
//        Function parameters:                                          
//        source-pointer to the array of source spectrum                
//        ssize-length of source array                                 
//        averWindow-width of averaging smoothing window               
//                                                                     
/////////////////////////////////////////////////////////////////////////////
//Begin_Html <!--
/* -->
<div class=Section16>

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
src="gif/TSpectrum_Smoothing1.gif"></sub><span style='font-size:16.0pt;
font-family:Arial'>     </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt;
font-family:Arial'>        </span><sub><img width=28 height=36
src="gif/TSpectrum_Smoothing2.gif"></sub><span style='font-size:16.0pt;
font-family:Arial'>  being defined from the normalization condition </span><sub><img
width=70 height=52 src="gif/TSpectrum_Smoothing3.gif"></sub></p>

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
  <td><img width=258 height=60 src="gif/TSpectrum_Smoothing4.gif"></td>
 </tr>
</table>

<span style='font-size:16.0pt;font-family:Arial'>&nbsp;</span></p>

<p class=MsoNormal><span style='font-size:18.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<br clear=ALL>

<p class=MsoNormal style='margin-left:34.2pt;text-align:justify'><span
style='font-size:16.0pt;font-family:Arial'>is the probability of the change of
the peak position from channel i to the channel i+1. </span> <sub><img
width=28 height=36 src="gif/TSpectrum_Smoothing5.gif"></sub><span
style='font-size:16.0pt'>is the normalization constant so that </span><sub><img
width=133 height=34 src="gif/TSpectrum_Smoothing6.gif"></sub> <span
style='font-size:16.0pt'>and m is a width of smoothing window. </span></p>

<p class=MsoNormal><i><span style='font-size:18.0pt'>&nbsp;</span></i></p>

<p class=MsoNormal><i><span style='font-size:18.0pt'>Function:</span></i></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:18.0pt'>const
<a href="http://root.cern.ch/root/html/ListOtransTypes.html#char" target="_parent">char</a>*
SmoothMarkov(<a href="http://root.cern.ch/root/html/ListOtransTypes.html#float"
target="_parent">float</a> *spectrum, <a
href="http://root.cern.ch/root/html/ListOtransTypes.html#int" target="_parent">int</a>
ssize,  <a href="http://root.cern.ch/root/html/ListOtransTypes.html#int"
target="_parent">int</a> averWindow)  </span></b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>This
function calculates smoothed spectrum from the source spectrum based on Markov
chain method. The result is placed in the vector pointed by source pointer. On
successful completion it returns 0. On error it returns pointer to the string
describing error.</span></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><i><span style='font-size:16.0pt;color:red'>Parameters:</span></i></p>

<p class=MsoNormal>        <b><span style='font-size:14.0pt'>spectrum</span></b>-pointer
to the vector of source spectrum                  </p>

<p class=MsoNormal>        <b><span style='font-size:14.0pt'>ssize</span></b>-length
of the spectrum vector                                 </p>

<p class=MsoNormal>        <b><span style='font-size:14.0pt'>averWindow</span></b>-width
of averaging smoothing window </p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><b><i><span style='font-size:18.0pt'>Reference:</span></i></b></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>[1]
Z.K. Silagadze, A new algorithm for automatic photopeak searches. NIM A 376
(1996), 45<b>1.</b>  </span></p>

</div>

<!-- */
// --> End_Html
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
   if(maxch == 0)
      return 0 ;
      
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

//Begin_Html <!--
/* -->
<div class=Section17>

<p class=MsoNormal><i><span style='font-size:16.0pt'>Example 14 – script Smoothing.c
:</span></i></p>

<p class=MsoNormal><span style='font-size:16.0pt'> <img width=296 height=182
src="gif/TSpectrum_Smoothing1.jpg"><img width=296 height=182
src="gif/TSpectrum_Smoothing2.jpg"></span></p>

<p class=MsoNormal><b><span style='font-size:16.0pt'>Fig. 23 Original noisy spectrum</span></b><b><span
style='font-size:14.0pt'>    </span></b><b><span style='font-size:16.0pt'>Fig.
24 Smoothed spectrum m=3</span></b></p>

<p class=MsoNormal><b><span style='font-size:14.0pt'><img width=299 height=184
src="gif/TSpectrum_Smoothing3.jpg"><img width=299 height=184
src="gif/TSpectrum_Smoothing4.jpg"></span></b></p>

<p class=MsoNormal><b><span style='font-size:16.0pt'>Fig. 25 Smoothed spectrum
m=7 Fig.26 Smoothed spectrum m=10</span></b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:#339966'>Script:</span></b></p>

<p class=MsoNormal>// Example to illustrate smoothing using Markov algorithm
(class TSpectrum).</p>

<p class=MsoNormal>// To execute this example, do</p>

<p class=MsoNormal>// root &gt; .x Smoothing.C</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>//#include &lt;TSpectrum&gt;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>void Smoothing() {</p>

<p class=MsoNormal>   Int_t i;</p>

<p class=MsoNormal>   Double_t nbins = 1024;</p>

<p class=MsoNormal>   Double_t xmin  = 0;</p>

<p class=MsoNormal>   Double_t xmax  = (Double_t)nbins;</p>

<p class=MsoNormal>   Float_t * source = new float[nbins];</p>

<p class=MsoNormal>   TH1F *h = new TH1F(&quot;h&quot;,&quot;Smoothed spectrum
for m=3&quot;,nbins,xmin,xmax);</p>

<p class=MsoNormal>   TFile *f = new
TFile(&quot;spectra\\TSpectrum.root&quot;);</p>

<p class=MsoNormal>   h=(TH1F*) f-&gt;Get(&quot;smooth1;1&quot;);   </p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++)
source[i]=h-&gt;GetBinContent(i + 1);   </p>

<p class=MsoNormal>   TCanvas *Smooth1 =
gROOT-&gt;GetListOfCanvases()-&gt;FindObject(&quot;Smooth1&quot;);</p>

<p class=MsoNormal>   if (!Smooth1) Smooth1 = new
TCanvas(&quot;Smooth1&quot;,&quot;Smooth1&quot;,10,10,1000,700);</p>

<p class=MsoNormal>   TSpectrum *s = new TSpectrum();</p>

<p class=MsoNormal>   s-&gt;SmoothMarkov(source,1024,3);  //3, 7, 10</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++) h-&gt;SetBinContent(i +
1,source[i]);   </p>

<p class=MsoNormal>   h-&gt;SetAxisRange(330,880);</p>

<p class=MsoNormal>   h-&gt;Draw(&quot;L&quot;);</p>

<p class=MsoNormal>}</p>

</div>

<!-- */
// --> End_Html


//_______________________________________________________________________________
const char *TSpectrum::Deconvolution(float *source, const float *response,
                                      int ssize, int numberIterations,
                                      int numberRepetitions, double boost ) 
{   
/////////////////////////////////////////////////////////////////////////////
//   ONE-DIMENSIONAL DECONVOLUTION FUNCTION                                //
//   This function calculates deconvolution from source spectrum           //
//   according to response spectrum using Gold algorithm                   //
//   The result is placed in the vector pointed by source pointer.         //
//                                                                         //
//   Function parameters:                                                  //
//   source:  pointer to the vector of source spectrum                     //
//   response:     pointer to the vector of response spectrum              //
//   ssize:    length of source and response spectra                       //
//   numberIterations, for details we refer to the reference given below   //
//   numberRepetitions, for repeated boosted deconvolution                 //
//   boost, boosting coefficient                                           //
//                                                                         //
//    M. Morhac, J. Kliman, V. Matousek, M. Veselský, I. Turzo.:           //
//    Efficient one- and two-dimensional Gold deconvolution and its        //
//    application to gamma-ray spectra decomposition.                      //
//    NIM, A401 (1997) 385-408.                                            //
//                                                                         //
/////////////////////////////////////////////////////////////////////////////
//
//Begin_Html <!--
/* -->
<div class=Section9>
<p class=MsoNormal style='text-align:justify'><b><span style='font-size:20.0pt'>Deconvolution</span></b></p>

<p class=MsoNormal style='text-align:justify'><i><span style='font-size:18.0pt'>&nbsp;</span></i></p>

<p class=MsoNormal style='text-align:justify'><i><span style='font-size:18.0pt'>Goal:
Improvement of the resolution in spectra, decomposition of multiplets</span></i></p>

<p class=MsoNormal><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal><span style='font-size:16.0pt'>Mathematical formulation of
the convolution system is</span></p>

<p class=MsoNormal style='margin-left:18.0pt'>

<table cellpadding=0 cellspacing=0 align=left>
 <tr>
  <td width=0 height=17></td>
 </tr>
 <tr>
  <td></td>
  <td><img width=585 height=84 src="gif/TSpectrum_Deconvolution1.gif"></td>
 </tr>
</table>

<span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>&nbsp;</p>

<br clear=ALL>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>where
h(i) is the impulse response function, x, y are input and output vectors, respectively,
N is the length of x and h vectors. In matrix form we have</span></p>

<p class=MsoNormal style='text-align:justify'>

<table cellpadding=0 cellspacing=0 align=left>
 <tr>
  <td width=4 height=8></td>
 </tr>
 <tr>
  <td></td>
  <td><img width=597 height=360 src="gif/TSpectrum_Deconvolution2.gif"></td>
 </tr>
</table>

<span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal><i><span style='font-size:18.0pt'>&nbsp;</span></i></p>

<p class=MsoNormal><i><span style='font-size:18.0pt'>&nbsp;</span></i></p>

<p class=MsoNormal><i><span style='font-size:18.0pt'>&nbsp;</span></i></p>

<p class=MsoNormal><i><span style='font-size:18.0pt'>&nbsp;</span></i></p>

<br clear=ALL>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>let us assume that we know the
response and the output vector (spectrum) of the above given system. </span></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>the deconvolution represents
solution of the overdetermined system of linear equations, i.e.,  the
calculation of the vector <b>x.</b></span></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>from numerical stability point of
view the operation of deconvolution is extremely critical (ill-posed  problem)
as well as time consuming operation. </span></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>the Gold deconvolution algorithm
proves to work very well, other methods (Fourier, VanCittert etc) oscillate. </span></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>it is suitable to process positive
definite data (e.g. histograms). </span></p>

<p class=MsoNormal><b><i><span style='font-size:16.0pt'>&nbsp;</span></i></b></p>

<p class=MsoNormal><b><i><span style='font-size:16.0pt'>Gold deconvolution
algorithm</span></i></b></p>

<p class=MsoNormal>

<table cellpadding=0 cellspacing=0 align=left>
 <tr>
  <td width=46 height=21></td>
 </tr>
 <tr>
  <td></td>
  <td><img width=551 height=233 src="gif/TSpectrum_Deconvolution3.gif"></td>
 </tr>
</table>

<span style='font-size:18.0pt'>&nbsp;</span></p>

<p class=MsoNormal><span style='font-size:18.0pt'>&nbsp;</span></p>

<p class=MsoNormal><span style='font-size:18.0pt'>&nbsp;</span></p>

<p class=MsoNormal><span style='font-size:18.0pt'>&nbsp;</span></p>

<p class=MsoNormal><span style='font-size:18.0pt'>&nbsp;</span></p>

<p class=MsoNormal><span style='font-size:18.0pt'>&nbsp;</span></p>

<p class=MsoNormal><span style='font-size:18.0pt'>&nbsp;</span></p>

<p class=MsoNormal><i><span style='font-size:18.0pt'>&nbsp;</span></i></p>

<p class=MsoNormal><i><span style='font-size:18.0pt'>&nbsp;</span></i></p>

<p class=MsoNormal><i><span style='font-size:18.0pt'>&nbsp;</span></i></p>

<br clear=ALL>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt;
font-family:Arial'>where L is given number of iterations (numberIterations
parameter).</span></p>

<p class=MsoNormal><b><i><span style='font-size:16.0pt'>&nbsp;</span></i></b></p>

<p class=MsoNormal><span style='position:absolute;z-index:4;margin-left:247px;
margin-top:17px;width:144px;height:36px'><img width=144 height=36
src="gif/TSpectrum_Deconvolution4.gif"></span><b><i><span style='font-size:
16.0pt'>Boosted deconvolution</span></i></b></p>

<p class=MsoNormal style='margin-left:36.0pt;text-indent:-18.0pt'><span
style='font-size:16.0pt'>1.<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>Set the initial solution </span></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:16.0pt'>2.<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>Set required number of repetitions
R and iterations L</span></p>

<p class=MsoNormal style='margin-left:36.0pt;text-indent:-18.0pt'><span
style='font-size:16.0pt'>3.<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>Set r = 1.</span></p>

<p class=MsoNormal style='margin-left:36.0pt;text-indent:-18.0pt'><span
style='font-size:16.0pt'>4.<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>Using Gold deconvolution algorithm
for k=1,2,...,L  find </span><sub><img width=30 height=24
src="gif/TSpectrum_Deconvolution5.gif"></sub></p>

<p class=MsoNormal style='margin-left:36.0pt;text-indent:-18.0pt'><span
style='font-size:16.0pt'>5.<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>If  r = R stop calculation, else</span></p>

<p class=MsoNormal style='text-indent:36.0pt'><span style='font-size:16.0pt'>a.
apply boosting operation, i.e., set </span><sub><img width=201 height=39
src="gif/TSpectrum_Deconvolution6.gif"></sub></p>

<p class=MsoNormal style='margin-left:36.0pt'><span style='font-size:16.0pt'>   
i=0,1,...N-1 and p is boosting coefficient &gt;0.</span></p>

<p class=MsoNormal style='margin-left:36.0pt'><span style='font-size:16.0pt'>b.
r = r + 1</span></p>

<p class=MsoNormal style='margin-left:36.0pt'><span style='font-size:16.0pt'>c.
continue in 4.</span></p>

<p class=MsoNormal><i><span style='font-size:18.0pt'>&nbsp;</span></i></p>

<p class=MsoNormal><i><span style='font-size:18.0pt'>Function:</span></i></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:18.0pt'>const
<a href="http://root.cern.ch/root/html/ListOtransTypes.html#char">char</a>* <a
name="TSpectrum:Deconvolution1">Deconvolution</a>(<a
href="http://root.cern.ch/root/html/ListOtransTypes.html#float">float</a> *source,
const <a href="http://root.cern.ch/root/html/ListOtransTypes.html#float">float</a>
*respMatrix, <a href="http://root.cern.ch/root/html/ListOtransTypes.html#int">int</a> ssize,
<a href="http://root.cern.ch/root/html/ListOtransTypes.html#int">int</a>
numberIterations, <a href="http://root.cern.ch/root/html/ListOtransTypes.html#int">int</a>
numberRepetitions, <a
href="http://root.cern.ch/root/html/ListOtransTypes.html#double">double</a></span></b><b><span
style='font-size:16.0pt'> </span></b><b><span style='font-size:18.0pt'>boost)</span></b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>This
function calculates deconvolution from source spectrum according to response
spectrum using Gold deconvolution algorithm. The result is placed in the vector
pointed by source pointer. On successful completion it returns 0. On error it
returns pointer to the string describing error. If desired after every
numberIterations one can apply boosting operation (exponential function with
exponent given by boost coefficient) and repeat it numberRepetitions times.</span></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><i><span style='font-size:16.0pt;color:red'>Parameters:</span></i></p>

<p class=MsoNormal>        <b><span style='font-size:14.0pt'>source</span></b>-pointer
to the vector of source spectrum                  </p>

<p class=MsoNormal>        <b><span style='font-size:14.0pt'>respMatrix</span></b>-pointer
to the vector of response spectrum                  </p>

<p class=MsoNormal>        <b><span style='font-size:14.0pt'>ssize</span></b>-length
of the spectrum vector                                 </p>

<p class=MsoNormal style='text-align:justify'>        <b><span
style='font-size:14.0pt'>numberIterations</span></b>-number of iterations
(parameter l in the Gold deconvolution  </p>

<p class=MsoNormal style='text-align:justify'>        algorithm)</p>

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

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>[1]
Gold R., ANL-6984, Argonne National Laboratories, Argonne Ill, 1964. </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>[2]
Coote G.E., Iterative smoothing and deconvolution of one- and two-dimensional
elemental distribution data, NIM B 130 (1997) 118.</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>[3]
</span><span lang=SK style='font-size:16.0pt'>M. Morhá&#269;, J. Kliman, V.
Matoušek, M. Veselský, I. Turzo</span><span style='font-size:16.0pt'>.:
Efficient one- and two-dimensional Gold deconvolution and its application to
gamma-ray spectra decomposition. NIM, A401 (1997) 385-408.</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>[4]
Morhá&#269; M., Matoušek V., Kliman J., Efficient algorithm of multidimensional
deconvolution and its application to nuclear data processing, Digital Signal
Processing 13 (2003) 144. </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

</div>

<!-- */
// --> End_Html

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
   if (lh_gold == -1)
      return "ZERO RESPONSE VECTOR";
   
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
//Begin_Html <!--
/* -->
<div class=Section10>

<p class=MsoNormal><i><span style='font-size:16.0pt'>Example 8 – script
Deconvolution.c :</span></i></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>response function (usually peak)
should be shifted left to the first non-zero channel (bin) (see Figure 9)</span></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><img width=600 height=340
src="gif/TSpectrum_Deconvolution1.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Figure
9 Response spectrum</span></b></p>

<p class=MsoNormal><img width=946 height=407
src="gif/TSpectrum_Deconvolution2.jpg"></p>

<p class=MsoNormal><b><span style='font-size:16.0pt'>Figure 10 Principle how the
response matrix is composed inside of the Deconvolution function</span></b></p>

<p class=MsoNormal><img width=601 height=407
src="gif/TSpectrum_Deconvolution3.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Figure
11 Example of Gold deconvolution. The original source spectrum is drawn with
black color, the spectrum after the deconvolution (10000 iterations) with red
color</span></b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:#339966'>Script:</span></b></p>

<p class=MsoNormal>// Example to illustrate deconvolution function (class
TSpectrum).</p>

<p class=MsoNormal>// To execute this example, do</p>

<p class=MsoNormal>// root &gt; .x Deconvolution.C</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>#include &lt;TSpectrum&gt;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>void Deconvolution() {</p>

<p class=MsoNormal>             Int_t i;</p>

<p class=MsoNormal>   Double_t nbins = 256;</p>

<p class=MsoNormal>   Double_t xmin  = 0;</p>

<p class=MsoNormal>   Double_t xmax  = (Double_t)nbins;</p>

<p class=MsoNormal>   Float_t * source = new float[nbins];</p>

<p class=MsoNormal>   Float_t * response = new float[nbins];   </p>

<p class=MsoNormal>   TH1F *h = new
TH1F(&quot;h&quot;,&quot;Deconvolution&quot;,nbins,xmin,xmax);</p>

<p class=MsoNormal>   TH1F *d = new
TH1F(&quot;d&quot;,&quot;&quot;,nbins,xmin,xmax);      </p>

<p class=MsoNormal>   TFile *f = new
TFile(&quot;spectra\\TSpectrum.root&quot;);</p>

<p class=MsoNormal>   h=(TH1F*) f-&gt;Get(&quot;decon1;1&quot;);</p>

<p class=MsoNormal>   TFile *fr = new
TFile(&quot;spectra\\TSpectrum.root&quot;);</p>

<p class=MsoNormal>   d=(TH1F*) fr-&gt;Get(&quot;decon_response;1&quot;);   </p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++)
source[i]=h-&gt;GetBinContent(i + 1);</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++)
response[i]=d-&gt;GetBinContent(i + 1);   </p>

<p class=MsoNormal>   TCanvas *Decon1 =
gROOT-&gt;GetListOfCanvases()-&gt;FindObject(&quot;Decon1&quot;);</p>

<p class=MsoNormal>   if (!Decon1) Decon1 = new
TCanvas(&quot;Decon1&quot;,&quot;Decon1&quot;,10,10,1000,700);</p>

<p class=MsoNormal>   h-&gt;Draw(&quot;L&quot;);</p>

<p class=MsoNormal>   TSpectrum *s = new TSpectrum();</p>

<p class=MsoNormal>   s-&gt;Deconvolution(source,response,256,1000,1,1);</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++) d-&gt;SetBinContent(i +
1,source[i]);   </p>

<p class=MsoNormal>   d-&gt;SetLineColor(kRed);</p>

<p class=MsoNormal>   d-&gt;Draw(&quot;SAME L&quot;);   </p>

<p class=MsoNormal>}</p>

</div>

<!-- */
// --> End_Html

//Begin_Html <!--
/* -->
<div class=Section11>

<p class=MsoNormal><b><span style='font-size:18.0pt'>Examples of Gold
deconvolution method</span></b></p>

<p class=MsoNormal><b><span style='font-size:18.0pt'>&nbsp;</span></b></p>

<p class=MsoNormal style='margin-left:36.0pt;text-indent:-18.0pt'><span
style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>first let us study the influence
of the number of iterations on the deconvolved spectrum  (Figure 12)</span></p>

<p class=MsoNormal><img width=602 height=409
src="gif/TSpectrum_Deconvolution_wide1.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Figure
12 Study of Gold deconvolution algorithm. The original source spectrum is drawn
with black color, spectrum after 100 iterations with red color,</span></b><span
style='font-size:16.0pt'> <b>spectrum after 1000 iterations with blue color,</b>
<b>spectrum after 10000 iterations with green color and </b> <b>spectrum after
100000 iterations with magenta color.</b></span></p>

<p class=MsoNormal style='margin-left:19.95pt;text-align:justify'><span
style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='margin-left:37.05pt;text-align:justify;text-indent:
-17.1pt'><span style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>for relatively narrow peaks in the
above given example the Gold deconvolution method is able to decompose 
overlapping peaks practically to delta - functions.</span></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>in the next example we have chosen
a synthetic data (spectrum, 256 channels) consisting of 5 very closely
positioned, relatively wide peaks (sigma =5), with added noise (Figure 13). </span></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>thin lines represent pure
Gaussians (see Table 1); thick line is a resulting spectrum with additive noise
(10% of the amplitude of small peaks).</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'><img
width=600 height=367 src="gif/TSpectrum_Deconvolution_wide2.jpg"></span></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Figure
13 Testing example of synthetic spectrum composed of 5 Gaussians with added
noise</span></b></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:18.0pt'>&nbsp;</span></p>

<table class=MsoNormalTable border=0 cellspacing=0 cellpadding=0 width=445
 style='width:333.65pt;margin-left:34.95pt'>
 <tr style='height:16.6pt'>
  <td width=80 valign=top style='width:60.3pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:16.6pt'>
  <p class=MsoNormal style='text-align:justify'>Peak #</p>
  </td>
  <td width=132 valign=top style='width:99.2pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:16.6pt'>
  <p class=MsoNormal style='text-align:justify'>Position</p>
  </td>
  <td width=111 valign=top style='width:83.05pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:16.6pt'>
  <p class=MsoNormal style='text-align:justify'>Height </p>
  </td>
  <td width=121 valign=top style='width:91.1pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:16.6pt'>
  <p class=MsoNormal style='text-align:justify'>Area</p>
  </td>
 </tr>
 <tr style='height:16.6pt'>
  <td width=80 valign=top style='width:60.3pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:16.6pt'>
  <p class=MsoNormal style='text-align:justify'>1</p>
  </td>
  <td width=132 valign=top style='width:99.2pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:16.6pt'>
  <p class=MsoNormal style='text-align:justify'>50</p>
  </td>
  <td width=111 valign=top style='width:83.05pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:16.6pt'>
  <p class=MsoNormal style='text-align:justify'>500</p>
  </td>
  <td width=121 valign=top style='width:91.1pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:16.6pt'>
  <p class=MsoNormal style='text-align:justify'>10159</p>
  </td>
 </tr>
 <tr style='height:16.6pt'>
  <td width=80 valign=top style='width:60.3pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:16.6pt'>
  <p class=MsoNormal style='text-align:justify'>2</p>
  </td>
  <td width=132 valign=top style='width:99.2pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:16.6pt'>
  <p class=MsoNormal style='text-align:justify'>70</p>
  </td>
  <td width=111 valign=top style='width:83.05pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:16.6pt'>
  <p class=MsoNormal style='text-align:justify'>3000</p>
  </td>
  <td width=121 valign=top style='width:91.1pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:16.6pt'>
  <p class=MsoNormal style='text-align:justify'>60957</p>
  </td>
 </tr>
 <tr style='height:16.6pt'>
  <td width=80 valign=top style='width:60.3pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:16.6pt'>
  <p class=MsoNormal style='text-align:justify'>3</p>
  </td>
  <td width=132 valign=top style='width:99.2pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:16.6pt'>
  <p class=MsoNormal style='text-align:justify'>80</p>
  </td>
  <td width=111 valign=top style='width:83.05pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:16.6pt'>
  <p class=MsoNormal style='text-align:justify'>1000</p>
  </td>
  <td width=121 valign=top style='width:91.1pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:16.6pt'>
  <p class=MsoNormal style='text-align:justify'>20319</p>
  </td>
 </tr>
 <tr style='height:16.6pt'>
  <td width=80 valign=top style='width:60.3pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:16.6pt'>
  <p class=MsoNormal style='text-align:justify'>4</p>
  </td>
  <td width=132 valign=top style='width:99.2pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:16.6pt'>
  <p class=MsoNormal style='text-align:justify'>100</p>
  </td>
  <td width=111 valign=top style='width:83.05pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:16.6pt'>
  <p class=MsoNormal style='text-align:justify'>5000</p>
  </td>
  <td width=121 valign=top style='width:91.1pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:16.6pt'>
  <p class=MsoNormal style='text-align:justify'>101596</p>
  </td>
 </tr>
 <tr style='height:16.6pt'>
  <td width=80 valign=top style='width:60.3pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:16.6pt'>
  <p class=MsoNormal style='text-align:justify'>5</p>
  </td>
  <td width=132 valign=top style='width:99.2pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:16.6pt'>
  <p class=MsoNormal style='text-align:justify'>110</p>
  </td>
  <td width=111 valign=top style='width:83.05pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:16.6pt'>
  <p class=MsoNormal style='text-align:justify'>500</p>
  </td>
  <td width=121 valign=top style='width:91.1pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:16.6pt'>
  <p class=MsoNormal style='text-align:justify'>10159</p>
  </td>
 </tr>
</table>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Table
1 Positions, heights and areas of peaks in the spectrum shown in Figure 13</span></b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>in ideal case, we should obtain
the result given in Figure 14. The areas of the Gaussian components of the
spectrum are concentrated completely to delta –functions</span></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>when solving the overdetermined
system of linear equations with data from Figure 13 in the sense of minimum
least squares criterion without any regularization we obtain the result with
large oscillations  (Figure 15). </span></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>from mathematical point of view,
it is the optimal solution in the unconstrained space of independent variables.
>From physical point of view we are interested only in a meaningful solution. </span></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:16.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>therefore, we have to employ
regularization techniques (e.g. Gold deconvolution) and/or to confine the space
of allowed solutions to subspace of positive solutions. </span></p>

<p class=MsoNormal style='margin-left:18.0pt;text-align:justify'><span
style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-indent:5.7pt'><span style='font-size:16.0pt'><img
width=589 height=189 src="gif/TSpectrum_Deconvolution_wide3.jpg"></span></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Figure
14 The same spectrum like in Figure 13, outlined bars show the contents of
present components (peaks) </span></b></p>

<p class=MsoNormal style='text-align:justify;text-indent:8.55pt'><span
style='font-size:16.0pt'><img width=585 height=183
src="gif/TSpectrum_Deconvolution_wide4.jpg"></span></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Figure
15 Least squares solution of the system of linear equations without
regularization</span></b></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal><i><span style='font-size:16.0pt'>Example 9 – script Deconvolution_wide.c
:</span></i></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:14.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>when we employ Gold deconvolution
algorithm we obtain the result given in Fig. 16. One can observe that the
resulting spectrum is smooth. On the other hand the method is not able to
decompose completely the peaks in the spectrum.</span></p>

<p class=MsoNormal style='text-align:justify'> <img width=601 height=407
src="gif/TSpectrum_Deconvolution_wide5.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Figure
16 Example of Gold deconvolution for closely positioned wide peaks. The
original source spectrum is drawn with black color, the spectrum after the
deconvolution (10000 iterations) with red color</span></b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:#339966'>Script:</span></b></p>

<p class=MsoNormal>// Example to illustrate deconvolution function (class
TSpectrum).</p>

<p class=MsoNormal>// To execute this example, do</p>

<p class=MsoNormal>// root &gt; .x Deconvolution_wide.C</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>#include &lt;TSpectrum&gt;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>void Deconvolution_wide() {</p>

<p class=MsoNormal>   Int_t i;</p>

<p class=MsoNormal>   Double_t nbins = 256;</p>

<p class=MsoNormal>   Double_t xmin  = 0;</p>

<p class=MsoNormal>   Double_t xmax  = (Double_t)nbins;</p>

<p class=MsoNormal>   Float_t * source = new float[nbins];</p>

<p class=MsoNormal>   Float_t * response = new float[nbins];   </p>

<p class=MsoNormal>   TH1F *h = new
TH1F(&quot;h&quot;,&quot;Deconvolution&quot;,nbins,xmin,xmax);</p>

<p class=MsoNormal>   TH1F *d = new
TH1F(&quot;d&quot;,&quot;&quot;,nbins,xmin,xmax);      </p>

<p class=MsoNormal>   TFile *f = new
TFile(&quot;spectra\\TSpectrum.root&quot;);</p>

<p class=MsoNormal>   h=(TH1F*) f-&gt;Get(&quot;decon3;1&quot;);</p>

<p class=MsoNormal>   TFile *fr = new
TFile(&quot;spectra\\TSpectrum.root&quot;);</p>

<p class=MsoNormal>   d=(TH1F*)
fr-&gt;Get(&quot;decon_response_wide;1&quot;);   </p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++)
source[i]=h-&gt;GetBinContent(i + 1);</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++)
response[i]=d-&gt;GetBinContent(i + 1);   </p>

<p class=MsoNormal>   TCanvas *Decon1 =
gROOT-&gt;GetListOfCanvases()-&gt;FindObject(&quot;Decon1&quot;);</p>

<p class=MsoNormal>   if (!Decon1) Decon1 = new
TCanvas(&quot;Decon1&quot;,&quot;Deconvolution of closely positioned
overlapping peaks using Gold deconvolution method&quot;,10,10,1000,700);</p>

<p class=MsoNormal>   h-&gt;SetMaximum(30000);</p>

<p class=MsoNormal>   h-&gt;Draw(&quot;L&quot;);</p>

<p class=MsoNormal>   TSpectrum *s = new TSpectrum();</p>

<p class=MsoNormal>   s-&gt;Deconvolution(source,response,256,10000,1,1);</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++) d-&gt;SetBinContent(i +
1,source[i]);   </p>

<p class=MsoNormal>   d-&gt;SetLineColor(kRed);</p>

<p class=MsoNormal>   d-&gt;Draw(&quot;SAME L&quot;);   </p>

<p class=MsoNormal>}</p>

<p class=MsoNormal><i><span style='font-size:16.0pt'>&nbsp;</span></i></p>

<p class=MsoNormal><i><span style='font-size:16.0pt'>Example 10 – script
Deconvolution_wide_boost.c :</span></i></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:14.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>further let us employ boosting
operation into deconvolution (Fig. 17)</span></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><img width=601 height=407
src="gif/TSpectrum_Deconvolution_wide6.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Figure
17 The original source spectrum is drawn with black color, the spectrum after
the deconvolution with red color. Number of iterations = 200, number of
repetitions = 50 and boosting coefficient = 1.2.</span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>&nbsp;</span></b></p>

<table class=MsoTableGrid border=1 cellspacing=0 cellpadding=0 width=469
 style='width:351.65pt;margin-left:50.55pt;border-collapse:collapse;border:
 none'>
 <tr style='height:17.15pt'>
  <td valign=top style='border:solid windowtext 1.0pt;padding:0mm 5.4pt 0mm 5.4pt;
  height:17.15pt'>
  <p class=MsoNormal style='text-align:justify;text-indent:50.55pt'>Peak #</p>
  </td>
  <td valign=top style='border:solid windowtext 1.0pt;border-left:none;
  padding:0mm 5.4pt 0mm 5.4pt;height:17.15pt'>
  <p class=MsoNormal style='text-align:justify'>Original/Estimated (max)
  position</p>
  </td>
  <td valign=top style='border:solid windowtext 1.0pt;border-left:none;
  padding:0mm 5.4pt 0mm 5.4pt;height:17.15pt'>
  <p class=MsoNormal style='text-align:justify'>Original/Estimated area</p>
  </td>
 </tr>
 <tr style='height:17.15pt'>
  <td valign=top style='border:solid windowtext 1.0pt;border-top:none;
  padding:0mm 5.4pt 0mm 5.4pt;height:17.15pt'>
  <p class=MsoNormal style='text-align:justify'>1</p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0mm 5.4pt 0mm 5.4pt;height:17.15pt'>
  <p class=MsoNormal style='text-align:justify'>50/49</p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0mm 5.4pt 0mm 5.4pt;height:17.15pt'>
  <p class=MsoNormal style='text-align:justify'>10159/10419</p>
  </td>
 </tr>
 <tr style='height:17.15pt'>
  <td valign=top style='border:solid windowtext 1.0pt;border-top:none;
  padding:0mm 5.4pt 0mm 5.4pt;height:17.15pt'>
  <p class=MsoNormal style='text-align:justify'>2</p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0mm 5.4pt 0mm 5.4pt;height:17.15pt'>
  <p class=MsoNormal style='text-align:justify'>70/70</p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0mm 5.4pt 0mm 5.4pt;height:17.15pt'>
  <p class=MsoNormal style='text-align:justify'>60957/58933</p>
  </td>
 </tr>
 <tr style='height:17.15pt'>
  <td valign=top style='border:solid windowtext 1.0pt;border-top:none;
  padding:0mm 5.4pt 0mm 5.4pt;height:17.15pt'>
  <p class=MsoNormal style='text-align:justify'>3</p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0mm 5.4pt 0mm 5.4pt;height:17.15pt'>
  <p class=MsoNormal style='text-align:justify'>80/79</p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0mm 5.4pt 0mm 5.4pt;height:17.15pt'>
  <p class=MsoNormal style='text-align:justify'>20319/19935</p>
  </td>
 </tr>
 <tr style='height:17.15pt'>
  <td valign=top style='border:solid windowtext 1.0pt;border-top:none;
  padding:0mm 5.4pt 0mm 5.4pt;height:17.15pt'>
  <p class=MsoNormal style='text-align:justify'>4</p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0mm 5.4pt 0mm 5.4pt;height:17.15pt'>
  <p class=MsoNormal style='text-align:justify'>100/100</p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0mm 5.4pt 0mm 5.4pt;height:17.15pt'>
  <p class=MsoNormal style='text-align:justify'>101596/105413</p>
  </td>
 </tr>
 <tr style='height:18.1pt'>
  <td valign=top style='border:solid windowtext 1.0pt;border-top:none;
  padding:0mm 5.4pt 0mm 5.4pt;height:18.1pt'>
  <p class=MsoNormal style='text-align:justify'>5</p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0mm 5.4pt 0mm 5.4pt;height:18.1pt'>
  <p class=MsoNormal style='text-align:justify'>110/117</p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0mm 5.4pt 0mm 5.4pt;height:18.1pt'>
  <p class=MsoNormal style='text-align:justify;page-break-after:avoid'>10159/6676</p>
  </td>
 </tr>
</table>

<p class=MsoCaption style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoCaption style='text-align:justify'><span style='font-size:16.0pt'>Table
2 Results of the estimation of peaks in spectrum shown in Figure 17</span></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>&nbsp;</span></b></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-36.0pt'><span style='font-size:14.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>one can observe that peaks are
decomposed practically to delta functions. Number of peaks is correct,
positions of big peaks as well as their areas are relatively well estimated.
However there is a considerable error in the estimation of the position of
small right hand peak.</span></p>

<p class=MsoNormal style='margin-left:18.0pt;text-align:justify'><span
style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:#339966'>Script:</span></b></p>

<p class=MsoNormal style='text-align:justify'>// Example to illustrate
deconvolution function (class TSpectrum).</p>

<p class=MsoNormal style='text-align:justify'>// To execute this example, do</p>

<p class=MsoNormal style='text-align:justify'>// root &gt; .x
Deconvolution_wide_boost.C</p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'>#include &lt;TSpectrum&gt;</p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'>void Deconvolution_wide_boost() {</p>

<p class=MsoNormal style='text-align:justify'>   Int_t i;</p>

<p class=MsoNormal style='text-align:justify'>   Double_t nbins = 256;</p>

<p class=MsoNormal style='text-align:justify'>   Double_t xmin  = 0;</p>

<p class=MsoNormal style='text-align:justify'>   Double_t xmax  =
(Double_t)nbins;</p>

<p class=MsoNormal style='text-align:justify'>   Float_t * source = new
float[nbins];</p>

<p class=MsoNormal style='text-align:justify'>   Float_t * response = new
float[nbins];   </p>

<p class=MsoNormal style='text-align:justify'>   TH1F *h = new
TH1F(&quot;h&quot;,&quot;Deconvolution&quot;,nbins,xmin,xmax);</p>

<p class=MsoNormal style='text-align:justify'>   TH1F *d = new
TH1F(&quot;d&quot;,&quot;&quot;,nbins,xmin,xmax);      </p>

<p class=MsoNormal style='text-align:justify'>   TFile *f = new
TFile(&quot;spectra\\TSpectrum.root&quot;);</p>

<p class=MsoNormal style='text-align:justify'>   h=(TH1F*)
f-&gt;Get(&quot;decon3;1&quot;);</p>

<p class=MsoNormal style='text-align:justify'>   TFile *fr = new
TFile(&quot;spectra\\TSpectrum.root&quot;);</p>

<p class=MsoNormal style='text-align:justify'>   d=(TH1F*)
fr-&gt;Get(&quot;decon_response_wide;1&quot;);   </p>

<p class=MsoNormal style='text-align:justify'>   for (i = 0; i &lt; nbins; i++)
source[i]=h-&gt;GetBinContent(i + 1);</p>

<p class=MsoNormal style='text-align:justify'>   for (i = 0; i &lt; nbins; i++)
response[i]=d-&gt;GetBinContent(i + 1);   </p>

<p class=MsoNormal style='text-align:justify'>   TCanvas *Decon1 =
gROOT-&gt;GetListOfCanvases()-&gt;FindObject(&quot;Decon1&quot;);</p>

<p class=MsoNormal style='text-align:justify'>   if (!Decon1) Decon1 = new
TCanvas(&quot;Decon1&quot;,&quot;Deconvolution of closely positioned
overlapping peaks using boosted Gold deconvolution
method&quot;,10,10,1000,700);</p>

<p class=MsoNormal style='text-align:justify'>   h-&gt;SetMaximum(110000);</p>

<p class=MsoNormal style='text-align:justify'>   h-&gt;Draw(&quot;L&quot;);</p>

<p class=MsoNormal style='text-align:justify'>   TSpectrum *s = new
TSpectrum();</p>

<p class=MsoNormal style='text-align:justify'>  
s-&gt;Deconvolution(source,response,256,200,50,1.2);</p>

<p class=MsoNormal style='text-align:justify'>   for (i = 0; i &lt; nbins; i++)
d-&gt;SetBinContent(i + 1,source[i]);   </p>

<p class=MsoNormal style='text-align:justify'>   d-&gt;SetLineColor(kRed);</p>

<p class=MsoNormal style='text-align:justify'>   d-&gt;Draw(&quot;SAME
L&quot;);   </p>

<p class=MsoNormal style='text-align:justify'>}</p>

</div>

<!-- */
// --> End_Html

//_______________________________________________________________________________
const char *TSpectrum::DeconvolutionRL(float *source, const float *response,
                                      int ssize, int numberIterations,
                                      int numberRepetitions, double boost ) 
{   
/////////////////////////////////////////////////////////////////////////////
//   ONE-DIMENSIONAL DECONVOLUTION FUNCTION                                //
//   This function calculates deconvolution from source spectrum           //
//   according to response spectrum using Richardson-Lucy algorithm        //
//   The result is placed in the vector pointed by source pointer.         //
//                                                                         //
//   Function parameters:                                                  //
//   source:  pointer to the vector of source spectrum                     //
//   response:     pointer to the vector of response spectrum              //
//   ssize:    length of source and response spectra                       //
//   numberIterations, for details we refer to the reference given above   //
//   numberRepetitions, for repeated boosted deconvolution                 //
//   boost, boosting coefficient                                           //
//                                                                         //
/////////////////////////////////////////////////////////////////////////////
//
//Begin_Html <!--
/* -->
<div class=Section12>
<p class=MsoNormal><b><i><span style='font-size:16.0pt'>Richardson-Lucy
deconvolution algorithm</span></i></b></p>

<p class=MsoNormal style='margin-left:32.2pt;text-align:justify;text-indent:
-32.2pt'><span lang=SK style='font-size:14.0pt;font-family:Symbol'>·<span
style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span lang=SK style='font-size:16.0pt'>for discrete systems it
has the form</span></p>

<p class=MsoNormal style='margin-left:18.0pt;text-align:justify'><span lang=SK>      <sub><img
width=438 height=98 src="gif/TSpectrum_DeconvolutionRL1.gif"></sub>                       
</span></p>

<p class=MsoNormal style='margin-left:27.0pt;text-align:justify'><span lang=SK> <sub><img
width=124 height=39 src="gif/TSpectrum_DeconvolutionRL2.gif"></sub></span></p>

<p class=MsoNormal style='margin-left:28.5pt;text-align:justify;text-indent:
-28.5pt'><span lang=SK style='font-size:14.0pt;font-family:Symbol'>·<span
style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span lang=SK style='font-size:16.0pt'>for positive input data
and response matrix this iterative method forces the deconvoluted spectra to be
non-negative. </span></p>

<p class=MsoNormal style='margin-left:27.0pt;text-align:justify;text-indent:
-27.0pt'><span lang=SK style='font-size:14.0pt;font-family:Symbol'>·<span
style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span lang=SK style='font-size:16.0pt'>the Richardson-Lucy
iteration converges to the maximum likelihood solution for Poisson statistics
in the data.</span></p>

<p class=MsoNormal><i><span style='font-size:18.0pt'>&nbsp;</span></i></p>

<p class=MsoNormal><i><span style='font-size:18.0pt'>Function:</span></i></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:18.0pt'>const
<a href="http://root.cern.ch/root/html/ListOtransTypes.html#char">char</a>* <a
name="TSpectrum:Deconvolution1">Deconvolution</a>RL(<a
href="http://root.cern.ch/root/html/ListOtransTypes.html#float">float</a> *source,
const <a href="http://root.cern.ch/root/html/ListOtransTypes.html#float">float</a>
*respMatrix, <a href="http://root.cern.ch/root/html/ListOtransTypes.html#int">int</a> ssize,
<a href="http://root.cern.ch/root/html/ListOtransTypes.html#int">int</a>
numberIterations, <a href="http://root.cern.ch/root/html/ListOtransTypes.html#int">int</a>
numberRepetitions, <a
href="http://root.cern.ch/root/html/ListOtransTypes.html#double">double</a> boost)</span></b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>This
function calculates deconvolution from source spectrum according to response
spectrum using Richardson-Lucy deconvolution algorithm. The result is placed in
the vector pointed by source pointer. On successful completion it returns 0. On
error it returns pointer to the string describing error. If desired after every
numberIterations one can apply boosting operation (exponential function with
exponent given by boost coefficient) and repeat it numberRepetitions times
(see Gold deconvolution).</span></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><i><span style='font-size:16.0pt;color:red'>Parameters:</span></i></p>

<p class=MsoNormal>        <b><span style='font-size:14.0pt'>source</span></b>-pointer
to the vector of source spectrum                  </p>

<p class=MsoNormal>        <b><span style='font-size:14.0pt'>respMatrix</span></b>-pointer
to the vector of response spectrum                  </p>

<p class=MsoNormal>        <b><span style='font-size:14.0pt'>ssize</span></b>-length
of the spectrum vector                                 </p>

<p class=MsoNormal style='text-align:justify'>        <b><span
style='font-size:14.0pt'>numberIterations</span></b>-number of iterations
(parameter l in the Gold deconvolution  </p>

<p class=MsoNormal style='text-align:justify'>        algorithm)</p>

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

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>[1]
Abreu M.C. et al., A four-dimensional deconvolution method to correct NA38
experimental data, NIM A 405 (1998) 139.</span></p>

<p class=MsoNormal style='margin-left:18.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:16.0pt'>[2] Lucy L.B., A.J. 79 (1974) 745.</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>[3]
  Richardson W.H., J. Opt. Soc. Am. 62 (1972) 55.</span></p>

</div>

<!-- */
// --> End_Html

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
   if (lh_gold == -1)
      return "ZERO RESPONSE VECTOR";
   
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
//Begin_Html <!--
/* -->
<div class=Section13>

<p class=MsoNormal><b><span style='font-size:18.0pt'>Examples of Richardson-Lucy
deconvolution method</span></b></p>

<p class=MsoNormal><i><span style='font-size:16.0pt'>&nbsp;</span></i></p>

<p class=MsoNormal><i><span style='font-size:16.0pt'>Example 11 – script DeconvolutionRL_wide.c
:</span></i></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:14.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>when we employ Richardson-Lucy
deconvolution algorithm to our data from Fig. 13 we obtain the result given in
Fig. 18. One can observe improvements as compared to the result achieved by
Gold deconvolution.</span></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:14.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>neverthless it is unable to
decompose the multiplet.</span></p>

<p class=MsoNormal style='text-align:justify'> <img width=601 height=407
src="gif/TSpectrum_DeconvolutionRL_wide1.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Figure
18 Example of Richardson-Lucy deconvolution for closely positioned wide peaks.
The original source spectrum is drawn with black color, the spectrum after the
deconvolution (10000 iterations) with red color</span></b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:#339966'>Script:</span></b></p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:#339966'>&nbsp;</span></b></p>

<p class=MsoNormal>// Example to illustrate deconvolution function (class
TSpectrum).</p>

<p class=MsoNormal>// To execute this example, do</p>

<p class=MsoNormal>// root &gt; .x DeconvolutionRL_wide.C</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>#include &lt;TSpectrum&gt;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>void DeconvolutionRL_wide() {</p>

<p class=MsoNormal>   Int_t i;</p>

<p class=MsoNormal>   Double_t nbins = 256;</p>

<p class=MsoNormal>   Double_t xmin  = 0;</p>

<p class=MsoNormal>   Double_t xmax  = (Double_t)nbins;</p>

<p class=MsoNormal>   Float_t * source = new float[nbins];</p>

<p class=MsoNormal>   Float_t * response = new float[nbins];   </p>

<p class=MsoNormal>   TH1F *h = new
TH1F(&quot;h&quot;,&quot;Deconvolution&quot;,nbins,xmin,xmax);</p>

<p class=MsoNormal>   TH1F *d = new TH1F(&quot;d&quot;,&quot;&quot;,nbins,xmin,xmax);     
</p>

<p class=MsoNormal>   TFile *f = new
TFile(&quot;spectra\\TSpectrum.root&quot;);</p>

<p class=MsoNormal>   h=(TH1F*) f-&gt;Get(&quot;decon3;1&quot;);</p>

<p class=MsoNormal>   TFile *fr = new
TFile(&quot;spectra\\TSpectrum.root&quot;);</p>

<p class=MsoNormal>   d=(TH1F*)
fr-&gt;Get(&quot;decon_response_wide;1&quot;);   </p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++) source[i]=h-&gt;GetBinContent(i
+ 1);</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++)
response[i]=d-&gt;GetBinContent(i + 1);   </p>

<p class=MsoNormal>   TCanvas *Decon1 =
gROOT-&gt;GetListOfCanvases()-&gt;FindObject(&quot;Decon1&quot;);</p>

<p class=MsoNormal>   if (!Decon1) Decon1 = new
TCanvas(&quot;Decon1&quot;,&quot;Deconvolution of closely positioned
overlapping peaks using Richardson-Lucy deconvolution
method&quot;,10,10,1000,700);</p>

<p class=MsoNormal>   h-&gt;SetMaximum(30000);</p>

<p class=MsoNormal>   h-&gt;Draw(&quot;L&quot;);</p>

<p class=MsoNormal>   TSpectrum *s = new TSpectrum();</p>

<p class=MsoNormal>   s-&gt;DeconvolutionRL(source,response,256,10000,1,1);</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++) d-&gt;SetBinContent(i +
1,source[i]);   </p>

<p class=MsoNormal>   d-&gt;SetLineColor(kRed);</p>

<p class=MsoNormal>   d-&gt;Draw(&quot;SAME L&quot;);   </p>

<p class=MsoNormal>}</p>

<p class=MsoNormal><i><span style='font-size:16.0pt'>&nbsp;</span></i></p>

<p class=MsoNormal><i><span style='font-size:16.0pt'>Example 12 – script
DeconvolutionRL_wide_boost.c :</span></i></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'><span style='font-size:14.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>further let us employ boosting
operation into deconvolution (Fig. 19)</span></p>

<p class=MsoNormal><img width=601 height=407
src="gif/TSpectrum_DeconvolutionRL_wide2.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Figure
19 The original source spectrum is drawn with black color, the spectrum after
the deconvolution with red color. Number of iterations = 200, number of
repetitions = 50 and boosting coefficient = 1.2.</span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>&nbsp;</span></b></p>

<table class=MsoTableGrid border=1 cellspacing=0 cellpadding=0
 style='margin-left:28.5pt;border-collapse:collapse;border:none'>
 <tr>
  <td valign=top style='border:solid windowtext 1.0pt;padding:0mm 5.4pt 0mm 5.4pt'>
  <p class=MsoNormal style='text-align:justify'>Peak #</p>
  </td>
  <td valign=top style='border:solid windowtext 1.0pt;border-left:none;
  padding:0mm 5.4pt 0mm 5.4pt'>
  <p class=MsoNormal style='text-align:justify'>Original/Estimated (max)
  position</p>
  </td>
  <td valign=top style='border:solid windowtext 1.0pt;border-left:none;
  padding:0mm 5.4pt 0mm 5.4pt'>
  <p class=MsoNormal style='text-align:justify'>Original/Estimated area</p>
  </td>
 </tr>
 <tr>
  <td valign=top style='border:solid windowtext 1.0pt;border-top:none;
  padding:0mm 5.4pt 0mm 5.4pt'>
  <p class=MsoNormal style='text-align:justify'>1</p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0mm 5.4pt 0mm 5.4pt'>
  <p class=MsoNormal style='text-align:justify'>50/51</p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0mm 5.4pt 0mm 5.4pt'>
  <p class=MsoNormal style='text-align:justify'>10159/11426</p>
  </td>
 </tr>
 <tr>
  <td valign=top style='border:solid windowtext 1.0pt;border-top:none;
  padding:0mm 5.4pt 0mm 5.4pt'>
  <p class=MsoNormal style='text-align:justify'>2</p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0mm 5.4pt 0mm 5.4pt'>
  <p class=MsoNormal style='text-align:justify'>70/71</p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0mm 5.4pt 0mm 5.4pt'>
  <p class=MsoNormal style='text-align:justify'>60957/65003</p>
  </td>
 </tr>
 <tr>
  <td valign=top style='border:solid windowtext 1.0pt;border-top:none;
  padding:0mm 5.4pt 0mm 5.4pt'>
  <p class=MsoNormal style='text-align:justify'>3</p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0mm 5.4pt 0mm 5.4pt'>
  <p class=MsoNormal style='text-align:justify'>80/81</p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0mm 5.4pt 0mm 5.4pt'>
  <p class=MsoNormal style='text-align:justify'>20319/12813</p>
  </td>
 </tr>
 <tr>
  <td valign=top style='border:solid windowtext 1.0pt;border-top:none;
  padding:0mm 5.4pt 0mm 5.4pt'>
  <p class=MsoNormal style='text-align:justify'>4</p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0mm 5.4pt 0mm 5.4pt'>
  <p class=MsoNormal style='text-align:justify'>100/100</p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0mm 5.4pt 0mm 5.4pt'>
  <p class=MsoNormal style='text-align:justify'>101596/101851</p>
  </td>
 </tr>
 <tr>
  <td valign=top style='border:solid windowtext 1.0pt;border-top:none;
  padding:0mm 5.4pt 0mm 5.4pt'>
  <p class=MsoNormal style='text-align:justify'>5</p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0mm 5.4pt 0mm 5.4pt'>
  <p class=MsoNormal style='text-align:justify'>110/111</p>
  </td>
  <td valign=top style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;padding:0mm 5.4pt 0mm 5.4pt'>
  <p class=MsoNormal style='text-align:justify;page-break-after:avoid'>10159/8920</p>
  </td>
 </tr>
</table>

<p class=MsoCaption style='text-align:justify'><span style='font-size:16.0pt'>Table
3 Results of the estimation of peaks in spectrum shown in Figure 19</span></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>&nbsp;</span></b></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-36.0pt'><span style='font-size:14.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span><span style='font-size:16.0pt'>one can observe improvements in
the estimation of peak positions as compared to the results achieved by Gold
deconvolution</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:#339966'>Script:</span></b></p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'>// Example to illustrate
deconvolution function (class TSpectrum).</p>

<p class=MsoNormal style='text-align:justify'>// To execute this example, do</p>

<p class=MsoNormal style='text-align:justify'>// root &gt; .x
DeconvolutionRL_wide_boost.C</p>

<p class=MsoNormal style='text-align:justify'>#include &lt;TSpectrum&gt;</p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'>void DeconvolutionRL_wide_boost()
{</p>

<p class=MsoNormal style='text-align:justify'>   Int_t i;</p>

<p class=MsoNormal style='text-align:justify'>   Double_t nbins = 256;</p>

<p class=MsoNormal style='text-align:justify'>   Double_t xmin  = 0;</p>

<p class=MsoNormal style='text-align:justify'>   Double_t xmax  =
(Double_t)nbins;</p>

<p class=MsoNormal style='text-align:justify'>   Float_t * source = new
float[nbins];</p>

<p class=MsoNormal style='text-align:justify'>   Float_t * response = new
float[nbins];   </p>

<p class=MsoNormal style='text-align:justify'>   TH1F *h = new
TH1F(&quot;h&quot;,&quot;Deconvolution&quot;,nbins,xmin,xmax);</p>

<p class=MsoNormal style='text-align:justify'>   TH1F *d = new
TH1F(&quot;d&quot;,&quot;&quot;,nbins,xmin,xmax);      </p>

<p class=MsoNormal style='text-align:justify'>   TFile *f = new
TFile(&quot;spectra\\TSpectrum.root&quot;);</p>

<p class=MsoNormal style='text-align:justify'>   h=(TH1F*)
f-&gt;Get(&quot;decon3;1&quot;);</p>

<p class=MsoNormal style='text-align:justify'>   TFile *fr = new
TFile(&quot;spectra\\TSpectrum.root&quot;);</p>

<p class=MsoNormal style='text-align:justify'>   d=(TH1F*)
fr-&gt;Get(&quot;decon_response_wide;1&quot;);   </p>

<p class=MsoNormal style='text-align:justify'>   for (i = 0; i &lt; nbins; i++)
source[i]=h-&gt;GetBinContent(i + 1);</p>

<p class=MsoNormal style='text-align:justify'>   for (i = 0; i &lt; nbins; i++)
response[i]=d-&gt;GetBinContent(i + 1);   </p>

<p class=MsoNormal style='text-align:justify'>   TCanvas *Decon1 =
gROOT-&gt;GetListOfCanvases()-&gt;FindObject(&quot;Decon1&quot;);</p>

<p class=MsoNormal style='text-align:justify'>   if (!Decon1) Decon1 = new
TCanvas(&quot;Decon1&quot;,&quot;Deconvolution of closely positioned
overlapping peaks using boosted Richardson-Lucy deconvolution
method&quot;,10,10,1000,700);</p>

<p class=MsoNormal style='text-align:justify'>   h-&gt;SetMaximum(110000);</p>

<p class=MsoNormal style='text-align:justify'>   h-&gt;Draw(&quot;L&quot;);</p>

<p class=MsoNormal style='text-align:justify'>   TSpectrum *s = new
TSpectrum();</p>

<p class=MsoNormal style='text-align:justify'>   s-&gt;DeconvolutionRL(source,response,256,200,50,1.2);</p>

<p class=MsoNormal style='text-align:justify'>   for (i = 0; i &lt; nbins; i++)
d-&gt;SetBinContent(i + 1,source[i]);   </p>

<p class=MsoNormal style='text-align:justify'>   d-&gt;SetLineColor(kRed);</p>

<p class=MsoNormal style='text-align:justify'>   d-&gt;Draw(&quot;SAME
L&quot;);   </p>

<p class=MsoNormal style='text-align:justify'>}</p>

</div>

<!-- */
// --> End_Html

//_______________________________________________________________________________

const char *TSpectrum::Unfolding(float *source,
                                               const float **respMatrix,
                                               int ssizex, int ssizey,
                                               int numberIterations,
                                               int numberRepetitions, double boost) 
{   
/////////////////////////////////////////////////////////////////////////////
//        ONE-DIMENSIONAL UNFOLDING FUNCTION                                 
//        This function unfolds source spectrum                              
//        according to response matrix columns.                              
//        The result is placed in the vector pointed by source pointer.  
//                                                                         
//        Function parameters:                                               
//        source-pointer to the vector of source spectrum                  
//        respMatrix-pointer to the matrix of response spectra             
//        ssizex-length of source spectrum and # of columns of response matrix
//        ssizey-length of destination spectrum and # of rows of            
//              response matrix                                            
//        numberIterations, for details we refer to manual                
//        Note!!! ssizex must be >= ssizey                               
/////////////////////////////////////////////////////////////////////////////
//Begin_Html <!--
/* -->
<div class=Section14>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:20.0pt'>Unfolding</span></b></p>

<p class=MsoNormal style='text-align:justify'><i><span style='font-size:18.0pt'>&nbsp;</span></i></p>

<p class=MsoNormal style='text-align:justify'><i><span style='font-size:18.0pt'>Goal:
Decomposition of spectrum to a given set of component spectra</span></i></p>

<p class=MsoNormal><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal><span style='font-size:16.0pt'>Mathematical formulation of
the discrete linear system is</span></p>

<p class=MsoNormal style='margin-left:18.0pt'>

<table cellpadding=0 cellspacing=0 align=left>
 <tr>
  <td width=0 height=17></td>
 </tr>
 <tr>
  <td></td>
  <td><img width=588 height=89 src="gif/TSpectrum_Unfolding1.gif"></td>
 </tr>
</table>

<span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'>

<table cellpadding=0 cellspacing=0 align=left>
 <tr>
  <td width=0 height=3></td>
 </tr>
 <tr>
  <td></td>
  <td><img width=597 height=228 src="gif/TSpectrum_Unfolding2.gif"></td>
 </tr>
</table>

<span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<br clear=ALL>

<p class=MsoNormal><i><span style='font-size:18.0pt'>Function:</span></i></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:18.0pt'>const
<a href="http://root.cern.ch/root/html/ListOtransTypes.html#char">char</a>* Unfolding(<a
href="http://root.cern.ch/root/html/ListOtransTypes.html#float">float</a> *source,
const <a href="http://root.cern.ch/root/html/ListOtransTypes.html#float">float</a>
**respMatrix, <a href="http://root.cern.ch/root/html/ListOtransTypes.html#int">int</a> ssizex,
<a href="http://root.cern.ch/root/html/ListOtransTypes.html#int">int</a> ssizey, <a
href="http://root.cern.ch/root/html/ListOtransTypes.html#int">int</a>
numberIterations, <a href="http://root.cern.ch/root/html/ListOtransTypes.html#int">int</a>
numberRepetitions, <a
href="http://root.cern.ch/root/html/ListOtransTypes.html#double">double</a></span></b><b><span
style='font-size:16.0pt'> </span></b><b><span style='font-size:18.0pt'>boost)</span></b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>This
function unfolds source spectrum according to response matrix columns. The
result is placed in the vector pointed by source pointer.  The coefficients of
the resulting vector represent contents of the columns (weights) in the input
vector. On successful completion it returns 0. On error it returns pointer to
the string describing error. If desired after every numberIterations one can
apply boosting operation (exponential function with exponent given by boost
coefficient) and repeat it numberRepetitions times. For details we refer to
[1].</span></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><i><span style='font-size:16.0pt;color:red'>Parameters:</span></i></p>

<p class=MsoNormal>        <b><span style='font-size:14.0pt'>source</span></b>-pointer
to the vector of source spectrum                  </p>

<p class=MsoNormal>        <b><span style='font-size:14.0pt'>respMatrix</span></b>-pointer
to the matrix of response spectra                  </p>

<p class=MsoNormal>        <b><span style='font-size:14.0pt'>ssizex</span></b>-length
of source spectrum and # of columns of the response matrix</p>

<p class=MsoNormal>        <b><span style='font-size:14.0pt'>ssizey</span></b>-length
of destination spectrum and # of rows of the response matrix                                
</p>

<p class=MsoNormal style='text-align:justify'>        <b><span
style='font-size:14.0pt'>numberIterations</span></b>-number of iterations </p>

<p class=MsoNormal style='text-align:justify'>        <b><span
style='font-size:14.0pt'>numberRepetitions</span></b>-number of repetitions
for boosted deconvolution. It must be </p>

<p class=MsoNormal style='text-align:justify'>        greater or equal to one.</p>

<p class=MsoNormal style='text-align:justify'>        <b><span
style='font-size:14.0pt'>boost</span></b>-boosting coefficient, applies only
if numberRepetitions is greater than one.  </p>

<p class=MsoNormal style='text-align:justify'>       <span style='font-size:
14.0pt;color:fuchsia'> Recommended range &lt;1,2&gt;.</span></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:14.0pt;
color:fuchsia'>Note!!! sizex must be &gt;= sizey After decomposition the
resulting channels are written back to the first sizey channels of the source
spectrum. </span></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><b><i><span style='font-size:18.0pt'>Reference:</span></i></b></p>

<p class=MsoNormal style='text-align:justify'><span lang=SK style='font-size:
16.0pt'>[1] Jandel M., </span><span style='font-size:16.0pt'>Morhá&#269; M., </span><span
lang=SK style='font-size:16.0pt'>Kliman J., Krupa L.,</span><span
style='font-size:16.0pt'> Matou</span><span lang=SK style='font-size:16.0pt'>šek
V., Hamilton J. H., Ramaya A. V.</span><span style='font-size:16.0pt'>:
Decomposition of continuum gamma-ray spectra using synthetized response matrix.
NIM A 516 (2004), 172-183.</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

</div>

<!-- */
// --> End_Html
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
   if (lhx == -1)
      return ("ZERO COLUMN IN RESPONSE MATRIX");
   
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
//Begin_Html <!--
/* -->
<div class=Section15>

<p class=MsoNormal><b><span style='font-size:18.0pt'>Example of unfolding</span></b></p>

<p class=MsoNormal><i><span style='font-size:16.0pt'>&nbsp;</span></i></p>

<p class=MsoNormal><i><span style='font-size:16.0pt'>Example 13 – script Unfolding.c
:</span></i></p>

<p class=MsoNormal><i><span style='font-size:16.0pt'><img width=442 height=648
src="gif/TSpectrum_Unfolding3.gif"></span></i></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt;
font-family:Arial'>Fig. 20  Response matrix composed of neutron spectra of pure
chemical elements</span></b><b><span style='font-size:16.0pt'> </span></b></p>

<p class=MsoNormal><img width=604 height=372
src="gif/TSpectrum_Unfolding2.jpg"></p>

<p class=MsoNormal><b><span style='font-size:16.0pt;font-family:Arial'>Fig. 21
Source neutron spectrum to be decomposed</span></b></p>

<p class=MsoNormal><img width=600 height=360
src="gif/TSpectrum_Unfolding3.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt;
font-family:Arial'>Fig. 22 Spectrum after decomposition, contains 10
coefficients, which correspond to contents of chemical components (dominant
8-th and 10-th components, i.e. O, Si)</span></b></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:#339966'>Script:</span></b></p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'>// Example to illustrate
unfolding function (class TSpectrum).</p>

<p class=MsoNormal style='text-align:justify'>// To execute this example, do</p>

<p class=MsoNormal style='text-align:justify'>// root &gt; .x Unfolding.C</p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'>#include &lt;TSpectrum&gt;</p>

<p class=MsoNormal style='text-align:justify'>void Unfolding() {</p>

<p class=MsoNormal style='text-align:justify'>   Int_t i, j;</p>

<p class=MsoNormal style='text-align:justify'>   Int_t nbinsx = 2048;</p>

<p class=MsoNormal style='text-align:justify'>   Int_t nbinsy = 10;   </p>

<p class=MsoNormal style='text-align:justify'>   Double_t xmin  = 0;</p>

<p class=MsoNormal style='text-align:justify'>   Double_t xmax  =
(Double_t)nbinsx;</p>

<p class=MsoNormal style='text-align:justify'>   Double_t ymin  = 0;</p>

<p class=MsoNormal style='text-align:justify'>   Double_t ymax  =
(Double_t)nbinsy;   </p>

<p class=MsoNormal style='text-align:justify'>   Float_t * source = new
float[nbinsx];</p>

<p class=MsoNormal style='text-align:justify'>   Float_t ** response = new
float *[nbinsy];   </p>

<p class=MsoNormal style='text-align:justify'>   for (i=0;i&lt;nbinsy;i++)</p>

<p class=MsoNormal style='text-align:justify'>                                    response[i]=new
float[nbinsx];   </p>

<p class=MsoNormal style='text-align:justify'>   TH1F *h = new
TH1F(&quot;h&quot;,&quot;&quot;,nbinsx,xmin,xmax);</p>

<p class=MsoNormal style='text-align:justify'>   TH1F *d = new
TH1F(&quot;d&quot;,&quot;Decomposition - unfolding&quot;,nbinsx,xmin,xmax);   </p>

<p class=MsoNormal style='text-align:justify'>   TH2F *decon_unf_resp = new
TH2F(&quot;decon_unf_resp&quot;,&quot;Root File&quot;,nbinsy,ymin,ymax,nbinsx,xmin,xmax);</p>

<p class=MsoNormal style='text-align:justify'>   TFile *f = new
TFile(&quot;spectra\\TSpectrum.root&quot;);</p>

<p class=MsoNormal style='text-align:justify'>   h=(TH1F*)
f-&gt;Get(&quot;decon_unf_in;1&quot;);</p>

<p class=MsoNormal style='text-align:justify'>   TFile *fr = new
TFile(&quot;spectra\\TSpectrum.root&quot;);</p>

<p class=MsoNormal style='text-align:justify'>   decon_unf_resp = (TH2F*)
fr-&gt;Get(&quot;decon_unf_resp;1&quot;);     </p>

<p class=MsoNormal style='text-align:justify'>   for (i = 0; i &lt; nbinsx;
i++) source[i] = h-&gt;GetBinContent(i + 1);</p>

<p class=MsoNormal style='text-align:justify'>   for (i = 0; i &lt; nbinsy;
i++){</p>

<p class=MsoNormal style='text-align:justify'>      for (j = 0; j&lt; nbinsx;
j++){</p>

<p class=MsoNormal style='text-align:justify'>             response[i][j] =
decon_unf_resp-&gt;GetBinContent(i + 1, j + 1);</p>

<p class=MsoNormal style='text-align:justify'>      }</p>

<p class=MsoNormal style='text-align:justify'>   }     </p>

<p class=MsoNormal style='text-align:justify'>   TCanvas *Decon1 =
gROOT-&gt;GetListOfCanvases()-&gt;FindObject(&quot;Decon1&quot;);</p>

<p class=MsoNormal style='text-align:justify'>   if (!Decon1) Decon1 = new
TCanvas(&quot;Decon1&quot;,&quot;Decon1&quot;,10,10,1000,700);</p>

<p class=MsoNormal style='text-align:justify'>   h-&gt;Draw(&quot;L&quot;);   </p>

<p class=MsoNormal style='text-align:justify'>   TSpectrum *s = new
TSpectrum();</p>

<p class=MsoNormal style='text-align:justify'>   s-&gt;Unfolding(source,response,nbinsx,nbinsy,1000,1,1);</p>

<p class=MsoNormal style='text-align:justify'>   for (i = 0; i &lt; nbinsy;
i++) d-&gt;SetBinContent(i + 1,source[i]); </p>

<p class=MsoNormal style='text-align:justify'>   d-&gt;SetLineColor(kRed);   </p>

<p class=MsoNormal style='text-align:justify'>  
d-&gt;SetAxisRange(0,nbinsy);     </p>

<p class=MsoNormal style='text-align:justify'>   d-&gt;Draw(&quot;&quot;);</p>

<p class=MsoNormal style='text-align:justify'>}</p>

</div>

<!-- */
// --> End_Html

//_____________________________________________________________________________

Int_t TSpectrum::SearchHighRes(float *source,float *destVector, int ssize,
                                     float sigma, double threshold,
                                     bool backgroundRemove,int deconIterations,
                                     bool markov, int averWindow)
{
/////////////////////////////////////////////////////////////////////////////
//        ONE-DIMENSIONAL HIGH-RESOLUTION PEAK SEARCH FUNCTION            
//        This function searches for peaks in source spectrum             
//      It is based on deconvolution method. First the background is       
//      removed (if desired), then Markov spectrum is calculated          
//      (if desired), then the response function is generated             
//      according to given sigma and deconvolution is carried out.        
//                                                                        
//        Function parameters:                                            
//        source-pointer to the vector of source spectrum                  
//        destVector-pointer to the vector of resulting deconvolved spectrum     */
//        ssize-length of source spectrum                                
//        sigma-sigma of searched peaks, for details we refer to manual  
//        threshold-threshold value in % for selected peaks, peaks with  
//                amplitude less than threshold*highest_peak/100          
//                are ignored, see manual                                 
//      backgroundRemove-logical variable, set if the removal of          
//                background before deconvolution is desired               
//      deconIterations-number of iterations in deconvolution operation   
//      markov-logical variable, if it is true, first the source spectrum 
//             is replaced by new spectrum calculated using Markov         
//             chains method.                                              
//        averWindow-averanging window of searched peaks, for details     
//                  we refer to manual (applies only for Markov method)    
//                                                                        
/////////////////////////////////////////////////////////////////////////////
//
//Begin_Html <!--
/* -->
<div class=Section18>

<p class=MsoNormal><b><span style='font-size:20.0pt'>Peaks searching</span></b></p>

<p class=MsoNormal style='text-align:justify'><i><span style='font-size:18.0pt'>&nbsp;</span></i></p>

<p class=MsoNormal style='text-align:justify'><i><span style='font-size:18.0pt'>Goal:
to identify automatically the peaks in spectrum with the presence of the
continuous background and statistical fluctuations - noise.</span></i><span
style='font-size:18.0pt'> </span></p>

<p class=MsoNormal><span style='font-size:16.0pt;font-family:Arial'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt;
font-family:Arial'>The common problems connected with correct peak
identification are</span></p>

<ul style='margin-top:0mm' type=disc>
 <li class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt;
     font-family:Arial'>non-sensitivity to noise, i.e., only statistically
     relevant peaks should be identified.</span></li>
 <li class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt;
     font-family:Arial'>non-sensitivity of the algorithm to continuous
     background</span></li>
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
href="http://root.cern.ch/root/html/ListOtransTypes.html#Int_t">Int_t</a> <a
name="TSpectrum:SearchHighRes"></a><a
href="http://root.cern.ch/root/html/src/TSpectrum.cxx.html#TSpectrum:SearchHighRes">SearchHighRes</a>(<a
href="http://root.cern.ch/root/html/ListOtransTypes.html#float">float</a> *source,<a
href="http://root.cern.ch/root/html/ListOtransTypes.html#float">float</a> *destVector, <a
href="http://root.cern.ch/root/html/ListOtransTypes.html#int">int</a> ssize, <a
href="http://root.cern.ch/root/html/ListOtransTypes.html#float">float</a> sigma, <a
href="http://root.cern.ch/root/html/ListOtransTypes.html#double">double</a> threshold,
<a href="http://root.cern.ch/root/html/ListOtransTypes.html#bool">bool</a> backgroundRemove,<a
href="http://root.cern.ch/root/html/ListOtransTypes.html#int">int</a> deconIterations,
<a href="http://root.cern.ch/root/html/ListOtransTypes.html#bool">bool</a> markov,
<a href="http://root.cern.ch/root/html/ListOtransTypes.html#int">int</a> averWindow)
  </span></b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>This
function searches for peaks in source spectrum. It is based on deconvolution
method. First the background is removed (if desired), then Markov smoothed spectrum
is calculated (if desired), then the response function is generated according
to given sigma and deconvolution is carried out. The order of peaks is arranged
according to their heights in the spectrum after background elimination. The
highest peak is the first in the list. On success it returns number of found
peaks.</span></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><i><span style='font-size:16.0pt;color:red'>Parameters:</span></i></p>

<p class=MsoNormal style='text-align:justify'>        <b><span
style='font-size:14.0pt'>source</span></b>-pointer to the vector of source
spectrum                  </p>

<p class=MsoNormal style='text-align:justify'>        <b><span
style='font-size:14.0pt'>destVector</span></b>-resulting spectrum after
deconvolution</p>

<p class=MsoNormal style='text-align:justify'>        <b><span
style='font-size:14.0pt'>ssize</span></b>-length of the source and destination
spectra                </p>

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

<p class=MsoNormal><img width=600 height=375 src="gif/TSpectrum_Searching1.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Fig.
27 An example of one-dimensional synthetic spectrum with found peaks denoted by
markers</span></b></p>

<p class=MsoNormal><b><i><span style='font-size:18.0pt'>&nbsp;</span></i></b></p>

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
         	  if(b < 0)
         	     b = 0;
            working_space[size_ext + j] = b - working_space[size_ext + j];
         }

         else if(j >= ssize + shift){
         	  a = j - (ssize - 1 + shift);
         	  b = source[ssize - 1];
         	  if(b < 0)
         	     b = 0;         	  
            working_space[size_ext + j] = b - working_space[size_ext + j];
         }

         else{
            working_space[size_ext + j] = source[j - shift] - working_space[size_ext + j];
         }
      }      
      for(j = 0;j < size_ext; j++){
      	if(working_space[size_ext + j] < 0)
      	   working_space[size_ext + j] = 0;
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

//_____________________________________________________________________________

Int_t TSpectrum::Search1HighRes(float *source,float *destVector, int ssize,
                                     float sigma, double threshold,
                                     bool backgroundRemove,int deconIterations,
                                     bool markov, int averWindow)
{
//  Old name of SearcHighRes introduced for back compatibility
// This function will be removed after the June 2006 release
   
   return SearchHighRes(source,destVector,ssize,sigma,threshold,backgroundRemove,
                        deconIterations,markov,averWindow);
}


//Begin_Html <!--
/* -->
<div class=Section19>

<p class=MsoNormal><b><span style='font-size:18.0pt'>Examples of peak searching
method</span></b></p>

<p class=MsoNormal><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'><a
href="http://root.cern.ch/root/html/src/TSpectrum.cxx.html#TSpectrum:SearchHighRes"
target="_parent">SearchHighRes</a> function provides users with the possibility
to vary the input parameters and with the access to the output deconvolved data
in the destination spectrum. Based on the output data one can tune the
parameters. </span></p>

<p class=MsoNormal><i><span style='font-size:16.0pt'>Example 15 – script SearchHR1.c:</span></i></p>

<p class=MsoNormal><span style='font-size:16.0pt'><img width=600 height=321
src="gif/TSpectrum_Searching1.jpg"></span></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Fig.
28 One-dimensional spectrum with found peaks denoted by markers, 3 iterations
steps in the deconvolution</span></b></p>

<p class=MsoNormal><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal><span style='font-size:16.0pt'><img width=600 height=323
src="gif/TSpectrum_Searching2.jpg"></span></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>Fig.
29 One-dimensional spectrum with found peaks denoted by markers, 8 iterations
steps in the deconvolution</span></b></p>

<p class=MsoNormal><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:#339966'>Script:</span></b></p>

<p class=MsoNormal>// Example to illustrate high resolution peak searching
function (class TSpectrum).</p>

<p class=MsoNormal>// To execute this example, do</p>

<p class=MsoNormal>// root &gt; .x SearchHR1.C</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>#include &lt;TSpectrum&gt;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>void SearchHR1() {</p>

<p class=MsoNormal>            </p>

<p class=MsoNormal>   Float_t fPositionX[100];</p>

<p class=MsoNormal>   Float_t fPositionY[100];   </p>

<p class=MsoNormal>   Int_t fNPeaks = 0;     </p>

<p class=MsoNormal>   Int_t i,nfound,bin;</p>

<p class=MsoNormal>   Double_t nbins = 1024,a;</p>

<p class=MsoNormal>   Double_t xmin  = 0;</p>

<p class=MsoNormal>   Double_t xmax  = (Double_t)nbins;</p>

<p class=MsoNormal>   Float_t * source = new float[nbins];</p>

<p class=MsoNormal>   Float_t * dest = new float[nbins];   </p>

<p class=MsoNormal>   TH1F *h = new TH1F(&quot;h&quot;,&quot;High resolution
peak searching, number of iterations = 3&quot;,nbins,xmin,xmax);</p>

<p class=MsoNormal>   TH1F *d = new
TH1F(&quot;d&quot;,&quot;&quot;,nbins,xmin,xmax);      </p>

<p class=MsoNormal>   TFile *f = new
TFile(&quot;spectra\\TSpectrum.root&quot;);</p>

<p class=MsoNormal>   h=(TH1F*) f-&gt;Get(&quot;search2;1&quot;);   </p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++) source[i]=h-&gt;GetBinContent(i
+ 1);   </p>

<p class=MsoNormal>   TCanvas *Search =
gROOT-&gt;GetListOfCanvases()-&gt;FindObject(&quot;Search&quot;);</p>

<p class=MsoNormal>   if (!Search) Search = new
TCanvas(&quot;Search&quot;,&quot;Search&quot;,10,10,1000,700);</p>

<p class=MsoNormal>   h-&gt;SetMaximum(4000);      </p>

<p class=MsoNormal>   h-&gt;Draw(&quot;L&quot;);</p>

<p class=MsoNormal>   TSpectrum *s = new TSpectrum();</p>

<p class=MsoNormal>   nfound = s-&gt;SearchHighRes(source, dest, nbins, 8, 2,
kTRUE, 3, kTRUE, 3);</p>

<p class=MsoNormal>   Float_t *xpeaks = s-&gt;GetPositionX(); </p>

<p class=MsoNormal>   for (i = 0; i &lt; nfound; i++) {</p>

<p class=MsoNormal>        a=xpeaks[i];</p>

<p class=MsoNormal>        bin = 1 + Int_t(a + 0.5);</p>

<p class=MsoNormal>        fPositionX[i] = h-&gt;GetBinCenter(bin);</p>

<p class=MsoNormal>        fPositionY[i] = h-&gt;GetBinContent(bin);</p>

<p class=MsoNormal>   }</p>

<p class=MsoNormal>   TPolyMarker * pm = (TPolyMarker*)h-&gt;GetListOfFunctions()-&gt;FindObject(&quot;TPolyMarker&quot;);</p>

<p class=MsoNormal>   if (pm) {</p>

<p class=MsoNormal>      h-&gt;GetListOfFunctions()-&gt;Remove(pm);</p>

<p class=MsoNormal>      delete pm;</p>

<p class=MsoNormal>   }</p>

<p class=MsoNormal>   pm = new TPolyMarker(nfound, fPositionX, fPositionY);</p>

<p class=MsoNormal>   h-&gt;GetListOfFunctions()-&gt;Add(pm);</p>

<p class=MsoNormal>   pm-&gt;SetMarkerStyle(23);</p>

<p class=MsoNormal>   pm-&gt;SetMarkerColor(kRed);</p>

<p class=MsoNormal>   pm-&gt;SetMarkerSize(1.3);</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++) d-&gt;SetBinContent(i +
1,dest[i]);</p>

<p class=MsoNormal>   d-&gt;SetLineColor(kRed);   </p>

<p class=MsoNormal>   d-&gt;Draw(&quot;SAME&quot;); </p>

<p class=MsoNormal>   printf(&quot;Found %d candidate peaks\n&quot;,nfound);  </p>

<p class=MsoNormal>   for(i=0;i&lt;nfound;i++)</p>

<p class=MsoNormal>      printf(&quot;posx= %d, posy= %d\n&quot;,fPositionX[i],
fPositionY[i]);        </p>

<p class=MsoNormal>   }</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><i><span style='font-size:16.0pt'>Example 16 – script SearchHR3.c:</span></i></p>

<table class=MsoNormalTable border=0 cellspacing=0 cellpadding=0 width=131
 style='width:97.9pt;margin-left:131.85pt'>
 <tr style='height:12.75pt'>
  <td width=33 valign=top style='width:24.85pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:12.75pt'>
  <p class=MsoNormal style='text-align:justify'>Peak #</p>
  </td>
  </nobr>
  <td width=54 valign=top style='width:40.85pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:12.75pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>Position</nobr></p>
  </td>
  <td width=43 valign=top style='width:32.2pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:12.75pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>Sigma</nobr></p>
  </td>
 </tr>
 <tr style='height:12.75pt'>
  <td width=33 valign=top style='width:24.85pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:12.75pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>1</nobr></p>
  </td>
  <td width=54 valign=top style='width:40.85pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:12.75pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>118</nobr></p>
  </td>
  <td width=43 valign=top style='width:32.2pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:12.75pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>26</nobr></p>
  </td>
 </tr>
 <tr style='height:12.75pt'>
  <td width=33 valign=top style='width:24.85pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:12.75pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>2</nobr></p>
  </td>
  <td width=54 valign=top style='width:40.85pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:12.75pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>162</nobr></p>
  </td>
  <td width=43 valign=top style='width:32.2pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:12.75pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>41</nobr></p>
  </td>
 </tr>
 <tr style='height:12.75pt'>
  <td width=33 valign=top style='width:24.85pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:12.75pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>3</nobr></p>
  </td>
  <td width=54 valign=top style='width:40.85pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:12.75pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>310</nobr></p>
  </td>
  <td width=43 valign=top style='width:32.2pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:12.75pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>4</nobr></p>
  </td>
 </tr>
 <tr style='height:13.5pt'>
  <td width=33 valign=top style='width:24.85pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:13.5pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>4</nobr></p>
  </td>
  <td width=54 valign=top style='width:40.85pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:13.5pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>330</nobr></p>
  </td>
  <td width=43 valign=top style='width:32.2pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:13.5pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>8</nobr></p>
  </td>
 </tr>
 <tr style='height:12.75pt'>
  <td width=33 valign=top style='width:24.85pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:12.75pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>5</nobr></p>
  </td>
  <td width=54 valign=top style='width:40.85pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:12.75pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>482</nobr></p>
  </td>
  <td width=43 valign=top style='width:32.2pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:12.75pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>22</nobr></p>
  </td>
 </tr>
 <tr style='height:12.75pt'>
  <td width=33 valign=top style='width:24.85pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:12.75pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>6</nobr></p>
  </td>
  <td width=54 valign=top style='width:40.85pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:12.75pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>491</nobr></p>
  </td>
  <td width=43 valign=top style='width:32.2pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:12.75pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>26</nobr></p>
  </td>
 </tr>
 <tr style='height:12.75pt'>
  <td width=33 valign=top style='width:24.85pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:12.75pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>7</nobr></p>
  </td>
  <td width=54 valign=top style='width:40.85pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:12.75pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>740</nobr></p>
  </td>
  <td width=43 valign=top style='width:32.2pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:12.75pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>21</nobr></p>
  </td>
 </tr>
 <tr style='height:12.75pt'>
  <td width=33 valign=top style='width:24.85pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:12.75pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>8</nobr></p>
  </td>
  <td width=54 valign=top style='width:40.85pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:12.75pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>852</nobr></p>
  </td>
  <td width=43 valign=top style='width:32.2pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:12.75pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>15</nobr></p>
  </td>
 </tr>
 <tr style='height:12.75pt'>
  <td width=33 valign=top style='width:24.85pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:12.75pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>9</nobr></p>
  </td>
  <td width=54 valign=top style='width:40.85pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:12.75pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>954</nobr></p>
  </td>
  <td width=43 valign=top style='width:32.2pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:12.75pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>12</nobr></p>
  </td>
 </tr>
 <tr style='height:12.75pt'>
  <td width=33 valign=top style='width:24.85pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:12.75pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>10</nobr></p>
  </td>
  <td width=54 valign=top style='width:40.85pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:12.75pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>989</nobr></p>
  </td>
  <td width=43 valign=top style='width:32.2pt;border:solid black 1.0pt;
  padding:0mm 0mm 0mm 0mm;height:12.75pt'>
  <p class=MsoNormal style='text-align:justify'><nobr>13</nobr></p>
  </td>
 </tr>
</table>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><b><span style='font-size:14.0pt'>Table 4 Positions and
sigma of peaks in the following examples</span></b></p>

<p class=MsoNormal><span style='font-size:18.0pt'>&nbsp;</span></p>

<p class=MsoNormal><img width=600 height=328
src="gif/TSpectrum_Searching3.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:14.0pt'>Fig.
30 Influence of number of iterations (3-red, 10-blue, 100- green,
1000-magenta), sigma=8, smoothing width=3</span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:14.0pt'>&nbsp;</span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:14.0pt'><img
width=600 height=321 src="gif/TSpectrum_Searching4.jpg"></span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:14.0pt'>Fig.
31 Influence of sigma (3-red, 8-blue, 20- green, 43-magenta), num. iter.=10,
sm. width=3</span></b></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:18.0pt'>&nbsp;</span></p>

<p class=MsoNormal><img width=600 height=323
src="gif/TSpectrum_Searching5.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:14.0pt'>Fig.
32 Influence smoothing width (0-red, 3-blue, 7- green, 20-magenta), num.
iter.=10, sigma=8</span></b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:14.0pt'>&nbsp;</span></b></p>

<p class=MsoNormal><b><span style='font-size:16.0pt;color:#339966'>Script:</span></b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>// Example to illustrate the influence of number of
iterations in deconvolution in high resolution peak searching function (class
TSpectrum).</p>

<p class=MsoNormal>// To execute this example, do</p>

<p class=MsoNormal>// root &gt; .x SearchHR3.C</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>#include &lt;TSpectrum&gt;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>void SearchHR3() {</p>

<p class=MsoNormal>   Float_t fPositionX[100];</p>

<p class=MsoNormal>   Float_t fPositionY[100];   </p>

<p class=MsoNormal>   Int_t fNPeaks = 0;     </p>

<p class=MsoNormal>   Int_t i,nfound,bin;</p>

<p class=MsoNormal>   Double_t nbins = 1024,a;</p>

<p class=MsoNormal>   Double_t xmin  = 0;</p>

<p class=MsoNormal>   Double_t xmax  = (Double_t)nbins;</p>

<p class=MsoNormal>   Float_t * source = new float[nbins];</p>

<p class=MsoNormal>   Float_t * dest = new float[nbins];   </p>

<p class=MsoNormal>   TH1F *h = new TH1F(&quot;h&quot;,&quot;Influence of # of
iterations in deconvolution in peak searching&quot;,nbins,xmin,xmax);</p>

<p class=MsoNormal>   TH1F *d1 = new
TH1F(&quot;d1&quot;,&quot;&quot;,nbins,xmin,xmax);      </p>

<p class=MsoNormal>   TH1F *d2 = new
TH1F(&quot;d2&quot;,&quot;&quot;,nbins,xmin,xmax);      </p>

<p class=MsoNormal>   TH1F *d3 = new
TH1F(&quot;d3&quot;,&quot;&quot;,nbins,xmin,xmax);      </p>

<p class=MsoNormal>   TH1F *d4 = new
TH1F(&quot;d4&quot;,&quot;&quot;,nbins,xmin,xmax);               </p>

<p class=MsoNormal>   TFile *f = new
TFile(&quot;spectra\\TSpectrum.root&quot;);</p>

<p class=MsoNormal>   h=(TH1F*) f-&gt;Get(&quot;search3;1&quot;);   </p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++)
source[i]=h-&gt;GetBinContent(i + 1);   </p>

<p class=MsoNormal>   TCanvas *Search =
gROOT-&gt;GetListOfCanvases()-&gt;FindObject(&quot;Search&quot;);</p>

<p class=MsoNormal>   if (!Search) Search = new
TCanvas(&quot;Search&quot;,&quot;Search&quot;,10,10,1000,700);</p>

<p class=MsoNormal>   h-&gt;SetMaximum(1300);         </p>

<p class=MsoNormal>   h-&gt;Draw(&quot;L&quot;);</p>

<p class=MsoNormal>   TSpectrum *s = new TSpectrum();</p>

<p class=MsoNormal>   nfound = s-&gt;SearchHighRes(source, dest, nbins, 8, 2,
kTRUE, 3, kTRUE, 3);</p>

<p class=MsoNormal>   Float_t *xpeaks = s-&gt;GetPositionX(); </p>

<p class=MsoNormal>   for (i = 0; i &lt; nfound; i++) {</p>

<p class=MsoNormal>        a=xpeaks[i];</p>

<p class=MsoNormal>        bin = 1 + Int_t(a + 0.5);</p>

<p class=MsoNormal>        fPositionX[i] = h-&gt;GetBinCenter(bin);</p>

<p class=MsoNormal>        fPositionY[i] = h-&gt;GetBinContent(bin);</p>

<p class=MsoNormal>   }   </p>

<p class=MsoNormal>   TPolyMarker * pm =
(TPolyMarker*)h-&gt;GetListOfFunctions()-&gt;FindObject(&quot;TPolyMarker&quot;);</p>

<p class=MsoNormal>   if (pm) {</p>

<p class=MsoNormal>      h-&gt;GetListOfFunctions()-&gt;Remove(pm);</p>

<p class=MsoNormal>      delete pm;</p>

<p class=MsoNormal>   }</p>

<p class=MsoNormal>   pm = new TPolyMarker(nfound, fPositionX, fPositionY);</p>

<p class=MsoNormal>   h-&gt;GetListOfFunctions()-&gt;Add(pm);</p>

<p class=MsoNormal>   pm-&gt;SetMarkerStyle(23);</p>

<p class=MsoNormal>   pm-&gt;SetMarkerColor(kRed);</p>

<p class=MsoNormal>   pm-&gt;SetMarkerSize(1.3);</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++) d1-&gt;SetBinContent(i +
1,dest[i]);</p>

<p class=MsoNormal>   h-&gt;Draw(&quot;&quot;);</p>

<p class=MsoNormal>   d1-&gt;SetLineColor(kRed);      </p>

<p class=MsoNormal>   d1-&gt;Draw(&quot;SAME&quot;); </p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++)
source[i]=h-&gt;GetBinContent(i + 1);</p>

<p class=MsoNormal>   s-&gt;SearchHighRes(source, dest, nbins, 8, 2, kTRUE, 10,
kTRUE, 3);      </p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++) d2-&gt;SetBinContent(i +
1,dest[i]);</p>

<p class=MsoNormal>   d2-&gt;SetLineColor(kBlue);      </p>

<p class=MsoNormal>   d2-&gt;Draw(&quot;SAME&quot;);</p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++)
source[i]=h-&gt;GetBinContent(i + 1);</p>

<p class=MsoNormal>   s-&gt;SearchHighRes(source, dest, nbins, 8, 2, kTRUE,
100, kTRUE, 3);      </p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++) d3-&gt;SetBinContent(i +
1,dest[i]);</p>

<p class=MsoNormal>   d3-&gt;SetLineColor(kGreen);      </p>

<p class=MsoNormal>   d3-&gt;Draw(&quot;SAME&quot;);       </p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++) source[i]=h-&gt;GetBinContent(i
+ 1);</p>

<p class=MsoNormal>   s-&gt;SearchHighRes(source, dest, nbins, 8, 2, kTRUE,
1000, kTRUE, 3);      </p>

<p class=MsoNormal>   for (i = 0; i &lt; nbins; i++) d4-&gt;SetBinContent(i +
1,dest[i]);</p>

<p class=MsoNormal>   d4-&gt;SetLineColor(kMagenta);      </p>

<p class=MsoNormal>   d4-&gt;Draw(&quot;SAME&quot;);   </p>

<p class=MsoNormal>   printf(&quot;Found %d candidate peaks\n&quot;,nfound);  </p>

<p class=MsoNormal>}</p>

</div>

<!-- */
// --> End_Html
