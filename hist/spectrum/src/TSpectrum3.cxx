// @(#)root/spectrum:$Id$
// Author: Miroslav Morhac   25/09/2006

/////////////////////////////////////////////////////////////////////////////
//   THIS CLASS CONTAINS ADVANCED SPECTRA PROCESSING FUNCTIONS.            //
//                                                                         //
//   THREE-DIMENSIONAL BACKGROUND ESTIMATION FUNCTIONS                     //
//   THREE-DIMENSIONAL SMOOTHING FUNCTIONS                                 //
//   THREE-DIMENSIONAL DECONVOLUTION FUNCTIONS                             //
//   THREE-DIMENSIONAL PEAK SEARCH FUNCTIONS                               //
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
//   [3] M. Morhac et al.: Efficient algorithm of multidimensional         //
//   deconvolution and its application to nuclear data processing. Digital //
//   Signal Processing, Vol. 13, No. 1, (2003), 144-171.                   //
//                                                                         //
//   [4]  M.Morhac et al.: Identification of peaks in multidimensional     //
//   coincidence gamma-ray spectra. Nuclear Instruments and Methods in     //
//   Research Physics A  443(2000), 108-125.                               //
//                                                                         //
//   These NIM papers are also available as Postscript files from:         //
//
//
//   ftp://root.cern.ch/root/SpectrumDec.ps.gz
//   ftp://root.cern.ch/root/SpectrumSrc.ps.gz
//   ftp://root.cern.ch/root/SpectrumBck.ps.gz
//
//
/////////////////////////////////////////////////////////////////////////////


#include "TSpectrum3.h"
#include "TH1.h"
#include "TMath.h"
#define PEAK_WINDOW 1024

ClassImp(TSpectrum3)

//______________________________________________________________________________
TSpectrum3::TSpectrum3() :TNamed("Spectrum", "Miroslav Morhac peak finder")
{
   // Constructor.

   Int_t n = 100;
   fMaxPeaks   = n;
   fPosition   = new Double_t[n];
   fPositionX  = new Double_t[n];
   fPositionY  = new Double_t[n];
   fPositionZ  = new Double_t[n];
   fResolution = 1;
   fHistogram  = 0;
   fNPeaks     = 0;
}


//______________________________________________________________________________
TSpectrum3::TSpectrum3(Int_t maxpositions, Double_t resolution) :TNamed("Spectrum", "Miroslav Morhac peak finder")
{
//  maxpositions:  maximum number of peaks
//  resolution:    determines resolution of the neighboring peaks
//                 default value is 1 correspond to 3 sigma distance
//                 between peaks. Higher values allow higher resolution
//                 (smaller distance between peaks.
//                 May be set later through SetResolution.
   Int_t n = TMath::Max(maxpositions, 100);
   fMaxPeaks  = n;
   fPosition  = new Double_t[n];
   fPositionX = new Double_t[n];
   fPositionY = new Double_t[n];
   fPositionZ = new Double_t[n];
   fHistogram = 0;
   fNPeaks    = 0;
   SetResolution(resolution);
}


//______________________________________________________________________________
TSpectrum3::~TSpectrum3()
{
   // Destructor.

   delete [] fPosition;
   delete [] fPositionX;
   delete [] fPositionY;
   delete [] fPositionZ;
   delete    fHistogram;
}

//______________________________________________________________________________
const char *TSpectrum3::Background(const TH1 * h, Int_t number_of_iterations,
                                   Option_t * option)
{
/////////////////////////////////////////////////////////////////////////////
//   ONE-DIMENSIONAL BACKGROUND ESTIMATION FUNCTION                        //
//   This function calculates background spectrum from source in h.        //
//   The result is placed in the vector pointed by spectrum pointer.       //
//                                                                         //
//   Function parameters:                                                  //
//   spectrum:  pointer to the vector of source spectrum                   //
//   size:      length of spectrum and working space vectors               //
//   number_of_iterations, for details we refer to manual                  //
//                                                                         //
/////////////////////////////////////////////////////////////////////////////
   Error("Background","function not yet implemented: h=%s, iter=%d, option=%sn"
        , h->GetName(), number_of_iterations, option);
   return 0;
}

//______________________________________________________________________________
void TSpectrum3::Print(Option_t *) const
{
   // Print the array of positions

   printf("\nNumber of positions = %d\n",fNPeaks);
   for (Int_t i=0;i<fNPeaks;i++) {
      printf(" x[%d] = %g, y[%d] = %g, z[%d] = %g\n",i,fPositionX[i],i,fPositionY[i],i,fPositionZ[i]);
   }
}



//______________________________________________________________________________
Int_t TSpectrum3::Search(const TH1 * hin, Double_t sigma,
                             Option_t * option, Double_t threshold)
{
/////////////////////////////////////////////////////////////////////////////
//   ONE-DIMENSIONAL PEAK SEARCH FUNCTION                                  //
//   This function searches for peaks in source spectrum in hin            //
//   The number of found peaks and their positions are written into        //
//   the members fNpeaks and fPositionX.                                   //
//                                                                         //
//   Function parameters:                                                  //
//   hin:       pointer to the histogram of source spectrum                //
//   sigma:   sigma of searched peaks, for details we refer to manual      //
//            Note that sigma is in number of bins                         //
//   threshold: (default=0.05)  peaks with amplitude less than             //
//       threshold*highest_peak are discarded.                             //
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
   if (hin == 0)
      return 0;
   Int_t dimension = hin->GetDimension();
   if (dimension != 3) {
      Error("Search", "Must be a 3-d histogram");
      return 0;
   }

   Int_t sizex = hin->GetXaxis()->GetNbins();
   Int_t sizey = hin->GetYaxis()->GetNbins();
   Int_t sizez = hin->GetZaxis()->GetNbins();
   Int_t i, j, k, binx,biny,binz, npeaks;
   Double_t*** source = new Double_t**[sizex];
   Double_t*** dest   = new Double_t**[sizex];
   for (i = 0; i < sizex; i++) {
      source[i] = new Double_t*[sizey];
      dest[i]   = new Double_t*[sizey];
      for (j = 0; j < sizey; j++) {
         source[i][j] = new Double_t[sizez];
         dest[i][j]   = new Double_t[sizez];
         for (k = 0; k < sizez; k++)
            source[i][j][k] = (Double_t) hin->GetBinContent(i + 1, j + 1, k + 1);
      }
   }
   //the smoothing option is used for 1-d but not for 2-d histograms
   npeaks = SearchHighRes((const Double_t***)source, dest, sizex, sizey, sizez, sigma, 100*threshold, kTRUE, 3, kFALSE, 3);

   //The logic in the loop should be improved to use the fact
   //that fPositionX,Y give a precise position inside a bin.
   //The current algorithm takes the center of the bin only.
   for (i = 0; i < npeaks; i++) {
      binx = 1 + Int_t(fPositionX[i] + 0.5);
      biny = 1 + Int_t(fPositionY[i] + 0.5);
      binz = 1 + Int_t(fPositionZ[i] + 0.5);
      fPositionX[i] = hin->GetXaxis()->GetBinCenter(binx);
      fPositionY[i] = hin->GetYaxis()->GetBinCenter(biny);
      fPositionZ[i] = hin->GetZaxis()->GetBinCenter(binz);
   }
   for (i = 0; i < sizex; i++) {
      for (j = 0; j < sizey; j++){
         delete [] source[i][j];
         delete [] dest[i][j];
      }
      delete [] source[i];
      delete [] dest[i];
   }
   delete [] source;
   delete [] dest;

   if (strstr(option, "goff"))
      return npeaks;
   if (!npeaks) return 0;
   return npeaks;
}


//______________________________________________________________________________
void TSpectrum3::SetResolution(Double_t resolution)
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

//______________________________________________________________________________
const char *TSpectrum3::Background(Double_t***spectrum,
                       Int_t ssizex, Int_t ssizey, Int_t ssizez,
                       Int_t numberIterationsX,
                       Int_t numberIterationsY,
                       Int_t numberIterationsZ,
                       Int_t direction,
                       Int_t filterType)
{
///////////////////////////////////////////////////////////////////////////////
// THREE-DIMENSIONAL BACKGROUND ESTIMATION FUNCTIONS                         //
// This function calculates background spectrum from source spectrum.        //
// The result is placed to the array pointed by spectrum pointer.            //
//                                                                           //
// Function parameters:                                                      //
// spectrum-pointer to the array of source spectrum                          //
// ssizex-x length of spectrum                                               //
// ssizey-y length of spectrum                                               //
// ssizez-z length of spectrum                                               //
// numberIterationsX-maximal x width of clipping window                      //
// numberIterationsY-maximal y width of clipping window                      //
// numberIterationsZ-maximal z width of clipping window                      //
//                           for details we refer to manual                  //
// direction- direction of change of clipping window                         //
//               - possible values=kBackIncreasingWindow                     //
//                                 kBackDecreasingWindow                     //
// filterType-determines the algorithm of the filtering                      //
//                  -possible values=kBackSuccessiveFiltering                //
//                                   kBackOneStepFiltering                   //
//                                                                           //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
//Begin_Html <!--
/* -->
<div class=Section1>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:14.0pt'>Background
estimation</span></b></p>

<p class=MsoNormal style='text-align:justify'><i>&nbsp;</i></p>

<p class=MsoNormal style='text-align:justify'><i>Goal: Separation of useful
information (peaks) from useless information (background)</i> </p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>method is based on Sensitive Nonlinear Iterative Peak (SNIP) clipping
algorithm [1]</p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>there exist two algorithms for the estimation of new value in the
channel “<sub><img width=43 height=24 src="gif/spectrum3_background_image001.gif"></sub>”</p>

<p class=MsoNormal style='margin-left:18.0pt;text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><i>Algorithm based on Successive
Comparisons</i></p>

<p class=MsoNormal style='text-align:justify'>It is an extension of
one-dimensional SNIP algorithm to another dimension. For details we refer to
[2].</p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><i>Algorithm based on One Step
Filtering</i></p>

<p class=MsoNormal style='text-align:justify'>The algorithm is analogous to
that for 2-dimensional data. For details we refer to TSpectrum2. New value in
the estimated channel is calculated as</p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><sub><img width=103 height=26
src="gif/spectrum3_background_image002.gif"></sub></p>

<p class=MsoNormal style='text-align:justify'><sub><img width=621 height=408
src="gif/spectrum3_background_image003.gif"></sub><sub><img width=148 height=27
src="gif/spectrum3_background_image004.gif"></sub></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'>where p = 1, 2, …,
number_of_iterations. </p>

<p class=MsoNormal><i>&nbsp;</i></p>

<p class=MsoNormal><i>Function:</i></p>

<p class=MsoNormal style='text-align:justify'><b>const <a
href="http://root.cern.ch/root/html/ListOfTypes.html#char" target="_parent">char</a>*
</b><a href="http://root.cern.ch/root/html/TSpectrum.html#TSpectrum:Fit1Awmi"><b>TSpectrum3::Background</b></a><b>
(<a href="http://root.cern.ch/root/html/ListOfTypes.html#double" target="_parent">double</a>
***fSpectrum, <a href="http://root.cern.ch/root/html/ListOfTypes.html#int"
target="_parent">int</a> fSizex, <a
href="http://root.cern.ch/root/html/ListOfTypes.html#int" target="_parent">int</a>
fSizey, int fSizez, <a href="http://root.cern.ch/root/html/ListOfTypes.html#int"
target="_parent">int</a> fNumberIterationsX, <a
href="http://root.cern.ch/root/html/ListOfTypes.html#int" target="_parent">int</a>
fNumberIterationsY, <a href="http://root.cern.ch/root/html/ListOfTypes.html#int"
target="_parent">int</a> fNumberIterationsZ,  <a
href="http://root.cern.ch/root/html/ListOfTypes.html#int" target="_parent">int</a>
fDirection, <a href="http://root.cern.ch/root/html/ListOfTypes.html#int"
target="_parent">int</a> fFilterType)  </b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'>This function calculates
background spectrum from the source spectrum.  The result is placed in the matrix
pointed by fSpectrum pointer.  One can also switch the direction of the change
of the clipping window and to select one of the two above given algorithms. On
successful completion it returns 0. On error it returns pointer to the string
describing error.</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><i><span style='color:red'>Parameters:</span></i></p>

<p class=MsoNormal>        <b>fSpectrum</b>-pointer to the matrix of source
spectrum                  </p>

<p class=MsoNormal>        <b>fSizex, fSizey, fSizez </b>-lengths of the
spectrum matrix                                 </p>

<p class=MsoNormal style='text-align:justify'>        <b>fNumberIterationsX,
fNumberIterationsY, fNumberIterationsZ </b>maximal</p>

<p class=MsoNormal style='text-align:justify'>        widths of clipping window,                                
</p>

<p class=MsoNormal>        <b>fDirection</b>- direction of change of clipping
window                  </p>

<p class=MsoNormal>               - possible
values=kBackIncreasingWindow                      </p>

<p class=MsoNormal>                                           
kBackDecreasingWindow                      </p>

<p class=MsoNormal>        <b>fFilterType</b>-type of the clipping algorithm,          
                   </p>

<p class=MsoNormal>                  -possible values=kBack SuccessiveFiltering</p>

<p class=MsoNormal>                                             
kBackOneStepFiltering                              </p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><b><i>References:</i></b></p>

<p class=MsoNormal style='text-align:justify'>[1]  C. G Ryan et al.: SNIP, a
statistics-sensitive background treatment for the quantitative analysis of PIXE
spectra in geoscience applications. NIM, B34 (1988), 396-402.</p>

<p class=MsoNormal style='text-align:justify'>[2] <span lang=SK> M.
Morhá&#269;, J. Kliman, V. Matoušek, M. Veselský, I. Turzo</span>.: Background
elimination methods for multidimensional gamma-ray spectra. NIM, A401 (1997)
113-132.</p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal><i>Example 1– script Back3.c :</i></p>

<p class=MsoNormal><i><span style='font-size:18.0pt'><img border=0 width=601
height=368 src="gif/spectrum3_background_image005.jpg"></span></i></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><b>Fig. 1 Original three-dimensional
gamma-gamma-gamma-ray spectrum</b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'><img
border=0 width=601 height=368 src="gif/spectrum3_background_image006.jpg"></span></b></p>

<p class=MsoNormal style='text-align:justify'><b>Fig. 2 Background estimated
from data from Fig. 1 using decreasing clipping window with widths 5, 5, 5 and
algorithm based on successive comparisons. The estimate includes not only
continuously changing background but also one- and two-dimensional ridges.</b></p>

<p class=MsoNormal><b><span style='font-size:14.0pt;color:green'>&nbsp;</span></b></p>

<p class=MsoNormal><b><span style='font-size:14.0pt;color:green'><img border=0
width=601 height=368 src="gif/spectrum3_background_image007.jpg"></span></b></p>

<p class=MsoNormal style='text-align:justify'><b>Fig. 3 Resulting peaks after
subtraction of the estimated background (Fig. 2) from original three-dimensional
gamma-gamma-gamma-ray spectrum (Fig. 1).</b></p>

<p class=MsoNormal><b><span style='color:green'>&nbsp;</span></b></p>

<p class=MsoNormal><b><span style='color:green'>&nbsp;</span></b></p>

<p class=MsoNormal><b><span style='color:green'>Script:</span></b></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// Example to illustrate the
background estimator (class TSpectrum3).</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// To execute this example,
do</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// root &gt; .x Back3.C</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>&nbsp;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>void Back3() {</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Int_t i, j, k;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   </span><span lang=FR
style='font-size:10.0pt'>Int_t nbinsx = 64;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t nbinsy = 64;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t nbinsz =
64;   </span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t xmin  = 0;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t xmax  =
nbinsx;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t ymin  = 0;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t ymax  =
nbinsy;   </span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t zmin  = 0;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   </span><span
style='font-size:10.0pt'>Int_t zmax  = nbinsz;      </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t*** source = new
Double_t**[nbinsx];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t*** dest = new Double_t
**[nbinsx];      </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for(i=0;i&lt;nbinsx;i++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>      source[i]=new Double_t*
[nbinsy];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     
for(j=0;j&lt;nbinsy;j++)</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>         source[i][j]=new
Double_t[nbinsz];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }           </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for(i=0;i&lt;nbinsx;i++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>      dest[i]=new Double_t*
[nbinsy];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     
for(j=0;j&lt;nbinsy;j++)</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>         dest[i][j]=new Double_t
[nbinsz];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }              </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TH3F *back = new
TH3F(&quot;back&quot;,&quot;Background
estimation&quot;,nbinsx,xmin,xmax,nbinsy,ymin,ymax,nbinsz,zmin,zmax);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TFile *f = new
TFile(&quot;TSpectrum3.root&quot;);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   back=(TH3F*)
f-&gt;Get(&quot;back;1&quot;);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TCanvas *Background = new
TCanvas(&quot;Background&quot;,&quot;Estimation of background with decreasing
window&quot;,10,10,1000,700);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TSpectrum3 *s = new
TSpectrum3();</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for (i = 0; i &lt; nbinsx;
i++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     for (j = 0; j &lt;
nbinsy; j++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                  for (k = 0;
k &lt; nbinsz; k++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                       source[i][j][k]
= back-&gt;GetBinContent(i + 1,j + 1,k + 1);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                       dest[i][j][k]
= back-&gt;GetBinContent(i + 1,j + 1,k + 1);                     </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                    } </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                 }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
s-&gt;Background(dest,nbinsx,nbinsy,nbinsz,5,5,5,s-&gt;kBackDecreasingWindow,s-&gt;kBackSuccessiveFiltering);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for (i = 0; i &lt; nbinsx;
i++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     for (j = 0; j &lt;
nbinsy; j++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>        for (k = 0; k &lt;
nbinsz; k++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>          
back-&gt;SetBinContent(i + 1,j + 1,k + 1, dest[i][j][k]);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>        }    </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>&nbsp;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   FILE *out;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   char PATH[80];   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
strcpy(PATH,&quot;spectra3\\back_output_5ds.spe&quot;);   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   out=fopen(PATH,&quot;wb&quot;);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for(i=0;i&lt;nbinsx;i++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     
for(j=0;j&lt;nbinsy;j++){                   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>         fwrite(dest[i][j],
sizeof(dest[0][0][0]),nbinsz,out);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>      }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   fclose(out);   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for (i = 0; i &lt; nbinsx;
i++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     for (j = 0; j &lt;
nbinsy; j++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>        for (k = 0; k &lt;
nbinsz; k++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>           source[i][j][k] =
source[i][j][k] - dest[i][j][k];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>        }    </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for (i = 0; i &lt; nbinsx;
i++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     for (j = 0; j &lt;
nbinsy; j++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>        for (k = 0; k &lt;
nbinsz; k++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>          
back-&gt;SetBinContent(i + 1,j + 1,k + 1, source[i][j][k]);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>        }    </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
strcpy(PATH,&quot;spectra3\\back_peaks_5ds.spe&quot;);   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
out=fopen(PATH,&quot;wb&quot;);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for(i=0;i&lt;nbinsx;i++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     
for(j=0;j&lt;nbinsy;j++){                   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>         fwrite(source[i][j],
sizeof(source[0][0][0]),nbinsz,out);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>      }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   fclose(out);      </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
back-&gt;Draw(&quot;&quot;);  </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>}</span></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

</div>

<!-- */
// --> End_Html
   Int_t i, j, x, y, z, sampling, q1, q2, q3;
   Double_t a, b, c, d, p1, p2, p3, p4, p5, p6, p7, p8, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, r1, r2, r3, r4, r5, r6;
   if (ssizex <= 0 || ssizey <= 0 || ssizez <= 0)
      return "Wrong parameters";
   if (numberIterationsX < 1 || numberIterationsY < 1 || numberIterationsZ < 1)
      return "Width of Clipping Window Must Be Positive";
   if (ssizex < 2 * numberIterationsX + 1 || ssizey < 2 * numberIterationsY + 1 || ssizey < 2 * numberIterationsZ + 1)
      return ("Too Large Clipping Window");
   Double_t*** working_space=new Double_t**[ssizex];
   for(i=0;i<ssizex;i++){
      working_space[i] =new Double_t*[ssizey];
      for(j=0;j<ssizey;j++)
         working_space[i][j]=new Double_t[ssizez];
   }
   sampling =(Int_t) TMath::Max(numberIterationsX, numberIterationsY);
   sampling =(Int_t) TMath::Max(sampling, numberIterationsZ);
   if (direction == kBackIncreasingWindow) {
      if (filterType == kBackSuccessiveFiltering) {
         for (i = 1; i <= sampling; i++) {
            q1 = (Int_t) TMath::Min(i, numberIterationsX), q2 =(Int_t) TMath::Min(i, numberIterationsY), q3 =(Int_t) TMath::Min(i, numberIterationsZ);
            for (z = q3; z < ssizez - q3; z++) {
               for (y = q2; y < ssizey - q2; y++) {
                  for (x = q1; x < ssizex - q1; x++) {
                     a = spectrum[x][y][z];
                     p1 = spectrum[x + q1][y + q2][z - q3];
                     p2 = spectrum[x - q1][y + q2][z - q3];
                     p3 = spectrum[x + q1][y - q2][z - q3];
                     p4 = spectrum[x - q1][y - q2][z - q3];
                     p5 = spectrum[x + q1][y + q2][z + q3];
                     p6 = spectrum[x - q1][y + q2][z + q3];
                     p7 = spectrum[x + q1][y - q2][z + q3];
                     p8 = spectrum[x - q1][y - q2][z + q3];
                     s1 = spectrum[x + q1][y     ][z - q3];
                     s2 = spectrum[x     ][y + q2][z - q3];
                     s3 = spectrum[x - q1][y     ][z - q3];
                     s4 = spectrum[x     ][y - q2][z - q3];
                     s5 = spectrum[x + q1][y     ][z + q3];
                     s6 = spectrum[x     ][y + q2][z + q3];
                     s7 = spectrum[x - q1][y     ][z + q3];
                     s8 = spectrum[x     ][y - q2][z + q3];
                     s9 = spectrum[x - q1][y + q2][z     ];
                     s10 = spectrum[x - q1][y - q2][z     ];
                     s11 = spectrum[x + q1][y + q2][z     ];
                     s12 = spectrum[x + q1][y - q2][z     ];
                     r1 = spectrum[x     ][y     ][z - q3];
                     r2 = spectrum[x     ][y     ][z + q3];
                     r3 = spectrum[x - q1][y     ][z     ];
                     r4 = spectrum[x + q1][y     ][z     ];
                     r5 = spectrum[x     ][y + q2][z     ];
                     r6 = spectrum[x     ][y - q2][z     ];
                     b = (p1 + p3) / 2.0;
                     if(b > s1)
                        s1 = b;
                     b = (p1 + p2) / 2.0;
                     if(b > s2)
                        s2 = b;
                     b = (p2 + p4) / 2.0;
                     if(b > s3)
                        s3 = b;
                     b = (p3 + p4) / 2.0;
                     if(b > s4)
                        s4 = b;
                     b = (p5 + p7) / 2.0;
                     if(b > s5)
                        s5 = b;
                     b = (p5 + p6) / 2.0;
                     if(b > s6)
                        s6 = b;
                     b = (p6 + p8) / 2.0;
                     if(b > s7)
                        s7 = b;
                     b = (p7 + p8) / 2.0;
                     if(b > s8)
                        s8 = b;
                     b = (p2 + p6) / 2.0;
                     if(b > s9)
                        s9 = b;
                     b = (p4 + p8) / 2.0;
                     if(b > s10)
                        s10 = b;
                     b = (p1 + p5) / 2.0;
                     if(b > s11)
                        s11 = b;
                     b = (p3 + p7) / 2.0;
                     if(b > s12)
                        s12 = b;
                     s1 = s1 - (p1 + p3) / 2.0;
                     s2 = s2 - (p1 + p2) / 2.0;
                     s3 = s3 - (p2 + p4) / 2.0;
                     s4 = s4 - (p3 + p4) / 2.0;
                     s5 = s5 - (p5 + p7) / 2.0;
                     s6 = s6 - (p5 + p6) / 2.0;
                     s7 = s7 - (p6 + p8) / 2.0;
                     s8 = s8 - (p7 + p8) / 2.0;
                     s9 = s9 - (p2 + p6) / 2.0;
                     s10 = s10 - (p4 + p8) / 2.0;
                     s11 = s11 - (p1 + p5) / 2.0;
                     s12 = s12 - (p3 + p7) / 2.0;
                     b = (s1 + s3) / 2.0 + (s2 + s4) / 2.0 + (p1 + p2 + p3 + p4) / 4.0;
                     if(b > r1)
                        r1 = b;
                     b = (s5 + s7) / 2.0 + (s6 + s8) / 2.0 + (p5 + p6 + p7 + p8) / 4.0;
                     if(b > r2)
                        r2 = b;
                     b = (s3 + s7) / 2.0 + (s9 + s10) / 2.0 + (p2 + p4 + p6 + p8) / 4.0;
                     if(b > r3)
                        r3 = b;
                     b = (s1 + s5) / 2.0 + (s11 + s12) / 2.0 + (p1 + p3 + p5 + p7) / 4.0;
                     if(b > r4)
                        r4 = b;
                     b = (s9 + s11) / 2.0 + (s2 + s6) / 2.0 + (p1 + p2 + p5 + p6) / 4.0;
                     if(b > r5)
                        r5 = b;
                     b = (s4 + s8) / 2.0 + (s10 + s12) / 2.0 + (p3 + p4 + p7 + p8) / 4.0;
                     if(b > r6)
                        r6 = b;
                     r1 = r1 - ((s1 + s3) / 2.0 + (s2 + s4) / 2.0 + (p1 + p2 + p3 + p4) / 4.0);
                     r2 = r2 - ((s5 + s7) / 2.0 + (s6 + s8) / 2.0 + (p5 + p6 + p7 + p8) / 4.0);
                     r3 = r3 - ((s3 + s7) / 2.0 + (s9 + s10) / 2.0 + (p2 + p4 + p6 + p8) / 4.0);
                     r4 = r4 - ((s1 + s5) / 2.0 + (s11 + s12) / 2.0 + (p1 + p3 + p5 + p7) / 4.0);
                     r5 = r5 - ((s9 + s11) / 2.0 + (s2 + s6) / 2.0 + (p1 + p2 + p5 + p6) / 4.0);
                     r6 = r6 - ((s4 + s8) / 2.0 + (s10 + s12) / 2.0 + (p3 + p4 + p7 + p8) / 4.0);
                     b = (r1 + r2) / 2.0 + (r3 + r4) / 2.0 + (r5 + r6) / 2.0 + (s1 + s3 + s5 + s7) / 4.0 + (s2 + s4 + s6 + s8) / 4.0 + (s9 + s10 + s11 + s12) / 4.0 + (p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8) / 8.0;
                     if(b < a)
                        a = b;
                     working_space[x][y][z] = a;
                  }
               }
            }
            for (z = q3; z < ssizez - q3; z++) {
               for (y = q2; y < ssizey - q2; y++) {
                  for (x = q1; x < ssizex - q1; x++) {
                     spectrum[x][y][z] = working_space[x][y][z];
                  }
               }
            }
         }
      }

      else if (filterType == kBackOneStepFiltering) {
         for (i = 1; i <= sampling; i++) {
            q1 = (Int_t) TMath::Min(i, numberIterationsX), q2 =(Int_t) TMath::Min(i, numberIterationsY), q3 =(Int_t) TMath::Min(i, numberIterationsZ);
            for (z = q3; z < ssizez - q3; z++) {
               for (y = q2; y < ssizey - q2; y++) {
                  for (x = q1; x < ssizex - q1; x++) {
                     a = spectrum[x][y][z];
                     p1 = spectrum[x + q1][y + q2][z - q3];
                     p2 = spectrum[x - q1][y + q2][z - q3];
                     p3 = spectrum[x + q1][y - q2][z - q3];
                     p4 = spectrum[x - q1][y - q2][z - q3];
                     p5 = spectrum[x + q1][y + q2][z + q3];
                     p6 = spectrum[x - q1][y + q2][z + q3];
                     p7 = spectrum[x + q1][y - q2][z + q3];
                     p8 = spectrum[x - q1][y - q2][z + q3];
                     s1 = spectrum[x + q1][y     ][z - q3];
                     s2 = spectrum[x     ][y + q2][z - q3];
                     s3 = spectrum[x - q1][y     ][z - q3];
                     s4 = spectrum[x     ][y - q2][z - q3];
                     s5 = spectrum[x + q1][y     ][z + q3];
                     s6 = spectrum[x     ][y + q2][z + q3];
                     s7 = spectrum[x - q1][y     ][z + q3];
                     s8 = spectrum[x     ][y - q2][z + q3];
                     s9 = spectrum[x - q1][y + q2][z     ];
                     s10 = spectrum[x - q1][y - q2][z     ];
                     s11 = spectrum[x + q1][y + q2][z     ];
                     s12 = spectrum[x + q1][y - q2][z     ];
                     r1 = spectrum[x     ][y     ][z - q3];
                     r2 = spectrum[x     ][y     ][z + q3];
                     r3 = spectrum[x - q1][y     ][z     ];
                     r4 = spectrum[x + q1][y     ][z     ];
                     r5 = spectrum[x     ][y + q2][z     ];
                     r6 = spectrum[x     ][y - q2][z     ];
                     b=(p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8) / 8 - (s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10 + s11 + s12) / 4 + (r1 + r2 + r3 + r4 + r5 + r6) / 2;
                     c = -(s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10 + s11 + s12) / 4 + (r1 + r2 + r3 + r4 + r5 + r6) / 2;
                     d = -(p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8) / 8 + (s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10 + s11 + s12) / 12;
                     if(b < a && b >= 0 && c >=0 && d >= 0)
                        a = b;
                     working_space[x][y][z] = a;
                  }
               }
            }
            for (z = q3; z < ssizez - q3; z++) {
               for (y = q2; y < ssizey - q2; y++) {
                  for (x = q1; x < ssizex - q1; x++) {
                     spectrum[x][y][z] = working_space[x][y][z];
                  }
               }
            }
         }
      }
   }

   else if (direction == kBackDecreasingWindow) {
      if (filterType == kBackSuccessiveFiltering) {
         for (i = sampling; i >= 1; i--) {
            q1 = (Int_t) TMath::Min(i, numberIterationsX), q2 =(Int_t) TMath::Min(i, numberIterationsY), q3 =(Int_t) TMath::Min(i, numberIterationsZ);
            for (z = q3; z < ssizez - q3; z++) {
               for (y = q2; y < ssizey - q2; y++) {
                  for (x = q1; x < ssizex - q1; x++) {
                     a = spectrum[x][y][z];
                     p1 = spectrum[x + q1][y + q2][z - q3];
                     p2 = spectrum[x - q1][y + q2][z - q3];
                     p3 = spectrum[x + q1][y - q2][z - q3];
                     p4 = spectrum[x - q1][y - q2][z - q3];
                     p5 = spectrum[x + q1][y + q2][z + q3];
                     p6 = spectrum[x - q1][y + q2][z + q3];
                     p7 = spectrum[x + q1][y - q2][z + q3];
                     p8 = spectrum[x - q1][y - q2][z + q3];
                     s1 = spectrum[x + q1][y     ][z - q3];
                     s2 = spectrum[x     ][y + q2][z - q3];
                     s3 = spectrum[x - q1][y     ][z - q3];
                     s4 = spectrum[x     ][y - q2][z - q3];
                     s5 = spectrum[x + q1][y     ][z + q3];
                     s6 = spectrum[x     ][y + q2][z + q3];
                     s7 = spectrum[x - q1][y     ][z + q3];
                     s8 = spectrum[x     ][y - q2][z + q3];
                     s9 = spectrum[x - q1][y + q2][z     ];
                     s10 = spectrum[x - q1][y - q2][z     ];
                     s11 = spectrum[x + q1][y + q2][z     ];
                     s12 = spectrum[x + q1][y - q2][z     ];
                     r1 = spectrum[x     ][y     ][z - q3];
                     r2 = spectrum[x     ][y     ][z + q3];
                     r3 = spectrum[x - q1][y     ][z     ];
                     r4 = spectrum[x + q1][y     ][z     ];
                     r5 = spectrum[x     ][y + q2][z     ];
                     r6 = spectrum[x     ][y - q2][z     ];
                     b = (p1 + p3) / 2.0;
                     if(b > s1)
                        s1 = b;
                     b = (p1 + p2) / 2.0;
                     if(b > s2)
                        s2 = b;
                     b = (p2 + p4) / 2.0;
                     if(b > s3)
                        s3 = b;
                     b = (p3 + p4) / 2.0;
                     if(b > s4)
                        s4 = b;
                     b = (p5 + p7) / 2.0;
                     if(b > s5)
                        s5 = b;
                     b = (p5 + p6) / 2.0;
                     if(b > s6)
                        s6 = b;
                     b = (p6 + p8) / 2.0;
                     if(b > s7)
                        s7 = b;
                     b = (p7 + p8) / 2.0;
                     if(b > s8)
                        s8 = b;
                     b = (p2 + p6) / 2.0;
                     if(b > s9)
                        s9 = b;
                     b = (p4 + p8) / 2.0;
                     if(b > s10)
                        s10 = b;
                     b = (p1 + p5) / 2.0;
                     if(b > s11)
                        s11 = b;
                     b = (p3 + p7) / 2.0;
                     if(b > s12)
                        s12 = b;
                     s1 = s1 - (p1 + p3) / 2.0;
                     s2 = s2 - (p1 + p2) / 2.0;
                     s3 = s3 - (p2 + p4) / 2.0;
                     s4 = s4 - (p3 + p4) / 2.0;
                     s5 = s5 - (p5 + p7) / 2.0;
                     s6 = s6 - (p5 + p6) / 2.0;
                     s7 = s7 - (p6 + p8) / 2.0;
                     s8 = s8 - (p7 + p8) / 2.0;
                     s9 = s9 - (p2 + p6) / 2.0;
                     s10 = s10 - (p4 + p8) / 2.0;
                     s11 = s11 - (p1 + p5) / 2.0;
                     s12 = s12 - (p3 + p7) / 2.0;
                     b = (s1 + s3) / 2.0 + (s2 + s4) / 2.0 + (p1 + p2 + p3 + p4) / 4.0;
                     if(b > r1)
                        r1 = b;
                     b = (s5 + s7) / 2.0 + (s6 + s8) / 2.0 + (p5 + p6 + p7 + p8) / 4.0;
                     if(b > r2)
                        r2 = b;
                     b = (s3 + s7) / 2.0 + (s9 + s10) / 2.0 + (p2 + p4 + p6 + p8) / 4.0;
                     if(b > r3)
                        r3 = b;
                     b = (s1 + s5) / 2.0 + (s11 + s12) / 2.0 + (p1 + p3 + p5 + p7) / 4.0;
                     if(b > r4)
                        r4 = b;
                     b = (s9 + s11) / 2.0 + (s2 + s6) / 2.0 + (p1 + p2 + p5 + p6) / 4.0;
                     if(b > r5)
                        r5 = b;
                     b = (s4 + s8) / 2.0 + (s10 + s12) / 2.0 + (p3 + p4 + p7 + p8) / 4.0;
                     if(b > r6)
                        r6 = b;
                     r1 = r1 - ((s1 + s3) / 2.0 + (s2 + s4) / 2.0 + (p1 + p2 + p3 + p4) / 4.0);
                     r2 = r2 - ((s5 + s7) / 2.0 + (s6 + s8) / 2.0 + (p5 + p6 + p7 + p8) / 4.0);
                     r3 = r3 - ((s3 + s7) / 2.0 + (s9 + s10) / 2.0 + (p2 + p4 + p6 + p8) / 4.0);
                     r4 = r4 - ((s1 + s5) / 2.0 + (s11 + s12) / 2.0 + (p1 + p3 + p5 + p7) / 4.0);
                     r5 = r5 - ((s9 + s11) / 2.0 + (s2 + s6) / 2.0 + (p1 + p2 + p5 + p6) / 4.0);
                     r6 = r6 - ((s4 + s8) / 2.0 + (s10 + s12) / 2.0 + (p3 + p4 + p7 + p8) / 4.0);
                     b = (r1 + r2) / 2.0 + (r3 + r4) / 2.0 + (r5 + r6) / 2.0 + (s1 + s3 + s5 + s7) / 4.0 + (s2 + s4 + s6 + s8) / 4.0 + (s9 + s10 + s11 + s12) / 4.0 + (p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8) / 8.0;
                     if(b < a)
                        a = b;
                     working_space[x][y][z] = a;
                  }
               }
            }
            for (z = q3; z < ssizez - q3; z++) {
               for (y = q2; y < ssizey - q2; y++) {
                  for (x = q1; x < ssizex - q1; x++) {
                     spectrum[x][y][z] = working_space[x][y][z];
                  }
               }
            }
         }
      }

      else if (filterType == kBackOneStepFiltering) {
         for (i = sampling; i >= 1; i--) {
            q1 = (Int_t) TMath::Min(i, numberIterationsX), q2 =(Int_t) TMath::Min(i, numberIterationsY), q3 =(Int_t) TMath::Min(i, numberIterationsZ);
            for (z = q3; z < ssizez - q3; z++) {
               for (y = q2; y < ssizey - q2; y++) {
                  for (x = q1; x < ssizex - q1; x++) {
                     a = spectrum[x][y][z];
                     p1 = spectrum[x + q1][y + q2][z - q3];
                     p2 = spectrum[x - q1][y + q2][z - q3];
                     p3 = spectrum[x + q1][y - q2][z - q3];
                     p4 = spectrum[x - q1][y - q2][z - q3];
                     p5 = spectrum[x + q1][y + q2][z + q3];
                     p6 = spectrum[x - q1][y + q2][z + q3];
                     p7 = spectrum[x + q1][y - q2][z + q3];
                     p8 = spectrum[x - q1][y - q2][z + q3];
                     s1 = spectrum[x + q1][y     ][z - q3];
                     s2 = spectrum[x     ][y + q2][z - q3];
                     s3 = spectrum[x - q1][y     ][z - q3];
                     s4 = spectrum[x     ][y - q2][z - q3];
                     s5 = spectrum[x + q1][y     ][z + q3];
                     s6 = spectrum[x     ][y + q2][z + q3];
                     s7 = spectrum[x - q1][y     ][z + q3];
                     s8 = spectrum[x     ][y - q2][z + q3];
                     s9 = spectrum[x - q1][y + q2][z     ];
                     s10 = spectrum[x - q1][y - q2][z     ];
                     s11 = spectrum[x + q1][y + q2][z     ];
                     s12 = spectrum[x + q1][y - q2][z     ];
                     r1 = spectrum[x     ][y     ][z - q3];
                     r2 = spectrum[x     ][y     ][z + q3];
                     r3 = spectrum[x - q1][y     ][z     ];
                     r4 = spectrum[x + q1][y     ][z     ];
                     r5 = spectrum[x     ][y + q2][z     ];
                     r6 = spectrum[x     ][y - q2][z     ];
                     b = (p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8) / 8 - (s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10 + s11 + s12) / 4 + (r1 + r2 + r3 + r4 + r5 + r6) / 2;
                     c = -(s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10 + s11 + s12) / 4+(r1 + r2 + r3 + r4 + r5 + r6) / 2;
                     d = -(p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8)/8 + (s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10 + s11 + s12) / 12;
                     if(b < a && b >= 0 && c >=0 && d >= 0)
                        a = b;
                     working_space[x][y][z] = a;
                  }
               }
            }
            for (z = q3; z < ssizez - q3; z++) {
               for (y = q2; y < ssizey - q2; y++) {
                  for (x = q1; x < ssizex - q1; x++) {
                     spectrum[x][y][z] = working_space[x][y][z];
                  }
               }
            }
         }
      }
   }
   for(i = 0;i < ssizex; i++){
      for(j = 0;j < ssizey; j++)
         delete[] working_space[i][j];
      delete[] working_space[i];
   }
   delete[] working_space;
   return 0;
}

//_____________________________________________________________________________
const char* TSpectrum3::SmoothMarkov(Double_t***source, Int_t ssizex, Int_t ssizey, Int_t ssizez, Int_t averWindow)
{
/////////////////////////////////////////////////////////////////////////////
// THREE-DIMENSIONAL MARKOV SPECTRUM SMOOTHING FUNCTION                    //
//                                                                         //
// This function calculates smoothed spectrum from source spectrum         //
//      based on Markov chain method.                                      //
// The result is placed in the array pointed by spectrum pointer.          //
//                                                                         //
// Function parameters:                                                    //
// source-pointer to the array of source spectrum                          //
// working_space-pointer to the working array                              //
// ssizex-x length of spectrum and working space arrays                    //
// ssizey-y length of spectrum and working space arrays                    //
// ssizey-z length of spectrum and working space arrays                    //
// averWindow-width of averaging smoothing window                          //
//                                                                         //
/////////////////////////////////////////////////////////////////////////////
//Begin_Html <!--
/* -->

<div class=Section2>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:14.0pt'>Smoothing</span></b></p>

<p class=MsoNormal style='text-align:justify'><i>&nbsp;</i></p>

<p class=MsoNormal><i>Goal: Suppression of statistical fluctuations</i></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>the algorithm is based on discrete Markov chain, which has very simple
invariant distribution</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>            <sub><img width=404 height=42
src="Smoothing_files/image001.gif"></sub><span style='font-family:Arial'>     </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-family:Arial'>       
</span><sub><img width=22 height=28 src="Smoothing_files/image002.gif"></sub><span
style='font-family:Arial'>  </span>being defined from the normalization
condition <sub><img width=50 height=37 src="Smoothing_files/image003.gif"></sub></p>

<p class=MsoNormal>         n is the length of the smoothed spectrum and </p>

<p class=MsoNormal>

<table cellpadding=0 cellspacing=0 align=left>
 <tr>
  <td width=65 height=9></td>
 </tr>
 <tr>
  <td></td>
  <td><img width=205 height=48 src="Smoothing_files/image004.gif"></td>
 </tr>
</table>

 &nbsp;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<br clear=ALL>

<p class=MsoNormal style='margin-left:34.2pt;text-align:justify'>is the
probability of the change of the peak position from channel i to the channel
i+1.  <sub><img width=24 height=31 src="Smoothing_files/image005.gif"></sub>is
the normalization constant so that <sub><img width=99 height=25
src="Smoothing_files/image006.gif"></sub> and m is a width of smoothing window.
We have extended this algorithm to three dimensions. </p>

<p class=MsoNormal><i>&nbsp;</i></p>

<p class=MsoNormal><i>Function:</i></p>

<p class=MsoNormal><b>const <a
href="http://root.cern.ch/root/html/ListOfTypes.html#char" target="_parent">char</a>*
</b><a href="http://root.cern.ch/root/html/TSpectrum.html#TSpectrum:Fit1Awmi"><b>TSpectrum3::SmoothMarkov</b></a><b>(<a
href="http://root.cern.ch/root/html/ListOfTypes.html#double" target="_parent">double</a>
***fSpectrum, <a href="http://root.cern.ch/root/html/ListOfTypes.html#int"
target="_parent">int</a> fSizex, <a
href="http://root.cern.ch/root/html/ListOfTypes.html#int" target="_parent">int</a>
fSizey, <a href="http://root.cern.ch/root/html/ListOfTypes.html#int"
target="_parent">int</a> fSizey,  <a
href="http://root.cern.ch/root/html/ListOfTypes.html#int" target="_parent">int</a>
fAverWindow)  </b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'>This function calculates smoothed
spectrum from the source spectrum based on Markov chain method. The result is
placed in the field pointed by source pointer. On successful completion it
returns 0. On error it returns pointer to the string describing error.</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><i><span style='color:red'>Parameters:</span></i></p>

<p class=MsoNormal>        <b>fSpectrum</b>-pointer to the matrix of source
spectrum                  </p>

<p class=MsoNormal>        <b>fSizex, fSizey, fSizez</b> -lengths of the
spectrum matrix                                 </p>

<p class=MsoNormal>        <b>fAverWindow</b>-width of averaging smoothing
window </p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><b><i>Reference:</i></b></p>

<p class=MsoNormal style='text-align:justify'>[1] Z.K. Silagadze, A new
algorithm for automatic photopeak searches. NIM A 376 (1996), 451<b>.</b>  </p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal><i>Example 1 – script SmootMarkov3.c :</i></p>

<p class=MsoNormal><i><img border=0 width=601 height=368
src="Smoothing_files/image007.jpg"></i><b>Fig. 1 Original noisy spectrum.    </b></p>

<p class=MsoNormal><b><img border=0 width=601 height=368
src="Smoothing_files/image008.jpg"></b></p>

<p class=MsoNormal><b>Fig. 2 Smoothed spectrum with averaging window m=3.</b></p>

<p class=MsoNormal><b><span style='color:#339966'>&nbsp;</span></b></p>

<p class=MsoNormal><b><span style='color:#339966'>Script:</span></b></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// Example to illustrate the
Markov smoothing (class TSpectrum3).</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// To execute this example,
do</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// root &gt; .x
SmoothMarkov3.C</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>&nbsp;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>void SmoothMarkov3()
{</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t i, j, k;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t nbinsx = 64;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t nbinsy = 64;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t nbinsz =
64;   </span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t xmin  = 0;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t xmax  =
nbinsx;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t ymin  = 0;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t ymax  =
nbinsy;   </span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t zmin  = 0;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   </span><span
style='font-size:10.0pt'>Int_t zmax  = nbinsz;      </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t*** source = new
Double_t**[nbinsx];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for(i=0;i&lt;nbinsx;i++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>      source[i]=new Double_t*
[nbinsy];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     
for(j=0;j&lt;nbinsy;j++)</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>         source[i][j]=new
Double_t[nbinsz];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }           </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TH3F *sm = new
TH3F(&quot;Smoothing&quot;,&quot;Markov
smoothing&quot;,nbinsx,xmin,xmax,nbinsy,ymin,ymax,nbinsz,zmin,zmax);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TFile *f = new
TFile(&quot;TSpectrum3.root&quot;);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   sm=(TH3F*)
f-&gt;Get(&quot;back;1&quot;);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TCanvas *Background = new
TCanvas(&quot;Smoothing&quot;,&quot;Markov smoothing&quot;,10,10,1000,700);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TSpectrum3 *s = new
TSpectrum3();</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for (i = 0; i &lt; nbinsx;
i++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     for (j = 0; j &lt;
nbinsy; j++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                  for (k = 0;
k &lt; nbinsz; k++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                       source[i][j][k]
= sm-&gt;GetBinContent(i + 1,j + 1,k + 1);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                    } </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                 }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
s-&gt;SmoothMarkov(source,nbinsx,nbinsy,nbinsz,3);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for (i = 0; i &lt; nbinsx;
i++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     for (j = 0; j &lt;
nbinsy; j++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>        for (k = 0; k &lt;
nbinsz; k++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>          
sm-&gt;SetBinContent(i + 1,j + 1,k + 1, source[i][j][k]);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>        }    </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
sm-&gt;Draw(&quot;&quot;);  </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>}</span></p>

</div>

<!-- */
// --> End_Html

   Int_t xmin,xmax,ymin,ymax,zmin,zmax,i,j,k,l;
   Double_t a,b,maxch;
   Double_t nom,nip,nim,sp,sm,spx,smx,spy,smy,spz,smz,plocha=0;
   if(averWindow<=0)
      return "Averaging Window must be positive";
   Double_t***working_space = new Double_t**[ssizex];
   for(i = 0;i < ssizex; i++){
      working_space[i] = new Double_t*[ssizey];
      for(j = 0;j < ssizey; j++)
         working_space[i][j] = new Double_t[ssizez];
   }
   xmin = 0;
   xmax = ssizex - 1;
   ymin = 0;
   ymax = ssizey - 1;
   zmin = 0;
   zmax = ssizez - 1;
   for(i = 0,maxch = 0;i < ssizex; i++){
      for(j = 0;j < ssizey; j++){
         for(k = 0;k < ssizez; k++){
            working_space[i][j][k] = 0;
            if(maxch < source[i][j][k])
               maxch = source[i][j][k];
            plocha += source[i][j][k];
         }
      }
   }
   if(maxch == 0) {
      delete [] working_space;
      return 0;
   }

   nom = 0;
   working_space[xmin][ymin][zmin] = 1;
   for(i = xmin;i < xmax; i++){
      nip = source[i][ymin][zmin] / maxch;
      nim = source[i + 1][ymin][zmin] / maxch;
      sp = 0,sm = 0;
      for(l = 1;l <= averWindow; l++){
         if((i + l) > xmax)
            a = source[xmax][ymin][zmin] / maxch;

         else
            a = source[i + l][ymin][zmin] / maxch;

         b = a - nip;
         if(a + nip <= 0)
            a = 1;

         else
            a = TMath::Sqrt(a + nip);

         b = b / a;
         b = TMath::Exp(b);
         sp = sp + b;
         if(i - l + 1 < xmin)
            a = source[xmin][ymin][zmin] / maxch;

         else
            a = source[i - l + 1][ymin][zmin] / maxch;

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
      a = working_space[i + 1][ymin][zmin] = a * working_space[i][ymin][zmin];
      nom = nom + a;
   }
   for(i = ymin;i < ymax; i++){
      nip = source[xmin][i][zmin] / maxch;
      nim = source[xmin][i + 1][zmin] / maxch;
      sp = 0,sm = 0;
      for(l = 1;l <= averWindow; l++){
         if((i + l) > ymax)
            a = source[xmin][ymax][zmin] / maxch;

         else
            a = source[xmin][i + l][zmin] / maxch;

         b = a - nip;
         if(a + nip <= 0)
            a = 1;

         else
            a = TMath::Sqrt(a + nip);

         b = b / a;
         b = TMath::Exp(b);
         sp = sp + b;
         if(i - l + 1 < ymin)
            a = source[xmin][ymin][zmin] / maxch;

         else
            a = source[xmin][i - l + 1][zmin] / maxch;

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
      a = working_space[xmin][i + 1][zmin] = a * working_space[xmin][i][zmin];
      nom = nom + a;
   }
   for(i = zmin;i < zmax; i++){
      nip = source[xmin][ymin][i] / maxch;
      nim = source[xmin][ymin][i + 1] / maxch;
      sp = 0,sm = 0;
      for(l = 1;l <= averWindow; l++){
         if((i + l) > zmax)
            a = source[xmin][ymin][zmax] / maxch;

         else
            a = source[xmin][ymin][i + l] / maxch;

         b = a - nip;
         if(a + nip <= 0)
            a = 1;

         else
            a = TMath::Sqrt(a + nip);

         b = b / a;
         b = TMath::Exp(b);
         sp = sp + b;
         if(i - l + 1 < zmin)
            a = source[xmin][ymin][zmin] / maxch;

         else
            a = source[xmin][ymin][i - l + 1] / maxch;

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
      a = working_space[xmin][ymin][i + 1] = a * working_space[xmin][ymin][i];
      nom = nom + a;
   }
   for(i = xmin;i < xmax; i++){
      for(j = ymin;j < ymax; j++){
         nip = source[i][j + 1][zmin] / maxch;
         nim = source[i + 1][j + 1][zmin] / maxch;
         spx = 0,smx = 0;
         for(l = 1;l <= averWindow; l++){
            if(i + l > xmax)
               a = source[xmax][j][zmin] / maxch;

            else
               a = source[i + l][j][zmin] / maxch;

            b = a - nip;
            if(a + nip <= 0)
               a = 1;

            else
               a = TMath::Sqrt(a + nip);

            b = b / a;
            b = TMath::Exp(b);
            spx = spx + b;
            if(i - l + 1 < xmin)
               a = source[xmin][j][zmin] / maxch;

            else
               a = source[i - l + 1][j][zmin] / maxch;

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
         nip = source[i + 1][j][zmin] / maxch;
         nim = source[i + 1][j + 1][zmin] / maxch;
         for(l = 1;l <= averWindow; l++){
            if(j + l > ymax)
               a = source[i][ymax][zmin] / maxch;

            else
               a = source[i][j + l][zmin] / maxch;

            b = a - nip;
            if(a + nip <= 0)
               a = 1;

            else
               a = TMath::Sqrt(a + nip);

            b = b / a;
            b = TMath::Exp(b);
            spy = spy + b;
            if(j - l + 1 < ymin)
               a = source[i][ymin][zmin] / maxch;

            else
               a = source[i][j - l + 1][zmin] / maxch;

            b = a - nim;
            if(a + nim <= 0)
               a = 1;

            else
               a = TMath::Sqrt(a + nim);

            b = b / a;
            b = TMath::Exp(b);
            smy = smy + b;
         }
         a = (spx * working_space[i][j + 1][zmin] + spy * working_space[i + 1][j][zmin]) / (smx + smy);
         working_space[i + 1][j + 1][zmin] = a;
         nom = nom + a;
      }
   }
   for(i = xmin;i < xmax; i++){
      for(j = zmin;j < zmax; j++){
         nip = source[i][ymin][j + 1] / maxch;
         nim = source[i + 1][ymin][j + 1] / maxch;
         spx = 0,smx = 0;
         for(l = 1;l <= averWindow; l++){
            if(i + l > xmax)
               a = source[xmax][ymin][j] / maxch;

            else
               a = source[i + l][ymin][j] / maxch;

            b = a - nip;
            if(a + nip <= 0)
               a = 1;

            else
               a = TMath::Sqrt(a + nip);

            b = b / a;
            b = TMath::Exp(b);
            spx = spx + b;
            if(i - l + 1 < xmin)
               a = source[xmin][ymin][j] / maxch;

            else
               a = source[i - l + 1][ymin][j] / maxch;

            b = a - nim;
            if(a + nim <= 0)
               a = 1;

            else
               a = TMath::Sqrt(a + nim);

            b = b / a;
            b = TMath::Exp(b);
            smx = smx + b;
         }
         spz = 0,smz = 0;
         nip = source[i + 1][ymin][j] / maxch;
         nim = source[i + 1][ymin][j + 1] / maxch;
         for(l = 1;l <= averWindow; l++){
            if(j + l > zmax)
               a = source[i][ymin][zmax] / maxch;

            else
               a = source[i][ymin][j + l] / maxch;

            b = a - nip;
            if(a + nip <= 0)
               a = 1;

            else
               a = TMath::Sqrt(a + nip);

            b = b / a;
            b = TMath::Exp(b);
            spz = spz + b;
            if(j - l + 1 < zmin)
               a = source[i][ymin][zmin] / maxch;

            else
               a = source[i][ymin][j - l + 1] / maxch;

            b = a - nim;
            if(a + nim <= 0)
               a = 1;

            else
               a = TMath::Sqrt(a + nim);

            b = b / a;
            b = TMath::Exp(b);
            smz = smz + b;
         }
         a = (spx * working_space[i][ymin][j + 1] + spz * working_space[i + 1][ymin][j]) / (smx + smz);
         working_space[i + 1][ymin][j + 1] = a;
         nom = nom + a;
      }
   }
   for(i = ymin;i < ymax; i++){
      for(j = zmin;j < zmax; j++){
         nip = source[xmin][i][j + 1] / maxch;
         nim = source[xmin][i + 1][j + 1] / maxch;
         spy = 0,smy = 0;
         for(l = 1;l <= averWindow; l++){
            if(i + l > ymax)
               a = source[xmin][ymax][j] / maxch;

            else
               a = source[xmin][i + l][j] / maxch;

            b = a - nip;
            if(a + nip <= 0)
               a = 1;

            else
               a = TMath::Sqrt(a + nip);

            b = b / a;
            b = TMath::Exp(b);
            spy = spy + b;
            if(i - l + 1 < ymin)
               a = source[xmin][ymin][j] / maxch;

            else
               a = source[xmin][i - l + 1][j] / maxch;

            b = a - nim;
            if(a + nim <= 0)
               a = 1;

            else
               a = TMath::Sqrt(a + nim);

            b = b / a;
            b = TMath::Exp(b);
            smy = smy + b;
         }
         spz = 0,smz = 0;
         nip = source[xmin][i + 1][j] / maxch;
         nim = source[xmin][i + 1][j + 1] / maxch;
         for(l = 1;l <= averWindow; l++){
            if(j + l > zmax)
               a = source[xmin][i][zmax] / maxch;

            else
               a = source[xmin][i][j + l] / maxch;

            b = a - nip;
            if(a + nip <= 0)
               a = 1;

            else
               a = TMath::Sqrt(a + nip);

            b = b / a;
            b = TMath::Exp(b);
            spz = spz + b;
            if(j - l + 1 < zmin)
               a = source[xmin][i][zmin] / maxch;

            else
               a = source[xmin][i][j - l + 1] / maxch;

            b = a - nim;
            if(a + nim <= 0)
               a = 1;

            else
               a = TMath::Sqrt(a + nim);

            b = b / a;
            b = TMath::Exp(b);
            smz = smz + b;
         }
         a = (spy * working_space[xmin][i][j + 1] + spz * working_space[xmin][i + 1][j]) / (smy + smz);
         working_space[xmin][i + 1][j + 1] = a;
         nom = nom + a;
      }
   }
   for(i = xmin;i < xmax; i++){
      for(j = ymin;j < ymax; j++){
         for(k = zmin;k < zmax; k++){
            nip = source[i][j + 1][k + 1] / maxch;
            nim = source[i + 1][j + 1][k + 1] / maxch;
            spx = 0,smx = 0;
            for(l = 1;l <= averWindow; l++){
               if(i + l > xmax)
                  a = source[xmax][j][k] / maxch;

               else
                  a = source[i + l][j][k] / maxch;

               b = a - nip;
               if(a + nip <= 0)
                  a = 1;

               else
                  a = TMath::Sqrt(a + nip);

               b = b / a;
               b = TMath::Exp(b);
               spx = spx + b;
               if(i - l + 1 < xmin)
                  a = source[xmin][j][k] / maxch;

               else
                  a = source[i - l + 1][j][k] / maxch;

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
            nip = source[i + 1][j][k + 1] / maxch;
            nim = source[i + 1][j + 1][k + 1] / maxch;
            for(l = 1;l <= averWindow; l++){
               if(j + l > ymax)
                  a = source[i][ymax][k] / maxch;

               else
                  a = source[i][j + l][k] / maxch;

               b = a - nip;
               if(a + nip <= 0)
                  a = 1;

               else
                  a = TMath::Sqrt(a + nip);

               b = b / a;
               b = TMath::Exp(b);
               spy = spy + b;
               if(j - l + 1 < ymin)
                  a = source[i][ymin][k] / maxch;

               else
                  a = source[i][j - l + 1][k] / maxch;

               b = a - nim;
               if(a + nim <= 0)
                  a = 1;

               else
                  a = TMath::Sqrt(a + nim);

               b = b / a;
               b = TMath::Exp(b);
               smy = smy + b;
            }
            spz = 0,smz = 0;
            nip = source[i + 1][j + 1][k] / maxch;
            nim = source[i + 1][j + 1][k + 1] / maxch;
            for(l = 1;l <= averWindow; l++){
               if(j + l > zmax)
                  a = source[i][j][zmax] / maxch;

               else
                  a = source[i][j][k + l] / maxch;

               b = a - nip;
               if(a + nip <= 0)
                  a = 1;

               else
                  a = TMath::Sqrt(a + nip);

               b = b / a;
               b = TMath::Exp(b);
               spz = spz + b;
               if(j - l + 1 < ymin)
                  a = source[i][j][zmin] / maxch;

               else
                  a = source[i][j][k - l + 1] / maxch;

               b = a - nim;
               if(a + nim <= 0)
                  a = 1;

               else
                  a = TMath::Sqrt(a + nim);

               b = b / a;
               b = TMath::Exp(b);
               smz = smz+b;
            }
            a = (spx * working_space[i][j + 1][k + 1] + spy * working_space[i + 1][j][k + 1] + spz * working_space[i + 1][j + 1][k]) / (smx + smy + smz);
            working_space[i + 1][j + 1][k + 1] = a;
            nom = nom + a;
         }
      }
   }
   for(i = xmin;i <= xmax; i++){
      for(j = ymin;j <= ymax; j++){
         for(k = zmin;k <= zmax; k++){
            working_space[i][j][k] = working_space[i][j][k] / nom;
         }
      }
   }
   for(i = 0;i < ssizex; i++){
      for(j = 0;j < ssizey; j++){
         for(k = 0;k < ssizez; k++){
            source[i][j][k] = plocha * working_space[i][j][k];
         }
      }
   }
   for(i = 0;i < ssizex; i++){
      for(j = 0;j < ssizey; j++)
         delete[] working_space[i][j];
      delete[] working_space[i];
   }
   delete[] working_space;
   return 0;
}

//______________________________________________________________________________________________________________________________
const char *TSpectrum3::Deconvolution(Double_t***source, const Double_t***resp,
                                       Int_t ssizex, Int_t ssizey, Int_t ssizez,
                                       Int_t numberIterations,
                                       Int_t numberRepetitions,
                                       Double_t boost)
{
/////////////////////////////////////////////////////////////////////////////
// THREE-DIMENSIONAL DECONVOLUTION FUNCTION                                //
// This function calculates deconvolution from source spectrum             //
// according to response spectrum                                          //
// The result is placed in the cube pointed by source pointer.             //
//                                                                         //
// Function parameters:                                                    //
// source-pointer to the cube of source spectrum                           //
// resp-pointer to the cube of response spectrum                           //
// ssizex-x length of source and response spectra                          //
// ssizey-y length of source and response spectra                          //
// ssizey-y length of source and response spectra                          //
// numberIterations, for details we refer to manual                        //
// numberRepetitions, for details we refer to manual                       //
// boost, boosting factor, for details we refer to manual                  //
//                                                                         //
/////////////////////////////////////////////////////////////////////////////
//Begin_Html <!--
/* -->

<div class=Section3>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:14.0pt'>Deconvolution</span></b></p>

<p class=MsoNormal style='text-align:justify'><i>&nbsp;</i></p>

<p class=MsoNormal style='text-align:justify'><i>Goal: Improvement of the
resolution in spectra, decomposition of multiplets</i></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>Mathematical formulation of the 3-dimensional convolution
system is</p>

<p class=MsoNormal style='margin-left:18.0pt'>

<table cellpadding=0 cellspacing=0 align=left>
 <tr>
  <td width=69 height=1></td>
 </tr>
 <tr>
  <td></td>
  <td><img width=334 height=67 src="gif/spectrum3_deconvolution_image001.gif"></td>
 </tr>
</table>

<span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>&nbsp;</p>

<br clear=ALL>

<p class=MsoNormal>where h(i,j,k) is the impulse response function, x, y are
input and output fields, respectively, <sub><img width=69 height=24
src="gif/spectrum3_deconvolution_image002.gif"></sub>, are the lengths of x and h fields<i>
</i></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>let us assume that we know the response and the output fields (spectra)
of the above given system. </p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>the deconvolution represents solution of the overdetermined system of
linear equations, i.e.,  the calculation of the field <b>x.</b></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>from numerical stability point of view the operation of deconvolution is
extremely critical (ill-posed  problem) as well as time consuming operation. </p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>the Gold deconvolution algorithm proves to work very well even for
2-dimensional systems. Generalization of the algorithm for 2-dimensional
systems was presented in [1], and for multidimensional systems in [2].</p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>for Gold deconvolution algorithm as well as for boosted deconvolution
algorithm we refer also to TSpectrum and TSpectrum2 </p>

<p class=MsoNormal><i>&nbsp;</i></p>

<p class=MsoNormal><i>Function:</i></p>

<p class=MsoNormal style='text-align:justify'><b>const <a
href="http://root.cern.ch/root/html/ListOfTypes.html#char">char</a>* </b><a
name="TSpectrum:Deconvolution1"></a><a
href="http://root.cern.ch/root/html/TSpectrum.html#TSpectrum:Fit1Awmi"><b>TSpectrum3::Deconvolution</b></a><b>(<a
href="http://root.cern.ch/root/html/ListOfTypes.html#double">double</a> ***fSource,
const <a href="http://root.cern.ch/root/html/ListOfTypes.html#double">double</a>
***fResp, <a href="http://root.cern.ch/root/html/ListOfTypes.html#int">int</a>
fSizex, <a href="http://root.cern.ch/root/html/ListOfTypes.html#int">int</a>
fSizey, <a href="http://root.cern.ch/root/html/ListOfTypes.html#int">int</a>
fSizez, <a href="http://root.cern.ch/root/html/ListOfTypes.html#int">int</a>
fNumberIterations, <a href="http://root.cern.ch/root/html/ListOfTypes.html#int">int</a>
fNumberRepetitions, <a
href="http://root.cern.ch/root/html/ListOfTypes.html#double">double</a> fBoost)</b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'>This function calculates
deconvolution from source spectrum according to response spectrum using Gold
deconvolution algorithm. The result is placed in the field pointed by source
pointer. On successful completion it returns 0. On error it returns pointer to
the string describing error. If desired after every fNumberIterations one can apply
boosting operation (exponential function with exponent given by fBoost
coefficient) and repeat it fNumberRepetitions times.</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><i><span style='color:red'>Parameters:</span></i></p>

<p class=MsoNormal>        <b>fSource</b>-pointer to the matrix of source
spectrum                  </p>

<p class=MsoNormal>        <b>fResp</b>-pointer to the matrix of response
spectrum                  </p>

<p class=MsoNormal>        <b>fSizex, fSizey, fSizez</b> -lengths of the
spectrum matrix                                 </p>

<p class=MsoNormal style='text-align:justify'>        <b>fNumberIterations</b>-number
of iterations </p>

<p class=MsoNormal style='text-align:justify'>        <b>fNumberRepetitions</b>-number
of repetitions for boosted deconvolution. It must be </p>

<p class=MsoNormal style='text-align:justify'>        greater or equal to one.</p>

<p class=MsoNormal style='text-align:justify'>        <b>fBoost</b>-boosting
coefficient, applies only if fNumberRepetitions is greater than one.  </p>

<p class=MsoNormal style='text-align:justify'>        <span style='color:fuchsia'>Recommended
range &lt;1,2&gt;.</span></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><b><i>References:</i></b></p>

<p class=MsoNormal style='text-align:justify'> [1] <span lang=SK>M.
Morhá&#269;, J. Kliman, V. Matoušek, M. Veselský, I. Turzo</span>.: Efficient
one- and two-dimensional Gold deconvolution and its application to gamma-ray
spectra decomposition. NIM, A401 (1997) 385-408.</p>

<p class=MsoNormal style='text-align:justify'>[2] Morhá&#269; M., Matoušek V.,
Kliman J., Efficient algorithm of multidimensional deconvolution and its
application to nuclear data processing, Digital Signal Processing 13 (2003)
144. </p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal><i>Example 1 – script Decon.c :</i></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>response function (usually peak) should be shifted to the beginning of
the coordinate system (see Fig. 1)</p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'><img
border=0 width=601 height=368 src="gif/spectrum3_deconvolution_image003.jpg"></span></b></p>

<p class=MsoNormal style='text-align:justify'><b>Fig. 1 Three-dimensional
response spectrum</b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><img border=0 width=601 height=368
src="gif/spectrum3_deconvolution_image004.jpg"></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><b>Fig. 2 Three-dimensional input
spectrum (before deconvolution)</b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><img border=0 width=601 height=368
src="gif/spectrum3_deconvolution_image005.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b>Fig. 3 Spectrum from Fig. 2
after deconvolution (100 iterations)</b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><b><span style='color:#339966'>Script:</span></b></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// Example to illustrate the
Gold deconvolution (class TSpectrum3).</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// To execute this example,
do</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// root &gt; .x Decon3.C</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>&nbsp;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>#include &lt;TSpectrum3&gt;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>&nbsp;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>void Decon3() {</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Int_t i, j, k;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   </span><span lang=FR
style='font-size:10.0pt'>Int_t nbinsx = 32;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t nbinsy = 32;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t nbinsz =
32;   </span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t xmin  = 0;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t xmax  =
nbinsx;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t ymin  = 0;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t ymax  =
nbinsy;   </span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t zmin  = 0;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   </span><span
style='font-size:10.0pt'>Int_t zmax  = nbinsz;      </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t*** source = new
Double_t**[nbinsx];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t*** resp = new Double_t
**[nbinsx];      </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for(i=0;i&lt;nbinsx;i++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>      source[i]=new Double_t*
[nbinsy];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     
for(j=0;j&lt;nbinsy;j++)</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>         source[i][j]=new
Double_t[nbinsz];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }           </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for(i=0;i&lt;nbinsx;i++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>      resp[i]=new Double_t*
[nbinsy];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     
for(j=0;j&lt;nbinsy;j++)</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>         resp[i][j]=new Double_t
[nbinsz];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }              </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TH3F *decon_in = new
TH3F(&quot;decon_in&quot;,&quot;Deconvolution&quot;,nbinsx,xmin,xmax,nbinsy,ymin,ymax,nbinsz,zmin,zmax);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TH3F *decon_resp = new
TH3F(&quot;decon_resp&quot;,&quot;Deconvolution&quot;,nbinsx,xmin,xmax,nbinsy,ymin,ymax,nbinsz,zmin,zmax);  
</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TFile *f = new
TFile(&quot;TSpectrum3.root&quot;);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   decon_in=(TH3F*)
f-&gt;Get(&quot;decon_in;1&quot;);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   decon_resp=(TH3F*)
f-&gt;Get(&quot;decon_resp;1&quot;);   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TCanvas *Deconvolution =
new TCanvas(&quot;Deconvolution&quot;,&quot;Deconvolution of 3-dimensional
spectra&quot;,10,10,1000,700);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TSpectrum3 *s = new
TSpectrum3();</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for (i = 0; i &lt; nbinsx;
i++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     for (j = 0; j &lt;
nbinsy; j++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                  for (k = 0;
k &lt; nbinsz; k++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                       source[i][j][k]
= decon_in-&gt;GetBinContent(i + 1,j + 1,k + 1);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                       resp[i][j][k]
= decon_resp-&gt;GetBinContent(i + 1,j + 1,k + 1);                        </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                    } </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                 }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
s-&gt;Deconvolution(source,resp,nbinsx,nbinsy,nbinsz,100,1,1);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for (i = 0; i &lt; nbinsx;
i++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     for (j = 0; j &lt;
nbinsy; j++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>        for (k = 0; k &lt;
nbinsz; k++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>          
decon_in-&gt;SetBinContent(i + 1,j + 1,k + 1, source[i][j][k]);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>        }    </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }</span></p>

<p class=MsoNormal>   decon_in-&gt;Draw(&quot;&quot;);  </p>

<p class=MsoNormal>}</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><i>Example 2 – script Decon_hr.c :</i></p>

<p class=MsoNormal style='text-align:justify'>This example illustrates repeated
Gold deconvolution with boosting. After every 10 iterations we apply power
function with exponent = 2 to the spectrum given in Fig. 2.</p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><img border=0 width=601
height=368 src="gif/spectrum3_deconvolution_image006.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b>Fig. 4 Spectrum from Fig. 2
after boosted deconvolution (10 iterations repeated 10 times). It decomposes
completely cluster of peaks from Fig 2.</b></p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal><b><span style='color:#339966'>Script:</span></b></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// Example to illustrate the
Gold deconvolution (class TSpectrum3).</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// To execute this example,
do</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// root &gt; .x Decon3_hr.C</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>void Decon3_hr() {</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Int_t i, j, k;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   </span><span lang=FR
style='font-size:10.0pt'>Int_t nbinsx = 32;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t nbinsy = 32;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t nbinsz =
32;   </span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t xmin  = 0;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t xmax  =
nbinsx;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t ymin  = 0;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t ymax  =
nbinsy;   </span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t zmin  = 0;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   </span><span
style='font-size:10.0pt'>Int_t zmax  = nbinsz;      </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t*** source = new
Double_t**[nbinsx];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t*** resp = new Double_t
**[nbinsx];      </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for(i=0;i&lt;nbinsx;i++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>      source[i]=new Double_t*
[nbinsy];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     
for(j=0;j&lt;nbinsy;j++)</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>         source[i][j]=new
Double_t[nbinsz];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }           </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for(i=0;i&lt;nbinsx;i++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>      resp[i]=new Double_t*
[nbinsy];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     
for(j=0;j&lt;nbinsy;j++)</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>         resp[i][j]=new Double_t
[nbinsz];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }              </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TH3F *decon_in = new
TH3F(&quot;decon_in&quot;,&quot;Deconvolution&quot;,nbinsx,xmin,xmax,nbinsy,ymin,ymax,nbinsz,zmin,zmax);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TH3F *decon_resp = new
TH3F(&quot;decon_resp&quot;,&quot;Deconvolution&quot;,nbinsx,xmin,xmax,nbinsy,ymin,ymax,nbinsz,zmin,zmax);  
</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TFile *f = new
TFile(&quot;TSpectrum3.root&quot;);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   decon_in=(TH3F*)
f-&gt;Get(&quot;decon_in;1&quot;);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   decon_resp=(TH3F*)
f-&gt;Get(&quot;decon_resp;1&quot;);   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TCanvas *Deconvolution =
new TCanvas(&quot;Deconvolution&quot;,&quot;High resolution deconvolution of
3-dimensional spectra&quot;,10,10,1000,700);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TSpectrum3 *s = new
TSpectrum3();</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for (i = 0; i &lt; nbinsx;
i++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     for (j = 0; j &lt;
nbinsy; j++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                  for (k = 0;
k &lt; nbinsz; k++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                       source[i][j][k]
= decon_in-&gt;GetBinContent(i + 1,j + 1,k + 1);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                       resp[i][j][k]
= decon_resp-&gt;GetBinContent(i + 1,j + 1,k + 1);                        </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                    } </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                 }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
s-&gt;Deconvolution(source,resp,nbinsx,nbinsy,nbinsz,10,10,2);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for (i = 0; i &lt; nbinsx;
i++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     for (j = 0; j &lt;
nbinsy; j++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>        for (k = 0; k &lt;
nbinsz; k++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>          
decon_in-&gt;SetBinContent(i + 1,j + 1,k + 1, source[i][j][k]);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>        }    </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
decon_in-&gt;Draw(&quot;&quot;);  </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>}</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

</div>

<!-- */
// --> End_Html

   Int_t i, j, k, lhx, lhy, lhz, i1, i2, i3, j1, j2, j3, k1, k2, k3, lindex, i1min, i1max, i2min, i2max, i3min, i3max, j1min, j1max, j2min, j2max, j3min, j3max, positx = 0, posity = 0, positz = 0, repet;
   Double_t lda, ldb, ldc, area, maximum = 0;
   if (ssizex <= 0 || ssizey <= 0 || ssizez <= 0)
      return "Wrong parameters";
   if (numberIterations <= 0)
      return "Number of iterations must be positive";
   if (numberRepetitions <= 0)
      return "Number of repetitions must be positive";
   Double_t ***working_space=new Double_t** [ssizex];
   for(i=0;i<ssizex;i++){
      working_space[i]=new Double_t* [ssizey];
      for(j=0;j<ssizey;j++)
         working_space[i][j]=new Double_t [5*ssizez];
   }
   area = 0;
   lhx = -1, lhy = -1, lhz = -1;
   for (i = 0; i < ssizex; i++) {
      for (j = 0; j < ssizey; j++) {
         for (k = 0; k < ssizez; k++) {
            lda = resp[i][j][k];
            if (lda != 0) {
               if ((i + 1) > lhx)
                  lhx = i + 1;
               if ((j + 1) > lhy)
                  lhy = j + 1;
               if ((k + 1) > lhz)
                  lhz = k + 1;
            }
            working_space[i][j][k] = lda;
            area = area + lda;
            if (lda > maximum) {
               maximum = lda;
               positx = i, posity = j, positz = k;
            }
         }
      }
   }
   if (lhx == -1 || lhy == -1 || lhz == -1)
      return ("Zero response data");

//calculate ht*y and write into p
   for (i3 = 0; i3 < ssizez; i3++) {
      for (i2 = 0; i2 < ssizey; i2++) {
         for (i1 = 0; i1 < ssizex; i1++) {
            ldc = 0;
            for (j3 = 0; j3 <= (lhz - 1); j3++) {
               for (j2 = 0; j2 <= (lhy - 1); j2++) {
                  for (j1 = 0; j1 <= (lhx - 1); j1++) {
                     k3 = i3 + j3, k2 = i2 + j2, k1 = i1 + j1;
                     if (k3 >= 0 && k3 < ssizez && k2 >= 0 && k2 < ssizey && k1 >= 0 && k1 < ssizex) {
                        lda = working_space[j1][j2][j3];
                        ldb = source[k1][k2][k3];
                        ldc = ldc + lda * ldb;
                     }
                  }
               }
            }
            working_space[i1][i2][i3 + ssizez] = ldc;
         }
      }
   }

//calculate matrix b=ht*h
   i1min = -(lhx - 1), i1max = lhx - 1;
   i2min = -(lhy - 1), i2max = lhy - 1;
   i3min = -(lhz - 1), i3max = lhz - 1;
   for (i3 = i3min; i3 <= i3max; i3++) {
      for (i2 = i2min; i2 <= i2max; i2++) {
         for (i1 = i1min; i1 <= i1max; i1++) {
            ldc = 0;
            j3min = -i3;
            if (j3min < 0)
               j3min = 0;
            j3max = lhz - 1 - i3;
            if (j3max > lhz - 1)
               j3max = lhz - 1;
            for (j3 = j3min; j3 <= j3max; j3++) {
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
                     lda = working_space[j1][j2][j3];
                     if (i1 + j1 < ssizex && i2 + j2 < ssizey)
                        ldb = working_space[i1 + j1][i2 + j2][i3 + j3];
                     else
                        ldb = 0;
                     ldc = ldc + lda * ldb;
                  }
               }
            }
            working_space[i1 - i1min][i2 - i2min][i3 - i3min + 2 * ssizez ] = ldc;
         }
      }
   }

//initialization in x1 matrix
   for (i3 = 0; i3 < ssizez; i3++) {
      for (i2 = 0; i2 < ssizey; i2++) {
         for (i1 = 0; i1 < ssizex; i1++) {
            working_space[i1][i2][i3 + 3 * ssizez] = 1;
            working_space[i1][i2][i3 + 4 * ssizez] = 0;
         }
      }
   }

 //START OF ITERATIONS
   for (repet = 0; repet < numberRepetitions; repet++) {
      if (repet != 0) {
         for (i = 0; i < ssizex; i++) {
            for (j = 0; j < ssizey; j++) {
               for (k = 0; k < ssizez; k++) {
                  working_space[i][j][k + 3 * ssizez] = TMath::Power(working_space[i][j][k + 3 * ssizez],boost);
               }
            }
         }
      }
      for (lindex = 0; lindex < numberIterations; lindex++) {
         for (i3 = 0; i3 < ssizez; i3++) {
            for (i2 = 0; i2 < ssizey; i2++) {
               for (i1 = 0; i1 < ssizex; i1++) {
                  ldb = 0;
                  j3min = i3;
                  if (j3min > lhz - 1)
                     j3min = lhz - 1;
                  j3min = -j3min;
                  j3max = ssizez - i3 - 1;
                  if (j3max > lhz - 1)
                     j3max = lhz - 1;
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
                  for (j3 = j3min; j3 <= j3max; j3++) {
                     for (j2 = j2min; j2 <= j2max; j2++) {
                        for (j1 = j1min; j1 <= j1max; j1++) {
                           ldc =  working_space[j1 - i1min][j2 - i2min][j3 - i3min + 2 * ssizez];
                           lda = working_space[i1 + j1][i2 + j2][i3 + j3 + 3 * ssizez];
                           ldb = ldb + lda * ldc;
                        }
                     }
                  }
                  lda = working_space[i1][i2][i3 + 3 * ssizez];
                  ldc = working_space[i1][i2][i3 + 1 * ssizez];
                  if (ldc * lda != 0 && ldb != 0) {
                     lda = lda * ldc / ldb;
                  }

                  else
                     lda = 0;
                  working_space[i1][i2][i3 + 4 * ssizez] = lda;
               }
            }
         }
         for (i3 = 0; i3 < ssizez; i3++) {
            for (i2 = 0; i2 < ssizey; i2++) {
               for (i1 = 0; i1 < ssizex; i1++)
                  working_space[i1][i2][i3 + 3 * ssizez] = working_space[i1][i2][i3 + 4 * ssizez];
            }
         }
      }
   }
   for (i = 0; i < ssizex; i++) {
      for (j = 0; j < ssizey; j++){
         for (k = 0; k < ssizez; k++)
            source[(i + positx) % ssizex][(j + posity) % ssizey][(k + positz) % ssizez] = area * working_space[i][j][k + 3 * ssizez];
      }
   }
   delete [] working_space;
   return 0;
}

//__________________________________________________________________
Int_t TSpectrum3::SearchHighRes(const Double_t***source,Double_t***dest, Int_t ssizex, Int_t ssizey, Int_t ssizez,
                                 Double_t sigma, Double_t threshold,
                                 Bool_t backgroundRemove,Int_t deconIterations,
                                 Bool_t markov, Int_t averWindow)

{
/////////////////////////////////////////////////////////////////////////////
// THREE-DIMENSIONAL HIGH-RESOLUTION PEAK SEARCH FUNCTION                  //
// This function searches for peaks in source spectrum                     //
//  It is based on deconvolution method. First the background is           //
//  removed (if desired), then Markov spectrum is calculated               //
//  (if desired), then the response function is generated                  //
//  according to given sigma and deconvolution is carried out.             //
// It returns number of found peaks.                                       //
//                                                                         //
// Function parameters:                                                    //
// source-pointer to the matrix of source spectrum                         //
// dest-pointer to the matrix of resulting deconvolved spectrum            //
// ssizex-x length of source spectrum                                      //
// ssizey-y length of source spectrum                                      //
// ssizez-z length of source spectrum                                      //
// sigma-sigma of searched peaks, for details we refer to manual           //
// threshold-threshold value in % for selected peaks, peaks with           //
//                amplitude less than threshold*highest_peak/100           //
//                are ignored, see manual                                  //
//  backgroundRemove-logical variable, set if the removal of               //
//                background before deconvolution is desired               //
//  deconIterations-number of iterations in deconvolution operation        //
//  markov-logical variable, if it is true, first the source spectrum      //
//             is replaced by new spectrum calculated using Markov         //
//             chains method.                                              //
// averWindow-averanging window of searched peaks, for details             //
//                  we refer to manual (applies only for Markov method)    //
//                                                                         //
/////////////////////////////////////////////////////////////////////////////
//Begin_Html <!--
/* -->

<div class=Section4>

<p class=MsoNormal><b><span style='font-size:14.0pt'>Peaks searching</span></b></p>

<p class=MsoNormal style='text-align:justify'><i>&nbsp;</i></p>

<p class=MsoNormal style='text-align:justify'><i>Goal: to identify
automatically the peaks in spectrum with the presence of the continuous
background, one- and two-fold coincidences (ridges) and statistical
fluctuations - noise.</i> </p>

<p class=MsoNormal><span style='font-family:Arial'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'>The common problems connected
with correct peak identification in three-dimensional coincidence spectra are</p>

<ul style='margin-top:0mm' type=disc>
 <li class=MsoNormal style='text-align:justify'>non-sensitivity to noise, i.e.,
     only statistically relevant peaks should be identified</li>
 <li class=MsoNormal style='text-align:justify'>non-sensitivity of the
     algorithm to continuous background</li>
 <li class=MsoNormal style='text-align:justify'>non-sensitivity to one-fold coincidences
     (coincidences peak – peak – background in all dimensions) and their
     crossings</li>
 <li class=MsoNormal style='text-align:justify'>non-sensitivity to two-fold
     coincidences (coincidences peak – background – background in all
     dimensions) and their crossings</li>
 <li class=MsoNormal style='text-align:justify'>ability to identify peaks close
     to the edges of the spectrum region</li>
 <li class=MsoNormal style='text-align:justify'>resolution, decomposition of
     doublets and multiplets. The algorithm should be able to recognize close
     positioned peaks. </li>
</ul>

<p class=MsoNormal><i>&nbsp;</i></p>

<p class=MsoNormal><i>Function:</i></p>

<p class=MsoNormal style='text-align:justify'><b><a
href="http://root.cern.ch/root/html/ListOfTypes.html#Int_t">Int_t</a> </b><a
name="TSpectrum:Search1HighRes"></a><a
href="http://root.cern.ch/root/html/TSpectrum.html#TSpectrum:Fit1Awmi"><b>TSpectrum3::SearchHighRes</b></a><b>
(const <a href="http://root.cern.ch/root/html/ListOfTypes.html#double">double</a>
***fSource,<a href="http://root.cern.ch/root/html/ListOfTypes.html#double">double</a>
***fDest, <a href="http://root.cern.ch/root/html/ListOfTypes.html#int">int</a>
fSizex, <a href="http://root.cern.ch/root/html/ListOfTypes.html#int">int</a>
fSizey, <a href="http://root.cern.ch/root/html/ListOfTypes.html#int">int</a>
fSizez, <a href="http://root.cern.ch/root/html/ListOfTypes.html#double">double</a>
fSigma, <a href="http://root.cern.ch/root/html/ListOfTypes.html#double">double</a>
fThreshold, <a href="http://root.cern.ch/root/html/ListOfTypes.html#bool">bool</a>
fBackgroundRemove,<a href="http://root.cern.ch/root/html/ListOfTypes.html#int">int</a>
fDeconIterations, <a href="http://root.cern.ch/root/html/ListOfTypes.html#bool">bool</a>
fMarkov, <a href="http://root.cern.ch/root/html/ListOfTypes.html#int">int</a>
fAverWindow)   </b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'>This function searches for peaks
in source spectrum. It is based on deconvolution method. First the background
is removed (if desired), then Markov smoothed spectrum is calculated (if
desired), then the response function is generated according to given sigma and
deconvolution is carried out. On success it returns number of found peaks.</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><i><span style='color:red'>Parameters:</span></i></p>

<p class=MsoNormal style='text-align:justify'>        <b>fSource</b>-pointer to
the matrix of source spectrum                  </p>

<p class=MsoNormal style='text-align:justify'>        <b>fDest</b>-resulting
spectrum after deconvolution</p>

<p class=MsoNormal style='text-align:justify'>        <b>fSizex, fSizey, fSizez</b>
-lengths of the source and destination spectra                </p>

<p class=MsoNormal style='text-align:justify'>        <b>fSigma</b>-sigma of
searched peaks</p>

<p class=MsoNormal style='margin-left:22.8pt;text-align:justify'><b>fThreshold</b>-
threshold value in % for selected peaks, peaks with amplitude less than
threshold*highest_peak/100 are ignored</p>

<p class=MsoNormal style='margin-left:22.8pt;text-align:justify'><b>fBackgroundRemove</b>-
background_remove-logical variable, true if the removal of background before
deconvolution is desired  </p>

<p class=MsoNormal style='margin-left:22.8pt;text-align:justify'><b>fDeconIterations</b>-number
of iterations in deconvolution operation</p>

<p class=MsoNormal style='margin-left:22.8pt;text-align:justify'><b>fMarkov</b>-logical
variable, if it is true, first the source spectrum is replaced by new spectrum
calculated using Markov chains method </p>

<p class=MsoNormal style='margin-left:19.95pt;text-align:justify;text-indent:
2.85pt'><b>fAverWindow</b>-width of averaging smoothing window </p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><b><i>References:</i></b></p>

<p class=MsoNormal style='text-align:justify'>[1] M.A. Mariscotti: A method for
identification of peaks in the presence of background and its application to
spectrum analysis. NIM 50 (1967), 309-320.</p>

<p class=MsoNormal style='text-align:justify'>[2] <span lang=SK> M.
Morhá&#269;, J. Kliman, V. Matoušek, M. Veselský, I. Turzo</span>.:Identification
of peaks in multidimensional coincidence gamma-ray spectra. NIM, A443 (2000)
108-125.</p>

<p class=MsoNormal style='text-align:justify'>[3] Z.K. Silagadze, A new
algorithm for automatic photopeak searches. NIM A 376 (1996), 451.</p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal><b>Example of peak searching method</b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><a
href="http://root.cern.ch/root/html/src/TSpectrum.cxx.html#TSpectrum:Search1HighRes"
target="_parent">SearchHighRes</a> function provides users with the possibility
to vary the input parameters and with the access to the output deconvolved data
in the destination spectrum. Based on the output data one can tune the
parameters. </p>

<p class=MsoNormal><i>Example 1 – script Search3.c:</i></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><img border=0 width=601 height=368
src="gif/spectrum3_searching_image001.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b>Fig. 1 Three-dimensional
spectrum with 5 peaks (<sub><img border=0 width=40 height=19
src="gif/spectrum3_searching_image002.gif"></sub>, threshold=5%, 3 iterations steps in
the deconvolution)</b></p>

<p class=MsoNormal style='text-align:justify'><b>&nbsp;</b></p>

<p class=MsoNormal style='text-align:justify'><b><img border=0 width=601
height=368 src="gif/spectrum3_searching_image003.jpg"></b></p>

<p class=MsoNormal style='text-align:justify'><b>Fig. 2 Spectrum from Fig. 1
after background elimination and deconvolution</b></p>

<p class=MsoNormal><b><span style='color:#339966'>&nbsp;</span></b></p>

<p class=MsoNormal><b><span style='color:#339966'>Script:</span></b></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// Example to illustrate high
resolution peak searching function (class TSpectrum3).</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// To execute this example,
do</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// root &gt; .x Search3.C</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>void Search3() {</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Int_t i, j, k, nfound;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   </span><span lang=FR
style='font-size:10.0pt'>Int_t nbinsx = 32;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t nbinsy = 32;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t nbinsz =
32;   </span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t xmin  = 0;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t xmax  =
nbinsx;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t ymin  = 0;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t ymax  =
nbinsy;   </span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t zmin  = 0;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   </span><span
style='font-size:10.0pt'>Int_t zmax  = nbinsz;      </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t*** source = new
Double_t**[nbinsx];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t*** dest = new Double_t
**[nbinsx];      </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for(i=0;i&lt;nbinsx;i++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>      source[i]=new Double_t*
[nbinsy];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     
for(j=0;j&lt;nbinsy;j++)</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>         source[i][j]=new
Double_t[nbinsz];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }           </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for(i=0;i&lt;nbinsx;i++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>      dest[i]=new Double_t*
[nbinsy];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     
for(j=0;j&lt;nbinsy;j++)</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>         dest[i][j]=new Double_t
[nbinsz];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }              </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TH3F *search = new
TH3F(&quot;Search&quot;,&quot;Peak
searching&quot;,nbinsx,xmin,xmax,nbinsy,ymin,ymax,nbinsz,zmin,zmax);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TFile *f = new
TFile(&quot;TSpectrum3.root&quot;);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   search=(TH3F*)
f-&gt;Get(&quot;search2;1&quot;);   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TCanvas *Search = new
TCanvas(&quot;Search&quot;,&quot;Peak searching&quot;,10,10,1000,700);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TSpectrum3 *s = new
TSpectrum3();</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for (i = 0; i &lt; nbinsx;
i++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     for (j = 0; j &lt;
nbinsy; j++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                  for (k = 0;
k &lt; nbinsz; k++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                       source[i][j][k]
= search-&gt;GetBinContent(i + 1,j + 1,k + 1);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                    } </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                 }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   nfound =
s-&gt;SearchHighRes(source, dest, nbinsx, nbinsy, nbinsz, 2, 5, kTRUE, 3,
kFALSE, 3);   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   printf(&quot;Found %d
candidate peaks\n&quot;,nfound);   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for (i = 0; i &lt; nbinsx;
i++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     for (j = 0; j &lt;
nbinsy; j++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>        for (k = 0; k &lt;
nbinsz; k++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>          
search-&gt;SetBinContent(i + 1,j + 1,k + 1, dest[i][j][k]);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>        }    </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t *PosX = new
Double_t[nfound];         </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t *PosY = new
Double_t[nfound];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t *PosZ = new
Double_t[nfound];      </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   PosX =
s-&gt;GetPositionX();</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   PosY =
s-&gt;GetPositionY();         </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   PosZ =
s-&gt;GetPositionZ();            </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for(i=0;i&lt;nfound;i++)</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                    </span><span
lang=PL style='font-size:10.0pt'>printf(&quot;posx= %d, posy= %d, posz=
%d\n&quot;,(Int_t)(PosX[i]+0.5), (Int_t)(PosY[i]+0.5),
(Int_t)(PosZ[i]+0.5));           </span></p>

<p class=MsoNormal><span lang=PL style='font-size:10.0pt'>   </span><span
style='font-size:10.0pt'>search-&gt;Draw(&quot;&quot;);  </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>}</span></p>

</div>

<!-- */
// --> End_Html

   Int_t number_of_iterations = (Int_t)(4 * sigma + 0.5);
   Int_t k,lindex;
   Double_t lda,ldb,ldc,area,maximum;
   Int_t xmin,xmax,l,peak_index = 0,sizex_ext=ssizex + 4 * number_of_iterations,sizey_ext = ssizey + 4 * number_of_iterations,sizez_ext = ssizez + 4 * number_of_iterations,shift = 2 * number_of_iterations;
   Int_t ymin,ymax,zmin,zmax,i,j;
   Double_t a,b,maxch,plocha = 0,plocha_markov = 0;
   Double_t nom,nip,nim,sp,sm,spx,spy,smx,smy,spz,smz;
   Double_t p1,p2,p3,p4,p5,p6,p7,p8,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,r1,r2,r3,r4,r5,r6;
   Int_t x,y,z;
   Double_t pocet_sigma = 5;
   Int_t lhx,lhy,lhz,i1,i2,i3,j1,j2,j3,k1,k2,k3,i1min,i1max,i2min,i2max,i3min,i3max,j1min,j1max,j2min,j2max,j3min,j3max,positx,posity,positz;
   if(sigma < 1){
      Error("SearchHighRes", "Invalid sigma, must be greater than or equal to 1");
      return 0;
   }

   if(threshold<=0||threshold>=100){
      Error("SearchHighRes", "Invalid threshold, must be positive and less than 100");
      return 0;
   }

   j = (Int_t)(pocet_sigma*sigma+0.5);
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
      if(sizex_ext < 2 * number_of_iterations + 1 || sizey_ext < 2 * number_of_iterations + 1 || sizez_ext < 2 * number_of_iterations + 1){
         Error("SearchHighRes", "Too large clipping window");
         return 0;
      }
   }

   i = (Int_t)(4 * sigma + 0.5);
   i = 4 * i;
   Double_t ***working_space = new Double_t** [ssizex + i];
   for(j = 0;j < ssizex + i; j++){
      working_space[j] = new Double_t* [ssizey + i];
      for(k = 0;k < ssizey + i; k++)
         working_space[j][k] = new Double_t [5 * (ssizez + i)];
   }
   for(k = 0;k < sizez_ext; k++){
      for(j = 0;j < sizey_ext; j++){
        for(i = 0;i < sizex_ext; i++){
            if(i < shift){
               if(j < shift){
                  if(k < shift)
                     working_space[i][j][k + sizez_ext] = source[0][0][0];

                  else if(k >= ssizez + shift)
                     working_space[i][j][k + sizez_ext] = source[0][0][ssizez - 1];

                  else
                     working_space[i][j][k + sizez_ext] = source[0][0][k - shift];
               }

               else if(j >= ssizey + shift){
                  if(k < shift)
                     working_space[i][j][k + sizez_ext] = source[0][ssizey - 1][0];

                  else if(k >= ssizez + shift)
                     working_space[i][j][k + sizez_ext] = source[0][ssizey - 1][ssizez - 1];

                  else
                     working_space[i][j][k + sizez_ext] = source[0][ssizey - 1][k - shift];
               }

               else{
                  if(k < shift)
                     working_space[i][j][k + sizez_ext] = source[0][j - shift][0];

                  else if(k >= ssizez + shift)
                     working_space[i][j][k + sizez_ext] = source[0][j - shift][ssizez - 1];

                  else
                     working_space[i][j][k + sizez_ext] = source[0][j - shift][k - shift];
               }
            }

            else if(i >= ssizex + shift){
               if(j < shift){
                  if(k < shift)
                     working_space[i][j][k + sizez_ext] = source[ssizex - 1][0][0];

                  else if(k >= ssizez + shift)
                     working_space[i][j][k + sizez_ext] = source[ssizex - 1][0][ssizez - 1];

                  else
                     working_space[i][j][k + sizez_ext] = source[ssizex - 1][0][k - shift];
               }

               else if(j >= ssizey + shift){
                  if(k < shift)
                     working_space[i][j][k + sizez_ext] = source[ssizex - 1][ssizey - 1][0];

                  else if(k >= ssizez + shift)
                     working_space[i][j][k + sizez_ext] = source[ssizex - 1][ssizey - 1][ssizez - 1];

                  else
                     working_space[i][j][k + sizez_ext] = source[ssizex - 1][ssizey - 1][k - shift];
               }

               else{
                  if(k < shift)
                     working_space[i][j][k + sizez_ext] = source[ssizex - 1][j - shift][0];

                  else if(k >= ssizez + shift)
                     working_space[i][j][k + sizez_ext] = source[ssizex - 1][j - shift][ssizez - 1];

                  else
                     working_space[i][j][k + sizez_ext] = source[ssizex - 1][j - shift][k - shift];
               }
            }

            else{
               if(j < shift){
                  if(k < shift)
                     working_space[i][j][k + sizez_ext] = source[i - shift][0][0];

                  else if(k >= ssizez + shift)
                     working_space[i][j][k + sizez_ext] = source[i - shift][0][ssizez - 1];

                  else
                     working_space[i][j][k + sizez_ext] = source[i - shift][0][k - shift];
               }

               else if(j >= ssizey + shift){
                  if(k < shift)
                     working_space[i][j][k + sizez_ext] = source[i - shift][ssizey - 1][0];

                  else if(k >= ssizez + shift)
                     working_space[i][j][k + sizez_ext] = source[i - shift][ssizey - 1][ssizez - 1];

                  else
                     working_space[i][j][k + sizez_ext] = source[i - shift][ssizey - 1][k - shift];
               }

               else{
                  if(k < shift)
                     working_space[i][j][k + sizez_ext] = source[i - shift][j - shift][0];

                  else if(k >= ssizez + shift)
                     working_space[i][j][k + sizez_ext] = source[i - shift][j - shift][ssizez - 1];

                  else
                     working_space[i][j][k + sizez_ext] = source[i - shift][j - shift][k - shift];
               }
            }
         }
      }
   }
   if(backgroundRemove == true){
      for(i = 1;i <= number_of_iterations; i++){
        for(z = i;z < sizez_ext - i; z++){
           for(y = i;y < sizey_ext - i; y++){
             for(x = i;x < sizex_ext - i; x++){
               a = working_space[x][y][z + sizez_ext];
                  p1 = working_space[x + i][y + i][z - i + sizez_ext];
                  p2 = working_space[x - i][y + i][z - i + sizez_ext];
                  p3 = working_space[x + i][y - i][z - i + sizez_ext];
                  p4 = working_space[x - i][y - i][z - i + sizez_ext];
                  p5 = working_space[x + i][y + i][z + i + sizez_ext];
                  p6 = working_space[x - i][y + i][z + i + sizez_ext];
                  p7 = working_space[x + i][y - i][z + i + sizez_ext];
                  p8 = working_space[x - i][y - i][z + i + sizez_ext];
                  s1 = working_space[x + i][y    ][z - i + sizez_ext];
                  s2 = working_space[x    ][y + i][z - i + sizez_ext];
                  s3 = working_space[x - i][y    ][z - i + sizez_ext];
                  s4 = working_space[x    ][y - i][z - i + sizez_ext];
                  s5 = working_space[x + i][y    ][z + i + sizez_ext];
                  s6 = working_space[x    ][y + i][z + i + sizez_ext];
                  s7 = working_space[x - i][y    ][z + i + sizez_ext];
                  s8 = working_space[x    ][y - i][z + i + sizez_ext];
                  s9 = working_space[x - i][y + i][z     + sizez_ext];
                  s10 = working_space[x - i][y - i][z     +sizez_ext];
                  s11 = working_space[x + i][y + i][z     +sizez_ext];
                  s12 = working_space[x + i][y - i][z     +sizez_ext];
                  r1 = working_space[x    ][y    ][z - i + sizez_ext];
                  r2 = working_space[x    ][y    ][z + i + sizez_ext];
                  r3 = working_space[x - i][y    ][z     + sizez_ext];
                  r4 = working_space[x + i][y    ][z     + sizez_ext];
                  r5 = working_space[x    ][y + i][z     + sizez_ext];
                  r6 = working_space[x    ][y - i][z     + sizez_ext];
                  b = (p1 + p3) / 2.0;
                  if(b > s1)
                     s1 = b;

                  b = (p1 + p2) / 2.0;
                  if(b > s2)
                     s2 = b;

                  b = (p2 + p4) / 2.0;
                  if(b > s3)
                     s3 = b;

                  b = (p3 + p4) / 2.0;
                  if(b > s4)
                     s4 = b;

                  b = (p5 + p7) / 2.0;
                  if(b > s5)
                     s5 = b;

                  b = (p5 + p6) / 2.0;
                  if(b > s6)
                     s6 = b;

                  b = (p6 + p8) / 2.0;
                  if(b > s7)
                     s7 = b;

                  b = (p7 + p8) / 2.0;
                  if(b > s8)
                     s8 = b;

                  b = (p2 + p6) / 2.0;
                  if(b > s9)
                     s9 = b;

                  b = (p4 + p8) / 2.0;
                  if(b > s10)
                     s10 = b;

                  b = (p1 + p5) / 2.0;
                  if(b > s11)
                     s11 = b;

                  b = (p3 + p7) / 2.0;
                  if(b > s12)
                     s12 = b;

                  s1 = s1 - (p1 + p3) / 2.0;
                  s2 = s2 - (p1 + p2) / 2.0;
                  s3 = s3 - (p2 + p4) / 2.0;
                  s4 = s4 - (p3 + p4) / 2.0;
                  s5 = s5 - (p5 + p7) / 2.0;
                  s6 = s6 - (p5 + p6) / 2.0;
                  s7 = s7 - (p6 + p8) / 2.0;
                  s8 = s8 - (p7 + p8) / 2.0;
                  s9 = s9 - (p2 + p6) / 2.0;
                  s10 = s10 - (p4 + p8) / 2.0;
                  s11 = s11 - (p1 + p5) / 2.0;
                  s12 = s12 - (p3 + p7) / 2.0;
                  b = (s1 + s3) / 2.0 + (s2 + s4) / 2.0 + (p1 + p2 + p3 + p4) / 4.0;
                  if(b > r1)
                     r1 = b;

                  b = (s5 + s7) / 2.0 + (s6 + s8) / 2.0 + (p5 + p6 + p7 + p8) / 4.0;
                  if(b > r2)
                     r2 = b;

                  b = (s3 + s7) / 2.0 + (s9 + s10) / 2.0 + (p2 + p4 + p6 + p8) / 4.0;
                  if(b > r3)
                     r3 = b;

                  b = (s1 + s5) / 2.0 + (s11 + s12) / 2.0 + (p1 + p3 + p5 + p7) / 4.0;
                  if(b > r4)
                     r4 = b;

                  b = (s9 + s11) / 2.0 + (s2 + s6) / 2.0 + (p1 + p2 + p5 + p6) / 4.0;
                  if(b > r5)
                     r5 = b;

                  b = (s4 + s8) / 2.0 + (s10 + s12) / 2.0 + (p3 + p4 + p7 + p8) / 4.0;
                  if(b > r6)
                     r6 = b;

                  r1 = r1 - ((s1 + s3) / 2.0 + (s2 + s4) / 2.0 + (p1 + p2 + p3 + p4) / 4.0);
                  r2 = r2 - ((s5 + s7) / 2.0 + (s6 + s8) / 2.0 + (p5 + p6 + p7 + p8) / 4.0);
                  r3 = r3 - ((s3 + s7) / 2.0 + (s9 + s10) / 2.0 + (p2 + p4 + p6 + p8) / 4.0);
                  r4 = r4 - ((s1 + s5) / 2.0 + (s11 + s12) / 2.0 + (p1 + p3 + p5 + p7) / 4.0);
                  r5 = r5 - ((s9 + s11) / 2.0 + (s2 + s6) / 2.0 + (p1 + p2 + p5 + p6) / 4.0);
                  r6 = r6 - ((s4 + s8) / 2.0 + (s10 + s12) / 2.0 + (p3 + p4 + p7 + p8) / 4.0);
                  b = (r1 + r2) / 2.0 + (r3 + r4) / 2.0 + (r5 + r6) / 2.0 + (s1 + s3 + s5 + s7) / 4.0 + (s2 + s4 + s6 + s8) / 4.0 + (s9 + s10 + s11 + s12) / 4.0 + (p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8) / 8.0;
                  if(b < a)
                     a = b;

                  working_space[x][y][z] = a;
               }
            }
         }
         for(z = i;z < sizez_ext - i; z++){
            for(y = i;y < sizey_ext - i; y++){
               for(x = i;x < sizex_ext - i; x++){
                  working_space[x][y][z + sizez_ext] = working_space[x][y][z];
               }
            }
         }
      }
      for(k = 0;k < sizez_ext; k++){
         for(j = 0;j < sizey_ext; j++){
            for(i = 0;i < sizex_ext; i++){
               if(i < shift){
                  if(j < shift){
                     if(k < shift)
                        working_space[i][j][k + sizez_ext] = source[0][0][0] - working_space[i][j][k + sizez_ext];

                     else if(k >= ssizez + shift)
                        working_space[i][j][k + sizez_ext] = source[0][0][ssizez - 1] - working_space[i][j][k + sizez_ext];

                     else
                        working_space[i][j][k + sizez_ext] = source[0][0][k - shift] - working_space[i][j][k + sizez_ext];
                  }

                  else if(j >= ssizey + shift){
                     if(k < shift)
                        working_space[i][j][k + sizez_ext] = source[0][ssizey - 1][0] - working_space[i][j][k + sizez_ext];

                     else if(k >= ssizez + shift)
                        working_space[i][j][k + sizez_ext] = source[0][ssizey - 1][ssizez - 1] - working_space[i][j][k + sizez_ext];

                     else
                        working_space[i][j][k + sizez_ext] = source[0][ssizey - 1][k - shift] - working_space[i][j][k + sizez_ext];
                  }

                  else{
                     if(k < shift)
                        working_space[i][j][k + sizez_ext] = source[0][j - shift][0] - working_space[i][j][k + sizez_ext];

                     else if(k >= ssizez + shift)
                        working_space[i][j][k + sizez_ext] = source[0][j - shift][ssizez - 1] - working_space[i][j][k + sizez_ext];

                     else
                        working_space[i][j][k + sizez_ext] = source[0][j - shift][k - shift] - working_space[i][j][k + sizez_ext];
                  }
               }

               else if(i >= ssizex + shift){
                  if(j < shift){
                     if(k < shift)
                        working_space[i][j][k + sizez_ext] = source[ssizex - 1][0][0] - working_space[i][j][k + sizez_ext];

                     else if(k >= ssizez + shift)
                        working_space[i][j][k + sizez_ext] = source[ssizex - 1][0][ssizez - 1] - working_space[i][j][k + sizez_ext];

                     else
                        working_space[i][j][k + sizez_ext] = source[ssizex - 1][0][k - shift] - working_space[i][j][k + sizez_ext];
                  }

                  else if(j >= ssizey + shift){
                     if(k < shift)
                        working_space[i][j][k + sizez_ext] = source[ssizex - 1][ssizey - 1][0] - working_space[i][j][k + sizez_ext];

                     else if(k >= ssizez + shift)
                        working_space[i][j][k + sizez_ext] = source[ssizex - 1][ssizey - 1][ssizez - 1] - working_space[i][j][k + sizez_ext];

                     else
                        working_space[i][j][k + sizez_ext] = source[ssizex - 1][ssizey - 1][k - shift] - working_space[i][j][k + sizez_ext];
                  }

                  else{
                     if(k < shift)
                        working_space[i][j][k + sizez_ext] = source[ssizex - 1][j - shift][0] - working_space[i][j][k + sizez_ext];

                     else if(k >= ssizez + shift)
                        working_space[i][j][k + sizez_ext] = source[ssizex - 1][j - shift][ssizez - 1] - working_space[i][j][k + sizez_ext];

                     else
                        working_space[i][j][k + sizez_ext] = source[ssizex - 1][j - shift][k - shift] - working_space[i][j][k + sizez_ext];
                  }
               }

               else{
                  if(j < shift){
                     if(k < shift)
                        working_space[i][j][k + sizez_ext] = source[i - shift][0][0] - working_space[i][j][k + sizez_ext];

                     else if(k >= ssizez + shift)
                        working_space[i][j][k + sizez_ext] = source[i - shift][0][ssizez - 1] - working_space[i][j][k + sizez_ext];

                     else
                        working_space[i][j][k + sizez_ext] = source[i - shift][0][k - shift] - working_space[i][j][k + sizez_ext];
                  }

                  else if(j >= ssizey + shift){
                     if(k < shift)
                        working_space[i][j][k + sizez_ext] = source[i - shift][ssizey - 1][0] - working_space[i][j][k + sizez_ext];

                     else if(k >= ssizez + shift)
                        working_space[i][j][k + sizez_ext] = source[i - shift][ssizey - 1][ssizez - 1] - working_space[i][j][k + sizez_ext];

                     else
                        working_space[i][j][k + sizez_ext] = source[i - shift][ssizey - 1][k - shift] - working_space[i][j][k + sizez_ext];
                  }

                  else{
                     if(k < shift)
                        working_space[i][j][k + sizez_ext] = source[i - shift][j - shift][0] - working_space[i][j][k + sizez_ext];

                     else if(k >= ssizez + shift)
                        working_space[i][j][k + sizez_ext] = source[i - shift][j - shift][ssizez - 1] - working_space[i][j][k + sizez_ext];

                     else
                        working_space[i][j][k + sizez_ext] = source[i - shift][j - shift][k - shift] - working_space[i][j][k + sizez_ext];
                  }
               }
            }
         }
      }
   }

   if(markov == true){
      for(i = 0;i < sizex_ext; i++){
         for(j = 0;j < sizey_ext; j++){
            for(k = 0;k < sizez_ext; k++){
               working_space[i][j][k + 2 * sizez_ext] = working_space[i][j][sizez_ext + k];
               plocha_markov = plocha_markov + working_space[i][j][sizez_ext + k];
            }
         }
      }
      xmin = 0;
      xmax = sizex_ext - 1;
      ymin = 0;
      ymax = sizey_ext - 1;
      zmin = 0;
      zmax = sizez_ext - 1;
      for(i = 0,maxch = 0;i < sizex_ext; i++){
         for(j = 0;j < sizey_ext;j++){
            for(k = 0;k < sizez_ext;k++){
               working_space[i][j][k] = 0;
               if(maxch < working_space[i][j][k + 2 * sizez_ext])
                  maxch = working_space[i][j][k + 2 * sizez_ext];

               plocha += working_space[i][j][k + 2 * sizez_ext];
            }
         }
      }
      if(maxch == 0) {
         delete [] working_space;
         return 0;
      }
      nom = 0;
      working_space[xmin][ymin][zmin] = 1;
      for(i = xmin;i < xmax; i++){
         nip = working_space[i][ymin][zmin + 2 * sizez_ext] / maxch;
         nim = working_space[i + 1][ymin][zmin + 2 * sizez_ext] / maxch;
         sp = 0,sm = 0;
         for(l = 1;l <= averWindow; l++){
            if((i + l) > xmax)
               a = working_space[xmax][ymin][zmin + 2 * sizez_ext] / maxch;

            else
               a = working_space[i + l][ymin][zmin + 2 * sizez_ext] / maxch;

            b = a - nip;
            if(a + nip <= 0)
               a = 1;

            else
               a = TMath::Sqrt(a + nip);

            b = b / a;
            b = TMath::Exp(b);
            sp = sp + b;
            if(i - l + 1 < xmin)
               a = working_space[xmin][ymin][zmin + 2 * sizez_ext] / maxch;

            else
               a = working_space[i - l + 1][ymin][zmin + 2 * sizez_ext] / maxch;

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
         a = working_space[i + 1][ymin][zmin] = a * working_space[i][ymin][zmin];
         nom = nom + a;
      }
      for(i = ymin;i < ymax; i++){
         nip = working_space[xmin][i][zmin + 2 * sizez_ext] / maxch;
         nim = working_space[xmin][i + 1][zmin + 2 * sizez_ext] / maxch;
         sp = 0,sm = 0;
         for(l = 1;l <= averWindow; l++){
            if((i + l) > ymax)
               a = working_space[xmin][ymax][zmin + 2 * sizez_ext] / maxch;

            else
               a = working_space[xmin][i + l][zmin + 2 * sizez_ext] / maxch;

            b = a - nip;
            if(a + nip <= 0)
               a = 1;

            else
               a = TMath::Sqrt(a + nip);

            b = b / a;
            b = TMath::Exp(b);
            sp = sp + b;
            if(i - l + 1 < ymin)
               a = working_space[xmin][ymin][zmin + 2 * sizez_ext] / maxch;

            else
               a = working_space[xmin][i - l + 1][zmin + 2 * sizez_ext] / maxch;

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
         a = working_space[xmin][i + 1][zmin] = a * working_space[xmin][i][zmin];
         nom = nom + a;
      }
      for(i = zmin;i < zmax;i++){
         nip = working_space[xmin][ymin][i + 2 * sizez_ext] / maxch;
         nim = working_space[xmin][ymin][i + 1 + 2 * sizez_ext] / maxch;
         sp = 0,sm = 0;
         for(l = 1;l <= averWindow;l++){
            if((i + l) > zmax)
               a = working_space[xmin][ymin][zmax + 2 * sizez_ext] / maxch;

            else
               a = working_space[xmin][ymin][i + l + 2 * sizez_ext] / maxch;

            b = a - nip;
            if(a + nip <= 0)
               a = 1;

            else
               a = TMath::Sqrt(a + nip);

            b = b / a;
            b = TMath::Exp(b);
            sp = sp + b;
            if(i - l + 1 < zmin)
               a = working_space[xmin][ymin][zmin + 2 * sizez_ext] / maxch;

            else
               a = working_space[xmin][ymin][i - l + 1 + 2 * sizez_ext] / maxch;

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
         a = working_space[xmin][ymin][i + 1] = a * working_space[xmin][ymin][i];
         nom = nom + a;
      }
      for(i = xmin;i < xmax; i++){
         for(j = ymin;j < ymax; j++){
            nip = working_space[i][j + 1][zmin + 2 * sizez_ext] / maxch;
            nim = working_space[i + 1][j + 1][zmin + 2 * sizez_ext] / maxch;
            spx = 0,smx = 0;
            for(l = 1;l <= averWindow; l++){
               if(i + l > xmax)
                  a = working_space[xmax][j][zmin + 2 * sizez_ext] / maxch;

               else
                  a = working_space[i + l][j][zmin + 2 * sizez_ext] / maxch;

               b = a - nip;
               if(a + nip <= 0)
                  a = 1;

               else
                  a = TMath::Sqrt(a + nip);

               b = b / a;
               b = TMath::Exp(b);
               spx = spx + b;
               if(i - l + 1 < xmin)
                  a = working_space[xmin][j][zmin + 2 * sizez_ext] / maxch;

               else
                  a = working_space[i - l + 1][j][zmin + 2 * sizez_ext] / maxch;

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
            nip = working_space[i + 1][j][zmin + 2 * sizez_ext] / maxch;
            nim = working_space[i + 1][j + 1][zmin + 2 * sizez_ext] / maxch;
            for(l = 1;l <= averWindow; l++){
               if(j + l > ymax)
                  a = working_space[i][ymax][zmin + 2 * sizez_ext] / maxch;

               else
                  a = working_space[i][j + l][zmin + 2 * sizez_ext] / maxch;

               b = a - nip;
               if(a + nip <= 0)
                  a = 1;

               else
                  a = TMath::Sqrt(a + nip);

               b = b / a;
               b = TMath::Exp(b);
               spy = spy + b;
               if(j - l + 1 < ymin)
                  a = working_space[i][ymin][zmin + 2 * sizez_ext] / maxch;

               else
                  a = working_space[i][j - l + 1][zmin + 2 * sizez_ext] / maxch;

               b = a - nim;
               if(a + nim <= 0)
                  a = 1;

               else
                  a = TMath::Sqrt(a + nim);

               b = b / a;
               b = TMath::Exp(b);
               smy = smy + b;
            }
            a = (spx * working_space[i][j + 1][zmin] + spy * working_space[i + 1][j][zmin]) / (smx + smy);
            working_space[i + 1][j + 1][zmin] = a;
            nom = nom + a;
         }
      }
      for(i = xmin;i < xmax;i++){
         for(j = zmin;j < zmax;j++){
            nip = working_space[i][ymin][j + 1 + 2 * sizez_ext] / maxch;
            nim = working_space[i + 1][ymin][j + 1 + 2 * sizez_ext] / maxch;
            spx = 0,smx = 0;
            for(l = 1;l <= averWindow; l++){
               if(i + l > xmax)
                 a = working_space[xmax][ymin][j + 2 * sizez_ext] / maxch;

               else
                  a = working_space[i + l][ymin][j + 2 * sizez_ext] / maxch;

               b = a - nip;
               if(a + nip <= 0)
                  a = 1;

               else
                  a = TMath::Sqrt(a + nip);

               b = b / a;
               b = TMath::Exp(b);
               spx = spx + b;
               if(i - l + 1 < xmin)
                  a = working_space[xmin][ymin][j + 2 * sizez_ext] / maxch;

               else
                  a = working_space[i - l + 1][ymin][j + 2 * sizez_ext] / maxch;

               b = a - nim;
               if(a + nim <= 0)
                  a = 1;

               else
                  a = TMath::Sqrt(a + nim);

               b = b / a;
               b = TMath::Exp(b);
               smx = smx + b;
            }
            spz = 0,smz = 0;
            nip = working_space[i + 1][ymin][j + 2 * sizez_ext] / maxch;
            nim = working_space[i + 1][ymin][j + 1 + 2 * sizez_ext] / maxch;
            for(l = 1;l <= averWindow; l++){
               if(j + l > zmax)
                  a = working_space[i][ymin][zmax + 2 * sizez_ext] / maxch;

               else
                  a = working_space[i][ymin][j + l + 2 * sizez_ext] / maxch;

               b = a - nip;
               if(a + nip <= 0)
                  a = 1;

               else
                  a = TMath::Sqrt(a + nip);

               b = b / a;
               b = TMath::Exp(b);
               spz = spz + b;
               if(j - l + 1 < zmin)
                  a = working_space[i][ymin][zmin + 2 * sizez_ext] / maxch;

               else
                  a = working_space[i][ymin][j - l + 1 + 2 * sizez_ext] / maxch;

               b = a - nim;
               if(a + nim <= 0)
                  a = 1;

               else
                  a = TMath::Sqrt(a + nim);

               b = b / a;
               b = TMath::Exp(b);
               smz = smz + b;
            }
            a = (spx * working_space[i][ymin][j + 1] + spz * working_space[i + 1][ymin][j]) / (smx + smz);
            working_space[i + 1][ymin][j + 1] = a;
            nom = nom + a;
         }
      }
      for(i = ymin;i < ymax;i++){
         for(j = zmin;j < zmax;j++){
            nip = working_space[xmin][i][j + 1 + 2 * sizez_ext] / maxch;
            nim = working_space[xmin][i + 1][j + 1 + 2 * sizez_ext] / maxch;
            spy = 0,smy = 0;
            for(l = 1;l <= averWindow; l++){
               if(i + l > ymax)
                  a = working_space[xmin][ymax][j + 2 * sizez_ext] / maxch;

               else
                  a = working_space[xmin][i + l][j + 2 * sizez_ext] / maxch;

               b = a - nip;
               if(a + nip <= 0)
                  a = 1;

               else
                  a = TMath::Sqrt(a + nip);

               b = b / a;
               b = TMath::Exp(b);
               spy = spy + b;
               if(i - l + 1 < ymin)
                  a = working_space[xmin][ymin][j + 2 * sizez_ext] / maxch;

               else
                  a = working_space[xmin][i - l + 1][j + 2 * sizez_ext] / maxch;

               b = a - nim;
               if(a + nim <= 0)
                  a = 1;

               else
                  a = TMath::Sqrt(a + nim);

               b = b / a;
               b = TMath::Exp(b);
               smy = smy + b;
            }
            spz = 0,smz = 0;
            nip = working_space[xmin][i + 1][j + 2 * sizez_ext] / maxch;
            nim = working_space[xmin][i + 1][j + 1 + 2 * sizez_ext] / maxch;
            for(l = 1;l <= averWindow; l++){
               if(j + l > zmax)
                  a = working_space[xmin][i][zmax + 2 * sizez_ext] / maxch;

               else
                  a = working_space[xmin][i][j + l + 2 * sizez_ext] / maxch;

               b = a - nip;
               if(a + nip <= 0)
                  a = 1;

               else
                  a = TMath::Sqrt(a + nip);

               b = b / a;
               b = TMath::Exp(b);
               spz = spz + b;
               if(j - l + 1 < zmin)
                  a = working_space[xmin][i][zmin + 2 * sizez_ext] / maxch;

               else
                  a = working_space[xmin][i][j - l + 1 + 2 * sizez_ext] / maxch;

               b = a - nim;
               if(a + nim <= 0)
                  a = 1;

               else
                  a = TMath::Sqrt(a + nim);

               b = b / a;
               b = TMath::Exp(b);
               smz = smz + b;
            }
            a = (spy * working_space[xmin][i][j + 1] + spz * working_space[xmin][i + 1][j]) / (smy + smz);
            working_space[xmin][i + 1][j + 1] = a;
            nom = nom + a;
         }
      }
      for(i = xmin;i < xmax; i++){
         for(j = ymin;j < ymax; j++){
            for(k = zmin;k < zmax; k++){
               nip = working_space[i][j + 1][k + 1 + 2 * sizez_ext] / maxch;
               nim = working_space[i + 1][j + 1][k + 1 + 2 * sizez_ext] / maxch;
               spx = 0,smx = 0;
               for(l = 1;l <= averWindow; l++){
                  if(i + l > xmax)
                     a = working_space[xmax][j][k + 2 * sizez_ext] / maxch;

                  else
                     a = working_space[i + l][j][k + 2 * sizez_ext] / maxch;

                  b = a - nip;
                  if(a + nip <= 0)
                     a = 1;

                  else
                     a = TMath::Sqrt(a + nip);

                  b = b / a;
                  b = TMath::Exp(b);
                  spx = spx + b;
                  if(i - l + 1 < xmin)
                     a = working_space[xmin][j][k + 2 * sizez_ext] / maxch;

                  else
                     a = working_space[i - l + 1][j][k + 2 * sizez_ext] / maxch;

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
               nip = working_space[i + 1][j][k + 1 + 2 * sizez_ext] / maxch;
               nim = working_space[i + 1][j + 1][k + 1 + 2 * sizez_ext] / maxch;
               for(l = 1;l <= averWindow; l++){
                  if(j + l > ymax)
                     a = working_space[i][ymax][k + 2 * sizez_ext] / maxch;

                  else
                     a = working_space[i][j + l][k + 2 * sizez_ext] / maxch;

                  b = a - nip;
                  if(a + nip <= 0)
                     a = 1;

                  else
                     a = TMath::Sqrt(a + nip);

                  b = b / a;
                  b = TMath::Exp(b);
                  spy = spy + b;
                  if(j - l + 1 < ymin)
                     a = working_space[i][ymin][k + 2 * sizez_ext] / maxch;

                  else
                     a = working_space[i][j - l + 1][k + 2 * sizez_ext] / maxch;

                  b = a - nim;
                  if(a + nim <= 0)
                     a = 1;

                  else
                     a = TMath::Sqrt(a + nim);

                  b = b / a;
                  b = TMath::Exp(b);
                  smy = smy + b;
               }
               spz = 0,smz = 0;
               nip = working_space[i + 1][j + 1][k + 2 * sizez_ext] / maxch;
               nim = working_space[i + 1][j + 1][k + 1 + 2 * sizez_ext] / maxch;
               for(l = 1;l <= averWindow; l++ ){
                  if(j + l > zmax)
                     a = working_space[i][j][zmax + 2 * sizez_ext] / maxch;

                  else
                     a = working_space[i][j][k + l + 2 * sizez_ext] / maxch;

                  b = a - nip;
                  if(a + nip <= 0)
                     a = 1;

                  else
                     a = TMath::Sqrt(a + nip);

                  b = b / a;
                  b = TMath::Exp(b);
                  spz = spz + b;
                  if(j - l + 1 < ymin)
                     a = working_space[i][j][zmin + 2 * sizez_ext] / maxch;

                  else
                     a = working_space[i][j][k - l + 1 + 2 * sizez_ext] / maxch;

                  b = a - nim;
                  if(a + nim <= 0)
                     a = 1;

                  else
                     a = TMath::Sqrt(a + nim);

                  b = b / a;
                  b = TMath::Exp(b);
                  smz = smz + b;
               }
               a = (spx * working_space[i][j + 1][k + 1] + spy * working_space[i + 1][j][k + 1] + spz * working_space[i + 1][j + 1][k]) / (smx + smy + smz);
               working_space[i + 1][j + 1][k + 1] = a;
               nom = nom + a;
            }
         }
      }
      a = 0;
      for(i = xmin;i <= xmax; i++){
         for(j = ymin;j <= ymax; j++){
            for(k = zmin;k <= zmax; k++){
               working_space[i][j][k] = working_space[i][j][k] / nom;
               a+=working_space[i][j][k];
            }
         }
      }
      for(i = 0;i < sizex_ext; i++){
         for(j = 0;j < sizey_ext; j++){
            for(k = 0;k < sizez_ext; k++){
               working_space[i][j][k + sizez_ext] = working_space[i][j][k] * plocha_markov / a;
            }
         }
      }
   }
   //deconvolution starts
   area = 0;
   lhx = -1,lhy = -1,lhz = -1;
   positx = 0,posity = 0,positz = 0;
   maximum = 0;
   //generate response cube
   for(i = 0;i < sizex_ext; i++){
      for(j = 0;j < sizey_ext; j++){
         for(k = 0;k < sizez_ext; k++){
            lda = (Double_t)i - 3 * sigma;
            ldb = (Double_t)j - 3 * sigma;
            ldc = (Double_t)k - 3 * sigma;
            lda = (lda * lda + ldb * ldb + ldc * ldc) / (2 * sigma * sigma);
            l = (Int_t)(1000 * exp(-lda));
            lda = l;
            if(lda!=0){
               if((i + 1) > lhx)
                  lhx = i + 1;

               if((j + 1) > lhy)
                  lhy = j + 1;

               if((k + 1) > lhz)
                  lhz = k + 1;
            }
            working_space[i][j][k] = lda;
            area = area + lda;
            if(lda > maximum){
               maximum = lda;
               positx = i,posity = j,positz = k;
            }
         }
      }
   }
   //read source cube
   for(i = 0;i < sizex_ext; i++){
      for(j = 0;j < sizey_ext; j++){
         for(k = 0;k < sizez_ext; k++){
            working_space[i][j][k + 2 * sizez_ext] = TMath::Abs(working_space[i][j][k + sizez_ext]);
         }
      }
   }
   //calculate ht*y and write into p
   for (i3 = 0; i3 < sizez_ext; i3++) {
      for (i2 = 0; i2 < sizey_ext; i2++) {
         for (i1 = 0; i1 < sizex_ext; i1++) {
            ldc = 0;
            for (j3 = 0; j3 <= (lhz - 1); j3++) {
               for (j2 = 0; j2 <= (lhy - 1); j2++) {
                  for (j1 = 0; j1 <= (lhx - 1); j1++) {
                     k3 = i3 + j3, k2 = i2 + j2, k1 = i1 + j1;
                     if (k3 >= 0 && k3 < sizez_ext && k2 >= 0 && k2 < sizey_ext && k1 >= 0 && k1 < sizex_ext) {
                        lda = working_space[j1][j2][j3];
                        ldb = working_space[k1][k2][k3+2*sizez_ext];
                        ldc = ldc + lda * ldb;
                     }
                  }
               }
            }
            working_space[i1][i2][i3 + sizez_ext] = ldc;
         }
      }
   }
//calculate b=ht*h
   i1min = -(lhx - 1), i1max = lhx - 1;
   i2min = -(lhy - 1), i2max = lhy - 1;
   i3min = -(lhz - 1), i3max = lhz - 1;
   for (i3 = i3min; i3 <= i3max; i3++) {
      for (i2 = i2min; i2 <= i2max; i2++) {
         for (i1 = i1min; i1 <= i1max; i1++) {
            ldc = 0;
            j3min = -i3;
            if (j3min < 0)
               j3min = 0;

            j3max = lhz - 1 - i3;
            if (j3max > lhz - 1)
               j3max = lhz - 1;

            for (j3 = j3min; j3 <= j3max; j3++) {
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
                     lda = working_space[j1][j2][j3];
                     if (i1 + j1 < sizex_ext && i2 + j2 < sizey_ext)
                        ldb = working_space[i1 + j1][i2 + j2][i3 + j3];

                     else
                        ldb = 0;

                     ldc = ldc + lda * ldb;
                  }
               }
            }
            working_space[i1 - i1min][i2 - i2min][i3 - i3min + 2 * sizez_ext ] = ldc;
         }
      }
   }
//initialization in x1 cube
   for (i3 = 0; i3 < sizez_ext; i3++) {
      for (i2 = 0; i2 < sizey_ext; i2++) {
         for (i1 = 0; i1 < sizex_ext; i1++) {
            working_space[i1][i2][i3 + 3 * sizez_ext] = 1;
            working_space[i1][i2][i3 + 4 * sizez_ext] = 0;
         }
      }
   }

//START OF ITERATIONS
   for (lindex=0;lindex<deconIterations;lindex++){
      for (i3 = 0; i3 < sizez_ext; i3++) {
         for (i2 = 0; i2 < sizey_ext; i2++) {
            for (i1 = 0; i1 < sizex_ext; i1++) {
               if (TMath::Abs(working_space[i1][i2][i3 + 3 * sizez_ext])>1e-6 && TMath::Abs(working_space[i1][i2][i3 + 1 * sizez_ext])>1e-6){
                  ldb = 0;
                  j3min = i3;
                  if (j3min > lhz - 1)
                     j3min = lhz - 1;

                  j3min = -j3min;
                  j3max = sizez_ext - i3 - 1;
                  if (j3max > lhz - 1)
                     j3max = lhz - 1;

                  j2min = i2;
                  if (j2min > lhy - 1)
                     j2min = lhy - 1;

                  j2min = -j2min;
                  j2max = sizey_ext - i2 - 1;
                  if (j2max > lhy - 1)
                     j2max = lhy - 1;

                  j1min = i1;
                  if (j1min > lhx - 1)
                     j1min = lhx - 1;

                  j1min = -j1min;
                  j1max = sizex_ext - i1 - 1;
                  if (j1max > lhx - 1)
                     j1max = lhx - 1;

                  for (j3 = j3min; j3 <= j3max; j3++) {
                     for (j2 = j2min; j2 <= j2max; j2++) {
                        for (j1 = j1min; j1 <= j1max; j1++) {
                           ldc =  working_space[j1 - i1min][j2 - i2min][j3 - i3min + 2 * sizez_ext];
                           lda = working_space[i1 + j1][i2 + j2][i3 + j3 + 3 * sizez_ext];
                           ldb = ldb + lda * ldc;
                        }
                     }
                  }
                  lda = working_space[i1][i2][i3 + 3 * sizez_ext];
                  ldc = working_space[i1][i2][i3 + 1 * sizez_ext];
                  if (ldc * lda != 0 && ldb != 0) {
                     lda = lda * ldc / ldb;
                  }

                  else
                     lda = 0;
                  working_space[i1][i2][i3 + 4 * sizez_ext] = lda;
               }
            }
         }
      }
      for (i3 = 0; i3 < sizez_ext; i3++) {
         for (i2 = 0; i2 < sizey_ext; i2++) {
            for (i1 = 0; i1 < sizex_ext; i1++)
               working_space[i1][i2][i3 + 3 * sizez_ext] = working_space[i1][i2][i3 + 4 * sizez_ext];
         }
      }
   }
//write back resulting spectrum
   maximum=0;
  for(i = 0;i < sizex_ext; i++){
      for(j = 0;j < sizey_ext; j++){
         for(k = 0;k < sizez_ext; k++){
            working_space[(i + positx) % sizex_ext][(j + posity) % sizey_ext][(k + positz) % sizez_ext] = area * working_space[i][j][k + 3 * sizez_ext];
            if(maximum < working_space[(i + positx) % sizex_ext][(j + posity) % sizey_ext][(k + positz) % sizez_ext])
               maximum = working_space[(i + positx) % sizex_ext][(j + posity) % sizey_ext][(k + positz) % sizez_ext];
         }
      }
   }
//searching for peaks in deconvolved spectrum
   for(i = 1;i < sizex_ext - 1; i++){
      for(j = 1;j < sizey_ext - 1; j++){
         for(l = 1;l < sizez_ext - 1; l++){
            a = working_space[i][j][l];
            if(a > working_space[i][j][l - 1] && a > working_space[i - 1][j][l - 1] && a > working_space[i - 1][j - 1][l - 1] && a > working_space[i][j - 1][l - 1] && a > working_space[i + 1][j - 1][l - 1] && a > working_space[i + 1][j][l - 1] && a > working_space[i + 1][j + 1][l - 1] && a > working_space[i][j + 1][l - 1] && a > working_space[i - 1][j + 1][l - 1] && a > working_space[i - 1][j][l] && a > working_space[i - 1][j - 1][l] && a > working_space[i][j - 1][l] && a > working_space[i + 1][j - 1][l] && a > working_space[i + 1][j][l] && a > working_space[i + 1][j + 1][l] && a > working_space[i][j + 1][l] && a > working_space[i - 1][j + 1][l] && a > working_space[i][j][l + 1] && a > working_space[i - 1][j][l + 1] && a > working_space[i - 1][j - 1][l + 1] && a > working_space[i][j - 1][l + 1] && a > working_space[i + 1][j - 1][l + 1] && a > working_space[i + 1][j][l + 1] && a > working_space[i + 1][j + 1][l + 1] && a > working_space[i][j + 1][l + 1] && a > working_space[i - 1][j + 1][l + 1]){
               if(i >= shift && i < ssizex + shift && j >= shift && j < ssizey + shift && l >= shift && l < ssizez + shift){
                  if(working_space[i][j][l] > threshold * maximum / 100.0){
                     if(peak_index < fMaxPeaks){
                        for(k = i - 1,a = 0,b = 0;k <= i + 1; k++){
                           a += (Double_t)(k - shift) * working_space[k][j][l];
                           b += working_space[k][j][l];
                        }
                     fPositionX[peak_index] = a / b;
                        for(k = j - 1,a = 0,b = 0;k <= j + 1; k++){
                           a += (Double_t)(k - shift) * working_space[i][k][l];
                           b += working_space[i][k][l];
                        }
                        fPositionY[peak_index] = a / b;
                        for(k = l - 1,a = 0,b = 0;k <= l + 1; k++){
                           a += (Double_t)(k - shift) * working_space[i][j][k];
                           b += working_space[i][j][k];
                        }
                        fPositionZ[peak_index] = a / b;
                        peak_index += 1;
                     }
                  }
               }
            }
         }
      }
   }
   for(i = 0;i < ssizex; i++){
      for(j = 0;j < ssizey; j++){
         for(k = 0;k < ssizez; k++){
            dest[i][j][k] = working_space[i + shift][j + shift][k + shift];
         }
      }
   }
   k = (Int_t)(4 * sigma + 0.5);
   k = 4 * k;
   for(i = 0;i < ssizex + k; i++){
      for(j = 0;j < ssizey + k; j++)
         delete[] working_space[i][j];
      delete[] working_space[i];
   }
   delete[] working_space;
   fNPeaks = peak_index;
   return fNPeaks;
}

//______________________________________________________________________________
Int_t TSpectrum3::SearchFast(const Double_t***source, Double_t***dest, Int_t ssizex, Int_t ssizey, Int_t ssizez,
                                 Double_t sigma, Double_t threshold,
                                 Bool_t markov, Int_t averWindow)

{

/////////////////////////////////////////////////////////////////////////////
// THREE-DIMENSIONAL CLASSICAL PEAK SEARCH FUNCTION                        //
// This function searches for peaks in source spectrum using               //
//  the algorithm based on smoothed second differences.                    //
//                                                                         //
// Function parameters:                                                    //
// source-pointer to the matrix of source spectrum                         //
// ssizex-x length of source spectrum                                      //
// ssizey-y length of source spectrum                                      //
// ssizez-z length of source spectrum                                      //
// sigma-sigma of searched peaks, for details we refer to manual           //
// threshold-threshold value in % for selected peaks, peaks with           //
//                amplitude less than threshold*highest_peak/100           //
//                are ignored, see manual                                  //
//  markov-logical variable, if it is true, first the source spectrum      //
//             is replaced by new spectrum calculated using Markov         //
//             chains method.                                              //
// averWindow-averanging window of searched peaks, for details             //
//                  we refer to manual (applies only for Markov method)    //
/////////////////////////////////////////////////////////////////////////////

   Int_t i,j,k,l,li,lj,lk,lmin,lmax,xmin,xmax,ymin,ymax,zmin,zmax;
   Double_t maxch,plocha = 0,plocha_markov = 0;
   Double_t nom,nip,nim,sp,sm,spx,spy,smx,smy,spz,smz;
   Double_t norma,val,val1,val2,val3,val4,val5,val6,val7,val8,val9,val10,val11,val12,val13,val14,val15,val16,val17,val18,val19,val20,val21,val22,val23,val24,val25,val26;
   Double_t a,b,s,f,maximum;
   Int_t x,y,z,peak_index=0;
   Double_t p1,p2,p3,p4,p5,p6,p7,p8,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,r1,r2,r3,r4,r5,r6;
   Double_t pocet_sigma = 5;
   Int_t number_of_iterations=(Int_t)(4 * sigma + 0.5);
   Int_t sizex_ext=ssizex + 4 * number_of_iterations,sizey_ext = ssizey + 4 * number_of_iterations,sizez_ext = ssizez + 4 * number_of_iterations,shift = 2 * number_of_iterations;
   Double_t c[PEAK_WINDOW],s_f_ratio_peaks = 5;
   if(sigma < 1){
      Error("SearchFast", "Invalid sigma, must be greater than or equal to 1");
      return 0;
   }

   if(threshold<=0||threshold>=100){
      Error("SearchFast", "Invalid threshold, must be positive and less than 100");
      return 0;
   }

   j = (Int_t)(pocet_sigma*sigma+0.5);
   if (j >= PEAK_WINDOW / 2) {
      Error("SearchFast", "Too large sigma");
      return 0;
   }

   if (markov == true) {
      if (averWindow <= 0) {
         Error("SearchFast", "Averanging window must be positive");
         return 0;
      }
   }

   if(sizex_ext < 2 * number_of_iterations + 1 || sizey_ext < 2 * number_of_iterations + 1 || sizez_ext < 2 * number_of_iterations + 1){
      Error("SearchFast", "Too large clipping window");
      return 0;
   }

   i = (Int_t)(4 * sigma + 0.5);
   i = 4 * i;
   Double_t ***working_space = new Double_t** [ssizex + i];
   for(j = 0;j < ssizex + i; j++){
      working_space[j] = new Double_t* [ssizey + i];
      for(k = 0;k < ssizey + i; k++)
         working_space[j][k] = new Double_t [4 * (ssizez + i)];
   }

   for(k = 0;k < sizez_ext; k++){
      for(j = 0;j < sizey_ext; j++){
        for(i = 0;i < sizex_ext; i++){
            if(i < shift){
               if(j < shift){
                  if(k < shift)
                     working_space[i][j][k + sizez_ext] = source[0][0][0];

                  else if(k >= ssizez + shift)
                     working_space[i][j][k + sizez_ext] = source[0][0][ssizez - 1];

                  else
                     working_space[i][j][k + sizez_ext] = source[0][0][k - shift];
               }

               else if(j >= ssizey + shift){
                  if(k < shift)
                     working_space[i][j][k + sizez_ext] = source[0][ssizey - 1][0];

                  else if(k >= ssizez + shift)
                     working_space[i][j][k + sizez_ext] = source[0][ssizey - 1][ssizez - 1];

                  else
                     working_space[i][j][k + sizez_ext] = source[0][ssizey - 1][k - shift];
               }

               else{
                  if(k < shift)
                     working_space[i][j][k + sizez_ext] = source[0][j - shift][0];

                  else if(k >= ssizez + shift)
                     working_space[i][j][k + sizez_ext] = source[0][j - shift][ssizez - 1];

                  else
                     working_space[i][j][k + sizez_ext] = source[0][j - shift][k - shift];
               }
            }

            else if(i >= ssizex + shift){
               if(j < shift){
                  if(k < shift)
                     working_space[i][j][k + sizez_ext] = source[ssizex - 1][0][0];

                  else if(k >= ssizez + shift)
                     working_space[i][j][k + sizez_ext] = source[ssizex - 1][0][ssizez - 1];

                  else
                     working_space[i][j][k + sizez_ext] = source[ssizex - 1][0][k - shift];
               }

               else if(j >= ssizey + shift){
                  if(k < shift)
                     working_space[i][j][k + sizez_ext] = source[ssizex - 1][ssizey - 1][0];

                  else if(k >= ssizez + shift)
                     working_space[i][j][k + sizez_ext] = source[ssizex - 1][ssizey - 1][ssizez - 1];

                  else
                     working_space[i][j][k + sizez_ext] = source[ssizex - 1][ssizey - 1][k - shift];
               }

               else{
                  if(k < shift)
                     working_space[i][j][k + sizez_ext] = source[ssizex - 1][j - shift][0];

                  else if(k >= ssizez + shift)
                     working_space[i][j][k + sizez_ext] = source[ssizex - 1][j - shift][ssizez - 1];

                  else
                     working_space[i][j][k + sizez_ext] = source[ssizex - 1][j - shift][k - shift];
               }
            }

            else{
               if(j < shift){
                  if(k < shift)
                     working_space[i][j][k + sizez_ext] = source[i - shift][0][0];

                  else if(k >= ssizez + shift)
                     working_space[i][j][k + sizez_ext] = source[i - shift][0][ssizez - 1];

                  else
                     working_space[i][j][k + sizez_ext] = source[i - shift][0][k - shift];
               }

               else if(j >= ssizey + shift){
                  if(k < shift)
                     working_space[i][j][k + sizez_ext] = source[i - shift][ssizey - 1][0];

                  else if(k >= ssizez + shift)
                     working_space[i][j][k + sizez_ext] = source[i - shift][ssizey - 1][ssizez - 1];

                  else
                     working_space[i][j][k + sizez_ext] = source[i - shift][ssizey - 1][k - shift];
               }

               else{
                  if(k < shift)
                     working_space[i][j][k + sizez_ext] = source[i - shift][j - shift][0];

                  else if(k >= ssizez + shift)
                     working_space[i][j][k + sizez_ext] = source[i - shift][j - shift][ssizez - 1];

                  else
                     working_space[i][j][k + sizez_ext] = source[i - shift][j - shift][k - shift];
               }
            }
         }
      }
   }
   for(i = 1;i <= number_of_iterations; i++){
      for(z = i;z < sizez_ext - i; z++){
         for(y = i;y < sizey_ext - i; y++){
            for(x = i;x < sizex_ext - i; x++){
               a = working_space[x][y][z + sizez_ext];
               p1 = working_space[x + i][y + i][z - i + sizez_ext];
               p2 = working_space[x - i][y + i][z - i + sizez_ext];
               p3 = working_space[x + i][y - i][z - i + sizez_ext];
               p4 = working_space[x - i][y - i][z - i + sizez_ext];
               p5 = working_space[x + i][y + i][z + i + sizez_ext];
               p6 = working_space[x - i][y + i][z + i + sizez_ext];
               p7 = working_space[x + i][y - i][z + i + sizez_ext];
               p8 = working_space[x - i][y - i][z + i + sizez_ext];
               s1 = working_space[x + i][y    ][z - i + sizez_ext];
               s2 = working_space[x    ][y + i][z - i + sizez_ext];
               s3 = working_space[x - i][y    ][z - i + sizez_ext];
               s4 = working_space[x    ][y - i][z - i + sizez_ext];
               s5 = working_space[x + i][y    ][z + i + sizez_ext];
               s6 = working_space[x    ][y + i][z + i + sizez_ext];
               s7 = working_space[x - i][y    ][z + i + sizez_ext];
               s8 = working_space[x    ][y - i][z + i + sizez_ext];
               s9 = working_space[x - i][y + i][z     + sizez_ext];
               s10 = working_space[x - i][y - i][z     +sizez_ext];
               s11 = working_space[x + i][y + i][z     +sizez_ext];
               s12 = working_space[x + i][y - i][z     +sizez_ext];
               r1 = working_space[x    ][y    ][z - i + sizez_ext];
               r2 = working_space[x    ][y    ][z + i + sizez_ext];
               r3 = working_space[x - i][y    ][z     + sizez_ext];
               r4 = working_space[x + i][y    ][z     + sizez_ext];
               r5 = working_space[x    ][y + i][z     + sizez_ext];
               r6 = working_space[x    ][y - i][z     + sizez_ext];
               b = (p1 + p3) / 2.0;
               if(b > s1)
                  s1 = b;

               b = (p1 + p2) / 2.0;
               if(b > s2)
                  s2 = b;

               b = (p2 + p4) / 2.0;
               if(b > s3)
                  s3 = b;

               b = (p3 + p4) / 2.0;
               if(b > s4)
                  s4 = b;

               b = (p5 + p7) / 2.0;
               if(b > s5)
                  s5 = b;

               b = (p5 + p6) / 2.0;
               if(b > s6)
                  s6 = b;

               b = (p6 + p8) / 2.0;
               if(b > s7)
                  s7 = b;

               b = (p7 + p8) / 2.0;
               if(b > s8)
                  s8 = b;

               b = (p2 + p6) / 2.0;
               if(b > s9)
                  s9 = b;

               b = (p4 + p8) / 2.0;
               if(b > s10)
                  s10 = b;

               b = (p1 + p5) / 2.0;
               if(b > s11)
                  s11 = b;

               b = (p3 + p7) / 2.0;
               if(b > s12)
                  s12 = b;

               s1 = s1 - (p1 + p3) / 2.0;
               s2 = s2 - (p1 + p2) / 2.0;
               s3 = s3 - (p2 + p4) / 2.0;
               s4 = s4 - (p3 + p4) / 2.0;
               s5 = s5 - (p5 + p7) / 2.0;
               s6 = s6 - (p5 + p6) / 2.0;
               s7 = s7 - (p6 + p8) / 2.0;
               s8 = s8 - (p7 + p8) / 2.0;
               s9 = s9 - (p2 + p6) / 2.0;
               s10 = s10 - (p4 + p8) / 2.0;
               s11 = s11 - (p1 + p5) / 2.0;
               s12 = s12 - (p3 + p7) / 2.0;
               b = (s1 + s3) / 2.0 + (s2 + s4) / 2.0 + (p1 + p2 + p3 + p4) / 4.0;
               if(b > r1)
                  r1 = b;

               b = (s5 + s7) / 2.0 + (s6 + s8) / 2.0 + (p5 + p6 + p7 + p8) / 4.0;
               if(b > r2)
                  r2 = b;

               b = (s3 + s7) / 2.0 + (s9 + s10) / 2.0 + (p2 + p4 + p6 + p8) / 4.0;
               if(b > r3)
                  r3 = b;

               b = (s1 + s5) / 2.0 + (s11 + s12) / 2.0 + (p1 + p3 + p5 + p7) / 4.0;
               if(b > r4)
                  r4 = b;

               b = (s9 + s11) / 2.0 + (s2 + s6) / 2.0 + (p1 + p2 + p5 + p6) / 4.0;
               if(b > r5)
                  r5 = b;

               b = (s4 + s8) / 2.0 + (s10 + s12) / 2.0 + (p3 + p4 + p7 + p8) / 4.0;
               if(b > r6)
                  r6 = b;

               r1 = r1 - ((s1 + s3) / 2.0 + (s2 + s4) / 2.0 + (p1 + p2 + p3 + p4) / 4.0);
               r2 = r2 - ((s5 + s7) / 2.0 + (s6 + s8) / 2.0 + (p5 + p6 + p7 + p8) / 4.0);
               r3 = r3 - ((s3 + s7) / 2.0 + (s9 + s10) / 2.0 + (p2 + p4 + p6 + p8) / 4.0);
               r4 = r4 - ((s1 + s5) / 2.0 + (s11 + s12) / 2.0 + (p1 + p3 + p5 + p7) / 4.0);
               r5 = r5 - ((s9 + s11) / 2.0 + (s2 + s6) / 2.0 + (p1 + p2 + p5 + p6) / 4.0);
               r6 = r6 - ((s4 + s8) / 2.0 + (s10 + s12) / 2.0 + (p3 + p4 + p7 + p8) / 4.0);
               b = (r1 + r2) / 2.0 + (r3 + r4) / 2.0 + (r5 + r6) / 2.0 + (s1 + s3 + s5 + s7) / 4.0 + (s2 + s4 + s6 + s8) / 4.0 + (s9 + s10 + s11 + s12) / 4.0 + (p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8) / 8.0;
               if(b < a)
                  a = b;

               working_space[x][y][z] = a;
            }
         }
      }
      for(z = i;z < sizez_ext - i; z++){
         for(y = i;y < sizey_ext - i; y++){
            for(x = i;x < sizex_ext - i; x++){
               working_space[x][y][z + sizez_ext] = working_space[x][y][z];
            }
         }
      }
   }
   for(k = 0;k < sizez_ext; k++){
      for(j = 0;j < sizey_ext; j++){
         for(i = 0;i < sizex_ext; i++){
            if(i < shift){
               if(j < shift){
                  if(k < shift)
                     working_space[i][j][k + 3 * sizez_ext] = source[0][0][0] - working_space[i][j][k + sizez_ext];

                  else if(k >= ssizez + shift)
                     working_space[i][j][k + 3 * sizez_ext] = source[0][0][ssizez - 1] - working_space[i][j][k + sizez_ext];

                  else
                     working_space[i][j][k + 3 * sizez_ext] = source[0][0][k - shift] - working_space[i][j][k + sizez_ext];
               }

               else if(j >= ssizey + shift){
                  if(k < shift)
                     working_space[i][j][k + 3 * sizez_ext] = source[0][ssizey - 1][0] - working_space[i][j][k + sizez_ext];

                  else if(k >= ssizez + shift)
                     working_space[i][j][k + 3 * sizez_ext] = source[0][ssizey - 1][ssizez - 1] - working_space[i][j][k + sizez_ext];

                  else
                     working_space[i][j][k + 3 * sizez_ext] = source[0][ssizey - 1][k - shift] - working_space[i][j][k + sizez_ext];
               }

               else{
                  if(k < shift)
                     working_space[i][j][k + 3 * sizez_ext] = source[0][j - shift][0] - working_space[i][j][k + sizez_ext];

                  else if(k >= ssizez + shift)
                     working_space[i][j][k + 3 * sizez_ext] = source[0][j - shift][ssizez - 1] - working_space[i][j][k + sizez_ext];

                  else
                     working_space[i][j][k + 3 * sizez_ext] = source[0][j - shift][k - shift] - working_space[i][j][k + sizez_ext];
               }
            }

            else if(i >= ssizex + shift){
               if(j < shift){
                  if(k < shift)
                     working_space[i][j][k + 3 * sizez_ext] = source[ssizex - 1][0][0] - working_space[i][j][k + sizez_ext];

                  else if(k >= ssizez + shift)
                     working_space[i][j][k + 3 * sizez_ext] = source[ssizex - 1][0][ssizez - 1] - working_space[i][j][k + sizez_ext];

                  else
                     working_space[i][j][k + 3 * sizez_ext] = source[ssizex - 1][0][k - shift] - working_space[i][j][k + sizez_ext];
               }

               else if(j >= ssizey + shift){
                  if(k < shift)
                     working_space[i][j][k + 3 * sizez_ext] = source[ssizex - 1][ssizey - 1][0] - working_space[i][j][k + sizez_ext];

                  else if(k >= ssizez + shift)
                     working_space[i][j][k + 3 * sizez_ext] = source[ssizex - 1][ssizey - 1][ssizez - 1] - working_space[i][j][k + sizez_ext];

                  else
                     working_space[i][j][k + 3 * sizez_ext] = source[ssizex - 1][ssizey - 1][k - shift] - working_space[i][j][k + sizez_ext];
               }

               else{
                  if(k < shift)
                     working_space[i][j][k + 3 * sizez_ext] = source[ssizex - 1][j - shift][0] - working_space[i][j][k + sizez_ext];

                  else if(k >= ssizez + shift)
                     working_space[i][j][k + 3 * sizez_ext] = source[ssizex - 1][j - shift][ssizez - 1] - working_space[i][j][k + sizez_ext];

                  else
                     working_space[i][j][k + 3 * sizez_ext] = source[ssizex - 1][j - shift][k - shift] - working_space[i][j][k + sizez_ext];
               }
            }

            else{
               if(j < shift){
                  if(k < shift)
                     working_space[i][j][k + 3 * sizez_ext] = source[i - shift][0][0] - working_space[i][j][k + sizez_ext];

                  else if(k >= ssizez + shift)
                     working_space[i][j][k + 3 * sizez_ext] = source[i - shift][0][ssizez - 1] - working_space[i][j][k + sizez_ext];

                  else
                     working_space[i][j][k + 3 * sizez_ext] = source[i - shift][0][k - shift] - working_space[i][j][k + sizez_ext];
               }

               else if(j >= ssizey + shift){
                  if(k < shift)
                     working_space[i][j][k + 3 * sizez_ext] = source[i - shift][ssizey - 1][0] - working_space[i][j][k + sizez_ext];

                  else if(k >= ssizez + shift)
                     working_space[i][j][k + 3 * sizez_ext] = source[i - shift][ssizey - 1][ssizez - 1] - working_space[i][j][k + sizez_ext];

                  else
                     working_space[i][j][k + 3 * sizez_ext] = source[i - shift][ssizey - 1][k - shift] - working_space[i][j][k + sizez_ext];
               }

               else{
                  if(k < shift)
                     working_space[i][j][k + 3 * sizez_ext] = source[i - shift][j - shift][0] - working_space[i][j][k + sizez_ext];

                  else if(k >= ssizez + shift)
                     working_space[i][j][k + 3 * sizez_ext] = source[i - shift][j - shift][ssizez - 1] - working_space[i][j][k + sizez_ext];

                  else
                     working_space[i][j][k + 3 * sizez_ext] = source[i - shift][j - shift][k - shift] - working_space[i][j][k + sizez_ext];
               }
            }
         }
      }
   }

   for(i = 0;i < sizex_ext; i++){
      for(j = 0;j < sizey_ext; j++){
         for(k = 0;k < sizez_ext; k++){
            if(i >= shift && i < ssizex + shift && j >= shift && j < ssizey + shift && k >= shift && k < ssizez + shift){
               working_space[i][j][k + 2 * sizez_ext] = source[i - shift][j - shift][k - shift];
               plocha_markov = plocha_markov + source[i - shift][j - shift][k - shift];
            }
            else
               working_space[i][j][k + 2 * sizez_ext] = 0;
         }
      }
   }

   if(markov == true){
      xmin = 0;
      xmax = sizex_ext - 1;
      ymin = 0;
      ymax = sizey_ext - 1;
      zmin = 0;
      zmax = sizez_ext - 1;
      for(i = 0,maxch = 0;i < sizex_ext; i++){
         for(j = 0;j < sizey_ext;j++){
            for(k = 0;k < sizez_ext;k++){
               working_space[i][j][k] = 0;
               if(maxch < working_space[i][j][k + 2 * sizez_ext])
                  maxch = working_space[i][j][k + 2 * sizez_ext];

               plocha += working_space[i][j][k + 2 * sizez_ext];
            }
         }
      }
      if(maxch == 0) {
         delete [] working_space;
         return 0;
      }

      nom = 0;
      working_space[xmin][ymin][zmin] = 1;
      for(i = xmin;i < xmax; i++){
         nip = working_space[i][ymin][zmin + 2 * sizez_ext] / maxch;
         nim = working_space[i + 1][ymin][zmin + 2 * sizez_ext] / maxch;
         sp = 0,sm = 0;
         for(l = 1;l <= averWindow; l++){
            if((i + l) > xmax)
               a = working_space[xmax][ymin][zmin + 2 * sizez_ext] / maxch;

            else
               a = working_space[i + l][ymin][zmin + 2 * sizez_ext] / maxch;

            b = a - nip;
            if(a + nip <= 0)
               a = 1;

            else
               a = TMath::Sqrt(a + nip);

            b = b / a;
            b = TMath::Exp(b);
            sp = sp + b;
            if(i - l + 1 < xmin)
               a = working_space[xmin][ymin][zmin + 2 * sizez_ext] / maxch;

            else
               a = working_space[i - l + 1][ymin][zmin + 2 * sizez_ext] / maxch;

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
         a = working_space[i + 1][ymin][zmin] = a * working_space[i][ymin][zmin];
         nom = nom + a;
      }
      for(i = ymin;i < ymax; i++){
         nip = working_space[xmin][i][zmin + 2 * sizez_ext] / maxch;
         nim = working_space[xmin][i + 1][zmin + 2 * sizez_ext] / maxch;
         sp = 0,sm = 0;
         for(l = 1;l <= averWindow; l++){
            if((i + l) > ymax)
               a = working_space[xmin][ymax][zmin + 2 * sizez_ext] / maxch;

            else
               a = working_space[xmin][i + l][zmin + 2 * sizez_ext] / maxch;

            b = a - nip;
            if(a + nip <= 0)
               a = 1;

            else
               a = TMath::Sqrt(a + nip);

            b = b / a;
            b = TMath::Exp(b);
            sp = sp + b;
            if(i - l + 1 < ymin)
               a = working_space[xmin][ymin][zmin + 2 * sizez_ext] / maxch;

            else
               a = working_space[xmin][i - l + 1][zmin + 2 * sizez_ext] / maxch;

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
         a = working_space[xmin][i + 1][zmin] = a * working_space[xmin][i][zmin];
         nom = nom + a;
      }
      for(i = zmin;i < zmax;i++){
         nip = working_space[xmin][ymin][i + 2 * sizez_ext] / maxch;
         nim = working_space[xmin][ymin][i + 1 + 2 * sizez_ext] / maxch;
         sp = 0,sm = 0;
         for(l = 1;l <= averWindow;l++){
            if((i + l) > zmax)
               a = working_space[xmin][ymin][zmax + 2 * sizez_ext] / maxch;

            else
               a = working_space[xmin][ymin][i + l + 2 * sizez_ext] / maxch;

            b = a - nip;
            if(a + nip <= 0)
               a = 1;

            else
               a = TMath::Sqrt(a + nip);

            b = b / a;
            b = TMath::Exp(b);
            sp = sp + b;
            if(i - l + 1 < zmin)
               a = working_space[xmin][ymin][zmin + 2 * sizez_ext] / maxch;

            else
               a = working_space[xmin][ymin][i - l + 1 + 2 * sizez_ext] / maxch;

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
         a = working_space[xmin][ymin][i + 1] = a * working_space[xmin][ymin][i];
         nom = nom + a;
      }
      for(i = xmin;i < xmax; i++){
         for(j = ymin;j < ymax; j++){
            nip = working_space[i][j + 1][zmin + 2 * sizez_ext] / maxch;
            nim = working_space[i + 1][j + 1][zmin + 2 * sizez_ext] / maxch;
            spx = 0,smx = 0;
            for(l = 1;l <= averWindow; l++){
               if(i + l > xmax)
                 a = working_space[xmax][j][zmin + 2 * sizez_ext] / maxch;

               else
                  a = working_space[i + l][j][zmin + 2 * sizez_ext] / maxch;

               b = a - nip;
               if(a + nip <= 0)
                  a = 1;

               else
                  a = TMath::Sqrt(a + nip);

               b = b / a;
               b = TMath::Exp(b);
               spx = spx + b;
               if(i - l + 1 < xmin)
                  a = working_space[xmin][j][zmin + 2 * sizez_ext] / maxch;

               else
                  a = working_space[i - l + 1][j][zmin + 2 * sizez_ext] / maxch;

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
            nip = working_space[i + 1][j][zmin + 2 * sizez_ext] / maxch;
            nim = working_space[i + 1][j + 1][zmin + 2 * sizez_ext] / maxch;
            for(l = 1;l <= averWindow; l++){
               if(j + l > ymax)
                  a = working_space[i][ymax][zmin + 2 * sizez_ext] / maxch;

               else
                  a = working_space[i][j + l][zmin + 2 * sizez_ext] / maxch;

               b = a - nip;
               if(a + nip <= 0)
                  a = 1;

               else
                  a = TMath::Sqrt(a + nip);

               b = b / a;
               b = TMath::Exp(b);
               spy = spy + b;
               if(j - l + 1 < ymin)
                  a = working_space[i][ymin][zmin + 2 * sizez_ext] / maxch;

               else
                  a = working_space[i][j - l + 1][zmin + 2 * sizez_ext] / maxch;

               b = a - nim;
               if(a + nim <= 0)
                  a = 1;

               else
                  a = TMath::Sqrt(a + nim);

               b = b / a;
               b = TMath::Exp(b);
               smy = smy + b;
            }
            a = (spx * working_space[i][j + 1][zmin] + spy * working_space[i + 1][j][zmin]) / (smx + smy);
            working_space[i + 1][j + 1][zmin] = a;
            nom = nom + a;
         }
      }
      for(i = xmin;i < xmax;i++){
         for(j = zmin;j < zmax;j++){
            nip = working_space[i][ymin][j + 1 + 2 * sizez_ext] / maxch;
            nim = working_space[i + 1][ymin][j + 1 + 2 * sizez_ext] / maxch;
            spx = 0,smx = 0;
            for(l = 1;l <= averWindow; l++){
               if(i + l > xmax)
                  a = working_space[xmax][ymin][j + 2 * sizez_ext] / maxch;

               else
                  a = working_space[i + l][ymin][j + 2 * sizez_ext] / maxch;

               b = a - nip;
               if(a + nip <= 0)
                  a = 1;

               else
                  a = TMath::Sqrt(a + nip);

               b = b / a;
               b = TMath::Exp(b);
               spx = spx + b;
               if(i - l + 1 < xmin)
                  a = working_space[xmin][ymin][j + 2 * sizez_ext] / maxch;

               else
                  a = working_space[i - l + 1][ymin][j + 2 * sizez_ext] / maxch;

               b = a - nim;
               if(a + nim <= 0)
                  a = 1;

               else
                  a = TMath::Sqrt(a + nim);

               b = b / a;
               b = TMath::Exp(b);
               smx = smx + b;
            }
            spz = 0,smz = 0;
            nip = working_space[i + 1][ymin][j + 2 * sizez_ext] / maxch;
            nim = working_space[i + 1][ymin][j + 1 + 2 * sizez_ext] / maxch;
            for(l = 1;l <= averWindow; l++){
               if(j + l > zmax)
                  a = working_space[i][ymin][zmax + 2 * sizez_ext] / maxch;

               else
                  a = working_space[i][ymin][j + l + 2 * sizez_ext] / maxch;

               b = a - nip;
               if(a + nip <= 0)
                  a = 1;

               else
                  a = TMath::Sqrt(a + nip);

               b = b / a;
               b = TMath::Exp(b);
               spz = spz + b;
               if(j - l + 1 < zmin)
                  a = working_space[i][ymin][zmin + 2 * sizez_ext] / maxch;

               else
                  a = working_space[i][ymin][j - l + 1 + 2 * sizez_ext] / maxch;

               b = a - nim;
               if(a + nim <= 0)
                  a = 1;

               else
                  a = TMath::Sqrt(a + nim);

               b = b / a;
               b = TMath::Exp(b);
               smz = smz + b;
            }
            a = (spx * working_space[i][ymin][j + 1] + spz * working_space[i + 1][ymin][j]) / (smx + smz);
            working_space[i + 1][ymin][j + 1] = a;
            nom = nom + a;
         }
      }
      for(i = ymin;i < ymax;i++){
         for(j = zmin;j < zmax;j++){
            nip = working_space[xmin][i][j + 1 + 2 * sizez_ext] / maxch;
            nim = working_space[xmin][i + 1][j + 1 + 2 * sizez_ext] / maxch;
            spy = 0,smy = 0;
            for(l = 1;l <= averWindow; l++){
               if(i + l > ymax)
                  a = working_space[xmin][ymax][j + 2 * sizez_ext] / maxch;

               else
                  a = working_space[xmin][i + l][j + 2 * sizez_ext] / maxch;

               b = a - nip;
               if(a + nip <= 0)
                  a = 1;

               else
                  a = TMath::Sqrt(a + nip);

               b = b / a;
               b = TMath::Exp(b);
               spy = spy + b;
               if(i - l + 1 < ymin)
                  a = working_space[xmin][ymin][j + 2 * sizez_ext] / maxch;

               else
                  a = working_space[xmin][i - l + 1][j + 2 * sizez_ext] / maxch;

               b = a - nim;
               if(a + nim <= 0)
                  a = 1;

               else
                  a = TMath::Sqrt(a + nim);

               b = b / a;
               b = TMath::Exp(b);
               smy = smy + b;
            }
            spz = 0,smz = 0;
            nip = working_space[xmin][i + 1][j + 2 * sizez_ext] / maxch;
            nim = working_space[xmin][i + 1][j + 1 + 2 * sizez_ext] / maxch;
            for(l = 1;l <= averWindow; l++){
               if(j + l > zmax)
                  a = working_space[xmin][i][zmax + 2 * sizez_ext] / maxch;

               else
                  a = working_space[xmin][i][j + l + 2 * sizez_ext] / maxch;

               b = a - nip;
               if(a + nip <= 0)
                  a = 1;

               else
                  a = TMath::Sqrt(a + nip);

               b = b / a;
               b = TMath::Exp(b);
               spz = spz + b;
               if(j - l + 1 < zmin)
                  a = working_space[xmin][i][zmin + 2 * sizez_ext] / maxch;

               else
                  a = working_space[xmin][i][j - l + 1 + 2 * sizez_ext] / maxch;

               b = a - nim;
               if(a + nim <= 0)
                  a = 1;

               else
                  a = TMath::Sqrt(a + nim);

               b = b / a;
               b = TMath::Exp(b);
               smz = smz + b;
            }
            a = (spy * working_space[xmin][i][j + 1] + spz * working_space[xmin][i + 1][j]) / (smy + smz);
            working_space[xmin][i + 1][j + 1] = a;
            nom = nom + a;
         }
      }
      for(i = xmin;i < xmax; i++){
         for(j = ymin;j < ymax; j++){
            for(k = zmin;k < zmax; k++){
               nip = working_space[i][j + 1][k + 1 + 2 * sizez_ext] / maxch;
               nim = working_space[i + 1][j + 1][k + 1 + 2 * sizez_ext] / maxch;
               spx = 0,smx = 0;
               for(l = 1;l <= averWindow; l++){
                  if(i + l > xmax)
                     a = working_space[xmax][j][k + 2 * sizez_ext] / maxch;

                  else
                     a = working_space[i + l][j][k + 2 * sizez_ext] / maxch;

                  b = a - nip;
                  if(a + nip <= 0)
                     a = 1;

                  else
                     a = TMath::Sqrt(a + nip);

                  b = b / a;
                  b = TMath::Exp(b);
                  spx = spx + b;
                  if(i - l + 1 < xmin)
                     a = working_space[xmin][j][k + 2 * sizez_ext] / maxch;

                  else
                     a = working_space[i - l + 1][j][k + 2 * sizez_ext] / maxch;

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
               nip = working_space[i + 1][j][k + 1 + 2 * sizez_ext] / maxch;
               nim = working_space[i + 1][j + 1][k + 1 + 2 * sizez_ext] / maxch;
               for(l = 1;l <= averWindow; l++){
                  if(j + l > ymax)
                     a = working_space[i][ymax][k + 2 * sizez_ext] / maxch;

                  else
                     a = working_space[i][j + l][k + 2 * sizez_ext] / maxch;

                  b = a - nip;
                  if(a + nip <= 0)
                     a = 1;

                  else
                     a = TMath::Sqrt(a + nip);

                  b = b / a;
                  b = TMath::Exp(b);
                  spy = spy + b;
                  if(j - l + 1 < ymin)
                     a = working_space[i][ymin][k + 2 * sizez_ext] / maxch;

                  else
                     a = working_space[i][j - l + 1][k + 2 * sizez_ext] / maxch;

                  b = a - nim;
                  if(a + nim <= 0)
                     a = 1;

                  else
                     a = TMath::Sqrt(a + nim);

                  b = b / a;
                  b = TMath::Exp(b);
                  smy = smy + b;
               }
               spz = 0,smz = 0;
               nip = working_space[i + 1][j + 1][k + 2 * sizez_ext] / maxch;
               nim = working_space[i + 1][j + 1][k + 1 + 2 * sizez_ext] / maxch;
               for(l = 1;l <= averWindow; l++ ){
                  if(j + l > zmax)
                     a = working_space[i][j][zmax + 2 * sizez_ext] / maxch;

                  else
                     a = working_space[i][j][k + l + 2 * sizez_ext] / maxch;

                  b = a - nip;
                  if(a + nip <= 0)
                     a = 1;

                  else
                     a = TMath::Sqrt(a + nip);

                  b = b / a;
                  b = TMath::Exp(b);
                  spz = spz + b;
                  if(j - l + 1 < ymin)
                     a = working_space[i][j][zmin + 2 * sizez_ext] / maxch;

                  else
                     a = working_space[i][j][k - l + 1 + 2 * sizez_ext] / maxch;

                  b = a - nim;
                  if(a + nim <= 0)
                     a = 1;

                  else
                     a = TMath::Sqrt(a + nim);

                  b = b / a;
                  b = TMath::Exp(b);
                  smz = smz + b;
               }
               a = (spx * working_space[i][j + 1][k + 1] + spy * working_space[i + 1][j][k + 1] + spz * working_space[i + 1][j + 1][k]) / (smx + smy + smz);
               working_space[i + 1][j + 1][k + 1] = a;
               nom = nom + a;
            }
         }
      }
      a = 0;
      for(i = xmin;i <= xmax; i++){
         for(j = ymin;j <= ymax; j++){
            for(k = zmin;k <= zmax; k++){
               working_space[i][j][k] = working_space[i][j][k] / nom;
               a+=working_space[i][j][k];
            }
         }
      }
      for(i = 0;i < sizex_ext; i++){
         for(j = 0;j < sizey_ext; j++){
            for(k = 0;k < sizez_ext; k++){
               working_space[i][j][k + 2 * sizez_ext] = working_space[i][j][k] * plocha_markov / a;
            }
         }
      }
   }

   maximum = 0;
   for(k = 0;k < ssizez; k++){
      for(j = 0;j < ssizey; j++){
         for(i = 0;i < ssizex; i++){
            working_space[i][j][k] = 0;
            working_space[i][j][sizez_ext + k] = 0;
            if(working_space[i][j][k + 3 * sizez_ext] > maximum)
               maximum=working_space[i][j][k+3*sizez_ext];
         }
      }
   }
   for(i = 0;i < PEAK_WINDOW; i++){
      c[i] = 0;
   }
   j = (Int_t)(pocet_sigma * sigma + 0.5);
   for(i = -j;i <= j; i++){
      a=i;
      a = -a * a;
      b = 2.0 * sigma * sigma;
      a = a / b;
      a = TMath::Exp(a);
      s = i;
      s = s * s;
      s = s - sigma * sigma;
      s = s / (sigma * sigma * sigma * sigma);
      s = s * a;
      c[PEAK_WINDOW / 2 + i] = s;
   }
   norma = 0;
   for(i = 0;i < PEAK_WINDOW; i++){
      norma = norma + TMath::Abs(c[i]);
   }
   for(i = 0;i < PEAK_WINDOW; i++){
      c[i] = c[i] / norma;
   }
   a = pocet_sigma * sigma + 0.5;
   i = (Int_t)a;
   zmin = i;
   zmax = sizez_ext - i - 1;
   ymin = i;
   ymax = sizey_ext - i - 1;
   xmin = i;
   xmax = sizex_ext - i - 1;
   lmin = PEAK_WINDOW / 2 - i;
   lmax = PEAK_WINDOW / 2 + i;
   for(i = xmin;i <= xmax; i++){
      for(j = ymin;j <= ymax; j++){
         for(k = zmin;k <= zmax; k++){
            s = 0,f = 0;
            for(li = lmin;li <= lmax; li++){
               for(lj = lmin;lj <= lmax; lj++){
                  for(lk = lmin;lk <= lmax; lk++){
                     a = working_space[i + li - PEAK_WINDOW / 2][j + lj - PEAK_WINDOW / 2][k + lk - PEAK_WINDOW / 2 + 2 * sizez_ext];
                     b = c[li] * c[lj] * c[lk];
                     s += a * b;
                     f += a * b * b;
                  }
               }
            }
            working_space[i][j][k] = s;
            working_space[i][j][k + sizez_ext] = TMath::Sqrt(f);
         }
      }
   }
   for(x = xmin;x <= xmax; x++){
      for(y = ymin + 1;y < ymax; y++){
         for(z = zmin + 1;z < zmax; z++){
            val = working_space[x][y][z];
            val1 =  working_space[x - 1][y - 1][z - 1];
            val2 =  working_space[x    ][y - 1][z - 1];
            val3 =  working_space[x + 1][y - 1][z - 1];
            val4 =  working_space[x - 1][y    ][z - 1];
            val5 =  working_space[x    ][y    ][z - 1];
            val6 =  working_space[x + 1][y    ][z - 1];
            val7 =  working_space[x - 1][y + 1][z - 1];
            val8 =  working_space[x    ][y + 1][z - 1];
            val9 =  working_space[x + 1][y + 1][z - 1];
            val10 = working_space[x - 1][y - 1][z    ];
            val11 = working_space[x    ][y - 1][z    ];
            val12 = working_space[x + 1][y - 1][z    ];
            val13 = working_space[x - 1][y    ][z    ];
            val14 = working_space[x + 1][y    ][z    ];
            val15 = working_space[x - 1][y + 1][z    ];
            val16 = working_space[x    ][y + 1][z    ];
            val17 = working_space[x + 1][y + 1][z    ];
            val18 = working_space[x - 1][y - 1][z + 1];
            val19 = working_space[x    ][y - 1][z + 1];
            val20 = working_space[x + 1][y - 1][z + 1];
            val21 = working_space[x - 1][y    ][z + 1];
            val22 = working_space[x    ][y    ][z + 1];
            val23 = working_space[x + 1][y    ][z + 1];
            val24 = working_space[x - 1][y + 1][z + 1];
            val25 = working_space[x    ][y + 1][z + 1];
            val26 = working_space[x + 1][y + 1][z + 1];
            f = -s_f_ratio_peaks * working_space[x][y][z + sizez_ext];
            if(val < f && val < val1 && val < val2 && val < val3 && val < val4 && val < val5 && val < val6 && val < val7 && val < val8 && val < val9 && val < val10 && val < val11 && val < val12 && val < val13 && val < val14 && val < val15 && val < val16 && val < val17 && val < val18 && val < val19 && val < val20 && val < val21 && val < val22 && val < val23 && val < val24 && val < val25 && val < val26){
               s=0,f=0;
            for(li = lmin;li <= lmax; li++){
               a = working_space[x + li - PEAK_WINDOW / 2][y][z + 2 * sizez_ext];
               s += a * c[li];
               f += a * c[li] * c[li];
            }
            f = -s_f_ratio_peaks * sqrt(f);
            if(s < f){
               s = 0,f = 0;
               for(li = lmin;li <= lmax; li++){
                  a = working_space[x][y + li - PEAK_WINDOW / 2][z + 2 * sizez_ext];
                  s += a * c[li];
                  f += a * c[li] * c[li];
               }
               f = -s_f_ratio_peaks * sqrt(f);
               if(s < f){
                  s = 0,f = 0;
                  for(li = lmin;li <= lmax; li++){
                     a = working_space[x][y][z + li - PEAK_WINDOW / 2 + 2 * sizez_ext];
                     s += a * c[li];
                     f += a * c[li] * c[li];
                  }
                  f = -s_f_ratio_peaks * sqrt(f);
                  if(s < f){
                     s = 0,f = 0;
                     for(li = lmin;li <= lmax; li++){
                        for(lj = lmin;lj <= lmax; lj++){
                           a = working_space[x + li - PEAK_WINDOW / 2][y + lj - PEAK_WINDOW / 2][z + 2 * sizez_ext];
                           s += a * c[li] * c[lj];
                           f += a * c[li] * c[li] * c[lj] * c[lj];
                        }
                     }
                     f = s_f_ratio_peaks * sqrt(f);
                     if(s > f){
                        s = 0,f = 0;
                        for(li = lmin;li <= lmax; li++){
                           for(lj = lmin;lj <= lmax; lj++){
                              a = working_space[x + li - PEAK_WINDOW / 2][y][z + lj - PEAK_WINDOW / 2 + 2 * sizez_ext];
                              s += a * c[li] * c[lj];
                              f += a * c[li] * c[li] * c[lj] * c[lj];
                           }
                        }
                        f = s_f_ratio_peaks * sqrt(f);
                        if(s > f){
                           s = 0,f = 0;
                           for(li = lmin;li <= lmax; li++){
                              for(lj=lmin;lj<=lmax;lj++){
                                 a = working_space[x][y + li - PEAK_WINDOW / 2][z + lj - PEAK_WINDOW / 2 + 2 * sizez_ext];
                                 s += a * c[li] * c[lj];
                                 f += a * c[li] * c[li] * c[lj] * c[lj];
                              }
                           }
                           f = s_f_ratio_peaks * sqrt(f);
                              if(s > f){
                                 if(x >= shift && x < ssizex + shift && y >= shift && y < ssizey + shift && z >= shift && z < ssizez + shift){
                                    if(working_space[x][y][z + 3 * sizez_ext] > threshold * maximum / 100.0){
                                       if(peak_index<fMaxPeaks){
                                          for(k = x - 1,a = 0,b = 0;k <= x + 1; k++){
                                             a += (Double_t)(k - shift) * working_space[k][y][z];
                                             b += working_space[k][y][z];
                                          }
                                          fPositionX[peak_index] = a / b;
                                          for(k = y - 1,a = 0,b = 0;k <= y + 1; k++){
                                             a += (Double_t)(k - shift) * working_space[x][k][z];
                                             b += working_space[x][k][z];
                                          }
                                          fPositionY[peak_index] = a / b;
                                          for(k = z - 1,a = 0,b = 0;k <= z + 1; k++){
                                             a += (Double_t)(k - shift) * working_space[x][y][k];
                                             b += working_space[x][y][k];
                                          }
                                          fPositionZ[peak_index] = a / b;
                                          peak_index += 1;
                                       }
                                    }
                                 }
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }
   for(i = 0;i < ssizex; i++){
      for(j = 0;j < ssizey; j++){
         for(k = 0;k < ssizez; k++){
            val = -working_space[i + shift][j + shift][k + shift];
            if( val < 0)
               val = 0;
            dest[i][j][k] = val;
         }
      }
   }
   k = (Int_t)(4 * sigma + 0.5);
   k = 4 * k;
   for(i = 0;i < ssizex + k; i++){
      for(j = 0;j < ssizey + k; j++)
         delete[] working_space[i][j];
      delete[] working_space[i];
   }
   delete[] working_space;
   fNPeaks = peak_index;
   return fNPeaks;
}
