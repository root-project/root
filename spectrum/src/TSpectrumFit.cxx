// @(#)root/spectrum:$Name:  $:$Id: TSpectrumFit.cxx,v 1.1 2006/09/28 19:19:52 brun Exp $
// Author: Miroslav Morhac   25/09/06

//__________________________________________________________________________
//   THIS CLASS CONTAINS ADVANCED SPECTRA FITTING FUNCTIONS.               //
//                                                                         //
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
//   [1] M. Morhac et al.: Efficient fitting algorithms applied to         //
//   analysis of coincidence gamma-ray spectra. Computer Physics           //
//   Communications, Vol 172/1 (2005) pp. 19-41.                           //
//                                                                         //
//   [2]  M. Morhac et al.: Study of fitting algorithms applied to         //
//   simultaneous analysis of large number of peaks in gamma-ray spectra.  //
//   Applied Spectroscopy, Vol. 57, No. 7, pp. 753-760, 2003.              //
//                                                                         //
//                                                                         //
//____________________________________________________________________________
    
#include "TSpectrumFit.h"
#include "TMath.h"

ClassImp(TSpectrumFit)  
    
//______________________________________________________________________________
TSpectrumFit::TSpectrumFit() :TNamed("SpectrumFit", "Miroslav Morhac peak fitter") 
{
   //default constructor
   fNPeaks = 0;   
   fPositionInit  = 0;
   fPositionCalc  = 0;   
   fPositionErr   = 0;   
   fFixPosition   = 0;   
   fAmpInit   = 0;   
   fAmpCalc   = 0;   
   fAmpErr    = 0;   
   fFixAmp    = 0;     
   fArea      = 0;   
   fAreaErr   = 0;      
}

//______________________________________________________________________________
TSpectrumFit::TSpectrumFit(Int_t numberPeaks) :TNamed("SpectrumFit", "Miroslav Morhac peak fitter") 
{
   //numberPeaks: number of fitted peaks (must be greater than zero)
   //the constructor allocates arrays for all fitted parameters (peak positions, amplitudes etc) and sets the member
   //variables to their default values. One can change these variables by member functions (setters) of TSpectrumFit class.
//Begin_Html <!--
/* -->
   
<div class=3DSection1>

<p class=3DMsoNormal style=3D'text-align:justify;tab-stops:.75in'><span lan=
g=3DEN-US>Shape
function of the fitted peaks is </span></p>

<p class=3DMsoNormal style=3D'text-align:justify;tab-stops:.75in'><!--[if g=
te vml 1]><v:shapetype
 id=3D"_x0000_t75" coordsize=3D"21600,21600" o:spt=3D"75" o:preferrelative=
=3D"t"
 path=3D"m@4@5l@4@11@9@11@9@5xe" filled=3D"f" stroked=3D"f">
 <v:stroke joinstyle=3D"miter"/>
 <v:formulas>
  <v:f eqn=3D"if lineDrawn pixelLineWidth 0"/>
  <v:f eqn=3D"sum @0 1 0"/>
  <v:f eqn=3D"sum 0 0 @1"/>
  <v:f eqn=3D"prod @2 1 2"/>
  <v:f eqn=3D"prod @3 21600 pixelWidth"/>
  <v:f eqn=3D"prod @3 21600 pixelHeight"/>
  <v:f eqn=3D"sum @0 0 1"/>
  <v:f eqn=3D"prod @6 1 2"/>
  <v:f eqn=3D"prod @7 21600 pixelWidth"/>
  <v:f eqn=3D"sum @8 21600 0"/>
  <v:f eqn=3D"prod @7 21600 pixelHeight"/>
  <v:f eqn=3D"sum @10 21600 0"/>
 </v:formulas>
 <v:path o:extrusionok=3D"f" gradientshapeok=3D"t" o:connecttype=3D"rect"/>
 <o:lock v:ext=3D"edit" aspectratio=3D"t"/>
</v:shapetype><v:shape id=3D"_x0000_s1037" type=3D"#_x0000_t75" style=3D'po=
sition:absolute;
 left:0;text-align:left;margin-left:51.3pt;margin-top:4.2pt;width:290.7pt;
 height:99.2pt;z-index:1' fillcolor=3D"window">
 <v:imagedata src=3D"FitConstructor_files/image001.wmz" o:title=3D""/>
</v:shape><![if gte mso 9]><o:OLEObject Type=3D"Embed" ProgID=3D"Equation.D=
SMT4"
 ShapeID=3D"_x0000_s1037" DrawAspect=3D"Content" ObjectID=3D"_1220794183">
</o:OLEObject>
<![endif]><![endif]--><![if !vml]><span style=3D'mso-ignore:vglayout'>

<table cellpadding=3D0 cellspacing=3D0 align=3Dleft>
 <tr>
  <td width=3D68 height=3D6></td>
 </tr>
 <tr>
  <td></td>
  <td><img width=3D388 height=3D132 src=3D"FitConstructor_files/image002.gi=
f" v:shapes=3D"_x0000_s1037"></td>
 </tr>
</table>

</span><![endif]><span lang=3DEN-US style=3D'font-family:Arial;mso-bidi-fon=
t-family:
"Times New Roman"'><o:p>&nbsp;</o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><o:p>&=
nbsp;</o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><o:p>&=
nbsp;</o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><o:p>&=
nbsp;</o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><o:p>&=
nbsp;</o:p></span></p>

<p class=3DMsoNormal><i><span lang=3DEN-US><o:p>&nbsp;</o:p></span></i></p>

<p class=3DMsoNormal><i><span lang=3DEN-US><o:p>&nbsp;</o:p></span></i></p>

<p class=3DMsoNormal><i><span lang=3DEN-US><o:p>&nbsp;</o:p></span></i></p>

<br style=3D'mso-ignore:vglayout' clear=3DALL>

<p class=3DMsoNormal style=3D'text-align:justify'><span class=3DGramE><span
lang=3DEN-US>where</span></span><span lang=3DEN-US> a represents vector of =
fitted
parameters (positions p(j), amplitudes A(j), sigma, relative amplitudes T, S
and slope B.</span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:16.0pt'><o:p>&nb=
sp;</o:p></span></p>

</div>
<!-- */
// --> End_Html
    
   if (numberPeaks <= 0){
      Error ("TSpectrumFit","Invalid number of peaks, must be > than 0");
      return;
   }
   fNPeaks = numberPeaks;   
   fPositionInit   = new Double_t[numberPeaks];
   fPositionCalc   = new Double_t[numberPeaks];   
   fPositionErr   = new Double_t[numberPeaks];   
   fFixPosition   = new Bool_t[numberPeaks];   
   fAmpInit   = new Double_t[numberPeaks];   
   fAmpCalc   = new Double_t[numberPeaks];   
   fAmpErr    = new Double_t[numberPeaks];   
   fFixAmp    = new Bool_t[numberPeaks];     
   fArea      = new Double_t[numberPeaks];   
   fAreaErr   = new Double_t[numberPeaks];      
   fXmin=0,fXmax=100,fSigmaInit = 2,fFixSigma = false;
   fAlpha =1;                     
   fStatisticType = kFitOptimChiCounts;
   fAlphaOptim = kFitAlphaHalving;     
   fPower = kFitPower2;                
   fFitTaylor = kFitTaylorOrderFirst;  
   fTInit = 0;  
   fFixT = true;
   fBInit = 1;  
   fFixB = true;
   fSInit = 0;  
   fFixS = true;
   fA0Init = 0; 
   fFixA0 = true;
   fA1Init = 0;
   fFixA1 = true;
   fA2Init = 0;
   fFixA2 = true;
}
    
    

//______________________________________________________________________________
TSpectrumFit::~TSpectrumFit() 
{
   //destructor
   delete [] fPositionInit;   
   delete [] fPositionCalc;   
   delete [] fPositionErr;
   delete [] fFixPosition;      
   delete [] fAmpInit;  
   delete [] fAmpCalc;  
   delete [] fAmpErr;   
   delete [] fFixAmp;
   delete [] fArea;  
   delete [] fAreaErr;         
}

//_____________________________________________________________________________
/////////////////BEGINNING OF AUXILIARY FUNCTIONS USED BY FITTING FUNCTION Fit1//////////////////////////
Double_t TSpectrumFit::Erfc(Double_t x) 
{   
//////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                      //
//                                                                          //
//   This function calculates error function of x.                           //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////
   Double_t da1 = 0.1740121, da2 = -0.0479399, da3 = 0.3739278, dap = 0.47047;
   Double_t a, t, c, w;
   a = TMath::Abs(x);
   w = 1. + dap * a;
   t = 1. / w;
   w = a * a;
   if (w < 700)
      c = exp(-w);
   
   else {
      c = 0;
   }
   c = c * t * (da1 + t * (da2 + t * da3));
   if (x < 0)
      c = 1. - c;
   return (c);
}

//____________________________________________________________________________
Double_t TSpectrumFit::Derfc(Double_t x) 
{
//////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                      //
//                                                                          //
//   This function calculates derivative of error function of x.             //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////
   Double_t a, t, c, w;
   Double_t da1 = 0.1740121, da2 = -0.0479399, da3 = 0.3739278, dap = 0.47047;
   a = TMath::Abs(x);
   w = 1. + dap * a;
   t = 1. / w;
   w = a * a;
   if (w < 700)
      c = exp(-w);
   
   else {
      c = 0;
   }
   c = (-1.) * dap * c * t * t * (da1 + t * (2. * da2 + t * 3. * da3)) -
       2. * a * Erfc(a);
   return (c);
}

//____________________________________________________________________________
Double_t TSpectrumFit::Deramp(Double_t i, Double_t i0, Double_t sigma, Double_t t,
                           Double_t s, Double_t b) 
{   
//////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                      //
//                                                                          //
//   This function calculates derivative of peak shape function (see manual) //
//   according to amplitude of peak.                                        //
//      Function parameters:                                                //
//              -i-channel                                                  //
//              -i0-position of peak                                        //
//              -sigma-sigma of peak                                        //
//              -t, s-relative amplitudes                                   //
//              -b-slope                                                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////
   Double_t p, q, r, a;
   p = (i - i0) / sigma;
   if ((p * p) < 700)
      q = exp(-p * p);
   
   else {
      q = 0;
   }
   r = 0;
   if (t != 0) {
      a = p / b;
      if (a > 700)
         a = 700;
      r = t * exp(a) / 2.;
   }
   if (r != 0)
      r = r * Erfc(p + 1. / (2. * b));
   q = q + r;
   if (s != 0)
      q = q + s * Erfc(p) / 2.;
   return (q);
}

//____________________________________________________________________________
Double_t TSpectrumFit::Deri0(Double_t i, Double_t amp, Double_t i0, Double_t sigma,
                          Double_t t, Double_t s, Double_t b) 
{  
//////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                      //
//                                                                          //
//   This function calculates derivative of peak shape function (see manual) //
//   according to peak position.                                            //
//      Function parameters:                                                //
//              -i-channel                                                  //
//              -amp-amplitude of peak                                      //
//              -i0-position of peak                                        //
//              -sigma-sigma of peak                                        //
//              -t, s-relative amplitudes                                   //
//              -b-slope                                                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////
   Double_t p, r1, r2, r3, r4, c, d, e;
   p = (i - i0) / sigma;
   d = 2. * sigma;
   if ((p * p) < 700)
      r1 = 2. * p * exp(-p * p) / sigma;
   
   else {
      r1 = 0;
   }
   r2 = 0, r3 = 0;
   if (t != 0) {
      c = p + 1. / (2. * b);
      e = p / b;
      if (e > 700)
         e = 700;
      r2 = -t * exp(e) * Erfc(c) / (d * b);
      r3 = -t * exp(e) * Derfc(c) / d;
   }
   r4 = 0;
   if (s != 0)
      r4 = -s * Derfc(p) / d;
   r1 = amp * (r1 + r2 + r3 + r4);
   return (r1);
}

//____________________________________________________________________________
Double_t TSpectrumFit::Derderi0(Double_t i, Double_t amp, Double_t i0,
                             Double_t sigma) 
{  
//////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                      //
//                                                                          //
//   This function calculates second derivative of peak shape function       //
//   (see manual) according to peak position.                               //
//      Function parameters:                                                //
//              -i-channel                                                  //
//              -amp-amplitude of peak                                      //
//              -i0-position of peak                                        //
//              -sigma-width of peak                                        //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////
   Double_t p, r1, r2, r3, r4;
   p = (i - i0) / sigma;
   if ((p * p) < 700)
      r1 = exp(-p * p);
   
   else {
      r1 = 0;
   }
   r1 = r1 * (4 * p * p - 2) / (sigma * sigma);
   r2 = 0, r3 = 0, r4 = 0;
   r1 = amp * (r1 + r2 + r3 + r4);
   return (r1);
}

//____________________________________________________________________________
Double_t TSpectrumFit::Dersigma(Int_t num_of_fitted_peaks, Double_t i,
                             const Double_t *parameter, Double_t sigma,
                             Double_t t, Double_t s, Double_t b) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates derivative of peaks shape function (see manual)    //
//   according to sigma of peaks.                                               //
//      Function parameters:                                                    //
//              -num_of_fitted_peaks-number of fitted peaks                     //
//              -i-channel                                                      //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigma-sigma of peak                                            //
//              -t, s-relative amplitudes                                       //
//              -b-slope                                                        //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Int_t j;
   Double_t r, p, r1, r2, r3, r4, c, d, e;
   r = 0;
   d = 2. * sigma;
   for (j = 0; j < num_of_fitted_peaks; j++) {
      p = (i - parameter[2 * j + 1]) / sigma;
      r1 = 0;
      if (TMath::Abs(p) < 3) {
         if ((p * p) < 700)
            r1 = 2. * p * p * exp(-p * p) / sigma;
         
         else {
            r1 = 0;
         }
      }
      r2 = 0, r3 = 0;
      if (t != 0) {
         c = p + 1. / (2. * b);
         e = p / b;
         if (e > 700)
            e = 700;
         r2 = -t * p * exp(e) * Erfc(c) / (d * b);
         r3 = -t * p * exp(e) * Derfc(c) / d;
      }
      r4 = 0;
      if (s != 0)
         r4 = -s * p * Derfc(p) / d;
      r = r + parameter[2 * j] * (r1 + r2 + r3 + r4);
   }
   return (r);
}

//____________________________________________________________________________
Double_t TSpectrumFit::Derdersigma(Int_t num_of_fitted_peaks, Double_t i,
                               const Double_t *parameter, Double_t sigma) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates second derivative of peaks shape function          //
//   (see manual) according to sigma of peaks.                                  //
//      Function parameters:                                                    //
//              -num_of_fitted_peaks-number of fitted peaks                     //
//              -i-channel                                                      //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigma-sigma of peak                                            //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Int_t j;
   Double_t r, p, r1, r2, r3, r4;
   r = 0;
   for (j = 0; j < num_of_fitted_peaks; j++) {
      p = (i - parameter[2 * j + 1]) / sigma;
      r1 = 0;
      if (TMath::Abs(p) < 3) {
         if ((p * p) < 700)
            r1 = exp(-p * p) * p * p * (4. * p * p - 6) / (sigma * sigma);
         
         else {
            r1 = 0;
         }
      }
      r2 = 0, r3 = 0, r4 = 0;
      r = r + parameter[2 * j] * (r1 + r2 + r3 + r4);
   }
   return (r);
}

//____________________________________________________________________________
Double_t TSpectrumFit::Dert(Int_t num_of_fitted_peaks, Double_t i,
                        const Double_t *parameter, Double_t sigma, Double_t b) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates derivative of peaks shape function (see manual)    //
//   according to relative amplitude t.                                         //
//      Function parameters:                                                    //
//              -num_of_fitted_peaks-number of fitted peaks                     //
//              -i-channel                                                      //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigma-sigma of peak                                            //
//              -b-slope                                                        //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Int_t j;
   Double_t r, p, r1, c, e;
   r = 0;
   for (j = 0; j < num_of_fitted_peaks; j++) {
      p = (i - parameter[2 * j + 1]) / sigma;
      c = p + 1. / (2. * b);
      e = p / b;
      if (e > 700)
         e = 700;
      r1 = exp(e) * Erfc(c);
      r = r + parameter[2 * j] * r1;
   }
   r = r / 2.;
   return (r);
}

//____________________________________________________________________________
Double_t TSpectrumFit::Ders(Int_t num_of_fitted_peaks, Double_t i,
                        const Double_t *parameter, Double_t sigma) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates derivative of peaks shape function (see manual)    //
//   according to relative amplitude s.                                               //
//      Function parameters:                                                    //
//              -num_of_fitted_peaks-number of fitted peaks                     //
//              -i-channel                                                      //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigma-sigma of peak                                            //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Int_t j;
   Double_t r, p, r1;
   r = 0;
   for (j = 0; j < num_of_fitted_peaks; j++) {
      p = (i - parameter[2 * j + 1]) / sigma;
      r1 = Erfc(p);
      r = r + parameter[2 * j] * r1;
   }
   r = r / 2.;
   return (r);
}

//____________________________________________________________________________
Double_t TSpectrumFit::Derb(Int_t num_of_fitted_peaks, Double_t i,
                        const Double_t *parameter, Double_t sigma, Double_t t,
                        Double_t b) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates derivative of peaks shape function (see manual)    //
//   according to slope b.                                                      //
//      Function parameters:                                                    //
//              -num_of_fitted_peaks-number of fitted peaks                     //
//              -i-channel                                                      //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigma-sigma of peak                                            //
//              -t-relative amplitude                                           //
//              -b-slope                                                        //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Int_t j;
   Double_t r, p, r1, c, e;
   r = 0;
   for (j = 0; j < num_of_fitted_peaks && t != 0; j++) {
      p = (i - parameter[2 * j + 1]) / sigma;
      c = p + 1. / (2. * b);
      e = p / b;
      r1 = p * Erfc(c);
      r1 = r1 + Derfc(c) / 2.;
      if (e > 700)
         e = 700;
      if (e < -700)
         r1 = 0;
      
      else
         r1 = r1 * exp(e);
      r = r + parameter[2 * j] * r1;
   }
   r = -r * t / (2. * b * b);
   return (r);
}

//____________________________________________________________________________
Double_t TSpectrumFit::Dera1(Double_t i)       
{
   //derivative of background according to a1
   return (i);
}

//____________________________________________________________________________
Double_t TSpectrumFit::Dera2(Double_t i)      
{
   //derivative of background according to a2
   return (i * i);
}

//____________________________________________________________________________
Double_t TSpectrumFit::Shape(Int_t num_of_fitted_peaks, Double_t i,
                         const Double_t *parameter, Double_t sigma, Double_t t,
                         Double_t s, Double_t b, Double_t a0, Double_t a1,
                         Double_t a2) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates peaks shape function (see manual)                  //
//      Function parameters:                                                    //
//              -num_of_fitted_peaks-number of fitted peaks                     //
//              -i-channel                                                      //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigma-sigma of peak                                            //
//              -t, s-relative amplitudes                                       //
//              -b-slope                                                        //
//              -a0, a1, a2- background coefficients                            //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Int_t j;
   Double_t r, p, r1, r2, r3, c, e;
   r = 0;
   for (j = 0; j < num_of_fitted_peaks; j++) {
      if (sigma > 0.0001)
         p = (i - parameter[2 * j + 1]) / sigma;
      
      else {
         if (i == parameter[2 * j + 1])
            p = 0;
         
         else
            p = 10;
      }
      r1 = 0;
      if (TMath::Abs(p) < 3) {
         if ((p * p) < 700)
            r1 = exp(-p * p);
         
         else {
            r1 = 0;
         }
      }
      r2 = 0;
      if (t != 0) {
         c = p + 1. / (2. * b);
         e = p / b;
         if (e > 700)
            e = 700;
         r2 = t * exp(e) * Erfc(c) / 2.;
      }
      r3 = 0;
      if (s != 0)
         r3 = s * Erfc(p) / 2.;
      r = r + parameter[2 * j] * (r1 + r2 + r3);
   }
   r = r + a0 + a1 * i + a2 * i * i;
   return (r);
}

//____________________________________________________________________________
Double_t TSpectrumFit::Area(Double_t a, Double_t sigma, Double_t t, Double_t b) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates area of a peak                                     //
//      Function parameters:                                                    //
//              -a-amplitude of the peak                                        //
//              -sigma-sigma of peak                                            //
//              -t-relative amplitude                                           //
//              -b-slope                                                        //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Double_t odm_pi = 1.7724538, r = 0;
   if (b != 0)
      r = 0.5 / b;
   r = (-1.) * r * r;
   if (TMath::Abs(r) < 700)
      r = a * sigma * (odm_pi + t * b * exp(r));
   
   else {
      r = a * sigma * odm_pi;
   }
   return (r);
}

//____________________________________________________________________________
Double_t TSpectrumFit::Derpa(Double_t sigma, Double_t t, Double_t b) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates derivative of the area of peak                     //
//   according to its amplitude.                                                //
//      Function parameters:                                                    //
//              -sigma-sigma of peak                                            //
//              -t-relative amplitudes                                          //
//              -b-slope                                                        //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Double_t odm_pi = 1.7724538, r;
   r = 0.5 / b;
   r = (-1.) * r * r;
   if (TMath::Abs(r) < 700)
      r = sigma * (odm_pi + t * b * exp(r));
   
   else {
      r = sigma * odm_pi;
   }
   return (r);
}
Double_t TSpectrumFit::Derpsigma(Double_t a, Double_t t, Double_t b) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates derivative of the area of peak                     //
//   according to sigma of peaks.                                               //
//      Function parameters:                                                    //
//              -a-amplitude of peak                                            //
//              -t-relative amplitudes                                          //
//              -b-slope                                                        //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Double_t odm_pi = 1.7724538, r;
   r = 0.5 / b;
   r = (-1.) * r * r;
   if (TMath::Abs(r) < 700)
      r = a * (odm_pi + t * b * exp(r));
   
   else {
      r = a * odm_pi;
   }
   return (r);
}

//______________________________________________________________________________
Double_t TSpectrumFit::Derpt(Double_t a, Double_t sigma, Double_t b) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates derivative of the area of peak                     //
//   according to t parameter.                                                  //
//      Function parameters:                                                    //
//              -sigma-sigma of peak                                            //
//              -t-relative amplitudes                                          //
//              -b-slope                                                        //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Double_t r;
   r = 0.5 / b;
   r = (-1.) * r * r;
   if (TMath::Abs(r) < 700)
      r = a * sigma * b * exp(r);
   
   else {
      r = 0;
   }
   return (r);
}

//______________________________________________________________________________
Double_t TSpectrumFit::Derpb(Double_t a, Double_t sigma, Double_t t, Double_t b) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates derivative of the area of peak                     //
//   according to b parameter.                                                  //
//      Function parameters:                                                    //
//              -sigma-sigma of peak                                            //
//              -t-relative amplitudes                                          //
//              -b-slope                                                        //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Double_t r;
   r = (-1) * 0.25 / (b * b);
   if (TMath::Abs(r) < 700)
      r = a * sigma * t * exp(r) * (1 - 2 * r);
   
   else {
      r = 0;
   }
   return (r);
}

//______________________________________________________________________________
Double_t TSpectrumFit::Ourpowl(Double_t a, Int_t pw)
{                               
   //power function
   Double_t c;
   c = 1;
   if (pw > 0)
      c = c * a * a;
   
   else if (pw > 2)
      c = c * a * a;
   
   else if (pw > 4)
      c = c * a * a;
   
   else if (pw > 6)
      c = c * a * a;
   
   else if (pw > 8)
      c = c * a * a;
   
   else if (pw > 10)
      c = c * a * a;
   
   else if (pw > 12)
      c = c * a * a;
   return (c);
}

/////////////////END OF AUXILIARY FUNCTIONS USED BY FITTING FUNCTIONS FitAWMI, FitStiefel//////////////////////////
/////////////////FITTING FUNCTION WITHOUT MATRIX INVERSION///////////////////////////////////////

//____________________________________________________________________________
void TSpectrumFit::FitAwmi(Float_t *source) 
{
/////////////////////////////////////////////////////////////////////////////
//        ONE-DIMENSIONAL FIT FUNCTION                                 
//        ALGORITHM WITHOUT MATRIX INVERSION                                  
//        This function fits the source spectrum. The calling program should 
//        fill in input parameters of the TSpectrumFit class            
//        The fitted parameters are written into           
//        TSpectrumFit class output parameters and fitted data are written into 
//        source spectrum.                                                    
//                                                                          
//        Function parameters:                                             
//        source-pointer to the vector of source spectrum                  
//                                                                         
/////////////////////////////////////////////////////////////////////////////

//Begin_Html <!--
/* -->
<div class=3DSection1>

<p class=3DMsoNormal><b><span lang=3DEN-US style=3D'font-size:14.0pt'>Fitti=
ng</span></b><span
lang=3DEN-US style=3D'font-size:14.0pt'><o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><i><span lang=3DEN-US>Goa=
l: to
estimate simultaneously peak shape parameters in spectra with large number =
of
peaks</span></i></p>

<p class=3DMsoNormal style=3D'margin-left:.5in;text-align:justify;text-inde=
nt:-.25in;
mso-list:l5 level1 lfo7;tab-stops:list .5in'><![if !supportLists]><span
lang=3DEN-US><span style=3D'mso-list:Ignore'>&#8226;<span style=3D'font:7.0=
pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span></span><![endif]><span class=3DGramE><span lang=3DEN-US>peaks=
</span></span><span
lang=3DEN-US> can be fitted separately, each peak (or <span class=3DSpellE>=
multiplets</span>)
in a region or together all peaks in a spectrum. To fit separately each peak
one needs to determine the fitted region. However it can happen that the
regions of neighboring peaks are overlapping. Then the results of fitting a=
re
very poor. On the other hand, when fitting together all peaks found in a<sp=
an
style=3D'mso-spacerun:yes'>&nbsp; </span>spectrum, one needs to have a meth=
od
that is<span style=3D'mso-spacerun:yes'>&nbsp; </span>stable (converges) an=
d fast
enough to carry out fitting in reasonable time </span></p>

<p class=3DMsoNormal style=3D'margin-left:.5in;text-align:justify;text-inde=
nt:-.25in;
mso-list:l5 level1 lfo7;tab-stops:list .5in'><![if !supportLists]><span
lang=3DEN-US><span style=3D'mso-list:Ignore'>&#8226;<span style=3D'font:7.0=
pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span></span><![endif]><span lang=3DEN-US>we have implemented the
nonsymmetrical <span class=3DSpellE>semiempirical</span> peak shape functio=
n [1]</span></p>

<p class=3DMsoNormal style=3D'margin-left:.5in;text-align:justify;text-inde=
nt:-.25in;
mso-list:l5 level1 lfo7;tab-stops:list .5in'><![if !supportLists]><span
lang=3DEN-US><span style=3D'mso-list:Ignore'>&#8226;<span style=3D'font:7.0=
pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span></span><![endif]><span class=3DGramE><span lang=3DEN-US>it</s=
pan></span><span
lang=3DEN-US> contains the symmetrical Gaussian as well as nonsymmetrical t=
erms.</span></p>

<p class=3DMsoNormal style=3D'text-align:justify;tab-stops:.75in'><!--[if g=
te vml 1]><v:shapetype
 id=3D"_x0000_t75" coordsize=3D"21600,21600" o:spt=3D"75" o:preferrelative=
=3D"t"
 path=3D"m@4@5l@4@11@9@11@9@5xe" filled=3D"f" stroked=3D"f">
 <v:stroke joinstyle=3D"miter"/>
 <v:formulas>
  <v:f eqn=3D"if lineDrawn pixelLineWidth 0"/>
  <v:f eqn=3D"sum @0 1 0"/>
  <v:f eqn=3D"sum 0 0 @1"/>
  <v:f eqn=3D"prod @2 1 2"/>
  <v:f eqn=3D"prod @3 21600 pixelWidth"/>
  <v:f eqn=3D"prod @3 21600 pixelHeight"/>
  <v:f eqn=3D"sum @0 0 1"/>
  <v:f eqn=3D"prod @6 1 2"/>
  <v:f eqn=3D"prod @7 21600 pixelWidth"/>
  <v:f eqn=3D"sum @8 21600 0"/>
  <v:f eqn=3D"prod @7 21600 pixelHeight"/>
  <v:f eqn=3D"sum @10 21600 0"/>
 </v:formulas>
 <v:path o:extrusionok=3D"f" gradientshapeok=3D"t" o:connecttype=3D"rect"/>
 <o:lock v:ext=3D"edit" aspectratio=3D"t"/>
</v:shapetype><v:shape id=3D"_x0000_s1037" type=3D"#_x0000_t75" style=3D'po=
sition:absolute;
 left:0;text-align:left;margin-left:62.7pt;margin-top:13.2pt;width:279.3pt;
 height:95.35pt;z-index:1' fillcolor=3D"window">
 <v:imagedata src=3D"FitAwmi_files/image001.wmz" o:title=3D""/>
</v:shape><![if gte mso 9]><o:OLEObject Type=3D"Embed" ProgID=3D"Equation.D=
SMT4"
 ShapeID=3D"_x0000_s1037" DrawAspect=3D"Content" ObjectID=3D"_1220797124">
</o:OLEObject>
<![endif]><![endif]--><![if !vml]><span style=3D'mso-ignore:vglayout'>

<table cellpadding=3D0 cellspacing=3D0 align=3Dleft>
 <tr>
  <td width=3D84 height=3D18></td>
 </tr>
 <tr>
  <td></td>
  <td><img width=3D372 height=3D127 src=3D"FitAwmi_files/image002.gif" v:sh=
apes=3D"_x0000_s1037"></td>
 </tr>
</table>

</span><![endif]><span lang=3DEN-US style=3D'font-size:16.0pt'><o:p>&nbsp;<=
/o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:16.0pt'><o:p>&nbsp;</o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:16.0pt'><o:p>&nbsp;</o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:16.0pt'><o:p>&nbsp;</o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:16.0pt'><o:p>&nbsp;</o:p></span></p>

<br style=3D'mso-ignore:vglayout' clear=3DALL>

<p class=3DMsoNormal style=3D'text-indent:34.2pt'><span class=3DGramE><span
lang=3DEN-US>where</span></span><span lang=3DEN-US> T and S are relative am=
plitudes
and B is slope.</span></p>

<p class=3DMsoNormal><span lang=3DEN-US><o:p>&nbsp;</o:p></span></p>

<p class=3DMsoNormal style=3D'margin-left:.5in;text-align:justify;text-inde=
nt:-.25in;
mso-list:l6 level1 lfo8;tab-stops:list .5in'><![if !supportLists]><span
lang=3DEN-US><span style=3D'mso-list:Ignore'>&#8226;<span style=3D'font:7.0=
pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span></span><![endif]><span class=3DGramE><span lang=3DEN-US>algor=
ithm</span></span><span
lang=3DEN-US> without matrix inversion (AWMI) allows fitting tens, hundreds=
 of
peaks simultaneously that represent sometimes thousands of parameters [2], =
[5].
</span></p>

<p class=3DMsoNormal><i><span lang=3DEN-US>Function:</span></i></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span class=3DGramE><span
lang=3DEN-US>void</span></span><span lang=3DEN-US> <a
href=3D"http://root.cern.ch/root/html/TSpectrum.html#TSpectrum:Fit1Awmi"><s=
pan
class=3DSpellE><b style=3D'mso-bidi-font-weight:normal'>TSpectrumFit::FitAw=
mi</b></span></a>(<a
href=3D"http://root.cern.ch/root/html/ListOfTypes.html#float"><b
style=3D'mso-bidi-font-weight:normal'>float</b></a> *<span class=3DSpellE>f=
Source</span>)<span
style=3D'mso-bidi-font-weight:bold'> </span></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US>This f=
unction
fits the source spectrum using AWMI algorithm. The calling program should f=
ill
in input fitting parameters of the <span class=3DSpellE>TSpectrumFit</span>=
 class
using a set of <span class=3DSpellE>TSpectrumFit</span> setters. The fitted
parameters are written into the class and the fitted data are written into
source spectrum. </span></p>

<p class=3DMsoNormal><span lang=3DEN-US><o:p>&nbsp;</o:p></span></p>

<p class=3DMsoNormal><i style=3D'mso-bidi-font-style:normal'><span lang=3DE=
N-US
style=3D'color:red'>Parameter:<o:p></o:p></span></i></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </spa=
n><span
class=3DSpellE><span class=3DGramE><b style=3D'mso-bidi-font-weight:normal'=
>fSource</b></span></span><span
class=3DGramE>-pointer</span> to the vector of source spectrum<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><span
style=3D'mso-spacerun:yes'>&nbsp;</span><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span></sp=
an></p>

<p class=3DMsoNormal><i style=3D'mso-bidi-font-style:normal'><span lang=3DE=
N-US
style=3D'color:red'>Member variables of the <span class=3DSpellE>TSpectrumF=
it</span>
class:<o:p></o:p></span></i></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Int_t</span><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp; </span><span class=3DSp=
ellE>fNPeaks</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//number of peaks present in fit, input parameter, it should be &gt;=
 0<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Int_t</span><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp; </span><span class=3DSp=
ellE>fNumberIterations</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;
</span>//number of iterations in fitting procedure, input parameter, it sho=
uld
be &gt; 0<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Int_t</span><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp; </span><span class=3DSp=
ellE>fXmin</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nb=
sp;
</span>//first fitted channel<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Int_t</span><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp; </span><span class=3DSp=
ellE>fXmax</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nb=
sp;
</span>//last fitted channel<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Int_t</span><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp; </span><span class=3DSp=
ellE>fStatisticType</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;
</span>//type of statistics, possible values <span class=3DSpellE>kFitOptim=
ChiCounts</span>
(chi square statistics with counts as weighting coefficients), <span
class=3DSpellE>kFitOptimChiFuncValues</span> (chi square statistics with fu=
nction
values as weighting coefficients)<span class=3DGramE>,<span class=3DSpellE>=
kFitOptimMaxLikelihood</span></span><o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Int_t</span><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp; </span><span class=3DSp=
ellE>fAlphaOptim</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//optimization of convergence algorithm, possible values <span
class=3DSpellE>kFitAlphaHalving</span>, <span class=3DSpellE>kFitAlphaOptim=
al</span><o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Int_t</span><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp; </span><span class=3DSp=
ellE>fPower</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//possible values kFitPower2<span class=3DGramE>,4,6,8,10,12</span>,=
 for
details see references. It applies only for <span class=3DSpellE>Awmi</span>
fitting function.<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE><span class=3DGramE>Int_t</span></span><span class=3DGramE><=
span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp; </span><span class=3DSp=
ellE>fFitTaylor</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//order of <st1:place w:st=3D"on"><st1:City w:st=3D"on">Taylor</st1:=
City></st1:place>
expansion, possible values <span class=3DSpellE>kFitTaylorOrderFirst</span>=
, <span
class=3DSpellE>kFitTaylorOrderSecond</span>.</span> It applies only for <sp=
an
class=3DSpellE>Awmi</span> fitting function.<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Double_<span class=3DGramE>t</span></span><span class=3DGram=
E><span
style=3D'mso-spacerun:yes'>&nbsp; </span><span class=3DSpellE>fAlpha</span>=
</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//convergence coefficient, input parameter, it should be positive nu=
mber
and &lt;=3D1, for details see references<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Double_<span class=3DGramE>t</span></span><span class=3DGram=
E><span
style=3D'mso-spacerun:yes'>&nbsp; </span><span class=3DSpellE>fChi</span></=
span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nb=
sp;&nbsp;
</span>//here the fitting functions return resulting chi square<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Double_t</span> *<span class=3DSpellE>fPositionInit</span>;<=
span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//[<span class=3DSpellE>fNPeaks</span>] array of initial values of p=
eaks
positions, input parameters<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Double_t</span> *<span class=3DSpellE>fPositionCalc</span>;<=
span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//[<span class=3DSpellE>fNPeaks</span>] array of calculated values of
fitted positions, output parameters<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Double_t</span> *<span class=3DSpellE>fPositionErr</span>;<s=
pan
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//[<span class=3DSpellE>fNPeaks</span>] array of position errors<o:p=
></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Double_t</span> *<span class=3DSpellE>fAmpInit</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//[<span class=3DSpellE>fNPeaks</span>] array of initial values of p=
eaks
amplitudes, input parameters<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Double_t</span> *<span class=3DSpellE>fAmpCalc</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//[<span class=3DSpellE>fNPeaks</span>] array of calculated values of
fitted amplitudes, output parameters<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Double_t</span> *<span class=3DSpellE>fAmpErr</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//[<span class=3DSpellE>fNPeaks</span>] array of amplitude errors<o:=
p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Double_t</span> *<span class=3DSpellE>fArea</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>//[<span
class=3DSpellE>fNPeaks</span>] array of calculated areas of peaks<o:p></o:p=
></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Double_t</span> *<span class=3DSpellE>fAreaErr</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//[<span class=3DSpellE>fNPeaks</span>] array of errors of peak area=
s<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Double_<span class=3DGramE>t</span></span><span class=3DGram=
E><span
style=3D'mso-spacerun:yes'>&nbsp; </span><span class=3DSpellE>fSigmaInit</s=
pan></span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//initial value of sigma parameter<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Double_<span class=3DGramE>t</span></span><span class=3DGram=
E><span
style=3D'mso-spacerun:yes'>&nbsp; </span><span class=3DSpellE>fSigmaCalc</s=
pan></span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;
</span><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span=
>//calculated
value of sigma parameter<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Double_<span class=3DGramE>t</span></span><span class=3DGram=
E><span
style=3D'mso-spacerun:yes'>&nbsp; </span><span class=3DSpellE>fSigmaErr</sp=
an></span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//error value of sigma parameter<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Double_<span class=3DGramE>t</span></span><span class=3DGram=
E><span
style=3D'mso-spacerun:yes'>&nbsp; </span><span class=3DSpellE>fTInit</span>=
</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//initial value of t parameter (relative amplitude of tail), for det=
ails
see html manual and references<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Double_<span class=3DGramE>t</span></span><span class=3DGram=
E><span
style=3D'mso-spacerun:yes'>&nbsp; </span><span class=3DSpellE>fTCalc</span>=
</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//calculated value of t parameter<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Double_<span class=3DGramE>t</span></span><span class=3DGram=
E><span
style=3D'mso-spacerun:yes'>&nbsp; </span><span class=3DSpellE>fTErr</span><=
/span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nb=
sp;
</span>//error value of t parameter<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Double_<span class=3DGramE>t</span></span><span class=3DGram=
E><span
style=3D'mso-spacerun:yes'>&nbsp; </span><span class=3DSpellE>fBInit</span>=
</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//initial value of b parameter (slope), for details see html manual =
and
references<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Double_<span class=3DGramE>t</span></span><span class=3DGram=
E><span
style=3D'mso-spacerun:yes'>&nbsp; </span><span class=3DSpellE>fBCalc</span>=
</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//calculated value of b parameter<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Double_<span class=3DGramE>t</span></span><span class=3DGram=
E><span
style=3D'mso-spacerun:yes'>&nbsp; </span><span class=3DSpellE>fBErr</span><=
/span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nb=
sp;
</span>//error value of b parameter<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Double_<span class=3DGramE>t</span></span><span class=3DGram=
E><span
style=3D'mso-spacerun:yes'>&nbsp; </span><span class=3DSpellE>fSInit</span>=
</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//initial value of s parameter (relative amplitude of step), for det=
ails
see html manual and references<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Double_<span class=3DGramE>t</span></span><span class=3DGram=
E><span
style=3D'mso-spacerun:yes'>&nbsp; </span><span class=3DSpellE>fSCalc</span>=
</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//calculated value of s parameter<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Double_<span class=3DGramE>t</span></span><span class=3DGram=
E><span
style=3D'mso-spacerun:yes'>&nbsp; </span><span class=3DSpellE>fSErr</span><=
/span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nb=
sp;
</span>//error value of s parameter<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Double_<span class=3DGramE>t</span></span><span class=3DGram=
E><span
style=3D'mso-spacerun:yes'>&nbsp; </span>fA0Init</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//initial value of background a0 parameter(<span class=3DSpellE>back=
groud</span>
is estimated as a0+a1*x+a2*x*x)<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Double_<span class=3DGramE>t</span></span><span class=3DGram=
E><span
style=3D'mso-spacerun:yes'>&nbsp; </span>fA0Calc</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//calculated value of background a0 parameter<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Double_<span class=3DGramE>t</span></span><span class=3DGram=
E><span
style=3D'mso-spacerun:yes'>&nbsp; </span>fA0Err</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//error value of background a0 parameter<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Double_<span class=3DGramE>t</span></span><span class=3DGram=
E><span
style=3D'mso-spacerun:yes'>&nbsp; </span>fA1Init</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//initial value of background a1 parameter(<span class=3DSpellE>back=
groud</span>
is estimated as a0+a1*x+a2*x*x)<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Double_<span class=3DGramE>t</span></span><span class=3DGram=
E><span
style=3D'mso-spacerun:yes'>&nbsp; </span>fA1Calc</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//calculated value of background a1 parameter<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Double_<span class=3DGramE>t</span></span><span class=3DGram=
E><span
style=3D'mso-spacerun:yes'>&nbsp; </span>fA1Err</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//error value of background a1 parameter<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Double_<span class=3DGramE>t</span></span><span class=3DGram=
E><span
style=3D'mso-spacerun:yes'>&nbsp; </span>fA2Init</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//initial value of background a2 parameter(<span class=3DSpellE>back=
groud</span>
is estimated as a0+a1*x+a2*x*x)<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Double_<span class=3DGramE>t</span></span><span class=3DGram=
E><span
style=3D'mso-spacerun:yes'>&nbsp; </span>fA2Calc</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//calculated value of background a2 parameter<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Double_<span class=3DGramE>t</span></span><span class=3DGram=
E><span
style=3D'mso-spacerun:yes'>&nbsp; </span>fA2Err</span>; <span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</s=
pan>//error
value of background a2 parameter<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE><span class=3DGramE>Bool_t</span></span><span class=3DGramE>=
<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>*<span class=3DSpellE>fFixPo=
sition</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//[<span class=3DSpellE>fNPeaks</span>] array of logical values which
allow to fix appropriate positions (not fit).</span> However they are prese=
nt
in the estimated functional<span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE><span class=3DGramE>Bool_t</span></span><span class=3DGramE>=
 <span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;</span>*<span class=3DSpellE>fFixAmp=
</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//[<span class=3DSpellE>fNPeaks</span>] array of logical values which
allow to fix appropriate amplitudes (not fit).</span> However they are pres=
ent
in the estimated functional<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span><o:p></o:p=
></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Bool_t</span><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&n=
bsp;
</span><span class=3DSpellE>fFixSigma</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//logical value of sigma parameter, which allows <span class=3DGramE=
>to
fix</span> the parameter (not to fit).<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Bool_t</span><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&n=
bsp;
</span><span class=3DSpellE>fFixT</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nb=
sp;
</span>//logical value of t parameter, which allows <span class=3DGramE>to =
fix</span>
the parameter (not to fit).<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span><o:p></o:p=
></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Bool_t</span><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&n=
bsp;
</span><span class=3DSpellE>fFixB</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nb=
sp;
</span>//logical value of b parameter, which allows <span class=3DGramE>to =
fix</span>
the parameter (not to fit).<span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Bool_t</span><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&n=
bsp;
</span><span class=3DSpellE>fFixS</span>;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nb=
sp;
</span>//logical value of s parameter, which allows <span class=3DGramE>to =
fix</span>
the parameter (not to fit).<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span><o:p></o:p=
></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Bool_t</span><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&n=
bsp;
</span>fFixA0;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//logical value of a0 parameter, which allows <span class=3DGramE>to=
 fix</span>
the parameter (not to fit).<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Bool_t</span><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&n=
bsp;
</span>fFixA1;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//logical value of a1 parameter, which allows <span class=3DGramE>to=
 fix</span>
the parameter (not to fit).<span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US
style=3D'font-size:10.0pt'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </=
span><span
class=3DSpellE>Bool_t</span><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&n=
bsp;
</span>fFixA2;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>//logical value of a2 parameter, which allows <span class=3DGramE>to=
 fix</span>
the parameter (not to fit).<o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><b><i><span lang=3DEN-US>=
<o:p>&nbsp;</o:p></span></i></b></p>

<p class=3DMsoNormal style=3D'text-align:justify'><b><i><span lang=3DEN-US>=
References:<o:p></o:p></span></i></b></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US>[1] <s=
pan
class=3DSpellE>Phillps</span> G.W., Marlow K.W., NIM 137 (1976) 525.</span>=
</p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US>[2] I.=
 A.
Slavic: Nonlinear least-squares fitting without matrix inversion applied to
complex Gaussian spectra analysis. NIM 134 (1976) 285-289.</span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US>[3] T.=
 <span
class=3DSpellE>Awaya</span>: A new method for curve fitting to the data wit=
h low
statistics not using chi-square method. NIM 165 (1979) 317-323.</span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US>[4] T.=
 <span
class=3DSpellE>Hauschild</span>, M. <span class=3DSpellE>Jentschel</span>:
Comparison of maximum likelihood estimation and chi-square statistics appli=
ed
to counting experiments. NIM A 457 (2001) 384-401.</span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><span
style=3D'mso-spacerun:yes'>&nbsp;</span>[5<span class=3DGramE>]<span
style=3D'mso-spacerun:yes'>&nbsp; </span>M</span>. <span class=3DSpellE>Mor=
h&aacute;&#269;</span>,<span
style=3D'mso-spacerun:yes'>&nbsp; </span>J. <span class=3DSpellE>Kliman</sp=
an>,<span
style=3D'mso-spacerun:yes'>&nbsp; </span>M. <span class=3DSpellE>Jandel</sp=
an>,<span
style=3D'mso-spacerun:yes'>&nbsp; </span>&#317;. <span class=3DSpellE>Krupa=
</span>,
V. <span class=3DSpellE>Matou&#353;ek</span>: Study of fitting algorithms a=
pplied
to simultaneous analysis of large number of peaks in -ray spectra. </span><=
span
lang=3DEN-GB style=3D'mso-ansi-language:EN-GB'>Applied Spectroscopy, Vol. 5=
7, No.
7, pp. 753-760, 2003</span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><span
style=3D'mso-spacerun:yes'>&nbsp;</span></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span class=3DGramE><i><s=
pan
lang=3DEN-US>Example<span style=3D'mso-spacerun:yes'>&nbsp; </span>&#8211;<=
/span></i></span><i><span
lang=3DEN-US> script <span class=3DSpellE>FitAwmi.c</span>:<o:p></o:p></spa=
n></i></p>

<p class=3DMsoNormal style=3D'text-align:justify'><i><span lang=3DEN-US
style=3D'font-size:16.0pt'><!--[if gte vml 1]><v:shape id=3D"_x0000_i1025" =
type=3D"#_x0000_t75"
 style=3D'width:450.75pt;height:301.5pt'>
 <v:imagedata src=3D"FitAwmi_files/image003.jpg" o:title=3D"Fit1"/>
</v:shape><![endif]--><![if !vml]><img border=3D0 width=3D601 height=3D402
src=3D"FitAwmi_files/image004.jpg" v:shapes=3D"_x0000_i1025"><![endif]><o:p=
></o:p></span></i></p>

<p class=3DMsoNormal style=3D'text-align:justify'><b><span lang=3DEN-US>Fig=
. 1
Original spectrum (black line) and fitted spectrum using AWMI algorithm (red
line) and number of iteration steps =3D 1000. Positions of fitted peaks are
denoted by markers<o:p></o:p></span></b></p>

<p class=3DMsoNormal><b style=3D'mso-bidi-font-weight:normal'><span lang=3D=
EN-US
style=3D'color:#339966'>Script:<o:p></o:p></span></b></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'>// Examp=
le to
illustrate fitting function using AWMI algorithm.<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'>// <span
class=3DGramE>To</span> execute this example, do<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'>// root =
&gt; .x <span
class=3DSpellE>FitAwmi.C</span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><o:p>&nbsp;</o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'>void FitAwmi() {<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>Double_t a;<o:p></=
o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp; </span><span
style=3D'mso-spacerun:yes'>&nbsp;</span>Int_t i,nfound=3D0,bin;<o:p></o:p><=
/span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>Int_t nbins =3D 25=
6;<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>Int_t xmin<span
style=3D'mso-spacerun:yes'>&nbsp; </span>=3D 0;<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>Int_t xmax<span
style=3D'mso-spacerun:yes'>&nbsp; </span>=3D nbins;<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>Float_t * source =
=3D new
float[nbins];<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>Float_t * dest =3D=
 new
float[nbins];<span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><o:p></o:=
p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>TH1F *h =3D new
TH1F(&quot;h&quot;,&quot;Fitting using AWMI algorithm&quot;,nbins,xmin,xmax=
);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>TH1F *d =3D new
TH1F(&quot;d&quot;,&quot;&quot;,nbins,xmin,xmax);<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span><o:p></o:p=
></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>TFile *f =3D new
TFile(&quot;TSpectrum.root&quot;);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>h=3D(TH1F*)
f-&gt;Get(&quot;fit;1&quot;);<span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;
</span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>for (i =3D 0; i &l=
t;
nbins; i++) source[i]=3Dh-&gt;GetBinContent(i + 1);<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span><o:p></o:p=
></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>TCanvas *Fit1 =3D
gROOT-&gt;GetListOfCanvases()-&gt;FindObject(&quot;Fit1&quot;);<o:p></o:p><=
/span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>if (!Fit1) Fit1 =
=3D new
TCanvas(&quot;Fit1&quot;,&quot;Fit1&quot;,10,10,1000,700);<o:p></o:p></span=
></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;
</span>h-&gt;Draw(&quot;L&quot;);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>TSpectrum *s =3D n=
ew
TSpectrum();<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>//searching for
candidate peaks positions<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>nfound =3D
s-&gt;SearchHighRes(source, dest, nbins, 2, 0.1, kFALSE, 10000, kFALSE, 0);=
<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>Bool_t *FixPos =3D=
 new
Bool_t[nfound];<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>Bool_t *FixAmp =3D=
 new
Bool_t[nfound];<span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nb=
sp;
</span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>for(i =3D 0; i&lt;=
 nfound
; i++){<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>FixPos[i] =3D kFALSE;<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>FixAmp[i] =3D kFALSE;<span style=3D'mso-tab-count:1'>&nbsp;&nbsp;&nb=
sp; </span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>}<o:p></o:p></span=
></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>//filling in the i=
nitial
estimates of the input parameters<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>Float_t *PosX =3D =
new
Float_t[nfound];<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>Float_t *PosY =3D =
new
Float_t[nfound];<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>PosX =3D
s-&gt;GetPositionX();<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>for (i =3D 0; i &l=
t;
nfound; i++) {<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span style=3D'mso=
-tab-count:
1'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
 </span><span
style=3D'mso-spacerun:yes'>&nbsp;</span><span style=3D'mso-tab-count:2'>&nb=
sp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nb=
sp;&nbsp;&nbsp;&nbsp;&nbsp; </span>a=3DPosX[i];<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&n=
bsp;
</span>bin =3D 1 + Int_t(a + 0.5);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&n=
bsp;
</span>PosY[i] =3D h-&gt;GetBinContent(bin);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>}<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>TSpectrumFit *pfit=
=3Dnew
TSpectrumFit(nfound);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;
</span>pfit-&gt;SetFitParameters(xmin, xmax-1, 1000, 0.1,
pfit-&gt;kFitOptimChiCounts, pfit-&gt;kFitAlphaHalving, pfit-&gt;kFitPower2,
pfit-&gt;kFitTaylorOrderFirst);<span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;
</span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;
</span>pfit-&gt;SetPeakParameters(2, kFALSE, PosX, (Bool_t *) FixPos, PosY,
(Bool_t *) FixAmp);<span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><o:=
p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;
</span>pfit-&gt;FitAwmi(source);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>Double_t *CalcPosi=
tions
=3D new Double_t[nfound];<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span><o:p></o:p=
></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>Double_t *CalcAmpl=
itudes
=3D new Double_t[nfound];<span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;
</span><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</spa=
n><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;
</span>CalcPositions=3Dpfit-&gt;GetPositions();<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;
</span>CalcAmplitudes=3Dpfit-&gt;GetAmplitudes();<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>for (i =3D 0; i &l=
t;
nbins; i++) d-&gt;SetBinContent(i + 1,source[i]);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;
</span>d-&gt;SetLineColor(kRed);<span style=3D'mso-spacerun:yes'>&nbsp;&nbs=
p;
</span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>d-&gt;Draw(&quot;S=
AME
L&quot;);<span style=3D'mso-spacerun:yes'>&nbsp; </span><o:p></o:p></span><=
/p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>for (i =3D 0; i &l=
t;
nfound; i++) {<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span style=3D'mso=
-tab-count:
1'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
 </span><span
style=3D'mso-spacerun:yes'>&nbsp;</span><span style=3D'mso-tab-count:2'>&nb=
sp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nb=
sp;&nbsp;&nbsp;&nbsp;&nbsp; </span>a=3DCalcPositions[i];<o:p></o:p></span><=
/p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&n=
bsp;
</span>bin =3D 1 + Int_t(a + 0.5);<span style=3D'mso-spacerun:yes'>&nbsp;&n=
bsp;
</span><span style=3D'mso-tab-count:1'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&n=
bsp;
</span>PosX[i] =3D d-&gt;GetBinCenter(bin);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&n=
bsp;
</span>PosY[i] =3D d-&gt;GetBinContent(bin);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>}<o:p></o:p></span=
></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>TPolyMarker * pm =
=3D
(TPolyMarker*)h-&gt;GetListOfFunctions()-&gt;FindObject(&quot;TPolyMarker&q=
uot;);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>if (pm) {<o:p></o:=
p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>h-&gt;GetListOfFunctions()-&gt;Remove(pm);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span>=
delete
pm;<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>}<o:p></o:p></span=
></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>pm =3D new
TPolyMarker(nfound, PosX, PosY);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;
</span>h-&gt;GetListOfFunctions()-&gt;Add(pm);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;
</span>pm-&gt;SetMarkerStyle(23);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;
</span>pm-&gt;SetMarkerColor(kRed);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'><span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;
</span>pm-&gt;SetMarkerSize(1);<span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;
</span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DFR style=3D'font-size:10.0pt;mso-ansi-lan=
guage:
FR'>}<o:p></o:p></span></p>

</div>

<!-- */
// --> End_Html

   Int_t i, j, k, shift =
       2 * fNPeaks + 7, peak_vel, rozmer, iter, pw, regul_cycle,
       flag;
   Double_t a, b, c, d = 0, alpha, chi_opt, yw, ywm, f, chi2, chi_min, chi =
       0, pi, pmin = 0, chi_cel = 0, chi_er;
   Double_t *working_space = new Double_t[5 * (2 * fNPeaks + 7)];
   for (i = 0, j = 0; i < fNPeaks; i++) {
      working_space[2 * i] = fAmpInit[i];        //vector parameter
      if (fFixAmp[i] == false) {
         working_space[shift + j] = fAmpInit[i];        //vector xk
         j += 1;
      }
      working_space[2 * i + 1] = fPositionInit[i];        //vector parameter
      if (fFixPosition[i] == false) {
         working_space[shift + j] = fPositionInit[i];        //vector xk
         j += 1;
      }
   }
   peak_vel = 2 * i;
   working_space[2 * i] = fSigmaInit;        //vector parameter
   if (fFixSigma == false) {
      working_space[shift + j] = fSigmaInit;        //vector xk
      j += 1;
   }
   working_space[2 * i + 1] = fTInit;        //vector parameter
   if (fFixT == false) {
      working_space[shift + j] = fTInit;        //vector xk
      j += 1;
   }
   working_space[2 * i + 2] = fBInit;        //vector parameter
   if (fFixB == false) {
      working_space[shift + j] = fBInit;        //vector xk
      j += 1;
   }
   working_space[2 * i + 3] = fSInit;        //vector parameter
   if (fFixS == false) {
      working_space[shift + j] = fSInit;        //vector xk
      j += 1;
   }
   working_space[2 * i + 4] = fA0Init;        //vector parameter
   if (fFixA0 == false) {
      working_space[shift + j] = fA0Init;        //vector xk
      j += 1;
   }
   working_space[2 * i + 5] = fA1Init;        //vector parameter
   if (fFixA1 == false) {
      working_space[shift + j] = fA1Init;        //vector xk
      j += 1;
   }
   working_space[2 * i + 6] = fA2Init;        //vector parameter
   if (fFixA2 == false) {
      working_space[shift + j] = fA2Init;        //vector xk
      j += 1;
   }
   rozmer = j;
   if (rozmer == 0){
      Error ("FitAwmi","All parameters are fixed");   
      return;
   }
   if (rozmer >= fXmax - fXmin + 1){
      Error ("FitAwmi","Number of fitted parameters is larger than # of fitted points");   
      return;    
   }
   for (iter = 0; iter < fNumberIterations; iter++) {
      for (j = 0; j < rozmer; j++) {
         working_space[2 * shift + j] = 0, working_space[3 * shift + j] = 0;        //der,temp
      }
      
          //filling vectors
      alpha = fAlpha;
      chi_opt = 0, pw = fPower - 2;
      for (i = fXmin; i <= fXmax; i++) {
         yw = source[i];
         ywm = yw;
         f = Shape(fNPeaks, (Double_t) i, working_space,
                    working_space[peak_vel], working_space[peak_vel + 1],
                    working_space[peak_vel + 3],
                    working_space[peak_vel + 2],
                    working_space[peak_vel + 4],
                    working_space[peak_vel + 5],
                    working_space[peak_vel + 6]);
         if (fStatisticType == kFitOptimMaxLikelihood) {
            if (f > 0.00001)
               chi_opt += yw * TMath::Log(f) - f;
         }
         
         else {
            if (ywm != 0)
               chi_opt += (yw - f) * (yw - f) / ywm;
         }
         if (fStatisticType == kFitOptimChiFuncValues) {
            ywm = f;
            if (f < 0.00001)
               ywm = 0.00001;
         }
         
         else if (fStatisticType == kFitOptimMaxLikelihood) {
            ywm = f;
            if (f < 0.001)
               ywm = 0.001;
         }
         
         else {
            if (ywm == 0)
               ywm = 1;
         }
         
             //calculation of gradient vector
             for (j = 0, k = 0; j < fNPeaks; j++) {
            if (fFixAmp[j] == false) {
               a = Deramp((Double_t) i, working_space[2 * j + 1],
                           working_space[peak_vel],
                           working_space[peak_vel + 1],
                           working_space[peak_vel + 3],
                           working_space[peak_vel + 2]);
               if (ywm != 0) {
                  c = Ourpowl(a, pw);
                  if (fStatisticType == kFitOptimChiFuncValues) {
                     b = a * (yw * yw - f * f) / (ywm * ywm);
                     working_space[2 * shift + k] += b * c;        //der
                     b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                     working_space[3 * shift + k] += b * c;        //temp
                  }
                  
                  else {
                     b = a * (yw - f) / ywm;
                     working_space[2 * shift + k] += b * c;        //der
                     b = a * a / ywm;
                     working_space[3 * shift + k] += b * c;        //temp
                  }
               }
               k += 1;
            }
            if (fFixPosition[j] == false) {
               a = Deri0((Double_t) i, working_space[2 * j],
                          working_space[2 * j + 1],
                          working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 3],
                          working_space[peak_vel + 2]);
               if (fFitTaylor == kFitTaylorOrderSecond)
                  d = Derderi0((Double_t) i, working_space[2 * j],
                                working_space[2 * j + 1],
                                working_space[peak_vel]);
               if (ywm != 0) {
                  c = Ourpowl(a, pw);
                  if (TMath::Abs(a) > 0.00000001
                       && fFitTaylor == kFitTaylorOrderSecond) {
                     d = d * TMath::Abs(yw - f) / (2 * a * ywm);
                     if ((a + d) <= 0 && a >= 0 || (a + d) >= 0 && a <= 0)
                        d = 0;
                  }
                  
                  else
                     d = 0;
                  a = a + d;
                  if (fStatisticType == kFitOptimChiFuncValues) {
                     b = a * (yw * yw - f * f) / (ywm * ywm);
                     working_space[2 * shift + k] += b * c;        //der
                     b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                     working_space[3 * shift + k] += b * c;        //temp
                  }
                  
                  else {
                     b = a * (yw - f) / ywm;
                     working_space[2 * shift + k] += b * c;        //der
                     b = a * a / ywm;
                     working_space[3 * shift + k] += b * c;        //temp
                  }
               }
               k += 1;
            }
         }
         if (fFixSigma == false) {
            a = Dersigma(fNPeaks, (Double_t) i, working_space,
                          working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 3],
                          working_space[peak_vel + 2]);
            if (fFitTaylor == kFitTaylorOrderSecond)
               d = Derdersigma(fNPeaks, (Double_t) i,
                                working_space, working_space[peak_vel]);
            if (ywm != 0) {
               c = Ourpowl(a, pw);
               if (TMath::Abs(a) > 0.00000001
                    && fFitTaylor == kFitTaylorOrderSecond) {
                  d = d * TMath::Abs(yw - f) / (2 * a * ywm);
                  if ((a + d) <= 0 && a >= 0 || (a + d) >= 0 && a <= 0)
                     d = 0;
               }
               
               else
                  d = 0;
               a = a + d;
               if (fStatisticType == kFitOptimChiFuncValues) {
                  b = a * (yw * yw - f * f) / (ywm * ywm);
                  working_space[2 * shift + k] += b * c;        //der
                  b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                  working_space[3 * shift + k] += b * c;        //temp
               }
               
               else {
                  b = a * (yw - f) / ywm;
                  working_space[2 * shift + k] += b * c;        //der
                  b = a * a / ywm;
                  working_space[3 * shift + k] += b * c;        //temp
               }
            }
            k += 1;
         }
         if (fFixT == false) {
            a = Dert(fNPeaks, (Double_t) i, working_space,
                      working_space[peak_vel],
                      working_space[peak_vel + 2]);
            if (ywm != 0) {
               c = Ourpowl(a, pw);
               if (fStatisticType == kFitOptimChiFuncValues) {
                  b = a * (yw * yw - f * f) / (ywm * ywm);
                  working_space[2 * shift + k] += b * c;        //der
                  b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                  working_space[3 * shift + k] += b * c;        //temp
               }
               
               else {
                  b = a * (yw - f) / ywm;
                  working_space[2 * shift + k] += b * c;        //der
                  b = a * a / ywm;
                  working_space[3 * shift + k] += b * c;        //temp
               }
            }
            k += 1;
         }
         if (fFixB == false) {
            a = Derb(fNPeaks, (Double_t) i, working_space,
                      working_space[peak_vel], working_space[peak_vel + 1],
                      working_space[peak_vel + 2]);
            if (ywm != 0) {
               c = Ourpowl(a, pw);
               if (fStatisticType == kFitOptimChiFuncValues) {
                  b = a * (yw * yw - f * f) / (ywm * ywm);
                  working_space[2 * shift + k] += b * c;        //der
                  b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                  working_space[3 * shift + k] += b * c;        //temp
               }
               
               else {
                  b = a * (yw - f) / ywm;
                  working_space[2 * shift + k] += b * c;        //der
                  b = a * a / ywm;
                  working_space[3 * shift + k] += b * c;        //temp
               }
            }
            k += 1;
         }
         if (fFixS == false) {
            a = Ders(fNPeaks, (Double_t) i, working_space,
                      working_space[peak_vel]);
            if (ywm != 0) {
               c = Ourpowl(a, pw);
               if (fStatisticType == kFitOptimChiFuncValues) {
                  b = a * (yw * yw - f * f) / (ywm * ywm);
                  working_space[2 * shift + k] += b * c;        //der
                  b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                  working_space[3 * shift + k] += b * c;        //temp
               }
               
               else {
                  b = a * (yw - f) / ywm;
                  working_space[2 * shift + k] += b * c;        //der
                  b = a * a / ywm;
                  working_space[3 * shift + k] += b * c;        //temp
               }
            }
            k += 1;
         }
         if (fFixA0 == false) {
            a = 1.;
            if (ywm != 0) {
               c = Ourpowl(a, pw);
               if (fStatisticType == kFitOptimChiFuncValues) {
                  b = a * (yw * yw - f * f) / (ywm * ywm);
                  working_space[2 * shift + k] += b * c;        //der
                  b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                  working_space[3 * shift + k] += b * c;        //temp
               }
               
               else {
                  b = a * (yw - f) / ywm;
                  working_space[2 * shift + k] += b * c;        //der
                  b = a * a / ywm;
                  working_space[3 * shift + k] += b * c;        //temp
               }
            }
            k += 1;
         }
         if (fFixA1 == false) {
            a = Dera1((Double_t) i);
            if (ywm != 0) {
               c = Ourpowl(a, pw);
               if (fStatisticType == kFitOptimChiFuncValues) {
                  b = a * (yw * yw - f * f) / (ywm * ywm);
                  working_space[2 * shift + k] += b * c;        //der
                  b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                  working_space[3 * shift + k] += b * c;        //temp
               }
               
               else {
                  b = a * (yw - f) / ywm;
                  working_space[2 * shift + k] += b * c;        //der
                  b = a * a / ywm;
                  working_space[3 * shift + k] += b * c;        //temp
               }
            }
            k += 1;
         }
         if (fFixA2 == false) {
            a = Dera2((Double_t) i);
            if (ywm != 0) {
               c = Ourpowl(a, pw);
               if (fStatisticType == kFitOptimChiFuncValues) {
                  b = a * (yw * yw - f * f) / (ywm * ywm);
                  working_space[2 * shift + k] += b * c;        //der
                  b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                  working_space[3 * shift + k] += b * c;        //temp
               }
               
               else {
                  b = a * (yw - f) / ywm;
                  working_space[2 * shift + k] += b * c;        //der
                  b = a * a / ywm;
                  working_space[3 * shift + k] += b * c;        //temp
               }
            }
            k += 1;
         }
      }
      for (j = 0; j < rozmer; j++) {
         if (TMath::Abs(working_space[3 * shift + j]) > 0.000001)
            working_space[2 * shift + j] = working_space[2 * shift + j] / TMath::Abs(working_space[3 * shift + j]);        //der[j]=der[j]/temp[j]
         else
            working_space[2 * shift + j] = 0;        //der[j]
      }
      
      //calculate chi_opt
      chi2 = chi_opt;
      chi_opt = TMath::Sqrt(TMath::Abs(chi_opt));
      
      //calculate new parameters
      regul_cycle = 0;
      for (j = 0; j < rozmer; j++) {
         working_space[4 * shift + j] = working_space[shift + j];        //temp_xk[j]=xk[j]
      }
      
      do {
         if (fAlphaOptim == kFitAlphaOptimal) {
            if (fStatisticType != kFitOptimMaxLikelihood)
               chi_min = 10000 * chi2;
            
            else
               chi_min = 0.1 * chi2;
            flag = 0;
            for (pi = 0.1; flag == 0 && pi <= 100; pi += 0.1) {
               for (j = 0; j < rozmer; j++) {
                  working_space[shift + j] = working_space[4 * shift + j] + pi * alpha * working_space[2 * shift + j];        //xk[j]=temp_xk[j]+pi*alpha*der[j]
               }
               for (i = 0, j = 0; i < fNPeaks; i++) {
                  if (fFixAmp[i] == false) {
                     if (working_space[shift + j] < 0)        //xk[j]
                        working_space[shift + j] = 0;        //xk[j]
                     working_space[2 * i] = working_space[shift + j];        //parameter[2*i]=xk[j]
                     j += 1;
                  }
                  if (fFixPosition[i] == false) {
                     if (working_space[shift + j] < fXmin)        //xk[j]
                        working_space[shift + j] = fXmin;        //xk[j]
                     if (working_space[shift + j] > fXmax)        //xk[j]
                        working_space[shift + j] = fXmax;        //xk[j]
                     working_space[2 * i + 1] = working_space[shift + j];        //parameter[2*i+1]=xk[j]
                     j += 1;
                  }
               }
               if (fFixSigma == false) {
                  if (working_space[shift + j] < 0.001) {        //xk[j]
                     working_space[shift + j] = 0.001;        //xk[j]
                  }
                  working_space[peak_vel] = working_space[shift + j];        //parameter[peak_vel]=xk[j]
                  j += 1;
               }
               if (fFixT == false) {
                  working_space[peak_vel + 1] = working_space[shift + j];        //parameter[peak_vel+1]=xk[j]
                  j += 1;
               }
               if (fFixB == false) {
                  if (TMath::Abs(working_space[shift + j]) < 0.001) {        //xk[j]
                     if (working_space[shift + j] < 0)        //xk[j]
                        working_space[shift + j] = -0.001;        //xk[j]
                     else
                        working_space[shift + j] = 0.001;        //xk[j]
                  }
                  working_space[peak_vel + 2] = working_space[shift + j];        //parameter[peak_vel+2]=xk[j]
                  j += 1;
               }
               if (fFixS == false) {
                  working_space[peak_vel + 3] = working_space[shift + j];        //parameter[peak_vel+3]=xk[j]
                  j += 1;
               }
               if (fFixA0 == false) {
                  working_space[peak_vel + 4] = working_space[shift + j];        //parameter[peak_vel+4]=xk[j]
                  j += 1;
               }
               if (fFixA1 == false) {
                  working_space[peak_vel + 5] = working_space[shift + j];        //parameter[peak_vel+5]=xk[j]
                  j += 1;
               }
               if (fFixA2 == false) {
                  working_space[peak_vel + 6] = working_space[shift + j];        //parameter[peak_vel+6]=xk[j]
                  j += 1;
               }
               chi2 = 0;
               for (i = fXmin; i <= fXmax; i++) {
                  yw = source[i];
                  ywm = yw;
                  f = Shape(fNPeaks, (Double_t) i, working_space,
                  working_space[peak_vel],
                  working_space[peak_vel + 1],
                  working_space[peak_vel + 3],
                  working_space[peak_vel + 2],
                  working_space[peak_vel + 4],
                  working_space[peak_vel + 5],
                  working_space[peak_vel + 6]);
                  if (fStatisticType == kFitOptimChiFuncValues) {
                     ywm = f;
                     if (f < 0.00001)
                        ywm = 0.00001;
                  }
                  if (fStatisticType == kFitOptimMaxLikelihood) {
                     if (f > 0.00001)
                        chi2 += yw * TMath::Log(f) - f;
                  }
                  
                  else {
                     if (ywm != 0)
                        chi2 += (yw - f) * (yw - f) / ywm;
                  }
               }
               if (chi2 < chi_min
                    && fStatisticType != kFitOptimMaxLikelihood
                    || chi2 > chi_min
                    && fStatisticType == kFitOptimMaxLikelihood) {
                  pmin = pi, chi_min = chi2;
               }
               
               else
                  flag = 1;
               if (pi == 0.1)
                  chi_min = chi2;
               chi = chi_min;
            }
            if (pmin != 0.1) {
               for (j = 0; j < rozmer; j++) {
                  working_space[shift + j] = working_space[4 * shift + j] + pmin * alpha * working_space[2 * shift + j];        //xk[j]=temp_xk[j]+pmin*alpha*der[j]
               }
               for (i = 0, j = 0; i < fNPeaks; i++) {
                  if (fFixAmp[i] == false) {
                     if (working_space[shift + j] < 0)        //xk[j]
                        working_space[shift + j] = 0;        //xk[j]
                     working_space[2 * i] = working_space[shift + j];        //parameter[2*i]=xk[j]
                     j += 1;
                  }
                  if (fFixPosition[i] == false) {
                     if (working_space[shift + j] < fXmin)        //xk[j]
                        working_space[shift + j] = fXmin;        //xk[j]
                     if (working_space[shift + j] > fXmax)        //xk[j]
                        working_space[shift + j] = fXmax;        //xk[j]
                     working_space[2 * i + 1] = working_space[shift + j];        //parameter[2*i+1]=xk[j]
                     j += 1;
                  }
               }
               if (fFixSigma == false) {
                  if (working_space[shift + j] < 0.001) {        //xk[j]
                     working_space[shift + j] = 0.001;        //xk[j]
                  }
                  working_space[peak_vel] = working_space[shift + j];        //parameter[peak_vel]=xk[j]
                  j += 1;
               }
               if (fFixT == false) {
                  working_space[peak_vel + 1] = working_space[shift + j];        //parameter[peak_vel+1]=xk[j]
                  j += 1;
               }
               if (fFixB == false) {
                  if (TMath::Abs(working_space[shift + j]) < 0.001) {        //xk[j]
                     if (working_space[shift + j] < 0)        //xk[j]
                        working_space[shift + j] = -0.001;        //xk[j]
                     else
                        working_space[shift + j] = 0.001;        //xk[j]
                  }
                  working_space[peak_vel + 2] = working_space[shift + j];        //parameter[peak_vel+2]=xk[j]
                  j += 1;
               }
               if (fFixS == false) {
                  working_space[peak_vel + 3] = working_space[shift + j];        //parameter[peak_vel+3]=xk[j]
                  j += 1;
               }
               if (fFixA0 == false) {
                  working_space[peak_vel + 4] = working_space[shift + j];        //parameter[peak_vel+4]=xk[j]
                  j += 1;
               }
               if (fFixA1 == false) {
                  working_space[peak_vel + 5] = working_space[shift + j];        //parameter[peak_vel+5]=xk[j]
                  j += 1;
               }
               if (fFixA2 == false) {
                  working_space[peak_vel + 6] = working_space[shift + j];        //parameter[peak_vel+6]=xk[j]
                  j += 1;
               }
               chi = chi_min;
            }
         }
         
         else {
            for (j = 0; j < rozmer; j++) {
               working_space[shift + j] = working_space[4 * shift + j] + alpha * working_space[2 * shift + j];        //xk[j]=temp_xk[j]+pi*alpha*der[j]
            }
            for (i = 0, j = 0; i < fNPeaks; i++) {
               if (fFixAmp[i] == false) {
                  if (working_space[shift + j] < 0)        //xk[j]
                     working_space[shift + j] = 0;        //xk[j]
                  working_space[2 * i] = working_space[shift + j];        //parameter[2*i]=xk[j]
                  j += 1;
               }
               if (fFixPosition[i] == false) {
                  if (working_space[shift + j] < fXmin)        //xk[j]
                     working_space[shift + j] = fXmin;        //xk[j]
                  if (working_space[shift + j] > fXmax)        //xk[j]
                     working_space[shift + j] = fXmax;        //xk[j]
                  working_space[2 * i + 1] = working_space[shift + j];        //parameter[2*i+1]=xk[j]
                  j += 1;
               }
            }
            if (fFixSigma == false) {
               if (working_space[shift + j] < 0.001) {        //xk[j]
                  working_space[shift + j] = 0.001;        //xk[j]
               }
               working_space[peak_vel] = working_space[shift + j];        //parameter[peak_vel]=xk[j]
               j += 1;
            }
            if (fFixT == false) {
               working_space[peak_vel + 1] = working_space[shift + j];        //parameter[peak_vel+1]=xk[j]
               j += 1;
            }
            if (fFixB == false) {
               if (TMath::Abs(working_space[shift + j]) < 0.001) {        //xk[j]
                  if (working_space[shift + j] < 0)        //xk[j]
                     working_space[shift + j] = -0.001;        //xk[j]
                  else
                     working_space[shift + j] = 0.001;        //xk[j]
               }
               working_space[peak_vel + 2] = working_space[shift + j];        //parameter[peak_vel+2]=xk[j]
               j += 1;
            }
            if (fFixS == false) {
               working_space[peak_vel + 3] = working_space[shift + j];        //parameter[peak_vel+3]=xk[j]
               j += 1;
            }
            if (fFixA0 == false) {
               working_space[peak_vel + 4] = working_space[shift + j];        //parameter[peak_vel+4]=xk[j]
               j += 1;
            }
            if (fFixA1 == false) {
               working_space[peak_vel + 5] = working_space[shift + j];        //parameter[peak_vel+5]=xk[j]
               j += 1;
            }
            if (fFixA2 == false) {
               working_space[peak_vel + 6] = working_space[shift + j];        //parameter[peak_vel+6]=xk[j]
               j += 1;
            }
            chi = 0;
            for (i = fXmin; i <= fXmax; i++) {
               yw = source[i];
               ywm = yw;
               f = Shape(fNPeaks, (Double_t) i, working_space,
               working_space[peak_vel],
               working_space[peak_vel + 1],
               working_space[peak_vel + 3],
               working_space[peak_vel + 2],
               working_space[peak_vel + 4],
               working_space[peak_vel + 5],
               working_space[peak_vel + 6]);
               if (fStatisticType == kFitOptimChiFuncValues) {
                  ywm = f;
                  if (f < 0.00001)
                     ywm = 0.00001;
               }
               if (fStatisticType == kFitOptimMaxLikelihood) {
                  if (f > 0.00001)
                     chi += yw * TMath::Log(f) - f;
               }
               
               else {
                  if (ywm != 0)
                     chi += (yw - f) * (yw - f) / ywm;
               }
            }
         }
         chi2 = chi;
         chi = TMath::Sqrt(TMath::Abs(chi));
         if (fAlphaOptim == kFitAlphaHalving && chi > 1E-6)
            alpha = alpha * chi_opt / (2 * chi);
         
         else if (fAlphaOptim == kFitAlphaOptimal)
            alpha = alpha / 10.0;
         iter += 1;
         regul_cycle += 1;
      } while ((chi > chi_opt
                 && fStatisticType != kFitOptimMaxLikelihood
                 || chi < chi_opt
                 && fStatisticType == kFitOptimMaxLikelihood)
                && regul_cycle < kFitNumRegulCycles);
      for (j = 0; j < rozmer; j++) {
         working_space[4 * shift + j] = 0;        //temp_xk[j]
         working_space[2 * shift + j] = 0;        //der[j]
      }
      for (i = fXmin, chi_cel = 0; i <= fXmax; i++) {
         yw = source[i];
         if (yw == 0)
            yw = 1;
         f = Shape(fNPeaks, (Double_t) i, working_space,
         working_space[peak_vel], working_space[peak_vel + 1],
         working_space[peak_vel + 3],
         working_space[peak_vel + 2],
         working_space[peak_vel + 4],
         working_space[peak_vel + 5],
         working_space[peak_vel + 6]);
         chi_opt = (yw - f) * (yw - f) / yw;
         chi_cel += (yw - f) * (yw - f) / yw;
         
             //calculate gradient vector
         for (j = 0, k = 0; j < fNPeaks; j++) {
            if (fFixAmp[j] == false) {
               a = Deramp((Double_t) i, working_space[2 * j + 1],
               working_space[peak_vel],
               working_space[peak_vel + 1],
               working_space[peak_vel + 3],
               working_space[peak_vel + 2]);
               if (yw != 0) {
                  c = Ourpowl(a, pw);
                  working_space[2 * shift + k] += chi_opt * c;        //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b * c;        //temp_xk[k]
               }
               k += 1;
            }
            if (fFixPosition[j] == false) {
               a = Deri0((Double_t) i, working_space[2 * j],
               working_space[2 * j + 1],
               working_space[peak_vel],
               working_space[peak_vel + 1],
               working_space[peak_vel + 3],
               working_space[peak_vel + 2]);
               if (yw != 0) {
                  c = Ourpowl(a, pw);
                  working_space[2 * shift + k] += chi_opt * c;        //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b * c;        //temp_xk[k]
               }
               k += 1;
            }
         }
         if (fFixSigma == false) {
            a = Dersigma(fNPeaks, (Double_t) i, working_space,
            working_space[peak_vel],
            working_space[peak_vel + 1],
            working_space[peak_vel + 3],
            working_space[peak_vel + 2]);
            if (yw != 0) {
               c = Ourpowl(a, pw);
               working_space[2 * shift + k] += chi_opt * c;        //der[k]
               b = a * a / yw;
               working_space[4 * shift + k] += b * c;        //temp_xk[k]
            }
            k += 1;
         }
         if (fFixT == false) {
            a = Dert(fNPeaks, (Double_t) i, working_space,
            working_space[peak_vel],
            working_space[peak_vel + 2]);
            if (yw != 0) {
               c = Ourpowl(a, pw);
               working_space[2 * shift + k] += chi_opt * c;        //der[k]
               b = a * a / yw;
               working_space[4 * shift + k] += b * c;        //temp_xk[k]
            }
            k += 1;
         }
         if (fFixB == false) {
            a = Derb(fNPeaks, (Double_t) i, working_space,
            working_space[peak_vel], working_space[peak_vel + 1],
            working_space[peak_vel + 2]);
            if (yw != 0) {
               c = Ourpowl(a, pw);
               working_space[2 * shift + k] += chi_opt * c;        //der[k]
               b = a * a / yw;
               working_space[4 * shift + k] += b * c;        //temp_xk[k]
            }
            k += 1;
         }
         if (fFixS == false) {
            a = Ders(fNPeaks, (Double_t) i, working_space,
            working_space[peak_vel]);
            if (yw != 0) {
               c = Ourpowl(a, pw);
               working_space[2 * shift + k] += chi_opt * c;        //der[k]
               b = a * a / yw;
               working_space[4 * shift + k] += b * c;        //tem_xk[k]
            }
            k += 1;
         }
         if (fFixA0 == false) {
            a = 1.0;
            if (yw != 0) {
               c = Ourpowl(a, pw);
               working_space[2 * shift + k] += chi_opt * c;        //der[k]
               b = a * a / yw;
               working_space[4 * shift + k] += b * c;        //temp_xk[k]
            }
            k += 1;
         }
         if (fFixA1 == false) {
            a = Dera1((Double_t) i);
            if (yw != 0) {
               c = Ourpowl(a, pw);
               working_space[2 * shift + k] += chi_opt * c;        //der[k]
               b = a * a / yw;
               working_space[4 * shift + k] += b * c;        //temp_xk[k]
            }
            k += 1;
         }
         if (fFixA2 == false) {
            a = Dera2((Double_t) i);
            if (yw != 0) {
               c = Ourpowl(a, pw);
               working_space[2 * shift + k] += chi_opt * c;        //der[k]
               b = a * a / yw;
               working_space[4 * shift + k] += b * c;        //temp_xk[k]
            }
            k += 1;
         }
      }
   }
   b = fXmax - fXmin + 1 - rozmer;
   chi_er = chi_cel / b;
   for (i = 0, j = 0; i < fNPeaks; i++) {
      fArea[i] =
          Area(working_space[2 * i], working_space[peak_vel],
               working_space[peak_vel + 1], working_space[peak_vel + 2]);
      if (fFixAmp[i] == false) {
         fAmpCalc[i] = working_space[shift + j];        //xk[j]
         if (working_space[3 * shift + j] != 0)
            fAmpErr[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));        //der[j]/temp[j]
         if (fArea[i] > 0) {
            a = Derpa(working_space[peak_vel],
                       working_space[peak_vel + 1],
                       working_space[peak_vel + 2]);
            b = working_space[4 * shift + j];        //temp_xk[j]
            if (b == 0)
               b = 1;
            
            else
               b = 1 / b;
            fAreaErr[i] = TMath::Sqrt(TMath::Abs(a * a * b * chi_er));
         }
         
         else
            fAreaErr[i] = 0;
         j += 1;
      }
      
      else {
         fAmpCalc[i] = fAmpInit[i];
         fAmpErr[i] = 0;
         fAreaErr[i] = 0;
      }
      if (fFixPosition[i] == false) {
         fPositionCalc[i] = working_space[shift + j];        //xk[j]
         if (working_space[3 * shift + j] != 0)        //temp[j]
            fPositionErr[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));        //der[j]/temp[j]
         j += 1;
      }
      
      else {
         fPositionCalc[i] = fPositionInit[i];
         fPositionErr[i] = 0;
      }
   }
   if (fFixSigma == false) {
      fSigmaCalc = working_space[shift + j];        //xk[j]
      if (working_space[3 * shift + j] != 0)        //temp[j]
         fSigmaErr = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));        //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fSigmaCalc = fSigmaInit;
      fSigmaErr = 0;
   }
   if (fFixT == false) {
      fTCalc = working_space[shift + j];        //xk[j]
      if (working_space[3 * shift + j] != 0)        //temp[j]
         fTErr = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));        //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fTCalc = fTInit;
      fTErr = 0;
   }
   if (fFixB == false) {
      fBCalc = working_space[shift + j];        //xk[j]
      if (working_space[3 * shift + j] != 0)        //temp[j]
         fBErr = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));        //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fBCalc = fBInit;
      fBErr = 0;
   }
   if (fFixS == false) {
      fSCalc = working_space[shift + j];        //xk[j]
      if (working_space[3 * shift + j] != 0)        //temp[j]
         fSErr = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));        //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fSCalc = fSInit;
      fSErr = 0;
   }
   if (fFixA0 == false) {
      fA0Calc = working_space[shift + j];        //xk[j]
      if (working_space[3 * shift + j] != 0)        //temp[j]
         fA0Err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));        //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fA0Calc = fA0Init;
      fA0Err = 0;
   }
   if (fFixA1 == false) {
      fA1Calc = working_space[shift + j];        //xk[j]
      if (working_space[3 * shift + j] != 0)        //temp[j]
         fA1Err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));        //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fA1Calc = fA1Init;
      fA1Err = 0;
   }
   if (fFixA2 == false) {
      fA2Calc = working_space[shift + j];        //xk[j]
      if (working_space[3 * shift + j] != 0)        //temp[j]
         fA2Err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));        //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fA2Calc = fA2Init;
      fA2Err = 0;
   }
   b = fXmax - fXmin + 1 - rozmer;
   fChi = chi_cel / b;
   for (i = fXmin; i <= fXmax; i++) {
      f = Shape(fNPeaks, (Double_t) i, working_space,
                 working_space[peak_vel], working_space[peak_vel + 1],
                 working_space[peak_vel + 3], working_space[peak_vel + 2],
                 working_space[peak_vel + 4], working_space[peak_vel + 5],
                 working_space[peak_vel + 6]);
      source[i] = f;
   }
   delete[]working_space;
   return;
}
    
/////////////////FITTING FUNCTION WITH MATRIX INVERSION///////////////////////////////////////

//_______________________________________________________________________________
void TSpectrumFit::StiefelInversion(Double_t **a, Int_t size)
{   
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates solution of the system of linear equations.        //
//   The matrix a should have a dimension size*(size+4)                         //
//   The calling function should fill in the matrix, the column size should     //
//   contain vector y (right side of the system of equations). The result is    //
//   placed into size+1 column of the matrix.                                   //
//   according to sigma of peaks.                                               //
//      Function parameters:                                                    //
//              -a-matrix with dimension size*(size+4)                          //                                            //
//              -size-number of rows of the matrix                              //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Int_t i, j, k = 0;
   Double_t sk = 0, b, lambdak, normk, normk_old = 0;
   
   do {
      normk = 0;
      
          //calculation of rk and norm
          for (i = 0; i < size; i++) {
         a[i][size + 2] = -a[i][size];        //rk=-C
         for (j = 0; j < size; j++) {
            a[i][size + 2] += a[i][j] * a[j][size + 1];        //A*xk-C
         }
         normk += a[i][size + 2] * a[i][size + 2];        //calculation normk
      }
      
      //calculation of sk
      if (k != 0) {
         sk = normk / normk_old;
      }
      
      //calculation of uk
      for (i = 0; i < size; i++) {
         a[i][size + 3] = -a[i][size + 2] + sk * a[i][size + 3];        //uk=-rk+sk*uk-1
      }
      
      //calculation of lambdak
      lambdak = 0;
      for (i = 0; i < size; i++) {
         for (j = 0, b = 0; j < size; j++) {
            b += a[i][j] * a[j][size + 3];        //A*uk
         }
         lambdak += b * a[i][size + 3];
      }
      if (TMath::Abs(lambdak) > 1e-50)        //computer zero
         lambdak = normk / lambdak;
      
      else
         lambdak = 0;
      for (i = 0; i < size; i++)
         a[i][size + 1] += lambdak * a[i][size + 3];        //xk+1=xk+lambdak*uk
      normk_old = normk;
      k += 1;
   } while (k < size && TMath::Abs(normk) > 1e-50);        //computer zero
   return;
}

//_______________________________________________________________________________
void TSpectrumFit::FitStiefel(Float_t *source) 
{
/////////////////////////////////////////////////////////////////////////////
//        ONE-DIMENSIONAL FIT FUNCTION                                    
//        ALGORITHM WITH MATRIX INVERSION (STIEFEL-HESTENS METHOD)            
//        This function fits the source spectrum. The calling program should 
//        fill in input parameters
//        The fitted parameters are written into           
//        output parameters and fitted data are written into 
//        source spectrum.                                                    
//                                                                          
//        Function parameters:                                             
//        source-pointer to the vector of source spectrum                  
//                                                                         
/////////////////////////////////////////////////////////////////////////////
//Begin_Html <!--
/* -->
<div class=3DSection2>

<p class=3DMsoNormal><span class=3DSpellE><b><span lang=3DEN-US style=3D'fo=
nt-size:
14.0pt'>Stiefel</span></b></span><b><span lang=3DEN-US style=3D'font-size:1=
4.0pt'>
fitting algorithm</span></b><span lang=3DEN-US style=3D'font-size:14.0pt'><=
o:p></o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><i><span lang=3DEN-US
style=3D'font-size:18.0pt'><o:p>&nbsp;</o:p></span></i></p>

<p class=3DMsoNormal><i><span lang=3DEN-US>Function:</span></i></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span class=3DGramE><span
lang=3DEN-US>void</span></span><span lang=3DEN-US> <a
href=3D"http://root.cern.ch/root/html/TSpectrum.html#TSpectrum:Fit1Awmi"><s=
pan
class=3DSpellE><b style=3D'mso-bidi-font-weight:normal'>TSpectrumFit::</b><=
/span></a><span
class=3DSpellE>FitStiefel</span>(<a
href=3D"http://root.cern.ch/root/html/ListOfTypes.html#float"><b
style=3D'mso-bidi-font-weight:normal'>float</b></a> *<span class=3DSpellE>f=
Source</span>)<span
style=3D'mso-bidi-font-weight:bold'> </span></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><o:p>&=
nbsp;</o:p></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US>This f=
unction
fits the source spectrum using <span class=3DSpellE>Stiefel-Hestens</span> =
method
[1] (see <span class=3DSpellE>Awmi</span> function). <span
style=3D'mso-spacerun:yes'>&nbsp;</span>The calling program should fill in =
input
fitting parameters of the <span class=3DSpellE>TSpectrumFit</span> class us=
ing a
set of <span class=3DSpellE>TSpectrumFit</span> setters. The fitted paramet=
ers
are written into the class and the fitted data are written into source
spectrum. It converges faster than <span class=3DSpellE>Awmi</span> method.=
</span></p>

<p class=3DMsoNormal><span lang=3DEN-US><o:p>&nbsp;</o:p></span></p>

<p class=3DMsoNormal><i style=3D'mso-bidi-font-style:normal'><span lang=3DE=
N-US
style=3D'color:red'>Parameter:<o:p></o:p></span></i></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </spa=
n><span
class=3DSpellE><span class=3DGramE><b style=3D'mso-bidi-font-weight:normal'=
>fSource</b></span></span><span
class=3DGramE>-pointer</span> to the vector of source spectrum<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><span lang=3DEN-US><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </spa=
n></span></p>

<p class=3DMsoNormal style=3D'text-align:justify'><i><span lang=3DEN-US>Exa=
mple
&#8211; script <span class=3DSpellE>FitStiefel.c</span>:<o:p></o:p></span><=
/i></p>

<p class=3DMsoNormal style=3D'text-align:justify'><i><span lang=3DEN-US
style=3D'font-size:16.0pt'><!--[if gte vml 1]><v:shapetype id=3D"_x0000_t75"
 coordsize=3D"21600,21600" o:spt=3D"75" o:preferrelative=3D"t" path=3D"m@4@=
5l@4@11@9@11@9@5xe"
 filled=3D"f" stroked=3D"f">
 <v:stroke joinstyle=3D"miter"/>
 <v:formulas>
  <v:f eqn=3D"if lineDrawn pixelLineWidth 0"/>
  <v:f eqn=3D"sum @0 1 0"/>
  <v:f eqn=3D"sum 0 0 @1"/>
  <v:f eqn=3D"prod @2 1 2"/>
  <v:f eqn=3D"prod @3 21600 pixelWidth"/>
  <v:f eqn=3D"prod @3 21600 pixelHeight"/>
  <v:f eqn=3D"sum @0 0 1"/>
  <v:f eqn=3D"prod @6 1 2"/>
  <v:f eqn=3D"prod @7 21600 pixelWidth"/>
  <v:f eqn=3D"sum @8 21600 0"/>
  <v:f eqn=3D"prod @7 21600 pixelHeight"/>
  <v:f eqn=3D"sum @10 21600 0"/>
 </v:formulas>
 <v:path o:extrusionok=3D"f" gradientshapeok=3D"t" o:connecttype=3D"rect"/>
 <o:lock v:ext=3D"edit" aspectratio=3D"t"/>
</v:shapetype><v:shape id=3D"_x0000_i1025" type=3D"#_x0000_t75" style=3D'wi=
dth:450.75pt;
 height:301.5pt'>
 <v:imagedata src=3D"FitStiefel_files/image001.jpg" o:title=3D"Fit1"/>
</v:shape><![endif]--><![if !vml]><img border=3D0 width=3D601 height=3D402
src=3D"FitStiefel_files/image002.jpg" v:shapes=3D"_x0000_i1025"><![endif]><=
o:p></o:p></span></i></p>

<p class=3DMsoNormal style=3D'text-align:justify'><b><span lang=3DEN-US>Fig=
. 2
Original spectrum (black line) and fitted spectrum using <span class=3DSpel=
lE>Stiefel-Hestens</span>
method (red line) and number of iteration steps =3D 100. Positions of fitted
peaks are denoted by markers</span></b></p>

<p class=3DMsoNormal><b style=3D'mso-bidi-font-weight:normal'><span lang=3D=
EN-US
style=3D'color:#339966'><o:p>&nbsp;</o:p></span></b></p>

<p class=3DMsoNormal><b style=3D'mso-bidi-font-weight:normal'><span lang=3D=
EN-US
style=3D'color:#339966'>Script:<o:p></o:p></span></b></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'>// Examp=
le to
illustrate fitting function using <span class=3DSpellE>Stiefel-Hestens</spa=
n>
method.<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'>// <span
class=3DGramE>To</span> execute this example, do<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'>// root =
&gt; .x <span
class=3DSpellE>FitStiefel.C</span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><o:p>&nb=
sp;</o:p></span></p>

<p class=3DMsoNormal><span class=3DGramE><span lang=3DEN-US style=3D'font-s=
ize:10.0pt'>void</span></span><span
lang=3DEN-US style=3D'font-size:10.0pt'> <span class=3DSpellE>FitStiefel</s=
pan>() {<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>Double_=
t</span>
a;<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp; </span><span
style=3D'mso-spacerun:yes'>&nbsp;</span><span class=3DSpellE>Int_t</span> <=
span
class=3DSpellE>i<span class=3DGramE>,nfound</span></span>=3D0,bin;<o:p></o:=
p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>Int_t</=
span> <span
class=3DSpellE>nbins</span> =3D 256;<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>Int_t</=
span> <span
class=3DSpellE><span class=3DGramE>xmin</span></span><span class=3DGramE><s=
pan
style=3D'mso-spacerun:yes'>&nbsp; </span>=3D</span> 0;<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>Int_t</=
span> <span
class=3DSpellE><span class=3DGramE>xmax</span></span><span class=3DGramE><s=
pan
style=3D'mso-spacerun:yes'>&nbsp; </span>=3D</span> <span class=3DSpellE>nb=
ins</span>;<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>Float_t=
</span>
* source =3D new <span class=3DGramE>float[</span><span class=3DSpellE>nbin=
s</span>];<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>Float_t=
</span>
* <span class=3DSpellE>dest</span> =3D new <span class=3DGramE>float[</span=
><span
class=3DSpellE>nbins</span>];<span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; =
</span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>TH1F *h =3D new <span class=
=3DGramE>TH1F(</span>&quot;<span
class=3DSpellE>h&quot;,&quot;Fitting</span> using AWMI <span class=3DSpellE=
>algorithm&quot;,nbins,xmin,xmax</span>);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>TH1F *d =3D new <span class=
=3DGramE>TH1F(</span>&quot;<span
class=3DSpellE>d&quot;,&quot;&quot;,nbins,xmin,xmax</span>);<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span><o:p></o:p=
></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>TFile</=
span> *f
=3D new <span class=3DSpellE><span class=3DGramE>TFile</span></span><span
class=3DGramE>(</span>&quot;<span class=3DSpellE>TSpectrum.root</span>&quot=
;);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>h<span class=3DGramE>=3D(</s=
pan>TH1F*)
f-&gt;Get(&quot;fit;1&quot;);<span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;
</span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>for</spa=
n> (<span
class=3DSpellE>i</span> =3D 0; <span class=3DSpellE>i</span> &lt; <span cla=
ss=3DSpellE>nbins</span>;
<span class=3DSpellE>i</span>++) source[<span class=3DSpellE>i</span>]=3Dh-=
&gt;<span
class=3DSpellE>GetBinContent</span>(<span class=3DSpellE>i</span> + 1);<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span><o:p></o:p=
></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>TCanvas=
</span>
*Fit1 =3D <span class=3DSpellE>gROOT</span>-&gt;<span class=3DSpellE><span
class=3DGramE>GetListOfCanvases</span></span><span class=3DGramE>(</span>)-=
&gt;<span
class=3DSpellE>FindObject</span>(&quot;Fit1&quot;);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>if</span>
(!Fit1) Fit1 =3D new <span class=3DSpellE>TCanvas</span>(&quot;Fit1&quot;,&=
quot;Fit1&quot;,10,10,1000,700);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>h</span>=
-&gt;Draw(&quot;L&quot;);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>TSpectr=
um</span>
*s =3D new <span class=3DSpellE><span class=3DGramE>TSpectrum</span></span>=
<span
class=3DGramE>(</span>);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>//searching for candidate pe=
aks
positions<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE><span
class=3DGramE>nfound</span></span> =3D s-&gt;<span class=3DSpellE>SearchHig=
hRes</span>(source,
<span class=3DSpellE>dest</span>, <span class=3DSpellE>nbins</span>, 2, 0.1=
, <span
class=3DSpellE>kFALSE</span>, 10000, <span class=3DSpellE>kFALSE</span>, 0)=
;<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>Bool_t<=
/span> *<span
class=3DSpellE>FixPos</span> =3D new <span class=3DSpellE>Bool_<span class=
=3DGramE>t</span></span><span
class=3DGramE>[</span><span class=3DSpellE>nfound</span>];<o:p></o:p></span=
></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>Bool_t<=
/span> *<span
class=3DSpellE>FixAmp</span> =3D new <span class=3DSpellE>Bool_<span class=
=3DGramE>t</span></span><span
class=3DGramE>[</span><span class=3DSpellE>nfound</span>];<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span><o:p></o:p=
></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>for(</sp=
an><span
class=3DSpellE>i</span> =3D 0; <span class=3DSpellE>i</span>&lt; <span clas=
s=3DSpellE>nfound</span>
; <span class=3DSpellE>i</span>++){<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span><span
class=3DSpellE>FixPos</span>[<span class=3DSpellE>i</span>] =3D <span class=
=3DSpellE>kFALSE</span>;<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span><span
class=3DSpellE>FixAmp</span>[<span class=3DSpellE>i</span>] =3D <span class=
=3DSpellE>kFALSE</span>;<span
style=3D'mso-tab-count:1'>&nbsp;&nbsp;&nbsp; </span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>}<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>//filling in the initial est=
imates
of the input parameters<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>Float_t=
</span>
*<span class=3DSpellE>PosX</span> =3D new <span class=3DSpellE>Float_<span
class=3DGramE>t</span></span><span class=3DGramE>[</span><span class=3DSpel=
lE>nfound</span>];<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>Float_t=
</span>
*<span class=3DSpellE>PosY</span> =3D new <span class=3DSpellE>Float_<span
class=3DGramE>t</span></span><span class=3DGramE>[</span><span class=3DSpel=
lE>nfound</span>];<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>PosX</s=
pan> =3D
s-&gt;<span class=3DSpellE><span class=3DGramE>GetPositionX</span></span><s=
pan
class=3DGramE>(</span>);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>for</spa=
n> (<span
class=3DSpellE>i</span> =3D 0; <span class=3DSpellE>i</span> &lt; <span cla=
ss=3DSpellE>nfound</span>;
<span class=3DSpellE>i</span>++) {<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span style=3D'mso-tab-count=
:1'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp=
; </span><span
style=3D'mso-spacerun:yes'>&nbsp;</span><span style=3D'mso-tab-count:2'>&nb=
sp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nb=
sp;&nbsp;&nbsp;&nbsp;&nbsp; </span>a=3D<span
class=3DSpellE>PosX</span>[<span class=3DSpellE>i</span>];<o:p></o:p></span=
></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </spa=
n><span
class=3DGramE>bin</span> =3D 1 + <span class=3DSpellE>Int_t</span>(a + 0.5)=
;<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </spa=
n><span
class=3DSpellE>PosY</span>[<span class=3DSpellE>i</span>] =3D h-&gt;<span
class=3DSpellE><span class=3DGramE>GetBinContent</span></span><span class=
=3DGramE>(</span>bin);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>}<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>TSpectr=
umFit</span>
*<span class=3DSpellE>pfit</span>=3Dnew <span class=3DSpellE><span class=3D=
GramE>TSpectrumFit</span></span><span
class=3DGramE>(</span><span class=3DSpellE>nfound</span>);<o:p></o:p></span=
></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE><span
class=3DGramE>pfit</span></span>-&gt;<span class=3DSpellE>SetFitParameters<=
/span>(<span
class=3DSpellE>xmin</span>, xmax-1, 1000, 0.1, <span class=3DSpellE>pfit</s=
pan>-&gt;<span
class=3DSpellE>kFitOptimChiCounts</span>, <span class=3DSpellE>pfit</span>-=
&gt;<span
class=3DSpellE>kFitAlphaHalving</span>, <span class=3DSpellE>pfit</span>-&g=
t;kFitPower2,
<span class=3DSpellE>pfit</span>-&gt;<span class=3DSpellE>kFitTaylorOrderFi=
rst</span>);<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE><span
class=3DGramE>pfit</span></span>-&gt;<span class=3DSpellE>SetPeakParameters=
</span>(2,
<span class=3DSpellE>kFALSE</span>, <span class=3DSpellE>PosX</span>, (<span
class=3DSpellE>Bool_t</span> *) <span class=3DSpellE>FixPos</span>, <span
class=3DSpellE>PosY</span>, (<span class=3DSpellE>Bool_t</span> *) <span
class=3DSpellE>FixAmp</span>);<span style=3D'mso-spacerun:yes'>&nbsp;&nbsp;=
 </span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE><span
class=3DGramE>pfit</span></span>-&gt;<span class=3DSpellE>FitStiefel</span>=
(source);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>Double_=
t</span>
*<span class=3DSpellE>CalcPositions</span> =3D new <span class=3DSpellE>Dou=
ble_<span
class=3DGramE>t</span></span><span class=3DGramE>[</span><span class=3DSpel=
lE>nfound</span>];<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span><o:p></o:p=
></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>Double_=
t</span>
*<span class=3DSpellE>CalcAmplitudes</span> =3D new <span class=3DSpellE>Do=
uble_<span
class=3DGramE>t</span></span><span class=3DGramE>[</span><span class=3DSpel=
lE>nfound</span>];<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>CalcPos=
itions</span>=3D<span
class=3DSpellE>pfit</span>-&gt;<span class=3DSpellE><span class=3DGramE>Get=
Positions</span></span><span
class=3DGramE>(</span>);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>CalcAmp=
litudes</span>=3D<span
class=3DSpellE>pfit</span>-&gt;<span class=3DSpellE><span class=3DGramE>Get=
Amplitudes</span></span><span
class=3DGramE>(</span>);<span style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </spa=
n><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>for</spa=
n> (<span
class=3DSpellE>i</span> =3D 0; <span class=3DSpellE>i</span> &lt; <span cla=
ss=3DSpellE>nbins</span>;
<span class=3DSpellE>i</span>++) d-&gt;<span class=3DSpellE>SetBinContent</=
span>(<span
class=3DSpellE>i</span> + 1,source[<span class=3DSpellE>i</span>]);<o:p></o=
:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>d</span>=
-&gt;<span
class=3DSpellE>SetLineColor</span>(<span class=3DSpellE>kRed</span>);<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>d</span>=
-&gt;Draw(&quot;SAME
L&quot;);<span style=3D'mso-spacerun:yes'>&nbsp; </span><o:p></o:p></span><=
/p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>for</spa=
n> (<span
class=3DSpellE>i</span> =3D 0; <span class=3DSpellE>i</span> &lt; <span cla=
ss=3DSpellE>nfound</span>;
<span class=3DSpellE>i</span>++) {<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span style=3D'mso-tab-count=
:1'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp=
; </span><span
style=3D'mso-spacerun:yes'>&nbsp;</span><span style=3D'mso-tab-count:2'>&nb=
sp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nb=
sp;&nbsp;&nbsp;&nbsp;&nbsp; </span>a=3D<span
class=3DSpellE>CalcPositions</span>[<span class=3DSpellE>i</span>];<o:p></o=
:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </spa=
n><span
class=3DGramE>bin</span> =3D 1 + <span class=3DSpellE>Int_t</span>(a + 0.5)=
;<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span style=3D'mso-tab-count=
:1'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp=
; </span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </spa=
n><span
class=3DSpellE>PosX</span>[<span class=3DSpellE>i</span>] =3D d-&gt;<span
class=3DSpellE><span class=3DGramE>GetBinCenter</span></span><span class=3D=
GramE>(</span>bin);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </spa=
n><span
class=3DSpellE>PosY</span>[<span class=3DSpellE>i</span>] =3D d-&gt;<span
class=3DSpellE><span class=3DGramE>GetBinContent</span></span><span class=
=3DGramE>(</span>bin);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>}<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DSpellE>TPolyMa=
rker</span>
* pm =3D (<span class=3DSpellE>TPolyMarker</span>*<span class=3DGramE>)h</s=
pan>-&gt;<span
class=3DSpellE>GetListOfFunctions</span>()-&gt;<span class=3DSpellE>FindObj=
ect</span>(&quot;<span
class=3DSpellE>TPolyMarker</span>&quot;);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>if</span=
> (pm) {<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span><span
class=3DGramE>h</span>-&gt;<span class=3DSpellE>GetListOfFunctions</span>()=
-&gt;Remove(pm);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span><span
class=3DGramE>delete</span> pm;<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span>}<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>pm</span=
> =3D new <span
class=3DSpellE>TPolyMarker</span>(<span class=3DSpellE>nfound</span>, <span
class=3DSpellE>PosX</span>, <span class=3DSpellE>PosY</span>);<o:p></o:p></=
span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>h</span>=
-&gt;<span
class=3DSpellE>GetListOfFunctions</span>()-&gt;Add(pm);<o:p></o:p></span></=
p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>pm</span=
>-&gt;<span
class=3DSpellE>SetMarkerStyle</span>(23);<o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>pm</span=
>-&gt;<span
class=3DSpellE>SetMarkerColor</span>(<span class=3DSpellE>kRed</span>);<o:p=
></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'><span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><span class=3DGramE>pm</span=
>-&gt;<span
class=3DSpellE>SetMarkerSize</span>(1);<span
style=3D'mso-spacerun:yes'>&nbsp;&nbsp; </span><o:p></o:p></span></p>

<p class=3DMsoNormal><span lang=3DEN-US style=3D'font-size:10.0pt'>}<o:p></=
o:p></span></p>

</div>

<!-- */
// --> End_Html

   Int_t i, j, k, shift =
       2 * fNPeaks + 7, peak_vel, rozmer, iter, regul_cycle,
       flag;
   Double_t a, b, alpha, chi_opt, yw, ywm, f, chi2, chi_min, chi =
       0, pi, pmin = 0, chi_cel = 0, chi_er;
   Double_t *working_space = new Double_t[5 * (2 * fNPeaks + 7)];
   for (i = 0, j = 0; i < fNPeaks; i++) {
      working_space[2 * i] = fAmpInit[i];        //vector parameter
      if (fFixAmp[i] == false) {
         working_space[shift + j] = fAmpInit[i];        //vector xk
         j += 1;
      }
      working_space[2 * i + 1] = fPositionInit[i];        //vector parameter
      if (fFixPosition[i] == false) {
         working_space[shift + j] = fPositionInit[i];        //vector xk
         j += 1;
      }
   }
   peak_vel = 2 * i;
   working_space[2 * i] = fSigmaInit;        //vector parameter
   if (fFixSigma == false) {
      working_space[shift + j] = fSigmaInit;        //vector xk
      j += 1;
   }
   working_space[2 * i + 1] = fTInit;        //vector parameter
   if (fFixT == false) {
      working_space[shift + j] = fTInit;        //vector xk
      j += 1;
   }
   working_space[2 * i + 2] = fBInit;        //vector parameter
   if (fFixB == false) {
      working_space[shift + j] = fBInit;        //vector xk
      j += 1;
   }
   working_space[2 * i + 3] = fSInit;        //vector parameter
   if (fFixS == false) {
      working_space[shift + j] = fSInit;        //vector xk
      j += 1;
   }
   working_space[2 * i + 4] = fA0Init;        //vector parameter
   if (fFixA0 == false) {
      working_space[shift + j] = fA0Init;        //vector xk
      j += 1;
   }
   working_space[2 * i + 5] = fA1Init;        //vector parameter
   if (fFixA1 == false) {
      working_space[shift + j] = fA1Init;        //vector xk
      j += 1;
   }
   working_space[2 * i + 6] = fA2Init;        //vector parameter
   if (fFixA2 == false) {
      working_space[shift + j] = fA2Init;        //vector xk
      j += 1;
   }
   rozmer = j;
   if (rozmer == 0){
      Error ("FitAwmi","All parameters are fixed");   
      return;    
   }
   if (rozmer >= fXmax - fXmin + 1){
      Error ("FitAwmi","Number of fitted parameters is larger than # of fitted points");   
      return;    
   }
   Double_t **working_matrix = new Double_t *[rozmer];
   for (i = 0; i < rozmer; i++)
      working_matrix[i] = new Double_t[rozmer + 4];
   for (iter = 0; iter < fNumberIterations; iter++) {
      for (j = 0; j < rozmer; j++) {
         working_space[3 * shift + j] = 0;        //temp
         for (k = 0; k <= rozmer; k++) {
            working_matrix[j][k] = 0;
         }
      }
      
      //filling working matrix
      alpha = fAlpha;
      chi_opt = 0;
      for (i = fXmin; i <= fXmax; i++) {
         
             //calculation of gradient vector
             for (j = 0, k = 0; j < fNPeaks; j++) {
            if (fFixAmp[j] == false) {
               working_space[2 * shift + k] =
                   Deramp((Double_t) i, working_space[2 * j + 1],
               working_space[peak_vel],
               working_space[peak_vel + 1],
               working_space[peak_vel + 3],
               working_space[peak_vel + 2]);
               k += 1;
            }
            if (fFixPosition[j] == false) {
               working_space[2 * shift + k] =
                   Deri0((Double_t) i, working_space[2 * j],
               working_space[2 * j + 1], working_space[peak_vel],
               working_space[peak_vel + 1],
               working_space[peak_vel + 3],
               working_space[peak_vel + 2]);
               k += 1;
            }
         } if (fFixSigma == false) {
            working_space[2 * shift + k] =
                Dersigma(fNPeaks, (Double_t) i, working_space,
            working_space[peak_vel],
            working_space[peak_vel + 1],
            working_space[peak_vel + 3],
            working_space[peak_vel + 2]);
            k += 1;
         }
         if (fFixT == false) {
            working_space[2 * shift + k] =
                Dert(fNPeaks, (Double_t) i, working_space,
            working_space[peak_vel], working_space[peak_vel + 2]);
            k += 1;
         }
         if (fFixB == false) {
            working_space[2 * shift + k] =
                Derb(fNPeaks, (Double_t) i, working_space,
            working_space[peak_vel], working_space[peak_vel + 1],
            working_space[peak_vel + 2]);
            k += 1;
         }
         if (fFixS == false) {
            working_space[2 * shift + k] =
                Ders(fNPeaks, (Double_t) i, working_space,
            working_space[peak_vel]);
            k += 1;
         }
         if (fFixA0 == false) {
            working_space[2 * shift + k] = 1.;
            k += 1;
         }
         if (fFixA1 == false) {
            working_space[2 * shift + k] = Dera1((Double_t) i);
            k += 1;
         }
         if (fFixA2 == false) {
            working_space[2 * shift + k] = Dera2((Double_t) i);
            k += 1;
         }
         yw = source[i];
         ywm = yw;
         f = Shape(fNPeaks, (Double_t) i, working_space,
         working_space[peak_vel], working_space[peak_vel + 1],
         working_space[peak_vel + 3],
         working_space[peak_vel + 2],
         working_space[peak_vel + 4],
         working_space[peak_vel + 5],
         working_space[peak_vel + 6]);
         if (fStatisticType == kFitOptimMaxLikelihood) {
            if (f > 0.00001)
               chi_opt += yw * TMath::Log(f) - f;
         }
         
         else {
            if (ywm != 0)
               chi_opt += (yw - f) * (yw - f) / ywm;
         }
         if (fStatisticType == kFitOptimChiFuncValues) {
            ywm = f;
            if (f < 0.00001)
               ywm = 0.00001;
         }
         
         else if (fStatisticType == kFitOptimMaxLikelihood) {
            ywm = f;
            if (f < 0.00001)
               ywm = 0.00001;
         }
         
         else {
            if (ywm == 0)
               ywm = 1;
         }
         for (j = 0; j < rozmer; j++) {
            for (k = 0; k < rozmer; k++) {
               b = working_space[2 * shift +
                                  j] * working_space[2 * shift + k] / ywm;
               if (fStatisticType == kFitOptimChiFuncValues)
                  b = b * (4 * yw - 2 * f) / ywm;
               working_matrix[j][k] += b;
               if (j == k)
                  working_space[3 * shift + j] += b;
            }
         }
         if (fStatisticType == kFitOptimChiFuncValues)
            b = (f * f - yw * yw) / (ywm * ywm);
         
         else
            b = (f - yw) / ywm;
         for (j = 0; j < rozmer; j++) {
            working_matrix[j][rozmer] -= b * working_space[2 * shift + j];
         }
      }
      for (i = 0; i < rozmer; i++) {
         working_matrix[i][rozmer + 1] = 0;        //xk
      }
      StiefelInversion(working_matrix, rozmer);
      for (i = 0; i < rozmer; i++) {
         working_space[2 * shift + i] = working_matrix[i][rozmer + 1];        //der
      }
      
      //calculate chi_opt
      chi2 = chi_opt;
      chi_opt = TMath::Sqrt(TMath::Abs(chi_opt));
      
      //calculate new parameters
      regul_cycle = 0;
      for (j = 0; j < rozmer; j++) {
         working_space[4 * shift + j] = working_space[shift + j];        //temp_xk[j]=xk[j]
      }
      
      do {
         if (fAlphaOptim == kFitAlphaOptimal) {
            if (fStatisticType != kFitOptimMaxLikelihood)
               chi_min = 10000 * chi2;
            
            else
               chi_min = 0.1 * chi2;
            flag = 0;
            for (pi = 0.1; flag == 0 && pi <= 100; pi += 0.1) {
               for (j = 0; j < rozmer; j++) {
                  working_space[shift + j] = working_space[4 * shift + j] + pi * alpha * working_space[2 * shift + j];        //xk[j]=temp_xk[j]+pi*alpha*der[j]
               }
               for (i = 0, j = 0; i < fNPeaks; i++) {
                  if (fFixAmp[i] == false) {
                     if (working_space[shift + j] < 0)        //xk[j]
                        working_space[shift + j] = 0;        //xk[j]
                     working_space[2 * i] = working_space[shift + j];        //parameter[2*i]=xk[j]
                     j += 1;
                  }
                  if (fFixPosition[i] == false) {
                     if (working_space[shift + j] < fXmin)        //xk[j]
                        working_space[shift + j] = fXmin;        //xk[j]
                     if (working_space[shift + j] > fXmax)        //xk[j]
                        working_space[shift + j] = fXmax;        //xk[j]
                     working_space[2 * i + 1] = working_space[shift + j];        //parameter[2*i+1]=xk[j]
                     j += 1;
                  }
               }
               if (fFixSigma == false) {
                  if (working_space[shift + j] < 0.001) {        //xk[j]
                     working_space[shift + j] = 0.001;        //xk[j]
                  }
                  working_space[peak_vel] = working_space[shift + j];        //parameter[peak_vel]=xk[j]
                  j += 1;
               }
               if (fFixT == false) {
                  working_space[peak_vel + 1] = working_space[shift + j];        //parameter[peak_vel+1]=xk[j]
                  j += 1;
               }
               if (fFixB == false) {
                  if (TMath::Abs(working_space[shift + j]) < 0.001) {        //xk[j]
                     if (working_space[shift + j] < 0)        //xk[j]
                        working_space[shift + j] = -0.001;        //xk[j]
                     else
                        working_space[shift + j] = 0.001;        //xk[j]
                  }
                  working_space[peak_vel + 2] = working_space[shift + j];        //parameter[peak_vel+2]=xk[j]
                  j += 1;
               }
               if (fFixS == false) {
                  working_space[peak_vel + 3] = working_space[shift + j];        //parameter[peak_vel+3]=xk[j]
                  j += 1;
               }
               if (fFixA0 == false) {
                  working_space[peak_vel + 4] = working_space[shift + j];        //parameter[peak_vel+4]=xk[j]
                  j += 1;
               }
               if (fFixA1 == false) {
                  working_space[peak_vel + 5] = working_space[shift + j];        //parameter[peak_vel+5]=xk[j]
                  j += 1;
               }
               if (fFixA2 == false) {
                  working_space[peak_vel + 6] = working_space[shift + j];        //parameter[peak_vel+6]=xk[j]
                  j += 1;
               }
               chi2 = 0;
               for (i = fXmin; i <= fXmax; i++) {
                  yw = source[i];
                  ywm = yw;
                  f = Shape(fNPeaks, (Double_t) i, working_space,
                  working_space[peak_vel],
                  working_space[peak_vel + 1],
                  working_space[peak_vel + 3],
                  working_space[peak_vel + 2],
                  working_space[peak_vel + 4],
                  working_space[peak_vel + 5],
                  working_space[peak_vel + 6]);
                  if (fStatisticType == kFitOptimChiFuncValues) {
                     ywm = f;
                     if (f < 0.00001)
                        ywm = 0.00001;
                  }
                  if (fStatisticType == kFitOptimMaxLikelihood) {
                     if (f > 0.00001)
                        chi2 += yw * TMath::Log(f) - f;
                  }
                  
                  else {
                     if (ywm != 0)
                        chi2 += (yw - f) * (yw - f) / ywm;
                  }
               }
               if (chi2 < chi_min
                    && fStatisticType != kFitOptimMaxLikelihood
                    || chi2 > chi_min
                    && fStatisticType == kFitOptimMaxLikelihood) {
                  pmin = pi, chi_min = chi2;
               }
               
               else
                  flag = 1;
               if (pi == 0.1)
                  chi_min = chi2;
               chi = chi_min;
            }
            if (pmin != 0.1) {
               for (j = 0; j < rozmer; j++) {
                  working_space[shift + j] = working_space[4 * shift + j] + pmin * alpha * working_space[2 * shift + j];        //xk[j]=temp_xk[j]+pmin*alpha*der[j]
               }
               for (i = 0, j = 0; i < fNPeaks; i++) {
                  if (fFixAmp[i] == false) {
                     if (working_space[shift + j] < 0)        //xk[j]
                        working_space[shift + j] = 0;        //xk[j]
                     working_space[2 * i] = working_space[shift + j];        //parameter[2*i]=xk[j]
                     j += 1;
                  }
                  if (fFixPosition[i] == false) {
                     if (working_space[shift + j] < fXmin)        //xk[j]
                        working_space[shift + j] = fXmin;        //xk[j]
                     if (working_space[shift + j] > fXmax)        //xk[j]
                        working_space[shift + j] = fXmax;        //xk[j]
                     working_space[2 * i + 1] = working_space[shift + j];        //parameter[2*i+1]=xk[j]
                     j += 1;
                  }
               }
               if (fFixSigma == false) {
                  if (working_space[shift + j] < 0.001) {        //xk[j]
                     working_space[shift + j] = 0.001;        //xk[j]
                  }
                  working_space[peak_vel] = working_space[shift + j];        //parameter[peak_vel]=xk[j]
                  j += 1;
               }
               if (fFixT == false) {
                  working_space[peak_vel + 1] = working_space[shift + j];        //parameter[peak_vel+1]=xk[j]
                  j += 1;
               }
               if (fFixB == false) {
                  if (TMath::Abs(working_space[shift + j]) < 0.001) {        //xk[j]
                     if (working_space[shift + j] < 0)        //xk[j]
                        working_space[shift + j] = -0.001;        //xk[j]
                     else
                        working_space[shift + j] = 0.001;        //xk[j]
                  }
                  working_space[peak_vel + 2] = working_space[shift + j];        //parameter[peak_vel+2]=xk[j]
                  j += 1;
               }
               if (fFixS == false) {
                  working_space[peak_vel + 3] = working_space[shift + j];        //parameter[peak_vel+3]=xk[j]
                  j += 1;
               }
               if (fFixA0 == false) {
                  working_space[peak_vel + 4] = working_space[shift + j];        //parameter[peak_vel+4]=xk[j]
                  j += 1;
               }
               if (fFixA1 == false) {
                  working_space[peak_vel + 5] = working_space[shift + j];        //parameter[peak_vel+5]=xk[j]
                  j += 1;
               }
               if (fFixA2 == false) {
                  working_space[peak_vel + 6] = working_space[shift + j];        //parameter[peak_vel+6]=xk[j]
                  j += 1;
               }
               chi = chi_min;
            }
         }
         
         else {
            for (j = 0; j < rozmer; j++) {
               working_space[shift + j] = working_space[4 * shift + j] + alpha * working_space[2 * shift + j];        //xk[j]=temp_xk[j]+alpha*der[j]
            }
            for (i = 0, j = 0; i < fNPeaks; i++) {
               if (fFixAmp[i] == false) {
                  if (working_space[shift + j] < 0)        //xk[j]
                     working_space[shift + j] = 0;        //xk[j]
                  working_space[2 * i] = working_space[shift + j];        //parameter[2*i]=xk[j]
                  j += 1;
               }
               if (fFixPosition[i] == false) {
                  if (working_space[shift + j] < fXmin)        //xk[j]
                     working_space[shift + j] = fXmin;        //xk[j]
                  if (working_space[shift + j] > fXmax)        //xk[j]
                     working_space[shift + j] = fXmax;        //xk[j]
                  working_space[2 * i + 1] = working_space[shift + j];        //parameter[2*i+1]=xk[j]
                  j += 1;
               }
            }
            if (fFixSigma == false) {
               if (working_space[shift + j] < 0.001) {        //xk[j]
                  working_space[shift + j] = 0.001;        //xk[j]
               }
               working_space[peak_vel] = working_space[shift + j];        //parameter[peak_vel]=xk[j]
               j += 1;
            }
            if (fFixT == false) {
               working_space[peak_vel + 1] = working_space[shift + j];        //parameter[peak_vel+1]=xk[j]
               j += 1;
            }
            if (fFixB == false) {
               if (TMath::Abs(working_space[shift + j]) < 0.001) {        //xk[j]
                  if (working_space[shift + j] < 0)        //xk[j]
                     working_space[shift + j] = -0.001;        //xk[j]
                  else
                     working_space[shift + j] = 0.001;        //xk[j]
               }
               working_space[peak_vel + 2] = working_space[shift + j];        //parameter[peak_vel+2]=xk[j]
               j += 1;
            }
            if (fFixS == false) {
               working_space[peak_vel + 3] = working_space[shift + j];        //parameter[peak_vel+3]=xk[j]
               j += 1;
            }
            if (fFixA0 == false) {
               working_space[peak_vel + 4] = working_space[shift + j];        //parameter[peak_vel+4]=xk[j]
               j += 1;
            }
            if (fFixA1 == false) {
               working_space[peak_vel + 5] = working_space[shift + j];        //parameter[peak_vel+5]=xk[j]
               j += 1;
            }
            if (fFixA2 == false) {
               working_space[peak_vel + 6] = working_space[shift + j];        //parameter[peak_vel+6]=xk[j]
               j += 1;
            }
            chi = 0;
            for (i = fXmin; i <= fXmax; i++) {
               yw = source[i];
               ywm = yw;
               f = Shape(fNPeaks, (Double_t) i, working_space,
               working_space[peak_vel],
               working_space[peak_vel + 1],
               working_space[peak_vel + 3],
               working_space[peak_vel + 2],
               working_space[peak_vel + 4],
               working_space[peak_vel + 5],
               working_space[peak_vel + 6]);
               if (fStatisticType == kFitOptimChiFuncValues) {
                  ywm = f;
                  if (f < 0.00001)
                     ywm = 0.00001;
               }
               if (fStatisticType == kFitOptimMaxLikelihood) {
                  if (f > 0.00001)
                     chi += yw * TMath::Log(f) - f;
               }
               
               else {
                  if (ywm != 0)
                     chi += (yw - f) * (yw - f) / ywm;
               }
            }
         }
         chi2 = chi;
         chi = TMath::Sqrt(TMath::Abs(chi));
         if (fAlphaOptim == kFitAlphaHalving && chi > 1E-6)
            alpha = alpha * chi_opt / (2 * chi);
         
         else if (fAlphaOptim == kFitAlphaOptimal)
            alpha = alpha / 10.0;
         iter += 1;
         regul_cycle += 1;
      } while ((chi > chi_opt
                 && fStatisticType != kFitOptimMaxLikelihood
                 || chi < chi_opt
                 && fStatisticType == kFitOptimMaxLikelihood)
                && regul_cycle < kFitNumRegulCycles);
      for (j = 0; j < rozmer; j++) {
         working_space[4 * shift + j] = 0;        //temp_xk[j]
         working_space[2 * shift + j] = 0;        //der[j]
      }
      for (i = fXmin, chi_cel = 0; i <= fXmax; i++) {
         yw = source[i];
         if (yw == 0)
            yw = 1;
         f = Shape(fNPeaks, (Double_t) i, working_space,
         working_space[peak_vel], working_space[peak_vel + 1],
         working_space[peak_vel + 3],
         working_space[peak_vel + 2],
         working_space[peak_vel + 4],
         working_space[peak_vel + 5],
         working_space[peak_vel + 6]);
         chi_opt = (yw - f) * (yw - f) / yw;
         chi_cel += (yw - f) * (yw - f) / yw;
         
             //calculate gradient vector
             for (j = 0, k = 0; j < fNPeaks; j++) {
            if (fFixAmp[j] == false) {
               a = Deramp((Double_t) i, working_space[2 * j + 1],
               working_space[peak_vel],
               working_space[peak_vel + 1],
               working_space[peak_vel + 3],
               working_space[peak_vel + 2]);
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt;        //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b;        //temp_xk[k]
               }
               k += 1;
            }
            if (fFixPosition[j] == false) {
               a = Deri0((Double_t) i, working_space[2 * j],
               working_space[2 * j + 1],
               working_space[peak_vel],
               working_space[peak_vel + 1],
               working_space[peak_vel + 3],
               working_space[peak_vel + 2]);
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt;        //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b;        //temp_xk[k]
               }
               k += 1;
            }
         }
         if (fFixSigma == false) {
            a = Dersigma(fNPeaks, (Double_t) i, working_space,
            working_space[peak_vel],
            working_space[peak_vel + 1],
            working_space[peak_vel + 3],
           working_space[peak_vel + 2]);
            if (yw != 0) {
               working_space[2 * shift + k] += chi_opt;        //der[k]
               b = a * a / yw;
               working_space[4 * shift + k] += b;        //temp_xk[k]
            }
            k += 1;
         }
         if (fFixT == false) {
            a = Dert(fNPeaks, (Double_t) i, working_space,
            working_space[peak_vel],
            working_space[peak_vel + 2]);
            if (yw != 0) {
               working_space[2 * shift + k] += chi_opt;        //der[k]
               b = a * a / yw;
               working_space[4 * shift + k] += b;        //temp_xk[k]
            }
            k += 1;
         }
         if (fFixB == false) {
            a = Derb(fNPeaks, (Double_t) i, working_space,
            working_space[peak_vel], working_space[peak_vel + 1],
            working_space[peak_vel + 2]);
            if (yw != 0) {
               working_space[2 * shift + k] += chi_opt;        //der[k]
               b = a * a / yw;
               working_space[4 * shift + k] += b;        //temp_xk[k]
            }
            k += 1;
         }
         if (fFixS == false) {
            a = Ders(fNPeaks, (Double_t) i, working_space,
            working_space[peak_vel]);
            if (yw != 0) {
               working_space[2 * shift + k] += chi_opt;        //der[k]
               b = a * a / yw;
               working_space[4 * shift + k] += b;        //tem_xk[k]
            }
            k += 1;
         }
         if (fFixA0 == false) {
            a = 1.0;
            if (yw != 0) {
               working_space[2 * shift + k] += chi_opt;        //der[k]
               b = a * a / yw;
               working_space[4 * shift + k] += b;        //temp_xk[k]
            }
            k += 1;
         }
         if (fFixA1 == false) {
            a = Dera1((Double_t) i);
            if (yw != 0) {
               working_space[2 * shift + k] += chi_opt;        //der[k]
               b = a * a / yw;
               working_space[4 * shift + k] += b;        //temp_xk[k]
            }
            k += 1;
         }
         if (fFixA2 == false) {
            a = Dera2((Double_t) i);
            if (yw != 0) {
               working_space[2 * shift + k] += chi_opt;        //der[k]
               b = a * a / yw;
               working_space[4 * shift + k] += b;        //temp_xk[k]
            }
            k += 1;
         }
      }
   }
   b = fXmax - fXmin + 1 - rozmer;
   chi_er = chi_cel / b;
   for (i = 0, j = 0; i < fNPeaks; i++) {
      fArea[i] =
          Area(working_space[2 * i], working_space[peak_vel],
               working_space[peak_vel + 1], working_space[peak_vel + 2]);
      if (fFixAmp[i] == false) {
         fAmpCalc[i] = working_space[shift + j];        //xk[j]
         if (working_space[3 * shift + j] != 0)
            fAmpErr[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));        //der[j]/temp[j]
         if (fArea[i] > 0) {
            a = Derpa(working_space[peak_vel],
            working_space[peak_vel + 1],
            working_space[peak_vel + 2]);
            b = working_space[4 * shift + j];        //temp_xk[j]
            if (b == 0)
               b = 1;
            
            else
               b = 1 / b;
            fAreaErr[i] = TMath::Sqrt(TMath::Abs(a * a * b * chi_er));
         }
         
         else
            fAreaErr[i] = 0;
         j += 1;
      }
      
      else {
         fAmpCalc[i] = fAmpInit[i];
         fAmpErr[i] = 0;
         fAreaErr[i] = 0;
      }
      if (fFixPosition[i] == false) {
         fPositionCalc[i] = working_space[shift + j];        //xk[j]
         if (working_space[3 * shift + j] != 0)        //temp[j]
            fPositionErr[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));        //Der[j]/temp[j]
         j += 1;
      }
      
      else {
         fPositionCalc[i] = fPositionInit[i];
         fPositionErr[i] = 0;
      }
   }
   if (fFixSigma == false) {
      fSigmaCalc = working_space[shift + j];        //xk[j]
      if (working_space[3 * shift + j] != 0)        //temp[j]
         fSigmaErr = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));        //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fSigmaCalc = fSigmaInit;
      fSigmaErr = 0;
   }
   if (fFixT == false) {
      fTCalc = working_space[shift + j];        //xk[j]
      if (working_space[3 * shift + j] != 0)        //temp[j]
         fTErr = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));        //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fTCalc = fTInit;
      fTErr = 0;
   }
   if (fFixB == false) {
      fBCalc = working_space[shift + j];        //xk[j]
      if (working_space[3 * shift + j] != 0)        //temp[j]
         fBErr = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));        //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fBCalc = fBInit;
      fBErr = 0;
   }
   if (fFixS == false) {
      fSCalc = working_space[shift + j];        //xk[j]
      if (working_space[3 * shift + j] != 0)        //temp[j]
         fSErr = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));        //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fSCalc = fSInit;
      fSErr = 0;
   }
   if (fFixA0 == false) {
      fA0Calc = working_space[shift + j];        //xk[j]
      if (working_space[3 * shift + j] != 0)        //temp[j]
         fA0Err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));        //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fA0Calc = fA0Init;
      fA0Err = 0;
   }
   if (fFixA1 == false) {
      fA1Calc = working_space[shift + j];        //xk[j]
      if (working_space[3 * shift + j] != 0)        //temp[j]
         fA1Err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));        //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fA1Calc = fA1Init;
      fA1Err = 0;
   }
   if (fFixA2 == false) {
      fA2Calc = working_space[shift + j];        //xk[j]
      if (working_space[3 * shift + j] != 0)        //temp[j]
         fA2Err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j]));        //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fA2Calc = fA2Init;
      fA2Err = 0;
   }
   b = fXmax - fXmin + 1 - rozmer;
   fChi = chi_cel / b;
   for (i = fXmin; i <= fXmax; i++) {
      f = Shape(fNPeaks, (Double_t) i, working_space,
      working_space[peak_vel], working_space[peak_vel + 1],
      working_space[peak_vel + 3], working_space[peak_vel + 2],
      working_space[peak_vel + 4], working_space[peak_vel + 5],
      working_space[peak_vel + 6]);
      source[i] = f;
   } 
   for (i = 0; i < rozmer; i++)
      delete [] working_matrix[i];
   delete [] working_matrix;
   delete [] working_space;
   return;
}

//____________________________________________________________________________
void TSpectrumFit::SetFitParameters(Int_t xmin,Int_t xmax, Int_t numberIterations, Double_t alpha, Int_t statisticType, Int_t alphaOptim, Int_t power, Int_t fitTaylor)
{
//////////////////////////////////////////////////////////////////////////////
//   SETTER FUNCTION                                                      
//                                                     
//   This function sets the following fitting parameters:
//         -xmin, xmax - fitting region
//         -numberIterations - # of desired iterations in the fit
//         -alpha - convergence coefficient, it should be positive number and <=1, for details see references
//         -statisticType - type of statistics, possible values kFitOptimChiCounts (chi square statistics with counts as weighting coefficients), kFitOptimChiFuncValues (chi square statistics with function values as weighting coefficients),kFitOptimMaxLikelihood 
//         -alphaOptim - optimization of convergence algorithm, possible values kFitAlphaHalving, kFitAlphaOptimal
//         -power - possible values kFitPower2,4,6,8,10,12, for details see references. It applies only for Awmi fitting function.
//         -fitTaylor - order of Taylor expansion, possible values kFitTaylorOrderFirst, kFitTaylorOrderSecond. It applies only for Awmi fitting function.
//////////////////////////////////////////////////////////////////////////////   
   if(xmin<0 || xmax <= xmin){ 
      Error("SetFitParameters", "Wrong range");
      return;
   }      
   if (numberIterations <= 0){
      Error("SetFitParameters","Invalid number of iterations, must be positive");
      return;
   }      
   if (alpha <= 0 || alpha > 1){
      Error ("SetFitParameters","Invalid step coefficient alpha, must be > than 0 and <=1");
      return;
   }      
   if (statisticType != kFitOptimChiCounts
        && statisticType != kFitOptimChiFuncValues
        && statisticType != kFitOptimMaxLikelihood){
      Error("SetFitParameters","Wrong type of statistic");
      return;
   }      
   if (alphaOptim != kFitAlphaHalving
        && alphaOptim != kFitAlphaOptimal){
      Error("SetFitParameters","Wrong optimization algorithm");
      return;
   }      
   if (power != kFitPower2 && power != kFitPower4
        && power != kFitPower6 && power != kFitPower8
        && power != kFitPower10 && power != kFitPower12){
      Error("SetFitParameters","Wrong power");
      return;
   }      
   if (fitTaylor != kFitTaylorOrderFirst
        && fitTaylor != kFitTaylorOrderSecond){
      Error("SetFitParameters","Wrong order of Taylor development");
      return;
   }      
   fXmin=xmin,fXmax=xmax,fNumberIterations=numberIterations,fAlpha=alpha,fStatisticType=statisticType,fAlphaOptim=alphaOptim,fPower=power,fFitTaylor=fitTaylor;
}

//_______________________________________________________________________________
void TSpectrumFit::SetPeakParameters(Double_t sigma, Bool_t fixSigma, const Float_t *positionInit, const Bool_t *fixPosition, const Float_t *ampInit, const Bool_t *fixAmp)
{
//////////////////////////////////////////////////////////////////////////////
//   SETTER FUNCTION                                                      
//                                                     
//   This function sets the following fitting parameters of peaks:
//         -sigma - initial value of sigma parameter
//         -fixSigma - logical value of sigma parameter, which allows to fix the parameter (not to fit)   
//         -positionInit - aray of initial values of peaks positions
//         -fixPosition - array of logical values which allow to fix appropriate positions (not fit). However they are present in the estimated functional.
//         -ampInit - aray of initial values of peaks amplitudes
//         -fixAmp - aray of logical values which allow to fix appropriate amplitudes (not fit). However they are present in the estimated functional      
//////////////////////////////////////////////////////////////////////////////   
   
   Int_t i;
   if (sigma <= 0){
      Error ("SetPeakParameters","Invalid sigma, must be > than 0");
      return;
   }      
   for(i=0; i < fNPeaks; i++){
      if((Int_t) positionInit[i] < fXmin || (Int_t) positionInit[i] > fXmax){
         Error ("SetPeakParameters","Invalid peak position, must be in the range fXmin, fXmax");
         return;
      }         
      if(ampInit[i] < 0){
         Error ("SetPeakParameters","Invalid peak amplitude, must be > than 0");        
         return;
      }         
   }
   fSigmaInit = sigma,fFixSigma = fixSigma;
   for(i=0; i < fNPeaks; i++){
      fPositionInit[i] = (Double_t) positionInit[i];
      fFixPosition[i] = fixPosition[i];
      fAmpInit[i] = (Double_t) ampInit[i];
      fFixAmp[i] = fixAmp[i]; 
   }
}

//_______________________________________________________________________________
void TSpectrumFit::SetBackgroundParameters(Double_t a0Init, Bool_t fixA0, Double_t a1Init, Bool_t fixA1, Double_t a2Init, Bool_t fixA2)
{
//////////////////////////////////////////////////////////////////////////////
//   SETTER FUNCTION                                                      
//                                                     
//   This function sets the following fitting parameters of background:
//         -a0Init - initial value of a0 parameter (backgroud is estimated as a0+a1*x+a2*x*x)
//         -fixA0 - logical value of a0 parameter, which allows to fix the parameter (not to fit)  
//         -a1Init - initial value of a1 parameter
//         -fixA1 - logical value of a1 parameter, which allows to fix the parameter (not to fit)   
//         -a2Init - initial value of a2 parameter
//         -fixA2 - logical value of a2 parameter, which allows to fix the parameter (not to fit)     
//////////////////////////////////////////////////////////////////////////////   
   
   fA0Init = a0Init;
   fFixA0 = fixA0;
   fA1Init = a1Init;
   fFixA1 = fixA1;
   fA2Init = a2Init;
   fFixA2 = fixA2;    
}

//_______________________________________________________________________________
void TSpectrumFit::SetTailParameters(Double_t tInit, Bool_t fixT, Double_t bInit, Bool_t fixB, Double_t sInit, Bool_t fixS)
{
//////////////////////////////////////////////////////////////////////////////
//   SETTER FUNCTION                                                      
//                                                     
//   This function sets the following fitting parameters of tails of peaks
//         -tInit - initial value of t parameter
//         -fixT - logical value of t parameter, which allows to fix the parameter (not to fit)  
//         -bInit - initial value of b parameter
//         -fixB - logical value of b parameter, which allows to fix the parameter (not to fit)  
//         -sInit - initial value of s parameter
//         -fixS - logical value of s parameter, which allows to fix the parameter (not to fit)  
//////////////////////////////////////////////////////////////////////////////   
   
   fTInit = tInit;
   fFixT = fixT;
   fBInit = bInit;
   fFixB = fixB;
   fSInit = sInit;
   fFixS = fixS;     
}

//_______________________________________________________________________________
void TSpectrumFit::GetSigma(Double_t &sigma, Double_t &sigmaErr)
{
//////////////////////////////////////////////////////////////////////////////
//   GETTER FUNCTION                                                      
//                                                     
//   This function gets the sigma parameter and its error
//         -sigma - gets the fitted value of sigma parameter
//         -sigmaErr - gets error value of sigma parameter
//////////////////////////////////////////////////////////////////////////////   
   sigma=fSigmaCalc;
   sigmaErr=fSigmaErr;
}

//_______________________________________________________________________________
void TSpectrumFit::GetBackgroundParameters(Double_t &a0, Double_t &a0Err, Double_t &a1, Double_t &a1Err, Double_t &a2, Double_t &a2Err)
{
//////////////////////////////////////////////////////////////////////////////
//   GETTER FUNCTION                                                      
//                                                     
//   This function gets the background parameters and their errors
//         -a0 - gets the fitted value of a0 parameter
//         -a0Err - gets error value of a0 parameter
//         -a1 - gets the fitted value of a1 parameter
//         -a1Err - gets error value of a1 parameter
//         -a2 - gets the fitted value of a2 parameter
//         -a2Err - gets error value of a2 parameter
//////////////////////////////////////////////////////////////////////////////      
   a0 = fA0Calc;
   a0Err = fA0Err;
   a1 = fA1Calc;
   a1Err = fA1Err;
   a2 = fA2Calc;
   a2Err = fA2Err;    
}

//_______________________________________________________________________________
void TSpectrumFit::GetTailParameters(Double_t &t, Double_t &tErr, Double_t &b, Double_t &bErr, Double_t &s, Double_t &sErr)
{
//////////////////////////////////////////////////////////////////////////////
//   GETTER FUNCTION                                                      
//                                                     
//   This function gets the tail parameters and their errors
//         -t - gets the fitted value of t parameter
//         -tErr - gets error value of t parameter
//         -b - gets the fitted value of b parameter
//         -bErr - gets error value of b parameter
//         -s - gets the fitted value of s parameter
//         -sErr - gets error value of s parameter
//////////////////////////////////////////////////////////////////////////////         
   t = fTCalc;
   tErr = fTErr;
   b = fBCalc;
   bErr = fBErr;  
   s = fSCalc;
   sErr = fSErr;  
}

