// @(#)root/spectrum:$Id$
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
   fNumberIterations = 1;
   fXmin = 0;
   fXmax = 100;
   fStatisticType = kFitOptimChiCounts;
   fAlphaOptim = kFitAlphaHalving;
   fPower = kFitPower2;
   fFitTaylor = kFitTaylorOrderFirst;
   fAlpha =1;
   fChi = 0;
   fPositionInit   = 0;
   fPositionCalc   = 0;
   fPositionErr   = 0;
   fFixPosition   = 0;
   fAmpInit   = 0;
   fAmpCalc   = 0;
   fAmpErr    = 0;
   fFixAmp    = 0;
   fArea      = 0;
   fAreaErr   = 0;
   fSigmaInit = 2;
   fSigmaCalc = 1;
   fSigmaErr  = 0;
   fTInit = 0;
   fTCalc = 0;
   fTErr = 0;
   fBInit = 1;
   fBCalc = 0;
   fBErr = 0;
   fSInit = 0;
   fSCalc = 0;
   fSErr = 0;
   fA0Init = 0;
   fA0Calc = 0;
   fA0Err = 0;
   fA1Init = 0;
   fA1Calc = 0;
   fA1Err = 0;
   fA2Init = 0;
   fA2Calc = 0;
   fA2Err = 0;
   fFixSigma = false;
   fFixT = true;
   fFixB = true;
   fFixS = true;
   fFixA0 = true;
   fFixA1 = true;
   fFixA2 = true;
}

//______________________________________________________________________________
TSpectrumFit::TSpectrumFit(Int_t numberPeaks) :TNamed("SpectrumFit", "Miroslav Morhac peak fitter")
{
   //numberPeaks: number of fitted peaks (must be greater than zero)
   //the constructor allocates arrays for all fitted parameters (peak positions, amplitudes etc) and sets the member
   //variables to their default values. One can change these variables by member functions (setters) of TSpectrumFit class.
//Begin_Html <!--
/* -->
<div class=Section1>

<p class=MsoNormal style='text-align:justify'>Shape function of the fitted
peaks is </p>

<p class=MsoNormal style='text-align:justify'>

<table cellpadding=0 cellspacing=0 align=left>
 <tr>
  <td width=68 height=6></td>
 </tr>
 <tr>
  <td></td>
  <td><img width=388 height=132 src="gif/spectrumfit_constructor_image001.gif"></td>
 </tr>
</table>

<span style='font-family:Arial'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal><i>&nbsp;</i></p>

<p class=MsoNormal><i>&nbsp;</i></p>

<p class=MsoNormal><i>&nbsp;</i></p>

<br clear=ALL>

<p class=MsoNormal style='text-align:justify'>where a represents vector of
fitted parameters (positions p(j), amplitudes A(j), sigma, relative amplitudes
T, S and slope B).</p>

<p class=MsoNormal><span style='font-size:16.0pt'>&nbsp;</span></p>

</div>

<!-- */
// --> End_Html

   if (numberPeaks <= 0){
      Error ("TSpectrumFit","Invalid number of peaks, must be > than 0");
      return;
   }
   fNPeaks = numberPeaks;
   fNumberIterations = 1;
   fXmin = 0;
   fXmax = 100;
   fStatisticType = kFitOptimChiCounts;
   fAlphaOptim = kFitAlphaHalving;
   fPower = kFitPower2;
   fFitTaylor = kFitTaylorOrderFirst;
   fAlpha =1;
   fChi = 0;
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
   fSigmaInit = 2;
   fSigmaCalc = 1;
   fSigmaErr  = 0;
   fTInit = 0;
   fTCalc = 0;
   fTErr = 0;
   fBInit = 1;
   fBCalc = 0;
   fBErr = 0;
   fSInit = 0;
   fSCalc = 0;
   fSErr = 0;
   fA0Init = 0;
   fA0Calc = 0;
   fA0Err = 0;
   fA1Init = 0;
   fA1Calc = 0;
   fA1Err = 0;
   fA2Init = 0;
   fA2Calc = 0;
   fA2Err = 0;
   fFixSigma = false;
   fFixT = true;
   fFixB = true;
   fFixS = true;
   fFixA0 = true;
   fFixA1 = true;
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
   Double_t a2 = a*a;
   c = 1;
   if (pw >  0) c *= a2;
   if (pw >  2) c *= a2;
   if (pw >  4) c *= a2;
   if (pw >  6) c *= a2;
   if (pw >  8) c *= a2;
   if (pw > 10) c *= a2;
   if (pw > 12) c *= a2;
   return (c);
}

/////////////////END OF AUXILIARY FUNCTIONS USED BY FITTING FUNCTIONS FitAWMI, FitStiefel//////////////////////////
/////////////////FITTING FUNCTION WITHOUT MATRIX INVERSION///////////////////////////////////////

//____________________________________________________________________________
void TSpectrumFit::FitAwmi(Double_t *source)
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
//
//Begin_Html <!--
/* -->
<div class=Section2>

<p class=MsoNormal><b><span style='font-size:14.0pt'>Fitting</span></b></p>

<p class=MsoNormal style='text-align:justify'><i>Goal: to estimate
simultaneously peak shape parameters in spectra with large number of peaks</i></p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>peaks can be fitted separately, each peak (or multiplets) in a region or
together all peaks in a spectrum. To fit separately each peak one needs to
determine the fitted region. However it can happen that the regions of
neighboring peaks are overlapping. Then the results of fitting are very poor.
On the other hand, when fitting together all peaks found in a  spectrum, one
needs to have a method that is  stable (converges) and fast enough to carry out
fitting in reasonable time </p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>we have implemented the nonsymmetrical semiempirical peak shape function
[1]</p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>it contains the symmetrical Gaussian as well as nonsymmetrical terms.</p>

<p class=MsoNormal style='text-align:justify'>

<table cellpadding=0 cellspacing=0 align=left>
 <tr>
  <td width=84 height=18></td>
 </tr>
 <tr>
  <td></td>
  <td><img width=372 height=127 src="gif/spectrumfit_awni_image001.gif"></td>
 </tr>
</table>

<span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:16.0pt'>&nbsp;</span></p>

<br clear=ALL>

<p class=MsoNormal style='text-indent:34.2pt'>where T and S are relative amplitudes
and B is slope.</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>algorithm without matrix inversion (AWMI) allows fitting tens, hundreds
of peaks simultaneously that represent sometimes thousands of parameters [2],
[5]. </p>

<p class=MsoNormal><i>Function:</i></p>

<p class=MsoNormal style='text-align:justify'>void <a
href="http://root.cern.ch/root/html/TSpectrum.html#TSpectrum:Fit1Awmi"><b>TSpectrumFit::FitAwmi</b></a>(<a
href="http://root.cern.ch/root/html/ListOfTypes.html#double"><b>double</b></a> *fSource)
</p>

<p class=MsoNormal style='text-align:justify'>This function fits the source
spectrum using AWMI algorithm. The calling program should fill in input fitting
parameters of the TSpectrumFit class using a set of TSpectrumFit setters. The
fitted parameters are written into the class and the fitted data are written
into source spectrum. </p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><i><span style='color:red'>Parameter:</span></i></p>

<p class=MsoNormal style='text-align:justify'>        <b>fSource</b>-pointer to
the vector of source spectrum                  </p>

<p class=MsoNormal style='text-align:justify'>        </p>

<p class=MsoNormal><i><span style='color:red'>Member variables of the
TSpectrumFit class:</span></i></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Int_t     fNPeaks;                    //number of peaks present in fit, input
parameter, it should be &gt; 0</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Int_t     fNumberIterations;          //number of iterations in fitting
procedure, input parameter, it should be &gt; 0</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Int_t     fXmin;                      //first fitted channel</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Int_t     fXmax;                      //last fitted channel</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Int_t     fStatisticType;             //type of statistics, possible values
kFitOptimChiCounts (chi square statistics with counts as weighting
coefficients), kFitOptimChiFuncValues (chi square statistics with function
values as weighting coefficients),kFitOptimMaxLikelihood</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Int_t     fAlphaOptim;                //optimization of convergence algorithm, possible
values kFitAlphaHalving, kFitAlphaOptimal</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Int_t     fPower;                     //possible values kFitPower2,4,6,8,10,12,
for details see references. It applies only for Awmi fitting function.</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Int_t     fFitTaylor;                 //order of Taylor expansion, possible
values kFitTaylorOrderFirst, kFitTaylorOrderSecond. It applies only for Awmi
fitting function.</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Double_t  fAlpha;                     //convergence coefficient, input
parameter, it should be positive number and &lt;=1, for details see references</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Double_t  fChi;                       //here the fitting functions return
resulting chi square   </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Double_t *fPositionInit;              //[fNPeaks] array of initial values of
peaks positions, input parameters</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Double_t *fPositionCalc;              //[fNPeaks] array of calculated values of
fitted positions, output parameters</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Double_t *fPositionErr;               //[fNPeaks] array of position errors</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Double_t *fAmpInit;                   //[fNPeaks] array of initial values of
peaks amplitudes, input parameters</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Double_t *fAmpCalc;                   //[fNPeaks] array of calculated values of
fitted amplitudes, output parameters</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Double_t *fAmpErr;                    //[fNPeaks] array of amplitude errors</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Double_t *fArea;                      //[fNPeaks] array of calculated areas of
peaks</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Double_t *fAreaErr;                   //[fNPeaks] array of errors of peak areas</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Double_t  fSigmaInit;                 //initial value of sigma parameter</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Double_t  fSigmaCalc;                 //calculated value of sigma parameter</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Double_t  fSigmaErr;                  //error value of sigma parameter</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Double_t  fTInit;                     //initial value of t parameter (relative
amplitude of tail), for details see html manual and references</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Double_t  fTCalc;                     //calculated value of t parameter</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Double_t  fTErr;                      //error value of t parameter</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Double_t  fBInit;                     //initial value of b parameter (slope),
for details see html manual and references</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Double_t  fBCalc;                     //calculated value of b parameter</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Double_t  fBErr;                      //error value of b parameter</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Double_t  fSInit;                     //initial value of s parameter (relative
amplitude of step), for details see html manual and references</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Double_t  fSCalc;                     //calculated value of s parameter</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Double_t  fSErr;                      //error value of s parameter</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Double_t  fA0Init;                    //initial value of background a0
parameter(backgroud is estimated as a0+a1*x+a2*x*x)</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Double_t  fA0Calc;                    //calculated value of background a0
parameter</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Double_t  fA0Err;                     //error value of background a0 parameter</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Double_t  fA1Init;                    //initial value of background a1
parameter(backgroud is estimated as a0+a1*x+a2*x*x)</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Double_t  fA1Calc;                    //calculated value of background a1
parameter</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Double_t  fA1Err;                     //error value of background a1 parameter</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Double_t  fA2Init;                    //initial value of background a2
parameter(backgroud is estimated as a0+a1*x+a2*x*x)</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Double_t  fA2Calc;                    //calculated value of background a2
parameter</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Double_t  fA2Err;                     //error value of background a2 parameter</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Bool_t   *fFixPosition;               //[fNPeaks] array of logical values which
allow to fix appropriate positions (not fit). However they are present in the
estimated functional   </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Bool_t   *fFixAmp;                    //[fNPeaks] array of logical values which
allow to fix appropriate amplitudes (not fit). However they are present in the
estimated functional      </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Bool_t    fFixSigma;                  //logical value of sigma parameter, which
allows to fix the parameter (not to fit).   </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Bool_t    fFixT;                      //logical value of t parameter, which
allows to fix the parameter (not to fit).      </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Bool_t    fFixB;                      //logical value of b parameter, which
allows to fix the parameter (not to fit).   </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Bool_t    fFixS;                      //logical value of s parameter, which
allows to fix the parameter (not to fit).      </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Bool_t    fFixA0;                     //logical value of a0 parameter, which
allows to fix the parameter (not to fit).</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Bool_t    fFixA1;                     //logical value of a1 parameter, which
allows to fix the parameter (not to fit).   </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Bool_t    fFixA2;                     //logical value of a2 parameter, which
allows to fix the parameter (not to fit).</span></p>

<p class=MsoNormal style='text-align:justify'><b><i>&nbsp;</i></b></p>

<p class=MsoNormal style='text-align:justify'><b><i>References:</i></b></p>

<p class=MsoNormal style='text-align:justify'>[1] Phillps G.W., Marlow K.W.,
NIM 137 (1976) 525.</p>

<p class=MsoNormal style='text-align:justify'>[2] I. A. Slavic: Nonlinear
least-squares fitting without matrix inversion applied to complex Gaussian
spectra analysis. NIM 134 (1976) 285-289.</p>

<p class=MsoNormal style='text-align:justify'>[3] T. Awaya: A new method for
curve fitting to the data with low statistics not using chi-square method. NIM
165 (1979) 317-323.</p>

<p class=MsoNormal style='text-align:justify'>[4] T. Hauschild, M. Jentschel:
Comparison of maximum likelihood estimation and chi-square statistics applied
to counting experiments. NIM A 457 (2001) 384-401.</p>

<p class=MsoNormal style='text-align:justify'> [5]  M. Morhá&#269;,  J.
Kliman,  M. Jandel,  &#317;. Krupa, V. Matoušek: Study of fitting algorithms
applied to simultaneous analysis of large number of peaks in -ray spectra. <span
lang=EN-GB>Applied Spectroscopy, Vol. 57, No. 7, pp. 753-760, 2003</span></p>

<p class=MsoNormal style='text-align:justify'> </p>

<p class=MsoNormal style='text-align:justify'><i>Example  – script FitAwmi.c:</i></p>

<p class=MsoNormal style='text-align:justify'><i><span style='font-size:16.0pt'><img
border=0 width=601 height=402 src="gif/spectrumfit_awni_image002.jpg"></span></i></p>

<p class=MsoNormal style='text-align:justify'><b>Fig. 1 Original spectrum
(black line) and fitted spectrum using AWMI algorithm (red line) and number of
iteration steps = 1000. Positions of fitted peaks are denoted by markers</b></p>

<p class=MsoNormal><b><span style='color:#339966'>Script:</span></b></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// Example to illustrate
fitting function using AWMI algorithm.</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// To execute this example,
do</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// root &gt; .x FitAwmi.C</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>&nbsp;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>void FitAwmi() {</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Double_t a;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t
i,nfound=0,bin;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t nbins = 256;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t xmin  = 0;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t xmax  =
nbins;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Double_t * source =
new Double_t[nbins];</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Double_t * dest =
new Double_t[nbins];   </span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   TH1F *h = new
TH1F(&quot;h&quot;,&quot;Fitting using AWMI algorithm&quot;,nbins,xmin,xmax);</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   TH1F *d = new
TH1F(&quot;d&quot;,&quot;&quot;,nbins,xmin,xmax);      </span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   TFile *f = new
TFile(&quot;TSpectrum.root&quot;);</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   h=(TH1F*)
f-&gt;Get(&quot;fit;1&quot;);   </span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   for (i = 0; i &lt;
nbins; i++) source[i]=h-&gt;GetBinContent(i + 1);      </span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   TCanvas *Fit1 =
gROOT-&gt;GetListOfCanvases()-&gt;FindObject(&quot;Fit1&quot;);</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   if (!Fit1) Fit1 =
new TCanvas(&quot;Fit1&quot;,&quot;Fit1&quot;,10,10,1000,700);</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>  
h-&gt;Draw(&quot;L&quot;);</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   TSpectrum *s = new
TSpectrum();</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   //searching for
candidate peaks positions</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   nfound =
s-&gt;SearchHighRes(source, dest, nbins, 2, 0.1, kFALSE, 10000, kFALSE, 0);</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Bool_t *FixPos =
new Bool_t[nfound];</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Bool_t *FixAmp =
new Bool_t[nfound];      </span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   for(i = 0; i&lt;
nfound ; i++){</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>      FixPos[i] =
kFALSE;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>      FixAmp[i] =
kFALSE;    </span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   }</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   //filling in the
initial estimates of the input parameters</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Double_t *PosX =
new Double_t[nfound];         </span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Double_t *PosY =
new Double_t[nfound];</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   PosX =
s-&gt;GetPositionX();</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   for (i = 0; i &lt;
nfound; i++) {</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>                                                a=PosX[i];</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>        bin = 1 +
Int_t(a + 0.5);</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>        PosY[i] =
h-&gt;GetBinContent(bin);</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   }   </span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   TSpectrumFit
*pfit=new TSpectrumFit(nfound);</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>  
pfit-&gt;SetFitParameters(xmin, xmax-1, 1000, 0.1, pfit-&gt;kFitOptimChiCounts,
pfit-&gt;kFitAlphaHalving, pfit-&gt;kFitPower2,
pfit-&gt;kFitTaylorOrderFirst);   </span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   pfit-&gt;SetPeakParameters(2,
kFALSE, PosX, (Bool_t *) FixPos, PosY, (Bool_t *) FixAmp);   </span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>  
pfit-&gt;FitAwmi(source);</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Double_t
*CalcPositions = new Double_t[nfound];      </span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Double_t
*CalcAmplitudes = new Double_t[nfound];         </span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>  
CalcPositions=pfit-&gt;GetPositions();</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>  
CalcAmplitudes=pfit-&gt;GetAmplitudes();   </span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   for (i = 0; i &lt;
nbins; i++) d-&gt;SetBinContent(i + 1,source[i]);</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>  
d-&gt;SetLineColor(kRed);   </span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>  
d-&gt;Draw(&quot;SAME L&quot;);  </span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   for (i = 0; i &lt;
nfound; i++) {</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>                                                a=CalcPositions[i];</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>        bin = 1 +
Int_t(a + 0.5);                </span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>        PosX[i] =
d-&gt;GetBinCenter(bin);</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>        PosY[i] =
d-&gt;GetBinContent(bin);</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   }</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   TPolyMarker * pm =
(TPolyMarker*)h-&gt;GetListOfFunctions()-&gt;FindObject(&quot;TPolyMarker&quot;);</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   if (pm) {</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>     
h-&gt;GetListOfFunctions()-&gt;Remove(pm);</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>      delete pm;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   }</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   pm = new
TPolyMarker(nfound, PosX, PosY);</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>  
h-&gt;GetListOfFunctions()-&gt;Add(pm);</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>  
pm-&gt;SetMarkerStyle(23);</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>  
pm-&gt;SetMarkerColor(kRed);</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>  
pm-&gt;SetMarkerSize(1);   </span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>}</span></p>

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
      delete [] working_space;
      Error ("FitAwmi","All parameters are fixed");
      return;
   }
   if (rozmer >= fXmax - fXmin + 1){
      delete [] working_space;
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
                     if (((a + d) <= 0 && a >= 0) || ((a + d) >= 0 && a <= 0))
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
                  if (((a + d) <= 0 && a >= 0) || ((a + d) >= 0 && a <= 0))
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
               if ((chi2 < chi_min
                    && fStatisticType != kFitOptimMaxLikelihood)
                    || (chi2 > chi_min
                    && fStatisticType == kFitOptimMaxLikelihood)) {
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
      } while (((chi > chi_opt
                 && fStatisticType != kFitOptimMaxLikelihood)
                 || (chi < chi_opt
                 && fStatisticType == kFitOptimMaxLikelihood))
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
void TSpectrumFit::FitStiefel(Double_t *source)
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
<div class=Section3>

<p class=MsoNormal><b><span style='font-size:14.0pt'>Stiefel fitting algorithm</span></b></p>

<p class=MsoNormal style='text-align:justify'><i><span style='font-size:18.0pt'>&nbsp;</span></i></p>

<p class=MsoNormal><i>Function:</i></p>

<p class=MsoNormal style='text-align:justify'>void <a
href="http://root.cern.ch/root/html/TSpectrum.html#TSpectrum:Fit1Awmi"><b>TSpectrumFit::</b></a>FitStiefel(<a
href="http://root.cern.ch/root/html/ListOfTypes.html#double"><b>double</b></a> *fSource)
</p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'>This function fits the source
spectrum using Stiefel-Hestens method [1] (see Awmi function).  The calling
program should fill in input fitting parameters of the TSpectrumFit class using
a set of TSpectrumFit setters. The fitted parameters are written into the class
and the fitted data are written into source spectrum. It converges faster than
Awmi method.</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><i><span style='color:red'>Parameter:</span></i></p>

<p class=MsoNormal style='text-align:justify'>        <b>fSource</b>-pointer to
the vector of source spectrum                  </p>

<p class=MsoNormal style='text-align:justify'>        </p>

<p class=MsoNormal style='text-align:justify'><i>Example – script FitStiefel.c:</i></p>

<p class=MsoNormal style='text-align:justify'><i><span style='font-size:16.0pt'><img
border=0 width=601 height=402 src="gif/spectrumfit_stiefel_image001.jpg"></span></i></p>

<p class=MsoNormal style='text-align:justify'><b>Fig. 2 Original spectrum
(black line) and fitted spectrum using Stiefel-Hestens method (red line) and
number of iteration steps = 100. Positions of fitted peaks are denoted by
markers</b></p>

<p class=MsoNormal><b><span style='color:#339966'>&nbsp;</span></b></p>

<p class=MsoNormal><b><span style='color:#339966'>Script:</span></b></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// Example to illustrate
fitting function using Stiefel-Hestens method.</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// To execute this example, do</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// root &gt; .x FitStiefel.C</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>&nbsp;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>void FitStiefel() {</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t a;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Int_t i,nfound=0,bin;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Int_t nbins = 256;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   </span><span lang=FR
style='font-size:10.0pt'>Int_t xmin  = 0;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t xmax  =
nbins;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   </span><span
style='font-size:10.0pt'>Double_t * source = new Double_t[nbins];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t * dest = new
Double_t[nbins];   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TH1F *h = new TH1F(&quot;h&quot;,&quot;Fitting
using AWMI algorithm&quot;,nbins,xmin,xmax);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TH1F *d = new
TH1F(&quot;d&quot;,&quot;&quot;,nbins,xmin,xmax);      </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TFile *f = new
TFile(&quot;TSpectrum.root&quot;);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   h=(TH1F*)
f-&gt;Get(&quot;fit;1&quot;);   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for (i = 0; i &lt; nbins;
i++) source[i]=h-&gt;GetBinContent(i + 1);      </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TCanvas *Fit1 =
gROOT-&gt;GetListOfCanvases()-&gt;FindObject(&quot;Fit1&quot;);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   if (!Fit1) Fit1 = new
TCanvas(&quot;Fit1&quot;,&quot;Fit1&quot;,10,10,1000,700);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   h-&gt;Draw(&quot;L&quot;);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TSpectrum *s = new
TSpectrum();</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   //searching for candidate
peaks positions</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   nfound =
s-&gt;SearchHighRes(source, dest, nbins, 2, 0.1, kFALSE, 10000, kFALSE, 0);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Bool_t *FixPos = new
Bool_t[nfound];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Bool_t *FixAmp = new
Bool_t[nfound];      </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for(i = 0; i&lt; nfound ;
i++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>      FixPos[i] = kFALSE;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>      FixAmp[i] = kFALSE;    </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   //filling in the initial
estimates of the input parameters</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t *PosX = new
Double_t[nfound];         </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t *PosY = new
Double_t[nfound];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   PosX =
s-&gt;GetPositionX();</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for (i = 0; i &lt; nfound;
i++) {</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                                                a=PosX[i];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>        bin = 1 + Int_t(a +
0.5);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>        PosY[i] =
h-&gt;GetBinContent(bin);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TSpectrumFit *pfit=new
TSpectrumFit(nfound);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
pfit-&gt;SetFitParameters(xmin, xmax-1, 1000, 0.1, pfit-&gt;kFitOptimChiCounts,
pfit-&gt;kFitAlphaHalving, pfit-&gt;kFitPower2,
pfit-&gt;kFitTaylorOrderFirst);   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   pfit-&gt;SetPeakParameters(2,
kFALSE, PosX, (Bool_t *) FixPos, PosY, (Bool_t *) FixAmp);   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
pfit-&gt;FitStiefel(source);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t *CalcPositions =
new Double_t[nfound];      </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   </span><span lang=FR
style='font-size:10.0pt'>Double_t *CalcAmplitudes = new
Double_t[nfound];         </span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   </span><span
style='font-size:10.0pt'>CalcPositions=pfit-&gt;GetPositions();</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
CalcAmplitudes=pfit-&gt;GetAmplitudes();   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for (i = 0; i &lt; nbins;
i++) d-&gt;SetBinContent(i + 1,source[i]);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
d-&gt;SetLineColor(kRed);   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   d-&gt;Draw(&quot;SAME
L&quot;);  </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for (i = 0; i &lt; nfound;
i++) {</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                                                a=CalcPositions[i];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>        bin = 1 + Int_t(a +
0.5);                </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>        PosX[i] =
d-&gt;GetBinCenter(bin);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>        PosY[i] =
d-&gt;GetBinContent(bin);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TPolyMarker * pm =
(TPolyMarker*)h-&gt;GetListOfFunctions()-&gt;FindObject(&quot;TPolyMarker&quot;);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   if (pm) {</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     
h-&gt;GetListOfFunctions()-&gt;Remove(pm);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>      delete pm;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   pm = new
TPolyMarker(nfound, PosX, PosY);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
h-&gt;GetListOfFunctions()-&gt;Add(pm);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   pm-&gt;SetMarkerStyle(23);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
pm-&gt;SetMarkerColor(kRed);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   pm-&gt;SetMarkerSize(1);  
</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>}</span></p>

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
      delete [] working_space;
      return;
   }
   if (rozmer >= fXmax - fXmin + 1){
      Error ("FitAwmi","Number of fitted parameters is larger than # of fitted points");
      delete [] working_space;
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
               if ((chi2 < chi_min
                    && fStatisticType != kFitOptimMaxLikelihood)
                    || (chi2 > chi_min
                    && fStatisticType == kFitOptimMaxLikelihood)) {
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
      } while (((chi > chi_opt
                 && fStatisticType != kFitOptimMaxLikelihood)
                 || (chi < chi_opt
                 && fStatisticType == kFitOptimMaxLikelihood))
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
void TSpectrumFit::SetPeakParameters(Double_t sigma, Bool_t fixSigma, const Double_t *positionInit, const Bool_t *fixPosition, const Double_t *ampInit, const Bool_t *fixAmp)
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

