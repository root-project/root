// @(#)root/spectrum:$Id$
// Author: Miroslav Morhac   25/09/2006

/////////////////////////////////////////////////////////////////////////////
//   THIS CLASS CONTAINS ADVANCED TWO-DIMENSIONAL SPECTRA                  //
//   FITTING FUNCTIONS                                                     //
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

#include "TSpectrum2Fit.h"
#include "TMath.h"

ClassImp(TSpectrum2Fit)  
    
//______________________________________________________________________________
TSpectrum2Fit::TSpectrum2Fit() :TNamed("Spectrum2Fit", "Miroslav Morhac peak fitter") 
{
   //default constructor
   fNPeaks = 0;
   fNumberIterations = 1;  
   fXmin = 0;
   fXmax = 100;
   fYmin = 0;
   fYmax = 100;
   fStatisticType = kFitOptimChiCounts;
   fAlphaOptim    = kFitAlphaHalving;     
   fPower         = kFitPower2;                
   fFitTaylor     = kFitTaylorOrderFirst;  
   fAlpha = 1; 
   fChi   = 0;                    
   fPositionInitX   = 0;
   fPositionCalcX   = 0;
   fPositionErrX    = 0;
   fPositionInitY   = 0;
   fPositionCalcY   = 0;
   fPositionErrY    = 0;
   fPositionInitX1  = 0;
   fPositionCalcX1  = 0;
   fPositionErrX1   = 0;
   fPositionInitY1  = 0;
   fPositionCalcY1  = 0;
   fPositionErrY1   = 0;
   fAmpInit    = 0;
   fAmpCalc    = 0;
   fAmpErr     = 0;
   fAmpInitX1  = 0;
   fAmpCalcX1  = 0;
   fAmpErrX1   = 0;
   fAmpInitY1  = 0;
   fAmpCalcY1  = 0;
   fAmpErrY1   = 0;
   fVolume     = 0;
   fVolumeErr  = 0;
   fSigmaInitX = 2;
   fSigmaCalcX = 0;
   fSigmaErrX  = 0;
   fSigmaInitY = 2;
   fSigmaCalcY = 0;
   fSigmaErrY  = 0;
   fRoInit  = 0;
   fRoCalc  = 0;
   fRoErr   = 0;
   fTxyInit = 0;
   fTxyCalc = 0;
   fTxyErr  = 0;
   fTxInit  = 0; 
   fTxCalc  = 0; 
   fTxErr   = 0; 
   fTyInit  = 0; 
   fTyCalc  = 0; 
   fTyErr   = 0; 
   fBxInit  = 1; 
   fBxCalc  = 0; 
   fBxErr   = 0; 
   fByInit  = 1; 
   fByCalc  = 0; 
   fByErr   = 0; 
   fSxyInit = 0;
   fSxyCalc = 0;
   fSxyErr  = 0;
   fSxInit  = 0; 
   fSxCalc  = 0; 
   fSxErr   = 0; 
   fSyInit  = 0; 
   fSyCalc  = 0; 
   fSyErr   = 0; 
   fA0Init  = 0; 
   fA0Calc  = 0; 
   fA0Err   = 0; 
   fAxInit  = 0;
   fAxCalc  = 0;
   fAxErr   = 0;
   fAyInit  = 0;
   fAyCalc  = 0;
   fAyErr   = 0;
   fFixPositionX    = 0;
   fFixPositionY    = 0;
   fFixPositionX1   = 0;
   fFixPositionY1   = 0;
   fFixAmp     = 0;
   fFixAmpX1   = 0;
   fFixAmpY1   = 0;
   fFixSigmaX  = false;
   fFixSigmaY  = false;
   fFixRo  = true;
   fFixTxy = true;
   fFixTx  = true;   
   fFixTy  = true;   
   fFixBx  = true;
   fFixBy  = true;   
   fFixSxy = true;
   fFixSx  = true;   
   fFixSy  = true;   
   fFixA0  = true;
   fFixAx  = true;
   fFixAy  = true;
         
}   
//______________________________________________________________________________
TSpectrum2Fit::TSpectrum2Fit(Int_t numberPeaks) :TNamed("Spectrum2Fit", "Miroslav Morhac peak fitter") 
{
   //numberPeaks: number of fitted peaks (must be greater than zero)
   //the constructor allocates arrays for all fitted parameters (peak positions, amplitudes etc) and sets the member
   //variables to their default values. One can change these variables by member functions (setters) of TSpectrumFit class.    
//Begin_Html <!--
/* -->
<div class=Section1>

<p class=MsoNormal style='text-align:justify'>Shape function of the fitted
peaks contains the two-dimensional symmetrical Gaussian two one-dimensional
symmetrical Gaussian ridges as well as nonsymmetrical terms and background.</p>

<p class=MsoNormal style='text-align:justify'><sub><span style='font-size:10.0pt'><img
width=600 height=401 src="gif/spectrum2fit_constructor_image001.gif"></span></sub></p>

</div>

<!-- */
// --> End_Html   
   if (numberPeaks <= 0){
      Error ("TSpectrum2Fit","Invalid number of peaks, must be > than 0");
      return;
   }
   fNPeaks = numberPeaks; 
   fNumberIterations = 1;  
   fXmin = 0;
   fXmax = 100;
   fYmin = 0;
   fYmax = 100;
   fStatisticType = kFitOptimChiCounts;
   fAlphaOptim = kFitAlphaHalving;     
   fPower = kFitPower2;                
   fFitTaylor = kFitTaylorOrderFirst;  
   fAlpha = 1; 
   fChi   = 0;                    
   fPositionInitX   = new Double_t[numberPeaks];
   fPositionCalcX   = new Double_t[numberPeaks];   
   fPositionErrX    = new Double_t[numberPeaks];   
   fPositionInitY   = new Double_t[numberPeaks];
   fPositionCalcY   = new Double_t[numberPeaks];   
   fPositionErrY    = new Double_t[numberPeaks];   
   fPositionInitX1  = new Double_t[numberPeaks];
   fPositionCalcX1  = new Double_t[numberPeaks];   
   fPositionErrX1   = new Double_t[numberPeaks];   
   fPositionInitY1  = new Double_t[numberPeaks];
   fPositionCalcY1  = new Double_t[numberPeaks];   
   fPositionErrY1   = new Double_t[numberPeaks];   
   fAmpInit    = new Double_t[numberPeaks];   
   fAmpCalc    = new Double_t[numberPeaks];   
   fAmpErr     = new Double_t[numberPeaks];   
   fAmpInitX1  = new Double_t[numberPeaks];   
   fAmpCalcX1  = new Double_t[numberPeaks];   
   fAmpErrX1   = new Double_t[numberPeaks];   
   fAmpInitY1  = new Double_t[numberPeaks];   
   fAmpCalcY1  = new Double_t[numberPeaks];   
   fAmpErrY1   = new Double_t[numberPeaks];   
   fVolume     = new Double_t[numberPeaks];   
   fVolumeErr  = new Double_t[numberPeaks];      
   fSigmaInitX = 2;
   fSigmaCalcX = 0;
   fSigmaErrX  = 0;
   fSigmaInitY = 2;
   fSigmaCalcY = 0;
   fSigmaErrY  = 0;
   fRoInit  = 0;
   fRoCalc  = 0;
   fRoErr   = 0;
   fTxyInit = 0;
   fTxyCalc = 0;
   fTxyErr  = 0;
   fTxInit  = 0; 
   fTxCalc  = 0; 
   fTxErr   = 0; 
   fTyInit  = 0; 
   fTyCalc  = 0; 
   fTyErr   = 0; 
   fBxInit  = 1; 
   fBxCalc  = 0; 
   fBxErr   = 0; 
   fByInit  = 1; 
   fByCalc  = 0; 
   fByErr   = 0; 
   fSxyInit = 0;
   fSxyCalc = 0;
   fSxyErr  = 0;
   fSxInit  = 0; 
   fSxCalc  = 0; 
   fSxErr   = 0; 
   fSyInit  = 0; 
   fSyCalc  = 0; 
   fSyErr   = 0; 
   fA0Init  = 0; 
   fA0Calc  = 0; 
   fA0Err   = 0; 
   fAxInit  = 0;
   fAxCalc  = 0;
   fAxErr   = 0;
   fAyInit  = 0;
   fAyCalc  = 0;
   fAyErr   = 0;
   fFixPositionX    = new Bool_t[numberPeaks];   
   fFixPositionY    = new Bool_t[numberPeaks];      
   fFixPositionX1   = new Bool_t[numberPeaks];     
   fFixPositionY1   = new Bool_t[numberPeaks];          
   fFixAmp     = new Bool_t[numberPeaks];     
   fFixAmpX1   = new Bool_t[numberPeaks];        
   fFixAmpY1   = new Bool_t[numberPeaks];           
   fFixSigmaX  = false;
   fFixSigmaY  = false;
   fFixRo  = true;
   fFixTxy = true;
   fFixTx  = true;   
   fFixTy  = true;   
   fFixBx  = true;
   fFixBy  = true;   
   fFixSxy = true;
   fFixSx  = true;   
   fFixSy  = true;   
   fFixA0  = true;
   fFixAx  = true;
   fFixAy  = true;
}

//______________________________________________________________________________
TSpectrum2Fit::~TSpectrum2Fit() 
{
   // destructor
   delete [] fPositionInitX;   
   delete [] fPositionCalcX;   
   delete [] fPositionErrX;
   delete [] fFixPositionX;      
   delete [] fPositionInitY;   
   delete [] fPositionCalcY;   
   delete [] fPositionErrY;
   delete [] fFixPositionY;
   delete [] fPositionInitX1;   
   delete [] fPositionCalcX1;   
   delete [] fPositionErrX1;
   delete [] fFixPositionX1;      
   delete [] fPositionInitY1;   
   delete [] fPositionCalcY1;   
   delete [] fPositionErrY1;
   delete [] fFixPositionY1;            
   delete [] fAmpInit;  
   delete [] fAmpCalc;  
   delete [] fAmpErr;   
   delete [] fFixAmp;
   delete [] fAmpInitX1;  
   delete [] fAmpCalcX1;  
   delete [] fAmpErrX1;   
   delete [] fFixAmpX1;
   delete [] fAmpInitY1;  
   delete [] fAmpCalcY1;  
   delete [] fAmpErrY1;   
   delete [] fFixAmpY1;    
   delete [] fVolume;  
   delete [] fVolumeErr;         
}


/////////////////BEGINNING OF AUXILIARY FUNCTIONS USED BY FITTING FUNCTIONS//////////////////////////
//______________________________________________________________________________
Double_t TSpectrum2Fit::Erfc(Double_t x) 
{   
//////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                      //
//                                                                          //
//   This function calculates error function of x.                           //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////
   Double_t da1 = 0.1740121, da2 = -0.0479399, da3 = 0.3739278, dap =
       0.47047;
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

//______________________________________________________________________________
Double_t TSpectrum2Fit::Derfc(Double_t x) 
{  
//////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                      //
//                                                                          //
//   This function calculates derivative of error function of x.             //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////
   Double_t a, t, c, w;
   Double_t da1 = 0.1740121, da2 = -0.0479399, da3 = 0.3739278, dap =
       0.47047;
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

//______________________________________________________________________________
Double_t TSpectrum2Fit::Ourpowl(Double_t a, Int_t pw)
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
void TSpectrum2Fit::StiefelInversion(Double_t **a, Int_t size)
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
         a[i][size + 2] = -a[i][size]; //rk=-C
         for (j = 0; j < size; j++) {
            a[i][size + 2] += a[i][j] * a[j][size + 1]; //A*xk-C
         }
         normk += a[i][size + 2] * a[i][size + 2]; //calculation normk
      }
      
      //calculation of sk
      if (k != 0) {
         sk = normk / normk_old;
      }
      
      //calculation of uk
      for (i = 0; i < size; i++) {
         a[i][size + 3] = -a[i][size + 2] + sk * a[i][size + 3]; //uk=-rk+sk*uk-1
      }
      
      //calculation of lambdak
      lambdak = 0;
      for (i = 0; i < size; i++) {
         for (j = 0, b = 0; j < size; j++) {
            b += a[i][j] * a[j][size + 3]; //A*uk
         }
         lambdak += b * a[i][size + 3];
      }
      if (TMath::Abs(lambdak) > 1e-50) //computer zero
         lambdak = normk / lambdak;
      
      else
         lambdak = 0;
      for (i = 0; i < size; i++)
         a[i][size + 1] += lambdak * a[i][size + 3]; //xk+1=xk+lambdak*uk
      normk_old = normk;
      k += 1;
   } while (k < size && TMath::Abs(normk) > 1e-50); //computer zero
   return;
}

//______________________________________________________________________________
Double_t TSpectrum2Fit::Shape2(Int_t numOfFittedPeaks, Double_t x, Double_t y,
                            const Double_t *parameter, Double_t sigmax,
                            Double_t sigmay, Double_t ro, Double_t a0, Double_t ax,
                            Double_t ay, Double_t txy, Double_t sxy, Double_t tx,
                            Double_t ty, Double_t sx, Double_t sy, Double_t bx,
                            Double_t by) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates 2D peaks shape function (see manual)               //
//      Function parameters:                                                    //
//              -numOfFittedPeaks-number of fitted peaks                        //
//              -x-channel in x-dimension                                       //
//              -y-channel in y-dimension                                       //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigmax-sigmax of peaks                                         //
//              -sigmay-sigmay of peaks                                         //
//              -ro-correlation coefficient                                     //
//              -a0,ax,ay-bac kground coefficients                              //
//              -txy,tx,ty, sxy,sy,sx-relative amplitudes                       //
//              -bx, by-slopes                                                  //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Int_t j;
   Double_t r, p, r1, e, ex, ey, vx, s2, px, py, rx, ry, erx, ery;
   vx = 0;
   s2 = TMath::Sqrt(2.0);
   for (j = 0; j < numOfFittedPeaks; j++) {
      p = (x - parameter[7 * j + 1]) / sigmax;
      r = (y - parameter[7 * j + 2]) / sigmay;
      if (TMath::Abs(p) < 3 && TMath::Abs(r) < 3) {
         e = (p * p - 2 * ro * p * r + r * r) / (2 * (1 - ro * ro));
         if (e < 700)
            r1 = exp(-e);
         
         else {
            r1 = 0;
         }
         if (txy != 0) {
            px = 0, py = 0;
            erx = Erfc(p / s2 + 1 / (2 * bx)), ery =
                Erfc(r / s2 + 1 / (2 * by));
            ex = p / (s2 * bx), ey = r / (s2 * by);
            if (TMath::Abs(ex) < 9 && TMath::Abs(ey) < 9) {
               px = exp(ex) * erx, py = exp(ey) * ery;
            }
            r1 += 0.5 * txy * px * py;
         }
         if (sxy != 0) {
            rx = Erfc(p / s2), ry = Erfc(r / s2);
            r1 += 0.5 * sxy * rx * ry;
         }
         vx = vx + parameter[7 * j] * r1;
      }
      p = (x - parameter[7 * j + 5]) / sigmax;
      if (TMath::Abs(p) < 3) {
         e = p * p / 2;
         if (e < 700)
            r1 = exp(-e);
         
         else {
            r1 = 0;
         }
         if (tx != 0) {
            px = 0;
            erx = Erfc(p / s2 + 1 / (2 * bx));
            ex = p / (s2 * bx);
            if (TMath::Abs(ex) < 9) {
               px = exp(ex) * erx;
            }
            r1 += 0.5 * tx * px;
         }
         if (sx != 0) {
            rx = Erfc(p / s2);
            r1 += 0.5 * sx * rx;
         }
         vx = vx + parameter[7 * j + 3] * r1;
      }
      r = (y - parameter[7 * j + 6]) / sigmay;
      if (TMath::Abs(r) < 3) {
         e = r * r / 2;
         if (e < 700)
            r1 = exp(-e);
         
         else {
            r1 = 0;
         }
         if (ty != 0) {
            py = 0;
            ery = Erfc(r / s2 + 1 / (2 * by));
            ey = r / (s2 * by);
            if (TMath::Abs(ey) < 9) {
               py = exp(ey) * ery;
            }
            r1 += 0.5 * ty * py;
         }
         if (sy != 0) {
            ry = Erfc(r / s2);
            r1 += 0.5 * sy * ry;
         }
         vx = vx + parameter[7 * j + 4] * r1;
      }
   }
   vx = vx + a0 + ax * x + ay * y;
   return (vx);
}

//______________________________________________________________________________
Double_t TSpectrum2Fit::Deramp2(Double_t x, Double_t y, Double_t x0, Double_t y0,
                            Double_t sigmax, Double_t sigmay, Double_t ro,
                            Double_t txy, Double_t sxy, Double_t bx, Double_t by) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates derivative of 2D peaks shape function (see manual) //
//   according to amplitude of 2D peak                                          //
//      Function parameters:                                                    //
//              -x-channel in x-dimension                                       //
//              -y-channel in y-dimension                                       //
//              -x0-position of peak in x-dimension                             //
//              -y0-position of peak in y-dimension                             //
//              -sigmax-sigmax of peaks                                         //
//              -sigmay-sigmay of peaks                                         //
//              -ro-correlation coefficient                                     //
//              -txy, sxy-relative amplitudes                                   //
//              -bx, by-slopes                                                  //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Double_t p, r, r1 = 0, e, ex, ey, px, py, rx, ry, erx, ery, s2;
   p = (x - x0) / sigmax;
   r = (y - y0) / sigmay;
   if (TMath::Abs(p) < 3 && TMath::Abs(r) < 3) {
      s2 = TMath::Sqrt(2.0);
      e = (p * p - 2 * ro * p * r + r * r) / (2 * (1 - ro * ro));
      if (e < 700)
         r1 = exp(-e);
      
      else {
         r1 = 0;
      }
      if (txy != 0) {
         px = 0, py = 0;
         erx = Erfc(p / s2 + 1 / (2 * bx)), ery =
             Erfc(r / s2 + 1 / (2 * by));
         ex = p / (s2 * bx), ey = r / (s2 * by);
         if (TMath::Abs(ex) < 9 && TMath::Abs(ey) < 9) {
            px = exp(ex) * erx, py = exp(ey) * ery;
         }
         r1 += 0.5 * txy * px * py;
      }
      if (sxy != 0) {
         rx = Erfc(p / s2), ry = Erfc(r / s2);
         r1 += 0.5 * sxy * rx * ry;
      }
   }
   return (r1);
}

//______________________________________________________________________________
Double_t TSpectrum2Fit::Derampx(Double_t x, Double_t x0, Double_t sigmax, Double_t tx,
                             Double_t sx, Double_t bx) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates derivative of 2D peaks shape function (see manual) //
//   according to amplitude of the ridge                                        //
//      Function parameters:                                                    //
//              -x-channel in x-dimension                                       //
//              -x0-position of peak in x-dimension                             //
//              -y0-position of peak in y-dimension                             //
//              -sigmax-sigmax of peaks                                         //
//              -ro-correlation coefficient                                     //
//              -tx, sx-relative amplitudes                                     //
//              -bx-slope                                                       //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Double_t p, r1 = 0, px, erx, rx, ex, s2;
   p = (x - x0) / sigmax;
   if (TMath::Abs(p) < 3) {
      s2 = TMath::Sqrt(2.0);
      p = p * p / 2;
      if (p < 700)
         r1 = exp(-p);
      
      else {
         r1 = 0;
      }
      if (tx != 0) {
         px = 0;
         erx = Erfc(p / s2 + 1 / (2 * bx));
         ex = p / (s2 * bx);
         if (TMath::Abs(ex) < 9) {
            px = exp(ex) * erx;
         }
         r1 += 0.5 * tx * px;
      }
      if (sx != 0) {
         rx = Erfc(p / s2);
         r1 += 0.5 * sx * rx;
      }
   }
   return (r1);
}

//______________________________________________________________________________
Double_t TSpectrum2Fit::Deri02(Double_t x, Double_t y, Double_t a, Double_t x0,
                            Double_t y0, Double_t sigmax, Double_t sigmay,
                            Double_t ro, Double_t txy, Double_t sxy, Double_t bx,
                            Double_t by) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates derivative of 2D peaks shape function (see manual) //
//   according to x position of 2D peak                                         //
//      Function parameters:                                                    //
//              -x-channel in x-dimension                                       //
//              -y-channel in y-dimension                                       //
//              -a-amplitude                                                    //
//              -x0-position of peak in x-dimension                             //
//              -y0-position of peak in y-dimension                             //
//              -sigmax-sigmax of peaks                                         //
//              -sigmay-sigmay of peaks                                         //
//              -ro-correlation coefficient                                     //
//              -txy, sxy-relative amplitudes                                   //
//              -bx, by-slopes                                                  //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Double_t p, r, r1 = 0, e, ex, ey, px, py, rx, ry, erx, ery, s2;
   p = (x - x0) / sigmax;
   r = (y - y0) / sigmay;
   if (TMath::Abs(p) < 3 && TMath::Abs(r) < 3) {
      s2 = TMath::Sqrt(2.0);
      e = (p * p - 2 * ro * p * r + r * r) / (2 * (1 - ro * ro));
      if (e < 700)
         r1 = exp(-e);
      
      else {
         r1 = 0;
      }
      e = -(ro * r - p) / sigmax;
      e = e / (1 - ro * ro);
      r1 = r1 * e;
      if (txy != 0) {
         px = 0, py = 0;
         erx =
             (-Erfc(p / s2 + 1 / (2 * bx)) / (s2 * bx * sigmax) -
              Derfc(p / s2 + 1 / (2 * bx)) / (s2 * sigmax)), ery =
             Erfc(r / s2 + 1 / (2 * by));
         ex = p / (s2 * bx), ey = r / (s2 * by);
         if (TMath::Abs(ex) < 9 && TMath::Abs(ey) < 9) {
            px = exp(ex) * erx, py = exp(ey) * ery;
         }
         r1 += 0.5 * txy * px * py;
      }
      if (sxy != 0) {
         rx = -Derfc(p / s2) / (s2 * sigmax), ry = Erfc(r / s2);
         r1 += 0.5 * sxy * rx * ry;
      }
      r1 = a * r1;
   }
   return (r1);
}

//______________________________________________________________________________
Double_t TSpectrum2Fit::Derderi02(Double_t x, Double_t y, Double_t a, Double_t x0,
                              Double_t y0, Double_t sigmax, Double_t sigmay,
                              Double_t ro) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates second derivative of 2D peaks shape function       //
//   (see manual) according to x position of 2D peak                            //
//      Function parameters:                                                    //
//              -x-channel in x-dimension                                       //
//              -y-channel in y-dimension                                       //
//              -a-amplitude                                                    //
//              -x0-position of peak in x-dimension                             //
//              -y0-position of peak in y-dimension                             //
//              -sigmax-sigmax of peaks                                         //
//              -sigmay-sigmay of peaks                                         //
//              -ro-correlation coefficient                                     //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Double_t p, r, r1 = 0, e;
   p = (x - x0) / sigmax;
   r = (y - y0) / sigmay;
   if (TMath::Abs(p) < 3 && TMath::Abs(r) < 3) {
      e = (p * p - 2 * ro * p * r + r * r) / (2 * (1 - ro * ro));
      if (e < 700)
         r1 = exp(-e);
      
      else {
         r1 = 0;
      }
      e = -(ro * r - p) / sigmax;
      e = e / (1 - ro * ro);
      r1 = r1 * (e * e - 1 / ((1 - ro * ro) * sigmax * sigmax));
      r1 = a * r1;
   }
   return (r1);
}
Double_t TSpectrum2Fit::Derj02(Double_t x, Double_t y, Double_t a, Double_t x0,
                            Double_t y0, Double_t sigmax, Double_t sigmay,
                            Double_t ro, Double_t txy, Double_t sxy, Double_t bx,
                            Double_t by) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates derivative of 2D peaks shape function (see manual) //
//   according to y position of 2D peak                                         //
//      Function parameters:                                                    //
//              -x-channel in x-dimension                                       //
//              -y-channel in y-dimension                                       //
//              -a-amplitude                                                    //
//              -x0-position of peak in x-dimension                             //
//              -y0-position of peak in y-dimension                             //
//              -sigmax-sigmax of peaks                                         //
//              -sigmay-sigmay of peaks                                         //
//              -ro-correlation coefficient                                     //
//              -txy, sxy-relative amplitudes                                   //
//              -bx, by-slopes                                                  //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Double_t p, r, r1 = 0, e, ex, ey, px, py, rx, ry, erx, ery, s2;
   p = (x - x0) / sigmax;
   r = (y - y0) / sigmay;
   if (TMath::Abs(p) < 3 && TMath::Abs(r) < 3) {
      s2 = TMath::Sqrt(2.0);
      e = (p * p - 2 * ro * p * r + r * r) / (2 * (1 - ro * ro));
      if (e < 700)
         r1 = exp(-e);
      
      else {
         r1 = 0;
      }
      e = -(ro * p - r) / sigmay;
      e = e / (1 - ro * ro);
      r1 = r1 * e;
      if (txy != 0) {
         px = 0, py = 0;
         ery =
             (-Erfc(r / s2 + 1 / (2 * by)) / (s2 * by * sigmay) -
              Derfc(r / s2 + 1 / (2 * by)) / (s2 * sigmay)), erx =
             Erfc(p / s2 + 1 / (2 * bx));
         ex = p / (s2 * bx), ey = r / (s2 * by);
         if (TMath::Abs(ex) < 9 && TMath::Abs(ey) < 9) {
            px = exp(ex) * erx, py = exp(ey) * ery;
         }
         r1 += 0.5 * txy * px * py;
      }
      if (sxy != 0) {
         ry = -Derfc(r / s2) / (s2 * sigmay), rx = Erfc(p / s2);
         r1 += 0.5 * sxy * rx * ry;
      }
      r1 = a * r1;
   }
   return (r1);
}

//______________________________________________________________________________
Double_t TSpectrum2Fit::Derderj02(Double_t x, Double_t y, Double_t a, Double_t x0,
                               Double_t y0, Double_t sigmax, Double_t sigmay,
                               Double_t ro) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates second derivative of 2D peaks shape function       //
//   (see manual) according to y position of 2D peak                            //
//      Function parameters:                                                    //
//              -x-channel in x-dimension                                       //
//              -y-channel in y-dimension                                       //
//              -a-amplitude                                                    //
//              -x0-position of peak in x-dimension                             //
//              -y0-position of peak in y-dimension                             //
//              -sigmax-sigmax of peaks                                         //
//              -sigmay-sigmay of peaks                                         //
//              -ro-correlation coefficient                                     //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Double_t p, r, r1 = 0, e;
   p = (x - x0) / sigmax;
   r = (y - y0) / sigmay;
   if (TMath::Abs(p) < 3 && TMath::Abs(r) < 3) {
      e = (p * p - 2 * ro * p * r + r * r) / (2 * (1 - ro * ro));
      if (e < 700)
         r1 = exp(-e);
      
      else {
         r1 = 0;
      }
      e = -(ro * p - r) / sigmay;
      e = e / (1 - ro * ro);
      r1 = r1 * (e * e - 1 / ((1 - ro * ro) * sigmay * sigmay));
      r1 = a * r1;
   }
   return (r1);
}

//______________________________________________________________________________
Double_t TSpectrum2Fit::Deri01(Double_t x, Double_t ax, Double_t x0, Double_t sigmax,
                            Double_t tx, Double_t sx, Double_t bx) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates derivative of 2D peaks shape function (see manual) //
//   according to x position of 1D ridge                                        //
//      Function parameters:                                                    //
//              -x-channel in x-dimension                                       //
//              -ax-amplitude of ridge                                          //
//              -x0-position of peak in x-dimension                             //
//              -sigmax-sigmax of peaks                                         //
//              -ro-correlation coefficient                                     //
//              -tx, sx-relative amplitudes                                     //
//              -bx-slope                                                       //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Double_t p, e, r1 = 0, px, rx, erx, ex, s2;
   p = (x - x0) / sigmax;
   if (TMath::Abs(p) < 3) {
      s2 = TMath::Sqrt(2.0);
      e = p * p / 2;
      if (e < 700)
         r1 = exp(-e);
      
      else {
         r1 = 0;
      }
      r1 = r1 * p / sigmax;
      if (tx != 0) {
         px = 0;
         erx =
             (-Erfc(p / s2 + 1 / (2 * bx)) / (s2 * bx * sigmax) -
              Derfc(p / s2 + 1 / (2 * bx)) / (s2 * sigmax));
         ex = p / (s2 * bx);
         if (TMath::Abs(ex) < 9)
            px = exp(ex) * erx;
         r1 += 0.5 * tx * px;
      }
      if (sx != 0) {
         rx = -Derfc(p / s2) / (s2 * sigmax);
         r1 += 0.5 * sx * rx;
      }
      r1 = ax * r1;
   }
   return (r1);
}

//______________________________________________________________________________
Double_t TSpectrum2Fit::Derderi01(Double_t x, Double_t ax, Double_t x0,
                               Double_t sigmax) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates second derivative of 2D peaks shape function       //
//   (see manual) according to x position of 1D ridge                           //
//      Function parameters:                                                    //
//              -x-channel in x-dimension                                       //
//              -ax-amplitude of ridge                                          //
//              -x0-position of peak in x-dimension                             //
//              -sigmax-sigmax of peaks                                         //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Double_t p, e, r1 = 0;
   p = (x - x0) / sigmax;
   if (TMath::Abs(p) < 3) {
      e = p * p / 2;
      if (e < 700)
         r1 = exp(-e);
      
      else {
         r1 = 0;
      }
      r1 = r1 * (p * p / (sigmax * sigmax) - 1 / (sigmax * sigmax));
      r1 = ax * r1;
   }
   return (r1);
}

//______________________________________________________________________________
Double_t TSpectrum2Fit::Dersigmax(Int_t numOfFittedPeaks, Double_t x, Double_t y,
                               const Double_t *parameter, Double_t sigmax,
                               Double_t sigmay, Double_t ro, Double_t txy,
                               Double_t sxy, Double_t tx, Double_t sx, Double_t bx,
                               Double_t by) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates derivative of peaks shape function (see manual)    //
//   according to sigmax of peaks.                                              //
//      Function parameters:                                                    //
//              -numOfFittedPeaks-number of fitted peaks                        //
//              -x,y-position of channel                                        //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigmax-sigmax of peaks                                         //
//              -sigmay-sigmay of peaks                                         //
//              -ro-correlation coefficient                                     //
//              -txy, sxy, tx, sx-relative amplitudes                           //
//              -bx, by-slopes                                                  //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Double_t p, r, r1 =
       0, e, a, b, x0, y0, s2, px, py, rx, ry, erx, ery, ex, ey;
   Int_t j;
   s2 = TMath::Sqrt(2.0);
   for (j = 0; j < numOfFittedPeaks; j++) {
      a = parameter[7 * j];
      x0 = parameter[7 * j + 1];
      y0 = parameter[7 * j + 2];
      p = (x - x0) / sigmax;
      r = (y - y0) / sigmay;
      if (TMath::Abs(p) < 3 && TMath::Abs(r) < 3) {
         e = (p * p - 2 * ro * p * r + r * r) / (2 * (1 - ro * ro));
         if (e < 700)
            e = exp(-e);
         
         else {
            e = 0;
         }
         b = -(ro * p * r - p * p) / sigmax;
         e = e * b / (1 - ro * ro);
         if (txy != 0) {
            px = 0, py = 0;
            erx =
                -Erfc(p / s2 + 1 / (2 * bx)) * p / (s2 * bx * sigmax) -
                Derfc(p / s2 + 1 / (2 * bx)) * p / (s2 * sigmax), ery =
                Erfc(r / s2 + 1 / (2 * by));
            ex = p / (s2 * bx), ey = r / (s2 * by);
            if (TMath::Abs(ex) < 9 && TMath::Abs(ey) < 9) {
               px = exp(ex) * erx, py = exp(ey) * ery;
            }
            e += 0.5 * txy * px * py;
         }
         if (sxy != 0) {
            rx = -Derfc(p / s2) * p / (s2 * sigmax), ry = Erfc(r / s2);
            e += 0.5 * sxy * rx * ry;
         }
         r1 = r1 + a * e;
      }
      if (TMath::Abs(p) < 3) {
         x0 = parameter[7 * j + 5];
         p = (x - x0) / sigmax;
         b = p * p / 2;
         if (b < 700)
            e = exp(-b);
         
         else {
            e = 0;
         }
         e = 2 * b * e / sigmax;
         if (tx != 0) {
            px = 0;
            erx =
                (-Erfc(p / s2 + 1 / (2 * bx)) * p / (s2 * bx * sigmax) -
                 Derfc(p / s2 + 1 / (2 * bx)) * p / (s2 * sigmax));
            ex = p / (s2 * bx);
            if (TMath::Abs(ex) < 9)
               px = exp(ex) * erx;
            e += 0.5 * tx * px;
         }
         if (sx != 0) {
            rx = -Derfc(p / s2) * p / (s2 * sigmax);
            e += 0.5 * sx * rx;
         }
         r1 += parameter[7 * j + 3] * e;
      }
   }
   return (r1);
}

//______________________________________________________________________________
Double_t TSpectrum2Fit::Derdersigmax(Int_t numOfFittedPeaks, Double_t x,
                                  Double_t y, const Double_t *parameter,
                                  Double_t sigmax, Double_t sigmay,
                                  Double_t ro) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates second derivative of peaks shape function          //
//   (see manual) according to sigmax of peaks.                                 //
//      Function parameters:                                                    //
//              -numOfFittedPeaks-number of fitted peaks                        //
//              -x,y-position of channel                                        //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigmax-sigmax of peaks                                         //
//              -sigmay-sigmay of peaks                                         //
//              -ro-correlation coefficient                                     //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Double_t p, r, r1 = 0, e, a, b, x0, y0;
   Int_t j;
   for (j = 0; j < numOfFittedPeaks; j++) {
      a = parameter[7 * j];
      x0 = parameter[7 * j + 1];
      y0 = parameter[7 * j + 2];
      p = (x - x0) / sigmax;
      r = (y - y0) / sigmay;
      if (TMath::Abs(p) < 3 && TMath::Abs(r) < 3) {
         e = (p * p - 2 * ro * p * r + r * r) / (2 * (1 - ro * ro));
         if (e < 700)
            e = exp(-e);
         
         else {
            e = 0;
         }
         b = -(ro * p * r - p * p) / sigmax;
         e = e * (b * b / (1 - ro * ro) -
                   (3 * p * p - 2 * ro * p * r) / (sigmax * sigmax)) / (1 -
                                                                        ro
                                                                        *
                                                                        ro);
         r1 = r1 + a * e;
      }
      if (TMath::Abs(p) < 3) {
         x0 = parameter[7 * j + 5];
         p = (x - x0) / sigmax;
         b = p * p / 2;
         if (b < 700)
            e = exp(-b);
         
         else {
            e = 0;
         }
         e = e * (4 * b * b - 6 * b) / (sigmax * sigmax);
         r1 += parameter[7 * j + 3] * e;
      }
   }
   return (r1);
}

//______________________________________________________________________________
Double_t TSpectrum2Fit::Dersigmay(Int_t numOfFittedPeaks, Double_t x, Double_t y,
                               const Double_t *parameter, Double_t sigmax,
                               Double_t sigmay, Double_t ro, Double_t txy,
                               Double_t sxy, Double_t ty, Double_t sy, Double_t bx,
                               Double_t by) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates derivative of peaks shape function (see manual)    //
//   according to sigmax of peaks.                                              //
//      Function parameters:                                                    //
//              -numOfFittedPeaks-number of fitted peaks                        //
//              -x,y-position of channel                                        //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigmax-sigmax of peaks                                         //
//              -sigmay-sigmay of peaks                                         //
//              -ro-correlation coefficient                                     //
//              -txy, sxy, ty, sy-relative amplitudes                           //
//              -bx, by-slopes                                                  //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Double_t p, r, r1 =
       0, e, a, b, x0, y0, s2, px, py, rx, ry, erx, ery, ex, ey;
   Int_t j;
   s2 = TMath::Sqrt(2.0);
   for (j = 0; j < numOfFittedPeaks; j++) {
      a = parameter[7 * j];
      x0 = parameter[7 * j + 1];
      y0 = parameter[7 * j + 2];
      p = (x - x0) / sigmax;
      r = (y - y0) / sigmay;
      if (TMath::Abs(p) < 3 && TMath::Abs(r) < 3) {
         e = (p * p - 2 * ro * p * r + r * r) / (2 * (1 - ro * ro));
         if (e < 700)
            e = exp(-e);
         
         else {
            e = 0;
         }
         b = -(ro * p * r - r * r) / sigmay;
         e = e * b / (1 - ro * ro);
         if (txy != 0) {
            px = 0, py = 0;
            ery =
                -Erfc(r / s2 + 1 / (2 * by)) * r / (s2 * by * sigmay) -
                Derfc(r / s2 + 1 / (2 * by)) * r / (s2 * sigmay), erx =
                Erfc(p / s2 + 1 / (2 * bx));
            ex = p / (s2 * bx), ey = r / (s2 * by);
            if (TMath::Abs(ex) < 9 && TMath::Abs(ey) < 9) {
               px = exp(ex) * erx, py = exp(ey) * ery;
            }
            e += 0.5 * txy * px * py;
         }
         if (sxy != 0) {
            ry = -Derfc(r / s2) * r / (s2 * sigmay), rx = Erfc(p / s2);
            e += 0.5 * sxy * rx * ry;
         }
         r1 = r1 + a * e;
      }
      if (TMath::Abs(r) < 3) {
         y0 = parameter[7 * j + 6];
         r = (y - y0) / sigmay;
         b = r * r / 2;
         if (b < 700)
            e = exp(-b);
         
         else {
            e = 0;
         }
         e = 2 * b * e / sigmay;
         if (ty != 0) {
            py = 0;
            ery =
                (-Erfc(r / s2 + 1 / (2 * by)) * r / (s2 * by * sigmay) -
                 Derfc(r / s2 + 1 / (2 * by)) * r / (s2 * sigmay));
            ey = r / (s2 * by);
            if (TMath::Abs(ey) < 9)
               py = exp(ey) * ery;
            e += 0.5 * ty * py;
         }
         if (sy != 0) {
            ry = -Derfc(r / s2) * r / (s2 * sigmay);
            e += 0.5 * sy * ry;
         }
         r1 += parameter[7 * j + 4] * e;
      }
   }
   return (r1);
}

//______________________________________________________________________________
Double_t TSpectrum2Fit::Derdersigmay(Int_t numOfFittedPeaks, Double_t x,
                                  Double_t y, const Double_t *parameter,
                                  Double_t sigmax, Double_t sigmay,
                                  Double_t ro) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates second derivative of peaks shape function          //
//   (see manual) according to sigmay of peaks.                                 //
//      Function parameters:                                                    //
//              -numOfFittedPeaks-number of fitted peaks                        //
//              -x,y-position of channel                                        //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigmax-sigmax of peaks                                         //
//              -sigmay-sigmay of peaks                                         //
//              -ro-correlation coefficient                                     //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Double_t p, r, r1 = 0, e, a, b, x0, y0;
   Int_t j;
   for (j = 0; j < numOfFittedPeaks; j++) {
      a = parameter[7 * j];
      x0 = parameter[7 * j + 1];
      y0 = parameter[7 * j + 2];
      p = (x - x0) / sigmax;
      r = (y - y0) / sigmay;
      if (TMath::Abs(p) < 3 && TMath::Abs(r) < 3) {
         e = (p * p - 2 * ro * p * r + r * r) / (2 * (1 - ro * ro));
         if (e < 700)
            e = exp(-e);
         
         else {
            e = 0;
         }
         b = -(ro * p * r - r * r) / sigmay;
         e = e * (b * b / (1 - ro * ro) -
                   (3 * r * r - 2 * ro * r * p) / (sigmay * sigmay)) / (1 -
                                                                        ro
                                                                        *
                                                                        ro);
         r1 = r1 + a * e;
      }
      if (TMath::Abs(r) < 3) {
         y0 = parameter[7 * j + 6];
         r = (y - y0) / sigmay;
         b = r * r / 2;
         if (b < 700)
            e = exp(-b);
         
         else {
            e = 0;
         }
         e = e * (4 * b * b - 6 * b) / (sigmay * sigmay);
         r1 += parameter[7 * j + 4] * e;
      }
   }
   return (r1);
}

//______________________________________________________________________________
Double_t TSpectrum2Fit::Derro(Int_t numOfFittedPeaks, Double_t x, Double_t y,
                           const Double_t *parameter, Double_t sx, Double_t sy,
                           Double_t r) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates derivative of peaks shape function (see manual)    //
//   according to correlation coefficient ro.                                   //
//      Function parameters:                                                    //
//              -numOfFittedPeaks-number of fitted peaks                        //
//              -x,y-position of channel                                        //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sx-sigmax of peaks                                             //
//              -sy-sigmay of peaks                                             //
//              -r-correlation coefficient ro                                   //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Double_t px, qx, rx, vx, x0, y0, a, ex, tx;
   Int_t j;
   vx = 0;
   for (j = 0; j < numOfFittedPeaks; j++) {
      a = parameter[7 * j];
      x0 = parameter[7 * j + 1];
      y0 = parameter[7 * j + 2];
      px = (x - x0) / sx;
      qx = (y - y0) / sy;
      if (TMath::Abs(px) < 3 && TMath::Abs(qx) < 3) {
         rx = (px * px - 2 * r * px * qx + qx * qx);
         ex = rx / (2 * (1 - r * r));
         if ((ex) < 700)
            ex = exp(-ex);
         
         else {
            ex = 0;
         }
         tx = px * qx / (1 - r * r);
         tx = tx - r * rx / ((1 - r * r) * (1 - r * r));
         vx = vx + a * ex * tx;
      }
   }
   return (vx);
}

//______________________________________________________________________________
Double_t TSpectrum2Fit::Dertxy(Int_t numOfFittedPeaks, Double_t x, Double_t y,
                            const Double_t *parameter, Double_t sigmax,
                            Double_t sigmay, Double_t bx, Double_t by) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates derivative of peaks shape function (see manual)    //
//   according to relative amplitude txy.                                       //
//      Function parameters:                                                    //
//              -numOfFittedPeaks-number of fitted peaks                        //
//              -x,y-position of channel                                        //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigmax-sigmax of peaks                                         //
//              -sigmay-sigmay of peaks                                         //
//              -bx, by-slopes                                                  //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Double_t p, r, r1 = 0, ex, ey, px, py, erx, ery, s2, x0, y0, a;
   Int_t j;
   s2 = TMath::Sqrt(2.0);
   for (j = 0; j < numOfFittedPeaks; j++) {
      a = parameter[7 * j];
      x0 = parameter[7 * j + 1];
      y0 = parameter[7 * j + 2];
      p = (x - x0) / sigmax;
      r = (y - y0) / sigmay;
      px = 0, py = 0;
      erx = Erfc(p / s2 + 1 / (2 * bx)), ery =
          Erfc(r / s2 + 1 / (2 * by));
      ex = p / (s2 * bx), ey = r / (s2 * by);
      if (TMath::Abs(ex) < 9 && TMath::Abs(ey) < 9) {
         px = exp(ex) * erx, py = exp(ey) * ery;
      }
      r1 += 0.5 * a * px * py;
   }
   return (r1);
}

//______________________________________________________________________________
Double_t TSpectrum2Fit::Dersxy(Int_t numOfFittedPeaks, Double_t x, Double_t y,
                            const Double_t *parameter, Double_t sigmax,
                            Double_t sigmay) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates derivative of peaks shape function (see manual)    //
//   according to relative amplitude sxy.                                       //
//      Function parameters:                                                    //
//              -numOfFittedPeaks-number of fitted peaks                        //
//              -x,y-position of channel                                        //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigmax-sigmax of peaks                                         //
//              -sigmay-sigmay of peaks                                         //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Double_t p, r, r1 = 0, rx, ry, x0, y0, a, s2;
   Int_t j;
   s2 = TMath::Sqrt(2.0);
   for (j = 0; j < numOfFittedPeaks; j++) {
      a = parameter[7 * j];
      x0 = parameter[7 * j + 1];
      y0 = parameter[7 * j + 2];
      p = (x - x0) / sigmax;
      r = (y - y0) / sigmay;
      rx = Erfc(p / s2), ry = Erfc(r / s2);
      r1 += 0.5 * a * rx * ry;
   }
   return (r1);
}

//______________________________________________________________________________
Double_t TSpectrum2Fit::Dertx(Int_t numOfFittedPeaks, Double_t x,
                           const Double_t *parameter, Double_t sigmax,
                           Double_t bx) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates derivative of peaks shape function (see manual)    //
//   according to relative amplitude tx.                                        //
//      Function parameters:                                                    //
//              -numOfFittedPeaks-number of fitted peaks                        //
//              -x-position of channel                                          //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigmax-sigma of 1D ridge                                       //
//              -bx-slope                                                       //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Double_t p, r1 = 0, ex, px, erx, s2, ax, x0;
   Int_t j;
   s2 = TMath::Sqrt(2.0);
   for (j = 0; j < numOfFittedPeaks; j++) {
      ax = parameter[7 * j + 3];
      x0 = parameter[7 * j + 5];
      p = (x - x0) / sigmax;
      px = 0;
      erx = Erfc(p / s2 + 1 / (2 * bx));
      ex = p / (s2 * bx);
      if (TMath::Abs(ex) < 9) {
         px = exp(ex) * erx;
      }
      r1 += 0.5 * ax * px;
   }
   return (r1);
}

//______________________________________________________________________________
Double_t TSpectrum2Fit::Derty(Int_t numOfFittedPeaks, Double_t x,
                           const Double_t *parameter, Double_t sigmax,
                           Double_t bx) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates derivative of peaks shape function (see manual)    //
//   according to relative amplitude ty.                                        //
//      Function parameters:                                                    //
//              -numOfFittedPeaks-number of fitted peaks                        //
//              -x-position of channel                                          //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigmax-sigma of 1D ridge                                       //
//              -bx-slope                                                       //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Double_t p, r1 = 0, ex, px, erx, s2, ax, x0;
   Int_t j;
   s2 = TMath::Sqrt(2.0);
   for (j = 0; j < numOfFittedPeaks; j++) {
      ax = parameter[7 * j + 4];
      x0 = parameter[7 * j + 6];
      p = (x - x0) / sigmax;
      px = 0;
      erx = Erfc(p / s2 + 1 / (2 * bx));
      ex = p / (s2 * bx);
      if (TMath::Abs(ex) < 9) {
         px = exp(ex) * erx;
      }
      r1 += 0.5 * ax * px;
   }
   return (r1);
}

//______________________________________________________________________________
Double_t TSpectrum2Fit::Dersx(Int_t numOfFittedPeaks, Double_t x,
                           const Double_t *parameter, Double_t sigmax) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates derivative of peaks shape function (see manual)    //
//   according to relative amplitude sx.                                        //
//      Function parameters:                                                    //
//              -numOfFittedPeaks-number of fitted peaks                        //
//              -x-position of channel                                          //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigmax-sigma of 1D ridge                                       //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Double_t p, r1 = 0, rx, ax, x0, s2;
   Int_t j;
   s2 = TMath::Sqrt(2.0);
   for (j = 0; j < numOfFittedPeaks; j++) {
      ax = parameter[7 * j + 3];
      x0 = parameter[7 * j + 5];
      p = (x - x0) / sigmax;
      s2 = TMath::Sqrt(2.0);
      rx = Erfc(p / s2);
      r1 += 0.5 * ax * rx;
   }
   return (r1);
}

//______________________________________________________________________________
Double_t TSpectrum2Fit::Dersy(Int_t numOfFittedPeaks, Double_t x,
                           const Double_t *parameter, Double_t sigmax) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates derivative of peaks shape function (see manual)    //
//   according to relative amplitude sy.                                        //
//      Function parameters:                                                    //
//              -numOfFittedPeaks-number of fitted peaks                        //
//              -x-position of channel                                          //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigmax-sigma of 1D ridge                                       //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Double_t p, r1 = 0, rx, ax, x0, s2;
   Int_t j;
   s2 = TMath::Sqrt(2.0);
   for (j = 0; j < numOfFittedPeaks; j++) {
      ax = parameter[7 * j + 4];
      x0 = parameter[7 * j + 6];
      p = (x - x0) / sigmax;
      s2 = TMath::Sqrt(2.0);
      rx = Erfc(p / s2);
      r1 += 0.5 * ax * rx;
   }
   return (r1);
}

//______________________________________________________________________________
Double_t TSpectrum2Fit::Derbx(Int_t numOfFittedPeaks, Double_t x, Double_t y,
                           const Double_t *parameter, Double_t sigmax,
                           Double_t sigmay, Double_t txy, Double_t tx, Double_t bx,
                           Double_t by) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates derivative of peaks shape function (see manual)    //
//   according to slope bx.                                                     //
//      Function parameters:                                                    //
//              -numOfFittedPeaks-number of fitted peaks                        //
//              -x,y-position of channel                                        //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigmax-sigmax of peaks                                         //
//              -sigmay-sigmay of peaks                                         //
//              -txy, tx-relative amplitudes                                    //
//              -bx, by-slopes                                                  //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Double_t p, r, r1 = 0, a, x0, y0, s2, px, py, erx, ery, ex, ey;
   Int_t j;
   s2 = TMath::Sqrt(2.0);
   for (j = 0; j < numOfFittedPeaks; j++) {
      a = parameter[7 * j];
      x0 = parameter[7 * j + 1];
      y0 = parameter[7 * j + 2];
      p = (x - x0) / sigmax;
      r = (y - y0) / sigmay;
      if (txy != 0) {
         px = 0, py = 0;
         erx =
             -Erfc(p / s2 + 1 / (2 * bx)) * p / (s2 * bx * bx) -
             Derfc(p / s2 + 1 / (2 * bx)) / (s2 * bx * bx), ery =
             Erfc(r / s2 + 1 / (2 * by));
         ex = p / (s2 * bx), ey = r / (s2 * by);
         if (TMath::Abs(ex) < 9 && TMath::Abs(ey) < 9) {
            px = exp(ex) * erx, py = exp(ey) * ery;
         }
         r1 += 0.5 * a * txy * px * py;
      }
      a = parameter[7 * j + 3];
      x0 = parameter[7 * j + 5];
      p = (x - x0) / sigmax;
      if (tx != 0) {
         px = 0;
         erx =
             (-Erfc(p / s2 + 1 / (2 * bx)) * p / (s2 * bx * bx) -
              Derfc(p / s2 + 1 / (2 * bx)) / (s2 * bx * bx));
         ex = p / (s2 * bx);
         if (TMath::Abs(ex) < 9)
            px = exp(ex) * erx;
         r1 += 0.5 * a * tx * px;
      }
   }
   return (r1);
}

//______________________________________________________________________________
Double_t TSpectrum2Fit::Derby(Int_t numOfFittedPeaks, Double_t x, Double_t y,
                           const Double_t *parameter, Double_t sigmax,
                           Double_t sigmay, Double_t txy, Double_t ty, Double_t bx,
                           Double_t by) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates derivative of peaks shape function (see manual)    //
//   according to slope by.                                                     //
//      Function parameters:                                                    //
//              -numOfFittedPeaks-number of fitted peaks                        //
//              -x,y-position of channel                                        //
//              -parameter-array of peaks parameters (amplitudes and positions) //
//              -sigmax-sigmax of peaks                                         //
//              -sigmay-sigmay of peaks                                         //
//              -txy, ty-relative amplitudes                                    //
//              -bx, by-slopes                                                  //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Double_t p, r, r1 = 0, a, x0, y0, s2, px, py, erx, ery, ex, ey;
   Int_t j;
   s2 = TMath::Sqrt(2.0);
   for (j = 0; j < numOfFittedPeaks; j++) {
      a = parameter[7 * j];
      x0 = parameter[7 * j + 1];
      y0 = parameter[7 * j + 2];
      p = (x - x0) / sigmax;
      r = (y - y0) / sigmay;
      if (txy != 0) {
         px = 0, py = 0;
         ery =
             -Erfc(r / s2 + 1 / (2 * by)) * r / (s2 * by * by) -
             Derfc(r / s2 + 1 / (2 * by)) / (s2 * by * by), erx =
             Erfc(p / s2 + 1 / (2 * bx));
         ex = p / (s2 * bx), ey = r / (s2 * by);
         if (TMath::Abs(ex) < 9 && TMath::Abs(ey) < 9) {
            px = exp(ex) * erx, py = exp(ey) * ery;
         }
         r1 += 0.5 * a * txy * px * py;
      }
      a = parameter[7 * j + 4];
      y0 = parameter[7 * j + 6];
      r = (y - y0) / sigmay;
      if (ty != 0) {
         py = 0;
         ery =
             (-Erfc(r / s2 + 1 / (2 * by)) * r / (s2 * by * by) -
              Derfc(r / s2 + 1 / (2 * by)) / (s2 * by * by));
         ey = r / (s2 * by);
         if (TMath::Abs(ey) < 9)
            py = exp(ey) * ery;
         r1 += 0.5 * a * ty * py;
      }
   }
   return (r1);
}

//______________________________________________________________________________
Double_t TSpectrum2Fit::Volume(Double_t a, Double_t sx, Double_t sy, Double_t ro) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates volume of a peak                                   //
//      Function parameters:                                                    //
//              -a-amplitude of the peak                                        //
//              -sx,sy-sigmas of peak                                           //
//              -ro-correlation coefficient                                     //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Double_t pi = 3.1415926535, r;
   r = 1 - ro * ro;
   if (r > 0)
      r = TMath::Sqrt(r);
   
   else {
      return (0);
   }
   r = 2 * a * pi * sx * sy * r;
   return (r);
}

//______________________________________________________________________________
Double_t TSpectrum2Fit::Derpa2(Double_t sx, Double_t sy, Double_t ro) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates derivative of the volume of a peak                 //
//   according to amplitute                                                     //
//      Function parameters:                                                    //
//              -sx,sy-sigmas of peak                                           //
//              -ro-correlation coefficient                                     //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Double_t pi = 3.1415926535, r;
   r = 1 - ro * ro;
   if (r > 0)
      r = TMath::Sqrt(r);
   
   else {
      return (0);
   }
   r = 2 * pi * sx * sy * r;
   return (r);
}

//______________________________________________________________________________
Double_t TSpectrum2Fit::Derpsigmax(Double_t a, Double_t sy, Double_t ro) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates derivative of the volume of a peak                 //
//   according to sigmax                                                        //
//      Function parameters:                                                    //
//              -a-amplitude of peak                                            //
//              -sy-sigma of peak                                               //
//              -ro-correlation coefficient                                     //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Double_t pi = 3.1415926535, r;
   r = 1 - ro * ro;
   if (r > 0)
      r = TMath::Sqrt(r);
   
   else {
      return (0);
   }
   r = a * 2 * pi * sy * r;
   return (r);
}

//______________________________________________________________________________
Double_t TSpectrum2Fit::Derpsigmay(Double_t a, Double_t sx, Double_t ro) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates derivative of the volume of a peak                 //
//   according to sigmay                                                        //
//      Function parameters:                                                    //
//              -a-amplitude of peak                                            //
//              -sx-sigma of peak                                               //
//              -ro-correlation coefficient                                     //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Double_t pi = 3.1415926535, r;
   r = 1 - ro * ro;
   if (r > 0)
      r = TMath::Sqrt(r);
   
   else {
      return (0);
   }
   r = a * 2 * pi * sx * r;
   return (r);
}

//______________________________________________________________________________
Double_t TSpectrum2Fit::Derpro(Double_t a, Double_t sx, Double_t sy, Double_t ro) 
{  
//////////////////////////////////////////////////////////////////////////////////
//   AUXILIARY FUNCTION                                                          //
//                                                                              //
//   This function calculates derivative of the volume of a peak                 //
//   according to ro                                                            //
//      Function parameters:                                                    //
//              -a-amplitude of peak                                            //
//              -sx,sy-sigmas of peak                                           //
//              -ro-correlation coefficient                                     //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
   Double_t pi = 3.1415926535, r;
   r = 1 - ro * ro;
   if (r > 0)
      r = TMath::Sqrt(r);
   
   else {
      return (0);
   }
   r = -a * 2 * pi * sx * sy * ro / r;
   return (r);
}


/////////////////END OF AUXILIARY FUNCTIONS USED BY FITTING FUNCTION fit2//////////////////////////
/////////////////FITTING FUNCTION WITHOUT MATRIX INVERSION///////////////////////////////////////
//______________________________________________________________________________
void TSpectrum2Fit::FitAwmi(Float_t **source) 
{
/////////////////////////////////////////////////////////////////////////////
// TWO-DIMENSIONAL FIT FUNCTION                
//  ALGORITHM WITHOUT MATRIX INVERSION 
//  This function fits the source spectrum. The calling program should
//  fill in input parameters of the TSpectrum2Fit class. 
//  The fitted parameters are written into           
//  TSpectrum2Fit class output parameters and fitted data are written into 
//  source spectrum.                                                    
//                                                                          
//        Function parameters:                                             
//        source-pointer to the matrix of source spectrum                  
//                                                                         
/////////////////////////////////////////////////////////////////////////////
//
//Begin_Html <!--
/* -->
<div class=Section2>

<p class=MsoNormal><b><span style='font-size:14.0pt'>Fitting</span></b></p>

<p class=MsoNormal style='text-align:justify'><i>&nbsp;</i></p>

<p class=MsoNormal style='text-align:justify'><i>Goal: to estimate
simultaneously peak shape parameters in spectra with large number of peaks</i></p>

<p class=MsoNormal>&nbsp;</p>

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
</span>we have implemented the nonsymmetrical semiempirical peak shape function</p>

<p class=MsoNormal style='margin-left:36.0pt;text-align:justify;text-indent:
-18.0pt'>•<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>it contains the two-dimensional symmetrical Gaussian two one-dimensional
symmetrical Gaussian ridges as well as nonsymmetrical terms and background.</p>

<p class=MsoNormal style='text-align:justify'><sub><img width=600 height=315
src="gif/spectrum2fit_awmi_image001.gif"></sub></p>

<p class=MsoNormal style='margin-left:37.05pt'>where Txy, Tx, Ty, Sxy, Sx, Sy
are relative amplitudes and Bx, By are slopes.</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal style='margin-left:36.0pt;text-indent:-18.0pt'>•<span
style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span>algorithm without matrix inversion (AWMI) allows fitting tens, hundreds
of peaks simultaneously that represent sometimes thousands of parameters [2],
[5]. </p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><i>Function:</i></p>

<p class=MsoNormal style='text-align:justify'>void <a
href="http://root.cern.ch/root/html/TSpectrum.html#TSpectrum:Fit1Awmi"><b>TSpectrumFit2::FitAwmi</b></a>(<a
href="http://root.cern.ch/root/html/ListOfTypes.html#float"><b>float</b></a> **fSource)</p>

<p class=MsoNormal style='text-align:justify'> </p>

<p class=MsoNormal style='text-align:justify'>This function fits the source
spectrum using AWMI algorithm. The calling program should fill in input
parameters of the TSpectrumFit2 class using a set of TSpectrumFit2 setters. The
fitted parameters are written into the class and fitted data are written into
source spectrum. </p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><i><span style='color:red'>Parameter:</span></i></p>

<p class=MsoNormal style='text-align:justify'>        <b>fSource</b>-pointer to
the matrix of source spectrum                  </p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><i><span style='font-size:10.0pt;color:red'>Member variables
of  TSpectrumFit2 class:</span></i></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Int_t    
fNPeaks;                        //number of peaks present in fit, input
parameter, it should be &gt; 0</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Int_t    
fNumberIterations;              //number of iterations in fitting procedure,
input parameter, it should be &gt; 0</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Int_t    
fXmin;                          //first fitted channel in x direction</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Int_t    
fXmax;                          //last fitted channel in x direction</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Int_t    
fYmin;                          //first fitted channel in y direction</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Int_t    
fYmax;                          //last fitted channel in y direction</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Int_t    
fStatisticType;                 //type of statistics, possible values
kFitOptimChiCounts (chi square statistics with counts as weighting
coefficients), kFitOptimChiFuncValues (chi square statistics with function
values as weighting coefficients),kFitOptimMaxLikelihood</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Int_t     fAlphaOptim;                   
//optimization of convergence algorithm, possible values kFitAlphaHalving,
kFitAlphaOptimal</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Int_t    
fPower;                         //possible values kFitPower2,4,6,8,10,12, for
details see references. It applies only for Awmi fitting function.</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Int_t    
fFitTaylor;                     //order of Taylor expansion, possible values
kFitTaylorOrderFirst, kFitTaylorOrderSecond. It applies only for Awmi fitting
function.</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fAlpha;                         //convergence coefficient, input parameter, it
should be positive number and &lt;=1, for details see references</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fChi;                           //here the fitting functions return resulting
chi square   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t
*fPositionInitX;                 //[fNPeaks] array of initial values of x
positions of 2D peaks, input parameters</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t
*fPositionCalcX;                 //[fNPeaks] array of calculated values of x
positions of 2D peaks, output parameters</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t
*fPositionErrX;                  //[fNPeaks] array of error values of x
positions of 2D peaks, output parameters</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t
*fPositionInitY;                 //[fNPeaks] array of initial values of y
positions of 2D peaks, input parameters</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t
*fPositionCalcY;                 //[fNPeaks] array of calculated values of y
positions of 2D peaks, output parameters</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t
*fPositionErrY;                  //[fNPeaks] array of error values of y
positions of 2D peaks, output parameters</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t
*fPositionInitX1;                //[fNPeaks] array of initial x positions of 1D
ridges, input parameters</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t
*fPositionCalcX1;                //[fNPeaks] array of calculated x positions of
1D ridges, output parameters</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t
*fPositionErrX1;                 //[fNPeaks] array of x positions errors of 1D
ridges, output parameters</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t
*fPositionInitY1;                //[fNPeaks] array of initial y positions of 1D
ridges, input parameters</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t
*fPositionCalcY1;                //[fNPeaks] array of calculated y positions of
1D ridges, output parameters</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t
*fPositionErrY1;                 //[fNPeaks] array of y positions errors of 1D
ridges, output parameters</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t
*fAmpInit;                       //[fNPeaks] array of initial values of
amplitudes of 2D peaks, input parameters</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t
*fAmpCalc;                       //[fNPeaks] array of calculated values of
amplitudes of 2D peaks, output parameters</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t
*fAmpErr;                        //[fNPeaks] array of amplitudes errors of 2D
peaks, output parameters</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t
*fAmpInitX1;                     //[fNPeaks] array of initial values of
amplitudes of 1D ridges in x direction, input parameters</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t
*fAmpCalcX1;                     //[fNPeaks] array of calculated values of
amplitudes of 1D ridges in x direction, output parameters</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t
*fAmpErrX1;                      //[fNPeaks] array of amplitudes errors of 1D
ridges in x direction, output parameters</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t
*fAmpInitY1;                     //[fNPeaks] array of initial values of
amplitudes of 1D ridges in y direction, input parameters</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t
*fAmpCalcY1;                     //[fNPeaks] array of calculated values of
amplitudes of 1D ridges in y direction, output parameters</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t
*fAmpErrY1;                      //[fNPeaks] array of amplitudes errors of 1D
ridges in y direction, output parameters</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t
*fVolume;                        //[fNPeaks] array of calculated volumes of 2D
peaks, output parameters</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t
*fVolumeErr;                     //[fNPeaks] array of volumes errors of 2D
peaks, output parameters</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fSigmaInitX;                    //initial value of sigma x parameter</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fSigmaCalcX;                    //calculated value of sigma x parameter</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fSigmaErrX;                     //error value of sigma x parameter</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fSigmaInitY;                    //initial value of sigma y parameter</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fSigmaCalcY;                    //calculated value of sigma y parameter</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fSigmaErrY;                     //error value of sigma y parameter</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fRoInit;                        //initial value of correlation coefficient</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fRoCalc;                        //calculated value of correlation coefficient</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fRoErr;                         //error value of correlation coefficient</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fTxyInit;                       //initial value of t parameter for 2D peaks
(relative amplitude of tail), for details see html manual and references</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fTxyCalc;                       //calculated value of t parameter for 2D peaks</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fTxyErr;                        //error value of t parameter for 2D peaks</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fSxyInit;                       //initial value of s parameter for 2D peaks
(relative amplitude of step), for details see html manual and references</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fSxyCalc;                       //calculated value of s parameter for 2D peaks</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fSxyErr;                        //error value of s parameter for 2D peaks</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fTxInit;                        //initial value of t parameter for 1D ridges in
x direction (relative amplitude of tail), for details see html manual and
references</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fTxCalc;                        //calculated value of t parameter for 1D ridges
in x direction</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fTxErr;                         //error value of t parameter for 1D ridges in x
direction</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fTyInit;                        //initial value of t parameter for 1D ridges in
y direction (relative amplitude of tail), for details see html manual and
references</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t  fTyCalc;                       
//calculated value of t parameter for 1D ridges in y direction</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fTyErr;                         //error value of t parameter for 1D ridges in y
direction</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fSxInit;                        //initial value of s parameter for 1D ridges in
x direction (relative amplitude of step), for details see html manual and
references</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fSxCalc;                        //calculated value of s parameter for 1D ridges
in x direction</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t  fSxErr;                         //error
value of s parameter for 1D ridges in x direction</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fSyInit;                        //initial value of s parameter for 1D ridges in
y direction (relative amplitude of step), for details see html manual and
references</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fSyCalc;                        //calculated value of s parameter for 1D ridges
in y direction</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fSyErr;                         //error value of s parameter for 1D ridges in y
direction</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fBxInit;                        //initial value of b parameter for 1D ridges in
x direction (slope), for details see html manual and references</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fBxCalc;                        //calculated value of b parameter for 1D ridges
in x direction</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fBxErr;                         //error value of b parameter for 1D ridges in x
direction</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fByInit;                        //initial value of b parameter for 1D ridges in
y direction (slope), for details see html manual and references</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fByCalc;                        //calculated value of b parameter for 1D ridges
in y direction</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fByErr;                         //error value of b parameter for 1D ridges in y
direction</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fA0Init;                        //initial value of background a0 parameter(backgroud
is estimated as a0+ax*x+ay*y)</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fA0Calc;                        //calculated value of background a0 parameter</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fA0Err;                         //error value of background a0 parameter</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t  fAxInit;        
               //initial value of background ax parameter(backgroud is
estimated as a0+ax*x+ay*y)</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fAxCalc;                        //calculated value of background ax parameter</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fAxErr;                         //error value of background ax parameter</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fAyInit;                        //initial value of background ay
parameter(backgroud is estimated as a0+ax*x+ay*y)</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t 
fAyCalc;                        //calculated value of background ay parameter</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Double_t  fAyErr;                        
//error value of background ay parameter   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Bool_t  
*fFixPositionX;                  //[fNPeaks] array of logical values which
allow to fix appropriate x positions of 2D peaks (not fit). However they are
present in the estimated functional</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Bool_t  
*fFixPositionY;                  //[fNPeaks] array of logical values which
allow to fix appropriate y positions of 2D peaks (not fit). However they are
present in the estimated functional</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Bool_t  
*fFixPositionX1;                 //[fNPeaks] array of logical values which
allow to fix appropriate x positions of 1D ridges (not fit). However they are
present in the estimated functional</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Bool_t  
*fFixPositionY1;                 //[fNPeaks] array of logical values which
allow to fix appropriate y positions of 1D ridges (not fit). However they are
present in the estimated functional</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Bool_t  
*fFixAmp;                        //[fNPeaks] array of logical values which
allow to fix appropriate amplitudes of 2D peaks (not fit). However they are
present in the estimated functional</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Bool_t  
*fFixAmpX1;                      //[fNPeaks] array of logical values which
allow to fix appropriate amplitudes of 1D ridges in x direction (not fit).
However they are present in the estimated functional</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Bool_t  
*fFixAmpY1;                      //[fNPeaks] array of logical values which
allow to fix appropriate amplitudes of 1D ridges in y direction (not fit).
However they are present in the estimated functional</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Bool_t   
fFixSigmaX;                     //logical value of sigma x parameter, which
allows to fix the parameter (not to fit).</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Bool_t   
fFixSigmaY;                     //logical value of sigma y parameter, which
allows to fix the parameter (not to fit).</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Bool_t    fFixRo;                         //logical
value of correlation coefficient, which allows to fix the parameter (not to
fit).</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Bool_t   
fFixTxy;                        //logical value of t parameter for 2D peaks,
which allows to fix the parameter (not to fit).</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Bool_t   
fFixSxy;                        //logical value of s parameter for 2D peaks,
which allows to fix the parameter (not to fit).</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Bool_t   
fFixTx;                         //logical value of t parameter for 1D ridges in
x direction, which allows to fix the parameter (not to fit).</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Bool_t   
fFixTy;                         //logical value of t parameter for 1D ridges in
y direction, which allows to fix the parameter (not to fit).</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Bool_t   
fFixSx;                         //logical value of s parameter for 1D ridges in
x direction, which allows to fix the parameter (not to fit).</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Bool_t   
fFixSy;                         //logical value of s parameter for 1D ridges in
y direction, which allows to fix the parameter (not to fit).</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Bool_t    fFixBx;                         //logical
value of b parameter for 1D ridges in x direction, which allows to fix the
parameter (not to fit).</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Bool_t   
fFixBy;                         //logical value of b parameter for 1D ridges in
y direction, which allows to fix the parameter (not to fit).</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Bool_t   
fFixA0;                         //logical value of a0 parameter, which allows
to fix the parameter (not to fit).</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Bool_t   
fFixAx;                         //logical value of ax parameter, which allows
to fix the parameter (not to fit).</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Bool_t   
fFixAy;                         //logical value of ay parameter, which allows
to fix the parameter (not to fit).</span></p>

<p class=MsoNormal> </p>

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

<p class=MsoNormal style='text-align:justify'><i>Example  – script FitAwmi2.c:</i></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:18.0pt'><img
border=0 width=602 height=455 src="gif/spectrum2fit_awmi_image002.jpg"></span></p>

<p class=MsoNormal style='text-align:justify'><b>Fig. 1 Original two-dimensional
spectrum with found peaks (using TSpectrum2 peak searching function). The
positions of peaks were used as initial estimates in fitting procedure.</b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'><img
border=0 width=602 height=455 src="gif/spectrum2fit_awmi_image003.jpg"></span></b></p>

<p class=MsoBodyText2 style='text-align:justify'>Fig. 2 Fitted (generated from
fitted parameters) spectrum of the data from Fig. 1 using Algorithm Without
Matrix Inversion. Each peak was represented by 7 parameters, which together
with Sigmax, Sigmay and a0 resulted in 38 fitted parameters. The chi-square
after 1000 iterations was 0.642342.</p>

<p class=MsoNormal><b><span style='color:#339966'>&nbsp;</span></b></p>

<p class=MsoNormal><b><span style='color:#339966'>Script:</span></b></p>

<p class=MsoNormal>&nbsp;</p>

<p class=MsoNormal><span style='font-size:10.0pt'>// Example to illustrate fitting
function, algorithm without matrix inversion (AWMI) (class TSpectrumFit2).</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// To execute this example,
do</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// root &gt; .x FitAwmi2.C</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>&nbsp;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>void FitAwmi2() {</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Int_t i, j, nfound;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Int_t nbinsx = 64;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Int_t nbinsy = 64;   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Int_t xmin  = 0;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Int_t xmax  = nbinsx;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Int_t ymin  = 0;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Int_t ymax  = nbinsy;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Float_t ** source = new
float *[nbinsx];   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Float_t ** dest = new
float *[nbinsx];      </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for (i=0;i&lt;nbinsx;i++)</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                                                source[i]=new
float[nbinsy];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for (i=0;i&lt;nbinsx;i++)</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                                                dest[i]=new
float[nbinsy];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TH2F *search = new
TH2F(&quot;search&quot;,&quot;High resolution peak
searching&quot;,nbinsx,xmin,xmax,nbinsy,ymin,ymax);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TFile *f = new
TFile(&quot;TSpectrum2.root&quot;);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   search=(TH2F*)
f-&gt;Get(&quot;search4;1&quot;);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TCanvas *Searching = new
TCanvas(&quot;Searching&quot;,&quot;Two-dimensional fitting using Algorithm
Without Matrix Inversion&quot;,10,10,1000,700);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TSpectrum2 *s = new
TSpectrum2();</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for (i = 0; i &lt; nbinsx;
i++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     for (j = 0; j &lt;
nbinsy; j++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                    source[i][j]
= search-&gt;GetBinContent(i + 1,j + 1); </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                 }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   } </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   //searching for candidate
peaks positions     </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   nfound =
s-&gt;SearchHighRes(source, dest, nbinsx, nbinsy, 2, 5, kTRUE, 3, kFALSE, 3);  
</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Bool_t *FixPosX = new
Bool_t[nfound];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Bool_t *FixPosY = new
Bool_t[nfound];   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Bool_t *FixAmp = new
Bool_t[nfound];      </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Float_t *PosX = new
Float_t[nfound];         </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Float_t *PosY = new
Float_t[nfound];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Float_t *Amp = new
Float_t[nfound];      </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Float_t *AmpXY = new
Float_t[nfound];         </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   PosX =
s-&gt;GetPositionX();</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   PosY =
s-&gt;GetPositionY();      </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   printf(&quot;Found %d
candidate peaks\n&quot;,nfound);   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for(i = 0; i&lt; nfound ;
i++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>      FixPosX[i] = kFALSE;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>      FixPosY[i] =
kFALSE;      </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>      FixAmp[i] = kFALSE;    </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>      Amp[i] =
source[(int)(PosX[i]+0.5)][(int)(PosY[i]+0.5)];      //initial values of peaks
amplitudes, input parameters          </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>      AmpXY[i] = 0;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   //filling in the initial
estimates of the input parameters</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TSpectrumFit2 *pfit=new
TSpectrumFit2(nfound);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
pfit-&gt;SetFitParameters(xmin, xmax-1, ymin, ymax-1, 1000, 0.1, pfit-&gt;kFitOptimChiCounts,
pfit-&gt;kFitAlphaHalving, pfit-&gt;kFitPower2,
pfit-&gt;kFitTaylorOrderFirst);   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
pfit-&gt;SetPeakParameters(2, kFALSE, 2, kFALSE, 0, kTRUE, PosX, (Bool_t *)
FixPosX, PosY, (Bool_t *) FixPosY, PosX, (Bool_t *) FixPosX, PosY, (Bool_t *)
FixPosY, Amp, (Bool_t *) FixAmp, AmpXY, (Bool_t *) FixAmp, AmpXY, (Bool_t *)
FixAmp);      </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
pfit-&gt;SetBackgroundParameters(0, kFALSE, 0, kTRUE, 0, kTRUE);   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   pfit-&gt;FitAwmi(source);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>    for (i = 0; i &lt;
nbinsx; i++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     for (j = 0; j &lt;
nbinsy; j++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                  search-&gt;SetBinContent(i
+ 1, j + 1,source[i][j]);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                 }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
search-&gt;Draw(&quot;SURF&quot;);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>}</span></p>

<p class=MsoNormal style='text-align:justify'><i><span style='font-size:16.0pt'>&nbsp;</span></i></p>

<p class=MsoNormal style='text-align:justify'><i>Example 2 – script FitA2.c:</i></p>

<p class=MsoNormal><img border=0 width=602 height=455
src="gif/spectrum2fit_awmi_image004.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b>Fig. 3 Original
two-dimensional gamma-gamma-ray spectrum with found peaks (using TSpectrum2
peak searching function). </b></p>

<p class=MsoNormal style='text-align:justify'><img border=0 width=602
height=455 src="gif/spectrum2fit_awmi_image005.jpg"></p>

<p class=MsoNormal style='text-align:justify'><b>Fig. 4 Fitted (generated from
fitted parameters) spectrum of the data from Fig. 3 using Algorithm Without
Matrix Inversion. 152 peaks were identified. </b><b>Each peak was represented
by 7 parameters, which together with Sigmax, Sigmay and a0 resulted in 1067
fitted parameters. The chi-square after 1000 iterations was 0.728675. One can
observe good correspondence with the original data.</b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'>&nbsp;</span></b></p>

<p class=MsoNormal><b><span style='color:#339966'>Script:</span></b></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>//
Example to illustrate fitting function, algorithm without matrix inversion
(AWMI) (class TSpectrumFit2).</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>//
To execute this example, do</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>//
root &gt; .x FitA2.C</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>void
FitA2() {</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Int_t i, j, nfound;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Int_t nbinsx = 256;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Int_t nbinsy = 256;   </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Int_t xmin  = 0;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Int_t xmax  = nbinsx;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Int_t ymin  = 0;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Int_t ymax  = nbinsy;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Float_t ** source = new float *[nbinsx];   </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Float_t ** dest = new float *[nbinsx];      </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
for (i=0;i&lt;nbinsx;i++)</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>                                                source[i]=new
float[nbinsy];</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
for (i=0;i&lt;nbinsx;i++)</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>                                                dest[i]=new
float[nbinsy];   </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
TH2F *search = new TH2F(&quot;search&quot;,&quot;High resolution peak
searching&quot;,nbinsx,xmin,xmax,nbinsy,ymin,ymax);</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
TFile *f = new TFile(&quot;TSpectrum2.root&quot;);</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
search=(TH2F*) f-&gt;Get(&quot;fit1;1&quot;);</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
TCanvas *Searching = new TCanvas(&quot;Searching&quot;,&quot;Two-dimensional
fitting using Algorithm Without Matrix Inversion&quot;,10,10,1000,700);</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
TSpectrum2 *s = new TSpectrum2(1000,1);</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
for (i = 0; i &lt; nbinsx; i++){</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>    
for (j = 0; j &lt; nbinsy; j++){</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
                 source[i][j] = search-&gt;GetBinContent(i + 1,j + 1); </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
              }</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
}   </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
nfound = s-&gt;SearchHighRes(source, dest, nbinsx, nbinsy, 2, 2, kTRUE, 100,
kFALSE, 3);   </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
printf(&quot;Found %d candidate peaks\n&quot;,nfound);</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Bool_t *FixPosX = new Bool_t[nfound];</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Bool_t *FixPosY = new Bool_t[nfound];   </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Bool_t *FixAmp = new Bool_t[nfound];      </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Float_t *PosX = new Float_t[nfound];         </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Float_t *PosY = new Float_t[nfound];</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Float_t *Amp = new Float_t[nfound];      </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
Float_t *AmpXY = new Float_t[nfound];         </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
PosX = s-&gt;GetPositionX();</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
PosY = s-&gt;GetPositionY();      </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
for(i = 0; i&lt; nfound ; i++){</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>     
FixPosX[i] = kFALSE;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>     
FixPosY[i] = kFALSE;      </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>     
FixAmp[i] = kFALSE;    </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>     
Amp[i] = source[(int)(PosX[i]+0.5)][(int)(PosY[i]+0.5)];      //initial values
of peaks amplitudes, input parameters          </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>     
AmpXY[i] = 0;</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
}</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
//filling in the initial estimates of the input parameters</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
TSpectrumFit2 *pfit=new TSpectrumFit2(nfound);</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
pfit-&gt;SetFitParameters(xmin, xmax-1, ymin, ymax-1, 1000, 0.1,
pfit-&gt;kFitOptimChiCounts, pfit-&gt;kFitAlphaHalving, pfit-&gt;kFitPower2,
pfit-&gt;kFitTaylorOrderFirst);   </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
pfit-&gt;SetPeakParameters(2, kFALSE, 2, kFALSE, 0, kTRUE, PosX, (Bool_t *)
FixPosX, PosY, (Bool_t *) FixPosY, PosX, (Bool_t *) FixPosX, PosY, (Bool_t *)
FixPosY, Amp, (Bool_t *) FixAmp, AmpXY, (Bool_t *) FixAmp, AmpXY, (Bool_t *)
FixAmp);      </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
pfit-&gt;SetBackgroundParameters(0, kFALSE, 0, kTRUE, 0, kTRUE);   </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
pfit-&gt;FitAwmi(source);</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
for (i = 0; i &lt; nbinsx; i++){</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>    
for (j = 0; j &lt; nbinsy; j++){</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>    
             search-&gt;SetBinContent(i + 1, j + 1,source[i][j]);</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
              }</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
}   </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>  
search-&gt;Draw(&quot;SURF&quot;);   </span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>}</span></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:10.0pt'>&nbsp;</span></p>

</div>

<!-- */
// --> End_Html

   Int_t i, i1, i2, j, k, shift =
       7 * fNPeaks + 14, peak_vel, size, iter, pw,
       regul_cycle, flag;
   Double_t a, b, c, d = 0, alpha, chi_opt, yw, ywm, f, chi2, chi_min, chi =
       0, pi, pmin = 0, chi_cel = 0, chi_er;
   Double_t *working_space = new Double_t[5 * (7 * fNPeaks + 14)];
   for (i = 0, j = 0; i < fNPeaks; i++) {
      working_space[7 * i] = fAmpInit[i]; //vector parameter
      if (fFixAmp[i] == false) {
         working_space[shift + j] = fAmpInit[i]; //vector xk
         j += 1;
      }
      working_space[7 * i + 1] = fPositionInitX[i]; //vector parameter
      if (fFixPositionX[i] == false) {
         working_space[shift + j] = fPositionInitX[i]; //vector xk
         j += 1;
      }
      working_space[7 * i + 2] = fPositionInitY[i]; //vector parameter
      if (fFixPositionY[i] == false) {
         working_space[shift + j] = fPositionInitY[i]; //vector xk
         j += 1;
      }
      working_space[7 * i + 3] = fAmpInitX1[i]; //vector parameter
      if (fFixAmpX1[i] == false) {
         working_space[shift + j] = fAmpInitX1[i]; //vector xk
         j += 1;
      }
      working_space[7 * i + 4] = fAmpInitY1[i]; //vector parameter
      if (fFixAmpY1[i] == false) {
         working_space[shift + j] = fAmpInitY1[i]; //vector xk
         j += 1;
      }
      working_space[7 * i + 5] = fPositionInitX1[i]; //vector parameter
      if (fFixPositionX1[i] == false) {
         working_space[shift + j] = fPositionInitX1[i]; //vector xk
         j += 1;
      }
      working_space[7 * i + 6] = fPositionInitY1[i]; //vector parameter
      if (fFixPositionY1[i] == false) {
         working_space[shift + j] = fPositionInitY1[i]; //vector xk
         j += 1;
      }
   }
   peak_vel = 7 * i;
   working_space[7 * i] = fSigmaInitX; //vector parameter
   if (fFixSigmaX == false) {
      working_space[shift + j] = fSigmaInitX; //vector xk
      j += 1;
   }
   working_space[7 * i + 1] = fSigmaInitY; //vector parameter
   if (fFixSigmaY == false) {
      working_space[shift + j] = fSigmaInitY; //vector xk
      j += 1;
   }
   working_space[7 * i + 2] = fRoInit; //vector parameter
   if (fFixRo == false) {
      working_space[shift + j] = fRoInit; //vector xk
      j += 1;
   }
   working_space[7 * i + 3] = fA0Init; //vector parameter
   if (fFixA0 == false) {
      working_space[shift + j] = fA0Init; //vector xk
      j += 1;
   }
   working_space[7 * i + 4] = fAxInit; //vector parameter
   if (fFixAx == false) {
      working_space[shift + j] = fAxInit; //vector xk
      j += 1;
   }
   working_space[7 * i + 5] = fAyInit; //vector parameter
   if (fFixAy == false) {
      working_space[shift + j] = fAyInit; //vector xk
      j += 1;
   }
   working_space[7 * i + 6] = fTxyInit; //vector parameter
   if (fFixTxy == false) {
      working_space[shift + j] = fTxyInit; //vector xk
      j += 1;
   }
   working_space[7 * i + 7] = fSxyInit; //vector parameter
   if (fFixSxy == false) {
      working_space[shift + j] = fSxyInit; //vector xk
      j += 1;
   }
   working_space[7 * i + 8] = fTxInit; //vector parameter
   if (fFixTx == false) {
      working_space[shift + j] = fTxInit; //vector xk
      j += 1;
   }
   working_space[7 * i + 9] = fTyInit; //vector parameter
   if (fFixTy == false) {
      working_space[shift + j] = fTyInit; //vector xk
      j += 1;
   }
   working_space[7 * i + 10] = fSxyInit; //vector parameter
   if (fFixSx == false) {
      working_space[shift + j] = fSxInit; //vector xk
      j += 1;
   }
   working_space[7 * i + 11] = fSyInit; //vector parameter
   if (fFixSy == false) {
      working_space[shift + j] = fSyInit; //vector xk
      j += 1;
   }
   working_space[7 * i + 12] = fBxInit; //vector parameter
   if (fFixBx == false) {
      working_space[shift + j] = fBxInit; //vector xk
      j += 1;
   }
   working_space[7 * i + 13] = fByInit; //vector parameter
   if (fFixBy == false) {
      working_space[shift + j] = fByInit; //vector xk
      j += 1;
   }
   size = j;
   for (iter = 0; iter < fNumberIterations; iter++) {
      for (j = 0; j < size; j++) {
         working_space[2 * shift + j] = 0, working_space[3 * shift + j] = 0; //der,temp
      }
      
          //filling vectors
      alpha = fAlpha;
      chi_opt = 0, pw = fPower - 2;
      for (i1 = fXmin; i1 <= fXmax; i1++) {
         for (i2 = fYmin; i2 <= fYmax; i2++) {
            yw = source[i1][i2];
            ywm = yw;
            f = Shape2(fNPeaks, (Double_t) i1, (Double_t) i2,
                        working_space, working_space[peak_vel],
                        working_space[peak_vel + 1],
                        working_space[peak_vel + 2],
                        working_space[peak_vel + 3],
                        working_space[peak_vel + 4],
                        working_space[peak_vel + 5],
                        working_space[peak_vel + 6],
                        working_space[peak_vel + 7],
                        working_space[peak_vel + 8],
                        working_space[peak_vel + 9],
                        working_space[peak_vel + 10],
                        working_space[peak_vel + 11],
                        working_space[peak_vel + 12],
                        working_space[peak_vel + 13]);
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
            
                //calculation of gradient vector
                for (j = 0, k = 0; j < fNPeaks; j++) {
               if (fFixAmp[j] == false) {
                  a = Deramp2((Double_t) i1, (Double_t) i2,
                               working_space[7 * j + 1],
                               working_space[7 * j + 2],
                               working_space[peak_vel],
                               working_space[peak_vel + 1],
                               working_space[peak_vel + 2],
                               working_space[peak_vel + 6],
                               working_space[peak_vel + 7],
                               working_space[peak_vel + 12],
                               working_space[peak_vel + 13]);
                  if (ywm != 0) {
                     c = Ourpowl(a, pw);
                     if (fStatisticType == kFitOptimChiFuncValues) {
                        b = a * (yw * yw - f * f) / (ywm * ywm);
                        working_space[2 * shift + k] += b * c; //der
                        b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                        working_space[3 * shift + k] += b * c; //temp
                     }
                     
                     else {
                        b = a * (yw - f) / ywm;
                        working_space[2 * shift + k] += b * c; //der
                        b = a * a / ywm;
                        working_space[3 * shift + k] += b * c; //temp
                     }
                  }
                  k += 1;
               }
               if (fFixPositionX[j] == false) {
                  a = Deri02((Double_t) i1, (Double_t) i2,
                              working_space[7 * j],
                              working_space[7 * j + 1],
                              working_space[7 * j + 2],
                              working_space[peak_vel],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 2],
                              working_space[peak_vel + 6],
                              working_space[peak_vel + 7],
                              working_space[peak_vel + 12],
                              working_space[peak_vel + 13]);
                  if (fFitTaylor == kFitTaylorOrderSecond)
                     d = Derderi02((Double_t) i1, (Double_t) i2,
                                    working_space[7 * j],
                                    working_space[7 * j + 1],
                                    working_space[7 * j + 2],
                                    working_space[peak_vel],
                                    working_space[peak_vel + 1],
                                    working_space[peak_vel + 2]);
                  if (ywm != 0) {
                     c = Ourpowl(a, pw);
                     if (TMath::Abs(a) > 0.00000001
                          && fFitTaylor == kFitTaylorOrderSecond) {
                        d = d * TMath::Abs(yw - f) / (2 * a * ywm);
                        if (((a + d) <= 0 && a >= 0) || ((a + d) >= 0
                             && a <= 0))
                           d = 0;
                     }
                     
                     else
                        d = 0;
                     a = a + d;
                     if (fStatisticType == kFitOptimChiFuncValues) {
                        b = a * (yw * yw - f * f) / (ywm * ywm);
                        working_space[2 * shift + k] += b * c; //der
                        b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                        working_space[3 * shift + k] += b * c; //temp
                     }
                     
                     else {
                        b = a * (yw - f) / ywm;
                        working_space[2 * shift + k] += b * c; //der
                        b = a * a / ywm;
                        working_space[3 * shift + k] += b * c; //temp
                     }
                  }
                  k += 1;
               }
               if (fFixPositionY[j] == false) {
                  a = Derj02((Double_t) i1, (Double_t) i2,
                              working_space[7 * j],
                              working_space[7 * j + 1],
                              working_space[7 * j + 2],
                              working_space[peak_vel],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 2],
                              working_space[peak_vel + 6],
                              working_space[peak_vel + 7],
                              working_space[peak_vel + 12],
                              working_space[peak_vel + 13]);
                  if (fFitTaylor == kFitTaylorOrderSecond)
                     d = Derderj02((Double_t) i1, (Double_t) i2,
                                    working_space[7 * j],
                                    working_space[7 * j + 1],
                                    working_space[7 * j + 2],
                                    working_space[peak_vel],
                                    working_space[peak_vel + 1],
                                    working_space[peak_vel + 2]);
                  if (ywm != 0) {
                     c = Ourpowl(a, pw);
                     if (TMath::Abs(a) > 0.00000001
                          && fFitTaylor == kFitTaylorOrderSecond) {
                        d = d * TMath::Abs(yw - f) / (2 * a * ywm);
                        if (((a + d) <= 0 && a >= 0) || ((a + d) >= 0
                             && a <= 0))
                           d = 0;
                     }
                     
                     else
                        d = 0;
                     a = a + d;
                     if (fStatisticType == kFitOptimChiFuncValues) {
                        b = a * (yw * yw - f * f) / (ywm * ywm);
                        working_space[2 * shift + k] += b * c; //der
                        b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                        working_space[3 * shift + k] += b * c; //temp
                     }
                     
                     else {
                        b = a * (yw - f) / ywm;
                        working_space[2 * shift + k] += b * c; //der
                        b = a * a / ywm;
                        working_space[3 * shift + k] += b * c; //temp
                     }
                  }
                  k += 1;
               }
               if (fFixAmpX1[j] == false) {
                  a = Derampx((Double_t) i1, working_space[7 * j + 5],
                               working_space[peak_vel],
                               working_space[peak_vel + 8],
                               working_space[peak_vel + 10],
                               working_space[peak_vel + 12]);
                  if (ywm != 0) {
                     c = Ourpowl(a, pw);
                     if (fStatisticType == kFitOptimChiFuncValues) {
                        b = a * (yw * yw - f * f) / (ywm * ywm);
                        working_space[2 * shift + k] += b * c; //der
                        b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                        working_space[3 * shift + k] += b * c; //temp
                     }
                     
                     else {
                        b = a * (yw - f) / ywm;
                        working_space[2 * shift + k] += b * c; //der
                        b = a * a / ywm;
                        working_space[3 * shift + k] += b * c; //temp
                     }
                  }
                  k += 1;
               }
               if (fFixAmpY1[j] == false) {
                  a = Derampx((Double_t) i2, working_space[7 * j + 6],
                               working_space[peak_vel + 1],
                               working_space[peak_vel + 9],
                               working_space[peak_vel + 11],
                               working_space[peak_vel + 13]);
                  if (ywm != 0) {
                     c = Ourpowl(a, pw);
                     if (fStatisticType == kFitOptimChiFuncValues) {
                        b = a * (yw * yw - f * f) / (ywm * ywm);
                        working_space[2 * shift + k] += b * c; //der
                        b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                        working_space[3 * shift + k] += b * c; //temp
                     }
                     
                     else {
                        b = a * (yw - f) / ywm;
                        working_space[2 * shift + k] += b * c; //der
                        b = a * a / ywm;
                        working_space[3 * shift + k] += b * c; //temp
                     }
                  }
                  k += 1;
               }
               if (fFixPositionX1[j] == false) {
                  a = Deri01((Double_t) i1, working_space[7 * j + 3],
                              working_space[7 * j + 5],
                              working_space[peak_vel],
                              working_space[peak_vel + 8],
                              working_space[peak_vel + 10],
                              working_space[peak_vel + 12]);
                  if (fFitTaylor == kFitTaylorOrderSecond)
                     d = Derderi01((Double_t) i1, working_space[7 * j + 3],
                                    working_space[7 * j + 5],
                                    working_space[peak_vel]);
                  if (ywm != 0) {
                     c = Ourpowl(a, pw);
                     if (TMath::Abs(a) > 0.00000001
                          && fFitTaylor == kFitTaylorOrderSecond) {
                        d = d * TMath::Abs(yw - f) / (2 * a * ywm);
                        if (((a + d) <= 0 && a >= 0) || ((a + d) >= 0
                             && a <= 0))
                           d = 0;
                     }
                     
                     else
                        d = 0;
                     a = a + d;
                     if (fStatisticType == kFitOptimChiFuncValues) {
                        b = a * (yw * yw - f * f) / (ywm * ywm);
                        working_space[2 * shift + k] += b * c; //Der
                        b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                        working_space[3 * shift + k] += b * c; //temp
                     }
                     
                     else {
                        b = a * (yw - f) / ywm;
                        working_space[2 * shift + k] += b * c; //Der
                        b = a * a / ywm;
                        working_space[3 * shift + k] += b * c; //temp
                     }
                  }
                  k += 1;
               }
               if (fFixPositionY1[j] == false) {
                  a = Deri01((Double_t) i2, working_space[7 * j + 4],
                              working_space[7 * j + 6],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 9],
                              working_space[peak_vel + 11],
                              working_space[peak_vel + 13]);
                  if (fFitTaylor == kFitTaylorOrderSecond)
                     d = Derderi01((Double_t) i2, working_space[7 * j + 4],
                                    working_space[7 * j + 6],
                                    working_space[peak_vel + 1]);
                  if (ywm != 0) {
                     c = Ourpowl(a, pw);
                     if (TMath::Abs(a) > 0.00000001
                          && fFitTaylor == kFitTaylorOrderSecond) {
                        d = d * TMath::Abs(yw - f) / (2 * a * ywm);
                        if (((a + d) <= 0 && a >= 0) || ((a + d) >= 0
                             && a <= 0))
                           d = 0;
                     }
                     
                     else
                        d = 0;
                     a = a + d;
                     if (fStatisticType == kFitOptimChiFuncValues) {
                        b = a * (yw * yw - f * f) / (ywm * ywm);
                        working_space[2 * shift + k] += b * c; //der
                        b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                        working_space[3 * shift + k] += b * c; //temp
                     }
                     
                     else {
                        b = a * (yw - f) / ywm;
                        working_space[2 * shift + k] += b * c; //der
                        b = a * a / ywm;
                        working_space[3 * shift + k] += b * c; //temp
                     }
                  }
                  k += 1;
               }
            }
            if (fFixSigmaX == false) {
               a = Dersigmax(fNPeaks, (Double_t) i1, (Double_t) i2,
                              working_space, working_space[peak_vel],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 2],
                              working_space[peak_vel + 6],
                              working_space[peak_vel + 7],
                              working_space[peak_vel + 8],
                              working_space[peak_vel + 10],
                              working_space[peak_vel + 12],
                              working_space[peak_vel + 13]);
               if (fFitTaylor == kFitTaylorOrderSecond)
                  d = Derdersigmax(fNPeaks, (Double_t) i1,
                                    (Double_t) i2, working_space,
                                    working_space[peak_vel],
                                    working_space[peak_vel + 1],
                                    working_space[peak_vel + 2]);
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
                     working_space[2 * shift + k] += b * c; //der
                     b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                     working_space[3 * shift + k] += b * c; //temp
                  }
                  
                  else {
                     b = a * (yw - f) / ywm;
                     working_space[2 * shift + k] += b * c; //der
                     b = a * a / ywm;
                     working_space[3 * shift + k] += b * c; //temp
                  }
               }
               k += 1;
            }
            if (fFixSigmaY == false) {
               a = Dersigmay(fNPeaks, (Double_t) i1, (Double_t) i2,
                              working_space, working_space[peak_vel],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 2],
                              working_space[peak_vel + 6],
                              working_space[peak_vel + 7],
                              working_space[peak_vel + 9],
                              working_space[peak_vel + 11],
                              working_space[peak_vel + 12],
                              working_space[peak_vel + 13]);
               if (fFitTaylor == kFitTaylorOrderSecond)
                  d = Derdersigmay(fNPeaks, (Double_t) i1,
                                    (Double_t) i2, working_space,
                                    working_space[peak_vel],
                                    working_space[peak_vel + 1],
                                    working_space[peak_vel + 2]);
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
                     working_space[2 * shift + k] += b * c; //der
                     b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                     working_space[3 * shift + k] += b * c; //temp
                  }
                  
                  else {
                     b = a * (yw - f) / ywm;
                     working_space[2 * shift + k] += b * c; //der
                     b = a * a / ywm;
                     working_space[3 * shift + k] += b * c; //temp
                  }
               }
               k += 1;
            }
            if (fFixRo == false) {
               a = Derro(fNPeaks, (Double_t) i1, (Double_t) i2,
                          working_space, working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 2]);
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
                     working_space[2 * shift + k] += b * c; //der
                     b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                     working_space[3 * shift + k] += b * c; //temp
                  }
                  
                  else {
                     b = a * (yw - f) / ywm;
                     working_space[2 * shift + k] += b * c; //der
                     b = a * a / ywm;
                     working_space[3 * shift + k] += b * c; //temp
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
                     working_space[2 * shift + k] += b * c; //der
                     b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                     working_space[3 * shift + k] += b * c; //temp
                  }
                  
                  else {
                     b = a * (yw - f) / ywm;
                     working_space[2 * shift + k] += b * c; //der
                     b = a * a / ywm;
                     working_space[3 * shift + k] += b * c; //temp
                  }
               }
               k += 1;
            }
            if (fFixAx == false) {
               a = i1;
               if (ywm != 0) {
                  c = Ourpowl(a, pw);
                  if (fStatisticType == kFitOptimChiFuncValues) {
                     b = a * (yw * yw - f * f) / (ywm * ywm);
                     working_space[2 * shift + k] += b * c; //der
                     b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                     working_space[3 * shift + k] += b * c; //temp
                  }
                  
                  else {
                     b = a * (yw - f) / ywm;
                     working_space[2 * shift + k] += b * c; //der
                     b = a * a / ywm;
                     working_space[3 * shift + k] += b * c; //temp
                  }
               }
               k += 1;
            }
            if (fFixAy == false) {
               a = i2;
               if (ywm != 0) {
                  c = Ourpowl(a, pw);
                  if (fStatisticType == kFitOptimChiFuncValues) {
                     b = a * (yw * yw - f * f) / (ywm * ywm);
                     working_space[2 * shift + k] += b * c; //der
                     b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                     working_space[3 * shift + k] += b * c; //temp
                  }
                  
                  else {
                     b = a * (yw - f) / ywm;
                     working_space[2 * shift + k] += b * c; //der
                     b = a * a / ywm;
                     working_space[3 * shift + k] += b * c; //temp
                  }
               }
               k += 1;
            }
            if (fFixTxy == false) {
               a = Dertxy(fNPeaks, (Double_t) i1, (Double_t) i2,
                           working_space, working_space[peak_vel],
                           working_space[peak_vel + 1],
                           working_space[peak_vel + 12],
                           working_space[peak_vel + 13]);
               if (ywm != 0) {
                  c = Ourpowl(a, pw);
                  if (fStatisticType == kFitOptimChiFuncValues) {
                     b = a * (yw * yw - f * f) / (ywm * ywm);
                     working_space[2 * shift + k] += b * c; //der
                     b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                     working_space[3 * shift + k] += b * c; //temp
                  }
                  
                  else {
                     b = a * (yw - f) / ywm;
                     working_space[2 * shift + k] += b * c; //der
                     b = a * a / ywm;
                     working_space[3 * shift + k] += b * c; //temp
                  }
               }
               k += 1;
            }
            if (fFixSxy == false) {
               a = Dersxy(fNPeaks, (Double_t) i1, (Double_t) i2,
                           working_space, working_space[peak_vel],
                           working_space[peak_vel + 1]);
               if (ywm != 0) {
                  c = Ourpowl(a, pw);
                  if (fStatisticType == kFitOptimChiFuncValues) {
                     b = a * (yw * yw - f * f) / (ywm * ywm);
                     working_space[2 * shift + k] += b * c; //der
                     b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                     working_space[3 * shift + k] += b * c; //temp
                  }
                  
                  else {
                     b = a * (yw - f) / ywm;
                     working_space[2 * shift + k] += b * c; //der
                     b = a * a / ywm;
                     working_space[3 * shift + k] += b * c; //temp
                  }
               }
               k += 1;
            }
            if (fFixTx == false) {
               a = Dertx(fNPeaks, (Double_t) i1, working_space,
                          working_space[peak_vel],
                          working_space[peak_vel + 12]);
               if (ywm != 0) {
                  c = Ourpowl(a, pw);
                  if (fStatisticType == kFitOptimChiFuncValues) {
                     b = a * (yw * yw - f * f) / (ywm * ywm);
                     working_space[2 * shift + k] += b * c; //der
                     b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                     working_space[3 * shift + k] += b * c; //temp
                  }
                  
                  else {
                     b = a * (yw - f) / ywm;
                     working_space[2 * shift + k] += b * c; //der
                     b = a * a / ywm;
                     working_space[3 * shift + k] += b * c; //temp
                  }
               }
               k += 1;
            }
            if (fFixTy == false) {
               a = Derty(fNPeaks, (Double_t) i2, working_space,
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 13]);
               if (ywm != 0) {
                  c = Ourpowl(a, pw);
                  if (fStatisticType == kFitOptimChiFuncValues) {
                     b = a * (yw * yw - f * f) / (ywm * ywm);
                     working_space[2 * shift + k] += b * c; //der
                     b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                     working_space[3 * shift + k] += b * c; //temp
                  }
                  
                  else {
                     b = a * (yw - f) / ywm;
                     working_space[2 * shift + k] += b * c; //der
                     b = a * a / ywm;
                     working_space[3 * shift + k] += b * c; //temp
                  }
               }
               k += 1;
            }
            if (fFixSx == false) {
               a = Dersx(fNPeaks, (Double_t) i1, working_space,
                          working_space[peak_vel]);
               if (ywm != 0) {
                  c = Ourpowl(a, pw);
                  if (fStatisticType == kFitOptimChiFuncValues) {
                     b = a * (yw * yw - f * f) / (ywm * ywm);
                     working_space[2 * shift + k] += b * c; //der
                     b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                     working_space[3 * shift + k] += b * c; //temp
                  }
                  
                  else {
                     b = a * (yw - f) / ywm;
                     working_space[2 * shift + k] += b * c; //der
                     b = a * a / ywm;
                     working_space[3 * shift + k] += b * c; //temp
                  }
               }
               k += 1;
            }
            if (fFixSy == false) {
               a = Dersy(fNPeaks, (Double_t) i2, working_space,
                          working_space[peak_vel + 1]);
               if (ywm != 0) {
                  c = Ourpowl(a, pw);
                  if (fStatisticType == kFitOptimChiFuncValues) {
                     b = a * (yw * yw - f * f) / (ywm * ywm);
                     working_space[2 * shift + k] += b * c; //der
                     b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                     working_space[3 * shift + k] += b * c; //temp
                  }
                  
                  else {
                     b = a * (yw - f) / ywm;
                     working_space[2 * shift + k] += b * c; //der
                     b = a * a / ywm;
                     working_space[3 * shift + k] += b * c; //temp
                  }
               }
               k += 1;
            }
            if (fFixBx == false) {
               a = Derbx(fNPeaks, (Double_t) i1, (Double_t) i2,
                          working_space, working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 6],
                          working_space[peak_vel + 8],
                          working_space[peak_vel + 12],
                          working_space[peak_vel + 13]);
               if (ywm != 0) {
                  c = Ourpowl(a, pw);
                  if (fStatisticType == kFitOptimChiFuncValues) {
                     b = a * (yw * yw - f * f) / (ywm * ywm);
                     working_space[2 * shift + k] += b * c; //der
                     b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                     working_space[3 * shift + k] += b * c; //temp
                  }
                  
                  else {
                     b = a * (yw - f) / ywm;
                     working_space[2 * shift + k] += b * c; //der
                     b = a * a / ywm;
                     working_space[3 * shift + k] += b * c; //temp
                  }
               }
               k += 1;
            }
            if (fFixBy == false) {
               a = Derby(fNPeaks, (Double_t) i1, (Double_t) i2,
                          working_space, working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 6],
                          working_space[peak_vel + 8],
                          working_space[peak_vel + 12],
                          working_space[peak_vel + 13]);
               if (ywm != 0) {
                  c = Ourpowl(a, pw);
                  if (fStatisticType == kFitOptimChiFuncValues) {
                     b = a * (yw * yw - f * f) / (ywm * ywm);
                     working_space[2 * shift + k] += b * c; //der
                     b = a * a * (4 * yw - 2 * f) / (ywm * ywm);
                     working_space[3 * shift + k] += b * c; //temp
                  }
                  
                  else {
                     b = a * (yw - f) / ywm;
                     working_space[2 * shift + k] += b * c; //der
                     b = a * a / ywm;
                     working_space[3 * shift + k] += b * c; //temp
                  }
               }
               k += 1;
            }
         }
      }
      for (j = 0; j < size; j++) {
         if (TMath::Abs(working_space[3 * shift + j]) > 0.000001)
            working_space[2 * shift + j] = working_space[2 * shift + j] / TMath::Abs(working_space[3 * shift + j]); //der[j]=der[j]/temp[j]
         else
            working_space[2 * shift + j] = 0; //der[j]
      }
      
      //calculate chi_opt
      chi2 = chi_opt;
      chi_opt = TMath::Sqrt(TMath::Abs(chi_opt));
      
      //calculate new parameters
      regul_cycle = 0;
      for (j = 0; j < size; j++) {
         working_space[4 * shift + j] = working_space[shift + j]; //temp_xk[j]=xk[j]
      }
      
      do {
         if (fAlphaOptim == kFitAlphaOptimal) {
            if (fStatisticType != kFitOptimMaxLikelihood)
               chi_min = 10000 * chi2;
            
            else
               chi_min = 0.1 * chi2;
            flag = 0;
            for (pi = 0.1; flag == 0 && pi <= 100; pi += 0.1) {
               for (j = 0; j < size; j++) {
                  working_space[shift + j] = working_space[4 * shift + j] + pi * alpha * working_space[2 * shift + j]; //xk[j]=temp_xk[j]+pi*alpha*der[j]
               }
               for (i = 0, j = 0; i < fNPeaks; i++) {
                  if (fFixAmp[i] == false) {
                     if (working_space[shift + j] < 0) //xk[j]
                        working_space[shift + j] = 0; //xk[j]
                     working_space[7 * i] = working_space[shift + j]; //parameter[7*i]=xk[j]
                     j += 1;
                  }
                  if (fFixPositionX[i] == false) {
                     if (working_space[shift + j] < fXmin) //xk[j]
                        working_space[shift + j] = fXmin; //xk[j]
                     if (working_space[shift + j] > fXmax) //xk[j]
                        working_space[shift + j] = fXmax; //xk[j]
                     working_space[7 * i + 1] = working_space[shift + j]; //parameter[7*i+1]=xk[j]
                     j += 1;
                  }
                  if (fFixPositionY[i] == false) {
                     if (working_space[shift + j] < fYmin) //xk[j]
                        working_space[shift + j] = fYmin; //xk[j]
                     if (working_space[shift + j] > fYmax) //xk[j]
                        working_space[shift + j] = fYmax; //xk[j]
                     working_space[7 * i + 2] = working_space[shift + j]; //parameter[7*i+2]=xk[j]
                     j += 1;
                  }
                  if (fFixAmpX1[i] == false) {
                     if (working_space[shift + j] < 0) //xk[j]
                        working_space[shift + j] = 0; //xk[j]
                     working_space[7 * i + 3] = working_space[shift + j]; //parameter[7*i+3]=xk[j]
                     j += 1;
                  }
                  if (fFixAmpY1[i] == false) {
                     if (working_space[shift + j] < 0) //xk[j]
                        working_space[shift + j] = 0; //xk[j]
                     working_space[7 * i + 4] = working_space[shift + j]; //parameter[7*i+4]=xk[j]
                     j += 1;
                  }
                  if (fFixPositionX1[i] == false) {
                     if (working_space[shift + j] < fXmin) //xk[j]
                        working_space[shift + j] = fXmin; //xk[j]
                     if (working_space[shift + j] > fXmax) //xk[j]
                        working_space[shift + j] = fXmax; //xk[j]
                     working_space[7 * i + 5] = working_space[shift + j]; //parameter[7*i+5]=xk[j]
                     j += 1;
                  }
                  if (fFixPositionY1[i] == false) {
                     if (working_space[shift + j] < fYmin) //xk[j]
                        working_space[shift + j] = fYmin; //xk[j]
                     if (working_space[shift + j] > fYmax) //xk[j]
                        working_space[shift + j] = fYmax; //xk[j]
                     working_space[7 * i + 6] = working_space[shift + j]; //parameter[7*i+6]=xk[j]
                     j += 1;
                  }
               }
               if (fFixSigmaX == false) {
                  if (working_space[shift + j] < 0.001) { //xk[j]
                     working_space[shift + j] = 0.001; //xk[j]
                  }
                  working_space[peak_vel] = working_space[shift + j]; //parameter[peak_vel]=xk[j]
                  j += 1;
               }
               if (fFixSigmaY == false) {
                  if (working_space[shift + j] < 0.001) { //xk[j]
                     working_space[shift + j] = 0.001; //xk[j]
                  }
                  working_space[peak_vel + 1] = working_space[shift + j]; //parameter[peak_vel+1]=xk[j]
                  j += 1;
               }
               if (fFixRo == false) {
                  if (working_space[shift + j] < -1) { //xk[j]
                     working_space[shift + j] = -1; //xk[j]
                  }
                  if (working_space[shift + j] > 1) { //xk[j]
                     working_space[shift + j] = 1; //xk[j]
                  }
                  working_space[peak_vel + 2] = working_space[shift + j]; //parameter[peak_vel+2]=xk[j]
                  j += 1;
               }
               if (fFixA0 == false) {
                  working_space[peak_vel + 3] = working_space[shift + j]; //parameter[peak_vel+3]=xk[j]
                  j += 1;
               }
               if (fFixAx == false) {
                  working_space[peak_vel + 4] = working_space[shift + j]; //parameter[peak_vel+4]=xk[j]
                  j += 1;
               }
               if (fFixAy == false) {
                  working_space[peak_vel + 5] = working_space[shift + j]; //parameter[peak_vel+5]=xk[j]
                  j += 1;
               }
               if (fFixTxy == false) {
                  working_space[peak_vel + 6] = working_space[shift + j]; //parameter[peak_vel+6]=xk[j]
                  j += 1;
               }
               if (fFixSxy == false) {
                  working_space[peak_vel + 7] = working_space[shift + j]; //parameter[peak_vel+7]=xk[j]
                  j += 1;
               }
               if (fFixTx == false) {
                  working_space[peak_vel + 8] = working_space[shift + j]; //parameter[peak_vel+8]=xk[j]
                  j += 1;
               }
               if (fFixTy == false) {
                  working_space[peak_vel + 9] = working_space[shift + j]; //parameter[peak_vel+9]=xk[j]
                  j += 1;
               }
               if (fFixSx == false) {
                  working_space[peak_vel + 10] = working_space[shift + j]; //parameter[peak_vel+10]=xk[j]
                  j += 1;
               }
               if (fFixSy == false) {
                  working_space[peak_vel + 11] = working_space[shift + j]; //parameter[peak_vel+11]=xk[j]
                  j += 1;
               }
               if (fFixBx == false) {
                  if (TMath::Abs(working_space[shift + j]) < 0.001) { //xk[j]
                     if (working_space[shift + j] < 0) //xk[j]
                        working_space[shift + j] = -0.001; //xk[j]
                     else
                        working_space[shift + j] = 0.001; //xk[j]
                  }
                  working_space[peak_vel + 12] = working_space[shift + j]; //parameter[peak_vel+12]=xk[j]
                  j += 1;
               }
               if (fFixBy == false) {
                  if (TMath::Abs(working_space[shift + j]) < 0.001) { //xk[j]
                     if (working_space[shift + j] < 0) //xk[j]
                        working_space[shift + j] = -0.001; //xk[j]
                     else
                        working_space[shift + j] = 0.001; //xk[j]
                  }
                  working_space[peak_vel + 13] = working_space[shift + j]; //parameter[peak_vel+13]=xk[j]
                  j += 1;
               }
               chi2 = 0;
               for (i1 = fXmin; i1 <= fXmax; i1++) {
                  for (i2 = fYmin; i2 <= fYmax; i2++) {
                     yw = source[i1][i2];
                     ywm = yw;
                     f = Shape2(fNPeaks, (Double_t) i1,
                                 (Double_t) i2, working_space,
                                 working_space[peak_vel],
                                 working_space[peak_vel + 1],
                                 working_space[peak_vel + 2],
                                 working_space[peak_vel + 3],
                                 working_space[peak_vel + 4],
                                 working_space[peak_vel + 5],
                                 working_space[peak_vel + 6],
                                 working_space[peak_vel + 7],
                                 working_space[peak_vel + 8],
                                 working_space[peak_vel + 9],
                                 working_space[peak_vel + 10],
                                 working_space[peak_vel + 11],
                                 working_space[peak_vel + 12],
                                 working_space[peak_vel + 13]);
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
               for (j = 0; j < size; j++) {
                  working_space[shift + j] = working_space[4 * shift + j] + pmin * alpha * working_space[2 * shift + j]; //xk[j]=temp_xk[j]+pmin*alpha*der[j]
               }
               for (i = 0, j = 0; i < fNPeaks; i++) {
                  if (fFixAmp[i] == false) {
                     if (working_space[shift + j] < 0) //xk[j]
                        working_space[shift + j] = 0; //xk[j]
                     working_space[7 * i] = working_space[shift + j]; //parameter[7*i]=xk[j]
                     j += 1;
                  }
                  if (fFixPositionX[i] == false) {
                     if (working_space[shift + j] < fXmin) //xk[j]
                        working_space[shift + j] = fXmin; //xk[j]
                     if (working_space[shift + j] > fXmax) //xk[j]
                        working_space[shift + j] = fXmax; //xk[j]
                     working_space[7 * i + 1] = working_space[shift + j]; //parameter[7*i+1]=xk[j]
                     j += 1;
                  }
                  if (fFixPositionY[i] == false) {
                     if (working_space[shift + j] < fYmin) //xk[j]
                        working_space[shift + j] = fYmin; //xk[j]
                     if (working_space[shift + j] > fYmax) //xk[j]
                        working_space[shift + j] = fYmax; //xk[j]
                     working_space[7 * i + 2] = working_space[shift + j]; //parameter[7*i+2]=xk[j]
                     j += 1;
                  }
                  if (fFixAmpX1[i] == false) {
                     if (working_space[shift + j] < 0) //xk[j]
                        working_space[shift + j] = 0; //xk[j]
                     working_space[7 * i + 3] = working_space[shift + j]; //parameter[7*i+3]=xk[j]
                     j += 1;
                  }
                  if (fFixAmpY1[i] == false) {
                     if (working_space[shift + j] < 0) //xk[j]
                        working_space[shift + j] = 0; //xk[j]
                     working_space[7 * i + 4] = working_space[shift + j]; //parameter[7*i+4]=xk[j]
                     j += 1;
                  }
                  if (fFixPositionX1[i] == false) {
                     if (working_space[shift + j] < fXmin) //xk[j]
                        working_space[shift + j] = fXmin; //xk[j]
                     if (working_space[shift + j] > fXmax) //xk[j]
                        working_space[shift + j] = fXmax; //xk[j]
                     working_space[7 * i + 5] = working_space[shift + j]; //parameter[7*i+5]=xk[j]
                     j += 1;
                  }
                  if (fFixPositionY1[i] == false) {
                     if (working_space[shift + j] < fYmin) //xk[j]
                        working_space[shift + j] = fYmin; //xk[j]
                     if (working_space[shift + j] > fYmax) //xk[j]
                        working_space[shift + j] = fYmax; //xk[j]
                     working_space[7 * i + 6] = working_space[shift + j]; //parameter[7*i+6]=xk[j]
                     j += 1;
                  }
               }
               if (fFixSigmaX == false) {
                  if (working_space[shift + j] < 0.001) { //xk[j]
                     working_space[shift + j] = 0.001; //xk[j]
                  }
                  working_space[peak_vel] = working_space[shift + j]; //parameter[peak_vel]=xk[j]
                  j += 1;
               }
               if (fFixSigmaY == false) {
                  if (working_space[shift + j] < 0.001) { //xk[j]
                     working_space[shift + j] = 0.001; //xk[j]
                  }
                  working_space[peak_vel + 1] = working_space[shift + j]; //parameter[peak_vel+1]=xk[j]
                  j += 1;
               }
               if (fFixRo == false) {
                  if (working_space[shift + j] < -1) { //xk[j]
                     working_space[shift + j] = -1; //xk[j]
                  }
                  if (working_space[shift + j] > 1) { //xk[j]
                     working_space[shift + j] = 1; //xk[j]
                  }
                  working_space[peak_vel + 2] = working_space[shift + j]; //parameter[peak_vel+2]=xk[j]
                  j += 1;
               }
               if (fFixA0 == false) {
                  working_space[peak_vel + 3] = working_space[shift + j]; //parameter[peak_vel+3]=xk[j]
                  j += 1;
               }
               if (fFixAx == false) {
                  working_space[peak_vel + 4] = working_space[shift + j]; //parameter[peak_vel+4]=xk[j]
                  j += 1;
               }
               if (fFixAy == false) {
                  working_space[peak_vel + 5] = working_space[shift + j]; //parameter[peak_vel+5]=xk[j]
                  j += 1;
               }
               if (fFixTxy == false) {
                  working_space[peak_vel + 6] = working_space[shift + j]; //parameter[peak_vel+6]=xk[j]
                  j += 1;
               }
               if (fFixSxy == false) {
                  working_space[peak_vel + 7] = working_space[shift + j]; //parameter[peak_vel+7]=xk[j]
                  j += 1;
               }
               if (fFixTx == false) {
                  working_space[peak_vel + 8] = working_space[shift + j]; //parameter[peak_vel+8]=xk[j]
                  j += 1;
               }
               if (fFixTy == false) {
                  working_space[peak_vel + 9] = working_space[shift + j]; //parameter[peak_vel+9]=xk[j]
                  j += 1;
               }
               if (fFixSx == false) {
                  working_space[peak_vel + 10] = working_space[shift + j]; //parameter[peak_vel+10]=xk[j]
                  j += 1;
               }
               if (fFixSy == false) {
                  working_space[peak_vel + 11] = working_space[shift + j]; //parameter[peak_vel+11]=xk[j]
                  j += 1;
               }
               if (fFixBx == false) {
                  if (TMath::Abs(working_space[shift + j]) < 0.001) { //xk[j]
                     if (working_space[shift + j] < 0) //xk[j]
                        working_space[shift + j] = -0.001; //xk[j]
                     else
                        working_space[shift + j] = 0.001; //xk[j]
                  }
                  working_space[peak_vel + 12] = working_space[shift + j]; //parameter[peak_vel+12]=xk[j]
                  j += 1;
               }
               if (fFixBy == false) {
                  if (TMath::Abs(working_space[shift + j]) < 0.001) { //xk[j]
                     if (working_space[shift + j] < 0) //xk[j]
                        working_space[shift + j] = -0.001; //xk[j]
                     else
                        working_space[shift + j] = 0.001; //xk[j]
                  }
                  working_space[peak_vel + 13] = working_space[shift + j]; //parameter[peak_vel+13]=xk[j]
                  j += 1;
               }
               chi = chi_min;
            }
         }
         
         else {
            for (j = 0; j < size; j++) {
               working_space[shift + j] = working_space[4 * shift + j] + alpha * working_space[2 * shift + j]; //xk[j]=temp_xk[j]+pi*alpha*der[j]
            }
            for (i = 0, j = 0; i < fNPeaks; i++) {
               if (fFixAmp[i] == false) {
                  if (working_space[shift + j] < 0) //xk[j]
                     working_space[shift + j] = 0; //xk[j]
                  working_space[7 * i] = working_space[shift + j]; //parameter[7*i]=xk[j]
                  j += 1;
               }
               if (fFixPositionX[i] == false) {
                  if (working_space[shift + j] < fXmin) //xk[j]
                     working_space[shift + j] = fXmin; //xk[j]
                  if (working_space[shift + j] > fXmax) //xk[j]
                     working_space[shift + j] = fXmax; //xk[j]
                  working_space[7 * i + 1] = working_space[shift + j]; //parameter[7*i+1]=xk[j]
                  j += 1;
               }
               if (fFixPositionY[i] == false) {
                  if (working_space[shift + j] < fYmin) //xk[j]
                     working_space[shift + j] = fYmin; //xk[j]
                  if (working_space[shift + j] > fYmax) //xk[j]
                     working_space[shift + j] = fYmax; //xk[j]
                  working_space[7 * i + 2] = working_space[shift + j]; //parameter[7*i+2]=xk[j]
                  j += 1;
               }
               if (fFixAmpX1[i] == false) {
                  if (working_space[shift + j] < 0) //xk[j]
                     working_space[shift + j] = 0; //xk[j]
                  working_space[7 * i + 3] = working_space[shift + j]; //parameter[7*i+3]=xk[j]
                  j += 1;
               }
               if (fFixAmpY1[i] == false) {
                  if (working_space[shift + j] < 0) //xk[j]
                     working_space[shift + j] = 0; //xk[j]
                  working_space[7 * i + 4] = working_space[shift + j]; //parameter[7*i+4]=xk[j]
                  j += 1;
               }
               if (fFixPositionX1[i] == false) {
                  if (working_space[shift + j] < fXmin) //xk[j]
                     working_space[shift + j] = fXmin; //xk[j]
                  if (working_space[shift + j] > fXmax) //xk[j]
                     working_space[shift + j] = fXmax; //xk[j]
                  working_space[7 * i + 5] = working_space[shift + j]; //parameter[7*i+5]=xk[j]
                  j += 1;
               }
               if (fFixPositionY1[i] == false) {
                  if (working_space[shift + j] < fYmin) //xk[j]
                     working_space[shift + j] = fYmin; //xk[j]
                  if (working_space[shift + j] > fYmax) //xk[j]
                     working_space[shift + j] = fYmax; //xk[j]
                  working_space[7 * i + 6] = working_space[shift + j]; //parameter[7*i+6]=xk[j]
                  j += 1;
               }
            }
            if (fFixSigmaX == false) {
               if (working_space[shift + j] < 0.001) { //xk[j]
                  working_space[shift + j] = 0.001; //xk[j]
               }
               working_space[peak_vel] = working_space[shift + j]; //parameter[peak_vel]=xk[j]
               j += 1;
            }
            if (fFixSigmaY == false) {
               if (working_space[shift + j] < 0.001) { //xk[j]
                  working_space[shift + j] = 0.001; //xk[j]
               }
               working_space[peak_vel + 1] = working_space[shift + j]; //parameter[peak_vel+1]=xk[j]
               j += 1;
            }
            if (fFixRo == false) {
               if (working_space[shift + j] < -1) { //xk[j]
                  working_space[shift + j] = -1; //xk[j]
               }
               if (working_space[shift + j] > 1) { //xk[j]
                  working_space[shift + j] = 1; //xk[j]
               }
               working_space[peak_vel + 2] = working_space[shift + j]; //parameter[peak_vel+2]=xk[j]
               j += 1;
            }
            if (fFixA0 == false) {
               working_space[peak_vel + 3] = working_space[shift + j]; //parameter[peak_vel+3]=xk[j]
               j += 1;
            }
            if (fFixAx == false) {
               working_space[peak_vel + 4] = working_space[shift + j]; //parameter[peak_vel+4]=xk[j]
               j += 1;
            }
            if (fFixAy == false) {
               working_space[peak_vel + 5] = working_space[shift + j]; //parameter[peak_vel+5]=xk[j]
               j += 1;
            }
            if (fFixTxy == false) {
               working_space[peak_vel + 6] = working_space[shift + j]; //parameter[peak_vel+6]=xk[j]
               j += 1;
            }
            if (fFixSxy == false) {
               working_space[peak_vel + 7] = working_space[shift + j]; //parameter[peak_vel+7]=xk[j]
               j += 1;
            }
            if (fFixTx == false) {
               working_space[peak_vel + 8] = working_space[shift + j]; //parameter[peak_vel+8]=xk[j]
               j += 1;
            }
            if (fFixTy == false) {
               working_space[peak_vel + 9] = working_space[shift + j]; //parameter[peak_vel+9]=xk[j]
               j += 1;
            }
            if (fFixSx == false) {
               working_space[peak_vel + 10] = working_space[shift + j]; //parameter[peak_vel+10]=xk[j]
               j += 1;
            }
            if (fFixSy == false) {
               working_space[peak_vel + 11] = working_space[shift + j]; //parameter[peak_vel+11]=xk[j]
               j += 1;
            }
            if (fFixBx == false) {
               if (TMath::Abs(working_space[shift + j]) < 0.001) { //xk[j]
                  if (working_space[shift + j] < 0) //xk[j]
                     working_space[shift + j] = -0.001; //xk[j]
                  else
                     working_space[shift + j] = 0.001; //xk[j]
               }
               working_space[peak_vel + 12] = working_space[shift + j]; //parameter[peak_vel+12]=xk[j]
               j += 1;
            }
            if (fFixBy == false) {
               if (TMath::Abs(working_space[shift + j]) < 0.001) { //xk[j]
                  if (working_space[shift + j] < 0) //xk[j]
                     working_space[shift + j] = -0.001; //xk[j]
                  else
                     working_space[shift + j] = 0.001; //xk[j]
               }
               working_space[peak_vel + 13] = working_space[shift + j]; //parameter[peak_vel+13]=xk[j]
               j += 1;
            }
            chi = 0;
            for (i1 = fXmin; i1 <= fXmax; i1++) {
               for (i2 = fYmin; i2 <= fYmax; i2++) {
                  yw = source[i1][i2];
                  ywm = yw;
                  f = Shape2(fNPeaks, (Double_t) i1, (Double_t) i2,
                              working_space, working_space[peak_vel],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 2],
                              working_space[peak_vel + 3],
                              working_space[peak_vel + 4],
                              working_space[peak_vel + 5],
                              working_space[peak_vel + 6],
                              working_space[peak_vel + 7],
                              working_space[peak_vel + 8],
                              working_space[peak_vel + 9],
                              working_space[peak_vel + 10],
                              working_space[peak_vel + 11],
                              working_space[peak_vel + 12],
                              working_space[peak_vel + 13]);
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
      for (j = 0; j < size; j++) {
         working_space[4 * shift + j] = 0; //temp_xk[j]
         working_space[2 * shift + j] = 0; //der[j]
      }
      for (i1 = fXmin, chi_cel = 0; i1 <= fXmax; i1++) {
         for (i2 = fYmin; i2 <= fYmax; i2++) {
            yw = source[i1][i2];
            if (yw == 0)
               yw = 1;
            f = Shape2(fNPeaks, (Double_t) i1, (Double_t) i2,
                        working_space, working_space[peak_vel],
                        working_space[peak_vel + 1],
                        working_space[peak_vel + 2],
                        working_space[peak_vel + 3],
                        working_space[peak_vel + 4],
                        working_space[peak_vel + 5],
                        working_space[peak_vel + 6],
                        working_space[peak_vel + 7],
                        working_space[peak_vel + 8],
                        working_space[peak_vel + 9],
                        working_space[peak_vel + 10],
                        working_space[peak_vel + 11],
                        working_space[peak_vel + 12],
                        working_space[peak_vel + 13]);
            chi_opt = (yw - f) * (yw - f) / yw;
            chi_cel += (yw - f) * (yw - f) / yw;
            
                //calculate gradient vector
                for (j = 0, k = 0; j < fNPeaks; j++) {
               if (fFixAmp[j] == false) {
                  a = Deramp2((Double_t) i1, (Double_t) i2,
                               working_space[7 * j + 1],
                               working_space[7 * j + 2],
                               working_space[peak_vel],
                               working_space[peak_vel + 1],
                               working_space[peak_vel + 2],
                               working_space[peak_vel + 6],
                               working_space[peak_vel + 7],
                               working_space[peak_vel + 12],
                               working_space[peak_vel + 13]);
                  if (yw != 0) {
                     c = Ourpowl(a, pw);
                     working_space[2 * shift + k] += chi_opt * c; //der[k]
                     b = a * a / yw;
                     working_space[4 * shift + k] += b * c; //temp_xk[k]
                  }
                  k += 1;
               }
               if (fFixPositionX[j] == false) {
                  a = Deri02((Double_t) i1, (Double_t) i2,
                              working_space[7 * j],
                              working_space[7 * j + 1],
                              working_space[7 * j + 2],
                              working_space[peak_vel],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 2],
                              working_space[peak_vel + 6],
                              working_space[peak_vel + 7],
                              working_space[peak_vel + 12],
                              working_space[peak_vel + 13]);
                  if (yw != 0) {
                     c = Ourpowl(a, pw);
                     working_space[2 * shift + k] += chi_opt * c; //der[k]
                     b = a * a / yw;
                     working_space[4 * shift + k] += b * c; //temp_xk[k]
                  }
                  k += 1;
               }
               if (fFixPositionY[j] == false) {
                  a = Derj02((Double_t) i1, (Double_t) i2,
                              working_space[7 * j],
                              working_space[7 * j + 1],
                              working_space[7 * j + 2],
                              working_space[peak_vel],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 2],
                              working_space[peak_vel + 6],
                              working_space[peak_vel + 7],
                              working_space[peak_vel + 12],
                              working_space[peak_vel + 13]);
                  if (yw != 0) {
                     c = Ourpowl(a, pw);
                     working_space[2 * shift + k] += chi_opt * c; //der[k]
                     b = a * a / yw;
                     working_space[4 * shift + k] += b * c; //temp_xk[k]
                  }
                  k += 1;
               }
               if (fFixAmpX1[j] == false) {
                  a = Derampx((Double_t) i1, working_space[7 * j + 5],
                               working_space[peak_vel],
                               working_space[peak_vel + 8],
                               working_space[peak_vel + 10],
                               working_space[peak_vel + 12]);
                  if (yw != 0) {
                     c = Ourpowl(a, pw);
                     working_space[2 * shift + k] += chi_opt * c; //der[k]
                     b = a * a / yw;
                     working_space[4 * shift + k] += b * c; //temp_xk[k]
                  }
                  k += 1;
               }
               if (fFixAmpY1[j] == false) {
                  a = Derampx((Double_t) i2, working_space[7 * j + 6],
                               working_space[peak_vel + 1],
                               working_space[peak_vel + 9],
                               working_space[peak_vel + 11],
                               working_space[peak_vel + 13]);
                  if (yw != 0) {
                     c = Ourpowl(a, pw);
                     working_space[2 * shift + k] += chi_opt * c; //der[k]
                     b = a * a / yw;
                     working_space[4 * shift + k] += b * c; //temp_xk[k]
                  }
                  k += 1;
               }
               if (fFixPositionX1[j] == false) {
                  a = Deri01((Double_t) i1, working_space[7 * j + 3],
                              working_space[7 * j + 5],
                              working_space[peak_vel],
                              working_space[peak_vel + 8],
                              working_space[peak_vel + 10],
                              working_space[peak_vel + 12]);
                  if (yw != 0) {
                     c = Ourpowl(a, pw);
                     working_space[2 * shift + k] += chi_opt * c; //der[k]
                     b = a * a / yw;
                     working_space[4 * shift + k] += b * c; //temp_xk[k]
                  }
                  k += 1;
               }
               if (fFixPositionY1[j] == false) {
                  a = Deri01((Double_t) i2, working_space[7 * j + 4],
                              working_space[7 * j + 6],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 9],
                              working_space[peak_vel + 11],
                              working_space[peak_vel + 13]);
                  if (yw != 0) {
                     c = Ourpowl(a, pw);
                     working_space[2 * shift + k] += chi_opt * c; //der[k]
                     b = a * a / yw;
                     working_space[4 * shift + k] += b * c; //temp_xk[k]
                  }
                  k += 1;
               }
            }
            if (fFixSigmaX == false) {
               a = Dersigmax(fNPeaks, (Double_t) i1, (Double_t) i2,
                              working_space, working_space[peak_vel],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 2],
                              working_space[peak_vel + 6],
                              working_space[peak_vel + 7],
                              working_space[peak_vel + 8],
                              working_space[peak_vel + 10],
                              working_space[peak_vel + 12],
                              working_space[peak_vel + 13]);
               if (yw != 0) {
                  c = Ourpowl(a, pw);
                  working_space[2 * shift + k] += chi_opt * c; //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b * c; //temp_xk[k]
               }
               k += 1;
            }
            if (fFixSigmaY == false) {
               a = Dersigmay(fNPeaks, (Double_t) i1, (Double_t) i2,
                              working_space, working_space[peak_vel],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 2],
                              working_space[peak_vel + 6],
                              working_space[peak_vel + 7],
                              working_space[peak_vel + 9],
                              working_space[peak_vel + 11],
                              working_space[peak_vel + 12],
                              working_space[peak_vel + 13]);
               if (yw != 0) {
                  c = Ourpowl(a, pw);
                  working_space[2 * shift + k] += chi_opt * c; //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b * c; //temp_xk[k]
               }
               k += 1;
            }
            if (fFixRo == false) {
               a = Derro(fNPeaks, (Double_t) i1, (Double_t) i2,
                          working_space, working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 2]);
               if (yw != 0) {
                  c = Ourpowl(a, pw);
                  working_space[2 * shift + k] += chi_opt * c; //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b * c; //temp_xk[k]
               }
               k += 1;
            }
            if (fFixA0 == false) {
               a = 1.;
               if (yw != 0) {
                  c = Ourpowl(a, pw);
                  working_space[2 * shift + k] += chi_opt * c; //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b * c; //temp_xk[k]
               }
               k += 1;
            }
            if (fFixAx == false) {
               a = i1;
               if (yw != 0) {
                  c = Ourpowl(a, pw);
                  working_space[2 * shift + k] += chi_opt * c; //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b * c; //temp_xk[k]
               }
               k += 1;
            }
            if (fFixAy == false) {
               a = i2;
               if (yw != 0) {
                  c = Ourpowl(a, pw);
                  working_space[2 * shift + k] += chi_opt * c; //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b * c; //temp_xk[k]
               }
               k += 1;
            }
            if (fFixTxy == false) {
               a = Dertxy(fNPeaks, (Double_t) i1, (Double_t) i2,
                           working_space, working_space[peak_vel],
                           working_space[peak_vel + 1],
                           working_space[peak_vel + 12],
                           working_space[peak_vel + 13]);
               if (yw != 0) {
                  c = Ourpowl(a, pw);
                  working_space[2 * shift + k] += chi_opt * c; //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b * c; //temp_xk[k]
               }
               k += 1;
            }
            if (fFixSxy == false) {
               a = Dersxy(fNPeaks, (Double_t) i1, (Double_t) i2,
                           working_space, working_space[peak_vel],
                           working_space[peak_vel + 1]);
               if (yw != 0) {
                  c = Ourpowl(a, pw);
                  working_space[2 * shift + k] += chi_opt * c; //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b * c; //temp_xk[k]
               }
               k += 1;
            }
            if (fFixTx == false) {
               a = Dertx(fNPeaks, (Double_t) i1, working_space,
                          working_space[peak_vel],
                          working_space[peak_vel + 12]);
               if (yw != 0) {
                  c = Ourpowl(a, pw);
                  working_space[2 * shift + k] += chi_opt * c; //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b * c; //temp_xk[k]
               }
               k += 1;
            }
            if (fFixTy == false) {
               a = Derty(fNPeaks, (Double_t) i2, working_space,
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 13]);
               if (yw != 0) {
                  c = Ourpowl(a, pw);
                  working_space[2 * shift + k] += chi_opt * c; //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b * c; //temp_xk[k]
               }
               k += 1;
            }
            if (fFixSx == false) {
               a = Dersx(fNPeaks, (Double_t) i1, working_space,
                          working_space[peak_vel]);
               if (yw != 0) {
                  c = Ourpowl(a, pw);
                  working_space[2 * shift + k] += chi_opt * c; //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b * c; //temp_xk[k]
               }
               k += 1;
            }
            if (fFixSy == false) {
               a = Dersy(fNPeaks, (Double_t) i2, working_space,
                          working_space[peak_vel + 1]);
               if (yw != 0) {
                  c = Ourpowl(a, pw);
                  working_space[2 * shift + k] += chi_opt * c; //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b * c; //temp_xk[k]
               }
               k += 1;
            }
            if (fFixBx == false) {
               a = Derbx(fNPeaks, (Double_t) i1, (Double_t) i2,
                          working_space, working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 6],
                          working_space[peak_vel + 8],
                          working_space[peak_vel + 12],
                          working_space[peak_vel + 13]);
               if (yw != 0) {
                  c = Ourpowl(a, pw);
                  working_space[2 * shift + k] += chi_opt * c; //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b * c; //temp_xk[k]
               }
               k += 1;
            }
            if (fFixBy == false) {
               a = Derby(fNPeaks, (Double_t) i1, (Double_t) i2,
                          working_space, working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 6],
                          working_space[peak_vel + 8],
                          working_space[peak_vel + 12],
                          working_space[peak_vel + 13]);
               if (yw != 0) {
                  c = Ourpowl(a, pw);
                  working_space[2 * shift + k] += chi_opt * c; //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b * c; //temp_xk[k]
               }
               k += 1;
            }
         }
      }
   }
   b = (fXmax - fXmin + 1) * (fYmax - fYmin + 1) - size;
   chi_er = chi_cel / b;
   for (i = 0, j = 0; i < fNPeaks; i++) {
      fVolume[i] =
          Volume(working_space[7 * i], working_space[peak_vel],
                 working_space[peak_vel + 1], working_space[peak_vel + 2]);
      if (fVolume[i] > 0) {
         c = 0;
         if (fFixAmp[i] == false) {
            a = Derpa2(working_space[peak_vel],
                        working_space[peak_vel + 1],
                        working_space[peak_vel + 2]);
            b = working_space[4 * shift + j]; //temp_xk[j]
            if (b == 0)
               b = 1;
            
            else
               b = 1 / b;
            c = c + a * a * b;
         }
         if (fFixSigmaX == false) {
            a = Derpsigmax(working_space[shift + j],
                            working_space[peak_vel + 1],
                            working_space[peak_vel + 2]);
            b = working_space[4 * shift + peak_vel]; //temp_xk[j]
            if (b == 0)
               b = 1;
            
            else
               b = 1 / b;
            c = c + a * a * b;
         }
         if (fFixSigmaY == false) {
            a = Derpsigmay(working_space[shift + j],
                            working_space[peak_vel],
                            working_space[peak_vel + 2]);
            b = working_space[4 * shift + peak_vel + 1]; //temp_xk[j]
            if (b == 0)
               b = 1;
            
            else
               b = 1 / b;
            c = c + a * a * b;
         }
         if (fFixRo == false) {
            a = Derpro(working_space[shift + j], working_space[peak_vel],
                        working_space[peak_vel + 1],
                        working_space[peak_vel + 2]);
            b = working_space[4 * shift + peak_vel + 2]; //temp_xk[j]
            if (b == 0)
               b = 1;
            
            else
               b = 1 / b;
            c = c + a * a * b;
         }
         fVolumeErr[i] = TMath::Sqrt(TMath::Abs(chi_er * c));
      }
      
      else {
         fVolumeErr[i] = 0;
      }
      if (fFixAmp[i] == false) {
         fAmpCalc[i] = working_space[shift + j]; //xk[j]
         if (working_space[3 * shift + j] != 0)
            fAmpErr[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
         j += 1;
      }
      
      else {
         fAmpCalc[i] = fAmpInit[i];
         fAmpErr[i] = 0;
      }
      if (fFixPositionX[i] == false) {
         fPositionCalcX[i] = working_space[shift + j]; //xk[j]
         if (working_space[3 * shift + j] != 0)
            fPositionErrX[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
         j += 1;
      }
      
      else {
         fPositionCalcX[i] = fPositionInitX[i];
         fPositionErrX[i] = 0;
      }
      if (fFixPositionY[i] == false) {
         fPositionCalcY[i] = working_space[shift + j]; //xk[j]
         if (working_space[3 * shift + j] != 0)
            fPositionErrY[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
         j += 1;
      }
      
      else {
         fPositionCalcY[i] = fPositionInitY[i];
         fPositionErrY[i] = 0;
      }
      if (fFixAmpX1[i] == false) {
         fAmpCalcX1[i] = working_space[shift + j]; //xk[j]
         if (working_space[3 * shift + j] != 0)
            fAmpErrX1[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
         j += 1;
      }
      
      else {
         fAmpCalcX1[i] = fAmpInitX1[i];
         fAmpErrX1[i] = 0;
      }
      if (fFixAmpY1[i] == false) {
         fAmpCalcY1[i] = working_space[shift + j]; //xk[j]
         if (working_space[3 * shift + j] != 0)
            fAmpErrY1[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
         j += 1;
      }
      
      else {
         fAmpCalcY1[i] = fAmpInitY1[i];
         fAmpErrY1[i] = 0;
      }
      if (fFixPositionX1[i] == false) {
         fPositionCalcX1[i] = working_space[shift + j]; //xk[j]
         if (working_space[3 * shift + j] != 0)
            fPositionErrX1[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
         j += 1;
      }
      
      else {
         fPositionCalcX1[i] = fPositionInitX1[i];
         fPositionErrX1[i] = 0;
      }
      if (fFixPositionY1[i] == false) {
         fPositionCalcY1[i] = working_space[shift + j]; //xk[j]
         if (working_space[3 * shift + j] != 0)
            fPositionErrY1[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
         j += 1;
      }
      
      else {
         fPositionCalcY1[i] = fPositionInitY1[i];
         fPositionErrY1[i] = 0;
      }
   }
   if (fFixSigmaX == false) {
      fSigmaCalcX = working_space[shift + j]; //xk[j]
      if (working_space[3 * shift + j] != 0) //temp[j]
         fSigmaErrX = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fSigmaCalcX = fSigmaInitX;
      fSigmaErrX = 0;
   }
   if (fFixSigmaY == false) {
      fSigmaCalcY = working_space[shift + j]; //xk[j]
      if (working_space[3 * shift + j] != 0) //temp[j]
         fSigmaErrY = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fSigmaCalcY = fSigmaInitY;
      fSigmaErrY = 0;
   }
   if (fFixRo == false) {
      fRoCalc = working_space[shift + j]; //xk[j]
      if (working_space[3 * shift + j] != 0) //temp[j]
         fRoErr = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fRoCalc = fRoInit;
      fRoErr = 0;
   }
   if (fFixA0 == false) {
      fA0Calc = working_space[shift + j]; //xk[j]
      if (working_space[3 * shift + j] != 0) //temp[j]
         fA0Err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fA0Calc = fA0Init;
      fA0Err = 0;
   }
   if (fFixAx == false) {
      fAxCalc = working_space[shift + j]; //xk[j]
      if (working_space[3 * shift + j] != 0) //temp[j]
         fAxErr = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fAxCalc = fAxInit;
      fAxErr = 0;
   }
   if (fFixAy == false) {
      fAyCalc = working_space[shift + j]; //xk[j]
      if (working_space[3 * shift + j] != 0) //temp[j]
         fAyErr = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fAyCalc = fAyInit;
      fAyErr = 0;
   }
   if (fFixTxy == false) {
      fTxyCalc = working_space[shift + j]; //xk[j]
      if (working_space[3 * shift + j] != 0) //temp[j]
         fTxyErr = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fTxyCalc = fTxyInit;
      fTxyErr = 0;
   }
   if (fFixSxy == false) {
      fSxyCalc = working_space[shift + j]; //xk[j]
      if (working_space[3 * shift + j] != 0) //temp[j]
         fSxyErr = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fSxyCalc = fSxyInit;
      fSxyErr = 0;
   }
   if (fFixTx == false) {
      fTxCalc = working_space[shift + j]; //xk[j]
      if (working_space[3 * shift + j] != 0) //temp[j]
         fTxErr = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fTxCalc = fTxInit;
      fTxErr = 0;
   }
   if (fFixTy == false) {
      fTyCalc = working_space[shift + j]; //xk[j]
      if (working_space[3 * shift + j] != 0) //temp[j]
         fTyErr = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fTyCalc = fTyInit;
      fTyErr = 0;
   }
   if (fFixSx == false) {
      fSxCalc = working_space[shift + j]; //xk[j]
      if (working_space[3 * shift + j] != 0) //temp[j]
         fSxErr = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fSxCalc = fSxInit;
      fSxErr = 0;
   }
   if (fFixSy == false) {
      fSyCalc = working_space[shift + j]; //xk[j]
      if (working_space[3 * shift + j] != 0) //temp[j]
         fSyErr = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fSyCalc = fSyInit;
      fSyErr = 0;
   }
   if (fFixBx == false) {
      fBxCalc = working_space[shift + j]; //xk[j]
      if (working_space[3 * shift + j] != 0) //temp[j]
         fBxErr = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fBxCalc = fBxInit;
      fBxErr = 0;
   }
   if (fFixBy == false) {
      fByCalc = working_space[shift + j]; //xk[j]
      if (working_space[3 * shift + j] != 0) //temp[j]
         fByErr = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fByCalc = fByInit;
      fByErr = 0;
   }
   b = (fXmax - fXmin + 1) * (fYmax - fYmin + 1) - size;
   fChi = chi_cel / b;
   for (i1 = fXmin; i1 <= fXmax; i1++) {
      for (i2 = fYmin; i2 <= fYmax; i2++) {
         f = Shape2(fNPeaks, (Double_t) i1, (Double_t) i2,
                     working_space, working_space[peak_vel],
                     working_space[peak_vel + 1],
                     working_space[peak_vel + 2],
                     working_space[peak_vel + 3],
                     working_space[peak_vel + 4],
                     working_space[peak_vel + 5],
                     working_space[peak_vel + 6],
                     working_space[peak_vel + 7],
                     working_space[peak_vel + 8],
                     working_space[peak_vel + 9],
                     working_space[peak_vel + 10],
                     working_space[peak_vel + 11],
                     working_space[peak_vel + 12],
                     working_space[peak_vel + 13]);
         source[i1][i2] = f;
      } 
   } 
   delete [] working_space;
   return;
}




// ____________________________________________________________________________________________________________________________
void TSpectrum2Fit::FitStiefel(Float_t **source) 
{
/////////////////////////////////////////////////////////////////////////////
// TWO-DIMENSIONAL FIT FUNCTION USING STIEFEL-HESTENS               
//  ALGORITHM.  
//  This function fits the source spectrum. The calling program should
//  fill in input parameters of the TSpectrum2Fit class. 
//  The fitted parameters are written into           
//  TSpectrum2Fit class output parameters and fitted data are written into 
//  source spectrum.                                                    
//                                                                          
//        Function parameters:                                             
//        source-pointer to the matrix of source spectrum                  
//                                                                         
/////////////////////////////////////////////////////////////////////////////
//   
//Begin_Html <!--
/* -->
<div class=Section1>

<p class=MsoNormal><b><span style='font-size:14.0pt'>Stiefel fitting algorithm</span></b></p>

<p class=MsoNormal style='text-align:justify'><i><span style='font-size:18.0pt'>&nbsp;</span></i></p>

<p class=MsoNormal><i>Function:</i></p>

<p class=MsoNormal style='text-align:justify'>void <a
href="http://root.cern.ch/root/html/TSpectrum.html#TSpectrum:Fit1Awmi"><b>TSpectrumFit2::FitStiefel</b></a>(<a
href="http://root.cern.ch/root/html/ListOfTypes.html#float"><b>float</b></a>
**fSource) </p>

<p class=MsoNormal style='text-align:justify'>This function fits the source
spectrum using Stiefel-Hestens method [1].  The calling program should fill in
input fitting parameters of the TSpectrumFit2 class using a set of
TSpectrumFit2 setters. The fitted parameters are written into the class and the
fitted data are written into source spectrum. It converges faster than Awmi
method.</p>

<p class=MsoNormal><i><span style='color:red'>&nbsp;</span></i></p>

<p class=MsoNormal><i><span style='color:red'>Parameter:</span></i></p>

<p class=MsoNormal style='text-align:justify'>        <b>fSource</b>-pointer to
the matrix of source spectrum                  </p>

<p class=MsoNormal style='text-align:justify'>        </p>

<p class=MsoNormal style='text-align:justify'><b><i>Reference:</i></b></p>

<p class=MsoNormal style='text-align:justify'>[1] B. Mihaila: Analysis of
complex gamma spectra, Rom. Jorn. Phys., Vol. 39, No. 2, (1994), 139-148.</p>

<p class=MsoNormal style='text-align:justify'>&nbsp;</p>

<p class=MsoNormal style='text-align:justify'><i>Example 1 – script FitS.c:</i></p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:18.0pt'><img
border=0 width=602 height=455 src="gif/spectrum2fit_stiefel_image001.jpg"></span></p>

<p class=MsoNormal style='text-align:justify'><b>Fig. 1 Original
two-dimensional spectrum with found peaks (using TSpectrum2 peak searching
function). The positions of peaks were used as initial estimates in fitting
procedure.</b></p>

<p class=MsoNormal style='text-align:justify'><b><span style='font-size:16.0pt'><img
border=0 width=602 height=455 src="gif/spectrum2fit_stiefel_image002.jpg"></span></b></p>

<p class=MsoBodyText2 style='text-align:justify'>Fig. 2 Fitted (generated from
fitted parameters) spectrum of the data from Fig. 1 using Stiefel-Hestens
method. Each peak was represented by 7 parameters, which together with Sigmax,
Sigmay and a0 resulted in 38 fitted parameters. The chi-square after 1000
iterations was 0.642157.</p>

<p class=MsoNormal style='text-align:justify'><span style='font-size:18.0pt'>&nbsp;</span></p>

<p class=MsoNormal><b><span style='color:#339966'>Script:</span></b></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// Example to illustrate
fitting function, algorithm without matrix inversion (AWMI) (class
TSpectrumFit2).</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// To execute this example,
do</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>// root &gt; .x FitStiefel2.C</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>&nbsp;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>void FitStiefel2() {</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Int_t i, j, nfound;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Int_t nbinsx = 64;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Int_t nbinsy = 64;   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   </span><span lang=FR
style='font-size:10.0pt'>Int_t xmin  = 0;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t xmax  =
nbinsx;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t ymin  = 0;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   Int_t ymax  =
nbinsy;</span></p>

<p class=MsoNormal><span lang=FR style='font-size:10.0pt'>   </span><span
style='font-size:10.0pt'>Float_t ** source = new float *[nbinsx];   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Float_t ** dest = new
float *[nbinsx];      </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for (i=0;i&lt;nbinsx;i++)</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                                                source[i]=new
float[nbinsy];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for (i=0;i&lt;nbinsx;i++)</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                                                dest[i]=new
float[nbinsy];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TH2F *search = new
TH2F(&quot;search&quot;,&quot;High resolution peak
searching&quot;,nbinsx,xmin,xmax,nbinsy,ymin,ymax);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TFile *f = new
TFile(&quot;TSpectrum2.root&quot;);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   search=(TH2F*)
f-&gt;Get(&quot;search4;1&quot;);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TCanvas *Searching = new
TCanvas(&quot;Searching&quot;,&quot;Two-dimensional fitting using
Stiefel-Hestens method&quot;,10,10,1000,700);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TSpectrum2 *s = new
TSpectrum2();</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for (i = 0; i &lt; nbinsx;
i++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     for (j = 0; j &lt;
nbinsy; j++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                    source[i][j]
= search-&gt;GetBinContent(i + 1,j + 1); </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                 }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   nfound =
s-&gt;SearchHighRes(source, dest, nbinsx, nbinsy, 2, 5, kTRUE, 3, kFALSE, 3);  
</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   printf(&quot;Found %d
candidate peaks\n&quot;,nfound);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Bool_t *FixPosX = new
Bool_t[nfound];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Bool_t *FixPosY = new
Bool_t[nfound];   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Bool_t *FixAmp = new
Bool_t[nfound];      </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Float_t *PosX = new
Float_t[nfound];         </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Float_t *PosY = new
Float_t[nfound];</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Float_t *Amp = new
Float_t[nfound];      </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   Float_t *AmpXY = new
Float_t[nfound];         </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   PosX = s-&gt;GetPositionX();</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   PosY =
s-&gt;GetPositionY();      </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   for(i = 0; i&lt; nfound ;
i++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>      FixPosX[i] = kFALSE;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>      FixPosY[i] =
kFALSE;      </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>      FixAmp[i] = kFALSE;    </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>      Amp[i] =
source[(int)(PosX[i]+0.5)][(int)(PosY[i]+0.5)];      //initial values of peaks
amplitudes, input parameters          </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>      AmpXY[i] = 0;</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   //filling in the initial
estimates of the input parameters</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   TSpectrumFit2 *pfit=new
TSpectrumFit2(nfound);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
pfit-&gt;SetFitParameters(xmin, xmax-1, ymin, ymax-1, 1000, 0.1,
pfit-&gt;kFitOptimChiCounts, pfit-&gt;kFitAlphaHalving, pfit-&gt;kFitPower2,
pfit-&gt;kFitTaylorOrderFirst);   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
pfit-&gt;SetPeakParameters(2, kFALSE, 2, kFALSE, 0, kTRUE, PosX, (Bool_t *)
FixPosX, PosY, (Bool_t *) FixPosY, PosX, (Bool_t *) FixPosX, PosY, (Bool_t *)
FixPosY, Amp, (Bool_t *) FixAmp, AmpXY, (Bool_t *) FixAmp, AmpXY, (Bool_t *)
FixAmp);      </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
pfit-&gt;SetBackgroundParameters(0, kFALSE, 0, kTRUE, 0, kTRUE);   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
pfit-&gt;FitStiefel(source);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>    for (i = 0; i &lt;
nbinsx; i++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>     for (j = 0; j &lt;
nbinsy; j++){</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                  search-&gt;SetBinContent(i
+ 1, j + 1,source[i][j]);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>                 }</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>   }   </span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>  
search-&gt;Draw(&quot;SURF&quot;);</span></p>

<p class=MsoNormal><span style='font-size:10.0pt'>}</span></p>

</div>

<!-- */
// --> End_Html

   Int_t i, i1, i2, j, k, shift =
       7 * fNPeaks + 14, peak_vel, size, iter, regul_cycle,
       flag;
   Double_t a, b, c, alpha, chi_opt, yw, ywm, f, chi2, chi_min, chi = 0
       , pi, pmin = 0, chi_cel = 0, chi_er;
   Double_t *working_space = new Double_t[5 * (7 * fNPeaks + 14)];
   for (i = 0, j = 0; i < fNPeaks; i++) {
      working_space[7 * i] = fAmpInit[i]; //vector parameter
      if (fFixAmp[i] == false) {
         working_space[shift + j] = fAmpInit[i]; //vector xk
         j += 1;
      }
      working_space[7 * i + 1] = fPositionInitX[i]; //vector parameter
      if (fFixPositionX[i] == false) {
         working_space[shift + j] = fPositionInitX[i]; //vector xk
         j += 1;
      }
      working_space[7 * i + 2] = fPositionInitY[i]; //vector parameter
      if (fFixPositionY[i] == false) {
         working_space[shift + j] = fPositionInitY[i]; //vector xk
         j += 1;
      }
      working_space[7 * i + 3] = fAmpInitX1[i]; //vector parameter
      if (fFixAmpX1[i] == false) {
         working_space[shift + j] = fAmpInitX1[i]; //vector xk
         j += 1;
      }
      working_space[7 * i + 4] = fAmpInitY1[i]; //vector parameter
      if (fFixAmpY1[i] == false) {
         working_space[shift + j] = fAmpInitY1[i]; //vector xk
         j += 1;
      }
      working_space[7 * i + 5] = fPositionInitX1[i]; //vector parameter
      if (fFixPositionX1[i] == false) {
         working_space[shift + j] = fPositionInitX1[i]; //vector xk
         j += 1;
      }
      working_space[7 * i + 6] = fPositionInitY1[i]; //vector parameter
      if (fFixPositionY1[i] == false) {
         working_space[shift + j] = fPositionInitY1[i]; //vector xk
         j += 1;
      }
   }
   peak_vel = 7 * i;
   working_space[7 * i] = fSigmaInitX; //vector parameter
   if (fFixSigmaX == false) {
      working_space[shift + j] = fSigmaInitX; //vector xk
      j += 1;
   }
   working_space[7 * i + 1] = fSigmaInitY; //vector parameter
   if (fFixSigmaY == false) {
      working_space[shift + j] = fSigmaInitY; //vector xk
      j += 1;
   }
   working_space[7 * i + 2] = fRoInit; //vector parameter
   if (fFixRo == false) {
      working_space[shift + j] = fRoInit; //vector xk
      j += 1;
   }
   working_space[7 * i + 3] = fA0Init; //vector parameter
   if (fFixA0 == false) {
      working_space[shift + j] = fA0Init; //vector xk
      j += 1;
   }
   working_space[7 * i + 4] = fAxInit; //vector parameter
   if (fFixAx == false) {
      working_space[shift + j] = fAxInit; //vector xk
      j += 1;
   }
   working_space[7 * i + 5] = fAyInit; //vector parameter
   if (fFixAy == false) {
      working_space[shift + j] = fAyInit; //vector xk
      j += 1;
   }
   working_space[7 * i + 6] = fTxyInit; //vector parameter
   if (fFixTxy == false) {
      working_space[shift + j] = fTxyInit; //vector xk
      j += 1;
   }
   working_space[7 * i + 7] = fSxyInit; //vector parameter
   if (fFixSxy == false) {
      working_space[shift + j] = fSxyInit; //vector xk
      j += 1;
   }
   working_space[7 * i + 8] = fTxInit; //vector parameter
   if (fFixTx == false) {
      working_space[shift + j] = fTxInit; //vector xk
      j += 1;
   }
   working_space[7 * i + 9] = fTyInit; //vector parameter
   if (fFixTy == false) {
      working_space[shift + j] = fTyInit; //vector xk
      j += 1;
   }
   working_space[7 * i + 10] = fSxyInit; //vector parameter
   if (fFixSx == false) {
      working_space[shift + j] = fSxInit; //vector xk
      j += 1;
   }
   working_space[7 * i + 11] = fSyInit; //vector parameter
   if (fFixSy == false) {
      working_space[shift + j] = fSyInit; //vector xk
      j += 1;
   }
   working_space[7 * i + 12] = fBxInit; //vector parameter
   if (fFixBx == false) {
      working_space[shift + j] = fBxInit; //vector xk
      j += 1;
   }
   working_space[7 * i + 13] = fByInit; //vector parameter
   if (fFixBy == false) {
      working_space[shift + j] = fByInit; //vector xk
      j += 1;
   }
   size = j;
   Double_t **working_matrix = new Double_t *[size];
   for (i = 0; i < size; i++)
      working_matrix[i] = new Double_t[size + 4];
   for (iter = 0; iter < fNumberIterations; iter++) {
      for (j = 0; j < size; j++) {
         working_space[3 * shift + j] = 0; //temp
         for (k = 0; k <= size; k++) {
            working_matrix[j][k] = 0;
         }
      }
      
      //filling working matrix
      alpha = fAlpha;
      chi_opt = 0;
      for (i1 = fXmin; i1 <= fXmax; i1++) {
         for (i2 = fYmin; i2 <= fYmax; i2++) {
            //calculation of gradient vector
            for (j = 0, k = 0; j < fNPeaks; j++) {
               if (fFixAmp[j] == false) {
                  working_space[2 * shift + k] =
                      Deramp2((Double_t) i1, (Double_t) i2,
                              working_space[7 * j + 1],
                              working_space[7 * j + 2],
                              working_space[peak_vel],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 2],
                              working_space[peak_vel + 6],
                              working_space[peak_vel + 7],
                              working_space[peak_vel + 12],
                              working_space[peak_vel + 13]);
                  k += 1;
               }
               if (fFixPositionX[j] == false) {
                  working_space[2 * shift + k] =
                      Deri02((Double_t) i1, (Double_t) i2,
                             working_space[7 * j],
                             working_space[7 * j + 1],
                             working_space[7 * j + 2],
                             working_space[peak_vel],
                             working_space[peak_vel + 1],
                             working_space[peak_vel + 2],
                             working_space[peak_vel + 6],
                             working_space[peak_vel + 7],
                             working_space[peak_vel + 12],
                             working_space[peak_vel + 13]);
                  k += 1;
               }
               if (fFixPositionY[j] == false) {
                  working_space[2 * shift + k] =
                      Derj02((Double_t) i1, (Double_t) i2,
                             working_space[7 * j],
                             working_space[7 * j + 1],
                             working_space[7 * j + 2],
                             working_space[peak_vel],
                             working_space[peak_vel + 1],
                             working_space[peak_vel + 2],
                             working_space[peak_vel + 6],
                             working_space[peak_vel + 7],
                             working_space[peak_vel + 12],
                             working_space[peak_vel + 13]);
                  k += 1;
               }
               if (fFixAmpX1[j] == false) {
                  working_space[2 * shift + k] =
                      Derampx((Double_t) i1, working_space[7 * j + 5],
                              working_space[peak_vel],
                              working_space[peak_vel + 8],
                              working_space[peak_vel + 10],
                              working_space[peak_vel + 12]);
                  k += 1;
               }
               if (fFixAmpY1[j] == false) {
                  working_space[2 * shift + k] =
                      Derampx((Double_t) i2, working_space[7 * j + 6],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 9],
                              working_space[peak_vel + 11],
                              working_space[peak_vel + 13]);
                  k += 1;
               }
               if (fFixPositionX1[j] == false) {
                  working_space[2 * shift + k] =
                      Deri01((Double_t) i1, working_space[7 * j + 3],
                             working_space[7 * j + 5],
                             working_space[peak_vel],
                             working_space[peak_vel + 8],
                             working_space[peak_vel + 10],
                             working_space[peak_vel + 12]);
                  k += 1;
               }
               if (fFixPositionY1[j] == false) {
                  working_space[2 * shift + k] =
                      Deri01((Double_t) i2, working_space[7 * j + 4],
                             working_space[7 * j + 6],
                             working_space[peak_vel + 1],
                             working_space[peak_vel + 9],
                             working_space[peak_vel + 11],
                             working_space[peak_vel + 13]);
                  k += 1;
               }
            } if (fFixSigmaX == false) {
               working_space[2 * shift + k] =
                   Dersigmax(fNPeaks, (Double_t) i1, (Double_t) i2,
                             working_space, working_space[peak_vel],
                             working_space[peak_vel + 1],
                             working_space[peak_vel + 2],
                             working_space[peak_vel + 6],
                             working_space[peak_vel + 7],
                             working_space[peak_vel + 8],
                             working_space[peak_vel + 10],
                             working_space[peak_vel + 12],
                             working_space[peak_vel + 13]);
               k += 1;
            }
            if (fFixSigmaY == false) {
               working_space[2 * shift + k] =
                   Dersigmay(fNPeaks, (Double_t) i1, (Double_t) i2,
                             working_space, working_space[peak_vel],
                             working_space[peak_vel + 1],
                             working_space[peak_vel + 2],
                             working_space[peak_vel + 6],
                             working_space[peak_vel + 7],
                             working_space[peak_vel + 9],
                             working_space[peak_vel + 11],
                             working_space[peak_vel + 12],
                             working_space[peak_vel + 13]);
               k += 1;
            }
            if (fFixRo == false) {
               working_space[2 * shift + k] =
                   Derro(fNPeaks, (Double_t) i1, (Double_t) i2,
                         working_space, working_space[peak_vel],
                         working_space[peak_vel + 1],
                         working_space[peak_vel + 2]);
               k += 1;
            }
            if (fFixA0 == false) {
               working_space[2 * shift + k] = 1.;
               k += 1;
            }
            if (fFixAx == false) {
               working_space[2 * shift + k] = i1;
               k += 1;
            }
            if (fFixAy == false) {
               working_space[2 * shift + k] = i2;
               k += 1;
            }
            if (fFixTxy == false) {
               working_space[2 * shift + k] =
                   Dertxy(fNPeaks, (Double_t) i1, (Double_t) i2,
                          working_space, working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 12],
                          working_space[peak_vel + 13]);
               k += 1;
            }
            if (fFixSxy == false) {
               working_space[2 * shift + k] =
                   Dersxy(fNPeaks, (Double_t) i1, (Double_t) i2,
                          working_space, working_space[peak_vel],
                          working_space[peak_vel + 1]);
               k += 1;
            }
            if (fFixTx == false) {
               working_space[2 * shift + k] =
                   Dertx(fNPeaks, (Double_t) i1, working_space,
                         working_space[peak_vel],
                         working_space[peak_vel + 12]);
               k += 1;
            }
            if (fFixTy == false) {
               working_space[2 * shift + k] =
                   Derty(fNPeaks, (Double_t) i2, working_space,
                         working_space[peak_vel + 1],
                         working_space[peak_vel + 13]);
               k += 1;
            }
            if (fFixSx == false) {
               working_space[2 * shift + k] =
                   Dersx(fNPeaks, (Double_t) i1, working_space,
                         working_space[peak_vel]);
               k += 1;
            }
            if (fFixSy == false) {
               working_space[2 * shift + k] =
                   Dersy(fNPeaks, (Double_t) i2, working_space,
                         working_space[peak_vel + 1]);
               k += 1;
            }
            if (fFixBx == false) {
               working_space[2 * shift + k] =
                   Derbx(fNPeaks, (Double_t) i1, (Double_t) i2,
                         working_space, working_space[peak_vel],
                         working_space[peak_vel + 1],
                         working_space[peak_vel + 6],
                         working_space[peak_vel + 8],
                         working_space[peak_vel + 12],
                         working_space[peak_vel + 13]);
               k += 1;
            }
            if (fFixBy == false) {
               working_space[2 * shift + k] =
                   Derby(fNPeaks, (Double_t) i1, (Double_t) i2,
                         working_space, working_space[peak_vel],
                         working_space[peak_vel + 1],
                         working_space[peak_vel + 6],
                         working_space[peak_vel + 8],
                         working_space[peak_vel + 12],
                         working_space[peak_vel + 13]);
               k += 1;
            }
            yw = source[i1][i2];
            ywm = yw;
            f = Shape2(fNPeaks, (Double_t) i1, (Double_t) i2,
                        working_space, working_space[peak_vel],
                        working_space[peak_vel + 1],
                        working_space[peak_vel + 2],
                        working_space[peak_vel + 3],
                        working_space[peak_vel + 4],
                        working_space[peak_vel + 5],
                        working_space[peak_vel + 6],
                        working_space[peak_vel + 7],
                        working_space[peak_vel + 8],
                        working_space[peak_vel + 9],
                        working_space[peak_vel + 10],
                        working_space[peak_vel + 11],
                        working_space[peak_vel + 12],
                        working_space[peak_vel + 13]);
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
            for (j = 0; j < size; j++) {
               for (k = 0; k < size; k++) {
                  b = working_space[2 * shift +
                                     j] * working_space[2 * shift +
                                                        k] / ywm;
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
            for (j = 0; j < size; j++) {
               working_matrix[j][size] -=
                   b * working_space[2 * shift + j];
            }
         }
      }
      for (i = 0; i < size; i++) {
         working_matrix[i][size + 1] = 0; //xk
      }
      StiefelInversion(working_matrix, size);
      for (i = 0; i < size; i++) {
         working_space[2 * shift + i] = working_matrix[i][size + 1]; //der
      }
      
      //calculate chi_opt
      chi2 = chi_opt;
      chi_opt = TMath::Sqrt(TMath::Abs(chi_opt));
      
      //calculate new parameters
      regul_cycle = 0;
      for (j = 0; j < size; j++) {
         working_space[4 * shift + j] = working_space[shift + j]; //temp_xk[j]=xk[j]
      }
      
      do {
         if (fAlphaOptim == kFitAlphaOptimal) {
            if (fStatisticType != kFitOptimMaxLikelihood)
               chi_min = 10000 * chi2;
            
            else
               chi_min = 0.1 * chi2;
            flag = 0;
            for (pi = 0.1; flag == 0 && pi <= 100; pi += 0.1) {
               for (j = 0; j < size; j++) {
                  working_space[shift + j] = working_space[4 * shift + j] + pi * alpha * working_space[2 * shift + j]; //xk[j]=temp_xk[j]+pi*alpha*der[j]
               }
               for (i = 0, j = 0; i < fNPeaks; i++) {
                  if (fFixAmp[i] == false) {
                     if (working_space[shift + j] < 0) //xk[j]
                        working_space[shift + j] = 0; //xk[j]
                     working_space[7 * i] = working_space[shift + j]; //parameter[7*i]=xk[j]
                     j += 1;
                  }
                  if (fFixPositionX[i] == false) {
                     if (working_space[shift + j] < fXmin) //xk[j]
                        working_space[shift + j] = fXmin; //xk[j]
                     if (working_space[shift + j] > fXmax) //xk[j]
                        working_space[shift + j] = fXmax; //xk[j]
                     working_space[7 * i + 1] = working_space[shift + j]; //parameter[7*i+1]=xk[j]
                     j += 1;
                  }
                  if (fFixPositionY[i] == false) {
                     if (working_space[shift + j] < fYmin) //xk[j]
                        working_space[shift + j] = fYmin; //xk[j]
                     if (working_space[shift + j] > fYmax) //xk[j]
                        working_space[shift + j] = fYmax; //xk[j]
                     working_space[7 * i + 2] = working_space[shift + j]; //parameter[7*i+2]=xk[j]
                     j += 1;
                  }
                  if (fFixAmpX1[i] == false) {
                     if (working_space[shift + j] < 0) //xk[j]
                        working_space[shift + j] = 0; //xk[j]
                     working_space[7 * i + 3] = working_space[shift + j]; //parameter[7*i+3]=xk[j]
                     j += 1;
                  }
                  if (fFixAmpY1[i] == false) {
                     if (working_space[shift + j] < 0) //xk[j]
                        working_space[shift + j] = 0; //xk[j]
                     working_space[7 * i + 4] = working_space[shift + j]; //parameter[7*i+4]=xk[j]
                     j += 1;
                  }
                  if (fFixPositionX1[i] == false) {
                     if (working_space[shift + j] < fXmin) //xk[j]
                        working_space[shift + j] = fXmin; //xk[j]
                     if (working_space[shift + j] > fXmax) //xk[j]
                        working_space[shift + j] = fXmax; //xk[j]
                     working_space[7 * i + 5] = working_space[shift + j]; //parameter[7*i+5]=xk[j]
                     j += 1;
                  }
                  if (fFixPositionY1[i] == false) {
                     if (working_space[shift + j] < fYmin) //xk[j]
                        working_space[shift + j] = fYmin; //xk[j]
                     if (working_space[shift + j] > fYmax) //xk[j]
                        working_space[shift + j] = fYmax; //xk[j]
                     working_space[7 * i + 6] = working_space[shift + j]; //parameter[7*i+6]=xk[j]
                     j += 1;
                  }
               }
               if (fFixSigmaX == false) {
                  if (working_space[shift + j] < 0.001) { //xk[j]
                     working_space[shift + j] = 0.001; //xk[j]
                  }
                  working_space[peak_vel] = working_space[shift + j]; //parameter[peak_vel]=xk[j]
                  j += 1;
               }
               if (fFixSigmaY == false) {
                  if (working_space[shift + j] < 0.001) { //xk[j]
                     working_space[shift + j] = 0.001; //xk[j]
                  }
                  working_space[peak_vel + 1] = working_space[shift + j]; //parameter[peak_vel+1]=xk[j]
                  j += 1;
               }
               if (fFixRo == false) {
                  if (working_space[shift + j] < -1) { //xk[j]
                     working_space[shift + j] = -1; //xk[j]
                  }
                  if (working_space[shift + j] > 1) { //xk[j]
                     working_space[shift + j] = 1; //xk[j]
                  }
                  working_space[peak_vel + 2] = working_space[shift + j]; //parameter[peak_vel+2]=xk[j]
                  j += 1;
               }
               if (fFixA0 == false) {
                  working_space[peak_vel + 3] = working_space[shift + j]; //parameter[peak_vel+3]=xk[j]
                  j += 1;
               }
               if (fFixAx == false) {
                  working_space[peak_vel + 4] = working_space[shift + j]; //parameter[peak_vel+4]=xk[j]
                  j += 1;
               }
               if (fFixAy == false) {
                  working_space[peak_vel + 5] = working_space[shift + j]; //parameter[peak_vel+5]=xk[j]
                  j += 1;
               }
               if (fFixTxy == false) {
                  working_space[peak_vel + 6] = working_space[shift + j]; //parameter[peak_vel+6]=xk[j]
                  j += 1;
               }
               if (fFixSxy == false) {
                  working_space[peak_vel + 7] = working_space[shift + j]; //parameter[peak_vel+7]=xk[j]
                  j += 1;
               }
               if (fFixTx == false) {
                  working_space[peak_vel + 8] = working_space[shift + j]; //parameter[peak_vel+8]=xk[j]
                  j += 1;
               }
               if (fFixTy == false) {
                  working_space[peak_vel + 9] = working_space[shift + j]; //parameter[peak_vel+9]=xk[j]
                  j += 1;
               }
               if (fFixSx == false) {
                  working_space[peak_vel + 10] = working_space[shift + j]; //parameter[peak_vel+10]=xk[j]
                  j += 1;
               }
               if (fFixSy == false) {
                  working_space[peak_vel + 11] = working_space[shift + j]; //parameter[peak_vel+11]=xk[j]
                  j += 1;
               }
               if (fFixBx == false) {
                  if (TMath::Abs(working_space[shift + j]) < 0.001) { //xk[j]
                     if (working_space[shift + j] < 0) //xk[j]
                        working_space[shift + j] = -0.001; //xk[j]
                     else
                        working_space[shift + j] = 0.001; //xk[j]
                  }
                  working_space[peak_vel + 12] = working_space[shift + j]; //parameter[peak_vel+12]=xk[j]
                  j += 1;
               }
               if (fFixBy == false) {
                  if (TMath::Abs(working_space[shift + j]) < 0.001) { //xk[j]
                     if (working_space[shift + j] < 0) //xk[j]
                        working_space[shift + j] = -0.001; //xk[j]
                     else
                        working_space[shift + j] = 0.001; //xk[j]
                  }
                  working_space[peak_vel + 13] = working_space[shift + j]; //parameter[peak_vel+13]=xk[j]
                  j += 1;
               }
               chi2 = 0;
               for (i1 = fXmin; i1 <= fXmax; i1++) {
                  for (i2 = fYmin; i2 <= fYmax; i2++) {
                     yw = source[i1][i2];
                     ywm = yw;
                     f = Shape2(fNPeaks, (Double_t) i1,
                                 (Double_t) i2, working_space,
                                 working_space[peak_vel],
                                 working_space[peak_vel + 1],
                                 working_space[peak_vel + 2],
                                 working_space[peak_vel + 3],
                                 working_space[peak_vel + 4],
                                 working_space[peak_vel + 5],
                                 working_space[peak_vel + 6],
                                 working_space[peak_vel + 7],
                                 working_space[peak_vel + 8],
                                 working_space[peak_vel + 9],
                                 working_space[peak_vel + 10],
                                 working_space[peak_vel + 11],
                                 working_space[peak_vel + 12],
                                 working_space[peak_vel + 13]);
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
               for (j = 0; j < size; j++) {
                  working_space[shift + j] = working_space[4 * shift + j] + pmin * alpha * working_space[2 * shift + j]; //xk[j]=temp_xk[j]+pmin*alpha*der[j]
               }
               for (i = 0, j = 0; i < fNPeaks; i++) {
                  if (fFixAmp[i] == false) {
                     if (working_space[shift + j] < 0) //xk[j]
                        working_space[shift + j] = 0; //xk[j]
                     working_space[7 * i] = working_space[shift + j]; //parameter[7*i]=xk[j]
                     j += 1;
                  }
                  if (fFixPositionX[i] == false) {
                     if (working_space[shift + j] < fXmin) //xk[j]
                        working_space[shift + j] = fXmin; //xk[j]
                     if (working_space[shift + j] > fXmax) //xk[j]
                        working_space[shift + j] = fXmax; //xk[j]
                     working_space[7 * i + 1] = working_space[shift + j]; //parameter[7*i+1]=xk[j]
                     j += 1;
                  }
                  if (fFixPositionY[i] == false) {
                     if (working_space[shift + j] < fYmin) //xk[j]
                        working_space[shift + j] = fYmin; //xk[j]
                     if (working_space[shift + j] > fYmax) //xk[j]
                        working_space[shift + j] = fYmax; //xk[j]
                     working_space[7 * i + 2] = working_space[shift + j]; //parameter[7*i+2]=xk[j]
                     j += 1;
                  }
                  if (fFixAmpX1[i] == false) {
                     if (working_space[shift + j] < 0) //xk[j]
                        working_space[shift + j] = 0; //xk[j]
                     working_space[7 * i + 3] = working_space[shift + j]; //parameter[7*i+3]=xk[j]
                     j += 1;
                  }
                  if (fFixAmpY1[i] == false) {
                     if (working_space[shift + j] < 0) //xk[j]
                        working_space[shift + j] = 0; //xk[j]
                     working_space[7 * i + 4] = working_space[shift + j]; //parameter[7*i+4]=xk[j]
                     j += 1;
                  }
                  if (fFixPositionX1[i] == false) {
                     if (working_space[shift + j] < fXmin) //xk[j]
                        working_space[shift + j] = fXmin; //xk[j]
                     if (working_space[shift + j] > fXmax) //xk[j]
                        working_space[shift + j] = fXmax; //xk[j]
                     working_space[7 * i + 5] = working_space[shift + j]; //parameter[7*i+5]=xk[j]
                     j += 1;
                  }
                  if (fFixPositionY1[i] == false) {
                     if (working_space[shift + j] < fYmin) //xk[j]
                        working_space[shift + j] = fYmin; //xk[j]
                     if (working_space[shift + j] > fYmax) //xk[j]
                        working_space[shift + j] = fYmax; //xk[j]
                     working_space[7 * i + 6] = working_space[shift + j]; //parameter[7*i+6]=xk[j]
                     j += 1;
                  }
               }
               if (fFixSigmaX == false) {
                  if (working_space[shift + j] < 0.001) { //xk[j]
                     working_space[shift + j] = 0.001; //xk[j]
                  }
                  working_space[peak_vel] = working_space[shift + j]; //parameter[peak_vel]=xk[j]
                  j += 1;
               }
               if (fFixSigmaY == false) {
                  if (working_space[shift + j] < 0.001) { //xk[j]
                     working_space[shift + j] = 0.001; //xk[j]
                  }
                  working_space[peak_vel + 1] = working_space[shift + j]; //parameter[peak_vel+1]=xk[j]
                  j += 1;
               }
               if (fFixRo == false) {
                  if (working_space[shift + j] < -1) { //xk[j]
                     working_space[shift + j] = -1; //xk[j]
                  }
                  if (working_space[shift + j] > 1) { //xk[j]
                     working_space[shift + j] = 1; //xk[j]
                  }
                  working_space[peak_vel + 2] = working_space[shift + j]; //parameter[peak_vel+2]=xk[j]
                  j += 1;
               }
               if (fFixA0 == false) {
                  working_space[peak_vel + 3] = working_space[shift + j]; //parameter[peak_vel+3]=xk[j]
                  j += 1;
               }
               if (fFixAx == false) {
                  working_space[peak_vel + 4] = working_space[shift + j]; //parameter[peak_vel+4]=xk[j]
                  j += 1;
               }
               if (fFixAy == false) {
                  working_space[peak_vel + 5] = working_space[shift + j]; //parameter[peak_vel+5]=xk[j]
                  j += 1;
               }
               if (fFixTxy == false) {
                  working_space[peak_vel + 6] = working_space[shift + j]; //parameter[peak_vel+6]=xk[j]
                  j += 1;
               }
               if (fFixSxy == false) {
                  working_space[peak_vel + 7] = working_space[shift + j]; //parameter[peak_vel+7]=xk[j]
                  j += 1;
               }
               if (fFixTx == false) {
                  working_space[peak_vel + 8] = working_space[shift + j]; //parameter[peak_vel+8]=xk[j]
                  j += 1;
               }
               if (fFixTy == false) {
                  working_space[peak_vel + 9] = working_space[shift + j]; //parameter[peak_vel+9]=xk[j]
                  j += 1;
               }
               if (fFixSx == false) {
                  working_space[peak_vel + 10] = working_space[shift + j]; //parameter[peak_vel+10]=xk[j]
                  j += 1;
               }
               if (fFixSy == false) {
                  working_space[peak_vel + 11] = working_space[shift + j]; //parameter[peak_vel+11]=xk[j]
                  j += 1;
               }
               if (fFixBx == false) {
                  if (TMath::Abs(working_space[shift + j]) < 0.001) { //xk[j]
                     if (working_space[shift + j] < 0) //xk[j]
                        working_space[shift + j] = -0.001; //xk[j]
                     else
                        working_space[shift + j] = 0.001; //xk[j]
                  }
                  working_space[peak_vel + 12] = working_space[shift + j]; //parameter[peak_vel+12]=xk[j]
                  j += 1;
               }
               if (fFixBy == false) {
                  if (TMath::Abs(working_space[shift + j]) < 0.001) { //xk[j]
                     if (working_space[shift + j] < 0) //xk[j]
                        working_space[shift + j] = -0.001; //xk[j]
                     else
                        working_space[shift + j] = 0.001; //xk[j]
                  }
                  working_space[peak_vel + 13] = working_space[shift + j]; //parameter[peak_vel+13]=xk[j]
                  j += 1;
               }
               chi = chi_min;
            }
         }
         
         else {
            for (j = 0; j < size; j++) {
               working_space[shift + j] = working_space[4 * shift + j] + alpha * working_space[2 * shift + j]; //xk[j]=temp_xk[j]+pi*alpha*der[j]
            }
            for (i = 0, j = 0; i < fNPeaks; i++) {
               if (fFixAmp[i] == false) {
                  if (working_space[shift + j] < 0) //xk[j]
                     working_space[shift + j] = 0; //xk[j]
                  working_space[7 * i] = working_space[shift + j]; //parameter[7*i]=xk[j]
                  j += 1;
               }
               if (fFixPositionX[i] == false) {
                  if (working_space[shift + j] < fXmin) //xk[j]
                     working_space[shift + j] = fXmin; //xk[j]
                  if (working_space[shift + j] > fXmax) //xk[j]
                     working_space[shift + j] = fXmax; //xk[j]
                  working_space[7 * i + 1] = working_space[shift + j]; //parameter[7*i+1]=xk[j]
                  j += 1;
               }
               if (fFixPositionY[i] == false) {
                  if (working_space[shift + j] < fYmin) //xk[j]
                     working_space[shift + j] = fYmin; //xk[j]
                  if (working_space[shift + j] > fYmax) //xk[j]
                     working_space[shift + j] = fYmax; //xk[j]
                  working_space[7 * i + 2] = working_space[shift + j]; //parameter[7*i+2]=xk[j]
                  j += 1;
               }
               if (fFixAmpX1[i] == false) {
                  if (working_space[shift + j] < 0) //xk[j]
                     working_space[shift + j] = 0; //xk[j]
                  working_space[7 * i + 3] = working_space[shift + j]; //parameter[7*i+3]=xk[j]
                  j += 1;
               }
               if (fFixAmpY1[i] == false) {
                  if (working_space[shift + j] < 0) //xk[j]
                     working_space[shift + j] = 0; //xk[j]
                  working_space[7 * i + 4] = working_space[shift + j]; //parameter[7*i+4]=xk[j]
                  j += 1;
               }
               if (fFixPositionX1[i] == false) {
                  if (working_space[shift + j] < fXmin) //xk[j]
                     working_space[shift + j] = fXmin; //xk[j]
                  if (working_space[shift + j] > fXmax) //xk[j]
                     working_space[shift + j] = fXmax; //xk[j]
                  working_space[7 * i + 5] = working_space[shift + j]; //parameter[7*i+5]=xk[j]
                  j += 1;
               }
               if (fFixPositionY1[i] == false) {
                  if (working_space[shift + j] < fYmin) //xk[j]
                     working_space[shift + j] = fYmin; //xk[j]
                  if (working_space[shift + j] > fYmax) //xk[j]
                     working_space[shift + j] = fYmax; //xk[j]
                  working_space[7 * i + 6] = working_space[shift + j]; //parameter[7*i+6]=xk[j]
                  j += 1;
               }
            }
            if (fFixSigmaX == false) {
               if (working_space[shift + j] < 0.001) { //xk[j]
                  working_space[shift + j] = 0.001; //xk[j]
               }
               working_space[peak_vel] = working_space[shift + j]; //parameter[peak_vel]=xk[j]
               j += 1;
            }
            if (fFixSigmaY == false) {
               if (working_space[shift + j] < 0.001) { //xk[j]
                  working_space[shift + j] = 0.001; //xk[j]
               }
               working_space[peak_vel + 1] = working_space[shift + j]; //parameter[peak_vel+1]=xk[j]
               j += 1;
            }
            if (fFixRo == false) {
               if (working_space[shift + j] < -1) { //xk[j]
                  working_space[shift + j] = -1; //xk[j]
               }
               if (working_space[shift + j] > 1) { //xk[j]
                  working_space[shift + j] = 1; //xk[j]
               }
               working_space[peak_vel + 2] = working_space[shift + j]; //parameter[peak_vel+2]=xk[j]
               j += 1;
            }
            if (fFixA0 == false) {
               working_space[peak_vel + 3] = working_space[shift + j]; //parameter[peak_vel+3]=xk[j]
               j += 1;
            }
            if (fFixAx == false) {
               working_space[peak_vel + 4] = working_space[shift + j]; //parameter[peak_vel+4]=xk[j]
               j += 1;
            }
            if (fFixAy == false) {
               working_space[peak_vel + 5] = working_space[shift + j]; //parameter[peak_vel+5]=xk[j]
               j += 1;
            }
            if (fFixTxy == false) {
               working_space[peak_vel + 6] = working_space[shift + j]; //parameter[peak_vel+6]=xk[j]
               j += 1;
            }
            if (fFixSxy == false) {
               working_space[peak_vel + 7] = working_space[shift + j]; //parameter[peak_vel+7]=xk[j]
               j += 1;
            }
            if (fFixTx == false) {
               working_space[peak_vel + 8] = working_space[shift + j]; //parameter[peak_vel+8]=xk[j]
               j += 1;
            }
            if (fFixTy == false) {
               working_space[peak_vel + 9] = working_space[shift + j]; //parameter[peak_vel+9]=xk[j]
               j += 1;
            }
            if (fFixSx == false) {
               working_space[peak_vel + 10] = working_space[shift + j]; //parameter[peak_vel+10]=xk[j]
               j += 1;
            }
            if (fFixSy == false) {
               working_space[peak_vel + 11] = working_space[shift + j]; //parameter[peak_vel+11]=xk[j]
               j += 1;
            }
            if (fFixBx == false) {
               if (TMath::Abs(working_space[shift + j]) < 0.001) { //xk[j]
                  if (working_space[shift + j] < 0) //xk[j]
                     working_space[shift + j] = -0.001; //xk[j]
                  else
                     working_space[shift + j] = 0.001; //xk[j]
               }
               working_space[peak_vel + 12] = working_space[shift + j]; //parameter[peak_vel+12]=xk[j]
               j += 1;
            }
            if (fFixBy == false) {
               if (TMath::Abs(working_space[shift + j]) < 0.001) { //xk[j]
                  if (working_space[shift + j] < 0) //xk[j]
                     working_space[shift + j] = -0.001; //xk[j]
                  else
                     working_space[shift + j] = 0.001; //xk[j]
               }
               working_space[peak_vel + 13] = working_space[shift + j]; //parameter[peak_vel+13]=xk[j]
               j += 1;
            }
            chi = 0;
            for (i1 = fXmin; i1 <= fXmax; i1++) {
               for (i2 = fYmin; i2 <= fYmax; i2++) {
                  yw = source[i1][i2];
                  ywm = yw;
                  f = Shape2(fNPeaks, (Double_t) i1, (Double_t) i2,
                              working_space, working_space[peak_vel],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 2],
                              working_space[peak_vel + 3],
                              working_space[peak_vel + 4],
                              working_space[peak_vel + 5],
                              working_space[peak_vel + 6],
                              working_space[peak_vel + 7],
                              working_space[peak_vel + 8],
                              working_space[peak_vel + 9],
                              working_space[peak_vel + 10],
                              working_space[peak_vel + 11],
                              working_space[peak_vel + 12],
                              working_space[peak_vel + 13]);
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
      for (j = 0; j < size; j++) {
         working_space[4 * shift + j] = 0; //temp_xk[j]
         working_space[2 * shift + j] = 0; //der[j]
      }
      for (i1 = fXmin, chi_cel = 0; i1 <= fXmax; i1++) {
         for (i2 = fYmin; i2 <= fYmax; i2++) {
            yw = source[i1][i2];
            if (yw == 0)
               yw = 1;
            f = Shape2(fNPeaks, (Double_t) i1, (Double_t) i2,
                        working_space, working_space[peak_vel],
                        working_space[peak_vel + 1],
                        working_space[peak_vel + 2],
                        working_space[peak_vel + 3],
                        working_space[peak_vel + 4],
                        working_space[peak_vel + 5],
                        working_space[peak_vel + 6],
                        working_space[peak_vel + 7],
                        working_space[peak_vel + 8],
                        working_space[peak_vel + 9],
                        working_space[peak_vel + 10],
                        working_space[peak_vel + 11],
                        working_space[peak_vel + 12],
                        working_space[peak_vel + 13]);
            chi_opt = (yw - f) * (yw - f) / yw;
            chi_cel += (yw - f) * (yw - f) / yw;
            
                //calculate gradient vector
                for (j = 0, k = 0; j < fNPeaks; j++) {
               if (fFixAmp[j] == false) {
                  a = Deramp2((Double_t) i1, (Double_t) i2,
                               working_space[7 * j + 1],
                               working_space[7 * j + 2],
                               working_space[peak_vel],
                               working_space[peak_vel + 1],
                               working_space[peak_vel + 2],
                               working_space[peak_vel + 6],
                               working_space[peak_vel + 7],
                               working_space[peak_vel + 12],
                               working_space[peak_vel + 13]);
                  if (yw != 0) {
                     working_space[2 * shift + k] += chi_opt; //der[k]
                     b = a * a / yw;
                     working_space[4 * shift + k] += b; //temp_xk[k]
                  }
                  k += 1;
               }
               if (fFixPositionX[j] == false) {
                  a = Deri02((Double_t) i1, (Double_t) i2,
                              working_space[7 * j],
                              working_space[7 * j + 1],
                              working_space[7 * j + 2],
                              working_space[peak_vel],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 2],
                              working_space[peak_vel + 6],
                              working_space[peak_vel + 7],
                              working_space[peak_vel + 12],
                              working_space[peak_vel + 13]);
                  if (yw != 0) {
                     working_space[2 * shift + k] += chi_opt; //der[k]
                     b = a * a / yw;
                     working_space[4 * shift + k] += b; //temp_xk[k]
                  }
                  k += 1;
               }
               if (fFixPositionY[j] == false) {
                  a = Derj02((Double_t) i1, (Double_t) i2,
                              working_space[7 * j],
                              working_space[7 * j + 1],
                              working_space[7 * j + 2],
                              working_space[peak_vel],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 2],
                              working_space[peak_vel + 6],
                              working_space[peak_vel + 7],
                              working_space[peak_vel + 12],
                              working_space[peak_vel + 13]);
                  if (yw != 0) {
                     working_space[2 * shift + k] += chi_opt; //der[k]
                     b = a * a / yw;
                     working_space[4 * shift + k] += b; //temp_xk[k]
                  }
                  k += 1;
               }
               if (fFixAmpX1[j] == false) {
                  a = Derampx((Double_t) i1, working_space[7 * j + 5],
                               working_space[peak_vel],
                               working_space[peak_vel + 8],
                               working_space[peak_vel + 10],
                               working_space[peak_vel + 12]);
                  if (yw != 0) {
                     working_space[2 * shift + k] += chi_opt; //der[k]
                     b = a * a / yw;
                     working_space[4 * shift + k] += b; //temp_xk[k]
                  }
                  k += 1;
               }
               if (fFixAmpY1[j] == false) {
                  a = Derampx((Double_t) i2, working_space[7 * j + 6],
                               working_space[peak_vel + 1],
                               working_space[peak_vel + 9],
                               working_space[peak_vel + 11],
                               working_space[peak_vel + 13]);
                  if (yw != 0) {
                     working_space[2 * shift + k] += chi_opt; //der[k]
                     b = a * a / yw;
                     working_space[4 * shift + k] += b; //temp_xk[k]
                  }
                  k += 1;
               }
               if (fFixPositionX1[j] == false) {
                  a = Deri01((Double_t) i1, working_space[7 * j + 3],
                              working_space[7 * j + 5],
                              working_space[peak_vel],
                              working_space[peak_vel + 8],
                              working_space[peak_vel + 10],
                              working_space[peak_vel + 12]);
                  if (yw != 0) {
                     working_space[2 * shift + k] += chi_opt; //der[k]
                     b = a * a / yw;
                     working_space[4 * shift + k] += b; //temp_xk[k]
                  }
                  k += 1;
               }
               if (fFixPositionY1[j] == false) {
                  a = Deri01((Double_t) i2, working_space[7 * j + 4],
                              working_space[7 * j + 6],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 9],
                              working_space[peak_vel + 11],
                              working_space[peak_vel + 13]);
                  if (yw != 0) {
                     working_space[2 * shift + k] += chi_opt; //der[k]
                     b = a * a / yw;
                     working_space[4 * shift + k] += b; //temp_xk[k]
                  }
                  k += 1;
               }
            }
            if (fFixSigmaX == false) {
               a = Dersigmax(fNPeaks, (Double_t) i1, (Double_t) i2,
                              working_space, working_space[peak_vel],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 2],
                              working_space[peak_vel + 6],
                              working_space[peak_vel + 7],
                              working_space[peak_vel + 8],
                              working_space[peak_vel + 10],
                              working_space[peak_vel + 12],
                              working_space[peak_vel + 13]);
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt; //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b; //temp_xk[k]
               }
               k += 1;
            }
            if (fFixSigmaY == false) {
               a = Dersigmay(fNPeaks, (Double_t) i1, (Double_t) i2,
                              working_space, working_space[peak_vel],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 2],
                              working_space[peak_vel + 6],
                              working_space[peak_vel + 7],
                              working_space[peak_vel + 9],
                              working_space[peak_vel + 11],
                              working_space[peak_vel + 12],
                              working_space[peak_vel + 13]);
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt; //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b; //temp_xk[k]
               }
               k += 1;
            }
            if (fFixRo == false) {
               a = Derro(fNPeaks, (Double_t) i1, (Double_t) i2,
                          working_space, working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 2]);
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt; //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b; //temp_xk[k]
               }
               k += 1;
            }
            if (fFixA0 == false) {
               a = 1.;
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt; //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b; //temp_xk[k]
               }
               k += 1;
            }
            if (fFixAx == false) {
               a = i1;
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt; //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b; //temp_xk[k]
               }
               k += 1;
            }
            if (fFixAy == false) {
               a = i2;
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt; //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b; //temp_xk[k]
               }
               k += 1;
            }
            if (fFixTxy == false) {
               a = Dertxy(fNPeaks, (Double_t) i1, (Double_t) i2,
                           working_space, working_space[peak_vel],
                           working_space[peak_vel + 1],
                           working_space[peak_vel + 12],
                           working_space[peak_vel + 13]);
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt; //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b; //temp_xk[k]
               }
               k += 1;
            }
            if (fFixSxy == false) {
               a = Dersxy(fNPeaks, (Double_t) i1, (Double_t) i2,
                           working_space, working_space[peak_vel],
                           working_space[peak_vel + 1]);
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt; //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b; //temp_xk[k]
               }
               k += 1;
            }
            if (fFixTx == false) {
               a = Dertx(fNPeaks, (Double_t) i1, working_space,
                          working_space[peak_vel],
                          working_space[peak_vel + 12]);
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt; //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b; //temp_xk[k]
               }
               k += 1;
            }
            if (fFixTy == false) {
               a = Derty(fNPeaks, (Double_t) i2, working_space,
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 13]);
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt; //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b; //temp_xk[k]
               }
               k += 1;
            }
            if (fFixSx == false) {
               a = Dersx(fNPeaks, (Double_t) i1, working_space,
                          working_space[peak_vel]);
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt; //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b; //temp_xk[k]
               }
               k += 1;
            }
            if (fFixSy == false) {
               a = Dersy(fNPeaks, (Double_t) i2, working_space,
                          working_space[peak_vel + 1]);
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt; //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b; //temp_xk[k]
               }
               k += 1;
            }
            if (fFixBx == false) {
               a = Derbx(fNPeaks, (Double_t) i1, (Double_t) i2,
                          working_space, working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 6],
                          working_space[peak_vel + 8],
                          working_space[peak_vel + 12],
                          working_space[peak_vel + 13]);
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt; //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b; //temp_xk[k]
               }
               k += 1;
            }
            if (fFixBy == false) {
               a = Derby(fNPeaks, (Double_t) i1, (Double_t) i2,
                          working_space, working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 6],
                          working_space[peak_vel + 8],
                          working_space[peak_vel + 12],
                          working_space[peak_vel + 13]);
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt; //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b; //temp_xk[k]
               }
               k += 1;
            }
         }
      }
   }
   b = (fXmax - fXmin + 1) * (fYmax - fYmin + 1) - size;
   chi_er = chi_cel / b;
   for (i = 0, j = 0; i < fNPeaks; i++) {
      fVolume[i] =
          Volume(working_space[7 * i], working_space[peak_vel],
                 working_space[peak_vel + 1], working_space[peak_vel + 2]);
      if (fVolume[i] > 0) {
         c = 0;
         if (fFixAmp[i] == false) {
            a = Derpa2(working_space[peak_vel],
                        working_space[peak_vel + 1],
                        working_space[peak_vel + 2]);
            b = working_space[4 * shift + j]; //temp_xk[j]
            if (b == 0)
               b = 1;
            
            else
               b = 1 / b;
            c = c + a * a * b;
         }
         if (fFixSigmaX == false) {
            a = Derpsigmax(working_space[shift + j],
                            working_space[peak_vel + 1],
                            working_space[peak_vel + 2]);
            b = working_space[4 * shift + peak_vel]; //temp_xk[j]
            if (b == 0)
               b = 1;
            
            else
               b = 1 / b;
            c = c + a * a * b;
         }
         if (fFixSigmaY == false) {
            a = Derpsigmay(working_space[shift + j],
                            working_space[peak_vel],
                            working_space[peak_vel + 2]);
            b = working_space[4 * shift + peak_vel + 1]; //temp_xk[j]
            if (b == 0)
               b = 1;
            
            else
               b = 1 / b;
            c = c + a * a * b;
         }
         if (fFixRo == false) {
            a = Derpro(working_space[shift + j], working_space[peak_vel],
                        working_space[peak_vel + 1],
                        working_space[peak_vel + 2]);
            b = working_space[4 * shift + peak_vel + 2]; //temp_xk[j]
            if (b == 0)
               b = 1;
            
            else
               b = 1 / b;
            c = c + a * a * b;
         }
         fVolumeErr[i] = TMath::Sqrt(TMath::Abs(chi_er * c));
      }
      
      else {
         fVolumeErr[i] = 0;
      }
      if (fFixAmp[i] == false) {
         fAmpCalc[i] = working_space[shift + j]; //xk[j]
         if (working_space[3 * shift + j] != 0)
            fAmpErr[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
         j += 1;
      }
      
      else {
         fAmpCalc[i] = fAmpInit[i];
         fAmpErr[i] = 0;
      }
      if (fFixPositionX[i] == false) {
         fPositionCalcX[i] = working_space[shift + j]; //xk[j]
         if (working_space[3 * shift + j] != 0)
            fPositionErrX[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
         j += 1;
      }
      
      else {
         fPositionCalcX[i] = fPositionInitX[i];
         fPositionErrX[i] = 0;
      }
      if (fFixPositionY[i] == false) {
         fPositionCalcY[i] = working_space[shift + j]; //xk[j]
         if (working_space[3 * shift + j] != 0)
            fPositionErrY[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
         j += 1;
      }
      
      else {
         fPositionCalcY[i] = fPositionInitY[i];
         fPositionErrY[i] = 0;
      }
      if (fFixAmpX1[i] == false) {
         fAmpCalcX1[i] = working_space[shift + j]; //xk[j]
         if (working_space[3 * shift + j] != 0)
            fAmpErrX1[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
         j += 1;
      }
      
      else {
         fAmpCalcX1[i] = fAmpInitX1[i];
         fAmpErrX1[i] = 0;
      }
      if (fFixAmpY1[i] == false) {
         fAmpCalcY1[i] = working_space[shift + j]; //xk[j]
         if (working_space[3 * shift + j] != 0)
            fAmpErrY1[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
         j += 1;
      }
      
      else {
         fAmpCalcY1[i] = fAmpInitY1[i];
         fAmpErrY1[i] = 0;
      }
      if (fFixPositionX1[i] == false) {
         fPositionCalcX1[i] = working_space[shift + j]; //xk[j]
         if (working_space[3 * shift + j] != 0)
            fPositionErrX1[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
         j += 1;
      }
      
      else {
         fPositionCalcX1[i] = fPositionInitX1[i];
         fPositionErrX1[i] = 0;
      }
      if (fFixPositionY1[i] == false) {
         fPositionCalcY1[i] = working_space[shift + j]; //xk[j]
         if (working_space[3 * shift + j] != 0)
            fPositionErrY1[i] = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
         j += 1;
      }
      
      else {
         fPositionCalcY1[i] = fPositionInitY1[i];
         fPositionErrY1[i] = 0;
      }
   }
   if (fFixSigmaX == false) {
      fSigmaCalcX = working_space[shift + j]; //xk[j]
      if (working_space[3 * shift + j] != 0) //temp[j]
         fSigmaErrX = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fSigmaCalcX = fSigmaInitX;
      fSigmaErrX = 0;
   }
   if (fFixSigmaY == false) {
      fSigmaCalcY = working_space[shift + j]; //xk[j]
      if (working_space[3 * shift + j] != 0) //temp[j]
         fSigmaErrY = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fSigmaCalcY = fSigmaInitY;
      fSigmaErrY = 0;
   }
   if (fFixRo == false) {
      fRoCalc = working_space[shift + j]; //xk[j]
      if (working_space[3 * shift + j] != 0) //temp[j]
         fRoErr = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fRoCalc = fRoInit;
      fRoErr = 0;
   }
   if (fFixA0 == false) {
      fA0Calc = working_space[shift + j]; //xk[j]
      if (working_space[3 * shift + j] != 0) //temp[j]
         fA0Err = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fA0Calc = fA0Init;
      fA0Err = 0;
   }
   if (fFixAx == false) {
      fAxCalc = working_space[shift + j]; //xk[j]
      if (working_space[3 * shift + j] != 0) //temp[j]
         fAxErr = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fAxCalc = fAxInit;
      fAxErr = 0;
   }
   if (fFixAy == false) {
      fAyCalc = working_space[shift + j]; //xk[j]
      if (working_space[3 * shift + j] != 0) //temp[j]
         fAyErr = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fAyCalc = fAyInit;
      fAyErr = 0;
   }
   if (fFixTxy == false) {
      fTxyCalc = working_space[shift + j]; //xk[j]
      if (working_space[3 * shift + j] != 0) //temp[j]
         fTxyErr = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fTxyCalc = fTxyInit;
      fTxyErr = 0;
   }
   if (fFixSxy == false) {
      fSxyCalc = working_space[shift + j]; //xk[j]
      if (working_space[3 * shift + j] != 0) //temp[j]
         fSxyErr = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fSxyCalc = fSxyInit;
      fSxyErr = 0;
   }
   if (fFixTx == false) {
      fTxCalc = working_space[shift + j]; //xk[j]
      if (working_space[3 * shift + j] != 0) //temp[j]
         fTxErr = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fTxCalc = fTxInit;
      fTxErr = 0;
   }
   if (fFixTy == false) {
      fTyCalc = working_space[shift + j]; //xk[j]
      if (working_space[3 * shift + j] != 0) //temp[j]
         fTyErr = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fTyCalc = fTyInit;
      fTyErr = 0;
   }
   if (fFixSx == false) {
      fSxCalc = working_space[shift + j]; //xk[j]
      if (working_space[3 * shift + j] != 0) //temp[j]
         fSxErr = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fSxCalc = fSxInit;
      fSxErr = 0;
   }
   if (fFixSy == false) {
      fSyCalc = working_space[shift + j]; //xk[j]
      if (working_space[3 * shift + j] != 0) //temp[j]
         fSyErr = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fSyCalc = fSyInit;
      fSyErr = 0;
   }
   if (fFixBx == false) {
      fBxCalc = working_space[shift + j]; //xk[j]
      if (working_space[3 * shift + j] != 0) //temp[j]
         fBxErr = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fBxCalc = fBxInit;
      fBxErr = 0;
   }
   if (fFixBy == false) {
      fByCalc = working_space[shift + j]; //xk[j]
      if (working_space[3 * shift + j] != 0) //temp[j]
         fByErr = TMath::Sqrt(TMath::Abs(working_space[2 * shift + j])) / TMath::Sqrt(TMath::Abs(working_space[3 * shift + j])); //der[j]/temp[j]
      j += 1;
   }
   
   else {
      fByCalc = fByInit;
      fByErr = 0;
   }
   b = (fXmax - fXmin + 1) * (fYmax - fYmin + 1) - size;
   fChi = chi_cel / b;
   for (i1 = fXmin; i1 <= fXmax; i1++) {
      for (i2 = fYmin; i2 <= fYmax; i2++) {
         f = Shape2(fNPeaks, (Double_t) i1, (Double_t) i2,
                     working_space, working_space[peak_vel],
                     working_space[peak_vel + 1],
                     working_space[peak_vel + 2],
                     working_space[peak_vel + 3],
                     working_space[peak_vel + 4],
                     working_space[peak_vel + 5],
                     working_space[peak_vel + 6],
                     working_space[peak_vel + 7],
                     working_space[peak_vel + 8],
                     working_space[peak_vel + 9],
                     working_space[peak_vel + 10],
                     working_space[peak_vel + 11],
                     working_space[peak_vel + 12],
                     working_space[peak_vel + 13]);
         source[i1][i2] = f;
   
      } 
   } 
   for (i = 0; i < size; i++) delete [] working_matrix[i];
   delete [] working_matrix;
   delete [] working_space;
   return;
}

void TSpectrum2Fit::SetFitParameters(Int_t xmin,Int_t xmax,Int_t ymin,Int_t ymax, Int_t numberIterations, Double_t alpha, Int_t statisticType, Int_t alphaOptim, Int_t power, Int_t fitTaylor)
{
//////////////////////////////////////////////////////////////////////////////
//   SETTER FUNCTION                                                      
//                                                     
//   This function sets the following fitting parameters:
//         -xmin, xmax, ymin, ymax - fitting region
//         -numberIterations - # of desired iterations in the fit
//         -alpha - convergence coefficient, it should be positive number and <=1, for details see references
//         -statisticType - type of statistics, possible values kFitOptimChiCounts (chi square statistics with counts as weighting coefficients), kFitOptimChiFuncValues (chi square statistics with function values as weighting coefficients),kFitOptimMaxLikelihood 
//         -alphaOptim - optimization of convergence algorithm, possible values kFitAlphaHalving, kFitAlphaOptimal
//         -power - possible values kFitPower2,4,6,8,10,12, for details see references. It applies only for Awmi fitting function.
//         -fitTaylor - order of Taylor expansion, possible values kFitTaylorOrderFirst, kFitTaylorOrderSecond. It applies only for Awmi fitting function.
//////////////////////////////////////////////////////////////////////////////   
   
   if(xmin<0 || xmax <= xmin || ymin<0 || ymax <= ymin){ 
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
   fXmin=xmin,fXmax=xmax,fYmin=ymin,fYmax=ymax,fNumberIterations=numberIterations,fAlpha=alpha,fStatisticType=statisticType,fAlphaOptim=alphaOptim,fPower=power,fFitTaylor=fitTaylor;
}

//______________________________________________________________________________
void TSpectrum2Fit::SetPeakParameters(Double_t sigmaX, Bool_t fixSigmaX, Double_t sigmaY, Bool_t fixSigmaY, Double_t ro, Bool_t fixRo, const Float_t *positionInitX, const Bool_t *fixPositionX, const Float_t *positionInitY, const Bool_t *fixPositionY, const Float_t *positionInitX1, const Bool_t *fixPositionX1, const Float_t *positionInitY1, const Bool_t *fixPositionY1, const Float_t *ampInit, const Bool_t *fixAmp, const Float_t *ampInitX1, const Bool_t *fixAmpX1, const Float_t *ampInitY1, const Bool_t *fixAmpY1)
{
//////////////////////////////////////////////////////////////////////////////
//   SETTER FUNCTION                                                      
//                                                     
//   This function sets the following fitting parameters of peaks:
//         -sigmaX - initial value of sigma x parameter
//         -fixSigmaX - logical value of sigma x parameter, which allows to fix the parameter (not to fit)   
//         -sigmaY - initial value of sigma y parameter
//         -fixSigmaY - logical value of sigma y parameter, which allows to fix the parameter (not to fit) 
//         -ro - initial value of ro parameter (correlation coefficient)
//         -fixRo - logical value of ro parameter, which allows to fix the parameter (not to fit)     
//         -positionInitX - aray of initial values of peaks x positions
//         -fixPositionX - array of logical values which allow to fix appropriate x positions (not fit). However they are present in the estimated functional.
//         -positionInitY - aray of initial values of peaks y positions
//         -fixPositionY - array of logical values which allow to fix appropriate y positions (not fit). However they are present in the estimated functional.
//         -ampInit - aray of initial values of  2D peaks amplitudes
//         -fixAmp - aray of logical values which allow to fix appropriate amplitudes of 2D peaks (not fit). However they are present in the estimated functional      
//         -ampInitX1 - aray of initial values of amplitudes of  1D ridges in x direction
//         -fixAmpX1 - aray of logical values which allow to fix appropriate amplitudes of 1D ridges in x direction (not fit). However they are present in the estimated functional      
//         -ampInitY1 - aray of initial values of amplitudes of  1D ridges in y direction
//         -fixAmpY1 - aray of logical values which allow to fix appropriate amplitudes of 1D ridges in y direction (not fit). However they are present in the estimated functional      
//////////////////////////////////////////////////////////////////////////////   
   
   if (sigmaX <= 0 || sigmaY <= 0){
      Error ("SetPeakParameters","Invalid sigma, must be > than 0");
      return;
   }
   if (ro < -1 || ro > 1){
      Error ("SetPeakParameters","Invalid ro, must be from region <-1,1>");
      return;
   }    
   Int_t i;
   for(i=0; i < fNPeaks; i++){
      if(positionInitX[i] < fXmin || positionInitX[i] > fXmax){
         Error ("SetPeakParameters","Invalid peak position, must be in the range fXmin, fXmax");
         return;
      }         
      if(positionInitY[i] < fYmin || positionInitY[i] > fYmax){
         Error ("SetPeakParameters","Invalid peak position, must be in the range fYmin, fYmax");
         return;
      }               
      if(positionInitX1[i] < fXmin || positionInitX1[i] > fXmax){
         Error ("SetPeakParameters","Invalid ridge position, must be in the range fXmin, fXmax");
         return;
      }         
      if(positionInitY1[i] < fYmin || positionInitY1[i] > fYmax){
         Error ("SetPeakParameters","Invalid ridge position, must be in the range fYmin, fYmax");
         return;
      }                     
      if(ampInit[i] < 0){
         Error ("SetPeakParameters","Invalid peak amplitude, must be > than 0");        
         return;
      }         
      if(ampInitX1[i] < 0){
         Error ("SetPeakParameters","Invalid x ridge amplitude, must be > than 0");        
         return;
      }               
      if(ampInitY1[i] < 0){
         Error ("SetPeakParameters","Invalid y ridge amplitude, must be > than 0");        
         return;
      }               
   }
   fSigmaInitX = sigmaX, fFixSigmaX = fixSigmaX, fSigmaInitY = sigmaY, fFixSigmaY = fixSigmaY, fRoInit = ro, fFixRo = fixRo;
   for(i=0; i < fNPeaks; i++){
      fPositionInitX[i] = (Double_t) positionInitX[i];
      fFixPositionX[i] = fixPositionX[i];
      fPositionInitY[i] = (Double_t) positionInitY[i];
      fFixPositionY[i] = fixPositionY[i];      
      fPositionInitX1[i] = (Double_t) positionInitX1[i];
      fFixPositionX1[i] = fixPositionX1[i];
      fPositionInitY1[i] = (Double_t) positionInitY1[i];
      fFixPositionY1[i] = fixPositionY1[i];            
      fAmpInit[i] = (Double_t) ampInit[i];
      fFixAmp[i] = fixAmp[i]; 
      fAmpInitX1[i] = (Double_t) ampInitX1[i];
      fFixAmpX1[i] = fixAmpX1[i];       
      fAmpInitY1[i] = (Double_t) ampInitY1[i];
      fFixAmpY1[i] = fixAmpY1[i];             
   }
}

//______________________________________________________________________________
void TSpectrum2Fit::SetBackgroundParameters(Double_t a0Init, Bool_t fixA0, Double_t axInit, Bool_t fixAx, Double_t ayInit, Bool_t fixAy)
{
//////////////////////////////////////////////////////////////////////////////
//   SETTER FUNCTION                                                      
//                                                     
//   This function sets the following fitting parameters of background:
//         -a0Init - initial value of a0 parameter (backgroud is estimated as a0+ax*x+ay*y)
//         -fixA0 - logical value of a0 parameter, which allows to fix the parameter (not to fit)  
//         -axInit - initial value of ax parameter
//         -fixAx - logical value of ax parameter, which allows to fix the parameter (not to fit)   
//         -ayInit - initial value of ay parameter
//         -fixAy - logical value of ay parameter, which allows to fix the parameter (not to fit)     
//////////////////////////////////////////////////////////////////////////////   
   
   fA0Init = a0Init;
   fFixA0 = fixA0;
   fAxInit = axInit;
   fFixAx = fixAx;
   fAyInit = ayInit;
   fFixAy = fixAy;    
}

//______________________________________________________________________________
void TSpectrum2Fit::SetTailParameters(Double_t tInitXY, Bool_t fixTxy, Double_t tInitX, Bool_t fixTx, Double_t tInitY, Bool_t fixTy, Double_t bInitX, Bool_t fixBx, Double_t bInitY, Bool_t fixBy, Double_t sInitXY, Bool_t fixSxy, Double_t sInitX, Bool_t fixSx, Double_t sInitY, Bool_t fixSy)
{
//////////////////////////////////////////////////////////////////////////////
//   SETTER FUNCTION                                                      
//                                                     
//   This function sets the following fitting parameters of tails of peaks
//         -tInitXY - initial value of txy parameter
//         -fixTxy - logical value of txy parameter, which allows to fix the parameter (not to fit)  
//         -tInitX - initial value of tx parameter
//         -fixTx - logical value of tx parameter, which allows to fix the parameter (not to fit)
//         -tInitY - initial value of ty parameter
//         -fixTy - logical value of ty parameter, which allows to fix the parameter (not to fit)    
//         -bInitX - initial value of bx parameter
//         -fixBx - logical value of bx parameter, which allows to fix the parameter (not to fit) 
//         -bInitY - initial value of by parameter
//         -fixBy - logical value of by parameter, which allows to fix the parameter (not to fit)   
//         -sInitXY - initial value of sxy parameter
//         -fixSxy - logical value of sxy parameter, which allows to fix the parameter (not to fit) 
//         -sInitX - initial value of sx parameter
//         -fixSx - logical value of sx parameter, which allows to fix the parameter (not to fit)
//         -sInitY - initial value of sy parameter
//         -fixSy - logical value of sy parameter, which allows to fix the parameter (not to fit)     
//////////////////////////////////////////////////////////////////////////////      
   fTxyInit = tInitXY;
   fFixTxy = fixTxy;
   fTxInit = tInitX;
   fFixTx = fixTx;
   fTyInit = tInitY;
   fFixTy = fixTy;    
   fBxInit = bInitX;
   fFixBx = fixBx;
   fByInit = bInitY;
   fFixBy = fixBy;  
   fSxyInit = sInitXY;
   fFixSxy = fixSxy;     
   fSxInit = sInitX;
   fFixSx = fixSx;
   fSyInit = sInitY;
   fFixSy = fixSy;              
}

//______________________________________________________________________________
void TSpectrum2Fit::GetPositions(Float_t *positionsX, Float_t *positionsY, Float_t *positionsX1, Float_t *positionsY1)
{
//////////////////////////////////////////////////////////////////////////////
//   GETTER FUNCTION                                                      
//                                                     
//   This function gets the positions of fitted 2D peaks and 1D ridges
//         -positionX - gets vector of x positions of 2D peaks
//         -positionY - gets vector of y positions of 2D peaks
//         -positionX1 - gets vector of x positions of 1D ridges
//         -positionY1 - gets vector of y positions of 1D ridges
//////////////////////////////////////////////////////////////////////////////      
   for( Int_t i=0; i < fNPeaks; i++){
      positionsX[i]  = (Float_t) fPositionCalcX[i];
      positionsY[i]  = (Float_t) fPositionCalcY[i];      
      positionsX1[i] = (Float_t) fPositionCalcX1[i];
      positionsY1[i] = (Float_t) fPositionCalcY1[i];            
   }
}

//______________________________________________________________________________
void TSpectrum2Fit::GetPositionErrors(Float_t *positionErrorsX, Float_t *positionErrorsY, Float_t *positionErrorsX1, Float_t *positionErrorsY1)
{
//////////////////////////////////////////////////////////////////////////////
//   GETTER FUNCTION                                                      
//                                                     
//   This function gets the errors of positions of fitted 2D peaks and 1D ridges
//         -positionErrorsX - gets vector of errors of x positions of 2D peaks
//         -positionErrorsY - gets vector of errors of y positions of 2D peaks
//         -positionErrorsX1 - gets vector of errors of x positions of 1D ridges
//         -positionErrorsY1 - gets vector of errors of y positions of 1D ridges
//////////////////////////////////////////////////////////////////////////////      
   
   for( Int_t i=0; i < fNPeaks; i++){
      positionErrorsX[i] = (Float_t) fPositionErrX[i];
      positionErrorsY[i] = (Float_t) fPositionErrY[i];      
      positionErrorsX1[i] = (Float_t) fPositionErrX1[i];      
      positionErrorsY1[i] = (Float_t) fPositionErrY1[i];      
   }
}

//______________________________________________________________________________
void TSpectrum2Fit::GetAmplitudes(Float_t *amplitudes, Float_t *amplitudesX1, Float_t *amplitudesY1)
{
//////////////////////////////////////////////////////////////////////////////
//   GETTER FUNCTION                                                      
//                                                     
//   This function gets the amplitudes of fitted 2D peaks and 1D ridges
//         -amplitudes - gets vector of amplitudes of 2D peaks
//         -amplitudesX1 - gets vector of amplitudes of 1D ridges in x direction
//         -amplitudesY1 - gets vector of amplitudes of 1D ridges in y direction
//////////////////////////////////////////////////////////////////////////////      
   
   for( Int_t i=0; i < fNPeaks; i++){
      amplitudes[i] = (Float_t) fAmpCalc[i];
      amplitudesX1[i] = (Float_t) fAmpCalcX1[i];      
      amplitudesY1[i] = (Float_t) fAmpCalcY1[i];      
   }
}

//______________________________________________________________________________
void TSpectrum2Fit::GetAmplitudeErrors(Float_t *amplitudeErrors, Float_t *amplitudeErrorsX1, Float_t *amplitudeErrorsY1)
{
//////////////////////////////////////////////////////////////////////////////
//   GETTER FUNCTION                                                      
//                                                     
//   This function gets the amplitudes of fitted 2D peaks and 1D ridges
//         -amplitudeErrors - gets vector of amplitudes errors of 2D peaks
//         -amplitudeErrorsX1 - gets vector of amplitudes errors of 1D ridges in x direction
//         -amplitudesErrorY1 - gets vector of amplitudes errors of 1D ridges in y direction
//////////////////////////////////////////////////////////////////////////////      
   
   for( Int_t i=0; i < fNPeaks; i++){
      amplitudeErrors[i] = (Float_t) fAmpErr[i];
      amplitudeErrorsX1[i] = (Float_t) fAmpErrX1[i];      
      amplitudeErrorsY1[i] = (Float_t) fAmpErrY1[i];      
   }
}

//______________________________________________________________________________
void TSpectrum2Fit::GetVolumes(Float_t *volumes)
{
//////////////////////////////////////////////////////////////////////////////
//   GETTER FUNCTION                                                      
//                                                     
//   This function gets the volumes of fitted 2D peaks
//         -volumes - gets vector of volumes of 2D peaks
//////////////////////////////////////////////////////////////////////////////         
   for( Int_t i=0; i < fNPeaks; i++){
      volumes[i] = (Float_t) fVolume[i];
   }
}

//______________________________________________________________________________
void TSpectrum2Fit::GetVolumeErrors(Float_t *volumeErrors)
{
//////////////////////////////////////////////////////////////////////////////
//   GETTER FUNCTION                                                      
//                                                     
//   This function gets errors of the volumes of fitted 2D peaks
//         -volumeErrors - gets vector of volumes errors of 2D peaks
//////////////////////////////////////////////////////////////////////////////            
   for( Int_t i=0; i < fNPeaks; i++){
      volumeErrors[i] = (Float_t) fVolumeErr[i];
   }
}

//______________________________________________________________________________
void TSpectrum2Fit::GetSigmaX(Double_t &sigmaX, Double_t &sigmaErrX)
{
//////////////////////////////////////////////////////////////////////////////
//   GETTER FUNCTION                                                      
//                                                     
//   This function gets the sigma x parameter and its error
//         -sigmaX - gets the fitted value of sigma x parameter
//         -sigmaErrX - gets error value of sigma x parameter
//////////////////////////////////////////////////////////////////////////////      
   sigmaX=fSigmaCalcX;
   sigmaErrX=fSigmaErrX;
}

//______________________________________________________________________________
void TSpectrum2Fit::GetSigmaY(Double_t &sigmaY, Double_t &sigmaErrY)
{
//////////////////////////////////////////////////////////////////////////////
//   GETTER FUNCTION                                                      
//                                                     
//   This function gets the sigma y parameter and its error
//         -sigmaY - gets the fitted value of sigma y parameter
//         -sigmaErrY - gets error value of sigma y parameter
//////////////////////////////////////////////////////////////////////////////         
   sigmaY=fSigmaCalcY;
   sigmaErrY=fSigmaErrY;
}

//______________________________________________________________________________
void TSpectrum2Fit::GetRo(Double_t &ro, Double_t &roErr)
{
//////////////////////////////////////////////////////////////////////////////
//   GETTER FUNCTION                                                      
//                                                     
//   This function gets the ro parameter and its error
//         -ro - gets the fitted value of ro parameter
//         -roErr - gets error value of ro parameter
//////////////////////////////////////////////////////////////////////////////         
   ro=fRoCalc;
   roErr=fRoErr;
}

//______________________________________________________________________________
void TSpectrum2Fit::GetBackgroundParameters(Double_t &a0, Double_t &a0Err, Double_t &ax, Double_t &axErr, Double_t &ay, Double_t &ayErr)
{
//////////////////////////////////////////////////////////////////////////////
//   GETTER FUNCTION                                                      
//                                                     
//   This function gets the background parameters and their errors
//         -a0 - gets the fitted value of a0 parameter
//         -a0Err - gets error value of a0 parameter
//         -ax - gets the fitted value of ax parameter
//         -axErr - gets error value of ax parameter
//         -ay - gets the fitted value of ay parameter
//         -ayErr - gets error value of ay parameter
//////////////////////////////////////////////////////////////////////////////      
   
   a0 = fA0Calc;
   a0Err = fA0Err;
   ax = fAxCalc;
   axErr = fAxErr;
   ay = fAyCalc;
   ayErr = fAyErr;    
}

//______________________________________________________________________________
void TSpectrum2Fit::GetTailParameters(Double_t &txy, Double_t &txyErr, Double_t &tx, Double_t &txErr, Double_t &ty, Double_t &tyErr, Double_t &bx, Double_t &bxErr, Double_t &by, Double_t &byErr, Double_t &sxy, Double_t &sxyErr, Double_t &sx, Double_t &sxErr, Double_t &sy, Double_t &syErr)
{
//////////////////////////////////////////////////////////////////////////////
//   GETTER FUNCTION                                                      
//                                                     
//   This function gets the tail parameters and their errors
//         -txy - gets the fitted value of txy parameter
//         -txyErr - gets error value of txy parameter
//         -tx - gets the fitted value of tx parameter
//         -txErr - gets error value of tx parameter
//         -ty - gets the fitted value of ty parameter
//         -tyErr - gets error value of ty parameter
//         -bx - gets the fitted value of bx parameter
//         -bxErr - gets error value of bx parameter
//         -by - gets the fitted value of by parameter
//         -byErr - gets error value of by parameter
//         -sxy - gets the fitted value of sxy parameter
//         -sxyErr - gets error value of sxy parameter
//         -sx - gets the fitted value of sx parameter
//         -sxErr - gets error value of sx parameter
//         -sy - gets the fitted value of sy parameter
//         -syErr - gets error value of sy parameter
//////////////////////////////////////////////////////////////////////////////            
   txy = fTxyCalc;
   txyErr = fTxyErr;
   tx = fTxCalc;
   txErr = fTxErr;  
   ty = fTyCalc;
   tyErr = fTyErr;  
   bx = fBxCalc;
   bxErr = fBxErr;  
   by = fByCalc;
   byErr = fByErr;    
   sxy = fSxyCalc;
   sxyErr = fSxyErr;  
   sx = fSxCalc;
   sxErr = fSxErr;    
   sy = fSyCalc;
   syErr = fSyErr;    
}

