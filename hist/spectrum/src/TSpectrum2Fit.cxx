// @(#)root/spectrum:$Id$
// Author: Miroslav Morhac   25/09/2006

/** \class TSpectrum2Fit
    \ingroup Spectrum
    \brief Advanced 2-dimensional spectra fitting functions
    \author Miroslav Morhac

 \legacy{TSpectrum2Fit}

Class for fitting 2D spectra using AWMI (algorithm without matrix
inversion) and conjugate gradient algorithms for symmetrical
matrices (Stiefel-Hestens method). AWMI method allows to fit
simultaneously 100s up to 1000s peaks. Stiefel method is very stable,
it converges faster, but is more time consuming.


 The algorithms in this class have been published in the following references:

 1. M. Morhac et al.: Efficient fitting algorithms applied to
    analysis of coincidence gamma-ray spectra. Computer Physics
    Communications, Vol 172/1 (2005) pp. 19-41.

 2.  M. Morhac et al.: Study of fitting algorithms applied to
    simultaneous analysis of large number of peaks in gamma-ray spectra.
    Applied Spectroscopy, Vol. 57, No. 7, pp. 753-760, 2003.
*/

#include "TSpectrum2Fit.h"
#include "TMath.h"

ClassImp(TSpectrum2Fit);

////////////////////////////////////////////////////////////////////////////////
/// Default constructor

TSpectrum2Fit::TSpectrum2Fit() :TNamed("Spectrum2Fit", "Miroslav Morhac peak fitter")
{
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

////////////////////////////////////////////////////////////////////////////////
/// numberPeaks: number of fitted peaks (must be greater than zero)
/// the constructor allocates arrays for all fitted parameters (peak positions,
/// amplitudes etc) and sets the member variables to their default values. One
/// can change these variables by member functions (setters) of TSpectrumFit class.
///
/// Shape function of the fitted
/// peaks contains the two-dimensional symmetrical Gaussian two one-dimensional
/// symmetrical Gaussian ridges as well as non-symmetrical terms and background.
///
/// \image html spectrum2fit_constructor_image001.gif

TSpectrum2Fit::TSpectrum2Fit(Int_t numberPeaks) :TNamed("Spectrum2Fit", "Miroslav Morhac peak fitter")
{
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

////////////////////////////////////////////////////////////////////////////////
/// Destructor

TSpectrum2Fit::~TSpectrum2Fit()
{
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


////////////////////////////////////////////////////////////////////////////////
///   This function calculates error function of x.


Double_t TSpectrum2Fit::Erfc(Double_t x)
{
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

////////////////////////////////////////////////////////////////////////////////
///   This function calculates derivative of error function of x.

Double_t TSpectrum2Fit::Derfc(Double_t x)
{
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

////////////////////////////////////////////////////////////////////////////////
/// power function

Double_t TSpectrum2Fit::Ourpowl(Double_t a, Int_t pw)
{
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

////////////////////////////////////////////////////////////////////////////////
///   This function calculates solution of the system of linear equations.
///   The matrix a should have a dimension size*(size+4)
///   The calling function should fill in the matrix, the column size should
///   contain vector y (right side of the system of equations). The result is
///   placed into size+1 column of the matrix.
///   according to sigma of peaks.
///
///      Function parameters:
///              - a-matrix with dimension size*(size+4)
///              - size-number of rows of the matrix

void TSpectrum2Fit::StiefelInversion(Double_t **a, Int_t size)
{
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

////////////////////////////////////////////////////////////////////////////////
///   This function calculates 2D peaks shape function (see manual)
///
///      Function parameters:
///              - numOfFittedPeaks-number of fitted peaks
///              - x-channel in x-dimension
///              - y-channel in y-dimension
///              - parameter-array of peaks parameters (amplitudes and positions)
///              - sigmax-sigmax of peaks
///              - sigmay-sigmay of peaks
///              - ro-correlation coefficient
///              - a0,ax,ay-bac kground coefficients
///              - txy,tx,ty, sxy,sy,sx-relative amplitudes
///              - bx, by-slopes

Double_t TSpectrum2Fit::Shape2(Int_t numOfFittedPeaks, Double_t x, Double_t y,
                            const Double_t *parameter, Double_t sigmax,
                            Double_t sigmay, Double_t ro, Double_t a0, Double_t ax,
                            Double_t ay, Double_t txy, Double_t sxy, Double_t tx,
                            Double_t ty, Double_t sx, Double_t sy, Double_t bx,
                            Double_t by)
{
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

////////////////////////////////////////////////////////////////////////////////
///   This function calculates derivative of 2D peaks shape function (see manual)
///   according to amplitude of 2D peak
///
///      Function parameters:
///              - x-channel in x-dimension
///              - y-channel in y-dimension
///              - x0-position of peak in x-dimension
///              - y0-position of peak in y-dimension
///              - sigmax-sigmax of peaks
///              - sigmay-sigmay of peaks
///              - ro-correlation coefficient
///              - txy, sxy-relative amplitudes
///              - bx, by-slopes

Double_t TSpectrum2Fit::Deramp2(Double_t x, Double_t y, Double_t x0, Double_t y0,
                            Double_t sigmax, Double_t sigmay, Double_t ro,
                            Double_t txy, Double_t sxy, Double_t bx, Double_t by)
{
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

////////////////////////////////////////////////////////////////////////////////
///   This function calculates derivative of 2D peaks shape function (see manual)
///   according to amplitude of the ridge
///
///      Function parameters:
///              - x-channel in x-dimension
///              - x0-position of peak in x-dimension
///              - y0-position of peak in y-dimension
///              - sigmax-sigmax of peaks
///              - ro-correlation coefficient
///              - tx, sx-relative amplitudes
///              - bx-slope

Double_t TSpectrum2Fit::Derampx(Double_t x, Double_t x0, Double_t sigmax, Double_t tx,
                             Double_t sx, Double_t bx)
{
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

////////////////////////////////////////////////////////////////////////////////
///   This function calculates derivative of 2D peaks shape function (see manual)
///   according to x position of 2D peak
///
///      Function parameters:
///              - x-channel in x-dimension
///              - y-channel in y-dimension
///              - a-amplitude
///              - x0-position of peak in x-dimension
///              - y0-position of peak in y-dimension
///              - sigmax-sigmax of peaks
///              - sigmay-sigmay of peaks
///              - ro-correlation coefficient
///              - txy, sxy-relative amplitudes
///              - bx, by-slopes

Double_t TSpectrum2Fit::Deri02(Double_t x, Double_t y, Double_t a, Double_t x0,
                            Double_t y0, Double_t sigmax, Double_t sigmay,
                            Double_t ro, Double_t txy, Double_t sxy, Double_t bx,
                            Double_t by)
{
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

////////////////////////////////////////////////////////////////////////////////
///   This function calculates second derivative of 2D peaks shape function
///   (see manual) according to x position of 2D peak
///
///      Function parameters:
///              - x-channel in x-dimension
///              - y-channel in y-dimension
///              - a-amplitude
///              - x0-position of peak in x-dimension
///              - y0-position of peak in y-dimension
///              - sigmax-sigmax of peaks
///              - sigmay-sigmay of peaks
///              - ro-correlation coefficient

Double_t TSpectrum2Fit::Derderi02(Double_t x, Double_t y, Double_t a, Double_t x0,
                              Double_t y0, Double_t sigmax, Double_t sigmay,
                              Double_t ro)
{
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

////////////////////////////////////////////////////////////////////////////////
///   This function calculates derivative of 2D peaks shape function (see manual)
///   according to y position of 2D peak
///      Function parameters:
///              - x-channel in x-dimension
///              - y-channel in y-dimension
///              - a-amplitude
///              - x0-position of peak in x-dimension
///              - y0-position of peak in y-dimension
///              - sigmax-sigmax of peaks
///              - sigmay-sigmay of peaks
///              - ro-correlation coefficient
///              - txy, sxy-relative amplitudes
///              - bx, by-slopes

Double_t TSpectrum2Fit::Derj02(Double_t x, Double_t y, Double_t a, Double_t x0,
                            Double_t y0, Double_t sigmax, Double_t sigmay,
                            Double_t ro, Double_t txy, Double_t sxy, Double_t bx,
                            Double_t by)
{


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

////////////////////////////////////////////////////////////////////////////////
///   This function calculates second derivative of 2D peaks shape function
///   (see manual) according to y position of 2D peak
///
///      Function parameters:
///              - x-channel in x-dimension
///              - y-channel in y-dimension
///              - a-amplitude
///              - x0-position of peak in x-dimension
///              - y0-position of peak in y-dimension
///              - sigmax-sigmax of peaks
///              - sigmay-sigmay of peaks
///              - ro-correlation coefficient

Double_t TSpectrum2Fit::Derderj02(Double_t x, Double_t y, Double_t a, Double_t x0,
                               Double_t y0, Double_t sigmax, Double_t sigmay,
                               Double_t ro)
{
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

////////////////////////////////////////////////////////////////////////////////
///   This function calculates derivative of 2D peaks shape function (see manual)
///   according to x position of 1D ridge
///
///      Function parameters:
///              - x-channel in x-dimension
///              - ax-amplitude of ridge
///              - x0-position of peak in x-dimension
///              - sigmax-sigmax of peaks
///              - ro-correlation coefficient
///              - tx, sx-relative amplitudes
///              - bx-slope

Double_t TSpectrum2Fit::Deri01(Double_t x, Double_t ax, Double_t x0, Double_t sigmax,
                            Double_t tx, Double_t sx, Double_t bx)
{
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

////////////////////////////////////////////////////////////////////////////////
///   This function calculates second derivative of 2D peaks shape function
///   (see manual) according to x position of 1D ridge
///
///      Function parameters:
///              - x-channel in x-dimension
///              - ax-amplitude of ridge
///              - x0-position of peak in x-dimension
///              - sigmax-sigmax of peaks

Double_t TSpectrum2Fit::Derderi01(Double_t x, Double_t ax, Double_t x0,
                               Double_t sigmax)
{
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

////////////////////////////////////////////////////////////////////////////////
///   This function calculates derivative of peaks shape function (see manual)
///   according to sigmax of peaks.
///
///      Function parameters:
///              - numOfFittedPeaks-number of fitted peaks
///              - x,y-position of channel
///              - parameter-array of peaks parameters (amplitudes and positions)
///              - sigmax-sigmax of peaks
///              - sigmay-sigmay of peaks
///              - ro-correlation coefficient
///              - txy, sxy, tx, sx-relative amplitudes
///              - bx, by-slopes

Double_t TSpectrum2Fit::Dersigmax(Int_t numOfFittedPeaks, Double_t x, Double_t y,
                               const Double_t *parameter, Double_t sigmax,
                               Double_t sigmay, Double_t ro, Double_t txy,
                               Double_t sxy, Double_t tx, Double_t sx, Double_t bx,
                               Double_t by)
{
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

////////////////////////////////////////////////////////////////////////////////
///   This function calculates second derivative of peaks shape function
///   (see manual) according to sigmax of peaks.
///
///      Function parameters:
///              - numOfFittedPeaks-number of fitted peaks
///              - x,y-position of channel
///              - parameter-array of peaks parameters (amplitudes and positions)
///              - sigmax-sigmax of peaks
///              - sigmay-sigmay of peaks
///              - ro-correlation coefficient

Double_t TSpectrum2Fit::Derdersigmax(Int_t numOfFittedPeaks, Double_t x,
                                  Double_t y, const Double_t *parameter,
                                  Double_t sigmax, Double_t sigmay,
                                  Double_t ro)
{
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

////////////////////////////////////////////////////////////////////////////////
///   This function calculates derivative of peaks shape function (see manual)
///   according to sigmax of peaks.
///
///      Function parameters:
///              - numOfFittedPeaks-number of fitted peaks
///              - x,y-position of channel
///              - parameter-array of peaks parameters (amplitudes and positions)
///              - sigmax-sigmax of peaks
///              - sigmay-sigmay of peaks
///              - ro-correlation coefficient
///              - txy, sxy, ty, sy-relative amplitudes
///              - bx, by-slopes

Double_t TSpectrum2Fit::Dersigmay(Int_t numOfFittedPeaks, Double_t x, Double_t y,
                               const Double_t *parameter, Double_t sigmax,
                               Double_t sigmay, Double_t ro, Double_t txy,
                               Double_t sxy, Double_t ty, Double_t sy, Double_t bx,
                               Double_t by)
{
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

////////////////////////////////////////////////////////////////////////////////
///   This function calculates second derivative of peaks shape function
///   (see manual) according to sigmay of peaks.
///
///      Function parameters:
///              - numOfFittedPeaks-number of fitted peaks
///              - x,y-position of channel
///              - parameter-array of peaks parameters (amplitudes and positions)
///              - sigmax-sigmax of peaks
///              - sigmay-sigmay of peaks
///              - ro-correlation coefficient

Double_t TSpectrum2Fit::Derdersigmay(Int_t numOfFittedPeaks, Double_t x,
                                  Double_t y, const Double_t *parameter,
                                  Double_t sigmax, Double_t sigmay,
                                  Double_t ro)
{
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

////////////////////////////////////////////////////////////////////////////////
///   This function calculates derivative of peaks shape function (see manual)
///   according to correlation coefficient ro.
///
///      Function parameters:
///              - numOfFittedPeaks-number of fitted peaks
///              - x,y-position of channel
///              - parameter-array of peaks parameters (amplitudes and positions)
///              - sx-sigmax of peaks
///              - sy-sigmay of peaks
///              - r-correlation coefficient ro

Double_t TSpectrum2Fit::Derro(Int_t numOfFittedPeaks, Double_t x, Double_t y,
                           const Double_t *parameter, Double_t sx, Double_t sy,
                           Double_t r)
{
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

////////////////////////////////////////////////////////////////////////////////
///   This function calculates derivative of peaks shape function (see manual)
///   according to relative amplitude txy.
///
///      Function parameters:
///              - numOfFittedPeaks-number of fitted peaks
///              - x,y-position of channel
///              - parameter-array of peaks parameters (amplitudes and positions)
///              - sigmax-sigmax of peaks
///              - sigmay-sigmay of peaks
///              - bx, by-slopes

Double_t TSpectrum2Fit::Dertxy(Int_t numOfFittedPeaks, Double_t x, Double_t y,
                            const Double_t *parameter, Double_t sigmax,
                            Double_t sigmay, Double_t bx, Double_t by)
{
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

////////////////////////////////////////////////////////////////////////////////
///   This function calculates derivative of peaks shape function (see manual)
///   according to relative amplitude sxy.
///
///      Function parameters:
///              - numOfFittedPeaks-number of fitted peaks
///              - x,y-position of channel
///              - parameter-array of peaks parameters (amplitudes and positions)
///              - sigmax-sigmax of peaks
///              - sigmay-sigmay of peaks

Double_t TSpectrum2Fit::Dersxy(Int_t numOfFittedPeaks, Double_t x, Double_t y,
                            const Double_t *parameter, Double_t sigmax,
                            Double_t sigmay)
{
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

////////////////////////////////////////////////////////////////////////////////
///   This function calculates derivative of peaks shape function (see manual)
///   according to relative amplitude tx.
///
///      Function parameters:
///              - numOfFittedPeaks-number of fitted peaks
///              - x-position of channel
///              - parameter-array of peaks parameters (amplitudes and positions)
///              - sigmax-sigma of 1D ridge
///              - bx-slope

Double_t TSpectrum2Fit::Dertx(Int_t numOfFittedPeaks, Double_t x,
                           const Double_t *parameter, Double_t sigmax,
                           Double_t bx)
{
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

////////////////////////////////////////////////////////////////////////////////
///   This function calculates derivative of peaks shape function (see manual)
///   according to relative amplitude ty.
///
///      Function parameters:
///              - numOfFittedPeaks-number of fitted peaks
///              - x-position of channel
///              - parameter-array of peaks parameters (amplitudes and positions)
///              - sigmax-sigma of 1D ridge
///              - bx-slope

Double_t TSpectrum2Fit::Derty(Int_t numOfFittedPeaks, Double_t x,
                           const Double_t *parameter, Double_t sigmax,
                           Double_t bx)
{
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

////////////////////////////////////////////////////////////////////////////////
///   This function calculates derivative of peaks shape function (see manual)
///   according to relative amplitude sx.
///
///      Function parameters:
///              - numOfFittedPeaks-number of fitted peaks
///              - x-position of channel
///              - parameter-array of peaks parameters (amplitudes and positions)
///              - sigmax-sigma of 1D ridge

Double_t TSpectrum2Fit::Dersx(Int_t numOfFittedPeaks, Double_t x,
                           const Double_t *parameter, Double_t sigmax)
{
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

////////////////////////////////////////////////////////////////////////////////
///   This function calculates derivative of peaks shape function (see manual)
///   according to relative amplitude sy.
///
///      Function parameters:
///              - numOfFittedPeaks-number of fitted peaks
///              - x-position of channel
///              - parameter-array of peaks parameters (amplitudes and positions)
///              - sigmax-sigma of 1D ridge

Double_t TSpectrum2Fit::Dersy(Int_t numOfFittedPeaks, Double_t x,
                           const Double_t *parameter, Double_t sigmax)
{
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

////////////////////////////////////////////////////////////////////////////////
///   This function calculates derivative of peaks shape function (see manual)
///   according to slope bx.
///
///      Function parameters:
///              - numOfFittedPeaks-number of fitted peaks
///              - x,y-position of channel
///              - parameter-array of peaks parameters (amplitudes and positions)
///              - sigmax-sigmax of peaks
///              - sigmay-sigmay of peaks
///              - txy, tx-relative amplitudes
///              - bx, by-slopes

Double_t TSpectrum2Fit::Derbx(Int_t numOfFittedPeaks, Double_t x, Double_t y,
                           const Double_t *parameter, Double_t sigmax,
                           Double_t sigmay, Double_t txy, Double_t tx, Double_t bx,
                           Double_t by)
{
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

////////////////////////////////////////////////////////////////////////////////
///   This function calculates derivative of peaks shape function (see manual)
///   according to slope by.
///
///      Function parameters:
///              - numOfFittedPeaks-number of fitted peaks
///              - x,y-position of channel
///              - parameter-array of peaks parameters (amplitudes and positions)
///              - sigmax-sigmax of peaks
///              - sigmay-sigmay of peaks
///              - txy, ty-relative amplitudes
///              - bx, by-slopes

Double_t TSpectrum2Fit::Derby(Int_t numOfFittedPeaks, Double_t x, Double_t y,
                           const Double_t *parameter, Double_t sigmax,
                           Double_t sigmay, Double_t txy, Double_t ty, Double_t bx,
                           Double_t by)
{
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

////////////////////////////////////////////////////////////////////////////////
///   This function calculates volume of a peak
///
///      Function parameters:
///              - a-amplitude of the peak
///              - sx,sy-sigmas of peak
///              - ro-correlation coefficient

Double_t TSpectrum2Fit::Volume(Double_t a, Double_t sx, Double_t sy, Double_t ro)
{
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

////////////////////////////////////////////////////////////////////////////////
///   This function calculates derivative of the volume of a peak
///   according to amplitude
///
///      Function parameters:
///              - sx,sy-sigmas of peak
///              - ro-correlation coefficient

Double_t TSpectrum2Fit::Derpa2(Double_t sx, Double_t sy, Double_t ro)
{
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

////////////////////////////////////////////////////////////////////////////////
///   This function calculates derivative of the volume of a peak
///   according to sigmax
///
///      Function parameters:
///              - a-amplitude of peak
///              - sy-sigma of peak
///              - ro-correlation coefficient

Double_t TSpectrum2Fit::Derpsigmax(Double_t a, Double_t sy, Double_t ro)
{
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

////////////////////////////////////////////////////////////////////////////////
///   This function calculates derivative of the volume of a peak
///   according to sigmay
///
///      Function parameters:
///              - a-amplitude of peak
///              - sx-sigma of peak
///              - ro-correlation coefficient

Double_t TSpectrum2Fit::Derpsigmay(Double_t a, Double_t sx, Double_t ro)
{
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

////////////////////////////////////////////////////////////////////////////////
///   This function calculates derivative of the volume of a peak
///   according to ro
///
///      Function parameters:
///              - a-amplitude of peak
///              - sx,sy-sigmas of peak
///              - ro-correlation coefficient

Double_t TSpectrum2Fit::Derpro(Double_t a, Double_t sx, Double_t sy, Double_t ro)
{
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


////////////////////////////////////////////////////////////////////////////////
///  This function fits the source spectrum. The calling program should
///  fill in input parameters of the TSpectrum2Fit class.
///  The fitted parameters are written into
///  TSpectrum2Fit class output parameters and fitted data are written into
///  source spectrum.
///
///   Function parameters:
///     - source-pointer to the matrix of source spectrum
///
/// ### Fitting
///
/// Goal: to estimate simultaneously peak shape parameters in spectra with large
/// number of peaks
///
///  - peaks can be fitted separately, each peak (or multiplets) in a region or
///    together all peaks in a spectrum. To fit separately each peak one needs to
///    determine the fitted region. However it can happen that the regions of
///    neighbouring peaks are overlapping. Then the results of fitting are very poor.
///    On the other hand, when fitting together all peaks found in a spectrum, one
///    needs to have a method that is stable (converges) and fast enough to carry out
///    fitting in reasonable time
///
///  - we have implemented the non-symmetrical semiempirical peak shape function
///
///  - it contains the two-dimensional symmetrical Gaussian two one-dimensional
///    symmetrical Gaussian ridges as well as non-symmetrical terms and background.
///
///    \image html spectrum2fit_awmi_image001.gif
///
///     where Txy, Tx, Ty, Sxy, Sx, Sy are relative amplitudes and Bx, By are slopes.
///
///  - algorithm without matrix inversion (AWMI) allows fitting tens, hundreds
///    of peaks simultaneously that represent sometimes thousands of parameters [2],[5].
///
/// #### References:
///
/// [1] Phillps G.W., Marlow K.W.,
/// NIM 137 (1976) 525.
///
/// [2] I. A. Slavic: Nonlinear
/// least-squares fitting without matrix inversion applied to complex Gaussian
/// spectra analysis. NIM 134 (1976) 285-289.
///
/// [3] T. Awaya: A new method for
/// curve fitting to the data with low statistics not using chi-square method. NIM
/// 165 (1979) 317-323.
///
/// [4] T. Hauschild, M. Jentschel:
/// Comparison of maximum likelihood estimation and chi-square statistics applied
/// to counting experiments. NIM A 457 (2001) 384-401.
///
/// [5] M. Morh, J.
/// Kliman, M. Jandel, Krupa, V. Matouoek: Study of fitting algorithms
/// applied to simultaneous analysis of large number of peaks in -ray spectra.
/// Applied Spectroscopy, Vol. 57, No. 7, pp. 753-760, 2003
///
/// ### Example 1 - script FitAwmi2.c:
///
/// \image html spectrum2fit_awmi_image002.jpg Original Fig. 1 two-dimensional spectrum with found peaks (using TSpectrum2 peak searching function). The positions of peaks were used as initial estimates in fitting procedure.
///
/// \image html spectrum2fit_awmi_image003.jpg Fig. 2 Fitted (generated from fitted parameters) spectrum of the data from Fig. 1 using Algorithm Without Matrix Inversion. Each peak was represented by 7 parameters, which together with Sigmax, Sigmay and a0 resulted in 38 fitted parameters. The chi-squareafter 1000 iterations was 0.642342.
///
/// #### Script:
///
/// Example to illustrate fitting function, algorithm without matrix inversion (AWMI) (class TSpectrumFit2).
/// To execute this example, do
///
/// `root > .x FitAwmi2.C`
///
/// ~~~ {.cpp}
///   void FitAwmi2() {
///      Int_t i, j, nfound;
///      Int_t nbinsx = 64;
///      Int_t nbinsy = 64;
///      Int_t xmin = 0;
///      Int_t xmax = nbinsx;
///      Int_t ymin = 0;
///      Int_t ymax = nbinsy;
///      Double_t ** source = new float *[nbinsx];
///      Double_t ** dest = new float *[nbinsx];
///      for (i=0;i<nbinsx;i++)
///         source[i]=new float[nbinsy];
///      for (i=0;i<nbinsx;i++)
///         dest[i]=new float[nbinsy];
///      TH2F *search = new TH2F("search","High resolution peak searching",nbinsx,xmin,xmax,nbinsy,ymin,ymax);
///      TFile *f = new TFile("TSpectrum2.root");
///      search=(TH2F*) f->Get("search4;1");
///      TCanvas *Searching = new TCanvas("Searching","Two-dimensional fitting using Algorithm Without Matrix Inversion",10,10,1000,700);
///      TSpectrum2 *s = new TSpectrum2();
///      for (i = 0; i < nbinsx; i++){
///         for (j = 0; j < nbinsy; j++){
///            source[i][j] = search->GetBinContent(i + 1,j + 1);
///         }
///      }
///      //searching for candidate peaks positions
///      nfound = s->SearchHighRes(source, dest, nbinsx, nbinsy, 2, 5, kTRUE, 3, kFALSE, 3);
///      Bool_t *FixPosX = new Bool_t[nfound];
///      Bool_t *FixPosY = new Bool_t[nfound];
///      Bool_t *FixAmp = new Bool_t[nfound];
///      Double_t *PosX = new Double_t[nfound];
///      Double_t *PosY = new Double_t[nfound];
///      Double_t *Amp = new Double_t[nfound];
///      Double_t *AmpXY = new Double_t[nfound];
///      PosX = s->GetPositionX();
///      PosY = s->GetPositionY();
///      printf("Found %d candidate peaks\n",nfound);
///      for(i = 0; i< nfound ; i++){
///         FixPosX[i] = kFALSE;
///         FixPosY[i] = kFALSE;
///         FixAmp[i] = kFALSE;
///         Amp[i] = source[(Int_t)(PosX[i]+0.5)][(Int_t)(PosY[i]+0.5)]; //initial values of peaks amplitudes, input parameters
///         AmpXY[i] = 0;
///      }
///      //filling in the initial estimates of the input parameters
///      TSpectrumFit2 *pfit=new TSpectrumFit2(nfound);
///      pfit->SetFitParameters(xmin, xmax-1, ymin, ymax-1, 1000, 0.1, pfit->kFitOptimChiCounts,
///      pfit->kFitAlphaHalving, pfit->kFitPower2,
///      pfit->kFitTaylorOrderFirst);
///      pfit->SetPeakParameters(2, kFALSE, 2, kFALSE, 0, kTRUE, PosX, (Bool_t *)
///      FixPosX, PosY, (Bool_t *) FixPosY, PosX, (Bool_t *) FixPosX, PosY, (Bool_t *)
///      FixPosY, Amp, (Bool_t *) FixAmp, AmpXY, (Bool_t *) FixAmp, AmpXY, (Bool_t *)
///      FixAmp);
///      pfit->SetBackgroundParameters(0, kFALSE, 0, kTRUE, 0, kTRUE);
///      pfit->FitAwmi(source);
///      for (i = 0; i < nbinsx; i++){
///         for (j = 0; j < nbinsy; j++){
///            search->SetBinContent(i + 1, j + 1,source[i][j]);
///          }
///      }
///   search->Draw("SURF");
///   }
/// ~~~
///
/// ### Example 2 - script FitAwmi2.c:
///
/// \image html spectrum2fit_awmi_image004.jpg Fig. 3 Original two-dimensional gamma-gamma-ray spectrum with found peaks (using TSpectrum2 peak searching function).
///
/// \image html spectrum2fit_awmi_image005.jpg Fig. 4 Fitted (generated from fitted parameters) spectrum of the data from Fig. 3 using Algorithm Without Matrix Inversion. 152 peaks were identified. Each peak was represented by 7 parameters, which together with Sigmax, Sigmay and a0 resulted in 1067 fitted parameters. The chi-square after 1000 iterations was 0.728675. One can observe good correspondence with the original data.
///
/// #### Script:
///
////
/// Example to illustrate fitting function, algorithm without matrix inversion
/// (AWMI) (class TSpectrumFit2). To execute this example, do:
///
/// `root > .x FitA2.C`
///
/// ~~~ {.cpp}
///   void FitA2() {
///      Int_t i, j, nfound;
///      Int_t nbinsx = 256;
///      Int_t nbinsy = 256;
///      Int_t xmin = 0;
///      Int_t xmax = nbinsx;
///      Int_t ymin = 0;
///      Int_t ymax = nbinsy;
///      Double_t ** source = new float *[nbinsx];
///      Double_t ** dest = new float *[nbinsx];
///      for (i=0;i<nbinsx;i++)
///         source[i]=new
///      float[nbinsy];
///      for (i=0;i<nbinsx;i++)
///         dest[i]=new
///      float[nbinsy];
///      TH2F *search = new TH2F("search","High resolution peak
///      searching",nbinsx,xmin,xmax,nbinsy,ymin,ymax);
///      TFile *f = new TFile("TSpectrum2.root");
///      search=(TH2F*) f->Get("fit1;1");
///      TCanvas *Searching = new TCanvas("Searching","Two-dimensional fitting using Algorithm Without Matrix Inversion",10,10,1000,700);
///      TSpectrum2 *s = new TSpectrum2(1000,1);
///      for (i = 0; i < nbinsx; i++){
///         for (j = 0; j < nbinsy; j++){
///            source[i][j] = search->GetBinContent(i + 1,j + 1);
///         }
///      }
///      nfound = s->SearchHighRes(source, dest, nbinsx, nbinsy, 2, 2, kTRUE, 100, kFALSE, 3);
///      printf("Found %d candidate peaks\n",nfound);
///      Bool_t *FixPosX = new Bool_t[nfound];
///      Bool_t *FixPosY = new Bool_t[nfound];
///      Bool_t *FixAmp = new Bool_t[nfound];
///      Double_t *PosX = new Double_t[nfound];
///      Double_t *PosY = new Double_t[nfound];
///      Double_t *Amp = new Double_t[nfound];
///      Double_t *AmpXY = new Double_t[nfound];
///      PosX = s->GetPositionX();
///      PosY = s->GetPositionY();
///      for(i = 0; i< nfound ; i++){
///         FixPosX[i] = kFALSE;
///         FixPosY[i] = kFALSE;
///         FixAmp[i] = kFALSE;
///         Amp[i] = source[(Int_t)(PosX[i]+0.5)][(Int_t)(PosY[i]+0.5)]; //initial values of peaks amplitudes, input parameters
///         AmpXY[i] = 0;
///      }
///      //filling in the initial estimates of the input parameters
///      TSpectrumFit2 *pfit=new TSpectrumFit2(nfound);
///      pfit->SetFitParameters(xmin, xmax-1, ymin, ymax-1, 1000, 0.1,
///      pfit->kFitOptimChiCounts, pfit->kFitAlphaHalving, pfit->kFitPower2,
///      pfit->kFitTaylorOrderFirst);
///      pfit->SetPeakParameters(2, kFALSE, 2, kFALSE, 0, kTRUE, PosX, (Bool_t *)
///      FixPosX, PosY, (Bool_t *) FixPosY, PosX, (Bool_t *) FixPosX, PosY, (Bool_t *)
///      FixPosY, Amp, (Bool_t *) FixAmp, AmpXY, (Bool_t *) FixAmp, AmpXY, (Bool_t *)
///      FixAmp);
///      pfit->SetBackgroundParameters(0, kFALSE, 0, kTRUE, 0, kTRUE);
///      pfit->FitAwmi(source);
///      for (i = 0; i < nbinsx; i++){
///         for (j = 0; j < nbinsy; j++){
///            search->SetBinContent(i + 1, j + 1,source[i][j]);
///         }
///      }
///      search->Draw("SURF");
///   }
/// ~~~

void TSpectrum2Fit::FitAwmi(Double_t **source)
{


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
            f = Shape2(fNPeaks, i1, i2,
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
                  a = Deramp2(i1, i2,
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
                  a = Deri02(i1, i2,
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
                     d = Derderi02(i1, i2,
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
                  a = Derj02(i1, i2,
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
                     d = Derderj02(i1, i2,
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
                  a = Derampx(i1, working_space[7 * j + 5],
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
                  a = Derampx(i2, working_space[7 * j + 6],
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
                  a = Deri01(i1, working_space[7 * j + 3],
                              working_space[7 * j + 5],
                              working_space[peak_vel],
                              working_space[peak_vel + 8],
                              working_space[peak_vel + 10],
                              working_space[peak_vel + 12]);
                  if (fFitTaylor == kFitTaylorOrderSecond)
                     d = Derderi01(i1, working_space[7 * j + 3],
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
                  a = Deri01(i2, working_space[7 * j + 4],
                              working_space[7 * j + 6],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 9],
                              working_space[peak_vel + 11],
                              working_space[peak_vel + 13]);
                  if (fFitTaylor == kFitTaylorOrderSecond)
                     d = Derderi01(i2, working_space[7 * j + 4],
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
               a = Dersigmax(fNPeaks, i1, i2,
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
                  d = Derdersigmax(fNPeaks, i1,
                                    i2, working_space,
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
               a = Dersigmay(fNPeaks, i1, i2,
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
                  d = Derdersigmay(fNPeaks, i1,
                                    i2, working_space,
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
               a = Derro(fNPeaks, i1, i2,
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
               a = Dertxy(fNPeaks, i1, i2,
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
               a = Dersxy(fNPeaks, i1, i2,
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
               a = Dertx(fNPeaks, i1, working_space,
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
               a = Derty(fNPeaks, i2, working_space,
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
               a = Dersx(fNPeaks, i1, working_space,
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
               a = Dersy(fNPeaks, i2, working_space,
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
               a = Derbx(fNPeaks, i1, i2,
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
               a = Derby(fNPeaks, i1, i2,
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
                     f = Shape2(fNPeaks, i1,
                                 i2, working_space,
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
                  f = Shape2(fNPeaks, i1, i2,
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
            f = Shape2(fNPeaks, i1, i2,
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
                  a = Deramp2(i1, i2,
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
                  a = Deri02(i1, i2,
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
                  a = Derj02(i1, i2,
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
                  a = Derampx(i1, working_space[7 * j + 5],
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
                  a = Derampx(i2, working_space[7 * j + 6],
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
                  a = Deri01(i1, working_space[7 * j + 3],
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
                  a = Deri01(i2, working_space[7 * j + 4],
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
               a = Dersigmax(fNPeaks, i1, i2,
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
               a = Dersigmay(fNPeaks, i1, i2,
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
               a = Derro(fNPeaks, i1, i2,
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
               a = Dertxy(fNPeaks, i1, i2,
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
               a = Dersxy(fNPeaks, i1, i2,
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
               a = Dertx(fNPeaks, i1, working_space,
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
               a = Derty(fNPeaks, i2, working_space,
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
               a = Dersx(fNPeaks, i1, working_space,
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
               a = Dersy(fNPeaks, i2, working_space,
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
               a = Derbx(fNPeaks, i1, i2,
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
               a = Derby(fNPeaks, i1, i2,
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
         f = Shape2(fNPeaks, i1, i2,
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



////////////////////////////////////////////////////////////////////////////////
/// This function fits the source spectrum. The calling program should
/// fill in input parameters of the TSpectrum2Fit class.
/// The fitted parameters are written into
/// TSpectrum2Fit class output parameters and fitted data are written into
/// source spectrum.
///
/// Function parameters:
///  - source-pointer to the matrix of source spectrum
///
/// ### Stiefel fitting algorithm
///
/// This function fits the source
/// spectrum using Stiefel-Hestens method [1]. The calling program should fill in
/// input fitting parameters of the TSpectrumFit2 class using a set of
/// TSpectrumFit2 setters. The fitted parameters are written into the class and the
/// fitted data are written into source spectrum. It converges faster than Awmi
/// method.
///
/// #### Reference:
///
/// [1] B. Mihaila: Analysis of
/// complex gamma spectra, Rom. Jorn. Phys., Vol. 39, No. 2, (1994), 139-148.
///
/// Example 1 - script FitS.c:
///
/// \image html spectrum2fit_stiefel_image001.jpg Fig. 1 Original two-dimensional spectrum with found peaks (using TSpectrum2 peak searching function). The positions of peaks were used as initial estimates in fitting procedure.
///
/// \image html spectrum2fit_stiefel_image002.jpg Fig. 2 Fitted (generated from fitted parameters) spectrum of the data from Fig. 1 using Stiefel-Hestens method. Each peak was represented by 7 parameters, which together with Sigmax, Sigmay and a0 resulted in 38 fitted parameters. The chi-square after 1000 iterations was 0.642157.
///
/// #### Script:
///
/// Example to illustrate fitting function, algorithm without matrix inversion (AWMI) (class
/// TSpectrumFit2). To execute this example, do
///
/// `root > .x FitStiefel2.C`
///
/// ~~~ {.cpp}
///   void FitStiefel2() {
///      Int_t i, j, nfound;
///      Int_t nbinsx = 64;
///      Int_t nbinsy = 64;
///      Int_t xmin = 0;
///      Int_t xmax = nbinsx;
///      Int_t ymin = 0;
///      Int_t ymax = nbinsy;
///      Double_t ** source = new float *[nbinsx];
///      Double_t ** dest = new float *[nbinsx];
///      for (i=0;i<nbinsx;i++)
///         source[i]=new float[nbinsy];
///      for (i=0;i<nbinsx;i++)
///         dest[i]=  new float[nbinsy];
///      TH2F *search = new TH2F("search","High resolution peak searching",nbinsx,xmin,xmax,nbinsy,ymin,ymax);
///      TFile *f = new TFile("TSpectrum2.root");
///      search=(TH2F*)f->Get("search4;1");
///      TCanvas *Searching = new TCanvas("Searching","Two-dimensional fitting using Stiefel-Hestens method",10,10,1000,700);
///      TSpectrum2 *s = new TSpectrum2();
///      for (i = 0; i < nbinsx; i++){
///         for (j = 0; j < nbinsy; j++){
///            source[i][j] = search->GetBinContent(i + 1,j + 1);
///         }
///      }
///      nfound = s->SearchHighRes(source, dest, nbinsx, nbinsy, 2, 5, kTRUE, 3, kFALSE, 3);
///      printf("Found %d candidate peaks\n",nfound);
///      Bool_t *FixPosX = new Bool_t[nfound];
///      Bool_t *FixPosY = new Bool_t[nfound];
///      Bool_t *FixAmp = new Bool_t[nfound];
///      Double_t *PosX = new Double_t[nfound];
///      Double_t *PosY = new Double_t[nfound];
///      Double_t *Amp = new Double_t[nfound];
///      Double_t *AmpXY = new Double_t[nfound];
///      PosX = s->GetPositionX();
///      PosY = s->GetPositionY();
///      for(i = 0; i< nfound ; i++){
///         FixPosX[i] = kFALSE;
///         FixPosY[i] = kFALSE;
///         FixAmp[i] = kFALSE;
///         Amp[i] = source[(Int_t)(PosX[i]+0.5)][(Int_t)(PosY[i]+0.5)]; //initial values of peaks amplitudes, input parameters
///         AmpXY[i] = 0;
///      }
///      //filling in the initial estimates of the input parameters
///      TSpectrumFit2 *pfit=new
///      TSpectrumFit2(nfound);
///      pfit->SetFitParameters(xmin, xmax-1, ymin, ymax-1, 1000, 0.1,
///      pfit->kFitOptimChiCounts, pfit->kFitAlphaHalving, pfit->kFitPower2,
///      pfit->kFitTaylorOrderFirst);
///      pfit->SetPeakParameters(2, kFALSE, 2, kFALSE, 0, kTRUE, PosX, (Bool_t *)
///      FixPosX, PosY, (Bool_t *) FixPosY, PosX, (Bool_t *) FixPosX, PosY, (Bool_t *)
///      FixPosY, Amp, (Bool_t *) FixAmp, AmpXY, (Bool_t *) FixAmp, AmpXY, (Bool_t *)
///      FixAmp);
///      pfit->SetBackgroundParameters(0, kFALSE, 0, kTRUE, 0, kTRUE);
///      pfit->FitStiefel(source);
///      for (i = 0; i < nbinsx; i++){
///         for (j = 0; j < nbinsy; j++){
///            search->SetBinContent(i + 1, j + 1,source[i][j]);
///         }
///      }
///      search->Draw("SURF");
/// }
/// ~~~

void TSpectrum2Fit::FitStiefel(Double_t **source)
{

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
         for (k = 0; k < (size + 4); k++) {
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
                      Deramp2(i1, i2,
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
                      Deri02(i1, i2,
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
                      Derj02(i1, i2,
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
                      Derampx(i1, working_space[7 * j + 5],
                              working_space[peak_vel],
                              working_space[peak_vel + 8],
                              working_space[peak_vel + 10],
                              working_space[peak_vel + 12]);
                  k += 1;
               }
               if (fFixAmpY1[j] == false) {
                  working_space[2 * shift + k] =
                      Derampx(i2, working_space[7 * j + 6],
                              working_space[peak_vel + 1],
                              working_space[peak_vel + 9],
                              working_space[peak_vel + 11],
                              working_space[peak_vel + 13]);
                  k += 1;
               }
               if (fFixPositionX1[j] == false) {
                  working_space[2 * shift + k] =
                      Deri01(i1, working_space[7 * j + 3],
                             working_space[7 * j + 5],
                             working_space[peak_vel],
                             working_space[peak_vel + 8],
                             working_space[peak_vel + 10],
                             working_space[peak_vel + 12]);
                  k += 1;
               }
               if (fFixPositionY1[j] == false) {
                  working_space[2 * shift + k] =
                      Deri01(i2, working_space[7 * j + 4],
                             working_space[7 * j + 6],
                             working_space[peak_vel + 1],
                             working_space[peak_vel + 9],
                             working_space[peak_vel + 11],
                             working_space[peak_vel + 13]);
                  k += 1;
               }
            } if (fFixSigmaX == false) {
               working_space[2 * shift + k] =
                   Dersigmax(fNPeaks, i1, i2,
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
                   Dersigmay(fNPeaks, i1, i2,
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
                   Derro(fNPeaks, i1, i2,
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
                   Dertxy(fNPeaks, i1, i2,
                          working_space, working_space[peak_vel],
                          working_space[peak_vel + 1],
                          working_space[peak_vel + 12],
                          working_space[peak_vel + 13]);
               k += 1;
            }
            if (fFixSxy == false) {
               working_space[2 * shift + k] =
                   Dersxy(fNPeaks, i1, i2,
                          working_space, working_space[peak_vel],
                          working_space[peak_vel + 1]);
               k += 1;
            }
            if (fFixTx == false) {
               working_space[2 * shift + k] =
                   Dertx(fNPeaks, i1, working_space,
                         working_space[peak_vel],
                         working_space[peak_vel + 12]);
               k += 1;
            }
            if (fFixTy == false) {
               working_space[2 * shift + k] =
                   Derty(fNPeaks, i2, working_space,
                         working_space[peak_vel + 1],
                         working_space[peak_vel + 13]);
               k += 1;
            }
            if (fFixSx == false) {
               working_space[2 * shift + k] =
                   Dersx(fNPeaks, i1, working_space,
                         working_space[peak_vel]);
               k += 1;
            }
            if (fFixSy == false) {
               working_space[2 * shift + k] =
                   Dersy(fNPeaks, i2, working_space,
                         working_space[peak_vel + 1]);
               k += 1;
            }
            if (fFixBx == false) {
               working_space[2 * shift + k] =
                   Derbx(fNPeaks, i1, i2,
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
                   Derby(fNPeaks, i1, i2,
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
            f = Shape2(fNPeaks, i1, i2,
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
                     f = Shape2(fNPeaks, i1,
                                 i2, working_space,
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
                  f = Shape2(fNPeaks, i1, i2,
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
            f = Shape2(fNPeaks, i1, i2,
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
                  a = Deramp2(i1, i2,
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
                  a = Deri02(i1, i2,
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
                  a = Derj02(i1, i2,
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
                  a = Derampx(i1, working_space[7 * j + 5],
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
                  a = Derampx(i2, working_space[7 * j + 6],
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
                  a = Deri01(i1, working_space[7 * j + 3],
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
                  a = Deri01(i2, working_space[7 * j + 4],
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
               a = Dersigmax(fNPeaks, i1, i2,
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
               a = Dersigmay(fNPeaks, i1, i2,
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
               a = Derro(fNPeaks, i1, i2,
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
               a = Dertxy(fNPeaks, i1, i2,
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
               a = Dersxy(fNPeaks, i1, i2,
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
               a = Dertx(fNPeaks, i1, working_space,
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
               a = Derty(fNPeaks, i2, working_space,
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
               a = Dersx(fNPeaks, i1, working_space,
                          working_space[peak_vel]);
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt; //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b; //temp_xk[k]
               }
               k += 1;
            }
            if (fFixSy == false) {
               a = Dersy(fNPeaks, i2, working_space,
                          working_space[peak_vel + 1]);
               if (yw != 0) {
                  working_space[2 * shift + k] += chi_opt; //der[k]
                  b = a * a / yw;
                  working_space[4 * shift + k] += b; //temp_xk[k]
               }
               k += 1;
            }
            if (fFixBx == false) {
               a = Derbx(fNPeaks, i1, i2,
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
               a = Derby(fNPeaks, i1, i2,
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
         f = Shape2(fNPeaks, i1, i2,
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

////////////////////////////////////////////////////////////////////////////////
///   This function sets the following fitting parameters:
///         - xmin, xmax, ymin, ymax - fitting region
///         - numberIterations - # of desired iterations in the fit
///         - alpha - convergence coefficient, it should be positive number and <=1, for details see references
///         - statisticType - type of statistics, possible values kFitOptimChiCounts (chi square statistics with counts as weighting coefficients), kFitOptimChiFuncValues (chi square statistics with function values as weighting coefficients),kFitOptimMaxLikelihood
///         - alphaOptim - optimization of convergence algorithm, possible values kFitAlphaHalving, kFitAlphaOptimal
///         - power - possible values kFitPower2,4,6,8,10,12, for details see references. It applies only for Awmi fitting function.
///         - fitTaylor - order of Taylor expansion, possible values kFitTaylorOrderFirst, kFitTaylorOrderSecond. It applies only for Awmi fitting function.

void TSpectrum2Fit::SetFitParameters(Int_t xmin,Int_t xmax,Int_t ymin,Int_t ymax, Int_t numberIterations, Double_t alpha, Int_t statisticType, Int_t alphaOptim, Int_t power, Int_t fitTaylor)
{
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

////////////////////////////////////////////////////////////////////////////////
///   This function sets the following fitting parameters of peaks:
///         - sigmaX - initial value of sigma x parameter
///         - fixSigmaX - logical value of sigma x parameter, which allows to fix the parameter (not to fit)
///         - sigmaY - initial value of sigma y parameter
///         - fixSigmaY - logical value of sigma y parameter, which allows to fix the parameter (not to fit)
///         - ro - initial value of ro parameter (correlation coefficient)
///         - fixRo - logical value of ro parameter, which allows to fix the parameter (not to fit)
///         - positionInitX - array of initial values of peaks x positions
///         - fixPositionX - array of logical values which allow to fix appropriate x positions (not fit). However they are present in the estimated functional.
///         - positionInitY - array of initial values of peaks y positions
///         - fixPositionY - array of logical values which allow to fix appropriate y positions (not fit). However they are present in the estimated functional.
///         - ampInit - array of initial values of  2D peaks amplitudes
///         - fixAmp - array of logical values which allow to fix appropriate amplitudes of 2D peaks (not fit). However they are present in the estimated functional
///         - ampInitX1 - array of initial values of amplitudes of  1D ridges in x direction
///         - fixAmpX1 - array of logical values which allow to fix appropriate amplitudes of 1D ridges in x direction (not fit). However they are present in the estimated functional
///         - ampInitY1 - array of initial values of amplitudes of  1D ridges in y direction
///         - fixAmpY1 - array of logical values which allow to fix appropriate amplitudes of 1D ridges in y direction (not fit). However they are present in the estimated functional

void TSpectrum2Fit::SetPeakParameters(Double_t sigmaX, Bool_t fixSigmaX, Double_t sigmaY, Bool_t fixSigmaY, Double_t ro, Bool_t fixRo, const Double_t *positionInitX, const Bool_t *fixPositionX, const Double_t *positionInitY, const Bool_t *fixPositionY, const Double_t *positionInitX1, const Bool_t *fixPositionX1, const Double_t *positionInitY1, const Bool_t *fixPositionY1, const Double_t *ampInit, const Bool_t *fixAmp, const Double_t *ampInitX1, const Bool_t *fixAmpX1, const Double_t *ampInitY1, const Bool_t *fixAmpY1)
{
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
      fPositionInitX[i] = positionInitX[i];
      fFixPositionX[i] = fixPositionX[i];
      fPositionInitY[i] = positionInitY[i];
      fFixPositionY[i] = fixPositionY[i];
      fPositionInitX1[i] = positionInitX1[i];
      fFixPositionX1[i] = fixPositionX1[i];
      fPositionInitY1[i] = positionInitY1[i];
      fFixPositionY1[i] = fixPositionY1[i];
      fAmpInit[i] = ampInit[i];
      fFixAmp[i] = fixAmp[i];
      fAmpInitX1[i] = ampInitX1[i];
      fFixAmpX1[i] = fixAmpX1[i];
      fAmpInitY1[i] = ampInitY1[i];
      fFixAmpY1[i] = fixAmpY1[i];
   }
}

////////////////////////////////////////////////////////////////////////////////
///   This function sets the following fitting parameters of background:
///         - a0Init - initial value of a0 parameter (background is estimated as a0+ax*x+ay*y)
///         - fixA0 - logical value of a0 parameter, which allows to fix the parameter (not to fit)
///         - axInit - initial value of ax parameter
///         - fixAx - logical value of ax parameter, which allows to fix the parameter (not to fit)
///         - ayInit - initial value of ay parameter
///         - fixAy - logical value of ay parameter, which allows to fix the parameter (not to fit)

void TSpectrum2Fit::SetBackgroundParameters(Double_t a0Init, Bool_t fixA0, Double_t axInit, Bool_t fixAx, Double_t ayInit, Bool_t fixAy)
{
   fA0Init = a0Init;
   fFixA0 = fixA0;
   fAxInit = axInit;
   fFixAx = fixAx;
   fAyInit = ayInit;
   fFixAy = fixAy;
}

////////////////////////////////////////////////////////////////////////////////
///   This function sets the following fitting parameters of tails of peaks
///         - tInitXY - initial value of txy parameter
///         - fixTxy - logical value of txy parameter, which allows to fix the parameter (not to fit)
///         - tInitX - initial value of tx parameter
///         - fixTx - logical value of tx parameter, which allows to fix the parameter (not to fit)
///         - tInitY - initial value of ty parameter
///         - fixTy - logical value of ty parameter, which allows to fix the parameter (not to fit)
///         - bInitX - initial value of bx parameter
///         - fixBx - logical value of bx parameter, which allows to fix the parameter (not to fit)
///         - bInitY - initial value of by parameter
///         - fixBy - logical value of by parameter, which allows to fix the parameter (not to fit)
///         - sInitXY - initial value of sxy parameter
///         - fixSxy - logical value of sxy parameter, which allows to fix the parameter (not to fit)
///         - sInitX - initial value of sx parameter
///         - fixSx - logical value of sx parameter, which allows to fix the parameter (not to fit)
///         - sInitY - initial value of sy parameter
///         - fixSy - logical value of sy parameter, which allows to fix the parameter (not to fit)

void TSpectrum2Fit::SetTailParameters(Double_t tInitXY, Bool_t fixTxy, Double_t tInitX, Bool_t fixTx, Double_t tInitY, Bool_t fixTy, Double_t bInitX, Bool_t fixBx, Double_t bInitY, Bool_t fixBy, Double_t sInitXY, Bool_t fixSxy, Double_t sInitX, Bool_t fixSx, Double_t sInitY, Bool_t fixSy)
{
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

////////////////////////////////////////////////////////////////////////////////
///   This function gets the positions of fitted 2D peaks and 1D ridges
///         - positionX - gets vector of x positions of 2D peaks
///         - positionY - gets vector of y positions of 2D peaks
///         - positionX1 - gets vector of x positions of 1D ridges
///         - positionY1 - gets vector of y positions of 1D ridges

void TSpectrum2Fit::GetPositions(Double_t *positionsX, Double_t *positionsY, Double_t *positionsX1, Double_t *positionsY1)
{
   for( Int_t i=0; i < fNPeaks; i++){
      positionsX[i]  = fPositionCalcX[i];
      positionsY[i]  = fPositionCalcY[i];
      positionsX1[i] = fPositionCalcX1[i];
      positionsY1[i] = fPositionCalcY1[i];
   }
}

////////////////////////////////////////////////////////////////////////////////
///   This function gets the errors of positions of fitted 2D peaks and 1D ridges
///         - positionErrorsX - gets vector of errors of x positions of 2D peaks
///         - positionErrorsY - gets vector of errors of y positions of 2D peaks
///         - positionErrorsX1 - gets vector of errors of x positions of 1D ridges
///         - positionErrorsY1 - gets vector of errors of y positions of 1D ridges

void TSpectrum2Fit::GetPositionErrors(Double_t *positionErrorsX, Double_t *positionErrorsY, Double_t *positionErrorsX1, Double_t *positionErrorsY1)
{
   for( Int_t i=0; i < fNPeaks; i++){
      positionErrorsX[i] = fPositionErrX[i];
      positionErrorsY[i] = fPositionErrY[i];
      positionErrorsX1[i] = fPositionErrX1[i];
      positionErrorsY1[i] = fPositionErrY1[i];
   }
}

////////////////////////////////////////////////////////////////////////////////
///   This function gets the amplitudes of fitted 2D peaks and 1D ridges
///         - amplitudes - gets vector of amplitudes of 2D peaks
///         - amplitudesX1 - gets vector of amplitudes of 1D ridges in x direction
///         - amplitudesY1 - gets vector of amplitudes of 1D ridges in y direction

void TSpectrum2Fit::GetAmplitudes(Double_t *amplitudes, Double_t *amplitudesX1, Double_t *amplitudesY1)
{
   for( Int_t i=0; i < fNPeaks; i++){
      amplitudes[i] = fAmpCalc[i];
      amplitudesX1[i] = fAmpCalcX1[i];
      amplitudesY1[i] = fAmpCalcY1[i];
   }
}

////////////////////////////////////////////////////////////////////////////////
///   This function gets the amplitudes of fitted 2D peaks and 1D ridges
///         - amplitudeErrors - gets vector of amplitudes errors of 2D peaks
///         - amplitudeErrorsX1 - gets vector of amplitudes errors of 1D ridges in x direction
///         - amplitudesErrorY1 - gets vector of amplitudes errors of 1D ridges in y direction

void TSpectrum2Fit::GetAmplitudeErrors(Double_t *amplitudeErrors, Double_t *amplitudeErrorsX1, Double_t *amplitudeErrorsY1)
{
   for( Int_t i=0; i < fNPeaks; i++){
      amplitudeErrors[i] = fAmpErr[i];
      amplitudeErrorsX1[i] = fAmpErrX1[i];
      amplitudeErrorsY1[i] = fAmpErrY1[i];
   }
}

////////////////////////////////////////////////////////////////////////////////
///   This function gets the volumes of fitted 2D peaks
///         - volumes - gets vector of volumes of 2D peaks

void TSpectrum2Fit::GetVolumes(Double_t *volumes)
{
   for( Int_t i=0; i < fNPeaks; i++){
      volumes[i] = fVolume[i];
   }
}

////////////////////////////////////////////////////////////////////////////////
///   This function gets errors of the volumes of fitted 2D peaks
///         - volumeErrors - gets vector of volumes errors of 2D peaks

void TSpectrum2Fit::GetVolumeErrors(Double_t *volumeErrors)
{
   for( Int_t i=0; i < fNPeaks; i++){
      volumeErrors[i] = fVolumeErr[i];
   }
}

////////////////////////////////////////////////////////////////////////////////
///   This function gets the sigma x parameter and its error
///         - sigmaX - gets the fitted value of sigma x parameter
///         - sigmaErrX - gets error value of sigma x parameter

void TSpectrum2Fit::GetSigmaX(Double_t &sigmaX, Double_t &sigmaErrX)
{
   sigmaX=fSigmaCalcX;
   sigmaErrX=fSigmaErrX;
}

////////////////////////////////////////////////////////////////////////////////
///   This function gets the sigma y parameter and its error
///         - sigmaY - gets the fitted value of sigma y parameter
///         - sigmaErrY - gets error value of sigma y parameter

void TSpectrum2Fit::GetSigmaY(Double_t &sigmaY, Double_t &sigmaErrY)
{
   sigmaY=fSigmaCalcY;
   sigmaErrY=fSigmaErrY;
}

////////////////////////////////////////////////////////////////////////////////
///   This function gets the ro parameter and its error
///         - ro - gets the fitted value of ro parameter
///         - roErr - gets error value of ro parameter

void TSpectrum2Fit::GetRo(Double_t &ro, Double_t &roErr)
{
   ro=fRoCalc;
   roErr=fRoErr;
}

////////////////////////////////////////////////////////////////////////////////
///   This function gets the background parameters and their errors
///         - a0 - gets the fitted value of a0 parameter
///         - a0Err - gets error value of a0 parameter
///         - ax - gets the fitted value of ax parameter
///         - axErr - gets error value of ax parameter
///         - ay - gets the fitted value of ay parameter
///         - ayErr - gets error value of ay parameter

void TSpectrum2Fit::GetBackgroundParameters(Double_t &a0, Double_t &a0Err, Double_t &ax, Double_t &axErr, Double_t &ay, Double_t &ayErr)
{
   a0 = fA0Calc;
   a0Err = fA0Err;
   ax = fAxCalc;
   axErr = fAxErr;
   ay = fAyCalc;
   ayErr = fAyErr;
}

////////////////////////////////////////////////////////////////////////////////
///   This function gets the tail parameters and their errors
///         - txy - gets the fitted value of txy parameter
///         - txyErr - gets error value of txy parameter
///         - tx - gets the fitted value of tx parameter
///         - txErr - gets error value of tx parameter
///         - ty - gets the fitted value of ty parameter
///         - tyErr - gets error value of ty parameter
///         - bx - gets the fitted value of bx parameter
///         - bxErr - gets error value of bx parameter
///         - by - gets the fitted value of by parameter
///         - byErr - gets error value of by parameter
///         - sxy - gets the fitted value of sxy parameter
///         - sxyErr - gets error value of sxy parameter
///         - sx - gets the fitted value of sx parameter
///         - sxErr - gets error value of sx parameter
///         - sy - gets the fitted value of sy parameter
///         - syErr - gets error value of sy parameter

void TSpectrum2Fit::GetTailParameters(Double_t &txy, Double_t &txyErr, Double_t &tx, Double_t &txErr, Double_t &ty, Double_t &tyErr, Double_t &bx, Double_t &bxErr, Double_t &by, Double_t &byErr, Double_t &sxy, Double_t &sxyErr, Double_t &sx, Double_t &sxErr, Double_t &sy, Double_t &syErr)
{
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

