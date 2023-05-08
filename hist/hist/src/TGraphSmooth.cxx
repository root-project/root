// @(#)root/hist:$Id$
// Author: Christian Stratowa 30/09/2001

/*************************************************************************
 * Copyright (C) 2006, Rene Brun and Fons Rademakers.                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/******************************************************************************
* Copyright(c) 2001-    , Dr. Christian Stratowa, Vienna, Austria.            *
* Author: Christian Stratowa with help from Rene Brun.                        *
*                                                                             *
* Algorithms for smooth regression adapted from:                              *
* R: A Computer Language for Statistical Data Analysis                        *
* Copyright (C) 1996 Robert Gentleman and Ross Ihaka                          *
* Copyright (C) 1999-2001 Robert Gentleman, Ross Ihaka and the                *
* R Development Core Team                                                     *
* R is free software, for licensing see the GNU General Public License        *
* http://www.ci.tuwien.ac.at/R-project/welcome.html                           *
*                                                                             *
******************************************************************************/


#include "Riostream.h"
#include "TMath.h"
#include "TGraphSmooth.h"
#include "TGraphErrors.h"

ClassImp(TGraphSmooth);

//______________________________________________________________________
/** \class TGraphSmooth
    \ingroup Graph
A helper class to smooth TGraph.
see examples in $ROOTSYS/tutorials/graphs/motorcycle.C and approx.C
*/

TGraphSmooth::TGraphSmooth(): TNamed()
{
   fNin  = 0;
   fNout = 0;
   fGin  = 0;
   fGout = 0;
   fMinX = 0;
   fMaxX = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// GraphSmooth constructor

TGraphSmooth::TGraphSmooth(const char *name): TNamed(name,"")
{
   fNin  = 0;
   fNout = 0;
   fGin  = 0;
   fGout = 0;
   fMinX = 0;
   fMaxX = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// GraphSmooth destructor

TGraphSmooth::~TGraphSmooth()
{
   if (fGout) delete fGout;
   fGin  = 0;
   fGout = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Sort input data points

void TGraphSmooth::Smoothin(TGraph *grin)
{
   if (fGout) {delete fGout; fGout = 0;}
   fGin = grin;

   fNin = fGin->GetN();
   Double_t *xin = new Double_t[fNin];
   Double_t *yin = new Double_t[fNin];
   Int_t i;
   for (i=0;i<fNin;i++) {
      xin[i] = fGin->GetX()[i];
      yin[i] = fGin->GetY()[i];
   }

// sort input x, y
   Int_t *index = new Int_t[fNin];
   TMath::Sort(fNin, xin, index, kFALSE);
   for (i=0;i<fNin;i++) {
      fGin->SetPoint(i, xin[index[i]], yin[index[i]]);
   }

   fMinX = fGin->GetX()[0];  //already sorted!
   fMaxX = fGin->GetX()[fNin-1];

   delete [] index;
   delete [] xin;
   delete [] yin;
}

////////////////////////////////////////////////////////////////////////////////
/// Smooth data with Kernel smoother. Smooth grin with the Nadaraya-Watson kernel regression estimate.
///
/// \param[in] grin input graph
/// \param[in] option the kernel to be used: "box", "normal"
/// \param[in] bandwidth the bandwidth. The kernels are scaled so that their quartiles
///    (viewed as probability densities) are at +/- 0.25*bandwidth.
/// \param[in] nout If xout is not specified, interpolation takes place at equally
///     spaced points spanning the interval [min(x), max(x)], where nout = max(nout, number of input data).
/// \param[in] xout an optional set of values at which to evaluate the fit

TGraph *TGraphSmooth::SmoothKern(TGraph *grin, Option_t *option,
                      Double_t bandwidth, Int_t nout, Double_t *xout)
{
   TString opt = option;
   opt.ToLower();
   Int_t kernel = 1;
   if (opt.Contains("normal")) kernel = 2;

   Smoothin(grin);

   Double_t delta = 0;
   Int_t *index = 0;
   if (xout == 0) {
      fNout = TMath::Max(nout, fNin);
      delta = (fMaxX - fMinX)/(fNout - 1);
   } else {
      fNout = nout;
      index = new Int_t[nout];
      TMath::Sort(nout, xout, index, kFALSE);
   }

   fGout = new TGraph(fNout);
   for (Int_t i=0;i<fNout;i++) {
      if (xout == 0) fGout->SetPoint(i,fMinX + i*delta, 0);
      else           fGout->SetPoint(i,xout[index[i]], 0);
   }

   BDRksmooth(fGin->GetX(), fGin->GetY(), fNin, fGout->GetX(),
                 fGout->GetY(), fNout, kernel, bandwidth);

   if (index) {delete [] index; index = 0;}

   return fGout;
}

////////////////////////////////////////////////////////////////////////////////
/// Smooth data with specified kernel.
/// Based on R function ksmooth: Translated to C++ by C. Stratowa
/// (R source file: ksmooth.c by B.D.Ripley Copyright (C) 1998)

void TGraphSmooth::BDRksmooth(Double_t *x, Double_t *y, Int_t n, Double_t *xp,
                   Double_t *yp, Int_t np, Int_t kernel, Double_t bw)
{
   Int_t    imin = 0;
   Double_t cutoff = 0.0;

// bandwidth is in units of half inter-quartile range
   if (kernel == 1) {
      bw *= 0.5;
      cutoff = bw;
   }
   if (kernel == 2) {
      bw *= 0.3706506;
      cutoff = 4*bw;
   }

   while ((imin < n) && (x[imin] < xp[0] - cutoff))
      imin++;

   for (Int_t j=0;j<np;j++) {
      Double_t xx, w;
      Double_t num = 0.0;
      Double_t den = 0.0;
      Double_t x0 = xp[j];
      for (Int_t i=imin;i<n;i++) {
         if (x[i] < x0 - cutoff) imin = i;
         if (x[i] > x0 + cutoff) break;
         xx = TMath::Abs(x[i] - x0)/bw;
         if (kernel == 1) w = 1;
         else             w = TMath::Exp(-0.5*xx*xx);
         num += w*y[i];
         den += w;
      }
      if (den > 0) {
         yp[j] = num/den;
      } else {
         yp[j] = 0; //should be NA_REAL (see R.h) or nan("NAN")
      }
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Smooth data with Lowess smoother
///
/// This function performs the computations for the LOWESS smoother
/// (see the reference below). Lowess returns the output points
/// x and y which give the coordinates of the smooth.
///
/// \param[in] grin Input graph
/// \param[in] option specific options
/// \param[in] span the smoother span. This gives the proportion of points in the plot
///     which influence the smooth at each value. Larger values give more smoothness.
/// \param[in] iter the number of robustifying iterations which should be performed.
///     Using smaller values of iter will make lowess run faster.
/// \param[in] delta values of x which lie within delta of each other replaced by a
///     single value in the output from lowess.
///     For delta = 0, delta will be calculated.
///
/// References:
///
/// - Cleveland, W. S. (1979) Robust locally weighted regression and smoothing
///        scatterplots. J. Amer. Statist. Assoc. 74, 829-836.
/// - Cleveland, W. S. (1981) LOWESS: A program for smoothing scatterplots
///        by robust locally weighted regression.
///        The American Statistician, 35, 54.

TGraph *TGraphSmooth::SmoothLowess(TGraph *grin, Option_t *option ,
                      Double_t span, Int_t iter, Double_t delta)
{
   TString opt = option;
   opt.ToLower();

   Smoothin(grin);

   if (delta == 0) {delta = 0.01*(TMath::Abs(fMaxX - fMinX));}

// output X, Y
   fNout = fNin;
   fGout = new TGraphErrors(fNout);

   for (Int_t i=0;i<fNout;i++) {
      fGout->SetPoint(i,fGin->GetX()[i], 0);
   }

   Lowess(fGin->GetX(), fGin->GetY(), fNin, fGout->GetY(), span, iter, delta);

   return fGout;
}

////////////////////////////////////////////////////////////////////////////////
/// Lowess regression smoother.
/// Based on R function clowess: Translated to C++ by C. Stratowa
/// (R source file: lowess.c by R Development Core Team (C) 1999-2001)

void TGraphSmooth::Lowess(Double_t *x, Double_t *y, Int_t n, Double_t *ys,
                   Double_t span, Int_t iter, Double_t delta)
{
   Int_t    i, iiter, j, last, m1, m2, nleft, nright, ns;
   Double_t alpha, c1, c9, cmad, cut, d1, d2, denom, r;
   Bool_t   ok;

   if (n < 2) {
      ys[0] = y[0];
      return;
   }

// nleft, nright, last, etc. must all be shifted to get rid of these:
   x--;
   y--;
   ys--;

   Double_t *rw  = ((TGraphErrors*)fGout)->GetEX();
   Double_t *res = ((TGraphErrors*)fGout)->GetEY();

// at least two, at most n poInt_ts
   ns = TMath::Max(2, TMath::Min(n, (Int_t)(span*n + 1e-7)));

// robustness iterations
   iiter = 1;
   while (iiter <= iter+1) {
      nleft = 1;
      nright = ns;
      last = 0;   // index of prev estimated poInt_t
      i = 1;      // index of current poInt_t

      for(;;) {
         if (nright < n) {
         // move nleft,  nright to right if radius decreases
            d1 = x[i] - x[nleft];
            d2 = x[nright+1] - x[i];

         // if d1 <= d2 with x[nright+1] == x[nright], lowest fixes
            if (d1 > d2) {
            // radius will not decrease by move right
               nleft++;
               nright++;
               continue;
            }
         }

      // fitted value at x[i]
         Bool_t iterg1 = iiter>1;
         Lowest(&x[1], &y[1], n, x[i], ys[i], nleft, nright,
                      res, iterg1, rw, ok);
         if (!ok) ys[i] = y[i];

      // all weights zero copy over value (all rw==0)
         if (last < i-1) {
            denom = x[i]-x[last];

         // skipped poInt_ts -- Int_terpolate non-zero - proof?
            for(j = last+1; j < i; j++) {
               alpha = (x[j]-x[last])/denom;
               ys[j] = alpha*ys[i] + (1.-alpha)*ys[last];
            }
      }

      // last poInt_t actually estimated
         last = i;

      // x coord of close poInt_ts
         cut = x[last] + delta;
         for (i = last+1; i <= n; i++) {
            if (x[i] > cut)
               break;
            if (x[i] == x[last]) {
               ys[i] = ys[last];
               last = i;
            }
         }
         i = TMath::Max(last+1, i-1);
         if (last >= n)
            break;
      }

   // residuals
      for(i=0; i < n; i++)
         res[i] = y[i+1] - ys[i+1];

   // compute robustness weights except last time
      if (iiter > iter)
         break;
      for(i=0 ; i<n ; i++)
         rw[i] = TMath::Abs(res[i]);

   // compute cmad := 6 * median(rw[], n)
      m1 = n/2;
   // partial sort, for m1 & m2
      Psort(rw, n, m1);
      if(n % 2 == 0) {
         m2 = n-m1-1;
         Psort(rw, n, m2);
         cmad = 3.*(rw[m1]+rw[m2]);
      } else { /* n odd */
         cmad = 6.*rw[m1];
      }

      c9 = 0.999*cmad;
      c1 = 0.001*cmad;
      for(i=0 ; i<n ; i++) {
         r = TMath::Abs(res[i]);
         if (r <= c1)
            rw[i] = 1.;
         else if (r <= c9)
            rw[i] = (1.-(r/cmad)*(r/cmad))*(1.-(r/cmad)*(r/cmad));
         else
            rw[i] = 0.;
      }
      iiter++;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Fit value at x[i]
///  Based on R function lowest: Translated to C++ by C. Stratowa
///  (R source file: lowess.c by R Development Core Team (C) 1999-2001)

void TGraphSmooth::Lowest(Double_t *x, Double_t *y, Int_t n, Double_t &xs,
                   Double_t &ys, Int_t nleft, Int_t nright, Double_t *w,
                   Bool_t userw, Double_t *rw, Bool_t &ok)
{
   Int_t    nrt, j;
   Double_t a, b, c, d, h, h1, h9, r, range;

   x--;
   y--;
   w--;
   rw--;

   range = x[n]-x[1];
   h = TMath::Max(xs-x[nleft], x[nright]-xs);
   h9 = 0.999*h;
   h1 = 0.001*h;

// sum of weights
   a = 0.;
   j = nleft;
   while (j <= n) {
   // compute weights (pick up all ties on right)
      w[j] = 0.;
      r = TMath::Abs(x[j] - xs);
      if (r <= h9) {
         if (r <= h1) {
            w[j] = 1.;
         } else {
            d = (r/h)*(r/h)*(r/h);
            w[j] = (1.- d)*(1.- d)*(1.- d);
         }
         if (userw)
            w[j] *= rw[j];
         a += w[j];
      } else if (x[j] > xs)
         break;
      j = j+1;
   }

// rightmost pt (may be greater than nright because of ties)
   nrt = j-1;
   if (a <= 0.)
      ok = kFALSE;
   else {
      ok = kTRUE;
   // weighted least squares: make sum of w[j] == 1
      for(j=nleft ; j<=nrt ; j++)
         w[j] /= a;
      if (h > 0.) {
         a = 0.;
      // use linear fit weighted center of x values
         for(j=nleft ; j<=nrt ; j++)
            a += w[j] * x[j];
         b = xs - a;
         c = 0.;
         for(j=nleft ; j<=nrt ; j++)
            c += w[j]*(x[j]-a)*(x[j]-a);
         if (TMath::Sqrt(c) > 0.001*range) {
            b /= c;
         // poInt_ts are spread out enough to compute slope
            for(j=nleft; j <= nrt; j++)
               w[j] *= (b*(x[j]-a) + 1.);
         }
      }
      ys = 0.;
      for(j=nleft; j <= nrt; j++)
         ys += w[j] * y[j];
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Smooth data with Super smoother.
/// Smooth the (x, y) values by Friedman's ``super smoother''.
///
/// \param[in] grin graph for smoothing
/// \param[in] option specific options
/// \param[in] span the fraction of the observations in the span of the running lines
///    smoother, or 0 to choose this by leave-one-out cross-validation.
/// \param[in] bass controls the smoothness of the fitted curve.
///    Values of up to 10 indicate increasing smoothness.
/// \param[in] isPeriodic if TRUE, the x values are assumed to be in [0, 1]
///    and of period 1.
/// \param[in] w case weights
///
/// Details:
///
/// supsmu is a running lines smoother which chooses between three spans for
/// the lines. The running lines smoothers are symmetric, with k/2 data points
/// each side of the predicted point, and values of k as 0.5 * n, 0.2 * n and
/// 0.05 * n, where n is the number of data points. If span is specified,
/// a single smoother with span span * n is used.
///
/// The best of the three smoothers is chosen by cross-validation for each
/// prediction. The best spans are then smoothed by a running lines smoother
/// and the final prediction chosen by linear interpolation.
///
/// The FORTRAN code says: ``For small samples (n < 40) or if there are
/// substantial serial correlations between observations close in x - value,
/// then a prespecified fixed span smoother (span > 0) should be used.
/// Reasonable span values are 0.2 to 0.4.''
///
/// References:
/// - Friedman, J. H. (1984) SMART User's Guide.
///           Laboratory for Computational Statistics,
///           Stanford University Technical Report No. 1.
/// - Friedman, J. H. (1984) A variable span scatterplot smoother.
///           Laboratory for Computational Statistics,
///           Stanford University Technical Report No. 5.

TGraph *TGraphSmooth::SmoothSuper(TGraph *grin, Option_t * option,
        Double_t bass, Double_t span, Bool_t isPeriodic, Double_t *w)
{
   if (span < 0 || span > 1) {
      std::cout << "Error: Span must be between 0 and 1" << std::endl;
      return 0;
   }
   TString opt = option;
   opt.ToLower();

   Smoothin(grin);

   Int_t iper = 1;
   if (isPeriodic) {
      iper = 2;
      if (fMinX < 0 || fMaxX > 1) {
         std::cout << "Error: x must be between 0 and 1 for periodic smooth" << std::endl;
         return 0;
      }
   }

// output X, Y
   fNout = fNin;
   fGout = new TGraph(fNout);
   Int_t i;
   for (i=0; i<fNout; i++) {
      fGout->SetPoint(i,fGin->GetX()[i], 0);
   }

// weights
   Double_t *weight = new Double_t[fNin];
   for (i=0; i<fNin; i++) {
      if (w == 0) weight[i] = 1;
      else        weight[i] = w[i];
   }

// temporary storage array
   Int_t nTmp = (fNin+1)*8;
   Double_t *tmp = new Double_t[nTmp];
   for (i=0; i<nTmp; i++) {
      tmp[i] = 0;
   }

   BDRsupsmu(fNin, fGin->GetX(), fGin->GetY(), weight, iper, span, bass, fGout->GetY(), tmp);

   delete [] tmp;
   delete [] weight;

   return fGout;
}

////////////////////////////////////////////////////////////////////////////////
/// Friedmanns super smoother (Friedman, 1984).
///
///  version 10/10/84
///  coded and copyright (c) 1984 by:
///
///                         Jerome H. Friedman
///                      department of statistics
///                                and
///                 stanford linear accelerator center
///                         stanford university
///
///  all rights reserved.
///
///  \param[in] n number of observations (x,y - pairs).
///  \param[in] x ordered abscissa values.
///  \param[in] y corresponding ordinate (response) values.
///  \param[in] w weight for each (x,y) observation.
///  \param[in] iper periodic variable flag.
///     - iper=1 => x is ordered interval variable.
///     - iper=2 => x is a periodic variable with values
///       in the range (0.0,1.0) and period 1.0.
///  \param[in] span smoother span (fraction of observations in window).
///     - span=0.0 => automatic (variable) span selection.
///  \param[in] alpha controls high frequency (small span) penality
///     used with automatic span selection (bass tone control).
///     (alpha.le.0.0 or alpha.gt.10.0 => no effect.)
///  \param[out] smo smoothed ordinate (response) values.
///  \param sc internal working storage.
///
///  note:
///
///     for small samples (n < 40) or if there are substantial serial
///     correlations between observations close in x - value, then
///     a prespecified fixed span smoother (span > 0) should be
///     used. reasonable span values are 0.2 to 0.4.
///
/// current implementation:
///
///   Based on R function supsmu: Translated to C++ by C. Stratowa
///   (R source file: ppr.f by B.D.Ripley Copyright (C) 1994-97)

void TGraphSmooth::BDRsupsmu(Int_t n, Double_t *x, Double_t *y, Double_t *w,
     Int_t iper, Double_t span, Double_t alpha, Double_t *smo, Double_t *sc)
{
// Local variables
   Int_t sc_offset;
   Int_t i, j, jper;
   Double_t a, f;
   Double_t sw, sy, resmin, vsmlsq;
   Double_t scale;
   Double_t d1, d2;

   Double_t spans[3] = { 0.05, 0.2, 0.5 };
   Double_t big = 1e20;
   Double_t sml = 1e-7;
   Double_t eps = 0.001;

// Parameter adjustments
   sc_offset = n + 1;
   sc -= sc_offset;
   --smo;
   --w;
   --y;
   --x;

// Function Body
   if (x[n] <= x[1]) {
      sy = 0.0;
      sw = sy;
      for (j=1;j<=n;++j) {
         sy += w[j] * y[j];
         sw += w[j];
      }

      a = 0.0;
      if (sw > 0.0) a = sy / sw;
      for (j=1;j<=n;++j) smo[j] = a;
      return;
   }

   i = (Int_t)(n / 4);
   j = i * 3;
   scale = x[j] - x[i];
   while (scale <= 0.0) {
      if (j < n) ++j;
      if (i > 1) --i;
      scale = x[j] - x[i];
   }

// Computing 2nd power
   d1 = eps * scale;
   vsmlsq = d1 * d1;
   jper = iper;
   if (iper == 2 && (x[1] < 0.0 || x[n] > 1.0)) {
      jper = 1;
   }
   if (jper < 1 || jper > 2) {
      jper = 1;
   }
   if (span > 0.0) {
      BDRsmooth(n, &x[1], &y[1], &w[1], span, jper, vsmlsq,
                      &smo[1], &sc[sc_offset]);
      return;
   }

   Double_t *h = new Double_t[n+1];
   for (i = 1; i <= 3; ++i) {
      BDRsmooth(n, &x[1], &y[1], &w[1], spans[i - 1], jper, vsmlsq,
                      &sc[((i<<1)-1)*n + 1], &sc[n*7 + 1]);
      BDRsmooth(n, &x[1], &sc[n*7 + 1], &w[1], spans[1], -jper, vsmlsq,
                      &sc[(i<<1)*n + 1], &h[1]);
   }

   for (j=1; j<=n; ++j) {
      resmin = big;
      for (i=1; i<=3; ++i) {
         if (sc[j + (i<<1)*n] < resmin) {
            resmin = sc[j + (i<<1)*n];
            sc[j + n*7] = spans[i-1];
         }
      }

      if (alpha>0.0 && alpha<=10.0 && resmin<sc[j + n*6]  && resmin>0.0) {
      // Computing MAX
         d1 = TMath::Max(sml,(resmin/sc[j + n*6]));
         d2 = 10. - alpha;
         sc[j + n*7] += (spans[2] - sc[j + n*7]) * TMath::Power(d1, d2);
      }
   }

   BDRsmooth(n, &x[1], &sc[n*7 + 1], &w[1], spans[1], -jper, vsmlsq,
                   &sc[(n<<1) + 1], &h[1]);

   for (j=1; j<=n; ++j) {
      if (sc[j + (n<<1)] <= spans[0]) {
         sc[j + (n<<1)] = spans[0];
      }
      if (sc[j + (n<<1)] >= spans[2]) {
         sc[j + (n<<1)] = spans[2];
      }
      f = sc[j + (n<<1)] - spans[1];
      if (f < 0.0) {
         f = -f / (spans[1] - spans[0]);
         sc[j + (n<<2)] = (1.0 - f) * sc[j + n*3] + f * sc[j + n];
      } else {
         f /= spans[2] - spans[1];
         sc[j + (n<<2)] = (1.0 - f) * sc[j + n*3] + f * sc[j + n*5];
      }
   }

   BDRsmooth(n, &x[1], &sc[(n<<2) + 1], &w[1], spans[0], -jper, vsmlsq,
                   &smo[1], &h[1]);

   delete [] h;
   return;
}

////////////////////////////////////////////////////////////////////////////////
/// Function for super smoother
/// Based on R function supsmu: Translated to C++ by C. Stratowa
/// (R source file: ppr.f by B.D.Ripley Copyright (C) 1994-97)

void TGraphSmooth::BDRsmooth(Int_t n, Double_t *x, Double_t *y, Double_t *w,
     Double_t span, Int_t iper, Double_t vsmlsq, Double_t *smo, Double_t *acvr)
{
// Local variables
   Int_t i, j, j0, in, out, it, jper, ibw;
   Double_t a, h1, d1;
   Double_t xm, ym, wt, sy, fbo, fbw;
   Double_t cvar, var, tmp, xti, xto;

// Parameter adjustments
   --acvr;
   --smo;
   --w;
   --y;
   --x;

// Function Body
   xm = 0.;
   ym = xm;
   var = ym;
   cvar = var;
   fbw = cvar;
   jper = TMath::Abs(iper);

   ibw = (Int_t)(span * 0.5 * n + 0.5);
   if (ibw < 2) {
      ibw = 2;
   }

   it = 2*ibw + 1;
   for (i=1; i<=it; ++i) {
      j = i;
      if (jper == 2) {
         j = i - ibw - 1;
      }
      xti = x[j];
      if (j < 1) {
         j = n + j;
         xti = x[j] - 1.0;
      }
      wt = w[j];
      fbo = fbw;
      fbw += wt;
      if (fbw > 0.0) {
         xm = (fbo * xm + wt * xti) / fbw;
         ym = (fbo * ym + wt * y[j]) / fbw;
      }
      tmp = 0.0;
      if (fbo > 0.0) {
         tmp = fbw * wt * (xti - xm) / fbo;
      }
      var += tmp * (xti - xm);
      cvar += tmp * (y[j] - ym);
   }

   for (j=1; j<=n; ++j) {
      out = j - ibw - 1;
      in = j + ibw;
      if (!(jper != 2 && (out < 1 || in > n))) {
         if (out < 1) {
            out = n + out;
            xto = x[out] - 1.0;
            xti = x[in];
         } else if (in > n) {
            in -= n;
            xti = x[in] + 1.0;
            xto = x[out];
         } else {
            xto = x[out];
            xti = x[in];
         }

         wt = w[out];
         fbo = fbw;
         fbw -= wt;
         tmp = 0.0;
         if (fbw > 0.0) {
            tmp = fbo * wt * (xto - xm) / fbw;
         }
         var -= tmp * (xto - xm);
         cvar -= tmp * (y[out] - ym);
         if (fbw > 0.0) {
            xm = (fbo * xm - wt * xto) / fbw;
            ym = (fbo * ym - wt * y[out]) / fbw;
         }
         wt = w[in];
         fbo = fbw;
         fbw += wt;
         if (fbw > 0.0) {
            xm = (fbo * xm + wt * xti) / fbw;
            ym = (fbo * ym + wt * y[in]) / fbw;
         }
         tmp = 0.0;
         if (fbo > 0.0) {
            tmp = fbw * wt * (xti - xm) / fbo;
         }
         var += tmp * (xti - xm);
         cvar += tmp * (y[in] - ym);
      }

      a = 0.0;
      if (var > vsmlsq) {
         a = cvar / var;
      }
      smo[j] = a * (x[j] - xm) + ym;

      if (iper <= 0) {
         continue;
      }

      h1 = 0.0;
      if (fbw > 0.0) {
         h1 = 1.0 / fbw;
      }
      if (var > vsmlsq) {
      // Computing 2nd power
         d1 = x[j] - xm;
         h1 += d1 * d1 / var;
      }

      acvr[j] = 0.0;
      a = 1.0 - w[j] * h1;
      if (a > 0.0) {
         acvr[j] = TMath::Abs(y[j] - smo[j]) / a;
         continue;
      }
      if (j > 1) {
         acvr[j] = acvr[j-1];
      }
   }

   j = 1;
   do {
      j0 = j;
      sy = smo[j] * w[j];
      fbw = w[j];
      if (j < n) {
         do {
            if (x[j + 1] > x[j]) {
               break;
            }
            ++j;
            sy += w[j] * smo[j];
            fbw += w[j];
         } while (j < n);
      }

      if (j > j0) {
         a = 0.0;
         if (fbw > 0.0) {
            a = sy / fbw;
         }
         for (i=j0; i<=j; ++i) {
            smo[i] = a;
         }
      }
      ++j;
   } while (j <= n);

   return;
}

////////////////////////////////////////////////////////////////////////////////
/// Sort data points and eliminate double x values

void TGraphSmooth::Approxin(TGraph *grin, Int_t /*iKind*/, Double_t &ylow,
     Double_t &yhigh, Int_t rule, Int_t iTies)
{
   if (fGout) {delete fGout; fGout = 0;}
   fGin = grin;

   fNin = fGin->GetN();
   Double_t *xin = new Double_t[fNin];
   Double_t *yin = new Double_t[fNin];
   Int_t i;
   for (i=0;i<fNin;i++) {
      xin[i] = fGin->GetX()[i];
      yin[i] = fGin->GetY()[i];
   }

// sort/rank input x, y
   Int_t *index = new Int_t[fNin];
   Int_t *rank  = new Int_t[fNin];
   Rank(fNin, xin, index, rank, kFALSE);

// input X, Y
   Int_t vNDup = 0;
   Int_t k = 0;
   Int_t *dup = new Int_t[fNin];
   Double_t *x = new Double_t[fNin];
   Double_t *y = new Double_t[fNin];
   Double_t vMean, vMin, vMax;
   for (i=1;i<fNin+1;i++) {
      Int_t ndup = 1;
      vMin = vMean = vMax = yin[index[i-1]];
      while ((i < fNin) && (rank[index[i]] == rank[index[i-1]])) {
         vMean += yin[index[i]];
         vMax = (vMax < yin[index[i]]) ? yin[index[i]] : vMax;
         vMin = (vMin > yin[index[i]]) ? yin[index[i]] : vMin;
         dup[vNDup] = i;
         i++;
         ndup++;
         vNDup++;
      }
      x[k] = xin[index[i-1]];
      if (ndup == 1) {y[k++] = yin[index[i-1]];}
      else switch(iTies) {
         case 1:
            y[k++] = vMean/ndup;
            break;
         case 2:
            y[k++] = vMin;
            break;
         case 3:
            y[k++] = vMax;
            break;
         default:
            y[k++] = vMean/ndup;
            break;
      }
   }
   fNin = k;

// set unique sorted input data x,y as final graph points
   fGin->Set(fNin);
   for (i=0;i<fNin;i++) {
      fGin->SetPoint(i, x[i], y[i]);
   }

   fMinX = fGin->GetX()[0];  //already sorted!
   fMaxX = fGin->GetX()[fNin-1];

// interpolate outside interval [min(x),max(x)]
   switch(rule) {
      case 1:
         ylow  = 0;   // = nan("NAN") ??
         yhigh = 0;   // = nan("NAN") ??
         break;
      case 2:
         ylow  = fGin->GetY()[0];
         yhigh = fGin->GetY()[fNin-1];
         break;
      default:
         break;
   }

// cleanup
   delete [] x;
   delete [] y;
   delete [] dup;
   delete [] rank;
   delete [] index;
   delete [] xin;
   delete [] yin;
}

////////////////////////////////////////////////////////////////////////////////
/// Approximate data points
/// \param[in] grin graph giving the coordinates of the points to be interpolated.
///    Alternatively a single plotting structure can be specified:
/// \param[in] option specifies the interpolation method to be used.
///    Choices are "linear" (iKind = 1) or "constant" (iKind = 2).
/// \param[in] nout If xout is not specified, interpolation takes place at n equally
///    spaced points spanning the interval [min(x), max(x)], where
///    nout = max(nout, number of input data).
/// \param[in] xout  an optional set of values specifying where interpolation is to
///    take place.
/// \param[in] yleft the value to be returned when input x values less than min(x).
///            The default is defined by the value of rule given below.
/// \param[in] yright the value to be returned when input x values greater than max(x).
///            The default is defined by the value of rule given below.
/// \param[in] rule an integer describing how interpolation is to take place outside
///            the interval [min(x), max(x)]. If rule is 0 then the given yleft
///            and yright values are returned, if it is 1 then 0 is returned
///            for such points and if it is 2, the value at the closest data
///            extreme is used.
/// \param[in] f For method="constant" a number between 0 and 1 inclusive,
///            indicating a compromise between left- and right-continuous step
///            functions. If y0 and y1 are the values to the left and right of
///            the point then the value is y0*f+y1*(1-f) so that f=0 is
///            right-continuous and f=1 is left-continuous
/// \param[in] ties Handling of tied x values. An integer describing a function with
///            a single vector argument returning a single number result:
///            - ties = "ordered" (iTies = 0): input x are "ordered"
///            - ties = "mean"    (iTies = 1): function "mean"
///            - ties = "min"     (iTies = 2): function "min"
///            - ties = "max"     (iTies = 3): function "max"
///
/// Details:
///
/// At least two complete (x, y) pairs are required.
/// If there are duplicated (tied) x values and ties is a function it is
/// applied to the y values for each distinct x value. Useful functions in
/// this context include mean, min, and max.
/// If ties="ordered" the x values are assumed to be already ordered. The
/// first y value will be used for interpolation to the left and the last
/// one for interpolation to the right.
///
/// Value:
///
/// approx returns a graph with components x and y, containing n coordinates
/// which interpolate the given data points according to the method (and rule)
/// desired.

TGraph *TGraphSmooth::Approx(TGraph *grin, Option_t *option, Int_t nout, Double_t *xout,
        Double_t yleft, Double_t yright, Int_t rule, Double_t f, Option_t *ties)
{
   TString opt = option;
   opt.ToLower();
   Int_t iKind = 0;
   if (opt.Contains("linear")) iKind = 1;
   else if (opt.Contains("constant")) iKind = 2;

   if (f < 0 || f > 1) {
      std::cout << "Error: Invalid f value" << std::endl;
      return 0;
   }

   opt = ties;
   opt.ToLower();
   Int_t iTies = 0;
   if (opt.Contains("ordered")) {
      iTies = 0;
   } else if (opt.Contains("mean")) {
      iTies = 1;
   } else if (opt.Contains("min")) {
      iTies = 2;
   } else if (opt.Contains("max")) {
      iTies = 3;
   } else {
      std::cout << "Error: Method not known: " << ties << std::endl;
      return 0;
   }

// input X, Y
   Double_t ylow  = yleft;
   Double_t yhigh = yright;
   Approxin(grin, iKind, ylow, yhigh, rule, iTies);

// output X, Y
   Double_t delta = 0;
   fNout = nout;
   if (xout == 0) {
      fNout = TMath::Max(nout, fNin);
      delta = (fMaxX - fMinX)/(fNout - 1);
   }

   fGout = new TGraph(fNout);

   Double_t x;
   for (Int_t i=0;i<fNout;i++) {
      if (xout == 0) x = fMinX + i*delta;
      else           x = xout[i];
      Double_t yout = Approx1(x, f, fGin->GetX(), fGin->GetY(), fNin, iKind, ylow, yhigh);
      fGout->SetPoint(i,x, yout);
   }

   return fGout;
}

////////////////////////////////////////////////////////////////////////////////
/// Approximate one data point.
/// Approximate  y(v),  given (x,y)[i], i = 0,..,n-1
/// Based on R function approx1: Translated to C++ by Christian Stratowa
/// (R source file: approx.c by R Development Core Team (C) 1999-2001)

Double_t TGraphSmooth::Approx1(Double_t v, Double_t f, Double_t *x, Double_t *y,
         Int_t n, Int_t iKind, Double_t ylow, Double_t yhigh)
{
   Int_t i = 0;
   Int_t j = n - 1;

// handle out-of-domain points
   if(v < x[i]) return ylow;
   if(v > x[j]) return yhigh;

// find the correct interval by bisection
   while(i < j - 1) {
      Int_t ij = (i + j)/2;
      if(v < x[ij]) j = ij;
      else i = ij;
   }

// interpolation
   if(v == x[j]) return y[j];
   if(v == x[i]) return y[i];

   if(iKind == 1) { // linear
      return y[i] + (y[j] - y[i]) * ((v - x[i])/(x[j] - x[i]));
   } else { // 2 : constant
      return y[i] * (1-f) + y[j] * f;
   }
}

// helper functions
////////////////////////////////////////////////////////////////////////////////
/// Static function
///     if (ISNAN(x))   return 1;
///     if (ISNAN(y))   return -1;

Int_t TGraphSmooth::Rcmp(Double_t x, Double_t y)
{
   if (x < y)      return -1;
   if (x > y)      return 1;
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Static function
/// based on R function rPsort: adapted to C++ by Christian Stratowa
/// (R source file: R_sort.c by R Development Core Team (C) 1999-2001)

void TGraphSmooth::Psort(Double_t *x, Int_t n, Int_t k)
{
   Double_t v, w;
   Int_t pL, pR, i, j;

   for (pL = 0, pR = n - 1; pL < pR; ) {
      v = x[k];
      for(i = pL, j = pR; i <= j;) {
         while (TGraphSmooth::Rcmp(x[i], v) < 0) i++;
         while (TGraphSmooth::Rcmp(v, x[j]) < 0) j--;
         if (i <= j) { w = x[i]; x[i++] = x[j]; x[j--] = w; }
      }
      if (j < k) pL = i;
      if (k < i) pR = j;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// static function

void TGraphSmooth::Rank(Int_t n, Double_t *a, Int_t *index, Int_t *rank, Bool_t down)
{
   if (n <= 0) return;
   if (n == 1) {
      index[0] = 0;
      rank[0] = 0;
      return;
   }

   TMath::Sort(n,a,index,down);

   Int_t k = 0;
   for (Int_t i=0;i<n;i++) {
      if ((i > 0) && (a[index[i]] == a[index[i-1]])) {
         rank[index[i]] = i-1;
         k++;
      }
      rank[index[i]] = i-k;
   }
}
