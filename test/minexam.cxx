// @(#)root/test:$Id$
// Author: Rene Brun   22/08/95

//______________________________________________________________________________
//*-*-*-*-*-*-*-*-*-*-*-*The Minuit standard test program-*-*-*-*-*-*-*-*-*
//*-*                    ========================                         *
//*-*                                                                     *
//*-*    This program is the translation to C++ of the minexam program    *
//*-*    distributed with the Minuit/Fortran source file.                 *
//*-*         original author Fred James                                  *
//*-*                                                                     *
//*-*       Fit randomly-generated leptonic K0 decays to the              *
//*-*       time distribution expected for interfering K1 and K2,         *
//*-*       with free parameters Re(X), Im(X), DeltaM, and GammaS.        *
//*-*                                                                     *
//*-*   This program can be run in batch mode with the makefile           *
//*-*   or executed interactively with the command:                       *
//*-*         Root > .x minexam.cxx                                       *
//*-*                                                                     *
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


void fcnk0(int &npar, double *gin, double &f, double *x, int iflag);
int minexam();

#ifndef __CINT__
#include "TVirtualFitter.h"
#include "TMath.h"
#include "TStopwatch.h"

#include <stdlib.h>
#include <stdio.h>

////////////////////////////////////////////////////////////////////////////////

int main()
{
   return minexam();
}
#endif

int minexam()
{
   TStopwatch timer;

   // Initialize TMinuit via generic fitter interface with a maximum of 5 params
   TVirtualFitter *minuit = TVirtualFitter::Fitter(0, 5);
   printf("Starting timer\n");
   timer.Start();
   minuit->SetFCN(fcnk0);

   minuit->SetParameter(0, "Re(X)",    0,     0.1, 0,0);
   minuit->SetParameter(1, "Im(X)",    0,     0.1, 0,0);
   minuit->SetParameter(2, "Delta M",  0.535, 0.1, 0,0);
   minuit->SetParameter(3, "T Kshort", 0.892, 0  , 0,0);
   minuit->SetParameter(4, "T Klong",  518.3, 0  , 0,0);

//*-*-       Request FCN to read in (or generate random) data (IFLAG=1)
   Double_t arglist[100];
   arglist[0] = 1;
   minuit->ExecuteCommand("CALL FCN", arglist, 1);
   minuit->FixParameter(2);
   arglist[0] = 0;
   minuit->ExecuteCommand("SET PRINT", arglist, 1);
   minuit->ExecuteCommand("MIGRAD", arglist, 0);
   minuit->ExecuteCommand("MINOS", arglist, 0);
   minuit->ReleaseParameter(2);
   minuit->ExecuteCommand("MIGRAD", arglist, 0);
   minuit->ExecuteCommand("MINOS", arglist, 0);
   arglist[0] = 3;
   minuit->ExecuteCommand("CALL FCN", arglist, 1);

   printf("Time at the end of job = %f seconds\n",timer.CpuTime());
   return 0;
}

////////////////////////////////////////////////////////////////////////////////

void fcnk0(Int_t &, Double_t *, Double_t &f, Double_t *x, Int_t iflag)
{
   static Double_t thplu[50],thmin[50],t[50];

   static Double_t evtp[30] = {
              11.,  9., 13., 13., 17.,  9.,  1.,  7.,  8.,  9.,
               6.,  4.,  6.,  3.,  7.,  4.,  7.,  3.,  8.,  4.,
               6.,  5.,  7.,  2.,  7.,  1.,  4.,  1.,  4.,  5.};
   static Double_t evtm[30] = {
               0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,
               0.,  2.,  1.,  4.,  4.,  2.,  4.,  2.,  2.,  0.,
               2.,  3.,  7.,  2.,  3.,  6.,  2.,  4.,  1.,  5.};
   static Double_t sevtp, sevtm;

   const Int_t nbins = 30;
   const Int_t nevtot = 250;

   Int_t i, nevplu, nevmin;

   Double_t xre, xim, dm, gams, gaml,gamls,xr1,xr2,em,ep;
   Double_t chisq, ti, thp, thm, evp, evm, chi1, chi2;
   Double_t sthplu, sthmin, ehalf, th, sterm;

   xre   = x[0];
   xim   = x[1];
   dm    = x[2];
   gams  = 1/x[3];
   gaml  = 1/x[4];
   gamls = 0.5*(gaml+gams);
      //  generate random data
   if (iflag == 1) {
      sthplu = sthmin = 0;
      for (i=0;i<nbins; i++) {
         t[i]     = 0.1*(i+1);
         ti       = t[i];         ehalf    = TMath::Exp(-ti*gamls);
         xr1      = 1-xre;
         xr2      = 1+xre;
         th       = (xr1*xr1 + xim*xim)*TMath::Exp(-ti*gaml);
         th      += (xr2*xr2 + xim*xim)*TMath::Exp(-ti*gams);
         th      -= 4*ehalf*xim*TMath::Sin(dm*ti);
         sterm    = 2*ehalf*(1-xre*xre-xim*xim)*TMath::Cos(dm*ti);
         thplu[i] = th + sterm;
         thmin[i] = th - sterm;
         sthplu  += thplu[i];
         sthmin  += thmin[i];
      }
      nevplu = Int_t(nevtot*(sthplu/(sthplu+sthmin)));
      nevmin = Int_t(nevtot*(sthmin/(sthplu+sthmin)));
      printf("  LEPTONIC K ZERO DECAYS\n");
      printf(" PLUS, MINUS, TOTAL=%5d %5d %5d\n",nevplu,nevmin,nevtot);
      printf("0      TIME       THEOR+      EXPTL+      THEOR-      EXPTL-\n");
      sevtp = sevtm = 0;
      for (i=0;i<nbins; i++) {
         thplu[i] = thplu[i]*nevplu/ sthplu;
         thmin[i] = thmin[i]*nevmin/ sthmin;
         sevtp   += evtp[i];
         sevtm   += evtm[i];
         printf("%12.4f%12.4f%12.4f%12.4f%12.4f\n",t[i],thplu[i],evtp[i],thmin[i],evtm[i]);
      }
      printf(" DATA EVTS PLUS, MINUS= %10.2f%10.2f\n", sevtp,sevtm);
   }
//                      calculate chisquared
   chisq = sthplu = sthmin = 0;
   for (i=0;i<nbins; i++) {
      ti    = t[i];
      ehalf = TMath::Exp(-ti*gamls);
      xr1   = 1-xre;
      xr2   = 1+xre;
      th    = (xr1*xr1 + xim*xim)*TMath::Exp(-ti*gaml);
      th   += (xr2*xr2 + xim*xim)*TMath::Exp(-ti*gams);
      th   -= 4*ehalf*xim*TMath::Sin(dm*ti);
      sterm = 2*ehalf*(1-xre*xre-xim*xim)*TMath::Cos(dm*ti);
      thplu[i] = th + sterm;
      thmin[i] = th - sterm;
      sthplu  += thplu[i];
      sthmin  += thmin[i];
  }
   thp = thm = evp = evm = 0;
   if (iflag != 4) {
      printf("          POSITIVE LEPTONS                    ,NEGATIVE LEPTONS\n");
      printf("      TIME    THEOR    EXPTL    chisq         TIME    THEOR    EXPTL    chisq\n");
   }
   for (i=0;i<nbins; i++) {
      thplu[i] = thplu[i]*sevtp / sthplu;
      thmin[i] = thmin[i]*sevtm / sthmin;
      thp += thplu[i];
      thm += thmin[i];
      evp += evtp[i];
      evm += evtm[i];
         //  Sum over bins until at least four events found
      if (evp > 3)  {
         ep     = evp-thp;
         chi1   = (ep*ep)/evp;
         chisq += chi1;
         if (iflag != 4) printf(" %9.3f%9.3f%9.3f%9.3f\n",t[i],thp,evp,chi1);
         thp = evp = 0;
      }
      if (evm > 3) {
         em     = evm-thm;
         chi2   = (em*em)/evm;
         chisq += chi2;
         if (iflag != 4) {
            printf("                                          %9.3f%9.3f%9.3f%9.3f\n"
                             ,t[i],thm,evm,chi2);
         }
         thm = evm = 0;
      }
  }
  f = chisq;
}
