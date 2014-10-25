// @(#)root/mathcore:$Id$
// Author: Rene Brun, Lorenzo Moneta   15/12/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//
// TRandom
//
// basic Random number generator class (periodicity = 10**9).
// Note that this is a very simple generator (linear congruential)
// which is known to have defects (the lower random bits are correlated)
// and therefore should NOT be used in any statistical study.
// One should use instead TRandom1, TRandom2 or TRandom3.
// TRandom3, is based on the "Mersenne Twister generator", and is the recommended one,
// since it has good random proprieties (period of about 10**6000 ) and it is fast.
// TRandom1, based on the RANLUX algorithm, has mathematically proven random proprieties
// and a period of about 10**171. It is however slower than the others.
// TRandom2, is based on the Tausworthe generator of L'Ecuyer, and it has the advantage
// of being fast and using only 3 words (of 32 bits) for the state. The period is 10**26.
//
// The following table shows some timings (in nanoseconds/call)
// for the random numbers obtained using an Intel Pentium 3.0 GHz running Linux
// and using the gcc 3.2.3 compiler
//
//    TRandom           34   ns/call     (BAD Generator)
//    TRandom1          242  ns/call
//    TRandom2          37   ns/call
//    TRandom3          45   ns/call
//
//
// The following basic Random distributions are provided:
// ===================================================
//   -Exp(tau)
//   -Integer(imax)
//   -Gaus(mean,sigma)
//   -Rndm()
//   -Uniform(x1)
//   -Landau(mpv,sigma)
//   -Poisson(mean)
//   -Binomial(ntot,prob)
//
// Random numbers distributed according to 1-d, 2-d or 3-d distributions
// =====================================================================
// contained in TF1, TF2 or TF3 objects.
// For example, to get a random number distributed following abs(sin(x)/x)*sqrt(x)
// you can do :
//   TF1 *f1 = new TF1("f1","abs(sin(x)/x)*sqrt(x)",0,10);
//   double r = f1->GetRandom();
// or you can use the UNURAN package. You need in this case to initialize UNURAN
// to the function you would like to generate.
//   TUnuran u;
//   u.Init(TUnuranDistrCont(f1));
//   double r = u.Sample();
//
// The techniques of using directly a TF1,2 or 3 function is powerful and
// can be used to generate numbers in the defined range of the function.
// Getting a number from a TF1,2,3 function is also quite fast.
// UNURAN is a  powerful and flexible tool which containes various methods for
// generate random numbers for continuous distributions of one and multi-dimension.
// It requires some set-up (initialization) phase and can be very fast when the distribution
// parameters are not changed for every call.
//
// The following table shows some timings (in nanosecond/call)
// for basic functions,  TF1 functions and using UNURAN obtained running
// the tutorial math/testrandom.C
// Numbers have been obtained on an Intel Xeon Quad-core Harpertown (E5410) 2.33 GHz running
// Linux SLC4 64 bit and compiled with gcc 3.4
//
// Distribution            nanoseconds/call
//                     TRandom  TRandom1 TRandom2 TRandom3
// Rndm..............    5.000  105.000    7.000   10.000
// RndmArray.........    4.000  104.000    6.000    9.000
// Gaus..............   36.000  180.000   40.000   48.000
// Rannor............  118.000  220.000  120.000  124.000
// Landau............   22.000  123.000   26.000   31.000
// Exponential.......   93.000  198.000   98.000  104.000
// Binomial(5,0.5)...   30.000  548.000   46.000   65.000
// Binomial(15,0.5)..   75.000 1615.000  125.000  178.000
// Poisson(3)........   96.000  494.000  109.000  125.000
// Poisson(10).......  138.000 1236.000  165.000  203.000
// Poisson(70).......  818.000 1195.000  835.000  844.000
// Poisson(100)......  837.000 1218.000  849.000  864.000
// GausTF1...........   83.000  180.000   87.000   88.000
// LandauTF1.........   80.000  180.000   83.000   86.000
// GausUNURAN........   40.000  139.000   41.000   44.000
// PoissonUNURAN(10).   85.000  271.000   92.000  102.000
// PoissonUNURAN(100)   62.000  256.000   69.000   78.000
//
//  Note that the time to generate a number from an arbitrary TF1 function
//  using TF1::GetRandom or using TUnuran is  independent of the complexity of the function.
//
//  TH1::FillRandom(TH1 *) or TH1::FillRandom(const char *tf1name)
//  ==============================================================
//  can be used to fill an histogram (1-d, 2-d, 3-d from an existing histogram
//  or from an existing function.
//
//  Note this interesting feature when working with objects
//  =======================================================
//  You can use several TRandom objects, each with their "independent"
//  random sequence. For example, one can imagine
//     TRandom *eventGenerator = new TRandom();
//     TRandom *tracking       = new TRandom();
//  eventGenerator can be used to generate the event kinematics.
//  tracking can be used to track the generated particles with random numbers
//  independent from eventGenerator.
//  This very interesting feature gives the possibility to work with simple
//  and very fast random number generators without worrying about
//  random number periodicity as it was the case with Fortran.
//  One can use TRandom::SetSeed to modify the seed of one generator.
//
//  a TRandom object may be written to a Root file
//  ==============================================
//    -as part of another object
//    -or with its own key (example gRandom->Write("Random");
//
//////////////////////////////////////////////////////////////////////////

#include "TROOT.h"
#include "TMath.h"
#include "TRandom.h"
#include "TRandom3.h"
#include "TSystem.h"
#include "TDirectory.h"
#include "Math/QuantFuncMathCore.h"
#include "TUUID.h"

ClassImp(TRandom)

//______________________________________________________________________________
TRandom::TRandom(UInt_t seed): TNamed("Random","Default Random number generator")
{
   // Default constructor. For seed see SetSeed().

   SetSeed(seed);
}

//______________________________________________________________________________
TRandom::~TRandom()
{
   // Default destructor. Can reset gRandom to 0 if gRandom points to this
   // generator.

   if (gRandom == this) gRandom = 0;
}

//______________________________________________________________________________
Int_t TRandom::Binomial(Int_t ntot, Double_t prob)
{
   // Generates a random integer N according to the binomial law.
   // Coded from Los Alamos report LA-5061-MS.
   //
   // N is binomially distributed between 0 and ntot inclusive
   // with mean prob*ntot and prob is between 0 and 1.
   //
   // Note: This function should not be used when ntot is large (say >100).
   // The normal approximation is then recommended instead
   // (with mean =*ntot+0.5 and standard deviation sqrt(ntot*prob*(1-prob)).

   if (prob < 0 || prob > 1) return 0;
   Int_t n = 0;
   for (Int_t i=0;i<ntot;i++) {
      if (Rndm() > prob) continue;
      n++;
   }
   return n;
}

//______________________________________________________________________________
Double_t TRandom::BreitWigner(Double_t mean, Double_t gamma)
{
   // Return a number distributed following a BreitWigner function with mean and gamma.

   Double_t rval, displ;
   rval = 2*Rndm() - 1;
   displ = 0.5*gamma*TMath::Tan(rval*TMath::PiOver2());

   return (mean+displ);
}

//______________________________________________________________________________
void TRandom::Circle(Double_t &x, Double_t &y, Double_t r)
{
   // Generates random vectors, uniformly distributed over a circle of given radius.
   //   Input : r = circle radius
   //   Output: x,y a random 2-d vector of length r

   Double_t phi = Uniform(0,TMath::TwoPi());
   x = r*TMath::Cos(phi);
   y = r*TMath::Sin(phi);
}

//______________________________________________________________________________
Double_t TRandom::Exp(Double_t tau)
{
   // Returns an exponential deviate.
   //
   //          exp( -t/tau )

   Double_t x = Rndm();              // uniform on ] 0, 1 ]
   Double_t t = -tau * TMath::Log( x ); // convert to exponential distribution
   return t;
}

//______________________________________________________________________________
Double_t TRandom::Gaus(Double_t mean, Double_t sigma)
{
   // Samples a random number from the standard Normal (Gaussian) Distribution
   // with the given mean and sigma.
   // Uses the Acceptance-complement ratio from W. Hoermann and G. Derflinger
   // This is one of the fastest existing method for generating normal random variables.
   // It is a factor 2/3 faster than the polar (Box-Muller) method used in the previous
   // version of TRandom::Gaus. The speed is comparable to the Ziggurat method (from Marsaglia)
   // implemented for example in GSL and available in the MathMore library.
   //
   // REFERENCE:  - W. Hoermann and G. Derflinger (1990):
   //              The ACR Method for generating normal random variables,
   //              OR Spektrum 12 (1990), 181-185.
   //
   // Implementation taken from
   // UNURAN (c) 2000  W. Hoermann & J. Leydold, Institut f. Statistik, WU Wien

   const Double_t kC1 = 1.448242853;
   const Double_t kC2 = 3.307147487;
   const Double_t kC3 = 1.46754004;
   const Double_t kD1 = 1.036467755;
   const Double_t kD2 = 5.295844968;
   const Double_t kD3 = 3.631288474;
   const Double_t kHm = 0.483941449;
   const Double_t kZm = 0.107981933;
   const Double_t kHp = 4.132731354;
   const Double_t kZp = 18.52161694;
   const Double_t kPhln = 0.4515827053;
   const Double_t kHm1 = 0.516058551;
   const Double_t kHp1 = 3.132731354;
   const Double_t kHzm = 0.375959516;
   const Double_t kHzmp = 0.591923442;
   /*zhm 0.967882898*/

   const Double_t kAs = 0.8853395638;
   const Double_t kBs = 0.2452635696;
   const Double_t kCs = 0.2770276848;
   const Double_t kB  = 0.5029324303;
   const Double_t kX0 = 0.4571828819;
   const Double_t kYm = 0.187308492 ;
   const Double_t kS  = 0.7270572718 ;
   const Double_t kT  = 0.03895759111;

   Double_t result;
   Double_t rn,x,y,z;

   do {
      y = Rndm();

      if (y>kHm1) {
         result = kHp*y-kHp1; break; }

      else if (y<kZm) {
         rn = kZp*y-1;
         result = (rn>0) ? (1+rn) : (-1+rn);
         break;
      }

      else if (y<kHm) {
         rn = Rndm();
         rn = rn-1+rn;
         z = (rn>0) ? 2-rn : -2-rn;
         if ((kC1-y)*(kC3+TMath::Abs(z))<kC2) {
            result = z; break; }
         else {
            x = rn*rn;
            if ((y+kD1)*(kD3+x)<kD2) {
               result = rn; break; }
            else if (kHzmp-y<exp(-(z*z+kPhln)/2)) {
               result = z; break; }
            else if (y+kHzm<exp(-(x+kPhln)/2)) {
               result = rn; break; }
         }
      }

      while (1) {
         x = Rndm();
         y = kYm * Rndm();
         z = kX0 - kS*x - y;
         if (z>0)
            rn = 2+y/x;
         else {
            x = 1-x;
            y = kYm-y;
            rn = -(2+y/x);
         }
         if ((y-kAs+x)*(kCs+x)+kBs<0) {
            result = rn; break; }
         else if (y<x+kT)
            if (rn*rn<4*(kB-log(x))) {
               result = rn; break; }
      }
   } while(0);

   return mean + sigma * result;
}

//______________________________________________________________________________
UInt_t TRandom::Integer(UInt_t imax)
{
   // Returns a random integer on [ 0, imax-1 ].

   UInt_t ui;
   ui = (UInt_t)(imax*Rndm());
   return ui;
}

//______________________________________________________________________________
Double_t TRandom::Landau(Double_t mu, Double_t sigma)
{
   // Generate a random number following a Landau distribution
   // with location parameter mu and scale parameter sigma:
   //      Landau( (x-mu)/sigma )
   // Note that mu is not the mpv(most probable value) of the Landa distribution
   // and sigma is not the standard deviation of the distribution which is not defined.
   // For mu  =0 and sigma=1, the mpv = -0.22278
   //
   // The Landau random number generation is implemented using the
   // function landau_quantile(x,sigma), which provides
   // the inverse of the landau cumulative distribution.
   // landau_quantile has been converted from CERNLIB ranlan(G110).

   if (sigma <= 0) return 0;
   Double_t x = Rndm();
   Double_t res = mu + ROOT::Math::landau_quantile(x, sigma);
   return res;
}

//______________________________________________________________________________
Int_t TRandom::Poisson(Double_t mean)
{
   // Generates a random integer N according to a Poisson law.
   // Prob(N) = exp(-mean)*mean^N/Factorial(N)
   //
   // Use a different procedure according to the mean value.
   // The algorithm is the same used by CLHEP.
   // For lower value (mean < 25) use the rejection method based on
   // the exponential.
   // For higher values use a rejection method comparing with a Lorentzian
   // distribution, as suggested by several authors.
   // This routine since is returning 32 bits integer will not work for values
   // larger than 2*10**9.
   // One should then use the Trandom::PoissonD for such large values.

   Int_t n;
   if (mean <= 0) return 0;
   if (mean < 25) {
      Double_t expmean = TMath::Exp(-mean);
      Double_t pir = 1;
      n = -1;
      while(1) {
         n++;
         pir *= Rndm();
         if (pir <= expmean) break;
      }
      return n;
   }
   // for large value we use inversion method
   else if (mean < 1E9) {
      Double_t em, t, y;
      Double_t sq, alxm, g;
      Double_t pi = TMath::Pi();

      sq = TMath::Sqrt(2.0*mean);
      alxm = TMath::Log(mean);
      g = mean*alxm - TMath::LnGamma(mean + 1.0);

      do {
         do {
            y = TMath::Tan(pi*Rndm());
            em = sq*y + mean;
         } while( em < 0.0 );

         em = TMath::Floor(em);
         t = 0.9*(1.0 + y*y)* TMath::Exp(em*alxm - TMath::LnGamma(em + 1.0) - g);
      } while( Rndm() > t );

      return static_cast<Int_t> (em);

   }
   else {
      // use Gaussian approximation vor very large values
      n = Int_t(Gaus(0,1)*TMath::Sqrt(mean) + mean +0.5);
      return n;
   }
}

//______________________________________________________________________________
Double_t TRandom::PoissonD(Double_t mean)
{
   // Generates a random number according to a Poisson law.
   // Prob(N) = exp(-mean)*mean^N/Factorial(N)
   //
   // This function is a variant of TRandom::Poisson returning a double
   // instead of an integer.

   Int_t n;
   if (mean <= 0) return 0;
   if (mean < 25) {
      Double_t expmean = TMath::Exp(-mean);
      Double_t pir = 1;
      n = -1;
      while(1) {
         n++;
         pir *= Rndm();
         if (pir <= expmean) break;
      }
      return static_cast<Double_t>(n);
   }
   // for large value we use inversion method
   else if (mean < 1E9) {
      Double_t em, t, y;
      Double_t sq, alxm, g;
      Double_t pi = TMath::Pi();

      sq = TMath::Sqrt(2.0*mean);
      alxm = TMath::Log(mean);
      g = mean*alxm - TMath::LnGamma(mean + 1.0);

      do {
         do {
            y = TMath::Tan(pi*Rndm());
            em = sq*y + mean;
         } while( em < 0.0 );

         em = TMath::Floor(em);
         t = 0.9*(1.0 + y*y)* TMath::Exp(em*alxm - TMath::LnGamma(em + 1.0) - g);
      } while( Rndm() > t );

      return em;

   } else {
      // use Gaussian approximation vor very large values
      return Gaus(0,1)*TMath::Sqrt(mean) + mean +0.5;
   }
}

//______________________________________________________________________________
void TRandom::Rannor(Float_t &a, Float_t &b)
{
   // Return 2 numbers distributed following a gaussian with mean=0 and sigma=1.

   Double_t r, x, y, z;

   y = Rndm();
   z = Rndm();
   x = z * 6.28318530717958623;
   r = TMath::Sqrt(-2*TMath::Log(y));
   a = (Float_t)(r * TMath::Sin(x));
   b = (Float_t)(r * TMath::Cos(x));
}

//______________________________________________________________________________
void TRandom::Rannor(Double_t &a, Double_t &b)
{
   // Return 2 numbers distributed following a gaussian with mean=0 and sigma=1.

   Double_t r, x, y, z;

   y = Rndm();
   z = Rndm();
   x = z * 6.28318530717958623;
   r = TMath::Sqrt(-2*TMath::Log(y));
   a = r * TMath::Sin(x);
   b = r * TMath::Cos(x);
}

//_____________________________________________________________________________
void TRandom::ReadRandom(const char *filename)
{
   // Reads saved random generator status from filename.

   if (!gDirectory) return;
   char *fntmp = gSystem->ExpandPathName(filename);
   TDirectory *file = (TDirectory*)gROOT->ProcessLine(Form("TFile::Open(\"%s\");",fntmp));
   delete [] fntmp;
   if(file && file->GetFile()) {
      gDirectory->ReadTObject(this,GetName());
      delete file;
   }
}

//______________________________________________________________________________
Double_t TRandom::Rndm(Int_t)
{
   //  Machine independent random number generator.
   //  Based on the BSD Unix (Rand) Linear congrential generator.
   //  Produces uniformly-distributed floating points between 0 and 1.
   //  Identical sequence on all machines of >= 32 bits.
   //  Periodicity = 2**31, generates a number in (0,1).
   //  Note that this is a generator which is known to have defects
   //  (the lower random bits are correlated) and therefore should NOT be
   //  used in any statistical study).

#ifdef OLD_TRANDOM_IMPL
   const Double_t kCONS = 4.6566128730774E-10;
   const Int_t kMASK24  = 2147483392;

   fSeed *= 69069;
   UInt_t jy = (fSeed&kMASK24); // Set lower 8 bits to zero to assure exact float
   if (jy) return kCONS*jy;
   return Rndm();
#endif

   // kCONS = 1./2147483648 = 1./(RAND_MAX+1) and RAND_MAX= 0x7fffffffUL
   const Double_t kCONS = 4.6566128730774E-10; // (1/pow(2,31)
   fSeed = (1103515245 * fSeed + 12345) & 0x7fffffffUL;

   if (fSeed) return  kCONS*fSeed;
   return Rndm();
}

//______________________________________________________________________________
void TRandom::RndmArray(Int_t n, Double_t *array)
{
   // Return an array of n random numbers uniformly distributed in ]0,1].

   const Double_t kCONS = 4.6566128730774E-10; // (1/pow(2,31))
   Int_t i=0;
   while (i<n) {
      fSeed = (1103515245 * fSeed + 12345) & 0x7fffffffUL;
      if (fSeed) {array[i] = kCONS*fSeed; i++;}
   }
}

//______________________________________________________________________________
void TRandom::RndmArray(Int_t n, Float_t *array)
{
   // Return an array of n random numbers uniformly distributed in ]0,1].

   const Double_t kCONS = 4.6566128730774E-10; // (1/pow(2,31))
   Int_t i=0;
   while (i<n) {
      fSeed = (1103515245 * fSeed + 12345) & 0x7fffffffUL;
      if (fSeed) {array[i] = Float_t(kCONS*fSeed); i++;}
   }
}

//______________________________________________________________________________
void TRandom::SetSeed(UInt_t seed)
{
   // Set the random generator seed. Note that default value is zero, which is
   // different than the default value used when constructing the class.
   // If the seed is zero the seed is set to a random value
   // which in case of TRandom depends on the lowest 4 bytes of TUUID
   // The UUID will be identical if SetSeed(0) is called with time smaller than 100 ns
   // Instead if a different generator implementation is used (TRandom1, 2 or 3)
   // the seed is generated using a 128 bit UUID. This results in different seeds
   // and then random sequence for every SetSeed(0) call.

   if( seed==0 ) {
      TUUID u;
      UChar_t uuid[16];
      u.GetUUID(uuid);
      fSeed  =  UInt_t(uuid[3])*16777216 + UInt_t(uuid[2])*65536 + UInt_t(uuid[1])*256 + UInt_t(uuid[0]);
   } else {
      fSeed = seed;
   }
}

//______________________________________________________________________________
void TRandom::Sphere(Double_t &x, Double_t &y, Double_t &z, Double_t r)
{
   // Generates random vectors, uniformly distributed over the surface
   // of a sphere of given radius.
   //   Input : r = sphere radius
   //   Output: x,y,z a random 3-d vector of length r
   // Method: (based on algorithm suggested by Knuth and attributed to Robert E Knop)
   //         which uses less random numbers than the CERNLIB RN23DIM algorithm

   Double_t a=0,b=0,r2=1;
   while (r2 > 0.25) {
      a  = Rndm() - 0.5;
      b  = Rndm() - 0.5;
      r2 =  a*a + b*b;
   }
   z = r* ( -1. + 8.0 * r2 );

   Double_t scale = 8.0 * r * TMath::Sqrt(0.25 - r2);
   x = a*scale;
   y = b*scale;
}

//______________________________________________________________________________
Double_t TRandom::Uniform(Double_t x1)
{
   // Returns a uniform deviate on the interval  (0, x1).

   Double_t ans = Rndm();
   return x1*ans;
}

//______________________________________________________________________________
Double_t TRandom::Uniform(Double_t x1, Double_t x2)
{
   // Returns a uniform deviate on the interval (x1, x2).

   Double_t ans= Rndm();
   return x1 + (x2-x1)*ans;
}

//_____________________________________________________________________________
void TRandom::WriteRandom(const char *filename)
{
   // Writes random generator status to filename.

   if (!gDirectory) return;
   char *fntmp = gSystem->ExpandPathName(filename);
   TDirectory *file = (TDirectory*)gROOT->ProcessLine(Form("TFile::Open(\"%s\",\"recreate\");",fntmp));
   delete [] fntmp;
   if(file && file->GetFile()) {
      gDirectory->WriteTObject(this,GetName());
      delete file;
   }
}
