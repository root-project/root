// @(#)root/mathcore:$Id$
// Author: L. Moneta Nov 2010 

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2010  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/
// helper class for binomial Neyman intervals
// author Jordan Tucker
//  integration in CMSSW: Luca Lista
//   modified and integrated in ROOT: Lorenzo Moneta


#ifndef TEFFiciencyHelper_h
#define TEFFiciencyHelper_h

#include <algorithm>
#include <cmath>
#include <vector>

#include "Math/PdfFuncMathCore.h"


// Helper class impelementing the
// binomial probability and the likelihood ratio
// used for ordering the interval in the FeldmanCousins interval class 
class BinomialProbHelper {
public:
   BinomialProbHelper(double rho, int x, int n)
      : fRho(rho), fX(x), fN(n),
        fRho_hat(double(x)/n),
        fProb(ROOT::Math::binomial_pdf(x, rho, n)) {
      // Cache the likelihood ratio L(\rho)/L(\hat{\rho}), too.
      if (x == 0)
         fLRatio = pow(1 - rho, n);
      else if (x == n)
         fLRatio = pow(rho, n);
      else
         fLRatio = pow(rho/fRho_hat, x) * pow((1 - rho)/(1 - fRho_hat), n - x);
   }

   double Rho   () const { return fRho;    };
   int    X     () const { return fX;      };
   int    N     () const { return fN;      };
   double Prob  () const { return fProb;   };
   double LRatio() const { return fLRatio; };

private:
   double fRho;
   int fX;
   int fN;
   double fRho_hat;
   double fProb;
   double fLRatio;
};



// Implement noncentral binomial confidence intervals using the Neyman
// construction. The Sorter class gives the ordering of points,
// i.e. it must be a functor implementing a greater-than relationship
// between two prob_helper instances. See feldman_cousins for an
// example.
template <typename Sorter>
class BinomialNeymanInterval  {
public:

   BinomialNeymanInterval() :
      fLower(0), 
      fUpper(1),
      fAlpha(0)
   {}

   void Init(double alpha) { 
      fAlpha = alpha;
   }

   // Given a true value of rho and ntot trials, calculate the
   // acceptance set [x_l, x_r] for use in a Neyman construction.
   bool Find_rho_set(const double rho, const int ntot, int& x_l, int& x_r) const {
      // Get the binomial probabilities for every x = 0..n, and sort them
      // in decreasing order, determined by the Sorter class.
      std::vector<BinomialProbHelper> probs;
      for (int i = 0; i <= ntot; ++i)
         probs.push_back(BinomialProbHelper(rho, i, ntot));
      std::sort(probs.begin(), probs.end(), fSorter);

      // Add up the probabilities until the total is 1 - alpha or
      // bigger, adding the biggest point first, then the next biggest,
      // etc. "Biggest" is given by the Sorter class and is taken care
      // of by the sort above. JMTBAD need to find equal probs and use
      // the sorter to differentiate between them.
      const double target = 1 - fAlpha;
      // An invalid interval.
      x_l = ntot;
      x_r = 0;
      double sum = 0;
      for (int i = 0; i <= ntot && sum < target; ++i) {
         sum += probs[i].Prob();
         const int& x = probs[i].X();
         if (x < x_l) x_l = x;
         if (x > x_r) x_r = x;
      }
  
      return x_l <= x_r;
   }

   // Construct nrho acceptance sets in rho = [0,1] given ntot trials
   // and put the results in already-allocated x_l and x_r.
   bool Neyman(const int ntot, const int nrho, double* rho, double* x_l, double* x_r) {
      int xL, xR;
      for (int i = 0; i < nrho; ++i) {
         rho[i] = double(i)/nrho;
         Find_rho_set(rho[i], ntot, xL, xR);
         x_l[i] = xL;
         x_r[i] = xR;
      }
      return true;
   }

   // Given X successes and n trials, calculate the interval using the
   // rho acceptance sets implemented above.
   void Calculate(const double X, const double n) {
      Set(0, 1);

      const double tol = 1e-9;
      double rho_min, rho_max, rho;
      int x_l, x_r;
  
      // Binary search for the smallest rho whose acceptance set has right
      // endpoint X; this is the lower endpoint of the rho interval.
      rho_min = 0; rho_max = 1;
      while (std::abs(rho_max - rho_min) > tol) {
         rho = (rho_min + rho_max)/2;
         Find_rho_set(rho, int(n), x_l, x_r);
         if (x_r < X)
            rho_min = rho;
         else
            rho_max = rho;
      }
      fLower = rho;
  
      // Binary search for the largest rho whose acceptance set has left
      // endpoint X; this is the upper endpoint of the rho interval.
      rho_min = 0; rho_max = 1;
      while (std::abs(rho_max - rho_min) > tol) {
         rho = (rho_min + rho_max)/2;
         Find_rho_set(rho, int(n), x_l, x_r);
         if (x_l > X)
            rho_max = rho;
         else
            rho_min = rho;
      }
      fUpper = rho;
   }

   double Lower() const { return fLower; }
   double Upper() const { return fUpper; }

private:
   Sorter fSorter;

   double fLower;
   double fUpper;

   double    fAlpha;

   void Set(double l, double u) { fLower = l; fUpper = u; }

};




struct FeldmanCousinsSorter {
   bool operator()(const BinomialProbHelper& l, const BinomialProbHelper& r) const {
      return l.LRatio() > r.LRatio();
   }
};

class FeldmanCousinsBinomialInterval : public BinomialNeymanInterval<FeldmanCousinsSorter> {
   //const char* name() const { return "Feldman-Cousins"; }

};




#endif
