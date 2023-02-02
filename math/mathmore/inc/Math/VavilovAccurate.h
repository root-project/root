// @(#)root/mathmore:$Id$
// Authors: B. List 29.4.2010

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2004 ROOT Foundation,  CERN/PH-SFT                   *
  *                                                                    *
  * This library is free software; you can redistribute it and/or      *
  * modify it under the terms of the GNU General Public License        *
  * as published by the Free Software Foundation; either version 2     *
  * of the License, or (at your option) any later version.             *
  *                                                                    *
  * This library is distributed in the hope that it will be useful,    *
  * but WITHOUT ANY WARRANTY; without even the implied warranty of     *
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU   *
  * General Public License for more details.                           *
  *                                                                    *
  * You should have received a copy of the GNU General Public License  *
  * along with this library (see file COPYING); if not, write          *
  * to the Free Software Foundation, Inc., 59 Temple Place, Suite      *
  * 330, Boston, MA 02111-1307 USA, or contact the author.             *
  *                                                                    *
  **********************************************************************/

// Header file for class VavilovAccurate
//
// Created by: blist  at Thu Apr 29 11:19:00 2010
//
// Last update: Thu Apr 29 11:19:00 2010
//
#ifndef ROOT_Math_VavilovAccurate
#define ROOT_Math_VavilovAccurate


#include "Math/Vavilov.h"

namespace ROOT {
namespace Math {

//____________________________________________________________________________
/**
   Class describing a Vavilov distribution.

   The probability density function of the Vavilov distribution
   as function of Landau's parameter is given by:
  \f[ p(\lambda_L; \kappa, \beta^2) =
  \frac{1}{2 \pi i}\int_{c-i\infty}^{c+i\infty} \phi(s) e^{\lambda_L s} ds\f]
   where \f$\phi(s) = e^{C} e^{\psi(s)}\f$
   with  \f$ C = \kappa (1+\beta^2 \gamma )\f$
   and \f$\psi(s)= s \ln \kappa + (s+\beta^2 \kappa)
               \cdot \left ( \int \limits_{0}^{1}
               \frac{1 - e^{\frac{-st}{\kappa}}}{t} \,d t- \gamma \right )
               - \kappa \, e^{\frac{-s}{\kappa}}\f$.
   \f$ \gamma = 0.5772156649\dots\f$ is Euler's constant.

   For the class VavilovAccurate,
   Pdf returns the Vavilov distribution as function of Landau's parameter
   \f$\lambda_L = \lambda_V/\kappa  - \ln \kappa\f$,
   which is the convention used in the CERNLIB routines, and in the tables
   by S.M. Seltzer and M.J. Berger: Energy loss stragglin of protons and mesons:
   Tabulation of the Vavilov distribution, pp 187-203
   in: National Research Council (U.S.), Committee on Nuclear Science:
   Studies in penetration of charged particles in matter,
   Nat. Akad. Sci. Publication 1133,
   Nucl. Sci. Series Report No. 39,
   Washington (Nat. Akad. Sci.) 1964, 388 pp.
   Available from
   <A HREF="http://books.google.de/books?id=kmMrAAAAYAAJ&lpg=PP9&pg=PA187#v=onepage&q&f=false">Google books</A>

   Therefore, for small values of \f$\kappa < 0.01\f$,
   pdf approaches the Landau distribution.

   For values \f$\kappa > 10\f$, the Gauss approximation should be used
   with \f$\mu\f$ and \f$\sigma\f$ given by Vavilov::mean(kappa, beta2)
   and sqrt(Vavilov::variance(kappa, beta2).

   The original Vavilov pdf is obtained by
   v.Pdf(lambdaV/kappa-log(kappa))/kappa.

   For detailed description see
   B. Schorr, Programs for the Landau and the Vavilov distributions and the corresponding random numbers,
   <A HREF="http://dx.doi.org/10.1016/0010-4655(74)90091-5">Computer Phys. Comm. 7 (1974) 215-224</A>,
   which has been implemented in
   <A HREF="https://cern-tex.web.cern.ch/cern-tex/shortwrupsdir/g116/top.html">
   CERNLIB (G116)</A>.

   The class stores coefficients needed to calculate \f$p(\lambda; \kappa, \beta^2)\f$
   for fixed values of \f$\kappa\f$ and \f$\beta^2\f$.
   Changing these values is computationally expensive.

   The parameter \f$\kappa\f$ should be in the range \f$0.01 \le \kappa \le 10\f$.
   In contrast to the CERNLIB implementation, all values of \f$\kappa \ge 0.001\f$ may be used,
   but may result in slower running and/or inaccurate results.

   The parameter \f$\beta^2\f$ must be in the range \f$0 \le \beta^2 \le 1\f$.

   Two parameters which are fixed in the CERNLIB implementation may be set by the user:
   - epsilonPM corresponds to \f$\epsilon^+ = \epsilon^-\f$ in Eqs. (2.1) and (2.2) of Schorr's paper.
   epsilonPM gives an estimate on the integral of the cumulative distribution function
   outside the range \f$\lambda_{min} \le \lambda \le \lambda_{max}\f$
   where the approximation is valid.
   Thus, it determines the support of the approximation used here (called $T_0 - T_1$ in the paper).
   Schorr recommends  \f$\epsilon^+ = \epsilon^- = 5\cdot 10^{-4}\f$.
   The code from CERNLIB has been extended such that also smaller values are possible.

   - epsilon corresponds to \f$\epsilon\f$ in Eq. (4.10) of Schorr's paper.
   It determines the accuracy of the series expansion.
   Schorr recommends  \f$\epsilon = 10^{-5}\f$.

   For the quantile calculation, the algorithm given by Schorr is not used,
   because it turns out to be very slow and still inaccurate.
   Instead, an initial estimate is calculated based on a pre-calculated table,
   which is subsequently improved by Newton iterations.

   While the CERNLIB implementation calculates at most 156 terms in the series expansion
   for the pdf and cdf calculation, this class calculates up to 500 terms, depending
   on the values of epsilonPM and epsilon.

   Average times on a Pentium Core2 Duo P8400 2.26GHz:
   - 38us per call to SetKappaBeta2 or constructor
   - 0.49us per call to Pdf, Cdf
   - 8.2us per first call to Quantile after SetKappaBeta2 or constructor
   - 0.83us per subsequent call to Quantile

   Benno List, June 2010

   @ingroup StatFunc
 */


class VavilovAccurate: public Vavilov {

public:


   /**
      Initialize an object to calculate the Vavilov distribution

       @param kappa The parameter \f$\kappa\f$, which must be in the range \f$\kappa \ge 0.001 \f$
       @param beta2 The parameter \f$\beta^2\f$, which must be in the range \f$0 \le \beta^2 \le 1 \f$
       @param epsilonPM: \f$\epsilon^+ = \epsilon^-\f$ in Eqs. (2.1) and (2.2) of Schorr's paper; gives an estimate on the integral of the cumulative distribution function
              outside the range \f$\lambda_{min} \le \lambda \le \lambda_{max}\f$
              where the approximation is valid.
       @param epsilon: \f$\epsilon\f$ in Eq. (4.10) of Schorr's paper; determines the accuracy of the series expansion.
   */

  VavilovAccurate(double kappa=1, double beta2=1, double epsilonPM=5E-4, double epsilon=1E-5);

   /**
     Destructor
   */
   ~VavilovAccurate() override;


public:

   /**
       Evaluate the Vavilov probability density function

       @param x The Landau parameter \f$x = \lambda_L\f$
   */
   double Pdf (double x) const override;

   /**
       Evaluate the Vavilov probability density function,
       and set kappa and beta2, if necessary

       @param x The Landau parameter \f$x = \lambda_L\f$
       @param kappa The parameter \f$\kappa\f$, which must be in the range \f$\kappa \ge 0.001 \f$
       @param beta2 The parameter \f$\beta^2\f$, which must be in the range \f$0 \le \beta^2 \le 1 \f$
   */
   double Pdf (double x, double kappa, double beta2) override;

   /**
       Evaluate the Vavilov cumulative probability density function

       @param x The Landau parameter \f$x = \lambda_L\f$
   */
   double Cdf (double x) const override;

   /**
       Evaluate the Vavilov cumulative probability density function,
       and set kappa and beta2, if necessary

       @param x The Landau parameter \f$x = \lambda_L\f$
       @param kappa The parameter \f$\kappa\f$, which must be in the range \f$\kappa \ge 0.001 \f$
       @param beta2 The parameter \f$\beta^2\f$, which must be in the range \f$0 \le \beta^2 \le 1 \f$
   */
   double Cdf (double x, double kappa, double beta2) override;

   /**
       Evaluate the Vavilov complementary cumulative probability density function

       @param x The Landau parameter \f$x = \lambda_L\f$
   */
   double Cdf_c (double x) const override;

   /**
       Evaluate the Vavilov complementary cumulative probability density function,
       and set kappa and beta2, if necessary

       @param x The Landau parameter \f$x = \lambda_L\f$
       @param kappa The parameter \f$\kappa\f$, which must be in the range \f$\kappa \ge 0.001 \f$
       @param beta2 The parameter \f$\beta^2\f$, which must be in the range \f$0 \le \beta^2 \le 1 \f$
   */
   double Cdf_c (double x, double kappa, double beta2) override;

   /**
       Evaluate the inverse of the Vavilov cumulative probability density function

       @param z The argument \f$z\f$, which must be in the range \f$0 \le z \le 1\f$
   */
   double Quantile (double z) const override;

   /**
       Evaluate the inverse of the Vavilov cumulative probability density function,
       and set kappa and beta2, if necessary

       @param z The argument \f$z\f$, which must be in the range \f$0 \le z \le 1\f$
       @param kappa The parameter \f$\kappa\f$, which must be in the range \f$\kappa \ge 0.001 \f$
       @param beta2 The parameter \f$\beta^2\f$, which must be in the range \f$0 \le \beta^2 \le 1 \f$
   */
   double Quantile (double z, double kappa, double beta2) override;

   /**
       Evaluate the inverse of the complementary Vavilov cumulative probability density function

       @param z The argument \f$z\f$, which must be in the range \f$0 \le z \le 1\f$
   */
   double Quantile_c (double z) const override;

   /**
       Evaluate the inverse of the complementary Vavilov cumulative probability density function,
       and set kappa and beta2, if necessary

       @param z The argument \f$z\f$, which must be in the range \f$0 \le z \le 1\f$
       @param kappa The parameter \f$\kappa\f$, which must be in the range \f$\kappa \ge 0.001 \f$
       @param beta2 The parameter \f$\beta^2\f$, which must be in the range \f$0 \le \beta^2 \le 1 \f$
   */
   double Quantile_c (double z, double kappa, double beta2) override;

   /**
      Change \f$\kappa\f$ and \f$\beta^2\f$ and recalculate coefficients if necessary

       @param kappa The parameter \f$\kappa\f$, which must be in the range \f$\kappa \ge 0.001 \f$
       @param beta2 The parameter \f$\beta^2\f$, which must be in the range \f$0 \le \beta^2 \le 1 \f$
   */
   void SetKappaBeta2 (double kappa, double beta2) override;


   /**
      (Re)Initialize the object

       @param kappa The parameter \f$\kappa\f$, which must be in the range \f$\kappa \ge 0.001 \f$
       @param beta2 The parameter \f$\beta^2\f$, which must be in the range \f$0 \le \beta^2 \le 1 \f$
       @param epsilonPM \f$\epsilon^+ = \epsilon^-\f$ in Eqs. (2.1) and (2.2) of Schorr's paper; gives an estimate on the integral of the cumulative distribution function
              outside the range \f$\lambda_{min} \le \lambda \le \lambda_{max}\f$
              where the approximation is valid.
       @param epsilon \f$\epsilon\f$ in Eq. (4.10) of Schorr's paper; determines the accuracy of the series expansion.
   */
   void Set(double kappa, double beta2, double epsilonPM=5E-4, double epsilon=1E-5);


   /**
      Return the minimum value of \f$\lambda\f$ for which \f$p(\lambda; \kappa, \beta^2)\f$
      is nonzero in the current approximation
   */
   double GetLambdaMin() const override;

   /**
      Return the maximum value of \f$\lambda\f$ for which \f$p(\lambda; \kappa, \beta^2)\f$
      is nonzero in the current approximation
   */
   double GetLambdaMax() const override;

   /**
      Return the current value of \f$\kappa\f$
   */
   double GetKappa()     const override;

   /**
      Return the current value of \f$\beta^2\f$
   */
   double GetBeta2()     const override;

   /**
      Return the value of \f$\lambda\f$ where the pdf is maximal
   */
   double Mode() const override;

   /**
      Return the value of \f$\lambda\f$ where the pdf is maximal function,
       and set kappa and beta2, if necessary

       @param kappa The parameter \f$\kappa\f$, which must be in the range \f$\kappa \ge 0.001 \f$
       @param beta2 The parameter \f$\beta^2\f$, which must be in the range \f$0 \le \beta^2 \le 1 \f$
   */
   double Mode(double kappa, double beta2) override;

   /**
      Return the current value of \f$\epsilon^+ = \epsilon^-\f$
   */

   double GetEpsilonPM() const;

   /**
      Return the current value of \f$\epsilon\f$
   */
   double GetEpsilon()   const;

   /**
      Return the number of terms used in the series expansion
   */
   double GetNTerms()    const;

   /**
      Returns a static instance of class VavilovFast
   */
   static VavilovAccurate *GetInstance();

   /**
      Returns a static instance of class VavilovFast,
      and sets the values of kappa and beta2

       @param kappa The parameter \f$\kappa\f$, which must be in the range \f$\kappa \ge 0.001 \f$
       @param beta2 The parameter \f$\beta^2\f$, which must be in the range \f$0 \le \beta^2 \le 1 \f$
   */
   static VavilovAccurate *GetInstance(double kappa, double beta2);


private:
   enum{MAXTERMS=500};
   double fH[8], fT0, fT1, fT, fOmega, fA_pdf[MAXTERMS+1], fB_pdf[MAXTERMS+1], fA_cdf[MAXTERMS+1], fB_cdf[MAXTERMS+1], fX0;
   double fKappa, fBeta2;
   double fEpsilonPM, fEpsilon;

   mutable bool fQuantileInit;
   mutable int fNQuant;
   enum{kNquantMax=32};
   mutable double fQuant[kNquantMax];
   mutable double fLambda[kNquantMax];

   void InitQuantile() const;

   static VavilovAccurate *fgInstance;

   double G116f1 (double x) const;
   double G116f2 (double x) const;

   int Rzero (double a, double b, double& x0,
              double eps, int mxf, double (VavilovAccurate::*f)(double)const) const;
   static double E1plLog (double x); // Calculates log(|x|)+E_1(x)

};

   /**
       The Vavilov probability density function

       @param x The Landau parameter \f$x = \lambda_L\f$
       @param kappa The parameter \f$\kappa\f$, which must be in the range \f$\kappa \ge 0.001 \f$
       @param beta2 The parameter \f$\beta^2\f$, which must be in the range \f$0 \le \beta^2 \le 1 \f$

       @ingroup PdfFunc
   */
double vavilov_accurate_pdf (double x, double kappa, double beta2);

   /**
       The Vavilov cumulative probability density function

       @param x The Landau parameter \f$x = \lambda_L\f$
       @param kappa The parameter \f$\kappa\f$, which must be in the range \f$\kappa \ge 0.001 \f$
       @param beta2 The parameter \f$\beta^2\f$, which must be in the range \f$0 \le \beta^2 \le 1 \f$

       @ingroup ProbFunc
   */
double vavilov_accurate_cdf (double x, double kappa, double beta2);

   /**
       The Vavilov complementary cumulative probability density function

       @param x The Landau parameter \f$x = \lambda_L\f$
       @param kappa The parameter \f$\kappa\f$, which must be in the range \f$\kappa \ge 0.001 \f$
       @param beta2 The parameter \f$\beta^2\f$, which must be in the range \f$0 \le \beta^2 \le 1 \f$

       @ingroup ProbFunc
   */
double vavilov_accurate_cdf_c (double x, double kappa, double beta2);

   /**
       The inverse of the Vavilov cumulative probability density function

       @param z The argument \f$z\f$, which must be in the range \f$0 \le z \le 1\f$
       @param kappa The parameter \f$\kappa\f$, which must be in the range \f$\kappa \ge 0.001 \f$
       @param beta2 The parameter \f$\beta^2\f$, which must be in the range \f$0 \le \beta^2 \le 1 \f$

      @ingroup QuantFunc
   */
double vavilov_accurate_quantile (double z, double kappa, double beta2);

   /**
       The inverse of the complementary Vavilov cumulative probability density function

       @param z The argument \f$z\f$, which must be in the range \f$0 \le z \le 1\f$
       @param kappa The parameter \f$\kappa\f$, which must be in the range \f$\kappa \ge 0.001 \f$
       @param beta2 The parameter \f$\beta^2\f$, which must be in the range \f$0 \le \beta^2 \le 1 \f$

      @ingroup QuantFunc
   */
double vavilov_accurate_quantile_c (double z, double kappa, double beta2);

} // namespace Math
} // namespace ROOT

#endif /* ROOT_Math_VavilovAccurate */
