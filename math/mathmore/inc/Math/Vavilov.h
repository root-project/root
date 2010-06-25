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

// Header file for class Vavilov
// 
// Created by: blist  at Thu Apr 29 11:19:00 2010
// 
// Last update: Thu Apr 29 11:19:00 2010
// 
#ifndef ROOT_Math_Vavilov
#define ROOT_Math_Vavilov



/**
   @ingroup StatFunc
 */


#include <iostream>

namespace ROOT {
namespace Math {

//____________________________________________________________________________
/**
   Base class describing a Vavilov distribution
   
   The Vavilov distribution is defined in
   P.V. Vavilov: Ionization losses of high-energy heavy particles,
   Sov. Phys. JETP 5 (1957) 749 [Zh. Eksp. Teor. Fiz. 32 (1957) 920].
   
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
   
   For the class Vavilov, 
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
   with \f$\mu\f$ and \f$\sigma\f$ given by Vavilov::Mean(kappa, beta2)
   and sqrt(Vavilov::Variance(kappa, beta2).
   
   The original Vavilov pdf is obtained by
   v.Pdf(lambdaV/kappa-log(kappa))/kappa.
   
   Two subclasses are provided:
   - VavilovFast uses the algorithm by 
   A. Rotondi and P. Montagna, Fast calculation of Vavilov distribution, 
   <A HREF="http://dx.doi.org/10.1016/0168-583X(90)90749-K">Nucl. Instr. and Meth. B47 (1990) 215-224</A>,
   which has been implemented in 
   <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/g115/top.html">
   CERNLIB (G115)</A>.
    
   - VavilovAccurate uses the algorithm by
   B. Schorr, Programs for the Landau and the Vavilov distributions and the corresponding random numbers, 
   <A HREF="http://dx.doi.org/10.1016/0010-4655(74)90091-5">Computer Phys. Comm. 7 (1974) 215-224</A>,
   which has been implemented in 
   <A HREF="http://wwwasdoc.web.cern.ch/wwwasdoc/shortwrupsdir/g116/top.html">
   CERNLIB (G116)</A>.
   
   Both subclasses store coefficients needed to calculate \f$p(\lambda; \kappa, \beta^2)\f$
   for fixed values of \f$\kappa\f$ and \f$\beta^2\f$.
   Changing these values is computationally expensive.
   
   VavilovFast is about 5 times faster for the calculation of the Pdf than VavilovAccurate;
   initialization takes about 100 times longer than calculation of the Pdf value.
   For the quantile calculation, VavilovFast
   is 30 times faster for the initialization, and 6 times faster for 
   subsequent calculations. Initialization for Quantile takes
   27 (11) times longer than subsequent calls for VavilovFast (VavilovAccurate).

   @ingroup StatFunc
   
   */
   

class Vavilov {

public: 


   /**
     Default constructor
   */
   Vavilov();

   /**
     Destructor
   */
   virtual ~Vavilov(); 
   
public: 
  
   /** 
       Evaluate the Vavilov probability density function
       
       @param x The Landau parameter \f$x = \lambda_L\f$ 
       
   */
   virtual double Pdf (double x) const = 0;
  
   /** 
       Evaluate the Vavilov probability density function,
       and set kappa and beta2, if necessary
       
       @param x The Landau parameter \f$x = \lambda_L\f$ 
       @param kappa The parameter \f$\kappa\f$, which should be in the range \f$0.01 \le \kappa \le 10 \f$ 
       @param beta2 The parameter \f$\beta^2\f$, which must be in the range \f$0 \le \beta^2 \le 1 \f$ 
   */
   virtual double Pdf (double x, double kappa, double beta2) = 0;
  
   /** 
       Evaluate the Vavilov cummulative probability density function
       
       @param x The Landau parameter \f$x = \lambda_L\f$ 
   */
   virtual double Cdf (double x) const = 0;
 
   /** 
       Evaluate the Vavilov cummulative probability density function,
       and set kappa and beta2, if necessary
       
       @param x The Landau parameter \f$x = \lambda_L\f$ 
       @param kappa The parameter \f$\kappa\f$, which should be in the range \f$0.01 \le \kappa \le 10 \f$ 
       @param beta2 The parameter \f$\beta^2\f$, which must be in the range \f$0 \le \beta^2 \le 1 \f$ 
   */
   virtual double Cdf (double x, double kappa, double beta2) = 0;
   
   /** 
       Evaluate the Vavilov complementary cummulative probability density function
       
       @param x The Landau parameter \f$x = \lambda_L\f$ 
   */
   virtual double Cdf_c (double x) const = 0;
   
   /** 
       Evaluate the Vavilov complementary cummulative probability density function,
       and set kappa and beta2, if necessary
       
       @param x The Landau parameter \f$x = \lambda_L\f$ 
       @param kappa The parameter \f$\kappa\f$, which should be in the range \f$0.01 \le \kappa \le 10 \f$ 
       @param beta2 The parameter \f$\beta^2\f$, which must be in the range \f$0 \le \beta^2 \le 1 \f$ 
   */
   virtual double Cdf_c (double x, double kappa, double beta2) = 0;
  
   /** 
       Evaluate the inverse of the Vavilov cummulative probability density function
       
       @param z The argument \f$z\f$, which must be in the range \f$0 \le z \le 1\f$
   */
   virtual double Quantile (double z) const = 0;
  
   /** 
       Evaluate the inverse of the Vavilov cummulative probability density function,
       and set kappa and beta2, if necessary
       
       @param z The argument \f$z\f$, which must be in the range \f$0 \le z \le 1\f$
       @param kappa The parameter \f$\kappa\f$, which should be in the range \f$0.01 \le \kappa \le 10 \f$ 
       @param beta2 The parameter \f$\beta^2\f$, which must be in the range \f$0 \le \beta^2 \le 1 \f$ 
   */
   virtual double Quantile (double z, double kappa, double beta2) = 0;
  
   /** 
       Evaluate the inverse of the complementary Vavilov cummulative probability density function
       
       @param z The argument \f$z\f$, which must be in the range \f$0 \le z \le 1\f$
   */
   virtual double Quantile_c (double z) const = 0;
  
   /** 
       Evaluate the inverse of the complementary Vavilov cummulative probability density function,
       and set kappa and beta2, if necessary
       
       @param z The argument \f$z\f$, which must be in the range \f$0 \le z \le 1\f$
       @param kappa The parameter \f$\kappa\f$, which should be in the range \f$0.01 \le \kappa \le 10 \f$ 
       @param beta2 The parameter \f$\beta^2\f$, which must be in the range \f$0 \le \beta^2 \le 1 \f$ 
   */
   virtual double Quantile_c (double z, double kappa, double beta2) = 0;

   /**
      Change \f$\kappa\f$ and \f$\beta^2\f$ and recalculate coefficients if necessary

       @param kappa The parameter \f$\kappa\f$, which should be in the range \f$0.01 \le \kappa \le 10 \f$ 
       @param beta2 The parameter \f$\beta^2\f$, which must be in the range \f$0 \le \beta^2 \le 1 \f$ 
   */
   virtual void SetKappaBeta2 (double kappa, double beta2) = 0; 

   /**
      Return the minimum value of \f$\lambda\f$ for which \f$p(\lambda; \kappa, \beta^2)\f$
      is nonzero in the current approximation
   */
   virtual double GetLambdaMin() const = 0;

   /**
      Return the maximum value of \f$\lambda\f$ for which \f$p(\lambda; \kappa, \beta^2)\f$
      is nonzero in the current approximation
   */
   virtual double GetLambdaMax() const = 0;

   /**
      Return the current value of \f$\kappa\f$
   */
   virtual double GetKappa()     const = 0;

   /**
      Return the current value of \f$\beta^2\f$
   */
   virtual double GetBeta2()     const = 0;

   /**
      Return the value of \f$\lambda\f$ where the pdf is maximal
   */
   virtual double Mode() const;
 
   /**
      Return the value of \f$\lambda\f$ where the pdf is maximal function,
       and set kappa and beta2, if necessary

       @param kappa The parameter \f$\kappa\f$, which should be in the range \f$0.01 \le \kappa \le 10 \f$ 
       @param beta2 The parameter \f$\beta^2\f$, which must be in the range \f$0 \le \beta^2 \le 1 \f$ 
   */
   virtual double Mode(double kappa, double beta2);

   /**
      Return the theoretical mean \f$\mu = \gamma-1- \ln \kappa - \beta^2\f$,
      where \f$\gamma = 0.5772\dots\f$ is Euler's constant
   */
   virtual double Mean() const;

   /**
      Return the theoretical variance \f$\sigma^2 = \frac{1 - \beta^2/2}{\kappa}\f$
   */
   virtual double Variance() const;

   /**
      Return the theoretical skewness
      \f$\gamma_1 = \frac{1/2 - \beta^2/3}{\kappa^2 \sigma^3} \f$
   */
   virtual double Skewness() const;

   /**
      Return the theoretical kurtosis
      \f$\gamma_2 = \frac{1/3 - \beta^2/4}{\kappa^3 \sigma^4}\f$
   */
   virtual double Kurtosis() const;
    
   /**
      Return the theoretical Mean \f$\mu = \gamma-1- \ln \kappa - \beta^2\f$

       @param kappa The parameter \f$\kappa\f$, which should be in the range \f$0.01 \le \kappa \le 10 \f$ 
       @param beta2 The parameter \f$\beta^2\f$, which must be in the range \f$0 \le \beta^2 \le 1 \f$ 
   */
   static double Mean(double kappa, double beta2);
 
   /**
      Return the theoretical Variance \f$\sigma^2 = \frac{1 - \beta^2/2}{\kappa}\f$

       @param kappa The parameter \f$\kappa\f$, which should be in the range \f$0.01 \le \kappa \le 10 \f$ 
       @param beta2 The parameter \f$\beta^2\f$, which must be in the range \f$0 \le \beta^2 \le 1 \f$ 
   */
   static double Variance(double kappa, double beta2);
 
   /**
      Return the theoretical skewness
      \f$\gamma_1 = \frac{1/2 - \beta^2/3}{\kappa^2 \sigma^3} \f$

       @param kappa The parameter \f$\kappa\f$, which should be in the range \f$0.01 \le \kappa \le 10 \f$ 
       @param beta2 The parameter \f$\beta^2\f$, which must be in the range \f$0 \le \beta^2 \le 1 \f$ 
   */
   static double Skewness(double kappa, double beta2);
 
   /**
      Return the theoretical kurtosis
      \f$\gamma_2 = \frac{1/3 - \beta^2/4}{\kappa^3 \sigma^4}\f$

       @param kappa The parameter \f$\kappa\f$, which should be in the range \f$0.01 \le \kappa \le 10 \f$ 
       @param beta2 The parameter \f$\beta^2\f$, which must be in the range \f$0 \le \beta^2 \le 1 \f$ 
   */
   static double Kurtosis(double kappa, double beta2);
     
                
}; 

} // namespace Math
} // namespace ROOT

#endif /* ROOT_Math_Vavilov */
