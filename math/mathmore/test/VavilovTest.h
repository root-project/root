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

// Header file for class VavilovTest
//
// Created by: blist  at Thu Apr 29 11:19:00 2010
//
// Last update: Thu Apr 29 11:19:00 2010
//
#ifndef ROOT_Math_VavilovTest
#define ROOT_Math_VavilovTest


#include <iostream>

namespace ROOT {
namespace Math {

class Vavilov;

//____________________________________________________________________________
/**
   Test class for class Vavilov and its subclasses

   For test purposes,
   the class contains a number of static function that return the tabulated
   values of the Vavilov pdf given by
   S.M. Seltzer and M.J. Berger: Energy loss stragglin of protons and mesons:
   Tabulation of the Vavilov distribution, pp 187-203
   in: National Research Council (U.S.), Committee on Nuclear Science:
  Studies in penetration of charged particles in matter,
  Nat. Akad. Sci. Publication 1133,
  Nucl. Sci. Series Report No. 39,
  Washington (Nat. Akad. Sci.) 1964, 388 pp.
  Available from
  <A HREF="http://books.google.de/books?id=kmMrAAAAYAAJ&lpg=PP9&pg=PA187#v=onepage&q&f=false">Google books</A>

   B. List 24.6.2010

   */


class VavilovTest {

public:


   /**
      Test the pdf values against the tables of Seltzer and Berger.
      Returns 0 if the test is passed

       @param v The Vavilov test object
       @param os Output stream
   */
   static int PdfTest (Vavilov& v, std::ostream& os,
                       double maxabsdiff, double maxdiffmantissa,
                       double agreefraction, double agreediffmantissa);

   /**
      Test the pdf values against the tables of Seltzer and Berger.
      Returns 0 if the test is passed

       @param v The Vavilov test object
       @param os Output stream
   */
   static int PdfTest (Vavilov& v, std::ostream& os);

   /**
      Test the cdf values against the integral of the pdf
      Returns 0 if the test is passed

       @param v The Vavilov test object
       @param os Output stream
   */
   static int CdfTest (Vavilov& v, std::ostream& os,
                       double maxabsdiff, double maxcdfdiff);

   /**
      Test the cdf values against the integral of the pdf
      Returns 0 if the test is passed

       @param v The Vavilov test object
       @param os Output stream
   */
   static int CdfTest (Vavilov& v, std::ostream& os);

   /**
      Test the quantile values against the cdf
      Returns 0 if the test is passed

       @param v The Vavilov test object
       @param os Output stream
   */
   static int QuantileTest (Vavilov& v, std::ostream& os, double maxabsdiff);

   /**
      Test the quantile values against the cdf
      Returns 0 if the test is passed

       @param v The Vavilov test object
       @param os Output stream
   */
   static int QuantileTest (Vavilov& v, std::ostream& os);

   /**
      Print a table of the pdf values to stream os

       @param v The Vavilov object
       @param os Output stream
       @param digits: Number of digits to be printed
   */
   static void PrintPdfTable (Vavilov& v, std::ostream& os, int digits = 3);


   /**
      Return the number of \f$\kappa\f$ values in the tables of Seltzer and Berger
   */
   static int GetSBNKappa ();

   /**
      Return the \f$\kappa\f$ value for ikappa in the tables of Seltzer and Berger
   */
   static double GetSBKappa (int ikappa);

   /**
      Return the number of \f$\beta^2\f$ values in the tables of Seltzer and Berger
   */
   static int GetSBNBeta2 ();

   /**
      Return the \f$\beta^2\f$ value in the tables of Seltzer and Berger
   */
   static double GetSBBeta2 (int ibeta2);

   /**
      Return the number of \f$\lambda\f$ values in the tables of Seltzer and Berger
   */
   static int GetSBNLambda (int ikappa);

   /**
      Return the \f$\lambda\f$ value in the tables of Seltzer and Berger
   */
   static double GetSBLambda (int ikappa, int ilambda);

   /**
      Return the value of \f$p(\lambda)\f$ in the tables of Seltzer and Berger
   */
   static double GetSBVavilovPdf (int ikappa, int ibeta2, int ilambda);

   static void GetPdfTestParams (const Vavilov& v, double& maxabsdiff, double& maxdiffmantissa, double& agreefraction, double& agreediffmantissa);

   static void GetCdfTestParams (const Vavilov& v, double& maxabsdiff, double& maxcdfdiff);

   static void GetQuantileTestParams (const Vavilov& v, double& maxabsdiff);


};

} // namespace Math
} // namespace ROOT

#endif /* ROOT_Math_VavilovTest */
