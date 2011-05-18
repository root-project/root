// @(#)root/mathmore:$Id$

// CodeCogs GNU General Public License Agreement
// Copyright (C) 2004-2005 CodeCogs, Zyba Ltd, Broadwood, Holford, TA5 1DU,
// England.
//
// This program is free software; you can redistribute it and/or modify it
// under
// the terms of the GNU General Public License as published by CodeCogs. 
// You must retain a copy of this licence in all copies. 
//
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A
// PARTICULAR PURPOSE. See the Adapted GNU General Public License for more
// details.
//
// *** THIS SOFTWARE CAN NOT BE USED FOR COMMERCIAL GAIN. ***
// ---------------------------------------------------------------------------------

#ifndef ROOT_Math_KelvinFunctions
#define ROOT_Math_KelvinFunctions

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// KelvinFunctions                                                      //
//                                                                      //
// Calculates the Kelvin Functions Ber(x), Bei(x), Ker(x), Kei(x), and  //
// their first derivatives.                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


namespace ROOT {
namespace Math {
   
class KelvinFunctions
{
  public:
    // The Kelvin functions and their first derivatives
    static double Ber(double x);
    static double Bei(double x);
    static double Ker(double x);
    static double Kei(double x);
    static double DBer(double x);
    static double DBei(double x);
    static double DKer(double x);
    static double DKei(double x);

    // Utility functions appearing in the calculations of the Kelvin
    // functions.
    static double F1(double x);
    static double F2(double x);
    static double G1(double x);
    static double G2(double x);
    static double M(double x);
    static double Theta(double x);
    static double N(double x);
    static double Phi(double x);

    // Include and empty virtual desctructor to eliminate compiler warnings
    virtual ~KelvinFunctions() {}

  protected:
    // Internal parameters used to control calculation method and convegence
    static double fgMin;     
    static double fgEpsilon;

};

} // namespace Math
} // namespace ROOT
 
 
#endif 
 
