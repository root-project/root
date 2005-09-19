// @(#)root/mathmore:$Name:  $:$Id: Polynomial.cxx,v 1.2 2005/09/18 20:41:25 brun Exp $
// Authors: L. Moneta, A. Zsenei   08/2005 

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

// Implementation file for class Polynomial
// 
// Created by: Lorenzo Moneta  at Wed Nov 10 17:46:19 2004
// 
// Last update: Wed Nov 10 17:46:19 2004
// 

#include "Math/Polynomial.h"


#include "gsl/gsl_math.h"
#include "gsl/gsl_errno.h"
#include "gsl/gsl_poly.h"



namespace ROOT {
namespace Math {


Polynomial::Polynomial(unsigned int n) : 
  // number of par is order + 1
  ParamFunction( n+1, true, true), 
  fOrder(n), 
  fDerived_params(std::vector<double>(n) )
{
}


Polynomial::~Polynomial() 
{
}

// Polynomial::Polynomial(const Polynomial &) 
// {
// }

// Polynomial & Polynomial::operator = (const Polynomial &rhs) 
// {
//    if (this == &rhs) return *this;  // time saving self-test

//    return *this;
// }


double  Polynomial::operator() (double x) { 
  
    return gsl_poly_eval( &fParams.front(), fOrder + 1, x); 

}


double  Polynomial::operator() (double x, const std::vector<double> & p) { 
  
    return gsl_poly_eval( &p.front(), fOrder + 1, x); 

}


double  Polynomial::Gradient(double x) { 

   for ( unsigned int i = 0; i < fOrder; ++i ) 
    fDerived_params[i] =  (i + 1) * Parameters()[i+1]; 

   return gsl_poly_eval( &fDerived_params.front(), fOrder, x); 

}

const std::vector<double> &  Polynomial::ParameterGradient (double x) { 

    for (unsigned int i = 0; i < fParGradient.size(); ++i) 
      fParGradient[i] = gsl_pow_int(x, i); 
  
    return fParGradient; 

}



IGenFunction * Polynomial::Clone() const { 
    Polynomial * f =  new Polynomial(fOrder);
    f->fDerived_params = fDerived_params; 
    f->SetParameters( Parameters() ); 
    return f; 
}
	

const std::vector< std::complex <double> > &  Polynomial::FindRoots(){


    // check if order is correct
    unsigned int n = fOrder;
    while ( Parameters()[n] == 0 ) { 
      n--;
    } 

    fRoots.reserve(n);  


    if (n == 0) {   
      return fRoots;
    } 
    else if (n == 1 ) { 
      if ( Parameters()[1] == 0) return fRoots;
      double r = - Parameters()[0]/ Parameters()[1]; 
      fRoots.push_back( std::complex<double> ( r, 0.0) ); 
    }
    // quadratic equations
    else if (n == 2 ) { 
      gsl_complex z1, z2;
      int status = gsl_poly_complex_solve_quadratic(Parameters()[2], Parameters()[1], Parameters()[0], &z1, &z2); 
      if (status != GSL_SUCCESS) return fRoots; 
      fRoots.push_back(std::complex<double>(z1.dat[0],z1.dat[1]) ); 
      fRoots.push_back(std::complex<double>(z2.dat[0],z2.dat[1]) );       
    }
    // cubic equations
    else if (n == 3 ) { 
      gsl_complex  z1, z2, z3;   
      // renormmalize params in this case
      double w = Parameters()[3]; 
      double a = Parameters()[2]/w;
      double b = Parameters()[1]/w;
      double c = Parameters()[0]/w;
      int status = gsl_poly_complex_solve_cubic(a, b, c, &z1, &z2, &z3); 
      if (status != GSL_SUCCESS) return fRoots; 
      fRoots.push_back(std::complex<double> (z1.dat[0],z1.dat[1]) ); 
      fRoots.push_back(std::complex<double> (z2.dat[0],z2.dat[1]) ); 
      fRoots.push_back(std::complex<double> (z3.dat[0],z3.dat[1]) );       
    }
    // cubic equations
    //else if (n == 4 ) { 
      // quartic eq. (t.b.d.) 
    //}
    // for higher order polynomial use numerical fRoots
    else { 
      gsl_poly_complex_workspace * w = gsl_poly_complex_workspace_alloc( n + 1); 
      std::vector<double> z(2*n);
      int status = gsl_poly_complex_solve (&Parameters().front(), n+1, w, &z.front() );  
      gsl_poly_complex_workspace_free(w);
      if (status != GSL_SUCCESS) return fRoots; 
      for (unsigned int i = 0; i < n; ++i) 
	fRoots.push_back(std::complex<double> (z[2*i],z[2*i+1] ) ); 
    }      

    return fRoots; 

  }




} // namespace Math
} // namespace ROOT
