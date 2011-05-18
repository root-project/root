// @(#)root/mathmore:$Id$
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

// Header file for class GSLChebSeries
// 
// Created by: moneta  at Thu Dec  2 16:50:07 2004
// 
// Last update: Thu Dec  2 16:50:07 2004
// 
#ifndef ROOT_Math_GSLChebSeries
#define ROOT_Math_GSLChebSeries

#include "gsl/gsl_chebyshev.h"


namespace ROOT {
namespace Math {

  /**
     wrapper class for C struct gsl_cheb_series 
  */

class GSLChebSeries {

public: 
  GSLChebSeries(size_t n)  
  { 
    m_cs = gsl_cheb_alloc(n); 
  }

  virtual ~GSLChebSeries() { 
    gsl_cheb_free(m_cs); 
  }

private:
// usually copying is non trivial, so we make this unaccessible
  GSLChebSeries(const GSLChebSeries &); 
  GSLChebSeries & operator = (const GSLChebSeries &); 

public: 

  gsl_cheb_series * get() const { return m_cs; }

private: 

  gsl_cheb_series * m_cs; 

}; 


} // namespace Math
} // namespace ROOT


#endif /* ROOT_Math_GSLChebSeries */
