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

// Header file for class GSLRootFSolver
//
// Created by: moneta  at Sun Nov 14 17:23:06 2004
//
// Last update: Sun Nov 14 17:23:06 2004
//
#ifndef ROOT_Math_GSLRootFSolver
#define ROOT_Math_GSLRootFSolver


#include "gsl/gsl_roots.h"


namespace ROOT {
namespace Math {


/**
   Root-Finder implementation class using  GSL

   @ingroup RootFinders
 */
class GSLRootFSolver {

public:

  GSLRootFSolver(const gsl_root_fsolver_type * type) {
    fSolver = gsl_root_fsolver_alloc( type);
  }
  virtual ~GSLRootFSolver() {
    gsl_root_fsolver_free( fSolver);
  }

/* private: */
/* // usually copying is non trivial, so we make this unaccessible */
/*   GSLRootFSolver(const GSLRootFSolver &);  */
/*   GSLRootFSolver & operator = (const GSLRootFSolver &);  */

/* public:  */

  gsl_root_fsolver * Solver() const { return fSolver; }


protected:


private:

  gsl_root_fsolver *fSolver;


};

} // namespace Math
} // namespace ROOT


#endif /* ROOT_Math_GSLRootFSolver */
