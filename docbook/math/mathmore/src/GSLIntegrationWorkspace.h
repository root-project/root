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

// Header file for class GSLIntegratorWorkspace
// 
// Created by: moneta  at Sat Nov 13 14:55:27 2004
// 
// Last update: Sat Nov 13 14:55:27 2004
// 
#ifndef ROOT_Math_GSLIntegrationWorkspace
#define ROOT_Math_GSLIntegrationWorkspace


namespace ROOT {
namespace Math {

#include "gsl/gsl_math.h"


#include "gsl/gsl_integration.h"


  class GSLIntegrationWorkspace { 

    public :
      
      GSLIntegrationWorkspace(size_t n) {
            fWs = gsl_integration_workspace_alloc( n);
    }
    ~GSLIntegrationWorkspace() {
            gsl_integration_workspace_free( fWs);
    }

    gsl_integration_workspace * GetWS() { return fWs; }

  private: 
    gsl_integration_workspace * fWs; 

  };


} // namespace Math
} // namespace ROOT


#endif /* ROOT_Math_GSLIntegrationWorkspace */
