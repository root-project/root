// @(#)root/mathmore:$Name:  $:$Id: InterpolationTypes.h,v 1.1 2005/09/18 17:33:47 brun Exp $
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

// Header file for class IntegrationTypes
// 
// Created by: moneta  at Fri Nov 26 15:40:58 2004
// 
// Last update: Fri Nov 26 15:40:58 2004
// 
#ifndef ROOT_Math_IntegrationTypes
#define ROOT_Math_IntegrationTypes


namespace ROOT {
namespace Math {



  namespace Integration {

    /**
	 enumeration specifying the integration types.
	 <ul>
	 <li>NONADAPTIVE : to be used for smooth functions
	 <li>ADAPTIVE : to be used for general functions without singularities.
	 <li>ADAPTIVESINGULAR: default adaptive integration type which can be used in the case of the presence of singularities.
	 </ul>
	 @ingroup Integration
    */

    // type of integration

    enum Type { NONADAPTIVE, ADAPTIVE, ADAPTIVESINGULAR } ;


    /**
	 enumeration specifying the Gauss-KronRod integration rule for ADAPTIVE integration type
	 @ingroup Integration
    */
    // Gauss KronRod Adaptive rule

    enum GKRule { GAUSS15 = 1,
		  GAUSS21 = 2,
		  GAUSS31 = 3,
		  GAUSS41 = 4,
		  GAUSS51 = 5,
		  GAUSS61 = 6
    };


  }    // end namespace Integration


} // namespace Math
} // namespace ROOT

#endif /* ROOT_Math_InterpolationTypes */
