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

// Header file for class GSLError

#ifndef ROOT_Math_GSLError
#define ROOT_Math_GSLError


#if defined(G__DICTIONARY) 

#include "gsl/gsl_errno.h"

#include "TError.h"
#include "TSystem.h"


namespace ROOT { 
   namespace Math { 

      /**
         class to change GSL Error handler to use ROOT one. 
         It is used only when building the dictionary (G__DICTIONARY is defined) 
         and not in the stand-alone version of the library. 
         In that case the default GSL error handler is used  
       */

      class GSLError { 

      public: 
         
         GSLError() { 
               gsl_set_error_handler(&GSLError::Handler);
            // set a new handler for GSL 
         }

         static void Handler(const char * reason, const char * file, int line, int gsl_errno)  { 

            Error("GSLError","Error %d in %s at %d : %s",gsl_errno,file,line,reason);
            
         }
      }; 

   }
}

// re-define the default error handler when loading the library
ROOT::Math::GSLError gGSLError; 


#endif

#endif /* ROOT_Math_GSLError */
