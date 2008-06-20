// @(#)root/mathmore:$Id$
// Author: Magdalena Slawinska  08/2007

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2007 ROOT Foundation,  CERN/PH-SFT                   *
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
// Author: Magdalena Slawinska
// 



#ifndef ROOT_Math_GSLMCIntegrationWorkspace
#define ROOT_Math_GSLMCIntegrationWorkspace

#include "gsl/gsl_math.h"
#include "gsl/gsl_monte.h"
#include "gsl/gsl_monte_vegas.h"
#include "gsl/gsl_monte_miser.h"
#include "gsl/gsl_monte_plain.h"

#include "Math/MCParameters.h"
#include "Math/MCIntegrationTypes.h"

namespace ROOT {
namespace Math {


   
   class GSLMCIntegrationWorkspace { 

   public :

      GSLMCIntegrationWorkspace(unsigned int dim) : 
         fDim(dim)
      {}

      virtual ~GSLMCIntegrationWorkspace() { }

      virtual MCIntegration::Type Type() const = 0;  

      unsigned int NDim() const { return fDim; } 


   private:

      unsigned int fDim;  // workspace dimension (must be equal o functino dimension)

   };

   class GSLVegasIntegrationWorkspace : public GSLMCIntegrationWorkspace { 

   public :
      
      GSLVegasIntegrationWorkspace(size_t dim) : 
         GSLMCIntegrationWorkspace (dim)
      {
         fWs = gsl_monte_vegas_alloc( dim);
      }
      ~GSLVegasIntegrationWorkspace() {
         gsl_monte_vegas_free( fWs);
      }

      gsl_monte_vegas_state * GetWS() { return fWs; }
      void SetParameters();
      void SetParameters(const struct VegasParameters &p);
      double Sigma() const;

      MCIntegration::Type Type() const { return MCIntegration::kVEGAS; }
      
   private: 
      gsl_monte_vegas_state * fWs; 

   };

   void GSLVegasIntegrationWorkspace::SetParameters(const struct VegasParameters &p)
   {
    
      fWs->alpha = p.alpha;
      fWs->iterations= p.iterations;

   }
   double GSLVegasIntegrationWorkspace::Sigma()const {return fWs->sigma;}


   class GSLMiserIntegrationWorkspace : public GSLMCIntegrationWorkspace { 

   public :
      
      GSLMiserIntegrationWorkspace(size_t dim) : 
         GSLMCIntegrationWorkspace (dim)
      {
         fWs = gsl_monte_miser_alloc( dim);
      }
      ~GSLMiserIntegrationWorkspace() {
         gsl_monte_miser_free( fWs);
      }

      gsl_monte_miser_state * GetWS() { return fWs; }
      void SetParameters();
      void SetParameters(const struct MiserParameters &p);

      MCIntegration::Type Type() const { return MCIntegration::kMISER; }
    
   private: 
      gsl_monte_miser_state * fWs; 

   };

   void GSLMiserIntegrationWorkspace::SetParameters(const struct MiserParameters &p)
   {
      fWs->estimate_frac = p.estimate_frac;
      fWs->min_calls = p.min_calls;
      fWs->min_calls_per_bisection = p.min_calls_per_bisection;
      fWs->alpha = p.alpha;

   }




   class GSLPlainIntegrationWorkspace : public GSLMCIntegrationWorkspace{ 

   public :
      
      GSLPlainIntegrationWorkspace(size_t dim) : 
         GSLMCIntegrationWorkspace (dim)
      {
         fWs = gsl_monte_plain_alloc( dim);
      }
      ~GSLPlainIntegrationWorkspace() {
         gsl_monte_plain_free( fWs);
      }

      gsl_monte_plain_state * GetWS() { return fWs; }
      //void SetParameters();
      //void SetParameters(const struct PlainParameters &p);

      MCIntegration::Type Type() const { return MCIntegration::kPLAIN; }

   private: 
      gsl_monte_plain_state * fWs; 

   
   };


} // namespace Math
} // namespace ROOT




#endif /* ROOT_Math_GSLMCIntegrationWorkspace */
