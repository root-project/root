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

      GSLMCIntegrationWorkspace()  {}

      virtual ~GSLMCIntegrationWorkspace() { Clear(); }

      virtual MCIntegration::Type Type() const = 0;

      virtual size_t NDim() const { return 0; }

      /// initialize the workspace creating the GSL pointer if it is not there
      virtual bool Init(size_t dim) = 0;

      /// re-initialize an existing the workspace
      virtual bool ReInit() = 0;

      /// free the workspace deleting the GSL pointer
      virtual void Clear() {}

      /// retrieve option pointer corresponding to parameters
      /// create a new object to be managed by the user
      virtual std::unique_ptr<ROOT::Math::IOptions>  Options() const = 0;

      /// set options
      virtual void SetOptions(const ROOT::Math::IOptions &)  = 0;

   private:


   };

   /**
      workspace for VEGAS
    */
   class GSLVegasIntegrationWorkspace : public GSLMCIntegrationWorkspace {

   public :

      GSLVegasIntegrationWorkspace(size_t dim = 0) :
         fWs(nullptr)
      {
         if (dim > 0) Init(dim);
      }

      bool Init(size_t dim) override {
         fWs = gsl_monte_vegas_alloc( dim);
         if (fWs) SetVegasParameters();
         return (fWs != nullptr);
      }

      bool ReInit() override {
         // according to the code - reinit just reset default GSL values
         if (!fWs) return false;
         int iret = gsl_monte_vegas_init( fWs );
         SetVegasParameters();
         return (iret == 0);
      }

      void Clear() override {
         if (fWs) gsl_monte_vegas_free( fWs);
         fWs = nullptr;
      }

      gsl_monte_vegas_state * GetWS() { return fWs; }

      void SetParameters(const struct VegasParameters &p) {
         fParams = p;
         if (fWs) SetVegasParameters();
      }

      size_t NDim() const override { return (fWs) ? fWs->dim : 0; }

      double Result() const {  return (fWs) ? fWs->result : -1;}

      double Sigma() const {  return (fWs) ? fWs->sigma : 0;}

      double Chisq() const {  return (fWs) ? fWs->chisq: -1;}

      MCIntegration::Type Type() const override { return MCIntegration::kVEGAS; }

      const VegasParameters & Parameters() const { return fParams; }
      VegasParameters & Parameters()  { return fParams; }

      std::unique_ptr<IOptions> Options() const override {
         return fParams();
      }
      /// set options
      virtual void SetOptions(const ROOT::Math::IOptions & opt) override {
         SetParameters(VegasParameters(opt));
      }

   private:

      void SetVegasParameters() {
         fWs->alpha       = fParams.alpha;
         fWs->iterations  = fParams.iterations;
         fWs->stage       = fParams.stage;
         fWs->mode        = fParams.mode;
         fWs->verbose     = fParams.verbose;
      }


      gsl_monte_vegas_state * fWs;
      VegasParameters fParams;

   };


   /**
      Workspace for MISER
    */
   class GSLMiserIntegrationWorkspace : public GSLMCIntegrationWorkspace {

   public :

      GSLMiserIntegrationWorkspace(size_t dim = 0) :
         fHaveNewParams(false),
         fWs(nullptr)
      {
         if (dim > 0) Init(dim);
      }


      bool Init(size_t dim) override {
         fWs = gsl_monte_miser_alloc( dim);
         // need this to set parameters according to dimension
         if (!fHaveNewParams) fParams = MiserParameters(dim);
         if (fWs) SetMiserParameters();
         return (fWs != nullptr);
      }

      bool ReInit() override {
         // according to the code - reinit just reset default GSL values
         if (!fWs) return false;
         int iret = gsl_monte_miser_init( fWs );
         SetMiserParameters();
         return (iret == 0);
      }

      void Clear() override {
         if (fWs) gsl_monte_miser_free( fWs);
         fWs = nullptr;
      }

      gsl_monte_miser_state * GetWS() { return fWs; }

      void SetParameters(const MiserParameters &p) {
         fParams = p;
         fHaveNewParams = true;
         if (fWs) SetMiserParameters();
      }

      size_t NDim() const override { return (fWs) ? fWs->dim : 0; }

      MCIntegration::Type Type() const override { return MCIntegration::kMISER; }


      const MiserParameters & Parameters() const { return fParams; }
      MiserParameters & Parameters()  { return fParams; }

      std::unique_ptr<ROOT::Math::IOptions> Options() const override {
         return fParams();
      }
      virtual void SetOptions(const ROOT::Math::IOptions & opt) override {
         SetParameters(MiserParameters(opt));
      }

   private:

      void SetMiserParameters()
      {
         fWs->estimate_frac           = fParams.estimate_frac;
         fWs->min_calls               = fParams.min_calls;
         fWs->min_calls_per_bisection = fParams.min_calls_per_bisection;
         fWs->alpha                   = fParams.alpha;
         fWs->dither                  = fParams.dither;
      }


      bool fHaveNewParams;
      gsl_monte_miser_state * fWs;
      MiserParameters fParams;

   };




   class GSLPlainIntegrationWorkspace : public GSLMCIntegrationWorkspace{

   public :

      GSLPlainIntegrationWorkspace() :
         fWs(nullptr)
      {  }

      bool Init(size_t dim) override {
         fWs = gsl_monte_plain_alloc( dim);
         // no parameter exists for plain
         return (fWs != nullptr);
      }

      bool ReInit() override {
         if (!fWs) return false;
         int iret = gsl_monte_plain_init( fWs );
         return (iret == GSL_SUCCESS);
      }

      void Clear() override {
         if (fWs) gsl_monte_plain_free( fWs);
         fWs = nullptr;
      }

      gsl_monte_plain_state * GetWS() { return fWs; }

      //void SetParameters(const struct PlainParameters &p);

      MCIntegration::Type Type() const override { return MCIntegration::kPLAIN; }

      size_t NDim() const override { return (fWs) ? fWs->dim : 0; }

      std::unique_ptr<ROOT::Math::IOptions>  Options() const override {
         return std::unique_ptr<ROOT::Math::IOptions>();
      }

      virtual void SetOptions(const ROOT::Math::IOptions &) override {}


   private:

      gsl_monte_plain_state * fWs;


   };


} // namespace Math
} // namespace ROOT




#endif /* ROOT_Math_GSLMCIntegrationWorkspace */
