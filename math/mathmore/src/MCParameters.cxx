// @(#)root/mathmore:$Id$
// Author: Lorenzo Moneta  11/2010

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
//
// implementation file for class MCParameters
// Author: Lorenzo Moneta , Nov 2010
//
//
#include "Math/MCParameters.h"
#include "Math/GenAlgoOptions.h"

#include "gsl/gsl_monte_vegas.h"

namespace ROOT {
namespace Math {


   /// default VEGAS parameters (copied from gsl/monte/vegas.c)
   void VegasParameters::SetDefaultValues() {
      // init default values
      alpha        =  1.5;
      iterations   = 5;
      stage        = 0;
      mode         = GSL_VEGAS_MODE_IMPORTANCE;
      verbose      = -1;
   }

   VegasParameters::VegasParameters(const IOptions & opt) {
      SetDefaultValues();
      (*this) = opt;
   }

   VegasParameters & VegasParameters::operator= (const IOptions & opt) {
      // set parameters from IOptions
      double val = 0;
      int ival = 0;
      bool ret = false;

      ret = opt.GetRealValue("alpha",val);
      if (ret) alpha = val;
      ret = opt.GetIntValue("iterations",ival);
      if (ret) iterations = ival;
      ret = opt.GetIntValue("stage",ival);
      if (ret) stage = ival;
      ret = opt.GetIntValue("mode",ival);
      if (ret) mode = ival;
      ret = opt.GetIntValue("verbose",ival);
      if (ret) verbose = ival;
      return *this;
   }

   std::unique_ptr<ROOT::Math::IOptions> VegasParameters::operator() () const {
      // convert to options (return object is managed by the user)
      GenAlgoOptions * opt = new GenAlgoOptions();
      opt->SetRealValue("alpha",alpha);
      opt->SetIntValue("iterations",iterations);
      opt->SetIntValue("stage",stage);
      opt->SetIntValue("mode",mode);
      opt->SetIntValue("verbose",verbose);
      return std::unique_ptr<ROOT::Math::IOptions>(opt);
   }



   /// default MISER parameters (copied from gsl/monte/vegas.c)


   void MiserParameters::SetDefaultValues(size_t dim) {
      // init default values
      estimate_frac           = 0.1;
      min_calls               = (dim>0) ? 16*dim : 160; // use default dim = 10
      min_calls_per_bisection = 32*min_calls;
      dither                  = 0;
      alpha                   = 2.0;
   }


   MiserParameters::MiserParameters(const IOptions & opt, size_t dim) {
      SetDefaultValues(dim);
      (*this) = opt;
   }

   MiserParameters & MiserParameters::operator= (const IOptions & opt) {
      // set parameters from IOptions
      double val = 0;
      int ival = 0;
      bool ret = false;

      ret = opt.GetRealValue("alpha",val);
      if (ret) alpha = val;
      ret = opt.GetRealValue("dither",val);
      if (ret) dither = val;
      ret = opt.GetRealValue("estimate_frac",val);
      if (ret) estimate_frac = val;
      ret = opt.GetIntValue("min_calls",ival);
      if (ret) min_calls = ival;
      ret = opt.GetIntValue("min_calls_per_bisection",ival);
      if (ret) min_calls_per_bisection = ival;
      return *this;
   }

   std::unique_ptr<ROOT::Math::IOptions>  MiserParameters::operator() () const {
      // convert to options (return object is managed by the user)
      GenAlgoOptions * opt = new GenAlgoOptions();
      opt->SetRealValue("alpha",alpha);
      opt->SetRealValue("dither",dither);
      opt->SetRealValue("estimate_frac",estimate_frac);
      opt->SetIntValue("min_calls",min_calls);
      opt->SetIntValue("min_calls_per_bisection",min_calls_per_bisection);
      return std::unique_ptr<ROOT::Math::IOptions>(opt);
   }


} // namespace Math
} // namespace ROOT



