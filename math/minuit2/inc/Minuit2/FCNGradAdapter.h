// @(#)root/minuit2:$Id$
// Authors: L. Moneta, E.G.P. Bos   2006-2017

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006 ROOT Foundation,  CERN/PH-SFT                   *
 * Copyright (c) 2017 Patrick Bos, Netherlands eScience Center        *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_FCNGradAdapter
#define ROOT_Minuit2_FCNGradAdapter

#include "Minuit2/FCNGradientBase.h"

//#define DEBUG
#ifdef DEBUG
#include <iostream>
#endif

namespace ROOT {

  namespace Minuit2 {

    /**

    template wrapped class for adapting to FCNBase signature a IGradFunction

    @author Lorenzo Moneta, Patrick Bos

    @ingroup Minuit

    */

    template< class Function>
    class FCNGradAdapter : public FCNGradientBase {

    public:

      FCNGradAdapter(const Function & f, double up = 1.) :
          fFunc(f) ,
          fUp (up) ,
          fGrad(std::vector<double>(fFunc.NDim() ) ),
          fG2(fFunc.hasG2ndDerivative() ? std::vector<double>(fFunc.NDim()) : std::vector<double>(0) ),
          fGStep(fFunc.hasGStepSize()   ? std::vector<double>(fFunc.NDim()) : std::vector<double>(0) )
      {}

      ~FCNGradAdapter() {}


      double operator()(const std::vector<double>& v) const {
        return fFunc.operator()(&v[0]);
      }
      double operator()(const double *  v) const {
        return fFunc.operator()(v);
      }

      double Up() const {return fUp;}

      virtual std::vector<double> Gradient(const std::vector<double>& v) const {
        fFunc.Gradient(v.data(), fGrad.data());

#ifdef DEBUG
        std::cout << " gradient in FCNAdapter = { " ;
      for (unsigned int i = 0; i < fGrad.size(); ++i)
         std::cout << fGrad[i] << "\t";
      std::cout << "}" << std::endl;
#endif
        return fGrad;
      }
      // forward interface
      //virtual double operator()(int npar, double* params,int iflag = 4) const;
      bool CheckGradient() const { return false; }

      virtual std::vector<double> G2ndDerivative(const std::vector<double>& v) const {
        fFunc.G2ndDerivative(v.data(), fG2.data());
        return fG2;
      };

      virtual std::vector<double> GStepSize(const std::vector<double>& v) const {
        fFunc.GStepSize(v.data(), fGStep.data());
        return fGStep;
      };

      virtual bool hasG2ndDerivative() const {
        return fFunc.hasG2ndDerivative();
      }

      virtual bool hasGStepSize() const {
        return fFunc.hasGStepSize();
      }

    private:
      const Function & fFunc;
      double fUp;
      mutable std::vector<double> fGrad;
      mutable std::vector<double> fG2;
      mutable std::vector<double> fGStep;
    };

  } // end namespace Minuit2

} // end namespace ROOT



#endif //ROOT_Minuit2_FCNGradAdapter
