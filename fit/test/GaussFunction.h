
#include "Math/IParamFunction.h"
#include <cmath>


class GaussFunction : public ROOT::Math::IParamMultiGradFunction { 

public: 

   enum { 
      kNPar = 3 
   }; 

   GaussFunction(double amp = 1, double mean = 0, double sigma = 1) {
      fParams[0] = amp;
      fParams[1] = mean; 
      fParams[2] = sigma; 
      fLogAmp = std::log(amp);
   }

   unsigned int NDim() const { return 1; }

   unsigned int NPar() const { return kNPar; }

   inline double amp()   const { return fParams[0]; }
   inline double logamp()   const { return fLogAmp; }
   inline double mean()  const { return fParams[1]; }
   inline double sigma() const { return fParams[2]; }

   const double * Parameters() const { return fParams; } 
   
   void SetParameters(const double * p) { std::copy(p,p+kNPar,fParams); /* fLogAmp = std::log( p[0] ); */ }


   ROOT::Math::IMultiGenFunction * Clone() const { return new GaussFunction(amp(), mean(), sigma() ); }


   // implementing this is much faster
   double operator()(const double *x, const double * p) { 
      double y = (x[0]-p[1])/p[2];
      return p[0]*std::exp(-0.5*y*y);
   }

   using  ROOT::Math::IParamMultiGradFunction::operator();

   void ParameterGradient(const double *x, double * g) const { 
      double y = (x[0]-mean())/sigma();
      g[0] = std::exp(-0.5*y*y);
      g[1] =  amp()*g[0]*y/sigma();
      g[2] = g[1]*y; 
   }


private: 


   double DoEval(const double * x) const { 
      double y = (x[0]-mean())/sigma();
      return amp()*std::exp(-0.5*y*y);
   }

   double DoDerivative(const double *x, unsigned int icoord) const { 
      assert (icoord == 0); 
      double dGdx = -(*this)(x) * (x[0]-mean())/(sigma()*sigma());
      return dGdx; 
   }

   double DoParameterDerivative(const double *x, unsigned int ipar) const { 
      double grad[3];
      ParameterGradient(x, &grad[0] ); 
      return grad[ipar]; 
   }


   double fParams[kNPar]; 
   double fLogAmp; 
}; 
