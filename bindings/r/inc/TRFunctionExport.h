// @(#)root/r:$Id$
// Author: Omar Zapata  Omar.Zapata@cern.ch   16/06/2013


/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_R_TRFunctionExport
#define ROOT_R_TRFunctionExport

#include <TRInternalFunction.h>


namespace ROOT {
   namespace R {

      /**
      \class TRFunctionExport

      This is a class to pass functions from ROOT to R
      <center><h2>TRFunctionExport class</h2></center>
      <p>
      The TRFunctionExport class lets you pass ROOT's functions to R's environment<br>
      </p>
      <p>
      The next example was based in <br>
      <a href="https://root.cern/doc/master/NumericalMinimization_8C.html">
      https://root.cern/doc/master/NumericalMinimization_8C.html
      </a><br>
      <a href="http://stat.ethz.ch/R-manual/R-devel/library/stats/html/optim.html">
      http://stat.ethz.ch/R-manual/R-devel/library/stats/html/optim.html</a><br>

      </p>

      Let \f$ f(x,y)=(x-1)^{2} + 100(y-x^{2})^{2} \f$ , which is called the Rosenbrock
      function.

      It's known that this function has a minimum when \f$ y = x^{2}\f$ , and \f$ x = 1.\f$
      Let's get the minimum using R's optim package through ROOTR's interface.
      In the code this function was called "Double_t RosenBrock(const TVectorD xx )", because for
      optim, the input in your function definition must be a single vector.

      The Gradient is formed by

      \f$ \frac{\partial f}{\partial x} =  -400x(y - x^{2}) - 2(1 - x) \f$

      \f$  \frac{\partial f}{\partial y} =  200(y - x^{2}); \f$

      The "TVectorD RosenBrockGrad(const TVectorD xx )" function
      must have  a single vector as the argument a it will return a single vetor.

      \code{.cpp}
      #include<TRInterface.h>

      //in the next function the pointer *double must be changed by TVectorD, because the pointer has no
      //sense in R's environment.
      Double_t RosenBrock(const TVectorD xx )
      {
        const Double_t x = xx[0];
        const Double_t y = xx[1];
        const Double_t tmp1 = y-x*x;
        const Double_t tmp2 = 1-x;
        return 100*tmp1*tmp1+tmp2*tmp2;
      }

      TVectorD RosenBrockGrad(const TVectorD xx )
      {
        const Double_t x = xx[0];
        const Double_t y = xx[1];
        TVectorD grad(2);
        grad[0]=-400 * x * (y - x * x) - 2 * (1 - x);
        grad[1]=200 * (y - x * x);
        return grad;
      }


      void Minimization()
      {
       ROOT::R::TRInterface &r=ROOT::R::TRInterface::Instance();
       //passing RosenBrock function to R
       r["RosenBrock"]<<ROOT::R::TRFunctionExport(RosenBrock);

       //passing RosenBrockGrad function to R
       r["RosenBrockGrad"]<<ROOT::R::TRFunctionExport(RosenBrockGrad);

       //the option "method" could be "Nelder-Mead", "BFGS", "CG", "L-BFGS-B", "SANN","Brent"
       //the option "control" lets you put some constraints like:
       //"maxit" The maximum number of iterations
       //"abstol" The absolute convergence tolerance.
       //"reltol" Relative convergence tolerance.
       r<<"result <- optim( c(0.01,0.01), RosenBrock,method='BFGS',control = list(maxit = 1000000) )";

       //Getting results from R
       TVectorD  min=r.Eval("result$par");

       std::cout.precision(8);
       //printing results
       std::cout<<"-----------------------------------------"<<std::endl;
       std::cout<<"Minimum x="<<min[0]<<" y="<<min[1]<<std::endl;
       std::cout<<"Value at minimum ="<<RosenBrock(min)<<std::endl;

       //using the gradient
       r<<"optimHess(result$par, RosenBrock, RosenBrockGrad)";
       r<<"hresult <- optim(c(-1.2,1), RosenBrock, NULL, method = 'BFGS', hessian = TRUE)";
       //getting the minimum calculated with the gradient
       TVectorD  hmin=r.Eval("hresult$par");

       //printing results
       std::cout<<"-----------------------------------------"<<std::endl;
       std::cout<<"Minimization with the Gradient"<<endl;
       std::cout<<"Minimum x="<<hmin[0]<<" y="<<hmin[1]<<std::endl;
       std::cout<<"Value at minimum ="<<RosenBrock(hmin)<<std::endl;

      }
      \endcode

      Output
      \code
      Processing Minimization.C...
      -----------------------------------------
      Minimum x=0.99980006 y=0.99960016
      Value at minimum =3.9974288e-08
      -----------------------------------------
      Minimization with the Gradient
      Minimum x=0.99980443 y=0.99960838
      Value at minimum =3.8273828e-08
      \endcode
      <h2>Users Guide </h2>
      <a href="https://oproject.org/pages/ROOT%20R%20Users%20Guide"> https://oproject.org/pages/ROOT R Users Guide</a><br>

         @ingroup R
      */


      class TRInterface;
      class TRFunctionExport: public TObject {
         friend class TRInterface;
         friend SEXP Rcpp::wrap<TRFunctionExport>(const TRFunctionExport &f);
      protected:
         TRInternalFunction *f; //Internar Function to export
      public:
         /**
         Default TRFunctionExport constructor
         */
         TRFunctionExport();

         /**
         Default TRFunctionExport destructor
         */
         ~TRFunctionExport()
         {
            if (f) delete f;
         }
         /**
         TRFunctionExport copy constructor
         \param fun other TRFunctionExport
         */
         TRFunctionExport(const TRFunctionExport &fun);

         /**
         TRFunctionExport template constructor that supports a lot of function's prototypes
         \param fun supported function to be wrapped by Rcpp
         */
         template<class T> TRFunctionExport(T fun)
         {
            f = new TRInternalFunction(fun);
         }

         /**
         function to assign function to export,
         template method that supports a lot of function's prototypes
         \param fun supported function to be wrapped by Rcpp
         */
         template<class T> void SetFunction(T fun)
         {
            f = new TRInternalFunction(fun);
         }

         ClassDef(TRFunctionExport, 0) //
      };
   }
}



#endif
