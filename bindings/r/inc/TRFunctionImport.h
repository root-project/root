// @(#)root/r:$Id$
// Author: Omar Zapata  Omar.Zapata@cern.ch  28/06/2015


/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_R_TRFunctionImport
#define ROOT_R_TRFunctionImport


#include <RExports.h>

#include <TRObject.h>

namespace ROOT {
   namespace R {

      /**
      \class TRFunctionImport
      This is a class to pass functions from ROOT to R

      <center><h2>TRFunctionImport class</h2></center>
      <p>
      The TRFunctionImport class lets you call R's functions to ROOT's environment<br>
      The object associated to this class have a set of overloaded operators to use the object like function<br>
      </p>
      \code{.cpp}
      #include<TRInterface.h>

      using namespace ROOT::R;
      void Function()
      {
        TRInterface &r = TRInterface::Instance();
        r.SetVerbose(1);
        ////////////////////////////////////////
        //defining functions to be used from R//
        ////////////////////////////////////////
        TRFunctionImport c("c");
        TRFunctionImport list("list");
        TRFunctionImport asformula("as.formula");
        TRFunctionImport nls("nls");
        TRFunctionImport confint("confint");
        TRFunctionImport summary("summary");
        TRFunctionImport print("print");
        TRFunctionImport plot("plot");
        TRFunctionImport lines("lines");
        TRFunctionImport devnew("dev.new");
        TRFunctionImport devoff("dev.off");
        TRFunctionImport min("min");
        TRFunctionImport max("max");
        TRFunctionImport seq("seq");
        TRFunctionImport predict("predict");

      r<<"options(device='png')";//enable plot in png file

        ////////////////////////
        //doing the procedure //
        ////////////////////////
         TRObject xdata = c(-2,-1.64,-1.33,-0.7,0,0.45,1.2,1.64,2.32,2.9);
         TRObject ydata = c(0.699369,0.700462,0.695354,1.03905,1.97389,2.41143,1.91091,0.919576,-0.730975,-1.42001);

         TRDataFrame data;
         data["xdata"]=xdata;
         data["ydata"]=ydata;

         //fit = nls(ydata ~ p1*cos(p2*xdata) + p2*sin(p1*xdata), start=list(p1=1,p2=0.2)) <- R code
         TRObject fit = nls(asformula("ydata ~ p1*cos(p2*xdata) + p2*sin(p1*xdata)"),Label["data"]=data, Label["start"]=list(Label["p1"]=1,Label["p2"]=0.2));
         print(summary(fit));

         print(confint(fit));

         devnew("Fitting Regression");
         plot(xdata,ydata);

         TRObject xgrid=seq(min(xdata),max(xdata),Label["len"]=10);
         lines(xgrid,predict(fit,xgrid),Label["col"] = "green");
         devoff();
      }
      \endcode

      Output
      \code
      Formula: ydata ~ p1 * cos(p2 * xdata) + p2 * sin(p1 * xdata)

      Parameters:
         Estimate Std. Error t value Pr(>|t|)
      p1 1.881851   0.027430   68.61 2.27e-12 ***
      p2 0.700230   0.009153   76.51 9.50e-13 ***
      ---
      Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

      Residual standard error: 0.08202 on 8 degrees of freedom

      Number of iterations to convergence: 7
      Achieved convergence tolerance: 2.189e-06

      Waiting for profiling to be done...
              2.5%     97.5%
      p1 1.8206081 1.9442365
      p2 0.6794193 0.7209843
      \endcode
      <h2>Users Guide </h2>
      <a href="https://oproject.org/pages/ROOT%20R%20Users%20Guide"> https://oproject.org/pages/ROOT R Users Guide</a><br>

      @ingroup R
      */

      class TRInterface;
      class TRFunctionImport: public TObject {
         friend class TRInterface;
         friend SEXP Rcpp::wrap<TRFunctionImport>(const TRFunctionImport &f);
         friend TRFunctionImport Rcpp::as<>(SEXP);

      protected:
         Rcpp::Function *f;//Internal Rcpp function to import
         /**
         TRFunctionImport constructor for Rcpp::DataFrame
         \param fun raw function object from Rcpp
         */

         TRFunctionImport(const Rcpp::Function &fun)
         {
            *f = fun;
         }

      public:
         /**
         TRFunctionImport constructor
         \param name name of function from R
         */
         TRFunctionImport(const TString &name);
         /**
         TRFunctionImport constructor
         \param name name of function from R
         \param ns   namespace of function from R
         */
         TRFunctionImport(const TString &name, const TString &ns);
         /**
         TRFunctionImport copy constructor
         \param fun other TRFunctionImport
         */
         TRFunctionImport(const TRFunctionImport &fun);
         /**
         TRFunctionImport constructor
         \param obj raw R object
         */
         TRFunctionImport(SEXP obj);
         /**
         TRFunctionImport  constructor
         \param obj  TRObject object
         */
         TRFunctionImport(TRObject &obj);

         ~TRFunctionImport()
         {
            if (f) delete f;
         }
         SEXP operator()()
         {
            return (*f)();
         }
#include<TRFunctionImport__oprtr.h>
         ClassDef(TRFunctionImport, 0) //
      };

      template<> inline TRObject::operator TRFunctionImport()
      {
         return (SEXP)fObj;
      }
   }
}



#endif
