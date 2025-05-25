// wrapper class for a RooPdf

#ifndef ROOT_WrapperRooPdf
#define ROOT_WrapperRooPdf

#include <RooAbsPdf.h>
#include <RooRealVar.h>
#include <RooArgSet.h>
#include <RooGaussian.h>
#include <TF1.h>

#include <Math/IParamFunction.h>

#include <cassert>

class WrapperRooPdf : public ROOT::Math::IParamMultiFunction {

public:
   /**
      for pdf with only 1D observables using as default the name x
    */
   WrapperRooPdf(RooAbsPdf *pdf, const std::string xvar = "x", bool norm = true) : fNorm(norm), fPdf(pdf)
   {
      assert(fPdf != nullptr);

      std::unique_ptr<RooArgSet> vars{fPdf->getVariables()};
      RooAbsArg *arg = vars->find(xvar.c_str()); // code should abort if not found
      if (!arg)
         std::cout << "Error - observable " << xvar << "is not in the list of pdf variables" << std::endl;
      assert(arg != nullptr);
      RooArgSet obsList(*arg);
      // arg.setDirtyInhibit(true); // do have faster setter of values
      fX = std::unique_ptr<RooArgSet>{fPdf->getObservables(obsList)};
      fParams = std::unique_ptr<RooArgSet>{fPdf->getParameters(obsList)};
      assert(fX != nullptr);
      assert(fParams != nullptr);
#ifdef DEBUG
      fX->Print("v");
      fParams->Print("v");
#endif
   }

   /**
      for pdf with multi-dim  observables specifying observables in the RooArgSet
    */
   WrapperRooPdf(RooAbsPdf *pdf, const RooArgSet &obsList, bool norm = true) : fNorm(norm), fPdf(pdf)
   {
      assert(fPdf != nullptr);

      fX = std::unique_ptr<RooArgSet>{fPdf->getObservables(obsList)};
      fParams = std::unique_ptr<RooArgSet>{fPdf->getParameters(obsList)};
      assert(fX != nullptr);
      assert(fParams != nullptr);
#ifdef DEBUG
      fX->Print("v");
      fParams->Print("v");
#endif
      //       // iterate on fX
      //       for (auto *arg : *fX) {
      //          assert(arg != 0);
      //          arg->setDirtyInhibit(true); // for having faster setter later  in DoEval
      //       }
   }

   /**
      clone the function
    */
#ifndef _WIN32
   WrapperRooPdf
#else
   ROOT::Math::IMultiGenFunction
#endif
      *
      Clone() const override
   {
      // copy the pdf function pointer
      return new WrapperRooPdf(fPdf, *fX, fNorm);
   }

   unsigned int NPar() const override { return fParams->size(); }
   unsigned int NDim() const override { return fX->size(); }
   const double *Parameters() const override
   {
      fParamValues.resize(0);
      // iterate on parameters and set values
      for (auto *var : dynamic_range_cast<RooRealVar *>(*fParams)) {
         assert(var != nullptr);
         fParamValues.push_back(var->getVal());
      }
      return fParamValues.data();
   }

   std::string ParameterName(unsigned int i) const override
   {
      return i < fParams->size() ? (*fParams)[i]->GetName() : "not_found";
   }

   /**
      set parameters. Order of parameter is the one defined by the RooPdf and must be checked by user
    */

   void SetParameters(const double *p) override { DoSetParameters(p); }

   //    double operator() (double *x, double * p = 0)  {
   //       if (p != 0) SetParameters(p);
   //       // iterate on observables
   //       for (auto *var : dynamic_range_cast<RooRealVar *>(*fX)) {
   //          assert(var != 0);
   //          var->setVal(*x++);
   //       }
   //       // debug
   //       //fX->Print("v");

   //       if (fNorm)
   //          return fPdf->getVal(fX);
   //       else
   //          return fPdf->getVal();  // get unnormalized value
   //    }

private:
   double DoEvalPar(const double *x, const double *p) const override
   {

      // should maybe be optimized ???
      DoSetParameters(p);

      // iterate on observables
      for (auto *var : dynamic_range_cast<RooRealVar *>(*fX)) {
         assert(var != nullptr);
#ifndef _WIN32
         var->setDirtyInhibit(true);
#endif
         var->setVal(*x++);
      }

      if (fNorm)
         return fPdf->getVal(fX.get());
      else
         return fPdf->getVal(); // get unnormalized value
   }

   void DoSetParameters(const double *p) const
   {
      // iterate on parameters and set values
      for (auto *var : dynamic_range_cast<RooRealVar *>(*fParams)) {
         assert(var != nullptr);
         var->setVal(*p++);
      }
   }

   bool fNorm;
   mutable RooAbsPdf *fPdf;
   mutable std::unique_ptr<RooArgSet> fX;
   mutable std::unique_ptr<RooArgSet> fParams;
   mutable std::vector<double> fParamValues;
};

#endif
