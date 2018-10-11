// wrapper class for a RooPdf

#ifndef  ROOT_WrapperRooPdf
#define  ROOT_WrapperRooPdf

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
   WrapperRooPdf(RooAbsPdf * pdf, const std::string xvar = "x", bool norm = true) :
      fNorm(norm),
      fPdf(pdf),
      fX(0),
      fParams(0)
   {
      assert(fPdf != 0);

      RooArgSet *vars = fPdf->getVariables();
      RooAbsArg * arg = vars->find(xvar.c_str());  // code should abort if not found
      if (!arg) std::cout <<"Error - observable " << xvar << "is not in the list of pdf variables" << std::endl;
      assert(arg != 0);
      RooArgSet obsList(*arg);
      //arg.setDirtyInhibit(true); // do have faster setter of values
      fX = fPdf->getObservables(obsList);
      fParams = fPdf->getParameters(obsList);
      assert(fX!=0);
      assert(fParams!=0);
      delete vars;
#ifdef DEBUG
      fX->Print("v");
      fParams->Print("v");
#endif
   }




   /**
      for pdf with multi-dim  observables specifying observables in the RooArgSet
    */
   WrapperRooPdf(RooAbsPdf * pdf, const RooArgSet & obsList, bool norm = true ) :
      fNorm(norm),
      fPdf(pdf),
      fX(0),
      fParams(0)
   {
      assert(fPdf != 0);

      fX = fPdf->getObservables(obsList);
      fParams = fPdf->getParameters(obsList);
      assert(fX!=0);
      assert(fParams!=0);
#ifdef DEBUG
      fX->Print("v");
      fParams->Print("v");
#endif
//       // iterate on fX
//       TIterator* itr = fX->createIterator() ;
//       RooAbsArg* arg = 0;
//       while( ( arg = dynamic_cast<RooAbsArg*>(itr->Next() ) ) ) {
//          assert(arg != 0);
//          arg->setDirtyInhibit(true); // for having faster setter later  in DoEval
//       }

   }


   ~WrapperRooPdf() {
      // need to delete observables and parameter list
      if (fX) delete fX;
      if (fParams) delete fParams;
   }

   /**
      clone the function
    */
#ifndef _WIN32
   WrapperRooPdf
#else
     ROOT::Math::IMultiGenFunction
#endif
     * Clone() const {
      // copy the pdf function pointer
      return new WrapperRooPdf(fPdf, *fX, fNorm);
   }

   unsigned int NPar() const {
      return fParams->getSize();
   }
   unsigned int NDim() const {
      return fX->getSize();
   }
   const double * Parameters() const {
      if (fParamValues.size() != NPar() )
         fParamValues.resize(NPar() );

      // iterate on parameters and set values
      TIterator* itr = fParams->createIterator() ;
      std::vector<double>::iterator vpitr = fParamValues.begin();

      RooRealVar* var = 0;
      while( ( var = dynamic_cast<RooRealVar*>(itr->Next() ) ) ) {
         assert(var != 0);
         *vpitr++ = var->getVal();
      }
      return &fParamValues.front();
   }

   std::string ParameterName(unsigned int i) const {
      // iterate on parameters and set values
      TIterator* itr = fParams->createIterator() ;
      RooRealVar* var = 0;
      unsigned int index = 0;
      while( ( var = dynamic_cast<RooRealVar*>(itr->Next() ) ) ) {
         assert(var != 0);
         if (index == i) return std::string(var->GetName() );
         index++;
      }
      return "not_found";
   }


   /**
      set parameters. Order of parameter is the one defined by the RooPdf and must be checked by user
    */

   void SetParameters(const double * p) {
      DoSetParameters(p);
   }

//    double operator() (double *x, double * p = 0)  {
//       if (p != 0) SetParameters(p);
//       // iterate on observables
//       TIterator* itr = fX->createIterator() ;
//       RooRealVar* var = 0;
//       while( ( var = dynamic_cast<RooRealVar*>(itr->Next() ) ) ) {
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

   double DoEvalPar(const double * x, const double * p) const {

      // should maybe be optimized ???
      DoSetParameters(p);

      // iterate on observables
      TIterator* itr = fX->createIterator() ;
      RooRealVar* var = 0;
      while( ( var = dynamic_cast<RooRealVar*>(itr->Next() ) ) ) {
         assert(var != 0);
#ifndef _WIN32
         var->setDirtyInhibit(true);
#endif
         var->setVal(*x++);
      }
      // debug
      //fX->Print("v");

      if (fNorm)
         return fPdf->getVal(fX);
      else
         return fPdf->getVal();  // get unnormalized value

   }


   void DoSetParameters(const double * p) const {
      // iterate on parameters and set values
      TIterator* itr = fParams->createIterator() ;
      RooRealVar* var = 0;
      while( ( var = dynamic_cast<RooRealVar*>(itr->Next() ) ) ) {
         assert(var != 0);
         var->setVal(*p++);
      }
      // debug
      //fParams->Print("v");
   }


   bool fNorm;
   mutable RooAbsPdf * fPdf;
   mutable RooArgSet * fX;
   mutable RooArgSet * fParams;
   mutable std::vector<double> fParamValues;


};



#endif
