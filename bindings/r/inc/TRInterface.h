// @(#)root/r:$Id$
// Author: Omar Zapata   29/05/2013


/*************************************************************************
 * Copyright (C) 2013-2014, Omar Andres Zapata Mesa                      *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_R_TRInterface
#define ROOT_R_TRInterface

#ifndef ROOT_R_TRObjectProxy
#include<TRObjectProxy.h>
#endif

#ifndef ROOT_R_TRDataFrame
#include<TRDataFrame.h>
#endif

#ifndef ROOT_R_TFunction
#include<TRFunction.h>
#endif

#ifndef ROOT_TThread
#include<TThread.h>
#endif

/**
   @defgroup R R Interface for Statistical Computing
   \ref ROOTR was implemented using the
   <A HREF="http://www.r-project.org/">R Project</A> library and the modules
   <A HREF="http://cran.r-project.org/web/packages/Rcpp/index.html">Rcpp</A> and
   <A HREF="http://cran.r-project.org/web/packages/RInside/index.html">RInside</A>
   @ingroup R
 */

/**
   @defgroup R R Interface for Statistical Computing
   @ingroup R
 */

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRInterface                                                          //
//                                                                      //
// R Interface class for Statistical Computing.                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

namespace ROOT {
   namespace R {
      class TRInterface: public TObject {
      protected:
         RInside *fR;
         TThread *th;
      public:
         //Proxy class to use operators for assignation Ex: r["name"]=object
         class Binding {
         public:
            Binding(TRInterface *rnt, TString name): fInterface(rnt), fName(name) {}
            Binding &operator=(const Binding &obj) {
               fInterface = obj.fInterface;
               fName = obj.fName;
               return *this;
            }
            template <class T> Binding &operator=(const T &data) {
               fInterface->Assign<T>(data, fName);
               return *this;
            }
            Binding &operator=(const TRFunction &fun) {
               //The method assign is not a template for a function
               fInterface->Assign(fun, fName);
               return *this;
            }

            Binding &operator<<(const TRFunction &fun) {
               //The method assign is not a template for a function
               fInterface->Assign(fun, fName);
               return *this;
            }

            template <class T> Binding &operator >>(T &var) {
               var = fInterface->Eval(fName).As<T>();
               return *this;
            }
            
            template <class T> Binding &operator <<(T var) {
               fInterface->Assign<T>(var, fName);
               return *this;
            }
            #include<TRInterface_Binding.h>
            template <class T> operator T() {
               return fInterface->Eval(fName);
            }

         private:
            TRInterface *fInterface;
            TString fName;
         };
      private:
         TRInterface(const int argc = 0, const char *argv[] = NULL, const bool loadRcpp = true, const bool verbose = false, const bool interactive = true);
      public:
         ~TRInterface();

         void SetVerbose(Bool_t status);
         Int_t Eval(const TString &code, TRObjectProxy  &ans); // parse line, returns in ans; error code rc
         
         void  Execute(const TString &code);

         TRObjectProxy Eval(const TString &code);

         static void LoadModule(TString name);

         //______________________________________________________________________________
         template<typename T >void Assign(const T &var, const TString &name) {
            // This method lets you pass variables from ROOT to R.
            // The template T should be a supported ROOT datatype and
            // the TString's name is the name of the variable in the R enviroment.
            fR->assign<T>(var, name.Data());
         }
         void Assign(const TRFunction &fun, const TString &name);

         void Interactive();
         void ProcessEventsLoop();
         Bool_t IsInstalled(TString pkg);
         Bool_t Require(TString pkg);
         Bool_t Install(TString pkg,TString repos="http://cran.r-project.org");
         Binding operator[](const TString &name);
         static TRInterface &Instance();
         static TRInterface *InstancePtr();

         ClassDef(TRInterface, 0)
      };
   }
}

inline ROOT::R::TRInterface &operator<<(ROOT::R::TRInterface &r, TString code)
{
   r.Execute(code);
   return r;
}

#endif
