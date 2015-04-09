// @(#)root/r:$Id$
// Author: Omar Zapata   29/05/2013

/*************************************************************************
 * Copyright (C) 2013-2014, Omar Andres Zapata Mesa                      *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_R_TRObjectProxy
#define ROOT_R_TRObjectProxy

#ifndef ROOT_R_RExports
#include<RExports.h>
#endif

//________________________________________________________________________________________________________
/**
   This is a class to get ROOT's objects from R's objects


   @ingroup R
*/

namespace ROOT {
   namespace R {
      class TRObjectProxy: public TObject {
	friend SEXP Rcpp::wrap<TRObjectProxy>(const TRObjectProxy &f);
      private:
         Rcpp::RObject x;
         Bool_t fStatus;//status tell if is a valid object
      public:
         TRObjectProxy(): TObject() {};
         TRObjectProxy(SEXP xx);
         TRObjectProxy(SEXP xx, Bool_t status);
	 
	 void SetStatus(Bool_t status){ fStatus = status;}
         
         Bool_t GetStatus() { return fStatus;}

         void operator=(SEXP xx);

	 template<class T> TRObjectProxy& Wrap(T obj) {
            x=::Rcpp::wrap(obj);
	    return *this;
         }
         
         template<class T> T As() {
	   if(fStatus)
	   {
	    T data=::Rcpp::as<T>(x);
            return data;
	   }else
	   {
	     Error("Cast Operator", "Can not make the requested data, returning an unknow value");
             return T();
	   }
         }

         template<class T> T operator=(TRObjectProxy &obj) {
            return ::Rcpp::as<T>(obj);
         }

         template <class T> operator T() {
	     
	   if(fStatus)
	   {
	     T data=::Rcpp::as<T>(x);
             return data;
	   }else
	   {
	     Error("Cast Operator", "Can not make the requested data, returning an unknow value");
             return T();
	   }
         }
         ClassDef(TRObjectProxy, 0) //
      };

   }
}


#endif
