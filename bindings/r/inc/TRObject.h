// @(#)root/r:$Id$
// Author: Omar Zapata   29/05/2013

/*************************************************************************
 * Copyright (C) 2013-2014, Omar Andres Zapata Mesa                      *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_R_TRObject
#define ROOT_R_TRObject

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
      class TRObject: public TObject {
	friend SEXP Rcpp::wrap<TRObject>(const TRObject &f);
      private:
         Rcpp::RObject fObj;
         Bool_t fStatus;//status tell if is a valid object
         class Attribute
         {
             Rcpp::RObject obj;
             TString fAttName;
         public:
             Attribute(){}
             void SetObject(Rcpp::RObject o){obj=o;}
             
             Attribute& operator[](const char* attname)
             {
                 fAttName=attname;
                 return *this;
             }
             template<class T,class RT> RT operator=(T o)
             {
                 obj.attr(fAttName.Data())=o;
                 return obj;
             }
         };
      public:
          Attribute Attr;
      public:
         TRObject(): TObject() {};
         TRObject(SEXP xx);
         TRObject(SEXP xx, Bool_t status);
	 
	 void SetStatus(Bool_t status){ fStatus = status;}
         
         Bool_t GetStatus() { return fStatus;}
         
                  
         void operator=(SEXP xx);

	 template<class T> TRObject& Wrap(T obj) {
            fObj=::Rcpp::wrap(obj);
	    return *this;
         }
         
         template<class T> T As() {
	   if(fStatus)
	   {
	    T data=::Rcpp::as<T>(fObj);
            return data;
	   }else
	   {
	     Error("Cast Operator", "Can not make the requested data, returning an unknow value");
             return T();
	   }
         }

         template<class T> T operator=(TRObject &obj) {
            return ::Rcpp::as<T>(obj);
         }

         template <class T> operator T() {
	     
	   if(fStatus)
	   {
	     T data=::Rcpp::as<T>(fObj);
             return data;
	   }else
	   {
	     Error("Cast Operator", "Can not make the requested data, returning an unknow value");
             return T();
	   }
         }
         ClassDef(TRObject, 0) //
      };

   }
}


#endif
