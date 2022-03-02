// @(#)root/r:$Id$
// Author: Omar Zapata  Omar.Zapata@cern.ch   29/05/2013

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_R_TRObject
#define ROOT_R_TRObject

#include <RExports.h>


namespace ROOT {
   namespace R {

      /**

      This is a class to get ROOT's objects from R's objects
      <center><h2>TRObject class</h2></center>

      <p>
      The TRObject class lets you obtain ROOT's objects from R's objects.<br>
      It has some basic template operators to convert R's objects into ROOT's datatypes<br>
      </p>
      A simple example<br>
      <p>

      </p>

      \code{.cpp}
      #include<TRInterface.h>
      void Proxy()
      {
      ROOT::R::TRInterface &r=ROOT::R::TRInterface::Instance();
      ROOT::R::TRObject obj;
      obj=r.Eval("seq(1,10)");
      TVectorD v=obj;
      v.Print();
      }
      \endcode
      Output
      \code

      Vector (10)  is as follows

      |        1  |
      ------------------
      0 |1
      1 |2
      2 |3
      3 |4
      4 |5
      5 |6
      6 |7
      7 |8
      8 |9
      9 |10

      \endcode

      <h2>Users Guide </h2>
      <a href="https://oproject.org/pages/ROOT%20R%20Users%20Guide"> https://oproject.org/pages/ROOT R Users Guide</a><br>

      @ingroup R
      */
      class TRObject: public TObject {
         friend SEXP Rcpp::wrap<TRObject>(const TRObject &f);
      private:
         Rcpp::RObject fObj; //internal Rcpp::RObject
         Bool_t fStatus;//status tell if is a valid object
      public:
         /**
              Default constructor
              */
         TRObject(): TObject() {};
         /**
              Construct a TRObject given a R base object
              \param robj raw R object
              */
         TRObject(SEXP robj);
         /**
              Construct a TRObject given a R base object
              \param robj raw R object
              \param status if the raw object is valid obj
              */
         TRObject(SEXP robj, Bool_t status);

         /**
               TRObject is a current valid object?
              \param status if the current object is valid obj
              */
         void SetStatus(Bool_t status)
         {
            fStatus = status;
         }

         /**
               TRObject is a current valid object?
              \return status if the current object
              */
         Bool_t GetStatus()
         {
            return fStatus;
         }

         /**
               The R objects can to have associate attributes
               with this method you can added attribute to TRObject given an object in the template argument.
              \param name attribute name
              \param obj  object associated to the attribute name in the current TRObject
              */
         template<class T> void SetAttribute(const TString name, T obj)
         {
            fObj.attr(name.Data()) = obj;
         }

         /**
               The R objects can to have associate attributes
               with this method you can added attribute to TRObject given an object in the template argument.
              \param name attribute name
              \return object associated to the attribute name in the current TRObject
              */
         TRObject GetAttribute(const TString name)
         {
            return fObj.attr(name.Data());
         }

         void operator=(SEXP xx);

         /**
          Some datatypes of ROOT or c++ can be wrapped in to a TRObject,
          this method lets you wrap those datatypes
              \param obj template object to be wrapped
              \return TRObject reference of wrapped object
              */
         template<class T> TRObject &Wrap(T obj)
         {
            fObj =::Rcpp::wrap(obj);
            return *this;
         }

         /**
          Some datatypes of ROOT or c++ can be wrapped in to a TRObject,
          this method lets you unwrap those datatypes encapsulate into this TRObject.
         \note If the current TRObject is not a valid object it will return and empty object and it will print an error message
              \return template return with the require datatype
              */
         template<class T> T As()
         {
            if (fStatus) {
               T data =::Rcpp::as<T>(fObj);
               return data;
            } else {
               Error("Cast Operator", "Can not make the requested data, returning an unknown value");
               return T();
            }
         }

         template<class T> T operator=(TRObject &obj)
         {
            return ::Rcpp::as<T>(obj);
         }

         operator SEXP()
         {
            return fObj;
         }

         operator SEXP() const
         {
            return fObj;
         }

         operator Rcpp::RObject()
         {
            return fObj;
         }

         template <class T> operator T()
         {

            if (fStatus) {
               T data =::Rcpp::as<T>(fObj);
               return data;
            } else {
               Error("Cast Operator", "Can not make the requested data, returning an unknown value");
               return T();
            }
         }
         ClassDef(TRObject, 0) //
      };

   }
}


#endif
