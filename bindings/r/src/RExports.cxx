// Author: Omar Zapata  Omar.Zapata@cern.ch   2014

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include<RExports.h>
#include<TRFunctionExport.h>
#include<TRObject.h>
#include<TRDataFrame.h>

namespace ROOT {
   namespace R {
     const Rcpp::internal::NamedPlaceHolder &Label(Rcpp::_);
   }
}

namespace Rcpp {
//TVectorT
   template<>  SEXP wrap(const TVectorT<Double_t> &v)
   {
      std::vector<Double_t> vec(v.GetMatrixArray(), v.GetMatrixArray() + v.GetNoElements());
      return wrap(vec);
   }

   template<> TVectorT<Double_t> as(SEXP v)
   {
      std::vector<Double_t> vec =::Rcpp::as<std::vector<Double_t> >(v);
      return TVectorT<Double_t>(vec.size(), vec.data());
   }

   template<> SEXP wrap(const TVectorT<Float_t> &v)
   {
      std::vector<Float_t> vec(v.GetMatrixArray(), v.GetMatrixArray() + v.GetNoElements());
      return wrap(vec);
   }

   template<> TVectorT<Float_t> as(SEXP v)
   {
      std::vector<Float_t> vec =::Rcpp::as<std::vector<Float_t> >(v);
      return TVectorT<Float_t>(vec.size(), vec.data());
   }

//TMatrixT
   template<> SEXP wrap(const TMatrixT<Double_t> &m)
   {
      Int_t rows = m.GetNrows();
      Int_t cols = m.GetNcols();
      Double_t *data = new Double_t[rows * cols];
      m.GetMatrix2Array(data, "F"); //ROOT has a bug here(Fixed)
      NumericMatrix mat(rows, cols, data);
      return wrap(mat);
   }

   template<> TMatrixT<Double_t> as(SEXP m)
   {
      NumericMatrix mat =::Rcpp::as<NumericMatrix>(m);
      return TMatrixT<Double_t>(mat.rows(), mat.cols(), mat.begin(), "F");
   }

   template<> SEXP wrap(const TMatrixT<Float_t> &m)
   {
      Int_t rows = m.GetNrows();
      Int_t cols = m.GetNcols();
      Float_t *data = new Float_t[rows * cols];
      m.GetMatrix2Array(data, "F"); //ROOT has a bug here(Fixed)
      NumericMatrix mat(rows, cols, data);
      return wrap(mat);
   }

   template<> TMatrixT<Float_t> as(SEXP m)
   {
      NumericMatrix mat =::Rcpp::as<NumericMatrix>(m);
      std::vector<Float_t> dat = Rcpp::as<std::vector<Float_t>>(mat);
      return TMatrixT<Float_t>(mat.rows(), mat.cols(), &dat[0], "F");
   }

//TRObject
   template<> SEXP wrap(const ROOT::R::TRObject &obj)
   {
      return obj.fObj;
   }

   template<> ROOT::R::TRObject as(SEXP obj)
   {
      return ROOT::R::TRObject(obj);
   }
//TRDataFrame
   template<> SEXP wrap(const ROOT::R::TRDataFrame &obj)
   {
      return obj.df;
   }

   template<> ROOT::R::TRDataFrame as(SEXP obj)
   {
      return ROOT::R::TRDataFrame(Rcpp::as<Rcpp::DataFrame>(obj));
   }

//TRFunctionImport
   template<> SEXP wrap(const ROOT::R::TRFunctionImport &obj)
   {
      return *obj.f;
   }

   template<> ROOT::R::TRFunctionImport as(SEXP obj)
   {
      return ROOT::R::TRFunctionImport(Rcpp::as<Rcpp::Function>(obj));
   }

}
namespace ROOT {
   namespace R {
      VARIABLE_IS_NOT_USED SEXP ModuleSymRef = NULL;
   }
}
