// @(#)root/matrix:$Name:  $:$Id: TMatrixDLazy.h,v 1.17 2003/08/18 16:40:33 rdm Exp $
// Authors: Fons Rademakers, Eddy Offermann   Nov 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMatrixDLazy
#define ROOT_TMatrixDLazy

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Lazy Matrix classes.                                                 //
//                                                                      //
//   TMatrixDLazy                                                       //
//   TMatrixDSymLazy                                                    //
//   THilbertMatrixD                                                    //
//   THaarMatrixD                                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMatrixDBase
#include "TMatrixDBase.h"
#endif

class TVectorD;
class TMatrixDBase;
class TMatrixD;
class TMatrixDSym;

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixDLazy                                                         //
//                                                                      //
// Class used to make a lazy copy of a matrix, i.e. only copy matrix    //
// when really needed (when accessed).                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TMatrixDLazy : public TObject {

friend class TMatrixDBase;
friend class TMatrixD;
friend class TVectorD;

protected:
  Int_t fRowUpb;
  Int_t fRowLwb;
  Int_t fColUpb;
  Int_t fColLwb;

private:
  virtual void FillIn(TMatrixD &m) const = 0;

  TMatrixDLazy(const TMatrixDLazy &) : TObject() { }
  void operator=(const TMatrixDLazy &) { }

public:
  TMatrixDLazy() { fRowUpb = fRowLwb = fColUpb = fColLwb = 0; }
  TMatrixDLazy(Int_t nrows, Int_t ncols)
     : fRowUpb(nrows-1),fRowLwb(0),fColUpb(ncols-1),fColLwb(0) { }
  TMatrixDLazy(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb)
     : fRowUpb(row_upb),fRowLwb(row_lwb),fColUpb(col_upb),fColLwb(col_lwb) { }
  virtual ~TMatrixDLazy() {}

  inline Int_t GetRowLwb() const { return fRowLwb; }
  inline Int_t GetRowUpb() const { return fRowUpb; }
  inline Int_t GetColLwb() const { return fColLwb; }
  inline Int_t GetColUpb() const { return fColUpb; }

  ClassDef(TMatrixDLazy,2)  // Lazy matrix with double precision
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixDSymLazy                                                      //
//                                                                      //
// Class used to make a lazy copy of a matrix, i.e. only copy matrix    //
// when really needed (when accessed).                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TMatrixDSymLazy : public TObject {

friend class TMatrixDBase;
friend class TMatrixDSym;
friend class TVectorD;

protected:
  Int_t fRowUpb;
  Int_t fRowLwb;

private:
  virtual void FillIn(TMatrixDSym &m) const = 0;

  TMatrixDSymLazy(const TMatrixDSymLazy &) : TObject() { }
  void operator=(const TMatrixDSymLazy &) { }

public:
  TMatrixDSymLazy() { fRowUpb = fRowLwb = 0; }
  TMatrixDSymLazy(Int_t nrows)
     : fRowUpb(nrows-1),fRowLwb(0) { }
  TMatrixDSymLazy(Int_t row_lwb,Int_t row_upb)
     : fRowUpb(row_upb),fRowLwb(row_lwb) { }
  virtual ~TMatrixDSymLazy() {}

  inline Int_t GetRowLwb() const { return fRowLwb; }
  inline Int_t GetRowUpb() const { return fRowUpb; }

  ClassDef(TMatrixDSymLazy,1)  // Lazy symmeytric matrix with double precision
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THaarMatrixD                                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class THaarMatrixD : public TMatrixDLazy {

private:
  void FillIn(TMatrixD &m) const;

public:
  THaarMatrixD() {}
  THaarMatrixD(Int_t n,Int_t no_cols = 0);
  virtual ~THaarMatrixD() {}

  ClassDef(THaarMatrixD,1)  // Haar matrix with double precision
};

void MakeHaarMat(TMatrixD &m);

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THilbertMatrixD                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class THilbertMatrixD : public TMatrixDLazy {

private:
  void FillIn(TMatrixD &m) const;

public:
  THilbertMatrixD() {}
  THilbertMatrixD(Int_t no_rows,Int_t no_cols);
  THilbertMatrixD(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb);
  virtual ~THilbertMatrixD() {}

  ClassDef(THilbertMatrixD,1)  // (no_rows x no_cols) Hilbert matrix with double precision
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THilbertMatrixDSym                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class THilbertMatrixDSym : public TMatrixDSymLazy {

private:
  void FillIn(TMatrixDSym &m) const;

public:
  THilbertMatrixDSym() {}
  THilbertMatrixDSym(Int_t no_rows);
  THilbertMatrixDSym(Int_t row_lwb,Int_t row_upb);
  virtual ~THilbertMatrixDSym() {}
  
  ClassDef(THilbertMatrixDSym,1)  // (no_rows x no_rows) Hilbert matrix with double precision
};

void MakeHilbertMat(TMatrixD &m);
void MakeHilbertMat(TMatrixDSym  &m);

#endif
