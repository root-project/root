// @(#)root/matrix:$Name:  $:$Id: TMatrixFLazy.h,v 1.1 2004/01/25 20:33:32 brun Exp $
// Authors: Fons Rademakers, Eddy Offermann   Nov 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMatrixFLazy
#define ROOT_TMatrixFLazy

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Lazy Matrix classes.                                                 //
//                                                                      //
//   TMatrixFLazy                                                       //
//   TMatrixFSymLazy                                                    //
//   THilbertMatrixF                                                    //
//   THaarMatrixF                                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMatrixFBase
#include "TMatrixFBase.h"
#endif

class TVectorF;
class TMatrixFBase;
class TMatrixF;
class TMatrixFSym;

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixFLazy                                                         //
//                                                                      //
// Class used to make a lazy copy of a matrix, i.e. only copy matrix    //
// when really needed (when accessed).                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TMatrixFLazy : public TObject {

friend class TMatrixFBase;
friend class TMatrixF;
friend class TVectorF;

protected:
  Int_t fRowUpb;
  Int_t fRowLwb;
  Int_t fColUpb;
  Int_t fColLwb;

private:
  virtual void FillIn(TMatrixF &m) const = 0;

  TMatrixFLazy(const TMatrixFLazy &) : TObject() { }
  void operator=(const TMatrixFLazy &) { }

public:
  TMatrixFLazy() { fRowUpb = fRowLwb = fColUpb = fColLwb = 0; }
  TMatrixFLazy(Int_t nrows, Int_t ncols)
     : fRowUpb(nrows-1),fRowLwb(0),fColUpb(ncols-1),fColLwb(0) { }
  TMatrixFLazy(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb)
     : fRowUpb(row_upb),fRowLwb(row_lwb),fColUpb(col_upb),fColLwb(col_lwb) { }
  virtual ~TMatrixFLazy() {}

  inline Int_t GetRowLwb() const { return fRowLwb; }
  inline Int_t GetRowUpb() const { return fRowUpb; }
  inline Int_t GetColLwb() const { return fColLwb; }
  inline Int_t GetColUpb() const { return fColUpb; }

  ClassDef(TMatrixFLazy,2)  // Lazy matrix with double precision
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixFSymLazy                                                      //
//                                                                      //
// Class used to make a lazy copy of a matrix, i.e. only copy matrix    //
// when really needed (when accessed).                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TMatrixFSymLazy : public TObject {

friend class TMatrixFBase;
friend class TMatrixFSym;
friend class TVectorF;

protected:
  Int_t fRowUpb;
  Int_t fRowLwb;

private:
  virtual void FillIn(TMatrixFSym &m) const = 0;

  TMatrixFSymLazy(const TMatrixFSymLazy &) : TObject() { }
  void operator=(const TMatrixFSymLazy &) { }

public:
  TMatrixFSymLazy() { fRowUpb = fRowLwb = 0; }
  TMatrixFSymLazy(Int_t nrows)
     : fRowUpb(nrows-1),fRowLwb(0) { }
  TMatrixFSymLazy(Int_t row_lwb,Int_t row_upb)
     : fRowUpb(row_upb),fRowLwb(row_lwb) { }
  virtual ~TMatrixFSymLazy() {}

  inline Int_t GetRowLwb() const { return fRowLwb; }
  inline Int_t GetRowUpb() const { return fRowUpb; }

  ClassDef(TMatrixFSymLazy,1)  // Lazy symmeytric matrix with double precision
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THaarMatrixF                                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class THaarMatrixF : public TMatrixFLazy {

private:
  void FillIn(TMatrixF &m) const;

public:
  THaarMatrixF() {}
  THaarMatrixF(Int_t n,Int_t no_cols = 0);
  virtual ~THaarMatrixF() {}

  ClassDef(THaarMatrixF,1)  // Haar matrix with double precision
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THilbertMatrixF                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class THilbertMatrixF : public TMatrixFLazy {

private:
  void FillIn(TMatrixF &m) const;

public:
  THilbertMatrixF() {}
  THilbertMatrixF(Int_t no_rows,Int_t no_cols);
  THilbertMatrixF(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb);
  virtual ~THilbertMatrixF() {}

  ClassDef(THilbertMatrixF,1)  // (no_rows x no_cols) Hilbert matrix with double precision
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THilbertMatrixFSym                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class THilbertMatrixFSym : public TMatrixFSymLazy {

private:
  void FillIn(TMatrixFSym &m) const;

public:
  THilbertMatrixFSym() {}
  THilbertMatrixFSym(Int_t no_rows);
  THilbertMatrixFSym(Int_t row_lwb,Int_t row_upb);
  virtual ~THilbertMatrixFSym() {}
  
  ClassDef(THilbertMatrixFSym,1)  // (no_rows x no_rows) Hilbert matrix with double precision
};

#endif
