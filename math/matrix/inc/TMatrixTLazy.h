// @(#)root/matrix:$Id$
// Authors: Fons Rademakers, Eddy Offermann   Nov 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMatrixTLazy
#define ROOT_TMatrixTLazy

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Templates of Lazy Matrix classes.                                    //
//                                                                      //
//   TMatrixTLazy                                                       //
//   TMatrixTSymLazy                                                    //
//   THaarMatrixT                                                       //
//   THilbertMatrixT                                                    //
//   THilbertMatrixTSym                                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMatrixTBase
#include "TMatrixTBase.h"
#endif

template<class Element> class TVectorT;
template<class Element> class TMatrixTBase;
template<class Element> class TMatrixT;
template<class Element> class TMatrixTSym;

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixTLazy                                                         //
//                                                                      //
// Class used to make a lazy copy of a matrix, i.e. only copy matrix    //
// when really needed (when accessed).                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

template<class Element> class TMatrixTLazy : public TObject {

friend class TMatrixTBase<Element>;
friend class TMatrixT    <Element>;
friend class TVectorT    <Element>;

protected:
   Int_t fRowUpb;
   Int_t fRowLwb;
   Int_t fColUpb;
   Int_t fColLwb;

   TMatrixTLazy(const TMatrixTLazy<Element> &) : TObject(), fRowUpb(0),fRowLwb(0),fColUpb(0),fColLwb(0) { }
   void operator=(const TMatrixTLazy<Element> &) { }

private:
   virtual void FillIn(TMatrixT<Element> &m) const = 0;

public:
   TMatrixTLazy() { fRowUpb = fRowLwb = fColUpb = fColLwb = 0; }
   TMatrixTLazy(Int_t nrows, Int_t ncols)
       : fRowUpb(nrows-1),fRowLwb(0),fColUpb(ncols-1),fColLwb(0) { }
   TMatrixTLazy(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb)
       : fRowUpb(row_upb),fRowLwb(row_lwb),fColUpb(col_upb),fColLwb(col_lwb) { }
   virtual ~TMatrixTLazy() {}

   inline Int_t GetRowLwb() const { return fRowLwb; }
   inline Int_t GetRowUpb() const { return fRowUpb; }
   inline Int_t GetColLwb() const { return fColLwb; }
   inline Int_t GetColUpb() const { return fColUpb; }

   ClassDef(TMatrixTLazy,3)  // Template of Lazy Matrix class
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixTSymLazy                                                      //
//                                                                      //
// Class used to make a lazy copy of a matrix, i.e. only copy matrix    //
// when really needed (when accessed).                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

template<class Element> class TMatrixTSymLazy : public TObject {

friend class TMatrixTBase<Element>;
friend class TMatrixTSym <Element>;
friend class TVectorT    <Element>;

protected:
   Int_t fRowUpb;
   Int_t fRowLwb;

   TMatrixTSymLazy(const TMatrixTSymLazy<Element> &) : TObject(), fRowUpb(0),fRowLwb(0)  { }
   void operator=(const TMatrixTSymLazy<Element> &) { }

private:
   virtual void FillIn(TMatrixTSym<Element> &m) const = 0;

public:
   TMatrixTSymLazy() { fRowUpb = fRowLwb = 0; }
   TMatrixTSymLazy(Int_t nrows)
       : fRowUpb(nrows-1),fRowLwb(0) { }
   TMatrixTSymLazy(Int_t row_lwb,Int_t row_upb)
       : fRowUpb(row_upb),fRowLwb(row_lwb) { }
   virtual ~TMatrixTSymLazy() {}

   inline Int_t GetRowLwb() const { return fRowLwb; }
   inline Int_t GetRowUpb() const { return fRowUpb; }

   ClassDef(TMatrixTSymLazy,2)  // Template of Lazy Symmeytric class
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THaarMatrixT                                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

template<class Element> class THaarMatrixT: public TMatrixTLazy<Element> {

private:
   void FillIn(TMatrixT<Element> &m) const;

public:
   THaarMatrixT() {}
   THaarMatrixT(Int_t n,Int_t no_cols = 0);
   virtual ~THaarMatrixT() {}

   ClassDef(THaarMatrixT,2)  // Template of Haar Matrix class
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THilbertMatrixT                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

template<class Element> class THilbertMatrixT : public TMatrixTLazy<Element> {

private:
   void FillIn(TMatrixT<Element> &m) const;

public:
   THilbertMatrixT() {}
   THilbertMatrixT(Int_t no_rows,Int_t no_cols);
   THilbertMatrixT(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb);
   virtual ~THilbertMatrixT() {}

   ClassDef(THilbertMatrixT,2)  // Template of Hilbert Matrix class
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THilbertMatrixTSym                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

template<class Element> class THilbertMatrixTSym : public TMatrixTSymLazy<Element> {

private:
   void FillIn(TMatrixTSym<Element> &m) const;

public:
   THilbertMatrixTSym() {}
   THilbertMatrixTSym(Int_t no_rows);
   THilbertMatrixTSym(Int_t row_lwb,Int_t row_upb);
   virtual ~THilbertMatrixTSym() {}

   ClassDef(THilbertMatrixTSym,2)  // Template of Symmetric Hilbert Matrix class
};

#endif
