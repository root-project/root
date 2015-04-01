// @(#)root/matrix:$Id$
// Authors: Fons Rademakers, Eddy Offermann   Nov 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMatrixT
#define ROOT_TMatrixT

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixT                                                             //
//                                                                      //
// Template class of a general matrix in the linear algebra package     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMatrixTBase
#include "TMatrixTBase.h"
#endif
#ifndef ROOT_TMatrixTUtils
#include "TMatrixTUtils.h"
#endif

#ifdef CBLAS
#include <vecLib/vBLAS.h>
//#include <cblas.h>
#endif


template<class Element> class TMatrixTSym;
template<class Element> class TMatrixTSparse;
template<class Element> class TMatrixTLazy;

template<class Element> class TMatrixT : public TMatrixTBase<Element> {

protected:

   Element  fDataStack[TMatrixTBase<Element>::kSizeMax]; //! data container
   Element *fElements;                                   //[fNelems] elements themselves

   Element *New_m   (Int_t size);
   void     Delete_m(Int_t size,Element*&);
   Int_t    Memcpy_m(Element *newp,const Element *oldp,Int_t copySize,
                      Int_t newSize,Int_t oldSize);
   void     Allocate(Int_t nrows,Int_t ncols,Int_t row_lwb = 0,Int_t col_lwb = 0,Int_t init = 0,
                     Int_t /*nr_nonzeros*/ = -1);


public:


   enum {kWorkMax = 100};
   enum EMatrixCreatorsOp1 { kZero,kUnit,kTransposed,kInverted,kAtA };
   enum EMatrixCreatorsOp2 { kMult,kTransposeMult,kInvMult,kMultTranspose,kPlus,kMinus };

   TMatrixT(): fDataStack(), fElements(0) { }
   TMatrixT(Int_t nrows,Int_t ncols);
   TMatrixT(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb);
   TMatrixT(Int_t nrows,Int_t ncols,const Element *data,Option_t *option="");
   TMatrixT(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,const Element *data,Option_t *option="");
   TMatrixT(const TMatrixT      <Element> &another);
   TMatrixT(const TMatrixTSym   <Element> &another);
   TMatrixT(const TMatrixTSparse<Element> &another);
   template <class Element2> TMatrixT(const TMatrixT<Element2> &another): fElements(0)
   {
      R__ASSERT(another.IsValid());
      Allocate(another.GetNrows(),another.GetNcols(),another.GetRowLwb(),another.GetColLwb());
      *this = another;
   }

   TMatrixT(EMatrixCreatorsOp1 op,const TMatrixT<Element> &prototype);
   TMatrixT(const TMatrixT    <Element> &a,EMatrixCreatorsOp2 op,const TMatrixT   <Element> &b);
   TMatrixT(const TMatrixT    <Element> &a,EMatrixCreatorsOp2 op,const TMatrixTSym<Element> &b);
   TMatrixT(const TMatrixTSym <Element> &a,EMatrixCreatorsOp2 op,const TMatrixT   <Element> &b);
   TMatrixT(const TMatrixTSym <Element> &a,EMatrixCreatorsOp2 op,const TMatrixTSym<Element> &b);
   TMatrixT(const TMatrixTLazy<Element> &lazy_constructor);

   virtual ~TMatrixT() { Clear(); }

   // Elementary constructors

   void Plus (const TMatrixT   <Element> &a,const TMatrixT   <Element> &b);
   void Plus (const TMatrixT   <Element> &a,const TMatrixTSym<Element> &b);
   void Plus (const TMatrixTSym<Element> &a,const TMatrixT   <Element> &b) { Plus(b,a); }

   void Minus(const TMatrixT   <Element> &a,const TMatrixT   <Element> &b);
   void Minus(const TMatrixT   <Element> &a,const TMatrixTSym<Element> &b);
   void Minus(const TMatrixTSym<Element> &a,const TMatrixT   <Element> &b) { Minus(b,a); }

   void Mult (const TMatrixT   <Element> &a,const TMatrixT   <Element> &b);
   void Mult (const TMatrixT   <Element> &a,const TMatrixTSym<Element> &b);
   void Mult (const TMatrixTSym<Element> &a,const TMatrixT   <Element> &b);
   void Mult (const TMatrixTSym<Element> &a,const TMatrixTSym<Element> &b);

   void TMult(const TMatrixT   <Element> &a,const TMatrixT   <Element> &b);
   void TMult(const TMatrixT   <Element> &a,const TMatrixTSym<Element> &b);
   void TMult(const TMatrixTSym<Element> &a,const TMatrixT   <Element> &b) { Mult(a,b); }
   void TMult(const TMatrixTSym<Element> &a,const TMatrixTSym<Element> &b) { Mult(a,b); }

   void MultT(const TMatrixT   <Element> &a,const TMatrixT   <Element> &b);
   void MultT(const TMatrixT   <Element> &a,const TMatrixTSym<Element> &b) { Mult(a,b); }
   void MultT(const TMatrixTSym<Element> &a,const TMatrixT   <Element> &b);
   void MultT(const TMatrixTSym<Element> &a,const TMatrixTSym<Element> &b) { Mult(a,b); }

   virtual const Element *GetMatrixArray  () const;
   virtual       Element *GetMatrixArray  ();
   virtual const Int_t   *GetRowIndexArray() const { return 0; }
   virtual       Int_t   *GetRowIndexArray()       { return 0; }
   virtual const Int_t   *GetColIndexArray() const { return 0; }
   virtual       Int_t   *GetColIndexArray()       { return 0; }

   virtual       TMatrixTBase<Element> &SetRowIndexArray(Int_t * /*data*/) { MayNotUse("SetRowIndexArray(Int_t *)"); return *this; }
   virtual       TMatrixTBase<Element> &SetColIndexArray(Int_t * /*data*/) { MayNotUse("SetColIndexArray(Int_t *)"); return *this; }

   virtual void Clear(Option_t * /*option*/ ="") { if (this->fIsOwner) Delete_m(this->fNelems,fElements);
                                                   else fElements = 0;  this->fNelems = 0; }

           TMatrixT    <Element> &Use     (Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,Element *data);
   const   TMatrixT    <Element> &Use     (Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,const Element *data) const
                                            { return (const TMatrixT<Element>&)
                                                     ((const_cast<TMatrixT<Element> *>(this))->Use(row_lwb,row_upb,col_lwb,col_upb, const_cast<Element *>(data))); }
           TMatrixT    <Element> &Use     (Int_t nrows,Int_t ncols,Element *data);
   const   TMatrixT    <Element> &Use     (Int_t nrows,Int_t ncols,const Element *data) const;
           TMatrixT    <Element> &Use     (TMatrixT<Element> &a);
   const   TMatrixT    <Element> &Use     (const TMatrixT<Element> &a) const;

   virtual TMatrixTBase<Element> &GetSub  (Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,
                                           TMatrixTBase<Element> &target,Option_t *option="S") const;
           TMatrixT    <Element>  GetSub  (Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,Option_t *option="S") const;
   virtual TMatrixTBase<Element> &SetSub  (Int_t row_lwb,Int_t col_lwb,const TMatrixTBase<Element> &source);

   virtual TMatrixTBase<Element> &ResizeTo(Int_t nrows,Int_t ncols,Int_t /*nr_nonzeros*/ =-1);
   virtual TMatrixTBase<Element> &ResizeTo(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,Int_t /*nr_nonzeros*/ =-1);
   inline  TMatrixTBase<Element> &ResizeTo(const TMatrixT<Element> &m) {
                                            return ResizeTo(m.GetRowLwb(),m.GetRowUpb(),m.GetColLwb(),m.GetColUpb());
                                 }

   virtual Double_t Determinant  () const;
   virtual void     Determinant  (Double_t &d1,Double_t &d2) const;

           TMatrixT<Element> &Invert      (Double_t *det=0);
           TMatrixT<Element> &InvertFast  (Double_t *det=0);
           TMatrixT<Element> &Transpose   (const TMatrixT<Element> &source);
   inline  TMatrixT<Element> &T           () { return this->Transpose(*this); }
           TMatrixT<Element> &Rank1Update (const TVectorT<Element> &v,Element alpha=1.0);
           TMatrixT<Element> &Rank1Update (const TVectorT<Element> &v1,const TVectorT<Element> &v2,Element alpha=1.0);
           Element            Similarity  (const TVectorT<Element> &v) const;

   TMatrixT<Element> &NormByColumn(const TVectorT<Element> &v,Option_t *option="D");
   TMatrixT<Element> &NormByRow   (const TVectorT<Element> &v,Option_t *option="D");

   // Either access a_ij as a(i,j)
   inline       Element                     operator()(Int_t rown,Int_t coln) const;
   inline       Element                    &operator()(Int_t rown,Int_t coln);

   // or as a[i][j]
   inline const TMatrixTRow_const<Element>  operator[](Int_t rown) const { return TMatrixTRow_const<Element>(*this,rown); }
   inline       TMatrixTRow      <Element>  operator[](Int_t rown)       { return TMatrixTRow      <Element>(*this,rown); }

   TMatrixT<Element> &operator= (const TMatrixT      <Element> &source);
   TMatrixT<Element> &operator= (const TMatrixTSym   <Element> &source);
   TMatrixT<Element> &operator= (const TMatrixTSparse<Element> &source);
   TMatrixT<Element> &operator= (const TMatrixTLazy  <Element> &source);
   template <class Element2> TMatrixT<Element> &operator= (const TMatrixT<Element2> &source)
   {
      if (!AreCompatible(*this,source)) {
         Error("operator=(const TMatrixT2 &)","matrices not compatible");
         return *this;
      }

     TObject::operator=(source);
     const Element2 * const ps = source.GetMatrixArray();
           Element  * const pt = this->GetMatrixArray();
     for (Int_t i = 0; i < this->fNelems; i++)
        pt[i] = ps[i];
     this->fTol = source.GetTol();
     return *this;
   }

   TMatrixT<Element> &operator= (Element val);
   TMatrixT<Element> &operator-=(Element val);
   TMatrixT<Element> &operator+=(Element val);
   TMatrixT<Element> &operator*=(Element val);

   TMatrixT<Element> &operator+=(const TMatrixT   <Element> &source);
   TMatrixT<Element> &operator+=(const TMatrixTSym<Element> &source);
   TMatrixT<Element> &operator-=(const TMatrixT   <Element> &source);
   TMatrixT<Element> &operator-=(const TMatrixTSym<Element> &source);

   TMatrixT<Element> &operator*=(const TMatrixT            <Element> &source);
   TMatrixT<Element> &operator*=(const TMatrixTSym         <Element> &source);
   TMatrixT<Element> &operator*=(const TMatrixTDiag_const  <Element> &diag);
   TMatrixT<Element> &operator/=(const TMatrixTDiag_const  <Element> &diag);
   TMatrixT<Element> &operator*=(const TMatrixTRow_const   <Element> &row);
   TMatrixT<Element> &operator/=(const TMatrixTRow_const   <Element> &row);
   TMatrixT<Element> &operator*=(const TMatrixTColumn_const<Element> &col);
   TMatrixT<Element> &operator/=(const TMatrixTColumn_const<Element> &col);

   const TMatrixT<Element> EigenVectors(TVectorT<Element> &eigenValues) const;

   ClassDef(TMatrixT,4) // Template of General Matrix class
};

template <class Element> inline const Element           *TMatrixT<Element>::GetMatrixArray() const { return fElements; }
template <class Element> inline       Element           *TMatrixT<Element>::GetMatrixArray()       { return fElements; }

template <class Element> inline       TMatrixT<Element> &TMatrixT<Element>::Use           (Int_t nrows,Int_t ncols,Element *data)
                                                                                          { return Use(0,nrows-1,0,ncols-1,data); }
template <class Element> inline const TMatrixT<Element> &TMatrixT<Element>::Use           (Int_t nrows,Int_t ncols,const Element *data) const
                                                                                          { return Use(0,nrows-1,0,ncols-1,data); }
template <class Element> inline       TMatrixT<Element> &TMatrixT<Element>::Use           (TMatrixT &a)
                                                                                          {
                                                                                            R__ASSERT(a.IsValid());
                                                                                            return Use(a.GetRowLwb(),a.GetRowUpb(),
                                                                                                       a.GetColLwb(),a.GetColUpb(),a.GetMatrixArray());
                                                                                          }
template <class Element> inline const TMatrixT<Element> &TMatrixT<Element>::Use           (const TMatrixT &a) const
                                                                                          {
                                                                                            R__ASSERT(a.IsValid());
                                                                                            return Use(a.GetRowLwb(),a.GetRowUpb(),
                                                                                                       a.GetColLwb(),a.GetColUpb(),a.GetMatrixArray());
                                                                                          }

template <class Element> inline       TMatrixT<Element>  TMatrixT<Element>::GetSub        (Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,
                                                                                           Option_t *option) const
                                                                                          {
                                                                                            TMatrixT tmp;
                                                                                            this->GetSub(row_lwb,row_upb,col_lwb,col_upb,tmp,option);
                                                                                            return tmp;
                                                                                          }

template <class Element> inline Element TMatrixT<Element>::operator()(Int_t rown,Int_t coln) const
{
   R__ASSERT(this->IsValid());
   const Int_t arown = rown-this->fRowLwb;
   const Int_t acoln = coln-this->fColLwb;
   if (arown >= this->fNrows || arown < 0) {
      Error("operator()","Request row(%d) outside matrix range of %d - %d",rown,this->fRowLwb,this->fRowLwb+this->fNrows);
      return TMatrixTBase<Element>::NaNValue();
   }
   if (acoln >= this->fNcols || acoln < 0) {
      Error("operator()","Request column(%d) outside matrix range of %d - %d",coln,this->fColLwb,this->fColLwb+this->fNcols);
      return TMatrixTBase<Element>::NaNValue();

   }
   return (fElements[arown*this->fNcols+acoln]);
}

template <class Element> inline Element &TMatrixT<Element>::operator()(Int_t rown,Int_t coln)
{
   R__ASSERT(this->IsValid());
   const Int_t arown = rown-this->fRowLwb;
   const Int_t acoln = coln-this->fColLwb;
   if (arown >= this->fNrows || arown < 0) {
      Error("operator()","Request row(%d) outside matrix range of %d - %d",rown,this->fRowLwb,this->fRowLwb+this->fNrows);
      return TMatrixTBase<Element>::NaNValue();
   }
   if (acoln >= this->fNcols || acoln < 0) {
      Error("operator()","Request column(%d) outside matrix range of %d - %d",coln,this->fColLwb,this->fColLwb+this->fNcols);
      return TMatrixTBase<Element>::NaNValue();
   }
   return (fElements[arown*this->fNcols+acoln]);
}

template <class Element> TMatrixT<Element>  operator+  (const TMatrixT   <Element> &source1,const TMatrixT   <Element> &source2);
template <class Element> TMatrixT<Element>  operator+  (const TMatrixT   <Element> &source1,const TMatrixTSym<Element> &source2);
template <class Element> TMatrixT<Element>  operator+  (const TMatrixTSym<Element> &source1,const TMatrixT   <Element> &source2);
template <class Element> TMatrixT<Element>  operator+  (const TMatrixT   <Element> &source ,      Element               val    );
template <class Element> TMatrixT<Element>  operator+  (      Element               val    ,const TMatrixT   <Element> &source );
template <class Element> TMatrixT<Element>  operator-  (const TMatrixT   <Element> &source1,const TMatrixT   <Element> &source2);
template <class Element> TMatrixT<Element>  operator-  (const TMatrixT   <Element> &source1,const TMatrixTSym<Element> &source2);
template <class Element> TMatrixT<Element>  operator-  (const TMatrixTSym<Element> &source1,const TMatrixT   <Element> &source2);
template <class Element> TMatrixT<Element>  operator-  (const TMatrixT   <Element> &source ,      Element               val    );
template <class Element> TMatrixT<Element>  operator-  (      Element               val    ,const TMatrixT   <Element> &source );
template <class Element> TMatrixT<Element>  operator*  (      Element               val    ,const TMatrixT   <Element> &source );
template <class Element> TMatrixT<Element>  operator*  (const TMatrixT   <Element> &source ,      Element               val    );
template <class Element> TMatrixT<Element>  operator*  (const TMatrixT   <Element> &source1,const TMatrixT   <Element> &source2);
template <class Element> TMatrixT<Element>  operator*  (const TMatrixT   <Element> &source1,const TMatrixTSym<Element> &source2);
template <class Element> TMatrixT<Element>  operator*  (const TMatrixTSym<Element> &source1,const TMatrixT   <Element> &source2);
template <class Element> TMatrixT<Element>  operator*  (const TMatrixTSym<Element> &source1,const TMatrixTSym<Element> &source2);
// Preventing warnings with -Weffc++ in GCC since overloading the || and && operators was a design choice.
#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) >= 40600
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#endif
template <class Element> TMatrixT<Element>  operator&& (const TMatrixT   <Element> &source1,const TMatrixT   <Element> &source2);
template <class Element> TMatrixT<Element>  operator&& (const TMatrixT   <Element> &source1,const TMatrixTSym<Element> &source2);
template <class Element> TMatrixT<Element>  operator&& (const TMatrixTSym<Element> &source1,const TMatrixT   <Element> &source2);
template <class Element> TMatrixT<Element>  operator|| (const TMatrixT   <Element> &source1,const TMatrixT   <Element> &source2);
template <class Element> TMatrixT<Element>  operator|| (const TMatrixT   <Element> &source1,const TMatrixTSym<Element> &source2);
template <class Element> TMatrixT<Element>  operator|| (const TMatrixTSym<Element> &source1,const TMatrixT   <Element> &source2);
#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) >= 40600
#pragma GCC diagnostic pop
#endif
template <class Element> TMatrixT<Element>  operator>  (const TMatrixT   <Element> &source1,const TMatrixT   <Element> &source2);
template <class Element> TMatrixT<Element>  operator>  (const TMatrixT   <Element> &source1,const TMatrixTSym<Element> &source2);
template <class Element> TMatrixT<Element>  operator>  (const TMatrixTSym<Element> &source1,const TMatrixT   <Element> &source2);
template <class Element> TMatrixT<Element>  operator>= (const TMatrixT   <Element> &source1,const TMatrixT   <Element> &source2);
template <class Element> TMatrixT<Element>  operator>= (const TMatrixT   <Element> &source1,const TMatrixTSym<Element> &source2);
template <class Element> TMatrixT<Element>  operator>= (const TMatrixTSym<Element> &source1,const TMatrixT   <Element> &source2);
template <class Element> TMatrixT<Element>  operator<= (const TMatrixT   <Element> &source1,const TMatrixT   <Element> &source2);
template <class Element> TMatrixT<Element>  operator<= (const TMatrixT   <Element> &source1,const TMatrixTSym<Element> &source2);
template <class Element> TMatrixT<Element>  operator<= (const TMatrixTSym<Element> &source1,const TMatrixT   <Element> &source2);
template <class Element> TMatrixT<Element>  operator<  (const TMatrixT   <Element> &source1,const TMatrixT   <Element> &source2);
template <class Element> TMatrixT<Element>  operator<  (const TMatrixT   <Element> &source1,const TMatrixTSym<Element> &source2);
template <class Element> TMatrixT<Element>  operator<  (const TMatrixTSym<Element> &source1,const TMatrixT   <Element> &source2);
template <class Element> TMatrixT<Element>  operator!= (const TMatrixT   <Element> &source1,const TMatrixT   <Element> &source2);
template <class Element> TMatrixT<Element>  operator!= (const TMatrixT   <Element> &source1,const TMatrixTSym<Element> &source2);
template <class Element> TMatrixT<Element>  operator!= (const TMatrixTSym<Element> &source1,const TMatrixT   <Element> &source2);

template <class Element> TMatrixT<Element> &Add        (TMatrixT<Element> &target,      Element               scalar,const TMatrixT   <Element> &source);
template <class Element> TMatrixT<Element> &Add        (TMatrixT<Element> &target,      Element               scalar,const TMatrixTSym<Element> &source);
template <class Element> TMatrixT<Element> &ElementMult(TMatrixT<Element> &target,const TMatrixT   <Element> &source);
template <class Element> TMatrixT<Element> &ElementMult(TMatrixT<Element> &target,const TMatrixTSym<Element> &source);
template <class Element> TMatrixT<Element> &ElementDiv (TMatrixT<Element> &target,const TMatrixT   <Element> &source);
template <class Element> TMatrixT<Element> &ElementDiv (TMatrixT<Element> &target,const TMatrixTSym<Element> &source);

template <class Element> void AMultB (const Element * const ap,Int_t na,Int_t ncolsa,
                                      const Element * const bp,Int_t nb,Int_t ncolsb,Element *cp);
template <class Element> void AtMultB(const Element * const ap,Int_t ncolsa,
                                      const Element * const bp,Int_t nb,Int_t ncolsb,Element *cp);
template <class Element> void AMultBt(const Element * const ap,Int_t na,Int_t ncolsa,
                                      const Element * const bp,Int_t nb,Int_t ncolsb,Element *cp);

#endif
