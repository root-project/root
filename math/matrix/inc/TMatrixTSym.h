// @(#)root/matrix:$Id$
// Authors: Fons Rademakers, Eddy Offermann   Nov 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMatrixTSym
#define ROOT_TMatrixTSym

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixTSym                                                          //
//                                                                      //
// Implementation of a symmetric matrix in the linear algebra package   //
//                                                                      //
// Note that in this implementation both matrix element m[i][j] and     //
// m[j][i] are updated and stored in memory . However, when making the  //
// object persistent only the upper right triangle is stored .          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMatrixTBase.h"
#include "TMatrixTUtils.h"

template<class Element>class TMatrixT;
template<class Element>class TMatrixTSymLazy;
template<class Element>class TVectorT;

template<class Element> class TMatrixTSym : public TMatrixTBase<Element> {

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

   enum {kWorkMax = 100}; // size of work array
   enum EMatrixCreatorsOp1 { kZero,kUnit,kTransposed,kInverted,kAtA };
   enum EMatrixCreatorsOp2 { kPlus,kMinus };

   TMatrixTSym() { fElements = nullptr; }
   explicit TMatrixTSym(Int_t nrows);
   TMatrixTSym(Int_t row_lwb,Int_t row_upb);
   TMatrixTSym(Int_t nrows,const Element *data,Option_t *option="");
   TMatrixTSym(Int_t row_lwb,Int_t row_upb,const Element *data,Option_t *option="");
   TMatrixTSym(const TMatrixTSym<Element> &another);
   template <class Element2> TMatrixTSym(const TMatrixTSym<Element2> &another)
   {
      R__ASSERT(another.IsValid());
      Allocate(another.GetNrows(),another.GetNcols(),another.GetRowLwb(),another.GetColLwb());
      *this = another;
   }

   TMatrixTSym(EMatrixCreatorsOp1 op,const TMatrixTSym<Element> &prototype);
   TMatrixTSym(EMatrixCreatorsOp1 op,const TMatrixT   <Element> &prototype);
   TMatrixTSym(const TMatrixTSym<Element> &a,EMatrixCreatorsOp2 op,const TMatrixTSym<Element> &b);
   TMatrixTSym(const TMatrixTSymLazy<Element> &lazy_constructor);

   ~TMatrixTSym() override { TMatrixTSym::Clear(); }

   // Elementary constructors
   void TMult(const TMatrixT   <Element> &a);
   void TMult(const TMatrixTSym<Element> &a);
   void Mult (const TMatrixTSym<Element> &a) { TMult(a); }

   void Plus (const TMatrixTSym<Element> &a,const TMatrixTSym<Element> &b);
   void Minus(const TMatrixTSym<Element> &a,const TMatrixTSym<Element> &b);

   const Element *GetMatrixArray  () const override;
         Element *GetMatrixArray  () override;
   const Int_t   *GetRowIndexArray() const override { return nullptr; }
         Int_t   *GetRowIndexArray() override       { return nullptr; }
   const Int_t   *GetColIndexArray() const override { return nullptr; }
         Int_t   *GetColIndexArray() override       { return nullptr; }

         TMatrixTBase<Element> &SetRowIndexArray(Int_t * /*data*/) override { MayNotUse("SetRowIndexArray(Int_t *)"); return *this; }
         TMatrixTBase<Element> &SetColIndexArray(Int_t * /*data*/) override { MayNotUse("SetColIndexArray(Int_t *)"); return *this; }

   void   Clear      (Option_t * /*option*/ ="") override { if (this->fIsOwner) Delete_m(this->fNelems,fElements);
                                                           else fElements = nullptr;
                                                           this->fNelems = 0; }
   Bool_t IsSymmetric() const override { return kTRUE; }

           TMatrixTSym <Element> &Use           (Int_t row_lwb,Int_t row_upb,Element *data);
   const   TMatrixTSym <Element> &Use           (Int_t row_lwb,Int_t row_upb,const Element *data) const
                                                  { return (const TMatrixTSym<Element>&)
                                                           ((const_cast<TMatrixTSym<Element> *>(this))->Use(row_lwb,row_upb,const_cast<Element *>(data))); }
           TMatrixTSym <Element> &Use           (Int_t nrows,Element *data);
   const   TMatrixTSym <Element> &Use           (Int_t nrows,const Element *data) const;
           TMatrixTSym <Element> &Use           (TMatrixTSym<Element> &a);
   const   TMatrixTSym <Element> &Use           (const TMatrixTSym<Element> &a) const;

           TMatrixTSym <Element> &GetSub        (Int_t row_lwb,Int_t row_upb,TMatrixTSym<Element> &target,Option_t *option="S") const;
   TMatrixTBase<Element> &GetSub        (Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,
                                                TMatrixTBase<Element> &target,Option_t *option="S") const override;
           TMatrixTSym <Element>  GetSub        (Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,Option_t *option="S") const;
           TMatrixTSym <Element> &SetSub        (Int_t row_lwb,const TMatrixTBase<Element> &source);
   TMatrixTBase<Element> &SetSub        (Int_t row_lwb,Int_t col_lwb,const TMatrixTBase<Element> &source) override;

   TMatrixTBase<Element> &SetMatrixArray(const Element *data, Option_t *option="") override;

   TMatrixTBase<Element> &Shift         (Int_t row_shift,Int_t col_shift) override;
   TMatrixTBase<Element> &ResizeTo      (Int_t nrows,Int_t ncols,Int_t /*nr_nonzeros*/ =-1) override;
   TMatrixTBase<Element> &ResizeTo      (Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,Int_t /*nr_nonzeros*/ =-1) override;
   inline  TMatrixTBase<Element> &ResizeTo      (const TMatrixTSym<Element> &m) {
                                                return ResizeTo(m.GetRowLwb(),m.GetRowUpb(),m.GetColLwb(),m.GetColUpb()); }

   Double_t      Determinant   () const override;
   void          Determinant   (Double_t &d1,Double_t &d2) const override;

           TMatrixTSym<Element>  &Invert        (Double_t *det=nullptr);
           TMatrixTSym<Element>  &InvertFast    (Double_t *det=nullptr);
           TMatrixTSym<Element>  &Transpose     (const TMatrixTSym<Element> &source);
   inline  TMatrixTSym<Element>  &T             () { return this->Transpose(*this); }
           TMatrixTSym<Element>  &Rank1Update   (const TVectorT   <Element> &v,Element alpha=1.0);
           TMatrixTSym<Element>  &Similarity    (const TMatrixT   <Element> &n);
           TMatrixTSym<Element>  &Similarity    (const TMatrixTSym<Element> &n);
           Element                Similarity    (const TVectorT   <Element> &v) const;
           TMatrixTSym<Element>  &SimilarityT   (const TMatrixT   <Element> &n);

   // Either access a_ij as a(i,j)
   inline       Element                    operator()(Int_t rown,Int_t coln) const override;
   inline       Element                   &operator()(Int_t rown,Int_t coln) override;

   // or as a[i][j]
   inline const TMatrixTRow_const<Element> operator[](Int_t rown) const { return TMatrixTRow_const<Element>(*this,rown); }
   inline       TMatrixTRow      <Element> operator[](Int_t rown)       { return TMatrixTRow      <Element>(*this,rown); }

   TMatrixTSym<Element> &operator= (const TMatrixTSym    <Element> &source);
   TMatrixTSym<Element> &operator= (const TMatrixTSymLazy<Element> &source);
   template <class Element2> TMatrixTSym<Element> &operator= (const TMatrixTSym<Element2> &source)
   {
      if (!AreCompatible(*this,source)) {
         Error("operator=(const TMatrixTSym2 &)","matrices not compatible");
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

   TMatrixTSym<Element> &operator= (Element val);
   TMatrixTSym<Element> &operator-=(Element val);
   TMatrixTSym<Element> &operator+=(Element val);
   TMatrixTSym<Element> &operator*=(Element val);

   TMatrixTSym &operator+=(const TMatrixTSym &source);
   TMatrixTSym &operator-=(const TMatrixTSym &source);

   TMatrixTBase<Element> &Apply(const TElementActionT   <Element> &action) override;
   TMatrixTBase<Element> &Apply(const TElementPosActionT<Element> &action) override;

   TMatrixTBase<Element> &Randomize  (Element alpha,Element beta,Double_t &seed) override;
   virtual TMatrixTSym <Element> &RandomizePD(Element alpha,Element beta,Double_t &seed);

   const TMatrixT<Element> EigenVectors(TVectorT<Element> &eigenValues) const;

   ClassDefOverride(TMatrixTSym,2) // Template of Symmetric Matrix class
};
#ifndef __CINT__
// When building with -fmodules, it instantiates all pending instantiations,
// instead of delaying them until the end of the translation unit.
// We 'got away with' probably because the use and the definition of the
// explicit specialization do not occur in the same TU.
//
// In case we are building with -fmodules, we need to forward declare the
// specialization in order to compile the dictionary G__Matrix.cxx.
template <> TClass *TMatrixTSym<double>::Class();
#endif // __CINT__

template <class Element> inline const Element               *TMatrixTSym<Element>::GetMatrixArray() const { return fElements; }
template <class Element> inline       Element               *TMatrixTSym<Element>::GetMatrixArray()       { return fElements; }

template <class Element> inline       TMatrixTSym<Element>  &TMatrixTSym<Element>::Use           (Int_t nrows,Element *data) { return Use(0,nrows-1,data); }
template <class Element> inline const TMatrixTSym<Element>  &TMatrixTSym<Element>::Use           (Int_t nrows,const Element *data) const
                                                                                                   { return Use(0,nrows-1,data); }
template <class Element> inline       TMatrixTSym<Element>  &TMatrixTSym<Element>::Use           (TMatrixTSym<Element> &a)
                                                                                                 { return Use(a.GetRowLwb(),a.GetRowUpb(),a.GetMatrixArray()); }
template <class Element> inline const TMatrixTSym<Element>  &TMatrixTSym<Element>::Use           (const TMatrixTSym<Element> &a) const
                                                                                                 { return Use(a.GetRowLwb(),a.GetRowUpb(),a.GetMatrixArray()); }

template <class Element> inline       TMatrixTSym<Element>   TMatrixTSym<Element>::GetSub        (Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,
                                                                                                  Option_t *option) const
                                                                                                 {
                                                                                                   TMatrixTSym<Element> tmp;
                                                                                                   this->GetSub(row_lwb,row_upb,col_lwb,col_upb,tmp,option);
                                                                                                   return tmp;
                                                                                                 }

template <class Element> inline Element TMatrixTSym<Element>::operator()(Int_t rown,Int_t coln) const
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

template <class Element> inline Element &TMatrixTSym<Element>::operator()(Int_t rown,Int_t coln)
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

template <class Element> Bool_t                operator== (const TMatrixTSym<Element> &source1,const TMatrixTSym<Element>  &source2);
template <class Element> TMatrixTSym<Element>  operator+  (const TMatrixTSym<Element> &source1,const TMatrixTSym<Element>  &source2);
template <class Element> TMatrixTSym<Element>  operator+  (const TMatrixTSym<Element> &source1,      Element                val);
template <class Element> TMatrixTSym<Element>  operator+  (      Element               val    ,const TMatrixTSym<Element>  &source2);
template <class Element> TMatrixTSym<Element>  operator-  (const TMatrixTSym<Element> &source1,const TMatrixTSym<Element>  &source2);
template <class Element> TMatrixTSym<Element>  operator-  (const TMatrixTSym<Element> &source1,      Element                val);
template <class Element> TMatrixTSym<Element>  operator-  (      Element               val    ,const TMatrixTSym<Element>  &source2);
template <class Element> TMatrixTSym<Element>  operator*  (const TMatrixTSym<Element> &source,       Element                val    );
template <class Element> TMatrixTSym<Element>  operator*  (      Element               val,    const TMatrixTSym<Element>  &source );
// Preventing warnings with -Weffc++ in GCC since overloading the || and && operators was a design choice.
#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) >= 40600
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#endif
template <class Element> TMatrixTSym<Element>  operator&& (const TMatrixTSym<Element> &source1,const TMatrixTSym<Element>  &source2);
template <class Element> TMatrixTSym<Element>  operator|| (const TMatrixTSym<Element> &source1,const TMatrixTSym<Element>  &source2);
#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) >= 40600
#pragma GCC diagnostic pop
#endif
template <class Element> TMatrixTSym<Element>  operator>  (const TMatrixTSym<Element> &source1,const TMatrixTSym<Element>  &source2);
template <class Element> TMatrixTSym<Element>  operator>= (const TMatrixTSym<Element> &source1,const TMatrixTSym<Element>  &source2);
template <class Element> TMatrixTSym<Element>  operator<= (const TMatrixTSym<Element> &source1,const TMatrixTSym<Element>  &source2);
template <class Element> TMatrixTSym<Element>  operator<  (const TMatrixTSym<Element> &source1,const TMatrixTSym<Element>  &source2);

template <class Element> TMatrixTSym<Element> &Add        (TMatrixTSym<Element> &target,      Element               scalar,const TMatrixTSym<Element> &source);
template <class Element> TMatrixTSym<Element> &ElementMult(TMatrixTSym<Element> &target,const TMatrixTSym<Element> &source);
template <class Element> TMatrixTSym<Element> &ElementDiv (TMatrixTSym<Element> &target,const TMatrixTSym<Element> &source);

#endif
