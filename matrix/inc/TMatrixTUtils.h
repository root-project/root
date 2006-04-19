// @(#)root/matrix:$Name:  $:$Id: TMatrixTUtils.h,v 1.2 2005/12/23 07:20:10 brun Exp $
// Authors: Fons Rademakers, Eddy Offermann   Nov 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMatrixTUtils
#define ROOT_TMatrixTUtils

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Matrix utility classes.                                              //
//                                                                      //
// Templates of utility classes in the Linear Algebra Package.          //
// The following classes are defined here:                              //
//                                                                      //
// Different matrix views without copying data elements :               //
//   TMatrixTRow_const        TMatrixTRow                               //
//   TMatrixTColumn_const     TMatrixTColumn                            //
//   TMatrixTDiag_const       TMatrixTDiag                              //
//   TMatrixTFlat_const       TMatrixTFlat                              //
//   TMatrixTSub_const        TMatrixTSub                               //
//   TMatrixTSparseRow_const  TMatrixTSparseRow                         //
//   TMatrixTSparseDiag_const TMatrixTSparseDiag                        //
//                                                                      //
//   TElementActionT                                                    //
//   TElementPosActionT                                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMatrixTBase
#include "TMatrixTBase.h"
#endif

template<class Element> class TVectorT;
template<class Element> class TMatrixT;
template<class Element> class TMatrixTSym;
template<class Element> class TMatrixTSparse;

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TElementActionT                                                      //
//                                                                      //
// A class to do a specific operation on every vector or matrix element //
// (regardless of it position) as the object is being traversed.        //
// This is an abstract class. Derived classes need to implement the     //
// action function Operation().                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

template<class Element> class TElementActionT {

#ifndef __CINT__
friend class TMatrixTBase  <Element>;
friend class TMatrixT      <Element>;
friend class TMatrixTSym   <Element>;
friend class TMatrixTSparse<Element>;
friend class TVectorT      <Element>;
#endif

protected:
   virtual ~TElementActionT() { }
   virtual void Operation(Element &element) const = 0;

private:
   void operator=(const TElementActionT<Element> &) { }
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TElementPosActionT                                                   //
//                                                                      //
// A class to do a specific operation on every vector or matrix element //
// as the object is being traversed. This is an abstract class.         //
// Derived classes need to implement the action function Operation().   //
// In the action function the location of the current element is        //
// known (fI=row, fJ=columns).                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

template<class Element> class TElementPosActionT {

#ifndef __CINT__
friend class TMatrixTBase  <Element>;
friend class TMatrixT      <Element>;
friend class TMatrixTSym   <Element>;
friend class TMatrixTSparse<Element>;
friend class TVectorT      <Element>;
#endif

protected:
   mutable Int_t fI; // i position of element being passed to Operation()
   mutable Int_t fJ; // j position of element being passed to Operation()
   virtual ~TElementPosActionT() { }
   virtual void Operation(Element &element) const = 0;

private:
   void operator=(const TElementPosActionT<Element> &) { }
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixTRow_const                                                    //
//                                                                      //
// Template class represents a row of a TMatrixT/TMatrixTSym            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

template<class Element> class TMatrixTRow_const {

protected:
  const TMatrixTBase<Element> *fMatrix;  //  the matrix I am a row of
        Int_t                  fRowInd;  //  effective row index
        Int_t                  fInc;     //  if ptr = @a[row,i], then ptr+inc = @a[row,i+1]
  const Element               *fPtr;     //  pointer to the a[row,0]

public:
  TMatrixTRow_const() { fMatrix = 0; fInc = 0; fPtr = 0; }
  TMatrixTRow_const(const TMatrixT   <Element> &matrix,Int_t row);
  TMatrixTRow_const(const TMatrixTSym<Element> &matrix,Int_t row);
  virtual ~TMatrixTRow_const() { }

  inline const TMatrixTBase<Element> *GetMatrix  () const { return fMatrix; }
  inline       Int_t                  GetRowIndex() const { return fRowInd; }
  inline       Int_t                  GetInc     () const { return fInc; }
  inline const Element               *GetPtr     () const { return fPtr; }
  inline const Element               &operator   ()(Int_t i) const { R__ASSERT(fMatrix->IsValid());
                                                                     const Int_t acoln = i-fMatrix->GetColLwb();
                                                                     R__ASSERT(acoln < fMatrix->GetNcols() && acoln >= 0);
                                                                     return fPtr[acoln]; }
  inline const Element               &operator   [](Int_t i) const { return (*(const TMatrixTRow_const<Element> *)this)(i); }

  ClassDef(TMatrixTRow_const,0)  // Template of General Matrix Row Access class
};

template<class Element> class TMatrixTRow : public TMatrixTRow_const<Element> {

public:
  TMatrixTRow() {}
  TMatrixTRow(TMatrixT   <Element> &matrix,Int_t row);
  TMatrixTRow(TMatrixTSym<Element> &matrix,Int_t row);
  TMatrixTRow(const TMatrixTRow<Element> &mr);

  inline Element *GetPtr() const { return const_cast<Element *>(this->fPtr); }

  inline const Element &operator()(Int_t i) const { R__ASSERT(this->fMatrix->IsValid());
                                                    const Int_t acoln = i-this->fMatrix->GetColLwb();
                                                    R__ASSERT(acoln < this->fMatrix->GetNcols() && acoln >= 0);
                                                    return this->fPtr[acoln]; }
  inline       Element &operator()(Int_t i)       { R__ASSERT(this->fMatrix->IsValid());
                                                    const Int_t acoln = i-this->fMatrix->GetColLwb();
                                                    R__ASSERT(acoln < this->fMatrix->GetNcols() && acoln >= 0);
                                                    return (const_cast<Element *>(this->fPtr))[acoln]; }
  inline const Element &operator[](Int_t i) const { return (*(const TMatrixTRow<Element> *)this)(i); }
  inline       Element &operator[](Int_t i)       { return (*(      TMatrixTRow<Element> *)this)(i); }

  void operator= (Element val);
  void operator+=(Element val);
  void operator*=(Element val);

  void operator=(const TMatrixTRow_const<Element> &r);
  void operator=(const TMatrixTRow      <Element> &r) { operator=((TMatrixTRow_const<Element> &)r); }
  void operator=(const TVectorT         <Element> &vec);

  void operator+=(const TMatrixTRow_const<Element> &r);
  void operator*=(const TMatrixTRow_const<Element> &r);

  ClassDef(TMatrixTRow,0)  // Template of General Matrix Row Access class
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixTColumn_const                                                 //
//                                                                      //
// Template class represents a column of a TMatrixT/TMatrixTSym         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

template<class Element> class TMatrixTColumn_const {

protected:
  const TMatrixTBase<Element> *fMatrix;  //  the matrix I am a column of
        Int_t                  fColInd;  //  effective column index
        Int_t                  fInc;     //  if ptr = @a[i,col], then ptr+inc = @a[i+1,col]
  const Element               *fPtr;     //  pointer to the a[0,col] column

public:
  TMatrixTColumn_const() { fMatrix = 0; fInc = 0; fPtr = 0; }
  TMatrixTColumn_const(const TMatrixT   <Element> &matrix,Int_t col);
  TMatrixTColumn_const(const TMatrixTSym<Element> &matrix,Int_t col);
  virtual ~TMatrixTColumn_const() { }

  inline const TMatrixTBase <Element> *GetMatrix  () const { return fMatrix; }
  inline       Int_t                   GetColIndex() const { return fColInd; }
  inline       Int_t                   GetInc     () const { return fInc; }
  inline const Element                *GetPtr     () const { return fPtr; }
  inline const Element                &operator   ()(Int_t i) const { R__ASSERT(fMatrix->IsValid());
                                                                      const Int_t arown = i-fMatrix->GetRowLwb();
                                                                      R__ASSERT(arown < fMatrix->GetNrows() && arown >= 0);
                                                                      return fPtr[arown*fInc]; }
  inline const Element      &operator [](Int_t i) const { return (*(const TMatrixTColumn_const<Element> *)this)(i); }

  ClassDef(TMatrixTColumn_const,0)  // Template of General Matrix Column Access class
};

template<class Element> class TMatrixTColumn : public TMatrixTColumn_const<Element> {

public:
  TMatrixTColumn() {}
  TMatrixTColumn(TMatrixT   <Element>&matrix,Int_t col);
  TMatrixTColumn(TMatrixTSym<Element>&matrix,Int_t col);
  TMatrixTColumn(const TMatrixTColumn <Element>&mc);

  inline Element *GetPtr() const { return const_cast<Element *>(this->fPtr); }

  inline const Element &operator()(Int_t i) const { R__ASSERT(this->fMatrix->IsValid());
                                                    const Int_t arown = i-this->fMatrix->GetRowLwb();
                                                    R__ASSERT(arown < this->fMatrix->GetNrows() && arown >= 0);
                                                    return this->fPtr[arown]; }
  inline       Element &operator()(Int_t i)       { R__ASSERT(this->fMatrix->IsValid());
                                                    const Int_t arown = i-this->fMatrix->GetRowLwb();
                                                    R__ASSERT(arown < this->fMatrix->GetNrows() && arown >= 0);
                                                    return (const_cast<Element *>(this->fPtr))[arown*this->fInc]; }
  inline const Element &operator[](Int_t i) const { return (*(const TMatrixTColumn<Element> *)this)(i); }
  inline       Element &operator[](Int_t i)       { return (*(      TMatrixTColumn<Element> *)this)(i); }

  void operator= (Element val);
  void operator+=(Element val);
  void operator*=(Element val);

  void operator=(const TMatrixTColumn_const<Element> &c);
  void operator=(const TMatrixTColumn      <Element> &c) { operator=((TMatrixTColumn_const<Element> &)c); }
  void operator=(const TVectorT            <Element> &vec);

  void operator+=(const TMatrixTColumn_const<Element> &c);
  void operator*=(const TMatrixTColumn_const<Element> &c);

  ClassDef(TMatrixTColumn,0)  // Template of General Matrix Column Access class
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixTDiag_const                                                   //
//                                                                      //
// Template class represents the diagonal of a TMatrixT/TMatrixTSym     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

template<class Element> class TMatrixTDiag_const {

protected:
  const TMatrixTBase<Element> *fMatrix;  //  the matrix I am the diagonal of
        Int_t                  fInc;     //  if ptr=@a[i,i], then ptr+inc = @a[i+1,i+1]
        Int_t                  fNdiag;   //  number of diag elems, min(nrows,ncols)
  const Element               *fPtr;     //  pointer to the a[0,0]

public:
  TMatrixTDiag_const() { fMatrix = 0; fInc = 0; fNdiag = 0; fPtr = 0; }
  TMatrixTDiag_const(const TMatrixT   <Element> &matrix);
  TMatrixTDiag_const(const TMatrixTSym<Element> &matrix);
  virtual ~TMatrixTDiag_const() { }

  inline const TMatrixTBase<Element> *GetMatrix() const { return fMatrix; }
  inline const Element               *GetPtr   () const { return fPtr; }
  inline       Int_t                  GetInc   () const { return fInc; }
  inline const Element               &operator ()(Int_t i) const { R__ASSERT(fMatrix->IsValid());
                                                          R__ASSERT(i < fNdiag && i >= 0); return fPtr[i*fInc]; }
  inline const Element               &operator [](Int_t i) const { return (*(const TMatrixTDiag_const<Element> *)this)(i); }

  Int_t GetNdiags() const { return fNdiag; }

  ClassDef(TMatrixTDiag_const,0)  // Template of General Matrix Diagonal Access class
};

template<class Element> class TMatrixTDiag : public TMatrixTDiag_const<Element> {

public:
  TMatrixTDiag() {}
  TMatrixTDiag(TMatrixT   <Element>&matrix);
  TMatrixTDiag(TMatrixTSym<Element>&matrix);
  TMatrixTDiag(const TMatrixTDiag<Element> &md);

  inline Element *GetPtr() const { return const_cast<Element *>(this->fPtr); }

  inline const Element &operator()(Int_t i) const { R__ASSERT(this->fMatrix->IsValid());
                                                    R__ASSERT(i < this->fNdiag && i >= 0); return this->fPtr[i*this->fInc]; }
  inline       Element &operator()(Int_t i)       { R__ASSERT(this->fMatrix->IsValid());
                                                    R__ASSERT(i < this->fNdiag && i >= 0);
                                                    return (const_cast<Element *>(this->fPtr))[i*this->fInc]; }
  inline const Element &operator[](Int_t i) const { return (*(const TMatrixTDiag<Element> *)this)(i); }
  inline       Element &operator[](Int_t i)       { return (*(      TMatrixTDiag *)this)(i); }

  void operator= (Element val);
  void operator+=(Element val);
  void operator*=(Element val);

  void operator=(const TMatrixTDiag_const<Element> &d);
  void operator=(const TMatrixTDiag      <Element> &d) { operator=((TMatrixTDiag_const<Element> &)d); }
  void operator=(const TVectorT          <Element> &vec);

  void operator+=(const TMatrixTDiag_const<Element> &d);
  void operator*=(const TMatrixTDiag_const<Element> &d);

  ClassDef(TMatrixTDiag,0)  // Template of General Matrix Diagonal Access class
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixTFlat_const                                                   //
//                                                                      //
// Template class represents a flat TMatrixT/TMatrixTSym                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

template<class Element> class TMatrixTFlat_const {

protected:
  const TMatrixTBase<Element> *fMatrix;  //  the matrix I am the diagonal of
        Int_t                  fNelems;  //
  const Element               *fPtr;     //  pointer to the a[0,0]

public:
  TMatrixTFlat_const() { fMatrix = 0; fNelems = 0; fPtr = 0; }
  TMatrixTFlat_const(const TMatrixT   <Element> &matrix);
  TMatrixTFlat_const(const TMatrixTSym<Element> &matrix);
  virtual ~TMatrixTFlat_const() { }

  inline const TMatrixTBase<Element> *GetMatrix() const { return fMatrix; }
  inline const Element               *GetPtr   () const { return fPtr; }
  inline const Element               &operator ()(Int_t i) const { R__ASSERT(fMatrix->IsValid());
                                                           R__ASSERT(i >=0 && i < fNelems); return fPtr[i]; }
  inline const Element               &operator [](Int_t i) const { return (*(const TMatrixTFlat_const<Element> *)this)(i); }

  ClassDef(TMatrixTFlat_const,0)  // Template of General Matrix Flat Representation class
};

template<class Element> class TMatrixTFlat : public TMatrixTFlat_const<Element> {

public:
  TMatrixTFlat() {}
  TMatrixTFlat(TMatrixT   <Element> &matrix);
  TMatrixTFlat(TMatrixTSym<Element> &matrix);
  TMatrixTFlat(const TMatrixTFlat<Element> &mf);

  inline Element *GetPtr() const { return const_cast<Element *>(this->fPtr); }

  inline const Element &operator()(Int_t i) const { R__ASSERT(this->fMatrix->IsValid());
                                                    R__ASSERT(i >=0 && i < this->fNelems); return this->fPtr[i]; }
  inline       Element &operator()(Int_t i)       { R__ASSERT(this->fMatrix->IsValid());
                                                    R__ASSERT(i >=0 && i < this->fNelems);
                                                    return (const_cast<Element *>(this->fPtr))[i]; }
  inline const Element &operator[](Int_t i) const { return (*(const TMatrixTFlat<Element> *)this)(i); }
  inline       Element &operator[](Int_t i)       { return (*(      TMatrixTFlat<Element> *)this)(i); }

  void operator= (Element val);
  void operator+=(Element val);
  void operator*=(Element val);

  void operator=(const TMatrixTFlat_const<Element> &f);
  void operator=(const TMatrixTFlat      <Element> &f) { operator=((TMatrixTFlat_const<Element> &)f); }
  void operator=(const TVectorT          <Element> &vec);

  void operator+=(const TMatrixTFlat_const<Element> &f);
  void operator*=(const TMatrixTFlat_const<Element> &f);

  ClassDef(TMatrixTFlat,0)  // Template of General Matrix Flat Representation class
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixTSub_const                                                    //
//                                                                      //
// Template class represents a sub matrix of TMatrixT/TMatrixTSym       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

template<class Element> class TMatrixTSub_const {

protected:
  const TMatrixTBase<Element> *fMatrix;    //  the matrix I am a submatrix of
        Int_t                  fRowOff;    //
        Int_t                  fColOff;    //
        Int_t                  fNrowsSub;  //
        Int_t                  fNcolsSub;  //

public:
  TMatrixTSub_const() { fRowOff = fColOff = fNrowsSub = fNcolsSub = 0; fMatrix = 0; }
  TMatrixTSub_const(const TMatrixT   <Element> &matrix,Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb);
  TMatrixTSub_const(const TMatrixTSym<Element> &matrix,Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb);
  virtual ~TMatrixTSub_const() { }

  inline const TMatrixTBase<Element> *GetMatrix() const { return fMatrix; }
  inline       Int_t                  GetRowOff() const { return fRowOff; }
  inline       Int_t                  GetColOff() const { return fColOff; }
  inline       Int_t                  GetNrows () const { return fNrowsSub; }
  inline       Int_t                  GetNcols () const { return fNcolsSub; }
  inline const Element               &operator ()(Int_t rown,Int_t coln) const
                                                    { R__ASSERT(fMatrix->IsValid());
                                                      R__ASSERT(rown < fNrowsSub && rown >= 0);
                                                      R__ASSERT(coln < fNcolsSub && coln >= 0);
                                                      const Int_t index = (rown+fRowOff)*fMatrix->GetNcols()+coln+fColOff;
                                                      const Element *ptr = fMatrix->GetMatrixArray();
                                                      return ptr[index]; }

  ClassDef(TMatrixTSub_const,0)  // Template of Sub Matrix Access class
};

template<class Element> class TMatrixTSub : public TMatrixTSub_const<Element> {

public:

  enum {kWorkMax = 100};

  TMatrixTSub() {}
  TMatrixTSub(TMatrixT   <Element> &matrix,Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb);
  TMatrixTSub(TMatrixTSym<Element> &matrix,Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb);
  TMatrixTSub(const TMatrixTSub<Element> &ms);

  inline       Element &operator()(Int_t rown,Int_t coln)
                                                    { R__ASSERT(this->fMatrix->IsValid());
                                                      R__ASSERT(rown < this->fNrowsSub && rown >= 0);
                                                      R__ASSERT(coln < this->fNcolsSub && coln >= 0);
                                                      const Int_t index = (rown+this->fRowOff)*this->fMatrix->GetNcols()+
                                                                           coln+this->fColOff;
                                                      const Element *ptr = this->fMatrix->GetMatrixArray();
                                                      return (const_cast<Element *>(ptr))[index]; }

  void Rank1Update(const TVectorT<Element> &vec,Element alpha=1.0);

  void operator= (Element val);
  void operator+=(Element val);
  void operator*=(Element val);

  void operator=(const TMatrixTSub_const<Element> &s);
  void operator=(const TMatrixTSub      <Element> &s) { operator=((TMatrixTSub_const<Element> &)s); }
  void operator=(const TMatrixTBase     <Element> &m);

  void operator+=(const TMatrixTSub_const<Element> &s);
  void operator*=(const TMatrixTSub_const<Element> &s);
  void operator+=(const TMatrixTBase     <Element> &m);
  void operator*=(const TMatrixT         <Element> &m);
  void operator*=(const TMatrixTSym      <Element> &m);

  ClassDef(TMatrixTSub,0)  // Template of Sub Matrix Access class
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixTSparseRow_const                                              //
//                                                                      //
// Template class represents a row of TMatrixTSparse                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

template<class Element> class TMatrixTSparseRow_const {

protected:
  const TMatrixTBase<Element> *fMatrix;  // the matrix I am a row of
        Int_t                  fRowInd;  // effective row index
        Int_t                  fNindex;  // index range
  const Int_t                 *fColPtr;  // column index pointer
  const Element               *fDataPtr; // data pointer

public:
  TMatrixTSparseRow_const() { fMatrix = 0; fRowInd = 0; fNindex = 0; fColPtr = 0; fDataPtr = 0; }
  TMatrixTSparseRow_const(const TMatrixTSparse<Element> &matrix,Int_t row);
  virtual ~TMatrixTSparseRow_const() { }

  inline const TMatrixTBase<Element> *GetMatrix  () const { return fMatrix; }
  inline const Element               *GetDataPtr () const { return fDataPtr; }
  inline const Int_t                 *GetColPtr  () const { return fColPtr; }
  inline       Int_t                  GetRowIndex() const { return fRowInd; }
  inline       Int_t                  GetNindex  () const { return fNindex; }

  inline Element operator()(Int_t i) const { R__ASSERT(fMatrix->IsValid());
                                             const Int_t acoln = i-fMatrix->GetColLwb();
                                             R__ASSERT(acoln < fMatrix->GetNcols() && acoln >= 0);
                                             const Int_t index = TMath::BinarySearch(fNindex,fColPtr,acoln);
                                             if (index >= 0 && fColPtr[index] == acoln) return fDataPtr[index];
                                             else                                       return 0.0; }
  inline Element operator[](Int_t i) const { return (*(const TMatrixTSparseRow_const<Element> *)this)(i); }

  ClassDef(TMatrixTSparseRow_const,0)  // Template of Sparse Matrix Row Access class
};

template<class Element> class TMatrixTSparseRow : public TMatrixTSparseRow_const<Element> {

public:
  TMatrixTSparseRow() {}
  TMatrixTSparseRow(TMatrixTSparse<Element> &matrix,Int_t row);
  TMatrixTSparseRow(const TMatrixTSparseRow<Element> &mr);

  inline Element *GetDataPtr() const { return const_cast<Element *>(this->fDataPtr); }

  inline Element  operator()(Int_t i) const { R__ASSERT(this->fMatrix->IsValid());
                                              const Int_t acoln = i-this->fMatrix->GetColLwb();
                                              R__ASSERT(acoln < this->fMatrix->GetNcols() && acoln >= 0);
                                              const Int_t index = TMath::BinarySearch(this->fNindex,this->fColPtr,acoln);
                                              if (index >= 0 && this->fColPtr[index] == acoln) return this->fDataPtr[index];
                                              else                                             return 0.0; }
         Element &operator()(Int_t i);
  inline Element  operator[](Int_t i) const { return (*(const TMatrixTSparseRow<Element> *)this)(i); }
  inline Element &operator[](Int_t i)       { return (*(TMatrixTSparseRow<Element> *)this)(i); }

  void operator= (Element val);
  void operator+=(Element val);
  void operator*=(Element val);

  void operator=(const TMatrixTSparseRow_const<Element> &r);
  void operator=(const TMatrixTSparseRow      <Element> &r) { operator=((TMatrixTSparseRow_const<Element> &)r); }
  void operator=(const TVectorT               <Element> &vec);

  void operator+=(const TMatrixTSparseRow_const<Element> &r);
  void operator*=(const TMatrixTSparseRow_const<Element> &r);

  ClassDef(TMatrixTSparseRow,0)  // Template of Sparse Matrix Row Access class
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixTSparseDiag_const                                             //
//                                                                      //
// Template class represents the diagonal of TMatrixTSparse             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

template<class Element> class TMatrixTSparseDiag_const {

protected:
  const TMatrixTBase<Element> *fMatrix;  //  the matrix I am the diagonal of
        Int_t                  fNdiag;   //  number of diag elems, min(nrows,ncols)
  const Element               *fDataPtr; //  data pointer

public:
  TMatrixTSparseDiag_const() { fMatrix = 0; fNdiag = 0; fDataPtr = 0; }
  TMatrixTSparseDiag_const(const TMatrixTSparse<Element> &matrix);
  virtual ~TMatrixTSparseDiag_const() { }

  inline const TMatrixTBase<Element> *GetMatrix () const { return fMatrix; }
  inline const Element               *GetDataPtr() const { return fDataPtr; }
  inline       Int_t                  GetNdiags () const { return fNdiag; }

  inline Element operator ()(Int_t i) const { R__ASSERT(fMatrix->IsValid());
                                              R__ASSERT(i < fNdiag && i >= 0);
                                              const Int_t   * const pR = fMatrix->GetRowIndexArray();
                                              const Int_t   * const pC = fMatrix->GetColIndexArray();
                                              const Element * const pD = fMatrix->GetMatrixArray();
                                              const Int_t sIndex = pR[i];
                                              const Int_t eIndex = pR[i+1];
                                              const Int_t index = TMath::BinarySearch(eIndex-sIndex,pC+sIndex,i)+sIndex;
                                              if (index >= sIndex && pC[index] == i) return pD[index];
                                              else                                   return 0.0; }

  inline Element operator [](Int_t i) const { return (*(const TMatrixTSparseRow_const<Element> *)this)(i); }

  ClassDef(TMatrixTSparseDiag_const,0)  // Template of Sparse Matrix Diagonal Access class
};

template<class Element> class TMatrixTSparseDiag : public TMatrixTSparseDiag_const<Element> {

public:
  TMatrixTSparseDiag() {}
  TMatrixTSparseDiag(TMatrixTSparse<Element> &matrix);
  TMatrixTSparseDiag(const TMatrixTSparseDiag<Element> &md);

  inline Element *GetDataPtr() const { return const_cast<Element *>(this->fDataPtr); }

  inline       Element  operator()(Int_t i) const { R__ASSERT(this->fMatrix->IsValid());
                                                    R__ASSERT(i < this->fNdiag && i >= 0);
                                                    const Int_t   * const pR = this->fMatrix->GetRowIndexArray();
                                                    const Int_t   * const pC = this->fMatrix->GetColIndexArray();
                                                    const Element * const pD = this->fMatrix->GetMatrixArray();
                                                    const Int_t sIndex = pR[i];
                                                    const Int_t eIndex = pR[i+1];
                                                    const Int_t index = TMath::BinarySearch(eIndex-sIndex,pC+sIndex,i)+sIndex;
                                                    if (index >= sIndex && pC[index] == i) return pD[index];
                                                    else                                   return 0.0; }
               Element &operator()(Int_t i);
  inline       Element  operator[](Int_t i) const { return (*(const TMatrixTSparseDiag<Element> *)this)(i); }
  inline       Element &operator[](Int_t i)       { return (*(TMatrixTSparseDiag<Element> *)this)(i); }

  void operator= (Element val);
  void operator+=(Element val);
  void operator*=(Element val);

  void operator=(const TMatrixTSparseDiag_const<Element> &d);
  void operator=(const TMatrixTSparseDiag      <Element> &d) { operator=((TMatrixTSparseDiag_const<Element> &)d); }
  void operator=(const TVectorT                <Element> &vec);

  void operator+=(const TMatrixTSparseDiag_const<Element> &d);
  void operator*=(const TMatrixTSparseDiag_const<Element> &d);

  ClassDef(TMatrixTSparseDiag,0)  // Template of Sparse Matrix Diagonal Access class
};

Double_t Drand(Double_t &ix);
#endif
