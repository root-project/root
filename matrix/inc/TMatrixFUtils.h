// @(#)root/matrix:$Name:  $:$Id: TMatrixFUtils.h,v 1.1 2004/01/25 20:33:32 brun Exp $
// Authors: Fons Rademakers, Eddy Offermann   Nov 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMatrixFUtils
#define ROOT_TMatrixFUtils

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Matrix utility classes.                                              //
//                                                                      //
// This file defines utility classes for the Linear Algebra Package.    //
// The following classes are defined here:                              //
//                                                                      //
// Different matrix views without copying data elements :               //
//   TMatrixFRow_const    TMatrixFRow                                   //
//   TMatrixFColumn_const TMatrixFColumn                                //
//   TMatrixFDiag_const   TMatrixFDiag                                  //
//   TMatrixFFlat_const   TMatrixFFlat                                  //
//                                                                      //
//   TElementActionF                                                    //
//   TElementPosActionF                                                 //
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
// TElementActionF                                                      //
//                                                                      //
// A class to do a specific operation on every vector or matrix element //
// (regardless of it position) as the object is being traversed.        //
// This is an abstract class. Derived classes need to implement the     //
// action function Operation().                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TElementActionF {

friend class TMatrixF;
friend class TMatrixFSym;
friend class TVectorF;

private:
  virtual void Operation(Float_t &element) const = 0;
  void operator=(const TElementActionF &) { }
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TElementPosActionF                                                   //
//                                                                      //
// A class to do a specific operation on every vector or matrix element //
// as the object is being traversed. This is an abstract class.         //
// Derived classes need to implement the action function Operation().   //
// In the action function the location of the current element is        //
// known (fI=row, fJ=columns).                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TElementPosActionF {

friend class TMatrixF;
friend class TMatrixFSym;
friend class TVectorF;

protected:
  mutable Int_t fI; // i position of element being passed to Operation()
  mutable Int_t fJ; // j position of element being passed to Operation()

private:
  virtual void Operation(Float_t &element) const = 0;
  void operator=(const TElementPosActionF &) { }
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixFRow_const                                                    //
//                                                                      //
// Class represents a row of a TMatrixFBase                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TMatrixFRow_const : public TObject {

protected:
  const TMatrixFBase *fMatrix;  //! the matrix I am a row of
        Int_t         fRowInd;  //  effective row index
        Int_t         fInc;     //  if ptr = @a[row,i], then ptr+inc = @a[row,i+1]
  const Float_t      *fPtr;     //! pointer to the a[row,0]

public:
  TMatrixFRow_const() { fMatrix = 0; fInc = 0; fPtr = 0; }
  TMatrixFRow_const(const TMatrixFBase &matrix,Int_t row);

  inline const TMatrixFBase *GetMatrix() const { return fMatrix; }
  inline const Float_t      *GetPtr   () const { return fPtr; }
  inline       Int_t         GetInc   () const { return fInc; }
  inline const Float_t      &operator ()(Int_t i) const { const Int_t acoln = i-fMatrix->GetColLwb();
                                                          Assert(acoln < fMatrix->GetNcols() && acoln >= 0);                    
                                                          return fPtr[acoln]; }
  inline const Float_t      &operator [](Int_t i) const { return (*(const TMatrixFRow_const *)this)(i); }

  ClassDef(TMatrixFRow_const,1)  // One row of a matrix (double precision)
};

class TMatrixFRow : public TMatrixFRow_const {

public: 
  TMatrixFRow() {}
  TMatrixFRow(TMatrixFBase &matrix,Int_t row);
  TMatrixFRow(const TMatrixFRow &mr);
   
  inline Float_t  *GetPtr() const { return const_cast<Float_t  *>(fPtr); }

  inline Float_t  &operator()(Int_t i) { const Int_t acoln = i-fMatrix->GetColLwb();
                                         Assert(acoln < fMatrix->GetNcols() && acoln >= 0);
                                         return (const_cast<Float_t  *>(fPtr))[acoln]; }
  inline Float_t  &operator[](Int_t i) { return (Float_t &)((*(TMatrixFRow *)this)(i)); }

  void operator= (Float_t  val);
  void operator+=(Float_t  val);
  void operator*=(Float_t  val);

  void operator=(const TMatrixFRow_const &r);
  void operator=(const TMatrixFRow       &r);
  void operator=(const TVectorF          &vec);

  void operator+=(const TMatrixFRow_const &r);
  void operator*=(const TMatrixFRow_const &r);

  ClassDef(TMatrixFRow,2)  // One row of a matrix (double precision)
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixFColumn_const                                                 //
//                                                                      //
// Class represents a column of a TMatrixFBase                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TMatrixFColumn_const : public TObject {

protected:
  const TMatrixFBase *fMatrix;  //! the matrix I am a column of
        Int_t         fColInd;  //  effective column index
        Int_t         fInc;     //  if ptr = @a[i,col], then ptr+inc = @a[i+1,col]
  const Float_t      *fPtr;     //! pointer to the a[0,col] column

public:
  TMatrixFColumn_const() { fMatrix = 0; fInc = 0; fPtr = 0; }
  TMatrixFColumn_const(const TMatrixFBase &matrix,Int_t col);

  inline const TMatrixFBase *GetMatrix() const { return fMatrix; }
  inline const Float_t      *GetPtr   () const { return fPtr; }
  inline       Int_t         GetInc   () const { return fInc; }
  inline const Float_t      &operator ()(Int_t i) const { const Int_t arown = i-fMatrix->GetRowLwb(); 
                                                          Assert(arown < fMatrix->GetNrows() && arown >= 0);
                                                          return fPtr[arown*fInc]; }
  inline const Float_t  &operator [](Int_t i) const { return ((*(const TMatrixFColumn_const *)this)(i)); }

  ClassDef(TMatrixFColumn_const,2)  // One column of a matrix (double precision)
};

class TMatrixFColumn : public TMatrixFColumn_const {

public:
  TMatrixFColumn() {}
  TMatrixFColumn(TMatrixFBase &matrix,Int_t col);
  TMatrixFColumn(const TMatrixFColumn &mc);

  inline Float_t  *GetPtr() const { return const_cast<Float_t  *>(fPtr); }

  inline Float_t  &operator()(Int_t i) { const Int_t arown = i-fMatrix->GetRowLwb();
                                         Assert(arown < fMatrix->GetNrows() && arown >= 0);
                                         return (const_cast<Float_t  *>(fPtr))[arown*fInc]; }
  inline Float_t  &operator[](Int_t i) { return (Float_t &)((*(TMatrixFColumn *)this)(i)); }

  void operator= (Float_t  val);
  void operator+=(Float_t  val);
  void operator*=(Float_t  val);

  void operator=(const TMatrixFColumn_const &c);
  void operator=(const TMatrixFColumn       &c);
  void operator=(const TVectorF             &vec);

  void operator+=(const TMatrixFColumn_const &c);
  void operator*=(const TMatrixFColumn_const &c);

  ClassDef(TMatrixFColumn,2)  // One column of a matrix (double precision)
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixFDiag_const                                                   //
//                                                                      //
// Class represents the diagonal of a matrix (for easy manipulation).   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TMatrixFDiag_const : public TObject {

protected:
  const TMatrixFBase *fMatrix;  //! the matrix I am the diagonal of
        Int_t         fInc;     //  if ptr=@a[i,i], then ptr+inc = @a[i+1,i+1]
        Int_t         fNdiag;   //  number of diag elems, min(nrows,ncols)
  const Float_t      *fPtr;     //! pointer to the a[0,0]

public:
  TMatrixFDiag_const() { fMatrix = 0; fInc = 0; fNdiag = 0; fPtr = 0; }
  TMatrixFDiag_const(const TMatrixFBase &matrix,Int_t dummy=0);

  inline const TMatrixFBase *GetMatrix() const { return fMatrix; }
  inline const Float_t      *GetPtr   () const { return fPtr; }
  inline       Int_t         GetInc   () const { return fInc; }
  inline const Float_t      &operator ()(Int_t i) const { Assert(i < fNdiag && i >= 0);
                                                          return fPtr[i*fInc]; }
  inline const Float_t      &operator [](Int_t i) const { return ((*(const TMatrixFDiag_const *)this)(i)); }

  Int_t GetNdiags() const { return fNdiag; }

  ClassDef(TMatrixFDiag_const,1)  // Diagonal of a matrix (double  precision)
};

class TMatrixFDiag : public TMatrixFDiag_const {

public:
  TMatrixFDiag() {}
  TMatrixFDiag(TMatrixFBase &matrix,Int_t dummy=0);
  TMatrixFDiag(const TMatrixFDiag &md);

  inline Float_t  *GetPtr() const { return const_cast<Float_t  *>(fPtr); }

  inline Float_t  &operator()(Int_t i) { Assert(i < fNdiag && i >= 0);
                                         return (const_cast<Float_t  *>(fPtr))[i*fInc]; }
  inline Float_t  &operator[](Int_t i) { return (Float_t &)((*(TMatrixFDiag *)this)(i)); }

  void operator= (Float_t  val);
  void operator+=(Float_t  val);
  void operator*=(Float_t  val);

  void operator=(const TMatrixFDiag_const &d);
  void operator=(const TMatrixFDiag       &d);
  void operator=(const TVectorF           &vec);

  void operator+=(const TMatrixFDiag_const &d);
  void operator*=(const TMatrixFDiag_const &d);

  ClassDef(TMatrixFDiag,2)  // Diagonal of a matrix (double  precision)
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixFFlat_const                                                   //
//                                                                      //
// Class represents a flat matrix (for easy manipulation).              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TMatrixFFlat_const : public TObject {

protected:
  const TMatrixFBase *fMatrix;  //! the matrix I am the diagonal of
        Int_t         fNelems;  //
  const Float_t      *fPtr;     //! pointer to the a[0,0]

public:
  TMatrixFFlat_const() { fMatrix = 0; fPtr = 0; }
  TMatrixFFlat_const(const TMatrixFBase &matrix,Int_t dummy=0);

  inline const TMatrixFBase *GetMatrix() const { return fMatrix; }
  inline const Float_t      *GetPtr   () const { return fPtr; }
  inline const Float_t      &operator ()(Int_t i) { Assert(i >=0 && i < fNelems); return GetPtr()[i]; }
  inline const Float_t      &operator [](Int_t i) { Assert(i >=0 && i < fNelems); return GetPtr()[i]; }

  ClassDef(TMatrixFFlat_const,1)  // Flat representation of a matrix
};

class TMatrixFFlat : public TMatrixFFlat_const {

public:
  TMatrixFFlat() {}
  TMatrixFFlat(TMatrixFBase &matrix,Int_t dummy=0);
  TMatrixFFlat(const TMatrixFFlat &mf);

  inline Float_t  *GetPtr() const { return const_cast<Float_t  *>(fPtr); }

  inline Float_t  &operator()(Int_t i) { Assert(i >=0 && i < fNelems);
                                         return (const_cast<Float_t  *>(fPtr))[i]; }
  inline Float_t  &operator[](Int_t i) { Assert(i >=0 && i < fNelems);
                                         return (const_cast<Float_t  *>(fPtr))[i]; }

  void operator= (Float_t  val);
  void operator+=(Float_t  val);
  void operator*=(Float_t  val);

  void operator=(const TMatrixFFlat_const &f);
  void operator=(const TMatrixFFlat       &f);
  void operator=(const TVectorF           &vec);

  void operator+=(const TMatrixFFlat_const &f);
  void operator*=(const TMatrixFFlat_const &f);

  ClassDef(TMatrixFFlat,2)  // Flat representation of a matrix
};

#endif
