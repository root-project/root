// @(#)root/matrix:$Name:  $:$Id: TMatrixDUtils.h,v 1.19 2004/01/27 06:36:45 brun Exp $
// Authors: Fons Rademakers, Eddy Offermann   Nov 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMatrixDUtils
#define ROOT_TMatrixDUtils

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Matrix utility classes.                                              //
//                                                                      //
// This file defines utility classes for the Linear Algebra Package.    //
// The following classes are defined here:                              //
//                                                                      //
// Different matrix views without copying data elements :               //
//   TMatrixDRow_const    TMatrixDRow                                   //
//   TMatrixDColumn_const TMatrixDColumn                                //
//   TMatrixDDiag_const   TMatrixDDiag                                  //
//   TMatrixDFlat_const   TMatrixDFlat                                  //
//                                                                      //
//   TElementActionD                                                    //
//   TElementPosActionD                                                 //
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
// TElementActionD                                                      //
//                                                                      //
// A class to do a specific operation on every vector or matrix element //
// (regardless of it position) as the object is being traversed.        //
// This is an abstract class. Derived classes need to implement the     //
// action function Operation().                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TElementActionD {

friend class TMatrixD;
friend class TMatrixDSym;
friend class TVectorD;

protected:
  virtual void Operation(Double_t &element) const = 0;

private:
  void operator=(const TElementActionD &) { }
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TElementPosActionD                                                   //
//                                                                      //
// A class to do a specific operation on every vector or matrix element //
// as the object is being traversed. This is an abstract class.         //
// Derived classes need to implement the action function Operation().   //
// In the action function the location of the current element is        //
// known (fI=row, fJ=columns).                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TElementPosActionD {

friend class TMatrixD;
friend class TMatrixDSym;
friend class TVectorD;

protected:
  mutable Int_t fI; // i position of element being passed to Operation()
  mutable Int_t fJ; // j position of element being passed to Operation()
  virtual void Operation(Double_t &element) const = 0;

private:
  void operator=(const TElementPosActionD &) { }
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixDRow_const                                                    //
//                                                                      //
// Class represents a row of a TMatrixDBase                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TMatrixDRow_const : public TObject {

protected:
  const TMatrixDBase *fMatrix;  //! the matrix I am a row of
        Int_t         fRowInd;  //  effective row index
        Int_t         fInc;     //  if ptr = @a[row,i], then ptr+inc = @a[row,i+1]
  const Double_t     *fPtr;     //! pointer to the a[row,0]

public:
  TMatrixDRow_const() { fMatrix = 0; fInc = 0; fPtr = 0; }
  TMatrixDRow_const(const TMatrixDBase &matrix,Int_t row);

  inline const TMatrixDBase *GetMatrix() const { return fMatrix; }
  inline const Double_t     *GetPtr   () const { return fPtr; }
  inline       Int_t         GetInc   () const { return fInc; }
  inline const Double_t     &operator ()(Int_t i) const { const Int_t acoln = i-fMatrix->GetColLwb();
                                                          Assert(acoln < fMatrix->GetNcols() && acoln >= 0);                    
                                                          return fPtr[acoln]; }
  inline const Double_t     &operator [](Int_t i) const { return (*(const TMatrixDRow_const *)this)(i); }

  ClassDef(TMatrixDRow_const,1)  // One row of a matrix (double precision)
};

class TMatrixDRow : public TMatrixDRow_const {

public: 
  TMatrixDRow() {}
  TMatrixDRow(TMatrixDBase &matrix,Int_t row);
  TMatrixDRow(const TMatrixDRow &mr);
   
  inline Double_t *GetPtr() const { return const_cast<Double_t *>(fPtr); }

  inline Double_t &operator()(Int_t i) { const Int_t acoln = i-fMatrix->GetColLwb();
                                         Assert(acoln < fMatrix->GetNcols() && acoln >= 0);
                                         return (const_cast<Double_t *>(fPtr))[acoln]; }
  inline Double_t &operator[](Int_t i) { return (Double_t&)((*(TMatrixDRow *)this)(i)); }

  void operator= (Double_t val);
  void operator+=(Double_t val);
  void operator*=(Double_t val);

  void operator=(const TMatrixDRow_const &r);
  void operator=(const TMatrixDRow       &r);
  void operator=(const TVectorD          &vec);

  void operator+=(const TMatrixDRow_const &r);
  void operator*=(const TMatrixDRow_const &r);

  ClassDef(TMatrixDRow,2)  // One row of a matrix (double precision)
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixDColumn_const                                                 //
//                                                                      //
// Class represents a column of a TMatrixDBase                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TMatrixDColumn_const : public TObject {

protected:
  const TMatrixDBase *fMatrix;  //! the matrix I am a column of
        Int_t         fColInd;  //  effective column index
        Int_t         fInc;     //  if ptr = @a[i,col], then ptr+inc = @a[i+1,col]
  const Double_t     *fPtr;     //! pointer to the a[0,col] column

public:
  TMatrixDColumn_const() { fMatrix = 0; fInc = 0; fPtr = 0; }
  TMatrixDColumn_const(const TMatrixDBase &matrix,Int_t col);

  inline const TMatrixDBase *GetMatrix() const { return fMatrix; }
  inline const Double_t     *GetPtr   () const { return fPtr; }
  inline       Int_t         GetInc   () const { return fInc; }
  inline const Double_t     &operator ()(Int_t i) const { const Int_t arown = i-fMatrix->GetRowLwb(); 
                                                          Assert(arown < fMatrix->GetNrows() && arown >= 0);
                                                          return fPtr[arown*fInc]; }
  inline const Double_t &operator [](Int_t i) const { return ((*(const TMatrixDColumn_const *)this)(i)); }

  ClassDef(TMatrixDColumn_const,2)  // One column of a matrix (double precision)
};

class TMatrixDColumn : public TMatrixDColumn_const {

public:
  TMatrixDColumn() {}
  TMatrixDColumn(TMatrixDBase &matrix,Int_t col);
  TMatrixDColumn(const TMatrixDColumn &mc);

  inline Double_t *GetPtr() const { return const_cast<Double_t *>(fPtr); }

  inline Double_t &operator()(Int_t i) { const Int_t arown = i-fMatrix->GetRowLwb();
                                         Assert(arown < fMatrix->GetNrows() && arown >= 0);
                                         return (const_cast<Double_t *>(fPtr))[arown*fInc]; }
  inline Double_t &operator[](Int_t i) { return (Double_t&)((*(TMatrixDColumn *)this)(i)); }

  void operator= (Double_t val);
  void operator+=(Double_t val);
  void operator*=(Double_t val);

  void operator=(const TMatrixDColumn_const &c);
  void operator=(const TMatrixDColumn       &c);
  void operator=(const TVectorD             &vec);

  void operator+=(const TMatrixDColumn_const &c);
  void operator*=(const TMatrixDColumn_const &c);

  ClassDef(TMatrixDColumn,2)  // One column of a matrix (double precision)
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixDDiag_const                                                   //
//                                                                      //
// Class represents the diagonal of a matrix (for easy manipulation).   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TMatrixDDiag_const : public TObject {

protected:
  const TMatrixDBase *fMatrix;  //! the matrix I am the diagonal of
        Int_t         fInc;     //  if ptr=@a[i,i], then ptr+inc = @a[i+1,i+1]
        Int_t         fNdiag;   //  number of diag elems, min(nrows,ncols)
  const Double_t     *fPtr;     //! pointer to the a[0,0]

public:
  TMatrixDDiag_const() { fMatrix = 0; fInc = 0; fNdiag = 0; fPtr = 0; }
  TMatrixDDiag_const(const TMatrixDBase &matrix,Int_t dummy=0);

  inline const TMatrixDBase *GetMatrix() const { return fMatrix; }
  inline const Double_t     *GetPtr   () const { return fPtr; }
  inline       Int_t         GetInc   () const { return fInc; }
  inline const Double_t     &operator ()(Int_t i) const { Assert(i < fNdiag && i >= 0);
                                                          return fPtr[i*fInc]; }
  inline const Double_t     &operator [](Int_t i) const { return ((*(const TMatrixDDiag_const *)this)(i)); }

  Int_t GetNdiags() const { return fNdiag; }

  ClassDef(TMatrixDDiag_const,1)  // Diagonal of a matrix (double  precision)
};

class TMatrixDDiag : public TMatrixDDiag_const {

public:
  TMatrixDDiag() {}
  TMatrixDDiag(TMatrixDBase &matrix,Int_t dummy=0);
  TMatrixDDiag(const TMatrixDDiag &md);

  inline Double_t *GetPtr() const { return const_cast<Double_t *>(fPtr); }

  inline Double_t &operator()(Int_t i) { Assert(i < fNdiag && i >= 0);
                                         return (const_cast<Double_t *>(fPtr))[i*fInc]; }
  inline Double_t &operator[](Int_t i) { return (Double_t&)((*(TMatrixDDiag *)this)(i)); }

  void operator= (Double_t val);
  void operator+=(Double_t val);
  void operator*=(Double_t val);

  void operator=(const TMatrixDDiag_const &d);
  void operator=(const TMatrixDDiag       &d);
  void operator=(const TVectorD           &vec);

  void operator+=(const TMatrixDDiag_const &d);
  void operator*=(const TMatrixDDiag_const &d);

  ClassDef(TMatrixDDiag,2)  // Diagonal of a matrix (double  precision)
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixDFlat_const                                                   //
//                                                                      //
// Class represents a flat matrix (for easy manipulation).              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TMatrixDFlat_const : public TObject {

protected:
  const TMatrixDBase *fMatrix;  //! the matrix I am the diagonal of
        Int_t         fNelems;  //
  const Double_t     *fPtr;     //! pointer to the a[0,0]

public:
  TMatrixDFlat_const() { fMatrix = 0; fPtr = 0; }
  TMatrixDFlat_const(const TMatrixDBase &matrix,Int_t dummy=0);

  inline const TMatrixDBase *GetMatrix() const { return fMatrix; }
  inline const Double_t     *GetPtr   () const { return fPtr; }
  inline const Double_t     &operator ()(Int_t i) { Assert(i >=0 && i < fNelems); return GetPtr()[i]; }
  inline const Double_t     &operator [](Int_t i) { Assert(i >=0 && i < fNelems); return GetPtr()[i]; }

  ClassDef(TMatrixDFlat_const,1)  // Flat representation of a matrix
};

class TMatrixDFlat : public TMatrixDFlat_const {

public:
  TMatrixDFlat() {}
  TMatrixDFlat(TMatrixDBase &matrix,Int_t dummy=0);
  TMatrixDFlat(const TMatrixDFlat &mf);

  inline Double_t *GetPtr() const { return const_cast<Double_t *>(fPtr); }

  inline Double_t &operator()(Int_t i) { Assert(i >=0 && i < fNelems);
                                         return (const_cast<Double_t *>(fPtr))[i]; }
  inline Double_t &operator[](Int_t i) { Assert(i >=0 && i < fNelems);
                                         return (const_cast<Double_t *>(fPtr))[i]; }

  void operator= (Double_t val);
  void operator+=(Double_t val);
  void operator*=(Double_t val);

  void operator=(const TMatrixDFlat_const &f);
  void operator=(const TMatrixDFlat       &f);
  void operator=(const TVectorD           &vec);

  void operator+=(const TMatrixDFlat_const &f);
  void operator*=(const TMatrixDFlat_const &f);

  ClassDef(TMatrixDFlat,2)  // Flat representation of a matrix
};

#endif
