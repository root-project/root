// @(#)root/matrix:$Name:  $:$Id: TMatrixDUtils.h,v 1.25 2004/05/12 18:24:58 brun Exp $
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
//   TMatrixDRow_const        TMatrixDRow                               //
//   TMatrixDColumn_const     TMatrixDColumn                            //
//   TMatrixDDiag_const       TMatrixDDiag                              //
//   TMatrixDFlat_const       TMatrixDFlat                              //
//   TMatrixDSparseRow_const  TMatrixDSparseRow                         //
//   TMatrixDSparseDiag_const TMatrixDSparseDiag                        //
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

friend class TMatrixDBase;
friend class TMatrixD;
friend class TMatrixDSym;
friend class TMatrixDSparse;
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

friend class TMatrixDBase;
friend class TMatrixD;
friend class TMatrixDSym;
friend class TMatrixDSparse;
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

class TMatrixDRow_const {

protected:
  const TMatrixDBase *fMatrix;  //  the matrix I am a row of
        Int_t         fRowInd;  //  effective row index
        Int_t         fInc;     //  if ptr = @a[row,i], then ptr+inc = @a[row,i+1]
  const Double_t     *fPtr;     //  pointer to the a[row,0]

public:
  TMatrixDRow_const() { fMatrix = 0; fInc = 0; fPtr = 0; }
  TMatrixDRow_const(const TMatrixD    &matrix,Int_t row);
  TMatrixDRow_const(const TMatrixDSym &matrix,Int_t row);

  inline const TMatrixDBase *GetMatrix() const { return fMatrix; }
  inline const Double_t     *GetPtr   () const { return fPtr; }
  inline       Int_t         GetInc   () const { return fInc; }
  inline const Double_t     &operator ()(Int_t i) const { const Int_t acoln = i-fMatrix->GetColLwb();
                                                          Assert(acoln < fMatrix->GetNcols() && acoln >= 0);
                                                          return fPtr[acoln]; }
  inline const Double_t     &operator [](Int_t i) const { return (*(const TMatrixDRow_const *)this)(i); }

  ClassDef(TMatrixDRow_const,0)  // One row of a dense matrix (double precision)
};

class TMatrixDRow : public TMatrixDRow_const {

public:
  TMatrixDRow() {}
  TMatrixDRow(TMatrixD    &matrix,Int_t row);
  TMatrixDRow(TMatrixDSym &matrix,Int_t row);
  TMatrixDRow(const TMatrixDRow &mr);

  inline Double_t *GetPtr() const { return const_cast<Double_t *>(fPtr); }

  inline const Double_t &operator()(Int_t i) const { const Int_t acoln = i-fMatrix->GetColLwb();
                                                     Assert(acoln < fMatrix->GetNcols() && acoln >= 0);
                                                     return fPtr[acoln]; }
  inline       Double_t &operator()(Int_t i)       { const Int_t acoln = i-fMatrix->GetColLwb();
                                                     Assert(acoln < fMatrix->GetNcols() && acoln >= 0);
                                                     return (const_cast<Double_t *>(fPtr))[acoln]; }
  inline const Double_t &operator[](Int_t i) const { return (*(const TMatrixDRow *)this)(i); }
  inline       Double_t &operator[](Int_t i)       { return (*(      TMatrixDRow *)this)(i); }

  void operator= (Double_t val);
  void operator+=(Double_t val);
  void operator*=(Double_t val);

  void operator=(const TMatrixDRow_const &r);
  void operator=(const TMatrixDRow       &r);
  void operator=(const TVectorD          &vec);

  void operator+=(const TMatrixDRow_const &r);
  void operator*=(const TMatrixDRow_const &r);

  ClassDef(TMatrixDRow,0)  // One row of a dense matrix (double precision)
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixDColumn_const                                                 //
//                                                                      //
// Class represents a column of a TMatrixDBase                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TMatrixDColumn_const {

protected:
  const TMatrixDBase *fMatrix;  //  the matrix I am a column of
        Int_t         fColInd;  //  effective column index
        Int_t         fInc;     //  if ptr = @a[i,col], then ptr+inc = @a[i+1,col]
  const Double_t     *fPtr;     //  pointer to the a[0,col] column

public:
  TMatrixDColumn_const() { fMatrix = 0; fInc = 0; fPtr = 0; }
  TMatrixDColumn_const(const TMatrixD    &matrix,Int_t col);
  TMatrixDColumn_const(const TMatrixDSym &matrix,Int_t col);

  inline const TMatrixDBase *GetMatrix() const { return fMatrix; }
  inline const Double_t     *GetPtr   () const { return fPtr; }
  inline       Int_t         GetInc   () const { return fInc; }
  inline const Double_t     &operator ()(Int_t i) const { const Int_t arown = i-fMatrix->GetRowLwb();
                                                          Assert(arown < fMatrix->GetNrows() && arown >= 0);
                                                          return fPtr[arown*fInc]; }
  inline const Double_t     &operator [](Int_t i) const { return (*(const TMatrixDColumn_const *)this)(i); }

  ClassDef(TMatrixDColumn_const,0)  // One column of a dense matrix (double precision)
};

class TMatrixDColumn : public TMatrixDColumn_const {

public:
  TMatrixDColumn() {}
  TMatrixDColumn(TMatrixD    &matrix,Int_t col);
  TMatrixDColumn(TMatrixDSym &matrix,Int_t col);
  TMatrixDColumn(const TMatrixDColumn &mc);

  inline Double_t *GetPtr() const { return const_cast<Double_t *>(fPtr); }

  inline const Double_t &operator()(Int_t i) const { const Int_t arown = i-fMatrix->GetRowLwb();
                                                     Assert(arown < fMatrix->GetNrows() && arown >= 0);
                                                     return fPtr[arown]; }
  inline       Double_t &operator()(Int_t i)       { const Int_t arown = i-fMatrix->GetRowLwb();
                                                     Assert(arown < fMatrix->GetNrows() && arown >= 0);
                                                     return (const_cast<Double_t *>(fPtr))[arown*fInc]; }
  inline const Double_t &operator[](Int_t i) const { return (*(const TMatrixDColumn *)this)(i); }
  inline       Double_t &operator[](Int_t i)       { return (*(      TMatrixDColumn *)this)(i); }

  void operator= (Double_t val);
  void operator+=(Double_t val);
  void operator*=(Double_t val);

  void operator=(const TMatrixDColumn_const &c);
  void operator=(const TMatrixDColumn       &c);
  void operator=(const TVectorD             &vec);

  void operator+=(const TMatrixDColumn_const &c);
  void operator*=(const TMatrixDColumn_const &c);

  ClassDef(TMatrixDColumn,0)  // One column of a dense matrix (double precision)
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixDDiag_const                                                   //
//                                                                      //
// Class represents the diagonal of a matrix (for easy manipulation).   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TMatrixDDiag_const {

protected:
  const TMatrixDBase *fMatrix;  //  the matrix I am the diagonal of
        Int_t         fInc;     //  if ptr=@a[i,i], then ptr+inc = @a[i+1,i+1]
        Int_t         fNdiag;   //  number of diag elems, min(nrows,ncols)
  const Double_t     *fPtr;     //  pointer to the a[0,0]

public:
  TMatrixDDiag_const() { fMatrix = 0; fInc = 0; fNdiag = 0; fPtr = 0; }
  TMatrixDDiag_const(const TMatrixD    &matrix);
  TMatrixDDiag_const(const TMatrixDSym &matrix);

  inline const TMatrixDBase *GetMatrix() const { return fMatrix; }
  inline const Double_t     *GetPtr   () const { return fPtr; }
  inline       Int_t         GetInc   () const { return fInc; }
  inline const Double_t     &operator ()(Int_t i) const { Assert(i < fNdiag && i >= 0); return fPtr[i*fInc]; }
  inline const Double_t     &operator [](Int_t i) const { return (*(const TMatrixDDiag_const *)this)(i); }

  Int_t GetNdiags() const { return fNdiag; }

  ClassDef(TMatrixDDiag_const,0)  // Diagonal of a dense matrix (double  precision)
};

class TMatrixDDiag : public TMatrixDDiag_const {

public:
  TMatrixDDiag() {}
  TMatrixDDiag(TMatrixD    &matrix);
  TMatrixDDiag(TMatrixDSym &matrix);
  TMatrixDDiag(const TMatrixDDiag &md);

  inline Double_t *GetPtr() const { return const_cast<Double_t *>(fPtr); }

  inline const Double_t &operator()(Int_t i) const { Assert(i < fNdiag && i >= 0); return fPtr[i*fInc]; }
  inline       Double_t &operator()(Int_t i)       { Assert(i < fNdiag && i >= 0);
                                                     return (const_cast<Double_t *>(fPtr))[i*fInc]; }
  inline const Double_t &operator[](Int_t i) const { return (*(const TMatrixDDiag *)this)(i); }
  inline       Double_t &operator[](Int_t i)       { return (*(      TMatrixDDiag *)this)(i); }

  void operator= (Double_t val);
  void operator+=(Double_t val);
  void operator*=(Double_t val);

  void operator=(const TMatrixDDiag_const &d);
  void operator=(const TMatrixDDiag       &d);
  void operator=(const TVectorD           &vec);

  void operator+=(const TMatrixDDiag_const &d);
  void operator*=(const TMatrixDDiag_const &d);

  ClassDef(TMatrixDDiag,0)  // Diagonal of a dense matrix (double  precision)
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixDFlat_const                                                   //
//                                                                      //
// Class represents a flat matrix (for easy manipulation).              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TMatrixDFlat_const {

protected:
  const TMatrixDBase *fMatrix;  //  the matrix I am the diagonal of
        Int_t         fNelems;  //
  const Double_t     *fPtr;     //  pointer to the a[0,0]

public:
  TMatrixDFlat_const() { fMatrix = 0; fPtr = 0; }
  TMatrixDFlat_const(const TMatrixD    &matrix);
  TMatrixDFlat_const(const TMatrixDSym &matrix);

  inline const TMatrixDBase *GetMatrix() const { return fMatrix; }
  inline const Double_t     *GetPtr   () const { return fPtr; }
  inline const Double_t     &operator ()(Int_t i) const { Assert(i >=0 && i < fNelems); return fPtr[i]; }
  inline const Double_t     &operator [](Int_t i) const { return (*(const TMatrixDFlat_const *)this)(i); }

  ClassDef(TMatrixDFlat_const,0)  // Flat representation of a dense matrix
};

class TMatrixDFlat : public TMatrixDFlat_const {

public:
  TMatrixDFlat() {}
  TMatrixDFlat(TMatrixD    &matrix);
  TMatrixDFlat(TMatrixDSym &matrix);
  TMatrixDFlat(const TMatrixDFlat &mf);

  inline Double_t *GetPtr() const { return const_cast<Double_t *>(fPtr); }

  inline const Double_t &operator()(Int_t i) const { Assert(i >=0 && i < fNelems); return fPtr[i]; }
  inline       Double_t &operator()(Int_t i)       { Assert(i >=0 && i < fNelems);
                                                     return (const_cast<Double_t *>(fPtr))[i]; }
  inline const Double_t &operator[](Int_t i) const { return (*(const TMatrixDFlat *)this)(i); }
  inline       Double_t &operator[](Int_t i)       { return (*(      TMatrixDFlat *)this)[i]; }

  void operator= (Double_t val);
  void operator+=(Double_t val);
  void operator*=(Double_t val);

  void operator=(const TMatrixDFlat_const &f);
  void operator=(const TMatrixDFlat       &f);
  void operator=(const TVectorD           &vec);

  void operator+=(const TMatrixDFlat_const &f);
  void operator*=(const TMatrixDFlat_const &f);

  ClassDef(TMatrixDFlat,0)  // Flat representation of a dense matrix
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixDSparseRow_const                                              //
//                                                                      //
// Class represents a row of a TMatrixDSparse                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TMatrixDSparse;

class TMatrixDSparseRow_const {

protected:
  const TMatrixDBase *fMatrix;  // the matrix I am a row of
        Int_t         fRowInd;  // effective row index
        Int_t         fNindex;  // index range
  const Int_t        *fColPtr;  // column index pointer
  const Double_t     *fDataPtr; // data pointer

public:
  TMatrixDSparseRow_const() { fMatrix = 0; fRowInd = 0; fNindex = 0; fColPtr = 0; fDataPtr = 0; }
  TMatrixDSparseRow_const(const TMatrixDSparse &matrix,Int_t row);

  inline const TMatrixDBase *GetMatrix  () const { return fMatrix; }
  inline const Double_t     *GetDataPtr () const { return fDataPtr; }
  inline const Int_t        *GetColPtr  () const { return fColPtr; }
  inline       Int_t         GetRowIndex() const { return fRowInd; }
  inline       Int_t         GetNindex  () const { return fNindex; }

  inline Double_t operator()(Int_t i) const { const Int_t acoln = i-fMatrix->GetColLwb();
                                              Assert(acoln < fMatrix->GetNcols() && acoln >= 0);
                                              const Int_t index = TMath::BinarySearch(fNindex,fColPtr,acoln);
                                              if (index >= 0 && fColPtr[index] == acoln) return fDataPtr[index];
                                              else                                       return 0.0; }
  inline Double_t operator[](Int_t i) const { return (*(const TMatrixDSparseRow_const *)this)(i); }

  ClassDef(TMatrixDSparseRow_const,0)  // One row of a sparse matrix (double precision)
};

class TMatrixDSparseRow : public TMatrixDSparseRow_const {

public:
  TMatrixDSparseRow() {}
  TMatrixDSparseRow(TMatrixDSparse &matrix,Int_t row);
  TMatrixDSparseRow(const TMatrixDSparseRow &mr);

  inline Double_t *GetDataPtr() const { return const_cast<Double_t *>(fDataPtr); }

  inline Double_t  operator()(Int_t i) const { const Int_t acoln = i-fMatrix->GetColLwb();
                                               Assert(acoln < fMatrix->GetNcols() && acoln >= 0);
                                               const Int_t index = TMath::BinarySearch(fNindex,fColPtr,acoln);
                                               if (index >= 0 && fColPtr[index] == acoln) return fDataPtr[index];
                                               else                                       return 0.0; }
         Double_t &operator()(Int_t i);
  inline Double_t  operator[](Int_t i) const { return (*(const TMatrixDSparseRow *)this)(i); }
  inline Double_t &operator[](Int_t i)       { return (*(TMatrixDSparseRow *)this)(i); }

  void operator= (Double_t val);
  void operator+=(Double_t val);
  void operator*=(Double_t val);

  void operator=(const TMatrixDSparseRow_const &r);
  void operator=(const TMatrixDSparseRow       &r);
  void operator=(const TVectorD                &vec);

  void operator+=(const TMatrixDSparseRow_const &r);
  void operator*=(const TMatrixDSparseRow_const &r);

  ClassDef(TMatrixDSparseRow,0)  // One row of a sparse matrix (double precision)
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMatrixDSparseDiag_const                                             //
//                                                                      //
// Class represents the diagonal of a matrix (for easy manipulation).   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TMatrixDSparseDiag_const {

protected:
  const TMatrixDBase *fMatrix;  //  the matrix I am the diagonal of
        Int_t         fNdiag;   //  number of diag elems, min(nrows,ncols)
  const Double_t     *fDataPtr; //  data pointer

public:
  TMatrixDSparseDiag_const() { fMatrix = 0; fNdiag = 0; fDataPtr = 0; }
  TMatrixDSparseDiag_const(const TMatrixDSparse &matrix);

  inline const TMatrixDBase *GetMatrix () const { return fMatrix; }
  inline const Double_t     *GetDataPtr() const { return fDataPtr; }
  inline       Int_t         GetNdiags () const { return fNdiag; }

  inline Double_t operator ()(Int_t i) const { Assert(i < fNdiag && i >= 0);
                                               const Int_t    * const pR = fMatrix->GetRowIndexArray();
                                               const Int_t    * const pC = fMatrix->GetColIndexArray();
                                               const Double_t * const pD = fMatrix->GetMatrixArray();
                                               const Int_t sIndex = pR[i];
                                               const Int_t eIndex = pR[i+1];
                                               const Int_t index = TMath::BinarySearch(eIndex-sIndex,pC+sIndex,i)+sIndex;
                                               if (index >= sIndex && pC[index] == i) return pD[index];
                                               else                                   return 0.0; }

  inline Double_t operator [](Int_t i) const { return (*(const TMatrixDSparseRow_const *)this)(i); }

  ClassDef(TMatrixDSparseDiag_const,0)  // Diagonal of a sparse matrix (double  precision)
};

class TMatrixDSparseDiag : public TMatrixDSparseDiag_const {

public:
  TMatrixDSparseDiag() {}
  TMatrixDSparseDiag(TMatrixDSparse &matrix);
  TMatrixDSparseDiag(const TMatrixDSparseDiag &md);

  inline Double_t *GetDataPtr() const { return const_cast<Double_t *>(fDataPtr); }
  
  inline       Double_t  operator()(Int_t i) const { Assert(i < fNdiag && i >= 0);
                                                     const Int_t    * const pR = fMatrix->GetRowIndexArray();
                                                     const Int_t    * const pC = fMatrix->GetColIndexArray();
                                                     const Double_t * const pD = fMatrix->GetMatrixArray();
                                                     const Int_t sIndex = pR[i];
                                                     const Int_t eIndex = pR[i+1];
                                                     const Int_t index = TMath::BinarySearch(eIndex-sIndex,pC+sIndex,i)+sIndex;
                                                     if (index >= sIndex && pC[index] == i) return pD[index];
                                                     else                                   return 0.0; }
               Double_t &operator()(Int_t i);
  inline       Double_t  operator[](Int_t i) const { return (*(const TMatrixDSparseDiag *)this)(i); }
  inline       Double_t &operator[](Int_t i)       { return (Double_t&)((*(TMatrixDSparseDiag *)this)(i)); }

  void operator= (Double_t val);
  void operator+=(Double_t val);
  void operator*=(Double_t val);

  void operator=(const TMatrixDSparseDiag_const &d);
  void operator=(const TMatrixDSparseDiag       &d);
  void operator=(const TVectorD                 &vec);

  void operator+=(const TMatrixDSparseDiag_const &d);
  void operator*=(const TMatrixDSparseDiag_const &d);

  ClassDef(TMatrixDSparseDiag,0)  // Diagonal of a dense matrix (double  precision)
};

Double_t Drand(Double_t &ix);
#endif
