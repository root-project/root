// @(#)root/matrix:$Name:  $:$Id: TMatrix.h,v 1.24 2003/08/18 16:40:33 rdm Exp $
// Authors: Oleg E. Kiselyov, Fons Rademakers   03/11/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMatrix
#define ROOT_TMatrix


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Linear Algebra Package                                               //
//                                                                      //
// The present package implements all the basic algorithms dealing      //
// with vectors, matrices, matrix columns, rows, diagonals, etc.        //
//                                                                      //
// Matrix elements are arranged in memory in a COLUMN-wise              //
// fashion (in FORTRAN's spirit). In fact, it makes it very easy to     //
// feed the matrices to FORTRAN procedures, which implement more        //
// elaborate algorithms.                                                //
//                                                                      //
// Unless otherwise specified, matrix and vector indices always start   //
// with 0, spanning up to the specified limit-1.                        //
//                                                                      //
// The present package provides all facilities to completely AVOID      //
// returning matrices. Use "TMatrix A(TMatrix::kTransposed,B);" and     //
// other fancy constructors as much as possible. If one really needs    //
// to return a matrix, return a TLazyMatrix object instead. The         //
// conversion is completely transparent to the end user, e.g.           //
// "TMatrix m = THaarMatrix(5);" and _is_ efficient.                    //
//                                                                      //
// For usage examples see $ROOTSYS/test/vmatrix.cxx and vvector.cxx     //
// and also:                                                            //
// http://root.cern.ch/root/html/TMatrix.html#TMatrix:description       //
//                                                                      //
// The implementation is based on original code by                      //
// Oleg E. Kiselyov (oleg@pobox.com).                                   //
// Several additions/optimisations by  Eddy Offermann <eddy@rentec.com> //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TVector
#include "TVector.h"
#endif

class TMatrix;
class TLazyMatrix;
class TMatrixRow;
class TMatrixColumn;
class TMatrixDiag;
class TMatrixFlat;
class TMatrixPivoting;

TMatrix &operator+=(TMatrix &target, const TMatrix &source);
TMatrix &operator-=(TMatrix &target, const TMatrix &source);
TMatrix  operator+(const TMatrix &source1, const TMatrix &source2);
TMatrix  operator-(const TMatrix &source1, const TMatrix &source2);
TMatrix  operator*(const TMatrix &source1, const TMatrix &source2);
TMatrix &Add(TMatrix &target, Double_t scalar, const TMatrix &source);
TMatrix &ElementMult(TMatrix &target, const TMatrix &source);
TMatrix &ElementDiv(TMatrix &target, const TMatrix &source);
Bool_t   operator==(const TMatrix &im1, const TMatrix &im2);
void     Compare(const TMatrix &im1, const TMatrix &im2);
Bool_t   AreCompatible(const TMatrix &im1, const TMatrix &im2);
Double_t E2Norm(const TMatrix &m1, const TMatrix &m2);


class TMatrix : public TObject {

friend class TVector;
friend class TMatrixRow;
friend class TMatrixColumn;
friend class TMatrixDiag;
friend class TMatrixFlat;
friend class TMatrixPivoting;

protected:
   Int_t     fNrows;            // number of rows
   Int_t     fNcols;            // number of columns
   Int_t     fNelems;           // number of elements in matrix
   Int_t     fRowLwb;           // lower bound of the row index
   Int_t     fColLwb;           // lower bound of the col index
   Real_t   *fElements;	        //[fNelems] elements themselves
   Real_t  **fIndex;            //! index[i] = &matrix(0,i) (col index)

   void Allocate(Int_t nrows, Int_t ncols, Int_t row_lwb = 0, Int_t col_lwb = 0);
   void Invalidate() { fNrows = fNcols = fNelems = -1; fElements = 0; fIndex = 0; }

   static Int_t Pdcholesky(const Real_t *a, Real_t *u, const Int_t n);
   static void  MakeTridiagonal(TMatrix &a,TVector &d,TVector &e);
   static void  MakeEigenVectors(TVector &d,TVector &e,TMatrix &z);
   static void  EigenSort(TMatrix &eigenVectors,TVector &eigenValues);

   // Elementary constructors
   void Transpose(const TMatrix &m);
   void Invert(const TMatrix &m);
   void InvertPosDef(const TMatrix &m);
   void AMultB(const TMatrix &a, const TMatrix &b);
   void AtMultB(const TMatrix &a, const TMatrix &b);

   friend void MakeHaarMatrix(TMatrix &m);
   friend void MakeHilbertMatrix(TMatrix &m);

public:
   enum EMatrixCreatorsOp1 { kZero, kUnit, kTransposed, kInverted, kInvertedPosDef };
   enum EMatrixCreatorsOp2 { kMult, kTransposeMult, kInvMult, kInvPosDefMult, kAtBA };

   TMatrix() { Invalidate(); }
   TMatrix(Int_t nrows, Int_t ncols);
   TMatrix(Int_t row_lwb, Int_t row_upb, Int_t col_lwb, Int_t col_upb);
   TMatrix(Int_t nrows, Int_t ncols, const Real_t *elements, Option_t *option="");
   TMatrix(Int_t row_lwb, Int_t row_upb, Int_t col_lwb, Int_t col_upb,
           const Real_t *elements, Option_t *option="");
   TMatrix(const TMatrix &another);
   TMatrix(EMatrixCreatorsOp1 op, const TMatrix &prototype);
   TMatrix(const TMatrix &a, EMatrixCreatorsOp2 op, const TMatrix &b);
   TMatrix(const TLazyMatrix &lazy_constructor);

   virtual ~TMatrix();

   void Clear(Option_t *option="");
   void Draw(Option_t *option="");  // *MENU*
   void ResizeTo(Int_t nrows, Int_t ncols);
   void ResizeTo(Int_t row_lwb, Int_t row_upb, Int_t col_lwb, Int_t col_upb);
   void ResizeTo(const TMatrix &m);

   Bool_t IsValid() const;
   Bool_t IsSymmetric() const;

   Int_t    GetRowLwb()     const { return fRowLwb; }
   Int_t    GetRowUpb()     const { return fNrows+fRowLwb-1; }
   Int_t    GetNrows()      const { return fNrows; }
   Int_t    GetColLwb()     const { return fColLwb; }
   Int_t    GetColUpb()     const { return fNcols+fColLwb-1; }
   Int_t    GetNcols()      const { return fNcols; }
   Int_t    GetNoElements() const { return fNelems; }
   const Real_t *GetElements() const { return fElements; }
         Real_t *GetElements()       { return fElements; }
   void     GetElements(Real_t *elements, Option_t *option="") const;
   void     SetElements(const Real_t *elements, Option_t *option="");

   const Real_t &operator()(Int_t rown, Int_t coln) const;
   Real_t &operator()(Int_t rown, Int_t coln);
   const TMatrixRow operator[](Int_t rown) const;
   TMatrixRow operator[](Int_t rown);

   TMatrix &operator=(const TMatrix &source);
   TMatrix &operator=(const TLazyMatrix &source);
   TMatrix &operator=(Real_t val);
   TMatrix &operator-=(Double_t val);
   TMatrix &operator+=(Double_t val);
   TMatrix &operator*=(Double_t val);

   Bool_t operator==(Real_t val) const;
   Bool_t operator!=(Real_t val) const;
   Bool_t operator<(Real_t val) const;
   Bool_t operator<=(Real_t val) const;
   Bool_t operator>(Real_t val) const;
   Bool_t operator>=(Real_t val) const;

   TMatrix &Zero();
   TMatrix &Abs();
   TMatrix &Sqr();
   TMatrix &Sqrt();

   TMatrix &Apply(const TElementAction &action);
   TMatrix &Apply(const TElementPosAction &action);

   TMatrix &Invert(Double_t *determ_ptr = 0);
   TMatrix &InvertPosDef();

   const TMatrix EigenVectors(TVector &eigenValues) const;

   TMatrix &MakeSymmetric();
   TMatrix &UnitMatrix();

   TMatrix &operator*=(const TMatrix &source);
   TMatrix &operator*=(const TMatrixDiag &diag);
   TMatrix &operator/=(const TMatrixDiag &diag);
   TMatrix &operator*=(const TMatrixRow &row);
   TMatrix &operator/=(const TMatrixRow &row);
   TMatrix &operator*=(const TMatrixColumn &col);
   TMatrix &operator/=(const TMatrixColumn &col);

   void Mult(const TMatrix &a, const TMatrix &b);

   Double_t RowNorm() const;
   Double_t NormInf() const { return RowNorm(); }
   Double_t ColNorm() const;
   Double_t Norm1() const { return ColNorm(); }
   Double_t E2Norm() const;
   TMatrix &NormByDiag(const TVector &v, Option_t *option="D");
   TMatrix &NormByColumn(const TVector &v, Option_t *option="D");
   TMatrix &NormByRow(const TVector &v, Option_t *option="D");

   Double_t Determinant() const;

   void Print(Option_t *option="") const;  // *MENU*

   friend TMatrix &operator+=(TMatrix &target, const TMatrix &source);
   friend TMatrix &operator-=(TMatrix &target, const TMatrix &source);
   friend TMatrix &Add(TMatrix &target, Double_t scalar, const TMatrix &source);
   friend TMatrix &ElementMult(TMatrix &target, const TMatrix &source);
   friend TMatrix &ElementDiv(TMatrix &target, const TMatrix &source);

   friend TMatrix  operator+(const TMatrix &source1, const TMatrix &source2);
   friend TMatrix  operator-(const TMatrix &source1, const TMatrix &source2);
   friend TMatrix  operator*(const TMatrix &source1, const TMatrix &source2);

   friend Bool_t operator==(const TMatrix &im1, const TMatrix &im2);
   friend void Compare(const TMatrix &im1, const TMatrix &im2);
   friend Bool_t AreCompatible(const TMatrix &im1, const TMatrix &im2);
   friend Double_t E2Norm(const TMatrix &m1, const TMatrix &m2);

   ClassDef(TMatrix,2)  // Matrix class
};


// Service functions (useful in the verification code).
// They print some detail info if the validation condition fails
void VerifyElementValue(const TMatrix &m, Real_t val);
void VerifyMatrixIdentity(const TMatrix &m1, const TMatrix &m2);


inline Bool_t TMatrix::IsValid() const
   { if (fNrows == -1) return kFALSE; return kTRUE; }


#ifndef ROOT_TMatrixUtils
#include "TMatrixUtils.h"
#endif

#endif
