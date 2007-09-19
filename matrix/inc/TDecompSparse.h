// @(#)root/matrix:$Id$
// Authors: Fons Rademakers, Eddy Offermann   Apr 2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDecompSparse
#define ROOT_TDecompSparse

///////////////////////////////////////////////////////////////////////////
//                                                                       //
// Sparse Decomposition class                                            //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TDecompBase
#include "TDecompBase.h"
#endif
#ifndef ROOT_TMatrixDSparse
#include "TMatrixDSparse.h"
#endif    
#ifndef ROOT_TArrayD
#include "TArrayD.h"
#endif
#ifndef ROOT_TArrayI
#include "TArrayI.h"
#endif

// globals
const Double_t kInitTreatAsZero         = 1.0e-12;
const Double_t kInitThresholdPivoting   = 1.0e-8;
const Double_t kInitPrecision           = 1.0e-7;

// the Threshold Pivoting parameter may need to be increased during
// the algorithm if poor precision is obtained from the linear
// solves.  kThresholdPivoting indicates the largest value we are
// willing to tolerate.

const Double_t kThresholdPivotingMax    = 1.0e-2;

// the factor in the range (1,inf) by which kThresholdPivoting is
// increased when it is found to be inadequate.

const Double_t kThresholdPivotingFactor = 10.0;

class TDecompSparse : public TDecompBase
{
protected :

   Int_t     fVerbose;

   Int_t     fIcntl[31]; // integer control numbers
   Double_t  fCntl[6];   // float control numbers
   Int_t     fInfo[21];  // array used for communication between programs

   Double_t  fPrecision; // precision we demand from the linear system solver. If it isn't
                         // attained on the first solve, we use iterative refinement and
                         // possibly refactorization with a higher value of kThresholdPivoting.

   TArrayI   fIkeep;     // pivot sequence and temporary storage information
   TArrayI   fIw;
   TArrayI   fIw1;
   TArrayI   fIw2;
   Int_t     fNsteps;
   Int_t     fMaxfrt;
   TArrayD   fW;         // temporary storage for the factorization

   Double_t  fIPessimism; // amounts by which to increase allocated factorization space when
   Double_t  fRPessimism; // inadequate space is detected. fIPessimism is for array "fIw",
                          // fRPessimism is for the array "fact".

   TMatrixDSparse fA; // original matrix; needed for the iterative solving procedure
   Int_t          fNrows;
   Int_t          fNnonZeros;
   TArrayD        fFact; // size of fFact array; may be increased during the numerical factorization
                         // if the estimate obtained during the symbolic factorization proves to be inadequate.
   TArrayI        fRowFact;
   TArrayI        fColFact;

   static Int_t NonZerosUpperTriang(const TMatrixDSparse &a);
   static void  CopyUpperTriang    (const TMatrixDSparse &a,Double_t *b);

          void  InitParam();
   static void  InitPivot (const Int_t n,const Int_t nz,TArrayI &Airn,TArrayI &Aicn,
                           TArrayI &Aiw,TArrayI &Aikeep,TArrayI &Aiw1,Int_t &nsteps,
                           const Int_t iflag,Int_t *icntl,Double_t *cntl,Int_t *info,Double_t &ops);
   static void   Factor   (const Int_t n,const Int_t nz,TArrayI &Airn,TArrayI &Aicn,TArrayD &Aa,
                           TArrayI &Aiw,TArrayI &Aikeep,const Int_t nsteps,Int_t &maxfrt,
                           TArrayI &Aiw1,Int_t *icntl,Double_t *cntl,Int_t *info);
   static void   Solve    (const Int_t n,TArrayD &Aa,TArrayI &Aiw,TArrayD &Aw,const Int_t maxfrt,
                           TVectorD &b,TArrayI &Aiw1,const Int_t nsteps,Int_t *icntl,Int_t *info);
 
   static void   InitPivot_sub1 (const Int_t n,const Int_t nz,Int_t *irn,Int_t *icn,Int_t *iw,Int_t *ipe,
                                 Int_t *iq,Int_t *flag,Int_t &iwfr,Int_t *icntl,Int_t *info);
   static void   InitPivot_sub2 (const Int_t n,Int_t *ipe,Int_t *iw,const Int_t lw,Int_t &iwfr,Int_t *nv,
                                 Int_t *nxt,Int_t *lst,Int_t *ipd,Int_t *flag,const Int_t iovflo,Int_t &ncmpa,
                                 const Double_t fratio);
   static void   InitPivot_sub2a(const Int_t n,Int_t *ipe,Int_t *iw,const Int_t lw,Int_t &iwfr,Int_t &ncmpa);
   static void   InitPivot_sub3 (const Int_t n,const Int_t nz,Int_t *irn,Int_t *icn,Int_t *perm,Int_t *iw,
                                 Int_t *ipe,Int_t *iq,Int_t *flag,Int_t &iwfr,Int_t *icntl,Int_t *info);
   static void   InitPivot_sub4 (const Int_t n,Int_t *ipe,Int_t *iw,const Int_t lw,Int_t &iwfr,Int_t *ips,
                                 Int_t *ipv,Int_t *nv,Int_t *flag,Int_t &ncmpa);
   static void   InitPivot_sub5 (const Int_t n,Int_t *ipe,Int_t *nv,Int_t *ips,Int_t *ne,Int_t *na,Int_t *nd,
                                 Int_t &nsteps,const Int_t nemin);
   static void   InitPivot_sub6 (const Int_t n,const Int_t nz,Int_t *irn,Int_t *icn,Int_t *perm,Int_t *na,
                                 Int_t *ne,Int_t *nd,const Int_t nsteps,Int_t *lstki,Int_t *lstkr,Int_t *iw,
                                 Int_t *info,Double_t &ops);

   static void   Factor_sub1    (const Int_t n,const Int_t nz,Int_t &nz1,Double_t *a,const Int_t la,
                                 Int_t *irn,Int_t *icn,Int_t *iw,const Int_t liw,Int_t *perm,Int_t *iw2,
                                 Int_t *icntl,Int_t *info);
   static void   Factor_sub2    (const Int_t n,const Int_t nz,Double_t *a,const Int_t la,Int_t *iw,
                                 const Int_t liw,Int_t *perm,Int_t *nstk,const Int_t nsteps,Int_t &maxfrt,
                                 Int_t *nelim,Int_t *iw2,Int_t *icntl,Double_t *cntl,Int_t *info);
   static void   Factor_sub3    (Double_t *a,Int_t *iw,Int_t &j1,Int_t &j2,const Int_t itop,const Int_t ireal,
                                 Int_t &ncmpbr,Int_t &ncmpbi);

   static void   Solve_sub1     (const Int_t n,Double_t *a,Int_t *iw,Double_t *w,Double_t *rhs,Int_t *iw2,
                                 const Int_t nblk,Int_t &latop,Int_t *icntl);
   static void   Solve_sub2     (const Int_t n,Double_t *a,Int_t *iw,Double_t *w,Double_t *rhs,Int_t *iw2,
                                 const Int_t nblk,const Int_t latop,Int_t *icntl);
   static Int_t  IDiag          (Int_t ix,Int_t iy) { return ((iy-1)*(2*ix-iy+2))/2; }

   inline Int_t IError          () { return fInfo[2]; }
   inline Int_t MinRealWorkspace() { return fInfo[5]; }
   inline Int_t MinIntWorkspace () { return fInfo[6]; }
   inline Int_t ErrorFlag       () { return fInfo[1]; }

   // Takes values in the range [0,1]. Larger values enforce greater stability in
   // the factorization as they insist on larger pivots. Smaller values preserve
   // sparsity at the cost of using smaller pivots.

   inline Double_t GetThresholdPivoting() { return fCntl[1]; }
   inline Double_t GetTreatAsZero      () { return fCntl[3]; }

   // The factorization will not accept a pivot whose absolute value is less than fCntl[3] as
   // a 1x1 pivot or as the off-diagonal in a 2x2 pivot.

   inline void     SetThresholdPivoting(Double_t piv) { fCntl[1] = piv; }
   inline void     SetTreatAsZero      (Double_t tol) { fCntl[3] = tol; }

   virtual const TMatrixDBase &GetDecompMatrix() const { MayNotUse("GetDecompMatrix()"); return fA; }

public :

   TDecompSparse();
   TDecompSparse(Int_t nRows,Int_t nr_nonZeros,Int_t verbose);
   TDecompSparse(Int_t row_lwb,Int_t row_upb,Int_t nr_nonZeros,Int_t verbose);
   TDecompSparse(const TMatrixDSparse &a,Int_t verbose);
   TDecompSparse(const TDecompSparse &another);
   virtual ~TDecompSparse() {}

   inline  void     SetVerbose (Int_t v) { fVerbose = (v) ? 1 : 0;
                                            if (fVerbose) { fIcntl[1] = fIcntl[2] = 1; fIcntl[3] = 2; }
                                            else          { fIcntl[1] = fIcntl[2] = fIcntl[3] = 0; }
                                         }
   virtual Int_t    GetNrows   () const { return fA.GetNrows(); }
   virtual Int_t    GetNcols   () const { return fA.GetNcols(); }

   virtual void     SetMatrix  (const TMatrixDSparse &a);

   virtual Bool_t   Decompose  ();
   virtual Bool_t   Solve      (      TVectorD &b);
   virtual TVectorD Solve      (const TVectorD& b,Bool_t &ok) { TVectorD x = b; ok = Solve(x); return x; }
   virtual Bool_t   Solve      (      TMatrixDColumn & /*b*/)
                               { MayNotUse("Solve(TMatrixDColumn &)"); return kFALSE; }
   virtual Bool_t   TransSolve (      TVectorD &b)            { return Solve(b); }
   virtual TVectorD TransSolve (const TVectorD& b,Bool_t &ok) { TVectorD x = b; ok = Solve(x); return x; }
   virtual Bool_t   TransSolve (      TMatrixDColumn & /*b*/)
                               { MayNotUse("TransSolve(TMatrixDColumn &)"); return kFALSE; }

   virtual void     Det        (Double_t &/*d1*/,Double_t &/*d2*/)
                                { MayNotUse("Det(Double_t&,Double_t&)"); }

   void Print(Option_t *opt ="") const; // *MENU*

   TDecompSparse &operator= (const TDecompSparse &source);

   ClassDef(TDecompSparse,1) // Matrix Decompositition LU
};

#endif
