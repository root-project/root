/* @(#)root/matrix:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ global gMatrixCheck;

#pragma link C++ namespace TMatrixTCramerInv;
#pragma link C++ function  TMatrixTCramerInv::Inv2x2(TMatrixT<float>&,Double_t*);
#pragma link C++ function  TMatrixTCramerInv::Inv2x2(TMatrixT<double>&,Double_t*);
#pragma link C++ function  TMatrixTCramerInv::Inv3x3(TMatrixT<float>&,Double_t*);
#pragma link C++ function  TMatrixTCramerInv::Inv3x3(TMatrixT<double>&,Double_t*);
#pragma link C++ function  TMatrixTCramerInv::Inv4x4(TMatrixT<float>&,Double_t*);
#pragma link C++ function  TMatrixTCramerInv::Inv4x4(TMatrixT<double>&,Double_t*);
#pragma link C++ function  TMatrixTCramerInv::Inv5x5(TMatrixT<float>&,Double_t*);
#pragma link C++ function  TMatrixTCramerInv::Inv5x5(TMatrixT<double>&,Double_t*);
#pragma link C++ function  TMatrixTCramerInv::Inv6x6(TMatrixT<float>&,Double_t*);
#pragma link C++ function  TMatrixTCramerInv::Inv6x6(TMatrixT<double>&,Double_t*);

#pragma link C++ namespace TMatrixTSymCramerInv;
#pragma link C++ function  TMatrixTSymCramerInv::Inv2x2(TMatrixTSym<float>&,Double_t*);
#pragma link C++ function  TMatrixTSymCramerInv::Inv2x2(TMatrixTSym<double>&,Double_t*);
#pragma link C++ function  TMatrixTSymCramerInv::Inv3x3(TMatrixTSym<float>&,Double_t*);
#pragma link C++ function  TMatrixTSymCramerInv::Inv3x3(TMatrixTSym<double>&,Double_t*);
#pragma link C++ function  TMatrixTSymCramerInv::Inv4x4(TMatrixTSym<float>&,Double_t*);
#pragma link C++ function  TMatrixTSymCramerInv::Inv4x4(TMatrixTSym<double>&,Double_t*);
#pragma link C++ function  TMatrixTSymCramerInv::Inv5x5(TMatrixTSym<float>&,Double_t*);
#pragma link C++ function  TMatrixTSymCramerInv::Inv5x5(TMatrixTSym<double>&,Double_t*);
#pragma link C++ function  TMatrixTSymCramerInv::Inv6x6(TMatrixTSym<float>&,Double_t*);
#pragma link C++ function  TMatrixTSymCramerInv::Inv6x6(TMatrixTSym<double>&,Double_t*);

#pragma link C++ class TVectorT                <float>-;
#pragma link C++ class TMatrixTBase            <float>-;
#pragma link C++ class TMatrixT                <float>-;
#pragma link C++ class TMatrixTSym             <float>-;
#pragma link C++ class TMatrixTSparse          <float>-;

#pragma link C++ class TMatrixTLazy            <float>+;
#pragma link C++ class TMatrixTSymLazy         <float>+;
#pragma link C++ class THaarMatrixT            <float>+;
#pragma link C++ class THilbertMatrixT         <float>+;
#pragma link C++ class THilbertMatrixTSym      <float>+;

#pragma link C++ class TMatrixTRow_const       <float>;
#pragma link C++ class TMatrixTColumn_const    <float>;
#pragma link C++ class TMatrixTDiag_const      <float>;
#pragma link C++ class TMatrixTFlat_const      <float>;
#pragma link C++ class TMatrixTSub_const       <float>;

#pragma link C++ class TMatrixTRow             <float>;
#pragma link C++ class TMatrixTColumn          <float>;
#pragma link C++ class TMatrixTDiag            <float>;
#pragma link C++ class TMatrixTFlat            <float>;
#pragma link C++ class TMatrixTSub             <float>;

#pragma link C++ class TMatrixTSparseRow_const <float>;
#pragma link C++ class TMatrixTSparseRow       <float>;
#pragma link C++ class TMatrixTSparseDiag_const<float>;
#pragma link C++ class TMatrixTSparseDiag      <float>;

#pragma link C++ typedef TVector;
#pragma link C++ typedef TVectorF;
#pragma link C++ typedef TMatrix;
#pragma link C++ typedef TMatrixF;
#pragma link C++ typedef TMatrixFSym;
#pragma link C++ typedef TMatrixFSparse;

#pragma link C++ typedef TMatrixFLazy;
#pragma link C++ typedef TMatrixFSymLazy;
#pragma link C++ typedef THaarMatrixF;
#pragma link C++ typedef THilbertMatrixF;
#pragma link C++ typedef THilbertMatrixFSym;

#pragma link C++ typedef TMatrixFRow_const;
#pragma link C++ typedef TMatrixFColumn_const;
#pragma link C++ typedef TMatrixFDiag_const;
#pragma link C++ typedef TMatrixFFlat_const;
#pragma link C++ typedef TMatrixFSub_const;
#pragma link C++ typedef TMatrixFRow;
#pragma link C++ typedef TMatrixFColumn;
#pragma link C++ typedef TMatrixFDiag;
#pragma link C++ typedef TMatrixFFlat;
#pragma link C++ typedef TMatrixFSub;

#pragma link C++ typedef TMatrixFSparseRow_const;
#pragma link C++ typedef TMatrixFSparseRow;
#pragma link C++ typedef TMatrixFSparseDiag_const;
#pragma link C++ typedef TMatrixFSparseDiag;

#pragma link C++ class TVectorT                <double>-;
#pragma link C++ class TMatrixTBase            <double>-;
#pragma link C++ class TMatrixT                <double>-;
#pragma link C++ class TMatrixTSym             <double>-;
#pragma link C++ class TMatrixTSparse          <double>-;

#pragma link C++ class TMatrixTLazy            <double>+;
#pragma link C++ class TMatrixTSymLazy         <double>+;
#pragma link C++ class THaarMatrixT            <double>+;
#pragma link C++ class THilbertMatrixT         <double>+;
#pragma link C++ class THilbertMatrixTSym      <double>+;

#pragma link C++ class TMatrixTRow_const       <double>;
#pragma link C++ class TMatrixTColumn_const    <double>;
#pragma link C++ class TMatrixTDiag_const      <double>;
#pragma link C++ class TMatrixTFlat_const      <double>;
#pragma link C++ class TMatrixTSub_const       <double>;

#pragma link C++ class TMatrixTRow             <double>;
#pragma link C++ class TMatrixTColumn          <double>;
#pragma link C++ class TMatrixTDiag            <double>;
#pragma link C++ class TMatrixTFlat            <double>;
#pragma link C++ class TMatrixTSub             <double>;

#pragma link C++ class TMatrixTSparseRow_const <double>;
#pragma link C++ class TMatrixTSparseRow       <double>;
#pragma link C++ class TMatrixTSparseDiag_const<double>;
#pragma link C++ class TMatrixTSparseDiag      <double>;

#pragma link C++ typedef TVectorD;
#pragma link C++ typedef TMatrixD;
#pragma link C++ typedef TMatrixDSym;
#pragma link C++ typedef TMatrixDSparse;

#pragma link C++ typedef TMatrixDLazy;
#pragma link C++ typedef TMatrixDSymLazy;
#pragma link C++ typedef THaarMatrixD;
#pragma link C++ typedef THilbertMatrixD;
#pragma link C++ typedef THilbertMatrixDSym;

#pragma link C++ typedef TMatrixDRow_const;
#pragma link C++ typedef TMatrixDColumn_const;
#pragma link C++ typedef TMatrixDDiag_const;
#pragma link C++ typedef TMatrixDFlat_const;
#pragma link C++ typedef TMatrixDSub_const;

#pragma link C++ typedef TMatrixDRow;
#pragma link C++ typedef TMatrixDColumn;
#pragma link C++ typedef TMatrixDDiag;
#pragma link C++ typedef TMatrixDFlat;
#pragma link C++ typedef TMatrixDSub;

#pragma link C++ typedef TMatrixDSparseRow_const;
#pragma link C++ typedef TMatrixDSparseRow;
#pragma link C++ typedef TMatrixDSparseDiag_const;
#pragma link C++ typedef TMatrixDSparseDiag;

#pragma link C++ class TMatrixDEigen+;
#pragma link C++ class TMatrixDSymEigen+;

#pragma link C++ class TDecompBase+;
#pragma link C++ class TDecompBK+;
#pragma link C++ class TDecompChol+;
#pragma link C++ class TDecompLU+;
#pragma link C++ class TDecompQRH+;
#pragma link C++ class TDecompSVD+;
#pragma link C++ class TDecompSparse+;

//TVectorT<float>
#pragma link C++ function operator==          (const TVectorF       &,const TVectorF &);
#pragma link C++ function operator+           (const TVectorF       &,const TVectorF &);
#pragma link C++ function operator-           (const TVectorF       &,const TVectorF &);
#pragma link C++ function operator*           (const TVectorF       &,const TVectorF &);
#pragma link C++ function operator*           (const TMatrixF       &,const TVectorF &);
#pragma link C++ function operator*           (const TMatrixFSym    &,const TVectorF &);
#pragma link C++ function operator*           (const TMatrixFSparse &,const TVectorF &);
#pragma link C++ function operator*           (      Float_t         ,const TVectorF &);
#pragma link C++ function Dot                 (const TVectorF       &,const TVectorF &);
#pragma link C++ function OuterProduct        (const TVectorF       &,const TVectorF &);
#pragma link C++ function OuterProduct        (      TMatrixF       &,const TVectorF &,const TVectorF &);
#pragma link C++ function Mult                (const TVectorF       &,const TMatrixF &,const TVectorF &);
#pragma link C++ function Add                 (      TVectorF       &,      Float_t   ,const TVectorF &);
#pragma link C++ function Add                 (      TVectorF       &,      Float_t   ,const TMatrixF       &,const TVectorF &);
#pragma link C++ function Add                 (      TVectorF       &,      Float_t   ,const TMatrixFSym    &,const TVectorF &);
#pragma link C++ function Add                 (      TVectorF       &,      Float_t   ,const TMatrixFSparse &,const TVectorF &);
#pragma link C++ function AddElemMult         (      TVectorF       &,      Float_t   ,const TVectorF       &,const TVectorF &);
#pragma link C++ function AddElemMult         (      TVectorF       &,      Float_t   ,const TVectorF       &,const TVectorF &, \
                                               const TVectorF &);
#pragma link C++ function AddElemDiv          (      TVectorF       &,      Float_t   ,const TVectorF       &,const TVectorF &);
#pragma link C++ function AddElemDiv          (      TVectorF       &,      Float_t   ,const TVectorF       &,const TVectorF &, \
                                               const TVectorF &);
#pragma link C++ function ElementMult         (      TVectorF       &,const TVectorF &);
#pragma link C++ function ElementMult         (      TVectorF       &,const TVectorF &,const TVectorF       &);
#pragma link C++ function ElementDiv          (      TVectorF       &,const TVectorF &);
#pragma link C++ function ElementDiv          (      TVectorF       &,const TVectorF &,const TVectorF       &);
#pragma link C++ function AreCompatible       (const TVectorF       &,const TVectorF &,      Int_t);
#pragma link C++ function AreCompatible       (const TVectorF       &,const TVectorD &,      Int_t);
#pragma link C++ function AreCompatible       (const TMatrixF       &,const TVectorF &,      Int_t);
#pragma link C++ function AreCompatible       (const TVectorF       &,const TMatrixF &,      Int_t);

#pragma link C++ function Compare             (const TVectorF       &,const TVectorF &);
#pragma link C++ function VerifyVectorValue   (const TVectorF       &,      Float_t   ,      Int_t,          Float_t);
#pragma link C++ function VerifyVectorIdentity(const TVectorF       &,const TVectorF &,      Int_t,          Float_t);

//TMatrixTBase<float>
#pragma link C++ function operator==          (const TMatrixFBase   &,const TMatrixFBase &);
#pragma link C++ function E2Norm              (const TMatrixFBase   &,const TMatrixFBase &);
#pragma link C++ function AreCompatible       (const TMatrixFBase   &,const TMatrixFBase &,Int_t);
#pragma link C++ function AreCompatible       (const TMatrixFBase   &,const TMatrixDBase &,Int_t);
#pragma link C++ function Compare             (const TMatrixFBase   &,const TMatrixFBase &);
#pragma link C++ function VerifyMatrixValue   (const TMatrixFBase   &,      Float_t       ,Int_t,Float_t);
#pragma link C++ function VerifyMatrixIdentity(const TMatrixFBase   &,const TMatrixFBase &,Int_t,Float_t);

//TMatrixT<float>
#pragma link C++ function operator+  (const TMatrixF    &,const TMatrixF    &);
#pragma link C++ function operator+  (const TMatrixF    &,const TMatrixFSym &);
#pragma link C++ function operator+  (const TMatrixFSym &,const TMatrixF    &);
#pragma link C++ function operator+  (const TMatrixF    &,      Float_t      );
#pragma link C++ function operator+  (      Float_t      ,const TMatrixF    &);
#pragma link C++ function operator-  (const TMatrixF    &,const TMatrixF    &);
#pragma link C++ function operator-  (const TMatrixF    &,const TMatrixFSym &);
#pragma link C++ function operator-  (const TMatrixFSym &,const TMatrixF    &);
#pragma link C++ function operator-  (const TMatrixF    &,      Float_t      );
#pragma link C++ function operator-  (      Float_t      ,const TMatrixF    &);

#pragma link C++ function operator*  (      Float_t      ,const TMatrixF    &);
#pragma link C++ function operator*  (const TMatrixF    &,      Float_t      );
#pragma link C++ function operator*  (const TMatrixF    &,const TMatrixF    &);
#pragma link C++ function operator*  (const TMatrixF    &,const TMatrixFSym &);
#pragma link C++ function operator*  (const TMatrixFSym &,const TMatrixF    &);
#pragma link C++ function operator*  (const TMatrixFSym &,const TMatrixFSym &);
#pragma link C++ function operator&& (const TMatrixF    &,const TMatrixF    &);
#pragma link C++ function operator&& (const TMatrixF    &,const TMatrixFSym &);
#pragma link C++ function operator&& (const TMatrixFSym &,const TMatrixF    &);
#pragma link C++ function operator|| (const TMatrixF    &,const TMatrixF    &);

#pragma link C++ function operator|| (const TMatrixF    &,const TMatrixFSym &);
#pragma link C++ function operator|| (const TMatrixFSym &,const TMatrixF    &);
#pragma link C++ function operator>  (const TMatrixF    &,const TMatrixF    &);
#pragma link C++ function operator>  (const TMatrixF    &,const TMatrixFSym &);
#pragma link C++ function operator>  (const TMatrixFSym &,const TMatrixF    &);
#pragma link C++ function operator>= (const TMatrixF    &,const TMatrixF    &);
#pragma link C++ function operator>= (const TMatrixF    &,const TMatrixFSym &);
#pragma link C++ function operator>= (const TMatrixFSym &,const TMatrixF    &);
#pragma link C++ function operator<= (const TMatrixF    &,const TMatrixF    &);
#pragma link C++ function operator<= (const TMatrixF    &,const TMatrixFSym &);
#pragma link C++ function operator<= (const TMatrixFSym &,const TMatrixF    &);
#pragma link C++ function operator<  (const TMatrixF    &,const TMatrixF    &);
#pragma link C++ function operator<  (const TMatrixF    &,const TMatrixFSym &);
#pragma link C++ function operator<  (const TMatrixFSym &,const TMatrixF    &);
#pragma link C++ function operator!= (const TMatrixF    &,const TMatrixF    &);
#pragma link C++ function operator!= (const TMatrixF    &,const TMatrixFSym &);
#pragma link C++ function operator!= (const TMatrixFSym &,const TMatrixF    &);

#pragma link C++ function Add        (      TMatrixF    &,      Float_t      ,const TMatrixF    &);
#pragma link C++ function Add        (      TMatrixF    &,      Float_t      ,const TMatrixFSym &);
#pragma link C++ function ElementMult(      TMatrixF    &,const TMatrixF    &);
#pragma link C++ function ElementMult(      TMatrixF    &,const TMatrixFSym &);
#pragma link C++ function ElementDiv (      TMatrixF    &,const TMatrixF    &);
#pragma link C++ function ElementDiv (      TMatrixF    &,const TMatrixFSym &);

//TMatrixTSym<float>
#pragma link C++ function operator== (const TMatrixFSym &,const TMatrixFSym &);
#pragma link C++ function operator+  (const TMatrixFSym &,const TMatrixFSym &);
#pragma link C++ function operator+  (const TMatrixFSym &,      Float_t      );
#pragma link C++ function operator+  (      Float_t      ,const TMatrixFSym &);
#pragma link C++ function operator-  (const TMatrixFSym &,const TMatrixFSym &);
#pragma link C++ function operator-  (const TMatrixFSym &,      Float_t     );
#pragma link C++ function operator-  (      Float_t      ,const TMatrixFSym &);
#pragma link C++ function operator*  (const TMatrixFSym &,      Float_t      );
#pragma link C++ function operator*  (      Float_t      ,const TMatrixFSym &);
#pragma link C++ function operator&& (const TMatrixFSym &,const TMatrixFSym &);
#pragma link C++ function operator|| (const TMatrixFSym &,const TMatrixFSym &);
#pragma link C++ function operator>  (const TMatrixFSym &,const TMatrixFSym &);
#pragma link C++ function operator>= (const TMatrixFSym &,const TMatrixFSym &);
#pragma link C++ function operator<= (const TMatrixFSym &,const TMatrixFSym &);
#pragma link C++ function operator<  (const TMatrixFSym &,const TMatrixFSym &);
#pragma link C++ function Add        (      TMatrixFSym &,      Float_t      ,const TMatrixFSym &);
#pragma link C++ function ElementMult(      TMatrixFSym &,const TMatrixFSym &);
#pragma link C++ function ElementDiv (      TMatrixFSym &,const TMatrixFSym &);

//TMatrixTSparse<float>
#pragma link C++ function operator+    (const TMatrixFSparse &,const TMatrixFSparse &);
#pragma link C++ function operator+    (const TMatrixFSparse &,const TMatrixF       &);
#pragma link C++ function operator+    (const TMatrixF       &,const TMatrixFSparse &);
#pragma link C++ function operator+    (const TMatrixFSparse &,      Float_t         );
#pragma link C++ function operator+    (      Float_t         ,const TMatrixFSparse &);
#pragma link C++ function operator-    (const TMatrixFSparse &,const TMatrixFSparse &);
#pragma link C++ function operator-    (const TMatrixFSparse &,const TMatrixF       &);
#pragma link C++ function operator-    (const TMatrixF       &,const TMatrixFSparse &);
#pragma link C++ function operator-    (const TMatrixFSparse &,      Float_t         );
#pragma link C++ function operator-    (      Float_t         ,const TMatrixFSparse &);
#pragma link C++ function operator*    (const TMatrixFSparse &,const TMatrixFSparse &);
#pragma link C++ function operator*    (const TMatrixFSparse &,const TMatrixF       &);
#pragma link C++ function operator*    (const TMatrixF       &,const TMatrixFSparse &);
#pragma link C++ function operator*    (      Float_t         ,const TMatrixFSparse &);
#pragma link C++ function operator*    (const TMatrixFSparse &,      Float_t         );
#pragma link C++ function Add          (      TMatrixFSparse &,      Float_t         ,const TMatrixFSparse &);
#pragma link C++ function ElementMult  (      TMatrixFSparse &,const TMatrixFSparse &);
#pragma link C++ function ElementDiv   (      TMatrixFSparse &,const TMatrixFSparse &);
#pragma link C++ function AreCompatible(const TMatrixFSparse &,const TMatrixFSparse &,Int_t);

//TVectorT<double>
#pragma link C++ function operator==          (const TVectorD       &,const TVectorD &);
#pragma link C++ function operator+           (const TVectorD       &,const TVectorD &);
#pragma link C++ function operator-           (const TVectorD       &,const TVectorD &);
#pragma link C++ function operator*           (const TVectorD       &,const TVectorD &);
#pragma link C++ function operator*           (const TMatrixD       &,const TVectorD &);
#pragma link C++ function operator*           (const TMatrixDSym    &,const TVectorD &);
#pragma link C++ function operator*           (const TMatrixDSparse &,const TVectorD &);
#pragma link C++ function operator*           (      Double_t        ,const TVectorD &);
#pragma link C++ function Dot                 (const TVectorD       &,const TVectorD &);
#pragma link C++ function OuterProduct        (const TVectorD       &,const TVectorD &);
#pragma link C++ function OuterProduct        (      TMatrixD       &,const TVectorD &,const TVectorD &);
#pragma link C++ function Mult                (const TVectorD       &,const TMatrixD &,const TVectorD &);
#pragma link C++ function Add                 (      TVectorD       &,      Double_t  ,const TVectorD &);
#pragma link C++ function Add                 (      TVectorD       &,      Double_t  ,const TMatrixD       &,const TVectorD &);
#pragma link C++ function Add                 (      TVectorD       &,      Double_t  ,const TMatrixDSym    &,const TVectorD &);
#pragma link C++ function Add                 (      TVectorD       &,      Double_t  ,const TMatrixDSparse &,const TVectorD &);
#pragma link C++ function AddElemMult         (      TVectorD       &,      Double_t  ,const TVectorD       &,const TVectorD &);
#pragma link C++ function AddElemMult         (      TVectorD       &,      Double_t  ,const TVectorD       &,const TVectorD &, \
                                               const TVectorD &);
#pragma link C++ function AddElemDiv          (      TVectorD       &,      Double_t  ,const TVectorD       &,const TVectorD &);
#pragma link C++ function AddElemDiv          (      TVectorD       &,      Double_t  ,const TVectorD       &,const TVectorD &, \
                                               const TVectorD &);
#pragma link C++ function ElementMult         (      TVectorD       &,const TVectorD &);
#pragma link C++ function ElementMult         (      TVectorD       &,const TVectorD &,const TVectorD       &);
#pragma link C++ function ElementDiv          (      TVectorD       &,const TVectorD &);
#pragma link C++ function ElementDiv          (      TVectorD       &,const TVectorD &,const TVectorD       &);
#pragma link C++ function AreCompatible       (const TVectorD       &,const TVectorD &,      Int_t);
#pragma link C++ function AreCompatible       (const TVectorD       &,const TVectorF &,      Int_t);
#pragma link C++ function AreCompatible       (const TMatrixD       &,const TVectorD &,      Int_t);
#pragma link C++ function AreCompatible       (const TVectorD       &,const TMatrixD &,      Int_t);
#pragma link C++ function Compare             (const TVectorD       &,const TVectorD &);
#pragma link C++ function VerifyVectorValue   (const TVectorD       &,      Double_t  ,      Int_t,          Double_t);
#pragma link C++ function VerifyVectorIdentity(const TVectorD       &,const TVectorD &,      Int_t,          Double_t);

//TMatrixTBase<double>
#pragma link C++ function operator==          (const TMatrixDBase   &,const TMatrixDBase &);
#pragma link C++ function E2Norm              (const TMatrixDBase   &,const TMatrixDBase &);
#pragma link C++ function AreCompatible       (const TMatrixDBase   &,const TMatrixDBase &,Int_t);
#pragma link C++ function AreCompatible       (const TMatrixDBase   &,const TMatrixFBase &,Int_t);
#pragma link C++ function Compare             (const TMatrixDBase   &,const TMatrixDBase &);
#pragma link C++ function VerifyMatrixValue   (const TMatrixDBase   &,      Double_t      ,Int_t,Double_t);
#pragma link C++ function VerifyMatrixIdentity(const TMatrixDBase   &,const TMatrixDBase &,Int_t,Double_t);

//TMatrixT<double>
#pragma link C++ function operator+  (const TMatrixD    &,const TMatrixD    &);
#pragma link C++ function operator+  (const TMatrixD    &,const TMatrixDSym &);
#pragma link C++ function operator+  (const TMatrixDSym &,const TMatrixD    &);
#pragma link C++ function operator+  (const TMatrixD    &,      Double_t     );
#pragma link C++ function operator+  (      Double_t     ,const TMatrixD    &);
#pragma link C++ function operator-  (const TMatrixD    &,const TMatrixD    &);
#pragma link C++ function operator-  (const TMatrixD    &,const TMatrixDSym &);
#pragma link C++ function operator-  (const TMatrixDSym &,const TMatrixD    &);
#pragma link C++ function operator-  (const TMatrixD    &,      Double_t     );
#pragma link C++ function operator-  (      Double_t     ,const TMatrixD    &);

#pragma link C++ function operator*  (      Double_t     ,const TMatrixD    &);
#pragma link C++ function operator*  (const TMatrixD    &,      Double_t     );
#pragma link C++ function operator*  (const TMatrixD    &,const TMatrixD    &);
#pragma link C++ function operator*  (const TMatrixD    &,const TMatrixDSym &);
#pragma link C++ function operator*  (const TMatrixDSym &,const TMatrixD    &);
#pragma link C++ function operator*  (const TMatrixDSym &,const TMatrixDSym &);
#pragma link C++ function operator&& (const TMatrixD    &,const TMatrixD    &);
#pragma link C++ function operator&& (const TMatrixD    &,const TMatrixDSym &);
#pragma link C++ function operator&& (const TMatrixDSym &,const TMatrixD    &);
#pragma link C++ function operator|| (const TMatrixD    &,const TMatrixD    &);

#pragma link C++ function operator|| (const TMatrixD    &,const TMatrixDSym &);
#pragma link C++ function operator|| (const TMatrixDSym &,const TMatrixD    &);
#pragma link C++ function operator>  (const TMatrixD    &,const TMatrixD    &);
#pragma link C++ function operator>  (const TMatrixD    &,const TMatrixDSym &);
#pragma link C++ function operator>  (const TMatrixDSym &,const TMatrixD    &);
#pragma link C++ function operator>= (const TMatrixD    &,const TMatrixD    &);
#pragma link C++ function operator>= (const TMatrixD    &,const TMatrixDSym &);
#pragma link C++ function operator>= (const TMatrixDSym &,const TMatrixD    &);
#pragma link C++ function operator<= (const TMatrixD    &,const TMatrixD    &);
#pragma link C++ function operator<= (const TMatrixD    &,const TMatrixDSym &);
#pragma link C++ function operator<= (const TMatrixDSym &,const TMatrixD    &);
#pragma link C++ function operator<  (const TMatrixD    &,const TMatrixD    &);
#pragma link C++ function operator<  (const TMatrixD    &,const TMatrixDSym &);
#pragma link C++ function operator<  (const TMatrixDSym &,const TMatrixD    &);
#pragma link C++ function operator!= (const TMatrixD    &,const TMatrixD    &);
#pragma link C++ function operator!= (const TMatrixD    &,const TMatrixDSym &);
#pragma link C++ function operator!= (const TMatrixDSym &,const TMatrixD    &);

#pragma link C++ function Add        (      TMatrixD    &,      Double_t     ,const TMatrixD    &);
#pragma link C++ function Add        (      TMatrixD    &,      Double_t     ,const TMatrixDSym &);
#pragma link C++ function ElementMult(      TMatrixD    &,const TMatrixD    &);
#pragma link C++ function ElementMult(      TMatrixD    &,const TMatrixDSym &);
#pragma link C++ function ElementDiv (      TMatrixD    &,const TMatrixD    &);
#pragma link C++ function ElementDiv (      TMatrixD    &,const TMatrixDSym &);

//TMatrixTSym<double>
#pragma link C++ function operator== (const TMatrixDSym &,const TMatrixDSym &);
#pragma link C++ function operator+  (const TMatrixDSym &,const TMatrixDSym &);
#pragma link C++ function operator+  (const TMatrixDSym &,      Double_t     );
#pragma link C++ function operator+  (      Double_t     ,const TMatrixDSym &);
#pragma link C++ function operator-  (const TMatrixDSym &,const TMatrixDSym &);
#pragma link C++ function operator-  (const TMatrixDSym &,      Double_t    );
#pragma link C++ function operator-  (      Double_t     ,const TMatrixDSym &);
#pragma link C++ function operator*  (const TMatrixDSym &,      Double_t     );
#pragma link C++ function operator*  (      Double_t     ,const TMatrixDSym &);
#pragma link C++ function operator&& (const TMatrixDSym &,const TMatrixDSym &);
#pragma link C++ function operator|| (const TMatrixDSym &,const TMatrixDSym &);
#pragma link C++ function operator>  (const TMatrixDSym &,const TMatrixDSym &);
#pragma link C++ function operator>= (const TMatrixDSym &,const TMatrixDSym &);
#pragma link C++ function operator<= (const TMatrixDSym &,const TMatrixDSym &);
#pragma link C++ function operator<  (const TMatrixDSym &,const TMatrixDSym &);
#pragma link C++ function Add        (      TMatrixDSym &,      Double_t     ,const TMatrixDSym &);
#pragma link C++ function ElementMult(      TMatrixDSym &,const TMatrixDSym &);
#pragma link C++ function ElementDiv (      TMatrixDSym &,const TMatrixDSym &);

//TMatrixTSparse<double>
#pragma link C++ function operator+    (const TMatrixDSparse &,const TMatrixDSparse &);
#pragma link C++ function operator+    (const TMatrixDSparse &,const TMatrixD       &);
#pragma link C++ function operator+    (const TMatrixD       &,const TMatrixDSparse &);
#pragma link C++ function operator+    (const TMatrixDSparse &,      Double_t        );
#pragma link C++ function operator+    (      Double_t        ,const TMatrixDSparse &);
#pragma link C++ function operator-    (const TMatrixDSparse &,const TMatrixDSparse &);
#pragma link C++ function operator-    (const TMatrixDSparse &,const TMatrixD       &);
#pragma link C++ function operator-    (const TMatrixD       &,const TMatrixDSparse &);
#pragma link C++ function operator-    (const TMatrixDSparse &,      Double_t        );
#pragma link C++ function operator-    (      Double_t        ,const TMatrixDSparse &);
#pragma link C++ function operator*    (const TMatrixDSparse &,const TMatrixDSparse &);
#pragma link C++ function operator*    (const TMatrixDSparse &,const TMatrixD       &);
#pragma link C++ function operator*    (const TMatrixD       &,const TMatrixDSparse &);
#pragma link C++ function operator*    (      Double_t        ,const TMatrixDSparse &);
#pragma link C++ function operator*    (const TMatrixDSparse &,      Double_t        );
#pragma link C++ function Add          (      TMatrixDSparse &,      Double_t        ,const TMatrixDSparse &);
#pragma link C++ function ElementMult  (      TMatrixDSparse &,const TMatrixDSparse &);
#pragma link C++ function ElementDiv   (      TMatrixDSparse &,const TMatrixDSparse &);
#pragma link C++ function AreCompatible(const TMatrixDSparse &,const TMatrixDSparse &,Int_t);

#pragma link C++ function NormalEqn (const TMatrixD &,const TVectorD &                 );
#pragma link C++ function NormalEqn (const TMatrixD &,const TVectorD &,const TVectorD &);
#pragma link C++ function NormalEqn (const TMatrixD &,const TMatrixD &                 );
#pragma link C++ function NormalEqn (const TMatrixD &,const TMatrixD &,const TVectorD &);

#endif
