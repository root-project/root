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

#pragma link C++ namespace TMatrixTAutoloadOps;

#endif
