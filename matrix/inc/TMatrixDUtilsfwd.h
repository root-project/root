// @(#)root/matrix:$Name:  $:$Id: TMatrixDUtils.h,v 1.30 2005/04/07 14:43:35 rdm Exp $
// Authors: Fons Rademakers, Eddy Offermann   Nov 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMatrixDUtilsfwd
#define ROOT_TMatrixDUtilsfwd

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Matrix utility classes.                                              //
//                                                                      //
//  Forward declaration of                                              //
//   TMatrixTRow_const       <Double_t>  TMatrixTRow       <Double_t>   //
//   TMatrixTColumn_const    <Double_t>  TMatrixTColumn    <Double_t>   //
//   TMatrixTDiag_const      <Double_t>  TMatrixTDiag      <Double_t>   //
//   TMatrixTFlat_const      <Double_t>  TMatrixTFlat      <Double_t>   //
//   TMatrixTSub_const       <Double_t>  TMatrixTSub       <Double_t>   //
//   TMatrixTSparseRow_const <Double_t>  TMatrixTSparseRow <Double_t>   //
//   TMatrixTSparseDiag_const<Double_t>  TMatrixTSparseDiag<Double_t>   //
//                                                                      //
//   TElementActionT   <Double_t>                                       //
//   TElementPosActionT<Double_t>                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TMatrixDRow_const;
class TMatrixDColumn_const;
class TMatrixDDiag_const;
class TMatrixDFlat_const;
class TMatrixDSub_const;
class TMatrixDSparseRow_const;
class TMatrixDSparseDiag_const;

class TMatrixDRow;
class TMatrixDColumn;
class TMatrixDDiag;
class TMatrixDFlat;
class TMatrixDSub;
class TMatrixDSparseRow;
class TMatrixDSparseDiag;

class TElementActionD;
class TElementPosActionD;

#endif
