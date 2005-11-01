// @(#)root/matrix:$Name:  $:$Id: TMatrixFUtils.h,v 1.30 2005/04/07 14:43:35 rdm Exp $
// Authors: Fons Rademakers, Eddy Offermann   Nov 2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMatrixFUtilsfwd
#define ROOT_TMatrixFUtilsfwd

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Matrix utility classes.                                              //
//                                                                      //
//  Forward declaration of                                              //
//   TMatrixTRow_const       <Float_t>  TMatrixTRow       <Float_t>     //
//   TMatrixTColumn_const    <Float_t>  TMatrixTColumn    <Float_t>     //
//   TMatrixTDiag_const      <Float_t>  TMatrixTDiag      <Float_t>     //
//   TMatrixTFlat_const      <Float_t>  TMatrixTFlat      <Float_t>     //
//   TMatrixTSub_const       <Float_t>  TMatrixTSub       <Float_t>     //
//   TMatrixTSparseRow_const <Float_t>  TMatrixTSparseRow <Float_t>     //
//   TMatrixTSparseDiag_const<Float_t>  TMatrixTSparseDiag<Float_t>     //
//                                                                      //
//   TElementActionT   <Float_t>                                        //
//   TElementPosActionT<Float_t>                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TMatrixFRow_const;
class TMatrixFColumn_const;
class TMatrixFDiag_const;
class TMatrixFFlat_const;
class TMatrixFSub_const;
class TMatrixFSparseRow_const;
class TMatrixFSparseDiag_const;

class TMatrixFRow;
class TMatrixFColumn;
class TMatrixFDiag;
class TMatrixFFlat;
class TMatrixFSub;
class TMatrixFSparseRow;
class TMatrixFSparseDiag;

class TElementActionF;
class TElementPosActionF;

#endif
