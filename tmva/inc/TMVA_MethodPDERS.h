// @(#)root/tmva $Id: TMVA_MethodPDERS.h,v 1.8 2006/05/02 23:27:40 helgevoss Exp $    
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_MethodPDERS                                                      *
 *                                                                                *
 * Description:                                                                   *
 *      Multidimensional Likelihood using the "Probability density estimator      *
 *      range search" (PDERS) method suggested in                                 *
 *      T. Carli and B. Koblitz, NIM A 501, 576 (2003)                            *
 *                                                                                *
 *      The multidimensional PDFs for signal and background are modeled           *
 *      by counting the events in the "vicinity" of a test point. The volume      *
 *      that describes "vicinity" is user-defined through the option string.      *
 *      A search method based on binary-trees is used to improve the selection    *
 *      efficiency of the volume search.                                          *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-KP Heidelberg, Germany                                                * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 * File and Version Information:                                                  *
 * $Id: TMVA_MethodPDERS.h,v 1.8 2006/05/02 23:27:40 helgevoss Exp $    
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodPDERS
#define ROOT_TMVA_MethodPDERS

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMVA_MethodPDERS                                                     //
//                                                                      //
// Multidimensional Likelihood using the "Probability density           //
// estimator range search" (PDERS) method                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_MethodBase
#include "TMVA_MethodBase.h"
#endif
#ifndef ROOT_TMVA_BinarySearchTree
#include "TMVA_BinarySearchTree.h"
#endif
#ifndef ROOT_TVector
#include "TVector.h"
#endif

class TMVA_Volume;
class TMVA_Event;

class TMVA_MethodPDERS : public TMVA_MethodBase {

 public:

  TMVA_MethodPDERS( TString jobName, 
		    vector<TString>* theVariables, 
		    TTree* theTree = 0, 
		    TString theOption = "Adaptive:100:200:50:0.99",
		    TDirectory* theTargetDir = 0 );

  TMVA_MethodPDERS( vector<TString> *theVariables, 
		    TString theWeightFile,  
		    TDirectory* theTargetDir = NULL );

  virtual ~TMVA_MethodPDERS( void );
    
  // training method
  virtual void Train( void );

  // write weights to file
  virtual void WriteWeightsToFile( void );
  
  // read weights from file
  virtual void ReadWeightsFromFile( void  );

  // calculate the MVA value
  virtual Double_t GetMvaValue( TMVA_Event *e );

  // write method specific histos to target file
  virtual void WriteHistosToFile( void );

 public:

  // for root finder 
  static Double_t IGetVolumeContentForRoot( Double_t );
  Double_t         GetVolumeContentForRoot( Double_t );

  // static pointer to this object
  static TMVA_MethodPDERS* ThisPDERS( void ) { return fThisPDERS; }  

 protected:

  TMVA_Volume* fHelpVolume;
  Int_t        fFcnCall;

  // accessors
  TMVA_BinarySearchTree* GetBinaryTreeSig( void ) const { return fBinaryTreeS; }
  TMVA_BinarySearchTree* GetBinaryTreeBkg( void ) const { return fBinaryTreeB; }

  Double_t KernelEstimate( TMVA_Event&, std::vector<TMVA_Event*>&, TMVA_Volume& );

 private:

  enum VolumeRangeMode { MinMax = 0, RMS, Adaptive } fVRangeMode;

  TMVA_BinarySearchTree*       fBinaryTreeS;
  TMVA_BinarySearchTree*       fBinaryTreeB;

  vector<Float_t>*   fDelta;
  vector<Float_t>*   fShift;

  Float_t            fScaleS;
  Float_t            fScaleB;
  Float_t            fDeltaFrac;

  // input for adaptive volume adjustment
  Float_t            fNEventsMin;
  Float_t            fNEventsMax;
  Float_t            fMaxVIterations;
  Float_t            fInitialScale;          

  TFile*             fFin;

  Bool_t             fInitializedVolumeEle;

  void    SetVolumeElement ( void );
  Float_t RScalc           ( TMVA_Event *e );
  Float_t GetError         ( Float_t countS, Float_t countB, 
			     Float_t sumW2S, Float_t sumW2B ) const;

  // this carrier
  static TMVA_MethodPDERS* fThisPDERS;

  void InitPDERS( void );

  ClassDef(TMVA_MethodPDERS,0) //Multidimensional Likelihood using the "Probability density estimator range search" (PDERS) method  
};

#endif // TMVA_MethodPDERS_H
