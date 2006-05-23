// @(#)root/tmva $Id: MethodPDERS.h,v 1.5 2006/05/23 09:53:10 stelzer Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodPDERS                                                           *
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
 * $Id: MethodPDERS.h,v 1.5 2006/05/23 09:53:10 stelzer Exp $
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodPDERS
#define ROOT_TMVA_MethodPDERS

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodPDERS                                                          //
//                                                                      //
// Multidimensional Likelihood using the "Probability density           //
// estimator range search" (PDERS) method                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif
#ifndef ROOT_TMVA_BinarySearchTree
#include "TMVA/BinarySearchTree.h"
#endif
#ifndef ROOT_TMVA_TVector
#include "TVector.h"
#endif

namespace TMVA {

   class Volume;
   class Event;

   class MethodPDERS : public MethodBase {

   public:

      MethodPDERS( TString jobName,
                   vector<TString>* theVariables,
                   TTree* theTree = 0,
                   TString theOption = "Adaptive:100:200:50:0.99",
                   TDirectory* theTargetDir = 0 );

      MethodPDERS( vector<TString> *theVariables,
                   TString theWeightFile,
                   TDirectory* theTargetDir = NULL );

      virtual ~MethodPDERS( void );

      // training method
      virtual void Train( void );

      // write weights to file
      virtual void WriteWeightsToFile( void );

      // read weights from file
      virtual void ReadWeightsFromFile( void  );

      // calculate the MVA value
      virtual Double_t GetMvaValue( Event *e );

      // write method specific histos to target file
      virtual void WriteHistosToFile( void );

   public:

      // for root finder
      static Double_t IGetVolumeContentForRoot( Double_t );
      Double_t         GetVolumeContentForRoot( Double_t );

      // static pointer to this object
      static MethodPDERS* ThisPDERS( void ) { return fgThisPDERS; }

   protected:

      Volume* fHelpVolume;
      Int_t        fFcnCall;

      // accessors
      BinarySearchTree* GetBinaryTreeSig( void ) const { return fBinaryTreeS; }
      BinarySearchTree* GetBinaryTreeBkg( void ) const { return fBinaryTreeB; }

      Double_t KernelEstimate( Event&, std::vector<Event*>&, Volume& );

   private:

      enum VolumeRangeMode { kMinMax = 0, kRMS, kAdaptive } fVRangeMode;

      BinarySearchTree*       fBinaryTreeS;
      BinarySearchTree*       fBinaryTreeB;

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
      Float_t RScalc           ( Event *e );
      Float_t GetError         ( Float_t countS, Float_t countB,
                                 Float_t sumW2S, Float_t sumW2B ) const;

      // this carrier
      static MethodPDERS* fgThisPDERS;

      void InitPDERS( void );

      ClassDef(MethodPDERS,0) //Multidimensional Likelihood using the "Probability density estimator range search" (PDERS) method
         };

} // namespace TMVA

#endif // MethodPDERS_H
