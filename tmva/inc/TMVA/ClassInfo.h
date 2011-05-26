// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : ClassInfo                                                             *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Contains all the data information                                         *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Peter Speckmayer <Peter.Speckmayer@cern.ch>  - CERN, Switzerland          *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2006:                                                            *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_ClassInfo
#define ROOT_TMVA_ClassInfo

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// ClassInfo                                                            //
//                                                                      //
// Class that contains all the information of a class                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TCut
#include "TCut.h"
#endif
#ifndef ROOT_TMatrixDfwd
#include "TMatrixDfwd.h"
#endif

namespace TMVA {
   
   class MsgLogger;

   class ClassInfo {

   public:
      
      ClassInfo( const TString& name = "default" );
      ~ClassInfo();

      // setters
      void               SetName  ( const TString& name   ) { fName = name; }
      void               SetWeight( const TString& weight ) { fWeight = weight; }
      void               SetCut   ( const TCut&    cut    ) { fCut = cut; }
      void               SetNumber( const UInt_t   index  ) { fNumber = index; }
      void               SetCorrelationMatrix( TMatrixD *matrix ) { fCorrMatrix = matrix; }

      // getters
      const TString&     GetName()              const { return fName; }
      const TString&     GetWeight()            const { return fWeight; }
      const TCut&        GetCut()               const { return fCut; }
      UInt_t             GetNumber()            const { return fNumber; }
      const TMatrixD*    GetCorrelationMatrix() const { return fCorrMatrix; }

   private:

      TString            fName;             //! name of the class
      TString            fWeight;           //! the input formula string that is the weight for the class
      TCut               fCut;              //! pretraining cut for the class
      UInt_t             fNumber;           //! index in of this class in vectors

      TMatrixD*          fCorrMatrix;       //! Correlation matrix for this class

      mutable MsgLogger* fLogger;   // message logger
      MsgLogger& Log() const { return *fLogger; }
   };
}

#endif
