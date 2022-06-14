// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : VariableDecorrTransform                                               *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Decorrelation of input variables                                          *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_VariableDecorrTransform
#define ROOT_TMVA_VariableDecorrTransform

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// VariableDecorrTransform                                              //
//                                                                      //
// Linear interpolation class                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMatrixDfwd.h"

#include "TMatrixDSymfwd.h"

#include "TMVA/VariableTransformBase.h"

#include <vector>

namespace TMVA {

   class VariableDecorrTransform : public VariableTransformBase {

   public:

      VariableDecorrTransform( DataSetInfo& dsi );
      virtual ~VariableDecorrTransform( void );

      void   Initialize();
      Bool_t PrepareTransformation (const std::vector<Event*>&);

      //      virtual const Event* Transform(const Event* const, Types::ESBType type = Types::kMaxSBType) const;
      virtual const Event* Transform(const Event* const, Int_t cls ) const;
      virtual const Event* InverseTransform(const Event* const, Int_t cls ) const;

      void WriteTransformationToStream ( std::ostream& ) const;
      void ReadTransformationFromStream( std::istream&, const TString& );

      virtual void AttachXMLTo(void* parent);
      virtual void ReadFromXML( void* trfnode );

      virtual void PrintTransformation( std::ostream & o );

      // writer of function code
      virtual void MakeFunction( std::ostream& fout, const TString& fncName, Int_t part, UInt_t trCounter, Int_t cls );

      // provides string vector giving explicit transformation
      std::vector<TString>* GetTransformationStrings( Int_t cls ) const;

   private:

      //      mutable Event*          fTransformedEvent;   ///<! local event copy
      std::vector<TMatrixD*>  fDecorrMatrices;     ///<! Decorrelation matrix [class0/class1/.../all classes]

      void CalcSQRMats( const std::vector< Event*>&, Int_t maxCls );
      std::vector<TMatrixDSym*>* CalcCovarianceMatrices( const std::vector<const Event*>& events, Int_t maxCls );

      ClassDef(VariableDecorrTransform,0); // Variable transformation: decorrelation
   };

} // namespace TMVA

#endif

