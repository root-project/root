// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : VariableIdentityTransform                                             *
 *                                             *
 *                                                                                *
 * Description:                                                                   *
 *      Identity transform                                                        *
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
 * (see tmva/doc/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_VariableIdentityTransform
#define ROOT_TMVA_VariableIdentityTransform

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// VariableIdentityTransform                                            //
//                                                                      //
// Linear interpolation class                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMVA/VariableTransformBase.h"

#include <vector>

namespace TMVA {

   class VariableIdentityTransform : public VariableTransformBase {

   public:

      VariableIdentityTransform( DataSetInfo& dsi );
      virtual ~VariableIdentityTransform( void ) {}

      void   Initialize() override;
      Bool_t PrepareTransformation (const std::vector<Event*>& ) override;

      void WriteTransformationToStream ( std::ostream& ) const override {}
      void ReadTransformationFromStream( std::istream&, const TString& ) override { SetCreated(); }

      void AttachXMLTo(void* parent) override;
      void ReadFromXML( void* trfnode ) override;

      const Event* Transform(const Event* const, Int_t cls ) const override;
      const Event* InverseTransform(const Event* const ev, Int_t cls ) const override { return Transform( ev, cls ); }

      // writer of function code
      void MakeFunction(std::ostream& fout, const TString& fncName, Int_t part, UInt_t trCounter, Int_t cls ) override;

      ClassDefOverride(VariableIdentityTransform,0); // Variable transformation: identity
   };

} // namespace TMVA

#endif
