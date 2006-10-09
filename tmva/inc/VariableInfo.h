// @(#)root/tmva $Id: VariableInfo.h,v 1.12 2006/10/04 22:29:27 andreas.hoecker Exp $   
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Option                                                                *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Variable type info                                                        *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *                                                                                *
 * Copyright (c) 2006:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-KP Heidelberg, Germany,                                               * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 **********************************************************************************/

#ifndef ROOT_TMVA_VariableInfo
#define ROOT_TMVA_VariableInfo

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// VariableInfo                                                         //
//                                                                      //
// Class for type info of MVA input variable                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TMVA_Types
#include "TMVA/Types.h"
#endif

namespace TMVA {

   class VariableInfo {

   public:

      VariableInfo(const TString& expression, int varCounter, char varTypeOriginal = 'F', void * external = 0);
      VariableInfo();
      ~VariableInfo() {}
      const TString& GetExpression()       const { return fExpression; }
      const TString& GetInternalVarName()  const { return fInternalVarName; }
      char           VarType()             const { return fVarType; }
      char           VarTypeOriginal()     const { return fVarTypeOriginal; }
      Double_t       GetMin (Types::PreprocessingMethod corr = Types::kNone) const { return fXminNorm[(Int_t) corr]; }
      Double_t       GetMax (Types::PreprocessingMethod corr = Types::kNone) const { return fXmaxNorm[(Int_t) corr]; }
      Double_t       GetMean(Types::PreprocessingMethod corr = Types::kNone) const { return fXmeanNorm[(Int_t) corr]; }
      Double_t       GetRMS (Types::PreprocessingMethod corr = Types::kNone) const { return fXrmsNorm[(Int_t) corr]; }

      void           SetExpression(const TString& s)      { fExpression = s; }
      void           SetInternalVarName(const TString& s) { fInternalVarName = s; }
      void           SetVarType(char c)                    { fVarType = c; }
      void           SetMin (Double_t v, Types::PreprocessingMethod corr = Types::kNone) { fXminNorm[(Int_t) corr] = v; }
      void           SetMax (Double_t v, Types::PreprocessingMethod corr = Types::kNone) { fXmaxNorm[(Int_t) corr] = v; }
      void           SetMean(Double_t v, Types::PreprocessingMethod corr = Types::kNone) { fXmeanNorm[(Int_t) corr] = v; }
      void           SetRMS (Double_t v, Types::PreprocessingMethod corr = Types::kNone) { fXrmsNorm[(Int_t) corr] = v; }
      void           SetExternalLink(void* p) { fExternalData = p; }

      void           WriteToStream(std::ostream& o, Types::PreprocessingMethod corr) const;
      void           ReadFromStream(std::istream& istr, Types::PreprocessingMethod corr);
      void*          GetExternalLink() const { return fExternalData; }

      // assignment operator (does not copy external link)
      VariableInfo&   operator=(const TMVA::VariableInfo& rhs);
   private:

      TString  fExpression;      //! 
      TString  fInternalVarName; //! 
      Char_t   fVarType;         //! the variable type to be used internally ('F'-default or 'I')
      Char_t   fVarTypeOriginal; //! the variable type to be used internally ('F'-default or 'I')
      Double_t fXminNorm[3];     //! minimum value for correlated/decorrelated/PCA variable
      Double_t fXmaxNorm[3];     //! maximum value for correlated/decorrelated/PCA variable
      Double_t fXmeanNorm[3];    //! mean value for correlated/decorrelated/PCA variable
      Double_t fXrmsNorm[3];     //! rms value for correlated/decorrelated/PCA variable
      void*    fExternalData;    //! if the variable content is linked to an external pointer      
      Int_t    fVarCounter;      //! dummy variable
   };

}

#endif
