// @(#)root/tmva $Id: MethodSVM.h,v 1.9 2006/08/30 22:19:59 andreas.hoecker Exp $    
// Author: Marcin ....

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodSVM                                                             *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Support Vector Machines
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Marcin ... and student                                                    *            
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-KP Heidelberg, Germany                                                * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodSVM
#define ROOT_TMVA_MethodSVM

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodSVM                                                            //
//                                                                      //
// Friedman's SVM method -- not yet implemented -- dummy class --       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif
#ifndef ROOT_TMVA_TMatrixD
#include "TMatrixD.h"
#endif
#ifndef ROOT_TMVA_TVectorD
#include "TVectorD.h"
#endif

namespace TMVA {

   class MethodSVM : public MethodBase {

   public:

      MethodSVM( TString jobName, 
                 TString methodTitle, 
                 DataSet& theData,
                 TString theOption = "",
                 TDirectory* theTargetDir = 0 );
      
      MethodSVM( DataSet& theData, 
                 TString theWeightFile,  
                 TDirectory* theTargetDir = NULL );

      virtual ~MethodSVM( void );
    
      // training method
      virtual void Train( void );

      // write weights to file
      virtual void WriteWeightsToStream( ostream& o ) const;

      // read weights from file
      virtual void ReadWeightsFromStream( istream& istr );

      // calculate the MVA value
      virtual Double_t GetMvaValue();

      // write method specific histos to target file
      virtual void WriteHistosToFile( void ) const;

      void InitSVM( void );

      // ranking of input variables
      const Ranking* CreateRanking() { return 0; }

   private:

      // the option handling methods
      virtual void DeclareOptions();
      virtual void ProcessOptions();

      ClassDef(MethodSVM,0)  // Friedman's SVM method 
   };

} // namespace TMVA

#endif // MethodSVM_H
