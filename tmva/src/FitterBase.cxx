// @(#)root/tmva $Id: FitterBase.cxx,v 1.16 2007/06/07 12:29:30 stelzer Exp $ 
// Author: Andreas Hoecker, Peter Speckmayer, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : FitterBase                                                            *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation                                                            *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker  <Andreas.Hocker@cern.ch> - CERN, Switzerland             *
 *      Peter Speckmayer <speckmay@mail.cern.ch>  - CERN, Switzerland             *
 *      Joerg Stelzer    <Joerg.Stelzer@cern.ch>  - CERN, Switzerland             *
 *      Helge Voss       <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany     *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
//                                                                      
//  Baseclass for TMVA fitters
//  Also defines generalised fitting interface
//_______________________________________________________________________

#include "TMVA/FitterBase.h"
#ifndef ROOT_TMVA_Interval
#include "TMVA/Interval.h"
#endif
#ifndef ROOT_TMVA_IFitterTarget
#include "TMVA/IFitterTarget.h"
#endif

ClassImp(TMVA::FitterBase)

//_______________________________________________________________________
TMVA::FitterBase::FitterBase( IFitterTarget& target, 
                              const TString& name, 
                              const std::vector<Interval*> ranges, 
                              const TString& theOption ) 
   : Configurable( theOption ),
     fFitterTarget( target ),
     fRanges( ranges ),
     fNpars( ranges.size() ),
     fLogger( this ),
     fClassName( name )
{
   // constructor   
}

//_______________________________________________________________________
Double_t TMVA::FitterBase::Run()
{
   // estimator function interface for fitting 
   std::vector<Double_t> pars;
   for (std::vector<Interval*>::const_iterator parIt = fRanges.begin(); parIt != fRanges.end(); parIt++) {
      pars.push_back( (*parIt)->GetMean() );
   }
                                                                   
   return this->Run( pars );
}

//_______________________________________________________________________
Double_t TMVA::FitterBase::EstimatorFunction( std::vector<Double_t>& parameters )
{
   // estimator function interface for fitting 
   return GetFitterTarget().EstimatorFunction( parameters );
}

