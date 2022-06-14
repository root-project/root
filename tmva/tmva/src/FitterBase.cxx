// @(#)root/tmva $Id$
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

/*! \class TMVA::FitterBase
\ingroup TMVA

Base class for TMVA fitters. Also defines generalised fitting interface

*/

#include "TMVA/FitterBase.h"
#include "TMVA/Interval.h"
#include "TMVA/IFitterTarget.h"

#include "TMVA/Configurable.h"
#include "TMVA/MsgLogger.h"
#include "TMVA/Types.h"

#include "Rtypes.h"
#include "TString.h"

ClassImp(TMVA::FitterBase);

#ifdef _WIN32
/*Disable warning C4355: 'this' : used in base member initializer list*/
#pragma warning ( disable : 4355 )
#endif

////////////////////////////////////////////////////////////////////////////////
/// constructor

TMVA::FitterBase::FitterBase( IFitterTarget& target,
                              const TString& name,
                              const std::vector<Interval*> ranges,
                              const TString& theOption )
: Configurable( theOption ),
   fFitterTarget( target ),
   fRanges( ranges ),
   fNpars( ranges.size() ),
   fLogger( new MsgLogger("FitterBase", kINFO) ),
   fClassName( name )
{
   SetConfigName( GetName() );
   SetConfigDescription( "Configuration options for setup and tuning of specific fitter" );
}

////////////////////////////////////////////////////////////////////////////////
/// estimator function interface for fitting

Double_t TMVA::FitterBase::Run()
{
   std::vector<Double_t> pars;
   for (std::vector<Interval*>::const_iterator parIt = fRanges.begin(); parIt != fRanges.end(); ++parIt) {
      pars.push_back( (*parIt)->GetMean() );
   }

   //   delete fLogger;
   return this->Run( pars );
}

////////////////////////////////////////////////////////////////////////////////
/// estimator function interface for fitting

Double_t TMVA::FitterBase::EstimatorFunction( std::vector<Double_t>& parameters )
{
   return GetFitterTarget().EstimatorFunction( parameters );
}

