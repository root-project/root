// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Peter Speckmayer, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : ClassInfo                                                             *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Peter Speckmayer <Peter.Speckmayer@cern.ch>  - CERN, Switzerland          *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2006:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

/*! \class TMVA::ClassInfo
\ingroup TMVA

Class that contains all the information of a class.

*/

#include "TMVA/ClassInfo.h"

#include <vector>

#include "TCut.h"
#include "TMatrixD.h"

#include "TMVA/MsgLogger.h"
#include "TMVA/Types.h"

////////////////////////////////////////////////////////////////////////////////
/// constructor

TMVA::ClassInfo::ClassInfo( const TString& name )
    : TNamed(name.Data(),name.Data()),
     fWeight( "" ),
     fCut( "" ),
     fNumber( 0 ),
     fCorrMatrix( 0 ),
     fLogger( new MsgLogger("ClassInfo", kINFO) )
{
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TMVA::ClassInfo::~ClassInfo()
{
   if (fCorrMatrix) delete fCorrMatrix;
   delete fLogger;
}



