// @(#)root/tmva $Id$
// Author: Dominik Dannheim, Alexander Voigt

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: PDEFoamKernelTrivial                                                  *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation of trivial PDEFoam kernel                                  *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      S. Jadach        - Institute of Nuclear Physics, Cracow, Poland           *
 *      Tancredi Carli   - CERN, Switzerland                                      *
 *      Dominik Dannheim - CERN, Switzerland                                      *
 *      Alexander Voigt  - TU Dresden, Germany                                    *
 *                                                                                *
 * Copyright (c) 2010:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

/*! \class TMVA::PDEFoamKernelTrivial
\ingroup TMVA
This class is a trivial PDEFoam kernel estimator.  The Estimate()
function returns the cell value, given an event 'txvec'.
*/

#include "TMVA/PDEFoamKernelTrivial.h"

#include "TMVA/MsgLogger.h"
#include "TMVA/PDEFoam.h"
#include "TMVA/PDEFoamKernelBase.h"
#include "TMVA/Types.h"

#include "Rtypes.h"

#include <vector>

ClassImp(TMVA::PDEFoamKernelTrivial);

////////////////////////////////////////////////////////////////////////////////
/// Default constructor for streamer

TMVA::PDEFoamKernelTrivial::PDEFoamKernelTrivial()
: PDEFoamKernelBase()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

TMVA::PDEFoamKernelTrivial::PDEFoamKernelTrivial(const PDEFoamKernelTrivial &other)
   : PDEFoamKernelBase(other)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Simple kernel estimator.  It returns the cell value 'cv',
/// corresponding to the event vector 'txvec' (in foam coordinates).
///
/// Parameters:
///
/// - foam - the pdefoam to search in
///
/// - txvec - event vector in foam coordinates [0,1]
///
/// - cv - cell value to estimate

Float_t TMVA::PDEFoamKernelTrivial::Estimate(PDEFoam *foam, std::vector<Float_t> &txvec, ECellValue cv)
{
   if (foam == NULL)
      Log() << kFATAL << "<PDEFoamKernelTrivial::Estimate>: PDEFoam not set!" << Endl;

   return foam->GetCellValue(foam->FindCell(txvec), cv);
}
