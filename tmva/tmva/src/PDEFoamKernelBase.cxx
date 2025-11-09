// @(#)root/tmva $Id$
// Author: Dominik Dannheim, Alexander Voigt

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: PDEFoamKernelBase                                                     *
 *                                             *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation of PDEFoam kernel interface                                *
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
 * (see tmva/doc/LICENSE)                                          *
 **********************************************************************************/

/*! \class TMVA::PDEFoamKernelBase
\ingroup TMVA
This class is the abstract kernel interface for PDEFoam.  The
kernel can be used for manipulating (smearing) the cell values of a
PDEFoam, by passing it as an argument to
PDEFoam::GetCellValue(...).

Derived classes must implement the Estimate() function to provide a
specific kernel behaviour.
*/

#include "TMVA/PDEFoamKernelBase.h"

#include "TMVA/MsgLogger.h"

#include "Rtypes.h"
#include "TObject.h"


////////////////////////////////////////////////////////////////////////////////
/// Default constructor for streamer

TMVA::PDEFoamKernelBase::PDEFoamKernelBase()
: TObject()
   , fLogger(new MsgLogger("PDEFoamKernelBase"))
{
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

TMVA::PDEFoamKernelBase::PDEFoamKernelBase(const PDEFoamKernelBase &other)
   : TObject()
   , fLogger(new MsgLogger(*other.fLogger))
{
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor

TMVA::PDEFoamKernelBase::~PDEFoamKernelBase()
{
   if (fLogger != NULL)
      delete fLogger;
}
