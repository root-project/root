// @(#)root/tmva $Id$
// Author: Matt Jachowski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::TNeuronInputSqSum                                               *
 *                                             *
 *                                                                                *
 * Description:                                                                   *
 *       TNeuron input calculator -- calculates the square                        *
 *       of the weighted sum of inputs.                                           *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Matt Jachowski  <jachowski@stanford.edu> - Stanford University, USA       *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (see tmva/doc/LICENSE)                                          *
 **********************************************************************************/

/*! \class TMVA::TNeuronInputSqSum
\ingroup TMVA
TNeuron input calculator -- calculates the squared weighted sum of inputs.
*/

#include "TMVA/TNeuronInputSqSum.h"

#include "Rtypes.h"

