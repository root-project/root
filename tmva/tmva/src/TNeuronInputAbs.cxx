// @(#)root/tmva $Id$
// Author: Matt Jachowski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::TNeuronInputAbs                                                 *
 *                                             *
 *                                                                                *
 * Description:                                                                   *
 *      TNeuron input calculator -- calculates the sum of the absolute values     *
 *      of the weighted inputs                                                    *
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

/*! \class TMVA::TNeuronInputAbs
\ingroup TMVA
TNeuron input calculator -- calculates the sum of the absolute
values of the weighted inputs
*/

#include "TMVA/TNeuronInputAbs.h"

#include "Rtypes.h"

