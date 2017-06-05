// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::GiniIndex                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description: Implementation of the GiniIndex With Laplace correction           *
 *              as separation criterion                                           *
 *              Gini(Sample M) = 1 - (c(1)/N)^2 - (c(2)/N)^2 .... - (c(k)/N)^2    *
 *              Where: M is a sample of whatever N elements (events)              *
 *                     that belong to K different classes                         *
 *                     c(k) is the number of elements that belong to class k      *
 *              Laplace's correction to the prob.density c/N --> (c+1)/(N+2)      *
 *              for just Signal and Background classes this then boils down to:   *
 *              Gini(Sample) = 2(s*b+s+b+1)/(s+b+2)^2                             *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      Heidelberg U., Germany                                                    *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

/*! \class TMVA::GiniIndexWithLaplace
\ingroup TMVA

Implementation of the GiniIndex With Laplace correction as separation criterion

Large Gini Indices (maximum 0.5) mean , that the sample is well mixed (same
amount of signal and bkg) bkg.

Small Indices mean, well separated.

#### General definition:

\f[
Gini(Sample M) = 1 - (\frac{c(1)}{N})^2 - (\frac{c(2)}{N})^2 .... - (\frac{c(k)}{N})^2
\f]

Where:

\f$ M \f$ is a sample of whatever \f$ N \f$ elements (events) that belong
to \f$ K \f$ different classes.

\f$ c(k) \f$ is the number of elements that belong to class \f$ k \f$ for just
Signal and Background classes this boils down to:

The Laplace's correction to the probability distribution would turn the
\f$ \frac{c(1)}{N} \f$ into \f$ \frac{(c(1)+1)}{(N+2)} \f$ using this the
simple Gini Index  for two classes

\f[
Gini(Sample) = \frac{2sb}{(s+b)^2}
\f]

turns into

\f[
GiniLaplace(Sample) = \frac{2(sb+s+b+1)}{(s+b+2)^2}
\f]
*/

#include "TMVA/GiniIndexWithLaplace.h"

#include "Rtypes.h"

ClassImp(TMVA::GiniIndexWithLaplace);

////////////////////////////////////////////////////////////////////////////////

Double_t TMVA::GiniIndexWithLaplace::GetSeparationIndex( const Double_t s, const Double_t b )
{
   if (s+b <= 0)      return 0;
   if (s<=0 || b <=0) return 0;
   else               return (s*b+s+b+1)/(s+b+2)/(s+b+2);
}


