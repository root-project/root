/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
 * Authors:                                                                  *
 *   PB, Patrick Bos,     NL eScience Center, p.bos@esciencecenter.nl        *
 *                                                                           *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#include <RooMinimizerType.h>

namespace RooFit {
  std::string minimizer_type(MinimizerType type) {
    switch (type) {
      case MinimizerType::Minuit:
        return "Minuit";
      case MinimizerType::Minuit2:
        return "Minuit2";
    }
  }
}
