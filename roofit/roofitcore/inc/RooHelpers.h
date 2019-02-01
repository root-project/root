// Author: Stephan Hageboeck, CERN  01/2019

/*****************************************************************************
 * RooFit
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2019, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef ROOFIT_ROOFITCORE_INC_ROOHELPERS_H_
#define ROOFIT_ROOFITCORE_INC_ROOHELPERS_H_

#include "RooMsgService.h"

namespace RooHelpers {

class MakeVerbose {
  public:
    MakeVerbose() {
      auto& msg = RooMsgService::instance();
      fOldKillBelow = msg.globalKillBelow();
      msg.setGlobalKillBelow(RooFit::DEBUG);
      fOldConf = msg.getStream(0);
      msg.getStream(0).minLevel= RooFit::DEBUG;
      msg.setStreamStatus(0, true);
    }

    ~MakeVerbose() {
      auto& msg = RooMsgService::instance();
      msg.setGlobalKillBelow(fOldKillBelow);
      msg.getStream(0) = fOldConf;
      msg.setStreamStatus(0, true);
    }

  private:
    RooFit::MsgLevel fOldKillBelow;
    RooMsgService::StreamConfig fOldConf;
};

std::vector<std::string> tokenise(const std::string &str, const std::string &delims);

}

#endif /* ROOFIT_ROOFITCORE_INC_ROOHELPERS_H_ */
