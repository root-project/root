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

#include "RooHelpers.h"
#include "RooAbsRealLValue.h"

namespace RooHelpers {

/// Tokenise the string by splitting at the characters in delims.
/// Consecutive delimiters are collapsed, so that no delimiters will appear in the
/// tokenised strings, and no emtpy strings are returned.
std::vector<std::string> tokenise(const std::string &str, const std::string &delims) {
  std::vector<std::string> tokens;

  auto beg = str.find_first_not_of(delims, 0);
  auto end = str.find_first_of(delims, beg);
  do {
    tokens.emplace_back(str.substr(beg, end-beg));
    beg = str.find_first_not_of(delims, end);
    end = str.find_first_of(delims, beg);
  } while (beg != std::string::npos);

  return tokens;
}



HijackMessageStream::HijackMessageStream(RooFit::MsgLevel level, RooFit::MsgTopic topics, const char* objectName) :
  std::ostringstream()
{
  auto& msg = RooMsgService::instance();
  _oldKillBelow = msg.globalKillBelow();
  msg.setGlobalKillBelow(level);
  for (int i = 0; i < msg.numStreams(); ++i) {
    _oldConf.push_back(msg.getStream(i));
    msg.getStream(i).removeTopic(topics);
    msg.setStreamStatus(i, true);
  }

  _thisStream = msg.addStream(level,
      RooFit::Topic(topics),
      RooFit::OutputStream(*this),
      objectName ? RooFit::ObjectName(objectName) : RooCmdArg());
}

HijackMessageStream::~HijackMessageStream() {
  auto& msg = RooMsgService::instance();
  msg.setGlobalKillBelow(_oldKillBelow);
  for (unsigned int i = 0; i < _oldConf.size(); ++i) {
    msg.getStream(i) = _oldConf[i];
  }
  msg.deleteStream(_thisStream);
}


/// \param[in] Class that's calling. Needed to include name and type name of the class in error message.
/// \param[in] pars List of all parameters to be checked.
/// \param[in] min Minimum of allowed range. `min` itself counts is disallowed.
/// \param[in] max Maximum of allowed range. `max` itself counts is disallowed.
void checkRangeOfParameters(const RooAbsReal* callingClass, std::initializer_list<const RooAbsReal*> pars,
    double min, double max) {
  for (auto parameter : pars) {
    auto par = dynamic_cast<const RooAbsRealLValue*>(parameter);
    if (par && (par->getMin() <= min || par->getMax() >= max) ) {
      std::stringstream rangeMsg;
      rangeMsg << "(";
      if (min > -std::numeric_limits<double>::max())
        rangeMsg << min << ", ";
      else
        rangeMsg << "-inf, ";

      if (max < std::numeric_limits<double>::max())
        rangeMsg << max << ")";
      else
        rangeMsg << "inf)";

      oocoutW(callingClass, InputArguments) << "The parameter '" << par->GetName() << "' with range [" << par->getMin("") << ", "
          << par->getMax() << "] of the " << callingClass->IsA()->GetName() << " '" << callingClass->GetName()
          << "' exceeds the safe range of " << rangeMsg.str() << ". Advise to limit its range." << std::endl;
    }
  }
}

}
