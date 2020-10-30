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
#include "RooAbsPdf.h"
#include "RooAbsData.h"
#include "RooDataHist.h"
#include "RooDataSet.h"
#include "RooAbsRealLValue.h"

#include "TClass.h"

namespace RooHelpers {

LocalChangeMsgLevel::LocalChangeMsgLevel(RooFit::MsgLevel lvl,
    unsigned int extraTopics, unsigned int removeTopics, bool overrideExternalLevel) {
  auto& msg = RooMsgService::instance();
  fOldKillBelow = msg.globalKillBelow();
  if (overrideExternalLevel) msg.setGlobalKillBelow(lvl);

  for (int i = 0; i < msg.numStreams(); ++i) {
    fOldConf.push_back(msg.getStream(i));
    if (overrideExternalLevel) msg.getStream(i).minLevel = lvl;
    msg.getStream(i).removeTopic(static_cast<RooFit::MsgTopic>(removeTopics));
    msg.setStreamStatus(i, true);
  }

  if (extraTopics != 0) {
    fExtraStream = msg.addStream(lvl);
    msg.getStream(fExtraStream).addTopic(static_cast<RooFit::MsgTopic>(extraTopics));
  }
}

LocalChangeMsgLevel::~LocalChangeMsgLevel() {
  auto& msg = RooMsgService::instance();
  msg.setGlobalKillBelow(fOldKillBelow);
  for (int i=0; i < msg.numStreams(); ++i) {
    if (i < static_cast<int>(fOldConf.size()))
      msg.getStream(i) = fOldConf[i];
  }

  if (fExtraStream > 0)
    msg.deleteStream(fExtraStream);
}


/// Tokenise the string by splitting at the characters in delims.
/// Consecutive delimiters are collapsed, so that no delimiters will appear in the
/// tokenised strings, and no emtpy strings are returned.
/// \param[in] str String to tokenise.
/// \param[in] delims One or more delimiters used to split the string.
/// \param[in] returnEmptyToken If the string is empty, return one empty token. Default is to return an empty vector.
std::vector<std::string> tokenise(const std::string &str, const std::string &delims, bool returnEmptyToken /*= true*/) {
  if (str.empty())
    return std::vector<std::string>(returnEmptyToken ? 1 : 0);

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


/// Hijack all messages with given level and topics while this object is alive.
/// \param[in] level Minimum level to hijack. Higher levels also get captured.
/// \param[in] topics Topics to hijack. Use `|` to combine different topics, and cast to `RooFit::MsgTopic` if necessary.
/// \param[in] objectName Only hijack messages from an object with the given name. Defaults to any object.
HijackMessageStream::HijackMessageStream(RooFit::MsgLevel level, RooFit::MsgTopic topics, const char* objectName)
{
  auto& msg = RooMsgService::instance();
  _oldKillBelow = msg.globalKillBelow();
  if (_oldKillBelow > level)
    msg.setGlobalKillBelow(level);

  std::vector<RooMsgService::StreamConfig> tmpStreams;
  for (int i = 0; i < msg.numStreams(); ++i) {
    _oldConf.push_back(msg.getStream(i));
    if (msg.getStream(i).match(level, topics, static_cast<RooAbsArg*>(nullptr))) {
      tmpStreams.push_back(msg.getStream(i));
      msg.setStreamStatus(i, false);
    }
  }

  _thisStream = msg.addStream(level,
      RooFit::Topic(topics),
      RooFit::OutputStream(_str),
      objectName ? RooFit::ObjectName(objectName) : RooCmdArg());

  for (RooMsgService::StreamConfig& st : tmpStreams) {
    msg.addStream(st.minLevel,
        RooFit::Topic(st.topic),
        RooFit::OutputStream(*st.os),
        RooFit::ObjectName(st.objectName.c_str()),
        RooFit::ClassName(st.className.c_str()),
        RooFit::BaseClassName(st.baseClassName.c_str()),
        RooFit::TagName(st.tagName.c_str()));
  }
}

/// Deregister the hijacked stream and restore the stream state of all previous streams.
HijackMessageStream::~HijackMessageStream() {
  auto& msg = RooMsgService::instance();
  msg.setGlobalKillBelow(_oldKillBelow);
  for (unsigned int i = 0; i < _oldConf.size(); ++i) {
    msg.getStream(i) = _oldConf[i];
  }

  while (_thisStream < msg.numStreams()) {
    msg.deleteStream(_thisStream);
  }
}


/// \param[in] callingClass Class that's calling. Needed to include name and type name of the class in error message.
/// \param[in] pars List of all parameters to be checked.
/// \param[in] min Minimum of allowed range. `min` itself counts as disallowed.
/// \param[in] max Maximum of allowed range. `max` itself counts as disallowed.
/// \param[in] limitsInAllowedRange If true, the limits passed as parameters are part of the allowed range.
/// \param[in] extraMessage Message that should be appended to the warning.
void checkRangeOfParameters(const RooAbsReal* callingClass, std::initializer_list<const RooAbsReal*> pars,
    double min, double max, bool limitsInAllowedRange, std::string const& extraMessage) {
  const char openBr = limitsInAllowedRange ? '[' : '(';
  const char closeBr = limitsInAllowedRange ? ']' : ')';

  for (auto parameter : pars) {
    auto par = dynamic_cast<const RooAbsRealLValue*>(parameter);
    if (par && (
        (par->getMin() < min || par->getMax() > max)
        || (!limitsInAllowedRange && (par->getMin() == min || par->getMax() == max)) )) {
      std::stringstream rangeMsg;
      rangeMsg << openBr;
      if (min > -std::numeric_limits<double>::max())
        rangeMsg << min << ", ";
      else
        rangeMsg << "-inf, ";

      if (max < std::numeric_limits<double>::max())
        rangeMsg << max << closeBr;
      else
        rangeMsg << "inf" << closeBr;

      oocoutW(callingClass, InputArguments) << "The parameter '" << par->GetName() << "' with range [" << par->getMin("") << ", "
          << par->getMax() << "] of the " << callingClass->IsA()->GetName() << " '" << callingClass->GetName()
          << "' exceeds the safe range of " << rangeMsg.str() << ". Advise to limit its range."
          << (!extraMessage.empty() ? "\n" : "") << extraMessage << std::endl;
    }
  }
}


namespace {
  std::pair<double, double> getBinningInterval(RooAbsBinning const& binning) {
    if (!binning.isParameterized()) {
      return {binning.lowBound(), binning.highBound()};
    } else {
      return {binning.lowBoundFunc()->getVal(), binning.highBoundFunc()->getVal()};
    }
  }
} // namespace


/// Get the lower and upper bound of parameter range if arg can be casted to RooAbsRealLValue.
/// If no range with rangeName is defined for the argument, this will check if a binning of the
/// same name exists and return the interval covered by the binning.
/// Returns `{-infinity, infinity}` if agument can't be casted to RooAbsRealLValue* or if no
/// range or binning with the requested name exists.
/// \param[in] arg RooAbsArg for which to get the range.
/// \param[in] rangeName The name of the range.
std::pair<double, double> getRangeOrBinningInterval(RooAbsArg const* arg, const char* rangeName) {
  auto rlv = dynamic_cast<RooAbsRealLValue const*>(arg);
  if (rlv) {
    if (rangeName && rlv->hasRange(rangeName)) {
      return {rlv->getMin(rangeName), rlv->getMax(rangeName)};
    } else if (auto binning = rlv->getBinningPtr(rangeName)) {
      return getBinningInterval(*binning);
    }
  }
  return {-std::numeric_limits<double>::infinity(), +std::numeric_limits<double>::infinity()};
}


/// Check if there is any overlap when a list of ranges is applied to a set of observables.
/// \param[in] arg RooAbsCollection with the observables to check for overlap.
/// \param[in] rangeName The names of the ranges.
bool checkIfRangesOverlap(RooAbsPdf const& pdf, RooAbsData const& data, std::vector<std::string> const& rangeNames) {

  auto observables = *pdf.getObservables(data);

  auto getLimits = [&](RooAbsRealLValue const& rlv, const char* rangeName) {
    
    // RooDataHistCase
    if(dynamic_cast<RooDataHist const*>(&data)) {
      if (auto binning = rlv.getBinningPtr(rangeName)) {
        return getBinningInterval(*binning);
      } else {
        // default binning if range is not defined
        return getBinningInterval(*rlv.getBinningPtr(nullptr));
      }
    }

    // RooDataSet and other cases
    if (rlv.hasRange(rangeName)) {
      return std::pair<double, double>{rlv.getMin(rangeName), rlv.getMax(rangeName)};
    }
    // default range if range with given name is not defined
    return std::pair<double, double>{rlv.getMin(), rlv.getMax()};
  };

  auto nObs = observables.size();
  auto nRanges = rangeNames.size();

  // cache the range limits in a flat vector
  std::vector<std::pair<double,double>> limits;
  limits.reserve(nRanges * nObs);

  for (auto const& range : rangeNames) {
    for (auto const& obs : observables) {
      auto rlv = dynamic_cast<RooAbsRealLValue const*>(obs);
      if(!rlv) {
        throw std::logic_error("Classes that represent observables are expected to inherit from RooAbsRealLValue!");
      }
      limits.push_back(getLimits(*rlv, range.c_str()));
    }
  }

  // loop over pairs of ranges
  for(size_t ir1 = 0; ir1 < nRanges; ++ir1) {
    for(size_t ir2 = ir1 + 1; ir2 < nRanges; ++ir2) {

      // Loop over observables. If all observables have overlapping limits for
      // these ranges, the hypercubes defining the range are overlapping and we
      // can return `true`.
      size_t overlaps = 0;
      for(size_t io1 = 0; io1 < nObs; ++io1) {
        auto r1 = limits[ir1 * nObs + io1];
        auto r2 = limits[ir2 * nObs + io1];
        overlaps += (r1.second > r2.first && r1.first < r2.second)
                 || (r2.second > r1.first && r2.first < r1.second);
      }
      if(overlaps == nObs) return true;
    }
  }

  return false;
}


}
