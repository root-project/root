/*
 * Project: RooFit
 * Author:
 *   Stephan Hageboeck, CERN 2019
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include <RooHelpers.h>

#include <RooAbsCategory.h>
#include <RooAbsData.h>
#include <RooAbsPdf.h>
#include <RooAbsRealLValue.h>
#include <RooArgList.h>
#include <RooDataHist.h>
#include <RooDataSet.h>
#include <RooProdPdf.h>
#include <RooRealSumPdf.h>
#include <RooSimultaneous.h>

#include <ROOT/StringUtils.hxx>
#include <TClass.h>

#include <unordered_map>

namespace RooHelpers {

LocalChangeMsgLevel::LocalChangeMsgLevel(RooFit::MsgLevel lvl, unsigned int extraTopics, unsigned int removeTopics,
                                         bool overrideExternalLevel)
{
   auto &msg = RooMsgService::instance();
   fOldKillBelow = msg.globalKillBelow();
   if (overrideExternalLevel)
      msg.setGlobalKillBelow(lvl);

   for (int i = 0; i < msg.numStreams(); ++i) {
      fOldConf.push_back(msg.getStream(i));
      if (overrideExternalLevel)
         msg.getStream(i).minLevel = lvl;
      msg.getStream(i).removeTopic(static_cast<RooFit::MsgTopic>(removeTopics));
      msg.setStreamStatus(i, true);
   }

   if (extraTopics != 0) {
      fExtraStream = msg.addStream(lvl);
      msg.getStream(fExtraStream).addTopic(static_cast<RooFit::MsgTopic>(extraTopics));
   }
}

LocalChangeMsgLevel::~LocalChangeMsgLevel()
{
   auto &msg = RooMsgService::instance();
   msg.setGlobalKillBelow(fOldKillBelow);
   for (int i = 0; i < msg.numStreams(); ++i) {
      if (i < static_cast<int>(fOldConf.size()))
         msg.getStream(i) = fOldConf[i];
   }

   if (fExtraStream > 0)
      msg.deleteStream(fExtraStream);
}

/// Hijack all messages with given level and topics while this object is alive.
/// \param[in] level Minimum level to hijack. Higher levels also get captured.
/// \param[in] topics Topics to hijack. Use `|` to combine different topics, and cast to `RooFit::MsgTopic` if
/// necessary. \param[in] objectName Only hijack messages from an object with the given name. Defaults to any object.
HijackMessageStream::HijackMessageStream(RooFit::MsgLevel level, RooFit::MsgTopic topics, const char *objectName)
{
   auto &msg = RooMsgService::instance();
   _oldKillBelow = msg.globalKillBelow();
   if (_oldKillBelow > level)
      msg.setGlobalKillBelow(level);

   std::vector<RooMsgService::StreamConfig> tmpStreams;
   for (int i = 0; i < msg.numStreams(); ++i) {
      _oldConf.push_back(msg.getStream(i));
      if (msg.getStream(i).match(level, topics, static_cast<RooAbsArg *>(nullptr))) {
         tmpStreams.push_back(msg.getStream(i));
         msg.setStreamStatus(i, false);
      }
   }

   _thisStream = msg.addStream(level, RooFit::Topic(topics), RooFit::OutputStream(_str),
                               objectName ? RooFit::ObjectName(objectName) : RooCmdArg());

   for (RooMsgService::StreamConfig &st : tmpStreams) {
      msg.addStream(st.minLevel, RooFit::Topic(st.topic), RooFit::OutputStream(*st.os),
                    RooFit::ObjectName(st.objectName.c_str()), RooFit::ClassName(st.className.c_str()),
                    RooFit::BaseClassName(st.baseClassName.c_str()), RooFit::TagName(st.tagName.c_str()));
   }
}

/// Deregister the hijacked stream and restore the stream state of all previous streams.
HijackMessageStream::~HijackMessageStream()
{
   auto &msg = RooMsgService::instance();
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
void checkRangeOfParameters(const RooAbsReal *callingClass, std::initializer_list<const RooAbsReal *> pars, double min,
                            double max, bool limitsInAllowedRange, std::string const &extraMessage)
{
   const char openBr = limitsInAllowedRange ? '[' : '(';
   const char closeBr = limitsInAllowedRange ? ']' : ')';

   for (auto parameter : pars) {
      auto par = dynamic_cast<const RooAbsRealLValue *>(parameter);
      if (par && ((par->getMin() < min || par->getMax() > max) ||
                  (!limitsInAllowedRange && (par->getMin() == min || par->getMax() == max)))) {
         std::stringstream rangeMsg;
         rangeMsg << openBr;
         if (min > -std::numeric_limits<double>::max()) {
            rangeMsg << min << ", ";
         } else {
            rangeMsg << "-inf, ";
         }

         if (max < std::numeric_limits<double>::max()) {
            rangeMsg << max << closeBr;
         } else {
            rangeMsg << "inf" << closeBr;
         }

         oocoutW(callingClass, InputArguments)
            << "The parameter '" << par->GetName() << "' with range [" << par->getMin("") << ", " << par->getMax()
            << "] of the " << callingClass->ClassName() << " '" << callingClass->GetName()
            << "' exceeds the safe range of " << rangeMsg.str() << ". Advise to limit its range."
            << (!extraMessage.empty() ? "\n" : "") << extraMessage << std::endl;
      }
   }
}

} // namespace RooHelpers
