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
#include "RooAbsArg.h"

#include <sstream>

namespace RooHelpers {

/// Switches the message service to verbose while the instance alive.
class MakeVerbose {
  public:
    MakeVerbose(RooFit::MsgTopic extraTopics = static_cast<RooFit::MsgTopic>(0u)) {
      auto& msg = RooMsgService::instance();
      fOldKillBelow = msg.globalKillBelow();
      msg.setGlobalKillBelow(RooFit::DEBUG);
      fOldConf = msg.getStream(0);
      msg.getStream(0).minLevel= RooFit::DEBUG;
      msg.getStream(0).addTopic(extraTopics);
      msg.setStreamStatus(0, true);
    }

    ~MakeVerbose() {
      auto& msg = RooMsgService::instance();
      msg.setGlobalKillBelow(fOldKillBelow);
      msg.getStream(0) = fOldConf;
    }

  private:
    RooFit::MsgLevel fOldKillBelow;
    RooMsgService::StreamConfig fOldConf;
};


/// Hijacks all messages with given level and topic (and optionally object name) while alive.
/// Use this like an ostringstream afterwards. Useful for unit tests and debugging.
class HijackMessageStream : public std::ostringstream {
  public:
    HijackMessageStream(RooFit::MsgLevel level, RooFit::MsgTopic topics, const char* objectName = nullptr);

    virtual ~HijackMessageStream();

  private:
    RooFit::MsgLevel _oldKillBelow;
    std::vector<RooMsgService::StreamConfig> _oldConf;
    Int_t _thisStream;
};


std::vector<std::string> tokenise(const std::string &str, const std::string &delims);



class CachingError : public std::exception {
  public:
    CachingError(const std::string& newMessage) :
      std::exception(),
      _messages()
    {
      _messages.push_back(newMessage);
    }

    CachingError(CachingError&& previous, const std::string& newMessage) :
    std::exception(),
    _messages{std::move(previous._messages)}
    {
      _messages.push_back(newMessage);
    }

    const char* what() const noexcept override {
      std::stringstream out;
      out << "**Caching Error** in\n";

      std::string indent;
      for (auto it = _messages.rbegin(); it != _messages.rend(); ++it) {
        std::string message = *it;
        auto pos = message.find('\n', 0);
        while (pos != std::string::npos) {
          message.insert(pos+1, indent);
          pos = (message.find('\n', pos+1));
        }

        out << indent << message << "\n";
        indent += " ";
      }

      out << std::endl;

      std::string* ret = new std::string(out.str()); //Make it survive this method

      return ret->c_str();
    }


  private:
    std::vector<std::string> _messages;
};


class FormatPdfTree {
  public:
    template <class T,
    typename std::enable_if<std::is_base_of<RooAbsArg, T>::value>::type* = nullptr >
    FormatPdfTree& operator<<(const T& arg) {
      _stream << arg.ClassName() << "::" << arg.GetName() << " " << &arg << " ";
      arg.printArgs(_stream);
      return *this;
    }

    template <class T,
    typename std::enable_if< ! std::is_base_of<RooAbsArg, T>::value>::type* = nullptr >
    FormatPdfTree& operator<<(const T& arg) {
      _stream << arg;
      return *this;
    }

    operator std::string() const {
      return _stream.str();
    }

    std::ostream& stream() {
      return _stream;
    }

  private:
    std::ostringstream _stream;
};


/// Check if the parameters have a range, and warn if the range extends below / above the set limits.
void checkRangeOfParameters(const RooAbsReal* callingClass, std::initializer_list<const RooAbsReal*> pars,
    double min = -std::numeric_limits<double>::max(), double max = std::numeric_limits<double>::max());

}

#endif /* ROOFIT_ROOFITCORE_INC_ROOHELPERS_H_ */
