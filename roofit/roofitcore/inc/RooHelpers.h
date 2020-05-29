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
#include "RooAbsReal.h"

#include <sstream>
#include <vector>
#include <string>

namespace RooHelpers {

/// Switches the message service to a different level while the instance is alive.
/// Can also temporarily activate / deactivate message topics.
/// Use as
/// ~~~{.cpp}
/// RooHelpers::LocalChangeMessageLevel changeMsgLvl(RooFit::WARNING);
/// [ statements that normally generate a lot of output ]
/// ~~~
class LocalChangeMsgLevel {
  public:
    /// Change message level (and topics) while this object is alive, reset when it goes out of scope.
    /// \param[in] lvl The desired message level. Defaults to verbose.
    /// \param[in] extraTopics Extra topics to be switched on. These will only switched on in the last stream to prevent all streams are printing.
    /// \param[in] removeTopics Message topics to be switched off
    /// \param[in] overrideExternalLevel Override the user message level.
    LocalChangeMsgLevel(RooFit::MsgLevel lvl = RooFit::DEBUG,
        unsigned int extraTopics = 0u,
        unsigned int removeTopics = 0u,
        bool overrideExternalLevel = true);

    ~LocalChangeMsgLevel();

  private:
    RooFit::MsgLevel fOldKillBelow;
    std::vector<RooMsgService::StreamConfig> fOldConf;
    int fExtraStream{-1};
};


/// Hijacks all messages with given level and topic (and optionally object name) while alive.
/// Use this like an ostringstream afterwards. The messages can e.g. be retrieved using `str()`.
/// Useful for unit tests / debugging.
class HijackMessageStream : public std::ostringstream {
  public:
    HijackMessageStream(RooFit::MsgLevel level, RooFit::MsgTopic topics, const char* objectName = nullptr);

    virtual ~HijackMessageStream();

  private:
    RooFit::MsgLevel _oldKillBelow;
    std::vector<RooMsgService::StreamConfig> _oldConf;
    Int_t _thisStream;
};


std::vector<std::string> tokenise(const std::string &str, const std::string &delims, bool returnEmptyToken = true);



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
    double min = -std::numeric_limits<double>::max(), double max = std::numeric_limits<double>::max(),
    bool limitsInAllowedRange = false, std::string extraMessage = "");


/// Helper class to access a batch-related part of RooAbsReal's interface, which should not leak to the outside world.
class BatchInterfaceAccessor {
  public:
    static void clearBatchMemory(RooAbsReal& theReal) {
      theReal.clearBatchMemory();
    }

    static void checkBatchComputation(const RooAbsReal& theReal, std::size_t evtNo,
        const RooArgSet* normSet = nullptr, double relAccuracy = 1.E-13) {
      theReal.checkBatchComputation(evtNo, normSet, relAccuracy);
    }
};


}


#endif /* ROOFIT_ROOFITCORE_INC_ROOHELPERS_H_ */
