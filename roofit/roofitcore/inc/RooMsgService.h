/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooMsgService.h,v 1.2 2007/07/13 21:50:24 wouter Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_MSG_SERVICE
#define ROO_MSG_SERVICE

#include <RooCmdArg.h>
#include <RooGlobalFunc.h>

#include <TObject.h>

#include <cstddef>
#include <string>
#include <vector>
#include <stack>
#include <map>

class RooAbsArg ;
class RooWorkspace ;

// Shortcut definitions
#define coutI(a) RooMsgService::instance().log(this,RooFit::INFO,RooFit::a)
#define coutP(a) RooMsgService::instance().log(this,RooFit::PROGRESS,RooFit::a)
#define coutW(a) RooMsgService::instance().log(this,RooFit::WARNING,RooFit::a)
#define coutE(a) RooMsgService::instance().log(this,RooFit::ERROR,RooFit::a)
#define coutF(a) RooMsgService::instance().log(this,RooFit::FATAL,RooFit::a)

// Skip the message prefix
#define ccoutD(a) RooMsgService::instance().log(this,RooFit::DEBUG,RooFit::a,true)
#define ccoutI(a) RooMsgService::instance().log(this,RooFit::INFO,RooFit::a,true)
#define ccoutP(a) RooMsgService::instance().log(this,RooFit::PROGRESS,RooFit::a,true)
#define ccoutW(a) RooMsgService::instance().log(this,RooFit::WARNING,RooFit::a,true)
#define ccoutE(a) RooMsgService::instance().log(this,RooFit::ERROR,RooFit::a,true)
#define ccoutF(a) RooMsgService::instance().log(this,RooFit::FATAL,RooFit::a,true)

// Message from given object instead of "this"
#define oocoutI(o,a) RooMsgService::instance().log(o,RooFit::INFO,RooFit::a)
#define oocoutP(o,a) RooMsgService::instance().log(o,RooFit::PROGRESS,RooFit::a)
#define oocoutW(o,a) RooMsgService::instance().log(o,RooFit::WARNING,RooFit::a)
#define oocoutE(o,a) RooMsgService::instance().log(o,RooFit::ERROR,RooFit::a)
#define oocoutF(o,a) RooMsgService::instance().log(o,RooFit::FATAL,RooFit::a)

// Message from given object instead of "this" and skip message prefix
#define ooccoutD(o,a) RooMsgService::instance().log(o,RooFit::DEBUG,RooFit::a,true)
#define ooccoutI(o,a) RooMsgService::instance().log(o,RooFit::INFO,RooFit::a,true)
#define ooccoutP(o,a) RooMsgService::instance().log(o,RooFit::PROGRESS,RooFit::a,true)
#define ooccoutW(o,a) RooMsgService::instance().log(o,RooFit::WARNING,RooFit::a,true)
#define ooccoutE(o,a) RooMsgService::instance().log(o,RooFit::ERROR,RooFit::a,true)
#define ooccoutF(o,a) RooMsgService::instance().log(o,RooFit::FATAL,RooFit::a,true)

#ifndef _WIN32
#define ANYDEBUG (RooMsgService::_debugCount>0)
#else
#define ANYDEBUG (RooMsgService::anyDebug())
#endif

#define dologD(a) (ANYDEBUG && RooMsgService::instance().isActive(this,RooFit::a,RooFit::DEBUG))
#define dologI(a) (RooMsgService::instance().isActive(this,RooFit::a,RooFit::INFO))
#define dologP(a) (RooMsgService::instance().isActive(this,RooFit::a,RooFit::PROGRESS))
#define dologW(a) (RooMsgService::instance().isActive(this,RooFit::a,RooFit::WARNING))
#define dologE(a) (RooMsgService::instance().isActive(this,RooFit::a,RooFit::ERROR))
#define dologF(a) (RooMsgService::instance().isActive(this,RooFit::a,RooFit::FATAL))

#define oodologD(o,a) (ANYDEBUG && RooMsgService::instance().isActive(o,RooFit::a,RooFit::DEBUG))
#define oodologI(o,a) (RooMsgService::instance().isActive(o,RooFit::a,RooFit::INFO))
#define oodologP(o,a) (RooMsgService::instance().isActive(o,RooFit::a,RooFit::PROGRESS))
#define oodologW(o,a) (RooMsgService::instance().isActive(o,RooFit::a,RooFit::WARNING))
#define oodologE(o,a) (RooMsgService::instance().isActive(o,RooFit::a,RooFit::ERROR))
#define oodologF(o,a) (RooMsgService::instance().isActive(o,RooFit::a,RooFit::FATAL))

// Shortcuts definitions with conditional execution of print expression -- USE WITH CAUTION

#define cxcoutD(a) if (ANYDEBUG && RooMsgService::instance().isActive(this,RooFit::a,RooFit::DEBUG)) RooMsgService::instance().log(this,RooFit::DEBUG,RooFit::a)
#define ccxcoutD(a) if (ANYDEBUG && RooMsgService::instance().isActive(this,RooFit::a,RooFit::DEBUG)) RooMsgService::instance().log(this,RooFit::DEBUG,RooFit::a,true)
#define oocxcoutD(o,a) if (ANYDEBUG && RooMsgService::instance().isActive(o,RooFit::a,RooFit::DEBUG)) RooMsgService::instance().log(o,RooFit::DEBUG,RooFit::a)
#define ooccxcoutD(o,a) if (ANYDEBUG && RooMsgService::instance().isActive(o,RooFit::a,RooFit::DEBUG)) RooMsgService::instance().log(o,RooFit::DEBUG,RooFit::a,true)
#define cxcoutI(a) if (RooMsgService::instance().isActive(this,RooFit::a,RooFit::INFO)) RooMsgService::instance().log(this,RooFit::INFO,RooFit::a)
#define ccxcoutI(a) if (RooMsgService::instance().isActive(this,RooFit::a,RooFit::INFO)) RooMsgService::instance().log(this,RooFit::INFO,RooFit::a,true)
#define oocxcoutI(o,a) if (RooMsgService::instance().isActive(o,RooFit::a,RooFit::INFO)) RooMsgService::instance().log(o,RooFit::INFO,RooFit::a)
#define ooccxcoutI(o,a) if (RooMsgService::instance().isActive(o,RooFit::a,RooFit::INFO)) RooMsgService::instance().log(o,RooFit::INFO,RooFit::a,true)
#define cxcoutP(a) if (RooMsgService::instance().isActive(this,RooFit::a,RooFit::PROGRESS)) RooMsgService::instance().log(this,RooFit::PROGRESS,RooFit::a)
#define ccxcoutP(a) if (RooMsgService::instance().isActive(this,RooFit::a,RooFit::PROGRESS)) RooMsgService::instance().log(this,RooFit::PROGRESS,RooFit::a,true)
#define oocxcoutP(o,a) if (RooMsgService::instance().isActive(o,RooFit::a,RooFit::PROGRESS)) RooMsgService::instance().log(o,RooFit::PROGRESS,RooFit::a)
#define ooccxcoutP(o,a) if (RooMsgService::instance().isActive(o,RooFit::a,RooFit::PROGRESS)) RooMsgService::instance().log(o,RooFit::PROGRESS,RooFit::a,true)
#define cxcoutW(a) if (RooMsgService::instance().isActive(this,RooFit::a,RooFit::WARNING)) RooMsgService::instance().log(this,RooFit::WARNING,RooFit::a)
#define ccxcoutW(a) if (RooMsgService::instance().isActive(this,RooFit::a,RooFit::WARNING)) RooMsgService::instance().log(this,RooFit::WARNING,RooFit::a,true)
#define oocxcoutW(o,a) if (RooMsgService::instance().isActive(o,RooFit::a,RooFit::WARNING)) RooMsgService::instance().log(o,RooFit::WARNING,RooFit::a)
#define ooccxcoutW(o,a) if (RooMsgService::instance().isActive(o,RooFit::a,RooFit::WARNING)) RooMsgService::instance().log(o,RooFit::WARNING,RooFit::a,true)
#define cxcoutE(a) if (RooMsgService::instance().isActive(this,RooFit::a,RooFit::ERROR)) RooMsgService::instance().log(this,RooFit::ERROR,RooFit::a)
#define ccxcoutE(a) if (RooMsgService::instance().isActive(this,RooFit::a,RooFit::ERROR)) RooMsgService::instance().log(this,RooFit::ERROR,RooFit::a,true)
#define oocxcoutE(o,a) if (RooMsgService::instance().isActive(o,RooFit::a,RooFit::ERROR)) RooMsgService::instance().log(o,RooFit::ERROR,RooFit::a)
#define ooccxcoutE(o,a) if (RooMsgService::instance().isActive(o,RooFit::a,RooFit::ERROR)) RooMsgService::instance().log(o,RooFit::ERROR,RooFit::a,true)
#define cxcoutF(a) if (RooMsgService::instance().isActive(this,RooFit::a,RooFit::FATAL)) RooMsgService::instance().log(this,RooFit::FATAL,RooFit::a)
#define ccxcoutF(a) if (RooMsgService::instance().isActive(this,RooFit::a,RooFit::FATAL)) RooMsgService::instance().log(this,RooFit::FATAL,RooFit::a,true)
#define oocxcoutF(o,a) if (RooMsgService::instance().isActive(o,RooFit::a,RooFit::FATAL)) RooMsgService::instance().log(o,RooFit::FATAL,RooFit::a)
#define ooccxcoutF(o,a) if (RooMsgService::instance().isActive(o,RooFit::a,RooFit::FATAL)) RooMsgService::instance().log(o,RooFit::FATAL,RooFit::a,true)

class RooMsgService : public TObject {
public:

  ~RooMsgService() override ;

  struct StreamConfig {
    public:

    void addTopic(RooFit::MsgTopic newTopic) {
      topic |= newTopic ;
    }

    void removeTopic(RooFit::MsgTopic oldTopic) {
      topic &= ~oldTopic ;
    }


    friend class RooMsgService ;

    bool match(RooFit::MsgLevel level, RooFit::MsgTopic facility, const RooAbsArg* obj) ;
    bool match(RooFit::MsgLevel level, RooFit::MsgTopic facility, const TObject* obj) ;

    inline bool match(RooFit::MsgLevel level, RooFit::MsgTopic facility, std::nullptr_t obj)
    {
       return match(level, facility, static_cast<TObject const *>(obj));
    }

    bool active ;
    bool universal ;

    RooFit::MsgLevel minLevel ;
    Int_t    topic ;
    std::string objectName ;
    std::string className ;
    std::string baseClassName ;
    std::string tagName ;
    Color_t color ;
    bool prefix ;

    std::ostream* os ;

  } ;

  // Access to instance
  static RooMsgService& instance();
  static bool anyDebug() ;

  // User interface -- Add or delete reporting streams ;
  Int_t addStream(RooFit::MsgLevel level, const RooCmdArg& arg1={}, const RooCmdArg& arg2={}, const RooCmdArg& arg3={},
                          const RooCmdArg& arg4={}, const RooCmdArg& arg5={}, const RooCmdArg& arg6={});
  void deleteStream(Int_t id) ;
  StreamConfig& getStream(Int_t id) { return _streams[id] ; }

  Int_t numStreams() const { return _streams.size() ; }
  void setStreamStatus(Int_t id, bool active) ;
  bool getStreamStatus(Int_t id) const ;

  void reset();

  void setGlobalKillBelow(RooFit::MsgLevel level) { _globMinLevel = level ; }
  RooFit::MsgLevel globalKillBelow() const { return _globMinLevel ; }

  void Print(Option_t *options= nullptr) const override ;
  void showPid(bool flag) { _showPid = flag ; }

  // Back end -- Send message or check if particular logging configuration is active
  std::ostream& log(const RooAbsArg* self, RooFit::MsgLevel level, RooFit::MsgTopic facility, bool forceSkipPrefix=false) ;
  std::ostream& log(const TObject* self, RooFit::MsgLevel level, RooFit::MsgTopic facility, bool forceSkipPrefix=false) ;
  // Overload to resolve the ambiguity when passing a `nullptr`. Without this,
  // one would have to explicitly cast the `nullptr` to TObject* or RooAbsArg*.
  inline std::ostream& log(std::nullptr_t, RooFit::MsgLevel level, RooFit::MsgTopic facility, bool forceSkipPrefix=false) {
      return log(static_cast<TObject*>(nullptr), level, facility, forceSkipPrefix);
  }

  /// Check if logging is active for given object/topic/RooFit::%MsgLevel combination.
  template <class T>
  bool isActive(T self, RooFit::MsgTopic topic, RooFit::MsgLevel level)
  {
     return activeStream(self, topic, level) >= 0;
  }

  static Int_t _debugCount ;
  std::map<int,std::string> _levelNames ;
  std::map<int,std::string> _topicNames ;

  // Print level support for RooFit-related messages that are not routed through RooMsgService (such as Minuit printouts)
  bool silentMode() const { return _silentMode ; }
  void setSilentMode(bool flag) { _silentMode = flag ; }

  Int_t errorCount() const { return _errorCount ; }
  void clearErrorCount() { _errorCount = 0 ; }

  void saveState() ;
  void restoreState() ;

  RooWorkspace* debugWorkspace() ;

  Int_t& debugCode() { return _debugCode ; }

protected:

  /// Find appropriate logging stream for message from given object with given topic and message level.
  template <class T>
  Int_t activeStream(T self, RooFit::MsgTopic topic, RooFit::MsgLevel level)
  {
     if (level < _globMinLevel)
        return -1;
     for (UInt_t i = 0; i < _streams.size(); i++) {
        if (_streams[i].match(level, topic, self)) {
           return i;
        }
     }
     return -1;
  }

  std::vector<StreamConfig> _streams ;
  std::stack<std::vector<StreamConfig> > _streamsSaved ;
  std::unique_ptr<std::ofstream> _devnull ;

  std::map<std::string,std::unique_ptr<std::ostream>> _files ;
  RooFit::MsgLevel _globMinLevel ;
  RooFit::MsgLevel _lastMsgLevel ;

  bool _silentMode ;
  bool _showPid ;

  Int_t _errorCount ;

  // Private constructor -- singleton class
  RooMsgService() ;
  RooMsgService(const RooMsgService&) ;

  std::unique_ptr<RooWorkspace> _debugWorkspace;

  Int_t _debugCode ;

  ClassDefOverride(RooMsgService,0) // RooFit Message Service Singleton class
};

#endif
