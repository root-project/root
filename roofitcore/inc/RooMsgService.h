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

#include "Riostream.h"
#include <assert.h>
#include "TObject.h"
#include <map>
#include <string>
#include <vector>
#include "RooCmdArg.h"
class RooAbsArg ;


// Shortcut definitions 
#define coutD(a) RooMsgService::instance().log(this,RooMsgService::DEBUG,a) 
#define coutI(a) RooMsgService::instance().log(this,RooMsgService::INFO,a) 
#define coutW(a) RooMsgService::instance().log(this,RooMsgService::WARNING,a) 
#define coutE(a) RooMsgService::instance().log(this,RooMsgService::ERROR,a) 
#define coutF(a) RooMsgService::instance().log(this,RooMsgService::FATAL,a) 

#define oocoutD(o,a) RooMsgService::instance().log(o,RooMsgService::DEBUG,a) 
#define oocoutI(o,a) RooMsgService::instance().log(o,RooMsgService::INFO,a) 
#define oocoutW(o,a) RooMsgService::instance().log(o,RooMsgService::WARNING,a) 
#define oocoutE(o,a) RooMsgService::instance().log(o,RooMsgService::ERROR,a) 
#define oocoutF(o,a) RooMsgService::instance().log(o,RooMsgService::FATAL,a) 

#define dologD(a) (RooMsgService::instance().isActive(this,a,RooMsgService::DEBUG))
#define dologI(a) (RooMsgService::instance().isActive(this,a,RooMsgService::INFO))
#define dologW(a) (RooMsgService::instance().isActive(this,a,RooMsgService::WARNING))
#define dologE(a) (RooMsgService::instance().isActive(this,a,RooMsgService::ERROR))
#define dologF(a) (RooMsgService::instance().isActive(this,a,RooMsgService::FATAL))

// Shortcuts definitions with conditional execution of print expression -- USE WITH CAUTION 
#define cxcoutD(a) if (RooMsgService::anyDebug() && RooMsgService::instance().isActive(this,a,RooMsgService::DEBUG)) RooMsgService::instance().log(this,RooMsgService::DEBUG,a) 
#define ccxcoutD(a) if (RooMsgService::anyDebug() && RooMsgService::instance().isActive(this,a,RooMsgService::DEBUG)) RooMsgService::instance().log(this,RooMsgService::DEBUG,a,kTRUE) 
#define oocxcoutD(o,a) if (RooMsgService::anyDebug() && RooMsgService::instance().isActive(o,a,RooMsgService::DEBUG)) RooMsgService::instance().log(o,RooMsgService::DEBUG,a) 
#define ooccxcoutD(o,a) if (RooMsgService::anyDebug() && RooMsgService::instance().isActive(o,a,RooMsgService::DEBUG)) RooMsgService::instance().log(o,RooMsgService::DEBUG,a,kTRUE) 
#define cxcoutI(a) if (RooMsgService::instance().isActive(this,a,RooMsgService::INFO)) RooMsgService::instance().log(this,RooMsgService::INFO,a) 
#define ccxcoutI(a) if (RooMsgService::instance().isActive(this,a,RooMsgService::INFO)) RooMsgService::instance().log(this,RooMsgService::INFO,a,kTRUE) 
#define oocxcoutI(o,a) if (RooMsgService::instance().isActive(o,a,RooMsgService::INFO)) RooMsgService::instance().log(o,RooMsgService::INFO,a) 
#define ooccxcoutI(o,a) if (RooMsgService::instance().isActive(o,a,RooMsgService::INFO)) RooMsgService::instance().log(o,RooMsgService::INFO,a,kTRUE) 
#define cxcoutW(a) if (RooMsgService::instance().isActive(this,a,RooMsgService::WARNING)) RooMsgService::instance().log(this,RooMsgService::WARNING,a) 
#define ccxcoutW(a) if (RooMsgService::instance().isActive(this,a,RooMsgService::WARNING)) RooMsgService::instance().log(this,RooMsgService::WARNING,a,kTRUE) 
#define oocxcoutW(o,a) if (RooMsgService::instance().isActive(o,a,RooMsgService::WARNING)) RooMsgService::instance().log(o,RooMsgService::WARNING,a) 
#define ooccxcoutW(o,a) if (RooMsgService::instance().isActive(o,a,RooMsgService::WARNING)) RooMsgService::instance().log(o,RooMsgService::WARNING,a,kTRUE) 
#define cxcoutE(a) if (RooMsgService::instance().isActive(this,a,RooMsgService::ERROR)) RooMsgService::instance().log(this,RooMsgService::ERROR,a) 
#define ccxcoutE(a) if (RooMsgService::instance().isActive(this,a,RooMsgService::ERROR)) RooMsgService::instance().log(this,RooMsgService::ERROR,a,kTRUE) 
#define oocxcoutE(o,a) if (RooMsgService::instance().isActive(o,a,RooMsgService::ERROR)) RooMsgService::instance().log(to,RooMsgService::ERROR,a) 
#define ooccxcoutE(o,a) if (RooMsgService::instance().isActive(o,a,RooMsgService::ERROR)) RooMsgService::instance().log(o,RooMsgService::ERROR,a,kTRUE) 
#define cxcoutF(a) if (RooMsgService::instance().isActive(this,a,RooMsgService::FATAL)) RooMsgService::instance().log(this,RooMsgService::FATAL,a) 
#define ccxcoutF(a) if (RooMsgService::instance().isActive(this,a,RooMsgService::FATAL)) RooMsgService::instance().log(this,RooMsgService::FATAL,a,kTRUE) 
#define oocxcoutF(o,a) if (RooMsgService::instance().isActive(o,a,RooMsgService::FATAL)) RooMsgService::instance().log(o,RooMsgService::FATAL,a) 
#define ooccxcoutF(o,a) if (RooMsgService::instance().isActive(o,a,RooMsgService::FATAL)) RooMsgService::instance().log(o,RooMsgService::FATAL,a,kTRUE) 

class RooMsgService : public TObject {
public:

  virtual ~RooMsgService() ;

  enum MsgLevel { DEBUG=0, INFO=1, WARNING=2, ERROR=3, FATAL=4 } ;

  // Access to instance
  static RooMsgService& instance() ;
  static Bool_t anyDebug() { return instance()._debugCount>0 ; }

  // User interface -- Add or delete reporting streams ;
  Int_t addStream(MsgLevel level, const RooCmdArg& arg1=RooCmdArg(), const RooCmdArg& arg2=RooCmdArg(), const RooCmdArg& arg3=RooCmdArg(),
                    		  const RooCmdArg& arg4=RooCmdArg(), const RooCmdArg& arg5=RooCmdArg(), const RooCmdArg& arg6=RooCmdArg()); 
  void deleteStream(Int_t id) ;

  void setStreamStatus(Int_t id, Bool_t active) ;
  Bool_t getStreamStatus(Int_t id) const ;

  void Print(Option_t *options= 0) const ;

  // Back end -- Send message or check if particular logging configuration is active
  ostream& log(const RooAbsArg* self, MsgLevel level, const char* facility, Bool_t forceSkipPrefix=kFALSE) ;
  ostream& log(const TObject* self, MsgLevel level, const char* facility, Bool_t forceSkipPrefix=kFALSE) ;
  Bool_t isActive(const RooAbsArg* self, const char* facility, MsgLevel level) ;
  Bool_t isActive(const TObject* self, const char* facility, MsgLevel level) ;

protected:

  Int_t activeStream(const RooAbsArg* self, const char* facility, MsgLevel level) ;
  Int_t activeStream(const TObject* self, const char* facility, MsgLevel level) ;

  struct StreamConfig {

    Bool_t active ;

    MsgLevel minLevel ;
    std::string topic ;
    std::string objectName ;
    std::string className ;
    std::string baseClassName ;
    std::string tagName ;
    Color_t color ;
    Bool_t prefix ;

    ostream* os ;

    Bool_t match(MsgLevel level, const char* facility, const RooAbsArg* obj) ;
    Bool_t match(MsgLevel level, const char* facility, const TObject* obj) ;

  } ;

  std::vector<StreamConfig> _streams ;
  ostream* _devnull ;

  std::map<string,ostream*> _files ;

  // Private ctor -- singleton class
  RooMsgService() ;
  RooMsgService(const RooMsgService&) ;

  static RooMsgService* _instance ;
  
  Int_t _debugCount ;

  ClassDef(RooMsgService,0) // RooFit Message Service Singleton class
};


#endif
