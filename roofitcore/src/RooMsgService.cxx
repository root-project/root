/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id$
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

// -- CLASS DESCRIPTION [AUX] --

#include "RooFit.h"
#include "RooAbsArg.h"
#include "TClass.h"

#include "RooMsgService.h"
#include "RooCmdArg.h"
#include "RooCmdConfig.h"
#include "RooGlobalFunc.h"

#include <iostream>
#include <iomanip>
#include <fstream>
using namespace std ;
using namespace RooFit ;

ClassImp(RooMsgService)
;

RooMsgService* RooMsgService::_instance = 0 ;


RooMsgService::RooMsgService() 
{
  _devnull = new ofstream("/dev/null") ;
  addStream(WARNING) ;
  addStream(INFO,Topic("Generation")) ;
}

RooMsgService::~RooMsgService() 
{
  // Delete all ostreams we own ;

  map<string,ostream*>::iterator iter = _files.begin() ;
  for (; iter != _files.end() ; ++iter) {
    delete iter->second ;
  }

  delete _devnull ;
}


Int_t RooMsgService::addStream(MsgLevel level, const RooCmdArg& arg1, const RooCmdArg& arg2, const RooCmdArg& arg3, 
      					                const RooCmdArg& arg4, const RooCmdArg& arg5, const RooCmdArg& arg6) 
{
  // Aggregate all arguments in a list
  RooLinkedList l ;
  l.Add((TObject*)&arg1) ;  l.Add((TObject*)&arg2) ;  
  l.Add((TObject*)&arg3) ;  l.Add((TObject*)&arg4) ;
  l.Add((TObject*)&arg5) ;  l.Add((TObject*)&arg6) ;  

  // Define configuration for this method
  RooCmdConfig pc(Form("RooMsgService::addReportingStream(%s)",GetName())) ;
  pc.defineInt("prefix","Prefix",0,kTRUE) ;
  pc.defineInt("color","Color",0,static_cast<Int_t>(kBlack)) ;
  pc.defineString("topic","Topic",0,"") ;
  pc.defineString("objName","ObjectName",0,"") ;
  pc.defineString("className","ClassName",0,"") ;
  pc.defineString("baseClassName","BaseClassName",0,"") ;
  pc.defineString("tagName","LabelName",0,"") ;
  pc.defineString("outFile","OutputFile",0,"") ;
  pc.defineObject("outStream","OutputStream",0,0) ;
  pc.defineMutex("OutputFile","OutputStream") ;

  // Process & check varargs 
  pc.process(l) ;
  if (!pc.ok(kTRUE)) {
    return -1 ;
  }

  // Extract values from named arguments
  const char* topic =  pc.getString("topic") ;
  const char* objName =  pc.getString("objName") ;
  const char* className =  pc.getString("className") ;
  const char* baseClassName =  pc.getString("baseClassName") ;
  const char* tagName =  pc.getString("tagName") ;
  const char* outFile = pc.getString("outFile") ;
  Bool_t prefix = pc.getInt("prefix") ;
  Color_t color = static_cast<Color_t>(pc.getInt("color")) ;
  ostream* os = reinterpret_cast<ostream*>(pc.getObject("outStream")) ;

  // Create new stream object
  StreamConfig newStream ;

  // Store configuration info
  newStream.active = kTRUE ;
  newStream.minLevel = level ;
  newStream.topic = (topic ? topic : "" ) ;
  newStream.objectName = (objName ? objName : "" ) ;
  newStream.className = (className ? className : "" ) ;
  newStream.baseClassName = (baseClassName ? baseClassName : "" ) ;
  newStream.tagName = (tagName ? tagName : "" ) ;
  newStream.color = color ;
  newStream.prefix = prefix ;

  // Configure output
  if (os) {

    // To given non-owned stream
    newStream.os = os ;

  } else if (string(outFile).size()>0) {

    // See if we already opened the file
    ostream* os = _files["outFile"] ;

    if (!os) {

      // To given file name, create owned stream for it
      os = new ofstream(outFile) ;

      if (!*os) {
	cout << "RooMsgService::addReportingStream ERROR: cannot open output log file " << outFile << " reverting stream to stdout" << endl ;
	delete os ;
	newStream.os = &cout ;
      }
    }
    _files["outFile"] = os ;
        
  } else {

    // To stdout
    newStream.os = &cout ;

  }


  // Add it to list of active streams ;
  _streams.push_back(newStream) ;

  // Return stream identifier
  return _streams.size()-1 ;
}



void RooMsgService::deleteStream(Int_t id) 
{
  vector<StreamConfig>::iterator iter = _streams.begin() ;
  iter += id ;
  _streams.erase(iter) ;
}


void RooMsgService::setStreamStatus(Int_t id, Bool_t flag) 
{
  if (id<0 || id>=static_cast<Int_t>(_streams.size())) {
    cout << "RooMsgService::setStreamStatus() ERROR: invalid stream ID " << id << endl ;
    return ;
  }
  _streams[id].active = flag ;
}


Bool_t RooMsgService::getStreamStatus(Int_t id) const 
{
  if (id<0 || id>= static_cast<Int_t>(_streams.size())) {
    cout << "RooMsgService::getStreamStatus() ERROR: invalid stream ID " << id << endl ;
    return kFALSE ;
  }
  return _streams[id].active ;
}


RooMsgService& RooMsgService::instance() 
{
  if (!_instance) {
    _instance = new RooMsgService() ;    
  }
  return *_instance ;
}


Bool_t RooMsgService::isActive(const RooAbsArg* self, const char* topic, MsgLevel level) 
{
  return (activeStream(self,topic,level)>=0) ;
}

Bool_t RooMsgService::isActive(const TObject* self, const char* topic, MsgLevel level) 
{
  return (activeStream(self,topic,level)>=0) ;
}

Int_t RooMsgService::activeStream(const RooAbsArg* self, const char* topic, MsgLevel level) 
{
  for (UInt_t i=0 ; i<_streams.size() ; i++) {
    if (_streams[i].match(level,topic,self)) {
      return i ;
    }
  }
  return -1 ;
}

Int_t RooMsgService::activeStream(const TObject* self, const char* topic, MsgLevel level) 
{
  for (UInt_t i=0 ; i<_streams.size() ; i++) {
    if (_streams[i].match(level,topic,self)) {
      return i ;
    }
  }
  return -1 ;
}

Bool_t RooMsgService::StreamConfig::match(MsgLevel level, const char* top, const RooAbsArg* obj) 
{
  if (level<minLevel) return kFALSE ;
  if (topic.size()>0 && topic!=top) return kFALSE ;
  if (objectName.size()>0 && objectName != obj->GetName()) return kFALSE ;
  if (className.size()>0 && className != obj->IsA()->GetName()) return kFALSE ;
  if (baseClassName.size()>0 && !obj->IsA()->InheritsFrom(baseClassName.c_str())) return kFALSE ;
  if (tagName.size()>0 && !obj->getAttribute(tagName.c_str())) return kFALSE ;
  
  return kTRUE ;
}

Bool_t RooMsgService::StreamConfig::match(MsgLevel level, const char* top, const TObject* obj) 
{
  if (level<minLevel) return kFALSE ;
  if (topic.size()>0 && topic!=top) return kFALSE ;
  if (objectName.size()>0 && objectName != obj->GetName()) return kFALSE ;
  if (className.size()>0 && className != obj->IsA()->GetName()) return kFALSE ;
  if (baseClassName.size()>0 && !obj->IsA()->InheritsFrom(baseClassName.c_str())) return kFALSE ;
  
  return kTRUE ;
}


ostream& RooMsgService::log(const RooAbsArg* self, MsgLevel level, const char* topic, Bool_t skipPrefix) 
{
  Int_t as = activeStream(self,topic,level) ;
  if (as==-1) {
    return *_devnull ;
  }

  // Flush any previous messages
  (*_streams[as].os).flush() ;
    
  const char* levelName[5] = { "DEBUG", "INFO", "WARNING", "ERROR", "FATAL" } ;
  if (_streams[as].prefix && !skipPrefix) {
    (*_streams[as].os) << "[#" << as << "] " << levelName[level] << ":" << topic  << " -- " ;
  }
  return (*_streams[as].os) ;
}


ostream& RooMsgService::log(const TObject* self, MsgLevel level, const char* topic, Bool_t skipPrefix) 
{
  Int_t as = activeStream(self,topic,level) ;
  if (as==-1) {
    return *_devnull ;
  }

  // Flush any previous messages
  (*_streams[as].os).flush() ;
    
  const char* levelName[5] = { "DEBUG", "INFO", "WARNING", "ERROR", "FATAL" } ;
  if (_streams[as].prefix && !skipPrefix) {
    (*_streams[as].os) << "[#" << as << "] " << levelName[level] << ":" << topic  << " -- " ;
  }
  return (*_streams[as].os) ;
}


void RooMsgService::Print(Option_t *options) const 
{
  const char* levelName[5] = { "DEBUG", "INFO", "WARNING", "ERROR", "FATAL" } ;

  Bool_t activeOnly = kTRUE ;
  if (TString(options).Contains("V") || TString(options).Contains("v")) {
    activeOnly = kFALSE ;
  }

  cout << (activeOnly?"Active Message streams":"All Message streams") << endl ;
  for (UInt_t i=0 ; i<_streams.size() ; i++) {

    // Skip passive streams in active only mode
    if (activeOnly && !_streams[i].active) {
      continue ;
    }

    
    cout << "[" << i << "] MinLevel = " << levelName[_streams[i].minLevel] ;
    if (_streams[i].topic.size()>0) {
      cout << " Topic = " << _streams[i].topic ;
    }
    if (_streams[i].objectName.size()>0) {
      cout << " ObjectName = " << _streams[i].objectName ;
    }
    if (_streams[i].className.size()>0) {
      cout << " ClassName = " << _streams[i].className ;
    }
    if (_streams[i].baseClassName.size()>0) {
      cout << " BaseClassName = " << _streams[i].baseClassName ;
    }
    if (_streams[i].tagName.size()>0) {
      cout << " TagLabel = " << _streams[i].tagName ;
    }
    
    // Postfix status when printing all
    if (!activeOnly && !_streams[i].active) {
      cout << " (NOT ACTIVE)"  ;
    }
    
    cout << endl ; 
  }
  
}
