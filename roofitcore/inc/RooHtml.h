/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$                                                             *
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2002, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_HTML
#define ROO_HTML

#include "THtml.h"
#include "TString.h"

#include <fstream.h>

class RooHtml : public THtml {
public:
  inline RooHtml(const char *version) : _version(version) { };
  inline virtual ~RooHtml() { };
  virtual void WriteHtmlHeader(ofstream &out, const char *title);
  virtual void  WriteHtmlFooter(ofstream &out, const char *dir="", const char *lastUpdate="",
				const char *author="", const char *copyright="");

  inline const char *getVersion() const { return _version.Data(); }
  void MakeIndexNew(const char *filter="*");
  
  void addTopic(const char* tag, const char* description) ;
  void MakeIndexOfTopics() ;
  
protected:
  TString _version;

  TList _topicTagList ;
  TList _topicDescList ;

  char* getClassGroup(const char* fileName) ;

private:
  RooHtml(const RooHtml&) ;

  ClassDef(RooHtml,0) // Convert Roo classes to HTML web pages
};

#endif
