/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooHtml.rdl,v 1.13 2005/12/08 13:19:55 wverkerke Exp $
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
#ifndef ROO_HTML
#define ROO_HTML

#include "THtml.h"
#include "TString.h"
#include "TList.h"

#include <fstream>

class RooHtml : public THtml {
public:
  inline RooHtml(const char *version) : _version(version), _hfColor("#FFCC00") { };
  inline virtual ~RooHtml() { };

   virtual void WriteHtmlHeader(ofstream &out, const char *title, const char* dir="", TClass* cls=0);
  virtual void  WriteHtmlFooter(ofstream &out, const char *dir="", const char *lastUpdate="",
				const char *author="", const char *copyright="");

  inline const char *getVersion() const { return _version.Data(); }
  void GetModuleName(TString& module, const char* filename) const;
  
  void addTopic(const char* tag, const char* description) ;
  void MakeIndexOfTopics() ;

  void setHeaderColor(const char* string) { _hfColor = string ; }
  
protected:
  TString _version;
  TString _hfColor ;

  TList _topicTagList ;
  TList _topicDescList ;

  char* getClassGroup(const char* fileName) const;

private:
  RooHtml(const RooHtml&) ;

  ClassDef(RooHtml,0) // Convert Roo classes to HTML web pages
};

#endif
