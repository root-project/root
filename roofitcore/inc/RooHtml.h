/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooHtml.rdl,v 1.1 2001/10/04 00:37:19 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   26-Sep-2001 DK Created initial version
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/
#ifndef ROO_HTML
#define ROO_HTML

#include "THtml.h"
#include "TString.h"

class ofstream;

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
