/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooMath.rdl,v 1.4 2001/09/24 23:05:59 verkerke Exp $
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

protected:
  TString _version;

  ClassDef(RooHtml,0) // Convert Roo classes to HTML web pages
};

#endif
