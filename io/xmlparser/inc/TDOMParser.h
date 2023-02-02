// @(#)root/xmlparser:$Id$
// Author: Jose Lo   12/4/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDOMParser
#define ROOT_TDOMParser

#include "TXMLParser.h"

#include "TXMLDocument.h"


class TDOMParser : public TXMLParser {

private:
   TXMLDocument *fTXMLDoc;      ///< xmlDoc

   TDOMParser(const TDOMParser&) = delete;
   TDOMParser& operator=(const TDOMParser&) = delete;
   Int_t ParseContext();

public:
   TDOMParser();
   virtual ~TDOMParser();

   Int_t   ParseFile(const char *filename) override;
   Int_t   ParseBuffer(const char *buffer, Int_t len) override;
   void    ReleaseUnderlying() override;

   virtual TXMLDocument *GetXMLDocument() const;

   ClassDefOverride(TDOMParser, 0); // DOM Parser
};

#endif
