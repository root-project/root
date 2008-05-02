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

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TDOMParser                                                           //
//                                                                      //
// DOM stands for the Document Object Model; this is an API for         //
// accessing XML or HTML structured documents.                          //
// The Document Object Model is a platform and language-neutral         //
// interface that will allow programs and scripts to dynamically        //
// access and update the content, structure and style of documents.     //
//                                                                      //
// The parser returns a tree built during the document analysis.        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TXMLParser
#include "TXMLParser.h"
#endif

#ifndef ROOT_TXMLDocument
#include "TXMLDocument.h"
#endif


class TDOMParser : public TXMLParser {

private:
   TXMLDocument *fTXMLDoc;      // xmlDoc

   TDOMParser(const TDOMParser&);            // Not implemented
   TDOMParser& operator=(const TDOMParser&); // Not implemented
   Int_t ParseContext();

public:
   TDOMParser();
   virtual ~TDOMParser();

   virtual Int_t   ParseFile(const char *filename);
   virtual Int_t   ParseBuffer(const char *buffer, Int_t len);
   virtual void    ReleaseUnderlying();

   virtual TXMLDocument *GetXMLDocument() const;

   ClassDef(TDOMParser, 0); // DOM Parser
};

#endif
