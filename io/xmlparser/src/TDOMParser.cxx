// @(#)root/xmlparser:$Id$
// Author: Jose Lo   12/4/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
\class TDomParser
\ingroup IO

DOM stands for the Document Object Model; this is an API for
accessing XML or HTML structured documents.
The Document Object Model is a platform and language-neutral
interface that will allow programs and scripts to dynamically
access and update the content, structure and style of documents.

The parser returns a tree built during the document analysis.
*/

#include "TDOMParser.h"
#include "TXMLDocument.h"

#include <libxml/tree.h>
#include <libxml/parserInternals.h>


ClassImp(TDOMParser);

////////////////////////////////////////////////////////////////////////////////
/// TDOMParser constructor.

TDOMParser::TDOMParser() : fTXMLDoc(0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// TDOMParser destructor, it calls ReleaseUnderlying().

TDOMParser::~TDOMParser()
{
   ReleaseUnderlying();
}

////////////////////////////////////////////////////////////////////////////////
/// Release any existing document.

void TDOMParser::ReleaseUnderlying()
{
   if (fTXMLDoc) {
      delete fTXMLDoc;
      fTXMLDoc = 0;
   }

   SetParseCode(0);

   TXMLParser::ReleaseUnderlying();
}

////////////////////////////////////////////////////////////////////////////////
/// Parse the XML file where filename is the XML file name.
/// It will create a TXMLDocument if the file is parsed without
/// any error. It returns parse code error in case of parse error,
/// see TXMLParser.

Int_t TDOMParser::ParseFile(const char *filename)
{
   ReleaseUnderlying();

   fContext = xmlCreateFileParserCtxt(filename);

   if (!fContext) {
      SetParseCode(-2);
      return -2;
   }

   InitializeContext();

   if (!fContext->directory) {
      const char *dir = xmlParserGetDirectory(filename);
      fContext->directory = (char *)xmlStrdup((const xmlChar *)dir);
   }

   return ParseContext();
}

////////////////////////////////////////////////////////////////////////////////
/// It parses a buffer, much like ParseFile().

Int_t TDOMParser::ParseBuffer(const char *buffer, Int_t len)
{
   ReleaseUnderlying();

   fContext = xmlCreateMemoryParserCtxt(buffer, len);

   if (!fContext) {
      SetParseCode(-2);
      return -2;
   }

   InitializeContext();

   return ParseContext();
}

////////////////////////////////////////////////////////////////////////////////
/// Creates a XML document for the parser.
/// It returns 0 on success, and
///   - \b -1 if no XML document was created,
///   - \b -5 if the document is not well formated,
///   - \b -6 if document is not valid.

Int_t TDOMParser::ParseContext()
{
   xmlParseDocument(fContext);

   if (!fContext->myDoc) {
      SetParseCode(-1);
      return -1;
   }

   if (!fContext->wellFormed) {
      SetParseCode(-5);
      return -5;
   }

   if (!fContext->valid) {
      SetParseCode(-6);
      return -6;
   }

   fTXMLDoc = new TXMLDocument(fContext->myDoc);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the TXMLDocument.

TXMLDocument *TDOMParser::GetXMLDocument() const
{
   return fTXMLDoc;
}
