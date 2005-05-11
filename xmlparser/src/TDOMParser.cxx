// @(#)root/xmlparser:$Name:  $:$Id: TSAXParser.h,v 1.4 2005/05/08 13:55:19 rdm Exp $
// Author: Jose Lo   12/4/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TDomParser                                                           //
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

#include "TDOMParser.h"
#include "TXMLDocument.h"

#include <libxml/tree.h>
#include <libxml/parserInternals.h>


ClassImp(TDOMParser);

//______________________________________________________________________________
TDOMParser::TDOMParser() : fTXMLDoc(0)
{
   // TDOMParser constructor
}

//______________________________________________________________________________
TDOMParser::~TDOMParser()
{
   // TDOMParser destructor, it calls ReleaseUnderlying().

   ReleaseUnderlying();
}

//______________________________________________________________________________
void TDOMParser::ReleaseUnderlying()
{
   // Release any existing document.

   if (fTXMLDoc)
      delete fTXMLDoc;

   TXMLParser::ReleaseUnderlying();
}

//______________________________________________________________________________
Int_t TDOMParser::ParseFile(const char *filename)
{
   // Parse the XML file where filename is the XML file name.
   // It will create a TXMLDocument if the file is parsed without
   // any error. It returns parse code error in case of parse error,
   // see TXMLParser.

   ReleaseUnderlying();

   fContext = xmlCreateFileParserCtxt(filename);

   if (!fContext)
      return -2;

   InitializeContext();

   if (!fContext->directory) {
      const char *dir = xmlParserGetDirectory(filename);
      fContext->directory = (char *)xmlStrdup((const xmlChar *)dir);
   }

   return ParseContext();
}

//______________________________________________________________________________
Int_t TDOMParser::ParseBuffer(const char *buffer, Int_t len)
{
   // It parses a buffer, much like ParseFile().

   ReleaseUnderlying();

   fContext = xmlCreateMemoryParserCtxt(buffer, len);

   if (!fContext)
      return -2;

   InitializeContext();

   return ParseContext();
}

//______________________________________________________________________________
Int_t TDOMParser::ParseContext()
{
   // Creates a XML document for the parser.
   // It returns 0 on success, -1 if no XML document was created and
   // -5 if the document is not well formated.

   xmlParseDocument(fContext);

   if (!fContext->myDoc)
      return -1;

   if (!fContext->wellFormed)
      return -5;

   fTXMLDoc = new TXMLDocument(fContext->myDoc);

   return 0;
}

//______________________________________________________________________________
TXMLDocument *TDOMParser::GetXMLDocument() const
{
   // Returns the TXMLDocument.

   return fTXMLDoc;
}
