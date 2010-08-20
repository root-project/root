// @(#)root/xmlparser:$Id$
// Author: Jose Lo   12/1/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXMLParser                                                           //
//                                                                      //
// TXMLParser is an abstract class which interfaces with Libxml2.       //
// Libxml2 is the XML C parser and toolkit developed for the Gnome      //
// project.                                                             //
//                                                                      //
// The libxml library provides two interfaces to the parser, a DOM      //
// style tree interface and a SAX style event based interface.          //
//                                                                      //
// TXMLParser is parent class of TSAXParser and TDOMParser, which are   //
// a SAX interface and DOM interface of libxml.                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

/*************************************************************************
  This source is based on libxml++, a C++ wrapper for the libxml XML
  parser library. Copyright (C) 2000 by Ari Johnson.

  libxml++ are copyright (C) 2000 by Ari Johnson, and are covered by the
  GNU Lesser General Public License, which should be included with
  libxml++ as the file COPYING.
 *************************************************************************/

#include "Riostream.h"
#include "TXMLParser.h"

#include <libxml/parser.h>


ClassImp(TXMLParser);

//______________________________________________________________________________
TXMLParser::TXMLParser()
   : fContext(0), fValidate(kTRUE), fReplaceEntities(kFALSE), fStopError(kFALSE), fParseCode(0)
{
   // Initializes parser variables.

   xmlInitParser();
}

//______________________________________________________________________________
TXMLParser::~TXMLParser()
{
   // Cleanup.

   ReleaseUnderlying();
   fParseCode = 0;
}

//______________________________________________________________________________
void TXMLParser::SetValidate(Bool_t val)
{
   // The parser will validate the xml file if val = true.

   fValidate = val;
}

//______________________________________________________________________________
void TXMLParser::SetReplaceEntities(Bool_t val)
{
   // The parser will replace/expand entities.

   fReplaceEntities = val;
}

//______________________________________________________________________________
void TXMLParser::ReleaseUnderlying()
{
   // To release any existing document.

   if (fContext) {
      fContext->_private = 0;
      xmlFreeParserCtxt(fContext);
      fContext = 0;
   }
   xmlCleanupParser();
}

//______________________________________________________________________________
void TXMLParser::OnValidateError(const TString& message)
{
   // This function is called when an error from the parser has occured.
   // Message is the parse error.

   fValidateError += message;
}

//______________________________________________________________________________
void TXMLParser::OnValidateWarning(const TString& message)
{
   // This function is called when a warning from the parser has occured.
   // Message is the parse error.

   fValidateWarning += message;
}

//______________________________________________________________________________
const char *TXMLParser::GetParseCodeMessage(Int_t parseCode) const
{
   // Returns the parse code message.

   switch (parseCode) {
      case -1:
         return "Attempt to parse a second file while a parse is in progress";
         break;
      case -2:
         return "Parse context is not created";
         break;
      case -3:
         return "An error occured while parsing file";
         break;
      case -4:
         return "A fatal error occured while parsing file";
         break;
      case -5:
         return "Document is not well-formed";
         break;
      case -6:
         return "Document is not valid";
         break;
      default:
         return "Parse code does not exist";
   }
}

//______________________________________________________________________________
void TXMLParser::InitializeContext()
{
   // Initialize parser parameters, such as, disactivate non-standards libxml1
   // features, on/off validation, clear error and warning messages.

   fContext->linenumbers = 1; // TRUE - This is the default anyway.
   fContext->validate = fValidate ? 1 : 0;
   fContext->replaceEntities = fReplaceEntities ? 1 : 0;
   fContext->_private = this;

   fValidateError = "";
   fValidateWarning = "";
}

//______________________________________________________________________________
void TXMLParser::StopParser()
{
   // Stops parsing.

   if (fContext)
      xmlStopParser(fContext);
}

//______________________________________________________________________________
void TXMLParser::SetParseCode(Int_t errorcode)
{
   // Set the parse code:
   //  0: Parse succesfull
   // -1: Attempt to parse a second file while a parse is in progress
   // -2: Parse context is not created
   // -3: An error occured while parsing file
   // -4: A fatal error occured while parsing file
   // -5: Document is not well-formed

   fParseCode = errorcode;
}

//______________________________________________________________________________
void TXMLParser::SetStopOnError(Bool_t stop)
{
   // Set parser stops in case of error:
   // stop = true, stops on error
   // stop = false, continue parsing on error...

   fStopError = stop;
}
