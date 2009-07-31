// @(#)root/xmlparser:$Id$
// Author: Jose Lo   12/4/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXMLParser
#define ROOT_TXMLParser

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

#ifndef ROOT_TQObject
#include "TQObject.h"
#endif

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TString
#include "TString.h"
#endif

struct   _xmlParserCtxt;


class TXMLParser : public TObject, public TQObject {

private:
   TXMLParser(const TXMLParser&);            // Not implemented
   TXMLParser& operator=(const TXMLParser&); // Not implemented

protected:
   _xmlParserCtxt     *fContext;          // parse the xml file
   Bool_t              fValidate;         // to validate the parse context
   Bool_t              fReplaceEntities;  // replace entities
   Bool_t              fStopError;        // stop when parse error occurs
   TString             fValidateError;    // parse error
   TString             fValidateWarning;  // parse warning
   Int_t               fParseCode;        // to keep track of the errorcodes

   virtual void        InitializeContext();
   virtual void        ReleaseUnderlying();
   virtual void        OnValidateError(const TString& message);
   virtual void        OnValidateWarning(const TString& message);
   virtual void        SetParseCode(Int_t code);

public:
   TXMLParser();
   virtual ~TXMLParser();

   void                SetValidate(Bool_t val = kTRUE);
   Bool_t              GetValidate() const { return fValidate; }

   void                SetReplaceEntities(Bool_t val = kTRUE);
   Bool_t              GetReplaceEntities() const { return fReplaceEntities; }

   virtual Int_t       ParseFile(const char *filename) = 0;
   virtual Int_t       ParseBuffer(const char *contents, Int_t len) = 0;
   virtual void        StopParser();

   Int_t               GetParseCode() const { return fParseCode; }

   const char         *GetParseCodeMessage(Int_t parseCode) const;

   void                SetStopOnError(Bool_t stop = kTRUE);
   Bool_t              GetStopOnError() const { return fStopError; }

   const char         *GetValidateError() const { return fValidateError; }
   const char         *GetValidateWarning() const { return fValidateWarning; }

   ClassDef(TXMLParser,0);  // XML SAX parser
};

#endif
