// @(#)root/xmlparser:$Name:$:$Id:$
// Author: Jose Lo   12/1/2005

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
// TXMLParser is parent class of TSAXParser, which is a SAX interface   //
// of libxml.                                                           //
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

extern "C" {
  struct   _xmlParserCtxt;
  struct   exception;
}


class TXMLParser : public TObject, public TQObject {

protected:
   _xmlParserCtxt     *fContext;          // parse the xml file
   Bool_t              fValidate;         // to validate the parse context
   Bool_t              fStopError;        // stop when parse error occurs
   TString             fValidateError;    // parse error
   TString             fValidateWarning;  // parse warning
   Int_t               fParseCode;        // to keep track of the errorcodes

   virtual void        InitializeContext();
   virtual void        ReleaseUnderlying();
   virtual void        OnValidateError(const TString& message);
   virtual void        OnValidateWarning(const TString& message);
   virtual void        StopParser();
   virtual void        SetParseCode(Int_t code);

public:
   TXMLParser();
   virtual ~TXMLParser();

   virtual void        SetValidate(Bool_t val = kTRUE);
   virtual Bool_t      GetValidate() const { return fValidate; }

   virtual Int_t       ParseFile(const char *filename) = 0;
   virtual Int_t       ParseBuffer(const char *contents, Int_t len) = 0;

   Int_t               GetParseCode() const { return fParseCode; }

   void                SetStopOnError(Bool_t stop = kTRUE);
   Bool_t              GetStopOnError() const { return fStopError; }

   const char         *GetValidateError() const { return fValidateError; }
   const char         *GetValidateWarning() const { return fValidateWarning; }

   ClassDef(TXMLParser,0);
};


class TXMLAttr : public TObject {
private:
   const char *fKey;     // XML attribute key
   const char *fValue;   // XML attribute value

public:
   TXMLAttr(const char *key, const char *value) : fKey(key), fValue(value) { }
   const char  *GetName() const { return fKey; }
   const char  *Key() const { return fKey; }
   const char  *Value() const { return fValue; }

   ClassDef(TXMLAttr,0)  //XML attribute pair
};


#endif

