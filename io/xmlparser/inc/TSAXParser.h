// @(#)root/xmlparser:$Id$
// Author: Jose Lo   12/1/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSAXParser
#define ROOT_TSAXParser

#include "TXMLParser.h"


class TList;
class TSAXParserCallback;
struct _xmlSAXHandler;


class TSAXParser : public TXMLParser {

friend class TSAXParserCallback;

private:
   _xmlSAXHandler         *fSAXHandler;  ///< libxml2 SAX handler

   virtual Int_t           Parse();

   TSAXParser(const TSAXParser&) = delete;
   TSAXParser& operator=(const TSAXParser&) = delete;

public:
   TSAXParser();
   virtual ~TSAXParser();

           Int_t           ParseFile(const char *filename) override;
           Int_t           ParseBuffer(const char *contents, Int_t len) override;

   virtual void            OnStartDocument();  //*SIGNAL*
   virtual void            OnEndDocument();  //*SIGNAL*
   virtual void            OnStartElement(const char *name, const TList *attr);  //*SIGNAL*
   virtual void            OnEndElement(const char *name);  //*SIGNAL*
   virtual void            OnCharacters(const char *characters);  //*SIGNAL*
   virtual void            OnComment(const char *text);  //*SIGNAL*
   virtual void            OnWarning(const char *text);  //*SIGNAL*
   virtual Int_t           OnError(const char *text);  //*SIGNAL*
   virtual Int_t           OnFatalError(const char *text);  //*SIGNAL*
   virtual void            OnCdataBlock(const char *text, Int_t len);  //*SIGNAL*

   virtual void            ConnectToHandler(const char *handlerName, void *handler);

   ClassDefOverride(TSAXParser,0); // SAX Parser
};

#endif
