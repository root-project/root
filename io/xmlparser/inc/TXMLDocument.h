// @(#)root/xmlparser:$Id$
// Author: Jose Lo   12/4/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXMLDocument
#define ROOT_TXMLDocument

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXMLDocument                                                         //
//                                                                      //
// TXMLDocument contains a pointer to an xmlDoc structure, after the    //
// parser returns a tree built during the document analysis.            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TString
#include "TString.h"
#endif


struct _xmlDoc;
class TXMLNode;


class TXMLDocument : public TObject {

private:
   TXMLDocument(const TXMLDocument&);            // Not implemented
   TXMLDocument& operator=(const TXMLDocument&); // Not implemented

   _xmlDoc  *fXMLDoc;           // libxml xml doc
   TXMLNode *fRootNode;         // the root node

public:
   TXMLDocument(_xmlDoc *doc);
   virtual ~TXMLDocument();

   TXMLNode   *GetRootNode() const;

   const char *Version() const;
   const char *Encoding() const;
   const char *URL() const;

   ClassDef(TXMLDocument,0)  // XML document created by the DOM parser
};

#endif
