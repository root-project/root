// @(#)root/xmlparser:$Id$
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
// TXMLDocument                                                         //
//                                                                      //
// TXMLDocument contains a pointer to an xmlDoc structure, after the    //
// parser returns a tree built during the document analysis.            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TXMLDocument.h"
#include "TXMLNode.h"
#include <libxml/tree.h>


ClassImp(TXMLDocument);

//______________________________________________________________________________
TXMLDocument::TXMLDocument(_xmlDoc *doc) : fXMLDoc(doc)
{
   // TXMLDocument constructor.

   if (fXMLDoc) {
      fRootNode = new TXMLNode(xmlDocGetRootElement(fXMLDoc));
   } else {
      fRootNode = 0;
   }
}

//______________________________________________________________________________
TXMLDocument::~TXMLDocument()
{
   // TXMLDocument destructor.
   // Free the global variables that may
   // have been allocated by the parser.

   delete fRootNode;
   xmlFreeDoc(fXMLDoc);
}

//______________________________________________________________________________
TXMLNode *TXMLDocument::GetRootNode() const
{
   // Returns the root element node.

   return fRootNode;
}

//______________________________________________________________________________
const char *TXMLDocument::Version() const
{
   // Returns the XML version string or 0 in case there is no document set.

   if (fXMLDoc)
      return (const char *) fXMLDoc->version;
   return 0;
}

//______________________________________________________________________________
const char *TXMLDocument::Encoding() const
{
   // Returns external initial encoding, if any or 0 in case there is no
   // document set.

   if (fXMLDoc)
      return (const char *) fXMLDoc->encoding;
   return 0;
}

//______________________________________________________________________________
const char *TXMLDocument::URL() const
{
   // Returns the URI for the document or 0 in case there is no document set.

   if (fXMLDoc)
      return (const char *) fXMLDoc->URL;
   return 0;
}
