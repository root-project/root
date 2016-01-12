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
\class TXMLDocument
\ingroup IO

TXMLDocument contains a pointer to an xmlDoc structure, after the
parser returns a tree built during the document analysis.
*/

#include "TXMLDocument.h"
#include "TXMLNode.h"
#include <libxml/tree.h>


ClassImp(TXMLDocument);

////////////////////////////////////////////////////////////////////////////////
/// TXMLDocument constructor.

TXMLDocument::TXMLDocument(_xmlDoc *doc) : fXMLDoc(doc)
{
   if (fXMLDoc) {
      fRootNode = new TXMLNode(xmlDocGetRootElement(fXMLDoc));
   } else {
      fRootNode = 0;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// TXMLDocument destructor.
/// Free the global variables that may
/// have been allocated by the parser.

TXMLDocument::~TXMLDocument()
{
   delete fRootNode;
   xmlFreeDoc(fXMLDoc);
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the root element node.

TXMLNode *TXMLDocument::GetRootNode() const
{
   return fRootNode;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the XML version string or 0 in case there is no document set.

const char *TXMLDocument::Version() const
{
   if (fXMLDoc)
      return (const char *) fXMLDoc->version;
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns external initial encoding, if any or 0 in case there is no
/// document set.

const char *TXMLDocument::Encoding() const
{
   if (fXMLDoc)
      return (const char *) fXMLDoc->encoding;
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the URI for the document or 0 in case there is no document set.

const char *TXMLDocument::URL() const
{
   if (fXMLDoc)
      return (const char *) fXMLDoc->URL;
   return 0;
}
