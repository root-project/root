// @(#)root/xmlparser:$Name:  $:$Id: TXMLParser.h,v 1.2 2005/03/14 20:02:41 rdm Exp $
// Author: Jose Lo   12/4/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXMLNode
#define ROOT_TXMLNode

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXMLNode                                                             //
//                                                                      //
// TXMLNode contains a pointer to xmlNode, which is a node under the    //
// DOM tree. A node can be an Element, an Attribute, a Text Node        //
// or a Comment Node.                                                   //
// One can navigate the DOM tree by accessing the siblings and          //
// parent or child nodes. Also retriving the Attribute or the Text in   //
// an Element node.                                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TString
#include "TString.h"
#endif

class TList;
struct _xmlNode;




class TXMLNode : public TObject {

private:
   _xmlNode *fXMLNode;     // libxml node

public:
   // This enum is based on libxml tree Enum xmlElementType
   enum EXMLElementType {
      kXMLElementNode = 1,
      kXMLAttributeNode = 2,
      kXMLTextNode = 3,
      kXMLCommentNode = 8
   };

   TXMLNode(_xmlNode *node);
   virtual ~TXMLNode();

   EXMLElementType GetNodeType() const;
   const char *GetNodeName() const;
   TXMLNode   *GetChildren() const;
   TXMLNode   *GetParent() const;
   TXMLNode   *GetNextNode() const;
   TXMLNode   *GetPreviousNode() const;
   const char *GetContent() const;
   const char *GetText() const;
   TList      *GetAttributes() const;

   Bool_t      HasChildren() const;
   Bool_t      HasNextNode() const;
   Bool_t      HasParent() const;
   Bool_t      HasPreviousNode() const;
   Bool_t      HasAttributes() const;

   const char *GetNamespaceHref() const;
   const char *GetNamespacePrefix() const;

   ClassDef(TXMLNode,0);  // XML node under DOM tree
};

#endif
