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

#include "TXMLNode.h"
#include "TXMLAttr.h"
#include "TList.h"
#include <libxml/tree.h>


ClassImp(TXMLNode);

//______________________________________________________________________________
TXMLNode::TXMLNode(xmlNode *node, TXMLNode *parent, TXMLNode *previous) :
   fXMLNode(node), fParent(parent), fChildren(0), fNextNode(0),
   fPreviousNode(previous), fAttrList(0)
{
   // TXMLNode constructor.
}

//______________________________________________________________________________
TXMLNode::~TXMLNode()
{
   // Destructor. It deletes the node's child, next sibling and the
   // attribute list.

   delete fChildren;
   delete fNextNode;
   if (fAttrList)
      fAttrList->Delete();
   delete fAttrList;

}

//______________________________________________________________________________
TXMLNode::EXMLElementType TXMLNode::GetNodeType() const
{
   // Returns the node's type.

   return (TXMLNode::EXMLElementType) fXMLNode->type;
}

//______________________________________________________________________________
const char *TXMLNode::GetNodeName() const
{
   // Returns the node's name.

   return (const char *) fXMLNode->name;
}

//______________________________________________________________________________
TXMLNode *TXMLNode::GetChildren()
{
   // Returns the node's child if any, returns 0 if no child.

   if (fChildren)
      return fChildren;

   if (fXMLNode->children){
      fChildren = new TXMLNode(fXMLNode->children, this);
      return fChildren;
   }
   return 0;
}

//______________________________________________________________________________
TXMLNode *TXMLNode::GetParent() const
{
   // Returns the node's parent if any, returns 0 if no parent.

   return fParent;
}

//______________________________________________________________________________
const char *TXMLNode::GetContent() const
{
   // Returns the content if any, or 0.

   if (fXMLNode->content)
      return (const char *) fXMLNode->content;
   return 0;
}

//______________________________________________________________________________
TList *TXMLNode::GetAttributes()
{
   // Returns a list of node's attribute if any,
   // returns 0 if no attribute.

   if (fAttrList)
      return fAttrList;

   if (!HasAttributes())
      return 0;

   fAttrList = new TList();
   xmlAttr *attr_node = fXMLNode->properties;
   for (; attr_node; attr_node = attr_node->next) {
      fAttrList->Add(new TXMLAttr((const char *) attr_node->name,
                                  (const char *) attr_node->children->content));
   }

   return fAttrList;
}

//______________________________________________________________________________
TXMLNode *TXMLNode::GetNextNode()
{
   // Returns the next sibling XMLNode in the DOM tree, if any
   // return 0 if no next node.

   if (fNextNode)
      return fNextNode;

   if (fXMLNode->next) {
      fNextNode = new TXMLNode(fXMLNode->next, fParent, this);
      return fNextNode;
   }
   return 0;
}

//______________________________________________________________________________
TXMLNode *TXMLNode::GetPreviousNode() const
{
   // Returns the previous sibling XMLNode in the DOM tree, if any
   // return 0 if no previous node

   return fPreviousNode;
}

//______________________________________________________________________________
const char *TXMLNode::GetText() const
{
   // Returns the content of a Text node if node is a TextNode, 0 otherwise.

   if (GetNodeType() == kXMLElementNode && HasChildren()) {
      if (fXMLNode->children->type == XML_TEXT_NODE)
         return (const char *) fXMLNode->children->content;
   }
   return 0;
}

//______________________________________________________________________________
Bool_t TXMLNode::HasChildren() const
{
   // Returns true if node has children.

   return fXMLNode->children ? kTRUE : kFALSE;
}

//______________________________________________________________________________
Bool_t TXMLNode::HasNextNode() const
{
   // Returns true if has next node.

   return fXMLNode->next ? kTRUE : kFALSE;
}

//______________________________________________________________________________
Bool_t TXMLNode::HasParent() const
{
   // Returns true if node has parent.

   return fXMLNode->parent ? kTRUE : kFALSE;
}

//______________________________________________________________________________
Bool_t TXMLNode::HasPreviousNode() const
{
   // Returns true if has previous node.

   return fXMLNode->prev ? kTRUE : kFALSE;
}

//______________________________________________________________________________
Bool_t TXMLNode::HasAttributes() const
{
   // Returns true if Element node has attribute.

   return fXMLNode->properties ? kTRUE : kFALSE;
}

//______________________________________________________________________________
const char *TXMLNode::GetNamespaceHref() const
{
   // Returns the URL for the namespace, or 0 if no namespace.

   if (fXMLNode->ns) {
      return (const char *) fXMLNode->ns->href;
   }
   return 0;
}

//______________________________________________________________________________
const char *TXMLNode::GetNamespacePrefix() const
{
   // Returns prefix for the namespace, or 0 if no namespace.

   if (fXMLNode->ns) {
      return (const char *) fXMLNode->ns->prefix;
   }
   return 0;
}
