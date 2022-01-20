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
\class TXMLNode
\ingroup IO

TXMLNode contains a pointer to xmlNode, which is a node under the
DOM tree. A node can be an Element, an Attribute, a Text Node
or a Comment Node.
One can navigate the DOM tree by accessing the siblings and
parent or child nodes. Also retriving the Attribute or the Text in
an Element node.
*/

#include "TXMLNode.h"
#include "TXMLAttr.h"
#include "TList.h"
#include <libxml/tree.h>


ClassImp(TXMLNode);

////////////////////////////////////////////////////////////////////////////////
/// TXMLNode constructor.

TXMLNode::TXMLNode(_xmlNode *node, TXMLNode *parent, TXMLNode *previous) :
   fXMLNode(node), fParent(parent), fChildren(0), fNextNode(0),
   fPreviousNode(previous), fAttrList(0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor. It deletes the node's child, next sibling and the
/// attribute list.

TXMLNode::~TXMLNode()
{
   delete fChildren;
   delete fNextNode;
   if (fAttrList)
      fAttrList->Delete();
   delete fAttrList;

}

////////////////////////////////////////////////////////////////////////////////
/// Returns the node's type.

TXMLNode::EXMLElementType TXMLNode::GetNodeType() const
{
   return (TXMLNode::EXMLElementType) fXMLNode->type;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the node's name.

const char *TXMLNode::GetNodeName() const
{
   return (const char *) fXMLNode->name;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the node's child if any, returns 0 if no child.

TXMLNode *TXMLNode::GetChildren()
{
   if (fChildren)
      return fChildren;

   if (fXMLNode->children){
      fChildren = new TXMLNode(fXMLNode->children, this);
      return fChildren;
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the node's parent if any, returns 0 if no parent.

TXMLNode *TXMLNode::GetParent() const
{
   return fParent;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the content if any, or 0.

const char *TXMLNode::GetContent() const
{
   if (fXMLNode->content)
      return (const char *) fXMLNode->content;
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns a list of node's attribute if any,
/// returns 0 if no attribute.

TList *TXMLNode::GetAttributes()
{
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

////////////////////////////////////////////////////////////////////////////////
/// Returns the next sibling XMLNode in the DOM tree, if any
/// return 0 if no next node.

TXMLNode *TXMLNode::GetNextNode()
{
   if (fNextNode)
      return fNextNode;

   if (fXMLNode->next) {
      fNextNode = new TXMLNode(fXMLNode->next, fParent, this);
      return fNextNode;
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the previous sibling XMLNode in the DOM tree, if any
/// return 0 if no previous node

TXMLNode *TXMLNode::GetPreviousNode() const
{
   return fPreviousNode;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the content of a Text node if node is a TextNode, 0 otherwise.

const char *TXMLNode::GetText() const
{
   if (GetNodeType() == kXMLElementNode && HasChildren()) {
      if (fXMLNode->children->type == XML_TEXT_NODE)
         return (const char *) fXMLNode->children->content;
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if node has children.

Bool_t TXMLNode::HasChildren() const
{
   return fXMLNode->children ? kTRUE : kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if has next node.

Bool_t TXMLNode::HasNextNode() const
{
   return fXMLNode->next ? kTRUE : kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if node has parent.

Bool_t TXMLNode::HasParent() const
{
   return fXMLNode->parent ? kTRUE : kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if has previous node.

Bool_t TXMLNode::HasPreviousNode() const
{
   return fXMLNode->prev ? kTRUE : kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if Element node has attribute.

Bool_t TXMLNode::HasAttributes() const
{
   return fXMLNode->properties ? kTRUE : kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the URL for the namespace, or 0 if no namespace.

const char *TXMLNode::GetNamespaceHref() const
{
   if (fXMLNode->ns) {
      return (const char *) fXMLNode->ns->href;
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns prefix for the namespace, or 0 if no namespace.

const char *TXMLNode::GetNamespacePrefix() const
{
   if (fXMLNode->ns) {
      return (const char *) fXMLNode->ns->prefix;
   }
   return 0;
}
