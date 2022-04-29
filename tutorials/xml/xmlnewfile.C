/// \file
/// \ingroup tutorial_xml
///
/// Example to create a new xml file with the TXMLEngine class
///
/// \macro_code
///
/// \author Sergey Linev

#include "TXMLEngine.h"

void xmlnewfile(const char* filename = "example.xml")
{
   // First create engine
   TXMLEngine xml;

   // Create main node of document tree
   XMLNodePointer_t mainnode = xml.NewChild(0, 0, "main");

   // Simple child node with content inside
   xml.NewChild(mainnode, 0, "child1", "Content of child1 node");

   // Other child node with attributes
   XMLNodePointer_t child2 = xml.NewChild(mainnode, 0, "child2");
   xml.NewAttr(child2, 0, "attr1","value1");
   xml.NewAttr(child2, 0, "attr2","value2");

   // Child node with subnodes
   XMLNodePointer_t child3 = xml.NewChild(mainnode, 0, "child3");
   xml.NewChild(child3, 0, "subchild1", "subchild1 content");
   xml.NewChild(child3, 0, "subchild2", "subchild2 content");
   xml.NewChild(child3, 0, "subchild3", "subchild3 content");

   // Child node with subnodes and namespace
   XMLNodePointer_t child4 = xml.NewChild(mainnode, 0, "child4");
   XMLNsPointer_t ns4 = xml.NewNS(child4, "http://website/webpage");
   xml.NewChild(child4, ns4, "subchild1", "subchild1 content");
   xml.NewChild(child4, ns4, "subchild2", "subchild2 content");
   xml.NewChild(child4, ns4, "subchild3", "subchild3 content");

   // now create document and assign main node of document
   XMLDocPointer_t xmldoc = xml.NewDoc();
   xml.DocSetRootElement(xmldoc, mainnode);

   // Save document to file
   xml.SaveDoc(xmldoc, filename);

   // Release memory before exit
   xml.FreeDoc(xmldoc);
}
