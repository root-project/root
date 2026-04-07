// Test string parsing with TXMLEngine
// Also keep file include via ENTITY
// Author: Sergey Linev   7.04.2026

#include "TXMLEngine.h"

void DisplayNode(TXMLEngine &xml, XMLNodePointer_t node, Int_t level)
{
   // this function display all accessible information about xml node and its children

   // display node with content
   if (xml.IsContentNode(node)) {
      printf("%*c content: %s\n",level,' ', xml.GetNodeName(node));
      return;
   }

   if (xml.IsCommentNode(node)) {
      printf("%*c comment: %s\n",level,' ', xml.GetNodeName(node));
      return;
   }

   if (!xml.IsXmlNode(node))
      return;

   printf("%*c node: %s\n",level,' ', xml.GetNodeName(node));

   // display namespace
   XMLNsPointer_t ns = xml.GetNS(node);
   if (ns)
      printf("%*c namespace: %s refer: %s\n",level+2,' ', xml.GetNSName(ns), xml.GetNSReference(ns));

   // display attributes
   XMLAttrPointer_t attr = xml.GetFirstAttr(node);
   while (attr!=0) {
       printf("%*c attr: %s value: %s\n",level+2,' ', xml.GetAttrName(attr), xml.GetAttrValue(attr));
       attr = xml.GetNextAttr(attr);
   }

   // display all child nodes (including special nodes)
   XMLNodePointer_t child = xml.GetChild(node, kFALSE);
   while (child) {
      DisplayNode(xml, child, level+2);
      child = xml.GetNext(child, kFALSE);
   }
}


void runenginestr()
{
   // First create engine
   TXMLEngine xml;

   // Now try to parse xml string
   // Only file with restricted xml syntax are supported
   XMLDocPointer_t xmldoc = xml.ParseString(
    "<?xml version=\"1.0\"?>\n"
    "   <!DOCTYPE main [\n"
    "     <!ENTITY child4 SYSTEM \"include.xml\">\n"
    "  ]>\n"
    "<main>\n"
    "  <child1>Content of child1 string node</child1>\n"
    "  <child2 attr1=\"strvalue1\" attr2=\"strvalue2\"/>\n"
    "  <child3>\n"
    "     <subchild1>subchild1 string content</subchild1>\n"
    "     <subchild2>subchild2 string content</subchild2>\n"
    "     <subchild3>subchild3 string content</subchild3>\n"
    "  </child3>\n"
    "  &child4;"
    "</main>");
   if (!xmldoc) {
      std::cerr << "Fail to parse string" << std::endl;
      return;
   }

   // take access to main node
   XMLNodePointer_t mainnode = xml.DocGetRootElement(xmldoc);

   // display recursively all nodes and subnodes
   DisplayNode(xml, mainnode, 1);

   // Release memory before exit
   xml.FreeDoc(xmldoc);
}
