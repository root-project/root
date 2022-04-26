/// \file
/// \ingroup tutorial_xml
///
/// Example to read and parse any xml file, supported by TXMLEngine class
/// The input file, produced by xmlnewfile.C macro is used
/// If you need full xml syntax support, use TXMLParser instead
///
/// \macro_output
/// \macro_code
///
/// \author Sergey Linev

#include "TXMLEngine.h"
#include <stdio.h>

void DisplayNode(TXMLEngine &xml, XMLNodePointer_t node, Int_t level)
{
   // this function display all accessible information about xml node and its children

   printf("%*c node: %s\n", level, ' ', xml.GetNodeName(node));

   // display namespace
   XMLNsPointer_t ns = xml.GetNS(node);
   if (ns != 0)
      printf("%*c namespace: %s refer: %s\n", level + 2, ' ', xml.GetNSName(ns), xml.GetNSReference(ns));

   // display attributes
   XMLAttrPointer_t attr = xml.GetFirstAttr(node);
   while (attr != 0) {
      printf("%*c attr: %s value: %s\n", level + 2, ' ', xml.GetAttrName(attr), xml.GetAttrValue(attr));
      attr = xml.GetNextAttr(attr);
   }

   // display content (if exists)
   const char *content = xml.GetNodeContent(node);
   if (content != 0)
      printf("%*c cont: %s\n", level + 2, ' ', content);

   // display all child nodes
   XMLNodePointer_t child = xml.GetChild(node);
   while (child != 0) {
      DisplayNode(xml, child, level + 2);
      child = xml.GetNext(child);
   }
}

void xmlreadfile(const char* filename = "example.xml")
{
   // First create engine
   TXMLEngine xml;

   // Now try to parse xml file
   // Only file with restricted xml syntax are supported
   XMLDocPointer_t xmldoc = xml.ParseFile(filename);
   if (!xmldoc) return;

   // take access to main node
   XMLNodePointer_t mainnode = xml.DocGetRootElement(xmldoc);

   // display recursively all nodes and subnodes
   DisplayNode(xml, mainnode, 1);

   // Release memory before exit
   xml.FreeDoc(xmldoc);
}
