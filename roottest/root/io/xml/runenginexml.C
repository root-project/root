// Test TXMLEngine and include capability
// Author: Sergey Linev

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


void runenginexml(const char* filename = "main.xml")
{
   // First create engine
   TXMLEngine xml;

   // Now try to parse xml file
   // Only file with restricted xml syntax are supported
   XMLDocPointer_t xmldoc = xml.ParseFile(filename);
   if (!xmldoc) {
      std::cerr << "Fail to parse " << filename << std::endl;
      return;
   }

   // take access to main node
   XMLNodePointer_t mainnode = xml.DocGetRootElement(xmldoc);

   // display recursively all nodes and subnodes
   DisplayNode(xml, mainnode, 1);

   // Release memory before exit
   xml.FreeDoc(xmldoc);
}
