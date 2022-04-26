/// \file
/// \ingroup tutorial_xml
///
/// Example to read, modify and store xml file, using TXMLEngine class
/// The input file, produced by xmlnewfile.C macro is used
/// If you need full xml syntax support, use TXMLParser instead
///
/// \macro_output
/// \macro_code
///
/// \author Sergey Linev

#include "TXMLEngine.h"

// scan node and returns number of childs
// for each child create info node with name and number of childs
int ScanNode(TXMLEngine &xml, XMLNodePointer_t node)
{
   int cnt = 0;
   XMLNodePointer_t child = xml.GetChild(node);
   while (child) {
      cnt++;

      int numsub = ScanNode(xml, child);

      // create new <info> node
      XMLNodePointer_t info = xml.NewChild(node, xml.GetNS(child), "info");

      // set name and num attributes of info node
      xml.NewAttr(info, 0, "name", xml.GetNodeName(child));
      if (numsub > 0) xml.NewIntAttr(info, "num", numsub);

      // move it after current node
      xml.AddChildAfter(node, info, child);

      // set pointer to new node
      child = info;

      xml.ShiftToNext(child);
   }
   return cnt;
}

void xmlmodifyfile(const char* filename = "example.xml")
{
   // First create engine
   TXMLEngine xml;

   // Now try to parse xml file
   XMLDocPointer_t xmldoc = xml.ParseFile(filename);
   if (xmldoc) {
      // recursively scan all nodes, insert new when required
      ScanNode(xml, xml.DocGetRootElement(xmldoc));

      // Save document to file
      xml.SaveDoc(xmldoc, "modify.xml");

      // Release memory before exit
      xml.FreeDoc(xmldoc);
   }
}
