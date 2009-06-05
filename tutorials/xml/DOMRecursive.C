//////////////////////////////////////////////////////////////////////////////
//
// ROOT implementation of a XML DOM Parser
//
// This is an example of how Dom Parser walks the DOM tree recursively.
// This example will parse any xml file.
// 
// To run this program
// .x DOMRecursive.C+
// 
// Requires: person.xml 
// 
//////////////////////////////////////////////////////////////////////////////

#include <Riostream.h>
#include <TDOMParser.h>
#include <TXMLNode.h>
#include <TXMLAttr.h>
#include <TList.h>


void ParseContext(TXMLNode *node)
{
   for ( ; node; node = node->GetNextNode()) {
      if (node->GetNodeType() == TXMLNode::kXMLElementNode) { // Element Node
         cout << node->GetNodeName() << ": ";
         if (node->HasAttributes()) {
            TList* attrList = node->GetAttributes();
            TIter next(attrList);
            TXMLAttr *attr;
            while ((attr =(TXMLAttr*)next())) {
               cout << attr->GetName() << ":" << attr->GetValue();
            }
         }
     }
     if (node->GetNodeType() == TXMLNode::kXMLTextNode) { // Text node
        cout << node->GetContent();
     }
     if (node->GetNodeType() == TXMLNode::kXMLCommentNode) { //Comment node
        cout << "Comment: " << node->GetContent();
     }

     ParseContext(node->GetChildren());
   }
}


void DOMRecursive()
{
  TDOMParser *domParser = new TDOMParser();

  domParser->SetValidate(false); // do not validate with DTD
  domParser->ParseFile("person.xml");

  TXMLNode *node = domParser->GetXMLDocument()->GetRootNode();

  ParseContext(node);
}
