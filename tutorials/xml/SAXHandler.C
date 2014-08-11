//////////////////////////////////////////////////////////////////////////////
//
// ROOT implementation of a simple SAX Handler.
//
// This handler uses TSAXParser, a SAX Parser using the SAX interface
// of libxml2. This script will output all elements of the original xml
// file, if sucessfully parsed.
//
// To run this program do:
// .x SAXHandler.C
//
// Requires: saxexample.xml
//
//////////////////////////////////////////////////////////////////////////////

#include <Riostream.h>
#include <TList.h>
#include <TSAXParser.h>
#include <TXMLAttr.h>


class SaxHandler {
public:
   SaxHandler() { }

   void     OnStartDocument() { }
   void     OnEndDocument();
   void     OnStartElement(const char*, const TList*);
   void     OnEndElement(const char*);
   void     OnCharacters(const char*);
   void     OnComment(const char*);
   void     OnWarning(const char*);
   void     OnError(const char*);
   void     OnFatalError(const char*);
   void     OnCdataBlock(const char*, Int_t);
};

void SaxHandler::OnEndDocument()
{
   cout << endl;
}

void SaxHandler::OnStartElement(const char *name, const TList *attributes)
{
   cout << "<" << name;

   TXMLAttr *attr;

   TIter next(attributes);
   while ((attr = (TXMLAttr*) next())) {
      cout << " " << attr->GetName() << "=\"" << attr->GetValue() << "\"";
   }

   cout  << ">";
}

void SaxHandler::OnEndElement(const char *name)
{
   cout << "</" << name << ">";
}

void SaxHandler::OnCharacters(const char *characters)
{
   cout << characters;
}

void SaxHandler::OnComment(const char *text)
{
   cout << "<!--" << text << "-->";
}

void SaxHandler::OnWarning(const char *text)
{
   cout << "Warning: " << text << endl;
}

void SaxHandler::OnError(const char *text)
{
   cerr << "Error: " << text << endl ;
}

void SaxHandler::OnFatalError(const char *text)
{
   cerr << "FatalError: " << text << endl ;
}

void SaxHandler::OnCdataBlock(const char *text, Int_t len)
{
   cout << "OnCdataBlock() " << text;
}



void SAXHandler()
{
   TSAXParser *saxParser = new TSAXParser();
   SaxHandler *saxHandler = new SaxHandler();

   saxParser->ConnectToHandler("SaxHandler", saxHandler);
   TString dir = gSystem->DirName(__FILE__);
   saxParser->ParseFile(dir+"/saxexample.xml");
}
