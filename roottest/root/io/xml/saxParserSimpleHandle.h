#ifndef __SAXHandler_h__
#define __SAXHandler_h__

#include <Rtypes.h>
#include <TQObject.h>
#include <TXMLAttr.h>

#include <iostream>

class TList;

class SAXHandler : public TQObject {
private:
   Bool_t   fQuiet = false;

public:
   SAXHandler() { }
   void     Quiet() {
      fQuiet = true;
   };
   void     OnStartDocument() { }
   void     OnEndDocument();
   void     OnStartElement(const char *, const TList *);
   void     OnEndElement(const char *);
   void     OnCharacters(const char *);
   void     OnComment(const char *);
   void     OnWarning(const char *);
   void     OnError(const char *);
   void     OnFatalError(const char *);
   void     OnCdataBlock(const char *, Int_t);
   ClassDef(SAXHandler, 0);
};

void SAXHandler::OnEndDocument()
{
   if (!fQuiet) std::cout << std::endl;
}

void SAXHandler::OnStartElement(const char *name, const TList *attributes)
{
   if (!fQuiet) std::cout << "<" << name;

   TXMLAttr *attr;

   TIter next(attributes);
   if (!fQuiet) {
      while ((attr = (TXMLAttr *) next())) {
         std::cout << " " << attr->GetName() << "=\"" << attr->GetValue() << "\"";
      }
   }

   if (!fQuiet) std::cout  << ">";
}

void SAXHandler::OnEndElement(const char *name)
{
   if (!fQuiet) std::cout << "</" << name << ">";
}

void SAXHandler::OnCharacters(const char *characters)
{
   if (!fQuiet) std::cout << characters;
}

void SAXHandler::OnComment(const char *text)
{
   if (!fQuiet) std::cout << "<!--" << text << "-->";
}

void SAXHandler::OnWarning(const char *text)
{
   if (!fQuiet) std::cout << "Warning: " << text << std::endl;
}

void SAXHandler::OnError(const char *text)
{
   if (!fQuiet) std::cerr << "Error: " << text << std::endl ;
}

void SAXHandler::OnFatalError(const char *text)
{
   if (!fQuiet) std::cerr << "FatalError: " << text << std::endl ;
}

void SAXHandler::OnCdataBlock(const char *text, Int_t len)
{
   if (!fQuiet) std::cout << "OnCdataBlock() " << text;
}


#endif /* __SAXHandler_h__ */
