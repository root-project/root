#ifndef ROOT_TXMLEngine
#define ROOT_TXMLEngine

#ifndef ROOT_TObject
#include "TObject.h"
#endif

typedef void* xmlNodePointer;
typedef void* xmlNsPointer;
typedef void* xmlAttrPointer;
typedef void* xmlDocPointer;

class TXMLInputStream;
class TXMLOutputStream;

class TXMLEngine : public TObject {
   public:
      TXMLEngine();
      virtual ~TXMLEngine();
      
      Bool_t            HasAttr(xmlNodePointer xmlnode, const char* name);
      const char*       GetAttr(xmlNodePointer xmlnode, const char* name);
      Int_t             GetIntAttr(xmlNodePointer node, const char* name);
      xmlAttrPointer    NewAttr(xmlNodePointer xmlnode, xmlNsPointer,
                                const char* name, const char* value);
      xmlAttrPointer    NewIntAttr(xmlNodePointer xmlnode, const char* name, Int_t value);
      void              FreeAttr(xmlNodePointer xmlnode, const char* name);
      xmlNodePointer    NewChild(xmlNodePointer parent, xmlNsPointer ns,
                                 const char* name, const char* content = 0);
      xmlNsPointer      NewNS(xmlNodePointer xmlnode, const char* reference, const char* name = 0);
      void              AddChild(xmlNodePointer parent, xmlNodePointer child);
      void              UnlinkNode(xmlNodePointer node);
      void              FreeNode(xmlNodePointer xmlnode);
      void              UnlinkFreeNode(xmlNodePointer xmlnode);
      const char*       GetNodeName(xmlNodePointer xmlnode);
      const char*       GetNodeContent(xmlNodePointer xmlnode);
      xmlNodePointer    GetChild(xmlNodePointer xmlnode);
      xmlNodePointer    GetParent(xmlNodePointer xmlnode);
      xmlNodePointer    GetNext(xmlNodePointer xmlnode);
      void              ShiftToNext(xmlNodePointer &xmlnode);
      Bool_t            IsEmptyNode(xmlNodePointer) { return kFALSE; } // obsolete, should not be used
      void              SkipEmpty(xmlNodePointer &) {}                 // obsolete, should not be used
      void              CleanNode(xmlNodePointer xmlnode);
      xmlDocPointer     NewDoc(const char* version = 0);
      void              AssignDtd(xmlDocPointer xmldoc, const char* dtdname, const char* rootname);
      void              FreeDoc(xmlDocPointer xmldoc);
      void              SaveDoc(xmlDocPointer xmldoc, const char* filename, Int_t layout = 1);
      void              DocSetRootElement(xmlDocPointer xmldoc, xmlNodePointer xmlnode);
      xmlNodePointer    DocGetRootElement(xmlDocPointer xmldoc);
      xmlDocPointer     ParseFile(const char* filename);
      Bool_t            ValidateDocument(xmlDocPointer, Bool_t = kFALSE) { return kFALSE; } // obsolete
   protected:
      char*             makestr(const char* str);   
      char*             makenstr(const char* start, int len);
      xmlNodePointer    AllocateNode(int namelen, xmlNodePointer parent);
      xmlAttrPointer    AllocateAttr(int namelen, int valuelen, xmlNodePointer xmlnode);
      xmlNsPointer      FindNs(xmlNodePointer xmlnode, const char* nsname);
      void              TruncateNsExtension(xmlNodePointer xmlnode);
      void              UnpackSpecialCharacters(char* target, const char* source, int srclen);
      void              OutputValue(Char_t* value, TXMLOutputStream* out);
      void              SaveNode(xmlNodePointer xmlnode, TXMLOutputStream* out, Int_t layout, Int_t level);
      xmlNodePointer    ReadNode(xmlNodePointer xmlparent, TXMLInputStream* inp, Int_t& resvalue);
  
   ClassDef(TXMLEngine,1);
};

#endif

