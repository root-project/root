// @(#)root/xml:$Name$:$Id$
// Author: Sergey Linev  10.05.2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TXMLEngine.h"

#include "Riostream.h"

ClassImp(TXMLEngine);

struct SXmlAttr {
   SXmlAttr  *next;
   char      name; // this is first symbol of attribute name, if 0 this is special attribute 
};

struct SXmlNode {
   SXmlAttr  *attr;
   SXmlAttr  *ns;
   SXmlNode  *next;
   SXmlNode  *child;
   SXmlNode  *lastchild;
   SXmlNode  *parent;
   char       name;    // this is start of node name, if 0 next byte is start of content
};

struct SXmlDoc {
   SXmlNode  *fRootNode;
   char      *version;
   char      *dtdname;
   char      *dtdroot;
};

//Int_t TXMLEngine::fNumNodes = 0;

class TXMLOutputStream {
   protected:
   
      ostream *fOut;   
      char    *fBuf;
      char    *fCurrent;
      char    *fMaxAddr; 
      char    *fLimitAddr;
   
   public:
      TXMLOutputStream(const char* filename, Int_t bufsize = 20000)
      {
        fOut = new ofstream(filename);
        fBuf = (char*) malloc(bufsize);
        fCurrent = fBuf;
        fMaxAddr = fBuf + bufsize;
        fLimitAddr = fBuf + int(bufsize*0.75);
      }
      
      virtual ~TXMLOutputStream() 
      {
        if (fCurrent!=fBuf) OutputCurrent();
        delete fOut;
      }
      
      void OutputCurrent()
      {
         if (fCurrent!=fBuf)
           fOut->write(fBuf, fCurrent-fBuf);
         fCurrent = fBuf;  
      }
      
      void Write(const char* str)
      {
         int len = strlen(str);
         if (fCurrent+len>=fMaxAddr) {
           OutputCurrent();
           fOut->write(str,len); 
         } else {
           while (*str)
             *fCurrent++ = *str++;
           if (fCurrent>fLimitAddr)
             OutputCurrent();
         }
      }
      
      void Put(char symb, Int_t cnt=1)
      {
        if (fCurrent+cnt>=fMaxAddr) 
          OutputCurrent();
        if (fCurrent+cnt>=fMaxAddr)
          for(int n=0;n<cnt;n++)
            fOut->put(symb);
          else {
            for(int n=0;n<cnt;n++)
              *fCurrent++ = symb;  
            if (fCurrent>fLimitAddr)
              OutputCurrent();  
          }
      }
};

class TXMLInputStream {
   protected:
      istream  *fInp;
      char     *fBuf;
      Int_t     fBufSize;
      Int_t     fBufLength;
      
      char     *fMaxAddr;
      char     *fLimitAddr;
      
      Int_t     fTotalPos;
      Int_t     fCurrentLine;
   public:

     char      *fCurrent;
      
   TXMLInputStream(const char* filename, Int_t ibufsize) 
   {
      fInp = new ifstream(filename);
      
      fBufSize = ibufsize;
      fBuf = (char*) malloc(fBufSize);
      fBufLength = 0;
      
      fCurrent = 0;
      fMaxAddr = 0;
      
      int len = DoRead(fBuf, fBufSize);
      fCurrent = fBuf;
      fMaxAddr = fBuf+len;
      fLimitAddr = fBuf + int(len*0.75);
            
      fTotalPos = 0;
      fCurrentLine = 1;
   }
   
   virtual ~TXMLInputStream() 
   {
      delete fInp;
      free(fBuf);
   }
   
   bool eof() { return fInp->eof(); }
   
   int DoRead(char* buf, int maxsize)
   {
     if (eof()) return 0;
     fInp->get(buf,maxsize-1,0);
     return strlen(buf);
   }
   
   Bool_t ExpandStream() {
     if (eof()) return kFALSE;
     fBufSize*=2;
     int curlength = fMaxAddr - fBuf;
     realloc(fBuf, fBufSize);
     int len = DoRead(fMaxAddr, fBufSize-curlength);
     if (len==0) return kFALSE;
     fMaxAddr+=len;
     fLimitAddr += int(len*0.75);
     return kTRUE;
   }
   
   Bool_t ShiftStream() {
     if (fCurrent<fLimitAddr) return kTRUE; // everything ok, can cntinue
     if (eof()) return kTRUE;
     int curlength = fMaxAddr - fCurrent;
     memcpy(fBuf, fCurrent, curlength+1); // copy with end 0
     fCurrent = fBuf;
     fMaxAddr = fBuf + curlength;
     fLimitAddr = fBuf + int(curlength*0.75);
     int len = DoRead(fMaxAddr, fBufSize - curlength);
     fMaxAddr+=len;
     fLimitAddr += int(len*0.75);
     return kTRUE;
   }

   Int_t  TotalPos() { return fTotalPos; }
   
   Int_t CurrentLine() { return fCurrentLine; }
   
   Bool_t ShiftCurrent(Int_t sz = 1) 
   {
      for(int n=0;n<sz;n++) {
        if (*fCurrent==10) fCurrentLine++;
        if (fCurrent>=fLimitAddr) {
           ShiftStream();
           if (fCurrent>=fMaxAddr) return kFALSE;
        }
        fCurrent++;
      }
      fTotalPos+=sz;
      return kTRUE;
   }
   
   Bool_t SkipSpaces(Bool_t tillendl = kFALSE) 
   {
     do {  
         char symb = *fCurrent;
         if ((symb>26) && (symb!=' ')) return kTRUE;
         
         if (!ShiftCurrent()) return kFALSE;
         
         if (tillendl && (symb==10)) return kTRUE;
     } while (fCurrent<fMaxAddr);
     return kFALSE;
   }
   
   Bool_t CheckFor(const char* str)
   {
      int len = strlen(str); 
      while (fCurrent+len>fMaxAddr)
         if (!ExpandStream()) return kFALSE;
      char* curr = fCurrent;   
      while (*str != 0)     
        if (*str++ != *curr++) return kFALSE;
      return ShiftCurrent(len);  
   }
   
   Bool_t SearchFor(const char* str) 
   {
      int len = strlen(str); 
      
      do {
        while (fCurrent+len>fMaxAddr)
         if (!ExpandStream()) return kFALSE;
        char* curr = fCurrent;
        const char* chk = str;
        Bool_t find = kTRUE;   
        while (*chk != 0)     
          if (*chk++ != *curr++) find = kFALSE;
        if (find) return kTRUE;  
        if (!ShiftCurrent()) return kFALSE;
      } while (fCurrent<fMaxAddr);
      return kFALSE;      
   }
   
   Int_t LocateIdentifier()
   {
      char symb = *fCurrent;
      bool ok = (((symb>='a') && (symb<='z')) ||
                 ((symb>='A') && (symb<='Z')) || 
                 (symb=='_'));
     if (!ok) return 0;
      
      char* curr = fCurrent;
      
      do {
        curr++;
        if (curr>=fMaxAddr)
          if (!ExpandStream()) return 0;
        symb = *curr;
        ok = ((symb>='a') && (symb<='z')) ||
              ((symb>='A') && (symb<='Z')) || 
              ((symb>='0') && (symb<='9')) ||
              (symb==':') || (symb=='_');
        if (!ok) return curr-fCurrent;
      } while (curr<fMaxAddr);
      return 0;
   }
   
   Int_t LocateContent()
   {
      char* curr = fCurrent;
      while (curr<fMaxAddr) {
         char symb = *curr;
         if (symb=='<') return curr - fCurrent;
         curr++;
         if (curr>=fMaxAddr)
          if (!ExpandStream()) return -1;
      }
      return -1; 
   }
   
   Int_t LocateAttributeValue(char* start)
   {
      char* curr = start;
      if (curr>=fMaxAddr)
        if (!ExpandStream()) return 0;
      if (*curr!='=') return 0;
      curr++;
      if (curr>=fMaxAddr)
        if (!ExpandStream()) return 0;
      if (*curr!='"') return 0;
      do {
        curr++;
        if (curr>=fMaxAddr)
          if (!ExpandStream()) return 0;
        if (*curr=='"') return curr-start+1;
      } while (curr<fMaxAddr); 
      return 0;
   }
};


//______________________________________________________________________________
TXMLEngine::TXMLEngine()
{

}


//______________________________________________________________________________
TXMLEngine::~TXMLEngine()
{
}

//______________________________________________________________________________
Bool_t TXMLEngine::HasAttr(xmlNodePointer xmlnode, const char* name)
{
   if (xmlnode==0) return kFALSE;
   SXmlAttr* attr = ((SXmlNode*)xmlnode)->attr;
   while (attr!=0) {
     if (strcmp(&(attr->name),name)==0) return kTRUE;
     attr = attr->next;
   }
   return kFALSE;
}

//______________________________________________________________________________
const char* TXMLEngine::GetAttr(xmlNodePointer xmlnode, const char* name)
{
   if (xmlnode==0) return 0;
   SXmlAttr* attr = ((SXmlNode*)xmlnode)->attr;
   while (attr!=0) {
     if (strcmp(&(attr->name),name)==0)
       return &(attr->name) + strlen(name) + 1;
     attr = attr->next;
   }
   return 0;
}

//______________________________________________________________________________
Int_t TXMLEngine::GetIntAttr(xmlNodePointer xmlnode, const char* name) 
{
   if (xmlnode==0) return 0;
   Int_t res = 0;
   const char* attr = GetAttr(xmlnode, name);
   if (attr) sscanf(attr, "%d", &res);
   return res;
}

//______________________________________________________________________________
xmlAttrPointer TXMLEngine::NewAttr(xmlNodePointer xmlnode, xmlNsPointer,
                                         const char* name, const char* value)
{
   if (xmlnode==0) return 0;
   
   int namelen = strlen(name), valuelen = strlen(value);
   SXmlAttr* attr = (SXmlAttr*) AllocateAttr(namelen, valuelen, xmlnode);

   char* attrname = &(attr->name);   
   strcpy(attrname, name);
   attrname += (namelen + 1);
   if ((value!=0) && (valuelen>0))
     strcpy(attrname, value);
   else
     *attrname=0;  
   
   return (xmlAttrPointer) attr;
}

//______________________________________________________________________________
xmlAttrPointer TXMLEngine::NewIntAttr(xmlNodePointer xmlnode,
                                      const char* name,
                                      Int_t value)
{
  char sbuf[30];
  sprintf(sbuf,"%d",value);
  return NewAttr(xmlnode, 0, name, sbuf);
}                                      

//______________________________________________________________________________
void TXMLEngine::FreeAttr(xmlNodePointer xmlnode, const char* name)
{
   if (xmlnode==0) return;
   SXmlAttr* attr = ((SXmlNode*) xmlnode)->attr;
   SXmlAttr* prev = 0;
   while (attr!=0) {
     if (strcmp(&(attr->name),name)==0) {
       if (prev!=0) prev->next = attr->next;
               else ((SXmlNode*) xmlnode)->attr = attr->next;
       //fNumNodes--;        
       free(attr);
       return;
     }
     
     prev = attr;
     attr = attr->next;
   }
}
      
//______________________________________________________________________________
xmlNodePointer TXMLEngine::NewChild(xmlNodePointer parent, xmlNsPointer ns,
                                          const char* name, const char* content)
{
   SXmlNode* node = (SXmlNode*) AllocateNode(strlen(name), parent);
   
   strcpy(&(node->name), name);
   node->ns = (SXmlAttr*) ns;
   if (content!=0) {
      int contlen = strlen(content);
      if (contlen>0) {
         SXmlNode* contnode = (SXmlNode*) AllocateNode(contlen+1, node);
         char* cptr = &(contnode->name);
         *cptr = 0;
         cptr++;
         strcpy(cptr,content);
      }
   }
   
   return (xmlNodePointer) node;
}

//______________________________________________________________________________
xmlNsPointer TXMLEngine::NewNS(xmlNodePointer xmlnode, const char* reference, const char* name)
{
   SXmlNode* node = (SXmlNode*) xmlnode;
   if (name==0) name = &(node->name);
   char* nsname = new char[strlen(name)+7];
   strcpy(nsname, "xmlns:");
   strcat(nsname, name);
   
   SXmlAttr* first = node->attr;
   node->attr = 0;
   
   SXmlAttr* nsattr = (SXmlAttr*) NewAttr(xmlnode, 0, nsname, reference);
   
   node->attr = nsattr;
   nsattr->next = first;
   
   node->ns = nsattr;
   delete[] nsname;
   return (xmlNsPointer) nsattr;
   
}      

//______________________________________________________________________________
void TXMLEngine::AddChild(xmlNodePointer parent, xmlNodePointer child)
{
   if ((parent==0) || (child==0)) return;
   SXmlNode* pnode = (SXmlNode*) parent;
   SXmlNode* cnode = (SXmlNode*) child;
   cnode->parent = pnode;
   if (pnode->lastchild==0) {
     pnode->child = cnode;
     pnode->lastchild = cnode;
   } else {
     //SXmlNode* ch = pnode->child;
     //while (ch->next!=0) ch=ch->next;
     pnode->lastchild->next = cnode;
     pnode->lastchild = cnode;
   }
}

//______________________________________________________________________________
void TXMLEngine::UnlinkNode(xmlNodePointer xmlnode)
{
   if (xmlnode==0) return;
   SXmlNode* node = (SXmlNode*) xmlnode;
   
   SXmlNode* parent = node->parent;
   
   if (parent==0) return;
   
   if (parent->child==node) {
      parent->child = node->next;
      if (parent->lastchild==node)
        parent->lastchild = node->next;
   } else {
      SXmlNode* ch = parent->child;
      while (ch->next!=node) ch = ch->next;
      ch->next = node->next;
      if (parent->lastchild == node)
        parent->lastchild = ch;
   }
}

//______________________________________________________________________________
void TXMLEngine::FreeNode(xmlNodePointer xmlnode)
{
   if (xmlnode==0) return;
   SXmlNode* node = (SXmlNode*) xmlnode;
   
   SXmlNode* child = node->child;
   while (child!=0) {
     SXmlNode* next = child->next;
     FreeNode((xmlNodePointer)child);
     child = next;
   }
   
   SXmlAttr* attr = node->attr;
   while (attr!=0) {
     SXmlAttr* next = attr->next;
     //fNumNodes--;
     free(attr);
     attr = next;
   }
   
   //delete[] node->name;
   // delete[] node->content;
   free(node);
   
   //fNumNodes--;
}

//______________________________________________________________________________
void TXMLEngine::UnlinkFreeNode(xmlNodePointer xmlnode)
{
   UnlinkNode(xmlnode); 
   FreeNode(xmlnode);
}
      
//______________________________________________________________________________
const char* TXMLEngine::GetNodeName(xmlNodePointer xmlnode)
{
  return xmlnode==0 ? 0 : & (((SXmlNode*) xmlnode)->name);
}

//______________________________________________________________________________
const char* TXMLEngine::GetNodeContent(xmlNodePointer xmlnode)
{
   if (xmlnode==0) return 0;
   SXmlNode* node = (SXmlNode*) xmlnode;
   if ((node->child==0) || (node->child->name!=0)) return 0;
   return &(node->child->name) + 1;
}

//______________________________________________________________________________
xmlNodePointer TXMLEngine::GetChild(xmlNodePointer xmlnode)
{
   SXmlNode* res = xmlnode==0 ? 0 :((SXmlNode*) xmlnode)->child;
   // skip content node
   if ((res!=0) && (res->name==0)) res = res->next;
   return (xmlNodePointer) res;
}

//______________________________________________________________________________
xmlNodePointer TXMLEngine::GetParent(xmlNodePointer xmlnode)
{
   return xmlnode==0 ? 0 : (xmlNodePointer) ((SXmlNode*) xmlnode)->parent;
}

//______________________________________________________________________________
xmlNodePointer TXMLEngine::GetNext(xmlNodePointer xmlnode)
{
   return xmlnode==0 ? 0 : (xmlNodePointer) ((SXmlNode*) xmlnode)->next;
}
 
//______________________________________________________________________________
void TXMLEngine::ShiftToNext(xmlNodePointer &xmlnode)
{
   xmlnode = xmlnode==0 ? 0 : (xmlNodePointer) ((SXmlNode*) xmlnode)->next;
}

//______________________________________________________________________________
void TXMLEngine::CleanNode(xmlNodePointer xmlnode)
{
   if (xmlnode==0) return;
   SXmlNode* node = (SXmlNode*) xmlnode;
   
   SXmlNode* child = node->child;
   while (child!=0) {
     SXmlNode* next = child->next;
     FreeNode((xmlNodePointer)child);
     child = next;
   }
   
   node->child = 0;
   node->lastchild = 0;
}

//______________________________________________________________________________
xmlDocPointer TXMLEngine::NewDoc(const char* version)
{
   SXmlDoc* doc = new SXmlDoc;
   doc->fRootNode = 0;
   doc->version = makestr(version);
   doc->dtdname = 0;
   doc->dtdroot = 0;
   return (xmlDocPointer) doc;
}

//______________________________________________________________________________
void TXMLEngine::AssignDtd(xmlDocPointer xmldoc, const char* dtdname, const char* rootname)
{
   if (xmldoc==0) return;
   SXmlDoc* doc = (SXmlDoc*) xmldoc;
   delete[] doc->dtdname; 
   doc->dtdname = makestr(dtdname);
   delete[] doc->dtdroot;
   doc->dtdroot = makestr(rootname);
}
            
//______________________________________________________________________________
void TXMLEngine::FreeDoc(xmlDocPointer xmldoc)
{
   if (xmldoc==0) return;
   SXmlDoc* doc = (SXmlDoc*) xmldoc;
   FreeNode((xmlNodePointer) doc->fRootNode);
   delete[] doc->version;
   delete[] doc->dtdname; 
   delete[] doc->dtdroot;
   delete doc;
}
      
//______________________________________________________________________________
void TXMLEngine::SaveDoc(xmlDocPointer xmldoc, const char* filename, Int_t layout)
{
  if (xmldoc==0) return;
  
  SXmlDoc* doc = (SXmlDoc*) xmldoc;
  
  TXMLOutputStream out(filename, 100000);
  out.Write("<?xml version=\"");
  if (doc->version!=0) out.Write(doc->version);
                  else out.Write("1.0");
  out.Write("\"?>\n");
  
  SaveNode((xmlNodePointer) doc->fRootNode, &out, layout, 0);
}

//______________________________________________________________________________
void TXMLEngine::DocSetRootElement(xmlDocPointer xmldoc, xmlNodePointer xmlnode)
{
   if (xmldoc==0) return;
   SXmlDoc* doc = (SXmlDoc*) xmldoc;
   FreeNode((xmlNodePointer) doc->fRootNode);
   doc->fRootNode = (SXmlNode*) xmlnode;
}

//______________________________________________________________________________
xmlNodePointer TXMLEngine::DocGetRootElement(xmlDocPointer xmldoc)
{
   return (xmldoc==0) ? 0 : (xmlNodePointer) ((SXmlDoc*)xmldoc)->fRootNode;
}

//______________________________________________________________________________
xmlDocPointer TXMLEngine::ParseFile(const char* filename)
{
   if ((filename==0) || (strlen(filename)==0)) return 0;
   
   TXMLInputStream inp(filename, 100000);
   
   if (!inp.SkipSpaces()) return 0;
   if (!inp.CheckFor("<?xml")) return 0;
   if (!inp.SkipSpaces()) return 0;
   if (!inp.SearchFor("?>")) return 0;
   if (!inp.ShiftCurrent(2)) return 0; 
   inp.SkipSpaces(kTRUE); // locate start of next string
   
   Int_t resvalue = 0;
   
   xmlNodePointer mainnode = ReadNode(0, &inp, resvalue);

   if (resvalue<=0) {
     switch(resvalue) {
       case -9: Error("ParseFile", "Multiple name space definitions not allowed, line %d", inp.CurrentLine()); break;
       case -8: Error("ParseFile", "Invalid namespace specification, line %d", inp.CurrentLine()); break;
       case -7: Error("ParseFile", "Invalid attribute value, line %d", inp.CurrentLine()); break;
       case -6: Error("ParseFile", "Invalid identifier for node attribute, line %d", inp.CurrentLine()); break;
       case -5: Error("ParseFile", "Missmatch between open and close nodes, line %d", inp.CurrentLine()); break;
       case -4: Error("ParseFile", "Unexpected close node, line %d", inp.CurrentLine()); break;
       case -3: Error("ParseFile", "Valid identifier for close node is missing, line %d", inp.CurrentLine()); break;
       case -2: Error("ParseFile", "No multiple content entries allowed, line %d", inp.CurrentLine()); break;
       case -1: Error("ParseFile", "Unexpected end of xml file"); break;
       default: Error("ParseFile", "XML syntax error at line %d", inp.CurrentLine()); break;
     }
     FreeNode(mainnode);
     return 0;
   }
      
   if (mainnode==0) return 0;
   xmlDocPointer xmldoc = NewDoc();
   DocSetRootElement(xmldoc, mainnode);
   return xmldoc;
}
      
//______________________________________________________________________________
char* TXMLEngine::makestr(const char* str)
{
   if (str==0) return 0;
   int len = strlen(str);
   if (len==0) return 0;
   char* res = new char[len+1];
   strcpy(res, str);
   return res;
}

//______________________________________________________________________________
char* TXMLEngine::makenstr(const char* str, int len)
{
   if ((str==0) || (len==0)) return 0; 
   char* res = new char[len+1];
   strncpy(res, str, len);
   *(res+len) = 0;
   return res;
}

//______________________________________________________________________________
xmlNodePointer TXMLEngine::AllocateNode(int namelen, xmlNodePointer parent)
{
   //fNumNodes++;
   SXmlNode* node = (SXmlNode*) malloc(sizeof(SXmlNode) + namelen);
   
   node->parent = 0;
   node->ns = 0;
   node->attr = 0;
   node->child = 0;
   node->lastchild = 0;
   node->next = 0;
   
   if (parent!=0)
     AddChild(parent, (xmlNodePointer) node);
   
   return (xmlNodePointer) node;
}

//______________________________________________________________________________
xmlAttrPointer TXMLEngine::AllocateAttr(int namelen, int valuelen, xmlNodePointer xmlnode) {
   
   //fNumNodes++;
   SXmlAttr* attr = (SXmlAttr*) malloc(sizeof(SXmlAttr) + namelen + valuelen + 1);
   
   SXmlNode* node = (SXmlNode*) xmlnode;
   
   attr->next = 0;
   
   if (node->attr==0) 
      node->attr = attr;
   else {
      SXmlAttr* d = node->attr;
      while (d->next!=0) d = d->next;
      d->next = attr;
   }   
   
   return (xmlAttrPointer) attr;
}

//______________________________________________________________________________
xmlNsPointer TXMLEngine::FindNs(xmlNodePointer xmlnode, const char* name)
{
   SXmlNode* node = (SXmlNode*) xmlnode;
   while (node!=0) {
      if (node->ns!=0) {
         const char* nsname = &(node->ns->name) + 6;
         if (strcmp(nsname, name)==0) return node->ns;
      } 
      node = node->parent;
   }
   return 0;
}

//______________________________________________________________________________
void TXMLEngine::TruncateNsExtension(xmlNodePointer xmlnode)
{
   SXmlNode* node = (SXmlNode*) xmlnode;
   if (node==0) return;
   char* colon = strchr(&(node->name),':');
   if (colon==0) return;
   
   char* copyname = &(node->name);
   
   while (*colon!=0)
     *(copyname++) = *(++colon);
   
   //cout << "new name = " << &(node->name) << endl;
}

//______________________________________________________________________________
void TXMLEngine::UnpackSpecialCharacters(char* target, const char* source, int srclen)
{
   while (srclen>0) {
      if (*source=='&') {
        if ((*(source+1)=='l') && (*(source+2)=='t') && (*(source+3)==';')) {
           *target++ = '<'; source+=4; srclen-=4;
        } else
        if ((*(source+1)=='g') && (*(source+2)=='t') && (*(source+3)==';')) {
           *target++ = '>'; source+=4; srclen-=4;
        } else
        if ((*(source+1)=='a') && (*(source+2)=='m') && (*(source+3)=='p') && (*(source+4)==';')) {
           *target++ = '&'; source+=5; srclen-=5;
        } else
         { *target++ = *source++; srclen--; }
      } else { 
        *target++ = *source++; 
        srclen--; 
      }
   }
   *target = 0;
}

//______________________________________________________________________________
void TXMLEngine::OutputValue(char* value, TXMLOutputStream* out)
{
   if (value==0) return; 
   char* find = value;
   while (*find!=0) {
   if ((*find=='<') || 
       (*find=='>') || 
       (*find=='&')) break;
     find++;
   }
   if (*find==0) {
     out->Write(value);  
     return;
   }
   
   char* last = value;
   find = 0;
   while ((find=strpbrk(last,"<&>")) !=0 ) {
      char symb = *find;
      *find = 0;
      out->Write(last);
      *find = symb;
      last = find+1;
      if (symb=='<') out->Write("&lt;"); else
      if (symb=='>') out->Write("&gt;"); else out->Write("&amp;");
   }
   if (*last!=0)
     out->Write(last);
}

//______________________________________________________________________________
void TXMLEngine::SaveNode(xmlNodePointer xmlnode, TXMLOutputStream* out, Int_t layout, Int_t level)
{
   if (xmlnode==0) return;
   SXmlNode* node = (SXmlNode*) xmlnode;
   
   bool issingleline = (node->name!=0) && (node->child==0);
   
   if (layout>0) out->Put(' ', level);
   
   if (node->name!=0) {    
      
      out->Put('<');
      // we suppose that ns is always first attribute
      if ((node->ns!=0) && (node->ns!=node->attr)) {
         out->Write(&(node->ns->name)+6);
         out->Put(':');
      }
      out->Write(&(node->name));
      
      SXmlAttr* attr = node->attr;
      while (attr!=0) {
         out->Put(' ');
         char* attrname = &(attr->name);
         out->Write(attrname);
         out->Write("=\"");
         attrname += strlen(attrname) + 1;
         OutputValue(attrname, out);
         out->Put('\"');      
         attr = attr->next;
      }
      if (issingleline) out->Write("/>");
                   else out->Put('>');
      if (layout>0) out->Put('\n');
   } else {
      // this is output for content
      out->Write(&(node->name)+1);
      return;
   }
   
   if (issingleline) return;
   
   SXmlNode* child = node->child;
   while (child!=0) {
      SaveNode((xmlNodePointer) child, out, layout, level+2);
      child = child->next;
   }   
   
   if (node->name!=0) {    
     if (layout>0)
        out->Put(' ',level);
     out->Write("</");
     // we suppose that ns is always first attribute
     if ((node->ns!=0) && (node->ns!=node->attr)) {
        out->Write(&(node->ns->name)+6);
        out->Put(':');
     }
     out->Write(&(node->name));
     out->Put('>');
     if (layout>0) out->Put('\n');
   }
}

//______________________________________________________________________________
xmlNodePointer TXMLEngine::ReadNode(xmlNodePointer xmlparent, TXMLInputStream* inp, Int_t& resvalue)
{
// resvalue <= 0 if error
// resvalue == 1 if this is endnode of parent 
// resvalue == 2 if this is child   
   
   resvalue = 0; 
    
   if (inp==0) return 0;
   if (!inp->SkipSpaces()) { resvalue = -1; return 0; }
   SXmlNode* parent = (SXmlNode*) xmlparent;

   SXmlNode* node = 0;

   if (*inp->fCurrent!='<') {
      // here should be reading of element content
      // only one entry for content is supported, only before any other childs
      if ((parent==0) || (parent->child!=0)) { resvalue = -2; return 0; }
      int contlen = inp->LocateContent();
      if (contlen<0) return 0;
      
      SXmlNode* contnode = (SXmlNode*) AllocateNode(contlen+1, xmlparent);
      char* contptr = &(contnode->name);
      *contptr = 0;
      contptr++;
      UnpackSpecialCharacters(contptr, inp->fCurrent, contlen);
      if (!inp->ShiftCurrent(contlen)) return 0;
      resvalue = 2;
      return contnode;
   } else
     // skip "<" symbol
     if (!inp->ShiftCurrent()) return 0;
   
   if (*inp->fCurrent=='/') {
      // this is a starting of closing node
      if (!inp->ShiftCurrent()) return 0;
      if (!inp->SkipSpaces()) return 0;
      Int_t len = inp->LocateIdentifier();
      if (len<=0) { resvalue = -3; return 0; }

      if (parent==0) { resvalue = -4; return 0; }
      
      if (strncmp(&(parent->name), inp->fCurrent, len)!=0) {
         resvalue = -5;
         return 0; 
      }
      
      if (!inp->ShiftCurrent(len)) return 0;
      
      if (!inp->SkipSpaces()) return 0;
      if (*inp->fCurrent!='>') return 0;
      if (!inp->ShiftCurrent()) return 0;
      
      if (parent->ns!=0)
        TruncateNsExtension((xmlNodePointer)parent);
      
      inp->SkipSpaces(kTRUE); // locate start of next string 
      resvalue = 1; 
      return 0;
   }
   
   
   if (!inp->SkipSpaces()) return 0;
   Int_t len = inp->LocateIdentifier();
   if (len<=0) return 0;
   node = (SXmlNode*) AllocateNode(len, xmlparent);
   char* nameptr = &(node->name);
   
   strncpy(nameptr, inp->fCurrent, len);
   nameptr+=len;
   *nameptr = 0;
   
   char* colon = strchr(&(node->name),':');
   if ((colon!=0) && (parent!=0)) {
      *colon = 0;
      node->ns = (SXmlAttr*) FindNs(xmlparent, &(node->name));
      *colon =':';      
   }
      
   if (!inp->ShiftCurrent(len)) return 0;
   
   do {
      if (!inp->SkipSpaces()) return 0;
    
      char nextsymb = *inp->fCurrent;
        
      if (nextsymb=='/') {  // this is end of short node like <node ... />
         if (!inp->ShiftCurrent()) return 0;
         if (*inp->fCurrent=='>') {
            if (!inp->ShiftCurrent()) return 0; 
            
            if (node->ns!=0)
               TruncateNsExtension((xmlNodePointer) node);
            
            inp->SkipSpaces(kTRUE); // locate start of next string 
            resvalue = 2; 
            return node;
         } else return 0;
      } else 
      if (nextsymb=='>') { // this is end of parent node, lets find all childs
          
         if (!inp->ShiftCurrent()) return 0; 
          
         do {
            ReadNode(node, inp, resvalue);
         } while (resvalue==2);
         
         if (resvalue==1) {
           resvalue = 2;
           return node;
         } else return 0;
      } else {
        Int_t attrlen = inp->LocateIdentifier();
        if (attrlen<=0) { resvalue = -6; return 0; }
        
        char* valuestart = inp->fCurrent+attrlen;
        
        int valuelen = inp->LocateAttributeValue(valuestart);
        if (valuelen<3) { resvalue = -7; return 0; }
        
        SXmlAttr* attr = (SXmlAttr*) AllocateAttr(attrlen, valuelen-3, (xmlNodePointer) node);
        
        char* attrname = &(attr->name);
        strncpy(attrname, inp->fCurrent, attrlen);
        attrname+=attrlen;
        *attrname = 0;
        attrname++;
        UnpackSpecialCharacters(attrname, valuestart+2, valuelen-3);
        
        if (!inp->ShiftCurrent(attrlen+valuelen)) return 0;
        
        attrname = &(attr->name);
         
        if ((strlen(attrname)>6) && (strstr(attrname,"xmlns:")==attrname)) {
          if (strcmp(&(node->name), attrname + 6)!=0) {
             resvalue = -8;
             return 0;
          }
          if (node->ns!=0) {
             resvalue = -9;
             return 0;   
          }  
          node->ns = attr;   
        }
      }
   } while (true);

   return 0;
}
