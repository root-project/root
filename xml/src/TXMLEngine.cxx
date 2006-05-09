// @(#)root/xml:$Name:  $:$Id: TXMLEngine.cxx,v 1.17 2006/01/20 01:12:13 pcanal Exp $
// Author: Sergey Linev  10.05.2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//________________________________________________________________________
//
//  TXMLEngine class is used to write and read ROOT XML files - TXMLFile.
//  It does not conform to complete xml standard and cannot be used
//  as parser for arbitrary XML files. For such cases TXMLParser should
//  be used. This class was introduced to exclude dependency from
//  external libraries (like libxml2) and improve speed / memory consumption.
//
//________________________________________________________________________

#include "TXMLEngine.h"

#include "Riostream.h"
#include "TString.h"

ClassImp(TXMLEngine);

struct SXmlAttr_t {
   SXmlAttr_t  *fNext;
   char         fName; // this is first symbol of attribute name, if 0 this is special attribute
};

struct SXmlNode_t {
   SXmlAttr_t  *fAttr;
   SXmlAttr_t  *fNs;
   SXmlNode_t  *fNext;
   SXmlNode_t  *fChild;
   SXmlNode_t  *fLastChild;
   SXmlNode_t  *fParent;
   char         fName;    // this is start of node name, if 0 next byte is start of content
};

struct SXmlDoc_t {
   SXmlNode_t  *fRootNode;
   char        *fVersion;
   char        *fDtdName;
   char        *fDtdRoot;
};

class TXMLOutputStream {
protected:

   std::ostream  *fOut;
   TString       *fOutStr;
   char          *fBuf;
   char          *fCurrent;
   char          *fMaxAddr;
   char          *fLimitAddr;

public:
   TXMLOutputStream(const char* filename, Int_t bufsize = 20000)
   {
      fOut = new std::ofstream(filename);
      fOutStr = 0;
      Init(bufsize);
   }

   TXMLOutputStream(TString* outstr, Int_t bufsize = 20000)
   {
      fOut = 0;
      fOutStr = outstr;
      Init(bufsize);
   }
   
   void Init(Int_t bufsize)
   {
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
         if (fOut!=0)
            fOut->write(fBuf, fCurrent-fBuf);
         else
         if (fOutStr!=0) 
            fOutStr->Append(fBuf, fCurrent-fBuf);
      fCurrent = fBuf;
   }
   
   void OutputChar(char symb) 
   {
      if (fOut!=0) fOut->put(symb); else
      if (fOutStr!=0) fOutStr->Append(symb);
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
            OutputChar(symb);
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

   std::istream  *fInp;
   const char    *fInpStr;
   Int_t          fInpStrLen;    
   
   char          *fBuf;
   Int_t          fBufSize;
   Int_t          fBufLength;

   char          *fMaxAddr;
   char          *fLimitAddr;

   Int_t          fTotalPos;
   Int_t          fCurrentLine;
   
public:

   char           *fCurrent;

   TXMLInputStream(const char* filename, Int_t ibufsize)
   {
      fInp = new std::ifstream(filename);
      fInpStr = 0;
      fInpStrLen = 0;
   
      Init(ibufsize);
   }

   TXMLInputStream(const char* str)
   {
      fInp = 0;
      fInpStr = str;
      fInpStrLen = str==0 ? 0 : strlen(str);
   
      Init(20000);
   }
   
   void Init(Int_t ibufsize)
   {
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

   Bool_t EndOfFile() { return (fInp!=0) ? fInp->eof() : (fInpStrLen<=0); }

   int DoRead(char* buf, int maxsize)
   {
      if (EndOfFile()) return 0;
      if (fInp!=0) {
         fInp->get(buf,maxsize-1,0);
         maxsize = strlen(buf);
      } else {
         if (maxsize>fInpStrLen) maxsize = fInpStrLen;
         strncpy(buf, fInpStr, maxsize);
         fInpStr+=maxsize;
         fInpStrLen-=maxsize;
         *(buf+maxsize) = 0;
      }
      return maxsize;
   }

   Bool_t ExpandStream()
   {
      if (EndOfFile()) return kFALSE;
      fBufSize*=2;
      int curlength = fMaxAddr - fBuf;
      fBuf = (char*) realloc(fBuf, fBufSize);
      int len = DoRead(fMaxAddr, fBufSize-curlength);
      if (len==0) return kFALSE;
      fMaxAddr+=len;
      fLimitAddr += int(len*0.75);
      return kTRUE;
   }

   Bool_t ShiftStream()
   {
      if (fCurrent<fLimitAddr) return kTRUE; // everything ok, can cntinue
      if (EndOfFile()) return kTRUE;
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
      Bool_t ok = (((symb>='a') && (symb<='z')) ||
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
   // default (normal) constructor of TXMLEngine class

}


//______________________________________________________________________________
TXMLEngine::~TXMLEngine()
{
   // destructor for TXMLEngine object

}

//______________________________________________________________________________
Bool_t TXMLEngine::HasAttr(XMLNodePointer_t xmlnode, const char* name)
{
   // checks if node has attribute of specified name

   if (xmlnode==0) return kFALSE;
   SXmlAttr_t* attr = ((SXmlNode_t*)xmlnode)->fAttr;
   while (attr!=0) {
      if (strcmp(&(attr->fName),name)==0) return kTRUE;
      attr = attr->fNext;
   }
   return kFALSE;
}

//______________________________________________________________________________
const char* TXMLEngine::GetAttr(XMLNodePointer_t xmlnode, const char* name)
{
   // returns value of attribute for xmlnode

   if (xmlnode==0) return 0;
   SXmlAttr_t* attr = ((SXmlNode_t*)xmlnode)->fAttr;
   while (attr!=0) {
      if (strcmp(&(attr->fName),name)==0)
         return &(attr->fName) + strlen(name) + 1;
      attr = attr->fNext;
   }
   return 0;
}

//______________________________________________________________________________
Int_t TXMLEngine::GetIntAttr(XMLNodePointer_t xmlnode, const char* name)
{
   // returns value of attribute as integer

   if (xmlnode==0) return 0;
   Int_t res = 0;
   const char* attr = GetAttr(xmlnode, name);
   if (attr) sscanf(attr, "%d", &res);
   return res;
}

//______________________________________________________________________________
XMLAttrPointer_t TXMLEngine::NewAttr(XMLNodePointer_t xmlnode, XMLNsPointer_t,
                                         const char* name, const char* value)
{
   // creates new attribute for xmlnode,
   // namespaces are not supported for attributes

   if (xmlnode==0) return 0;

   int namelen = strlen(name), valuelen = strlen(value);
   SXmlAttr_t* attr = (SXmlAttr_t*) AllocateAttr(namelen, valuelen, xmlnode);

   char* attrname = &(attr->fName);
   strcpy(attrname, name);
   attrname += (namelen + 1);
   if ((value!=0) && (valuelen>0))
      strcpy(attrname, value);
   else
      *attrname=0;

   return (XMLAttrPointer_t) attr;
}

//______________________________________________________________________________
XMLAttrPointer_t TXMLEngine::NewIntAttr(XMLNodePointer_t xmlnode,
                                      const char* name,
                                      Int_t value)
{
   // create node attribute with integer value

   char sbuf[30];
   sprintf(sbuf,"%d",value);
   return NewAttr(xmlnode, 0, name, sbuf);
}

//______________________________________________________________________________
void TXMLEngine::FreeAttr(XMLNodePointer_t xmlnode, const char* name)
{
   // remove attribute from xmlnode

   if (xmlnode==0) return;
   SXmlAttr_t* attr = ((SXmlNode_t*) xmlnode)->fAttr;
   SXmlAttr_t* prev = 0;
   while (attr!=0) {
      if (strcmp(&(attr->fName),name)==0) {
         if (prev!=0)
            prev->fNext = attr->fNext;
         else
            ((SXmlNode_t*) xmlnode)->fAttr = attr->fNext;
         //fNumNodes--;
         free(attr);
         return;
      }

      prev = attr;
      attr = attr->fNext;
   }
}

//______________________________________________________________________________
XMLNodePointer_t TXMLEngine::NewChild(XMLNodePointer_t parent, XMLNsPointer_t ns,
                                      const char* name, const char* content)
{
   // create new child element for parent node

   SXmlNode_t* node = (SXmlNode_t*) AllocateNode(strlen(name), parent);

   strcpy(&(node->fName), name);
   node->fNs = (SXmlAttr_t*) ns;
   if (content!=0) {
      int contlen = strlen(content);
      if (contlen>0) {
         SXmlNode_t* contnode = (SXmlNode_t*) AllocateNode(contlen+1, node);
         char* cptr = &(contnode->fName);
         *cptr = 0;
         cptr++;
         strcpy(cptr,content);
      }
   }

   return (XMLNodePointer_t) node;
}

//______________________________________________________________________________
XMLNsPointer_t TXMLEngine::NewNS(XMLNodePointer_t xmlnode, const char* reference, const char* name)
{
   // create namespace attribute for xmlnode.
   // namespace attribute will be always the first in list of node attributes

   SXmlNode_t* node = (SXmlNode_t*) xmlnode;
   if (name==0) name = &(node->fName);
   char* nsname = new char[strlen(name)+7];
   strcpy(nsname, "xmlns:");
   strcat(nsname, name);

   SXmlAttr_t* first = node->fAttr;
   node->fAttr = 0;

   SXmlAttr_t* nsattr = (SXmlAttr_t*) NewAttr(xmlnode, 0, nsname, reference);

   node->fAttr = nsattr;
   nsattr->fNext = first;

   node->fNs = nsattr;
   delete[] nsname;
   return (XMLNsPointer_t) nsattr;
}

//______________________________________________________________________________
void TXMLEngine::AddChild(XMLNodePointer_t parent, XMLNodePointer_t child)
{
   // add child element to xmlnode

   if ((parent==0) || (child==0)) return;
   SXmlNode_t* pnode = (SXmlNode_t*) parent;
   SXmlNode_t* cnode = (SXmlNode_t*) child;
   cnode->fParent = pnode;
   if (pnode->fLastChild==0) {
      pnode->fChild = cnode;
      pnode->fLastChild = cnode;
   } else {
      //SXmlNode_t* ch = pnode->fChild;
      //while (ch->fNext!=0) ch=ch->fNext;
      pnode->fLastChild->fNext = cnode;
      pnode->fLastChild = cnode;
   }
}

//______________________________________________________________________________
void TXMLEngine::UnlinkNode(XMLNodePointer_t xmlnode)
{
   // unlink (dettach) xml node from parent

   if (xmlnode==0) return;
   SXmlNode_t* node = (SXmlNode_t*) xmlnode;

   SXmlNode_t* parent = node->fParent;

   if (parent==0) return;

   if (parent->fChild==node) {
      parent->fChild = node->fNext;
      if (parent->fLastChild==node)
         parent->fLastChild = node->fNext;
   } else {
      SXmlNode_t* ch = parent->fChild;
      while (ch->fNext!=node) ch = ch->fNext;
      ch->fNext = node->fNext;
      if (parent->fLastChild == node)
         parent->fLastChild = ch;
   }
}

//______________________________________________________________________________
void TXMLEngine::FreeNode(XMLNodePointer_t xmlnode)
{
   // release all memory, allocated fro this node and
   // destroyes node itself

   if (xmlnode==0) return;
   SXmlNode_t* node = (SXmlNode_t*) xmlnode;

   SXmlNode_t* child = node->fChild;
   while (child!=0) {
      SXmlNode_t* next = child->fNext;
      FreeNode((XMLNodePointer_t)child);
      child = next;
   }

   SXmlAttr_t* attr = node->fAttr;
   while (attr!=0) {
      SXmlAttr_t* next = attr->fNext;
      //fNumNodes--;
      free(attr);
      attr = next;
   }

   //delete[] node->fName;
   // delete[] node->content;
   free(node);

   //fNumNodes--;
}

//______________________________________________________________________________
void TXMLEngine::UnlinkFreeNode(XMLNodePointer_t xmlnode)
{
   // combined operation. Unlink node and free used memory

   UnlinkNode(xmlnode);
   FreeNode(xmlnode);
}

//______________________________________________________________________________
const char* TXMLEngine::GetNodeName(XMLNodePointer_t xmlnode)
{
   // returns name of xmlnode

   return xmlnode==0 ? 0 : & (((SXmlNode_t*) xmlnode)->fName);
}

//______________________________________________________________________________
const char* TXMLEngine::GetNodeContent(XMLNodePointer_t xmlnode)
{
   // get contents (if any) of xml node

   if (xmlnode==0) return 0;
   SXmlNode_t* node = (SXmlNode_t*) xmlnode;
   if ((node->fChild==0) || (node->fChild->fName!=0)) return 0;
   return &(node->fChild->fName) + 1;
}

//______________________________________________________________________________
XMLNodePointer_t TXMLEngine::GetChild(XMLNodePointer_t xmlnode)
{
   // returns first child of xml node

   SXmlNode_t* res = xmlnode==0 ? 0 :((SXmlNode_t*) xmlnode)->fChild;
   // skip content node
   if ((res!=0) && (res->fName==0)) res = res->fNext;
   return (XMLNodePointer_t) res;
}

//______________________________________________________________________________
XMLNodePointer_t TXMLEngine::GetParent(XMLNodePointer_t xmlnode)
{
   // returns parent of xmlnode

   return xmlnode==0 ? 0 : (XMLNodePointer_t) ((SXmlNode_t*) xmlnode)->fParent;
}

//______________________________________________________________________________
XMLNodePointer_t TXMLEngine::GetNext(XMLNodePointer_t xmlnode)
{
   // return next to xmlnode node

   return xmlnode==0 ? 0 : (XMLNodePointer_t) ((SXmlNode_t*) xmlnode)->fNext;
}

//______________________________________________________________________________
void TXMLEngine::ShiftToNext(XMLNodePointer_t &xmlnode)
{
   // shifts specified node to next

   xmlnode = xmlnode==0 ? 0 : (XMLNodePointer_t) ((SXmlNode_t*) xmlnode)->fNext;
}

//______________________________________________________________________________
void TXMLEngine::CleanNode(XMLNodePointer_t xmlnode)
{
   // remove all childs node from xmlnode

   if (xmlnode==0) return;
   SXmlNode_t* node = (SXmlNode_t*) xmlnode;

   SXmlNode_t* child = node->fChild;
   while (child!=0) {
      SXmlNode_t* next = child->fNext;
      FreeNode((XMLNodePointer_t)child);
      child = next;
   }

   node->fChild = 0;
   node->fLastChild = 0;
}

//______________________________________________________________________________
XMLDocPointer_t TXMLEngine::NewDoc(const char* version)
{
   // creates new xml document with provided version

   SXmlDoc_t* doc = new SXmlDoc_t;
   doc->fRootNode = 0;
   doc->fVersion = Makestr(version);
   doc->fDtdName = 0;
   doc->fDtdRoot = 0;
   return (XMLDocPointer_t) doc;
}

//______________________________________________________________________________
void TXMLEngine::AssignDtd(XMLDocPointer_t xmldoc, const char* dtdname, const char* rootname)
{
   // assignes dtd filename to document

   if (xmldoc==0) return;
   SXmlDoc_t* doc = (SXmlDoc_t*) xmldoc;
   delete[] doc->fDtdName;
   doc->fDtdName = Makestr(dtdname);
   delete[] doc->fDtdRoot;
   doc->fDtdRoot = Makestr(rootname);
}

//______________________________________________________________________________
void TXMLEngine::FreeDoc(XMLDocPointer_t xmldoc)
{
   // frees allocated document data and deletes document itself

   if (xmldoc==0) return;
   SXmlDoc_t* doc = (SXmlDoc_t*) xmldoc;
   FreeNode((XMLNodePointer_t) doc->fRootNode);
   delete[] doc->fVersion;
   delete[] doc->fDtdName;
   delete[] doc->fDtdRoot;
   delete doc;
}

//______________________________________________________________________________
void TXMLEngine::SaveDoc(XMLDocPointer_t xmldoc, const char* filename, Int_t layout)
{
   // store document content to file
   // if layout<=0, no any spaces or newlines will be placed between
   // xmlnodes. Xml file will have minimum size, but nonreadable structure
   // if (layout>0) each node will be started from new line,
   // and number of spaces will correspond to structure depth.

   if (xmldoc==0) return;

   SXmlDoc_t* doc = (SXmlDoc_t*) xmldoc;

   TXMLOutputStream out(filename, 100000);
   out.Write("<?xml version=\"");
   if (doc->fVersion!=0)
      out.Write(doc->fVersion);
   else
      out.Write("1.0");
   out.Write("\"?>\n");

   SaveNode((XMLNodePointer_t) doc->fRootNode, &out, layout, 0);
}

//______________________________________________________________________________
void TXMLEngine::DocSetRootElement(XMLDocPointer_t xmldoc, XMLNodePointer_t xmlnode)
{
   // set main (root) node for document

   if (xmldoc==0) return;
   SXmlDoc_t* doc = (SXmlDoc_t*) xmldoc;
   FreeNode((XMLNodePointer_t) doc->fRootNode);
   doc->fRootNode = (SXmlNode_t*) xmlnode;
}

//______________________________________________________________________________
XMLNodePointer_t TXMLEngine::DocGetRootElement(XMLDocPointer_t xmldoc)
{
   // returns root node of document

   return (xmldoc==0) ? 0 : (XMLNodePointer_t) ((SXmlDoc_t*)xmldoc)->fRootNode;
}

//______________________________________________________________________________
XMLDocPointer_t TXMLEngine::ParseFile(const char* filename)
{
   // parses content of file and tries to produce xml structures

   if ((filename==0) || (strlen(filename)==0)) return 0;

   TXMLInputStream inp(filename, 100000);

   if (!inp.SkipSpaces()) return 0;
   if (!inp.CheckFor("<?xml")) return 0;
   if (!inp.SkipSpaces()) return 0;
   if (!inp.SearchFor("?>")) return 0;
   if (!inp.ShiftCurrent(2)) return 0;
   inp.SkipSpaces(kTRUE); // locate start of next string

   Int_t resvalue = 0;

   XMLNodePointer_t mainnode = ReadNode(0, &inp, resvalue);

   if (resvalue<=0) {
      DisplayError(resvalue, inp.CurrentLine());
      FreeNode(mainnode);
      return 0;
   }

   if (mainnode==0) return 0;
   XMLDocPointer_t xmldoc = NewDoc();
   DocSetRootElement(xmldoc, mainnode);
   return xmldoc;
}

//______________________________________________________________________________
void TXMLEngine::SaveSingleNode(XMLNodePointer_t xmlnode, TString* res, Int_t layout)
{
   // convert single xml node (and its child node) to string 
   // if layout<=0, no any spaces or newlines will be placed between
   // xmlnodes. Xml file will have minimum size, but nonreadable structure
   // if (layout>0) each node will be started from new line,
   // and number of spaces will correspond to structure depth.
   
   if ((res==0) || (xmlnode==0)) return; 
   
   TXMLOutputStream out(res, 10000);

   SaveNode(xmlnode, &out, layout, 0);
}

//______________________________________________________________________________
XMLNodePointer_t TXMLEngine::ReadSingleNode(const char* src)
{
   // read snigle xml node from provided string
    
   if (src==0) return 0;
   
   TXMLInputStream inp(src);

   Int_t resvalue;

   XMLNodePointer_t xmlnode = ReadNode(0, &inp, resvalue);

   if (resvalue<=0) {
      DisplayError(resvalue, inp.CurrentLine());
      FreeNode(xmlnode);
      return 0;
   }
   
   return xmlnode; 
}

//______________________________________________________________________________
char* TXMLEngine::Makestr(const char* str)
{
   // creates char* variable with copy of provided string

   if (str==0) return 0;
   int len = strlen(str);
   if (len==0) return 0;
   char* res = new char[len+1];
   strcpy(res, str);
   return res;
}

//______________________________________________________________________________
char* TXMLEngine::Makenstr(const char* str, int len)
{
   // creates char* variable with copy of len symbols from provided string

   if ((str==0) || (len==0)) return 0;
   char* res = new char[len+1];
   strncpy(res, str, len);
   *(res+len) = 0;
   return res;
}

//______________________________________________________________________________
XMLNodePointer_t TXMLEngine::AllocateNode(int namelen, XMLNodePointer_t parent)
{
   // Allocates new xml node with specified namelength

   //fNumNodes++;

   SXmlNode_t* node = (SXmlNode_t*) malloc(sizeof(SXmlNode_t) + namelen);

   node->fParent = 0;
   node->fNs = 0;
   node->fAttr = 0;
   node->fChild = 0;
   node->fLastChild = 0;
   node->fNext = 0;

   if (parent!=0)
      AddChild(parent, (XMLNodePointer_t) node);

   return (XMLNodePointer_t) node;
}

//______________________________________________________________________________
XMLAttrPointer_t TXMLEngine::AllocateAttr(int namelen, int valuelen, XMLNodePointer_t xmlnode)
{
   // Allocate new attribute with specified name length and value length

   //fNumNodes++;

   SXmlAttr_t* attr = (SXmlAttr_t*) malloc(sizeof(SXmlAttr_t) + namelen + valuelen + 1);

   SXmlNode_t* node = (SXmlNode_t*) xmlnode;

   attr->fNext = 0;

   if (node->fAttr==0)
      node->fAttr = attr;
   else {
      SXmlAttr_t* d = node->fAttr;
      while (d->fNext!=0) d = d->fNext;
      d->fNext = attr;
   }

   return (XMLAttrPointer_t) attr;
}

//______________________________________________________________________________
XMLNsPointer_t TXMLEngine::FindNs(XMLNodePointer_t xmlnode, const char* name)
{
   // define if namespace of that name exists for xmlnode

   SXmlNode_t* node = (SXmlNode_t*) xmlnode;
   while (node!=0) {
      if (node->fNs!=0) {
         const char* nsname = &(node->fNs->fName) + 6;
         if (strcmp(nsname, name)==0) return node->fNs;
      }
      node = node->fParent;
   }
   return 0;
}

//______________________________________________________________________________
void TXMLEngine::TruncateNsExtension(XMLNodePointer_t xmlnode)
{
   // removes namespace extension of nodename

   SXmlNode_t* node = (SXmlNode_t*) xmlnode;
   if (node==0) return;
   char* colon = strchr(&(node->fName),':');
   if (colon==0) return;

   char* copyname = &(node->fName);

   while (*colon!=0)
     *(copyname++) = *(++colon);
}

//______________________________________________________________________________
void TXMLEngine::UnpackSpecialCharacters(char* target, const char* source, int srclen)
{
   // unpack special symbols, used in xml syntax to code characters
   // these symbols: '<' - &lt, '>' - &gt, '&' - &amp, '"' - &quot

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
         if ((*(source+1)=='q') && (*(source+2)=='u') && (*(source+3)=='o') && (*(source+4)=='t') && (*(source+5)==';')) {
            *target++ = '\"'; source+=6; srclen-=6;
         } else {
            *target++ = *source++; srclen--;
         }
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
   // output value to output stream
   // if symbols '<' '&' '>' '"' appears in the string, they
   // will be encoded to appropriate xml symbols: &lt, &amp, &gt, &quot

   if (value==0) return;

   char* last = value;
   char* find = 0;
   while ((find=strpbrk(last,"<&>\"")) !=0 ) {
      char symb = *find;
      *find = 0;
      out->Write(last);
      *find = symb;
      last = find+1;
      if (symb=='<') out->Write("&lt;"); else
      if (symb=='>') out->Write("&gt;"); else
      if (symb=='&') out->Write("&amp;"); else
                     out->Write("&quot;");
   }
   if (*last!=0)
      out->Write(last);
}

//______________________________________________________________________________
void TXMLEngine::SaveNode(XMLNodePointer_t xmlnode, TXMLOutputStream* out, Int_t layout, Int_t level)
{
   // stream data of xmlnode to output

   if (xmlnode==0) return;
   SXmlNode_t* node = (SXmlNode_t*) xmlnode;

   Bool_t issingleline = (node->fName!=0) && (node->fChild==0);

   if (layout>0) out->Put(' ', level);

   if (node->fName!=0) {

      out->Put('<');
      // we suppose that ns is always first attribute
      if ((node->fNs!=0) && (node->fNs!=node->fAttr)) {
         out->Write(&(node->fNs->fName)+6);
         out->Put(':');
      }
      out->Write(&(node->fName));

      SXmlAttr_t* attr = node->fAttr;
      while (attr!=0) {
         out->Put(' ');
         char* attrname = &(attr->fName);
         out->Write(attrname);
         out->Write("=\"");
         attrname += strlen(attrname) + 1;
         OutputValue(attrname, out);
         out->Put('\"');
         attr = attr->fNext;
      }
      if (issingleline) out->Write("/>"); else out->Put('>');
      if (layout>0) out->Put('\n');
   } else {
      // this is output for content
      out->Write(&(node->fName)+1);
      return;
   }

   if (issingleline) return;

   SXmlNode_t* child = node->fChild;
   while (child!=0) {
      SaveNode((XMLNodePointer_t) child, out, layout, level+2);
      child = child->fNext;
   }

   if (node->fName!=0) {
      if (layout>0)
         out->Put(' ',level);
      out->Write("</");
      // we suppose that ns is always first attribute
      if ((node->fNs!=0) && (node->fNs!=node->fAttr)) {
         out->Write(&(node->fNs->fName)+6);
         out->Put(':');
      }
      out->Write(&(node->fName));
      out->Put('>');
      if (layout>0) out->Put('\n');
   }
}

//______________________________________________________________________________
XMLNodePointer_t TXMLEngine::ReadNode(XMLNodePointer_t xmlparent, TXMLInputStream* inp, Int_t& resvalue)
{
   // Tries to construct xml node from input stream. Node should be
   // child of xmlparent node or it can be closing tag of xmlparent.
   // resvalue <= 0 if error
   // resvalue == 1 if this is endnode of parent
   // resvalue == 2 if this is child

   resvalue = 0;

   if (inp==0) return 0;
   if (!inp->SkipSpaces()) { resvalue = -1; return 0; }
   SXmlNode_t* parent = (SXmlNode_t*) xmlparent;

   SXmlNode_t* node = 0;

   if (*inp->fCurrent!='<') {
      // here should be reading of element content
      // only one entry for content is supported, only before any other childs
      if ((parent==0) || (parent->fChild!=0)) { resvalue = -2; return 0; }
      int contlen = inp->LocateContent();
      if (contlen<0) return 0;

      SXmlNode_t* contnode = (SXmlNode_t*) AllocateNode(contlen+1, xmlparent);
      char* contptr = &(contnode->fName);
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

      if (strncmp(&(parent->fName), inp->fCurrent, len)!=0) {
         resvalue = -5;
         return 0;
      }

      if (!inp->ShiftCurrent(len)) return 0;

      if (!inp->SkipSpaces()) return 0;
      if (*inp->fCurrent!='>') return 0;
      if (!inp->ShiftCurrent()) return 0;

      if (parent->fNs!=0)
         TruncateNsExtension((XMLNodePointer_t)parent);

      inp->SkipSpaces(kTRUE); // locate start of next string
      resvalue = 1;
      return 0;
   }


   if (!inp->SkipSpaces()) return 0;
   Int_t len = inp->LocateIdentifier();
   if (len<=0) return 0;
   node = (SXmlNode_t*) AllocateNode(len, xmlparent);
   char* nameptr = &(node->fName);

   strncpy(nameptr, inp->fCurrent, len);
   nameptr+=len;
   *nameptr = 0;

   char* colon = strchr(&(node->fName),':');
   if ((colon!=0) && (parent!=0)) {
      *colon = 0;
      node->fNs = (SXmlAttr_t*) FindNs(xmlparent, &(node->fName));
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

            if (node->fNs!=0)
               TruncateNsExtension((XMLNodePointer_t) node);

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

         SXmlAttr_t* attr = (SXmlAttr_t*) AllocateAttr(attrlen, valuelen-3, (XMLNodePointer_t) node);

         char* attrname = &(attr->fName);
         strncpy(attrname, inp->fCurrent, attrlen);
         attrname+=attrlen;
         *attrname = 0;
         attrname++;
         UnpackSpecialCharacters(attrname, valuestart+2, valuelen-3);

         if (!inp->ShiftCurrent(attrlen+valuelen)) return 0;

         attrname = &(attr->fName);

         if ((strlen(attrname)>6) && (strstr(attrname,"xmlns:")==attrname)) {
            if (strcmp(&(node->fName), attrname + 6)!=0) {
               resvalue = -8;
               return 0;
            }
            if (node->fNs!=0) {
               resvalue = -9;
               return 0;
            }
            node->fNs = attr;
         }
      }
   } while (true);

   return 0;
}

//______________________________________________________________________________
void TXMLEngine::DisplayError(Int_t error, Int_t linenumber)
{
   // Dsiplays error, occured during parsing of xml file
   switch(error) {
      case -9: Error("ParseFile", "Multiple name space definitions not allowed, line %d", linenumber); break;
      case -8: Error("ParseFile", "Invalid namespace specification, line %d", linenumber); break;
      case -7: Error("ParseFile", "Invalid attribute value, line %d", linenumber); break;
      case -6: Error("ParseFile", "Invalid identifier for node attribute, line %d", linenumber); break;
      case -5: Error("ParseFile", "Missmatch between open and close nodes, line %d", linenumber); break;
      case -4: Error("ParseFile", "Unexpected close node, line %d", linenumber); break;
      case -3: Error("ParseFile", "Valid identifier for close node is missing, line %d", linenumber); break;
      case -2: Error("ParseFile", "No multiple content entries allowed, line %d", linenumber); break;
      case -1: Error("ParseFile", "Unexpected end of xml file"); break;
      default: Error("ParseFile", "XML syntax error at line %d", linenumber); break;
   }
   
}

