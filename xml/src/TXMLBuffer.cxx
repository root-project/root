// @(#)root/xml:$Name:  $:$Id: TXMLBuffer.cxx,v 1.6 2004/05/14 14:30:46 brun Exp $
// Author: Sergey Linev, Rene Brun  10.05.2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//________________________________________________________________________
//
// Class for serializing/deserializing object to/from xml.
// It redefines most of TBuffer class function to convert simple types,
// array of simple types and objects to/from xml.
// Instead of writing a binary data it creates a set of xml structures as
// nodes and attributes
// TXMLBuffer class uses streaming mechanism, provided by ROOT system,
// therefore most of ROOT and user classes can be stored to xml. There are 
// limitations for complex objects like TTree, which can not be yet converted to xml.
//________________________________________________________________________


#include "TXMLBuffer.h"
#include "TXMLFile.h"

#include "TObjArray.h"
#include "TROOT.h"
#include "TClass.h"
#include "TClassTable.h"
#include "TExMap.h"
#include "TMethodCall.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TProcessID.h"
#include "TFile.h"
#include "TMemberStreamer.h"
#include "TStreamer.h"

#include <Riostream.h>

extern "C" void R__zip(int cxlevel, int *srcsize, char *src, int *tgtsize, char *tgt, int *irep);

extern "C" void R__unzip(int *srcsize, unsigned char *src, int *tgtsize, unsigned char *tgt, int *irep);

ClassImp(TXMLBuffer);

//______________________________________________________________________________
TXMLBuffer::TXMLBuffer() : 
    TBuffer(),
    TXMLSetup(),
    fXML(0)
{
// Default constructor
}

//______________________________________________________________________________
TXMLBuffer::TXMLBuffer(TBuffer::EMode mode) :
    TBuffer(mode),
    TXMLSetup(),
    fXML(0),
    fStack(),
    fVersionBuf(-111),
    fObjMap(0),
    fIdArray(0),
    fErrorFlag(0),
    fCanUseCompact(kFALSE),
    fExpectedChain(kFALSE),
    fExpectedBaseClass(0),
    fCompressLevel(0)
{
// Creates buffer object to serailize/deserialize data to/from xml. 
// Mode should be either TBuffer::kRead or TBuffer::kWrite.

  SetParent(0);
}

//______________________________________________________________________________
TXMLBuffer::TXMLBuffer(TBuffer::EMode mode, TXMLFile* file) :
    TBuffer(mode),
    TXMLSetup(*file),
    fStack(),
    fVersionBuf(-111),
    fObjMap(0),
    fIdArray(0),
    fErrorFlag(0),
    fCanUseCompact(kFALSE),
    fExpectedChain(kFALSE),
    fExpectedBaseClass(0),
    fCompressLevel(0)
{
// Creates buffer object to serailize/deserialize data to/from xml. 
// This constructor should be used, if data from buffer supposed to be stored in file.
// Mode should be either TBuffer::kRead or TBuffer::kWrite.

  SetParent(file);
  if (XmlFile()) {
    SetXML(XmlFile()->XML());
    SetCompressionLevel(XmlFile()->GetCompressionLevel());
  }
}


//______________________________________________________________________________
TXMLBuffer::~TXMLBuffer() 
{
// destroy xml buffer     

   if (fObjMap) delete fObjMap;
   if (fIdArray) delete fIdArray;
   fStack.Delete();
}

//______________________________________________________________________________
TXMLFile* TXMLBuffer::XmlFile() 
{
// returns pointer to TXMLFile object
// access to file is necessary to produce unique identifier for object references

   return dynamic_cast<TXMLFile*>(GetParent());
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::XmlWrite(const TObject* obj) 
{
// Convert object, derived from TObject class to xml structures
// Return pointer on top xml element    
   
   if (obj==0) return XmlWrite(0,0);
          else return XmlWrite(obj, obj->IsA());
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::XmlWrite(const void* obj, const TClass* cl) 
{
// Convert object of any class to xml structures
// Return pointer on top xml element    
    
   fErrorFlag = 0;
   
   if (fXML==0) return 0;

   xmlNodePointer res = XmlWriteObject(obj, cl);
   
   return res;
}

//______________________________________________________________________________
TObject* TXMLBuffer::XmlRead(xmlNodePointer node) 
{
// Recreate object from xml structure. 
// Return pointer to read object.
// If object class is not inherited from TObject, 
// object is deleted and function return 0
    
   TClass* cl = 0; 
   void* obj = XmlReadAny(node, &cl);
   
   if ((cl!=0) && !cl->InheritsFrom(TObject::Class())) {
      cl->Destructor(obj);
      obj = 0; 
   }

   return (TObject*) obj;
}

//______________________________________________________________________________
void* TXMLBuffer::XmlReadAny(xmlNodePointer node, TClass** cl) 
{
// Recreate object from xml structure. 
// Return pointer to read object.
// if (cl!=0) returns pointer to class of object
    
   if (node==0) return 0;
   if (cl) *cl = 0;

   fErrorFlag = 0;

   if (fXML==0) return 0;
   
   PushStack(node, kTRUE);

   void* obj = XmlReadObject(0, cl);
   
   PopStack();

   return obj;
}

//______________________________________________________________________________
void TXMLBuffer::WriteObject(const TObject *obj) 
{
// Convert object into xml structures.
// !!! Should be used only by TXMLBuffer itself. 
// Use XmlWrite() functions to convert your object to xml
// Redefined here to avoid gcc 3.x warning
    
   TBuffer::WriteObject(obj);
}

class TXMLStackObj : public TObject {
   public:
      TXMLStackObj(xmlNodePointer node) :
         TObject(), 
         fNode(node), 
         fInfo(0), 
         fElem(0), 
         fElemNumber(0),
         fCompressedClassNode(kFALSE),
         fClassNs(0) {}

      xmlNodePointer    fNode;
      TStreamerInfo*    fInfo;
      TStreamerElement* fElem;
      Int_t             fElemNumber;
      Bool_t            fCompressedClassNode;
      xmlNsPointer      fClassNs;
};

//______________________________________________________________________________
TXMLStackObj* TXMLBuffer::PushStack(xmlNodePointer current, Bool_t simple) 
{
// add new level to xml stack
    
  if (IsReading() && !simple) {
    current = fXML->GetChild(current);
    fXML->SkipEmpty(current);
  }

  TXMLStackObj* stack = new TXMLStackObj(current);
  fStack.Add(stack);
  return stack;
}

//______________________________________________________________________________
TXMLStackObj* TXMLBuffer::PopStack() 
{
// remove one level from xml stack    
    
  TObject* last = fStack.Last();
  if (last!=0) {
    fStack.Remove(last);
    delete last;
    fStack.Compress();
  }
  return dynamic_cast<TXMLStackObj*> (fStack.Last());
}

//______________________________________________________________________________
TXMLStackObj* TXMLBuffer::Stack(Int_t depth) 
{
// return xml stack object of specified depth
    
  TXMLStackObj* stack = 0;
  if (depth<=fStack.GetLast())
    stack = dynamic_cast<TXMLStackObj*> (fStack.At(fStack.GetLast()-depth));
  return stack;
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::StackNode() 
{
// return pointer on current xml node    
    
  TXMLStackObj* stack = dynamic_cast<TXMLStackObj*> (fStack.Last());
  return (stack==0) ? 0 : stack->fNode;
}

//______________________________________________________________________________
void TXMLBuffer::ShiftStack(const char* errinfo) 
{
// shift stack node to next
    
   TXMLStackObj* stack = dynamic_cast<TXMLStackObj*> (fStack.Last());
   if (stack) {
     if (gDebug>4)
       cout << "       Shift " << errinfo << " from " << fXML->GetNodeName(stack->fNode);
     fXML->ShiftToNext(stack->fNode);
     if (gDebug>4)
       cout << " to " << fXML->GetNodeName(stack->fNode) << endl;
   }
}

//______________________________________________________________________________
void TXMLBuffer::XmlWriteBlock(xmlNodePointer node) 
{
// write binary data block from buffer to xml
// this data can be produced only by direct call of TBuffer::WriteBuf() functions
    
   if ((node==0) || (Length()==0)) return;

   const char* src = Buffer();
   int srcSize = Length();

   char* fZipBuffer = 0;
   
   Int_t complevel = fCompressLevel;

   if ((Length() > 512) && (complevel>0)) {
      int ZipBufferSize = Length();
      fZipBuffer = new char[ZipBufferSize];
      int DataSize = Length();
      int CompressedSize = 0;
      if (complevel>9) complevel = 9;
      R__zip(complevel, &DataSize, Buffer(), &ZipBufferSize, fZipBuffer, &CompressedSize);
      src = fZipBuffer;
      srcSize = CompressedSize;
   }

   TString res; 
   char sbuf[500];
   int block = 0;
   char* tgt = sbuf;
   int srcCnt = 0;

   while (srcCnt++<srcSize) {
      tgt+=sprintf(tgt, " %02x", (unsigned char) *src);
      src++;
      if (block++==100) {
         res += sbuf;
         block = 0;
         tgt = sbuf;
      }
   }

   if (block>0) res += sbuf;

   xmlNodePointer blocknode = fXML->NewChild(node, 0, xmlNames_XmlBlock, res);
   fXML->NewIntAttr(blocknode, xmlNames_Size, Length());

   if (fZipBuffer) {
      fXML->NewIntAttr(blocknode, xmlNames_Zip, srcSize);
      delete[] fZipBuffer;
   }
}

//______________________________________________________________________________
void TXMLBuffer::XmlReadBlock(xmlNodePointer blocknode) 
{
// read binary block of data from xml

   if (blocknode==0) return;

   Int_t blockSize = fXML->GetIntAttr(blocknode, xmlNames_Size);
   bool blockCompressed = fXML->HasAttr(blocknode, xmlNames_Zip);
   char* fUnzipBuffer = 0;

   if (gDebug>2) {
      cout << " TXMLBuffer::XmlReadBlock " << endl;
      cout << "    block size = " << blockSize << endl;
      cout << "    Length = " << Length() << endl;
      cout << "    compressed = " << blockCompressed << endl;
   }

   if (blockSize>BufferSize()) Expand(blockSize);

   char* tgt = Buffer();
   Int_t readSize = blockSize;

   TString content = fXML->GetNodeContent(blocknode);

   if (blockCompressed) {
      Int_t zipSize = fXML->GetIntAttr(blocknode, xmlNames_Zip);
      fUnzipBuffer = new char[zipSize];

      tgt = fUnzipBuffer;
      readSize = zipSize;
   }

   if (gDebug>3)
     cout << "Content = " << content << endl;

   char* ptr = (char*) content.Data();

   for (int i=0;i<readSize;i++) {
     while ((*ptr<48) || ((*ptr>57) && (*ptr<97)) || (*ptr>102)) ptr++;

     char* z = ptr;
     int b_hi = (*ptr>57) ? *ptr-87 : *ptr-48;
     ptr++;
     int b_lo = (*ptr>57) ? *ptr-87 : *ptr-48;
     ptr++;

     *tgt=b_hi*16+b_lo;
     tgt++;

     if (gDebug>3)
       cout << "Buf[" << i << "] = " << b_hi*16.+b_lo << "  " << *z << *(z+1) << " "<< b_hi << "  " << b_lo << endl;
   }

   if (fUnzipBuffer) {
      int unzipRes = 0;
      R__unzip(&readSize, (unsigned char*) fUnzipBuffer, &blockSize,
                          (unsigned char*) Buffer(), &unzipRes);
      if (gDebug>2)
         if (unzipRes!=blockSize) cout << "decompression error " << unzipRes << endl;
                             else cout << "unzip ok" << endl;
      delete[] fUnzipBuffer;
   }
}

//______________________________________________________________________________
Bool_t TXMLBuffer::ProcessPointer(const void* ptr, xmlNodePointer node) 
{
// Add "ptr" attribute to node, if ptr is null or 
// if ptr is pointer on object, which is already saved in buffer
// Automatically add "ref" attribute to node, where referenced object is stored

  if (node==0) return kFALSE;

  TString refvalue;

  if (ptr==0)
     refvalue = xmlNames_Null;   //null
  else {
     if (fObjMap==0) return kFALSE;

     ULong_t hash = TMath::Hash(&ptr, sizeof(void*));

     xmlNodePointer refnode = (xmlNodePointer) fObjMap->GetValue(hash, (Long_t) ptr);
     if (refnode==0) return kFALSE;

     if (fXML->HasAttr(refnode, xmlNames_Ref))
        refvalue = fXML->GetAttr(refnode, xmlNames_Ref);
     else {
        refvalue = xmlNames_IdBase;
        if (XmlFile()) refvalue += XmlFile()->GetNextRefCounter();
                  else refvalue += GetNextRefCounter();
        fXML->NewAttr(refnode, 0, xmlNames_Ref, refvalue.Data());
     }
  }
  if (refvalue.Length()>0) {
     fXML->NewAttr(node, 0, xmlNames_Ptr, refvalue.Data());
     return kTRUE;
  }

  return kFALSE;
}

//______________________________________________________________________________
void TXMLBuffer::RegisterPointer(const void* ptr, xmlNodePointer node) 
{
// Register pair of object pointer and node, where this object is saved, 
// in object map
    
   if ((node==0) || (ptr==0)) return;

   ULong_t hash = TMath::Hash(&ptr, sizeof(void*));

   if (fObjMap==0) fObjMap = new TExMap();

   if (fObjMap->GetValue(hash, (Long_t) ptr)==0)
     fObjMap->Add(hash, (Long_t) ptr, (Long_t) node);
}

//______________________________________________________________________________
Bool_t TXMLBuffer::ExtractPointer(xmlNodePointer node, void* &ptr, TClass* &cl) 
{
// Searches for "ptr" attribute and returns pointer to object and class,
// if "ptr" attribute reference to read object 

   cl = 0;

   if (!fXML->HasAttr(node,xmlNames_Ptr)) return kFALSE;

   const char* ptrid = fXML->GetAttr(node, xmlNames_Ptr);

   if (ptrid==0) return kFALSE;

   // null
   if (strcmp(ptrid, xmlNames_Null)==0) {
      ptr = 0;
      return kTRUE;
   }

   if ((fIdArray==0) || (fObjMap==0)) return kFALSE;

   TNamed* obj = (TNamed*) fIdArray->FindObject(ptrid);
   if (obj) {
      ptr = (void*) fObjMap->GetValue((Long_t) fIdArray->IndexOf(obj));
      cl = gROOT->GetClass(obj->GetTitle());
      return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
void TXMLBuffer::ExtractReference(xmlNodePointer node, const void* ptr, const TClass* cl) 
{
// Analyse, if node has "ref" attribute and register it to object map
    
   if ((node==0) || (ptr==0)) return;

   const char* refid = fXML->GetAttr(node, xmlNames_Ref);

   if (refid==0) return;

   if (fIdArray==0) {
      fIdArray = new TObjArray; 
      fIdArray->SetOwner(kTRUE);
   }
   TNamed* nid = new TNamed(refid, cl->GetName());
   fIdArray->Add(nid);

   if (fObjMap==0) fObjMap = new TExMap();

   fObjMap->Add((Long_t) fIdArray->IndexOf(nid), (Long_t) ptr);

   if (gDebug>2)
     cout << "----- Find reference " << refid << " for object " << ptr << " ----- "<< endl;
}

//______________________________________________________________________________
Bool_t TXMLBuffer::VerifyNode(xmlNodePointer node, const char* name, const char* errinfo) 
{
// check, if node has specified name

   if ((name==0) || (node==0)) return kFALSE;

   if (strcmp(fXML->GetNodeName(node), name)!=0) {
      if (errinfo) {
         cout << "   Error reading XML file (" << errinfo << "). Get: " << fXML->GetNodeName(node) << "   expects " << name << endl;
        fErrorFlag = 1;
      }
      return kFALSE;
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TXMLBuffer::VerifyStackNode(const char* name, const char* errinfo) 
{
// check, if stack node has specified name
    
   return VerifyNode(StackNode(), name, errinfo);
}


//______________________________________________________________________________
Bool_t TXMLBuffer::VerifyAttr(xmlNodePointer node, const char* name, const char* value, const char* errinfo) 
{
// checks, that attribute of specified name exists and has specified value    
    
   if ((node==0) || (name==0) || (value==0)) return kFALSE;
   const char* cont = fXML->GetAttr(node, name);
   if (((cont==0) || (strcmp(cont, value)!=0))) {
       if  (errinfo) {
         Error("VerifyAttr","%s : attr %s = %s, expected: %s", errinfo, name, cont, value); 
         fErrorFlag = 1;
       }
       return kFALSE;
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TXMLBuffer::VerifyStackAttr(const char* name, const char* value, const char* errinfo) 
{
// checks stack attribute    
    
   return VerifyAttr(StackNode(), name, value, errinfo);
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::CreateItemNode(const char* name) 
{
// create item node of specified name
    
   xmlNodePointer node = 0; 
   if (GetXmlLayout()==kGeneralized) {
      node = fXML->NewChild(StackNode(), 0, xmlNames_Item, 0);
      fXML->NewAttr(node, 0, xmlNames_Name, name);  
   } else
       node = fXML->NewChild(StackNode(), 0, name, 0);
   return node;    
}

//______________________________________________________________________________
Bool_t TXMLBuffer::VerifyItemNode(const char* name, const char* errinfo) 
{
// checks, if stack node is item and has specified name
    
   Bool_t res = kTRUE; 
   if (GetXmlLayout()==kGeneralized) 
      res = VerifyStackNode(xmlNames_Item, errinfo) &&
            VerifyStackAttr(xmlNames_Name, name, errinfo);
   else
      res = VerifyStackNode(name, errinfo);
   return res;
}   

//______________________________________________________________________________
void TXMLBuffer::CreateElemNode(const TStreamerElement* elem, Int_t number) 
{
// create xml node correspondent to TStreamerElement object

    xmlNodePointer elemnode = 0;

    if (GetXmlLayout()==kGeneralized) {
      elemnode = fXML->NewChild(StackNode(), 0, xmlNames_Member, 0);
      fXML->NewAttr(elemnode, 0, xmlNames_Name, XmlGetElementName(elem));
    } else {
       // take namesapce for element only if it is not a base class or class name  
       xmlNsPointer ns = Stack()->fClassNs; 
       if ((elem->GetType()==TStreamerInfo::kBase) 
           || ((elem->GetType()==TStreamerInfo::kTNamed) && !strcmp(elem->GetName(), TNamed::Class()->GetName()))
           || ((elem->GetType()==TStreamerInfo::kTObject) && !strcmp(elem->GetName(), TObject::Class()->GetName()))
           || ((elem->GetType()==TStreamerInfo::kTString) && !strcmp(elem->GetName(), TString::Class()->GetName())))
         ns = 0;  
       
       elemnode = fXML->NewChild(StackNode(), ns, XmlGetElementName(elem), 0);
    }

    TXMLStackObj* curr = PushStack(elemnode);
    curr->fElem = (TStreamerElement*)elem;
    curr->fElemNumber = number;
}

//______________________________________________________________________________
Bool_t TXMLBuffer::VerifyElemNode(const TStreamerElement* elem, Int_t number) 
{
// Checks, if stack node correspond to TStreamerElement object
    
    if (GetXmlLayout()==kGeneralized) {
       if (!VerifyStackNode(xmlNames_Member)) return kFALSE;
       if (!VerifyStackAttr(xmlNames_Name, XmlGetElementName(elem))) return kFALSE;
    } else {
       if (!VerifyStackNode(XmlGetElementName(elem))) return kFALSE;
    }
    
    PerformPreProcessing(elem, StackNode());
    
    TXMLStackObj* curr = PushStack(StackNode()); // set pointer to first data inside element
    curr->fElem = (TStreamerElement*)elem;
    curr->fElemNumber = number;
    return kTRUE;
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::XmlWriteObject(const void* obj, const TClass* cl) 
{
// Write object to buffer
// If object was written before, only pointer will be stored
// Return pointer to top xml node, representing object
    
   xmlNodePointer objnode = fXML->NewChild(StackNode(), 0, xmlNames_Object, 0);

   if (!cl) obj = 0;
   if (ProcessPointer(obj, objnode)) return objnode;

   TString clname = XmlConvertClassName(cl ? cl->GetName() : "");

   fXML->NewAttr(objnode, 0, xmlNames_ObjClass, clname);

   RegisterPointer(obj, objnode);

   PushStack(objnode);

   ((TClass*)cl)->Streamer((void*)obj, *this);

   PopStack();

   if (gDebug>1)
      cout << "Done write of " << cl->GetName() << endl;

   return objnode;
}

//______________________________________________________________________________
void* TXMLBuffer::XmlReadObject(void* obj, TClass** cl) 
{
// Read object from the buffer    

   if (cl) *cl = 0;
   
   xmlNodePointer objnode = StackNode();

   if (fErrorFlag>0) return obj;

   if (objnode==0) return obj;

   if (!VerifyNode(objnode, xmlNames_Object, "XmlReadObjectNew")) return obj;

   TClass* objClass = 0;

   if (ExtractPointer(objnode, obj, objClass)) {
      ShiftStack("readobjptr");
      if (cl) *cl = objClass;
      return obj;
   }

   TString clname = fXML->GetAttr(objnode, xmlNames_ObjClass);
   objClass = XmlDefineClass(clname);

   if (objClass==0) {
      Error("XmlReadObject", "Cannot find class %s", clname.Data());
      ShiftStack("readobjerr");
      return obj;
   }

   if (gDebug>1)
     cout << "Reading object of class " << clname << endl;

   if (obj==0) obj = objClass->New();

   ExtractReference(objnode, obj, objClass);

   PushStack(objnode);

   objClass->Streamer((void*)obj, *this);

   PopStack();

   ShiftStack("readobj");

   if (gDebug>1)
     cout << "Read object of class " << clname << " done" << endl << endl;
     
   if (cl) *cl = objClass;

   return obj;
}

//______________________________________________________________________________
void  TXMLBuffer::IncrementLevel(TStreamerInfo* info) 
{
// Function is called from TStreamerInfo WriteBuffer and Readbuffer functions
// and indent new level in xml structure.
// This call indicates, that TStreamerInfo functions starts streaming
// object data of correspondent class
    
   if (info==0) return;

   fCanUseCompact = kFALSE;
   fExpectedChain = kFALSE;

   TString clname = XmlConvertClassName(info->GetClass()->GetName());

   if (gDebug>2)
     cout << " IncrementLevel " << clname << endl;
     
   Bool_t compressClassNode = fExpectedBaseClass==info->GetClass();
//   cout << "Compressed class = " << compressClassNode << endl;
   fExpectedBaseClass= 0;
   
   TXMLStackObj* stack = Stack();

   if (IsWriting()) {
       
      xmlNodePointer classnode = 0;
      if (compressClassNode) {
        classnode = StackNode();
      } else {
        if (GetXmlLayout()==kGeneralized) {
           classnode = fXML->NewChild(StackNode(), 0, xmlNames_Class, 0);
           fXML->NewAttr(classnode, 0, "name", clname);
        } else
          classnode = fXML->NewChild(StackNode(), 0, clname, 0);
        stack = PushStack(classnode);
     }

     if (fVersionBuf>=0) {
        fXML->NewIntAttr(classnode, xmlNames_ClassVersion, fVersionBuf);
        fVersionBuf = -111;
      }
      
     if (IsUseNamespaces() && (GetXmlLayout()!=kGeneralized))
       stack->fClassNs = fXML->NewNS(classnode, XmlClassNameSpaceRef(info->GetClass()), clname);

   } else {
      if (!compressClassNode) {
        if (GetXmlLayout()==kGeneralized) {
          if (!VerifyStackNode(xmlNames_Class, "StartInfo")) return;
          if (!VerifyStackAttr("name", clname, "StartInfo")) return;
        } else
          if (!VerifyStackNode(clname, "StartInfo")) return;
        stack = PushStack(StackNode());
      }
   }
   
   stack->fCompressedClassNode = compressClassNode;
   if (gDebug>3)
     cout << "### Inc.lvl Set stack = " << stack << "  sz =" << fStack.GetLast()+1 << endl;
   
   stack->fInfo = info;
}

//______________________________________________________________________________
void  TXMLBuffer::DecrementLevel(TStreamerInfo* info) 
{
// Function is called from TStreamerInfo WriteBuffer and Readbuffer functions
// and decrease level in xml structure.
    
   CheckVersionBuf();

   if (info==0) return;

   fCanUseCompact = kFALSE;
   fExpectedChain = kFALSE;

   if (gDebug>2)
      cout << " DecrementLevel " << info->GetClass()->GetName() << endl;

   TXMLStackObj* stack = Stack();

   if (stack->fInfo==0) {
     PerformPostProcessing();
     stack = PopStack();  // remove stack of last element
   }

   if (gDebug>3)
     cout << "### Dec.lvl Set stack = " << stack << "  sz =" << fStack.GetLast()+1 << endl;

   
   if (stack->fCompressedClassNode) {
     stack->fInfo = 0;
     stack->fCompressedClassNode = kFALSE;
   } else {
     PopStack();                       // back from data of stack info
     if (IsReading()) ShiftStack("declevel"); // shift to next element after streamer info
   }
}

//______________________________________________________________________________
void TXMLBuffer::SetStreamerElementNumber(Int_t number) 
{

// Function is called from TStreamerInfo WriteBuffer and Readbuffer functions
// and add/verify next element of xml structure
// This calls allows separate data, correspondent to one class member, from another
    
   CheckVersionBuf();

   fExpectedChain = kFALSE;
   fCanUseCompact = kFALSE;
   fExpectedBaseClass = 0;

   TXMLStackObj* stack = Stack();
   if (stack==0) {
       Error("SetStreamerElementNumber", "stack is empty");
       return;
    }
   
   if (stack->fInfo==0) {  // this is not a first element
     PerformPostProcessing();
     PopStack();           // go level back
     if (IsReading()) ShiftStack("startelem");   // shift to next element, only for reading
     stack = dynamic_cast<TXMLStackObj*> (fStack.Last());
   }
   
   if (stack==0) {
      Error("SetStreamerElementNumber", "Lost of stack");
      return;   
   }
   
   TStreamerInfo* info = stack->fInfo;
   if (info==0) {
      Error("SetStreamerElementNumber", "Problem in Inc/Dec level");
      return;   
   }
   
   TStreamerElement* elem = info->GetStreamerElementReal(number, 0);
      
   if (elem==0) {
      Error("SetStreamerElementNumber", "streamer info returns elem = 0");
      return;
   }
   
   if (gDebug>4)
     cout << "         Next element " << elem->GetName() << endl;

   Int_t comp_type = info->GetTypes()[number];

   Bool_t isBasicType = (elem->GetType()>0) && (elem->GetType()<20);

   fCanUseCompact = isBasicType && (elem->GetType()==comp_type);

   fExpectedChain = isBasicType && (comp_type - elem->GetType() == TStreamerInfo::kOffsetL);
   
   if ((elem->GetType()==TStreamerInfo::kBase) ||
       ((elem->GetType()==TStreamerInfo::kTNamed) && !strcmp(elem->GetName(), TNamed::Class()->GetName())))
     fExpectedBaseClass = elem->GetClassPointer();
   
   if (fExpectedChain && (gDebug>3))
     cout << "Expects chain for class " << info->GetName()
          << " in elem " << elem->GetName() << " number " << number << endl;
          
   if (fExpectedBaseClass && (gDebug>3))
     cout << "Expects base class " << fExpectedBaseClass->GetName() << " with standard streamer" << endl;

   if (IsWriting()) {
      CreateElemNode(elem, number);
   } else {
      if (!VerifyElemNode(elem, number)) return;
   }
}

//______________________________________________________________________________
void TXMLBuffer::PerformPostProcessing() 
{
// Function is converts TObject and TString structures to more compact representation

   //cout << "PerformPostProcessing " << Stack() << "  " << Stack(1) << endl; 

   if (GetXmlLayout()==kGeneralized) return;

   const TStreamerElement* elem = Stack()->fElem;
   xmlNodePointer elemnode = IsWriting() ? Stack()->fNode : Stack(1)->fNode;
   
   if ((elem==0) || (elemnode==0)) return;
   
   if (elem->GetType()==TStreamerInfo::kTString) { 
//      cout << "Bad are " << int('<') << " " << int('>') << "  " << int ('\"') << endl;  
   
//      cout << "Has TString " << elem->GetName() << endl;
   
      xmlNodePointer node = fXML->GetChild(elemnode);
      fXML->SkipEmpty(node);
      
      xmlNodePointer nodecharstar = 0;
      xmlNodePointer nodeuchar = 0;
      xmlNodePointer nodeint = 0;
      
      while (node!=0) {
         const char* name = fXML->GetNodeName(node);
//         cout << "Has name " << name << endl;
         if (strcmp(name, xmlNames_UChar)==0) {
            if (nodeuchar) return;
            nodeuchar = node;
         } else 
         if (strcmp(name, xmlNames_Int)==0) {
            if (nodeint) return;
            nodeint = node;
         } else 
         if (strcmp(name, xmlNames_CharStar)==0) {
            if (nodecharstar!=0) return;
            nodecharstar = node;
         } else return; // can not be something else
         fXML->ShiftToNext(node);
      }
//      cout << "Looks good" << endl;
      
      if (nodeuchar==0) return;

      TString str;
      if (nodecharstar!=0)
        str = fXML->GetAttr(nodecharstar, xmlNames_v);
      fXML->NewAttr(elemnode, 0, "str", str);
      
//      cout << "Make string: " << str << endl;
      
      fXML->UnlinkFreeNode(nodeuchar);
      fXML->UnlinkFreeNode(nodeint);  
      fXML->UnlinkFreeNode(nodecharstar);
   } else
   if (elem->GetType()==TStreamerInfo::kTObject) { 
      xmlNodePointer node = fXML->GetChild(elemnode);
      fXML->SkipEmpty(node);
      
      xmlNodePointer vnode = 0;
      xmlNodePointer idnode = 0;
      xmlNodePointer bitsnode = 0;
      xmlNodePointer prnode = 0;
      while (node!=0) {
         const char* name = fXML->GetNodeName(node);
         
         if (strcmp(name, xmlNames_OnlyVersion)==0) {
            if (vnode) return;
            vnode = node;
         } else 
         if (strcmp(name, xmlNames_UInt)==0) {
            if (idnode==0) idnode = node; else
            if (bitsnode==0) bitsnode = node; else return;
         } else
         if (strcmp(name, xmlNames_UShort)==0) {
            if (prnode) return;
            prnode = node;
         } else return; 
         fXML->ShiftToNext(node);
      }
      
      if ((vnode==0) || (idnode==0) || (bitsnode==0)) return;
      
      TString str = fXML->GetAttr(idnode,xmlNames_v);
      fXML->NewAttr(elemnode, 0, "fUniqueID", str);
      
      str = fXML->GetAttr(bitsnode, xmlNames_v);
      UInt_t bits;
      sscanf(str.Data(),"%u", &bits);
      
      char sbuf[20];
      sprintf(sbuf,"%x",bits);
      fXML->NewAttr(elemnode, 0, "fBits", sbuf);
      
      if (prnode!=0) {
         str = fXML->GetAttr(prnode,xmlNames_v);   
         fXML->NewAttr(elemnode, 0, "fProcessID", str);
      }
      
      fXML->UnlinkFreeNode(vnode);
      fXML->UnlinkFreeNode(idnode);
      fXML->UnlinkFreeNode(bitsnode);
      fXML->UnlinkFreeNode(prnode);
   }
}

//______________________________________________________________________________
void TXMLBuffer::PerformPreProcessing(const TStreamerElement* elem, xmlNodePointer elemnode) 
{
// Function is unpack TObject and TString structures to be able read 
// them from custom streamers of this objects
  
   if (GetXmlLayout()==kGeneralized) return;
   if ((elem==0) || (elemnode==0)) return;

   if (elem->GetType()==TStreamerInfo::kTString) { 
//     cout << "Has TString " << elem->GetName() << "  node : " << fXML->GetNodeName(elemnode) << endl;

     if (!fXML->HasAttr(elemnode,"str")) return;
     TString str = fXML->GetAttr(elemnode, "str");
     fXML->FreeAttr(elemnode, "str");
     Int_t len = str.Length();
     
//     cout << "Unpack string : " << str << endl;
     
     xmlNodePointer ucharnode = fXML->NewChild(elemnode, 0, xmlNames_UChar,0);
     
     char sbuf[20];
     sprintf(sbuf,"%d", len);
     if (len<255) {
        fXML->NewAttr(ucharnode,0,xmlNames_v,sbuf); 
     }
     else {
        fXML->NewAttr(ucharnode,0,xmlNames_v,"255"); 
        xmlNodePointer intnode = fXML->NewChild(elemnode, 0, xmlNames_Int, 0); 
        fXML->NewAttr(intnode, 0, xmlNames_v, sbuf); 
        
     }   
     if (len>0) {
       xmlNodePointer node = fXML->NewChild(elemnode, 0, xmlNames_CharStar, 0);
       fXML->NewAttr(node, 0, xmlNames_v, str); 
     }
   } else
   if (elem->GetType()==TStreamerInfo::kTObject) { 
       if (!fXML->HasAttr(elemnode, "fUniqueID")) return;
       if (!fXML->HasAttr(elemnode, "fBits")) return;
       
       TString idstr = fXML->GetAttr(elemnode, "fUniqueID");
       TString bitsstr = fXML->GetAttr(elemnode, "fBits");
       TString prstr = fXML->GetAttr(elemnode, "fProcessID");

       fXML->FreeAttr(elemnode, "fUniqueID");
       fXML->FreeAttr(elemnode, "fBits");
       fXML->FreeAttr(elemnode, "fProcessID");
       
       xmlNodePointer node = fXML->NewChild(elemnode, 0, xmlNames_OnlyVersion, 0);
       fXML->NewAttr(node, 0, xmlNames_v, "1");  
       
       node = fXML->NewChild(elemnode, 0, xmlNames_UInt, 0);
       fXML->NewAttr(node, 0, xmlNames_v, idstr);  
       
       UInt_t bits;
       sscanf(bitsstr.Data(),"%x", &bits);
       char sbuf[20];
       sprintf(sbuf,"%u", bits);
       
       node = fXML->NewChild(elemnode, 0, xmlNames_UInt, 0);
       fXML->NewAttr(node, 0, xmlNames_v, sbuf);  
       
       if (prstr.Length()>0) {
         node = fXML->NewChild(elemnode, 0, xmlNames_UShort, 0);
         fXML->NewAttr(node, 0, xmlNames_v, sbuf);  
       }
   }
}


//______________________________________________________________________________
void TXMLBuffer::BeforeIOoperation() 
{
  // Function is called before any IO operation of TBuffer
  // Now is used to store version value if no proper calls are discovered
  
   CheckVersionBuf();
}

//______________________________________________________________________________
TClass* TXMLBuffer::ReadClass(const TClass*, UInt_t*) 
{
// suppressed function of TBuffer
    
   return 0;
}

//______________________________________________________________________________
void TXMLBuffer::WriteClass(const TClass*) 
{
// suppressed function of TBuffer
    
}

//______________________________________________________________________________
Int_t TXMLBuffer::CheckByteCount(UInt_t /*r_s */, UInt_t /*r_c*/, const TClass* /*cl*/) 
{
// suppressed function of TBuffer    
    
   return 0;
}

//______________________________________________________________________________
Int_t  TXMLBuffer::CheckByteCount(UInt_t, UInt_t, const char*) 
{
// suppressed function of TBuffer    
    
   return 0;
}

//______________________________________________________________________________
void TXMLBuffer::SetByteCount(UInt_t, Bool_t) 
{
// suppressed function of TBuffer    
    
}

//______________________________________________________________________________
Version_t TXMLBuffer::ReadVersion(UInt_t *start, UInt_t *bcnt, const TClass * /*cl*/) 
{
// read version value from buffer
    
    
   if (gDebug>3) 
     cout << "TXMLBuffer::ReadVersion " << endl;

   BeforeIOoperation();

   Version_t res = 0;

   if (start) *start = 0;
   if (bcnt) *bcnt = 0;
   
   if (VerifyItemNode(xmlNames_OnlyVersion)) {
      res = AtoI(XmlReadValue(xmlNames_OnlyVersion));
   } else
   if ((fExpectedBaseClass!=0) && (fXML->HasAttr(Stack(1)->fNode, xmlNames_ClassVersion))) {
      res = fXML->GetIntAttr(Stack(1)->fNode, xmlNames_ClassVersion);
   } else
   if (fXML->HasAttr(StackNode(), xmlNames_ClassVersion)) {
     res = fXML->GetIntAttr(StackNode(), xmlNames_ClassVersion);
   } else {
      Error("ReadVersion", "No correspondent tags to read version");;
      fErrorFlag = 1;
   }

   if (gDebug>2)
     cout << "    version = " << res << endl;

   return res;
}

//______________________________________________________________________________
void TXMLBuffer::CheckVersionBuf() 
{
// checks buffer, filled by WriteVersion 
// if next data is arriving, version should be stored in buffer
    
  if (IsWriting() && (fVersionBuf>=-100)) {
     char sbuf[20];
     sprintf(sbuf, "%d", fVersionBuf);
     XmlWriteValue(sbuf, xmlNames_OnlyVersion);
     fVersionBuf = -111;
  }
}


//______________________________________________________________________________
UInt_t TXMLBuffer::WriteVersion(const TClass *cl, Bool_t /* useBcnt */) 
{
// Copies class version to buffer, but not writes it to xml
// Version will be written with next I/O operation or 
// will be added as attribute of class tag, created by IncrementLevel call
    
   BeforeIOoperation();
   
   if (fExpectedBaseClass!=cl)
     fExpectedBaseClass = 0;

   fVersionBuf = cl->GetClassVersion();

   if (gDebug>2)
      cout << "TXMLBuffer::WriteVersion " << (cl ? cl->GetName() : "null") << "   ver = " << fVersionBuf << endl;

//   XmlWriteBasic(cl->GetClassVersion(), "Version");

   return 0;
}

//______________________________________________________________________________
void* TXMLBuffer::ReadObjectAny(const TClass*) 
{
// Read object from buffer. Only used from TBuffer
    
   BeforeIOoperation();
   if (gDebug>2)
      cout << "TXMLBuffer::ReadObjectAny   " << fXML->GetNodeName(StackNode()) << endl;
   void* res = XmlReadObject(0);
   return res;
}

//______________________________________________________________________________
void TXMLBuffer::WriteObject(const void *actualObjStart, const TClass *actualClass) 
{
// Write object to buffer. Only used from TBuffer
    
   BeforeIOoperation();
   if (gDebug>2)
      cout << "TXMLBuffer::WriteObject of class " << (actualClass ? actualClass->GetName() : " null") << endl;
   XmlWriteObject(actualObjStart, actualClass);
}

// Macro to read content of uncompressed array 
#define TXMLReadArrayNoncompress(vname) \
{ \
   for(Int_t indx=0;indx<n;indx++) \
     XmlReadBasic(vname[indx]); \
}

// macro to read content of array with compression
#define TXMLReadArrayContent(vname) \
{ \
   Int_t indx = 0; \
   while(indx<n) { \
     Int_t cnt = 1; \
     if (fXML->HasAttr(StackNode(), xmlNames_cnt)) \
        cnt = fXML->GetIntAttr(StackNode(), xmlNames_cnt); \
     XmlReadBasic(vname[indx]); \
     Int_t curr = indx; indx++; \
     while(cnt>1) {\
       vname[indx] = vname[curr]; \
       cnt--; indx++; \
     } \
   } \
}

// macro to read array, which include size attribute
#define TXMLBuffer_ReadArray(tname, vname) \
{ \
   BeforeIOoperation(); \
   if (!VerifyItemNode(xmlNames_Array,"ReadArray")) return 0; \
   Int_t n = fXML->GetIntAttr(StackNode(), xmlNames_Size); \
   if (n<=0) return 0; \
   if (!vname) vname = new tname[n]; \
   PushStack(StackNode()); \
   TXMLReadArrayContent(vname); \
   PopStack(); \
   ShiftStack("readarr"); \
   return n; \
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadArray(Bool_t    *&b) 
{
// Read array of Bool_t from buffer
    
   TXMLBuffer_ReadArray(Bool_t,b);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadArray(Char_t    *&c) 
{
// Read array of Char_t from buffer
    
   TXMLBuffer_ReadArray(Char_t,c);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadArray(UChar_t   *&c) 
{
// Read array of UChar_t from buffer
    
   TXMLBuffer_ReadArray(UChar_t,c);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadArray(Short_t   *&h) 
{
// Read array of Short_t from buffer
    
   TXMLBuffer_ReadArray(Short_t,h);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadArray(UShort_t  *&h) 
{
// Read array of UShort_t from buffer
    
   TXMLBuffer_ReadArray(UShort_t,h);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadArray(Int_t     *&i) 
{
// Read array of Int_t from buffer
    
   TXMLBuffer_ReadArray(Int_t,i);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadArray(UInt_t    *&i) 
{
// Read array of UInt_t from buffer
    
   TXMLBuffer_ReadArray(UInt_t,i);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadArray(Long_t    *&l) 
{
// Read array of Long_t from buffer
    
   TXMLBuffer_ReadArray(Long_t,l);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadArray(ULong_t   *&l) 
{
// Read array of ULong_t from buffer
    
   TXMLBuffer_ReadArray(ULong_t,l);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadArray(Long64_t  *&l) 
{
// Read array of Long64_t from buffer
    
   TXMLBuffer_ReadArray(Long64_t,l);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadArray(ULong64_t *&l) 
{
// Read array of ULong64_t from buffer
    
   TXMLBuffer_ReadArray(ULong64_t,l);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadArray(Float_t   *&f) 
{
// Read array of Float_t from buffer
    
   TXMLBuffer_ReadArray(Float_t,f);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadArray(Double_t  *&d) 
{
// Read array of Double_t from buffer
    
   TXMLBuffer_ReadArray(Double_t,d);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadArrayDouble32(Double_t  *&d) 
{
// Read array of Double32_t from buffer
    
   TXMLBuffer_ReadArray(Double_t,d);
}


// macro to read array from xml buffer
#define TXMLBuffer_ReadStaticArray(vname) \
{ \
   BeforeIOoperation(); \
   if (!VerifyItemNode(xmlNames_Array,"ReadStaticArray")) return 0; \
   Int_t n = fXML->GetIntAttr(StackNode(), xmlNames_Size); \
   if (n<=0) return 0; \
   if (!vname) return 0; \
   PushStack(StackNode()); \
   TXMLReadArrayContent(vname); \
   PopStack(); \
   ShiftStack("readstatarr"); \
   return n; \
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadStaticArray(Bool_t    *b) 
{
// Read array of Bool_t from buffer    
    
   TXMLBuffer_ReadStaticArray(b);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadStaticArray(Char_t    *c) 
{
// Read array of Char_t from buffer    
    
   TXMLBuffer_ReadStaticArray(c);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadStaticArray(UChar_t   *c) 
{
// Read array of UChar_t from buffer    
    
   TXMLBuffer_ReadStaticArray(c);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadStaticArray(Short_t   *h) 
{
// Read array of Short_t from buffer    
    
   TXMLBuffer_ReadStaticArray(h);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadStaticArray(UShort_t  *h) 
{
// Read array of UShort_t from buffer    
    
   TXMLBuffer_ReadStaticArray(h);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadStaticArray(Int_t     *i) 
{
// Read array of Int_t from buffer    
    
   TXMLBuffer_ReadStaticArray(i);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadStaticArray(UInt_t    *i) 
{
// Read array of UInt_t from buffer    
    
   TXMLBuffer_ReadStaticArray(i);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadStaticArray(Long_t    *l) 
{
// Read array of Long_t from buffer    
    
   TXMLBuffer_ReadStaticArray(l);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadStaticArray(ULong_t   *l) 
{
// Read array of ULong_t from buffer    
    
   TXMLBuffer_ReadStaticArray(l);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadStaticArray(Long64_t  *l) 
{
// Read array of Long64_t from buffer    
    
   TXMLBuffer_ReadStaticArray(l);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadStaticArray(ULong64_t *l) 
{
// Read array of ULong64_t from buffer    
    
   TXMLBuffer_ReadStaticArray(l);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadStaticArray(Float_t   *f) 
{
// Read array of Float_t from buffer    
    
   TXMLBuffer_ReadStaticArray(f);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadStaticArray(Double_t  *d) 
{
// Read array of Double_t from buffer    
    
   TXMLBuffer_ReadStaticArray(d);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadStaticArrayDouble32(Double_t  *d) 
{
// Read array of Double32_t from buffer    
    
   TXMLBuffer_ReadStaticArray(d);
}

// macro to read content of array, which not include size of array
#define TXMLBuffer_ReadFastArray(vname) \
{ \
   BeforeIOoperation(); \
   if (n<=0) return; \
   if (fExpectedChain) { \
      fExpectedChain = kFALSE; \
      TStreamerInfo* info = Stack(1)->fInfo; \
      Int_t startnumber = Stack(0)->fElemNumber; \
      fCanUseCompact = kTRUE; \
      XmlReadBasic(vname[0]); \
      for(Int_t indx=1;indx<n; indx++) { \
          PopStack(); \
          ShiftStack("chainreader"); \
          TStreamerElement* elem = info->GetStreamerElementReal(startnumber, indx); \
          fCanUseCompact = kTRUE; \
          VerifyElemNode(elem); \
          XmlReadBasic(vname[indx]); \
      } \
   } else { \
      if (!VerifyItemNode(xmlNames_Array,"ReadFastArray")) return; \
      PushStack(StackNode()); \
      TXMLReadArrayContent(vname); \
      PopStack(); \
      ShiftStack("readfastarr"); \
   } \
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArray(Bool_t    *b, Int_t n) 
{
// read array of Bool_t from buffer
    
   TXMLBuffer_ReadFastArray(b);
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArray(Char_t    *c, Int_t n) 
{
// read array of Char_t from buffer
// if nodename==CharStar, read all array as string

   if ((n>0) && VerifyItemNode(xmlNames_CharStar)) {
      const char* buf = XmlReadValue(xmlNames_CharStar);
      Int_t size = strlen(buf);
      if (size<n) size = n;
      memcpy(c, buf, size);
   } else
     TXMLBuffer_ReadFastArray(c);
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArray(UChar_t   *c, Int_t n) 
{
// read array of UChar_t from buffer
    
   TXMLBuffer_ReadFastArray(c);
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArray(Short_t   *h, Int_t n) 
{
// read array of Short_t from buffer
    
   TXMLBuffer_ReadFastArray(h);
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArray(UShort_t  *h, Int_t n) 
{
// read array of UShort_t from buffer
    
   TXMLBuffer_ReadFastArray(h);
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArray(Int_t     *i, Int_t n) 
{
// read array of Int_t from buffer
    
   TXMLBuffer_ReadFastArray(i);
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArray(UInt_t    *i, Int_t n) 
{
// read array of UInt_t from buffer
    
   TXMLBuffer_ReadFastArray(i);
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArray(Long_t    *l, Int_t n) 
{
// read array of Long_t from buffer
    
   TXMLBuffer_ReadFastArray(l);
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArray(ULong_t   *l, Int_t n) 
{
// read array of ULong_t from buffer
    
   TXMLBuffer_ReadFastArray(l);
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArray(Long64_t  *l, Int_t n) 
{
// read array of Long64_t from buffer
    
   TXMLBuffer_ReadFastArray(l);
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArray(ULong64_t *l, Int_t n) 
{
// read array of ULong64_t from buffer
    
   TXMLBuffer_ReadFastArray(l);
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArray(Float_t   *f, Int_t n) 
{
// read array of Float_t from buffer
    
   TXMLBuffer_ReadFastArray(f);
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArray(Double_t  *d, Int_t n) 
{
// read array of Double_t from buffer
    
   TXMLBuffer_ReadFastArray(d);
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArrayDouble32(Double_t  *d, Int_t n) 
{
// read array of Double32_t from buffer
    
   TXMLBuffer_ReadFastArray(d);
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArray(void  *start, const TClass *cl, Int_t n, TMemberStreamer *s) 
{
// redefined here to avoid warning message from gcc
    
    
   TBuffer::ReadFastArray(start, cl, n, s);
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArray(void **startp, const TClass *cl, Int_t n, Bool_t isPreAlloc, TMemberStreamer *s) 
{
// redefined here to avoid warning message from gcc
    
   TBuffer::ReadFastArray(startp, cl, n, isPreAlloc, s);
}

// macro to write content of noncompressed array, not used
#define TXMLWriteArrayNoncompress(vname) \
{ \
   for(Int_t indx=0;indx<n;indx++) \
     XmlWriteBasic(vname[indx]); \
}

// macro to write content of compressed array
#define TXMLWriteArrayCompress(vname) \
{ \
   Int_t indx = 0; \
   while(indx<n) { \
      xmlNodePointer elemnode = XmlWriteBasic(vname[indx]); \
      Int_t curr = indx; indx++; \
      while ((indx<n) && (vname[indx]==vname[curr])) indx++; \
      if (indx-curr > 1)  \
         fXML->NewIntAttr(elemnode, xmlNames_cnt, indx-curr); \
   } \
}

#define TXMLWriteArrayContent(vname) \
{ \
   if (fCompressLevel>0) { \
     TXMLWriteArrayCompress(vname) \
   } else {\
     TXMLWriteArrayNoncompress(vname) \
   } \
}

// macro to write array, which include size
#define TXMLBuffer_WriteArray(vname) \
{ \
   BeforeIOoperation(); \
   xmlNodePointer arrnode = CreateItemNode(xmlNames_Array); \
   fXML->NewIntAttr(arrnode, xmlNames_Size, n); \
   PushStack(arrnode); \
   TXMLWriteArrayContent(vname); \
   PopStack(); \
}

//______________________________________________________________________________
void TXMLBuffer::WriteArray(const Bool_t    *b, Int_t n) 
{
// Write array of Bool_t to buffer
    
   TXMLBuffer_WriteArray(b);
}

//______________________________________________________________________________
void TXMLBuffer::WriteArray(const Char_t    *c, Int_t n) 
{
// Write array of Char_t to buffer
    
   TXMLBuffer_WriteArray(c);
}

//______________________________________________________________________________
void TXMLBuffer::WriteArray(const UChar_t   *c, Int_t n) 
{
// Write array of UChar_t to buffer
    
    TXMLBuffer_WriteArray(c);
}

//______________________________________________________________________________
void TXMLBuffer::WriteArray(const Short_t   *h, Int_t n) 
{
// Write array of Short_t to buffer
    
    TXMLBuffer_WriteArray(h);
}

//______________________________________________________________________________
void TXMLBuffer::WriteArray(const UShort_t  *h, Int_t n) 
{
// Write array of UShort_t to buffer
    
    TXMLBuffer_WriteArray(h);
}

//______________________________________________________________________________
void TXMLBuffer::WriteArray(const Int_t     *i, Int_t n) 
{
// Write array of Int_ to buffer
    
    TXMLBuffer_WriteArray(i);
}

//______________________________________________________________________________
void TXMLBuffer::WriteArray(const UInt_t    *i, Int_t n) 
{
// Write array of UInt_t to buffer
    
    TXMLBuffer_WriteArray(i);
}

//______________________________________________________________________________
void TXMLBuffer::WriteArray(const Long_t    *l, Int_t n) 
{
// Write array of Long_t to buffer
    
    TXMLBuffer_WriteArray(l);
}

//______________________________________________________________________________
void TXMLBuffer::WriteArray(const ULong_t   *l, Int_t n) 
{
// Write array of ULong_t to buffer
    
    TXMLBuffer_WriteArray(l);
}

//______________________________________________________________________________
void TXMLBuffer::WriteArray(const Long64_t  *l, Int_t n) 
{
// Write array of Long64_t to buffer
    
    TXMLBuffer_WriteArray(l);
}

//______________________________________________________________________________
void TXMLBuffer::WriteArray(const ULong64_t *l, Int_t n) 
{
// Write array of ULong64_t to buffer
    
    TXMLBuffer_WriteArray(l);
}

//______________________________________________________________________________
void TXMLBuffer::WriteArray(const Float_t   *f, Int_t n) 
{
// Write array of Float_t to buffer
    
    TXMLBuffer_WriteArray(f);
}

//______________________________________________________________________________
void TXMLBuffer::WriteArray(const Double_t  *d, Int_t n) 
{
// Write array of Double_t to buffer
    
    TXMLBuffer_WriteArray(d);
}

//______________________________________________________________________________
void TXMLBuffer::WriteArrayDouble32(const Double_t  *d, Int_t n) 
{
// Write array of Double32_t to buffer
    
   TXMLBuffer_WriteArray(d);
}

// write array without size attribute
#define TXMLBuffer_WriteFastArray(vname) \
{ \
   BeforeIOoperation(); \
   if (n<=0) return; \
   if (fExpectedChain) { \
      fExpectedChain = kFALSE; \
      TStreamerInfo* info = Stack(1)->fInfo; \
      Int_t startnumber = Stack(0)->fElemNumber; \
      fCanUseCompact = kTRUE; \
      XmlWriteBasic(vname[0]); \
      for(Int_t indx=1;indx<n; indx++) { \
          PopStack(); \
          TStreamerElement* elem = info->GetStreamerElementReal(startnumber, indx); \
          CreateElemNode(elem); \
          fCanUseCompact = kTRUE; \
          XmlWriteBasic(vname[indx]); \
      } \
   } else {\
      xmlNodePointer arrnode = CreateItemNode(xmlNames_Array); \
      PushStack(arrnode); \
      TXMLWriteArrayContent(vname); \
      PopStack(); \
   } \
}

//______________________________________________________________________________
void TXMLBuffer::WriteFastArray(const Bool_t    *b, Int_t n) 
{
// Write array of Bool_t to buffer
    
   TXMLBuffer_WriteFastArray(b);
}

//______________________________________________________________________________
void TXMLBuffer::WriteFastArray(const Char_t    *c, Int_t n) 
{
// Write array of Char_t to buffer
// If array does not include any special characters, 
// it will be reproduced as CharStar node with string as attribute
    
   Bool_t usedefault = (n==0) || fExpectedChain;
   const Char_t* buf = c;
   if (!usedefault)
     for (int i=0;i<n;i++) {
        if ((*buf<26)/* || (*buf=='<') || (*buf=='>') */ || (*buf=='\"'))
         { usedefault = kTRUE; break; }
        buf++;
     }
   if (usedefault) {
      TXMLBuffer_WriteFastArray(c);
   } else {
      Char_t* buf = new Char_t[n+1];
      memcpy(buf, c, n);
      buf[n] = 0;
      XmlWriteValue(buf, xmlNames_CharStar);
      delete[] buf;
   }
}

//______________________________________________________________________________
void TXMLBuffer::WriteFastArray(const UChar_t   *c, Int_t n) 
{
// Write array of UChar_t to buffer
    
   TXMLBuffer_WriteFastArray(c);
}

//______________________________________________________________________________
void TXMLBuffer::WriteFastArray(const Short_t   *h, Int_t n) 
{
// Write array of Short_t to buffer
    
   TXMLBuffer_WriteFastArray(h);
}

//______________________________________________________________________________
void TXMLBuffer::WriteFastArray(const UShort_t  *h, Int_t n) 
{
// Write array of UShort_t to buffer
    
   TXMLBuffer_WriteFastArray(h);
}

//______________________________________________________________________________
void TXMLBuffer::WriteFastArray(const Int_t     *i, Int_t n) 
{
// Write array of Int_t to buffer
    
   TXMLBuffer_WriteFastArray(i);
}

//______________________________________________________________________________
void TXMLBuffer::WriteFastArray(const UInt_t    *i, Int_t n) 
{
// Write array of UInt_t to buffer
    
   TXMLBuffer_WriteFastArray(i);
}

//______________________________________________________________________________
void TXMLBuffer::WriteFastArray(const Long_t    *l, Int_t n) 
{
// Write array of Long_t to buffer
    
   TXMLBuffer_WriteFastArray(l);
}

//______________________________________________________________________________
void TXMLBuffer::WriteFastArray(const ULong_t   *l, Int_t n) 
{
// Write array of ULong_t to buffer
    
   TXMLBuffer_WriteFastArray(l);
}

//______________________________________________________________________________
void TXMLBuffer::WriteFastArray(const Long64_t  *l, Int_t n) 
{
// Write array of Long64_t to buffer
    
   TXMLBuffer_WriteFastArray(l);
}

//______________________________________________________________________________
void TXMLBuffer::WriteFastArray(const ULong64_t *l, Int_t n) 
{
// Write array of ULong64_t to buffer
    
   TXMLBuffer_WriteFastArray(l);
}

//______________________________________________________________________________
void TXMLBuffer::WriteFastArray(const Float_t   *f, Int_t n) 
{
// Write array of Float_t to buffer
    
   TXMLBuffer_WriteFastArray(f);
}

//______________________________________________________________________________
void TXMLBuffer::WriteFastArray(const Double_t  *d, Int_t n) 
{
// Write array of Double_t to buffer
    
   TXMLBuffer_WriteFastArray(d);
}

//______________________________________________________________________________
void TXMLBuffer::WriteFastArrayDouble32(const Double_t  *d, Int_t n) 
{
// Write array of Double32_t to buffer
    
   TXMLBuffer_WriteFastArray(d);
}

//______________________________________________________________________________
void  TXMLBuffer::WriteFastArray(void  *start,  const TClass *cl, Int_t n, TMemberStreamer *s) 
{
// Recall TBuffer function to avoid gcc warning message
    
   TBuffer::WriteFastArray(start, cl, n, s);
}

//______________________________________________________________________________
Int_t TXMLBuffer::WriteFastArray(void **startp, const TClass *cl, Int_t n, Bool_t isPreAlloc, TMemberStreamer *s) 
{
// Recall TBuffer function to avoid gcc warning message
    
   return TBuffer::WriteFastArray(startp, cl, n, isPreAlloc, s);
}

//______________________________________________________________________________
void TXMLBuffer::StreamObject(void *obj, const type_info &typeinfo) 
{
// steram object to/from buffer
    
   StreamObject(obj, gROOT->GetClass(typeinfo));
}

//______________________________________________________________________________
void TXMLBuffer::StreamObject(void *obj, const char *className) 
{
// steram object to/from buffer
    
   StreamObject(obj, gROOT->GetClass(className));
}

//______________________________________________________________________________
void TXMLBuffer::StreamObject(void *obj, const TClass *cl) 
{
// steram object to/from buffer
    
   BeforeIOoperation();
   if (gDebug>1)
     cout << " TXMLBuffer::StreamObject class = " << (cl ? cl->GetName() : "none") << endl;
   if (IsReading()) 
      XmlReadObject(obj);
   else
      XmlWriteObject(obj, cl);
}

// macro for right shift operator for basic type 
#define TXMLBuffer_operatorin(vname) \
{ \
  BeforeIOoperation(); \
  XmlReadBasic(vname); \
  return *this; \
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator>>(Bool_t    &b) 
{
// Reads Bool_t value from buffer
    
   TXMLBuffer_operatorin(b);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator>>(Char_t    &c) 
{
// Reads Char_t value from buffer
    
   TXMLBuffer_operatorin(c);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator>>(UChar_t   &c) 
{
// Reads UChar_t value from buffer
    
   TXMLBuffer_operatorin(c);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator>>(Short_t   &h) 
{
// Reads Short_t value from buffer
    
   TXMLBuffer_operatorin(h);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator>>(UShort_t  &h) 
{
// Reads UShort_t value from buffer
    
   TXMLBuffer_operatorin(h);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator>>(Int_t     &i) 
{
// Reads Int_t value from buffer
    
   TXMLBuffer_operatorin(i);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator>>(UInt_t    &i) 
{
// Reads UInt_t value from buffer
    
   TXMLBuffer_operatorin(i);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator>>(Long_t    &l) 
{
// Reads Long_t value from buffer
    
   TXMLBuffer_operatorin(l);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator>>(ULong_t   &l) 
{
// Reads ULong_t value from buffer
    
   TXMLBuffer_operatorin(l);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator>>(Long64_t  &l) 
{
// Reads Long64_t value from buffer
    
   TXMLBuffer_operatorin(l);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator>>(ULong64_t &l) 
{
// Reads ULong64_t value from buffer
    
   TXMLBuffer_operatorin(l);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator>>(Float_t   &f) 
{
// Reads Float_t value from buffer
    
   TXMLBuffer_operatorin(f);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator>>(Double_t  &d) 
{
// Reads Double_t value from buffer
    
   TXMLBuffer_operatorin(d);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator>>(Char_t    *c) 
{
// Reads array of characters from buffer
    
   BeforeIOoperation();
   const char* buf = XmlReadValue(xmlNames_CharStar);
   strcpy(c, buf);
   return *this;
}

// macro for right shift operator for basic types
#define TXMLBuffer_operatorout(vname) \
{ \
  BeforeIOoperation(); \
  XmlWriteBasic(vname); \
  return *this; \
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator<<(Bool_t    b) 
{
// Writes Bool_t value to buffer
    
   TXMLBuffer_operatorout(b);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator<<(Char_t    c)  
{
// Writes Char_t value to buffer
    
   TXMLBuffer_operatorout(c);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator<<(UChar_t   c)  
{
// Writes UChar_t value to buffer
    
   TXMLBuffer_operatorout(c);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator<<(Short_t   h)  
{
// Writes Short_t value to buffer
    
   TXMLBuffer_operatorout(h);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator<<(UShort_t  h)  
{
// Writes UShort_t value to buffer
    
   TXMLBuffer_operatorout(h);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator<<(Int_t     i)  
{
// Writes Int_t value to buffer
    
   TXMLBuffer_operatorout(i);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator<<(UInt_t    i)  
{
// Writes UInt_t value to buffer
    
   TXMLBuffer_operatorout(i);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator<<(Long_t    l)  
{
// Writes Long_t value to buffer
    
   TXMLBuffer_operatorout(l);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator<<(ULong_t   l)  
{
// Writes ULong_t value to buffer
    
   TXMLBuffer_operatorout(l);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator<<(Long64_t  l)  
{
// Writes Long64_t value to buffer
    
   TXMLBuffer_operatorout(l);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator<<(ULong64_t l)  
{
// Writes ULong64_t value to buffer
    
   TXMLBuffer_operatorout(l);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator<<(Float_t   f)  
{
// Writes Float_t value to buffer
    
   TXMLBuffer_operatorout(f);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator<<(Double_t  d)  
{
// Writes Double_t value to buffer
    
   TXMLBuffer_operatorout(d);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator<<(const Char_t *c) 
{
// Writes array of characters to buffer
    
   BeforeIOoperation();
   XmlWriteValue(c, xmlNames_CharStar);
   return *this;
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::XmlWriteBasic(Char_t value) 
{
// converts Char_t to string and add xml node to buffer

   char buf[50];
   sprintf(buf,"%d",value);
   return XmlWriteValue(buf, xmlNames_Char);
}

//______________________________________________________________________________
xmlNodePointer  TXMLBuffer::XmlWriteBasic(Short_t value) 
{
// converts Short_t to string and add xml node to buffer
    
   char buf[50];
   sprintf(buf,"%hd", value);
   return XmlWriteValue(buf, xmlNames_Short);
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::XmlWriteBasic(Int_t value) 
{
// converts Int_t to string and add xml node to buffer
    
   char buf[50];
   sprintf(buf,"%d", value);
   return XmlWriteValue(buf, xmlNames_Int);
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::XmlWriteBasic(Long_t value) 
{
// converts Long_t to string and add xml node to buffer
    
   char buf[50];
   sprintf(buf,"%ld", value);
   return XmlWriteValue(buf, xmlNames_Long);
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::XmlWriteBasic(Long64_t value) 
{
// converts Long64_t to string and add xml node to buffer
    
   char buf[50];
   sprintf(buf,"%lld", value);
   return XmlWriteValue(buf, xmlNames_Long64);
}

//______________________________________________________________________________
xmlNodePointer  TXMLBuffer::XmlWriteBasic(Float_t value) 
{
// converts Float_t to string and add xml node to buffer
    
   char buf[200];
   sprintf(buf,"%f", value);
   return XmlWriteValue(buf, xmlNames_Float);
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::XmlWriteBasic(Double_t value) 
{
// converts Double_t to string and add xml node to buffer
    
   char buf[1000];
   sprintf(buf,"%f", value);
   return XmlWriteValue(buf, xmlNames_Double);
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::XmlWriteBasic(Bool_t value) 
{
// converts Bool_t to string and add xml node to buffer
    
   return XmlWriteValue(value ? xmlNames_true : xmlNames_false, xmlNames_Bool);
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::XmlWriteBasic(UChar_t value) 
{
// converts UChar_t to string and add xml node to buffer
    
   char buf[50];
   sprintf(buf,"%u", value);
   return XmlWriteValue(buf, xmlNames_UChar);
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::XmlWriteBasic(UShort_t value) 
{
// converts UShort_t to string and add xml node to buffer
    
   char buf[50];
   sprintf(buf,"%hu", value);
   return XmlWriteValue(buf, xmlNames_UShort);
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::XmlWriteBasic(UInt_t value) 
{
// converts UInt_t to string and add xml node to buffer
    
   char buf[50];
   sprintf(buf,"%u", value);
   return XmlWriteValue(buf, xmlNames_UInt);
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::XmlWriteBasic(ULong_t value) 
{
// converts ULong_t to string and add xml node to buffer
    
   char buf[50];
   sprintf(buf,"%lu", value);
   return XmlWriteValue(buf, xmlNames_ULong);
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::XmlWriteBasic(ULong64_t value) 
{
// converts ULong64_t to string and add xml node to buffer
    
   char buf[50];
   sprintf(buf,"%llu", value);
   return XmlWriteValue(buf, xmlNames_ULong64);
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::XmlWriteValue(const char* value, const char* name) 
{
// create xml node with specified name and adds it to stack node
    
   xmlNodePointer node = 0;
   
   if (fCanUseCompact) 
     node = StackNode();
   else 
     node = CreateItemNode(name);

   fXML->NewAttr(node, 0, xmlNames_v, value);

   fCanUseCompact = kFALSE;

   return node;
}

//______________________________________________________________________________
void TXMLBuffer::XmlReadBasic(Char_t& value) 
{
// reads string from current xml node and convert it to Char_t value    
    
   const char* res = XmlReadValue(xmlNames_Char);
   if (res) {
     int n;
     sscanf(res,"%d", &n);
     value = n;
   } else
     value = 0;
}

//______________________________________________________________________________
void TXMLBuffer::XmlReadBasic(Short_t& value) 
{
// reads string from current xml node and convert it to Short_t value    
    
   const char* res = XmlReadValue(xmlNames_Short);
   if (res)
     sscanf(res,"%hd", &value);
   else
     value = 0;
}

//______________________________________________________________________________
void TXMLBuffer::XmlReadBasic(Int_t& value) 
{
// reads string from current xml node and convert it to Int_t value    
    
   const char* res = XmlReadValue(xmlNames_Int);
   if (res)
      sscanf(res,"%d", &value);
   else
      value = 0;
}

//______________________________________________________________________________
void TXMLBuffer::XmlReadBasic(Long_t& value) 
{
// reads string from current xml node and convert it to Long_t value    
    
   const char* res = XmlReadValue(xmlNames_Long);
   if (res)
     sscanf(res,"%ld", &value);
   else
     value = 0;
}

//______________________________________________________________________________
void TXMLBuffer::XmlReadBasic(Long64_t& value) 
{
// reads string from current xml node and convert it to Long64_t value    
    
   const char* res = XmlReadValue(xmlNames_Long64);
   if (res)
     sscanf(res,"%lld", &value);
   else
     value = 0;
}

//______________________________________________________________________________
void TXMLBuffer::XmlReadBasic(Float_t& value) 
{
// reads string from current xml node and convert it to Float_t value    
    
   const char* res = XmlReadValue(xmlNames_Float);
   if (res)
     sscanf(res,"%f", &value);
   else
     value = 0.;
}

//______________________________________________________________________________
void TXMLBuffer::XmlReadBasic(Double_t& value) 
{
// reads string from current xml node and convert it to Double_t value    
    
   const char* res = XmlReadValue(xmlNames_Double);
   if (res)
     sscanf(res,"%lf", &value);
   else
     value = 0.;
}

//______________________________________________________________________________
void TXMLBuffer::XmlReadBasic(Bool_t& value) 
{
// reads string from current xml node and convert it to Bool_t value    
    
   const char* res = XmlReadValue(xmlNames_Bool);
   if (res)
     value = (strcmp(res, xmlNames_true)==0);
   else
     value = kFALSE;
}

//______________________________________________________________________________
void TXMLBuffer::XmlReadBasic(UChar_t& value) 
{
// reads string from current xml node and convert it to UChar_t value    
    
   const char* res = XmlReadValue(xmlNames_UChar);
   if (res) {
     unsigned int n;
     sscanf(res,"%ud", &n);
     value = n;
   } else
     value = 0;
}

//______________________________________________________________________________
void TXMLBuffer::XmlReadBasic(UShort_t& value) 
{
// reads string from current xml node and convert it to UShort_t value    
    
   const char* res = XmlReadValue(xmlNames_UShort);
   if (res)
     sscanf(res,"%hud", &value);
   else
     value = 0;
}

//______________________________________________________________________________
void TXMLBuffer::XmlReadBasic(UInt_t& value) 
{
// reads string from current xml node and convert it to UInt_t value    
    
   const char* res = XmlReadValue(xmlNames_UInt);
   if (res)
     sscanf(res,"%u", &value);
   else
     value = 0;
}

//______________________________________________________________________________
void TXMLBuffer::XmlReadBasic(ULong_t& value) 
{
// reads string from current xml node and convert it to ULong_t value    
    
   const char* res = XmlReadValue(xmlNames_ULong);
   if (res)
     sscanf(res,"%lu", &value);
   else
     value = 0;
}

//______________________________________________________________________________
void TXMLBuffer::XmlReadBasic(ULong64_t& value) 
{
// reads string from current xml node and convert it to ULong64_t value    
    
   const char* res = XmlReadValue(xmlNames_ULong64);
   if (res)
     sscanf(res,"%llu", &value);
   else
     value = 0;
}

//______________________________________________________________________________
const char* TXMLBuffer::XmlReadValue(const char* name) 
{
// read string value from current stack node     

   if (fErrorFlag>0) return 0;

   if (gDebug>4)
      cout << "     read value " << name << " = " ;

   Bool_t trysimple = fCanUseCompact;
   fCanUseCompact = kFALSE;

   if (trysimple)
     if (fXML->HasAttr(Stack(1)->fNode,xmlNames_v))
       fValueBuf = fXML->GetAttr(Stack(1)->fNode, xmlNames_v);
     else
       trysimple = kFALSE;

   if (!trysimple) {
     if (!VerifyItemNode(name, "XmlReadValue")) return 0;
     fValueBuf = fXML->GetAttr(StackNode(), xmlNames_v);
   }

   if (gDebug>4)
     cout << fValueBuf << endl;

   if (!trysimple)
      ShiftStack("readvalue");

   return fValueBuf.Data();
}

