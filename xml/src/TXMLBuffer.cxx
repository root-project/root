// @(#)root/xml:$Name:  $:$Id: TXMLBuffer.cxx,v 1.0 2004/04/21 15:06:45 brun Exp $
// Author: Sergey Linev, Rene Brun  10.05.2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TXMLBuffer.h"
#include "TXMLDtdGenerator.h"
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
    TXMLSetup() {
}

//______________________________________________________________________________
TXMLBuffer::TXMLBuffer(TBuffer::EMode mode, const TXMLSetup& setup, TXMLFile* file) :
    TBuffer(mode),
    TXMLSetup(setup),
    fStoredBuffePos(0),
    fStack(),
    fVersionBuf(-111),
    fDtdGener(0),
    fXmlFile(file),
    fObjMap(0),
    fIdArray(0),
    fErrorFlag(0),
    fCanUseCompact(kFALSE),
    fExpectedChain(kFALSE) {
}

//______________________________________________________________________________
TXMLBuffer::~TXMLBuffer() {
   if (fObjMap) delete fObjMap;
   if (fIdArray) delete fIdArray;
   fStack.Delete();
}

//______________________________________________________________________________
TXMLFile* TXMLBuffer::XmlFile() {
   return fXmlFile;
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::XmlWrite(const TObject* obj) {
   if (obj==0) return XmlWrite(0,0);
          else return XmlWrite(obj, obj->IsA());
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::XmlWrite(const void* obj, const TClass* cl) {
   fErrorFlag = 0;
   
   fStoredBuffePos = Length();

   xmlNodePointer res = XmlWriteObjectNew(obj, cl);
   
   XmlWriteBlock(kTRUE);
   return res;
}

//______________________________________________________________________________
TObject* TXMLBuffer::XmlRead(xmlNodePointer node) {
   void* obj = XmlReadAny(node);

   return (TObject*) obj;
}
    
//______________________________________________________________________________
void* TXMLBuffer::XmlReadAny(xmlNodePointer node) {
   if (node==0) return 0;
   
   fErrorFlag = 0;

   PushStack(node, kTRUE);

   if (IsaSolidDataBlock())
     XmlReadBlock(gXML->GetChild(node));
   
   void* obj = XmlReadObjectNew(0);

   return obj;
}

//______________________________________________________________________________
void TXMLBuffer::WriteObject(const TObject *obj) {
   TBuffer::WriteObject(obj);
}

// **************************** stack functions *****************************

class TXMLStackObj : public TObject {
   public:
      TXMLStackObj(xmlNodePointer node) :
         TObject(), fNode(node), fInfo(0), fLastElem(0), fElemNumber(0) {}

      xmlNodePointer    fNode;
      TStreamerInfo*    fInfo;
      TStreamerElement* fLastElem;
      Int_t             fElemNumber;
};

//______________________________________________________________________________
TXMLStackObj* TXMLBuffer::PushStack(xmlNodePointer current, Bool_t simple) {
  if (IsReading() && !simple) {
    current = gXML->GetChild(current);
    gXML->SkipEmpty(current);
  } 
  
  TXMLStackObj* stack = new TXMLStackObj(current);
  fStack.Add(stack);
  return stack;
}

//______________________________________________________________________________
void TXMLBuffer::PopStack() {
  TObject* last = fStack.Last();
  if (last!=0) {
    fStack.Remove(last);
    delete last;
    fStack.Compress();
  }
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::StackNode() {
  TXMLStackObj* stack = dynamic_cast<TXMLStackObj*> (fStack.Last());
  if (stack==0) return 0;
  return stack->fNode;
}

//______________________________________________________________________________
TXMLStackObj* TXMLBuffer::Stack(Int_t depth) {
  TXMLStackObj* stack = 0;
  if (depth<=fStack.GetLast())
    stack = dynamic_cast<TXMLStackObj*> (fStack.At(fStack.GetLast()-depth));
  return stack;
}

//______________________________________________________________________________
void TXMLBuffer::ShiftStack(const char* errinfo) {
   TXMLStackObj* stack = dynamic_cast<TXMLStackObj*> (fStack.Last());
   if (stack) {
     if (gDebug>4)
       cout << "       Shift " << errinfo << " from " << gXML->GetNodeName(stack->fNode);
     gXML->ShiftToNext(stack->fNode);
     if (gDebug>4)
       cout << " to " << gXML->GetNodeName(stack->fNode) << endl;
   }
}

//______________________________________________________________________________
void TXMLBuffer::XmlWriteBlock(Bool_t force) {
   if ((StackNode()==0) || (fStoredBuffePos>=Length()) || (IsaSolidDataBlock() && !force)) return;

   int BlockSize = Length() - fStoredBuffePos;

   const char* src = Buffer() + fStoredBuffePos;
   int srcSize = BlockSize;

   char* fZipBuffer = 0;

   if ((BlockSize > 512) && (GetXmlLayout()!=kGeneralized)) {
      int ZipBufferSize = BlockSize;
      fZipBuffer = new char[ZipBufferSize];
      int CompressedSize = 0;
      R__zip(1, &BlockSize, Buffer() + fStoredBuffePos, &ZipBufferSize, fZipBuffer, &CompressedSize);

      src = fZipBuffer;
      srcSize = CompressedSize;
   }


   TString res; //("\n");
   char sbuf[200];
   int block = 0;
   char* tgt = sbuf;
   int srcCnt = 0;

   while (srcCnt++<srcSize) {
      tgt+=sprintf(tgt, " %02x", (unsigned char) *src);
      src++;
      if (block++==15) {
         res += sbuf;
         block = 0;
         tgt = sbuf;
      }
   }

   if (block>0) res += sbuf;

   xmlNodePointer node = 0;

   if (GetXmlLayout()==kGeneralized) {
      node = XmlCreateMember(0, "block", res, BlockSize);
   } else {
      node = gXML->NewChild(StackNode(), 0, xmlNames_XmlBlock, res);
      sprintf(sbuf, "%d", BlockSize);
     gXML->NewProp(node, 0, xmlNames_Size, sbuf);

     if (fZipBuffer) {
        sprintf(sbuf, "%d", srcSize);
        gXML->NewProp(node, 0, xmlNames_Zip, sbuf);
        delete[] fZipBuffer;
     }
   }

   fStoredBuffePos = Length();
}

//______________________________________________________________________________
void TXMLBuffer::XmlReadBlock(xmlNodePointer node) {
   return;

   xmlNodePointer blocknode = 0;

   bool usestacknode = (node==0);
   if (usestacknode) node = StackNode();
               else gXML->SkipEmpty(node);

   while (node!=0) {
     if (GetXmlLayout()==kGeneralized) {
        if (VerifyNode(node, xmlNames_Item) && VerifyProp(node, xmlNames_Type, "block")) {
           blocknode = node;
           break;
        }
     } else {
       if (VerifyNode(node, xmlNames_XmlBlock)) {
          blocknode = node;
          break;
       }
     }
     if (usestacknode) break;
                  else gXML->ShiftToNext(node);
   }

   if (blocknode==0) return;

   Int_t blockSize = atoi(gXML->GetProp(blocknode, xmlNames_Size));
   bool blockCompressed = gXML->HasProp(blocknode, xmlNames_Zip);
   char* fUnzipBuffer = 0;

   if (gDebug>2) {
      cout << " TXMLBuffer::XmlReadBlock " << endl;
      cout << "    block size = " << blockSize << endl;
      cout << "    Length = " << Length() << endl;
      cout << "    compressed = " << blockCompressed << endl;
   }

   int pos = fStoredBuffePos;
   if (pos+blockSize>BufferSize()) Expand(pos+blockSize);

   char* tgt = Buffer()+pos;
   Int_t readSize = blockSize;

   TString content;

   if (GetXmlLayout()==kGeneralized)
      content = gXML->GetProp(blocknode, xmlNames_Value);
   else
      content = gXML->GetNodeContent(blocknode);

   if (blockCompressed) {
      Int_t zipSize = atoi(gXML->GetProp(blocknode, xmlNames_Zip));
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
       cout << "Buf[" << i+pos << "] = " << b_hi*16.+b_lo << "  " << *z << *(z+1) << " "<< b_hi << "  " << b_lo << endl;
   }

   if (fUnzipBuffer) {
      int unzipRes = 0;
      R__unzip(&readSize, (unsigned char*) fUnzipBuffer, &blockSize,
                          (unsigned char*) (Buffer()+pos), &unzipRes);
      if (gDebug>2)
         if (unzipRes!=blockSize) cout << "decompression error " << unzipRes << endl;
                             else cout << "unzip ok" << endl;
      delete[] fUnzipBuffer;
   }

   fStoredBuffePos += blockSize;

   if (usestacknode) ShiftStack("blockread");
}


//______________________________________________________________________________
Bool_t TXMLBuffer::ProcessPointer(const void* ptr, xmlNodePointer node) {
  if (node==0) return kFALSE;

  TString refvalue;

  if (ptr==0)
     refvalue = xmlNames_Null;   //null
  else {
     if (fObjMap==0) return kFALSE;

     ULong_t hash = TMath::Hash(&ptr, sizeof(void*));

     xmlNodePointer refnode = (xmlNodePointer) fObjMap->GetValue(hash, (Long_t) ptr);
     if (refnode==0) return kFALSE;

     if (gXML->HasProp(refnode, xmlNames_Ref))
        refvalue = gXML->GetProp(refnode, xmlNames_Ref);
     else {
        refvalue = xmlNames_IdBase;
        if (XmlFile()) refvalue += XmlFile()->GetNextRefCounter();
                  else refvalue += GetNextRefCounter();
        gXML->NewProp(refnode, 0, xmlNames_Ref, refvalue.Data());
     }
  }
  if (refvalue.Length()>0) {
     gXML->NewProp(node, 0, xmlNames_Ptr, refvalue.Data());
     return kTRUE;
  }

  return kFALSE;
}


//______________________________________________________________________________
Bool_t TXMLBuffer::ExtractPointer(xmlNodePointer node, void* &ptr) {

   if (!gXML->HasProp(node,xmlNames_Ptr)) return kFALSE;

   const char* ptrid = gXML->GetProp(node, xmlNames_Ptr);

   if (ptrid==0) return kFALSE;

   // null
   if (strcmp(ptrid, xmlNames_Null)==0) {
      ptr = 0;
      return kTRUE;
   }

   if ((fIdArray==0) || (fObjMap==0)) return kFALSE;

   TObject* obj = fIdArray->FindObject(ptrid);
   if (obj) {
      ptr = (void*) fObjMap->GetValue((Long_t) fIdArray->IndexOf(obj));
      return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
void TXMLBuffer::RegisterPointer(const void* ptr, xmlNodePointer node) {
   if ((node==0) || (ptr==0)) return;

   ULong_t hash = TMath::Hash(&ptr, sizeof(void*));

   if (fObjMap==0) fObjMap = new TExMap();

   if (fObjMap->GetValue(hash, (Long_t) ptr)==0)
     fObjMap->Add(hash, (Long_t) ptr, (Long_t) node);
}

//______________________________________________________________________________
void TXMLBuffer::ExtractObjectId(xmlNodePointer node, const void* ptr) {
   if ((node==0) || (ptr==0)) return;

   const char* refid = gXML->GetProp(node, xmlNames_Ref);

   if (refid==0) return;

   if (fIdArray==0) {
      fIdArray = new TObjArray; fIdArray->SetOwner(kTRUE);
   }
   TNamed* nid = new TNamed(refid,0);
   fIdArray->Add(nid);

   if (fObjMap==0) fObjMap = new TExMap();

   fObjMap->Add((Long_t) fIdArray->IndexOf(nid), (Long_t) ptr);

   if (gDebug>2)
     cout << "----- Find reference " << refid << " for object " << ptr << " ----- "<< endl;
}

//______________________________________________________________________________
Bool_t TXMLBuffer::VerifyNode(xmlNodePointer node, const char* name, const char* errinfo) {
   if ((name==0) || (node==0)) return kFALSE;

   if (strcmp(gXML->GetNodeName(node), name)!=0) {
      if (errinfo) {
         cout << "   Error reading XML file (" << errinfo << "). Get: " << gXML->GetNodeName(node) << "   expects " << name << endl;
        fErrorFlag = 1;
      }
      return kFALSE;
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TXMLBuffer::VerifyStackNode(const char* name, const char* errinfo) {
   return VerifyNode(StackNode(), name, errinfo);
}

//______________________________________________________________________________
Bool_t TXMLBuffer::VerifyProp(xmlNodePointer node, const char* propname, const char* propvalue, const char* errinfo) {
   if ((node==0) || (propname==0) || (propvalue==0)) return kFALSE;
   const char* cont = gXML->GetProp(node, propname);
   if (((cont==0) || (strcmp(cont, propvalue)!=0))) {
       if  (errinfo) {
         cout << "     Error reading XML file (" << errinfo << ") expected " << propvalue <<
                 " for tag property " << propname << " obtain " << cont << endl;
          fErrorFlag = 1;
       }
       return kFALSE;
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TXMLBuffer::VerifyStackProp(const char* propname, const char* propvalue, const char* errinfo) {
   return VerifyProp(StackNode(), propname, propvalue, errinfo);
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::XmlCreateMember(const char* name,
                                           const char* type,
                                           const char* value,
                                           Int_t size) {
   xmlNodePointer node = 0;
   if (GetXmlLayout()==kGeneralized) {
      if (name!=0) {
         node = gXML->NewChild(StackNode(), 0, xmlNames_Member, 0);
         gXML->NewProp(node, 0, xmlNames_Name, name);
      } else
         node = gXML->NewChild(StackNode(), 0, xmlNames_Item, 0);
      gXML->NewProp(node, 0, xmlNames_Type, type);
      if (value) gXML->NewProp(node, 0, xmlNames_Value, value);
      if (size>=0) {
         char ssize[30];
         sprintf(ssize,"%d",size);
         gXML->NewProp(node, 0, xmlNames_Size, ssize);
      }
   } else {
      if (name!=0)
         node = gXML->NewChild(StackNode(), 0, name, value);
      else
         node = gXML->NewChild(StackNode(), 0, type, value);
   }
   return node;
}

//______________________________________________________________________________
Bool_t TXMLBuffer::VerifyMember(const char* name,
                                const char* type,
                                Int_t size,
                                const char* errinfo) {
   if (GetXmlLayout()==kGeneralized) {
      if (name!=0) {
        if (!VerifyStackNode(xmlNames_Member, errinfo)) return kFALSE;
        if (!VerifyStackProp(xmlNames_Name, name, errinfo)) return kFALSE;
      } else
        if (!VerifyStackNode(xmlNames_Item, errinfo)) return kFALSE;

      if (!VerifyStackProp(xmlNames_Type, type, errinfo)) return kFALSE;
      if (size>=0) {
        const char* ssize = gXML->GetProp(StackNode(), xmlNames_Size);
        if ((ssize==0) || (atoi(ssize) != size)) {
           cout << errinfo << " : invalid Member " << name <<" size property. Expected "
                << size << ", get " << ssize << endl;
           fErrorFlag = 1;
           return kFALSE;
       }
     }
     return kTRUE;
   } else
     if (name!=0)
       return VerifyStackNode(name, errinfo);
     else
       return VerifyStackNode(type, errinfo);
}


// ################ redefined virtual functions ###########################

//______________________________________________________________________________
TClass* TXMLBuffer::ReadClass(const TClass*, UInt_t*) {
   if (gDebug>2)
     cout << " TXMLBuffer::ReadClass SUPPRESSED ???????????????????? " << endl;
   return 0;
}

//______________________________________________________________________________
void TXMLBuffer::WriteClass(const TClass*) {
   if (gDebug>2)
      cout << "TXMLBuffer_first::WriteClass SUPPRESSED!!!!!!" << endl;
}


// ########################### XMLStyle1 ################################


//______________________________________________________________________________
Int_t TXMLBuffer::CheckByteCount(UInt_t /*r_s */, UInt_t /*r_c*/, const TClass* /*cl*/) {
   return 0;
}

//______________________________________________________________________________
Int_t  TXMLBuffer::CheckByteCount(UInt_t, UInt_t, const char*) {
   return 0;
}

//______________________________________________________________________________
void TXMLBuffer::SetByteCount(UInt_t, Bool_t) {
}

Version_t TXMLBuffer::ReadVersion(UInt_t *start, UInt_t *bcnt, const TClass * /*cl*/) {

   BeforeIOoperation();

   Version_t res = 0;

   if (start) *start = 0;
   if (bcnt) *bcnt = 0;

   if (VerifyNode(StackNode(),"Version")) {
      res = atoi(XmlReadValue("Version"));
     //      cout << "Reading version from Version tag" << endl;
   } else
   if (VerifyNode(StackNode(),xmlNames_Class)) {
     res = atoi(gXML->GetProp(StackNode(), xmlNames_Version));
   } else {
      cerr << "Error reading version " << endl;
      fErrorFlag = 1;
   }

   if (gDebug>2)
     cout << "    version = " << res << endl;

   return res;
}

//______________________________________________________________________________
void TXMLBuffer::CheckVersionBuf() {
  if (IsWriting() && (fVersionBuf>=-100)) {
     char sbuf[20];
     sprintf(sbuf, "%d", fVersionBuf);
     XmlWriteValue(sbuf, "Version");
     fVersionBuf = -111;
  }
}


//______________________________________________________________________________
UInt_t TXMLBuffer::WriteVersion(const TClass *cl, Bool_t /* useBcnt */) {
//   cout << " ----> " << cl->GetName() << endl;

   BeforeIOoperation();
   
   fVersionBuf = cl->GetClassVersion();


   if (gDebug>2)
      cout << "##### TXMLBuffer::WriteVersion " << endl;

//   XmlWriteBasic(cl->GetClassVersion(), "Version");

   return 0;
}

//______________________________________________________________________________
void* TXMLBuffer::ReadObjectAny(const TClass*) {
   BeforeIOoperation();
   if (gDebug>2)
      cout << "TXMLBuffer::ReadObjectAny   " << gXML->GetNodeName(StackNode()) << endl;
   void* res = XmlReadObjectNew(0);
   return res;
}

//______________________________________________________________________________
void TXMLBuffer::WriteObject(const void *actualObjStart, const TClass *actualClass) {
   BeforeIOoperation();
   if (gDebug>2)
      cout << "TXMLBuffer::WriteObject of class " << (actualClass ? actualClass->GetName() : " null") << endl;
   XmlWriteObjectNew(actualObjStart, actualClass);
}

// ########################### XMLStyle2 ################################

#define TXMLReadArrayCompress0(vname) { \
   for(Int_t indx=0;indx<n;indx++) \
     XmlReadBasic(vname[indx]); \
}

#define TXMLReadArrayCompress(vname) { \
   Int_t indx = 0; \
   while(indx<n) { \
     Int_t cnt = 1; \
     if (gXML->HasProp(StackNode(), "cnt")) { \
        cnt = atoi(gXML->GetProp(StackNode(), "cnt")); \
     } \
     XmlReadBasic(vname[indx]); \
     Int_t curr = indx; indx++; \
     while(cnt>1) {\
       vname[indx] = vname[curr]; \
       cnt--; indx++; \
     } \
   } \
}



#define TXMLBuffer_ReadArray(tname, vname) \
{ \
   BeforeIOoperation(); \
   if (!IsConvertBasicTypes()) \
      return TBuffer::ReadArray(vname); \
   Int_t n = atoi(gXML->GetProp(StackNode(), xmlNames_Size)); \
   if (n<=0) return 0; \
   if (!vname) vname = new tname[n]; \
   PushStack(StackNode()); \
   TXMLReadArrayCompress(vname); \
   PopStack(); \
   ShiftStack("readarr"); \
   return n; \
}


//______________________________________________________________________________
Int_t TXMLBuffer::ReadArray(Bool_t    *&b) {
   TXMLBuffer_ReadArray(Bool_t,b);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadArray(Char_t    *&c) {
   TXMLBuffer_ReadArray(Char_t,c);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadArray(UChar_t   *&c) {
   TXMLBuffer_ReadArray(UChar_t,c);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadArray(Short_t   *&h) {
   TXMLBuffer_ReadArray(Short_t,h);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadArray(UShort_t  *&h) {
   TXMLBuffer_ReadArray(UShort_t,h);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadArray(Int_t     *&i) {
   TXMLBuffer_ReadArray(Int_t,i);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadArray(UInt_t    *&i) {
   TXMLBuffer_ReadArray(UInt_t,i);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadArray(Long_t    *&l) {
   TXMLBuffer_ReadArray(Long_t,l);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadArray(ULong_t   *&l) {
   TXMLBuffer_ReadArray(ULong_t,l);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadArray(Long64_t  *&l) {
   TXMLBuffer_ReadArray(Long64_t,l);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadArray(ULong64_t *&l) {
   TXMLBuffer_ReadArray(ULong64_t,l);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadArray(Float_t   *&f) {
   TXMLBuffer_ReadArray(Float_t,f);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadArray(Double_t  *&d) {
   TXMLBuffer_ReadArray(Double_t,d);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadArrayDouble32(Double_t  *&d) {
   if (!IsConvertBasicTypes())
      return TBuffer::ReadArrayDouble32(d);
   TXMLBuffer_ReadArray(Double_t,d);
}


#define TXMLBuffer_ReadStaticArray(vname) \
{ \
   BeforeIOoperation(); \
   if (!IsConvertBasicTypes()) \
      return TBuffer::ReadStaticArray(vname); \
   Int_t n = atoi(gXML->GetProp(StackNode(), xmlNames_Size)); \
   if (n<=0) return 0; \
   if (!vname) return 0; \
   PushStack(StackNode()); \
   TXMLReadArrayCompress(vname); \
   PopStack(); \
   ShiftStack("readstatarr"); \
   return n; \
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadStaticArray(Bool_t    *b) {
   TXMLBuffer_ReadStaticArray(b);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadStaticArray(Char_t    *c) {
   TXMLBuffer_ReadStaticArray(c);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadStaticArray(UChar_t   *c) {
   TXMLBuffer_ReadStaticArray(c);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadStaticArray(Short_t   *h) {
   TXMLBuffer_ReadStaticArray(h);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadStaticArray(UShort_t  *h) {
   TXMLBuffer_ReadStaticArray(h);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadStaticArray(Int_t     *i) {
   TXMLBuffer_ReadStaticArray(i);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadStaticArray(UInt_t    *i) {
   TXMLBuffer_ReadStaticArray(i);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadStaticArray(Long_t    *l) {
   TXMLBuffer_ReadStaticArray(l);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadStaticArray(ULong_t   *l) {
   TXMLBuffer_ReadStaticArray(l);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadStaticArray(Long64_t  *l) {
   TXMLBuffer_ReadStaticArray(l);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadStaticArray(ULong64_t *l) {
   TXMLBuffer_ReadStaticArray(l);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadStaticArray(Float_t   *f) {
   TXMLBuffer_ReadStaticArray(f);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadStaticArray(Double_t  *d) {
   TXMLBuffer_ReadStaticArray(d);
}

//______________________________________________________________________________
Int_t TXMLBuffer::ReadStaticArrayDouble32(Double_t  *d) {
   if (!IsConvertBasicTypes())
      return TBuffer::ReadStaticArrayDouble32(d);
   TXMLBuffer_ReadStaticArray(d);
}

#define TXMLBuffer_ReadFastArray(vname) \
{ \
   BeforeIOoperation(); \
   if (!IsConvertBasicTypes()) { \
      TBuffer::ReadFastArray(vname, n); \
      return; \
   } \
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
      PushStack(StackNode()); \
      TXMLReadArrayCompress(vname); \
      PopStack(); \
      ShiftStack("readfastarr"); \
   } \
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArray(Bool_t    *b, Int_t n) {
   TXMLBuffer_ReadFastArray(b);
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArray(Char_t    *c, Int_t n) {
   if ((n>0) && VerifyNode(StackNode(), xmlNames_CharStar)) {
      const char* buf = XmlReadValue(xmlNames_CharStar);
      Int_t size = strlen(buf);
      if (size<n) size = n;
      memcpy(c, buf, size);
   } else
     TXMLBuffer_ReadFastArray(c);
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArray(UChar_t   *c, Int_t n) {
   TXMLBuffer_ReadFastArray(c);
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArray(Short_t   *h, Int_t n) {
   TXMLBuffer_ReadFastArray(h);
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArray(UShort_t  *h, Int_t n) {
   TXMLBuffer_ReadFastArray(h);
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArray(Int_t     *i, Int_t n) {
   TXMLBuffer_ReadFastArray(i);
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArray(UInt_t    *i, Int_t n) {
   TXMLBuffer_ReadFastArray(i);
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArray(Long_t    *l, Int_t n) {
   TXMLBuffer_ReadFastArray(l);
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArray(ULong_t   *l, Int_t n) {
   TXMLBuffer_ReadFastArray(l);
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArray(Long64_t  *l, Int_t n) {
   TXMLBuffer_ReadFastArray(l);
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArray(ULong64_t *l, Int_t n) {
   TXMLBuffer_ReadFastArray(l);
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArray(Float_t   *f, Int_t n) {
   TXMLBuffer_ReadFastArray(f);
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArray(Double_t  *d, Int_t n) {
   TXMLBuffer_ReadFastArray(d);
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArrayDouble32(Double_t  *d, Int_t n) {
   BeforeIOoperation();
   if (!IsConvertBasicTypes()) {
      TBuffer::ReadFastArrayDouble32(d, n);
      return;
   }
   TXMLBuffer_ReadFastArray(d);
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArray(void  *start, const TClass *cl, Int_t n, TMemberStreamer *s) {
   TBuffer::ReadFastArray(start, cl, n, s);
}

//______________________________________________________________________________
void TXMLBuffer::ReadFastArray(void **startp, const TClass *cl, Int_t n, Bool_t isPreAlloc, TMemberStreamer *s) {
   TBuffer::ReadFastArray(startp, cl, n, isPreAlloc, s);
}

#define TXMLWriteArrayCompress0(vname) { \
   for(Int_t indx=0;indx<n;indx++) \
     XmlWriteBasic(vname[indx]); \
}


#define TXMLWriteArrayCompress(vname) { \
   Int_t indx = 0; \
   while(indx<n) { \
      xmlNodePointer elemnode = XmlWriteBasic(vname[indx]); \
      Int_t curr = indx; indx++; \
      while ((indx<n) && (vname[indx]==vname[curr])) indx++; \
      if (indx-curr > 1) { \
         char sbuf[10]; \
         sprintf(sbuf,"%d",indx-curr); \
         gXML->NewProp(elemnode,0,"cnt",sbuf); \
      } \
   } \
}


#define TXMLBuffer_WriteArray(vname) \
{ \
   BeforeIOoperation(); \
   if (!IsConvertBasicTypes()) { \
      TBuffer::WriteArray(vname, n); \
      return; \
   } \
   xmlNodePointer arrnode = gXML->NewChild(StackNode(), 0, xmlNames_Array, 0); \
   char sbuf[50]; \
   sprintf(sbuf,"%d",n); \
   gXML->NewProp(arrnode, 0, xmlNames_Size, sbuf); \
   PushStack(arrnode); \
   TXMLWriteArrayCompress(vname); \
   PopStack(); \
}

//______________________________________________________________________________
void TXMLBuffer::WriteArray(const Bool_t    *b, Int_t n) {
    TXMLBuffer_WriteArray(b);
}

//______________________________________________________________________________
void TXMLBuffer::WriteArray(const Char_t    *c, Int_t n) {
    TXMLBuffer_WriteArray(c);
}

//______________________________________________________________________________
void TXMLBuffer::WriteArray(const UChar_t   *c, Int_t n) {
    TXMLBuffer_WriteArray(c);
}

//______________________________________________________________________________
void TXMLBuffer::WriteArray(const Short_t   *h, Int_t n) {
    TXMLBuffer_WriteArray(h);
}

//______________________________________________________________________________
void TXMLBuffer::WriteArray(const UShort_t  *h, Int_t n) {
    TXMLBuffer_WriteArray(h);
}

//______________________________________________________________________________
void TXMLBuffer::WriteArray(const Int_t     *i, Int_t n) {
    TXMLBuffer_WriteArray(i);
}

//______________________________________________________________________________
void TXMLBuffer::WriteArray(const UInt_t    *i, Int_t n) {
    TXMLBuffer_WriteArray(i);
}

//______________________________________________________________________________
void TXMLBuffer::WriteArray(const Long_t    *l, Int_t n) {
    TXMLBuffer_WriteArray(l);
}

//______________________________________________________________________________
void TXMLBuffer::WriteArray(const ULong_t   *l, Int_t n) {
    TXMLBuffer_WriteArray(l);
}

//______________________________________________________________________________
void TXMLBuffer::WriteArray(const Long64_t  *l, Int_t n) {
    TXMLBuffer_WriteArray(l);
}

//______________________________________________________________________________
void TXMLBuffer::WriteArray(const ULong64_t *l, Int_t n) {
    TXMLBuffer_WriteArray(l);
}

//______________________________________________________________________________
void TXMLBuffer::WriteArray(const Float_t   *f, Int_t n) {
    TXMLBuffer_WriteArray(f);
}

//______________________________________________________________________________
void TXMLBuffer::WriteArray(const Double_t  *d, Int_t n) {
    TXMLBuffer_WriteArray(d);
}

//______________________________________________________________________________
void TXMLBuffer::WriteArrayDouble32(const Double_t  *d, Int_t n) {
   if (!IsConvertBasicTypes()) {
      TBuffer::WriteArrayDouble32(d, n);
      return;
   }
   TXMLBuffer_WriteArray(d);
}


#define TXMLBuffer_WriteFastArray(vname) \
{ \
   BeforeIOoperation(); \
   if (!IsConvertBasicTypes()) { \
      TBuffer::WriteFastArray(vname, n); \
      return; \
   } \
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
      xmlNodePointer arrnode = gXML->NewChild(StackNode(), 0, xmlNames_Array, 0); \
      PushStack(arrnode); \
      TXMLWriteArrayCompress(vname); \
      PopStack(); \
   } \
}

//______________________________________________________________________________
void TXMLBuffer::WriteFastArray(const Bool_t    *b, Int_t n) {
   TXMLBuffer_WriteFastArray(b);
}

//______________________________________________________________________________
void TXMLBuffer::WriteFastArray(const Char_t    *c, Int_t n) {
   Bool_t usedefault = (n==0) || fExpectedChain;
   const Char_t* buf = c;
   if (!usedefault)
     for (int i=0;i<n;i++) {
        if ((*buf<26) || (*buf=='<') || (*buf=='>') || (*buf=='\"'))
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
void TXMLBuffer::WriteFastArray(const UChar_t   *c, Int_t n) {
   TXMLBuffer_WriteFastArray(c);
}

//______________________________________________________________________________
void TXMLBuffer::WriteFastArray(const Short_t   *h, Int_t n) {
   TXMLBuffer_WriteFastArray(h);
}

//______________________________________________________________________________
void TXMLBuffer::WriteFastArray(const UShort_t  *h, Int_t n) {
   TXMLBuffer_WriteFastArray(h);
}

//______________________________________________________________________________
void TXMLBuffer::WriteFastArray(const Int_t     *i, Int_t n) {
   TXMLBuffer_WriteFastArray(i);
}

//______________________________________________________________________________
void TXMLBuffer::WriteFastArray(const UInt_t    *i, Int_t n) {
   TXMLBuffer_WriteFastArray(i);
}

//______________________________________________________________________________
void TXMLBuffer::WriteFastArray(const Long_t    *l, Int_t n) {
   TXMLBuffer_WriteFastArray(l);
}

//______________________________________________________________________________
void TXMLBuffer::WriteFastArray(const ULong_t   *l, Int_t n) {
   TXMLBuffer_WriteFastArray(l);
}

//______________________________________________________________________________
void TXMLBuffer::WriteFastArray(const Long64_t  *l, Int_t n) {
   TXMLBuffer_WriteFastArray(l);
}

//______________________________________________________________________________
void TXMLBuffer::WriteFastArray(const ULong64_t *l, Int_t n) {
   TXMLBuffer_WriteFastArray(l);
}

//______________________________________________________________________________
void TXMLBuffer::WriteFastArray(const Float_t   *f, Int_t n) {
   TXMLBuffer_WriteFastArray(f);
}

//______________________________________________________________________________
void TXMLBuffer::WriteFastArray(const Double_t  *d, Int_t n) {
   TXMLBuffer_WriteFastArray(d);
}

//______________________________________________________________________________
void TXMLBuffer::WriteFastArrayDouble32(const Double_t  *d, Int_t n) {
   BeforeIOoperation();
   if (!IsConvertBasicTypes()) {
      TBuffer::WriteFastArrayDouble32(d, n);
      return;
   }
   TXMLBuffer_WriteFastArray(d);
}

//______________________________________________________________________________
void  TXMLBuffer::WriteFastArray(void  *start,  const TClass *cl, Int_t n, TMemberStreamer *s) {
   TBuffer::WriteFastArray(start, cl, n, s);
}

Int_t TXMLBuffer::WriteFastArray(void **startp, const TClass *cl, Int_t n, Bool_t isPreAlloc, TMemberStreamer *s) {
   return TBuffer::WriteFastArray(startp, cl, n, isPreAlloc, s);
}


//______________________________________________________________________________
void TXMLBuffer::StreamObject(void *obj, const type_info &typeinfo) {
   StreamObject(obj, gROOT->GetClass(typeinfo));
}

//______________________________________________________________________________
void TXMLBuffer::StreamObject(void *obj, const char *className) {
   StreamObject(obj, gROOT->GetClass(className));
}

//______________________________________________________________________________
void TXMLBuffer::StreamObject(void *obj, const TClass *cl) {
   BeforeIOoperation();
   if (gDebug>1)
     cout << " TXMLBuffer::StreamObject class = " << (cl ? cl->GetName() : "none") << endl;
   if (IsReading()) {
      XmlReadObjectNew(obj);
   } else {
      XmlWriteObjectNew(obj, cl);
   }
}

#define TXMLBuffer_operatorin(vname) \
{ \
  BeforeIOoperation(); \
  if (IsConvertBasicTypes()) { \
    XmlReadBasic(vname); \
    return *this; \
  } else \
    return TBuffer::operator>>(vname); \
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator>>(Bool_t    &b) {
   TXMLBuffer_operatorin(b);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator>>(Char_t    &c) {
   TXMLBuffer_operatorin(c);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator>>(UChar_t   &c) {
   TXMLBuffer_operatorin(c);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator>>(Short_t   &h) {
   TXMLBuffer_operatorin(h);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator>>(UShort_t  &h) {
   TXMLBuffer_operatorin(h);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator>>(Int_t     &i) {
   TXMLBuffer_operatorin(i);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator>>(UInt_t    &i) {
   TXMLBuffer_operatorin(i);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator>>(Long_t    &l) {
   TXMLBuffer_operatorin(l);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator>>(ULong_t   &l) {
   TXMLBuffer_operatorin(l);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator>>(Long64_t  &l) {
   TXMLBuffer_operatorin(l);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator>>(ULong64_t &l) {
   TXMLBuffer_operatorin(l);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator>>(Float_t   &f) {
   TXMLBuffer_operatorin(f);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator>>(Double_t  &d) {
   TXMLBuffer_operatorin(d);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator>>(Char_t    *c) {
   BeforeIOoperation();
   if (IsConvertBasicTypes()) {
      const char* buf = XmlReadValue(xmlNames_CharStar);
      strcpy(c, buf);
      return *this;
   } else return TBuffer::operator>>(c);
}

#define TXMLBuffer_operatorout(vname) \
{ \
  BeforeIOoperation(); \
  if (IsConvertBasicTypes()) { \
      XmlWriteBasic(vname); \
      return *this; \
   } else { \
     return TBuffer::operator<<(vname); \
  } \
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator<<(Bool_t    b) {
   TXMLBuffer_operatorout(b);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator<<(Char_t    c)  {
   TXMLBuffer_operatorout(c);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator<<(UChar_t   c)  {
   TXMLBuffer_operatorout(c);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator<<(Short_t   h)  {
   TXMLBuffer_operatorout(h);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator<<(UShort_t  h)  {
   TXMLBuffer_operatorout(h);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator<<(Int_t     i)  {
   TXMLBuffer_operatorout(i);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator<<(UInt_t    i)  {
   TXMLBuffer_operatorout(i);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator<<(Long_t    l)  {
   TXMLBuffer_operatorout(l);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator<<(ULong_t   l)  {
   TXMLBuffer_operatorout(l);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator<<(Long64_t  l)  {
   TXMLBuffer_operatorout(l);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator<<(ULong64_t l)  {
   TXMLBuffer_operatorout(l);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator<<(Float_t   f)  {
   TXMLBuffer_operatorout(f);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator<<(Double_t  d)  {
   TXMLBuffer_operatorout(d);
}

//______________________________________________________________________________
TBuffer& TXMLBuffer::operator<<(const Char_t *c) {
   BeforeIOoperation();
   if (IsConvertBasicTypes()) {
      XmlWriteValue(c, xmlNames_CharStar);
      return *this;
   } else return TBuffer::operator<<(c);
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::XmlWriteBasic(Char_t value) {
   char buf[50];
   sprintf(buf,"%d", value);
   return XmlWriteValue(buf, xmlNames_Char);
}

//______________________________________________________________________________
xmlNodePointer  TXMLBuffer::XmlWriteBasic(Short_t value) {
   char buf[50];
   sprintf(buf,"%hd", value);
   return XmlWriteValue(buf, xmlNames_Short);
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::XmlWriteBasic(Int_t value) {
   char buf[50];
   sprintf(buf,"%d", value);
   return XmlWriteValue(buf, xmlNames_Int);
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::XmlWriteBasic(Long_t value) {
   char buf[50];
   sprintf(buf,"%ld", value);
   return XmlWriteValue(buf, xmlNames_Long);
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::XmlWriteBasic(Long64_t value) {
   char buf[50];
   sprintf(buf,"%lld", value);
   return XmlWriteValue(buf, xmlNames_Long64);
}

//______________________________________________________________________________
xmlNodePointer  TXMLBuffer::XmlWriteBasic(Float_t value) {
   char buf[200];
   sprintf(buf,"%f", value);
   return XmlWriteValue(buf, xmlNames_Float);
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::XmlWriteBasic(Double_t value) {
   char buf[1000];
   sprintf(buf,"%f", value);
   return XmlWriteValue(buf, xmlNames_Double);
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::XmlWriteBasic(Bool_t value) {
   return XmlWriteValue(value ? "true" : "false", xmlNames_Bool);
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::XmlWriteBasic(UChar_t value) {
   char buf[50];
   sprintf(buf,"%u", value);
   return XmlWriteValue(buf, xmlNames_UChar);
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::XmlWriteBasic(UShort_t value) {
   char buf[50];
   sprintf(buf,"%hu", value);
   return XmlWriteValue(buf, xmlNames_UShort);
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::XmlWriteBasic(UInt_t value) {
   char buf[50];
   sprintf(buf,"%u", value);
   return XmlWriteValue(buf, xmlNames_UInt);
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::XmlWriteBasic(ULong_t value) {
   char buf[50];
   sprintf(buf,"%lu", value);
   return XmlWriteValue(buf, xmlNames_ULong);
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::XmlWriteBasic(ULong64_t value) {
   char buf[50];
   sprintf(buf,"%llu", value);
   return XmlWriteValue(buf, xmlNames_ULong64);
}

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::XmlWriteValue(const char* value,
                                         const char* name) {
   TXMLLayout mode = GetXmlLayout();

   xmlNodePointer node = 0;

   if (mode==kGeneralized) {
      //      node = XmlCreateMember(name, name_t, value);
   } else {
     if (fCanUseCompact) node = StackNode();
                    else node = gXML->NewChild(StackNode(), 0, name, 0);

     gXML->NewProp(node, 0, "v", value);
   }

   fCanUseCompact = kFALSE;
   
   return node;
}

//______________________________________________________________________________
void TXMLBuffer::XmlReadBasic(Char_t& value) {
   const char* res = XmlReadValue(xmlNames_Char);
   if (res) {
     int n;
     sscanf(res,"%d", &n);
     value = n;
   } else
     value = 0;

}

//______________________________________________________________________________
void TXMLBuffer::XmlReadBasic(Short_t& value) {
   const char* res = XmlReadValue(xmlNames_Short);
   if (res)
     sscanf(res,"%hd", &value);
   else
     value = 0;
}

//______________________________________________________________________________
void TXMLBuffer::XmlReadBasic(Int_t& value) {
   const char* res = XmlReadValue(xmlNames_Int);
   if (res)
      sscanf(res,"%d", &value);
   else
      value = 0;
}

//______________________________________________________________________________
void TXMLBuffer::XmlReadBasic(Long_t& value) {
   const char* res = XmlReadValue(xmlNames_Long);
   if (res)
     sscanf(res,"%ld", &value);
   else
     value = 0;
}

//______________________________________________________________________________
void TXMLBuffer::XmlReadBasic(Long64_t& value) {
   const char* res = XmlReadValue(xmlNames_Long64);
   if (res)
     sscanf(res,"%lld", &value);
   else
     value = 0;
}

//______________________________________________________________________________
void TXMLBuffer::XmlReadBasic(Float_t& value) {
   const char* res = XmlReadValue(xmlNames_Float);
   if (res)
     sscanf(res,"%f", &value);
   else
     value = 0.;
}

//______________________________________________________________________________
void TXMLBuffer::XmlReadBasic(Double_t& value) {
   const char* res = XmlReadValue(xmlNames_Double);
   if (res)
     sscanf(res,"%lf", &value);
   else
     value = 0.;
}

//______________________________________________________________________________
void TXMLBuffer::XmlReadBasic(Bool_t& value) {
   const char* res = XmlReadValue(xmlNames_Bool);
   if (res)
     value = (strcmp(res,"true")==0);
   else
     value = kFALSE;
}

//______________________________________________________________________________
void TXMLBuffer::XmlReadBasic(UChar_t& value) {
   const char* res = XmlReadValue(xmlNames_UChar);
   if (res) {
     unsigned int n;
     sscanf(res,"%ud", &n);
     value = n;
   } else
     value = 0;
}

//______________________________________________________________________________
void TXMLBuffer::XmlReadBasic(UShort_t& value) {
   const char* res = XmlReadValue(xmlNames_UShort);
   if (res)
     sscanf(res,"%hud", &value);
   else
     value = 0;
}

//______________________________________________________________________________
void TXMLBuffer::XmlReadBasic(UInt_t& value) {
   const char* res = XmlReadValue(xmlNames_UInt);
   if (res)
     sscanf(res,"%u", &value);
   else
     value = 0;
}

//______________________________________________________________________________
void TXMLBuffer::XmlReadBasic(ULong_t& value) {
   const char* res = XmlReadValue(xmlNames_ULong);
   if (res)
     sscanf(res,"%lu", &value);
   else
     value = 0;
}

//______________________________________________________________________________
void TXMLBuffer::XmlReadBasic(ULong64_t& value) {
   const char* res = XmlReadValue(xmlNames_ULong64);
   if (res)
     sscanf(res,"%llu", &value);
   else
     value = 0;
}

//______________________________________________________________________________
const char* TXMLBuffer::XmlReadValue(const char* name) {
   if (fErrorFlag>0) return 0;

   TXMLLayout mode = GetXmlLayout();

   if (mode==kGeneralized) {
      //      if (!VerifyMember(name, name_t, -1, "XmlReadValue")) return 0;
      //      fValueBuf = gXML->GetProp(StackNode(), xmlNames_Value);
      //      ShiftStack("general");
   } else {

      if (gDebug>4)
         cout << "     read value " << name << " = " ;

      Bool_t trysimple = fCanUseCompact;
      fCanUseCompact = kFALSE;

      if (trysimple)
        if (gXML->HasProp(Stack(1)->fNode,"v"))
          fValueBuf = gXML->GetProp(Stack(1)->fNode, "v");
        else
          trysimple = kFALSE;

      if (!trysimple) {
        if (!VerifyStackNode(name, "XmlReadValue")) return 0;
        fValueBuf = gXML->GetProp(StackNode(), "v");
      }

      if (gDebug>4)
         cout << fValueBuf << endl;
         
      if (!trysimple)
         ShiftStack("readvalue");
   }
            
   return fValueBuf.Data();
}


// ******************************** new stuff ***********************************8

//______________________________________________________________________________
xmlNodePointer TXMLBuffer::XmlWriteObjectNew(const void* obj,
                                             const TClass* cl) {
   xmlNodePointer objnode = gXML->NewChild(StackNode(), 0, xmlNames_Object, 0);

   if (!cl) obj = 0;
   if (ProcessPointer(obj, objnode)) return objnode;

   TString clname = XmlConvertClassName(cl);

   gXML->NewProp(objnode, 0, "class", clname);

   RegisterPointer(obj, objnode);

   PushStack(objnode);

   ((TClass*)cl)->Streamer((void*)obj, *this);

   PopStack();

   if (gDebug>1)
      cout << "Done write of " << cl->GetName() << endl;

   return objnode;
}

//______________________________________________________________________________
void* TXMLBuffer::XmlReadObjectNew(void* obj) {

   xmlNodePointer objnode = StackNode();

   if (fErrorFlag>0) return obj;

   if (objnode==0) return obj;

   if (!VerifyNode(objnode, xmlNames_Object, "XmlReadObjectNew")) return obj;

   if (ExtractPointer(objnode, obj))
      return obj;

   TString clname = gXML->GetProp(objnode, "class");
   TClass* objClass = XmlDefineClass(clname);

   if (objClass==0) {
      cerr << "Cannot find class " << clname << endl;
      return obj;
   }

   if (gDebug>1)
     cout << "Reading object of class " << clname << endl;

   if (obj==0) obj = objClass->New();

   ExtractObjectId(objnode, obj);

   PushStack(objnode);

   objClass->Streamer((void*)obj, *this);

   PopStack();

   ShiftStack("readobj");

   if (gDebug>1)
     cout << "Read object of class " << clname << " done" << endl << endl;

   return obj;
}



//______________________________________________________________________________
void  TXMLBuffer::IncrementLevel(TStreamerInfo* info) {
   if (info==0) return;

   fCanUseCompact = kFALSE;
   fExpectedChain = kFALSE;

   TString clname = XmlConvertClassName(info->GetClass());

   if (gDebug>2)
     cout << " TXMLBuffer::StartInfo " << clname << endl;

   if (IsWriting()) {
      xmlNodePointer classnode = gXML->NewChild(StackNode(), 0, xmlNames_Class, 0);
      gXML->NewProp(classnode, 0, "name", clname);

      if (fVersionBuf>=0) {
         char sbuf[20];
         sprintf(sbuf,"%d", fVersionBuf);
         gXML->NewProp(classnode, 0, xmlNames_Version, sbuf);
         fVersionBuf = -111;
      }

      TXMLStackObj* stack = PushStack(classnode);
      stack->fInfo = info;
   } else {
      if (!VerifyNode(StackNode(), xmlNames_Class, "StartInfo")) return;
      if (!VerifyProp(StackNode(), "name", clname, "StartInfo")) return;

      TXMLStackObj* stack = PushStack(StackNode());
      stack->fInfo = info;

   }
}

//______________________________________________________________________________
void TXMLBuffer::CreateElemNode(const TStreamerElement* elem, Int_t number) {
    xmlNodePointer elemnode = 0;

    if (GetXmlLayout()==kGeneralized) {
      elemnode = gXML->NewChild(StackNode(), 0, xmlNames_Member, 0);
      gXML->NewProp(elemnode, 0, xmlNames_Name, elem->GetName());
      char sbuf[20];
      sprintf(sbuf,"%d", elem->GetType());
      gXML->NewProp(elemnode, 0, xmlNames_Type, sbuf);
    } else {
      
       elemnode = gXML->NewChild(StackNode(), 0, elem->GetName(), 0);
    }

    TXMLStackObj* curr = PushStack(elemnode);
    curr->fLastElem = (TStreamerElement*)elem;
    curr->fElemNumber = number;
}

//______________________________________________________________________________
Bool_t TXMLBuffer::VerifyElemNode(const TStreamerElement* elem, Int_t number) {
    if (GetXmlLayout()==kGeneralized) {
       if (!VerifyNode(StackNode(), xmlNames_Member)) return kFALSE;
       if (!VerifyProp(StackNode(), xmlNames_Name, elem->GetName())) return kFALSE;
       char sbuf[20];
       sprintf(sbuf,"%d", elem->GetType());
       if (!VerifyProp(StackNode(), xmlNames_Type, sbuf)) return kFALSE;
    } else {
       if (!VerifyNode(StackNode(), elem->GetName())) return kFALSE;
    }
       
    TXMLStackObj* curr = PushStack(StackNode()); // set pointer to first data inside element
    curr->fLastElem = (TStreamerElement*)elem;
    curr->fElemNumber = number;
    return kTRUE;
}



//______________________________________________________________________________
void TXMLBuffer::SetStreamerElementNumber(Int_t number) {
   CheckVersionBuf();

   TXMLStackObj* stack = dynamic_cast<TXMLStackObj*> (fStack.Last());
   if (stack==0) {
       cerr << "fatal error in SeStreamerElementNumber" << endl;
       return;
    }

    fExpectedChain = kFALSE;
    fCanUseCompact = kFALSE;

   if (IsWriting()) {

      if (stack->fInfo==0) {  // this is not a first element
         PopStack();
         stack = dynamic_cast<TXMLStackObj*> (fStack.Last());
      }

      TStreamerInfo* info = stack->fInfo;

      TStreamerElement* elem = info->GetStreamerElementReal(number, 0);

      Int_t comp_type = info->GetTypes()[number];

      Bool_t isBasicType = (elem->GetType()>0) && (elem->GetType()<20);
      
      fCanUseCompact = isBasicType && (elem->GetType()==comp_type);

      fExpectedChain = isBasicType && (comp_type - elem->GetType() == TStreamerInfo::kOffsetL);

      CreateElemNode(elem, number);

      if (fExpectedChain && (gDebug>3)) {
         cout << "Expects chain for class " << info->GetName()
              << " in elem " << elem->GetName() << " number " << number << endl;
      }
      
   } else {
       if (stack->fInfo==0) {  // this is not a first element
          PopStack();         // go level back
          ShiftStack("startelem");   // shift to next element
          stack = dynamic_cast<TXMLStackObj*> (fStack.Last());
       }

      TStreamerInfo* info = stack->fInfo;

      TStreamerElement* elem = info->GetStreamerElementReal(number, 0);
      
      Int_t comp_type = info->GetTypes()[number];

      Bool_t isBasicType = (elem->GetType()>0) && (elem->GetType()<20);

      fCanUseCompact = isBasicType && (elem->GetType()==comp_type);

      fExpectedChain = isBasicType && (comp_type - elem->GetType() == TStreamerInfo::kOffsetL);

      if (!VerifyElemNode(elem, number)) return;
   }
}


//______________________________________________________________________________
void  TXMLBuffer::StartElement(TStreamerElement* elem) {
   CheckVersionBuf();

   TXMLStackObj* stack = dynamic_cast<TXMLStackObj*> (fStack.Last());
   if (stack==0) {
       cerr << "Fatal error in StartElement" << endl;
       return;
    }

   TString elemname = elem->GetName();

   fCanUseCompact = (elem->GetType()>0) && (elem->GetType()<20);

   if (gDebug>2)
      cout << "   TXMLBuffer::StartElement " << elemname << endl;

   if (IsWriting()) {

      if (stack->fInfo==0)  // this is not a first element
        PopStack();
      xmlNodePointer elemnode = gXML->NewChild(StackNode(), 0, elemname , 0);

      TXMLStackObj* curr = PushStack(elemnode);
      curr->fLastElem = elem;
   } else {
       if (stack->fInfo==0) {  // this is not a first element
          PopStack();         // go level back
          ShiftStack("startelem");   // shift to next element
       }

      if (!VerifyNode(StackNode(), elemname, "StartElement")) return;

      TXMLStackObj* curr = PushStack(StackNode()); // set pointer to first data inside element
      curr->fLastElem = elem;
   }
}

//______________________________________________________________________________
void  TXMLBuffer::DecrementLevel(TStreamerInfo* info) {
   CheckVersionBuf();

   if (info==0) return;

   fCanUseCompact = kFALSE;
   fExpectedChain = kFALSE;

   //   cout << " <++++ " << info->GetName() << endl;

   if (gDebug>2)
      cout << " TXMLBuffer::StopInfo " << info->GetClass()->GetName() << endl;

   TXMLStackObj* stack = dynamic_cast<TXMLStackObj*> (fStack.Last());

   if (IsWriting()) {
      if (stack->fInfo==0) PopStack();  // remove stack of last element
      PopStack();
   } else {
      if (stack->fInfo==0) PopStack();  // back from data of last element
      PopStack();                       // back from data of stack info
      ShiftStack("declevel");                 // shift to next element after streamer info
   }
}


//______________________________________________________________________________
void TXMLBuffer::BeforeIOoperation() {
  // this function is called before any IO operation of TBuffer
   CheckVersionBuf();
}

