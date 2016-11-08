// @(#)root/:$Id: 5400e36954e1dc109fcfc306242c30234beb7312 $
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
// TBufferXML class uses streaming mechanism, provided by ROOT system,
// therefore most of ROOT and user classes can be stored to xml. There are
// limitations for complex objects like TTree, which can not be yet converted to xml.
//________________________________________________________________________


#include "TBufferXML.h"
#include "Compression.h"
#include "TXMLFile.h"

#include "TObjArray.h"
#include "TROOT.h"
#include "TClass.h"
#include "TClassTable.h"
#include "TDataType.h"
#include "TExMap.h"
#include "TMethodCall.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TProcessID.h"
#include "TFile.h"
#include "TMemberStreamer.h"
#include "TStreamer.h"
#include "TStreamerInfoActions.h"
#include "RZip.h"

#ifdef R__VISUAL_CPLUSPLUS
#define FLong64    "%I64d"
#define FULong64   "%I64u"
#else
#define FLong64    "%lld"
#define FULong64   "%llu"
#endif

ClassImp(TBufferXML);


const char* TBufferXML::fgFloatFmt = "%e";

//______________________________________________________________________________
TBufferXML::TBufferXML() :
   TBufferFile(),
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
   fCompressLevel(0),
   fIOVersion(3)
{
   // Default constructor
}

//______________________________________________________________________________
TBufferXML::TBufferXML(TBuffer::EMode mode) :
   TBufferFile(mode),
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
   fCompressLevel(0),
   fIOVersion(3)
{
   // Creates buffer object to serailize/deserialize data to/from xml.
   // Mode should be either TBuffer::kRead or TBuffer::kWrite.

   fBufSize = 1000000000;

   SetParent(0);
   SetBit(kCannotHandleMemberWiseStreaming);
   SetBit(kTextBasedStreaming);
}

//______________________________________________________________________________
TBufferXML::TBufferXML(TBuffer::EMode mode, TXMLFile* file) :
   TBufferFile(mode),
   TXMLSetup(*file),
   fXML(0),
   fStack(),
   fVersionBuf(-111),
   fObjMap(0),
   fIdArray(0),
   fErrorFlag(0),
   fCanUseCompact(kFALSE),
   fExpectedChain(kFALSE),
   fExpectedBaseClass(0),
   fCompressLevel(0),
   fIOVersion(3)
{
   // Creates buffer object to serailize/deserialize data to/from xml.
   // This constructor should be used, if data from buffer supposed to be stored in file.
   // Mode should be either TBuffer::kRead or TBuffer::kWrite.

   // this is for the case when StreamerInfo reads elements from
   // buffer as ReadFastArray. When it checks if size of buffer is
   // too small and skip reading. Actually, more improved method should
   // be used here.
   fBufSize = 1000000000;

   SetParent(file);
   SetBit(kCannotHandleMemberWiseStreaming);
   SetBit(kTextBasedStreaming);
   if (XmlFile()) {
      SetXML(XmlFile()->XML());
      SetCompressionSettings(XmlFile()->GetCompressionSettings());
      SetIOVersion(XmlFile()->GetIOVersion());
   }
}

//______________________________________________________________________________
TBufferXML::~TBufferXML()
{
   // destroy xml buffer

   if (fObjMap) delete fObjMap;
   if (fIdArray) delete fIdArray;
   fStack.Delete();
}

//______________________________________________________________________________
TXMLFile* TBufferXML::XmlFile()
{
   // returns pointer to TXMLFile object
   // access to file is necessary to produce unique identifier for object references

   return dynamic_cast<TXMLFile*>(GetParent());
}

//______________________________________________________________________________
TString TBufferXML::ConvertToXML(const TObject* obj, Bool_t GenericLayout, Bool_t UseNamespaces)
{
   // converts object, inherited from TObject class, to XML string
   // fmt contains configuration of XML layout. See TXMLSetup class for details

   TClass *clActual = 0;
   void *ptr = (void *) obj;

   if (obj!=0) {
      clActual = TObject::Class()->GetActualClass(obj);
      if (!clActual) clActual = TObject::Class(); else
      if (clActual != TObject::Class())
         ptr = (void *) ((Long_t) obj - clActual->GetBaseClassOffset(TObject::Class()));
   }

   return ConvertToXML(ptr, clActual, GenericLayout, UseNamespaces);
}

//______________________________________________________________________________
TString TBufferXML::ConvertToXML(const void* obj, const TClass* cl, Bool_t GenericLayout, Bool_t UseNamespaces)
{
   // converts any type of object to XML string
   // fmt contains configuration of XML layout. See TXMLSetup class for details

   TXMLEngine xml;

   TBufferXML buf(TBuffer::kWrite);
   buf.SetXML(&xml);

   buf.SetXmlLayout(GenericLayout ? TXMLSetup::kGeneralized : TXMLSetup::kSpecialized);
   buf.SetUseNamespaces(UseNamespaces);

   XMLNodePointer_t xmlnode = buf.XmlWriteAny(obj, cl);

   TString res;

   xml.SaveSingleNode(xmlnode, &res);

   xml.FreeNode(xmlnode);

   return res;
}

//______________________________________________________________________________
TObject* TBufferXML::ConvertFromXML(const char* str, Bool_t GenericLayout, Bool_t UseNamespaces)
{
   // Read object from XML, produced by ConvertToXML() method.
   // If object does not inherit from TObject class, return 0.
   // GenericLayout and UseNamespaces should be the same as in ConvertToXML()

   TClass* cl = 0;
   void* obj = ConvertFromXMLAny(str, &cl, GenericLayout, UseNamespaces);

   if ((cl==0) || (obj==0)) return 0;

   Int_t delta = cl->GetBaseClassOffset(TObject::Class());

   if (delta<0) {
      cl->Destructor(obj);
      return 0;
   }

   return (TObject*) ( ( (char*)obj ) + delta );
}

//______________________________________________________________________________
void* TBufferXML::ConvertFromXMLAny(const char* str, TClass** cl, Bool_t GenericLayout, Bool_t UseNamespaces)
{
   // Read object of any class from XML, produced by ConvertToXML() method.
   // If cl!=0, return actual class of object.
   // GenericLayout and UseNamespaces should be the same as in ConvertToXML()

   TXMLEngine xml;
   TBufferXML buf(TBuffer::kRead);

   buf.SetXML(&xml);

   buf.SetXmlLayout(GenericLayout ? TXMLSetup::kGeneralized : TXMLSetup::kSpecialized);
   buf.SetUseNamespaces(UseNamespaces);

   XMLNodePointer_t xmlnode = xml.ReadSingleNode(str);

   void* obj = buf.XmlReadAny(xmlnode, 0, cl);

   xml.FreeNode(xmlnode);

   return obj;
}

//______________________________________________________________________________
XMLNodePointer_t TBufferXML::XmlWriteAny(const void* obj, const TClass* cl)
{
   // Convert object of any class to xml structures
   // Return pointer on top xml element

   fErrorFlag = 0;

   if (fXML==0) return 0;

   XMLNodePointer_t res = XmlWriteObject(obj, cl);

   return res;
}

//______________________________________________________________________________
void* TBufferXML::XmlReadAny(XMLNodePointer_t node, void* obj, TClass** cl)
{
   // Recreate object from xml structure.
   // Return pointer to read object.
   // if (cl!=0) returns pointer to class of object

   if (node==0) return 0;
   if (cl) *cl = 0;

   fErrorFlag = 0;

   if (fXML==0) return 0;

   PushStack(node, kTRUE);

   void* res = XmlReadObject(obj, cl);

   PopStack();

   return res;
}

//______________________________________________________________________________
void TBufferXML::WriteObject(const TObject *obj)
{
   // Convert object into xml structures.
   // !!! Should be used only by TBufferXML itself.
   // Use ConvertToXML() methods to convert your object to xml
   // Redefined here to avoid gcc 3.x warning

   TBufferFile::WriteObject(obj);
}

// TXMLStackObj is used to keep stack of object hierarchy,
// stored in TBuffer. For instnace, data for parent class(es)
// stored in subnodes, but initial object node will be kept.

class TXMLStackObj : public TObject {
   public:
      TXMLStackObj(XMLNodePointer_t node) :
         TObject(),
         fNode(node),
         fInfo(0),
         fElem(0),
         fElemNumber(0),
         fCompressedClassNode(kFALSE),
         fClassNs(0),
         fIsStreamerInfo(kFALSE),
         fIsElemOwner(kFALSE)
          {}

      virtual ~TXMLStackObj()
      {
         if (fIsElemOwner) delete fElem;
      }

      Bool_t IsStreamerInfo() const { return fIsStreamerInfo; }

      XMLNodePointer_t  fNode;
      TStreamerInfo*    fInfo;
      TStreamerElement* fElem;
      Int_t             fElemNumber;
      Bool_t            fCompressedClassNode;
      XMLNsPointer_t    fClassNs;
      Bool_t            fIsStreamerInfo;
      Bool_t            fIsElemOwner;
};

//______________________________________________________________________________
TXMLStackObj* TBufferXML::PushStack(XMLNodePointer_t current, Bool_t simple)
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
TXMLStackObj* TBufferXML::PopStack()
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
TXMLStackObj* TBufferXML::Stack(Int_t depth)
{
   // return xml stack object of specified depth

   TXMLStackObj* stack = 0;
   if (depth<=fStack.GetLast())
      stack = dynamic_cast<TXMLStackObj*> (fStack.At(fStack.GetLast()-depth));
   return stack;
}

//______________________________________________________________________________
XMLNodePointer_t TBufferXML::StackNode()
{
   // return pointer on current xml node

   TXMLStackObj* stack = dynamic_cast<TXMLStackObj*> (fStack.Last());
   return (stack==0) ? 0 : stack->fNode;
}

//______________________________________________________________________________
void TBufferXML::ShiftStack(const char* errinfo)
{
   // shift stack node to next

   TXMLStackObj* stack = dynamic_cast<TXMLStackObj*> (fStack.Last());
   if (stack) {
      fXML->ShiftToNext(stack->fNode);
      if (gDebug>4) Info("ShiftStack","%s to node %s", errinfo, fXML->GetNodeName(stack->fNode));
   }
}

//______________________________________________________________________________
void TBufferXML::SetCompressionAlgorithm(Int_t algorithm)
{
   // See comments for function SetCompressionSettings
   if (algorithm < 0 || algorithm >= ROOT::kUndefinedCompressionAlgorithm) algorithm = 0;
   if (fCompressLevel < 0) {
      // if the level is not defined yet use 1 as a default
      fCompressLevel = 100 * algorithm + 1;
   } else {
      int level = fCompressLevel % 100;
      fCompressLevel = 100 * algorithm + level;
   }
}

//______________________________________________________________________________
void TBufferXML::SetCompressionLevel(Int_t level)
{
   // See comments for function SetCompressionSettings
   if (level < 0) level = 0;
   if (level > 99) level = 99;
   if (fCompressLevel < 0) {
      // if the algorithm is not defined yet use 0 as a default
      fCompressLevel = level;
   } else {
      int algorithm = fCompressLevel / 100;
      if (algorithm >= ROOT::kUndefinedCompressionAlgorithm) algorithm = 0;
      fCompressLevel = 100 * algorithm + level;
   }
}

//______________________________________________________________________________
void TBufferXML::SetCompressionSettings(Int_t settings)
{
   // Used to specify the compression level and algorithm:
   //  settings = 100 * algorithm + level
   //
   //  level = 0 no compression.
   //  level = 1 minimal compression level but fast.
   //  ....
   //  level = 9 maximal compression level but slower and might use more memory.
   // (For the currently supported algorithms, the maximum level is 9)
   // If compress is negative it indicates the compression level is not set yet.
   //
   // The enumeration ROOT::ECompressionAlgorithm associates each
   // algorithm with a number. There is a utility function to help
   // to set the value of compress. For example,
   //   ROOT::CompressionSettings(ROOT::kLZMA, 1)
   // will build an integer which will set the compression to use
   // the LZMA algorithm and compression level 1.  These are defined
   // in the header file Compression.h.

   fCompressLevel = settings;
}

//______________________________________________________________________________
void TBufferXML::XmlWriteBlock(XMLNodePointer_t node)
{
   // write binary data block from buffer to xml
   // this data can be produced only by direct call of TBuffer::WriteBuf() functions

   if ((node==0) || (Length()==0)) return;

   const char* src = Buffer();
   int srcSize = Length();

   char* fZipBuffer = 0;

   Int_t compressionLevel = GetCompressionLevel();
   Int_t compressionAlgorithm = GetCompressionAlgorithm();

   if ((Length() > 512) && (compressionLevel > 0)) {
      int zipBufferSize = Length();
      fZipBuffer = new char[zipBufferSize + 9];
      int dataSize = Length();
      int compressedSize = 0;
      R__zipMultipleAlgorithm(compressionLevel, &dataSize, Buffer(), &zipBufferSize,
                              fZipBuffer, &compressedSize, compressionAlgorithm);
      if (compressedSize > 0) {
        src = fZipBuffer;
        srcSize = compressedSize;
      } else {
        delete[] fZipBuffer;
        fZipBuffer = 0;
      }
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

   XMLNodePointer_t blocknode = fXML->NewChild(node, 0, xmlio::XmlBlock, res);
   fXML->NewIntAttr(blocknode, xmlio::Size, Length());

   if (fZipBuffer) {
      fXML->NewIntAttr(blocknode, xmlio::Zip, srcSize);
      delete[] fZipBuffer;
   }
}

//______________________________________________________________________________
void TBufferXML::XmlReadBlock(XMLNodePointer_t blocknode)
{
   // read binary block of data from xml

   if (blocknode==0) return;

   Int_t blockSize = fXML->GetIntAttr(blocknode, xmlio::Size);
   Bool_t blockCompressed = fXML->HasAttr(blocknode, xmlio::Zip);
   char* fUnzipBuffer = 0;

   if (gDebug>2)
      Info("XmlReadBlock","Block size = %d, Length = %d, Compressed = %d",
                           blockSize, Length(), blockCompressed);

   if (blockSize>BufferSize()) Expand(blockSize);

   char* tgt = Buffer();
   Int_t readSize = blockSize;

   TString content = fXML->GetNodeContent(blocknode);

   if (blockCompressed) {
      Int_t zipSize = fXML->GetIntAttr(blocknode, xmlio::Zip);
      fUnzipBuffer = new char[zipSize];

      tgt = fUnzipBuffer;
      readSize = zipSize;
   }

   char* ptr = (char*) content.Data();

   if (gDebug>3)
      Info("XmlReadBlock","Content %s", ptr);

   for (int i=0;i<readSize;i++) {
      while ((*ptr<48) || ((*ptr>57) && (*ptr<97)) || (*ptr>102)) ptr++;

      int b_hi = (*ptr>57) ? *ptr-87 : *ptr-48;
      ptr++;
      int b_lo = (*ptr>57) ? *ptr-87 : *ptr-48;
      ptr++;

      *tgt=b_hi*16+b_lo;
      tgt++;

      if (gDebug>4) Info("XmlReadBlock","    Buf[%d] = %d", i, b_hi*16+b_lo);
   }

   if (fUnzipBuffer) {

      int srcsize;
      int tgtsize;
      int status = R__unzip_header(&srcsize, (UChar_t*) fUnzipBuffer, &tgtsize);

      int unzipRes = 0;
      if (status == 0) {
        R__unzip(&readSize, (unsigned char*) fUnzipBuffer, &blockSize,
                            (unsigned char*) Buffer(), &unzipRes);
      }
      if (status != 0 || unzipRes!=blockSize)
         Error("XmlReadBlock", "Decompression error %d", unzipRes);
      else
         if (gDebug>2) Info("XmlReadBlock","Unzip ok");
      delete[] fUnzipBuffer;
   }
}

//______________________________________________________________________________
Bool_t TBufferXML::ProcessPointer(const void* ptr, XMLNodePointer_t node)
{
   // Add "ptr" attribute to node, if ptr is null or
   // if ptr is pointer on object, which is already saved in buffer
   // Automatically add "ref" attribute to node, where referenced object is stored

   if (node==0) return kFALSE;

   TString refvalue;

   if (ptr==0)
      refvalue = xmlio::Null;   //null
   else {
      if (fObjMap==0) return kFALSE;

      ULong_t hash = TString::Hash(&ptr, sizeof(void*));

      XMLNodePointer_t refnode = (XMLNodePointer_t) (Long_t)fObjMap->GetValue(hash, (Long_t) ptr);
      if (refnode==0) return kFALSE;

      if (fXML->HasAttr(refnode, xmlio::Ref))
         refvalue = fXML->GetAttr(refnode, xmlio::Ref);
      else {
         refvalue = xmlio::IdBase;
         if (XmlFile())
            refvalue += XmlFile()->GetNextRefCounter();
         else
            refvalue += GetNextRefCounter();
         fXML->NewAttr(refnode, 0, xmlio::Ref, refvalue.Data());
      }
   }
   if (refvalue.Length()>0) {
      fXML->NewAttr(node, 0, xmlio::Ptr, refvalue.Data());
      return kTRUE;
   }

   return kFALSE;
}

//______________________________________________________________________________
void TBufferXML::RegisterPointer(const void* ptr, XMLNodePointer_t node)
{
   // Register pair of object pointer and node, where this object is saved,
   // in object map

   if ((node==0) || (ptr==0)) return;

   ULong_t hash = TString::Hash(&ptr, sizeof(void*));

   if (fObjMap==0) fObjMap = new TExMap();

   if (fObjMap->GetValue(hash, (Long_t) ptr)==0)
      fObjMap->Add(hash, (Long_t) ptr, (Long_t) node);
}

//______________________________________________________________________________
Bool_t TBufferXML::ExtractPointer(XMLNodePointer_t node, void* &ptr, TClass* &cl)
{
   // Searches for "ptr" attribute and returns pointer to object and class,
   // if "ptr" attribute reference to read object

   cl = 0;

   if (!fXML->HasAttr(node,xmlio::Ptr)) return kFALSE;

   const char* ptrid = fXML->GetAttr(node, xmlio::Ptr);

   if (ptrid==0) return kFALSE;

   // null
   if (strcmp(ptrid, xmlio::Null)==0) {
      ptr = 0;
      return kTRUE;
   }

   if ((fIdArray==0) || (fObjMap==0)) return kFALSE;

   TNamed* obj = (TNamed*) fIdArray->FindObject(ptrid);
   if (obj) {
      ptr = (void*) (Long_t)fObjMap->GetValue((Long_t) fIdArray->IndexOf(obj));
      cl = TClass::GetClass(obj->GetTitle());
      return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
void TBufferXML::ExtractReference(XMLNodePointer_t node, const void* ptr, const TClass* cl)
{
   // Analyse, if node has "ref" attribute and register it to object map

   if ((node==0) || (ptr==0)) return;

   const char* refid = fXML->GetAttr(node, xmlio::Ref);

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
      Info("ExtractReference","Find reference %s for object %p", refid, ptr);
}

//______________________________________________________________________________
Bool_t TBufferXML::VerifyNode(XMLNodePointer_t node, const char* name, const char* errinfo)
{
   // check, if node has specified name

   if ((name==0) || (node==0)) return kFALSE;

   if (strcmp(fXML->GetNodeName(node), name)!=0) {
      if (errinfo) {
         Error("VerifyNode","Reading XML file (%s). Get: %s, expects: %s",
                errinfo, fXML->GetNodeName(node), name);
         fErrorFlag = 1;
      }
      return kFALSE;
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TBufferXML::VerifyStackNode(const char* name, const char* errinfo)
{
   // check, if stack node has specified name

   return VerifyNode(StackNode(), name, errinfo);
}


//______________________________________________________________________________
Bool_t TBufferXML::VerifyAttr(XMLNodePointer_t node, const char* name, const char* value, const char* errinfo)
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
Bool_t TBufferXML::VerifyStackAttr(const char* name, const char* value, const char* errinfo)
{
   // checks stack attribute

   return VerifyAttr(StackNode(), name, value, errinfo);
}

//______________________________________________________________________________
XMLNodePointer_t TBufferXML::CreateItemNode(const char* name)
{
   // create item node of specified name

   XMLNodePointer_t node = 0;
   if (GetXmlLayout()==kGeneralized) {
      node = fXML->NewChild(StackNode(), 0, xmlio::Item, 0);
      fXML->NewAttr(node, 0, xmlio::Name, name);
   } else
      node = fXML->NewChild(StackNode(), 0, name, 0);
   return node;
}

//______________________________________________________________________________
Bool_t TBufferXML::VerifyItemNode(const char* name, const char* errinfo)
{
   // checks, if stack node is item and has specified name

   Bool_t res = kTRUE;
   if (GetXmlLayout()==kGeneralized)
      res = VerifyStackNode(xmlio::Item, errinfo) &&
            VerifyStackAttr(xmlio::Name, name, errinfo);
   else
      res = VerifyStackNode(name, errinfo);
   return res;
}

//______________________________________________________________________________
void TBufferXML::CreateElemNode(const TStreamerElement* elem)
{
   // create xml node correspondent to TStreamerElement object

   XMLNodePointer_t elemnode = 0;

   const char* elemxmlname = XmlGetElementName(elem);

   if (GetXmlLayout()==kGeneralized) {
      elemnode = fXML->NewChild(StackNode(), 0, xmlio::Member, 0);
      fXML->NewAttr(elemnode, 0, xmlio::Name, elemxmlname);
   } else {
      // take namesapce for element only if it is not a base class or class name
      XMLNsPointer_t ns = Stack()->fClassNs;
      if ((elem->GetType()==TStreamerInfo::kBase)
           || ((elem->GetType()==TStreamerInfo::kTNamed) && !strcmp(elem->GetName(), TNamed::Class()->GetName()))
           || ((elem->GetType()==TStreamerInfo::kTObject) && !strcmp(elem->GetName(), TObject::Class()->GetName()))
           || ((elem->GetType()==TStreamerInfo::kTString) && !strcmp(elem->GetName(), TString::Class()->GetName())))
         ns = 0;

      elemnode = fXML->NewChild(StackNode(), ns, elemxmlname, 0);
   }

   TXMLStackObj* curr = PushStack(elemnode);
   curr->fElem = (TStreamerElement*)elem;
}

//______________________________________________________________________________
Bool_t TBufferXML::VerifyElemNode(const TStreamerElement* elem)
{
   // Checks, if stack node correspond to TStreamerElement object

   const char* elemxmlname = XmlGetElementName(elem);

   if (GetXmlLayout()==kGeneralized) {
      if (!VerifyStackNode(xmlio::Member)) return kFALSE;
      if (!VerifyStackAttr(xmlio::Name, elemxmlname)) return kFALSE;
   } else {
      if (!VerifyStackNode(elemxmlname)) return kFALSE;
   }

   PerformPreProcessing(elem, StackNode());

   TXMLStackObj* curr = PushStack(StackNode()); // set pointer to first data inside element
   curr->fElem = (TStreamerElement*)elem;
   return kTRUE;
}

//______________________________________________________________________________
XMLNodePointer_t TBufferXML::XmlWriteObject(const void* obj, const TClass* cl)
{
   // Write object to buffer
   // If object was written before, only pointer will be stored
   // Return pointer to top xml node, representing object

   XMLNodePointer_t objnode = fXML->NewChild(StackNode(), 0, xmlio::Object, 0);

   if (!cl) obj = 0;
   if (ProcessPointer(obj, objnode)) return objnode;

   TString clname = XmlConvertClassName(cl->GetName());

   fXML->NewAttr(objnode, 0, xmlio::ObjClass, clname);

   RegisterPointer(obj, objnode);

   PushStack(objnode);

   ((TClass*)cl)->Streamer((void*)obj, *this);

   PopStack();

   if (gDebug>1)
      Info("XmlWriteObject","Done write for class: %s", cl ? cl->GetName() : "null");

   return objnode;
}

//______________________________________________________________________________
void* TBufferXML::XmlReadObject(void* obj, TClass** cl)
{
   // Read object from the buffer

   if (cl) *cl = 0;

   XMLNodePointer_t objnode = StackNode();

   if (fErrorFlag>0) return obj;

   if (objnode==0) return obj;

   if (!VerifyNode(objnode, xmlio::Object, "XmlReadObjectNew")) return obj;

   TClass* objClass = 0;

   if (ExtractPointer(objnode, obj, objClass)) {
      ShiftStack("readobjptr");
      if (cl) *cl = objClass;
      return obj;
   }

   TString clname = fXML->GetAttr(objnode, xmlio::ObjClass);
   objClass = XmlDefineClass(clname);
   if (objClass == TDirectory::Class()) objClass = TDirectoryFile::Class();

   if (objClass==0) {
      Error("XmlReadObject", "Cannot find class %s", clname.Data());
      ShiftStack("readobjerr");
      return obj;
   }

   if (gDebug>1)
      Info("XmlReadObject", "Reading object of class %s", clname.Data());

   if (obj==0) obj = objClass->New();

   ExtractReference(objnode, obj, objClass);

   PushStack(objnode);

   objClass->Streamer((void*)obj, *this);

   PopStack();

   ShiftStack("readobj");

   if (gDebug>1)
      Info("XmlReadObject", "Reading object of class %s done", clname.Data());

   if (cl) *cl = objClass;

   return obj;
}

//______________________________________________________________________________
void TBufferXML::IncrementLevel(TVirtualStreamerInfo* info)
{
   // Function is called from TStreamerInfo WriteBuffer and Readbuffer functions
   // and indent new level in xml structure.
   // This call indicates, that TStreamerInfo functions starts streaming
   // object data of correspondent class

   WorkWithClass((TStreamerInfo*)info);
}

//______________________________________________________________________________
void  TBufferXML::WorkWithClass(TStreamerInfo* sinfo, const TClass* cl)
{
   // Prepares buffer to stream data of specified class

   fCanUseCompact = kFALSE;
   fExpectedChain = kFALSE;

   if (sinfo!=0) cl = sinfo->GetClass();

   if (cl==0) return;

   TString clname = XmlConvertClassName(cl->GetName());

   if (gDebug>2) Info("IncrementLevel","Class: %s", clname.Data());

   Bool_t compressClassNode = fExpectedBaseClass==cl;
   fExpectedBaseClass = 0;

   TXMLStackObj* stack = Stack();

   if (IsWriting()) {

      XMLNodePointer_t classnode = 0;
      if (compressClassNode) {
         classnode = StackNode();
      } else {
         if (GetXmlLayout()==kGeneralized) {
            classnode = fXML->NewChild(StackNode(), 0, xmlio::Class, 0);
            fXML->NewAttr(classnode, 0, "name", clname);
         } else
            classnode = fXML->NewChild(StackNode(), 0, clname, 0);
         stack = PushStack(classnode);
      }

      if (fVersionBuf>=-1) {
         if (fVersionBuf == -1) fVersionBuf = 1;
         fXML->NewIntAttr(classnode, xmlio::ClassVersion, fVersionBuf);
         fVersionBuf = -111;
      }

      if (IsUseNamespaces() && (GetXmlLayout()!=kGeneralized))
         stack->fClassNs = fXML->NewNS(classnode, XmlClassNameSpaceRef(cl), clname);

   } else {
      if (!compressClassNode) {
         if (GetXmlLayout()==kGeneralized) {
            if (!VerifyStackNode(xmlio::Class, "StartInfo")) return;
            if (!VerifyStackAttr("name", clname, "StartInfo")) return;
         } else
            if (!VerifyStackNode(clname, "StartInfo")) return;
         stack = PushStack(StackNode());
      }
   }

   stack->fCompressedClassNode = compressClassNode;
   stack->fInfo = sinfo;
   stack->fIsStreamerInfo = kTRUE;
}

//______________________________________________________________________________
void TBufferXML::DecrementLevel(TVirtualStreamerInfo* info)
{
   // Function is called from TStreamerInfo WriteBuffer and Readbuffer functions
   // and decrease level in xml structure.

   CheckVersionBuf();

   fCanUseCompact = kFALSE;
   fExpectedChain = kFALSE;

   if (gDebug>2)
      Info("DecrementLevel","Class: %s", (info ? info->GetClass()->GetName() : "custom"));

   TXMLStackObj* stack = Stack();

   if (!stack->IsStreamerInfo()) {
      PerformPostProcessing();
      stack = PopStack();  // remove stack of last element
   }

   if (stack->fCompressedClassNode) {
      stack->fInfo = 0;
      stack->fIsStreamerInfo = kFALSE;
      stack->fCompressedClassNode = kFALSE;
   } else {
      PopStack();                       // back from data of stack info
      if (IsReading()) ShiftStack("declevel"); // shift to next element after streamer info
   }
}

//______________________________________________________________________________
void TBufferXML::SetStreamerElementNumber(TStreamerElement *elem, Int_t comptype)
{
   // Function is called from TStreamerInfo WriteBuffer and Readbuffer functions
   // and add/verify next element of xml structure
   // This calls allows separate data, correspondent to one class member, from another

   WorkWithElement(elem, comptype);
}

//______________________________________________________________________________
void TBufferXML::WorkWithElement(TStreamerElement* elem, Int_t comp_type)
{
   // This function is a part of SetStreamerElementNumber method.
   // It is introduced for reading of data for specified data memeber of class.
   // Used also in ReadFastArray methods to resolve problem of compressed data,
   // when several data memebers of the same basic type streamed with single ...FastArray call

   CheckVersionBuf();

   fExpectedChain = kFALSE;
   fCanUseCompact = kFALSE;
   fExpectedBaseClass = 0;

   TXMLStackObj* stack = Stack();
   if (stack==0) {
      Error("SetStreamerElementNumber", "stack is empty");
      return;
   }

   if (!stack->IsStreamerInfo()) {  // this is not a first element
      PerformPostProcessing();
      PopStack();           // go level back
      if (IsReading()) ShiftStack("startelem");   // shift to next element, only for reading
      stack = dynamic_cast<TXMLStackObj*> (fStack.Last());
   }

   if (stack==0) {
      Error("SetStreamerElementNumber", "Lost of stack");
      return;
   }

   if (!elem) {
      Error("SetStreamerElementNumber", "Problem in Inc/Dec level");
      return;
   }

   TStreamerInfo* info = stack->fInfo;

   if (!stack->IsStreamerInfo()) {
      Error("SetStreamerElementNumber", "Problem in Inc/Dec level");
      return;
   }
   Int_t number = info ? info->GetElements()->IndexOf(elem) : -1;

   if (gDebug>4) Info("SetStreamerElementNumber", "    Next element %s", elem->GetName());

   Bool_t isBasicType = (elem->GetType()>0) && (elem->GetType()<20);

   fExpectedChain = isBasicType && (comp_type - elem->GetType() == TStreamerInfo::kOffsetL);

   if (fExpectedChain && (gDebug>3)) {
      Info("SetStreamerElementNumber",
           "    Expects chain for elem %s number %d",
            elem->GetName(), number);
   }

   fCanUseCompact = isBasicType && ((elem->GetType()==comp_type) ||
                                    (elem->GetType()==comp_type-TStreamerInfo::kConv) ||
                                    (elem->GetType()==comp_type-TStreamerInfo::kSkip));

   if ((elem->GetType()==TStreamerInfo::kBase) ||
       ((elem->GetType()==TStreamerInfo::kTNamed) && !strcmp(elem->GetName(), TNamed::Class()->GetName())))
      fExpectedBaseClass = elem->GetClassPointer();

   if (fExpectedBaseClass && (gDebug>3))
      Info("SetStreamerElementNumber",
           "   Expects base class %s with standard streamer",
               fExpectedBaseClass->GetName());

   if (IsWriting()) {
      CreateElemNode(elem);
   } else {
      if (!VerifyElemNode(elem)) return;
   }

   stack = Stack();
   stack->fElemNumber = number;
   stack->fIsElemOwner = (number<0);
}

//______________________________________________________________________________
void TBufferXML::ClassBegin(const TClass* cl, Version_t)
{
   // Should be called in the beginning of custom class streamer.
   // Informs buffer data about class which will be streamed now.
   //
   // ClassBegin(), ClassEnd() and ClassMemeber() should be used in
   // custom class streamers to specify which kind of data are
   // now streamed. Such information is used to correctly
   // convert class data to XML. Without that functions calls
   // classes with custom streamers cannot be used with TBufferXML

   WorkWithClass(0, cl);
}

//______________________________________________________________________________
void TBufferXML::ClassEnd(const TClass*)
{
   // Should be called at the end of custom streamer
   // See TBufferXML::ClassBegin for more details

   DecrementLevel(0);
}

//______________________________________________________________________________
void TBufferXML::ClassMember(const char* name, const char* typeName, Int_t arrsize1, Int_t arrsize2)
{
   // Method indicates name and typename of class member,
   // which should be now streamed in custom streamer
   // Following combinations are supported:
   // 1. name = "ClassName", typeName = 0 or typename==ClassName
   //    This is a case, when data of parent class "ClassName" should be streamed.
   //     For instance, if class directly inherited from TObject, custom
   //     streamer should include following code:
   //       b.ClassMember("TObject");
   //       TObject::Streamer(b);
   // 2. Basic data type
   //      b.ClassMember("fInt","Int_t");
   //      b >> fInt;
   // 3. Array of basic data types
   //      b.ClassMember("fArr","Int_t", 5);
   //      b.ReadFastArray(fArr, 5);
   // 4. Object as data member
   //      b.ClassMemeber("fName","TString");
   //      fName.Streamer(b);
   // 5. Pointer on object as data member
   //      b.ClassMemeber("fObj","TObject*");
   //      b.StreamObject(fObj);
   //  arrsize1 and arrsize2 arguments (when specified) indicate first and
   //  second dimension of array. Can be used for array of basic types.
   //  See ClassBegin() method for more details.

   if (typeName==0) typeName = name;

   if ((name==0) || (strlen(name)==0)) {
      Error("ClassMember","Invalid member name");
      fErrorFlag = 1;
      return;
   }

   TString tname = typeName;

   Int_t typ_id(-1), comp_type(-1);

   if (strcmp(typeName,"raw:data")==0)
      typ_id = TStreamerInfo::kMissing;

   if (typ_id<0) {
      TDataType *dt = gROOT->GetType(typeName);
      if (dt!=0)
         if ((dt->GetType()>0) && (dt->GetType()<20))
            typ_id = dt->GetType();
   }

   if (typ_id<0)
      if (strcmp(name, typeName)==0) {
         TClass* cl = TClass::GetClass(tname.Data());
         if (cl!=0) typ_id = TStreamerInfo::kBase;
      }

   if (typ_id<0) {
      Bool_t isptr = kFALSE;
      if (tname[tname.Length()-1]=='*') {
         tname.Resize(tname.Length()-1);
         isptr = kTRUE;
      }
      TClass* cl = TClass::GetClass(tname.Data());
      if (cl==0) {
         Error("ClassMember","Invalid class specifier %s", typeName);
         fErrorFlag = 1;
         return;
      }

      if (cl->IsTObject())
         typ_id = isptr ? TStreamerInfo::kObjectp : TStreamerInfo::kObject;
      else
         typ_id = isptr ? TStreamerInfo::kAnyp : TStreamerInfo::kAny;

      if ((cl==TString::Class()) && !isptr)
         typ_id = TStreamerInfo::kTString;
   }

   TStreamerElement* elem = 0;

   if (typ_id == TStreamerInfo::kMissing) {
      elem = new TStreamerElement(name,"title",0, typ_id, "raw:data");
   } else

   if (typ_id==TStreamerInfo::kBase) {
      TClass* cl = TClass::GetClass(tname.Data());
      if (cl!=0) {
         TStreamerBase* b = new TStreamerBase(tname.Data(), "title", 0);
         b->SetBaseVersion(cl->GetClassVersion());
         elem = b;
      }
   } else

   if ((typ_id>0) && (typ_id<20)) {
      elem = new TStreamerBasicType(name, "title", 0, typ_id, typeName);
      comp_type = typ_id;
   } else

   if ((typ_id==TStreamerInfo::kObject) ||
       (typ_id==TStreamerInfo::kTObject) ||
       (typ_id==TStreamerInfo::kTNamed)) {
      elem = new TStreamerObject(name, "title", 0, tname.Data());
   } else

   if (typ_id==TStreamerInfo::kObjectp) {
      elem = new TStreamerObjectPointer(name, "title", 0, tname.Data());
   } else

   if (typ_id==TStreamerInfo::kAny) {
      elem = new TStreamerObjectAny(name, "title", 0, tname.Data());
   } else

   if (typ_id==TStreamerInfo::kAnyp) {
      elem = new TStreamerObjectAnyPointer(name, "title", 0, tname.Data());
   } else

   if (typ_id==TStreamerInfo::kTString) {
      elem = new TStreamerString(name, "title", 0);
   }

   if (elem==0) {
      Error("ClassMember","Invalid combination name = %s type = %s", name, typeName);
      fErrorFlag = 1;
      return;
   }

   if (arrsize1>0) {
      elem->SetArrayDim(arrsize2>0 ? 2 : 1);
      elem->SetMaxIndex(0, arrsize1);
      if (arrsize2>0)
         elem->SetMaxIndex(1, arrsize2);
   }

   // we indicate that there is no streamerinfo
   WorkWithElement(elem, comp_type);
}

//______________________________________________________________________________
void TBufferXML::PerformPostProcessing()
{
   // Function is converts TObject and TString structures to more compact representation

   if (GetXmlLayout()==kGeneralized) return;

   const TStreamerElement* elem = Stack()->fElem;
   XMLNodePointer_t elemnode = IsWriting() ? Stack()->fNode : Stack(1)->fNode;

   if ((elem==0) || (elemnode==0)) return;

   if (elem->GetType()==TStreamerInfo::kTString)  {

      XMLNodePointer_t node = fXML->GetChild(elemnode);
      fXML->SkipEmpty(node);

      XMLNodePointer_t nodecharstar(0), nodeuchar(0), nodeint(0), nodestring(0);

      while (node!=0) {
         const char* name = fXML->GetNodeName(node);
         if (strcmp(name, xmlio::String)==0) {
            if (nodestring) return;
            nodestring = node;
         } else
         if (strcmp(name, xmlio::UChar)==0) {
            if (nodeuchar) return;
            nodeuchar = node;
         } else
         if (strcmp(name, xmlio::Int)==0) {
            if (nodeint) return;
            nodeint = node;
         } else
         if (strcmp(name, xmlio::CharStar)==0) {
            if (nodecharstar!=0) return;
            nodecharstar = node;
         } else return; // can not be something else
         fXML->ShiftToNext(node);
      }

      TString str;

      if (GetIOVersion()<3) {
         if (nodeuchar==0) return;
         if (nodecharstar!=0)
            str = fXML->GetAttr(nodecharstar, xmlio::v);
	 fXML->UnlinkFreeNode(nodeuchar);
         fXML->UnlinkFreeNode(nodeint);
         fXML->UnlinkFreeNode(nodecharstar);
      } else {
         if (nodestring!=0)
            str = fXML->GetAttr(nodestring, xmlio::v);
	 fXML->UnlinkFreeNode(nodestring);
      }

      fXML->NewAttr(elemnode, 0, "str", str);
   } else
   if (elem->GetType()==TStreamerInfo::kTObject) {
      XMLNodePointer_t node = fXML->GetChild(elemnode);
      fXML->SkipEmpty(node);

      XMLNodePointer_t vnode = 0;
      XMLNodePointer_t idnode = 0;
      XMLNodePointer_t bitsnode = 0;
      XMLNodePointer_t prnode = 0;
      while (node!=0) {
         const char* name = fXML->GetNodeName(node);

         if (strcmp(name, xmlio::OnlyVersion)==0) {
            if (vnode) return;
            vnode = node;
         } else
         if (strcmp(name, xmlio::UInt)==0) {
            if (idnode==0) idnode = node; else
            if (bitsnode==0) bitsnode = node; else return;
         } else
         if (strcmp(name, xmlio::UShort)==0) {
            if (prnode) return;
            prnode = node;
         } else return;
         fXML->ShiftToNext(node);
      }

      if ((vnode==0) || (idnode==0) || (bitsnode==0)) return;

      TString str = fXML->GetAttr(idnode,xmlio::v);
      fXML->NewAttr(elemnode, 0, "fUniqueID", str);

      str = fXML->GetAttr(bitsnode, xmlio::v);
      UInt_t bits;
      sscanf(str.Data(),"%u", &bits);

      char sbuf[20];
      snprintf(sbuf, sizeof(sbuf), "%x",bits);
      fXML->NewAttr(elemnode, 0, "fBits", sbuf);

      if (prnode!=0) {
         str = fXML->GetAttr(prnode,xmlio::v);
         fXML->NewAttr(elemnode, 0, "fProcessID", str);
      }

      fXML->UnlinkFreeNode(vnode);
      fXML->UnlinkFreeNode(idnode);
      fXML->UnlinkFreeNode(bitsnode);
      fXML->UnlinkFreeNode(prnode);
   }
}

//______________________________________________________________________________
void TBufferXML::PerformPreProcessing(const TStreamerElement* elem, XMLNodePointer_t elemnode)
{
   // Function is unpack TObject and TString structures to be able read
   // them from custom streamers of this objects

   if (GetXmlLayout()==kGeneralized) return;
   if ((elem==0) || (elemnode==0)) return;

   if (elem->GetType()==TStreamerInfo::kTString) {

      if (!fXML->HasAttr(elemnode,"str")) return;
      TString str = fXML->GetAttr(elemnode, "str");
      fXML->FreeAttr(elemnode, "str");

      if (GetIOVersion()<3) {
         Int_t len = str.Length();
         XMLNodePointer_t ucharnode = fXML->NewChild(elemnode, 0, xmlio::UChar,0);
         char sbuf[20];
         snprintf(sbuf, sizeof(sbuf), "%d", len);
         if (len<255) {
            fXML->NewAttr(ucharnode,0,xmlio::v,sbuf);
	 } else {
            fXML->NewAttr(ucharnode,0,xmlio::v,"255");
            XMLNodePointer_t intnode = fXML->NewChild(elemnode, 0, xmlio::Int, 0);
            fXML->NewAttr(intnode, 0, xmlio::v, sbuf);
         }
         if (len>0) {
            XMLNodePointer_t node = fXML->NewChild(elemnode, 0, xmlio::CharStar, 0);
            fXML->NewAttr(node, 0, xmlio::v, str);
         }
      } else {
         XMLNodePointer_t node = fXML->NewChild(elemnode, 0, xmlio::String, 0);
         fXML->NewAttr(node, 0, xmlio::v, str);
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

      XMLNodePointer_t node = fXML->NewChild(elemnode, 0, xmlio::OnlyVersion, 0);
      fXML->NewAttr(node, 0, xmlio::v, "1");

      node = fXML->NewChild(elemnode, 0, xmlio::UInt, 0);
      fXML->NewAttr(node, 0, xmlio::v, idstr);

      UInt_t bits;
      sscanf(bitsstr.Data(),"%x", &bits);
      char sbuf[20];
      snprintf(sbuf, sizeof(sbuf), "%u", bits);

      node = fXML->NewChild(elemnode, 0, xmlio::UInt, 0);
      fXML->NewAttr(node, 0, xmlio::v, sbuf);

      if (prstr.Length()>0) {
         node = fXML->NewChild(elemnode, 0, xmlio::UShort, 0);
         fXML->NewAttr(node, 0, xmlio::v, prstr.Data());
      }
   }
}

//______________________________________________________________________________
void TBufferXML::BeforeIOoperation()
{
  // Function is called before any IO operation of TBuffer
  // Now is used to store version value if no proper calls are discovered

   CheckVersionBuf();
}

//______________________________________________________________________________
TClass* TBufferXML::ReadClass(const TClass*, UInt_t*)
{
   // function to read class from buffer, used in old-style streamers

   const char* clname = 0;

   if (VerifyItemNode(xmlio::Class)) {
      clname = XmlReadValue(xmlio::Class);
   }

   if (gDebug>2) Info("ReadClass", "Try to read class %s", clname ? clname : "---");

   return clname ? gROOT->GetClass(clname) : 0;
}

//______________________________________________________________________________
void TBufferXML::WriteClass(const TClass* cl)
{
   // function to write class into buffer, used in old-style streamers

   if (gDebug>2) Info("WriteClass", "Try to write class %s", cl->GetName());

   XmlWriteValue(cl->GetName(), xmlio::Class);
}

//______________________________________________________________________________
Int_t TBufferXML::CheckByteCount(UInt_t /*r_s */, UInt_t /*r_c*/, const TClass* /*cl*/)
{
   // suppressed function of TBuffer

   return 0;
}

//______________________________________________________________________________
Int_t  TBufferXML::CheckByteCount(UInt_t, UInt_t, const char*)
{
   // suppressed function of TBuffer

   return 0;
}

//______________________________________________________________________________
void TBufferXML::SetByteCount(UInt_t, Bool_t)
{
   // suppressed function of TBuffer

}

//______________________________________________________________________________
void TBufferXML::SkipVersion(const TClass *cl)
{
   // Skip class version from I/O buffer.
   ReadVersion(0,0,cl);
}

//______________________________________________________________________________
Version_t TBufferXML::ReadVersion(UInt_t *start, UInt_t *bcnt, const TClass * /*cl*/)
{
   // read version value from buffer

   BeforeIOoperation();

   Version_t res = 0;

   if (start) *start = 0;
   if (bcnt) *bcnt = 0;

   if (VerifyItemNode(xmlio::OnlyVersion)) {
      res = AtoI(XmlReadValue(xmlio::OnlyVersion));
   } else
   if ((fExpectedBaseClass!=0) && (fXML->HasAttr(Stack(1)->fNode, xmlio::ClassVersion))) {
      res = fXML->GetIntAttr(Stack(1)->fNode, xmlio::ClassVersion);
   } else
   if (fXML->HasAttr(StackNode(), xmlio::ClassVersion)) {
      res = fXML->GetIntAttr(StackNode(), xmlio::ClassVersion);
   } else {
      Error("ReadVersion", "No correspondent tags to read version");;
      fErrorFlag = 1;
   }

   if (gDebug>2) Info("ReadVersion","Version = %d", res);

   return res;
}

//______________________________________________________________________________
void TBufferXML::CheckVersionBuf()
{
   // checks buffer, filled by WriteVersion
   // if next data is arriving, version should be stored in buffer

   if (IsWriting() && (fVersionBuf>=-100)) {
      char sbuf[20];
      snprintf(sbuf, sizeof(sbuf), "%d", fVersionBuf);
      XmlWriteValue(sbuf, xmlio::OnlyVersion);
      fVersionBuf = -111;
   }
}

//______________________________________________________________________________
UInt_t TBufferXML::WriteVersion(const TClass *cl, Bool_t /* useBcnt */)
{
   // Copies class version to buffer, but not writes it to xml
   // Version will be written with next I/O operation or
   // will be added as attribute of class tag, created by IncrementLevel call

   BeforeIOoperation();

   if (fExpectedBaseClass!=cl)
      fExpectedBaseClass = 0;

   fVersionBuf = cl->GetClassVersion();

   if (gDebug>2)
      Info("WriteVersion", "Class: %s, version = %d",
           cl->GetName(), fVersionBuf);

   return 0;
}

//______________________________________________________________________________
void* TBufferXML::ReadObjectAny(const TClass*)
{
   // Read object from buffer. Only used from TBuffer

   BeforeIOoperation();
   if (gDebug>2)
      Info("ReadObjectAny","From node %s", fXML->GetNodeName(StackNode()));
   void* res = XmlReadObject(0);
   return res;
}

//______________________________________________________________________________
void TBufferXML::SkipObjectAny()
{
   // Skip any kind of object from buffer
   // Actually skip only one node on current level of xml structure

   ShiftStack("skipobjectany");                                          \
}

//______________________________________________________________________________
void TBufferXML::WriteObjectClass(const void *actualObjStart, const TClass *actualClass)
{
   // Write object to buffer. Only used from TBuffer

   BeforeIOoperation();
   if (gDebug>2)
      Info("WriteObject","Class %s", (actualClass ? actualClass->GetName() : " null"));
   XmlWriteObject(actualObjStart, actualClass);
}

// Macro to read content of uncompressed array
#define TXMLReadArrayNoncompress(vname) \
{                                       \
   for(Int_t indx=0;indx<n;indx++)      \
     XmlReadBasic(vname[indx]);         \
}

// macro to read content of array with compression
#define TXMLReadArrayContent(vname, arrsize)               \
{                                                          \
   Int_t indx = 0;                                         \
   while(indx<arrsize) {                                   \
     Int_t cnt = 1;                                        \
     if (fXML->HasAttr(StackNode(), xmlio::cnt))         \
        cnt = fXML->GetIntAttr(StackNode(), xmlio::cnt); \
     XmlReadBasic(vname[indx]);                            \
     Int_t curr = indx; indx++;                            \
     while(cnt>1) {                                        \
       vname[indx] = vname[curr];                          \
       cnt--; indx++;                                      \
     }                                                     \
   }                                                       \
}

// macro to read array, which include size attribute
#define TBufferXML_ReadArray(tname, vname)                    \
{                                                             \
   BeforeIOoperation();                                       \
   if (!VerifyItemNode(xmlio::Array,"ReadArray")) return 0; \
   Int_t n = fXML->GetIntAttr(StackNode(), xmlio::Size);    \
   if (n<=0) return 0;                                        \
   if (!vname) vname = new tname[n];                          \
   PushStack(StackNode());                                    \
   TXMLReadArrayContent(vname, n);                            \
   PopStack();                                                \
   ShiftStack("readarr");                                     \
   return n;                                                  \
}

//______________________________________________________________________________
void TBufferXML::ReadFloat16 (Float_t *f, TStreamerElement * /*ele*/)
{
   // read a Float16_t from the buffer
   BeforeIOoperation();
   XmlReadBasic(*f);
}

//______________________________________________________________________________
void TBufferXML::ReadDouble32 (Double_t *d, TStreamerElement * /*ele*/)
{
   // read a Double32_t from the buffer
   BeforeIOoperation();
   XmlReadBasic(*d);
}

//______________________________________________________________________________
void TBufferXML::ReadWithFactor(Float_t *ptr, Double_t /* factor */, Double_t /* minvalue */)
{
   // Read a Double32_t from the buffer when the factor and minimun value have been specified
   // see comments about Double32_t encoding at TBufferFile::WriteDouble32().
   // Currently TBufferXML does not optimize space in this case.

   BeforeIOoperation();
   XmlReadBasic(*ptr);
}

//______________________________________________________________________________
void TBufferXML::ReadWithNbits(Float_t *ptr, Int_t /* nbits */)
{
   // Read a Float16_t from the buffer when the number of bits is specified (explicitly or not)
   // see comments about Float16_t encoding at TBufferFile::WriteFloat16().
   // Currently TBufferXML does not optimize space in this case.

   BeforeIOoperation();
   XmlReadBasic(*ptr);
}

//______________________________________________________________________________
void TBufferXML::ReadWithFactor(Double_t *ptr, Double_t /* factor */, Double_t /* minvalue */)
{
   // Read a Double32_t from the buffer when the factor and minimun value have been specified
   // see comments about Double32_t encoding at TBufferFile::WriteDouble32().
   // Currently TBufferXML does not optimize space in this case.

   BeforeIOoperation();
   XmlReadBasic(*ptr);
}

//______________________________________________________________________________
void TBufferXML::ReadWithNbits(Double_t *ptr, Int_t /* nbits */)
{
   // Read a Double32_t from the buffer when the number of bits is specified (explicitly or not)
   // see comments about Double32_t encoding at TBufferFile::WriteDouble32().
   // Currently TBufferXML does not optimize space in this case.

   BeforeIOoperation();
   XmlReadBasic(*ptr);
}

//______________________________________________________________________________
void TBufferXML::WriteFloat16 (Float_t *f, TStreamerElement * /*ele*/)
{
   // write a Float16_t to the buffer
   BeforeIOoperation();
   XmlWriteBasic(*f);
}

//______________________________________________________________________________
void TBufferXML::WriteDouble32 (Double_t *d, TStreamerElement * /*ele*/)
{
   // write a Double32_t to the buffer
   BeforeIOoperation();
   XmlWriteBasic(*d);
}

//______________________________________________________________________________
Int_t TBufferXML::ReadArray(Bool_t    *&b)
{
   // Read array of Bool_t from buffer

   TBufferXML_ReadArray(Bool_t,b);
}

//______________________________________________________________________________
Int_t TBufferXML::ReadArray(Char_t    *&c)
{
   // Read array of Char_t from buffer

   TBufferXML_ReadArray(Char_t,c);
}

//______________________________________________________________________________
Int_t TBufferXML::ReadArray(UChar_t   *&c)
{
   // Read array of UChar_t from buffer

   TBufferXML_ReadArray(UChar_t,c);
}

//______________________________________________________________________________
Int_t TBufferXML::ReadArray(Short_t   *&h)
{
   // Read array of Short_t from buffer

   TBufferXML_ReadArray(Short_t,h);
}

//______________________________________________________________________________
Int_t TBufferXML::ReadArray(UShort_t  *&h)
{
   // Read array of UShort_t from buffer

   TBufferXML_ReadArray(UShort_t,h);
}

//______________________________________________________________________________
Int_t TBufferXML::ReadArray(Int_t     *&i)
{
   // Read array of Int_t from buffer

   TBufferXML_ReadArray(Int_t,i);
}

//______________________________________________________________________________
Int_t TBufferXML::ReadArray(UInt_t    *&i)
{
   // Read array of UInt_t from buffer

   TBufferXML_ReadArray(UInt_t,i);
}

//______________________________________________________________________________
Int_t TBufferXML::ReadArray(Long_t    *&l)
{
   // Read array of Long_t from buffer

   TBufferXML_ReadArray(Long_t,l);
}

//______________________________________________________________________________
Int_t TBufferXML::ReadArray(ULong_t   *&l)
{
   // Read array of ULong_t from buffer

   TBufferXML_ReadArray(ULong_t,l);
}

//______________________________________________________________________________
Int_t TBufferXML::ReadArray(Long64_t  *&l)
{
   // Read array of Long64_t from buffer

   TBufferXML_ReadArray(Long64_t,l);
}

//______________________________________________________________________________
Int_t TBufferXML::ReadArray(ULong64_t *&l)
{
   // Read array of ULong64_t from buffer

   TBufferXML_ReadArray(ULong64_t,l);
}

//______________________________________________________________________________
Int_t TBufferXML::ReadArray(Float_t   *&f)
{
   // Read array of Float_t from buffer

   TBufferXML_ReadArray(Float_t,f);
}

//______________________________________________________________________________
Int_t TBufferXML::ReadArray(Double_t  *&d)
{
   // Read array of Double_t from buffer

   TBufferXML_ReadArray(Double_t,d);
}

//______________________________________________________________________________
Int_t TBufferXML::ReadArrayFloat16(Float_t  *&f, TStreamerElement * /*ele*/)
{
   // Read array of Float16_t from buffer

   TBufferXML_ReadArray(Float_t,f);
}

//______________________________________________________________________________
Int_t TBufferXML::ReadArrayDouble32(Double_t  *&d, TStreamerElement * /*ele*/)
{
   // Read array of Double32_t from buffer

   TBufferXML_ReadArray(Double_t,d);
}

// macro to read array from xml buffer
#define TBufferXML_ReadStaticArray(vname)                           \
{                                                                   \
   BeforeIOoperation();                                             \
   if (!VerifyItemNode(xmlio::Array,"ReadStaticArray")) return 0; \
   Int_t n = fXML->GetIntAttr(StackNode(), xmlio::Size);          \
   if (n<=0) return 0;                                              \
   if (!vname) return 0;                                            \
   PushStack(StackNode());                                          \
   TXMLReadArrayContent(vname, n);                                  \
   PopStack();                                                      \
   ShiftStack("readstatarr");                                       \
   return n;                                                        \
}

//______________________________________________________________________________
Int_t TBufferXML::ReadStaticArray(Bool_t    *b)
{
   // Read array of Bool_t from buffer

   TBufferXML_ReadStaticArray(b);
}

//______________________________________________________________________________
Int_t TBufferXML::ReadStaticArray(Char_t    *c)
{
   // Read array of Char_t from buffer

   TBufferXML_ReadStaticArray(c);
}

//______________________________________________________________________________
Int_t TBufferXML::ReadStaticArray(UChar_t   *c)
{
   // Read array of UChar_t from buffer

   TBufferXML_ReadStaticArray(c);
}

//______________________________________________________________________________
Int_t TBufferXML::ReadStaticArray(Short_t   *h)
{
   // Read array of Short_t from buffer

   TBufferXML_ReadStaticArray(h);
}

//______________________________________________________________________________
Int_t TBufferXML::ReadStaticArray(UShort_t  *h)
{
   // Read array of UShort_t from buffer

   TBufferXML_ReadStaticArray(h);
}

//______________________________________________________________________________
Int_t TBufferXML::ReadStaticArray(Int_t     *i)
{
   // Read array of Int_t from buffer

   TBufferXML_ReadStaticArray(i);
}

//______________________________________________________________________________
Int_t TBufferXML::ReadStaticArray(UInt_t    *i)
{
   // Read array of UInt_t from buffer

   TBufferXML_ReadStaticArray(i);
}

//______________________________________________________________________________
Int_t TBufferXML::ReadStaticArray(Long_t    *l)
{
   // Read array of Long_t from buffer

   TBufferXML_ReadStaticArray(l);
}

//______________________________________________________________________________
Int_t TBufferXML::ReadStaticArray(ULong_t   *l)
{
   // Read array of ULong_t from buffer

   TBufferXML_ReadStaticArray(l);
}

//______________________________________________________________________________
Int_t TBufferXML::ReadStaticArray(Long64_t  *l)
{
   // Read array of Long64_t from buffer

   TBufferXML_ReadStaticArray(l);
}

//______________________________________________________________________________
Int_t TBufferXML::ReadStaticArray(ULong64_t *l)
{
   // Read array of ULong64_t from buffer

   TBufferXML_ReadStaticArray(l);
}

//______________________________________________________________________________
Int_t TBufferXML::ReadStaticArray(Float_t   *f)
{
   // Read array of Float_t from buffer

   TBufferXML_ReadStaticArray(f);
}

//______________________________________________________________________________
Int_t TBufferXML::ReadStaticArray(Double_t  *d)
{
   // Read array of Double_t from buffer

   TBufferXML_ReadStaticArray(d);
}

//______________________________________________________________________________
Int_t TBufferXML::ReadStaticArrayFloat16(Float_t  *f, TStreamerElement * /*ele*/)
{
   // Read array of Float16_t from buffer

   TBufferXML_ReadStaticArray(f);
}

//______________________________________________________________________________
Int_t TBufferXML::ReadStaticArrayDouble32(Double_t  *d, TStreamerElement * /*ele*/)
{
   // Read array of Double32_t from buffer

   TBufferXML_ReadStaticArray(d);
}

// macro to read content of array, which not include size of array
// macro also treat situation, when instead of one single array chain
// of several elements should be produced
#define TBufferXML_ReadFastArray(vname)                                   \
{                                                                         \
   BeforeIOoperation();                                                   \
   if (n<=0) return;                                                      \
   TStreamerElement* elem = Stack(0)->fElem;                              \
   if ((elem!=0) && (elem->GetType()>TStreamerInfo::kOffsetL) &&          \
       (elem->GetType()<TStreamerInfo::kOffsetP) &&                       \
       (elem->GetArrayLength()!=n)) fExpectedChain = kTRUE;               \
   if (fExpectedChain) {                                                  \
      fExpectedChain = kFALSE;                                            \
      Int_t startnumber = Stack(0)->fElemNumber;                          \
      TStreamerInfo* info = Stack(1)->fInfo;                              \
      Int_t index = 0;                                                    \
      while (index<n) {                                                   \
        elem = (TStreamerElement*)info->GetElements()->At(startnumber++); \
        if (elem->GetType()<TStreamerInfo::kOffsetL) {                    \
           if (index>0) { PopStack(); ShiftStack("chainreader"); VerifyElemNode(elem); }  \
           fCanUseCompact = kTRUE;                                        \
           XmlReadBasic(vname[index]);                                    \
           index++;                                                       \
        } else {                                                          \
           if (!VerifyItemNode(xmlio::Array,"ReadFastArray")) return;     \
           PushStack(StackNode());                                        \
           Int_t elemlen = elem->GetArrayLength();                        \
           TXMLReadArrayContent((vname+index), elemlen);                  \
           PopStack();                                                    \
           ShiftStack("readfastarr");                                     \
           index+=elemlen;                                                \
        }                                                                 \
      }                                                                   \
   } else {                                                               \
      if (!VerifyItemNode(xmlio::Array,"ReadFastArray")) return;          \
      PushStack(StackNode());                                             \
      TXMLReadArrayContent(vname, n);                                     \
      PopStack();                                                         \
      ShiftStack("readfastarr");                                          \
   }                                                                      \
}

//______________________________________________________________________________
void TBufferXML::ReadFastArray(Bool_t    *b, Int_t n)
{
   // read array of Bool_t from buffer

   TBufferXML_ReadFastArray(b);
}

//______________________________________________________________________________
void TBufferXML::ReadFastArray(Char_t    *c, Int_t n)
{
   // read array of Char_t from buffer
   // if nodename==CharStar, read all array as string

   if ((n>0) && VerifyItemNode(xmlio::CharStar)) {
      const char* buf;
      if ((buf = XmlReadValue(xmlio::CharStar))) {
         Int_t size = strlen(buf);
         if (size<n) size = n;
         memcpy(c, buf, size);
      }
   } else
      TBufferXML_ReadFastArray(c);
}

//______________________________________________________________________________
void TBufferXML::ReadFastArray(UChar_t   *c, Int_t n)
{
   // read array of UChar_t from buffer

   TBufferXML_ReadFastArray(c);
}

//______________________________________________________________________________
void TBufferXML::ReadFastArray(Short_t   *h, Int_t n)
{
   // read array of Short_t from buffer

   TBufferXML_ReadFastArray(h);
}

//______________________________________________________________________________
void TBufferXML::ReadFastArray(UShort_t  *h, Int_t n)
{
   // read array of UShort_t from buffer

   TBufferXML_ReadFastArray(h);
}

//______________________________________________________________________________
void TBufferXML::ReadFastArray(Int_t     *i, Int_t n)
{
   // read array of Int_t from buffer

   TBufferXML_ReadFastArray(i);
}

//______________________________________________________________________________
void TBufferXML::ReadFastArray(UInt_t    *i, Int_t n)
{
   // read array of UInt_t from buffer

   TBufferXML_ReadFastArray(i);
}

//______________________________________________________________________________
void TBufferXML::ReadFastArray(Long_t    *l, Int_t n)
{
   // read array of Long_t from buffer

   TBufferXML_ReadFastArray(l);
}

//______________________________________________________________________________
void TBufferXML::ReadFastArray(ULong_t   *l, Int_t n)
{
   // read array of ULong_t from buffer

   TBufferXML_ReadFastArray(l);
}

//______________________________________________________________________________
void TBufferXML::ReadFastArray(Long64_t  *l, Int_t n)
{
   // read array of Long64_t from buffer

   TBufferXML_ReadFastArray(l);
}

//______________________________________________________________________________
void TBufferXML::ReadFastArray(ULong64_t *l, Int_t n)
{
   // read array of ULong64_t from buffer

   TBufferXML_ReadFastArray(l);
}

//______________________________________________________________________________
void TBufferXML::ReadFastArray(Float_t   *f, Int_t n)
{
   // read array of Float_t from buffer

   TBufferXML_ReadFastArray(f);
}

//______________________________________________________________________________
void TBufferXML::ReadFastArray(Double_t  *d, Int_t n)
{
   // read array of Double_t from buffer

   TBufferXML_ReadFastArray(d);
}

//______________________________________________________________________________
void TBufferXML::ReadFastArrayFloat16(Float_t  *f, Int_t n, TStreamerElement * /*ele*/)
{
   // read array of Float16_t from buffer

   TBufferXML_ReadFastArray(f);
}

//______________________________________________________________________________
void TBufferXML::ReadFastArrayWithFactor(Float_t  *f, Int_t n, Double_t /* factor */, Double_t /* minvalue */)
{
   // read array of Float16_t from buffer

   TBufferXML_ReadFastArray(f);
}

//______________________________________________________________________________
void TBufferXML::ReadFastArrayWithNbits(Float_t  *f, Int_t n, Int_t /*nbits*/)
{
   // read array of Float16_t from buffer

   TBufferXML_ReadFastArray(f);
}

//______________________________________________________________________________
void TBufferXML::ReadFastArrayDouble32(Double_t  *d, Int_t n, TStreamerElement * /*ele*/)
{
   // read array of Double32_t from buffer

   TBufferXML_ReadFastArray(d);
}

//______________________________________________________________________________
void TBufferXML::ReadFastArrayWithFactor(Double_t  *d, Int_t n, Double_t /* factor */, Double_t /* minvalue */)
{
   // read array of Double32_t from buffer

   TBufferXML_ReadFastArray(d);
}

//______________________________________________________________________________
void TBufferXML::ReadFastArrayWithNbits(Double_t  *d, Int_t n, Int_t /*nbits*/)
{
   // read array of Double32_t from buffer

   TBufferXML_ReadFastArray(d);
}

//______________________________________________________________________________
void TBufferXML::ReadFastArray(void  *start, const TClass *cl, Int_t n, TMemberStreamer *s, const TClass *onFileClass)
{
   // redefined here to avoid warning message from gcc

   TBufferFile::ReadFastArray(start, cl, n, s, onFileClass);
}

//______________________________________________________________________________
void TBufferXML::ReadFastArray(void **startp, const TClass *cl, Int_t n, Bool_t isPreAlloc, TMemberStreamer *s, const TClass *onFileClass)
{
   // redefined here to avoid warning message from gcc

   TBufferFile::ReadFastArray(startp, cl, n, isPreAlloc, s, onFileClass);
}

// macro to write content of noncompressed array
#define TXMLWriteArrayNoncompress(vname, arrsize) \
{                                                 \
   for(Int_t indx=0;indx<arrsize;indx++)          \
     XmlWriteBasic(vname[indx]);                  \
}

// macro to write content of compressed array
#define TXMLWriteArrayCompress(vname, arrsize)                     \
{                                                                  \
   Int_t indx = 0;                                                 \
   while(indx<arrsize) {                                           \
      XMLNodePointer_t elemnode = XmlWriteBasic(vname[indx]);      \
      Int_t curr = indx; indx++;                                   \
      while ((indx<arrsize) && (vname[indx]==vname[curr])) indx++; \
      if (indx-curr > 1)                                           \
         fXML->NewIntAttr(elemnode, xmlio::cnt, indx-curr);      \
   }                                                               \
}

#define TXMLWriteArrayContent(vname, arrsize)   \
{                                               \
   if (fCompressLevel>0) {                      \
     TXMLWriteArrayCompress(vname, arrsize)     \
   } else {                                     \
     TXMLWriteArrayNoncompress(vname, arrsize)  \
   }                                            \
}

// macro to write array, which include size
#define TBufferXML_WriteArray(vname)                          \
{                                                             \
   BeforeIOoperation();                                       \
   XMLNodePointer_t arrnode = CreateItemNode(xmlio::Array); \
   fXML->NewIntAttr(arrnode, xmlio::Size, n);               \
   PushStack(arrnode);                                        \
   TXMLWriteArrayContent(vname, n);                           \
   PopStack();                                                \
}

//______________________________________________________________________________
void TBufferXML::WriteArray(const Bool_t    *b, Int_t n)
{
   // Write array of Bool_t to buffer

   TBufferXML_WriteArray(b);
}

//______________________________________________________________________________
void TBufferXML::WriteArray(const Char_t    *c, Int_t n)
{
   // Write array of Char_t to buffer

   TBufferXML_WriteArray(c);
}

//______________________________________________________________________________
void TBufferXML::WriteArray(const UChar_t   *c, Int_t n)
{
   // Write array of UChar_t to buffer

   TBufferXML_WriteArray(c);
}

//______________________________________________________________________________
void TBufferXML::WriteArray(const Short_t   *h, Int_t n)
{
   // Write array of Short_t to buffer

   TBufferXML_WriteArray(h);
}

//______________________________________________________________________________
void TBufferXML::WriteArray(const UShort_t  *h, Int_t n)
{
   // Write array of UShort_t to buffer

   TBufferXML_WriteArray(h);
}

//______________________________________________________________________________
void TBufferXML::WriteArray(const Int_t     *i, Int_t n)
{
   // Write array of Int_ to buffer

   TBufferXML_WriteArray(i);
}

//______________________________________________________________________________
void TBufferXML::WriteArray(const UInt_t    *i, Int_t n)
{
   // Write array of UInt_t to buffer

   TBufferXML_WriteArray(i);
}

//______________________________________________________________________________
void TBufferXML::WriteArray(const Long_t    *l, Int_t n)
{
   // Write array of Long_t to buffer

   TBufferXML_WriteArray(l);
}

//______________________________________________________________________________
void TBufferXML::WriteArray(const ULong_t   *l, Int_t n)
{
   // Write array of ULong_t to buffer

   TBufferXML_WriteArray(l);
}

//______________________________________________________________________________
void TBufferXML::WriteArray(const Long64_t  *l, Int_t n)
{
   // Write array of Long64_t to buffer

   TBufferXML_WriteArray(l);
}

//______________________________________________________________________________
void TBufferXML::WriteArray(const ULong64_t *l, Int_t n)
{
   // Write array of ULong64_t to buffer

   TBufferXML_WriteArray(l);
}

//______________________________________________________________________________
void TBufferXML::WriteArray(const Float_t   *f, Int_t n)
{
   // Write array of Float_t to buffer

   TBufferXML_WriteArray(f);
}

//______________________________________________________________________________
void TBufferXML::WriteArray(const Double_t  *d, Int_t n)
{
   // Write array of Double_t to buffer

   TBufferXML_WriteArray(d);
}

//______________________________________________________________________________
void TBufferXML::WriteArrayFloat16(const Float_t  *f, Int_t n, TStreamerElement * /*ele*/)
{
   // Write array of Float16_t to buffer

   TBufferXML_WriteArray(f);
}

//______________________________________________________________________________
void TBufferXML::WriteArrayDouble32(const Double_t  *d, Int_t n, TStreamerElement * /*ele*/)
{
   // Write array of Double32_t to buffer

   TBufferXML_WriteArray(d);
}

// write array without size attribute
// macro also treat situation, when instead of one single array
// chain of several elements should be produced
#define TBufferXML_WriteFastArray(vname)                                  \
{                                                                         \
   BeforeIOoperation();                                                   \
   if (n<=0) return;                                                      \
   TStreamerElement* elem = Stack(0)->fElem;                              \
   if ((elem!=0) && (elem->GetType()>TStreamerInfo::kOffsetL) &&          \
       (elem->GetType()<TStreamerInfo::kOffsetP) &&                       \
       (elem->GetArrayLength()!=n)) fExpectedChain = kTRUE;               \
   if (fExpectedChain) {                                                  \
      TStreamerInfo* info = Stack(1)->fInfo;                              \
      Int_t startnumber = Stack(0)->fElemNumber;                          \
      fExpectedChain = kFALSE;                                            \
      Int_t index = 0;                                                    \
      while (index<n) {                                                   \
        elem =(TStreamerElement*)info->GetElements()->At(startnumber++);  \
        if (elem->GetType()<TStreamerInfo::kOffsetL) {                    \
          if(index>0) { PopStack(); CreateElemNode(elem); }               \
          fCanUseCompact = kTRUE;                                         \
          XmlWriteBasic(vname[index]);                                    \
          index++;                                                        \
        } else {                                                          \
          XMLNodePointer_t arrnode = CreateItemNode(xmlio::Array);        \
          Int_t elemlen = elem->GetArrayLength();                         \
          PushStack(arrnode);                                             \
          TXMLWriteArrayContent((vname+index), elemlen);                  \
          index+=elemlen;                                                 \
          PopStack();                                                     \
        }                                                                 \
      }                                                                   \
   } else {                                                               \
      XMLNodePointer_t arrnode = CreateItemNode(xmlio::Array);            \
      PushStack(arrnode);                                                 \
      TXMLWriteArrayContent(vname, n);                                    \
      PopStack();                                                         \
   }                                                                      \
}

//______________________________________________________________________________
void TBufferXML::WriteFastArray(const Bool_t    *b, Int_t n)
{
   // Write array of Bool_t to buffer

   TBufferXML_WriteFastArray(b);
}

//______________________________________________________________________________
void TBufferXML::WriteFastArray(const Char_t    *c, Int_t n)
{
   // Write array of Char_t to buffer
   // If array does not include any special characters,
   // it will be reproduced as CharStar node with string as attribute

   Bool_t usedefault = (n==0) || fExpectedChain;
   const Char_t* buf = c;
   if (!usedefault)
      for (int i=0;i<n;i++) {
         if (*buf < 27) { usedefault = kTRUE; break; }
         buf++;
      }
   if (usedefault) {
      TBufferXML_WriteFastArray(c);
   } else {
      Char_t* buf2 = new Char_t[n+1];
      memcpy(buf2, c, n);
      buf2[n] = 0;
      XmlWriteValue(buf2, xmlio::CharStar);
      delete[] buf2;
   }
}

//______________________________________________________________________________
void TBufferXML::WriteFastArray(const UChar_t   *c, Int_t n)
{
   // Write array of UChar_t to buffer

   TBufferXML_WriteFastArray(c);
}

//______________________________________________________________________________
void TBufferXML::WriteFastArray(const Short_t   *h, Int_t n)
{
   // Write array of Short_t to buffer

   TBufferXML_WriteFastArray(h);
}

//______________________________________________________________________________
void TBufferXML::WriteFastArray(const UShort_t  *h, Int_t n)
{
   // Write array of UShort_t to buffer

   TBufferXML_WriteFastArray(h);
}

//______________________________________________________________________________
void TBufferXML::WriteFastArray(const Int_t     *i, Int_t n)
{
   // Write array of Int_t to buffer

   TBufferXML_WriteFastArray(i);
}

//______________________________________________________________________________
void TBufferXML::WriteFastArray(const UInt_t    *i, Int_t n)
{
   // Write array of UInt_t to buffer

   TBufferXML_WriteFastArray(i);
}

//______________________________________________________________________________
void TBufferXML::WriteFastArray(const Long_t    *l, Int_t n)
{
   // Write array of Long_t to buffer

   TBufferXML_WriteFastArray(l);
}

//______________________________________________________________________________
void TBufferXML::WriteFastArray(const ULong_t   *l, Int_t n)
{
   // Write array of ULong_t to buffer

   TBufferXML_WriteFastArray(l);
}

//______________________________________________________________________________
void TBufferXML::WriteFastArray(const Long64_t  *l, Int_t n)
{
   // Write array of Long64_t to buffer

   TBufferXML_WriteFastArray(l);
}

//______________________________________________________________________________
void TBufferXML::WriteFastArray(const ULong64_t *l, Int_t n)
{
   // Write array of ULong64_t to buffer

   TBufferXML_WriteFastArray(l);
}

//______________________________________________________________________________
void TBufferXML::WriteFastArray(const Float_t   *f, Int_t n)
{
   // Write array of Float_t to buffer

   TBufferXML_WriteFastArray(f);
}

//______________________________________________________________________________
void TBufferXML::WriteFastArray(const Double_t  *d, Int_t n)
{
   // Write array of Double_t to buffer

   TBufferXML_WriteFastArray(d);
}

//______________________________________________________________________________
void TBufferXML::WriteFastArrayFloat16(const Float_t  *f, Int_t n, TStreamerElement * /*ele*/)
{
   // Write array of Float16_t to buffer

   TBufferXML_WriteFastArray(f);
}

//______________________________________________________________________________
void TBufferXML::WriteFastArrayDouble32(const Double_t  *d, Int_t n, TStreamerElement * /*ele*/)
{
   // Write array of Double32_t to buffer

   TBufferXML_WriteFastArray(d);
}

//______________________________________________________________________________
void  TBufferXML::WriteFastArray(void  *start,  const TClass *cl, Int_t n, TMemberStreamer *s)
{
   // Recall TBuffer function to avoid gcc warning message

   TBufferFile::WriteFastArray(start, cl, n, s);
}

//______________________________________________________________________________
Int_t TBufferXML::WriteFastArray(void **startp, const TClass *cl, Int_t n, Bool_t isPreAlloc, TMemberStreamer *s)
{
   // Recall TBuffer function to avoid gcc warning message

   return TBufferFile::WriteFastArray(startp, cl, n, isPreAlloc, s);
}

//______________________________________________________________________________
void TBufferXML::StreamObject(void *obj, const type_info &typeinfo, const TClass* /* onFileClass */ )
{
   // steram object to/from buffer

   StreamObject(obj, TClass::GetClass(typeinfo));
}

//______________________________________________________________________________
void TBufferXML::StreamObject(void *obj, const char *className, const TClass* /* onFileClass */ )
{
   // steram object to/from buffer

   StreamObject(obj, TClass::GetClass(className));
}

void TBufferXML::StreamObject(TObject *obj)
{
   // steram object to/from buffer

   StreamObject(obj, obj ? obj->IsA() : TObject::Class());
}

//______________________________________________________________________________
void TBufferXML::StreamObject(void *obj, const TClass *cl, const TClass* /* onfileClass */ )
{
   // steram object to/from buffer

   BeforeIOoperation();
   if (gDebug>1)
      Info("StreamObject","Class: %s", (cl ? cl->GetName() : "none"));
   if (IsReading())
      XmlReadObject(obj);
   else
      XmlWriteObject(obj, cl);
}

// macro for right shift operator for basic type
#define TBufferXML_operatorin(vname) \
{                                    \
  BeforeIOoperation();               \
  XmlReadBasic(vname);               \
}

//______________________________________________________________________________
void TBufferXML::ReadBool(Bool_t    &b)
{
   // Reads Bool_t value from buffer

   TBufferXML_operatorin(b);
}

//______________________________________________________________________________
void TBufferXML::ReadChar(Char_t    &c)
{
   // Reads Char_t value from buffer

   TBufferXML_operatorin(c);
}

//______________________________________________________________________________
void TBufferXML::ReadUChar(UChar_t   &c)
{
   // Reads UChar_t value from buffer

   TBufferXML_operatorin(c);
}

//______________________________________________________________________________
void TBufferXML::ReadShort(Short_t   &h)
{
   // Reads Short_t value from buffer

   TBufferXML_operatorin(h);
}

//______________________________________________________________________________
void TBufferXML::ReadUShort(UShort_t  &h)
{
   // Reads UShort_t value from buffer

   TBufferXML_operatorin(h);
}

//______________________________________________________________________________
void TBufferXML::ReadInt(Int_t     &i)
{
   // Reads Int_t value from buffer

   TBufferXML_operatorin(i);
}

//______________________________________________________________________________
void TBufferXML::ReadUInt(UInt_t    &i)
{
   // Reads UInt_t value from buffer

   TBufferXML_operatorin(i);
}

//______________________________________________________________________________
void TBufferXML::ReadLong(Long_t    &l)
{
   // Reads Long_t value from buffer

   TBufferXML_operatorin(l);
}

//______________________________________________________________________________
void TBufferXML::ReadULong(ULong_t   &l)
{
   // Reads ULong_t value from buffer

   TBufferXML_operatorin(l);
}

//______________________________________________________________________________
void TBufferXML::ReadLong64(Long64_t  &l)
{
   // Reads Long64_t value from buffer

   TBufferXML_operatorin(l);
}

//______________________________________________________________________________
void TBufferXML::ReadULong64(ULong64_t &l)
{
   // Reads ULong64_t value from buffer

   TBufferXML_operatorin(l);
}

//______________________________________________________________________________
void TBufferXML::ReadFloat(Float_t   &f)
{
   // Reads Float_t value from buffer

   TBufferXML_operatorin(f);
}

//______________________________________________________________________________
void TBufferXML::ReadDouble(Double_t  &d)
{
   // Reads Double_t value from buffer

   TBufferXML_operatorin(d);
}

//______________________________________________________________________________
void TBufferXML::ReadCharP(Char_t    *c)
{
   // Reads array of characters from buffer

   BeforeIOoperation();
   const char* buf;
   if ((buf = XmlReadValue(xmlio::CharStar)))
      strcpy(c, buf);
}

//______________________________________________________________________________
void TBufferXML::ReadTString(TString &s)
{
   // Reads a TString

   if (GetIOVersion()<3) {
      TBufferFile::ReadTString(s);
   } else {
      BeforeIOoperation();
      const char* buf;
      if ((buf = XmlReadValue(xmlio::String)))
         s = buf;
   }
}

//______________________________________________________________________________
void TBufferXML::ReadStdString(std::string *s)
{
   // Reads a std::string

   if (GetIOVersion()<3) {
      TBufferFile::ReadStdString(s);
   } else {
      BeforeIOoperation();
      const char* buf;
      if ((buf = XmlReadValue(xmlio::String)))
         if (s) *s = buf;
   }
}

//______________________________________________________________________________
void TBufferXML::ReadCharStar(char* &s)
{
   // Read a char* string

   TBufferFile::ReadCharStar(s);
}


// macro for left shift operator for basic types
#define TBufferXML_operatorout(vname) \
{                                     \
  BeforeIOoperation();                \
  XmlWriteBasic(vname);               \
}

//______________________________________________________________________________
void TBufferXML::WriteBool(Bool_t    b)
{
   // Writes Bool_t value to buffer

   TBufferXML_operatorout(b);
}

//______________________________________________________________________________
void TBufferXML::WriteChar(Char_t    c)
{
   // Writes Char_t value to buffer

   TBufferXML_operatorout(c);
}

//______________________________________________________________________________
void TBufferXML::WriteUChar(UChar_t   c)
{
   // Writes UChar_t value to buffer

   TBufferXML_operatorout(c);
}

//______________________________________________________________________________
void TBufferXML::WriteShort(Short_t   h)
{
   // Writes Short_t value to buffer

   TBufferXML_operatorout(h);
}

//______________________________________________________________________________
void TBufferXML::WriteUShort(UShort_t  h)
{
   // Writes UShort_t value to buffer

   TBufferXML_operatorout(h);
}

//______________________________________________________________________________
void TBufferXML::WriteInt(Int_t     i)
{
   // Writes Int_t value to buffer

   TBufferXML_operatorout(i);
}

//______________________________________________________________________________
void TBufferXML::WriteUInt(UInt_t    i)
{
   // Writes UInt_t value to buffer

   TBufferXML_operatorout(i);
}

//______________________________________________________________________________
void TBufferXML::WriteLong(Long_t    l)
{
   // Writes Long_t value to buffer

   TBufferXML_operatorout(l);
}

//______________________________________________________________________________
void TBufferXML::WriteULong(ULong_t   l)
{
   // Writes ULong_t value to buffer

   TBufferXML_operatorout(l);
}

//______________________________________________________________________________
void TBufferXML::WriteLong64(Long64_t  l)
{
   // Writes Long64_t value to buffer

   TBufferXML_operatorout(l);
}

//______________________________________________________________________________
void TBufferXML::WriteULong64(ULong64_t l)
{
   // Writes ULong64_t value to buffer

   TBufferXML_operatorout(l);
}

//______________________________________________________________________________
void TBufferXML::WriteFloat(Float_t   f)
{
   // Writes Float_t value to buffer

   TBufferXML_operatorout(f);
}

//______________________________________________________________________________
void TBufferXML::WriteDouble(Double_t  d)
{
   // Writes Double_t value to buffer

   TBufferXML_operatorout(d);
}

//______________________________________________________________________________
void TBufferXML::WriteCharP(const Char_t *c)
{
   // Writes array of characters to buffer

   BeforeIOoperation();
   XmlWriteValue(c, xmlio::CharStar);
}

//______________________________________________________________________________
void TBufferXML::WriteTString(const TString &s)
{
   // Writes a TString

   if (GetIOVersion()<3) {
      TBufferFile::WriteTString(s);
   } else {
      BeforeIOoperation();
      XmlWriteValue(s.Data(), xmlio::String);
   }
}

//______________________________________________________________________________
void TBufferXML::WriteStdString(const std::string *s)
{
   // Writes a TString

   if (GetIOVersion()<3) {
      TBufferFile::WriteStdString(s);
   } else {
      BeforeIOoperation();
      if (s) XmlWriteValue(s->c_str(), xmlio::String);
        else XmlWriteValue("", xmlio::String);
   }
}

//______________________________________________________________________________
void TBufferXML::WriteCharStar(char *s)
{
   // Write a char* string

   TBufferFile::WriteCharStar(s);
}

//______________________________________________________________________________
XMLNodePointer_t TBufferXML::XmlWriteBasic(Char_t value)
{
   // converts Char_t to string and add xml node to buffer

   char buf[50];
   snprintf(buf, sizeof(buf), "%d",value);
   return XmlWriteValue(buf, xmlio::Char);
}

//______________________________________________________________________________
XMLNodePointer_t  TBufferXML::XmlWriteBasic(Short_t value)
{
   // converts Short_t to string and add xml node to buffer

   char buf[50];
   snprintf(buf, sizeof(buf), "%hd", value);
   return XmlWriteValue(buf, xmlio::Short);
}

//______________________________________________________________________________
XMLNodePointer_t TBufferXML::XmlWriteBasic(Int_t value)
{
   // converts Int_t to string and add xml node to buffer

   char buf[50];
   snprintf(buf, sizeof(buf), "%d", value);
   return XmlWriteValue(buf, xmlio::Int);
}

//______________________________________________________________________________
XMLNodePointer_t TBufferXML::XmlWriteBasic(Long_t value)
{
   // converts Long_t to string and add xml node to buffer

   char buf[50];
   snprintf(buf, sizeof(buf), "%ld", value);
   return XmlWriteValue(buf, xmlio::Long);
}

//______________________________________________________________________________
XMLNodePointer_t TBufferXML::XmlWriteBasic(Long64_t value)
{
   // converts Long64_t to string and add xml node to buffer

   char buf[50];
   snprintf(buf, sizeof(buf), FLong64, value);
   return XmlWriteValue(buf, xmlio::Long64);
}

//______________________________________________________________________________
XMLNodePointer_t  TBufferXML::XmlWriteBasic(Float_t value)
{
   // converts Float_t to string and add xml node to buffer

   char buf[200];
   snprintf(buf, sizeof(buf), fgFloatFmt, value);
   return XmlWriteValue(buf, xmlio::Float);
}

//______________________________________________________________________________
XMLNodePointer_t TBufferXML::XmlWriteBasic(Double_t value)
{
   // converts Double_t to string and add xml node to buffer

   char buf[1000];
   snprintf(buf, sizeof(buf), fgFloatFmt, value);
   return XmlWriteValue(buf, xmlio::Double);
}

//______________________________________________________________________________
XMLNodePointer_t TBufferXML::XmlWriteBasic(Bool_t value)
{
   // converts Bool_t to string and add xml node to buffer

   return XmlWriteValue(value ? xmlio::True : xmlio::False, xmlio::Bool);
}

//______________________________________________________________________________
XMLNodePointer_t TBufferXML::XmlWriteBasic(UChar_t value)
{
   // converts UChar_t to string and add xml node to buffer

   char buf[50];
   snprintf(buf, sizeof(buf), "%u", value);
   return XmlWriteValue(buf, xmlio::UChar);
}

//______________________________________________________________________________
XMLNodePointer_t TBufferXML::XmlWriteBasic(UShort_t value)
{
   // converts UShort_t to string and add xml node to buffer

   char buf[50];
   snprintf(buf, sizeof(buf), "%hu", value);
   return XmlWriteValue(buf, xmlio::UShort);
}

//______________________________________________________________________________
XMLNodePointer_t TBufferXML::XmlWriteBasic(UInt_t value)
{
   // converts UInt_t to string and add xml node to buffer

   char buf[50];
   snprintf(buf, sizeof(buf), "%u", value);
   return XmlWriteValue(buf, xmlio::UInt);
}

//______________________________________________________________________________
XMLNodePointer_t TBufferXML::XmlWriteBasic(ULong_t value)
{
   // converts ULong_t to string and add xml node to buffer

   char buf[50];
   snprintf(buf, sizeof(buf), "%lu", value);
   return XmlWriteValue(buf, xmlio::ULong);
}

//______________________________________________________________________________
XMLNodePointer_t TBufferXML::XmlWriteBasic(ULong64_t value)
{
   // converts ULong64_t to string and add xml node to buffer

   char buf[50];
   snprintf(buf, sizeof(buf), FULong64, value);
   return XmlWriteValue(buf, xmlio::ULong64);
}

//______________________________________________________________________________
XMLNodePointer_t TBufferXML::XmlWriteValue(const char* value, const char* name)
{
   // create xml node with specified name and adds it to stack node

   XMLNodePointer_t node = 0;

   if (fCanUseCompact)
      node = StackNode();
   else
      node = CreateItemNode(name);

   fXML->NewAttr(node, 0, xmlio::v, value);

   fCanUseCompact = kFALSE;

   return node;
}

//______________________________________________________________________________
void TBufferXML::XmlReadBasic(Char_t& value)
{
   // reads string from current xml node and convert it to Char_t value

   const char* res = XmlReadValue(xmlio::Char);
   if (res) {
      int n;
      sscanf(res,"%d", &n);
      value = n;
   } else
      value = 0;
}

//______________________________________________________________________________
void TBufferXML::XmlReadBasic(Short_t& value)
{
   // reads string from current xml node and convert it to Short_t value

   const char* res = XmlReadValue(xmlio::Short);
   if (res)
      sscanf(res,"%hd", &value);
   else
      value = 0;
}

//______________________________________________________________________________
void TBufferXML::XmlReadBasic(Int_t& value)
{
   // reads string from current xml node and convert it to Int_t value

   const char* res = XmlReadValue(xmlio::Int);
   if (res)
      sscanf(res,"%d", &value);
   else
      value = 0;
}

//______________________________________________________________________________
void TBufferXML::XmlReadBasic(Long_t& value)
{
   // reads string from current xml node and convert it to Long_t value

   const char* res = XmlReadValue(xmlio::Long);
   if (res)
      sscanf(res,"%ld", &value);
   else
      value = 0;
}

//______________________________________________________________________________
void TBufferXML::XmlReadBasic(Long64_t& value)
{
   // reads string from current xml node and convert it to Long64_t value

   const char* res = XmlReadValue(xmlio::Long64);
   if (res)
      sscanf(res, FLong64, &value);
   else
      value = 0;
}

//______________________________________________________________________________
void TBufferXML::XmlReadBasic(Float_t& value)
{
   // reads string from current xml node and convert it to Float_t value

   const char* res = XmlReadValue(xmlio::Float);
   if (res)
      sscanf(res, "%f", &value);
   else
      value = 0.;
}

//______________________________________________________________________________
void TBufferXML::XmlReadBasic(Double_t& value)
{
   // reads string from current xml node and convert it to Double_t value

   const char* res = XmlReadValue(xmlio::Double);
   if (res)
      sscanf(res, "%lf", &value);
   else
      value = 0.;
}

//______________________________________________________________________________
void TBufferXML::XmlReadBasic(Bool_t& value)
{
   // reads string from current xml node and convert it to Bool_t value

   const char* res = XmlReadValue(xmlio::Bool);
   if (res)
      value = (strcmp(res, xmlio::True)==0);
   else
      value = kFALSE;
}

//______________________________________________________________________________
void TBufferXML::XmlReadBasic(UChar_t& value)
{
   // reads string from current xml node and convert it to UChar_t value

   const char* res = XmlReadValue(xmlio::UChar);
   if (res) {
      unsigned int n;
      sscanf(res,"%ud", &n);
      value = n;
   } else
      value = 0;
}

//______________________________________________________________________________
void TBufferXML::XmlReadBasic(UShort_t& value)
{
   // reads string from current xml node and convert it to UShort_t value

   const char* res = XmlReadValue(xmlio::UShort);
   if (res)
      sscanf(res,"%hud", &value);
   else
      value = 0;
}

//______________________________________________________________________________
void TBufferXML::XmlReadBasic(UInt_t& value)
{
   // reads string from current xml node and convert it to UInt_t value

   const char* res = XmlReadValue(xmlio::UInt);
   if (res)
      sscanf(res,"%u", &value);
   else
      value = 0;
}

//______________________________________________________________________________
void TBufferXML::XmlReadBasic(ULong_t& value)
{
   // reads string from current xml node and convert it to ULong_t value

   const char* res = XmlReadValue(xmlio::ULong);
   if (res)
      sscanf(res,"%lu", &value);
   else
      value = 0;
}

//______________________________________________________________________________
void TBufferXML::XmlReadBasic(ULong64_t& value)
{
   // reads string from current xml node and convert it to ULong64_t value

   const char* res = XmlReadValue(xmlio::ULong64);
   if (res)
      sscanf(res, FULong64, &value);
   else
      value = 0;
}

//______________________________________________________________________________
const char* TBufferXML::XmlReadValue(const char* name)
{
   // read string value from current stack node

   if (fErrorFlag>0) return 0;

   Bool_t trysimple = fCanUseCompact;
   fCanUseCompact = kFALSE;

   if (trysimple) {
      if (fXML->HasAttr(Stack(1)->fNode,xmlio::v))
         fValueBuf = fXML->GetAttr(Stack(1)->fNode, xmlio::v);
      else
         trysimple = kFALSE;
   }

   if (!trysimple) {
      if (!VerifyItemNode(name, "XmlReadValue")) return 0;
      fValueBuf = fXML->GetAttr(StackNode(), xmlio::v);
   }

   if (gDebug>4)
      Info("XmlReadValue","     Name = %s value = %s", name, fValueBuf.Data());

   if (!trysimple)
      ShiftStack("readvalue");

   return fValueBuf.Data();
}

void TBufferXML::SetFloatFormat(const char* fmt)
{
   // set printf format for float/double members, default "%e"

   if (fmt==0) fmt = "%e";
   fgFloatFmt = fmt;
}

const char* TBufferXML::GetFloatFormat()
{
   // return current printf format for float/double members, default "%e"

   return fgFloatFmt;
}

//______________________________________________________________________________
Int_t TBufferXML::ApplySequence(const TStreamerInfoActions::TActionSequence &sequence, void *obj)
{
   // Read one collection of objects from the buffer using the StreamerInfoLoopAction.
   // The collection needs to be a split TClonesArray or a split vector of pointers.

   TVirtualStreamerInfo *info = sequence.fStreamerInfo;
   IncrementLevel(info);

   if (gDebug) {
      //loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for(TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin();
          iter != end;
          ++iter) {
         // Idea: Try to remove this function call as it is really needed only for XML streaming.
         SetStreamerElementNumber((*iter).fConfiguration->fCompInfo->fElem,(*iter).fConfiguration->fCompInfo->fType);
         (*iter).PrintDebug(*this,obj);
         (*iter)(*this,obj);
      }

   } else {
      //loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for(TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin();
          iter != end;
          ++iter) {
         // Idea: Try to remove this function call as it is really needed only for XML streaming.
         SetStreamerElementNumber((*iter).fConfiguration->fCompInfo->fElem,(*iter).fConfiguration->fCompInfo->fType);
         (*iter)(*this,obj);
      }
   }

   DecrementLevel(info);
   return 0;
}

//______________________________________________________________________________
Int_t TBufferXML::ApplySequenceVecPtr(const TStreamerInfoActions::TActionSequence &sequence, void *start_collection, void *end_collection)
{
   // Read one collection of objects from the buffer using the StreamerInfoLoopAction.
   // The collection needs to be a split TClonesArray or a split vector of pointers.

   TVirtualStreamerInfo *info = sequence.fStreamerInfo;
   IncrementLevel(info);

   if (gDebug) {
      //loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for(TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin();
          iter != end;
          ++iter) {
         // Idea: Try to remove this function call as it is really needed only for XML streaming.
         SetStreamerElementNumber((*iter).fConfiguration->fCompInfo->fElem,(*iter).fConfiguration->fCompInfo->fType);
         (*iter).PrintDebug(*this,*(char**)start_collection);  // Warning: This limits us to TClonesArray and vector of pointers.
         (*iter)(*this,start_collection,end_collection);
      }

   } else {
      //loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for(TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin();
          iter != end;
          ++iter) {
         // Idea: Try to remove this function call as it is really needed only for XML streaming.
         SetStreamerElementNumber((*iter).fConfiguration->fCompInfo->fElem,(*iter).fConfiguration->fCompInfo->fType);
         (*iter)(*this,start_collection,end_collection);
      }
   }

   DecrementLevel(info);
   return 0;
}

//______________________________________________________________________________
Int_t TBufferXML::ApplySequence(const TStreamerInfoActions::TActionSequence &sequence, void *start_collection, void *end_collection)
{
   // Read one collection of objects from the buffer using the StreamerInfoLoopAction.

   TVirtualStreamerInfo *info = sequence.fStreamerInfo;
   IncrementLevel(info);

   TStreamerInfoActions::TLoopConfiguration *loopconfig = sequence.fLoopConfig;
   if (gDebug) {

      // Get the address of the first item for the PrintDebug.
      // (Performance is not essential here since we are going to print to
      // the screen anyway).
      void *arr0 = loopconfig->GetFirstAddress(start_collection,end_collection);
      // loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for(TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin();
          iter != end;
          ++iter) {
         // Idea: Try to remove this function call as it is really needed only for XML streaming.
         SetStreamerElementNumber((*iter).fConfiguration->fCompInfo->fElem,(*iter).fConfiguration->fCompInfo->fType);
         (*iter).PrintDebug(*this,arr0);
         (*iter)(*this,start_collection,end_collection,loopconfig);
      }

   } else {
      //loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for(TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin();
          iter != end;
          ++iter) {
         // Idea: Try to remove this function call as it is really needed only for XML streaming.
         SetStreamerElementNumber((*iter).fConfiguration->fCompInfo->fElem,(*iter).fConfiguration->fCompInfo->fType);
         (*iter)(*this,start_collection,end_collection,loopconfig);
      }
   }

   DecrementLevel(info);
   return 0;
}
