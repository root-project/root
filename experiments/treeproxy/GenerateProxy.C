/*
  TODO:
  Have separate names for the wrapper classes in the cases of: [done]
    clones/non clones
    split/non split
    split levels

  Have a solution for passing top+"."+middle to the parents classes [probably done .. need testing]

  Have a solution for the return by references of abstract classes [not done]

  Have object inside ClonesArray properly treated! [done]
  Why is there 2 TRef proxy classes? [done]

  check why some inheritance are TObjProxy and not TPx_

  Be smart enough to avoid issue about having 2 classes one unrolled and one non unrolled!

  When using in interpreted mode understand why the reloading reloads the calling script and then crashes :(

  CINT does not properly call the custom operators when doing return fNtrack.

  CINT does not handle fMatrix[2][1] well.

  The user's function in script.h are not exposed by ACLiC.

  Add a method to avoid the useless refreshing of the generated file
    - this implies a predictable algorithm for the generated file name
    - for most efficiency it would require a different name for each tree
    - it would need to refresh the file only if it changed (i.e. create in a temporary file and copy over only if needed).
 */

#include "TList.h"
#include <stdio.h>

class TTree;
class TBranch;
class TStreamerElement;

#include "TClass.h"
#include "TClonesArray.h"
#include "TError.h"
#include "TROOT.h"

#include "TTreeFormula.h"
#include "TFormLeafInfo.h"

class TProxyDescriptor : public TNamed {
   TString fBranchName;
   bool fIsSplit;

public:
   TProxyDescriptor(const char *dataname, const char *type, const char *branchname, bool split = true) :
      TNamed(dataname,type),fBranchName(branchname),fIsSplit(split) {}
   const char *GetDataName() { return GetName(); }
   const char *GetTypeName() { return GetTitle(); }
   const char *GetBranchName() { return fBranchName.Data(); }

   using TObject::IsEqual;
   bool IsEqual(const TProxyDescriptor *other) {
      if ( !other ) return false;
      if ( fBranchName != other->fBranchName ) return false;
      if ( fIsSplit != other->fIsSplit ) return false;
      if ( strcmp(GetName(),other->GetName()) ) return false;
      if ( strcmp(GetTitle(),other->GetTitle()) ) return false;
      return true;
   }
   bool IsSplit() const { return fIsSplit; }

   void OutputDecl(FILE *hf, int offset, UInt_t maxVarname){
      fprintf(hf,"%-*s%-*s %s;\n",  offset," ",  maxVarname, GetTypeName(), GetDataName()); // might want to add a comment
   }

   void OutputInit(FILE *hf, int offset, UInt_t maxVarname,
                   const char *prefix) {
      if (fIsSplit) {
         const char *subbranchname = GetBranchName();
         const char *above = "";
         if (strncmp(prefix,subbranchname,strlen(prefix))==0
             && strcmp(prefix,subbranchname)!=0)  {
            subbranchname += strlen(prefix)+1; // +1 for the dot "."
            above = "ffPrefix, ";
         }

         fprintf(hf,"\n%-*s      %-*s(director, %s\"%s\")",
                 offset," ", maxVarname, GetName(), above, subbranchname);
      } else {

         fprintf(hf,"\n%-*s      %-*s(director, obj.proxy(), \"%s\")",
                 offset," ", maxVarname, GetName(), GetBranchName() );

         //fprintf(hf,"\n%-*s      %-*s(director, ffPrefix, \"\", \"%s\")",
         //        offset," ", maxVarname, GetName(), GetBranchName() );

      }
   }

   ClassDef(TProxyDescriptor,0);
};

class TProxyClassDescriptor : public TNamed {
public:
   typedef enum { kOut=0, kClones, kInsideClones } EInClones; // for IsClones
private:
   TList   fListOfSubProxies;
   TList   fListOfBaseProxies;
   UInt_t  fIsClones;   // 1 for the general case, 2 when this a split clases inside a TClonesArray.
   bool    fIsLeafList; // true if the branch was constructed from a leaf list.

   UInt_t  fSplitLevel;

   TString fRawSymbol;
   TString fBranchName;
   TString fSubBranchPrefix;

   UInt_t  fMaxDatamemberType;

   void NameToSymbol() {

      // Make the typename a proper class name without having the really deal with
      // namespace and templates.

      fRawSymbol = GetName();
      fRawSymbol.ReplaceAll(":","_");
      fRawSymbol.ReplaceAll("<","_");
      fRawSymbol.ReplaceAll(">","_");
      fRawSymbol.ReplaceAll(" ","");
      fRawSymbol.ReplaceAll("*","st");
      fRawSymbol.ReplaceAll("&","rf");
      if (fIsClones!=kOut)
         fRawSymbol.Prepend("TClaPx_");
      else
         fRawSymbol.Prepend("TPx_");

      SetName(fRawSymbol);
   }

public:

   TProxyClassDescriptor(const char *type, const char *branchname,
                         UInt_t isclones, UInt_t splitlevel) :
      TNamed(type,type),
      fIsClones(isclones),
      fIsLeafList(false),
      fSplitLevel(splitlevel),
      fBranchName(branchname),
      fSubBranchPrefix(branchname),
      fMaxDatamemberType(3)
   {
      NameToSymbol();
   }

   TProxyClassDescriptor(const char *branchname) :
      TNamed(branchname,branchname),
      fIsClones(false),
      fIsLeafList(true),
      fSplitLevel(0),
      fBranchName(branchname),
      fSubBranchPrefix(branchname),
      fMaxDatamemberType(3)
   {
      // Constructor for a branch constructed from a leaf list.
      NameToSymbol();
   }

   TProxyClassDescriptor(const char *type, const char *branchname,
                         const char *branchPrefix, UInt_t isclones,
                         UInt_t splitlevel) :
      TNamed(type,type),
      fIsClones(isclones),
      fIsLeafList(true),
      fSplitLevel(splitlevel),
      fBranchName(branchname),
      fSubBranchPrefix(branchPrefix),
      fMaxDatamemberType(3)
   {
      NameToSymbol();
   }

   const char* GetBranchName() const {
      return fBranchName.Data();
   }

   const char* GetSubBranchPrefix() const {
      return fSubBranchPrefix.Data();
   }

   const char* GetRawSymbol() const {
      return fRawSymbol;
   }

   UInt_t GetSplitLevel() const { return fSplitLevel; }
   using TObject::IsEqual;
   bool IsEqual(const TProxyClassDescriptor* other) {
      if ( !other ) return false;
      // Purposely do not test on the name!
      if ( strcmp(GetTitle(),other->GetTitle()) ) return false;
      // if ( fBranchName != other->fBranchName ) return false;
      // if ( fSubBranchPrefix != other->fSubBranchPrefix ) return false;

      if (fIsClones != other->fIsClones) return false;

      TProxyDescriptor *desc;
      TProxyDescriptor *othdesc;

      if ( fListOfBaseProxies.GetSize() != other->fListOfBaseProxies.GetSize() ) return false;
      TIter next(&fListOfBaseProxies);
      TIter othnext(&other->fListOfBaseProxies);
      while ( (desc=(TProxyDescriptor*)next()) ) {
         othdesc=(TProxyDescriptor*)othnext();
         if (!desc->IsEqual(othdesc) ) return false;
      }

      if ( fListOfSubProxies.GetSize() != other->fListOfSubProxies.GetSize() ) return false;
      next = &fListOfSubProxies;
      othnext = &(other->fListOfSubProxies);

      while ( (desc=(TProxyDescriptor*)next()) ) {
         othdesc=(TProxyDescriptor*)othnext();
         if (!desc->IsEqual(othdesc)) return false;
      }
      return true;
   }

   void AddDescriptor(TProxyDescriptor *desc, bool isBase) {
      if (desc) {
         if (isBase) {
            fListOfBaseProxies.Add(desc);
         } else {
            fListOfSubProxies.Add(desc);
            UInt_t len = strlen(desc->GetTypeName());
            if ((len+2)>fMaxDatamemberType) fMaxDatamemberType = len+2;
         }
      }
   }

   Bool_t IsLoaded() const {
      TClass *cl = gROOT->GetClass(GetTitle());
      return (cl && cl->IsLoaded());
   }

   Bool_t IsClones() const { return fIsClones!=kOut; }
   UInt_t GetIsClones() const { return fIsClones; }

   void OutputDecl(FILE *hf, int offset, UInt_t /* maxVarname */){
      // Output the declaration and implementation of this emulation class

      TProxyDescriptor *desc;


      // Start the class declaration with the eventual list of base classes
      fprintf(hf,"%-*sstruct %s\n", offset," ", GetName() );

      if (fListOfBaseProxies.GetSize()) {
         fprintf(hf,"%-*s   : ", offset," ");

         TIter next(&fListOfBaseProxies);

         desc = (TProxyDescriptor*)next();
         fprintf(hf,"public %s", desc->GetTypeName());

         while ( (desc = (TProxyDescriptor*)next()) ) {
            fprintf(hf,",\n%-*spublic %s", offset+5," ", desc->GetTypeName());
         }

         fprintf(hf,"\n");
      }
      fprintf(hf,"%-*s{\n", offset," ");


      // Write the constructor
      fprintf(hf,"%-*s   %s(TProxyDirector* director,const char *top,const char *mid=0) :",
              offset," ", GetName());

      bool wroteFirst = false;

      if (fListOfBaseProxies.GetSize()) {

         TIter next(&fListOfBaseProxies);

         desc = (TProxyDescriptor*)next();
         fprintf(hf,"\n%-*s%-*s(director, top, mid)",  offset+6, " ", fMaxDatamemberType,desc->GetTypeName());
         wroteFirst = true;

         while ( (desc = (TProxyDescriptor*)next()) ) {
            fprintf(hf,",\n%-*s%-*s(director, top, mid)",  offset+6, " ", fMaxDatamemberType,desc->GetTypeName());
         }

      }
      fprintf(hf,"%s\n%-*s      %-*s(top,mid)",wroteFirst?",":"",offset," ",fMaxDatamemberType,"ffPrefix");
      wroteFirst = true;


      TString objInit = "top, mid";
      if ( GetIsClones() == kInsideClones ) {
         if (fListOfSubProxies.GetSize()) {
            desc = (TProxyDescriptor*)fListOfSubProxies.At(0);
            if (desc && desc->IsSplit()) {

               // In the case of a split sub object is TClonesArray, the
               // object itself does not have its own branch, so we need to
               // use its first (semantic) sub-branch as a proxy

               TString main = GetBranchName();
               TString sub = desc->GetBranchName();
               sub.Remove(0,main.Length()+1);

               objInit  = "ffPrefix, ";
               objInit += "\"";
               objInit += sub;
               objInit += "\"";

               objInit = "top, \"\", mid";
            }
         }
      }

      fprintf(hf,"%s\n%-*s      %-*s(director, %s)",
              wroteFirst?",":"",offset," ",fMaxDatamemberType,"obj",objInit.Data());
      wroteFirst = true;

      TIter next(&fListOfSubProxies);
      while ( (desc = (TProxyDescriptor*)next()) ) {
         if (wroteFirst) fprintf(hf,",");
         desc->OutputInit(hf,offset,fMaxDatamemberType,GetSubBranchPrefix());
         wroteFirst = true;
      }
      fprintf(hf,"\n%-*s   {};\n",offset," ");


      // Write the 2nd constructor
      fprintf(hf,"%-*s   %s(TProxyDirector* director, TProxy *parent, const char *membername) :",
              offset," ", GetName());

      wroteFirst = false;

      if (fListOfBaseProxies.GetSize()) {

         TIter next(&fListOfBaseProxies);

         desc = (TProxyDescriptor*)next();
         fprintf(hf,"\n%-*s%-*s(director, parent, membername)",  offset+6, " ", fMaxDatamemberType,desc->GetTypeName());
         wroteFirst = true;

         while ( (desc = (TProxyDescriptor*)next()) ) {
            fprintf(hf,",\n%-*s%-*s(director, parent, membername)",  offset+6, " ", fMaxDatamemberType,desc->GetTypeName());
         }

      }
      fprintf(hf,"%s\n%-*s      %-*s(\"\")",wroteFirst?",":"",offset," ",fMaxDatamemberType,"ffPrefix");
      wroteFirst = true;

      if ( true ||  IsLoaded() || IsClones() ) {
         fprintf(hf,"%s\n%-*s      %-*s(director, parent, membername)",
                 wroteFirst?",":"",offset," ",fMaxDatamemberType,"obj");
         wroteFirst = true;
      }

      next = &fListOfSubProxies;
      while ( (desc = (TProxyDescriptor*)next()) ) {
         if (wroteFirst) fprintf(hf,",");
         desc->OutputInit(hf,offset,fMaxDatamemberType,GetSubBranchPrefix());
         wroteFirst = true;
      }
      fprintf(hf,"\n%-*s   {};\n",offset," ");


      // Declare the data members.
      fprintf(hf,"%-*s%-*s %s;\n",  offset+3," ",  fMaxDatamemberType, "TProxyHelper", "ffPrefix");

      // If the real class is available, make it available via the arrow operator:
      if (IsLoaded()) {

         const char *type = GetTitle(); /* IsClones() ? "TClonesArray" : GetTitle(); */
         fprintf(hf,"%-*sInjectProxyInterface();\n", offset+3," ");
         //Can the real type contain a leading 'const'? If so the following is incorrect.
         if ( IsClones() ) {
            fprintf(hf,"%-*sconst %s* operator[](int i) { return obj.at(i); }\n", offset+3," ",type);
            fprintf(hf,"%-*sTClaObjProxy<%s > obj;\n", offset+3, " ", type);
         } else {
            fprintf(hf,"%-*sconst %s* operator->() { return obj.ptr(); }\n", offset+3," ",type);
            fprintf(hf,"%-*sTObjProxy<%s > obj;\n", offset+3, " ", type);
         }

      } else if ( IsClones()) {

         fprintf(hf,"%-*sInjectProxyInterface();\n", offset+3," ");
         fprintf(hf,"%-*sconst TClonesArray* operator->() { return obj.ptr(); }\n", offset+3," ");
         fprintf(hf,"%-*sTClaProxy obj;\n", offset+3," ");

      } else {

         fprintf(hf,"%-*sInjectProxyInterface();\n", offset+3," ");
         fprintf(hf,"%-*sTProxy obj;\n", offset+3," ");

      }

      fprintf(hf,"\n");

      next.Reset();
      while( (desc = ( TProxyDescriptor *)next()) ) {
         desc->OutputDecl(hf,offset+3,fMaxDatamemberType);
      }
      fprintf(hf,"%-*s};\n",offset," ");

      //TProxyDescriptor::OutputDecl(hf,offset,maxVarname);
   }
   ClassDef(TProxyClassDescriptor,0);
};

class TGenerateProxy /* change this name please */
{
public:
   enum EContainer { kNone, kClones };
   UInt_t   fMaxDatamemberType;
   TString  fScript;
   TString  fCutScript;
   TString  fPrefix;
   TString  fHeaderFilename;
   UInt_t   fMaxUnrolling;
   TTree   *fTree;
   TList    fListOfHeaders;
   TList    fListOfClasses;
   TList    fListOfTopProxies;
   TList    fListOfForwards;
   TGenerateProxy(TTree* tree, const char *script, const char *fileprefix, UInt_t maxUnrolling);
   TGenerateProxy(TTree* tree, const char *script, const char *cutscript, const char *fileprefix, UInt_t maxUnrolling);

   TProxyClassDescriptor* AddClass(TProxyClassDescriptor *desc);
   void AddForward(TClass *cl);
   void AddForward(const char *classname);
   void AddHeader(TClass *cl);
   void AddHeader(const char *classname);
   void AddDescriptor(TProxyDescriptor *desc);

   bool NeedToEmulate(TClass *cl, UInt_t level);

   UInt_t AnalyzeBranch(TBranch *branch, UInt_t level, TProxyClassDescriptor *desc);
   UInt_t AnalyzeOldBranch(TBranch *branch, UInt_t level, TProxyClassDescriptor *desc);
   UInt_t AnalyzeOldLeaf(TLeaf *leaf, UInt_t level, TProxyClassDescriptor *topdesc);
   void   AnalyzeElement(TBranch *branch, TStreamerElement *element, UInt_t level, TProxyClassDescriptor *desc, const char* path);
   void   AnalyzeTree();
   void   WriteProxy();

   const char *GetFilename() { return fHeaderFilename; }
};

#include "TBranchElement.h"
#include "TChain.h"
#include "TFile.h"
#include "TLeaf.h"
#include "TTree.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TSystem.h"
#include "TLeafObject.h"

TString GetArrayType(TStreamerElement *element, const char *subtype, TGenerateProxy::EContainer container)
{
   TString result;
   int ndim = 0;
   if (element && element->InheritsFrom(TStreamerBasicPointer::Class())) {
      TStreamerBasicPointer * elem = (TStreamerBasicPointer*)element;
      const char *countname = elem->GetCountName();
      if (countname && strlen(countname)>0) ndim = 1;
   }
   ndim += element->GetArrayDim();

   TString middle;
   if (container == TGenerateProxy::kClones) {
      middle = "Cla";
   }

   if (ndim==0) {
      result = "T";
      result += middle;
      result += subtype;
      result += "Proxy";
   } else if (ndim==1) {
      result = "T";
      result += middle;
      result += "Array";
      result += subtype;
      result += "Proxy";
   } else if (ndim==2) {
      result = "T";
      result += middle;
      result += "Array2Proxy<";
      result += element->GetTypeName();
      result += ",";
      result += element->GetMaxIndex(1);
      result += " >";
   }  else if (ndim==3) {
      result = "T";
      result += middle;
      result += "Array3Proxy<";
      result += element->GetTypeName();
      result += ",";
      result += element->GetMaxIndex(1);
      result += ",";
      result += element->GetMaxIndex(2);
      result += " >";
   } else {
      fprintf(stderr,"array of more than 3 dimentsions not implemented yet\n");
   }
   return result;

   /*
   if (!strcmp("unsigned int", name))
      sprintf(line, "%u", *(unsigned int *)buf);
   else if (!strcmp("int", name))
      sprintf(line, "%d", *(int *)buf);
   else if (!strcmp("unsigned long", name))
      sprintf(line, "%lu", *(unsigned long *)buf);
   else if (!strcmp("long", name))
      sprintf(line, "%ld", *(long *)buf);
   else if (!strcmp("unsigned short", name))
      sprintf(line, "%hu", *(unsigned short *)buf);
   else if (!strcmp("short", name))
      sprintf(line, "%hd", *(short *)buf);
   else if (!strcmp("unsigned char", name))
      sprintf(line, "%u", *(unsigned char *)buf);
   else if (!strcmp("bool", name))
      sprintf(line, "%u", *(unsigned char *)buf);
   else if (!strcmp("char", name))
      sprintf(line, "%d", *(char *)buf);
   else if (!strcmp("float", name))
      sprintf(line, "%g", *(float *)buf);
   else if (!strcmp("double", name))
      sprintf(line, "%g", *(double *)buf);
   */
}

TGenerateProxy::TGenerateProxy(TTree* tree, const char *script, const char *fileprefix, UInt_t maxUnrolling) :
   fMaxDatamemberType(2),
   fScript(script),
   fCutScript(),
   fPrefix(fileprefix),
   fHeaderFilename(),
   fMaxUnrolling(maxUnrolling),
   fTree(tree)
{

   AnalyzeTree();

   WriteProxy();
}

TGenerateProxy::TGenerateProxy(TTree* tree, const char *script, const char *cutscript, const char *fileprefix, UInt_t maxUnrolling) :
   fMaxDatamemberType(2),
   fScript(script),
   fCutScript(cutscript),
   fPrefix(fileprefix),
   fHeaderFilename(),
   fMaxUnrolling(maxUnrolling),
   fTree(tree)
{

   AnalyzeTree();

   WriteProxy();
}

Bool_t TGenerateProxy::NeedToEmulate(TClass *cl, UInt_t /* level */) {
   // Return true if we should create a nested class representing this class

   return cl->TestBit(TClass::kIsEmulation);
}

TProxyClassDescriptor*
TGenerateProxy::AddClass( TProxyClassDescriptor* desc )
{
   if (desc==0) return 0;

   TProxyClassDescriptor *existing =
      (TProxyClassDescriptor*)fListOfClasses(desc->GetName());

   int count = 0;
   while (existing) {
      if (! existing->IsEqual( desc )  ) {
         TString newname = desc->GetRawSymbol();
         count++;
         newname += "_";
         newname += count;

         desc->SetName(newname);
         existing = (TProxyClassDescriptor*)fListOfClasses(desc->GetName());
      } else {
         // we already have the exact same class
         delete desc;
         return existing;
      }
   }
   fListOfClasses.Add(desc);
   return desc;

}

void TGenerateProxy::AddForward( const char *classname )
{
   TObject *obj = fListOfForwards.FindObject(classname);
   if (obj) return;

   if (strstr(classname,"<")!=0) {
      // this is a template instantiation.
      // let's ignore it for now

      fprintf(stderr,"forward declaration of templated class not implemented yet\n");
   } else {
      fListOfForwards.Add(new TNamed(classname,Form("class %s;\n",classname)));
   }
   return;
}

void TGenerateProxy::AddForward(TClass *cl)
{
   if (cl) AddForward(cl->GetName());
}

void TGenerateProxy::AddHeader(TClass *cl) {
   fprintf(stderr,"adding header info for %s\n",cl?cl->GetName():"unknown class");
   if (cl==0) return;

   TObject *obj = fListOfHeaders.FindObject(cl->GetName());
   if (obj) return;

   if (cl->GetDeclFileName() && strlen(cl->GetDeclFileName()) ) {
      // Actually we probably should look for the file ..
      TString header = gSystem->BaseName(cl->GetDeclFileName());
      fListOfHeaders.Add(new TNamed(cl->GetName(),Form("#include \"%s\"\n",
                                                       header.Data())));
   }

}

void TGenerateProxy::AddHeader(const char *classname) {
   AddHeader(gROOT->GetClass(classname));
}

void TGenerateProxy::AddDescriptor(TProxyDescriptor *desc) {
   if (desc) {
      fListOfTopProxies.Add(desc);
      UInt_t len = strlen(desc->GetTypeName());
      if ((len+2)>fMaxDatamemberType) fMaxDatamemberType = len+2;
   }

}

UInt_t TGenerateProxy::AnalyzeBranch(TBranch *branch, UInt_t level, TProxyClassDescriptor *topdesc)
{
   // Analyze the branch and populate the TGenerateProxy or the topdesc with
   // its findings.  Sometimes several branch of the mom are also analyzed,
   // the number of such branches is returned (this happens in the case of
   // embedded objects inside an object inside a clones array split more than
   // one level.

   TString proxyTypeName;
   TString prefix;
   bool isBase = false;
   TString dataMemberName;
   TString cname;
   TString middle;
   UInt_t  extraLookedAt = 0;
   Bool_t  isclones = false;
   EContainer container = kNone;

   if (topdesc && topdesc->IsClones()) {
      container = kClones;
      middle = "Cla";
      isclones = true;
   }

   if (branch->IsA()==TBranchElement::Class()) {

      TBranchElement *be = (TBranchElement*)branch;

      Int_t bid = be->GetID();

      TStreamerElement *element = 0;
      TStreamerInfo *info = be->GetInfo();

      if (bid==-2) {
         fprintf(stderr,"support for branch ID: %d not yet implement\n",
                 bid);
      } else if (bid==-1) {
         fprintf(stderr,"support for branch ID: %d not yet implement\n",
                 bid);
      } else if (bid>=0) {

         element = (TStreamerElement *)info->GetElements()->At(bid);

      } else {
         fprintf(stderr,"support for branch ID: %d not yet implement\n",
                 bid);
      }

      if (element) {
         bool ispointer = false;
         switch(element->GetType()) {

            case TStreamerInfo::kChar:    { proxyTypeName = "T" + middle + "CharProxy"; break; }
            case TStreamerInfo::kShort:   { proxyTypeName = "T" + middle + "ShortProxy"; break; }
            case TStreamerInfo::kInt:     { proxyTypeName = "T" + middle + "IntProxy"; break; }
            case TStreamerInfo::kLong:    { proxyTypeName = "T" + middle + "LongProxy"; break; }
            case TStreamerInfo::kFloat:   { proxyTypeName = "T" + middle + "FloatProxy"; break; }
            case TStreamerInfo::kDouble:  { proxyTypeName = "T" + middle + "DoubleProxy"; break; }
            case TStreamerInfo::kDouble32:{ proxyTypeName = "T" + middle + "DoubleProxy"; break; }
            case TStreamerInfo::kUChar:   { proxyTypeName = "T" + middle + "UCharProxy"; break; }
            case TStreamerInfo::kUShort:  { proxyTypeName = "T" + middle + "UShortProxy"; break; }
            case TStreamerInfo::kUInt:    { proxyTypeName = "T" + middle + "UIntProxy"; break; }
            case TStreamerInfo::kULong:   { proxyTypeName = "T" + middle + "ULongProxy"; break; }
            case TStreamerInfo::kBits:    { proxyTypeName = "T" + middle + "UIntProxy"; break; }

            case TStreamerInfo::kCharStar: { proxyTypeName = GetArrayType(element,"Char",container); break; }

            // array of basic types  array[8]
            case TStreamerInfo::kOffsetL + TStreamerInfo::kChar:    { proxyTypeName = GetArrayType(element,"Char",container ); break; }
            case TStreamerInfo::kOffsetL + TStreamerInfo::kShort:   { proxyTypeName = GetArrayType(element,"Short",container ); break; }
            case TStreamerInfo::kOffsetL + TStreamerInfo::kInt:     { proxyTypeName = GetArrayType(element,"Int",container ); break; }
            case TStreamerInfo::kOffsetL + TStreamerInfo::kLong:    { proxyTypeName = GetArrayType(element,"Long",container ); break; }
            case TStreamerInfo::kOffsetL + TStreamerInfo::kFloat:   { proxyTypeName = GetArrayType(element,"Float",container ); break; }
            case TStreamerInfo::kOffsetL + TStreamerInfo::kDouble:  { proxyTypeName = GetArrayType(element,"Double",container ); break; }
            case TStreamerInfo::kOffsetL + TStreamerInfo::kDouble32:{ proxyTypeName = GetArrayType(element,"Double",container ); break; }
            case TStreamerInfo::kOffsetL + TStreamerInfo::kUChar:   { proxyTypeName = GetArrayType(element,"UChar",container ); break; }
            case TStreamerInfo::kOffsetL + TStreamerInfo::kUShort:  { proxyTypeName = GetArrayType(element,"UShort",container ); break; }
            case TStreamerInfo::kOffsetL + TStreamerInfo::kUInt:    { proxyTypeName = GetArrayType(element,"UInt",container ); break; }
            case TStreamerInfo::kOffsetL + TStreamerInfo::kULong:   { proxyTypeName = GetArrayType(element,"ULong",container ); break; }
            case TStreamerInfo::kOffsetL + TStreamerInfo::kBits:    { proxyTypeName = GetArrayType(element,"UInt",container ); break; }

            // pointer to an array of basic types  array[n]
            case TStreamerInfo::kOffsetP + TStreamerInfo::kChar:    { proxyTypeName = GetArrayType(element,"Char",container ); break; }
            case TStreamerInfo::kOffsetP + TStreamerInfo::kShort:   { proxyTypeName = GetArrayType(element,"Short",container ); break; }
            case TStreamerInfo::kOffsetP + TStreamerInfo::kInt:     { proxyTypeName = GetArrayType(element,"Int",container ); break; }
            case TStreamerInfo::kOffsetP + TStreamerInfo::kLong:    { proxyTypeName = GetArrayType(element,"Long",container ); break; }
            case TStreamerInfo::kOffsetP + TStreamerInfo::kFloat:   { proxyTypeName = GetArrayType(element,"Float",container ); break; }
            case TStreamerInfo::kOffsetP + TStreamerInfo::kDouble:  { proxyTypeName = GetArrayType(element,"Double",container ); break; }
            case TStreamerInfo::kOffsetP + TStreamerInfo::kDouble32:{ proxyTypeName = GetArrayType(element,"Double",container ); break; }
            case TStreamerInfo::kOffsetP + TStreamerInfo::kUChar:   { proxyTypeName = GetArrayType(element,"UChar",container ); break; }
            case TStreamerInfo::kOffsetP + TStreamerInfo::kUShort:  { proxyTypeName = GetArrayType(element,"UShort",container ); break; }
            case TStreamerInfo::kOffsetP + TStreamerInfo::kUInt:    { proxyTypeName = GetArrayType(element,"UInt",container ); break; }
            case TStreamerInfo::kOffsetP + TStreamerInfo::kULong:   { proxyTypeName = GetArrayType(element,"ULong",container ); break; }
            case TStreamerInfo::kOffsetP + TStreamerInfo::kBits:    { proxyTypeName = GetArrayType(element,"UInt",container ); break; }

            // array counter //[n]
            case TStreamerInfo::kCounter: { proxyTypeName = "T" + middle + "IntProxy"; break; }


            case TStreamerInfo::kOffsetL + TStreamerInfo::kObjectp:
            case TStreamerInfo::kOffsetL + TStreamerInfo::kObjectP:
            case TStreamerInfo::kObjectp:
            case TStreamerInfo::kObjectP:
            case TStreamerInfo::kAnyp:
            case TStreamerInfo::kAnyP:
               // set as pointers and fall through to the next switches
               ispointer = true;
            case TStreamerInfo::kOffsetL + TStreamerInfo::kObject:
            case TStreamerInfo::kObject:
            case TStreamerInfo::kTString:
            case TStreamerInfo::kTNamed:
            case TStreamerInfo::kTObject:
            case TStreamerInfo::kAny: {
               TClass *cl = element->GetClassPointer();
               if (cl) {
                  proxyTypeName = Form("T%sObjProxy<%s >", middle.Data(), cl->GetName());
                  cname = cl->GetName();
                  if (cl==TClonesArray::Class()) {
                     isclones = true;
                     cname = be->GetClonesName();
                     if (cname.Length()==0) {
                        // We may have any unsplit clones array
                        Int_t i = branch->GetTree()->GetReadEntry();
                        if (i<0) i = 0;
                        branch->GetEntry(i);
                        char *obj = be->GetObject();

                        const char *ename = 0;
                        ename = element->GetName();


                        TBranchElement *parent = (TBranchElement*)be->GetMother()->GetSubBranch(be);
                        const char *pclname = parent->GetClassName();

                        TClass *clparent = gROOT->GetClass(pclname);
                        // TClass *clm = gROOT->GetClass(GetClassName());
                        Int_t lOffset; // offset in the local streamerInfo.
                        if (clparent) lOffset = clparent->GetStreamerInfo()->GetOffset(ename);
                        else fprintf(stderr,"missing parent for %s\n",branch->GetName());

                        TClonesArray *arr;
                        if (ispointer) {
                           arr = (TClonesArray*)*(void**)(obj+lOffset);
                        } else {
                           arr = (TClonesArray*)(obj+lOffset);
                        }
                        cname = arr->GetClass()->GetName();

                     }
                     if (cname.Length()==0) {
                        fprintf(stderr,"introspection of TClonesArray in older file not implemented yet\n");
                     }
                  }
               }
               else fprintf(stderr,"missing class for %s\n",branch->GetName());
               if (element->IsA()==TStreamerBase::Class()) {
                  isBase = true;
                  prefix  = "base";
               }
               AddForward(cl);
               AddHeader(cl);
               break;
            }

            default:
               fprintf(stderr,"Unsupported type for %s (%d)\n",branch->GetName(),element->GetType());

         }

      }

   } else {

      fprintf(stderr,"non TBranchElement not implemented yet in AnalyzeBranch (this should not happen)\n");
      return extraLookedAt;

   }

   if ( branch->GetListOfBranches()->GetEntries() > 0 ) {
      // The branch has sub-branch corresponding the split data member of a class


      // See AnalyzeTree for similar code!
      TProxyClassDescriptor *cldesc;

      TClass *cl = gROOT->GetClass(cname);
      if (cl) {
         cldesc = new TProxyClassDescriptor(cl->GetName(), branch->GetName(), isclones, branch->GetSplitLevel());
      }
      //fprintf(stderr,"nesting br %s of class %s and type %s\n",
      //        branchname,cname.Data(),type.Data());

      if (cldesc) {
         TBranch *subbranch;
         TIter subnext( branch->GetListOfBranches() );
         while ( (subbranch = (TBranch*)subnext()) ) {
            Int_t skipped = AnalyzeBranch(subbranch,level+1,cldesc);
            Int_t s = 0;
            while( s<skipped && subnext() ) { s++; };
         }

         TProxyClassDescriptor *added = AddClass(cldesc);
         if (added) proxyTypeName = added->GetName();
         // this codes and the previous 2 lines move from inside the if (cl)
         // aboce and this line was used to avoid unecessary work:
         // if (added!=cldesc) cldesc = 0;
      }


   } else if ( cname.Length() ) {
      // The branch contains a non-split object that we are unfolding!

      // See AnalyzeTree for similar code!
      TProxyClassDescriptor *cldesc;

      TClass *cl = gROOT->GetClass(cname);
      if (cl) {
         cldesc = new TProxyClassDescriptor(cl->GetName(), branch->GetName(), isclones, 0 /* unsplit object */);
      }
      if (cldesc && cl) {
         TStreamerInfo *info = cl->GetStreamerInfo();
         TStreamerElement *elem = 0;

         TIter next(info->GetElements());
         while( (elem = (TStreamerElement*)next()) ) {
            AnalyzeElement(branch,elem,level+1,cldesc,"");
         }

         TProxyClassDescriptor *added = AddClass(cldesc);
         if (added) proxyTypeName = added->GetName();
         // this codes and the previous 2 lines move from inside the if (cl)
         // aboce and this line was used to avoid unecessary work:
         // if (added!=cldesc) cldesc = 0;
      }

   }

   TLeaf *leaf = (TLeaf*)branch->GetListOfLeaves()->At(0);

   if (leaf && strlen(leaf->GetTypeName()) == 0) {
      // no type is know for the first leaf (which case is that?)
      return extraLookedAt;
   }

   if (leaf && proxyTypeName.Length()==0) {
      proxyTypeName = leaf->GetTypeName() ;
   }

   if (leaf && !isclones) dataMemberName = leaf->GetName();
   else dataMemberName = branch->GetName();

   Int_t pos;
   pos = dataMemberName.Index(".");
   if (pos != -1) {
      dataMemberName.Remove(0,pos+1);
   }
   pos = dataMemberName.Index("[");
   if (pos != -1) {
      dataMemberName.Remove(pos);
   }
   pos = dataMemberName.Index(".");

   TString branchName = branch->GetName();

   if (pos != -1 && container==kClones && branch->IsA()==TBranchElement::Class()) {
      // We still have a "." in the name, we assume that we are in the case
      // where we reach an embedded object in the object contained in the
      // TClonesArray

      // Discover the type of this object.
      TString name = dataMemberName(0,pos);

      TBranchElement *mom = (TBranchElement*)branch->GetMother()->GetSubBranch(branch);
      TString cname = mom->GetClonesName();
      TString prefix = mom->GetName();
      prefix += ".";
      prefix += name;
      // prefix += ".";


      if ( topdesc && strcmp(topdesc->GetBranchName(),prefix.Data())==0 ) {

         // Assume we coming recursively from the previous case!
         dataMemberName.Remove(0,pos+1);

      } else {

         TStreamerElement* branchStreamerElem = 0;

         TStreamerInfo *momInfo = mom->GetInfo();
         if (cname != momInfo->GetName()) {
            // We do not have the correct TStreamerInfo, this is
            // because there is no proper 'branch' holding this sub-object
            // they are all stored in the branch for the owner of the object

            // We need to discover if 'branch' represents a direct datamember
            // of the class in 'mom' or if it is an indirect one (with a
            // missing branch in the hierachy)

            TClass *momBranchClass = TClass::GetClass(cname);

            // We are in the case where there is a missing branch in the hiearchy

            momInfo = momBranchClass->GetStreamerInfo();

            // remove the main branch name (if present)
            TString parentDataName = branch->GetName();
            if (parentDataName.Index(mom->GetName())==0) {
               parentDataName.Remove(0,strlen(mom->GetName()));
               if (parentDataName[0]=='.') {
                  parentDataName.Remove(0,1);
               }
            }
            
            // remove the current data member name
            Ssiz_t pos = parentDataName.Last('.');
            if (pos>0) {
               // We had a branch name of the style:
               //     [X.]Y.Z
               // and we are looking up 'Y'
               parentDataName.Remove(pos);
               branchStreamerElem = (TStreamerElement*)momInfo->GetElements()->FindObject( parentDataName.Data() );
               
            } else {
               // We had a branch name of the style:
               //     [X.]Z
               // and we are looking up 'Z'
               // Because we are missing 'Y' (or more exactly the name of the
               // thing that contains Z, ... we don't know what do yet.
               
               fprintf(stderr,"The case of the branch '%s' is not implemented yet.\nPlease send your data file to the root developers\n",
                       branch->GetName());
            }

         } else {

            branchStreamerElem = (TStreamerElement*)
               momInfo->GetElements()->FindObject(name.Data());

         }


         if (branchStreamerElem==0) {
            fprintf(stderr,"ERROR: We did not find %s when looking into %s.\n",
                    name.Data(),mom->GetName());
//             mom->GetInfo()->Print();
            return extraLookedAt;
         } else {
//             fprintf(stderr,"SUCC: We did find %s when looking into %s %p.\n",
//                     name.Data(),mom->GetName(),branchStreamerElem);
//             mom->GetInfo()->Print();
         }

         TClass *cl = branchStreamerElem->GetClassPointer();

         cname = cl->GetName();
         proxyTypeName = Form("TClaObjProxy<%s >",cname.Data());

         TProxyClassDescriptor *cldesc;

         cldesc = new TProxyClassDescriptor( cl->GetName(), prefix.Data(), prefix.Data(),
                                             TProxyClassDescriptor::kInsideClones, 
                                             branch->GetSplitLevel()-1);

         TIter next(mom->GetListOfBranches());
         TBranch *subbranch;
         while ( (subbranch = (TBranch*)next()) && subbranch!=branch ) {};

         Assert( subbranch == branch );

         do {
            TString subname = subbranch->GetName();
            if ( subname.BeginsWith( prefix ) ) {
               Int_t skipped = 0;
               if (cldesc) {

                  skipped = AnalyzeBranch( subbranch, level+1, cldesc);
                  Int_t s = 0;
                  while( s<skipped && next() ) { s++; };

               }
               extraLookedAt += 1 + skipped;
            } else {
               break;
            }
         } while ( (subbranch = (TBranch*)next()) );

         dataMemberName.Remove(pos);
         //fprintf(stderr,"will use %s\n", dataMemberName.Data());

         // this codes and the previous 2 lines move from inside the if (cl)
         // aboce and this line was used to avoid unecessary work:
         TProxyClassDescriptor *added = AddClass(cldesc);
         if (added) proxyTypeName = added->GetName();
         // if (added!=cldesc) cldesc = 0;

         pos = branchName.Last('.');
         if (pos != -1) {
            branchName.Remove(pos);
         }

      }
   }

   TProxyDescriptor *desc;
   if (topdesc) {
      topdesc->AddDescriptor( desc = new TProxyDescriptor( dataMemberName.Data(), proxyTypeName, branchName.Data() ), isBase );
   } else {
      dataMemberName.Prepend(prefix);
      AddDescriptor( desc = new TProxyDescriptor( dataMemberName.Data(), proxyTypeName, branchName.Data() ) );
   }
   //fprintf(stderr,"%-*s      %-*s(director,\"%s\")\n",
   //        0," ",10,desc->GetName(), desc->GetBranchName());
   return extraLookedAt;
}

UInt_t TGenerateProxy::AnalyzeOldLeaf(TLeaf *leaf, UInt_t /* level */, TProxyClassDescriptor *topdesc)
{
   // Analyze the leaf and populate the TGenerateProxy or the topdesc with
   // its findings.

   if (leaf->IsA()==TLeafObject::Class()) {
      fprintf(stderr,"We do not support TLeafObject yet");
      return 0;
   }

   TString leafTypeName = leaf->GetTypeName();
   Int_t pos = leafTypeName.Last('_');
   if (pos!=-1) leafTypeName.Remove(pos);

   Int_t len = leaf->GetLen();
   TLeaf *leafcount = leaf->GetLeafCount();

   UInt_t dim = 0;
   Int_t maxDim[3];
   maxDim[0] = maxDim[1] = maxDim[2] = 1;

   TString dimensions;
   TString temp = leaf->GetName();
   pos = temp.Index("[");
   if (pos!=-1) {
      if (pos) temp.Remove(0,pos);
      dimensions.Append(temp);
   }
   temp = leaf->GetTitle();
   pos = temp.Index("[");
   if (pos!=-1) {
      if (pos) temp.Remove(0,pos);
      dimensions.Append(temp);
   }

   Int_t dimlen = dimensions.Length();

   if (dimlen) {
      // fprintf(stderr,"dimension not implemented yet: %s from %s %s\n",dimensions.Data(),leaf->GetName(),leaf->GetTitle());
      const char *current = dimensions.Data();

      Int_t index;
      Int_t scanindex ;
      while (current) {
         current++;
         if (current[0] == ']') {
            maxDim[dim] = -1; // Loop over all elements;
         } else {
            scanindex = sscanf(current,"%d",&index);
            if (scanindex) {
               maxDim[dim] = index;
            } else {
               maxDim[dim] = -2; // Index is calculated via a variable.
            }
         }
         dim ++;
         if (dim >= 3) {
            // NOTE: test that dim this is NOT too big!!
            break;
         }
         current = (char*)strstr( current, "[" );
      }

   }
   //char *twodim = (char*)strstr(leaf->GetTitle(),"][");

   if (leafcount) {
      len = leafcount->GetMaximum();
   }


   TString type;
   switch (dim) {
      case 0: {
         type = "T";
         type += leafTypeName;
         type += "Proxy";
         break;
      }
      case 1: {
         type = "TArray";
         type += leafTypeName;
         type += "Proxy";
         break;
      }
      case 2: {
         type = "TArray2Proxy<";
         type += leaf->GetTypeName();
         type += ",";
         type += maxDim[1];
         type += " >";
         break;
      }
      case 3: {
         type = "TArray3Proxy<";
         type += leaf->GetTypeName();
         type += ",";
         type += maxDim[1];
         type += ",";
         type += maxDim[2];
         type += " >";
         break;
      }
      default:  {
         fprintf(stderr,"array of more than 3 dimentsions not implemented yet\n");
         return 0;
      }
   }

   TString branchName = leaf->GetBranch()->GetName();
   TProxyDescriptor *desc;
   if (topdesc) {
      topdesc->AddDescriptor( desc = new TProxyDescriptor( branchName.Data(), type, branchName.Data() ), 0 );
   } else {
      // leafname.Prepend(prefix);
      AddDescriptor( desc = new TProxyDescriptor( branchName.Data(), type, branchName.Data() ) );
   }

   return 0;

}

UInt_t TGenerateProxy::AnalyzeOldBranch(TBranch *branch, UInt_t level, TProxyClassDescriptor *topdesc)
{
   // Analyze the branch and populate the TGenerateProxy or the topdesc with
   // its findings.  Sometimes several branch of the mom are also analyzed,
   // the number of such branches is returned (this happens in the case of
   // embedded objects inside an object inside a clones array split more than
   // one level.

   UInt_t extraLookedAt = 0;
   TString prefix;

   TString branchName = branch->GetName();

   TObjArray *leaves = branch->GetListOfLeaves();
   Int_t nleaves = leaves ? leaves->GetEntriesFast() : 0;

   if (nleaves>1) {

      // Create a holder
      TString type = "unknown";
      TProxyClassDescriptor *cldesc = new TProxyClassDescriptor(branch->GetName());
      TProxyClassDescriptor *added = AddClass(cldesc);
      if (added) type = added->GetName();

      for(int l=0;l<nleaves;l++) {
         TLeaf *leaf = (TLeaf*)leaves->UncheckedAt(l);
         extraLookedAt += AnalyzeOldLeaf(leaf,level+1,cldesc);
      }

      TProxyDescriptor *desc;
      if (topdesc) {
         topdesc->AddDescriptor( desc = new TProxyDescriptor( branchName.Data(), type, branchName.Data() ), 0 );
      } else {
         // leafname.Prepend(prefix);
         AddDescriptor( desc = new TProxyDescriptor( branchName.Data(), type, branchName.Data() ) );
      }

   } else {

      TLeaf *leaf = (TLeaf*)leaves->UncheckedAt(0);
      extraLookedAt += AnalyzeOldLeaf(leaf,level,topdesc);

   }


   return extraLookedAt;

}

void TGenerateProxy::AnalyzeTree() {

   TIter next( fTree->GetListOfBranches() );
   TBranch *branch;
   while ( (branch = (TBranch*)next()) ) {
      const char *branchname = branch->GetName();
      const char *classname = branch->GetClassName();
      if (classname && strlen(classname)) {
         AddForward( classname );
         AddHeader( classname );
      }

      TProxyClassDescriptor *desc = 0;
      TClass *cl = gROOT->GetClass(classname);
      TString type = "unknown";
      if (cl) {
         Bool_t isclones = false;
         if (cl==TClonesArray::Class()) {
            isclones = true;
            if (branch->IsA()==TBranchElement::Class()) {
               const char *cname = ((TBranchElement*)branch)->GetClonesName();
               TClass *ncl = gROOT->GetClass(cname);
               if (ncl) {
                  cl = ncl;
               } else {
                  fprintf(stderr,"introspection of TClonesArray in older file not implemented yet\n");
               }
            } else {
               fprintf(stderr,"introspection of TClonesArray in older file not implemented yet\n");
            }

         }
         if (NeedToEmulate(cl,0) || branchname[strlen(branchname)-1] == '.' ) {
            desc = new TProxyClassDescriptor(cl->GetName(), branchname, isclones, branch->GetSplitLevel());
         } else {
            type = Form("TObjProxy<%s >",cl->GetName());
         }
      }

      if ( branch->GetListOfBranches()->GetEntries() == 0 ) {

         if (cl) {
            // We have a non-splitted object!

            TStreamerInfo *info = cl->GetStreamerInfo();
            TStreamerElement *elem = 0;

            TIter next(info->GetElements());
            while( (elem = (TStreamerElement*)next()) ) {
               AnalyzeElement(branch,elem,1,desc,"");
            }

            desc = AddClass(desc);
            type = desc->GetName();
            AddDescriptor( new TProxyDescriptor( branchname, type, branchname ) );
         } else {

            // We have a top level raw type.

            AnalyzeOldBranch(branch, 0, 0);

         }

      } else {

         // We have a splitted object

         TBranch *subbranch;
         TIter subnext( branch->GetListOfBranches() );
         UInt_t skipped = 0;
         if (desc) {
            while ( (subbranch = (TBranch*)subnext()) ) {
               skipped = AnalyzeBranch(subbranch,1,desc);
               if (skipped != 0) fprintf(stderr,"unexpectly read more than one branch in AnalyzeTree\n");
            }
         }
         desc = AddClass(desc);
         type = desc->GetName();
         AddDescriptor( new TProxyDescriptor( branchname, type, branchname ) );

         if ( branchname[strlen(branchname)-1] != '.' ) {
            // If there is no dot also included the data member directly
            subnext.Reset();
            while ( (subbranch = (TBranch*)subnext()) ) {
               skipped = AnalyzeBranch(subbranch,1,0);
               if (skipped != 0) fprintf(stderr,"unexpectly read more than one branch in AnalyzeTree\n");
            }
         }

      } // if split or non split
   }

}

void TGenerateProxy::AnalyzeElement(TBranch *branch, TStreamerElement *element,
                                    UInt_t level, TProxyClassDescriptor *topdesc,
                                    const char *path)
{
   // Analyze the element and populate the TGenerateProxy or the topdesc with
   // its findings.

   TString dataMemberName;
   TString pxDataMemberName;
   TString type;

   // TString prefix;
   bool isBase = false;
   TString cname;
   TString middle;
   Bool_t  isclones = false;
   EContainer container = kNone;
   if (topdesc && topdesc->IsClones()) {
      container = kClones;
      middle = "Cla";
      isclones = true;
   }

   if (!element) return;
   bool ispointer = false;
   switch(element->GetType()) {

      case TStreamerInfo::kChar:  { type = "T" + middle + "CharProxy"; break; }
      case TStreamerInfo::kShort: { type = "T" + middle + "ShortProxy"; break; }
      case TStreamerInfo::kInt:   { type = "T" + middle + "IntProxy"; break; }
      case TStreamerInfo::kLong:  { type = "T" + middle + "LongProxy"; break; }
      case TStreamerInfo::kFloat: { type = "T" + middle + "FloatProxy"; break; }
      case TStreamerInfo::kDouble:{ type = "T" + middle + "DoubleProxy"; break; }
      case TStreamerInfo::kUChar: { type = "T" + middle + "UCharProxy"; break; }
      case TStreamerInfo::kUShort:{ type = "T" + middle + "UShortProxy"; break; }
      case TStreamerInfo::kUInt:  { type = "T" + middle + "UIntProxy"; break; }
      case TStreamerInfo::kULong: { type = "T" + middle + "ULongProxy"; break; }
      case TStreamerInfo::kBits:  { type = "T" + middle + "UIntProxy"; break; }

      case TStreamerInfo::kCharStar: { type = GetArrayType(element,"Char",container); break; }

         // array of basic types  array[8]
      case TStreamerInfo::kOffsetL + TStreamerInfo::kChar:  { type = GetArrayType(element,"Char",container ); break; }
      case TStreamerInfo::kOffsetL + TStreamerInfo::kShort: { type = GetArrayType(element,"Short",container ); break; }
      case TStreamerInfo::kOffsetL + TStreamerInfo::kInt:   { type = GetArrayType(element,"Int",container ); break; }
      case TStreamerInfo::kOffsetL + TStreamerInfo::kLong:  { type = GetArrayType(element,"Long",container ); break; }
      case TStreamerInfo::kOffsetL + TStreamerInfo::kFloat: { type = GetArrayType(element,"Float",container ); break; }
      case TStreamerInfo::kOffsetL + TStreamerInfo::kDouble:{ type = GetArrayType(element,"Double",container ); break; }
      case TStreamerInfo::kOffsetL + TStreamerInfo::kUChar: { type = GetArrayType(element,"UChar",container ); break; }
      case TStreamerInfo::kOffsetL + TStreamerInfo::kUShort:{ type = GetArrayType(element,"UShort",container ); break; }
      case TStreamerInfo::kOffsetL + TStreamerInfo::kUInt:  { type = GetArrayType(element,"UInt",container ); break; }
      case TStreamerInfo::kOffsetL + TStreamerInfo::kULong: { type = GetArrayType(element,"ULong",container ); break; }
      case TStreamerInfo::kOffsetL + TStreamerInfo::kBits:  { type = GetArrayType(element,"UInt",container ); break; }

         // pointer to an array of basic types  array[n]
      case TStreamerInfo::kOffsetP + TStreamerInfo::kChar:  { type = GetArrayType(element,"Char",container ); break; }
      case TStreamerInfo::kOffsetP + TStreamerInfo::kShort: { type = GetArrayType(element,"Short",container ); break; }
      case TStreamerInfo::kOffsetP + TStreamerInfo::kInt:   { type = GetArrayType(element,"Int",container ); break; }
      case TStreamerInfo::kOffsetP + TStreamerInfo::kLong:  { type = GetArrayType(element,"Long",container ); break; }
      case TStreamerInfo::kOffsetP + TStreamerInfo::kFloat: { type = GetArrayType(element,"Float",container ); break; }
      case TStreamerInfo::kOffsetP + TStreamerInfo::kDouble:{ type = GetArrayType(element,"Double",container ); break; }
      case TStreamerInfo::kOffsetP + TStreamerInfo::kUChar: { type = GetArrayType(element,"UChar",container ); break; }
      case TStreamerInfo::kOffsetP + TStreamerInfo::kUShort:{ type = GetArrayType(element,"UShort",container ); break; }
      case TStreamerInfo::kOffsetP + TStreamerInfo::kUInt:  { type = GetArrayType(element,"UInt",container ); break; }
      case TStreamerInfo::kOffsetP + TStreamerInfo::kULong: { type = GetArrayType(element,"ULong",container ); break; }
      case TStreamerInfo::kOffsetP + TStreamerInfo::kBits:  { type = GetArrayType(element,"UInt",container ); break; }

         // array counter //[n]
      case TStreamerInfo::kCounter: { type = "T" + middle + "IntProxy"; break; }


      case TStreamerInfo::kOffsetL + TStreamerInfo::kObjectp:
      case TStreamerInfo::kOffsetL + TStreamerInfo::kObjectP:
      case TStreamerInfo::kObjectp:
      case TStreamerInfo::kObjectP:
      case TStreamerInfo::kAnyp:
      case TStreamerInfo::kAnyP:
         // set as pointers and fall through to the next switches
         ispointer = true;
      case TStreamerInfo::kOffsetL + TStreamerInfo::kObject:
      case TStreamerInfo::kObject:
      case TStreamerInfo::kTString:
      case TStreamerInfo::kTNamed:
      case TStreamerInfo::kTObject:
      case TStreamerInfo::kAny:
      case TStreamerInfo::kBase: {
         TClass *cl = element->GetClassPointer();
         if (cl) {
            type = Form("T%sObjProxy<%s >",
                        middle.Data(),cl->GetName());
            cname = cl->GetName();
            if (cl==TClonesArray::Class()) {
               isclones = true;

               Int_t i = branch->GetTree()->GetReadEntry();
               if (i<0) i = 0;
               branch->GetEntry(i);

               //char *obj = branch->GetObject();

               // now need to follow it through to this pointer!

               TClonesArray *arr;

               TString fullpath = branch->GetName();
               fullpath += ".";
               if (path && strlen(path)>0) fullpath.Append(path).Append(".");
               fullpath += element->GetName();

               TTreeFormula *formula = new TTreeFormula("clones",fullpath,branch->GetTree());

               TFormLeafInfo *leafinfo = formula->GetLeafInfo(0);
               TLeaf *leaf = formula->GetLeaf(0);
               Assert(leaf && leafinfo);

               arr = (TClonesArray*)leafinfo->GetLocalValuePointer(leaf,0);

               /*
               if (ispointer) {
                  arr = (TClonesArray*)*(void**)(obj+lOffset);
               } else {
                  arr = (TClonesArray*)(obj+lOffset);
               }
               */
               cname = arr->GetClass()->GetName();

               if (cname.Length()==0) {
                  fprintf(stderr,"introspection of TClonesArray in older file not implemented yet\n");
               }
            }
         }
         else fprintf(stderr,"missing class for %s\n",branch->GetName());
         if (element->IsA()==TStreamerBase::Class()) {
            // prefix  = "base";
            isBase = true;
         }
         AddForward(cl);
         AddHeader(cl);
         break;
      }

      default:
         fprintf(stderr,"Unsupported type for %s %s %d\n",branch->GetName(), element->GetName(), element->GetType());

   }

   dataMemberName = element->GetName();

   if (level<=fMaxUnrolling) {

      // See AnalyzeTree for similar code!
      TProxyClassDescriptor *cldesc;

      TClass *cl = gROOT->GetClass(cname);
      if (cl) {
         cldesc = new TProxyClassDescriptor(cl->GetName(), branch->GetName(), isclones, 0 /* non-split object */);

         TStreamerInfo *info = cl->GetStreamerInfo();
         TStreamerElement *elem = 0;

         TString subpath = path;
         if (subpath.Length()>0) subpath += ".";
         subpath += dataMemberName;

         TIter next(info->GetElements());
         while( (elem = (TStreamerElement*)next()) ) {
            AnalyzeElement(branch, elem, level+1, cldesc, subpath.Data());
         }

         TProxyClassDescriptor *added = AddClass(cldesc);
         if (added) type = added->GetName();
      }

   }

   pxDataMemberName = /* prefix + */ dataMemberName;
   TProxyDescriptor *desc;
   if (topdesc) {
      topdesc->AddDescriptor( desc = new TProxyDescriptor( pxDataMemberName.Data(), type, dataMemberName.Data(), false), isBase );
   } else {
      fprintf(stderr,"topdesc should not be null in TGenerateProxy::AnalyzeElement\n");
   }


}

void TGenerateProxy::WriteProxy() {

   // Check whether the file exist and do something useful if it does
   TString fileLocation = gSystem->DirName(fScript);

   TString incPath = gSystem->GetIncludePath(); // of the form -Idir1  -Idir2 -Idir3
   incPath.Append(":").Prepend(" ");
   incPath.ReplaceAll(" -I",":");       // of form :dir1 :dir2:dir3
   while ( incPath.Index(" :") != -1 ) {
      incPath.ReplaceAll(" :",":");
   }
   incPath.Prepend(fileLocation+":.:");

   const char *filename = gSystem->Which(incPath,fScript);
   if (filename==0) {
      fprintf(stderr,"Can not find the user's script: %s\n",fScript.Data());
      return;
   }
   const char *cutfilename = 0;
   if (fCutScript.Length()) {
      fileLocation = gSystem->DirName(fCutScript);
      incPath.Prepend(fileLocation+":.:");
      cutfilename = gSystem->Which(incPath,fCutScript);
      if (cutfilename==0) {
         fprintf(stderr,"Can not find the user's cut script: %s\n",fCutScript.Data());
         return;
      }
   }

   fHeaderFilename = fPrefix;
   fHeaderFilename.Append(".h");

   TString classname = fPrefix;

   TString treefile;
   bool ischain = fTree->InheritsFrom(TChain::Class());
   if (fTree->GetDirectory() && fTree->GetDirectory()->GetFile())
      treefile = fTree->GetDirectory()->GetFile()->GetName();
   else
      treefile = "Memory Directory";

   TString scriptfunc = fScript;
   Ssiz_t dot_pos = scriptfunc.Last('.');
   scriptfunc.Replace( dot_pos, fScript.Length()-dot_pos, "");
   TString scriptHeader = scriptfunc;
   const char * extensions[] = { ".h", ".hh", ".hpp", ".hxx",  ".hPP", ".hXX" };

   int i;
   for (i = 0; i < 6; i++ ) {
      TString possible = scriptHeader;
      possible.Append(extensions[i]);
      const char *name = gSystem->Which(incPath,possible);
      if (name) {
         scriptHeader = possible;
         fListOfHeaders.Add(new TNamed("script",Form("#include \"%s\"\n",
                                                     scriptHeader.Data())));
         break;
      }
   }
   scriptfunc = gSystem->BaseName(scriptfunc);


   TString cutscriptfunc = fCutScript;
   if (cutfilename) {
      Ssiz_t dot_pos = cutscriptfunc.Last('.');
      cutscriptfunc.Replace( dot_pos, fCutScript.Length()-dot_pos, "");
      TString cutscriptHeader = cutscriptfunc;
      const char * extensions[] = { ".h", ".hh", ".hpp", ".hxx",  ".hPP", ".hXX" };

      for (i = 0; i < 6; i++ ) {
         TString possible = cutscriptHeader;
         possible.Append(extensions[i]);
         const char *name = gSystem->Which(incPath,possible);
         if (name) {
            cutscriptHeader = possible;
            fListOfHeaders.Add(new TNamed("cutscript",Form("#include \"%s\"\n",
                                                           cutscriptHeader.Data())));
            break;
         }
      }
      cutscriptfunc = gSystem->BaseName(cutscriptfunc);
   }

   FILE *hf = fopen(fHeaderFilename.Data(),"w");

   TDatime td;
   fprintf(hf,   "//////////////////////////////////////////////////////////\n");
   fprintf(hf,   "//   This class has been automatically generated \n");
   fprintf(hf,   "//     (%s by ROOT version%s)\n",td.AsString(),gROOT->GetVersion());
   if (!ischain) {
      fprintf(hf,"//   from TTree %s/%s\n",fTree->GetName(),fTree->GetTitle());
      fprintf(hf,"//   found on file: %s\n",treefile.Data());
   } else {
      fprintf(hf,"//   from TChain %s/%s\n",fTree->GetName(),fTree->GetTitle());
   }
   fprintf(hf,   "//////////////////////////////////////////////////////////\n");
   fprintf(hf,"\n");
   fprintf(hf,"\n");

   fprintf(hf,"#ifndef %s_h\n",classname.Data());
   fprintf(hf,"#define %s_h\n",classname.Data());
   fprintf(hf,"\n");


   fprintf(hf,"// System Headers needed by the proxy\n");
   fprintf(hf,"#include <TROOT.h>\n");
   fprintf(hf,"#include <TChain.h>\n");
   fprintf(hf,"#include <TFile.h>\n");
   fprintf(hf,"#include <TSelectorDraw.h>\n");
   fprintf(hf,"#include <TPad.h>\n");
   fprintf(hf,"#include <TH1.h>\n");
   fprintf(hf,"#include <TProxy.h>\n");
   fprintf(hf,"#include <TProxyDirector.h>\n");
   fprintf(hf,"#include <TProxyTemplate.h>\n");
   fprintf(hf,"#if defined(__CINT__) && !defined(__MAKECINT__)\n");
   fprintf(hf,"   #define ROOT_Rtypes\n");
   fprintf(hf,"#endif\n");
   fprintf(hf,"using namespace ROOT;\n");       // questionable
   fprintf(hf,"\n\n");

   fprintf(hf,"// forward declarations needed by this particular proxy\n");
   TIter next( &fListOfForwards );
   TObject *current;
   while ( (current=next()) ) {
      fprintf(hf,current->GetTitle());
   }

   fprintf(hf,"\n\n");
   fprintf(hf,"// Header needed by this particular proxy\n");
   next = &fListOfHeaders;
   TObject *header;
   while ( (header = next()) ) {
      fprintf(hf,header->GetTitle());
   }
   fprintf(hf,"\n\n");

   fprintf(hf, "class %s : public TSelector {\n", classname.Data());
   fprintf(hf, "   public :\n");
   fprintf(hf, "   TTree          *fChain;    //!pointer to the analyzed TTree or TChain\n");
   fprintf(hf, "   TSelectorDraw  *fHelper;   //!helper class to create the default histogram\n");
   fprintf(hf, "   TList          *fInput;    //!input list of the helper\n");
   fprintf(hf, "   TH1            *htemp;     //!pointer to the histogram\n");
   fprintf(hf, "   TProxyDirector  fDirector; //!Manages the proxys\n\n");

   fprintf(hf, "   // Wrapper class for each unwounded class\n");
   next = &fListOfClasses;
   TProxyClassDescriptor *clp;
   while ( (clp = (TProxyClassDescriptor*)next()) ) {
      clp->OutputDecl(hf, 3, fMaxDatamemberType);
   }
   fprintf(hf,"\n\n");

   fprintf(hf, "   // Proxy for each of the branches and leaves of the tree\n");
   next = &fListOfTopProxies;
   TProxyDescriptor *data;
   while ( (data = (TProxyDescriptor*)next()) ) {
      data->OutputDecl(hf, 3, fMaxDatamemberType);
   }
   fprintf(hf,"\n\n");

   // Constructor
   fprintf(hf,      "   %s(TTree *tree=0) : \n",classname.Data());
   fprintf(hf,      "      fChain(0)");
   fprintf(hf,   ",\n      fHelper(0)");
   fprintf(hf,   ",\n      fInput(0)");
   fprintf(hf,   ",\n      htemp(0)");
   fprintf(hf,   ",\n      fDirector(tree,-1)");
   next.Reset();
   while ( (data = (TProxyDescriptor*)next()) ) {
      fprintf(hf,",\n      %-*s(&fDirector,\"%s\")",fMaxDatamemberType,data->GetName(), data->GetBranchName());
   }

   fprintf(hf,    "\n      { }\n");

   // Other functions.
   fprintf(hf,"   ~%s();\n",classname.Data());
   fprintf(hf,"   void    Begin(::TTree *tree);\n");
   fprintf(hf,"   void    Init(::TTree *tree);\n");
   fprintf(hf,"   Bool_t  Notify();\n");
   fprintf(hf,"   Bool_t  Process(Int_t entry);\n");
   fprintf(hf,"   Bool_t  ProcessCut(Int_t entry);\n");
   fprintf(hf,"   void    ProcessFill(Int_t entry);\n");
   fprintf(hf,"   void    SetOption(const char *option) { fOption = option; }\n");
   fprintf(hf,"   void    SetObject(TObject *obj) { fObject = obj; }\n");
   fprintf(hf,"   void    SetInputList(TList *input) {fInput = input;}\n");
   fprintf(hf,"   TList  *GetOutputList() const { return fOutput; }\n");
   fprintf(hf,"   void    Terminate();\n");
   fprintf(hf,"\n");
   fprintf(hf,"   ClassDef(%s,0);\n",classname.Data());
   fprintf(hf,"\n\n");

   fprintf(hf,"//inject the user's code\n");
   fprintf(hf,"#include \"%s\"\n",fScript.Data());

   if (cutfilename) {
      fprintf(hf,"#include \"%s\"\n",fCutScript.Data());
   }

   // Close the class.
   fprintf(hf,"};\n");
   fprintf(hf,"\n");
   fprintf(hf,"#endif\n");
   fprintf(hf,"\n\n");

   fprintf(hf,"#ifdef __MAKECINT__\n");
   next = &fListOfClasses;
   while ( (clp = (TProxyClassDescriptor*)next()) ) {
      fprintf(hf,"#pragma link C++ class %s::%s-;\n",classname.Data(),clp->GetName());
   }
   fprintf(hf,"#pragma link C++ class %s;\n",classname.Data());
   fprintf(hf,"#endif\n");
   fprintf(hf,"\n\n");

   // Write the implementations.
   fprintf(hf,"%s::~%s() {\n",classname.Data(),classname.Data());
   fprintf(hf,"   // destructor. Clean up helpers.\n");
   fprintf(hf,"\n");
   fprintf(hf,"   delete fHelper;\n");
   fprintf(hf,"   delete fInput;\n");
   fprintf(hf,"}\n");
   fprintf(hf,"\n");
   fprintf(hf,"void %s::Init(TTree *tree)\n",classname.Data());
   fprintf(hf,"{\n");
   fprintf(hf,"//   Set branch addresses\n");
   fprintf(hf,"   if (tree == 0) return;\n");
   fprintf(hf,"   fChain = tree;\n");
   fprintf(hf,"   fDirector.SetTree(fChain);\n");
   fprintf(hf,"   fHelper = new TSelectorDraw();\n");
   fprintf(hf,"   fInput  = new TList();\n");
   fprintf(hf,"   fInput->Add(new TNamed(\"varexp\",\"0.0\")); // Fake a double size histogram\n");
   fprintf(hf,"   fInput->Add(new TNamed(\"selection\",\"\"));\n");
   fprintf(hf,"   fHelper->SetInputList(fInput);\n");
   fprintf(hf,"}\n");
   fprintf(hf,"\n");
   fprintf(hf,"Bool_t %s::Notify()\n",classname.Data());
   fprintf(hf,"{\n");
   fprintf(hf,"   // Called when loading a new file.\n");
   fprintf(hf,"   // Get branch pointers.\n");
   fprintf(hf,"   fDirector.SetTree(fChain);\n");
   fprintf(hf,"   \n");
   fprintf(hf,"   return kTRUE;\n");
   fprintf(hf,"}\n");
   fprintf(hf,"   \n");
   fprintf(hf,"void %s::Begin(TTree *tree)\n",classname.Data());
   fprintf(hf,"{\n");
   fprintf(hf,"   // Function called before starting the event loop.\n");
   fprintf(hf,"   // Initialize the tree branches.\n");
   fprintf(hf,"\n");
   fprintf(hf,"   Init(tree);\n");
   fprintf(hf,"\n");
   fprintf(hf,"   TString option = GetOption();\n");
   fprintf(hf,"   fHelper->SetOption(option);\n");
   fprintf(hf,"   fHelper->Begin(tree);\n");
   fprintf(hf,"   htemp = (TH1*)fHelper->GetObject();\n");
   if (cutfilename) {
      fprintf(hf,"   htemp->SetTitle(\"%s {%s}\");\n",fScript.Data(),fCutScript.Data());
   } else {
      fprintf(hf,"   htemp->SetTitle(\"%s\");\n",fScript.Data());
   }
   fprintf(hf,"   fObject = htemp;\n");
   fprintf(hf,"\n");
   fprintf(hf,"}\n");
   fprintf(hf,"\n");
   fprintf(hf,"Bool_t %s::Process(Int_t /* entry */)\n",classname.Data());
   fprintf(hf,"{\n");
   fprintf(hf,"   // Processing function.\n");
   fprintf(hf,"   // Entry is the entry number in the current tree.\n");
   fprintf(hf,"   // Read only the necessary branches to select entries.\n");
   fprintf(hf,"   // To read complete event, call fChain->GetTree()->GetEntry(entry).\n");
   fprintf(hf,"   // Return kFALSE to stop processing.\n");
   fprintf(hf,"\n");
   fprintf(hf,"   return kTRUE;\n");
   fprintf(hf,"}\n");
   fprintf(hf,"\n");
   fprintf(hf,"Bool_t %s::ProcessCut(Int_t /* entry */)\n",classname.Data());
   fprintf(hf,"{\n");
   fprintf(hf,"   // Selection function.\n");
   fprintf(hf,"   // Entry is the entry number in the current tree.\n");
   fprintf(hf,"   // Read only the necessary branches to select entries.\n");
   fprintf(hf,"   // Return kFALSE as soon as a bad entry is detected.\n");
   fprintf(hf,"   // To read complete event, call fChain->GetTree()->GetEntry(entry).\n");
   fprintf(hf,"\n");
   fprintf(hf,"   return kTRUE;\n");
   fprintf(hf,"}\n");
   fprintf(hf,"\n");
   fprintf(hf,"void %s::ProcessFill(Int_t entry)\n",classname.Data());
   fprintf(hf,"{\n");
   fprintf(hf,"   // Function called for selected entries only.\n");
   fprintf(hf,"   // Entry is the entry number in the current tree.\n");
   fprintf(hf,"   // Read branches not processed in ProcessCut() and fill histograms.\n");
   fprintf(hf,"   // To read complete event, call fChain->GetTree()->GetEntry(entry).\n");

   fprintf(hf,"   fDirector.SetReadEntry(entry);\n");
   if (cutfilename) {
      fprintf(hf,"   if (%s()) htemp->Fill(%s());\n",cutscriptfunc.Data(),scriptfunc.Data());
   } else {
      fprintf(hf,"   htemp->Fill(%s());\n",scriptfunc.Data());
   }

   fprintf(hf,"\n");
   fprintf(hf,"}\n");
   fprintf(hf,"\n");
   fprintf(hf,"void %s::Terminate()\n",classname.Data());
   fprintf(hf,"{\n");
   fprintf(hf,"   // Function called at the end of the event loop.\n");
   fprintf(hf,"   Int_t drawflag = (htemp && htemp->GetEntries()>0);\n");
   fprintf(hf,"   \n");
   fprintf(hf,"   if (!drawflag && !fOption.Contains(\"goff\") && !fOption.Contains(\"same\")) {\n");
   fprintf(hf,"      gPad->Clear();\n");
   fprintf(hf,"      return;\n");
   fprintf(hf,"  }\n");
   fprintf(hf,"   if (fOption.Contains(\"goff\")) drawflag = false;\n");
   fprintf(hf,"   if (drawflag) htemp->Draw(fOption);\n");
   fprintf(hf,"\n");
   fprintf(hf,"}\n");

   fclose(hf);
}



Int_t draw(const char* prefix, TTree *tree, const char *filename, const char *cutfilename = "", Option_t *option = "", Int_t nentries=1000000000, Int_t firstentry=0) {

   if (!filename || strlen(filename)==0) return 0;

   TString aclicMode;
   TString arguments;
   TString io;
   TString realcutname;
   if (cutfilename && strlen(cutfilename))
      realcutname =  gSystem->SplitAclicMode(cutfilename, aclicMode, arguments, io);

   // we ignore the aclicMode for the cutfilename!
   TString    realname = gSystem->SplitAclicMode(filename, aclicMode, arguments, io);

   TString selname = prefix;

   TGenerateProxy gp(tree,realname,realcutname,selname,3);

   // should check on the existence of selname+".h"

   selname = gp.GetFilename();
   selname.Append(aclicMode);

   fprintf(stderr,"will process %s\n",selname.Data());
   Int_t result = tree->Process(selname,option,nentries,firstentry);

   // could delete the file selname+".h"

   return result;
}

Int_t draw(TTree *tree, const char *filename, const char *cutfilename = "", Option_t *option = "", Int_t nentries=1000000000, Int_t firstentry=0) {
   return draw("generatedSel",tree,filename,cutfilename,option,nentries,firstentry);
}

