#include "TList.h"
#include <stdio.h>

class TTree;
class TBranch;

#include "TClass.h"
#include "TClonesArray.h"
#include "TError.h"

class TProxyDescriptor : public TNamed {
   TString fBranchName;
public:
   TProxyDescriptor(const char *dataname, const char *type, const char *branchname) :
      TNamed(dataname,type),fBranchName(branchname) {}
   const char *GetDataName() { return GetName(); }
   const char *GetTypeName() { return GetTitle(); }
   const char *GetBranchName() { return fBranchName.Data(); }

   void OutputDecl(FILE *hf, int offset, UInt_t maxVarname){
      fprintf(hf,"%-*s%-*s %s;\n",  offset," ",  maxVarname, GetTypeName(), GetDataName()); // might want to add a comment      
   };

   ClassDef(TProxyDescriptor,0); 
};

class TProxyClassDescriptor : public TNamed {
   TList   fListOfSubProxies;
   UInt_t  fMaxDatamemberType;
   UInt_t  fSplitLevel;
   TString fBranchName;
   TString fSubBranchPrefix;
   Bool_t  fIsClones;

   void FixName() { 

      // Make the typename a proper class name without having the really deal with 
      // namespace and templates.
            
      TString newname = GetName();
      newname.ReplaceAll(":","_");
      newname.ReplaceAll("<","_");
      newname.ReplaceAll(">","_");
      newname.ReplaceAll(" ","");
      newname.ReplaceAll("*","st");
      newname.ReplaceAll("&","rf");
      newname.Prepend("TPx_");
      SetName(newname);
   }

public:
   TProxyClassDescriptor(const char *type, const char *branchname, Bool_t isclones) :
      TNamed(type,type), 
      fMaxDatamemberType(3),
      fSplitLevel(0),
      fBranchName(branchname),
      fSubBranchPrefix(branchname),
      fIsClones(isclones)
   {
      FixName();
   }

   TProxyClassDescriptor(const char *type, const char *branchname, 
                         const char *branchPrefix, Bool_t isclones) :
      TNamed(type,type), 
      fMaxDatamemberType(3),
      fSplitLevel(0),
      fBranchName(branchname),
      fSubBranchPrefix(branchPrefix),
      fIsClones(isclones)
   {
      FixName();
   }

   void AddDescriptor(TProxyDescriptor *desc) {
      if (desc) {
         fListOfSubProxies.Add(desc);
         UInt_t len = strlen(desc->GetTypeName());
         if ((len+2)>fMaxDatamemberType) fMaxDatamemberType = len+2;
      }
   }

   Bool_t IsLoaded() const {
      TClass *cl = gROOT->GetClass(GetTitle());
      return (cl && cl->IsLoaded());
   }
   
   const char* GetBranchName() const {
      return fBranchName.Data();
   }

   const char* GetSubBranchPrefix() const {
      return fSubBranchPrefix.Data();
   }

   UInt_t GetSplitLevel() const { return fSplitLevel; }
   Bool_t IsClones() const { return fIsClones; }

   void OutputDecl(FILE *hf, int offset, UInt_t maxVarname){
      // Output the declaration and implementation of this emulation class

      TProxyDescriptor *desc;
      fprintf(hf,"%-*sstruct %s {\n", offset," ", GetName() );

      fprintf(hf,"%-*s   %s(TProxyDirector* director,const char *top,const char *mid=0) :",
              offset," ", GetName());

      fprintf(hf,"\n%-*s      %-*s(top,mid)",offset," ",fMaxDatamemberType,"ffPrefix");

      bool wroteFirst = true;
      if (IsLoaded() || IsClones() ) {
         fprintf(hf,"%s\n%-*s      %-*s(director,\"%s\")",
                 wroteFirst?",":"",offset," ",fMaxDatamemberType,"obj",GetBranchName());
         wroteFirst = true;

      }
      TIter next(&fListOfSubProxies);
      while ( (desc = (TProxyDescriptor*)next()) ) {
         const char *subbranchname = desc->GetBranchName();
         const char *above = "";
         if (strncmp(GetSubBranchPrefix(),desc->GetBranchName(),strlen(GetSubBranchPrefix()))==0) {
            subbranchname += strlen(GetSubBranchPrefix())+1; // +1 for the dot "."
            above = "ffPrefix, ";
         }
         fprintf(hf,"%s\n%-*s      %-*s(director, %s\"%s\")",
                 wroteFirst?",":"",offset," ",fMaxDatamemberType,desc->GetName(), above, subbranchname);
         wroteFirst = true;
      }
      fprintf(hf,"\n%-*s   {};\n",offset," ");

      fprintf(hf,"%-*s%-*s %s;\n",  offset+3," ",  fMaxDatamemberType, "TProxyHelper", "ffPrefix");

      // If the real class is available, make it available via the arrow operator:
      if (IsLoaded() || IsClones()) {
         const char *type = IsClones() ? "TClonesArray" : GetTitle();
         fprintf(hf,"%-*sInjectProxyInterface();\n", offset+3," ");
         //Can the real type contain a leading 'const'? If so the following is incorrect.
         fprintf(hf,"%-*sconst %s* operator->() { return obj.ptr(); }\n", offset+3," ",type);
         fprintf(hf,"%-*sTObjProxy< %s > obj;\n", offset+3, " ", type);
      }

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
   TString  fPrefix;
   TTree   *fTree;
   TList    fListOfHeaders;
   TList    fListOfClasses;
   TList    fListOfTopProxies;
   TList    fListOfForwards;
   TGenerateProxy(TTree* tree, const char *script, const char *fileprefix);

   TProxyClassDescriptor* AddClass(TProxyClassDescriptor *desc);
   void AddForward(TClass *cl);
   void AddForward(const char *classname);
   void AddHeader(TClass *cl);
   void AddHeader(const char *classname);
   void AddDescriptor(TProxyDescriptor *desc);

   bool NeedToEmulate(TClass *cl, UInt_t level);

   UInt_t AnalyzeBranch(TBranch *branch, UInt_t level, TProxyClassDescriptor *desc);
   void   AnalyzeTree();
   void   WriteHeader();
};

#include "TBranchElement.h"
#include "TChain.h"
#include "TFile.h"
#include "TLeaf.h"
#include "TTree.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TSystem.h"

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

TGenerateProxy::TGenerateProxy(TTree* tree, const char *script, const char *fileprefix) :
   fMaxDatamemberType(2),
   fScript(script),
   fPrefix(fileprefix),
   fTree(tree)
{
   
   AnalyzeTree();

   WriteHeader();
}

Bool_t TGenerateProxy::NeedToEmulate(TClass *cl, UInt_t level) {
   // Return true if we should create a nested class representing this class

   return cl->TestBit(TClass::kIsEmulation);
}

TProxyClassDescriptor*
TGenerateProxy::AddClass( TProxyClassDescriptor* desc ) 
{
   if (desc==0) return 0;

   TProxyClassDescriptor *existing =
      (TProxyClassDescriptor*)fListOfClasses(desc->GetName());
   
   if (existing) {
      if (existing->GetSplitLevel() != desc->GetSplitLevel()) {
         TString newname = desc->GetName();
         newname += "_l";
         newname += desc->GetSplitLevel();
         desc->SetName(newname);
         return AddClass(desc);
      } else {
         // we already have the exact same class 
         delete desc;
         return existing;
      }
   } else {
      fListOfClasses.AddFirst(desc);
   }
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

   TString type;
   TString prefix;
   TString dataMemberName;
   TString cname;
   TString middle;
   UInt_t  extraLookedAt = 0;
   Bool_t  isclones = false;
   EContainer container = kNone;
   if (topdesc && topdesc->IsClones()) {
      container = kClones;
      middle = "Cla";
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
            case TStreamerInfo::kOffsetL + TStreamerInfo::kObject:
            case TStreamerInfo::kObject:
            case TStreamerInfo::kObjectp:
            case TStreamerInfo::kObjectP:
            case TStreamerInfo::kTString:
            case TStreamerInfo::kTNamed:
            case TStreamerInfo::kTObject:
            case TStreamerInfo::kAny:
            case TStreamerInfo::kAnyp:
            case TStreamerInfo::kAnyP: {
               TClass *cl = element->GetClassPointer();
               if (cl) {
                  type = Form("T%sObjProxy<%s >",
                              middle.Data(),cl->GetName());
                  cname = cl->GetName();
                  if (cl==TClonesArray::Class()) {
                     isclones = true;
                     cname = be->GetClonesName();
                     if (!cname) {
                        fprintf(stderr,"introspection of TClonesArray in older file not implemented yet\n");
                     }
                  }
               }
               else fprintf(stderr,"missing class for %s\n",branch->GetName());
               if (element->IsA()==TStreamerBase::Class()) {
                  prefix  = "base";
               }
               AddForward(cl);
               AddHeader(cl);
               break;
            }
            
            default:
               fprintf(stderr,"Unsupported type for %s\n",branch->GetName());
                
         }

      }      
      
   } else {
      
      fprintf(stderr,"non TBranchElement not implemented yet\n");
      return extraLookedAt;
      
   }
   
   if ( branch->GetListOfBranches()->GetEntries() > 0 ) {
      
      // See AnalyzeTree for similar code!
      TProxyClassDescriptor *cldesc;

      TClass *cl = gROOT->GetClass(cname);
      if (cl) {            
         cldesc = new TProxyClassDescriptor(cl->GetName(), branch->GetName(),isclones);
         TProxyClassDescriptor *added = AddClass(cldesc);
         if (added) type = added->GetName();
         if (added!=cldesc) cldesc = 0;
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
      }

   }
   TLeaf *leaf = (TLeaf*)branch->GetListOfLeaves()->At(0);
   
   if (leaf && strlen(leaf->GetTypeName()) == 0) return extraLookedAt;
   
   if (leaf && type.Length()==0) type=leaf->GetTypeName() ;
   
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

   if (pos != -1 && container==kClones && branch->IsA()==TBranchElement::Class()) {
      // We still have a "." in the name, we assume that we are in the case
      // where we reach an embedded object in the object contained in the
      // TClonesArray
      
      // Discover the type of this object.
      TString name = dataMemberName(0,pos);
      TBranchElement *mom = (TBranchElement*)branch->GetMother();
      TString cname = mom->GetClonesName();
      TString prefix = mom->GetName();
      prefix += ".";
      prefix += name;
      // prefix += ".";
      TStreamerElement* elem = (TStreamerElement*)
         mom->GetInfo()->GetElements()->FindObject("fTriggerBits");
      TClass *cl = elem->GetClassPointer();

      cname = cl->GetName();
      type = Form("TClaObjProxy<%s >",cname.Data());
      
      TProxyClassDescriptor *cldesc;

      cldesc = new TProxyClassDescriptor( cl->GetName(), prefix.Data(), prefix.Data(), isclones);
      TProxyClassDescriptor *added = AddClass(cldesc);
      if (added) type = added->GetName();
      if (added!=cldesc) cldesc = 0;
      
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

   } else {

      // Assume we coming recursively from the previous case!
      dataMemberName.Remove(0,pos+1);
   }

   dataMemberName.Prepend(prefix);

   TProxyDescriptor *desc;
   if (topdesc) {
      topdesc->AddDescriptor( desc = new TProxyDescriptor( dataMemberName.Data(), type, branch->GetName() ) );
   } else {
      AddDescriptor( desc = new TProxyDescriptor( dataMemberName.Data(), type, branch->GetName() ) );
   } 
   //fprintf(stderr,"%-*s      %-*s(director,\"%s\")\n",
   //        0," ",10,desc->GetName(), desc->GetBranchName());
   return extraLookedAt;
}

void TGenerateProxy::AnalyzeTree() {
   
   TIter next( fTree->GetListOfBranches() );
   TBranch *branch;
   while ( (branch = (TBranch*)next()) ) {
      const char *branchname = branch->GetName();
      const char *classname = branch->GetClassName();
      AddForward( classname );
      AddHeader( classname );
      
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
                  fprintf(stderr,"introspection of TClonesArrya in older file not implemented yet\n");
               }
            } else {
               fprintf(stderr,"introspection of TClonesArrya in older file not implemented yet\n");
            }

         }
         if (NeedToEmulate(cl,0) || branchname[strlen(branchname)-1] == '.' ) {
            desc = new TProxyClassDescriptor(cl->GetName(),branchname,isclones);
            desc = AddClass(desc);
            type = desc->GetName();
         } else {
            type = Form("TObjProxy<%s >",cl->GetName());
         }
      }

      AddDescriptor( new TProxyDescriptor( branchname, type, branchname ) );

      TBranch *subbranch;
      TIter subnext( branch->GetListOfBranches() );
      UInt_t skipped = 0;
      if (desc) {
         while ( (subbranch = (TBranch*)subnext()) ) {
            skipped = AnalyzeBranch(subbranch,1,desc);
            if (skipped != 0) fprintf(stderr,"unexpectly read more than one branch in AnalyzeTree\n");
         }         
      }
      if ( branchname[strlen(branchname)-1] != '.' ) {
         // If there is no dot also included the data member directly
         subnext.Reset();
         while ( (subbranch = (TBranch*)subnext()) ) {
            skipped = AnalyzeBranch(subbranch,1,0);
            if (skipped != 0) fprintf(stderr,"unexpectly read more than one branch in AnalyzeTree\n");
         }
      }
   }

}

void TGenerateProxy::WriteHeader() {
   
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
   
   TString headerFilename = fPrefix;
   headerFilename.Append(".h");

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

   FILE *hf = fopen(headerFilename.Data(),"w");

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
   fprintf(hf,"#include <TSelector.h>\n");
   fprintf(hf,"#include <TProxy.h>\n");
   fprintf(hf,"#include <TProxyTemplate.h>\n");
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
   fprintf(hf,      "      fDirector(tree,-1)");
   next.Reset();
   while ( (data = (TProxyDescriptor*)next()) ) {
      fprintf(hf,",\n      %-*s(&fDirector,\"%s\")",fMaxDatamemberType,data->GetName(), data->GetBranchName());
   }
   
   fprintf(hf,    "\n      { }\n");

   // Other functions.
   fprintf(hf,"   ~%s() { }\n",classname.Data());
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
   fprintf(hf,"   #include \"%s\"\n",fScript.Data());

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
   fprintf(hf,"void %s::Init(TTree *tree)\n",classname.Data());
   fprintf(hf,"{\n");
   fprintf(hf,"//   Set branch addresses\n");
   fprintf(hf,"   if (tree == 0) return;\n");
   fprintf(hf,"   fChain = tree;\n");
   fprintf(hf,"   fDirector.SetTree(fChain);\n");
   fprintf(hf,"\n");
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
   fprintf(hf,"\n");
   fprintf(hf,"}\n");
   fprintf(hf,"\n");
   fprintf(hf,"Bool_t %s::Process(Int_t entry)\n",classname.Data());
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
   fprintf(hf,"Bool_t %s::ProcessCut(Int_t entry)\n",classname.Data());
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

   fprintf(hf,"   fDirector.fEntry = entry;\n");
   fprintf(hf,"   %s();\n",scriptfunc.Data());

   fprintf(hf,"\n");
   fprintf(hf,"}\n");
   fprintf(hf,"\n");
   fprintf(hf,"void %s::Terminate()\n",classname.Data());
   fprintf(hf,"{\n");
   fprintf(hf,"   // Function called at the end of the event loop.\n");
   fprintf(hf,"\n");
   fprintf(hf,"}\n");

   fclose(hf);
}
