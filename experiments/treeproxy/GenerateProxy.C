#include "TList.h"
#include <stdio.h>

class TTree;


class TProxyDescriptor : public TNamed {
   TString fBranchName;
   TList fListOfSubProxies;
public:
   TProxyDescriptor(const char *dataname, const char *type, const char *branchname) :
      TNamed(dataname,type),fBranchName(branchname) {}
   const char *GetDataName() { return GetName(); }
   const char *GetTypeName() { return GetTitle(); }
   const char *GetBranchName() { return fBranchName.Data(); }

   void OutputDecl(FILE *hf, int offset, UInt_t maxVarname){
      fprintf(hf,"%-*s%-*s %s;\n",  offset," ",  maxVarname,GetTypeName(),  GetDataName()); // might want to add a comment      
   };

   ClassDef(TProxyDescriptor,0); 
};

class TGenerateProxy /* change this name please */ 
{
public:
   UInt_t   fMaxDatamemberType;
   TString  fScript;
   TString  fPrefix;
   TTree   *fTree;
   TList    fListOfHeaders;
   TList    fListOfTopProxies;
   TList    fListOfForwards;
   TGenerateProxy(TTree* tree, const char *script, const char *fileprefix);

   void AddForward(TClass *cl);
   void AddForward(const char *classname);
   void AddHeader(TClass *cl);
   void AddHeader(const char *classname);
   void AddDescriptor(TProxyDescriptor *desc);

   void AnalyzeTree();
   void WriteHeader();
};

#include "TBranchElement.h"
#include "TChain.h"
#include "TFile.h"
#include "TLeaf.h"
#include "TTree.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TSystem.h"

TString GetArrayType(TStreamerElement *element, const char *subtype) 
{
   TString result;
   int ndim = 0;
   if (element && element->InheritsFrom(TStreamerBasicPointer::Class())) {
      TStreamerBasicPointer * elem = (TStreamerBasicPointer*)element;
      const char *countname = elem->GetCountName();
      if (countname && strlen(countname)>0) ndim = 1;
   }
   ndim += element->GetArrayDim();
   
   if (ndim==0) {
      result = "T";
      result += subtype;
      result += "Proxy";
   } else if (ndim==1) {
      result = "TArray";
      result += subtype;
      result += "Proxy";
   } else if (ndim==2) {
      result = "TArray2Proxy<";
      result += element->GetTypeName();
      result += ",";
      result += element->GetMaxIndex(1);
      result += " >";
   }  else if (ndim==3) {
      result = "TArray3Proxy<";
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
void TGenerateProxy::AnalyzeTree() {
   
   TIter next( fTree->GetListOfBranches() );
   TBranch *branch;
   while ( (branch = (TBranch*)next()) ) {
      const char *branchname = branch->GetName();
      if ( branchname[strlen(branchname)-1] == '.' ) {
         // The branch was qualified with a '.'.  The intend is usually to 
         // disambiguate the names.  So we need to nested it.

         fprintf(stderr,"branche end with a '.' are not treated yet\n");

         continue;
      }

      const char *classname = branch->GetClassName();
      AddForward( classname );
      AddHeader( classname );
      
      {
         TClass *cl = gROOT->GetClass(classname);
         TString type = "unknown";
         if (cl) type = Form("TObjProxy<%s >",cl->GetName());
         AddDescriptor( new TProxyDescriptor( branchname, type, branchname ) );
      }

      TBranch *subbranch;
      TIter subnext( branch->GetListOfBranches() );
      while ( (subbranch = (TBranch*)subnext()) ) {
         
         TString type;

         if (subbranch->IsA()==TBranchElement::Class()) {
            TBranchElement *be = (TBranchElement*)subbranch;

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

                  case TStreamerInfo::kChar:  { type = "TCharProxy"; break; } 
                  case TStreamerInfo::kShort: { type = "TShortProxy"; break; } 
                  case TStreamerInfo::kInt:   { type = "TIntProxy"; break; } 
                  case TStreamerInfo::kLong:  { type = "TLongProxy"; break; } 
                  case TStreamerInfo::kFloat: { type = "TFloatProxy"; break; } 
                  case TStreamerInfo::kDouble:{ type = "TDoubleProxy"; break; } 
                  case TStreamerInfo::kUChar: { type = "TUCharProxy"; break; } 
                  case TStreamerInfo::kUShort:{ type = "TUShortProxy"; break; } 
                  case TStreamerInfo::kUInt:  { type = "TUIntProxy"; break; } 
                  case TStreamerInfo::kULong: { type = "TULongProxy"; break; } 
                  case TStreamerInfo::kBits:  { type = "TUIntProxy"; break; } 

                  case TStreamerInfo::kCharStar: { type = GetArrayType(element,"Char"); break; } 
                     
                     // array of basic types  array[8]
                  case TStreamerInfo::kOffsetL + TStreamerInfo::kChar:  { type = GetArrayType(element,"Char"); break; } 
                  case TStreamerInfo::kOffsetL + TStreamerInfo::kShort: { type = GetArrayType(element,"Short"); break; } 
                  case TStreamerInfo::kOffsetL + TStreamerInfo::kInt:   { type = GetArrayType(element,"Int"); break; } 
                  case TStreamerInfo::kOffsetL + TStreamerInfo::kLong:  { type = GetArrayType(element,"Long"); break; } 
                  case TStreamerInfo::kOffsetL + TStreamerInfo::kFloat: { type = GetArrayType(element,"Float"); break; } 
                  case TStreamerInfo::kOffsetL + TStreamerInfo::kDouble:{ type = GetArrayType(element,"Double"); break; } 
                  case TStreamerInfo::kOffsetL + TStreamerInfo::kUChar: { type = GetArrayType(element,"UChar"); break; } 
                  case TStreamerInfo::kOffsetL + TStreamerInfo::kUShort:{ type = GetArrayType(element,"UShort"); break; } 
                  case TStreamerInfo::kOffsetL + TStreamerInfo::kUInt:  { type = GetArrayType(element,"UInt"); break; } 
                  case TStreamerInfo::kOffsetL + TStreamerInfo::kULong: { type = GetArrayType(element,"ULong"); break; } 
                  case TStreamerInfo::kOffsetL + TStreamerInfo::kBits:  { type = GetArrayType(element,"UInt"); break; } 
                     
                     // pointer to an array of basic types  array[n]
                  case TStreamerInfo::kOffsetP + TStreamerInfo::kChar:  { type = GetArrayType(element,"Char"); break; } 
                  case TStreamerInfo::kOffsetP + TStreamerInfo::kShort: { type = GetArrayType(element,"Short"); break; } 
                  case TStreamerInfo::kOffsetP + TStreamerInfo::kInt:   { type = GetArrayType(element,"Int"); break; } 
                  case TStreamerInfo::kOffsetP + TStreamerInfo::kLong:  { type = GetArrayType(element,"Long"); break; } 
                  case TStreamerInfo::kOffsetP + TStreamerInfo::kFloat: { type = GetArrayType(element,"Float"); break; } 
                  case TStreamerInfo::kOffsetP + TStreamerInfo::kDouble:{ type = GetArrayType(element,"Double"); break; } 
                  case TStreamerInfo::kOffsetP + TStreamerInfo::kUChar: { type = GetArrayType(element,"UChar"); break; } 
                  case TStreamerInfo::kOffsetP + TStreamerInfo::kUShort:{ type = GetArrayType(element,"UShort"); break; } 
                  case TStreamerInfo::kOffsetP + TStreamerInfo::kUInt:  { type = GetArrayType(element,"UInt"); break; } 
                  case TStreamerInfo::kOffsetP + TStreamerInfo::kULong: { type = GetArrayType(element,"ULong"); break; } 
                  case TStreamerInfo::kOffsetP + TStreamerInfo::kBits:  { type = GetArrayType(element,"UInt"); break; } 
                     
                     // array counter //[n]
                  case TStreamerInfo::kCounter: { type = "TIntProxy"; break; } 
                     
                     
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
                     if (cl) type = Form("TObjProxy<%s >",cl->GetName());
                     else fprintf(stderr,"missing class for %s\n",subbranch->GetName());
                     AddForward(cl);
                     AddHeader(cl);
                     break;
                  }
                   
                  default:
                     fprintf(stderr,"Unsupported type for %s\n",subbranch->GetName());
                
               }


            }
            

         } else {
            
            fprintf(stderr,"non TBranchElement not implemented yet\n");
            continue;

         }
        
         if ( subbranch->GetListOfLeaves()->GetEntries() != 1 ) {
            fprintf(stderr,"%s unexpectedly has more or less than one leaf (%d)\n",
                    subbranch->GetName(),  subbranch->GetListOfLeaves()->GetEntries() );
            continue;
         }
         TLeaf *leaf = (TLeaf*)subbranch->GetListOfLeaves()->At(0);
         
         if (strlen(leaf->GetTypeName()) == 0) continue;
         
         if (type.Length()==0) type=leaf->GetTypeName() ;
         
         AddDescriptor( new TProxyDescriptor( leaf->GetName(), type, subbranch->GetName() ) );

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
      fprintf(hf,",\n      %-20s(&fDirector,\"%s\")",data->GetName(), data->GetBranchName());
   }
   
   fprintf(hf,    "\n      { }\n");

   // Other functions.
   fprintf(hf,"   ~%s() { }\n",classname.Data());
   fprintf(hf,"   void    Begin(TTree *tree);\n");
   fprintf(hf,"   void    Init(TTree *tree);\n");
   fprintf(hf,"   Bool_t  Notify();\n");
   fprintf(hf,"   Bool_t  Process(Int_t entry);\n");
   fprintf(hf,"   Bool_t  ProcessCut(Int_t entry);\n");
   fprintf(hf,"   void    ProcessFill(Int_t entry);\n");
   fprintf(hf,"   void    SetOption(const char *option) { fOption = option; }\n");
   fprintf(hf,"   void    SetObject(TObject *obj) { fObject = obj; }\n");
   fprintf(hf,"   void    SetInputList(TList *input) {fInput = input;}\n");
   fprintf(hf,"   TList  *GetOutputList() const { return fOutput; }\n");
   fprintf(hf,"   void    Terminate();\n");
   fprintf(hf,"\n\n");

   fprintf(hf,"//inject the user's code\n");
   fprintf(hf,"   #include \"%s\"\n",fScript.Data());

   // Close the class.
   fprintf(hf,"};\n");
   fprintf(hf,"\n");
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
