// @(#)root/html:$Name:  $:$Id: THtml.h,v 1.5 2001/07/06 07:34:39 brun Exp $
// Author: Nenad Buncic   18/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_THtml
#define ROOT_THtml


////////////////////////////////////////////////////////////////////////////
//                                                                        //
// THtml                                                                  //
//                                                                        //
// Html makes a documentation for all ROOT classes                        //
// using Hypertext Markup Language 2.0                                    //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include "TCint.h"
#include "Api.h"
#undef G__FPROTO_H
#include "fproto.h"
#endif
//#include "Type.h"
//#include "G__ci.h"
//#include "Typedf.h"
//#include "Class.h"

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TDictionary
#include "TDictionary.h"
#endif
#ifndef ROOT_TMap
#include "TMap.h"
#endif
#ifndef ROOT_TList
#include "TList.h"
#endif
#ifndef ROOT_THashList
#include "THashList.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TClass;
class TVirtualPad;
class TPaveText;

class THtml : public TObject {

public:
//_____________________________________________________
// doc for TDocElement
//
   class TDocElement: public TObject {
   public:
      inline TDocElement(const TString doc, 
         const TString sourceLink="", const TString docLink=""): 
      fDoc(doc), fSourceLink(sourceLink),
         fDocLink(docLink) {};

      inline const TString& GetDoc() const {
         return fDoc;}
      inline void AddToDoc(const char* doc) {
         fDoc+=doc;
      }
      inline const TString& GetDocLink() const {
         return fDocLink;}
      inline const TString& GetSourceLink() const {
         return fSourceLink;}

      inline TDocElement& SetDoc(const TString doc) {
         fDoc=doc; 
         return *this; }
      inline TDocElement& SetDocLink(const TString docLink) {
         fDocLink=docLink; 
         return *this; }

   private:
      TString fDoc; // documentation for the element
      TString fSourceLink; // link to source where this element is defined
      TString fDocLink; // link to documentation of this element
      ClassDef(TDocElement, 0)
   };

   //______________________________________________________
   // 
   // Addendum to TROOT's list of classes, containing
   // * additional, non-linkdef'ed classes
   // * enums 
   // * structs
   // * typedefs
   class TLocalType: public TDictionary {
   public:
      enum ESpec {
         kUndefined,
         kClass,
         kEnum,
         kStruct,
         kTypedef,
         kNumSpecs
      };

      inline TLocalType(const char* nameFQI, 
         ESpec spec,
         const char* declFileName, Int_t declFileLine,
         const char* realTypdefName = 0):
         fName(nameFQI), 
         fDeclName(declFileName),
	 fTypedefReal(realTypdefName), 
         fSpec(spec), fDeclLine(declFileLine) {}

      inline virtual ~TLocalType() {}
      inline const char* GetName() const {
         return fName; }
      inline const char* GetDeclFileName() const {
         return fDeclName;
      }
      inline Int_t GetDeclFileLine() const {
         return fDeclLine;
      }
      inline const char* RealTypedefName() const {
         return fTypedefReal; }
      inline ESpec Spec() const {
         return fSpec;
      }
      inline int Compare(const TObject *obj) const {
         return strcmp(fName, ((TDictionary*)obj)->GetName());
      }
      const char *GetTitle() const {
         return 0;
      }
      ULong_t Hash() const {
         return fName.Hash();
      }
      Long_t Property() const {
         return 0;
      }

   private:
      TString fName; // type name
      TString fDeclName; // file name containing declaration
      TString fTypedefReal; // "A" for "typedef A B;"
      ESpec fSpec; // type of type
      Int_t fDeclLine; // line number in file containing declaration
      ClassDef(TLocalType,0) // additional types found while parsing
   };

   
//_____________________________________________________
// doc for TParseStack
//
   class TParseStack{
   public:
      enum EContext {
         kTop, // topmost - i.e. no - context
         kComment, // within comment /* ... */
         kBlock, // { ... }
         kParameter, // ( ... )
         kTemplate, // < ... >
         kArray, // [ ... ]
         kString, // " ... "
         kUndefined
      };
      enum EBlockSpec {
         kClassDecl, // class ...{ (or enum or struct)
         kMethodDef, // some method is defined here
         kMethodDefCand, // some method might be defined here (not sure, e.g. unknown param type)
         kNamespace, // within namespace ...{
         kBlkUndefined
      };

      //____________________________________________
      // doc for TParseElement
      //
      class TParseElement: public TNamed {
      public:
         inline TParseElement(EContext ctx=kUndefined, EBlockSpec bsp=kBlkUndefined, 
            const char* fName=0, const char* fTitle=0, TDictionary* dict=0):
         TNamed(fName, fTitle), fCtx(ctx), fBsp(bsp), fPStrUsing(0), fDict(dict) {};
         inline TParseElement(char cStart, EBlockSpec bsp=kBlkUndefined,
            const char* fName=0, const char* fTitle=0, TDictionary* dict=0):
         TNamed(fName, fTitle), fBsp(bsp), fPStrUsing(0), fDict(dict) {
            if(cStart=='/') fCtx=kComment;
            else if (cStart=='{') fCtx=kBlock;
            else if (cStart=='(') fCtx=kParameter;
            else if (cStart=='[') fCtx=kArray;
            else if (cStart=='<') fCtx=kTemplate;
            else if (cStart=='"') fCtx=kString;
         }

         virtual ~TParseElement() {
            if (fPStrUsing) delete fPStrUsing;
         }

         inline void AddUsing(const char* cIdentifier) {
            if (!fPStrUsing) fPStrUsing=new TString(" ");
            *fPStrUsing+=cIdentifier;
            *fPStrUsing+=" "; // as delimiter
         }
         inline EContext Context() const { 
            return fCtx; }
         inline EBlockSpec BlockSpec() const {
            return fBsp; }
         inline TDictionary* Dict() const {
            return fDict; }
         inline TParseElement& SetContext(EContext ctx) {
            fCtx=ctx;
            return *this;
         }
         inline Bool_t IsUsing(const char* cIdentifier) const {
            if (strcmp(cIdentifier, fName)==0) return kTRUE;
            if (!fPStrUsing) return kFALSE;
            TString strSearch(" ");
            strSearch+=cIdentifier;
            strSearch+=" ";
            return (fPStrUsing->Contains(strSearch));
         }
         inline TString* GetUsing() const {
            return fPStrUsing;
         }

         inline const char* GetCloseTag() const {
            switch (fCtx) {
            case kComment: return "*/"; 
            case kBlock: return "}";
            case kParameter: return ")";
            case kTemplate: return ">";
            case kArray: return "]";
            case kString: return "\"";
	    default: return NULL;
            }
            return NULL;
         }

      private:
         EContext     fCtx;
         EBlockSpec   fBsp;
         TString*     fPStrUsing;
         TDictionary* fDict;
         ClassDef(TParseElement, 0)
      };

      inline TParseStack(){
         Push(kTop);

         // fill fTypes
         // start with classes
         G__ClassInfo clinfo;
         while (clinfo.Next())
            fTypes.Add(new TNamed(clinfo.Fullname(), clinfo.Title()));

         G__TypedefInfo typeinfo;
         while (typeinfo.Next())
            fTypedefs.Add(new TNamed(typeinfo.Name(), typeinfo.TrueName()));
      };
      inline virtual ~TParseStack(){
         if (fStack.GetSize()>1) {
            TString strStack;
            TIter psi(&fStack);
            TParseElement* pe;
            while ((pe=(TParseElement*) psi())) {
               strStack+=" - ";
               switch (pe->Context()){
                  case kComment: strStack+="Comment"; break; 
                  case kBlock: strStack+="Block ("; 
                     switch (pe->BlockSpec()){
                        case kClassDecl: 
                           strStack+="ClassDecl ";
                           strStack+=pe->GetName(); break;
                        case kNamespace: 
                           strStack+="Namespace)"; 
                           strStack+=pe->GetName(); break;
                        case kBlkUndefined: strStack+="BlkUndefined: "; 
                           strStack+=pe->GetName(); strStack+="***";
                           strStack+=pe->GetTitle(); strStack+="***";
                           break;
		     default: break;
                     }
                     strStack+=")";
                     break; 
                  case kParameter: strStack+="Parameter"; break; 
                  case kTemplate: strStack+="Template"; break; 
                  case kArray: strStack+="Array"; break; 
                  case kString: strStack+="String"; break; 
	       default: break;
               }
               strStack+="\n";
               fStack.Remove(pe);
            }
            strStack.Remove(strStack.Length()-3);

//printf("Warning in <TParseStack::~TParseStack>: Stack not empty! Elements:\n %s\n", strStack.Data());
         }
         fStack.Remove(fStack.LastLink());
      };
      inline TParseElement& Push(EContext ctx, EBlockSpec bsp=kBlkUndefined, 
         const char* name="", const char* title="") {
         return Push(new TParseElement(ctx, bsp, name, title));
      }
      inline TParseElement& Push(char cStart, EBlockSpec bsp=kBlkUndefined,
         const char* name="", const char* title="") {
         return Push(new TParseElement(cStart, bsp, name, title));
      }
      inline TParseElement& Push(TParseElement* pe) {
         fStack.AddLast(pe);
         return *pe;
      }
      inline TParseElement* Pop() {
         if (fStack.GetSize()==1) {
//printf("Warning in <TParseStack::Pop>: Stack is already empty!\n");
            return 0;
         } else
            return (TParseElement*) fStack.Remove(fStack.LastLink());
      }
      inline void PopAndDel() {
         delete Pop();
      }
      inline TParseElement& Top() const{
         return *((TParseElement*) fStack.Last());
      }
      inline TList* Stack() {
         return &fStack;
      }

      inline EContext Context() const {
         return Top().Context();
      }

      inline EBlockSpec BlockSpec() const {
         return Top().BlockSpec();
      }

      inline TDictionary* Dict() const {
         return Top().Dict();
      }

      inline void GetFQI(TString& fqi) const{
         // prepend the surrounding class and namepace names 
         // (separated by "::") to whatever is in fqi
         TIter psi(&fStack, kIterBackward);
         TParseElement* pe;
         while ((pe=(TParseElement*) psi()))
            if (pe->Context()==kBlock && 
               (pe->BlockSpec()==kClassDecl || pe->BlockSpec()==kNamespace)){
            fqi.Prepend("::");
            fqi.Prepend(pe->GetName());
            }
      }

      inline const char* IsUsing(const char* cIdentifier) const{
      // returns the part of cIdentifier that is not covered by a using directive
         static char cID[1024];
         char* cFindNext=cID;
         strcpy(cID, cIdentifier);
         
         TIter psi(&fStack);
         TParseElement* pe;

         while (cFindNext) {
            // only look for next part of identifier 
            // this doesn't work if there's a using namespace::class directive...
            char* cCol=strchr(cFindNext, ':');
            if (cCol) *cCol=0;

            while ((pe=(TParseElement*) psi()) && !pe->IsUsing(cFindNext));
            if (pe && cCol) cFindNext=&cCol[2];
            if (!pe) return &cIdentifier[cFindNext-cID];
         }
         return 0;
      }

      inline Bool_t FindTypeFQI(TString& type, TDictionary*& dict) {
         // first try local type
         TString fqi(type);
         GetFQI(fqi);
         // if they are the same this doesn't get us anywhere
         if (strcmp(fqi, type)) {
            if (fTypes.FindObject(fqi)) {
               dict=THtml::GetType(fqi);
               type=fqi;
               return kTRUE;
            }
            if (fTypedefs.FindObject(fqi)) {
               type=fqi;
               return kTRUE;
            }
         }

         dict=THtml::GetType(type);
         // maybe THtml found its own type
         if (dict)
            return kTRUE;

         return (fTypedefs.FindObject(type)) ;
      }

      inline Bool_t FindType(TString& strClassName){
         TDictionary* dict;
         return FindType(strClassName, dict);
      }

      inline Bool_t FindType(TString& strClassName, TDictionary*& dict){
         // find a class named strClassName, if necessary add
         // "using"-used or surrounding classes / namespaces
         Bool_t bFound=FindTypeFQI(strClassName, dict);

         // iterate through parse stack, try to find FQI
         TParseElement* pe=0;
         TIter iPE(&fStack, kFALSE);
         while (!bFound && (pe=(TParseElement*) iPE())) {
            TString *strUsing=pe->GetUsing();
            if (strUsing) {
               const char* pos=strUsing->Data();
	       Int_t end;
               TString strClassUsing;
               ParseWord(pos, end, strClassUsing, ":");
               strClassUsing+=strClassName;
               bFound=FindTypeFQI(strClassUsing, dict);
               if (bFound) strClassName=strClassUsing;
            }
         } // for (parse elements in stack)
         return bFound;
      }

      Bool_t IsInStack(EContext ctx, EBlockSpec bsp=kBlkUndefined) const {
         TIter psi(&fStack, kIterBackward);
         TParseElement* pe;
         while ((pe=(TParseElement*) psi()))
            if (pe->Context()==ctx && 
               (bsp==kBlkUndefined || pe->BlockSpec()==bsp))
               return kTRUE;
         return kFALSE;
      }

      inline void AddCustomType(const char* type) {
         fTypes.Add(new TNamed(type,0));
      }

   private:
      TList fStack;
      THashList fTypes; // hashed list of known types
      THashList fTypedefs; // hashed list of known typedefs

      ClassDef(TParseStack, 0);
   };

protected:
    TString      fXwho;            // by default http://xwho.cern.ch/WHO/people?
  const char    *fSourcePrefix;    // prefix to relative source path
  const char    *fSourceDir;       // source path
  const char    *fOutputDir;       // output directory
        char    *fLine;            // current line
        Int_t    fLen;             // maximum line length
        char    *fCounter;         // counter string
        Bool_t   fEscFlag;         // Flag to mark the symbol must be written "as is"
        char     fEsc;             // The special symbol ("backslash" by default) to mark "the next symbol should not be converted
        TMap    *fMapDocElements;  // map of <TDictionary*, TDocElement*> for all objects for which doc was parsed
        static THashList fgLocalTypes;    // list of types that are not in TROOT::GetClass
        TList fFilesParsed; // list of files on which ExtractDocumentatoin was run

        void    Class2Html(TClass *classPtr, Bool_t force=kFALSE);
        void    ClassDescription(ofstream &out, TClass *classPtr, Bool_t &flag);
        void    ClassTree(TVirtualPad *canvas, TClass *classPtr, Bool_t force=kFALSE);
        Bool_t  CopyHtmlFile(const char *sourceName, const char *destName="");
        void    CreateIndex(const char **classNames, Int_t numberOfClasses);
        void    CreateIndexByTopic(char **filenames, Int_t numberOfNames, Int_t maxLen);
        void    CreateListOfTypes();
        void    DerivedClasses(ofstream &out, TClass *classPtr);
        void    ExpandKeywords(ofstream &out, char *text, TClass *ptr2class, Bool_t &flag, const char *dir="");
        void    ExpandPpLine(ofstream &out, char *line);
   TClass      *GetClass(const char *name, Bool_t load=kTRUE);
  const char   *GetFileName(const char *filename);
        char   *GetSourceFileName(const char *filename);
        char   *GetHtmlFileName(TClass *classPtr);
        Bool_t  IsModified(TClass *classPtr, const Int_t type);
        static Bool_t  IsName(Int_t c);
        static Bool_t  IsWord(Int_t c);
		void	NameSpace2FileName(char *name);
        void    ReplaceSpecialChars(ofstream &out, const char c);
        void    ReplaceSpecialChars(ofstream &out, const char *string);
        void    SortNames(const char **strings, Int_t num, Bool_t type=0);
        char   *StrDup(const char *s1, Int_t n = 1);

   friend Int_t CaseSensitiveSort(const void *name1, const void *name2);
   friend Int_t CaseInsensitiveSort(const void *name1, const void *name2);

   TClass* ParseClassDecl(char* &cfirstLinePos, 
      const TParseStack& parseStack, TString& strClassName);
   TDocElement* AddDocElement(TDictionary* dict, TString& strDoc, const char* filename);
   TDocElement* GetDocElement(TDictionary* dict) const {
      return (TDocElement*)(fMapDocElements?fMapDocElements->GetValue(dict):0);
   }

   Bool_t FindMethodImpl(TString strMethFullName, TList& listMethodSameName, 
      TList& listArgs, TParseStack& parseStack, Bool_t done=kFALSE) const;

public:
                 THtml();
       virtual   ~THtml();
          void   Convert(const char *filename, const char *title, const char *dirname = "");
    const char  *GetSourceDir()  { return fSourceDir; }
    const char  *GetOutputDir()  { return fOutputDir; }
    const char  *GetXwho() const { return fXwho.Data(); }
          void   MakeAll(Bool_t force=kFALSE, const char *filter="*");
          void   MakeClass(const char *className, Bool_t force=kFALSE);
          void   MakeIndex(const char *filter="*");
          void   MakeTree(const char *className, Bool_t force=kFALSE);
          void   SetEscape(char esc='\\') { fEsc = esc; }
          void   SetSourcePrefix(const char *prefix) { fSourcePrefix = prefix; }
          void   SetSourceDir(const char *dir) { fSourceDir = dir; }
          void   SetOutputDir(const char *dir) { fOutputDir = dir; }
          void   SetXwho(const char *xwho) { fXwho = xwho; }
   virtual void  WriteHtmlHeader(ofstream &out, const char *title);
   virtual void  WriteHtmlFooter(ofstream &out, const char *dir="", const char *lastUpdate="",
                                 const char *author="", const char *copyright="");

   void ExtractDocumentation(const char* cFileName, TList* listClassesFound);
   void ExtractClassDocumentation(const TClass* classPtr);
   static Bool_t ParseWord(const char* begin, Int_t &step, 
      const char* allowedChars=0);
   static Bool_t ParseWord(const char* begin, Int_t &step, 
      TString &strWord, const char* allowedChars=0);

   static inline TDictionary* GetType(const char* type) {
      TDictionary* dict=(TDictionary*) gROOT->GetClass(type);
      if (!dict) return (TDictionary*) fgLocalTypes.FindObject(type);
      else return dict;
   }

   const char* GetDoc(TDictionary* dict) const {
      TDocElement* de=GetDocElement(dict);
      return de?de->GetDoc():0;
   }
   TMap* MakeHelp(TClass* cl);
   TPaveText* GetDocPave(TDictionary* dict);


   ClassDef(THtml,0)  //Convert class(es) into HTML file(s)
};

R__EXTERN THtml *gHtml;

#endif
