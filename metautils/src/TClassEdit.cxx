// @(#)root/base:$Name:  $:$Id: TClassEdit.cxx,v 1.8 2004/02/03 22:19:45 brun Exp $
// Author: Victor Perev   04/10/2003
//         Philippe Canal 05/2004

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "TClassEdit.h"
#include <ctype.h>
#include "Rstrstream.h"

// CINT's API.
#include "Api.h"

namespace std {} using namespace std;

//______________________________________________________________________________
int   TClassEdit::STLKind(const char *type)
{
//      Converts STL container name to number. vector -> 1, etc..

   static const char *stls[] =                  //container names
   {"any","vector","list","deque","map","multimap","set","multiset",0};

//              kind of stl container
   for(int k=1;stls[k];k++) {if (strcmp(type,stls[k])==0) return k;}
   return 0;
}

//______________________________________________________________________________
int   TClassEdit::STLArgs(int kind)
{
//      Return number of arguments for STL container before allocator

   static const char  stln[] =// min number of container arguments
   {    1,       1,     1,      1,    3,         3,    2,        2 };

   return stln[kind];
}

//______________________________________________________________________________
bool TClassEdit::IsDefAlloc(const char *allocname, const char *classname)
{
   // return whether or not 'allocname' is the STL default allocator for type
   // 'classname'

   string a = CleanType(allocname);
   string k = CleanType(classname);
   if (a=="alloc")                              return true;
   if (a=="__default_alloc_template<true,0>")   return true;
   if (a=="__malloc_alloc_template<0>")         return true;

   string ts("allocator<"); ts += k; ts+=">";
   if (a==ts) return true;

   ts = "allocator<"; ts += k; ts+=" >";
   if (a==ts) return true;

   return false;
}

//______________________________________________________________________________
bool TClassEdit::IsDefAlloc(const char *allocname,
                            const char *keyclassname,
                            const char *valueclassname)
{
   // return whether or not 'allocname' is the STL default allocator for a key
   // of type 'keyclassname' and a value of type 'valueclassname'

   if (IsDefAlloc(allocname,keyclassname)) return true;


   string a = CleanType(allocname);
   string k = CleanType(keyclassname);
   string v = CleanType(valueclassname);

   string stem("allocator<pair<");
   stem += k;
   stem += ",";
   stem += v;

   string ts(stem);
   ts += "> >";

   if (a==ts) return true;

   ts = stem;
   ts += " > >";

   if (a==ts) return true;

   stem = "allocator<pair<const ";
   stem += k;
   stem += ",";
   stem += v;

   ts = stem;
   ts += "> >";

   if (a==ts) return true;

   ts = stem;
   ts += " > >";

   if (a==ts) return true;

   if ( keyclassname[strlen(keyclassname)-1] == '*' ) {

      stem = "allocator<pair<";
      stem += k;
      stem += "const";
      stem += ",";
      stem += v;

      string ts(stem);
      ts += "> >";

      if (a==ts) return true;

      ts = stem;
      ts += " > >";

      if (a==ts) return true;

      stem = "allocator<pair<const ";
      stem += k;
      stem += "const";
      stem += ",";
      stem += v;

      ts = stem;
      ts += "> >";

      if (a==ts) return true;

      ts = stem;
      ts += " > >";

      if (a==ts) return true;

   }

   return false;
}

//______________________________________________________________________________
bool TClassEdit::IsDefComp(const char *compname, const char *classname)
{
   // return whether or not 'compare' is the STL default comparator for type
   // 'classname'

   string c = CleanType(compname);
   string k = CleanType(classname);

   // The default compartor is std::less<classname> which is usually stored
   // in CINT as less<classname>

   string stdless("less<");
   stdless += k;
   if (stdless[stdless.size()-1]=='>') stdless += " >";
   else stdless += ">";

   if (stdless == c) return true;

   stdless.insert(0,"std::");
   if (stdless == c) return true;

   return false;
}

//______________________________________________________________________________
int TClassEdit::GetSplit(const char *type, vector<string>& output)
{
   ///////////////////////////////////////////////////////////////////////////
   //  Stores in output (after emptying it) the splited type.
   //  Return the number of elements stored.
   //
   //  First in list is the template name or is empty
   //         "vector<list<int>,alloc>**" to "vector" "list<int>" "alloc" "**"
   //   or    "TNamed*" to "" "TNamed" "*"
   ///////////////////////////////////////////////////////////////////////////

   int keepConst = 0;
   int keepInnerConst = 1;

   output.clear();
   if (strlen(type)==0) return 0;

   string full = CleanType(type, keepInnerConst );
   const char *t = full.c_str();
   const char *c = strchr(t,'<');

   string stars;
   const char *starloc = t+strlen(t)-1;
   if ( (*starloc)=='*' ) {
      while( (*(starloc-1))=='*' ) { starloc--; }
      stars = starloc;
      full.replace(strlen(t)-strlen(starloc),strlen(starloc),1,'\0');
   }

   if (c) {
      //we have 'something<'
      output.push_back(string(full,0,c-t));
   } else {
      //empty
      output.push_back(string()); c=t-1;
   }
   do {
      output.push_back(CleanType(c+1,keepConst,&c));
   } while(*c!='>' && *c);

   if (*c=='>') {
      // See what's next!
      if (*(c+1)==':') {
         // we have a name specified inside the class/namespace
         // For now we keep it in one piece
         output.push_back((c+1));
      }
   }
   if (stars.length()) output.push_back(stars);
   return output.size();
}


//______________________________________________________________________________
string TClassEdit::CleanType(const char *typeDesc, int mode, const char **tail)
{
   ///////////////////////////////////////////////////////////////////////////
   //      Cleanup type description, redundant blanks removed
   //      and redundant tail ignored
   //      return *tail = pointer to last used character
   //      if (mode==0) keep keywords
   //      if (mode==1) remove keywords outside the template params
   //      if (mode>=2) remove the keywords everywhere.
   //      if (tail!=0) cut before the trailing *
   //
   //      The keywords currently are: "const" , "volatile" removed
   //
   //
   //      CleanType(" A<B, C< D, E> > *,F,G>") returns "A<B,C<D,E> >*"
   ///////////////////////////////////////////////////////////////////////////

   static const char* remove[] = {"class","const","volatile",0};


   string result;
   int lev=0,kbl=1;
   const char* c;

   for(c=typeDesc;*c;c++) {
      if (c[0]==' ') {
         if (kbl)       continue;
         if (!isalnum(c[ 1]) && c[ 1] !='_')    continue;
      }
      if (kbl && (mode>=2 || lev==0)) { //remove "const' etc...
         int done = 0;
         int n = (mode) ? 999 : 1;

         // loop on all the keywords we want to remove
         for (int k=0; k<n && remove[k]; k++) {
           int rlen = strlen(remove[k]);

           // Do we have a match
           if (strncmp(remove[k],c,rlen)) continue;

           // make sure that the 'keyword' is not part of a longer indentifier
           if (isalnum(c[rlen]) || c[rlen]=='_' ||  c[rlen]=='$') continue;

           c+=rlen-1; done = 1; break;
         }
         if (done) continue;
      }

      kbl = (!isalnum(c[ 0]) && c[ 0]!='_' && c[ 0]!='$');

      if (*c == '<')   lev++;
      if (lev==0 && !isalnum(*c)) {
         if (!strchr("*:_$ ",*c)) break;
      }
      if (c[0]=='>' && result.size() && result[result.size()-1]=='>') result+=" ";

      result += c[0];

      if (*c == '>')    lev--;
   }
   if(tail) *tail=c;
   return result;
}

//______________________________________________________________________________
string TClassEdit::ShortType(const char *typeDesc, int mode)
{
   /////////////////////////////////////////////////////////////////////////////
   // Return the absolute type of typeDesc.
   // E.g.: typeDesc = "class const volatile TNamed**", returns "TNamed**".
   // if (mode&1) remove last "*"s                     returns "TNamed"
   // if (mode&2) remove default allocators from STL containers
   // if (mode&4) remove all     allocators from STL containers
   // if (mode&8) return inner class of stl container. list<innerClass>
   // if (mode&16) return deapest class of stl container. vector<list<deapest>>
   /////////////////////////////////////////////////////////////////////////////

   //fprintf(stderr,"calling ShortType with mode %d\n",mode);
//    int imode = mode;

   string full = CleanType(typeDesc, 1);
   string answ;
   int tailLoc=0;

   // get list of all arguments
   vector<string> arglist;
   int narg = GetSplit(full.c_str(),arglist);

   if (narg==0) return typeDesc;

//      fprintf(stderr,"calling ShortType %d for %s with narg %d\n",imode,typeDesc,narg);
//      {for (int i=0;i<narg;i++) fprintf(stderr,"calling ShortType %d for %s with %d %s \n",
//                                        imode,typeDesc,i,arglist[i].c_str());
//      }
   if (arglist[narg-1][0]=='*') {
      if ((mode&1)==0) tailLoc = narg-1;
      narg--;
   }
   mode &= (~1);

//    fprintf(stderr,"calling ShortType %d for %s with narg %d tail %d\n",imode,typeDesc,narg,tailLoc);

   //kind of stl container
   int kind = STLKind(arglist[0].c_str());
   int iall = STLArgs(kind);

   // Only class is needed
   if (mode&(8|16)) {
      while(narg-1>iall) {arglist.pop_back(); narg--;}
      if (!arglist[0].empty() && tailLoc) {
         tailLoc = 0;
      }
      arglist[0].erase(0,999);
      mode&=(~8);
   }

   if (mode & kDropStlDefault) mode |= kDropDefaultAlloc;

   if (kind) {
      bool allocRemoved = false;

      if ( mode & (kDropDefaultAlloc|kDropAlloc) ) {
         // remove allocators


         if (narg-1 == iall+1) {
            // has an allocator specified
            bool dropAlloc = false;
            if (mode & kDropAlloc) {

               dropAlloc = true;

            } else if (mode & kDropDefaultAlloc) {
               switch (kind) {
                  case kVector:
                  case kList:
                  case kDeque:
                  case kSet:
                  case kMultiSet:
                     dropAlloc = IsDefAlloc(arglist[iall+1].c_str(),arglist[1].c_str());
                     break;
                  case kMap:
                  case kMultiMap:
                     dropAlloc = IsDefAlloc(arglist[iall+1].c_str(),arglist[1].c_str(),arglist[2].c_str());
                     break;
                  default:
                     dropAlloc = false;
               }

            }
            if (dropAlloc) {
               narg--;
               allocRemoved = true;
            }
         } else {
            // has no allocator specified (hence it is already removed!)
            allocRemoved = true;
         }
      }

      if ( allocRemoved && (mode & kDropStlDefault) && narg-1 == iall) { // remove default comparator
         if ( IsDefComp( arglist[iall].c_str(), arglist[1].c_str() ) ) {
            narg--;
         }
      } else if ( mode & kDropComparator ) {

         switch (kind) {
            case kVector:
            case kList:
            case kDeque:
               break;
            case kSet:
            case kMultiSet:
            case kMap:
            case kMultiMap:
               if (!allocRemoved && narg-1 == iall+1) {
                  narg--;
                  allocRemoved = true;
               }
               if (narg-1 == iall) narg--;
               break;
            default:
               break;
         }
      }
   }

   //   do the same for all inside
   for (int i=1;i<narg; i++) {
      if (strchr(arglist[i].c_str(),'<')==0) continue;
      arglist[i] = ShortType(arglist[i].c_str(),mode);
   }
   if (!arglist[0].empty()) {answ = arglist[0]; answ +="<";}

   { for (int i=1;i<narg-1; i++) { answ += arglist[i]; answ+=",";} }
   if (narg>1) { answ += arglist[narg-1]; }

   if (!arglist[0].empty()) {
      if ( answ.at(answ.size()-1) == '>') {
         answ += " >";
      } else {
         answ += '>';
      }
   }
   if (tailLoc) answ += arglist[tailLoc];

//     fprintf(stderr,"2. mode %d reduce \"%s\" into \"%s\"\n",
//             imode, typeDesc,answ.c_str());
   return answ;
}

//______________________________________________________________________________
int TClassEdit::IsSTLCont(const char *ty,int testAlloc)
{
   //  ty:  type name like: vector<list<classA,allocator>,allocator>
   //  testAlloc: 1=test allocator, if it is not default result is negative
   //  result:     0=not stl container
   //              abs(result): code of container 1=vector,2=list,3=deque,4=map
   //                           5=multimap,6=set,7=multiset
   //              +ve: it is vector or list with default allocator to any depth
   //                   like vector<list<vector<int>>>
   //  -ve: if other then vector or int, or non default allocator
   //             like vector<deque<int>> has answer -1
   ////////////////////////////////////////////////////////////////////////////////

   if (strchr(ty,'<')==0) return 0;

   int k = (testAlloc) ? 2:0;
   string full = ShortType(ty,k);

   vector<string> arglist;
   int numb = GetSplit(full.c_str(),arglist);

   if ( arglist[0].length()>0 && arglist[numb-1][0]=='*' ) numb--;
   
   if ( arglist[numb-1][0]==':' ) {
      // we have a nested name
      return 0;
   }

   int kind = STLKind(arglist[0].c_str());

   if (kind==kVector || kind==kList ) {

      if (testAlloc && numb-1 > STLArgs(kind)) {

         kind = -kind;

      } else {

         k = IsSTLCont(arglist[1].c_str(),testAlloc);
         if (k<0) kind = -kind;

      }
   }

   if(kind>2) kind = - kind;
   return kind;
}

//______________________________________________________________________________
bool TClassEdit::IsStdClass(const char *classname)
{
  // return true if the class belond to the std namespace

  if ( strcmp(classname,"string")==0 ) return true;
  if ( strncmp(classname,"pair<",strlen("pair<"))==0) return true;
  if ( strcmp(classname,"allocator")==0) return true;
  if ( strncmp(classname,"allocator<",strlen("allocator<"))==0) return true;

  return IsSTLCont(classname) != 0;

}


//______________________________________________________________________________
bool TClassEdit::IsVectorBool(const char *name) {
   vector<string> splitName;
   TClassEdit::GetSplit(name,splitName);

   return ( TClassEdit::STLKind( splitName[0].c_str() ) == TClassEdit::kVector)
      && ( splitName[1] == "bool" || splitName[1]=="Bool_t");
};

//______________________________________________________________________________
namespace {
   static bool ShouldReplace(const char *name)
   {
      // This helper function indicates whether we really want to replace
      // a type.

      const char *excludelist [] = {"Char_t","Short_t","Int_t","Long_t","Float_t",
                                    "Int_t","Double_t","Double32_t",
                                    "UChar_t","UShort_t","UInt_t","ULong_t","UInt_t",
                                    "Long64_t","ULong64_t","Bool_t"};

      for (unsigned int i=0; i < sizeof(excludelist)/sizeof(excludelist[0]); ++i) {
         if (strcmp(name,excludelist[i])==0) return false;
      }

      return true;
   }
}

//______________________________________________________________________________
string TClassEdit::ResolveTypedef(const char *tname, bool resolveAll)
{

   // Return the name of type 'tname' with all its typedef components replaced
   // by the actual type its points to
   // For example for "typedef MyObj MyObjTypedef;"
   //    vector<MyObjTypedef> return vector<MyObjTypedef>
   //

   if ( strchr(tname,'<')==0 && (tname[strlen(tname)-1]!='*') ) {

      if ( strchr(tname,':')!=0 ) {
         // We have a namespace an we have to check it first :(
         
         int slen = strlen(tname);
         for(int k=0;k<slen;++k) {
            if (tname[k]==':') {
               if (k+1>=slen || tname[k+1]!=':') {
                  // we expected another ':'
                  return tname;
               }
               if (k) {
                  string base(tname, 0, k);
                  G__ClassInfo info(base.c_str());
                  if (!info.IsLoaded()) {
                     // the nesting namespace is not declared
                     return tname;
                  }
               }
            }
         }
      }

      // We have a very simple type

      if (resolveAll || ShouldReplace(tname)) {
         G__TypedefInfo t;
         t.Init(tname);
         if (t.IsValid()) return t.TrueName();
      }
      return tname;
   } 

   int len = strlen(tname);
   string input(tname);
 #ifdef R__SSTREAM
   // This is the modern implementation
   stringstream answ;
#else
   // This is deprecated in the C++ standard
   strstream answ;
#endif

   int prev = 0;
   for (int i=0; i<len; ++i) {
      switch (tname[i]) {
         case '<':
         case '>':
         case '*':
         case ' ':
         case '&':
         case ',':
         {
            char keep = input[i];
            string temp( input, prev,i-prev );

            if (resolveAll || ShouldReplace(temp.c_str())) {
               answ << ResolveTypedef( temp.c_str(), resolveAll);
            } else {
               answ << temp;
            }
            answ << keep;
            prev = i+1;
         }
      }
   }
   const char *last = &(input.c_str()[prev]);
   if (resolveAll || ShouldReplace(last)) {
      answ << ResolveTypedef( last, resolveAll);
   } else {
      answ << last;
   }
#ifndef R__SSTREAM
   // Deprecated case
   answ << ends;
#endif
   return answ.str();

}
