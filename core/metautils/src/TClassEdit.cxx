// @(#)root/metautils:$Id$
// Author: Victor Perev   04/10/2003
//         Philippe Canal 05/2004

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "TClassEdit.h"
#include <ctype.h>
#include "Rstrstream.h"
#include <set>
// for shared_ptr
#include <memory>

namespace {
   static TClassEdit::TInterpreterLookupHelper *gInterpreterHelper = 0;
}

namespace std {} using namespace std;

//______________________________________________________________________________
static size_t StdLen(const std::string &name, size_t pos = 0)
{
   // Return the length, if any, taken by std:: and any
   // potential inline namespace (well compiler detail namespace).

   size_t len = 0;
   if (name.compare(pos,5,"std::")==0) {
      len = 5;
   }

   return len;
}

//______________________________________________________________________________
static size_t StdLen(const char *name, size_t pos = 0)
{
   // Return the length, if any, taken by std:: and any
   // potential inline namespace (well compiler detail namespace).

   return StdLen(std::string(name),pos);
}

//______________________________________________________________________________
static void RemoveStd(std::string &name, size_t pos = 0)
{
   // Remove std:: and any potential inline namespace (well compiler detail
   // namespace.

   size_t len = StdLen(name,pos);
   if (len) {
      name.erase(pos,len);
   }
}


//______________________________________________________________________________
void TClassEdit::Init(TClassEdit::TInterpreterLookupHelper *helper)
{
   gInterpreterHelper = helper;
}

//______________________________________________________________________________
TClassEdit::TSplitType::TSplitType(const char *type2split, EModType mode) : fName(type2split), fNestedLocation(0)
{
   // default constructor
   TClassEdit::GetSplit(type2split, fElements, fNestedLocation, mode);
}

//______________________________________________________________________________
ROOT::ESTLType TClassEdit::TSplitType::IsInSTL() const
{
   //  type     : type name: vector<list<classA,allocator>,allocator>[::iterator]
   //  result:    0          : not stl container and not declared inside an stl container.
   //             result: code of container that the type or is the scope of the type

   if (fElements[0].empty()) return ROOT::kNotSTL;
   return STLKind(fElements[0].c_str());
}

//______________________________________________________________________________
int TClassEdit::TSplitType::IsSTLCont(int testAlloc) const
{
   //  type     : type name: vector<list<classA,allocator>,allocator>
   //  testAlloc: if true, we test allocator, if it is not default result is negative
   //  result:    0          : not stl container
   //             abs(result): code of container 1=vector,2=list,3=deque,4=map
   //                           5=multimap,6=set,7=multiset
   //             positive val: we have a vector or list with default allocator to any depth
   //                   like vector<list<vector<int>>>
   //             negative val: STL container other than vector or list, or non default allocator
   //                           For example: vector<deque<int>> has answer -1


   if (fElements[0].empty()) return 0;
   int numb = fElements.size();
   if (!fElements[numb-1].empty() && fElements[numb-1][0]=='*') --numb;

   if ( fNestedLocation ) {
      // The type has been defined inside another namespace and/or class
      // this couldn't possibly be an STL container
      return 0;
   }

   int kind = STLKind(fElements[0].c_str());

   if (kind==ROOT::kSTLvector || kind==ROOT::kSTLlist ) {

      int nargs = STLArgs(kind);
      if (testAlloc && (numb-1 > nargs) && !IsDefAlloc(fElements[numb-1].c_str(),fElements[1].c_str())) {

         // We have a non default allocator,
         // let's return a negative value.

         kind = -kind;

      } else {

         // We has a default allocator, let's continue to
         // look inside the argument list.
         int k = TClassEdit::IsSTLCont(fElements[1].c_str(),testAlloc);
         if (k<0) kind = -kind;

      }
   }

   // We return a negative value for anything which is not a vector or a list.
   if(kind>2) kind = - kind;
   return kind;
}

//______________________________________________________________________________
void TClassEdit::TSplitType::ShortType(std::string &answ, int mode)
{
   /////////////////////////////////////////////////////////////////////////////
   // Return the absolute type of typeDesc into the string answ.

   // E.g.: typeDesc = "class const volatile TNamed**", returns "TNamed**".
   // if (mode&1) remove last "*"s                     returns "TNamed"
   // if (mode&2) remove default allocators from STL containers
   // if (mode&4) remove all     allocators from STL containers
   // if (mode&8) return inner class of stl container. list<innerClass>
   // if (mode&16) return deepest class of stl container. vector<list<deepest>>
   // if (mode&kDropAllDefault) remove default template arguments
   /////////////////////////////////////////////////////////////////////////////

   answ.clear();
   int narg = fElements.size();
   int tailLoc = 0;

   if (narg == 0) {
      answ = fName;
      return ;
   }
   //      fprintf(stderr,"calling ShortType %d for %s with narg %d\n",mode,typeDesc,narg);
   //      {for (int i=0;i<narg;i++) fprintf(stderr,"calling ShortType %d for %s with %d %s \n",
   //                                        mode,typeDesc,i,arglist[i].c_str());
   //      }
   if (fElements[narg-1].empty() == false &&
       (fElements[narg-1][0]=='*'
        || fElements[narg-1][0]=='&'
        || fElements[narg-1][0]=='['
        || 0 == fElements[narg-1].compare(0,6,"const*")
        || 0 == fElements[narg-1].compare(0,6,"const&")
        )
       ) {
      if ((mode&1)==0) tailLoc = narg-1;
      narg--;
   }
   mode &= (~1);

   if (fNestedLocation) narg--;

   //    fprintf(stderr,"calling ShortType %d for %s with narg %d tail %d\n",imode,typeDesc,narg,tailLoc);

   //kind of stl container
   int kind = STLKind(fElements[0].c_str());
   int iall = STLArgs(kind);

   // Only class is needed
   if (mode&(8|16)) {
      while(narg-1>iall) { fElements.pop_back(); narg--;}
      if (!fElements[0].empty() && tailLoc) {
         tailLoc = 0;
      }
      fElements[0].clear();
      mode&=(~8);
   }

   if (mode & kDropAllDefault) mode |= kDropStlDefault;
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
                  case ROOT::kSTLvector:
                  case ROOT::kSTLlist:
                  case ROOT::kSTLdeque:
                  case ROOT::kSTLset:
                  case ROOT::kSTLmultiset:
                     dropAlloc = IsDefAlloc(fElements[iall+1].c_str(),fElements[1].c_str());
                     break;
                  case ROOT::kSTLmap:
                  case ROOT::kSTLmultimap:
                     dropAlloc = IsDefAlloc(fElements[iall+1].c_str(),fElements[1].c_str(),fElements[2].c_str());
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
         if ( IsDefComp( fElements[iall].c_str(), fElements[1].c_str() ) ) {
            narg--;
         }
      } else if ( mode & kDropComparator ) {

         switch (kind) {
            case ROOT::kSTLvector:
            case ROOT::kSTLlist:
            case ROOT::kSTLdeque:
               break;
            case ROOT::kSTLset:
            case ROOT::kSTLmultiset:
            case ROOT::kSTLmap:
            case ROOT::kSTLmultimap:
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
#if __cplusplus >= 201103L
   else {
      if ( (mode & kDropStlDefault) && (narg >= 3)) {
         unsigned int offset = (0==strncmp("const ",fElements[0].c_str(),6)) ? 6 : 0;
         offset += (0==strncmp("std::",fElements[0].c_str()+offset,5)) ? 5 : 0;
         if (0 == strcmp(fElements[0].c_str()+offset,"__shared_ptr"))
         {
#ifdef _CONCURRENCE_H
            static const std::string sharedPtrDef = std::to_string(__gnu_cxx::__default_lock_policy); // to_string is C++11
#else
            static const std::string sharedPtrDef = std::to_string(2); // to_string is C++11
#endif
            if (fElements[2] == sharedPtrDef) {
               narg--;
            }
         }
      }
   }
#endif

   //   do the same for all inside
   for (int i=1;i<narg; i++) {
      if (strchr(fElements[i].c_str(),'<')==0) {
         if (mode&kDropStd) {
            unsigned int offset = (0==strncmp("const ",fElements[i].c_str(),6)) ? 6 : 0;
            RemoveStd( fElements[i], offset );
         }
         continue;
      }
      bool hasconst = 0==strncmp("const ",fElements[i].c_str(),6);
      //NOTE: Should we also check the end of the type for 'const'?
      fElements[i] = TClassEdit::ShortType(fElements[i].c_str(),mode);
      if (hasconst && !(mode & TClassEdit::kKeepOuterConst)) {
         // if mode is set to keep the outer const, it will be kept
         // and we do not need to put it back ... 
         // FIXME: why are passing a flag meant for the outer
         // to the handling of the inner?
         fElements[i] = "const " + fElements[i];
      }
   }

   if (!fElements[0].empty()) {answ += fElements[0]; answ +="<";}

   if (mode & kDropAllDefault) {
      int nargNonDefault = 0;
      std::string nonDefName = answ;
      // "superlong" because tLong might turn fName into an even longer name
      std::string nameSuperLong = fName;
      if (gInterpreterHelper)
         gInterpreterHelper->GetPartiallyDesugaredName(nameSuperLong);
      while (++nargNonDefault < narg) {
         // If T<a> is a "typedef" (aka default template params)
         // to T<a,b> then we can strip the "b".
         const char* closeTemplate = " >";
         if (nonDefName[nonDefName.length() - 1] != '>')
            ++closeTemplate;
         string nondef = nonDefName + closeTemplate;
         if (gInterpreterHelper &&
             gInterpreterHelper->IsAlreadyPartiallyDesugaredName(nondef, nameSuperLong))
            break;
         if (nargNonDefault>1) nonDefName += ",";
         nonDefName += fElements[nargNonDefault];
      }
      if (nargNonDefault < narg)
         narg = nargNonDefault;
   }

   { for (int i=1;i<narg-1; i++) { answ += fElements[i]; answ+=",";} }
   if (narg>1) { answ += fElements[narg-1]; }

   if (!fElements[0].empty()) {
      if ( answ.at(answ.size()-1) == '>') {
         answ += " >";
      } else {
         answ += '>';
      }
   }
   if (fNestedLocation) {
      // Treat X pf A<B>::X
      fElements[fNestedLocation] = TClassEdit::ShortType(fElements[fNestedLocation].c_str(),mode);
      answ += fElements[fNestedLocation];
   }
   // tail is not a type name, just [2], &, * etc.
   if (tailLoc) answ += fElements[tailLoc];
}


//______________________________________________________________________________
ROOT::ESTLType TClassEdit::STLKind(const char *type, size_t len)
{
   // Converts STL container name to number. vector -> 1, etc..
   // If len is greater than 0, only look at that many characters in the string.

   unsigned char offset = 0;
   if (strncmp(type,"const ",6)==0) { offset += 6; }
   offset += StdLen(type,offset);

   //container names
   static const char *stls[] =
      { "any", "vector", "list", "deque", "map", "multimap", "set", "multiset", "bitset", 0};
   static const size_t stllen[] =
      { 3, 6, 4, 5, 3, 8, 3, 8, 6, 0};
   static const ROOT::ESTLType values[] =
      {  ROOT::kNotSTL, ROOT::kSTLvector,
         ROOT::kSTLlist, ROOT::kSTLdeque,
         ROOT::kSTLmap, ROOT::kSTLmultimap,
         ROOT::kSTLset, ROOT::kSTLmultiset,
         ROOT::kSTLbitset, ROOT::kNotSTL
      };

   // kind of stl container
   if (len) {
      len -= offset;
      for(int k=1;stls[k];k++) {
         if (len == stllen[k]) {
            if (strncmp(type+offset,stls[k],len)==0) return values[k];
         }
      }         
   } else {
      for(int k=1;stls[k];k++) {if (strcmp(type+offset,stls[k])==0) return values[k];}
   }
   return ROOT::kNotSTL;
}

//______________________________________________________________________________
int   TClassEdit::STLArgs(int kind)
{
//      Return number of arguments for STL container before allocator

   static const char  stln[] =// min number of container arguments
      //     vector, list, deque, map, multimap, set, multiset, bitset
      {    1,     1,    1,     1,   3,        3,   2,        2,      1 };

   return stln[kind];
}

//______________________________________________________________________________
size_t findNameEnd(std::string &full, size_t pos)
{
   int level = 0;
   for(size_t i = pos; i < full.length(); ++i) {
      switch(full[i]) {
         case '<': { ++level; break; }
         case '>': {
            if (level == 0) return i;
            else --level;
            break;
         }
         case ',': {
            if (level == 0) return i;
            break;
         }
         default: break;
      }
   }
   return full.length();
}

//______________________________________________________________________________
bool TClassEdit::IsDefAlloc(const char *allocname, const char *classname)
{
   // return whether or not 'allocname' is the STL default allocator for type
   // 'classname'

   string a = allocname;
   RemoveStd(a);

   if (a=="alloc")                              return true;
   if (a=="__default_alloc_template<true,0>")   return true;
   if (a=="__malloc_alloc_template<0>")         return true;

   const static int alloclen = strlen("allocator<");
   if (a.compare(0,alloclen,"allocator<") != 0) {
      return false;
   }
   size_t pos = alloclen;

   pos += StdLen(a,pos);

   string k = classname;
   size_t pos2 = StdLen(k);
   if (pos2) k = classname + pos2;

   if (a.compare(pos,k.length(),k) != 0) {
      // Now we need to compare the normalized name.
      size_t end = findNameEnd(a,pos);

      std::string valuepart;
      GetNormalizedName(valuepart,a.substr(pos,end-pos).c_str());

      std::string norm_value;
      GetNormalizedName(norm_value,k.c_str());

      if (valuepart != norm_value) {
         return false;
      }
      pos = end;
   } else {
      pos += k.length();
   }

   if (a.compare(pos,1,">")!=0 && a.compare(pos,2," >")!=0) {
      return false;
   }

   return true;
}

//______________________________________________________________________________
bool TClassEdit::IsDefAlloc(const char *allocname,
                            const char *keyclassname,
                            const char *valueclassname)
{
   // return whether or not 'allocname' is the STL default allocator for a key
   // of type 'keyclassname' and a value of type 'valueclassname'

   if (IsDefAlloc(allocname,keyclassname)) return true;

   string a = allocname;
   RemoveStd(a);

   const static int alloclen = strlen("allocator<");
   if (a.compare(0,alloclen,"allocator<") != 0) {
      return false;
   }
   size_t pos = alloclen;

   pos += StdLen(a,pos);

   const static int pairlen = strlen("pair<");
   if (a.compare(pos,pairlen,"pair<") != 0) {
      return false;
   }
   pos += pairlen;

   const static int constlen = strlen("const ");
   if (a.compare(pos,constlen,"const ") == 0) {
      pos += constlen;
   }

   pos += StdLen(a,pos);

   string k = keyclassname;
   size_t pos2 = StdLen(k);
   if (pos2) k = keyclassname + pos2;

   if (a.compare(pos,k.length(),k) != 0) {
      // Now we need to compare the normalized name.
      size_t end = findNameEnd(a,pos);

      std::string keypart;
      GetNormalizedName(keypart,a.substr(pos,end-pos).c_str());

      std::string norm_key;
      GetNormalizedName(norm_key,k.c_str());

      if (keypart != norm_key) {
         if ( k[k.length()-1] == '*' ) {
            // also check with a trailing 'const'.
            keypart += "const";
            if (keypart != norm_key) {
               return false;
            }
         } else {
            return false;
         }
      }
      pos = end;
   } else {
      pos += k.length();
   }

   if (a[pos] != ',') {
      return false;
   }
   ++pos;

   pos += StdLen(a,pos);

   string v = valueclassname;
   size_t pos3 = StdLen(v);
   if (pos3) v = valueclassname + pos3;

   if (a.compare(pos,v.length(),v) != 0) {
      // Now we need to compare the normalized name.
      size_t end = findNameEnd(a,pos);

      std::string valuepart;
      GetNormalizedName(valuepart,a.substr(pos,end-pos).c_str());

      std::string norm_value;
      GetNormalizedName(norm_value,k.c_str());

      if (valuepart != norm_value) {
         return false;
      }
      pos = end;
   } else {
      pos += v.length();
   }

   if (a.compare(pos,1,">")!=0 && a.compare(pos,2," >")!=0) {
      return false;
   }

   return true;
}

//______________________________________________________________________________
bool TClassEdit::IsDefComp(const char *compname, const char *classname)
{
   // return whether or not 'compare' is the STL default comparator for type
   // 'classname'

   string c = compname;

   size_t pos = StdLen(c);

   const static int lesslen = strlen("less<");
   if (c.compare(pos,lesslen,"less<") != 0) {
      return false;
   }
   pos += lesslen;

   string k = classname;
   if (c.compare(pos,k.length(),k) != 0) {
      // Now we need to compare the normalized name.
      size_t end = findNameEnd(c,pos);

      std::string keypart;
      GetNormalizedName(keypart,c.substr(pos,end-pos).c_str());

      std::string norm_key;
      GetNormalizedName(norm_key,k.c_str());

      if (keypart != norm_key) {
         return false;
      }
      pos = end;
   } else {
      pos += k.length();
   }

   if (c.compare(pos,1,">")!=0 && c.compare(pos,2," >")!=0) {
      return false;
   }

   return true;
}

//______________________________________________________________________________
void TClassEdit::GetNormalizedName(std::string &norm_name, const char *name)
{
   // Return the normalized name.  See TMetaUtils::GetNormalizedName.
   //
   // Return the type name normalized for ROOT,
   // keeping only the ROOT opaque typedef (Double32_t, etc.) and
   // removing the STL collections default parameter if any.
   //
   // Compare to TMetaUtils::GetNormalizedName, this routines does not
   // and can not add default template parameters.

   norm_name = name;

   // Remove the std:: and default template argument and insert the Long64_t and change basic_string to string.
   TClassEdit::TSplitType splitname(norm_name.c_str(),(TClassEdit::EModType)(TClassEdit::kLong64 | TClassEdit::kDropStd | TClassEdit::kDropStlDefault | TClassEdit::kKeepOuterConst));
   splitname.ShortType(norm_name,TClassEdit::kDropStd | TClassEdit::kDropStlDefault );

   // Depending on how the user typed their code, in particular typedef
   // declarations, we may end up with an explicit '::' being
   // part of the result string.  For consistency, we must remove it.
   if (norm_name.length()>2 && norm_name[0]==':' && norm_name[1]==':') {
      norm_name.erase(0,2);
   }
}

//______________________________________________________________________________
string TClassEdit::GetLong64_Name(const char* original)
{
   // Replace 'long long' and 'unsigned long long' by 'Long64_t' and 'ULong64_t'

   if (original==0)
      return "";
   else
      return GetLong64_Name(string(original));
}

//______________________________________________________________________________
string TClassEdit::GetLong64_Name(const string& original)
{
   // Replace 'long long' and 'unsigned long long' by 'Long64_t' and 'ULong64_t'

   static const char* longlong_s  = "long long";
   static const char* ulonglong_s = "unsigned long long";
   static const unsigned int longlong_len  = strlen(longlong_s);
   static const unsigned int ulonglong_len = strlen(ulonglong_s);

   string result = original;

   int pos = 0;
   while( (pos = result.find(ulonglong_s,pos) ) >=0 ) {
      result.replace(pos, ulonglong_len, "ULong64_t");
   }
   pos = 0;
   while( (pos = result.find(longlong_s,pos) ) >=0 ) {
      result.replace(pos, longlong_len, "Long64_t");
   }
   return result;
}

//______________________________________________________________________________
int TClassEdit::GetSplit(const char *type, vector<string>& output, int &nestedLoc, EModType mode)
{
   ///////////////////////////////////////////////////////////////////////////
   //  Stores in output (after emptying it) the splited type.
   //  Stores the location of the tail (nested names) in nestedLoc (0 indicates no tail).
   //  Return the number of elements stored.
   //
   //  First in list is the template name or is empty
   //         "vector<list<int>,alloc>**" to "vector" "list<int>" "alloc" "**"
   //   or    "TNamed*" to "" "TNamed" "*"
   ///////////////////////////////////////////////////////////////////////////

   nestedLoc = 0;
   output.clear();
   if (strlen(type)==0) return 0;

   int cleantypeMode = 1 /* keepInnerConst */;
   if (mode & kKeepOuterConst) {
      cleantypeMode = 0; /* remove only the outer class keyword */
   }
   string full( mode & kLong64 ? TClassEdit::GetLong64_Name( CleanType(type, cleantypeMode) )
               : CleanType(type, cleantypeMode) );

   // We need to replace basic_string with string.
   bool isString = false;
   bool isStdString = false;
   static const char* basic_string_std = "std::basic_string<char";
   static const unsigned int basic_string_std_len = strlen(basic_string_std);

   if (full.compare(0,basic_string_std_len,basic_string_std) == 0
       && full.size() > basic_string_std_len) {
      isString = true;
      isStdString = true;
   } else if (full.compare(0,basic_string_std_len-5,basic_string_std+5) == 0
              && full.size() > (basic_string_std_len-5)) {
      // no std.
      isString = true;
   }
   if (isString) {
      size_t offset = isStdString ? basic_string_std_len : basic_string_std_len - 5;
      if ( full[offset] == '>' ) {
         // done.
      } else if (full[offset] == ',') {
         ++offset;
         if (full.compare(offset, 5, "std::") == 0) {
            offset += 5;
         }
         static const char* char_traits_s = "char_traits<char>";
         static const unsigned int char_traits_len = strlen(char_traits_s);
         if (full.compare(offset, char_traits_len, char_traits_s) == 0) {
            offset += char_traits_len;
            if ( full[offset] == '>') {
               // done.
            } else if (full[offset] == ' ' && full[offset+1] == '>') {
               ++offset;
               // done.
            } else if (full[offset] == ',') {
               ++offset;
               if (full.compare(offset, 5, "std::") == 0) {
                  offset += 5;
               }
               static const char* allocator_s = "allocator<char>";
               static const unsigned int allocator_len = strlen(allocator_s);
               if (full.compare(offset, allocator_len, allocator_s) == 0) {
                  offset += allocator_len;
                  if ( full[offset] == '>') {
                     // done.
                  } else if (full[offset] == ' ' && full[offset+1] == '>') {
                     ++offset;
                     // done.
                  } else {
                     // Not std::string
                     isString = false;
                  }
               }
            } else {
               // Not std::string
               isString = false;
            }
         } else {
            // Not std::string.
            isString = false;
         }
      } else {
         // Not std::string.
         isString = false;
      }
      if (isString) {
         output.push_back(string());
         if (isStdString && !(mode & kDropStd)) {
            output.push_back("std::string");
         } else {
            output.push_back("string");
         }
         if (offset < full.length()) {
            // Copy the trailing text.
            output.back().append(full.substr(offset+1));
         }
         return output.size();
      }
   }

   if ( mode & kDropStd) {
      unsigned int offset = (0==strncmp("const ",full.c_str(),6)) ? 6 : 0;
      RemoveStd( full, offset );
   }
   const char *t = full.c_str();
   const char *c = strchr(t,'<');

   string stars;
   const unsigned int tlen( full.size() );
   if ( tlen > 0 ) {
      const char *starloc = t + tlen - 1;
      bool hasconst = false;
      if ( (*starloc)=='t'
          && (starloc-t) > 5 && 0 == strncmp((starloc-5),"const",5)
          && ( (*(starloc-6)) == ' ' || (*(starloc-6)) == '*' || (*(starloc-6)) == '&') ) {
         // we are ending on a const.
         starloc -= 4;
         if ((*starloc-1)==' ') {
            // Take the space too.
            starloc--;
         }
         hasconst = true;
      }
      if ( hasconst || (*starloc)=='*' || (*starloc)=='&' || (*starloc)==']' ) {
         bool isArray = ( (*starloc)==']' );
         while( (*(starloc-1))=='*' || (*(starloc-1))=='&' || (*(starloc-1))=='t' || isArray) {
            if (isArray) {
               starloc--;
               isArray = ! ( (*starloc)=='[' );
            } else if ( (*(starloc-1))=='t' ) {
               if ( (starloc-1-t) > 5 && 0 == strncmp((starloc-5),"const",5)
                   && ( (*(starloc-6)) == ' ' || (*(starloc-6)) == '*' || (*(starloc-6)) == '&') ) {
                  // we have a const.
                  starloc -= 5;
                  if ((*starloc-1)==' ') {
                     // Take the space too.
                     starloc--;
                  }
               } else {
                  break;
               }
            } else {
               starloc--;
            }
         }
         stars = starloc;
         const unsigned int starlen = strlen(starloc);
         full.erase(tlen-starlen,starlen);
      } else if (hasconst) {
         stars = starloc;
         const unsigned int starlen = strlen(starloc);
         full.erase(tlen-starlen,starlen);
      }
   }

   if (c) {
      //we have 'something<'
      output.push_back(string(full,0,c-t));

      const char *cursor;
      int level = 0;
      for(cursor = c + 1; *cursor != '\0' && !(level==0 && *cursor == '>'); ++cursor) {
         switch (*cursor) {
            case '<': ++level; break;
            case '>': --level; break;
            case ',':
               if (level == 0) {
                  output.push_back(std::string(c+1,cursor));
                  c = cursor;
               }
               break;
         }
      }
      if (*cursor=='>') {
         if (*(cursor-1) == ' ') {
            output.push_back(std::string(c+1,cursor-1));
         } else {
            output.push_back(std::string(c+1,cursor));
         }
         // See what's next!
         if (*(cursor+1)==':') {
            // we have a name specified inside the class/namespace
            // For now we keep it in one piece
            nestedLoc = output.size();
            output.push_back((cursor+1));
         }
      } else if (level >= 0) {
         // Unterminated template
         output.push_back(std::string(c+1,cursor));
      }
   } else {
      //empty
      output.push_back(string());
      output.push_back(full);
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
   static bool isinit = false;
   static std::vector<size_t> lengths;
   if (!isinit) {
      for (int k=0; remove[k]; ++k) {
         lengths.push_back(strlen(remove[k]));
      }
      isinit = true;
   }

   string result;
   result.reserve(strlen(typeDesc)*2);
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
            int rlen = lengths[k];

            // Do we have a match
            if (strncmp(remove[k],c,rlen)) continue;

            // make sure that the 'keyword' is not part of a longer indentifier
            if (isalnum(c[rlen]) || c[rlen]=='_' ||  c[rlen]=='$') continue;

            c+=rlen-1; done = 1; break;
         }
         if (done) continue;
      }

      kbl = (!isalnum(c[ 0]) && c[ 0]!='_' && c[ 0]!='$' && c[0]!='[' && c[0]!=']' && c[0]!='-' && c[0]!='@');
      // '@' is special character used only the artifical class name used by ROOT to implement the
      // I/O customization rules that requires caching of the input data.

      if (*c == '<' || *c == '(')   lev++;
      if (lev==0 && !isalnum(*c)) {
         if (!strchr("*&:_$ []-@",*c)) break;
      }
      if (c[0]=='>' && result.size() && result[result.size()-1]=='>') result+=" ";

      result += c[0];

      if (*c == '>' || *c == ')')    lev--;
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
   // if (mode&kDropAllDefault) remove default template arguments
   /////////////////////////////////////////////////////////////////////////////

   string answer;

   // get list of all arguments
   if (typeDesc) {
      TSplitType arglist(typeDesc, (EModType) mode);
      arglist.ShortType(answer, mode);
   }

   return answer;
}

//______________________________________________________________________________
bool TClassEdit::IsInterpreterDetail(const char *type)
{
   // Return true if the type is one the interpreter details which are
   // only forward declared (ClassInfo_t etc..)

   size_t len = strlen(type);
   if (len < 2 || strncmp(type+len-2,"_t",2) != 0) return false;

   unsigned char offset = 0;
   if (strncmp(type,"const ",6)==0) { offset += 6; }
   static const char *names[] = { "CallFunc_t","ClassInfo_t","BaseClassInfo_t",
      "DataMemberInfo_t","FuncTempInfo_t","MethodInfo_t","MethodArgInfo_t",
      "TypeInfo_t","TypedefInfo_t",0};

   for(int k=1;names[k];k++) {if (strcmp(type+offset,names[k])==0) return true;}
   return false;
}

//______________________________________________________________________________
bool TClassEdit::IsSTLBitset(const char *classname)
{
   // Return true is the name is std::bitset<number> or bitset<number>

   size_t offset = StdLen(classname);
   if ( strncmp(classname+offset,"bitset<",strlen("bitset<"))==0) return true;
   return false;
}

//______________________________________________________________________________
ROOT::ESTLType TClassEdit::IsSTLCont(const char *type)
{
   //  type     : type name: vector<list<classA,allocator>,allocator>
   //  result:    0          : not stl container
   //             code of container 1=vector,2=list,3=deque,4=map
   //                     5=multimap,6=set,7=multiset

   const char *pos = strchr(type,'<');
   if (pos==0) return ROOT::kNotSTL;

   return STLKind(type,pos-type);
}

//______________________________________________________________________________
int TClassEdit::IsSTLCont(const char *type, int testAlloc)
{
   //  type     : type name: vector<list<classA,allocator>,allocator>
   //  testAlloc: if true, we test allocator, if it is not default result is negative
   //  result:    0          : not stl container
   //             abs(result): code of container 1=vector,2=list,3=deque,4=map
   //                           5=multimap,6=set,7=multiset
   //             positive val: we have a vector or list with default allocator to any depth
   //                   like vector<list<vector<int>>>
   //             negative val: STL container other than vector or list, or non default allocator
   //                           For example: vector<deque<int>> has answer -1

   if (strchr(type,'<')==0) return 0;

   TSplitType arglist( type );
   return arglist.IsSTLCont(testAlloc);
}

//______________________________________________________________________________
bool TClassEdit::IsStdClass(const char *classname)
{
   // return true if the class belond to the std namespace

   classname += StdLen( classname );
   if ( strcmp(classname,"string")==0 ) return true;
   if ( strncmp(classname,"bitset<",strlen("bitset<"))==0) return true;
   if ( strncmp(classname,"pair<",strlen("pair<"))==0) return true;
   if ( strcmp(classname,"allocator")==0) return true;
   if ( strncmp(classname,"allocator<",strlen("allocator<"))==0) return true;
   if ( strncmp(classname,"greater<",strlen("greater<"))==0) return true;
   if ( strncmp(classname,"less<",strlen("less<"))==0) return true;
   if ( strncmp(classname,"auto_ptr<",strlen("auto_ptr<"))==0) return true;

   if ( strncmp(classname,"vector<",strlen("vector<"))==0) return true;
   if ( strncmp(classname,"list<",strlen("list<"))==0) return true;
   if ( strncmp(classname,"deque<",strlen("deque<"))==0) return true;
   if ( strncmp(classname,"map<",strlen("map<"))==0) return true;
   if ( strncmp(classname,"multimap<",strlen("multimap<"))==0) return true;
   if ( strncmp(classname,"set<",strlen("set<"))==0) return true;
   if ( strncmp(classname,"multiset<",strlen("multiset<"))==0) return true;
   if ( strncmp(classname,"bitset<",strlen("bitset<"))==0) return true;

   return false;
}


//______________________________________________________________________________
bool TClassEdit::IsVectorBool(const char *name) {
   TSplitType splitname( name );

   return ( TClassEdit::STLKind( splitname.fElements[0].c_str() ) == ROOT::kSTLvector)
      && ( splitname.fElements[1] == "bool" || splitname.fElements[1]=="Bool_t");
}

//______________________________________________________________________________
namespace {
   static bool ShouldReplace(const char *name)
   {
      // This helper function indicates whether we really want to replace
      // a type.

      // In cling, looking up 'unsigned' by itself will point to a type
      // 'unsigned int' ... so because of the simplistic parsing of ResolveTypedef
      // this is cause 'unsigned int' (as input) to be replace with 'unsigned int int'
      const char *excludelist [] = {"Char_t","Short_t","Int_t","Long_t","Float_t",
                                    "Int_t","Double_t","Double32_t","Float16_t",
                                    "UChar_t","UShort_t","UInt_t","ULong_t","UInt_t",
                                    "Long64_t","ULong64_t","Bool_t","unsigned"};

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
   //    vector<MyObjTypedef> return vector<MyObj>
   //

   if ( tname==0 || tname[0]==0 ) return "";

   if ( strchr(tname,'<')==0 && (tname[strlen(tname)-1]!='*') ) {

      if (strcmp(tname,"Double32_t")==0 || strcmp(tname,"Float16_t")==0) {
         return tname;
      }

      if ( strchr(tname,':')!=0 ) {
         // We have a namespace and we have to check it first :(

         int slen = strlen(tname);
         for(int k=0;k<slen;++k) {
            if (tname[k]==':') {
               // NOTE: there is a missing increment of k, which means that
               // this next steps prevents to look at typedef define in a scope.
               if (k+1>=slen || tname[k+1]!=':') {
                  // we expected another ':'
                  return tname;
               }
               if (k) {
                  string base(tname, 0, k);
                  if (base=="std") {
                     // std is not declared but is also ignored by CINT!
                     // NOTE: this is probably no longer necessary.
                     tname += 5;
                     break;
                  } else {
                     if (base.compare(0,6,"const ") == 0) {
                        base.erase(0,6);
                     }
                     if (gInterpreterHelper &&
                         !gInterpreterHelper->IsDeclaredScope(base)) {
                        // the nesting namespace is not declared
                        return tname;
                     }
                  }
                  // Consume the 2nd semi colon
                  ++k;
               }
            }
         }
      }

      // We have a very simple type

      if (resolveAll || ShouldReplace(tname)) {
         string result, tsnam = tname;
         if (gInterpreterHelper &&
             gInterpreterHelper->GetPartiallyDesugaredNameWithScopeHandling(tsnam, result))
            return result;
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
   if (len > 5 && strncmp(tname,"std::",5) == 0) {
      prev = 5;
   }
   for (int i=prev; i<len; ++i) {
      switch (tname[i]) {
         case ':': {
           if ( strncmp(tname+prev,"std::",5) == 0 ) {
              prev += 5;
              ++i;
           }
           break;
         }
         case '<': {
            char keep = input[i];
            string temp( input, prev,i-prev );
            answ << temp;
            answ << keep;
            prev = i+1;
            break; // We do not have a complete type yet.
         }
         case '>':
         case '*':
         case ' ':
         case '&':
         case ',':
         {
            char keep = input[i];
            string temp( input, prev,i-prev );

            if ( (resolveAll&&(temp!="Double32_t")&&(temp!="Float16_t")) || ShouldReplace(temp.c_str())) {
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
   if ((resolveAll&&(strcmp(last,"Double32_t")!=0)&&(strcmp(last,"Float16_t")!=0)) || ShouldReplace(last)) {
      answ << ResolveTypedef( last, resolveAll);
   } else {
      answ << last;
   }
#ifndef R__SSTREAM
   // Deprecated case
   answ << ends;
   std::string ret = answ.str();
   answ.freeze(false);
   return ret;
#else
   return answ.str();
#endif

}


//______________________________________________________________________________
string TClassEdit::InsertStd(const char *tname)
{

   // Return the name of type 'tname' with all STL classes prepended by "std::".
   // For example for "vector<set<auto_ptr<int*> > >" it returns
   //    "std::vector<std::set<std::auto_ptr<int*> > >"
   //

   static const char* sSTLtypes[] = {
      "allocator",
      "auto_ptr",
      "bad_alloc",
      "bad_cast",
      "bad_exception",
      "bad_typeid",
      "basic_filebuf",
      "basic_fstream",
      "basic_ifstream",
      "basic_ios",
      "basic_iostream",
      "basic_istream",
      "basic_istringstream",
      "basic_ofstream",
      "basic_ostream",
      "basic_ostringstream",
      "basic_streambuf",
      "basic_string",
      "basic_stringbuf",
      "basic_stringstream",
      "binary_function",
      "binary_negate",
      "bitset",
      "char_traits",
      "codecvt_byname",
      "codecvt",
      "collate",
      "collate_byname",
      "compare",
      "complex",
      "ctype_byname",
      "ctype",
      "deque",
      "divides",
      "domain_error",
      "equal_to",
      "exception",
      "fpos",
      "greater_equal",
      "greater",
      "gslice_array",
      "gslice",
      "indirect_array",
      "invalid_argument",
      "ios_base",
      "istream_iterator",
      "istreambuf_iterator",
      "istrstream",
      "iterator_traits",
      "iterator",
      "length_error",
      "less_equal",
      "less",
      "list",
      "locale",
      "localedef utility",
      "locale utility",
      "logic_error",
      "logical_and",
      "logical_not",
      "logical_or",
      "map",
      "mask_array",
      "mem_fun",
      "mem_fun_ref",
      "messages",
      "messages_byname",
      "minus",
      "modulus",
      "money_get",
      "money_put",
      "moneypunct",
      "moneypunct_byname",
      "multimap",
      "multiplies",
      "multiset",
      "negate",
      "not_equal_to",
      "num_get",
      "num_put",
      "numeric_limits",
      "numpunct",
      "numpunct_byname",
      "ostream_iterator",
      "ostreambuf_iterator",
      "ostrstream",
      "out_of_range",
      "overflow_error",
      "pair",
      "plus",
      "pointer_to_binary_function",
      "pointer_to_unary_function",
      "priority_queue",
      "queue",
      "range_error",
      "raw_storage_iterator",
      "reverse_iterator",
      "runtime_error",
      "set",
      "slice_array",
      "slice",
      "stack",
      "string",
      "strstream",
      "strstreambuf",
      "time_get_byname",
      "time_get",
      "time_put_byname",
      "time_put",
      "unary_function",
      "unary_negate",
      "underflow_error",
      "valarray",
      "vector",
      "wstring"
   };
   static set<string> sSetSTLtypes;

   if (tname==0 || tname[0]==0) return "";

   if (sSetSTLtypes.empty()) {
      // set up static set
      const size_t nSTLtypes = sizeof(sSTLtypes) / sizeof(const char*);
      for (size_t i = 0; i < nSTLtypes; ++i)
         sSetSTLtypes.insert(sSTLtypes[i]);
   }

   size_t b = 0;
   size_t len = strlen(tname);
   string ret;
   ret.reserve(len + 20); // expect up to 4 extra "std::" to insert
   string id;
   while (b < len) {
      // find beginning of next identifier
      bool precScope = false; // whether the identifier was preceded by "::"
      while (!(isalnum(tname[b]) || tname[b] == '_') && b < len) {
         precScope = (b < len - 2) && (tname[b] == ':') && (tname[b + 1] == ':');
         if (precScope) {
            ret += "::";
            b += 2;
         } else
            ret += tname[b++];
      }

      // now b is at the beginning of an identifier or len
      size_t e = b;
      // find end of identifier
      id.clear();
      while (e < len && (isalnum(tname[e]) || tname[e] == '_'))
         id += tname[e++];
      if (!id.empty()) {
         if (!precScope) {
            set<string>::const_iterator iSTLtype = sSetSTLtypes.find(id);
            if (iSTLtype != sSetSTLtypes.end())
               ret += "std::";
         }

         ret += id;
         b = e;
      }
   }
   return ret;
}


