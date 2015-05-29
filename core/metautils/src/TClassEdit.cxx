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
#include "RStringView.h"

namespace {
   static TClassEdit::TInterpreterLookupHelper *gInterpreterHelper = 0;
}

namespace std {} using namespace std;

//______________________________________________________________________________
static size_t StdLen(const std::string_view name)
{
   // Return the length, if any, taken by std:: and any
   // potential inline namespace (well compiler detail namespace).

   size_t len = 0;
   if (name.compare(0,5,"std::")==0) {
      len = 5;

      // TODO: This is likely to induce unwanted autoparsing, those are reduced
      // by the caching of the result.
      if (gInterpreterHelper) {
         for(size_t i = 5; i < name.length(); ++i) {
            if (name[i] == '<') break;
            if (name[i] == ':') {
               bool isInlined;
               std::string scope(name.data(),i);
               std::string scoperesult;
               // We assume that we are called in already serialized code.
               // Note: should we also cache the negative answers?
               static std::set<std::string> gInlined;

               if (gInlined.find(scope) != gInlined.end()) {
                  len = i;
                  if (i+1<name.length() && name[i+1]==':') {
                     len += 2;
                  }
               }
               if (!gInterpreterHelper->ExistingTypeCheck(scope, scoperesult)
                   && gInterpreterHelper->IsDeclaredScope(scope,isInlined)) {
                  if (isInlined) {
                     gInlined.insert(scope);
                     len = i;
                     if (i+1<name.length() && name[i+1]==':') {
                        len += 2;
                     }
                  }
               }
            }
         }
      }
   }

   return len;
}

//______________________________________________________________________________
static void RemoveStd(std::string &name, size_t pos = 0)
{
   // Remove std:: and any potential inline namespace (well compiler detail
   // namespace.

   size_t len = StdLen({name.data()+pos,name.length()-pos});
   if (len) {
      name.erase(pos,len);
   }
}

//______________________________________________________________________________
static void RemoveStd(std::string_view &name)
{
   // Remove std:: and any potential inline namespace (well compiler detail
   // namespace.

   size_t len = StdLen(name);
   if (len) {
      name.remove_prefix(len);
   }
}

//______________________________________________________________________________
TClassEdit::EComplexType TClassEdit::GetComplexType(const char* clName)
{
   if (0 == strncmp(clName, "complex<", 8)) {
      const char *clNamePlus8 = clName + 8;
      if (0 == strcmp("float>", clNamePlus8)) {
         return EComplexType::kFloat;
         }
      if (0 == strcmp("double>", clNamePlus8)) {
         return EComplexType::kDouble;
      }
      if (0 == strcmp("int>", clNamePlus8)) {
         return EComplexType::kInt;
      }
      if (0 == strcmp("long>", clNamePlus8)) {
         return EComplexType::kLong;
      }
   }
   return EComplexType::kNone;
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
   return STLKind(fElements[0]);
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

   int kind = STLKind(fElements[0]);

   if (kind==ROOT::kSTLvector || kind==ROOT::kSTLlist || kind==ROOT::kSTLforwardlist) {

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
#include <iostream>
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
        || 0 == fElements[narg-1].compare(0,6,"const[")
        || 0 == fElements[narg-1].compare("const")
        )
       ) {
      if ((mode&1)==0) tailLoc = narg-1;
   }
   else { assert(fElements[narg-1].empty()); };
   narg--;
   mode &= (~1);

   if (fNestedLocation) narg--;

   //    fprintf(stderr,"calling ShortType %d for %s with narg %d tail %d\n",imode,typeDesc,narg,tailLoc);

   //kind of stl container
   const int kind = STLKind(fElements[0]);
   const int iall = STLArgs(kind);

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
                  case ROOT::kSTLforwardlist:
                  case ROOT::kSTLdeque:
                  case ROOT::kSTLset:
                  case ROOT::kSTLmultiset:
                  case ROOT::kSTLunorderedset:
                  case ROOT::kSTLunorderedmultiset:
                     dropAlloc = IsDefAlloc(fElements[iall+1].c_str(),fElements[1].c_str());
                     break;
                  case ROOT::kSTLmap:
                  case ROOT::kSTLmultimap:
                  case ROOT::kSTLunorderedmap:
                  case ROOT::kSTLunorderedmultimap:
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
            case ROOT::kSTLforwardlist:
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

      // Treat now Pred and Hash for unordered set/map containers. Signature is:
      // template < class Key,
      //            class Hash = hash<Key>,
      //            class Pred = equal_to<Key>,
      //            class Alloc = allocator<Key>
      //          > class unordered_{set,multiset}
      // template < class Key,
      //            class Val,
      //            class Hash = hash<Key>,
      //            class Pred = equal_to<Key>,
      //            class Alloc = allocator<Key>
      //          > class unordered_{map,multimap}


      if (kind == ROOT::kSTLunorderedset || kind == ROOT::kSTLunorderedmultiset || kind == ROOT::kSTLunorderedmap || kind == ROOT::kSTLunorderedmultimap){

         bool predRemoved = false;

         if ( allocRemoved && (mode & kDropStlDefault) && narg-1 == iall) { // remove default predicate
            if ( IsDefPred( fElements[iall].c_str(), fElements[1].c_str() ) ) {
               predRemoved=true;
               narg--;
            }
         }

         if ( predRemoved && (mode & kDropStlDefault) && narg == iall) { // remove default hash
            if ( IsDefHash( fElements[iall-1].c_str(), fElements[1].c_str() ) ) {
               narg--;
            }
         }
      }
   } // End of treatment of stl containers
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

   //   do the same for all inside
   for (int i=1;i<narg; i++) {
      if (strchr(fElements[i].c_str(),'<')==0) {
         if (mode&kDropStd) {
            unsigned int offset = (0==strncmp("const ",fElements[i].c_str(),6)) ? 6 : 0;
            RemoveStd( fElements[i], offset );
         }
         if (mode&kResolveTypedef) {
            fElements[i] = ResolveTypedef(fElements[i].c_str(),true);
         }
         continue;
      }
      fElements[i] = TClassEdit::ShortType(fElements[i].c_str(),mode | TClassEdit::kKeepOuterConst);
      if (mode&kResolveTypedef) {
         // We 'just' need to check whether the outer type is a typedef or not;
         // this also will add the default template parameter if any needs to
         // be added.
         string typeresult;
         if (gInterpreterHelper->ExistingTypeCheck(fElements[i], typeresult)
             || gInterpreterHelper->GetPartiallyDesugaredNameWithScopeHandling(fElements[i], typeresult)) {
            if (!typeresult.empty()) fElements[i] = typeresult;
         }
      }
   }

   unsigned int tailOffset = 0;
   if (tailLoc && fElements[tailLoc].compare(0,5,"const") == 0) {
      if (mode & kKeepOuterConst) answ += "const ";
      tailOffset = 5;
   }
   if (!fElements[0].empty()) {answ += fElements[0]; answ +="<";}

#if 0
   // This code is no longer use, the moral equivalent would be to get
   // the 'fixed' number of argument the user told us to ignore and drop those.
   // However, the name we get here might be (usually) normalized enough that
   // this is not necessary (at the very least nothing break in roottest without
   // the aforementioned new code or this old code).
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
#endif

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
   if (tailLoc) answ += fElements[tailLoc].c_str()+tailOffset;
}

//______________________________________________________________________________
ROOT::ESTLType TClassEdit::STLKind(std::string_view type)
{
   // Converts STL container name to number. vector -> 1, etc..
   // If len is greater than 0, only look at that many characters in the string.

   unsigned char offset = 0;
   if (type.compare(0,6,"const ")==0) { offset += 6; }
   offset += StdLen(type.substr(offset));

   //container names
   static const char *stls[] =
      { "any", "vector", "list", "deque", "map", "multimap", "set", "multiset", "bitset",
         "forward_list", "unordered_set", "unordered_multiset", "unordered_map", "unordered_multimap", 0};
   static const size_t stllen[] =
      { 3, 6, 4, 5, 3, 8, 3, 8, 6,
         12, 13, 18, 13, 18, 0};
   static const ROOT::ESTLType values[] =
      {  ROOT::kNotSTL, ROOT::kSTLvector,
         ROOT::kSTLlist, ROOT::kSTLdeque,
         ROOT::kSTLmap, ROOT::kSTLmultimap,
         ROOT::kSTLset, ROOT::kSTLmultiset,
         ROOT::kSTLbitset,
         // New C++11
         ROOT::kSTLforwardlist,
         ROOT::kSTLunorderedset, ROOT::kSTLunorderedmultiset,
         ROOT::kSTLunorderedmap, ROOT::kSTLunorderedmultimap,
         ROOT::kNotSTL
      };

   // kind of stl container
   auto len = type.length();
   if (len) {
      len -= offset;
      for(int k=1;stls[k];k++) {
         if (len == stllen[k]) {
            if (type.compare(offset,len,stls[k])==0) return values[k];
         }
      }
   } else {
      for(int k=1;stls[k];k++) {if (type.compare(offset,len,stls[k])==0) return values[k];}
   }
   return ROOT::kNotSTL;
}

//______________________________________________________________________________
int   TClassEdit::STLArgs(int kind)
{
//      Return number of arguments for STL container before allocator

   static const char  stln[] =// min number of container arguments
      //     vector, list, deque, map, multimap, set, multiset, bitset,
      {    1,     1,    1,     1,   3,        3,   2,        2,      1,
      // forward_list, unordered_set, unordered_multiset, unordered_map, unordered_multimap
                    1,             3,                  3,             4,                  4};

   return stln[kind];
}

//______________________________________________________________________________
static size_t findNameEnd(const std::string_view full)
{
   int level = 0;
   for(size_t i = 0; i < full.length(); ++i) {
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
static size_t findNameEnd(const std::string &full, size_t pos)
{
   return pos + findNameEnd( {full.data()+pos,full.length()-pos} );
}

//______________________________________________________________________________
bool TClassEdit::IsDefAlloc(const char *allocname, const char *classname)
{
   // return whether or not 'allocname' is the STL default allocator for type
   // 'classname'

   string_view a( allocname );
   RemoveStd(a);

   if (a=="alloc")                              return true;
   if (a=="__default_alloc_template<true,0>")   return true;
   if (a=="__malloc_alloc_template<0>")         return true;

   const static int alloclen = strlen("allocator<");
   if (a.compare(0,alloclen,"allocator<") != 0) {
      return false;
   }
   a.remove_prefix(alloclen);

   RemoveStd(a);

   string_view k = classname;
   RemoveStd(k);

   if (a.compare(0,k.length(),k) != 0) {
      // Now we need to compare the normalized name.
      size_t end = findNameEnd(a);

      std::string valuepart;
      GetNormalizedName(valuepart,std::string_view(a.data(),end));

      std::string norm_value;
      GetNormalizedName(norm_value,k);

      if (valuepart != norm_value) {
         return false;
      }
      a.remove_prefix(end);
   } else {
      a.remove_prefix(k.length());
   }

   if (a.compare(0,1,">")!=0 && a.compare(0,2," >")!=0) {
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

   string_view a( allocname );
   RemoveStd(a);

   const static int alloclen = strlen("allocator<");
   if (a.compare(0,alloclen,"allocator<") != 0) {
      return false;
   }
   a.remove_prefix(alloclen);

   RemoveStd(a);

   const static int pairlen = strlen("pair<");
   if (a.compare(0,pairlen,"pair<") != 0) {
      return false;
   }
   a.remove_prefix(pairlen);

   const static int constlen = strlen("const ");
   if (a.compare(0,constlen,"const ") == 0) {
      a.remove_prefix(constlen);
   }

   RemoveStd(a);

   string_view k = keyclassname;
   RemoveStd(k);

   if (a.compare(0,k.length(),k) != 0) {
      // Now we need to compare the normalized name.
      size_t end = findNameEnd(a);

      std::string keypart;
      GetNormalizedName(keypart,std::string_view(a.data(),end));

      std::string norm_key;
      GetNormalizedName(norm_key,k);

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
      a.remove_prefix(end);
   } else {
      a.remove_prefix(k.length());
   }

   if (a[0] != ',') {
      return false;
   }
   a.remove_prefix(1);
   RemoveStd(a);

   string_view v = valueclassname;
   RemoveStd(v);

   if (a.compare(0,v.length(),v) != 0) {
      // Now we need to compare the normalized name.
      size_t end = findNameEnd(a);

      std::string valuepart;
      GetNormalizedName(valuepart,std::string_view(a.data(),end));

      std::string norm_value;
      GetNormalizedName(norm_value,v);

      if (valuepart != norm_value) {
         return false;
      }
      a.remove_prefix(end);
   } else {
      a.remove_prefix(v.length());
   }

   if (a.compare(0,1,">")!=0 && a.compare(0,2," >")!=0) {
      return false;
   }

   return true;
}

//______________________________________________________________________________
static bool IsDefElement(const char *elementName, const char* defaultElementName, const char *classname)
{
   // return whether or not 'elementName' is the STL default Element for type
   // 'classname'
   string c = elementName;

   size_t pos = StdLen(c);

   const int elementlen = strlen(defaultElementName);
   if (c.compare(pos,elementlen,defaultElementName) != 0) {
      return false;
   }
   pos += elementlen;

   string k = classname;
   if (c.compare(pos,k.length(),k) != 0) {
      // Now we need to compare the normalized name.
      size_t end = findNameEnd(c,pos);

      std::string keypart;
      TClassEdit::GetNormalizedName(keypart,std::string_view(c.c_str()+pos,end-pos));

      std::string norm_key;
      TClassEdit::GetNormalizedName(norm_key,k.c_str());

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
bool TClassEdit::IsDefComp(const char *compname, const char *classname)
{
   // return whether or not 'compare' is the STL default comparator for type
   // 'classname'

   return IsDefElement(compname, "less<", classname);
}

//______________________________________________________________________________
bool TClassEdit::IsDefPred(const char *predname, const char *classname)
{
   // return whether or not 'predname' is the STL default predicate for type
   // 'classname'
   return IsDefElement(predname, "equal_to<", classname);
}

//______________________________________________________________________________
bool TClassEdit::IsDefHash(const char *hashname, const char *classname)
{
   // return whether or not 'hashname' is the STL default hash for type
   // 'classname'

   return IsDefElement(hashname, "hash<", classname);
}

//______________________________________________________________________________
void TClassEdit::GetNormalizedName(std::string &norm_name, std::string_view name)
{
   // Return the normalized name.  See TMetaUtils::GetNormalizedName.
   //
   // Return the type name normalized for ROOT,
   // keeping only the ROOT opaque typedef (Double32_t, etc.) and
   // removing the STL collections default parameter if any.
   //
   // Compare to TMetaUtils::GetNormalizedName, this routines does not
   // and can not add default template parameters.

   norm_name = std::string(name); // NOTE: Is that the shortest version?

   // Remove the std:: and default template argument and insert the Long64_t and change basic_string to string.
   TClassEdit::TSplitType splitname(norm_name.c_str(),(TClassEdit::EModType)(TClassEdit::kLong64 | TClassEdit::kDropStd | TClassEdit::kDropStlDefault | TClassEdit::kKeepOuterConst));
   splitname.ShortType(norm_name,TClassEdit::kDropStd | TClassEdit::kDropStlDefault | TClassEdit::kResolveTypedef | TClassEdit::kKeepOuterConst);

   // Depending on how the user typed their code, in particular typedef
   // declarations, we may end up with an explicit '::' being
   // part of the result string.  For consistency, we must remove it.
   if (norm_name.length()>2 && norm_name[0]==':' && norm_name[1]==':') {
      norm_name.erase(0,2);
   }

   if (gInterpreterHelper) {
      // See if the expanded name itself is a typedef.
      std::string typeresult;
      if (gInterpreterHelper->ExistingTypeCheck(norm_name, typeresult)
          || gInterpreterHelper->GetPartiallyDesugaredNameWithScopeHandling(norm_name, typeresult)) {

         if (!typeresult.empty()) norm_name = typeresult;
      }
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
const char *TClassEdit::GetUnqualifiedName(const char *original)
{
   // Return the start of the unqualified name include in 'original'.

   const char *lastPos = original;
   {
      long depth = 0;
      for(auto cursor = original; *cursor != '\0'; ++cursor) {
         if ( *cursor == '<' || *cursor == '(') ++depth;
         else if ( *cursor == '>' || *cursor == ')' ) --depth;
         else if ( *cursor == ':' ) {
            if (depth==0 && *(cursor+1) == ':' && *(cursor+2) != '\0') {
               lastPos = cursor+2;
            }
         }
      }
   }
   return lastPos;
}

//______________________________________________________________________________
static void R__FindTrailing(std::string &full,  /*modified*/
                            std::string &stars /* the literal output */
                            )
{
   const char *t = full.c_str();
   const unsigned int tlen( full.size() );

   const char *starloc = t + tlen - 1;
   bool hasconst = false;
   if ( (*starloc)=='t'
       && (starloc-t) > 4 && 0 == strncmp((starloc-4),"const",5)
       && ( (*(starloc-5)) == ' ' || (*(starloc-5)) == '*' || (*(starloc-5)) == '&'
           || (*(starloc-5)) == '>' || (*(starloc-5)) == ']') ) {
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
      while( t<=(starloc-1) && ((*(starloc-1))=='*' || (*(starloc-1))=='&' || (*(starloc-1))=='t' || isArray)) {
         if (isArray) {
            starloc--;
            isArray = ! ( (*starloc)=='[' );
         } else if ( (*(starloc-1))=='t' ) {
            if ( (starloc-1-t) > 5 && 0 == strncmp((starloc-5),"const",5)
                && ( (*(starloc-6)) == ' ' || (*(starloc-6)) == '*' || (*(starloc-6)) == '&'
                    || (*(starloc-6)) == '>' || (*(starloc-6)) == ']')) {
               // we have a const.
               starloc -= 5;
            } else {
               break;
            }
         } else {
            starloc--;
         }
      }
      stars = starloc;
      if ((*(starloc-1))==' ') {
         // erase the space too.
         starloc--;
      }

      const unsigned int starlen = strlen(starloc);
      full.erase(tlen-starlen,starlen);
   } else if (hasconst) {
      stars = starloc;
      const unsigned int starlen = strlen(starloc);
      full.erase(tlen-starlen,starlen);
   }

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
   {
      unsigned int const_offset = (0==strncmp("const ",full.c_str(),6)) ? 6 : 0;
      bool isString = false;
      bool isStdString = false;
      static const char* basic_string_std = "std::basic_string<char";
      static const unsigned int basic_string_std_len = strlen(basic_string_std);

      if (full.compare(const_offset,basic_string_std_len,basic_string_std) == 0
          && full.size() > basic_string_std_len) {
         isString = true;
         isStdString = true;
      } else if (full.compare(const_offset,basic_string_std_len-5,basic_string_std+5) == 0
                 && full.size() > (basic_string_std_len-5)) {
         // no std.
         isString = true;
      }
      if (isString) {
         size_t offset = isStdString ? basic_string_std_len : basic_string_std_len - 5;
         offset += const_offset;
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
            if (const_offset && (mode & kKeepOuterConst)) {
               if (isStdString && !(mode & kDropStd)) {
                  output.push_back("const std::string");
               } else {
                  output.push_back("const string");
               }
            } else {
               if (isStdString && !(mode & kDropStd)) {
                  output.push_back("std::string");
               } else {
                  output.push_back("string");
               }
            }
            if (offset < full.length()) {
               // Copy the trailing text.
               // keep the '>' inside right for R__FindTrailing to work
               string right( full.substr(offset) );
               string stars;
               R__FindTrailing(right, stars);
               output.back().append(right.c_str()+1); // skip the '>'
               output.push_back(stars);
            } else {
               output.push_back("");
            }
            return output.size();
         }
      }
   }

   if ( mode & kDropStd) {
      unsigned int offset = (0==strncmp("const ",full.c_str(),6)) ? 6 : 0;
      RemoveStd( full, offset );
   }

   string stars;
   if ( !full.empty() ) {
      R__FindTrailing(full, stars);
   }

   const char *c = strchr(full.c_str(),'<');
   if (c) {
      //we have 'something<'
      output.push_back(string(full,0,c - full.c_str()));

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

   if (!output.empty()) output.push_back(stars);
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
         if (!strchr("*&:._$ []-@",*c)) break;
         // '.' is used as a module/namespace separator by PyROOT, see
         // TPyClassGenerator::GetClass.
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
ROOT::ESTLType TClassEdit::UnderlyingIsSTLCont(std::string_view type)
{
   // Return the type of STL collection, if any, that is the underlying type
   // of the given type.   Namely return the value of IsSTLCont after stripping
   // pointer, reference and constness from the type.
   //    UnderlyingIsSTLCont("vector<int>*") == IsSTLCont("vector<int>")
   // See TClassEdit::IsSTLCont
   //
   //  type     : type name: vector<list<classA,allocator>,allocator>*
   //  result:    0          : not stl container
   //             code of container 1=vector,2=list,3=deque,4=map
   //                     5=multimap,6=set,7=multiset

   if (type.compare(0,6,"const ",6) == 0)
      type.remove_prefix(6);

   while(type[type.length()-1]=='*' ||
         type[type.length()-1]=='&' ||
         type[type.length()-1]==' ') {
      type.remove_suffix(1);
   }
   return IsSTLCont(type);
}

//______________________________________________________________________________
ROOT::ESTLType TClassEdit::IsSTLCont(std::string_view type)
{
   //  type     : type name: vector<list<classA,allocator>,allocator>
   //  result:    0          : not stl container
   //             code of container 1=vector,2=list,3=deque,4=map
   //                     5=multimap,6=set,7=multiset

   auto pos = type.find('<');
   if (pos==std::string_view::npos) return ROOT::kNotSTL;

   auto c = pos+1;
   for (decltype(type.length()) level = 1; c < type.length(); ++c) {
      if (type[c] == '<') ++level;
      if (type[c] == '>') --level;
      if (level == 0) break;
   }
   if (c != (type.length()-1) ) {
      return ROOT::kNotSTL;
   }

   return STLKind({type.data(),pos});
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
   // return true if the class belongs to the std namespace

   classname += StdLen( classname );
   if ( strcmp(classname,"string")==0 ) return true;
   if ( strncmp(classname,"bitset<",strlen("bitset<"))==0) return true;
   if ( strncmp(classname,"pair<",strlen("pair<"))==0) return true;
   if ( strcmp(classname,"allocator")==0) return true;
   if ( strncmp(classname,"allocator<",strlen("allocator<"))==0) return true;
   if ( strncmp(classname,"greater<",strlen("greater<"))==0) return true;
   if ( strncmp(classname,"less<",strlen("less<"))==0) return true;
   if ( strncmp(classname,"equal_to<",strlen("equal_to<"))==0) return true;
   if ( strncmp(classname,"hash<",strlen("hash<"))==0) return true;
   if ( strncmp(classname,"auto_ptr<",strlen("auto_ptr<"))==0) return true;

   if ( strncmp(classname,"vector<",strlen("vector<"))==0) return true;
   if ( strncmp(classname,"list<",strlen("list<"))==0) return true;
   if ( strncmp(classname,"forward_list<",strlen("forward_list<"))==0) return true;
   if ( strncmp(classname,"deque<",strlen("deque<"))==0) return true;
   if ( strncmp(classname,"map<",strlen("map<"))==0) return true;
   if ( strncmp(classname,"multimap<",strlen("multimap<"))==0) return true;
   if ( strncmp(classname,"set<",strlen("set<"))==0) return true;
   if ( strncmp(classname,"multiset<",strlen("multiset<"))==0) return true;
   if ( strncmp(classname,"unordered_set<",strlen("unordered_set<"))==0) return true;
   if ( strncmp(classname,"unordered_multiset<",strlen("unordered_multiset<"))==0) return true;
   if ( strncmp(classname,"unordered_map<",strlen("unordered_map<"))==0) return true;
   if ( strncmp(classname,"unordered_multimap<",strlen("unordered_multimap<"))==0) return true;
   if ( strncmp(classname,"bitset<",strlen("bitset<"))==0) return true;

   return false;
}


//______________________________________________________________________________
bool TClassEdit::IsVectorBool(const char *name) {
   TSplitType splitname( name );

   return ( TClassEdit::STLKind( splitname.fElements[0] ) == ROOT::kSTLvector)
      && ( splitname.fElements[1] == "bool" || splitname.fElements[1]=="Bool_t");
}

//______________________________________________________________________________
static void ResolveTypedefProcessType(const char *tname,
                                      unsigned int /* len */,
                                      unsigned int cursor,
                                      bool constprefix,
                                      unsigned int start_of_type,
                                      unsigned int end_of_type,
                                      unsigned int mod_start_of_type,
                                      bool &modified,
                                      std::string &result)
{
   std::string type(modified && (mod_start_of_type < result.length()) ?
                    result.substr(mod_start_of_type, string::npos)
                    : string(tname, start_of_type, end_of_type == 0 ? cursor - start_of_type : end_of_type - start_of_type));  // we need to try to avoid this copy
   string typeresult;
   if (gInterpreterHelper->ExistingTypeCheck(type, typeresult)
       || gInterpreterHelper->GetPartiallyDesugaredNameWithScopeHandling(type, typeresult)) {
      // it is a known type
      if (!typeresult.empty()) {
         // and it is a typedef, we need to replace it in the output.
         if (modified) {
            result.replace(mod_start_of_type, string::npos,
                           typeresult);
         }
         else {
            modified = true;
            mod_start_of_type = start_of_type;
            result += string(tname,0,start_of_type);
            if (constprefix && typeresult.compare(0,6,"const ",6) == 0) {
               result += typeresult.substr(6,string::npos);
            } else {
               result += typeresult;
            }
         }
      } else if (modified) {
         result.replace(mod_start_of_type, string::npos,
                        type);
      }
      if (modified) {
         if (end_of_type != 0 && end_of_type!=cursor) {
            result += std::string(tname,end_of_type,cursor-end_of_type);
         }
      }
   } else {
      // no change needed.
      if (modified) {
         // result += type;
         if (end_of_type != 0 && end_of_type!=cursor) {
            result += std::string(tname,end_of_type,cursor-end_of_type);
         }
      }
   }
}

//______________________________________________________________________________
static void ResolveTypedefImpl(const char *tname,
                               unsigned int len,
                               unsigned int &cursor,
                               bool &modified,
                               std::string &result)
{

   // Need to parse and deal with
   // A::B::C< D, E::F, G::H<I,J>::K::L >::M
   // where E might be replace by N<O,P>
   // and G::H<I,J>::K or G might be a typedef.

   bool constprefix = false;

   if (tname[cursor]==' ') {
      if (!modified) {
         modified = true;
         result += string(tname,0,cursor);
      }
      while (tname[cursor]==' ') ++cursor;
   }

   if (tname[cursor]=='c' && (cursor+6<len)) {
      if (strncmp(tname+cursor,"const ",6) == 0) {
         cursor += 6;
         if (modified) result += "const ";
      }
      constprefix = true;

   }

   // When either of those two is true, we should probably go to modified
   // mode. (Otherwise we rely on somebody else to strip the std::)
   if (len > 5 && strncmp(tname+cursor,"std::",5) == 0) {
      cursor += 5;
   }
   if (len > 2 && strncmp(tname+cursor,"::",2) == 0) {
      cursor += 2;
      len -= 2;
   }

   unsigned int start_of_type = cursor;
   unsigned int end_of_type = 0;
   unsigned int mod_start_of_type = result.length();
   unsigned int prevScope = cursor;
   for ( ; cursor<len; ++cursor) {
      switch (tname[cursor]) {
         case ':': {
            if ((cursor+1)>=len || tname[cursor+1]!=':') {
               // we expected another ':', malformed, give up.
               if (modified) result += (tname+prevScope);
               return;
            }
            string scope;
            if (modified) {
               scope = result.substr(mod_start_of_type, string::npos);
               scope += std::string(tname+prevScope,cursor-prevScope);
            } else {
               scope = std::string(tname, start_of_type, cursor - start_of_type); // we need to try to avoid this copy
            }
            std::string scoperesult;
            bool isInlined = false;
            if (gInterpreterHelper->ExistingTypeCheck(scope, scoperesult)
                ||gInterpreterHelper->GetPartiallyDesugaredNameWithScopeHandling(scope, scoperesult)) {
               // it is a known type
               if (!scoperesult.empty()) {
                  // and it is a typedef
                  if (modified) {
                     if (constprefix && scoperesult.compare(0,6,"const ",6) != 0) mod_start_of_type -= 6;
                     result.replace(mod_start_of_type, string::npos,
                                    scoperesult);
                     result += "::";
                  } else {
                     modified = true;
                     mod_start_of_type = start_of_type;
                     result += string(tname,0,start_of_type);
                     //if (constprefix) result += "const ";
                     result += scoperesult;
                     result += "::";
                  }
               } else if (modified) {
                  result += std::string(tname+prevScope,cursor+1-prevScope);
               }
            } else if (!gInterpreterHelper->IsDeclaredScope(scope,isInlined)) {
               // the nesting namespace is not declared
               if (modified) result += (tname+prevScope);
               // Unfortunately, this is too harsh .. what about:
               //    unknown::wrapper<Int_t>
               return;
            } else if (isInlined) {
               // humm ... just skip it.
               if (!modified) {
                  modified = true;
                  mod_start_of_type = start_of_type;
                  result += string(tname,0,start_of_type);
                  //if (constprefix) result += "const ";
                  result += string(tname,start_of_type,prevScope - start_of_type);
               }
            } else if (modified) {
               result += std::string(tname+prevScope,cursor+1-prevScope);
            }
            // Consume the 1st semi colon, the 2nd will be consume by the for loop.
            ++cursor;
            prevScope = cursor+1;
            break;
         }
         case '<': {
            // push information on stack
            if (modified) {
               result += std::string(tname+prevScope,cursor+1-prevScope);
               // above includes the '<' .... result += '<';
            }
            do {
               ++cursor;
               ResolveTypedefImpl(tname,len,cursor,modified,result);
            } while( cursor<len && tname[cursor] == ',' );

            while (cursor<len && tname[cursor+1]==' ') ++cursor;

            // Since we already checked the type, skip the next section
            // (respective the scope section and final type processing section)
            // as they would re-do the same job.
            if (cursor+2<len && tname[cursor+1]==':' && tname[cursor+2]==':') {
               if (modified) result += "::";
               cursor += 2;
               prevScope = cursor+1;
            }
            if ( (cursor+1)<len && tname[cursor+1] == ',') {
               ++cursor;
               if (modified) result += ',';
               return;
            }
            if ( (cursor+1)<len && tname[cursor+1] == '>') {
               ++cursor;
               if (modified) result += " >";
               return;
            }
            if ( (cursor+1) >= len) {
               return;
            }
            if (tname[cursor] != ' ') break;
            if (modified) prevScope = cursor+1;
            // If the 'current' character is a space we need to treat it,
            // since this the next case statement, we can just fall through,
            // otherwise we should need to do:
            // --cursor; break;
         }
         case ' ': {
            end_of_type = cursor;
            // let's see if we have 'long long' or 'unsigned int' or 'signed char' or what not.
            while ((cursor+1)<len && tname[cursor+1] == ' ') ++cursor;

            auto next = cursor+1;
            if (strncmp(tname+next,"const",5) == 0 && ((next+5)==len || tname[next+5] == ' ' || tname[next+5] == '*' || tname[next+5] == '&' || tname[next+5] == ',' || tname[next+5] == '>' || tname[next+5] == ']'))
            {
               // A first const after the type needs to be move in the front.
               if (!modified) {
                  modified = true;
                  result += string(tname,0,start_of_type);
                  result += "const ";
                  mod_start_of_type = start_of_type + 6;
                  result += string(tname,start_of_type,end_of_type-start_of_type);
               } else if (mod_start_of_type < result.length()) {
                  result.insert(mod_start_of_type,"const ");
                  mod_start_of_type += 6;
               } else {
                  result += "const ";
                  mod_start_of_type += 6;
                  result += string(tname,start_of_type,end_of_type-start_of_type);
               }
               cursor += 5;
               end_of_type = cursor+1;
               prevScope = end_of_type;
               if (tname[next+5] == ',' || tname[next+5] == '>' || tname[next+5] == '[') {
                  break;
               }
            } else if (next!=len && tname[next] != '*' && tname[next] != '&') {
               // the type is not ended yet.
               end_of_type = 0;
               break;
            }
            ++cursor;
            // Intentional fall through;
         }
         case '*':
         case '&': {
            if (tname[cursor] != ' ') end_of_type = cursor;
            // check and skip const (followed by *,&, ,) ... what about followed by ':','['?
            auto next = cursor+1;
            if (strncmp(tname+next,"const",5) == 0) {
               if ((next+5)==len || tname[next+5] == ' ' || tname[next+5] == '*' || tname[next+5] == '&' || tname[next+5] == ',' || tname[next+5] == '>' || tname[next+5] == '[') {
                  next += 5;
               }
            }
            while (next<len &&
                   (tname[next] == ' ' || tname[next] == '*' || tname[next] == '&')) {
               ++next;
               // check and skip const (followed by *,&, ,) ... what about followed by ':','['?
               if (strncmp(tname+next,"const",5) == 0) {
                  if ((next+5)==len || tname[next+5] == ' ' || tname[next+5] == '*' || tname[next+5] == '&' || tname[next+5] == ',' || tname[next+5] == '>' || tname[next+5] == '[') {
                     next += 5;
                  }
               }
            }
            cursor = next-1;
//            if (modified && mod_start_of_type < result.length()) {
//               result += string(tname,end_of_type,cursor-end_of_type);
//            }
            break;
         }
         case ',': {
            if (modified && prevScope) {
               result += std::string(tname+prevScope,(end_of_type == 0 ? cursor : end_of_type)-prevScope);
            }
            ResolveTypedefProcessType(tname,len,cursor,constprefix,start_of_type,end_of_type,mod_start_of_type,
                                      modified, result);
            if (modified) result += ',';
            return;
         }
         case '>': {
            if (modified && prevScope) {
               result += std::string(tname+prevScope,(end_of_type == 0 ? cursor : end_of_type)-prevScope);
            }
            ResolveTypedefProcessType(tname,len,cursor,constprefix,start_of_type,end_of_type,mod_start_of_type,
                                      modified, result);
            if (modified) result += '>';
            return;
         }
         default:
            end_of_type = 0;
      }
   }

   if (prevScope && modified) result += std::string(tname+prevScope,(end_of_type == 0 ? cursor : end_of_type)-prevScope);

   ResolveTypedefProcessType(tname,len,cursor,constprefix,start_of_type,end_of_type,mod_start_of_type,
                             modified, result);
}


//______________________________________________________________________________
string TClassEdit::ResolveTypedef(const char *tname, bool /* resolveAll */)
{

   // Return the name of type 'tname' with all its typedef components replaced
   // by the actual type its points to
   // For example for "typedef MyObj MyObjTypedef;"
   //    vector<MyObjTypedef> return vector<MyObj>
   //

   if ( tname==0 || tname[0]==0 || !gInterpreterHelper) return "";

   std::string result;

   // Check if we already know it is a normalized typename or a registered
   // typedef (i.e. known to gROOT).
   if (gInterpreterHelper->ExistingTypeCheck(tname, result))
   {
      if (result.empty()) return tname;
      else return result;
   }

   unsigned int len = strlen(tname);

   unsigned int cursor = 0;
   bool modified = false;
   ResolveTypedefImpl(tname,len,cursor,modified,result);

   if (!modified) return tname;
   else return result;
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
      "forward_list",
      "fpos",
      "greater_equal",
      "greater",
      "gslice_array",
      "gslice",
      "hash",
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
      "unordered_map",
      "unordered_multimap",
      "unordered_multiset",
      "unordered_set",
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

//______________________________________________________________________________
char* TClassEdit::DemangleTypeIdName(const std::type_info& ti, int& errorCode)
{
   // Demangle in a portable way the type id name.
   // IMPORTANT: The caller is responsible for freeing the returned const char*

   const char* mangled_name = ti.name();
   return DemangleName(mangled_name, errorCode);
}
