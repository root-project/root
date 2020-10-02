#ifndef instHeader_h
#define instHeader_h

#include <string>
#include <utility>
using namespace std;

template <typename T> class Inner;
template <> class Inner<int>;


#include "TInterpreter.h"

int instHeaderTestValid(bool expect)
{
   auto c = gInterpreter->ClassInfo_Factory("std::pair<std::string,Inner<int>>");
   if (gInterpreter->ClassInfo_IsValid(c) != expect) {
      if (expect) printf("ERROR: Class info for instantiation is not valid!\n");
      else printf("ERROR: Class info for failed instantiation is valid!\n");
      return 1;
   }
   gInterpreter->ClassInfo_Delete(c);
   return 0;
}

#ifdef __CLING__
//.L decltest.cxx+
// Private testing functions declared in TCling.cxx
class ClassInfo_t;
bool TCling__TEST_isInvalidDecl(ClassInfo_t *input);

int instHeaderTestDecl(bool expect, const char *tname = "std::pair<std::string,Inner<int>>")
{
   auto c = gInterpreter->ClassInfo_Factory("std::pair<std::string,Inner<int>>");
   if (!gInterpreter->ClassInfo_IsValid(c)) {
      printf("Info: ClassInfo is not valid\n");
      return 1;
   }
   if (TCling__TEST_isInvalidDecl(c))
      printf("Info: Decl for instantation is not valid\n");
   else
      printf("Info: Decl for instantation is valid\n");

   if (TCling__TEST_isInvalidDecl(c) != expect) {
      if (expect) printf("ERROR: Decl for instantiation is valid!\n");
      else printf("ERROR: Decl for failed instantiation is not valid!\n");
      return 1;
   }
   gInterpreter->ClassInfo_Delete(c);
   return 0;
}
#endif

int instHeaderTestMembers()
{
   auto c = gInterpreter->ClassInfo_Factory("std::pair<std::string,Inner<int>>");
   if (!gInterpreter->ClassInfo_IsValid(c)) {
      printf("ClassInfo is invalid.\n");
      return 0;
   }
   auto d = gInterpreter->DataMemberInfo_Factory(c, TDictionary::EMemberSelection::kNoUsingDecls);
   while (gInterpreter->DataMemberInfo_Next(d)) {
      if (gInterpreter->DataMemberInfo_IsValid(d)) {
         printf("DataMember: %s\n",gInterpreter->DataMemberInfo_Name(d));
      }
   }
   gInterpreter->DataMemberInfo_Delete(d);
   gInterpreter->ClassInfo_Delete(c);
   return 0;
}

#endif
