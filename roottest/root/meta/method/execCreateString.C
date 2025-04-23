#include "TClass.h"
#include "TInterpreter.h"

#include <iostream>
#include <string>


void execCreateString() {
   std::cerr << "running create_string" << std::endl;

   ClassInfo_t* gcl = TClass::GetClass("string")->GetClassInfo();
   CallFunc_t* mc = gInterpreter->CallFunc_Factory();
   Longptr_t offset = 0;

// Bug 1: can not get a valid function for "string", but can get one for "basic_string<>"
   std::cerr << "Bug 1:\n";
   gInterpreter->CallFunc_SetFuncProto(mc, gcl, "string", "", &offset);
   std::cerr << "string default ctor is valid:       " << gInterpreter->CallFunc_IsValid(mc) << std::endl;
   gInterpreter->CallFunc_SetFuncProto(mc, gcl, "basic_string<char,char_traits<char>,allocator<char> >", "", &offset);
   std::cerr << "basic_string default ctor is valid: " << gInterpreter->CallFunc_IsValid(mc) << std::endl;
// -- Bug 1

// Bug 2: although "basic_string<>" works, the namespace must have been dropped
   std::cerr << "\nBug 2:\n";
   gInterpreter->CallFunc_SetFuncProto(mc, gcl, "std::basic_string<char,char_traits<char>,allocator<char> >", "", &offset);
   std::cerr << "std::basic_string default ctor is valid: " << gInterpreter->CallFunc_IsValid(mc) << std::endl;
// -- Bug 2

// Bug 3: also, exact matching fails, if not default constructor
   std::cerr << "\nBug 3:\n";
   gInterpreter->CallFunc_SetFuncProto(mc, gcl, "basic_string<char,char_traits<char>,allocator<char> >", "", &offset, ROOT::kExactMatch);
   std::cerr << "basic_string default ctor (exact match) is valid: " << gInterpreter->CallFunc_IsValid(mc) << std::endl;
   gInterpreter->CallFunc_SetFuncProto(mc, gcl, "basic_string<char,char_traits<char>,allocator<char> >",
                                          "const basic_string<char,char_traits<char>,allocator<char> >&", &offset, ROOT::kExactMatch);
   std::cerr << "basic_string copy ctor 1 (exact match) is valid:  " << gInterpreter->CallFunc_IsValid(mc) << std::endl;
   gInterpreter->CallFunc_SetFuncProto(mc, gcl, "basic_string<char,char_traits<char>,allocator<char> >", 
                                          "const string&", &offset, ROOT::kExactMatch);
   std::cerr << "basic_string copy ctor 2 (exact match) is valid:  " << gInterpreter->CallFunc_IsValid(mc) << std::endl;
   // the above all work for conversion match
// -- Bug 3
}
