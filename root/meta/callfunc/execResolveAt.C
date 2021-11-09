#include "TClass.h"
#include "TInterpreter.h"

#include <iostream>
#include <string>
#include <vector>

std::vector<float> gvf;

void execResolveAt() {

   std::cerr << "running resolve_at" << std::endl;

   ClassInfo_t* gcl = TClass::GetClass("std::vector<float>")->GetClassInfo();
   //ClassInfo_t* gcl = gInterpreter->ClassInfo_Factory("vector<float>");
   CallFunc_t* mc = gInterpreter->CallFunc_Factory();
   Longptr_t offset = 0;
   gInterpreter->CallFunc_SetFuncProto(mc, gcl, "at", "vector<_Tp,_Alloc>::size_type", &offset);
   std::cerr << "is valid vector<_Tp,_Alloc>::size_type: " << gInterpreter->CallFunc_IsValid(mc) << std::endl;

   gInterpreter->CallFunc_SetFuncProto(mc, gcl, "at", "std::vector<float>::size_type", &offset);
   std::cerr << "is valid std::vector<float>::size_type: " << gInterpreter->CallFunc_IsValid(mc) << std::endl;

   gInterpreter->CallFunc_SetFuncProto(mc, gcl, "at", "vector<float>::size_type", &offset);
   std::cerr << "is valid vector<float>::size_type:      " << gInterpreter->CallFunc_IsValid(mc) << std::endl;

   gInterpreter->CallFunc_SetFuncProto(mc, gcl, "at", "unsigned int", &offset);
   std::cerr << "is valid unsigned int:                  " << gInterpreter->CallFunc_IsValid(mc) << std::endl;
}

