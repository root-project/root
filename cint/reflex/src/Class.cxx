// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef REFLEX_BUILD
# define REFLEX_BUILD
#endif

#include "Class.h"

#include "Reflex/Object.h"
#include "Reflex/Type.h"

#include "DataMember.h"
#include "FunctionMember.h"
#include "Reflex/Tools.h"
#include "Reflex/DictionaryGenerator.h"

#include <typeinfo>
#include <iostream>
#include <sstream>
#include <algorithm>
#if defined(__linux) || defined(__APPLE__)
# include <cxxabi.h>
#endif
#if defined(__APPLE__)
# include <AvailabilityMacros.h>
#endif

#if defined(__APPLE__) && defined(MAC_OS_X_VERSION_10_7)
namespace __cxxabiv1 {
extern "C" void*
__dynamic_cast(const void* __src_ptr, // Starting object.
               const __class_type_info* __src_type, // Static type of object.
               const __class_type_info* __dst_type, // Desired target type.
               ptrdiff_t __src2dst); // How src and dst are related.
}
#endif


//-------------------------------------------------------------------------------
Reflex::Class::Class(const char* typ,
                     size_t size,
                     const std::type_info& ti,
                     unsigned int modifiers,
                     TYPE classType)
//-------------------------------------------------------------------------------
// Construct a Class instance.
   : ScopedType(typ, size, classType, ti, Type(), modifiers,
                ((typ[0] == 'F') && !strcmp(typ, "FILE")) ? (REPRESTYPE) 'e': REPRES_STRUCT),
   fAllBases(0),
   fCompleteType(false),
   fInherited(0) {
}


//-------------------------------------------------------------------------------
Reflex::Class::~Class() {
//-------------------------------------------------------------------------------
   for (PathsToBase::iterator it = fPathsToBase.begin(); it != fPathsToBase.end(); ++it) {
      delete it->second;
   }
   delete fInherited;
}


//-------------------------------------------------------------------------------
void
Reflex::Class::AddBase(const Type& bas,
                       OffsetFunction offsFP,
                       unsigned int modifiers) const {
//-------------------------------------------------------------------------------
// Add a base class information.
   Base b(bas, offsFP, modifiers);
   fBases.push_back(b);
}


//-------------------------------------------------------------------------------
Reflex::Object
Reflex::Class::CastObject(const Type& to,
                          const Object& obj) const {
//-------------------------------------------------------------------------------
// Cast an object. Will do up and down cast. Cross cast missing.
   std::vector<Base> path = std::vector<Base>();

   if (HasBase(to, path)) {    // up cast
      // in case of up cast the Offset has to be calculated by Reflex
      size_t obj2 = (size_t) obj.Address();

      for (std::vector<Base>::reverse_iterator bIter = path.rbegin();
           bIter != path.rend(); ++bIter) {
         obj2 += bIter->Offset((void*) obj2);
      }
      return Object(to, (void*) obj2);
   }
   path.clear();
   Type t = *this;

   if (to.HasBase(t)) {      // down cast
      // use the internal dynamic casting of the compiler (e.g. libstdc++.so)
      void* obj3 = 0;
#if defined(__linux) || defined(__APPLE__)

      obj3 = abi::__dynamic_cast(obj.Address(),
                                 (const abi::__class_type_info*) &this->TypeInfo(),
                                 (const abi::__class_type_info*) &to.TypeInfo(),
                                 -1);
#elif defined(_WIN32)
      obj3 = __RTDynamicCast(obj.Address(),
                             0,
                             (void*) &this->TypeInfo(),
                             (void*) &to.TypeInfo(),
                             0);
#endif
      return Object(to, obj3);
   }
   // fixme cross cast missing ?? internal cast possible ??

   // if the same At was passed return the object
   if ((Type) (*this) == to) {
      return obj;
   }

   // if everything fails return the dummy object
   return Object();
} // CastObject


/*/-------------------------------------------------------------------------------
   Reflex::Object Reflex::Class::Construct( const Type & signature,
                                                       const std::vector < Object > & args,
                                                       void * mem ) const {
   //-------------------------------------------------------------------------------
   static Type defSignature = Type::ByName("void (void)");
   Type signature2 = signature;

   Member constructor = Member();
   if ( !signature &&  fConstructors.size() > 1 )
   signature2 = defSignature;

   for (size_t i = 0; i < fConstructors.size(); ++ i) {
   if ( !signature2 || fConstructors[i].TypeOf().Id() == signature2.Id()) {
   constructor = fConstructors[i];
   break;
   }
   }

   if ( constructor.TypeOf() ) {
   // no memory Address passed -> Allocate memory for class
   if ( mem == 0 ) mem = Allocate();
   Object obj = Object( TypeOf(), mem );
   constructor.Invoke( obj, args );
   return obj;
   }
   else {
   throw RuntimeError("No suitable constructor found");
   }
   }
 */


//-------------------------------------------------------------------------------
Reflex::Object
Reflex::Class::Construct(const Type& sig,
                         const std::vector<void*>& args,
                         void* mem) const {
//-------------------------------------------------------------------------------
// Construct an object of this class type. The signature of the constructor function
// can be given as the first argument. Furhter arguments are a vector of memory
// addresses for non default constructors and a memory address for in place construction.
   static Type defSignature = Type::ByName("void (void)");

   // trigger setup of function members for constructor
   ExecuteFunctionMemberDelayLoad();
   Type signature = (!sig && fConstructors.size() > 1) ? defSignature : sig;

   for (size_t i = 0; i < fConstructors.size(); ++i) {
      if (!signature || fConstructors[i].TypeOf().Id() == signature.Id()) {
         Member constructor = fConstructors[i];

         if (mem == 0) {
            mem = Allocate();
         }
         Object obj = Object(ThisType(), mem);
         constructor.Invoke(obj, 0, args);
         return obj;
      }
   }
   std::stringstream s;
   s << "No suitable constructor found with signature '" << signature.Name() << "'";
   throw RuntimeError(s.str());
} // Construct


//-------------------------------------------------------------------------------
void
Reflex::Class::Destruct(void* instance,
                        bool dealloc) const {
//-------------------------------------------------------------------------------
// Call the destructor for this class type on a memory address (instance). Deallocate
// memory if dealloc = true (i.e. default).

   // trigger setup of function members for constructor
   ExecuteFunctionMemberDelayLoad();
   if (!fDestructor.TypeOf()) {
      // destructor for this class not yet revealed
      for (size_t i = 0; i < ScopeBase::FunctionMemberSize(); ++i) {
         Member fm = ScopeBase::FunctionMemberAt(i);

         // constructor found Set the cache pointer
         if (fm.IsDestructor()) {
            fDestructor = fm;
            break;
         }
      }
   }

   if (fDestructor.TypeOf()) {
      // we found a destructor -> Invoke it
      Object dummy = Object(Type(), instance);
      fDestructor.Invoke(dummy, (Object*) 0);
   }

   // if deallocation of memory wanted
   if (dealloc) {
      Deallocate(instance);
   }
} // Destruct


//-------------------------------------------------------------------------------
struct DynType_t {
//-------------------------------------------------------------------------------
   virtual ~DynType_t() {
      // dummy type with vtable.
   }


};


//-------------------------------------------------------------------------------
Reflex::Type
Reflex::Class::DynamicType(const Object& obj) const {
//-------------------------------------------------------------------------------
// Discover the dynamic type of a class object and return it.
// If no virtual_function_table return itself
   if (IsVirtual()) {
      // Avoid the case that the first word is a virtual_base_offset_table instead of
      // a virtual_function_table
      long int offset = **(long**) obj.Address();

      if (offset == 0) {
         return ThisType();
      } else {
         const Type& dytype = Type::ByTypeInfo(typeid(*(DynType_t*) obj.Address()));

         if (dytype && dytype.IsClass()) {
            return dytype;
         } else { return ThisType(); }
      }
   } else {
      return ThisType();
   }
} // DynamicType


//-------------------------------------------------------------------------------
bool
Reflex::Class::HasBase(const Type& cl) const {
//-------------------------------------------------------------------------------
// Return true if this class has a base class of type cl.
   std::vector<Base> v = std::vector<Base>();
   return HasBase(cl, v);
}


//-------------------------------------------------------------------------------
bool
Reflex::Class::HasBase(const Type& cl,
                       std::vector<Base>& path) const {
//-------------------------------------------------------------------------------
// Return true if this class has a base class of type cl. Return also the path
// to this type.
   if (!cl.Id()) {
      return false;
   }

   for (size_t i = 0; i < BaseSize(); ++i) {
      Base b = BaseAt(i);
      Type basetype = b.ToType();

      if (basetype.Id() == cl.Id() || basetype.FinalType().Id() == cl.Id()) {
         // remember the path to this class
         path.push_back(b);
         return true;
      } else if (basetype) {
         const Class* clbase = dynamic_cast<const Class*>(basetype.FinalType().ToTypeBase());

         if (clbase && clbase->HasBase(cl, path)) {
            // is the final base class one of the current class ?
            // if successfull remember path
            path.push_back(b);
            return true;
         }
      }
   }
   return false;
} // HasBase


//-------------------------------------------------------------------------------
bool
Reflex::Class::UpdateMembers() const {
//-------------------------------------------------------------------------------
// Initialize the vector of inherited members, accessible by
// ...Member(INHERITEDMEMBERS_ALSO)
// Return false if one of the bases is not complete.
//
// This function recurses over all bases in left-to-right order, i.e.
//   class A: public A0, public A1 {int a;};
//   class A0: public A01 { int a0;}
//   class A1 {int a1;};
//   class A01 {int a01;}
// will fill the members in the order a, a0, a01, a1.
// Members of virtual bases and of their bases will only be enumerated once
// (the left-most occurrence).
// Function members with the same name and signature that occurr multiple times
// will only be enumerated once. NOTE: the left-most occurrence is taken,
// which does NOT correspond to C++ virtual function resolution (for
// backward compatibility reasons). And anyway multiple occurrences of the
// "same" non-virtual function are fine and should not be suppressed, but again
// for backward compatibility reasons that's what UpdateMembers does.

   if (fInherited) {
      return true;
   }

   if (!IsComplete()) {
      return false;
   }

   // flatten the bases, where the current class is at front
   // and the basiest base classes are at the end:
   typedef std::list<std::pair<Scope, std::pair<bool, size_t> > > BaseList_t;
   BaseList_t bases;
   {
      size_t numDataMembers = 0;
      size_t numFunctionMembers = 0;
      // bases are ordered by inheritance level
      BaseList_t basesToProcess;
      basesToProcess.push_back(std::make_pair(operator Scope(), std::make_pair(false, (size_t) 0)));

      while (!basesToProcess.empty()) {
         Scope s = basesToProcess.front().first;
         bool virt = basesToProcess.front().second.first;
         size_t level = basesToProcess.front().second.second;
         bool duplicateVirtualBase = false;

         if (virt) {
            for (BaseList_t::iterator iV = bases.begin(), iVe = bases.end(); iV != iVe; ++iV) {
               if (iV->second.first && iV->first == s) {
                  // duplicate virtual base, skip this.
                  duplicateVirtualBase = true;
               }
            }
         }
         basesToProcess.pop_front();

         if (!duplicateVirtualBase) {
            bases.push_back(std::make_pair(s, std::make_pair(virt, level)));

            for (Reverse_Base_Iterator iB = s.Base_RBegin(), iBe = s.Base_REnd(); iB != iBe; ++iB) {
               basesToProcess.push_front(std::make_pair(iB->ToScope(),
                                                        std::make_pair(virt || iB->IsVirtual(), level + 1)));
            }
            numDataMembers += s.DataMemberSize(INHERITEDMEMBERS_NO);
            numFunctionMembers += s.FunctionMemberSize(INHERITEDMEMBERS_NO);
         }
      }
      fInherited = new InheritedMembersInfo_t(numDataMembers, numFunctionMembers);
   }

   for (BaseList_t::const_iterator iS = bases.begin(), iSe = bases.end();
        iS != iSe; ++iS) {
      const Scope sc = iS->first;
      // we collect all data members...
      fInherited->fDataMembers.insert(fInherited->fDataMembers.end(),
                                      sc.DataMember_Begin(INHERITEDMEMBERS_NO),
                                      sc.DataMember_End(INHERITEDMEMBERS_NO));
      fInherited->fMembers.insert(fInherited->fMembers.end(),
                                  sc.DataMember_Begin(INHERITEDMEMBERS_NO),
                                  sc.DataMember_End(INHERITEDMEMBERS_NO));
      // ...but only non-existing function members.
      // There is no use searching for an existing function in its own scope,
      // so only iterate over members collected before the current scope:
      size_t numMembersBefore = fInherited->fFunctionMembers.size();

      for (Member_Iterator iM = sc.FunctionMember_Begin(INHERITEDMEMBERS_NO), iMe = sc.FunctionMember_End(INHERITEDMEMBERS_NO);
           iM != iMe; ++iM) {
         std::vector<Member>::iterator iExists = std::find(fInherited->fFunctionMembers.begin(), fInherited->fFunctionMembers.begin() + numMembersBefore, *iM);

         if (iExists == fInherited->fFunctionMembers.end()
             || iExists == fInherited->fFunctionMembers.begin() + numMembersBefore) {
            fInherited->fFunctionMembers.push_back(*iM);
            fInherited->fMembers.push_back(*iM);
         } else {
            // The function already exists. Let's keep the one with the lowest inheritance level.
            if (iExists != fInherited->fFunctionMembers.end()) {
               const Scope declExists = iExists->DeclaringScope();

               for (BaseList_t::const_iterator iES = bases.begin(), iESe = bases.end(); iES != iESe; ++iES) {
                  if (declExists == iES->first) {
                     if (iES->second.second > iS->second.second) {
                        // the existing one has a larger inheritance distance, so replace it:
                        *iExists = *iM;
                        // "end" is just fine, we will find it before that anyway.
                        iExists = std::find(fInherited->fMembers.begin(), fInherited->fMembers.end(), *iM);

                        if (iExists != fInherited->fMembers.end()) {
                           *iExists = *iM;
                        }
                     }
                     break;
                  }
               }
            }
         }
      }
   }

   return true;
} // UpdateMembers


//-------------------------------------------------------------------------------
bool
Reflex::Class::IsComplete() const {
//-------------------------------------------------------------------------------
// Return true if this class is complete. I.e. all dictionary information for all
// data and function member types and base classes is available.
   if (!fCompleteType) {
      fCompleteType = IsComplete2();
   }
   return fCompleteType;
}


//-------------------------------------------------------------------------------
bool
Reflex::Class::IsComplete2() const {
//-------------------------------------------------------------------------------
// Return true if this class is complete. I.e. all dictionary information for all
// data and function member types and base classes is available (internal function).
   for (size_t i = 0; i < BaseSize(); ++i) {
      Type baseType = BaseAt(i).ToType().FinalType();

      if (!baseType) {
         return false;
      }

      if (!baseType.IsComplete()) {
         return false;
      }
   }
   return true;
} // IsComplete2


//-------------------------------------------------------------------------------
size_t
Reflex::Class::AllBases() const {
//-------------------------------------------------------------------------------
// Return the number of base classes.
   size_t aBases = 0;

   for (size_t i = 0; i < BaseSize(); ++i) {
      ++aBases;

      if (BaseAt(i)) {
         aBases += BaseAt(i).BaseClass()->AllBases();
      }
   }
   return aBases;
}


//-------------------------------------------------------------------------------
bool
Reflex::Class::NewBases() const {
//-------------------------------------------------------------------------------
// Check if information for new base classes has been added.
   if (!fCompleteType) {
      size_t numBases = AllBases();

      if (fAllBases != numBases) {
         fCompleteType = IsComplete2();
         fAllBases = numBases;
         return true;
      }
   }
   return false;
}


//-------------------------------------------------------------------------------
const std::vector<Reflex::OffsetFunction>&
Reflex::Class::PathToBase(const Scope& bas) const {
//-------------------------------------------------------------------------------
// Return a vector of offset functions from the current class to the base class.
   const BasePath_t* pathToBase = fPathsToBase[bas.Id()];

   if (!pathToBase) {
      static const BasePath_t sEmptyBasePath;
      // if bas is a base, it muts be one of our direct bases,
      // or one of them must have it as a base:
      bool isDirectBase = false;

      for (std::vector<Base>::const_iterator iBase = fBases.begin(), endBase = fBases.end();
           !isDirectBase && iBase != endBase; ++iBase) {
         isDirectBase |= iBase->ToScope() == bas;
      }

      for (std::vector<Base>::const_iterator iBase = fBases.begin(), endBase = fBases.end();
           !pathToBase && iBase != endBase; ++iBase) {
         const Scope scBase = iBase->ToScope();

         if (scBase == bas || (!isDirectBase && scBase.HasBase(bas))) {
            const Class* clBase = dynamic_cast<const Class*>(scBase.ToScopeBase());

            if (clBase) {
               BasePath_t* newPathToBase = new BasePath_t(1, iBase->OffsetFP());

               if (scBase != bas) {
                  const BasePath_t& baseOffset = clBase->PathToBase(bas);
                  newPathToBase->insert(newPathToBase->begin() + 1, baseOffset.begin(), baseOffset.end());
               }
               fPathsToBase[bas.Id()] = newPathToBase;
               pathToBase = newPathToBase;
            } else {
               pathToBase = &sEmptyBasePath;
            }
         }
      }

      if (!pathToBase) {
         pathToBase = &sEmptyBasePath;
      }
   }
   return *pathToBase;
} // PathToBase


//-------------------------------------------------------------------------------
void
Reflex::Class::AddDataMember(const Member& dm) const {
//-------------------------------------------------------------------------------
// Add data member dm to this class
   ScopeBase::AddDataMember(dm);
}


//-------------------------------------------------------------------------------
Reflex::Member
Reflex::Class::AddDataMember(const char* nam,
                             const Type& typ,
                             size_t offs,
                             unsigned int modifiers /* = 0 */,
                             char* interpreterOffset /* = 0 */) const {
//-------------------------------------------------------------------------------
// Add data member to this class
   return ScopeBase::AddDataMember(nam, typ, offs, modifiers, interpreterOffset);
}


//-------------------------------------------------------------------------------
void
Reflex::Class::RemoveDataMember(const Member& dm) const {
//-------------------------------------------------------------------------------
// Remove data member dm from this class
   ScopeBase::RemoveDataMember(dm);
}


//-------------------------------------------------------------------------------
void
Reflex::Class::AddFunctionMember(const Member& fm) const {
//-------------------------------------------------------------------------------
// Add function member fm to this class
   ScopeBase::AddFunctionMember(fm);

   if (fm.IsConstructor()) {
      fConstructors.push_back(fm);
   } else if (fm.IsDestructor()) {
      fDestructor = fm;
   }
}


//-------------------------------------------------------------------------------
Reflex::Member
Reflex::Class::AddFunctionMember(const char* nam,
                                 const Type& typ,
                                 StubFunction stubFP,
                                 void* stubCtx,
                                 const char* params,
                                 unsigned int modifiers) const {
//-------------------------------------------------------------------------------
// Add function member to this class
   Member fm(ScopeBase::AddFunctionMember(nam, typ, stubFP, stubCtx, params, modifiers));

   if (fm.IsConstructor()) {
      fConstructors.push_back(fm);
   } else if (fm.IsDestructor()) {
      fDestructor = fm;
   }
   return fm;
}


//-------------------------------------------------------------------------------
void
Reflex::Class::RemoveFunctionMember(const Member& fm) const {
//-------------------------------------------------------------------------------
// Remove function member from this class.
   ScopeBase::RemoveFunctionMember(fm);
}


//-------------------------------------------------------------------------------
void
Reflex::Class::GenerateDict(DictionaryGenerator& generator) const {
//-------------------------------------------------------------------------------
// Generate Dictionary information about itself.

   // Selection file usage
   bool selected = true;

   /*
      // selection file used
      if (generator.fSelections.size() != 0 || generator.fPattern_selections.size() != 0) {
      selected = false;

      // normal selection
      for (unsigned i = 0; i < generator.fSelections.size(); ++i) {
         if (generator.fSelections.at(i) == (*this).Name(SCOPED)) {
            selected = true;
         }
      }

      // pattern selection
      for (unsigned i = 0; i < generator.fPattern_selections.size(); ++i) {
         if ((*this).Name(SCOPED).find(generator.fPattern_selections.at(i)) != std::string::npos) {
            selected = true;
         }
      }

      }
      // EndOf Selection file usage
    */

   if (selected == true) {
      std::string typenumber = generator.GetTypeNumber(ThisType());

      if (generator.fSelections.size() != 0 || generator.fPattern_selections.size() != 0) {
         std::cout << "  * selecting class " << (*this).Name(SCOPED) << "\n";
      }

      generator.AddIntoInstances("      " + generator.Replace_colon(ThisType().Name(SCOPED)) + "_dict();\n");

      // Outputten only, if inside a namespace
      if (ThisType().DeclaringScope().IsTopScope() && (!DeclaringScope().IsNamespace())) {
         generator.AddIntoShadow("\nnamespace " + ThisType().Name() + " {");
      }

      // new
      if (ThisType().DeclaringScope().IsClass()) {
         generator.AddIntoShadow("};");
      }


      // begin of the Dictionary-part
      generator.AddIntoShadow("\nclass " + generator.Replace_colon(ThisType().Name(SCOPED)) + " {\n");
      generator.AddIntoShadow("public:\n");

      if ((ThisType().DeclaringScope().IsClass())) {
         generator.AddIntoFree(";\n}\n");
      }

      generator.AddIntoFree("\n\n// ------ Dictionary for class " + ThisType().Name() + "\n");
      generator.AddIntoFree("void " + generator.Replace_colon(ThisType().Name(SCOPED)) + "_dict() {\n");
      generator.AddIntoFree("ClassBuilder(\"" + ThisType().Name(SCOPED));

      if (IsPublic()) {
         generator.AddIntoFree("\", typeid(" + ThisType().Name(SCOPED) + "), sizeof(" + ThisType().Name(SCOPED) + "), ");
      } else if (IsProtected()) {
         generator.AddIntoFree("\", typeid(Reflex::ProtectedClass), 0,");
      } else if (IsPrivate()) {
         generator.AddIntoFree("\", typeid(Reflex::PrivateClass), 0,");
      }

      if (ThisType().IsPublic()) {
         generator.AddIntoFree("PUBLIC");
      }

      if (ThisType().IsPrivate()) {
         generator.AddIntoFree("PRIVATE");
      }

      if (ThisType().IsProtected()) {
         generator.AddIntoFree("PROTECTED");
      }

      if (ThisType().IsVirtual()) {
         generator.AddIntoFree(" | VIRTUAL");
      }
      generator.AddIntoFree(" | CLASS)\n");

      generator.AddIntoClasses("\n// -- Stub functions for class " + ThisType().Name() + "--\n");

      for (Member_Iterator mi = (*this).Member_Begin();
           mi != (*this).Member_End(); ++mi) {
         (*mi).GenerateDict(generator);      // call Members' own gendict
      }

      if (ThisType().DeclaringScope().IsTopScope() && (!DeclaringScope().IsNamespace())) {
         generator.AddIntoShadow("\nnamespace " + ThisType().Name() + " {");
      }

//       std::stringstream tempcounter;
//       tempcounter << generator.fMethodCounter;
//       generator.AddIntoClasses("\nstatic void* method_x" + tempcounter.str());
//       generator.AddIntoClasses(" ( void*, const std::vector<void*>&, void*)\n{\n");
//       generator.AddIntoClasses("  static NewDelFunctions s_funcs;\n");

//       generator.AddIntoFree(".AddFunctionMember<void*(void)>(\"__getNewDelFunctions\", method_x" + tempcounter.str());
//       generator.AddIntoFree(", 0, 0, PUBLIC | ARTIFICIAL)");

//       std::string temp = "NewDelFunctionsT< ::" + ThisType().Name(SCOPED) + " >::";
//       generator.AddIntoClasses("  s_funcs.fNew         = " + temp + "new_T;\n");
//       generator.AddIntoClasses("  s_funcs.fNewArray    = " + temp + "newArray_T;\n");
//       generator.AddIntoClasses("  s_funcs.fDelete      = " + temp + "delete_T;\n");
//       generator.AddIntoClasses("  s_funcs.fDeleteArray = " + temp + "deleteArray_T;\n");
//       generator.AddIntoClasses("  s_funcs.fDestructor  = " + temp + "destruct_T;\n");
//       generator.AddIntoClasses("  return &s_funcs;\n}\n ");

//       ++generator.fMethodCounter;

      if (ThisType().DeclaringScope().IsTopScope() && (!DeclaringScope().IsNamespace())) {
         generator.AddIntoShadow("}\n");        // End of top namespace
      }

      // Recursive call
      this->ScopeBase::GenerateDict(generator);

      if (!(ThisType().DeclaringScope().IsClass())) {
         generator.AddIntoShadow("};\n");
      }

      if (!(ThisType().DeclaringScope().IsClass())) {
         generator.AddIntoFree(";\n}\n");
      }

   } //new type
} // GenerateDict
