/* @(#)root/reflex:$Id$ */

/*************************************************************************
* Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
* All rights reserved.                                                  *
*                                                                       *
* For the licensing terms see $ROOTSYS/LICENSE.                         *
* For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/

#ifdef __CINT__

#pragma link off class Reflex::DictionaryGenerator;

#pragma link C++ nestedclasses;

#pragma link C++ namespace Reflex;
#pragma link C++ namespace Reflex::Dummy;
#pragma link C++ namespace Reflex::Tools;

#pragma link C++ class Reflex::Any;
#pragma link C++ class Reflex::Base;
#pragma link C++ class Reflex::ICallback;
#pragma link C++ class Reflex::Instance;
#pragma link C++ class Reflex::Member;
#pragma link C++ class Reflex::MemberTemplate;
#pragma link C++ class Reflex::Object;
#pragma link C++ class Reflex::Object;
#pragma link C++ class Reflex::Scope;
#pragma link C++ class Reflex::Type;
#pragma link C++ class Reflex::TypeTemplate;

#pragma link C++ class Reflex::NullType;
#pragma link C++ class Reflex::UnknownType;
#pragma link C++ class Reflex::ProtectedClass;
#pragma link C++ class Reflex::ProtectedEnum;
#pragma link C++ class Reflex::ProtectedStruct;
#pragma link C++ class Reflex::ProtectedUnion;
#pragma link C++ class Reflex::PrivateClass;
#pragma link C++ class Reflex::PrivateEnum;
#pragma link C++ class Reflex::PrivateStruct;
#pragma link C++ class Reflex::PrivateUnion;
#pragma link C++ class Reflex::UnnamedClass;
#pragma link C++ class Reflex::UnnamedEnum;
#pragma link C++ class Reflex::UnnamedNamespace;
#pragma link C++ class Reflex::UnnamedStruct;
#pragma link C++ class Reflex::UnnamedUnion;

#ifndef G__SUNPRO_CC
// cannot distinguish ::exception and std::exception
# pragma link C++ class Reflex::RuntimeError;
#endif

#pragma link C++ function Reflex::NPos();


#pragma link C++ option=nomap enum Reflex::ENTITY_DESCRIPTION;
#pragma link C++ option=nomap enum Reflex::ENTITY_HANDLING;
#pragma link C++ option=nomap enum Reflex::TYPE;
#pragma link C++ option=nomap enum Reflex::EFUNDAMENTALTYPE;

#pragma link C++ option=nomap typedef Reflex::StdString_Cont_Type_t;
#pragma link C++ option=nomap typedef Reflex::StdString_Iterator;
#pragma link C++ option=nomap typedef Reflex::Reverse_StdString_Iterator;

#pragma link C++ option=nomap typedef Reflex::Type_Cont_Type_t;
#pragma link C++ option=nomap typedef Reflex::Type_Iterator;
#pragma link C++ option=nomap typedef Reflex::Reverse_Type_Iterator;

#pragma link C++ option=nomap typedef Reflex::Base_Cont_Type_t;
#pragma link C++ option=nomap typedef Reflex::Base_Iterator;
#pragma link C++ option=nomap typedef Reflex::Reverse_Base_Iterator;

#pragma link C++ option=nomap typedef Reflex::Scope_Cont_Type_t;
#pragma link C++ option=nomap typedef Reflex::Scope_Iterator;
#pragma link C++ option=nomap typedef Reflex::Reverse_Scope_Iterator;

#pragma link C++ option=nomap typedef Reflex::Object_Cont_Type_t;
#pragma link C++ option=nomap typedef Reflex::Object_Iterator;
#pragma link C++ option=nomap typedef Reflex::Reverse_Object_Iterator;

#pragma link C++ option=nomap typedef Reflex::Member_Cont_Type_t;
#pragma link C++ option=nomap typedef Reflex::Member_Iterator;
#pragma link C++ option=nomap typedef Reflex::Reverse_Member_Iterator;

#pragma link C++ option=nomap typedef Reflex::TypeTemplate_Cont_Type_t;
#pragma link C++ option=nomap typedef Reflex::TypeTemplate_Iterator;
#pragma link C++ option=nomap typedef Reflex::Reverse_TypeTemplate_Iterator;

#pragma link C++ option=nomap typedef Reflex::MemberTemplate_Cont_Type_t;
#pragma link C++ option=nomap typedef Reflex::MemberTemplate_Iterator;
#pragma link C++ option=nomap typedef Reflex::Reverse_MemberTemplate_Iterator;


#pragma link C++ class Reflex::ClassBuilder;
#pragma link C++ class Reflex::ClassBuilderImpl;
#pragma link C++ class Reflex::EnumBuilder;
#pragma link C++ class Reflex::FunctionBuilder;
#pragma link C++ class Reflex::FunctionBuilderImpl;
#pragma link C++ class Reflex::NamespaceBuilder;
#pragma link C++ class Reflex::TypedefBuilderImpl;
#pragma link C++ class Reflex::UnionBuilderImpl;
#pragma link C++ class Reflex::VariableBuilder;
#pragma link C++ class Reflex::VariableBuilderImpl;

#pragma link C++ function Reflex::TypeBuilder( const char *,unsigned int );
#pragma link C++ function Reflex::ConstBuilder( const Type & );
#pragma link C++ function Reflex::VolatileBuilder( const Type & );
#pragma link C++ function Reflex::PointerBuilder( const Type &, const std::type_info &);
#pragma link C++ function Reflex::PointerToMemberBuilder( const Type &, const Scope &, const std::type_info &);
#pragma link C++ function Reflex::ReferenceBuilder( const Type& );
#pragma link C++ function Reflex::ArrayBuilder( const Type&, size_t, const std::type_info & );
#pragma link C++ function Reflex::EnumTypeBuilder( const char *, const char *, const std::type_info &, unsigned int );
#pragma link C++ function Reflex::TypedefTypeBuilder( const char *, const Type& );
#pragma link C++ function Reflex::FunctionTypeBuilder( const Type&, const std::vector<Type> &, const std::type_info & );

#pragma link C++ function Reflex::FunctionTypeBuilder(const Type&);
#pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&);
#pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&);
#pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);

/*
 #pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
 #pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
 #pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
 #pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
 #pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
 #pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
 #pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
 #pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
 #pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
 #pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
 #pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
 #pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
 #pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
 #pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
 #pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
 #pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
 #pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
 #pragma link C++ function Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
 */

#pragma link C++ struct Reflex::NewDelFunctions;
#pragma link C++ struct Reflex::CollFuncTable;

#pragma link C++ class Reflex::Selection::NO_SELF_AUTOSELECT;
#pragma link C++ class Reflex::Selection::TRANSIENT;
#pragma link C++ class Reflex::Selection::AUTOSELECT;
#pragma link C++ class Reflex::Selection::NODEFAULT;

#endif // __CINT__
