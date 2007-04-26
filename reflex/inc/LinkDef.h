/* @(#)root/reflex:$Name:  $:$Id: LinkDef.h,v 1.1 2007/04/25 16:52:43 axel Exp $ */

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef __CINT__

#pragma link C++ nestedclasses;

#pragma link C++ namespace ROOT::Reflex;
#pragma link C++ namespace ROOT::Reflex::Dummy;
#pragma link C++ namespace ROOT::Reflex::Tools;

#pragma link C++ class ROOT::Reflex::Any;
#pragma link C++ class ROOT::Reflex::Base;
#pragma link C++ class ROOT::Reflex::ICallback;
#pragma link C++ class ROOT::Reflex::Reflex;
#pragma link C++ class ROOT::Reflex::Member;
#pragma link C++ class ROOT::Reflex::MemberTemplate;
#pragma link C++ class ROOT::Reflex::Object;
#pragma link C++ class ROOT::Reflex::Object;
#pragma link C++ class ROOT::Reflex::Scope;
#pragma link C++ class ROOT::Reflex::Type;
#pragma link C++ class ROOT::Reflex::TypeTemplate;

#pragma link C++ class ROOT::Reflex::NullType;
#pragma link C++ class ROOT::Reflex::UnknownType;
#pragma link C++ class ROOT::Reflex::ProtectedClass;
#pragma link C++ class ROOT::Reflex::ProtectedEnum;
#pragma link C++ class ROOT::Reflex::ProtectedStruct;
#pragma link C++ class ROOT::Reflex::ProtectedUnion;
#pragma link C++ class ROOT::Reflex::PrivateClass;
#pragma link C++ class ROOT::Reflex::PrivateEnum;
#pragma link C++ class ROOT::Reflex::PrivateStruct;
#pragma link C++ class ROOT::Reflex::PrivateUnion;
#pragma link C++ class ROOT::Reflex::UnnamedClass;
#pragma link C++ class ROOT::Reflex::UnnamedEnum;
#pragma link C++ class ROOT::Reflex::UnnamedNamespace;
#pragma link C++ class ROOT::Reflex::UnnamedStruct;
#pragma link C++ class ROOT::Reflex::UnnamedUnion;

#ifndef G__SUNPRO_CC
// cannot distinguish ::exception and std::exception
#pragma link C++ class ROOT::Reflex::RuntimeError;
#endif

#pragma link C++ function ROOT::Reflex::NPos();


#pragma link C++ option=nomap enum ROOT::Reflex::ENTITY_DESCRIPTION;
#pragma link C++ option=nomap enum ROOT::Reflex::ENTITY_HANDLING;
#pragma link C++ option=nomap enum ROOT::Reflex::TYPE;
#pragma link C++ option=nomap enum ROOT::Reflex::EFUNDAMENTALTYPE;

#pragma link C++ option=nomap typedef ROOT::Reflex::StdString_Cont_Type_t;
#pragma link C++ option=nomap typedef ROOT::Reflex::StdString_Iterator;
#pragma link C++ option=nomap typedef ROOT::Reflex::Reverse_StdString_Iterator;

#pragma link C++ option=nomap typedef ROOT::Reflex::Type_Cont_Type_t;
#pragma link C++ option=nomap typedef ROOT::Reflex::Type_Iterator;
#pragma link C++ option=nomap typedef ROOT::Reflex::Reverse_Type_Iterator;

#pragma link C++ option=nomap typedef ROOT::Reflex::Base_Cont_Type_t;
#pragma link C++ option=nomap typedef ROOT::Reflex::Base_Iterator;
#pragma link C++ option=nomap typedef ROOT::Reflex::Reverse_Base_Iterator;

#pragma link C++ option=nomap typedef ROOT::Reflex::Scope_Cont_Type_t;
#pragma link C++ option=nomap typedef ROOT::Reflex::Scope_Iterator;
#pragma link C++ option=nomap typedef ROOT::Reflex::Reverse_Scope_Iterator;

#pragma link C++ option=nomap typedef ROOT::Reflex::Object_Cont_Type_t;
#pragma link C++ option=nomap typedef ROOT::Reflex::Object_Iterator;
#pragma link C++ option=nomap typedef ROOT::Reflex::Reverse_Object_Iterator;

#pragma link C++ option=nomap typedef ROOT::Reflex::Member_Cont_Type_t;
#pragma link C++ option=nomap typedef ROOT::Reflex::Member_Iterator;
#pragma link C++ option=nomap typedef ROOT::Reflex::Reverse_Member_Iterator;

#pragma link C++ option=nomap typedef ROOT::Reflex::TypeTemplate_Cont_Type_t;
#pragma link C++ option=nomap typedef ROOT::Reflex::TypeTemplate_Iterator;
#pragma link C++ option=nomap typedef ROOT::Reflex::Reverse_TypeTemplate_Iterator;

#pragma link C++ option=nomap typedef ROOT::Reflex::MemberTemplate_Cont_Type_t;
#pragma link C++ option=nomap typedef ROOT::Reflex::MemberTemplate_Iterator;
#pragma link C++ option=nomap typedef ROOT::Reflex::Reverse_MemberTemplate_Iterator;


#pragma link C++ class ROOT::Reflex::ClassBuilder;
#pragma link C++ class ROOT::Reflex::ClassBuilderImpl;
#pragma link C++ class ROOT::Reflex::EnumBuilder;
#pragma link C++ class ROOT::Reflex::FunctionBuilder;
#pragma link C++ class ROOT::Reflex::FunctionBuilderImpl;
#pragma link C++ class ROOT::Reflex::NamespaceBuilder;
#pragma link C++ class ROOT::Reflex::TypedefBuilderImpl;
#pragma link C++ class ROOT::Reflex::UnionBuilderImpl;
#pragma link C++ class ROOT::Reflex::VariableBuilder;
#pragma link C++ class ROOT::Reflex::VariableBuilderImpl;

#pragma link C++ function ROOT::Reflex::TypeBuilder( const char *,unsigned int );
#pragma link C++ function ROOT::Reflex::ConstBuilder( const Type & );
#pragma link C++ function ROOT::Reflex::VolatileBuilder( const Type & );
#pragma link C++ function ROOT::Reflex::PointerBuilder( const Type &, const std::type_info &);
#pragma link C++ function ROOT::Reflex::PointerToMemberBuilder( const Type &, const Scope &, const std::type_info &);
#pragma link C++ function ROOT::Reflex::ReferenceBuilder( const Type& );
#pragma link C++ function ROOT::Reflex::ArrayBuilder( const Type&, size_t, const std::type_info & );
#pragma link C++ function ROOT::Reflex::EnumTypeBuilder( const char *, const char *, const std::type_info &, unsigned int );
#pragma link C++ function ROOT::Reflex::TypedefTypeBuilder( const char *, const Type& );
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder( const Type&, const std::vector<Type> &, const std::type_info & );

#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&);
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&);
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&);
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
/*
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
#pragma link C++ function ROOT::Reflex::FunctionTypeBuilder(const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&, const Type&);
*/

#pragma link C++ struct ROOT::Reflex::NewDelFunctions;
#pragma link C++ struct ROOT::Reflex::CollFuncTable;

#pragma link C++ class ROOT::Reflex::Selection::NO_SELF_AUTOSELECT;
#pragma link C++ class ROOT::Reflex::Selection::TRANSIENT;
#pragma link C++ class ROOT::Reflex::Selection::AUTOSELECT;
#pragma link C++ class ROOT::Reflex::Selection::NODEFAULT;

#endif // __CINT__
