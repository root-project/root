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

#include "FuncHandler.h"

#include "Reflex/Tools.h"

#include <typeinfo>


//-------------------------------------------------------------------------------
template <class R, class C>
const std::string
Reflex::FuncHandler::DemangleFunRetType(R (C::*)()) {
//-------------------------------------------------------------------------------
// Demangle return type of  a member function with 0 arguments.
   return Tools::Demangle(typeid(R));
}


//-------------------------------------------------------------------------------
template <class R, class C, class T0>
const std::string
Reflex::FuncHandler::DemangleFunRetType(R (C::*)(T0)) {
//-------------------------------------------------------------------------------
// Demangle return type of  a member function with 1 argument.
   return Tools::Demangle(typeid(R));
}


//-------------------------------------------------------------------------------
template <class R, class C, class T0, class T1>
const std::string
Reflex::FuncHandler::DemangleFunRetType(R (C::*)(T0, T1)) {
//-------------------------------------------------------------------------------
// Demangle return type of  a member function with 2 arguments.
   return Tools::Demangle(typeid(R));
}


//-------------------------------------------------------------------------------
template <class R, class C, class T0, class T1, class T2>
const std::string
Reflex::FuncHandler::DemangleFunRetType(R (C::*)(T0, T1, T2)) {
//-------------------------------------------------------------------------------
// Demangle return type of  a member function with 3 arguments.
   return Tools::Demangle(typeid(R));
}


//-------------------------------------------------------------------------------
template <class R, class C, class T0, class T1, class T2, class T3>
const std::string
Reflex::FuncHandler::DemangleFunRetType(R (C::*)(T0, T1, T2, T3)) {
//-------------------------------------------------------------------------------
// Demangle return type of  a member function with 4 arguments.
   return Tools::Demangle(typeid(R));
}


//-------------------------------------------------------------------------------
template <class R, class C, class T0, class T1, class T2, class T3, class T4>
const std::string
Reflex::FuncHandler::DemangleFunRetType(R (C::*)(T0, T1, T2, T3, T4)) {
//-------------------------------------------------------------------------------
// Demangle return type of  a member function with 5 arguments.
   return Tools::Demangle(typeid(R));
}


//-------------------------------------------------------------------------------
template <class R, class C>
const std::string
Reflex::FuncHandler::DemangleFunParTypes(R (C::*)()) {
//-------------------------------------------------------------------------------
// Demangle parameters of a member function with 0 arguments.
   return "void";
}


//-------------------------------------------------------------------------------
template <class R, class C, class T0>
const std::string
Reflex::FuncHandler::DemangleFunParTypes(R (C::*)(T0)) {
//-------------------------------------------------------------------------------
// Demangle parameters of a member function with 1 argument.
   return Tools::Demangle(typeid(T0));
}


//-------------------------------------------------------------------------------
template <class R, class C, class T0, class T1>
const std::string
Reflex::FuncHandler::DemangleFunParTypes(R (C::*)(T0, T1)) {
//-------------------------------------------------------------------------------
// Demangle parameters of a member function with 2 arguments.
   std::string s =
      Tools::Demangle(typeid(T0)) + ";" +
      Tools::Demangle(typeid(T1));
   return s;
}


//-------------------------------------------------------------------------------
template <class R, class C, class T0, class T1, class T2>
const std::string
Reflex::FuncHandler::DemangleFunParTypes(R (C::*)(T0, T1, T2)) {
//-------------------------------------------------------------------------------
// Demangle parameters of a member function with 3 arguments.
   std::string s =
      Tools::Demangle(typeid(T0)) + ";" +
      Tools::Demangle(typeid(T1)) + ";" +
      Tools::Demangle(typeid(T2));
   return s;
}


//-------------------------------------------------------------------------------
template <class R, class C, class T0, class T1, class T2, class T3>
const std::string
Reflex::FuncHandler::DemangleFunParTypes(R (C::*)(T0, T1, T2, T3)) {
//-------------------------------------------------------------------------------
// Demangle parameters of a member function with 4 arguments.
   std::string s =
      Tools::Demangle(typeid(T0)) + ";" +
      Tools::Demangle(typeid(T1)) + ";" +
      Tools::Demangle(typeid(T2)) + ";" +
      Tools::Demangle(typeid(T3));
   return s;
}


//-------------------------------------------------------------------------------
template <class R, class C, class T0, class T1, class T2, class T3, class T4>
const std::string
Reflex::FuncHandler::DemangleFunParTypes(R (C::*)(T0, T1, T2, T3, T4)) {
//-------------------------------------------------------------------------------
// Demangle parameters of a member function with 5 arguments.
   std::string s =
      Tools::Demangle(typeid(T0)) + ";" +
      Tools::Demangle(typeid(T1)) + ";" +
      Tools::Demangle(typeid(T2)) + ";" +
      Tools::Demangle(typeid(T3)) + ";" +
      Tools::Demangle(typeid(T4));
   return s;
}


//-------------------------------------------------------------------------------
template <class R>
const std::string
Reflex::FuncHandler::DemangleFunRetType(R(*) ()) {
//-------------------------------------------------------------------------------
// Demangle return type of a free function with 0 arguments.
   return Tools::Demangle(typeid(R));
}


//-------------------------------------------------------------------------------
template <class R, class T0>
const std::string
Reflex::FuncHandler::DemangleFunRetType(R(*) (T0)) {
//-------------------------------------------------------------------------------
// Demangle return type of a free function with 0 arguments.
   return Tools::Demangle(typeid(R));
}


//-------------------------------------------------------------------------------
template <class R, class T0, class T1>
const std::string
Reflex::FuncHandler::DemangleFunRetType(R(*) (T0, T1)) {
//-------------------------------------------------------------------------------
// Demangle return type of a free function with 0 arguments.
   return Tools::Demangle(typeid(R));
}


//-------------------------------------------------------------------------------
template <class R, class T0, class T1, class T2>
const std::string
Reflex::FuncHandler::DemangleFunRetType(R(*) (T0, T1, T2)) {
//-------------------------------------------------------------------------------
// Demangle return type of a free function with 0 arguments.
   return Tools::Demangle(typeid(R));
}


//-------------------------------------------------------------------------------
template <class R, class T0, class T1, class T2, class T3>
const std::string
Reflex::FuncHandler::DemangleFunRetType(R(*) (T0, T1, T2, T3)) {
//-------------------------------------------------------------------------------
// Demangle return type of a free function with 0 arguments.
   return Tools::Demangle(typeid(R));
}


//-------------------------------------------------------------------------------
template <class R, class T0, class T1, class T2, class T3, class T4>
const std::string
Reflex::FuncHandler::DemangleFunRetType(R(*) (T0, T1, T2, T3, T4)) {
//-------------------------------------------------------------------------------
// Demangle return type of a free function with 0 arguments.
   return Tools::Demangle(typeid(R));
}


//-------------------------------------------------------------------------------
template <class R>
const std::string
Reflex::FuncHandler::DemangleFunParTypes(R(*) ()) {
//-------------------------------------------------------------------------------
// Demangle parameter types of a free function with 0 arguments.
   return "void";
}


//-------------------------------------------------------------------------------
template <class R, class T0>
const std::string
Reflex::FuncHandler::DemangleFunParTypes(R(*) (T0)) {
//-------------------------------------------------------------------------------
// Demangle parameter types of a free function with 1 argument.
   return Tools::Demangle(typeid(T0));
}


//-------------------------------------------------------------------------------
template <class R, class T0, class T1>
const std::string
Reflex::FuncHandler::DemangleFunParTypes(R(*) (T0, T1)) {
//-------------------------------------------------------------------------------
// Demangle parameter types of a free function with 2 arguments.
   std::string s =
      Tools::Demangle(typeid(T0)) + ";" +
      Tools::Demangle(typeid(T1));
   return s;
}


//-------------------------------------------------------------------------------
template <class R, class T0, class T1, class T2>
const std::string
Reflex::FuncHandler::DemangleFunParTypes(R(*) (T0, T1, T2)) {
//-------------------------------------------------------------------------------
// Demangle parameter types of a free function with 3 arguments.
   std::string s =
      Tools::Demangle(typeid(T0)) + ";" +
      Tools::Demangle(typeid(T1)) + ";" +
      Tools::Demangle(typeid(T2));
   return s;
}


//-------------------------------------------------------------------------------
template <class R, class T0, class T1, class T2, class T3>
const std::string
Reflex::FuncHandler::DemangleFunParTypes(R(*) (T0, T1, T2, T3)) {
//-------------------------------------------------------------------------------
// Demangle parameter types of a free function with 4 arguments.
   std::string s =
      Tools::Demangle(typeid(T0)) + ";" +
      Tools::Demangle(typeid(T1)) + ";" +
      Tools::Demangle(typeid(T2)) + ";" +
      Tools::Demangle(typeid(T3));
   return s;
}


//-------------------------------------------------------------------------------
template <class R, class T0, class T1, class T2, class T3, class T4>
const std::string
Reflex::FuncHandler::DemangleFunParTypes(R(*) (T0, T1, T2, T3, T4)) {
//-------------------------------------------------------------------------------
// Demangle parameter types of a free function with 5 arguments.
   std::string s =
      Tools::Demangle(typeid(T0)) + ";" +
      Tools::Demangle(typeid(T1)) + ";" +
      Tools::Demangle(typeid(T2)) + ";" +
      Tools::Demangle(typeid(T3)) + ";" +
      Tools::Demangle(typeid(T4));
   return s;
}
