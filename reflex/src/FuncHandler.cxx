// @(#)root/reflex:$Name: HEAD $:$Id: FuncHandler.cxx,v 1.7 2006/03/20 09:46:18 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef REFLEX_BUILD
#define REFLEX_BUILD
#endif

#include "FuncHandler.h"

#include "Reflex/Tools.h"

#include <typeinfo>


//-------------------------------------------------------------------------------
template < class R, class C >
const std::string ROOT::Reflex::FuncHandler::DemangleFunRetType(R(C::*)()) {
//-------------------------------------------------------------------------------
   return Tools::Demangle(typeid(R));
}


//-------------------------------------------------------------------------------
template < class R, class C, class T0>
const std::string ROOT::Reflex::FuncHandler::DemangleFunRetType(R(C::*)(T0)) {
//-------------------------------------------------------------------------------
   return Tools::Demangle(typeid(R));
}


//-------------------------------------------------------------------------------
template < class R, class C, class T0, class T1 >
const std::string ROOT::Reflex::FuncHandler::DemangleFunRetType(R(C::*)(T0,T1)) {
//-------------------------------------------------------------------------------
   return Tools::Demangle(typeid(R));
}


//-------------------------------------------------------------------------------
template < class R, class C, class T0, class T1, class T2 >
const std::string ROOT::Reflex::FuncHandler::DemangleFunRetType(R(C::*)(T0,T1,T2)) {
//-------------------------------------------------------------------------------
   return Tools::Demangle(typeid(R));
}


//-------------------------------------------------------------------------------
template < class R, class C, class T0, class T1, class T2, class T3 >
const std::string ROOT::Reflex::FuncHandler::DemangleFunRetType(R(C::*)(T0,T1,T2,T3)) {
//-------------------------------------------------------------------------------
   return Tools::Demangle(typeid(R));
}


//-------------------------------------------------------------------------------
template < class R, class C, class T0, class T1, class T2, class T3, class T4 >
const std::string ROOT::Reflex::FuncHandler::DemangleFunRetType(R(C::*)(T0,T1,T2,T3,T4)) {
//-------------------------------------------------------------------------------
   return Tools::Demangle(typeid(R));
}


//-------------------------------------------------------------------------------
template < class R, class C >
const std::string ROOT::Reflex::FuncHandler::DemangleFunParTypes(R(C::*)()) {
//-------------------------------------------------------------------------------
   return "void";
}


//-------------------------------------------------------------------------------
template < class R, class C, class T0>
const std::string ROOT::Reflex::FuncHandler::DemangleFunParTypes(R(C::*)(T0)) {
//-------------------------------------------------------------------------------
   return Tools::Demangle(typeid(T0));
}


//-------------------------------------------------------------------------------
template < class R, class C, class T0, class T1 >
const std::string ROOT::Reflex::FuncHandler::DemangleFunParTypes(R(C::*)(T0,T1)) {
//-------------------------------------------------------------------------------
   std::string s = 
      Tools::Demangle(typeid(T0)) + ";" +
      Tools::Demangle(typeid(T1));
   return s;
}


//-------------------------------------------------------------------------------
template < class R, class C, class T0, class T1, class T2 >
const std::string ROOT::Reflex::FuncHandler::DemangleFunParTypes(R(C::*)(T0,T1,T2)) {
//-------------------------------------------------------------------------------
   std::string s =
      Tools::Demangle(typeid(T0)) + ";" +
      Tools::Demangle(typeid(T1)) + ";" +
      Tools::Demangle(typeid(T2));
   return s;
}


//-------------------------------------------------------------------------------
template < class R, class C, class T0, class T1, class T2, class T3 >
const std::string ROOT::Reflex::FuncHandler::DemangleFunParTypes(R(C::*)(T0,T1,T2,T3)) {
//-------------------------------------------------------------------------------
   std::string s = 
      Tools::Demangle(typeid(T0)) + ";" +
      Tools::Demangle(typeid(T1)) + ";" +
      Tools::Demangle(typeid(T2)) + ";" +
      Tools::Demangle(typeid(T3));
   return s;
}


//-------------------------------------------------------------------------------
template < class R, class C, class T0, class T1, class T2, class T3, class T4 >
const std::string ROOT::Reflex::FuncHandler::DemangleFunParTypes(R(C::*)(T0,T1,T2,T3,T4)) {
//-------------------------------------------------------------------------------
   std::string s = 
      Tools::Demangle(typeid(T0)) + ";" +
      Tools::Demangle(typeid(T1)) + ";" +
      Tools::Demangle(typeid(T2)) + ";" +
      Tools::Demangle(typeid(T3)) + ";" +
      Tools::Demangle(typeid(T4));
   return s;
}


//-------------------------------------------------------------------------------
template < class R >
const std::string ROOT::Reflex::FuncHandler::DemangleFunRetType(R(*)()) {
//-------------------------------------------------------------------------------
   return Tools::Demangle(typeid(R));
}


//-------------------------------------------------------------------------------
template < class R, class T0>
const std::string ROOT::Reflex::FuncHandler::DemangleFunRetType(R(*)(T0)) {
//-------------------------------------------------------------------------------
   return Tools::Demangle(typeid(R));
}


//-------------------------------------------------------------------------------
template < class R, class T0, class T1 >
const std::string ROOT::Reflex::FuncHandler::DemangleFunRetType(R(*)(T0,T1)) {
//-------------------------------------------------------------------------------
   return Tools::Demangle(typeid(R));
}


//-------------------------------------------------------------------------------
template < class R, class T0, class T1, class T2 >
const std::string ROOT::Reflex::FuncHandler::DemangleFunRetType(R(*)(T0,T1,T2)) {
//-------------------------------------------------------------------------------
   return Tools::Demangle(typeid(R));
}


//-------------------------------------------------------------------------------
template < class R, class T0, class T1, class T2, class T3 >
const std::string ROOT::Reflex::FuncHandler::DemangleFunRetType(R(*)(T0,T1,T2,T3)) {
//-------------------------------------------------------------------------------
   return Tools::Demangle(typeid(R));
}


//-------------------------------------------------------------------------------
template < class R, class T0, class T1, class T2, class T3, class T4 >
const std::string ROOT::Reflex::FuncHandler::DemangleFunRetType(R(*)(T0,T1,T2,T3,T4)) {
//-------------------------------------------------------------------------------
   return Tools::Demangle(typeid(R));
}


//-------------------------------------------------------------------------------
template < class R >
const std::string ROOT::Reflex::FuncHandler::DemangleFunParTypes(R(*)()) {
//-------------------------------------------------------------------------------
   return "void";
}


//-------------------------------------------------------------------------------
template < class R, class T0>
const std::string ROOT::Reflex::FuncHandler::DemangleFunParTypes(R(*)(T0)) {
//-------------------------------------------------------------------------------
   return Tools::Demangle(typeid(T0));
}


//-------------------------------------------------------------------------------
template < class R, class T0, class T1 >
const std::string ROOT::Reflex::FuncHandler::DemangleFunParTypes(R(*)(T0,T1)) {
//-------------------------------------------------------------------------------
   std::string s = 
      Tools::Demangle(typeid(T0)) + ";" +
      Tools::Demangle(typeid(T1));
   return s;
}


//-------------------------------------------------------------------------------
template < class R, class T0, class T1, class T2 >
const std::string ROOT::Reflex::FuncHandler::DemangleFunParTypes(R(*)(T0,T1,T2)) {
//-------------------------------------------------------------------------------
   std::string s = 
      Tools::Demangle(typeid(T0)) + ";" +
      Tools::Demangle(typeid(T1)) + ";" +
      Tools::Demangle(typeid(T2));
   return s;
}


//-------------------------------------------------------------------------------
template < class R, class T0, class T1, class T2, class T3 >
const std::string ROOT::Reflex::FuncHandler::DemangleFunParTypes(R(*)(T0,T1,T2,T3)) {
//-------------------------------------------------------------------------------
   std::string s = 
      Tools::Demangle(typeid(T0)) + ";" +
      Tools::Demangle(typeid(T1)) + ";" +
      Tools::Demangle(typeid(T2)) + ";" +
      Tools::Demangle(typeid(T3));
   return s;
}


//-------------------------------------------------------------------------------
template < class R, class T0, class T1, class T2, class T3, class T4 >
const std::string ROOT::Reflex::FuncHandler::DemangleFunParTypes(R(*)(T0,T1,T2,T3,T4)) {
//-------------------------------------------------------------------------------
   std::string s = 
      Tools::Demangle(typeid(T0)) + ";" +
      Tools::Demangle(typeid(T1)) + ";" +
      Tools::Demangle(typeid(T2)) + ";" +
      Tools::Demangle(typeid(T3)) + ";" +
      Tools::Demangle(typeid(T4));
   return s;
}
