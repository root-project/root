//===------- HackForDefaultTemplateArg.h - Make template argument substitution mroe permissive -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//===----------------------------------------------------------------------===//
//
//  Enabling this hack, make the template substitution more permissive and
//  allows for replacement with non-canonical types.  This is usefull in the
//  case of client code emulating opaque typedefs and/or wanting to recover
//  the template instance name as the user would have written if (s)he 
//  expanded the default paramater explicitly.   For example the user might
//  have type: vector<int32_t> and the client wants to see:
//  std::vector<int32_t,std::allocator<int32_t> >
//
//  For convenience purposes the implementation is located in
//  SemaTemplate.cpp
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_HACKFORDEFAULTTEMPLATEARG_H
#define LLVM_CLANG_SEMA_HACKFORDEFAULTTEMPLATEARG_H

namespace clang {
namespace sema {

///  \brief Enabling this hack makes the template substitution more permissive
///  and allows for replacement with non-canonical types.  This is usefull in
///  the case of client code emulating opaque typedefs and/or wanting to recover
///  the template instance name as the user would have written if (s)he
///  expanded the default paramater explicitly.   For example the user might
///  have type: \c vector<int32_t> and the client wants to see:
///  \c std::vector<int32_t,std::allocator<int32_t> >
   
class HackForDefaultTemplateArg {
  /// \brief Private RAII object that set and reset the hack state.

  static bool AllowNonCanonicalSubstEnabled;
  bool OldValue;
public:

  HackForDefaultTemplateArg();
  ~HackForDefaultTemplateArg();
  
  static bool AllowNonCanonicalSubst();
};
  
} // sema
} // clang

#endif // LLVM_CLANG_SEMA_HACKFORDEFAULTTEMPLATEARG_H
