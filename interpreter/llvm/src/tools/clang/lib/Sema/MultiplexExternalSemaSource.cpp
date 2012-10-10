//===--- MultiplexExternalSemaSource.cpp  ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the event dispatching to the subscribed clients.
//
//===----------------------------------------------------------------------===//
#include "clang/Sema/MultiplexExternalSemaSource.h"
#include "clang/Sema/Lookup.h"

using namespace clang;

///\brief Constructs a new multiplexing external sema source and appends the
/// given element to it.
///
///\param[in] source - An ExternalSemaSource.
///
MultiplexExternalSemaSource::MultiplexExternalSemaSource(ExternalSemaSource &s1,
                                                         ExternalSemaSource &s2){
  Sources.push_back(&s1);
  Sources.push_back(&s2);
}

// pin the vtable here.
MultiplexExternalSemaSource::~MultiplexExternalSemaSource() {}

///\brief Appends new source to the source list.
///
///\param[in] source - An ExternalSemaSource.
///
void MultiplexExternalSemaSource::addSource(ExternalSemaSource &source) {
  Sources.push_back(&source);
}

void MultiplexExternalSemaSource::InitializeSema(Sema &S) {
  for(size_t i = 0; i < Sources.size(); ++i)
    Sources[i]->InitializeSema(S);
}

void MultiplexExternalSemaSource::ForgetSema() {
  for(size_t i = 0; i < Sources.size(); ++i)
    Sources[i]->ForgetSema();
}

void MultiplexExternalSemaSource::ReadMethodPool(Selector Sel) {
  for(size_t i = 0; i < Sources.size(); ++i)
    Sources[i]->ReadMethodPool(Sel);
}

void MultiplexExternalSemaSource::ReadKnownNamespaces(
                                   SmallVectorImpl<NamespaceDecl*> &Namespaces){
  for(size_t i = 0; i < Sources.size(); ++i)
    Sources[i]->ReadKnownNamespaces(Namespaces);
}
  
bool MultiplexExternalSemaSource::LookupUnqualified(LookupResult &R, Scope *S){ 
  for(size_t i = 0; i < Sources.size(); ++i)
    Sources[i]->LookupUnqualified(R, S);
  
  return !R.empty();
}

void MultiplexExternalSemaSource::ReadTentativeDefinitions(
                                     SmallVectorImpl<VarDecl*> &TentativeDefs) {
  for(size_t i = 0; i < Sources.size(); ++i)
    Sources[i]->ReadTentativeDefinitions(TentativeDefs);
}
  
void MultiplexExternalSemaSource::ReadUnusedFileScopedDecls(
                                SmallVectorImpl<const DeclaratorDecl*> &Decls) {
  for(size_t i = 0; i < Sources.size(); ++i)
    Sources[i]->ReadUnusedFileScopedDecls(Decls);
}
  
void MultiplexExternalSemaSource::ReadDelegatingConstructors(
                                  SmallVectorImpl<CXXConstructorDecl*> &Decls) {
  for(size_t i = 0; i < Sources.size(); ++i)
    Sources[i]->ReadDelegatingConstructors(Decls);
}

void MultiplexExternalSemaSource::ReadExtVectorDecls(
                                     SmallVectorImpl<TypedefNameDecl*> &Decls) {
  for(size_t i = 0; i < Sources.size(); ++i)
    Sources[i]->ReadExtVectorDecls(Decls);
}

void MultiplexExternalSemaSource::ReadDynamicClasses(
                                       SmallVectorImpl<CXXRecordDecl*> &Decls) {
  for(size_t i = 0; i < Sources.size(); ++i)
    Sources[i]->ReadDynamicClasses(Decls);
}

void MultiplexExternalSemaSource::ReadLocallyScopedExternalDecls(
                                           SmallVectorImpl<NamedDecl*> &Decls) {
  for(size_t i = 0; i < Sources.size(); ++i)
    Sources[i]->ReadLocallyScopedExternalDecls(Decls);
}

void MultiplexExternalSemaSource::ReadReferencedSelectors(
                  SmallVectorImpl<std::pair<Selector, SourceLocation> > &Sels) {
  for(size_t i = 0; i < Sources.size(); ++i)
    Sources[i]->ReadReferencedSelectors(Sels);
}

void MultiplexExternalSemaSource::ReadWeakUndeclaredIdentifiers(
                   SmallVectorImpl<std::pair<IdentifierInfo*, WeakInfo> > &WI) {
  for(size_t i = 0; i < Sources.size(); ++i)
    Sources[i]->ReadWeakUndeclaredIdentifiers(WI);
}

void MultiplexExternalSemaSource::ReadUsedVTables(
                                  SmallVectorImpl<ExternalVTableUse> &VTables) {
  for(size_t i = 0; i < Sources.size(); ++i)
    Sources[i]->ReadUsedVTables(VTables);
}

void MultiplexExternalSemaSource::ReadPendingInstantiations(
                                           SmallVectorImpl<std::pair<ValueDecl*,
                                                   SourceLocation> > &Pending) {
  for(size_t i = 0; i < Sources.size(); ++i)
    Sources[i]->ReadPendingInstantiations(Pending);
}
