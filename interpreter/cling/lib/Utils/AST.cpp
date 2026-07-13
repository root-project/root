//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vasil.georgiev.vasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "cling/Utils/AST.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "clang/AST/Mangle.h"

#include <memory>
#include <stdio.h>

using namespace clang;

namespace {
  template<typename D>
  static D* LookupResult2Decl(clang::LookupResult& R)
  {
    if (R.empty())
      return nullptr;

    R.resolveKind();

    if (R.isSingleResult())
      return dyn_cast<D>(R.getFoundDecl());
    return (D*)-1;
  }

  // LLVM22: Replaces the old nns->getPrefix() pointer call
  static NestedNameSpecifier getNNSPrefix(NestedNameSpecifier NNS) {
    switch (NNS.getKind()) {
    case NestedNameSpecifier::Kind::Namespace:
      return NNS.getAsNamespaceAndPrefix().Prefix;
    case NestedNameSpecifier::Kind::Type:
      return NNS.getAsType()->getPrefix();
    default:
      return NestedNameSpecifier{};
    }
  }

  // LLVM22: Replaces old nns->getAsNamespace()
  static const NamespaceDecl* getNNSNamespace(NestedNameSpecifier NNS) {
    if (NNS.getKind() != NestedNameSpecifier::Kind::Namespace)
      return nullptr;
    return dyn_cast<NamespaceDecl>(NNS.getAsNamespaceAndPrefix().Namespace);
  }

  // LLVM22: Replaces old nns->getAsNamespaceAlias()
  static const NamespaceAliasDecl* getNNSNamespaceAlias(NestedNameSpecifier NNS) {
    if (NNS.getKind() != NestedNameSpecifier::Kind::Namespace)
      return nullptr;
    return dyn_cast<NamespaceAliasDecl>(NNS.getAsNamespaceAndPrefix().Namespace);
  }
}

namespace cling {
namespace utils {
namespace TypeName {

  // Forward declare
  QualType getFullyQualifiedType(QualType QT, const ASTContext &Ctx,
                                 bool WithGlobalNsPrefix);

  // The code below is mostly copied from clang/lib/AST/QualTypeNames.cpp
  // as it became harder and diverged after the NestedNameSpecifier changes
  // some of the functions that we need are not exposed (static).
  // For the upgrade to proceed, it is better to have this here than patch
  // clang.
  // We also need some modifications in getFullyQualifiedType, mainly
  // for auto-type and decl-type handling that is not done upstream.
  // Rest of the functions are simply copied.

  /// Create a NestedNameSpecifier for Namesp and its enclosing
  /// scopes.
  ///
  /// \param[in] Ctx - the AST Context to be used.
  /// \param[in] Namesp - the NamespaceDecl for which a NestedNameSpecifier
  /// is requested.
  /// \param[in] WithGlobalNsPrefix - Indicate whether the global namespace
  /// specifier "::" should be prepended or not.
  static NestedNameSpecifier
  createNestedNameSpecifier(const ASTContext &Ctx, const NamespaceDecl *Namesp,
                            bool WithGlobalNsPrefix);

  /// Create a NestedNameSpecifier for TagDecl and its enclosing
  /// scopes.
  ///
  /// \param[in] Ctx - the AST Context to be used.
  /// \param[in] TD - the TagDecl for which a NestedNameSpecifier is
  /// requested.
  /// \param[in] FullyQualify - Convert all template arguments into fully
  /// qualified names.
  /// \param[in] WithGlobalNsPrefix - Indicate whether the global namespace
  /// specifier "::" should be prepended or not.
  static NestedNameSpecifier createNestedNameSpecifier(const ASTContext &Ctx,
                                                       const TypeDecl *TD,
                                                       bool FullyQualify,
                                                       bool WithGlobalNsPrefix);

  static NestedNameSpecifier
  createNestedNameSpecifierForScopeOf(const ASTContext &Ctx, const Decl *decl,
                                      bool FullyQualified,
                                      bool WithGlobalNsPrefix);

  static NestedNameSpecifier getFullyQualifiedNestedNameSpecifier(
      const ASTContext &Ctx, NestedNameSpecifier NNS, bool WithGlobalNsPrefix);

  static bool getFullyQualifiedTemplateName(const ASTContext &Ctx,
                                            TemplateName &TName,
                                            bool WithGlobalNsPrefix) {
    bool Changed = false;
    NestedNameSpecifier NNS = std::nullopt;

    TemplateDecl *ArgTDecl = TName.getAsTemplateDecl();
    if (!ArgTDecl) // ArgTDecl can be null in dependent contexts.
      return false;

    QualifiedTemplateName *QTName = TName.getAsQualifiedTemplateName();

    if (QTName &&
        !QTName->hasTemplateKeyword() &&
        (NNS = QTName->getQualifier())) {
      NestedNameSpecifier QNNS =
          getFullyQualifiedNestedNameSpecifier(Ctx, NNS, WithGlobalNsPrefix);
      if (QNNS != NNS) {
        Changed = true;
        NNS = QNNS;
      } else {
        NNS = std::nullopt;
      }
    } else {
      NNS = createNestedNameSpecifierForScopeOf(
          Ctx, ArgTDecl, true, WithGlobalNsPrefix);
    }
    if (NNS) {
      TemplateName UnderlyingTN(ArgTDecl);
      if (UsingShadowDecl *USD = TName.getAsUsingShadowDecl())
        UnderlyingTN = TemplateName(USD);
      TName =
          Ctx.getQualifiedTemplateName(NNS,
                                       /*TemplateKeyword=*/false, UnderlyingTN);
      Changed = true;
    }
    return Changed;
  }

  static bool getFullyQualifiedTemplateArgument(const ASTContext &Ctx,
                                                TemplateArgument &Arg,
                                                bool WithGlobalNsPrefix) {
    bool Changed = false;

    // Note: we do not handle TemplateArgument::Expression, to replace it
    // we need the information for the template instance decl.

    if (Arg.getKind() == TemplateArgument::Template) {
      TemplateName TName = Arg.getAsTemplate();
      Changed = getFullyQualifiedTemplateName(Ctx, TName, WithGlobalNsPrefix);
      if (Changed) {
        Arg = TemplateArgument(TName);
      }
    } else if (Arg.getKind() == TemplateArgument::Type) {
      QualType SubTy = Arg.getAsType();
      // Check if the type needs more desugaring and recurse.
      QualType QTFQ = getFullyQualifiedType(SubTy, Ctx, WithGlobalNsPrefix);
      if (QTFQ != SubTy) {
        Arg = TemplateArgument(QTFQ);
        Changed = true;
      }
    }
    return Changed;
  }

  static const Type *getFullyQualifiedTemplateType(const ASTContext &Ctx,
                                                   const TagType *TSTRecord,
                                                   ElaboratedTypeKeyword Keyword,
                                                   NestedNameSpecifier Qualifier,
                                                   bool WithGlobalNsPrefix) {
    // We are asked to fully qualify and we have a Record Type,
    // which can point to a template instantiation with no sugar in any of
    // its template argument, however we still need to fully qualify them.

    const auto *TD = TSTRecord->getDecl();
    const auto *TSTDecl = dyn_cast<ClassTemplateSpecializationDecl>(TD);
    if (!TSTDecl)
      return Ctx.getTagType(Keyword, Qualifier, TD, /*OwnsTag=*/false)
          .getTypePtr();

    const TemplateArgumentList &TemplateArgs = TSTDecl->getTemplateArgs();

    bool MightHaveChanged = false;
    SmallVector<TemplateArgument, 4> FQArgs;
    for (unsigned int I = 0, E = TemplateArgs.size(); I != E; ++I) {
      // cheap to copy and potentially modified by
      // getFullyQualifedTemplateArgument
      TemplateArgument Arg(TemplateArgs[I]);
      MightHaveChanged |=
          getFullyQualifiedTemplateArgument(Ctx, Arg, WithGlobalNsPrefix);
      FQArgs.push_back(Arg);
    }

    if (!MightHaveChanged)
      return Ctx.getTagType(Keyword, Qualifier, TD, /*OwnsTag=*/false)
          .getTypePtr();
    // If a fully qualified arg is different from the unqualified arg,
    // allocate new type in the AST.
    TemplateName TN = Ctx.getQualifiedTemplateName(
        Qualifier, /*TemplateKeyword=*/false,
        TemplateName(TSTDecl->getSpecializedTemplate()));
    QualType QT = Ctx.getTemplateSpecializationType(
        Keyword, TN, FQArgs,
        /*CanonicalArgs=*/{}, TSTRecord->getCanonicalTypeInternal());
    // getTemplateSpecializationType returns a fully qualified
    // version of the specialization itself, so no need to qualify
    // it.
    return QT.getTypePtr();
  }

  static const Type *
  getFullyQualifiedTemplateType(const ASTContext &Ctx,
                                const TemplateSpecializationType *TST,
                                bool WithGlobalNsPrefix) {
    TemplateName TName = TST->getTemplateName();
    bool MightHaveChanged =
        getFullyQualifiedTemplateName(Ctx, TName, WithGlobalNsPrefix);
    SmallVector<TemplateArgument, 4> FQArgs;
    // Cheap to copy and potentially modified by
    // getFullyQualifedTemplateArgument.
    for (TemplateArgument Arg : TST->template_arguments()) {
      MightHaveChanged |=
          getFullyQualifiedTemplateArgument(Ctx, Arg, WithGlobalNsPrefix);
      FQArgs.push_back(Arg);
    }

    if (!MightHaveChanged)
      return TST;

    QualType NewQT =
        Ctx.getTemplateSpecializationType(TST->getKeyword(), TName, FQArgs,
                                          /*CanonicalArgs=*/{}, TST->desugar());
    // getTemplateSpecializationType returns a fully qualified
    // version of the specialization itself, so no need to qualify
    // it.
    return NewQT.getTypePtr();
  }

  static NestedNameSpecifier createOuterNNS(const ASTContext &Ctx, const Decl *D,
                                            bool FullyQualify,
                                            bool WithGlobalNsPrefix) {
    const DeclContext *DC = D->getDeclContext();
    if (const auto *NS = dyn_cast<NamespaceDecl>(DC)) {
      while (NS && NS->isInline()) {
        // Ignore inline namespace;
        NS = dyn_cast<NamespaceDecl>(NS->getDeclContext());
      }
      if (NS && NS->getDeclName()) {
        return createNestedNameSpecifier(Ctx, NS, WithGlobalNsPrefix);
      }
      return std::nullopt; // no starting '::', no anonymous
    }
    if (const auto *TD = dyn_cast<TagDecl>(DC))
      return createNestedNameSpecifier(Ctx, TD, FullyQualify, WithGlobalNsPrefix);
    if (const auto *TDD = dyn_cast<TypedefNameDecl>(DC))
      return createNestedNameSpecifier(Ctx, TDD, FullyQualify,
                                       WithGlobalNsPrefix);
    if (WithGlobalNsPrefix && DC->isTranslationUnit())
      return NestedNameSpecifier::getGlobal();
    return std::nullopt; // no starting '::' if |WithGlobalNsPrefix| is false
  }

  /// Return a fully qualified version of this name specifier.
  static NestedNameSpecifier getFullyQualifiedNestedNameSpecifier(
      const ASTContext &Ctx, NestedNameSpecifier Scope, bool WithGlobalNsPrefix) {
    switch (Scope.getKind()) {
    case NestedNameSpecifier::Kind::Null:
      llvm_unreachable("can't fully qualify the empty nested name specifier");
    case NestedNameSpecifier::Kind::Global:
    case NestedNameSpecifier::Kind::MicrosoftSuper:
      // Already fully qualified
      return Scope;
    case NestedNameSpecifier::Kind::Namespace:
      return TypeName::createNestedNameSpecifier(
          Ctx, Scope.getAsNamespaceAndPrefix().Namespace->getNamespace(),
          WithGlobalNsPrefix);
    case NestedNameSpecifier::Kind::Type: {
      const Type *Type = Scope.getAsType();
      // Find decl context.
      const TypeDecl *TD;
      if (const TagType *TagDeclType = Type->getAs<TagType>())
        TD = TagDeclType->getDecl();
      else if (const auto *D = dyn_cast<TypedefType>(Type))
        TD = D->getDecl();
      else
        return Scope;
      return TypeName::createNestedNameSpecifier(Ctx, TD, /*FullyQualify=*/true,
                                                 WithGlobalNsPrefix);
    }
    }
    llvm_unreachable("bad NNS kind");
  }

  /// Create a nested name specifier for the declaring context of
  /// the type.
  static NestedNameSpecifier
  createNestedNameSpecifierForScopeOf(const ASTContext &Ctx, const Decl *Decl,
                                      bool FullyQualified,
                                      bool WithGlobalNsPrefix) {
    assert(Decl);

    // Some declaration cannot be qualified.
    if (Decl->isTemplateParameter())
      return std::nullopt;
    const DeclContext *DC = Decl->getDeclContext()->getRedeclContext();
    const auto *Outer = dyn_cast<NamedDecl>(DC);
    const auto *OuterNS = dyn_cast<NamespaceDecl>(DC);
    if (OuterNS && OuterNS->isAnonymousNamespace())
      OuterNS = dyn_cast<NamespaceDecl>(OuterNS->getParent());
    if (Outer) {
      if (const auto *CxxDecl = dyn_cast<CXXRecordDecl>(DC)) {
        if (ClassTemplateDecl *ClassTempl =
                CxxDecl->getDescribedClassTemplate()) {
          // We are in the case of a type(def) that was declared in a
          // class template but is *not* type dependent.  In clang, it
          // gets attached to the class template declaration rather than
          // any specific class template instantiation.  This result in
          // 'odd' fully qualified typename:
          //
          //    vector<_Tp,_Alloc>::size_type
          //
          // Make the situation is 'useable' but looking a bit odd by
          // picking a random instance as the declaring context.
          if (!ClassTempl->specializations().empty()) {
            Decl = *(ClassTempl->spec_begin());
            Outer = dyn_cast<NamedDecl>(Decl);
            OuterNS = dyn_cast<NamespaceDecl>(Decl);
          }
        }
      }

      if (OuterNS) {
        return createNestedNameSpecifier(Ctx, OuterNS, WithGlobalNsPrefix);
      } else if (const auto *TD = dyn_cast<TagDecl>(Outer)) {
        return createNestedNameSpecifier(
            Ctx, TD, FullyQualified, WithGlobalNsPrefix);
      } else if (isa<TranslationUnitDecl>(Outer)) {
        // Context is the TU. Nothing needs to be done.
        return std::nullopt;
      } else {
        // Decl's context was neither the TU, a namespace, nor a
        // TagDecl, which means it is a type local to a scope, and not
        // accessible at the end of the TU.
        return std::nullopt;
      }
    } else if (WithGlobalNsPrefix && DC->isTranslationUnit()) {
      return NestedNameSpecifier::getGlobal();
    }
    return std::nullopt;
  }

  /// Create a nested name specifier for the declaring context of
  /// the type.
  static NestedNameSpecifier
  createNestedNameSpecifierForScopeOf(const ASTContext &Ctx, const Type *TypePtr,
                                      bool FullyQualified,
                                      bool WithGlobalNsPrefix) {
    if (!TypePtr)
      return std::nullopt;

    Decl *Decl = nullptr;
    // There are probably other cases ...
    if (const auto *TDT = dyn_cast<TypedefType>(TypePtr)) {
      Decl = TDT->getDecl();
    } else if (const auto *TagDeclType = dyn_cast<TagType>(TypePtr)) {
      Decl = TagDeclType->getDecl();
    } else if (const auto *TST = dyn_cast<TemplateSpecializationType>(TypePtr)) {
      Decl = TST->getTemplateName().getAsTemplateDecl();
    } else {
      Decl = TypePtr->getAsCXXRecordDecl();
    }

    if (!Decl)
      return std::nullopt;

    return createNestedNameSpecifierForScopeOf(
        Ctx, Decl, FullyQualified, WithGlobalNsPrefix);
  }

  static NestedNameSpecifier
  createNestedNameSpecifier(const ASTContext &Ctx, const NamespaceDecl *Namespace,
                            bool WithGlobalNsPrefix) {
    while (Namespace && Namespace->isInline()) {
      // Ignore inline namespace;
      Namespace = dyn_cast<NamespaceDecl>(Namespace->getDeclContext());
    }
    if (!Namespace)
      return std::nullopt;

    bool FullyQualify = true; // doesn't matter, DeclContexts are namespaces
    return NestedNameSpecifier(
        Ctx, Namespace,
        createOuterNNS(Ctx, Namespace, FullyQualify, WithGlobalNsPrefix));
  }

  NestedNameSpecifier createNestedNameSpecifier(const ASTContext &Ctx,
                                                const TypeDecl *TD,
                                                bool FullyQualify,
                                                bool WithGlobalNsPrefix) {
    const Type *TypePtr = Ctx.getTypeDeclType(TD).getTypePtr();
    if (auto *RD = dyn_cast<TagType>(TypePtr)) {
      // We are asked to fully qualify and we have a Record Type (which
      // may point to a template specialization) or Template
      // Specialization Type. We need to fully qualify their arguments.
      TypePtr = getFullyQualifiedTemplateType(
          Ctx, RD, ElaboratedTypeKeyword::None,
          createOuterNNS(Ctx, TD, FullyQualify, WithGlobalNsPrefix),
          WithGlobalNsPrefix);
    } else if (auto *TST = dyn_cast<TemplateSpecializationType>(TypePtr)) {
      TypePtr = getFullyQualifiedTemplateType(Ctx, TST, WithGlobalNsPrefix);
    }
    return NestedNameSpecifier(TypePtr);
  }

  /// Return the fully qualified type, including fully-qualified
  /// versions of any template parameters.
  QualType getFullyQualifiedType(QualType QT, const ASTContext &Ctx,
                                 bool WithGlobalNsPrefix) {
    // Use the underlying deduced type for AutoType
    if (const auto *AT = dyn_cast<AutoType>(QT.getTypePtr())) {
      if (AT->isDeduced()) {
        // Get the qualifiers.
        Qualifiers Quals = QT.getQualifiers();
        QT = AT->getDeducedType();
        // Add back the qualifiers.
        QT = Ctx.getQualifiedType(QT, Quals);
      }
    }

    // In case of myType* we need to strip the pointer first, fully
    // qualify and attach the pointer once again.
    if (isa<PointerType>(QT.getTypePtr())) {
      // Get the qualifiers.
      Qualifiers Quals = QT.getQualifiers();
      QT = getFullyQualifiedType(QT->getPointeeType(), Ctx, WithGlobalNsPrefix);
      QT = Ctx.getPointerType(QT);
      // Add back the qualifiers.
      QT = Ctx.getQualifiedType(QT, Quals);
      return QT;
    }

    if (auto *MPT = dyn_cast<MemberPointerType>(QT.getTypePtr())) {
      // Get the qualifiers.
      Qualifiers Quals = QT.getQualifiers();
      // Fully qualify the pointee and class types.
      QT = getFullyQualifiedType(QT->getPointeeType(), Ctx, WithGlobalNsPrefix);
      NestedNameSpecifier Qualifier = getFullyQualifiedNestedNameSpecifier(
          Ctx, MPT->getQualifier(), WithGlobalNsPrefix);
      QT = Ctx.getMemberPointerType(QT, Qualifier,
                                    MPT->getMostRecentCXXRecordDecl());
      // Add back the qualifiers.
      QT = Ctx.getQualifiedType(QT, Quals);
      return QT;
    }

    // In case of myType& we need to strip the reference first, fully
    // qualify and attach the reference once again.
    if (isa<ReferenceType>(QT.getTypePtr())) {
      // Get the qualifiers.
      bool IsLValueRefTy = isa<LValueReferenceType>(QT.getTypePtr());
      Qualifiers Quals = QT.getQualifiers();
      QT = getFullyQualifiedType(QT->getPointeeType(), Ctx, WithGlobalNsPrefix);
      // Add the r- or l-value reference type back to the fully
      // qualified one.
      if (IsLValueRefTy)
        QT = Ctx.getLValueReferenceType(QT);
      else
        QT = Ctx.getRValueReferenceType(QT);
      // Add back the qualifiers.
      QT = Ctx.getQualifiedType(QT, Quals);
      return QT;
    }

    // Handle types with attributes such as `unique_ptr<int> _Nonnull`.
    if (auto *AT = dyn_cast<AttributedType>(QT.getTypePtr())) {
      QualType NewModified =
          getFullyQualifiedType(AT->getModifiedType(), Ctx, WithGlobalNsPrefix);
      QualType NewEquivalent =
          getFullyQualifiedType(AT->getEquivalentType(), Ctx, WithGlobalNsPrefix);
      Qualifiers Qualifiers = QT.getLocalQualifiers();
      return Ctx.getQualifiedType(
          Ctx.getAttributedType(AT->getAttrKind(), NewModified, NewEquivalent),
          Qualifiers);
    }

    bool Changed;
    do {
      Changed = false;
      // Remove the part of the type related to the type being a template
      // parameter (we won't report it as part of the 'type name' and it
      // is actually make the code below to be more complex (to handle
      // those)
      while (isa<SubstTemplateTypeParmType>(QT.getTypePtr())) {
        // Get the qualifiers.
        Qualifiers Quals = QT.getQualifiers();
        QT = cast<SubstTemplateTypeParmType>(QT.getTypePtr())->desugar();
        // Add back the qualifiers.
        QT = Ctx.getQualifiedType(QT, Quals);
        Changed = true;
      }

      // Try to get to the underlying type for DecltypeType
      while (const auto *DT = dyn_cast<DecltypeType>(QT.getTypePtr())) {
        // Get the qualifiers.
        Qualifiers Quals = QT.getQualifiers();
        QualType Underlying = DT->getUnderlyingType();
        if (Underlying.isNull() || Underlying->isDependentType())
          break;
        // Add back the qualifiers.
        QT = Ctx.getQualifiedType(Underlying, Quals);
        Changed = true;
      }

      // UnaryTransformType represents compiler built-ins like __remove_extent(T).
      // We must peel these layers back to reach the underlying type so it can be
      // fully qualified.
      while (const auto *UTT = dyn_cast<UnaryTransformType>(QT.getTypePtr())) {
        if (!UTT->isSugared())
          break;
        Qualifiers Quals = QT.getQualifiers();
        QT = Ctx.getQualifiedType(UTT->desugar(), Quals);
        Changed = true;
      }
    } while (Changed);

    if (const auto *TST =
            dyn_cast<const TemplateSpecializationType>(QT.getTypePtr())) {

      const Type *T = getFullyQualifiedTemplateType(Ctx, TST, WithGlobalNsPrefix);
      if (T == TST)
        return QT;
      return Ctx.getQualifiedType(T, QT.getQualifiers());
    }

    // Local qualifiers are attached to the QualType outside of the
    // elaborated type.  Retrieve them before descending into the
    // elaborated type.
    Qualifiers PrefixQualifiers = QT.getLocalQualifiers();
    QT = QualType(QT.getTypePtr(), 0);

    // Create a nested name specifier if needed.
    NestedNameSpecifier Prefix = createNestedNameSpecifierForScopeOf(
        Ctx, QT.getTypePtr(), true /*FullyQualified*/, WithGlobalNsPrefix);

    // In case of template specializations iterate over the arguments and
    // fully qualify them as well.
    if (const auto *TT = dyn_cast<TagType>(QT.getTypePtr())) {
      // We are asked to fully qualify and we have a Record Type (which
      // may point to a template specialization) or Template
      // Specialization Type. We need to fully qualify their arguments.

      const Type *TypePtr = getFullyQualifiedTemplateType(
          Ctx, TT, TT->getKeyword(), Prefix, WithGlobalNsPrefix);
      QT = QualType(TypePtr, 0);
    } else if (const auto *TT = dyn_cast<TypedefType>(QT.getTypePtr())) {
      QT = Ctx.getTypedefType(
          TT->getKeyword(), Prefix, TT->getDecl(),
          getFullyQualifiedType(TT->desugar(), Ctx, WithGlobalNsPrefix));
    } else if (const auto* UT = dyn_cast<UsingType>(QT.getTypePtr())) {
      QT = Ctx.getUsingType(UT->getKeyword(), Prefix, UT->getDecl(),
                            getFullyQualifiedType(UT->desugar(), Ctx,
                                                  WithGlobalNsPrefix));
    } else {
      assert(!Prefix && "Unhandled type node");
    }
    QT = Ctx.getQualifiedType(QT, PrefixQualifiers);
    return QT;
  }

  std::string getFullyQualifiedName(QualType QT,
                                    const ASTContext &Ctx,
                                    const PrintingPolicy &Policy,
                                    bool WithGlobalNsPrefix) {
    QualType FQQT = getFullyQualifiedType(QT, Ctx, WithGlobalNsPrefix);
    return FQQT.getAsString(Policy);
  }

  NestedNameSpecifier getFullyQualifiedDeclaredContext(const ASTContext &Ctx,
                                                       const Decl *Decl,
                                                       bool WithGlobalNsPrefix) {
    return createNestedNameSpecifierForScopeOf(Ctx, Decl, /*FullyQualified=*/true,
                                               WithGlobalNsPrefix);
  }
} // end namespace TypeName
} // namespace utils
} // namespace cling

namespace cling {
namespace utils {
  static bool GetFullyQualifiedTemplateName(const ASTContext& Ctx,
                                            TemplateName& tname) {
    return TypeName::getFullyQualifiedTemplateName(
        Ctx, tname, /*WithGlobalNsPrefix=*/false);
  }

  static NestedNameSpecifier
  GetFullyQualifiedNameSpecifier(const ASTContext& Ctx,
                                 NestedNameSpecifier scope) {
    return TypeName::getFullyQualifiedNestedNameSpecifier(
        Ctx, scope, /*WithGlobalNsPrefix=*/false);
  }

  NestedNameSpecifier
  TypeName::CreateNestedNameSpecifier(const ASTContext& Ctx,
                                      const NamespaceDecl* Namesp) {
    return TypeName::createNestedNameSpecifier(
        Ctx, Namesp, /*WithGlobalNsPrefix=*/false);
  }

  // FIXME: Doesn't exist upstream
  NestedNameSpecifier TypeName::CreateNestedNameSpecifier(
      const ASTContext& Ctx, const TypedefNameDecl* TD, bool FullyQualify) {
    return TypeName::createNestedNameSpecifier(
        Ctx, TD, FullyQualify, /*WithGlobalNsPrefix=*/false);
  }

  NestedNameSpecifier
  TypeName::CreateNestedNameSpecifier(const ASTContext& Ctx, const TagDecl* TD,
                                      bool FullyQualify) {
    return TypeName::createNestedNameSpecifier(
        Ctx, TD, FullyQualify, /*WithGlobalNsPrefix=*/false);
  }

  QualType TypeName::GetFullyQualifiedType(QualType QT, const ASTContext& Ctx) {
    return TypeName::getFullyQualifiedType(QT, Ctx,
                                                  /*WithGlobalNsPrefix=*/false);
  }

  std::string TypeName::GetFullyQualifiedName(QualType QT,
                                              const ASTContext& Ctx) {
    QualType FQQT =
        TypeName::getFullyQualifiedType(QT, Ctx,
                                               /*WithGlobalNsPrefix=*/false);

    PrintingPolicy Policy(Ctx.getPrintingPolicy());
    Policy.SuppressScope = false;
    Policy.AnonymousTagLocations = false;
    return FQQT.getAsString(Policy);
  }

  // Does not exist upstream. Maybe upstream?
  QualType TypeName::QualifyTypeUnderPrefix(const ASTContext& Ctx, QualType QT,
                                            NestedNameSpecifier prefix,
                                            bool WithGlobalNsPrefix) {
    if (const auto* TT = dyn_cast<TagType>(QT.getTypePtr())) {
      const Type* TypePtr =
          TypeName::getFullyQualifiedTemplateType(Ctx, TT, TT->getKeyword(),
                                                  prefix, WithGlobalNsPrefix);
      QT = QualType(TypePtr, 0);
    } else if (const auto* TT = dyn_cast<TypedefType>(QT.getTypePtr())) {
      QT = Ctx.getTypedefType(TT->getKeyword(), prefix, TT->getDecl(),
                              TypeName::getFullyQualifiedType(
                                  TT->desugar(), Ctx, WithGlobalNsPrefix));
    } else if (const auto* UT = dyn_cast<UsingType>(QT.getTypePtr())) {
      QT =
          Ctx.getUsingType(UT->getKeyword(), prefix, UT->getDecl(),
                           getFullyQualifiedType(UT->desugar(), Ctx,
                                                 WithGlobalNsPrefix));
    }
    return QT;
  }

  static
  QualType GetPartiallyDesugaredTypeImpl(const ASTContext& Ctx,
                                         QualType QT,
                               const Transform::Config& TypeConfig,
                                         bool fullyQualifyType,
                                         bool fullyQualifyTmpltArg);

  static
  NestedNameSpecifier GetPartiallyDesugaredNNS(const ASTContext& Ctx,
                                                NestedNameSpecifier scope,
                                           const Transform::Config& TypeConfig);

  bool Analyze::IsWrapper(const FunctionDecl* ND) {
    if (!ND)
      return false;
    if (!ND->getDeclName().isIdentifier())
      return false;

    return ND->getName().starts_with(Synthesize::UniquePrefix);
  }

  void Analyze::maybeMangleDeclName(const GlobalDecl& GD,
                                    std::string& mangledName) {
    // copied and adapted from CodeGen::CodeGenModule::getMangledName

    NamedDecl* D
      = cast<NamedDecl>(const_cast<Decl*>(GD.getDecl()));
    std::unique_ptr<MangleContext> mangleCtx;
    mangleCtx.reset(D->getASTContext().createMangleContext());
    if (!mangleCtx->shouldMangleDeclName(D)) {
      IdentifierInfo *II = D->getIdentifier();
      assert(II && "Attempt to mangle unnamed decl.");
      mangledName = II->getName().str();
      return;
    }

    llvm::raw_string_ostream RawStr(mangledName);

#if defined(_WIN32)
    // MicrosoftMangle.cpp:954 calls llvm_unreachable when mangling Dtor_Comdat
    if (isa<CXXDestructorDecl>(GD.getDecl()) &&
        GD.getDtorType() == Dtor_Comdat) {
      if (const IdentifierInfo* II = D->getIdentifier())
        RawStr << II->getName();
    } else
#endif
      mangleCtx->mangleName(GD, RawStr);
    RawStr.flush();
  }

  Expr* Analyze::GetOrCreateLastExpr(FunctionDecl* FD,
                                     int* FoundAt /*=0*/,
                                     bool omitDeclStmts /*=true*/,
                                     Sema* S /*=0*/) {
    assert(FD && "We need a function declaration!");
    assert((omitDeclStmts || S)
           && "Sema needs to be set when omitDeclStmts is false");
    if (FoundAt)
      *FoundAt = -1;

    Expr* result = nullptr;
    if (CompoundStmt* CS = dyn_cast<CompoundStmt>(FD->getBody())) {
      ArrayRef<Stmt*> Stmts(CS->body_begin(), CS->size());
      int indexOfLastExpr = Stmts.size();
      while(indexOfLastExpr--) {
        if (!isa<NullStmt>(Stmts[indexOfLastExpr]))
          break;
      }

      if (FoundAt)
        *FoundAt = indexOfLastExpr;

      if (indexOfLastExpr < 0)
        return nullptr;

      if ( (result = dyn_cast<Expr>(Stmts[indexOfLastExpr])) )
        return result;
      if (!omitDeclStmts)
        if (DeclStmt* DS = dyn_cast<DeclStmt>(Stmts[indexOfLastExpr])) {
          std::vector<Stmt*> newBody = Stmts.vec();
          for (DeclStmt::reverse_decl_iterator I = DS->decl_rbegin(),
                 E = DS->decl_rend(); I != E; ++I) {
            if (VarDecl* VD = dyn_cast<VarDecl>(*I)) {
              // Change the void function's return type
              // We can't PushDeclContext, because we don't have scope.
              Sema::ContextRAII pushedDC(*S, FD);

              QualType VDTy = VD->getType().getNonReferenceType();
              // Get the location of the place we will insert.
              SourceLocation Loc
                = newBody[indexOfLastExpr]->getEndLoc().getLocWithOffset(1);
              DeclRefExpr* DRE = S->BuildDeclRefExpr(VD, VDTy,VK_LValue, Loc);
              assert(DRE && "Cannot be null");
              indexOfLastExpr++;
              newBody.insert(newBody.begin() + indexOfLastExpr, DRE);

              // Attach a new body.
              FPOptionsOverride FPFeatures;
              if (CS->hasStoredFPFeatures()) {
                FPFeatures = CS->getStoredFPFeatures();
              }
              auto newCS = CompoundStmt::Create(S->getASTContext(), newBody,
                                                FPFeatures, CS->getLBracLoc(),
                                                CS->getRBracLoc());
              FD->setBody(newCS);
              if (FoundAt)
                *FoundAt = indexOfLastExpr;
              return DRE;
            }
          }
        }

      return result;
    }

    return result;
  }

  const char* const Synthesize::UniquePrefix = "__cling_Un1Qu3";

  IntegerLiteral* Synthesize::IntegerLiteralExpr(ASTContext& C, uintptr_t Ptr) {
    const llvm::APInt Addr(8 * sizeof(void*), Ptr);
    return IntegerLiteral::Create(C, Addr, C.getUIntPtrType(),
                                  SourceLocation());
  }

  Expr* Synthesize::CStyleCastPtrExpr(Sema* S, QualType Ty, uintptr_t Ptr) {
    ASTContext& Ctx = S->getASTContext();
    return CStyleCastPtrExpr(S, Ty, Synthesize::IntegerLiteralExpr(Ctx, Ptr));
  }

  Expr* Synthesize::CStyleCastPtrExpr(Sema* S, QualType Ty, Expr* E) {
    ASTContext& Ctx = S->getASTContext();
    if (!Ty->isPointerType())
      Ty = Ctx.getPointerType(Ty);

    TypeSourceInfo* TSI = Ctx.getTrivialTypeSourceInfo(Ty, SourceLocation());
    Expr* Result
      = S->BuildCStyleCastExpr(SourceLocation(), TSI,SourceLocation(),E).get();
    assert(Result && "Cannot create CStyleCastPtrExpr");
    return Result;
  }

  static
  NestedNameSpecifier SelectPrefix(const ASTContext& Ctx,
                                    const DeclContext *declContext,
                                    NestedNameSpecifier original_prefix,
                             const Transform::Config& TypeConfig) {
    // We have to also desugar the prefix.
    NestedNameSpecifier prefix = std::nullopt;
    if (declContext) {
      // We had a scope prefix as input, let see if it is still
      // the same as the scope of the result and if it is, then
      // we use it.
      if (declContext->isNamespace()) {
        // Deal with namespace.  This is mostly about dealing with
        // namespace aliases (i.e. keeping the one the user used).
        const NamespaceDecl *new_ns =dyn_cast<NamespaceDecl>(declContext);
        if (new_ns) {
          new_ns = new_ns->getCanonicalDecl();
          const NamespaceDecl *old_ns = nullptr;
          if (original_prefix) {
            if (const NamespaceAliasDecl* alias = getNNSNamespaceAlias(original_prefix)) {
              old_ns = alias->getNamespace()->getCanonicalDecl();
            }
          }
          if (old_ns == new_ns) {
            // This is the same namespace, use the original prefix
            // as a starting point.
            prefix = GetFullyQualifiedNameSpecifier(Ctx,original_prefix);
          } else {
            prefix = TypeName::CreateNestedNameSpecifier(Ctx,
                                               dyn_cast<NamespaceDecl>(new_ns));
          }
        }
      } else {
        const CXXRecordDecl* newtype=dyn_cast<CXXRecordDecl>(declContext);
        if (newtype && original_prefix) {
          // Deal with a class
          const Type *oldtype = original_prefix.getAsType();
          if (original_prefix.getKind() == NestedNameSpecifier::Kind::Type &&
              // NOTE: Should we compare the RecordDecl instead?
              oldtype->getAsCXXRecordDecl() == newtype)
          {
            // This is the same type, use the original prefix as a starting
            // point.
            prefix = GetPartiallyDesugaredNNS(Ctx,original_prefix,TypeConfig);
          } else {
            const TagDecl *tdecl = dyn_cast<TagDecl>(declContext);
            if (tdecl) {
              prefix = TypeName::CreateNestedNameSpecifier(Ctx, tdecl,
                                                      false /*FullyQualified*/);
            }
          }
        } else {
          // We should only create the nested name specifier
          // if the outer scope is really a TagDecl.
          // It could also be a CXXMethod for example.
          const TagDecl *tdecl = dyn_cast<TagDecl>(declContext);
          if (tdecl) {
            prefix = TypeName::CreateNestedNameSpecifier(Ctx,tdecl,
                                                      false /*FullyQualified*/);
          }
        }
      }
    } else {
      prefix = GetFullyQualifiedNameSpecifier(Ctx,original_prefix);
    }
    return prefix;
  }

  static
  NestedNameSpecifier GetPartiallyDesugaredNNS(const ASTContext& Ctx,
                                                NestedNameSpecifier scope,
                                          const Transform::Config& TypeConfig) {
    // Desugar the scope qualifier if needed.

    if (scope.getKind() == NestedNameSpecifier::Kind::Type) {

      const Type* scope_type = scope.getAsType();

      // this is not a namespace, so we might need to desugar
      QualType desugared = GetPartiallyDesugaredTypeImpl(Ctx,
                                                         QualType(scope_type,0),
                                                         TypeConfig,
                                                         /*qualifyType=*/false,
                                                      /*qualifyTmpltArg=*/true);

      NestedNameSpecifier outer_scope = getNNSPrefix(scope);
      // LLVM22: ElaboratedType no longer exists.
      {
        Decl* decl = nullptr;
        if (!desugared.isNull()) {
          const Type* desugaredTy = desugared.getTypePtr();
          switch (desugaredTy->getTypeClass()) {
            case Type::Typedef:
              decl = cast<TypedefType>(desugaredTy)->getDecl();
              break;
            case Type::Using:
              decl = cast<UsingType>(desugaredTy)->getDecl();
              break;
            case Type::Record:
            case Type::Enum:
              decl = cast<TagType>(desugaredTy)->getDecl();
              break;
            default: decl = desugared->getAsCXXRecordDecl(); break;
          }
        }
        if (decl) {
          NamedDecl* outer
            = dyn_cast_or_null<NamedDecl>(decl->getDeclContext());
          NamespaceDecl* outer_ns
            = dyn_cast_or_null<NamespaceDecl>(decl->getDeclContext());
          if (outer
              && !(outer_ns && outer_ns->isAnonymousNamespace())
              && outer->getName().size() ) {
            outer_scope = SelectPrefix(Ctx,decl->getDeclContext(),
                                       outer_scope,TypeConfig);
          } else {
            outer_scope = std::nullopt;
          }
        } else if (outer_scope) {
          outer_scope = GetPartiallyDesugaredNNS(Ctx, outer_scope, TypeConfig);
        }
      }

      // LLVM22: In the old API this was:
      // return NestedNameSpecifier::Create(Ctx,outer_scope,
      //                                    false /* template keyword wanted */,
      //                                    desugared.getTypePtr());
      if (outer_scope) {
        desugared = TypeName::QualifyTypeUnderPrefix(Ctx, desugared, outer_scope);
      }
      return NestedNameSpecifier(desugared.getTypePtr());
    } else {
      return  GetFullyQualifiedNameSpecifier(Ctx,scope);
    }
  }

  bool Analyze::IsStdOrCompilerDetails(const NamedDecl &decl)
  {
    // Return true if the TagType is a 'details' of the std implementation
    // or declared within std.
    // Details means (For now) declared in __gnu_cxx or starting with
    // underscore.

    IdentifierInfo *info = decl.getDeclName().getAsIdentifierInfo();
    if (info && info->getNameStart()[0] == '_') {
      // We have a name starting by _, this is reserve for compiler
      // implementation, so let's not desugar to it.
      return true;
    }
    // And let's check if it is in one of the know compiler implementation
    // namespace.
    const NamedDecl *outer =dyn_cast_or_null<NamedDecl>(decl.getDeclContext());
    while (outer && outer->getName().size() ) {
      if (outer->getName().compare("std") == 0 ||
          outer->getName().compare("__gnu_cxx") == 0) {
        return true;
      }
      outer = dyn_cast_or_null<NamedDecl>(outer->getDeclContext());
    }
    return false;
  }

  bool Analyze::IsStdClass(const clang::NamedDecl &cl)
  {
    // Return true if the class or template is declared directly in the
    // std namespace (modulo inline namespace).

    return cl.getDeclContext()->isStdNamespace();
  }

  // See Sema::PushOnScopeChains
  bool Analyze::isOnScopeChains(const NamedDecl* ND, Sema& SemaR) {

    // Named decls without name shouldn't be in. Eg: struct {int a};
    if (!ND->getDeclName())
      return false;

    // Out-of-line definitions shouldn't be pushed into scope in C++.
    // Out-of-line variable and function definitions shouldn't even in C.
    if ((isa<VarDecl>(ND) || isa<FunctionDecl>(ND)) && ND->isOutOfLine() &&
        !ND->getDeclContext()->getRedeclContext()->Equals(
                        ND->getLexicalDeclContext()->getRedeclContext()))
      return false;

    // Template instantiations should also not be pushed into scope.
    if (isa<FunctionDecl>(ND) &&
        cast<FunctionDecl>(ND)->isFunctionTemplateSpecialization())
      return false;

    // Using directives are not registered onto the scope chain
    if (isa<UsingDirectiveDecl>(ND))
      return false;

    IdentifierResolver::iterator
      IDRi = SemaR.IdResolver.begin(ND->getDeclName()),
      IDRiEnd = SemaR.IdResolver.end();

    for (; IDRi != IDRiEnd; ++IDRi) {
      if (ND == *IDRi)
        return true;
    }


    // Check if the declaration is template instantiation, which is not in
    // any DeclContext yet, because it came from
    // Sema::PerformPendingInstantiations
    // if (isa<FunctionDecl>(D) &&
    //     cast<FunctionDecl>(D)->getTemplateInstantiationPattern())
    //   return false;


    return false;
  }

  unsigned int
  Transform::Config::DropDefaultArg(clang::TemplateDecl &Template) const
  {
    /// Return the number of default argument to drop.

    if (Analyze::IsStdClass(Template)) {
      static const char *stls[] =  //container names
        {"vector","list","deque","map","multimap","set","multiset",nullptr};
      static unsigned int values[] =       //number of default arg.
        {1,1,1,2,2,2,2};
      StringRef name = Template.getName();
      for(int k=0;stls[k];k++) {
        if (name == stls[k])
          return values[k];
      }
    }
    // Check in some struct if the Template decl is registered something like
    /*
     DefaultCollection::const_iterator iter;
     iter = m_defaultArgs.find(&Template);
     if (iter != m_defaultArgs.end()) {
        return iter->second;
     }
    */
    return 0;
  }

  static bool ShouldKeepTypedef(const TypedefType* TT,
                                const llvm::SmallSet<const Decl*, 4>& ToSkip)
  {
    // Return true, if we should keep this typedef rather than desugaring it.

    return 0 != ToSkip.count(TT->getDecl()->getCanonicalDecl());
  }

  static bool SingleStepPartiallyDesugarTypeImpl(QualType& QT)
  {
    //  WARNING:
    //
    //  The large blocks of commented-out code in this routine
    //  are there to support doing more desugaring in the future,
    //  we will probably have to.
    //
    //  Do not delete until we are completely sure we will
    //  not be changing this routine again!
    //
    const Type* QTy = QT.getTypePtr();
    Type::TypeClass TC = QTy->getTypeClass();
    switch (TC) {
      //
      //  Unconditionally sugared types.
      //
      case Type::Paren: {
        return false;
        //const ParenType* Ty = llvm::cast<ParenType>(QTy);
        //QT = Ty->desugar();
        //return true;
      }
      case Type::Typedef: {
        const TypedefType* Ty = llvm::cast<TypedefType>(QTy);
        QT = Ty->desugar();
        return true;
      }
      case Type::Using: {
        const UsingType* Ty = llvm::cast<UsingType>(QTy);
        QT = Ty->desugar();
        return true;
      }
      case Type::PredefinedSugar: {
        // e.g. "__size_t", "__signed_size_t", "__ptrdiff_t" are
        // synthetic sugar names with no corresponding declaration, so they
        // cannot legally appear as bare identifiers in generated code and
        // must always be desugared to the underlying builtin type.
        // See: https://github.com/llvm/llvm-project/commit/7c402b8
        const PredefinedSugarType* Ty = llvm::cast<PredefinedSugarType>(QTy);
        QT = Ty->desugar();
        return true;
      }
      case Type::TypeOf: {
        const TypeOfType* Ty = llvm::cast<TypeOfType>(QTy);
        QT = Ty->desugar();
        return true;
      }
      case Type::Attributed: {
        return false;
        //const AttributedType* Ty = llvm::cast<AttributedType>(QTy);
        //QT = Ty->desugar();
        //return true;
      }
      case Type::SubstTemplateTypeParm: {
        const SubstTemplateTypeParmType* Ty =
          llvm::cast<SubstTemplateTypeParmType>(QTy);
        QT = Ty->desugar();
        return true;
      }
      // LLVM22: There is no ElaboratedType anymore.
      case Type::Record: {
        const RecordType* Ty = llvm::cast<RecordType>(QTy);

        // In LLVM20, a type "::Object" was represented as:
        //   ElaboratedType ('::Object')
        //     `- RecordType ('class Object')
        //
        // Desugaring the ElaboratedType would "strip" the syntactic
        // qualifier ("::") and gives just "class Object".
        //
        // In LLVM22, ElaboratedType has been removed and its information
        // is now stored directly on the underlying type (RecordType here):
        //
        //   RecordType ('::Object')
        //     |- NestedNameSpecifier Global  (i.e. "::")
        //     `- CXXRecordDecl 'Object'
        //
        // Now, the previous desugaring step that implicitly removed the
        // global "::" prefix no longer happens.
        // Older behavior produced "Object", but now we get "::Object".
        //
        // To preserve the previous semantics of "partial desugaring", we
        // explicitly detect and remove the unintended global prefix here.
        // We only do this for the global namespace, since other prefixes
        // (like "std::") must be preserved.

        // Detect unwanted global qualifier (::)
        if (const auto prefix = QT->getPrefix()) {
          if (prefix.getKind() == NestedNameSpecifier::Kind::Global) { // global namespace ("::")
            // Rebuild without prefix by going through the declaration
            // Old ElaboratedType::desugar() behavior.
            if (const CXXRecordDecl* RD = Ty->getAsCXXRecordDecl()) {
              QT = RD->getASTContext().getCanonicalTagType(RD);
              return true;
            }
          }
        }

        return false;
      }
      //
      //  Conditionally sugared types.
      //
      case Type::TypeOfExpr: {
        const TypeOfExprType* Ty = llvm::cast<TypeOfExprType>(QTy);
        if (Ty->isSugared()) {
          QT = Ty->desugar();
          return true;
        }
        return false;
      }
      case Type::Decltype: {
        const DecltypeType* Ty = llvm::cast<DecltypeType>(QTy);
        if (Ty->isSugared()) {
          QT = Ty->desugar();
          return true;
        }
        return false;
      }
      case Type::UnaryTransform: {
        const UnaryTransformType* Ty = llvm::cast<UnaryTransformType>(QTy);
        if (Ty->isSugared()) {
          QT = Ty->desugar();
          return true;
        }
        return false;
      }
      case Type::Auto: {
        return false;
        //const AutoType* Ty = llvm::cast<AutoType>(QTy);
        //if (Ty->isSugared()) {
        //  QT = Ty->desugar();
        //  return true;
        //}
        //return false;
      }
      case Type::TemplateSpecialization: {
        //const TemplateSpecializationType* Ty =
        //  llvm::cast<TemplateSpecializationType>(QTy);
        // Too broad, this returns a the target template but with
        // canonical argument types.
        //if (Ty->isTypeAlias()) {
        //  QT = Ty->getAliasedType();
        //  return true;
        //}
        // Too broad, this returns the canonical type
        //if (Ty->isSugared()) {
        //  QT = Ty->desugar();
        //  return true;
        //}
        return false;
      }
      // Not a sugared type.
      default: {
        break;
      }
    }
    return false;
  }

  bool Transform::SingleStepPartiallyDesugarType(QualType &QT,
                                                 const ASTContext &Context) {
    Qualifiers quals = QT.getQualifiers();
    bool desugared = SingleStepPartiallyDesugarTypeImpl( QT );
    if (desugared) {
      // If the types has been desugared it also lost its qualifiers.
      QT = Context.getQualifiedType(QT, quals);
    }
    return desugared;
  }

  static bool GetPartiallyDesugaredTypeImpl(const ASTContext& Ctx,
                                            TemplateArgument &arg,
                                            const Transform::Config& TypeConfig,
                                            bool fullyQualifyTmpltArg) {
    bool changed = false;

    if (arg.getKind() == TemplateArgument::Template) {
      TemplateName tname = arg.getAsTemplate();
      // Desugar before fully qualifying.
      if (std::optional<TemplateName> UnderlyingOrNone = tname.desugar(/*IgnoreDeduced=*/false)) {
        if (*UnderlyingOrNone != tname) {
          tname = *UnderlyingOrNone;
          changed = true;
        }
      }
      changed = GetFullyQualifiedTemplateName(Ctx, tname);
      if (changed) {
        arg = TemplateArgument(tname);
      }
    } else if (arg.getKind() == TemplateArgument::Type) {

      QualType SubTy = arg.getAsType();
      // Check if the type needs more desugaring and recurse.
      if (isa<TypedefType>(SubTy)
          || isa<UsingType>(SubTy)
          || isa<TemplateSpecializationType>(SubTy)
          || SubTy->getPrefix() // LLVM22: was isa<ElaboratedType>
          || fullyQualifyTmpltArg) {
        changed = true;
        QualType PDQT
              = GetPartiallyDesugaredTypeImpl(Ctx, SubTy, TypeConfig,
                    /*fullyQualifyType=*/true,
                    /*fullyQualifyTmpltArg=*/true);
        arg = TemplateArgument(PDQT);
      }
    } else if (arg.getKind() == TemplateArgument::Pack) {
      SmallVector<TemplateArgument, 2> desArgs;
      for (auto I = arg.pack_begin(), E = arg.pack_end(); I != E; ++I) {
        TemplateArgument pack_arg(*I);
        changed = GetPartiallyDesugaredTypeImpl(Ctx,pack_arg,
                                                TypeConfig,
                                                fullyQualifyTmpltArg);
        desArgs.push_back(pack_arg);
      }
      if (changed) {
        // The allocator in ASTContext is mutable ...
        // Keep the argument const to be inline will all the other interfaces
        // like:  NestedNameSpecifier::Create
        ASTContext &mutableCtx( const_cast<ASTContext&>(Ctx) );
        arg = TemplateArgument::CreatePackCopy(mutableCtx, desArgs);
      }
    }
    return changed;
  }

  static const TemplateArgument*
  GetTmpltArgDeepFirstIndexPack(size_t &cur,
                                const TemplateArgument& arg,
                                size_t idx) {
    SmallVector<TemplateArgument, 2> desArgs;
    for (auto I = arg.pack_begin(), E = arg.pack_end();
         cur < idx && I != E; ++cur,++I) {
      if ((*I).getKind() == TemplateArgument::Pack) {
        auto p_arg = GetTmpltArgDeepFirstIndexPack(cur,(*I),idx);
        if (cur == idx) return p_arg;
      } else if (cur == idx) {
        return I;
      }
    }
    return nullptr;
  }

  // Return the template argument corresponding to the index (idx)
  // when the composite list of arguement is seen flattened out deep
  // first (where depth is provided by template argument packs)
  static const TemplateArgument*
  GetTmpltArgDeepFirstIndex(const TemplateArgumentList& templateArgs,
                            size_t idx) {

    for (size_t cur = 0, I = 0, E = templateArgs.size();
         cur <= idx && I < E; ++I, ++cur) {
      auto &arg = templateArgs[I];
      if (arg.getKind() == TemplateArgument::Pack) {
        // Need to recurse.
        auto p_arg = GetTmpltArgDeepFirstIndexPack(cur,arg,idx);
        if (cur == idx) return p_arg;
     } else if (cur == idx) {
        return &arg;
      }
    }
    return nullptr;
  }

  static QualType GetPartiallyDesugaredTypeImpl(const ASTContext& Ctx,
    QualType QT, const Transform::Config& TypeConfig,
    bool fullyQualifyType, bool fullyQualifyTmpltArg)
  {
    if (QT.isNull())
      return QT;
    // If there are no constraints, then use the standard desugaring.
    if (TypeConfig.empty() && !fullyQualifyType && !fullyQualifyTmpltArg)
      return QT.getDesugaredType(Ctx);

    // Use the underlying deduced type for AutoType
    if (const auto* AT = dyn_cast<AutoType>(QT.getTypePtr())) {
      if (AT->isDeduced()) {
        // Get the qualifiers.
        Qualifiers Quals = QT.getQualifiers();
        QT = AT->getDeducedType();
        // Add back the qualifiers.
        QT = Ctx.getQualifiedType(QT, Quals);
      }
    }

    // In case of Int_t* we need to strip the pointer first, desugar and attach
    // the pointer once again.
    if (isa<PointerType>(QT.getTypePtr())) {
      // Get the qualifiers.
      Qualifiers quals = QT.getQualifiers();
      QualType nQT;
      nQT = GetPartiallyDesugaredTypeImpl(Ctx, QT->getPointeeType(), TypeConfig,
                                          fullyQualifyType,fullyQualifyTmpltArg);
      if (nQT == QT->getPointeeType()) return QT;

      QT = Ctx.getPointerType(nQT);
      // Add back the qualifiers.
      QT = Ctx.getQualifiedType(QT, quals);
      return QT;
    }

    while (isa<SubstTemplateTypeParmType>(QT.getTypePtr())) {
      // Get the qualifiers.
      Qualifiers quals = QT.getQualifiers();

      QT = dyn_cast<SubstTemplateTypeParmType>(QT.getTypePtr())->desugar();

      // Add back the qualifiers.
      QT = Ctx.getQualifiedType(QT, quals);
    }

    // In case of Int_t& we need to strip the pointer first, desugar and attach
    // the reference once again.
    if (isa<ReferenceType>(QT.getTypePtr())) {
      // Get the qualifiers.
      bool isLValueRefTy = isa<LValueReferenceType>(QT.getTypePtr());
      Qualifiers quals = QT.getQualifiers();
      QualType nQT;
      nQT = GetPartiallyDesugaredTypeImpl(Ctx, QT->getPointeeType(), TypeConfig,
                                         fullyQualifyType,fullyQualifyTmpltArg);
      if (nQT == QT->getPointeeType()) return QT;

      // Add the r- or l-value reference type back to the desugared one.
      if (isLValueRefTy)
        QT = Ctx.getLValueReferenceType(nQT);
      else
        QT = Ctx.getRValueReferenceType(nQT);
      // Add back the qualifiers.
      QT = Ctx.getQualifiedType(QT, quals);
      return QT;
    }

    // In case of Int_t[2] we need to strip the array first, desugar and attach
    // the array once again.
    if (isa<ArrayType>(QT.getTypePtr())) {
      // Get the qualifiers.
      Qualifiers quals = QT.getQualifiers();

      if (isa<ConstantArrayType>(QT.getTypePtr())) {
        const ConstantArrayType *arr
          = dyn_cast<ConstantArrayType>(QT.getTypePtr());
        QualType newQT
           = GetPartiallyDesugaredTypeImpl(Ctx,arr->getElementType(), TypeConfig,
                                         fullyQualifyType,fullyQualifyTmpltArg);
        if (newQT == arr->getElementType()) return QT;
        QT = Ctx.getConstantArrayType (newQT,
                                       arr->getSize(),
                                       arr->getSizeExpr(),
                                       arr->getSizeModifier(),
                                       arr->getIndexTypeCVRQualifiers());

      } else if (isa<DependentSizedArrayType>(QT.getTypePtr())) {
        const DependentSizedArrayType *arr
          = dyn_cast<DependentSizedArrayType>(QT.getTypePtr());
        QualType newQT
          = GetPartiallyDesugaredTypeImpl(Ctx,arr->getElementType(), TypeConfig,
                                          fullyQualifyType,fullyQualifyTmpltArg);
        if (newQT == QT) return QT;
        QT = Ctx.getDependentSizedArrayType (newQT,
                                            arr->getSizeExpr(),
                                            arr->getSizeModifier(),
                                            arr->getIndexTypeCVRQualifiers());

      } else if (isa<IncompleteArrayType>(QT.getTypePtr())) {
        const IncompleteArrayType *arr
          = dyn_cast<IncompleteArrayType>(QT.getTypePtr());
        QualType newQT
          = GetPartiallyDesugaredTypeImpl(Ctx,arr->getElementType(), TypeConfig,
                                          fullyQualifyType,fullyQualifyTmpltArg);
        if (newQT == arr->getElementType()) return QT;
        QT = Ctx.getIncompleteArrayType (newQT,
                                         arr->getSizeModifier(),
                                         arr->getIndexTypeCVRQualifiers());

      } else if (isa<VariableArrayType>(QT.getTypePtr())) {
        const VariableArrayType *arr
          = dyn_cast<VariableArrayType>(QT.getTypePtr());
        QualType newQT
          = GetPartiallyDesugaredTypeImpl(Ctx,arr->getElementType(), TypeConfig,
                                          fullyQualifyType,fullyQualifyTmpltArg);
        if (newQT == arr->getElementType()) return QT;
        QT = Ctx.getVariableArrayType (newQT,
                                       arr->getSizeExpr(),
                                       arr->getSizeModifier(),
                                       arr->getIndexTypeCVRQualifiers());
      }

      // Add back the qualifiers.
      QT = Ctx.getQualifiedType(QT, quals);
      return QT;
    }

    // If the type is elaborated, first remove the prefix and then
    // when we are done we will as needed add back the (new) prefix.
    // for example for std::vector<int>::iterator, we work on
    // just 'iterator' (which remember which scope its from)
    // and remove the typedef to get (for example),
    //   __gnu_cxx::__normal_iterator
    // which is *not* in the std::vector<int> scope and it is
    // the __gnu__cxx part we should use as the prefix.
    // NOTE: however we problably want to add the std::vector typedefs
    // to the list of things to skip!

    NestedNameSpecifier original_prefix = std::nullopt;
    Qualifiers prefix_qualifiers;
    NestedNameSpecifier embedded_prefix = QT->getPrefix();

    if (embedded_prefix) {
      // Intentionally, we do not care about the other compononent of
      // the elaborated type (the keyword) as part of the partial
      // desugaring (and/or name normaliztation) is to remove it.

        const NamespaceDecl* ns = getNNSNamespace(embedded_prefix);
        if (!(ns && ns->isAnonymousNamespace())) {
          // We have to also desugar the prefix unless
          // it does not have a name (anonymous namespaces).
          fullyQualifyType = true;
          prefix_qualifiers = QT.getLocalQualifiers();
          original_prefix = embedded_prefix;
          // TODO: LLVM22: In the old API this was:
          // QT = QualType(etype_input->getNamedType().getTypePtr(), 0);

          // Strip the prefix from QT by rebuilding the TemplateSpecializationType
          // with an unqualified template name, so the desugaring can resolve
          // e.g. ROOT::RVec (UsingShadowDecl) -> ROOT::VecOps::RVec
          if (const TemplateSpecializationType* TST = QT->getAs<TemplateSpecializationType>()) {
            TemplateName TName = TST->getTemplateName();
            if (TemplateDecl* TD = TName.getAsTemplateDecl()) {
                // Rebuild TemplateName WITHOUT prefix
                TemplateName NewName(TD);
                // Recreate the type
                QT = Ctx.getTemplateSpecializationType(
                    ElaboratedTypeKeyword::None,
                    NewName,
                    TST->template_arguments(),
                    /*CanonicalArgs=*/{},
                    TST->getCanonicalTypeInternal());
            }
          }
        }
    }

    // Desugar QT until we cannot desugar any more, or
    // we hit one of the special typedefs.
    while (1) {
      if (const TypedefType* TT = llvm::dyn_cast<TypedefType>(QT.getTypePtr())){
        if (ShouldKeepTypedef(TT, TypeConfig.m_toSkip)) {
          if (!fullyQualifyType && !fullyQualifyTmpltArg) {
            return QT;
          }
          // We might have stripped the namespace/scope part,
          // so we must go on to add it back.
          break;
        }
      }
      bool wasDesugared = Transform::SingleStepPartiallyDesugarType(QT,Ctx);

      // Did we get to a basic_string, let's get back to std::string
      Transform::Config::ReplaceCollection::const_iterator
      iter = TypeConfig.m_toReplace.find(QT->getCanonicalTypeInternal().getTypePtr());
      if (iter != TypeConfig.m_toReplace.end()) {
        Qualifiers quals = QT.getQualifiers();
        QT = QualType( iter->second, 0);
        QT = Ctx.getQualifiedType(QT,quals);
        break;
      }
      if (!wasDesugared) {
        // No more work to do, stop now.
        break;
      }
    }

    // If we have a reference, array or pointer we still need to
    // desugar what they point to.
    if (isa<PointerType>(QT.getTypePtr()) ||
        isa<ReferenceType>(QT.getTypePtr()) ||
        isa<ArrayType>(QT.getTypePtr())) {
      return GetPartiallyDesugaredTypeImpl(Ctx, QT, TypeConfig,
                                           fullyQualifyType,
                                           fullyQualifyTmpltArg);
    }

    NestedNameSpecifier prefix = std::nullopt;
    // LLVM22: ElaboratedType no longer exists.
    if (fullyQualifyType) {
      // Let's check whether this type should have been an elaborated type.
      // in which case we want to add it ... but we can't really preserve
      // the typedef in this case ...

      Decl* decl = nullptr;
      if (!QT.isNull()) {
        const Type* QTTy = QT.getTypePtr();
        switch (QTTy->getTypeClass()) {
          case Type::Typedef: decl = cast<TypedefType>(QTTy)->getDecl(); break;
          case Type::Using: decl = cast<UsingType>(QTTy)->getDecl(); break;
          case Type::Record:
          case Type::Enum: decl = cast<TagType>(QTTy)->getDecl(); break;
          default: decl = QT->getAsCXXRecordDecl(); break;
        }
      }
      if (decl) {
        NamedDecl* outer
           = dyn_cast_or_null<NamedDecl>(decl->getDeclContext());
        NamespaceDecl* outer_ns
           = dyn_cast_or_null<NamespaceDecl>(decl->getDeclContext());
        if (outer
            && !(outer_ns && outer_ns->isAnonymousNamespace())
            && !outer->getNameAsString().empty() ) {
          if (original_prefix) {
            if (original_prefix.getKind() == NestedNameSpecifier::Kind::Type) {
              const Type *oldtype = original_prefix.getAsType();
              if (oldtype->getAsCXXRecordDecl() == outer) {
                // Same type, use the original spelling
                prefix
                  = GetPartiallyDesugaredNNS(Ctx, original_prefix, TypeConfig);
                outer = nullptr; // Cancel the later creation.
              }
            } else {
              const NamespaceDecl *old_ns = getNNSNamespace(original_prefix);
              if (old_ns) {
                old_ns = old_ns->getCanonicalDecl();
              }
              else if (const NamespaceAliasDecl *alias =
                       getNNSNamespaceAlias(original_prefix))
              {
                old_ns = alias->getNamespace()->getCanonicalDecl();
              }
              const NamespaceDecl *new_ns = dyn_cast<NamespaceDecl>(outer);
              if (new_ns) new_ns = new_ns->getCanonicalDecl();
              if (old_ns == new_ns) {
                // This is the same namespace, use the original prefix
                // as a starting point.
                prefix = GetFullyQualifiedNameSpecifier(Ctx,original_prefix);
                outer = nullptr; // Cancel the later creation.
              }
            }
          } else { // if (!original_prefix)
            // move qualifiers on the outer type (avoid 'std::const string'!)
            prefix_qualifiers = QT.getLocalQualifiers();
            QT = QualType(QT.getTypePtr(),0);
          }
          if (outer) {
            if (decl->getDeclContext()->isNamespace()) {
              prefix = TypeName::CreateNestedNameSpecifier(Ctx,
                                                dyn_cast<NamespaceDecl>(outer));
            } else {
              // We should only create the nested name specifier
              // if the outer scope is really a TagDecl.
              // It could also be a CXXMethod for example.
              TagDecl *tdecl = dyn_cast<TagDecl>(outer);
              if (tdecl) {
                prefix = TypeName::CreateNestedNameSpecifier(Ctx,tdecl,
                                                      false /*FullyQualified*/);
                prefix = GetPartiallyDesugaredNNS(Ctx,prefix,TypeConfig);
              }
            }
          }
        }
      }
    }

    // In case of template specializations iterate over the arguments and
    // desugar them as well.
    if (const TemplateSpecializationType* TST
       = dyn_cast<const TemplateSpecializationType>(QT.getTypePtr())) {

      if (TST->isTypeAlias()) {
        QualType targetType = TST->getAliasedType();
        /*
        // We really need to find a way to propagate/keep the opaque typedef
        // that are available in TST to the aliased type.  We would need
        // to do something like:

        QualType targetType = TST->getAliasedType();
        QualType resubst = ReSubstTemplateArg(targetType,TST);
        return GetPartiallyDesugaredTypeImpl(Ctx, resubst, TypeConfig,
                                             fullyQualifyType,
                                             fullyQualifyTmpltArg);

        // But this is not quite right (ReSubstTemplateArg is from TMetaUtils)
        // as it does not resubst for

          template <typename T> using myvector = std::vector<T>;
          myvector<Double32_t> vd32d;

        // and does not work at all for

          template<class T> using ptr = T*;
          ptr<Double32_t> p2;

        // as the target is not a template.
        */
        // So for now just return move on with the least lose we can do
        return GetPartiallyDesugaredTypeImpl(Ctx, targetType, TypeConfig,
                                           fullyQualifyType,
                                           fullyQualifyTmpltArg);
      }

      bool mightHaveChanged = false;
      llvm::SmallVector<TemplateArgument, 4> desArgs;
      llvm::ArrayRef<clang::TemplateArgument> template_arguments =
          TST->template_arguments();
      unsigned int argi = 0;
      for (const clang::TemplateArgument *I = template_arguments.begin(),
                                         *E = template_arguments.end();
           I != E; ++I, ++argi) {
        if (I->getKind() == TemplateArgument::Expression) {
          // If we have an expression, we need to replace it / desugar it
          // as it could contain unqualifed (or partially qualified or
          // private) parts.

          QualType canon = QT->getCanonicalTypeInternal();
          const RecordType *TSTRecord
            = dyn_cast<const RecordType>(canon.getTypePtr());
          if (TSTRecord) {
            if (const ClassTemplateSpecializationDecl* TSTdecl =
               dyn_cast<ClassTemplateSpecializationDecl>(TSTRecord->getDecl()))
            {
              const TemplateArgumentList& templateArgs
                = TSTdecl->getTemplateArgs();

              mightHaveChanged = true;
              const TemplateArgument *match
                  = GetTmpltArgDeepFirstIndex(templateArgs,argi);
              if (match) desArgs.push_back(*match);
              continue;
            }
          }
        }

        if (I->getKind() == TemplateArgument::Template) {
          TemplateName tname = I->getAsTemplate();
          bool changed = false;

          // Sometimes tname can still be sugared and can cause lookup failures.
          // The following example fixed:
          // __common_pool_policy<__pool,true> -> __common_pool_policy<__gnu_cxx::__pool, true>
          if (std::optional<TemplateName> UnderlyingOrNone = tname.desugar(/*IgnoreDeduced=*/false)) {
            if (*UnderlyingOrNone != tname) {
              tname = *UnderlyingOrNone;
              changed = true;
            }
          }
          changed |= GetFullyQualifiedTemplateName(Ctx, tname);
          if (changed) {
            desArgs.push_back(TemplateArgument(tname));
            mightHaveChanged = true;
          } else
            desArgs.push_back(*I);
          continue;
        }

        if (I->getKind() != TemplateArgument::Type) {
          desArgs.push_back(*I);
          continue;
        }

        QualType SubTy = I->getAsType();
        // Check if the type needs more desugaring and recurse.
        if (isa<TypedefType>(SubTy)
            || isa<UsingType>(SubTy)
            || isa<TemplateSpecializationType>(SubTy)
            || SubTy->getPrefix() // LLVM22: was isa<ElaboratedType>
            || fullyQualifyTmpltArg) {
          QualType PDQT
            = GetPartiallyDesugaredTypeImpl(Ctx, SubTy, TypeConfig,
                                            fullyQualifyType,
                                            fullyQualifyTmpltArg);
          mightHaveChanged |= (SubTy != PDQT);
          desArgs.push_back(TemplateArgument(PDQT));
        } else {
          desArgs.push_back(*I);
        }
      }

      // If desugaring happened allocate new type in the AST.
      if (mightHaveChanged) {
        Qualifiers qualifiers = QT.getLocalQualifiers();
        QT = Ctx.getTemplateSpecializationType(TST->getKeyword(),
                                               TST->getTemplateName(),
                                               desArgs,
                                               /*CanonicalArgs=*/{},
                                               TST->getCanonicalTypeInternal());
        QT = Ctx.getQualifiedType(QT, qualifiers);
      }

      // Attach the prefix to the TST now, before the generic prefix block
      // below. The generic block's QualifyTypeUnderPrefix does not handle TSTs,
      // so we must do it here while we still know QT is a TST.
      if (prefix) {
        if (const auto* NewTST =
                dyn_cast<TemplateSpecializationType>(QT.getTypePtr())) {
          const Type* TypePtr = TypeName::getFullyQualifiedTemplateType(
              Ctx, NewTST, /*WithGlobalNsPrefix=*/false);
          QT = Ctx.getQualifiedType(QualType(TypePtr, 0), prefix_qualifiers);
          return QT;
        }
      }
    } else if (fullyQualifyTmpltArg) {

      if (const RecordType *TSTRecord
          = dyn_cast<const RecordType>(QT.getTypePtr())) {
        // We are asked to fully qualify and we have a Record Type,
        // which can point to a template instantiation with no sugar in any of
        // its template argument, however we still need to fully qualify them.

        if (const ClassTemplateSpecializationDecl* TSTdecl =
            dyn_cast<ClassTemplateSpecializationDecl>(TSTRecord->getDecl()))
        {
          const TemplateArgumentList& templateArgs
            = TSTdecl->getTemplateArgs();

          bool mightHaveChanged = false;
          llvm::SmallVector<TemplateArgument, 4> desArgs;
          for(unsigned int I = 0, E = templateArgs.size();
              I != E; ++I) {

#if 1

            // cheap to copy and potentially modified by
            // GetPartiallyDesugaredTypeImpl
            TemplateArgument arg(templateArgs[I]);
            mightHaveChanged |= GetPartiallyDesugaredTypeImpl(Ctx,arg,
                                                              TypeConfig,
                                                          fullyQualifyTmpltArg);
            desArgs.push_back(arg);
#else
            if (templateArgs[I].getKind() == TemplateArgument::Template) {
               TemplateName tname = templateArgs[I].getAsTemplate();
               // Note: should we not also desugar?
               bool changed = GetFullyQualifiedTemplateName(Ctx, tname);
               if (changed) {
                  desArgs.push_back(TemplateArgument(tname));
                  mightHaveChanged = true;
               } else
                  desArgs.push_back(templateArgs[I]);
               continue;
            }

            if (templateArgs[I].getKind() != TemplateArgument::Type) {
              desArgs.push_back(templateArgs[I]);
              continue;
            }

            QualType SubTy = templateArgs[I].getAsType();
            // Check if the type needs more desugaring and recurse.
            if (isa<TypedefType>(SubTy)
                || isa<UsingType>(SubTy)
                || isa<TemplateSpecializationType>(SubTy)
                || SubTy->getPrefix() // LLVM22: was isa<ElaboratedType>
                || fullyQualifyTmpltArg) {
              mightHaveChanged = true;
              QualType PDQT
                = GetPartiallyDesugaredTypeImpl(Ctx, SubTy, TypeConfig,
                                                /*fullyQualifyType=*/true,
                                                /*fullyQualifyTmpltArg=*/true);
              desArgs.push_back(TemplateArgument(PDQT));
            } else {
              desArgs.push_back(templateArgs[I]);
            }
#endif
          }

          // If desugaring happened allocate new type in the AST.
          if (mightHaveChanged) {
            Qualifiers qualifiers = QT.getLocalQualifiers();
            TemplateName TN(TSTdecl->getSpecializedTemplate());
            QT = Ctx.getTemplateSpecializationType(ElaboratedTypeKeyword::None, TN, desArgs, /*CanonicalArgs=*/{},
                                         TSTRecord->getCanonicalTypeInternal());
            QT = Ctx.getQualifiedType(QT, qualifiers);
          }
        }
      }
    }
    // TODO: Find a way to avoid creating new types, if the input is already
    // fully qualified.
    if (prefix) {
      QT = TypeName::QualifyTypeUnderPrefix(Ctx, QT, prefix);
      QT = Ctx.getQualifiedType(QT, prefix_qualifiers);
    } else if (original_prefix) {
      QT = Ctx.getQualifiedType(QT, prefix_qualifiers);
    }
    return QT;
  }

  QualType Transform::GetPartiallyDesugaredType(const ASTContext& Ctx,
    QualType QT, const Transform::Config& TypeConfig,
    bool fullyQualify/*=true*/)
  {
    return GetPartiallyDesugaredTypeImpl(Ctx,QT,TypeConfig,
                                         /*qualifyType*/fullyQualify,
                                         /*qualifyTmpltArg*/fullyQualify);
  }

  NamespaceDecl* Lookup::Namespace(Sema* S, const char* Name,
                                   const DeclContext* Within) {
    DeclarationName DName = &S->Context.Idents.get(Name);
    LookupResult R(*S, DName, SourceLocation(),
                   Sema::LookupNestedNameSpecifierName);
    R.suppressDiagnostics();
    if (!Within)
      S->LookupName(R, S->TUScope);
    else {
      if (const clang::TagDecl* TD = dyn_cast<clang::TagDecl>(Within)) {
        if (!TD->getDefinition()) {
          // No definition, no lookup result.
          return nullptr;
        }
      }
      S->LookupQualifiedName(R, const_cast<DeclContext*>(Within));
    }

    if (R.empty())
      return nullptr;

    R.resolveKind();

    return dyn_cast<NamespaceDecl>(R.getFoundDecl());
  }

  NamedDecl* Lookup::Named(Sema* S, llvm::StringRef Name,
                           const DeclContext* Within) {
    DeclarationName DName = &S->Context.Idents.get(Name);
    return Lookup::Named(S, DName, Within);
  }

  NamedDecl* Lookup::Named(Sema* S, const char* Name,
                           const DeclContext* Within) {
    return Lookup::Named(S, llvm::StringRef(Name), Within);
  }

  NamedDecl* Lookup::Named(Sema* S, const clang::DeclarationName& Name,
                           const DeclContext* Within) {
    LookupResult R(*S, Name, SourceLocation(), Sema::LookupOrdinaryName,
                   RedeclarationKind::ForVisibleRedeclaration);
    Lookup::Named(S, R, Within);
    return LookupResult2Decl<clang::NamedDecl>(R);
  }

  TagDecl* Lookup::Tag(Sema* S, llvm::StringRef Name,
                       const DeclContext* Within) {
    DeclarationName DName = &S->Context.Idents.get(Name);
    return Lookup::Tag(S, DName, Within);
  }

  TagDecl* Lookup::Tag(Sema* S, const char* Name,
                       const DeclContext* Within) {
    return Lookup::Tag(S, llvm::StringRef(Name), Within);
  }

  TagDecl* Lookup::Tag(Sema* S, const clang::DeclarationName& Name,
                       const DeclContext* Within) {
    LookupResult R(*S, Name, SourceLocation(), Sema::LookupTagName,
                   RedeclarationKind::ForVisibleRedeclaration);
    Lookup::Named(S, R, Within);
    return LookupResult2Decl<clang::TagDecl>(R);
  }

  void Lookup::Named(Sema* S, LookupResult& R, const DeclContext* Within) {
    R.suppressDiagnostics();
    if (!Within)
      S->LookupName(R, S->TUScope);
    else {
      const DeclContext* primaryWithin = nullptr;
      if (const clang::TagDecl *TD = dyn_cast<clang::TagDecl>(Within)) {
        primaryWithin = dyn_cast_or_null<DeclContext>(TD->getDefinition());
      } else {
        primaryWithin = Within->getPrimaryContext();
      }
      if (!primaryWithin) {
        // No definition, no lookup result.
        return;
      }
      bool res =
          S->LookupQualifiedName(R, const_cast<DeclContext*>(primaryWithin));

      // If the lookup fails and the context is a namespace, try to lookup in
      // the namespaces by setting NotForRedeclaration.
      if (!res && primaryWithin->isNamespace()) {
        R.setRedeclarationKind(RedeclarationKind::NotForRedeclaration);
        S->LookupQualifiedName(R, const_cast<DeclContext*>(primaryWithin));
      }
    }
  }

} // end namespace utils
} // end namespace cling
