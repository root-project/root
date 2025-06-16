//===- Mips.cpp -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ABIInfoImpl.h"
#include "TargetInfo.h"

using namespace clang;
using namespace clang::CodeGen;

//===----------------------------------------------------------------------===//
// Elbrus ABI Implementation.
//===----------------------------------------------------------------------===//

static bool
isRecordReturnIndirect( QualType T, CodeGen::CodeGenTypes &CGT)
{
  const CXXRecordDecl *RD = T->getAsCXXRecordDecl();

  if ( !RD )
    return false;

  // Return indirectly if we have a non-trivial copy ctor or non-trivial dtor.
  // FIXME: Use canCopyArgument() when it is fixed to handle lazily declared
  // special members.
  if ( (RD->hasNonTrivialDestructor()
        || RD->hasNonTrivialCopyConstructor())
       && !RD->canPassInRegisters() )
    return true;

  return false;
}

namespace {
class ElbrusABIInfo : public DefaultABIInfo {
public:
  ElbrusABIInfo(CodeGenTypes &CGT) : DefaultABIInfo(CGT) {}

  //ABIArgInfo classifyReturnType(QualType RetTy) const;
  ABIArgInfo classifyArgumentType(QualType RetTy) const;

  void computeInfo(CGFunctionInfo &FI) const override {
    CGFunctionInfo::arg_iterator it, ie;
    QualType rtype = FI.getReturnType();

    DefaultABIInfo::computeInfo( FI);

    FI.getReturnInfo() = classifyReturnType( rtype);
    if ( !rtype->isVoidType() )
      FI.getReturnInfo() = ABIArgInfo::getDirect();

    if ( isAggregateTypeForABI( rtype) )
      if ( isRecordReturnIndirect( rtype, CGT) )
        FI.getReturnInfo() = ABIInfo::getNaturalAlignIndirect( rtype);

    for ( it = FI.arg_begin(), ie = FI.arg_end(); it != ie; ++it )
      if ( isAggregateTypeForABI( it->type)
           && !isRecordReturnIndirect( it->type, CGT) ) {
        it->info = ABIArgInfo::getDirect( CGT.ConvertType( it->type));
        it->info.setCoerceToType( CGT.ConvertType( it->type));
      }
  }

  Address EmitVAArg(CodeGenFunction &CGF, Address VAListAddr,
                    QualType Ty) const override {
    return EmitVAArgInstr(CGF, VAListAddr, Ty, classifyArgumentType(Ty));
  }
};

class ElbrusTargetCodeGenInfo : public TargetCodeGenInfo {
public:
  ElbrusTargetCodeGenInfo( CodeGenTypes &CGT)
      : TargetCodeGenInfo( std::make_unique<ElbrusABIInfo>( CGT)) {}
};

ABIArgInfo ElbrusABIInfo::classifyArgumentType( QualType Ty) const {
    ABIArgInfo r;

    Ty = useFirstFieldIfTransparentUnion(Ty);

    if ( isAggregateTypeForABI( Ty) )
    {
        // Records with non-trivial destructors/copy-constructors should not be
        // passed by value.
        if (CGCXXABI::RecordArgABI RAA = getRecordArgABI(Ty, getCXXABI()))
            return getNaturalAlignIndirect(Ty, RAA == CGCXXABI::RAA_DirectInMemory);

        return ABIArgInfo::getDirect();
    }

    // Treat an enum type as its underlying type.
    if (const EnumType *EnumTy = Ty->getAs<EnumType>())
        Ty = EnumTy->getDecl()->getIntegerType();

    if ( isPromotableIntegerTypeForABI( Ty) ) {
       r = ABIArgInfo::getExtend(Ty);
    } else {
       r = ABIArgInfo::getDirect();
    }

    return (r);
}

}

std::unique_ptr<TargetCodeGenInfo>
CodeGen::createElbrusTargetCodeGenInfo(CodeGenModule &CGM) {
  return std::make_unique<ElbrusTargetCodeGenInfo>(CGM.getTypes());
}
