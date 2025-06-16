//===--- Elbrus.h - Declare Elbrus target feature support -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares Elbrus TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_BASIC_TARGETS_ELBRUS_H
#define LLVM_CLANG_LIB_BASIC_TARGETS_ELBRUS_H

#include "OSTargets.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Compiler.h"
#include "llvm/TargetParser/ElbrusTargetParser.h"

namespace clang {
namespace targets {

// Elbrus abstract base class
class LLVM_LIBRARY_VISIBILITY ElbrusTargetInfo : public TargetInfo
{
    static const Builtin::Info BuiltinInfo[];
    static const char * const GCCRegNames[];
    static const TargetInfo::GCCRegAlias GCCRegAliases[];
    std::string CPU;
    bool HasMMX = true;
    bool HasPOPCNT = true;
    bool HasSSE = true;
    bool HasSSE2 = true;
    bool HasSSE3 = true;
    bool HasSSSE3 = true;
    bool HasSSE4_1 = true;
    bool HasSSE4_2 = true;
    bool HasAVX = true;
    bool HasAVX2 = true;
    bool HasSSE4A = true;
    bool HasFMA4 = true;
    bool HasXOP = true;
    bool HasFMA = true;
    bool HasBMI = true;
    bool HasBMI2 = true;
    bool HasAES = true;
    bool HasPCLMUL = true;
    bool Has3DNOW = true;
    bool Has3DNOWA = true;
    bool HasCLFLUSHOPT = true;
    bool HasCLWB = true;
    bool HasCLZERO = true;
    bool HasF16C = true;
    bool HasLZCNT = true;
    bool HasMWAITX = true;
    bool HasRDRND = true;
    bool HasRDSEED = true;
    bool HasSHA2 = true;
    bool HasABM = true;
    bool HasTBM = true;
    bool HasAVXVNNI = true;

  public:
    ElbrusTargetInfo( const llvm::Triple &triple, const TargetOptions &) : TargetInfo(triple)
    {
#if 1
        //LongDoubleWidth = 80;
        LongDoubleWidth = 128;
        LongDoubleAlign = 128;
        LongDoubleFormat = &llvm::APFloatBase::x87DoubleExtended();
#else
        LongDoubleWidth = 64;
        LongDoubleAlign = 64;
#endif
        HasFloat128 = true;
    }

    /// \brief Flags for architecture specific defines.
    typedef enum
    {
        ArchDefineNone  = 0,
        ArchDefineName  = 1 << 0, // <name> is substituted for arch name.
        ArchDefine2c_p  = ArchDefineName << 1,
        ArchDefine4c    = ArchDefine2c_p << 1,
        ArchDefine8c    = ArchDefine4c   << 1,
        ArchDefine1c_p  = ArchDefine8c   << 1,
        ArchDefine8c2   = ArchDefine1c_p << 1,
        ArchDefine12c   = ArchDefine8c2  << 1,
        ArchDefine16c   = ArchDefine12c  << 1,
        ArchDefine2c3   = ArchDefine16c  << 1,
    } ArchDefineTypes;

#if 0
    // Note: GCC recognizes the following additional cpus:
    //  elbrus-2c+, elbrus-4c, elbrus-8c
    virtual bool setCPU(const std::string &Name)
    {
        bool CPUKnown = llvm::StringSwitch<bool>(Name)
            .Case("generic", true)
            .Case("elbrus-2c+", true)
            .Case("elbrus-4c", true)
            .Case("elbrus-8c", true)
            .Default(false);

        bool CPUKnown = llvm::StringSwitch<bool>(Name)
            .Case("generic", true)
            .Case("elbrus-2c+", true)
            .Case("elbrus-4c", true)
            .Case("elbrus-8c", true)
            .Default(false);

        if ( CPUKnown )
        {
            CPU = Name;
        }

        return CPUKnown;
    }
#endif

    // Note: GCC recognizes the following additional cpus:
    //  elbrus-2c+, elbrus-4c, elbrus-8c
    virtual bool setCPU(const std::string &Name)
    {
        bool CPUKnown = llvm::StringSwitch<bool>(Name)
            .Case("native", true)
            .Case("elbrus-v2", true)  // v2
            .Case("elbrus-2c+", true)
            .Case("elbrus-v3", true)  // v3
            .Case("elbrus-4c", true)
            .Case("elbrus-v4", true)  // v4
            .Case("elbrus-8c", true)
            .Case("elbrus-1c+", true)
            .Case("elbrus-v5", true)  // v5
            .Case("elbrus-8c2", true)
            .Case("elbrus-v6", true)  // v6
            .Case("elbrus-16c", true)
            .Case("elbrus-2c3", true)
            .Case("elbrus-12c", true)
            .Case("elbrus-v7", true)  // v7
            .Case("elbrus-48c", true)
            .Case("elbrus-8v7", true)
            .Default(false);

        if ( CPUKnown ) {
            CPU = Name;
        }

        return CPUKnown;
    }

    virtual ArrayRef<Builtin::Info> getTargetBuiltins() const;

    ArrayRef<const char *> getGCCRegNames() const override;
    ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override;

    virtual bool isCLZForZeroUndef() const { return false; }
    virtual void getTargetDefines( const LangOptions &Opts, MacroBuilder &Builder) const;
    virtual void getDefaultFeatures( llvm::StringMap<bool> &Features) const;
    virtual void setFeatureEnabled( llvm::StringMap<bool> &Features,
                                    StringRef Name,
                                    bool Enabled) const;
    bool handleTargetFeatures( std::vector<std::string> &Features,
                               DiagnosticsEngine &Diags) override;
    virtual bool hasFeature( StringRef Feature) const;
    std::optional<std::string> handleAsmEscapedChar(char EscChar) const override;
    std::string_view getClobbers() const {
        return "";
    }
    int getEHDataRegisterNumber( unsigned RegNo) const {
        return -1;
    }
    virtual bool validateAsmConstraint( const char *&Name, TargetInfo::ConstraintInfo &Info) const;
};

class LLVM_LIBRARY_VISIBILITY Elbrus32TargetInfo : public ElbrusTargetInfo
{
    public:
        Elbrus32TargetInfo( const llvm::Triple &triple, const TargetOptions &Opts)
            : ElbrusTargetInfo( triple, Opts)
        {
            resetDataLayout( "e-m:e-p:32:32-i64:64:64-f80:128:128-n32:64-S128");

            switch ( getTriple().getOS() )
            {
              case llvm::Triple::Linux:
                SizeType = UnsignedInt;
                PtrDiffType = SignedInt;
                IntPtrType = SignedInt;
                break;
              default:
                break;
            }

            MaxAtomicPromoteWidth = MaxAtomicInlineWidth = 64;
        }

        virtual BuiltinVaListKind getBuiltinVaListKind() const
        {
            return TargetInfo::CharPtrBuiltinVaList;
        }
};

class LLVM_LIBRARY_VISIBILITY Elbrus64TargetInfo : public ElbrusTargetInfo
{
    public:
        Elbrus64TargetInfo( const llvm::Triple &triple, const TargetOptions &Opts)
          : ElbrusTargetInfo( triple, Opts)
        {
            LongWidth = LongAlign = PointerWidth = PointerAlign = 64;
            IntMaxType = SignedLong;
            //UIntMaxType = UnsignedLong;
            Int64Type = SignedLong;

            resetDataLayout( "e-m:e-p:64:64-i64:64:64-f80:128:128-n32:64-S128");

            MaxAtomicPromoteWidth = MaxAtomicInlineWidth = 64;
        }

        virtual BuiltinVaListKind getBuiltinVaListKind() const
        {
            return TargetInfo::CharPtrBuiltinVaList;
        }
};

class LLVM_LIBRARY_VISIBILITY Elbrus128TargetInfo : public ElbrusTargetInfo
{
    public:
        Elbrus128TargetInfo( const llvm::Triple &triple, const TargetOptions &Opts)
          : ElbrusTargetInfo( triple, Opts)
        {
            LongWidth = LongAlign = 64;
            PointerWidth = PointerAlign = 128;
            IntMaxType = SignedLong;
            //UIntMaxType = UnsignedLong;
            Int64Type = SignedLong;

            resetDataLayout( "e-m:e-p:128:128-i64:64:64-f80:128:128-n32:64-S128");

            MaxAtomicPromoteWidth = MaxAtomicInlineWidth = 64;
        }

        virtual BuiltinVaListKind getBuiltinVaListKind() const
        {
            return TargetInfo::CharPtrBuiltinVaList;
        }
};


} // namespace targets
} // namespace clang
#endif // LLVM_CLANG_LIB_BASIC_TARGETS_PPC_H
