//===--- Elbrus.cpp - Implement Elbrus target feature support -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Elbrus TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "Elbrus.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/MacroBuilder.h"
#include "clang/Basic/TargetBuiltins.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/CodeGen/Lccrt.h"

using namespace llvm;
using namespace llvm::Elbrus;
using namespace clang;
using namespace clang::targets;

const Builtin::Info ElbrusTargetInfo::BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS) \
    {#ID, TYPE, ATTRS, 0,       HeaderDesc::NO_HEADER, ALL_LANGUAGES},
#define TARGET_BUILTIN(ID, TYPE, ATTRS, FEATURE) \
    {#ID, TYPE, ATTRS, FEATURE, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
#include "clang/Basic/BuiltinsElbrus.def"
};

typedef struct {
    const char *fname;
    const char *macro;
} IntrinFile_t;

IntrinFile_t IntrinFiles[] = {
    {"float.h",            "float_h__"},
    {"adxintrin.h",        "adxintrin_h__"},
    {"ammintrin.h",        "ammintrin_h__"},
    {"avx2intrin.h",       "avx2intrin_h__"},
    {"avxintrin.h",        "avxintrin_h__"},
    {"avxvnniintrin.h",    "avxvnniintrin_h__"},
    {"bmi2intrin.h",       "bmi2intrin_h__"},
    {"bmiintrin.h",        "bmiintrin_h__"},
    {"bmmintrin.h",        "bmmintrin_h__"},
    {"clflushoptintrin.h", "clflushoptintrin_h__"},
    {"clwbintrin.h",       "clwbintrin_h__"},
    {"clzerointrin.h",     "clzerointrin_h__"},
    {"e2kbuiltin.h",       "e2kbuiltin_h__"},
    {"e2kintrin.h",        "e2kintrin_h__"},
    {"emmintrin.h",        "emmintrin_h__"},
    {"f16cintrin.h",       "f16cintrin_h__"},
    {"fma4intrin.h",       "fma4intrin_h__"},
    {"fmaintrin.h",        "fmaintrin_h__"},
    {"ia32intrin.h",       "ia32intrin_h__"},
    {"immintrin.h",        "immintrin_h__"},
    {"lzcntintrin.h",      "lzcntintrin_h__"},
    {"mm3dnow.h",          "mm3dnow_h__"},
    {"mmintrin.h",         "mmintrin_h__"},
    {"mwaitxintrin.h",     "mwaitxintrin_h__"},
    {"nmmintrin.h",        "nmmintrin_h__"},
    {"pmmintrin.h",        "pmmintrin_h__"},
    {"popcntintrin.h",     "popcntintrin_h__"},
    {"prfchwintrin.h",     "prfchwintrin_h__"},
    {"rdseedintrin.h",     "rdseedintrin_h__"},
    {"shaintrin.h",        "shaintrin_h__"},
    {"smmintrin.h",        "smmintrin_h__"},
    {"tbmintrin.h",        "tbmintrin_h__"},
    {"tmmintrin.h",        "tmmintrin_h__"},
    {"wmmintrin.h",        "wmmintrin_h__"},
    {"x86intrin.h",        "x86intrin_h__"},
    {"x86gprintrin.h",     "x86gprintrin_h__"},
    {"xmmintrin.h",        "xmmintrin_h__"},
    {"xopintrin.h",        "xopintrin_h__"},
    {0, 0}
};

/// ElbrusTargetInfo::getTargetDefines - Return a set of the Elbrus-specific
/// #defines that are not tied to a specific subtarget.
void
ElbrusTargetInfo::getTargetDefines( const LangOptions &Opts,
                                    MacroBuilder &Builder) const
{
    int iset = 0;
    std::string include_sys = llvm::Lccrt::getIncludePath( getTriple(), "c");

    // Target identification.
    Builder.defineMacro( "__e2k__");
    Builder.defineMacro( "_ARCH_E2K");
    Builder.defineMacro( "__elbrus__");
    Builder.defineMacro( "__ELBRUS__");
    if ( (PointerWidth == 128) ) {
        Builder.defineMacro( "_ARCH_E2K128");
        Builder.defineMacro( "__elbrus128__");
        Builder.defineMacro( "__e2k128__");
        Builder.defineMacro( "__ptr128__");
    } else if ( (PointerWidth == 64) ) {
        Builder.defineMacro( "_ARCH_E2K64");
        Builder.defineMacro( "__elbrus64__");
        Builder.defineMacro( "__e2k64__");
        Builder.defineMacro( "__ptr64__");
    } else {
        Builder.defineMacro( "_ARCH_E2K32");
        Builder.defineMacro( "__elbrus32__");
        Builder.defineMacro( "__e2k__");
        Builder.defineMacro( "__ptr32__");
    }

    // Target properties.
    Builder.defineMacro( "_LITTLE_ENDIAN");
    Builder.defineMacro( "__LITTLE_ENDIAN__");

    Builder.defineMacro( "__LCC_MAS_NO", "0");
    Builder.defineMacro( "__LCC_MAS_SPEC", "((int)0x10000000)");
    Builder.defineMacro( "__LCC_MAS_VOLATILE", "((int)0x20000000)");
    Builder.defineMacro( "__LCC_MAS_CLEARTAG", "((int)0x40000000)");
    Builder.defineMacro( "__LCC_CHAN_ANY", "(-1)");

    if ( (CPU == "elbrus-v2") ) {
        iset = 2;
        Builder.defineMacro( "__iset__", "2");
    } else if ( (CPU == "elbrus-2c+") ) {
        iset = 2;
        Builder.defineMacro( "__iset__", "2");
        Builder.defineMacro( "__elbrus_2cplus__");
    } else if ( (CPU == "elbrus-v3") ) {
        iset = 3;
        Builder.defineMacro( "__iset__", "3");
    } else if ( (CPU == "elbrus-4c") ) {
        iset = 3;
        Builder.defineMacro( "__iset__", "3");
        Builder.defineMacro( "__elbrus_4c__");
    } else if ( (CPU == "elbrus-v4") ) {
        iset = 4;
        Builder.defineMacro( "__iset__", "4");
    } else if ( (CPU == "elbrus-8c") ) {
        iset = 4;
        Builder.defineMacro( "__iset__", "4");
        Builder.defineMacro( "__elbrus_8c__");
    } else if ( (CPU == "elbrus-1c+") ) {
        iset = 4;
        Builder.defineMacro( "__iset__", "4");
        Builder.defineMacro( "__elbrus_1cplus__");
    } else if ( (CPU == "elbrus-v5") ) {
        iset = 5;
        Builder.defineMacro( "__iset__", "5");
    } else if ( (CPU == "elbrus-8c2") ) {
        iset = 5;
        Builder.defineMacro( "__iset__", "5");
        Builder.defineMacro( "__elbrus_8c2__");
    } else if ( (CPU == "elbrus-v6") ) {
        iset = 6;
        Builder.defineMacro( "__iset__", "6");
    } else if ( (CPU == "elbrus-16c") ) {
        iset = 6;
        Builder.defineMacro( "__iset__", "6");
        Builder.defineMacro( "__elbrus_16c__");
    } else if ( (CPU == "elbrus-2c3") ) {
        iset = 6;
        Builder.defineMacro( "__iset__", "6");
        Builder.defineMacro( "__elbrus_2c3__");
    } else if ( (CPU == "elbrus-12c") ) {
        iset = 6;
        Builder.defineMacro( "__iset__", "6");
        Builder.defineMacro( "__elbrus_12c__");
    } else if ( (CPU == "elbrus-v7") ) {
        iset = 7;
        Builder.defineMacro( "__iset__", "7");
    } else if ( (CPU == "elbrus-48c") ) {
        iset = 7;
        Builder.defineMacro( "__iset__", "7");
        Builder.defineMacro( "__elbrus_48c__");
    } else if ( (CPU == "elbrus-8v7") ) {
        iset = 7;
        Builder.defineMacro( "__iset__", "7");
        Builder.defineMacro( "__elbrus_8v7__");
    }

    if ( HasMMX ) Builder.defineMacro( "__MMX__");
    if ( HasSSE ) Builder.defineMacro( "__SSE__");
    if ( HasSSE2 ) Builder.defineMacro( "__SSE2__");
    if ( HasSSE3 ) Builder.defineMacro( "__SSE3__");
    if ( HasSSSE3 ) Builder.defineMacro( "__SSSE3__");
    if ( HasSSE4_1 ) Builder.defineMacro( "__SSE4_1__");
    if ( HasSSE4_2 ) Builder.defineMacro( "__SSE4_2__");
    if ( HasAVX ) Builder.defineMacro( "__AVX__");
    if ( Has3DNOW ) Builder.defineMacro( "__3dNOW__");
    if ( Has3DNOWA ) Builder.defineMacro( "__3dNOW_A__");
    if ( HasSSE4A ) Builder.defineMacro( "__SSE4A__");
    if ( HasFMA4 ) Builder.defineMacro( "__FMA4__");
    if ( HasXOP ) Builder.defineMacro( "__XOP__");
    if ( HasAES ) Builder.defineMacro( "__AES__");
    if ( HasPCLMUL ) Builder.defineMacro( "__PCLMUL__");
    if ( HasRDRND ) Builder.defineMacro( "__RDRND__");
    if ( HasBMI ) Builder.defineMacro( "__BMI__");
    if ( HasTBM ) Builder.defineMacro( "__TBM__");
    if ( HasABM ) Builder.defineMacro( "__ABM__");
    if ( HasF16C ) Builder.defineMacro( "__F16C__");
    if ( HasPOPCNT ) Builder.defineMacro( "__POPCNT__");
    if ( HasRDSEED ) Builder.defineMacro( "__RDSEED__");
    if ( HasLZCNT ) Builder.defineMacro( "__LZCNT__");
    if ( HasMWAITX ) Builder.defineMacro( "__MWAITX__");
    if ( HasCLZERO ) Builder.defineMacro( "__CLZERO__");
    if ( HasCLFLUSHOPT ) Builder.defineMacro( "__CLFLUSHOPT__");
    if ( HasCLWB ) Builder.defineMacro( "__CLWB__");
    if ( HasBMI2 ) Builder.defineMacro( "__BMI2__");
    if ( HasFMA ) Builder.defineMacro( "__FMA__");
    if ( HasAVX2 ) Builder.defineMacro( "__AVX2__");
    if ( HasSHA2 ) Builder.defineMacro( "__SHA2__");

    for ( int i = 0; IntrinFiles[i].fname; ++i ) {
        std::string macro = "__LLVM_LCCRT_INTRIN_";
        std::string fname = "\"" + include_sys;

        macro += IntrinFiles[i].macro;
        fname += "/";
        fname += IntrinFiles[i].fname;
        fname += "\"";
        Builder.defineMacro( macro, fname);
    }

    // All of the __sync_(bool|val)_compare_and_swap_(1|2|4|8) builtins work.
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_1");
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_2");
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4");
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8");
    if ( iset >= 5 ) Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_16");

    return;
} /* ElbrusTargetInfo::getTargetDefines */

void
ElbrusTargetInfo::getDefaultFeatures( llvm::StringMap<bool> &Features) const
{
    return;
} /* ElbrusTargetInfo::getDefaultFeatures */

void
ElbrusTargetInfo::setFeatureEnabled( llvm::StringMap<bool> &Features,
                                     StringRef Name,
                                     bool Enabled) const
{
    uint64_t fmask = 0;
    XVecOptInfo *xvoi = 0;

    if (Name == "sse4") {
        if (Enabled) {
            Name = "sse4.2";
        } else {
            Name = "sse4.1";
        }
    }

    for ( int i = 0; XVecOpts[i].name; ++i ) {
        if ( Name == XVecOpts[i].name ) {
            xvoi = XVecOpts + i;
            break;
        }
    }

    if ( xvoi ) {
        llvm::Elbrus::updateXVecFeatures( fmask, xvoi, Features, Enabled);
    }

    return;
} /* ElbrusTargetInfo::setFeatureEnabled */

bool
ElbrusTargetInfo::hasFeature( StringRef Feature) const
{
    return (false);
} /* ElbrusTargetInfo::hasFeature */

const char *const ElbrusTargetInfo::GCCRegNames[] = {
    "g0",  "g1",  "g2",  "g3",  "g4",  "g5",  "g6",  "g7",  "g8",  "g9",
    "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g18", "g19",
    "g20", "g21", "g22", "g23", "g24", "g25", "g26", "g27", "g28", "g29",
    "g30", "g31",

    "r0",  "r1",  "r2",  "r3",  "r4",  "r5",  "r6",  "r7",  "r8",  "r9",
    "r10", "r11", "r12", "r13", "r14", "r15", "r16", "r17", "r18", "r19",
    "r20", "r21", "r22", "r23", "r24", "r25", "r26", "r27", "r28", "r29",
    "r30", "r31", "r32", "r33", "r34", "r35", "r36", "r37", "r38", "r39",
    "r40", "r41", "r42", "r43", "r44", "r45", "r46", "r47", "r48", "r49",
    "r50", "r51", "r52", "r53", "r54", "r55", "r56", "r57", "r58", "r59",
    "r60", "r61", "r62", "r63",

    "b[0]",   "b[1]",   "b[2]",   "b[3]",   "b[4]",   "b[5]",   "b[6]",   "b[7]",   "b[8]",   "b[9]",
    "b[10]",  "b[11]",  "b[12]",  "b[13]",  "b[14]",  "b[15]",  "b[16]",  "b[17]",  "b[18]",  "b[19]",
    "b[20]",  "b[21]",  "b[22]",  "b[23]",  "b[24]",  "b[25]",  "b[26]",  "b[27]",  "b[28]",  "b[29]",
    "b[30]",  "b[31]",  "b[32]",  "b[33]",  "b[34]",  "b[35]",  "b[36]",  "b[37]",  "b[38]",  "b[39]",
    "b[40]",  "b[41]",  "b[42]",  "b[43]",  "b[44]",  "b[45]",  "b[46]",  "b[47]",  "b[48]",  "b[49]",
    "b[50]",  "b[51]",  "b[52]",  "b[53]",  "b[54]",  "b[55]",  "b[56]",  "b[57]",  "b[58]",  "b[59]",
    "b[60]",  "b[61]",  "b[62]",  "b[63]",  "b[64]",  "b[65]",  "b[66]",  "b[67]",  "b[68]",  "b[69]",
    "b[70]",  "b[71]",  "b[72]",  "b[73]",  "b[74]",  "b[75]",  "b[76]",  "b[77]",  "b[78]",  "b[79]",
    "b[80]",  "b[81]",  "b[82]",  "b[83]",  "b[84]",  "b[85]",  "b[86]",  "b[87]",  "b[88]",  "b[89]",
    "b[90]",  "b[91]",  "b[92]",  "b[93]",  "b[94]",  "b[95]",  "b[96]",  "b[97]",  "b[98]",  "b[99]",
    "b[100]", "b[101]", "b[102]", "b[103]", "b[104]", "b[105]", "b[106]", "b[107]", "b[108]", "b[109]",
    "b[110]", "b[111]", "b[112]", "b[113]", "b[114]", "b[115]", "b[116]", "b[117]", "b[118]", "b[119]",
    "b[120]", "b[121]", "b[122]", "b[123]", "b[124]", "b[125]", "b[126]", "b[127]",

    "ctpr1", "ctpr2", "ctpr3",

    "pred0",  "pred1",  "pred2",  "pred3",  "pred4",  "pred5",  "pred6",  "pred7",
    "pred8",  "pred9",  "pred10", "pred11", "pred12", "pred13", "pred14", "pred15",
    "pred16", "pred17", "pred18", "pred19", "pred20", "pred21", "pred22", "pred23",
    "pred24", "pred25", "pred26", "pred27", "pred28", "pred29", "pred30", "pred31",
};

ArrayRef<const char *>
ElbrusTargetInfo::getGCCRegNames() const {
    return llvm::ArrayRef( GCCRegNames);
}

const TargetInfo::GCCRegAlias ElbrusTargetInfo::GCCRegAliases[] = {
    // While some of these aliases do map to different registers
    // they still share the same register name.
    {{""}, ""},
};

ArrayRef<TargetInfo::GCCRegAlias>
ElbrusTargetInfo::getGCCRegAliases() const {
    return llvm::ArrayRef( GCCRegAliases);
}

ArrayRef<Builtin::Info>
ElbrusTargetInfo::getTargetBuiltins() const {
  return llvm::ArrayRef( BuiltinInfo, clang::Elbrus::LastTSBuiltin - Builtin::FirstTSBuiltin);
}

std::optional<std::string>
ElbrusTargetInfo::handleAsmEscapedChar(char EscChar) const {
  std::optional<std::string> r;
  switch (EscChar) {
  case '#':
    r = std::string( "%#");
    break;
  default:
    break;
  }

  return r;
}

bool
ElbrusTargetInfo::validateAsmConstraint( const char *&Name,
                                         TargetInfo::ConstraintInfo &Info) const {
  switch (*Name) {
  default:
    return false;
  case 'I':
  case 'J':
    return true;
  case 'x': // long double x-register, qp-register
    Info.setAllowsRegister();
    return true;
#if 0
  case 'e': // long double x-register, qp-register
    Info.setAllowsRegister();
    return true;
  // x86-registers (from gcc)
  case 'R':
  case 'q':
  case 'f':
  case 't':
  case 'u':
  case 'a':
  case 'b':
  case 'c':
  case 'd':
  //case 'x':
  case 'y':
  case 'A':
  case 'D':
  case 'S':
    Info.setAllowsRegister();
    return true;
#endif
  }
  return false;
}

bool
ElbrusTargetInfo::handleTargetFeatures( std::vector<std::string> &Features,
                                        DiagnosticsEngine &Diags)
{
    bool r = true;
    struct {
        const char *name;
        bool *value;
    } opts[] = {
        {"mmx",        &HasMMX},
        {"popcnt",     &HasPOPCNT},
        {"sse",        &HasSSE},
        {"sse2",       &HasSSE2},
        {"sse3",       &HasSSE3},
        {"ssse3",      &HasSSSE3},
        {"sse4.1",     &HasSSE4_1},
        {"sse4.2",     &HasSSE4_2},
        {"avx",        &HasAVX},
        {"avx2",       &HasAVX2},
        {"sse4a",      &HasSSE4A},
        {"fma4",       &HasFMA4},
        {"xop",        &HasXOP},
        {"fma",        &HasFMA},
        {"bmi",        &HasBMI},
        {"bmi2",       &HasBMI2},
        {"aes",        &HasAES},
        {"pclmul",     &HasPCLMUL},
        {"3dnow",      &Has3DNOW},
        {"3dnowa",     &Has3DNOWA},
        {"clflushopt", &HasCLFLUSHOPT},
        {"clwb",       &HasCLWB},
        {"clzero",     &HasCLZERO},
        {"f16c",       &HasF16C},
        {"lzcnt",      &HasLZCNT},
        {"mwaitx",     &HasMWAITX},
        {"rdrnd",      &HasRDRND},
        {"rdseed",     &HasRDSEED},
        {"sha2",       &HasSHA2},
        {"abm",        &HasABM},
        {"tbm",        &HasTBM},
        {"avxvnni",    &HasAVXVNNI},
        {0,            0},
    };

    //llvm::dbgs() << "Features\n";
    for ( const auto &Feature : Features ) {
        bool value = (Feature[0] == '+');
        StringRef name = Feature.substr( 1);

        //llvm::dbgs() << "  feature: " << Feature << "\n";
        for ( int i = 0; opts[i].name; ++i ) {
            if ( name == opts[i].name ) {
                opts[i].value[0] = value;
                break;
            }
        }
    }

    return (r);
} /* ElbrusTargetInfo::handleTargetFeatures */

