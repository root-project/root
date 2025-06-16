//===--- Elbrus.cpp - Elbrus Helpers for Tools ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Elbrus.h"
#include "clang/Driver/Options.h"
#include "llvm/Option/ArgList.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Support/Debug.h"

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;
using namespace llvm;

const char *XVecOptsNames[] = {
    "3dnow", "3dnowa",
    "abm", "aes", "avx", "avx2", "avxvnni",
    "bmi", "bmi2",
    "clflushopt", "clwb", "clzero",
    "f16c", "fma", "fma4",
    "lzcnt",
    "mmx", "mwaitx",
    "pclmul", "popcnt",
    "rdrnd", "rdseed",
    "sha", "sse", "sse2", "sse3", "sse4.1", "sse4.2", "sse4a", "ssse3",
    "tbm",
    "xop",
    "xvec-emul",
    0
};

std::string elbrus::getElbrusTargetCPU(const ArgList &Args) {
  const char *cpu_arch = "";

  if ( const Arg *A = Args.getLastArg( clang::driver::options::OPT_march_EQ) )
    cpu_arch = A->getValue();

  if ( (strcmp( cpu_arch, "native") == 0) ) {
      auto cpuHost = llvm::sys::getHostCPUName();
      if ( !cpuHost.empty() && (cpuHost != "generic") ) {
          return std::string( cpuHost.data());
      } else {
          return "elbrus-v2";
      }
  }

  if ( !cpu_arch
       || (strcmp( cpu_arch, "") == 0)
       || (strcmp( cpu_arch, "generic") == 0) ) {
      cpu_arch = "elbrus-v2";
  }

  return cpu_arch;
}

void elbrus::getElbrusTargetFeatures(const Driver &D, const llvm::Triple &Triple,
                                     const ArgList &Args,
									 std::vector<llvm::StringRef> &Features)
{
#if 0
    // -m(no-)htm overrides use of the transactional-execution facility.
    if (Arg *A = Args.getLastArg(options::OPT_mhtm, options::OPT_mno_htm)) {
        if (A->getOption().matches(options::OPT_mhtm))
            Features.push_back("+transactional-execution");
        else
            Features.push_back("-transactional-execution");
    }
    // -m(no-)vx overrides use of the vector facility.
    if (Arg *A = Args.getLastArg(options::OPT_mvx, options::OPT_mno_vx)) {
        if (A->getOption().matches(options::OPT_mvx))
            Features.push_back("+vector");
        else
            Features.push_back("-vector");
    }
#endif

    // Now add any that the user explicitly requested on the command line,
    // which may override the defaults.
    for ( const Arg *A : Args.filtered( options::OPT_m_x86_Features_Group,
                                        options::OPT_m_e2k_Features_Group) )
    {
        int opti = -1;
        StringRef Name = A->getOption().getName();

        // Skip over "-m".
        assert(Name.starts_with("m") && "Invalid feature name.");
        Name = Name.substr(1);

        bool IsNegative = Name.starts_with("no-");
        if (IsNegative)
            Name = Name.substr(3);

        if ( Name == "sse4" ) {
            Name = IsNegative ? "sse4.1" : "sse4.2";
        }

        for ( int i = 0; XVecOptsNames[i]; ++i ) {
            if ( Name == StringRef( XVecOptsNames[i]) ) {
                opti = i;
            }
        }

        if ( opti == -1 ) {
            continue;
        }

        A->claim();

        Features.push_back( Args.MakeArgString( (IsNegative ? "-" : "+") + Name));
    }
}
