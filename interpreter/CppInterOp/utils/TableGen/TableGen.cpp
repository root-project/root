//===- TableGen.cpp - Top-level TableGen for CppInterOp -------------------===//
//
// Part of the compiler-research project, under the Apache License v2.0 with
// LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;

void EmitCppInterOpAPI(const RecordKeeper& Records, raw_ostream& OS);
void EmitCppInterOpDecl(const RecordKeeper& Records, raw_ostream& OS);
void EmitCXCppInterOpDecl(const RecordKeeper& Records, raw_ostream& OS);
void EmitCXCppInterOpImpl(const RecordKeeper& Records, raw_ostream& OS);

enum ActionType {
  GenCppInterOpAPI,
  GenCppInterOpDecl,
  GenCXCppInterOpDecl,
  GenCXCppInterOpImpl,
};

namespace {
cl::opt<ActionType> Action(
    cl::desc("Action to perform:"),
    cl::values(
        clEnumValN(GenCppInterOpAPI, "gen-cppinterop-api",
                   "Generate CppInterOpAPI.inc X-macro file"),
        clEnumValN(GenCppInterOpDecl, "gen-cppinterop-decl",
                   "Generate CppInterOpDecl.inc function declarations"),
        clEnumValN(GenCXCppInterOpDecl, "gen-cx-cppinterop-decl",
                   "Generate CXCppInterOpDecl.inc C API declarations"),
        clEnumValN(GenCXCppInterOpImpl, "gen-cx-cppinterop-impl",
                   "Generate CXCppInterOpImpl.inc C API implementations")));

bool CppInterOpTableGenMain(raw_ostream& OS, const RecordKeeper& Records) {
  switch (Action) {
  case GenCppInterOpAPI:
    EmitCppInterOpAPI(Records, OS);
    break;
  case GenCppInterOpDecl:
    EmitCppInterOpDecl(Records, OS);
    break;
  case GenCXCppInterOpDecl:
    EmitCXCppInterOpDecl(Records, OS);
    break;
  case GenCXCppInterOpImpl:
    EmitCXCppInterOpImpl(Records, OS);
    break;
  }
  return false;
}
} // end anonymous namespace

int main(int argc, char** argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  llvm_shutdown_obj Y;

  return TableGenMain(argv[0], &CppInterOpTableGenMain);
}
