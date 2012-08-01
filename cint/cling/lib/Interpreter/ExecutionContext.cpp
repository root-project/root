//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// version: $Id$
// author:  Axel Naumann <axel@cern.ch>
//------------------------------------------------------------------------------

#include "ExecutionContext.h"

#include "cling/Interpreter/Value.h"

#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/ExecutionEngine/JIT.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/DynamicLibrary.h"

#include <cstdio>
#include <iostream>

using namespace cling;

namespace {
  class JITtedFunctionCollector : public llvm::JITEventListener {
  private:
    std::vector<llvm::Function *> m_vec_functions;
    llvm::ExecutionEngine *m_engine;

  public:
    JITtedFunctionCollector(): m_vec_functions(), m_engine(0) { }
    virtual ~JITtedFunctionCollector() { }

    virtual void NotifyFunctionEmitted(const llvm::Function&, void *, size_t,
                               const JITEventListener::EmittedFunctionDetails&);
    virtual void NotifyFreeingMachineCode(void *OldPtr) {}

    void CleanupList();
    void UnregisterFunctionMapping(llvm::ExecutionEngine&);
  };
}


void JITtedFunctionCollector::NotifyFunctionEmitted(const llvm::Function &F,
                                                    void *Code, size_t Size,
                        const JITEventListener::EmittedFunctionDetails &Details)
{
  //BB std::cerr << "JITtedFunctionCollector::NotifyFunctionEmitted: "
  //BB              "m_vec_functions.push_back("
  //BB           << F.getName().data() << "); Code @ " << Code << std::endl;
  m_vec_functions.push_back(const_cast<llvm::Function *>(&F));
}

void JITtedFunctionCollector::CleanupList()
{
  m_vec_functions.clear();
}

void JITtedFunctionCollector::UnregisterFunctionMapping(
                                                  llvm::ExecutionEngine &engine)
{
  std::vector<llvm::Function *>::reverse_iterator it;

  for (it=m_vec_functions.rbegin(); it < m_vec_functions.rend(); it++) {
    llvm::Function *ff = (llvm::Function *)*it;
    //BB std::cerr << "JITtedFunctionCollector::UnregisterFunctionMapping: "
    //BB              "updateGlobalMapping("
    //BB           << ff->getName().data() << ", 0); Global @"
    //BB           << engine.getPointerToGlobalIfAvailable(ff) << std::endl;
    engine.freeMachineCodeForFunction(ff);
    engine.updateGlobalMapping(ff, 0);
    //BB std::cerr << "Global after delete @"
    //BB           << engine.getPointerToGlobalIfAvailable(ff) << std::endl;
  }
  m_vec_functions.clear();
}


std::vector<std::string> ExecutionContext::m_vec_unresolved;
std::vector<ExecutionContext::LazyFunctionCreatorFunc_t>
  ExecutionContext::m_vec_lazy_function;

ExecutionContext::ExecutionContext():
  m_engine(0),
  m_posInitGlobals(0),
  m_RunningStaticInits(false)
{
}

void
ExecutionContext::InitializeBuilder(llvm::Module* m)
{
  //
  //  Create an execution engine to use.
  //
  // Note: Engine takes ownership of the module.
  assert(m && "Module cannot be null");

  llvm::EngineBuilder builder(m);
  builder.setOptLevel(llvm::CodeGenOpt::Less);
  std::string errMsg;
  builder.setErrorStr(&errMsg);
  builder.setEngineKind(llvm::EngineKind::JIT);
  builder.setAllocateGVsWithCode(false);
  m_engine = builder.create();
  assert(m_engine && "Cannot initialize builder without module!");

  //m_engine->addModule(m); // Note: The engine takes ownership of the module.

  // install lazy function
  m_engine->InstallLazyFunctionCreator(NotifyLazyFunctionCreators);
}

ExecutionContext::~ExecutionContext()
{
}

void unresolvedSymbol()
{
  // throw exception?
  std::cerr << "Error: calling unresolved symbol (should never happen)!"
            << std::endl;
}

void* ExecutionContext::HandleMissingFunction(const std::string& mangled_name)
{
  // Not found in the map, add the symbol in the list of unresolved symbols
  std::vector<std::string>::iterator it;
  it = find (m_vec_unresolved.begin(), m_vec_unresolved.end(), mangled_name);
  if (it == m_vec_unresolved.end())
    m_vec_unresolved.push_back(mangled_name);
  // Avoid "ISO C++ forbids casting between pointer-to-function and
  // pointer-to-object":
  return (void*)reinterpret_cast<size_t>(unresolvedSymbol);
}

void ExecutionContext::ResetUnresolved()
{
    m_vec_unresolved.clear();
}

void*
ExecutionContext::NotifyLazyFunctionCreators(const std::string& mangled_name)
{
  void *ret = 0;

  std::vector<LazyFunctionCreatorFunc_t>::iterator it;
  for (it=m_vec_lazy_function.begin(); it < m_vec_lazy_function.end(); it++) {
    ret = (void*)((LazyFunctionCreatorFunc_t)*it)(mangled_name);
    if (ret != 0) return ret;
  }
  return HandleMissingFunction(mangled_name);
}

void
ExecutionContext::executeFunction(llvm::StringRef funcname,
                                  llvm::GenericValue* returnValue)
{
  // Call a function without arguments, or with an SRet argument, see SRet below

  // Rewire atexit:
  llvm::Function* atExit = m_engine->FindFunctionNamed("__cxa_atexit");
  llvm::Function* clingAtExit = m_engine->FindFunctionNamed("cling_cxa_atexit");
  if (atExit && clingAtExit) {
    void* clingAtExitAddr = m_engine->getPointerToFunction(clingAtExit);
    if (clingAtExitAddr) {
      m_engine->updateGlobalMapping(atExit, clingAtExitAddr);
    }
  }

  llvm::Function* f = m_engine->FindFunctionNamed(funcname.data());
  if (!f) {
    fprintf(stderr, "executeFunction: Could not find function named: %s\n",
            funcname.data()
            );
    return;
  }
  JITtedFunctionCollector listener;
  // register the listener
  m_engine->RegisterJITEventListener(&listener);
  m_engine->getPointerToFunction(f);
  // check if there is any unresolved symbol in the list
  if (!m_vec_unresolved.empty()) {
    std::cerr << "ExecutionContext::executeFunction:" << std::endl;
    for (size_t i = 0, e = m_vec_unresolved.size(); i != e; ++i) {
      std::cerr << "Error: Symbol \'" << m_vec_unresolved[i] <<
        "\' unresolved!" << std::endl;
      llvm::Function *ff
        = m_engine->FindFunctionNamed(m_vec_unresolved[i].c_str());
      if (ff) {
        m_engine->updateGlobalMapping(ff, 0);
        m_engine->freeMachineCodeForFunction(ff);
      }
      else {
        std::cerr << "Error: Canot find symbol \'" << m_vec_unresolved[i] <<
          std::endl;
      }
    }
    m_vec_unresolved.clear();
    // cleanup functions
    listener.UnregisterFunctionMapping(*m_engine);
    m_engine->UnregisterJITEventListener(&listener);
    return;
  }
  // cleanup list and unregister our listener
  listener.CleanupList();
  m_engine->UnregisterJITEventListener(&listener);

  std::vector<llvm::GenericValue> args;
  bool wantReturn = (returnValue);

  if (f->hasStructRetAttr()) {
    // Function expects to receive the storage for the returned aggregate as
    // first argument. Make sure returnValue is able to receive it, i.e.
    // that it has the pointer set:
    assert(returnValue && GVTOP(*returnValue) || "must call function returning aggregate with returnValue pointing to the storage space for return value!");

    args.push_back(*returnValue);
    // will get set as arg0, must not assign.
    wantReturn = false;
  }

  if (wantReturn) {
    *returnValue = m_engine->runFunction(f, args);
  } else {
    m_engine->runFunction(f, args);
  }

  m_engine->freeMachineCodeForFunction(f);
}


void
ExecutionContext::runStaticInitializersOnce(llvm::Module* m) {
  assert(m && "Module must not be null");

  if (!m_engine)
    InitializeBuilder(m);

  assert(m_engine && "Code generation did not create an engine!");

  if (!m_RunningStaticInits) {
    m_RunningStaticInits = true;

    llvm::GlobalVariable* gctors
      = m->getGlobalVariable("llvm.global_ctors", true);
    if (gctors) {
      m_engine->runStaticConstructorsDestructors(false);
      gctors->eraseFromParent();
    }

    m_RunningStaticInits = false;
  }
}

void
ExecutionContext::runStaticDestructorsOnce(llvm::Module* m) {
  assert(m && "Module must not be null");
  assert(m_engine && "Code generation did not create an engine!");

  llvm::GlobalVariable* gdtors
    = m->getGlobalVariable("llvm.global_dtors", true);
  if (gdtors) {
    m_engine->runStaticConstructorsDestructors(true);
  }

}

int
ExecutionContext::verifyModule(llvm::Module* m)
{
  //
  //  Verify generated module.
  //
  bool mod_has_errs = llvm::verifyModule(*m, llvm::PrintMessageAction);
  if (mod_has_errs) {
    return 1;
  }
  return 0;
}

void
ExecutionContext::printModule(llvm::Module* m)
{
  //
  //  Print module LLVM code in human-readable form.
  //
  llvm::PassManager PM;
  PM.add(llvm::createPrintModulePass(&llvm::outs()));
  PM.run(*m);
}

void
ExecutionContext::installLazyFunctionCreator(LazyFunctionCreatorFunc_t fp)
{
  m_vec_lazy_function.push_back(fp);
  //m_engine->InstallLazyFunctionCreator(fp);
}

bool ExecutionContext::addSymbol(const char* symbolName,  void* symbolAddress){

  void* actualAdress
    = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(symbolName);
  if (actualAdress)
    return false;

  llvm::sys::DynamicLibrary::AddSymbol(symbolName, symbolAddress);
  return true;
}
