//===-- Lccrt.cpp - Common Lccrt code ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LccrtPass class.
//
//===----------------------------------------------------------------------===//

#include <unistd.h>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <sstream>

#include "LccrtEmitter.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/CodeGen/Lccrt.h"
#include "llvm/IR/EHPersonalities.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;

#ifndef LLVM_WITH_LCCRT

#ifdef NDEBUG
#define assert_define( x)
#else /* !NDEBUG */
#define assert_define( x) x
#endif /* NDEBUG */

std::string
Lccrt::getVersion( const Triple &T)
{
    return ("");
} /* Lccrt::getVersion */

std::string
Lccrt::getToolchain( const Triple &T, const char *name)
{
    return ("");
} /* Lccrt::getToolchain */

std::string
Lccrt::getToolchainPath( const Triple &T, const char *name)
{
    return ("");
} /* Lccrt::getToolchainPath */

std::string
Lccrt::getLibPath( const Triple &T, const char *name)
{
    return ("");
} /* Lccrt::getLibPath */

std::string
Lccrt::getIncludePath( const Triple &T, const char *name)
{
    return ("");
} /* Lccrt::getIncludePath */

LLVM_EXTERNAL_VISIBILITY Pass *
Lccrt::createAsmPass( TargetMachine &TM, raw_pwrite_stream &Out, CodeGenFileType type)
{
    return (0);
} /* Lccrt::createAsmPass */

#else /* LLVM_WITH_LCCRT */

#define DEBUG_TYPE "lccrt"

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
#define debug_print( v) v
#else /* defined(NDEBUG) && !defined(LLVM_ENABLE_DUMP) */
#define debug_print( v)
#endif /* !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP) */

char LccrtPass::ID = 0;

LccrtContext::LccrtContext()
{
    ctx = lccrt_context_new( 0, 0);

    return;
} /* LccrtContext::LccrtContext */

LccrtContext::~LccrtContext()
{
    lccrt_context_delete( ctx);
    ctx = 0;

    return;
} /* LccrtContext::~LccrtContext */

lccrt_context_ptr
LccrtContext::get()
{
    static thread_local LccrtContext lc;

    return (lc);
} /* LccrtContext::get */

std::string
LccrtContext::getLibName( const Triple &T)
{
    std::string r;

    if ( (T.getOS() == Triple::Linux) )
    {
        if ( (T.getArch() == Triple::x86)
             || (T.getArch() == Triple::x86_64) )
        {
            //r = "gccjit";
            r = "lcbe";

        } else if ( (T.getArch() == Triple::e2k32)
                    || (T.getArch() == Triple::e2k64)
                    || (T.getArch() == Triple::e2k128) )
        {
            r = "lccopt-";
            r += getArchName( T);
        }
    }

    return (r);
} /* LccrtContext::getLibName */

std::string
LccrtContext::getArchName( const Triple &T)
{
    std::string r;

    if ( (T.getOS() == Triple::Linux) )
    {
        if ( (T.getArch() == Triple::x86) ) {
            r = "x86_32";
        } else if ( (T.getArch() == Triple::x86_64) ) {
            r = "x86_64";
            //r = "c";
        } else if ( (T.getArch() == Triple::e2k32) ) {
            r = "e2k32";
        } else if ( (T.getArch() == Triple::e2k64) ) {
            r = "e2k64";
        } else if ( (T.getArch() == Triple::e2k128) ) {
            r = "e2k128";
        }
    }

    return (r);
} /* LccrtContext::getArchName */

std::string
Lccrt::getVersion( const Triple &T)
{
    lccrt_context_ptr c = LccrtContext::get();
    std::string l = LccrtContext::getLibName( T);
    std::string a = LccrtContext::getArchName( T);
    std::string r = lccrt_context_get_toolchain( c, l.c_str(), a.c_str(), "version", "plugin");

    return (r);
} /* Lccrt::getVersion */

std::string
Lccrt::getToolchain( const Triple &T, const char *name)
{
    lccrt_context_ptr c = LccrtContext::get();
    std::string l = LccrtContext::getLibName( T);
    std::string a = LccrtContext::getArchName( T);
    std::string r = lccrt_context_get_toolchain( c, l.c_str(), a.c_str(), "tool", name);

    return (r);
} /* Lccrt::getToolchain */

std::string
Lccrt::getToolchainPath( const Triple &T, const char *name)
{
    lccrt_context_ptr c = LccrtContext::get();
    std::string l = LccrtContext::getLibName( T);
    std::string a = LccrtContext::getArchName( T);
    std::string r = lccrt_context_get_toolchain( c, l.c_str(), a.c_str(), "tool_path", name);

    return (r);
} /* Lccrt::getToolchainPath */

std::string
Lccrt::getLibPath( const Triple &T, const char *name)
{
    lccrt_context_ptr c = LccrtContext::get();
    std::string l = LccrtContext::getLibName( T);
    std::string a = LccrtContext::getArchName( T);
    std::string r = lccrt_context_get_toolchain( c, l.c_str(), a.c_str(), "lib_path", name);

    return (r);
} /* Lccrt::getLibPath */

std::string
Lccrt::getIncludePath( const Triple &T, const char *name)
{
    lccrt_context_ptr c = LccrtContext::get();
    std::string l = LccrtContext::getLibName( T);
    std::string a = LccrtContext::getArchName( T);
    std::string r = lccrt_context_get_toolchain( c, l.c_str(), a.c_str(), "include_path", name);

    return (r);
} /* Lccrt::getIncludePath */

LLVM_EXTERNAL_VISIBILITY Pass *
Lccrt::createAsmPass( TargetMachine &TM, raw_pwrite_stream &Out, CodeGenFileType type)
{
    Pass *P;

    P = new LccrtPass( TM, Out, type);

    return (P);
} /* Lccrt::createAsmPass */

//INITIALIZE_PASS( LccrtPass, "lccrt", "Lccrt asm-emitter", false, true)

LccrtPass::LccrtPass( TargetMachine &VTM, raw_pwrite_stream &vOut, CodeGenFileType type)
    : ModulePass( LccrtPass::ID), TM(VTM), Out(vOut), OutType( type)
{
    //initializeLccrtPassPass( *PassRegistry::getPassRegistry());
    return;
} /* LccrtPass::LccrtPass */

LccrtPass::~LccrtPass()
{
    return;
} /* LccrtPass::~LccrtPass */

static std::string
intToStr( int64_t value) {
    char buff[256];
    std::string r;

    snprintf( buff, 256, "%jd", value);
    r = buff;

    return (r);
} /* intToStr */

Instruction *
LccrtPass::findBBlockHead( BasicBlock &BB) {
    Instruction *r = 0;

    for ( auto &J : BB ) {
        Instruction &O = J;

        if ( !isa<PHINode>( O) ) {
            r = &O;
            break;
        }
    }

    return (r);
} /* LccrtPass::findBBlockHead */

void
LccrtPass::addModuleIRLogging( Module &M) {
    char fname[1024];
    std::error_code ec;
    std::map<std::string, bool> funcs;
    std::map<void *, int64_t> ctxgmap;
    std::string s;
    std::vector<Type *> vtypes;
    std::vector<Constant *> vcnsts;
    const char *env_lldir = getenv( "LLVM_LCCRT_LOGIR_DIR");
    const char *env_funcs = getenv( "LLVM_LCCRT_FUNCS_LOGIR");
    std::istringstream elog_funcs( env_funcs ? env_funcs : "");
    LLVMContext &Ctx = M.getContext();
    ModuleSlotTracker MST( &M);
    Type *tyv = Type::getVoidTy( Ctx);
    Type *tyi8 = Type::getInt8Ty( Ctx);
    Type *tyi8p = PointerType::get( tyi8, 0);
    FunctionType *tyfv = FunctionType::get( tyv, {tyi8p, tyi8p}, false);
    Function *F_logir_v = Function::Create( tyfv, GlobalValue::ExternalLinkage, "__lccrt_logir_void");

    snprintf( fname, 1024, "%s/llvm.logir.XXXXXX.ll", env_lldir ? env_lldir : ".");
    mkstemps( fname, 3);
    raw_fd_ostream *fs = new raw_fd_ostream( fname, ec, sys::fs::OpenFlags::OF_None);
    M.print( *fs, 0);
    delete fs;

    for ( Module::global_iterator I = M.global_begin(), E = M.global_end(); I != E; ++I ) {
        if ( !I->hasAppendingLinkage() ) {
            ctxgmap[(void *)&(*I)] = vtypes.size();
            vtypes.push_back( I->getType());
            vcnsts.push_back( &(*I));
        }
    }

    for ( Module::alias_iterator I = M.alias_begin(), E = M.alias_end(); I != E; ++I ) {
        ctxgmap[(void *)&(*I)] = vtypes.size();
        vtypes.push_back( I->getType());
        vcnsts.push_back( &(*I));
    }

    for ( Module::iterator I = M.begin(), E = M.end(); I != E; ++I ) {
        ctxgmap[(void *)&(*I)] = vtypes.size();
        vtypes.push_back( I->getType());
        vcnsts.push_back( &(*I));
    }

    StructType *ctxgty = StructType::create( vtypes);
    Constant *ctxgiv = ConstantStruct::get( ctxgty, vcnsts);
    new GlobalVariable( M, ctxgty, true, GlobalValue::InternalLinkage,
                        ctxgiv, "__lccrt_logir_ctxaddrs");
    //ValueToValueMapTy vmap;
    auto CM = CloneModule( M);
    dbgs() << CM.get();

    while ( std::getline( elog_funcs, s, ',') ) {
        funcs[s] = true;
    }

    for ( auto &I : M ) {
        if ( !I.isDeclaration() ) {
            Function &F = I;

            if ( F.hasName()
                 && (funcs.empty() || (funcs.find( F.getName().str()) != funcs.end())) )
            {
                addFunctionIRLogging( MST, F, F_logir_v);
            }
        }
    }

    return;
} /* LccrtPass::addModuleIRLogging */

void
LccrtPass::addFunctionIRLogging( ModuleSlotTracker &MST, Function &F, Function *F_logir_v) {
    LLVMContext &Ctx = F.getContext();
    Module *M = F.getParent();
    FunctionType *tyfv = F_logir_v->getFunctionType();
    IRBuilder<> Builder( Ctx);
    Value *funcsptr = Builder.CreateGlobalStringPtr( F.getName().str(), "", 0, M);

    MST.incorporateFunction( F);

    for ( auto &I : F ) {
        BasicBlock &BB = I;
        Instruction *beg = findBBlockHead( BB);
        std::string sbname = (BB.hasName() ? BB.getName().str() : intToStr( MST.getLocalSlot( &BB))) + ":";
        Value *bbsptr = Builder.CreateGlobalStringPtr( sbname, "", 0, M);

        //errs() << sbname << "\n";
        //errs() << *vbname << "\n";
        //errs() << *bname << "\n";

        for ( auto &J : BB ) {
            std::string so;
            raw_string_ostream RSO( so);
            Value *osptr = 0;
            Instruction &O = J;
           
            if ( !isa<PHINode>( O) ) { 
                O.print( RSO, MST, true);
                osptr = Builder.CreateGlobalStringPtr( so, "", 0, M);
                CallInst::Create( tyfv, F_logir_v, {funcsptr, osptr}, "__lccrt_logir_void", beg);
                if ( !O.getType()->isVoidTy() ) {
                }
            }
        }

        CallInst::Create( tyfv, F_logir_v, {funcsptr, bbsptr}, "__lccrt_logir_void", findBBlockHead( BB));
    }

    return;
} /* LccrtPass::addFunctionIRLogging */

/**
 * Запись блока данных ассемблера.
 */
static int
funcWriteAsm( void *vlp, char *data, uint64_t len)
{
    int r = len;
    LccrtPass *lp = (LccrtPass *)vlp;

    lp->Out.write( data, len);

    return (r);
} /* funcWriteAsm */

bool
LccrtPass::extCompile( Module &M, std::string llc)
{
    std::error_code ec;
    std::string cmd;
    bool r = false;
    bool is_asm = (OutType == CodeGenFileType::AssemblyFile);
    char iname[64] = "/tmp/llvm.lccrt.XXXXXX.ll";
    char oname[128] = "";
    raw_fd_ostream *f0 = 0;
    FILE *f1 = 0;

    mkstemps( iname, 3);
    snprintf( oname, 128, "%s.%s", iname, is_asm ? "s" : "o");
    mkstemps( oname, 5);

    f0 = new raw_fd_ostream( iname, ec, sys::fs::OpenFlags::OF_None);
    M.print( *f0, 0);
    delete f0;

    //cmd += "LCCRT_VERBOSE=1 ";
    //cmd += "LCCOPT_VERBOSE=1 ";
    cmd += "LLVM_LCCRT_LLC= ";
    cmd += llc;
    cmd += " ";
    cmd += iname;
    cmd += " -o ";
    cmd += oname;
    cmd += " -filetype=";
    cmd += (is_asm ? "asm" : "obj");

    if ( (system( cmd.c_str()) != 0) )
    {
        fprintf( stderr, "compile: %s\n", cmd.c_str());
        abort();
    }

    f1 = fopen( oname, "r");
    if ( !f1 )
    {
        fprintf( stderr, "read: %s\n", oname);
        abort();
    } else
    {
        char b[1024];
        int l = 0;

        while ( !feof( f1) )
        {
            l = fread( b, 1, 1024, f1);
            if ( (l > 0) )
            {
                Out.write( b, l);

            } else if ( (l < 0) )
            {
                fprintf( stderr, "read: %s\n", oname);
                abort();
            }
        }

        fclose( f1);
    }

    unlink( iname);
    unlink( oname);

    return (r);
} /* LccrtPass::extCompile */

void
LccrtPass::getAnalysisUsage( AnalysisUsage &AU) const
{
    if ( TM.isLccrtIpa() )
    {
        AU.addRequired<LccrtIpaPass>();
    }
} /* LccrtPass::getAnalysisUsage */

bool
LccrtPass::runOnModule( Module &M)
{
    std::string alib;
    std::string targ;
    bool r = false;
    lccrt_asm_compile_config_t cnf = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    Triple descr = TM.getTargetTriple();
    char *llc = getenv( "LLVM_LCCRT_LLC");
    Reloc::Model reloc_model = TM.getRelocationModel();
    LccrtModuleIpaEmitter mipa = LccrtModuleIpaEmitter( this);
    ModuleSlotTracker MST( &M);

    llvm::SetPrintStack( false);
    reloc_model = (reloc_model == Reloc::Model::DynamicNoPIC) ? Reloc::Model::PIC_ : reloc_model;

    if ( !((reloc_model == Reloc::Model::Static)
           || (reloc_model == Reloc::Model::PIC_)) )
    {
        std::cerr << "\nERROR: reloc-model [" << reloc_model << "] isn't supported\n\n\n";
        abort();
    }

    if ( (M.getPIELevel() != 0)
         && (reloc_model != Reloc::Model::PIC_) )
    {
        std::cerr << "\nERROR: pie-level isn't zero in non-pic mode\n\n\n";
        abort();
    }

    if ( llc
         && llc[0] )
    {
        if ( extCompile( M, llc) )
        {
            return (false);
        } else
        {
            return (true);
        }
    }

    alib = LccrtContext::getLibName( descr);
    targ = LccrtContext::getArchName( descr);

    if ( TM.isLccrtBackendDebug() )
    {
        alib += "-debug";
    }

    if ( TM.isLccrtAsmtest() )
    {
        alib = "asmtest";
        targ = "asmtest";
    }

    if ( alib.empty()
         || targ.empty() )
    {
        return (true);
    }

    if ( getenv( "LLVM_LCCRT_LOGIR_DIR")
         || getenv( "LLVM_LCCRT_FUNCS_LOGIR") ) {
        addModuleIRLogging( M);
    }

    if ( (getenv( "LLVM_LCCRT_DUMP")
          && atoi( getenv( "LLVM_LCCRT_DUMP")))
         || getenv( "LLVM_LCCRT_DUMP_DIR") ) {
        std::error_code ec;
        char name[1024] = {};
        raw_fd_ostream *fs = 0;
        const char *dir0 = getenv( "LLVM_LCCRT_DUMP_DIR");
        const char *dir = dir0 ? (dir0[0] ? dir0 : ".") : "/tmp";

        snprintf( name, 1024, "%s/llvm.lccrt.dump.XXXXXX.ll", dir);

        mkstemps( name, 3);
        fs = new raw_fd_ostream( name, ec, sys::fs::OpenFlags::OF_None);
        M.print( *fs, 0);
        delete fs;
        fprintf( stderr, "SAVE IR-MODULE TO FILE: %s\n", name);
        fflush( stderr);
    }

    LccrtEmitter le( TM, M.getDataLayout(), descr, mipa, MST);
    lccrt_module_ptr m = le.newModule( M);
    Function *eh_func = 0;

    le.dbge_.makeModuleDbgMetadata( m, &M);
    le.dbgle_.makeModuleDbgMetadata( m, &M);

    mipa.open( m, &M);
    if ( !M.getModuleInlineAsm().empty() ) {
        lccrt_module_set_inline_asm( m, M.getModuleInlineAsm().c_str());
    }

    for ( Module::iterator I = M.begin(), E = M.end(); I != E; ++I ) {
        if ( I->isDeclaration()
             && !le.skipedIntrinsicDeclaration( &(*I)) ) {
            if ( !le.isDbg( &(*I)) ) {
                if ( !le.makeFunction( &(*I)) ) {
                    return (true);
                }
            }
        } else {
            Constant *fehp = I->hasPersonalityFn() ? I->getPersonalityFn() : 0;

            if ( fehp ) {
                Function *ehf = dyn_cast<Function>( fehp->stripPointerCasts());

                if ( !ehf ) {
                    fprintf( stderr, "Non function's personalities don't supported yet\n");
                    debug_print( fehp->dump());
                    abort();

                } if ( eh_func
                       && (eh_func != ehf) ) {
                    fprintf( stderr, "Different personalities in one module don't supported yet\n");
                    debug_print( eh_func->dump());
                    debug_print( fehp->dump());
                    abort();
                } else {
                    eh_func = ehf;
                }
            }

            le.numberLabels( &(*I));
        }
    }

    for ( Module::global_iterator I = M.global_begin(), E = M.global_end(); I != E; ++I ) {
        if ( !I->hasAppendingLinkage() ) {
            if ( !le.makeGlobal( &(*I), false) ) {
                return (true);
            }
        }
    }

    for ( Module::alias_iterator I = M.alias_begin(), E = M.alias_end(); I != E; ++I ) {
        if ( !le.makeAlias( &(*I)) ) {
            return (true);
        }
    }

    for ( Module::iterator I = M.begin(), E = M.end(); I != E; ++I ) {
        if ( !I->isDeclaration() ) {
            if ( !le.makeFunction( &(*I)) ) {
                return (true);
            }
        }
    }

    for ( Module::global_iterator I = M.global_begin(), E = M.global_end(); I != E; ++I ) {
        if ( I->hasAppendingLinkage() ) {
            if ( !le.makeAppendingGlobal( &(*I)) ) {
                return (true);
            }
        }
    }

    mipa.close();

    std::string cflags = TM.getLccrtBackendOptions();

    if ( TM.isLccrtCallLong() ) {
        cflags += " -flccrt-call-long";
    }

    if ( TM.isLccrtLlvmIREmbedStaticOnly() ) {
        cnf.is_llvmir_embed_static_only = 1;
    }

    cnf.target = targ.c_str();
    cnf.write = &funcWriteAsm;
    cnf.write_info = this;
    cnf.is_pic = (reloc_model == Reloc::Model::PIC_);
    cnf.pie_level = cnf.is_pic ? M.getPIELevel() : 0;
    cnf.is_jit = TM.isLccrtJit();
    cnf.function_sections = TM.getFunctionSections();
    cnf.data_sections = TM.getDataSections();
    if ( (OutType == CodeGenFileType::AssemblyFile) )
    {
        cnf.out_type = "asm";

    } else if ( (OutType == CodeGenFileType::ObjectFile) )
    {
        cnf.out_type = "obj";
    } else
    {
        cnf.out_type = "none";
    }
    cnf.eh_personality = eh_func ? eh_func->getName().data() : 0;
    cnf.cpu_arch = TM.getTargetCPU().data();
    cnf.cflags = cflags.c_str();
    cnf.asm_verbose = TM.Options.MCOptions.AsmVerbose;
    switch ( TM.getOptLevel() ) {
      case CodeGenOptLevel::None:       cnf.opt_level = 0; break;
      case CodeGenOptLevel::Less:       cnf.opt_level = 1; break;
      case CodeGenOptLevel::Default:    cnf.opt_level = 2; break;
      case CodeGenOptLevel::Aggressive: cnf.opt_level = 4; break;
      default:                          cnf.opt_level = 2; abort(); break;
    }

#ifndef LLVM_LCCRT_RELEASE
    if ( getenv( "LLVM_LCCRT_PRINT_MODULE") )
    {
        int sfd = -1;
        char *fname = getenv( "LLVM_LCCRT_PRINT_MODULE");

        if ( (strcmp( fname, "") == 0) )
        {
            char s[128];

            snprintf( s, 128, "/tmp/llvm.print_module.XXXXXX.txt");
            sfd = mkstemps( s, 4);
            fname = strdup( s);
            //printf( "print module: %s\n", fname);
        }

        {
            lccrt_asm_compile_config_t tcnf;
            lccrt_ctx_ptr tctx = lccrt_context_new( 0, 0);
            int fd = open( fname, O_WRONLY);

            lccrt_module_print( &cnf, m, fd);
            close( fd);
            fd = open( fname, O_RDONLY);
            lccrt_module_delete( lccrt_module_load( tctx, fd, &tcnf));
            close( fd);
        }

        if ( (sfd >= 0) )
        {
            close( sfd);
            unlink( fname);
            free( fname);
        }
    }
#endif /* LLVM_LCCRT_RELEASE */

    if ( (lccrt_module_compile_asm( m, alib.c_str(), &cnf) != 0) )
    {
        fprintf( stderr, "ERROR: backend's compilation fails\n\n");
        //fprintf( stderr, "  lccrt_module_compile_asm : %p\n", (void *)&lccrt_module_compile_asm);
        //fprintf( stderr, "    module   : [%p]\n", (void *)m);
        //fprintf( stderr, "    lib_name : [%s]\n", alib.c_str());
        //fprintf( stderr, "    cnf      : [%p]\n", (void *)&cnf);
        fflush( stderr);
        if ( getenv( "LLVM_LCCRT_FAIL_SAVE") )
        {
            std::error_code ec;
            char name[64] = "/tmp/llvm.lccrt.fail.XXXXXX.ll";
            raw_fd_ostream *fs = 0;

            mkstemps( name, 3);
            fs = new raw_fd_ostream( name, ec, sys::fs::OpenFlags::OF_None);
            M.print( *fs, 0);
            delete fs;
            fprintf( stderr, "MODULE: %s\n", name);
            fflush( stderr);
        }

        if ( getenv( "LLVM_LCCRT_FAIL_WAIT") )
        {
            fprintf( stderr, "PROCESS: %d\n", getpid());
            fflush( stderr);
            while ( 1 )
            {
                sleep( 10);
            }
        }

        //fprintf( stderr, "\n\n");
        abort();
        return (true);
    }

    le.deleteModule();
    llvm::SetPrintStack( true);

    return (r);
} /* LccrtPass::runOnModule */

MCCodeEmitter *
Lccrt::createMCCodeEmitter( const Triple &triple)
{
    MCCodeEmitter *r = new LccrtMCCodeEmitter();

    return (r);
} /* Lccrt::createMCCodeEmitter */

MCAsmBackend *
Lccrt::createMCAsmBackend( const Triple &triple)
{
    MCAsmBackend *r = new LccrtMCAsmBackend();

    return (r);
} /* Lccrt::createMCAsmBackend */
#endif /* LLVM_WITH_LCCRT */
