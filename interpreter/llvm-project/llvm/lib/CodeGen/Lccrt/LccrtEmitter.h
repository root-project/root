//=- llvm/CodeGen/Lccrt/LccrtEmitter.h - Lccrt-IR translation -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LccrtEmitter class, which
// implements translation from LLVM-IR to LCCRT-IR.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_LCCRT_LCCRTEMITTER_H
#define LLVM_LIB_CODEGEN_LCCRT_LCCRTEMITTER_H

#include "llvm/Support/Compiler.h"
#include "llvm/ADT/bit.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Pass.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCMachObjectWriter.h"
#include "LccrtDbg.h"
#include "LccrtDbglines.h"
#include "LccrtIpa.h"

#ifdef LLVM_WITH_LCCRT
#include "lccrt.h"

#include <map>

/**
 * Указатель на аргумент операции.
 */
class arg_ref_t
{
  public:
    lccrt_oper_ptr oper; /* операция */
    unsigned arg_num; /* номер аргумента операции */
    arg_ref_t() : oper( 0), arg_num( 0) {}
    arg_ref_t( lccrt_oper_ptr voper, int varg_num) : oper( voper), arg_num( varg_num) {}
};

//#define ARG_REF( op, an) ({ arg_ref_t aref = {(op), (an)}; aref; })
#define ARG_REF( op, an) (arg_ref_t( op, an))

namespace llvm {
class Triple;
class DataLayout;
class GlobalAlias;
class MCAsmLayout;

class LLVM_LIBRARY_VISIBILITY LccrtContext
{
    lccrt_context_ptr ctx;
  public:
    LccrtContext();
    virtual ~LccrtContext();
    static lccrt_context_ptr get();
    static std::string getLibName( const Triple &);
    static std::string getArchName( const Triple &);
  private:
    operator lccrt_context_ptr() const { return (ctx); }
};

/* This class is intended to be used as a driving class for all lccrt writers. */
class LLVM_LIBRARY_VISIBILITY LccrtPass : public ModulePass
{
  public:
    TargetMachine &TM;
    raw_pwrite_stream &Out; /* Stream for output assembler. */
    CodeGenFileType OutType;

  public:
    static char ID;

  public:
    //LccrtPass() : ModulePass( ID), TM(*(TargetMachine*)0), Out(*(raw_pwrite_stream*)0) {}
    explicit LccrtPass( TargetMachine &TM, raw_pwrite_stream &vOut, CodeGenFileType type);
    virtual ~LccrtPass() override;

    virtual bool runOnModule( Module &M) override;
    virtual void getAnalysisUsage( AnalysisUsage &Info) const override;
    bool extCompile( Module &M, std::string llc);
    void addModuleIRLogging( Module &M);
    void addFunctionIRLogging( ModuleSlotTracker &MST, Function &F, Function *F_logir_v);
    Instruction *findBBlockHead( BasicBlock &BB);
};

typedef struct
{
    lccrt_type_ptr type;
    lccrt_varinit_ptr vi;
} TypeVarinit;

struct CompareTypeVarinit
{
    bool operator()( const TypeVarinit &tv0, const TypeVarinit &tv1) const
    {
        bool r;

        if ( (tv0.type == tv1.type) )
        {
            if ( lccrt_varinit_is_zero_or_hex( tv0.vi)
                 && lccrt_varinit_is_zero_or_hex( tv1.vi) )
            {
                r = lccrt_varinit_get_zero_or_hex64( tv0.vi) < lccrt_varinit_get_zero_or_hex64( tv1.vi);

            } else if ( lccrt_varinit_is_zero_or_hex( tv0.vi) )
            {
                r = true;

            } else if ( lccrt_varinit_is_zero_or_hex( tv1.vi) )
            {
                r = false;

            } else if ( lccrt_varinit_is_addr_var( tv0.vi)
                        && lccrt_varinit_is_addr_var( tv1.vi) )
            {
                lccrt_var_ptr v0 = lccrt_varinit_get_addr_var( tv0.vi);
                lccrt_var_ptr v1 = lccrt_varinit_get_addr_var( tv1.vi);
                uint64_t n0 = lccrt_varinit_get_num_elems( tv0.vi);
                uint64_t n1 = lccrt_varinit_get_num_elems( tv1.vi);

                r = (v0 == v1) ? (n0 < n1) : ((uintptr_t)v0 < (uintptr_t)v1);

            } else if ( lccrt_varinit_is_addr_var( tv0.vi) ) 
            {
                r = true;

            } else if ( lccrt_varinit_is_addr_var( tv1.vi) )
            {
                r = false;

            } else if ( lccrt_varinit_is_addr_func( tv0.vi)
                        && lccrt_varinit_is_addr_func( tv1.vi) )
            {
                lccrt_f_ptr f0 = lccrt_varinit_get_addr_func( tv0.vi);
                lccrt_f_ptr f1 = lccrt_varinit_get_addr_func( tv1.vi);
                uint64_t n0 = lccrt_varinit_get_num_elems( tv0.vi);
                uint64_t n1 = lccrt_varinit_get_num_elems( tv1.vi);

                r = (f0 == f1) ? (n0 < n1) : ((uintptr_t)f0 < (uintptr_t)f1);

            } else if ( lccrt_varinit_is_addr_func( tv0.vi) ) 
            {
                r = true;

            } else if ( lccrt_varinit_is_addr_func( tv1.vi) )
            {
                r = false;
            } else
            {
                r = (uintptr_t)tv0.vi < (uintptr_t)tv1.vi;
            }
        } else
        {
            r = (uintptr_t)tv0.type < (uintptr_t)tv1.type;
        }

        return (r);
    }
};

typedef std::map<const GlobalVariable *, const lccrt_v_ptr> MapGVToVar;
typedef std::map<const GlobalAlias *, const lccrt_v_ptr> MapGAToVar;
typedef std::map<const Function *, const lccrt_function_ptr> MapLLVMFuncToFunc;
typedef std::map<const lccrt_v_ptr, const lccrt_v_ptr> MapVarToVar;
typedef std::map<int64_t, void *> MapIToV;
typedef std::map<const Constant *, const lccrt_varinit_ptr> MapCToVI;
typedef std::map<const std::string, lccrt_function_ptr> MapStringToFunc;
typedef std::map<TypeVarinit, lccrt_var_ptr, CompareTypeVarinit> MapVarinitToConstarg;
typedef std::pair<const GlobalAlias *, const lccrt_v_ptr> PairGAToVar;
typedef std::pair<const Function *, const lccrt_function_ptr> PairLLVMFuncToFunc;
typedef std::pair<const std::string, const lccrt_function_ptr> PairStringToFunc;
typedef std::pair<int, void *> PairIToV;

class LLVM_LIBRARY_VISIBILITY LccrtEmitter {
  public:
    typedef std::map<Type *, lccrt_type_ptr> NamedTypes;
    typedef std::map<const BasicBlock *, int> BBsNums;
    typedef std::map<const Function *, BBsNums> FuncsLbls;

  public:
    lccrt_context_ptr c;
    lccrt_module_ptr m;
    lccrt_einfo_category_t ecat_loop_count;
    lccrt_einfo_category_t ecat_func_attrs;
    lccrt_einfo_category_t ecat_prof;
    lccrt_einfo_tydescr_ptr etyde_i64;
    lccrt_einfo_tydescr_ptr etyde_raw;
    lccrt_einfo_tydescr_ptr etyde_loop_count;
    lccrt_einfo_tydescr_ptr etyde_func_attr;
    lccrt_einfo_tydescr_ptr etyde_func_attrs;
    lccrt_einfo_tydescr_ptr etyde_proffn;
    lccrt_einfo_tydescr_ptr etyde_profct;
    lccrt_einfo_field_id_t eifi_fattr_src;
    lccrt_einfo_field_id_t eifi_fattr_val;
    lccrt_einfo_field_id_t eifi_lcount_val;
    lccrt_einfo_field_id_t eifi_proffn_iv;
    lccrt_einfo_field_id_t eifi_profct_et;
    lccrt_einfo_field_id_t eifi_profct_ef;
    lccrt_einfo_reference_t eref_raw_llvm13;
    std::map<std::string,lccrt_einfo_reference_t> funcs_attrs;
    MapGVToVar gvars;
    MapGAToVar avars;
    MapVarToVar gvar_ptrs;
    MapLLVMFuncToFunc funcs;
    std::map<const Type *, lccrt_type_ptr> types;
    std::map<const Type *, lccrt_type_ptr> dense_types;
    MapCToVI cnsts;
    MapStringToFunc lib_funcs;
    uint64_t name_ident;
    FuncsLbls funcs_lbls;
    MapVarinitToConstarg carg_vars;
    bool verbose_ir;

  public:
    const DataLayout &DL;
    const Triple &archTriple;
    const TargetMachine &TM;
    LccrtModuleIpaEmitter &MIPA;
    ModuleSlotTracker &MST;
    LccrtDbgEmitter dbge_;
    LccrtDbglinesEmitter dbgle_;

  public:
    LccrtEmitter( const TargetMachine &, const DataLayout &, const Triple &archTriple,
                  LccrtModuleIpaEmitter &, ModuleSlotTracker &);
    virtual ~LccrtEmitter();
    lccrt_module_ptr newModule( const Module &M);
    void deleteModule();
    lccrt_module_ptr getModule() { return (m); }
    lccrt_v_ptr makeGlobal( GlobalVariable *GV, bool is_self = false);
    bool makeAppendingGlobal( const GlobalVariable *GV);
    lccrt_v_ptr makeAlias( GlobalAlias *GA);
    bool skipedIntrinsicDeclaration( const Function *F);
    lccrt_function_ptr makeFunction( Function *F);
    lccrt_v_ptr findGlobal( const GlobalVariable *G);
    lccrt_v_ptr findAlias( const GlobalAlias *GA);
    lccrt_function_ptr findFunc( const Function *F);
    lccrt_link_t makeLink( GlobalValue::LinkageTypes lt, GlobalValue::VisibilityTypes vt,
                           GlobalVariable::ThreadLocalMode tlm, int is_cnst, int is_alias,
                           int is_declaration);
    lccrt_f_ptr makeFunctionFast( const char *func_name, lccrt_t_ptr func_type);
    bool testVarinit( const Constant *c, uint64_t shift);
    bool testVarinitExpr( const ConstantExpr *c, uint64_t shift);
    lccrt_varinit_ptr makeVarinit( Constant *c, uint64_t shift);
    lccrt_varinit_ptr makeVarinitExpr( ConstantExpr *c, uint64_t shift);
    lccrt_var_ptr makeVarConst( lccrt_type_ptr type, lccrt_varinit_ptr vi);
    lccrt_var_ptr makeVarConstHex( lccrt_type_ptr type, uint64_t value);
    const GlobalValue *getConstExprGlobal( const Constant *C);
    int getTypeBitsize( Type *Ty);
    bool isTypeFloatNormal( const Type *Ty);
    bool isBitWidthNormal( int width);
    bool isIntBitWidthNormal( Type *Ty);
    bool isIntBitWidthBool( Type *Ty);
    bool isIntBitWidthNormalOrBool( Type *Ty);
    bool isTypeNonStdVector( Type *T);
    bool isVectorBuiltinInt( FixedVectorType *T, int elem_minbitsize);
    int getElementBitsize( Type *T);
    lccrt_type_ptr makeType( Type *T);
    lccrt_type_ptr makeType( Type *T, NamedTypes &ntypes);
    lccrt_type_ptr makeTypeStruct( StructType *ST, NamedTypes &ntypes);
    lccrt_type_ptr makeTypeDenseVector( Type *T);
    lccrt_type_ptr makeValueType( Value *V);
    lccrt_type_ptr makeTypeIntNormal( int bitwidth);
    Type *getValueElementType( Value *V);
    lccrt_function_ptr findLibCall( const std::string &S);
    void insertLibCall( const std::string &S, lccrt_function_ptr f);
    void numberLabels( const Function *F);
    std::string preprocessGlobalName( const GlobalValue *GV);

    static bool isVecUniform( const Value *V);
    static bool isDbgDeclare( const Function *F);
    static bool isDbg( const Function *F);
};

typedef std::map<const Value *, const lccrt_v_ptr> MapValueToVar;
typedef std::map<const BasicBlock *, const lccrt_oper_ptr> MapBBToOper;
typedef std::map<const PHINode *, const lccrt_v_ptr> MapPHINodeToVar;
typedef std::map<const Instruction *, arg_ref_t *> MapInstToArgs;
typedef std::pair<const Value *, const lccrt_v_ptr> PairValueToVar;
typedef std::pair<const BasicBlock *, const lccrt_oper_ptr> PairBBToOper;
typedef std::pair<const PHINode *, const lccrt_v_ptr> PairPHINodeToVar;
typedef std::pair<const Instruction *, arg_ref_t *> PairInstToArgs;

class LLVM_LIBRARY_VISIBILITY LccrtFunctionEmitter {
  public:
    typedef std::vector<lccrt_var_ptr> VecVars;
    typedef std::map<lccrt_type_ptr, VecVars *> MapTypeVecVars;
  private:
    LccrtEmitter &le;
    lccrt_function_ptr f;
    lccrt_module_ptr m;
    lccrt_context_ptr c;
    Function *F;
    MapValueToVar lvals;
    MapBBToOper lbls;
    MapTypeVecVars locals_type_pool; /* пул временных переменных, разделенных по типам */
    int num_lvals; /* количество локальных переменных (без имени) */
    int num_cnsts; /* количество локальных констант */
    int num_exprs; /* количество константных выражений */
    int num_alloc; /* количество локальных переменных (для alloca) */
    int num_rvars; /* количество локальных переменных (для временных значений) */
    LccrtFunctionIpaEmitter FIPA;
    LccrtDbgEmitter &dbge_;
    LccrtDbglinesEmitter &dbgle_;
  public:
    LccrtFunctionEmitter( LccrtEmitter &ale, lccrt_function_ptr af, Function *aF,
                          LccrtModuleIpaEmitter &MIPA, LccrtDbgEmitter &dbge,
                          LccrtDbglinesEmitter &dbgel);
    static std::string lowerCallName( lccrt_m_ptr m, const char *name, lccrt_t_ptr *type, int is_test = 0);
    static lccrt_cmp_name_t getCmpLccrtName( unsigned opcode, const char **ps = 0);
    static lccrt_t_ptr makeLibCallType( lccrt_m_ptr m, int num_args, bool is_vec);
    bool isFastLibCall( const User *O, const char *file, int line);
    Value *getShuffleMaskVector( const User &O);
    void makeArgs( lccrt_type_ptr types[]);
    void makeOpers();            
    lccrt_v_ptr makeNIntLocalVar( int bitsize);
    lccrt_v_ptr makeValue( Value *V, lccrt_oi_ptr i = 0, bool noncarg = false);
    lccrt_v_ptr makeValueConstExpr( lccrt_v_ptr v, ConstantExpr *E, lccrt_oi_ptr i);
    lccrt_v_ptr makeValueConst( lccrt_v_ptr v, Constant *C, lccrt_oi_ptr i);
    lccrt_v_ptr makeValuePtrcast( Value *V, lccrt_t_ptr rtype, lccrt_oi_ptr i);
    lccrt_oper_ptr findBlockLabel( const BasicBlock *BB);
    lccrt_oper_ptr makeArith1( unsigned opcode, Instruction *O,
                               lccrt_v_ptr a1, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeArith1( unsigned opcode, User &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    lccrt_oper_ptr makeArith2( unsigned opcode, Instruction *O,
                               lccrt_v_ptr a1, lccrt_v_ptr a2, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeArith2( unsigned opcode, User &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeBswap( User &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeCtpop( User &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeCttz( User &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeCtlz( User &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeFptoiSat( User &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeFrexp( User &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeFshl( User &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeFshr( User &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeFmuladd( User &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeSqrt( User &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeIntAbs( User &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeIntMinMax( std::string mname, User &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeBitrev( User &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeVectorReduce( const User &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    lccrt_v_ptr makeVarRes( lccrt_type_ptr type);
    void makeLibCall( const char *func_name, bool is_vec, const User &O, lccrt_v_ptr res,
                      bool ignore_last_arg, lccrt_oi_ptr i);
    void makeLibCallFast( const char *func_name, const User &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeLibCallFast( const char *func_name, lccrt_v_ptr a0, lccrt_v_ptr a1, lccrt_v_ptr a2,
                          lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeLibCallFast( const char *func_name, const std::vector<lccrt_var_ptr> &iargs,
                          lccrt_var_ptr res, lccrt_oi_ptr i);
    int makeBranch( Instruction &O, lccrt_oper_ptr &ct, lccrt_v_ptr res,
                    arg_ref_t *alts_to_opers, lccrt_oi_ptr i);
    int makeIndirectBranch( Instruction &O, lccrt_oper_ptr &ct, lccrt_v_ptr res,
                            arg_ref_t *alts_to_opers, lccrt_oi_ptr i);
    void makeCall( Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    int makeInvoke( Instruction &O, lccrt_oper_ptr &ct, lccrt_v_ptr res,
                    arg_ref_t *alts_to_opers, lccrt_oi_ptr i);
    void makeLandingpad( const Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeResume( const Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    std::string evalAsmConstraint( std::string ac);
    std::string evalAsmConstraintVector( const InlineAsm::ConstraintCodeVector &Codes);
    bool parseInt( const char *p, int &len, int64_t &value);
    bool parseAsmInlineArg( const char *p, int &len, std::string &arg);
    void makeAsmInline( const CallInst &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeReadWriteRegister( const CallInst &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    lccrt_v_ptr makeCallBuiltinAddr( const Function *F, lccrt_type_ptr type, const char *name);
    lccrt_v_ptr makeCallBuiltinAddr( const Function *F, Type *T, const char *name, int num_args);
    void makeCmp( const Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeLoadStore( Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeVecbitRepack( Instruction &O, Type *T, bool ispack,
                           lccrt_var_ptr a, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeVecbitpack( Instruction &O, const char *opname,
                         Type *T1, Type *T2, int elem_bitsize,
                         lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeBitcast( Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeCast( Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeGetelementptr( User &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeAlloca( Instruction &O, lccrt_v_ptr res, int is_start, lccrt_oi_ptr i);
    void makeSelect( const Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeExtractvalue( const Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeInsertvalue( const Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeExtractelement( Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeInsertelement( const Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeShufflevector( const Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeVaArg( const Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeFence( const Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeCmpXchg( const Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makeAtomicrmw( const Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i);
    void semiUnzipLandingpad( const BasicBlock *BB0, MapInstToArgs &oas, lccrt_o_ptr lbl_work, lccrt_oi_ptr i);
    void makePhi( Instruction &O, MapInstToArgs &oas, lccrt_o_ptr lbl_work, lccrt_v_ptr res, lccrt_oi_ptr i);
    void makePhiArg( lccrt_o_ptr lbl_work, Value *PV, lccrt_v_ptr next_res, lccrt_o_ptr ct, int ct_num, lccrt_oi_ptr i);
    lccrt_oper_ptr makeBitcastNIntToInt( int bitsize, lccrt_v_ptr arg, lccrt_v_ptr res, lccrt_oi_ptr i);
    lccrt_oper_ptr makeBitcastIntToNInt( int bitsize, lccrt_v_ptr arg, lccrt_v_ptr res, lccrt_oi_ptr i);
    lccrt_var_ptr makeExtInt( lccrt_var_ptr a, int bitwidth, bool sign, lccrt_oper_iterator_ptr i);
    lccrt_var_ptr makeExtNIntToInt( lccrt_var_ptr a, int bitwidth, bool sign, lccrt_oper_iterator_ptr i);
    lccrt_type_ptr makeDoublingStruct( lccrt_type_ptr type_long, lccrt_type_ptr type_short);
    lccrt_var_ptr allocLocalTmpVar( lccrt_type_ptr type);
    lccrt_var_ptr makeBitcastVar( Type *TA, Type *TR);
    lccrt_var_ptr makeBitcastPtr( lccrt_var_ptr v, lccrt_type_ptr rt, lccrt_oi_ptr i);
    void freeLocalTmpVar( lccrt_var_ptr v);
    void evalFuncProfile( const Function *F, lccrt_function_ptr f);
    void generateComment( const Instruction &O, lccrt_oper_iterator_ptr i);
    void evalBranchProfile( const BranchInst &BI, lccrt_oper_ptr ct);
  private:
    static thread_local int64_t fast_lib_cur_;
    static thread_local int64_t fast_lib_bnd_;
};

class LLVM_LIBRARY_VISIBILITY LccrtMCCodeEmitter : public MCCodeEmitter
{
  public:

    /// EncodeInstruction - Encode the given \p Inst to bytes on the output
    /// stream \p OS.
    void encodeInstruction(const MCInst &Inst, SmallVectorImpl<char> &CB,
                           SmallVectorImpl<MCFixup> &Fixups,
                           const MCSubtargetInfo &STI) const override;
};

class LLVM_LIBRARY_VISIBILITY LccrtMCAsmBackend : virtual public MCAsmBackend
{
  public:
    LccrtMCAsmBackend() : MCAsmBackend(llvm::endianness::little) {}

    /// Create a new MCObjectWriter instance for use by the assembler backend to
    /// emit the final object file.
    virtual std::unique_ptr<MCObjectWriter> createObjectWriter( raw_pwrite_stream &OS) const;

    /// \name Target Fixup Interfaces
    /// @{

    /// Get the number of target specific fixup kinds.
    unsigned getNumFixupKinds() const override;

    /// Apply the \p Value for given \p Fixup into the provided data fragment, at
    /// the offset specified by the fixup and following the fixup kind as
    /// appropriate. Errors (such as an out of range fixup value) should be
    /// reported via \p Ctx.
    /// The  \p STI is present only for fragments of type MCRelaxableFragment and
    /// MCDataFragment with hasInstructions() == true.
    void applyFixup(const MCAssembler &Asm, const MCFixup &Fixup,
                    const MCValue &Target, MutableArrayRef<char> Data,
                    uint64_t Value, bool IsResolved, const MCSubtargetInfo *STI) const override;

    /// @}

    /// \name Target Relaxation Interfaces
    /// @{

    /// Check whether the given instruction may need relaxation.
    ///
    /// \param Inst - The instruction to test.
    /// \param STI - The MCSubtargetInfo in effect when the instruction was
    /// encoded.
    bool mayNeedRelaxation(const MCInst &Inst, const MCSubtargetInfo &STI) const override;

    /// Simple predicate for targets where !Resolved implies requiring relaxation
    bool fixupNeedsRelaxation(const MCFixup &Fixup, uint64_t Value,
                              const MCRelaxableFragment *DF,
                              const MCAsmLayout &Layout) const override;

    /// Relax the instruction in the given fragment to the next wider instruction.
    ///
    /// \param Inst The instruction to relax, which may be the same as the
    /// output.
    /// \param STI the subtarget information for the associated instruction.
    /// \param [out] Res On return, the relaxed instruction.
    virtual void relaxInstruction(const MCInst &Inst, const MCSubtargetInfo &STI,
                                  MCInst &Res) const;

    /// @}

    /// Returns the minimum size of a nop in bytes on this target. The assembler
    /// will use this to emit excess padding in situations where the padding
    /// required for simple alignment would be less than the minimum nop size.
    ///
    unsigned getMinimumNopSize() const override { return 8; }

    /// Write an (optimal) nop sequence of Count bytes to the given output. If the
    /// target cannot generate such a sequence, it should return an error.
    ///
    /// \return - True on success.
    bool writeNopData(raw_ostream &OS, uint64_t Count, const MCSubtargetInfo *STI) const override;

    std::unique_ptr<MCObjectTargetWriter> createObjectTargetWriter() const override;
};

class LLVM_LIBRARY_VISIBILITY LccrtMCMachObjectTargetWriter : public MCMachObjectTargetWriter
{
  public:
    LccrtMCMachObjectTargetWriter();
    void recordRelocation( MachObjectWriter *Writer, MCAssembler &Asm, const MCAsmLayout &Layout,
                           const MCFragment *Fragment, const MCFixup &Fixup, MCValue Target,
                           uint64_t &FixedValue) override;
};
}
#endif /* LLVM_WITH_LCCRT */

#endif /* LLVM_LIB_CODEGEN_LCCRT_LCCRTEMITTER_H */
