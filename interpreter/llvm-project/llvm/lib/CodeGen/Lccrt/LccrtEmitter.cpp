//===-- LccrtEmitter.cpp - Lccrt emittion code ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LccrtEmitter class.
//
//===----------------------------------------------------------------------===//

#include <string.h>

#include <iostream>
#include <iomanip>

#include "LccrtEmitter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Operator.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/CodeGen/Lccrt.h"

#ifdef LLVM_WITH_LCCRT
using namespace llvm;

#define DEBUG_TYPE "lccrt"

#define isFastLibCallLine( O) isFastLibCall( &(O), __FILE__, __LINE__)

#ifdef NDEBUG
#define assert_define( x)
#else /* !NDEBUG */
#define assert_define( x) x
#endif /* NDEBUG */

#define is_str( s, t) (strcmp( s, t) == 0)
#define is_str_n( s, t, tn) (strncmp( s, t, tn) == 0)
#define is_str_head( s, t) is_str_n( s, t, strlen( t))

#define errorDumpHead( v) \
  { \
      fprintf( stderr, "  %s: %s:%d\n", __FUNCTION__, __FILE__, __LINE__); \
      fflush( stderr); \
      errs() << "\n" << *(v) << "\n\n"; \
  }
#define errorDump( v) \
  { \
      fprintf( stderr, "IRLite: error: fails to translate llvm-IR\n"); \
      errorDumpHead( v); \
      abort(); \
  }
#define errorDump2( v1, v2) \
  { \
      errorDumpHead( v1); \
      errorDumpHead( v2); \
      abort(); \
  }

/**
 * Информация об одном компоненте суффиксе функции.
 */
struct MathSuffixInfo
{
    int veclen; // количество элементов вектора или 0
    int bitlen; // битовый размер элемента
    bool is_float; // флаг плавающего значения
};

static uint32_t
get_floor2( uint32_t v)
{
    uint32_t r = v ? 1 : 0;

    while ( (2*r <= v) )
    {
        r = 2*r;
    }

    return (r);
} /* get_floor2 */

/**
 * Проверка значения на равенство степени числа 2.
 */
static bool
is_pow2( uint64_t v) {
    bool r = (__builtin_popcountll( v) == 1);

    return (r);
} /* is_pow2 */

static int
is_number( const char *s)
{
    char *p;
    int flag = 0;

    strtol( s, &p, 10);
    flag = (p[0] == 0);

    return (flag);
} /* is_number */

static bool
is_e2k_reg( const char *s0, int &roff) {
    bool r = false;
    int k = 2;
    const char *s = s0 + roff;

    if ( (s[0] == 'g')
         || (s[0] == 'r')
         || (s[0] == 'b') ) {
        if ( isdigit( s[1]) ) {
            if ( isdigit( s[k]) ) ++k;
            if ( isdigit( s[k]) ) ++k;
            if ( (k == 2)
                 || (s[1] != '0') ) {
                r = true;
                roff += k;
            }
        }
    }

    return (r);
} /* is_e2k_reg */

/**
 * Выравнивание величины на указанное значение.
 */
static int64_t
get_align_size( int64_t size, int64_t align)
{
    int64_t r = ((size + align - 1) / align) * align;

    return (r);
} /* get_align_size */

/**
 * Получение байтового выравнивания для байтового размера типа iN.
 */
static int64_t
get_int_align( int64_t bytesize)
{
    int r = 0;

    if ( (bytesize < 0) )
    {
        assert( 0);

    } else if ( (bytesize <= 2) )
    {
        r = bytesize;

    } else if ( (bytesize <= 4) )
    {
        r = 4;
    } else
    {
        r = 8;
    }

    return (r);
} /* get_int_align */

/**
 * Выравнивание байтового размера на 1, 2, 4 или 8 байт в соответствии с
 * величиной входноного значения.
 */
static int64_t
get_int_align_size( int64_t bytesize)
{
    int r = get_align_size( bytesize, get_int_align( bytesize));

    return (r);
} /* get_int_align_size */

/**
 * Вычислить для операции количество обычных альтернативых перехода.
 */
static int
get_num_normal_alts( const llvm::Instruction &O)
{
    int r = 0;

    if ( isa<BranchInst>(O) )
    {
        const BranchInst &BI = cast<BranchInst>(O);

        r = BI.isConditional() ? 2 : 1;

    } else if ( isa<SwitchInst>(O) )
    {
        const SwitchInst &SI = cast<SwitchInst>(O);

        r = SI.getNumCases() + 1;

    } else if ( isa<InvokeInst>(O) )
    {
        r = 2;

    } else if ( auto IBI = dyn_cast<IndirectBrInst>( &O) ) 
    {
        r = IBI->getNumDestinations() + 1;
    }

    return (r);
} /* get_num_normal_alts */

/**
 * Проверяем, что символ может быть напечатано в ассемблерном файле
 * (без использования кавычек).
 */
static bool
is_char_asm_printable( char c, bool is_quoted = false)
{
    bool r = true;

    if ( !(isalnum( c)
           || (c == '_')
           || (c == '$')
           || (c == '.')
           || (is_quoted
               && (c == ' '))) )
    {
        r = false;
    }

    return (r);
} /* is_char_asm_printable */

/**
 * Проверяем, что имя может быть напечатано в ассемблерном файле
 * (без использования кавычек).
 */
static bool
is_name_asm_printable( const char *s, bool is_quoted = false)
{
    bool r = true;

    if ( !s
         || (s[0] == 0) )
    {
        r = false;
    } else
    {
        for ( int i = 0; s[i]; ++i )
        {
            if ( !is_char_asm_printable( s[i], is_quoted) )
            {
                r = false;
                break;
            }
        }
    }

    return (r);
} /* is_name_asm_printable */

/**
 * Применение манглирования к имени для процесса ассемблирования,
 * если задан соответствующий режим.
 */
static std::string
apply_lccrt_mangling( std::string s)
{
    std::string r;
    static bool is_mangling = false;
    static bool is_init = false;

    if ( !s.empty()
         && (s[0] == 1) )
    {
        s.erase( 0, 1);
    }

    if ( !is_init )
    {
        const char *v = getenv( "LLVM_LCCRT_NAME_MANGLING");

        is_init = true;
#ifdef LLVM_LCCRT_RELEASE
        is_mangling = !v || atoi( v);
#else /* !LLVM_LCCRT_RELEASE */
        is_mangling = v && atoi( v);
#endif /* LLVM_LCCRT_RELEASE */
    }

    if ( is_mangling
         && !is_name_asm_printable( s.c_str(), false) )
    {
        char b[16];
        const char *p = s.c_str();

        r = "__llvm_lccrt_mangling_";
        for ( int i = 0; p[i] ; ++i )
        {
            if ( (p[i] == '$') )
            {
                r += "$$";

            } else if ( is_char_asm_printable( p[i], false) )
            {
                r += p[i];
            } else
            {
                snprintf( b, 16, "$%02x", (uint8_t)p[i]);
                r += b;
            }
        }
    } else
    {
        r = s;
    }

    return (r);
} /* apply_lccrt_mangling */

static bool
parse_math_suffix( const char *suffix, std::vector<MathSuffixInfo> &rvec)
{
    int i = 0;
    bool r = (suffix[0] != 0);

    rvec.clear();
    while ( suffix[i] )
    {
        if ( (suffix[i] == '.') )
        {
            MathSuffixInfo si = {};

            ++i;
            if ( (suffix[i] == 'v') )
            {
                char *p;

                ++i;
                si.veclen = strtol( suffix + i, &p, 10);
                if ( (si.veclen > 0)
                     && (p > suffix + i) )
                {
                    i += p - (suffix + i);
                } else
                {
                    r = false;
                    break;
                }
            }

            if      ( is_str_n( suffix + i, "f32",  3) ) { i += 3; si.bitlen = 32; si.is_float = true; }
            else if ( is_str_n( suffix + i, "f64",  3) ) { i += 3; si.bitlen = 64; si.is_float = true; }
            else if ( is_str_n( suffix + i, "f80",  3) ) { i += 3; si.bitlen = 80; si.is_float = true; }
            else if ( is_str_n( suffix + i, "i8",   2) ) { i += 2; si.bitlen = 8; }
            else if ( is_str_n( suffix + i, "i16",  3) ) { i += 3; si.bitlen = 16; }
            else if ( is_str_n( suffix + i, "i32",  3) ) { i += 3; si.bitlen = 32; }
            else if ( is_str_n( suffix + i, "i64",  3) ) { i += 3; si.bitlen = 64; }
            else if ( is_str_n( suffix + i, "i128", 4) ) { i += 4; si.bitlen = 128; }
            else                                         { r = false; break; }

            rvec.push_back( si);
        } else
        {
            r = false;
            break;
        }
    }

    if ( !r )
    {
        rvec.clear();
    }

    return (r);
} /* parse_math_suffix */

static bool
is_name_suff( const char *name, const char *head, const char *suffixes[])
{
    bool r = false;
    int hlen = strlen( head);

    if ( (strncmp( name, head, hlen) == 0) )
    {
        for ( int i = 0; suffixes[i]; ++i )
        {
            if ( (strcmp( name + hlen, suffixes[i]) == 0) )
            {
                r = true;
                break;
            }
        }
    }

    return (r);
} /* is_name_suff */

static bool
is_name_suff_stdint_bool( const char *name, const char *head)
{
    static const char *a[] = {".i1", ".i8", ".i16", ".i32", ".i64", ".i128", 0};
    bool r = is_name_suff( name, head, a);

    return (r);
} /* is_name_suff_stdint_bool */

static bool
is_name_suff_stdint( const char *name, const char *head)
{
    static const char *a[] = {".i8", ".i16", ".i32", ".i64", ".i128", 0};
    bool r = is_name_suff( name, head, a);

    return (r);
} /* is_name_suff_stdint */

static bool
is_name_suff_stdfloat( const char *name, const char *head)
{
    static const char *a[] = {".f32", ".f64", ".f80", 0};
    bool r = is_name_suff( name, head, a);

    return (r);
} /* is_name_suff_stdfloat */

static bool
is_name_math_vec( std::string call_name, std::string &math_name, int &num_args)
{
    bool r = false;
    const char *cn = (const char *)(call_name.c_str());

    math_name = "";
    if ( is_str_head( cn, "llvm.") )
    {
        int i;
        #define DEF_CNTF( s, na, t) {s, s ".", t, (int)strlen( s "."), na}
        #define DEF_CNTS( s, na) DEF_CNTF( s, na, s)
        struct
        {
            const char *math_name;
            const char *name_dot;
            const char *new_math_name;
            int dot_len;
            int num_args;
        } cnt[] =
        {
            DEF_CNTS( "fma",        3),
            DEF_CNTS( "floor",      1),
            DEF_CNTS( "ceil" ,      1),
            DEF_CNTS( "round",      1),
            DEF_CNTF( "trunc",      1, "ftrunc"),
            DEF_CNTS( "rint",       1),
            DEF_CNTS( "nearbyint",  1),
            DEF_CNTS( "fabs",       1),
            DEF_CNTS( "exp2",       1),
            DEF_CNTS( "exp",        1),
            DEF_CNTS( "pow",        2),
            DEF_CNTS( "powi",       2),
            DEF_CNTS( "sqrt",       1),
            DEF_CNTS( "log",        1),
            DEF_CNTS( "log2",       1),
            DEF_CNTS( "log10",      1),
            DEF_CNTS( "cos",        1),
            DEF_CNTS( "sin",        1),
            DEF_CNTS( "tan",        1),
            DEF_CNTS( "asin",       1),
            DEF_CNTS( "acos",       1),
            DEF_CNTS( "minnum",     2),
            DEF_CNTS( "maxnum",     2),
            DEF_CNTF( "uadd.sat",   2, "uadd_sat"),
            DEF_CNTF( "sadd.sat",   2, "sadd_sat"),
            DEF_CNTF( "usub.sat",   2, "usub_sat"),
            DEF_CNTF( "ssub.sat",   2, "ssub_sat"),
            DEF_CNTF( "is.fpclass", 2, "isfpclass"),
            {0, 0, 0, 0}
        };
        #undef DEF_CNTS
        #undef DEF_CNTF

        cn += strlen( "llvm.");
        for ( i = 0; cnt[i].math_name; ++i )
        {
            if ( is_str_n( cn, cnt[i].name_dot, cnt[i].dot_len) )
            {
                break;
            }
        }

        if ( cnt[i].math_name )
        {
            std::vector<MathSuffixInfo> sinfo;

            if ( parse_math_suffix( cn + cnt[i].dot_len - 1, sinfo) )
            {
                if ( is_str( cnt[i].math_name, "powi") )
                {
                    if ( (sinfo.size() == 2)
                         && (sinfo[0].veclen >= 1)
                         && (sinfo[1].veclen == 0)
                         && (sinfo[1].bitlen == 32)
                         && !sinfo[1].is_float )
                    {
                        r = true;
                    }
                } else
                {
                    if ( (sinfo.size() == 1)
                         && (sinfo[0].veclen >= 1) )
                    {
                        r = true;
                    }
                }

                if ( r )
                {
                    num_args = cnt[i].num_args;
                    math_name = cnt[i].new_math_name;
                }
            }
        }
    }

    return (r);
} /* is_name_math_vec */

LccrtEmitter::LccrtEmitter( const TargetMachine &vTM, const DataLayout &vDL, const Triple &vTriple,
                            LccrtModuleIpaEmitter &vMIPA, ModuleSlotTracker &vMST)
    : m(0), name_ident( 0), DL( vDL), archTriple( vTriple), TM( vTM), MIPA( vMIPA), MST( vMST)
{
    const char *env_verbose_ir = getenv( "LLVM_LCCRT_VERBOSE_IR");

    c = LccrtContext::get();
    verbose_ir = env_verbose_ir ? atoi( env_verbose_ir) : 0;

    return;
} /* LccrtEmitter::LccrtEmitter */

LccrtEmitter::~LccrtEmitter()
{
    return;
} /* LccrtEmitter::LccrtEmitter */

lccrt_module_ptr
LccrtEmitter::newModule( const Module &M)
{
    const char *lc_names[2] = {"value"};
    const char *fa_names[2] = {"src", "value"};
    const char *pf_names[1] = {"invalue"};
    const char *pa_names[2] = {"edgetrue", "edgefalse"};
    lccrt_eitd_ptr fa_types[2] = {};
    lccrt_eitd_ptr pa_types[2] = {};
    const char *name = M.getName().data();

    deleteModule();

    m = lccrt_module_new( c, name, DL.getPointerSize() == 4);
    ecat_loop_count = lccrt_module_new_einfo_category( m, "loop_count");
    ecat_func_attrs = lccrt_module_new_einfo_category( m, "func_attrs");
    ecat_prof = lccrt_module_new_einfo_category( m, "profile");

    etyde_i64 = lccrt_einfo_make_tydescr_i64( m);
    etyde_raw = lccrt_einfo_make_tydescr_raw( m);

    etyde_loop_count = lccrt_einfo_make_tydescr_struct( m, "loop_count.value", 1, lc_names, &etyde_i64);
    eifi_lcount_val = lccrt_einfo_find_tydescr_field( etyde_loop_count, "value");

    fa_types[0] = etyde_raw;
    fa_types[1] = etyde_raw;
    etyde_func_attr = lccrt_einfo_make_tydescr_struct( m, "func_attrs.value", 2, fa_names, fa_types);
    etyde_func_attrs = lccrt_einfo_make_tydescr_array( m, etyde_func_attr);
    eifi_fattr_src = lccrt_einfo_find_tydescr_field( etyde_func_attr, "src");
    eifi_fattr_val = lccrt_einfo_find_tydescr_field( etyde_func_attr, "value");

    pa_types[0] = etyde_i64;
    etyde_proffn = lccrt_einfo_make_tydescr_struct( m, "profile.funcentry", 1, pf_names, pa_types);
    eifi_proffn_iv = lccrt_einfo_find_tydescr_field( etyde_proffn, "invalue");

    pa_types[0] = etyde_i64;
    pa_types[1] = etyde_i64;
    etyde_profct = lccrt_einfo_make_tydescr_struct( m, "profile.branchif", 2, pa_names, pa_types);
    eifi_profct_et = lccrt_einfo_find_tydescr_field( etyde_profct, "edgetrue");
    eifi_profct_ef = lccrt_einfo_find_tydescr_field( etyde_profct, "edgefalse");

    eref_raw_llvm13 = lccrt_einfo_new_raw_by_string( m, "llvm-13");

    return (m);
} /* LccrtEmitter::newModule */

void
LccrtEmitter::deleteModule()
{
    gvars.clear();
    avars.clear();
    gvar_ptrs.clear();
    funcs.clear();
    types.clear();
    cnsts.clear();
    if ( m )
    {
        lccrt_module_delete( m);
        m = 0;
    }

    return;
} /* LccrtEmitter::deleteModule */

/**
 * Присвоить каждой метке функции идентификатор, который будет использоваться
 * в качестве blockaddress.
 */
void
LccrtEmitter::numberLabels( const Function *F)
{
    BBsNums sn;
    int k = 0;

    for ( auto bi = F->begin() ; bi != F->end() ; ++bi )
    {
        const BasicBlock *BB = &(*bi);

        sn[BB] = k;
        ++k;
    }

    funcs_lbls[F] = sn;

    return;
} /* LccrtEmitter::numberLabels */

/**
 * Возврат имени объекта или формирование имени для static-объекта без имени.
 */
std::string
LccrtEmitter::preprocessGlobalName( const GlobalValue *GV)
{
    std::string r = GV->getName().str();
    bool is_static = GV->hasInternalLinkage() || GV->hasPrivateLinkage();

    r = apply_lccrt_mangling( r);

    if ( !GV->hasName()
         || (is_static
             && !is_name_asm_printable( r.c_str(), true)) )
    {
        char bf[256];

        snprintf( bf, 256, "%jd", name_ident);
        r = "__llvm_lccrt_global_";
        r += bf;
        name_ident++;
    }

    return (r);
} /* LccrtEmitter::preprocessGlobalName */

/**
 * Функция относится к intrinsic-функциям, для которой следует пропустить декларацию.
 */
bool
LccrtEmitter::skipedIntrinsicDeclaration( const Function *F)
{
    bool r = false;
    std::string name = F->getName().str();

    if ( (name == "llvm.read_register.i32")
         || (name == "llvm.read_register.i64")
         || (name == "llvm.write_register.i32")
         || (name == "llvm.write_register.i64") )
    {
        r = true;
    }

    return (r);
} /* LccrtEmitter::skipedIntrinsicDeclaration */

lccrt_function_ptr
LccrtEmitter::makeFunction( Function *F)
{
    lccrt_link_t lnk;
    lccrt_type_ptr t = 0;
    lccrt_function_ptr f = 0;
    MapLLVMFuncToFunc::const_iterator k = funcs.find( F);
    std::string fname = F->hasName() ? F->getName().str() : "";

    if ( (k != funcs.end()) )
    {
        f = k->second;

    } else if ( (fname.find( "llvm.coro.") == 0) )
    {
        t = lccrt_type_make_void( m);
        //t = lccrt_type_make_func( t, 1, &t);
        t = lccrt_type_make_func( t, 0, 0);
        lnk = lccrt_link_get( LCCRT_LINK_BND_WEAK, LCCRT_LINK_VIS_DEFAULT, LCCRT_LINK_TLS_NO, 0, 0);
        f = lccrt_function_new( m, t, "__lccrt_coro_unsupported_yet", 0, lnk, 1, 0);
        funcs.insert( PairLLVMFuncToFunc( F, f));

    } else if ( (fname == "llvm.type.test") )
    {
        t = lccrt_type_make_func( lccrt_type_make_bool( m), 0, 0);
        f = makeFunctionFast( "__lccrt_typetest_unsupported_yet", t);
        funcs.insert( PairLLVMFuncToFunc( F, f));

    } else if ( (fname == "llvm.experimental.noalias.scope.decl") )
    {
        t = lccrt_type_make_func( lccrt_type_make_void( m), 0, 0);
        f = makeFunctionFast( "__lccrt_noalias_scope_unsupported", t);
        funcs.insert( PairLLVMFuncToFunc( F, f));
    } else
    {
        FunctionType *FT = F->getFunctionType();
        int num_args = FT->getNumParams() + FT->isVarArg();
        lccrt_type_ptr *targs = new lccrt_type_ptr[num_args];
        const AttributeList &Attrs = F->getAttributes();
        assert_define( CallingConv::ID cc = F->getCallingConv());
        std::string fname_prc = preprocessGlobalName( F);
        std::string fname_std = LccrtFunctionEmitter::lowerCallName( m, fname_prc.c_str(), &t);
        const char *fname = fname_std.empty() ? 0 : fname_std.c_str();

        assert( (cc == CallingConv::C) || (cc == CallingConv::Fast));
        if ( F->getAlignment() )
        {
            assert( 8 % F->getAlignment() == 0);
        }

        for ( unsigned i = 0, e = FT->getNumParams(); i != e; ++i )
        {
            targs[i] = makeType( FT->getParamType( i));
            if ( Attrs.hasAttributeAtIndex( i + 1, Attribute::ByVal) )
            {
                //targs[i] = lccrt_type_make_ptr_type( targs[i]);
            }
        }

        if ( FT->isVarArg() )
        {
            /* Добавляем тип аргумента '...' */
            targs[num_args - 1] = lccrt_type_make_ellipsis( m);
        }

        /* Создаем тип функции. */
        t = t ? t : lccrt_type_make_func( makeType( FT->getReturnType()), num_args, targs);

        /* Создаем функцию. */
        lnk = makeLink( F->getLinkage(), F->getVisibility(),
                        GlobalVariable::NotThreadLocal, 0, 0, F->isDeclaration());
        f = lccrt_function_new( m, t, fname, 0, lnk, F->isDeclaration(), 0);
        //dbge_.makeFunctionDbgMetadata( f, F);
        dbgle_.makeFunctionDbgMetadata( f, F);
        if ( F->doesNotThrow() )
        {
            lccrt_function_set_attr_does_not_throw( f, 1);
        }

        AttributeSet ats = F->getAttributes().getFnAttrs();
        if ( ats.hasAttributes() ) {
            std::string ats_str = ats.getAsString( true);
            auto fai = funcs_attrs.find( ats_str);

            // Проверяем, что ранее такого набора атрибутов не создавалось.
            if ( fai == funcs_attrs.end() ) {
                int k = 0;
                int arr_len = ats.end() - ats.begin();
                lccrt_einfo_reference_t attrs = lccrt_einfo_new_array( etyde_func_attrs, arr_len);

                for ( auto it : ats ) {
                    lccrt_einfo_reference_t attr;
                    lccrt_einfo_reference_t v0, v1;

                    attr = lccrt_einfo_new_struct( etyde_func_attr);
                    v0 = eref_raw_llvm13;
                    v1 = lccrt_einfo_new_raw_by_string( m, it.getAsString( true).c_str());
                    lccrt_einfo_set_field( attr, eifi_fattr_src, v0);
                    lccrt_einfo_set_field( attr, eifi_fattr_val, v1);
                    lccrt_einfo_set_elem( attrs, k, attr);
                    ++k;
                }

                // Сохраняем новый набор атрибутов.
                funcs_attrs[ats_str] = attrs;
                fai = funcs_attrs.find( ats_str);
            }

            // Устанавливаем для функции набор атрибутов.
            lccrt_function_set_einfo( f, ecat_func_attrs, fai->second);
        }

        if ( F->hasSection() )
        {
            if ( F->hasComdat() )
            {
                if ( F->getSection() == ".text.startup" )
                {
                } else if ( getenv( "LLVM_LCCRT_COMDAT_WITHOUT_SECTION")
                            && !atoi( getenv( "LLVM_LCCRT_COMDAT_WITHOUT_SECTION")) )
                {
                    printf( "\n%s:%s:%d\nllvm-lccrt:error: "
                            "function [%s] has both a section [%s] and a comdat [%s]\n\n",
                            __FUNCTION__, __FILE__, __LINE__, F->getName().data(),
                            F->getSection().data(), F->getComdat()->getName().data());
                    abort();
                }
            } else
            {
                lccrt_function_set_section( f, F->getSection().data());
            }
        }

        if ( (F->getLinkage() == GlobalValue::AvailableExternallyLinkage) )
        {
            lccrt_function_set_attr_extern_inline( f, 1);
        }

        if ( F->hasComdat() )
        {
            const Comdat *com = F->getComdat();

            lccrt_function_set_comdat( f, com->getName().str().c_str());
            if ( (com->getSelectionKind() != Comdat::Any)
                 && (com->getSelectionKind() != Comdat::NoDeduplicate) )
            {
                errorDump( com);
            }
        }

        funcs.insert( PairLLVMFuncToFunc( F, f));
        if ( !F->isDeclaration() )
        {
            LccrtFunctionEmitter lfe( *this, f, F, MIPA, dbge_, dbgle_);

            //F->dump();
            lfe.makeArgs( targs);
            lfe.makeOpers();
        }

        delete[] targs;

        dbge_.makeFunctionDbgMetadata( f, F);
    }

    return (f);
} /* LccrtEmitter::makeFunction */

lccrt_var_ptr
LccrtEmitter::makeGlobal( GlobalVariable *GV, bool is_self)
{
    lccrt_var_loc_t loc;
    lccrt_link_t lnk;
    lccrt_type_ptr t = 0;
    lccrt_var_ptr g = 0;
    lccrt_var_ptr gp = 0;
    lccrt_var_ptr r = 0;
    MapGVToVar::const_iterator k = gvars.find( GV);

    if ( (k != gvars.end()) )
    {
        g = k->second;
        gp = gvar_ptrs.find( g)->second;

    } else if ( GV->hasAppendingLinkage() )
    {
        return (0);
    } else
    {
        lccrt_type_ptr tp = 0;
        std::string name = preprocessGlobalName( GV);

        if ( GV->getType()->getAddressSpace() )
        {
            return (0);
        }

        /* Создаем тип. */
        t = makeType( GV->getValueType());

        if ( GV->hasAvailableExternallyLinkage()
             || (!GV->hasInitializer()
                 && (GV->hasExternalLinkage()
                     || GV->hasExternalWeakLinkage())) )
        {
            assert( GV->isDeclaration());
            loc = LCCRT_VAR_LOC_EXT;
        } else
        {
            assert( !GV->isDeclaration());
            loc = LCCRT_VAR_LOC_GLOB;
        }

        lnk = makeLink( GV->getLinkage(), GV->getVisibility(), GV->getThreadLocalMode(),
                        GV->isConstant(), 0, GV->isDeclaration());
        g = lccrt_var_new( m, loc, t, name.c_str(), 0, lnk, GV->getAlignment());
        // При наличии у глобальной переменной dwarf-информации выполняем ее преобразование.
        dbge_.makeGlobalVariableDbgMetadata( g, GV);

        if ( (GV->getLinkage() == GlobalValue::CommonLinkage) )
        {
            lccrt_var_set_attr_common( g, 1);
        }

        tp = lccrt_type_make_ptr_type( t);
        gp = makeVarConst( tp, lccrt_varinit_new_addr_var( g, 0));

        gvars.insert( std::pair<const GlobalVariable *, lccrt_var_ptr>( GV, g));
        gvar_ptrs.insert( std::pair<lccrt_var_ptr, lccrt_var_ptr>( g, gp));

        if ( !GV->hasAvailableExternallyLinkage()
             && GV->hasInitializer() )
        {
            Constant *c = GV->getInitializer();
            lccrt_varinit_ptr vi = makeVarinit( c, 0);

            if ( (GV->getLinkage() == GlobalValue::CommonLinkage) )
            {
                if ( lccrt_varinit_is_zero( vi)
                     || (lccrt_varinit_is_hex( vi)
                         && (lccrt_varinit_get_hex64( vi) == 0)) )
                {
                } else {
                    assert( 0);
                }
            } else {
                lccrt_var_set_init_value_reduce( g, vi);
            }
        }

        if ( GV->hasSection() && !GV->hasAvailableExternallyLinkage() ) {
            StringRef section = GV->getSection();

            if ( GV->hasComdat() && !section.starts_with( "__llvm_") ) {
                if ( getenv( "LLVM_LCCRT_COMDAT_WITHOUT_SECTION")
                     && !atoi( getenv( "LLVM_LCCRT_COMDAT_WITHOUT_SECTION")) )
                {
                    printf( "\n%s:%s:%d\nllvm-lccrt:error: "
                            "global [%s] has both a section [%s] and a comdat [%s]\n\n",
                            __FUNCTION__, __FILE__, __LINE__, GV->getName().data(),
                            GV->getSection().data(), GV->getComdat()->getName().data());
                    abort();
                }
            } else {
                lccrt_var_set_section( g, section.data());
            }
        }

        if ( GV->hasComdat() && !lccrt_var_get_section( g) ) {
            const Comdat *com = GV->getComdat();

            lccrt_var_set_comdat( g, com->getName().str().c_str());
            if ( (com->getSelectionKind() != Comdat::Any)
                 && (com->getSelectionKind() != Comdat::NoDeduplicate) )
            {
                errorDump( com);
            }            
        }
    }

    r = is_self ? g : gp;

    return (r);    
} /* LccrtEmitter::makeGlobal */

/**
 * Создание глобальной переменной.
 */
bool
LccrtEmitter::makeAppendingGlobal( const GlobalVariable *GV)
{
    bool r = true;
    std::string name = GV->getName().str();
    assert_define( ArrayType *TA = dyn_cast<ArrayType>( GV->getValueType()));

    assert( GV->hasAppendingLinkage());
    if ( (strcmp( name.c_str(), "llvm.used") == 0)
         || (strcmp( name.c_str(), "llvm.compiler.used") == 0) )
    {
        const Constant *C = GV->getInitializer();
        uint64_t n = C->getNumOperands();

        for ( uint64_t j = 0; j < n; ++j )
        {
            const GlobalValue *GV = getConstExprGlobal( dyn_cast<Constant>(C->getOperand( j)));

            if ( isa<const Function>(GV) )
            {
                lccrt_function_ptr f = findFunc( dyn_cast<const Function>(GV));

                assert( f);
                lccrt_function_set_attr_used( f, 1);

            } else if ( isa<const GlobalVariable>(GV) )
            {
                lccrt_var_ptr v = findGlobal( dyn_cast<const GlobalVariable>(GV));

                assert( v);
                lccrt_var_set_attr_used( v, 1);

            } else if ( isa<const GlobalAlias>(GV) )
            {
                lccrt_var_ptr v = findAlias( dyn_cast<const GlobalAlias>(GV));

                assert( v);
                lccrt_var_set_attr_used( v, 1);
            } else
            {
                errorDump( C);
            }
        }
    } else if ( (strcmp( name.c_str(), "llvm.global_ctors") == 0)
                || (strcmp( name.c_str(), "llvm.global_dtors") == 0) )
    {
        lccrt_function_init_type_t itf;
        const Constant *C = GV->getInitializer();
        uint64_t n = C->getNumOperands();

        assert( TA && TA->getElementType()->isStructTy());
        if ( (strcmp( name.c_str(), "llvm.global_ctors") == 0) )
        {
            itf = LCCRT_FUNC_INIT_CTOR;
        } else
        {
            itf = LCCRT_FUNC_INIT_DTOR;
        }

        for ( uint64_t j = 0; j < n; ++j )
        {
            const ConstantStruct *CS = dyn_cast<const ConstantStruct>( C->getOperand( j));
            const Function *F = dyn_cast<const Function>( CS->getOperand( 1));
            MapLLVMFuncToFunc::const_iterator k = funcs.find( F);

            if ( (k == funcs.end()) )
            {
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
                GV->dump();
                CS->dump();
#endif /* !NDEBUG || LLVM_ENABLE_DUMP */
                assert( 0);
            } else
            {
                const ConstantInt *PI = dyn_cast<ConstantInt>( CS->getOperand( 0));

                lccrt_function_set_init_type( k->second, itf);
                lccrt_function_set_init_priority( k->second, PI->getZExtValue());
            }
        }
    } else if ( (strcmp( name.c_str(), "llvm.global.annotations") == 0) ) {
        // Just ignore.        
    } else
    {
        errorDump( GV);
        r = false;
    }

    return (r);
} /* LccrtEmitter::makeAppendingGlobal */

/**
 * Создание алиаса.
 */
lccrt_var_ptr
LccrtEmitter::makeAlias( GlobalAlias *GA)
{
    lccrt_link_t lnk;
    lccrt_type_ptr t = 0;
    lccrt_var_ptr g = 0;
    std::string name = GA->getName().str();
    MapGAToVar::const_iterator k = avars.find( GA);

    if ( (k != avars.end()) )
    {
        g = k->second;
    } else
    {
        t = makeType( GA->getType());
        lnk = makeLink( GA->getLinkage(), GA->getVisibility(), GlobalVariable::NotThreadLocal, 0, 1,
                        GA->isDeclaration());
        g = lccrt_var_new( m, LCCRT_VAR_LOC_EXT, t, name.c_str(), 0, lnk, 0);
        lccrt_var_set_init_value_reduce( g, makeVarinit( GA->getAliasee(), 0));

        avars.insert( PairGAToVar( GA, g));
        if ( GA->hasComdat() )
        {
            const GlobalValue *gv = dyn_cast<GlobalValue>( GA->getAliasee());

            if ( !gv
                 || !gv->hasComdat()
                 || (GA->getComdat() != gv->getComdat()) )
            {
                errorDump( GA);
            }
        }
    }

    return (g);
} /* LccrtEmitter::makeAlias */

/**
 * Поиск элемента в таблице.
 */
lccrt_var_ptr
LccrtEmitter::findGlobal( const GlobalVariable *G)
{
    MapGVToVar::const_iterator k = gvars.find( G);
    lccrt_var_ptr r = (k == gvars.end()) ? 0 : k->second;

    return (r);
} /* LccrtEmitter::findGlobal */

/**
 * Поиск элемента в таблице.
 */
lccrt_var_ptr
LccrtEmitter::findAlias( const GlobalAlias *G)
{
    MapGAToVar::const_iterator k = avars.find( G);
    lccrt_var_ptr r = (k == avars.end()) ? 0 : k->second;

    return (r);
} /* LccrtEmitter::findAlias */

/**
 * Поиск элемента в таблице.
 */
lccrt_function_ptr
LccrtEmitter::findFunc( const Function *F)
{
    MapLLVMFuncToFunc::const_iterator k = funcs.find( F);
    lccrt_function_ptr r = (k == funcs.end()) ? 0 : k->second;

    return (r);
} /* LccrtEmitter::findFunc */

/**
 * Поиск глобала, на основе которого формируется адрес.
 */
const GlobalValue *
LccrtEmitter::getConstExprGlobal( const Constant *C)
{
    const GlobalValue *r = 0;

    if ( C )
    {
        while ( !isa<const GlobalValue>( C) )
        {
            const ConstantExpr *CE = dyn_cast<const ConstantExpr>(C);

            if ( !CE
                 || !((CE->getOpcode() == Instruction::BitCast)
                      || (CE->getOpcode() == Instruction::GetElementPtr)) )
            {
                errorDump( C);
            }

            C = CE->getOperand( 0);
        }

        r = dyn_cast<const GlobalValue>(C);
    }

    return (r);
} /* LccrtEmitter::getConstExprGlobal */

/**
 * Создание константы.
 */
bool
LccrtEmitter::testVarinitExpr( const ConstantExpr *e, uint64_t shift)
{
    bool r = false;
    bool single = true/*!e->hasIndices()*/ && (e->getNumOperands() == 1);
    Type *ety = e->getType();

    if ( (e->getOpcode() == Instruction::GetElementPtr) ) {
        unsigned k;
        Value *v0 = e->getOperand( 0);
        Type *ty = v0->getType();

        r = true/*!e->hasIndices()*/;

        for ( k = 1; r && (k < e->getNumOperands()); ++k ) {
            Value *v = e->getOperand( k);
            unsigned bw = v->getType()->getPrimitiveSizeInBits();

            if ( !isa<ConstantInt>( v) ) {
                r = false;
            } else if ( (bw == 32) ) {
            } else if ( (bw == 64) ) {
            } else {
                r = false;
            }

            if ( dyn_cast<PointerType>( ty) ) {
                if ( (k != 1) ) {
                    r = false;
                }
            } else if ( dyn_cast<StructType>( ty) ) {
            } else if ( dyn_cast<FixedVectorType>( ty) ) {
            } else {
                r = false;
            }
        }

        if ( r ) {
            r = testVarinit( e->getOperand( 0), shift);
        }
    } else if ( (e->getOpcode() == Instruction::BitCast) ) {
        r = single && ety->isPointerTy() && testVarinit( e->getOperand( 0), shift);

    } else if ( (e->getOpcode() == Instruction::IntToPtr) ) {
        r = single && testVarinit( e->getOperand( 0), shift);

    } else if ( (e->getOpcode() == Instruction::PtrToInt) ) {
        lccrt_type_ptr tintp = lccrt_type_make_intptr( m);

        if ( (e->getType()->getScalarSizeInBits() == 8*lccrt_type_get_bytesize( tintp)) ) {
            r = single && testVarinit( e->getOperand( 0), shift);
        }
    }

    return (r);
} /* LccrtEmitter::testVarinitExpr */

/**
 * Создание константы.
 */
bool
LccrtEmitter::testVarinit( const Constant *c, uint64_t shift)
{
    uint64_t j;
    bool r = false;

    if ( (shift != 0) )
    {
        if ( !(isa<GlobalValue>( c)
               || isa<Function>( c)
               || isa<GlobalAlias>( c)
               || isa<ConstantInt>( c)
               || isa<ConstantExpr>( c)) )
        {
            return (false);
        }
    }

    if ( 0
         && (shift == 0)
         && (cnsts.find( c) != cnsts.end()) )
    {
        r = true;
    } else
    {
        if ( const ConstantInt *ci = dyn_cast<ConstantInt>( c) )
        {
            APInt A = ci->getValue();
            int bw = A.getBitWidth();

            if ( (bw == 1)
                 || (isBitWidthNormal( bw)
                     && (bw <= 64)) )
            {
                r = true;
            } else
            {
                r = (shift == 0);
            }
        } else if ( const ConstantFP *cf = dyn_cast<ConstantFP>( c))
        {
            if ( (&cf->getValueAPF().getSemantics() == &APFloat::IEEEsingle())
                 || (&cf->getValueAPF().getSemantics() == &APFloat::IEEEdouble()) )
            {
                r = true;

            } else if ( (&cf->getValueAPF().getSemantics() == &APFloat::x87DoubleExtended()) )
            {
                r = true;
            }
        } else if ( isa<ConstantAggregateZero>( c)
                    || isa<ConstantPointerNull>( c)
                    || isa<UndefValue>( c) )
        {
            r = true;

        } else if ( (isa<ConstantDataArray>( c)
                     && !dyn_cast<ConstantDataArray>( c)->isString())
                    || isa<ConstantDataVector>( c) )
        {
            const ConstantDataSequential *cds = dyn_cast<ConstantDataSequential>( c);
            uint64_t n = cds->getNumElements();

            r = true;
            for ( j = 0; j < n; ++j )
            {
                const Constant *e = static_cast<const Constant *>( cds->getElementAsConstant( j));

                if ( !testVarinit( e, 0) )
                {
                    r = false;
                    break;
                }
            }
        } else if ( isa<ConstantArray>( c)
                    || isa<ConstantStruct>( c)
                    || isa<ConstantVector>( c) )
        {
            uint64_t n = c->getNumOperands();

            r = true;
            for ( j = 0; j < n; ++j )
            {
                const Constant *e = static_cast<const Constant *>( c->getOperand( j));

                if ( !testVarinit( e, 0) )
                {
                    r = false;
                    break;
                }
            }
        } else if ( isa<ConstantDataArray>( c)
                    && dyn_cast<ConstantDataArray>( c)->isString() )
        {
            r = true;

        } else if (const GlobalValue *gv = dyn_cast<GlobalValue>( c))
        {
            if ( dyn_cast<GlobalVariable>( gv) )
            {
                r = true;

            } else if ( dyn_cast<Function>( gv) )
            {
                r = true;

            } else if ( const GlobalAlias *a = dyn_cast<GlobalAlias>( gv) )
            {           
                r = (a->getAliasee() != 0);
            }
        } else if ( const ConstantExpr *ce = dyn_cast<ConstantExpr>( c) )
        {
            r = testVarinitExpr( ce, shift);
        }
    }

    return (r);
} /* LccrtEmitter::testVarinit */

/**
 * Создание константы.
 */
lccrt_varinit_ptr
LccrtEmitter::makeVarinitExpr( ConstantExpr *e, uint64_t shift)
{
    lccrt_varinit_ptr a0, a1;
    lccrt_type_ptr t0, t1;
    lccrt_varinit_ptr vi = 0;
    lccrt_type_ptr t = makeType( e->getType());
    lccrt_type_ptr ti32 = lccrt_type_make_int( m, 4, 1);
    lccrt_type_ptr tu32 = lccrt_type_make_u32( m);
    lccrt_type_ptr ti64 = lccrt_type_make_int( m, 8, 1);
    lccrt_type_ptr tu64 = lccrt_type_make_u64( m);

    if ( (e->getOpcode() == Instruction::GetElementPtr) ) {
        unsigned k;
        Value *v0 = e->getOperand( 0);
        Type *ty = v0->getType();
        GEPOperator *gep = dyn_cast<GEPOperator>( e);

        for ( k = 1; k < e->getNumOperands(); ++k ) {
            int64_t idx = 0;
            Value *v = e->getOperand( k);
            const ConstantInt *cvi = cast<ConstantInt>( v);
            unsigned bw = v->getType()->getPrimitiveSizeInBits();

            if ( (bw == 32) ) {
                idx = (int64_t)(int32_t)cvi->getZExtValue();

            } else if ( (bw == 64) ) {
                assert( bw == 64);
                idx = (int64_t)cvi->getZExtValue();
            } else {
                errorDump( e);
            }

            if ( isa<PointerType>( ty) ) {
                if ( (k == 1) ) {
                    ty = gep->getSourceElementType();
                    if ( idx ) {
                        shift += idx * DL.getTypeAllocSize( ty);
                    }
                } else {
                    errorDump( e);
                }
            } else if ( StructType *stty = dyn_cast<StructType>( ty) ) {
                const StructLayout *stlay = DL.getStructLayout( stty);

                shift += stlay->getElementOffset( cvi->getZExtValue());
                ty = stty->getElementType( idx);

            } else if ( ArrayType *arty = dyn_cast<ArrayType>( ty) ) {
                ty = arty->getElementType();
                shift += DL.getTypeAllocSize( ty) * idx;

            } else if ( FixedVectorType *fvty = dyn_cast<FixedVectorType>( ty) ) {
                ty = fvty->getElementType();
                shift += DL.getTypeAllocSize( ty) * idx;
            } else {
                errorDump( e);
            }
        }

        vi = makeVarinit( e->getOperand( 0), shift);

    } else if ( (e->getOpcode() == Instruction::BitCast) ) {
        vi = makeVarinit( e->getOperand( 0), shift);
        assert( e->getNumOperands() == 1);

    } else if ( (e->getOpcode() == Instruction::IntToPtr) ) {
        vi = makeVarinit( e->getOperand( 0), shift);
        assert( shift == 0);
        assert( e->getNumOperands() == 1);
        if ( lccrt_varinit_is_addr_var( vi) ) {
        } else if ( lccrt_varinit_is_hex( vi) ) {
            vi = lccrt_varinit_new_scalar( t, lccrt_varinit_get_hex64( vi));
        } else {
            vi = 0;
            errorDump( e);
        }
    } else if ( (e->getOpcode() == Instruction::PtrToInt) ) {
        vi = makeVarinit( e->getOperand( 0), shift);
        assert( e->getNumOperands() == 1);
        if ( lccrt_varinit_is_zero_or_hex( vi) ) {
            assert( shift == 0);
            vi = lccrt_varinit_new_scalar( t, lccrt_varinit_get_zero_or_hex64( vi));
        }
    } else if ( (e->getOpcode() == Instruction::Trunc) ) {
        a0 = makeVarinit( e->getOperand( 0), shift);
        t0 = lccrt_varinit_get_type( a0);
        assert( shift == 0);
        if ( ((t == ti32) || (t == tu32))
             && ((t0 == ti64) || (t0 == tu64)) ) {
            vi = lccrt_varinit_new_scalar( t, lccrt_varinit_get_zero_or_hex64( a0));
        }

        if ( !vi ) {
            errorDump( e);
        }
    } else if ( (e->getOpcode() == Instruction::Add) ) {
        Value *v0 = e->getOperand( 0);
        Value *v1 = e->getOperand( 1);
        Type *ty0 = v0->getType();
        Type *ty1 = v1->getType();

        if ( (shift == 0)
             && ty0->isIntegerTy()
             && ty1->isIntegerTy()
             && (ty0->getIntegerBitWidth() == 64)
             && (ty1->getIntegerBitWidth() == 64) ) {
            a0 = makeVarinit( e->getOperand( 0), shift);
            a1 = makeVarinit( e->getOperand( 1), 0);
            t0 = lccrt_varinit_get_type( a0);
            t1 = lccrt_varinit_get_type( a1);

            if ( lccrt_type_is_int( t1) ) {
                if ( lccrt_varinit_is_addr_var( a0)
                     && lccrt_varinit_is_zero_or_hex( a1) ) {
                    lccrt_var_ptr a0v = lccrt_varinit_get_addr_var( a0);
                    uint64_t nelems = lccrt_varinit_get_num_elems( a0);

                    nelems += lccrt_varinit_get_zero_or_hex64( a1);
                    vi = lccrt_varinit_new_addr_var( a0v, nelems);
                }
            }
        }

        if ( !vi ) {
            errorDump( e);
        }
    } else if ( (e->getOpcode() == Instruction::Sub) ) {
        a0 = makeVarinit( e->getOperand( 0), shift);
        a1 = makeVarinit( e->getOperand( 1), shift);
        t0 = lccrt_varinit_get_type( a0);
        t1 = lccrt_varinit_get_type( a1);
        if ( (t0 == t1)
             && lccrt_type_is_int( t0) ) {
            if ( (t0 == ti32)
                 || (t0 == tu64) ) {
                uint32_t x = lccrt_varinit_get_zero_or_hex64( a0);
                uint32_t y = lccrt_varinit_get_zero_or_hex64( a1);

                vi = lccrt_varinit_new_scalar( t0, x - y);

            } else if ( (t0 == ti64)
                        || (t0 == tu64) ) {
                uint64_t x = lccrt_varinit_get_zero_or_hex64( a0);
                uint64_t y = lccrt_varinit_get_zero_or_hex64( a1);

                vi = lccrt_varinit_new_scalar( t0, x - y);
            }
        }

        if ( !vi ) {
            errorDump( e);
        }
    } else {
        errorDump( e);
    }

    return (vi);
} /* LccrtEmitter::makeVarinitExpr */

/**
 * Создание константы.
 */
lccrt_varinit_ptr
LccrtEmitter::makeVarinit( Constant *c, uint64_t shift)
{
    uint64_t j;
    lccrt_varinit_ptr vi = 0;
    Type *Ty = c->getType();
    lccrt_type_ptr t = makeType( Ty);
    MapCToVI::const_iterator k = cnsts.find( c);
    lccrt_type_ptr tu8 = lccrt_type_make_u8( m);

    if ( (shift != 0) ) {
        if ( !(isa<GlobalValue>( c)
               || isa<Function>( c)
               || isa<GlobalAlias>( c)
               || isa<ConstantInt>( c)
               || isa<ConstantExpr>( c)
               || isa<ConstantPointerNull>( c)) )
        {
            errorDump( c);
        }
    }

    //WriteConstantInternal( Out, CV, *TypePrinter, Machine, Context);
    if ( (shift == 0)
         && (k != cnsts.end()) )
    {
        vi = k->second;
    } else
    {
        if ( ConstantInt *ci = dyn_cast<ConstantInt>( c) ) {
            APInt A = ci->getValue();
            int bw = A.getBitWidth();

            if ( (bw == 1)
                 || (isBitWidthNormal( bw)
                     && (bw <= 64)) )
            {
                vi = lccrt_varinit_new_scalar( t, ci->getZExtValue() + shift);

            } else if ( (bw == 128) ) {
                if ( ci->isZero() ) {
                    vi = lccrt_varinit_new_zero( t);
                } else {
                    vi = lccrt_varinit_new_str( t, 16, (const char *)A.getRawData());
                }

                if ( shift ) {
                    errorDump( c);
                }
            } else {
                lccrt_varinit_ptr di = 0;
                int ne = (bw + 7) / 8;
                lccrt_type_ptr ta = lccrt_type_make_array( tu8, ne);

                //assert( bw % 8 == 0);
                di = lccrt_varinit_new_str( ta, ne, (const char *)A.getRawData());
                vi = lccrt_varinit_new_array( t, 1, &di);
                if ( shift ) {
                    errorDump( c);
                }
            }
        } else if ( ConstantFP *cf = dyn_cast<ConstantFP>( c)) {
            APInt a = cf->getValueAPF().bitcastToAPInt();

            if ( (&cf->getValueAPF().getSemantics() == &APFloat::IEEEsingle())
                 || (&cf->getValueAPF().getSemantics() == &APFloat::IEEEdouble()) )
            {
                assert( (a.getBitWidth() == 32) || (a.getBitWidth() == 64));
                vi = lccrt_varinit_new_scalar( t, *(a.getRawData()));

            } else if ( (&cf->getValueAPF().getSemantics() == &APFloat::x87DoubleExtended())
                        || (&cf->getValueAPF().getSemantics() == &APFloat::IEEEquad()) )
            {
                assert( (a.getBitWidth() == 80) || (a.getBitWidth() == 128));
                if ( cf->isZero() ) {
                    vi = lccrt_varinit_new_zero( t);
                } else {
                    vi = lccrt_varinit_new_str( t, a.getBitWidth()/8, (const char *)a.getRawData());
                }
            } else {
                assert( 0);
            }
        } else if ( isa<ConstantAggregateZero>( c)
                    || isa<ConstantPointerNull>( c)
                    || isa<UndefValue>( c) )
        {
            if ( (shift == 0) ) {
                vi = lccrt_varinit_new_zero( t);
            } else {
                vi = lccrt_varinit_new_scalar( t, shift);
            }
        } else if ( (isa<ConstantDataArray>( c)
                     && !dyn_cast<ConstantDataArray>( c)->isString())
                    || isa<ConstantDataVector>( c) )
        {
            ConstantDataSequential *cds = dyn_cast<ConstantDataSequential>( c);
            uint64_t n = cds->getNumElements();
            lccrt_varinit_ptr *d = new lccrt_varinit_ptr[n];

            for ( j = 0; j < n; ++j )
            {
                Constant *e = static_cast<Constant *>( cds->getElementAsConstant( j));

                d[j] = makeVarinit( e, 0);
            }

            vi = lccrt_varinit_new_array( t, n, d);
            delete[] d;

        } else if ( isa<ConstantArray>( c)
                    || isa<ConstantStruct>( c)
                    || isa<ConstantVector>( c) )
        {
            uint64_t n = c->getNumOperands();
            lccrt_varinit_ptr *d = new lccrt_varinit_ptr[n];

            for ( j = 0; j < n; ++j )
            {
                Constant *e = static_cast<Constant *>( c->getOperand( j));

                d[j] = makeVarinit( e, 0);
            }

            vi = lccrt_varinit_new_array( t, n, d);
            delete[] d;

        } else if ( isa<ConstantDataArray>( c)
                    && dyn_cast<ConstantDataArray>( c)->isString() )
        {
            StringRef s = dyn_cast<ConstantDataArray>( c)->getAsString();

            vi = lccrt_varinit_new_str( t, s.size(), s.data());
        } else if ( GlobalValue *gv = dyn_cast<GlobalValue>( c) ) {
            if ( GlobalVariable *v = dyn_cast<GlobalVariable>( gv) ) {
                vi = lccrt_varinit_new_addr_var( makeGlobal( v, true), shift);

            } else if ( Function *f = dyn_cast<Function>( gv) ) {
                vi = lccrt_varinit_new_addr_func( makeFunction( f), shift);

            } else if ( GlobalAlias *a = dyn_cast<GlobalAlias>( gv) ) {           
                Constant *av = a->getAliasee();

                if ( av ) {
                    vi = makeVarinit( av, shift);
                } else {
                    errorDump( a);
                }
            } else
            {
                errorDump( gv);
            }
        } else if ( ConstantExpr *ce = dyn_cast<ConstantExpr>( c) ) {
            vi = makeVarinitExpr( ce, shift);

        } else if ( BlockAddress *ba = dyn_cast<BlockAddress>( c) ) {
            int ba_ident = funcs_lbls[ba->getFunction()][ba->getBasicBlock()];

            vi = lccrt_varinit_new_scalar( lccrt_type_make_ptr_type( tu8), ba_ident);
        } else {
            errorDump( c);
        }

        if ( (shift == 0) ) {
            cnsts.insert( std::pair<Constant *, lccrt_varinit_ptr>( c, vi));
        }
    }

    return (vi);
} /* LccrtEmitter::makeVarinit */

/**
 * Создание константной переменной с инициализацией с хешированием.
 */
lccrt_var_ptr
LccrtEmitter::makeVarConst( lccrt_type_ptr type, lccrt_varinit_ptr vi)
{
    lccrt_var_ptr r = 0;
    TypeVarinit tvi = {type, vi};
    auto it = carg_vars.find( tvi);

    if ( (it != carg_vars.end()) )
    {
        r = it->second;
    } else
    {
        r = lccrt_var_new_constarg( m, type, vi);
        carg_vars[tvi] = r;
    }

    return (r);
} /* LccrtEmitter::makeVarConst */

/**
 * Создание константной переменной с инициализацией с хешированием для целочисленных констант.
 */
lccrt_var_ptr
LccrtEmitter::makeVarConstHex( lccrt_type_ptr type, uint64_t value)
{
    lccrt_var_ptr r = makeVarConst( type, lccrt_varinit_new_scalar( type, value));

    return (r);
} /* LccrtEmitter::makeVarConstHex */

/**
 * Создание информации о линковке.
 */
lccrt_link_t
LccrtEmitter::makeLink( GlobalValue::LinkageTypes lt, GlobalValue::VisibilityTypes vt,
                        GlobalVariable::ThreadLocalMode tlm, int is_cnst, int is_alias,
                        int is_declaration)
{
    lccrt_link_bind_t bind;
    lccrt_link_visibility_t vis;
    lccrt_link_tls_t tls;
    lccrt_link_t r = 0;

    if ( is_declaration )
    {
        if ( !((lt == GlobalValue::ExternalLinkage)
               || (lt == GlobalValue::ExternalWeakLinkage)
               || (lt == GlobalValue::AvailableExternallyLinkage)) )
        {
            assert( 0);
        }
    }

    /* Заполняем данные о линковке переменной. */
    switch ( lt )
    {
      case GlobalValue::ExternalLinkage:
      case GlobalValue::CommonLinkage:
        bind = LCCRT_LINK_BND_GLOBAL;
        break;
      case GlobalValue::AvailableExternallyLinkage:
        bind = LCCRT_LINK_BND_GLOBAL;
        break;
      case GlobalValue::PrivateLinkage:
      case GlobalValue::InternalLinkage:
        bind = LCCRT_LINK_BND_LOCAL;
        break;
      case GlobalValue::ExternalWeakLinkage:
      case GlobalValue::WeakODRLinkage:
      case GlobalValue::WeakAnyLinkage:
      case GlobalValue::LinkOnceODRLinkage:
      case GlobalValue::LinkOnceAnyLinkage:
        bind = LCCRT_LINK_BND_WEAK;
        break;
      case GlobalValue::AppendingLinkage:
        return (~0);
        break;
      default:
        assert( 0);
        abort();
        break;
    }

    switch ( vt )
    {
      case GlobalValue::DefaultVisibility: vis = LCCRT_LINK_VIS_DEFAULT; break;
      case GlobalValue::HiddenVisibility: vis = LCCRT_LINK_VIS_HIDDEN; break;
      case GlobalValue::ProtectedVisibility: vis = LCCRT_LINK_VIS_PROTECTED; break;
      default: assert( 0); abort(); break;
    }

    switch ( tlm )
    {
      case GlobalVariable::NotThreadLocal:
        tls = LCCRT_LINK_TLS_NO;
        break;
      case GlobalVariable::GeneralDynamicTLSModel:
        if ( (TM.getRelocationModel() == Reloc::Model::PIC_) )
        {
            tls = LCCRT_LINK_TLS_DYNAMIC_G;
        } else
        {
            tls = is_declaration ? LCCRT_LINK_TLS_EXEC_I : LCCRT_LINK_TLS_EXEC_L;
        }
        break;
      case GlobalVariable::LocalDynamicTLSModel:
        tls = LCCRT_LINK_TLS_DYNAMIC_L;
        break;
      case GlobalVariable::InitialExecTLSModel:
        tls = LCCRT_LINK_TLS_EXEC_I;
        break;
      case GlobalVariable::LocalExecTLSModel:
        tls = LCCRT_LINK_TLS_EXEC_L;
        break;
      default:
        assert( 0);
        abort();
        break;
    }

    r = lccrt_link_get( bind, vis, tls, is_cnst, is_alias);

    return (r);
} /* LccrtEmitter::makeLink */

/**
 * Создание или поиск ранее созданной функции (для интринсика).
 */
lccrt_function_ptr
LccrtEmitter::makeFunctionFast( const char *func_name, lccrt_type_ptr func_type)
{
    lccrt_link_t lnk;
    lccrt_function_ptr r = 0;
    std::string func_name_s( func_name);
    auto ah = lib_funcs.find( func_name_s);

    lnk = makeLink( GlobalValue::ExternalLinkage, GlobalValue::DefaultVisibility,
                    GlobalVariable::NotThreadLocal, 0, 0, 1);
    if ( (ah != lib_funcs.end()) )
    {
        r = ah->second;
        assert( lnk == lccrt_function_get_link( r));
        assert( func_type == lccrt_function_get_type( r));
    } else
    {
        r = lccrt_function_new( m, func_type, func_name, 0, lnk, 1, 0);
        lib_funcs.insert( std::pair<const std::string, lccrt_function_ptr>( func_name_s, r));
        lccrt_function_set_attr_does_not_throw( r, 1);
    }

    return (r);
} /* LccrtEmitter::makeFunctionFast */

/**
 * Проверяем, что тип не является целым или битовая длина выражается одним из целых типов.
 */
bool
LccrtEmitter::isTypeFloatNormal( const Type *T)
{
    bool r = false;

    if ( T->isFloatTy()
         || T->isDoubleTy()
         || T->isX86_FP80Ty() )
    {
        r = true;
    }

    return (r);
} /* LccrtEmitter::isTypeFloatNormal */

/**
 * Проверяем, что битовая длина выражается одним из целых типов.
 */
bool
LccrtEmitter::isBitWidthNormal( int width)
{
    bool r = false;

    if ( (width == 8)
         || (width == 16)
         || (width == 32)
         || (width == 64)
         || (width == 128) )
    {
        r = true;
    }

    return (r);
} /* LccrtEmitter::isBitWidthNormal */

/**
 * Проверяем, что тип не является целым или битовая длина выражается одним из целых типов.
 */
bool
LccrtEmitter::isIntBitWidthNormal( Type *T)
{
    bool r = false;

    if ( !isa<IntegerType>(T)
         || isBitWidthNormal( T->getPrimitiveSizeInBits()) )
    {
        r = true;
    }

    return (r);
} /* LccrtEmitter::isIntBitWidthNormal */

/**
 * Проверяем, что тип не является целым или битовая длина равна 1 биту.
 */
bool
LccrtEmitter::isIntBitWidthBool( Type *T)
{
    bool r = false;

    if ( !isa<IntegerType>(T)
         || (T->getPrimitiveSizeInBits() == 1) )
    {
        r = true;
    }

    return (r);
} /* LccrtEmitter::isIntBitWidthBool */

/**
 * Объединение двух проверок.
 */
bool
LccrtEmitter::isIntBitWidthNormalOrBool( Type *T)
{
    bool r = false;

    if ( isIntBitWidthNormal( T)
         || isIntBitWidthBool( T) )
    {
        r = true;
    }

    return (r);
} /* LccrtEmitter::isIntBitWidthNormalOrBool */

/**
 * Проверяем, что тип является вектором нестандартных целых.
 */
bool
LccrtEmitter::isTypeNonStdVector( Type *T) {
    bool r = false;

    if ( isa<ScalableVectorType>( T) ) {
        errorDump( T);
    } else if ( isa<FixedVectorType>( T) ) {
        int elem_bitsize = getElementBitsize( T);

        if ( !isBitWidthNormal( elem_bitsize) ) {
            r = true;
        }
    }

    return (r);
} /* LccrtEmitter::isTypeNonStdVector */

/**
 * Проверяем, что для векторного билтина: T -> T допустимо создавать lccrt-билтин.
 */
bool
LccrtEmitter::isVectorBuiltinInt( FixedVectorType *T, int elem_minbitsize) {
    int bytesize;
    bool r = false;
    int num_elems = T->getNumElements();
    int bitsize = T->getScalarSizeInBits();

    if ( bitsize <= 8 ) {
        bytesize = 1;
    } else if ( bitsize <= 16 ) {
        bytesize = 2;
    } else if ( bitsize <= 32 ) {
        bytesize = 4;
    } else {
        bytesize = 8;
    }

    if ( isa<IntegerType>( T->getScalarType()) ) {
        if ( (bitsize >= elem_minbitsize)
             && (bitsize <= 64)
             && (num_elems*bytesize <= 256) )
        {
            r = true;
        }
    }

    return (r);
} /* LccrtEmitter::isVectorBuiltinInt */

/**
 * Создание типа для хранения вектора нестандартных целых в памяти
 * в плотном формате.
 */
lccrt_type_ptr
LccrtEmitter::makeTypeDenseVector( Type *T) {
    lccrt_type_ptr r = 0;
    auto k = dense_types.find( T);

    assert( isTypeNonStdVector( T));

    if ( (k != dense_types.end()) ) {
        r = k->second;
    } else {
        lccrt_type_ptr elems[1];
        int bitsize = DL.getTypeSizeInBits( T);
        int bytesize = (bitsize + 7) / 8;
        int bytealign = DL.getABITypeAlign( T).value();

        elems[0] = lccrt_type_make_array( lccrt_type_make_u8( m), bytesize);
        elems[0] = lccrt_type_make_field( elems[0], 1, 0, 0, 8*bytesize);

        r = lccrt_type_make_struct( m, bytealign, bytesize, 1, elems, 0);

        dense_types[T] = r;
    }

    return (r);
} /* LccrtEmitter::makeTypeDenseVector */

/**
 * Создание целого типа с выравниванием.
 */
lccrt_type_ptr
LccrtEmitter::makeTypeIntNormal( int bitwidth) {
    lccrt_type_ptr r = 0;

    if ( bitwidth <= 8 ) {
        r = lccrt_type_make_u8( m);
    } else if ( bitwidth <= 16 ) {
        r = lccrt_type_make_u16( m);
    } else if ( bitwidth <= 32 ) {
        r = lccrt_type_make_u32( m);
    } else if ( bitwidth <= 64 ) {
        r = lccrt_type_make_u64( m);
    } else if ( bitwidth <= 128 ) {
        r = lccrt_type_make_u128( m);
    } else {
        assert( 0);
    }

    return (r);
} /* LccrtEmitter::makeTypeIntNormal */

/**
 * Создание типа.
 */
lccrt_type_ptr
LccrtEmitter::makeTypeStruct( StructType *T, NamedTypes &ntypes)
{
    lccrt_type_ptr t = 0;
    int k;
    const StructLayout *lay = DL.getStructLayout( T);
    int num_elems = T->getNumElements();
    lccrt_type_ptr *elems = new lccrt_type_ptr[num_elems];

    for ( k = 0; k < num_elems; ++k )
    {
        Type *elem = T->getElementType( k);
        int shift = lay->getElementOffset( k);
        int bitsize = DL.getTypeSizeInBits( elem);
        int bitsubshift = lay->getElementOffsetInBits( k) - 8*shift;

        elems[k] = makeType( elem, ntypes);
        elems[k] = lccrt_type_make_field( elems[k], 1, shift, bitsubshift, bitsize);
    }

    t = lccrt_type_make_struct( m, lay->getAlignment().value(),
                                lay->getSizeInBytes(), num_elems, elems, 0);
    delete[] elems;

    return (t);
} /* LccrtEmitter::makeTypeStruct */

/**
 * Создание типа.
 */
lccrt_type_ptr
LccrtEmitter::makeType( Type *T, std::map<Type *, lccrt_type_ptr> &ntypes)
{
    lccrt_type_ptr t = 0;
    lccrt_type_ptr e = 0;
    StructType *Tst = 0;
    ArrayType *Ta = 0;
    FixedVectorType *Tv = 0;
    FunctionType *Tf = 0;
    auto k = types.find( T);
    int bitsize = T->getPrimitiveSizeInBits();

    if ( (k != types.end()) )
    {
        t = k->second;
    }

    if ( !t
         || (isa<StructType>( T)
             && lccrt_type_is_typename( t)
             && lccrt_type_get_parent( t)) )
    {
        switch ( T->getTypeID() )
        {
          case Type::VoidTyID:
            t = lccrt_type_make_void( m);
            break;
          case Type::FloatTyID:
          case Type::DoubleTyID:
          case Type::X86_FP80TyID:
          case Type::FP128TyID:
            if ( (bitsize % 8 == 0) ) {
                t = lccrt_type_make_float( m, bitsize / 8);
            } else {
                return (0);
            }
            break;
          case Type::IntegerTyID:
            if ( (bitsize == 1) ) {
                t = lccrt_type_make_bool( m);

            } else if ( isBitWidthNormal( bitsize) ) {
                t = lccrt_type_make_int( m, bitsize / 8, 0);
            } else {
                int ne = DL.getTypeStoreSize( T);

                t = lccrt_type_make_array( lccrt_type_make_u8( m), ne);
                t = lccrt_type_make_field( t, 1, 0, 0, ne * 8);
                t = lccrt_type_make_struct( m, DL.getABITypeAlign( T).value(), ne, 1, &t, 0);
            }
            break;
          case Type::PointerTyID:
            t = lccrt_type_make_pvoid( m);
            break;
          case Type::StructTyID:
            Tst = static_cast<StructType *>( T);
            if ( Tst->isOpaque() ) {
                if ( Tst->hasName() ) {
                    t = lccrt_type_make_typename( m, Tst->getName().data(), 0);
                    /* Делаем указатель, т.к. могут быть глобальные extern (!) переменные
                       с opaque-типом! В таком случае может использоваться только адрес. */
                    t = lccrt_type_make_ptr_type( t);
                } else {
                    return (0);
                }
            } else {
                if ( Tst->hasName() ) {
                    const StructLayout *lay = DL.getStructLayout( Tst);

                    t = lccrt_type_make_typename( m, Tst->getName().data(), 0);
                    ntypes[Tst] = t;
                    lccrt_type_set_typename_bytesize( t, lay->getSizeInBytes());
                    lccrt_type_set_typename_bytealign( t, lay->getAlignment().value());
                } else {
                    t = makeTypeStruct( Tst, ntypes);
                }
            }
            break;
          case Type::ArrayTyID:
            Ta = static_cast<ArrayType *>( T);
            e = makeType( Ta->getElementType(), ntypes);
            t = lccrt_type_make_array( e, Ta->getNumElements());
            break;
          case Type::FixedVectorTyID:
            Tv = static_cast<FixedVectorType *>( T);
            e = makeType( Tv->getElementType(), ntypes);
            //t = lccrt_type_make_vector( e, Tv->getNumElements());
            t = lccrt_type_make_array( e, Tv->getNumElements());
            break;
          case Type::FunctionTyID:
            Tf = static_cast<FunctionType *>( T);
            {
                int k;
                int is_varg = Tf->isVarArg();
                int num_elems = Tf->getNumParams();
                lccrt_type_ptr *elems = new lccrt_type_ptr[num_elems + is_varg];

                e = makeType( Tf->getReturnType(), ntypes);
                for ( k = 0; k < num_elems; ++k )
                {
                    elems[k] = makeType( Tf->getParamType( k), ntypes);
                }

                if ( is_varg )
                {
                    elems[k] = lccrt_type_make_ellipsis( m);
                }

                t = lccrt_type_make_func( e, num_elems + is_varg, elems);
                delete[] elems;
            }
            break;
          default:
            errorDump( T);
            return (0);
            break;
        }

        types[T] = t;
    }

    return (t);
} /* LccrtEmitter::makeType */

/**
 * Создание типа.
 */
lccrt_type_ptr
LccrtEmitter::makeType( Type *T)
{
    std::map<Type *, lccrt_type_ptr> ntypes;
    lccrt_type_ptr t = makeType( T, ntypes);

    while ( !ntypes.empty() ) {
        lccrt_type_ptr s = 0;
        auto it = ntypes.begin();
        auto S = static_cast<StructType *>( it->first);

        ntypes.erase( it);
        if ( S->hasName() ) {
            s = makeTypeStruct( S, ntypes);
            lccrt_type_make_typename( m, S->getName().data(), s);
        }
    }

    return (t);
} /* LccrtEmitter::makeType */

typedef struct
{
    const char *iname;
    const char *oname;
} subst_t;

std::string
LccrtFunctionEmitter::lowerCallName( lccrt_m_ptr m, const char *s, lccrt_type_ptr *t, int is_test)
{
    int i;
    std::string math_name;
    int num_args = 0;
    bool is_name = false;
    std::string r = is_test ? "" : s;
    static subst_t suff_stdint[6] = {{".i8", "_8"},   {".i16", "_16"},   {".i32", "_32"},
                                     {".i64", "_64"}, {".i128", "_128"}, {0, 0}};
    static subst_t suff_stdfloat[4] = {{".f32", "f"}, {".f64", ""}, {".f80", "l"}, {0, 0}};
    static subst_t suff_stdfloatn[4] = {{".f32", "_f32"}, {".f64", "_f64"}, {".f80", "_f80"}, {0, 0}};
    static struct
    {
        const char *iname;
        const char *oname;
        subst_t *suffixes;
        lccrt_type_ptr *type;
    } tbl[] =
    {
        {"llvm.memcmp.p0.p0.i64",           "__builtin_memcmp"},
        {"llvm.memcpy.p0.p0.i32",           "__builtin_memcpy"},
        {"llvm.memcpy.p0.p0.i64",           "__builtin_memcpy"},
        {"llvm.memmove.p0.p0.i64",          "__builtin_memmove"},
        {"llvm.memset.p0.i64",              "__builtin_memset"},
        {"llvm.memcmp.p0i8.p0i8.i64",       "__builtin_memcmp"},
        {"llvm.memcpy.p0i8.p0i8.i32",       "__builtin_memcpy"},
        {"llvm.memcpy.p0i8.p0i8.i64",       "__builtin_memcpy"},
        {"llvm.memmove.p0i8.p0i8.i64",      "__builtin_memmove"},
        {"llvm.memset.p0i8.i64",            "__builtin_memset"},
        {"llvm.trap",                       "abort"},
        {"llvm.debugtrap",                  "abort"},
        {"llvm.va_start",                   "__lccrt_va_start"},
        {"llvm.va_start.p0",                "__lccrt_va_start"},
        {"llvm.va_end",                     "__lccrt_va_end"},
        {"llvm.va_end.p0",                  "__lccrt_va_end"},
        {"llvm.va_copy",                    "__lccrt_va_copy"},
        {"llvm.va_copy.p0",                 "__lccrt_va_copy"},
        {"llvm.stacksave",                  "__lccrt_builtin_stacksave"},
        {"llvm.stacksave.p0",               "__lccrt_builtin_stacksave"},
        {"llvm.stackrestore",               "__lccrt_builtin_stackrestore"},
        {"llvm.stackrestore.p0",            "__lccrt_builtin_stackrestore"},
        {"llvm.eh.typeid.for",              "__lccrt_eh_typeid_for"},
        {"llvm.returnaddress",              "__builtin_return_address"},
        {"llvm.uadd.with.overflow",         "__lccrt_uadd_overflow",        suff_stdint},
        {"llvm.sadd.with.overflow",         "__lccrt_sadd_overflow",        suff_stdint},
        {"llvm.sadd.with.overflow.i65",     "__lccrt_sadd_overflow_65"},
        {"llvm.usub.with.overflow",         "__lccrt_usub_overflow",        suff_stdint},
        {"llvm.ssub.with.overflow",         "__lccrt_ssub_overflow",        suff_stdint},
        {"llvm.umul.with.overflow",         "__lccrt_umul_overflow",        suff_stdint},
        {"llvm.smul.with.overflow",         "__lccrt_smul_overflow",        suff_stdint},
        {"llvm.uadd.sat",                   "__lccrt_uadd_sat",             suff_stdint},
        {"llvm.sadd.sat",                   "__lccrt_sadd_sat",             suff_stdint},
        {"llvm.usub.sat",                   "__lccrt_usub_sat",             suff_stdint},
        {"llvm.ssub.sat",                   "__lccrt_ssub_sat",             suff_stdint},
        {"llvm.umul.sat",                   "__lccrt_umul_sat",             suff_stdint},
        {"llvm.smul.sat",                   "__lccrt_smul_sat",             suff_stdint},
        {"llvm.powi.f32",                   "__builtin_powif"},
        {"llvm.powi.f32.i32",               "__builtin_powif"},
        {"llvm.powi.f64",                   "__builtin_powi"},
        {"llvm.powi.f64.i32",               "__builtin_powi"},
        {"llvm.powi.f80",                   "__builtin_powil"},
        {"llvm.powi.f80.i32",               "__builtin_powil"},
        {"llvm.fabs",                       "__builtin_fabs",               suff_stdfloat},
        {"llvm.exp",                        "__builtin_exp",                suff_stdfloat},
        {"llvm.exp2",                       "__builtin_exp2",               suff_stdfloat},
        {"llvm.ceil",                       "__builtin_ceil",               suff_stdfloat},
        {"llvm.floor",                      "__builtin_floor",              suff_stdfloat},
        {"llvm.round",                      "__builtin_round",              suff_stdfloat},
        {"llvm.lround.i32",                 "__builtin_lround",             suff_stdfloat},
        {"llvm.lround.i64",                 "__builtin_llround",            suff_stdfloat},
        {"llvm.llround.i64",                "__builtin_llround",            suff_stdfloat},
        {"llvm.trunc",                      "__builtin_trunc",              suff_stdfloat},
        {"llvm.rint",                       "__builtin_rint",               suff_stdfloat},
        {"llvm.lrint.i32",                  "__builtin_lrint",              suff_stdfloat},
        {"llvm.lrint.i64",                  "__builtin_llrint",             suff_stdfloat},
        {"llvm.llrint.i64",                 "__builtin_llrint",             suff_stdfloat},
        {"llvm.nearbyint",                  "nearbyint",                    suff_stdfloat},
        {"llvm.fma",                        "__builtin_fma",                suff_stdfloat},
        {"llvm.log",                        "__builtin_log",                suff_stdfloat},
        {"llvm.log2",                       "__builtin_log2",               suff_stdfloat},
        {"llvm.log10",                      "__builtin_log10",              suff_stdfloat},
        {"llvm.pow",                        "__builtin_pow",                suff_stdfloat},
        {"llvm.sqrt",                       "__builtin_sqrt",               suff_stdfloat},
        {"llvm.sin",                        "__builtin_sin",                suff_stdfloat},
        {"llvm.cos",                        "__builtin_cos",                suff_stdfloat},
        {"llvm.tan",                        "__builtin_tan",                suff_stdfloat},
        {"llvm.minnum",                     "__builtin_fmin",               suff_stdfloat},
        {"llvm.maxnum",                     "__builtin_fmax",               suff_stdfloat},
        {"llvm.copysign",                   "__builtin_copysign",           suff_stdfloat},
        {"llvm.is.fpclass",                 "__lccrt_isfpclass",            suff_stdfloatn},
        {"llvm.mulsc3",                     ""},
        {"llvm.clear_cache",                "__builtin___clear_cache"},
        {"llvm.e2k.loadmas.8u",             "__builtin_loadmas_8u"},
        {"llvm.e2k.loadmas.16u",            "__builtin_loadmas_16u"},
        {"llvm.e2k.loadmas.32u",            "__builtin_loadmas_32u"},
        {"llvm.e2k.loadmas.64u",            "__builtin_loadmas_64u"},
        {"llvm.e2k.loadmas.128v",           "__builtin_loadmas_128v"},
        {"llvm.e2k.storemas.8u",            "__builtin_storemas_8u"},
        {"llvm.e2k.storemas.16u",           "__builtin_storemas_16u"},
        {"llvm.e2k.storemas.32u",           "__builtin_storemas_32u"},
        {"llvm.e2k.storemas.64u",           "__builtin_storemas_64u"},
        {"llvm.e2k.storemas.128v",          "__builtin_storemas_128v"},
        {"llvm.e2k.pstoremas.128v",         "__builtin_pstoremas_128v"},
        {0,                                 0}
    };
    lccrt_type_ptr tpv = lccrt_type_make_pvoid( m);

    if ( t ) {
        (*t) = 0;
    }

    for ( i = 0; tbl[i].iname; ++i )
    {
        if ( !tbl[i].suffixes )
        {
            if ( (strcmp( s, tbl[i].iname) == 0) )
            {
                is_name = true;
                r = tbl[i].oname;
                break;
            }
        } else
        {
            int ilen = strlen( tbl[i].iname);

            if ( (strncmp( s, tbl[i].iname, ilen) == 0) )
            {
                for ( int j = 0; tbl[i].suffixes[j].iname; ++j )
                {
                    if ( (strcmp( s + ilen, tbl[i].suffixes[j].iname) == 0) )
                    {
                        is_name = true;
                        r = tbl[i].oname;
                        r += tbl[i].suffixes[j].oname;
                        break;
                    }
                }
            }

            if ( is_name )
            {
                break;
            }
        }
    }

    if ( !is_name ) {
        const char *p0 = "llvm.e2k.";
        int p0_len = strlen( p0);

        if ( (strncmp( s, p0, p0_len) == 0) ) {
            is_name = true;
            r = "__builtin_e2k_";
            r += (s + p0_len);
            for ( int k = p0_len, k1 = r.length(); k <= k1 ; ++k ) {
                if ( r[k] == '.' ) {
                    r[k] = '_';
                }
            }
        }
    }

    if ( t )
    {
        lccrt_type_ptr targs[16];
        lccrt_type_ptr ti32 = lccrt_type_make_u32( m);
        lccrt_type_ptr ti64 = lccrt_type_make_u64( m);
        const char *p = r.c_str();

        if ( !tbl[i].iname )
        {
            (*t) = 0;

        } else if ( (strcmp( p, "__builtin_memcpy") == 0)
                    || (strcmp( p, "__builtin_memmove") == 0) )
        {
            targs[0] = tpv;
            targs[1] = tpv;
            targs[2] = ti64;
            (*t) = lccrt_type_make_func( tpv, 3, targs);

        } else if ( (strcmp( p, "__builtin_memcmp") == 0) )
        {
            targs[0] = tpv;
            targs[1] = ti32;
            targs[2] = ti64;
            (*t) = lccrt_type_make_func( ti32, 3, targs);

        } else if ( (strcmp( p, "__builtin_memset") == 0) )
        {
            targs[0] = tpv;
            targs[1] = ti32;
            targs[2] = ti64;
            (*t) = lccrt_type_make_func( tpv, 3, targs);
        }
    }

    if ( is_name_math_vec( s, math_name, num_args) )
    {
        r = "__lccrt_" + math_name + "_v";
        if ( t )
        {
            (*t) = makeLibCallType( m, num_args, true);
        }
    }

    r = apply_lccrt_mangling( r);

    return (r);
} /* LccrtEmitter::lowerCallName */

Type *
LccrtEmitter::getValueElementType( Value *V) {
    Type *r = 0;

    if ( isa<LoadInst>( V) ) {
        r = dyn_cast<LoadInst>( V)->getType();
    } else if ( isa<GetElementPtrInst>( V) ) {
        r = dyn_cast<GetElementPtrInst>( V)->getSourceElementType();
    } else {
        errorDump( V);
    }

    return (r);
} /* LccrtEmitter::getValueElementType */

lccrt_type_ptr
LccrtEmitter::makeValueType( Value *V) {
    lccrt_type_ptr r = 0;

    if ( isa<LoadInst>( V) ) {
        r = makeType( dyn_cast<LoadInst>( V)->getType());
    } else if ( isa<GetElementPtrInst>( V) ) {
        r = makeType( dyn_cast<GetElementPtrInst>( V)->getResultElementType());
        r = lccrt_type_make_ptr_type( r);
    } else if ( isa<AllocaInst>( V) ) {
        //r = makeType( dyn_cast<AllocaInst>( V)->getAllocatedType());
        //r = lccrt_type_make_ptr_type( r);
        r = lccrt_type_make_pvoid( m);
    } else if ( isa<VAArgInst>( V) ) {
        r = makeType( V->getType());
    } else if ( isa<CmpInst>( V) ) {
        r = makeType( V->getType());
    } else if ( isa<CastInst>( V) ) {
        r = makeType( V->getType());
    } else if ( isa<BinaryOperator>( V) ) {
        r = makeType( V->getType());
    } else if ( isa<Constant>( V) ) {
        if ( ConstantExpr *e = dyn_cast<ConstantExpr>(V) ) {
            if ( e->getOpcode() == Instruction::GetElementPtr ) {
                r = makeType( dyn_cast<GEPOperator>( e)->getResultElementType());
                r = lccrt_type_make_ptr_type( r);
            } else if ( e->getOpcode() == Instruction::IntToPtr ) {
                r = lccrt_type_make_pvoid( m);
            } else if ( e->getOpcode() == Instruction::PtrToInt ) {
                r = makeType( V->getType());
            } else if ( e->getOpcode() == Instruction::SExt ) {
                r = makeType( V->getType());
            } else if ( e->getOpcode() == Instruction::ZExt ) {
                r = makeType( V->getType());
            } else if ( e->getOpcode() == Instruction::Trunc ) {
                r = makeType( V->getType());
            } else if ( e->getOpcode() == Instruction::ICmp ) {
                r = makeType( V->getType());
            } else if ( e->getOpcode() == Instruction::Sub ) {
                r = makeType( V->getType());
            } else if ( e->getOpcode() == Instruction::Add ) {
                r = makeType( V->getType());
            } else if ( e->getOpcode() == Instruction::Mul ) {
                r = makeType( V->getType());
            } else if ( e->getOpcode() == Instruction::SDiv ) {
                r = makeType( V->getType());
            } else if ( e->getOpcode() == Instruction::UDiv ) {
                r = makeType( V->getType());
            } else if ( e->getOpcode() == Instruction::Or ) {
                r = makeType( V->getType());
            } else if ( e->getOpcode() == Instruction::And ) {
                r = makeType( V->getType());
            } else if ( e->getOpcode() == Instruction::Xor ) {
                r = makeType( V->getType());
            } else if ( e->getOpcode() == Instruction::Shl ) {
                r = makeType( V->getType());
            } else if ( e->getOpcode() == Instruction::LShr ) {
                r = makeType( V->getType());
            } else if ( e->getOpcode() == Instruction::AShr ) {
                r = makeType( V->getType());
            } else if ( e->getOpcode() == Instruction::BitCast ) {
                r = makeType( V->getType());
            } else {
                errorDump( V);
            }
        } else if ( isa<ConstantInt>(V) ) {
            r = makeType( V->getType());
        } else if ( isa<ConstantFP>(V) ) {
            r = makeType( V->getType());
        } else if ( isa<ConstantVector>(V) ) {
            r = makeType( V->getType());
        } else if ( isa<ConstantDataVector>(V) ) {
            r = makeType( V->getType());
        } else if ( isa<ConstantStruct>(V) ) {
            r = makeType( V->getType());
        } else if ( isa<ConstantArray>(V) ) {
            r = makeType( V->getType());
        } else if ( isa<GlobalValue>(V) ) {
            r = makeType( dyn_cast<GlobalValue>(V)->getValueType());
            r = lccrt_type_make_ptr_type( r);
        } else if ( isa<ConstantPointerNull>(V) ) {
            r = lccrt_type_make_pvoid( m);
        } else if ( isa<BlockAddress>(V) ) {
            r = lccrt_type_make_pvoid( m);
        } else if ( isa<ConstantAggregateZero>(V) ) {
            r = makeType( V->getType());
        } else if ( isa<ConstantTargetNone>(V) ) {
            r = makeType( V->getType());
        } else if ( isa<UndefValue>(V) ) {
            if ( V->getType()->isPointerTy() ) {
                r = lccrt_type_make_pvoid( m);
            } else {
                r = makeType( V->getType());
            }
        } else {
            errorDump( V);
        }
    } else if ( isa<PHINode>( V) ) {
        r = makeType( V->getType());
    } else if ( isa<CallInst>( V) ) {
        r = makeType( V->getType());
    } else if ( isa<InvokeInst>( V) ) {
        r = makeType( V->getType());
    } else if ( isa<LandingPadInst>( V) ) {
        r = makeType( V->getType());
    } else if ( isa<AtomicRMWInst>( V) ) {
        r = makeType( V->getType());
    } else if ( isa<AtomicCmpXchgInst>( V) ) {
        r = makeType( V->getType());
    } else if ( isa<SelectInst>( V) ) {
        r = makeType( V->getType());
    } else if ( isa<ExtractElementInst>( V) ) {
        r = makeType( V->getType());
    } else if ( isa<InsertElementInst>( V) ) {
        r = makeType( V->getType());
    } else if ( isa<ExtractValueInst>( V) ) {
        r = makeType( V->getType());
    } else if ( isa<InsertValueInst>( V) ) {
        r = makeType( V->getType());
    } else if ( isa<ShuffleVectorInst>( V) ) {
        r = makeType( V->getType());
    } else if ( isa<FreezeInst>( V) ) {
        r = makeType( V->getType());
    } else if ( Instruction *inst = dyn_cast<Instruction>( V) ) {
        if ( inst->getOpcode() == Instruction::FNeg ) {
            r = makeType( V->getType());
        } else {
            errorDump( V);
        }
    } else {
        errorDump( V);
    }

    return (r);
} /* LccrtEmitter::makeValueType */

bool
LccrtEmitter::isVecUniform( const Value *V)
{
    bool r = false;
    Type *T = V->getType();

    if ( isa<FixedVectorType>( T)
         && (isa<ConstantVector>( V)
             || isa<ConstantDataVector>( V)) )
    {
        FixedVectorType *TV = cast<FixedVectorType>( T);
        int num_elems = TV->getNumElements();
        const Constant *C = dyn_cast<Constant>( V);

        if ( (num_elems > 0) )
        {
            bool is_uni = true;
            const Value *C0 = C->getAggregateElement( 0U);

            for ( int i = 1; i < num_elems; ++i )
            {
                if ( (C0 != C->getAggregateElement( i)) )
                {
                    is_uni = false;
                    break;
                }
            }

            r = is_uni;
        }
    }

    return (r);
} /* LccrtEmitter::isVecUniform */
 
bool
LccrtEmitter::isDbgDeclare( const Function *F)
{
    const char *name = F->getName().data();
    bool r = false;

    if ( (strcmp( name, "llvm.dbg.declare") == 0) )
    {
        r = true;
    }

    return (r);
} /* LccrtEmitter::isDbgDeclare */

bool
LccrtEmitter::isDbg( const Function *F)
{
    const char *name = F->getName().data();
    bool r = false;
    const char *pttn = "llvm.dbg.";
    int pttn_len = strlen( pttn);

    if ( (strncmp( name, pttn, pttn_len) == 0) )
    {
        r = true;
    }

    return (r);
} /* LccrtEmitter::isDbg */

int64_t thread_local LccrtFunctionEmitter::fast_lib_cur_ = 0;
int64_t thread_local LccrtFunctionEmitter::fast_lib_bnd_ = -1;

LccrtFunctionEmitter::LccrtFunctionEmitter( LccrtEmitter &vle, lccrt_f_ptr vf, Function *vF,
                                            LccrtModuleIpaEmitter &MIPA, LccrtDbgEmitter &dbge,
                                            LccrtDbglinesEmitter &dbgle)
    : le( vle), FIPA( &MIPA, vf, vF), dbge_( dbge), dbgle_( dbgle)
{
    const char *fast_lib_str = getenv( "LLVM_LCCRT_FAST_LIB");

    f = vf;
    F = vF;
    num_lvals = 0;
    num_cnsts = 0;
    num_alloc = 0;
    num_rvars = 0;
    m = lccrt_function_get_module( f);
    c = lccrt_module_get_context( m);
    lccrt_context_set_verbose_ir( c, le.verbose_ir);
    fast_lib_bnd_ = fast_lib_str ? atoi( fast_lib_str) : -1;

    return;
} /* LccrtFunctionEmitter::LccrtFunctionEmitter */

bool
LccrtFunctionEmitter::isFastLibCall( const User *O, const char *file, int line)
{
    bool r = true;
    const Instruction *I = dyn_cast<Instruction>( O);

    if ( (fast_lib_bnd_ >= 0)
         && (fast_lib_cur_ + 1 > fast_lib_bnd_) )
    {
        r = false;
        if ( (fast_lib_cur_ == 0)
             || (fast_lib_cur_ == fast_lib_bnd_) )
        {
            fast_lib_cur_++;
            printf( "FAST_LIB function: %s, %s:%d\n",
                    lccrt_function_get_name( f), file, line);
            if ( I )
            {
                errs() << *I << "\n";
                //printf( "FAST_LIB oper: %s\n", O->getName().data());
            }
        }
    } else
    {
        fast_lib_cur_++;
        if ( (fast_lib_cur_ == fast_lib_bnd_) )
        {
            printf( "FAST_LIB function: %s, %s:%d\n",
                    lccrt_function_get_name( f), file, line);
            if ( I )
            {
                errs() << *I << "\n";
                //printf( "FAST_LIB oper: %s\n", O->getName().data());
            }
        }
    }

    return (r);
} /* LccrtFunctionEmitter::isFastLibCall */

void
LccrtFunctionEmitter::makeArgs( lccrt_type_ptr types[])
{
    unsigned arg_num = 0;
    const AttributeList &Attrs = F->getAttributes();

    for ( Function::const_arg_iterator I = F->arg_begin(), E = F->arg_end(); I != E; ++I, arg_num++ )
    {
        lccrt_var_ptr a;
        const char *name = I->hasName() ? I->getName().data() : 0;

        a = lccrt_var_new( f, LCCRT_VAR_LOC_ARG, types[arg_num], name, 0, 0, 0);
        lvals.insert( PairValueToVar( I, a));
        lccrt_function_set_arg( f, arg_num, a);
        if ( Attrs.hasAttributeAtIndex( arg_num + 1, Attribute::NoAlias) )
        {
            lccrt_var_set_attr_restrict( a, 1);
        }
    }

    assert( arg_num == F->getFunctionType()->getNumParams());

    return;
} /* LccrtFunctionEmitter::makeArgs */

/**
 * Транслировать профильную информацию для операции условного перехода.
 */
void
LccrtFunctionEmitter::evalFuncProfile( const Function *F, lccrt_function_ptr f) {
    if ( MDNode *prof = F->getMetadata( LLVMContext::MD_prof) ) {
        if ( prof->getNumOperands() == 2 ) {
            auto *pname = dyn_cast<MDString>( prof->getOperand( 0));
            auto *pm = dyn_cast<ValueAsMetadata>( prof->getOperand( 1));
            const Value *pv = pm ? pm->getValue() : 0;
            const ConstantInt *p = pv ? dyn_cast<ConstantInt>( pv) : 0;

            if ( pname && p && (pname->getString() == "function_entry_count") ) {
                lccrt_einfo_reference_t eprof;
                uint64_t pi = p->getZExtValue();
               
                eprof = lccrt_einfo_new_struct( le.etyde_proffn);
                lccrt_einfo_set_field( eprof, le.eifi_proffn_iv, lccrt_einfo_new_i64( pi));
                lccrt_function_set_einfo( f, le.ecat_prof, eprof);
            }
        }
    }

    return;
} /* LccrtFunctionEmitter::evalFuncProfile */

void
LccrtFunctionEmitter::generateComment( const Instruction &O, lccrt_oper_iterator_ptr i) {
    if ( !isa<PHINode>( O) ) {
        lccrt_var_ptr a1, a2;
        lccrt_type_ptr t1, t2;
        std::string so;
        raw_string_ostream RSO( so);

        O.print( RSO, le.MST, true);
        t1 = lccrt_type_make_array( lccrt_type_make_u8( m), so.size() + 1);
        a1 = le.makeVarConst( t1, lccrt_varinit_new_str( t1, so.size(), so.data()));
        t2 = lccrt_type_make_pbyte( m);
        a2 = le.makeVarConst( t2, lccrt_varinit_new_addr_var( a1, 0));
        makeLibCallFast( "__lccrt_builtin_ircomment", a2, 0, 0, 0, i);
    }

    return;
} /* LccrtFunctionEmitter::generateComment */

void
LccrtFunctionEmitter::makeOpers()
{
    MapPHINodeToVar phis;
    MapBBToOper cts;
    MapInstToArgs oas;
    lccrt_oper_ptr lbl_work;
    arg_ref_t *alts_pool;
    int num_ct_alts = 0;
    int cur_ct_alts = 0;
    int is_start = 1;
    lccrt_oper_iterator_ptr i = lccrt_oper_iterator_new( c);

    evalFuncProfile( F, f);

    for ( Function::const_iterator I = F->begin(), IE = F->end(); I != IE; ++I )
    {
        lccrt_oper_ptr o;
        const BasicBlock *BB = &(*I);
        const char *bname = BB->hasName() ? BB->getName().data() : 0;

        //if ( BB->hasName() )
        {
            o = lccrt_oper_new_label( f, bname, 0);
            lbls.insert( PairBBToOper( BB, o));
        }
    }

    lbl_work = lccrt_oper_new_label( f, 0, 0);

#if 1
    if ( le.verbose_ir )
    {
        errs() << "==============\n";
        errs() << F->getName().str() << ":\n";
        errs().flush();
    }
#endif

    /**
     * Вычисляем размер памяти необходимой для альтернатив всех операций перехода.
     */
    for ( Function::const_iterator I = F->begin(), IE = F->end(); I != IE; ++I )
    {
        const BasicBlock *BB = &(*I);

        for ( BasicBlock::const_iterator J = BB->begin(), JE = BB->end(); J != JE; ++J )
        {
            num_ct_alts += get_num_normal_alts( *J);
        }
    }

    alts_pool = new arg_ref_t[num_ct_alts];

    for ( Function::iterator I = F->begin(), IE = F->end(); I != IE; ++I )
    {
        BasicBlock *BB = &(*I);
        int is_ct = 0;

        //printBasicBlock( I);

        lccrt_oper_iterator_set( i, lbls.find( BB)->second);
        for ( BasicBlock::iterator J = BB->begin(), JE = BB->end(); J != JE; ++J )
        {
            Instruction &O = *J;
            lccrt_var_ptr res = 0;
            lccrt_type_ptr tr = 0;
            lccrt_oper_ptr ct = 0;
            arg_ref_t *alts = alts_pool + cur_ct_alts;
            Value *V1 = O.getNumOperands() ? O.getOperand(0) : 0;

            assert( !is_ct);
#if 1
            if ( le.verbose_ir )
            {
                errs() << "  -------------\n" << O << "\n";
                errs().flush();
                // printInstruction();
            }
#endif

            //generateComment( O, i);
           
            if ( !O.getType()->isVoidTy() ) {
                res = makeValue( cast<Value>(&O));
                tr = lccrt_var_get_type( res);
            }

            if ( (isa<AtomicCmpXchgInst>(O)
                  && cast<AtomicCmpXchgInst>(O).isVolatile())
                 || (isa<AtomicRMWInst>(O)
                     && cast<AtomicRMWInst>(O).isVolatile()) )
            {
                //assert( 0);
            }

            if ( isa<BranchInst>(O) || isa<SwitchInst>(O) ) {
                is_ct = 1;
                assert( cur_ct_alts + get_num_normal_alts( O) <= num_ct_alts);
                cur_ct_alts += makeBranch( O, ct, res, alts, i);

            } else if ( isa<IndirectBrInst>(O) ) {
                is_ct = 1;
                assert( cur_ct_alts + get_num_normal_alts( O) <= num_ct_alts);
                cur_ct_alts += makeIndirectBranch( O, ct, res, alts, i);

            } else if ( isa<CallInst>(O) ) {
                makeCall( O, res, i);

            } else if ( isa<InvokeInst>(O) ) {
                is_ct = 1;
                assert( cur_ct_alts + get_num_normal_alts( O) <= num_ct_alts);
                cur_ct_alts += makeInvoke( O, ct, res, alts, i);

            } else if ( isa<LandingPadInst>(O) )
            {
                makeLandingpad( O, res, i);

            } else if ( isa<ResumeInst>(O) )
            {
                makeResume( O, res, i);

            } else if ( isa<CmpInst>(O) )
            {
                makeCmp( O, res, i);

            } else if ( isa<LoadInst>(O)
                        || isa<StoreInst>(O) )
            {
                makeLoadStore( O, res, i);

            } else if ( isa<CastInst>(O) ) {
                makeCast( O, res, i);

            } else if ( isa<UnaryOperator>(O) )
            {
                makeArith1( O.getOpcode(), O, res, i);

            } else if ( isa<BinaryOperator>(O) ) {
                makeArith2( O.getOpcode(), O, res, i);

            } else if ( isa<ReturnInst>(O) )
            {
                is_ct = 1;
                ct = lccrt_oper_new_return( f, V1 ? makeValue( V1, i) : 0, i);
                //dbge_.makeOperDbgMetadata( ct, &O);
                dbgle_.makeOperDbgMetadata( ct, &O);

            } else if ( isa<GetElementPtrInst>(O) ) {
                makeGetelementptr( O, res, i);

            } else if ( isa<PHINode>(O) )
            {
                const PHINode &FI = cast<PHINode>(O);

                phis.insert( PairPHINodeToVar( &FI, res));

            } else if ( isa<AllocaInst>(O) ) {
                makeAlloca( O, res, is_start, i);

            } else if ( isa<SelectInst>(O) )
            {
                makeSelect( O, res, i);

            } else if ( isa<ExtractValueInst>(O) )
            {
                makeExtractvalue( O, res, i);

            } else if ( isa<InsertValueInst>(O) )
            {
                makeInsertvalue( O, res, i);

            } else if ( isa<ExtractElementInst>(O) ) {
                makeExtractelement( O, res, i);

            } else if ( isa<InsertElementInst>(O) )
            {
                makeInsertelement( O, res, i);

            } else if ( isa<ShuffleVectorInst>(O) )
            {
                makeShufflevector( O, res, i);

            } else if ( isa<VAArgInst>(O) )
            {
                makeVaArg( O, res, i);

            } else if ( isa<FenceInst>(O) )
            {
                makeFence( O, res, i);

            } else if ( isa<AtomicCmpXchgInst>(O) )
            {
                makeCmpXchg( O, res, i);

            } else if ( isa<AtomicRMWInst>(O) )
            {
                makeAtomicrmw( O, res, i);

            } else if ( isa<UnreachableInst>(O) )
            {
                is_ct = 1;
                lccrt_oper_new_return( f, 0, i);

            } else if ( isa<FreezeInst>(O) )
            {
                lccrt_oper_new_move( f, makeValuePtrcast( V1, tr, i), res, i);
            } else
            {
                errorDump( &O);
            }

            assert( cur_ct_alts <= num_ct_alts);

            if ( is_ct )
            {
                //lccrt_oper_ptr lo = lccrt_oper_iterator_get_prev( i);

                assert( cts.find( BB) == cts.end());
                //cts.insert( PairBBToOper( BB, lo));
                cts.insert( PairBBToOper( BB, ct));
                oas.insert( PairInstToArgs( &O, alts));
                is_ct = 0;
            }
        }

        is_start = 0;
    }

    for ( Function::iterator I = F->begin(), IE = F->end(); I != IE; ++I )
    {
        BasicBlock *BB0 = &(*I);
        BasicBlock::iterator J = BB0->begin();
        BasicBlock::iterator JE = BB0->end();

        if ( (J != JE) && isa<PHINode>( *J) ) {
            if ( BB0->isLandingPad() ) {
                semiUnzipLandingpad( BB0, oas, lbl_work, i);
            }

            /* Реализуем все фи-узлы в текущем линейном участке. */
            for ( ; (J != JE) && isa<PHINode>( *J); ++J )
            {
                PHINode &FI = cast<PHINode>( *J);
                Value *V = dyn_cast<Value>( &FI);

                makePhi( FI, oas, lbl_work, makeValue( V), i);
            }
        }
    }

    delete[] alts_pool;
    alts_pool = 0;

    lccrt_oper_iterator_delete( i);

    return;
} /* LccrtFunctionEmitter::makeOpers */

/**
 * Создание операции elemptr для операции getelementptr.
 */
void
LccrtFunctionEmitter::makeGetelementptr( User &E, lccrt_var_ptr v, lccrt_oper_iterator_ptr i)
{
    lccrt_oper_ptr res = 0;
    unsigned num_args = E.getNumOperands();
    GetElementPtrInst *G = dyn_cast<GetElementPtrInst>( &E);
    GEPOperator *GEP = dyn_cast<GEPOperator>( &E);
    bool is_vector = false;
    int *num_elems = new int[num_args];
    int max_elems = 1;
    lccrt_type_ptr tu64 = lccrt_type_make_u64( m);

    for ( unsigned k = 0; k < num_args; ++k ) {
        Type *kTy = E.getOperand( k)->getType();

        num_elems[k] = 1;
        if ( isa<ScalableVectorType>( kTy) ) {
            errorDump( &E);

        } else if ( isa<FixedVectorType>( kTy) ) {
            FixedVectorType *VecTy = cast<FixedVectorType>( kTy);

            is_vector = true;
            num_elems[k] = VecTy->getNumElements();
            max_elems = (max_elems > 1) ? max_elems : num_elems[k];
            if ( (num_elems[k] != 1) && (num_elems[k] != max_elems) ) {
                errorDump( &E);
            }
        }
    }

    assert( is_vector == isa<FixedVectorType>( E.getType()));
    if ( !is_vector )
    {
        Type *Ty = G ? G->getSourceElementType() : GEP->getSourceElementType();
        uint64_t alloc_size = le.DL.getTypeAllocSize( Ty);
        uint64_t store_size = le.DL.getTypeStoreSize( Ty);
        lccrt_var_ptr *args = new lccrt_var_ptr[num_args];
        lccrt_type_ptr tp = lccrt_type_make_ptr_type( le.makeType( Ty));

        args[0] = makeValuePtrcast( E.getOperand( 0), tp, i);
        for ( unsigned k = 1; k < num_args; ++k ) {
            args[k] = makeValue( E.getOperand( k), i);
        }

        if ( ((0 && (num_args == 2))
              || (alloc_size != store_size)) )
        {
            lccrt_oper_ptr mul, cnv, add, sxt;
            lccrt_type_ptr type0 = lccrt_var_get_type( args[0]);
            lccrt_type_ptr type1 = lccrt_var_get_type( args[1]);
            lccrt_type_ptr type2 = lccrt_type_make_int( m, lccrt_type_get_bytesize( type1), 1);
            lccrt_type_ptr type3 = lccrt_type_make_int( m, lccrt_type_get_bytesize( type0), 1);
            lccrt_var_ptr res_var = (num_args == 2) ? v : 0;

            mul = lccrt_oper_new_mul( f, args[1], le.makeVarConstHex( type1, alloc_size), 0, i);
            cnv = lccrt_oper_new_bitcast( f, args[0], lccrt_type_make_pbyte( m), 0, i);
            sxt = lccrt_oper_new_sext( f, lccrt_oper_get_res( mul), type2, 0, i);
            sxt = lccrt_oper_new_sext( f, lccrt_oper_get_res( sxt), type3, 0, i);
            add = lccrt_oper_new_add( f, lccrt_oper_get_res( cnv), lccrt_oper_get_res( sxt), 0, i);
            res = lccrt_oper_new_bitcast( f, lccrt_oper_get_res( add), type0, res_var, i);
            args[0] = lccrt_oper_get_res( res);
            if ( (num_args > 2) ) {
                args[1] = le.makeVarConstHex( type1, 0);
            } else {
                args[1] = 0;
            }
        }

        if ( (num_args > 2)
             || (alloc_size == store_size) )
        {
            res = lccrt_oper_new_elemptr( f, num_args, args, v, i);
        }

        v = lccrt_oper_get_res( res);
        delete[] args;

    } else if ( is_vector )
    {
        lccrt_var_ptr *args = new lccrt_var_ptr[num_args];
        lccrt_var_ptr *jargs = new lccrt_var_ptr[num_args];

        errorDump( &E);
        assert( (int)cast<FixedVectorType>( E.getType())->getNumElements() == max_elems);

        // Готовим аргументы для формирования элементов адресных цепочек.
        for ( int k = 0; k < (int)num_args; ++k )
        {
            args[k] = makeValue( E.getOperand( k), i);
        }

        // Для каждой координаты вектора строим цепочку адресации.
        for ( int j = 0; j < max_elems; ++j )
        {
            lccrt_var_ptr raj;
            struct { lccrt_var_ptr v[3]; } eargs;

            // Формируем аргументы для текущей цепочки.
            for ( int k = 0; k < (int)num_args; ++k )
            {
                Type *kTy = E.getOperand( k)->getType();

                if ( !isa<FixedVectorType>( kTy) )
                {
                    // Скаляр рассматриваем как вектор из одинаковых значений.
                    jargs[k] = args[k];
                } else
                {
                    assert_define( FixedVectorType *VecTy = cast<FixedVectorType>( kTy));

                    // Читаем соответствующую координату для очереднего элемента цепочки.
                    assert( max_elems = VecTy->getNumElements());
                    eargs = {{args[k], le.makeVarConstHex( tu64, j)}};
                    jargs[k] = lccrt_oper_get_res( lccrt_oper_new_elemread( f, 2, eargs.v, 0, i));
                }
            }

            // Формируем очередную цепочку адресации.
            raj = lccrt_oper_get_res( lccrt_oper_new_elemptr( f, num_args, jargs, 0, i));

            // Сохраняем результат цепочки адресации в текущую координату результата.
            eargs = {{v, raj, le.makeVarConstHex( tu64, j)}};
            lccrt_oper_new_elemwrite( f, 3, eargs.v, i);
        }

        delete[] args;
        delete[] jargs;
    } else
    {
        printf( "\n%s:%s:%d\n  Wrong getelementptr arguments types\n\n", __FUNCTION__, __FILE__, __LINE__);
        errorDump( &E);
    }

    delete[] num_elems;

    return;
} /* LccrtFunctionEmitter::makeGetelementptr */

/**
 * Создание константного выражения.
 */
lccrt_var_ptr
LccrtFunctionEmitter::makeValueConstExpr( lccrt_var_ptr v, ConstantExpr *E, lccrt_oi_ptr i)
{
    lccrt_type_ptr t = le.makeType( E->getType());

    if ( (E->getOpcode() == Instruction::GetElementPtr) )
    {
        bool is_regular = false;
        APInt offset( 64, 0);
        GetElementPtrInst *G = dyn_cast<GetElementPtrInst>( E->getAsInstruction());
        GlobalValue *e0 = dyn_cast<GlobalValue>( G->getOperand( 0));
        auto *COff = G->stripAndAccumulateConstantOffsets( le.DL, offset, true);
        auto *GV0 = dyn_cast<GlobalVariable>( COff);

        if ( e0 && GV0 && (offset.getSExtValue() >= 0) ) {
            lccrt_type_ptr t = le.makeType( E->getType());

            is_regular = true;
            v = le.makeVarConst( t, le.makeVarinit( e0, offset.getSExtValue()));
        }

        delete G;
        if ( !is_regular ) {
            makeGetelementptr( *E, v, i);
        }
    } else if ( (E->getOpcode() == Instruction::BitCast)
                || (E->getOpcode() == Instruction::IntToPtr)
                || (E->getOpcode() == Instruction::PtrToInt) )
    {
        lccrt_var_ptr a = makeValue( E->getOperand( 0), i);

        lccrt_oper_new_bitcast( f, a, t, v, i);

    } else if ( (E->getOpcode() == Instruction::Trunc)
                || (E->getOpcode() == Instruction::SExt)
                || (E->getOpcode() == Instruction::ZExt) )
    {
        CastInst *CI;
        
        CI = CastInst::Create( (Instruction::CastOps)E->getOpcode(),
                               E->getOperand( 0), E->getType());
        makeCast( *CI, v, i);
        delete CI;

    } else if ( (E->getOpcode() == Instruction::FNeg) )
    {
        makeArith1( E->getOpcode(), *E, v, i);

    } else if ( (E->getOpcode() == Instruction::Add)
                || (E->getOpcode() == Instruction::Sub)
                || (E->getOpcode() == Instruction::Mul)
                || (E->getOpcode() == Instruction::UDiv)
                || (E->getOpcode() == Instruction::SDiv)
                || (E->getOpcode() == Instruction::Shl)
                || (E->getOpcode() == Instruction::LShr)
                || (E->getOpcode() == Instruction::AShr)
                || (E->getOpcode() == Instruction::And)
                || (E->getOpcode() == Instruction::Or)
                || (E->getOpcode() == Instruction::Xor)
                || (E->getOpcode() == Instruction::FAdd)
                || (E->getOpcode() == Instruction::FSub)
                || (E->getOpcode() == Instruction::FMul)
                || (E->getOpcode() == Instruction::FDiv)
                || (E->getOpcode() == Instruction::URem)
                || (E->getOpcode() == Instruction::SRem) )
    {
        makeArith2( E->getOpcode(), *E, v, i);

    } else if ( (E->getOpcode() == Instruction::ICmp) )
    {
        lccrt_var_ptr a = makeValue( E->getOperand( 0), i);
        lccrt_var_ptr b = makeValue( E->getOperand( 1), i);
        const CmpInst *CI = cast<CmpInst>(E);
        lccrt_cmp_name_t cn = getCmpLccrtName( CI->getPredicate());
        lccrt_varinit_ptr n = lccrt_varinit_new_scalar( lccrt_type_make_u32( m), cn);
        Type *T1 = E->getOperand( 0)->getType();

        if ( isa<PointerType>( T1) )
        {
            lccrt_type_ptr t = lccrt_type_make_intptr( m);

            a = lccrt_oper_get_res( lccrt_oper_new_bitcast( f, a, t, 0, i));
            b = lccrt_oper_get_res( lccrt_oper_new_bitcast( f, b, t, 0, i));
        }

        lccrt_oper_new_cmp( f, n, a, b, v, i);

    } else if ( (E->getOpcode() == Instruction::Select) )
    {
        lccrt_var_ptr a = makeValue( E->getOperand( 0), i);
        lccrt_var_ptr b = makeValue( E->getOperand( 1), i);
        lccrt_var_ptr c = makeValue( E->getOperand( 2), i);

        lccrt_oper_new_select( f, a, b, c, v, i);
    } else
    {
        errorDump( E);
    }

    return (v);
} /* LccrtFunctionEmitter::makeValueConstExpr */

/**
 * Создание константного выражения.
 */
lccrt_var_ptr
LccrtFunctionEmitter::makeValueConst( lccrt_var_ptr v, Constant *C, lccrt_oi_ptr i)
{
    lccrt_type_ptr t = lccrt_var_get_type( v);
    lccrt_type_ptr tu64 = lccrt_type_make_u64( m);

    if ( ConstantExpr *E = dyn_cast<ConstantExpr>(C) ) {
        v = makeValueConstExpr( v, E, i);

    } else if ( GlobalAlias *a = dyn_cast<GlobalAlias>( C) ) {           
        Constant *av = a->getAliasee();

        if ( av ) {
            v = makeValueConst( v, av, i);
        } else {
            errorDump( a);
        }
    } else if ( (isa<ConstantDataArray>( C)
                 && !dyn_cast<ConstantDataArray>( C)->isString())
                || isa<ConstantDataVector>( C) )
    {
        ConstantDataSequential *cds = dyn_cast<ConstantDataSequential>( C);
        uint64_t n = cds->getNumElements();

        for ( uint64_t j = 0; j < n; ++j )
        {
            struct { lccrt_var_ptr v[3]; } args;
            Constant *e = static_cast<Constant *>( cds->getElementAsConstant( j));
            lccrt_type_ptr tj = le.makeType( e->getType());
            lccrt_var_ptr vj = lccrt_var_new_local( f, tj, 0);

            args = {{v, makeValueConst( vj, e, i), le.makeVarConstHex( tu64, j)}};
            lccrt_oper_new_elemwrite( f, 3, args.v, i);
        }
    } else if ( isa<ConstantArray>( C)
                || isa<ConstantStruct>( C)
                || isa<ConstantVector>( C) )
    {
        uint64_t n = C->getNumOperands();

        for ( uint64_t j = 0; j < n; ++j )
        {
            struct { lccrt_var_ptr v[3]; } args;
            Constant *e = static_cast<Constant *>( C->getOperand( j));
            lccrt_type_ptr tj = le.makeType( e->getType());
            lccrt_var_ptr vj = lccrt_var_new_local( f, tj, 0);

            args = {{v, makeValueConst( vj, e, i), le.makeVarConstHex( tu64, j)}};
            lccrt_oper_new_elemwrite( f, 3, args.v, i);
        }
    } else {
        v = le.makeVarConst( t, le.makeVarinit( C, 0));
    }

    return (v);
} /* LccrtFunctionEmitter::makeValueConst */

/**
 * Создание аргумента операции.
 */
lccrt_var_ptr
LccrtFunctionEmitter::makeValuePtrcast( Value *V, lccrt_type_ptr rtype, lccrt_oi_ptr i)
{
    lccrt_var_ptr a = makeValue( V, i);
    lccrt_var_ptr r = a;
    lccrt_type_ptr ta = lccrt_var_get_type( a);

    if ( lccrt_type_is_pointer( ta) ) {
        assert( lccrt_type_is_pointer( rtype));
        if ( ta != rtype ) {
            r = lccrt_var_new_local( f, rtype, 0);
            lccrt_oper_new_bitcast( f, a, rtype, r, i);
        }
    } else {
        assert( ta == rtype);
    }

    return (r);
} /* LccrtFunctionEmitter::makeValuePtrcast */

/**
 * Создание аргумента операции.
 */
lccrt_var_ptr
LccrtFunctionEmitter::makeValue( Value *V, lccrt_oi_ptr i, bool noncarg)
{
    lccrt_var_ptr v = 0;
    MapValueToVar::const_iterator l = lvals.find( V);

    if ( !V ) {
        ;
    } else if ( isa<GlobalVariable>(V) ) {
        v = le.makeGlobal( dyn_cast<GlobalVariable>(V), false);
    } else if ( !isa<ConstantExpr>(V) && (l != lvals.end()) ) {
        v = l->second;
    } else {
        lccrt_type_ptr t = le.makeValueType( V);
        InsertValueInst *IVI = dyn_cast<InsertValueInst>(V);
        InsertValueInst *IVI0 = IVI ? dyn_cast<InsertValueInst>(IVI->getOperand(0)) : 0;

        if ( isa<Constant>(V) ) {
            Constant *C = dyn_cast<Constant>(V);

            if ( le.testVarinit( C, 0) ) {
                v = le.makeVarConst( t, le.makeVarinit( C, 0));
                if ( noncarg ) {
                    lccrt_oper_ptr o = lccrt_oper_new_move( f, v, 0, i);

                    v = lccrt_oper_get_res( o);
                }
            } else {
                v = lccrt_var_new_local( f, t, 0);
                v = makeValueConst( v, C, i);
                return (v);
            }
        } else if ( isa<InlineAsm>(V) ) {
            assert( 0);
        } else if ( IVI && IVI0 && IVI0->hasOneUse() ) {
            v = makeValue( IVI0, i, noncarg);
            return (v);
        } else {
            const char *name = V->hasName() ? V->getName().data() : 0;

            assert( !isa<GlobalValue>(V));
            v = lccrt_var_new_local( f, t, name);
        }
        
        lvals.insert( PairValueToVar( V, v));
    }

    return (v);
} /* LccrtFunctionEmitter::makeValue */

/**
 * Создание локальной переменной для байтового преобразования типов.
 */
lccrt_var_ptr
LccrtFunctionEmitter::makeBitcastVar( Type *TA, Type *TR)
{
    lccrt_var_ptr v = 0;
    lccrt_type_ptr t = 0;
    lccrt_type_ptr tf[3] = {0, 0, 0};
    int data_size = 0;
    int dwlen = (le.DL.getTypeAllocSize( TA) + 7)/8;

    tf[0] = le.makeType( TA);
    tf[0] = lccrt_type_make_field( tf[0], 1, 0, 0, lccrt_type_get_bytesize( tf[0]) * 8);

    tf[1] = le.makeType( TR);
    tf[1] = lccrt_type_make_field( tf[1], 1, 0, 0, lccrt_type_get_bytesize( tf[1]) * 8);

    tf[2] = lccrt_type_make_array( lccrt_type_make_u64( m), dwlen);
    data_size = lccrt_type_get_bytesize( tf[2]);
    tf[2] = lccrt_type_make_field( tf[2], 1, 0, 0, data_size * 8);
    tf[2] = lccrt_type_make_struct( m, 8, data_size, 1, tf + 2, 0);
    tf[2] = lccrt_type_make_field( tf[2], 1, 0, 0, data_size * 8);

    t = lccrt_type_make_struct( m, 8, data_size, 3, tf, 1);
    v = lccrt_var_new_local( f, t, 0);

    return (v);
} /* LccrtFunctionEmitter::makeBitcastVar */

/**
 * Создание локальной переменной для преобразования структуры, реализующей iN, в целый тип
 * при N <= 128.
 */
lccrt_var_ptr
LccrtFunctionEmitter::makeNIntLocalVar( int bitsize)
{
    lccrt_var_ptr v = 0;
    lccrt_type_ptr t = 0;
    lccrt_type_ptr tf[2] = {0, 0};
    int data_size = (bitsize + 7) / 8;
    int align = get_int_align( data_size);
    int alloc_size = get_int_align_size( data_size);

    assert( (0 < bitsize) && (bitsize <= 128));

    tf[0] = lccrt_type_make_array( lccrt_type_make_u8( m), data_size);
    tf[0] = lccrt_type_make_field( tf[0], 1, 0, 0, data_size * 8);
    tf[0] = lccrt_type_make_struct( m, align, data_size, 1, tf, 0);
    tf[0] = lccrt_type_make_field( tf[0], 1, 0, 0, data_size * 8);
    tf[1] = lccrt_type_make_int( m, alloc_size, 0);
    tf[1] = lccrt_type_make_field( tf[1], 1, 0, 0, alloc_size * 8);
    t = lccrt_type_make_struct( m, align, alloc_size, 2, tf, 1);

    v = lccrt_var_new_local( f, t, 0);

    return (v);
} /* LccrtFunctionEmitter::makeNIntLocalVar */

/**
 * Преобразование имени операции сравнения.
 */
int
LccrtEmitter::getTypeBitsize( Type *Ty)
{
    int r = 0;

    if ( Ty->isIntegerTy() ) {
        r = Ty->getIntegerBitWidth();
    } else {
        r = 8*lccrt_type_get_bytesize( makeType( Ty));
        if ( isa<FixedVectorType>( Ty) ) {
            Type *Te = dyn_cast<FixedVectorType>( Ty)->getElementType();

            if ( Te->isIntegerTy() )
            {
                assert_define( int esize = Te->getIntegerBitWidth());

                assert( (esize == 1) || (esize % 8 == 0));
            }
        }
    }

    return (r);
} /* LccrtEmitter::getTypeBitsize */

/**
 * Преобразование имени операции сравнения.
 */
lccrt_cmp_name_t
LccrtFunctionEmitter::getCmpLccrtName( unsigned opcode, const char **ps)
{
    const char *sn = 0;
    lccrt_cmp_name_t cn = LCCRT_CMP_LAST;

    switch ( opcode )
    {
      //case FCmpInst::FCMP_FALSE: cn = ; break;
      case FCmpInst::FCMP_OEQ:   cn = LCCRT_CMP_EQ_FO; sn = "_eq_fo"; break;
      case FCmpInst::FCMP_OGT:   cn = LCCRT_CMP_GT_FO; sn = "_gt_fo"; break;
      case FCmpInst::FCMP_OGE:   cn = LCCRT_CMP_GE_FO; sn = "_ge_fo"; break;
      case FCmpInst::FCMP_OLT:   cn = LCCRT_CMP_LT_FO; sn = "_lt_fo"; break;
      case FCmpInst::FCMP_OLE:   cn = LCCRT_CMP_LE_FO; sn = "_le_fo"; break;
      case FCmpInst::FCMP_ONE:   cn = LCCRT_CMP_NE_FO; sn = "_ne_fo"; break;
      case FCmpInst::FCMP_ORD:   cn = LCCRT_CMP_FO;    sn = "_fo"; break;
      case FCmpInst::FCMP_UNO:   cn = LCCRT_CMP_FU;    sn = "_fu"; break;
      case FCmpInst::FCMP_UEQ:   cn = LCCRT_CMP_EQ_FU; sn = "_eq_fu"; break;
      case FCmpInst::FCMP_UGT:   cn = LCCRT_CMP_GT_FU; sn = "_gt_fu"; break;
      case FCmpInst::FCMP_UGE:   cn = LCCRT_CMP_GE_FU; sn = "_ge_fu"; break;
      case FCmpInst::FCMP_ULT:   cn = LCCRT_CMP_LT_FU; sn = "_lt_fu"; break;
      case FCmpInst::FCMP_ULE:   cn = LCCRT_CMP_LE_FU; sn = "_le_fu"; break;
      case FCmpInst::FCMP_UNE:   cn = LCCRT_CMP_NE_FU; sn = "_ne_fu"; break;
      //case FCmpInst::FCMP_TRUE:  cn = ; break;
      case ICmpInst::ICMP_EQ:    cn = LCCRT_CMP_EQ;    sn = "_eq_i"; break;
      case ICmpInst::ICMP_NE:    cn = LCCRT_CMP_NE;    sn = "_ne_i"; break;
      case ICmpInst::ICMP_SGT:   cn = LCCRT_CMP_GT_I;  sn = "_gt_i"; break;
      case ICmpInst::ICMP_SGE:   cn = LCCRT_CMP_GE_I;  sn = "_ge_i"; break;
      case ICmpInst::ICMP_SLT:   cn = LCCRT_CMP_LT_I;  sn = "_lt_i"; break;
      case ICmpInst::ICMP_SLE:   cn = LCCRT_CMP_LE_I;  sn = "_le_i"; break;
      case ICmpInst::ICMP_UGT:   cn = LCCRT_CMP_GT_U;  sn = "_gt_u"; break;
      case ICmpInst::ICMP_UGE:   cn = LCCRT_CMP_GE_U;  sn = "_ge_u"; break;
      case ICmpInst::ICMP_ULT:   cn = LCCRT_CMP_LT_U;  sn = "_lt_u"; break;
      case ICmpInst::ICMP_ULE:   cn = LCCRT_CMP_LE_U;  sn = "_le_u"; break;
      default: assert( 0); break;
    }

    if ( ps )
    {
        (*ps) = sn;
    }

    return (cn);
} /* LccrtFunctionEmitter::getCmpLccrtName */

/**
 * Найти lccrt-метку для узла.
 */
lccrt_oper_ptr
LccrtFunctionEmitter::findBlockLabel( const BasicBlock *BB)
{
    lccrt_oper_ptr r = 0;
    MapBBToOper::const_iterator k = lbls.find( BB);

    if ( (k != lbls.end()) )
    {
        r = k->second;
    }

    return (r);
} /* LccrtFunctionEmitter::findBlockLabel */

/**
 * Создание вызова из библиотеки поддержки с передачей операндов по косвенности.
 */
lccrt_type_ptr
LccrtFunctionEmitter::makeLibCallType( lccrt_module_ptr m, int n, bool is_vec)
{
    int k;
    lccrt_type_ptr tf = 0;
    lccrt_type_ptr targs[64] = {0};
    lccrt_type_ptr tv = lccrt_type_make_void( m);
    lccrt_type_ptr tvp = lccrt_type_make_ptr_type( tv);
    lccrt_type_ptr tu32 = lccrt_type_make_u32( m);

    assert( (n == 1) || (n == 2) || (n == 3));

    for ( k = 0; k < n + 1; k++ )
    {
        /* Тип параметра с битовым размером фактического аргумента/результата. */
        targs[k] = tu32;
        /* Тип параметра, указывающего на фактический аргумент/результата. */
        targs[k + (n + 1)] = tvp;
    }

    if ( is_vec )
    {
        for ( k = 0; k < n + 1; ++k )
        {
            targs[k + 2*(n + 1)] = tu32;
            targs[k + 3*(n + 1)] = tu32;
        }
    }    

    tf = lccrt_type_make_func( tv, (2 + 2*is_vec)*(n + 1), targs);

    return (tf);
} /* makeLibCallType */

/**
 * Создание вызова из библиотеки поддержки с передачей операндов по косвенности.
 */
Value *
LccrtFunctionEmitter::getShuffleMaskVector( const User &O)
{
    SmallVector<int, 16> mask;
    Value *r = 0;
    const ShuffleVectorInst &SV = cast<ShuffleVectorInst>(O);
    LLVMContext &lctx = F->getContext();
    Type *ty32i = Type::getInt32Ty( lctx);
    std::vector<Constant *> cvec;

    SV.getShuffleMask( mask);
    for ( int k = 0; k < (int)mask.size(); ++k )
    {
        cvec.push_back( ConstantInt::get( ty32i, (mask[k] < 0) ? 0 : mask[k]));
    }

    r = ConstantVector::get( cvec);

    return (r);
} /* LccrtFunctionEmitter::getShuffleMaskVector */

/**
 * Создание вызова из библиотеки поддержки с передачей операндов по косвенности.
 */
void
LccrtFunctionEmitter::makeLibCall( const char *func_name, bool is_vec, const User &O,
                                   lccrt_var_ptr res, bool ignore_last_arg, lccrt_oi_ptr i)
{
    int k;
    lccrt_var_ptr args[64];
    lccrt_link_t lnk;
    std::map<const std::string, lccrt_function_ptr>::const_iterator ah;
    lccrt_function_ptr af = 0;
    lccrt_type_ptr tf = 0;
    lccrt_type_ptr tfp = 0;
    lccrt_var_ptr a[64] = {0};
    lccrt_var_ptr b[64] = {0};
    lccrt_var_ptr p[64] = {0};
    lccrt_var_ptr q[64] = {0};
    int w[64] = {0};
    lccrt_type_ptr tv = lccrt_type_make_void( m);
    lccrt_type_ptr tvp = lccrt_type_make_ptr_type( tv);
    lccrt_type_ptr tu32 = lccrt_type_make_u32( m);
    bool is_call = isa<CallInst>( O);
    bool is_shuffle = isa<ShuffleVectorInst>( O);
    int n = O.getNumOperands();
    Type *T = O.getType();
    std::string func_name_s( func_name);
    Value **operands = new Value*[n+1];

    n = is_call ? (n - 1) : n;
    if ( ignore_last_arg ) --n;
    assert( (n == 1) || (n == 2) || (n == 3));

    /* Для операции shufflevector создаем операнд для маски. */
    for ( k = 0; k < n; ++k ) operands[k] = O.getOperand( k);
    if ( is_shuffle )
    {
        n++;
        operands[n-1] = getShuffleMaskVector( O);
    }

    /* Битовый размер фактического результата. */
    w[0] = le.getTypeBitsize( T);
    for ( k = 0; k < n; k++ )
    {
        /* Значение фактического аргумента. */
        a[k] = makeValue( operands[k], i);
        /* Битовый размер фактического аргумента. */
        w[k+1] = le.getTypeBitsize( operands[k]->getType());
    }

    tf = makeLibCallType( m, n, is_vec);
    tfp = lccrt_type_make_ptr_type( tf);

    /* Создаем временные переменные для передачи значений фактических результата/аргументов. */
    //b[0] = makeVarRes( le.makeType( T));
    b[0] = allocLocalTmpVar( le.makeType( T));
    for ( k = 0; k < n; ++k )
    {
        //b[k+1] = lccrt_oper_get_res( lccrt_oper_new_move( f, a[k], 0, i));
        b[k+1] = allocLocalTmpVar( lccrt_var_get_type( a[k]));
        lccrt_oper_new_move( f, a[k], b[k+1], i);
    }

    /* Берем адреса на временные переменные. */
    for ( k = 0; k < n + 1; ++k )
    {
        p[k] = lccrt_oper_get_res( lccrt_oper_new_varptr( f, b[k], 0, i));
    }

    /* Адреса временных переменных используем в качестве параметров вызова. */
    for ( k = 0; k < n + 1; ++k )
    {
        q[k] = lccrt_oper_get_res( lccrt_oper_new_bitcast( f, p[k], tvp, 0, i));
    }

    ah = le.lib_funcs.find( func_name_s);
    lnk = le.makeLink( GlobalValue::ExternalLinkage, GlobalValue::DefaultVisibility,
                       GlobalVariable::NotThreadLocal, 0, 0, 1);
    if ( (ah != le.lib_funcs.end()) )
    {
        af = ah->second;
        assert( lnk == lccrt_function_get_link( af));
    } else
    {
        af = lccrt_function_new( m, tf, func_name, 0, lnk, 1, 0);
        le.lib_funcs.insert( std::pair<const std::string, lccrt_function_ptr>( func_name_s, af));
        lccrt_function_set_attr_does_not_throw( af, 1);
    }

    /* Первый параметр - адрес вызываемой функции. */
    args[0] = le.makeVarConst( tfp, lccrt_varinit_new_addr_func( af, 0));
    for ( k = 0; k < n + 1; k++ )
    {
        /* Формируем параметры с размером фактических аргументов/результата. */
        args[k + 1] = le.makeVarConst( tu32, lccrt_varinit_new_scalar( tu32, w[k]));
        args[k + 1 + (n + 1)] = q[k];
        if ( is_vec )
        {
            lccrt_varinit_ptr vi1 = 0;
            lccrt_varinit_ptr vi2 = 0;
            Type *Tk = (k == 0) ? T : operands[k - 1]->getType();
            FixedVectorType *TV = dyn_cast<FixedVectorType>( Tk);

            if ( TV ) {
                vi1 = lccrt_varinit_new_scalar( tu32, TV->getNumElements());
                vi2 = lccrt_varinit_new_scalar( tu32, le.getTypeBitsize( TV->getElementType()));
            } else {
                vi1 = lccrt_varinit_new_scalar( tu32, 1);
                vi2 = lccrt_varinit_new_scalar( tu32, le.getTypeBitsize( Tk));
            }

            args[k + 1 + 2*(n + 1)] = le.makeVarConst( tu32, vi1);
            args[k + 1 + 3*(n + 1)] = le.makeVarConst( tu32, vi2);
        }
    }

    lccrt_oper_new_call( f, tf, (2 + 2*is_vec)*(n + 1) + 1, args, 0, 0, 0, i);
    lccrt_oper_new_move( f, b[0], res, i);

    // Освобождаем временные переменные.
    for ( int k = 0; k <= n; ++k ) {
        freeLocalTmpVar( b[k]);
    }

    delete[] operands;

    return;
} /* LccrtFunctionEmitter::makeLibCall */

/**
 * Создание вызова из библиотеки поддержки с передачей операндов по значению.
 */
void
LccrtFunctionEmitter::makeLibCallFast( const char *func_name, const User &O,
                                       lccrt_var_ptr res, lccrt_oi_ptr i)
{
    int k;
    lccrt_var_ptr args[64];
    lccrt_type_ptr tfa[64];
    lccrt_function_ptr af = 0;
    lccrt_type_ptr tf = 0;
    lccrt_type_ptr tfp = 0;
    int n = O.getNumOperands();
    std::string func_name_s( func_name);

    assert( (n == 1) || (n == 2) || (n == 3));

    for ( k = 0; k < n; k++ )
    {
        tfa[k] = le.makeType( O.getOperand( k)->getType());
    }
    tf = lccrt_type_make_func( res ? lccrt_var_get_type( res) : le.makeType( O.getType()), n, tfa);
    tfp = lccrt_type_make_ptr_type( tf);

    af = le.makeFunctionFast( func_name, tf);

    /* Первый параметр - адрес вызываемой функции. */
    args[0] = le.makeVarConst( tfp, lccrt_varinit_new_addr_func( af, 0));
    for ( k = 0; k < n; k++ )
    {
        /* Значение фактического аргумента. */
        args[k + 1] = makeValue( O.getOperand( k), i);
    }

    lccrt_oper_new_call( f, tf, n + 1, args, 0, 0, res, i);

    return;
} /* LccrtFunctionEmitter::makeLibCallFast */

/**
 * Создание вызова из библиотеки поддержки с передачей операндов по значению.
 */
void
LccrtFunctionEmitter::makeLibCallFast( const char *func_name,
                                       lccrt_var_ptr arg1, lccrt_var_ptr arg2,
                                       lccrt_var_ptr arg3,
                                       lccrt_var_ptr res, lccrt_oi_ptr i)
{
    lccrt_var_ptr args[4] = {0, arg1, arg2, arg3};
    lccrt_type_ptr ta1 = arg1 ? lccrt_var_get_type( arg1) : 0;
    lccrt_type_ptr ta2 = arg2 ? lccrt_var_get_type( arg2) : 0;
    lccrt_type_ptr ta3 = arg3 ? lccrt_var_get_type( arg3) : 0;
    lccrt_type_ptr tfa[3] = {ta1, ta2, ta3};
    lccrt_function_ptr af = 0;
    lccrt_type_ptr tf = 0;
    lccrt_type_ptr tfp = 0;
    std::string func_name_s( func_name);
    int num_args = arg3 ? 3 : (arg2 ? 2 : (arg1 ? 1 : 0));

    tf = lccrt_type_make_func( lccrt_var_get_type( res), num_args, tfa);
    tfp = lccrt_type_make_ptr_type( tf);

    af = le.makeFunctionFast( func_name, tf);

    /* Первый параметр - адрес вызываемой функции. */
    args[0] = le.makeVarConst( tfp, lccrt_varinit_new_addr_func( af, 0));

    lccrt_oper_new_call( f, tf, 1 + num_args, args, 0, 0, res, i);

    return;
} /* LccrtFunctionEmitter::makeLibCallFast */

/**
 * Создание вызова из библиотеки поддержки с передачей операндов по значению.
 */
void
LccrtFunctionEmitter::makeLibCallFast( const char *func_name,
                                       const std::vector<lccrt_var_ptr> &iargs,
                                       lccrt_var_ptr res, lccrt_oi_ptr i)
{
    lccrt_function_ptr af = 0;
    lccrt_type_ptr tf = 0;
    lccrt_type_ptr tfp = 0;
    std::string func_name_s( func_name);
    int num_args = iargs.size();
    std::vector<lccrt_type_ptr> tfa( num_args);
    std::vector<lccrt_var_ptr> args( num_args + 1);
    lccrt_type_ptr tr = res ? lccrt_var_get_type( res) : lccrt_type_make_void( m);

    for ( int i = 0; i < num_args; ++i ) {
        tfa[i] = lccrt_var_get_type( iargs[i]);
        args[1 + i] = iargs[i];
    }

    tf = lccrt_type_make_func( tr, num_args, tfa.data());
    tfp = lccrt_type_make_ptr_type( tf);

    af = le.makeFunctionFast( func_name, tf);

    /* Первый параметр - адрес вызываемой функции. */
    args[0] = le.makeVarConst( tfp, lccrt_varinit_new_addr_func( af, 0));

    lccrt_oper_new_call( f, tf, 1 + num_args, args.data(), 0, 0, res, i);

    return;
} /* LccrtFunctionEmitter::makeLibCallFast */

/**
 * Создание переменной под временный результат.
 */
lccrt_var_ptr
LccrtFunctionEmitter::makeVarRes( lccrt_type_ptr type)
{
    lccrt_var_ptr v = lccrt_var_new_local( f, type, 0);

    return (v);
} /* LccrtFunctionEmitter::makeVarRes */

/**
 * Выделение локальной переменной для временного использования (например, параметры
 * операции call).
 */
lccrt_var_ptr
LccrtFunctionEmitter::allocLocalTmpVar( lccrt_type_ptr type) {
    lccrt_var_ptr r = 0;
    VecVars *vecs = 0;
    auto it = locals_type_pool.find( type);

    if ( it == locals_type_pool.end() ) {
        // Первый раз выделяем временную переменную с таким типом.
        vecs = new VecVars;
        locals_type_pool[type] = vecs;
    } else {
        // Для переменных указанного типа, уже создавались переменные.
        vecs = it->second;
    }

    if ( vecs->empty() ) {
        // Создаем новую временную переменную.
        r = makeVarRes( type);
    } else {
        // Переиспользуем ранее созданную переменную.
        r = vecs->back();
        vecs->pop_back();
    }

    return (r);
} /* LccrtFunctionEmitter::allocLocalTmpVar */

/**
 * Освобождение ранее созданной локальной переменной для временного использования.
 */
void
LccrtFunctionEmitter::freeLocalTmpVar( lccrt_var_ptr v) {
    lccrt_type_ptr type = lccrt_var_get_type( v);
    auto it = locals_type_pool.find( type);

    it->second->push_back( v);

    return;
} /* LccrtFunctionEmitter::freeLocalTmpVar */

/**
 * Создание операций, преобразующих iN в соответствующий int.
 */
lccrt_oper_ptr
LccrtFunctionEmitter::makeBitcastNIntToInt( int bitsize, lccrt_v_ptr a0, lccrt_v_ptr res, lccrt_oi_ptr i)
{
    struct { lccrt_var_ptr v[3]; } args;
    lccrt_oper_ptr oper = 0;
    lccrt_type_ptr tu32 = lccrt_type_make_u32( m);
    lccrt_type_ptr tu64 = lccrt_type_make_u64( m);
    lccrt_type_ptr tu128 = lccrt_type_make_u128( m);
    lccrt_type_ptr rtype = (bitsize <= 32) ? tu32 : ((bitsize <= 64) ? tu64 : tu128);

    assert( bitsize <= 128);
    if ( le.isBitWidthNormal( bitsize) || (bitsize == 1) ) {
        if ( (bitsize == 1) ) {
            assert( lccrt_type_is_bool( lccrt_var_get_type( a0)));
        } else {
            assert( (unsigned)bitsize == 8*lccrt_type_get_bytesize( lccrt_var_get_type( a0)));
            assert( lccrt_type_is_int( lccrt_var_get_type( a0)));
        }
    } else {
        lccrt_var_ptr v = makeNIntLocalVar( bitsize);
        lccrt_type_ptr tvi = lccrt_type_make_int( m, get_int_align_size( (bitsize + 7)/8), 0);
        
        args = {{v, le.makeVarConstHex( tvi, 0), le.makeVarConstHex( tu64, 1)}};
        lccrt_oper_new_elemwrite( f, 3, args.v, i);

        args = {{v, a0, le.makeVarConstHex( tu64, 0)}};
        lccrt_oper_new_elemwrite( f, 3, args.v, i);

        args = {{v, le.makeVarConstHex( tu64, 1)}};
        a0 = lccrt_oper_get_res( lccrt_oper_new_elemread( f, 2, args.v, 0, i));
    }

    oper = lccrt_oper_new_zext( f, a0, rtype, res, i);

    return (oper);
} /* LccrtFunctionEmitter::makeBitcastNIntToInt */

/**
 * Создание операций, преобразующих соответствующий int в iN.
 */
lccrt_oper_ptr
LccrtFunctionEmitter::makeBitcastIntToNInt( int bitsize, lccrt_v_ptr a0, lccrt_v_ptr res, lccrt_oi_ptr i)
{
    struct { lccrt_var_ptr v[3]; } args;
    lccrt_oper_ptr oper = 0;
    lccrt_type_ptr tu64 = lccrt_type_make_u64( m);
    lccrt_type_ptr atype = lccrt_var_get_type( a0);
    lccrt_type_ptr rtype = lccrt_var_get_type( res);

    if ( le.isBitWidthNormal( bitsize)
         || (bitsize == 1) )
    {
        if ( (bitsize == 1) && lccrt_type_is_bool( atype) )
        {
            assert( lccrt_type_is_bool( rtype));
            oper = lccrt_oper_new_move( f, a0, res, i);
        } else
        {
            assert( lccrt_type_is_int( rtype));
            assert( (unsigned)bitsize == 8*lccrt_type_get_bytesize( rtype));
            oper = lccrt_oper_new_trunc( f, a0, rtype, res, i);
        }
    } else
    {
        lccrt_var_ptr v = makeNIntLocalVar( bitsize);
        lccrt_type_ptr tvi = lccrt_type_make_int( m, get_int_align_size( (bitsize + 7)/8), 0);

        a0 = lccrt_oper_get_res( lccrt_oper_new_trunc( f, a0, tvi, 0, i));

        args = {{v, a0, le.makeVarConstHex( tu64, 1)}};
        lccrt_oper_new_elemwrite( f, 3, args.v, i);

        args = {{v, le.makeVarConstHex( tu64, 0)}};
        oper = lccrt_oper_new_elemread( f, 2, args.v, res, i);
    }

    return (oper);
} /* LccrtFunctionEmitter::makeBitcastIntToNInt */

/**
 * Спропагировать знак с заданного бита.
 */
lccrt_var_ptr
LccrtFunctionEmitter::makeExtInt( lccrt_var_ptr a, int bitwidth, bool sign,
                                  lccrt_oper_iterator_ptr i)
{
    lccrt_type_ptr tint = lccrt_var_get_type( a);
    int outbitwidth = 8*lccrt_type_get_bytesize( tint);

    if ( outbitwidth != bitwidth ) {
        lccrt_var_ptr bl = le.makeVarConstHex( tint, outbitwidth - bitwidth);

        if ( sign ) {
            a = lccrt_oper_get_res( lccrt_oper_new_shl( f, a, bl, 0, i));
            a = lccrt_oper_get_res( lccrt_oper_new_sar( f, a, bl, 0, i));
        } else {
            a = lccrt_oper_get_res( lccrt_oper_new_shl( f, a, bl, 0, i));
            a = lccrt_oper_get_res( lccrt_oper_new_shr( f, a, bl, 0, i));
        }
    }

    return (a);
} /* LccrtFunctionEmitter::makeExtInt */

/**
 * Расширение с пропагацией знака нестандартного целого до стандартного целого.
 */
lccrt_var_ptr
LccrtFunctionEmitter::makeExtNIntToInt( lccrt_var_ptr a, int bitwidth, bool sign,
                                        lccrt_oper_iterator_ptr i)
{
    a = lccrt_oper_get_res( makeBitcastNIntToInt( bitwidth, a, 0, i));
    a = makeExtInt( a, bitwidth, sign, i);

    return (a);
} /* LccrtFunctionEmitter::makeExtNIntToInt */

/**
 * Создание операции.
 */
lccrt_oper_ptr
LccrtFunctionEmitter::makeArith1( unsigned opcode, Instruction *O,
                                  lccrt_var_ptr a1, lccrt_var_ptr res, lccrt_oi_ptr i)
{
    lccrt_oper_ptr r = 0;

    switch ( opcode )
    {
      case Instruction::FNeg:  r = lccrt_oper_new_fneg( f, a1, res, i); break;
      default: errorDump( O); break;
    }

    //dbge_.makeOperDbgMetadata( r, O);
    dbgle_.makeOperDbgMetadata( r, O);

    return (r);
} /* LccrtFunctionEmitter::makeArith1 */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeArith1( unsigned opcode, User &O, lccrt_var_ptr res, lccrt_oi_ptr i)
{
    Value *V1 = O.getOperand( 0);
    Instruction *I = dyn_cast<Instruction>( &O);    
    Type *T1 = V1->getType();
    std::string func_name = "";

    switch ( opcode )
    {
      case Instruction::FNeg: func_name = "fneg"; break;
      default:
        errorDump( &O);
        break;
    }        

    if ( isa<ScalableVectorType>( T1) )
    {
        errorDump( &O);

    } else if ( isa<FixedVectorType>( T1) )
    {
        char sf[256];
        FixedVectorType *TV1 = static_cast<FixedVectorType *>( T1);
        Type *TVE1 = TV1->getElementType();
        int num_elems = TV1->getNumElements();
        int elem_size = le.DL.getTypeAllocSize( TVE1);

        if ( 0 && ((elem_size == 4)
              || (elem_size == 8))
             && ((num_elems * elem_size == 8)
                 || (num_elems * elem_size == 16))
             && isFastLibCall( &O, __FILE__, __LINE__) )
        {
            errorDump( &O);
            snprintf( sf, 256, "__lccrt_builtin_%s_v%df%d",
                      func_name.c_str(), num_elems, 8*elem_size);
            makeLibCallFast( sf, O, res, i);
        } else
        {
            func_name = "__lccrt_" + func_name + "_v";
            makeLibCall( func_name.c_str(), true, O, res, false, i);
        }
    } else
    {
        lccrt_var_ptr a1 = makeValue( V1, i);

        if ( !makeArith1( opcode, I, a1, res, i) )
        {
            errorDump( &O);
        }
    }

    return;
} /* LccrtFunctionEmitter::makeArith1 */

/**
 * Создание операции.
 */
lccrt_oper_ptr
LccrtFunctionEmitter::makeArith2( unsigned opcode, Instruction *O,
                                  lccrt_var_ptr a1, lccrt_var_ptr a2,
                                  lccrt_var_ptr res, lccrt_oi_ptr i)
{
    lccrt_var_ptr b1 = 0;
    lccrt_var_ptr b2 = 0;
    lccrt_oper_ptr r = 0;
    lccrt_type_ptr t1 = lccrt_var_get_type( a1);

    switch ( opcode )
    {
      case Instruction::Add:  r = lccrt_oper_new_add( f, a1, a2, res, i); break;
      case Instruction::Sub:  r = lccrt_oper_new_sub( f, a1, a2, res, i); break;
      case Instruction::Mul:  r = lccrt_oper_new_mul( f, a1, a2, res, i); break;
      case Instruction::UDiv: r = lccrt_oper_new_udiv( f, a1, a2, res, i); break;
      case Instruction::SDiv: r = lccrt_oper_new_sdiv( f, a1, a2, res, i); break;
      case Instruction::Shl:  r = lccrt_oper_new_shl( f, a1, a2, res, i); break;
      case Instruction::LShr: r = lccrt_oper_new_shr( f, a1, a2, res, i); break;
      case Instruction::AShr: r = lccrt_oper_new_sar( f, a1, a2, res, i); break;
      case Instruction::And:  r = lccrt_oper_new_and( f, a1, a2, res, i); break;
      case Instruction::Or:   r = lccrt_oper_new_or( f, a1, a2, res, i); break;
      case Instruction::Xor:  r = lccrt_oper_new_xor( f, a1, a2, res, i); break;
      case Instruction::FAdd: r = lccrt_oper_new_fadd( f, a1, a2, res, i); break;
      case Instruction::FSub: r = lccrt_oper_new_fsub( f, a1, a2, res, i); break;
      case Instruction::FMul: r = lccrt_oper_new_fmul( f, a1, a2, res, i); break;
      case Instruction::FDiv: r = lccrt_oper_new_fdiv( f, a1, a2, res, i); break;
      case Instruction::URem:
      case Instruction::SRem:
        b1 = makeVarRes( t1);
        b2 = makeVarRes( t1);
        if ( (opcode == Instruction::URem) ) {
            lccrt_oper_new_udiv( f, a1, a2, b1, i);
        } else {
            lccrt_oper_new_sdiv( f, a1, a2, b1, i);
        }
        lccrt_oper_new_mul( f, b1, a2, b2, i);
        r = lccrt_oper_new_sub( f, a1, b2, res, i);
        break;
    }

    if ( O ) {
        //dbge_.makeOperDbgMetadata( r, O);
        dbgle_.makeOperDbgMetadata( r, O);
    }

    return (r);
} /* LccrtFunctionEmitter::makeArith2 */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeArith2( unsigned opcode, User &O, lccrt_var_ptr res, lccrt_oi_ptr i)
{
    Value *V1 = O.getOperand( 0);
    Value *V2 = O.getOperand( 1);
    Instruction *I = dyn_cast<Instruction>( &O);    
    Type *T1 = V1->getType();
    std::string func_name = "";
    bool is_unsig = false;
    bool is_float = false;
    int bytemin = -1;

    switch ( opcode )
    {
      case Instruction::Add:  func_name = "add";  is_unsig = 1; bytemin = 1; break;
      case Instruction::Sub:  func_name = "sub";  is_unsig = 1; bytemin = 1; break;
      case Instruction::Mul:  func_name = "mul";  is_unsig = 1; bytemin = 2; break;
      case Instruction::UDiv: func_name = "udiv"; break;
      case Instruction::SDiv: func_name = "sdiv"; break;
      case Instruction::URem: func_name = "umod"; break;
      case Instruction::SRem: func_name = "smod"; break;
      case Instruction::Shl:  func_name = "shl";  bytemin = 2; break;
      case Instruction::LShr: func_name = "shr";  bytemin = 2; break;
      case Instruction::AShr: func_name = "sar";  bytemin = 2; break;
      case Instruction::And:  func_name = "and";  is_unsig = 1; bytemin = 1; break;
      case Instruction::Or:   func_name = "or";   is_unsig = 1; bytemin = 1; break;
      case Instruction::Xor:  func_name = "xor";  is_unsig = 1; bytemin = 1; break;
      case Instruction::FAdd: func_name = "fadd"; is_float = true; break;
      case Instruction::FSub: func_name = "fsub"; is_float = true; break;
      case Instruction::FMul: func_name = "fmul"; is_float = true; break;
      case Instruction::FDiv: func_name = "fdiv"; is_float = true; break;
      case Instruction::FRem: func_name = "fmod"; is_float = true; break;
      default:
        errorDump( &O);
        break;
    }        

    if ( isa<ScalableVectorType>( T1) )
    {
        errorDump( &O);

    } else if ( isa<FixedVectorType>( T1) )
    {
        char sf[256];
        FixedVectorType *TV1 = static_cast<FixedVectorType *>( T1);
        Type *TVE1 = TV1->getElementType();
        int num_elems = TV1->getNumElements();
        int elem_size = le.DL.getTypeAllocSize( TVE1);

        if ( (opcode != Instruction::FRem)
             && is_float
             && ((elem_size == 4)
                 || (elem_size == 8))
             && ((num_elems * elem_size == 8)
                 || (num_elems * elem_size == 16)
                 || (num_elems * elem_size == 32)
                 || (num_elems * elem_size == 64))
             && isFastLibCall( &O, __FILE__, __LINE__) )
        {
            snprintf( sf, 256, "__lccrt_builtin_%s_v%df%d",
                      func_name.c_str(), num_elems, 8*elem_size);
            makeLibCallFast( sf, O, res, i);

        } else if ( isa<IntegerType>( TVE1)
                    && (((opcode != Instruction::Shl)
                         && (opcode != Instruction::LShr)
                         && (opcode != Instruction::AShr))
                        || le.isVecUniform( V2))
                    && le.isIntBitWidthNormal( TVE1)
                    && (((elem_size <= 8)
                         && (bytemin > 0)
                         && (elem_size >= bytemin)
                         && ((num_elems * elem_size == 4)
                             || (num_elems * elem_size == 8)
                             || (num_elems * elem_size == 16)
                             || (num_elems * elem_size == 32)))
                        || (((opcode == Instruction::Shl)
                             || (opcode == Instruction::LShr)
                             || (opcode == Instruction::AShr)
                             || (opcode == Instruction::Add)
                             || (opcode == Instruction::Sub)
                             || (opcode == Instruction::Mul))
                            && (((elem_size == 1)
                                 && ((opcode == Instruction::Add)
                                     || (opcode == Instruction::Sub)
                                     || (opcode == Instruction::Mul)))
                                || (elem_size == 2)
                                || (elem_size == 4)
                                || (elem_size == 8))
                            && (((num_elems * elem_size == 8)
                                 && (num_elems > 1))
                                || (num_elems * elem_size == 16)
                                || (num_elems * elem_size == 32))))
                    && isFastLibCall( &O, __FILE__, __LINE__) )
        {
            snprintf( sf, 256, "__lccrt_builtin_%s_v%di%d",
                      func_name.c_str(), num_elems, 8*elem_size);
            makeLibCallFast( sf, O, res, i);
        } else
        {
            func_name = "__lccrt_" + func_name + "_v";
            makeLibCall( func_name.c_str(), true, O, res, false, i);
            if ( (opcode == Instruction::FMul) ) {
                errorDump( &O);
            }
        }
    } else if ( !le.isIntBitWidthNormalOrBool( T1)
                || (opcode == Instruction::FRem) )
    {
        if ( isa<IntegerType>( T1)
             && (le.DL.getTypeAllocSize( T1) <= 8)
             && (is_unsig) )
        {
            lccrt_var_ptr b0, b1;
            lccrt_var_ptr a1 = makeValue( V1, i);
            lccrt_var_ptr a2 = makeValue( V2, i);
            uint64_t bitsize = le.DL.getTypeSizeInBits( T1);
            lccrt_type_ptr tu32 = lccrt_type_make_u32( m);
            lccrt_type_ptr tu64 = lccrt_type_make_u64( m);
            lccrt_type_ptr tu = (bitsize <= 32) ? tu32 : tu64;

            a1 = lccrt_oper_get_res( makeBitcastNIntToInt( bitsize, a1, 0, i));
            a2 = lccrt_oper_get_res( makeBitcastNIntToInt( bitsize, a2, 0, i));

            // b0 = (a1 op a2) & mask
            b1 = le.makeVarConstHex( tu, (~0ULL) >> (64ULL - bitsize));
            b0 = lccrt_oper_get_res( makeArith2( opcode, I, a1, a2, 0, i));
            b0 = lccrt_oper_get_res( lccrt_oper_new_and( f, b0, b1, 0, i));

            makeBitcastIntToNInt( bitsize, b0, res, i);
        } else
        {
            func_name = "__lccrt_" + func_name + "_n";
            makeLibCall( func_name.c_str(), false, O, res, false, i);
        }
    } else
    {
        lccrt_var_ptr a1 = makeValue( V1, i);
        lccrt_var_ptr a2 = makeValue( V2, i);

        if ( !makeArith2( opcode, I, a1, a2, res, i) ) {
            errorDump( &O);
        }
    }

    return;
} /* LccrtFunctionEmitter::makeArith2 */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeBswap( User &O, lccrt_var_ptr res, lccrt_oi_ptr i)
{
    char sf[256];
    Value *V1 = O.getOperand( 0);
    lccrt_var_ptr a1 = makeValue( V1, i);
    Type *TR = O.getType();
    Type *T1 = V1->getType();
    const CallInst &CI = cast<CallInst>(O);
    std::string cn = CI.getCalledOperand()->getName().str();

    if ( isa<ScalableVectorType>( T1) ) {
        errorDump( &O);
    } else if ( isa<FixedVectorType>( T1) ) {
        FixedVectorType *TV1 = static_cast<FixedVectorType *>( T1);
        int num_elems = TV1->getNumElements();
        int bitsize = TV1->getScalarSizeInBits();

        if ( !isa<IntegerType>( TV1->getScalarType()) ) {
            errorDump( &O);
            
        } else if ( le.isVectorBuiltinInt( TV1, 16)
                    && isFastLibCallLine( O) )
        {
            snprintf( sf, 256, "__lccrt_builtin_bswap_v%di%d", num_elems, bitsize);
            makeLibCallFast( sf, a1, 0, 0, res, i);
        } else {
            makeLibCall( "__lccrt_bswap_v", true, O, res, false, i);
        }
    } else if ( isa<IntegerType>( TR) ) {
        int bitsize = TR->getScalarSizeInBits();

        if ( (bitsize > 128) || (bitsize % 16 != 0) ) {
            errorDump( &O);
        } else {
            snprintf( sf, 256, "__lccrt_builtin_bswap_i%d", bitsize);
            makeLibCallFast( sf, a1, 0, 0, res, i);
        }
    } else {
        errorDump( &O);
    }

    return;
} /* LccrtFunctionEmitter::makeBswap */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeCtpop( User &O, lccrt_var_ptr res, lccrt_oi_ptr i)
{
    char sf[256];
    Value *V1 = O.getOperand( 0);
    lccrt_var_ptr a1 = makeValue( V1, i);
    Type *T1 = V1->getType();

    if ( isa<ScalableVectorType>( T1) )
    {
        errorDump( &O);

    } else if ( isa<FixedVectorType>( T1) )
    {
        FixedVectorType *TV1 = static_cast<FixedVectorType *>( T1);
        Type *TVE1 = TV1->getElementType();
        int num_elems = TV1->getNumElements();
        int elem_size = le.DL.getTypeAllocSize( TVE1);

        if ( ((elem_size == 4)
              || (elem_size == 8))
             && ((num_elems * elem_size == 4)
                 || (num_elems * elem_size == 8)
                 || (num_elems * elem_size == 16))
             && isFastLibCall( &O, __FILE__, __LINE__) )
        {
            snprintf( sf, 256, "__lccrt_builtin_ctpop_v%di%d",
                      num_elems, 8*elem_size);
            makeLibCallFast( sf, a1, 0, 0, res, i);
        } else
        {
            makeLibCall( "__lccrt_ctpop_v", true, O, res, false, i);
        }
    } else if ( le.isIntBitWidthNormal( T1) )
    {
        int bitsize = le.DL.getTypeSizeInBits( T1);

        snprintf( sf, 256, "__lccrt%s_ctpop_i%d",
                  (bitsize < 128) ? "_builtin" : "", bitsize);
        makeLibCallFast( sf, a1, 0, 0, res, i);
    } else
    {
        makeLibCall( "__lccrt_ctpop_n", false, O, res, false, i);
    }

    return;
} /* LccrtFunctionEmitter::makeCtpop */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeCttz( User &O, lccrt_var_ptr res, lccrt_oi_ptr i)
{
    char sf[256];
    Value *V1 = O.getOperand( 0);
    lccrt_var_ptr a1 = makeValue( V1, i);
    Type *T1 = V1->getType();

    if ( isa<ScalableVectorType>( T1) ) {
        errorDump( &O);

    } else if ( isa<FixedVectorType>( T1) ) {
        FixedVectorType *TV1 = static_cast<FixedVectorType *>( T1);
        Type *TVE1 = TV1->getElementType();
        int num_elems = TV1->getNumElements();
        int elem_size = le.DL.getTypeAllocSize( TVE1);

        if ( ((elem_size == 4)
              || (elem_size == 8))
             && ((num_elems * elem_size == 4)
                 || (num_elems * elem_size == 8)
                 || (num_elems * elem_size == 16))
             && isFastLibCall( &O, __FILE__, __LINE__) )
        {
            snprintf( sf, 256, "__lccrt_builtin_cttz_v%di%d",
                      num_elems, 8*elem_size);
            makeLibCallFast( sf, a1, 0, 0, res, i);
        } else {
            makeLibCall( "__lccrt_cttz_v", true, O, res, true, i);
        }
    } else if ( le.isIntBitWidthNormal( T1) ) {
        int bitsize = le.DL.getTypeSizeInBits( T1);

        snprintf( sf, 256, "__lccrt%s_cttz_i%d",
                  (bitsize < 128) ? "_builtin" : "", bitsize);
        makeLibCallFast( sf, a1, 0, 0, res, i);
    } else {
        makeLibCall( "__lccrt_cttz_n", false, O, res, true, i);
    }

    return;
} /* LccrtFunctionEmitter::makeCttz */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeCtlz( User &O, lccrt_var_ptr res, lccrt_oi_ptr i)
{
    char sf[256];
    Value *V1 = O.getOperand( 0);
    lccrt_var_ptr a1 = makeValue( V1, i);
    Type *T1 = V1->getType();

    if ( isa<ScalableVectorType>( T1) ) {
        errorDump( &O);

    } else if ( isa<FixedVectorType>( T1) ) {
        FixedVectorType *TV1 = static_cast<FixedVectorType *>( T1);
        Type *TVE1 = TV1->getElementType();
        int num_elems = TV1->getNumElements();
        int elem_size = le.DL.getTypeAllocSize( TVE1);

        if ( ((elem_size == 4)
              || (elem_size == 8))
             && ((num_elems * elem_size == 4)
                 || (num_elems * elem_size == 8)
                 || (num_elems * elem_size == 16))
             && isFastLibCall( &O, __FILE__, __LINE__) )
        {
            snprintf( sf, 256, "__lccrt_builtin_ctlz_v%di%d",
                      num_elems, 8*elem_size);
            makeLibCallFast( sf, a1, 0, 0, res, i);
        } else {
            makeLibCall( "__lccrt_ctlz_v", true, O, res, true, i);
        }
    } else if ( le.isIntBitWidthNormal( T1) ) {
        int bitsize = le.DL.getTypeSizeInBits( T1);

        snprintf( sf, 256, "__lccrt%s_ctlz_i%d",
                  (bitsize < 128) ? "_builtin" : "", bitsize);
        makeLibCallFast( sf, a1, 0, 0, res, i);
    } else {
        makeLibCall( "__lccrt_ctlz_n", false, O, res, true, i);
    }

    return;
} /* LccrtFunctionEmitter::makeCtlz */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeFptoiSat( User &O, lccrt_var_ptr res, lccrt_oi_ptr i)
{
    Value *V1 = O.getOperand( 0);
    lccrt_var_ptr a1 = makeValue( V1, i);
    Type *TR = O.getType();
    Type *T1 = V1->getType();
    const CallInst &CI = cast<CallInst>(O);
    std::string cn = CI.getCalledOperand()->getName().str();
    const char *func_name = (cn.find( "llvm.fptosi.sat.") == 0) ? "fptosi_sat" : "fptoui_sat";

    if ( isa<ScalableVectorType>( T1) )
    {
        errorDump( &O);

    } else if ( isa<FixedVectorType>( T1) )
    {
        char sf[256];

        snprintf( sf, 256, "__lccrt_%s_v", func_name);
        makeLibCall( sf, true, O, res, false, i);
    } else
    {
        if ( isa<IntegerType>( TR)
             && le.isTypeFloatNormal( T1)
             && le.isIntBitWidthNormal( TR) )
        {
            char sf[256];

            snprintf( sf, 256, "__lccrt_%s_i%df%d",
                      func_name, le.getTypeBitsize( TR), le.getTypeBitsize( T1));
            makeLibCallFast( sf, a1, 0, 0, res, i);
        } else
        {
            errorDump( &O);
        }
    }

    return;
} /* LccrtFunctionEmitter::makeFptoiSat */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeFrexp( User &O, lccrt_var_ptr res, lccrt_oi_ptr i)
{
    Value *V1 = O.getOperand( 0);
    lccrt_type_ptr tu32 = lccrt_type_make_u32( m);
    lccrt_var_ptr a1 = makeValue( V1, i);
    lccrt_var_ptr a2 = 0;
    lccrt_var_ptr b2 = lccrt_var_new_local( f, tu32, 0);
    lccrt_var_ptr a0 = lccrt_var_new_local( f, lccrt_var_get_type( a1), 0);
    struct { lccrt_var_ptr v[3]; } args = {};
    lccrt_var_ptr cv0 = le.makeVarConstHex( tu32, 0);
    lccrt_var_ptr cv1 = le.makeVarConstHex( tu32, 1);
    const CallInst &CI = cast<CallInst>(O);
    std::string cn = CI.getCalledOperand()->getName().str();
    const char *oname = 0;

    if ( cn == "llvm.frexp.f32.i32" ) {
        oname = "frexpf";
    } else if ( cn == "llvm.frexp.f64.i32" ) {
        oname = "frexp";
    } else if ( cn == "llvm.frexp.f80.i32" ) {
        oname = "frexpl";
    } else {
        errorDump( &O);
    }

    a2 = lccrt_oper_get_res( lccrt_oper_new_varptr( f, b2, 0, i));
    makeLibCallFast( oname, a1, a2, 0, a0, i);
    args = {{res, a0, cv0}};
    lccrt_oper_new_elemwrite( f, 3, args.v, i);
    args = {{res, b2, cv1}};
    lccrt_oper_new_elemwrite( f, 3, args.v, i);

    return;
} /* LccrtFunctionEmitter::makeFrexp */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeFshl( User &O, lccrt_var_ptr res, lccrt_oi_ptr i)
{
    Value *V1 = O.getOperand( 0);
    Value *V2 = O.getOperand( 1);
    Value *V3 = O.getOperand( 2);
    lccrt_var_ptr args[3] = {makeValue( V1, i), makeValue( V2, i), makeValue( V3, i)};
    Type *TR = O.getType();
    Type *T1 = V1->getType();
    const CallInst &CI = cast<CallInst>(O);
    std::string cn = CI.getCalledOperand()->getName().str();

    if ( isa<ScalableVectorType>( T1) ) {
        errorDump( &O);
    } else if ( isa<FixedVectorType>( T1) ) {
        makeLibCall( "__lccrt_fshl_v", true, O, res, false, i);
    } else {
        int bitsize = le.getTypeBitsize( TR);
        bool is_intpow2 = isa<IntegerType>( TR) && is_pow2( bitsize);

        if ( is_intpow2 && (8 <= bitsize) && (bitsize <= 64) ) {
            lccrt_var_ptr ya[6];
            lccrt_type_ptr t = lccrt_var_get_type( args[0]);
            lccrt_type_ptr tu32 = lccrt_type_make_u32( m);
            lccrt_type_ptr tb = lccrt_type_make_bool( m);
            lccrt_var_ptr p0 = lccrt_var_new_local( f, tb, 0);
            lccrt_var_ptr c0 = le.makeVarConstHex( t, 0);
            lccrt_var_ptr c1 = le.makeVarConstHex( t, bitsize - 1);
            lccrt_var_ptr c2 = le.makeVarConstHex( t, bitsize);
            lccrt_varinit_ptr an = lccrt_varinit_new_scalar( tu32, LCCRT_CMP_EQ);

            for ( int i = 0; i < 6; ++i ) ya[i] = lccrt_var_new_local( f, t, 0);

            lccrt_oper_new_and( f, args[2], c1, ya[0], i);
            lccrt_oper_new_sub( f, c2, ya[0], ya[1], i);
            lccrt_oper_new_and( f, ya[1], c1, ya[2], i);

            lccrt_oper_new_shl( f, args[0], ya[0], ya[3], i);
            lccrt_oper_new_shr( f, args[1], ya[2], ya[4], i);
            lccrt_oper_new_or( f, ya[3], ya[4], ya[5], i);

            lccrt_oper_new_cmp( f, an, ya[0], c0, p0, i);
            lccrt_oper_new_select( f, p0, args[0], ya[5], res, i);

        } else if ( is_intpow2 && ((bitsize == 4) || (bitsize == 128)) ) {
            char sf[256];

            //snprintf( sf, 256, "__lccrt_builtin_fshl_i%d", bitsize);
            snprintf( sf, 256, "__lccrt_fshl_i%d", bitsize);
            makeLibCallFast( sf, args[0], args[1], args[2], res, i);
        } else {
            makeLibCall( "__lccrt_fshl_n", false, O, res, false, i);
        }
    }

    return;
} /* LccrtFunctionEmitter::makeFshl */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeFshr( User &O, lccrt_var_ptr res, lccrt_oi_ptr i)
{
    Value *V1 = O.getOperand( 0);
    Value *V2 = O.getOperand( 1);
    Value *V3 = O.getOperand( 2);
    lccrt_var_ptr args[3] = {makeValue( V1, i), makeValue( V2, i), makeValue( V3, i)};
    Type *TR = O.getType();
    Type *T1 = V1->getType();
    const CallInst &CI = cast<CallInst>(O);
    std::string cn = CI.getCalledOperand()->getName().str();

    if ( isa<ScalableVectorType>( T1) ) {
        errorDump( &O);
    } else if ( isa<FixedVectorType>( T1) ) {
        makeLibCall( "__lccrt_fshr_v", true, O, res, false, i);
    } else {
        int bitsize = le.getTypeBitsize( TR);
        bool is_intpow2 = isa<IntegerType>( TR) && is_pow2( bitsize);

        if ( is_intpow2 && (8 <= bitsize) && (bitsize <= 64) ) {
            lccrt_var_ptr ya[6];
            lccrt_type_ptr t = lccrt_var_get_type( args[0]);
            lccrt_type_ptr tu32 = lccrt_type_make_u32( m);
            lccrt_type_ptr tb = lccrt_type_make_bool( m);
            lccrt_var_ptr p0 = lccrt_var_new_local( f, tb, 0);
            lccrt_var_ptr c0 = le.makeVarConstHex( t, 0);
            lccrt_var_ptr c1 = le.makeVarConstHex( t, bitsize - 1);
            lccrt_var_ptr c2 = le.makeVarConstHex( t, bitsize);
            lccrt_varinit_ptr an = lccrt_varinit_new_scalar( tu32, LCCRT_CMP_EQ);

            for ( int i = 0; i < 6; ++i ) ya[i] = lccrt_var_new_local( f, t, 0);

            lccrt_oper_new_and( f, args[2], c1, ya[0], i);
            lccrt_oper_new_sub( f, c2, ya[0], ya[1], i);
            lccrt_oper_new_and( f, ya[1], c1, ya[2], i);

            lccrt_oper_new_shr( f, args[1], ya[0], ya[3], i);
            lccrt_oper_new_shl( f, args[0], ya[2], ya[4], i);
            lccrt_oper_new_or( f, ya[3], ya[4], ya[5], i);

            lccrt_oper_new_cmp( f, an, ya[0], c0, p0, i);
            lccrt_oper_new_select( f, p0, args[1], ya[5], res, i);

        } else if ( is_intpow2 && ((bitsize == 4) || (bitsize == 128)) ) {
            char sf[256];

            //snprintf( sf, 256, "__lccrt_builtin_fshr_i%d", bitsize);
            snprintf( sf, 256, "__lccrt_fshr_i%d", bitsize);
            makeLibCallFast( sf, args[0], args[1], args[2], res, i);
        } else {
            makeLibCall( "__lccrt_fshr_n", false, O, res, false, i);
        }
    }

    return;
} /* LccrtFunctionEmitter::makeFshr */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeFmuladd( User &O, lccrt_var_ptr res, lccrt_oi_ptr i)
{
    Value *V1 = O.getOperand( 0);
    Value *V2 = O.getOperand( 1);
    Value *V3 = O.getOperand( 2);
    lccrt_var_ptr a1 = makeValue( V1, i);
    lccrt_var_ptr a2 = makeValue( V2, i);
    lccrt_var_ptr a3 = makeValue( V3, i);
    lccrt_var_ptr v0 = lccrt_var_new_local( f, lccrt_var_get_type( res), 0);
    Type *T1 = V1->getType();
    std::string func_name = "fmuladd";

    if ( isa<ScalableVectorType>( T1) )
    {
        errorDump( &O);

    } else if ( isa<FixedVectorType>( T1) )
    {
        char sf[256];
        FixedVectorType *TV1 = static_cast<FixedVectorType *>( T1);
        Type *TVE1 = TV1->getElementType();
        int num_elems = TV1->getNumElements();
        int elem_size = le.DL.getTypeAllocSize( TVE1);

        if ( ((elem_size == 4)
              || (elem_size == 8))
             && ((num_elems * elem_size == 8)
                 || (num_elems * elem_size == 16)
                 || (num_elems * elem_size == 32))
             && isFastLibCall( &O, __FILE__, __LINE__) )
        {
            snprintf( sf, 256, "__lccrt_builtin_fmul_v%df%d",
                      num_elems, 8*elem_size);
            makeLibCallFast( sf, a1, a2, 0, v0, i);

            snprintf( sf, 256, "__lccrt_builtin_fadd_v%df%d",
                      num_elems, 8*elem_size);
            makeLibCallFast( sf, v0, a3, 0, res, i);
        } else
        {
            func_name = "__lccrt_" + func_name + "_v";
            makeLibCall( func_name.c_str(), true, O, res, false, i);
        }
    } else
    {
        lccrt_oper_new_fmul( f, a1, a2, v0, i);
        lccrt_oper_new_fadd( f, v0, a3, res, i);
    }

    return;
} /* LccrtFunctionEmitter::makeFmuladd */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeSqrt( User &O, lccrt_var_ptr res, lccrt_oi_ptr i)
{
    Value *V1 = O.getOperand( 0);
    lccrt_var_ptr a1 = makeValue( V1, i);
    Type *T1 = V1->getType();
    std::string func_name = "sqrt";
    CallInst &CI = cast<CallInst>(O);
    Value *CV = CI.getCalledOperand();
    std::string cn = CV->hasName() ? CV->getName().str() : "";

    if ( isa<ScalableVectorType>( T1) )
    {
        errorDump( &O);

    } else if ( isa<FixedVectorType>( T1) )
    {
        char sf[256];
        FixedVectorType *TV1 = static_cast<FixedVectorType *>( T1);
        Type *TVE1 = TV1->getElementType();
        int num_elems = TV1->getNumElements();
        int elem_size = le.DL.getTypeAllocSize( TVE1);

        if ( ((elem_size == 4)
              || (elem_size == 8))
             && ((num_elems * elem_size == 8)
                 || (num_elems * elem_size == 16))
             && isFastLibCall( &O, __FILE__, __LINE__) )
        {
            snprintf( sf, 256, "__lccrt_builtin_fastsqrt_v%df%d",
                      num_elems, 8*elem_size);
            makeLibCallFast( sf, a1, 0, 0, res, i);
        } else
        {
            func_name = "__lccrt_" + func_name + "_v";
            makeLibCall( func_name.c_str(), true, O, res, false, i);
        }
    } else if ( (cn == "llvm.sqrt.f32") )
    {
        makeLibCallFast( "__builtin_sqrtf", a1, 0, 0, res, i);

    } else if ( (cn == "llvm.sqrt.f64") )
    {
        makeLibCallFast( "__builtin_sqrt", a1, 0, 0, res, i);

    } else if ( (cn == "llvm.sqrt.f80") )
    {
        makeLibCallFast( "__builtin_sqrtl", a1, 0, 0, res, i);
    } else
    {
        errorDump( &O);
    }

    return;
} /* LccrtFunctionEmitter::makeSqrt */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeIntAbs( User &O, lccrt_var_ptr res, lccrt_oi_ptr i)
{
    Value *V1 = O.getOperand( 0);
    lccrt_var_ptr a1 = makeValue( V1, i);
    Type *T1 = V1->getType();
    std::string func_name = "abs";
    lccrt_type_ptr tu32 = lccrt_type_make_u32( m);
    lccrt_type_ptr tu64 = lccrt_type_make_u64( m);
    int bitwidth = T1->getPrimitiveSizeInBits();
    bool normal_int = le.isIntBitWidthNormal( T1);
    bool isint = isa<IntegerType>( T1);

    if ( isa<ScalableVectorType>( T1) ) {
        errorDump( &O);
    } else if ( isa<FixedVectorType>( T1) ) {
        char sf[256];
        FixedVectorType *TV1 = static_cast<FixedVectorType *>( T1);
        Type *TVE1 = TV1->getElementType();
        int num_elems = TV1->getNumElements();
        int elem_bitsize = le.DL.getTypeSizeInBits( TVE1);

        if ( !isa<IntegerType>( TVE1) ) {
            errorDump( &O);
        } else if ( ((elem_bitsize == 32)
                     || (elem_bitsize == 64))
                    && ((num_elems * elem_bitsize == 8*8)
                        || (num_elems * elem_bitsize == 16*8))
                    && isFastLibCall( &O, __FILE__, __LINE__) )
        {
            const char *sns;
            lccrt_varinit_ptr zi = 0;
            lccrt_var_ptr z1 = 0;
            lccrt_var_ptr b1 = 0;
            lccrt_var_ptr p1 = 0;
            std::vector<lccrt_varinit_ptr> zarr( num_elems);
            lccrt_type_ptr te = (elem_bitsize == 32) ? tu32 : tu64;
            lccrt_type_ptr tr = le.makeType( TV1);

            for ( int l = 0; l < num_elems; ++l ) {
                zarr[l] = lccrt_varinit_new_zero( te);
            }

            zi = lccrt_varinit_new_array( tr, zarr.size(), zarr.data());
            z1 = le.makeVarConst( tr, zi);
            b1 = allocLocalTmpVar( tr);
            p1 = allocLocalTmpVar( tr);
            getCmpLccrtName( ICmpInst::ICMP_SGE, &sns);

            snprintf( sf, 256, "__lccrt_builtin_sub_v%di%d",
                      num_elems, elem_bitsize);
            makeLibCallFast( sf, z1, a1, 0, b1, i);

            snprintf( sf, 256, "__lccrt_builtin_cmp%s_v%di%d",
                      sns, num_elems, elem_bitsize);
            makeLibCallFast( sf, a1, b1, 0, p1, i);

            snprintf( sf, 256, "__lccrt_builtin_select_v%di%d",
                      num_elems, elem_bitsize);
            makeLibCallFast( sf, p1, a1, b1, res, i);

            freeLocalTmpVar( b1);
            freeLocalTmpVar( p1);
        } else {
            func_name = "__lccrt_" + func_name + "_v";
            makeLibCall( func_name.c_str(), true, O, res, true, i);
        }
    } else if ( isint && (normal_int || (bitwidth < 64)) ) {
        lccrt_varinit_ptr an;
        lccrt_var_ptr v0 = lccrt_var_new_local( f, lccrt_type_make_bool( m), 0);
        lccrt_var_ptr v1 = 0;
        lccrt_var_ptr v2 = 0;
        lccrt_var_ptr vr = res;

        if ( !normal_int ) {
            a1 = makeExtNIntToInt( a1, bitwidth, true, i);
            vr = lccrt_var_new_local( f, lccrt_var_get_type( a1), 0);
        }

        an = lccrt_varinit_new_scalar( lccrt_type_make_u32( m), LCCRT_CMP_GE_I);
        v2 = le.makeVarConstHex( lccrt_var_get_type( a1), 0);
        v1 = lccrt_oper_get_res( lccrt_oper_new_sub( f, v2, a1, 0, i));
        lccrt_oper_new_cmp( f, an, a1, v1, v0, i);
        lccrt_oper_new_select( f, v0, a1, v1, vr, i);

        if ( !normal_int ) {
            vr = makeExtInt( vr, bitwidth, false, i);
            makeBitcastIntToNInt( bitwidth, vr, res, i);
        }
    } else {
        errorDump( &O);
    }

    return;
} /* LccrtFunctionEmitter::makeIntAbs */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeIntMinMax( std::string mname, User &O,
                                     lccrt_var_ptr res, lccrt_oi_ptr i)
{
    Value *V1 = O.getOperand( 0);
    Value *V2 = O.getOperand( 1);
    lccrt_var_ptr a1 = makeValue( V1, i);
    lccrt_var_ptr a2 = makeValue( V2, i);
    Type *T1 = V1->getType();
    std::string func_name = mname;
    int bitwidth = T1->getPrimitiveSizeInBits();
    bool normal_int = le.isIntBitWidthNormal( T1);
    bool isint = isa<IntegerType>( T1);

    if ( isa<ScalableVectorType>( T1) ) {
        errorDump( &O);
    } else if ( isa<FixedVectorType>( T1) ) {
        char sf[256];
        FixedVectorType *TV1 = static_cast<FixedVectorType *>( T1);
        Type *TVE1 = TV1->getElementType();
        int num_elems = TV1->getNumElements();
        int elem_size = le.DL.getTypeAllocSize( TVE1);

        if ( !isa<IntegerType>( TVE1) ) {
            errorDump( &O);
        } else if ( ((elem_size == 1) || (elem_size == 2) || (elem_size == 4) || (elem_size == 8))
                    && ((num_elems * elem_size == 4)
                        || (num_elems * elem_size == 8)
                        || (num_elems * elem_size == 16)
                        || (num_elems * elem_size == 32)
                        || (num_elems * elem_size == 64))
                    && isFastLibCall( &O, __FILE__, __LINE__) )
        {
            snprintf( sf, 256, "__lccrt_builtin_%s_v%di%d",
                      mname.c_str(), num_elems, 8*elem_size);
            makeLibCallFast( sf, a1, a2, 0, res, i);
        } else
        {
            func_name = "__lccrt_" + func_name + "_v";
            makeLibCall( func_name.c_str(), true, O, res, false, i);
        }
    } else if ( isint && (normal_int || (bitwidth < 64)) ) {
        lccrt_varinit_ptr an;
        lccrt_cmp_name_t ncmp = LCCRT_CMP_LAST;
        lccrt_var_ptr v0 = lccrt_var_new_local( f, lccrt_type_make_bool( m), 0);
        lccrt_var_ptr vr = res;

        if ( !normal_int ) {
            bool sign = (mname == "smax") || (mname == "smin");

            a1 = makeExtNIntToInt( a1, bitwidth, sign, i);
            a2 = makeExtNIntToInt( a2, bitwidth, sign, i);
            vr = lccrt_var_new_local( f, lccrt_var_get_type( a1), 0);
        }

        if ( (mname == "smax") )      ncmp = LCCRT_CMP_GE_I;
        else if ( (mname == "smin") ) ncmp = LCCRT_CMP_LE_I;
        else if ( (mname == "umax") ) ncmp = LCCRT_CMP_GE_U;
        else if ( (mname == "umin") ) ncmp = LCCRT_CMP_LE_U;
        else errorDump( &O);

        an = lccrt_varinit_new_scalar( lccrt_type_make_u32( m), ncmp);
        lccrt_oper_new_cmp( f, an, a1, a2, v0, i);
        lccrt_oper_new_select( f, v0, a1, a2, vr, i);

        if ( !normal_int ) {
            vr = makeExtInt( vr, bitwidth, false, i);
            makeBitcastIntToNInt( bitwidth, vr, res, i);
        }
    } else if ( le.isIntBitWidthBool( T1) ) {
        if ( (mname == "umax") ) {
            lccrt_oper_new_or( f, a1, a2, res, i);
        } else if ( (mname == "umin") ) {
            lccrt_oper_new_and( f, a1, a2, res, i);
        } else {
            errorDump( &O);
        }
    } else {
        errorDump( &O);
    }

    return;
} /* LccrtFunctionEmitter::makeIntMinMax */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeBitrev( User &O, lccrt_var_ptr res, lccrt_oi_ptr i)
{
    char sf[256];
    Value *V1 = O.getOperand( 0);
    lccrt_var_ptr a1 = makeValue( V1, i);
    Type *T1 = V1->getType();
    std::string func_name = "bitreverse";

    if ( isa<ScalableVectorType>( T1) )
    {
        errorDump( &O);

    } else if ( isa<FixedVectorType>( T1) )
    {
        FixedVectorType *TV1 = static_cast<FixedVectorType *>( T1);
        Type *TVE1 = TV1->getElementType();
        int num_elems = TV1->getNumElements();
        int elem_size = le.DL.getTypeAllocSize( TVE1);

        if ( !isa<IntegerType>( TVE1) )
        {
            errorDump( &O);

        } else if ( ((elem_size == 4)
                     || (elem_size == 8))
                    && ((num_elems * elem_size == 8)
                        || (num_elems * elem_size == 16))
                    && isFastLibCall( &O, __FILE__, __LINE__) )
        {
            snprintf( sf, 256, "__lccrt_builtin_%s_v%di%d",
                      func_name.c_str(), num_elems, 8*elem_size);
            makeLibCallFast( sf, a1, 0, 0, res, i);
        } else
        {
            func_name = "__lccrt_" + func_name + "_v";
            makeLibCall( func_name.c_str(), true, O, res, false, i);
        }
    } else if ( (T1->getPrimitiveSizeInBits() <= 128) )
    {
        int bits1 = T1->getPrimitiveSizeInBits();

        snprintf( sf, 256, "__lccrt_builtin_%s_i%d",
                  func_name.c_str(), bits1);
        a1 = lccrt_oper_get_res( makeBitcastNIntToInt( bits1, a1, 0, i));
        makeLibCallFast( sf, a1, 0, 0, res, i);
    } else
    {
        errorDump( &O);
    }

    return;
} /* LccrtFunctionEmitter::makeBitrev */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeVectorReduce( const User &O, lccrt_var_ptr res, lccrt_oi_ptr i)
{
    //char sf[256];
    const Value *V1 = O.getOperand( 0);
    //lccrt_var_ptr a1 = makeValue( V1, i);
    Type *T1 = V1->getType();
    const CallInst &CI = cast<CallInst>(O);
    const Value *CV = CI.getCalledOperand();
    std::string cn = CV->hasName() ? CV->getName().str() : "";
    std::string func_name;

    if ( cn.find( "llvm.vector.reduce.fmin.") == 0 )      func_name = "reduce_fmin";
    else if ( cn.find( "llvm.vector.reduce.fmax.") == 0 ) func_name = "reduce_fmax";
    else errorDump( &O);

    if ( isa<ScalableVectorType>( T1) )
    {
        errorDump( &O);

    } else if ( isa<FixedVectorType>( T1) )
    {
#if 0
        FixedVectorType *TV1 = static_cast<FixedVectorType *>( T1);
        Type *TVE1 = TV1->getElementType();
        int num_elems = TV1->getNumElements();
        int elem_size = le.DL.getTypeAllocSize( TVE1);

        if ( !isa<IntegerType>( TVE1) )
        {
            errorDump( &O);

        } else if ( ((elem_size == 4)
                     || (elem_size == 8))
                    && ((num_elems * elem_size == 8)
                        || (num_elems * elem_size == 16))
                    && isFastLibCall( &O, __FILE__, __LINE__) )
        {
            snprintf( sf, 256, "__lccrt_builtin_%s_v%di%d",
                      func_name.c_str(), num_elems, 8*elem_size);
            makeLibCallFast( sf, a1, 0, 0, res, i);
        } else
#endif
        {
            func_name = "__lccrt_" + func_name + "_v";
            makeLibCall( func_name.c_str(), true, O, res, false, i);
        }
    } else
    {
        errorDump( &O);
    }

    return;
} /* LccrtFunctionEmitter::makeVectorReduce */

/**
 * Получить прагма-атрибут цикла, привязанный к операции перехода (по обратной дуге).
 */
MDNode *
getBranchLoopPragma( const BranchInst &BI, StringRef Name) {
    MDNode *r = 0;
    MDNode *MD = BI.getMetadata( LLVMContext::MD_loop);

    if ( MD ) {
        // First operand should refer to the loop id itself.
        assert( MD->getNumOperands() > 0 && "requires at least one operand");
        assert( MD->getOperand(0) == MD && "invalid loop id");

        for ( unsigned i = 1, e = MD->getNumOperands(); i < e; ++i ) {
            MDNode *cur_md = dyn_cast<MDNode>( MD->getOperand(i));

            if ( cur_md ) {
                MDString *S = dyn_cast<MDString>( cur_md->getOperand(0));

                if ( S ) {
                    if ( Name == S->getString() ) {
                        r = cur_md;
                        break;
                    }
                }
            }
        }
    }

    return (r);
} /* getBranchLoopPragma */

/**
 * Транслировать профильную информацию для операции условного перехода.
 */
void
LccrtFunctionEmitter::evalBranchProfile( const BranchInst &BI, lccrt_oper_ptr ct) {
    if ( MDNode *prof = BI.getMetadata( LLVMContext::MD_prof) ) {
        if ( prof->getNumOperands() == 3 ) {
            auto *pname = dyn_cast<MDString>( prof->getOperand( 0));
            auto *ptm = dyn_cast<ValueAsMetadata>( prof->getOperand( 1));
            auto *pfm = dyn_cast<ValueAsMetadata>( prof->getOperand( 2));
            const Value *ptv = ptm ? ptm->getValue() : 0;
            const Value *pfv = pfm ? pfm->getValue() : 0;
            const ConstantInt *pt = ptv ? dyn_cast<ConstantInt>( ptv) : 0;
            const ConstantInt *pf = pfv ? dyn_cast<ConstantInt>( pfv) : 0;

            if ( pname && pt && pf && (pname->getString() == "branch_weights") ) {
                lccrt_einfo_reference_t eprof;
                uint64_t pti = pt->getZExtValue();
                uint64_t pfi = pf->getZExtValue();
               
                eprof = lccrt_einfo_new_struct( le.etyde_profct);
                lccrt_einfo_set_field( eprof, le.eifi_profct_et, lccrt_einfo_new_i64( pti));
                lccrt_einfo_set_field( eprof, le.eifi_profct_ef, lccrt_einfo_new_i64( pfi));
                lccrt_oper_set_einfo( ct, le.ecat_prof, eprof);
            }
        }
    }

    return;
} /* LccrtFunctionEmitter::evalBranchProfile */

/**
 * Создание операции.
 */
int
LccrtFunctionEmitter::makeBranch( Instruction &O, lccrt_oper_ptr &ct, lccrt_v_ptr res,
                                  arg_ref_t *alts_to_opers, lccrt_oi_ptr i)
{
    int num_alts = 0;

    ct = 0;
    if ( isa<BranchInst>(O) )
    {
        lccrt_var_ptr v1 = 0;
        lccrt_oper_ptr o1 = 0;
        lccrt_oper_ptr o2 = 0;
        BranchInst &BI = cast<BranchInst>(O);
        MDNode *MD = getBranchLoopPragma( BI, "llvm.loop.loop.count");

        if ( cast<BranchInst>(O).isConditional() )
        {
            num_alts = 2;
            v1 = makeValue( BI.getCondition(), i);
            o1 = lbls.find( BI.getSuccessor( 0))->second;
            o2 = lbls.find( BI.getSuccessor( 1))->second;
            ct = lccrt_oper_new_branchif( f, v1, o2, o1, i);
            alts_to_opers[0] = ARG_REF( ct, 2);
            alts_to_opers[1] = ARG_REF( ct, 1);
            evalBranchProfile( BI, ct);
        } else
        {
            num_alts = 1;
            o1 = lbls.find( BI.getSuccessor( 0))->second;
            ct = lccrt_oper_new_branch( f, o1, i);
            alts_to_opers[0] = ARG_REF( ct, 0);
        }

        //dbge_.makeOperDbgMetadata( ct, &O);
        dbgle_.makeOperDbgMetadata( ct, &O);
        if ( MD )
        {
            assert( MD->getNumOperands() == 2 && "Unroll count hint metadata should have two operands.");
            unsigned Count = mdconst::extract<ConstantInt>( MD->getOperand( 1))->getZExtValue();
            lccrt_einfo_reference_t lcount = lccrt_einfo_new_struct( le.etyde_loop_count);

            lccrt_einfo_set_field( lcount, le.eifi_lcount_val, lccrt_einfo_new_i64( Count));
            lccrt_oper_set_einfo( ct, le.ecat_loop_count, lcount);
        }
    } else if ( isa<SwitchInst>(O) )
    {
        int d;
        lccrt_var_ptr a = 0;
        int j = 0;
        SwitchInst &SI = cast<SwitchInst>(O);
        lccrt_switch_alts_t *alts = new lccrt_switch_alts_t[SI.getNumCases() + 1];
        Type *ST = SI.getCondition()->getType();
        lccrt_type_ptr t = le.makeType( ST);

        num_alts = SI.getNumCases() + 1;
        a = makeValue( SI.getCondition(), i);
        for ( SwitchInst::ConstCaseIt k = SI.case_begin(), ke = SI.case_end(); k != ke; ++k, ++j )
        {
            alts[j].dst = findBlockLabel( k->getCaseSuccessor());
            if ( le.isIntBitWidthNormalOrBool( ST) ) {
                APInt A = k->getCaseValue()->getValue();
                int bw = A.getBitWidth();

                if ( (bw == 128) ) {
                    alts[j].val = lccrt_varinit_new_str( t, 16, (const char *)A.getRawData());
                } else {
                    alts[j].val = lccrt_varinit_new_scalar( t, A.getZExtValue());
                }
            } else {
                alts[j].val = (lccrt_varinit_ptr)( k->getCaseValue());
            }
        }

        alts[j].val = 0;
        alts[j].dst = findBlockLabel( SI.getDefaultDest());

        if ( le.isIntBitWidthNormalOrBool( ST) ) {
            ct = lccrt_oper_new_switch( f, a, num_alts, alts, i);
            for ( d = 0; d <= j; ++d ) {
                alts_to_opers[d] = ARG_REF( ct, 2 + 2*d);
            }
        } else {
            lccrt_oper_ptr ctd;

            for ( d = 0; d < j; ++d ) {
                lccrt_oper_ptr lbl;
                lccrt_oper_ptr prev = lccrt_oper_iterator_get_prev( i);
                lccrt_var_ptr v1 = lccrt_var_new_local( f, lccrt_type_make_bool( m), 0);
                ICmpInst *CI = new ICmpInst( CmpInst::ICMP_EQ, SI.getCondition(), (Value *)alts[d].val);

                lbl = lccrt_oper_new_label( f, 0, i);
                lccrt_oper_iterator_set( i, prev);
                makeCmp( *CI, v1, i);
                ctd = lccrt_oper_new_branchif( f, v1, lbl, alts[d].dst, i);
                alts_to_opers[d] = ARG_REF( ctd, 2);
                delete CI;
                lccrt_oper_iterator_set( i, lbl);
                if ( (d == 0) ) {
                    assert( !ct);
                    ct = ctd;
                }
            }

            ctd = lccrt_oper_new_branch( f, alts[j].dst, i);
            alts_to_opers[j] = ARG_REF( ctd, 0);
            if ( (j == 0) ) {
                assert( !ct);
                ct = ctd;
            }
        }

        delete[] alts;
    } else {
        errorDump( &O);
    }

    assert( ct);
    assert( num_alts == get_num_normal_alts( O));

    return (num_alts);
} /* LccrtFunctionEmitter::makeBranch */

/**
 * Создание операции.
 */
int
LccrtFunctionEmitter::makeIndirectBranch( Instruction &O, lccrt_oper_ptr &ct, lccrt_v_ptr res,
                                          arg_ref_t *alts_to_opers, lccrt_oi_ptr i)
{
    int num_alts = 0;
    IndirectBrInst &IB = cast<IndirectBrInst>(O);
    lccrt_oper_ptr prev = 0;
    lccrt_oper_ptr ldef = 0;
    lccrt_oper_ptr trap = 0;
    unsigned d = 0;
    unsigned j = 0;
    lccrt_var_ptr a = 0;
    lccrt_switch_alts_t *alts = new lccrt_switch_alts_t[IB.getNumDestinations() + 1];
    lccrt_type_ptr tu32 = lccrt_type_make_u32( m);
    lccrt_type_ptr tf = lccrt_type_make_func( lccrt_type_make_void( m), 0, 0);
    lccrt_var_ptr call_args[1] = {makeCallBuiltinAddr( 0, tf, "__builtin_trap")};

    ct = 0;
    num_alts = IB.getNumDestinations() + 1;
    a = makeValue( IB.getAddress(), i);
    a = lccrt_oper_get_res( lccrt_oper_new_bitcast( f, a, tu32, 0, i));
    for ( j = 0; j < IB.getNumDestinations(); ++j )
    {
        const BasicBlock *bb = IB.getDestination( j);
        int bb_ident = le.funcs_lbls[F][bb];

        alts[j].dst = findBlockLabel( bb);
        alts[j].val = lccrt_varinit_new_scalar( tu32, bb_ident);
    }

    prev = lccrt_oper_iterator_get_prev( i);
    ldef = lccrt_oper_new_label( f, 0, i);
    trap = lccrt_oper_new_call( f, tf, 1, call_args, 0, 0, 0, i);

    lccrt_oper_iterator_set( i, prev);
    alts[j].val = 0;
    alts[j].dst = ldef;

    ct = lccrt_oper_new_switch( f, a, num_alts, alts, i);
    lccrt_oper_iterator_set( i, trap);
    for ( d = 0; d <= j; ++d )
    {
        alts_to_opers[d] = ARG_REF( ct, 2 + 2*d);
    }

    delete[] alts;
    assert( ct);
    assert( num_alts == get_num_normal_alts( O));

    return (num_alts);
} /* LccrtFunctionEmitter::makeIndirectBranch */

/**
 * При несовпадении типов указателей создать bitcast-операцию
 * преобразования типа указателя.
 */
lccrt_var_ptr
LccrtFunctionEmitter::makeBitcastPtr( lccrt_var_ptr v, lccrt_type_ptr rt,
                                      lccrt_oi_ptr i)
{
    lccrt_oper_ptr p2p;
    lccrt_var_ptr r = v;

    assert( lccrt_type_is_pointer( rt));
    assert( lccrt_type_is_pointer( lccrt_var_get_type( v)));

    if ( lccrt_var_get_type( v) != rt ) {
        p2p = lccrt_oper_new_bitcast( f, v, rt, 0, i);
        r = lccrt_oper_get_res( p2p);
    }

    return (r);
} /* LccrtFunctionEmitter::makeBitcastPtr */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeCall( Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i)
{
    std::string math_name;
    int math_args;
    lccrt_type_ptr ct = 0;
    lccrt_oper_ptr oper = 0;
    CallInst &CI = cast<CallInst>(O);
    unsigned num_args = CI.arg_size();
    lccrt_var_ptr *args = new lccrt_var_ptr[num_args + 1];
    const AttributeList &AT = CI.getAttributes();
    Value *CV = CI.getCalledOperand();
    FunctionType *FTy = CI.getFunctionType();
    bool is_mem = false;
    lccrt_type_ptr tpv = lccrt_type_make_pvoid( m);
    lccrt_type_ptr tb = lccrt_type_make_bool( m);
    std::string cn = CV->hasName() ? CV->getName().str() : "";

    args[0] = 0;
    if ( CV->hasName() )
    {
        const Function *CF = (const Function *)CV;
        Type *CT = CF->getFunctionType();
        const char *cns = cn.c_str();
        std::string cnl_std = lowerCallName( m, cns, &ct, 1);
        const char *cnl = cnl_std.empty() ? 0 : cnl_std.c_str();

        if ( le.isDbgDeclare( CF) )
        {
            //dbge_.processDbgDeclare( &CI);
            return;

        } else if ( le.isDbg( CF) )
        {
            return;

        } else if ( (cn.compare( "llvm.lifetime.start") == 0)
                    || (cn.compare( "llvm.lifetime.end") == 0)
                    || (cn.compare( "llvm.lifetime.start.p0") == 0)
                    || (cn.compare( "llvm.lifetime.end.p0") == 0)
                    || (cn.compare( "llvm.lifetime.start.p0i8") == 0)
                    || (cn.compare( "llvm.lifetime.end.p0i8") == 0)
                    || (cn.compare( "llvm.invariant.start") == 0)
                    || (cn.compare( "llvm.invariant.end") == 0)
                    || (cn.compare( "llvm.invariant.start.p0") == 0)
                    || (cn.compare( "llvm.invariant.end.p0") == 0)
                    || (cn.compare( "llvm.invariant.start.p0i8") == 0)
                    || (cn.compare( "llvm.invariant.end.p0i8") == 0)
                    || (cn.compare( "llvm.prefetch") == 0)
                    || (cn.compare( "llvm.prefetch.p0") == 0)
                    || (cn.compare( "llvm.prefetch.p0i8") == 0)
                    || (cn.compare( "llvm.assume") == 0)
                    || (cn.compare( "llvm.experimental.noalias.scope.decl") == 0) )
        {
            return;

        } else if ( (cn == "llvm.type.test") )
        {
            res = res ? res : lccrt_var_new_local( f, lccrt_type_make_bool( m), 0);
            makeLibCallFast( "__lccrt_typetest_unsupported_yet", 0, 0, 0, res, i);
            return;

        } else if ( (cn.find( "llvm.ptr.annotation.") == 0)
                    || (cn.find( "llvm.annotation.") == 0) )
        {
            lccrt_oper_new_move( f, makeValue( O.getOperand( 0), i), res, i);
            return;
        } else if ( (cn.find( "llvm.fptosi.sat.") == 0)
                    || (cn.find( "llvm.fptoui.sat.") == 0) )
        {
            makeFptoiSat( O, res, i);
            return;
        } else if ( (cn == "llvm.frexp.f32.i32")
                    || (cn == "llvm.frexp.f64.i32")
                    || (cn == "llvm.frexp.f80.i32") )
        {
            makeFrexp( O, res, i);
            return;
        } else if ( (cn.find( "llvm.bswap.") == 0) ) {
            makeBswap( O, res, i);
            return;
        } else if ( (cn.find( "llvm.fshl.") == 0) ) {
            makeFshl( O, res, i);
            return;
        } else if ( (cn.find( "llvm.fshr.") == 0) ) {
            makeFshr( O, res, i);
            return;
        } else if ( (cn.find( "llvm.fmuladd.") == 0) ) {
            makeFmuladd( O, res, i);
            return;
        } else if ( (cn.find( "llvm.ctpop.") == 0) ) {
            makeCtpop( O, res, i);
            return;
        } else if ( (cn.find( "llvm.cttz.") == 0) ) {
            makeCttz( O, res, i);
            return;
        } else if ( (cn.find( "llvm.ctlz.") == 0) ) {
            makeCtlz( O, res, i);
            return;
        } else if ( (cn.find( "llvm.sqrt.") == 0) ) {
            makeSqrt( O, res, i);
            return;
        } else if ( (cn.find( "llvm.abs.") == 0) ) {
            makeIntAbs( O, res, i);
            return;
        } else if ( (cn.find( "llvm.smax.") == 0)
                    || (cn.find( "llvm.smin.") == 0)
                    || (cn.find( "llvm.umax.") == 0)
                    || (cn.find( "llvm.umin.") == 0) )
        {
            makeIntMinMax( cn.substr( 5, 4), O, res, i);
            return;

        } else if ( (cn.find( "llvm.bitreverse.") == 0) ) {
            makeBitrev( O, res, i);
            return;

        } else if ( (cn.find( "llvm.vector.reduce.") == 0) )
        {
            makeVectorReduce( O, res, i);
            return;

        } else if ( is_name_suff_stdint_bool( cns, "llvm.expect") )
        {
            lccrt_oper_new_move( f, makeValue( CI.getOperand( 0), i), res, i);
            return;

        } else if ( (cn.find( "llvm.is.constant.") == 0) )
        {
            lccrt_oper_new_move( f, le.makeVarConstHex( tb, 0), res, i);
            return;

        } else if ( (cn.compare( "llvm.memcpy.p0.p0.i32") == 0)
                    || (cn.compare( "llvm.memcpy.p0.p0.i64") == 0)
                    || (cn.compare( "llvm.memmove.p0.p0.i64") == 0)
                    || (cn.compare( "llvm.memcmp.p0.p0.i64") == 0)
                    || (cn.compare( "llvm.memset.p0.i64") == 0)
                    || (cn.compare( "llvm.memcpy.p0i8.p0i8.i32") == 0)
                    || (cn.compare( "llvm.memcpy.p0i8.p0i8.i64") == 0)
                    || (cn.compare( "llvm.memmove.p0i8.p0i8.i64") == 0)
                    || (cn.compare( "llvm.memcmp.p0i8.p0i8.i64") == 0)
                    || (cn.compare( "llvm.memset.p0i8.i64") == 0) )
        {
            num_args = 3;
            is_mem = true;
            args[0] = makeCallBuiltinAddr( CF, ct, cnl);

        } else if ( (cn.compare( "llvm.trap") == 0)
                    || (cn.compare( "llvm.debugtrap") == 0)
                    || (cn.compare( "llvm.stacksave") == 0)
                    || (cn.compare( "llvm.stacksave.p0") == 0) )
        {
            args[0] = makeCallBuiltinAddr( CF, CT, cnl, 0);

        } else if ( (cn.compare( "llvm.va_start") == 0)
                    || (cn.compare( "llvm.va_start.p0") == 0)
                    || (cn.compare( "llvm.va_end") == 0)
                    || (cn.compare( "llvm.va_end.p0") == 0)
                    || (cn.compare( "llvm.stackrestore") == 0)
                    || (cn.compare( "llvm.stackrestore.p0") == 0) )
        {
            args[0] = makeCallBuiltinAddr( CF, CT, cnl, 1);

        } else if ( (cn.compare( "llvm.va_copy") == 0) )
        {
            args[0] = makeCallBuiltinAddr( CF, CT, cnl, 2);

        } else if ( (cn.compare( "llvm.returnaddress") == 0) )
        {
            args[0] = makeCallBuiltinAddr( CF, CT, cnl, 1);

        } else if ( (cn == "llvm.powi.f32")
                    || (cn == "llvm.powi.f32.i32")
                    || (cn == "llvm.powi.f64")
                    || (cn == "llvm.powi.f64.i32")
                    || (cn == "llvm.powi.f80")
                    || (cn == "llvm.powi.f80.i32")
                    || is_name_suff_stdint( cns, "llvm.uadd.with.overflow")
                    || is_name_suff_stdint( cns, "llvm.sadd.with.overflow")
                    || (cn == "llvm.sadd.with.overflow.i65")
                    || is_name_suff_stdint( cns, "llvm.usub.with.overflow")
                    || is_name_suff_stdint( cns, "llvm.ssub.with.overflow")
                    || is_name_suff_stdint( cns, "llvm.umul.with.overflow")
                    || is_name_suff_stdint( cns, "llvm.smul.with.overflow")
                    || is_name_suff_stdint( cns, "llvm.uadd.sat")
                    || is_name_suff_stdint( cns, "llvm.sadd.sat")
                    || is_name_suff_stdint( cns, "llvm.usub.sat")
                    || is_name_suff_stdint( cns, "llvm.ssub.sat")
                    || is_name_suff_stdint( cns, "llvm.umul.sat")
                    || is_name_suff_stdint( cns, "llvm.smul.sat")
                    || is_name_suff_stdfloat( cns, "llvm.is.fpclass")
                    || is_name_suff_stdfloat( cns, "llvm.pow")
                    || is_name_suff_stdfloat( cns, "llvm.minnum")
                    || is_name_suff_stdfloat( cns, "llvm.maxnum")
                    || is_name_suff_stdfloat( cns, "llvm.copysign") )
        {
            args[0] = makeCallBuiltinAddr( CF, CT, cnl, 2);

        } else if ( is_name_suff_stdfloat( cns, "llvm.fabs")
                    || is_name_suff_stdfloat( cns, "llvm.exp")
                    || is_name_suff_stdfloat( cns, "llvm.exp2")
                    || is_name_suff_stdfloat( cns, "llvm.ceil")
                    || is_name_suff_stdfloat( cns, "llvm.floor")
                    || is_name_suff_stdfloat( cns, "llvm.round")
                    || is_name_suff_stdfloat( cns, "llvm.trunc")
                    || is_name_suff_stdfloat( cns, "llvm.rint")
                    || is_name_suff_stdfloat( cns, "llvm.nearbyint")
                    || is_name_suff_stdfloat( cns, "llvm.log")
                    || is_name_suff_stdfloat( cns, "llvm.log2")
                    || is_name_suff_stdfloat( cns, "llvm.log10")
                    || is_name_suff_stdfloat( cns, "llvm.sqrt")
                    || is_name_suff_stdfloat( cns, "llvm.sin")
                    || is_name_suff_stdfloat( cns, "llvm.cos")
                    || is_name_suff_stdfloat( cns, "llvm.tan") )
        {
            args[0] = makeCallBuiltinAddr( CF, CT, cnl, 1);

        } else if ( is_name_suff_stdfloat( cns, "llvm.fma") )
        {
            args[0] = makeCallBuiltinAddr( CF, CT, cnl, 3);
        }
    }

    if ( (CI.getCallingConv() != CallingConv::C)
         && (CI.getCallingConv() != CallingConv::Fast) )
    {
        errorDump( &CI);
    }

    if ( isa<InlineAsm>(CV) )
    {
        makeAsmInline( CI, res, i);

    } else if ( (cn == "llvm.threadlocal.address.p0") ) {
        args[0] = makeValue( CI.getArgOperand( 0), i);
        oper = lccrt_oper_new_bitcast( f, args[0], tpv, res, i);

    } else if ( (cn == "llvm.read_register.i32")
                || (cn == "llvm.read_register.i64")
                || (cn == "llvm.write_register.i32")
                || (cn == "llvm.write_register.i64") )
    {
        makeReadWriteRegister( CI, res, i);

    } else if ( CV->hasName()
                && is_name_math_vec( CV->getName().str(), math_name, math_args) )
    {
        std::string fname = "__lccrt_" + math_name + "_v";

        makeLibCall( fname.c_str(), true, CI, res, false, i);
    } else
    {
        lccrt_type_ptr a0ty = 0;

        ct = ct ? ct : le.makeType( FTy);

        if ( !args[0] ) {
            args[0] = makeValue( CV, i);
        }

        a0ty = lccrt_var_get_type( args[0]);
        if ( !lccrt_type_is_pointer( a0ty) ) {
            assert( 0);
        } else if ( !lccrt_type_is_function( lccrt_type_get_parent( a0ty)) ) {
            args[0] = makeBitcastPtr( args[0], lccrt_type_make_ptr_type( ct), i);
        }

        for ( unsigned k = 0; k < num_args; ++k )
        {
            args[k+1] = makeValue( CI.getArgOperand( k), i);
            assert( (k == 0) || !AT.hasAttributeAtIndex( k+1, Attribute::StructRet));
            if ( is_mem )
            {
                if ( lccrt_type_is_pointer( lccrt_var_get_type( args[k+1])) )
                {
                    args[k+1] = lccrt_oper_get_res( lccrt_oper_new_bitcast( f, args[k+1], tpv, 0, i));
                }
            }

            if ( AT.hasAttributeAtIndex( k+1, Attribute::ByVal) ) {
                if ( le.archTriple.isArchElbrus() ) {
                    errorDump( &CI);
                }
            }
        }

        oper = lccrt_oper_new_call( f, ct, num_args + 1, args, 0, 0, res, i);
        //dbge_.makeOperDbgMetadata( oper, &O);
        dbgle_.makeOperDbgMetadata( oper, &O);
    }

    delete[] args;

    return;
} /* LccrtFunctionEmitter::makeCall */

/**
 * Создание операции.
 */
int
LccrtFunctionEmitter::makeInvoke( Instruction &O, lccrt_oper_ptr &ct_res, lccrt_v_ptr res,
                                  arg_ref_t *alts_to_opers, lccrt_oi_ptr i)
{
    lccrt_oper_ptr o1, o2;
    lccrt_type_ptr a0ty = 0;
    InvokeInst &II = cast<InvokeInst>(O);
    unsigned num_args = II.arg_size();
    lccrt_var_ptr *args = new lccrt_var_ptr[num_args + 1];
    const AttributeList &AT = II.getAttributes();
    Value *CV = II.getCalledOperand();
    lccrt_type_ptr ct = le.makeType( II.getFunctionType());

    if ( (II.getCallingConv() != CallingConv::C)
         && (II.getCallingConv() != CallingConv::Fast) )
    {
        errorDump( &II);
    }

    o1 = lbls.find( II.getNormalDest())->second;
    o2 = lbls.find( II.getUnwindDest())->second;

    args[0] = makeValue( CV, i);
    a0ty = lccrt_var_get_type( args[0]);
    if ( !lccrt_type_is_pointer( a0ty) ) {
        assert( 0);
    } else if ( !lccrt_type_is_function( lccrt_type_get_parent( a0ty)) ) {
        args[0] = makeBitcastPtr( args[0], lccrt_type_make_ptr_type( ct), i);
    }

    for ( unsigned k = 0; k < num_args; ++k )
    {
        args[k+1] = makeValue( II.getArgOperand( k), i);
        assert( (k == 0) || !AT.hasAttributeAtIndex( k+1, Attribute::StructRet));
        if ( AT.hasAttributeAtIndex( k+1, Attribute::ByVal) )
        {
            errorDump( &II);
        }
    }

    ct_res = lccrt_oper_new_invoke( f, ct, num_args + 1, args, 0, 0, o1, o2, res, i);
    alts_to_opers[0] = ARG_REF( ct_res, 0);
    alts_to_opers[1] = ARG_REF( ct_res, 1);
    assert( 2 == get_num_normal_alts( O));

    delete[] args;

    return (2);
} /* LccrtFunctionEmitter::makeInvoke */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeLandingpad( const Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i)
{
    lccrt_oper_ptr oper;
    const LandingPadInst &LI = cast<LandingPadInst>(O);
    unsigned num_args = LI.getNumClauses();
    lccrt_var_ptr *args = new lccrt_var_ptr[2*num_args];
    lccrt_type_ptr t = le.makeType( LI.getType());

    for ( unsigned k = 0; k < num_args; ++k )
    {
        lccrt_varinit_ptr vi = le.makeVarinit( LI.getClause( k), 0);

        args[2*k+0] = le.makeVarConstHex( lccrt_type_make_u64( m), LI.isFilter( k));
        args[2*k+1] = le.makeVarConst( lccrt_varinit_get_type( vi), vi);;
    }

    oper = lccrt_oper_new_landingpad( f, 2*num_args, args, t, res, i);
    lccrt_oper_set_cleanup( oper, LI.isCleanup());

    delete[] args;

    return;
} /* LccrtFunctionEmitter::makeLandingpad */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeResume( const Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i)
{
    const ResumeInst &RI = cast<ResumeInst>(O);
    lccrt_var_ptr args[2] = {0, 0};
    lccrt_type_ptr t0 = 0;
    lccrt_type_ptr t1 = 0;

    t0 = le.makeType( RI.getType());
    t1 = lccrt_type_make_func( lccrt_type_make_void( m), 1, &t0);

    args[0] = makeCallBuiltinAddr( 0, t1, "__lccrt_eh_resume");
    args[1] = makeValue( RI.getValue(), i);

    lccrt_oper_new_call( f, t1, 2, args, 0, 0, 0, i);
    lccrt_oper_new_return( f, 0, i);
    lccrt_oper_new_label( f, 0, i);

    return;
} /* LccrtFunctionEmitter::makeResume */

/**
 * Преобразование asm-модидфикатора.
 */
std::string
LccrtFunctionEmitter::evalAsmConstraint( std::string ac) {
    std::string r = "";

    if ( ac.empty() ) r= "";
    else if ( is_number( ac.c_str()) ) r = ac;
    else if ( (ac.compare( "r") == 0) ) r = "r";
    else if ( (ac.compare( "x") == 0) ) r = "x";
    else if ( (ac.compare( "i") == 0) ) r = "i";
    else if ( (ac.compare( "J") == 0) ) r = "J";
    else if ( (ac.compare( "m") == 0) ) r = "m";
    else if ( (ac.compare( "{ax}") == 0) ) r = "a";
    else if ( (ac.compare( "{bx}") == 0) ) r = "b";
    else if ( (ac.compare( "{cx}") == 0) ) r = "c";
    else if ( (ac.compare( "{di}") == 0) ) r = "D";
    else {
        const char *s = ac.c_str();

        if ( s[0] == '{' ) {
            int off = 1;
            if ( is_e2k_reg( s, off)
                 && (s[off] == '}')
                 && (s[off+1] == 0) ) {
                //r = ac.substr( 1, off - 1);
                r = "r";
            }
        }
    }

    return (r);
} /* LccrtFunctionEmitter::evalAsmConstraint */

/**
 * Преобразование строки модификатороы. Возвращается пустая строка, если разбор
 * завершился не успешно.
 */
std::string
LccrtFunctionEmitter::evalAsmConstraintVector( const InlineAsm::ConstraintCodeVector &Codes) {
    std::string r;

    if ( (Codes.size() != 1) ) {
        bool r_code = false;

        for ( unsigned ci = 0; ci < Codes.size(); ++ci ) {
            if ( (Codes[ci] == "r")
                 || (Codes[ci] == "x")
                 || (Codes[ci] == "m") ) {
                r_code = true;
                r += Codes[ci];
            } else if ( (Codes[ci] == "i")
                        || (Codes[ci] == "I")
                        || (Codes[ci] == "J") ) {
                r += Codes[ci];
            } else {
                r_code = false;
                break;
            }
        }

        if ( !r_code ) {
            r = "";
        }
    } else {
        r = evalAsmConstraint( Codes[0]);
    }

    return (r);
} /* LccrtFunctionEmitter::evalAsmConstraintVector */

/**
 * Проверить, что в начале строки находится число. Вернуть значение числа и длину
 * префикса с числом.
 */
bool
LccrtFunctionEmitter::parseInt( const char *str, int &len, int64_t &value)
{
    bool r = false;

    if ( isdigit( str[0]) )
    {
        char *q;
        int karg = strtol( str, &q, 10);

        if ( (q > str) )
        {
            r = true;
            len = (q - str);
            value = karg;
        }
    }

    return (r);
} /* LccrtFunctionEmitter::parseInt */

/**
 * Проверяем, что входная строка начинается с ${K}, где  K - некоторое число.
 * Возвращается длина префикса и строка с аргументов вида %K.
 */
bool
LccrtFunctionEmitter::parseAsmInlineArg( const char *p, int &len, std::string &arg)
{
    char buff[64];
    bool r = false;
    int64_t karg = -1;
    int kl = 0;

    if ( (p[0] == '$')
         && isdigit( p[1])
         && parseInt( p + 1, kl, karg)
         && (karg >= 0) )
    {
        r = true;
        len = kl + 1;

    } else if ( (p[0] == '$')
                && (p[1] == '{')
                && isdigit( p[2])
                && parseInt( p + 2, kl, karg)
                && (karg >= 0)
                && (p[2 + kl] == '}') )
    {
        r = true;
        len = kl + 3;

    } else if ( is_str_head( p, "$$$(")
                && isdigit( p[4])
                && parseInt( p + 4, kl, karg)
                && (karg >= 0)
                && is_str_head( p + 4 + kl, "$)") )
    {
        r = true;
        len = kl + 6;
    } 

    if ( r )
    {
        snprintf( buff, 64, "%%%d", (int)karg);
        arg = buff;
    }

    return (r);
} /* LccrtFunctionEmitter::parseAsmInlineArg */

/**
 * Проверка набора constraint'ов параметров на равенство числовому индексу.
 */
static bool
isConstraintNumber( const std::vector<std::string> &codes, int &index) {
    bool r = false;

    if ( codes.size() == 1 ) {
        const char *p = codes[0].c_str();
        char *q = 0;

        index = strtol( p, &q, 10);
        r = (q[0] == 0);
    }

    return (r);
} /* isConstraintNumber */

/**
 * Создание аргументов операции.
 */
void
LccrtFunctionEmitter::makeAsmInline( const CallInst &O, lccrt_var_ptr res, lccrt_oi_ptr i)
{
    char buf[256];
    std::string asm_text;
    int k = 1;
    bool is_call_clobber = false;
    InlineAsm::ConstraintInfoVector::iterator j, j1;
    const Value *V = O.getCalledOperand();
    const InlineAsm *IA = dyn_cast<InlineAsm>(V);
    InlineAsm::ConstraintInfoVector civ = IA->ParseConstraints();
    std::vector<std::string> civstr;
    lccrt_var_ptr *args = new lccrt_var_ptr[civ.size() + 1];
    Type *T1 = O.getType();
    StructType *T2 = dyn_cast<StructType>(T1);
    int num_rargs = T2 ? T2->getNumElements() : (T1->isVoidTy() ? 0 : 1);
    lccrt_var_ptr *rvars = new lccrt_var_ptr[civ.size() + 1];
    lccrt_type_ptr *fte = new lccrt_type_ptr[civ.size() + 1];
    lccrt_type_ptr ct = 0;
    lccrt_type_ptr tu64 = lccrt_type_make_u64( m);
    lccrt_type_ptr tpv = lccrt_type_make_pvoid( m);
    lccrt_oper_ptr asm_oper = 0;
    int num_ins = 0;
    int num_dirouts = 0;
    int num_indirouts = 0;

    //assert( !IA->isAlignStack());
    assert( IA->getDialect() == InlineAsm::AD_ATT);
    snprintf( buf, 256, "' tag:%p\n", (const void *)IA);
    asm_text += buf;
    asm_text += IA->getAsmString();
    civstr.resize( civ.size());

    //if ( (civ.begin() != civ.end()) )
    {
        std::string s;

        for ( unsigned l = 0; l < asm_text.size(); ++l ) {
            std::string s_arg;
            int l_next = -1;
            int sym_curr = asm_text[l];
            int sym_next = (l + 1 < asm_text.size()) ? asm_text[l+1] : -1;

            if ( parseAsmInlineArg( asm_text.c_str() + l, l_next, s_arg) ) {
                assert( l_next >= 2);
                l += l_next - 1;
                s += s_arg;
            } else if ( is_str_head( asm_text.c_str() + l, "$$") ) {
                l += 1;
                s += "$";
            } else if ( is_str_head( asm_text.c_str() + l, "$(") ) {
                l += 1;
                s += "{";
            } else if ( is_str_head( asm_text.c_str() + l, "$)") ) {
                l += 1;
                s += "}";
            } else if ( is_str_head( asm_text.c_str() + l, "%#") ) {
                l += 1;
                s += "%#";
            } else if ( (sym_curr == '%')
                        && (sym_next >= 0)
                        && !isdigit( sym_next)
                        && ((l == 0)
                            || (asm_text[l-1] != '%')) ) {
                s += "%%";
            } else {
                s += asm_text[l];
            }
        }

        asm_text = s;
        asm_text += "!";
    }

    if ( (num_rargs > 0) ) {
        if ( !res ) {
            res = makeVarRes( le.makeType( T1));
        }
    }

    //dbgs() << O << "\n";

    for ( j = civ.begin(), j1 = civ.end(); j != j1; ++j )
    {
        std::string sj;
        int nargs = num_dirouts + num_indirouts + num_ins;

        if ( !((j->Type == InlineAsm::isOutput)
               || (j->Type == InlineAsm::isInput)
               || (j->Type == InlineAsm::isClobber)) ) {
            errorDump( &O);
        }

        if ( j->isCommutative
             || j->currentAlternativeIndex ) {
            errorDump( &O);
        }

        if ( j->isIndirect ) {
            if ( j->Type == InlineAsm::isClobber ) errorDump( &O);
        }

        if ( !j->isMultipleAlternative ) {
            if ( j->Codes.empty() ) {
                errorDump( &O);
            }

            sj = evalAsmConstraintVector( j->Codes);
            if ( sj.empty() ) {
                if ( (j->Type == InlineAsm::isInput)
                     || (j->Type == InlineAsm::isOutput) ) {
                    errorDump( &O);
                }
            }
        } else {
            if ( !j->Codes.empty()
                 || j->multipleAlternatives.empty() ) {
                errorDump( &O);
            }

            for ( auto jc : j->multipleAlternatives ) {
                if ( jc.MatchingInput >= 0 ) {
                    errorDump( &O);
                } else {
                    std::string tj = evalAsmConstraintVector( jc.Codes);

                    if ( tj.empty() ) {
                        errorDump( &O);
                    } else {
                        sj += tj;
                    }
                }
            }
        }

        if ( j->Type == InlineAsm::isOutput ) {
            int match = j->MatchingInput;

            civstr[nargs] = sj;
            if ( num_ins > 0 ) errorDump( &O);
            if ( sj.empty() ) errorDump( &O);
            if ( match >= 0 ) {
                int arg_ind = -1;

                if ( match >= (int)civ.size()
                     || !isConstraintNumber( civ[match].Codes, arg_ind)
                     || (arg_ind != k - 1) ) {
                    errorDump( &O);
                }
            }

            if ( j->isIndirect ) {
                num_indirouts++;
                asm_text += "`+";
                if ( j->isEarlyClobber ) errorDump( &O);
                args[k] = makeValue( O.getOperand( num_indirouts - 1));
                if ( (sj == "mr") || (sj == "rm") ) {
                    //sj = "r";
                    sj = "m";
                    if ( O.getOperand( num_indirouts - 1)->getType()->isPointerTy() ) {
                        //sj = "m";
                    } else {
                        //sj = "r";
                    }
                }
            } else {
                lccrt_type_ptr te = le.makeType( T2 ? T2->getElementType( num_dirouts) : T1);
                lccrt_oper_ptr vptr = 0;

                num_dirouts++;
                asm_text += "`=";
                if ( j->isEarlyClobber ) asm_text += "&";
                rvars[num_dirouts - 1] = makeVarRes( te);
                vptr = lccrt_oper_new_varptr( f, rvars[num_dirouts - 1], 0, i);
                args[k] = lccrt_oper_get_res( vptr);
            }

            asm_text += sj + "`";
            fte[k-1] = lccrt_var_get_type( args[k]);
            k++;
        } else if ( j->Type == InlineAsm::isInput ) {
            int arg_ind = -1;

            num_ins++;
            if ( sj.empty() ) errorDump( &O);
            if ( isConstraintNumber( j->Codes, arg_ind) ) {
                char buf[256];

                snprintf( buf, 256, "%d", arg_ind);
                sj = buf;
                if ( !((0 <= arg_ind)
                       && (arg_ind < (int)civ.size())
                       && (arg_ind < num_dirouts + num_indirouts)
                       && (civ[arg_ind].MatchingInput == nargs)) ) {
                    errorDump( &O);
                }
            }

            civstr[nargs] = sj;
            if ( j->isIndirect ) {
                std::string lj = sj;

                if ( arg_ind >= 0 ) {
                    lj = civstr[arg_ind];
                }

                if ( (lj == "rm") || (lj == "mr") || (lj == "imr") || (lj == "irm") ) {
                    //lj = "r";
                    lj = "m";
                }

                if ( j->isEarlyClobber ) errorDump( &O);
                if ( (lj != "m") && (lj != "r") ) errorDump( &O);
            } else {
                if ( (sj == "rm") || (sj == "mr") ) {
                    //sj = "r";
                    sj = "m";
                }
            }

            asm_text += "`" + sj + "`";
            args[k] = makeValue( O.getOperand( num_indirouts + num_ins - 1), i);
            fte[k-1] = lccrt_var_get_type( args[k]);
            k++;
        } else if ( (j->Type == InlineAsm::isClobber) ) {
            std::string clob = j->Codes[0].substr( 1, j->Codes[0].length() - 2);

            civstr[nargs] = clob;
            if ( !((le.archTriple.getArch() == Triple::e2k32)
                   || (le.archTriple.getArch() == Triple::e2k64)) ) {
                asm_text += "?" + clob + "?";
            } else {
                if ( !is_call_clobber ) {
                    is_call_clobber = true;
                    asm_text += "?call?";
                }

                if ( (clob.rfind( "r", 0) == 0)
                     || (clob.rfind( "g", 0) == 0)
                     || (clob.rfind( "pred", 0) == 0)
                     || (clob.rfind( "b[", 0) == 0)
                     || (clob.rfind( "ctpr", 0) == 0)
                     || (clob.rfind( "memory", 0) == 0) ) {
                    asm_text += "?" + clob + "?";
                }
            }
        } else {
            errorDump( &O);
        }
    }

    for ( int kt = num_dirouts; kt < k - 1; ++kt ) {
        if ( lccrt_type_is_pointer( fte[kt]) && (fte[kt] != tpv) ) {
            lccrt_oper_ptr p2p = lccrt_oper_new_bitcast( f, args[kt+1], tpv, 0, i);

            args[kt + 1] = lccrt_oper_get_res( p2p);
            fte[kt] = tpv;
        }
    }

    ct = lccrt_type_make_func( lccrt_type_make_void( m), k - 1, fte);
    args[0] = lccrt_var_new_asm( f, ct, asm_text.c_str(), 0);
    asm_oper = lccrt_oper_new_call( f, ct, k, args, 0, 0, 0, i);
    if ( IA->hasSideEffects() ) lccrt_oper_set_volatile( asm_oper, 1);
    if ( num_rargs != num_dirouts ) errorDump( &O);

    for ( int nr = 0; nr < num_rargs; ++nr ) {
        if ( T2 ) {
            lccrt_var_ptr wargs[3];

            wargs[0] = res;
            wargs[1] = rvars[nr];
            wargs[2] = le.makeVarConst( tu64, lccrt_varinit_new_scalar( tu64, nr));
            lccrt_oper_new_elemwrite( f, 3, wargs, i);
        } else {
            assert( num_rargs == 1);
            lccrt_oper_new_move( f, rvars[nr], res, i);
        }
    }

    delete[] fte;
    delete[] rvars;
    delete[] args;

    return;
} /* LccrtFunctionEmitter::makeAsmInline */

/**
 * Создание операций чтения/записи регистра.
 */
void
LccrtFunctionEmitter::makeReadWriteRegister( const CallInst &CI, lccrt_var_ptr res, lccrt_oi_ptr i)
{
    char buf[256];
    Value *reg = CI.getArgOperand( 0);
    MDNode *md = cast<MDNode>( cast<MetadataAsValue>( reg)->getMetadata());
    MDString *ms = dyn_cast<MDString>( md->getOperand( 0));
    std::string sreg = std::string( "%") + ms->getString().data();
    std::string name = std::string( CI.getCalledOperand()->getName());
    bool is_read = false;
    lccrt_type_ptr ti = 0;
    lccrt_type_ptr ti32 = lccrt_type_make_u32( m);
    lccrt_type_ptr ti64 = lccrt_type_make_u64( m);
    std::string asm_text = "";
    lccrt_var_ptr v = 0;
    lccrt_var_ptr args[2] = {};
    lccrt_type_ptr ct = 0;

    if ( (name == "llvm.read_register.i32") )       { is_read = true;  ti = ti32; }
    else if ( (name == "llvm.read_register.i64") )  { is_read = true;  ti = ti64; }
    else if ( (name == "llvm.write_register.i32") ) { is_read = false; ti = ti32; }
    else if ( (name == "llvm.write_register.i64") ) { is_read = false; ti = ti64; }
    else { assert( 0); }

    snprintf( buf, 256, "' tag:%p\n", (const void *)&CI);
    asm_text += buf;

    snprintf( buf, 256, "add%s %%%s, 0, %%%s!", (ti == ti32) ? "s" : "d",
              is_read ? sreg.c_str() : "0", is_read ? "0" : sreg.c_str());
    asm_text += buf;

    snprintf( buf, 256, "`%sr`?%s?", is_read ? "" : "=r", sreg.c_str());
    asm_text += buf;

    v = lccrt_var_new_local( f, ti, 0);
    ct = lccrt_type_make_func( lccrt_type_make_void( m), 1, &ti);
    args[0] = lccrt_var_new_asm( f, ct, asm_text.c_str(), 0);
    args[1] = v;
    lccrt_oper_new_call( f, ct, 2, args, 0, 0, 0, i);

    if ( is_read )
    {
        lccrt_oper_new_move( f, v, res, i);
    }

    return;
} /* LccrtFunctionEmitter::makeReadWriteRegister */

/**
 * Создание переменной с адресом функции.
 */
lccrt_var_ptr
LccrtFunctionEmitter::makeCallBuiltinAddr( const Function *F, lccrt_t_ptr type, const char *name)
{
    lccrt_link_t lnk;
    lccrt_function_ptr f;
    lccrt_var_ptr r = 0;
    MapLLVMFuncToFunc::const_iterator k = F ? le.funcs.find( F) : le.funcs.end();

    lnk = le.makeLink( GlobalValue::ExternalLinkage, GlobalValue::DefaultVisibility,
                       GlobalVariable::NotThreadLocal, 0, 0, 1);

    if ( (k != le.funcs.end()) )
    {
        f = k->second;
        assert( lnk == lccrt_function_get_link( f));
    } else
    {
        f = lccrt_function_new( m, type, name, 0, lnk, 1, 0);
        le.funcs.insert( PairLLVMFuncToFunc( F, f));
    }

    r = le.makeVarConst( lccrt_type_make_ptr_type( type),
                         lccrt_varinit_new_addr_func( f, 0));

    return (r);
} /* LccrtFunctionEmitter::makeCallBuiltinAddr */

/**
 * Создание переменной с адресом функции.
 */
lccrt_var_ptr
LccrtFunctionEmitter::makeCallBuiltinAddr( const Function *F, Type *T, const char *name, int num_args)
{
    int k;
    lccrt_type_ptr t, tres;
    lccrt_var_ptr r = 0;
    FunctionType *Tf = dyn_cast<FunctionType>( T);
    lccrt_type_ptr *elems = new lccrt_type_ptr[num_args];

    tres = le.makeType( Tf->getReturnType());
    for ( k = 0; k < num_args; ++k )
    {
        elems[k] = le.makeType( Tf->getParamType( k));
    }

    t = lccrt_type_make_func( tres, num_args, elems);
    delete[] elems;

    r = makeCallBuiltinAddr( F, t, name);

    return (r);
} /* LccrtFunctionEmitter::makeCallBuiltinAddr */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeCmp( const Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i)
{
    std::string sn;
    const char *sns = 0;
    const CmpInst &CI = cast<CmpInst>(O);
    lccrt_oper_ptr oper = 0;
    lccrt_var_ptr a = 0;
    lccrt_var_ptr b = 0;
    lccrt_varinit_ptr n = 0;
    lccrt_cmp_name_t cn = LCCRT_CMP_LAST;
    lccrt_type_ptr tu32 = lccrt_type_make_u32( m);
    lccrt_type_ptr tu64 = lccrt_type_make_u64( m);
    Value *V1 = CI.getOperand( 0);
    Value *V2 = CI.getOperand( 1);
    Type *T1 = V1->getType();    

    cn = getCmpLccrtName( CI.getPredicate(), &sns);
    if ( sns ) {
        sn = sns;
    }

    if ( (isa<IntegerType>( T1)
          && le.isIntBitWidthNormal( T1))
         || isa<PointerType>( T1)
         || T1->isFloatTy()
         || T1->isDoubleTy()
         || T1->isFP128Ty()
         || T1->isX86_FP80Ty() )
    {
        n = lccrt_varinit_new_scalar( lccrt_type_make_u32( m), cn);
        a = makeValue( V1, i);
        b = makeValue( V2, i);
        if ( isa<PointerType>( T1) )
        {
            lccrt_type_ptr t = lccrt_type_make_intptr( m);

            a = lccrt_oper_get_res( lccrt_oper_new_bitcast( f, a, t, 0, i));
            b = lccrt_oper_get_res( lccrt_oper_new_bitcast( f, b, t, 0, i));
        }

        oper = lccrt_oper_new_cmp( f, n, a, b, res, i);

    } else if ( isa<IntegerType>( T1) )
    {
        if ( (le.DL.getTypeAllocSize( T1) <= 8) )
        {
            lccrt_var_ptr a1 = makeValue( V1, i);
            lccrt_var_ptr a2 = makeValue( V2, i);
            uint64_t bitsize = le.DL.getTypeSizeInBits( T1);
            lccrt_type_ptr tur = (bitsize <= 32) ? tu32 : tu64;

            a1 = lccrt_oper_get_res( makeBitcastNIntToInt( bitsize, a1, 0, i));
            a2 = lccrt_oper_get_res( makeBitcastNIntToInt( bitsize, a2, 0, i));

            if ( (cn == LCCRT_CMP_LT_I)
                 || (cn == LCCRT_CMP_LE_I)
                 || (cn == LCCRT_CMP_GT_I)
                 || (cn == LCCRT_CMP_GE_I) )
            {
                int align_bitsize = (bitsize <= 32) ? 32 : 64;
                lccrt_var_ptr c0 = le.makeVarConstHex( tur, (align_bitsize - bitsize) % align_bitsize);

                a1 = lccrt_oper_get_res( lccrt_oper_new_shl( f, a1, c0, 0, i));
                a1 = lccrt_oper_get_res( lccrt_oper_new_sar( f, a1, c0, 0, i));
                a2 = lccrt_oper_get_res( lccrt_oper_new_shl( f, a2, c0, 0, i));
                a2 = lccrt_oper_get_res( lccrt_oper_new_sar( f, a2, c0, 0, i));
            }

            n = lccrt_varinit_new_scalar( lccrt_type_make_u32( m), cn);
            oper = lccrt_oper_new_cmp( f, n, a1, a2, res, i);
        } else
        {
            sn = "__lccrt_cmp_n" + sn;
            makeLibCall( sn.c_str(), false, CI, res, false, i);
        }
    } else if ( isa<FixedVectorType>( T1) )
    {
        char sf[256];
        FixedVectorType *TV1 = static_cast<FixedVectorType *>( T1);
        Type *TVE1 = TV1->getElementType();
        int num_elems = TV1->getNumElements();
        int elem_size = le.DL.getTypeAllocSize( TVE1);

        if ( (TVE1->isFloatingPointTy()
              && ((elem_size == 4)
                  || (elem_size == 8)))
             || (TVE1->isIntegerTy()
                 && le.isIntBitWidthNormal( TVE1)
                 && (elem_size <= 8))
             || (TVE1->isPointerTy()) )
        {
            if ( ((num_elems * elem_size == 8)
                  || (num_elems * elem_size == 16)
                  || (num_elems * elem_size == 32))
                 && isFastLibCall( &O, __FILE__, __LINE__) )
            {
                lccrt_type_ptr tye = lccrt_type_make_int( m, elem_size, 0);
                lccrt_type_ptr tyve = lccrt_type_make_array( tye, num_elems);
                lccrt_var_ptr v0 = lccrt_var_new_local( f, tyve, 0);
                char suff = TVE1->isFloatingPointTy() ? 'f' : 'i';

                snprintf( sf, 256, "__lccrt_builtin_cmp%s_v%d%c%d",
                          sn.c_str(), num_elems, suff, 8*elem_size);
                makeLibCallFast( sf, CI, v0, i);

                snprintf( sf, 256, "__lccrt_builtin_trunc_v%di%di1",
                          num_elems, 8*elem_size);
                makeLibCallFast( sf, v0, 0, 0, res, i);
            } else {
                sn = "__lccrt_cmp_v" + sn;
                makeLibCall( sn.c_str(), true, CI, res, false, i);
            }
        } else {
            errorDump( &O);
            assert( 0);
        }
    } else {
        errorDump( T1);
        assert( 0);
    }

    if ( oper )
    {
        //dbge_.makeOperDbgMetadata( oper, &O);
        dbgle_.makeOperDbgMetadata( oper, &O);
    }

    return;
} /* LccrtFunctionEmitter::makeCmp */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeLoadStore( Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i)
{
    lccrt_var_ptr a = 0;
    lccrt_var_ptr b = 0;
    lccrt_oper_ptr r = 0;
    lccrt_type_ptr tp = 0;
    const LoadInst *LD = dyn_cast<LoadInst>( &O);
    const StoreInst *ST = dyn_cast<StoreInst>( &O);
    AtomicOrdering ord = LD ? LD->getOrdering() : ST->getOrdering();
    int is_volatile = LD ? LD->isVolatile() : ST->isVolatile();
    int is_atomic = LD ? LD->isAtomic() : ST->isAtomic();
    Value *V0 = O.getOperand( 0);
    Type *T0 = LD ? O.getType() : V0->getType();

    if ( is_atomic ) {
    } else {
        if ( (ord != AtomicOrdering::NotAtomic) ) {
            assert( 0);
        }
    }

    if ( !le.isTypeNonStdVector( T0) ) {
        if ( LD ) {
            tp = lccrt_type_make_ptr_type( lccrt_var_get_type( res));
            a = makeValuePtrcast( LD->getOperand( 0), tp, i);
            r = lccrt_oper_new_load( f, a, res, i);
        } else {
            b = makeValue( ST->getOperand( 0), i);
            tp = lccrt_type_make_ptr_type( lccrt_var_get_type( b));
            a = makeValuePtrcast( ST->getOperand( 1), tp, i);
            r = lccrt_oper_new_store( f, a, b, i);
        }
    } else {
        lccrt_var_ptr p = 0;
        lccrt_var_ptr v = 0;
        lccrt_type_ptr td = le.makeTypeDenseVector( T0);
        lccrt_type_ptr tdp = lccrt_type_make_ptr_type( td);

        a = makeValue( V0, i);
        p = lccrt_var_new_local( f, tdp, 0);
        v = lccrt_var_new_local( f, td, 0);
        if ( LD ) {
            lccrt_oper_new_bitcast( f, a, tdp, p, i);
            r = lccrt_oper_new_load( f, p, v, i);
            makeVecbitRepack( O, T0, false, v, res, i);
        } else {
            b = makeValue( ST->getOperand( 1), i);
            lccrt_oper_new_bitcast( f, b, tdp, p, i);
            makeVecbitRepack( O, T0, true, a, v, i);
            r = lccrt_oper_new_store( f, p, v, i);
        }
    }

    if ( is_volatile ) {
        lccrt_oper_set_volatile( r, 1);
    }

    if ( is_atomic ) {
        lccrt_oper_set_atomic( r, 1);
    }

    FIPA.setOperIpaResult( r, O);
    //dbge_.makeOperDbgMetadata( r, &O);
    dbgle_.makeOperDbgMetadata( r, &O);

    return;
} /* LccrtFunctionEmitter::makeLoadStore */

/**
 * Для невекторного типа вернуь 0. Для фиксированного вектора вернуть битовый размер
 * элемента вектора. Для нефиксированного вектора вернуть -1.
 */
int
LccrtEmitter::getElementBitsize( Type *T) {
    int r = 0;

    if ( isa<ScalableVectorType>( T) ) {
        r = -1;
    } else if ( isa<FixedVectorType>( T) ) {
        FixedVectorType *TV = static_cast<FixedVectorType *>( T);
        Type *TVE = TV->getElementType();

        r = DL.getTypeSizeInBits( TVE);
    }

    return (r);
} /* LccrtEmitter::getElementBitsize */

void
LccrtFunctionEmitter::makeVecbitRepack( Instruction &O, Type *T, bool ispack,
                                        lccrt_var_ptr a, lccrt_v_ptr res, lccrt_oi_ptr i)
{
    lccrt_oper_ptr o1, o2;
    lccrt_var_ptr pa, pr, ne, be;
    const char *opname;
    lccrt_type_ptr tu64 = lccrt_type_make_u64( m);
    lccrt_type_ptr tpv = lccrt_type_make_pvoid( m);
    FixedVectorType *TV = dyn_cast<FixedVectorType>( T);
    int elem_bitsize = le.DL.getTypeSizeInBits( TV->getElementType());

    assert( le.isTypeNonStdVector( T));
    if ( ispack ) {
        opname = "__lccrt_vecbitpack";
    } else {
        opname = "__lccrt_vecbitunpack";
    }

    o1 = lccrt_oper_new_varptr( f, a, 0, i);
    o2 = lccrt_oper_new_varptr( f, res, 0, i);
    o1 = lccrt_oper_new_bitcast( f, lccrt_oper_get_res( o1), tpv, 0, i);
    o2 = lccrt_oper_new_bitcast( f, lccrt_oper_get_res( o2), tpv, 0, i);
    pa = lccrt_oper_get_res( o1);
    pr = lccrt_oper_get_res( o2);
    ne = le.makeVarConstHex( tu64, TV->getNumElements());
    be = le.makeVarConstHex( tu64, elem_bitsize);
    makeLibCallFast( opname, {pr, pa, ne, be}, 0, i);

    return;
} /* LccrtFunctionEmitter::makeVecbitRepack */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeBitcast( Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i)
{
    BitCastInst &BI = cast<BitCastInst>(O);
    Value *V1 = O.getOperand( 0);
    Type *TR = BI.getType();
    Type *T1 = V1->getType();
    lccrt_var_ptr a = makeValue( V1, i);
    lccrt_type_ptr t = le.makeType( TR);

    if ( le.isTypeNonStdVector( T1) ) {
        lccrt_type_ptr td = le.makeTypeDenseVector( T1);
        lccrt_var_ptr v = lccrt_var_new_local( f, td, 0);

        makeVecbitRepack( O, T1, true, a, v, i);
        a = v;
    }

    if ( !le.isTypeNonStdVector( TR) ) {
        lccrt_oper_new_bitcast( f, a, t, res, i);
    } else {
        lccrt_type_ptr td = le.makeTypeDenseVector( TR);
        lccrt_var_ptr v = lccrt_var_new_local( f, td, 0);

        lccrt_oper_new_bitcast( f, a, td, v, i);
        makeVecbitRepack( O, TR, false, v, res, i);
    }

    return;
} /* LccrtFunctionEmitter::makeBitcast */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeCast( Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i)
{
    CastInst &CI = cast<CastInst>(O);
    Value *V1 = O.getOperand( 0);
    Type *TR = CI.getType();
    Type *T1 = V1->getType();
    lccrt_type_ptr t = le.makeType( TR);
    std::string func_name = "";
    int is_unsig = 0;
    unsigned opcode = CI.getOpcode();

    switch ( CI.getOpcode() )
    {
      case Instruction::SExt:     func_name = "sext";    is_unsig = 1; break;
      case Instruction::ZExt:     func_name = "zext";    is_unsig = 1; break;
      case Instruction::Trunc:    func_name = "trunc";   is_unsig = 1; break;
      case Instruction::BitCast:  func_name = "bitcast"; break;
      case Instruction::PtrToInt: func_name = "bitcast"; break;
      case Instruction::IntToPtr: func_name = "bitcast"; break;
      case Instruction::FPExt:    func_name = "fptofp";  break;
      case Instruction::FPTrunc:  func_name = "fptofp";  break;
      case Instruction::FPToUI:   func_name = "fptoui";  break;
      case Instruction::FPToSI:   func_name = "fptosi";  break;
      case Instruction::UIToFP:   func_name = "uitofp";  break;
      case Instruction::SIToFP:   func_name = "sitofp";  break;
      default:
        errorDump( &CI);
        break;
    }

    if ( isa<ScalableVectorType>( T1) ) {
        errorDump( &O);
    } else if ( CI.getOpcode() == Instruction::BitCast ) {
        makeBitcast( O, res, i);
    } else if ( isa<FixedVectorType>( T1)
                && (CI.getOpcode() != Instruction::BitCast) )
    {
        char sf[256];
        FixedVectorType *TVR = static_cast<FixedVectorType *>( TR);
        FixedVectorType *TV1 = static_cast<FixedVectorType *>( T1);
        Type *TVER = TVR->getElementType();
        Type *TVE1 = TV1->getElementType();
        int num_elems = TV1->getNumElements();
        int elem_sizer = le.DL.getTypeAllocSize( TVER);
        int elem_size1 = le.DL.getTypeAllocSize( TVE1);
        int elem_bitsizer = le.DL.getTypeSizeInBits( TVER);
        int elem_bitsize1 = le.DL.getTypeSizeInBits( TVE1);
        int size1 = num_elems * elem_size1;

        switch ( CI.getOpcode() )
        {
          case Instruction::PtrToInt: func_name = "ptoi"; break;
          case Instruction::IntToPtr: func_name = "itop"; break;
          default: break;
        }

        if ( (((opcode == Instruction::SExt)
               && (elem_sizer >= 2))
              || (opcode == Instruction::ZExt)
              || (opcode == Instruction::Trunc))
             && (elem_sizer <= 8)
             && (elem_size1 <= 8)
             && ((elem_bitsizer == 8*elem_sizer)
                 || (elem_bitsizer == 1))
             && ((elem_bitsize1 == 8*elem_size1)
                 || (elem_bitsize1 == 1))
             && (num_elems * elem_sizer <= 64)
             && (num_elems * elem_size1 <= 64)
             && isFastLibCall( &O, __FILE__, __LINE__) )
        {
            snprintf( sf, 256, "__lccrt_builtin_%s_v%di%di%d",
                      func_name.c_str(), num_elems, elem_bitsize1, elem_bitsizer);
            makeLibCallFast( sf, O, res, i);

        } else if ( (opcode == Instruction::FPToSI)
                    && (elem_bitsizer == elem_bitsize1)
                    && (elem_bitsize1 == 32)
                    && ((size1 == 8)
                        || (size1 == 16))
                    && isFastLibCall( &O, __FILE__, __LINE__) )
        {
            snprintf( sf, 256, "__lccrt_builtin_%s_v%df%di%d",
                      func_name.c_str(), num_elems, elem_bitsize1, elem_bitsizer);
            makeLibCallFast( sf, O, res, i);

        } else if ( ((opcode == Instruction::UIToFP)
                     || (opcode == Instruction::SIToFP))
                    && ((elem_bitsize1 < elem_bitsizer)
                        || ((opcode == Instruction::SIToFP)
                            && (elem_bitsize1 == elem_bitsizer)))
                    && ((elem_bitsize1 == 8)
                        || (elem_bitsize1 == 16)
                        || (elem_bitsize1 == 32)
                        || (elem_bitsize1 == 64))
                    && ((elem_bitsizer == 32)
                        || (elem_bitsizer == 64))
                    && ((num_elems == 2)
                        || (num_elems == 4)
                        || (num_elems == 8)
                        || (num_elems == 16))
                    && (num_elems * elem_size1 <= 64)
                    && (num_elems * elem_sizer <= 64)
                    && isFastLibCall( &O, __FILE__, __LINE__) )
        {
            assert( (opcode == Instruction::UIToFP) || (opcode == Instruction::SIToFP));
            if ( (elem_bitsize1 < 32) )
            {
                Value *V1 = O.getOperand( 0);
                lccrt_var_ptr a1 = makeValue( V1, i);
                const char *name0 = (opcode == Instruction::UIToFP) ? "zext" : "sext";
                lccrt_type_ptr ti = lccrt_type_make_int( m, elem_bitsizer/8, 0);
                lccrt_type_ptr t1 = lccrt_type_make_array( ti, num_elems);
                lccrt_var_ptr v0 = lccrt_var_new_local( f, t1, 0);

                snprintf( sf, 256, "__lccrt_builtin_%s_v%di%di%d",
                          name0, num_elems, elem_bitsize1, 32);
                makeLibCallFast( sf, a1, 0, 0, v0, i);

                snprintf( sf, 256, "__lccrt_builtin_%s_v%di%df%d",
                          "sitofp", num_elems, 32, elem_bitsizer);
                makeLibCallFast( sf, v0, 0, 0, res, i);
            } else
            {
                snprintf( sf, 256, "__lccrt_builtin_%s_v%di%df%d",
                          func_name.c_str(), num_elems, elem_bitsize1, elem_bitsizer);
                makeLibCallFast( sf, O, res, i);
            }
        } else
        {
            func_name = "__lccrt_" + func_name + "_v";
            makeLibCall( func_name.c_str(), true, O, res, false, i);
        }
    } else if ( !le.isIntBitWidthNormalOrBool( TR)
                || !le.isIntBitWidthNormalOrBool( T1) )
    {
        if ( isa<IntegerType>( T1)
             && isa<IntegerType>( TR)
             && (le.DL.getTypeAllocSize( T1) <= 8)
             && (le.DL.getTypeAllocSize( TR) <= 8)
             && (is_unsig) )
        {
            uint64_t mask = 0;
            lccrt_var_ptr a2 = 0;
            lccrt_var_ptr a1 = makeValue( V1, i);
            uint64_t bitsize1 = le.DL.getTypeSizeInBits( T1);
            uint64_t bitsizer = le.DL.getTypeSizeInBits( TR);
            lccrt_type_ptr tu32 = lccrt_type_make_u32( m);
            lccrt_type_ptr tu64 = lccrt_type_make_u64( m);
            lccrt_type_ptr tur = (bitsizer <= 32) ? tu32 : tu64;

            a1 = lccrt_oper_get_res( makeBitcastNIntToInt( bitsize1, a1, 0, i));
            if ( (bitsize1 > 32)
                 && (bitsizer <= 32) )
            {
                a1 = lccrt_oper_get_res( lccrt_oper_new_trunc( f, a1, tur, 0, i));

            } else if ( (bitsize1 <= 32)
                        && (bitsizer > 32) )
            {
                a1 = lccrt_oper_get_res( lccrt_oper_new_zext( f, a1, tur, 0, i));
            }

            if ( (opcode == Instruction::SExt) )
            {
                int align_bitsizer = (bitsizer <= 32) ? 32 : 64;

                assert( bitsize1 <= bitsizer);
                a2 = le.makeVarConstHex( tur, (align_bitsizer - bitsize1) % align_bitsizer);
                a1 = lccrt_oper_get_res( lccrt_oper_new_shl( f, a1, a2, 0, i));
                a1 = lccrt_oper_get_res( lccrt_oper_new_sar( f, a1, a2, 0, i));
                mask = (~0ULL) >> (64ULL - bitsizer);

            } else if ( (opcode == Instruction::ZExt)
                        || (opcode == Instruction::Trunc) )
            {
                mask = (~0ULL) >> (64ULL - std::min( bitsize1, bitsizer));
            } else
            {
                assert( 0);
            }

            a2 = le.makeVarConstHex( tur, mask);
            a1 = lccrt_oper_get_res( lccrt_oper_new_and( f, a1, a2, 0, i));

            makeBitcastIntToNInt( bitsizer, a1, res, i);
        } else
        {
            func_name = "__lccrt_" + func_name + "_n";
            makeLibCall( func_name.c_str(), false, CI, res, false, i);
        }
    } else
    {
        lccrt_var_ptr a = makeValue( V1, i);

        switch ( CI.getOpcode() )
        {
          case Instruction::SExt: lccrt_oper_new_sext( f, a, t, res, i); break;
          case Instruction::ZExt: lccrt_oper_new_zext( f, a, t, res, i); break;
          case Instruction::Trunc: lccrt_oper_new_trunc( f, a, t, res, i); break;
          case Instruction::BitCast: lccrt_oper_new_bitcast( f, a, t, res, i); break;
          case Instruction::PtrToInt: lccrt_oper_new_bitcast( f, a, t, res, i); break;
          case Instruction::IntToPtr: lccrt_oper_new_bitcast( f, a, t, res, i); break;
          case Instruction::FPExt: lccrt_oper_new_fptofp( f, a, t, res, i); break;
          case Instruction::FPTrunc: lccrt_oper_new_fptofp( f, a, t, res, i); break;
          case Instruction::FPToUI: lccrt_oper_new_fptoui( f, a, t, res, i); break;
          case Instruction::FPToSI: lccrt_oper_new_fptosi( f, a, t, res, i); break;
          case Instruction::UIToFP: lccrt_oper_new_uitofp( f, a, t, res, i); break;
          case Instruction::SIToFP: lccrt_oper_new_sitofp( f, a, t, res, i); break;
          default:
            errorDump( &CI);
            break;
        }
    }

    return;
} /* LccrtFunctionEmitter::makeCast */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeExtractvalue( const Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i)
{
    unsigned j = 0;
    const ExtractValueInst &EV = cast<ExtractValueInst>(O);
    unsigned num_args = 1 + EV.getNumIndices();
    lccrt_var_ptr *args = new lccrt_var_ptr[num_args];
    lccrt_type_ptr tu64 = lccrt_type_make_u64( m);
    //lccrt_type_ptr t = le.makeType( EV.getOperand( 0)->getType());

    //t = lccrt_type_make_ptr_type( t);
    args[0] = makeValue( EV.getOperand( 0), i);
    //args[0] = le.makeVarConst( t, lccrt_varinit_new_addr_var( args[0]));
    for ( const unsigned *k = EV.idx_begin(), *ke = EV.idx_end(); k != ke; ++k, ++j )
    {
        args[j+1] = le.makeVarConst( tu64, lccrt_varinit_new_scalar( tu64, *k));
    }

    lccrt_oper_new_elemread( f, num_args, args, res, i);
    delete[] args;

    return;
} /* LccrtFunctionEmitter::makeExtractvalue */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeInsertvalue( const Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i)
{
    unsigned j = 0;
    const InsertValueInst &IV = cast<InsertValueInst>(O);
    unsigned num_args = 2 + IV.getNumIndices();
    lccrt_var_ptr *args = new lccrt_var_ptr[num_args];
    lccrt_type_ptr tu64 = lccrt_type_make_u64( m);
    //lccrt_type_ptr t = le.makeType( IV.getOperand( 0)->getType());
    lccrt_type_ptr tval = 0;
    lccrt_type_ptr tvp = lccrt_type_make_pvoid( m);

    args[0] = makeValue( IV.getOperand( 0), i);
    if ( (args[0] != res) )
    {
        lccrt_oper_new_move( f, args[0], res, i);
        args[0] = lccrt_oper_get_res( lccrt_oper_iterator_get_prev( i));
    }

    //t = lccrt_type_make_ptr_type( t);
    //args[0] = le.makeVarConst( t, lccrt_varinit_new_addr_var( args[0]));
    args[1] = makeValue( IV.getOperand( 1), i);
    tval = lccrt_var_get_type( args[1]);
    if ( lccrt_type_is_pointer( tval) && (tval != tvp) ) {
        args[1] = lccrt_oper_get_res( lccrt_oper_new_bitcast( f, args[1], tvp, 0, i));
    }

    for ( const unsigned *k = IV.idx_begin(), *ke = IV.idx_end(); k != ke; ++k, ++j )
    {
        args[j+2] = le.makeVarConst( tu64, lccrt_varinit_new_scalar( tu64, *k));
    }

    lccrt_oper_new_elemwrite( f, num_args, args, i);
    delete[] args;

    return;
} /* LccrtFunctionEmitter::makeInsertvalue */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeExtractelement( Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i)
{
    ExtractElementInst &EE = cast<ExtractElementInst>(O);
    lccrt_var_ptr args[2] = {0};

    args[0] = makeValue( EE.getVectorOperand(), i);
    args[1] = makeValue( EE.getIndexOperand(), i);

    lccrt_oper_new_elemread( f, 2, args, res, i);

    return;
} /* LccrtFunctionEmitter::makeExtractelement */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeInsertelement( const Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i)
{
    const InsertElementInst &IE = cast<InsertElementInst>(O);
    lccrt_var_ptr args[3] = {0};
    lccrt_type_ptr te = 0;
    Type *T0 = IE.getOperand( 0)->getType();

    if ( auto TV = dyn_cast<FixedVectorType>( T0) ) {
        te = le.makeType( TV->getElementType());
    } else {
        errorDump( &O);
    }

    lccrt_oper_new_move( f, makeValue( IE.getOperand( 0), i), res, i);

    args[0] = lccrt_oper_get_res( lccrt_oper_iterator_get_prev( i));
    args[1] = makeValuePtrcast( IE.getOperand( 1), te, i);
    args[2] = makeValue( IE.getOperand( 2), i);

    lccrt_oper_new_elemwrite( f, 3, args, i);

    return;
} /* LccrtFunctionEmitter::makeInsertelement */

/**
 * Создание структуры для преобразования типов с удвоением.
 */
lccrt_type_ptr
LccrtFunctionEmitter::makeDoublingStruct( lccrt_type_ptr type_long, lccrt_type_ptr type_short)
{
    lccrt_type_ptr r;
    lccrt_type_ptr pair_elems[2];
    lccrt_type_ptr r_elems[2];
    lccrt_type_ptr type_pair;
    int64_t align_long = lccrt_type_get_bytealign( type_long);
    int64_t align_short = lccrt_type_get_bytealign( type_short);
    int64_t bytes_long = lccrt_type_get_bytesize( type_long);
    int64_t bytes_short = lccrt_type_get_bytesize( type_short);

    assert( bytes_long == 2*bytes_short);

    pair_elems[0] = lccrt_type_make_field( type_short, align_short, 0*bytes_short, 0, 8*bytes_short);
    pair_elems[1] = lccrt_type_make_field( type_short, align_short, 1*bytes_short, 0, 8*bytes_short);
    type_pair = lccrt_type_make_struct( m, align_long, bytes_long, 2, pair_elems, 0);

    r_elems[0] = lccrt_type_make_field( type_long, align_long, 0, 0, 8*bytes_long);
    r_elems[1] = lccrt_type_make_field( type_pair, align_long, 0, 0, 8*bytes_long);
    r = lccrt_type_make_struct( m, align_long, bytes_long, 2, r_elems, 1);

    return (r);
} /* LccrtFunctionEmitter::makeDoublingStruct */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeShufflevector( const Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i)
{
    SmallVector<int, 16> mask;
    int mask_len = 0;
    lccrt_varinit_ptr *vi_args = 0;
    lccrt_type_ptr t = 0;
    lccrt_var_ptr vm = 0;
    lccrt_varinit_ptr vi = 0;
    bool is_opt = false;
    const ShuffleVectorInst &SV = cast<ShuffleVectorInst>(O);
    lccrt_type_ptr tu32 = lccrt_type_make_u32( m);
    lccrt_type_ptr tu64 = lccrt_type_make_u64( m);
    lccrt_var_ptr va = makeValue( SV.getOperand( 0), i);
    lccrt_var_ptr vb = makeValue( SV.getOperand( 1), i);
    int va_bytes = (lccrt_var_get_bytesize( va) + 7) / 8;
    int vb_bytes = (lccrt_var_get_bytesize( vb) + 7) / 8;
    lccrt_type_ptr ta = lccrt_var_get_type( va);
    int ta_len = lccrt_type_get_num_args( ta);
    lccrt_type_ptr tr = lccrt_var_get_type( res);
    Type *T0 = SV.getOperand( 0)->getType();
    FixedVectorType *TV0 = static_cast<FixedVectorType *>( T0);
    Type *TVE0 = TV0->getElementType();
    int num_elems = TV0->getNumElements();
    int elem_size = le.DL.getTypeAllocSize( TVE0);
    char tsym = TVE0->isIntegerTy() ? 'i' : 'f';

    SV.getShuffleMask( mask);
    mask_len = mask.size();
    vi_args = new lccrt_varinit_ptr[mask_len];
   
    memset( vi_args, 0, mask_len*sizeof( vi_args[0]));

    assert( lccrt_type_is_array( ta) && (ta == lccrt_var_get_type( vb)));
    if ( !is_opt
         && (mask_len == ta_len)
         && (mask_len == (int)get_floor2( mask_len)) )
    {
        is_opt = true;
        t = lccrt_type_make_array( tu32, mask_len);
        for ( int j = 0; j < mask_len; ++j )
        {
            int k = SV.getMaskValue( j);

            vi_args[j] = lccrt_varinit_new_scalar( tu32, (k >= 0) ? k : 0);
        }

        vi = lccrt_varinit_new_array( t, mask_len, vi_args);
        vm = le.makeVarConst( t, vi);

        lccrt_oper_new_shuffle( f, va, vb, vm, res, i);
    }

    if ( is_opt || !isFastLibCall( &O, __FILE__, __LINE__) ) {
        ;
    } else if ( (va_bytes <= 16) && (vb_bytes <= 16) && (mask_len <= 8) )
    {
        char sf[256];
        uint64_t w = 0;

        is_opt = true;
        for ( int j = mask_len - 1; j >= 0; --j ) {
            w = (w << 8) | (uint8_t)SV.getMaskValue( j);
        }

        if ( (mask_len == 4)
             && (num_elems == 3)
             && (w == 0xff020100) )
        {
            snprintf( sf, 256, "__lccrt_builtin_veczxt_v%dv%d%c%d",
                      mask_len, num_elems, tsym, 8*elem_size);
            makeLibCallFast( sf, va, vb, 0, res, i);
        } else {
            snprintf( sf, 256, "__lccrt_builtin_shuffle_v%dv%d%c%d",
                      mask_len, num_elems, tsym, 8*elem_size);
            makeLibCallFast( sf, va, vb, le.makeVarConstHex( tu64, w), res, i);
        }
    } else if ( (va_bytes + vb_bytes <= 2*64)
                && (mask_len*elem_size <= 2*64)
                && (mask_len % 4 == 0)
                && (2*ta_len < 255) )
    {
        char sf[256];
        std::vector<uint8_t> vind( mask_len);
        std::vector<lccrt_var_ptr> vargs;

        is_opt = true;
        snprintf( sf, 256, "__lccrt_builtin_shuffle_v%dv%d%c%d",
                  mask_len, num_elems, tsym, 8*elem_size);

        for ( int j = 0; j < mask_len; ++j ) {
            int k = SV.getMaskValue( j);

            k = ((k >= 0) && (k < 2*ta_len)) ? k : 0xff;
            vind[j] = k;
        }

        vargs.push_back( va);
        vargs.push_back( vb);
        for ( int j = 0; j < mask_len; j += 8 ) {
            int l;
            uint64_t xj = 0;

            for ( l = 0; (l < 8) && (j + l < mask_len); ++l ) {
                xj = xj | (((uint64_t)vind[j + l]) << 8*l) ;
            }

            if ( l <= 5 ) {
                vargs.push_back( le.makeVarConstHex( tu32, xj));
            } else {
                vargs.push_back( le.makeVarConstHex( tu64, xj));
            }
        }

        makeLibCallFast( sf, vargs, res, i);
    }

    if ( !is_opt
         && (2*mask_len == ta_len)
         && (mask_len == (int)get_floor2( mask_len)) )
    {
        int j;
        int mlo = 2*ta_len - 1;
        int mhi = 0;

        for ( j = 0; j < mask_len; ++j ) {
            int mj = SV.getMaskValue( j);

            if ( (mj < 0) ) {
            } else {
                mlo = std::min( mlo, mj);
                mhi = std::max( mhi, mj);
            }
        }

        if ( (mlo <= mhi)
             && ((mhi < ta_len)
                 || (mlo >= ta_len)) )
        {
            struct { lccrt_var_ptr v[4]; } args;
            lccrt_var_ptr vx = (mhi < ta_len) ? va : vb;
            int k0 = (mhi < ta_len) ? 0 : ta_len;
            lccrt_var_ptr cv0 = le.makeVarConstHex( tu32, 0);
            lccrt_var_ptr cv1 = le.makeVarConstHex( tu32, 1);
            lccrt_type_ptr loc_type = makeDoublingStruct( ta, tr);
            lccrt_var_ptr loc = lccrt_var_new_local( f, loc_type, 0);
            lccrt_var_ptr vlo = lccrt_var_new_local( f, tr, 0);
            lccrt_var_ptr vhi = lccrt_var_new_local( f, tr, 0);

            is_opt = true;

            args = {{loc, vx, cv0}};
            lccrt_oper_new_elemwrite( f, 3, args.v, i);

            args = {{loc, cv1, cv0}};
            lccrt_oper_new_elemread( f, 3, args.v, vlo, i);

            args = {{loc, cv1, cv1}};
            lccrt_oper_new_elemread( f, 3, args.v, vhi, i);

            t = lccrt_type_make_array( tu32, mask_len);
            for ( int j = 0; j < mask_len; ++j )
            {
                int k = SV.getMaskValue( j) - k0;

                k = ((k >= 0) && (k < ta_len)) ? k : 0;
                vi_args[j] = lccrt_varinit_new_scalar( tu32, k);
            }

            vi = lccrt_varinit_new_array( t, mask_len, vi_args);
            vm = le.makeVarConst( t, vi);

            lccrt_oper_new_shuffle( f, vlo, vhi, vm, res, i);
        }
    }

    if ( !is_opt
         && (mask_len == 2*ta_len)
         && (mask_len == (int)get_floor2( mask_len)) )
    {
        int j;

        for ( j = 0; j < mask_len; ++j )
        {
            if ( (SV.getMaskValue( j) != j)
                 && (SV.getMaskValue( j) >= 0) )
                break;
        }

        if ( (j == mask_len) )
        {
            lccrt_var_ptr loc;
            lccrt_type_ptr ta2, tam;
            struct { lccrt_var_ptr v[4]; } args;
            lccrt_type_ptr ta2_elems[2];
            lccrt_type_ptr tam_elems[2];
            lccrt_type_ptr rty = lccrt_var_get_type( res);
            int64_t rbytes = lccrt_type_get_bytesize( rty);
            int64_t ralign = lccrt_type_get_bytealign( rty);
            int64_t abytes = rbytes/2;
            int64_t aalign = ralign/2;
            lccrt_var_ptr cv0 = le.makeVarConstHex( tu32, 0);
            lccrt_var_ptr cv1 = le.makeVarConstHex( tu32, 1);

            is_opt = true;
            ta2_elems[0] = lccrt_type_make_field( ta, aalign, 0*abytes, 0, 8*abytes);
            ta2_elems[1] = lccrt_type_make_field( ta, aalign, 1*abytes, 0, 8*abytes);
            ta2 = lccrt_type_make_struct( m, ralign, rbytes, 2, ta2_elems, 0);
            tam_elems[0] = lccrt_type_make_field( ta2, ralign, 0, 0, 8*rbytes);
            tam_elems[1] = lccrt_type_make_field( rty, ralign, 0, 0, 8*rbytes);
            tam = lccrt_type_make_struct( m, ralign, rbytes, 2, tam_elems, 1);

            loc = lccrt_var_new_local( f, tam, 0);

            args = {{loc, va, cv0, cv0}};
            lccrt_oper_new_elemwrite( f, 4, args.v, i);

            args = {{loc, vb, cv0, cv1}};
            lccrt_oper_new_elemwrite( f, 4, args.v, i);

            args = {{loc, cv1}};
            lccrt_oper_new_elemread( f, 2, args.v, res, i);
        }
    }

    delete[] vi_args;

    if ( !is_opt )
    {
        makeLibCall( "__lccrt_shuffle_n", true, O, res, false, i);
#ifndef NDEBUG
        O.dump();
#endif /* !NDEBUG */
    }

    return;
} /* LccrtFunctionEmitter::makeShufflevector */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeAlloca( Instruction &O, lccrt_v_ptr res, int is_start, lccrt_oi_ptr i)
{
    char name[256];
    lccrt_var_ptr v = 0;
    AllocaInst &AI = cast<AllocaInst>(O);
    lccrt_type_ptr te = le.makeType( AI.getAllocatedType());    
    lccrt_type_ptr tp = lccrt_type_make_ptr_type( te);

    if ( !is_start
         || !AI.getArraySize()
         || AI.isArrayAllocation() )
    {
        Value *V = AI.getArraySize();

        if ( is_start && isa<ConstantInt>( V) ) {
            lccrt_oper_ptr vptr;
            ConstantInt *C = cast<ConstantInt>( V);
            uint64_t n = C->getUniqueInteger().getLimitedValue();

            te = lccrt_type_make_array( te, n);
            snprintf( name, 256, "__llvm_lccrt_a%d", num_alloc);
            v = lccrt_var_new_local( f, te, name);
            num_alloc++;
            vptr = lccrt_oper_new_varptr( f, v, 0, i);
            lccrt_oper_new_bitcast( f, lccrt_oper_get_res( vptr), tp, res, i);
        } else {
            lccrt_var_ptr z;
            lccrt_oper_ptr acall;
            lccrt_type_ptr tx = le.makeType( V->getType());
            lccrt_var_ptr x = makeValue( V, i);
            lccrt_varinit_ptr yi = lccrt_varinit_new_scalar( tx, lccrt_type_get_bytesize( te));
            lccrt_var_ptr y = le.makeVarConst( tx, yi);

            z = lccrt_oper_get_res( lccrt_oper_new_mul( f, x, y, 0, i));
            acall = lccrt_oper_new_alloca( f, z, 0, i);
            lccrt_oper_new_bitcast( f, lccrt_oper_get_res( acall), tp, res, i);
        }
    } else {
        lccrt_var_ptr pv = lccrt_var_new_local( f, lccrt_type_make_ptr_type( te), 0);

        snprintf( name, 256, "__llvm_lccrt_a%d", num_alloc);
        num_alloc++;
        v = lccrt_var_new_local( f, te, name);
        lccrt_oper_new_varptr( f, v, pv, i);
        lccrt_oper_new_bitcast( f, pv, lccrt_var_get_type( res), res, i);
    }

    return;
} /* LccrtFunctionEmitter::makeAlloca */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeSelect( const Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i)
{
    const SelectInst &S = cast<SelectInst>(O);
    Type *T0 = S.getOperand( 0)->getType();
    Type *T1 = S.getOperand( 1)->getType();

    if ( isa<ScalableVectorType>( T0) )
    {
        errorDump( &O);

    } else if ( isa<FixedVectorType>( T0) )
    {
        char sf[256];
        FixedVectorType *TV1 = static_cast<FixedVectorType *>( T1);
        Type *TVE1 = TV1->getElementType();
        int num_elems = TV1->getNumElements();
        int elem_size = le.DL.getTypeAllocSize( TVE1);

        if ( ((TVE1->isFloatingPointTy()
               && ((elem_size == 4)
                   || (elem_size == 8))
               && ((num_elems * elem_size == 8)
                   || (num_elems * elem_size == 16)))
              || (TVE1->isIntegerTy()
                  && le.isIntBitWidthNormal( TVE1)
                  && (elem_size <= 8)
                  && ((num_elems * elem_size == 8)
                      || (num_elems * elem_size == 16)
                      || (num_elems * elem_size == 32))))
             && (elem_size >= 2)
             && isFastLibCall( &O, __FILE__, __LINE__) )
        {
            lccrt_type_ptr tye = lccrt_type_make_int( m, elem_size, 0);
            lccrt_type_ptr tyve = lccrt_type_make_array( tye, num_elems);
            lccrt_var_ptr v0 = lccrt_var_new_local( f, tyve, 0);
            lccrt_var_ptr a = makeValue( S.getOperand( 0), i);
            lccrt_var_ptr b = makeValue( S.getOperand( 1), i);
            lccrt_var_ptr c = makeValue( S.getOperand( 2), i);

            snprintf( sf, 256, "__lccrt_builtin_sext_v%di1i%d",
                      num_elems, 8*elem_size);
            makeLibCallFast( sf, a, 0, 0, v0, i);

            snprintf( sf, 256, "__lccrt_builtin_select_v%d%c%d",
                      num_elems, TVE1->isIntegerTy() ? 'i' : 'f', 8*elem_size);
            makeLibCallFast( sf, v0, b, c, res, i);
        } else {
            makeLibCall( "__lccrt_select_v", true, S, res, false, i);
        }
    } else
    {
        lccrt_oper_ptr ocast;
        lccrt_var_ptr a = makeValue( S.getOperand( 0), i);
        lccrt_var_ptr b = makeValue( S.getOperand( 1), i);
        lccrt_var_ptr c = makeValue( S.getOperand( 2), i);
        lccrt_type_ptr tb = lccrt_var_get_type( b);
        lccrt_type_ptr tc = lccrt_var_get_type( c);
        lccrt_type_ptr tr = le.makeType( S.getType());

        if ( tb != tr ) {
            assert( lccrt_type_is_pointer( tb));
            assert( lccrt_type_is_pointer( tr));
            ocast = lccrt_oper_new_bitcast( f, b, tr, 0, i);
            b = lccrt_oper_get_res( ocast);
        }

        if ( tc != tr ) {
            assert( lccrt_type_is_pointer( tc));
            assert( lccrt_type_is_pointer( tr));
            ocast = lccrt_oper_new_bitcast( f, c, tr, 0, i);
            c = lccrt_oper_get_res( ocast);
        }

        lccrt_oper_new_select( f, a, b, c, res, i);
    }

    return;
} /* LccrtFunctionEmitter::makeSelect */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeVaArg( const Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i)
{
    const VAArgInst &VA = cast<VAArgInst>(O);
    lccrt_var_ptr a = 0;
    lccrt_type_ptr t = 0;

    a = makeValue( VA.getOperand( 0), i);
    t = le.makeType( VA.getType());

    lccrt_oper_new_va_arg( f, a, t, res, i);

    return;
} /* LccrtFunctionEmitter::makeVaArg */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeFence( const Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i)
{
    lccrt_type_ptr t = 0;
    lccrt_var_ptr args[1] = {0};
    const FenceInst &FE = cast<FenceInst>(O);
    lccrt_type_ptr te[1] = {lccrt_type_make_ellipsis( m)};

    if ( (FE.getOrdering() != AtomicOrdering::SequentiallyConsistent)
         && (FE.getOrdering() != AtomicOrdering::Release)
         && (FE.getOrdering() != AtomicOrdering::Acquire)
         && (FE.getOrdering() != AtomicOrdering::AcquireRelease) )
    {
        errorDump( &FE);
    }

    t = lccrt_type_make_func( lccrt_type_make_void( m), 1, te);
    args[0] = makeCallBuiltinAddr( 0, t, "__sync_synchronize");
    lccrt_oper_new_call( f, t, 1, args, 0, 0, 0, i);

    return;
} /* LccrtFunctionEmitter::makeFence */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeCmpXchg( const Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i)
{
    lccrt_type_ptr t = 0;
    int bw = 0;
    lccrt_var_ptr args[4] = {0};
    const AtomicCmpXchgInst &CX = cast<AtomicCmpXchgInst>(O);
    lccrt_type_ptr tu = 0;
    lccrt_type_ptr te[3] = {0};
    char name[256] = {0};
    lccrt_oper_ptr oper1 = 0;
    lccrt_oper_ptr oper2 = 0;
    lccrt_oper_ptr oper3 = 0;
    lccrt_var_ptr a1 = 0;
    Type *T = CX.getCompareOperand()->getType();
    lccrt_type_ptr tu8 = lccrt_type_make_u8( m);
    lccrt_type_ptr tu64 = lccrt_type_make_u64( m);
    lccrt_type_ptr tr = le.makeType( T);

    if ( isa<PointerType>( T) ) {
        bw = 64 / 8;
    } else if ( T->isFloatTy() ) {
        bw = 32 / 8;
    } else if ( T->isDoubleTy() ) {
        bw = 64 / 8;
    } else if ( le.isIntBitWidthNormalOrBool( T) ) {
        bw = (dyn_cast<IntegerType>(T)->getBitWidth() + 7) / 8;
    } else {
        assert( 0);
    }

    tu = lccrt_type_make_int( m, bw, 0);

    args[1] = makeValue( CX.getOperand( 0), i);
    args[2] = makeValue( CX.getOperand( 1), i);
    args[3] = makeValue( CX.getOperand( 2), i);

    //te[0] = lccrt_type_make_ptr_type( le.makeType( T));
    te[0] = le.makeType( CX.getOperand( 0)->getType());
    te[1] = tu;
    te[2] = tu;
    t = lccrt_type_make_func( tu, 3, te);

    snprintf( name, 256, "%s%d", "__sync_val_compare_and_swap_", bw);

    args[0] = makeCallBuiltinAddr( 0, t, name);
    oper2 = lccrt_oper_new_bitcast( f, args[2], tu, 0, i);
    oper3 = lccrt_oper_new_bitcast( f, args[3], tu, 0, i);
    args[2] = lccrt_oper_get_res( oper2);
    args[3] = lccrt_oper_get_res( oper3);
    a1 = args[2];
    oper1 = lccrt_oper_new_call( f, t, 4, args, 0, 0, 0, i);
    oper1 = lccrt_oper_new_bitcast( f, lccrt_oper_get_res( oper1), tr, 0, i);

    args[1] = lccrt_oper_get_res( oper1);
    args[2] = a1;
    oper2 = lccrt_oper_new_cmp( f, lccrt_varinit_new_scalar( tu64, LCCRT_CMP_EQ), args[1], args[2], 0, i);

    args[0] = lccrt_oper_get_res( oper2);
    args[1] = le.makeVarConstHex( tu8, 1);
    args[2] = le.makeVarConstHex( tu8, 0);
    oper2 = lccrt_oper_new_select( f, args[0], args[1], args[2], 0, i);

    args[0] = res;
    args[1] = lccrt_oper_get_res( oper1);
    args[2] = le.makeVarConstHex( tu64, 0);
    lccrt_oper_new_elemwrite( f, 3, args, i);

    args[0] = res;
    args[1] = lccrt_oper_get_res( oper2);
    args[2] = le.makeVarConstHex( tu64, 1);
    lccrt_oper_new_elemwrite( f, 3, args, i);

    return;
} /* LccrtFunctionEmitter::makeCmpXchg */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makeAtomicrmw( const Instruction &O, lccrt_v_ptr res, lccrt_oi_ptr i)
{
    lccrt_type_ptr t = 0;
    int bw = 0;
    lccrt_var_ptr args[4] = {0};
    const AtomicRMWInst &RMW = cast<AtomicRMWInst>(O);
    lccrt_type_ptr tu = 0;
    lccrt_type_ptr te[3] = {0};
    lccrt_oper_ptr oper1 = 0;
    lccrt_oper_ptr oper2 = 0;
    char name[256] = {0};
    const char *op = 0;
    Type *T = RMW.getType();
    lccrt_type_ptr tr = le.makeType( T);

    if ( isa<PointerType>( T) ) {
        bw = 64 / 8;
    } else if ( T->isFloatTy() ) {
        bw = 32 / 8;
    } else if ( T->isDoubleTy() ) {
        bw = 64 / 8;
    } else if ( le.isIntBitWidthNormalOrBool( T) ) {
        bw = (dyn_cast<IntegerType>(T)->getBitWidth() + 7) / 8;
    } else {
        errorDump( &O);
    }

    tu = lccrt_type_make_int( m, bw, 0);

    args[1] = makeValue( RMW.getOperand( 0), i);
    args[2] = makeValue( RMW.getOperand( 1), i);

    te[0] = le.makeType( RMW.getOperand( 0)->getType());
    te[1] = tu;
    te[2] = lccrt_type_make_u32( m);
    t = lccrt_type_make_func( tu, 3, te);

    switch ( RMW.getOperation() )
    {
      case AtomicRMWInst::Add:  op = "fetch_add";  break;
      case AtomicRMWInst::Sub:  op = "fetch_sub";  break;
      case AtomicRMWInst::And:  op = "fetch_and";  break;
      case AtomicRMWInst::Or:   op = "fetch_or";   break;
      case AtomicRMWInst::Xor:  op = "fetch_xor";  break;
      case AtomicRMWInst::Nand: op = "fetch_nand"; break;
      case AtomicRMWInst::FAdd: op = "fetch_fadd"; break;
      case AtomicRMWInst::FSub: op = "fetch_fsub"; break;
      case AtomicRMWInst::Max:  op = "fetch_max";  break;
      case AtomicRMWInst::Min:  op = "fetch_min";  break;
      case AtomicRMWInst::UMax: op = "fetch_umax"; break;
      case AtomicRMWInst::UMin: op = "fetch_umin"; break;
      case AtomicRMWInst::Xchg: op = "exchange";   break;
      default:
        errorDump( &O);
        break;
    }

    snprintf( name, 256, "%s%s_%d", "__atomic_", op, bw);

    args[0] = makeCallBuiltinAddr( 0, t, name);
    oper2 = lccrt_oper_new_bitcast( f, args[2], tu, 0, i);
    args[2] = lccrt_oper_get_res( oper2);
    args[3] = le.makeVarConst( te[2], lccrt_varinit_new_scalar( te[2], 0));
    oper1 = lccrt_oper_new_call( f, t, 4, args, 0, 0, res, i);
    oper1 = lccrt_oper_new_bitcast( f, lccrt_oper_get_res( oper1), tr, 0, i);

    return;
} /* LccrtFunctionEmitter::makeAtomicrmw */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::semiUnzipLandingpad( const BasicBlock *BB0, MapInstToArgs &oas, lccrt_o_ptr lbl_work, lccrt_oi_ptr i)
{
    lccrt_oper_ptr lpad = lbls.find( BB0)->second;
    lccrt_oper_ptr lpga = lccrt_oper_get_next( lpad);
    const Instruction *LP = BB0->getFirstNonPHI();

    assert( BB0->isLandingPad() && isa<LandingPadInst>( LP) && lccrt_oper_is_landingpad( lpga));

    for ( auto p = pred_begin(BB0), pe = pred_end(BB0); p != pe; ++p )
    {
        lccrt_o_ptr new_br, new_label;
        const BasicBlock *BB = *p;
        assert( isa<InvokeInst>( *(BB->getTerminator())));
        const InvokeInst &II = cast<InvokeInst>( *(BB->getTerminator()));
        MapInstToArgs::const_iterator l = oas.find( &II);

        /* Считаем, что на landingpad-участок можно попасть только через unwind-дугу
           операции invoke. */
        assert( (BB0 == II.getUnwindDest()) && (BB0 != II.getNormalDest()));
        assert( l != oas.end());
        arg_ref_t *arefs = l->second;
        assert( lccrt_oper_is_invoke( arefs[0].oper) && (arefs[0].oper == arefs[1].oper));
        assert( (arefs[0].arg_num == 0) && (arefs[1].arg_num == 1));

        /* Вставляем новый lpad-участок на входящую дугу. */
        lccrt_oper_iterator_set_prev( i, lbl_work);
        new_label = lccrt_oper_new_label( f, 0, i);
        makeLandingpad( *LP, lccrt_oper_get_res( lpga), i);
        new_br = lccrt_oper_new_branch( f, lpad, i);

        /* Перенаправляем unwind-дугу на новый lpad-участок. */
        lccrt_oper_invoke_set_unwind( arefs[1].oper, new_label);

        /* Обновляем ссылки на аргумент операции перехода. */
        arefs[1].oper = new_br;
        arefs[1].arg_num = 0;
    }

    lccrt_oper_delete( lpga);

    return;
} /* LccrtEmitter::semiUnzipLandingpad */

/**
 * Создание операции.
 */
void
LccrtFunctionEmitter::makePhi( Instruction &O, MapInstToArgs &oas, lccrt_o_ptr lbl_work,
                               lccrt_v_ptr res, lccrt_oi_ptr i)
{
    MapBBToOper pib;
    const BasicBlock *BB0 = O.getParent();
    const PHINode &FI = cast<PHINode>( O);
    lccrt_var_ptr next_res = 0;
    const Value *V = dyn_cast<Value>( &FI);
    lccrt_type_ptr t = le.makeType( V->getType());

    if ( le.verbose_ir ) {
        errs() << "  -----\n" << "  MATERIALIZE PHINODE " << O << "\n";
        errs().flush();
    }

    /* Создаем для текущего фи-узла промежуточную переменную для сведения результатов. */
    next_res = lccrt_var_new_local( f, t, 0);

    /* Для каждой альтернативы делаем соответствующую записи в промежуточную переменную. */
    for ( unsigned k = 0; k < FI.getNumOperands(); ++k )
    {
        unsigned j;
        BasicBlock *BB = FI.getIncomingBlock( k);
        Value *VK = FI.getOperand( k);
        Instruction &TI = *(BB->getTerminator());
        MapInstToArgs::const_iterator l = oas.find( &TI);

        if ( (l == oas.end()) )
        {
            /* Для источника альтернативы не смогли найти массив указателей на аргументы,
               чего быть не должно! */
            errorDump2( &TI, &FI);
        }

        if ( (pib.find( BB) != pib.end()) )
        {
            /* Данный источник альтернативы уже обработан. */                
            continue;
        }

        const arg_ref_t *arefs = l->second;

        /* Для каждой входящей дуги из текущего источника вставляем новые узлы, в которых
           записываем входное значение в промежуточную переменную. */
        if ( isa<BranchInst>( TI) )
        {
            assert( !BB0->isLandingPad());
            for ( j = 0; j < TI.getNumSuccessors(); ++j )
            {
                assert( lccrt_oper_is_branch( arefs[j].oper));
                assert( (arefs[0].oper == arefs[j].oper));
                //assert( (j == arefs[0].arg_num));
                if ( (TI.getSuccessor( j) == BB0) )
                {
                    makePhiArg( lbl_work, VK, next_res, arefs[j].oper, arefs[j].arg_num, i);
                }
            }
        } else if ( isa<SwitchInst>( TI) )
        {
            assert( !BB0->isLandingPad());
            const SwitchInst &SI = cast<SwitchInst>( TI);

            j = 0;
            for ( SwitchInst::ConstCaseIt k = SI.case_begin(), ke = SI.case_end(); k != ke; ++k, ++j )
            {
                assert( lccrt_oper_is_branch( arefs[j].oper) || lccrt_oper_is_switch( arefs[j].oper));
                assert( lccrt_oper_is_branch( arefs[j].oper) || (2 + 2*j == arefs[j].arg_num));
                if ( (k->getCaseSuccessor() == BB0) )
                {
                    makePhiArg( lbl_work, VK, next_res, arefs[j].oper, arefs[j].arg_num, i);
                }
            }

            if ( (SI.getDefaultDest() == BB0) )
            {
                assert( lccrt_oper_is_branch( arefs[j].oper) || lccrt_oper_is_switch( arefs[j].oper));
                assert( lccrt_oper_is_branch( arefs[j].oper) || (2 + 2*j == arefs[j].arg_num));
                makePhiArg( lbl_work, VK, next_res, arefs[j].oper, arefs[j].arg_num, i);
            }
        } else if ( isa<InvokeInst>( TI) )
        {
#ifndef NDEBUG
            const InvokeInst &II = cast<InvokeInst>( TI);
#endif /* !NDEBUG */

            assert( lccrt_oper_is_invoke( arefs[0].oper) && (arefs[0].arg_num == 0));
            if ( BB0->isLandingPad() )
            {
                assert( lccrt_oper_is_branch( arefs[1].oper) && (arefs[1].arg_num == 0));
                assert( (II.getNormalDest() != BB0) && (II.getUnwindDest() == BB0));
                makePhiArg( lbl_work, VK, next_res, arefs[1].oper, arefs[1].arg_num, i);
            } else
            {
                //assert( (arefs[0].oper == arefs[1].oper) && (arefs[1].arg_num == 1));
                assert( (II.getNormalDest() == BB0) && (II.getUnwindDest() != BB0));
                makePhiArg( lbl_work, VK, next_res, arefs[0].oper, arefs[0].arg_num, i);
            }
        } else if ( auto IB = dyn_cast<IndirectBrInst>( &TI) )
        {
            j = 0;
            for ( unsigned si = 0; si < IB->getNumDestinations(); ++si, ++j )
            {
                assert( lccrt_oper_is_switch( arefs[j].oper) && (2 + 2*j == arefs[j].arg_num));
                if ( (IB->getDestination( si) == BB0) )
                {
                    makePhiArg( lbl_work, VK, next_res, arefs[j].oper, arefs[j].arg_num, i);
                }
            }
        } else
        {
            /* Неожиданный тип операции перехода. */
            errorDump2( &TI, &FI);
        }

        /* Помечаем текущий источник как уже обработанный. */
        pib.insert( PairBBToOper( BB, 0));
    }

    if ( le.verbose_ir ) {
        errs() << "  MATERIALIZE PHINODE RES\n";
        errs().flush();
    }

    /* Копируем промежуточную переменную в итоговую. */
    lccrt_oper_iterator_set( i, lbls.find( BB0)->second);
    lccrt_oper_new_move( f, next_res, res, i);

    return;
} /* LccrtFunctionEmitter::makePhi */

/**
 * Создание аргумента фи-узла с вставкой нового линейного участка на дугу.
 */
void
LccrtFunctionEmitter::makePhiArg( lccrt_o_ptr lbl_work, Value *PV, lccrt_v_ptr next_res,
                                  lccrt_o_ptr ct, int ct_num, lccrt_oi_ptr i)
{
    lccrt_oper_ptr old_label, new_label;
    lccrt_var_ptr value;
    lccrt_type_ptr res_type = lccrt_var_get_type( next_res);

    if ( le.verbose_ir ) {
        errs() << "  MATERIALIZE PHINODE ARG " << *PV << "\n";
        errs().flush();
    }

    if ( lccrt_oper_is_invoke( ct) )
    {
        assert( ct_num == 0);
        old_label = lccrt_oper_invoke_get_normal( ct);
    } else
    {
        old_label = lccrt_oper_get_arg_oper( ct, ct_num);
    }

    lccrt_oper_iterator_set_prev( i, lbl_work);
    new_label = lccrt_oper_new_label( f, 0, i);
    value = makeValue( PV, i);
    if ( lccrt_var_get_type( value) != res_type ) {
        lccrt_oper_new_bitcast( f, value, res_type, next_res, i);
    } else {
        lccrt_oper_new_move( f, value, next_res, i);
    }

    lccrt_oper_new_branch( f, old_label, i);

    if ( lccrt_oper_is_invoke( ct) )
    {
        assert( ct_num == 0);
        lccrt_oper_invoke_set_normal( ct, new_label);
    } else
    {
        lccrt_oper_set_arg_oper( ct, ct_num, new_label);
    }

    return;
} /* LccrtFunctionEmitter::makePhiArg */
#endif /* LLVM_WITH_LCCRT */
