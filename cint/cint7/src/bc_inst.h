/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file bc_inst.h
 ************************************************************************
 * Description:
 *  stack buffer for automatic object
 ************************************************************************
 * Copyright(c) 2004~2004  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef BC_INST_H
#define BC_INST_H

#include "G__ci.h"
#include "common.h"
#include "Api.h"

#include <map>

/***********************************************************************
 * C wrapper to optimize G__LD_IFUNC
 ***********************************************************************/
extern "C" int G__LD_IFUNC_optimize(struct G__ifunc_table* ifunc,int ifn
                                   ,long *inst,int pc);

namespace Cint {
   namespace Bytecode {
   
/***********************************************************************
 * G__bc_inst
 ***********************************************************************/
class G__bc_inst {
#ifdef G__NOTYET
  // not used now
  long *m_asm_inst;
  G__value *m_asm_stack;
  int  *m_pasm_cp;
  int  *m_pasm_dt;
#endif

public:

#ifdef G__NOTYET
  G__bc_inst& operator=(const G__bc_inst& x) {
    // not used now
    m_asm_inst = x.m_asm_inst;
    m_asm_stack = x.m_asm_stack;
    m_pasm_cp = x.m_pasm_cp;
    m_pasm_dt = x.m_pasm_dt;
#else 
  G__bc_inst& operator=(const G__bc_inst& ) {
#endif
    return(*this);
  }

#ifdef G__NOTYET
  void setenv(long* asm_inst,int* pasm_cp,G__value* asm_stack,int* pasm_dt) {
    // not used now
    m_asm_inst = asm_inst;
    m_asm_stack = asm_stack;
    m_pasm_cp = pasm_cp;
    m_pasm_dt = pasm_dt;
#else 
    void setenv(long* ,int* ,G__value* ,int* ) {
#endif
  }
  int inc_cp_asm(int pcinc,int dtinc) ;
  void rewind(int pcin) { ::Cint::Internal::G__asm_cp=pcin; }
  int GetPC() const { return(::Cint::Internal::G__asm_cp); }
  int GetDT() const { return(::Cint::Internal::G__asm_dt); }

  // direct asccess
  long GetInstRel(int rpc) { return(::Cint::Internal::G__asm_inst[::Cint::Internal::G__asm_cp+rpc]); }
  long GetInst(int pc) { return(::Cint::Internal::G__asm_inst[pc]); }
  void Assign(int pc,long val) { ::Cint::Internal::G__asm_inst[pc] = val; }
  long& operator[](int pc) { return(::Cint::Internal::G__asm_inst[pc]); }

  // optimizer
  void optimizeloop(int start,int end); 
  void optimize(int start,int end); 

  // instruction
  void LD(G__value* pval);
  void LD(int a);
  void CL(void);
  void OP2(int opr);
  int CNDJMP(int addr=0);
  int JMP(int addr=0);
  void POP(void);
  void LD_FUNC(const char* fname,int hash,int paran,void* pfunc);
  void LD_FUNC_BC(const Reflex::Member& var,int paran,void *pfunc);
  void LD_FUNC_VIRTUAL(const Reflex::Member& var,int paran,void *pfunc);
  void RETURN(void);
  void CAST(const Reflex::Type& type);
  void CAST(G__TypeInfo& x);
  void OP1(int opr);
  void LETVVAL(void);
  void ADDSTROS(int os);
  void LETPVAL(void);
  void TOPNTR(void);
  void NOT(void);
  void BOOL(void);
  int ISDEFAULTPARA(int addr=0);
  void LD_VAR(const Reflex::Member& var,int paran,int var_type);
  void ST_VAR(const Reflex::Member& var,int paran,int var_type);
  void LD_MSTR(const Reflex::Member& var,int paran,int var_type);
  void ST_MSTR(const Reflex::Member& var,int paran,int var_type);
  void LD_LVAR(const Reflex::Member& var,int paran,int var_type);
  void ST_LVAR(const Reflex::Member& var,int paran,int var_type);
  void CMP2(int operator2);
  void PUSHSTROS(void);
  void SETSTROS(void);
  void POPSTROS(void);
  void SETTEMP(void);
  void FREETEMP(void);
  void GETRSVD(const char* item);
  void REWINDSTACK(int rewind);
  int CND1JMP(int addr=0);
 private:
  void LD_IFUNC(const Reflex::Member& var,int hash,int paran,int funcmatch,int memfunc_flag);
 public:
  void NEWALLOC(int size,int isclass_array);
  void SET_NEWALLOC(int tagnum,int var_type);
  void SET_NEWALLOC(const G__TypeInfo& type);
  void DELETEFREE(int isarray);
  void SWAP();
  void BASECONV(const Reflex::Scope& formal_tagnum,int baseoffset);
  void STORETEMP(void);
  void ALLOCTEMP(const Reflex::Scope& tagnum);
  void POPTEMP(const Reflex::Scope& tagnum);
  void REORDER(int paran,int ig25);
  void LD_THIS(int var_type);
  void RTN_FUNC(int isreturn);
  void SETMEMFUNCENV(void);
  void RECMEMFUNCENV(void);
  void ADDALLOCTABLE(void);
  void DELALLOCTABLE(void);
  void BASEDESTRUCT(int tagnum,int isarray);
  void REDECL(const Reflex::Member& var);
  void TOVALUE(G__value* pbuf);
  void INIT_REF(const Reflex::Member& var,int paran,int var_type);
  void PUSHCPY(void);
  void LETNEWVAL(void);
  void SETGVP(int pushpop);
  void TOPVALUE(void);
  void CTOR_SETGVP(const Reflex::Member& var,int mode); 
  int TRY(int first_catchblock=0,int endof_catchblock=0);
  void TYPEMATCH(G__value* pbuf);
  void ALLOCEXCEPTION(int tagnum);
  void DESTROYEXCEPTION(void);
  void THROW(void);
  void CATCH(void);
  void SETARYINDEX(int newauto);
  void RESETARYINDEX(int newauto);
  void GETARYINDEX(void);

  void PAUSE();

  void NOP(void);

  // new instructions
  void ENTERSCOPE(void);
  void EXITSCOPE(void);
  void PUTAUTOOBJ(const Reflex::Member& var);
  void CASE(void* x);
  /* void SETARYCTOR(int num); */
  void MEMCPY();
  void MEMSETINT(int mode,std::map<long,long>& x);
  int JMPIFVIRTUALOBJ(int offset,int addr=0);
  void VIRTUALADDSTROS(int tagnum,struct G__inheritance* baseclass,int basen);
  void cancel_VIRTUALADDSTROS();

};

   } // namespace Bytecode
} // namespace Cint

#endif

/*
 * Local Variables:
 * c-tab-always-indent:nil
 * c-indent-level:2
 * c-continued-statement-offset:2
 * c-brace-offset:-2
 * c-brace-imaginary-offset:0
 * c-argdecl-indent:0
 * c-label-offset:-2
 * compile-command:"make -k"
 * End:
 */
