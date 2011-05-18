/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file bc_cfunc.h
 ************************************************************************
 * Description:
 *  function scope, bytecode compiler
 ************************************************************************
 * Copyright(c) 2004~2005  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef BC_CFUNC_H
#define BC_CFUNC_H

#if !defined(__sun) && (!defined(_MSC_VER) || _MSC_VER > 1200) && !(defined(__xlC__) || defined(__xlc__))
//extern "C" {
#ifdef __CINT__
#include "../G__ci.h"
#else
#include "common.h"
#endif
//}
#else
#include "G__ci.h"
#include "common.h"
#endif

#include "bc_parse.h"

/////////////////////////////////////////////////////////////////////////
extern "C" int G__bc_compile_function(struct G__ifunc_table_internal *ifunc,int iexist);

/***********************************************************************
* G__functionscope
***********************************************************************/
class G__functionscope : public G__blockscope {

  static int sm_tagdefining;

  ////////////////////////////////////////////////////////////////////
  // function ID, moved to G__blockscope
  ////////////////////////////////////////////////////////////////////
  //struct G__ifunc_table *m_ifunc;
  //int m_iexist;

  ////////////////////////////////////////////////////////////////////
  // file position buffer to restore orignial file pos
  ////////////////////////////////////////////////////////////////////
  G__fstream store_fpos;

  ////////////////////////////////////////////////////////////////////
  // jump table
  ////////////////////////////////////////////////////////////////////
  G__gototable m_gototable;

  ////////////////////////////////////////////////////////////////////
  // global/member function execution environment
  ////////////////////////////////////////////////////////////////////
  int store_exec_memberfunc;
  int store_memberfunc_tagnum;
  long store_memberfunc_struct_offset;

  ////////////////////////////////////////////////////////////////////
  // others
  ////////////////////////////////////////////////////////////////////
  long store_struct_offset;

  int store_tagnum;
  int store_def_tagnum;

  int store_typenum;

  int store_def_struct_member;


  ////////////////////////////////////////////////////////////////////
  // in G__compile_bytecode
  ////////////////////////////////////////////////////////////////////
  //struct G__param para; /* This one is only dummy */
  struct G__input_file store_ifile;
  int store_prerun;
  int store_asm_index;
  int store_no_exec;
  int store_asm_exec;
  int store_tagdefining;
  int store_asm_noverflow;
  int store_asm_wholefunction;
  //int funcstatus;
  long store_globalvarpointer;
  //char funcname[G__ONELINE];

  ////////////////////////////////////////////////////////////////////
  // in G__interpret_func
  ////////////////////////////////////////////////////////////////////
  int store_no_exec_compile;

  // in G__interpret_func
#ifndef __CINT__
  G__value asm_stack_g[G__MAXSTACK]; /* data stack */
  char *asm_name; //[G__ASM_FUNCNAMEBUF];
#endif

  long *store_asm_inst;
  int store_asm_instsize;
  G__value *store_asm_stack;
  char *store_asm_name;
  int store_asm_name_p;
  struct G__param *store_asm_param;
  int store_asm_cp;
  int store_asm_dt;
  int store_func_now;
  int store_func_page;

  ////////////////////////////////////////////////////////////////////
 public:
   G__functionscope():
      store_exec_memberfunc(-1),
      store_memberfunc_tagnum(-1),
      store_memberfunc_struct_offset(-1),
      store_struct_offset(-1),
      store_tagnum(-1),
      store_def_tagnum(-1),
      store_typenum(-1),
      store_def_struct_member(-1),
      store_prerun(-1),
      store_asm_index(-1),
      store_no_exec(-1),
      store_asm_exec(-1),
      store_tagdefining(-1),
      store_asm_noverflow(-1),
      store_asm_wholefunction(-1),
      store_globalvarpointer(-1),
      store_no_exec_compile(-1),
#ifndef __CINT__
      asm_name(0),
#endif
      store_asm_inst(0),
      store_asm_instsize(-1),
      store_asm_stack(0),
      store_asm_name(0),
      store_asm_name_p(-1),
      store_asm_param(0),
      store_asm_cp(-1),
      store_asm_dt(-1),
      store_func_now(-1),
      store_func_page(-1)
   { 
      m_preader=0; 
      for(unsigned int i = 0; i < G__MAXSTACK; ++i) {
         asm_stack_g[i] = G__null;
      }
   }
  ~G__functionscope();

  int compile_normalfunction(struct G__ifunc_table_internal *ifunc,int iexist);
  int compile_implicitdefaultctor(struct G__ifunc_table_internal *ifunc,int iexist);
  int compile_implicitcopyctor(struct G__ifunc_table_internal *ifunc,int iexist);
  int compile_implicitassign(struct G__ifunc_table_internal *ifunc,int iexist);
  int compile_implicitdtor(struct G__ifunc_table_internal *ifunc,int iexist);

 private:
  void Store() ;
  void Init();
  void Restore();
  void Baseclassctor(int c);
  void Baseclasscopyctor(int c);
  void Baseclassassign(int c); 
  void Baseclassdtor();
  void ArgumentPassing();
  void EachArgumentPassing(G__TypeReader& type
			   ,const char* name,const char* def,G__value* val);
  // compile();
  int compile_function(struct G__ifunc_table_internal *ifunc,int iexist);
  void ReturnFromFunction();

  void Storefpos();
  void Setfpos();
  int  FposGetReady();
  void Restorefpos();
  void Setstatus();

  void Storebytecode();

 private:
  // setting virtual base class offset, this must be done after ctor_base
  void Baseclassctor_vbase(G__ClassInfo& cls); // This is not used here

  // Generating constructor
  int Readinitlist(map<string,string>& initlist,int c);
  void Baseclassctor_base(G__ClassInfo& cls,map<string,string>& initlist);
  void Baseclassctor_member(G__ClassInfo& cls,map<string,string>& initlist);
  void InitVirtualoffset(G__ClassInfo& cls,int tagnum,long offset);

  // Generating implicitly defined copy constructor
  void Baseclasscopyctor_base(G__ClassInfo& cls,struct G__param *libp);
  void Baseclasscopyctor_member(G__ClassInfo& cls,struct G__param *libp);

  // Generating implicitly defined operator=
  void Baseclassassign_base(G__ClassInfo& cls,struct G__param *libp);
  void Baseclassassign_member(G__ClassInfo& cls,struct G__param *libp);

  // Generating destructor
  void Baseclassdtor_base(G__ClassInfo& cls);
  void Baseclassdtor_member(G__ClassInfo& cls);
};



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
