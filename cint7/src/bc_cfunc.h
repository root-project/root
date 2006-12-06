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

#include "G__ci.h"
#include "common.h"

#include "bc_parse.h"

/////////////////////////////////////////////////////////////////////////
extern "C" int G__bc_compile_function(struct G__ifunc_table *ifunc,int iexist);

namespace Cint {
   namespace Bytecode {

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
  ::ROOT::Reflex::Type store_memberfunc_tagnum;
  long store_memberfunc_struct_offset;

  ////////////////////////////////////////////////////////////////////
  // others
  ////////////////////////////////////////////////////////////////////
  long store_struct_offset;

  ::ROOT::Reflex::Type store_tagnum;
  ::ROOT::Reflex::Type store_def_tagnum;

  ::ROOT::Reflex::Type store_typenum;

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
  ::ROOT::Reflex::Type store_tagdefining;
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
  G__functionscope() { m_preader=0; }
  ~G__functionscope();

  int compile_normalfunction(struct G__ifunc_table *ifunc,int iexist);
  int compile_implicitdefaultctor(struct G__ifunc_table *ifunc,int iexist);
  int compile_implicitcopyctor(struct G__ifunc_table *ifunc,int iexist);
  int compile_implicitassign(struct G__ifunc_table *ifunc,int iexist);
  int compile_implicitdtor(struct G__ifunc_table *ifunc,int iexist);

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
  int compile_function(struct G__ifunc_table *ifunc,int iexist);
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
  int Readinitlist(std::map<std::string,std::string>& initlist,int c);
  void Baseclassctor_base(G__ClassInfo& cls,std::map<std::string,std::string>& initlist);
  void Baseclassctor_member(G__ClassInfo& cls,std::map<std::string,std::string>& initlist);
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
