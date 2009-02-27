#if 0
/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file bc_cfunc.cxx
 ************************************************************************
 * Description:
 *  function scope, bytecode compiler
 ************************************************************************
 * Copyright(c) 2004~2004  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "bc_cfunc.h"
#include "bc_exec.h"
#include "Dict.h"

using namespace ::Cint::Internal;
using namespace ::Cint::Bytecode;

/***********************************************************************
* G__bc_compile_function()
***********************************************************************/
int Cint::Bytecode::G__bc_compile_function(struct G__ifunc_table *ifunc,int iexist){
   // ignore iexist, ifunc is Reflex::Member::Id()
  G__functionscope compiler;
  int store_dispsource = G__dispsource;
  if(G__step||G__stepover) G__dispsource=0;
  int result = compiler.compile_normalfunction(G__Dict::GetDict().GetFunction(ifunc,iexist));
  G__dispsource = store_dispsource;
  return(result);
}
int Cint::Bytecode::G__bc_compile_function(const ::Reflex::Member &func){
   // ignore iexist, ifunc is Reflex::Member::Id()
  G__functionscope compiler;
  int store_dispsource = G__dispsource;
  if(G__step||G__stepover) G__dispsource=0;
  int result = compiler.compile_normalfunction(func);
  G__dispsource = store_dispsource;
  return(result);
}



/***********************************************************************
* G__functionscope
***********************************************************************/

int Cint::Bytecode::G__functionscope::sm_tagdefining = G__MAXSTRUCT;

//////////////////////////////////////////////////////////////////////////
// storing/restoring environment
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
void Cint::Bytecode::G__functionscope::Store() {
 
  // in G__compile_bytecode
  store_tagdefining = G__tagdefining;

  store_prerun = G__prerun;
  store_asm_index = G__asm_index;
  store_no_exec = G__no_exec;
  store_asm_exec = G__asm_exec;
  store_asm_noverflow = G__asm_noverflow;
  store_globalvarpointer = G__globalvarpointer;
  store_asm_wholefunction = G__asm_wholefunction;

  // in G__interpret_func
  store_no_exec_compile = G__no_exec_compile;
  store_func_now = G__func_now;

  // store bytecode buffer
  store_asm_inst = G__asm_inst;
  store_asm_instsize = G__asm_instsize;
  store_asm_stack = G__asm_stack;
  store_asm_name = G__asm_name;
  store_asm_name_p = G__asm_name_p;
  store_asm_param  = G__asm_param ;
  store_asm_exec  = G__asm_exec ;
  store_asm_noverflow  = G__asm_noverflow ;
  store_asm_cp  = G__asm_cp ;
  store_asm_dt  = G__asm_dt ;
  store_asm_index  = G__asm_index ;

  // global/member function execution environment
  store_exec_memberfunc = G__exec_memberfunc;
  store_memberfunc_tagnum = G__memberfunc_tagnum;
  store_memberfunc_struct_offset = G__memberfunc_struct_offset;

  // others
  store_struct_offset = G__store_struct_offset;
  store_tagnum = G__tagnum;
  store_def_tagnum = G__def_tagnum;
  store_typenum = G__typenum;
  store_def_struct_member = G__def_struct_member;

  Storefpos();
}

//////////////////////////////////////////////////////////////////////////
void Cint::Bytecode::G__functionscope::Init() {
  // in G__compile_bytecode.
  // this is needed in order to use G__malloc() in 
  // G__blockscope::allocatevariable
  G__tagdefining = G__Dict::GetDict().GetScope(--sm_tagdefining);
  G__struct.type[G__get_tagnum(G__tagdefining)] = 's';
  G__struct.size[G__get_tagnum(G__tagdefining)] = 0;

  G__no_exec = 0;
  G__prerun = 0;
  G__asm_exec = 1;
  G__asm_wholefunction = G__ASM_FUNC_COMPILE;
  G__globalvarpointer = G__PVOID;
  G__asm_index = m_iexist;

  // in G__interpret_func
  G__func_now=m_iexist;
  //G__func_page=m_scope->page;
  G__no_exec_compile = 1;

  G__asm_instsize = G__MAXINST;
  G__asm_inst = (long*)malloc(sizeof(long)*G__asm_instsize);
  G__asm_stack = asm_stack_g;
  asm_name = (char*)malloc(G__ASM_FUNCNAMEBUF); // freed in G__free_bytecode
  G__asm_name = asm_name;
  G__asm_name_p = 0;
  /* G__asm_param ; */
  G__asm_exec = 0 ;
  G__asm_noverflow = 1;
  G__asm_cp = 0;
  G__asm_dt = G__MAXSTACK-1;

  // global/member function execution environment
  char* dmy_offset = (char*)1; // questionable
  G__exec_memberfunc = !m_scope.DeclaringScope().IsTopScope();
  G__memberfunc_tagnum = m_scope.DeclaringScope();
  G__memberfunc_struct_offset = G__exec_memberfunc?dmy_offset:(char*)0;

  // others
  G__store_struct_offset = G__memberfunc_struct_offset;
  G__tagnum = G__memberfunc_tagnum;
  G__def_tagnum = ::Reflex::Scope();
  G__typenum = ::Reflex::Type();
  G__def_struct_member = 0;

  // initialize blockscope compiler class (base class)
  G__blockscope::Init(0);
  m_scope = G__memberfunc_tagnum;
  setgototable(&m_gototable);

}

//////////////////////////////////////////////////////////////////////////
G__functionscope::~G__functionscope() {
  Restore(); // moved from compile_function, compile_implicitctor/assign
  if(m_preader) delete m_preader; 
  if(G__asm_instsize) free((void*)G__asm_inst); // should always be true
  G__asm_instsize = store_asm_instsize;
  G__asm_inst = store_asm_inst;
  if(asm_name) free((void*)asm_name);
}

//////////////////////////////////////////////////////////////////////////
void Cint::Bytecode::G__functionscope::Restore() {

  // in G__compile_bytecode
  G__tagdefining = store_tagdefining;

  G__prerun = store_prerun;
  G__asm_index = store_asm_index;
  G__no_exec = store_no_exec;
  G__asm_exec = store_asm_exec;
  G__asm_noverflow = store_asm_noverflow;
  G__globalvarpointer = store_globalvarpointer;
  G__asm_wholefunction = store_asm_wholefunction; // ??? G__ASM_FUNC_NOP ???

  // in G__interpret_func
  G__no_exec_compile = store_no_exec_compile;
  G__func_now=store_func_now;

  /* Pop loop compilation environment */
  G__asm_stack = store_asm_stack;
  G__asm_name = store_asm_name;
  G__asm_name_p = store_asm_name_p;
  G__asm_param  = store_asm_param ;
  G__asm_exec  = store_asm_exec ;
  G__asm_noverflow  = store_asm_noverflow ;
  G__asm_cp  = store_asm_cp ;
  G__asm_dt  = store_asm_dt ;
  G__asm_index  = store_asm_index ;

  // global/member function execution environment
  G__exec_memberfunc = store_exec_memberfunc;
  G__memberfunc_tagnum = store_memberfunc_tagnum;
  G__memberfunc_struct_offset = store_memberfunc_struct_offset;

  // others
  G__store_struct_offset = store_struct_offset;
  G__tagnum = store_tagnum;
  G__def_tagnum = store_def_tagnum;
  G__typenum = store_typenum;
  G__def_struct_member = store_def_struct_member;

  --sm_tagdefining;
}

//////////////////////////////////////////////////////////////////////////
// file position handling
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
void Cint::Bytecode::G__functionscope::Storefpos() {
  store_fpos.storepos();

  store_ifile.fp = G__ifile.fp;
  store_ifile.line_number = G__ifile.line_number;
  store_ifile.filenum = G__ifile.filenum;
  strcpy(store_ifile.name,G__ifile.name);
}
//////////////////////////////////////////////////////////////////////////
void Cint::Bytecode::G__functionscope::Setfpos() {
  const G__funcentry& entry = G__get_funcproperties(m_iexist)->entry;
  G__ifile.fp = (FILE*)entry.p ;
  G__ifile.line_number = entry.line_number;
  G__ifile.filenum=entry.filenum;
  strcpy(G__ifile.name,G__srcfile[entry.filenum].filename);
  fsetpos(G__ifile.fp,&entry.pos);

  // m_reader also has m_fpos. This implementation is not so clean.
  m_preader = new G__srcreader<G__fstream>;
  m_preader->Init(G__ifile);

  // todo , following line causes SEGV ??? Need debugging
  // For now, this is done above with fsetpos()
  //m_preader->setpos(m_scope->pentry[m_iexist]->pos);
}

//////////////////////////////////////////////////////////////////////////
int Cint::Bytecode::G__functionscope::FposGetReady() {
  //       type(args)       : member(expr), base(expr) {
  // type  func(args) const {
  //            ^ ---------> ^
  int c;
  c = m_preader->fignorestream(")");
  c = m_preader->fignorestream(":{");
  //m_preader->putback();
  return(c);
}

//////////////////////////////////////////////////////////////////////////
void Cint::Bytecode::G__functionscope::Restorefpos() {
  G__ifile.fp = store_ifile.fp;
  G__ifile.line_number = store_ifile.line_number;
  G__ifile.filenum = store_ifile.filenum;
  strcpy(G__ifile.name,store_ifile.name);

  store_fpos.rewindpos();

  // m_reader also has m_fpos. This implementation is not so clean.
  m_preader->Init(G__ifile);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
// bytecode instruction generation
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// constructor and baseclass and member objects
//////////////////////////////////////////////////////////////////////////
void Cint::Bytecode::G__functionscope::Baseclassctor(int c) {
  //   type(args)  : mem(expr), base(expr) {
  // void f(args)  {
  //                ^
  if(m_iexist.IsConstructor()) {

    G__ClassInfo cls(G__get_tagnum(m_iexist.DeclaringScope()));

    if(cls.Property()&G__BIT_ISCOMPILED) {
      // error if compiled class;,  should never happen
      G__genericerror("Internal Error: trying to compile natively compiled class's constructor");
    }

    std::map<std::string,std::string> initlist;
    c = Readinitlist(initlist,c);

    // iterate on direct base classes
    Baseclassctor_base(cls,initlist);
  
    // set virtual base offset, this must be done after ctor_base
    //Baseclassctor_vbase(cls); // not here. Do this before ctor call

    // iterate on members
    Baseclassctor_member(cls,initlist);

    // initialize G__virtualtag
    InitVirtualoffset(cls,cls.Tagnum(),0);
  }
  else if(c!='{') {
    // error;
    G__genericerror("Error: Syntax error");
  }

  //   type(args)  : mem(expr), base(expr)    {
  //                                      ^ -> ^
  // if(c!='{') c = m_preader->fignorestream("{"); // this should not be needed
}

//////////////////////////////////////////////////////////////////////////
void Cint::Bytecode::G__functionscope::Baseclassctor_vbase(G__ClassInfo& cls) {
  // THIS function should not be used. Use G__blockscope::Baseclassctor_vbase()
  // generate instruction for setting virtual base offset
  //  xxVVVV        yyvvvv
  //  AAAAAAAA ???? BBBBBBBB
  //  DDDDDDDDDDDDDDDDDDDDDDDDDD
  //  |------------>| baseoffset of B. (static)
  //    |<----------| virtual base offset of B. Contents of yy (dynamic)
  G__BaseClassInfo bas(cls);
  std::map<long,long> vbasetable;
  std::map<long,long> adrtable;
  while(bas.Next(0)) { // iterate all inheritance
    if(bas.Property()&G__BIT_ISVIRTUALBASE) {

      if(0==adrtable[bas.Tagnum()]) {
        // the first appearance of virtual base object. 
        vbasetable[bas.Offset()] = G__DOUBLEALLOC;
        adrtable[bas.Tagnum()] = bas.Offset() + G__DOUBLEALLOC;
      }
      else {
        // ghost area of virtual base object
        vbasetable[bas.Offset()] = adrtable[bas.Tagnum()] - bas.Offset();
      }
    }
  }

  m_bc_inst.MEMSETINT(1,vbasetable);
}
//////////////////////////////////////////////////////////////////////////
int Cint::Bytecode::G__functionscope::Readinitlist(std::map<std::string,std::string>& initlist,int c) {
  if(c==':') {  // read initialization list
    //   type(args)  : mem(args1) , base(args2)    {
    //                ^ ---------> ^ -------------> ^
    //     initlist["mem"]="args1"
    std::string element, expr;
    while(c!='{') {
      c = m_preader->fgetstream(element,"(");
      c = m_preader->fgetstream(expr,")");
      initlist[element] = expr;
      c = m_preader->fignorestream(",{");
    }
  }
  return(c);
}
//////////////////////////////////////////////////////////////////////////
void Cint::Bytecode::G__functionscope::Baseclassctor_base(G__ClassInfo& cls
                                          ,std::map<std::string,std::string>& initlist) {
  G__BaseClassInfo bas(cls);
  struct G__param para;
  std::string args;
  G__value ctorfound;
  int pc_skip=0;

  while(bas.Next()) { // iterate only direct inheritance

    if(bas.Property()&G__BIT_ISVIRTUALBASE) {
      pc_skip = m_bc_inst.JMPIFVIRTUALOBJ(bas.Offset());
    }

    ctorfound=G__null;

    // compile initialization list for base class ctor
    args = initlist[bas.Name()];
    para.paran = 0;
    para.para[0] = G__null;
    if(args!="") {
      // evaluate initialization list if exist
      compile_arglist(args,&para); 
      initlist[bas.Name()] = "";
    }

    // try calling ctor
    int store_pc= m_bc_inst.GetPC();
    if(bas.Property()&G__BIT_ISVIRTUALBASE)
      m_bc_inst.ADDSTROS(bas.Offset()+G__DOUBLEALLOC);
    else  
      if(bas.Offset()) m_bc_inst.ADDSTROS(bas.Offset());
    if(bas.Property()&G__BIT_ISCOMPILED) m_bc_inst.SETGVP(1);
    ctorfound=call_func(bas,bas.Name(),&para,G__TRYMEMFUNC);
    if(bas.Property()&G__BIT_ISCOMPILED) m_bc_inst.SETGVP(-1);
    if(bas.Property()&G__BIT_ISVIRTUALBASE) 
      m_bc_inst.ADDSTROS(-bas.Offset()-G__DOUBLEALLOC);
    else
      if(bas.Offset()) m_bc_inst.ADDSTROS(-bas.Offset());

    if(bas.Property()&G__BIT_ISVIRTUALBASE) {
      m_bc_inst.Assign(pc_skip,m_bc_inst.GetPC());
    }

    if(!G__value_typenum(ctorfound)) {
      m_bc_inst.rewind(store_pc);
      // Because implicit default and copy ctors are always generated as
      // far as they can exist, failure to find ctor here should be an error.
      G__fprinterr(G__serr
            ,"Error: %s, base class %s does not have appropriate constructor"
                  ,cls.Name(),bas.Name());
      G__genericerror((char*)NULL);
    }
  }
}
//////////////////////////////////////////////////////////////////////////
void Cint::Bytecode::G__functionscope::Baseclassctor_member(G__ClassInfo& cls
                                            ,std::map<std::string,std::string>& initlist) {
  G__DataMemberInfo dat(cls);
  struct G__param para;
  std::string args;
  G__value ctorfound;
  while(dat.Next()) {
    ctorfound=G__null;

    // compile initlization list for data member
    args = initlist[dat.Name()];
    para.paran = 0;
    para.para[0] = G__null;
    if(args!="") {
      // evaluate initialization list if exist
      compile_arglist(args,&para); 
      initlist[dat.Name()] = "";
    }

    Reflex::Member var = G__Dict::GetDict().GetDataMember(dat.Handle());
    const Reflex::Type& varType = var.TypeOf();
    if(varType.RawType().IsClass() 
       && !(varType.IsPointer() || varType.IsReference() || var.IsStatic())) {
      int store_pc= m_bc_inst.GetPC();
      if(dat.Type()->Property()&G__BIT_ISCOMPILED) {
        // TODO, compiled class object array as member
        m_bc_inst.CTOR_SETGVP(var,1); // init local block scope object
        ctorfound=call_func(*dat.Type(),dat.Type()->TrueName(),&para,G__TRYMEMFUNC);
        m_bc_inst.SETGVP(-1); // restoration from store_globalvarpointer stack
      }
      else {
        m_bc_inst.LD_MSTR(var,0,'p');
        m_bc_inst.PUSHSTROS();
        m_bc_inst.SETSTROS();
        if(dat.ArrayDim()) {
          m_bc_inst.LD(G__get_varlabel(var, 1) /* number of elements */);
          m_bc_inst.SETARYINDEX(1); // this is illegular, but (1) does --sp
          ctorfound
            =call_func(*dat.Type(),dat.Type()->TrueName(),&para,G__TRYMEMFUNC,1);
          m_bc_inst.RESETARYINDEX(0);
        }
        else {
          ctorfound
            =call_func(*dat.Type(),dat.Type()->TrueName(),&para,G__TRYMEMFUNC,0);
        }
        m_bc_inst.POPSTROS();
      }
      if(!G__value_typenum(ctorfound)) {
        // TODO, is this an error?
        m_bc_inst.rewind(store_pc);
        // Because implicit default and copy ctors are always generated as
        // far as they can exist, failure to find ctor here should be an error.
        G__fprinterr(G__serr,"Error: %s, data member %s does not have appropriate constructor"
                     ,cls.Name(),dat.Name());
        G__genericerror((char*)NULL);
      }
    }
    if(!G__value_typenum(ctorfound) && para.paran) {
      // fundamental type initialization. 
      // Since implicit default and copy ctors are always generated, class object
      // member should not come here.
      m_bc_inst.ST_MSTR(var,0,'p');
    }    
  }
}

//////////////////////////////////////////////////////////////////////////
void Cint::Bytecode::G__functionscope::InitVirtualoffset(G__ClassInfo& cls
                                        ,int tagnum
                                        ,long offset) {
  char* voffset = G__struct.virtual_offset[cls.Tagnum()];
  if((char*)-1!=voffset) {

    // This section is needed. Virtualoffset is already set by base class ctor.
    // But it has base class's tagnum. Need to overwrite tagnum of instantiated
    // class. 
    G__BaseClassInfo bas(cls);
    while(bas.Next()) { 
      if(bas.Property()&G__BIT_ISVIRTUALBASE) 
        InitVirtualoffset(bas,tagnum,bas.Offset()+offset+G__DOUBLEALLOC);
      else 
        InitVirtualoffset(bas,tagnum,bas.Offset()+offset);
    }

    // Initialize G__virtualinfo, if the first virtual function appears in
    // this class.
    long dmy;
    G__DataMemberInfo dat = cls.GetDataMember("G__virtualinfo",&dmy);
    if(dat.IsValid()) {
      // This should be fine for constructor. In constructor, everything 
      // has to be resolved statically, even for virtual base class
      if(offset) m_bc_inst.ADDSTROS(offset);
      m_bc_inst.LD(tagnum);
      Reflex::Member var = G__Dict::GetDict().GetDataMember(dat.Handle());
      m_bc_inst.ST_MSTR(var,0,'p');
      if(offset) m_bc_inst.ADDSTROS(-offset);
    }
  }
}


//////////////////////////////////////////////////////////////////////////
// copy ctor for baseclass and member objects
//////////////////////////////////////////////////////////////////////////
void Cint::Bytecode::G__functionscope::Baseclasscopyctor(int c) {
  // void copyctor(args)  {
  //                       ^

  if(c!='{') {
    // error;  never happen
    G__genericerror("Error: Syntax error");
  }

  if(m_iexist.IsConstructor()) {

    G__ClassInfo cls(G__get_tagnum(m_iexist.DeclaringScope()));
    struct G__param para;
    for(int i=0;i<m_iexist.FunctionParameterSize();++i) {
      G__value_typenum(para.para[i])=m_iexist.TypeOf().FunctionParameterAt(i);
      para.para[i].obj.i=1;   // dummy value
      para.para[i].ref=1; // dummy value
    }
    para.paran = m_iexist.FunctionParameterSize();

    if(cls.Property()&G__BIT_ISCOMPILED) {
      // error if compiled class;,  should never happen
      G__genericerror("Internal Error: trying to compile natively compiled class's constructor");
    }

    // iterate on direct base classes
    Baseclasscopyctor_base(cls,&para);
  
    // set virtual base offset, this must be done after ctor_base
    //Baseclassctor_vbase(cls); // not here. do this before ctor call

    // iterate on members
    Baseclasscopyctor_member(cls,&para);

    // initialize G__virtualtag
    InitVirtualoffset(cls,cls.Tagnum(),0);
  }

}

//////////////////////////////////////////////////////////////////////////
void Cint::Bytecode::G__functionscope::Baseclasscopyctor_base(G__ClassInfo& cls
                                              ,struct G__param *libp) {
  G__BaseClassInfo bas(cls);
  G__value found;
  while(bas.Next()) {
    found=G__null;
    int store_pc= m_bc_inst.GetPC();

    // prepare oprand
    m_bc_inst.PUSHCPY(); // duplicate arg obj in stack
    m_bc_inst.BASECONV(G__Dict::GetDict().GetScope(bas.Tagnum()),bas.Offset()); //cast arg to base class obj

    // TODO, how to deal with compiled class member? See Baseclassctor_base()

    // try calling copy ctor
    if(bas.Offset()) m_bc_inst.ADDSTROS(bas.Offset());
    if(bas.Property()&G__BIT_ISCOMPILED) m_bc_inst.SETGVP(1);
    found=call_func(bas,bas.Name(),libp,G__TRYMEMFUNC);
    if(bas.Property()&G__BIT_ISCOMPILED) m_bc_inst.SETGVP(-1);
    if(bas.Offset()) m_bc_inst.ADDSTROS(-bas.Offset());

    // pop oprand
    m_bc_inst.POP(); // ??? need this? copy ctor returns value???

    if(!G__value_typenum(found)) {
      m_bc_inst.rewind(store_pc);
      // Since implicit copy is generated for every class, this is
      // rather an error.
      G__fprinterr(G__serr
                   ,"Error: %s, base class %s has private copy constructor"
                   ,cls.Name(),bas.Name());
      G__genericerror((char*)NULL);
    }
  }
}
//////////////////////////////////////////////////////////////////////////
void Cint::Bytecode::G__functionscope::Baseclasscopyctor_member(G__ClassInfo& cls
                                                ,struct G__param *libp) {
  G__DataMemberInfo dat(cls);
  G__value found;
  while(dat.Next()) {
    found=G__null;
    Reflex::Member var=G__Dict::GetDict().GetDataMember(dat.Handle());
    m_bc_inst.PUSHCPY();
    m_bc_inst.PUSHSTROS();
    m_bc_inst.SETSTROS();
    m_bc_inst.LD_MSTR(var,0,'p');
    m_bc_inst.POPSTROS();
    if(var.TypeOf().RawType().IsClass() 
      && !(var.TypeOf().IsPointer() || var.TypeOf().IsReference() || var.IsStatic())) {
      int store_pc= m_bc_inst.GetPC();
      // TODO, how to deal with compiled class member? See Baseclassctor_member()
      m_bc_inst.LD_MSTR(var,0,'p');
      m_bc_inst.PUSHSTROS();
      m_bc_inst.SETSTROS();
      // libp->para[0].tagnum=var->p_tagtable[ig15]; // Masa's comment was: TODO, dirty workaround
      G__value_typenum(libp->para[0])=var.TypeOf();
      if(dat.ArrayDim()) {
        m_bc_inst.LD(G__get_varlabel(var, 1) /* number of elements */);
        m_bc_inst.SETARYINDEX(1); // this is illegular, but (1) does --sp
        found=call_func(*dat.Type(),dat.Type()->TrueName(),libp,G__TRYMEMFUNC,1
                        );
        m_bc_inst.RESETARYINDEX(0);
      }
      else {
        found=call_func(*dat.Type(),dat.Type()->TrueName(),libp,G__TRYMEMFUNC,0
                        );
      }
      m_bc_inst.POPSTROS();
      if(!G__value_typenum(found)) {
        m_bc_inst.rewind(store_pc);
        // Since implicit copy ctor is generated for every class, this is
        // rather an error.
        G__fprinterr(G__serr
                     ,"Error: %s, data member %s has private copy constructor"
                     ,cls.Name(),dat.Name());
        G__genericerror((char*)NULL);
      }
    }
    if(!G__value_typenum(found)) {
      if(dat.ArrayDim()) {
        // (LD SRC), LD DEST, LD SIZE, MEMCPY
        m_bc_inst.LD_MSTR(var,0,'p');
        m_bc_inst.LD((G__get_varlabel(var, 1) /* number of elements */)*dat.Type()->Size());
        m_bc_inst.MEMCPY();
      }
      else {
        m_bc_inst.ST_MSTR(var,0,'p');
      }
    }
    m_bc_inst.POP(); // ??? need this? copy ctor returns value???
  }
}

//////////////////////////////////////////////////////////////////////////
// operator= for baseclass and member objects
//////////////////////////////////////////////////////////////////////////
void Cint::Bytecode::G__functionscope::Baseclassassign(int c) {
  // void operator=(args)  {
  //                        ^

  if(c!='{') {
    // error; never happen
    G__genericerror("Error: Syntax error");
  }

  if(!m_iexist.DeclaringScope().IsTopScope()&&m_iexist.Name() == "operator="){

    G__ClassInfo cls(G__get_tagnum(m_iexist.DeclaringScope()));
    struct G__param para;
    for(int i=0;i<m_iexist.FunctionParameterSize();++i) {
      G__value_typenum(para.para[i])=m_iexist.TypeOf().FunctionParameterAt(i);
      para.para[i].obj.i=1;   // dummy value
      para.para[i].ref=1; // dummy value
    }
    para.paran = m_iexist.TypeOf().FunctionParameterSize();

    if(cls.Property()&G__BIT_ISCOMPILED) {
      // error if compiled class;,  should never happen
      G__genericerror("Internal Error: trying to compile natively compiled class's constructor");
    }

    // iterate on direct base classes
    Baseclassassign_base(cls,&para);
  
    // iterate on members
    Baseclassassign_member(cls,&para);
  }

  m_bc_inst.LD_THIS('v');
  m_bc_inst.RTN_FUNC(1);

}

//////////////////////////////////////////////////////////////////////////
void Cint::Bytecode::G__functionscope::Baseclassassign_base(G__ClassInfo& cls
                                          ,struct G__param *libp) {
  G__BaseClassInfo bas(cls);
  G__value found;
  while(bas.Next()) {
    found=G__null;
    int store_pc= m_bc_inst.GetPC();
    // prepare oprand
    m_bc_inst.PUSHCPY();
    m_bc_inst.BASECONV(G__Dict::GetDict().GetScope(bas.Tagnum()),bas.Offset());
    // try calling assignment operator
    if(bas.Offset()) m_bc_inst.ADDSTROS(bas.Offset());
    found=call_func(bas,"operator=",libp,G__TRYMEMFUNC);
    if(bas.Offset()) m_bc_inst.ADDSTROS(-bas.Offset());
    m_bc_inst.POP();
    if(!G__value_typenum(found)) {
      m_bc_inst.rewind(store_pc);
      // Since implicit operator= is generated for every class, this is
      // rather an error.
      G__fprinterr(G__serr,"Error: %s, base class %s has private operator="
                   ,cls.Name(),bas.Name());
      G__genericerror((char*)NULL);
    }
  }
}
//////////////////////////////////////////////////////////////////////////
void Cint::Bytecode::G__functionscope::Baseclassassign_member(G__ClassInfo& cls
                                          ,struct G__param *libp) {
  G__DataMemberInfo dat(cls);
  G__value found;
  while(dat.Next()) {
    found=G__null;
    Reflex::Member var = G__Dict::GetDict().GetDataMember(dat.Handle());
    m_bc_inst.PUSHCPY();
    m_bc_inst.PUSHSTROS();
    m_bc_inst.SETSTROS();
    m_bc_inst.LD_MSTR(var,0,'p');
    m_bc_inst.POPSTROS();
    if(var.TypeOf().RawType().IsClass() 
       && !(var.TypeOf().IsPointer() || var.TypeOf().IsReference() || var.IsStatic())) {
      int store_pc= m_bc_inst.GetPC();
      m_bc_inst.LD_MSTR(var,0,'p');
      m_bc_inst.PUSHSTROS();
      m_bc_inst.SETSTROS();
      // libp->para[0].tagnum=var->p_tagtable[ig15]; // Masa says: TODO, dirty workaround
      G__value_typenum(libp->para[0])=var.TypeOf();
      if(dat.ArrayDim()) {
        m_bc_inst.LD(G__get_varlabel(var, 1) /* number of elements */);
        m_bc_inst.SETARYINDEX(1); // this is illegular, but (1) does --sp
        found=call_func(*dat.Type(),"operator=",libp,G__TRYMEMFUNC,1);
        m_bc_inst.RESETARYINDEX(0);
      }
      else {
        found=call_func(*dat.Type(),"operator=",libp,G__TRYMEMFUNC,0);
      }
      m_bc_inst.POPSTROS();
      if(!G__value_typenum(found)) {
        m_bc_inst.rewind(store_pc);
        // Since implicit operator= is generated for every class, this is
        // rather an error.
        G__fprinterr(G__serr,"Error: %s, data member %s has private operator="
                     ,cls.Name(),dat.Name());
        G__genericerror((char*)NULL);
      }
    }
    if(!G__value_typenum(found)) {
      if(dat.ArrayDim()) {
        // (LD SRC), LD DEST, LD SIZE, MEMCPY
        m_bc_inst.LD_MSTR(var,0,'p');
        m_bc_inst.LD((G__get_varlabel(var, 1) /* number of elements */)*dat.Type()->Size());
        m_bc_inst.MEMCPY();
      }
      else {
        m_bc_inst.ST_MSTR(var,0,'p');
      }
    }
    m_bc_inst.POP();
  }
}


//////////////////////////////////////////////////////////////////////////
// destructor for baseclass and member objects
//////////////////////////////////////////////////////////////////////////
void Cint::Bytecode::G__functionscope::Baseclassdtor() {
  //   type(args)  : mem(expr), base(expr) {
  // void f(args)  {
  //                ^
  if(m_iexist.IsDestructor()){
    G__ClassInfo cls(G__get_tagnum(m_scope));

    // iterate on members
    Baseclassdtor_member(cls);

    // iterate on direct base classes
    Baseclassdtor_base(cls);

  }
}

//////////////////////////////////////////////////////////////////////////
void Cint::Bytecode::G__functionscope::Baseclassdtor_base(G__ClassInfo& cls) {
  G__BaseClassInfo bas(cls);
  struct G__param para;
  std::string args;
  G__value dtorfound;
  para.paran = 0;
  para.para[0] = G__null;
  std::string fname;
  while(bas.Prev()) {
    dtorfound=G__null;
    int store_pc= m_bc_inst.GetPC();
    if(bas.Offset()) m_bc_inst.ADDSTROS(bas.Offset());
    fname = "~";
    fname.append(G__struct.name[bas.Tagnum()]);
    dtorfound=call_func(bas,fname,&para,G__TRYMEMFUNC);
    if(bas.Offset()) m_bc_inst.ADDSTROS(-bas.Offset());
    if(!G__value_typenum(dtorfound)) m_bc_inst.rewind(store_pc);
  }
}
//////////////////////////////////////////////////////////////////////////
void Cint::Bytecode::G__functionscope::Baseclassdtor_member(G__ClassInfo& cls) {
  G__DataMemberInfo dat(cls);
  struct G__param para;
  G__value dtorfound;
  std::string fname;
  para.paran = 0;
  para.para[0] = G__null;
  while(dat.Prev()) {
    dtorfound=G__null;
    if((dat.Property()&(G__BIT_ISCLASS|G__BIT_ISSTRUCT)) &&
       !(dat.Property()&
        (G__BIT_ISPOINTER|G__BIT_ISREFERENCE|G__BIT_ISSTATIC))) {
      // in case of class/struct,  try 
      int store_pc= m_bc_inst.GetPC();
      if(dat.Offset()) m_bc_inst.ADDSTROS(dat.Offset());
      fname = "~";
      fname.append(G__struct.name[dat.Type()->Tagnum()]);
      if(dat.ArrayDim()) {
   Reflex::Member var=G__Dict::GetDict().GetDataMember(dat.Handle());
        m_bc_inst.LD(G__get_varlabel(var, 1) /* number of elements */);
        m_bc_inst.SETARYINDEX(1); // this is illegular, but (1) does --sp
        dtorfound=call_func(*dat.Type(),fname,&para,G__TRYMEMFUNC,1);
        m_bc_inst.RESETARYINDEX(0);
      }
      else {
        dtorfound=call_func(*dat.Type(),fname,&para,G__TRYMEMFUNC,0);
      }
      if(dat.Offset()) m_bc_inst.ADDSTROS(-dat.Offset());
      if(!G__value_typenum(dtorfound)) m_bc_inst.rewind(store_pc);
    }
  }
}

//////////////////////////////////////////////////////////////////////////
void Cint::Bytecode::G__functionscope::ArgumentPassing() {
  G__value dmy;
  G__TypeReader type;
  for(int i=0;i<m_iexist.FunctionParameterSize();i++) {
    G__value_typenum(dmy) = m_iexist.TypeOf().FunctionParameterAt(i);
    type.Init(dmy);
    // duplication, but just in case. G__ClassInfo::Init() may have error

    std::string name = m_iexist.FunctionParameterNameAt(i);
    std::string def = m_iexist.FunctionParameterDefaultAt(i);
    const G__funcentry& fentry = G__get_funcproperties(m_iexist)->entry;
    G__value *defv = 0;
    if (fentry.para_default.size())
       defv = (fentry.para_default)[i];
    EachArgumentPassing(type,name.c_str(),def.c_str(),defv);
  }
}

//////////////////////////////////////////////////////////////////////////
void Cint::Bytecode::G__functionscope::EachArgumentPassing(G__TypeReader& type
                                        ,const char* name
                                      ,const char* def,G__value* /* val */) {
  // allocate variable 
  std::deque<int> arysize; // dummy
  std::deque<int> typesize; // dummy
  Reflex::Member var = allocatevariable(type,name,arysize,typesize,0);

  // bytecode generation
  if(def) {
    std::string defs(def);
    int origin = m_bc_inst.ISDEFAULTPARA();
    compile_expression(defs);
    m_bc_inst.Assign(origin,m_bc_inst.GetPC());
  }
  if(type.Isreference()) {
    // todo, need to review this implementation for reference argument
    //  Maybe Okey, refer to malloc.c, pcode.c and bc_parse.cxx
    m_bc_inst.INIT_REF(var,0,'p');
  }
  else {
    m_bc_inst.ST_LVAR(var,0,'p');
  }
  m_bc_inst.POP();
}

//////////////////////////////////////////////////////////////////////////
void Cint::Bytecode::G__functionscope::ReturnFromFunction() {
  if(m_bc_inst.GetPC()>2 && m_bc_inst.GetInstRel(-2)!=G__RTN_FUNC)
    m_bc_inst.RTN_FUNC(0);
  m_bc_inst.RETURN();
}


//////////////////////////////////////////////////////////////////////////
// set compiled bytecode to ifunc table
//////////////////////////////////////////////////////////////////////////
void Cint::Bytecode::G__functionscope::Storebytecode() {
  // use legacy code in src/ifunc.c
   G__asm_storebytecodefunc(m_iexist,m_scope
                          ,G__asm_stack,G__asm_dt
                          ,G__asm_inst,G__asm_cp); 
  asm_name = (char*)NULL; //avoid deleting already stored bytecode information
}

//////////////////////////////////////////////////////////////////////////
void Cint::Bytecode::G__functionscope::Setstatus() {
  G__funcentry& fentry = G__get_funcproperties(m_iexist)->entry;
  if(fentry.bytecode) {
    if(0==G__xrefflag) 
      fentry.bytecodestatus = G__BYTECODE_SUCCESS;
    else
      fentry.bytecodestatus = G__BYTECODE_ANALYSIS;
  }
  else if(0==G__def_struct_member)
    fentry.bytecodestatus = G__BYTECODE_FAILURE;
}

//////////////////////////////////////////////////////////////////////////
// bytecode compiler entry function
//////////////////////////////////////////////////////////////////////////
int Cint::Bytecode::G__functionscope::compile_normalfunction(const Reflex::Member& func) {
  int store_cintv6 = G__cintv6;
  G__cintv6 |= G__BC_COMPILEERROR;
#if ENABLE_CPP_EXCEPTIONS
  try {
#endif //ENABLE_CPP_EXCEPTIONS
    int result=compile_function(func);
    G__cintv6 = store_cintv6;
    return(result);
  }
#if ENABLE_CPP_EXCEPTIONS
  catch(G__bc_compile_error& /*x*/) {
    G__cintv6 = store_cintv6;
    return(G__BYTECODE_FAILURE);
  }
  catch(...) {
    //throw;
    G__cintv6 = store_cintv6;
    return(G__BYTECODE_FAILURE);
  }
 
  G__cintv6 = store_cintv6;
  return(G__BYTECODE_FAILURE);
#endif //ENABLE_CPP_EXCEPTIONS
}

//////////////////////////////////////////////////////////////////////////
// bytecode compiler entry function
//////////////////////////////////////////////////////////////////////////
int Cint::Bytecode::G__functionscope::compile_function(const Reflex::Member& func) {

  m_iexist = func;
  m_scope = m_iexist.DeclaringScope();

  // store environment
  Store();

  // allocate buffer and initialize environment
  Init();

  //   type(args)  : mem(expr), base(expr) {
  // void f(args)  {
  //        ^ ----> ^
  Setfpos();
  if(G__dispsource) {
    if(!m_scope.IsTopScope()) {
      G__fprinterr(G__serr,"\n%-5d%s::%s(" ,G__ifile.line_number
                ,m_scope.Name().c_str() ,m_iexist.Name().c_str());
    }
    else {
      G__fprinterr(G__serr,"\n%-5d%s(" ,G__ifile.line_number
                ,m_iexist.Name().c_str());
    }
  }
  int c = FposGetReady();

  int addr_start = m_bc_inst.GetPC();

  // enter function scope
  m_bc_inst.ENTERSCOPE(); 

  // generate bytecode for argument passing
  ArgumentPassing();

  // if this is ctor, call that of base class and member
  Baseclassctor(c);

  // if this is operator=, call that of base class and member
  // Baseclassassign(c);

  // compile bytecode
  G__blockscope::compile_core( 1 );

  // if this is dtor, call that of base class and member
  Baseclassdtor();

  // exit from function scope
  m_bc_inst.EXITSCOPE(); 

  // return 
  ReturnFromFunction();

  // resolve goto table
  m_gototable.resolve(m_bc_inst);

  // loop optimization
  int addr_end = m_bc_inst.GetPC();
  m_bc_inst.optimize(addr_start,addr_end);

  // store bytecode
  Storebytecode();
  
  // set bytecode status
  Setstatus();

  // restore environment
  Restorefpos();
  //Restore(); // moved to dtor

  return(G__get_funcproperties(m_iexist)->entry.bytecodestatus);
}

//////////////////////////////////////////////////////////////////////////
// bytecode compiler entry function
//////////////////////////////////////////////////////////////////////////
int Cint::Bytecode::G__functionscope::compile_implicitdefaultctor(const Reflex::Member& func) {
  m_iexist = func;
  m_scope = func.DeclaringScope();

  // store environment
  Store();

  // allocate buffer and initialize environment
  Init();

  int addr_start = m_bc_inst.GetPC();

  // if this is ctor, call that of base class and member
  Baseclassctor('{');
  //m_bc_inst.CL(); 

  // return 
  ReturnFromFunction();

  // resolve goto table
  m_gototable.resolve(m_bc_inst);

  // loop optimization
  int addr_end = m_bc_inst.GetPC();
  m_bc_inst.optimize(addr_start,addr_end);

  // store bytecode
  Storebytecode();
  
  // set bytecode status
  Setstatus();

  // restore environment
  //Restorefpos();
  //Restore(); // moved to dtor

  return(G__get_funcproperties(m_iexist)->entry.bytecodestatus);
}

//////////////////////////////////////////////////////////////////////////
// bytecode compiler entry function
//////////////////////////////////////////////////////////////////////////
int Cint::Bytecode::G__functionscope::compile_implicitcopyctor(const Reflex::Member& func) {
  m_iexist = func;
  m_scope = func.DeclaringScope();

  // store environment
  Store();

  // allocate buffer and initialize environment
  Init();

  int addr_start = m_bc_inst.GetPC();

  //m_bc_inst.PAUSE(); // DEBUG
  // if this is copy ctor, call that of base class and member
  Baseclasscopyctor('{');
  //m_bc_inst.PAUSE(); // DEBUG
  //m_bc_inst.CL(); 

  // return 
  ReturnFromFunction();

  // resolve goto table
  m_gototable.resolve(m_bc_inst);

  // loop optimization
  int addr_end = m_bc_inst.GetPC();
  m_bc_inst.optimize(addr_start,addr_end);

  // store bytecode
  Storebytecode();
  
  // set bytecode status
  Setstatus();

  // restore environment
  //Restorefpos();
  //Restore(); // moved to dtor

  return(G__get_funcproperties(m_iexist)->entry.bytecodestatus);
}

//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// bytecode compiler entry function
//////////////////////////////////////////////////////////////////////////
int Cint::Bytecode::G__functionscope::compile_implicitassign(const Reflex::Member& func) {
  m_iexist = func;
  m_scope = func.DeclaringScope();

  // store environment
  Store();

  // allocate buffer and initialize environment
  Init();

  int addr_start = m_bc_inst.GetPC();

  //m_bc_inst.PAUSE(); // DEBUG
  // if this is operator=, call that of base class and member
  Baseclassassign('{');
  //m_bc_inst.PAUSE(); // DEBUG
  //m_bc_inst.CL(); 

  // return 
  ReturnFromFunction();

  // resolve goto table
  m_gototable.resolve(m_bc_inst);

  // loop optimization
  int addr_end = m_bc_inst.GetPC();
  m_bc_inst.optimize(addr_start,addr_end);

  // store bytecode
  Storebytecode();
  
  // set bytecode status
  Setstatus();

  // restore environment
  //Restorefpos();
  //Restore(); // moved to dtor

  return(G__get_funcproperties(m_iexist)->entry.bytecodestatus);
}

//////////////////////////////////////////////////////////////////////////
// bytecode compiler entry function
//////////////////////////////////////////////////////////////////////////
int Cint::Bytecode::G__functionscope::compile_implicitdtor(const Reflex::Member& func) {
  m_iexist = func;
  m_scope = func.DeclaringScope();

  // store environment
  Store();

  // allocate buffer and initialize environment
  Init();

  int addr_start = m_bc_inst.GetPC();

  // if this is dtor, call that of base class and member
  Baseclassdtor();
  //m_bc_inst.CL(); 

  // return 
  ReturnFromFunction();

  // resolve goto table
  m_gototable.resolve(m_bc_inst);

  // loop optimization
  int addr_end = m_bc_inst.GetPC();
  m_bc_inst.optimize(addr_start,addr_end);

  // store bytecode
  Storebytecode();
  
  // set bytecode status
  Setstatus();

  // restore environment
  //Restorefpos();
  //Restore(); // moved to dtor

  return(G__get_funcproperties(m_iexist)->entry.bytecodestatus);
}

//////////////////////////////////////////////////////////////////////////

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


#endif // 0
