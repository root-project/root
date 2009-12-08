/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file bc_assign.cxx
 ************************************************************************
 * Description:
 *  assignment 
 ************************************************************************
 * Copyright(c) 2004~2005  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "bc_assign.h"

// G__OLDIMPLEMENTATION2182

////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
// variable assignment
////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////
int G__bc_stvar(G__TypeReader& /*ltype*/,G__TypeReader& /*rtype*/
		,struct G__var_array *var,int ig15,int lparan,int lvar_type
		,G__value* /*prresult*/ ,G__bc_inst& inst
		,long struct_offset,long store_struct_offset) {
  if(struct_offset) {
    if(struct_offset!=store_struct_offset) 
      inst.ADDSTROS(struct_offset-store_struct_offset);
    inst.ST_MSTR(var,ig15,lparan,lvar_type);
    if(struct_offset!=store_struct_offset) 
      inst.ADDSTROS(-struct_offset+store_struct_offset);
  }
  else if(G__asm_wholefunction && G__ASM_VARLOCAL==store_struct_offset 
     && G__LOCALSTATIC!=var->statictype[ig15]) {
    inst.ST_LVAR(var,ig15,lparan,lvar_type);
  }
  else {
    inst.ST_VAR(var,ig15,lparan,lvar_type);
  }
  return(1); // always true
}

////////////////////////////////////////////////////////////////
void G__bc_indexoperator(G__TypeReader& ltype,G__value *ppara,int paran) {
  long dmy;
  struct G__param* para = new G__param();
  para->paran = paran;
  for(int i=0;i<paran;i++) para->para[i] = ppara[i];
  G__MethodInfo m = ltype.GetMethod("operator[]",para,&dmy
				    ,G__ClassInfo::ExactMatch);
  delete para;
  if(!m.IsValid()) {
    G__fprinterr(G__serr ,"Error: %s::operator[] not defined " ,ltype.Name());
    G__genericerror((char*)NULL);
    return;
  }
  G__bc_inst& inst = G__currentscope->GetInst();
  inst.PUSHSTROS();
  inst.SETSTROS();
  if(m.Property()&G__BIT_ISCOMPILED) 
    inst.LD_FUNC_BC((struct G__ifunc_table*)m.Handle(),m.Index(),1
		    ,(void*)m.InterfaceMethod());
  else
    inst.LD_FUNC_BC((struct G__ifunc_table*)m.Handle(),m.Index(),1
		    ,(void*)G__bc_exec_normal_bytecode);
  inst.POPSTROS();

  //G__value lval;

  ltype.Init(*m.Type());
  
  return;
}

////////////////////////////////////////////////////////////////
int G__bc_assignment_indexoperator(struct G__var_array *var,int ig15
				   ,int lparan,int lvar_type
				   ,G__TypeReader& ltype
				   ,G__value *ppara,int paran) {
  // TODO , NOT DONE YET, temporarily fix for test/maincmplx.cxxx
  G__bc_inst &inst = G__currentscope->GetInst();
  inst.LD_LVAR(var,ig15,0 /* ??? */ ,lvar_type);
  for(int i=0;i<(-paran);i++) {
    G__bc_indexoperator(ltype,ppara,lparan /* ??? */ ); // ltype is modified
  }
  inst.LETVVAL(); // depends on return type of operator[]
  return 0;
}

////////////////////////////////////////////////////////////////
int G__bc_assignmentopr(G__TypeReader& ltype,G__TypeReader& /*rtype*/
		    ,struct G__var_array *var,int ig15,int lparan,int lvar_type
		    ,G__value* prresult ,G__bc_inst& inst
		    ,long struct_offset,long store_struct_offset) {

  // look for ltype.operator=(rtype)
  struct G__param* para = new G__param();
  para->paran=1;
  para->para[0] = *prresult;
  long dmy=0;
  G__MethodInfo m = ltype.GetMethod("operator=",para,&dmy
				    ,G__ClassInfo::ExactMatch);

  if(m.IsValid()) {

    if(var) {
      if(struct_offset) {
	if(struct_offset!=store_struct_offset) 
	  inst.ADDSTROS(struct_offset-store_struct_offset);
	inst.LD_MSTR(var,ig15,lparan,lvar_type);
	if(struct_offset!=store_struct_offset) 
	  inst.ADDSTROS(-struct_offset+store_struct_offset);
      }
      else if(G__asm_wholefunction && G__ASM_VARLOCAL==store_struct_offset 
       && G__LOCALSTATIC!=var->statictype[ig15]) {
	inst.LD_LVAR(var,ig15,lparan,lvar_type);
      }
      else {
      inst.LD_VAR(var,ig15,lparan,lvar_type);
      }
    }

    inst.PUSHSTROS();
    inst.SETSTROS();

    struct G__ifunc_table *ifunc = (struct G__ifunc_table*)m.Handle();
    int ifn = m.Index();
    if(m.Property()&G__BIT_ISCOMPILED) {
      inst.LD_FUNC_BC(ifunc,ifn,para->paran,(void*)m.InterfaceMethod());
    }
    else if(m.Property()&G__BIT_ISVIRTUAL) {
      inst.LD_FUNC_VIRTUAL(ifunc,ifn,para->paran
			   ,(void*)G__bc_exec_virtual_bytecode);
    }
    else {
      inst.LD_FUNC_BC(ifunc,ifn,para->paran
		      ,(void*)G__bc_exec_normal_bytecode);
    }

    inst.POPSTROS();
    delete para;

    return(1); 
  }

  delete para;

  return(0); 
}

////////////////////////////////////////////////////////////////
int G__bc_conversionctor(G__TypeReader& ltype,G__TypeReader& rtype
		    ,struct G__var_array * /*var*/,int /*ig15*/,int lparan,int /*lvar_type*/
		    ,G__value* prresult ,G__bc_inst& inst
		    ,long /*struct_offset*/,long /*store_struct_offset*/) {

  // look for ltype::ltype(rtype)
  struct G__param* para = new G__param();
  para->paran=1;
  para->para[0] = *prresult;
  long dmy=0;
  G__MethodInfo m = ltype.GetMethod(ltype.TrueName(),para,&dmy
				    ,G__ClassInfo::ExactMatch);

  if(m.IsValid()) {

    if(lparan) inst.REWINDSTACK(lparan);

    inst.ALLOCTEMP(ltype.Tagnum());
    inst.SETTEMP(); // G__store_struct_offset = G__p_tempbuf->obj.obj.i

    struct G__ifunc_table *ifunc = (struct G__ifunc_table*)m.Handle();
    int ifn = m.Index();

    if(m.Property()&G__BIT_ISCOMPILED) {
      inst.SETGVP(1); // G__globalvarpointer = G__store_struct_offset
      inst.LD_FUNC_BC(ifunc,ifn,para->paran,(void*)m.InterfaceMethod());
      inst.SETGVP(-1);
    }
    else {
      inst.LD_FUNC_BC(ifunc,ifn,para->paran
			   ,(void*)G__bc_exec_ctor_bytecode);
    }

    inst.POPTEMP(ltype.Tagnum());

    if(lparan) inst.REWINDSTACK(-lparan);

    rtype = ltype;
    rtype.append_const();
    *prresult = rtype.Value();
    delete para;

    return(1); 
  }

  delete para;

  return(0); 
}

////////////////////////////////////////////////////////////////
int G__bc_baseconvobj(G__TypeReader& ltype,G__TypeReader& rtype
		      ,struct G__var_array * /*var*/,int /*ig15*/,int lparan
		      ,int /*lvar_type*/
		      ,G__value* /*prresult*/ ,G__bc_inst& inst
		      ,long /*struct_offset*/,long /*store_struct_offset*/) {

  long baseoffset=G__ispublicbase(ltype.Tagnum(),rtype.Tagnum(),(long)0);
  if(-1!=baseoffset) {
    if(lparan) inst.REWINDSTACK(lparan);
    inst.BASECONV(ltype.Tagnum(),baseoffset); // only if not virtual base
    if(lparan) inst.REWINDSTACK(-lparan);
    rtype.G__ClassInfo::Init(ltype.Tagnum()); // questionable
    return(1); 
  }

  // TODO, virtual base class
  //inst.CAST(ltype); // TODO

  return(0);
}

////////////////////////////////////////////////////////////////
int G__bc_membercopy(G__TypeReader& /*ltype*/,G__TypeReader& /*rtype*/
		    ,struct G__var_array * /*var*/,int /*ig15*/,int /*lparan*/,int /*lvar_type*/
		    ,G__value* /*prresult*/ ,G__bc_inst& /*inst*/
		    ,long /*struct_offset*/,long /*store_struct_offset*/) {
  return(1);  // returning 1, calls G__bc_stvar or G__bc_letvar
}

////////////////////////////////////////////////////////////////
int G__bc_conversionopr(G__TypeReader& ltype,G__TypeReader& rtype
			    ,struct G__var_array * /*var*/,int /*ig15*/,int lparan,int /*lvar_type*/
			    ,G__value* /*prresult*/ ,G__bc_inst& inst
			    ,long /*struct_offset*/,long /*store_struct_offset*/) {

  // look for rtype::operator ltype()
  struct G__param* para = new G__param();
  para->paran=0;
  long dmy=0;
  string fname ="operator ";
  fname.append(ltype.TrueName());
  G__MethodInfo m = rtype.GetMethod(fname.c_str(),para,&dmy
				    ,G__ClassInfo::ExactMatch);

  if(m.IsValid()) {

    if(lparan) inst.REWINDSTACK(lparan);

    inst.PUSHSTROS();
    inst.SETSTROS();

    struct G__ifunc_table *ifunc = (struct G__ifunc_table*)m.Handle();
    int ifn = m.Index();

    if(m.Property()&G__BIT_ISCOMPILED) {
      inst.LD_FUNC_BC(ifunc,ifn,para->paran,(void*)m.InterfaceMethod());
    }
    else if(m.Property()&G__BIT_ISVIRTUAL) {
      inst.LD_FUNC_VIRTUAL(ifunc,ifn,para->paran
			   ,(void*)G__bc_exec_virtual_bytecode);
    }
    else {
      inst.LD_FUNC_BC(ifunc,ifn,para->paran
			   ,(void*)G__bc_exec_normal_bytecode);
    }

    inst.POPSTROS();

    if(lparan) inst.REWINDSTACK(-lparan);

    rtype = ltype;
    delete para;
  
    return(1); 
  }
  
  delete para;

  return(0); 
}

////////////////////////////////////////////////////////////////
int G__bc_baseconvptr(G__TypeReader& ltype,G__TypeReader& rtype
		      ,struct G__var_array *var,int ig15,int lparan,int lvar_type
		      ,G__value* prresult ,G__bc_inst& /*inst*/
		      ,long struct_offset,long store_struct_offset) {
  
  return(G__bc_baseconvobj(ltype,rtype,var,ig15,lparan,lvar_type
			   ,prresult
			   ,G__currentscope->GetInst()
			   ,struct_offset,store_struct_offset));
}
////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////
extern "C" int G__bc_assignment(struct G__var_array *var,int ig15,int lparan
				,int lvar_type,G__value *prresult
				,long struct_offset,long store_struct_offset
				,G__value *ppara) {
  // error check
  // TODO

  // prepare lval object
  G__value lval = G__null;
  lval.isconst = 0;
  lval.type = var->type[ig15];
  lval.tagnum = var->p_tagtable[ig15];
  lval.typenum = var->p_typetable[ig15];
  lval.obj.reftype.reftype = var->reftype[ig15];

  // prepare ltype and rtype objects
  G__TypeReader ltype(lval), rtype(*prresult);
  int paran = var->paran[ig15] - lparan;
  if(paran>0) for(int i=0;i<paran;i++) ltype.incplevel();
  else for(int i=0;i<(-paran);i++) {
    if(ltype.Ispointer()>0) ltype.decplevel();
    else if(ltype.Type()=='u') {
      G__bc_assignment_indexoperator(var,ig15,lparan,lvar_type,ltype,ppara
				     ,paran);
      return(0);
    }
    else {
      G__fprinterr(G__serr
		   ,"Error: illegal use of operator[] to fundamental type %s"
		   ,var->varnamebuf[ig15]);
      G__genericerror((char*)NULL);
    }
  }
  if(ltype.Ispointer() && 'v'==lvar_type) ltype.decplevel();

  // fundamental types, normal assignment
  if(tolower(ltype.Type())!='u' && tolower(rtype.Type())!='u' &&
     G__Isvalidassignment(ltype,rtype,prresult) ) {
    // always true
    if(G__bc_stvar(ltype,rtype,var,ig15,lparan,lvar_type,prresult
		   ,G__currentscope->GetInst()
		   ,struct_offset,store_struct_offset)) return(0);
  }

  // assignment operator
  if(ltype.Type()=='u') {
    // search target.operator=(origin), return if found
    if(G__bc_assignmentopr(ltype,rtype,var,ig15,lparan,lvar_type,prresult
			   ,G__currentscope->GetInst() 
			   ,struct_offset,store_struct_offset)) return(0);
  }

  // conversion ctor
  if(ltype.Type()=='u') {
    // search target.target(origin)
    // return if found
    if(G__bc_conversionctor(ltype,rtype,var,ig15,lparan,lvar_type,prresult
			    ,G__currentscope->GetInst()
			    ,struct_offset,store_struct_offset)) {
      // questionable
      if(ltype.Type()=='u') {
	// search target.operator=(origin), return if found
	if(G__bc_assignmentopr(ltype,rtype,var,ig15,lparan,lvar_type,prresult
			       ,G__currentscope->GetInst() 
			       ,struct_offset,store_struct_offset)) return(0);
      }
      G__bc_stvar(ltype,rtype,var,ig15,lparan,lvar_type,prresult
		  ,G__currentscope->GetInst()
		  ,struct_offset,store_struct_offset);
      return(0);
    }
  }

  // class object assignment
  if(ltype.Type()=='u' && rtype.Type()=='u') {
    // if(ltype.Ispointer()==rtype.Ispointer()) ; given condition
    if(G__bc_baseconvobj(ltype,rtype,var,ig15,lparan,lvar_type
			 ,prresult ,G__currentscope->GetInst()
			 ,struct_offset,store_struct_offset)) {
      G__bc_stvar(ltype,rtype,var,ig15,lparan,lvar_type,prresult
		  ,G__currentscope->GetInst()
		  ,struct_offset,store_struct_offset);
      return(0);
    }
    if(ltype.Tagnum()==rtype.Tagnum()) {
      // memberwise copy, always true
      if(G__bc_membercopy(ltype,rtype,var,ig15,lparan,lvar_type
			  ,prresult ,G__currentscope->GetInst()
			  ,struct_offset,store_struct_offset)) {
	G__bc_stvar(ltype,rtype,var,ig15,lparan,lvar_type,prresult
		    ,G__currentscope->GetInst()
		    ,struct_offset,store_struct_offset);
	return(0);
      }
    }
  }

  // conversion operator from original class to target type
  if(rtype.Type()=='u' && 0==rtype.Ispointer()) {
    // search rtype.operator target(), return if found 
    if(G__bc_conversionopr(ltype,rtype,var,ig15,lparan,lvar_type
			   ,prresult ,G__currentscope->GetInst()
			   ,struct_offset,store_struct_offset)) {
      G__bc_stvar(ltype,ltype,var,ig15,lparan,lvar_type,prresult
		  ,G__currentscope->GetInst()
		  ,struct_offset,store_struct_offset);
      return(0);
    }
  }

  // class pointer assignment
  if(ltype.Type()=='U' && rtype.Type()=='U') { 
    if(1==ltype.Ispointer() && 1==rtype.Ispointer()) {
      if(ltype.Tagnum()==rtype.Tagnum()) {
	// simple assignment, always true
	if(G__bc_assignmentopr(ltype,rtype,var,ig15,lparan,lvar_type,prresult
			       ,G__currentscope->GetInst()
			       ,struct_offset,store_struct_offset)) return(0);
      }
      // add struct offset, this part is always true
      if(G__bc_baseconvptr(ltype,rtype,var,ig15,lparan,lvar_type
			   ,prresult
			   ,G__currentscope->GetInst()
			   ,struct_offset,store_struct_offset)) {
	G__bc_stvar(ltype,ltype,var,ig15,lparan,lvar_type,prresult
		    ,G__currentscope->GetInst()
		    ,struct_offset,store_struct_offset);
	return(0);
      }
    }
    else if(G__Isvalidassignment(ltype,rtype,prresult)) {
      // pointer to pointer of class object
      // always true
      if(G__bc_stvar(ltype,rtype,var,ig15,lparan,lvar_type,prresult
		     ,G__currentscope->GetInst()
		     ,struct_offset,store_struct_offset)) return(0);
    }
  } 

  // void* assignment
  if((ltype.Type()=='Y' && rtype.Ispointer()) 
     || (ltype.Ispointer()  && 0==G__int(*prresult))) {
    // simple assignment, this part is always true
    if(G__bc_stvar(ltype,rtype,var,ig15,lparan,lvar_type,prresult
		   ,G__currentscope->GetInst()
		   ,struct_offset,store_struct_offset)) return(0);
    return(0);
  }

  // if non of above, illegal assignment
  G__fprinterr(G__serr,"Error: illegal assignment to %s"
	       ,var->varnamebuf[ig15]);
  G__genericerror((char*)NULL);
  return(1);
}
////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
// object assignment
////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////
int G__bc_letvar(G__value * /*plresult*/ ,G__value * /*prresult*/
		 ,G__bc_inst& inst) {
  int pc = inst.GetPC();
  if(pc>2 && inst.GetInst(pc-2)==G__TOVALUE) {
    inst.inc_cp_asm(-2,0);
    inst.LETPVAL();
  }
  else {
    inst.LETVVAL();
  }
  return(1);
}


////////////////////////////////////////////////////////////////
extern "C" int G__bc_objassignment(G__value *plresult ,G__value *prresult) {

  // prepare ltype and rtype objects
  G__TypeReader ltype(*plresult), rtype(*prresult);

  // fundamental types, normal assignment
  if(tolower(ltype.Type())!='u' && tolower(rtype.Type())!='u' &&
     G__Isvalidassignment(ltype,rtype,prresult) ) {
    // always true
    if(G__bc_letvar(plresult,prresult,G__currentscope->GetInst())) return(0);
  }

  // assignment operator
  if(ltype.Type()=='u') {
    // search target.operator=(origin), return if found
    if(G__bc_assignmentopr(ltype,rtype,0,0,0,0,prresult
			   ,G__currentscope->GetInst(),0,0)) return(0);
  }

  // conversion ctor
  if(ltype.Type()=='u') {
    // search target.target(origin)
    if(G__bc_conversionctor(ltype,rtype,0,0,0,0,prresult
			    ,G__currentscope->GetInst() ,0,0)) {
      if(ltype.Type()=='u') {
	// search target.operator=(origin), return if found
	if(G__bc_assignmentopr(ltype,rtype,0,0,0,0,prresult
			       ,G__currentscope->GetInst(),0,0)) return(0);
      }
      G__bc_letvar(plresult,prresult,G__currentscope->GetInst());
      return(0);
    }
    // return if found
  }

  // class object assignment
  if(ltype.Type()=='u' && rtype.Type()=='u') {
    // if(ltype.Ispointer()==rtype.Ispointer()) ; given condition
    if(G__bc_baseconvobj(ltype,rtype,0,0,0,0
			 ,prresult ,G__currentscope->GetInst()
			 ,0,0)) {
      G__bc_letvar(plresult,prresult,G__currentscope->GetInst());
      return(0);
    }
    if(ltype.Tagnum()==rtype.Tagnum()) {
      // memberwise copy, always true
      if(G__bc_membercopy(ltype,rtype,0,0,0,0
			  ,prresult ,G__currentscope->GetInst()
			  ,0,0)) {
	G__bc_letvar(plresult,prresult,G__currentscope->GetInst());
	return(0);
      }
    }

  }

  // conversion operator from original class to target type
  if(rtype.Type()=='u' && 0==rtype.Ispointer()) {
    // search rtype.operator target(), return if found 
    if(G__bc_conversionopr(ltype,rtype,0,0,0,0
			   ,prresult,G__currentscope->GetInst(),0,0)) {
      G__bc_letvar(plresult,prresult,G__currentscope->GetInst());
      return(0);
    }
  }

  // class pointer assignment
  if(ltype.Type()=='U' && rtype.Type()=='U') { 
    if(1==ltype.Ispointer() && 1==rtype.Ispointer()) {
      if(ltype.Tagnum()==rtype.Tagnum()) {
	// simple assignment, always true
	if(G__bc_assignmentopr(ltype,rtype,0,0,0,0,0
			       ,G__currentscope->GetInst(),0,0)) return(0);
      }
      // add struct offset, this part is always true
      if(G__bc_baseconvptr(ltype,rtype,0,0,0,0
			   ,prresult,G__currentscope->GetInst()
			   ,0,0)) {
	G__bc_letvar(plresult,prresult,G__currentscope->GetInst());
	return(0);
      }
    }
    else if(G__Isvalidassignment(ltype,rtype,prresult)) {
      // pointer to pointer of class object
      // always true
      G__bc_letvar(plresult,prresult,G__currentscope->GetInst());
      return(0);
    }
  } 

  // void* assignment
  if((ltype.Type()=='Y' && rtype.Ispointer()) 
     || (ltype.Ispointer()  && 0==G__int(*prresult))) {
    // simple assignment, this part is always true
    G__bc_letvar(plresult,prresult,G__currentscope->GetInst());
    return(0);
  }

  // if non of above, illegal assignment
  G__fprinterr(G__serr,"Error: illegal assignment");
  G__genericerror((char*)NULL);
  return(1);
}
////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////




////////////////////////////////////////////////////////////////



