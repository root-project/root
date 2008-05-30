/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file stub.c
 ************************************************************************
 * Description:
 *  New style stub function interface
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"
#include "Dict.h"

#include "Reflex/Member.h"
#include "Reflex/Type.h"

using namespace Cint::Internal;

/**************************************************************************
* G__stubstoreenv()
*
*  Called from interface method source
*
**************************************************************************/
extern "C" void G__stubstoreenv(G__StoreEnv *env,void *p,int tagnum)
{
  env->store_struct_offset = (long)G__store_struct_offset;
  env->store_tagnum = G__get_tagnum(G__tagnum);
  env->store_memberfunc_tagnum = G__get_tagnum(G__memberfunc_tagnum);
  env->store_exec_memberfunc = G__exec_memberfunc;
  if(p) {
    G__store_struct_offset = (char*)p;
    G__tagnum = G__Dict::GetDict().GetScope(tagnum); 
    G__memberfunc_tagnum = G__Dict::GetDict().GetScope(tagnum);
    G__exec_memberfunc = 1;
  }
  else {
    G__store_struct_offset = 0;
    G__tagnum = ::Reflex::Scope();
    G__memberfunc_tagnum = ::Reflex::Scope();
    G__exec_memberfunc = 0;
  }
}

/**************************************************************************
* G__stubrestoreenv()
*
*  Called from interface method source
*
**************************************************************************/
extern "C" void G__stubrestoreenv(G__StoreEnv *env)
{
  G__store_struct_offset = (char*)env->store_struct_offset;
  G__tagnum = G__Dict::GetDict().GetScope(env->store_tagnum);
  G__memberfunc_tagnum = G__Dict::GetDict().GetScope(env->store_memberfunc_tagnum);
  G__exec_memberfunc = env->store_exec_memberfunc;
}

/**************************************************************************
* G__setdouble()
*
**************************************************************************/
extern "C" void G__setdouble(G__value *pbuf
                  ,double d,void *pd
                  ,int type,int tagnum,int typenum,int reftype)
{
  G__value_typenum(*pbuf) = G__get_Type(type,tagnum,typenum,0);
  if(reftype) pbuf->ref = (long)pd;
  else        pbuf->ref = 0;
  pbuf->obj.d = d;
}

/**************************************************************************
* G__setint()
*
**************************************************************************/
extern "C" void G__setint(G__value *pbuf,long l,void *pl
               ,int type,int tagnum,int typenum,int reftype)
{
  G__value_typenum(*pbuf) = G__get_Type(type,tagnum,typenum,0);
  if(reftype) pbuf->ref = (long)pl;
  else        pbuf->ref = 0;
  pbuf->obj.i = l;
}


/**************************************************************************
* G__cppstub_setparam()
*
**************************************************************************/
static void G__cppstub_setparam(char *pformat,char *pbody
                                ,const ::Reflex::Scope &/*tagnum*/
                                ,const ::Reflex::Member &ifunc,int k)
{
  G__StrBuf paraname_sb(G__MAXNAME);
  char *paraname = paraname_sb;
  G__StrBuf temp_sb(G__ONELINE);
  char *temp = temp_sb;

  if (!ifunc) return;

  strcpy(paraname,ifunc.FunctionParameterNameAt(k).c_str());
  if (paraname[0]==0) sprintf(paraname,"a%d",k);

  if(k) strcat(pformat,",");
  strcat(pbody,",");


  // NOTE: The CINT code was ignoring the 'reftype' of the argument
  ::Reflex::Type argType( ifunc.TypeOf().FunctionParameterAt(k) );
  std::string type_name = argType.Name(::Reflex::SCOPED);

  if(argType.IsReference()) {
     sprintf(temp,"*(%s*)(%%ld)",argType.Name(Reflex::SCOPED).c_str());
     strcat(pformat,temp);
     sprintf(temp,"(long)(&%s)",paraname);
     strcat(pbody,temp);
  }
  else {
     argType = argType.FinalType();
     if (argType.IsClass() || argType.IsStruct()) {
        sprintf(temp,"(%s)(%%ld)",type_name.c_str());
        strcat(pformat,temp);
        sprintf(temp,"&%s",paraname);
        strcat(pbody,temp);
     } else {
        switch ( ::Reflex::Tools::FundamentalType(argType.RawType()) ) {
        case ::Reflex::kFLOAT:
        case ::Reflex::kDOUBLE:
           sprintf(temp,"(%s)%%g",type_name.c_str());
           strcat(pformat,temp);
           sprintf(temp,"%s",paraname);
           strcat(pbody,temp);
           break;
        default:
           sprintf(temp,"(%s)(%%ld)",type_name.c_str());
           strcat(pformat,temp);
           sprintf(temp,"(long)%s",paraname);
           strcat(pbody,temp);
        }
     }
  }
}

/**************************************************************************
* G__cppstub_constructor()
*
**************************************************************************/
static void G__cppstub_genconstructor(FILE * /* fp */,const ::Reflex::Scope &tagnum
                                      ,const ::Reflex::Member&)
{
  G__fprinterr(G__serr,"Limitation: Can not make STUB constructor, class %s\n"
               ,tagnum.Name(::Reflex::SCOPED).c_str());
}

/**************************************************************************
* G__cppstub_destructor()
*
**************************************************************************/
static void G__cppstub_gendestructor(FILE * /* fp */,const ::Reflex::Scope &tagnum
                                     ,const ::Reflex::Member&)
{
  G__fprinterr(G__serr,"Limitation: Can not make STUB destructor, class %s\n"
               ,tagnum.Name(::Reflex::SCOPED).c_str());
}

/**************************************************************************
* G__cppstub_genfunc()
*
**************************************************************************/
static void G__cppstub_genfunc(FILE *fp,const ::Reflex::Scope &tagnum,
                               const ::Reflex::Member &ifunc)
{
  unsigned int k;
  G__StrBuf pformat_sb(G__ONELINE);
  char *pformat = pformat_sb;
  G__StrBuf pbody_sb(G__LONGLINE);
  char *pbody = pbody_sb;

  ::Reflex::Type argType( ifunc.TypeOf().ReturnType() );
  std::string type_name = argType.Name();

  /*******************************************************************
  * Function header
  *******************************************************************/
  if(!tagnum || tagnum.IsTopScope() ) {
     fprintf(fp,"%s %s(\n",type_name.c_str(),ifunc.Name().c_str());
  }
  else {
    fprintf(fp,"%s ",type_name.c_str());
    fprintf(fp,"%s::%s(\n",tagnum.Name(::Reflex::SCOPED).c_str(),ifunc.Name().c_str());
  }

  if(G__clock) {
     for(k=0;k<ifunc.FunctionParameterSize();++k) {
        if(k) fprintf(fp,",");
        std::string paraname( ifunc.FunctionParameterNameAt(k));
        if(paraname.length()) {
           fprintf(fp,"%s",paraname.c_str());
        }
        else {
           fprintf(fp,"a%d",k);
        }
     }
     fprintf(fp,")\n");
     for(k=0;k<ifunc.FunctionParameterSize();++k) {
        fprintf(fp,"%s" ,ifunc.TypeOf().FunctionParameterAt(k).Name(::Reflex::QUALIFIED).c_str());
        std::string paraname( ifunc.FunctionParameterNameAt(k));
        if(paraname.length()) {
           fprintf(fp," %s;\n",paraname.c_str());
        }
        else {
           fprintf(fp," a%d;\n",k);
        }
     }
     fprintf(fp,"{\n");
  }
  else {
     for(k=0;k<ifunc.FunctionParameterSize();++k) {
        if(k) fprintf(fp,",\n");
        fprintf(fp,"%s" ,ifunc.TypeOf().FunctionParameterAt(k).Name(::Reflex::QUALIFIED).c_str());
        std::string paraname( ifunc.FunctionParameterNameAt(k));
        if(paraname.length()) {
           fprintf(fp," %s",paraname.c_str());
        }
        else {
           fprintf(fp," a%d",k);
        }
     }
     if(ifunc.TypeOf().IsConst()) fprintf(fp,") const {\n");
     else fprintf(fp,") {\n");
  }

  /*******************************************************************
  * local variable declaration and initialization 
  *******************************************************************/
  fprintf(fp,"  G__value buf;\n");
  fprintf(fp,"  struct G__StoreEnv storeenv;\n");
  fprintf(fp,"  char funccall[G__LONGLINE];\n");

  if(tagnum && !tagnum.IsTopScope()) {
    fprintf(fp
   ,"  G__stubstoreenv(&storeenv,(void*)this,G__get_linked_tagnum(&%s));\n"
            ,G__get_link_tagname(G__get_tagnum(tagnum)));
  }
  else {
    fprintf(fp,"  G__stubstoreenv(&storeenv,(void*)NULL,-1);\n");
  }

  /*******************************************************************
  * stub function call
  *******************************************************************/
  pformat[0] = '\0';
  pbody[0] = '\0';
  for(k=0;k<ifunc.FunctionParameterSize();++k) {
    G__cppstub_setparam(pformat,pbody,tagnum,ifunc,k);
  }
  fprintf(fp,"  sprintf(funccall,\"%s(%s)\"%s);\n"
          ,ifunc.Name().c_str(),pformat,pbody);
  fprintf(fp,"  buf=G__calc(funccall);\n");

  /*******************************************************************
  * clean up
  *******************************************************************/
  fprintf(fp,"  G__stubrestoreenv(&storeenv);\n");

  /*******************************************************************
  * return value
  *******************************************************************/
  if(ifunc.TypeOf().ReturnType().IsReference()) {
     ::Reflex::Type t = ifunc.TypeOf().ReturnType().ToType(); // trim the 'reference'
     
     fprintf(fp,"  return(*(%s*)buf.obj.i);\n",t.Name().c_str());
  }
  else {
     ::Reflex::Type t( ifunc.TypeOf().ReturnType() );
     ::Reflex::Type f( t.FinalType() );
     if (f.IsClass() || f.IsStruct()) { 
        fprintf(fp,"  return(*(%s*)buf.obj.i);\n",t.Name().c_str());
     } else {
        switch ( ::Reflex::Tools::FundamentalType(f.RawType()) ) {
        case ::Reflex::kFLOAT:
        case ::Reflex::kDOUBLE:
           fprintf(fp,"  return((%s)buf.obj.d);\n",f.Name().c_str());
           break;
        case ::Reflex::kVOID:
           break;
        default:
           fprintf(fp,"  return((%s)buf.obj.i);\n",f.Name().c_str());
           break;
        }
     }
  }
  
  fprintf(fp,"}\n\n");
}


/**************************************************************************
* G__cppstub_memfunc() 
*
*  Generate stub member function. Not used
* 
**************************************************************************/
void Cint::Internal::G__cppstub_memfunc(FILE *fp)
{

   fprintf(fp,"\n/*********************************************************\n");
   fprintf(fp,"* Member function Stub\n");
   fprintf(fp,"*********************************************************/\n");


   for( ::Reflex::Type_Iterator i (::Reflex::Type::Type_Begin());
        i != ::Reflex::Type::Type_End();
        ++i) {
      G__RflxProperties *prop = G__get_properties(*i);
     
      if(prop && (G__CPPLINK==prop->globalcomp || G__CLINK==prop->globalcomp)&&
         /* -1==(int)G__struct.parent_tagnum[i]&& */
         -1!=prop->linenum &&
         /* G__struct.hash[i]&& */
         '$'!=i->Name()[0] && 
         !i->IsEnum()) {

         /* member function interface */
         fprintf(fp,"\n/* %s */\n",i->Name(::Reflex::SCOPED).c_str());

         for( ::Reflex::Member_Iterator m ( i->FunctionMember_Begin() );
              m != i->FunctionMember_End();
              ++m) { 
            G__RflxFuncProperties *fprop = G__get_funcproperties(*m);
            if( -1==fprop->entry.line_number
                && !m->TypeOf().IsAbstract()
                /* && *m->hash[j] */
                && (G__CPPSTUB==fprop->globalcomp||G__CSTUB==fprop->globalcomp)) {

               if(m->IsConstructor()) {
                  /* constructor need special handling */
                  G__cppstub_genconstructor(fp,*i,*m);
               }
               else if(m->IsDestructor()) {
                  G__cppstub_gendestructor(fp,*i,*m);
               }
               else {
                  G__cppstub_genfunc(fp,*i,*m);
               }
            } /* if(access) */
         } /* for each Function Member */
      } /* if(globalcomp) */
   } /* for each types */
}

/**************************************************************************
* G__cppstub_func() 
*
*  Generate stub global function
* 
**************************************************************************/
void Cint::Internal::G__cppstub_func(FILE *fp)
{
  fprintf(fp,"\n/*********************************************************\n");
  fprintf(fp,"* Global function Stub\n");
  fprintf(fp,"*********************************************************/\n");

  ::Reflex::Scope s( ::Reflex::Scope::GlobalScope() );

  /* member function interface */
  for( ::Reflex::Member_Iterator m ( s.FunctionMember_Begin() );
           m != s.FunctionMember_End();
           ++m) { 
     G__RflxFuncProperties *fprop = G__get_funcproperties(*m);
     if(G__CPPSTUB==fprop->globalcomp||G__CSTUB==fprop->globalcomp) {

        G__cppstub_genfunc(fp,s,*m);
     } /* if(access) */
  } /* for each Function Member */
} /* if(globalcomp) */

/**************************************************************************
* G__set_stubflags()
*
*  Set stub flags beginningg at certain dictionary position
**************************************************************************/
void Cint::Internal::G__set_stubflags(G__dictposition *dictpos)
{

  /* global variable */
  int i = 0;
  for(Reflex::Member_Iterator iter = dictpos->var.DataMember_Begin();
      iter != dictpos->var.DataMember_End();
      ++iter, ++i ) {

     if (i<dictpos->ig15) continue;

     if(G__is_cppmacro(*iter)) { // was: 'p'!=dictpos->var->type[ig15]) {
        if(G__dispmsg>=G__DISPWARN) {
           G__fprinterr(G__serr,
                        "Warning: global variable %s specified in stub file. Ignored\n"
                        ,iter->Name().c_str());
        }       
      } 
  }

  for(int tagnum=dictpos->tagnum;tagnum<G__struct.alltag;tagnum++) {
     ::Reflex::Scope s( G__Dict::GetDict().GetScope(tagnum) );
     
     for( ::Reflex::Member_Iterator m ( s.FunctionMember_Begin() );
          m != s.FunctionMember_End();
          ++m) { 
        G__RflxFuncProperties *fprop = G__get_funcproperties(*m);
        
        if(-1==fprop->linenum
           && !m->TypeOf().IsAbstract()) {
          switch(G__globalcomp) {
          case G__CPPLINK: fprop->globalcomp=G__CPPSTUB; break;
          case G__CLINK:   fprop->globalcomp=G__CSTUB; break;
          default: break;
          }
        }
     }
  }

  if(dictpos->ifunc) {
     i = 0;
     for(Reflex::Member_Iterator iter = dictpos->ifunc.FunctionMember_Begin();
         iter != dictpos->ifunc.FunctionMember_End();
         ++iter, ++i ) {
        if (i<dictpos->ifn) continue;

        G__RflxFuncProperties *fprop = G__get_funcproperties(*iter);
        switch(fprop->globalcomp) {
        case G__CPPLINK: fprop->globalcomp=G__CPPSTUB; break;
        case G__CLINK:   fprop->globalcomp=G__CSTUB; break;
        default: break;
        }
     }
  }
}

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
