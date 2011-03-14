/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file ifunc.c
 ************************************************************************
 * Description:
 *  interpret function and new style compiled function
 ************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#include "common.h"
#include <map>
#include <set>

extern "C" void G__exec_alloc_lock();
extern "C" void G__exec_alloc_unlock();

//______________________________________________________________________________
namespace {
static std::map<int /*tagnum*/, std::set<G__ifunc_table> > & G__ifunc_refs()
{
   static std::map<int /*tagnum*/, std::set<G__ifunc_table> > ifunc_refs;
   return ifunc_refs;
}
} // unnamed namespace

extern "C" {

static int G__readansiproto(G__ifunc_table_internal* ifunc, int func_now);

static int G__calldepth = 0;

#ifndef G__OLDIMPLEMENTATION1167
//______________________________________________________________________________
void G__reftypeparam(G__ifunc_table_internal* p_ifunc, int ifn, G__param* libp)
{
   // -- FIXME: Describe this function!
   for (int itemp = 0; itemp < p_ifunc->para_nu[ifn] && itemp < libp->paran; ++itemp) {
      if (
         G__PARAREFERENCE == p_ifunc->param[ifn][itemp]->reftype &&
         p_ifunc->param[ifn][itemp]->type != libp->para[itemp].type
      ) {
         switch (p_ifunc->param[ifn][itemp]->type) {
            case 'c':
               libp->para[itemp].ref = (long) G__Charref(&libp->para[itemp]);
               break;
            case 's':
               libp->para[itemp].ref = (long) G__Shortref(&libp->para[itemp]);
               break;
            case 'i':
               libp->para[itemp].ref = (long) G__Intref(&libp->para[itemp]);
               break;
            case 'l':
               libp->para[itemp].ref = (long) G__Longref(&libp->para[itemp]);
               break;
            case 'b':
               libp->para[itemp].ref = (long) G__UCharref(&libp->para[itemp]);
               break;
            case 'r':
               libp->para[itemp].ref = (long) G__UShortref(&libp->para[itemp]);
               break;
            case 'h':
               libp->para[itemp].ref = (long) G__UIntref(&libp->para[itemp]);
               break;
            case 'k':
               libp->para[itemp].ref = (long) G__ULongref(&libp->para[itemp]);
               break;
            case 'f':
               libp->para[itemp].ref = (long) G__Floatref(&libp->para[itemp]);
               break;
            case 'd':
               libp->para[itemp].ref = (long) G__Doubleref(&libp->para[itemp]);
               break;
            case 'g':
               libp->para[itemp].ref = (long) G__Boolref(&libp->para[itemp]);
               break;
            case 'n':
               libp->para[itemp].ref = (long) G__Longlongref(&libp->para[itemp]);
               break;
            case 'm':
               libp->para[itemp].ref = (long) G__ULonglongref(&libp->para[itemp]);
               break;
            case 'q':
               libp->para[itemp].ref = (long) G__Longdoubleref(&libp->para[itemp]);
               break;
         }
      }
   }
}
#endif // G__OLDIMPLEMENTATION1167

//______________________________________________________________________________
static void G__warn_refpromotion(G__ifunc_table_internal* p_ifunc, int ifn, int itemp, G__param* libp)
{
   // -- FIXME: Describe this function!
   if (
      G__PARAREFERENCE == p_ifunc->param[ifn][itemp]->reftype &&
      'u' != p_ifunc->param[ifn][itemp]->type &&
      p_ifunc->param[ifn][itemp]->type != libp->para[itemp].type &&
      0 != libp->para[itemp].obj.i &&
      G__VARIABLE == p_ifunc->param[ifn][itemp]->isconst
   ) {
#ifdef G__OLDIMPLEMENTATION1167
      if (G__dispmsg >= G__DISPWARN) {
         G__fprinterr(G__serr, "Warning: implicit type conversion of non-const reference arg %d", itemp);
         G__printlinenum();
      }
#endif // G__OLDIMPLEMENTATION1167
   }
}

#ifdef G__ASM_WHOLEFUNC
//______________________________________________________________________________
void G__free_bytecode(G__bytecodefunc* bytecode)
{
   // -- FIXME: Describe this function!
   if (bytecode) {
      if (bytecode->asm_name) {
         free((void*) bytecode->asm_name);
         bytecode->asm_name = 0;
      }
      if (bytecode->pstack) {
         free((void*) bytecode->pstack);
         bytecode->pstack = 0;
      }
      if (bytecode->pinst) {
         free((void*) bytecode->pinst);
         bytecode->pinst = 0;
      }
      if (bytecode->var) {
         G__destroy_upto(bytecode->var, G__BYTECODELOCAL_VAR, 0, -1);
         free((void*) bytecode->var);
         bytecode->var = 0;
      }
      free((void*) bytecode);
   }
}

//______________________________________________________________________________
void G__asm_storebytecodefunc(G__ifunc_table_internal* ifunc, int ifn, G__var_array* var, G__value* pstack, int sp, long* pinst, int instsize)
{
   // -- FIXME: Describe this function!
   struct G__bytecodefunc* bytecode;
   /* check if the function is already compiled, replace old one */
   if (ifunc->pentry[ifn]->bytecode) {
      G__genericerror("Internal error: G__asm_storebytecodefunc duplicated");
   }
   /* allocate bytecode buffer */
   bytecode = (struct G__bytecodefunc*)malloc(sizeof(struct G__bytecodefunc));
   ifunc->pentry[ifn]->bytecode = bytecode;
   /* store function ID */
   bytecode->ifunc = ifunc;
   bytecode->ifn = ifn;
   /* copy local variable table */
   bytecode->var = var;
   bytecode->varsize = G__struct.size[G__MAXSTRUCT-1];
   /* copy instruction */
   bytecode->pinst = (long*)malloc(sizeof(long) * instsize + 8);
   memcpy(bytecode->pinst, pinst, sizeof(long)*instsize + 1);
   bytecode->instsize = instsize;
   /* copy constant data stack */
   bytecode->stacksize = G__MAXSTACK - sp;
   bytecode->pstack = (G__value*)malloc(sizeof(G__value) * bytecode->stacksize);
   memcpy((void*)bytecode->pstack, (void*)(&pstack[sp]), sizeof(G__value)*bytecode->stacksize);
   /* copy compiled and library function name buffer */
   if (0 == G__asm_name_p) {
      if (G__asm_name) {
         free(G__asm_name);
         G__asm_name = 0;
      }
      bytecode->asm_name = 0;
   }
   else {
      bytecode->asm_name = G__asm_name;
   }
#ifdef G__OLDIMPLEMENtATION1578 /* Problem  t1048.cxx */
   /* store pointer to function */
   ifunc->pentry[ifn]->tp2f = (void*) bytecode;
#endif // G__OLDIMPLEMENtATION1578
   // --
}

//______________________________________________________________________________
void* G__allocheapobjectstack G__P((struct G__ifunc_table* ifunc, int ifn, int scopelevel));
void G__copyheapobjectstack G__P((void* p, G__value* result, struct G__ifunc_table* ifunc, int ifn));

//______________________________________________________________________________
int G__noclassargument(G__ifunc_table_internal* ifunc, int iexist)
{
   // -- Stops bytecode compilation if class object is passed as argument.
   for (int i = 0;i < ifunc->para_nu[iexist];i++) {
      if (
         'u' == ifunc->param[iexist][i]->type &&
         G__PARAREFERENCE != ifunc->param[iexist][i]->reftype
      ) {
         /* return false if class/struct object and non-reference type arg */
         return 0;
      }
   }
   return 1;
}

//______________________________________________________________________________
int G__compile_bytecode(G__ifunc_table* iref, int iexist)
{
   // -- FIXME: Describe this function!
#ifdef G__ASM
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(G__serr, "G__compile_bytecode: begin bytecode compilation ...\n");
   }
#endif // G__ASM_DBG
#endif // G__ASM
   G__value buf;
   struct G__param para;
   struct G__input_file store_ifile;
   int store_prerun = G__prerun;
   int store_asm_index = G__asm_index;
   int store_no_exec = G__no_exec;
   int store_asm_exec = G__asm_exec;
   int store_tagdefining = G__tagdefining;
   int store_asm_noverflow = G__asm_noverflow;
   int funcstatus;
   long store_globalvarpointer = G__globalvarpointer;
   G__FastAllocString funcname(G__ONELINE);
   int store_dispsource = G__dispsource;
   if (G__step || G__stepover) {
      G__dispsource = 0;
   }
   G__ifunc_table_internal* ifunc = G__get_ifunc_internal(iref);
   if (
      G__xrefflag ||
      (
         (ifunc->pentry[iexist]->size < G__ASM_BYTECODE_FUNC_LIMIT) &&
         !G__def_struct_member &&
         ((ifunc->type[iexist] != 'u') || (ifunc->reftype[iexist] == G__PARAREFERENCE)) &&
         (!ifunc->para_nu[iexist] || (ifunc->ansi[iexist] && G__noclassargument(ifunc, iexist)))
      )
   ) {
      para.paran = 0;
      para.para[0] = G__null;
      G__tagdefining = G__MAXSTRUCT - 1;
      G__struct.type[G__tagdefining] = 's';
      G__struct.size[G__tagdefining] = 0;
      G__no_exec = 0;
      G__prerun = 0;
      G__asm_exec = 1;
      G__asm_wholefunction = G__ASM_FUNC_COMPILE;
      G__asm_noverflow = 0;
      store_ifile = G__ifile;
      G__asm_index = iexist;
#ifdef G__ASM
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(
              G__serr
            , "\n!!!G__compile_bytecode: Increment G__templevel %d --> %d  %s:%d\n"
            , G__templevel
            , G__templevel + 1
            , __FILE__
            , __LINE__
         );
      }
#endif // G__ASM_DBG
#endif // G__ASM
      ++G__templevel;
      ++G__calldepth;
      funcname = ifunc->funcname[iexist];
      if (-1 == ifunc->tagnum) {
         funcstatus = G__TRYNORMAL;
      }
      else {
         funcstatus = G__CALLMEMFUNC;
      }
      G__init_jumptable_bytecode();
#ifdef G__ASM
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "G__compile_bytecode: calling G__interpret_func ...\n");
      }
#endif // G__ASM_DBG
#endif // G__ASM
      G__interpret_func(&buf, funcname, &para, ifunc->hash[iexist] , ifunc, G__EXACT, funcstatus);
#ifdef G__ASM
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "G__compile_bytecode: finished G__interpret_func.\n");
         if (ifunc->pentry[iexist]->bytecode) {
            G__fprinterr(G__serr, "G__compile_bytecode: success.\n");
         }
      }
#endif // G__ASM_DBG
#endif // G__ASM
      G__init_jumptable_bytecode();
      --G__calldepth;
#ifdef G__ASM
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(
              G__serr
            , "\n!!!G__compile_bytecode: Destroy temp objects now at G__templevel %d  %s:%d\n"
            , G__templevel
            , __FILE__
            , __LINE__
         );
      }
#endif // G__ASM_DBG
#endif // G__ASM
      G__free_tempobject();
#ifdef G__ASM
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(
              G__serr
            , "\n!!!G__compile_bytecode: Decrement G__templevel %d --> %d  %s:%d\n"
            , G__templevel
            , G__templevel - 1
            , __FILE__
            , __LINE__
         );
      }
#endif // G__ASM_DBG
#endif // G__ASM
      --G__templevel;
      G__tagdefining = store_tagdefining;
      G__asm_exec = store_asm_exec;
      G__no_exec = store_no_exec;
      G__prerun = store_prerun;
      G__asm_index = store_asm_index;
      G__asm_wholefunction = G__ASM_FUNC_NOP;
      G__ifile = store_ifile;
      G__asm_noverflow = store_asm_noverflow;
      G__globalvarpointer = store_globalvarpointer;
   }
#ifdef G__ASM
#ifdef G__ASM_DBG
   else if (G__asm_dbg) {
      G__fprinterr(G__serr, "!!!bytecode compilation %s not tried either because\n", ifunc->funcname[iexist]);
      G__fprinterr(G__serr, "    function is longer than %d lines\n", G__ASM_BYTECODE_FUNC_LIMIT);
      G__fprinterr(G__serr, "    function returns class object or reference type\n");
      G__fprinterr(G__serr, "    function is K&R style\n");
      G__printlinenum();
   }
#endif // G__ASM_DBG
#endif // G__ASM
   if (ifunc->pentry[iexist]->bytecode) {
      if (!G__xrefflag) {
         ifunc->pentry[iexist]->bytecodestatus = G__BYTECODE_SUCCESS;
      }
      else {
         ifunc->pentry[iexist]->bytecodestatus = G__BYTECODE_ANALYSIS;
      }
   }
   else if (!G__def_struct_member) {
      ifunc->pentry[iexist]->bytecodestatus = G__BYTECODE_FAILURE;
   }
   G__dispsource = store_dispsource;
#ifdef G__ASM
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(G__serr, "G__compile_bytecode: end bytecode compilation.\n");
   }
#endif // G__ASM_DBG
#endif // G__ASM
   return ifunc->pentry[iexist]->bytecodestatus;
}

//______________________________________________________________________________
//______________________________________________________________________________
#define G__MAXGOTOLABEL 30

struct G__gotolabel
{
   int pc;
   char* label;
};

static int G__ngoto  = 0;
static int G__nlabel = 0;
static struct G__gotolabel G__gototable[G__MAXGOTOLABEL];
static struct G__gotolabel G__labeltable[G__MAXGOTOLABEL];
//______________________________________________________________________________
//______________________________________________________________________________

//______________________________________________________________________________
static void G__free_gotolabel(struct G__gotolabel* pgotolabel, int* pn)
{
   // -- FIXME: Describe this function!
   while (*pn > 0)
   {
      --(*pn);
      free((char*) pgotolabel[*pn].label);
      pgotolabel[*pn].label = 0;
   }
}

//______________________________________________________________________________
void G__init_jumptable_bytecode()
{
   // -- FIXME: Describe this function!
   G__free_gotolabel(G__labeltable, &G__nlabel);
   G__free_gotolabel(G__gototable, &G__ngoto);
}

//______________________________________________________________________________
void G__add_label_bytecode(char* label)
{
   // -- FIXME: Describe this function!
   if (G__nlabel < G__MAXGOTOLABEL) {
      int len = strlen(label);
      if (len) {
         G__labeltable[G__nlabel].pc = G__asm_cp;
         label[len-1] = 0;
         G__labeltable[G__nlabel].label = (char*) malloc(strlen(label) + 1);
         strcpy(G__labeltable[G__nlabel].label, label); // Okay, we allocated enough space
         ++G__nlabel;
      }
   }
   else {
      G__abortbytecode();
   }
}

//______________________________________________________________________________
void G__add_jump_bytecode(char* label)
{
   // -- FIXME: Describe this function!
   if (G__ngoto < G__MAXGOTOLABEL) {
      int len = strlen(label);
      if (len) {
         G__gototable[G__ngoto].pc = G__asm_cp + 1;
         G__asm_inst[G__asm_cp] = G__JMP;
         G__inc_cp_asm(2, 0);
         G__gototable[G__ngoto].label = (char*) malloc(strlen(label) + 1);
         strcpy(G__gototable[G__ngoto].label, label); // Okay, we allocated enough space
         ++G__ngoto;
      }
   }
   else {
      G__abortbytecode();
   }
}

//______________________________________________________________________________
void G__resolve_jumptable_bytecode()
{
   // -- FIXME: Describe this function!
   if (G__asm_noverflow) {
      int i, j;
      for (j = 0; j < G__nlabel; j++) {
         for (i = 0; i < G__ngoto; i++) {
            if (strcmp(G__gototable[i].label, G__labeltable[j].label) == 0) {
               G__asm_inst[G__gototable[i].pc] = G__labeltable[j].pc;
            }
         }
      }
   }
   G__init_jumptable_bytecode();
}

#endif // G__ASM_WHOLEFUNC

//______________________________________________________________________________
int G__istypename(char* temp)
{
   // -- True if fundamental type, class, struct, typedef, template class name.
   if (strncmp(temp, "class ", 6) == 0) {
      temp += 6;
   }
   else if (strncmp(temp, "struct ", 7) == 0) {
      temp += 7;
   }
   else if (strncmp(temp, "enum ", 5) == 0) {
      temp += 5;
   }
   if (strchr(temp, '(') || strchr(temp, ')') || strchr(temp, '|')) {
      return 0;
   }
   if ('\0' == temp[0]) {
      return 0;
   }
   if (
      strcmp(temp, "int") == 0 ||
      strcmp(temp, "short") == 0 ||
      strcmp(temp, "char") == 0 ||
      strcmp(temp, "long") == 0 ||
      strcmp(temp, "float") == 0 ||
      strcmp(temp, "double") == 0 ||
      (strncmp(temp, "unsigned", 8) == 0 &&
       (!temp[8] ||
        strcmp(temp+9, "char") == 0 ||
        strcmp(temp+9, "short") == 0 ||
        strcmp(temp+9, "int") == 0 ||
        strcmp(temp+9, "long") == 0)) ||
      (strncmp(temp, "signed", 6) == 0 &&
       (!temp[6] ||
        strcmp(temp+7, "char") == 0 ||
        strcmp(temp+7, "short") == 0 ||
        strcmp(temp+7, "int") == 0 ||
        strcmp(temp+7, "long") == 0)) ||
      strcmp(temp, "const") == 0 ||
      strcmp(temp, "void") == 0 ||
      strcmp(temp, "FILE") == 0 ||
      strcmp(temp, "class") == 0 ||
      strcmp(temp, "struct") == 0 ||
      strcmp(temp, "union") == 0 ||
      strcmp(temp, "enum") == 0 ||
      strcmp(temp, "register") == 0 ||
      strcmp(temp, "bool") == 0 ||
      (G__iscpp && strcmp(temp, "typename") == 0) ||
      -1 != G__defined_typename(temp) ||
      -1 != G__defined_tagname(temp, 2) ||
      G__defined_templateclass(temp)
   ) {
      return 1;
   }
   if (G__fpundeftype) {
      return 1;
   }
   return 0;
}

} // Extern "C"

//______________________________________________________________________________
void G__make_ifunctable(G__FastAllocString &funcheader)
{
   // -- FIXME: Describe this function!
   //
   // Note: G__no_exec is always zero on entry.
   //
   int iin = 0;
   int cin = '\0';
   int func_now;
   int iexist;
   struct G__ifunc_table_internal* ifunc;
   char store_type;
   int store_tagnum;
   int store_typenum;
   int store_access;
   int paranu = 0;
   int dobody = 0;
#ifdef G__FRIEND
   struct G__friendtag* friendtag;
#endif // G__FRIEND
#ifdef G__NEWINHERIT
   int basen;
   struct G__inheritance* baseclass;
#endif // G__NEWINHERIT
   int isvoid = 0;
   /*****************************************************
    * to get type of function parameter
    *****************************************************/
   int iin2;
   fpos_t temppos;
   int store_line_number;
   int store_def_struct_member;
   struct G__ifunc_table_internal* store_ifunc;
   struct G__ifunc_table_internal* store_ifunc_tmp;
   /* system check */
   G__ASSERT(G__prerun);
   store_ifunc = G__p_ifunc;
   if (G__def_struct_member && G__def_tagnum != -1) {
      /* no need for incremental setup */
      G__p_ifunc = G__struct.memfunc[G__def_tagnum];
   }
   /* Store ifunc to check if same function already exists */
   ifunc = G__p_ifunc;
   /* Get to the last page of interpreted function list */
   while (G__p_ifunc->next) G__p_ifunc = G__p_ifunc->next;
   if (G__p_ifunc->allifunc == G__MAXIFUNC) {
      /* This case is used only when complicated template instantiation is done
       * during reading argument list 'f(vector<int> &x) { }' */
      G__p_ifunc->next = (struct G__ifunc_table_internal *)malloc(sizeof(struct G__ifunc_table_internal));
      memset(G__p_ifunc->next, 0, sizeof(struct G__ifunc_table_internal));
      G__p_ifunc->next->allifunc = 0;
      G__p_ifunc->next->next = (struct G__ifunc_table_internal *)NULL;
      G__p_ifunc->next->page = G__p_ifunc->page + 1;
      G__p_ifunc->next->tagnum = G__p_ifunc->tagnum;
      G__p_ifunc = G__p_ifunc->next;
      {
         int ix;
         for (ix = 0;ix < G__MAXIFUNC;ix++) {
            G__p_ifunc->funcname[ix] = (char*)NULL;
            G__p_ifunc->userparam[ix] = 0;
         }
      }
   }
   store_ifunc_tmp = G__p_ifunc;
   /* set funcname to G__p_ifunc */
   G__func_now = G__p_ifunc->allifunc;
   G__func_page = G__p_ifunc->page;
   func_now = G__func_now;
   G__FastAllocString funcname(G__LONGLINE);
   if ('~' == funcheader[0] && 0 == ifunc->hash[0]) {
      G__p_ifunc = ifunc;
      G__func_now = 0;
      func_now = 0;
      G__func_page = ifunc->page;
   }
   if ('*' == funcheader[0]) {
      if ('*' == funcheader[1]) {
         int numstar = 2;
         while ('*' == funcheader[numstar]) {
            ++numstar;
         }
         funcname = funcheader + numstar;
         if (isupper(G__var_type)) {
            switch (G__reftype) {
               case G__PARANORMAL:
                  G__reftype = G__PARAP2P2P;
                  break;
               default:
                  G__reftype += 2;
                  break;
            }
         }
         else {
            switch (G__reftype) {
               case G__PARANORMAL:
                  G__reftype = G__PARAP2P;
                  break;
               case G__PARAP2P:
                  G__reftype = G__PARAP2P2P;
                  break;
               default:
                  G__reftype += 1;
                  break;
            }
         }
         G__reftype += numstar - 2;
      }
      else {
         funcname = funcheader + 1;
         if (isupper(G__var_type)) {
            switch (G__reftype) {
               case G__PARANORMAL:
                  G__reftype = G__PARAP2P;
                  break;
               case G__PARAP2P:
                  G__reftype = G__PARAP2P2P;
                  break;
               default:
                  G__reftype += 1;
                  break;
            }
         }
      }
      G__var_type = toupper(G__var_type);
   }
   else {
      if (strncmp(funcheader, "operator ", 9) == 0) {
         char* oprtype = funcheader + 9;
         if (
            strcmp(oprtype, "char") == 0 ||
            strcmp(oprtype, "short") == 0 ||
            strcmp(oprtype, "int") == 0 ||
            strcmp(oprtype, "long") == 0 ||
            strcmp(oprtype, "unsigned char") == 0 ||
            strcmp(oprtype, "unsigned short") == 0 ||
            strcmp(oprtype, "unsigned int") == 0 ||
            strcmp(oprtype, "unsigned long") == 0 ||
            strcmp(oprtype, "float") == 0 ||
            strcmp(oprtype, "double") == 0
         ) {
            // Do nothing.
         }
         else {
            int oprtypenum;
            oprtype[strlen(oprtype)-1] = 0;
            oprtypenum = G__defined_typename(oprtype);
            if (
               -1 != oprtypenum &&
               -1 == G__newtype.tagnum[oprtypenum] &&
               -1 != G__newtype.parent_tagnum[oprtypenum]
            ) {
               const char* posEndType = oprtype;
               while (isalnum(*posEndType)
                      // NOTE this increases posEndType to skip '::'!
                      || (posEndType[0] == ':' && posEndType[1] == ':' && (++posEndType))
                      || *posEndType == '_')
                  ++posEndType;
               G__FastAllocString refbuf("");
               if (*posEndType) {
                  refbuf = posEndType;
               }
               funcheader.Replace(oprtype-funcheader(), G__type2string(G__newtype.type[oprtypenum] , -1, -1, G__newtype.reftype[oprtypenum], G__newtype.isconst[oprtypenum]));
               if (refbuf[0]) {
                  funcheader += refbuf;
               }
            }
            else {
               int oprtagnum = G__defined_tagname(oprtype, 2);
               if (oprtagnum > -1) {
                  const char* posEndType = oprtype;
                  bool isTemplate = (strchr(G__struct.name[oprtagnum], '<'));
                  while (isalnum(*posEndType)
                         || (posEndType[0] == ':' && posEndType[1] == ':' && (++posEndType))
                         || *posEndType == '_') {
                     ++posEndType;
                     if (isTemplate && *posEndType == '<') {
                        // also include "<T>" in type name
                        ++posEndType;
                        int templLevel = 1;
                        while (templLevel && *posEndType) {
                           if (*posEndType == '<') ++templLevel;
                           else if (*posEndType == '>') --templLevel;
                           ++posEndType;
                        }
                     }
                  }
                  G__FastAllocString refbuf("");
                  if (*posEndType) {
                     refbuf = posEndType;
                  }
                  funcheader.Replace(oprtype-funcheader(),G__fulltagname(oprtagnum, 0));
                  if (refbuf[0]) {
                     funcheader += refbuf;
                  }
               }
            }
            funcheader += "(";
         }
      }
      funcname = funcheader;
      if (
         (strstr(funcheader, ">>") != NULL && strchr(funcheader, '<') != NULL) ||
         (strstr(funcheader, "<<") != NULL && strchr(funcheader, '>') != NULL)
      ) {
         size_t pt1 = 0; // for funcheader;
         size_t pt2 = 0; // for funcname;
         if ((char*)NULL != strstr(funcheader, "operator<<") &&
               (char*)NULL != strchr(funcheader, '>')) {
            /* we might have operator< <> or operator< <double>
              or operator<< <> or operator<< <double>
              with the space missing */
            pt2 += strlen("operator<");
            pt1 += strlen("operator<");
            /*char *pt2 = G__p_ifunc->funcname[func_now] + strlen( "operator<" );*/
            if (funcname[pt2 + 1] == '<') {
               /* we have operator<< <...> */
               ++pt2;
               ++pt1;
            }
            funcname.Set( pt2, ' ' );
            ++pt2;
            funcname.Replace(pt2, funcheader() + pt1);
         }
         else if ((char*)NULL != strstr(funcname() + pt2, "operator>>") &&
                  (char*)NULL != strchr(funcname() + pt2, '<')) {
            /* we might have operator>><>  */
            /* we have nothing to do ... yet (we may have to do something
               for nested templates */
            pt2 += strlen("operator>>");
            pt1 += strlen("operator>>");
         }
         const char *next_pt1;
         while ((char*)NULL != (next_pt1 = strstr(funcheader() + pt1, ">>"))) {
            pt1 = next_pt1 - funcheader();
            size_t pt3 = strstr(funcname() + pt2, ">>") - funcname();
            ++pt3;
            funcname.Set( pt3, ' ');
            ++pt3;
            ++pt1;
            pt2 = pt3;
            funcname.Replace(pt3, funcheader() + pt1);
         }
      }
   }

   funcname[strlen(funcname) - 1] = 0; // remove trailing '('
   /******************************************************
    * conv<B>(x) -> conv<ns::B>(x)
    ******************************************************/
   G__rename_templatefunc(funcname);
   
   G__hash(funcname, G__p_ifunc->hash[func_now], iin2);
   G__paramfunc* param = G__p_ifunc->param[func_now][0];
   param->name = 0;
   /*************************************************************
    * check if the function is operator()(), if so, regenerate
    * hash value
    *************************************************************/
   if (
      (G__p_ifunc->hash[func_now] == G__HASH_OPERATOR) &&
      !strcmp(funcname, "operator")
   ) {
      funcname = "operator()";
      G__p_ifunc->hash[func_now] += '(' + ')';
   }
   size_t funcnamelen = strlen(funcname) + 1;
   G__p_ifunc->funcname[func_now] = (char*)malloc(funcnamelen);
   memcpy(G__p_ifunc->funcname[func_now], funcname(), funcnamelen);

   fgetpos(G__ifile.fp, &G__p_ifunc->entry[func_now].pos);
   G__p_ifunc->entry[func_now].p = (void*)G__ifile.fp;
   G__p_ifunc->entry[func_now].line_number = G__ifile.line_number;
   G__p_ifunc->entry[func_now].filenum = G__ifile.filenum;
#ifdef G__TRUEP2F
   G__p_ifunc->entry[func_now].tp2f = (void*)G__p_ifunc->funcname[func_now];
#endif // G__TRUEP2F
#ifdef G__ASM_FUNC
   G__p_ifunc->entry[func_now].size = 0;
#endif // G__ASM_FUNC
#ifdef G__ASM_WHOLEFUNC
   G__p_ifunc->entry[func_now].bytecode = (struct G__bytecodefunc*)NULL;
   G__p_ifunc->entry[func_now].bytecodestatus = G__BYTECODE_NOTYET;
#endif // G__ASM_WHOLEFUNC
   G__p_ifunc->pentry[func_now] = &G__p_ifunc->entry[func_now];
   if (-1 == G__p_ifunc->tagnum) {
      G__p_ifunc->globalcomp[func_now] = G__default_link ? G__globalcomp : G__NOLINK;
   }
   else {
      G__p_ifunc->globalcomp[func_now] = G__globalcomp;
   }
#ifdef G__FRIEND
   if (-1 == G__friendtagnum) {
      G__p_ifunc->friendtag[func_now] = (struct G__friendtag*)NULL;
   }
   else {
      G__p_ifunc->friendtag[func_now]
      = (struct G__friendtag*)malloc(sizeof(struct G__friendtag));
      G__p_ifunc->friendtag[func_now]->next = (struct G__friendtag*)NULL;
      G__p_ifunc->friendtag[func_now]->tagnum = G__friendtagnum;
   }
#endif // G__FRIEND
   /*************************************************************
    * set type struct and typedef information to G__ifile
    *************************************************************/
   if (
      G__def_struct_member &&
      G__def_tagnum != -1 &&
      !strcmp(G__struct.name[G__def_tagnum], G__p_ifunc->funcname[func_now])
   ) {
      /* constructor */
      /* illegular handling not to instaitiate temp object for return */
      G__p_ifunc->type[func_now] = 'i';
      G__p_ifunc->p_tagtable[func_now] = G__def_tagnum;
      G__p_ifunc->p_typetable[func_now] = G__typenum;
      G__struct.isctor[G__def_tagnum] = 1;
   }
   else {
      G__p_ifunc->type[func_now] = G__var_type;
      G__p_ifunc->p_tagtable[func_now] = G__tagnum;
      G__p_ifunc->p_typetable[func_now] = G__typenum;
   }
   G__p_ifunc->reftype[func_now] = G__reftype;
   G__p_ifunc->isconst[func_now] = (G__SIGNEDCHAR_T)G__constvar;
   G__p_ifunc->isexplicit[func_now] = (G__SIGNEDCHAR_T)G__isexplicit;
   G__isexplicit = 0;
   G__reftype = G__PARANORMAL;
   if (funcheader[0] == '~') {
      /* return type is void if destructor */
      G__p_ifunc->type[func_now] = 'y';
      G__p_ifunc->p_tagtable[func_now] = -1;
      G__p_ifunc->p_typetable[func_now] = -1;
   }
#ifndef G__NEWINHERIT
   G__p_ifunc->isinherit[func_now] = 0;
#endif // G__NEWINHERIT
   /*************************************************************
    * member access control
    *************************************************************/
   if (G__def_struct_member) {
      G__p_ifunc->access[func_now] = G__access;
   }
   else {
      G__p_ifunc->access[func_now] = G__PUBLIC;
   }
   G__p_ifunc->staticalloc[func_now] = (char) G__static_alloc;
   /*************************************************************
    * initiazlize baseoffset
    *************************************************************/
#ifndef G__NEWINHERIT
   G__p_ifunc->baseoffset[func_now] = 0;
   if (-1 != G__def_tagnum) G__p_ifunc->basetagnum[func_now] = G__def_tagnum;
   else G__p_ifunc->basetagnum[func_now] = G__tagdefining;
#endif // G__NEWINHERIT
   G__p_ifunc->isvirtual[func_now] = G__virtual;
   G__p_ifunc->ispurevirtual[func_now] = 0;
   /* for virtual function, allocate virtual identity member.
    * Set offset of the virtual identity member to
    * G__struct.virtual_offset[G__p_ifunc->basetagnum[func_now]].
    */
   if (
#ifdef G__NEWINHERIT
      G__virtual && -1 == G__struct.virtual_offset[G__tagdefining]
#else // G__NEWINHERIT
      G__virtual && -1 == G__struct.virtual_offset[G__p_ifunc->basetagnum[func_now]]
#endif // G__NEWINHERIT
   ) {
      store_tagnum = G__tagnum;
      store_typenum = G__typenum;
      store_type = G__var_type;
      G__tagnum = -1;
      G__typenum = -1;
      G__var_type = 'l';
      store_access = G__access;
#ifdef G__DEBUG2
      G__access = G__PUBLIC;
#else // G__DEBUG2
      G__access = G__PRIVATE;
#endif // G__DEBUG2
      G__FastAllocString vinfo("G__virtualinfo");
      G__letvariable(vinfo, G__null, &G__global, G__p_local);
      G__access = store_access;
      G__var_type = store_type;
      G__tagnum = store_tagnum;
      G__typenum = store_typenum;
#ifdef G__NEWINHERIT
      G__struct.virtual_offset[G__tagdefining] = G__struct.size[G__tagdefining] - G__LONGALLOC;
#else // G__NEWINHERIT
      G__struct.virtual_offset[G__p_ifunc->basetagnum[func_now]] = G__struct.size[G__p_ifunc->basetagnum[func_now]] - G__LONGALLOC;
#endif // G__NEWINHERIT
   }
   G__virtual = 0; /* this position is not very best */
   G__p_ifunc->comment[func_now].p.com = 0;
   G__p_ifunc->comment[func_now].filenum = -1;
   /*************************************************************
    * initialize virtual table index
    *  TODO, may need to this this in other places too, need investigation
    *************************************************************/
   G__p_ifunc->vtblindex[func_now] = -1;
   G__p_ifunc->vtblbasetagnum[func_now] = -1;
   /*************************************************************
    * initialize busy flag
    *************************************************************/
   G__p_ifunc->busy[func_now] = 0;
   /*************************************************************
    * store C++ or C
    *************************************************************/
   G__p_ifunc->iscpp[func_now] = (char)G__iscpp;
   /*****************************************************
    * to get type of function parameter
    *****************************************************/
   /* remember current file position
    *   func(   int   a   ,  double   b )
    *        ^
    *  if this is an ANSI stype header, the file will be rewinded
    */
   fgetpos(G__ifile.fp, &temppos);
   store_line_number = G__ifile.line_number;  /* bub fix 3 mar 1993 */
   /* Skip parameter field  'param,param,,,)'  until ')' is found */
   /* check if the header is written in ANSI format or not
    *   type func(param,param);
    *             ^
    *   func(   int   a   ,  double   b )
    *         -  -  - - - -
    */
   G__FastAllocString paraname(G__LONGLINE);
   cin = G__fgetname_template(paraname, 0, "<*&,()=");
   if (strlen(paraname) && isspace(cin)) {
      /* There was an argument and the parsing was stopped by a white
      * space rather than on of ",)*&<=", it is possible that
      * we have a namespace followed by '::' in which case we have
      * to grab more before stopping! */
      int namespace_tagnum;
      G__FastAllocString more(G__LONGLINE);

      namespace_tagnum = G__defined_tagname(paraname, 2);
      while ((((namespace_tagnum != -1)
               && (G__struct.type[namespace_tagnum] == 'n'))
              || (strcmp("std", paraname) == 0)
              || (paraname[strlen(paraname)-1] == ':'))
             && isspace(cin)) {
         cin = G__fgetname(more, 0, "<*&,)=");
         paraname += more;
         namespace_tagnum = G__defined_tagname(paraname, 2);
      }
   }
   if (paraname[0]) {
      if (strcmp("void", paraname) == 0) {
         if (isspace(cin)) cin = G__fgetspace();
         switch (cin) {
            case ',':
            case ')':
               G__p_ifunc->ansi[func_now] = 1;
               isvoid = 1;
               break;
            case '*':
            case '(':
               G__p_ifunc->ansi[func_now] = 1;
               isvoid = 0;
               break;
            default:
               G__genericerror("Syntax error");
               G__p_ifunc->ansi[func_now] = 0;
               isvoid = 1;
               break;
         }
      }
      else if (strcmp("register", paraname) == 0) {
         G__p_ifunc->ansi[func_now] = 1;
         isvoid = 0;
      }
      else if (G__istypename(paraname) || strchr(paraname, '[')
               || -1 != G__friendtagnum
              ) {
         G__p_ifunc->ansi[func_now] = 1;
         isvoid = 0;
      }
      else if (!strcmp(paraname, "...")) {
         G__p_ifunc->ansi[func_now] = 1;
         isvoid = 0;
      }
      else {
         if (G__def_struct_member) G__genericerror("Syntax error");
         if (G__globalcomp < G__NOLINK && !G__nonansi_func
#ifdef G__ROOT
               && strncmp(funcheader, "ClassDef", 8) != 0
#endif // G__ROOT
            ) {
            if (G__dispmsg >= G__DISPWARN) {
               G__fprinterr(G__serr, "Warning: Unknown type %s in function argument"
                            , paraname());
               G__printlinenum();
            }
         }
         G__p_ifunc->ansi[func_now] = 0;
         isvoid = 0;
      }
   }
   else {
      if (G__def_struct_member || G__iscpp) G__p_ifunc->ansi[func_now] = 1;
      else                     G__p_ifunc->ansi[func_now] = 0;
      isvoid = 1;
   }
   if (')' != cin) cin = G__fignorestream(")");
   G__static_alloc = 0;
   /*****************************************************
    * to get type of function parameter
    *****************************************************/
   /****************************************************************
    * If ANSI style header, rewind file position to
    *       func(int a ,double b )   ANSI
    *            ^
    * and check type of paramaters and store it into G__ifunc
    ****************************************************************/
   if (G__p_ifunc->ansi[func_now]) {
      // -- ANSI style function header
      if (isvoid || '~' == funcheader[0]) {
         G__p_ifunc->para_nu[func_now] = 0;
         G__p_ifunc->param[func_now][0]->def = (char*)NULL;
         G__p_ifunc->param[func_now][0]->pdefault = (G__value*)NULL;
      }
      else {
         if (G__dispsource) G__disp_mask = 1000;
         fsetpos(G__ifile.fp, &temppos);
         G__ifile.line_number = store_line_number;
         ++G__p_ifunc->allifunc;
         G__readansiproto(G__p_ifunc, func_now);
         if (store_ifunc_tmp != G__p_ifunc || func_now != G__p_ifunc->allifunc) {
            /* This is the normal case. This block is skipped only when
             * compicated template instantiation is done during reading
             * argument list 'f(vector<int> &x) { }' */
            // Or we we autoloaded a library while reading this func prototype.
            // In that case we might have added tons of ifunc tables - a simple
            //--G__p_ifunc->allifunc;
            // is not sufficient.
            // Instead, switch store_ifunc_tmp with the last one of its chain.
   
            G__ifunc_table_internal* ifunc_last = store_ifunc_tmp;
            while (ifunc_last->next && ifunc_last->next->next)
               ifunc_last = ifunc_last->next;
            if (ifunc_last != store_ifunc_tmp) {
               // only supported for G__MAX_IFUNC==1 for now.
               G__ASSERT(G__MAXIFUNC==1);
   
               // different, so switch
               G__ifunc_table_internal tmp = *store_ifunc_tmp;
               *store_ifunc_tmp = *ifunc_last;
               *ifunc_last = tmp;
   
               // fix link
               ifunc_last->next = store_ifunc_tmp->next;
               store_ifunc_tmp->next = tmp.next;
   
               // fix pageno
               ifunc_last->page = store_ifunc_tmp->page;
               store_ifunc_tmp->page = tmp.page;
   
               // 09-08-07
               ifunc_last->page_base = store_ifunc_tmp->page_base;
               store_ifunc_tmp->page_base = tmp.page_base;

               // fix pentry
               if (ifunc_last->pentry[0] == &store_ifunc_tmp->entry[0])
                  ifunc_last->pentry[0] = &ifunc_last->entry[0];
               if (store_ifunc_tmp->pentry[0] == &ifunc_last->entry[0])
                  store_ifunc_tmp->pentry[0] = &store_ifunc_tmp->entry[0];
   
               // for matching ++allifunc below:
               store_ifunc_tmp = ifunc_last;
   
               // prevent overzealous d'tor from destroying 
               // our precious default parameters:
               tmp.param[0].fparams = 0;
            }
            G__p_ifunc = ifunc_last;
            --G__p_ifunc->allifunc;
         }
         cin = ')';
         if (G__dispsource) G__disp_mask = 0;
      }
   }
   else {
      // -- K & R style function header
      if (isvoid) {
         G__p_ifunc->para_nu[func_now] = 0;
      }
      else {
         G__p_ifunc->para_nu[func_now] = -1;
      }
   }
   // Set G__no_exec to skip function body.
   // This statement can be placed after endif.
   G__no_exec = 1;
   // skip space characters after
   //   func(param)      int a; {...}
   //              ^
   if (G__isfuncreturnp2f) {
      /* function returning pointer to function
       *   type (*func(param1))(param2)  { } or;
       *                      ^ -----> ^   */
      cin = G__fignorestream(")");
      cin = G__fignorestream("(");
      cin = G__fignorestream(")");
   }
   cin = G__fgetstream_template(paraname, 0, ",;{(");
   if ('(' == cin) {
      int len = strlen(paraname);
      paraname.Resize(len + 10);
      paraname[len++] = cin;
      cin = G__fgetstream(paraname, len, ")");
      len = strlen(paraname);
      paraname.Resize(len + 10);
      paraname[len++] = cin;
      cin = G__fgetstream_template(paraname, len, ",;{");
   }
   // If header ignore following headers, else read function body.
   if ((paraname[0] == '\0'
#ifndef G__OLDIMPLEMETATION817
         || ((strncmp(paraname, "throw", 5) == 0
              || strncmp(paraname, "const throw", 11) == 0
              || strncmp(paraname, "_attribute_", 11) == 0) && 0 == strchr(paraname, '='))
#endif // G__OLDIMPLEMETATION817
       ) && ((cin == ',') || (cin == ';'))
         && strncmp(funcheader, "ClassDef", 8) != 0
      ) {
      if (cin == ',') {
         /* ignore other prototypes */
         G__fignorestream(";");
         if (G__globalcomp < G__NOLINK)
            G__genericerror("Limitation: Items in header must be separately specified");
      }
      /* entry fp = NULL means this is header */
      G__p_ifunc->entry[func_now].p = (void*)NULL;
      G__p_ifunc->entry[func_now].line_number = -1;
      G__p_ifunc->ispurevirtual[func_now] = 0;
      /* Key the class comment off of DeclFileLine rather than ClassDef
       * because ClassDef is removed by a preprocessor */
      if (G__fons_comment && G__def_struct_member &&
            (strncmp(G__p_ifunc->funcname[func_now], "DeclFileLine", 12) == 0
             || strncmp(G__p_ifunc->funcname[func_now], "DeclFileLine(", 13) == 0
             || strncmp(G__p_ifunc->funcname[func_now], "DeclFileLine", 12) == 0
             || strncmp(G__p_ifunc->funcname[func_now], "DeclFileLine(", 13) == 0
            )) {
         G__fsetcomment(&G__struct.comment[G__tagdefining]);
      }

      if (0 == strncmp(paraname, "const", 5))
         G__p_ifunc->isconst[func_now] |= G__CONSTFUNC;

      // 07-11-07
      // "throwness" is needed to declare the prototypes
      if (0 == strncmp(paraname, "throw", 5) || strncmp(paraname, "const throw", 11) == 0)
         G__p_ifunc->isconst[func_now] |= G__FUNCTHROW;
   }
   else if (
      strncmp(paraname, "=", 1) == 0 ||
      strncmp(paraname, "const =", 7) == 0 ||
      strncmp(paraname, "const=", 6) == 0
#ifndef G__OLDIMPLEMETATION817
      || (
         (
            strncmp(paraname, "throw", 5) == 0 ||
            strncmp(paraname, "const throw", 11) == 0 ||
            strncmp(paraname, "_attribute_", 11) == 0
         ) &&
         0 != strchr(paraname, '=')
      )
#endif // G__OLDIMPLEMETATION817
   ) {
      char *p;
      p = strchr(paraname, '=');
      // FIXME: We need to check for a textual zero, not eval an expression.
      if (0 != G__int(G__getexpr(p + 1))) {
         G__genericerror("Error: invalid pure virtual function initializer");
      }
      /* this is ANSI style func proto without param name */
      if (0 == G__p_ifunc->ansi[func_now]) G__p_ifunc->ansi[func_now] = 1;
      if (cin == ',') {
         /* ignore other prototypes */
         G__fignorestream(";");
         if (G__globalcomp < G__NOLINK)
            G__genericerror(
               "Limitation: Items in header must be separately specified");
      }
      /* entry fp = NULL means this is header */
      G__p_ifunc->entry[func_now].p = (void*)NULL;
      G__p_ifunc->entry[func_now].line_number = -1;
      G__p_ifunc->ispurevirtual[func_now] = 1;
      if (G__tagdefining >= 0) ++G__struct.isabstract[G__tagdefining];
      if ('~' == G__p_ifunc->funcname[func_now][0]) {
         if (G__dispmsg >= G__DISPWARN) {
            G__printlinenum();
            G__fprinterr(G__serr, "Warning: Pure virtual destructor may cause problem. Define as 'virtual %s() { }'\n"
                         , G__p_ifunc->funcname[func_now]
                        );
         }
      }
      if (0 == strncmp(paraname, "const", 5))
         G__p_ifunc->isconst[func_now] |= G__CONSTFUNC;

      // 07-11-07
      // "throwness" is needed to declare the prototypes
      if (0 == strncmp(paraname, "throw", 5) || strncmp(paraname, "const throw", 11) == 0)
         G__p_ifunc->isconst[func_now] |= G__FUNCTHROW;
   }
   else if (strcmp(paraname, "const") == 0 ||
            strcmp(paraname, "const ") == 0
           ) {
      // This is an ANSI style function prototype without a parameter name.
      if (!G__p_ifunc->ansi[func_now]) {
         G__p_ifunc->ansi[func_now] = 1;
      }
      if (cin == ',') {/* ignore other prototypes */
         G__fignorestream(";");
         if (G__globalcomp < G__NOLINK)
            G__genericerror(
               "Limitation: Items in header must be separately specified");
      }
      if ('{' == cin) {
         // It is possible that this is a function body.
         fseek(G__ifile.fp, -1, SEEK_CUR);
         if (G__dispsource) {
            G__disp_mask = 1;
         }
         dobody = 1;
      }
      else {
         // Entry fp = NULL means this is header.
         G__p_ifunc->entry[func_now].p = 0;
         G__p_ifunc->entry[func_now].line_number = -1;
      }
      G__p_ifunc->ispurevirtual[func_now] = 0;
      G__p_ifunc->isconst[func_now] |= G__CONSTFUNC;
   }
   else if (
      G__def_struct_member &&
      (
         '}' == cin ||
         (';' == cin && '\0' != paraname[0] && ':' != paraname[0]) ||
         (';' == cin && strncmp(funcheader, "ClassDef", 8) == 0)
      )
   ) {
      /* Function macro as member declaration */
      /* restore file position
       *   func(   int   a   ,  double   b )
       *        ^  <------------------------+
       */
      fsetpos(G__ifile.fp, &temppos);
      G__ifile.line_number = store_line_number;

      if (G__dispsource) G__disp_mask = 1000;
      paraname = funcheader;
      cin = G__fgetstream(paraname, strlen(paraname), ")");
      iin = strlen(paraname);
      paraname += ")";
      if (G__dispsource) G__disp_mask = 0;

      G__no_exec = 0; /* must be set to 1 again after return */
      G__func_now = -1;
      G__p_ifunc = store_ifunc;

      G__execfuncmacro(paraname, &iin);
      if (!iin) {
         G__genericerror("Error: unrecognized language construct");
      }
      else if (G__fons_comment && G__def_struct_member &&
               (strncmp(paraname, "ClassDef", 8) == 0 ||
                strncmp(paraname, "ClassDef(", 9) == 0 ||
                strncmp(paraname, "ClassDefT(", 10) == 0)
              ) {
         G__fsetcomment(&G__struct.comment[G__tagdefining]);
      }
      return;
   }
   else {
      // 03-12-07
      // Why wasn't the const here? can't a function be const and be implemented
      // in the declaration??
      if (0 == strncmp(paraname, "const", 5))
         G__p_ifunc->isconst[func_now] |= G__CONSTFUNC;

      // 07-11-07
      // "throwness" is needed to declare the prototypes
      if (0 == strncmp(paraname, "throw", 5) || strncmp(paraname, "const throw", 11) == 0)
         G__p_ifunc->isconst[func_now] |= G__FUNCTHROW;

      /* Body of the function, skip until
       * 'func(param)  type param;  {...} '
       *                             ^
       * and rewind file to just before the '{...}'
       */
      if (
         G__HASH_MAIN == G__p_ifunc->hash[func_now] &&
         !strcmp(G__p_ifunc->funcname[func_now], "main") &&
         -1 == G__def_tagnum
      ) {
         G__ismain = G__MAINEXIST;
      }
      // Following part is needed to detect inline new/delete in header.
      if (G__CPPLINK == G__globalcomp || R__CPPLINK == G__globalcomp) {
         if (strcmp(G__p_ifunc->funcname[func_now], "operator new") == 0 &&
               2 == G__p_ifunc->para_nu[func_now] &&
               0 == (G__is_operator_newdelete&G__MASK_OPERATOR_NEW))
            G__is_operator_newdelete |= G__IS_OPERATOR_NEW;
         if (strcmp(G__p_ifunc->funcname[func_now], "operator delete") == 0 &&
               0 == (G__is_operator_newdelete&G__MASK_OPERATOR_DELETE))
            G__is_operator_newdelete |= G__IS_OPERATOR_DELETE;
      }
      if (':' == paraname[0] && 0 == G__p_ifunc->ansi[func_now]) {
         G__p_ifunc->ansi[func_now] = 1;
      }
      if (cin != '{') {
         G__fignorestream("{");
      }
      fseek(G__ifile.fp, -1, SEEK_CUR);
      if (G__dispsource) {
         G__disp_mask = 1;
      }
      // Skip body of the function surrounded by '{' '}'.
      // G__exec_statement() does the job.
      G__p_ifunc->ispurevirtual[func_now] = 0;

      dobody = 1;
   }
   if (G__nonansi_func) {
      G__p_ifunc->ansi[func_now] = 0;
   }
#ifdef G__DETECT_NEWDEL
   /****************************************************************
    * operator new(size_t,void*) , operator delete(void*) detection
    * This is only needed for Linux g++
    ****************************************************************/
   if (G__CPPLINK == G__globalcomp || R__CPPLINK == G__globalcomp) {
      if (strcmp(G__p_ifunc->funcname[func_now], "operator new") == 0 &&
            2 == G__p_ifunc->para_nu[func_now] &&
            0 == (G__is_operator_newdelete&G__MASK_OPERATOR_NEW))
         G__is_operator_newdelete |= G__IS_OPERATOR_NEW;
      if (strcmp(G__p_ifunc->funcname[func_now], "operator delete") == 0 &&
            0 == (G__is_operator_newdelete&G__MASK_OPERATOR_DELETE))
         G__is_operator_newdelete |= G__IS_OPERATOR_DELETE;
   }
#endif // G__DETECT_NEWDEL
   if (
      (
         !strcmp(G__p_ifunc->funcname[func_now], "operator delete") ||
         !strcmp(G__p_ifunc->funcname[func_now], "operator delete[]")
      ) &&
      (G__p_ifunc->tagnum != -1)
   ) {
      G__p_ifunc->staticalloc[func_now] = 1;
   }
   /****************************************************************
    * Set constructor,copy constructor, destructor, operator= flags
    ****************************************************************/
   if (G__def_struct_member && G__def_tagnum != -1) {
      if ('~' == G__p_ifunc->funcname[func_now][0]) {
         /* Destructor */
         G__struct.funcs[G__def_tagnum] |= G__HAS_DESTRUCTOR;
      }
      else if (strcmp(G__struct.name[G__def_tagnum]
                      , G__p_ifunc->funcname[func_now]) == 0) {
         if (0 == G__p_ifunc->para_nu[func_now] ||
               G__p_ifunc->param[func_now][0]->pdefault) {
            /* Default constructor */
            G__struct.funcs[G__def_tagnum] |= G__HAS_DEFAULTCONSTRUCTOR;
         }
         else if ((1 == G__p_ifunc->para_nu[func_now] ||
                   G__p_ifunc->param[func_now][1]->pdefault) &&
                  G__def_tagnum == G__p_ifunc->param[func_now][0]->p_tagtable &&
                  G__p_ifunc->param[func_now][0]->reftype) {
            /* Copy constructor */
            G__struct.funcs[G__def_tagnum] |= G__HAS_COPYCONSTRUCTOR;
         }
         else {
            G__struct.funcs[G__def_tagnum] |= G__HAS_XCONSTRUCTOR;
         }
      }
      else if (strcmp("operator=", G__p_ifunc->funcname[func_now]) == 0) {
         /* operator= */
         G__struct.funcs[G__def_tagnum] |= G__HAS_ASSIGNMENTOPERATOR;
      }
      else if (!strcmp("operator new", G__p_ifunc->funcname[func_now])) {
         if (G__p_ifunc->para_nu[func_now] == 1) {
            G__struct.funcs[G__def_tagnum] |= G__HAS_OPERATORNEW1ARG;
         }
         else if (G__p_ifunc->para_nu[func_now] == 2) {
            G__struct.funcs[G__def_tagnum] |= G__HAS_OPERATORNEW2ARG;
         }
      }
      else if (strcmp("operator delete", G__p_ifunc->funcname[func_now]) == 0) {
         G__struct.funcs[G__def_tagnum] |= G__HAS_OPERATORDELETE;
      }
   }
   /****************************************************************
    * if same function already exists, then copy entry
    * else if body exists or ansi header, then increment ifunc
    ****************************************************************/
   ifunc = G__ifunc_exist(G__p_ifunc, func_now , ifunc, &iexist, 0xffff);
   if (ifunc == G__p_ifunc) {
      ifunc = 0;
   }
   if (G__ifile.filenum < G__nfile) {
      if (ifunc) {
#ifdef G__FRIEND
         if (G__p_ifunc->friendtag[func_now]) {
            if (ifunc->friendtag[iexist]) {
               friendtag = ifunc->friendtag[iexist];
               while (friendtag->next) friendtag = friendtag->next;
               friendtag->next = G__p_ifunc->friendtag[func_now];
            }
            else {
               ifunc->friendtag[iexist] = G__p_ifunc->friendtag[func_now];
            }
         }
#endif // G__FRIEND
         if (
            ((FILE*)G__p_ifunc->entry[func_now].p != (FILE*)NULL)
            /* C++ precompiled member function must not be overridden  */
            && (0 == G__def_struct_member ||
                G__CPPLINK != G__struct.iscpplink[G__def_tagnum])
         ) {
            ifunc->ansi[iexist] = G__p_ifunc->ansi[func_now];
            if (-1 == G__p_ifunc->para_nu[func_now]) paranu = 0;
            else paranu = ifunc->para_nu[iexist];
            if (0 == ifunc->ansi[iexist])
               ifunc->para_nu[iexist] = G__p_ifunc->para_nu[func_now];
            ifunc->type[iexist] = G__p_ifunc->type[func_now];
            ifunc->p_tagtable[iexist] = G__p_ifunc->p_tagtable[func_now];
            ifunc->p_typetable[iexist] = G__p_ifunc->p_typetable[func_now];
            ifunc->reftype[iexist] = G__p_ifunc->reftype[func_now];
            ifunc->isconst[iexist] |= G__p_ifunc->isconst[func_now];
            ifunc->isexplicit[iexist] |= G__p_ifunc->isexplicit[func_now];

            for (iin = 0;iin < paranu;iin++) {
               ifunc->param[iexist][iin]->reftype
               = G__p_ifunc->param[func_now][iin]->reftype;
               ifunc->param[iexist][iin]->p_typetable
               = G__p_ifunc->param[func_now][iin]->p_typetable;
               if (G__p_ifunc->param[func_now][iin]->pdefault) {
                  G__genericerror("Error: Redefinition of default argument");
                  if (-1 != (long)G__p_ifunc->param[func_now][iin]->pdefault)
                     free((void*)G__p_ifunc->param[func_now][iin]->pdefault);
                  free((void*)G__p_ifunc->param[func_now][iin]->def);
               }
               G__p_ifunc->param[func_now][iin]->pdefault = (G__value*)NULL;
               G__p_ifunc->param[func_now][iin]->def = (char*)NULL;
               if (ifunc->param[iexist][iin]->name) {
                  if (G__p_ifunc->param[func_now][iin]->name) {
                     if (dobody && 0 != strcmp(ifunc->param[iexist][iin]->name
                                               , G__p_ifunc->param[func_now][iin]->name)) {
                        free((void*)ifunc->param[iexist][iin]->name);
                        ifunc->param[iexist][iin]->name
                        = G__p_ifunc->param[func_now][iin]->name;
                     }
                     else {
                        free((void*)G__p_ifunc->param[func_now][iin]->name);
                     }
                     G__p_ifunc->param[func_now][iin]->name = (char*)NULL;
                  }
               }
               else {
                  ifunc->param[iexist][iin]->name = G__p_ifunc->param[func_now][iin]->name;
                  G__p_ifunc->param[func_now][iin]->name = (char*)NULL;
               }
            }
            iin = paranu;
            if (iin < 0) iin = 0;
            for (; iin < G__p_ifunc->para_nu[func_now]; ++iin) {
               if (G__p_ifunc->param[func_now][iin]->pdefault) {
                  if (-1 != (long)G__p_ifunc->param[func_now][iin]->pdefault)
                     free((void*)G__p_ifunc->param[func_now][iin]->pdefault);
                  free((void*)G__p_ifunc->param[func_now][iin]->def);
               }
               G__p_ifunc->param[func_now][iin]->pdefault=(G__value*)NULL;
               G__p_ifunc->param[func_now][iin]->def=(char*)NULL;
               if (G__p_ifunc->param[func_now][iin]->name) {
                  free ((void*)G__p_ifunc->param[func_now][iin]->name);
                  G__p_ifunc->param[func_now][iin]->name = (char*)NULL;
               }
            }
            ifunc->entry[iexist] = G__p_ifunc->entry[func_now];
            /* The copy in previous get the wrong tp2f ... let's restore it */
            ifunc->entry[iexist].tp2f = (void*)ifunc->funcname[iexist];
            ifunc->pentry[iexist] = &ifunc->entry[iexist];
            if (1 == ifunc->ispurevirtual[iexist]) {
               ifunc->ispurevirtual[iexist] = G__p_ifunc->ispurevirtual[func_now];
               if (G__tagdefining >= 0) --G__struct.isabstract[G__tagdefining];
            }
            else if (1 == G__p_ifunc->ispurevirtual[func_now]) {
               ifunc->ispurevirtual[iexist] = G__p_ifunc->ispurevirtual[func_now];
            }
            if ((ifunc != G__p_ifunc || iexist != func_now) &&
                  G__p_ifunc->funcname[func_now]) {
               free((void*)G__p_ifunc->funcname[func_now]);
               G__p_ifunc->funcname[func_now] = (char*)NULL;
            }
         } /* of if(G__p_ifunc->entry[func_now].p) */
         else {
            /* Entry not used, must free allocated default argument buffer */
            if (1 == G__p_ifunc->ispurevirtual[func_now]) {
               if (G__tagdefining >= 0) --G__struct.isabstract[G__tagdefining];
            }
            paranu = G__p_ifunc->para_nu[func_now];
            for (iin = 0;iin < paranu;iin++) {
               if (G__p_ifunc->param[func_now][iin]->name) {
                  free((void*)G__p_ifunc->param[func_now][iin]->name);
                  G__p_ifunc->param[func_now][iin]->name = (char*)NULL;
               }
               if (G__p_ifunc->param[func_now][iin]->pdefault &&
                     (&G__default_parameter) != G__p_ifunc->param[func_now][iin]->pdefault) {
                  free((void*)G__p_ifunc->param[func_now][iin]->pdefault);
                  G__p_ifunc->param[func_now][iin]->pdefault = (G__value*)NULL;
                  free((void*)G__p_ifunc->param[func_now][iin]->def);
                  G__p_ifunc->param[func_now][iin]->def = (char*)NULL;
               }
            }
            if ((ifunc != G__p_ifunc || iexist != func_now) &&
                  G__p_ifunc->funcname[func_now]) {
               free((void*)G__p_ifunc->funcname[func_now]);
               G__p_ifunc->funcname[func_now] = (char*)NULL;
            }
         }
         G__func_page = ifunc->page;
         G__func_now = iexist;
         G__p_ifunc = ifunc;
      } /* of if(ifunc) */
      else if ((G__p_ifunc->entry[func_now].p || G__p_ifunc->ansi[func_now] ||
                G__nonansi_func ||
                G__globalcomp < G__NOLINK || G__p_ifunc->friendtag[func_now])
               /* This block is skipped only when compicated template
                * instantiation is done during reading argument list
                * 'f(vector<int> &x) { }' */
               /* with 1706, do not skip this block with template instantiation
                * in function argument. Do not know exactly why... */
               && (store_ifunc_tmp == G__p_ifunc && func_now == G__p_ifunc->allifunc)
               && '~' != funcheader[0]
              ) {
         /* increment allifunc */
         ++G__p_ifunc->allifunc;

         /* Allocate and initialize function table list if needed */
         if (G__p_ifunc->allifunc == G__MAXIFUNC) {

            G__p_ifunc->next = (struct G__ifunc_table_internal *)malloc(sizeof(struct G__ifunc_table_internal));
            memset(G__p_ifunc->next, 0, sizeof(struct G__ifunc_table_internal));
            G__p_ifunc->next->allifunc = 0;
            G__p_ifunc->next->next = (struct G__ifunc_table_internal *)NULL;
            G__p_ifunc->next->page = G__p_ifunc->page + 1;
            G__p_ifunc->next->page_base = 0; // 09-08-07 (index of this func in the base class)
            {
               //int i,j;
               //for (i = 0; i < G__MAXIFUNC; i++) {
               //for (j = 0; j < G__MAXFUNCPARA; j++)
               //  G__p_ifunc->next->param[i][j]->p_tagtable = 0;
               //}
            }
#ifdef G__NEWINHERIT
            G__p_ifunc->next->tagnum = G__p_ifunc->tagnum;
#endif // G__NEWINHERIT
            {
               int ix;
               for (ix = 0;ix < G__MAXIFUNC;ix++) {
                  G__p_ifunc->next->funcname[ix] = (char*)NULL;
                  G__p_ifunc->next->userparam[ix] = 0;
               }
            }
         }
      } /* if(ifunc) */
      /* else: default parameter does not exist in K&R style
       * no need to free default parameter buffer */
   } /* of G__ifile.filenum<G__nfile */
   else {
      G__fprinterr(G__serr, "Limitation: Function can not be defined in a command line or a tempfile\n");
      G__genericerror("You need to write it in a source file");
   }
   if (dobody) {
      store_def_struct_member = G__def_struct_member;
      G__def_struct_member = 0;
      int brace_level = 0;
      G__exec_statement(&brace_level);
      G__def_struct_member = store_def_struct_member;

#ifdef G__ASM_FUNC
      if (ifunc) {
         ifunc->pentry[iexist]->size =
            G__ifile.line_number - ifunc->pentry[iexist]->line_number + 1;
      }
      else {
         G__p_ifunc->pentry[func_now]->size =
            G__ifile.line_number - G__p_ifunc->pentry[func_now]->line_number + 1;
      }
#endif // G__ASM_FUNC
#ifdef G__ASM_WHOLEFUNC
      /***************************************************************
      * compile as bytecode at load time if -O10 or #pragma bytecode
      ***************************************************************/
      if (G__asm_loopcompile >= 10
         ) {
         if (ifunc) G__compile_bytecode(G__get_ifunc_ref(ifunc), iexist);
         else G__compile_bytecode(G__get_ifunc_ref(G__p_ifunc), func_now);
      }
#endif // G__ASM_WHOLEFUNC
   }

   if (G__GetShlHandle()) {
      void *shlp2f = G__FindSymbol(G__p_ifunc, func_now);
      if (shlp2f) {
         G__p_ifunc->pentry[func_now]->tp2f = shlp2f;
         G__p_ifunc->pentry[func_now]->p = (void*)G__DLL_direct_globalfunc;
         G__p_ifunc->pentry[func_now]->filenum = G__GetShlFilenum();
         G__p_ifunc->pentry[func_now]->size = -1;
         G__p_ifunc->pentry[func_now]->line_number = -1;
      }
   }

   if (G__fons_comment && G__def_struct_member) {
      if ((ifunc && (strncmp(ifunc->funcname[iexist], "ClassDef", 8) == 0 ||
                     strncmp(ifunc->funcname[iexist], "ClassDef(", 9) == 0 ||
                     strncmp(ifunc->funcname[iexist], "ClassDefT(", 10) == 0 ||
                     strncmp(ifunc->funcname[iexist], "DeclFileLine", 12) == 0 ||
                     strncmp(ifunc->funcname[iexist], "DeclFileLine(", 13) == 0)) ||
            (!ifunc && (strncmp(G__p_ifunc->funcname[func_now], "ClassDef", 8) == 0 ||
                        strncmp(G__p_ifunc->funcname[func_now], "ClassDef(", 9) == 0 ||
                        strncmp(G__p_ifunc->funcname[func_now], "ClassDefT(", 10) == 0 ||
                        strncmp(G__p_ifunc->funcname[func_now], "DeclFileLine", 12) == 0 ||
                        strncmp(G__p_ifunc->funcname[func_now], "DeclFileLine(", 13) == 0))) {
         G__fsetcomment(&G__struct.comment[G__tagdefining]);
      }
      else {
         if (ifunc) G__fsetcomment(&ifunc->comment[iexist]);
         else      G__fsetcomment(&G__p_ifunc->comment[func_now]);
      }
   }

#ifdef G__NEWINHERIT
   /***********************************************************************
   * If this is a non-pure virtual member function declaration, decrement
   * isabstract flag in G__struct.
   ***********************************************************************/
   if (-1 != G__tagdefining && !ifunc) {
      baseclass = G__struct.baseclass[G__tagdefining];
      for (basen = 0;basen < baseclass->basen;basen++) {
         G__incsetup_memfunc(baseclass->herit[basen]->basetagnum);
         ifunc = G__struct.memfunc[baseclass->herit[basen]->basetagnum];
         ifunc = G__ifunc_exist(G__p_ifunc, func_now , ifunc, &iexist, G__CONSTFUNC);
         if (ifunc) {
            if (ifunc->ispurevirtual[iexist] &&
                  G__struct.isabstract[G__tagdefining]) {
               --G__struct.isabstract[G__tagdefining];
            }
            G__p_ifunc->isvirtual[func_now] |= ifunc->isvirtual[iexist];
            break; /* revived by Scott Snyder */
         }
      }
   }
#endif // G__NEWINHERIT

   G__p_ifunc->page_base = 0;

   /* finishing up */
   G__no_exec = 0;
   G__func_now = -1;
   G__p_ifunc = store_ifunc;

   return;
}

extern "C" {

//______________________________________________________________________________
static int G__readansiproto(G__ifunc_table_internal* ifunc, int func_now)
{
   //  func(type , type* , ...)
   //       ^
   ifunc->ansi[func_now] = 1;
   int iin = 0;
   int c = 0;
   for (; c != ')'; ++iin) {
      if (iin == G__MAXFUNCPARA) {
         G__fprinterr(G__serr, "Limitation: cint can not accept more than %d function arguments", G__MAXFUNCPARA);
         G__printlinenum();
         G__fignorestream(")");
         return 1;
      }
      G__paramfunc* param = ifunc->param[func_now][iin];
      param->isconst = G__VARIABLE;
      G__FastAllocString buf(G__LONGLINE);
      buf[0] = '\0';
      // Get first keyword, id, or separator of the type specification.
      c = G__fgetname_template(buf, 0, "&*[(=,)");
      if (strlen(buf) && isspace(c)) {
         // -- There was an argument and the parsing stopped at white space.
         // It is possible that we have a qualified name, check.
         // FIXME: A classname should be allowed here as well, the parameter type could be a nested type.
         int namespace_tagnum = G__defined_tagname(buf, 2);
         while (
            isspace(c) &&
            (
               (buf[strlen(buf)-1] == ':') ||
               ((namespace_tagnum != -1) && (G__struct.type[namespace_tagnum] == 'n')) ||
               !strcmp("std", buf)
            )
         ) {
            G__FastAllocString more(G__LONGLINE);
            c = G__fgetname(more, 0, "&*[(=,)");
            buf += more;
            namespace_tagnum = G__defined_tagname(buf, 2);
         }
      }
      //fprintf(stderr, "G__readansiproto: buf: '%s'\n", buf);
      // Check if we have reached an ellipsis (must be at end).
      if (!strcmp(buf, "...")) {
         ifunc->ansi[func_now] = 2;
         break;
      }
      // Check for and consume const, volatile, auto, register and typename qualifiers.
      while (
         !strcmp(buf, "const") ||
         !strcmp(buf, "volatile") ||
         !strcmp(buf, "auto") ||
         !strcmp(buf, "register") ||
         (G__iscpp && !strcmp(buf, "typename"))
      ) {
         if (!strcmp(buf, "const")) {
            ifunc->param[func_now][iin]->isconst |= G__CONSTVAR;
         }
         c = G__fgetname_template(buf, 0, "&*[(=,)");
         //fprintf(stderr, "G__readansiproto: buf: '%s'\n", buf);
      }
      //
      //  Determine type.
      //
      char type = '\0';
      int tagnum = -1;
      int typenum = -1;
      int ptrcnt = 0; // number of pointers seen (pointer count)
      int isunsigned = 0; // unsigned seen flag and offset for type code
      //
      //  Process first keyword of type specifier
      //  (most type specifiers have only one keyword).
      //
      {
         // Partially handle unsigned and sigend keywords here.  Also do some integral promotions.
         if (!strcmp(buf, "unsigned") || !strcmp(buf, "signed")) {
            if (buf[0] == 'u') {
               isunsigned = -1;
            }
            else {
               isunsigned = 0;
            }
            switch (c) {
               case ',':
               case ')':
               case '&':
               case '[':
               case '(':
               case '=':
               case '*':
                  buf = "int";
                  //fprintf(stderr, "G__readansiproto: buf: '%s'\n", buf);
                  break;
               default:
                  if (isspace(c)) {
                     c = G__fgetname(buf, 0, ",)&*[(="); // FIXME: Change to G__getname_template???
                     //fprintf(stderr, "G__readansiproto: buf: '%s'\n", buf);
                  }
                  else {
                     fpos_t pos;
                     fgetpos(G__ifile.fp, &pos);
                     int store_line = G__ifile.line_number;
                     c = G__fgetname(buf, 0, ",)&*[(=");
                     //fprintf(stderr, "G__readansiproto: buf: '%s'\n", buf);
                     if (strcmp(buf, "short") && strcmp(buf, "int") && strcmp(buf, "long")) {
                        G__ifile.line_number = store_line;
                        fsetpos(G__ifile.fp, &pos);
                        buf = "int";
                        //fprintf(stderr, "G__readansiproto: buf: '%s'\n", buf);
                        c = ' ';
                     }
                  }
                  break;
            }
         }
         if (!strcmp(buf, "class")) {
            c = G__fgetname_template(buf, 0, ",)&*[(=");
            //fprintf(stderr, "G__readansiproto: buf: '%s'\n", buf);
            tagnum = G__search_tagname(buf, 'c');
            type = 'u';
         }
         else if (!strcmp(buf, "struct")) {
            c = G__fgetname_template(buf, 0, ",)&*[(=");
            //fprintf(stderr, "G__readansiproto: buf: '%s'\n", buf);
            tagnum = G__search_tagname(buf, 's');
            type = 'u';
         }
         else if (!strcmp(buf, "union")) {
            c = G__fgetname_template(buf, 0, ",)&*[(=");
            //fprintf(stderr, "G__readansiproto: buf: '%s'\n", buf);
            tagnum = G__search_tagname(buf, 'u');
            type = 'u';
         }
         else if (!strcmp(buf, "enum")) {
            c = G__fgetname_template(buf, 0, ",)&*[(=");
            //fprintf(stderr, "G__readansiproto: buf: '%s'\n", buf);
            tagnum = G__search_tagname(buf, 'e');
            type = 'i';
         }
         else if (!strcmp(buf, "int")) {
            type = 'i' + isunsigned;
         }
         else if (!strcmp(buf, "char")) {
            type = 'c' + isunsigned;
         }
         else if (!strcmp(buf, "short")) {
            type = 's' + isunsigned;
         }
         else if (!strcmp(buf, "long")) {
            if ((c != ',') && (c != ')') && (c != '(')) {
               fpos_t pos;
               fgetpos(G__ifile.fp, &pos);
               int store_line = G__ifile.line_number;
               int store_c = c;
               c = G__fgetname(buf, 0, ",)&*[(="); // FIXME: Change to G__fgetname_template???
               //fprintf(stderr, "G__readansiproto: buf: '%s'\n", buf);
               if (!strcmp(buf, "long") || !strcmp(buf, "double")) {
                  if (!strcmp(buf, "long")) {
                     if (isunsigned) {
                        type = 'm';
                     }
                     else {
                        type = 'n';
                     }
                  }
                  else {
                     type = 'q';
                  }
               }
               else if (!strcmp(buf, "int")) {
                  type = 'l' + isunsigned;
               }
               else {
                  G__ifile.line_number = store_line;
                  fsetpos(G__ifile.fp, &pos);
                  c = store_c;
                  type = 'l' + isunsigned;
               }
            }
            else {
               type = 'l' + isunsigned;
            }
         }
         else if (!strcmp(buf, "float")) {
            type = 'f' + isunsigned;
         }
         else if (!strcmp(buf, "double")) {
            type = 'd' + isunsigned;
         }
         else if (!strcmp(buf, "bool")) {
            type = 'g';
         }
         else if (!strcmp(buf, "void")) {
            type = 'y';
         }
         else if (!strcmp(buf, "FILE")) {
            type = 'e';
         }
         else {
            int store_tagdefining = G__tagdefining;
            int store_def_tagnum = G__def_tagnum;
            if (G__friendtagnum != -1) {
               G__tagdefining = G__friendtagnum;
               G__def_tagnum = G__friendtagnum;
            }
            typenum = G__defined_typename(buf);
            if (typenum != -1) {
               tagnum = G__newtype.tagnum[typenum];
               type = G__newtype.type[typenum];
               ptrcnt += G__newtype.nindex[typenum];
               ifunc->param[func_now][iin]->isconst |= G__newtype.isconst[typenum];
            }
            else {
               tagnum = G__defined_tagname(buf, 1);
               if (tagnum != -1) {
                  if (G__struct.type[tagnum] == 'e') {
                     type = 'i';
                  }
                  else {
                     // re-evaluate typedef name in case of template class
                     if (strchr(buf, '<')) {
                        typenum = G__defined_typename(buf);
                     }
                     type = 'u';
                  }
               }
               else {
                  //fprintf(stderr, "G__readansiproto: failed to find tagname: '%s' ... \n", buf);
                  if (G__fpundeftype) {
                     tagnum = G__search_tagname(buf, 'c');
                     fprintf(G__fpundeftype, "class %s; /* %s %d */\n", buf(), G__ifile.name, G__ifile.line_number);
                     fprintf(G__fpundeftype, "#pragma link off class %s;\n\n", buf());
                     if (tagnum > -1) { // it could be -1 if we get too many classes.
                        G__struct.globalcomp[tagnum] = G__NOLINK;
                     }
                     type = 'u';
                  }
                  else {
                     // In case of f(unsigned x,signed y)
                     type = 'i' + isunsigned;
                     if (!isdigit(buf[0]) && !isunsigned) {
                        if (G__dispmsg >= G__DISPWARN) {
                           G__fprinterr(G__serr, "Warning: Unknown type '%s' in function argument handled as int", buf());
                           G__printlinenum();
                        }
                     }
                  }
               }
            }
            G__tagdefining = store_tagdefining;
            G__def_tagnum = store_def_tagnum;
         }
      }
      //
      //  Process the rest of the type specifier and the parameter name.
      //
      int is_a_reference = 0;
      int has_a_default = 0;
      G__FastAllocString param_name(G__LONGLINE);
      param_name[0] = '\0';
      {
         int arydim = 0; // Track which array bound we are processing, so we can handle an unspecified length array.
         while ((c != ',') && (c != ')')) {
            switch (c) {
               case EOF: // end-of-file
                  // -- Error, unexpected end of file, exit.
                  return 1;
               case ' ':
               case '\f':
               case '\n':
               case '\r':
               case '\t':
               case '\v':
                  // -- Whitespace, skip it.
                  c = G__fgetspace();
                  break;
               case '.': // ellipses
                  // -- We have reached an ellipses (must be at end), flag it and we are done.
                  ifunc->ansi[func_now] = 2;
                  c = G__fignorestream(",)");
                  // Note: The enclosing loop will terminate after we break.
                  break;
               case '&': // reference
                  ++is_a_reference;
                  c = G__fgetspace();
                  break;
               case '*': // pointer
                  ++ptrcnt;
                  c = G__fgetspace();
                  break;
               case '[': // array bound
                  ++arydim;
                  if (
                     (G__globalcomp < G__NOLINK) &&
                     (
                        !param_name[0] || // No parameter name given, first array bound.
                        (param_name[0] == '[') // We are processing a second or greater array bound.
                     )
                  ) {
                     // read 'MyFunc(int [][30])' or 'MyFunc(int [])'
                     int len = strlen(param_name);
                     param_name.Resize(len + 2);
                     param_name[len++] = c;
                     param_name[len++] = ']';
                     // Ignore the given array bound value.
                     c = G__fignorestream("],)"); // <<<
                     //
                     //  Peek ahead looking for another array bound.
                     //
                     {
                        G__disp_mask = 1000;
                        fpos_t tmp_pos;
                        fgetpos(G__ifile.fp, &tmp_pos);
                        int tmp_line = G__ifile.line_number;
                        c = G__fgetstream(param_name, len, "[=,)");
                        fsetpos(G__ifile.fp, &tmp_pos);
                        G__ifile.line_number = tmp_line;
                        G__disp_mask = 0;
                     }
                     if (c == '[') {
                        // Collect all the rest of the array bounds.
                        c = G__fgetstream(param_name, len, "=,)");
                        ptrcnt = 0; // FIXME: This erases all pointers (int* ary[d][d] is broken!)
                        break;
                     }
                     else {
                        // G__fignorestream("],)") already called above <<<
                        param_name[0] = 0;
                     }
                  }
                  else {
                     c = G__fignorestream("],)");
                  }
                  ++ptrcnt;
                  c = G__fgetspace();
                  break;
               case '=': // default value
                  // -- Parameter has a default value, collect it, and we are done.
                  // Collect the rest of the parameter specification as the default text.
                  has_a_default = 1;
                  c = G__fgetstream_template(buf, 0, ",)");
                  // Note: The enclosing loop will terminate after we break.
                  break;
               case '(': // Assume a function pointer type, e.g., MyFunc(int, int (*fp)(int, ...), int, ...)
                  // -- We have a function pointer type.
                  //
                  //  If the return type is a typedef or a class, struct, union, or enum,
                  //  then normalize the typename.
                  //
                  if (
                     ((typenum != -1) && (G__newtype.parent_tagnum[typenum] != -1)) ||
                     ((tagnum != -1) && (G__struct.parent_tagnum[tagnum] != -1))
                  ) {
                     char* p = strrchr(buf, ' ');
                     if (p) {
                        ++p;
                     }
                     else {
                        p = buf;
                     }
                     *p = 0;
                     buf += G__type2string(0, tagnum, typenum, 0, 0);
                  }
                  //
                  //  Normalize the rest of the parameter specification,
                  //  up to any default value, and collect the parameter name.
                  //
                  {
                     // Handle any ref part of the return type.
                     // FIXME: This is wrong, cannot have a pointer to a reference!
                     int i = strlen(buf);
                     if (type == 'm' || type == 'n' || type == 'q') {
                        // prepend "long":
                        G__FastAllocString tmplong(i + 5 /*long*/);
                        tmplong = "long ";
                        tmplong += buf;
                        buf.Swap(tmplong);
                        i += 5;
                     }
                     // Add ptr level: CINT cannot handle that many anyway
                     buf.Resize(i + ptrcnt * 2 + 10 + 5 /*possibly "long "*/);
                     if (is_a_reference) {
                        buf[i++] = '&';
                     }
                     is_a_reference = 0;
                     buf[i++] = ' ';
                     // Handle any pointer part of the return type.
                     // FIXME: This is wrong, cannot have a pointer to a reference!
                     for (int j = 0; j < ptrcnt; ++j) {
                        buf[i++] = '*';
                     }
                     ptrcnt = 0;
                     // Start constructing the parameter name part.
                     buf[i++] = '(';
                     c = G__fgetstream(buf, i, "*)");
                     if (c == '*') {
                        buf.Resize(i + 1);
                        buf[i++] = c;
                        c = G__fgetstream(param_name, 0, ")");
                        int j = 0;
                        for (; param_name[j] == '*'; ++j) {
                           buf.Resize(i + 1);
                           buf[i++] = '*';
                        }
                        if (j) {
                           int k = 0;
                           while (param_name[j]) {
                              param_name[k++] = param_name[j++];
                           }
                           param_name[k] = 0;
                        }
                     }
                     if (c == ')') {
                        buf.Resize(i + 1);
                        buf[i++] = ')';
                     }
                     // Copy out the rest of the parameter specification (up to a default value, if any).
                     c = G__fdumpstream(buf, i, ",)=");
                  }
                  //fprintf(stderr, "G__readansiproto: buf: '%s'\n", buf);
#ifndef G__OLDIMPLEMENTATION2191
                  typenum = G__search_typename(buf, '1', -1, 0);
                  type = '1';
#else // G__OLDIMPLEMENTATION2191
                  typenum = G__search_typename(buf, 'Q', -1, 0);
                  type = 'Q';
#endif // G__OLDIMPLEMENTATION2191
                  tagnum = -1;
                  break;
               default: // Consume "long long", "const", "const*", array bounds, param name (with def val)
                  // -- Not whitespace or separator.
                  if (!strcmp(param_name, "long") && !strcmp(buf, "long")) { // Process "long long" type.
                     type = 'n';
                     tagnum = -1;
                     typenum = -1;
                  }
                  // Collect next keyword or id into param_name.
                  param_name[0] = c;
                  c = G__fgetstream(param_name, 1, "[=,)& \t");
                  if (!strcmp(param_name, "const")) { // handle const keyword
                     ifunc->param[func_now][iin]->isconst |= G__PCONSTVAR; // FIXME: This is intentionally wrong!  Fix the code that depends on this!
                     param_name[0] = 0;
                  }
                  if (!strcmp(param_name, "const*")) { // handle const keyword and a single pointer spec
                     ifunc->param[func_now][iin]->isconst |= G__CONSTVAR; // FIXME: This is intentionally wrong!  Fix the code that depends on this!
                     ++ptrcnt;
                     param_name[0] = 0;
                  }
                  else { // Process any array bounds and possible default value.
                     while ((c == '[') || (c == ']')) { // We have array bounds.
                        if (c == '[') { // Consume an array bound, converting it into a pointer.
                           ++ptrcnt;
                           ++arydim;
                           if ((G__globalcomp < G__NOLINK) && (arydim == 2)) { // We are generating dictionaries and this is the second array bound.
                              // -- We are generating dictionaries and have just seen the beginning of the second array bound.
                              int len = strlen(param_name);
                              if (param_name[0] == ']') {
                                 len = 0;
                              }
                              param_name.Replace(len, "[]");
                              ptrcnt -= 2;
                              len = strlen(param_name);
                              fseek(G__ifile.fp, -1, SEEK_CUR);
                              if (G__dispsource) {
                                 G__disp_mask = 1;
                              }
                              c = G__fgetstream(param_name, len, "=,)");
                              // Note: Either we next process a default value, or enclosing loop terminates.
                              break;
                           }
                        }
                        // Skip till next array bound, default value, or end of parameter.
                        c = G__fignorestream("[=,)");
                     }
                     if (c == '=') { // We have a default value.
                        has_a_default = 1;
                        c = G__fgetstream_template(buf, 0, ",)");
                        //fprintf(stderr, "G__readansiproto: buf: '%s'\n", buf);
                        // Note: Enclosing loop will terminate after we break.
                     }
                  }
                  break;
            }
         }
      }
      ifunc->param[func_now][iin]->p_tagtable = tagnum;
      ifunc->param[func_now][iin]->p_typetable = typenum;
      if (!has_a_default) {
         // -- No default text or value.
         ifunc->param[func_now][iin]->pdefault = 0;
         ifunc->param[func_now][iin]->def = 0;
      } else {
         // -- Remember the default text, and then evaluate it.
         ifunc->param[func_now][iin]->pdefault = (G__value*) malloc(sizeof(G__value));
         ifunc->param[func_now][iin]->def = (char*) malloc(strlen(buf) + 1);
         strcpy(ifunc->param[func_now][iin]->def, buf); // Okay we allocated enough sapce
         if (buf[0] == '(') {
            int len = strlen(buf);
            if (
               (len > 5) &&
               (buf[len-4] != '*') &&
               !strcmp(")()", buf + len - 3) &&
               strchr(buf, '<')
            ) {
               int i = 1;
               for (; i < len - 3; ++i) {
                  buf[i-1] = buf[i];
               }
               buf.Replace(i - 1, "()");
            }
         }
         int store_def_tagnum = G__def_tagnum;
         int store_tagdefining = G__tagdefining;
         int store_prerun = G__prerun;
         int store_decl = G__decl;
         int store_var_type = G__var_type;
         G__var_type = 'p';
         int store_tagnum_default = 0;
         int store_def_struct_member_default = 0;
         int store_exec_memberfunc = 0;
         if (G__def_tagnum != -1) {
            store_tagnum_default = G__tagnum;
            store_exec_memberfunc = G__exec_memberfunc;
            store_def_struct_member_default = G__def_struct_member;
            G__tagnum = G__def_tagnum;
            G__exec_memberfunc = 1;
            G__def_struct_member = 0;
         }
         if (G__globalcomp == G__NOLINK) {
            G__prerun = 0;
            G__decl = 1;
         }
         {
            struct G__ifunc_table_internal* store_pifunc = G__p_ifunc;
            G__p_ifunc = &G__ifunc;
            if (G__decl && G__prerun && ((G__globalcomp == G__CPPLINK) || (G__globalcomp == R__CPPLINK))) {
               G__noerr_defined = 1;
            }
            // FIXME: These should be evaluated every function call!
            {
               // Note: Any temp created here must have a lifetime matching
               // the lifetime of the function declaration, so allocate at
               // temp level zero which survives until a scratch.
               int store_templevel = G__templevel;
               G__templevel = 0;
               *ifunc->param[func_now][iin]->pdefault = G__getexpr(buf);
               G__templevel = store_templevel;
            }
            if (G__decl && G__prerun && ((G__globalcomp == G__CPPLINK) || (G__globalcomp == R__CPPLINK))) {
               G__noerr_defined = 0;
            }
            G__value* val = ifunc->param[func_now][iin]->pdefault;
            if (is_a_reference && !ptrcnt && ((toupper(val->type) != toupper(type)) || (val->tagnum != tagnum))) {
               // -- If binding a reference to default rvalue and the types do not match, do a cast.
               G__FastAllocString tmp(G__ONELINE);
               tmp.Format("%s(%s)", G__type2string(type, tagnum, -1, 0, 0), buf());
               // Note: Any temp created here must have a lifetime matching
               // the lifetime of the function declaration, so allocate at
               // temp level zero which survives until a scratch.
               int store_templevel = G__templevel; 
               G__templevel = 0;
               *val = G__getexpr(tmp);
               G__templevel = store_templevel;
               if (val->type == 'u') {
                  val->ref = val->obj.i;
               }
            }
            G__p_ifunc = store_pifunc;
         }
         G__prerun = store_prerun;
         G__decl = store_decl;
         G__def_tagnum = store_def_tagnum;
         G__tagdefining = store_tagdefining;
         if (G__def_tagnum != -1) {
            G__tagnum = store_tagnum_default;
            G__exec_memberfunc = store_exec_memberfunc;
            G__def_struct_member = store_def_struct_member_default;
         }
         G__var_type = store_var_type;
      }
      if (isupper(type) && ptrcnt) {
         // Note: This can only happen if the type was a typedef to an array of pointers.
         ++ptrcnt;
      }
      if ((typenum != -1) && (G__newtype.reftype[typenum] >= G__PARAP2P)) {
         ptrcnt += G__newtype.reftype[typenum] - G__PARAP2P + 2;
         type = tolower(type);
      }
      switch (ptrcnt) {
         case 0:
            ifunc->param[func_now][iin]->type = type;
            if (!is_a_reference) {
               ifunc->param[func_now][iin]->reftype = G__PARANORMAL;
            }
            else {
               ifunc->param[func_now][iin]->reftype = G__PARAREFERENCE;
            }
            break;
         case 1:
            ifunc->param[func_now][iin]->type = toupper(type);
            if (!is_a_reference) {
               ifunc->param[func_now][iin]->reftype = G__PARANORMAL;
            }
            else {
               ifunc->param[func_now][iin]->reftype = G__PARAREFERENCE;
            }
            break;
         default:
            ifunc->param[func_now][iin]->type = toupper(type);
            if (!is_a_reference) {
               ifunc->param[func_now][iin]->reftype = ptrcnt - 2 + G__PARAP2P;
            }
            else {
               ifunc->param[func_now][iin]->reftype = ptrcnt - 2 + G__PARAREFP2P;
            }
            break;
      }
      // Remember the parameter name.
      if (!param_name[0]) {
         ifunc->param[func_now][iin]->name = 0;
      }
      else {
         ifunc->param[func_now][iin]->name = (char*) malloc(strlen(param_name) + 1);
         strcpy(ifunc->param[func_now][iin]->name, param_name); // Okay we allocated enough space
      }
   }
   ifunc->para_nu[func_now] = iin;
   return 0;
}

//______________________________________________________________________________
int G__matchpointlevel(int param_reftype, int formal_reftype)
{
// -- FIXME: Describe this function!
switch (param_reftype) {
   case G__PARANORMAL:
   case G__PARAREFERENCE:
      if (G__PARANORMAL == formal_reftype || G__PARAREFERENCE == formal_reftype)
         return(1);
      else
         return(0);
   default:
      return(formal_reftype == param_reftype);
}
}

//______________________________________________________________________________
int G__param_match(char formal_type, int formal_tagnum, G__value* default_parameter, char param_type, int param_tagnum, G__value* param, char* parameter, int funcmatch, int rewind_arg, int formal_reftype, int formal_isconst)
{
   // -- FIXME: Describe this function!
   if (default_parameter && (param_type == '\0')) {
      return 2;
   }
   static int recursive = 0; // FIXME: This is not thread-safe!
   int rewindflag = 0;
   G__value reg;
   G__FastAllocString conv(G__ONELINE);
   G__FastAllocString arg1(G__ONELINE);
   //
   //  Try exact match.
   //
   int match = 0;
   if (funcmatch >= G__EXACT) {
      if (
         !recursive && // Not recursive, and
         (param_tagnum == formal_tagnum) && // Argument tagnum matches formal
                                            // parameter tagnum, and
         (
            (param_type == formal_type) || // Argument type code matches
                                           // formal parameter typecode, or
            // FIXME: We do not handle references or multi-level pointers
            // correctly here!
            (
               (param_type == 'I') && // Argument type is int*, and
               (formal_type == 'U') && // Formal parameter type is pointer
                                       // to struct, and
               (formal_tagnum != -1) && // Formal parameter rawtype is valid,
                                        // and
               (G__struct.type[formal_tagnum] == 'e') // Formal parameter
                                                      // rawtype is an enum
            ) || // or,
            (
               (param_type == 'U') && // Argument type is pointer to struct, and
               (formal_type == 'I') && // Formal parameter type is int*, and
               (param_tagnum != -1) && // Argument rawtype is valid, and
               (G__struct.type[param_tagnum] == 'e') // Argument rawtype is
                                                     // an enum
            ) || // or,
            (
               (param_type == 'i') &&  // Argument type is int, and
               (formal_type == 'u') && // Formal parameter type is struct, and
               (formal_tagnum != -1) && // Formal parameter rawtype is valid,
                                        // and
               (G__struct.type[formal_tagnum] == 'e') // Formal parameter
                                                      // rawtype is an enum
            ) || // or,
            (
               (param_type == 'u') &&  // Argument type is struct, and
               (formal_type == 'i') && // Formal parameter type is int, and
               (param_tagnum != -1) && // Argument rawtype is valid, and
               (G__struct.type[param_tagnum] == 'e') // Argument rawtype is
                                                     // an enum
            )
         )
      ) {
         match = 1;
      }
   }
   //
   //  Try argument promotion.
   //
   if (!match && (funcmatch >= G__PROMOTION)) {
      switch (formal_type) {
         case 'd': // double
         case 'f': // float
            // Integral and float types promote to double, float.
            // FIXME: We should not allow double --> float!
            switch (param_type) {
               case 'b': // unsigned char
               case 'c': // char
               case 'd': // double // FIXME: Bad if formal_type is float!
               case 'f': // float
               case 'h': // unsigned int
               case 'i': // int
               case 'k': // unsigned long
               case 'l': // long
               case 'r': // unsigned short
               case 's': // short
                  match = 1;
                  break;
               default:
                  break;
            }
            break;
         case 'h': // unsigned int
            switch (param_type) {
               case 'b': // unsigned char
               //case 'c': // char
               case 'h': // unsigned int
               //case 'i': // int
               //case 'k': // unsigned long
               //case 'l': // long
               case 'r': // unsigned short
               //case 's': // short
                  match = 1;
                  break;
               default:
                  break;
            }
            break;
         case 'i': // int
            switch (param_type) {
               case 'b':
               case 'c':
               case 'r':
               case 's':
                  /* case 'h': */
               case 'i':
                  /* case 'k': */
                  /* case 'l': */
                  match = 1;
                  break;
               case 'u':
                  if (G__struct.type[param_tagnum] == 'e') {
                     if (param->ref) {
                        param->obj.i = *(long*)(param->ref);
                     }
                     match = 1;
                  }
                  break;
               default:
                  break;
            }
            break;
         case 'k': // unsigned long
            switch (param_type) {
               case 'b':
                  /* case 'c': */
               case 'r':
                  /* case 's': */
               case 'h':
                  /* case 'i': */
               case 'k':
                  /* case 'l': */
                  match = 1;
                  break;
               default:
                  break;
            }
            break;
         case 'l': // long
            switch (param_type) {
               case 'b':
               case 'c':
               case 'r':
               case 's':
                  /* case 'h': */
               case 'i':
                  /* case 'k': */
               case 'l':
                  match = 1;
                  break;
               default:
                  break;
            }
            break;
         case 'r': // unsigned short
            switch (param_type) {
               case 'b': // unsigned char
               //case 'c': // char
               //case 'h': // unsigned int
               //case 'i': // int
               //case 'k': // unsigned long
               //case 'l': // long
               case 'r': // unsigned short
               //case 's': // short
                  match = 1;
                  break;
               default:
                  break;
            }
            break;
         case 's': // short
            switch (param_type) {
               case 'b':
               case 'c':
                  /* case 'r': */
               case 's':
                  /* case 'h': */
                  /* case 'i': */
                  /* case 'k': */
                  /* case 'l': */
                  match = 1;
                  break;
               default:
                  break;
            }
            break;
         case 'u': // class, struct, union, enum
            if (
               (formal_tagnum != -1) &&
               (G__struct.type[formal_tagnum] == 'e')
            ) {
               switch (param_type) {
                  case 'b': // unsigned char
                  case 'c': // char
                  case 'h': // unsigned int
                  case 'i': // int
                  case 'k': // unsigned long
                  case 'l': // long
                  case 'r': // unsigned short
                  case 's': // short
                     match = 1;
                     break;
                  default:
                     break;
               }
            }
            break;
         default:
            break;
      }
   }
   //
   //  Try a standard conversion.
   //
   if (!match && (funcmatch >= G__STDCONV)) {
      switch (formal_type) {
         case 'b': // unsigned char
         case 'c': // char
         case 'd': // double
         case 'f': // float
         case 'h': // unsigned int
         case 'i': // int
         case 'k': // unsigned long
         case 'l': // long
         case 'r': // unsigned short
         case 's': // short
            switch (param_type) {
               case 'b': // unsigned char
               case 'c': // char
               case 'd': // double
               case 'f': // float
               case 'h': // unsigned int
               case 'i': // int
               case 'k': // unsigned long
               case 'l': // long
               case 'r': // unsigned short
               case 's': // short
                  match = 1;
                  break;
               default:
                  break;
            }
            break;
         case 'C':
            switch (param_type) {
               case 'i':
               case 'l':
                  if (!param->obj.i) {
                     match = 1;
                  }
                  break;
               case 'Y':
                  match = 1;
                  break;
               default:
                  break;
            }
            break;
         case 'Y':
            if (isupper(param_type) || !param->obj.i) {
               match = 1;
            }
            break;
#if !defined(G__OLDIMPLEMENTATION2191)
         case '1': /* questionable */
            if (
               (param_type == '1') ||
               (param_type == 'C') ||
               (param_type == 'Y') 
            ) {
               match = 1;
            }
            break;
#else // G__OLDIMPLEMENTATION2191
         case 'Q': /* questionable */
            if (
               'Q' == param_type ||
               'C' == param_type ||
               'Y' == param_type
            ) {
               match = 1;
            }
            break;
#endif // G__OLDIMPLEMENTATION2191
#ifdef G__WORKAROUND000209_1
         // reference type conversion should not be handled in this way.
         // difference was found from g++ when activating this part.
         //
         // Added condition for formal_reftype and recursive, then things
         // are working 1999/12/5
         case 'u':
            if ((formal_reftype == G__PARAREFERENCE) && recursive) {
               switch (param_type) {
                  case 'u': {
                     // reference to derived class can be converted to
                     // reference to base class. add offset, modify
                     // char *parameter and G__value *param
                     int baseoffset =
                        G__ispublicbase(formal_tagnum, param_tagnum,
                           param->obj.i);
                     if (baseoffset != -1) {
                        param->tagnum = formal_tagnum;
                        param->obj.i += baseoffset;
                        param->ref += baseoffset;
                        match = 1;
                     }
                  }
                  break;
               }
            }
            break;
#endif // G__WORKAROUND000209_1
            // --
         case 'U':
            switch (param_type) {
               case 'U':
                  {
                     // Pointer to derived class can be converted to
                     // pointer to base class.
                     // add offset, modify char *parameter and
                     // G__value *param
                     //
#ifdef G__VIRTUALBASE
                     int baseoffset =
                        G__ispublicbase(formal_tagnum, param_tagnum,
                           param->obj.i);
#else // G__VIRTUALBASE
                     int baseoffset =
                        G__ispublicbase(formal_tagnum, param_tagnum);
#endif // G__VIRTUALBASE
                     if (baseoffset != -1) {
                        param->tagnum = formal_tagnum;
                        param->obj.i += baseoffset;
                        param->ref = 0;
                        match = 1;
                     }
                  }
                  break;
               case 'Y':
#ifndef G__OLDIMPLEMENTATION2191
               case '1': // questionable
#else // G__OLDIMPLEMENTATION2191
               case 'Q': // questionable
#endif // G__OLDIMPLEMENTATION2191
                  match = 1;
                  break;
               case 'i':
               case 0:
                  if (!param->obj.i) {
                     match = 1;
                  }
                  break;
               default:
                  break;
            }
            break;
         default:
            // questionable
#ifndef G__OLDIMPLEMENTATION2191
            if (
               (
                  (param_type == 'Y') ||
                  (param_type == '1') ||
                  !param->obj.i
               ) &&
               (
                  isupper(formal_type) ||
                  (formal_type == 'a')
               )
            ) {
               match = 1;
            }
            if (
               (
                  (param_type == 'Y') ||
                  (param_type == 'Q') ||
                  !param->obj.i
               ) &&
               (
                  isupper(formal_type) ||
                  (formal_type == 'a')
               )
            ) {
               match = 1;
            }
            else {
               match = 0; // FIXME: This could undo the previous match!
            }
#endif // G__OLDIMPLEMENTATION2191
            break;
      }
   }
   //
   //  Try a user-specified conversion function.
   //
   if (!match && (funcmatch >= G__USERCONV)) {
      if ((formal_type == 'u') && !recursive) {
         // Create a temporary of type formal_tagnum.
         if (G__struct.iscpplink[formal_tagnum] != G__CPPLINK) {
            // The formal_tagnum class is interpreted.
#ifdef G__ASM
            if (G__asm_noverflow) {
               // We are generating bytecode.
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(
                       G__serr
                     , "%3x,%3x: ALLOCTEMP %s %d  %s:%d\n"
                     , G__asm_cp
                     , G__asm_dt
                     , G__struct.name[formal_tagnum]
                     , formal_tagnum
                     , __FILE__
                     , __LINE__
                  );
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__ALLOCTEMP;
               G__asm_inst[G__asm_cp+1] = formal_tagnum;
               G__inc_cp_asm(2, 0);
            }
#endif // G__ASM
            G__alloc_tempobject(formal_tagnum, -1);
         }
         // Try finding constructor.
         if (param_type != 'u') {
            G__valuemonitor(*param, arg1);
         }
         else {
            if (param->obj.i < 0) { // Protect against sign character.
               arg1.Format("(%s)(%ld)",
                  G__fulltagname(param_tagnum, 1), param->obj.i);
            }
            else {
               arg1.Format("(%s)%ld",
                  G__fulltagname(param_tagnum, 1), param->obj.i);
            }
         }
         conv.Format("%s(%s)", G__struct.name[formal_tagnum], arg1());
         if (G__dispsource) {
            G__fprinterr(
                 G__serr
               , "!!!Trying implicit conversion %s  %s:%d\n"
               , conv()
               , __FILE__
               , __LINE__
            );
         }
#ifdef G__ASM
         if (G__asm_noverflow) {
            // We are generating bytecode.
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(
                    G__serr
                  , "%3x,%3x: SETTEMP  %s:%d\n"
                  , G__asm_cp
                  , G__asm_dt
                  , __FILE__
                  , __LINE__
               );
            }
#endif // G__ASM_DBG
            // FIXME: This sets G__tagnum to G__p_tempbuf->obj.tagnum,
            //        but we want formal_tagnum instead!
            G__asm_inst[G__asm_cp] = G__SETTEMP;
            G__inc_cp_asm(1, 0);
         }
#endif // G__ASM
         // FIXME: We did not create a temporary if formal_tagnum is compiled!
         long store_struct_offset = G__store_struct_offset;
         G__store_struct_offset = G__p_tempbuf->obj.obj.i;
         int store_tagnum = G__tagnum;
         G__tagnum = formal_tagnum;
         int store_oprovld = G__oprovld;
         G__oprovld = 1; // Tell G__getfunction to not stack the args.
#ifdef G__ASM
         if (G__asm_noverflow && rewind_arg) {
            rewindflag = 1;
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(
                    G__serr
                  , "%3x,%3x: REWINDSTACK %d  %s:%d\n"
                  , G__asm_cp
                  , G__asm_dt
                  , rewind_arg
                  , __FILE__
                  , __LINE__
               );
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__REWINDSTACK;
            G__asm_inst[G__asm_cp+1] = rewind_arg;
            G__inc_cp_asm(2, 0);
         }
#endif // G__ASM
         ++recursive;
         if (G__struct.iscpplink[formal_tagnum] == G__CPPLINK) {
            // compiled class
            reg = G__getfunction(conv, &match, G__TRYIMPLICITCONSTRUCTOR);
            if (match) {
               // --
#ifdef G__ASM
               if (G__asm_noverflow) {
                  // We are generating code.
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(
                          G__serr
                        , "%3x,%3x: STORETEMP  %s:%d\n"
                        , G__asm_cp
                        , G__asm_dt
                        , __FILE__
                        , __LINE__
                     );
                  }
#endif // G__ASM_DBG
                  G__asm_inst[G__asm_cp] = G__STORETEMP;
                  G__inc_cp_asm(1, 0);
               }
#endif // G__ASM
               G__store_tempobject(reg);
            }
            else {
               conv.Format("operator %s()", G__fulltagname(formal_tagnum, 1));
               G__store_struct_offset = param->obj.i;
               G__tagnum = param->tagnum;
               if (G__tagnum != -1) {
                  reg = G__getfunction(conv, &match, G__TRYMEMFUNC);
               }
               if (!match) {
                  G__store_tempobject(G__null);
               }
            }
         }
         else {
            // interpreted class
            G__getfunction(conv, &match, G__TRYIMPLICITCONSTRUCTOR);
            if (match) {
               // --
#ifdef G__ASM
               if (G__asm_noverflow) {
                  // We are generating bytecode.
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(
                          G__serr
                        , "%3x,%3x: POPTEMP %d  %s:%d\n"
                        , G__asm_cp
                        , G__asm_dt
                        , formal_tagnum
                        , __FILE__
                        , __LINE__
                     );
                  }
#endif // G__ASM_DBG
                  G__asm_inst[G__asm_cp] = G__POPTEMP;
                  G__asm_inst[G__asm_cp+1] = formal_tagnum;
                  G__inc_cp_asm(2, 0);
               }
#endif // G__ASM
               // --
            }
            else {
               // No constructor found for interpreted class.
#ifdef G__ASM
               if (G__asm_noverflow) {
                  // FIXME: Write out message about what we are canceling!
                  G__inc_cp_asm(-3, 0);
               }
#endif // G__ASM
               // Try user-provided conversion function.
               conv.Format("operator %s()", G__fulltagname(formal_tagnum, 1));
               G__store_struct_offset = param->obj.i;
               G__tagnum = param->tagnum;
               reg = G__getfunction(conv, &match, G__TRYMEMFUNC);
               if (!match) {
                  // --
#ifdef G__ASM
                  if (G__asm_noverflow) {
                     // We are generating code.
                     if (rewindflag) {
                        G__asm_inst[G__asm_cp-2] = G__REWINDSTACK;
                        G__asm_inst[G__asm_cp-1] = rewind_arg;
                     }
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(
                             G__serr
                           , "ALLOCTEMP,SETTEMP canceled %x  %s:%d\n"
                           , G__asm_cp
                           , __FILE__
                           , __LINE__
                        );
                     }
#endif // G__ASM_DBG
                     // --
                  }
#endif // G__ASM
                  // --
               }
            }
         }
         --recursive;
         G__oprovld = store_oprovld;
         G__tagnum = store_tagnum;
         G__store_struct_offset = store_struct_offset;
         // If no constructor, try converting to base class.
         if (!match) {
            // --
#ifdef G__VIRTUALBASE
            int baseoffset =
               G__ispublicbase(formal_tagnum, param_tagnum, param->obj.i);
#else // G__VIRTUALBASE
            int baseoffset = G__ispublicbase(formal_tagnum, param_tagnum);
#endif // G__VIRTUALBASE
            if ((param_type == 'u') && (baseoffset != -1)) {
               if (G__dispsource) {
                  G__fprinterr(
                       G__serr
                     , "!!!Implicit conversion from %s to base %s  %s:%d\n"
                     , G__struct.name[param_tagnum]
                     , G__struct.name[formal_tagnum]
                     , __FILE__
                     , __LINE__
                  );
               }
#ifdef G__ASM
               if (G__asm_noverflow) {
                  // --
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(
                          G__serr
                        , "%3x,%3x: BASECONV %d %d  %s:%d\n"
                        , G__asm_cp
                        , G__asm_dt
                        , formal_tagnum
                        , baseoffset
                        , __FILE__
                        , __LINE__
                     );
                  }
#endif // G__ASM_DBG
                  G__asm_inst[G__asm_cp] = G__BASECONV;
                  G__asm_inst[G__asm_cp+1] = formal_tagnum;
                  G__asm_inst[G__asm_cp+2] = baseoffset;
                  G__inc_cp_asm(3, 0);
               }
#endif // G__ASM
               param->typenum = -1;
               param->tagnum = formal_tagnum;
               param->obj.i += baseoffset;
               param->ref += baseoffset;
#ifdef G__ASM
               if (G__asm_noverflow) {
                  // We are generating bytecode.
                  if (rewind_arg) {
                     rewindflag = 1;
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(
                             G__serr
                           , "%3x,%3x: REWINDSTACK %d  %s:%d\n"
                           , G__asm_cp
                           , G__asm_dt
                           , -rewind_arg
                           , __FILE__
                           , __LINE__
                        );
                     }
#endif // G__ASM_DBG
                     G__asm_inst[G__asm_cp] = G__REWINDSTACK;
                     G__asm_inst[G__asm_cp+1] = -rewind_arg;
                     G__inc_cp_asm(2, 0);
                  }
#endif // G__ASM
                  if (param->obj.i < 0) {
                     // parameter is G__param::parameter[index]
                     G__snprintf(parameter, G__ONELINE, "(%s)(%ld)",
                        G__struct.name[formal_tagnum], param->obj.i);
                  }
                  else {
                     // parameter is G__param::parameter[index]
                     G__snprintf(parameter, G__ONELINE, "(%s)%ld",
                        G__struct.name[formal_tagnum], param->obj.i);
                  }
               }
               match = 1;
               G__pop_tempobject();
            }
            else { // All conversions failed.
               if (G__dispsource) {
                  G__fprinterr(
                       G__serr
                     , "!!!Implicit conversion %s tried, but failed  %s:%d\n"
                     , conv()
                     , __FILE__
                     , __LINE__
                  );
               }
               G__pop_tempobject();
#ifdef G__ASM
               if (rewindflag) {
                  // --
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(
                          G__serr
                        , "REWINDSTACK cancelled.  %s:%d\n"
                        , __FILE__
                        , __LINE__
                     );
                  }
#endif // G__ASM_DBG
                  G__inc_cp_asm(-2, 0);
               }
            }
#else // G__ASM
               // All conversions failed.
               if (G__dispsource) {
                  G__fprinterr(
                       G__serr
                     , "!!!Implicit conversion %s tried, but failed  %s:%d\n"
                     , conv()
                     , __FILE__
                     , __LINE__
                  );
               }
               G__pop_tempobject();
#ifdef G__ASM
               if (rewindflag) {
                  // --
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(
                          G__serr
                        , "REWINDSTACK cancelled.  %s:%d\n"
                        , __FILE__
                        , __LINE__
                     );
                  }
#endif // G__ASM_DBG
                  G__inc_cp_asm(-2, 0);
               }
#endif //  G__ASM
#endif // G__ASM
            // --
         }
         else {
            // --
#ifdef G__ASM
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(
                    G__serr
                  , "!!!Create temp object (%s,%d,%d) 0x%lx %d for "
                    "implicit conversion %s  %s:%d\n"
                  , G__struct.name[G__p_tempbuf->obj.tagnum]
                  , G__p_tempbuf->obj.tagnum
                  , G__p_tempbuf->obj.typenum
                  , G__p_tempbuf->obj.obj.i
                  , G__templevel
                  , conv()
                  , __FILE__
                  , __LINE__
               );
            }
#endif // G__ASM_DBG
#endif // G__ASM
#ifdef G__ASM
            if (G__asm_noverflow && rewind_arg) {
               rewindflag = 1;
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(
                       G__serr
                     , "%3x,%3x: REWINDSTACK %d  %s:%d\n"
                     , G__asm_cp
                     , G__asm_dt
                     , -rewind_arg
                     , __FILE__
                     , __LINE__
                  );
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__REWINDSTACK;
               G__asm_inst[G__asm_cp+1] = -rewind_arg;
               G__inc_cp_asm(2, 0);
            }
#endif // G__ASM
            *param = G__p_tempbuf->obj;
            // parameter is G__param::parameter[index]
            G__snprintf(parameter, G__ONELINE, "(%s)%ld",
               G__struct.name[formal_tagnum], G__p_tempbuf->obj.obj.i);
         }
      }
      else if (param->tagnum != -1) {
         long store_struct_offset = G__store_struct_offset;
         int store_tagnum = G__tagnum;
         conv.Format("operator %s()",
            G__type2string(formal_type, formal_tagnum, -1, 0, 0));
         G__store_struct_offset = param->obj.i;
         G__tagnum = param->tagnum;
#ifdef G__ASM
         if (G__asm_noverflow) {
            if (rewind_arg) {
               rewindflag = 1;
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(
                       G__serr
                     , "%3x,%3x: REWINDSTACK %d  %s:%d\n"
                     , G__asm_cp
                     , G__asm_dt
                     , rewind_arg
                     , __FILE__
                     , __LINE__
                  );
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__REWINDSTACK;
               G__asm_inst[G__asm_cp+1] = rewind_arg;
               G__inc_cp_asm(2, 0);
            }
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(
                    G__serr
                  , "%3x,%3x: PUSHSTROS  %s:%d\n"
                  , G__asm_cp
                  , G__asm_dt
                  , __FILE__
                  , __LINE__
               );
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__PUSHSTROS;
            G__inc_cp_asm(1, 0);
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(
                    G__serr
                  , "%3x,%3x: SETSTROS  %s:%d\n"
                  , G__asm_cp
                  , G__asm_dt
                  , __FILE__
                  , __LINE__
               );
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__SETSTROS;
            G__inc_cp_asm(1, 0);
         }
#endif // G__ASM
         reg = G__getfunction(conv, &match, G__TRYMEMFUNC);
         if (!match && formal_isconst) {
            conv.Format("operator const %s()",
               G__type2string(formal_type, formal_tagnum, -1, 0, 0));
            G__store_struct_offset = param->obj.i;
            G__tagnum = param->tagnum;
            reg = G__getfunction(conv, &match, G__TRYMEMFUNC);
         }
         G__tagnum = store_tagnum;
         G__store_struct_offset = store_struct_offset;
#ifdef G__ASM
         if (G__asm_noverflow) {
            if (rewind_arg) {
               rewindflag = 1;
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(
                       G__serr
                     , "%3x,%3x: REWINDSTACK %d  %s:%d\n"
                     , G__asm_cp
                     , G__asm_dt
                     , -rewind_arg
                     , __FILE__
                     , __LINE__
                  );
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__REWINDSTACK;
               G__asm_inst[G__asm_cp+1] = -rewind_arg;
               G__inc_cp_asm(2, 0);
            }
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(
                    G__serr
                  , "%3x,%3x: POPSTROS  %s:%d\n"
                  , G__asm_cp
                  , G__asm_dt
                  , __FILE__
                  , __LINE__
               );
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__POPSTROS;
            G__inc_cp_asm(1, 0);
         }
#endif // G__ASM
         // fixing 'cout<<x' fundamental conversion opr with opr overloading
         // Not 100% sure if this is OK.
         if (match) {
            *param = reg;
         }
         else if (rewindflag) {
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(
                    G__serr
                  , "REWINDSTACK~ cancelled.  %s:%d\n"
                  , __FILE__
                  , __LINE__
               );
            }
#endif // G__ASM_DBG
            G__inc_cp_asm(-7, 0);
         }
         else {
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(
                    G__serr
                  , "PUSHSTROS~ cancelled.  %s:%d\n"
                  , __FILE__
                  , __LINE__
               );
            }
#endif // G__ASM_DBG
            G__inc_cp_asm(-3, 0);
         }
      }
      else {
         if (recursive && G__dispsource) {
            G__valuemonitor(*param, arg1);
            G__fprinterr(
                 G__serr
               , "!!!Recursive implicit conversion %s(%s) rejected  %s:%d\n"
               , G__struct.name[formal_tagnum]
               , arg1()
               , __FILE__
               , __LINE__
            );
         }
      }
   }
   //
   //  If we have a match between two pointer types,
   //  then do a final match on the pointer levels.
   //
   if (
      match &&
      isupper(param_type) &&
      isupper(formal_type) &&
      (param_type != 'Y') &&
      (formal_type != 'Y')
#if !defined(G__OLDIMPLEMENTATION2191)
      && (param_type != '1')
#else // G__OLDIMPLEMENTATION2191
      && (param_type != 'Q')
#endif // G__OLDIMPLEMENTATION2191
      // --
   ) {
      match = G__matchpointlevel(param->obj.reftype.reftype, formal_reftype);
   }
   return match;
}

//______________________________________________________________________________
//______________________________________________________________________________
#define G__NOMATCH        0xffffffff
#define G__EXACTMATCH     0x00000000
#define G__PROMOTIONMATCH 0x00000100
#define G__STDCONVMATCH   0x00010000
#define G__USRCONVMATCH   0x01000000
#define G__CVCONVMATCH    0x00000001
#define G__BASECONVMATCH  0x00000001
#define G__C2P2FCONVMATCH 0x00000001
#define G__I02PCONVMATCH  0x00000002
#define G__V2P2FCONVMATCH 0x00000002
#define G__TOVOIDPMATCH   0x00000003
//______________________________________________________________________________
//______________________________________________________________________________

//______________________________________________________________________________
struct G__funclist* G__funclist_add(G__funclist* last, G__ifunc_table_internal* ifunc, int ifn, int rate)
{
   // -- FIXME: Describe this function!
   struct G__funclist *latest = (struct G__funclist*)malloc(sizeof(struct G__funclist));
   latest->prev = last;
   latest->ifunc = ifunc;
   latest->ifn = ifn;
   latest->rate = rate;
   return(latest);
}

//______________________________________________________________________________
void G__funclist_delete(G__funclist* body)
{
   // -- FIXME: Describe this function!
   if (body) {
      if (body->prev) G__funclist_delete(body->prev);
      free((void*)body);
   }
}

//______________________________________________________________________________
unsigned int G__rate_inheritance(int basetagnum, int derivedtagnum)
{
   // -- FIXME: Describe this function!
   struct G__inheritance *derived;
   int i, n;

   if (0 > derivedtagnum || 0 > basetagnum) return(G__NOMATCH);
   if (basetagnum == derivedtagnum) return(G__EXACTMATCH);
   derived = G__struct.baseclass[derivedtagnum];
   n = derived->basen;

   for (i = 0;i < n;i++) {
      if (basetagnum == derived->herit[i]->basetagnum) {
         if (derived->herit[i]->baseaccess == G__PUBLIC ||
               (G__exec_memberfunc && G__tagnum == derivedtagnum &&
                G__GRANDPRIVATE != derived->herit[i]->baseaccess)) {
            if (G__ISDIRECTINHERIT&derived->herit[i]->property) {
               return(G__BASECONVMATCH);
            }
            else {
               int distance = 1;
               int ii = i; /* i is not 0, because !G__ISDIRECTINHERIT */
               struct G__inheritance *derived2 = derived;
               int derivedtagnum2 = derivedtagnum;
               while (0 == (derived2->herit[ii]->property&G__ISDIRECTINHERIT)) {
                  ++distance;
                  while (ii && 0 == (derived2->herit[--ii]->property&G__ISDIRECTINHERIT)) {}
                  derivedtagnum2 = derived2->herit[ii]->basetagnum;
                  derived2 = G__struct.baseclass[derivedtagnum2];
                  for (ii = 0;ii < derived2->basen;ii++) {
                     if (derived2->herit[ii]->basetagnum == basetagnum) break;
                  }
                  if (ii == derived2->basen) return(G__NOMATCH);
               }
               return(distance*G__BASECONVMATCH);
            }
         }
      }
   }
   return(G__NOMATCH);
}

//______________________________________________________________________________
//______________________________________________________________________________
#define G__promotiongrade(f,p) G__PROMOTIONMATCH*(G__igrd(f)-G__igrd(p))
//______________________________________________________________________________
//______________________________________________________________________________

//______________________________________________________________________________
static int G__igrd(int formal_type)
{
   // -- FIXME: Describe this function!

   // this looks good but is actually nowhere to be found in the standard;
   // it's not how conversions are to be ranked.
   switch (formal_type) {
      case 'g':
         return(1);
      case 'b':
      case 'c':
         return(2);
      case 'r':
      case 's':
         return(3);
      case 'h':
      case 'i':
         return(4);
      case 'k':
      case 'l':
         return(5);
      case 'n':
      case 'm':
         return(6);
   }
   return(0);
}

//______________________________________________________________________________
//______________________________________________________________________________
#ifndef __CINT__
struct G__ifunc_table_internal* G__overload_match G__P((const char* funcname, struct G__param* libp, int hash, struct G__ifunc_table_internal* p_ifunc, int memfunc_flag, int access, int* pifn, int recursive, int doconvert, int check_access));
#endif // __CINT__
//______________________________________________________________________________
//______________________________________________________________________________

//______________________________________________________________________________
void G__rate_parameter_match(G__param* libp, G__ifunc_table_internal* p_ifunc, int ifn, G__funclist* funclist, int recursive)
{
   // -- FIXME: Describe this function!
   int i;
   char param_type, formal_type;
   int param_tagnum, formal_tagnum;
   int param_reftype, formal_reftype;
   int param_isconst = 0, formal_isconst = 0;
   funclist->rate = 0;
   for (i = 0;i < libp->paran;i++) {
      param_type = libp->para[i].type;
      formal_type = p_ifunc->param[ifn][i]->type;
      param_tagnum = libp->para[i].tagnum;
      formal_tagnum = p_ifunc->param[ifn][i]->p_tagtable;
      param_reftype = libp->para[i].obj.reftype.reftype;
      formal_reftype = p_ifunc->param[ifn][i]->reftype;
      param_isconst = libp->para[i].isconst;
      formal_isconst = p_ifunc->param[ifn][i]->isconst;
      funclist->p_rate[i] = G__NOMATCH;

      if (param_type == 'i' && param_tagnum != -1
	  && formal_type == 'u' && formal_tagnum == param_tagnum
	  && G__struct.type[formal_tagnum] == 'e')
	// enum constants come in as param_type== 'i'
	param_type = 'u';
	

      /* exact match */
      if (param_type == formal_type) {
         if (tolower(param_type) == 'u') {
            /* If struct,class,union, check tagnum */
            if (formal_tagnum == param_tagnum) { /* match */
               funclist->p_rate[i] = G__EXACTMATCH;
            }
         }
         else if (isupper(param_type)) {
            if (param_reftype == formal_reftype ||
                  (param_reftype <= G__PARAREFERENCE &&
                   formal_reftype <= G__PARAREFERENCE)) {
               funclist->p_rate[i] = G__EXACTMATCH;
            }
            else if ((formal_reftype > G__PARAREF && formal_reftype == param_reftype + G__PARAREF)
                     || (param_reftype > G__PARAREF && param_reftype == formal_reftype + G__PARAREF)) {
               funclist->p_rate[i] = G__STDCONVMATCH;
            }
         }
         else if ('i' == param_type && (formal_tagnum != param_tagnum)) {
            funclist->p_rate[i] = G__PROMOTIONMATCH;
         }
         else { /* match */
            funclist->p_rate[i] = G__EXACTMATCH;
         }
      }
      else if (('I' == param_type || 'U' == param_type) &&
               ('I' == formal_type || 'U' == formal_type) &&
               param_tagnum == formal_tagnum &&
               -1 != formal_tagnum && 'e' == G__struct.type[formal_tagnum]) {
         funclist->p_rate[i] = G__EXACTMATCH;
      }
      else if (isupper(formal_type) &&
               ('i' == param_type || 'l' == param_type)
               && 0 == libp->para[i].obj.i) {
         funclist->p_rate[i] = G__STDCONVMATCH + G__I02PCONVMATCH;
      }

      /* promotion */
      if (G__NOMATCH == funclist->p_rate[i]) {
         switch (formal_type) {
         case 'd': /* 4.6: conv.fpprom */
            switch (param_type) {
            case 'f':
               funclist->p_rate[i] = G__PROMOTIONMATCH;
               break;
            default:
               break;
            }
            break;
         case 'i': /* 4.5: conv.prom */
         case 'h': /* 4.5: conv.prom */
            switch (param_type) {
            case 'b':
            case 'c':
            case 'r':
            case 's':
            case 'g':
               funclist->p_rate[i] = G__promotiongrade(formal_type, param_type);
               break;
            case 'u':
               if ('e' == G__struct.type[param_tagnum]) {
                  funclist->p_rate[i] = G__PROMOTIONMATCH;
               }
               break;
            default:
               break;
            }
            break;
         case 'l':
         case 'k': /* only enums get promoted to (u)long! */
            if (param_type == 'u' && 'e' == G__struct.type[param_tagnum]) {
                  funclist->p_rate[i] = G__PROMOTIONMATCH;
            }
            break;
         case 'Y':
            if (isupper(param_type) || 0 == libp->para[i].obj.i
#ifndef G__OLDIMPLEMENTATION2191
                || '1' == param_type
#endif // G__OLDIMPLEMENTATION2191
            ) {
               funclist->p_rate[i] = G__PROMOTIONMATCH + G__TOVOIDPMATCH;
            }
            break;
         default:
            break;
         }
      }

      /* standard conversion */
      if (G__NOMATCH == funclist->p_rate[i]) {
         switch (formal_type) {
         /* no; f(enum E) cannot be called as f(1)!
         case 'u':
            if (0 <= formal_tagnum && 'e' == G__struct.type[formal_tagnum]) {
               switch (param_type) {
               case 'i':
               case 's':
               case 'l':
               case 'c':
               case 'h':
               case 'r':
               case 'k':
               case 'b':
                  funclist->p_rate[i] = G__PROMOTIONMATCH;
                  break;
               default:
                  break;
               }
            }
            else {}
            break;
         */
         case 'b':
         case 'c':
         case 'r':
         case 's':
         case 'h':
         case 'i':
         case 'k':
         case 'l':
         case 'g':
         case 'n':
         case 'm':
         case 'd':
         case 'f':
            switch (param_type) {
            case 'd':
            case 'f':
            case 'b':
            case 'c':
            case 'r':
            case 's':
            case 'h':
            case 'i':
            case 'k':
            case 'l':
            case 'g':
            case 'n':
            case 'm':
            case 'q':
               funclist->p_rate[i] = G__STDCONVMATCH;
               break;
            case 'u':
               if ('e' == G__struct.type[param_tagnum]) {
                  funclist->p_rate[i] = G__PROMOTIONMATCH;
               }
               break;
            default:
               break;
            }
            break;
         case 'C':
            switch (param_type) {
            case 'i':
            case 'l':
               if (0 == libp->para[i].obj.i)
                  funclist->p_rate[i] = G__STDCONVMATCH + G__I02PCONVMATCH;
               break;
            case 'Y':
               if (G__PARANORMAL == param_reftype) {
                  funclist->p_rate[i] = G__STDCONVMATCH;
               }
               break;
            default:
               break;
            }
            break;
         case 'Y':
            if (isupper(param_type) || 0 == libp->para[i].obj.i) {
               funclist->p_rate[i] = G__STDCONVMATCH;
            }
            break;
#ifndef G__OLDIMPLEMENTATION2191
         case '1': /* questionable */
#else // G__OLDIMPLEMENTATION2191
         case 'Q': /* questionable */
#endif // G__OLDIMPLEMENTATION2191
            if (
#ifndef G__OLDIMPLEMENTATION2191
                '1' == param_type
#else // G__OLDIMPLEMENTATION2191
                'Q' == param_type
#endif // G__OLDIMPLEMENTATION2191
            )
               funclist->p_rate[i] = G__STDCONVMATCH;
            else if ('Y' == param_type)
               funclist->p_rate[i] = G__STDCONVMATCH + G__V2P2FCONVMATCH;
            else if ('C' == param_type) {
               if (
                   p_ifunc->pentry[ifn]->size >= 0
                   )
                  funclist->p_rate[i] = G__STDCONVMATCH - G__C2P2FCONVMATCH;
               else {
                  funclist->p_rate[i] = G__STDCONVMATCH + G__C2P2FCONVMATCH;/*???*/
               }
            }
            break;
         case 'u':
            switch (param_type) {
            case 'u':
               /* reference to derived class can be converted to reference to base
                * class. add offset, modify char *parameter and G__value *param */
               {
                  unsigned int rate_inheritance =
                     G__rate_inheritance(formal_tagnum, param_tagnum);
                  if (G__NOMATCH != rate_inheritance) {
                     funclist->p_rate[i] = G__STDCONVMATCH + rate_inheritance;
                  }
               }
               break;
            }
            break;
         case 'U':
            switch (param_type) {
            case 'U':
               /* Pointer to derived class can be converted to
                * pointer to base class.
                * add offset, modify char *parameter and
                * G__value *param
                */
               {
                  unsigned int rate_inheritance =
                     G__rate_inheritance(formal_tagnum, param_tagnum);
                  if (G__NOMATCH != rate_inheritance) {
                     funclist->p_rate[i] = G__STDCONVMATCH + rate_inheritance;
                  }
               }
               break;
            case 'Y':
               if (G__PARANORMAL == param_reftype) {
                  funclist->p_rate[i] = G__STDCONVMATCH;
               }
               break;
#ifndef G__OLDIMPLEMENTATION2191
            case '1': /* questionable */
#else // G__OLDIMPLEMENTATION2191
            case 'Q': /* questionable */
#endif // G__OLDIMPLEMENTATION2191
               funclist->p_rate[i] = G__STDCONVMATCH;
               break;
            case 'i':
            case 0:
               if (0 == libp->para[0].obj.i) funclist->p_rate[i] = G__STDCONVMATCH;
               break;
            default:
               break;
            }
            break;
         default:
            /* questionable */
#ifndef G__OLDIMPLEMENTATION2191
            if ((param_type == 'Y' || param_type == '1') &&
                (isupper(formal_type) || 'a' == formal_type)) {
               funclist->p_rate[i] = G__STDCONVMATCH;
            }
#else // G__OLDIMPLEMENTATION2191
            if ((param_type == 'Y' || param_type == 'Q' || 0 == libp->para[0].obj.i) &&
                (isupper(formal_type) || 'a' == formal_type)) {
               funclist->p_rate[i] = G__STDCONVMATCH;
            }
#endif // G__OLDIMPLEMENTATION2191
            break;
         }
      }

      /* user defined conversion */
      if (0 == recursive && G__NOMATCH == funclist->p_rate[i]) {
         if (formal_type == 'u') {
            struct G__ifunc_table_internal *ifunc2;
            int ifn2;
            int hash2;
            G__FastAllocString funcname2(G__ONELINE);
            struct G__param para;
            G__incsetup_memfunc(formal_tagnum);
            ifunc2 = G__struct.memfunc[formal_tagnum];
            para.paran = 1;
            para.para[0] = libp->para[i];
            long store_struct_offset = G__store_struct_offset;
            if (param_type == 'u') {
               G__store_struct_offset = libp->para[i].obj.i;
            } else {
               G__store_struct_offset = 0;
            }
            funcname2 = G__struct.name[formal_tagnum];
            G__hash(funcname2, hash2, ifn2);
            ifunc2 = G__overload_match(funcname2, &para, hash2, ifunc2
                                       , G__TRYCONSTRUCTOR, G__PUBLIC, &ifn2, 1
                                       , 1, false
                                      );
            G__store_struct_offset = store_struct_offset;
            if (ifunc2 && -1 != ifn2)
               funclist->p_rate[i] = G__USRCONVMATCH;
         }
      }

      if (0 == recursive && G__NOMATCH == funclist->p_rate[i]) {
         if (param_type == 'u' && -1 != param_tagnum) {
            struct G__ifunc_table_internal *ifunc2;
            int ifn2 = -1;
            int hash2;
            G__FastAllocString funcname2(G__ONELINE);
            struct G__param para;
            G__incsetup_memfunc(param_tagnum);
            para.paran = 0;
            long store_struct_offset = G__store_struct_offset;
            G__store_struct_offset = libp->para[i].obj.i;
            /* search for  operator type */
            funcname2.Format("operator %s"
                             , G__type2string(formal_type, formal_tagnum, -1, 0, 0));
            G__hash(funcname2, hash2, ifn2);
            ifunc2 = G__struct.memfunc[param_tagnum];
            ifunc2 = G__overload_match(funcname2, &para, hash2, ifunc2
                                       , G__TRYMEMFUNC, G__PUBLIC, &ifn2, 1
                                       , 1, false
                                      );
            if (!ifunc2) {
               /* search for  operator const type */
               funcname2.Format("operator %s"
                                , G__type2string(formal_type, formal_tagnum, -1, 0, 1));
               G__hash(funcname2, hash2, ifn2);
               ifunc2 = G__struct.memfunc[param_tagnum];
               ifunc2 = G__overload_match(funcname2, &para, hash2, ifunc2
                                          , G__TRYMEMFUNC, G__PUBLIC, &ifn2, 1
                                          , 1, false
                                         );
            }
            G__store_struct_offset = store_struct_offset;
            if (ifunc2 && -1 != ifn2)
               funclist->p_rate[i] = G__USRCONVMATCH;
         }
      }

      /* add up matching rate */
      if (G__NOMATCH == funclist->p_rate[i]) {
         funclist->rate = G__NOMATCH;
         break;
      }
      else {
         if (param_isconst != formal_isconst) {
            funclist->p_rate[i] += G__CVCONVMATCH;
         }
         /*
         if('u'==param_type && (0!=param_isconst&& 0==formal_isconst)) {
           funclist->p_rate[i]=G__NOMATCH;
           funclist->rate = G__NOMATCH;
         }
         else */
         if (G__NOMATCH != funclist->rate)
            funclist->rate += funclist->p_rate[i];
      }
   }
   if (G__NOMATCH != funclist->rate &&
         ((0 == G__isconst && (p_ifunc->isconst[ifn]&G__CONSTFUNC))
          || (G__isconst && 0 == (p_ifunc->isconst[ifn]&G__CONSTFUNC)))
      )
      funclist->rate += G__CVCONVMATCH;
}

//______________________________________________________________________________
int G__convert_param(G__param* libp, G__ifunc_table_internal* p_ifunc, int ifn, G__funclist* pmatch)
{
   // -- FIXME: Describe this function!
   int i;
   unsigned int rate;
   char param_type;
   char formal_type;
   int param_tagnum;
   int formal_tagnum;
   int formal_reftype;
   int formal_isconst;
   G__value* param;
   G__FastAllocString conv(G__ONELINE);
   G__FastAllocString arg1(G__ONELINE);
   long store_struct_offset;
   int store_tagnum;
   int store_isconst;
   int baseoffset;
   G__value reg;
   int store_oprovld;
   int rewindflag = 0;
   int recursive = 0;
   int rewind_arg;
   int match = 0;
   int store_exec_memberfunc = G__exec_memberfunc;
   // Allow for proper testing of static vs non-static calls.
   G__exec_memberfunc = 0;
   for (i = 0; i < libp->paran; ++i) {
      rate = pmatch->p_rate[i];
      param_type = libp->para[i].type;
      formal_type = p_ifunc->param[ifn][i]->type;
      param_tagnum = libp->para[i].tagnum;
      formal_tagnum = p_ifunc->param[ifn][i]->p_tagtable;
      param = &libp->para[i];
      formal_reftype = p_ifunc->param[ifn][i]->reftype;
#ifndef G__OLDIMPLEMENTATION
      rewind_arg = libp->paran - i - 1;
#else // G__OLDIMPLEMENTATION
      rewind_arg = p_ifunc->para_nu[ifn] - i - 1;
#endif // G__OLDIMPLEMENTATION
      formal_isconst = p_ifunc->param[ifn][i]->isconst;
      if (rate & G__USRCONVMATCH) {
         if (formal_type == 'u') {
            // try finding constructor
            if (param_type == 'u') {
               arg1.Format(
                    "(%s)0x%lx"
                  , G__fulltagname(param_tagnum, 1)
                  , param->obj.i
               );
            }
            else {
               G__valuemonitor(*param, arg1);
            }
            conv.Format("%s(%s)", G__struct.name[formal_tagnum], arg1());
            if (G__dispsource) {
               G__fprinterr(
                    G__serr
                  , "\n!!!Trying implicit conversion %s  %s:%d\n"
                  , conv()
                  , __FILE__
                  , __LINE__
               );
            }
            if (G__struct.iscpplink[formal_tagnum] != G__CPPLINK) {
               // Create a temp object to hold the result.
#ifdef G__ASM
               if (G__asm_noverflow) {
                  // We are generating bytecode.
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(
                          G__serr
                        , "%3x,%3x: ALLOCTEMP %s %d  %s:%d\n"
                        , G__asm_cp
                        , G__asm_dt
                        , G__struct.name[formal_tagnum]
                        , formal_tagnum
                        , __FILE__
                        , __LINE__
                     );
                  }
#endif // G__ASM_DBG
                  G__asm_inst[G__asm_cp] = G__ALLOCTEMP;
                  G__asm_inst[G__asm_cp+1] = formal_tagnum;
                  G__inc_cp_asm(2, 0);
               }
#endif // G__ASM
               G__alloc_tempobject(formal_tagnum, -1);
#ifdef G__ASM
               if (G__asm_noverflow) {
                  // We are generating bytecode.
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(
                          G__serr
                        , "%3x,%3x: SETTEMP  %s:%d\n"
                        , G__asm_cp
                        , G__asm_dt
                        , __FILE__
                        , __LINE__
                     );
                  }
#endif // G__ASM_DBG
                  G__asm_inst[G__asm_cp] = G__SETTEMP;
                  G__inc_cp_asm(1, 0);
               }
#endif // G__ASM
            }
            store_struct_offset = G__store_struct_offset;
            G__store_struct_offset = G__p_tempbuf->obj.obj.i;
            store_tagnum = G__tagnum;
            G__tagnum = formal_tagnum;
            store_isconst = G__isconst;
            G__isconst = formal_isconst;
            // --
            store_oprovld = G__oprovld;
            G__oprovld = 1; // Tell G__getfunction() to not stack args.
#ifdef G__ASM
            if (G__asm_noverflow && rewind_arg) {
               rewindflag = 1;
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(
                       G__serr
                     , "%3x,%3x: REWINDSTACK %d  %s:%d\n"
                     , G__asm_cp
                     , G__asm_dt
                     , rewind_arg
                     , __FILE__
                     , __LINE__
                  );
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__REWINDSTACK;
               G__asm_inst[G__asm_cp+1] = rewind_arg;
               G__inc_cp_asm(2, 0);
            }
#endif // G__ASM
            ++recursive;
            if (G__struct.iscpplink[formal_tagnum] == G__CPPLINK) {
               // compiled class
               reg = G__getfunction(conv, &match, G__TRYIMPLICITCONSTRUCTOR);
               if (match) {
                  // --
#ifdef G__ASM
                  if (G__asm_noverflow) {
                     // We are generating bytecode.
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(
                             G__serr
                           , "%3x,%3x: STORETEMP  %s:%d\n"
                           , G__asm_cp
                           , G__asm_dt
                           , __FILE__
                           , __LINE__
                        );
                     }
#endif // G__ASM_DBG
                     G__asm_inst[G__asm_cp] = G__STORETEMP;
                     G__inc_cp_asm(1, 0);
                  }
#endif // G__ASM
                  G__store_tempobject(reg);
               }
               else {
                  // FIXME: Generate bytecode for this!
                  G__pop_tempobject();
                  conv.Format(
                       "operator %s()"
                     , G__fulltagname(formal_tagnum, 1)
                  );
                  G__store_struct_offset = param->obj.i;
                  G__tagnum = param->tagnum;
                  if (G__tagnum != -1) {
                     reg = G__getfunction(conv, &match, G__TRYMEMFUNC);
                  }
                  if (!match) {
                     // FIXME: Generate bytecode for this!
                     G__store_tempobject(G__null);
                  }
               }
            }
            else {
               // interpreted class
               G__getfunction(conv, &match, G__TRYIMPLICITCONSTRUCTOR);
               if (match) {
                  // --
#ifdef G__ASM
                  if (G__asm_noverflow) {
                     // We are generating bytecode.
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(
                             G__serr
                           , "%3x,%3x: POPTEMP %d  %s:%d\n"
                           , G__asm_cp
                           , G__asm_dt
                           , formal_tagnum
                           , __FILE__
                           , __LINE__
                        );
                     }
#endif // G__ASM_DBG
                     G__asm_inst[G__asm_cp] = G__POPTEMP;
                     G__asm_inst[G__asm_cp+1] = formal_tagnum;
                     G__inc_cp_asm(2, 0);
                  }
#endif // G__ASM
                  G__p_tempbuf->obj.obj.i = G__store_struct_offset;
                  G__p_tempbuf->obj.ref = G__store_struct_offset;
               }
               else {
                  // No constructor found, try a user type conversion opr.
                  //--
                  // Discard the temporary object we allocated above.
                  G__pop_tempobject();
#ifdef G__ASM
                  if (G__asm_noverflow) {
                     // We are generating bytecode.
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(
                             G__serr
                           , "%3x,%3x: ALLOCTEMP,SETTEMP canceled  %s:%d\n"
                           , G__asm_cp
                           , G__asm_dt
                           , __FILE__
                           , __LINE__
                        );
                     }
#endif // G__ASM_DBG
                     G__inc_cp_asm(-3, 0);
                  }
#endif // G__ASM
                  // Generate the type conversion operator function call text.
                  conv.Format(
                       "operator %s()"
                     , G__fulltagname(formal_tagnum, 1)
                  );
                  // FIXME: Generate code to stack this for G__SETSTROS!
                  G__store_struct_offset = param->obj.i;
                  G__tagnum = param->tagnum;
#ifdef G__ASM
                  if (G__asm_noverflow) {
                     // We are generating bytecode.
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(
                             G__serr
                           , "%3x,%3x: PUSHSTROS  %s:%d\n"
                           , G__asm_cp
                           , G__asm_dt
                           , __FILE__
                           , __LINE__
                        );
                     }
#endif // G__ASM_DBG
                     G__asm_inst[G__asm_cp] = G__PUSHSTROS;
                     G__inc_cp_asm(1, 0);
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(
                             G__serr
                           , "%3x,%3x: SETSTROS  %s:%d\n"
                           , G__asm_cp
                           , G__asm_dt
                           , __FILE__
                           , __LINE__
                        );
                     }
#endif // G__ASM_DBG
                     G__asm_inst[G__asm_cp] = G__SETSTROS;
                     G__inc_cp_asm(1, 0);
                  }
#endif // G__ASM
                  reg = G__getfunction(conv, &match, G__TRYMEMFUNC);
                  if (!match) {
                     // --
#ifdef G__ASM
                     if (G__asm_noverflow) {
                        // We are generating bytecode.
#ifdef G__ASM_DBG
                        if (G__asm_dbg) {
                           G__fprinterr(
                                G__serr
                              , "ALLOCTEMP,SETTEMP canceled  %s:%d\n"
                              , __FILE__
                              , __LINE__
                           );
                        }
#endif // G__ASM_DBG
                        G__inc_cp_asm(-2, 0);
                        if (rewindflag) {
                           // --
#ifdef G__ASM_DBG
                           if (G__asm_dbg) {
                              G__fprinterr(
                                   G__serr
                                 , "%3x,%3x: REWINDSTACK  %s:%d\n"
                                 , G__asm_cp
                                 , G__asm_dt
                                 , __FILE__
                                 , __LINE__
                              );
                           }
#endif // G__ASM_DBG
                           //FIXME: These cp offsets might be wrong!
                           G__asm_inst[G__asm_cp-2] = G__REWINDSTACK;
                           G__asm_inst[G__asm_cp-1] = rewind_arg;
                        }
                     }
#endif // G__ASM
                     // --
                  }
                  else {
                     // User specified conversion function was found.
#ifdef G__ASM
                     if (G__asm_noverflow) {
                        // We are generating bytecode.
#ifdef G__ASM_DBG
                        if (G__asm_dbg) {
                           G__fprinterr(
                                G__serr
                              , "%3x,%3x: POPSTROS  %s:%d\n"
                              , G__asm_cp
                              , G__asm_dt
                              , __FILE__
                              , __LINE__
                           );
                        }
#endif // G__ASM_DBG
                        G__asm_inst[G__asm_cp] = G__POPSTROS;
                        G__inc_cp_asm(1, 0);
                     }
#endif // G__ASM
                     // --
                  }
               }
            }
            --recursive;
            G__oprovld = store_oprovld;
            G__isconst = store_isconst;
            G__tagnum = store_tagnum;
            G__store_struct_offset = store_struct_offset;
            // if no constructor, try converting to base class
            if (!match) {
               if (
                  (param_type == 'u') &&
#ifdef G__VIRTUALBASE
                  -1 != (baseoffset = G__ispublicbase(formal_tagnum,
                                          param_tagnum, param->obj.i))
#else // G__VIRTUALBASE
                  -1 != (baseoffset = G__ispublicbase(formal_tagnum,
                                          param_tagnum))
#endif // G__VIRTUALBASE
                  // --
               ) {
                  if (G__dispsource) {
                     G__fprinterr(
                          G__serr
                        , "!!!Implicit conversion from %s to base %s  %s:%d\n"
                        , G__struct.name[param_tagnum]
                        , G__struct.name[formal_tagnum]
                        , __FILE__
                        , __LINE__
                     );
                  }
                  param->typenum = -1;
                  param->tagnum = formal_tagnum;
                  param->obj.i += baseoffset;
                  param->ref += baseoffset;
#ifdef G__ASM
                  if (G__asm_noverflow) {
                     // We are generating bytecode.
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(
                             G__serr
                           , "%3x,%3x: BASECONV %d %d  %s:%d\n"
                           , G__asm_cp
                           , G__asm_dt
                           , formal_tagnum
                           , baseoffset
                           , __FILE__
                           , __LINE__
                        );
                     }
#endif // G__ASM_DBG
                     G__asm_inst[G__asm_cp] = G__BASECONV;
                     G__asm_inst[G__asm_cp+1] = formal_tagnum;
                     G__asm_inst[G__asm_cp+2] = baseoffset;
                     G__inc_cp_asm(3, 0);
                     if (rewind_arg) {
                        rewindflag = 1;
#ifdef G__ASM_DBG
                        if (G__asm_dbg) {
                           G__fprinterr(
                                G__serr
                              , "%3x,%3x: REWINDSTACK %d  %s:%d\n"
                              , G__asm_cp
                              , G__asm_dt
                              , -rewind_arg
                              , __FILE__
                              , __LINE__
                           );
                        }
#endif // G__ASM_DBG
                        G__asm_inst[G__asm_cp] = G__REWINDSTACK;
                        G__asm_inst[G__asm_cp+1] = -rewind_arg;
                        G__inc_cp_asm(2, 0);
                     }
#endif // G__ASM
                     // --
                  }
                  match = 1;
                  G__pop_tempobject();
               }
               else {
                  // all conversion failed
                  if (G__dispsource) {
                     G__fprinterr(
                          G__serr
                        , "!!!Implicit conversion %s tried, but failed  %s:%d\n"
                        , conv()
                        , __FILE__
                        , __LINE__
                     );
                  }
                  G__pop_tempobject();
#ifdef G__ASM
                  if (rewindflag) {
                     // --
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(
                             G__serr
                           , "REWINDSTACK canceled  %s:%d\n"
                           , __FILE__
                           , __LINE__
                        );
                     }
#endif // G__ASM_DBG
                     G__inc_cp_asm(-2, 0);
                  }
               }

#else // G__ASM
                  // all conversions failed
                  if (G__dispsource) {
                     G__fprinterr(
                          G__serr
                        , "!!!Implicit conversion %s tried, but failed\n"
                        , conv()
                     );
                  }
                  G__pop_tempobject();
#ifdef G__ASM
                  if (rewindflag) {
                     // --
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(G__serr, "REWINDSTACK cancelled\n");
                     }
#endif // G__ASM_DBG
                     G__inc_cp_asm(-2, 0);
                  }
#endif // G__ASM
#endif // G__ASM
                  // --
            }
            else {
               // Conversion successful.
#ifdef G__ASM
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(
                       G__serr
                     , "!!!Created temp object (%s) 0x%lx %d for "
                       "implicit conversion  %s:%d\n"
                     , conv()
                     , G__p_tempbuf->obj.obj.i
                     , G__templevel
                     , __FILE__
                     , __LINE__
                  );
               }
#endif // G__ASM_DBG
#endif // G__ASM
#ifdef G__ASM
               if (G__asm_noverflow && rewind_arg) {
                  rewindflag = 1;
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(
                          G__serr
                        , "%3x,%3x: REWINDSTACK %d  %s:%d\n"
                        , G__asm_cp
                        , G__asm_dt
                        , -rewind_arg
                        , __FILE__
                        , __LINE__
                     );
                  }
#endif // G__ASM_DBG
                  G__asm_inst[G__asm_cp] = G__REWINDSTACK;
                  G__asm_inst[G__asm_cp+1] = -rewind_arg;
                  G__inc_cp_asm(2, 0);
               }
#endif // G__ASM
               *param = G__p_tempbuf->obj;
            }
         }
         else if (param->tagnum != -1) {
            long store_struct_offset = G__store_struct_offset;
            int store_tagnum = G__tagnum;
            int store_isconst = G__isconst;
            conv.Format("operator %s()",
               G__type2string(formal_type, formal_tagnum, -1, 0, 0));
            G__store_struct_offset = param->obj.i;
            G__tagnum = param->tagnum;
#ifdef G__ASM
            if (G__asm_noverflow) {
               // We are generating bytecode.
               if (rewind_arg) {
                  rewindflag = 1;
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(
                          G__serr
                        , "%3x,%3x: REWINDSTACK %d  %s:%d\n"
                        , G__asm_cp
                        , G__asm_dt
                        , rewind_arg
                        , __FILE__
                        , __LINE__
                     );
                  }
#endif // G__ASM_DBG
                  G__asm_inst[G__asm_cp] = G__REWINDSTACK;
                  G__asm_inst[G__asm_cp+1] = rewind_arg;
                  G__inc_cp_asm(2, 0);
               }
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(
                       G__serr
                     , "%3x,%3x: PUSHSTROS  %s:%d\n"
                     , G__asm_cp
                     , G__asm_dt
                     , __FILE__
                     , __LINE__
                  );
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__PUSHSTROS;
               G__inc_cp_asm(1, 0);
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(
                       G__serr
                     , "%3x,%3x: SETSTROS  %s:%d\n"
                     , G__asm_cp
                     , G__asm_dt
                     , __FILE__
                     , __LINE__
                  );
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__SETSTROS;
               G__inc_cp_asm(1, 0);
            }
#endif // G__ASM
            reg = G__getfunction(conv, &match, G__TRYMEMFUNC);
            if (!match && formal_isconst) {
               conv.Format(
                    "operator const %s()"
                  , G__type2string(formal_type, formal_tagnum, -1, 0, 0)
               );
               G__store_struct_offset = param->obj.i;
               G__tagnum = param->tagnum;
               reg = G__getfunction(conv, &match, G__TRYMEMFUNC);
            }
            G__isconst = store_isconst;
            G__tagnum = store_tagnum;
            G__store_struct_offset = store_struct_offset;
#ifdef G__ASM
            if (G__asm_noverflow) {
               // We are generating bytecode.
               if (rewind_arg) {
                  rewindflag = 1;
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(
                          G__serr
                        , "%3x,%3x: REWINDSTACK %d  %s:%d\n"
                        , G__asm_cp
                        , G__asm_dt
                        , -rewind_arg
                        , __FILE__
                        , __LINE__
                     );
                  }
#endif // G__ASM_DBG
                  G__asm_inst[G__asm_cp] = G__REWINDSTACK;
                  G__asm_inst[G__asm_cp+1] = -rewind_arg;
                  G__inc_cp_asm(2, 0);
               }
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(
                       G__serr
                     , "%3x%3x: POPSTROS  %s:%d\n"
                     , G__asm_cp
                     , G__asm_dt
                     , __FILE__
                     , __LINE__
                  );
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__POPSTROS;
               G__inc_cp_asm(1, 0);
            }
#endif // G__ASM
            // fixing 'cout<<x' fundamental conversion opr with opr overloading
            // Not 100% sure if this is OK.
            if (match) {
               *param = reg;
            }
            else if (rewindflag) {
               // --
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(
                       G__serr
                     , "REWINDSTACK~ canceled  %s:%d\n"
                     , __FILE__
                     , __LINE__
                  );
               }
#endif // G__ASM_DBG
               G__inc_cp_asm(-7, 0);
            }
            else {
               // --
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(
                       G__serr
                     , "PUSHSTROS~ canceled  %s:%d\n"
                     , __FILE__
                     , __LINE__
                  );
               }
#endif // G__ASM_DBG
               G__inc_cp_asm(-3, 0);
            }
         }
         else {
            match = 0;
            if (G__dispsource && recursive) {
               G__valuemonitor(*param, arg1);
               G__fprinterr(
                    G__serr
                  , "!!!Recursive implicit conversion %s(%s) rejected  %s:%d\n"
                  , G__struct.name[formal_tagnum]
                  , arg1()
                  , __FILE__
                  , __LINE__
               );
            }
         }
         continue;
      }
      switch (formal_type) {
         case 'b':
         case 'c':
         case 'r':
         case 's':
         case 'h':
         case 'i':
         case 'k':
         case 'l':
            switch (param_type) {
               case 'd':
               case 'f':
                  /* std conv */
                  if (G__PARAREFERENCE == formal_reftype) {
                     param->obj.i = (long)param->obj.d;
                     param->type = formal_type;
                     param->ref = 0;
                  }
                  break;
            }
            break;
         case 'g':
            switch (param_type) {
               case 'd':
               case 'f':
                  /* std conv */
                  if (G__PARAREFERENCE == formal_reftype) {
                     param->obj.i = param->obj.d ? 1 : 0;
                     param->type = formal_type;
                     param->ref = 0;
                  }
                  break;
               case 'l':
               case 'i':
               case 's':
               case 'c':
               case 'h':
               case 'k':
               case 'r':
               case 'b':
                  if (G__PARAREFERENCE == formal_reftype) {
                     param->obj.i = (long)param->obj.i ? 1 : 0;
                     param->type = formal_type;
                     param->ref = 0;
                  }
                  break;
            }
            break;
         case 'n': /* long long */
            if (G__PARAREFERENCE == formal_reftype) {
               if (param->type != formal_type) param->ref = 0;
               param->type = formal_type;
               switch (param_type) {
                  case 'd':
                  case 'f':
                     param->obj.ll = (G__int64)param->obj.d;
                     break;
                  case 'g':
                  case 'c':
                  case 's':
                  case 'i':
                  case 'l':
                  case 'b':
                  case 'r':
                  case 'h':
                  case 'k':
                     param->obj.ll = param->obj.i;
                     break;
                  case 'm':
                     param->obj.ll = param->obj.ull;
                     break;
                  case 'q':
                     param->obj.ll = (G__int64)param->obj.ld;
                     break;
               }
            }
            break;
         case 'm': /* unsigned long long */
            if (G__PARAREFERENCE == formal_reftype) {
               if (param->type != formal_type) param->ref = 0;
               param->type = formal_type;
               switch (param_type) {
                  case 'd':
                  case 'f':
                     param->obj.ull = (G__uint64)param->obj.d;
                     break;
                  case 'g':
                  case 'c':
                  case 's':
                  case 'i':
                  case 'l':
                  case 'b':
                  case 'r':
                  case 'h':
                  case 'k':
                     param->obj.ull = param->obj.i;
                     break;
                  case 'n':
                     param->obj.ull = param->obj.ll;
                     break;
                  case 'q':
                     param->obj.ull = (G__int64)param->obj.ld;
                     break;
               }
            }
            break;
         case 'q': /* long double */
            if (G__PARAREFERENCE == formal_reftype) {
               if (param->type != formal_type) param->ref = 0;
               param->type = formal_type;
               switch (param_type) {
                  case 'd':
                  case 'f':
                     param->obj.ld = param->obj.d;
                     break;
                  case 'g':
                  case 'c':
                  case 's':
                  case 'i':
                  case 'l':
                  case 'b':
                  case 'r':
                  case 'h':
                  case 'k':
                     param->obj.ld = param->obj.i;
                     break;
                  case 'n':
                     param->obj.ld = (long double)param->obj.ll;
                     break;
                  case 'm':
                     param->obj.ld = (long double)param->obj.ld;
                     break;
               }
            }
            break;
         case 'd':
         case 'f':
            switch (param_type) {
               case 'b':
               case 'c':
               case 'r':
               case 's':
               case 'h':
               case 'i':
               case 'k':
               case 'l':
               case 'g':
               case 'n':
               case 'm':
                  /* std conv */
                  if (G__PARAREFERENCE == formal_reftype) {
                     param->obj.d = param->obj.i;
                     param->type = formal_type;
                     param->ref = 0;
                  }
                  break;
            }
            break;
         case 'u':
            switch (param_type) {
               case 'u':
                  if (0 == (rate&0xffffff00)) {
                     /* exact */
                     if ('e' == G__struct.type[param_tagnum]) {
                        if (param->ref) param->obj.i = *(long*)(param->ref);
                     }
                  }
                  else /* if(G__PARAREFERENCE==formal_reftype) */ {
                     if (-1 != (baseoffset = G__ispublicbase(formal_tagnum, param_tagnum
                                                             , param->obj.i))) {
                        param->tagnum = formal_tagnum;
                        param->obj.i += baseoffset;
                        param->ref = param->obj.i;
#ifdef G__ASM
                        if (G__asm_noverflow) {
                           if (rewind_arg
                              ) {
#ifdef G__ASM_DBG
                              if (G__asm_dbg) G__fprinterr(G__serr, "%3x: REWINDSTACK %d\n"
                                                              , G__asm_cp, rewind_arg);
#endif
                              G__asm_inst[G__asm_cp] = G__REWINDSTACK;
                              G__asm_inst[G__asm_cp+1] = rewind_arg;
                              G__inc_cp_asm(2, 0);
                           }
#ifdef G__ASM_DBG
                           if (G__asm_dbg) G__fprinterr(G__serr, "%3x: BASECONV %d %d\n"
                                                           , G__asm_cp, formal_tagnum, baseoffset);
#endif
                           G__asm_inst[G__asm_cp] = G__BASECONV;
                           G__asm_inst[G__asm_cp+1] = formal_tagnum;
                           G__asm_inst[G__asm_cp+2] = baseoffset;
                           G__inc_cp_asm(3, 0);
                           if (rewind_arg
                              ) {
#ifdef G__ASM_DBG
                              if (G__asm_dbg) G__fprinterr(G__serr, "%3x: REWINDSTACK %d\n"
                                                              , G__asm_cp, -rewind_arg);
#endif
                              G__asm_inst[G__asm_cp] = G__REWINDSTACK;
                              G__asm_inst[G__asm_cp+1] = -rewind_arg;
                              G__inc_cp_asm(2, 0);
                           }
                        }
#endif
                     }
                  }
                  break;
            }
            break;
         case 'U':
            switch (param_type) {
               case 'U':
                  /* Pointer to derived class can be converted to
                   * pointer to base class.
                   * add offset, modify char *parameter and
                   * G__value *param
                   */
                  if (-1 != (baseoffset = G__ispublicbase(formal_tagnum, param_tagnum
                                                          , param->obj.i))) {
                     param->tagnum = formal_tagnum;
                     param->obj.i += baseoffset;
                     param->ref += baseoffset;
#ifdef G__ASM
                     if (G__asm_noverflow) {
                        if (rewind_arg
                           ) {
#ifdef G__ASM_DBG
                           if (G__asm_dbg) G__fprinterr(G__serr, "%3x: REWINDSTACK %d\n"
                                                           , G__asm_cp, rewind_arg);
#endif
                           G__asm_inst[G__asm_cp] = G__REWINDSTACK;
                           G__asm_inst[G__asm_cp+1] = rewind_arg;
                           G__inc_cp_asm(2, 0);
                        }
#ifdef G__ASM_DBG
                        if (G__asm_dbg) G__fprinterr(G__serr, "%3x: BASECONV %d %d\n"
                                                        , G__asm_cp, formal_tagnum, baseoffset);
#endif
                        G__asm_inst[G__asm_cp] = G__BASECONV;
                        G__asm_inst[G__asm_cp+1] = formal_tagnum;
                        G__asm_inst[G__asm_cp+2] = baseoffset;
                        G__inc_cp_asm(3, 0);
                        if (rewind_arg
                           ) {
#ifdef G__ASM_DBG
                           if (G__asm_dbg) G__fprinterr(G__serr, "%3x: REWINDSTACK %d\n"
                                                           , G__asm_cp, -rewind_arg);
#endif
                           G__asm_inst[G__asm_cp] = G__REWINDSTACK;
                           G__asm_inst[G__asm_cp+1] = -rewind_arg;
                           G__inc_cp_asm(2, 0);
                        }
                     }
#endif
                  }
                  break;
            }
            break;
#ifndef G__OLDIMPLEMENTATION2191
         case '1':
#else // G__OLDIMPLEMENTATION2191
         case 'Q':
#endif // G__OLDIMPLEMENTATION2191
            if ('C' == param_type &&
                  p_ifunc->pentry[ifn]->size < 0
               ) {
               G__genericerror("Limitation: Precompiled function can not get pointer to interpreted function as argument");
               G__exec_memberfunc = store_exec_memberfunc;
               return(-1);
            }
      }
   }
   G__exec_memberfunc = store_exec_memberfunc;
   return 0;
}

//______________________________________________________________________________
void G__display_param(FILE* fp, int scopetagnum, const char* funcname, G__param* libp)
{
   // -- FIXME: Describe this function!
   int i;
#ifndef G__OLDIMPLEMENTATION1485
   if (G__serr == fp) {
      if (-1 != scopetagnum) G__fprinterr(G__serr, "%s::", G__fulltagname(scopetagnum, 1));
      G__fprinterr(G__serr, "%s(", funcname);
      for (i = 0;i < libp->paran;i++) {
         switch (libp->para[i].type) {
            case 'd':
            case 'f':
               G__fprinterr(G__serr, "%s", G__type2string(libp->para[i].type
                            , libp->para[i].tagnum
                            , libp->para[i].typenum
                            , 0
                            , 0));
               break;
            default:
               G__fprinterr(G__serr, "%s", G__type2string(libp->para[i].type
                            , libp->para[i].tagnum
                            , libp->para[i].typenum
                            , libp->para[i].obj.reftype.reftype
                            , 0));
               break;
         }
         if (i != libp->paran - 1) G__fprinterr(G__serr, ",");
      }
      G__fprinterr(G__serr, ");\n");
   }
   else {
#endif
      if (-1 != scopetagnum) fprintf(fp, "%s::", G__fulltagname(scopetagnum, 1));
      fprintf(fp, "%s(", funcname);
      for (i = 0;i < libp->paran;i++) {
         switch (libp->para[i].type) {
            case 'd':
            case 'f':
               fprintf(fp, "%s", G__type2string(libp->para[i].type
                                                , libp->para[i].tagnum
                                                , libp->para[i].typenum
                                                , 0
                                                , 0));
               break;
            default:
               fprintf(fp, "%s", G__type2string(libp->para[i].type
                                                , libp->para[i].tagnum
                                                , libp->para[i].typenum
                                                , libp->para[i].obj.reftype.reftype
                                                , 0));
               break;
         }
         if (i != libp->paran - 1) fprintf(fp, ",");
      }
      fprintf(fp, ");\n");
#ifndef G__OLDIMPLEMENTATION1485
   }
#endif
}

//______________________________________________________________________________
void G__display_func(FILE* fp, G__ifunc_table_internal* ifunc, int ifn)
{
   // -- FIXME: Describe this function!
   int i;
   int store_iscpp = G__iscpp;
   G__iscpp = 1;

   if (!ifunc || !ifunc->pentry[ifn]) return;

#ifndef G__OLDIMPLEMENTATION1485
   if (G__serr == fp) {
      if (ifunc->pentry[ifn]->filenum >= 0) { /* 2012 must leave this one */
         G__fprinterr(G__serr, "%-10s%4d "
                      , G__stripfilename(G__srcfile[ifunc->pentry[ifn]->filenum].filename)
                      , ifunc->pentry[ifn]->line_number);
      }
      else {
         G__fprinterr(G__serr, "%-10s%4d ", "(compiled)", 0);
      }
      G__fprinterr(G__serr, "%s ", G__type2string(ifunc->type[ifn]
                   , ifunc->p_tagtable[ifn]
                   , ifunc->p_typetable[ifn]
                   , ifunc->reftype[ifn]
                   , ifunc->isconst[ifn]));
      if (-1 != ifunc->tagnum) G__fprinterr(G__serr, "%s::", G__fulltagname(ifunc->tagnum, 1));
      G__fprinterr(G__serr, "%s(", ifunc->funcname[ifn]);
      for (i = 0;i < ifunc->para_nu[ifn];i++) {
         G__fprinterr(G__serr, "%s", G__type2string(ifunc->param[ifn][i]->type
                      , ifunc->param[ifn][i]->p_tagtable
                      , ifunc->param[ifn][i]->p_typetable
                      , ifunc->param[ifn][i]->reftype
                      , ifunc->param[ifn][i]->isconst));
         if (i != ifunc->para_nu[ifn] - 1) G__fprinterr(G__serr, ",");
      }
      G__fprinterr(G__serr, ");\n");
   }
   else {
#endif
      if (ifunc->pentry[ifn]->filenum >= 0) { /* 2012 must leave this one */
         fprintf(fp, "%-10s%4d "
                 , G__stripfilename(G__srcfile[ifunc->pentry[ifn]->filenum].filename)
                 , ifunc->pentry[ifn]->line_number);
      }
      else {
         fprintf(fp, "%-10s%4d ", "(compiled)", 0);
      }
      fprintf(fp, "%s ", G__type2string(ifunc->type[ifn]
                                        , ifunc->p_tagtable[ifn]
                                        , ifunc->p_typetable[ifn]
                                        , ifunc->reftype[ifn]
                                        , ifunc->isconst[ifn]));
      if (-1 != ifunc->tagnum) fprintf(fp, "%s::", G__fulltagname(ifunc->tagnum, 1));
      fprintf(fp, "%s(", ifunc->funcname[ifn]);
      for (i = 0;i < ifunc->para_nu[ifn];i++) {
         fprintf(fp, "%s", G__type2string(ifunc->param[ifn][i]->type
                                          , ifunc->param[ifn][i]->p_tagtable
                                          , ifunc->param[ifn][i]->p_typetable
                                          , ifunc->param[ifn][i]->reftype
                                          , ifunc->param[ifn][i]->isconst));
         if (i != ifunc->para_nu[ifn] - 1) fprintf(fp, ",");
      }
      fprintf(fp, ");\n");
#ifndef G__OLDIMPLEMENTATION1485
   }
#endif

   G__iscpp = store_iscpp;
}

//______________________________________________________________________________
void G__display_ambiguous(int scopetagnum, const char* funcname, G__param* libp, G__funclist* funclist, unsigned int bestmatch)
{
   // -- FIXME: Describe this function!
   G__fprinterr(G__serr, "Calling : ");
   G__display_param(G__serr, scopetagnum, funcname, libp);
   G__fprinterr(G__serr, "Match rank: file     line  signature\n");
   while (funclist) {
      struct G__ifunc_table_internal *ifunc = funclist->ifunc;
      int ifn = funclist->ifn;
      if (bestmatch == funclist->rate) G__fprinterr(G__serr, "* %8x ", funclist->rate);
      else                          G__fprinterr(G__serr, "  %8x ", funclist->rate);
      G__display_func(G__serr, ifunc, ifn);
      funclist = funclist->prev;
   }
}

/***********************************************************************
* G__add_templatefunc()
*
* Search matching template function, search by name then parameter.
* If match found, expand template, parse as pre-run
***********************************************************************/
//______________________________________________________________________________
struct G__funclist* G__add_templatefunc(const char* funcnamein, G__param* libp, int hash, G__funclist* funclist, G__ifunc_table_internal* p_ifunc, int isrecursive)
{
   // -- FIXME: Describe this function!
   struct G__Definetemplatefunc *deftmpfunc;
   struct G__Charlist call_para;
   /* int env_tagnum=G__get_envtagnum(); */
   int env_tagnum = p_ifunc->tagnum;
   struct G__inheritance *baseclass;
   int store_friendtagnum = G__friendtagnum;
   struct G__ifunc_table_internal *ifunc;
   int ifn;
   char *ptmplt;
   char *pexplicitarg = 0;

   G__FastAllocString funcname(funcnamein);

   if (-1 != env_tagnum) baseclass = G__struct.baseclass[env_tagnum];
   else               baseclass = &G__globalusingnamespace;
   if (0 == baseclass->basen) baseclass = (struct G__inheritance*)NULL;


   call_para.string = (char*)NULL;
   call_para.next = (struct G__Charlist*)NULL;
   deftmpfunc = &G__definedtemplatefunc;

   ptmplt = strchr(funcname, '<');
   if (ptmplt)
   {
      if (strncmp("operator", funcname, ptmplt - funcname) == 0) {
         /* We have operator< */
         if (ptmplt[1] == '<') ptmplt = strchr(ptmplt + 2, '<');
         else ptmplt = strchr(ptmplt + 1, '<');
      }
   }
   if (ptmplt)
   {
      if ((-1 != env_tagnum) && strcmp(funcname, G__struct.name[env_tagnum]) == 0) {
         /* this is probably a template constructor of a class template */
         ptmplt = (char*)0;
      }
      else {
         int tmp;
         *ptmplt = 0;
         if (G__defined_templatefunc(funcname)) {
            G__hash(funcname, hash, tmp);
         }
         else {
            pexplicitarg = ptmplt;
            *ptmplt = '<';
            ptmplt = (char*)0;
         }
      }
   }

   if (pexplicitarg)
   {
      /* funcname="f<int>" ->  funcname="f" , pexplicitarg="int>" */
      int tmp = 0;
      *pexplicitarg = 0;
      ++pexplicitarg;
      G__hash(funcname, hash, tmp);
   }

   /* Search matching template function name */
   while (deftmpfunc->next)
   {
      G__freecharlist(&call_para);
      if (ptmplt) {
         int itmp = 0;
         int ip = 1;
         int c;
         G__FastAllocString buf(G__ONELINE);
         do {
            c = G__getstream_template(ptmplt, &ip, buf, 0, ",>");
            G__checkset_charlist(buf, &call_para, ++itmp, 'u');
         }
         while (c != '>');
      }
      if (deftmpfunc->hash == hash && strcmp(deftmpfunc->name, funcname) == 0 &&
            (G__matchtemplatefunc(deftmpfunc, libp, &call_para, G__PROMOTION)
#ifndef G__OLDIMPLEMEMTATION2214
             || (pexplicitarg && libp->paran == 0)
#else
             || pexplicitarg
#endif
            )) {

         if (-1 != deftmpfunc->parent_tagnum &&
               env_tagnum != deftmpfunc->parent_tagnum) {
            if (baseclass) {
               int temp;
               for (temp = 0;temp < baseclass->basen;temp++) {
                  if (baseclass->herit[temp]->basetagnum == deftmpfunc->parent_tagnum) {
                     goto match_found;
                  }
               }
            }
            deftmpfunc = deftmpfunc->next;
            continue;
         }
match_found:
         G__friendtagnum = deftmpfunc->friendtagnum;

         if (pexplicitarg) {
            int npara = 0;
            G__gettemplatearglist(pexplicitarg, &call_para
                                  , deftmpfunc->def_para, &npara
                                  , -1
                                 );
         }

         if (pexplicitarg) {
            int tmp = 0;
            G__hash(funcname, hash, tmp);
         }

         /* matches funcname and parameter,
          * then expand the template and parse as prerun */
         G__replacetemplate(
            funcname
            , funcnamein
            , &call_para /* needs to make this up */
            , deftmpfunc->def_fp
            , deftmpfunc->line
            , deftmpfunc->filenum
            , &(deftmpfunc->def_pos)
            , deftmpfunc->def_para
            , 0
            , SHRT_MAX /* large enough number */
            , deftmpfunc->parent_tagnum
         );

         G__friendtagnum = store_friendtagnum;


         /* search for instantiated template function */
         ifunc = p_ifunc;
         while (ifunc && ifunc->next && ifunc->next->allifunc) ifunc = ifunc->next;
         if (ifunc) {
            ifn = ifunc->allifunc - 1;
            if (
               strcmp(funcnamein, ifunc->funcname[ifn]) == 0
            ) {
               if (ptmplt) {
                  int tmp;
                  *ptmplt = '<';
                  free((void*)ifunc->funcname[ifn]);
                  ifunc->funcname[ifn] = (char*)malloc(strlen(funcnamein) + 1);
                  strcpy(ifunc->funcname[ifn], funcnamein); // Okay we allocated enough space
                  G__hash(funcnamein, hash, tmp);
                  ifunc->hash[ifn] = hash;
               }
               if (0 == ifunc->pentry[ifn]->p && G__NOLINK == G__globalcomp) {
                  /* This was only a prototype template, search for definition
                   * template */
                  deftmpfunc = deftmpfunc->next;
                  continue;
               }
               funclist = G__funclist_add(funclist, ifunc, ifn, 0);
               if (ifunc->para_nu[ifn] < libp->paran ||
                     (ifunc->para_nu[ifn] > libp->paran &&
                      !ifunc->param[ifn][libp->paran]->pdefault)) {
                  funclist->rate = G__NOMATCH;
               }
               else {
                  G__rate_parameter_match(libp, ifunc, ifn, funclist, isrecursive);
               }
            }
         }
         G__freecharlist(&call_para);
      }
      deftmpfunc = deftmpfunc->next;
   }
   G__freecharlist(&call_para);

   return funclist;
}

//______________________________________________________________________________
struct G__funclist* G__rate_binary_operator(G__ifunc_table_internal* p_ifunc, G__param* libp, int tagnum, const char* funcname, int hash, G__funclist* funclist, int isrecursive)
{
   // -- FIXME: Describe this function!
   int i;
   struct G__param fpara;
#ifdef G__DEBUG
   {
      int jdbg;
      int sizedbg = sizeof(struct G__param);
      char *pcdbg = (char*)(&fpara);
      for (jdbg = 0;jdbg < (int)sizedbg;jdbg++) {
         *(pcdbg + jdbg) = (char)0xa3;
      }
   }
#endif

   /* set 1st argument as the object */
   fpara.para[0].type = 'u';
   fpara.para[0].tagnum = tagnum;
   fpara.para[0].typenum = -1;
   fpara.para[0].obj.i = G__store_struct_offset;
   fpara.para[0].ref = G__store_struct_offset;
   fpara.para[0].isconst = G__isconst;

   /* set 2nd to n arguments */
   fpara.paran = libp->paran + 1;
   for (i = 0;i < libp->paran;i++) fpara.para[i+1] = libp->para[i];

   /* Search for name match
    *  if reserved func or K&R, match immediately
    *  check number of arguments and default parameters
    *  rate parameter match */
   while (p_ifunc)
   {
      int ifn;
      for (ifn = 0;ifn < p_ifunc->allifunc;++ifn) {
         if (hash == p_ifunc->hash[ifn] && strcmp(funcname, p_ifunc->funcname[ifn]) == 0) {
            if (p_ifunc->para_nu[ifn] < fpara.paran ||
                  (p_ifunc->para_nu[ifn] > fpara.paran &&
                   !p_ifunc->param[ifn][fpara.paran]->pdefault)
                  || (isrecursive && p_ifunc->isexplicit[ifn])
            ) {}
            else {
               funclist = G__funclist_add(funclist, p_ifunc, ifn, 0);
               G__rate_parameter_match(&fpara, p_ifunc, ifn, funclist, isrecursive);
               funclist->ifunc = 0; /* added as dummy */
            }
         }
      }
      p_ifunc = p_ifunc->next;
   }

   return(funclist);
}

//______________________________________________________________________________
int G__identical_function(G__funclist* match, G__funclist* func)
{
   // -- FIXME: Describe this function!
   int ipara;
   if (!match || !match->ifunc || !func || !func->ifunc) return(0);
   for (ipara = 0;ipara < match->ifunc->para_nu[match->ifn];ipara++) {
      if (
         (match->ifunc->param[match->ifn][ipara]->type !=
          func->ifunc->param[func->ifn][ipara]->type) ||
         (match->ifunc->param[match->ifn][ipara]->p_tagtable !=
          func->ifunc->param[func->ifn][ipara]->p_tagtable) ||
         (match->ifunc->param[match->ifn][ipara]->p_typetable !=
          func->ifunc->param[func->ifn][ipara]->p_typetable) ||
         (match->ifunc->param[match->ifn][ipara]->isconst !=
          func->ifunc->param[func->ifn][ipara]->isconst) ||
         (match->ifunc->param[match->ifn][ipara]->reftype !=
          func->ifunc->param[func->ifn][ipara]->reftype)
      ) {
         return(0);
      }
   }

   return(1);
}

//______________________________________________________________________________
struct G__ifunc_table_internal* G__overload_match(const char* funcname, G__param* libp, int hash, G__ifunc_table_internal* p_ifunc, int memfunc_flag, int access, int* pifn, int isrecursive, int doconvert, int check_access)
{
   // -- FIXME: Describe this function!
   struct G__funclist* funclist = 0;
   struct G__funclist* match = 0;
   unsigned int bestmatch = G__NOMATCH;
   struct G__funclist* func;
   int ambiguous = 0;
   int scopetagnum = p_ifunc->tagnum;
   struct G__ifunc_table_internal *store_ifunc = p_ifunc;
   int ix = 0;
#ifdef G__ASM
   int active_run = doconvert && !G__asm_wholefunction && !G__asm_noverflow && !(G__no_exec_compile==1 && funcname[0]=='~' /* loop compilation of temporary destruction */);
#else
   int active_run = doconvert;
#endif

   /* Search for name match
    *  if reserved func or K&R, match immediately
    *  check number of arguments and default parameters
    *  rate parameter match */
   while (p_ifunc)
   {
      int ifn;
      for (ifn = 0;ifn < p_ifunc->allifunc;++ifn) {
         if (hash == p_ifunc->hash[ifn] && strcmp(funcname, p_ifunc->funcname[ifn]) == 0) {
            if (p_ifunc->ansi[ifn] == 0 || /* K&R C style header */
                  p_ifunc->ansi[ifn] == 2 || /* variable number of args */
                  (G__HASH_MAIN == hash && strcmp(funcname, "main") == 0)) {
               /* immediate return for special match */
               doconvert = 0;
               *pifn = ifn;
               goto end_of_function;
            }
            if (-1 != p_ifunc->tagnum &&
                  (memfunc_flag == G__TRYNORMAL && doconvert)
                  && strcmp(G__struct.name[p_ifunc->tagnum], funcname) == 0) {
               continue;
            }
            funclist = G__funclist_add(funclist, p_ifunc, ifn, 0);
            if (p_ifunc->para_nu[ifn] < libp->paran ||
                  (p_ifunc->para_nu[ifn] > libp->paran &&
                   !p_ifunc->param[ifn][libp->paran]->pdefault)
                  || (isrecursive && p_ifunc->isexplicit[ifn])
               ) {
               funclist->rate = G__NOMATCH;
            }
            else {
               G__rate_parameter_match(libp, p_ifunc, ifn, funclist, isrecursive);
            }
            if (G__EXACTMATCH == (funclist->rate&0xffffff00)) match = funclist;
         }
      }
      p_ifunc = p_ifunc->next;
      if (!p_ifunc && store_ifunc == G__p_ifunc &&
            ix < G__globalusingnamespace.basen) {
         p_ifunc = G__struct.memfunc[G__globalusingnamespace.herit[ix]->basetagnum];
         ++ix;
      }
   }

   /* If exact match does not exist
    *    search for template func
    *    rate parameter match */
   if (!match)
   {
      funclist =  G__add_templatefunc(funcname, libp, hash, funclist
                                      , store_ifunc, isrecursive);
   }

   if (!match && (G__TRYUNARYOPR == memfunc_flag || G__TRYBINARYOPR == memfunc_flag))
   {
      for (ix = 0;ix < G__globalusingnamespace.basen;ix++) {
         funclist = G__rate_binary_operator(
                       G__struct.memfunc[G__globalusingnamespace.herit[ix]->basetagnum]
                       , libp, G__tagnum, funcname, hash
                       , funclist, isrecursive);
      }
      funclist = G__rate_binary_operator(&G__ifunc, libp, G__tagnum, funcname, hash
                                         , funclist, isrecursive);
   }

   /* if there is no name match, return null */
   if ((struct G__funclist*)NULL == funclist) return((struct G__ifunc_table_internal*)NULL);
   /* else  there is function name match */


   /*  choose the best match
    *    display error if the call is ambiguous
    *    display error if there is no parameter match */
   func = funclist;
   ambiguous = 0;
   while (func)
   {
      if (func->rate < bestmatch) {
         bestmatch = func->rate;
         match = func;
         ambiguous = 0;
      }
      else if (func->rate == bestmatch && bestmatch != G__NOMATCH) {
         if (0 == G__identical_function(match, func)) ++ambiguous;
         match = func;
      }
      func = func->prev;
   }

   if ((G__TRYUNARYOPR == memfunc_flag || G__TRYBINARYOPR == memfunc_flag) &&
         match && 0 == match->ifunc)
   {
      G__funclist_delete(funclist);
      return((struct G__ifunc_table_internal*)NULL);
   }

#ifdef G__ASM_DBG
   /* #define G__ASM_DBG2 */
#endif
#ifdef G__ASM_DBG2
   if (G__dispsource)
      G__display_ambiguous(scopetagnum, funcname, libp, funclist, bestmatch);
#endif

   if (!match)
   {
#if G__NEVER
      G__genericerror("Error: No appropriate match in the scope");
      *pifn = -1;
#endif
      G__funclist_delete(funclist);
      return((struct G__ifunc_table_internal*)NULL);
   }

   if (ambiguous && G__EXACTMATCH != bestmatch
         && !isrecursive
      )
   {
      if (!G__mask_error) {
         /* error, ambiguous overloading resolution */
         G__fprinterr(G__serr, "Error: Ambiguous overload resolution (%x,%d)"
                      , bestmatch, ambiguous + 1);
         G__genericerror((char*)NULL);
         G__display_ambiguous(scopetagnum, funcname, libp, funclist, bestmatch);
      }
      *pifn = -1;
      G__funclist_delete(funclist);
      return((struct G__ifunc_table_internal*)NULL);
   }

   /* best match function found */
   p_ifunc = match->ifunc;
   *pifn = match->ifn;

end_of_function:
   /*  check private, protected access rights, and static-ness
    *    display error if no access right
    *    do parameter conversion if needed */
   if (check_access) {
      if (0 == (p_ifunc->access[*pifn]&access) && (!G__isfriend(p_ifunc->tagnum))
            && G__NOLINK == G__globalcomp
         )
      {
         /* no access right */
         G__fprinterr(G__serr, "Error: can not call private or protected function");
         G__genericerror((char*)NULL);
         G__fprinterr(G__serr, "  ");
         G__display_func(G__serr, p_ifunc, *pifn);
         G__display_ambiguous(scopetagnum, funcname, libp, funclist, bestmatch);
         *pifn = -1;
         G__funclist_delete(funclist);
         return((struct G__ifunc_table_internal*)NULL);
      }
      if (active_run && G__exec_memberfunc && G__getstructoffset()==0 && p_ifunc->tagnum != -1 && G__struct.type[p_ifunc->tagnum]!='n' && !p_ifunc->staticalloc[*pifn] && G__NOLINK == G__globalcomp
          && (G__TRYCONSTRUCTOR !=  memfunc_flag && G__CALLCONSTRUCTOR != memfunc_flag) ) {
         /* non static function called without an object */
         G__fprinterr(G__serr, "Error: cannot call member function without object");
         G__genericerror((char*)NULL);
         G__fprinterr(G__serr, "  ");
         G__display_func(G__serr, p_ifunc, *pifn);
         G__display_ambiguous(scopetagnum, funcname, libp, funclist, bestmatch);
         G__funclist_delete(funclist);
         *pifn = -1;
         return((struct G__ifunc_table_internal*)NULL);
      }
   }
   
   /* convert parameter */
   if (
      doconvert &&
      G__convert_param(libp, p_ifunc, *pifn, match))
      return((struct G__ifunc_table_internal*)NULL);

   G__funclist_delete(funclist);
   return(p_ifunc);
}

//______________________________________________________________________________
int G__interpret_func(G__value* result7, const char* funcname, G__param* libp, int hash, G__ifunc_table_internal* p_ifunc, int funcmatch, int memfunc_flag)
{
   // -- FIXME: Describe this function!
   //  return 1 if function is executed.
   //  return 0 if function isn't executed.
   int ifn = 0;
   struct G__var_array G_local;
   FILE *prev_fp;
   fpos_t prev_pos;
   // paraname[][] is used only for K&R func param. length should be OK.
   G__FastAllocString paraname[G__MAXFUNCPARA];
   int ipara = 0;
   int cin = '\0';
   int itemp = 0;
   int break_exit_func;
   int store_decl;
   G__value buf;
   int store_var_type;
   int store_tagnum;
   long store_struct_offset;
   int store_inherit_tagnum;
   long store_inherit_offset;
   int iexist, virtualtag;
   int store_def_struct_member;
   int store_var_typeB;
   int store_doingconstruction;
   int store_func_now;
   int store_func_page;
   int store_iscpp;
   int store_exec_memberfunc;
   G__UINT32 store_security;
#ifdef G__ASM_IFUNC
   G__FastAllocString asm_inst_g_sb(G__MAXINST * sizeof(long));
   long *asm_inst_g = (long*) asm_inst_g_sb.data(); // p-code instruction buffer.
   G__FastAllocString asm_stack_g_sb(G__MAXSTACK * sizeof(G__value));
   G__value *asm_stack_g = (G__value*) asm_stack_g_sb.data(); // data stack 
   G__FastAllocString asm_name_sb(G__ASM_FUNCNAMEBUF);
   char *asm_name = asm_name_sb;
   long *store_asm_inst;
   int store_asm_instsize;
   G__value *store_asm_stack;
   char *store_asm_name;
   int store_asm_name_p;
   struct G__param *store_asm_param;
   int store_asm_exec;
   int store_asm_noverflow;
   int store_asm_cp;
   int store_asm_dt;
   int store_asm_index;
#endif
#ifdef G__ASM_WHOLEFUNC
   int store_no_exec_compile = 0;
   struct G__var_array* localvar = 0;
#endif
#ifdef G__NEWINHERIT
   int basen = 0;
   int isbase;
   int access;
   int memfunc_or_friend = 0;
   struct G__inheritance* baseclass = 0;
#endif
   int local_tagnum = 0;
   struct G__ifunc_table_internal* store_p_ifunc = p_ifunc;
   int specialflag = 0;
   long store_memberfunc_struct_offset;
   int store_memberfunc_tagnum;
#ifndef G__OLDIMPLEMENTATION2038
   G_local.enclosing_scope = 0;
   G_local.inner_scope = 0;
#endif
#ifdef G__NEWINHERIT
   store_inherit_offset = G__store_struct_offset;
   store_inherit_tagnum = G__tagnum;
#endif
   store_asm_noverflow = G__asm_noverflow;
   //
   // Skip the search for the function if we are
   // called by running bytecode, the search was
   // done at bytecode compilation time.
   //
#ifdef G__ASM_IFUNC
   if (G__asm_exec) {
      ifn = G__asm_index;
      // Delete 0 ~destructor ignored.
      if (
         !G__store_struct_offset &&
         (p_ifunc->tagnum != -1) &&
         !p_ifunc->staticalloc[ifn] &&
         (p_ifunc->funcname[ifn][0] == '~')
      ) {
         return 1;
      }
      goto asm_ifunc_start;
   }
#endif
   //
   // Search for function.
   //
#ifdef G__NEWINHERIT
   if (
      (G__exec_memberfunc && (-1 != G__tagnum || -1 != G__memberfunc_tagnum)) ||
      G__TRYNORMAL != memfunc_flag
   ) {
      isbase = 1;
      basen = 0;
      if (G__exec_memberfunc && -1 == G__tagnum) {
         local_tagnum = G__memberfunc_tagnum;
      }
      else {
         local_tagnum = G__tagnum;
      }
      baseclass = G__struct.baseclass[local_tagnum];
      if (G__exec_memberfunc || G__isfriend(G__tagnum)) {
         access = G__PUBLIC_PROTECTED_PRIVATE;
         memfunc_or_friend = 1;
      }
      else {
         access = G__PUBLIC;
         memfunc_or_friend = 0;
      }
   }
   else {
      access = G__PUBLIC;
      isbase = 0;
      if (p_ifunc && p_ifunc == G__p_ifunc) {
         basen = 0;
         isbase = 1;
         baseclass = &G__globalusingnamespace;
      }
   }
   next_base:
#endif
   p_ifunc = G__overload_match(funcname, libp, hash, p_ifunc, memfunc_flag, access, &ifn, 0, 1, true);
   if (ifn == -1) {
      *result7 = G__null;
      return 1;
   }
#ifdef G__NEWINHERIT
   // Iteration for base class member function search.
   if (
      !p_ifunc ||
      (
         (p_ifunc->access[ifn] != G__PUBLIC) &&
         !G__isfriend(G__tagnum) &&
         (
            !G__exec_memberfunc ||
            (
               (local_tagnum != G__memberfunc_tagnum) &&
               (
                  (p_ifunc->access[ifn] != G__PROTECTED) ||
                  (G__ispublicbase(local_tagnum, G__memberfunc_tagnum, store_inherit_offset) == -1)
               )
            )
         )
      )
   ) {
      if (isbase) {
         while (baseclass && basen < baseclass->basen) {
            if (memfunc_or_friend) {
               if ((baseclass->herit[basen]->baseaccess&G__PUBLIC_PROTECTED) ||
                     baseclass->herit[basen]->property&G__ISDIRECTINHERIT) {
                  access = G__PUBLIC_PROTECTED;
                  G__incsetup_memfunc(baseclass->herit[basen]->basetagnum);
                  p_ifunc = G__struct.memfunc[baseclass->herit[basen]->basetagnum];
#ifdef G__VIRTUALBASE
                  // require !G__prerun, else store_inherit_offset might not point
                  // to a valid object with vtable, and G__getvirtualbaseoffset()
                  // might fail. We should not need the voffset in this case
                  // anyway, as we don't actually call the function.
                  if (!G__prerun &&
                      baseclass->herit[basen]->property&G__ISVIRTUALBASE) {
                     G__store_struct_offset = store_inherit_offset +
                                              G__getvirtualbaseoffset(store_inherit_offset, G__tagnum
                                                                      , baseclass, basen);
                     if (G__cintv6) {
                        G__bc_VIRTUALADDSTROS(G__tagnum, baseclass, basen);
                     }
                  }
                  else {
                     G__store_struct_offset
                     = store_inherit_offset + baseclass->herit[basen]->baseoffset;
                  }
#else
                  G__store_struct_offset
                  = store_inherit_offset + baseclass->herit[basen]->baseoffset;
#endif
                  G__tagnum = baseclass->herit[basen]->basetagnum;
                  ++basen;
                  store_p_ifunc = p_ifunc;
                  goto next_base; /* I know this is a bad manner */
               }
            }
            else {
               if (baseclass->herit[basen]->baseaccess&G__PUBLIC) {
                  access = G__PUBLIC;
                  G__incsetup_memfunc(baseclass->herit[basen]->basetagnum);
                  p_ifunc = G__struct.memfunc[baseclass->herit[basen]->basetagnum];
#ifdef G__VIRTUALBASE
                  // require !G__prerun, else store_inherit_offset might not point
                  // to a valid object with vtable, and G__getvirtualbaseoffset()
                  // might fail. We should not need the voffset in this case
                  // anyway, as we don't actually call the function.
                  if (!G__prerun &&
                      baseclass->herit[basen]->property&G__ISVIRTUALBASE) {
                     G__store_struct_offset = store_inherit_offset +
                                              G__getvirtualbaseoffset(store_inherit_offset, G__tagnum
                                                                      , baseclass, basen);
                     if (G__cintv6) {
                        G__bc_VIRTUALADDSTROS(G__tagnum, baseclass, basen);
                     }
                  }
                  else {
                     G__store_struct_offset
                     = store_inherit_offset + baseclass->herit[basen]->baseoffset;
                  }
#else
                  G__store_struct_offset
                  = store_inherit_offset + baseclass->herit[basen]->baseoffset;
#endif
                  G__tagnum = baseclass->herit[basen]->basetagnum;
                  ++basen;
                  store_p_ifunc = p_ifunc;
                  goto next_base; /* I know this is a bad manner */
               }
            }
            ++basen;
         }
         isbase = 0;
      }
      if (
         !specialflag &&
         (libp->paran == 1) &&
         (libp->para[0].tagnum != -1) &&
         (G__struct.parent_tagnum[libp->para[0].tagnum] != -1)
      ) {
         p_ifunc = G__struct.memfunc[G__struct.parent_tagnum[libp->para[0].tagnum]];
         switch (G__struct.type[p_ifunc->tagnum]) {
            case 's':
            case 'c':
               store_p_ifunc = p_ifunc;
               specialflag = 1;
               goto next_base;
         }
      }
      // Not found.
      G__store_struct_offset = store_inherit_offset;
      G__tagnum = store_inherit_tagnum;
      G__asm_noverflow = store_asm_noverflow;
      return 0;
   }
#else
   // If no such func, return 0.
   if (!p_ifunc) {
      return 0;
   }
   //
   // We have now found the function to execute.
   //
   // Member access control.
   if (G__PUBLIC != p_ifunc->access[ifn] && !G__isfriend(G__tagnum)) {
      return 0;
   }
#endif
   asm_ifunc_start: // Loop compilation execution label.
   if (!p_ifunc->hash[ifn]) {
      return 0;
   }
   // Parameter analysis with -c option, return without call.
   if (G__globalcomp) {
      // -- With -c-1 or -c-2 option, return immediately.
      result7->obj.d = 0.0;
      result7->ref = 0;
      result7->type = p_ifunc->type[ifn];
      result7->tagnum = p_ifunc->p_tagtable[ifn];
      result7->typenum = p_ifunc->p_typetable[ifn];
#ifndef G__OLDIMPLEMENTATION1259
      result7->isconst = p_ifunc->isconst[ifn];
#endif
      if (isupper(result7->type)) {
         result7->obj.reftype.reftype = p_ifunc->reftype[ifn];
      }
      return 1;
   }
   // Constructor or destructor call in G__make_ifunctable() parameter
   // type allocation.  Return without call.
   if (G__prerun) {
      // -- In G__make_ifunctable() parameter allocation, return immediately.
      result7->obj.i = p_ifunc->type[ifn];
      result7->ref = 0;
      result7->type = G__DEFAULT_FUNCCALL;
      result7->tagnum = p_ifunc->p_tagtable[ifn];
      result7->typenum = p_ifunc->p_typetable[ifn];
#ifndef G__OLDIMPLEMENTATION1259
      result7->isconst = p_ifunc->isconst[ifn];
#endif
      if (isupper(result7->type)) {
         result7->obj.reftype.reftype = p_ifunc->reftype[ifn];
      }
      return 1;
   }
   
   // 24-05-07
   // We had a check in ifunc to verify that it has a stub pointer.
   // For stub-less funcs this pointer may be null.
   // So check that and that the function has not been register yet;
   // error if body not defined.
   if (
      //
#ifdef G__ASM_WHOLEFUNC
      !p_ifunc->pentry[ifn]->p &&
      !p_ifunc->ispurevirtual[ifn] &&
      (G__asm_wholefunction  == G__ASM_FUNC_NOP) &&
      !G__get_funcptr(p_ifunc, ifn) &&
      p_ifunc->hash[ifn]
#else
      !p_ifunc->pentry[ifn]->p &&
      !G__get_funcptr(p_ifunc, ifn) &&
      !p_ifunc->ispurevirtual[ifn]
#endif
      //
   ) {
      if (!G__templatefunc(result7, funcname, libp, hash, funcmatch)) {
         if (funcmatch == G__USERCONV) {
            *result7 = G__null;
            bool isCompiled = (p_ifunc->pentry[ifn]->size == -1);
            if (p_ifunc->isvirtual[ifn] && p_ifunc->tagnum >= 0 && isCompiled && G__method_inbase(ifn, p_ifunc))
               G__fprinterr(G__serr, "Error: %s() declared but no dictionary for the base class", funcname);
            else if (isCompiled)
               G__fprinterr(G__serr, "Error: no dictionary for function %s()", funcname);
            else
               G__fprinterr(G__serr, "Error: %s() declared but not defined", funcname);
            G__genericerror(0);
            return 1;
         }
         return 0;
      }
      return 1;
   }
   // Add baseoffset if calling base class member function.
   // Resolution of virtual function is not done here. There is a
   // separate section down below. Search string 'virtual function'
   // to get there.
   G__tagnum = p_ifunc->tagnum;
   store_exec_memberfunc = G__exec_memberfunc;
   if ((G__tagnum == -1) && (G__memberfunc_tagnum == -1)) {
      G__exec_memberfunc = 0;
   }
   // Set the variable type for the function return value.
   store_var_typeB = G__var_typeB;
   G__var_typeB = 'p';
#ifdef G__NEWINHERIT
#ifdef G__ASM
   if (G__asm_noverflow && G__store_struct_offset &&
         G__store_struct_offset != store_inherit_offset) {
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x: ADDSTROS %ld\n", G__asm_cp, G__store_struct_offset - store_inherit_offset);
      }
#endif
      G__asm_inst[G__asm_cp] = G__ADDSTROS;
      G__asm_inst[G__asm_cp+1] = G__store_struct_offset - store_inherit_offset;
      G__inc_cp_asm(2, 0);
   }
#endif
#endif
   //
   //  Handle a linked-in, compiled function now.
   //
   if (p_ifunc->pentry[ifn]->size == -1) {
      // -- C++ compiled function.
#ifdef G__ROOT
      if (
         memfunc_flag == G__CALLCONSTRUCTOR ||
         memfunc_flag == G__TRYCONSTRUCTOR  ||
         memfunc_flag == G__TRYIMPLICITCONSTRUCTOR
      ) {
         G__exec_alloc_lock();
#ifdef G__ASM
         if (G__asm_noverflow) {
            //
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x: ROOTOBJALLOCBEGIN\n" , G__asm_cp);
            }
#endif
            G__asm_inst[G__asm_cp] = G__ROOTOBJALLOCBEGIN;
            G__inc_cp_asm(1, 0);
         }
#endif
         //
      }
#endif
      //
      //  Call the function now.
      //
      G__call_cppfunc(result7, libp, p_ifunc, ifn);
      //
      //  Function is done, cleanup.
      //
#ifdef G__ROOT
      if (
         memfunc_flag == G__CALLCONSTRUCTOR ||
         memfunc_flag == G__TRYCONSTRUCTOR  ||
         memfunc_flag == G__TRYIMPLICITCONSTRUCTOR
      ) {
         G__exec_alloc_unlock();
#ifdef G__ASM
         if (G__asm_noverflow) {
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3x: ROOTOBJALLOCEND\n" , G__asm_cp);
#endif
            G__asm_inst[G__asm_cp] = G__ROOTOBJALLOCEND;
            G__inc_cp_asm(1, 0);
         }
#endif
         //
      }
#endif
      // Recover tag environment.
      G__store_struct_offset = store_inherit_offset;
      G__tagnum = store_inherit_tagnum;
      if (G__tagnum != -1) {
         G__incsetup_memvar(G__tagnum);
         if (
            (G__struct.virtual_offset[G__tagnum] != -1) &&
            !strcmp(funcname, G__struct.name[G__tagnum])
         ) {
            long* pvtag = (long*) (result7->obj.i + G__struct.virtual_offset[G__tagnum]);
            *pvtag = G__tagnum;
         }
      }
      if (store_var_typeB == 'P') {
         G__val2pointer(result7);
      }
#ifdef G__NEWINHERIT
      //
#ifdef G__ASM
      if (G__asm_noverflow && G__store_struct_offset && (G__store_struct_offset != store_inherit_offset)) {
         //
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x: ADDSTROS %ld\n", G__asm_cp, -G__store_struct_offset + store_inherit_offset);
         }
#endif
         G__asm_inst[G__asm_cp] = G__ADDSTROS;
         G__asm_inst[G__asm_cp+1] = -G__store_struct_offset + store_inherit_offset;
         G__inc_cp_asm(2, 0);
      }
#endif // G__ASM
      //
#endif // G__NEWINHERIT
      G__exec_memberfunc = store_exec_memberfunc;
      return 1;
   }
#ifdef G__ASM
   // Create bytecode instruction for calling interpreted function.
   if (G__asm_noverflow) {
      if (G__cintv6) {
         if (p_ifunc->isvirtual[ifn] && !G__fixedscope) {
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(
                    G__serr
                  , "%3x,%3x: LD_FUNC virtual '%s' paran: %d  %s:%d\n"
                  , G__asm_cp
                  , G__asm_dt
                  , funcname
                  , libp->paran
                  , __FILE__
                  , __LINE__
               );
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__LD_FUNC;
            G__asm_inst[G__asm_cp+1] = p_ifunc->tagnum;
            G__asm_inst[G__asm_cp+2] = (p_ifunc->vtblindex[ifn] & 0xffff) + (p_ifunc->vtblbasetagnum[ifn] * 0x10000);
            G__asm_inst[G__asm_cp+3] = libp->paran;
            G__asm_inst[G__asm_cp+4] = (long) G__bc_exec_virtual_bytecode;
            G__asm_inst[G__asm_cp+5] = 0;
            if (p_ifunc && p_ifunc->pentry[ifn]) {
               G__asm_inst[G__asm_cp+5] = p_ifunc->pentry[ifn]->ptradjust;
            }
            G__asm_inst[G__asm_cp+6] = (long) p_ifunc;
            G__asm_inst[G__asm_cp+7] = (long) ifn;
            G__inc_cp_asm(8, 0);
         }
         else {
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(
                    G__serr
                  , "%3x,%3x: LD_FUNC '%s' paran: %d  %s:%d\n"
                  , G__asm_cp
                  , G__asm_dt
                  , funcname
                  , libp->paran
                  , __FILE__
                  , __LINE__
               );
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__LD_FUNC;
            G__asm_inst[G__asm_cp+1] = (long) p_ifunc;
            G__asm_inst[G__asm_cp+2] = ifn;
            G__asm_inst[G__asm_cp+3] = libp->paran;
            if ((p_ifunc->tagnum != -1) && strcmp(funcname, G__struct.name[p_ifunc->tagnum]) == 0) {
               //
#ifndef G__OOLDIMPLEMENTATION2150
               G__bc_Baseclassctor_vbase(p_ifunc->tagnum);
#endif
               G__asm_inst[G__asm_cp+4] = (long) G__bc_exec_ctor_bytecode;
            }
            else {
               G__asm_inst[G__asm_cp+4] = (long) G__bc_exec_normal_bytecode;
            }
            G__asm_inst[G__asm_cp+5] = 0;
            if (p_ifunc && p_ifunc->pentry[ifn]) {
               G__asm_inst[G__asm_cp+5] = p_ifunc->pentry[ifn]->ptradjust;
            }
           
            G__asm_inst[G__asm_cp+6] = (long) p_ifunc;
            G__asm_inst[G__asm_cp+7] = (long) ifn;
            G__inc_cp_asm(8, 0);
         }
      }
      else {
         // --
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(
                 G__serr
               , "%3x,%3x: LD_IFUNC '%s' paran: %d  %s:%d\n"
               , G__asm_cp
               , G__asm_dt
               , funcname
               , libp->paran
               , __FILE__
               , __LINE__
            );
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__LD_IFUNC;
         G__asm_inst[G__asm_cp+1] = (long) p_ifunc->funcname[ifn];
         G__asm_inst[G__asm_cp+2] = hash;
         G__asm_inst[G__asm_cp+3] = libp->paran;
         G__asm_inst[G__asm_cp+4] = (long) p_ifunc;
         G__asm_inst[G__asm_cp+5] = (long) funcmatch;
         G__asm_inst[G__asm_cp+6] = (long) memfunc_flag;
         G__asm_inst[G__asm_cp+7] = (long) ifn;
         G__inc_cp_asm(8, 0);
      }
      if (G__store_struct_offset && (G__store_struct_offset != store_inherit_offset)) {
         // --
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(
                 G__serr
               , "%3x,%3x: ADDSTROS 0x%08lx  %s:%d\n"
               , G__asm_cp
               , G__asm_dt
               , -G__store_struct_offset + store_inherit_offset
               , __FILE__
               , __LINE__
            );
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__ADDSTROS;
         G__asm_inst[G__asm_cp+1] = -G__store_struct_offset + store_inherit_offset;
         G__inc_cp_asm(2, 0);
      }
   }
#endif // G__ASM
   // G__oprovld is set when calling operator overload function after
   // evaluating its argument to avoid duplication in p-code stack data.
   // This must be reset when calling lower level interpreted function.
   G__oprovld = 0;
   //
   //  Return immediately if we are just bytecode compiling.
   //
#ifdef G__ASM
   if (G__no_exec_compile) {
      // -- We are doing whole function compilation right now, do not run function.
      G__store_struct_offset = store_inherit_offset;
      G__tagnum = store_inherit_tagnum;
      result7->tagnum = p_ifunc->p_tagtable[ifn];
      if ((result7->tagnum != -1) && (G__struct.type[result7->tagnum] != 'e')) {
         if (isupper(p_ifunc->type[ifn])) {
            result7->type = 'U';
         }
         else {
            result7->type = 'u';
         }
      }
      else {
         result7->type = p_ifunc->type[ifn];
      }
      result7->typenum = p_ifunc->p_typetable[ifn];
      if (result7->tagnum != -1) {
         result7->ref = 1;
      }
      else {
         result7->ref = 0;
      }
#ifndef G__OLDIMPLEMENTATION1259
      result7->isconst = p_ifunc->isconst[ifn];
#endif
      result7->obj.d = 0.0;
      result7->obj.i = 1;
      if (isupper(result7->type)) {
         result7->obj.reftype.reftype = p_ifunc->reftype[ifn];
      }
      result7->ref = p_ifunc->reftype[ifn];
      if ((p_ifunc->type[ifn] == 'u') && !result7->ref && (result7->tagnum != -1)) {
         G__store_tempobject(*result7); // To free tempobject in pcode.
      }
      // To be implemented.
      G__exec_memberfunc = store_exec_memberfunc;
      return 1;
   }
#endif // G__ASM
   // If virtual function flag is set, get actual tag identity by
   // taginfo member at offset of G__struct.virtual_offset[].
   // Then search for virtual function in actual tag. If found,
   // change p_ifunc, ifn, G__store_struct_offset and G__tagnum.
   // G__store_struct_offset and G__tagnum are already stored above,
   // so no need to store it to temporary here.
   if (p_ifunc->isvirtual[ifn] && !G__fixedscope) {
      if (-1 != G__struct.virtual_offset[G__tagnum])
         virtualtag = *(long*)(G__store_struct_offset /* NEED TO CHECK THIS PART */
                               + G__struct.virtual_offset[G__tagnum]);
      else {
         virtualtag = G__tagnum;
      }
      if (virtualtag != G__tagnum) {
         struct G__inheritance *baseclass = G__struct.baseclass[virtualtag];
         int xbase[G__MAXBASE], ybase[G__MAXBASE];
         int nxbase = 0, nybase;
         int basen;
         G__incsetup_memfunc(virtualtag);
         G__ifunc_table_internal* ifunc = G__ifunc_exist(p_ifunc, ifn, G__struct.memfunc[virtualtag], &iexist
                                , 0xffff);
         for (basen = 0;!ifunc && basen < baseclass->basen;basen++) {
            virtualtag = baseclass->herit[basen]->basetagnum;
            if (0 == (baseclass->herit[basen]->property&G__ISDIRECTINHERIT)) continue;
            xbase[nxbase++] = virtualtag;
            G__incsetup_memfunc(virtualtag);
            ifunc
            = G__ifunc_exist(p_ifunc, ifn, G__struct.memfunc[virtualtag], &iexist
                             , 0xffff);
         }
         while (!ifunc && nxbase) {
            int xxx;
            nybase = 0;
            for (xxx = 0;!ifunc && xxx < nxbase;xxx++) {
               baseclass = G__struct.baseclass[xbase[xxx]];
               for (basen = 0;!ifunc && basen < baseclass->basen;basen++) {
                  virtualtag = baseclass->herit[basen]->basetagnum;
                  if (0 == (baseclass->herit[basen]->property&G__ISDIRECTINHERIT)) continue;
                  ybase[nybase++] = virtualtag;
                  G__incsetup_memfunc(virtualtag);
                  ifunc
                  = G__ifunc_exist(p_ifunc, ifn, G__struct.memfunc[virtualtag]
                                   , &iexist, 0xffff);
               }
            }
            nxbase = nybase;
            memcpy((void*)xbase, (void*)ybase, sizeof(int)*nybase);
         }
         if (ifunc) {
            if (!ifunc->pentry[iexist]->p && !G__get_funcptr(p_ifunc, ifn)) {// 12-09-07 (stub-less calls)
               G__fprinterr(G__serr, "Error: virtual %s() header found but not defined", funcname);
               G__genericerror(0);
               G__exec_memberfunc = store_exec_memberfunc;
               return 1;
            }
            p_ifunc = ifunc;
            ifn = iexist;
            G__store_struct_offset -= G__find_virtualoffset(virtualtag
#ifdef G__VIRTUALBASE
                                                            ,G__store_struct_offset
#endif
                                                            );
            G__tagnum = virtualtag;
            if ('~' == funcname[0]) {
               G__hash(G__struct.name[G__tagnum], hash, itemp);
               hash += '~';
            }
         }
         else if (p_ifunc->ispurevirtual[ifn]) {
            G__fprinterr(G__serr, "Error: pure virtual %s() not defined", funcname);
            G__genericerror(0);
            G__exec_memberfunc = store_exec_memberfunc;
            return 1;
         }
      }
   }
   //
   // *** EXPERIMENTAL ***
   // Try to bytecode compile the whole function
   // using the v6 bytecode compilation machinery.
   //
   if (
      G__cintv6 &&
      (p_ifunc->pentry[ifn]->bytecodestatus == G__BYTECODE_NOTYET)
   ) {
      if (
         (G__bc_compile_function(p_ifunc, ifn) == G__BYTECODE_FAILURE)
      ) {
         G__exec_memberfunc = store_exec_memberfunc;
         return 1;
      }
   }
#ifdef G__ASM
#ifdef G__ASM_WHOLEFUNC
   //
   // Try bytecode compilation.
   //
   if (
      (p_ifunc->pentry[ifn]->bytecodestatus == G__BYTECODE_NOTYET) &&
      (G__asm_loopcompile > 3) &&
      (G__asm_wholefunction == G__ASM_FUNC_NOP) &&
#ifndef G__TO_BE_DELETED
      (memfunc_flag != G__CALLCONSTRUCTOR) &&
      (memfunc_flag != G__TRYCONSTRUCTOR) &&
      (memfunc_flag != G__TRYIMPLICITCONSTRUCTOR) &&
      (memfunc_flag != G__TRYDESTRUCTOR) &&
#endif
      !G__step &&
      (G__asm_noverflow || G__asm_exec || (G__asm_loopcompile > 4))
   ) {
      G__compile_bytecode(G__get_ifunc_ref(p_ifunc), ifn);
   }
   //
   //  If bytecode is available, run the bytecode.
   //
   if (
      p_ifunc->pentry[ifn]->bytecode &&
      (p_ifunc->pentry[ifn]->bytecodestatus != G__BYTECODE_ANALYSIS)
   ) {
      struct G__input_file store_ifile;
      store_ifile = G__ifile;
      G__ifile.filenum = p_ifunc->pentry[ifn]->filenum;
      G__ifile.line_number = p_ifunc->pentry[ifn]->line_number;
      G__exec_bytecode(result7, (char*) p_ifunc->pentry[ifn]->bytecode, libp, hash);
      G__ifile = store_ifile;
      G__tagnum = store_inherit_tagnum;
      G__store_struct_offset = store_inherit_offset;
      return 1;
   }
#endif // G__ASM_WHOLEFUNC
#endif // G__ASM
   //
   //  We are going to interpret the function.
   //
#ifndef G__OLDIMPLEMENTATION1167
   G__reftypeparam(p_ifunc, ifn, libp);
#endif
   //
   //
   //
#ifdef G__ASM
#ifdef G__ASM_IFUNC
   // Push bytecode environment stack.
   // Push loop compilation environment.
   store_asm_inst = G__asm_inst;
   store_asm_stack = G__asm_stack;
   store_asm_name = G__asm_name;
   store_asm_name_p = G__asm_name_p;
   store_asm_param  = G__asm_param;
   store_asm_exec  = G__asm_exec;
   store_asm_noverflow  = G__asm_noverflow;
   store_asm_cp  = G__asm_cp;
   store_asm_dt  = G__asm_dt;
   store_asm_index  = G__asm_index;
   store_asm_instsize = G__asm_instsize;
   G__asm_instsize = 0; // G__asm_inst is not resizable.
   G__asm_inst = asm_inst_g;
   G__asm_stack = asm_stack_g;
   G__asm_name = asm_name;
   G__asm_name_p = 0;
   G__asm_exec = 0;
#endif // G__ASM_IFUNC
#endif // G__ASM
   //
   //
   /////
   //
#ifdef G__ASM
#ifdef G__ASM_IFUNC
#ifdef G__ASM_WHOLEFUNC
   //
   // Bytecode function compilation start.
   //
   if (G__asm_wholefunction & G__ASM_FUNC_COMPILE) {
      // -- We are doing Whole function compilation.
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "!!!bytecode compilation of '%s' started at ", p_ifunc->funcname[ifn]);
         G__printlinenum();
      }
#endif // G__ASM_DBG
      G__asm_name = (char*) malloc(G__ASM_FUNCNAMEBUF);
      // Turn on bytecode generation.
      G__asm_noverflow = 1;
      // Flag that we are doing whole function compilation,
      // no execution should be done, but bytecode should be generated.
      // FIXME: Why do we need this flag, why not G__no_exec and G__asm_noverflow instead?
      // FIXME: Answer: G__no_exec means we are skipping code, so no bytecode can be generated.
      store_no_exec_compile = G__no_exec_compile;
      G__no_exec_compile = 1;
      //
      //  Allocate a local variable chain.
      //
      localvar = (struct G__var_array*) malloc(sizeof(struct G__var_array));
#ifndef G__OLDIMPLEMENTATION2038
      localvar->enclosing_scope = 0;
      localvar->inner_scope = 0;
#endif // G__OLDIMPLEMENTATION2038
      localvar->prev_local = G__p_local;
      localvar->ifunc = G__get_ifunc_ref(p_ifunc);
      localvar->ifn = ifn;
#ifdef G__VAARG
      localvar->libp = libp;
#endif // G__VAARG
      localvar->tagnum = G__tagnum;
      localvar->struct_offset = G__store_struct_offset;
      localvar->exec_memberfunc = G__exec_memberfunc;
      localvar->allvar = 0;
      localvar->varlabel[0][0] = 0;
      localvar->next = 0;
      localvar->prev_filenum = G__ifile.filenum;
      localvar->prev_line_number = G__ifile.line_number;
      for (int ix = 0; ix < G__MEMDEPTH; ++ix) {
         localvar->varnamebuf[ix] = 0;
         localvar->p[ix] = 0;
      }
   }
   else {
      // -- We are not doing whole function compilation.
      G__asm_noverflow = 0;
   }
#else // G__ASM_WHOLEFUNC
   G__asm_noverflow = 0;
#endif // G__ASM_WHOLEFUNC
   G__clear_asm();
#endif // G__ASM_IFUNC
#endif // G__ASM
   //
   //
   //
   /////
   //
   // G__exec_memberfunc and G__memberfunc_tagnum are stored in one
   // upper level G__getfunction() and G__parenthesisovld() and restored
   // when exit from these functions.
   // FIXME: They should be saved and restore here instead.
   if (p_ifunc->tagnum == -1) {
      // -- We are *not* a member function.
      G__exec_memberfunc = 0;
   }
   else {
      // -- We are a member function.
      G__exec_memberfunc = 1;
   }
   G__setclassdebugcond(G__tagnum, 0);
   store_memberfunc_tagnum = G__memberfunc_tagnum;
   store_memberfunc_struct_offset = G__memberfunc_struct_offset;
   G__memberfunc_tagnum = G__tagnum;
   G__memberfunc_struct_offset = G__store_struct_offset;
   //
   //  If the declared type of the return value is of an
   //  interpreted class type, then create a temporary object
   //  to hold it.
   //
   //  Note: We create this before we increment the temp level so that
   //        the temp will stay alive after the function call.
   //
   struct G__tempobject_list* store_p_tempobject = 0;
   if (
      (p_ifunc->type[ifn] == 'u') && // class, enum, struct, union, and
      (G__struct.type[p_ifunc->p_tagtable[ifn]] != 'e') && // not enum, and
      (G__struct.iscpplink[p_ifunc->p_tagtable[ifn]] != G__CPPLINK) && // intrp
      (p_ifunc->reftype[ifn] == G__PARANORMAL) // not reference
   ) {
      // Create temporary object buffer.
#ifdef G__ASM
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(
              G__serr
            , "\n!!!Create temp object (%s) for %s() return value "
              "at temp level %d  %s:%d\n"
            , G__struct.name[p_ifunc->p_tagtable[ifn]]
            , p_ifunc->funcname[ifn]
            , G__templevel
            , __FILE__
            , __LINE__
         );
      }
#endif // G__ASM_DBG
#endif // G__ASM
      G__alloc_tempobject(p_ifunc->p_tagtable[ifn], p_ifunc->p_typetable[ifn]);
      store_p_tempobject = G__p_tempbuf;
   }
   //
   //  Increment busy flag.
   //
   p_ifunc->busy[ifn]++;
   //
   //  Set global variable G__func_now.
   //
   //  This is used in G__malloc() when
   //  allocating function body static variables.
   //
   store_func_now = G__func_now;
   store_func_page = G__func_page;
   G__func_now = ifn;
   G__func_page = p_ifunc->page;
   //
   //  Setup a new local variable chain.
   //
   G_local.ifunc = G__get_ifunc_ref(p_ifunc);
   G_local.ifn = ifn;
#ifdef G__VAARG
   G_local.libp = libp;
#endif
   G_local.tagnum = G__tagnum;
   G_local.struct_offset = G__store_struct_offset;
   G_local.exec_memberfunc = G__exec_memberfunc;
   for (int ix = 0; ix < G__MEMDEPTH; ++ix) {
      G_local.varnamebuf[ix] = 0;
      G_local.p[ix] = 0;
      G_local.is_init_aggregate_array[ix] = 0;
   }
   G_local.prev_filenum = G__ifile.filenum;
   G_local.prev_line_number = G__ifile.line_number;
   G_local.prev_local = G__p_local;
   //
   //  Decide which local variable chain to use.
   //
   G__p_local = &G_local;
#ifdef G__ASM_WHOLEFUNC
   //  Whole body function compilation uses a block
   //  of bytes instead of a variable chain to contain
   //  the values of its local variables.
   if (G__asm_wholefunction & G__ASM_FUNC_COMPILE) {
      G__p_local = localvar;
   }
#endif
   //
   //  Cancel any break exit function.
   //
   break_exit_func = G__break_exit_func;
   G__break_exit_func = 0;
   //
   //  Initialize the local variable chain to empty.
   //
   G__p_local->allvar = 0;
   G__p_local->varlabel[0][0] = 0;
   G__p_local->next = 0;
   //
   //  Initialize the line number and filename and number.
   //
   G__ifile.line_number = p_ifunc->pentry[ifn]->line_number;
   if (p_ifunc->pentry[ifn]->filenum>=0) {
      G__strlcpy(G__ifile.name, G__srcfile[p_ifunc->pentry[ifn]->filenum].filename, G__MAXFILENAME);
   } else {
      G__strlcpy(G__ifile.name, "unknown", G__MAXFILENAME);
   }
   G__ifile.filenum = p_ifunc->pentry[ifn]->filenum;
   //
   //  Stop at a breakpoint if necessary.
   //
   if (
      !G__nobreak &&
      !G__no_exec_compile &&
      G__srcfile[G__ifile.filenum].breakpoint &&
      (G__srcfile[G__ifile.filenum].maxline > G__ifile.line_number) &&
      ((G__srcfile[G__ifile.filenum].breakpoint[G__ifile.line_number] |= G__TRACED) & G__TESTBREAK)
   ) {
      G__BREAKfgetc();
   }
   //
   //  Change to the file-specific security environment.
   //
   store_security = G__security;
   G__security = G__srcfile[G__ifile.filenum].security;
   //
   //  Store file pointer and fpos.
   //
   if (G__ifile.fp) {
      fgetpos(G__ifile.fp, &prev_pos);
   }
   prev_fp = G__ifile.fp;
   //
   //  Find the right file pointer.
   //
   if (G__mfp && (FILE*)p_ifunc->pentry[ifn]->p == G__mfp) {
      // -- In case of macro expanded by cint, we use the tmpfile.
      G__ifile.fp = (FILE*) p_ifunc->pentry[ifn]->p;
   }
   else if (G__srcfile[G__ifile.filenum].fp) {
      // -- The file is already open use that.
      G__ifile.fp = G__srcfile[G__ifile.filenum].fp;
   }
   else {
      // -- The file had been closed, let's reopen the proper file.
      // Resp from the preprocessor and raw.
      if (G__srcfile[G__ifile.filenum].prepname) {
         G__ifile.fp = fopen(G__srcfile[G__ifile.filenum].prepname, "r");
      }
      else {
         G__ifile.fp = fopen(G__srcfile[G__ifile.filenum].filename, "r");
      }
      G__srcfile[G__ifile.filenum].fp = G__ifile.fp;
      if (!G__ifile.fp) {
         G__ifile.fp = (FILE*)p_ifunc->pentry[ifn]->p;
      }
   }
   fsetpos(G__ifile.fp, &p_ifunc->pentry[ifn]->pos);
   //
   //  Print function header if debug mode.
   //
   if (G__dispsource) {
      G__disp_mask = 0;
      if ((G__debug || G__break || G__step
            || (strcmp(G__breakfile, G__ifile.name) == 0) || (strcmp(G__breakfile, "") == 0)
          ) && ((G__prerun != 0) || (G__no_exec == 0))) {
         if (/* G__ifile.name is an array so never null && */ G__ifile.name[0])
            G__fprinterr(G__serr, "\n# %s", G__ifile.name);
         if (-1 != p_ifunc->tagnum) {
            G__fprinterr(G__serr, "\n%-5d%s::%s(" , G__ifile.line_number
                         , G__struct.name[p_ifunc->tagnum] , funcname);
         }
         else {
            G__fprinterr(G__serr, "\n%-5d%s(" , G__ifile.line_number, funcname);
         }
      }
   }
   //
   // Now we have: funcname(para1, ...)
   //                       ^
   ////
   //
   //  Initialize function parameters.
   //
   store_doingconstruction = G__doingconstruction;
   if (!p_ifunc->ansi[ifn]) {
      // -- K&R C.
      ipara = 0;
      while (cin != ')') {
         G__FastAllocString temp(G__ONELINE);
         cin = G__fgetstream(temp, 0, ",)");
         if (temp[0] != '\0') {
            paraname[ipara] = temp;
            ++ipara;
         }
      }
      // read and exec parameter declaration, K&R C
      // G__exec_statement() returns at '{' if G__funcheader==1
      // Flag that we are initializing function parameters.
      G__funcheader = 1;
      do {
         int brace_level = 0;
         buf = G__exec_statement(&brace_level);
      }
      while ((buf.type == G__null.type) && (G__return < G__RETURN_EXIT1));
      // Set function parameters, K&R C.
      // Parameters can be constant. When G__funcheader==1,
      // error message of changing const doesn't appear.
      for (itemp = 0; itemp < ipara; ++itemp) {
         G__letvariable(paraname[itemp], libp->para[itemp], &G__global, G__p_local);
      }
      // Flag that we are done initializing function parameters.
      G__funcheader = 0;
      // Backup one character so that the opening '{'
      // of the function body will be read again.
      fseek(G__ifile.fp, -1, SEEK_CUR);
      if (G__dispsource) {
         G__disp_mask = 1;
      }
   }
   else {
      // -- ANSI C.
      G__value store_ansipara;
      store_ansipara = G__ansipara;
      // Flag that we processing an ANSI-style function header.
      G__ansiheader = 1;
      // Flag that we are initializing function parameters.
      G__funcheader = 1;
      ipara = 0;
      while (G__ansiheader && (G__return < G__RETURN_EXIT1)) {
         // -- Set G__ansipara and G__refansipara to pass argument value and text.
         if (ipara < libp->paran) {
            // -- We have an argument for the parameter.
            G__ansipara = libp->para[ipara];
            // Assigning reference for fundamental type reference argument.
            if (!G__ansipara.ref) {
               switch (p_ifunc->param[ifn][ipara]->type) {
                  case 'f':
                     G__Mfloat(libp->para[ipara]);
                     libp->para[ipara].type = p_ifunc->param[ifn][ipara]->type;
                     G__ansipara.ref = (long)(&libp->para[ipara].obj.fl);
                     break;
                  case 'd':
                     G__ansipara.ref = (long)(&libp->para[ipara].obj.d);
                     break;
                  case 'c':
                     G__Mchar(libp->para[ipara]);
                     libp->para[ipara].type = p_ifunc->param[ifn][ipara]->type;
                     G__ansipara.ref = (long)(&libp->para[ipara].obj.ch);
                     break;
                  case 's':
                     G__Mshort(libp->para[ipara]);
                     libp->para[ipara].type = p_ifunc->param[ifn][ipara]->type;
                     G__ansipara.ref = (long)(&libp->para[ipara].obj.sh);
                     break;
                  case 'i':
                     G__Mint(libp->para[ipara]);
                     libp->para[ipara].type = p_ifunc->param[ifn][ipara]->type;
                     G__ansipara.ref = (long)(&libp->para[ipara].obj.in);
                     break;
                  case 'l':
                     G__ansipara.ref = (long)(&libp->para[ipara].obj.i);
                     break;
                  case 'b':
                  case 'g':
#ifdef G__BOOL4BYTE
                     G__Mint(libp->para[ipara]);
                     G__ansipara.ref = (long)(&libp->para[ipara].obj.i);
#else
                     G__Muchar(libp->para[ipara]);
                     G__ansipara.ref = (long)(&libp->para[ipara].obj.uch);
#endif
                     libp->para[ipara].type = p_ifunc->param[ifn][ipara]->type;
                     break;
                  case 'r':
                     G__Mushort(libp->para[ipara]);
                     libp->para[ipara].type = p_ifunc->param[ifn][ipara]->type;
                     G__ansipara.ref = (long)(&libp->para[ipara].obj.ush);
                     break;
                  case 'h':
                     /* G__Muint(libp->para[ipara]); */
                     G__ansipara.ref = (long)(&libp->para[ipara].obj.i);
                     break;
                  case 'k':
                     G__ansipara.ref = (long)(&libp->para[ipara].obj.i);
                     break;
                  case 'u':
                     G__ansipara.ref = G__ansipara.obj.i;
                     break;
                  case 'n':
                     G__ansipara.ref = (long)(&libp->para[ipara].obj.ll);
                     break;
                  case 'm':
                     G__ansipara.ref = (long)(&libp->para[ipara].obj.ull);
                     break;
                  case 'q':
                     G__ansipara.ref = (long)(&libp->para[ipara].obj.ld);
                     break;
                  default:
                     G__ansipara.ref = (long)(&libp->para[ipara].obj.i);
                     break;
               }
            }
         }
         else {
            // -- We do not have an argument for the paramater.
            if ((ipara < p_ifunc->para_nu[ifn]) && p_ifunc->param[ifn][ipara]->pdefault) {
               // -- We have a default value for the parameter, use it.
               if (p_ifunc->param[ifn][ipara]->pdefault->type == G__DEFAULT_FUNCCALL) {
                  G__ASSERT(p_ifunc->param[ifn][ipara]->pdefault->ref);
                  {
                     // Note: Any temps created during the default expr eval
                     // need to stay alive for the entire function call.
                     // For example if the default value expr is MyString(),
                     // then the created temp and the passed arg will share
                     // storage, and so we must keep it alive for the whole
                     // function call.
                     *p_ifunc->param[ifn][ipara]->pdefault =
                        G__getexpr(
                           (char*) p_ifunc->param[ifn][ipara]->pdefault->ref
                        );
                  }
                  G__ansiheader = 1;
                  G__funcheader = 1;
               }
               G__ansipara = *p_ifunc->param[ifn][ipara]->pdefault;
            }
            else {
               // -- Flag that no argument was provided.
               G__ansipara = G__null;
            }
         }
         G__refansipara = libp->parameter[ipara];
         if (
            (G__asm_wholefunction == G__ASM_FUNC_COMPILE) &&
            p_ifunc->param[ifn][ipara]->pdefault
         ) {
            // -- Generate bytecode for stacking default parameter value.
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x: ISDEFAULTPARA %x\n", G__asm_cp, G__asm_cp + 4);
               G__fprinterr(G__serr, "%3x: LD %ld %g\n", G__asm_cp + 2, p_ifunc->param[ifn][ipara]->pdefault->obj.i, p_ifunc->param[ifn][ipara]->pdefault->obj.d);
            }
#endif
            G__asm_inst[G__asm_cp] = G__ISDEFAULTPARA;
            G__asm_wholefunc_default_cp = G__asm_cp + 1;
            G__inc_cp_asm(2, 0);
            // Set default param in stack.
            G__asm_inst[G__asm_cp] = G__LD;
            G__asm_inst[G__asm_cp+1] = G__asm_dt;
            G__asm_stack[G__asm_dt] = *p_ifunc->param[ifn][ipara]->pdefault;
            G__inc_cp_asm(2, 1);
            G__asm_inst[G__asm_wholefunc_default_cp] = G__asm_cp;
            G__asm_noverflow = 0;
            int brace_level = 0;
            G__exec_statement(&brace_level); // Create var entry and ST_LVAR inst.
            //{
            //   char buf[128];
            //   G__fgetstream_peek(buf, 30);
            //   fprintf(stderr, "G__interpret_func: finished default param, peek ahead: '%s'\n", buf);
            //}
            G__asm_wholefunc_default_cp = 0;
            G__asm_noverflow = 1;
         }
         else {
            // -- Initialize function parameter.
            int brace_level = 0;
            G__exec_statement(&brace_level);
            // Note: We do not destroy temps here, all func arg temps must
            // stay alive until the function completes execution.
         }
         ipara++;
      }
      // Flag that we are done initializing function parameters.
      G__funcheader = 0;
      // Process any base and member initialization part.
      switch (memfunc_flag) {
         case G__CALLCONSTRUCTOR:
         case G__TRYCONSTRUCTOR:
#ifndef G__OLDIMPLEMENTATIO1250
         case G__TRYIMPLICITCONSTRUCTOR:
#endif
            G__baseconstructorwp();
            G__doingconstruction = 1;
      }
      G__ansipara = store_ansipara;
   }
#ifdef G__SECURITY
   if (
      (G__security & G__SECURE_STACK_DEPTH) &&
      G__max_stack_depth &&
      (G__calldepth > G__max_stack_depth)
   ) {
      G__fprinterr(G__serr, "Error: Stack depth exceeded %d", G__max_stack_depth);
      G__genericerror(0);
      G__pause();
      G__return = G__RETURN_EXIT1;
   }
   if (G__return > G__RETURN_EXIT1) {
      G__exec_memberfunc = store_exec_memberfunc;
      G__security = store_security;
      G__memberfunc_tagnum = store_memberfunc_tagnum;
      G__memberfunc_struct_offset = store_memberfunc_struct_offset;
      return 1;
   }
#endif
   G__setclassdebugcond(G__memberfunc_tagnum, 1);
   //
   // Now get ready to execute the function body.
   //
   store_iscpp = G__iscpp;
   G__iscpp = p_ifunc->iscpp[ifn];
#ifdef G__ASM
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(
           G__serr
         , "\n!!!G__interpret_func: Increment G__templevel %d --> %d  %s:%d\n"
         , G__templevel
         , G__templevel + 1
         , __FILE__
         , __LINE__
      );
   }
#endif // G__ASM_DBG
#endif // G__ASM
   ++G__templevel;
   ++G__calldepth;
#ifdef G__ASM_DBG
   if (G__istrace > 1) {
      if (G__istrace > G__calldepth) {
         G__debug = 1;
         G__asm_dbg = 1;
      }
      else {
         G__debug = 0;
         G__asm_dbg = 0;
      }
   }
#endif
   G__ASSERT(!G__decl || (G__decl == 1));
   store_def_struct_member = G__def_struct_member;
   G__def_struct_member = 0;
   store_decl = G__decl;
   G__decl = 0;
   G__no_exec = 0;
   //
   // Execute body.
   //
   int brace_level = 0;
   *result7 = G__exec_statement(&brace_level);
   //
   //  After execution.
   //
   G__decl = store_decl;
   G__def_struct_member = store_def_struct_member;
   if (
      (G__return == G__RETURN_IMMEDIATE) &&
      G__interactivereturnvalue.type &&
      (result7->type == '\0')
   ) {
      *result7 = G__interactivereturnvalue;
      G__interactivereturnvalue = G__null;
   }
   --G__calldepth;
#ifdef G__ASM_DBG
   if (G__istrace > 1) {
      if (G__istrace > G__calldepth) {
         G__debug = 1;
         G__asm_dbg = 1;
      }
      else {
         G__debug = 0;
         G__asm_dbg = 0;
      }
   }
#endif // G__ASM_DBG
   G__iscpp = (short) store_iscpp;
   G__doingconstruction = store_doingconstruction;
   //
   //  Error if goto label not found.
   //
   if (G__gotolabel[0]) {
      G__fprinterr(
           G__serr
         , "Error: Goto label '%s' not found in %s()"
         , G__gotolabel
         , funcname
      );
      G__genericerror(0);
      G__gotolabel[0] = '\0';
   }
   //
   //  Do return value type conversion.
   //
   if (
      !G__xrefflag &&
      (G__return != G__RETURN_EXIT1) &&
      (result7->type != '\0') &&
      ((G__asm_wholefunction == G__ASM_FUNC_NOP) || G__asm_noverflow)
   ) {
      switch (p_ifunc->type[ifn]) {
         case 'd': // double
         case 'f': // float
         case 'w': // logic
            G__letdouble(result7, p_ifunc->type[ifn], G__double(*result7));
            if (p_ifunc->reftype[ifn] == G__PARANORMAL) {
               result7->ref = 0;
            }
#ifndef G__OLDIMPLEMENTATION1259
            result7->isconst = p_ifunc->isconst[ifn];
#endif // G__OLDIMPLEMENTATION1259
            break;
         case 'n': // unsigned long long
         case 'm': // long long
            G__letLonglong(result7, p_ifunc->type[ifn], G__Longlong(*result7));
            if (p_ifunc->reftype[ifn] == G__PARANORMAL) {
               result7->ref = 0;
            }
            result7->isconst = p_ifunc->isconst[ifn];
            break;
         case 'q': // long double
            G__letLongdouble(result7, p_ifunc->type[ifn],
               G__Longdouble(*result7));
            if (p_ifunc->reftype[ifn] == G__PARANORMAL) {
               result7->ref = 0;
            }
            result7->isconst = p_ifunc->isconst[ifn];
            break;
         case 'g': // bool
            G__letint(result7, p_ifunc->type[ifn], G__int(*result7) ? 1 : 0);
            if (p_ifunc->reftype[ifn] == G__PARANORMAL) {
               result7->ref = 0;
            }
            result7->isconst = p_ifunc->isconst[ifn];
            break;
         case 'y': // void
            //
            // in case of void, if return(); statement exists
            // it is illegal.
            // Maybe bug if return; statement exists without
            // return value.
            //
            if (G__return == G__RETURN_NORMAL) {
               if (G__dispmsg >= G__DISPWARN) {
                  G__fprinterr(
                       G__serr
                     , "Warning: Return value of void %s() ignored"
                     , p_ifunc->funcname[ifn]
                  );
                  G__printlinenum();
               }
            }
            *result7 = G__null;
            break;
         case 'u': // class, enum, struct, union by value
            //
            //  Note: result7 contains pointer to the local variable
            //  which will be destroyed right after this.
            //
            //--
            //
            //  Do not call a copy constructor if the declared
            //  return type is a reference.
            //
            if (p_ifunc->reftype[ifn] != G__PARANORMAL) {
               // The declared return type is a reference.
               if (p_ifunc->p_tagtable[ifn] != result7->tagnum) {
                  // The type of the returned value does not match the
                  // declared return type.  Check if the return type is
                  // a public base of the type of the returned value.
                  int offset =
                     G__ispublicbase(
                          p_ifunc->p_tagtable[ifn]
                        , result7->tagnum
                        , result7->obj.i
                     );
                  if (offset == -1) {
                     // The func return type is not a public base
                     // of the type of the returned value.
                     G__fprinterr(
                          G__serr
                        , "Error: Return type mismatch. %s "
                          "not a public base of %s"
                        , G__fulltagname(p_ifunc->p_tagtable[ifn], 1)
                        , G__fulltagname(result7->tagnum, 1)
                     );
                     G__genericerror(0);
                     result7->tagnum = p_ifunc->p_tagtable[ifn];
                     break;
                  }
                  else {
                     // The func return type is a public base of the
                     // type of the returned value.  Change the type
                     // of the returned value and adjust the this
                     // pointer, completing the cast-to-base operation.
                     result7->tagnum = p_ifunc->p_tagtable[ifn];
                     result7->obj.i += offset;
                     if (result7->ref) {
                        result7->ref += offset;
                     }
                  }
               }
               break;
            }
            //
            //  At this point we know the declared return type
            //  is not a pointer or reference.
            //
            //  Now check for enumerator return type, we do not
            //  need to do anything in that case.
            //
            if (G__struct.type[p_ifunc->p_tagtable[ifn]] == 'e') {
               // Enumerator return value, nothing to do here.
               break;
            }
            //
            //  At this point we know that the declared returned
            //  type is actually of class, struct, or union type.
            //
            {
               //
               //  Create a one-argument constructor call to convert
               //  the returned value to the declared return type.
               //
               G__FastAllocString constructor_call(G__ONELINE);
               if (
                  (result7->type == 'u') || // class type, or
                  (
                     (result7->type == 'i') && // int type, and
                     (result7->tagnum != -1) // has a tag (must be enum)
                  )
               ) {
                  constructor_call.Format(
                       "%s((%s)0x%lx)"
                     , G__struct.name[p_ifunc->p_tagtable[ifn]]
                     , G__fulltagname(result7->tagnum, 1)
                     , result7->obj.i
                  );
               }
               else {
                  G__FastAllocString buf2(G__ONELINE);
                  G__valuemonitor(*result7, buf2);
                  constructor_call.Format(
                       "%s(%s)"
                     , G__struct.name[p_ifunc->p_tagtable[ifn]]
                     , buf2()
                  );
               }
               //
               //  Setup for the constructor call.
               //
               store_struct_offset = G__store_struct_offset;
               store_tagnum = G__tagnum;
               G__tagnum = p_ifunc->p_tagtable[ifn];
               store_var_type = G__var_type;
#ifdef G__SECURITY
               G__castcheckoff = 1;
#endif // G__SECURITY
               if (G__return <= G__RETURN_IMMEDIATE) {
                  G__return = G__RETURN_NON;
               }
               itemp = 0;
               if (
                  G__struct.iscpplink[p_ifunc->p_tagtable[ifn]] != G__CPPLINK
               ) {
                  // The declared type of the func return is
                  // an interpreted class.
                  if (store_p_tempobject) {
                     // Declared return type is interpreted class.
                     G__store_struct_offset = store_p_tempobject->obj.obj.i;
                  }
                  else {
                     // TODO: Can this case ever happen?
                     // Declared return type is compiled class.
                     abort();
                     G__store_struct_offset = G__p_tempbuf->obj.obj.i;
                  }
                  if (G__dispsource) {
                     G__fprinterr(
                          G__serr
                        , "\n!!!Calling constructor to convert "
                          "returned value to declared return type "
                          "(%s) 0x%lx  %s:%d\n"
                        , constructor_call()
                        , G__store_struct_offset
                        , __FILE__
                        , __LINE__
                     );
                  }
                  G__getfunction(constructor_call(), &itemp, G__TRYCONSTRUCTOR);
               }
               else {
                  // The declared type of the func return is a compiled class.
                  long store_globalvarpointer = G__globalvarpointer;
                  G__globalvarpointer = G__PVOID;
                  G__store_struct_offset = 0xffff;
                  if (G__dispsource) {
                     G__fprinterr(
                          G__serr
                        , "\n!!!Calling constructor to convert "
                          "returned value to declared return type "
                          " (%s) 0x%lx  %s:%d\n"
                        , constructor_call()
                        , G__store_struct_offset
                        , __FILE__
                        , __LINE__
                     );
                  }
                  // It is not needed to explicitly create a STORETEMP
                  // instruction because it is preincluded in the compiled
                  // function call interface.
                  buf = G__getfunction(constructor_call(), &itemp,
                     G__TRYCONSTRUCTOR);
                  G__globalvarpointer = store_globalvarpointer;
                  if (itemp) {
                     //G__free_tempobject();
                     // We need to keep this temporary around.
                     --G__templevel;
                     G__store_tempobject(buf);
                     ++G__templevel;
                  }
#ifdef G__ASM
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(
                          G__serr
                        , "!!!Created temp object (%s) 0x%lx at "
                          "temp level %d to hold "
                          "%s() returned value  %s:%d\n"
                        , G__struct.name[G__tagnum]
                        , G__p_tempbuf->obj.obj.i
                        , G__templevel
                        , p_ifunc->funcname[ifn]
                        , __FILE__
                        , __LINE__
                     );
                  }
#endif // G__ASM_DBG
#endif // G__ASM
                  // --
               }
               //
               //  Cleanup after the constructor call.
               //
               G__store_struct_offset = store_struct_offset;
               G__tagnum = store_tagnum;
               G__var_type = store_var_type;
               //
               //  If no available constructor, do a memberwise copy if
               //  the types match, otherwise try a base class constructor.
               //
               if (!itemp && !G__xrefflag) {
                  long offset = 0;
                  if (result7->tagnum == p_ifunc->p_tagtable[ifn]) {
                     // Types match, do a memberwise copy.
                     if (store_p_tempobject) {
                        memcpy(
                             (void*) store_p_tempobject->obj.obj.i
                           , (void*) result7->obj.i
                           , (size_t) G__struct.size[result7->tagnum]
                        );
                     }
                     else {
                        memcpy(
                             (void*) G__p_tempbuf->obj.obj.i
                           , (void*) result7->obj.i
                           , (size_t) G__struct.size[result7->tagnum]
                        );
                     }
                  }
                  else {
                     // Types do not match, try a base class constructor.
                     offset = G__ispublicbase(p_ifunc->p_tagtable[ifn],
                        result7->tagnum, result7->obj.i);
                     if (offset != -1) {
                        // The declared return type is a base class of
                        // the type of the returned value, try a base
                        // class copy constructor using the offset this
                        // pointer of the returned value.
                        G__FastAllocString base_constructor_call(G__ONELINE);
                        base_constructor_call.Format(
                             "%s((%s)0x%lx)"
                           , G__struct.name[p_ifunc->p_tagtable[ifn]]
                           , G__fulltagname(p_ifunc->p_tagtable[ifn], 1)
                           , result7->obj.i + offset
                        );
                        if (G__struct.iscpplink[G__tagnum] != G__CPPLINK) {
                           // Declared return type is an interpreted class.
                           if (store_p_tempobject) {
                              G__store_struct_offset =
                                 store_p_tempobject->obj.obj.i;
                           }
                           else {
                              G__store_struct_offset = G__p_tempbuf->obj.obj.i;
                           }
                           G__getfunction(base_constructor_call(), &itemp,
                              G__TRYCONSTRUCTOR);
                        }
                        else {
                           // Declared return type is a compiled class.
                           G__store_struct_offset = 0xffff;
                           buf = G__getfunction(base_constructor_call(), &itemp,
                              G__TRYCONSTRUCTOR);
                           if (itemp) {
                              G__store_tempobject(buf);
                           }
                        }
                     }
                  }
               }
               //
               //  Make the result a shallow copy of the temporary
               //  created to hold the converted returned value.
               //
               //  Note: We must keep the temporary alive because
               //        it and the result share the this pointer!
               //
               if (store_p_tempobject) {
                  *result7 = store_p_tempobject->obj;
               }
               else {
                  *result7 = G__p_tempbuf->obj;
               }
            }
#ifndef G__OLDIMPLEMENTATION1259
            result7->isconst = p_ifunc->isconst[ifn];
#endif // G__OLDIMPLEMENTATION1259
            break;
         case 'i': // int
            // return value of constructor
            if (p_ifunc->p_tagtable[ifn] != -1) {
               if (
                  (
                     G__struct.iscpplink[p_ifunc->p_tagtable[ifn]] !=
                        G__CPPLINK
                  ) &&
                  (G__struct.type[p_ifunc->p_tagtable[ifn]] != 'e') &&
                  (G__store_struct_offset != 0) &&
                  (G__store_struct_offset != 1)
               ) {
                  result7->obj.i = G__store_struct_offset;
               }
               result7->ref = result7->obj.i;
#ifndef G__OLDIMPLEMENTATION1259
               result7->isconst = 0;
#endif // G__OLDIMPLEMENTATION1259
               break;
            }
            // INTENTIONAL FALLTHROUGH
         case 'U': // pointer to class, struct, union
            if ((p_ifunc->type[ifn] == 'U') && (result7->type == 'U')) {
               if (p_ifunc->p_tagtable[ifn] != result7->tagnum) {
                  int offset = G__ispublicbase(p_ifunc->p_tagtable[ifn],
                     result7->tagnum, result7->obj.i);
                  if (offset == -1) {
                     G__fprinterr(
                          G__serr
                        , "Error: Return type mismatch. %s "
                        , G__fulltagname(p_ifunc->p_tagtable[ifn], 1)
                     );
                     G__fprinterr(
                          G__serr
                        , "not a public base of %s"
                        , G__fulltagname(result7->tagnum, 1)
                     );
                     G__genericerror(0);
                     result7->tagnum = p_ifunc->p_tagtable[ifn];
                     break;
                  }
                  else {
                     result7->obj.i += offset;
                     if (result7->ref) {
                        result7->ref += offset;
                     }
                     result7->tagnum = p_ifunc->p_tagtable[ifn];
                  }
               }
            }
            // INTENTIONAL FALLTHROUGH
         default:
            //
            //  Everything else is returned as integer. This
            //  includes char,short,int,long,unsigned version
            //  of them, pointer and struct/union.
            //  If return value is struct/union, malloced memory
            //  area will be freed about 20 lines below by
            //  a destroy. To prevent any data loss, memory
            //  area has to be copied to left hand side memory
            //  area of assignment (or temp buffer of expression
            //  parser which doesn't exist in this version).
            //
#ifdef G__SECURITY
            if (
               isupper(p_ifunc->type[ifn]) &&
               islower(result7->type) &&
               result7->obj.i &&
               (G__asm_wholefunction == 0)
            ) {
               G__fprinterr(
                    G__serr
                  , "Error: Return type mismatch %s()"
                  , p_ifunc->funcname[ifn]
               );
               G__genericerror(0);
               break;
            }
#endif // G__SECURITY
            G__letint(result7, p_ifunc->type[ifn], G__int(*result7));
            if (p_ifunc->reftype[ifn] == G__PARANORMAL) {
               result7->ref = 0;
            }
            if (isupper(result7->type)) {
               result7->obj.reftype.reftype = p_ifunc->reftype[ifn];
            }
#ifdef G__SECURITY
            if (
               isupper(result7->type) &&
               (G__security & G__SECURE_GARBAGECOLLECTION) &&
               !G__no_exec_compile
            ) {
               // Add reference count to avoid garbage collection
               // when pointer is returned.
               G__add_refcount((void*) result7->obj.i, 0);
            }
#endif // G__SECURITY
#ifndef G__OLDIMPLEMENTATION1259
            result7->isconst = p_ifunc->isconst[ifn];
#endif // G__OLDIMPLEMENTATION1259
            break;
      }
   }
   if (G__return != G__RETURN_EXIT1) {
      // -- If not exit, return struct and typedef identity.
      result7->tagnum  = p_ifunc->p_tagtable[ifn];
      result7->typenum = p_ifunc->p_typetable[ifn];
   }
   if (G__return != G__RETURN_TRY) {
      G__no_exec = 0;
      // G__return is set to 1 if interpreted function returns by
      // return statement.  Until G__return is reset to 0,
      // execution flow exits from G__exec_statment().
      if (G__return <= G__RETURN_IMMEDIATE) {
         G__return = G__RETURN_NON;
      }
   }
#ifndef G__NEWINHERIT
   //
   //  Recover previous local variable chain.
   //
   G__p_local = G_local.prev_local; // same as G__p_local->prev_local
#endif // G__NEWINHERIT
#ifdef G__ASM_WHOLEFUNC
   if (G__asm_wholefunction & G__ASM_FUNC_COMPILE) {
      // -- End whole function bytecode compile.
      if (G__security_error != G__NOERROR) {
         G__resetbytecode();
      }
      if (G__asm_noverflow) {
         int pc = 0;
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(
                 G__serr
               , "%3x,%3x: RTN_FUNC  %s:%d\n"
               , G__asm_cp
               , G__asm_dt
               , __FILE__
               , __LINE__
            );
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__RTN_FUNC;
         G__asm_inst[G__asm_cp+1] = 0;
         G__inc_cp_asm(2, 0);
         G__asm_inst[G__asm_cp] = G__RETURN;
         G__resolve_jumptable_bytecode();
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(
                 G__serr
               , "Bytecode compilation of %s successful"
               , p_ifunc->funcname[ifn]
            );
            G__printlinenum();
         }
#endif // G__ASM_DBG
         if (G__asm_loopcompile >= 2) {
            G__asm_optimize(&pc);
         }
         G__resetbytecode();
         G__no_exec_compile = store_no_exec_compile;
         G__asm_storebytecodefunc(p_ifunc, ifn, localvar, G__asm_stack,
            G__asm_dt, G__asm_inst, G__asm_cp);
      }
      else {
         free(G__asm_name);
         G__resetbytecode();
         G__no_exec_compile = store_no_exec_compile;
         // Destroy local memory area.
         G__destroy_upto(localvar, G__BYTECODELOCAL_VAR, 0, -1);
         free((void*)localvar);
         if (G__asm_dbg) {
            if (G__dispmsg >= G__DISPWARN) {
               G__fprinterr(
                    G__serr
                  , "\nWarning: Bytecode compilation of %s failed. "
                    "Function execution may be slow."
                  , p_ifunc->funcname[ifn]
               );
               G__printlinenum();
            }
         }
         if (G__return >= G__RETURN_IMMEDIATE) {
            G__return = G__RETURN_NON;
         }
         G__security_error = G__NOERROR;
      }
   }
   else {
      // Destroy allocated local variable chain.
      int store_security_error = G__security_error;
      G__security_error = 0;
      G__destroy_upto(&G_local, G__LOCAL_VAR, 0, -1);
      G__security_error = store_security_error;
   }
#else // G__ASM_WHOLEFUNC
   //
   //  Destroy allocated local variable chain.
   //
   G__destroy_upto(&G_local, G__LOCAL_VAR, 0, -1);
#endif // G__ASM_WHOLEFUNC
   //
   //  If we have just finished a class destructor call,
   //  then call the destructors of the base classes and
   //  of the data members of class type.
   //
   if (memfunc_flag == G__TRYDESTRUCTOR) {
      G__basedestructor();
   }
#ifdef G__NEWINHERIT
   //
   //  Destroy all temporaries created during function call.
   //
#ifdef G__ASM
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(
           G__serr
         , "\n!!!G__interpret_func: Destroy temp objects now at G__templevel %d  %s:%d\n"
         , G__templevel
         , __FILE__
         , __LINE__
      );
   }
#endif // G__ASM_DBG
#endif // G__ASM
   G__free_tempobject();
#ifdef G__ASM
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(
           G__serr
         , "\n!!!G__interpret_func: Decrement G__templevel %d --> %d  %s:%d\n"
         , G__templevel
         , G__templevel - 1
         , __FILE__
         , __LINE__
      );
   }
#endif // G__ASM_DBG
#endif // G__ASM
   --G__templevel;
   //
   //  Recover previous local variable chain.
   //
   G__p_local = G_local.prev_local; // same as G__p_local->prev_local
#endif // G__NEWINHERIT
   G__tagnum = store_inherit_tagnum;
   G__store_struct_offset = store_inherit_offset;
   //
   //  Recover previous line number, filenum, and filename.
   //
   G__ifile.line_number = G_local.prev_line_number;
   G__ifile.filenum = G_local.prev_filenum;
   if (
      (G__ifile.filenum != -1) &&
      G__srcfile[G__ifile.filenum].filename
   ) {
      G__strlcpy(G__ifile.name,
         G__srcfile[G__ifile.filenum].filename, G__MAXFILENAME);
   }
   else {
      G__ifile.name[0] = '\0';
   }
   if (G__dispsource && G__ifile.name[0]) {
      G__fprinterr(G__serr, "\n# %s   ", G__ifile.name);
   }
   // Recover file pointer and fpos.
   G__ifile.fp = prev_fp;
   if (G__ifile.fp) {
      fsetpos(G__ifile.fp, &prev_pos);
   }
   if (G__dispsource) {
      if (
         (G__disp_mask == 0) &&
         (G__debug || G__break) &&
         (G__prerun || !G__no_exec)
      ) {
         G__fprinterr(G__serr, "\n");
      }
   }
   if (G__break_exit_func) {
      G__break = 1;
      G__break_exit_func = 0;
      G__setdebugcond();
   }
   G__break_exit_func = break_exit_func;
   // Decrement busy flag.
   --p_ifunc->busy[ifn];
   if (store_var_typeB == 'P') {
      G__val2pointer(result7);
   }
   G__func_page = store_func_page;
   G__func_now = store_func_now;
#ifdef G__ASM_IFUNC
   // Pop loop compilation environment.
   G__asm_inst = store_asm_inst;
   G__asm_instsize = store_asm_instsize;
   G__asm_stack = store_asm_stack;
   G__asm_name = store_asm_name;
   G__asm_name_p = store_asm_name_p;
   G__asm_param  = store_asm_param;
   G__asm_exec  = store_asm_exec;
   G__asm_noverflow  = store_asm_noverflow;
   G__asm_cp  = store_asm_cp;
   G__asm_dt  = store_asm_dt;
   G__asm_index  = store_asm_index;
#endif // G__ASM_IFUNC
   G__exec_memberfunc = store_exec_memberfunc;
   G__security = store_security;
   G__memberfunc_tagnum = store_memberfunc_tagnum;
   G__memberfunc_struct_offset = store_memberfunc_struct_offset;
   return 1;
}

//______________________________________________________________________________
struct G__ifunc_table_internal* G__ifunc_exist(G__ifunc_table_internal* ifunc_now, int allifunc, G__ifunc_table_internal* ifunc, int* piexist, int mask)
{
   // -- FIXME: Describe this function!
   int i, j, paran;
   int ref_diff;
   while (ifunc)
   {
      for (i = 0;i < ifunc->allifunc;i++) {
         if ('~' == ifunc_now->funcname[allifunc][0] &&
               '~' == ifunc->funcname[i][0]) { /* destructor matches with ~ */
            *piexist = i;
            return(ifunc);
         }
         if (ifunc_now->hash[allifunc] != ifunc->hash[i] ||
               strcmp(ifunc_now->funcname[allifunc], ifunc->funcname[i]) != 0 ||
               (ifunc_now->para_nu[allifunc] != ifunc->para_nu[i] &&
                ifunc_now->para_nu[allifunc] >= 0 && ifunc->para_nu[i] >= 0)
#ifndef G__OLDIMPLEMENTATION1258
               || ((ifunc_now->isconst[allifunc]&mask) /* 1798 */
                   != (ifunc->isconst[i]&mask))
#endif
            ) continue; /* unmatch */


         if (ifunc_now->para_nu[allifunc] >= 0 && ifunc->para_nu[i] >= 0)
            paran = ifunc_now->para_nu[allifunc];
         else
            paran = 0;
         ref_diff = 0;
         for (j = 0;j < paran;j++) {
            if (ifunc_now->param[allifunc][j]->type != ifunc->param[i][j]->type ||
                  ifunc_now->param[allifunc][j]->p_tagtable != ifunc->param[i][j]->p_tagtable
                  || (ifunc_now->param[allifunc][j]->reftype != ifunc->param[i][j]->reftype
                      && G__PARAREFERENCE !=
                      ifunc_now->param[allifunc][j]->reftype + ifunc->param[i][j]->reftype
                     )
                  || ifunc_now->param[allifunc][j]->isconst != ifunc->param[i][j]->isconst
               ) {
               break; /* unmatch */
            }
            if (ifunc_now->param[allifunc][j]->reftype != ifunc->param[i][j]->reftype)
               ++ref_diff;
         }
         if (j == paran) { /* all matched */
            if (ref_diff) {
               G__fprinterr(G__serr, "Warning: %s(), parameter only differs in reference type or not"
                            , ifunc->funcname[i]);
               G__printlinenum();
            }
            *piexist = i;
            return ifunc;
         }
      }
      ifunc = ifunc->next;
   }
   // Not found.
   return ifunc;
}

//______________________________________________________________________________
struct G__ifunc_table_internal* G__ifunc_ambiguous(G__ifunc_table_internal* ifunc_now, int allifunc, G__ifunc_table_internal* ifunc, int* piexist, int derivedtagnum)
{
   // -- FIXME: Describe this function!
   int i, j, paran;
   while (ifunc)
   {
      for (i = 0;i < ifunc->allifunc;i++) {
         if ('~' == ifunc_now->funcname[allifunc][0] &&
               '~' == ifunc->funcname[i][0]) { /* destructor matches with ~ */
            *piexist = i;
            return(ifunc);
         }
         if (ifunc_now->hash[allifunc] != ifunc->hash[i] ||
               strcmp(ifunc_now->funcname[allifunc], ifunc->funcname[i]) != 0
            ) continue; /* unmatch */
         if (ifunc_now->para_nu[allifunc] < ifunc->para_nu[i])
            paran = ifunc_now->para_nu[allifunc];
         else
            paran = ifunc->para_nu[i];
         if (paran < 0) paran = 0;
         for (j = 0;j < paran;j++) {
            if (ifunc_now->param[allifunc][j]->type != ifunc->param[i][j]->type)
               break; /* unmatch */
            if (ifunc_now->param[allifunc][j]->p_tagtable
                  == ifunc->param[i][j]->p_tagtable) continue; /* match */
#ifdef G__VIRTUALBASE
            if (-1 == G__ispublicbase(ifunc_now->param[allifunc][j]->p_tagtable
                                      , derivedtagnum, G__STATICRESOLUTION2) ||
                  -1 == G__ispublicbase(ifunc->param[i][j]->p_tagtable, derivedtagnum
                                        , G__STATICRESOLUTION2))
#else
            if (-1 == G__ispublicbase(ifunc_now->param[allifunc][j]->p_tagtable
                                      , derivedtagnum) ||
                  -1 == G__ispublicbase(ifunc->param[i][j]->p_tagtable, derivedtagnum))
#endif
               break; /* unmatch */
            /* else match */
         }
         if ((ifunc_now->para_nu[allifunc] < ifunc->para_nu[i] &&
               ifunc->param[i][paran]->pdefault) ||
               (ifunc_now->para_nu[allifunc] > ifunc->para_nu[i] &&
                ifunc_now->param[allifunc][paran]->pdefault)) {
            *piexist = i;
            return(ifunc);
         }
         else if (j == paran) { /* all matched */
            *piexist = i;
            return(ifunc);
         }
      }
      ifunc = ifunc->next;
   }
   return(ifunc); /* not found case */
}

//______________________________________________________________________________
// 05-02-08
// We add extraparameters to avoid code replication. We need no pass the noerror
// argument to G__argtype2param and isconst to G__get_ifunchandle_base
//
// Note: the constness was not checked before and I think it's important
// to check it, afterall const overloading is allowed. The problem is that
// Cint was ignoring it before so we need it an option to do exactly that.. 
// ignore it.
// isconst can be: 
// 0 (do everything as before, i.e. ignore the constness)
// 1 (look for a non-const function)
// 2 (look for a const function)
// Note: This should be put into something like an enum
//
struct G__ifunc_table_internal* G__get_ifunchandle(const char* funcname, G__param* libp, int hash, G__ifunc_table_internal* p_ifunc, long* pifn, int access, int funcmatch, int isconst)
{
   // -- FIXME: Describe this function!
   int ifn = 0;
   int ipara = 0;
   int itemp = 0;

   if (-1 != p_ifunc->tagnum) G__incsetup_memfunc(p_ifunc->tagnum);

   /*******************************************************
    * while interpreted function list exists
    *******************************************************/
   while (p_ifunc)
   {
      while ((ipara == 0) && (ifn < p_ifunc->allifunc)) {
         /* if hash (sum of funcname char) matchs */
         if (hash == p_ifunc->hash[ifn] && strcmp(funcname, p_ifunc->funcname[ifn]) == 0
               && (p_ifunc->access[ifn]&access)) {
            /**************************************************
             * for overloading of function and operator
             **************************************************/
            /**************************************************
             * check if parameter type matchs
             **************************************************/
            /* set(reset) match flag ipara temporarily */
            itemp = 0;
            ipara = 1;

            // 31-07-07
            // constness was not checked before
            if(isconst) { // This means we want to check for constness
               isconst -= 1; // a bit hacky... be aware
               if (isconst) isconst = G__CONSTFUNC;
               if( (p_ifunc->isconst[ifn] & G__CONSTFUNC) !=  isconst) {
                 ++ifn;
                 continue;
               }
            }

            if (p_ifunc->ansi[ifn] == 0) break; /* K&R C style header */
            /* main() no overloading */
            if (G__HASH_MAIN == hash && strcmp(funcname, "main") == 0) break;

            /* if more actual parameter than formal parameter, unmatch */
            if (p_ifunc->para_nu[ifn] < libp->paran) {
               ipara = 0;
               itemp = p_ifunc->para_nu[ifn]; /* end of this parameter */
               ++ifn; /* next function */
            }
            else {
               /* scan each parameter */
               while (itemp < p_ifunc->para_nu[ifn]) {
                  if ((G__value*)NULL == p_ifunc->param[ifn][itemp]->pdefault &&
                        itemp >= libp->paran
                     ) {
                     ipara = 0;
                  }
                  else if (p_ifunc->param[ifn][itemp]->pdefault && itemp >= libp->paran) {
                     ipara = 2; /* I'm not sure what this is, Fons. */
                  }
                  else {
                     ipara = G__param_match(p_ifunc->param[ifn][itemp]->type
                                            , p_ifunc->param[ifn][itemp]->p_tagtable
                                            , p_ifunc->param[ifn][itemp]->pdefault
                                            , libp->para[itemp].type
                                            , libp->para[itemp].tagnum
                                            , &(libp->para[itemp])
                                            , libp->parameter[itemp]
                                            , funcmatch
                                            , p_ifunc->para_nu[ifn] - itemp - 1
                                            , p_ifunc->param[ifn][itemp]->reftype
                                            , p_ifunc->param[ifn][itemp]->isconst
                                            /* ,p_ifunc->isexplicit[ifn] */
                                           );
                  }
                  switch (ipara) {
                     case 2: /* default parameter */
#ifdef G__ASM_DBG
                        if (G__asm_dbg) {
                           G__fprinterr(G__serr, " default%d %c tagnum%d %p : %c tagnum%d %d\n"
                                        , itemp
                                        , p_ifunc->param[ifn][itemp]->type
                                        , p_ifunc->param[ifn][itemp]->p_tagtable
                                        , p_ifunc->param[ifn][itemp]->pdefault
                                        , libp->para[itemp].type
                                        , libp->para[itemp].tagnum
                                        , funcmatch);
                        }
#endif
                        itemp = p_ifunc->para_nu[ifn]; /* end of this parameter */
                        break;
                     case 1: /* match this one, next parameter */
#ifdef G__ASM_DBG
                        if (G__asm_dbg) {
                           G__fprinterr(G__serr, " match%d %c tagnum%d %p : %c tagnum%d %d\n"
                                        , itemp
                                        , p_ifunc->param[ifn][itemp]->type
                                        , p_ifunc->param[ifn][itemp]->p_tagtable
                                        , p_ifunc->param[ifn][itemp]->pdefault
                                        , libp->para[itemp].type
                                        , libp->para[itemp].tagnum
                                        , funcmatch);
                        }
#endif
                        if (G__EXACT != funcmatch)
                           G__warn_refpromotion(p_ifunc, ifn, itemp, libp);
                        ++itemp; /* next function parameter */
                        break;
                     case 0: /* unmatch, next function */
#ifdef G__ASM_DBG
                        if (G__asm_dbg) {
                           G__fprinterr(G__serr, " unmatch%d %c tagnum%d %p : %c tagnum%d %d\n"
                                        , itemp
                                        , p_ifunc->param[ifn][itemp]->type
                                        , p_ifunc->param[ifn][itemp]->p_tagtable
                                        , p_ifunc->param[ifn][itemp]->pdefault
                                        , libp->para[itemp].type
                                        , libp->para[itemp].tagnum
                                        , funcmatch);
                        }
#endif
                        itemp = p_ifunc->para_nu[ifn];
                        /* exit from while loop */
                        break;
                  }

               } /* end of while(itemp<p_ifunc->para_nu[ifn]) */
               if (ipara == 0) { /* parameter doesn't match */
                  ++ifn; /* next function */
               }
            }
         }
         else {  /* funcname doesn't match */
            ++ifn;
         }
      }  /* end of while((ipara==0))&&(ifn<p_ifunc->allifunc)) */
      /******************************************************************
       * next page of interpreted function list
       *******************************************************************/
      if (ifn >= p_ifunc->allifunc) {
         p_ifunc = p_ifunc->next;
         ifn = 0;
      }
      else {
         break; /* get out from while(p_ifunc) loop */
      }
   } /* end of while(p_ifunc) */


   *pifn = ifn;
   return(p_ifunc);
}

//______________________________________________________________________________
struct G__ifunc_table_internal* G__get_ifunchandle_base(const char* funcname, G__param* libp, int hash, G__ifunc_table_internal* p_ifunc, long* pifn, long* poffset, int access, int funcmatch, int withInheritance, int isconst)
{
   // -- FIXME: Describe this function!
   int tagnum;
   struct G__ifunc_table_internal *ifunc;
   int basen = 0;
   struct G__inheritance *baseclass;

   /* Search for function */
   *poffset = 0;
   ifunc = G__get_ifunchandle(funcname, libp, hash, p_ifunc, pifn, access, funcmatch, isconst);
   if (ifunc || !withInheritance) return(ifunc);

   /* Search for base class function if member function */
   tagnum = p_ifunc->tagnum;
   if (-1 != tagnum)
   {
      baseclass = G__struct.baseclass[tagnum];
      while (basen < baseclass->basen) {
         if (baseclass->herit[basen]->baseaccess&G__PUBLIC) {
#ifdef G__VIRTUALBASE
            /* Can not handle virtual base class member function for ERTTI
             * because pointer to the object is not given  */
#endif
            *poffset = baseclass->herit[basen]->baseoffset;
            p_ifunc = G__struct.memfunc[baseclass->herit[basen]->basetagnum];
            ifunc = G__get_ifunchandle(funcname, libp, hash, p_ifunc, pifn
                                       , access, funcmatch, isconst);
            if (ifunc) return(ifunc);
         }
         ++basen;
      }
   }

   /* Not found , ifunc=NULL */
   return(ifunc);
}

//______________________________________________________________________________
//
// 05-02-08
// Add new parameters needed to avoid code replication, these are:
// "int noerror,int& error"
// If we want to obtain the old behaviour we have to call this function
// with noerror=0 and some placeholder to keep the value of error 
// (or zero if we want to ignore it too)
void G__argtype2param(const char* argtype, G__param* libp, int noerror, int* error)
{
   // -- FIXME: Describe this function!
   G__FastAllocString typenam(G__MAXNAME*2);
   int p = 0;
   int c;
   const char *endmark = ",);";

   libp->paran = 0;
   libp->para[0] = G__null;

   do {
      c = G__getstream_template(argtype, &p, typenam, 0, endmark);
      if (typenam[0]) {
         char* start = typenam;
         while (isspace(*start)) ++start;
         if (*start) {
            char* end = start + strlen(start) - 1;
            while (isspace(*end) && end != start) --end;
         }
         G__value buf = G__string2type_noerror(start, noerror);

         // LF 17-07-07
         if (error && buf.type==0 && buf.typenum==-1)
            *error = 1;

         // LF 20/04/07
         // This means the argument is "..."
         // How do we handle that properly?
         if (buf.type != -1) {
            libp->para[libp->paran] = buf;
            ++libp->paran;
         }
      }
   }
   while (',' == c);
}

//______________________________________________________________________________
//
// 05-02-08
// We add extraparameters to avoid code replication. We need no pass the noerror
// argument to G__argtype2param and isconst to G__get_ifunchandle_base
//
// Note: the constness was not checked before and I think it's important
// to check it, afterall const overloading is allowed. The problem is that
// Cint was ignoring it before so we need it an option to do exactly that.. 
// ignore it.
// isconst can be: 
// 0 (do everything as before, i.e. ignore the constness)
// 1 (look for a non-const function)
// 2 (look for a const function)
// Note: This should be put into something like an enum
struct G__ifunc_table* G__get_methodhandle_noerror(const char* funcname, const char* argtype, G__ifunc_table* p_iref, long* pifn, long* poffset, int withConversion, int withInheritance, int noerror, int isconst)
{
   // -- FIXME: Describe this function!
   struct G__ifunc_table_internal *ifunc;
   struct G__ifunc_table_internal *p_ifunc = G__get_ifunc_internal(p_iref);
   struct G__param para;
   int hash;
   int temp;
   struct G__funclist *funclist = (struct G__funclist*)NULL;

   int store_def_tagnum = G__def_tagnum;
   int store_tagdefining = G__tagdefining;
   G__def_tagnum = p_ifunc->tagnum;
   G__tagdefining = p_ifunc->tagnum;

   // 17-07-07
   int error = 0;
   G__argtype2param(argtype,&para, noerror, &error);
   //G__argtype2param(argtype, &para, 0, 0);
   G__def_tagnum = store_def_tagnum;
   G__tagdefining = store_tagdefining;
   G__hash(funcname, hash, temp);

   if(error)
      return 0;

   if (withConversion)
   {
      int tagnum = p_ifunc->tagnum;
      int ifn = (int)(*pifn);

      if (-1 != tagnum) G__incsetup_memfunc(tagnum);

      ifunc = G__overload_match(funcname, &para, hash, p_ifunc, G__TRYNORMAL
                                , G__PUBLIC_PROTECTED_PRIVATE, &ifn, 0
                                , (withConversion & 0x2) ? 1 : 0, false);
      *poffset = 0;
      *pifn = ifn;
      if (ifunc || !withInheritance) return G__get_ifunc_ref(ifunc);
      if (-1 != tagnum) {
         int basen = 0;
         struct G__inheritance *baseclass = G__struct.baseclass[tagnum];
         while (basen < baseclass->basen) {
            if (baseclass->herit[basen]->baseaccess&G__PUBLIC) {
               G__incsetup_memfunc(baseclass->herit[basen]->basetagnum);
               *poffset = baseclass->herit[basen]->baseoffset;
               p_ifunc = G__struct.memfunc[baseclass->herit[basen]->basetagnum];
               ifunc = G__overload_match(funcname, &para, hash, p_ifunc, G__TRYNORMAL
                                         , G__PUBLIC_PROTECTED_PRIVATE, &ifn, 0, 0, false);
               *pifn = ifn;
               if (ifunc) return G__get_ifunc_ref(ifunc);
            }
            ++basen;
         }
      }
   }
   else
   {
      /* first, search for exact match */
      ifunc = G__get_ifunchandle_base(funcname, &para, hash, p_ifunc, pifn, poffset
                                      , G__PUBLIC_PROTECTED_PRIVATE, G__EXACT
                                      , withInheritance, isconst
                                     );
      if (ifunc) return G__get_ifunc_ref(ifunc);

      /* if no exact match, try to instantiate template function */
      // 24-10-07 Don't try this when registering symbols...
      // the match should be exact.. shouldnt it?
      if(noerror==0) {
         funclist = G__add_templatefunc(funcname, &para, hash, funclist, p_ifunc, 0);
         if (funclist && funclist->rate == G__EXACTMATCH) {
           ifunc = funclist->ifunc;
            *pifn = funclist->ifn;
            G__funclist_delete(funclist);
            return G__get_ifunc_ref(ifunc);
         }
         G__funclist_delete(funclist);
      }

      // FIXME: Remove this code for now, we should not attempt conversions
      //        here, we have specified an exact prototype.
      //
      //for (int match = G__EXACT;match <= G__STDCONV;match++) {
      //   ifunc = G__get_ifunchandle_base(funcname, &para, hash, p_ifunc, pifn, poffset
      //                                   , G__PUBLIC_PROTECTED_PRIVATE
      //                                   , match
      //                                   , withInheritance
      //                                  );
      //   if (ifunc) return G__get_ifunc_ref(ifunc);
      //}
   }

   return G__get_ifunc_ref(ifunc);
}

//______________________________________________________________________________
//
// This will behave as the old function but is based in a new implementation
// where "noerror" and "isconst" must be given
struct G__ifunc_table* G__get_methodhandle(const char* funcname, const char* argtype, G__ifunc_table* p_iref, long* pifn, long* poffset, int withConversion, int withInheritance)
{
  return G__get_methodhandle_noerror(funcname, argtype, p_iref, pifn, poffset, withConversion, withInheritance, 0, 0);
}


//______________________________________________________________________________
struct G__ifunc_table* G__get_methodhandle2(char* funcname, G__param* libp, G__ifunc_table* p_iref, long* pifn, long* poffset, int withConversion, int withInheritance)
{
   // -- FIXME: Describe this function!
   struct G__ifunc_table_internal* ifunc;
   struct G__ifunc_table_internal* p_ifunc = G__get_ifunc_internal(p_iref);
   int hash;
   int temp;
   struct G__funclist* funclist = 0;
   int store_def_tagnum = G__def_tagnum;
   int store_tagdefining = G__tagdefining;
   G__def_tagnum = p_ifunc->tagnum;
   G__tagdefining = p_ifunc->tagnum;
   G__def_tagnum = store_def_tagnum;
   G__tagdefining = store_tagdefining;
   G__hash(funcname, hash, temp);
   if (withConversion)
   {
      int tagnum = p_ifunc->tagnum;
      int ifn = (int) (*pifn);
      if (-1 != tagnum) {
         G__incsetup_memfunc(tagnum);
      }
      ifunc = G__overload_match(funcname, libp, hash, p_ifunc, G__TRYNORMAL, G__PUBLIC_PROTECTED_PRIVATE, &ifn, 0, 0, false);
      *poffset = 0;
      *pifn = ifn;
      if (ifunc || !withInheritance) {
         return G__get_ifunc_ref(ifunc);
      }
      if (-1 != tagnum) {
         int basen = 0;
         struct G__inheritance* baseclass = G__struct.baseclass[tagnum];
         while (basen < baseclass->basen) {
            if (baseclass->herit[basen]->baseaccess&G__PUBLIC) {
               G__incsetup_memfunc(baseclass->herit[basen]->basetagnum);
               *poffset = baseclass->herit[basen]->baseoffset;
               p_ifunc = G__struct.memfunc[baseclass->herit[basen]->basetagnum];
               ifunc = G__overload_match(funcname, libp, hash, p_ifunc, G__TRYNORMAL, G__PUBLIC_PROTECTED_PRIVATE, &ifn, 0, 0, false);
               *pifn = ifn;
               if (ifunc) {
                  return G__get_ifunc_ref(ifunc);
               }
            }
            ++basen;
         }
      }
   }
   else
   {
      /* first, search for exact match */
      ifunc = G__get_ifunchandle_base(funcname, libp, hash, p_ifunc, pifn, poffset
                                      , G__PUBLIC_PROTECTED_PRIVATE, G__EXACT
                                      , withInheritance, 0
                                     );
      if (ifunc) return G__get_ifunc_ref(ifunc);

      /* if no exact match, try to instantiate template function */
      funclist = G__add_templatefunc(funcname, libp, hash, funclist, p_ifunc, 0);
      if (funclist && funclist->rate == G__EXACTMATCH) {
         ifunc = funclist->ifunc;
         *pifn = funclist->ifn;
         G__funclist_delete(funclist);
         return G__get_ifunc_ref(ifunc);
      }
      G__funclist_delete(funclist);

      //
      // Now see if we can call it using the standard type conversions.
      //
      for (int match = G__EXACT;match <= G__STDCONV;match++) {
         ifunc = G__get_ifunchandle_base(funcname, libp, hash, p_ifunc, pifn, poffset
                                         , G__PUBLIC_PROTECTED_PRIVATE
                                         , match
                                         , withInheritance, 0
                                        );
         if (ifunc) return G__get_ifunc_ref(ifunc);
      }
   }

   return G__get_ifunc_ref(ifunc);
}

/**************************************************************************
* G__get_methodhandle4
*
* 03-08-07
* Exactly the same as 'G__get_methodhandle2' except that we receive a 
* G__ifunc_table_internal instead of a G__ifunc_table
* and return the same (this makes things slightly faster)
* this code replication could be avoided but for the moment I prefer
* to keep things clear
* FIXME: Code replication
**************************************************************************/
struct G__ifunc_table_internal* G__get_methodhandle4(char* funcname
      , struct G__param* libp
      , G__ifunc_table_internal* p_ifunc
      , long* pifn, long* poffset
      , int withConversion
      , int withInheritance
      , int isconst)
{
   struct G__ifunc_table_internal* ifunc;
   //struct G__ifunc_table_internal *p_ifunc = G__get_ifunc_internal(p_iref);
   int hash;
   int temp;
   int match;

   int store_def_tagnum = G__def_tagnum;
   int store_tagdefining = G__tagdefining;
   G__def_tagnum = p_ifunc->tagnum;
   G__tagdefining = p_ifunc->tagnum;
   G__def_tagnum = store_def_tagnum;
   G__tagdefining = store_tagdefining;
   G__hash(funcname, hash, temp);


   if (withConversion) {
      int tagnum = p_ifunc->tagnum;
      int ifn = (int)(*pifn);
      if (-1 != tagnum) {
         G__incsetup_memfunc(tagnum);
      }
      ifunc = G__overload_match(funcname, libp, hash, p_ifunc, G__TRYNORMAL
                                , G__PUBLIC_PROTECTED_PRIVATE, &ifn, 1, 0, false) ;
      *poffset = 0;
      *pifn = ifn;
      if (ifunc || !withInheritance) {
         return ifunc;
      }
      if (-1 != tagnum) {
         int basen = 0;
         struct G__inheritance* baseclass = G__struct.baseclass[tagnum];
         while (basen < baseclass->basen) {
            if (baseclass->herit[basen]->baseaccess & G__PUBLIC) {
               G__incsetup_memfunc(baseclass->herit[basen]->basetagnum);
               *poffset = baseclass->herit[basen]->baseoffset;
               p_ifunc = G__struct.memfunc[baseclass->herit[basen]->basetagnum];
               ifunc = G__overload_match(funcname, libp, hash, p_ifunc, G__TRYNORMAL
                                         , G__PUBLIC_PROTECTED_PRIVATE, &ifn, 1, 0, false) ;
               *pifn = ifn;
               if (ifunc) {
                  return ifunc;
               }
            }
            ++basen;
         }
      }
   }
   else {
      /* first, search for exact match */
      ifunc = G__get_ifunchandle_base(funcname, libp, hash, p_ifunc, pifn, poffset
                                      , G__PUBLIC_PROTECTED_PRIVATE, G__EXACT
                                      , withInheritance, isconst
                                     );
      if (ifunc) {
         return ifunc;
      }
      /* if no exact match, try to instantiate template function */
      // LF 24-10-07 Don't try this when registering symbols...
      // the match should be exact.. shouldnt it?
      //funclist = G__add_templatefunc(funcname,libp,hash,funclist,p_ifunc,1,withInheritance);
      //if(funclist && funclist->rate==G__EXACTMATCH) {
      //  ifunc = funclist->ifunc;
      //  *pifn = funclist->ifn;
      //  G__funclist_delete(funclist);
      //  return ifunc;
      //}
      //G__funclist_delete(funclist);
      for (match = G__EXACT; match <= G__STDCONV; match++) {
         ifunc = G__get_ifunchandle_base(funcname, libp, hash, p_ifunc, pifn, poffset
                                         , G__PUBLIC_PROTECTED_PRIVATE
                                         , match
                                         , withInheritance, isconst
                                        );
         if (ifunc) {
            return ifunc;
         }
      }
   }

   return ifunc;
}


//______________________________________________________________________________
struct G__ifunc_table* G__get_ifunc_ref(struct G__ifunc_table_internal* ifunc)
{
   // -- Returns the G__ifunc_table reference object for an internal G__ifunc_table_internal and creates it if it doesn't exist.
   if (!ifunc) return 0;
   G__ifunc_table iref;
   iref.tagnum = ifunc->tagnum;
   iref.page = ifunc->page;
   std::set<G__ifunc_table>::const_iterator irefIter = G__ifunc_refs()[iref.tagnum].insert(iref).first;
   G__ifunc_table& irefInSet = const_cast<G__ifunc_table&>(*irefIter);
   irefInSet.ifunc_cached = ifunc;
   return &irefInSet;
}

//______________________________________________________________________________
void G__reset_ifunc_refs_for_tagnum(int tagnum)
{
   // -- Resets cache for the G__ifunc_table reference set for a tagnum.
   std::map<int, std::set<G__ifunc_table> >::const_iterator iRefSet = G__ifunc_refs().find(tagnum);
   if (iRefSet == G__ifunc_refs().end()) return;
   for (
      std::set<G__ifunc_table>::const_iterator iRef = iRefSet->second.begin();
      iRef != iRefSet->second.end();
      ++iRef
   ) {
      const_cast<G__ifunc_table&>(*iRef).ifunc_cached = 0;
   }
}

//______________________________________________________________________________
void G__reset_ifunc_refs(G__ifunc_table_internal* ifunc)
{
   // -- Resets cache for a (global) function's G__ifunc_table reference.
   if (!ifunc) return;
   std::map<int, std::set<G__ifunc_table> >::const_iterator iRefSet = G__ifunc_refs().find(ifunc->tagnum);
   if (iRefSet == G__ifunc_refs().end()) return;
   G__ifunc_table iref;
   iref.tagnum = ifunc->tagnum;
   iref.page = ifunc->page;
   std::set<G__ifunc_table>::const_iterator iRef = iRefSet->second.find(iref);
   if (iRef != iRefSet->second.end()) {
      const_cast<G__ifunc_table&>(*iRef).ifunc_cached = 0;
   }
}

//______________________________________________________________________________
struct G__ifunc_table_internal* G__get_ifunc_internal(struct G__ifunc_table* iref)
{
   // -- Returns the G__ifunc_table_internal object for a reference object.
   if (!iref) return 0;
   if (iref->ifunc_cached) return iref->ifunc_cached;
   int tagnum = iref->tagnum;
   if (tagnum != -1)
   {
      if (tagnum >= G__struct.alltag) return 0;
      G__incsetup_memfunc(tagnum); // make sure funcs are setup
      G__ifunc_table_internal* ifunc = G__struct.memfunc[tagnum];
      for (int page = 0; page < iref->page && ifunc; ++page)
         ifunc = ifunc->next;
      return ifunc;
   }
   // We cannot re-initialize global ifuncs when they are re-loaded.
   // We only have their position (page), and that might well change.
   return 0;
}

} // extern "C"

/*
 * Local Variables:
 * c-tab-always-indent:nil
 * c-indent-level:3
 * c-continued-statement-offset:3
 * c-brace-offset:-3
 * c-brace-imaginary-offset:0
 * c-argdecl-indent:0
 * c-label-offset:-3
 * compile-command:"make -k"
 * End:
 */
