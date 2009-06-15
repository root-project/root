#include "common.h"
#include "Dict.h"
using namespace ::Cint::Internal;
namespace Cint {
namespace Internal {
int G__exec_asm(int start, int stack, G__value* presult, char* localmem);
} // namespace Internal
} // namespace Cint

extern "C" int G__exec_bytecode(G__value* result7, G__CONST char* funcname, struct G__param* libp, int /*hash*/)
{
   int i;
   struct G__bytecodefunc *bytecode;
   G__value asm_stack_g[G__MAXSTACK]; /* data stack */
   long *store_asm_inst;
   G__value *store_asm_stack;
   char *store_asm_name;
   int store_asm_name_p;
   struct G__param *store_asm_param;
   int store_asm_exec;
   int store_asm_noverflow;
   int store_asm_cp;
   int store_asm_dt;
   Reflex::Member store_asm_index; /* maybe unneccessary */
   ::Reflex::Scope store_tagnum;
   char* localmem;
   int store_exec_memberfunc;
   char* store_memberfunc_struct_offset;
   ::Reflex::Scope store_memberfunc_tagnum;
   const int G__LOCALBUFSIZE = 32;
   G__StrBuf localbuf_sb(G__LOCALBUFSIZE);
   char *localbuf = localbuf_sb;
   if (G__asm_dbg) {
      G__fprinterr(G__serr, "G__exec_bytecode: starting bytecode execution ...\n");
   }
   // use funcname as bytecode struct
   bytecode = (struct G__bytecodefunc*) funcname;
#ifdef G__ASM_DBG
   if (G__asm_dbg || (G__dispsource && !G__stepover)) {
      if (bytecode->ifunc.DeclaringScope().IsClass()) {
         G__fprinterr(G__serr, "Running bytecode function %s inst=%lx->%lx stack=%lx->%lx stros=%lx %d  %s:%d\n", bytecode->ifunc.Name(::Reflex::SCOPED).c_str(), (long) G__asm_inst, (long) bytecode->pinst, (long) G__asm_stack, (long) asm_stack_g, G__store_struct_offset, G__get_tagnum(G__tagnum), __FILE__, __LINE__);
      }
      else {
         G__fprinterr(G__serr, "Running bytecode function %s inst=%lx->%lx stack=%lx->%lx  %s:%d\n", bytecode->ifunc.Name(::Reflex::SCOPED).c_str(), (long) G__asm_inst, (long) bytecode->pinst, (long) G__asm_stack, (long) asm_stack_g, __FILE__, __LINE__);
      }
   }
#endif // G__ASM_DBG
   // Push loop compilation environment
   store_asm_inst = G__asm_inst;
   store_asm_stack = G__asm_stack;
   store_asm_name = G__asm_name;
   store_asm_name_p = G__asm_name_p;
   store_asm_param  = G__asm_param ;
   store_asm_exec  = G__asm_exec ;
   store_asm_noverflow  = G__asm_noverflow ;
   store_asm_cp  = G__asm_cp ;
   store_asm_dt  = G__asm_dt ;
   store_asm_index  = G__asm_index ;
   store_tagnum = G__tagnum;
   store_exec_memberfunc = G__exec_memberfunc;
   store_memberfunc_struct_offset = G__memberfunc_struct_offset;
   store_memberfunc_tagnum = G__memberfunc_tagnum;
   // set new bytecode environment
   G__asm_inst = bytecode->pinst;
   G__asm_stack = asm_stack_g;
   G__asm_name = bytecode->asm_name;
   G__asm_name_p = 0;
   G__tagnum = bytecode->frame.DeclaringScope();
   G__asm_noverflow = 0; /* bug fix */
   if (bytecode->ifunc.DeclaringScope() && !bytecode->ifunc.DeclaringScope().IsTopScope()) {
      G__exec_memberfunc = 1;
   }
   else {
      G__exec_memberfunc = 0;
   }
   G__memberfunc_struct_offset = G__store_struct_offset;
   G__memberfunc_tagnum = G__tagnum;
   // copy constant buffer
   {
      int nx = bytecode->stacksize;
      int ny = G__MAXSTACK - nx;
      for (i = 0; i < nx; i++) {
         asm_stack_g[ny+i] = bytecode->pstack[i];
      }
   }
   // copy arguments to stack in reverse order
   for (i = 0; i < libp->paran; ++i) {
      int j = libp->paran - i - 1;
      G__asm_stack[j] = libp->para[i];
      if (
         bytecode->frame.DataMemberAt(i) &&
         (
            !G__asm_stack[j].ref ||
            (
               (G__get_reftype(bytecode->frame.DataMemberAt(i).TypeOf()) == G__PARAREFERENCE) &&
               (G__get_type(bytecode->frame.DataMemberAt(i).TypeOf()) != G__get_type(G__value_typenum(libp->para[i])))
            )
         )
      ) {
         switch (G__get_type(bytecode->frame.DataMemberAt(i).TypeOf())) {
            case 'f':
               // -- float
               G__asm_stack[j].ref = (long) G__Floatref(&libp->para[i]);
               break;
            case 'd':
               // -- double
               G__asm_stack[j].ref = (long) G__Doubleref(&libp->para[i]);
               break;
            case 'c':
               // -- char
               G__asm_stack[j].ref = (long) G__Charref(&libp->para[i]);
               break;
            case 's':
               // -- short
               G__asm_stack[j].ref = (long) G__Shortref(&libp->para[i]);
               break;
            case 'i':
               // -- int
               G__asm_stack[j].ref = (long) G__Intref(&libp->para[i]);
               break;
            case 'l':
               // -- long
               G__asm_stack[j].ref = (long) G__Longref(&libp->para[i]);
               break;
            case 'b':
               // -- unsigned char
               G__asm_stack[j].ref = (long) G__UCharref(&libp->para[i]);
               break;
            case 'r':
               // -- unsigned short
               G__asm_stack[j].ref = (long) G__UShortref(&libp->para[i]);
               break;
            case 'h':
               // -- unsigned int
               G__asm_stack[j].ref = (long) G__UIntref(&libp->para[i]);
               break;
            case 'k':
               // -- unsigned long
               G__asm_stack[j].ref = (long) G__ULongref(&libp->para[i]);
               break;
            case 'u':
               // -- class, enum, struct, union
               G__asm_stack[j].ref = libp->para[i].obj.i;
               break;
            case 'g':
               // -- bool
               G__asm_stack[j].ref = (long) G__UCharref(&libp->para[i]);
               break;
            case 'n':
               // -- long long
               G__asm_stack[j].ref = (long) G__Longlongref(&libp->para[i]);
               break;
            case 'm':
               // -- unsigned long long
               G__asm_stack[j].ref = (long) G__ULonglongref(&libp->para[i]);
               break;
            case 'q':
               // -- long double
               G__asm_stack[j].ref = (long) G__Longdoubleref(&libp->para[i]);
               break;
            default:
               // -- everything else
               G__asm_stack[j].ref = (long)(&libp->para[i].obj.i);
               break;
         }
      }
   }
   // allocate local memory
   if (bytecode->varsize > G__LOCALBUFSIZE) {
      localmem = (char*)malloc(bytecode->varsize);
   }
   else {
      localmem = localbuf;
   }
#ifdef G__DUMPFILE
   if (G__dumpfile)
   {
      int ipara;
      G__StrBuf resultx_sb(G__ONELINE);
      char *resultx = resultx_sb;
      for (ipara = 0;ipara < G__dumpspace;ipara++) fprintf(G__dumpfile, " ");
      fprintf(G__dumpfile, "%s(", bytecode->ifunc.Name().c_str());
      for (ipara = 1;ipara <= libp->paran;ipara++) {
         if (ipara != 1) fprintf(G__dumpfile, ",");
         G__valuemonitor(libp->para[ipara-1], resultx);
         fprintf(G__dumpfile, "%s", resultx);
      }
      fprintf(G__dumpfile, ");/*%s %d (bc)*/\n" , G__ifile.name, G__ifile.line_number);
      G__dumpspace += 3;

   }
#endif // G__DUMPFILE
   //
   // Run bytecode function
   //
   ++G__get_funcproperties(bytecode->ifunc)->entry.busy;
   G__exec_asm(0 /*start*/, libp->paran /*stack*/, result7, localmem);
   --G__get_funcproperties(bytecode->ifunc)->entry.busy;
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__StrBuf temp_sb(G__ONELINE);
      char *temp = temp_sb;
      G__fprinterr(G__serr, "bytecode function '%s' returns: %s  %s:%d\n", bytecode->ifunc.Name(::Reflex::SCOPED).c_str(), G__valuemonitor(*result7, temp), __FILE__, __LINE__);
   }
#endif // G__ASM_DBG
   if (G__RETURN_IMMEDIATE >= G__return) {
      G__return = G__RETURN_NON;
   }
#ifdef G__ASM_DBG
   if (G__asm_dbg || (G__dispsource && !G__stepover)) {
      if (bytecode->ifunc.DeclaringScope().IsClass()) {
         G__fprinterr(G__serr, "Exit bytecode function %s restore inst=%lx stack=%lx  %s:%d\n", bytecode->ifunc.Name(::Reflex::SCOPED).c_str(), (long) store_asm_inst, (long) store_asm_stack, __FILE__, __LINE__);
      }
      else {
         G__fprinterr(G__serr, "Exit bytecode function %s restore inst=%lx stack=%lx  %s:%d\n", bytecode->ifunc.Name(::Reflex::SCOPED).c_str(), (long) store_asm_inst, (long) store_asm_stack, __FILE__, __LINE__);
      }
   }
#endif // G__ASM_DBG
#ifdef G__DUMPFILE
   if (G__dumpfile) {
      int ipara;
      G__StrBuf resultx_sb(G__ONELINE);
      char *resultx = resultx_sb;
      G__dumpspace -= 3;
      for (ipara = 0;ipara < G__dumpspace;ipara++) fprintf(G__dumpfile, " ");
      G__valuemonitor(*result7, resultx);
      fprintf(G__dumpfile , "/* return(bc) %s()=%s*/\n", bytecode->ifunc.Name().c_str(), resultx);
   }
#endif // G__DUMPFILE
   // restore bytecode environment
   G__asm_inst = store_asm_inst;
   G__asm_stack = store_asm_stack;
   G__asm_name = store_asm_name;
   G__asm_name_p = store_asm_name_p;
   G__asm_param  = store_asm_param ;
   G__asm_exec  = store_asm_exec ;
   G__asm_noverflow  = store_asm_noverflow ;
   G__asm_cp  = store_asm_cp ;
   G__asm_dt  = store_asm_dt ;
   G__asm_index  = store_asm_index ;
   G__tagnum = store_tagnum;
   G__exec_memberfunc = store_exec_memberfunc;
   G__memberfunc_struct_offset = store_memberfunc_struct_offset;
   G__memberfunc_tagnum = store_memberfunc_tagnum;
   if (G__asm_dbg) {
      G__fprinterr(G__serr, "G__exec_bytecode: end bytecode execution ...\n");
   }
   return 0;
}

#include "bc_exec_asm.h"

