/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file malloc.c
 ************************************************************************
 * Description:
 *  Allocate automatic variable arena
 ************************************************************************
 * Copyright(c) 1995~2002  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"

#ifdef G__SHMGLOBAL

#include <unistd.h>
#include <sys/types.h>
#include <sys/shm.h>

//______________________________________________________________________________
//______________________________________________________________________________
//______________________________________________________________________________

extern "C" {

int G__myshmid = 0;
void* G__shmbuffer = 0;
int* G__pthreadnum = 0;
int G__shmsize = 0x10000;

//______________________________________________________________________________
int G__CreateThread()
{
   return (int) fork();
}

//______________________________________________________________________________
void* G__shmmalloc(int size)
{
   int* poffset = (int*)G__shmbuffer;
   void* result;
   int alignsize;

   if (size <= sizeof(double)) alignsize = size;
#if defined(G__64BIT)
   else                        alignsize = sizeof(double);
#else
   else                        alignsize = sizeof(long);
#endif
   if ((*poffset) % alignsize) *poffset += sizeof(alignsize) - (*poffset) % alignsize;

   result = (void*)((long)G__shmbuffer + (*poffset));

   (*poffset) += size;
   if ((*poffset) >= G__shmsize) return((void*)0);

   return(result);
}

//______________________________________________________________________________
void* G__shmcalloc(int atomsize, int num)
{
   int i;
   int size = atomsize * num;
   void* result = G__shmmalloc(size);
   if (!result) return(result);
   for (i = 0;i < size;i++) *((char*)result + i) = (char)0;
   return(result);
}

//______________________________________________________________________________
void G__shmfinish()
{
   /* free shared memory */
   switch (*G__pthreadnum) {
      case 0:
         break;
      case 1:
         (*G__pthreadnum)--;
         shmdt(G__shmbuffer);
         shmctl(G__myshmid, IPC_RMID, 0);
         break;
      default:
         (*G__pthreadnum)--;
         shmdt(G__shmbuffer);
         break;
   }
}

//______________________________________________________________________________
void G__shminit()
{
   /* Prepare keys */
   key_t mykey;
   const char projid = 'Q';
   G__FastAllocString keyfile(256);
   int* poffset;

   G__getcintsysdir();
   keyfile.Format("%s/cint", G__cintsysdir);
   mykey = ftok(keyfile, projid);
   /* printf("mykey=%x\n",mykey); */

   /* Shared Memory */
   G__myshmid = shmget(mykey, G__shmsize, SHM_R | SHM_W | IPC_CREAT);
   /* printf("myshmid=%x\n",G__myshmid); */
   if (-1 == G__myshmid) {
      fprintf(stderr, "shmget failed\n");
      throw std::runtime_error("CINT: shmget failed")
   }

   G__shmbuffer = shmat(G__myshmid, 0, 0);
   /* printf("shmbuffer=%p\n",G__shmbuffer); */
   if ((void*)(~0) == G__shmbuffer) {
      fprintf(stderr, "shmat failed\n");
      throw std::runtime_error("CINT: shmat failed")
   }

   /* set offset address */
   poffset = (int*)G__shmcalloc(sizeof(int), 1);
   *poffset = 0;

   /* set thread num count */
   G__pthreadnum = (int*)G__shmcalloc(sizeof(int), 1);
   *G__pthreadnum = 1;

   /* set finish function */
   atexit(G__shmfinish);
}

} // extern "C"

#endif // G__SHMGLOBAL

//______________________________________________________________________________
//______________________________________________________________________________
//______________________________________________________________________________

extern "C" {

//______________________________________________________________________________
static long G__getstaticobject()
{
   G__FastAllocString temp(G__ONELINE);
   if (G__memberfunc_tagnum != -1) {
      temp.Format("%s\\%x\\%x\\%x", G__varname_now, G__func_page, G__func_now, G__memberfunc_tagnum);
   }
   else {
      temp.Format("%s\\%x\\%x", G__varname_now, G__func_page, G__func_now);
   }
   int hash = 0;
   int i = 0;
   G__hash(temp, hash, i)
   struct G__var_array* var = &G__global;
   for (; var; var = var->next) {
      for (i = 0; i < var->allvar; ++i) {
         if ((var->hash[i] == hash) && !strcmp(var->varnamebuf[i], temp)) {
            //fprintf(stderr, "G__getstaticobject: Found '%s' addr %08X for '%s'\n", temp, var->p[i], G__varname_now);
            return var->p[i];
         }
      }
   }
   if (!G__const_noerror) {
      G__fprinterr(G__serr, "Error: No memory for static %s ", temp());
      G__genericerror(0);
   }
   return 0;
}

//______________________________________________________________________________
long G__malloc(int n, int bsize, const char* item)
{
   // -- Allocate memory.
#ifdef G__MEMTEST
   fprintf(G__memhist, "G__malloc(%d,%d,%s)\n", n, bsize, item);
#endif // G__MEMTEST
   // Calculate total malloc size.
   int size = n * bsize;
   // Experimental reference type in bytecode.
   if ((G__globalvarpointer != G__PVOID) && (G__asm_wholefunction == G__ASM_FUNC_COMPILE)) {
      G__globalvarpointer = G__PVOID;
      size = G__LONGALLOC;
   }
   // Do we already have allocated memory for object?
   if (G__globalvarpointer == G__PVOID) {
      // -- No, interpretively allocate memory.
      if (
#ifdef G__ASM_WHOLEFUNC
         !G__def_struct_member && (G__asm_wholefunction == G__ASM_FUNC_NOP)
#else // G__ASM_WHOLEFUNC
         !G__def_struct_member
#endif // G__ASM_WHOLEFUNC
      ) {
         // -- Not a member variable and we are not doing whole function bytecode generation.
         if (G__static_alloc && (G__func_now > -1) && !G__prerun) {
            // -- Static variable in function scope which is already allocated at pre-RUN.
            return G__getstaticobject();
         }
         // Allocate memory area. Normal case.
         long allocmem = 0;
         if (G__prerun) {
            allocmem = (long) calloc((size_t) n, (size_t) bsize);
         }
         else {
            allocmem = (long) malloc((size_t) size);
         }
         if (!allocmem) {
            G__malloc_error(item);
         }
         return allocmem;
      }
      else {
         // -- Interpretively calculate struct offset
         // In case of struct conservative padding strategy.
         if ((G__struct.type[G__tagdefining] == 's') || (G__struct.type[G__tagdefining] == 'c')) {
            // -- Allocate an offset and padding for a struct or class member.
            if (G__static_alloc) {
               if (G__ASM_FUNC_COMPILE == G__asm_wholefunction) {
                  return G__getstaticobject();
               }
               long allocmem = (long) calloc((size_t) n, (size_t) bsize);
               if (!allocmem) {
                  G__malloc_error(item);
               }
               return allocmem;
            }
            long allocmem = 0;
            // Get padding size.
#if defined(G__64BIT)
            if (bsize > ((int) G__DOUBLEALLOC)) {
               allocmem = G__DOUBLEALLOC;
            }
#else
            if (bsize > ((int) G__LONGALLOC)) {
               allocmem = sizeof(int);
            }
#endif
            else {
               allocmem = bsize;
            }
            // Get padding size.
            G__struct.size[G__tagdefining] += size;
            // Padding .
            if (allocmem && (G__struct.size[G__tagdefining] % allocmem)) {
               G__struct.size[G__tagdefining] += allocmem - (G__struct.size[G__tagdefining] % allocmem);
            }
            return G__struct.size[G__tagdefining] - size;
         }
         else if (G__struct.type[G__tagdefining] == 'u') {
            // -- Adjust size and padding of union for this new member.
            if (G__struct.size[G__tagdefining] < size) {
               G__struct.size[G__tagdefining] = size;
               if ((size % 2) == 1)
                  G__struct.size[G__tagdefining]++;
            }
            return 0;
         }
         else if (G__struct.type[G__tagdefining] == 'n') {
            // -- Actually allocate memory for a namespace member.
            long allocmem = (long) calloc((size_t) n, (size_t) bsize);
            if (!allocmem) {
               G__malloc_error(item);
            }
            return allocmem;
         }
      }
   }
   // Return address of already allocated memory.
   return G__globalvarpointer;
}

} /* extern "C" */

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
