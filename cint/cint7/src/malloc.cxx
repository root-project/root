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
using namespace Cint::Internal;

#ifdef G__SHMGLOBAL

#include <unistd.h>
#include <sys/types.h>
#include <sys/shm.h>

//______________________________________________________________________________
//______________________________________________________________________________
//______________________________________________________________________________

int G__myshmid;
void* G__shmbuffer;
int* G__pthreadnum;
int G__shmsize = 0x10000;

//______________________________________________________________________________
int Cint::Internal::G__CreateThread()
{
  return (int) fork();
}

//______________________________________________________________________________
void* Cint::Internal::G__shmmalloc(int size)
{
   int* poffset = (int*) G__shmbuffer;
   int alignsize = sizeof(double);
   if (size <= sizeof(double)) {
      alignsize = size;
   }
   if (*poffset % alignsize) {
      *poffset += sizeof(alignsize) - (*poffset % alignsize);
   }
   void* result = (void*) ((long) G__shmbuffer + *poffset);
   *poffset += size;
   if (*poffset >= G__shmsize) {
      return 0;
   }
   return result;
}

//______________________________________________________________________________
void* Cint::Internal::G__shmcalloc(int atomsize, int num)
{
   int size = atomsize * num;
   void* result = G__shmmalloc(size);
   if (!result) {
      return result;
   }
   for (int i = 0;i < size;i++) {
      *((char*)result + i) = '\0';
   }
   return result;
}

//______________________________________________________________________________
void Cint::Internal::G__shmfinish()
{
   // -- Free shared memory.
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
void Cint::Internal::G__shminit()
{
   // -- Prepare keys.
   const char projid = 'Q';
   char keyfile[256];
   G__getcintsysdir();
   sprintf(keyfile, "%s/cint", G__cintsysdir);
   key_t mykey = ftok(keyfile, projid);
   // Shared Memory
   G__myshmid = shmget(mykey, G__shmsize, SHM_R | SHM_W | IPC_CREAT);
   if (G__myshmid == -1) {
      fprintf(stderr, "shmget failed\n");
      exit(1);
   }
   G__shmbuffer = shmat(G__myshmid, 0, 0);
   if (G__shmbuffer == (void*)(~0)) {
      fprintf(stderr, "shmat failed\n");
      exit(1);
   }
   // set offset address
   int* poffset = (int*)G__shmcalloc(sizeof(int), 1);
   *poffset = 0;
   // set thread num count
   G__pthreadnum = (int*)G__shmcalloc(sizeof(int), 1);
   *G__pthreadnum = 1;
   // set finish function
   atexit(G__shmfinish);
}

#endif // G__SHMGLOBAL

//______________________________________________________________________________
//______________________________________________________________________________
//______________________________________________________________________________

//______________________________________________________________________________
static std::string& G__get_static_varname()
{
   static std::string name;
   return name;
}

//______________________________________________________________________________
void Cint::Internal::G__set_static_varname(const char* name) 
{
   G__get_static_varname() = name;
}

//______________________________________________________________________________
static void* G__getstaticobject()
{
   std::string temp;
   G__get_stack_varname(temp, G__get_static_varname().c_str(), G__func_now, G__get_tagnum(G__memberfunc_tagnum));
   ::Reflex::Member m = ::Reflex::Scope::GlobalScope().DataMemberByName(temp);
   if (m) {
      return G__get_offset(m);
   }
   if (!G__const_noerror) {
      G__fprinterr(G__serr, "Error: No memory for static %s ", temp.c_str());
      G__genericerror(0);
   }
   return 0;
}

//______________________________________________________________________________
void* Cint::Internal::G__malloc(int n, int bsize, const char* item)
{
   // -- Allocate memory.
#ifdef G__MEMTEST
   fprintf(G__memhist, "G__malloc(%d,%d,%s)\n", n, bsize, item);
#endif
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
         // --
#ifdef G__ASM_WHOLEFUNC
         !G__def_struct_member && (G__asm_wholefunction == G__ASM_FUNC_NOP)
#else // G__ASM_WHOLEFUNC
         !G__def_struct_member
#endif // G__ASM_WHOLEFUNC
         // --
      ) {
         // -- Not a member variable and we are not doing whole function bytecode generation.
         if (G__static_alloc && G__func_now && !G__prerun) {
            return G__getstaticobject();
         }
         // Allocate memory area. Normal case.
         void* allocmem = 0;
         if (G__prerun) {
            allocmem = calloc(n, bsize);
         }
         else {
            allocmem = malloc(size);
         }
         if (!allocmem) {
            G__malloc_error(item);
         }
         return allocmem;
      }
      else {
         // -- Interpretively calculate struct offset.
         //
         // In case of struct conservative padding strategy.
         if (
            (G__struct.type[G__get_tagnum(G__tagdefining)] == 's') ||
            (G__struct.type[G__get_tagnum(G__tagdefining)] == 'c')
         ) {
            // -- Allocate an offset and padding for a struct or class member.
            if (G__static_alloc) {
               if (G__asm_wholefunction == G__ASM_FUNC_COMPILE) {
                  return G__getstaticobject();
               }
               void* allocmem = calloc(n, bsize);
               if (!allocmem) {
                  G__malloc_error(item);
               }
               return allocmem;
            }
            long allocmem = 0;
            // Get padding size.
            if (bsize > (int) G__DOUBLEALLOC) {
               allocmem = G__DOUBLEALLOC;
            }
            else {
               allocmem = bsize;
            }
            // Get padding size.
            G__struct.size[G__get_tagnum(G__tagdefining)] += size;
            // Padding.
            if (allocmem && (G__struct.size[G__get_tagnum(G__tagdefining)] % allocmem)) {
               G__struct.size[G__get_tagnum(G__tagdefining)] += allocmem - (G__struct.size[G__get_tagnum(G__tagdefining)] % allocmem);
            }
            return (void*) (long) (G__struct.size[G__get_tagnum(G__tagdefining)] - size);
         }
         else if (G__struct.type[G__get_tagnum(G__tagdefining)] == 'u') {
            // -- Adjust size and padding of union for this new member.
            if (G__struct.size[G__get_tagnum(G__tagdefining)] < size) {
               G__struct.size[G__get_tagnum(G__tagdefining)] = size;
               if ((size % 2) == 1)
                  G__struct.size[G__get_tagnum(G__tagdefining)]++;
            }
            return 0;
         }
         else if (G__struct.type[G__get_tagnum(G__tagdefining)] == 'n') {
            // -- Actually allocate memory for a namespace member.
            void* allocmem = calloc(n , bsize);
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
