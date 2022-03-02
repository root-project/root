/* @(#)root/clib:$Id$ */
/* Author: */

/* Support for an sbrk-like function that uses mmap.
   Copyright 1992 Free Software Foundation, Inc.

   Contributed by Fred Fish at Cygnus Support.   fnf@cygnus.com

This file is part of the GNU C Library.

The GNU C Library is free software; you can redistribute it and/or
modify it under the terms of the GNU Library General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.

The GNU C Library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Library General Public License for more details.

You should have received a copy of the GNU Library General Public
License along with the GNU C Library; see the file COPYING.LIB.  If
not, write to the Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.  */

#include "mmprivate.h"

#if defined(R__HAVE_MMAP)

#include <stdio.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/types.h>
#ifdef WIN32
typedef char* caddr_t;
#else
#  include <unistd.h>
#  include <sys/mman.h>
#endif

#ifdef __CYGWIN__
#include <cygwin/version.h>
#endif /* __CYGWIN__ */

#ifndef SEEK_SET
#define SEEK_SET 0
#endif

/* Cache the pagesize for the current host machine.  Note that if the host
   does not readily provide a getpagesize() function, we need to emulate it
   elsewhere, not clutter up this file with lots of kluges to try to figure
   it out. */

static size_t pagesize;

#define PAGE_ALIGN(addr) (caddr_t) (((long)(addr) + pagesize - 1) & \
                                    ~(pagesize - 1))

/*  Get core for the memory region specified by MDP, using SIZE as the
    amount to either add to or subtract from the existing region.  Works
    like sbrk(), but using mmap(). */

PTR
__mmalloc_mmap_morecore (mdp, size)
  struct mdesc *mdp;
  int size;
{
  PTR result = NULL;
  off_t foffset;        /* File offset at which new mapping will start */
  size_t mapbytes;      /* Number of bytes to map */
  caddr_t moveto;       /* Address where we wish to move "break value" to */
  caddr_t mapto;        /* Address we actually mapped to */
  char buf = 0;         /* Single byte to write to extend mapped file */
#ifdef WIN32
  HANDLE hMap;
#endif

  if (pagesize == 0)
    {
#ifdef _SC_PAGE_SIZE
      pagesize = sysconf(_SC_PAGE_SIZE);
#else
# ifdef _SC_PAGESIZE
      pagesize = sysconf(_SC_PAGESIZE);
# else
      pagesize = getpagesize();
# endif
#endif
    }
  if (size == 0)
    {
      /* Just return the current "break" value. */
      result = mdp -> breakval;
    }
  else if (size < 0)
    {
      /* We are deallocating memory.  If the amount requested would cause
         us to try to deallocate back past the base of the mmap'd region
         then do nothing, and return NULL.  Otherwise, deallocate the
         memory and return the old break value. */

      /* only munmap whole mapping, called via mmalloc_detach, smaller size
         reductions cause the breakval to be reduced but not the mapping
         to be undone (rdm). */

      if (mdp -> breakval + size >= mdp -> base)
        {
          result = (PTR) mdp -> breakval;
          mdp -> breakval += size;

          if (mdp -> breakval == mdp -> base) {
            /* moveto = PAGE_ALIGN (mdp -> breakval); */
            moveto = PAGE_ALIGN (mdp -> base);
#ifndef WIN32
            munmap (moveto, (size_t) (mdp -> top - moveto));
#else
            UnmapViewOfFile(moveto);
#endif
            mdp -> top = moveto;
          }
        }
    }
  else
    {
      /* We are allocating memory.  Make sure we have an open file
         descriptor and then go on to get the memory. */
      if (mdp -> fd < 0)
        {
          result = NULL;
        }
      else if (mdp -> breakval + size > mdp -> top)
        {
          /* The request would move us past the end of the currently
             mapped memory, so map in enough more memory to satisfy
             the request.  This means we also have to grow the mapped-to
             file by an appropriate amount, since mmap cannot be used
             to extend a file. */
          moveto = PAGE_ALIGN (mdp -> breakval + size);
          mapbytes = moveto - mdp -> top;
          foffset = mdp -> top - mdp -> base;
#ifndef WIN32
          if (lseek (mdp -> fd, foffset + mapbytes - 1, SEEK_SET) == -1) {
             fprintf(stderr, "mmap_morecore: error in lseek (%d)\n", errno);
             return (result);
          }
          if (write (mdp -> fd, &buf, 1) == -1) {
             fprintf(stderr,
                     "mmap_morecore: error extending memory mapped file (%d)\n",
                     errno);
             return (result);
          }
          if (!mdp->base) {
            mapto = mmap (0, mapbytes, PROT_READ | PROT_WRITE,
                          MAP_SHARED, mdp -> fd, foffset);
#else
          if (!mdp->base) {
              hMap = CreateFileMapping(mdp -> fd, NULL, PAGE_READWRITE,
                  0, mapbytes, NULL);
              mapto = (char *) -1;
              if (hMap != NULL)
              {
                  mapto = MapViewOfFileEx(hMap, FILE_MAP_READ | FILE_MAP_WRITE,
                                          0, foffset,0, (LPVOID)0);
//                                          0, foffset,0, (LPVOID)0x1e70000);
                  if (!mapto) mapto = (char *)-1;
              }
#endif
            if (mapto != (char *)-1) {
              mdp->base = mapto;
              mdp->top  = mapto + mapbytes;
              mdp->breakval = mapto + size;
              result = (PTR) mapto;
            }
          } else {
            /*fprintf(stderr, "mmap_morecore: try to extend mapping by %d bytes, use bigger TMapFile\n", mapbytes);*/
#ifndef WIN32
            mapto = mmap (mdp -> top, mapbytes, PROT_READ | PROT_WRITE,
                          MAP_SHARED | MAP_FIXED, mdp -> fd, foffset);
#else
            hMap = CreateFileMapping(mdp -> fd, NULL, PAGE_READWRITE,
                          0, mapbytes, NULL);
            mapto = (char *) -1;
            if (hMap != NULL)
              mapto = MapViewOfFileEx(hMap, FILE_MAP_READ | FILE_MAP_WRITE, 0, foffset,0, mdp -> top);
#endif
            if (mapto == mdp -> top)
              {
                mdp -> top = moveto;
                result = (PTR) mdp -> breakval;
                mdp -> breakval += size;
              }
          }
        }
      else
        {
          result = (PTR) mdp -> breakval;
          mdp -> breakval += size;
        }
    }
  return (result);
}

PTR
__mmalloc_remap_core (mdp)
  struct mdesc *mdp;
{
  caddr_t base;
  int rdonly = 0;

#ifndef WIN32
  int val;
  if ((val = fcntl(mdp->fd, F_GETFL, 0)) < 0) {
     fprintf(stderr, "__mmalloc_remap_core: error calling fcntl(%d)\n", errno);
     return ((PTR)-1);
  }
  if ((val & O_ACCMODE) == O_RDONLY) rdonly = 1;
#else
  BY_HANDLE_FILE_INFORMATION FileInformation;
  if (!GetFileInformationByHandle(mdp->fd,&FileInformation))
  {
     fprintf(stderr, "__mmalloc_remap_core: error calling GetFileInformationByHandle(%d)\n",
                     GetLastError());
     return ((PTR)-1);
  }
  if (FileInformation.dwFileAttributes & FILE_ATTRIBUTE_READONLY) rdonly = 1;
  rdonly = 1;  // for NT always read-only for the time being
#endif

  if (rdonly) {
#ifndef WIN32
    base = mmap (mdp -> base, mdp -> top - mdp -> base,
                 PROT_READ, MAP_SHARED | MAP_FIXED,
                 mdp -> fd, 0);
    if (base == (char *)-1)
       base = mmap (0, mdp -> top - mdp -> base,
                    PROT_READ, MAP_SHARED, mdp -> fd, 0);
#else
    HANDLE hMap;
    hMap = CreateFileMapping(mdp -> fd, NULL, PAGE_READONLY,
                             0, mdp -> top - mdp -> base, NULL);

//    hMap = OpenFileMapping(FILE_MAP_READ, FALSE, (LPTSTR)mdp->magic);

    base = (char *)-1;
    if (hMap != NULL)
    {
        base = MapViewOfFileEx(hMap, FILE_MAP_READ, 0, 0, 0, 0);
        if (!base)
           fprintf(stderr, "__mmalloc_remap_core: can't get base address %p to map. Error code %d.\n",
                   mdp -> base,GetLastError());
    }
    else
    {
        fprintf(stderr, "__mmalloc_remap_core: can't map file. Error code %d.\n",GetLastError());
    }
#endif
    if (base != mdp->base) mdp->offset = base - mdp->base;
  } else {
#ifndef WIN32
    base = mmap (mdp -> base, mdp -> top - mdp -> base,
                 PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED,
                 mdp -> fd, 0);
#else
    HANDLE hMap;
    hMap = CreateFileMapping(mdp -> fd, NULL, PAGE_READWRITE,
        0, mdp -> top - mdp -> base, NULL);

//    hMap = OpenFileMapping(FILE_MAP_READ | FILE_MAP_WRITE, FALSE, (LPTSTR)mdp->magic);

    base = (char *)-1;
    if (hMap != NULL)
    {
       base = MapViewOfFileEx(hMap, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, 0, mdp -> base);
       if (!base)
           fprintf(stderr, "__mmalloc_remap_core: can't get base address %p to map. Error code %d.\n",
                   mdp -> base,GetLastError());
    }
    else {
        fprintf(stderr, "__mmalloc_remap_core: can't map file. Error code %d.\n",GetLastError());
    }
#endif
  }
  return ((PTR) base);
}

int
mmalloc_update_mapping(md)
  PTR md;
{
  /*
   * In case of a read-only mapping, we need to call this routine to
   * keep the mapping in sync with the mapping of the writer.
   */

  struct mdesc *mdp = (struct mdesc *)md;
  caddr_t oldtop, top, mapto;
  size_t  mapbytes;
  off_t   foffset;
  int     result;
#ifdef WIN32
    HANDLE hMap;
#endif

  oldtop = mdp->top;
  top    = ((struct mdesc *)mdp->base)->top;

  if (oldtop == top) return 0;

  if (top < oldtop) {

#ifndef WIN32
    munmap (top, (size_t) (oldtop - top));
#else
    UnmapViewOfFile((LPCVOID)top);
#endif
    result = 0;

  } else {

    mapbytes = top - oldtop;
    foffset = oldtop - mdp->base;
#ifndef WIN32
    mapto = mmap (oldtop, mapbytes, PROT_READ,
                  MAP_SHARED | MAP_FIXED, mdp -> fd, foffset);
#else
//    hMap = OpenFileMapping(FILE_MAP_READ, FALSE, (LPTSTR)mdp->magic);
    hMap = CreateFileMapping(mdp -> fd, NULL, PAGE_READWRITE,
        0, mapbytes, NULL);

    mapto = (char *) -1;
    if (hMap != NULL)
      mapto = MapViewOfFileEx(hMap, FILE_MAP_READ,(DWORD)0, (DWORD)foffset, 0, (LPVOID)oldtop);
#endif
    if (mapto == oldtop)
       result = 0;
    else
       result = -1;
  }
  mdp->top = top;

  return (result);
}

#else   /* defined(R__HAVE_MMAP) */

int
mmalloc_update_mapping(md)
  PTR md;
{
   return 0;
}

#endif  /* defined(R__HAVE_MMAP) */
