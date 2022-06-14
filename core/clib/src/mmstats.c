/* @(#)root/clib:$Id$ */
/* Author: */

/* Access the statistics maintained by `mmalloc'.
   Copyright 1990, 1991, 1992 Free Software Foundation

   Written May 1989 by Mike Haertel.
   Modified Mar 1992 by Fred Fish.  (fnf@cygnus.com)

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
Boston, MA 02111-1307, USA.

   The author may be reached (Email) at the address mike@ai.mit.edu,
   or (US mail) as Mike Haertel c/o Free Software Foundation. */

#include "mmprivate.h"

/* FIXME:  See the comment in mmprivate.h where struct mmstats_t is defined.
   None of the internal mmalloc structures should be externally visible
   outside the library. */

struct mmstats_t
mmstats (md)
  PTR md;
{
  struct mmstats_t result;
  struct mdesc *mdp;

  mdp = MD_TO_MDP (md);
  result.bytes_total =
      (char *) mdp -> morecore (mdp, 0) - mdp -> heapbase;
  result.chunks_used = mdp -> heapstats.chunks_used;
  result.bytes_used = mdp -> heapstats.bytes_used;
  result.chunks_free = mdp -> heapstats.chunks_free;
  result.bytes_free = mdp -> heapstats.bytes_free;
  return (result);
}
