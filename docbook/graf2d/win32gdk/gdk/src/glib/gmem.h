/* GLIB - Library of useful routines for C programming
 * Copyright (C) 1995-1997  Peter Mattis, Spencer Kimball and Josh MacDonald
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

/*
 * Modified by the GLib Team and others 1997-2000.  See the AUTHORS
 * file for a list of people on the GLib Team.  See the ChangeLog
 * files for a list of changes.  These files are distributed with
 * GLib at ftp://ftp.gtk.org/pub/gtk/. 
 */

#ifndef __G_MEM_H__
#define __G_MEM_H__

#include <g_types.h>

/* optionally feature DMALLOC memory allocation debugger
 */
#ifdef USE_DMALLOC
#include "dmalloc.h"
#endif

G_BEGIN_DECLS

typedef struct _GAllocator      GAllocator;
typedef struct _GMemChunk       GMemChunk;

/* Provide macros for easily allocating memory. The macros
 *  will cast the allocated memory to the specified type
 *  in order to avoid compiler warnings. (Makes the code neater).
 */

#ifdef __DMALLOC_H__
#  define g_new(type, count)		(ALLOC (type, count))
#  define g_new0(type, count)		(CALLOC (type, count))
#  define g_renew(type, mem, count)	(REALLOC (mem, type, count))
#else /* __DMALLOC_H__ */
#  define g_new(type, count)	  \
      ((type *) g_malloc ((unsigned) sizeof (type) * (count)))
#  define g_new0(type, count)	  \
      ((type *) g_malloc0 ((unsigned) sizeof (type) * (count)))
#  define g_renew(type, mem, count)	  \
      ((type *) g_realloc (mem, (unsigned) sizeof (type) * (count)))
#endif /* __DMALLOC_H__ */

#define g_mem_chunk_create(type, pre_alloc, alloc_type)	( \
  g_mem_chunk_new (#type " mem chunks (" #pre_alloc ")", \
		   sizeof (type), \
		   sizeof (type) * (pre_alloc), \
		   (alloc_type)) \
)
#define g_chunk_new(type, chunk)	( \
  (type *) g_mem_chunk_alloc (chunk) \
)
#define g_chunk_new0(type, chunk)	( \
  (type *) g_mem_chunk_alloc0 (chunk) \
)
#define g_chunk_free(mem, mem_chunk)	G_STMT_START { \
  g_mem_chunk_free ((mem_chunk), (mem)); \
} G_STMT_END

/* Memory allocation and debugging
 */
#ifdef USE_DMALLOC

#define g_malloc(size)	     ((gpointer) MALLOC (size))
#define g_malloc0(size)	     ((gpointer) CALLOC (char, size))
#define g_realloc(mem,size)  ((gpointer) REALLOC (mem, char, size))
#define g_free(mem)	     FREE (mem)

#else /* !USE_DMALLOC */

gpointer g_malloc      (gulong	  size);
gpointer g_malloc0     (gulong	  size);
gpointer g_realloc     (gpointer  mem,
			gulong	  size);
void	 g_free	       (gpointer  mem);

#endif /* !USE_DMALLOC */

void	 g_mem_profile (void);
void	 g_mem_check   (gpointer  mem);

/* Generic allocators
 */
GAllocator* g_allocator_new   (const gchar  *name,
			       guint         n_preallocs);
void        g_allocator_free  (GAllocator   *allocato);

#define	G_ALLOCATOR_LIST	(1)
#define	G_ALLOCATOR_SLIST	(2)
#define	G_ALLOCATOR_NODE	(3)

/* "g_mem_chunk_new" creates a new memory chunk.
 * Memory chunks are used to allocate pieces of memory which are
 *  always the same size. Lists are a good example of such a data type.
 * The memory chunk allocates and frees blocks of memory as needed.
 *  Just be sure to call "g_mem_chunk_free" and not "g_free" on data
 *  allocated in a mem chunk. ("g_free" will most likely cause a seg
 *  fault...somewhere).
 *
 * Oh yeah, GMemChunk is an opaque data type. (You don't really
 *  want to know what's going on inside do you?)
 */

/* ALLOC_ONLY MemChunk's can only allocate memory. The free operation
 *  is interpreted as a no op. ALLOC_ONLY MemChunk's save 4 bytes per
 *  atom. (They are also useful for lists which use MemChunk to allocate
 *  memory but are also part of the MemChunk implementation).
 * ALLOC_AND_FREE MemChunk's can allocate and free memory.
 */

#define G_ALLOC_ONLY	  1
#define G_ALLOC_AND_FREE  2

GMemChunk* g_mem_chunk_new     (gchar	  *name,
				gint	   atom_size,
				gulong	   area_size,
				gint	   type);
void	   g_mem_chunk_destroy (GMemChunk *mem_chunk);
gpointer   g_mem_chunk_alloc   (GMemChunk *mem_chunk);
gpointer   g_mem_chunk_alloc0  (GMemChunk *mem_chunk);
void	   g_mem_chunk_free    (GMemChunk *mem_chunk,
				gpointer   mem);
void	   g_mem_chunk_clean   (GMemChunk *mem_chunk);
void	   g_mem_chunk_reset   (GMemChunk *mem_chunk);
void	   g_mem_chunk_print   (GMemChunk *mem_chunk);
void	   g_mem_chunk_info    (void);

/* Ah yes...we have a "g_blow_chunks" function.
 * "g_blow_chunks" simply compresses all the chunks. This operation
 *  consists of freeing every memory area that should be freed (but
 *  which we haven't gotten around to doing yet). And, no,
 *  "g_blow_chunks" doesn't follow the naming scheme, but it is a
 *  much better name than "g_mem_chunk_clean_all" or something
 *  similar.
 */
void g_blow_chunks (void);

G_END_DECLS

#endif /* __G_MEM_H__ */

