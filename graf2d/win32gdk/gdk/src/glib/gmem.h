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

#include <glib/gtypes.h>

G_BEGIN_DECLS

typedef struct _GAllocator GAllocator;
typedef struct _GMemChunk  GMemChunk;
typedef struct _GMemVTable GMemVTable;


#if GLIB_SIZEOF_VOID_P > GLIB_SIZEOF_LONG
#  define G_MEM_ALIGN	GLIB_SIZEOF_VOID_P
#else	/* GLIB_SIZEOF_VOID_P <= GLIB_SIZEOF_LONG */
#  define G_MEM_ALIGN	GLIB_SIZEOF_LONG
#endif	/* GLIB_SIZEOF_VOID_P <= GLIB_SIZEOF_LONG */


/* Memory allocation functions
 */
gpointer g_malloc         (gulong	 n_bytes);
gpointer g_malloc0        (gulong	 n_bytes);
gpointer g_realloc        (gpointer	 mem,
			   gulong	 n_bytes);
void	 g_free	          (gpointer	 mem);
gpointer g_try_malloc     (gulong	 n_bytes);
gpointer g_try_realloc    (gpointer	 mem,
			   gulong	 n_bytes);


/* Convenience memory allocators
 */
#define g_new(struct_type, n_structs)		\
    ((struct_type *) g_malloc (((gsize) sizeof (struct_type)) * ((gsize) (n_structs))))
#define g_new0(struct_type, n_structs)		\
    ((struct_type *) g_malloc0 (((gsize) sizeof (struct_type)) * ((gsize) (n_structs))))
#define g_renew(struct_type, mem, n_structs)	\
    ((struct_type *) g_realloc ((mem), ((gsize) sizeof (struct_type)) * ((gsize) (n_structs))))


/* Memory allocation virtualization for debugging purposes
 * g_mem_set_vtable() has to be the very first GLib function called
 * if being used
 */
struct _GMemVTable
{
  gpointer (*malloc)      (gsize    n_bytes);
  gpointer (*realloc)     (gpointer mem,
			   gsize    n_bytes);
  void     (*free)        (gpointer mem);
  /* optional */
  gpointer (*calloc)      (gsize    n_blocks,
			   gsize    n_block_bytes);
  gpointer (*try_malloc)  (gsize    n_bytes);
  gpointer (*try_realloc) (gpointer mem,
			   gsize    n_bytes);
};
void	 g_mem_set_vtable (GMemVTable	*vtable);
gboolean g_mem_is_system_malloc (void);

/* Memory profiler and checker, has to be enabled via g_mem_set_vtable()
 */
GLIB_VAR GMemVTable	*glib_mem_profiler_table;
void	g_mem_profile	(void);


/* Memchunk convenience functions
 */
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

GMemChunk* g_mem_chunk_new     (const gchar *name,
				gint         atom_size,
				gulong       area_size,
				gint         type);
void       g_mem_chunk_destroy (GMemChunk   *mem_chunk);
gpointer   g_mem_chunk_alloc   (GMemChunk   *mem_chunk);
gpointer   g_mem_chunk_alloc0  (GMemChunk   *mem_chunk);
void       g_mem_chunk_free    (GMemChunk   *mem_chunk,
				gpointer     mem);
void       g_mem_chunk_clean   (GMemChunk   *mem_chunk);
void       g_mem_chunk_reset   (GMemChunk   *mem_chunk);
void       g_mem_chunk_print   (GMemChunk   *mem_chunk);
void       g_mem_chunk_info    (void);

/* Ah yes...we have a "g_blow_chunks" function.
 * "g_blow_chunks" simply compresses all the chunks. This operation
 *  consists of freeing every memory area that should be freed (but
 *  which we haven't gotten around to doing yet). And, no,
 *  "g_blow_chunks" doesn't follow the naming scheme, but it is a
 *  much better name than "g_mem_chunk_clean_all" or something
 *  similar.
 */
void	   g_blow_chunks (void);


/* Generic allocators
 */
GAllocator* g_allocator_new   (const gchar  *name,
			       guint         n_preallocs);
void        g_allocator_free  (GAllocator   *allocator);

/* internal */
#define	G_ALLOCATOR_LIST	(1)
#define	G_ALLOCATOR_SLIST	(2)
#define	G_ALLOCATOR_NODE	(3)


G_END_DECLS

#endif /* __G_MEM_H__ */
