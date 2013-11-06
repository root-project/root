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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
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

/* 
 * MT safe
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdlib.h>
#include <string.h>
#include "glib.h"


/* notes on macros:
 * having DISABLE_MEM_POOLS defined, disables mem_chunks alltogether, their
 * allocations are performed through ordinary g_malloc/g_free.
 * having G_DISABLE_CHECKS defined disables use of glib_mem_profiler_table and
 * g_mem_profile().
 * REALLOC_0_WORKS is defined if g_realloc (NULL, x) works.
 * SANE_MALLOC_PROTOS is defined if the systems malloc() and friends functions
 * match the corresponding GLib prototypes, keep configure.in and gmem.h in sync here.
 * if ENABLE_GC_FRIENDLY is defined, freed memory should be 0-wiped.
 */

#define MEM_PROFILE_TABLE_SIZE 4096

#define MEM_AREA_SIZE 4L

#ifdef	G_DISABLE_CHECKS
#  define ENTER_MEM_CHUNK_ROUTINE()
#  define LEAVE_MEM_CHUNK_ROUTINE()
#  define IN_MEM_CHUNK_ROUTINE()	FALSE
#else	/* !G_DISABLE_CHECKS */
static GPrivate* mem_chunk_recursion = NULL;
#  define MEM_CHUNK_ROUTINE_COUNT()	GPOINTER_TO_UINT (g_private_get (mem_chunk_recursion))
#  define ENTER_MEM_CHUNK_ROUTINE()	g_private_set (mem_chunk_recursion, GUINT_TO_POINTER (MEM_CHUNK_ROUTINE_COUNT () + 1))
#  define LEAVE_MEM_CHUNK_ROUTINE()	g_private_set (mem_chunk_recursion, GUINT_TO_POINTER (MEM_CHUNK_ROUTINE_COUNT () - 1))
#endif	/* !G_DISABLE_CHECKS */

#ifndef	REALLOC_0_WORKS
static gpointer
standard_realloc (gpointer mem,
		  gsize    n_bytes)
{
  if (!mem)
    return malloc (n_bytes);
  else
    return realloc (mem, n_bytes);
}
#endif	/* !REALLOC_0_WORKS */

#ifdef SANE_MALLOC_PROTOS
#  define standard_malloc	malloc
#  ifdef REALLOC_0_WORKS
#    define standard_realloc	realloc
#  endif /* REALLOC_0_WORKS */
#  define standard_free		free
#  define standard_calloc	calloc
#  define standard_try_malloc	malloc
#  define standard_try_realloc	realloc
#else	/* !SANE_MALLOC_PROTOS */
static gpointer
standard_malloc (gsize n_bytes)
{
  return malloc (n_bytes);
}
#  ifdef REALLOC_0_WORKS
static gpointer
standard_realloc (gpointer mem,
		  gsize    n_bytes)
{
  return realloc (mem, n_bytes);
}
#  endif /* REALLOC_0_WORKS */
static void
standard_free (gpointer mem)
{
  free (mem);
}
static gpointer
standard_calloc (gsize n_blocks,
		 gsize n_bytes)
{
  return calloc (n_blocks, n_bytes);
}
#define	standard_try_malloc	standard_malloc
#define	standard_try_realloc	standard_realloc
#endif	/* !SANE_MALLOC_PROTOS */


/* --- variables --- */
static GMemVTable glib_mem_vtable = {
  standard_malloc,
  standard_realloc,
  standard_free,
  standard_calloc,
  standard_try_malloc,
  standard_try_realloc,
};


/* --- functions --- */
gpointer
g_malloc (gulong n_bytes)
{
  if (n_bytes)
    {
      gpointer mem;

      mem = glib_mem_vtable.malloc (n_bytes);
      if (mem)
	return mem;

      g_error ("%s: failed to allocate %lu bytes", G_STRLOC, n_bytes);
    }

  return NULL;
}

gpointer
g_malloc0 (gulong n_bytes)
{
  if (n_bytes)
    {
      gpointer mem;

      mem = glib_mem_vtable.calloc (1, n_bytes);
      if (mem)
	return mem;

      g_error ("%s: failed to allocate %lu bytes", G_STRLOC, n_bytes);
    }

  return NULL;
}

gpointer
g_realloc (gpointer mem,
	   gulong   n_bytes)
{
  if (n_bytes)
    {
      mem = glib_mem_vtable.realloc (mem, n_bytes);
      if (mem)
	return mem;

      g_error ("%s: failed to allocate %lu bytes", G_STRLOC, n_bytes);
    }

  if (mem)
    glib_mem_vtable.free (mem);

  return NULL;
}

void
g_free (gpointer mem)
{
  if (mem)
    glib_mem_vtable.free (mem);
}

gpointer
g_try_malloc (gulong n_bytes)
{
  if (n_bytes)
    return glib_mem_vtable.try_malloc (n_bytes);
  else
    return NULL;
}

gpointer
g_try_realloc (gpointer mem,
	       gulong   n_bytes)
{
  if (n_bytes)
    return glib_mem_vtable.try_realloc (mem, n_bytes);

  if (mem)
    glib_mem_vtable.free (mem);

  return NULL;
}

static gpointer
fallback_calloc (gsize n_blocks,
		 gsize n_block_bytes)
{
  gsize l = n_blocks * n_block_bytes;
  gpointer mem = glib_mem_vtable.malloc (l);

  if (mem)
    memset (mem, 0, l);

  return mem;
}

static gboolean vtable_set = FALSE;

/**
 * g_mem_is_system_malloc
 * 
 * Checks whether the allocator used by g_malloc() is the system's
 * malloc implementation. If it returns %TRUE memory allocated with
 * malloc() can be used interchangeable with memory allocated using
 * g_malloc(). This function is useful for avoiding an extra copy
 * of allocated memory returned by a non-GLib-based API.
 *
 * A different allocator can be set using g_mem_set_vtable().
 *
 * Return value: if %TRUE, malloc() and g_malloc() can be mixed.
 **/
gboolean
g_mem_is_system_malloc (void)
{
  return vtable_set;
}

void
g_mem_set_vtable (GMemVTable *vtable)
{
  if (!vtable_set)
    {
      vtable_set = TRUE;
      if (vtable->malloc && vtable->realloc && vtable->free)
	{
	  glib_mem_vtable.malloc = vtable->malloc;
	  glib_mem_vtable.realloc = vtable->realloc;
	  glib_mem_vtable.free = vtable->free;
	  glib_mem_vtable.calloc = vtable->calloc ? vtable->calloc : fallback_calloc;
	  glib_mem_vtable.try_malloc = vtable->try_malloc ? vtable->try_malloc : glib_mem_vtable.malloc;
	  glib_mem_vtable.try_realloc = vtable->try_realloc ? vtable->try_realloc : glib_mem_vtable.realloc;
	}
      else
	g_warning (G_STRLOC ": memory allocation vtable lacks one of malloc(), realloc() or free()");
    }
  else
    g_warning (G_STRLOC ": memory allocation vtable can only be set once at startup");
}


/* --- memory profiling and checking --- */
#ifdef	G_DISABLE_CHECKS
GMemVTable *glib_mem_profiler_table = &glib_mem_vtable;
void
g_mem_profile (void)
{
}
#else	/* !G_DISABLE_CHECKS */
typedef enum {
  PROFILER_FREE		= 0,
  PROFILER_ALLOC	= 1,
  PROFILER_RELOC	= 2,
  PROFILER_ZINIT	= 4
} ProfilerJob;
static guint *profile_data = NULL;
static gulong profile_allocs = 0;
static gulong profile_mc_allocs = 0;
static gulong profile_zinit = 0;
static gulong profile_frees = 0;
static gulong profile_mc_frees = 0;
static GMutex *g_profile_mutex = NULL;
#ifdef  G_ENABLE_DEBUG
static volatile gulong g_trap_free_size = 0;
static volatile gulong g_trap_realloc_size = 0;
static volatile gulong g_trap_malloc_size = 0;
#endif  /* G_ENABLE_DEBUG */

#define	PROFILE_TABLE(f1,f2,f3)   ( ( ((f3) << 2) | ((f2) << 1) | (f1) ) * (MEM_PROFILE_TABLE_SIZE + 1))

static void
profiler_log (ProfilerJob job,
	      gulong      n_bytes,
	      gboolean    success)
{
  g_mutex_lock (g_profile_mutex);
  if (!profile_data)
    {
      profile_data = standard_malloc ((MEM_PROFILE_TABLE_SIZE + 1) * 8 * sizeof (profile_data[0]));
      if (!profile_data)	/* memory system kiddin' me, eh? */
	{
	  g_mutex_unlock (g_profile_mutex);
	  return;
	}
    }

  if (MEM_CHUNK_ROUTINE_COUNT () == 0)
    {
      if (n_bytes < MEM_PROFILE_TABLE_SIZE)
	profile_data[n_bytes + PROFILE_TABLE ((job & PROFILER_ALLOC) != 0,
					      (job & PROFILER_RELOC) != 0,
					      success != 0)] += 1;
      else
	profile_data[MEM_PROFILE_TABLE_SIZE + PROFILE_TABLE ((job & PROFILER_ALLOC) != 0,
							     (job & PROFILER_RELOC) != 0,
							     success != 0)] += 1;
      if (success)
	{
	  if (job & PROFILER_ALLOC)
	    {
	      profile_allocs += n_bytes;
	      if (job & PROFILER_ZINIT)
		profile_zinit += n_bytes;
	    }
	  else
	    profile_frees += n_bytes;
	}
    }
  else if (success)
    {
      if (job & PROFILER_ALLOC)
	profile_mc_allocs += n_bytes;
      else
	profile_mc_frees += n_bytes;
    }
  g_mutex_unlock (g_profile_mutex);
}

static void
profile_print_locked (guint   *local_data,
		      gboolean success)
{
  gboolean need_header = TRUE;
  guint i;

  for (i = 0; i <= MEM_PROFILE_TABLE_SIZE; i++)
    {
      glong t_malloc = local_data[i + PROFILE_TABLE (1, 0, success)];
      glong t_realloc = local_data[i + PROFILE_TABLE (1, 1, success)];
      glong t_free = local_data[i + PROFILE_TABLE (0, 0, success)];
      glong t_refree = local_data[i + PROFILE_TABLE (0, 1, success)];
      
      if (!t_malloc && !t_realloc && !t_free && !t_refree)
	continue;
      else if (need_header)
	{
	  need_header = FALSE;
	  g_print (" blocks of | allocated  | freed      | allocated  | freed      | n_bytes   \n");
	  g_print ("  n_bytes  | n_times by | n_times by | n_times by | n_times by | remaining \n");
	  g_print ("           | malloc()   | free()     | realloc()  | realloc()  |           \n");
	  g_print ("===========|============|============|============|============|===========\n");
	}
      if (i < MEM_PROFILE_TABLE_SIZE)
	g_print ("%10u | %10ld | %10ld | %10ld | %10ld |%+11ld\n",
		 i, t_malloc, t_free, t_realloc, t_refree,
		 (t_malloc - t_free + t_realloc - t_refree) * i);
      else if (i >= MEM_PROFILE_TABLE_SIZE)
	g_print ("   >%6u | %10ld | %10ld | %10ld | %10ld |        ***\n",
		 i, t_malloc, t_free, t_realloc, t_refree);
    }
  if (need_header)
    g_print (" --- none ---\n");
}

void
g_mem_profile (void)
{
  guint local_data[(MEM_PROFILE_TABLE_SIZE + 1) * 8 * sizeof (profile_data[0])];
  gulong local_allocs;
  gulong local_zinit;
  gulong local_frees;
  gulong local_mc_allocs;
  gulong local_mc_frees;

  g_mutex_lock (g_profile_mutex);

  local_allocs = profile_allocs;
  local_zinit = profile_zinit;
  local_frees = profile_frees;
  local_mc_allocs = profile_mc_allocs;
  local_mc_frees = profile_mc_frees;

  if (!profile_data)
    {
      g_mutex_unlock (g_profile_mutex);
      return;
    }

  memcpy (local_data, profile_data, 
	  (MEM_PROFILE_TABLE_SIZE + 1) * 8 * sizeof (profile_data[0]));
  
  g_mutex_unlock (g_profile_mutex);

  g_print ("GLib Memory statistics (successful operations):\n");
  profile_print_locked (local_data, TRUE);
  g_print ("GLib Memory statistics (failing operations):\n");
  profile_print_locked (local_data, FALSE);
  g_print ("Total bytes: allocated=%lu, zero-initialized=%lu (%.2f%%), freed=%lu (%.2f%%), remaining=%lu\n",
	   local_allocs,
	   local_zinit,
	   ((gdouble) local_zinit) / local_allocs * 100.0,
	   local_frees,
	   ((gdouble) local_frees) / local_allocs * 100.0,
	   local_allocs - local_frees);
  g_print ("MemChunk bytes: allocated=%lu, freed=%lu (%.2f%%), remaining=%lu\n",
	   local_mc_allocs,
	   local_mc_frees,
	   ((gdouble) local_mc_frees) / local_mc_allocs * 100.0,
	   local_mc_allocs - local_mc_frees);
}

static gpointer
profiler_try_malloc (gsize n_bytes)
{
  gulong *p;

#ifdef  G_ENABLE_DEBUG
  if (g_trap_malloc_size == n_bytes)
    G_BREAKPOINT ();
#endif  /* G_ENABLE_DEBUG */

  p = standard_malloc (sizeof (gulong) * 2 + n_bytes);

  if (p)
    {
      p[0] = 0;		/* free count */
      p[1] = n_bytes;	/* length */
      profiler_log (PROFILER_ALLOC, n_bytes, TRUE);
      p += 2;
    }
  else
    profiler_log (PROFILER_ALLOC, n_bytes, FALSE);
  
  return p;
}

static gpointer
profiler_malloc (gsize n_bytes)
{
  gpointer mem = profiler_try_malloc (n_bytes);

  if (!mem)
    g_mem_profile ();

  return mem;
}

static gpointer
profiler_calloc (gsize n_blocks,
		 gsize n_block_bytes)
{
  gsize l = n_blocks * n_block_bytes;
  gulong *p;

#ifdef  G_ENABLE_DEBUG
  if (g_trap_malloc_size == l)
    G_BREAKPOINT ();
#endif  /* G_ENABLE_DEBUG */
  
  p = standard_calloc (1, sizeof (gulong) * 2 + l);

  if (p)
    {
      p[0] = 0;		/* free count */
      p[1] = l;		/* length */
      profiler_log (PROFILER_ALLOC | PROFILER_ZINIT, l, TRUE);
      p += 2;
    }
  else
    {
      profiler_log (PROFILER_ALLOC | PROFILER_ZINIT, l, FALSE);
      g_mem_profile ();
    }

  return p;
}

static void
profiler_free (gpointer mem)
{
  gulong *p = mem;

  p -= 2;
  if (p[0])	/* free count */
    {
      g_warning ("free(%p): memory has been freed %lu times already", p + 2, p[0]);
      profiler_log (PROFILER_FREE,
		    p[1],	/* length */
		    FALSE);
    }
  else
    {
#ifdef  G_ENABLE_DEBUG
      if (g_trap_free_size == p[1])
	G_BREAKPOINT ();
#endif  /* G_ENABLE_DEBUG */

      profiler_log (PROFILER_FREE,
		    p[1],	/* length */
		    TRUE);
      memset (p + 2, 0xaa, p[1]);

      /* for all those that miss standard_free (p); in this place, yes,
       * we do leak all memory when profiling, and that is intentional
       * to catch double frees. patch submissions are futile.
       */
    }
  p[0] += 1;
}

static gpointer
profiler_try_realloc (gpointer mem,
		      gsize    n_bytes)
{
  gulong *p = mem;

  p -= 2;

#ifdef  G_ENABLE_DEBUG
  if (g_trap_realloc_size == n_bytes)
    G_BREAKPOINT ();
#endif  /* G_ENABLE_DEBUG */
  
  if (mem && p[0])	/* free count */
    {
      g_warning ("realloc(%p, %u): memory has been freed %lu times already", p + 2, n_bytes, p[0]);
      profiler_log (PROFILER_ALLOC | PROFILER_RELOC, n_bytes, FALSE);

      return NULL;
    }
  else
    {
      p = standard_realloc (mem ? p : NULL, sizeof (gulong) * 2 + n_bytes);

      if (p)
	{
	  if (mem)
	    profiler_log (PROFILER_FREE | PROFILER_RELOC, p[1], TRUE);
	  p[0] = 0;
	  p[1] = n_bytes;
	  profiler_log (PROFILER_ALLOC | PROFILER_RELOC, p[1], TRUE);
	  p += 2;
	}
      else
	profiler_log (PROFILER_ALLOC | PROFILER_RELOC, n_bytes, FALSE);

      return p;
    }
}

static gpointer
profiler_realloc (gpointer mem,
		  gsize    n_bytes)
{
  mem = profiler_try_realloc (mem, n_bytes);

  if (!mem)
    g_mem_profile ();

  return mem;
}

static GMemVTable profiler_table = {
  profiler_malloc,
  profiler_realloc,
  profiler_free,
  profiler_calloc,
  profiler_try_malloc,
  profiler_try_realloc,
};
GMemVTable *glib_mem_profiler_table = &profiler_table;

#endif	/* !G_DISABLE_CHECKS */


/* --- MemChunks --- */
typedef struct _GFreeAtom      GFreeAtom;
typedef struct _GMemArea       GMemArea;

struct _GFreeAtom
{
  GFreeAtom *next;
};

struct _GMemArea
{
  GMemArea *next;            /* the next mem area */
  GMemArea *prev;            /* the previous mem area */
  gulong index;              /* the current index into the "mem" array */
  gulong free;               /* the number of free bytes in this mem area */
  gulong allocated;          /* the number of atoms allocated from this area */
  gulong mark;               /* is this mem area marked for deletion */
  gchar mem[MEM_AREA_SIZE];  /* the mem array from which atoms get allocated
			      * the actual size of this array is determined by
			      *  the mem chunk "area_size". ANSI says that it
			      *  must be declared to be the maximum size it
			      *  can possibly be (even though the actual size
			      *  may be less).
			      */
};

struct _GMemChunk
{
  const gchar *name;         /* name of this MemChunk...used for debugging output */
  gint type;                 /* the type of MemChunk: ALLOC_ONLY or ALLOC_AND_FREE */
  gint num_mem_areas;        /* the number of memory areas */
  gint num_marked_areas;     /* the number of areas marked for deletion */
  guint atom_size;           /* the size of an atom */
  gulong area_size;          /* the size of a memory area */
  GMemArea *mem_area;        /* the current memory area */
  GMemArea *mem_areas;       /* a list of all the mem areas owned by this chunk */
  GMemArea *free_mem_area;   /* the free area...which is about to be destroyed */
  GFreeAtom *free_atoms;     /* the free atoms list */
  GTree *mem_tree;           /* tree of mem areas sorted by memory address */
  GMemChunk *next;           /* pointer to the next chunk */
  GMemChunk *prev;           /* pointer to the previous chunk */
};


#ifndef DISABLE_MEM_POOLS
static gulong g_mem_chunk_compute_size (gulong    size,
					gulong    min_size) G_GNUC_CONST;
static gint   g_mem_chunk_area_compare (GMemArea *a,
					GMemArea *b);
static gint   g_mem_chunk_area_search  (GMemArea *a,
					gchar    *addr);

/* here we can't use StaticMutexes, as they depend upon a working
 * g_malloc, the same holds true for StaticPrivate
 */
static GMutex        *mem_chunks_lock = NULL;
static GMemChunk     *mem_chunks = NULL;

GMemChunk*
g_mem_chunk_new (const gchar  *name,
		 gint          atom_size,
		 gulong        area_size,
		 gint          type)
{
  GMemChunk *mem_chunk;
  gulong rarea_size;

  g_return_val_if_fail (atom_size > 0, NULL);
  g_return_val_if_fail (area_size >= atom_size, NULL);

  ENTER_MEM_CHUNK_ROUTINE ();

  area_size = (area_size + atom_size - 1) / atom_size;
  area_size *= atom_size;

  mem_chunk = g_new (GMemChunk, 1);
  mem_chunk->name = name;
  mem_chunk->type = type;
  mem_chunk->num_mem_areas = 0;
  mem_chunk->num_marked_areas = 0;
  mem_chunk->mem_area = NULL;
  mem_chunk->free_mem_area = NULL;
  mem_chunk->free_atoms = NULL;
  mem_chunk->mem_tree = NULL;
  mem_chunk->mem_areas = NULL;
  mem_chunk->atom_size = atom_size;
  
  if (mem_chunk->type == G_ALLOC_AND_FREE)
    mem_chunk->mem_tree = g_tree_new ((GCompareFunc) g_mem_chunk_area_compare);
  
  if (mem_chunk->atom_size % G_MEM_ALIGN)
    mem_chunk->atom_size += G_MEM_ALIGN - (mem_chunk->atom_size % G_MEM_ALIGN);

  rarea_size = area_size + sizeof (GMemArea) - MEM_AREA_SIZE;
  rarea_size = g_mem_chunk_compute_size (rarea_size, atom_size + sizeof (GMemArea) - MEM_AREA_SIZE);
  mem_chunk->area_size = rarea_size - (sizeof (GMemArea) - MEM_AREA_SIZE);

  g_mutex_lock (mem_chunks_lock);
  mem_chunk->next = mem_chunks;
  mem_chunk->prev = NULL;
  if (mem_chunks)
    mem_chunks->prev = mem_chunk;
  mem_chunks = mem_chunk;
  g_mutex_unlock (mem_chunks_lock);

  LEAVE_MEM_CHUNK_ROUTINE ();

  return mem_chunk;
}

void
g_mem_chunk_destroy (GMemChunk *mem_chunk)
{
  GMemArea *mem_areas;
  GMemArea *temp_area;
  
  g_return_if_fail (mem_chunk != NULL);

  ENTER_MEM_CHUNK_ROUTINE ();

  mem_areas = mem_chunk->mem_areas;
  while (mem_areas)
    {
      temp_area = mem_areas;
      mem_areas = mem_areas->next;
      g_free (temp_area);
    }
  
  if (mem_chunk->next)
    mem_chunk->next->prev = mem_chunk->prev;
  if (mem_chunk->prev)
    mem_chunk->prev->next = mem_chunk->next;
  
  g_mutex_lock (mem_chunks_lock);
  if (mem_chunk == mem_chunks)
    mem_chunks = mem_chunks->next;
  g_mutex_unlock (mem_chunks_lock);
  
  if (mem_chunk->type == G_ALLOC_AND_FREE)
    g_tree_destroy (mem_chunk->mem_tree);  

  g_free (mem_chunk);

  LEAVE_MEM_CHUNK_ROUTINE ();
}

gpointer
g_mem_chunk_alloc (GMemChunk *mem_chunk)
{
  GMemArea *temp_area;
  gpointer mem;

  ENTER_MEM_CHUNK_ROUTINE ();

  g_return_val_if_fail (mem_chunk != NULL, NULL);
  
  while (mem_chunk->free_atoms)
    {
      /* Get the first piece of memory on the "free_atoms" list.
       * We can go ahead and destroy the list node we used to keep
       *  track of it with and to update the "free_atoms" list to
       *  point to its next element.
       */
      mem = mem_chunk->free_atoms;
      mem_chunk->free_atoms = mem_chunk->free_atoms->next;
      
      /* Determine which area this piece of memory is allocated from */
      temp_area = g_tree_search (mem_chunk->mem_tree,
				 (GCompareFunc) g_mem_chunk_area_search,
				 mem);
      
      /* If the area has been marked, then it is being destroyed.
       *  (ie marked to be destroyed).
       * We check to see if all of the segments on the free list that
       *  reference this area have been removed. This occurs when
       *  the ammount of free memory is less than the allocatable size.
       * If the chunk should be freed, then we place it in the "free_mem_area".
       * This is so we make sure not to free the mem area here and then
       *  allocate it again a few lines down.
       * If we don't allocate a chunk a few lines down then the "free_mem_area"
       *  will be freed.
       * If there is already a "free_mem_area" then we'll just free this mem area.
       */
      if (temp_area->mark)
        {
          /* Update the "free" memory available in that area */
          temp_area->free += mem_chunk->atom_size;
	  
          if (temp_area->free == mem_chunk->area_size)
            {
              if (temp_area == mem_chunk->mem_area)
                mem_chunk->mem_area = NULL;
	      
              if (mem_chunk->free_mem_area)
                {
                  mem_chunk->num_mem_areas -= 1;
		  
                  if (temp_area->next)
                    temp_area->next->prev = temp_area->prev;
                  if (temp_area->prev)
                    temp_area->prev->next = temp_area->next;
                  if (temp_area == mem_chunk->mem_areas)
                    mem_chunk->mem_areas = mem_chunk->mem_areas->next;
		  
		  if (mem_chunk->type == G_ALLOC_AND_FREE)
		    g_tree_remove (mem_chunk->mem_tree, temp_area);
                  g_free (temp_area);
                }
              else
                mem_chunk->free_mem_area = temp_area;
	      
	      mem_chunk->num_marked_areas -= 1;
	    }
	}
      else
        {
          /* Update the number of allocated atoms count.
	   */
          temp_area->allocated += 1;
	  
          /* The area wasn't marked...return the memory
	   */
	  goto outa_here;
        }
    }
  
  /* If there isn't a current mem area or the current mem area is out of space
   *  then allocate a new mem area. We'll first check and see if we can use
   *  the "free_mem_area". Otherwise we'll just malloc the mem area.
   */
  if ((!mem_chunk->mem_area) ||
      ((mem_chunk->mem_area->index + mem_chunk->atom_size) > mem_chunk->area_size))
    {
      if (mem_chunk->free_mem_area)
        {
          mem_chunk->mem_area = mem_chunk->free_mem_area;
	  mem_chunk->free_mem_area = NULL;
        }
      else
        {
#ifdef ENABLE_GC_FRIENDLY
	  mem_chunk->mem_area = (GMemArea*) g_malloc0 (sizeof (GMemArea) -
						       MEM_AREA_SIZE +
						       mem_chunk->area_size); 
#else /* !ENABLE_GC_FRIENDLY */
	  mem_chunk->mem_area = (GMemArea*) g_malloc (sizeof (GMemArea) -
						      MEM_AREA_SIZE +
						      mem_chunk->area_size);
#endif /* ENABLE_GC_FRIENDLY */
	  
	  mem_chunk->num_mem_areas += 1;
	  mem_chunk->mem_area->next = mem_chunk->mem_areas;
	  mem_chunk->mem_area->prev = NULL;
	  
	  if (mem_chunk->mem_areas)
	    mem_chunk->mem_areas->prev = mem_chunk->mem_area;
	  mem_chunk->mem_areas = mem_chunk->mem_area;
	  
	  if (mem_chunk->type == G_ALLOC_AND_FREE)
	    g_tree_insert (mem_chunk->mem_tree, mem_chunk->mem_area, mem_chunk->mem_area);
        }
      
      mem_chunk->mem_area->index = 0;
      mem_chunk->mem_area->free = mem_chunk->area_size;
      mem_chunk->mem_area->allocated = 0;
      mem_chunk->mem_area->mark = 0;
    }
  
  /* Get the memory and modify the state variables appropriately.
   */
  mem = (gpointer) &mem_chunk->mem_area->mem[mem_chunk->mem_area->index];
  mem_chunk->mem_area->index += mem_chunk->atom_size;
  mem_chunk->mem_area->free -= mem_chunk->atom_size;
  mem_chunk->mem_area->allocated += 1;

outa_here:

  LEAVE_MEM_CHUNK_ROUTINE ();

  return mem;
}

gpointer
g_mem_chunk_alloc0 (GMemChunk *mem_chunk)
{
  gpointer mem;

  mem = g_mem_chunk_alloc (mem_chunk);
  if (mem)
    {
      memset (mem, 0, mem_chunk->atom_size);
    }

  return mem;
}

void
g_mem_chunk_free (GMemChunk *mem_chunk,
		  gpointer   mem)
{
  GMemArea *temp_area;
  GFreeAtom *free_atom;
  
  g_return_if_fail (mem_chunk != NULL);
  g_return_if_fail (mem != NULL);

  ENTER_MEM_CHUNK_ROUTINE ();

#ifdef ENABLE_GC_FRIENDLY
  memset (mem, 0, mem_chunk->atom_size);
#endif /* ENABLE_GC_FRIENDLY */

  /* Don't do anything if this is an ALLOC_ONLY chunk
   */
  if (mem_chunk->type == G_ALLOC_AND_FREE)
    {
      /* Place the memory on the "free_atoms" list
       */
      free_atom = (GFreeAtom*) mem;
      free_atom->next = mem_chunk->free_atoms;
      mem_chunk->free_atoms = free_atom;
      
      temp_area = g_tree_search (mem_chunk->mem_tree,
				 (GCompareFunc) g_mem_chunk_area_search,
				 mem);
      
      temp_area->allocated -= 1;
      
      if (temp_area->allocated == 0)
	{
	  temp_area->mark = 1;
	  mem_chunk->num_marked_areas += 1;
	}
    }

  LEAVE_MEM_CHUNK_ROUTINE ();
}

/* This doesn't free the free_area if there is one */
void
g_mem_chunk_clean (GMemChunk *mem_chunk)
{
  GMemArea *mem_area;
  GFreeAtom *prev_free_atom;
  GFreeAtom *temp_free_atom;
  gpointer mem;
  
  g_return_if_fail (mem_chunk != NULL);
  
  ENTER_MEM_CHUNK_ROUTINE ();

  if (mem_chunk->type == G_ALLOC_AND_FREE)
    {
      prev_free_atom = NULL;
      temp_free_atom = mem_chunk->free_atoms;
      
      while (temp_free_atom)
	{
	  mem = (gpointer) temp_free_atom;
	  
	  mem_area = g_tree_search (mem_chunk->mem_tree,
				    (GCompareFunc) g_mem_chunk_area_search,
				    mem);
	  
          /* If this mem area is marked for destruction then delete the
	   *  area and list node and decrement the free mem.
           */
	  if (mem_area->mark)
	    {
	      if (prev_free_atom)
		prev_free_atom->next = temp_free_atom->next;
	      else
		mem_chunk->free_atoms = temp_free_atom->next;
	      temp_free_atom = temp_free_atom->next;
	      
	      mem_area->free += mem_chunk->atom_size;
	      if (mem_area->free == mem_chunk->area_size)
		{
		  mem_chunk->num_mem_areas -= 1;
		  mem_chunk->num_marked_areas -= 1;
		  
		  if (mem_area->next)
		    mem_area->next->prev = mem_area->prev;
		  if (mem_area->prev)
		    mem_area->prev->next = mem_area->next;
		  if (mem_area == mem_chunk->mem_areas)
		    mem_chunk->mem_areas = mem_chunk->mem_areas->next;
		  if (mem_area == mem_chunk->mem_area)
		    mem_chunk->mem_area = NULL;
		  
		  if (mem_chunk->type == G_ALLOC_AND_FREE)
		    g_tree_remove (mem_chunk->mem_tree, mem_area);
		  g_free (mem_area);
		}
	    }
	  else
	    {
	      prev_free_atom = temp_free_atom;
	      temp_free_atom = temp_free_atom->next;
	    }
	}
    }
  LEAVE_MEM_CHUNK_ROUTINE ();
}

void
g_mem_chunk_reset (GMemChunk *mem_chunk)
{
  GMemArea *mem_areas;
  GMemArea *temp_area;
  
  g_return_if_fail (mem_chunk != NULL);
  
  ENTER_MEM_CHUNK_ROUTINE ();

  mem_areas = mem_chunk->mem_areas;
  mem_chunk->num_mem_areas = 0;
  mem_chunk->mem_areas = NULL;
  mem_chunk->mem_area = NULL;
  
  while (mem_areas)
    {
      temp_area = mem_areas;
      mem_areas = mem_areas->next;
      g_free (temp_area);
    }
  
  mem_chunk->free_atoms = NULL;
  
  if (mem_chunk->mem_tree)
    g_tree_destroy (mem_chunk->mem_tree);
  mem_chunk->mem_tree = g_tree_new ((GCompareFunc) g_mem_chunk_area_compare);

  LEAVE_MEM_CHUNK_ROUTINE ();
}

void
g_mem_chunk_print (GMemChunk *mem_chunk)
{
  GMemArea *mem_areas;
  gulong mem;
  
  g_return_if_fail (mem_chunk != NULL);
  
  mem_areas = mem_chunk->mem_areas;
  mem = 0;
  
  while (mem_areas)
    {
      mem += mem_chunk->area_size - mem_areas->free;
      mem_areas = mem_areas->next;
    }

  g_log (g_log_domain_glib, G_LOG_LEVEL_INFO,
	 "%s: %ld bytes using %d mem areas",
	 mem_chunk->name, mem, mem_chunk->num_mem_areas);
}

void
g_mem_chunk_info (void)
{
  GMemChunk *mem_chunk;
  gint count;
  
  count = 0;
  g_mutex_lock (mem_chunks_lock);
  mem_chunk = mem_chunks;
  while (mem_chunk)
    {
      count += 1;
      mem_chunk = mem_chunk->next;
    }
  g_mutex_unlock (mem_chunks_lock);
  
  g_log (g_log_domain_glib, G_LOG_LEVEL_INFO, "%d mem chunks", count);
  
  g_mutex_lock (mem_chunks_lock);
  mem_chunk = mem_chunks;
  g_mutex_unlock (mem_chunks_lock);

  while (mem_chunk)
    {
      g_mem_chunk_print ((GMemChunk*) mem_chunk);
      mem_chunk = mem_chunk->next;
    }  
}

void
g_blow_chunks (void)
{
  GMemChunk *mem_chunk;
  
  g_mutex_lock (mem_chunks_lock);
  mem_chunk = mem_chunks;
  g_mutex_unlock (mem_chunks_lock);
  while (mem_chunk)
    {
      g_mem_chunk_clean ((GMemChunk*) mem_chunk);
      mem_chunk = mem_chunk->next;
    }
}

static gulong
g_mem_chunk_compute_size (gulong size,
			  gulong min_size)
{
  gulong power_of_2;
  gulong lower, upper;
  
  power_of_2 = 16;
  while (power_of_2 < size)
    power_of_2 <<= 1;
  
  lower = power_of_2 >> 1;
  upper = power_of_2;
  
  if (size - lower < upper - size && lower >= min_size)
    return lower;
  else
    return upper;
}

static gint
g_mem_chunk_area_compare (GMemArea *a,
			  GMemArea *b)
{
  if (a->mem > b->mem)
    return 1;
  else if (a->mem < b->mem)
    return -1;
  return 0;
}

static gint
g_mem_chunk_area_search (GMemArea *a,
			 gchar    *addr)
{
  if (a->mem <= addr)
    {
      if (addr < &a->mem[a->index])
	return 0;
      return 1;
    }
  return -1;
}

#else /* DISABLE_MEM_POOLS */

typedef struct {
  guint alloc_size;           /* the size of an atom */
}  GMinimalMemChunk;

GMemChunk*
g_mem_chunk_new (const gchar  *name,
		 gint          atom_size,
		 gulong        area_size,
		 gint          type)
{
  GMinimalMemChunk *mem_chunk;

  g_return_val_if_fail (atom_size > 0, NULL);

  mem_chunk = g_new (GMinimalMemChunk, 1);
  mem_chunk->alloc_size = atom_size;

  return ((GMemChunk*) mem_chunk);
}

void
g_mem_chunk_destroy (GMemChunk *mem_chunk)
{
  g_return_if_fail (mem_chunk != NULL);
  
  g_free (mem_chunk);
}

gpointer
g_mem_chunk_alloc (GMemChunk *mem_chunk)
{
  GMinimalMemChunk *minimal = (GMinimalMemChunk *)mem_chunk;
  
  g_return_val_if_fail (mem_chunk != NULL, NULL);
  
  return g_malloc (minimal->alloc_size);
}

gpointer
g_mem_chunk_alloc0 (GMemChunk *mem_chunk)
{
  GMinimalMemChunk *minimal = (GMinimalMemChunk *)mem_chunk;
  
  g_return_val_if_fail (mem_chunk != NULL, NULL);
  
  return g_malloc0 (minimal->alloc_size);
}

void
g_mem_chunk_free (GMemChunk *mem_chunk,
		  gpointer   mem)
{
  g_return_if_fail (mem_chunk != NULL);
  
  g_free (mem);
}

void	g_mem_chunk_clean	(GMemChunk *mem_chunk)	{}
void	g_mem_chunk_reset	(GMemChunk *mem_chunk)	{}
void	g_mem_chunk_print	(GMemChunk *mem_chunk)	{}
void	g_mem_chunk_info	(void)			{}
void	g_blow_chunks		(void)			{}

#endif /* DISABLE_MEM_POOLS */


/* generic allocators
 */
struct _GAllocator /* from gmem.c */
{
  gchar		*name;
  guint16	 n_preallocs;
  guint		 is_unused : 1;
  guint		 type : 4;
  GAllocator	*last;
  GMemChunk	*mem_chunk;
  gpointer	 dummy; /* implementation specific */
};

GAllocator*
g_allocator_new (const gchar *name,
		 guint        n_preallocs)
{
  GAllocator *allocator;

  g_return_val_if_fail (name != NULL, NULL);

  allocator = g_new0 (GAllocator, 1);
  allocator->name = g_strdup (name);
  allocator->n_preallocs = CLAMP (n_preallocs, 1, 65535);
  allocator->is_unused = TRUE;
  allocator->type = 0;
  allocator->last = NULL;
  allocator->mem_chunk = NULL;
  allocator->dummy = NULL;

  return allocator;
}

void
g_allocator_free (GAllocator *allocator)
{
  g_return_if_fail (allocator != NULL);
  g_return_if_fail (allocator->is_unused == TRUE);

  g_free (allocator->name);
  if (allocator->mem_chunk)
    g_mem_chunk_destroy (allocator->mem_chunk);

  g_free (allocator);
}

void
g_mem_init (void)
{
#ifndef DISABLE_MEM_POOLS
  mem_chunks_lock = g_mutex_new ();
#endif
#ifndef G_DISABLE_CHECKS
  mem_chunk_recursion = g_private_new (NULL);
  g_profile_mutex = g_mutex_new ();
#endif
}
