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

#ifndef __G_TIMER_H__
#define __G_TIMER_H__

#include <g_types.h>

G_BEGIN_DECLS

/* Timer
 */

/* microseconds per second */
typedef struct _GTimer		GTimer;

#define G_USEC_PER_SEC 1000000

GTimer* g_timer_new	(void);
void	g_timer_destroy (GTimer	 *timer);
void	g_timer_start	(GTimer	 *timer);
void	g_timer_stop	(GTimer	 *timer);
void	g_timer_reset	(GTimer	 *timer);
gdouble g_timer_elapsed (GTimer	 *timer,
			 gulong	 *microseconds);
void    g_usleep        (gulong microseconds);

G_END_DECLS

#endif /* __G_TIMER_H__ */
