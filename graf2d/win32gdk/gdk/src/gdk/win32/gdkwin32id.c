/* GDK - The GIMP Drawing Kit
 * Copyright (C) 1995-1997 Peter Mattis, Spencer Kimball and Josh MacDonald
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

/*
 * Modified by the GTK+ Team and others 1997-1999.  See the AUTHORS
 * file for a list of people on the GTK+ Team.  See the ChangeLog
 * files for a list of changes.  These files are distributed with
 * GTK+ at ftp://ftp.gtk.org/pub/gtk/. 
 */

#include "config.h"

#include <stdio.h>
#include <gdk/gdk.h>

#include "gdkprivate-win32.h"

static guint gdk_xid_hash(HANDLE * xid);
static gint gdk_xid_compare(HANDLE * a, HANDLE * b);


static GHashTable *xid_ht = NULL;


void gdk_xid_table_insert(HANDLE * xid, gpointer data)
{
   g_return_if_fail(xid != NULL);

   if (!xid_ht)
      xid_ht = g_hash_table_new((GHashFunc) gdk_xid_hash,
                                (GCompareFunc) gdk_xid_compare);

   g_hash_table_insert(xid_ht, xid, data);
}

void gdk_xid_table_remove(HANDLE xid)
{
   if (!xid_ht)
      xid_ht = g_hash_table_new((GHashFunc) gdk_xid_hash,
                                (GCompareFunc) gdk_xid_compare);

   g_hash_table_remove(xid_ht, &xid);
}

gpointer gdk_xid_table_lookup(HANDLE xid)
{
   gpointer data = NULL;

   if (xid_ht)
      data = g_hash_table_lookup(xid_ht, &xid);

   return data;
}


static guint gdk_xid_hash(HANDLE * xid)
{
   return (guint) * xid;
}

static gint gdk_xid_compare(HANDLE * a, HANDLE * b)
{
   return (*a == *b);
}
