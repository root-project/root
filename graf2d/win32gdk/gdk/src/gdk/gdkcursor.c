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

#include "gdkcursor.h"
#include "gdkprivate.h"

GdkCursor *gdk_cursor_ref(GdkCursor * cursor)
{
   g_return_val_if_fail(cursor != NULL, NULL);
   g_return_val_if_fail(cursor->ref_count > 0, NULL);

   cursor->ref_count += 1;

   return cursor;
}

void gdk_cursor_unref(GdkCursor * cursor)
{
   g_return_if_fail(cursor != NULL);
   g_return_if_fail(cursor->ref_count > 0);

   cursor->ref_count -= 1;

   if (cursor->ref_count == 0)
      _gdk_cursor_destroy(cursor);
}
