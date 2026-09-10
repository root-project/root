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

#include <stdlib.h>
#include <sys/types.h>

#include "gdkimage.h"
#include "gdkprivate.h"

GdkImage *gdk_image_ref(GdkImage * image)
{
   GdkImagePrivate *private = (GdkImagePrivate *) image;

   g_return_val_if_fail(image != NULL, NULL);

   private->ref_count++;

   return image;
}

void gdk_image_unref(GdkImage * image)
{
   GdkImagePrivate *private = (GdkImagePrivate *) image;

   g_return_if_fail(image != NULL);
   g_return_if_fail(private->ref_count > 0);

   private->ref_count--;
   if (private->ref_count == 0)
      private->klass->destroy(image);
}
