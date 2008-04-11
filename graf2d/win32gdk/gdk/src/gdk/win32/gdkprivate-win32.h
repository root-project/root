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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
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

#ifndef __GDK_PRIVATE_WIN32_H__
#define __GDK_PRIVATE_WIN32_H__

#ifndef STRICT
#define STRICT                  /* We want strict type checks */
#endif
#include <windows.h>
#include <commctrl.h>

/* Make up for mingw32 header lossage */

/* PS_JOIN_MASK is missing from the mingw32 headers */
#ifndef PS_JOIN_MASK
#define PS_JOIN_MASK (PS_JOIN_BEVEL|PS_JOIN_MITER|PS_JOIN_ROUND)
#endif

/* CLR_INVALID is missing */
#ifndef CLR_INVALID
#define CLR_INVALID CLR_NONE
#endif

/* Some charsets are missing */
#ifndef JOHAB_CHARSET
#define JOHAB_CHARSET 130
#endif
#ifndef VIETNAMESE_CHARSET
#define VIETNAMESE_CHARSET 163
#endif

#ifndef FS_VIETNAMESE
#define FS_VIETNAMESE 0x100
#endif

#ifndef VM_OEM_PLUS
#define VK_OEM_PLUS 0xBB
#endif
#ifndef VK_OEM_COMMA
#define VK_OEM_COMMA 0xBC
#endif
#ifndef VK_OEM_MINUS
#define VK_OEM_MINUS 0xBD
#endif
#ifndef VK_OEM_PERIOD
#define VK_OEM_PERIOD 0xBE
#endif

#ifndef VK_OEM_1
#define VK_OEM_1 0xBA
#endif
#ifndef VK_OEM_2
#define VK_OEM_2 0xBF
#endif
#ifndef VK_OEM_3
#define VK_OEM_3 0xC0
#endif
#ifndef VK_OEM_4
#define VK_OEM_4 0xDB
#endif
#ifndef VK_OEM_5
#define VK_OEM_5 0xDC
#endif
#ifndef VK_OEM_6
#define VK_OEM_6 0xDD
#endif
#ifndef VK_OEM_7
#define VK_OEM_7 0xDE
#endif
#ifndef VK_OEM_8
#define VK_OEM_8 0xDF
#endif

/* Missing messages */
#ifndef WM_MOUSEWHEEL
#define WM_MOUSEWHEEL 0X20A
#endif
#ifndef WM_GETOBJECT
#define WM_GETOBJECT 0x003D
#endif
#ifndef WM_NCXBUTTONDOWN
#define WM_NCXBUTTONDOWN 0x00AB
#endif
#ifndef WM_NCXBUTTONUP
#define WM_NCXBUTTONUP 0x00AC
#endif
#ifndef WM_NCXBUTTONDBLCLK
#define WM_NCXBUTTONDBLCLK 0x00AD
#endif
#ifndef WM_MENURBUTTONUP
#define WM_MENURBUTTONUP 0x0122
#endif
#ifndef WM_MENUDRAG
#define WM_MENUDRAG 0x0123
#endif
#ifndef WM_MENUGETOBJECT
#define WM_MENUGETOBJECT 0x0124
#endif
#ifndef WM_UNINITMENUPOPUP
#define WM_UNINITMENUPOPUP 0x0125
#endif
#ifndef WM_MENUCOMMAND
#define WM_MENUCOMMAND 0x0126
#endif
#ifndef WM_CHANGEUISTATE
#define WM_CHANGEUISTATE 0x0127
#endif
#ifndef WM_UPDATEUISTATE
#define WM_UPDATEUISTATE 0x0128
#endif
#ifndef WM_QUERYUISTATE
#define WM_QUERYUISTATE 0x0129
#endif
#ifndef WM_XBUTTONDOWN
#define WM_XBUTTONDOWN 0x020B
#endif
#ifndef WM_XBUTTONUP
#define WM_XBUTTONUP 0x020C
#endif
#ifndef WM_XBUTTONDBLCLK
#define WM_XBUTTONDBLCLK 0x020D
#endif
#ifndef WM_IME_REQUEST
#define WM_IME_REQUEST 0x0288
#endif
#ifndef WM_MOUSEHOVER
#define WM_MOUSEHOVER 0x02A1
#endif
#ifndef WM_MOUSELEAVE
#define WM_MOUSELEAVE 0x02A3
#endif
#ifndef WM_NCMOUSEHOVER
#define WM_NCMOUSEHOVER 0x02A0
#endif
#ifndef WM_NCMOUSELEAVE
#define WM_NCMOUSELEAVE 0x02A2
#endif
#ifndef WM_APPCOMMAND
#define WM_APPCOMMAND 0x0319
#endif
#ifndef WM_HANDHELDFIRST
#define WM_HANDHELDFIRST 0x0358
#endif
#ifndef WM_HANDHELDLAST
#define WM_HANDHELDLAST 0x035F
#endif
#ifndef WM_AFXFIRST
#define WM_AFXFIRST 0x0360
#endif
#ifndef WM_AFXLAST
#define WM_AFXLAST 0x037F
#endif

#ifndef CopyCursor
#define CopyCursor(pcur) ((HCURSOR)CopyIcon((HICON)(pcur)))
#endif

#include <time.h>

#include <gdk/gdktypes.h>
#include <gdk/gdkprivate.h>

#include <gdk/gdkcursor.h>
#include <gdk/gdkevents.h>
#include <gdk/gdkfont.h>
#include <gdk/gdkgc.h>
#include <gdk/gdkim.h>
#include <gdk/gdkimage.h>
#include <gdk/gdkvisual.h>
#include <gdk/gdkwindow.h>

#ifdef __cplusplus
extern "C" {
#endif                          /* __cplusplus */

/* Define corresponding Windows types for some X11 types, just for laziness.
 */

   typedef PALETTEENTRY XColor;
   typedef guint VisualID;
   typedef int Status;

/* Define some of the X11 constants also here, again just for laziness */

/* Generic null resource */
#define None 0

/* Error codes */
#define Success            0

/* Grabbing status */
#define GrabSuccess	   0
#define AlreadyGrabbed	   2

/* Some structs are somewhat useful to emulate internally, just to
   keep the code less #ifdefed.  */
   typedef struct {
      HPALETTE palette;         /* Palette handle used when drawing. */
      guint size;               /* Number of entries in the palette. */
      gboolean stale;           /* 1 if palette needs to be realized,
                                 * otherwise 0. */
      gboolean *in_use;
      gboolean rc_palette;      /* If RC_PALETTE is on in the RASTERCAPS */
      gulong sizepalette;       /* SIZEPALETTE if rc_palette */
   } ColormapStruct, *Colormap;

   typedef struct {
      gint map_entries;
      guint visualid;
      guint bitspixel;
   } Visual;

   typedef struct {
      Colormap colormap;
      unsigned long red_max;
      unsigned long red_mult;
      unsigned long green_max;
      unsigned long green_mult;
      unsigned long blue_max;
      unsigned long blue_mult;
      unsigned long base_pixel;
   } XStandardColormap;

   typedef struct _GdkGCWin32Data GdkGCWin32Data;
   typedef struct _GdkDrawableWin32Data GdkDrawableWin32Data;
   typedef struct _GdkWindowWin32Data GdkWindowWin32Data;
   typedef struct _GdkColormapPrivateWin32 GdkColormapPrivateWin32;
   typedef struct _GdkCursorPrivate GdkCursorPrivate;
   typedef struct _GdkWin32SingleFont GdkWin32SingleFont;
   typedef struct _GdkFontPrivateWin32 GdkFontPrivateWin32;
   typedef struct _GdkImagePrivateWin32 GdkImagePrivateWin32;
   typedef struct _GdkVisualPrivate GdkVisualPrivate;
   typedef struct _GdkRegionPrivate GdkRegionPrivate;
   typedef struct _GdkICPrivate GdkICPrivate;

#define GDK_DRAWABLE_WIN32DATA(win) ((GdkDrawableWin32Data *)(((GdkDrawablePrivate*)(win))->klass_data))
#define GDK_WINDOW_WIN32DATA(win) ((GdkWindowWin32Data *)(((GdkDrawablePrivate*)(win))->klass_data))
#define GDK_GC_WIN32DATA(gc) ((GdkGCWin32Data *)(((GdkGCPrivate*)(gc))->klass_data))

   struct _GdkGCWin32Data {
      /* A Windows Device Context (DC) is not equivalent to an X11
       * GC. We can use a DC only in the window for which it was
       * allocated, or (in the case of a memory DC) with the bitmap that
       * has been selected into it. Thus, we have to release and
       * reallocate a DC each time the GdkGC is used to paint into a new
       * window or pixmap. We thus keep all the necessary values in the
       * GdkGCWin32Data struct.
       */
      HDC xgc;
      GdkGCValuesMask values_mask;
      gulong foreground;        /* Pixel values from GdkColor, */
      gulong background;        /* not Win32 COLORREFs */
      GdkFont *font;
      gint rop2;
      GdkFill fill_style;
      GdkPixmap *tile;
      GdkPixmap *stipple;
      HRGN clip_region;
      GdkSubwindowMode subwindow_mode;
      gint ts_x_origin;
      gint ts_y_origin;
      gint clip_x_origin;
      gint clip_y_origin;
      gint graphics_exposures;
      gint pen_width;
      DWORD pen_style;
      HANDLE hwnd;              /* If a DC is allocated, for which window
                                 * or what bitmap is selected into it
                                 */
// bb add
      int luser_dash;           // length of array containing custom style bits
      int user_dash[32];        // array of custom style bits

      int saved_dc;
   };

   struct _GdkDrawableWin32Data {
      HANDLE xid;
   };

   struct _GdkWindowWin32Data {
      GdkDrawableWin32Data drawable;

      /* We must keep the event mask here to filter them ourselves */
      gint event_mask;

      /* Values for bg_type */
#define GDK_WIN32_BG_NORMAL 0
#define GDK_WIN32_BG_PIXEL 1
#define GDK_WIN32_BG_PIXMAP 2
#define GDK_WIN32_BG_PARENT_RELATIVE 3
#define GDK_WIN32_BG_TRANSPARENT 4

      /* We draw the background ourselves at WM_ERASEBKGND  */
      guchar bg_type;
      gulong bg_pixel;          /* GdkColor pixel, not COLORREF */
      GdkPixmap *bg_pixmap;

      HCURSOR xcursor;

      /* Window size hints */
      gint hint_flags;
      gint hint_x, hint_y;
      gint hint_min_width, hint_min_height;
      gint hint_max_width, hint_max_height;

      gboolean extension_events_selected;

      HKL input_locale;
      CHARSETINFO charset_info;

      gint grab_button;          //vo
      gint grab_event_mask;      //vo
      gint grab_owner_events;    //vo
      gint grab_modifiers;       //vo
      guint32 grab_time;         //vo
      GdkWindow *grab_confine;   //vo
      GdkCursor *grab_cursor;    //vo
      GList *grab_keys;          //vo
      gint grab_key_owner_events;//vo
   };

   struct _GdkCursorPrivate {
      GdkCursor cursor;
      HCURSOR xcursor;
   };

   struct _GdkWin32SingleFont {
      HFONT xfont;
      UINT charset;
      UINT codepage;
      FONTSIGNATURE fs;
   };

   struct _GdkFontPrivateWin32 {
      GdkFontPrivate base;
      GSList *fonts;            /* List of GdkWin32SingleFonts */
      GSList *names;
   };

   struct _GdkVisualPrivate {
      GdkVisual visual;
      Visual *xvisual;
   };

   struct _GdkColormapPrivateWin32 {
      GdkColormapPrivate base;
      Colormap xcolormap;
      gint private_val;

      GHashTable *hash;
      GdkColorInfo *info;
      time_t last_sync_time;
   };

   struct _GdkImagePrivateWin32 {
      GdkImagePrivate base;
      HBITMAP ximage;
   };

   struct _GdkRegionPrivate {
      GdkRegion region;
      HRGN xregion;
   };

   void gdk_win32_selection_init(void);
   void gdk_win32_dnd_exit(void);

   GdkColormap *gdk_colormap_lookup(Colormap xcolormap);
   GdkVisual *gdk_visual_lookup(Visual * xvisual);

   void gdk_xid_table_insert(HANDLE * hnd, gpointer data);
   void gdk_xid_table_remove(HANDLE xid);
   gpointer gdk_xid_table_lookup(HANDLE xid);

   GdkGC *_gdk_win32_gc_new(GdkDrawable * drawable,
                            GdkGCValues * values,
                            GdkGCValuesMask values_mask);
   COLORREF gdk_colormap_color(GdkColormapPrivateWin32 * colormap_private,
                               gulong pixel);
   HDC gdk_gc_predraw(GdkDrawable * drawable,
                      GdkGCPrivate * gc_private, GdkGCValuesMask usage);
   void gdk_gc_postdraw(GdkDrawable * drawable,
                        GdkGCPrivate * gc_private, GdkGCValuesMask usage);
   HRGN BitmapToRegion(HBITMAP hBmp);

   void gdk_sel_prop_store(GdkWindow * owner,
                           GdkAtom type,
                           gint format, guchar * data, gint length);

   gint gdk_nmbstowcs(GdkWChar * dest,
                      const gchar * src, gint src_len, gint dest_max);
   gint gdk_nmbstowchar_ts(wchar_t * dest,
                           const gchar * src, gint src_len, gint dest_max);

   void gdk_wchar_text_handle(GdkFont * font,
                              const wchar_t * wcstr,
                              int wclen,
                              void (*handler) (GdkWin32SingleFont *,
                                               const wchar_t *,
                                               int, void *), void *arg);

   void gdk_win32_api_failed(const gchar * where,
                             gint line, const gchar * api);
   void gdk_other_api_failed(const gchar * where,
                             gint line, const gchar * api);
   void gdk_win32_gdi_failed(const gchar * where,
                             gint line, const gchar * api);
#ifdef __GNUC__
#define WIN32_API_FAILED(api) gdk_win32_api_failed (__FILE__ ":" __PRETTY_FUNCTION__, __LINE__, api)
#define WIN32_GDI_FAILED(api) gdk_win32_gdi_failed (__FILE__ ":" __PRETTY_FUNCTION__, __LINE__, api)
#define OTHER_API_FAILED(api) gdk_other_api_failed (__FILE__ ":" __PRETTY_FUNCTION__, __LINE__, api)
#else
#define WIN32_API_FAILED(api) gdk_win32_api_failed (__FILE__, __LINE__, api)
#define WIN32_GDI_FAILED(api) gdk_win32_gdi_failed (__FILE__, __LINE__, api)
#define OTHER_API_FAILED(api) gdk_other_api_failed (__FILE__, __LINE__, api)
#endif

#ifdef G_ENABLE_DEBUG
   gchar *gdk_win32_color_to_string(const GdkColor * color);
   gchar *gdk_win32_cap_style_to_string(GdkCapStyle cap_style);
   gchar *gdk_win32_fill_style_to_string(GdkFill fill);
   gchar *gdk_win32_function_to_string(GdkFunction function);
   gchar *gdk_win32_join_style_to_string(GdkJoinStyle join_style);
   gchar *gdk_win32_line_style_to_string(GdkLineStyle line_style);
   gchar *gdk_win32_message_name(UINT msg);
#endif

   extern LRESULT CALLBACK gdk_WindowProc(HWND, UINT, WPARAM, LPARAM);

   extern GdkDrawableClass _gdk_win32_drawable_class;
   extern HWND gdk_root_window;
   GDKVAR ATOM gdk_selection_property;
   GDKVAR gchar *gdk_progclass;
   extern gboolean gdk_event_func_from_window_proc;

   extern HDC gdk_DC;
   extern HINSTANCE gdk_DLLInstance;
   extern HINSTANCE gdk_ProgInstance;

   extern UINT gdk_selection_notify_msg;
   extern UINT gdk_selection_request_msg;
   extern UINT gdk_selection_clear_msg;
   extern GdkAtom gdk_clipboard_atom;
   extern GdkAtom gdk_win32_dropfiles_atom;
   extern GdkAtom gdk_ole2_dnd_atom;

   extern DWORD windows_version;
#define IS_WIN_NT(dwVersion) (dwVersion < 0x80000000)

#ifdef __cplusplus
}
#endif                          /* __cplusplus */
#endif                          /* __GDK_PRIVATE_WIN32_H__ */
