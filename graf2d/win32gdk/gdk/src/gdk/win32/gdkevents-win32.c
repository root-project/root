/* GDK - The GIMP Drawing Kit
 * Copyright (C) 1995-1997 Peter Mattis, Spencer Kimball and Josh MacDonald
 * Copyright (C) 1998-1999 Tor Lillqvist
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
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

/* Cannot use TrackMouseEvent, as the stupid WM_MOUSELEAVE message
 * doesn't tell us where the mouse has gone. Thus we cannot use it to
 * generate a correct GdkNotifyType. Pity, as using TrackMouseEvent
 * otherwise would make it possible to reliably generate
 * GDK_LEAVE_NOTIFY events, which would help get rid of those pesky
 * tooltips sometimes popping up in the wrong place.
 */
/* define USE_TRACKMOUSEEVENT */

#include <stdio.h>

#include "gdk.h"
#include "gdkwin32.h"

#include "gdkkeysyms.h"

#include "gdkinputprivate.h"

#include <objbase.h>
#include <imm.h>

#ifdef HAVE_DIMM_H
#include <dimm.h>
#else
#include "surrogate-dimm.h"
#endif

#ifdef HAVE_WINTAB
#include "d:\development\wtkit\include\wintab.h"
#endif

#define PING() printf("%s: %d\n",__FILE__,__LINE__),fflush(stdout)

#define WINDOW_PRIVATE(wp) GDK_WINDOW_WIN32DATA (wp)

#define GET_X_LPARAM(lp)   ((int)(short)LOWORD(lp))
#define GET_Y_LPARAM(lp)   ((int)(short)HIWORD(lp))

typedef struct _GdkIOClosure GdkIOClosure;
typedef struct _GdkEventPrivate GdkEventPrivate;

typedef enum {
   /* Following flag is set for events on the event queue during
    * translation and cleared afterwards.
    */
   GDK_EVENT_PENDING = 1 << 0
} GdkEventFlags;

struct _GdkIOClosure {
   GdkInputFunction function;
   GdkInputCondition condition;
   GdkDestroyNotify notify;
   gpointer data;
};

struct _GdkEventPrivate {
   GdkEvent event;
   guint flags;
};

/*
 * Private function declarations
 */

static GdkFilterReturn
gdk_event_apply_filters(MSG * xevent, GdkEvent * event, GList * filters);
static gboolean gdk_event_translate(GdkEvent * event,
                                    MSG * xevent,
                                    gboolean * ret_val_flagp,
                                    gint * ret_valp);
static gboolean gdk_event_prepare(GSource * source, gint * timeout);
static gboolean gdk_event_check(GSource * source);
static gboolean gdk_event_dispatch(GSource * source,
                                   GSourceFunc callback,
                                   gpointer user_data);

static void gdk_synthesize_click(GdkEvent * event, gint nclicks);

/* Private variable declarations
 */

static GdkWindow *p_grab_window = NULL; /* Window that currently
                                         * holds the pointer grab
                                         */

static GdkWindow *k_grab_window = NULL; /* Window the holds the
                                         * keyboard grab
                                         */

static GList *client_filters;   /* Filters for client messages */

static gboolean p_grab_automatic;
static GdkEventMask p_grab_mask;
static gboolean p_grab_owner_events, k_grab_owner_events;
static HCURSOR p_grab_cursor;

static GList *client_filters;   /* Filters for client messages */

static GSourceFuncs event_funcs = {
   gdk_event_prepare,
   gdk_event_check,
   gdk_event_dispatch,
   (GDestroyNotify) g_free
};

GPollFD event_poll_fd;

static GdkWindow *curWnd = NULL;
static HWND active = NULL;
static gint curX, curY;
static gdouble curXroot, curYroot;
static UINT gdk_ping_msg;
static UINT msh_mousewheel_msg;
static gboolean ignore_WM_CHAR = FALSE;
static gboolean is_AltGr_key = FALSE;

static IActiveIMMApp *paimmapp = NULL;
static IActiveIMMMessagePumpOwner *paimmmpo = NULL;

typedef BOOL(WINAPI * PFN_TrackMouseEvent) (LPTRACKMOUSEEVENT);
static PFN_TrackMouseEvent p_TrackMouseEvent = NULL;

static gboolean use_IME_COMPOSITION = FALSE;

static gboolean first_move = FALSE; // bb add

static LRESULT
inner_window_proc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
   GdkEventPrivate event;
   GdkEvent *eventp;
   MSG msg;
   DWORD pos;
   LRESULT lres;
   gint ret_val;
   gboolean ret_val_flag;

   msg.hwnd = hWnd;
   msg.message = message;
   msg.wParam = wParam;
   msg.lParam = lParam;
   msg.time = GetTickCount();
   pos = GetMessagePos();
   msg.pt.x = GET_X_LPARAM(pos);
   msg.pt.y = GET_Y_LPARAM(pos);

   event.flags = GDK_EVENT_PENDING;
   if (gdk_event_translate(&event.event, &msg, &ret_val_flag, &ret_val)) {
      event.flags &= ~GDK_EVENT_PENDING;
#if 0
      if (event.event.any.type == GDK_CONFIGURE) {
         /* Compress configure events */
         GList *list = gdk_queued_events;

         while (list != NULL
                && (((GdkEvent *) list->data)->any.type != GDK_CONFIGURE
                    || ((GdkEvent *) list->data)->any.window !=
                    event.event.any.window))
            list = list->next;
         if (list != NULL) {
            GDK_NOTE(EVENTS,
                     g_print("... compressing an CONFIGURE event\n"));

            *((GdkEvent *) list->data) = event.event;
            gdk_window_unref(event.event.any.window);
            /* Wake up WaitMessage */
            PostMessage(NULL, gdk_ping_msg, 0, 0);
            return FALSE;
         }
      } else if (event.event.any.type == GDK_EXPOSE) {
         /* Compress expose events */
         GList *list = gdk_queued_events;

         while (list != NULL
                && (((GdkEvent *) list->data)->any.type != GDK_EXPOSE
                    || ((GdkEvent *) list->data)->any.window !=
                    event.event.any.window))
            list = list->next;
         if (list != NULL) {
            GdkRectangle u;

            GDK_NOTE(EVENTS, g_print("... compressing an EXPOSE event\n"));
            gdk_rectangle_union(&event.event.expose.area,
                                &((GdkEvent *) list->data)->expose.area,
                                &u);
            ((GdkEvent *) list->data)->expose.area = u;
            gdk_window_unref(event.event.any.window);
//#if 0
            /* Wake up WaitMessage */
            PostMessage(NULL, gdk_ping_msg, 0, 0);
//#endif
            return FALSE;
         }
      }
#endif
      eventp = gdk_event_new();
      *((GdkEventPrivate *) eventp) = event;

      /* Philippe Colantoni <colanton@aris.ss.uci.edu> suggests this
       * in order to handle events while opaque resizing neatly.  I
       * don't want it as default. Set the
       * GDK_EVENT_FUNC_FROM_WINDOW_PROC env var to get this
       * behaviour.
       */
      if (gdk_event_func_from_window_proc && gdk_event_func) {
         GDK_THREADS_ENTER();

         (*gdk_event_func) (eventp, gdk_event_data);
         gdk_event_free(eventp);

         GDK_THREADS_LEAVE();
      } else {
         gdk_event_queue_append(eventp);
#if 1
         /* Wake up WaitMessage */
         PostMessage(NULL, gdk_ping_msg, 0, 0);
#endif
      }

      if (ret_val_flag)
         return ret_val;
      else
         return FALSE;
   }

   if (ret_val_flag)
      return ret_val;
   else {
      if (paimmapp == NULL
          || (*paimmapp->lpVtbl->OnDefWindowProc) (paimmapp, hWnd, message,
                                                   wParam, lParam,
                                                   &lres) == S_FALSE)
         return DefWindowProc(hWnd, message, wParam, lParam);
      else
         return lres;
   }
}

LRESULT CALLBACK
gdk_WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
   LRESULT retval;
   gint tid = GetCurrentThreadId();

   GDK_NOTE(EVENTS, g_print("gdk_WindowProc: thread %#x hwnd %#x %s\n",
                            tid, hWnd, gdk_win32_message_name(message)));

   retval = inner_window_proc(hWnd, message, wParam, lParam);

   GDK_NOTE(EVENTS,
            g_print("gdk_WindowProc: thread %#x hwnd %#x returns %d\n",
                    tid, hWnd, retval));
   return retval;
}

void gdk_events_init(void)
{
   GSource *source;
   HRESULT hres;

   gdk_ping_msg = RegisterWindowMessage("gdk-ping");
   GDK_NOTE(EVENTS, g_print("gdk-ping = %#x\n", gdk_ping_msg));

   /* This is the string MSH_MOUSEWHEEL from zmouse.h,
    * http://www.microsoft.com/mouse/intellimouse/sdk/zmouse.h
    * This message is used by mouse drivers than cannot generate WM_MOUSEWHEEL
    * or on Win95.
    */
   msh_mousewheel_msg = RegisterWindowMessage("MSWHEEL_ROLLMSG");
   GDK_NOTE(EVENTS, g_print("MSH_MOUSEWHEEL = %#x\n", msh_mousewheel_msg));

   source = g_source_new(&event_funcs, sizeof(GSource));
   g_source_set_priority(source, GDK_PRIORITY_EVENTS);

   event_poll_fd.fd = G_WIN32_MSG_HANDLE;
   event_poll_fd.events = G_IO_IN;

   g_source_add_poll(source, &event_poll_fd);
   g_source_set_can_recurse(source, TRUE);
   g_source_attach(source, NULL);

   hres = CoCreateInstance(&CLSID_CActiveIMM,
                           NULL,
                           CLSCTX_ALL,
                           &IID_IActiveIMMApp, (LPVOID *) & paimmapp);

   if (hres == S_OK) {
      GDK_NOTE(EVENTS, g_print("IActiveIMMApp created %#x\n", paimmapp));
      (*paimmapp->lpVtbl->Activate) (paimmapp, TRUE);

      hres =
          (*paimmapp->lpVtbl->QueryInterface) (paimmapp,
                                               &IID_IActiveIMMMessagePumpOwner,
                                               &paimmmpo);
      GDK_NOTE(EVENTS,
               g_print("IActiveIMMMessagePumpOwner created %#x\n",
                       paimmmpo));
      (paimmmpo->lpVtbl->Start) (paimmmpo);
   }
#ifdef USE_TRACKMOUSEEVENT
   user32 = GetModuleHandle("user32.dll");
   if ((p_TrackMouseEvent =
        GetProcAddress(user32, "TrackMouseEvent")) == NULL) {
      if ((commctrl32 = LoadLibrary("commctrl32.dll")) != NULL)
         p_TrackMouseEvent = (PFN_TrackMouseEvent)
             GetProcAddress(commctrl32, "_TrackMouseEvent");
   }
   if (p_TrackMouseEvent != NULL)
      GDK_NOTE(EVENTS,
               g_print("Using TrackMouseEvent to detect leave events\n"));
#endif
   if (IS_WIN_NT(windows_version) && (windows_version & 0xFF) == 5) {
      /* On Win2k (Beta 3, at least) WM_IME_CHAR doesn't seem to work
       * correctly for non-Unicode applications. Handle
       * WM_IME_COMPOSITION with GCS_RESULTSTR instead, fetch the
       * Unicode char from the IME with ImmGetCompositionStringW().
       */
      use_IME_COMPOSITION = TRUE;
   }
}

/*
 *--------------------------------------------------------------
 * gdk_events_pending
 *
 *   Returns if events are pending on the queue.
 *
 * Arguments:
 *
 * Results:
 *   Returns TRUE if events are pending
 *
 * Side effects:
 *
 *--------------------------------------------------------------
 */

gboolean gdk_events_pending(void)
{
   MSG msg;

   return (gdk_event_queue_find_first()
           || PeekMessage(&msg, NULL, 0, 0, PM_NOREMOVE));
}

/*
 *--------------------------------------------------------------
 * gdk_event_get_graphics_expose
 *
 *   Waits for a GraphicsExpose or NoExpose event
 *
 * Arguments:
 *
 * Results:
 *   For GraphicsExpose events, returns a pointer to the event
 *   converted into a GdkEvent Otherwise, returns NULL.
 *
 * Side effects:
 *
 *-------------------------------------------------------------- */

GdkEvent *gdk_event_get_graphics_expose(GdkWindow * window)
{
   GdkWindowPrivate *private = (GdkWindowPrivate *) window;

   g_return_val_if_fail(window != NULL, NULL);

   GDK_NOTE(EVENTS, g_print("gdk_event_get_graphics_expose\n"));

#if 1
   /* Some nasty bugs here, just return NULL for now. */
   return NULL;
#else
   if (GetMessage(&xevent, private->xwindow, WM_PAINT, WM_PAINT)) {
      event = gdk_event_new();

      if (gdk_event_translate(event, &xevent, NULL, NULL))
         return event;
      else
         gdk_event_free(event);
   }

   return NULL;
#endif
}

static char *event_mask_string(GdkEventMask mask)
{
   static char bfr[500];
   char *p = bfr;

   *p = '\0';
#define BIT(x) \
  if (mask & GDK_##x##_MASK) \
    p += sprintf (p, "%s" #x, (p > bfr ? " " : ""))
   BIT(EXPOSURE);
   BIT(POINTER_MOTION);
   BIT(POINTER_MOTION_HINT);
   BIT(BUTTON_MOTION);
   BIT(BUTTON1_MOTION);
   BIT(BUTTON2_MOTION);
   BIT(BUTTON3_MOTION);
   BIT(BUTTON_PRESS);
   BIT(BUTTON_RELEASE);
   BIT(KEY_PRESS);
   BIT(KEY_RELEASE);
   BIT(ENTER_NOTIFY);
   BIT(LEAVE_NOTIFY);
   BIT(FOCUS_CHANGE);
   BIT(STRUCTURE);
   BIT(PROPERTY_CHANGE);
   BIT(VISIBILITY_NOTIFY);
   BIT(PROXIMITY_IN);
   BIT(PROXIMITY_OUT);
   BIT(SUBSTRUCTURE);
   BIT(SCROLL);
#undef BIT

   return bfr;
}

/*
 *--------------------------------------------------------------
 * gdk_pointer_grab
 *
 *   Grabs the pointer to a specific window
 *
 * Arguments:
 *   "window" is the window which will receive the grab
 *   "owner_events" specifies whether events will be reported as is,
 *     or relative to "window"
 *   "event_mask" masks only interesting events
 *   "confine_to" limits the cursor movement to the specified window
 *   "cursor" changes the cursor for the duration of the grab
 *   "time" specifies the time
 *
 * Results:
 *
 * Side effects:
 *   requires a corresponding call to gdk_pointer_ungrab
 *
 *--------------------------------------------------------------
 */

gint
gdk_pointer_grab(GdkWindow * window,
                 gint owner_events,
                 GdkEventMask event_mask,
                 GdkWindow * confine_to, GdkCursor * cursor, guint32 time)
{
   HWND xwindow;
   HWND xconfine_to;
   HCURSOR xcursor;
   GdkCursorPrivate *cursor_private;
   gint return_val;

   g_return_val_if_fail(window != NULL, 0);
   g_return_val_if_fail(GDK_IS_WINDOW(window), 0);
   g_return_val_if_fail(confine_to == NULL
                        || GDK_IS_WINDOW(confine_to), 0);

   cursor_private = (GdkCursorPrivate *) cursor;

   xwindow = GDK_DRAWABLE_XID(window);

   if (!confine_to || GDK_DRAWABLE_DESTROYED(confine_to))
      xconfine_to = NULL;
   else
      xconfine_to = GDK_DRAWABLE_XID(confine_to);

   if (!cursor)
      xcursor = NULL;
   else
      xcursor = cursor_private->xcursor;

      return_val = Success;

   if (return_val == Success) {
      if (!GDK_DRAWABLE_DESTROYED(window)) {
         GDK_NOTE(EVENTS, g_print("gdk_pointer_grab: %#x %s %#x %s\n",
                                  xwindow,
                                  (owner_events ? "TRUE" : "FALSE"),
                                  xcursor, event_mask_string(event_mask)));
         p_grab_mask = event_mask;
         p_grab_owner_events = (owner_events != 0);
         p_grab_automatic = FALSE;

         SetCapture(xwindow);
         return_val = GrabSuccess;
      } else
         return_val = AlreadyGrabbed;
   }

   if (return_val == GrabSuccess) {
      p_grab_window = window;
      if (p_grab_cursor != NULL) {
         if (GetCursor () == p_grab_cursor)
            SetCursor (NULL);
         DestroyCursor (p_grab_cursor);
      }
      p_grab_cursor = CopyCursor(xcursor);
      if(p_grab_cursor) {
         SetCursor(p_grab_cursor);
         ShowCursor(TRUE);
      }
   }

   return return_val;
}

/*
 *--------------------------------------------------------------
 * gdk_pointer_ungrab
 *
 *   Releases any pointer grab
 *
 * Arguments:
 *
 * Results:
 *
 * Side effects:
 *
 *--------------------------------------------------------------
 */

void gdk_pointer_ungrab(guint32 time)
{
   if (GetCapture() != NULL)
      ReleaseCapture();

   GDK_NOTE(EVENTS, g_print("gdk_pointer_ungrab\n"));

   p_grab_window = NULL;
   if (p_grab_cursor != NULL) {
      if (GetCursor () == p_grab_cursor)
            SetCursor (NULL);
      DestroyCursor (p_grab_cursor);
      p_grab_cursor = NULL;
   }

   p_grab_window = NULL;
}

/*
 *--------------------------------------------------------------
 * gdk_pointer_is_grabbed
 *
 *   Tell wether there is an active x pointer grab in effect
 *
 * Arguments:
 *
 * Results:
 *
 * Side effects:
 *
 *--------------------------------------------------------------
 */

gint gdk_pointer_is_grabbed(void)
{
   return p_grab_window != NULL;
}


//vo
gint gdk_button_grab(gint button, gint mod, GdkWindow * window,
                     gboolean owner_events,
                     GdkEventMask event_mask,
                     GdkWindow * confine_to,
                     GdkCursor * cursor)
{
   gint return_val;

   g_return_val_if_fail(window != NULL, 0);
   g_return_val_if_fail(GDK_IS_WINDOW(window), 0);
   g_return_val_if_fail(confine_to == NULL
                        || GDK_IS_WINDOW(confine_to), 0);

   return_val = GrabSuccess;

   GDK_WINDOW_WIN32DATA(window)->grab_event_mask = event_mask;
   GDK_WINDOW_WIN32DATA(window)->grab_button = button < 0 ? 0 : button;
   GDK_WINDOW_WIN32DATA(window)->grab_owner_events = owner_events;
   GDK_WINDOW_WIN32DATA(window)->grab_modifiers |= mod;
   GDK_WINDOW_WIN32DATA(window)->grab_confine = confine_to;
   GDK_WINDOW_WIN32DATA(window)->grab_cursor = cursor;

   return return_val;
}

//vo
void gdk_button_ungrab(gint button, gint mod, GdkWindow * window)
{
   if (window == NULL) return;
   if (!GDK_IS_WINDOW(window)) return;

   GDK_WINDOW_WIN32DATA(window)->grab_button = -1; // negative
   GDK_WINDOW_WIN32DATA(window)->grab_event_mask = 0;
}

//vo
typedef struct {
   gint key;
   gint mod;
} _Gdk_key_mod;

//vo
static _Gdk_key_mod *find_key_mod(GList *li, gint keycode, gint mod)
{
   GList *list;
   _Gdk_key_mod *result;

   if (li == NULL) return NULL;
   list = li;

   while (list) {
      result = (_Gdk_key_mod *)list->data;
      if (((result->key == keycode) || (keycode == 0)) &&
          ((result->mod == mod) ||
           (result->mod == GDK_MODIFIER_MASK) ||
           (mod == GDK_MODIFIER_MASK))) {
         return result;
      }
      list = list->next;
   }
   return NULL;
}

//vo
gint gdk_key_grab(gint keycode, gint mod, GdkWindow * window)
{
   gint return_val;
   _Gdk_key_mod *key_mod;

   g_return_val_if_fail(window != NULL, 0);
   g_return_val_if_fail(GDK_IS_WINDOW(window), 0);

   return_val = GrabSuccess;

   key_mod = find_key_mod(GDK_WINDOW_WIN32DATA(window)->grab_keys, keycode, mod);

   if (key_mod == NULL) {
      key_mod = g_new(_Gdk_key_mod, 1);
      key_mod->key = keycode;
      key_mod->mod = mod;
      GDK_WINDOW_WIN32DATA(window)->grab_keys =
         g_list_append(GDK_WINDOW_WIN32DATA(window)->grab_keys, key_mod);
   } else {
      return_val = AlreadyGrabbed;
   }

   return return_val;
}

//vo
void gdk_key_ungrab(gint keycode, gint mod, GdkWindow * window)
{
   _Gdk_key_mod *key_mod;

   if (window == NULL) return;
   if (!GDK_IS_WINDOW(window)) return;
   if (GDK_WINDOW_WIN32DATA(window)->grab_keys == NULL) return;

   if (keycode) {
      key_mod = find_key_mod(GDK_WINDOW_WIN32DATA(window)->grab_keys, keycode, mod);
      if (key_mod == NULL) return;

      GDK_WINDOW_WIN32DATA(window)->grab_keys =
         g_list_remove(GDK_WINDOW_WIN32DATA(window)->grab_keys, key_mod);
      g_free(key_mod);
   } else {
      while (1) {
         key_mod = find_key_mod(GDK_WINDOW_WIN32DATA(window)->grab_keys, 0, mod);
         if (key_mod) {
            GDK_WINDOW_WIN32DATA(window)->grab_keys =
               g_list_remove(GDK_WINDOW_WIN32DATA(window)->grab_keys, key_mod);
            g_free(key_mod);
         } else {
            break;
         }
      }
   }

   if (g_list_length(GDK_WINDOW_WIN32DATA(window)->grab_keys) == 0) {
      g_list_free(GDK_WINDOW_WIN32DATA(window)->grab_keys);
      GDK_WINDOW_WIN32DATA(window)->grab_keys = NULL;
      if (k_grab_window == window)
         k_grab_window = NULL;
   }
}

/*
 *--------------------------------------------------------------
 * find_window_for_pointer_event
 *
 *   Find the window a pointer event (mouse up, down, move) should
 *   be reported to.  If the return value != reported_window then
 *   the ref count of reported_window will be decremented and the
 *   ref count of the return value will be incremented.
 *
 * Arguments:
 *
 *  "reported_window" is the gdk window the xevent was reported relative to
 *  "xevent" is the win32 message
 *
 * Results:
 *
 * Side effects:
 *
 *--------------------------------------------------------------
 */

static GdkWindow*
find_window_for_pointer_event (GdkWindow*  reported_window,
                               MSG*        msg)
{
   HWND hwnd;
   POINTS points;
   POINT pt;
   GdkWindow* other_window;

   if (p_grab_window == NULL || !p_grab_owner_events)
      return reported_window;

   points = MAKEPOINTS (msg->lParam);
   pt.x = points.x;
   pt.y = points.y;
   ClientToScreen (msg->hwnd, &pt);

   GDK_NOTE (EVENTS, g_print ("Finding window for grabbed pointer event at (%ld, %ld)\n",
                             pt.x, pt.y));

   hwnd = WindowFromPoint (pt);
   if (hwnd == NULL) {
      return reported_window;
   }
   other_window = gdk_window_lookup(hwnd);

   if (other_window == NULL) {
      return reported_window;
   }
   GDK_NOTE (EVENTS, g_print ("Found window %p for point (%ld, %ld)\n",
                              hwnd, pt.x, pt.y));

   gdk_window_unref (reported_window);
   gdk_window_ref (other_window);

   return other_window;
}

static void
track_mouse_event(HWND  hwnd)
{
   typedef BOOL (WINAPI *PFN_TrackMouseEvent) (LPTRACKMOUSEEVENT);
   static PFN_TrackMouseEvent pTrackMouseEvent = NULL;
   static gboolean once = FALSE;

   if (!once) {
      HMODULE user32;
      HINSTANCE commctrl32;

      user32 = GetModuleHandle ("user32.dll");
      if ((pTrackMouseEvent = (PFN_TrackMouseEvent)GetProcAddress (user32, "TrackMouseEvent")) == NULL) {
         if ((commctrl32 = LoadLibrary ("commctrl32.dll")) != NULL)
            pTrackMouseEvent = (PFN_TrackMouseEvent) GetProcAddress (commctrl32, "_TrackMouseEvent");
      }
      once = TRUE;
   }

   if (pTrackMouseEvent) {
      TRACKMOUSEEVENT tme = {0};
      tme.cbSize = sizeof(TRACKMOUSEEVENT);
      tme.dwFlags = TME_LEAVE;
      tme.hwndTrack = hwnd;

      if (!pTrackMouseEvent (&tme))
         WIN32_API_FAILED ("TrackMouseEvent");
   }
}


/*
 *--------------------------------------------------------------
 * gdk_keyboard_grab
 *
 *   Grabs the keyboard to a specific window
 *
 * Arguments:
 *   "window" is the window which will receive the grab
 *   "owner_events" specifies whether events will be reported as is,
 *     or relative to "window"
 *   "time" specifies the time
 *
 * Results:
 *
 * Side effects:
 *   requires a corresponding call to gdk_keyboard_ungrab
 *
 *--------------------------------------------------------------
 */

gint gdk_keyboard_grab(GdkWindow * window, gint owner_events, guint32 time)
{
   gint return_val;

   g_return_val_if_fail(window != NULL, 0);
   g_return_val_if_fail(GDK_IS_WINDOW(window), 0);

   GDK_NOTE(EVENTS, g_print("gdk_keyboard_grab %#x\n",
                            GDK_DRAWABLE_XID(window)));

   if (!GDK_DRAWABLE_DESTROYED(window)) {
      k_grab_owner_events = owner_events != 0;
      return_val = GrabSuccess;
   } else
      return_val = AlreadyGrabbed;

   if (return_val == GrabSuccess) {
      k_grab_window = window;
      SetFocus((HWND)GDK_DRAWABLE_XID(k_grab_window));
   }
   return return_val;
}

/*
 *--------------------------------------------------------------
 * gdk_keyboard_ungrab
 *
 *   Releases any keyboard grab
 *
 * Arguments:
 *
 * Results:
 *
 * Side effects:
 *
 *--------------------------------------------------------------
 */

void gdk_keyboard_ungrab(guint32 time)
{
   GDK_NOTE(EVENTS, g_print("gdk_keyboard_ungrab\n"));

   k_grab_window = NULL;
}

static void gdk_io_destroy(gpointer data)
{
   GdkIOClosure *closure = data;

   if (closure->notify)
      closure->notify(closure->data);

   g_free(closure);
}

static gboolean
gdk_io_invoke(GIOChannel * source, GIOCondition condition, gpointer data)
{
   GdkIOClosure *closure = data;
   GdkInputCondition gdk_cond = 0;

   if (condition & (G_IO_IN | G_IO_PRI))
      gdk_cond |= GDK_INPUT_READ;
   if (condition & G_IO_OUT)
      gdk_cond |= GDK_INPUT_WRITE;
   if (condition & (G_IO_ERR | G_IO_HUP | G_IO_NVAL))
      gdk_cond |= GDK_INPUT_EXCEPTION;

   if (closure->condition & gdk_cond)
      closure->function(closure->data, g_io_channel_unix_get_fd(source),
                        gdk_cond);

   return TRUE;
}

static GdkFilterReturn
gdk_event_apply_filters(MSG * xevent, GdkEvent * event, GList * filters)
{
   GdkEventFilter *filter;
   GList *tmp_list;
   GdkFilterReturn result;

   tmp_list = filters;

   while (tmp_list) {
      filter = (GdkEventFilter *) tmp_list->data;

      result = (*filter->function) (xevent, event, filter->data);
      if (result != GDK_FILTER_CONTINUE)
         return result;

      tmp_list = tmp_list->next;
   }

   return GDK_FILTER_CONTINUE;
}

void
gdk_add_client_message_filter(GdkAtom message_type,
                              GdkFilterFunc func, gpointer data)
{
   GdkClientFilter *filter = g_new(GdkClientFilter, 1);

   filter->type = message_type;
   filter->function = func;
   filter->data = data;

   client_filters = g_list_prepend(client_filters, filter);
}

/* Thanks to Markus G. Kuhn <mkuhn@acm.org> for the ksysym<->Unicode
 * mapping functions, from the xterm sources.
 */

#if 0                           /* Keyval-to-Unicode isn't actually needed  */

struct k2u {
   unsigned short keysym;
   unsigned short ucs;
} k2utab[] = {
   {
   0x01a1, 0x0104},             /*                     Aogonek Ą LATIN CAPITAL LETTER A WITH OGONEK */
   {
   0x01a2, 0x02d8},             /*                       breve ˘ BREVE */
   {
   0x01a3, 0x0141},             /*                     Lstroke Ł LATIN CAPITAL LETTER L WITH STROKE */
   {
   0x01a5, 0x013d},             /*                      Lcaron Ľ LATIN CAPITAL LETTER L WITH CARON */
   {
   0x01a6, 0x015a},             /*                      Sacute Ś LATIN CAPITAL LETTER S WITH ACUTE */
   {
   0x01a9, 0x0160},             /*                      Scaron Š LATIN CAPITAL LETTER S WITH CARON */
   {
   0x01aa, 0x015e},             /*                    Scedilla Ş LATIN CAPITAL LETTER S WITH CEDILLA */
   {
   0x01ab, 0x0164},             /*                      Tcaron Ť LATIN CAPITAL LETTER T WITH CARON */
   {
   0x01ac, 0x0179},             /*                      Zacute Ź LATIN CAPITAL LETTER Z WITH ACUTE */
   {
   0x01ae, 0x017d},             /*                      Zcaron Ž LATIN CAPITAL LETTER Z WITH CARON */
   {
   0x01af, 0x017b},             /*                   Zabovedot Ż LATIN CAPITAL LETTER Z WITH DOT ABOVE */
   {
   0x01b1, 0x0105},             /*                     aogonek ą LATIN SMALL LETTER A WITH OGONEK */
   {
   0x01b2, 0x02db},             /*                      ogonek ˛ OGONEK */
   {
   0x01b3, 0x0142},             /*                     lstroke ł LATIN SMALL LETTER L WITH STROKE */
   {
   0x01b5, 0x013e},             /*                      lcaron ľ LATIN SMALL LETTER L WITH CARON */
   {
   0x01b6, 0x015b},             /*                      sacute ś LATIN SMALL LETTER S WITH ACUTE */
   {
   0x01b7, 0x02c7},             /*                       caron ˇ CARON */
   {
   0x01b9, 0x0161},             /*                      scaron š LATIN SMALL LETTER S WITH CARON */
   {
   0x01ba, 0x015f},             /*                    scedilla ş LATIN SMALL LETTER S WITH CEDILLA */
   {
   0x01bb, 0x0165},             /*                      tcaron ť LATIN SMALL LETTER T WITH CARON */
   {
   0x01bc, 0x017a},             /*                      zacute ź LATIN SMALL LETTER Z WITH ACUTE */
   {
   0x01bd, 0x02dd},             /*                 doubleacute ˝ DOUBLE ACUTE ACCENT */
   {
   0x01be, 0x017e},             /*                      zcaron ž LATIN SMALL LETTER Z WITH CARON */
   {
   0x01bf, 0x017c},             /*                   zabovedot ż LATIN SMALL LETTER Z WITH DOT ABOVE */
   {
   0x01c0, 0x0154},             /*                      Racute Ŕ LATIN CAPITAL LETTER R WITH ACUTE */
   {
   0x01c3, 0x0102},             /*                      Abreve Ă LATIN CAPITAL LETTER A WITH BREVE */
   {
   0x01c5, 0x0139},             /*                      Lacute Ĺ LATIN CAPITAL LETTER L WITH ACUTE */
   {
   0x01c6, 0x0106},             /*                      Cacute Ć LATIN CAPITAL LETTER C WITH ACUTE */
   {
   0x01c8, 0x010c},             /*                      Ccaron Č LATIN CAPITAL LETTER C WITH CARON */
   {
   0x01ca, 0x0118},             /*                     Eogonek Ę LATIN CAPITAL LETTER E WITH OGONEK */
   {
   0x01cc, 0x011a},             /*                      Ecaron Ě LATIN CAPITAL LETTER E WITH CARON */
   {
   0x01cf, 0x010e},             /*                      Dcaron Ď LATIN CAPITAL LETTER D WITH CARON */
   {
   0x01d0, 0x0110},             /*                     Dstroke Đ LATIN CAPITAL LETTER D WITH STROKE */
   {
   0x01d1, 0x0143},             /*                      Nacute Ń LATIN CAPITAL LETTER N WITH ACUTE */
   {
   0x01d2, 0x0147},             /*                      Ncaron Ň LATIN CAPITAL LETTER N WITH CARON */
   {
   0x01d5, 0x0150},             /*                Odoubleacute Ő LATIN CAPITAL LETTER O WITH DOUBLE ACUTE */
   {
   0x01d8, 0x0158},             /*                      Rcaron Ř LATIN CAPITAL LETTER R WITH CARON */
   {
   0x01d9, 0x016e},             /*                       Uring Ů LATIN CAPITAL LETTER U WITH RING ABOVE */
   {
   0x01db, 0x0170},             /*                Udoubleacute Ű LATIN CAPITAL LETTER U WITH DOUBLE ACUTE */
   {
   0x01de, 0x0162},             /*                    Tcedilla Ţ LATIN CAPITAL LETTER T WITH CEDILLA */
   {
   0x01e0, 0x0155},             /*                      racute ŕ LATIN SMALL LETTER R WITH ACUTE */
   {
   0x01e3, 0x0103},             /*                      abreve ă LATIN SMALL LETTER A WITH BREVE */
   {
   0x01e5, 0x013a},             /*                      lacute ĺ LATIN SMALL LETTER L WITH ACUTE */
   {
   0x01e6, 0x0107},             /*                      cacute ć LATIN SMALL LETTER C WITH ACUTE */
   {
   0x01e8, 0x010d},             /*                      ccaron č LATIN SMALL LETTER C WITH CARON */
   {
   0x01ea, 0x0119},             /*                     eogonek ę LATIN SMALL LETTER E WITH OGONEK */
   {
   0x01ec, 0x011b},             /*                      ecaron ě LATIN SMALL LETTER E WITH CARON */
   {
   0x01ef, 0x010f},             /*                      dcaron ď LATIN SMALL LETTER D WITH CARON */
   {
   0x01f0, 0x0111},             /*                     dstroke đ LATIN SMALL LETTER D WITH STROKE */
   {
   0x01f1, 0x0144},             /*                      nacute ń LATIN SMALL LETTER N WITH ACUTE */
   {
   0x01f2, 0x0148},             /*                      ncaron ň LATIN SMALL LETTER N WITH CARON */
   {
   0x01f5, 0x0151},             /*                odoubleacute ő LATIN SMALL LETTER O WITH DOUBLE ACUTE */
   {
   0x01f8, 0x0159},             /*                      rcaron ř LATIN SMALL LETTER R WITH CARON */
   {
   0x01f9, 0x016f},             /*                       uring ů LATIN SMALL LETTER U WITH RING ABOVE */
   {
   0x01fb, 0x0171},             /*                udoubleacute ű LATIN SMALL LETTER U WITH DOUBLE ACUTE */
   {
   0x01fe, 0x0163},             /*                    tcedilla ţ LATIN SMALL LETTER T WITH CEDILLA */
   {
   0x01ff, 0x02d9},             /*                    abovedot ˙ DOT ABOVE */
   {
   0x02a1, 0x0126},             /*                     Hstroke Ħ LATIN CAPITAL LETTER H WITH STROKE */
   {
   0x02a6, 0x0124},             /*                 Hcircumflex Ĥ LATIN CAPITAL LETTER H WITH CIRCUMFLEX */
   {
   0x02a9, 0x0130},             /*                   Iabovedot İ LATIN CAPITAL LETTER I WITH DOT ABOVE */
   {
   0x02ab, 0x011e},             /*                      Gbreve Ğ LATIN CAPITAL LETTER G WITH BREVE */
   {
   0x02ac, 0x0134},             /*                 Jcircumflex Ĵ LATIN CAPITAL LETTER J WITH CIRCUMFLEX */
   {
   0x02b1, 0x0127},             /*                     hstroke ħ LATIN SMALL LETTER H WITH STROKE */
   {
   0x02b6, 0x0125},             /*                 hcircumflex ĥ LATIN SMALL LETTER H WITH CIRCUMFLEX */
   {
   0x02b9, 0x0131},             /*                    idotless ı LATIN SMALL LETTER DOTLESS I */
   {
   0x02bb, 0x011f},             /*                      gbreve ğ LATIN SMALL LETTER G WITH BREVE */
   {
   0x02bc, 0x0135},             /*                 jcircumflex ĵ LATIN SMALL LETTER J WITH CIRCUMFLEX */
   {
   0x02c5, 0x010a},             /*                   Cabovedot Ċ LATIN CAPITAL LETTER C WITH DOT ABOVE */
   {
   0x02c6, 0x0108},             /*                 Ccircumflex Ĉ LATIN CAPITAL LETTER C WITH CIRCUMFLEX */
   {
   0x02d5, 0x0120},             /*                   Gabovedot Ġ LATIN CAPITAL LETTER G WITH DOT ABOVE */
   {
   0x02d8, 0x011c},             /*                 Gcircumflex Ĝ LATIN CAPITAL LETTER G WITH CIRCUMFLEX */
   {
   0x02dd, 0x016c},             /*                      Ubreve Ŭ LATIN CAPITAL LETTER U WITH BREVE */
   {
   0x02de, 0x015c},             /*                 Scircumflex Ŝ LATIN CAPITAL LETTER S WITH CIRCUMFLEX */
   {
   0x02e5, 0x010b},             /*                   cabovedot ċ LATIN SMALL LETTER C WITH DOT ABOVE */
   {
   0x02e6, 0x0109},             /*                 ccircumflex ĉ LATIN SMALL LETTER C WITH CIRCUMFLEX */
   {
   0x02f5, 0x0121},             /*                   gabovedot ġ LATIN SMALL LETTER G WITH DOT ABOVE */
   {
   0x02f8, 0x011d},             /*                 gcircumflex ĝ LATIN SMALL LETTER G WITH CIRCUMFLEX */
   {
   0x02fd, 0x016d},             /*                      ubreve ŭ LATIN SMALL LETTER U WITH BREVE */
   {
   0x02fe, 0x015d},             /*                 scircumflex ŝ LATIN SMALL LETTER S WITH CIRCUMFLEX */
   {
   0x03a2, 0x0138},             /*                         kra ĸ LATIN SMALL LETTER KRA */
   {
   0x03a3, 0x0156},             /*                    Rcedilla Ŗ LATIN CAPITAL LETTER R WITH CEDILLA */
   {
   0x03a5, 0x0128},             /*                      Itilde Ĩ LATIN CAPITAL LETTER I WITH TILDE */
   {
   0x03a6, 0x013b},             /*                    Lcedilla Ļ LATIN CAPITAL LETTER L WITH CEDILLA */
   {
   0x03aa, 0x0112},             /*                     Emacron Ē LATIN CAPITAL LETTER E WITH MACRON */
   {
   0x03ab, 0x0122},             /*                    Gcedilla Ģ LATIN CAPITAL LETTER G WITH CEDILLA */
   {
   0x03ac, 0x0166},             /*                      Tslash Ŧ LATIN CAPITAL LETTER T WITH STROKE */
   {
   0x03b3, 0x0157},             /*                    rcedilla ŗ LATIN SMALL LETTER R WITH CEDILLA */
   {
   0x03b5, 0x0129},             /*                      itilde ĩ LATIN SMALL LETTER I WITH TILDE */
   {
   0x03b6, 0x013c},             /*                    lcedilla ļ LATIN SMALL LETTER L WITH CEDILLA */
   {
   0x03ba, 0x0113},             /*                     emacron ē LATIN SMALL LETTER E WITH MACRON */
   {
   0x03bb, 0x0123},             /*                    gcedilla ģ LATIN SMALL LETTER G WITH CEDILLA */
   {
   0x03bc, 0x0167},             /*                      tslash ŧ LATIN SMALL LETTER T WITH STROKE */
   {
   0x03bd, 0x014a},             /*                         ENG Ŋ LATIN CAPITAL LETTER ENG */
   {
   0x03bf, 0x014b},             /*                         eng ŋ LATIN SMALL LETTER ENG */
   {
   0x03c0, 0x0100},             /*                     Amacron Ā LATIN CAPITAL LETTER A WITH MACRON */
   {
   0x03c7, 0x012e},             /*                     Iogonek Į LATIN CAPITAL LETTER I WITH OGONEK */
   {
   0x03cc, 0x0116},             /*                   Eabovedot Ė LATIN CAPITAL LETTER E WITH DOT ABOVE */
   {
   0x03cf, 0x012a},             /*                     Imacron Ī LATIN CAPITAL LETTER I WITH MACRON */
   {
   0x03d1, 0x0145},             /*                    Ncedilla Ņ LATIN CAPITAL LETTER N WITH CEDILLA */
   {
   0x03d2, 0x014c},             /*                     Omacron Ō LATIN CAPITAL LETTER O WITH MACRON */
   {
   0x03d3, 0x0136},             /*                    Kcedilla Ķ LATIN CAPITAL LETTER K WITH CEDILLA */
   {
   0x03d9, 0x0172},             /*                     Uogonek Ų LATIN CAPITAL LETTER U WITH OGONEK */
   {
   0x03dd, 0x0168},             /*                      Utilde Ũ LATIN CAPITAL LETTER U WITH TILDE */
   {
   0x03de, 0x016a},             /*                     Umacron Ū LATIN CAPITAL LETTER U WITH MACRON */
   {
   0x03e0, 0x0101},             /*                     amacron ā LATIN SMALL LETTER A WITH MACRON */
   {
   0x03e7, 0x012f},             /*                     iogonek į LATIN SMALL LETTER I WITH OGONEK */
   {
   0x03ec, 0x0117},             /*                   eabovedot ė LATIN SMALL LETTER E WITH DOT ABOVE */
   {
   0x03ef, 0x012b},             /*                     imacron ī LATIN SMALL LETTER I WITH MACRON */
   {
   0x03f1, 0x0146},             /*                    ncedilla ņ LATIN SMALL LETTER N WITH CEDILLA */
   {
   0x03f2, 0x014d},             /*                     omacron ō LATIN SMALL LETTER O WITH MACRON */
   {
   0x03f3, 0x0137},             /*                    kcedilla ķ LATIN SMALL LETTER K WITH CEDILLA */
   {
   0x03f9, 0x0173},             /*                     uogonek ų LATIN SMALL LETTER U WITH OGONEK */
   {
   0x03fd, 0x0169},             /*                      utilde ũ LATIN SMALL LETTER U WITH TILDE */
   {
   0x03fe, 0x016b},             /*                     umacron ū LATIN SMALL LETTER U WITH MACRON */
   {
   0x047e, 0x203e},             /*                    overline ‾ OVERLINE */
   {
   0x04a1, 0x3002},             /*               kana_fullstop 。 IDEOGRAPHIC FULL STOP */
   {
   0x04a2, 0x300c},             /*         kana_openingbracket 「 LEFT CORNER BRACKET */
   {
   0x04a3, 0x300d},             /*         kana_closingbracket 」 RIGHT CORNER BRACKET */
   {
   0x04a4, 0x3001},             /*                  kana_comma 、 IDEOGRAPHIC COMMA */
   {
   0x04a5, 0x30fb},             /*            kana_conjunctive ・ KATAKANA MIDDLE DOT */
   {
   0x04a6, 0x30f2},             /*                     kana_WO ヲ KATAKANA LETTER WO */
   {
   0x04a7, 0x30a1},             /*                      kana_a ァ KATAKANA LETTER SMALL A */
   {
   0x04a8, 0x30a3},             /*                      kana_i ィ KATAKANA LETTER SMALL I */
   {
   0x04a9, 0x30a5},             /*                      kana_u ゥ KATAKANA LETTER SMALL U */
   {
   0x04aa, 0x30a7},             /*                      kana_e ェ KATAKANA LETTER SMALL E */
   {
   0x04ab, 0x30a9},             /*                      kana_o ォ KATAKANA LETTER SMALL O */
   {
   0x04ac, 0x30e3},             /*                     kana_ya ャ KATAKANA LETTER SMALL YA */
   {
   0x04ad, 0x30e5},             /*                     kana_yu ュ KATAKANA LETTER SMALL YU */
   {
   0x04ae, 0x30e7},             /*                     kana_yo ョ KATAKANA LETTER SMALL YO */
   {
   0x04af, 0x30c3},             /*                    kana_tsu ッ KATAKANA LETTER SMALL TU */
   {
   0x04b0, 0x30fc},             /*              prolongedsound ー KATAKANA-HIRAGANA PROLONGED SOUND MARK */
   {
   0x04b1, 0x30a2},             /*                      kana_A ア KATAKANA LETTER A */
   {
   0x04b2, 0x30a4},             /*                      kana_I イ KATAKANA LETTER I */
   {
   0x04b3, 0x30a6},             /*                      kana_U ウ KATAKANA LETTER U */
   {
   0x04b4, 0x30a8},             /*                      kana_E エ KATAKANA LETTER E */
   {
   0x04b5, 0x30aa},             /*                      kana_O オ KATAKANA LETTER O */
   {
   0x04b6, 0x30ab},             /*                     kana_KA カ KATAKANA LETTER KA */
   {
   0x04b7, 0x30ad},             /*                     kana_KI キ KATAKANA LETTER KI */
   {
   0x04b8, 0x30af},             /*                     kana_KU ク KATAKANA LETTER KU */
   {
   0x04b9, 0x30b1},             /*                     kana_KE ケ KATAKANA LETTER KE */
   {
   0x04ba, 0x30b3},             /*                     kana_KO コ KATAKANA LETTER KO */
   {
   0x04bb, 0x30b5},             /*                     kana_SA サ KATAKANA LETTER SA */
   {
   0x04bc, 0x30b7},             /*                    kana_SHI シ KATAKANA LETTER SI */
   {
   0x04bd, 0x30b9},             /*                     kana_SU ス KATAKANA LETTER SU */
   {
   0x04be, 0x30bb},             /*                     kana_SE セ KATAKANA LETTER SE */
   {
   0x04bf, 0x30bd},             /*                     kana_SO ソ KATAKANA LETTER SO */
   {
   0x04c0, 0x30bf},             /*                     kana_TA タ KATAKANA LETTER TA */
   {
   0x04c1, 0x30c1},             /*                    kana_CHI チ KATAKANA LETTER TI */
   {
   0x04c2, 0x30c4},             /*                    kana_TSU ツ KATAKANA LETTER TU */
   {
   0x04c3, 0x30c6},             /*                     kana_TE テ KATAKANA LETTER TE */
   {
   0x04c4, 0x30c8},             /*                     kana_TO ト KATAKANA LETTER TO */
   {
   0x04c5, 0x30ca},             /*                     kana_NA ナ KATAKANA LETTER NA */
   {
   0x04c6, 0x30cb},             /*                     kana_NI ニ KATAKANA LETTER NI */
   {
   0x04c7, 0x30cc},             /*                     kana_NU ヌ KATAKANA LETTER NU */
   {
   0x04c8, 0x30cd},             /*                     kana_NE ネ KATAKANA LETTER NE */
   {
   0x04c9, 0x30ce},             /*                     kana_NO ノ KATAKANA LETTER NO */
   {
   0x04ca, 0x30cf},             /*                     kana_HA ハ KATAKANA LETTER HA */
   {
   0x04cb, 0x30d2},             /*                     kana_HI ヒ KATAKANA LETTER HI */
   {
   0x04cc, 0x30d5},             /*                     kana_FU フ KATAKANA LETTER HU */
   {
   0x04cd, 0x30d8},             /*                     kana_HE ヘ KATAKANA LETTER HE */
   {
   0x04ce, 0x30db},             /*                     kana_HO ホ KATAKANA LETTER HO */
   {
   0x04cf, 0x30de},             /*                     kana_MA マ KATAKANA LETTER MA */
   {
   0x04d0, 0x30df},             /*                     kana_MI ミ KATAKANA LETTER MI */
   {
   0x04d1, 0x30e0},             /*                     kana_MU ム KATAKANA LETTER MU */
   {
   0x04d2, 0x30e1},             /*                     kana_ME メ KATAKANA LETTER ME */
   {
   0x04d3, 0x30e2},             /*                     kana_MO モ KATAKANA LETTER MO */
   {
   0x04d4, 0x30e4},             /*                     kana_YA ヤ KATAKANA LETTER YA */
   {
   0x04d5, 0x30e6},             /*                     kana_YU ユ KATAKANA LETTER YU */
   {
   0x04d6, 0x30e8},             /*                     kana_YO ヨ KATAKANA LETTER YO */
   {
   0x04d7, 0x30e9},             /*                     kana_RA ラ KATAKANA LETTER RA */
   {
   0x04d8, 0x30ea},             /*                     kana_RI リ KATAKANA LETTER RI */
   {
   0x04d9, 0x30eb},             /*                     kana_RU ル KATAKANA LETTER RU */
   {
   0x04da, 0x30ec},             /*                     kana_RE レ KATAKANA LETTER RE */
   {
   0x04db, 0x30ed},             /*                     kana_RO ロ KATAKANA LETTER RO */
   {
   0x04dc, 0x30ef},             /*                     kana_WA ワ KATAKANA LETTER WA */
   {
   0x04dd, 0x30f3},             /*                      kana_N ン KATAKANA LETTER N */
   {
   0x04de, 0x309b},             /*                 voicedsound ゛ KATAKANA-HIRAGANA VOICED SOUND MARK */
   {
   0x04df, 0x309c},             /*             semivoicedsound ゜ KATAKANA-HIRAGANA SEMI-VOICED SOUND MARK */
   {
   0x05ac, 0x060c},             /*                Arabic_comma ، ARABIC COMMA */
   {
   0x05bb, 0x061b},             /*            Arabic_semicolon ؛ ARABIC SEMICOLON */
   {
   0x05bf, 0x061f},             /*        Arabic_question_mark ؟ ARABIC QUESTION MARK */
   {
   0x05c1, 0x0621},             /*                Arabic_hamza ء ARABIC LETTER HAMZA */
   {
   0x05c2, 0x0622},             /*          Arabic_maddaonalef آ ARABIC LETTER ALEF WITH MADDA ABOVE */
   {
   0x05c3, 0x0623},             /*          Arabic_hamzaonalef أ ARABIC LETTER ALEF WITH HAMZA ABOVE */
   {
   0x05c4, 0x0624},             /*           Arabic_hamzaonwaw ؤ ARABIC LETTER WAW WITH HAMZA ABOVE */
   {
   0x05c5, 0x0625},             /*       Arabic_hamzaunderalef إ ARABIC LETTER ALEF WITH HAMZA BELOW */
   {
   0x05c6, 0x0626},             /*           Arabic_hamzaonyeh ئ ARABIC LETTER YEH WITH HAMZA ABOVE */
   {
   0x05c7, 0x0627},             /*                 Arabic_alef ا ARABIC LETTER ALEF */
   {
   0x05c8, 0x0628},             /*                  Arabic_beh ب ARABIC LETTER BEH */
   {
   0x05c9, 0x0629},             /*           Arabic_tehmarbuta ة ARABIC LETTER TEH MARBUTA */
   {
   0x05ca, 0x062a},             /*                  Arabic_teh ت ARABIC LETTER TEH */
   {
   0x05cb, 0x062b},             /*                 Arabic_theh ث ARABIC LETTER THEH */
   {
   0x05cc, 0x062c},             /*                 Arabic_jeem ج ARABIC LETTER JEEM */
   {
   0x05cd, 0x062d},             /*                  Arabic_hah ح ARABIC LETTER HAH */
   {
   0x05ce, 0x062e},             /*                 Arabic_khah خ ARABIC LETTER KHAH */
   {
   0x05cf, 0x062f},             /*                  Arabic_dal د ARABIC LETTER DAL */
   {
   0x05d0, 0x0630},             /*                 Arabic_thal ذ ARABIC LETTER THAL */
   {
   0x05d1, 0x0631},             /*                   Arabic_ra ر ARABIC LETTER REH */
   {
   0x05d2, 0x0632},             /*                 Arabic_zain ز ARABIC LETTER ZAIN */
   {
   0x05d3, 0x0633},             /*                 Arabic_seen س ARABIC LETTER SEEN */
   {
   0x05d4, 0x0634},             /*                Arabic_sheen ش ARABIC LETTER SHEEN */
   {
   0x05d5, 0x0635},             /*                  Arabic_sad ص ARABIC LETTER SAD */
   {
   0x05d6, 0x0636},             /*                  Arabic_dad ض ARABIC LETTER DAD */
   {
   0x05d7, 0x0637},             /*                  Arabic_tah ط ARABIC LETTER TAH */
   {
   0x05d8, 0x0638},             /*                  Arabic_zah ظ ARABIC LETTER ZAH */
   {
   0x05d9, 0x0639},             /*                  Arabic_ain ع ARABIC LETTER AIN */
   {
   0x05da, 0x063a},             /*                Arabic_ghain غ ARABIC LETTER GHAIN */
   {
   0x05e0, 0x0640},             /*              Arabic_tatweel ـ ARABIC TATWEEL */
   {
   0x05e1, 0x0641},             /*                  Arabic_feh ف ARABIC LETTER FEH */
   {
   0x05e2, 0x0642},             /*                  Arabic_qaf ق ARABIC LETTER QAF */
   {
   0x05e3, 0x0643},             /*                  Arabic_kaf ك ARABIC LETTER KAF */
   {
   0x05e4, 0x0644},             /*                  Arabic_lam ل ARABIC LETTER LAM */
   {
   0x05e5, 0x0645},             /*                 Arabic_meem م ARABIC LETTER MEEM */
   {
   0x05e6, 0x0646},             /*                 Arabic_noon ن ARABIC LETTER NOON */
   {
   0x05e7, 0x0647},             /*                   Arabic_ha ه ARABIC LETTER HEH */
   {
   0x05e8, 0x0648},             /*                  Arabic_waw و ARABIC LETTER WAW */
   {
   0x05e9, 0x0649},             /*          Arabic_alefmaksura ى ARABIC LETTER ALEF MAKSURA */
   {
   0x05ea, 0x064a},             /*                  Arabic_yeh ي ARABIC LETTER YEH */
   {
   0x05eb, 0x064b},             /*             Arabic_fathatan ً ARABIC FATHATAN */
   {
   0x05ec, 0x064c},             /*             Arabic_dammatan ٌ ARABIC DAMMATAN */
   {
   0x05ed, 0x064d},             /*             Arabic_kasratan ٍ ARABIC KASRATAN */
   {
   0x05ee, 0x064e},             /*                Arabic_fatha َ ARABIC FATHA */
   {
   0x05ef, 0x064f},             /*                Arabic_damma ُ ARABIC DAMMA */
   {
   0x05f0, 0x0650},             /*                Arabic_kasra ِ ARABIC KASRA */
   {
   0x05f1, 0x0651},             /*               Arabic_shadda ّ ARABIC SHADDA */
   {
   0x05f2, 0x0652},             /*                Arabic_sukun ْ ARABIC SUKUN */
   {
   0x06a1, 0x0452},             /*                 Serbian_dje ђ CYRILLIC SMALL LETTER DJE */
   {
   0x06a2, 0x0453},             /*               Macedonia_gje ѓ CYRILLIC SMALL LETTER GJE */
   {
   0x06a3, 0x0451},             /*                 Cyrillic_io ё CYRILLIC SMALL LETTER IO */
   {
   0x06a4, 0x0454},             /*                Ukrainian_ie є CYRILLIC SMALL LETTER UKRAINIAN IE */
   {
   0x06a5, 0x0455},             /*               Macedonia_dse ѕ CYRILLIC SMALL LETTER DZE */
   {
   0x06a6, 0x0456},             /*                 Ukrainian_i і CYRILLIC SMALL LETTER BYELORUSSIAN-UKRAINIAN I */
   {
   0x06a7, 0x0457},             /*                Ukrainian_yi ї CYRILLIC SMALL LETTER YI */
   {
   0x06a8, 0x0458},             /*                 Cyrillic_je ј CYRILLIC SMALL LETTER JE */
   {
   0x06a9, 0x0459},             /*                Cyrillic_lje љ CYRILLIC SMALL LETTER LJE */
   {
   0x06aa, 0x045a},             /*                Cyrillic_nje њ CYRILLIC SMALL LETTER NJE */
   {
   0x06ab, 0x045b},             /*                Serbian_tshe ћ CYRILLIC SMALL LETTER TSHE */
   {
   0x06ac, 0x045c},             /*               Macedonia_kje ќ CYRILLIC SMALL LETTER KJE */
   {
   0x06ae, 0x045e},             /*         Byelorussian_shortu ў CYRILLIC SMALL LETTER SHORT U */
   {
   0x06af, 0x045f},             /*               Cyrillic_dzhe џ CYRILLIC SMALL LETTER DZHE */
   {
   0x06b0, 0x2116},             /*                  numerosign № NUMERO SIGN */
   {
   0x06b1, 0x0402},             /*                 Serbian_DJE Ђ CYRILLIC CAPITAL LETTER DJE */
   {
   0x06b2, 0x0403},             /*               Macedonia_GJE Ѓ CYRILLIC CAPITAL LETTER GJE */
   {
   0x06b3, 0x0401},             /*                 Cyrillic_IO Ё CYRILLIC CAPITAL LETTER IO */
   {
   0x06b4, 0x0404},             /*                Ukrainian_IE Є CYRILLIC CAPITAL LETTER UKRAINIAN IE */
   {
   0x06b5, 0x0405},             /*               Macedonia_DSE Ѕ CYRILLIC CAPITAL LETTER DZE */
   {
   0x06b6, 0x0406},             /*                 Ukrainian_I І CYRILLIC CAPITAL LETTER BYELORUSSIAN-UKRAINIAN I */
   {
   0x06b7, 0x0407},             /*                Ukrainian_YI Ї CYRILLIC CAPITAL LETTER YI */
   {
   0x06b8, 0x0408},             /*                 Cyrillic_JE Ј CYRILLIC CAPITAL LETTER JE */
   {
   0x06b9, 0x0409},             /*                Cyrillic_LJE Љ CYRILLIC CAPITAL LETTER LJE */
   {
   0x06ba, 0x040a},             /*                Cyrillic_NJE Њ CYRILLIC CAPITAL LETTER NJE */
   {
   0x06bb, 0x040b},             /*                Serbian_TSHE Ћ CYRILLIC CAPITAL LETTER TSHE */
   {
   0x06bc, 0x040c},             /*               Macedonia_KJE Ќ CYRILLIC CAPITAL LETTER KJE */
   {
   0x06be, 0x040e},             /*         Byelorussian_SHORTU Ў CYRILLIC CAPITAL LETTER SHORT U */
   {
   0x06bf, 0x040f},             /*               Cyrillic_DZHE Џ CYRILLIC CAPITAL LETTER DZHE */
   {
   0x06c0, 0x044e},             /*                 Cyrillic_yu ю CYRILLIC SMALL LETTER YU */
   {
   0x06c1, 0x0430},             /*                  Cyrillic_a а CYRILLIC SMALL LETTER A */
   {
   0x06c2, 0x0431},             /*                 Cyrillic_be б CYRILLIC SMALL LETTER BE */
   {
   0x06c3, 0x0446},             /*                Cyrillic_tse ц CYRILLIC SMALL LETTER TSE */
   {
   0x06c4, 0x0434},             /*                 Cyrillic_de д CYRILLIC SMALL LETTER DE */
   {
   0x06c5, 0x0435},             /*                 Cyrillic_ie е CYRILLIC SMALL LETTER IE */
   {
   0x06c6, 0x0444},             /*                 Cyrillic_ef ф CYRILLIC SMALL LETTER EF */
   {
   0x06c7, 0x0433},             /*                Cyrillic_ghe г CYRILLIC SMALL LETTER GHE */
   {
   0x06c8, 0x0445},             /*                 Cyrillic_ha х CYRILLIC SMALL LETTER HA */
   {
   0x06c9, 0x0438},             /*                  Cyrillic_i и CYRILLIC SMALL LETTER I */
   {
   0x06ca, 0x0439},             /*             Cyrillic_shorti й CYRILLIC SMALL LETTER SHORT I */
   {
   0x06cb, 0x043a},             /*                 Cyrillic_ka к CYRILLIC SMALL LETTER KA */
   {
   0x06cc, 0x043b},             /*                 Cyrillic_el л CYRILLIC SMALL LETTER EL */
   {
   0x06cd, 0x043c},             /*                 Cyrillic_em м CYRILLIC SMALL LETTER EM */
   {
   0x06ce, 0x043d},             /*                 Cyrillic_en н CYRILLIC SMALL LETTER EN */
   {
   0x06cf, 0x043e},             /*                  Cyrillic_o о CYRILLIC SMALL LETTER O */
   {
   0x06d0, 0x043f},             /*                 Cyrillic_pe п CYRILLIC SMALL LETTER PE */
   {
   0x06d1, 0x044f},             /*                 Cyrillic_ya я CYRILLIC SMALL LETTER YA */
   {
   0x06d2, 0x0440},             /*                 Cyrillic_er р CYRILLIC SMALL LETTER ER */
   {
   0x06d3, 0x0441},             /*                 Cyrillic_es с CYRILLIC SMALL LETTER ES */
   {
   0x06d4, 0x0442},             /*                 Cyrillic_te т CYRILLIC SMALL LETTER TE */
   {
   0x06d5, 0x0443},             /*                  Cyrillic_u у CYRILLIC SMALL LETTER U */
   {
   0x06d6, 0x0436},             /*                Cyrillic_zhe ж CYRILLIC SMALL LETTER ZHE */
   {
   0x06d7, 0x0432},             /*                 Cyrillic_ve в CYRILLIC SMALL LETTER VE */
   {
   0x06d8, 0x044c},             /*           Cyrillic_softsign ь CYRILLIC SMALL LETTER SOFT SIGN */
   {
   0x06d9, 0x044b},             /*               Cyrillic_yeru ы CYRILLIC SMALL LETTER YERU */
   {
   0x06da, 0x0437},             /*                 Cyrillic_ze з CYRILLIC SMALL LETTER ZE */
   {
   0x06db, 0x0448},             /*                Cyrillic_sha ш CYRILLIC SMALL LETTER SHA */
   {
   0x06dc, 0x044d},             /*                  Cyrillic_e э CYRILLIC SMALL LETTER E */
   {
   0x06dd, 0x0449},             /*              Cyrillic_shcha щ CYRILLIC SMALL LETTER SHCHA */
   {
   0x06de, 0x0447},             /*                Cyrillic_che ч CYRILLIC SMALL LETTER CHE */
   {
   0x06df, 0x044a},             /*           Cyrillic_hardsign ъ CYRILLIC SMALL LETTER HARD SIGN */
   {
   0x06e0, 0x042e},             /*                 Cyrillic_YU Ю CYRILLIC CAPITAL LETTER YU */
   {
   0x06e1, 0x0410},             /*                  Cyrillic_A А CYRILLIC CAPITAL LETTER A */
   {
   0x06e2, 0x0411},             /*                 Cyrillic_BE Б CYRILLIC CAPITAL LETTER BE */
   {
   0x06e3, 0x0426},             /*                Cyrillic_TSE Ц CYRILLIC CAPITAL LETTER TSE */
   {
   0x06e4, 0x0414},             /*                 Cyrillic_DE Д CYRILLIC CAPITAL LETTER DE */
   {
   0x06e5, 0x0415},             /*                 Cyrillic_IE Е CYRILLIC CAPITAL LETTER IE */
   {
   0x06e6, 0x0424},             /*                 Cyrillic_EF Ф CYRILLIC CAPITAL LETTER EF */
   {
   0x06e7, 0x0413},             /*                Cyrillic_GHE Г CYRILLIC CAPITAL LETTER GHE */
   {
   0x06e8, 0x0425},             /*                 Cyrillic_HA Х CYRILLIC CAPITAL LETTER HA */
   {
   0x06e9, 0x0418},             /*                  Cyrillic_I И CYRILLIC CAPITAL LETTER I */
   {
   0x06ea, 0x0419},             /*             Cyrillic_SHORTI Й CYRILLIC CAPITAL LETTER SHORT I */
   {
   0x06eb, 0x041a},             /*                 Cyrillic_KA К CYRILLIC CAPITAL LETTER KA */
   {
   0x06ec, 0x041b},             /*                 Cyrillic_EL Л CYRILLIC CAPITAL LETTER EL */
   {
   0x06ed, 0x041c},             /*                 Cyrillic_EM М CYRILLIC CAPITAL LETTER EM */
   {
   0x06ee, 0x041d},             /*                 Cyrillic_EN Н CYRILLIC CAPITAL LETTER EN */
   {
   0x06ef, 0x041e},             /*                  Cyrillic_O О CYRILLIC CAPITAL LETTER O */
   {
   0x06f0, 0x041f},             /*                 Cyrillic_PE П CYRILLIC CAPITAL LETTER PE */
   {
   0x06f1, 0x042f},             /*                 Cyrillic_YA Я CYRILLIC CAPITAL LETTER YA */
   {
   0x06f2, 0x0420},             /*                 Cyrillic_ER Р CYRILLIC CAPITAL LETTER ER */
   {
   0x06f3, 0x0421},             /*                 Cyrillic_ES С CYRILLIC CAPITAL LETTER ES */
   {
   0x06f4, 0x0422},             /*                 Cyrillic_TE Т CYRILLIC CAPITAL LETTER TE */
   {
   0x06f5, 0x0423},             /*                  Cyrillic_U У CYRILLIC CAPITAL LETTER U */
   {
   0x06f6, 0x0416},             /*                Cyrillic_ZHE Ж CYRILLIC CAPITAL LETTER ZHE */
   {
   0x06f7, 0x0412},             /*                 Cyrillic_VE В CYRILLIC CAPITAL LETTER VE */
   {
   0x06f8, 0x042c},             /*           Cyrillic_SOFTSIGN Ь CYRILLIC CAPITAL LETTER SOFT SIGN */
   {
   0x06f9, 0x042b},             /*               Cyrillic_YERU Ы CYRILLIC CAPITAL LETTER YERU */
   {
   0x06fa, 0x0417},             /*                 Cyrillic_ZE З CYRILLIC CAPITAL LETTER ZE */
   {
   0x06fb, 0x0428},             /*                Cyrillic_SHA Ш CYRILLIC CAPITAL LETTER SHA */
   {
   0x06fc, 0x042d},             /*                  Cyrillic_E Э CYRILLIC CAPITAL LETTER E */
   {
   0x06fd, 0x0429},             /*              Cyrillic_SHCHA Щ CYRILLIC CAPITAL LETTER SHCHA */
   {
   0x06fe, 0x0427},             /*                Cyrillic_CHE Ч CYRILLIC CAPITAL LETTER CHE */
   {
   0x06ff, 0x042a},             /*           Cyrillic_HARDSIGN Ъ CYRILLIC CAPITAL LETTER HARD SIGN */
   {
   0x07a1, 0x0386},             /*           Greek_ALPHAaccent Ά GREEK CAPITAL LETTER ALPHA WITH TONOS */
   {
   0x07a2, 0x0388},             /*         Greek_EPSILONaccent Έ GREEK CAPITAL LETTER EPSILON WITH TONOS */
   {
   0x07a3, 0x0389},             /*             Greek_ETAaccent Ή GREEK CAPITAL LETTER ETA WITH TONOS */
   {
   0x07a4, 0x038a},             /*            Greek_IOTAaccent Ί GREEK CAPITAL LETTER IOTA WITH TONOS */
   {
   0x07a5, 0x03aa},             /*         Greek_IOTAdiaeresis Ϊ GREEK CAPITAL LETTER IOTA WITH DIALYTIKA */
   {
   0x07a7, 0x038c},             /*         Greek_OMICRONaccent Ό GREEK CAPITAL LETTER OMICRON WITH TONOS */
   {
   0x07a8, 0x038e},             /*         Greek_UPSILONaccent Ύ GREEK CAPITAL LETTER UPSILON WITH TONOS */
   {
   0x07a9, 0x03ab},             /*       Greek_UPSILONdieresis Ϋ GREEK CAPITAL LETTER UPSILON WITH DIALYTIKA */
   {
   0x07ab, 0x038f},             /*           Greek_OMEGAaccent Ώ GREEK CAPITAL LETTER OMEGA WITH TONOS */
   {
   0x07ae, 0x0385},             /*        Greek_accentdieresis ΅ GREEK DIALYTIKA TONOS */
   {
   0x07af, 0x2015},             /*              Greek_horizbar ― HORIZONTAL BAR */
   {
   0x07b1, 0x03ac},             /*           Greek_alphaaccent ά GREEK SMALL LETTER ALPHA WITH TONOS */
   {
   0x07b2, 0x03ad},             /*         Greek_epsilonaccent έ GREEK SMALL LETTER EPSILON WITH TONOS */
   {
   0x07b3, 0x03ae},             /*             Greek_etaaccent ή GREEK SMALL LETTER ETA WITH TONOS */
   {
   0x07b4, 0x03af},             /*            Greek_iotaaccent ί GREEK SMALL LETTER IOTA WITH TONOS */
   {
   0x07b5, 0x03ca},             /*          Greek_iotadieresis ϊ GREEK SMALL LETTER IOTA WITH DIALYTIKA */
   {
   0x07b6, 0x0390},             /*    Greek_iotaaccentdieresis ΐ GREEK SMALL LETTER IOTA WITH DIALYTIKA AND TONOS */
   {
   0x07b7, 0x03cc},             /*         Greek_omicronaccent ό GREEK SMALL LETTER OMICRON WITH TONOS */
   {
   0x07b8, 0x03cd},             /*         Greek_upsilonaccent ύ GREEK SMALL LETTER UPSILON WITH TONOS */
   {
   0x07b9, 0x03cb},             /*       Greek_upsilondieresis ϋ GREEK SMALL LETTER UPSILON WITH DIALYTIKA */
   {
   0x07ba, 0x03b0},             /* Greek_upsilonaccentdieresis ΰ GREEK SMALL LETTER UPSILON WITH DIALYTIKA AND TONOS */
   {
   0x07bb, 0x03ce},             /*           Greek_omegaaccent ώ GREEK SMALL LETTER OMEGA WITH TONOS */
   {
   0x07c1, 0x0391},             /*                 Greek_ALPHA Α GREEK CAPITAL LETTER ALPHA */
   {
   0x07c2, 0x0392},             /*                  Greek_BETA Β GREEK CAPITAL LETTER BETA */
   {
   0x07c3, 0x0393},             /*                 Greek_GAMMA Γ GREEK CAPITAL LETTER GAMMA */
   {
   0x07c4, 0x0394},             /*                 Greek_DELTA Δ GREEK CAPITAL LETTER DELTA */
   {
   0x07c5, 0x0395},             /*               Greek_EPSILON Ε GREEK CAPITAL LETTER EPSILON */
   {
   0x07c6, 0x0396},             /*                  Greek_ZETA Ζ GREEK CAPITAL LETTER ZETA */
   {
   0x07c7, 0x0397},             /*                   Greek_ETA Η GREEK CAPITAL LETTER ETA */
   {
   0x07c8, 0x0398},             /*                 Greek_THETA Θ GREEK CAPITAL LETTER THETA */
   {
   0x07c9, 0x0399},             /*                  Greek_IOTA Ι GREEK CAPITAL LETTER IOTA */
   {
   0x07ca, 0x039a},             /*                 Greek_KAPPA Κ GREEK CAPITAL LETTER KAPPA */
   {
   0x07cb, 0x039b},             /*                Greek_LAMBDA Λ GREEK CAPITAL LETTER LAMDA */
   {
   0x07cc, 0x039c},             /*                    Greek_MU Μ GREEK CAPITAL LETTER MU */
   {
   0x07cd, 0x039d},             /*                    Greek_NU Ν GREEK CAPITAL LETTER NU */
   {
   0x07ce, 0x039e},             /*                    Greek_XI Ξ GREEK CAPITAL LETTER XI */
   {
   0x07cf, 0x039f},             /*               Greek_OMICRON Ο GREEK CAPITAL LETTER OMICRON */
   {
   0x07d0, 0x03a0},             /*                    Greek_PI Π GREEK CAPITAL LETTER PI */
   {
   0x07d1, 0x03a1},             /*                   Greek_RHO Ρ GREEK CAPITAL LETTER RHO */
   {
   0x07d2, 0x03a3},             /*                 Greek_SIGMA Σ GREEK CAPITAL LETTER SIGMA */
   {
   0x07d4, 0x03a4},             /*                   Greek_TAU Τ GREEK CAPITAL LETTER TAU */
   {
   0x07d5, 0x03a5},             /*               Greek_UPSILON Υ GREEK CAPITAL LETTER UPSILON */
   {
   0x07d6, 0x03a6},             /*                   Greek_PHI Φ GREEK CAPITAL LETTER PHI */
   {
   0x07d7, 0x03a7},             /*                   Greek_CHI Χ GREEK CAPITAL LETTER CHI */
   {
   0x07d8, 0x03a8},             /*                   Greek_PSI Ψ GREEK CAPITAL LETTER PSI */
   {
   0x07d9, 0x03a9},             /*                 Greek_OMEGA Ω GREEK CAPITAL LETTER OMEGA */
   {
   0x07e1, 0x03b1},             /*                 Greek_alpha α GREEK SMALL LETTER ALPHA */
   {
   0x07e2, 0x03b2},             /*                  Greek_beta β GREEK SMALL LETTER BETA */
   {
   0x07e3, 0x03b3},             /*                 Greek_gamma γ GREEK SMALL LETTER GAMMA */
   {
   0x07e4, 0x03b4},             /*                 Greek_delta δ GREEK SMALL LETTER DELTA */
   {
   0x07e5, 0x03b5},             /*               Greek_epsilon ε GREEK SMALL LETTER EPSILON */
   {
   0x07e6, 0x03b6},             /*                  Greek_zeta ζ GREEK SMALL LETTER ZETA */
   {
   0x07e7, 0x03b7},             /*                   Greek_eta η GREEK SMALL LETTER ETA */
   {
   0x07e8, 0x03b8},             /*                 Greek_theta θ GREEK SMALL LETTER THETA */
   {
   0x07e9, 0x03b9},             /*                  Greek_iota ι GREEK SMALL LETTER IOTA */
   {
   0x07ea, 0x03ba},             /*                 Greek_kappa κ GREEK SMALL LETTER KAPPA */
   {
   0x07eb, 0x03bb},             /*                Greek_lambda λ GREEK SMALL LETTER LAMDA */
   {
   0x07ec, 0x03bc},             /*                    Greek_mu μ GREEK SMALL LETTER MU */
   {
   0x07ed, 0x03bd},             /*                    Greek_nu ν GREEK SMALL LETTER NU */
   {
   0x07ee, 0x03be},             /*                    Greek_xi ξ GREEK SMALL LETTER XI */
   {
   0x07ef, 0x03bf},             /*               Greek_omicron ο GREEK SMALL LETTER OMICRON */
   {
   0x07f0, 0x03c0},             /*                    Greek_pi π GREEK SMALL LETTER PI */
   {
   0x07f1, 0x03c1},             /*                   Greek_rho ρ GREEK SMALL LETTER RHO */
   {
   0x07f2, 0x03c3},             /*                 Greek_sigma σ GREEK SMALL LETTER SIGMA */
   {
   0x07f3, 0x03c2},             /*       Greek_finalsmallsigma ς GREEK SMALL LETTER FINAL SIGMA */
   {
   0x07f4, 0x03c4},             /*                   Greek_tau τ GREEK SMALL LETTER TAU */
   {
   0x07f5, 0x03c5},             /*               Greek_upsilon υ GREEK SMALL LETTER UPSILON */
   {
   0x07f6, 0x03c6},             /*                   Greek_phi φ GREEK SMALL LETTER PHI */
   {
   0x07f7, 0x03c7},             /*                   Greek_chi χ GREEK SMALL LETTER CHI */
   {
   0x07f8, 0x03c8},             /*                   Greek_psi ψ GREEK SMALL LETTER PSI */
   {
   0x07f9, 0x03c9},             /*                 Greek_omega ω GREEK SMALL LETTER OMEGA */
/*  0x08a1                               leftradical ? ??? */
/*  0x08a2                            topleftradical ? ??? */
/*  0x08a3                            horizconnector ? ??? */
   {
   0x08a4, 0x2320},             /*                 topintegral ⌠ TOP HALF INTEGRAL */
   {
   0x08a5, 0x2321},             /*                 botintegral ⌡ BOTTOM HALF INTEGRAL */
   {
   0x08a6, 0x2502},             /*               vertconnector │ BOX DRAWINGS LIGHT VERTICAL */
/*  0x08a7                          topleftsqbracket ? ??? */
/*  0x08a8                          botleftsqbracket ? ??? */
/*  0x08a9                         toprightsqbracket ? ??? */
/*  0x08aa                         botrightsqbracket ? ??? */
/*  0x08ab                             topleftparens ? ??? */
/*  0x08ac                             botleftparens ? ??? */
/*  0x08ad                            toprightparens ? ??? */
/*  0x08ae                            botrightparens ? ??? */
/*  0x08af                      leftmiddlecurlybrace ? ??? */
/*  0x08b0                     rightmiddlecurlybrace ? ??? */
/*  0x08b1                          topleftsummation ? ??? */
/*  0x08b2                          botleftsummation ? ??? */
/*  0x08b3                 topvertsummationconnector ? ??? */
/*  0x08b4                 botvertsummationconnector ? ??? */
/*  0x08b5                         toprightsummation ? ??? */
/*  0x08b6                         botrightsummation ? ??? */
/*  0x08b7                      rightmiddlesummation ? ??? */
   {
   0x08bc, 0x2264},             /*               lessthanequal ≤ LESS-THAN OR EQUAL TO */
   {
   0x08bd, 0x2260},             /*                    notequal ≠ NOT EQUAL TO */
   {
   0x08be, 0x2265},             /*            greaterthanequal ≥ GREATER-THAN OR EQUAL TO */
   {
   0x08bf, 0x222b},             /*                    integral ∫ INTEGRAL */
   {
   0x08c0, 0x2234},             /*                   therefore ∴ THEREFORE */
   {
   0x08c1, 0x221d},             /*                   variation ∝ PROPORTIONAL TO */
   {
   0x08c2, 0x221e},             /*                    infinity ∞ INFINITY */
   {
   0x08c5, 0x2207},             /*                       nabla ∇ NABLA */
   {
   0x08c8, 0x2245},             /*                 approximate ≅ APPROXIMATELY EQUAL TO */
/*  0x08c9                              similarequal ? ??? */
   {
   0x08cd, 0x21d4},             /*                    ifonlyif ⇔ LEFT RIGHT DOUBLE ARROW */
   {
   0x08ce, 0x21d2},             /*                     implies ⇒ RIGHTWARDS DOUBLE ARROW */
   {
   0x08cf, 0x2261},             /*                   identical ≡ IDENTICAL TO */
   {
   0x08d6, 0x221a},             /*                     radical √ SQUARE ROOT */
   {
   0x08da, 0x2282},             /*                  includedin ⊂ SUBSET OF */
   {
   0x08db, 0x2283},             /*                    includes ⊃ SUPERSET OF */
   {
   0x08dc, 0x2229},             /*                intersection ∩ INTERSECTION */
   {
   0x08dd, 0x222a},             /*                       union ∪ UNION */
   {
   0x08de, 0x2227},             /*                  logicaland ∧ LOGICAL AND */
   {
   0x08df, 0x2228},             /*                   logicalor ∨ LOGICAL OR */
   {
   0x08ef, 0x2202},             /*           partialderivative ∂ PARTIAL DIFFERENTIAL */
   {
   0x08f6, 0x0192},             /*                    function ƒ LATIN SMALL LETTER F WITH HOOK */
   {
   0x08fb, 0x2190},             /*                   leftarrow ← LEFTWARDS ARROW */
   {
   0x08fc, 0x2191},             /*                     uparrow ↑ UPWARDS ARROW */
   {
   0x08fd, 0x2192},             /*                  rightarrow → RIGHTWARDS ARROW */
   {
   0x08fe, 0x2193},             /*                   downarrow ↓ DOWNWARDS ARROW */
   {
   0x09df, 0x2422},             /*                       blank ␢ BLANK SYMBOL */
   {
   0x09e0, 0x25c6},             /*                soliddiamond ◆ BLACK DIAMOND */
   {
   0x09e1, 0x2592},             /*                checkerboard ▒ MEDIUM SHADE */
   {
   0x09e2, 0x2409},             /*                          ht ␉ SYMBOL FOR HORIZONTAL TABULATION */
   {
   0x09e3, 0x240c},             /*                          ff ␌ SYMBOL FOR FORM FEED */
   {
   0x09e4, 0x240d},             /*                          cr ␍ SYMBOL FOR CARRIAGE RETURN */
   {
   0x09e5, 0x240a},             /*                          lf ␊ SYMBOL FOR LINE FEED */
   {
   0x09e8, 0x2424},             /*                          nl ␤ SYMBOL FOR NEWLINE */
   {
   0x09e9, 0x240b},             /*                          vt ␋ SYMBOL FOR VERTICAL TABULATION */
   {
   0x09ea, 0x2518},             /*              lowrightcorner ┘ BOX DRAWINGS LIGHT UP AND LEFT */
   {
   0x09eb, 0x2510},             /*               uprightcorner ┐ BOX DRAWINGS LIGHT DOWN AND LEFT */
   {
   0x09ec, 0x250c},             /*                upleftcorner ┌ BOX DRAWINGS LIGHT DOWN AND RIGHT */
   {
   0x09ed, 0x2514},             /*               lowleftcorner └ BOX DRAWINGS LIGHT UP AND RIGHT */
   {
   0x09ee, 0x253c},             /*               crossinglines ┼ BOX DRAWINGS LIGHT VERTICAL AND HORIZONTAL */
/*  0x09ef                            horizlinescan1 ? ??? */
/*  0x09f0                            horizlinescan3 ? ??? */
   {
   0x09f1, 0x2500},             /*              horizlinescan5 ─ BOX DRAWINGS LIGHT HORIZONTAL */
/*  0x09f2                            horizlinescan7 ? ??? */
/*  0x09f3                            horizlinescan9 ? ??? */
   {
   0x09f4, 0x251c},             /*                       leftt ├ BOX DRAWINGS LIGHT VERTICAL AND RIGHT */
   {
   0x09f5, 0x2524},             /*                      rightt ┤ BOX DRAWINGS LIGHT VERTICAL AND LEFT */
   {
   0x09f6, 0x2534},             /*                        bott ┴ BOX DRAWINGS LIGHT UP AND HORIZONTAL */
   {
   0x09f7, 0x252c},             /*                        topt ┬ BOX DRAWINGS LIGHT DOWN AND HORIZONTAL */
   {
   0x09f8, 0x2502},             /*                     vertbar │ BOX DRAWINGS LIGHT VERTICAL */
   {
   0x0aa1, 0x2003},             /*                     emspace   EM SPACE */
   {
   0x0aa2, 0x2002},             /*                     enspace   EN SPACE */
   {
   0x0aa3, 0x2004},             /*                    em3space   THREE-PER-EM SPACE */
   {
   0x0aa4, 0x2005},             /*                    em4space   FOUR-PER-EM SPACE */
   {
   0x0aa5, 0x2007},             /*                  digitspace   FIGURE SPACE */
   {
   0x0aa6, 0x2008},             /*                  punctspace   PUNCTUATION SPACE */
   {
   0x0aa7, 0x2009},             /*                   thinspace   THIN SPACE */
   {
   0x0aa8, 0x200a},             /*                   hairspace   HAIR SPACE */
   {
   0x0aa9, 0x2014},             /*                      emdash — EM DASH */
   {
   0x0aaa, 0x2013},             /*                      endash – EN DASH */
/*  0x0aac                               signifblank ? ??? */
   {
   0x0aae, 0x2026},             /*                    ellipsis … HORIZONTAL ELLIPSIS */
/*  0x0aaf                           doubbaselinedot ? ??? */
   {
   0x0ab0, 0x2153},             /*                    onethird ⅓ VULGAR FRACTION ONE THIRD */
   {
   0x0ab1, 0x2154},             /*                   twothirds ⅔ VULGAR FRACTION TWO THIRDS */
   {
   0x0ab2, 0x2155},             /*                    onefifth ⅕ VULGAR FRACTION ONE FIFTH */
   {
   0x0ab3, 0x2156},             /*                   twofifths ⅖ VULGAR FRACTION TWO FIFTHS */
   {
   0x0ab4, 0x2157},             /*                 threefifths ⅗ VULGAR FRACTION THREE FIFTHS */
   {
   0x0ab5, 0x2158},             /*                  fourfifths ⅘ VULGAR FRACTION FOUR FIFTHS */
   {
   0x0ab6, 0x2159},             /*                    onesixth ⅙ VULGAR FRACTION ONE SIXTH */
   {
   0x0ab7, 0x215a},             /*                  fivesixths ⅚ VULGAR FRACTION FIVE SIXTHS */
   {
   0x0ab8, 0x2105},             /*                      careof ℅ CARE OF */
   {
   0x0abb, 0x2012},             /*                     figdash ‒ FIGURE DASH */
   {
   0x0abc, 0x2329},             /*            leftanglebracket 〈 LEFT-POINTING ANGLE BRACKET */
   {
   0x0abd, 0x002e},             /*                decimalpoint . FULL STOP */
   {
   0x0abe, 0x232a},             /*           rightanglebracket 〉 RIGHT-POINTING ANGLE BRACKET */
/*  0x0abf                                    marker ? ??? */
   {
   0x0ac3, 0x215b},             /*                   oneeighth ⅛ VULGAR FRACTION ONE EIGHTH */
   {
   0x0ac4, 0x215c},             /*                threeeighths ⅜ VULGAR FRACTION THREE EIGHTHS */
   {
   0x0ac5, 0x215d},             /*                 fiveeighths ⅝ VULGAR FRACTION FIVE EIGHTHS */
   {
   0x0ac6, 0x215e},             /*                seveneighths ⅞ VULGAR FRACTION SEVEN EIGHTHS */
   {
   0x0ac9, 0x2122},             /*                   trademark ™ TRADE MARK SIGN */
   {
   0x0aca, 0x2613},             /*               signaturemark ☓ SALTIRE */
/*  0x0acb                         trademarkincircle ? ??? */
   {
   0x0acc, 0x25c1},             /*            leftopentriangle ◁ WHITE LEFT-POINTING TRIANGLE */
   {
   0x0acd, 0x25b7},             /*           rightopentriangle ▷ WHITE RIGHT-POINTING TRIANGLE */
   {
   0x0ace, 0x25cb},             /*                emopencircle ○ WHITE CIRCLE */
   {
   0x0acf, 0x25a1},             /*             emopenrectangle □ WHITE SQUARE */
   {
   0x0ad0, 0x2018},             /*         leftsinglequotemark ‘ LEFT SINGLE QUOTATION MARK */
   {
   0x0ad1, 0x2019},             /*        rightsinglequotemark ’ RIGHT SINGLE QUOTATION MARK */
   {
   0x0ad2, 0x201c},             /*         leftdoublequotemark “ LEFT DOUBLE QUOTATION MARK */
   {
   0x0ad3, 0x201d},             /*        rightdoublequotemark ” RIGHT DOUBLE QUOTATION MARK */
   {
   0x0ad4, 0x211e},             /*                prescription ℞ PRESCRIPTION TAKE */
   {
   0x0ad6, 0x2032},             /*                     minutes ′ PRIME */
   {
   0x0ad7, 0x2033},             /*                     seconds ″ DOUBLE PRIME */
   {
   0x0ad9, 0x271d},             /*                  latincross ✝ LATIN CROSS */
/*  0x0ada                                  hexagram ? ??? */
   {
   0x0adb, 0x25ac},             /*            filledrectbullet ▬ BLACK RECTANGLE */
   {
   0x0adc, 0x25c0},             /*         filledlefttribullet ◀ BLACK LEFT-POINTING TRIANGLE */
   {
   0x0add, 0x25b6},             /*        filledrighttribullet ▶ BLACK RIGHT-POINTING TRIANGLE */
   {
   0x0ade, 0x25cf},             /*              emfilledcircle ● BLACK CIRCLE */
   {
   0x0adf, 0x25a0},             /*                emfilledrect ■ BLACK SQUARE */
   {
   0x0ae0, 0x25e6},             /*            enopencircbullet ◦ WHITE BULLET */
   {
   0x0ae1, 0x25ab},             /*          enopensquarebullet ▫ WHITE SMALL SQUARE */
   {
   0x0ae2, 0x25ad},             /*              openrectbullet ▭ WHITE RECTANGLE */
   {
   0x0ae3, 0x25b3},             /*             opentribulletup △ WHITE UP-POINTING TRIANGLE */
   {
   0x0ae4, 0x25bd},             /*           opentribulletdown ▽ WHITE DOWN-POINTING TRIANGLE */
   {
   0x0ae5, 0x2606},             /*                    openstar ☆ WHITE STAR */
   {
   0x0ae6, 0x2022},             /*          enfilledcircbullet • BULLET */
   {
   0x0ae7, 0x25aa},             /*            enfilledsqbullet ▪ BLACK SMALL SQUARE */
   {
   0x0ae8, 0x25b2},             /*           filledtribulletup ▲ BLACK UP-POINTING TRIANGLE */
   {
   0x0ae9, 0x25bc},             /*         filledtribulletdown ▼ BLACK DOWN-POINTING TRIANGLE */
   {
   0x0aea, 0x261c},             /*                 leftpointer ☜ WHITE LEFT POINTING INDEX */
   {
   0x0aeb, 0x261e},             /*                rightpointer ☞ WHITE RIGHT POINTING INDEX */
   {
   0x0aec, 0x2663},             /*                        club ♣ BLACK CLUB SUIT */
   {
   0x0aed, 0x2666},             /*                     diamond ♦ BLACK DIAMOND SUIT */
   {
   0x0aee, 0x2665},             /*                       heart ♥ BLACK HEART SUIT */
   {
   0x0af0, 0x2720},             /*                maltesecross ✠ MALTESE CROSS */
   {
   0x0af1, 0x2020},             /*                      dagger † DAGGER */
   {
   0x0af2, 0x2021},             /*                doubledagger ‡ DOUBLE DAGGER */
   {
   0x0af3, 0x2713},             /*                   checkmark ✓ CHECK MARK */
   {
   0x0af4, 0x2717},             /*                 ballotcross ✗ BALLOT X */
   {
   0x0af5, 0x266f},             /*                musicalsharp ♯ MUSIC SHARP SIGN */
   {
   0x0af6, 0x266d},             /*                 musicalflat ♭ MUSIC FLAT SIGN */
   {
   0x0af7, 0x2642},             /*                  malesymbol ♂ MALE SIGN */
   {
   0x0af8, 0x2640},             /*                femalesymbol ♀ FEMALE SIGN */
   {
   0x0af9, 0x260e},             /*                   telephone ☎ BLACK TELEPHONE */
   {
   0x0afa, 0x2315},             /*           telephonerecorder ⌕ TELEPHONE RECORDER */
   {
   0x0afb, 0x2117},             /*         phonographcopyright ℗ SOUND RECORDING COPYRIGHT */
   {
   0x0afc, 0x2038},             /*                       caret ‸ CARET */
   {
   0x0afd, 0x201a},             /*          singlelowquotemark ‚ SINGLE LOW-9 QUOTATION MARK */
   {
   0x0afe, 0x201e},             /*          doublelowquotemark „ DOUBLE LOW-9 QUOTATION MARK */
/*  0x0aff                                    cursor ? ??? */
   {
   0x0ba3, 0x003c},             /*                   leftcaret < LESS-THAN SIGN */
   {
   0x0ba6, 0x003e},             /*                  rightcaret > GREATER-THAN SIGN */
   {
   0x0ba8, 0x2228},             /*                   downcaret ∨ LOGICAL OR */
   {
   0x0ba9, 0x2227},             /*                     upcaret ∧ LOGICAL AND */
   {
   0x0bc0, 0x00af},             /*                     overbar ¯ MACRON */
   {
   0x0bc2, 0x22a4},             /*                    downtack ⊤ DOWN TACK */
   {
   0x0bc3, 0x2229},             /*                      upshoe ∩ INTERSECTION */
   {
   0x0bc4, 0x230a},             /*                   downstile ⌊ LEFT FLOOR */
   {
   0x0bc6, 0x005f},             /*                    underbar _ LOW LINE */
   {
   0x0bca, 0x2218},             /*                         jot ∘ RING OPERATOR */
   {
   0x0bcc, 0x2395},             /*                        quad ⎕ APL FUNCTIONAL SYMBOL QUAD (Unicode 3.0) */
   {
   0x0bce, 0x22a5},             /*                      uptack ⊥ UP TACK */
   {
   0x0bcf, 0x25cb},             /*                      circle ○ WHITE CIRCLE */
   {
   0x0bd3, 0x2308},             /*                     upstile ⌈ LEFT CEILING */
   {
   0x0bd6, 0x222a},             /*                    downshoe ∪ UNION */
   {
   0x0bd8, 0x2283},             /*                   rightshoe ⊃ SUPERSET OF */
   {
   0x0bda, 0x2282},             /*                    leftshoe ⊂ SUBSET OF */
   {
   0x0bdc, 0x22a3},             /*                    lefttack ⊣ LEFT TACK */
   {
   0x0bfc, 0x22a2},             /*                   righttack ⊢ RIGHT TACK */
   {
   0x0cdf, 0x2017},             /*        hebrew_doublelowline ‗ DOUBLE LOW LINE */
   {
   0x0ce0, 0x05d0},             /*                hebrew_aleph א HEBREW LETTER ALEF */
   {
   0x0ce1, 0x05d1},             /*                  hebrew_bet ב HEBREW LETTER BET */
   {
   0x0ce2, 0x05d2},             /*                hebrew_gimel ג HEBREW LETTER GIMEL */
   {
   0x0ce3, 0x05d3},             /*                hebrew_dalet ד HEBREW LETTER DALET */
   {
   0x0ce4, 0x05d4},             /*                   hebrew_he ה HEBREW LETTER HE */
   {
   0x0ce5, 0x05d5},             /*                  hebrew_waw ו HEBREW LETTER VAV */
   {
   0x0ce6, 0x05d6},             /*                 hebrew_zain ז HEBREW LETTER ZAYIN */
   {
   0x0ce7, 0x05d7},             /*                 hebrew_chet ח HEBREW LETTER HET */
   {
   0x0ce8, 0x05d8},             /*                  hebrew_tet ט HEBREW LETTER TET */
   {
   0x0ce9, 0x05d9},             /*                  hebrew_yod י HEBREW LETTER YOD */
   {
   0x0cea, 0x05da},             /*            hebrew_finalkaph ך HEBREW LETTER FINAL KAF */
   {
   0x0ceb, 0x05db},             /*                 hebrew_kaph כ HEBREW LETTER KAF */
   {
   0x0cec, 0x05dc},             /*                hebrew_lamed ל HEBREW LETTER LAMED */
   {
   0x0ced, 0x05dd},             /*             hebrew_finalmem ם HEBREW LETTER FINAL MEM */
   {
   0x0cee, 0x05de},             /*                  hebrew_mem מ HEBREW LETTER MEM */
   {
   0x0cef, 0x05df},             /*             hebrew_finalnun ן HEBREW LETTER FINAL NUN */
   {
   0x0cf0, 0x05e0},             /*                  hebrew_nun נ HEBREW LETTER NUN */
   {
   0x0cf1, 0x05e1},             /*               hebrew_samech ס HEBREW LETTER SAMEKH */
   {
   0x0cf2, 0x05e2},             /*                 hebrew_ayin ע HEBREW LETTER AYIN */
   {
   0x0cf3, 0x05e3},             /*              hebrew_finalpe ף HEBREW LETTER FINAL PE */
   {
   0x0cf4, 0x05e4},             /*                   hebrew_pe פ HEBREW LETTER PE */
   {
   0x0cf5, 0x05e5},             /*            hebrew_finalzade ץ HEBREW LETTER FINAL TSADI */
   {
   0x0cf6, 0x05e6},             /*                 hebrew_zade צ HEBREW LETTER TSADI */
   {
   0x0cf7, 0x05e7},             /*                 hebrew_qoph ק HEBREW LETTER QOF */
   {
   0x0cf8, 0x05e8},             /*                 hebrew_resh ר HEBREW LETTER RESH */
   {
   0x0cf9, 0x05e9},             /*                 hebrew_shin ש HEBREW LETTER SHIN */
   {
   0x0cfa, 0x05ea},             /*                  hebrew_taw ת HEBREW LETTER TAV */
   {
   0x0da1, 0x0e01},             /*                  Thai_kokai ก THAI CHARACTER KO KAI */
   {
   0x0da2, 0x0e02},             /*                Thai_khokhai ข THAI CHARACTER KHO KHAI */
   {
   0x0da3, 0x0e03},             /*               Thai_khokhuat ฃ THAI CHARACTER KHO KHUAT */
   {
   0x0da4, 0x0e04},             /*               Thai_khokhwai ค THAI CHARACTER KHO KHWAI */
   {
   0x0da5, 0x0e05},             /*                Thai_khokhon ฅ THAI CHARACTER KHO KHON */
   {
   0x0da6, 0x0e06},             /*             Thai_khorakhang ฆ THAI CHARACTER KHO RAKHANG */
   {
   0x0da7, 0x0e07},             /*                 Thai_ngongu ง THAI CHARACTER NGO NGU */
   {
   0x0da8, 0x0e08},             /*                Thai_chochan จ THAI CHARACTER CHO CHAN */
   {
   0x0da9, 0x0e09},             /*               Thai_choching ฉ THAI CHARACTER CHO CHING */
   {
   0x0daa, 0x0e0a},             /*               Thai_chochang ช THAI CHARACTER CHO CHANG */
   {
   0x0dab, 0x0e0b},             /*                   Thai_soso ซ THAI CHARACTER SO SO */
   {
   0x0dac, 0x0e0c},             /*                Thai_chochoe ฌ THAI CHARACTER CHO CHOE */
   {
   0x0dad, 0x0e0d},             /*                 Thai_yoying ญ THAI CHARACTER YO YING */
   {
   0x0dae, 0x0e0e},             /*                Thai_dochada ฎ THAI CHARACTER DO CHADA */
   {
   0x0daf, 0x0e0f},             /*                Thai_topatak ฏ THAI CHARACTER TO PATAK */
   {
   0x0db0, 0x0e10},             /*                Thai_thothan ฐ THAI CHARACTER THO THAN */
   {
   0x0db1, 0x0e11},             /*          Thai_thonangmontho ฑ THAI CHARACTER THO NANGMONTHO */
   {
   0x0db2, 0x0e12},             /*             Thai_thophuthao ฒ THAI CHARACTER THO PHUTHAO */
   {
   0x0db3, 0x0e13},             /*                  Thai_nonen ณ THAI CHARACTER NO NEN */
   {
   0x0db4, 0x0e14},             /*                  Thai_dodek ด THAI CHARACTER DO DEK */
   {
   0x0db5, 0x0e15},             /*                  Thai_totao ต THAI CHARACTER TO TAO */
   {
   0x0db6, 0x0e16},             /*               Thai_thothung ถ THAI CHARACTER THO THUNG */
   {
   0x0db7, 0x0e17},             /*              Thai_thothahan ท THAI CHARACTER THO THAHAN */
   {
   0x0db8, 0x0e18},             /*               Thai_thothong ธ THAI CHARACTER THO THONG */
   {
   0x0db9, 0x0e19},             /*                   Thai_nonu น THAI CHARACTER NO NU */
   {
   0x0dba, 0x0e1a},             /*               Thai_bobaimai บ THAI CHARACTER BO BAIMAI */
   {
   0x0dbb, 0x0e1b},             /*                  Thai_popla ป THAI CHARACTER PO PLA */
   {
   0x0dbc, 0x0e1c},             /*               Thai_phophung ผ THAI CHARACTER PHO PHUNG */
   {
   0x0dbd, 0x0e1d},             /*                   Thai_fofa ฝ THAI CHARACTER FO FA */
   {
   0x0dbe, 0x0e1e},             /*                Thai_phophan พ THAI CHARACTER PHO PHAN */
   {
   0x0dbf, 0x0e1f},             /*                  Thai_fofan ฟ THAI CHARACTER FO FAN */
   {
   0x0dc0, 0x0e20},             /*             Thai_phosamphao ภ THAI CHARACTER PHO SAMPHAO */
   {
   0x0dc1, 0x0e21},             /*                   Thai_moma ม THAI CHARACTER MO MA */
   {
   0x0dc2, 0x0e22},             /*                  Thai_yoyak ย THAI CHARACTER YO YAK */
   {
   0x0dc3, 0x0e23},             /*                  Thai_rorua ร THAI CHARACTER RO RUA */
   {
   0x0dc4, 0x0e24},             /*                     Thai_ru ฤ THAI CHARACTER RU */
   {
   0x0dc5, 0x0e25},             /*                 Thai_loling ล THAI CHARACTER LO LING */
   {
   0x0dc6, 0x0e26},             /*                     Thai_lu ฦ THAI CHARACTER LU */
   {
   0x0dc7, 0x0e27},             /*                 Thai_wowaen ว THAI CHARACTER WO WAEN */
   {
   0x0dc8, 0x0e28},             /*                 Thai_sosala ศ THAI CHARACTER SO SALA */
   {
   0x0dc9, 0x0e29},             /*                 Thai_sorusi ษ THAI CHARACTER SO RUSI */
   {
   0x0dca, 0x0e2a},             /*                  Thai_sosua ส THAI CHARACTER SO SUA */
   {
   0x0dcb, 0x0e2b},             /*                  Thai_hohip ห THAI CHARACTER HO HIP */
   {
   0x0dcc, 0x0e2c},             /*                Thai_lochula ฬ THAI CHARACTER LO CHULA */
   {
   0x0dcd, 0x0e2d},             /*                   Thai_oang อ THAI CHARACTER O ANG */
   {
   0x0dce, 0x0e2e},             /*               Thai_honokhuk ฮ THAI CHARACTER HO NOKHUK */
   {
   0x0dcf, 0x0e2f},             /*              Thai_paiyannoi ฯ THAI CHARACTER PAIYANNOI */
   {
   0x0dd0, 0x0e30},             /*                  Thai_saraa ะ THAI CHARACTER SARA A */
   {
   0x0dd1, 0x0e31},             /*             Thai_maihanakat ั THAI CHARACTER MAI HAN-AKAT */
   {
   0x0dd2, 0x0e32},             /*                 Thai_saraaa า THAI CHARACTER SARA AA */
   {
   0x0dd3, 0x0e33},             /*                 Thai_saraam ำ THAI CHARACTER SARA AM */
   {
   0x0dd4, 0x0e34},             /*                  Thai_sarai ิ THAI CHARACTER SARA I */
   {
   0x0dd5, 0x0e35},             /*                 Thai_saraii ี THAI CHARACTER SARA II */
   {
   0x0dd6, 0x0e36},             /*                 Thai_saraue ึ THAI CHARACTER SARA UE */
   {
   0x0dd7, 0x0e37},             /*                Thai_sarauee ื THAI CHARACTER SARA UEE */
   {
   0x0dd8, 0x0e38},             /*                  Thai_sarau ุ THAI CHARACTER SARA U */
   {
   0x0dd9, 0x0e39},             /*                 Thai_sarauu ู THAI CHARACTER SARA UU */
   {
   0x0dda, 0x0e3a},             /*                Thai_phinthu ฺ THAI CHARACTER PHINTHU */
   {
   0x0dde, 0x0e3e},             /*      Thai_maihanakat_maitho ฾ ??? */
   {
   0x0ddf, 0x0e3f},             /*                   Thai_baht ฿ THAI CURRENCY SYMBOL BAHT */
   {
   0x0de0, 0x0e40},             /*                  Thai_sarae เ THAI CHARACTER SARA E */
   {
   0x0de1, 0x0e41},             /*                 Thai_saraae แ THAI CHARACTER SARA AE */
   {
   0x0de2, 0x0e42},             /*                  Thai_sarao โ THAI CHARACTER SARA O */
   {
   0x0de3, 0x0e43},             /*          Thai_saraaimaimuan ใ THAI CHARACTER SARA AI MAIMUAN */
   {
   0x0de4, 0x0e44},             /*         Thai_saraaimaimalai ไ THAI CHARACTER SARA AI MAIMALAI */
   {
   0x0de5, 0x0e45},             /*            Thai_lakkhangyao ๅ THAI CHARACTER LAKKHANGYAO */
   {
   0x0de6, 0x0e46},             /*               Thai_maiyamok ๆ THAI CHARACTER MAIYAMOK */
   {
   0x0de7, 0x0e47},             /*              Thai_maitaikhu ็ THAI CHARACTER MAITAIKHU */
   {
   0x0de8, 0x0e48},             /*                  Thai_maiek ่ THAI CHARACTER MAI EK */
   {
   0x0de9, 0x0e49},             /*                 Thai_maitho ้ THAI CHARACTER MAI THO */
   {
   0x0dea, 0x0e4a},             /*                 Thai_maitri ๊ THAI CHARACTER MAI TRI */
   {
   0x0deb, 0x0e4b},             /*            Thai_maichattawa ๋ THAI CHARACTER MAI CHATTAWA */
   {
   0x0dec, 0x0e4c},             /*            Thai_thanthakhat ์ THAI CHARACTER THANTHAKHAT */
   {
   0x0ded, 0x0e4d},             /*               Thai_nikhahit ํ THAI CHARACTER NIKHAHIT */
   {
   0x0df0, 0x0e50},             /*                 Thai_leksun ๐ THAI DIGIT ZERO */
   {
   0x0df1, 0x0e51},             /*                Thai_leknung ๑ THAI DIGIT ONE */
   {
   0x0df2, 0x0e52},             /*                Thai_leksong ๒ THAI DIGIT TWO */
   {
   0x0df3, 0x0e53},             /*                 Thai_leksam ๓ THAI DIGIT THREE */
   {
   0x0df4, 0x0e54},             /*                  Thai_leksi ๔ THAI DIGIT FOUR */
   {
   0x0df5, 0x0e55},             /*                  Thai_lekha ๕ THAI DIGIT FIVE */
   {
   0x0df6, 0x0e56},             /*                 Thai_lekhok ๖ THAI DIGIT SIX */
   {
   0x0df7, 0x0e57},             /*                Thai_lekchet ๗ THAI DIGIT SEVEN */
   {
   0x0df8, 0x0e58},             /*                Thai_lekpaet ๘ THAI DIGIT EIGHT */
   {
   0x0df9, 0x0e59},             /*                 Thai_lekkao ๙ THAI DIGIT NINE */
   {
   0x0ea1, 0x3131},             /*               Hangul_Kiyeog ㄱ HANGUL LETTER KIYEOK */
   {
   0x0ea2, 0x3132},             /*          Hangul_SsangKiyeog ㄲ HANGUL LETTER SSANGKIYEOK */
   {
   0x0ea3, 0x3133},             /*           Hangul_KiyeogSios ㄳ HANGUL LETTER KIYEOK-SIOS */
   {
   0x0ea4, 0x3134},             /*                Hangul_Nieun ㄴ HANGUL LETTER NIEUN */
   {
   0x0ea5, 0x3135},             /*           Hangul_NieunJieuj ㄵ HANGUL LETTER NIEUN-CIEUC */
   {
   0x0ea6, 0x3136},             /*           Hangul_NieunHieuh ㄶ HANGUL LETTER NIEUN-HIEUH */
   {
   0x0ea7, 0x3137},             /*               Hangul_Dikeud ㄷ HANGUL LETTER TIKEUT */
   {
   0x0ea8, 0x3138},             /*          Hangul_SsangDikeud ㄸ HANGUL LETTER SSANGTIKEUT */
   {
   0x0ea9, 0x3139},             /*                Hangul_Rieul ㄹ HANGUL LETTER RIEUL */
   {
   0x0eaa, 0x313a},             /*          Hangul_RieulKiyeog ㄺ HANGUL LETTER RIEUL-KIYEOK */
   {
   0x0eab, 0x313b},             /*           Hangul_RieulMieum ㄻ HANGUL LETTER RIEUL-MIEUM */
   {
   0x0eac, 0x313c},             /*           Hangul_RieulPieub ㄼ HANGUL LETTER RIEUL-PIEUP */
   {
   0x0ead, 0x313d},             /*            Hangul_RieulSios ㄽ HANGUL LETTER RIEUL-SIOS */
   {
   0x0eae, 0x313e},             /*           Hangul_RieulTieut ㄾ HANGUL LETTER RIEUL-THIEUTH */
   {
   0x0eaf, 0x313f},             /*          Hangul_RieulPhieuf ㄿ HANGUL LETTER RIEUL-PHIEUPH */
   {
   0x0eb0, 0x3140},             /*           Hangul_RieulHieuh ㅀ HANGUL LETTER RIEUL-HIEUH */
   {
   0x0eb1, 0x3141},             /*                Hangul_Mieum ㅁ HANGUL LETTER MIEUM */
   {
   0x0eb2, 0x3142},             /*                Hangul_Pieub ㅂ HANGUL LETTER PIEUP */
   {
   0x0eb3, 0x3143},             /*           Hangul_SsangPieub ㅃ HANGUL LETTER SSANGPIEUP */
   {
   0x0eb4, 0x3144},             /*            Hangul_PieubSios ㅄ HANGUL LETTER PIEUP-SIOS */
   {
   0x0eb5, 0x3145},             /*                 Hangul_Sios ㅅ HANGUL LETTER SIOS */
   {
   0x0eb6, 0x3146},             /*            Hangul_SsangSios ㅆ HANGUL LETTER SSANGSIOS */
   {
   0x0eb7, 0x3147},             /*                Hangul_Ieung ㅇ HANGUL LETTER IEUNG */
   {
   0x0eb8, 0x3148},             /*                Hangul_Jieuj ㅈ HANGUL LETTER CIEUC */
   {
   0x0eb9, 0x3149},             /*           Hangul_SsangJieuj ㅉ HANGUL LETTER SSANGCIEUC */
   {
   0x0eba, 0x314a},             /*                Hangul_Cieuc ㅊ HANGUL LETTER CHIEUCH */
   {
   0x0ebb, 0x314b},             /*               Hangul_Khieuq ㅋ HANGUL LETTER KHIEUKH */
   {
   0x0ebc, 0x314c},             /*                Hangul_Tieut ㅌ HANGUL LETTER THIEUTH */
   {
   0x0ebd, 0x314d},             /*               Hangul_Phieuf ㅍ HANGUL LETTER PHIEUPH */
   {
   0x0ebe, 0x314e},             /*                Hangul_Hieuh ㅎ HANGUL LETTER HIEUH */
   {
   0x0ebf, 0x314f},             /*                    Hangul_A ㅏ HANGUL LETTER A */
   {
   0x0ec0, 0x3150},             /*                   Hangul_AE ㅐ HANGUL LETTER AE */
   {
   0x0ec1, 0x3151},             /*                   Hangul_YA ㅑ HANGUL LETTER YA */
   {
   0x0ec2, 0x3152},             /*                  Hangul_YAE ㅒ HANGUL LETTER YAE */
   {
   0x0ec3, 0x3153},             /*                   Hangul_EO ㅓ HANGUL LETTER EO */
   {
   0x0ec4, 0x3154},             /*                    Hangul_E ㅔ HANGUL LETTER E */
   {
   0x0ec5, 0x3155},             /*                  Hangul_YEO ㅕ HANGUL LETTER YEO */
   {
   0x0ec6, 0x3156},             /*                   Hangul_YE ㅖ HANGUL LETTER YE */
   {
   0x0ec7, 0x3157},             /*                    Hangul_O ㅗ HANGUL LETTER O */
   {
   0x0ec8, 0x3158},             /*                   Hangul_WA ㅘ HANGUL LETTER WA */
   {
   0x0ec9, 0x3159},             /*                  Hangul_WAE ㅙ HANGUL LETTER WAE */
   {
   0x0eca, 0x315a},             /*                   Hangul_OE ㅚ HANGUL LETTER OE */
   {
   0x0ecb, 0x315b},             /*                   Hangul_YO ㅛ HANGUL LETTER YO */
   {
   0x0ecc, 0x315c},             /*                    Hangul_U ㅜ HANGUL LETTER U */
   {
   0x0ecd, 0x315d},             /*                  Hangul_WEO ㅝ HANGUL LETTER WEO */
   {
   0x0ece, 0x315e},             /*                   Hangul_WE ㅞ HANGUL LETTER WE */
   {
   0x0ecf, 0x315f},             /*                   Hangul_WI ㅟ HANGUL LETTER WI */
   {
   0x0ed0, 0x3160},             /*                   Hangul_YU ㅠ HANGUL LETTER YU */
   {
   0x0ed1, 0x3161},             /*                   Hangul_EU ㅡ HANGUL LETTER EU */
   {
   0x0ed2, 0x3162},             /*                   Hangul_YI ㅢ HANGUL LETTER YI */
   {
   0x0ed3, 0x3163},             /*                    Hangul_I ㅣ HANGUL LETTER I */
   {
   0x0ed4, 0x11a8},             /*             Hangul_J_Kiyeog ᆨ HANGUL JONGSEONG KIYEOK */
   {
   0x0ed5, 0x11a9},             /*        Hangul_J_SsangKiyeog ᆩ HANGUL JONGSEONG SSANGKIYEOK */
   {
   0x0ed6, 0x11aa},             /*         Hangul_J_KiyeogSios ᆪ HANGUL JONGSEONG KIYEOK-SIOS */
   {
   0x0ed7, 0x11ab},             /*              Hangul_J_Nieun ᆫ HANGUL JONGSEONG NIEUN */
   {
   0x0ed8, 0x11ac},             /*         Hangul_J_NieunJieuj ᆬ HANGUL JONGSEONG NIEUN-CIEUC */
   {
   0x0ed9, 0x11ad},             /*         Hangul_J_NieunHieuh ᆭ HANGUL JONGSEONG NIEUN-HIEUH */
   {
   0x0eda, 0x11ae},             /*             Hangul_J_Dikeud ᆮ HANGUL JONGSEONG TIKEUT */
   {
   0x0edb, 0x11af},             /*              Hangul_J_Rieul ᆯ HANGUL JONGSEONG RIEUL */
   {
   0x0edc, 0x11b0},             /*        Hangul_J_RieulKiyeog ᆰ HANGUL JONGSEONG RIEUL-KIYEOK */
   {
   0x0edd, 0x11b1},             /*         Hangul_J_RieulMieum ᆱ HANGUL JONGSEONG RIEUL-MIEUM */
   {
   0x0ede, 0x11b2},             /*         Hangul_J_RieulPieub ᆲ HANGUL JONGSEONG RIEUL-PIEUP */
   {
   0x0edf, 0x11b3},             /*          Hangul_J_RieulSios ᆳ HANGUL JONGSEONG RIEUL-SIOS */
   {
   0x0ee0, 0x11b4},             /*         Hangul_J_RieulTieut ᆴ HANGUL JONGSEONG RIEUL-THIEUTH */
   {
   0x0ee1, 0x11b5},             /*        Hangul_J_RieulPhieuf ᆵ HANGUL JONGSEONG RIEUL-PHIEUPH */
   {
   0x0ee2, 0x11b6},             /*         Hangul_J_RieulHieuh ᆶ HANGUL JONGSEONG RIEUL-HIEUH */
   {
   0x0ee3, 0x11b7},             /*              Hangul_J_Mieum ᆷ HANGUL JONGSEONG MIEUM */
   {
   0x0ee4, 0x11b8},             /*              Hangul_J_Pieub ᆸ HANGUL JONGSEONG PIEUP */
   {
   0x0ee5, 0x11b9},             /*          Hangul_J_PieubSios ᆹ HANGUL JONGSEONG PIEUP-SIOS */
   {
   0x0ee6, 0x11ba},             /*               Hangul_J_Sios ᆺ HANGUL JONGSEONG SIOS */
   {
   0x0ee7, 0x11bb},             /*          Hangul_J_SsangSios ᆻ HANGUL JONGSEONG SSANGSIOS */
   {
   0x0ee8, 0x11bc},             /*              Hangul_J_Ieung ᆼ HANGUL JONGSEONG IEUNG */
   {
   0x0ee9, 0x11bd},             /*              Hangul_J_Jieuj ᆽ HANGUL JONGSEONG CIEUC */
   {
   0x0eea, 0x11be},             /*              Hangul_J_Cieuc ᆾ HANGUL JONGSEONG CHIEUCH */
   {
   0x0eeb, 0x11bf},             /*             Hangul_J_Khieuq ᆿ HANGUL JONGSEONG KHIEUKH */
   {
   0x0eec, 0x11c0},             /*              Hangul_J_Tieut ᇀ HANGUL JONGSEONG THIEUTH */
   {
   0x0eed, 0x11c1},             /*             Hangul_J_Phieuf ᇁ HANGUL JONGSEONG PHIEUPH */
   {
   0x0eee, 0x11c2},             /*              Hangul_J_Hieuh ᇂ HANGUL JONGSEONG HIEUH */
   {
   0x0eef, 0x316d},             /*     Hangul_RieulYeorinHieuh ㅭ HANGUL LETTER RIEUL-YEORINHIEUH */
   {
   0x0ef0, 0x3171},             /*    Hangul_SunkyeongeumMieum ㅱ HANGUL LETTER KAPYEOUNMIEUM */
   {
   0x0ef1, 0x3178},             /*    Hangul_SunkyeongeumPieub ㅸ HANGUL LETTER KAPYEOUNPIEUP */
   {
   0x0ef2, 0x317f},             /*              Hangul_PanSios ㅿ HANGUL LETTER PANSIOS */
/*  0x0ef3                  Hangul_KkogjiDalrinIeung ? ??? */
   {
   0x0ef4, 0x3184},             /*   Hangul_SunkyeongeumPhieuf ㆄ HANGUL LETTER KAPYEOUNPHIEUPH */
   {
   0x0ef5, 0x3186},             /*          Hangul_YeorinHieuh ㆆ HANGUL LETTER YEORINHIEUH */
   {
   0x0ef6, 0x318d},             /*                Hangul_AraeA ㆍ HANGUL LETTER ARAEA */
   {
   0x0ef7, 0x318e},             /*               Hangul_AraeAE ㆎ HANGUL LETTER ARAEAE */
   {
   0x0ef8, 0x11eb},             /*            Hangul_J_PanSios ᇫ HANGUL JONGSEONG PANSIOS */
/*  0x0ef9                Hangul_J_KkogjiDalrinIeung ? ??? */
   {
   0x0efa, 0x11f9},             /*        Hangul_J_YeorinHieuh ᇹ HANGUL JONGSEONG YEORINHIEUH */
   {
   0x0eff, 0x20a9},             /*                  Korean_Won ₩ WON SIGN */
   {
   0x13bc, 0x0152},             /*                          OE Œ LATIN CAPITAL LIGATURE OE */
   {
   0x13bd, 0x0153},             /*                          oe œ LATIN SMALL LIGATURE OE */
   {
   0x13be, 0x0178},             /*                  Ydiaeresis Ÿ LATIN CAPITAL LETTER Y WITH DIAERESIS */
   {
   0x20a0, 0x20a0},             /*                     EcuSign ₠ EURO-CURRENCY SIGN */
   {
   0x20a1, 0x20a1},             /*                   ColonSign ₡ COLON SIGN */
   {
   0x20a2, 0x20a2},             /*                CruzeiroSign ₢ CRUZEIRO SIGN */
   {
   0x20a3, 0x20a3},             /*                  FFrancSign ₣ FRENCH FRANC SIGN */
   {
   0x20a4, 0x20a4},             /*                    LiraSign ₤ LIRA SIGN */
   {
   0x20a5, 0x20a5},             /*                    MillSign ₥ MILL SIGN */
   {
   0x20a6, 0x20a6},             /*                   NairaSign ₦ NAIRA SIGN */
   {
   0x20a7, 0x20a7},             /*                  PesetaSign ₧ PESETA SIGN */
   {
   0x20a8, 0x20a8},             /*                   RupeeSign ₨ RUPEE SIGN */
   {
   0x20a9, 0x20a9},             /*                     WonSign ₩ WON SIGN */
   {
   0x20aa, 0x20aa},             /*               NewSheqelSign ₪ NEW SHEQEL SIGN */
   {
   0x20ab, 0x20ab},             /*                    DongSign ₫ DONG SIGN */
   {
   0x20ac, 0x20ac},             /*                    EuroSign € EURO SIGN */
};

static guint keyval_to_unicode(guint keysym)
{
   int min = 0;
   int max = sizeof(k2utab) / sizeof(k2utab[0]) - 1;
   int mid;

   /* First check for Latin-1 characters (1:1 mapping) */
   if ((keysym >= 0x0020 && keysym <= 0x007e) ||
       (keysym >= 0x00a0 && keysym <= 0x00ff))
      return keysym;

   /* Also check for directly encoded 24-bit UCS characters */
   if ((keysym & 0xff000000) == 0x01000000)
      return keysym & 0x00ffffff;

   /* binary search in table */
   while (max >= min) {
      mid = (min + max) / 2;
      if (k2utab[mid].keysym < keysym)
         min = mid + 1;
      else if (k2utab[mid].keysym > keysym)
         max = mid - 1;
      else {
         /* found it */
         return k2utab[mid].ucs;
      }
   }

   /* No matching Unicode value found */
   return -1;
}

#endif                          /* 0 */

struct u2k {
   unsigned short keysym;
   unsigned short ucs;
} u2ktab[] = {
   {
   0x0abd, 0x002e},             /*                decimalpoint . FULL STOP */
   {
   0x0ba3, 0x003c},             /*                   leftcaret < LESS-THAN SIGN */
   {
   0x0ba6, 0x003e},             /*                  rightcaret > GREATER-THAN SIGN */
   {
   0x0bc6, 0x005f},             /*                    underbar _ LOW LINE */
   {
   0x0bc0, 0x00af},             /*                     overbar ¯ MACRON */
   {
   0x03c0, 0x0100},             /*                     Amacron Ā LATIN CAPITAL LETTER A WITH MACRON */
   {
   0x03e0, 0x0101},             /*                     amacron ā LATIN SMALL LETTER A WITH MACRON */
   {
   0x01c3, 0x0102},             /*                      Abreve Ă LATIN CAPITAL LETTER A WITH BREVE */
   {
   0x01e3, 0x0103},             /*                      abreve ă LATIN SMALL LETTER A WITH BREVE */
   {
   0x01a1, 0x0104},             /*                     Aogonek Ą LATIN CAPITAL LETTER A WITH OGONEK */
   {
   0x01b1, 0x0105},             /*                     aogonek ą LATIN SMALL LETTER A WITH OGONEK */
   {
   0x01c6, 0x0106},             /*                      Cacute Ć LATIN CAPITAL LETTER C WITH ACUTE */
   {
   0x01e6, 0x0107},             /*                      cacute ć LATIN SMALL LETTER C WITH ACUTE */
   {
   0x02c6, 0x0108},             /*                 Ccircumflex Ĉ LATIN CAPITAL LETTER C WITH CIRCUMFLEX */
   {
   0x02e6, 0x0109},             /*                 ccircumflex ĉ LATIN SMALL LETTER C WITH CIRCUMFLEX */
   {
   0x02c5, 0x010a},             /*                   Cabovedot Ċ LATIN CAPITAL LETTER C WITH DOT ABOVE */
   {
   0x02e5, 0x010b},             /*                   cabovedot ċ LATIN SMALL LETTER C WITH DOT ABOVE */
   {
   0x01c8, 0x010c},             /*                      Ccaron Č LATIN CAPITAL LETTER C WITH CARON */
   {
   0x01e8, 0x010d},             /*                      ccaron č LATIN SMALL LETTER C WITH CARON */
   {
   0x01cf, 0x010e},             /*                      Dcaron Ď LATIN CAPITAL LETTER D WITH CARON */
   {
   0x01ef, 0x010f},             /*                      dcaron ď LATIN SMALL LETTER D WITH CARON */
   {
   0x01d0, 0x0110},             /*                     Dstroke Đ LATIN CAPITAL LETTER D WITH STROKE */
   {
   0x01f0, 0x0111},             /*                     dstroke đ LATIN SMALL LETTER D WITH STROKE */
   {
   0x03aa, 0x0112},             /*                     Emacron Ē LATIN CAPITAL LETTER E WITH MACRON */
   {
   0x03ba, 0x0113},             /*                     emacron ē LATIN SMALL LETTER E WITH MACRON */
   {
   0x03cc, 0x0116},             /*                   Eabovedot Ė LATIN CAPITAL LETTER E WITH DOT ABOVE */
   {
   0x03ec, 0x0117},             /*                   eabovedot ė LATIN SMALL LETTER E WITH DOT ABOVE */
   {
   0x01ca, 0x0118},             /*                     Eogonek Ę LATIN CAPITAL LETTER E WITH OGONEK */
   {
   0x01ea, 0x0119},             /*                     eogonek ę LATIN SMALL LETTER E WITH OGONEK */
   {
   0x01cc, 0x011a},             /*                      Ecaron Ě LATIN CAPITAL LETTER E WITH CARON */
   {
   0x01ec, 0x011b},             /*                      ecaron ě LATIN SMALL LETTER E WITH CARON */
   {
   0x02d8, 0x011c},             /*                 Gcircumflex Ĝ LATIN CAPITAL LETTER G WITH CIRCUMFLEX */
   {
   0x02f8, 0x011d},             /*                 gcircumflex ĝ LATIN SMALL LETTER G WITH CIRCUMFLEX */
   {
   0x02ab, 0x011e},             /*                      Gbreve Ğ LATIN CAPITAL LETTER G WITH BREVE */
   {
   0x02bb, 0x011f},             /*                      gbreve ğ LATIN SMALL LETTER G WITH BREVE */
   {
   0x02d5, 0x0120},             /*                   Gabovedot Ġ LATIN CAPITAL LETTER G WITH DOT ABOVE */
   {
   0x02f5, 0x0121},             /*                   gabovedot ġ LATIN SMALL LETTER G WITH DOT ABOVE */
   {
   0x03ab, 0x0122},             /*                    Gcedilla Ģ LATIN CAPITAL LETTER G WITH CEDILLA */
   {
   0x03bb, 0x0123},             /*                    gcedilla ģ LATIN SMALL LETTER G WITH CEDILLA */
   {
   0x02a6, 0x0124},             /*                 Hcircumflex Ĥ LATIN CAPITAL LETTER H WITH CIRCUMFLEX */
   {
   0x02b6, 0x0125},             /*                 hcircumflex ĥ LATIN SMALL LETTER H WITH CIRCUMFLEX */
   {
   0x02a1, 0x0126},             /*                     Hstroke Ħ LATIN CAPITAL LETTER H WITH STROKE */
   {
   0x02b1, 0x0127},             /*                     hstroke ħ LATIN SMALL LETTER H WITH STROKE */
   {
   0x03a5, 0x0128},             /*                      Itilde Ĩ LATIN CAPITAL LETTER I WITH TILDE */
   {
   0x03b5, 0x0129},             /*                      itilde ĩ LATIN SMALL LETTER I WITH TILDE */
   {
   0x03cf, 0x012a},             /*                     Imacron Ī LATIN CAPITAL LETTER I WITH MACRON */
   {
   0x03ef, 0x012b},             /*                     imacron ī LATIN SMALL LETTER I WITH MACRON */
   {
   0x03c7, 0x012e},             /*                     Iogonek Į LATIN CAPITAL LETTER I WITH OGONEK */
   {
   0x03e7, 0x012f},             /*                     iogonek į LATIN SMALL LETTER I WITH OGONEK */
   {
   0x02a9, 0x0130},             /*                   Iabovedot İ LATIN CAPITAL LETTER I WITH DOT ABOVE */
   {
   0x02b9, 0x0131},             /*                    idotless ı LATIN SMALL LETTER DOTLESS I */
   {
   0x02ac, 0x0134},             /*                 Jcircumflex Ĵ LATIN CAPITAL LETTER J WITH CIRCUMFLEX */
   {
   0x02bc, 0x0135},             /*                 jcircumflex ĵ LATIN SMALL LETTER J WITH CIRCUMFLEX */
   {
   0x03d3, 0x0136},             /*                    Kcedilla Ķ LATIN CAPITAL LETTER K WITH CEDILLA */
   {
   0x03f3, 0x0137},             /*                    kcedilla ķ LATIN SMALL LETTER K WITH CEDILLA */
   {
   0x03a2, 0x0138},             /*                         kra ĸ LATIN SMALL LETTER KRA */
   {
   0x01c5, 0x0139},             /*                      Lacute Ĺ LATIN CAPITAL LETTER L WITH ACUTE */
   {
   0x01e5, 0x013a},             /*                      lacute ĺ LATIN SMALL LETTER L WITH ACUTE */
   {
   0x03a6, 0x013b},             /*                    Lcedilla Ļ LATIN CAPITAL LETTER L WITH CEDILLA */
   {
   0x03b6, 0x013c},             /*                    lcedilla ļ LATIN SMALL LETTER L WITH CEDILLA */
   {
   0x01a5, 0x013d},             /*                      Lcaron Ľ LATIN CAPITAL LETTER L WITH CARON */
   {
   0x01b5, 0x013e},             /*                      lcaron ľ LATIN SMALL LETTER L WITH CARON */
   {
   0x01a3, 0x0141},             /*                     Lstroke Ł LATIN CAPITAL LETTER L WITH STROKE */
   {
   0x01b3, 0x0142},             /*                     lstroke ł LATIN SMALL LETTER L WITH STROKE */
   {
   0x01d1, 0x0143},             /*                      Nacute Ń LATIN CAPITAL LETTER N WITH ACUTE */
   {
   0x01f1, 0x0144},             /*                      nacute ń LATIN SMALL LETTER N WITH ACUTE */
   {
   0x03d1, 0x0145},             /*                    Ncedilla Ņ LATIN CAPITAL LETTER N WITH CEDILLA */
   {
   0x03f1, 0x0146},             /*                    ncedilla ņ LATIN SMALL LETTER N WITH CEDILLA */
   {
   0x01d2, 0x0147},             /*                      Ncaron Ň LATIN CAPITAL LETTER N WITH CARON */
   {
   0x01f2, 0x0148},             /*                      ncaron ň LATIN SMALL LETTER N WITH CARON */
   {
   0x03bd, 0x014a},             /*                         ENG Ŋ LATIN CAPITAL LETTER ENG */
   {
   0x03bf, 0x014b},             /*                         eng ŋ LATIN SMALL LETTER ENG */
   {
   0x03d2, 0x014c},             /*                     Omacron Ō LATIN CAPITAL LETTER O WITH MACRON */
   {
   0x03f2, 0x014d},             /*                     omacron ō LATIN SMALL LETTER O WITH MACRON */
   {
   0x01d5, 0x0150},             /*                Odoubleacute Ő LATIN CAPITAL LETTER O WITH DOUBLE ACUTE */
   {
   0x01f5, 0x0151},             /*                odoubleacute ő LATIN SMALL LETTER O WITH DOUBLE ACUTE */
   {
   0x13bc, 0x0152},             /*                          OE Œ LATIN CAPITAL LIGATURE OE */
   {
   0x13bd, 0x0153},             /*                          oe œ LATIN SMALL LIGATURE OE */
   {
   0x01c0, 0x0154},             /*                      Racute Ŕ LATIN CAPITAL LETTER R WITH ACUTE */
   {
   0x01e0, 0x0155},             /*                      racute ŕ LATIN SMALL LETTER R WITH ACUTE */
   {
   0x03a3, 0x0156},             /*                    Rcedilla Ŗ LATIN CAPITAL LETTER R WITH CEDILLA */
   {
   0x03b3, 0x0157},             /*                    rcedilla ŗ LATIN SMALL LETTER R WITH CEDILLA */
   {
   0x01d8, 0x0158},             /*                      Rcaron Ř LATIN CAPITAL LETTER R WITH CARON */
   {
   0x01f8, 0x0159},             /*                      rcaron ř LATIN SMALL LETTER R WITH CARON */
   {
   0x01a6, 0x015a},             /*                      Sacute Ś LATIN CAPITAL LETTER S WITH ACUTE */
   {
   0x01b6, 0x015b},             /*                      sacute ś LATIN SMALL LETTER S WITH ACUTE */
   {
   0x02de, 0x015c},             /*                 Scircumflex Ŝ LATIN CAPITAL LETTER S WITH CIRCUMFLEX */
   {
   0x02fe, 0x015d},             /*                 scircumflex ŝ LATIN SMALL LETTER S WITH CIRCUMFLEX */
   {
   0x01aa, 0x015e},             /*                    Scedilla Ş LATIN CAPITAL LETTER S WITH CEDILLA */
   {
   0x01ba, 0x015f},             /*                    scedilla ş LATIN SMALL LETTER S WITH CEDILLA */
   {
   0x01a9, 0x0160},             /*                      Scaron Š LATIN CAPITAL LETTER S WITH CARON */
   {
   0x01b9, 0x0161},             /*                      scaron š LATIN SMALL LETTER S WITH CARON */
   {
   0x01de, 0x0162},             /*                    Tcedilla Ţ LATIN CAPITAL LETTER T WITH CEDILLA */
   {
   0x01fe, 0x0163},             /*                    tcedilla ţ LATIN SMALL LETTER T WITH CEDILLA */
   {
   0x01ab, 0x0164},             /*                      Tcaron Ť LATIN CAPITAL LETTER T WITH CARON */
   {
   0x01bb, 0x0165},             /*                      tcaron ť LATIN SMALL LETTER T WITH CARON */
   {
   0x03ac, 0x0166},             /*                      Tslash Ŧ LATIN CAPITAL LETTER T WITH STROKE */
   {
   0x03bc, 0x0167},             /*                      tslash ŧ LATIN SMALL LETTER T WITH STROKE */
   {
   0x03dd, 0x0168},             /*                      Utilde Ũ LATIN CAPITAL LETTER U WITH TILDE */
   {
   0x03fd, 0x0169},             /*                      utilde ũ LATIN SMALL LETTER U WITH TILDE */
   {
   0x03de, 0x016a},             /*                     Umacron Ū LATIN CAPITAL LETTER U WITH MACRON */
   {
   0x03fe, 0x016b},             /*                     umacron ū LATIN SMALL LETTER U WITH MACRON */
   {
   0x02dd, 0x016c},             /*                      Ubreve Ŭ LATIN CAPITAL LETTER U WITH BREVE */
   {
   0x02fd, 0x016d},             /*                      ubreve ŭ LATIN SMALL LETTER U WITH BREVE */
   {
   0x01d9, 0x016e},             /*                       Uring Ů LATIN CAPITAL LETTER U WITH RING ABOVE */
   {
   0x01f9, 0x016f},             /*                       uring ů LATIN SMALL LETTER U WITH RING ABOVE */
   {
   0x01db, 0x0170},             /*                Udoubleacute Ű LATIN CAPITAL LETTER U WITH DOUBLE ACUTE */
   {
   0x01fb, 0x0171},             /*                udoubleacute ű LATIN SMALL LETTER U WITH DOUBLE ACUTE */
   {
   0x03d9, 0x0172},             /*                     Uogonek Ų LATIN CAPITAL LETTER U WITH OGONEK */
   {
   0x03f9, 0x0173},             /*                     uogonek ų LATIN SMALL LETTER U WITH OGONEK */
   {
   0x13be, 0x0178},             /*                  Ydiaeresis Ÿ LATIN CAPITAL LETTER Y WITH DIAERESIS */
   {
   0x01ac, 0x0179},             /*                      Zacute Ź LATIN CAPITAL LETTER Z WITH ACUTE */
   {
   0x01bc, 0x017a},             /*                      zacute ź LATIN SMALL LETTER Z WITH ACUTE */
   {
   0x01af, 0x017b},             /*                   Zabovedot Ż LATIN CAPITAL LETTER Z WITH DOT ABOVE */
   {
   0x01bf, 0x017c},             /*                   zabovedot ż LATIN SMALL LETTER Z WITH DOT ABOVE */
   {
   0x01ae, 0x017d},             /*                      Zcaron Ž LATIN CAPITAL LETTER Z WITH CARON */
   {
   0x01be, 0x017e},             /*                      zcaron ž LATIN SMALL LETTER Z WITH CARON */
   {
   0x08f6, 0x0192},             /*                    function ƒ LATIN SMALL LETTER F WITH HOOK */
   {
   0x01b7, 0x02c7},             /*                       caron ˇ CARON */
   {
   0x01a2, 0x02d8},             /*                       breve ˘ BREVE */
   {
   0x01ff, 0x02d9},             /*                    abovedot ˙ DOT ABOVE */
   {
   0x01b2, 0x02db},             /*                      ogonek ˛ OGONEK */
   {
   0x01bd, 0x02dd},             /*                 doubleacute ˝ DOUBLE ACUTE ACCENT */
   {
   0x07ae, 0x0385},             /*        Greek_accentdieresis ΅ GREEK DIALYTIKA TONOS */
   {
   0x07a1, 0x0386},             /*           Greek_ALPHAaccent Ά GREEK CAPITAL LETTER ALPHA WITH TONOS */
   {
   0x07a2, 0x0388},             /*         Greek_EPSILONaccent Έ GREEK CAPITAL LETTER EPSILON WITH TONOS */
   {
   0x07a3, 0x0389},             /*             Greek_ETAaccent Ή GREEK CAPITAL LETTER ETA WITH TONOS */
   {
   0x07a4, 0x038a},             /*            Greek_IOTAaccent Ί GREEK CAPITAL LETTER IOTA WITH TONOS */
   {
   0x07a7, 0x038c},             /*         Greek_OMICRONaccent Ό GREEK CAPITAL LETTER OMICRON WITH TONOS */
   {
   0x07a8, 0x038e},             /*         Greek_UPSILONaccent Ύ GREEK CAPITAL LETTER UPSILON WITH TONOS */
   {
   0x07ab, 0x038f},             /*           Greek_OMEGAaccent Ώ GREEK CAPITAL LETTER OMEGA WITH TONOS */
   {
   0x07b6, 0x0390},             /*    Greek_iotaaccentdieresis ΐ GREEK SMALL LETTER IOTA WITH DIALYTIKA AND TONOS */
   {
   0x07c1, 0x0391},             /*                 Greek_ALPHA Α GREEK CAPITAL LETTER ALPHA */
   {
   0x07c2, 0x0392},             /*                  Greek_BETA Β GREEK CAPITAL LETTER BETA */
   {
   0x07c3, 0x0393},             /*                 Greek_GAMMA Γ GREEK CAPITAL LETTER GAMMA */
   {
   0x07c4, 0x0394},             /*                 Greek_DELTA Δ GREEK CAPITAL LETTER DELTA */
   {
   0x07c5, 0x0395},             /*               Greek_EPSILON Ε GREEK CAPITAL LETTER EPSILON */
   {
   0x07c6, 0x0396},             /*                  Greek_ZETA Ζ GREEK CAPITAL LETTER ZETA */
   {
   0x07c7, 0x0397},             /*                   Greek_ETA Η GREEK CAPITAL LETTER ETA */
   {
   0x07c8, 0x0398},             /*                 Greek_THETA Θ GREEK CAPITAL LETTER THETA */
   {
   0x07c9, 0x0399},             /*                  Greek_IOTA Ι GREEK CAPITAL LETTER IOTA */
   {
   0x07ca, 0x039a},             /*                 Greek_KAPPA Κ GREEK CAPITAL LETTER KAPPA */
   {
   0x07cb, 0x039b},             /*                Greek_LAMBDA Λ GREEK CAPITAL LETTER LAMDA */
   {
   0x07cc, 0x039c},             /*                    Greek_MU Μ GREEK CAPITAL LETTER MU */
   {
   0x07cd, 0x039d},             /*                    Greek_NU Ν GREEK CAPITAL LETTER NU */
   {
   0x07ce, 0x039e},             /*                    Greek_XI Ξ GREEK CAPITAL LETTER XI */
   {
   0x07cf, 0x039f},             /*               Greek_OMICRON Ο GREEK CAPITAL LETTER OMICRON */
   {
   0x07d0, 0x03a0},             /*                    Greek_PI Π GREEK CAPITAL LETTER PI */
   {
   0x07d1, 0x03a1},             /*                   Greek_RHO Ρ GREEK CAPITAL LETTER RHO */
   {
   0x07d2, 0x03a3},             /*                 Greek_SIGMA Σ GREEK CAPITAL LETTER SIGMA */
   {
   0x07d4, 0x03a4},             /*                   Greek_TAU Τ GREEK CAPITAL LETTER TAU */
   {
   0x07d5, 0x03a5},             /*               Greek_UPSILON Υ GREEK CAPITAL LETTER UPSILON */
   {
   0x07d6, 0x03a6},             /*                   Greek_PHI Φ GREEK CAPITAL LETTER PHI */
   {
   0x07d7, 0x03a7},             /*                   Greek_CHI Χ GREEK CAPITAL LETTER CHI */
   {
   0x07d8, 0x03a8},             /*                   Greek_PSI Ψ GREEK CAPITAL LETTER PSI */
   {
   0x07d9, 0x03a9},             /*                 Greek_OMEGA Ω GREEK CAPITAL LETTER OMEGA */
   {
   0x07a5, 0x03aa},             /*         Greek_IOTAdiaeresis Ϊ GREEK CAPITAL LETTER IOTA WITH DIALYTIKA */
   {
   0x07a9, 0x03ab},             /*       Greek_UPSILONdieresis Ϋ GREEK CAPITAL LETTER UPSILON WITH DIALYTIKA */
   {
   0x07b1, 0x03ac},             /*           Greek_alphaaccent ά GREEK SMALL LETTER ALPHA WITH TONOS */
   {
   0x07b2, 0x03ad},             /*         Greek_epsilonaccent έ GREEK SMALL LETTER EPSILON WITH TONOS */
   {
   0x07b3, 0x03ae},             /*             Greek_etaaccent ή GREEK SMALL LETTER ETA WITH TONOS */
   {
   0x07b4, 0x03af},             /*            Greek_iotaaccent ί GREEK SMALL LETTER IOTA WITH TONOS */
   {
   0x07ba, 0x03b0},             /* Greek_upsilonaccentdieresis ΰ GREEK SMALL LETTER UPSILON WITH DIALYTIKA AND TONOS */
   {
   0x07e1, 0x03b1},             /*                 Greek_alpha α GREEK SMALL LETTER ALPHA */
   {
   0x07e2, 0x03b2},             /*                  Greek_beta β GREEK SMALL LETTER BETA */
   {
   0x07e3, 0x03b3},             /*                 Greek_gamma γ GREEK SMALL LETTER GAMMA */
   {
   0x07e4, 0x03b4},             /*                 Greek_delta δ GREEK SMALL LETTER DELTA */
   {
   0x07e5, 0x03b5},             /*               Greek_epsilon ε GREEK SMALL LETTER EPSILON */
   {
   0x07e6, 0x03b6},             /*                  Greek_zeta ζ GREEK SMALL LETTER ZETA */
   {
   0x07e7, 0x03b7},             /*                   Greek_eta η GREEK SMALL LETTER ETA */
   {
   0x07e8, 0x03b8},             /*                 Greek_theta θ GREEK SMALL LETTER THETA */
   {
   0x07e9, 0x03b9},             /*                  Greek_iota ι GREEK SMALL LETTER IOTA */
   {
   0x07ea, 0x03ba},             /*                 Greek_kappa κ GREEK SMALL LETTER KAPPA */
   {
   0x07eb, 0x03bb},             /*                Greek_lambda λ GREEK SMALL LETTER LAMDA */
   {
   0x07ec, 0x03bc},             /*                    Greek_mu μ GREEK SMALL LETTER MU */
   {
   0x07ed, 0x03bd},             /*                    Greek_nu ν GREEK SMALL LETTER NU */
   {
   0x07ee, 0x03be},             /*                    Greek_xi ξ GREEK SMALL LETTER XI */
   {
   0x07ef, 0x03bf},             /*               Greek_omicron ο GREEK SMALL LETTER OMICRON */
   {
   0x07f0, 0x03c0},             /*                    Greek_pi π GREEK SMALL LETTER PI */
   {
   0x07f1, 0x03c1},             /*                   Greek_rho ρ GREEK SMALL LETTER RHO */
   {
   0x07f3, 0x03c2},             /*       Greek_finalsmallsigma ς GREEK SMALL LETTER FINAL SIGMA */
   {
   0x07f2, 0x03c3},             /*                 Greek_sigma σ GREEK SMALL LETTER SIGMA */
   {
   0x07f4, 0x03c4},             /*                   Greek_tau τ GREEK SMALL LETTER TAU */
   {
   0x07f5, 0x03c5},             /*               Greek_upsilon υ GREEK SMALL LETTER UPSILON */
   {
   0x07f6, 0x03c6},             /*                   Greek_phi φ GREEK SMALL LETTER PHI */
   {
   0x07f7, 0x03c7},             /*                   Greek_chi χ GREEK SMALL LETTER CHI */
   {
   0x07f8, 0x03c8},             /*                   Greek_psi ψ GREEK SMALL LETTER PSI */
   {
   0x07f9, 0x03c9},             /*                 Greek_omega ω GREEK SMALL LETTER OMEGA */
   {
   0x07b5, 0x03ca},             /*          Greek_iotadieresis ϊ GREEK SMALL LETTER IOTA WITH DIALYTIKA */
   {
   0x07b9, 0x03cb},             /*       Greek_upsilondieresis ϋ GREEK SMALL LETTER UPSILON WITH DIALYTIKA */
   {
   0x07b7, 0x03cc},             /*         Greek_omicronaccent ό GREEK SMALL LETTER OMICRON WITH TONOS */
   {
   0x07b8, 0x03cd},             /*         Greek_upsilonaccent ύ GREEK SMALL LETTER UPSILON WITH TONOS */
   {
   0x07bb, 0x03ce},             /*           Greek_omegaaccent ώ GREEK SMALL LETTER OMEGA WITH TONOS */
   {
   0x06b3, 0x0401},             /*                 Cyrillic_IO Ё CYRILLIC CAPITAL LETTER IO */
   {
   0x06b1, 0x0402},             /*                 Serbian_DJE Ђ CYRILLIC CAPITAL LETTER DJE */
   {
   0x06b2, 0x0403},             /*               Macedonia_GJE Ѓ CYRILLIC CAPITAL LETTER GJE */
   {
   0x06b4, 0x0404},             /*                Ukrainian_IE Є CYRILLIC CAPITAL LETTER UKRAINIAN IE */
   {
   0x06b5, 0x0405},             /*               Macedonia_DSE Ѕ CYRILLIC CAPITAL LETTER DZE */
   {
   0x06b6, 0x0406},             /*                 Ukrainian_I І CYRILLIC CAPITAL LETTER BYELORUSSIAN-UKRAINIAN I */
   {
   0x06b7, 0x0407},             /*                Ukrainian_YI Ї CYRILLIC CAPITAL LETTER YI */
   {
   0x06b8, 0x0408},             /*                 Cyrillic_JE Ј CYRILLIC CAPITAL LETTER JE */
   {
   0x06b9, 0x0409},             /*                Cyrillic_LJE Љ CYRILLIC CAPITAL LETTER LJE */
   {
   0x06ba, 0x040a},             /*                Cyrillic_NJE Њ CYRILLIC CAPITAL LETTER NJE */
   {
   0x06bb, 0x040b},             /*                Serbian_TSHE Ћ CYRILLIC CAPITAL LETTER TSHE */
   {
   0x06bc, 0x040c},             /*               Macedonia_KJE Ќ CYRILLIC CAPITAL LETTER KJE */
   {
   0x06be, 0x040e},             /*         Byelorussian_SHORTU Ў CYRILLIC CAPITAL LETTER SHORT U */
   {
   0x06bf, 0x040f},             /*               Cyrillic_DZHE Џ CYRILLIC CAPITAL LETTER DZHE */
   {
   0x06e1, 0x0410},             /*                  Cyrillic_A А CYRILLIC CAPITAL LETTER A */
   {
   0x06e2, 0x0411},             /*                 Cyrillic_BE Б CYRILLIC CAPITAL LETTER BE */
   {
   0x06f7, 0x0412},             /*                 Cyrillic_VE В CYRILLIC CAPITAL LETTER VE */
   {
   0x06e7, 0x0413},             /*                Cyrillic_GHE Г CYRILLIC CAPITAL LETTER GHE */
   {
   0x06e4, 0x0414},             /*                 Cyrillic_DE Д CYRILLIC CAPITAL LETTER DE */
   {
   0x06e5, 0x0415},             /*                 Cyrillic_IE Е CYRILLIC CAPITAL LETTER IE */
   {
   0x06f6, 0x0416},             /*                Cyrillic_ZHE Ж CYRILLIC CAPITAL LETTER ZHE */
   {
   0x06fa, 0x0417},             /*                 Cyrillic_ZE З CYRILLIC CAPITAL LETTER ZE */
   {
   0x06e9, 0x0418},             /*                  Cyrillic_I И CYRILLIC CAPITAL LETTER I */
   {
   0x06ea, 0x0419},             /*             Cyrillic_SHORTI Й CYRILLIC CAPITAL LETTER SHORT I */
   {
   0x06eb, 0x041a},             /*                 Cyrillic_KA К CYRILLIC CAPITAL LETTER KA */
   {
   0x06ec, 0x041b},             /*                 Cyrillic_EL Л CYRILLIC CAPITAL LETTER EL */
   {
   0x06ed, 0x041c},             /*                 Cyrillic_EM М CYRILLIC CAPITAL LETTER EM */
   {
   0x06ee, 0x041d},             /*                 Cyrillic_EN Н CYRILLIC CAPITAL LETTER EN */
   {
   0x06ef, 0x041e},             /*                  Cyrillic_O О CYRILLIC CAPITAL LETTER O */
   {
   0x06f0, 0x041f},             /*                 Cyrillic_PE П CYRILLIC CAPITAL LETTER PE */
   {
   0x06f2, 0x0420},             /*                 Cyrillic_ER Р CYRILLIC CAPITAL LETTER ER */
   {
   0x06f3, 0x0421},             /*                 Cyrillic_ES С CYRILLIC CAPITAL LETTER ES */
   {
   0x06f4, 0x0422},             /*                 Cyrillic_TE Т CYRILLIC CAPITAL LETTER TE */
   {
   0x06f5, 0x0423},             /*                  Cyrillic_U У CYRILLIC CAPITAL LETTER U */
   {
   0x06e6, 0x0424},             /*                 Cyrillic_EF Ф CYRILLIC CAPITAL LETTER EF */
   {
   0x06e8, 0x0425},             /*                 Cyrillic_HA Х CYRILLIC CAPITAL LETTER HA */
   {
   0x06e3, 0x0426},             /*                Cyrillic_TSE Ц CYRILLIC CAPITAL LETTER TSE */
   {
   0x06fe, 0x0427},             /*                Cyrillic_CHE Ч CYRILLIC CAPITAL LETTER CHE */
   {
   0x06fb, 0x0428},             /*                Cyrillic_SHA Ш CYRILLIC CAPITAL LETTER SHA */
   {
   0x06fd, 0x0429},             /*              Cyrillic_SHCHA Щ CYRILLIC CAPITAL LETTER SHCHA */
   {
   0x06ff, 0x042a},             /*           Cyrillic_HARDSIGN Ъ CYRILLIC CAPITAL LETTER HARD SIGN */
   {
   0x06f9, 0x042b},             /*               Cyrillic_YERU Ы CYRILLIC CAPITAL LETTER YERU */
   {
   0x06f8, 0x042c},             /*           Cyrillic_SOFTSIGN Ь CYRILLIC CAPITAL LETTER SOFT SIGN */
   {
   0x06fc, 0x042d},             /*                  Cyrillic_E Э CYRILLIC CAPITAL LETTER E */
   {
   0x06e0, 0x042e},             /*                 Cyrillic_YU Ю CYRILLIC CAPITAL LETTER YU */
   {
   0x06f1, 0x042f},             /*                 Cyrillic_YA Я CYRILLIC CAPITAL LETTER YA */
   {
   0x06c1, 0x0430},             /*                  Cyrillic_a а CYRILLIC SMALL LETTER A */
   {
   0x06c2, 0x0431},             /*                 Cyrillic_be б CYRILLIC SMALL LETTER BE */
   {
   0x06d7, 0x0432},             /*                 Cyrillic_ve в CYRILLIC SMALL LETTER VE */
   {
   0x06c7, 0x0433},             /*                Cyrillic_ghe г CYRILLIC SMALL LETTER GHE */
   {
   0x06c4, 0x0434},             /*                 Cyrillic_de д CYRILLIC SMALL LETTER DE */
   {
   0x06c5, 0x0435},             /*                 Cyrillic_ie е CYRILLIC SMALL LETTER IE */
   {
   0x06d6, 0x0436},             /*                Cyrillic_zhe ж CYRILLIC SMALL LETTER ZHE */
   {
   0x06da, 0x0437},             /*                 Cyrillic_ze з CYRILLIC SMALL LETTER ZE */
   {
   0x06c9, 0x0438},             /*                  Cyrillic_i и CYRILLIC SMALL LETTER I */
   {
   0x06ca, 0x0439},             /*             Cyrillic_shorti й CYRILLIC SMALL LETTER SHORT I */
   {
   0x06cb, 0x043a},             /*                 Cyrillic_ka к CYRILLIC SMALL LETTER KA */
   {
   0x06cc, 0x043b},             /*                 Cyrillic_el л CYRILLIC SMALL LETTER EL */
   {
   0x06cd, 0x043c},             /*                 Cyrillic_em м CYRILLIC SMALL LETTER EM */
   {
   0x06ce, 0x043d},             /*                 Cyrillic_en н CYRILLIC SMALL LETTER EN */
   {
   0x06cf, 0x043e},             /*                  Cyrillic_o о CYRILLIC SMALL LETTER O */
   {
   0x06d0, 0x043f},             /*                 Cyrillic_pe п CYRILLIC SMALL LETTER PE */
   {
   0x06d2, 0x0440},             /*                 Cyrillic_er р CYRILLIC SMALL LETTER ER */
   {
   0x06d3, 0x0441},             /*                 Cyrillic_es с CYRILLIC SMALL LETTER ES */
   {
   0x06d4, 0x0442},             /*                 Cyrillic_te т CYRILLIC SMALL LETTER TE */
   {
   0x06d5, 0x0443},             /*                  Cyrillic_u у CYRILLIC SMALL LETTER U */
   {
   0x06c6, 0x0444},             /*                 Cyrillic_ef ф CYRILLIC SMALL LETTER EF */
   {
   0x06c8, 0x0445},             /*                 Cyrillic_ha х CYRILLIC SMALL LETTER HA */
   {
   0x06c3, 0x0446},             /*                Cyrillic_tse ц CYRILLIC SMALL LETTER TSE */
   {
   0x06de, 0x0447},             /*                Cyrillic_che ч CYRILLIC SMALL LETTER CHE */
   {
   0x06db, 0x0448},             /*                Cyrillic_sha ш CYRILLIC SMALL LETTER SHA */
   {
   0x06dd, 0x0449},             /*              Cyrillic_shcha щ CYRILLIC SMALL LETTER SHCHA */
   {
   0x06df, 0x044a},             /*           Cyrillic_hardsign ъ CYRILLIC SMALL LETTER HARD SIGN */
   {
   0x06d9, 0x044b},             /*               Cyrillic_yeru ы CYRILLIC SMALL LETTER YERU */
   {
   0x06d8, 0x044c},             /*           Cyrillic_softsign ь CYRILLIC SMALL LETTER SOFT SIGN */
   {
   0x06dc, 0x044d},             /*                  Cyrillic_e э CYRILLIC SMALL LETTER E */
   {
   0x06c0, 0x044e},             /*                 Cyrillic_yu ю CYRILLIC SMALL LETTER YU */
   {
   0x06d1, 0x044f},             /*                 Cyrillic_ya я CYRILLIC SMALL LETTER YA */
   {
   0x06a3, 0x0451},             /*                 Cyrillic_io ё CYRILLIC SMALL LETTER IO */
   {
   0x06a1, 0x0452},             /*                 Serbian_dje ђ CYRILLIC SMALL LETTER DJE */
   {
   0x06a2, 0x0453},             /*               Macedonia_gje ѓ CYRILLIC SMALL LETTER GJE */
   {
   0x06a4, 0x0454},             /*                Ukrainian_ie є CYRILLIC SMALL LETTER UKRAINIAN IE */
   {
   0x06a5, 0x0455},             /*               Macedonia_dse ѕ CYRILLIC SMALL LETTER DZE */
   {
   0x06a6, 0x0456},             /*                 Ukrainian_i і CYRILLIC SMALL LETTER BYELORUSSIAN-UKRAINIAN I */
   {
   0x06a7, 0x0457},             /*                Ukrainian_yi ї CYRILLIC SMALL LETTER YI */
   {
   0x06a8, 0x0458},             /*                 Cyrillic_je ј CYRILLIC SMALL LETTER JE */
   {
   0x06a9, 0x0459},             /*                Cyrillic_lje љ CYRILLIC SMALL LETTER LJE */
   {
   0x06aa, 0x045a},             /*                Cyrillic_nje њ CYRILLIC SMALL LETTER NJE */
   {
   0x06ab, 0x045b},             /*                Serbian_tshe ћ CYRILLIC SMALL LETTER TSHE */
   {
   0x06ac, 0x045c},             /*               Macedonia_kje ќ CYRILLIC SMALL LETTER KJE */
   {
   0x06ae, 0x045e},             /*         Byelorussian_shortu ў CYRILLIC SMALL LETTER SHORT U */
   {
   0x06af, 0x045f},             /*               Cyrillic_dzhe џ CYRILLIC SMALL LETTER DZHE */
   {
   0x0ce0, 0x05d0},             /*                hebrew_aleph א HEBREW LETTER ALEF */
   {
   0x0ce1, 0x05d1},             /*                  hebrew_bet ב HEBREW LETTER BET */
   {
   0x0ce2, 0x05d2},             /*                hebrew_gimel ג HEBREW LETTER GIMEL */
   {
   0x0ce3, 0x05d3},             /*                hebrew_dalet ד HEBREW LETTER DALET */
   {
   0x0ce4, 0x05d4},             /*                   hebrew_he ה HEBREW LETTER HE */
   {
   0x0ce5, 0x05d5},             /*                  hebrew_waw ו HEBREW LETTER VAV */
   {
   0x0ce6, 0x05d6},             /*                 hebrew_zain ז HEBREW LETTER ZAYIN */
   {
   0x0ce7, 0x05d7},             /*                 hebrew_chet ח HEBREW LETTER HET */
   {
   0x0ce8, 0x05d8},             /*                  hebrew_tet ט HEBREW LETTER TET */
   {
   0x0ce9, 0x05d9},             /*                  hebrew_yod י HEBREW LETTER YOD */
   {
   0x0cea, 0x05da},             /*            hebrew_finalkaph ך HEBREW LETTER FINAL KAF */
   {
   0x0ceb, 0x05db},             /*                 hebrew_kaph כ HEBREW LETTER KAF */
   {
   0x0cec, 0x05dc},             /*                hebrew_lamed ל HEBREW LETTER LAMED */
   {
   0x0ced, 0x05dd},             /*             hebrew_finalmem ם HEBREW LETTER FINAL MEM */
   {
   0x0cee, 0x05de},             /*                  hebrew_mem מ HEBREW LETTER MEM */
   {
   0x0cef, 0x05df},             /*             hebrew_finalnun ן HEBREW LETTER FINAL NUN */
   {
   0x0cf0, 0x05e0},             /*                  hebrew_nun נ HEBREW LETTER NUN */
   {
   0x0cf1, 0x05e1},             /*               hebrew_samech ס HEBREW LETTER SAMEKH */
   {
   0x0cf2, 0x05e2},             /*                 hebrew_ayin ע HEBREW LETTER AYIN */
   {
   0x0cf3, 0x05e3},             /*              hebrew_finalpe ף HEBREW LETTER FINAL PE */
   {
   0x0cf4, 0x05e4},             /*                   hebrew_pe פ HEBREW LETTER PE */
   {
   0x0cf5, 0x05e5},             /*            hebrew_finalzade ץ HEBREW LETTER FINAL TSADI */
   {
   0x0cf6, 0x05e6},             /*                 hebrew_zade צ HEBREW LETTER TSADI */
   {
   0x0cf7, 0x05e7},             /*                 hebrew_qoph ק HEBREW LETTER QOF */
   {
   0x0cf8, 0x05e8},             /*                 hebrew_resh ר HEBREW LETTER RESH */
   {
   0x0cf9, 0x05e9},             /*                 hebrew_shin ש HEBREW LETTER SHIN */
   {
   0x0cfa, 0x05ea},             /*                  hebrew_taw ת HEBREW LETTER TAV */
   {
   0x05ac, 0x060c},             /*                Arabic_comma ، ARABIC COMMA */
   {
   0x05bb, 0x061b},             /*            Arabic_semicolon ؛ ARABIC SEMICOLON */
   {
   0x05bf, 0x061f},             /*        Arabic_question_mark ؟ ARABIC QUESTION MARK */
   {
   0x05c1, 0x0621},             /*                Arabic_hamza ء ARABIC LETTER HAMZA */
   {
   0x05c2, 0x0622},             /*          Arabic_maddaonalef آ ARABIC LETTER ALEF WITH MADDA ABOVE */
   {
   0x05c3, 0x0623},             /*          Arabic_hamzaonalef أ ARABIC LETTER ALEF WITH HAMZA ABOVE */
   {
   0x05c4, 0x0624},             /*           Arabic_hamzaonwaw ؤ ARABIC LETTER WAW WITH HAMZA ABOVE */
   {
   0x05c5, 0x0625},             /*       Arabic_hamzaunderalef إ ARABIC LETTER ALEF WITH HAMZA BELOW */
   {
   0x05c6, 0x0626},             /*           Arabic_hamzaonyeh ئ ARABIC LETTER YEH WITH HAMZA ABOVE */
   {
   0x05c7, 0x0627},             /*                 Arabic_alef ا ARABIC LETTER ALEF */
   {
   0x05c8, 0x0628},             /*                  Arabic_beh ب ARABIC LETTER BEH */
   {
   0x05c9, 0x0629},             /*           Arabic_tehmarbuta ة ARABIC LETTER TEH MARBUTA */
   {
   0x05ca, 0x062a},             /*                  Arabic_teh ت ARABIC LETTER TEH */
   {
   0x05cb, 0x062b},             /*                 Arabic_theh ث ARABIC LETTER THEH */
   {
   0x05cc, 0x062c},             /*                 Arabic_jeem ج ARABIC LETTER JEEM */
   {
   0x05cd, 0x062d},             /*                  Arabic_hah ح ARABIC LETTER HAH */
   {
   0x05ce, 0x062e},             /*                 Arabic_khah خ ARABIC LETTER KHAH */
   {
   0x05cf, 0x062f},             /*                  Arabic_dal د ARABIC LETTER DAL */
   {
   0x05d0, 0x0630},             /*                 Arabic_thal ذ ARABIC LETTER THAL */
   {
   0x05d1, 0x0631},             /*                   Arabic_ra ر ARABIC LETTER REH */
   {
   0x05d2, 0x0632},             /*                 Arabic_zain ز ARABIC LETTER ZAIN */
   {
   0x05d3, 0x0633},             /*                 Arabic_seen س ARABIC LETTER SEEN */
   {
   0x05d4, 0x0634},             /*                Arabic_sheen ش ARABIC LETTER SHEEN */
   {
   0x05d5, 0x0635},             /*                  Arabic_sad ص ARABIC LETTER SAD */
   {
   0x05d6, 0x0636},             /*                  Arabic_dad ض ARABIC LETTER DAD */
   {
   0x05d7, 0x0637},             /*                  Arabic_tah ط ARABIC LETTER TAH */
   {
   0x05d8, 0x0638},             /*                  Arabic_zah ظ ARABIC LETTER ZAH */
   {
   0x05d9, 0x0639},             /*                  Arabic_ain ع ARABIC LETTER AIN */
   {
   0x05da, 0x063a},             /*                Arabic_ghain غ ARABIC LETTER GHAIN */
   {
   0x05e0, 0x0640},             /*              Arabic_tatweel ـ ARABIC TATWEEL */
   {
   0x05e1, 0x0641},             /*                  Arabic_feh ف ARABIC LETTER FEH */
   {
   0x05e2, 0x0642},             /*                  Arabic_qaf ق ARABIC LETTER QAF */
   {
   0x05e3, 0x0643},             /*                  Arabic_kaf ك ARABIC LETTER KAF */
   {
   0x05e4, 0x0644},             /*                  Arabic_lam ل ARABIC LETTER LAM */
   {
   0x05e5, 0x0645},             /*                 Arabic_meem م ARABIC LETTER MEEM */
   {
   0x05e6, 0x0646},             /*                 Arabic_noon ن ARABIC LETTER NOON */
   {
   0x05e7, 0x0647},             /*                   Arabic_ha ه ARABIC LETTER HEH */
   {
   0x05e8, 0x0648},             /*                  Arabic_waw و ARABIC LETTER WAW */
   {
   0x05e9, 0x0649},             /*          Arabic_alefmaksura ى ARABIC LETTER ALEF MAKSURA */
   {
   0x05ea, 0x064a},             /*                  Arabic_yeh ي ARABIC LETTER YEH */
   {
   0x05eb, 0x064b},             /*             Arabic_fathatan ً ARABIC FATHATAN */
   {
   0x05ec, 0x064c},             /*             Arabic_dammatan ٌ ARABIC DAMMATAN */
   {
   0x05ed, 0x064d},             /*             Arabic_kasratan ٍ ARABIC KASRATAN */
   {
   0x05ee, 0x064e},             /*                Arabic_fatha َ ARABIC FATHA */
   {
   0x05ef, 0x064f},             /*                Arabic_damma ُ ARABIC DAMMA */
   {
   0x05f0, 0x0650},             /*                Arabic_kasra ِ ARABIC KASRA */
   {
   0x05f1, 0x0651},             /*               Arabic_shadda ّ ARABIC SHADDA */
   {
   0x05f2, 0x0652},             /*                Arabic_sukun ْ ARABIC SUKUN */
   {
   0x0da1, 0x0e01},             /*                  Thai_kokai ก THAI CHARACTER KO KAI */
   {
   0x0da2, 0x0e02},             /*                Thai_khokhai ข THAI CHARACTER KHO KHAI */
   {
   0x0da3, 0x0e03},             /*               Thai_khokhuat ฃ THAI CHARACTER KHO KHUAT */
   {
   0x0da4, 0x0e04},             /*               Thai_khokhwai ค THAI CHARACTER KHO KHWAI */
   {
   0x0da5, 0x0e05},             /*                Thai_khokhon ฅ THAI CHARACTER KHO KHON */
   {
   0x0da6, 0x0e06},             /*             Thai_khorakhang ฆ THAI CHARACTER KHO RAKHANG */
   {
   0x0da7, 0x0e07},             /*                 Thai_ngongu ง THAI CHARACTER NGO NGU */
   {
   0x0da8, 0x0e08},             /*                Thai_chochan จ THAI CHARACTER CHO CHAN */
   {
   0x0da9, 0x0e09},             /*               Thai_choching ฉ THAI CHARACTER CHO CHING */
   {
   0x0daa, 0x0e0a},             /*               Thai_chochang ช THAI CHARACTER CHO CHANG */
   {
   0x0dab, 0x0e0b},             /*                   Thai_soso ซ THAI CHARACTER SO SO */
   {
   0x0dac, 0x0e0c},             /*                Thai_chochoe ฌ THAI CHARACTER CHO CHOE */
   {
   0x0dad, 0x0e0d},             /*                 Thai_yoying ญ THAI CHARACTER YO YING */
   {
   0x0dae, 0x0e0e},             /*                Thai_dochada ฎ THAI CHARACTER DO CHADA */
   {
   0x0daf, 0x0e0f},             /*                Thai_topatak ฏ THAI CHARACTER TO PATAK */
   {
   0x0db0, 0x0e10},             /*                Thai_thothan ฐ THAI CHARACTER THO THAN */
   {
   0x0db1, 0x0e11},             /*          Thai_thonangmontho ฑ THAI CHARACTER THO NANGMONTHO */
   {
   0x0db2, 0x0e12},             /*             Thai_thophuthao ฒ THAI CHARACTER THO PHUTHAO */
   {
   0x0db3, 0x0e13},             /*                  Thai_nonen ณ THAI CHARACTER NO NEN */
   {
   0x0db4, 0x0e14},             /*                  Thai_dodek ด THAI CHARACTER DO DEK */
   {
   0x0db5, 0x0e15},             /*                  Thai_totao ต THAI CHARACTER TO TAO */
   {
   0x0db6, 0x0e16},             /*               Thai_thothung ถ THAI CHARACTER THO THUNG */
   {
   0x0db7, 0x0e17},             /*              Thai_thothahan ท THAI CHARACTER THO THAHAN */
   {
   0x0db8, 0x0e18},             /*               Thai_thothong ธ THAI CHARACTER THO THONG */
   {
   0x0db9, 0x0e19},             /*                   Thai_nonu น THAI CHARACTER NO NU */
   {
   0x0dba, 0x0e1a},             /*               Thai_bobaimai บ THAI CHARACTER BO BAIMAI */
   {
   0x0dbb, 0x0e1b},             /*                  Thai_popla ป THAI CHARACTER PO PLA */
   {
   0x0dbc, 0x0e1c},             /*               Thai_phophung ผ THAI CHARACTER PHO PHUNG */
   {
   0x0dbd, 0x0e1d},             /*                   Thai_fofa ฝ THAI CHARACTER FO FA */
   {
   0x0dbe, 0x0e1e},             /*                Thai_phophan พ THAI CHARACTER PHO PHAN */
   {
   0x0dbf, 0x0e1f},             /*                  Thai_fofan ฟ THAI CHARACTER FO FAN */
   {
   0x0dc0, 0x0e20},             /*             Thai_phosamphao ภ THAI CHARACTER PHO SAMPHAO */
   {
   0x0dc1, 0x0e21},             /*                   Thai_moma ม THAI CHARACTER MO MA */
   {
   0x0dc2, 0x0e22},             /*                  Thai_yoyak ย THAI CHARACTER YO YAK */
   {
   0x0dc3, 0x0e23},             /*                  Thai_rorua ร THAI CHARACTER RO RUA */
   {
   0x0dc4, 0x0e24},             /*                     Thai_ru ฤ THAI CHARACTER RU */
   {
   0x0dc5, 0x0e25},             /*                 Thai_loling ล THAI CHARACTER LO LING */
   {
   0x0dc6, 0x0e26},             /*                     Thai_lu ฦ THAI CHARACTER LU */
   {
   0x0dc7, 0x0e27},             /*                 Thai_wowaen ว THAI CHARACTER WO WAEN */
   {
   0x0dc8, 0x0e28},             /*                 Thai_sosala ศ THAI CHARACTER SO SALA */
   {
   0x0dc9, 0x0e29},             /*                 Thai_sorusi ษ THAI CHARACTER SO RUSI */
   {
   0x0dca, 0x0e2a},             /*                  Thai_sosua ส THAI CHARACTER SO SUA */
   {
   0x0dcb, 0x0e2b},             /*                  Thai_hohip ห THAI CHARACTER HO HIP */
   {
   0x0dcc, 0x0e2c},             /*                Thai_lochula ฬ THAI CHARACTER LO CHULA */
   {
   0x0dcd, 0x0e2d},             /*                   Thai_oang อ THAI CHARACTER O ANG */
   {
   0x0dce, 0x0e2e},             /*               Thai_honokhuk ฮ THAI CHARACTER HO NOKHUK */
   {
   0x0dcf, 0x0e2f},             /*              Thai_paiyannoi ฯ THAI CHARACTER PAIYANNOI */
   {
   0x0dd0, 0x0e30},             /*                  Thai_saraa ะ THAI CHARACTER SARA A */
   {
   0x0dd1, 0x0e31},             /*             Thai_maihanakat ั THAI CHARACTER MAI HAN-AKAT */
   {
   0x0dd2, 0x0e32},             /*                 Thai_saraaa า THAI CHARACTER SARA AA */
   {
   0x0dd3, 0x0e33},             /*                 Thai_saraam ำ THAI CHARACTER SARA AM */
   {
   0x0dd4, 0x0e34},             /*                  Thai_sarai ิ THAI CHARACTER SARA I */
   {
   0x0dd5, 0x0e35},             /*                 Thai_saraii ี THAI CHARACTER SARA II */
   {
   0x0dd6, 0x0e36},             /*                 Thai_saraue ึ THAI CHARACTER SARA UE */
   {
   0x0dd7, 0x0e37},             /*                Thai_sarauee ื THAI CHARACTER SARA UEE */
   {
   0x0dd8, 0x0e38},             /*                  Thai_sarau ุ THAI CHARACTER SARA U */
   {
   0x0dd9, 0x0e39},             /*                 Thai_sarauu ู THAI CHARACTER SARA UU */
   {
   0x0dda, 0x0e3a},             /*                Thai_phinthu ฺ THAI CHARACTER PHINTHU */
   {
   0x0ddf, 0x0e3f},             /*                   Thai_baht ฿ THAI CURRENCY SYMBOL BAHT */
   {
   0x0de0, 0x0e40},             /*                  Thai_sarae เ THAI CHARACTER SARA E */
   {
   0x0de1, 0x0e41},             /*                 Thai_saraae แ THAI CHARACTER SARA AE */
   {
   0x0de2, 0x0e42},             /*                  Thai_sarao โ THAI CHARACTER SARA O */
   {
   0x0de3, 0x0e43},             /*          Thai_saraaimaimuan ใ THAI CHARACTER SARA AI MAIMUAN */
   {
   0x0de4, 0x0e44},             /*         Thai_saraaimaimalai ไ THAI CHARACTER SARA AI MAIMALAI */
   {
   0x0de5, 0x0e45},             /*            Thai_lakkhangyao ๅ THAI CHARACTER LAKKHANGYAO */
   {
   0x0de6, 0x0e46},             /*               Thai_maiyamok ๆ THAI CHARACTER MAIYAMOK */
   {
   0x0de7, 0x0e47},             /*              Thai_maitaikhu ็ THAI CHARACTER MAITAIKHU */
   {
   0x0de8, 0x0e48},             /*                  Thai_maiek ่ THAI CHARACTER MAI EK */
   {
   0x0de9, 0x0e49},             /*                 Thai_maitho ้ THAI CHARACTER MAI THO */
   {
   0x0dea, 0x0e4a},             /*                 Thai_maitri ๊ THAI CHARACTER MAI TRI */
   {
   0x0deb, 0x0e4b},             /*            Thai_maichattawa ๋ THAI CHARACTER MAI CHATTAWA */
   {
   0x0dec, 0x0e4c},             /*            Thai_thanthakhat ์ THAI CHARACTER THANTHAKHAT */
   {
   0x0ded, 0x0e4d},             /*               Thai_nikhahit ํ THAI CHARACTER NIKHAHIT */
   {
   0x0df0, 0x0e50},             /*                 Thai_leksun ๐ THAI DIGIT ZERO */
   {
   0x0df1, 0x0e51},             /*                Thai_leknung ๑ THAI DIGIT ONE */
   {
   0x0df2, 0x0e52},             /*                Thai_leksong ๒ THAI DIGIT TWO */
   {
   0x0df3, 0x0e53},             /*                 Thai_leksam ๓ THAI DIGIT THREE */
   {
   0x0df4, 0x0e54},             /*                  Thai_leksi ๔ THAI DIGIT FOUR */
   {
   0x0df5, 0x0e55},             /*                  Thai_lekha ๕ THAI DIGIT FIVE */
   {
   0x0df6, 0x0e56},             /*                 Thai_lekhok ๖ THAI DIGIT SIX */
   {
   0x0df7, 0x0e57},             /*                Thai_lekchet ๗ THAI DIGIT SEVEN */
   {
   0x0df8, 0x0e58},             /*                Thai_lekpaet ๘ THAI DIGIT EIGHT */
   {
   0x0df9, 0x0e59},             /*                 Thai_lekkao ๙ THAI DIGIT NINE */
   {
   0x0ed4, 0x11a8},             /*             Hangul_J_Kiyeog ᆨ HANGUL JONGSEONG KIYEOK */
   {
   0x0ed5, 0x11a9},             /*        Hangul_J_SsangKiyeog ᆩ HANGUL JONGSEONG SSANGKIYEOK */
   {
   0x0ed6, 0x11aa},             /*         Hangul_J_KiyeogSios ᆪ HANGUL JONGSEONG KIYEOK-SIOS */
   {
   0x0ed7, 0x11ab},             /*              Hangul_J_Nieun ᆫ HANGUL JONGSEONG NIEUN */
   {
   0x0ed8, 0x11ac},             /*         Hangul_J_NieunJieuj ᆬ HANGUL JONGSEONG NIEUN-CIEUC */
   {
   0x0ed9, 0x11ad},             /*         Hangul_J_NieunHieuh ᆭ HANGUL JONGSEONG NIEUN-HIEUH */
   {
   0x0eda, 0x11ae},             /*             Hangul_J_Dikeud ᆮ HANGUL JONGSEONG TIKEUT */
   {
   0x0edb, 0x11af},             /*              Hangul_J_Rieul ᆯ HANGUL JONGSEONG RIEUL */
   {
   0x0edc, 0x11b0},             /*        Hangul_J_RieulKiyeog ᆰ HANGUL JONGSEONG RIEUL-KIYEOK */
   {
   0x0edd, 0x11b1},             /*         Hangul_J_RieulMieum ᆱ HANGUL JONGSEONG RIEUL-MIEUM */
   {
   0x0ede, 0x11b2},             /*         Hangul_J_RieulPieub ᆲ HANGUL JONGSEONG RIEUL-PIEUP */
   {
   0x0edf, 0x11b3},             /*          Hangul_J_RieulSios ᆳ HANGUL JONGSEONG RIEUL-SIOS */
   {
   0x0ee0, 0x11b4},             /*         Hangul_J_RieulTieut ᆴ HANGUL JONGSEONG RIEUL-THIEUTH */
   {
   0x0ee1, 0x11b5},             /*        Hangul_J_RieulPhieuf ᆵ HANGUL JONGSEONG RIEUL-PHIEUPH */
   {
   0x0ee2, 0x11b6},             /*         Hangul_J_RieulHieuh ᆶ HANGUL JONGSEONG RIEUL-HIEUH */
   {
   0x0ee3, 0x11b7},             /*              Hangul_J_Mieum ᆷ HANGUL JONGSEONG MIEUM */
   {
   0x0ee4, 0x11b8},             /*              Hangul_J_Pieub ᆸ HANGUL JONGSEONG PIEUP */
   {
   0x0ee5, 0x11b9},             /*          Hangul_J_PieubSios ᆹ HANGUL JONGSEONG PIEUP-SIOS */
   {
   0x0ee6, 0x11ba},             /*               Hangul_J_Sios ᆺ HANGUL JONGSEONG SIOS */
   {
   0x0ee7, 0x11bb},             /*          Hangul_J_SsangSios ᆻ HANGUL JONGSEONG SSANGSIOS */
   {
   0x0ee8, 0x11bc},             /*              Hangul_J_Ieung ᆼ HANGUL JONGSEONG IEUNG */
   {
   0x0ee9, 0x11bd},             /*              Hangul_J_Jieuj ᆽ HANGUL JONGSEONG CIEUC */
   {
   0x0eea, 0x11be},             /*              Hangul_J_Cieuc ᆾ HANGUL JONGSEONG CHIEUCH */
   {
   0x0eeb, 0x11bf},             /*             Hangul_J_Khieuq ᆿ HANGUL JONGSEONG KHIEUKH */
   {
   0x0eec, 0x11c0},             /*              Hangul_J_Tieut ᇀ HANGUL JONGSEONG THIEUTH */
   {
   0x0eed, 0x11c1},             /*             Hangul_J_Phieuf ᇁ HANGUL JONGSEONG PHIEUPH */
   {
   0x0eee, 0x11c2},             /*              Hangul_J_Hieuh ᇂ HANGUL JONGSEONG HIEUH */
   {
   0x0ef8, 0x11eb},             /*            Hangul_J_PanSios ᇫ HANGUL JONGSEONG PANSIOS */
   {
   0x0efa, 0x11f9},             /*        Hangul_J_YeorinHieuh ᇹ HANGUL JONGSEONG YEORINHIEUH */
   {
   0x0aa2, 0x2002},             /*                     enspace   EN SPACE */
   {
   0x0aa1, 0x2003},             /*                     emspace   EM SPACE */
   {
   0x0aa3, 0x2004},             /*                    em3space   THREE-PER-EM SPACE */
   {
   0x0aa4, 0x2005},             /*                    em4space   FOUR-PER-EM SPACE */
   {
   0x0aa5, 0x2007},             /*                  digitspace   FIGURE SPACE */
   {
   0x0aa6, 0x2008},             /*                  punctspace   PUNCTUATION SPACE */
   {
   0x0aa7, 0x2009},             /*                   thinspace   THIN SPACE */
   {
   0x0aa8, 0x200a},             /*                   hairspace   HAIR SPACE */
   {
   0x0abb, 0x2012},             /*                     figdash ‒ FIGURE DASH */
   {
   0x0aaa, 0x2013},             /*                      endash – EN DASH */
   {
   0x0aa9, 0x2014},             /*                      emdash — EM DASH */
   {
   0x07af, 0x2015},             /*              Greek_horizbar ― HORIZONTAL BAR */
   {
   0x0cdf, 0x2017},             /*        hebrew_doublelowline ‗ DOUBLE LOW LINE */
   {
   0x0ad0, 0x2018},             /*         leftsinglequotemark ‘ LEFT SINGLE QUOTATION MARK */
   {
   0x0ad1, 0x2019},             /*        rightsinglequotemark ’ RIGHT SINGLE QUOTATION MARK */
   {
   0x0afd, 0x201a},             /*          singlelowquotemark ‚ SINGLE LOW-9 QUOTATION MARK */
   {
   0x0ad2, 0x201c},             /*         leftdoublequotemark “ LEFT DOUBLE QUOTATION MARK */
   {
   0x0ad3, 0x201d},             /*        rightdoublequotemark ” RIGHT DOUBLE QUOTATION MARK */
   {
   0x0afe, 0x201e},             /*          doublelowquotemark „ DOUBLE LOW-9 QUOTATION MARK */
   {
   0x0af1, 0x2020},             /*                      dagger † DAGGER */
   {
   0x0af2, 0x2021},             /*                doubledagger ‡ DOUBLE DAGGER */
   {
   0x0ae6, 0x2022},             /*          enfilledcircbullet • BULLET */
   {
   0x0aae, 0x2026},             /*                    ellipsis … HORIZONTAL ELLIPSIS */
   {
   0x0ad6, 0x2032},             /*                     minutes ′ PRIME */
   {
   0x0ad7, 0x2033},             /*                     seconds ″ DOUBLE PRIME */
   {
   0x0afc, 0x2038},             /*                       caret ‸ CARET */
   {
   0x047e, 0x203e},             /*                    overline ‾ OVERLINE */
   {
   0x20a0, 0x20a0},             /*                     EcuSign ₠ EURO-CURRENCY SIGN */
   {
   0x20a1, 0x20a1},             /*                   ColonSign ₡ COLON SIGN */
   {
   0x20a2, 0x20a2},             /*                CruzeiroSign ₢ CRUZEIRO SIGN */
   {
   0x20a3, 0x20a3},             /*                  FFrancSign ₣ FRENCH FRANC SIGN */
   {
   0x20a4, 0x20a4},             /*                    LiraSign ₤ LIRA SIGN */
   {
   0x20a5, 0x20a5},             /*                    MillSign ₥ MILL SIGN */
   {
   0x20a6, 0x20a6},             /*                   NairaSign ₦ NAIRA SIGN */
   {
   0x20a7, 0x20a7},             /*                  PesetaSign ₧ PESETA SIGN */
   {
   0x20a8, 0x20a8},             /*                   RupeeSign ₨ RUPEE SIGN */
   {
   0x0eff, 0x20a9},             /*                  Korean_Won ₩ WON SIGN */
   {
   0x20a9, 0x20a9},             /*                     WonSign ₩ WON SIGN */
   {
   0x20aa, 0x20aa},             /*               NewSheqelSign ₪ NEW SHEQEL SIGN */
   {
   0x20ab, 0x20ab},             /*                    DongSign ₫ DONG SIGN */
   {
   0x20ac, 0x20ac},             /*                    EuroSign € EURO SIGN */
   {
   0x0ab8, 0x2105},             /*                      careof ℅ CARE OF */
   {
   0x06b0, 0x2116},             /*                  numerosign № NUMERO SIGN */
   {
   0x0afb, 0x2117},             /*         phonographcopyright ℗ SOUND RECORDING COPYRIGHT */
   {
   0x0ad4, 0x211e},             /*                prescription ℞ PRESCRIPTION TAKE */
   {
   0x0ac9, 0x2122},             /*                   trademark ™ TRADE MARK SIGN */
   {
   0x0ab0, 0x2153},             /*                    onethird ⅓ VULGAR FRACTION ONE THIRD */
   {
   0x0ab1, 0x2154},             /*                   twothirds ⅔ VULGAR FRACTION TWO THIRDS */
   {
   0x0ab2, 0x2155},             /*                    onefifth ⅕ VULGAR FRACTION ONE FIFTH */
   {
   0x0ab3, 0x2156},             /*                   twofifths ⅖ VULGAR FRACTION TWO FIFTHS */
   {
   0x0ab4, 0x2157},             /*                 threefifths ⅗ VULGAR FRACTION THREE FIFTHS */
   {
   0x0ab5, 0x2158},             /*                  fourfifths ⅘ VULGAR FRACTION FOUR FIFTHS */
   {
   0x0ab6, 0x2159},             /*                    onesixth ⅙ VULGAR FRACTION ONE SIXTH */
   {
   0x0ab7, 0x215a},             /*                  fivesixths ⅚ VULGAR FRACTION FIVE SIXTHS */
   {
   0x0ac3, 0x215b},             /*                   oneeighth ⅛ VULGAR FRACTION ONE EIGHTH */
   {
   0x0ac4, 0x215c},             /*                threeeighths ⅜ VULGAR FRACTION THREE EIGHTHS */
   {
   0x0ac5, 0x215d},             /*                 fiveeighths ⅝ VULGAR FRACTION FIVE EIGHTHS */
   {
   0x0ac6, 0x215e},             /*                seveneighths ⅞ VULGAR FRACTION SEVEN EIGHTHS */
   {
   0x08fb, 0x2190},             /*                   leftarrow ← LEFTWARDS ARROW */
   {
   0x08fc, 0x2191},             /*                     uparrow ↑ UPWARDS ARROW */
   {
   0x08fd, 0x2192},             /*                  rightarrow → RIGHTWARDS ARROW */
   {
   0x08fe, 0x2193},             /*                   downarrow ↓ DOWNWARDS ARROW */
   {
   0x08ce, 0x21d2},             /*                     implies ⇒ RIGHTWARDS DOUBLE ARROW */
   {
   0x08cd, 0x21d4},             /*                    ifonlyif ⇔ LEFT RIGHT DOUBLE ARROW */
   {
   0x08ef, 0x2202},             /*           partialderivative ∂ PARTIAL DIFFERENTIAL */
   {
   0x08c5, 0x2207},             /*                       nabla ∇ NABLA */
   {
   0x0bca, 0x2218},             /*                         jot ∘ RING OPERATOR */
   {
   0x08d6, 0x221a},             /*                     radical √ SQUARE ROOT */
   {
   0x08c1, 0x221d},             /*                   variation ∝ PROPORTIONAL TO */
   {
   0x08c2, 0x221e},             /*                    infinity ∞ INFINITY */
   {
   0x08de, 0x2227},             /*                  logicaland ∧ LOGICAL AND */
   {
   0x0ba9, 0x2227},             /*                     upcaret ∧ LOGICAL AND */
   {
   0x08df, 0x2228},             /*                   logicalor ∨ LOGICAL OR */
   {
   0x0ba8, 0x2228},             /*                   downcaret ∨ LOGICAL OR */
   {
   0x08dc, 0x2229},             /*                intersection ∩ INTERSECTION */
   {
   0x0bc3, 0x2229},             /*                      upshoe ∩ INTERSECTION */
   {
   0x08dd, 0x222a},             /*                       union ∪ UNION */
   {
   0x0bd6, 0x222a},             /*                    downshoe ∪ UNION */
   {
   0x08bf, 0x222b},             /*                    integral ∫ INTEGRAL */
   {
   0x08c0, 0x2234},             /*                   therefore ∴ THEREFORE */
   {
   0x08c8, 0x2245},             /*                 approximate ≅ APPROXIMATELY EQUAL TO */
   {
   0x08bd, 0x2260},             /*                    notequal ≠ NOT EQUAL TO */
   {
   0x08cf, 0x2261},             /*                   identical ≡ IDENTICAL TO */
   {
   0x08bc, 0x2264},             /*               lessthanequal ≤ LESS-THAN OR EQUAL TO */
   {
   0x08be, 0x2265},             /*            greaterthanequal ≥ GREATER-THAN OR EQUAL TO */
   {
   0x08da, 0x2282},             /*                  includedin ⊂ SUBSET OF */
   {
   0x0bda, 0x2282},             /*                    leftshoe ⊂ SUBSET OF */
   {
   0x08db, 0x2283},             /*                    includes ⊃ SUPERSET OF */
   {
   0x0bd8, 0x2283},             /*                   rightshoe ⊃ SUPERSET OF */
   {
   0x0bfc, 0x22a2},             /*                   righttack ⊢ RIGHT TACK */
   {
   0x0bdc, 0x22a3},             /*                    lefttack ⊣ LEFT TACK */
   {
   0x0bc2, 0x22a4},             /*                    downtack ⊤ DOWN TACK */
   {
   0x0bce, 0x22a5},             /*                      uptack ⊥ UP TACK */
   {
   0x0bd3, 0x2308},             /*                     upstile ⌈ LEFT CEILING */
   {
   0x0bc4, 0x230a},             /*                   downstile ⌊ LEFT FLOOR */
   {
   0x0afa, 0x2315},             /*           telephonerecorder ⌕ TELEPHONE RECORDER */
   {
   0x08a4, 0x2320},             /*                 topintegral ⌠ TOP HALF INTEGRAL */
   {
   0x08a5, 0x2321},             /*                 botintegral ⌡ BOTTOM HALF INTEGRAL */
   {
   0x0abc, 0x2329},             /*            leftanglebracket 〈 LEFT-POINTING ANGLE BRACKET */
   {
   0x0abe, 0x232a},             /*           rightanglebracket 〉 RIGHT-POINTING ANGLE BRACKET */
   {
   0x0bcc, 0x2395},             /*                        quad ⎕ APL FUNCTIONAL SYMBOL QUAD (Unicode 3.0) */
   {
   0x09e2, 0x2409},             /*                          ht ␉ SYMBOL FOR HORIZONTAL TABULATION */
   {
   0x09e5, 0x240a},             /*                          lf ␊ SYMBOL FOR LINE FEED */
   {
   0x09e9, 0x240b},             /*                          vt ␋ SYMBOL FOR VERTICAL TABULATION */
   {
   0x09e3, 0x240c},             /*                          ff ␌ SYMBOL FOR FORM FEED */
   {
   0x09e4, 0x240d},             /*                          cr ␍ SYMBOL FOR CARRIAGE RETURN */
   {
   0x09df, 0x2422},             /*                       blank ␢ BLANK SYMBOL */
   {
   0x09e8, 0x2424},             /*                          nl ␤ SYMBOL FOR NEWLINE */
   {
   0x09f1, 0x2500},             /*              horizlinescan5 ─ BOX DRAWINGS LIGHT HORIZONTAL */
   {
   0x08a6, 0x2502},             /*               vertconnector │ BOX DRAWINGS LIGHT VERTICAL */
   {
   0x09f8, 0x2502},             /*                     vertbar │ BOX DRAWINGS LIGHT VERTICAL */
   {
   0x09ec, 0x250c},             /*                upleftcorner ┌ BOX DRAWINGS LIGHT DOWN AND RIGHT */
   {
   0x09eb, 0x2510},             /*               uprightcorner ┐ BOX DRAWINGS LIGHT DOWN AND LEFT */
   {
   0x09ed, 0x2514},             /*               lowleftcorner └ BOX DRAWINGS LIGHT UP AND RIGHT */
   {
   0x09ea, 0x2518},             /*              lowrightcorner ┘ BOX DRAWINGS LIGHT UP AND LEFT */
   {
   0x09f4, 0x251c},             /*                       leftt ├ BOX DRAWINGS LIGHT VERTICAL AND RIGHT */
   {
   0x09f5, 0x2524},             /*                      rightt ┤ BOX DRAWINGS LIGHT VERTICAL AND LEFT */
   {
   0x09f7, 0x252c},             /*                        topt ┬ BOX DRAWINGS LIGHT DOWN AND HORIZONTAL */
   {
   0x09f6, 0x2534},             /*                        bott ┴ BOX DRAWINGS LIGHT UP AND HORIZONTAL */
   {
   0x09ee, 0x253c},             /*               crossinglines ┼ BOX DRAWINGS LIGHT VERTICAL AND HORIZONTAL */
   {
   0x09e1, 0x2592},             /*                checkerboard ▒ MEDIUM SHADE */
   {
   0x0adf, 0x25a0},             /*                emfilledrect ■ BLACK SQUARE */
   {
   0x0acf, 0x25a1},             /*             emopenrectangle □ WHITE SQUARE */
   {
   0x0ae7, 0x25aa},             /*            enfilledsqbullet ▪ BLACK SMALL SQUARE */
   {
   0x0ae1, 0x25ab},             /*          enopensquarebullet ▫ WHITE SMALL SQUARE */
   {
   0x0adb, 0x25ac},             /*            filledrectbullet ▬ BLACK RECTANGLE */
   {
   0x0ae2, 0x25ad},             /*              openrectbullet ▭ WHITE RECTANGLE */
   {
   0x0ae8, 0x25b2},             /*           filledtribulletup ▲ BLACK UP-POINTING TRIANGLE */
   {
   0x0ae3, 0x25b3},             /*             opentribulletup △ WHITE UP-POINTING TRIANGLE */
   {
   0x0add, 0x25b6},             /*        filledrighttribullet ▶ BLACK RIGHT-POINTING TRIANGLE */
   {
   0x0acd, 0x25b7},             /*           rightopentriangle ▷ WHITE RIGHT-POINTING TRIANGLE */
   {
   0x0ae9, 0x25bc},             /*         filledtribulletdown ▼ BLACK DOWN-POINTING TRIANGLE */
   {
   0x0ae4, 0x25bd},             /*           opentribulletdown ▽ WHITE DOWN-POINTING TRIANGLE */
   {
   0x0adc, 0x25c0},             /*         filledlefttribullet ◀ BLACK LEFT-POINTING TRIANGLE */
   {
   0x0acc, 0x25c1},             /*            leftopentriangle ◁ WHITE LEFT-POINTING TRIANGLE */
   {
   0x09e0, 0x25c6},             /*                soliddiamond ◆ BLACK DIAMOND */
   {
   0x0ace, 0x25cb},             /*                emopencircle ○ WHITE CIRCLE */
   {
   0x0bcf, 0x25cb},             /*                      circle ○ WHITE CIRCLE */
   {
   0x0ade, 0x25cf},             /*              emfilledcircle ● BLACK CIRCLE */
   {
   0x0ae0, 0x25e6},             /*            enopencircbullet ◦ WHITE BULLET */
   {
   0x0ae5, 0x2606},             /*                    openstar ☆ WHITE STAR */
   {
   0x0af9, 0x260e},             /*                   telephone ☎ BLACK TELEPHONE */
   {
   0x0aca, 0x2613},             /*               signaturemark ☓ SALTIRE */
   {
   0x0aea, 0x261c},             /*                 leftpointer ☜ WHITE LEFT POINTING INDEX */
   {
   0x0aeb, 0x261e},             /*                rightpointer ☞ WHITE RIGHT POINTING INDEX */
   {
   0x0af8, 0x2640},             /*                femalesymbol ♀ FEMALE SIGN */
   {
   0x0af7, 0x2642},             /*                  malesymbol ♂ MALE SIGN */
   {
   0x0aec, 0x2663},             /*                        club ♣ BLACK CLUB SUIT */
   {
   0x0aee, 0x2665},             /*                       heart ♥ BLACK HEART SUIT */
   {
   0x0aed, 0x2666},             /*                     diamond ♦ BLACK DIAMOND SUIT */
   {
   0x0af6, 0x266d},             /*                 musicalflat ♭ MUSIC FLAT SIGN */
   {
   0x0af5, 0x266f},             /*                musicalsharp ♯ MUSIC SHARP SIGN */
   {
   0x0af3, 0x2713},             /*                   checkmark ✓ CHECK MARK */
   {
   0x0af4, 0x2717},             /*                 ballotcross ✗ BALLOT X */
   {
   0x0ad9, 0x271d},             /*                  latincross ✝ LATIN CROSS */
   {
   0x0af0, 0x2720},             /*                maltesecross ✠ MALTESE CROSS */
   {
   0x04a4, 0x3001},             /*                  kana_comma 、 IDEOGRAPHIC COMMA */
   {
   0x04a1, 0x3002},             /*               kana_fullstop 。 IDEOGRAPHIC FULL STOP */
   {
   0x04a2, 0x300c},             /*         kana_openingbracket 「 LEFT CORNER BRACKET */
   {
   0x04a3, 0x300d},             /*         kana_closingbracket 」 RIGHT CORNER BRACKET */
   {
   0x04de, 0x309b},             /*                 voicedsound ゛ KATAKANA-HIRAGANA VOICED SOUND MARK */
   {
   0x04df, 0x309c},             /*             semivoicedsound ゜ KATAKANA-HIRAGANA SEMI-VOICED SOUND MARK */
   {
   0x04a7, 0x30a1},             /*                      kana_a ァ KATAKANA LETTER SMALL A */
   {
   0x04b1, 0x30a2},             /*                      kana_A ア KATAKANA LETTER A */
   {
   0x04a8, 0x30a3},             /*                      kana_i ィ KATAKANA LETTER SMALL I */
   {
   0x04b2, 0x30a4},             /*                      kana_I イ KATAKANA LETTER I */
   {
   0x04a9, 0x30a5},             /*                      kana_u ゥ KATAKANA LETTER SMALL U */
   {
   0x04b3, 0x30a6},             /*                      kana_U ウ KATAKANA LETTER U */
   {
   0x04aa, 0x30a7},             /*                      kana_e ェ KATAKANA LETTER SMALL E */
   {
   0x04b4, 0x30a8},             /*                      kana_E エ KATAKANA LETTER E */
   {
   0x04ab, 0x30a9},             /*                      kana_o ォ KATAKANA LETTER SMALL O */
   {
   0x04b5, 0x30aa},             /*                      kana_O オ KATAKANA LETTER O */
   {
   0x04b6, 0x30ab},             /*                     kana_KA カ KATAKANA LETTER KA */
   {
   0x04b7, 0x30ad},             /*                     kana_KI キ KATAKANA LETTER KI */
   {
   0x04b8, 0x30af},             /*                     kana_KU ク KATAKANA LETTER KU */
   {
   0x04b9, 0x30b1},             /*                     kana_KE ケ KATAKANA LETTER KE */
   {
   0x04ba, 0x30b3},             /*                     kana_KO コ KATAKANA LETTER KO */
   {
   0x04bb, 0x30b5},             /*                     kana_SA サ KATAKANA LETTER SA */
   {
   0x04bc, 0x30b7},             /*                    kana_SHI シ KATAKANA LETTER SI */
   {
   0x04bd, 0x30b9},             /*                     kana_SU ス KATAKANA LETTER SU */
   {
   0x04be, 0x30bb},             /*                     kana_SE セ KATAKANA LETTER SE */
   {
   0x04bf, 0x30bd},             /*                     kana_SO ソ KATAKANA LETTER SO */
   {
   0x04c0, 0x30bf},             /*                     kana_TA タ KATAKANA LETTER TA */
   {
   0x04c1, 0x30c1},             /*                    kana_CHI チ KATAKANA LETTER TI */
   {
   0x04af, 0x30c3},             /*                    kana_tsu ッ KATAKANA LETTER SMALL TU */
   {
   0x04c2, 0x30c4},             /*                    kana_TSU ツ KATAKANA LETTER TU */
   {
   0x04c3, 0x30c6},             /*                     kana_TE テ KATAKANA LETTER TE */
   {
   0x04c4, 0x30c8},             /*                     kana_TO ト KATAKANA LETTER TO */
   {
   0x04c5, 0x30ca},             /*                     kana_NA ナ KATAKANA LETTER NA */
   {
   0x04c6, 0x30cb},             /*                     kana_NI ニ KATAKANA LETTER NI */
   {
   0x04c7, 0x30cc},             /*                     kana_NU ヌ KATAKANA LETTER NU */
   {
   0x04c8, 0x30cd},             /*                     kana_NE ネ KATAKANA LETTER NE */
   {
   0x04c9, 0x30ce},             /*                     kana_NO ノ KATAKANA LETTER NO */
   {
   0x04ca, 0x30cf},             /*                     kana_HA ハ KATAKANA LETTER HA */
   {
   0x04cb, 0x30d2},             /*                     kana_HI ヒ KATAKANA LETTER HI */
   {
   0x04cc, 0x30d5},             /*                     kana_FU フ KATAKANA LETTER HU */
   {
   0x04cd, 0x30d8},             /*                     kana_HE ヘ KATAKANA LETTER HE */
   {
   0x04ce, 0x30db},             /*                     kana_HO ホ KATAKANA LETTER HO */
   {
   0x04cf, 0x30de},             /*                     kana_MA マ KATAKANA LETTER MA */
   {
   0x04d0, 0x30df},             /*                     kana_MI ミ KATAKANA LETTER MI */
   {
   0x04d1, 0x30e0},             /*                     kana_MU ム KATAKANA LETTER MU */
   {
   0x04d2, 0x30e1},             /*                     kana_ME メ KATAKANA LETTER ME */
   {
   0x04d3, 0x30e2},             /*                     kana_MO モ KATAKANA LETTER MO */
   {
   0x04ac, 0x30e3},             /*                     kana_ya ャ KATAKANA LETTER SMALL YA */
   {
   0x04d4, 0x30e4},             /*                     kana_YA ヤ KATAKANA LETTER YA */
   {
   0x04ad, 0x30e5},             /*                     kana_yu ュ KATAKANA LETTER SMALL YU */
   {
   0x04d5, 0x30e6},             /*                     kana_YU ユ KATAKANA LETTER YU */
   {
   0x04ae, 0x30e7},             /*                     kana_yo ョ KATAKANA LETTER SMALL YO */
   {
   0x04d6, 0x30e8},             /*                     kana_YO ヨ KATAKANA LETTER YO */
   {
   0x04d7, 0x30e9},             /*                     kana_RA ラ KATAKANA LETTER RA */
   {
   0x04d8, 0x30ea},             /*                     kana_RI リ KATAKANA LETTER RI */
   {
   0x04d9, 0x30eb},             /*                     kana_RU ル KATAKANA LETTER RU */
   {
   0x04da, 0x30ec},             /*                     kana_RE レ KATAKANA LETTER RE */
   {
   0x04db, 0x30ed},             /*                     kana_RO ロ KATAKANA LETTER RO */
   {
   0x04dc, 0x30ef},             /*                     kana_WA ワ KATAKANA LETTER WA */
   {
   0x04a6, 0x30f2},             /*                     kana_WO ヲ KATAKANA LETTER WO */
   {
   0x04dd, 0x30f3},             /*                      kana_N ン KATAKANA LETTER N */
   {
   0x04a5, 0x30fb},             /*            kana_conjunctive ・ KATAKANA MIDDLE DOT */
   {
   0x04b0, 0x30fc},             /*              prolongedsound ー KATAKANA-HIRAGANA PROLONGED SOUND MARK */
   {
   0x0ea1, 0x3131},             /*               Hangul_Kiyeog ㄱ HANGUL LETTER KIYEOK */
   {
   0x0ea2, 0x3132},             /*          Hangul_SsangKiyeog ㄲ HANGUL LETTER SSANGKIYEOK */
   {
   0x0ea3, 0x3133},             /*           Hangul_KiyeogSios ㄳ HANGUL LETTER KIYEOK-SIOS */
   {
   0x0ea4, 0x3134},             /*                Hangul_Nieun ㄴ HANGUL LETTER NIEUN */
   {
   0x0ea5, 0x3135},             /*           Hangul_NieunJieuj ㄵ HANGUL LETTER NIEUN-CIEUC */
   {
   0x0ea6, 0x3136},             /*           Hangul_NieunHieuh ㄶ HANGUL LETTER NIEUN-HIEUH */
   {
   0x0ea7, 0x3137},             /*               Hangul_Dikeud ㄷ HANGUL LETTER TIKEUT */
   {
   0x0ea8, 0x3138},             /*          Hangul_SsangDikeud ㄸ HANGUL LETTER SSANGTIKEUT */
   {
   0x0ea9, 0x3139},             /*                Hangul_Rieul ㄹ HANGUL LETTER RIEUL */
   {
   0x0eaa, 0x313a},             /*          Hangul_RieulKiyeog ㄺ HANGUL LETTER RIEUL-KIYEOK */
   {
   0x0eab, 0x313b},             /*           Hangul_RieulMieum ㄻ HANGUL LETTER RIEUL-MIEUM */
   {
   0x0eac, 0x313c},             /*           Hangul_RieulPieub ㄼ HANGUL LETTER RIEUL-PIEUP */
   {
   0x0ead, 0x313d},             /*            Hangul_RieulSios ㄽ HANGUL LETTER RIEUL-SIOS */
   {
   0x0eae, 0x313e},             /*           Hangul_RieulTieut ㄾ HANGUL LETTER RIEUL-THIEUTH */
   {
   0x0eaf, 0x313f},             /*          Hangul_RieulPhieuf ㄿ HANGUL LETTER RIEUL-PHIEUPH */
   {
   0x0eb0, 0x3140},             /*           Hangul_RieulHieuh ㅀ HANGUL LETTER RIEUL-HIEUH */
   {
   0x0eb1, 0x3141},             /*                Hangul_Mieum ㅁ HANGUL LETTER MIEUM */
   {
   0x0eb2, 0x3142},             /*                Hangul_Pieub ㅂ HANGUL LETTER PIEUP */
   {
   0x0eb3, 0x3143},             /*           Hangul_SsangPieub ㅃ HANGUL LETTER SSANGPIEUP */
   {
   0x0eb4, 0x3144},             /*            Hangul_PieubSios ㅄ HANGUL LETTER PIEUP-SIOS */
   {
   0x0eb5, 0x3145},             /*                 Hangul_Sios ㅅ HANGUL LETTER SIOS */
   {
   0x0eb6, 0x3146},             /*            Hangul_SsangSios ㅆ HANGUL LETTER SSANGSIOS */
   {
   0x0eb7, 0x3147},             /*                Hangul_Ieung ㅇ HANGUL LETTER IEUNG */
   {
   0x0eb8, 0x3148},             /*                Hangul_Jieuj ㅈ HANGUL LETTER CIEUC */
   {
   0x0eb9, 0x3149},             /*           Hangul_SsangJieuj ㅉ HANGUL LETTER SSANGCIEUC */
   {
   0x0eba, 0x314a},             /*                Hangul_Cieuc ㅊ HANGUL LETTER CHIEUCH */
   {
   0x0ebb, 0x314b},             /*               Hangul_Khieuq ㅋ HANGUL LETTER KHIEUKH */
   {
   0x0ebc, 0x314c},             /*                Hangul_Tieut ㅌ HANGUL LETTER THIEUTH */
   {
   0x0ebd, 0x314d},             /*               Hangul_Phieuf ㅍ HANGUL LETTER PHIEUPH */
   {
   0x0ebe, 0x314e},             /*                Hangul_Hieuh ㅎ HANGUL LETTER HIEUH */
   {
   0x0ebf, 0x314f},             /*                    Hangul_A ㅏ HANGUL LETTER A */
   {
   0x0ec0, 0x3150},             /*                   Hangul_AE ㅐ HANGUL LETTER AE */
   {
   0x0ec1, 0x3151},             /*                   Hangul_YA ㅑ HANGUL LETTER YA */
   {
   0x0ec2, 0x3152},             /*                  Hangul_YAE ㅒ HANGUL LETTER YAE */
   {
   0x0ec3, 0x3153},             /*                   Hangul_EO ㅓ HANGUL LETTER EO */
   {
   0x0ec4, 0x3154},             /*                    Hangul_E ㅔ HANGUL LETTER E */
   {
   0x0ec5, 0x3155},             /*                  Hangul_YEO ㅕ HANGUL LETTER YEO */
   {
   0x0ec6, 0x3156},             /*                   Hangul_YE ㅖ HANGUL LETTER YE */
   {
   0x0ec7, 0x3157},             /*                    Hangul_O ㅗ HANGUL LETTER O */
   {
   0x0ec8, 0x3158},             /*                   Hangul_WA ㅘ HANGUL LETTER WA */
   {
   0x0ec9, 0x3159},             /*                  Hangul_WAE ㅙ HANGUL LETTER WAE */
   {
   0x0eca, 0x315a},             /*                   Hangul_OE ㅚ HANGUL LETTER OE */
   {
   0x0ecb, 0x315b},             /*                   Hangul_YO ㅛ HANGUL LETTER YO */
   {
   0x0ecc, 0x315c},             /*                    Hangul_U ㅜ HANGUL LETTER U */
   {
   0x0ecd, 0x315d},             /*                  Hangul_WEO ㅝ HANGUL LETTER WEO */
   {
   0x0ece, 0x315e},             /*                   Hangul_WE ㅞ HANGUL LETTER WE */
   {
   0x0ecf, 0x315f},             /*                   Hangul_WI ㅟ HANGUL LETTER WI */
   {
   0x0ed0, 0x3160},             /*                   Hangul_YU ㅠ HANGUL LETTER YU */
   {
   0x0ed1, 0x3161},             /*                   Hangul_EU ㅡ HANGUL LETTER EU */
   {
   0x0ed2, 0x3162},             /*                   Hangul_YI ㅢ HANGUL LETTER YI */
   {
   0x0ed3, 0x3163},             /*                    Hangul_I ㅣ HANGUL LETTER I */
   {
   0x0eef, 0x316d},             /*     Hangul_RieulYeorinHieuh ㅭ HANGUL LETTER RIEUL-YEORINHIEUH */
   {
   0x0ef0, 0x3171},             /*    Hangul_SunkyeongeumMieum ㅱ HANGUL LETTER KAPYEOUNMIEUM */
   {
   0x0ef1, 0x3178},             /*    Hangul_SunkyeongeumPieub ㅸ HANGUL LETTER KAPYEOUNPIEUP */
   {
   0x0ef2, 0x317f},             /*              Hangul_PanSios ㅿ HANGUL LETTER PANSIOS */
   {
   0x0ef4, 0x3184},             /*   Hangul_SunkyeongeumPhieuf ㆄ HANGUL LETTER KAPYEOUNPHIEUPH */
   {
   0x0ef5, 0x3186},             /*          Hangul_YeorinHieuh ㆆ HANGUL LETTER YEORINHIEUH */
   {
   0x0ef6, 0x318d},             /*                Hangul_AraeA ㆍ HANGUL LETTER ARAEA */
   {
   0x0ef7, 0x318e},             /*               Hangul_AraeAE ㆎ HANGUL LETTER ARAEAE */
};

static guint unicode_to_keyval(wchar_t ucs)
{
   int min = 0;
   int max = sizeof(u2ktab) / sizeof(u2ktab[0]) - 1;
   int mid;

   /* First check for Latin-1 characters (1:1 mapping) */
   if ((ucs >= 0x0020 && ucs <= 0x007e) ||
       (ucs >= 0x00a0 && ucs <= 0x00ff))
      return ucs;

   /* Binary search in table */
   while (max >= min) {
      mid = (min + max) / 2;
      if (u2ktab[mid].ucs < ucs)
         min = mid + 1;
      else if (u2ktab[mid].ucs > ucs)
         max = mid - 1;
      else {
         /* found it */
         return u2ktab[mid].keysym;
      }
   }

   /*
    * No matching keysym value found, return Unicode value plus 0x01000000
    * (a convention introduced in the UTF-8 work on xterm).
    */
   return ucs | 0x01000000;
}

static void build_key_event_state(GdkEvent * event)
{
   if (GetKeyState(VK_SHIFT) < 0)
      event->key.state |= GDK_SHIFT_MASK;
   if (GetKeyState(VK_CAPITAL) & 0x1)
      event->key.state |= GDK_LOCK_MASK;
   if (!is_AltGr_key) {
      if (GetKeyState(VK_CONTROL) < 0) {
         event->key.state |= GDK_CONTROL_MASK;
#if 0
         if (event->key.keyval < ' ')
            event->key.keyval += '@';
#endif
      }
#if 0
      /* We never get here (?) */
      else if (event->key.keyval < ' ') {
         event->key.state |= GDK_CONTROL_MASK;
         event->key.keyval += '@';
      }
#endif
      if (GetKeyState(VK_MENU) < 0) {
         event->key.state |= GDK_MOD1_MASK;
      }
   }
}

static gint build_pointer_event_state(MSG * xevent)
{
   gint state;

   state = 0;
   if (xevent->wParam & MK_CONTROL)
      state |= GDK_CONTROL_MASK;
   if (xevent->wParam & MK_LBUTTON)
      state |= GDK_BUTTON1_MASK;
   if (xevent->wParam & MK_MBUTTON)
      state |= GDK_BUTTON2_MASK;
   if (xevent->wParam & MK_RBUTTON)
      state |= GDK_BUTTON3_MASK;
   if (xevent->wParam & MK_SHIFT)
      state |= GDK_SHIFT_MASK;
   if (GetKeyState(VK_MENU) < 0)
      state |= GDK_MOD1_MASK;
   if (GetKeyState(VK_CAPITAL) & 0x1)
      state |= GDK_LOCK_MASK;

   return state;
}
//vo
static gint get_key_value(MSG * xevent)
{
   gint ret = GDK_VoidSymbol;

   if (xevent->wParam < 0x20) {
      ret = (xevent->wParam + 0x60)  & 0xFF;
   } else {
      if (xevent->wParam & 0xFF00) {
         ret = ((xevent->wParam >> 8) & 0xFF);
      } else {
         ret = (xevent->wParam & 0xFF);
      }
   }
   return ret;
}

static void
build_keypress_event(GdkWindowWin32Data * windata,
                     GdkEvent * event, MSG * xevent)
{
   HIMC hIMC;
   gint i, bytecount, ucount, ucleft, len;
   guchar buf[100], *bp;
   wchar_t wbuf[100], *wcp;

   event->type = GDK_KEY_PRESS;
   event->any.type = GDK_KEY_PRESS;
   event->key.type = GDK_KEY_PRESS;
   event->key.time = xevent->time;
   event->key.state = 0;

   if (xevent->message == WM_IME_COMPOSITION) {
      hIMC = ImmGetContext(xevent->hwnd);

      bytecount = ImmGetCompositionStringW(hIMC, GCS_RESULTSTR,
                                           wbuf, sizeof(wbuf));
      ucount = bytecount / 2;
   } else {
      if (xevent->message == WM_CHAR || xevent->message == WM_SYSCHAR) {
         bytecount = MIN((xevent->lParam & 0xFFFF), sizeof(buf));
         for (i = 0; i < bytecount; i++)
            buf[i] = xevent->wParam;
      } else {                  /* WM_IME_CHAR */

         event->key.keyval = GDK_VoidSymbol;
         if (xevent->wParam & 0xFF00) {
            /* Contrary to some versions of the documentation,
             * the lead byte is the most significant byte.
             */
            buf[0] = ((xevent->wParam >> 8) & 0xFF);
            buf[1] = (xevent->wParam & 0xFF);
            bytecount = 2;
         } else {
            buf[0] = (xevent->wParam & 0xFF);
            bytecount = 1;
         }
      }

      /* Convert from the window's current code page
       * to Unicode. Then convert to UTF-8.
       * We don't handle the surrogate stuff. Should we?
       */
      ucount = MultiByteToWideChar(windata->charset_info.ciACP,
                                   0, buf, bytecount,
                                   wbuf, sizeof(wbuf) / sizeof(wbuf[0]));

   }
   if (ucount == 0)
      event->key.keyval = GDK_VoidSymbol;
   else if (xevent->message == WM_CHAR || xevent->message == WM_SYSCHAR)
      if (xevent->wParam < ' ') {
         event->key.keyval = xevent->wParam + '@';
         /* This is needed in case of Alt+nnn or Alt+0nnn (on the numpad)
          * where nnn<32
          */
         event->key.state |= GDK_CONTROL_MASK;
      } else
         event->key.keyval = unicode_to_keyval(wbuf[0]);

   build_key_event_state(event);

   /* Build UTF-8 string */
   ucleft = ucount;
   len = 0;
   wcp = wbuf;
   while (ucleft-- > 0) {
      wchar_t c = *wcp++;

      if (c < 0x80)
         len += 1;
      else if (c < 0x800)
         len += 2;
      else
         len += 3;
   }

   event->key.string = g_malloc(len + 1);
   event->key.length = len;

   ucleft = ucount;
   wcp = wbuf;
   bp = event->key.string;
   while (ucleft-- > 0) {
      int first;
      wchar_t c = *wcp++;

      if (c < 0x80) {
         first = 0;
         len = 1;
      } else if (c < 0x800) {
         first = 0xc0;
         len = 2;
      } else {
         first = 0xe0;
         len = 3;
      }

#if 1
      /* Woo-hoo! */
      switch (len) {
      case 3:
         bp[2] = (c & 0x3f) | 0x80;
         c >>= 6;               /* Fall through */
      case 2:
         bp[1] = (c & 0x3f) | 0x80;
         c >>= 6;               /* Fall through */
      case 1:
         bp[0] = c | first;
      }
#else
      for (i = len - 1; i > 0; --i) {
         bp[i] = (c & 0x3f) | 0x80;
         c >>= 6;
      }
      bp[0] = c | first;
#endif

      bp += len;
   }
   *bp = 0;
}

static void
build_keyrelease_event(GdkWindowWin32Data * windata,
                       GdkEvent * event, MSG * xevent)
{
   guchar buf;
   wchar_t wbuf;

   event->type = GDK_KEY_RELEASE;
   event->key.type = GDK_KEY_RELEASE;
   event->key.time = xevent->time;
   event->key.state = 0;

   if (xevent->message == WM_CHAR || xevent->message == WM_SYSCHAR)
      if (xevent->wParam < ' ')
         event->key.keyval = xevent->wParam + '@';
      else {
         buf = xevent->wParam;
         MultiByteToWideChar(windata->charset_info.ciACP,
                             0, &buf, 1, &wbuf, 1);

         event->key.keyval = unicode_to_keyval(wbuf);
   } else
      event->key.keyval = GDK_VoidSymbol;
   build_key_event_state(event);
   event->key.string = NULL;
   event->key.length = 0;
   k_grab_window = 0; //vo
}

static void print_event_state(gint state)
{
   if (state & GDK_SHIFT_MASK)
      g_print("SHIFT ");
   if (state & GDK_LOCK_MASK)
      g_print("LOCK ");
   if (state & GDK_CONTROL_MASK)
      g_print("CONTROL ");
   if (state & GDK_MOD1_MASK)
      g_print("MOD1 ");
   if (state & GDK_BUTTON1_MASK)
      g_print("BUTTON1 ");
   if (state & GDK_BUTTON2_MASK)
      g_print("BUTTON2 ");
   if (state & GDK_BUTTON3_MASK)
      g_print("BUTTON3 ");
}

static void print_event(GdkEvent * event)
{
   gchar *escaped, *kvname;

   switch (event->any.type) {
   case GDK_NOTHING:
      g_print("GDK_NOTHING ");
      break;
   case GDK_DELETE:
      g_print("GDK_DELETE ");
      break;
   case GDK_DESTROY:
      g_print("GDK_DESTROY ");
      break;
   case GDK_EXPOSE:
      g_print("GDK_EXPOSE ");
      break;
   case GDK_MOTION_NOTIFY:
      g_print("GDK_MOTION_NOTIFY ");
      break;
   case GDK_BUTTON_PRESS:
      g_print("GDK_BUTTON_PRESS ");
      break;
   case GDK_2BUTTON_PRESS:
      g_print("GDK_2BUTTON_PRESS ");
      break;
   case GDK_3BUTTON_PRESS:
      g_print("GDK_3BUTTON_PRESS ");
      break;
   case GDK_BUTTON_RELEASE:
      g_print("GDK_BUTTON_RELEASE ");
      break;
   case GDK_KEY_PRESS:
      g_print("GDK_KEY_PRESS ");
      break;
   case GDK_KEY_RELEASE:
      g_print("GDK_KEY_RELEASE ");
      break;
   case GDK_ENTER_NOTIFY:
      g_print("GDK_ENTER_NOTIFY ");
      break;
   case GDK_LEAVE_NOTIFY:
      g_print("GDK_LEAVE_NOTIFY ");
      break;
   case GDK_FOCUS_CHANGE:
      g_print("GDK_FOCUS_CHANGE ");
      break;
   case GDK_CONFIGURE:
      g_print("GDK_CONFIGURE ");
      break;
   case GDK_MAP:
      g_print("GDK_MAP ");
      break;
   case GDK_UNMAP:
      g_print("GDK_UNMAP ");
      break;
   case GDK_PROPERTY_NOTIFY:
      g_print("GDK_PROPERTY_NOTIFY ");
      break;
   case GDK_SELECTION_CLEAR:
      g_print("GDK_SELECTION_CLEAR ");
      break;
   case GDK_SELECTION_REQUEST:
      g_print("GDK_SELECTION_REQUEST ");
      break;
   case GDK_SELECTION_NOTIFY:
      g_print("GDK_SELECTION_NOTIFY ");
      break;
   case GDK_PROXIMITY_IN:
      g_print("GDK_PROXIMITY_IN ");
      break;
   case GDK_PROXIMITY_OUT:
      g_print("GDK_PROXIMITY_OUT ");
      break;
   case GDK_DRAG_ENTER:
      g_print("GDK_DRAG_ENTER ");
      break;
   case GDK_DRAG_LEAVE:
      g_print("GDK_DRAG_LEAVE ");
      break;
   case GDK_DRAG_MOTION:
      g_print("GDK_DRAG_MOTION ");
      break;
   case GDK_DRAG_STATUS:
      g_print("GDK_DRAG_STATUS ");
      break;
   case GDK_DROP_START:
      g_print("GDK_DROP_START ");
      break;
   case GDK_DROP_FINISHED:
      g_print("GDK_DROP_FINISHED ");
      break;
   case GDK_CLIENT_EVENT:
      g_print("GDK_CLIENT_EVENT ");
      break;
   case GDK_VISIBILITY_NOTIFY:
      g_print("GDK_VISIBILITY_NOTIFY ");
      break;
   case GDK_NO_EXPOSE:
      g_print("GDK_NO_EXPOSE ");
      break;
   case GDK_SCROLL:
      g_print("GDK_SCROLL ");
      break;
   }
   g_print("%#x ", GDK_DRAWABLE_XID(event->any.window));

   switch (event->any.type) {
   case GDK_EXPOSE:
      g_print("%dx%d@+%d+%d %d",
              event->expose.area.width,
              event->expose.area.height,
              event->expose.area.x,
              event->expose.area.y, event->expose.count);
      break;
   case GDK_MOTION_NOTIFY:
      g_print("(%.4g,%.4g) %s",
              event->motion.x, event->motion.y,
              event->motion.is_hint ? "HINT " : "");
      print_event_state(event->motion.state);
      break;
   case GDK_BUTTON_PRESS:
   case GDK_2BUTTON_PRESS:
   case GDK_3BUTTON_PRESS:
   case GDK_BUTTON_RELEASE:
      g_print("%d (%.4g,%.4g) ",
              event->button.button, event->button.x, event->button.y);
      print_event_state(event->button.state);
      break;
   case GDK_KEY_PRESS:
   case GDK_KEY_RELEASE:
      if (event->key.length == 0)
         escaped = g_strdup("");
      else
         escaped = g_strescape(event->key.string, NULL);
      kvname = gdk_keyval_name(event->key.keyval);
      g_print("%s %d:\"%s\" ",
              (kvname ? kvname : "??"), event->key.length, escaped);
      g_free(escaped);
      print_event_state(event->key.state);
      break;
   case GDK_ENTER_NOTIFY:
   case GDK_LEAVE_NOTIFY:
      g_print("%s ",
              (event->crossing.detail == GDK_NOTIFY_INFERIOR ? "INFERIOR" :
               (event->crossing.detail ==
                GDK_NOTIFY_ANCESTOR ? "ANCESTOR" : (event->crossing.
                                                    detail ==
                                                    GDK_NOTIFY_NONLINEAR ?
                                                    "NONLINEAR" :
                                                    "???"))));
      break;
   case GDK_SCROLL:
      g_print("%s ",
              (event->scroll.direction == GDK_SCROLL_UP ? "UP" :
               (event->scroll.direction == GDK_SCROLL_DOWN ? "DOWN" :
                (event->scroll.direction == GDK_SCROLL_LEFT ? "LEFT" :
                 (event->scroll.direction == GDK_SCROLL_RIGHT ? "RIGHT" :
                  "???")))));
      print_event_state(event->scroll.state);
      break;
   }
   g_print("\n");
}

static void synthesize_crossing_events(GdkWindow * window, MSG * xevent)
{
   GdkEvent *event;

   /* If we are not using TrackMouseEvent, generate a leave notify
    * event if necessary
    */
   if (p_TrackMouseEvent == NULL
       && curWnd
       && (GDK_WINDOW_WIN32DATA(curWnd)->
           event_mask & GDK_LEAVE_NOTIFY_MASK)) {
      GDK_NOTE(EVENTS, g_print("synthesizing LEAVE_NOTIFY event\n"));

      event = gdk_event_new();
      event->type = GDK_LEAVE_NOTIFY;
      event->crossing.type = GDK_LEAVE_NOTIFY;
      event->crossing.window = curWnd;
      gdk_window_ref(event->crossing.window);
      event->crossing.subwindow = NULL;
      event->crossing.time = xevent->time;
      event->crossing.x = curX;
      event->crossing.y = curY;
      event->crossing.x_root = curXroot;
      event->crossing.y_root = curYroot;
      event->crossing.mode = GDK_CROSSING_NORMAL;
      if (IsChild(GDK_DRAWABLE_XID(curWnd), GDK_DRAWABLE_XID(window)))
         event->crossing.detail = GDK_NOTIFY_INFERIOR;
      else if (IsChild(GDK_DRAWABLE_XID(window), GDK_DRAWABLE_XID(curWnd)))
         event->crossing.detail = GDK_NOTIFY_ANCESTOR;
      else
         event->crossing.detail = GDK_NOTIFY_NONLINEAR;

      event->crossing.focus = TRUE; /* ??? */
//      event->crossing.state = 0;    /* ??? */
      event->crossing.state = build_pointer_event_state(xevent);
      gdk_event_queue_append(event);
      GDK_NOTE(EVENTS, print_event(event));
   }

   if (GDK_WINDOW_WIN32DATA(window)->event_mask & GDK_ENTER_NOTIFY_MASK) {
      GDK_NOTE(EVENTS, g_print("synthesizing ENTER_NOTIFY event\n"));

      event = gdk_event_new();
      event->type = GDK_ENTER_NOTIFY;
      event->crossing.type = GDK_ENTER_NOTIFY;
      event->crossing.window = window;
      gdk_window_ref(event->crossing.window);
      event->crossing.subwindow = NULL;
      event->crossing.time = xevent->time;
      event->crossing.x = GET_X_LPARAM(xevent->lParam);
      event->crossing.y = GET_Y_LPARAM(xevent->lParam);
      event->crossing.x_root = (gfloat) xevent->pt.x;
      event->crossing.y_root = (gfloat) xevent->pt.y;
      event->crossing.mode = GDK_CROSSING_NORMAL;
      if (curWnd
          && IsChild(GDK_DRAWABLE_XID(curWnd), GDK_DRAWABLE_XID(window)))
         event->crossing.detail = GDK_NOTIFY_ANCESTOR;
      else if (curWnd
               && IsChild(GDK_DRAWABLE_XID(window),
                          GDK_DRAWABLE_XID(curWnd)))
         event->crossing.detail = GDK_NOTIFY_INFERIOR;
      else
         event->crossing.detail = GDK_NOTIFY_NONLINEAR;

      event->crossing.focus = TRUE; /* ??? */
//      event->crossing.state = 0;    /* ??? */
      event->crossing.state = build_pointer_event_state(xevent);

      gdk_event_queue_append(event);

      GDK_NOTE(EVENTS, print_event(event));

      if (((GdkWindowPrivate *) window)->extension_events != 0
          && gdk_input_vtable.enter_event)
         gdk_input_vtable.enter_event(&event->crossing, window);

   }

   if (curWnd)
      gdk_window_unref(curWnd);
   curWnd = window;
   gdk_window_ref(curWnd);
#ifdef USE_TRACKMOUSEEVENT
   if (p_TrackMouseEvent != NULL) {
      TRACKMOUSEEVENT tme;

      tme.cbSize = sizeof(TRACKMOUSEEVENT);
      tme.dwFlags = TME_LEAVE;
      tme.hwndTrack = GDK_DRAWABLE_XID(curWnd);
      tme.dwHoverTime = HOVER_DEFAULT;

      (*p_TrackMouseEvent) (&tme);
   }
#endif
}

static void
translate_mouse_coords(GdkWindow * window1,
                       GdkWindow * window2, MSG * xevent)
{
   POINT pt;

   pt.x = GET_X_LPARAM(xevent->lParam);
   pt.y = GET_Y_LPARAM(xevent->lParam);
   ClientToScreen(GDK_DRAWABLE_XID(window1), &pt);
   ScreenToClient(GDK_DRAWABLE_XID(window2), &pt);
   xevent->lParam = MAKELPARAM(pt.x, pt.y);
   GDK_NOTE(EVENTS, g_print("...new coords are (%d,%d)\n", pt.x, pt.y));
}

//vo
static gboolean
is_grabbed_button(GdkWindow **window, gint button, gint mod, guint32 time)
{
   GdkWindow *found = NULL;
   GdkWindow *sav = *window;

   while (*window) {
      if (((GDK_WINDOW_WIN32DATA(*window)->grab_button == 0) ||
           (GDK_WINDOW_WIN32DATA(*window)->grab_button == button))
         && ((GDK_WINDOW_WIN32DATA(*window)->grab_event_mask & GDK_BUTTON_PRESS_MASK)) //) {
         && (GDK_WINDOW_WIN32DATA(*window)->grab_modifiers & mod)) {
         found = *window;
      }
      if (((GdkWindowPrivate *) *window)->parent == gdk_parent_root) break;
      *window = ((GdkWindowPrivate *) *window)->parent;
   }

   if (found) {
      *window = found;
      gdk_window_ref(*window);
      return TRUE;
   } else {
      *window = sav;
      return FALSE;
   }
}

//vo
static gboolean
is_grabbed_key(GdkWindow **window, gint keycode, gint mod)
{
   GdkWindow *found = NULL;
   GdkWindow *sav = *window;

   while (1) {
      if (find_key_mod(GDK_WINDOW_WIN32DATA(*window)->grab_keys, keycode, mod)) {
         found = *window;
      }
      if (((GdkWindowPrivate *) *window)->parent == gdk_parent_root) break;
      *window = ((GdkWindowPrivate *) *window)->parent;
   }

   if (found) {
      *window = found;
      gdk_window_ref(*window);
      return TRUE;
   } else {
      *window = sav;
      return FALSE;
   }
}

static gboolean
propagate(GdkWindow ** window,
          MSG * xevent,
          GdkWindow * grab_window,
          gboolean grab_owner_events,
          gint grab_mask,
          gboolean(*doesnt_want_it) (gint mask, MSG * xevent))
{
   gboolean in_propagation = FALSE;

   if (grab_window != NULL && !grab_owner_events) {
      /* Event source is grabbed with owner_events FALSE */
      GDK_NOTE(EVENTS, g_print("...grabbed, owner_events FALSE, "));
      if ((*doesnt_want_it) (grab_mask, xevent)) {
         GDK_NOTE(EVENTS, g_print("...grabber doesn't want it\n"));
         return FALSE;
      } else {
         GDK_NOTE(EVENTS, g_print("...sending to grabber %#x\n",
                                  GDK_DRAWABLE_XID(grab_window)));
         gdk_window_unref(*window);
         *window = grab_window;
         gdk_window_ref(*window);
         return TRUE;
      }
   }

   while (TRUE) {
      if (GDK_DRAWABLE_TYPE(*window) == GDK_WINDOW_FOREIGN) return FALSE;
      if ((*doesnt_want_it)
          (GDK_WINDOW_WIN32DATA(*window)->event_mask, xevent) ) {
         /* Owner doesn't want it, propagate to parent. */
         if (((GdkWindowPrivate *) * window)->parent == gdk_parent_root) {
            /* No parent; check if grabbed */
            if (grab_window != NULL) {
               /* Event source is grabbed with owner_events TRUE */
               GDK_NOTE(EVENTS, g_print("...undelivered, but grabbed\n"));
               if ((*doesnt_want_it) (grab_mask, xevent)) {
                  /* Grabber doesn't want it either */
                  GDK_NOTE(EVENTS,
                           g_print("...grabber doesn't want it\n"));
                  return FALSE;
               } else {
                  /* Grabbed! */
                  GDK_NOTE(EVENTS, g_print("...sending to grabber %#x\n",
                                           GDK_DRAWABLE_XID(grab_window)));
                  gdk_window_unref(*window);
                  *window = grab_window;
                  gdk_window_ref(*window);
                  return TRUE;
               }
            } else {
               GDK_NOTE(EVENTS, g_print("...undelivered\n"));
               return FALSE;
            }
         } else {
            gdk_window_unref(*window);
            *window = ((GdkWindowPrivate *) * window)->parent;
            gdk_window_ref(*window);
            GDK_NOTE(EVENTS, g_print("...propagating to %#x\n",
                                     GDK_DRAWABLE_XID(*window)));
            /* The only branch where we actually continue the loop */
            in_propagation = TRUE;
         }
      } else {
         return TRUE;
      }
   }
}

static gboolean doesnt_want_key(gint mask, MSG * xevent)
{
   return (((xevent->message == WM_KEYUP || xevent->message == WM_SYSKEYUP)
            && !(mask & GDK_KEY_RELEASE_MASK))
           ||
           ((xevent->message == WM_KEYDOWN
             || xevent->message == WM_SYSKEYDOWN)
            && !(mask & GDK_KEY_PRESS_MASK)));
}

static gboolean doesnt_want_char(gint mask, MSG * xevent)
{
   return !(mask & (GDK_KEY_PRESS_MASK | GDK_KEY_RELEASE_MASK));
}

static gboolean doesnt_want_button_press(gint mask, MSG * xevent)
{
   return !(mask & GDK_BUTTON_PRESS_MASK);
}

static gboolean doesnt_want_button_release(gint mask, MSG * xevent)
{
   return !(mask & GDK_BUTTON_RELEASE_MASK);
}

static gboolean doesnt_want_button_motion(gint mask, MSG * xevent)
{
   return !((mask & GDK_POINTER_MOTION_MASK)
            || ((xevent->wParam & (MK_LBUTTON | MK_MBUTTON | MK_RBUTTON))
                && (mask & GDK_BUTTON_MOTION_MASK))
            || ((xevent->wParam & MK_LBUTTON)
                && (mask & GDK_BUTTON1_MOTION_MASK))
            || ((xevent->wParam & MK_MBUTTON)
                && (mask & GDK_BUTTON2_MOTION_MASK))
            || ((xevent->wParam & MK_RBUTTON)
                && (mask & GDK_BUTTON3_MOTION_MASK)));
}

static gboolean doesnt_want_scroll(gint mask, MSG * xevent)
{
#if 0
   return !(mask & GDK_SCROLL_MASK);
#else
   return !(mask & GDK_BUTTON_PRESS_MASK);
#endif
}

static char *decode_key_lparam(LPARAM lParam)
{
   static char buf[100];
   char *p = buf;

   if (HIWORD(lParam) & KF_UP)
      p += sprintf(p, "KF_UP ");
   if (HIWORD(lParam) & KF_REPEAT)
      p += sprintf(p, "KF_REPEAT ");
   if (HIWORD(lParam) & KF_ALTDOWN)
      p += sprintf(p, "KF_ALTDOWN ");
   if (HIWORD(lParam) & KF_EXTENDED)
      p += sprintf(p, "KF_EXTENDED ");
   p += sprintf(p, "sc%d rep%d", LOBYTE(HIWORD(lParam)), LOWORD(lParam));

   return buf;
}

static gboolean
gdk_event_translate(GdkEvent * event,
                    MSG * xevent,
                    gboolean * ret_val_flagp, gint * ret_valp)
{
   PAINTSTRUCT paintstruct;
   HDC hdc;
   HDC bgdc;
   HGDIOBJ oldbitmap;
   HBRUSH hbr;
   COLORREF bg;
   RECT rect;
   POINT pt;
   MINMAXINFO *lpmmi;
   HWND hwnd;
   HCURSOR xcursor;
   GdkWindow *window, *orig_window, *newwindow;
   GdkColormapPrivateWin32 *colormap_private;
   GdkPixmap *pixmap;
   GdkDrawablePrivate *pixmap_private;
   int button;
   int i, j;
   gchar buf[256];
   gboolean return_val;

   return_val = FALSE;

   if (ret_val_flagp)
      *ret_val_flagp = FALSE;

   window = gdk_window_lookup(xevent->hwnd);
   orig_window = window;

   if (window != NULL)
      gdk_window_ref(window);
   else {
      /* Handle WM_QUIT here ? */
      if (xevent->message == WM_QUIT) {
         GDK_NOTE(EVENTS, g_print("WM_QUIT: %d\n", xevent->wParam));
         exit(xevent->wParam);
      } else if (xevent->message == WM_MOVE || xevent->message == WM_SIZE) {
         /* It's quite normal to get these messages before we have
          * had time to register the window in our lookup table, or
          * when the window is being destroyed and we already have
          * removed it. Repost the same message to our queue so that
          * we will get it later when we are prepared.
          */
         PostMessage(xevent->hwnd, xevent->message,
                     xevent->wParam, xevent->lParam);
      }
      return FALSE;
   }

   if (!GDK_DRAWABLE_DESTROYED(window)) {
      /* Check for filters for this window */
      GdkFilterReturn result;
      event->any.window = window;
      result = gdk_event_apply_filters
          (xevent, event, ((GdkWindowPrivate *) window)->filters);

      if (result != GDK_FILTER_CONTINUE) {
         return (result == GDK_FILTER_TRANSLATE) ? TRUE : FALSE;
      }
   }

   if (xevent->message == gdk_selection_notify_msg) {
      GDK_NOTE(EVENTS, g_print("gdk_selection_notify_msg: %#x\n",
                               xevent->hwnd));

      event->type = GDK_SELECTION_NOTIFY;
      event->selection.type = GDK_SELECTION_NOTIFY;
      event->selection.window = window;
      event->selection.selection = xevent->wParam;
      event->selection.target = xevent->lParam;
      event->selection.property = gdk_selection_property;
      event->selection.time = xevent->time;

      return_val = !GDK_DRAWABLE_DESTROYED(window);

      goto bypass_switch;
   } else if (xevent->message == gdk_selection_request_msg) {
      GDK_NOTE(EVENTS, g_print("gdk_selection_request_msg: %#x\n",
                               xevent->hwnd));

      event->type = GDK_SELECTION_REQUEST;
      event->selection.type = GDK_SELECTION_REQUEST;
      event->selection.window = window;
      if (xevent->wParam == gdk_clipboard_atom) {
         event->selection.selection = gdk_clipboard_atom;
         event->selection.target = GDK_TARGET_STRING;
      }
      else {
         event->selection.selection = xevent->wParam;
         event->selection.target = 0x0200; // CF_PRIVATEFIRST
      }
      event->selection.property = gdk_selection_property;
      event->selection.requestor = (guint32) xevent->hwnd;
      event->selection.time = xevent->time;

      return_val = !GDK_DRAWABLE_DESTROYED(window);

      goto bypass_switch;
   } else if (xevent->message == gdk_selection_clear_msg) {
      GDK_NOTE(EVENTS, g_print("gdk_selection_clear_msg: %#x\n",
                               xevent->hwnd));

      event->type = GDK_SELECTION_CLEAR;
      event->selection.type = GDK_SELECTION_CLEAR;
      event->selection.window = window;
      event->selection.selection = xevent->wParam;
      event->selection.time = xevent->time;

      return_val = !GDK_DRAWABLE_DESTROYED(window);

      goto bypass_switch;
   } else if (xevent->message == msh_mousewheel_msg) {
      GDK_NOTE(EVENTS, g_print("MSH_MOUSEWHEEL: %#x %d\n",
                               xevent->hwnd, xevent->wParam));

      event->type = GDK_SCROLL;
      event->scroll.type = GDK_SCROLL;

      /* MSG_MOUSEWHEEL is delivered to the foreground window.  Work
       * around that. Also, the position is in screen coordinates, not
       * client coordinates as with the button messages.
       */
      pt.x = GET_X_LPARAM(xevent->lParam);
      pt.y = GET_Y_LPARAM(xevent->lParam);
      if ((hwnd = WindowFromPoint(pt)) == NULL)
         goto bypass_switch;

      xevent->hwnd = hwnd;
      if ((newwindow = gdk_window_lookup(xevent->hwnd)) == NULL)
         goto bypass_switch;

      if (newwindow != window) {
         gdk_window_unref(window);
         window = newwindow;
         gdk_window_ref(window);
      }

      if (((GdkWindowPrivate *) window)->extension_events != 0
          && gdk_input_ignore_core) {
         GDK_NOTE(EVENTS, g_print("...ignored\n"));
         goto bypass_switch;
      }

      if (!propagate(&window, xevent,
                     p_grab_window, p_grab_owner_events, p_grab_mask,
                     doesnt_want_scroll))
         goto bypass_switch;

      ScreenToClient(xevent->hwnd, &pt);
      event->button.window = window;
      event->scroll.direction = ((int) xevent->wParam > 0) ?
          GDK_SCROLL_UP : GDK_SCROLL_DOWN;
      event->scroll.window = window;
      event->scroll.time = xevent->time;
      event->scroll.x = (gint16) pt.x;
      event->scroll.y = (gint16) pt.y;
      event->scroll.x_root = (gint16) GET_X_LPARAM(xevent->lParam);
      event->scroll.y_root = (gint16) GET_Y_LPARAM(xevent->lParam);
      event->scroll.pressure = 0.5;
      event->scroll.xtilt = 0;
      event->scroll.ytilt = 0;
      event->scroll.state = 0;  /* No state information with MSH_MOUSEWHEEL */
      event->scroll.source = GDK_SOURCE_MOUSE;
      event->scroll.deviceid = GDK_CORE_POINTER;
      return_val = !GDK_DRAWABLE_DESTROYED(window);

      goto bypass_switch;
   } else {
      GList *tmp_list;
      GdkFilterReturn result = GDK_FILTER_CONTINUE;

      tmp_list = client_filters;
      while (tmp_list) {
         GdkClientFilter *filter = tmp_list->data;
         if (filter->type == xevent->message) {
            GDK_NOTE(EVENTS, g_print("client filter matched\n"));
            event->any.window = window;
            result = (*filter->function) (xevent, event, filter->data);
            switch (result) {
            case GDK_FILTER_REMOVE:
               return_val = FALSE;
               break;

            case GDK_FILTER_TRANSLATE:
               return_val = TRUE;
               break;

            case GDK_FILTER_CONTINUE:
               return_val = TRUE;
               event->type = GDK_CLIENT_EVENT;
               event->client.type = GDK_CLIENT_EVENT;
               event->client.window = window;
               event->client.message_type = xevent->message;
               event->client.data_format = 0;
               event->client.data.l[0] = xevent->wParam;
               event->client.data.l[1] = xevent->lParam;
               break;
            }
            goto bypass_switch; /* Ouch */
         }
         tmp_list = tmp_list->next;
      }
   }

   switch (xevent->message) {
   case WM_INPUTLANGCHANGE:
      GDK_NOTE(EVENTS,
               g_print("WM_INPUTLANGCHANGE: %#x  charset %d locale %x\n",
                       xevent->hwnd, xevent->wParam, xevent->lParam));
      GDK_WINDOW_WIN32DATA(window)->input_locale = (HKL) xevent->lParam;
      TranslateCharsetInfo((DWORD FAR *) xevent->wParam,
                           &GDK_WINDOW_WIN32DATA(window)->charset_info,
                           TCI_SRCCHARSET);
      break;

   case WM_SYSKEYUP:
   case WM_SYSKEYDOWN:
      //k_grab_window = 0;   //vo terminate active grab

      GDK_NOTE(EVENTS,
               g_print("WM_SYSKEY%s: %#x  %s %#x %s\n",
                       (xevent->message == WM_SYSKEYUP ? "UP" : "DOWN"),
                       xevent->hwnd,
                       (GetKeyNameText(xevent->lParam, buf,
                                       sizeof(buf)) > 0 ?
                        buf : ""),
                       xevent->wParam, decode_key_lparam(xevent->lParam)));

      /* Let the system handle Alt-Tab and Alt-Enter */
      if (xevent->wParam == VK_TAB
          || xevent->wParam == VK_RETURN || xevent->wParam == VK_F4)
         break;
      /* If posted without us having keyboard focus, ignore */
      if (!(xevent->lParam & 0x20000000))
         break;
#if 0
      /* don't generate events for just the Alt key */
      if (xevent->wParam == VK_MENU)
         break;
#endif
      /* Jump to code in common with WM_KEYUP and WM_KEYDOWN */
      goto keyup_or_down;

   case WM_KEYUP:
   case WM_KEYDOWN:
      //k_grab_window = 0;   //vo terminate active grab

      GDK_NOTE(EVENTS,
               printf("WM_KEY%s: %#x  %s %#x %s\n",
                       (xevent->message == WM_KEYUP ? "UP" : "DOWN"),
                       (unsigned) xevent->hwnd,
                       (GetKeyNameText(xevent->lParam, buf,
                                       sizeof(buf)) > 0 ?
                        buf : ""),
                       xevent->wParam, decode_key_lparam(xevent->lParam)));

      ignore_WM_CHAR = TRUE;

   keyup_or_down:

      event->key.window = window;
      switch (xevent->wParam) {
      case VK_LBUTTON:
         event->key.keyval = GDK_Pointer_Button1;
         break;
      case VK_RBUTTON:
         event->key.keyval = GDK_Pointer_Button3;
         break;
      case VK_MBUTTON:
         event->key.keyval = GDK_Pointer_Button2;
         break;
      case VK_CANCEL:
         event->key.keyval = GDK_Cancel;
         break;
      case VK_BACK:
         event->key.keyval = GDK_BackSpace;
         break;
      case VK_TAB:
         event->key.keyval = (GetKeyState(VK_SHIFT) < 0 ?
                              GDK_ISO_Left_Tab : GDK_Tab);
         break;
      case VK_CLEAR:
         event->key.keyval = GDK_Clear;
         break;
      case VK_RETURN:
         event->key.keyval = GDK_Return;
         break;
      case VK_SHIFT:
         /* Don't let Shift auto-repeat */
         if (xevent->message == WM_KEYDOWN
             && (HIWORD(xevent->lParam) & KF_REPEAT))
            ignore_WM_CHAR = FALSE;
         else
            event->key.keyval = GDK_Shift_L;
         break;
      case VK_CONTROL:
         /* And not Control either */
         if (xevent->message == WM_KEYDOWN
             && (HIWORD(xevent->lParam) & KF_REPEAT))
            ignore_WM_CHAR = FALSE;
         else if (HIWORD(xevent->lParam) & KF_EXTENDED)
            event->key.keyval = GDK_Control_R;
         else
            event->key.keyval = GDK_Control_L;
         break;
      case VK_MENU:
         /* And not Alt */
         if (xevent->message == WM_KEYDOWN
             && (HIWORD(xevent->lParam) & KF_REPEAT)) {
            ignore_WM_CHAR = FALSE;
         } else if (HIWORD(xevent->lParam) & KF_EXTENDED) {
            /* AltGr key comes in as Control+Right Alt */
            if (GetKeyState(VK_CONTROL) < 0) {
               ignore_WM_CHAR = FALSE;
               is_AltGr_key = TRUE;
            }
            event->key.keyval = GDK_Alt_R;
         } else {
            event->key.keyval = GDK_Alt_L;
            /* This needed in case she types Alt+nnn (on the numpad) */
            ignore_WM_CHAR = FALSE;
         }
         break;
      case VK_PAUSE:
         event->key.keyval = GDK_Pause;
         break;
      case VK_CAPITAL:
         event->key.keyval = GDK_Caps_Lock;
         break;
      case VK_ESCAPE:
         event->key.keyval = GDK_Escape;
         break;
      case VK_PRIOR:
         event->key.keyval = GDK_Prior;
         break;
      case VK_NEXT:
         event->key.keyval = GDK_Next;
         break;
      case VK_END:
         event->key.keyval = GDK_End;
         break;
      case VK_HOME:
         event->key.keyval = GDK_Home;
         break;
      case VK_LEFT:
         event->key.keyval = GDK_Left;
         break;
      case VK_UP:
         event->key.keyval = GDK_Up;
         break;
      case VK_RIGHT:
         event->key.keyval = GDK_Right;
         break;
      case VK_DOWN:
         event->key.keyval = GDK_Down;
         break;
      case VK_SELECT:
         event->key.keyval = GDK_Select;
         break;
      case VK_PRINT:
         event->key.keyval = GDK_Print;
         break;
      case VK_EXECUTE:
         event->key.keyval = GDK_Execute;
         break;
      case VK_INSERT:
         event->key.keyval = GDK_Insert;
         break;
      case VK_DELETE:
         event->key.keyval = GDK_Delete;
         break;
      case VK_HELP:
         event->key.keyval = GDK_Help;
         break;
      case VK_NUMPAD0:
      case VK_NUMPAD1:
      case VK_NUMPAD2:
      case VK_NUMPAD3:
      case VK_NUMPAD4:
      case VK_NUMPAD5:
      case VK_NUMPAD6:
      case VK_NUMPAD7:
      case VK_NUMPAD8:
      case VK_NUMPAD9:
         /* Apparently applications work better if we just pass numpad digits
          * on as real digits? So wait for the WM_CHAR instead.
          */
         ignore_WM_CHAR = FALSE;
         break;
      case VK_MULTIPLY:
         event->key.keyval = GDK_KP_Multiply;
         break;
      case VK_ADD:
         /* Pass it on as an ASCII plus in WM_CHAR. */
         ignore_WM_CHAR = FALSE;
         break;
      case VK_SEPARATOR:
         event->key.keyval = GDK_KP_Separator;
         break;
      case VK_SUBTRACT:
         /* Pass it on as an ASCII minus in WM_CHAR. */
         ignore_WM_CHAR = FALSE;
         break;
      case VK_DECIMAL:
         /* The keypad decimal key should also be passed on as the decimal
          * sign ('.' or ',' depending on the Windows locale settings,
          * apparently). So wait for the WM_CHAR here, also.
          */
         ignore_WM_CHAR = FALSE;
         break;
      case VK_DIVIDE:
         event->key.keyval = GDK_KP_Divide;
         break;
      case VK_F1:
         event->key.keyval = GDK_F1;
         break;
      case VK_F2:
         event->key.keyval = GDK_F2;
         break;
      case VK_F3:
         event->key.keyval = GDK_F3;
         break;
      case VK_F4:
         event->key.keyval = GDK_F4;
         break;
      case VK_F5:
         event->key.keyval = GDK_F5;
         break;
      case VK_F6:
         event->key.keyval = GDK_F6;
         break;
      case VK_F7:
         event->key.keyval = GDK_F7;
         break;
      case VK_F8:
         event->key.keyval = GDK_F8;
         break;
      case VK_F9:
         event->key.keyval = GDK_F9;
         break;
      case VK_F10:
         event->key.keyval = GDK_F10;
         break;
      case VK_F11:
         event->key.keyval = GDK_F11;
         break;
      case VK_F12:
         event->key.keyval = GDK_F12;
         break;
      case VK_F13:
         event->key.keyval = GDK_F13;
         break;
      case VK_F14:
         event->key.keyval = GDK_F14;
         break;
      case VK_F15:
         event->key.keyval = GDK_F15;
         break;
      case VK_F16:
         event->key.keyval = GDK_F16;
         break;
      case '0':
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
      case '7':
      case '8':
      case '9':
         if (!is_AltGr_key && (GetKeyState(VK_CONTROL) < 0
                               || GetKeyState(VK_MENU) < 0))
            /* Control- or Alt- digits won't come in as a WM_CHAR, but
             * beware of AltGr-digits, which are used for instance on
             * Finnish keyboards.
             */
            event->key.keyval = GDK_0 + (xevent->wParam - '0');
         else
            ignore_WM_CHAR = FALSE;
         break;
      case VK_OEM_PLUS:        /* On my Win98, the '+' key comes in
                                 * as VK_OEM_PLUS, etc
                                 */
      case VK_OEM_COMMA:
      case VK_OEM_MINUS:
      case VK_OEM_PERIOD:
      case VK_OEM_2:
      case VK_OEM_4:
      case VK_OEM_5:
      case VK_OEM_6:
         if (!is_AltGr_key && (GetKeyState(VK_CONTROL) < 0
                               || GetKeyState(VK_MENU) < 0))
            /* Control- or Alt- plus, comma, minus or period won't
             * come in as WM_CHAR, but beware of AltGr-plus which is
             * backslash on Finnish keyboards
             */
            /* All these VK_OEM keycodes are the corresponding ASCII
             * char + 0x90
             */
            event->key.keyval = xevent->wParam - 0x90;
         else
            ignore_WM_CHAR = FALSE;
         break;
      case VK_OEM_1:
         if (!is_AltGr_key && (GetKeyState(VK_CONTROL) < 0
                               || GetKeyState(VK_MENU) < 0))
            /* ;: on US keyboard */
            event->key.keyval = ';';
         else
            ignore_WM_CHAR = FALSE;
         break;
      case VK_OEM_3:
         if (!is_AltGr_key && (GetKeyState(VK_CONTROL) < 0
                               || GetKeyState(VK_MENU) < 0))
            /* `~ on US keyboard */
            event->key.keyval = '`';
         else
            ignore_WM_CHAR = FALSE;
         break;
      case VK_OEM_7:
         if (!is_AltGr_key && (GetKeyState(VK_CONTROL) < 0
                               || GetKeyState(VK_MENU) < 0))
            /* '" on US keyboard */
            event->key.keyval = '\'';
         else
            ignore_WM_CHAR = FALSE;
         break;
      default:
         if (xevent->message == WM_SYSKEYDOWN
             || xevent->message == WM_SYSKEYUP)
            event->key.keyval = xevent->wParam;
         else
            ignore_WM_CHAR = FALSE;
         break;
      }

      if (!ignore_WM_CHAR)
         break;

      event->key.state = 0;
      if (GetKeyState(VK_SHIFT) < 0)
         event->key.state |= GDK_SHIFT_MASK;
      if (GetKeyState(VK_CAPITAL) & 0x1)
         event->key.state |= GDK_LOCK_MASK;
      if (GetKeyState(VK_CONTROL) < 0)
         event->key.state |= GDK_CONTROL_MASK;
      if (xevent->wParam != VK_MENU && GetKeyState(VK_MENU) < 0)
         event->key.state |= GDK_MOD1_MASK;

      //vo check if key is grabbed
      if (!k_grab_window &&
          is_grabbed_key(&window, event->key.keyval, event->key.state)) {
         gdk_keyboard_grab(window, GDK_WINDOW_WIN32DATA(window)->grab_key_owner_events, 0);
      } else {
         POINT pt;
         HWND chw;
         GdkWindow* other_window = 0;
         GetCursorPos(&pt);
         chw = WindowFromPoint(pt);
         other_window = gdk_window_lookup(chw);
         if (other_window && IsChild(xevent->hwnd, chw)) {
            gdk_window_unref (window);
            gdk_window_ref (other_window);
            window = other_window;
         }
         if (!k_grab_window &&
            is_grabbed_key(&window, event->key.keyval, event->key.state)) {
            gdk_keyboard_grab(window, GDK_WINDOW_WIN32DATA(window)->grab_key_owner_events, 0);
         }
      }

      if (!propagate(&window, xevent, k_grab_window, k_grab_owner_events,
                     GDK_ALL_EVENTS_MASK, doesnt_want_key)) {
         event->key.state = 0;
         break;
      }

      is_AltGr_key = FALSE;
      event->key.window = window;
      event->key.type = ((xevent->message == WM_KEYDOWN ||
                          xevent->message == WM_SYSKEYDOWN) ?
                          GDK_KEY_PRESS : GDK_KEY_RELEASE);
      event->type = event->key.type;
      event->key.time = xevent->time;
      event->key.string = NULL;
      event->key.length = 0;
      if (event->key.keyval == GDK_Escape) {
         event->key.length = 1;
         event->key.string = g_strdup ("\033");
      }
      else if (event->key.keyval == GDK_Return ||
               event->key.keyval == GDK_KP_Enter) {
         event->key.length = 1;
         event->key.string = g_strdup ("\r");
      }
      return_val = !GDK_DRAWABLE_DESTROYED(window);
      break;

   case WM_IME_COMPOSITION:
      if (!use_IME_COMPOSITION)
         break;
      GDK_NOTE(EVENTS, g_print("WM_IME_COMPOSITION: %#x  %#x\n",
                               xevent->hwnd, xevent->lParam));
      if (xevent->lParam & GCS_RESULTSTR)
         goto wm_char;
      break;

   case WM_IME_CHAR:
      GDK_NOTE(EVENTS,
               g_print("WM_IME_CHAR: %#x  bytes: %#.04x\n",
                       xevent->hwnd, xevent->wParam));
      goto wm_char;

   case WM_CHAR:
   case WM_SYSCHAR:
      GDK_NOTE(EVENTS,
               g_print("WM_%sCHAR: %#x  %#x %#s %s\n",
                       (xevent->message == WM_CHAR ? "" : "SYS"),
                       xevent->hwnd, xevent->wParam,
                       decode_key_lparam(xevent->lParam),
                       (ignore_WM_CHAR ? "ignored" : "")));

      if (ignore_WM_CHAR) {
         ignore_WM_CHAR = FALSE;
         break;
      }

   wm_char:
      event->key.state = 0;
      if (GetKeyState(VK_SHIFT) < 0)
         event->key.state |= GDK_SHIFT_MASK;
      if (GetKeyState(VK_CAPITAL) & 0x1)
         event->key.state |= GDK_LOCK_MASK;
      if (GetKeyState(VK_CONTROL) < 0)
         event->key.state |= GDK_CONTROL_MASK;
      if (xevent->wParam != VK_MENU && GetKeyState(VK_MENU) < 0)
         event->key.state |= GDK_MOD1_MASK;

      event->key.keyval = get_key_value(xevent);

      //vo check if key is grabbed
      if (!k_grab_window &&
          is_grabbed_key(&window, event->key.keyval, event->key.state)) {
         gdk_keyboard_grab(window, GDK_WINDOW_WIN32DATA(window)->grab_key_owner_events, 0);
      } else {
         POINT pt;
         HWND chw;
         GdkWindow* other_window = 0;
         GetCursorPos(&pt);
         chw = WindowFromPoint(pt);
         other_window = gdk_window_lookup(chw);
         if (other_window && IsChild(xevent->hwnd, chw)) {
            gdk_window_unref (window);
            gdk_window_ref (other_window);
            window = other_window;
         }
         if (!k_grab_window &&
            is_grabbed_key(&window, event->key.keyval, event->key.state)) {
            gdk_keyboard_grab(window, GDK_WINDOW_WIN32DATA(window)->grab_key_owner_events, 0);
         }
      }

      if (!propagate(&window, xevent, k_grab_window, k_grab_owner_events,
                     GDK_ALL_EVENTS_MASK, doesnt_want_char)) {
         break;
      }

      event->key.window = window;
      return_val = !GDK_DRAWABLE_DESTROYED(window);

      if (return_val && (event->key.window == k_grab_window
         || (GDK_WINDOW_WIN32DATA(window)->event_mask & GDK_KEY_RELEASE_MASK))) {
         if (window == k_grab_window
             || (GDK_WINDOW_WIN32DATA(window)->event_mask & GDK_KEY_PRESS_MASK)) {
            /* Append a GDK_KEY_PRESS event to the pushback list
             * (from which it will be fetched before the release
             * event).
             */
            GdkEvent *event2 = gdk_event_new();
            build_keypress_event(GDK_WINDOW_WIN32DATA(window), event2, xevent);
            event2->key.window = window;
            gdk_window_ref(window);
            gdk_event_queue_append(event2);
            GDK_NOTE(EVENTS, print_event(event2));
         }
         /* Return the key release event.  */
         build_keyrelease_event(GDK_WINDOW_WIN32DATA(window), event, xevent);
      } else if (window == k_grab_window
             || (GDK_WINDOW_WIN32DATA(window)->event_mask & GDK_KEY_RELEASE_MASK)) {
         /* Return just the key press event. */
         build_keyrelease_event(GDK_WINDOW_WIN32DATA(window), event, xevent);
      } else if (return_val
                 && (GDK_WINDOW_WIN32DATA(window)->event_mask & GDK_KEY_PRESS_MASK)) {
         /* Return just the key press event. */
         build_keypress_event(GDK_WINDOW_WIN32DATA(window), event, xevent);
      } else {
         return_val = FALSE;
      }
#if 0                           /* Don't reset is_AltGr_key here. Othewise we can't type several
                                 * AltGr-accessed chars while keeping the AltGr pressed down
                                 * all the time.
                                 */
      is_AltGr_key = FALSE;
#endif
      break;

   case WM_LBUTTONDOWN:
      button = 1;
      goto buttondown0;
   case WM_MBUTTONDOWN:
      button = 2;
      goto buttondown0;
   case WM_RBUTTONDOWN:
      button = 3;

    buttondown0:
      GDK_NOTE(EVENTS,
               g_print("WM_%cBUTTONDOWN: %#x  (%d,%d)\n",
                       " LMR"[button],
                       xevent->hwnd,
                       GET_X_LPARAM(xevent->lParam), GET_Y_LPARAM(xevent->lParam)));

      if (((GdkWindowPrivate *) window)->extension_events != 0
          && gdk_input_ignore_core) {
         GDK_NOTE(EVENTS, g_print("...ignored\n"));
         break;
      }

      window = find_window_for_pointer_event (window, xevent);

      if (window != curWnd)
         synthesize_crossing_events(window, xevent);

      event->type = GDK_BUTTON_PRESS;
      event->button.type = GDK_BUTTON_PRESS;

      //vo check if button is grabbed
      if (!p_grab_window &&
          is_grabbed_button(&window, button, build_pointer_event_state(xevent), xevent->time)) {
         gdk_pointer_grab(window, GDK_WINDOW_WIN32DATA(window)->grab_owner_events,
                          GDK_WINDOW_WIN32DATA(window)->grab_event_mask,
                          GDK_WINDOW_WIN32DATA(window)->grab_confine,
                          GDK_WINDOW_WIN32DATA(window)->grab_cursor, 0);
         p_grab_automatic = TRUE;
      } else if (!propagate(&window, xevent, p_grab_window, p_grab_owner_events,  p_grab_mask,
                     doesnt_want_button_press)) {
         break;
      }
      event->button.window = window;

     /* Emulate X11's automatic active grab */
      if (!p_grab_window) {
         /* No explicit active grab, let's start one automatically */
         gint owner_events =
             GDK_WINDOW_WIN32DATA (window)->event_mask
             & (GDK_BUTTON_PRESS_MASK | GDK_BUTTON_RELEASE_MASK);

         GDK_NOTE(EVENTS, g_print("...automatic grab started\n"));
         gdk_pointer_grab(window,
                          owner_events,
                          GDK_WINDOW_WIN32DATA (window)->event_mask,
                          NULL, NULL, 0);
         p_grab_automatic = TRUE;
      }

      event->button.time = xevent->time;
      if (window != orig_window)
         translate_mouse_coords(orig_window, window, xevent);
      event->button.x = curX = (gint16) GET_X_LPARAM(xevent->lParam);
      event->button.y = curY = (gint16) GET_Y_LPARAM(xevent->lParam);
      event->button.x_root = xevent->pt.x;
      event->button.y_root = xevent->pt.y;
      event->button.pressure = 0.5;
      event->button.xtilt = 0;
      event->button.ytilt = 0;
      event->button.state = build_pointer_event_state(xevent);
      event->button.button = button;
      event->button.source = GDK_SOURCE_MOUSE;
      event->button.deviceid = GDK_CORE_POINTER;

      gdk_event_button_generate(event);

      return_val = !GDK_DRAWABLE_DESTROYED(window);
      break;

   case WM_LBUTTONUP:
      button = 1;
      goto buttonup0;
   case WM_MBUTTONUP:
      button = 2;
      goto buttonup0;
   case WM_RBUTTONUP:
      button = 3;

   buttonup0:
      GDK_NOTE(EVENTS,
               g_print("WM_%cBUTTONUP: %#x  (%d,%d)\n",
                       " LMR"[button],
                       xevent->hwnd,
                       GET_X_LPARAM(xevent->lParam), GET_Y_LPARAM(xevent->lParam)));

      window = find_window_for_pointer_event (window, xevent);

      if (((GdkWindowPrivate *) window)->extension_events != 0
          && gdk_input_ignore_core) {
         GDK_NOTE(EVENTS, g_print("...ignored\n"));
         break;
      }

      if (window != curWnd)
         synthesize_crossing_events(window, xevent);

      event->type = GDK_BUTTON_RELEASE;
      event->button.type = GDK_BUTTON_RELEASE;
      if (!propagate(&window, xevent,
                     p_grab_window, p_grab_owner_events, p_grab_mask,
                     doesnt_want_button_release)) {
      } else {
         if (window != orig_window) {
            translate_mouse_coords(orig_window, window, xevent);
         }

         event->button.window = window;
         event->button.time = xevent->time;
         event->button.x = (gint16) GET_X_LPARAM(xevent->lParam);
         event->button.y = (gint16) GET_Y_LPARAM(xevent->lParam);
         event->button.x_root = xevent->pt.x;
         event->button.y_root = xevent->pt.y;
         event->button.pressure = 0.5;
         event->button.xtilt = 0;
         event->button.ytilt = 0;
         event->button.state = build_pointer_event_state(xevent);
         event->button.button = button;
         event->button.source = GDK_SOURCE_MOUSE;
         event->button.deviceid = GDK_CORE_POINTER;

         return_val = !GDK_DRAWABLE_DESTROYED(window);
      }

      if (p_grab_window != NULL
          && p_grab_automatic
          && (xevent->wParam & (MK_LBUTTON | MK_MBUTTON | MK_RBUTTON)) == 0) {
         gdk_pointer_ungrab (0);
      }
      break;

   case WM_MOUSELEAVE:
      GDK_NOTE(EVENTS, g_print("WM_MOUSELEAVE: %#x\n", xevent->hwnd));

      if (!(GDK_WINDOW_WIN32DATA(window)->
           event_mask & GDK_LEAVE_NOTIFY_MASK))
         break;

      event->type = GDK_LEAVE_NOTIFY;
      event->crossing.type = GDK_LEAVE_NOTIFY;
      event->crossing.window = window;
      event->crossing.subwindow = NULL;
      event->crossing.time = xevent->time;
      event->crossing.x = curX;
      event->crossing.y = curY;
      event->crossing.x_root = curXroot;
      event->crossing.y_root = curYroot;
      event->crossing.mode = GDK_CROSSING_NORMAL;
      if (curWnd && IsChild(GDK_DRAWABLE_XID(curWnd),
                            GDK_DRAWABLE_XID(window)))
         event->crossing.detail = GDK_NOTIFY_INFERIOR;
      else if (curWnd && IsChild(GDK_DRAWABLE_XID(window),
                                 GDK_DRAWABLE_XID(curWnd)))
         event->crossing.detail = GDK_NOTIFY_ANCESTOR;
      else
         event->crossing.detail = GDK_NOTIFY_NONLINEAR;
      event->crossing.focus = TRUE;
      event->crossing.state = build_pointer_event_state(xevent);

      if (curWnd) {
         gdk_window_unref(curWnd);
         curWnd = NULL;
      }

      return_val = !GDK_DRAWABLE_DESTROYED(window);
      break;

   case WM_MOUSEMOVE:
      GDK_NOTE(EVENTS,
               g_print("WM_MOUSEMOVE: %#x  %#x (%d,%d)\n",
                       xevent->hwnd, xevent->wParam,
                       GET_X_LPARAM(xevent->lParam), GET_Y_LPARAM(xevent->lParam)));

      track_mouse_event(xevent->hwnd);

      window = find_window_for_pointer_event (window, xevent);

      if (window != curWnd)
         synthesize_crossing_events(window, xevent);

      if (((GdkWindowPrivate *) window)->extension_events != 0
          && gdk_input_ignore_core) {
         GDK_NOTE(EVENTS, g_print("...ignored\n"));
         break;
      }

      event->type = GDK_MOTION_NOTIFY;
      event->motion.type = GDK_MOTION_NOTIFY;
      if (!propagate(&window, xevent,
                     p_grab_window, p_grab_owner_events, p_grab_mask,
                     doesnt_want_button_motion))
         break;

      event->motion.window = window;
      event->motion.time = xevent->time;
      if (window != orig_window)
         translate_mouse_coords(orig_window, window, xevent);

      if (window == curWnd
         && (gint16) GET_X_LPARAM(xevent->lParam) == curX
         && (gint16) GET_Y_LPARAM(xevent->lParam) == curY) break;

      event->motion.x = curX = (gint16) GET_X_LPARAM(xevent->lParam);
      event->motion.y = curY = (gint16) GET_Y_LPARAM(xevent->lParam);
      event->motion.x_root = xevent->pt.x;
      event->motion.y_root = xevent->pt.y;
      curXroot = event->motion.x_root;
      curYroot = event->motion.y_root;
      event->motion.pressure = 0.5;
      event->motion.xtilt = 0;
      event->motion.ytilt = 0;
      event->motion.state = build_pointer_event_state(xevent);
#if 0
      event->motion.is_hint = (p_grab_window != NULL
                               && (p_grab_mask &
                                   GDK_POINTER_MOTION_HINT_MASK));
#else
      event->motion.is_hint = FALSE;
#endif
      event->motion.source = GDK_SOURCE_MOUSE;
      event->motion.deviceid = GDK_CORE_POINTER;

      return_val = !GDK_DRAWABLE_DESTROYED(window);
      break;

   case WM_NCMOUSEMOVE:
      GDK_NOTE(EVENTS,
               g_print("WM_NCMOUSEMOVE: %#x  x,y: %d %d\n",
                       xevent->hwnd,
                       GET_X_LPARAM(xevent->lParam), GET_Y_LPARAM(xevent->lParam)));
      if (p_TrackMouseEvent == NULL
          && curWnd != NULL
          && (GDK_WINDOW_WIN32DATA(curWnd)->
              event_mask & GDK_LEAVE_NOTIFY_MASK)) {
         GDK_NOTE(EVENTS, g_print("...synthesizing LEAVE_NOTIFY event\n"));

         event->type = GDK_LEAVE_NOTIFY;
         event->crossing.type = GDK_LEAVE_NOTIFY;
         event->crossing.window = curWnd;
         event->crossing.subwindow = NULL;
         event->crossing.time = xevent->time;
         event->crossing.x = curX;
         event->crossing.y = curY;
         event->crossing.x_root = curXroot;
         event->crossing.y_root = curYroot;
         event->crossing.mode = GDK_CROSSING_NORMAL;
         event->crossing.detail = GDK_NOTIFY_NONLINEAR;

         event->crossing.focus = TRUE; /* ??? */
         event->crossing.state = 0;    /* ??? */
         return_val = TRUE;
      }

      if (curWnd) {
         gdk_window_unref(curWnd);
         curWnd = NULL;
      }

      break;

   case WM_MOUSEWHEEL:
      GDK_NOTE(EVENTS, g_print("WM_MOUSEWHEEL: %#x %d\n",
                               xevent->hwnd, HIWORD(xevent->wParam)));

      event->type = GDK_SCROLL;
      event->scroll.type = GDK_SCROLL;

      /* WM_MOUSEWHEEL seems to be delivered to the focus window Work
       * around that, we want it to the window that the mouse is
       * in. Also, the position is in screen coordinates, not client
       * coordinates as with the button messages. I love the
       * consistency of Windows.
       */
      pt.x = GET_X_LPARAM(xevent->lParam);
      pt.y = GET_Y_LPARAM(xevent->lParam);
      if ((hwnd = WindowFromPoint(pt)) == NULL)
         break;

      xevent->hwnd = hwnd;
      if ((newwindow = gdk_window_lookup(xevent->hwnd)) == NULL)
         break;

      if (newwindow != window) {
         gdk_window_unref(window);
         window = newwindow;
         gdk_window_ref(window);
      }

      if (((GdkWindowPrivate *) window)->extension_events != 0
          && gdk_input_ignore_core) {
         GDK_NOTE(EVENTS, g_print("...ignored\n"));
         break;
      }

      //vo check if button is grabbed
      if (!p_grab_window &&
          is_grabbed_button(&window, 0, GDK_ALL_EVENTS_MASK, xevent->time)) {
      /* // bb removed 17.09.04 : side effects !
          gdk_pointer_grab(window, GDK_WINDOW_WIN32DATA(window)->grab_owner_events,
                          GDK_WINDOW_WIN32DATA(window)->grab_event_mask,
                          GDK_WINDOW_WIN32DATA(window)->grab_confine,
                          GDK_WINDOW_WIN32DATA(window)->grab_cursor, 0);
         p_grab_automatic = TRUE;
      */
      } else if (!propagate(&window, xevent,
                     p_grab_window, p_grab_owner_events, p_grab_mask,
                     doesnt_want_scroll)) {
         break;
      }

      ScreenToClient(xevent->hwnd, &pt);
      event->button.window = window;
      event->scroll.direction = (((short) HIWORD(xevent->wParam)) > 0) ?
          GDK_SCROLL_UP : GDK_SCROLL_DOWN;
      event->scroll.window = window;
      event->scroll.time = xevent->time;
      event->scroll.x = (gint16) pt.x;
      event->scroll.y = (gint16) pt.y;
      event->scroll.x_root = (gint16) GET_X_LPARAM(xevent->lParam);
      event->scroll.y_root = (gint16) GET_Y_LPARAM(xevent->lParam);
      event->scroll.pressure = 0.5;
      event->scroll.xtilt = 0;
      event->scroll.ytilt = 0;
      event->scroll.state = build_pointer_event_state(xevent);
      event->scroll.source = GDK_SOURCE_MOUSE;
      event->scroll.deviceid = GDK_CORE_POINTER;
      return_val = !GDK_DRAWABLE_DESTROYED(window);

      break;

#ifdef USE_TRACKMOUSEEVENT
   case WM_MOUSELEAVE:
      GDK_NOTE(EVENTS, g_print("WM_MOUSELEAVE: %#x\n", xevent->hwnd));

      if (!
          (GDK_WINDOW_WIN32DATA(window)->
           event_mask & GDK_LEAVE_NOTIFY_MASK))
         break;

      event->type = GDK_LEAVE_NOTIFY;
      event->crossing.type = GDK_LEAVE_NOTIFY;
      event->crossing.window = window;
      event->crossing.subwindow = NULL;
      event->crossing.time = xevent->time;
      event->crossing.x = curX;
      event->crossing.y = curY;
      event->crossing.x_root = curXroot;
      event->crossing.y_root = curYroot;
      event->crossing.mode = GDK_CROSSING_NORMAL;
      if (curWnd
          && IsChild(GDK_DRAWABLE_XID(curWnd), GDK_DRAWABLE_XID(window)))
         event->crossing.detail = GDK_NOTIFY_INFERIOR;
      else if (curWnd
               && IsChild(GDK_DRAWABLE_XID(window),
                          GDK_DRAWABLE_XID(curWnd)))
         event->crossing.detail = GDK_NOTIFY_ANCESTOR;
      else
         event->crossing.detail = GDK_NOTIFY_NONLINEAR;

      event->crossing.focus = TRUE; /* ??? */
      event->crossing.state = 0;    /* ??? */

      if (curWnd) {
         gdk_window_unref(curWnd);
         curWnd = NULL;
      }

      return_val = !GDK_DRAWABLE_DESTROYED(window);
      break;
#endif

   case WM_SETFOCUS:
   case WM_KILLFOCUS:
      GDK_NOTE(EVENTS, g_print("WM_%sFOCUS: %#x\n",
                               (xevent->message == WM_SETFOCUS ?
                                "SET" : "KILL"), xevent->hwnd));
      if (!
          (GDK_WINDOW_WIN32DATA(window)->
           event_mask & GDK_FOCUS_CHANGE_MASK))
         break;

      event->type = GDK_FOCUS_CHANGE;
      event->focus_change.type = GDK_FOCUS_CHANGE;
      event->focus_change.window = window;
      event->focus_change.in = (xevent->message == WM_SETFOCUS);
      return_val = !GDK_DRAWABLE_DESTROYED(window);
      break;

   case WM_ERASEBKGND:
      GDK_NOTE(EVENTS, g_print("WM_ERASEBKGND: %#x  dc %#x\n",
                               xevent->hwnd, xevent->wParam));

      if (GDK_DRAWABLE_DESTROYED(window))
         break;

      colormap_private =
          (GdkColormapPrivateWin32 *) ((GdkWindowPrivate *) window)->
          drawable.colormap;
      hdc = (HDC) xevent->wParam;
      if (colormap_private && colormap_private->xcolormap->rc_palette) {
         int k;

         if (SelectPalette(hdc, colormap_private->xcolormap->palette,
                           FALSE) == NULL)
            WIN32_GDI_FAILED("SelectPalette");
         if ((k = RealizePalette(hdc)) == GDI_ERROR)
            WIN32_GDI_FAILED("RealizePalette");
#if 0
         g_print("WM_ERASEBKGND: selected %#x, realized %d colors\n",
                 colormap_private->xcolormap->palette, k);
#endif
      }
      *ret_val_flagp = TRUE;
      *ret_valp = 1;

      if (GDK_WINDOW_WIN32DATA(window)->bg_type ==
          GDK_WIN32_BG_TRANSPARENT)
         break;

      if (GDK_WINDOW_WIN32DATA(window)->bg_type ==
          GDK_WIN32_BG_PARENT_RELATIVE) {
         /* If this window should have the same background as the
          * parent, fetch the parent. (And if the same goes for
          * the parent, fetch the grandparent, etc.)
          */
         while (window
                && GDK_WINDOW_WIN32DATA(window)->bg_type ==
                GDK_WIN32_BG_PARENT_RELATIVE) {
            gdk_window_unref(window);
            window = ((GdkWindowPrivate *) window)->parent;
            gdk_window_ref(window);
         }
      }

      if (GDK_WINDOW_WIN32DATA(window)->bg_type == GDK_WIN32_BG_PIXEL) {
         bg = gdk_colormap_color(colormap_private,
                                 GDK_WINDOW_WIN32DATA(window)->bg_pixel);

         GetClipBox(hdc, &rect);
         GDK_NOTE(EVENTS,
                  g_print("...%dx%d@+%d+%d BG_PIXEL %.06x\n",
                          rect.right - rect.left,
                          rect.bottom - rect.top,
                          rect.left, rect.top, bg));
         hbr = CreateSolidBrush(bg);
#if 0
         g_print("...CreateSolidBrush (%.08x) = %.08x\n", bg, hbr);
#endif
         if (!FillRect(hdc, &rect, hbr))
            WIN32_GDI_FAILED("FillRect");
         DeleteObject(hbr);
      } else if (GDK_WINDOW_WIN32DATA(window)->bg_type ==
                 GDK_WIN32_BG_PIXMAP) {
         pixmap = GDK_WINDOW_WIN32DATA(window)->bg_pixmap;
         pixmap_private = (GdkDrawablePrivate *) pixmap;
         GetClipBox(hdc, &rect);

         if (pixmap_private->width <= 8 && pixmap_private->height <= 8) {
            GDK_NOTE(EVENTS, g_print("...small pixmap, using brush\n"));
            hbr = CreatePatternBrush(GDK_DRAWABLE_XID(pixmap));
            if (!FillRect(hdc, &rect, hbr))
               WIN32_GDI_FAILED("FillRect");
            DeleteObject(hbr);
         } else {
            GDK_NOTE(EVENTS,
                     g_print("...blitting pixmap %#x (%dx%d) "
                             "all over the place,\n"
                             "...clip box = %dx%d@+%d+%d\n",
                             GDK_DRAWABLE_XID(pixmap),
                             pixmap_private->width, pixmap_private->height,
                             rect.right - rect.left,
                             rect.bottom - rect.top, rect.left, rect.top));

            if (!(bgdc = CreateCompatibleDC(hdc))) {
               WIN32_GDI_FAILED("CreateCompatibleDC");
               break;
            }
            if (!
                (oldbitmap =
                 SelectObject(bgdc, GDK_DRAWABLE_XID(pixmap)))) {
               WIN32_GDI_FAILED("SelectObject");
               DeleteDC(bgdc);
               break;
            }
            i = 0;
            while (i < rect.right) {
               j = 0;
               while (j < rect.bottom) {
                  if (i + pixmap_private->width >= rect.left
                      && j + pixmap_private->height >= rect.top) {
                     if (!BitBlt(hdc, i, j,
                                 pixmap_private->width,
                                 pixmap_private->height, bgdc, 0, 0,
                                 SRCCOPY)) {
                        WIN32_GDI_FAILED("BitBlt");
                        goto loopexit;
                     }
                  }
                  j += pixmap_private->height;
               }
               i += pixmap_private->width;
            }
          loopexit:
            SelectObject(bgdc, oldbitmap);
            DeleteDC(bgdc);
         }
      } else {
         GDK_NOTE(EVENTS, g_print("...BLACK_BRUSH (?)\n"));
         hbr = GetStockObject(BLACK_BRUSH);
         GetClipBox(hdc, &rect);
         if (!FillRect(hdc, &rect, hbr))
            WIN32_GDI_FAILED("FillRect");
      }
      break;

   case WM_PAINT:
      hdc = BeginPaint(xevent->hwnd, &paintstruct);

      GDK_NOTE(EVENTS,
               g_print("WM_PAINT: %#x  %dx%d@+%d+%d %s dc %#x\n",
                       xevent->hwnd,
                       paintstruct.rcPaint.right -
                       paintstruct.rcPaint.left,
                       paintstruct.rcPaint.bottom -
                       paintstruct.rcPaint.top, paintstruct.rcPaint.left,
                       paintstruct.rcPaint.top,
                       (paintstruct.fErase ? "erase" : ""), hdc));

      EndPaint(xevent->hwnd, &paintstruct);

      if (!(GDK_WINDOW_WIN32DATA(window)->event_mask & GDK_EXPOSURE_MASK))
         break;

      event->type = GDK_EXPOSE;
      event->expose.type = GDK_EXPOSE;
      event->expose.window = window;
      event->expose.area.x = paintstruct.rcPaint.left;
      event->expose.area.y = paintstruct.rcPaint.top;
      event->expose.area.width =
          paintstruct.rcPaint.right - paintstruct.rcPaint.left;
      event->expose.area.height =
          paintstruct.rcPaint.bottom - paintstruct.rcPaint.top;
      event->expose.count = 0;

      return_val = !GDK_DRAWABLE_DESTROYED(window);
      if (return_val) {
         GList *list = gdk_queued_events;
         while (list != NULL) {
            if ((((GdkEvent *) list->data)->any.type == GDK_EXPOSE) &&
                (((GdkEvent *) list->data)->any.window == window) &&
                !(((GdkEventPrivate *) list->data)->
                  flags & GDK_EVENT_PENDING))
               ((GdkEvent *) list->data)->expose.count++;

            list = list->next;
         }
      }
      break;

   case WM_SETCURSOR:
      GDK_NOTE(EVENTS, g_print("WM_SETCURSOR: %#x %#x %#x\n",
                               xevent->hwnd,
                               LOWORD(xevent->lParam),
                               HIWORD(xevent->lParam)));

      if (LOWORD(xevent->lParam) != HTCLIENT)
         break;

      if (p_grab_window != NULL && p_grab_cursor != NULL)
         xcursor = p_grab_cursor;
      else if (!GDK_DRAWABLE_DESTROYED(window))
         xcursor = GDK_WINDOW_WIN32DATA(window)->xcursor;
      else
         xcursor = NULL;

      /*if (p_grab_cursor != NULL) { //vo
         DestroyCursor (p_grab_cursor);
         p_grab_cursor = NULL;
      }*/

      if (xcursor != NULL) {
         GDK_NOTE(EVENTS, g_print("...SetCursor(%#x)\n", xcursor));
         SetCursor(xcursor);
         *ret_val_flagp = TRUE;
         *ret_valp = TRUE;
      }
      break;

   case WM_SHOWWINDOW:
      GDK_NOTE(EVENTS, g_print("WM_SHOWWINDOW: %#x  %d\n",
                               xevent->hwnd, xevent->wParam));

      if (!(GDK_WINDOW_WIN32DATA(window)->event_mask & GDK_STRUCTURE_MASK))
         break;

      event->any.type = (xevent->wParam ? GDK_MAP : GDK_UNMAP);
      event->any.window = window;
      event->type = event->any.type;

      if (event->any.type == GDK_UNMAP && p_grab_window == window)
         gdk_pointer_ungrab(xevent->time);

      if (event->any.type == GDK_UNMAP && k_grab_window == window)
         gdk_keyboard_ungrab(xevent->time);

      return_val = !GDK_DRAWABLE_DESTROYED(window);
      break;

   case WM_SIZE:
      GDK_NOTE(EVENTS,
               g_print("WM_SIZE: %#x  %s %dx%d\n",
                       xevent->hwnd,
                       (xevent->wParam == SIZE_MAXHIDE ? "MAXHIDE" :
                        (xevent->wParam == SIZE_MAXIMIZED ? "MAXIMIZED" :
                         (xevent->wParam == SIZE_MAXSHOW ? "MAXSHOW" :
                          (xevent->wParam == SIZE_MINIMIZED ? "MINIMIZED" :
                           (xevent->wParam ==
                            SIZE_RESTORED ? "RESTORED" : "?"))))),
                       LOWORD(xevent->lParam), HIWORD(xevent->lParam)));

      if (!(GDK_WINDOW_WIN32DATA(window)->event_mask & GDK_STRUCTURE_MASK))
         break;

      if (xevent->wParam == SIZE_MINIMIZED) {
         event->any.type = GDK_UNMAP;
         event->any.window = window;
         event->type = GDK_UNMAP;

         if (p_grab_window == window)
            gdk_pointer_ungrab(xevent->time);

         if (k_grab_window == window)
            gdk_keyboard_ungrab(xevent->time);

         return_val = !GDK_DRAWABLE_DESTROYED(window);
      } else if ((xevent->wParam == SIZE_RESTORED
                  || xevent->wParam == SIZE_MAXIMIZED)
#if 1
                 && GDK_DRAWABLE_TYPE(window) != GDK_WINDOW_CHILD
#endif
          ) {
         if (LOWORD(xevent->lParam) == 0)
            break;

         event->type = GDK_CONFIGURE;
         event->configure.type = GDK_CONFIGURE;
         event->configure.window = window;
         pt.x = 0;
         pt.y = 0;
         ClientToScreen(xevent->hwnd, &pt);
         event->configure.x = pt.x;
         event->configure.y = pt.y;
         event->configure.width = LOWORD(xevent->lParam);
         event->configure.height = HIWORD(xevent->lParam);
         ((GdkWindowPrivate *) window)->x = event->configure.x;
         ((GdkWindowPrivate *) window)->y = event->configure.y;
         ((GdkWindowPrivate *) window)->drawable.width =
             event->configure.width;
         ((GdkWindowPrivate *) window)->drawable.height =
             event->configure.height;
         if (((GdkWindowPrivate *) window)->resize_count > 1)
            ((GdkWindowPrivate *) window)->resize_count -= 1;

         return_val = !GDK_DRAWABLE_DESTROYED(window);
         if (return_val
             && ((GdkWindowPrivate *) window)->extension_events != 0
             && gdk_input_vtable.configure_event)
            gdk_input_vtable.configure_event(&event->configure, window);
      } else if ((xevent->wParam == SIZE_RESTORED
                  || xevent->wParam == SIZE_MAXIMIZED)
                 && GDK_DRAWABLE_TYPE(window) == GDK_WINDOW_CHILD) {
         event->type = GDK_CONFIGURE;
         event->configure.type = GDK_CONFIGURE;
         event->configure.window = window;
         GetClientRect(xevent->hwnd, &rect);
         event->configure.x = rect.left;
         event->configure.y = rect.top;
         event->configure.width = rect.right;
         event->configure.height = rect.bottom;
         ((GdkWindowPrivate *) window)->x = event->configure.x;
         ((GdkWindowPrivate *) window)->y = event->configure.y;
         ((GdkWindowPrivate *) window)->drawable.width =
             event->configure.width;
         ((GdkWindowPrivate *) window)->drawable.height =
             event->configure.height;

         return_val = !GDK_DRAWABLE_DESTROYED(window);
      }
      break;

   case WM_GETMINMAXINFO:
      GDK_NOTE(EVENTS, g_print("WM_GETMINMAXINFO: %#x\n", xevent->hwnd));

      lpmmi = (MINMAXINFO *) xevent->lParam;
      if (GDK_WINDOW_WIN32DATA(window)->hint_flags & GDK_HINT_MIN_SIZE) {
         lpmmi->ptMinTrackSize.x =
             GDK_WINDOW_WIN32DATA(window)->hint_min_width;
         lpmmi->ptMinTrackSize.y =
             GDK_WINDOW_WIN32DATA(window)->hint_min_height;
      }
      if (GDK_WINDOW_WIN32DATA(window)->hint_flags & GDK_HINT_MAX_SIZE) {
         lpmmi->ptMaxTrackSize.x =
             GDK_WINDOW_WIN32DATA(window)->hint_max_width;
         lpmmi->ptMaxTrackSize.y =
             GDK_WINDOW_WIN32DATA(window)->hint_max_height;

         lpmmi->ptMaxSize.x = GDK_WINDOW_WIN32DATA(window)->hint_max_width;
         lpmmi->ptMaxSize.y =
             GDK_WINDOW_WIN32DATA(window)->hint_max_height;
      }
      break;

   case WM_MOVE:
      GDK_NOTE(EVENTS, g_print("WM_MOVE: %#x  (%d,%d)\n",
                               xevent->hwnd,
                               GET_X_LPARAM(xevent->lParam),
                               GET_Y_LPARAM(xevent->lParam)));

      if (!(GDK_WINDOW_WIN32DATA(window)->event_mask & GDK_STRUCTURE_MASK))
         break;

      if (GDK_DRAWABLE_TYPE(window) != GDK_WINDOW_CHILD
          && !IsIconic(xevent->hwnd)
          && IsWindowVisible(xevent->hwnd)) {
         event->type = GDK_CONFIGURE;
         event->configure.type = GDK_CONFIGURE;
         event->configure.window = window;
         event->configure.x = GET_X_LPARAM(xevent->lParam);
         event->configure.y = GET_Y_LPARAM(xevent->lParam);
         GetClientRect(xevent->hwnd, &rect);
         event->configure.width = rect.right;
         event->configure.height = rect.bottom;
         ((GdkWindowPrivate *) window)->x = event->configure.x;
         ((GdkWindowPrivate *) window)->y = event->configure.y;
         ((GdkWindowPrivate *) window)->drawable.width =
             event->configure.width;
         ((GdkWindowPrivate *) window)->drawable.height =
             event->configure.height;

         return_val = !GDK_DRAWABLE_DESTROYED(window);
      }
      break;

   case WM_CLOSE:
      GDK_NOTE(EVENTS, g_print("WM_CLOSE: %#x\n", xevent->hwnd));

      event->type = GDK_DELETE;
      event->any.type = GDK_DELETE;
      event->any.window = window;

      return_val = !GDK_DRAWABLE_DESTROYED(window);
      break;

#if 0
      /* No, don't use delayed rendering after all. It works only if the
       * delayed SetClipboardData is called from the WindowProc, it
       * seems. (The #else part below is test code for that. It succeeds
       * in setting the clipboard data. But if I call SetClipboardData
       * in gdk_property_change (as a consequence of the
       * GDK_SELECTION_REQUEST event), it fails.  I deduce that this is
       * because delayed rendering requires that SetClipboardData is
       * called in the window procedure.)
       */
   case WM_RENDERFORMAT:
   case WM_RENDERALLFORMATS:
      flag = FALSE;
      GDK_NOTE(EVENTS, flag = TRUE);
      if (flag)
         g_print("WM_%s: %#x %#x (%s)\n",
                 (xevent->message == WM_RENDERFORMAT ? "RENDERFORMAT" :
                  "RENDERALLFORMATS"),
                 xevent->hwnd,
                 xevent->wParam,
                 (xevent->wParam == CF_TEXT ? "CF_TEXT" :
                  (xevent->wParam == CF_DIB ? "CF_DIB" :
                   (xevent->wParam == CF_UNICODETEXT ? "CF_UNICODETEXT" :
                    (GetClipboardFormatName
                     (xevent->wParam, buf, sizeof(buf)), buf)))));

#if 0
      event->selection.type = GDK_SELECTION_REQUEST;
      event->selection.window = window;
      event->selection.selection = gdk_clipboard_atom;
      if (xevent->wParam == CF_TEXT)
         event->selection.target = GDK_TARGET_STRING;
      else {
         GetClipboardFormatName(xevent->wParam, buf, sizeof(buf));
         event->selection.target = gdk_atom_intern(buf, FALSE);
      }
      event->selection.property = gdk_selection_property;
      event->selection.requestor = (guint32) xevent->hwnd;
      event->selection.time = xevent->time;
      return_val = !GDK_DRAWABLE_DESTROYED(window);
#else
      /* Test code, to see if SetClipboardData works when called from
       * the window procedure.
       */
      {
         HGLOBAL hdata = GlobalAlloc(GMEM_MOVEABLE | GMEM_DDESHARE, 10);
         char *ptr = GlobalLock(hdata);
         strcpy(ptr, "Huhhaa");
         GlobalUnlock(hdata);
         if (!SetClipboardData(CF_TEXT, hdata))
            WIN32_API_FAILED("SetClipboardData");
      }
      *ret_valp = 0;
      *ret_val_flagp = TRUE;
      return_val = FALSE;
#endif
      break;
#endif                          /* No delayed rendering */

   case WM_DROPFILES:
      {
         DWORD i, len;
         char tmp[256];
         char uri[256];
         HDROP hDrop;
         HGLOBAL hdata;
         char *ptr;
         POINT pt;
         GdkAtom *types;
         GdkEvent *event0, *event1, *event2;

         hDrop = (HDROP)xevent->wParam;
         types = calloc(sizeof(GdkAtom), 3);
         types[0] = gdk_atom_intern("text/uri-list", FALSE);
         types[1] = types[2] = 0;
         event0 = gdk_event_new();
         event0->type = GDK_CLIENT_EVENT;
         event0->any.window = window;
         event0->client.type = GDK_CLIENT_EVENT;
         event0->client.window = window;
         event0->client.message_type = gdk_atom_intern("XdndEnter", TRUE);
         event0->client.data_format = 0;
         event0->client.data.l[0] = (long)window;
         event0->client.data.l[1] = 0;
         event0->client.data.l[2] = (long)*types;
         event0->client.data.l[3] = 0;
         event0->client.data.l[4] = 0;
         gdk_window_ref(window);
         gdk_event_queue_append(event0);

         DragQueryPoint(hDrop, &pt);
         MapWindowPoints(xevent->hwnd, NULL, &pt, 1);
         event1 = gdk_event_new();
         event1->any.window = window;
         event1->type = GDK_CLIENT_EVENT;
         event1->client.type = GDK_CLIENT_EVENT;
         event1->client.window = window;
         event1->client.message_type = gdk_atom_intern("XdndPosition", TRUE);
         event1->client.data_format = 0;
         event1->client.data.l[0] = (long)window;
         event1->client.data.l[1] = 0;
         event1->client.data.l[2] = (long)(pt.y | (pt.x << 16));
         event1->client.data.l[3] = 0; // time
         event1->client.data.l[4] = 0; // action
         gdk_window_ref(window);
         gdk_event_queue_append(event1);

         event2 = gdk_event_new();
         event2->any.window = window;
         event2->type = GDK_CLIENT_EVENT;
         event2->client.type = GDK_CLIENT_EVENT;
         event2->client.window = window;
         event2->client.message_type = gdk_atom_intern("XdndDrop", TRUE);
         event2->client.data_format = 0;
         event2->client.data.l[0] = 0;
         if (OpenClipboard(xevent->hwnd)) {
            DragQueryFile(hDrop, 0, tmp, 256);
            sprintf(uri, "file://%s\r\n", tmp);
            len = strlen(uri);
             // Do anything with the data...
            hdata = GlobalAlloc(GMEM_MOVEABLE | GMEM_DDESHARE, len+1);
            ptr = (char *)GlobalLock(hdata);
            for (i = 0; i < len; i++) {
               ptr[i] = uri[i];
            }
            ptr[len] = '\0';
            GlobalUnlock(hdata);
            SetClipboardData(CF_PRIVATEFIRST, hdata);
            CloseClipboard();
         }
         DragFinish(hDrop);
         gdk_window_ref(window);
         gdk_event_queue_append(event2);
         return_val = FALSE;
      }
      break;

   case WM_DESTROY:
      GDK_NOTE(EVENTS, g_print("WM_DESTROY: %#x\n", xevent->hwnd));

      event->type = GDK_DESTROY;
      event->any.type = GDK_DESTROY;
      event->any.window = window;
      if (window != NULL && window == curWnd) {
         gdk_window_unref(curWnd);
         curWnd = NULL;
      }

      if (p_grab_window == window)
         gdk_pointer_ungrab(xevent->time);

      if (k_grab_window == window)
         gdk_keyboard_ungrab(xevent->time);

      return_val = 1;           //window != NULL && !GDK_DRAWABLE_DESTROYED (window); // bb change

      if (window != NULL)
         gdk_window_destroy_notify(window);

      break;

#ifdef HAVE_WINTAB
      /* Handle WINTAB events here, as we know that gdkinput.c will
       * use the fixed WT_DEFBASE as lcMsgBase, and we thus can use the
       * constants as case labels.
       */
   case WT_PACKET:
      GDK_NOTE(EVENTS, g_print("WT_PACKET: %#x %d %#x\n",
                               xevent->hwnd,
                               xevent->wParam, xevent->lParam));
      goto wintab;

   case WT_CSRCHANGE:
      GDK_NOTE(EVENTS, g_print("WT_CSRCHANGE: %#x %d %#x\n",
                               xevent->hwnd,
                               xevent->wParam, xevent->lParam));
      goto wintab;

   case WT_PROXIMITY:
      GDK_NOTE(EVENTS, g_print("WT_PROXIMITY: %#x %#x %d %d\n",
                               xevent->hwnd, xevent->wParam,
                               LOWORD(xevent->lParam),
                               HIWORD(xevent->lParam)));
      /* Fall through */
    wintab:
      event->any.window = window;
      return_val = gdk_input_vtable.other_event(event, xevent);
      break;
#endif

   default:
      GDK_NOTE(EVENTS, g_print("%s: %#x %#x %#x\n",
                               gdk_win32_message_name(xevent->message),
                               xevent->hwnd,
                               xevent->wParam, xevent->lParam));
   }

 bypass_switch:

   if (return_val) {
      if (event->any.window)
         gdk_window_ref(event->any.window);
      if (((event->any.type == GDK_ENTER_NOTIFY) ||
           (event->any.type == GDK_LEAVE_NOTIFY)) &&
          (event->crossing.subwindow != NULL))
         gdk_window_ref(event->crossing.subwindow);

      GDK_NOTE(EVENTS, print_event(event));
   } else {
      /* Mark this event as having no resources to be freed */
      event->any.window = NULL;
      event->any.type = GDK_NOTHING;
      event->type = GDK_NOTHING;
   }

   if (window)
      gdk_window_unref(window);

   return return_val;
}

void gdk_events_queue(void)
{
   MSG msg;

   while (!gdk_event_queue_find_first()
          && PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
      GDK_NOTE(EVENTS, g_print("PeekMessage: %#x %#x\n",
                               msg.hwnd,
                               gdk_win32_message_name(msg.message)));

      if (paimmmpo == NULL
          || (paimmmpo->lpVtbl->OnTranslateMessage) (paimmmpo,
                                                     &msg) != S_OK)
         TranslateMessage(&msg);

      DispatchMessage(&msg);
   }
}

static gboolean gdk_event_prepare(GSource * source, gint * timeout)
{
   MSG msg;
   gboolean retval;

   GDK_THREADS_ENTER();

   *timeout = -1;

   retval = (gdk_event_queue_find_first() != NULL)
       || PeekMessage(&msg, NULL, 0, 0, PM_NOREMOVE);

   GDK_THREADS_LEAVE();

   return retval;
}

static gboolean gdk_event_check(GSource * source)
{
   MSG msg;
   gboolean retval;

   GDK_THREADS_ENTER();

   if (event_poll_fd.revents & G_IO_IN)
      retval = (gdk_event_queue_find_first() != NULL)
          || PeekMessage(&msg, NULL, 0, 0, PM_NOREMOVE);
   else
      retval = FALSE;

   GDK_THREADS_LEAVE();

   return retval;
}

static gboolean
gdk_event_dispatch(GSource * source,
                   GSourceFunc callback, gpointer user_data)
{
   GdkEvent *event;

   GDK_THREADS_ENTER();

   gdk_events_queue();
   event = gdk_event_unqueue();

   if (event) {
      if (gdk_event_func)
         (*gdk_event_func) (event, gdk_event_data);

      gdk_event_free(event);
   }

   GDK_THREADS_LEAVE();

   return TRUE;
}

/* Sends a ClientMessage to all toplevel client windows */
gboolean gdk_event_send_client_message(GdkEvent * event, guint32 xid)
{
   /* XXX */
   return FALSE;
}

void gdk_event_send_clientmessage_toall(GdkEvent * event)
{
   /* XXX */
}

void gdk_flush(void)
{
   GdiFlush();
}

#ifdef G_ENABLE_DEBUG

gchar *gdk_win32_message_name(UINT msg)
{
   static gchar bfr[100];

   switch (msg) {
#define CASE(x) case x: return #x
      CASE(WM_NULL);
      CASE(WM_CREATE);
      CASE(WM_DESTROY);
      CASE(WM_MOVE);
      CASE(WM_SIZE);
      CASE(WM_ACTIVATE);
      CASE(WM_SETFOCUS);
      CASE(WM_KILLFOCUS);
      CASE(WM_ENABLE);
      CASE(WM_SETREDRAW);
      CASE(WM_SETTEXT);
      CASE(WM_GETTEXT);
      CASE(WM_GETTEXTLENGTH);
      CASE(WM_PAINT);
      CASE(WM_CLOSE);
      CASE(WM_QUERYENDSESSION);
      CASE(WM_QUERYOPEN);
      CASE(WM_ENDSESSION);
      CASE(WM_QUIT);
      CASE(WM_ERASEBKGND);
      CASE(WM_SYSCOLORCHANGE);
      CASE(WM_SHOWWINDOW);
      CASE(WM_WININICHANGE);
      CASE(WM_DEVMODECHANGE);
      CASE(WM_ACTIVATEAPP);
      CASE(WM_FONTCHANGE);
      CASE(WM_TIMECHANGE);
      CASE(WM_CANCELMODE);
      CASE(WM_SETCURSOR);
      CASE(WM_MOUSEACTIVATE);
      CASE(WM_CHILDACTIVATE);
      CASE(WM_QUEUESYNC);
      CASE(WM_GETMINMAXINFO);
      CASE(WM_PAINTICON);
      CASE(WM_ICONERASEBKGND);
      CASE(WM_NEXTDLGCTL);
      CASE(WM_SPOOLERSTATUS);
      CASE(WM_DRAWITEM);
      CASE(WM_MEASUREITEM);
      CASE(WM_DELETEITEM);
      CASE(WM_VKEYTOITEM);
      CASE(WM_CHARTOITEM);
      CASE(WM_SETFONT);
      CASE(WM_GETFONT);
      CASE(WM_SETHOTKEY);
      CASE(WM_GETHOTKEY);
      CASE(WM_QUERYDRAGICON);
      CASE(WM_COMPAREITEM);
      CASE(WM_GETOBJECT);
      CASE(WM_COMPACTING);
      CASE(WM_WINDOWPOSCHANGING);
      CASE(WM_WINDOWPOSCHANGED);
      CASE(WM_POWER);
      CASE(WM_COPYDATA);
      CASE(WM_CANCELJOURNAL);
      CASE(WM_NOTIFY);
      CASE(WM_INPUTLANGCHANGEREQUEST);
      CASE(WM_INPUTLANGCHANGE);
      CASE(WM_TCARD);
      CASE(WM_HELP);
      CASE(WM_USERCHANGED);
      CASE(WM_NOTIFYFORMAT);
      CASE(WM_CONTEXTMENU);
      CASE(WM_STYLECHANGING);
      CASE(WM_STYLECHANGED);
      CASE(WM_DISPLAYCHANGE);
      CASE(WM_GETICON);
      CASE(WM_SETICON);
      CASE(WM_NCCREATE);
      CASE(WM_NCDESTROY);
      CASE(WM_NCCALCSIZE);
      CASE(WM_NCHITTEST);
      CASE(WM_NCPAINT);
      CASE(WM_NCACTIVATE);
      CASE(WM_GETDLGCODE);
      CASE(WM_SYNCPAINT);
      CASE(WM_NCMOUSEMOVE);
      CASE(WM_NCLBUTTONDOWN);
      CASE(WM_NCLBUTTONUP);
      CASE(WM_NCLBUTTONDBLCLK);
      CASE(WM_NCRBUTTONDOWN);
      CASE(WM_NCRBUTTONUP);
      CASE(WM_NCRBUTTONDBLCLK);
      CASE(WM_NCMBUTTONDOWN);
      CASE(WM_NCMBUTTONUP);
      CASE(WM_NCMBUTTONDBLCLK);
      CASE(WM_NCXBUTTONDOWN);
      CASE(WM_NCXBUTTONUP);
      CASE(WM_NCXBUTTONDBLCLK);
      CASE(WM_KEYDOWN);
      CASE(WM_KEYUP);
      CASE(WM_CHAR);
      CASE(WM_DEADCHAR);
      CASE(WM_SYSKEYDOWN);
      CASE(WM_SYSKEYUP);
      CASE(WM_SYSCHAR);
      CASE(WM_SYSDEADCHAR);
      CASE(WM_KEYLAST);
      CASE(WM_IME_STARTCOMPOSITION);
      CASE(WM_IME_ENDCOMPOSITION);
      CASE(WM_IME_COMPOSITION);
      CASE(WM_INITDIALOG);
      CASE(WM_COMMAND);
      CASE(WM_SYSCOMMAND);
      CASE(WM_TIMER);
      CASE(WM_HSCROLL);
      CASE(WM_VSCROLL);
      CASE(WM_INITMENU);
      CASE(WM_INITMENUPOPUP);
      CASE(WM_MENUSELECT);
      CASE(WM_MENUCHAR);
      CASE(WM_ENTERIDLE);
      CASE(WM_MENURBUTTONUP);
      CASE(WM_MENUDRAG);
      CASE(WM_MENUGETOBJECT);
      CASE(WM_UNINITMENUPOPUP);
      CASE(WM_MENUCOMMAND);
      CASE(WM_CHANGEUISTATE);
      CASE(WM_UPDATEUISTATE);
      CASE(WM_QUERYUISTATE);
      CASE(WM_CTLCOLORMSGBOX);
      CASE(WM_CTLCOLOREDIT);
      CASE(WM_CTLCOLORLISTBOX);
      CASE(WM_CTLCOLORBTN);
      CASE(WM_CTLCOLORDLG);
      CASE(WM_CTLCOLORSCROLLBAR);
      CASE(WM_CTLCOLORSTATIC);
      CASE(WM_MOUSEMOVE);
      CASE(WM_LBUTTONDOWN);
      CASE(WM_LBUTTONUP);
      CASE(WM_LBUTTONDBLCLK);
      CASE(WM_RBUTTONDOWN);
      CASE(WM_RBUTTONUP);
      CASE(WM_RBUTTONDBLCLK);
      CASE(WM_MBUTTONDOWN);
      CASE(WM_MBUTTONUP);
      CASE(WM_MBUTTONDBLCLK);
      CASE(WM_MOUSEWHEEL);
      CASE(WM_XBUTTONDOWN);
      CASE(WM_XBUTTONUP);
      CASE(WM_XBUTTONDBLCLK);
      CASE(WM_PARENTNOTIFY);
      CASE(WM_ENTERMENULOOP);
      CASE(WM_EXITMENULOOP);
      CASE(WM_NEXTMENU);
      CASE(WM_SIZING);
      CASE(WM_CAPTURECHANGED);
      CASE(WM_MOVING);
      CASE(WM_POWERBROADCAST);
      CASE(WM_DEVICECHANGE);
      CASE(WM_MDICREATE);
      CASE(WM_MDIDESTROY);
      CASE(WM_MDIACTIVATE);
      CASE(WM_MDIRESTORE);
      CASE(WM_MDINEXT);
      CASE(WM_MDIMAXIMIZE);
      CASE(WM_MDITILE);
      CASE(WM_MDICASCADE);
      CASE(WM_MDIICONARRANGE);
      CASE(WM_MDIGETACTIVE);
      CASE(WM_MDISETMENU);
      CASE(WM_ENTERSIZEMOVE);
      CASE(WM_EXITSIZEMOVE);
      CASE(WM_DROPFILES);
      CASE(WM_MDIREFRESHMENU);
      CASE(WM_IME_SETCONTEXT);
      CASE(WM_IME_NOTIFY);
      CASE(WM_IME_CONTROL);
      CASE(WM_IME_COMPOSITIONFULL);
      CASE(WM_IME_SELECT);
      CASE(WM_IME_CHAR);
      CASE(WM_IME_REQUEST);
      CASE(WM_IME_KEYDOWN);
      CASE(WM_IME_KEYUP);
      CASE(WM_MOUSEHOVER);
      CASE(WM_MOUSELEAVE);
      CASE(WM_NCMOUSEHOVER);
      CASE(WM_NCMOUSELEAVE);
      CASE(WM_CUT);
      CASE(WM_COPY);
      CASE(WM_PASTE);
      CASE(WM_CLEAR);
      CASE(WM_UNDO);
      CASE(WM_RENDERFORMAT);
      CASE(WM_RENDERALLFORMATS);
      CASE(WM_DESTROYCLIPBOARD);
      CASE(WM_DRAWCLIPBOARD);
      CASE(WM_PAINTCLIPBOARD);
      CASE(WM_VSCROLLCLIPBOARD);
      CASE(WM_SIZECLIPBOARD);
      CASE(WM_ASKCBFORMATNAME);
      CASE(WM_CHANGECBCHAIN);
      CASE(WM_HSCROLLCLIPBOARD);
      CASE(WM_QUERYNEWPALETTE);
      CASE(WM_PALETTEISCHANGING);
      CASE(WM_PALETTECHANGED);
      CASE(WM_HOTKEY);
      CASE(WM_PRINT);
      CASE(WM_PRINTCLIENT);
      CASE(WM_APPCOMMAND);
      CASE(WM_HANDHELDFIRST);
      CASE(WM_HANDHELDLAST);
      CASE(WM_AFXFIRST);
      CASE(WM_AFXLAST);
      CASE(WM_PENWINFIRST);
      CASE(WM_PENWINLAST);
      CASE(WM_APP);
#undef CASE
   default:
      if (msg >= WM_HANDHELDFIRST && msg <= WM_HANDHELDLAST)
         sprintf(bfr, "WM_HANDHELDFIRST+%d", msg - WM_HANDHELDFIRST);
      else if (msg >= WM_AFXFIRST && msg <= WM_AFXLAST)
         sprintf(bfr, "WM_AFXFIRST+%d", msg - WM_AFXFIRST);
      else if (msg >= WM_PENWINFIRST && msg <= WM_PENWINLAST)
         sprintf(bfr, "WM_PENWINFIRST+%d", msg - WM_PENWINFIRST);
      else if (msg >= WM_USER && msg <= 0x7FFF)
         sprintf(bfr, "WM_USER+%d", msg - WM_USER);
      else if (msg >= 0xC000 && msg <= 0xFFFF)
         sprintf(bfr, "reg-%#x", msg);
      else
         sprintf(bfr, "unk-%#x", msg);
      return bfr;
   }
   g_assert_not_reached();
}

#endif                          /* G_ENABLE_DEBUG */
