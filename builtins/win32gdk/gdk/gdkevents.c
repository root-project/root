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

#include "gdk.h"
#include "gdkprivate.h"
#include <stdio.h>

typedef struct _GdkIOClosure GdkIOClosure;
typedef struct _GdkEventPrivate GdkEventPrivate;

#define DOUBLE_CLICK_TIME      250
#define TRIPLE_CLICK_TIME      500
#define DOUBLE_CLICK_DIST      5
#define TRIPLE_CLICK_DIST      5

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

GdkFilterReturn gdk_wm_protocols_filter(GdkXEvent * xev,
                                        GdkEvent * event, gpointer data);

/* Private variable declarations
 */

static guint32 button_click_time[2] = { 0, 0 };	/* The last 2 button click times. Used
	 * to determine if the latest button click
	 * is part of a double or triple click.
	 */
static GdkWindow *button_window[2] = { NULL, NULL };	/* The last 2 windows to receive button presses.
	 *                                                                                                                                                                                                         Also used to determine if the latest button
	 *                                                                                                                                                                                                         click is part of a double or triple click.
	 */
static guint button_number[2] = { -1, -1 };	/* The last 2 buttons to be pressed.
	 */
GdkEventFunc gdk_event_func = NULL;	/* Callback for events */
gpointer gdk_event_data = NULL;
GDestroyNotify gdk_event_notify = NULL;

GPollFD event_poll_fd;

/*********************************************
 * Functions for maintaining the event queue *
 *********************************************/

/*************************************************************
 * gdk_event_queue_find_first:
 *     Find the first event on the queue that is not still
 *     being filled in.
 *   arguments:
 *     
 *   results:
 *     Pointer to the list node for that event, or NULL
 *************************************************************/

GList *gdk_event_queue_find_first(void)
{
   GList *tmp_list = gdk_queued_events;

   while (tmp_list) {
      GdkEventPrivate *event = tmp_list->data;
      if (!(event->flags & GDK_EVENT_PENDING))
         return tmp_list;

      tmp_list = g_list_next(tmp_list);
   }

   return NULL;
}

/*************************************************************
 * gdk_event_queue_remove_link:
 *     Remove a specified list node from the event queue.
 *   arguments:
 *     node: Node to remove.
 *   results:
 *************************************************************/

void gdk_event_queue_remove_link(GList * node)
{
   if (node->prev)
      node->prev->next = node->next;
   else
      gdk_queued_events = node->next;

   if (node->next)
      node->next->prev = node->prev;
   else
      gdk_queued_tail = node->prev;

}

/*************************************************************
 * gdk_event_queue_append:
 *     Append an event onto the tail of the event queue.
 *   arguments:
 *     event: Event to append.
 *   results:
 *************************************************************/

void gdk_event_queue_append(GdkEvent * event)
{
   gdk_queued_tail = g_list_append(gdk_queued_tail, event);

   if (!gdk_queued_events)
      gdk_queued_events = gdk_queued_tail;
   else
      gdk_queued_tail = gdk_queued_tail->next;
}

/*************************************************************
 * gdk_event_handler_set:
 *     
 *   arguments:
 *     func: Callback function to be called for each event.
 *     data: Data supplied to the function
 *     notify: function called when function is no longer needed
 * 
 *   results:
 *************************************************************/

void
gdk_event_handler_set(GdkEventFunc func,
                      gpointer data, GDestroyNotify notify)
{
   if (gdk_event_notify)
      (*gdk_event_notify) (gdk_event_data);

   gdk_event_func = func;
   gdk_event_data = data;
   gdk_event_notify = notify;
}

/*
 *--------------------------------------------------------------
 * gdk_event_get
 *
 *   Gets the next event.
 *
 * Arguments:
 *
 * Results:
 *   If an event is waiting that we care about, returns 
 *   a pointer to that event, to be freed with gdk_event_free.
 *   Otherwise, returns NULL.
 *
 * Side effects:
 *
 *--------------------------------------------------------------
 */

GdkEvent *gdk_event_get(void)
{
   gdk_events_queue();

   return gdk_event_unqueue();
}

/*
 *--------------------------------------------------------------
 * gdk_event_peek
 *
 *   Gets the next event.
 *
 * Arguments:
 *
 * Results:
 *   If an event is waiting that we care about, returns 
 *   a copy of that event, but does not remove it from
 *   the queue. The pointer is to be freed with gdk_event_free.
 *   Otherwise, returns NULL.
 *
 * Side effects:
 *
 *--------------------------------------------------------------
 */

GdkEvent *gdk_event_peek(void)
{
   GList *tmp_list;

   tmp_list = gdk_event_queue_find_first();

   if (tmp_list)
      return gdk_event_copy(tmp_list->data);
   else
      return NULL;
}

void gdk_event_put(GdkEvent * event)
{
   GdkEvent *new_event;

   g_return_if_fail(event != NULL);

   new_event = gdk_event_copy(event);

   gdk_event_queue_append(new_event);
}

/*
 *--------------------------------------------------------------
 * gdk_event_copy
 *
 *   Copy a event structure into new storage.
 *
 * Arguments:
 *   "event" is the event struct to copy.
 *
 * Results:
 *   A new event structure.  Free it with gdk_event_free.
 *
 * Side effects:
 *   The reference count of the window in the event is increased.
 *
 *--------------------------------------------------------------
 */

static GMemChunk *event_chunk = NULL;

GdkEvent *gdk_event_new(void)
{
   GdkEventPrivate *new_event;

   if (event_chunk == NULL)
      event_chunk = g_mem_chunk_new("events",
                                    sizeof(GdkEventPrivate),
                                    4096, G_ALLOC_AND_FREE);

   new_event = g_chunk_new(GdkEventPrivate, event_chunk);
   new_event->flags = 0;

   return (GdkEvent *) new_event;
}

GdkEvent *gdk_event_copy(GdkEvent * event)
{
   GdkEvent *new_event;

   g_return_val_if_fail(event != NULL, NULL);

   new_event = gdk_event_new();

   *new_event = *event;
   gdk_window_ref(new_event->any.window);

   switch (event->any.type) {
   case GDK_KEY_PRESS:
   case GDK_KEY_RELEASE:
      new_event->key.string = g_strdup(event->key.string);
      break;

   case GDK_ENTER_NOTIFY:
   case GDK_LEAVE_NOTIFY:
      if (event->crossing.subwindow != NULL)
         gdk_window_ref(event->crossing.subwindow);
      break;

   case GDK_DRAG_ENTER:
   case GDK_DRAG_LEAVE:
   case GDK_DRAG_MOTION:
   case GDK_DRAG_STATUS:
   case GDK_DROP_START:
   case GDK_DROP_FINISHED:
      gdk_drag_context_ref(event->dnd.context);
      break;

   default:
      break;
   }

   return new_event;
}

/*
 *--------------------------------------------------------------
 * gdk_event_free
 *
 *   Free a event structure obtained from gdk_event_copy.  Do not use
 *   with other event structures.
 *
 * Arguments:
 *   "event" is the event struct to free.
 *
 * Results:
 *
 * Side effects:
 *   The reference count of the window in the event is decreased and
 *   might be freed, too.
 *
 *-------------------------------------------------------------- */

void gdk_event_free(GdkEvent * event)
{
   g_return_if_fail(event != NULL);

   g_assert(event_chunk != NULL);	/* paranoid */

   if ((event->any.window) && (event->type != GDK_DESTROY))
      gdk_window_unref(event->any.window);

   switch (event->any.type) {
   case GDK_KEY_PRESS:
   case GDK_KEY_RELEASE:
      g_free(event->key.string);
      break;

   case GDK_ENTER_NOTIFY:
   case GDK_LEAVE_NOTIFY:
      if (event->crossing.subwindow != NULL)
         gdk_window_unref(event->crossing.subwindow);
      break;

   case GDK_DRAG_ENTER:
   case GDK_DRAG_LEAVE:
   case GDK_DRAG_MOTION:
   case GDK_DRAG_STATUS:
   case GDK_DROP_START:
   case GDK_DROP_FINISHED:
      gdk_drag_context_unref(event->dnd.context);
      break;

   default:
      break;
   }

   g_mem_chunk_free(event_chunk, event);
}

/*
 *--------------------------------------------------------------
 * gdk_event_get_time:
 *    Get the timestamp from an event.
 *   arguments:
 *     event:
 *   results:
 *    The event's time stamp, if it has one, otherwise
 *    GDK_CURRENT_TIME.
 *--------------------------------------------------------------
 */

guint32 gdk_event_get_time(GdkEvent * event)
{
   if (event)
      switch (event->type) {
      case GDK_MOTION_NOTIFY:
         return event->motion.time;
      case GDK_BUTTON_PRESS:
      case GDK_2BUTTON_PRESS:
      case GDK_3BUTTON_PRESS:
      case GDK_BUTTON_RELEASE:
      case GDK_SCROLL:
         return event->button.time;
      case GDK_KEY_PRESS:
      case GDK_KEY_RELEASE:
         return event->key.time;
      case GDK_ENTER_NOTIFY:
      case GDK_LEAVE_NOTIFY:
         return event->crossing.time;
      case GDK_PROPERTY_NOTIFY:
         return event->property.time;
      case GDK_SELECTION_CLEAR:
      case GDK_SELECTION_REQUEST:
      case GDK_SELECTION_NOTIFY:
         return event->selection.time;
      case GDK_PROXIMITY_IN:
      case GDK_PROXIMITY_OUT:
         return event->proximity.time;
      case GDK_DRAG_ENTER:
      case GDK_DRAG_LEAVE:
      case GDK_DRAG_MOTION:
      case GDK_DRAG_STATUS:
      case GDK_DROP_START:
      case GDK_DROP_FINISHED:
         return event->dnd.time;
      default:                 /* use current time */
         break;
      }

   return GDK_CURRENT_TIME;
}

/*
 *--------------------------------------------------------------
 * gdk_set_show_events
 *
 *   Turns on/off the showing of events.
 *
 * Arguments:
 *   "show_events" is a boolean describing whether or
 *   not to show the events gdk receives.
 *
 * Results:
 *
 * Side effects:
 *   When "show_events" is TRUE, calls to "gdk_event_get"
 *   will output debugging informatin regarding the event
 *   received to stdout.
 *
 *--------------------------------------------------------------
 */

void gdk_set_show_events(gboolean show_events)
{
   if (show_events)
      gdk_debug_flags |= GDK_DEBUG_EVENTS;
   else
      gdk_debug_flags &= ~GDK_DEBUG_EVENTS;
}

gboolean gdk_get_show_events(void)
{
   return (gdk_debug_flags & GDK_DEBUG_EVENTS) != 0;
}

static void gdk_io_destroy(gpointer data)
{
   GdkIOClosure *closure = data;

   if (closure->notify)
      closure->notify(closure->data);

   g_free(closure);
}

/* What do we do with G_IO_NVAL?
 */
#define READ_CONDITION (G_IO_IN | G_IO_HUP | G_IO_ERR)
#define WRITE_CONDITION (G_IO_OUT | G_IO_ERR)
#define EXCEPTION_CONDITION (G_IO_PRI)

static gboolean
gdk_io_invoke(GIOChannel * source, GIOCondition condition, gpointer data)
{
   GdkIOClosure *closure = data;
   GdkInputCondition gdk_cond = 0;

   if (condition & READ_CONDITION)
      gdk_cond |= GDK_INPUT_READ;
   if (condition & WRITE_CONDITION)
      gdk_cond |= GDK_INPUT_WRITE;
   if (condition & EXCEPTION_CONDITION)
      gdk_cond |= GDK_INPUT_EXCEPTION;

   if (closure->condition & gdk_cond)
      closure->function(closure->data, g_io_channel_unix_get_fd(source),
                        gdk_cond);

   return TRUE;
}

gint
gdk_input_add_full(gint source,
                   GdkInputCondition condition,
                   GdkInputFunction function,
                   gpointer data, GdkDestroyNotify destroy)
{
   guint result;
   GdkIOClosure *closure = g_new(GdkIOClosure, 1);
   GIOChannel *channel;
   GIOCondition cond = 0;

   closure->function = function;
   closure->condition = condition;
   closure->notify = destroy;
   closure->data = data;

   if (condition & GDK_INPUT_READ)
      cond |= READ_CONDITION;
   if (condition & GDK_INPUT_WRITE)
      cond |= WRITE_CONDITION;
   if (condition & GDK_INPUT_EXCEPTION)
      cond |= EXCEPTION_CONDITION;

   channel = g_io_channel_unix_new(source);
   result = g_io_add_watch_full(channel, G_PRIORITY_DEFAULT, cond,
                                gdk_io_invoke, closure, gdk_io_destroy);
   g_io_channel_unref(channel);

   return result;
}

gint
gdk_input_add(gint source,
              GdkInputCondition condition,
              GdkInputFunction function, gpointer data)
{
   return gdk_input_add_full(source, condition, function, data, NULL);
}

void gdk_input_remove(gint tag)
{
   g_source_remove(tag);
}

GdkEvent *gdk_event_unqueue(void)
{
   GdkEvent *event = NULL;
   GList *tmp_list;

   tmp_list = gdk_event_queue_find_first();

   if (tmp_list) {
      event = tmp_list->data;
      gdk_event_queue_remove_link(tmp_list);
      g_list_free_1(tmp_list);
   }

   return event;
}

void gdk_synthesize_click(GdkEvent * event, gint nclicks)
{
   GdkEvent temp_event;

   g_return_if_fail(event != NULL);

   temp_event = *event;
   temp_event.type =
       (nclicks == 2) ? GDK_2BUTTON_PRESS : GDK_3BUTTON_PRESS;

   gdk_event_put(&temp_event);
}

void gdk_event_button_generate(GdkEvent * event)
{
   if ((event->button.time < (button_click_time[1] + TRIPLE_CLICK_TIME)) &&
       (event->button.window == button_window[1]) &&
       (event->button.button == button_number[1])) {
      gdk_synthesize_click(event, 3);

      button_click_time[1] = 0;
      button_click_time[0] = 0;
      button_window[1] = NULL;
      button_window[0] = 0;
      button_number[1] = -1;
      button_number[0] = -1;
   } else
       if ((event->button.time <
            (button_click_time[0] + DOUBLE_CLICK_TIME))
           && (event->button.window == button_window[0])
           && (event->button.button == button_number[0])) {
      gdk_synthesize_click(event, 2);

      button_click_time[1] = button_click_time[0];
      button_click_time[0] = event->button.time;
      button_window[1] = button_window[0];
      button_window[0] = event->button.window;
      button_number[1] = button_number[0];
      button_number[0] = event->button.button;
   } else {
      button_click_time[1] = 0;
      button_click_time[0] = event->button.time;
      button_window[1] = NULL;
      button_window[0] = event->button.window;
      button_number[1] = -1;
      button_number[0] = event->button.button;
   }
}

gboolean
gdk_check_typed_window_event(GdkWindow * w, gint type, GdkEvent * event)
{

//    GList *tmp_list = gdk_queued_events;
   GList *tmp_list = gdk_event_queue_find_first();
   while (tmp_list) {
      GdkEventPrivate *ev = tmp_list->data;
      if (ev->event.any.type == type) {
         if (ev->event.any.window == w) {
            *event = ev->event;
            gdk_event_queue_remove_link(tmp_list);
            g_list_free_1(tmp_list);
            return TRUE;
         }
      }
      tmp_list = g_list_next(tmp_list);
   }
   return FALSE;

}
