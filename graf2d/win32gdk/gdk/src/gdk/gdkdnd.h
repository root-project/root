#ifndef __GDK_DND_H__
#define __GDK_DND_H__

#include <gdk/gdktypes.h>

#ifdef __cplusplus
extern "C" {
#endif                          /* __cplusplus */

   typedef struct _GdkDragContext GdkDragContext;

   typedef enum {
      GDK_ACTION_DEFAULT = 1 << 0,
      GDK_ACTION_COPY = 1 << 1,
      GDK_ACTION_MOVE = 1 << 2,
      GDK_ACTION_LINK = 1 << 3,
      GDK_ACTION_PRIVATE = 1 << 4,
      GDK_ACTION_ASK = 1 << 5
   } GdkDragAction;

   typedef enum {
      GDK_DRAG_PROTO_MOTIF,
      GDK_DRAG_PROTO_XDND,
      GDK_DRAG_PROTO_ROOTWIN,   /* A root window with nobody claiming
                                 * drags */
      GDK_DRAG_PROTO_NONE,      /* Not a valid drag window */
      GDK_DRAG_PROTO_WIN32_DROPFILES,	/* The simple WM_DROPFILES dnd */
      GDK_DRAG_PROTO_OLE2,      /* The complex OLE2 dnd (not implemented) */
   } GdkDragProtocol;

/* Structure that holds information about a drag in progress.
 * this is used on both source and destination sides.
 */
   struct _GdkDragContext {
      GdkDragProtocol protocol;

      gboolean is_source;

      GdkWindow *source_window;
      GdkWindow *dest_window;

      GList *targets;
      GdkDragAction actions;
      GdkDragAction suggested_action;
      GdkDragAction action;

      guint32 start_time;
   };

/* Drag and Drop */

   GdkDragContext *gdk_drag_context_new(void);
   void gdk_drag_context_ref(GdkDragContext * context);
   void gdk_drag_context_unref(GdkDragContext * context);

/* Destination side */

   void gdk_drag_status(GdkDragContext * context,
                        GdkDragAction action, guint32 time);
   void gdk_drop_reply(GdkDragContext * context,
                       gboolean ok, guint32 time);
   void gdk_drop_finish(GdkDragContext * context,
                        gboolean success, guint32 time);
   GdkAtom gdk_drag_get_selection(GdkDragContext * context);

/* Source side */

   GdkDragContext *gdk_drag_begin(GdkWindow * window, GList * targets);
   guint32 gdk_drag_get_protocol(guint32 xid, GdkDragProtocol * protocol);
   void gdk_drag_find_window(GdkDragContext * context,
                             GdkWindow * drag_window,
                             gint x_root,
                             gint y_root,
                             GdkWindow ** dest_window,
                             GdkDragProtocol * protocol);
   gboolean gdk_drag_motion(GdkDragContext * context,
                            GdkWindow * dest_window,
                            GdkDragProtocol protocol,
                            gint x_root,
                            gint y_root,
                            GdkDragAction suggested_action,
                            GdkDragAction possible_actions, guint32 time);
   void gdk_drag_drop(GdkDragContext * context, guint32 time);
   void gdk_drag_abort(GdkDragContext * context, guint32 time);

#ifdef __cplusplus
}
#endif                          /* __cplusplus */
#endif	/* __GDK_DND_H__ */
