

#include <wchar.h>
#include "TROOT.h"
#include "TGWin32.h"
#include "gdk/gdkkeysyms.h"
#include "xatom.h"

#ifndef ROOT_GdkConstants
#include "GdkConstants.h"
#endif

//______________________________________________________________________________
BOOL CALLBACK EnumChildProc(HWND hwndChild, LPARAM lParam)
{
   // Make sure the child window is visible.

   ShowWindow(hwndChild, SW_SHOWNORMAL);

   return TRUE;
}

//______________________________________________________________________________
static void _ChangeProperty(HWND w, char *np, char *dp, int n, Atom_t type)
{
   HGLOBAL hMem;
   char *p;

   hMem = GetProp(w, np);
   if (hMem != NULL) {
      GlobalFree(hMem);
   }
   hMem = GlobalAlloc(GHND, n + sizeof(Atom_t));
   p = (char *) GlobalLock(hMem);
   memcpy(p, &type, sizeof(Atom_t));
   memcpy(p + sizeof(Atom_t), dp, n);
   GlobalUnlock(hMem);
   SetProp(w, np, hMem);
}

//______________________________________________________________________________
void W32ChangeProperty(HWND w, Atom_t property, Atom_t type,
                       int format, int mode, const unsigned char *data,
                       int nelements)
{
   char *atomName;
   char buffer[256];
   char *p, *s;
   int len;
   char propName[8];

   if (mode == GDK_PROP_MODE_REPLACE || mode == GDK_PROP_MODE_PREPEND) {
      len = (int) GlobalGetAtomName(property, buffer, sizeof(buffer));
      if ((atomName = (char *) malloc(len + 1)) == NULL) {
         return;
      } else {
         strcpy(atomName, buffer);
      }
      sprintf(propName, "#0x%0.4x", atomName);
      _ChangeProperty(w, propName, (char *) data, nelements, type);
   }
}


//______________________________________________________________________________
int _GetWindowProperty(GdkWindow * id, Atom_t property, Long_t long_offset,
                       Long_t long_length, Bool_t delete_it, Atom_t req_type,
                       Atom_t * actual_type_return,
                       Int_t * actual_format_return, ULong_t * nitems_return,
                       ULong_t * bytes_after_return, UChar_t ** prop_return)
{
   char *atomName;
   char *data, *destPtr;
   char propName[8];
   HGLOBAL handle;
   HGLOBAL hMem;
   HWND w;

   w = (HWND) GDK_DRAWABLE_XID(id);

   if (IsClipboardFormatAvailable(CF_TEXT) && OpenClipboard(NULL)) {
      handle = GetClipboardData(CF_TEXT);
      if (handle != NULL) {
         data = (char *) GlobalLock(handle);
         *nitems_return = strlen(data);
         *prop_return = (UChar_t *) malloc(*nitems_return + 1);
         destPtr = (char *) *prop_return;
         while (*data != '\0') {
            if (*data != '\r') {
               *destPtr = *data;
               destPtr++;
            }
            data++;
         }
         *destPtr = '\0';
         GlobalUnlock(handle);
         *actual_type_return = XA_STRING;
         *bytes_after_return = 0;
      }
      CloseClipboard();
      return 1;
   }
   if (delete_it)
      RemoveProp(w, propName);
   return 1;
}



// Thread handling GDK Calls
void TGWin32::GdkThread( )
{
    MSG msg;
    int erret;
    GdkColor fore, back;
    GdkEventMask masque;
    Bool_t EndLoop = FALSE;
    Int_t depth;
    POINT cpt, tmp;
    HDC hdc;
    RECT srct;
    HWND dw;
    
    PeekMessage(&msg, NULL, WM_USER, WM_USER, PM_NOREMOVE);
    ReleaseSemaphore(fThreadP.hThrSem, 1, NULL);
    while(!EndLoop) {
        erret = GetMessage(&msg, NULL, WIN32_GDK_INIT, WIN32_GDK_QUERY_POINTER1);
        if (erret <= 0) EndLoop = TRUE;
        switch (msg.message) {
            case WIN32_GDK_INIT:
                if(gdk_init_check(NULL, NULL))
                    fThreadP.iRet = kTRUE;
                else
                    fThreadP.iRet = kFALSE;
                break;
            case WIN32_GDK_EXIT:
                EndLoop = TRUE;
                break;
            case WIN32_GDK_ATOM:
                fThreadP.ulRet = gdk_atom_intern(fThreadP.sParam, fThreadP.iParam);
                break;
            case WIN32_GDK_DRAWABLE_GET_SIZE:
                gdk_drawable_get_size((GdkDrawable *) fThreadP.Drawable, &fThreadP.w, &fThreadP.h);
                break;
            case WIN32_GDK_DRAW_RECTANGLE:
                gdk_draw_rectangle((GdkDrawable *) fThreadP.Drawable,
                    (GdkGC *)(fThreadP.GC), fThreadP.bFill, fThreadP.x, fThreadP.y, fThreadP.w, fThreadP.h);
                break;
            case WIN32_GDK_FLUSH:
                gdk_flush();
                break;
            case WIN32_GDK_WIN_SET_BACKGROUND:
                gdk_window_set_background((GdkDrawable *) fThreadP.Drawable,
                    (GdkColor *) & fThreadP.color);
                break;
            case WIN32_GDK_WIN_CLEAR:
                gdk_window_clear((GdkDrawable *) fThreadP.Drawable);
                break;
            case WIN32_GDK_PIX_UNREF:
                gdk_pixmap_unref((GdkPixmap *) fThreadP.Drawable);
                break;
            case WIN32_GDK_WIN_DESTROY:
                gdk_window_destroy((GdkDrawable *) fThreadP.Drawable);
                break;
            case WIN32_GDK_WIN_COPY_AREA:
                gdk_window_copy_area((GdkDrawable *) fThreadP.Drawable,
                        fThreadP.GC, fThreadP.xpos, fThreadP.ypos, 
                        (GdkDrawable *) fThreadP.pParam, fThreadP.x, 
                        fThreadP.y, fThreadP.w, fThreadP.h);
                break;
            case WIN32_GDK_CMAP_FREE_COLORS:
                gdk_colormap_free_colors((GdkColormap *) fThreadP.pParam, 
                    (GdkColor *) fThreadP.pParam2, fThreadP.iParam);
                break;
            case WIN32_GDK_DRAW_LINES:
                gdk_draw_lines(fThreadP.Drawable, fThreadP.GC, (GdkPoint *)fThreadP.pParam, fThreadP.iParam);
                break;
            case WIN32_GDK_FILL_POLYGON:
                gdk_draw_polygon(fThreadP.Drawable, fThreadP.GC, 1, (GdkPoint *)fThreadP.pParam, fThreadP.iParam);
                break;
            case WIN32_GDK_DRAW_LINE:
                gdk_draw_line(fThreadP.Drawable, fThreadP.GC, fThreadP.x1, fThreadP.y1, fThreadP.x2, fThreadP.y2);
                break;
            case WIN32_GDK_GC_SET_DASHES:
                gdk_gc_set_dashes(fThreadP.GC, fThreadP.iParam, fThreadP.dashes, fThreadP.iParam2);
                break;
            case WIN32_GDK_DRAW_POINT:
                gdk_draw_point(fThreadP.Drawable, fThreadP.GC, fThreadP.x, fThreadP.y);
                break;
            case WIN32_GDK_DRAW_POINTS:
                gdk_draw_points(fThreadP.Drawable, fThreadP.GC, (GdkPoint *)fThreadP.pParam, fThreadP.iParam);
                break;
            case WIN32_GDK_DRAW_ARC:
                gdk_draw_arc(fThreadP.Drawable, fThreadP.GC, fThreadP.bFill, fThreadP.x, fThreadP.y,
                    fThreadP.w, fThreadP.h, fThreadP.angle1, fThreadP.angle2);
                break;
            case WIN32_GDK_GET_EVENT:
                fThreadP.pRet = gdk_event_get();
                break;
            case WIN32_GDK_SCREEN_WIDTH:
                fThreadP.w = gdk_screen_width();
                break;
            case WIN32_GDK_SCREEN_HEIGHT:
                fThreadP.h = gdk_screen_height();
                break;
            case WIN32_GDK_WIN_GEOMETRY:
                gdk_window_get_geometry(fThreadP.Drawable, &fThreadP.x, &fThreadP.y, 
                    &fThreadP.w, &fThreadP.h, &fThreadP.iRet);
                break;
            case WIN32_GDK_GET_DESK_ORIGIN:
                gdk_window_get_deskrelative_origin(fThreadP.Drawable, &fThreadP.x, &fThreadP.y);
                break;
            case WIN32_GDK_GET_DEPTH:
                fThreadP.iRet = gdk_visual_get_best_depth();
                break;
            case WIN32_GDK_GET_TEXT_WIDTH:
                fThreadP.iRet = gdk_text_width((GdkFont *)fThreadP.pParam, fThreadP.sParam, fThreadP.iParam);
                break;
            case WIN32_GDK_GET_TEXT_HEIGHT:
                fThreadP.iRet = gdk_text_height((GdkFont *)fThreadP.pParam, fThreadP.sParam, fThreadP.iParam);
                break;
            case WIN32_GDK_WIN_MOVE:
                gdk_window_move(fThreadP.Drawable, fThreadP.x, fThreadP.y);
                break;
            case WIN32_GDK_CMAP_GET_SYSTEM:
                fThreadP.pRet = gdk_colormap_get_system();
                break;
            case WIN32_GDK_COLOR_BLACK:
                gdk_color_black((GdkColormap *)fThreadP.pParam, (GdkColor *) &fThreadP.color);
                break;
            case WIN32_GDK_COLOR_WHITE:
                gdk_color_white((GdkColormap *)fThreadP.pParam, (GdkColor *) &fThreadP.color);
                break;
            case WIN32_GDK_GC_NEW:
                if(fThreadP.pParam == NULL)
                    fThreadP.pRet = gdk_gc_new(GDK_ROOT_PARENT());
                break;
            case WIN32_GDK_GC_SET_FOREGROUND:
                gdk_gc_set_foreground(fThreadP.GC, &fThreadP.color);
                break;
            case WIN32_GDK_GC_SET_BACKGROUND:
                gdk_gc_set_background(fThreadP.GC, &fThreadP.color);
                break;
            case WIN32_GDK_GC_GET_VALUES:
                gdk_gc_get_values(fThreadP.GC, &fThreadP.gcvals);
                break;
            case WIN32_GDK_GC_NEW_WITH_VAL:
                if(fThreadP.Drawable == NULL) {
                    fThreadP.pRet = gdk_gc_new_with_values((GdkWindow *) GDK_ROOT_PARENT(), 
                    &fThreadP.gcvals, (GdkGCValuesMask) fThreadP.iParam);
                }
                else {
                    fThreadP.pRet = gdk_gc_new_with_values((GdkWindow *) fThreadP.Drawable, 
                    &fThreadP.gcvals, (GdkGCValuesMask) fThreadP.iParam);
                }
                break;

            case WIN32_GDK_FONTLIST_NEW:
                fThreadP.pRet = gdk_font_list_new(fThreadP.sParam, &fThreadP.iRet);
                break;

            case WIN32_GDK_FONT_LOAD:
                fThreadP.pRet = gdk_font_load(fThreadP.sParam);
                break;
 
            case WIN32_GDK_FONTLIST_FREE:
                gdk_font_list_free((char **)fThreadP.pParam);
                break;

            case WIN32_GDK_BMP_CREATE_FROM_DATA:
                if(fThreadP.Drawable == NULL) {
                    fThreadP.pRet = gdk_bitmap_create_from_data(GDK_ROOT_PARENT(),
                        (const char *)fThreadP.pParam, fThreadP.w, fThreadP.h);
                }
                else {
                    fThreadP.pRet = gdk_bitmap_create_from_data((GdkWindow *) fThreadP.Drawable,
                        (const char *)fThreadP.pParam, fThreadP.w, fThreadP.h);
                }
                break;

            case WIN32_GDK_CURSOR_NEW_FROM_PIXMAP:
                fThreadP.pRet = gdk_cursor_new_from_pixmap(fThreadP.Drawable, 
                    (GdkDrawable *)fThreadP.pParam, &fore, &back, 0, 0);
                break;

            case WIN32_GDK_CURSOR_NEW:
                fThreadP.pRet = gdk_cursor_new((GdkCursorType)msg.wParam);
                break;

            case WIN32_GDK_PIXMAP_NEW:
                if(fThreadP.iParam == 0)
                    depth = gdk_visual_get_best_depth();
                else
                    depth = fThreadP.iParam;
                if(fThreadP.Drawable == NULL) {
                    fThreadP.pRet = gdk_pixmap_new(GDK_ROOT_PARENT(),
                                    fThreadP.w, fThreadP.h, depth);
                }
                else {
                    fThreadP.pRet = gdk_pixmap_new((GdkWindow *)fThreadP.Drawable,
                                    fThreadP.w, fThreadP.h, depth);
                }
                break;

            case WIN32_GDK_GC_SET_CLIP_MASK:
                gdk_gc_set_clip_mask(fThreadP.GC, (GdkDrawable *)fThreadP.pParam);
                break;

            case WIN32_GDK_WIN_RESIZE:
                gdk_window_resize((GdkWindow *) fThreadP.Drawable, fThreadP.w, fThreadP.h);
                break;

            case WIN32_GDK_GC_SET_CLIP_RECT:
                if(msg.wParam == 1)
                    gdk_gc_set_clip_rectangle(fThreadP.GC, (GdkRectangle *)fThreadP.pParam);
                else
                    gdk_gc_set_clip_rectangle(fThreadP.GC, &fThreadP.region);
                break;

            case WIN32_GDK_WIN_SET_CURSOR:
                gdk_window_set_cursor((GdkWindow *) fThreadP.Drawable, (GdkCursor *)fThreadP.pParam);
                break;

            case WIN32_GDK_GC_SET_FUNCTION:
                gdk_gc_set_function(fThreadP.GC, (GdkFunction)msg.wParam);
                break;

            case WIN32_GDK_GC_SET_FILL:
                gdk_gc_set_fill(fThreadP.GC, (GdkFill)msg.wParam);
                break;

            case WIN32_GDK_GC_SET_STIPPLE:
                gdk_gc_set_stipple(fThreadP.GC, (GdkPixmap *) fThreadP.pParam);
                break;

            case WIN32_GDK_GC_SET_LINE_ATTR:
                gdk_gc_set_line_attributes(fThreadP.GC, fThreadP.w,
                                 (GdkLineStyle) fThreadP.iParam,
                                 (GdkCapStyle) fThreadP.iParam1,
                                 (GdkJoinStyle) fThreadP.iParam2);
                break;

            case WIN32_GDK_DRAW_SEGMENTS:
                gdk_draw_segments(fThreadP.Drawable, fThreadP.GC, 
                    (GdkSegment *)fThreadP.pParam, fThreadP.iParam);
                break;

            case WIN32_GDK_IMAGE_NEW:
                fThreadP.pRet = gdk_image_new(GDK_IMAGE_SHARED, gdk_visual_get_best(), fThreadP.w, fThreadP.h);
                break;

            case WIN32_GDK_IMAGE_GET:
                fThreadP.pRet = gdk_image_get(fThreadP.Drawable, fThreadP.x, fThreadP.y, 
                                                fThreadP.w, fThreadP.h);
                break;

            case WIN32_GDK_IMAGE_GET_PIXEL:
                fThreadP.lRet = gdk_image_get_pixel((GdkImage *)fThreadP.pParam, fThreadP.x, fThreadP.y);
                break;

            case WIN32_GDK_IMAGE_PUT_PIXEL:
                gdk_image_put_pixel((GdkImage *)fThreadP.pParam, fThreadP.x, fThreadP.y, fThreadP.lParam);
                break;

            case WIN32_GDK_DRAW_IMAGE:
                gdk_draw_image(fThreadP.Drawable, fThreadP.GC, (GdkImage *)fThreadP.pParam, fThreadP.x, fThreadP.y, 
                                fThreadP.x1, fThreadP.y1, fThreadP.w, fThreadP.h);
                break;

            case WIN32_GDK_COLORS_FREE:
                gdk_colors_free((GdkColormap *)fThreadP.pParam, (unsigned long *)fThreadP.pParam2, fThreadP.iParam, 0);
                break;

            case WIN32_GDK_IMAGE_UNREF:
                gdk_image_unref((GdkImage *)fThreadP.pParam);
                break;

            case WIN32_GDK_COLOR_CONTEXT_NEW:
                fThreadP.pRet = gdk_color_context_new(gdk_visual_get_system(), (GdkColormap *)fThreadP.pParam);
                break;
  
            case WIN32_GDK_COLOR_CONTEXT_QUERY_COLORS:
                if(fThreadP.iParam == 1)
                    gdk_color_context_query_color((GdkColorContext *)fThreadP.pParam, &fThreadP.color);
                else
                    gdk_color_context_query_colors((GdkColorContext *)fThreadP.pParam, (GdkColor *)fThreadP.pParam1, fThreadP.iParam);
                break;

            case WIN32_GDK_COLOR_ALLOC:
                fThreadP.iRet = gdk_color_alloc((GdkColormap *)fThreadP.pParam, (GdkColor *)fThreadP.pRet);
                break;

            case WIN32_GDK_COLORMAP_ALLOC_COLOR:
                fThreadP.iRet = gdk_colormap_alloc_color((GdkColormap *)fThreadP.pParam, &fThreadP.color, 
                                    fThreadP.iParam, fThreadP.iParam1);
                break;

            case WIN32_GDK_GC_SET_FONT:
                gdk_gc_set_font(fThreadP.GC, (GdkFont *)fThreadP.pParam);
                break;

            case WIN32_GDK_SET_INPUT:
                if (msg.wParam == 1)
                    EnableWindow((HWND) GDK_DRAWABLE_XID((GdkWindow *) fThreadP.Drawable), TRUE);
                else
                    EnableWindow((HWND) GDK_DRAWABLE_XID((GdkWindow *) fThreadP.Drawable), FALSE);
                break;

            case WIN32_GDK_WARP:
                {
                HWND dw = (HWND) GDK_DRAWABLE_XID((GdkWindow *) fThreadP.Drawable);
                GetCursorPos(&cpt);
                tmp.x = fThreadP.x > 0 ? fThreadP.x : cpt.x;
                tmp.y = fThreadP.y > 0 ? fThreadP.y : cpt.y;
                ClientToScreen(dw, &tmp);
                // SetCursorPos(tmp.x, tmp.y);
                }
                break;



//______________________________________________________________________________
//
//              From GWin32GUI.cxx                
//______________________________________________________________________________


            case WIN32_GDK_WIN_SHOW:
                gdk_window_show((GdkWindow *) fThreadP.Drawable);
                break;

            case WIN32_GDK_MAP_SUBWINDOWS:
                HWND wp;
                EnumChildWindows((HWND) GDK_DRAWABLE_XID((GdkWindow *) fThreadP.Drawable),
                                EnumChildProc, (LPARAM) NULL);
                break;

            case WIN32_GDK_WIN_RAISE:
                gdk_window_raise((GdkWindow *) fThreadP.Drawable);
                break;

            case WIN32_GDK_WIN_HIDE:
                gdk_window_hide((GdkWindow *) fThreadP.Drawable);
                break;

            case WIN32_GDK_WIN_LOWER:
                gdk_window_lower((GdkWindow *) fThreadP.Drawable);
                break;

            case WIN32_GDK_WIN_MOVE_RESIZE:
                gdk_window_move_resize((GdkWindow *) fThreadP.Drawable, fThreadP.x, 
                                fThreadP.y, fThreadP.w, fThreadP.h);
                break;

            case WIN32_GDK_WIN_SET_BACK_PIXMAP:
                gdk_window_set_back_pixmap((GdkWindow *) fThreadP.Drawable, 
                                (GdkPixmap *) fThreadP.pParam, fThreadP.iParam);
                break;

            case WIN32_GDK_WIN_NEW:
                fThreadP.pRet = gdk_window_new((GdkWindow *) fThreadP.Drawable, &fThreadP.xattr, 
                    fThreadP.lParam);
                break;

            case WIN32_GDK_WIN_SET_EVENTS:
                gdk_window_set_events(fThreadP.Drawable, (GdkEventMask) msg.wParam);
                break;

            case WIN32_GDK_WIN_SET_DECOR:
                gdk_window_set_decorations(fThreadP.Drawable, (GdkWMDecoration) msg.wParam);
                break;

            case WIN32_GDK_WIN_GET_COLORMAP:
                fThreadP.pRet = gdk_window_get_colormap((GdkWindow *) fThreadP.Drawable);
                break;

            case WIN32_GDK_WIN_GET_VISUAL:
                fThreadP.pRet = gdk_window_get_visual((GdkWindow *) fThreadP.Drawable);
                break;

            case WIN32_GDK_WIN_IS_VISIBLE:
                fThreadP.iRet = gdk_window_is_visible((GdkWindow *) fThreadP.Drawable);
                break;

            case WIN32_GDK_WIN_IS_VIEWABLE:
                fThreadP.iRet = gdk_window_is_viewable((GdkWindow *) fThreadP.Drawable);
                break;

            case WIN32_GDK_WIN_GET_PARENT:
                fThreadP.pRet = gdk_window_get_parent((GdkWindow *) msg.lParam);//fThreadP.Drawable);
                break;

            case WIN32_GDK_FONT_REF:
                fThreadP.pRet = gdk_font_ref((GdkFont *) fThreadP.pParam);
                break;

            case WIN32_GDK_FONT_UNREF:
                gdk_font_unref((GdkFont *) fThreadP.pParam);
                break;

            case WIN32_GDK_GC_SET_TILE:
                gdk_gc_set_tile((GdkGC *) fThreadP.GC, (GdkDrawable *)fThreadP.pParam);
                break;

            case WIN32_GDK_GC_SET_TS_ORIGIN:
                gdk_gc_set_ts_origin((GdkGC *) fThreadP.GC, fThreadP.x, fThreadP.y);
                break;

            case WIN32_GDK_GC_SET_CLIP_ORIGIN:
                gdk_gc_set_clip_origin((GdkGC *) fThreadP.GC, fThreadP.x, fThreadP.y);
                break;

            case WIN32_GDK_GC_SET_EXPOSURES:
                gdk_gc_set_exposures((GdkGC *) fThreadP.GC, fThreadP.iParam);
                break;

            case WIN32_GDK_GC_SET_SUBWINDOW:
                gdk_gc_set_subwindow((GdkGC *) fThreadP.GC, (GdkSubwindowMode)fThreadP.iParam);
                break;

            case WIN32_GDK_GC_COPY:
                gdk_gc_copy((GdkGC *) fThreadP.GC, (GdkGC *) fThreadP.pParam);
                break;

            case WIN32_GDK_GC_UNREF:
                gdk_gc_unref((GdkGC *) fThreadP.GC);
                break;

            case WIN32_GDK_PIXMAP_CREATE_FROM_DATA:
                fore.pixel = fThreadP.lParam;
                fore.red = GetRValue(fore.pixel);
                fore.green = GetGValue(fore.pixel);
                fore.blue = GetBValue(fore.pixel);

                back.pixel = fThreadP.lParam1;
                back.red = GetRValue(back.pixel);
                back.green = GetGValue(back.pixel);
                back.blue = GetBValue(back.pixel);
                fThreadP.pRet = gdk_pixmap_create_from_data((GdkWindow *) fThreadP.Drawable,
                                                 (char *) fThreadP.pParam, fThreadP.w,
                                                 fThreadP.h, fThreadP.iParam, &fore, &back);
                break;


            case WIN32_GDK_PIXMAP_CREATE_FROM_XPM:
                GdkBitmap *gdk_pixmap_mask;
                fThreadP.pRet = gdk_pixmap_create_from_xpm((GdkWindow *) fThreadP.Drawable,
                                   &gdk_pixmap_mask, (GdkColor *)fThreadP.pParam1, 
                                   fThreadP.sParam);
                fThreadP.pParam = gdk_pixmap_mask;
                break;

            case WIN32_GDK_PIXMAP_CREATE_FROM_XPM_D:
                fThreadP.pRet = gdk_pixmap_create_from_xpm_d((GdkWindow *) fThreadP.Drawable,
                                  (GdkDrawable **)&fThreadP.pParam, 0, (char **)fThreadP.pParam2);
                break;

            case WIN32_GDK_COLOR_PARSE:
                fThreadP.iRet = gdk_color_parse((char *) fThreadP.sParam, &fThreadP.color);
                break;

            case WIN32_GDK_EVENTS_PENDING:
                fThreadP.iRet = gdk_events_pending();
                break;

            case WIN32_GDK_EVENT_GET:
                fThreadP.pRet = gdk_event_get();
                break;

            case WIN32_GDK_EVENT_GET_TIME:
                fThreadP.iRet = gdk_event_get_time((GdkEvent *)fThreadP.pParam);
                break;

            case WIN32_GDK_XID_TABLE_LOOKUP:
                fThreadP.lRet = (ULong_t) gdk_xid_table_lookup((GdkWindow *) fThreadP.Drawable);
                break;

            case WIN32_GDK_BEEP:
                gdk_beep();
                break;

            case WIN32_GDK_WIN_SET_COLORMAP:
                gdk_window_set_colormap((GdkWindow *) fThreadP.Drawable, 
                    (GdkColormap *) fThreadP.pParam);
                break;

            case WIN32_GDK_PROPERTY_CHANGE:
                gdk_property_change((GdkWindow *) fThreadP.Drawable, (GdkAtom) fThreadP.ulParam,
                                   (GdkAtom) fThreadP.ulParam1, fThreadP.iParam, 
                                   (GdkPropMode)fThreadP.iParam1, (unsigned char *) fThreadP.pParam, 
                                   fThreadP.iParam2);
                break;

            case WIN32_WIN32_PROPERTY_CHANGE:
                W32ChangeProperty((HWND) GDK_DRAWABLE_XID((GdkWindow *) fThreadP.Drawable),
                            (Atom) fThreadP.ulParam, (Atom) fThreadP.ulParam1, fThreadP.iParam,
                            fThreadP.iParam1,
                            (unsigned char *) fThreadP.sParam,
                            fThreadP.iParam2);
                break;

            case WIN32_WM_DELETE_NOTIFY:
                Atom prop;
                prop = (Atom) gdk_atom_intern("WM_DELETE_WINDOW", FALSE);

                W32ChangeProperty((HWND) GDK_DRAWABLE_XID((GdkWindow *) fThreadP.Drawable),
                            prop, XA_ATOM, 32, GDK_PROP_MODE_REPLACE,
                            (unsigned char *) &gWM_DELETE_WINDOW, 1);
                break;

            case WIN32_GDK_WIN_CLEAR_AREA:
                gdk_window_clear_area((GdkWindow *) fThreadP.Drawable, fThreadP.x, fThreadP.y,
                                        fThreadP.w, fThreadP.h);
                break;

            case WIN32_GDK_CHECK_TYPED_WIN_EVENT:
                fThreadP.iRet = gdk_check_typed_window_event((GdkWindow *) fThreadP.Drawable, 
//                                        fThreadP.iParam, &fThreadP.event);
                                        fThreadP.iParam, (GdkEvent *) fThreadP.pParam);
                break;

            case WIN32_GDK_EVENT_PUT:

                gdk_event_put((GdkEvent *)fThreadP.pParam);
//                gdk_event_put(&fThreadP.event);
                break;

            case WIN32_GDK_SET_KEY_AUTOREPEAT:
                if (msg.wParam == TRUE)
                    gdk_key_repeat_restore();
                else
                    gdk_key_repeat_disable();
                break;

            case WIN32_GDK_GRAB_KEY:
                if (msg.wParam == TRUE) {
                    masque = gdk_window_get_events((GdkWindow *) fThreadP.Drawable);
                    masque = (GdkEventMask) (masque | (GdkEventMask) fThreadP.uiParam);
                    gdk_window_set_events((GdkWindow *) fThreadP.Drawable, masque);
                    gdk_keyboard_grab((GdkWindow *) fThreadP.Drawable, 1, GDK_CURRENT_TIME);
                } else {
                    masque = gdk_window_get_events((GdkWindow *) fThreadP.Drawable);
                    masque = (GdkEventMask) (masque & (GdkEventMask) fThreadP.uiParam);
                    gdk_window_set_events((GdkWindow *) fThreadP.Drawable, masque);
                    gdk_keyboard_ungrab(GDK_CURRENT_TIME);
                }
                break;

            case WIN32_GDK_GRAB_BUTTON:
                if (msg.wParam == TRUE) {
                    masque = gdk_window_get_events((GdkWindow *) fThreadP.Drawable);
                    masque = (GdkEventMask) (masque | (GdkEventMask) fThreadP.uiParam);
                    gdk_window_set_events((GdkWindow *) fThreadP.Drawable, masque);
                } else {
                    masque = gdk_window_get_events((GdkWindow *) fThreadP.Drawable);
                    masque = (GdkEventMask) (masque & (GdkEventMask) fThreadP.uiParam);
                    gdk_window_set_events((GdkWindow *) fThreadP.Drawable, masque);
                }
                break;

            case WIN32_GDK_GRAB_POINTER:
                if (msg.wParam == TRUE) {
                    gdk_pointer_grab((GdkWindow *) fThreadP.Drawable, fThreadP.iParam,
                               (GdkEventMask) fThreadP.uiParam, (GdkWindow *) fThreadP.pParam1,
                               (GdkCursor *) fThreadP.pParam2, GDK_CURRENT_TIME);
                } else {
                    gdk_pointer_ungrab(GDK_CURRENT_TIME);
                }
                break;


            case WIN32_GDK_WIN_SET_TITLE:
                gdk_window_set_title((GdkWindow *) fThreadP.Drawable, fThreadP.sParam);
                break;

            case WIN32_GDK_WIN_SET_ICON_NAME:
                gdk_window_set_icon_name((GdkWindow *) fThreadP.Drawable, fThreadP.sParam);
                break;

            case WIN32_GDK_WIN_SET_ICON:
                gdk_window_set_icon((GdkWindow *) fThreadP.Drawable, NULL, 
                            (GdkPixmap *) fThreadP.pParam, (GdkPixmap *) fThreadP.pParam);
                break;

            case WIN32_GDK_WIN_SET_FUNCTIONS:
                gdk_window_set_functions((GdkWindow *) fThreadP.Drawable, 
                            (GdkWMFunction) fThreadP.iParam);
                break;

            case WIN32_GDK_WIN_SET_GEOM_HINTS:
                gdk_window_set_geometry_hints((GdkWindow *) fThreadP.Drawable, 
                            (GdkGeometry *)fThreadP.pParam, (GdkWindowHints)fThreadP.iParam);
                break;

            case WIN32_GDK_WIN_SET_TRANSIENT_FOR:
                gdk_window_set_transient_for((GdkWindow *) fThreadP.Drawable, 
                            (GdkWindow *) fThreadP.pParam);
                break;

            case WIN32_GDK_DRAW_TEXT:
                gdk_draw_text((GdkDrawable *) fThreadP.Drawable, (GdkFont *) fThreadP.pParam, 
                            (GdkGC *) fThreadP.GC, fThreadP.x, fThreadP.y, 
                            (const gchar *)fThreadP.sParam, fThreadP.iParam);
                break;

            case WIN32_GDK_KEYVAL_FROM_NAME:
                fThreadP.uiRet = gdk_keyval_from_name((const char *) &fThreadP.uiParam);
                break;

            case WIN32_GDK_SELECT_INPUT:
                {
                GdkEventMask tmp_masque = gdk_window_get_events((GdkWindow *) fThreadP.Drawable);
                masque = (GdkEventMask) (tmp_masque | (GdkEventMask) fThreadP.uiParam);
                gdk_window_set_events((GdkWindow *) fThreadP.Drawable, masque);
                }
                break;

            case WIN32_GDK_GET_INPUT_FOCUS:
                {
                HWND focuswindow = GetFocus();
                fThreadP.pRet = gdk_xid_table_lookup(focuswindow);
                }
                break;

            case WIN32_GDK_SET_INPUT_FOCUS:
                SetFocus((HWND) GDK_DRAWABLE_XID((GdkWindow *) fThreadP.Drawable));
                break;

            case WIN32_GDK_SELECTION_OWNER_GET:
                fThreadP.pRet = gdk_selection_owner_get(gdk_atom_intern ("GDK_SELECTION_PRIMARY", 0));
                break;

            case WIN32_GDK_SELECTION_OWNER_SET:
                gdk_selection_owner_set((GdkWindow *) fThreadP.Drawable, 
                            fThreadP.lParam, GDK_CURRENT_TIME, 0);
                break;

            case WIN32_GDK_SELECTION_CONVERT:
                gdk_selection_convert((GdkWindow *) fThreadP.Drawable, 
                            fThreadP.lParam, gdk_atom_intern("GDK_TARGET_STRING", 0), 
                            fThreadP.uiParam);
                break;

            case WIN32_GDK_SELECTION_PROP_GET:
                fThreadP.iRet = gdk_selection_property_get((GdkWindow *) fThreadP.Drawable,
                                      (unsigned char **) &fThreadP.sRet,
                                      (GdkAtom *) & fThreadP.lParam1, &fThreadP.iRet1);
                break;

            case WIN32_GDK_PROP_DELETE:
                gdk_property_delete((GdkWindow *) fThreadP.Drawable, 
                    gdk_atom_intern("GDK_SELECTION", FALSE));
                break;

            case WIN32_GDK_REGION_NEW:
                fThreadP.pRet = gdk_region_new();
                break;

            case WIN32_GDK_REGION_DESTROY:
                gdk_region_destroy((GdkRegion *) fThreadP.pParam);
                break;

            case WIN32_GDK_REGION_UNION_WITH_RECT:
                {
                GdkRectangle r;
                r.x = fThreadP.x;
                r.y = fThreadP.y;
                r.width = fThreadP.w;
                r.height = fThreadP.h;
                fThreadP.pRet = gdk_region_union_with_rect((GdkRegion *) fThreadP.pParam, &r);
                }
                break;

            case WIN32_GDK_REGION_POLYGON:
                fThreadP.pRet = gdk_region_polygon((GdkPoint *)fThreadP.pParam, 
                    fThreadP.iParam, fThreadP.iParam1 ? GDK_WINDING_RULE : GDK_EVEN_ODD_RULE);
                break;

            case WIN32_GDK_REGIONS_UNION:
                fThreadP.pRet = gdk_regions_union((GdkRegion *) fThreadP.pParam,
                            (GdkRegion *) fThreadP.pParam1);
                break;


            case WIN32_GDK_REGIONS_INTERSECT:
                fThreadP.pRet = gdk_regions_intersect((GdkRegion *) fThreadP.pParam,
                                                        (GdkRegion *) fThreadP.pParam1);
                break;


            case WIN32_GDK_REGIONS_SUBSTRACT:
                fThreadP.pRet = gdk_regions_subtract((GdkRegion *) fThreadP.pParam,
                                                       (GdkRegion *) fThreadP.pParam1);
                break;

            case WIN32_GDK_REGIONS_XOR:
                fThreadP.pRet = gdk_regions_xor((GdkRegion *) fThreadP.pParam, 
                                                        (GdkRegion *) fThreadP.pParam1);
                break;

            case WIN32_GDK_REGION_EMPTY:
                fThreadP.iRet = gdk_region_empty((GdkRegion *) fThreadP.pParam);
                break;

            case WIN32_GDK_REGION_POINT_IN:
                fThreadP.iRet = gdk_region_point_in((GdkRegion *) fThreadP.pParam, 
                                                    fThreadP.x, fThreadP.y);
                break;

            case WIN32_GDK_REGION_EQUAL:
                fThreadP.iRet = gdk_region_equal((GdkRegion *) fThreadP.pParam, 
                                                (GdkRegion *) fThreadP.pParam1);
                break;

            case WIN32_GDK_REGION_GET_CLIPBOX:
                {
                GdkRectangle r;
                gdk_region_get_clipbox((GdkRegion *) fThreadP.pParam, &r);
                fThreadP.x = r.x;
                fThreadP.y = r.y;
                fThreadP.w = r.width;
                fThreadP.h = r.height;
                }
                break;

            case WIN32_GDK_WIN_CHILD_FROM_POINT:
                {
                POINT tpoint;
                tpoint.x = fThreadP.x;
                tpoint.y = fThreadP.y;
                HWND tmpwin = ChildWindowFromPoint((HWND) GDK_DRAWABLE_XID((GdkWindow *)fThreadP.Drawable), tpoint);
                fThreadP.pRet = tmpwin;
                }
                break;

            case WIN32_GDK_TRANSLATE_COORDINATES:
                // TranslateCoordinates translates coordinates from the frame of
                // reference of one window to another. If the point is contained
                // in a mapped child of the destination, the id of that child is
                // returned as well.
                {
                HWND sw, dw, ch = NULL;
                POINT point;
                sw = (HWND) GDK_DRAWABLE_XID((GdkWindow *) fThreadP.Drawable);
                dw = (HWND) GDK_DRAWABLE_XID((GdkWindow *) fThreadP.pParam);
                point.x = fThreadP.x;
                point.y = fThreadP.y;
                MapWindowPoints(sw,          // handle of window to be mapped from
                                dw,          // handle to window to be mapped to
                                &point,      // pointer to array with points to map
                                1);          // number of structures in array
                ch = ChildWindowFromPoint(dw, point);
                fThreadP.pRet = gdk_xid_table_lookup(ch);
                fThreadP.x1 = point.x;
                fThreadP.y1 = point.y;
                }
                break;

            case WIN32_GDK_QUERY_POINTER1:
                {
                POINT mousePt, sPt, currPt;
                HWND chw, window;
                UInt_t ev_mask = 0;

                window = (HWND) GDK_DRAWABLE_XID((GdkWindow *) fThreadP.Drawable);
                fThreadP.pRet = GDK_ROOT_PARENT();
                GetCursorPos(&currPt);
                chw = ChildWindowFromPoint(window, currPt);
                ClientToScreen(window, &mousePt);
                fThreadP.x = mousePt.x;
                fThreadP.y = mousePt.y;
                sPt.x = mousePt.x;
                sPt.y = mousePt.y;
                ScreenToClient(window, &sPt);
                fThreadP.x1 = sPt.x;
                fThreadP.y1 = sPt.y;
                fThreadP.pRet1 = gdk_xid_table_lookup(chw);
                if (fThreadP.pRet1)
                    fThreadP.uiRet = (UInt_t) gdk_window_get_events((GdkWindow *) fThreadP.pRet1);
                }
                break;

            case WIN32_GDK_FONT_FULLNAME_GET:
                fThreadP.sRet = gdk_font_full_name_get((GdkFont *)fThreadP.pParam);
                break;
   
            case WIN32_GDK_GC_SET_TEXT_ALIGN:
                fThreadP.uiRet = gdk_gc_set_text_align((GdkGC *) fThreadP.GC, fThreadP.uiParam);
                break;

            case WIN32_GDK_DRAW_TEXT_WC:
                {
                int i;
                GdkWChar wctext[1024];
                for(i=0;i<fThreadP.iParam;i++)
                    wctext[i] = btowc((int)fThreadP.sParam[i]);
                wctext[fThreadP.iParam] = 0;
                gdk_draw_text_wc((GdkDrawable *) fThreadP.Drawable,
                          (GdkFont *) fThreadP.pParam, (GdkGC *)fThreadP.GC, 
                          fThreadP.x, fThreadP.y,
                          (const GdkWChar *) wctext, fThreadP.iParam);
                }
                break;
   
            case WIN32_GDK_FONT_FULLNAME_FREE:
                gdk_font_full_name_free(fThreadP.sRet);
                break;

            case WIN32_GDK_VISUAL_GET_SYSTEM:
                fThreadP.pRet = gdk_visual_get_system();
                break;

            case WIN32_GDK_QUERY_POINTER:
                {
                GdkWindow *root_return;
                int win_x_return, win_y_return;
                int root_x_return, root_y_return;
                GdkModifierType mask_return;

                root_return = gdk_window_get_pointer((GdkWindow *) fThreadP.Drawable,
                                        &root_x_return, &root_y_return,
                                        &mask_return);

                fThreadP.x = root_x_return;
                fThreadP.y = root_y_return;
                }
                break;

            case WIN32_GDK_EVENT_FREE:
                gdk_event_free((GdkEvent *)fThreadP.pParam);
                break;

            case WIN32_GDK_CURSOR_UNREF:
                gdk_cursor_unref((GdkCursor *)fThreadP.pParam);
                break;

            case WIN32_GW_CHILD:
                fThreadP.pRet = GetWindow((HWND) GDK_DRAWABLE_XID((GdkWindow *) fThreadP.Drawable), GW_CHILD);
                break;

            case WIN32_GDK_ROOT_PARENT:
                fThreadP.pRet = GDK_ROOT_PARENT();
                break;

//______________________________________________________________________________
//
//              For OpenGL                
//______________________________________________________________________________

            case WIN32_GDK_GET_WIN_DC:
                fThreadP.pRet = GetWindowDC((HWND)GDK_DRAWABLE_XID((GdkWindow *)fThreadP.Drawable));
                break;

            case WIN32_GDK_INIT_PIXEL_FORMAT:
                {
               	int pixelformat;
                static PIXELFORMATDESCRIPTOR pfd =
	                {
		                sizeof(PIXELFORMATDESCRIPTOR),  // size of this pfd
		                1,                              // version number
		                PFD_DRAW_TO_WINDOW |            // support window
		                    PFD_SUPPORT_OPENGL |          // support OpenGL
		                    PFD_DOUBLEBUFFER,             // double buffered
		                PFD_TYPE_RGBA,                  // RGBA type
		                24,                             // 24-bit color depth
		                0, 0, 0, 0, 0, 0,               // color bits ignored
		                0,                              // no alpha buffer
		                0,                              // shift bit ignored
		                0,                              // no accumulation buffer
		                0, 0, 0, 0,                     // accum bits ignored
		                32,                             // 32-bit z-buffer
		                0,                              // no stencil buffer
		                0,                              // no auxiliary buffer
		                PFD_MAIN_PLANE,                 // main layer
		                0,                              // reserved
		                0, 0, 0                         // layer masks ignored
	                };

                    fThreadP.iRet = 0;
	                if ( (pixelformat = ChoosePixelFormat(GetWindowDC((HWND)GDK_DRAWABLE_XID((GdkWindow *)fThreadP.Drawable)), 
                        &pfd)) == 0 )
                        fThreadP.iRet = -1;
	                if ( (SetPixelFormat(GetWindowDC((HWND)GDK_DRAWABLE_XID((GdkWindow *)fThreadP.Drawable)), 
                        pixelformat,
                        &pfd)) == FALSE )
                        fThreadP.iRet = -2;
                }
                break;

        }
        ReleaseSemaphore(fThreadP.hThrSem, 1, NULL);
    }    
    if (erret == -1) {
        erret = GetLastError();
        Error("MsgLoop", "Error in GetMessage");
        Printf(" %d \n", erret);
        fIDThread = 0;
        ExitThread(-1);
    }
    else {
        ExitThread(0);
        fIDThread = 0;
    }
}

