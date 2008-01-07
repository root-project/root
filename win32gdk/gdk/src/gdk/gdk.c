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

#include "config.h"

#include <string.h>
#include <stdlib.h>

#include "gdk.h"
#include "gdkprivate.h"

#ifndef HAVE_XCONVERTCASE
#include "gdkkeysyms.h"
#endif

typedef struct _GdkPredicate GdkPredicate;
typedef struct _GdkErrorTrap GdkErrorTrap;

struct _GdkPredicate {
   GdkEventFunc func;
   gpointer data;
};

struct _GdkErrorTrap {
   gint error_warnings;
   gint error_code;
};

/* 
 * Private function declarations
 */
static void gdk_exit_func(void);

GdkFilterReturn gdk_wm_protocols_filter(GdkXEvent * xev,
                                        GdkEvent * event, gpointer data);

/* Private variable declarations
 */
static int gdk_initialized = 0; /* 1 if the library is initialized,
                                 * 0 otherwise.
                                 */

static GSList *gdk_error_traps = NULL;	/* List of error traps */
static GSList *gdk_error_trap_free_list = NULL;	/* Free list */

#ifdef G_ENABLE_DEBUG
static const GDebugKey gdk_debug_keys[] = {
   {"events", GDK_DEBUG_EVENTS},
   {"misc", GDK_DEBUG_MISC},
   {"dnd", GDK_DEBUG_DND},
   {"color-context", GDK_DEBUG_COLOR_CONTEXT},
   {"xim", GDK_DEBUG_XIM}
};

static const int gdk_ndebug_keys =
    sizeof(gdk_debug_keys) / sizeof(GDebugKey);

#endif                          /* G_ENABLE_DEBUG */

GdkArgContext *gdk_arg_context_new(gpointer cb_data)
{
   GdkArgContext *result = g_new(GdkArgContext, 1);
   result->tables = g_ptr_array_new();
   result->cb_data = cb_data;

   return result;
}

void gdk_arg_context_destroy(GdkArgContext * context)
{
   g_ptr_array_free(context->tables, TRUE);
   g_free(context);
}

void gdk_arg_context_add_table(GdkArgContext * context, GdkArgDesc * table)
{
   g_ptr_array_add(context->tables, table);
}

void
gdk_arg_context_parse(GdkArgContext * context, gint * argc, gchar *** argv)
{
   int i, j, k;

   /* Save a copy of the original argc and argv */
   if (argc && argv) {
      for (i = 1; i < *argc; i++) {
         char *arg;

         if (!(*argv)[i][0] == '-' && (*argv)[i][1] == '-')
            continue;

         arg = (*argv)[i] + 2;

         /* '--' terminates list of arguments */
         if (arg == 0) {
            (*argv)[i] = NULL;
            break;
         }

         for (j = 0; j < (int)context->tables->len; j++) {
            GdkArgDesc *table = context->tables->pdata[j];
            for (k = 0; table[k].name; k++) {
               switch (table[k].type) {
               case GDK_ARG_STRING:
               case GDK_ARG_CALLBACK:
               case GDK_ARG_INT:
                  {
                     int len = strlen(table[k].name);

                     if (strncmp(arg, table[k].name, len) == 0 &&
                         (arg[len] == '=' || argc[len] == 0)) {
                        char *value = NULL;

                        (*argv)[i] = NULL;

                        if (arg[len] == '=')
                           value = arg + len + 1;
                        else if (i < *argc - 1) {
                           value = (*argv)[i + 1];
                           (*argv)[i + 1] = NULL;
                           i++;
                        } else
                           value = "";

                        switch (table[k].type) {
                        case GDK_ARG_STRING:
                           *(gchar **) table[k].location = g_strdup(value);
                           break;
                        case GDK_ARG_INT:
                           *(gint *) table[k].location = atoi(value);
                           break;
                        case GDK_ARG_CALLBACK:
                           (*table[k].callback) (table[k].name, value,
                                                 context->cb_data);
                           break;
                        default:
                           ;
                        }

                        goto next_arg;
                     }
                     break;
                  }
               case GDK_ARG_BOOL:
               case GDK_ARG_NOBOOL:
                  if (strcmp(arg, table[k].name) == 0) {
                     (*argv)[i] = NULL;

                     *(gboolean *) table[k].location =
                         (table[k].type == GDK_ARG_BOOL) ? TRUE : FALSE;
                     goto next_arg;
                  }
               }
            }
         }
       next_arg:
         ;
      }

      for (i = 1; i < *argc; i++) {
         for (k = i; k < *argc; k++)
            if ((*argv)[k] != NULL)
               break;

         if (k > i) {
            k -= i;
            for (j = i + k; j < *argc; j++)
               (*argv)[j - k] = (*argv)[j];
            *argc -= k;
         }
      }
   }
}

static void
gdk_arg_debug_cb(const char *key, const char *value, gpointer user_data)
{
   gdk_debug_flags |= g_parse_debug_string(value,
                                           (GDebugKey *) gdk_debug_keys,
                                           gdk_ndebug_keys);
}

static void
gdk_arg_no_debug_cb(const char *key, const char *value, gpointer user_data)
{
   gdk_debug_flags &= ~g_parse_debug_string(value,
                                            (GDebugKey *) gdk_debug_keys,
                                            gdk_ndebug_keys);
}

static void
gdk_arg_name_cb(const char *key, const char *value, gpointer user_data)
{
   g_set_prgname(value);
}

static GdkArgDesc gdk_args[] = {
   {"name", GDK_ARG_STRING, NULL, gdk_arg_name_cb},
#ifdef G_ENABLE_DEBUG
   {"gdk-debug", GDK_ARG_CALLBACK, NULL, gdk_arg_debug_cb},
   {"gdk-no-debug", GDK_ARG_CALLBACK, NULL, gdk_arg_no_debug_cb},
#endif                          /* G_ENABLE_DEBUG */
   {NULL}
};

/*
 *--------------------------------------------------------------
 * gdk_init_check
 *
 *   Initialize the library for use.
 *
 * Arguments:
 *   "argc" is the number of arguments.
 *   "argv" is an array of strings.
 *
 * Results:
 *   "argc" and "argv" are modified to reflect any arguments
 *   which were not handled. (Such arguments should either
 *   be handled by the application or dismissed). If initialization
 *   fails, returns FALSE, otherwise TRUE.
 *
 * Side effects:
 *   The library is initialized.
 *
 *--------------------------------------------------------------
 */

gboolean gdk_init_check(int *argc, char ***argv)
{
   gchar **argv_orig = NULL;
   gint argc_orig = 0;
   GdkArgContext *arg_context;
   gboolean result;
   int i;

   if (gdk_initialized)
      return TRUE;

   if (g_thread_supported())
      gdk_threads_mutex = g_mutex_new();

   if (argc && argv) {
      argc_orig = *argc;

      argv_orig = g_malloc((argc_orig + 1) * sizeof(char *));
      for (i = 0; i < argc_orig; i++)
         argv_orig[i] = g_strdup((*argv)[i]);
      argv_orig[argc_orig] = NULL;

      if (*argc > 0) {
         gchar *d;

         d = strrchr((*argv)[0], '/');
         if (d != NULL)
            g_set_prgname(d + 1);
         else
            g_set_prgname((*argv)[0]);
      }
   }

#ifdef G_ENABLE_DEBUG
   {
      gchar *debug_string = getenv("GDK_DEBUG");
      if (debug_string != NULL)
         gdk_debug_flags = g_parse_debug_string(debug_string,
                                (GDebugKey *) gdk_debug_keys, 
                                gdk_ndebug_keys);
   }
#endif                          /* G_ENABLE_DEBUG */

   arg_context = gdk_arg_context_new(NULL);
   gdk_arg_context_add_table(arg_context, gdk_args);
   gdk_arg_context_add_table(arg_context, _gdk_windowing_args);
   gdk_arg_context_parse(arg_context, argc, argv);
   gdk_arg_context_destroy(arg_context);

   GDK_NOTE(MISC, g_message("progname: \"%s\"", g_get_prgname()));

   result = _gdk_windowing_init_check(argc_orig, argv_orig);

   for (i = 0; i < argc_orig; i++)
      g_free(argv_orig[i]);
   g_free(argv_orig);

   if (!result)
      return FALSE;

   g_atexit(gdk_exit_func);

   gdk_events_init();
   gdk_visual_init();
   gdk_window_init();
   gdk_image_init();
   gdk_input_init();
   gdk_dnd_init();

#ifdef USE_XIM
   gdk_im_open();
#endif

   gdk_initialized = 1;

   return TRUE;
}

void gdk_init(int *argc, char ***argv)
{
   if (!gdk_init_check(argc, argv)) {
      g_warning("cannot open display: %s", gdk_get_display());
      exit(1);
   }
}

/*
 *--------------------------------------------------------------
 * gdk_exit
 *
 *   Restores the library to an un-itialized state and exits
 *   the program using the "exit" system call.
 *
 * Arguments:
 *   "errorcode" is the error value to pass to "exit".
 *
 * Results:
 *   Allocated structures are freed and the program exits
 *   cleanly.
 *
 * Side effects:
 *
 *--------------------------------------------------------------
 */

void gdk_exit(gint errorcode)
{
   /* de-initialisation is done by the gdk_exit_funct(),
      no need to do this here (Alex J.) */
   exit(errorcode);
}

/*
 *--------------------------------------------------------------
 * gdk_exit_func
 *
 *   This is the "atexit" function that makes sure the
 *   library gets a chance to cleanup.
 *
 * Arguments:
 *
 * Results:
 *
 * Side effects:
 *   The library is un-initialized and the program exits.
 *
 *--------------------------------------------------------------
 */

static void gdk_exit_func(void)
{
   static gboolean in_gdk_exit_func = FALSE;

   /* This is to avoid an infinite loop if a program segfaults in
      an atexit() handler (and yes, it does happen, especially if a program
      has trounced over memory too badly for even g_message to work) */
   if (in_gdk_exit_func == TRUE)
      return;
   in_gdk_exit_func = TRUE;

   if (gdk_initialized) {
#ifdef USE_XIM
      /* cleanup IC */
      gdk_ic_cleanup();
      /* close IM */
      gdk_im_close();
#endif
      gdk_image_exit();
      gdk_input_exit();
      gdk_key_repeat_restore();

      gdk_initialized = 0;
   }
}

/*************************************************************
 * gdk_error_trap_push:
 *     Push an error trap. X errors will be trapped until
 *     the corresponding gdk_error_pop(), which will return
 *     the error code, if any.
 *   arguments:
 *     
 *   results:
 *************************************************************/

void gdk_error_trap_push(void)
{
   GSList *node;
   GdkErrorTrap *trap;

   if (gdk_error_trap_free_list) {
      node = gdk_error_trap_free_list;
      gdk_error_trap_free_list = gdk_error_trap_free_list->next;
   } else {
      node = g_slist_alloc();
      node->data = g_new(GdkErrorTrap, 1);
   }

   node->next = gdk_error_traps;
   gdk_error_traps = node;

   trap = node->data;
   trap->error_code = gdk_error_code;
   trap->error_warnings = gdk_error_warnings;

   gdk_error_code = 0;
   gdk_error_warnings = 0;
}

/*************************************************************
 * gdk_error_trap_pop:
 *     Pop an error trap added with gdk_error_push()
 *   arguments:
 *     
 *   results:
 *     0, if no error occured, otherwise the error code.
 *************************************************************/

gint gdk_error_trap_pop(void)
{
   GSList *node;
   GdkErrorTrap *trap;
   gint result;

   g_return_val_if_fail(gdk_error_traps != NULL, 0);

   node = gdk_error_traps;
   gdk_error_traps = gdk_error_traps->next;

   node->next = gdk_error_trap_free_list;
   gdk_error_trap_free_list = node;

   result = gdk_error_code;

   trap = node->data;
   gdk_error_code = trap->error_code;
   gdk_error_warnings = trap->error_warnings;

   return result;
}

#ifndef HAVE_XCONVERTCASE
/* compatibility function from X11R6.3, since XConvertCase is not
 * supplied by X11R5.
 */
void gdk_keyval_convert_case(guint symbol, guint * lower, guint * upper)
{
   guint xlower = symbol;
   guint xupper = symbol;

   switch (symbol >> 8) {
#if	defined (GDK_A) && defined (GDK_Ooblique)
   case 0:                     /* Latin 1 */
      if ((symbol >= GDK_A) && (symbol <= GDK_Z))
         xlower += (GDK_a - GDK_A);
      else if ((symbol >= GDK_a) && (symbol <= GDK_z))
         xupper -= (GDK_a - GDK_A);
      else if ((symbol >= GDK_Agrave) && (symbol <= GDK_Odiaeresis))
         xlower += (GDK_agrave - GDK_Agrave);
      else if ((symbol >= GDK_agrave) && (symbol <= GDK_odiaeresis))
         xupper -= (GDK_agrave - GDK_Agrave);
      else if ((symbol >= GDK_Ooblique) && (symbol <= GDK_Thorn))
         xlower += (GDK_oslash - GDK_Ooblique);
      else if ((symbol >= GDK_oslash) && (symbol <= GDK_thorn))
         xupper -= (GDK_oslash - GDK_Ooblique);
      break;
#endif                          /* LATIN1 */

#if	defined (GDK_Aogonek) && defined (GDK_tcedilla)
   case 1:                     /* Latin 2 */
      /* Assume the KeySym is a legal value (ignore discontinuities) */
      if (symbol == GDK_Aogonek)
         xlower = GDK_aogonek;
      else if (symbol >= GDK_Lstroke && symbol <= GDK_Sacute)
         xlower += (GDK_lstroke - GDK_Lstroke);
      else if (symbol >= GDK_Scaron && symbol <= GDK_Zacute)
         xlower += (GDK_scaron - GDK_Scaron);
      else if (symbol >= GDK_Zcaron && symbol <= GDK_Zabovedot)
         xlower += (GDK_zcaron - GDK_Zcaron);
      else if (symbol == GDK_aogonek)
         xupper = GDK_Aogonek;
      else if (symbol >= GDK_lstroke && symbol <= GDK_sacute)
         xupper -= (GDK_lstroke - GDK_Lstroke);
      else if (symbol >= GDK_scaron && symbol <= GDK_zacute)
         xupper -= (GDK_scaron - GDK_Scaron);
      else if (symbol >= GDK_zcaron && symbol <= GDK_zabovedot)
         xupper -= (GDK_zcaron - GDK_Zcaron);
      else if (symbol >= GDK_Racute && symbol <= GDK_Tcedilla)
         xlower += (GDK_racute - GDK_Racute);
      else if (symbol >= GDK_racute && symbol <= GDK_tcedilla)
         xupper -= (GDK_racute - GDK_Racute);
      break;
#endif                          /* LATIN2 */

#if	defined (GDK_Hstroke) && defined (GDK_Cabovedot)
   case 2:                     /* Latin 3 */
      /* Assume the KeySym is a legal value (ignore discontinuities) */
      if (symbol >= GDK_Hstroke && symbol <= GDK_Hcircumflex)
         xlower += (GDK_hstroke - GDK_Hstroke);
      else if (symbol >= GDK_Gbreve && symbol <= GDK_Jcircumflex)
         xlower += (GDK_gbreve - GDK_Gbreve);
      else if (symbol >= GDK_hstroke && symbol <= GDK_hcircumflex)
         xupper -= (GDK_hstroke - GDK_Hstroke);
      else if (symbol >= GDK_gbreve && symbol <= GDK_jcircumflex)
         xupper -= (GDK_gbreve - GDK_Gbreve);
      else if (symbol >= GDK_Cabovedot && symbol <= GDK_Scircumflex)
         xlower += (GDK_cabovedot - GDK_Cabovedot);
      else if (symbol >= GDK_cabovedot && symbol <= GDK_scircumflex)
         xupper -= (GDK_cabovedot - GDK_Cabovedot);
      break;
#endif                          /* LATIN3 */

#if	defined (GDK_Rcedilla) && defined (GDK_Amacron)
   case 3:                     /* Latin 4 */
      /* Assume the KeySym is a legal value (ignore discontinuities) */
      if (symbol >= GDK_Rcedilla && symbol <= GDK_Tslash)
         xlower += (GDK_rcedilla - GDK_Rcedilla);
      else if (symbol >= GDK_rcedilla && symbol <= GDK_tslash)
         xupper -= (GDK_rcedilla - GDK_Rcedilla);
      else if (symbol == GDK_ENG)
         xlower = GDK_eng;
      else if (symbol == GDK_eng)
         xupper = GDK_ENG;
      else if (symbol >= GDK_Amacron && symbol <= GDK_Umacron)
         xlower += (GDK_amacron - GDK_Amacron);
      else if (symbol >= GDK_amacron && symbol <= GDK_umacron)
         xupper -= (GDK_amacron - GDK_Amacron);
      break;
#endif                          /* LATIN4 */

#if	defined (GDK_Serbian_DJE) && defined (GDK_Cyrillic_yu)
   case 6:                     /* Cyrillic */
      /* Assume the KeySym is a legal value (ignore discontinuities) */
      if (symbol >= GDK_Serbian_DJE && symbol <= GDK_Serbian_DZE)
         xlower -= (GDK_Serbian_DJE - GDK_Serbian_dje);
      else if (symbol >= GDK_Serbian_dje && symbol <= GDK_Serbian_dze)
         xupper += (GDK_Serbian_DJE - GDK_Serbian_dje);
      else if (symbol >= GDK_Cyrillic_YU
               && symbol <= GDK_Cyrillic_HARDSIGN)
         xlower -= (GDK_Cyrillic_YU - GDK_Cyrillic_yu);
      else if (symbol >= GDK_Cyrillic_yu
               && symbol <= GDK_Cyrillic_hardsign)
         xupper += (GDK_Cyrillic_YU - GDK_Cyrillic_yu);
      break;
#endif                          /* CYRILLIC */

#if	defined (GDK_Greek_ALPHAaccent) && defined (GDK_Greek_finalsmallsigma)
   case 7:                     /* Greek */
      /* Assume the KeySym is a legal value (ignore discontinuities) */
      if (symbol >= GDK_Greek_ALPHAaccent
          && symbol <= GDK_Greek_OMEGAaccent)
         xlower += (GDK_Greek_alphaaccent - GDK_Greek_ALPHAaccent);
      else if (symbol >= GDK_Greek_alphaaccent
               && symbol <= GDK_Greek_omegaaccent
               && symbol != GDK_Greek_iotaaccentdieresis
               && symbol != GDK_Greek_upsilonaccentdieresis)
         xupper -= (GDK_Greek_alphaaccent - GDK_Greek_ALPHAaccent);
      else if (symbol >= GDK_Greek_ALPHA && symbol <= GDK_Greek_OMEGA)
         xlower += (GDK_Greek_alpha - GDK_Greek_ALPHA);
      else if (symbol >= GDK_Greek_alpha && symbol <= GDK_Greek_omega &&
               symbol != GDK_Greek_finalsmallsigma)
         xupper -= (GDK_Greek_alpha - GDK_Greek_ALPHA);
      break;
#endif                          /* GREEK */
   }

   if (lower)
      *lower = xlower;
   if (upper)
      *upper = xupper;
}
#endif

guint gdk_keyval_to_upper(guint keyval)
{
   guint result;

   gdk_keyval_convert_case(keyval, NULL, &result);

   return result;
}

guint gdk_keyval_to_lower(guint keyval)
{
   guint result;

   gdk_keyval_convert_case(keyval, &result, NULL);

   return result;
}

gboolean gdk_keyval_is_upper(guint keyval)
{
   if (keyval) {
      guint upper_val = 0;

      gdk_keyval_convert_case(keyval, NULL, &upper_val);
      return upper_val == keyval;
   }
   return FALSE;
}

gboolean gdk_keyval_is_lower(guint keyval)
{
   if (keyval) {
      guint lower_val = 0;

      gdk_keyval_convert_case(keyval, &lower_val, NULL);
      return lower_val == keyval;
   }
   return FALSE;
}

void gdk_threads_enter()
{
   GDK_THREADS_ENTER();
}

void gdk_threads_leave()
{
   GDK_THREADS_LEAVE();
}
