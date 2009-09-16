// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*	$NetBSD: key.c,v 1.12 2001/05/17 01:02:17 christos Exp $	*/

/*-
 * Copyright (c) 1992, 1993
 *	The Regents of the University of California.  All rights reserved.
 *
 * This code is derived from software contributed to Berkeley by
 * Christos Zoulas of Cornell University.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#include "compat.h"

/*
 * key.c: This module contains the procedures for maintaining
 *	  the extended-key map.
 *
 *      An extended-key (key) is a sequence of keystrokes introduced
 *	with an sequence introducer and consisting of an arbitrary
 *	number of characters.  This module maintains a map (the el->fKey.fMap)
 *	to convert these extended-key sequences into input strs
 *	(XK_STR), editor functions (XK_CMD), or unix commands (XK_EXE).
 *
 *      Warning:
 *	  If key is a substr of some other keys, then the longer
 *	  keys are lost!!  That is, if the keys "abcd" and "abcef"
 *	  are in el->fKey.fMap, adding the key "abc" will cause the first two
 *	  definitions to be lost.
 *
 *      Restrictions:
 *      -------------
 *      1) It is not possible to have one key that is a
 *	   substr of another.
 */
#include "sys.h"
#include <string.h>
#include <stdlib.h>

#include "el.h"

/*
 * The Nodes of the el->fKey.fMap.  The el->fKey.fMap is a linked list
 * of these node elements
 */
struct KeyNode_t {
   char fCh;                             /* single character of key       */
   int fType;                            /* node type			 */
   KeyValue_t fVal;                     /* command code or pointer to str,  */
                                        /* if this is a leaf             */
   struct KeyNode_t* fNext;             /* ptr to next char of this key  */
   struct KeyNode_t* fSibling;          /* ptr to another key with same prefix*/
};

el_private int node_trav(EditLine_t*, KeyNode_t*, char*,
                         KeyValue_t*);
el_private int node__try(EditLine_t*, KeyNode_t*, const char*,
                         KeyValue_t*, int);
el_private KeyNode_t* node__get(int);
el_private void node__put(EditLine_t*, KeyNode_t*);
el_private int node__delete(EditLine_t*, KeyNode_t**, char*);
el_private int node_lookup(EditLine_t*, const char*, KeyNode_t*, int);
el_private int node_enum(EditLine_t*, KeyNode_t*, int);
el_private int key__decode_char(char*, int, int);

#define KEY_BUFSIZ EL_BUFSIZ


/* key_init():
 *	Initialize the key maps
 */
el_protected int
key_init(EditLine_t* el) {
   el->fKey.fBuf = (char*) el_malloc(KEY_BUFSIZ);

   if (el->fKey.fBuf == NULL) {
      return -1;
   }
   el->fKey.fMap = NULL;
   key_reset(el);
   return 0;
}


/* key_end():
 *	Free the key maps
 */
el_protected void
key_end(EditLine_t* el) {
   el_free((ptr_t) el->fKey.fBuf);
   el->fKey.fBuf = NULL;
   /* XXX: provide a function to clear the keys */
   el->fKey.fMap = NULL;
}


/* key_map_cmd():
 *	Associate cmd with a key value
 */
el_protected KeyValue_t*
key_map_cmd(EditLine_t* el, int cmd) {
   el->fKey.fVal.fCmd = (ElAction_t) cmd;
   return &el->fKey.fVal;
}


/* key_map_str():
 *	Associate str with a key value
 */
el_protected KeyValue_t*
key_map_str(EditLine_t* el, char* str) {
   el->fKey.fVal.fStr = str;
   return &el->fKey.fVal;
}


/* key_reset():
 *	Takes all nodes on el->fKey.fMap and puts them on free list.  Then
 *	initializes el->fKey.fMap with arrow keys
 *	[Always bind the ansi arrow keys?]
 */
el_protected void
key_reset(EditLine_t* el) {
   node__put(el, el->fKey.fMap);
   el->fKey.fMap = NULL;
   return;
}


/* key_get():
 *	Calls the recursive function with entry point el->fKey.fMap
 *      Looks up *ch in map and then reads characters until a
 *      complete match is found or a mismatch occurs. Returns the
 *      type of the match found (XK_STR, XK_CMD, or XK_EXE).
 *      Returns NULL in val.fStr and XK_STR for no match.
 *      The last character read is returned in *ch.
 */
el_protected int
key_get(EditLine_t* el, char* ch, KeyValue_t* val) {
   return node_trav(el, el->fKey.fMap, ch, val);
}


/* key_add():
 *      Adds key to the el->fKey.fMap and associates the value in val with it.
 *      If key is already is in el->fKey.fMap, the new code is applied to the
 *      existing key. Ntype specifies if code is a command, an
 *      out str or a unix command.
 */
el_protected void
key_add(EditLine_t* el, const char* key, KeyValue_t* val, int ntype) {
   if (key[0] == '\0') {
      (void) fprintf(el->fErrFile,
                     "key_add: Null extended-key not allowed.\n");
      return;
   }

   if (ntype == XK_CMD && val->fCmd == ED_SEQUENCE_LEAD_IN) {
      (void) fprintf(el->fErrFile,
                     "key_add: sequence-lead-in command not allowed\n");
      return;
   }

   if (el->fKey.fMap == NULL) {
      /* tree is initially empty.  Set up new node to match key[0] */
      el->fKey.fMap = node__get(key[0]);
   }
   /* it is properly initialized */

   /* Now recurse through el->fKey.fMap */
   (void) node__try(el, el->fKey.fMap, key, val, ntype);
   return;
} // key_add


/* key_clear():
 *
 */
el_protected void
key_clear(EditLine_t* el, ElAction_t* map, char* in) {
   if ((map[(unsigned char) *in] == ED_SEQUENCE_LEAD_IN) &&
       ((map == el->fMap.fKey &&
         el->fMap.fAlt[(unsigned char) *in] != ED_SEQUENCE_LEAD_IN) ||
        (map == el->fMap.fAlt &&
              el->fMap.fKey[(unsigned char) *in] != ED_SEQUENCE_LEAD_IN))) {
      (void) key_delete(el, in);
   }
}


/* key_delete():
 *      Delete the key and all longer keys staring with key, if
 *      they exists.
 */
el_protected int
key_delete(EditLine_t* el, char* key) {
   if (key[0] == '\0') {
      (void) fprintf(el->fErrFile,
                     "key_delete: Null extended-key not allowed.\n");
      return -1;
   }

   if (el->fKey.fMap == NULL) {
      return 0;
   }

   (void) node__delete(el, &el->fKey.fMap, key);
   return 0;
}


/* key_print():
 *	Print the binding associated with key key.
 *	Print entire el->fKey.fMap if null
 */
el_protected void
key_print(EditLine_t* el, const char* key) {
   /* do nothing if el->fKey.fMap is empty and null key specified */
   if (el->fKey.fMap == NULL && *key == 0) {
      return;
   }

   el->fKey.fBuf[0] = '"';

   if (node_lookup(el, key, el->fKey.fMap, 1) <= -1) {
      /* key is not bound */
      (void) fprintf(el->fErrFile, "Unbound extended key \"%s\"\n",
                     key);
   }
   return;
}


/* node_trav():
 *	recursively traverses node in tree until match or mismatch is
 *      found.  May read in more characters.
 */
el_private int
node_trav(EditLine_t* el, KeyNode_t* ptr, char* ch, KeyValue_t* val) {
   if (ptr->fCh == *ch) {
      /* match found */
      if (ptr->fNext) {
         /* key not complete so get next char */
         if (el_getc(el, ch) != 1) {                    /* if EOF or error */
            val->fCmd = ED_END_OF_FILE;
            return XK_CMD;
            /* PWP: Pretend we just read an end-of-file */
         }
         return node_trav(el, ptr->fNext, ch, val);
      } else {
         *val = ptr->fVal;

         if (ptr->fType != XK_CMD) {
            *ch = '\0';
         }
         return ptr->fType;
      }
   } else {
      /* no match found here */
      if (ptr->fSibling) {
         /* try next sibling */
         return node_trav(el, ptr->fSibling, ch, val);
      } else {
         /* no next sibling -- mismatch */
         val->fStr = NULL;
         return XK_STR;
      }
   }
} // node_trav


/* node__try():
 *      Find a node that matches *str or allocate a new one
 */
el_private int
node__try(EditLine_t* el, KeyNode_t* ptr, const char* str, KeyValue_t* val, int ntype) {
   if (ptr->fCh != *str) {
      KeyNode_t* xm;

      for (xm = ptr; xm->fSibling != NULL; xm = xm->fSibling) {
         if (xm->fSibling->fCh == *str) {
            break;
         }
      }

      if (xm->fSibling == NULL) {
         xm->fSibling = node__get(*str);                 /* setup new node */
      }
      ptr = xm->fSibling;
   }

   if (*++str == '\0') {
      /* we're there */
      if (ptr->fNext != NULL) {
         node__put(el, ptr->fNext);
         /* lose longer keys with this prefix */
         ptr->fNext = NULL;
      }

      switch (ptr->fType) {
      case XK_CMD:
      case XK_NOD:
         break;
      case XK_STR:
      case XK_EXE:

         if (ptr->fVal.fStr) {
            el_free((ptr_t) ptr->fVal.fStr);
         }
         break;
      default:
         EL_ABORT((el->fErrFile, "Bad XK_ type %d\n",
                   ptr->fType));
         break;
      }

      switch (ptr->fType = ntype) {
      case XK_CMD:
         ptr->fVal = *val;
         break;
      case XK_STR:
      case XK_EXE:
         ptr->fVal.fStr = strdup(val->fStr);
         break;
      default:
         EL_ABORT((el->fErrFile, "Bad XK_ type %d\n", ntype));
         break;
      }
   } else {
      /* still more chars to go */
      if (ptr->fNext == NULL) {
         ptr->fNext = node__get(*str);                   /* setup new node */
      }
      (void) node__try(el, ptr->fNext, str, val, ntype);
   }
   return 0;
} // node__try


/* node__delete():
 *	Delete node that matches str
 */
el_private int
node__delete(EditLine_t* el, KeyNode_t** inptr, char* str) {
   KeyNode_t* ptr;
   KeyNode_t* prev_ptr = NULL;

   ptr = *inptr;

   if (ptr->fCh != *str) {
      KeyNode_t* xm;

      for (xm = ptr; xm->fSibling != NULL; xm = xm->fSibling) {
         if (xm->fSibling->fCh == *str) {
            break;
         }
      }

      if (xm->fSibling == NULL) {
         return 0;
      }
      prev_ptr = xm;
      ptr = xm->fSibling;
   }

   if (*++str == '\0') {
      /* we're there */
      if (prev_ptr == NULL) {
         *inptr = ptr->fSibling;
      } else {
         prev_ptr->fSibling = ptr->fSibling;
      }
      ptr->fSibling = NULL;
      node__put(el, ptr);
      return 1;
   } else if (ptr->fNext != NULL &&
              node__delete(el, &ptr->fNext, str) == 1) {
      if (ptr->fNext != NULL) {
         return 0;
      }

      if (prev_ptr == NULL) {
         *inptr = ptr->fSibling;
      } else {
         prev_ptr->fSibling = ptr->fSibling;
      }
      ptr->fSibling = NULL;
      node__put(el, ptr);
      return 1;
   } else {
      return 0;
   }
} // node__delete


/* node__put():
 *	Puts a tree of nodes onto free list using free(3).
 */
el_private void
node__put(EditLine_t* el, KeyNode_t* ptr) {
   if (ptr == NULL) {
      return;
   }

   if (ptr->fNext != NULL) {
      node__put(el, ptr->fNext);
      ptr->fNext = NULL;
   }
   node__put(el, ptr->fSibling);

   switch (ptr->fType) {
   case XK_CMD:
   case XK_NOD:
      break;
   case XK_EXE:
   case XK_STR:

      if (ptr->fVal.fStr != NULL) {
         el_free((ptr_t) ptr->fVal.fStr);
      }
      break;
   default:
      EL_ABORT((el->fErrFile, "Bad XK_ type %d\n", ptr->fType));
      break;
   }
   el_free((ptr_t) ptr);
} // node__put


/* node__get():
 *	Returns pointer to an KeyNode_t for ch.
 */
el_private KeyNode_t*
node__get(int ch) {
   KeyNode_t* ptr;

   ptr = (KeyNode_t*) el_malloc((size_t) sizeof(KeyNode_t));

   if (ptr == NULL) {
      return NULL;
   }
   ptr->fCh = ch;
   ptr->fType = XK_NOD;
   ptr->fVal.fStr = NULL;
   ptr->fNext = NULL;
   ptr->fSibling = NULL;
   return ptr;
}


/* node_lookup():
 *	look for the str starting at node ptr.
 *	Print if last node
 */
el_private int
node_lookup(EditLine_t* el, const char* str, KeyNode_t* ptr, int cnt) {
   int ncnt;

   if (ptr == NULL) {
      return -1;                /* cannot have null ptr */

   }

   if (*str == 0) {
      /* no more chars in str.  node_enum from here. */
      (void) node_enum(el, ptr, cnt);
      return 0;
   } else {
      /* If match put this char into el->fKey.fBuf.  Recurse */
      if (ptr->fCh == *str) {
         /* match found */
         ncnt = key__decode_char(el->fKey.fBuf, cnt,
                                 (unsigned char) ptr->fCh);

         if (ptr->fNext != NULL) {
            /* not yet at leaf */
            return node_lookup(el, str + 1, ptr->fNext,
                               ncnt + 1);
         } else {
            /* next node is null so key should be complete */
            if (str[1] == 0) {
               el->fKey.fBuf[ncnt + 1] = '"';
               el->fKey.fBuf[ncnt + 2] = '\0';
               key_kprint(el, el->fKey.fBuf,
                          &ptr->fVal, ptr->fType);
               return 0;
            } else {
               return -1;
            }
            /* mismatch -- str still has chars */
         }
      } else {
         /* no match found try sibling */
         if (ptr->fSibling) {
            return node_lookup(el, str, ptr->fSibling,
                               cnt);
         } else {
            return -1;
         }
      }
   }
} // node_lookup


/* node_enum():
 *	Traverse the node printing the characters it is bound in buffer
 */
el_private int
node_enum(EditLine_t* el, KeyNode_t* ptr, int cnt) {
   int ncnt;

   if (cnt >= KEY_BUFSIZ - 5) {         /* buffer too small */
      el->fKey.fBuf[++cnt] = '"';
      el->fKey.fBuf[++cnt] = '\0';
      (void) fprintf(el->fErrFile,
                     "Some extended keys too long for internal print buffer");
      (void) fprintf(el->fErrFile, " \"%s...\"\n", el->fKey.fBuf);
      return 0;
   }

   if (ptr == NULL) {
#ifdef DEBUG_EDIT
         (void) fprintf(el->fErrFile,
                        "node_enum: BUG!! Null ptr passed\n!");
#endif
      return -1;
   }
   /* put this char at end of str */
   ncnt = key__decode_char(el->fKey.fBuf, cnt, (unsigned char) ptr->fCh);

   if (ptr->fNext == NULL) {
      /* print this key and function */
      el->fKey.fBuf[ncnt + 1] = '"';
      el->fKey.fBuf[ncnt + 2] = '\0';
      key_kprint(el, el->fKey.fBuf, &ptr->fVal, ptr->fType);
   } else {
      (void) node_enum(el, ptr->fNext, ncnt + 1);
   }

   /* go to sibling if there is one */
   if (ptr->fSibling) {
      (void) node_enum(el, ptr->fSibling, cnt);
   }
   return 0;
} // node_enum


/* key_kprint():
 *	Print the specified key and its associated
 *	function specified by val
 */
el_protected void
key_kprint(EditLine_t* el, const char* key, KeyValue_t* val, int ntype) {
   ElBindings_t* fp;
   char unparsbuf[EL_BUFSIZ];
   static const char fmt[] = "%-15s->  %s\n";

   if (val != NULL) {
      switch (ntype) {
      case XK_STR:
      case XK_EXE:
         (void) fprintf(el->fOutFile, fmt, key,
                        key__decode_str(val->fStr, unparsbuf,
                                        ntype == XK_STR ? "\"\"" : "[]"));
         break;
      case XK_CMD:

         for (fp = el->fMap.fHelp; fp->fName; fp++) {
            if (val->fCmd == fp->fFunc) {
               (void) fprintf(el->fOutFile, fmt,
                              key, fp->fName);
               break;
            }
         }
#ifdef DEBUG_KEY

         if (fp->fName == NULL) {
            (void) fprintf(el->fOutFile,
                           "BUG! Command not found.\n");
         }
#endif

         break;
      default:
         EL_ABORT((el->fErrFile, "Bad XK_ type %d\n", ntype));
         break;
      } // switch
   } else {
      (void) fprintf(el->fOutFile, fmt, key, "no input");
   }
} // key_kprint


/* key__decode_char():
 *	Put a printable form of char in buf.
 */
el_private int
key__decode_char(char* buf, int cnt, int ch) {
   if (ch == 0) {
      buf[cnt++] = '^';
      buf[cnt] = '@';
      return cnt;
   }

   if (iscntrl(ch)) {
      buf[cnt++] = '^';

      if (ch == '\177') {
         buf[cnt] = '?';
      } else {
         buf[cnt] = ch | 0100;
      }
   } else if (ch == '^') {
      buf[cnt++] = '\\';
      buf[cnt] = '^';
   } else if (ch == '\\') {
      buf[cnt++] = '\\';
      buf[cnt] = '\\';
   } else if (ch == ' ' || (isprint(ch) && !isspace(ch))) {
      buf[cnt] = ch;
   } else {
      buf[cnt++] = '\\';
      buf[cnt++] = (((unsigned int) ch >> 6) & 7) + '0';
      buf[cnt++] = (((unsigned int) ch >> 3) & 7) + '0';
      buf[cnt] = (ch & 7) + '0';
   }
   return cnt;
} // key__decode_char


/* key__decode_str():
 *	Make a printable version of the ey
 */
el_protected char*
key__decode_str(char* str, char* buf, const char* sep) {
   char* b, * p;

   b = buf;

   if (sep[0] != '\0') {
      *b++ = sep[0];
   }

   if (*str == 0) {
      *b++ = '^';
      *b++ = '@';

      if (sep[0] != '\0' && sep[1] != '\0') {
         *b++ = sep[1];
      }
      *b++ = 0;
      return buf;
   }

   for (p = str; *p != 0; p++) {
      if (iscntrl((unsigned char) *p)) {
         *b++ = '^';

         if (*p == '\177') {
            *b++ = '?';
         } else {
            *b++ = *p | 0100;
         }
      } else if (*p == '^' || *p == '\\') {
         *b++ = '\\';
         *b++ = *p;
      } else if (*p == ' ' || (isprint((unsigned char) *p) &&
                               !isspace((unsigned char) *p))) {
         *b++ = *p;
      } else {
         *b++ = '\\';
         *b++ = (((unsigned int) *p >> 6) & 7) + '0';
         *b++ = (((unsigned int) *p >> 3) & 7) + '0';
         *b++ = (*p & 7) + '0';
      }
   }

   if (sep[0] != '\0' && sep[1] != '\0') {
      *b++ = sep[1];
   }
   *b++ = 0;
   return buf;                  /* should check for overflow */
} // key__decode_str
