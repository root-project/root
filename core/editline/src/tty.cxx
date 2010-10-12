// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*	$NetBSD: tty.c,v 1.15 2001/05/17 01:02:17 christos Exp $	*/

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
 * tty.c: tty interface stuff
 */
#include "sys.h"
#include "tty.h"
#include "el.h"
#include "errno.h"
#include "stdio.h"


struct TTYModes_t {
   const char* fName;
   u_int fValue;
   int fType;
};

struct TTYMap_t {
   int fNCh;                /* Internal rep of chars */
   int fOCh;                /* Termio rep of chars */
   ElAction_t fBind[3];     /* emacs, vi, and vi-cmd */
};


el_private const TTYPerm_t ttyperm = {
   {
      { "iflag:", ICRNL, (INLCR | IGNCR) },
      { "oflag:", (OPOST | ONLCR), ONLRET },
      { "cflag:", 0, 0 },
      { "lflag:", (ISIG | ICANON | ECHO | ECHOE | ECHOCTL | IEXTEN),
        (NOFLSH | ECHONL | EXTPROC | FLUSHO) },
      { "chars:", 0, 0 },
   },
   {
      { "iflag:", (INLCR | ICRNL), IGNCR },
      { "oflag:", (OPOST | ONLCR), ONLRET },
      { "cflag:", 0, 0 },
      { "lflag:", ISIG,
        (NOFLSH | ICANON | ECHO | ECHOK | ECHONL | EXTPROC | IEXTEN | FLUSHO) },
      { "chars:", (C_SH(C_MIN) | C_SH(C_TIME) | C_SH(C_SWTCH) | C_SH(C_DSWTCH) |
                   C_SH(C_SUSP) | C_SH(C_DSUSP) | C_SH(C_EOL) | C_SH(C_DISCARD) |
                   C_SH(C_PGOFF) | C_SH(C_PAGE) | C_SH(C_STATUS)), 0 }
   },
   {
      { "iflag:", 0, IXON | IXOFF | INLCR | ICRNL },
      { "oflag:", 0, 0 },
      { "cflag:", 0, 0 },
      { "lflag:", 0, ISIG | IEXTEN },
      { "chars:", 0, 0 },
   }
};

el_private const TTYChar_t ttychar = {
   {
      CINTR, CQUIT, CERASE, CKILL,
      CEOF, CEOL, CEOL2, CSWTCH,
      CDSWTCH, CERASE2, CSTART, CSTOP,
      CWERASE, CSUSP, CDSUSP, CREPRINT,
      CDISCARD, CLNEXT, CSTATUS, CPAGE,
      CPGOFF, CKILL2, CBRK, CMIN,
      CTIME
   },
   {
      CINTR, CQUIT, CERASE, CKILL,
      _POSIX_VDISABLE, _POSIX_VDISABLE, _POSIX_VDISABLE, _POSIX_VDISABLE,
      _POSIX_VDISABLE, CERASE2, CSTART, CSTOP,
      _POSIX_VDISABLE, CSUSP, _POSIX_VDISABLE, _POSIX_VDISABLE,
      CDISCARD, _POSIX_VDISABLE, _POSIX_VDISABLE, _POSIX_VDISABLE,
      _POSIX_VDISABLE, _POSIX_VDISABLE, _POSIX_VDISABLE, 1,
      0
   },
   {
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0
   }
};

el_private const TTYMap_t tty_map[] = {
#ifdef VERASE
   { C_ERASE, VERASE,
     { ED_DELETE_PREV_CHAR, VI_DELETE_PREV_CHAR, ED_PREV_CHAR } },
#endif /* VERASE */
#ifdef VERASE2
   { C_ERASE2, VERASE2,
     { ED_DELETE_PREV_CHAR, VI_DELETE_PREV_CHAR, ED_PREV_CHAR } },
#endif /* VERASE2 */
#ifdef VKILL
   { C_KILL, VKILL,
     { EM_KILL_LINE, VI_KILL_LINE_PREV, ED_UNASSIGNED } },
#endif /* VKILL */
#ifdef VKILL2
   { C_KILL2, VKILL2,
     { EM_KILL_LINE, VI_KILL_LINE_PREV, ED_UNASSIGNED } },
#endif /* VKILL2 */
#ifdef VEOF
   { C_EOF, VEOF,
     { EM_DELETE_OR_LIST, VI_LIST_OR_EOF, ED_UNASSIGNED } },
#endif /* VEOF */
#ifdef VWERASE
   { C_WERASE, VWERASE,
     { ED_DELETE_PREV_WORD, ED_DELETE_PREV_WORD, ED_PREV_WORD } },
#endif /* VWERASE */
#ifdef VREPRINT
   { C_REPRINT, VREPRINT,
     { ED_REDISPLAY, ED_INSERT, ED_REDISPLAY } },
#endif /* VREPRINT */
#ifdef VLNEXT
   { C_LNEXT, VLNEXT,
     { ED_QUOTED_INSERT, ED_QUOTED_INSERT, ED_UNASSIGNED } },
#endif /* VLNEXT */
   { -1, -1,
     { ED_UNASSIGNED, ED_UNASSIGNED, ED_UNASSIGNED } }
};

el_private const TTYModes_t ttymodes[] = {
#ifdef  IGNBRK
   { "ignbrk", IGNBRK, MD_INP },
#endif /* IGNBRK */
#ifdef  BRKINT
   { "brkint", BRKINT, MD_INP },
#endif /* BRKINT */
#ifdef  IGNPAR
   { "ignpar", IGNPAR, MD_INP },
#endif /* IGNPAR */
#ifdef  PARMRK
   { "parmrk", PARMRK, MD_INP },
#endif /* PARMRK */
#ifdef  INPCK
   { "inpck", INPCK, MD_INP },
#endif /* INPCK */
#ifdef  ISTRIP
   { "istrip", ISTRIP, MD_INP },
#endif /* ISTRIP */
#ifdef  INLCR
   { "inlcr", INLCR, MD_INP },
#endif /* INLCR */
#ifdef  IGNCR
   { "igncr", IGNCR, MD_INP },
#endif /* IGNCR */
#ifdef  ICRNL
   { "icrnl", ICRNL, MD_INP },
#endif /* ICRNL */
#ifdef  IUCLC
   { "iuclc", IUCLC, MD_INP },
#endif /* IUCLC */
#ifdef  IXON
   { "ixon", IXON, MD_INP },
#endif /* IXON */
#ifdef  IXANY
   { "ixany", IXANY, MD_INP },
#endif /* IXANY */
#ifdef  IXOFF
   { "ixoff", IXOFF, MD_INP },
#endif /* IXOFF */
#ifdef  IMAXBEL
   { "imaxbel", IMAXBEL, MD_INP },
#endif /* IMAXBEL */

#ifdef  OPOST
   { "opost", OPOST, MD_OUT },
#endif /* OPOST */
#ifdef  OLCUC
   { "olcuc", OLCUC, MD_OUT },
#endif /* OLCUC */
#ifdef  ONLCR
   { "onlcr", ONLCR, MD_OUT },
#endif /* ONLCR */
#ifdef  OCRNL
   { "ocrnl", OCRNL, MD_OUT },
#endif /* OCRNL */
#ifdef  ONOCR
   { "onocr", ONOCR, MD_OUT },
#endif /* ONOCR */
#ifdef ONOEOT
   { "onoeot", ONOEOT, MD_OUT },
#endif /* ONOEOT */
#ifdef  ONLRET
   { "onlret", ONLRET, MD_OUT },
#endif /* ONLRET */
#ifdef  OFILL
   { "ofill", OFILL, MD_OUT },
#endif /* OFILL */
#ifdef  OFDEL
   { "ofdel", OFDEL, MD_OUT },
#endif /* OFDEL */
#ifdef  NLDLY
   { "nldly", NLDLY, MD_OUT },
#endif /* NLDLY */
#ifdef  CRDLY
   { "crdly", CRDLY, MD_OUT },
#endif /* CRDLY */
#ifdef  TABDLY
   { "tabdly", TABDLY, MD_OUT },
#endif /* TABDLY */
#ifdef  XTABS
   { "xtabs", XTABS, MD_OUT },
#endif /* XTABS */
#ifdef  BSDLY
   { "bsdly", BSDLY, MD_OUT },
#endif /* BSDLY */
#ifdef  VTDLY
   { "vtdly", VTDLY, MD_OUT },
#endif /* VTDLY */
#ifdef  FFDLY
   { "ffdly", FFDLY, MD_OUT },
#endif /* FFDLY */
#ifdef  PAGEOUT
   { "pageout", PAGEOUT, MD_OUT },
#endif /* PAGEOUT */
#ifdef  WRAP
   { "wrap", WRAP, MD_OUT },
#endif /* WRAP */

#ifdef  CIGNORE
   { "cignore", CIGNORE, MD_CTL },
#endif /* CBAUD */
#ifdef  CBAUD
   { "cbaud", CBAUD, MD_CTL },
#endif /* CBAUD */
#ifdef  CSTOPB
   { "cstopb", CSTOPB, MD_CTL },
#endif /* CSTOPB */
#ifdef  CREAD
   { "cread", CREAD, MD_CTL },
#endif /* CREAD */
#ifdef  PARENB
   { "parenb", PARENB, MD_CTL },
#endif /* PARENB */
#ifdef  PARODD
   { "parodd", PARODD, MD_CTL },
#endif /* PARODD */
#ifdef  HUPCL
   { "hupcl", HUPCL, MD_CTL },
#endif /* HUPCL */
#ifdef  CLOCAL
   { "clocal", CLOCAL, MD_CTL },
#endif /* CLOCAL */
#ifdef  LOBLK
   { "loblk", LOBLK, MD_CTL },
#endif /* LOBLK */
#ifdef  CIBAUD
   { "cibaud", CIBAUD, MD_CTL },
#endif /* CIBAUD */
#ifdef CRTSCTS
# ifdef CCTS_OFLOW
   { "ccts_oflow", CCTS_OFLOW, MD_CTL },
# else
   { "crtscts", CRTSCTS, MD_CTL },
# endif /* CCTS_OFLOW */
#endif /* CRTSCTS */
#ifdef CRTS_IFLOW
   { "crts_iflow", CRTS_IFLOW, MD_CTL },
#endif /* CRTS_IFLOW */
#ifdef CDTRCTS
   { "cdtrcts", CDTRCTS, MD_CTL },
#endif /* CDTRCTS */
#ifdef MDMBUF
   { "mdmbuf", MDMBUF, MD_CTL },
#endif /* MDMBUF */
#ifdef RCV1EN
   { "rcv1en", RCV1EN, MD_CTL },
#endif /* RCV1EN */
#ifdef XMT1EN
   { "xmt1en", XMT1EN, MD_CTL },
#endif /* XMT1EN */

#ifdef  ISIG
   { "isig", ISIG, MD_LIN },
#endif /* ISIG */
#ifdef  ICANON
   { "icanon", ICANON, MD_LIN },
#endif /* ICANON */
#ifdef  XCASE
   { "xcase", XCASE, MD_LIN },
#endif /* XCASE */
#ifdef  ECHO
   { "echo", ECHO, MD_LIN },
#endif /* ECHO */
#ifdef  ECHOE
   { "echoe", ECHOE, MD_LIN },
#endif /* ECHOE */
#ifdef  ECHOK
   { "echok", ECHOK, MD_LIN },
#endif /* ECHOK */
#ifdef  ECHONL
   { "echonl", ECHONL, MD_LIN },
#endif /* ECHONL */
#ifdef  NOFLSH
   { "noflsh", NOFLSH, MD_LIN },
#endif /* NOFLSH */
#ifdef  TOSTOP
   { "tostop", TOSTOP, MD_LIN },
#endif /* TOSTOP */
#ifdef  ECHOCTL
   { "echoctl", ECHOCTL, MD_LIN },
#endif /* ECHOCTL */
#ifdef  ECHOPRT
   { "echoprt", ECHOPRT, MD_LIN },
#endif /* ECHOPRT */
#ifdef  ECHOKE
   { "echoke", ECHOKE, MD_LIN },
#endif /* ECHOKE */
#ifdef  DEFECHO
   { "defecho", DEFECHO, MD_LIN },
#endif /* DEFECHO */
#ifdef  FLUSHO
   { "flusho", FLUSHO, MD_LIN },
#endif /* FLUSHO */
#ifdef  PENDIN
   { "pendin", PENDIN, MD_LIN },
#endif /* PENDIN */
#ifdef  IEXTEN
   { "iexten", IEXTEN, MD_LIN },
#endif /* IEXTEN */
#ifdef  NOKERNINFO
   { "nokerninfo", NOKERNINFO, MD_LIN },
#endif /* NOKERNINFO */
#ifdef  ALTWERASE
   { "altwerase", ALTWERASE, MD_LIN },
#endif /* ALTWERASE */
#ifdef  EXTPROC
   { "extproc", EXTPROC, MD_LIN },
#endif /* EXTPROC */

#if defined(VINTR)
   { "intr", C_SH(C_INTR), MD_CHAR },
#endif /* VINTR */
#if defined(VQUIT)
   { "quit", C_SH(C_QUIT), MD_CHAR },
#endif /* VQUIT */
#if defined(VERASE)
   { "erase", C_SH(C_ERASE), MD_CHAR },
#endif /* VERASE */
#if defined(VKILL)
   { "kill", C_SH(C_KILL), MD_CHAR },
#endif /* VKILL */
#if defined(VEOF)
   { "eof", C_SH(C_EOF), MD_CHAR },
#endif /* VEOF */
#if defined(VEOL)
   { "eol", C_SH(C_EOL), MD_CHAR },
#endif /* VEOL */
#if defined(VEOL2)
   { "eol2", C_SH(C_EOL2), MD_CHAR },
#endif /* VEOL2 */
#if defined(VSWTCH)
   { "swtch", C_SH(C_SWTCH), MD_CHAR },
#endif /* VSWTCH */
#if defined(VDSWTCH)
   { "dswtch", C_SH(C_DSWTCH), MD_CHAR },
#endif /* VDSWTCH */
#if defined(VERASE2)
   { "erase2", C_SH(C_ERASE2), MD_CHAR },
#endif /* VERASE2 */
#if defined(VSTART)
   { "start", C_SH(C_START), MD_CHAR },
#endif /* VSTART */
#if defined(VSTOP)
   { "stop", C_SH(C_STOP), MD_CHAR },
#endif /* VSTOP */
#if defined(VWERASE)
   { "werase", C_SH(C_WERASE), MD_CHAR },
#endif /* VWERASE */
#if defined(VSUSP)
   { "susp", C_SH(C_SUSP), MD_CHAR },
#endif /* VSUSP */
#if defined(VDSUSP)
   { "dsusp", C_SH(C_DSUSP), MD_CHAR },
#endif /* VDSUSP */
#if defined(VREPRINT)
   { "reprint", C_SH(C_REPRINT), MD_CHAR },
#endif /* VREPRINT */
#if defined(VDISCARD)
   { "discard", C_SH(C_DISCARD), MD_CHAR },
#endif /* VDISCARD */
#if defined(VLNEXT)
   { "lnext", C_SH(C_LNEXT), MD_CHAR },
#endif /* VLNEXT */
#if defined(VSTATUS)
   { "status", C_SH(C_STATUS), MD_CHAR },
#endif /* VSTATUS */
#if defined(VPAGE)
   { "page", C_SH(C_PAGE), MD_CHAR },
#endif /* VPAGE */
#if defined(VPGOFF)
   { "pgoff", C_SH(C_PGOFF), MD_CHAR },
#endif /* VPGOFF */
#if defined(VKILL2)
   { "kill2", C_SH(C_KILL2), MD_CHAR },
#endif /* VKILL2 */
#if defined(VBRK)
   { "brk", C_SH(C_BRK), MD_CHAR },
#endif /* VBRK */
#if defined(VMIN)
   { "min", C_SH(C_MIN), MD_CHAR },
#endif /* VMIN */
#if defined(VTIME)
   { "time", C_SH(C_TIME), MD_CHAR },
#endif /* VTIME */
   { NULL, 0, -1 },
};


#define tty_getty(el, td) tcgetattr((el)->fInFD, (td))
#define tty_setty(el, td) tcsetattr((el)->fInFD, TCSADRAIN, (td))

#define tty__gettabs(td) ((((td)->c_oflag & TAB3) == TAB3) ? 0 : 1)
#define tty__geteightbit(td) (((td)->c_cflag & CSIZE) == CS8)
#define tty__cooked_mode(td) ((td)->c_lflag & ICANON)

el_private void tty__getchar(struct termios*, unsigned char*);
el_private void tty__setchar(struct termios*, unsigned char*);
el_private speed_t tty__getspeed(struct termios*);
el_private int tty_setup(EditLine_t*);

#define t_qu t_ts

/* tty_canoutput():
 *   Indicate whether we are connected or not to the tty.
 *   In particular returns false if the process is in the background.
 */
int
tty_can_output(void)
{
   return (getpgrp() == tcgetpgrp(STDOUT_FILENO));
}

bool tty_need_to_run_setup = false;

/* tty_setup():
 *	Get the tty parameters and initialize the editing state
 */
el_private int
tty_setup(EditLine_t* el) {
   int rst = 1;
   if (!tty_can_output()) {
      tty_need_to_run_setup = true;
      return 0;
   }
   tty_need_to_run_setup = false;

   /*
   if (el->fFlags & EDIT_DISABLED) {
      return 0;
   }
   */

   if (tty_getty(el, &el->fTTY.t_ed) == -1) {
#ifdef DEBUG_TTY
         (void) fprintf(el->fErrFile,
                        "tty_setup: tty_getty: %s\n", strerror(errno));
#endif /* DEBUG_TTY */
      return -1;
   }
   el->fTTY.t_ts = el->fTTY.t_ex = el->fTTY.t_ed;

   el->fTTY.t_speed = tty__getspeed(&el->fTTY.t_ex);
   el->fTTY.t_tabs = tty__gettabs(&el->fTTY.t_ex);
   el->fTTY.t_eight = tty__geteightbit(&el->fTTY.t_ex);

   el->fTTY.t_ex.c_iflag &= ~el->fTTY.t_t[EX_IO][MD_INP].t_clrmask;
   el->fTTY.t_ex.c_iflag |= el->fTTY.t_t[EX_IO][MD_INP].t_setmask;

   el->fTTY.t_ex.c_oflag &= ~el->fTTY.t_t[EX_IO][MD_OUT].t_clrmask;
   el->fTTY.t_ex.c_oflag |= el->fTTY.t_t[EX_IO][MD_OUT].t_setmask;

   el->fTTY.t_ex.c_cflag &= ~el->fTTY.t_t[EX_IO][MD_CTL].t_clrmask;
   el->fTTY.t_ex.c_cflag |= el->fTTY.t_t[EX_IO][MD_CTL].t_setmask;

   el->fTTY.t_ex.c_lflag &= ~el->fTTY.t_t[EX_IO][MD_LIN].t_clrmask;
   el->fTTY.t_ex.c_lflag |= el->fTTY.t_t[EX_IO][MD_LIN].t_setmask;

   /*
    * Reset the tty chars to reasonable defaults
    * If they are disabled, then enable them.
    */
   if (rst) {
      if (tty__cooked_mode(&el->fTTY.t_ts)) {
         tty__getchar(&el->fTTY.t_ts, el->fTTY.t_c[TS_IO]);

         /*
          * Don't affect CMIN and CTIME for the editor mode
          */
         for (rst = 0; rst < C_NCC - 2; rst++) {
            if (el->fTTY.t_c[TS_IO][rst] !=
                el->fTTY.t_vdisable
                && el->fTTY.t_c[ED_IO][rst] !=
                el->fTTY.t_vdisable) {
               el->fTTY.t_c[ED_IO][rst] =
                  el->fTTY.t_c[TS_IO][rst];
            }
         }

         for (rst = 0; rst < C_NCC; rst++) {
            if (el->fTTY.t_c[TS_IO][rst] !=
                el->fTTY.t_vdisable) {
               el->fTTY.t_c[EX_IO][rst] =
                  el->fTTY.t_c[TS_IO][rst];
            }
         }
      }
      tty__setchar(&el->fTTY.t_ex, el->fTTY.t_c[EX_IO]);

      if (tty_setty(el, &el->fTTY.t_ex) == -1) {
#ifdef DEBUG_TTY
            (void) fprintf(el->fErrFile,
                           "tty_setup: tty_setty: %s\n",
                           strerror(errno));
#endif /* DEBUG_TTY */
         return -1;
      }
   } else {
      // This cannot be reached as rst is set to 1 above.
      // coverity[dead_error_line]
      tty__setchar(&el->fTTY.t_ex, el->fTTY.t_c[EX_IO]);
   }

   el->fTTY.t_ed.c_iflag &= ~el->fTTY.t_t[ED_IO][MD_INP].t_clrmask;
   el->fTTY.t_ed.c_iflag |= el->fTTY.t_t[ED_IO][MD_INP].t_setmask;

   el->fTTY.t_ed.c_oflag &= ~el->fTTY.t_t[ED_IO][MD_OUT].t_clrmask;
   el->fTTY.t_ed.c_oflag |= el->fTTY.t_t[ED_IO][MD_OUT].t_setmask;

   el->fTTY.t_ed.c_cflag &= ~el->fTTY.t_t[ED_IO][MD_CTL].t_clrmask;
   el->fTTY.t_ed.c_cflag |= el->fTTY.t_t[ED_IO][MD_CTL].t_setmask;

   el->fTTY.t_ed.c_lflag &= ~el->fTTY.t_t[ED_IO][MD_LIN].t_clrmask;
   el->fTTY.t_ed.c_lflag |= el->fTTY.t_t[ED_IO][MD_LIN].t_setmask;

   tty__setchar(&el->fTTY.t_ed, el->fTTY.t_c[ED_IO]);
   tty_bind_char(el, 1);

   el_set(el, EL_EDITMODE, 1);
   return 0;
} // tty_setup


el_protected int
tty_init(EditLine_t* el) {
   el->fTTY.t_mode = EX_IO;
   el->fTTY.t_vdisable = _POSIX_VDISABLE;
   (void) memcpy(el->fTTY.t_t, ttyperm, sizeof(TTYPerm_t));
   (void) memcpy(el->fTTY.t_c, ttychar, sizeof(TTYChar_t));
   return tty_setup(el);
}


/* tty_end():
 *	Restore the tty to its original settings
 */
el_protected void
/*ARGSUSED*/
tty_end(EditLine_t* /*el*/) {
   /* XXX: Maybe reset to an initial state? */
}


/* tty__getspeed():
 *	Get the tty speed
 */
el_private speed_t
tty__getspeed(struct termios* td) {
   speed_t spd;

   if ((spd = cfgetispeed(td)) == 0) {
      spd = cfgetospeed(td);
   }
   return spd;
}


/* tty__getchar():
 *	Get the tty characters
 */
el_private void
tty__getchar(struct termios* td, unsigned char* s) {
#ifdef VINTR
   s[C_INTR] = td->c_cc[VINTR];
#endif /* VINTR */
#ifdef VQUIT
   s[C_QUIT] = td->c_cc[VQUIT];
#endif /* VQUIT */
#ifdef VERASE
   s[C_ERASE] = td->c_cc[VERASE];
#endif /* VERASE */
#ifdef VKILL
   s[C_KILL] = td->c_cc[VKILL];
#endif /* VKILL */
#ifdef VEOF
   s[C_EOF] = td->c_cc[VEOF];
#endif /* VEOF */
#ifdef VEOL
   s[C_EOL] = td->c_cc[VEOL];
#endif /* VEOL */
#ifdef VEOL2
   s[C_EOL2] = td->c_cc[VEOL2];
#endif /* VEOL2 */
#ifdef VSWTCH
   s[C_SWTCH] = td->c_cc[VSWTCH];
#endif /* VSWTCH */
#ifdef VDSWTCH
   s[C_DSWTCH] = td->c_cc[VDSWTCH];
#endif /* VDSWTCH */
#ifdef VERASE2
   s[C_ERASE2] = td->c_cc[VERASE2];
#endif /* VERASE2 */
#ifdef VSTART
   s[C_START] = td->c_cc[VSTART];
#endif /* VSTART */
#ifdef VSTOP
   s[C_STOP] = td->c_cc[VSTOP];
#endif /* VSTOP */
#ifdef VWERASE
   s[C_WERASE] = td->c_cc[VWERASE];
#endif /* VWERASE */
#ifdef VSUSP
   s[C_SUSP] = td->c_cc[VSUSP];
#endif /* VSUSP */
#ifdef VDSUSP
   s[C_DSUSP] = td->c_cc[VDSUSP];
#endif /* VDSUSP */
#ifdef VREPRINT
   s[C_REPRINT] = td->c_cc[VREPRINT];
#endif /* VREPRINT */
#ifdef VDISCARD
   s[C_DISCARD] = td->c_cc[VDISCARD];
#endif /* VDISCARD */
#ifdef VLNEXT
   s[C_LNEXT] = td->c_cc[VLNEXT];
#endif /* VLNEXT */
#ifdef VSTATUS
   s[C_STATUS] = td->c_cc[VSTATUS];
#endif /* VSTATUS */
#ifdef VPAGE
   s[C_PAGE] = td->c_cc[VPAGE];
#endif /* VPAGE */
#ifdef VPGOFF
   s[C_PGOFF] = td->c_cc[VPGOFF];
#endif /* VPGOFF */
#ifdef VKILL2
   s[C_KILL2] = td->c_cc[VKILL2];
#endif /* KILL2 */
#ifdef VMIN
   s[C_MIN] = td->c_cc[VMIN];
#endif /* VMIN */
#ifdef VTIME
   s[C_TIME] = td->c_cc[VTIME];
#endif /* VTIME */
}                               /* tty__getchar */


/* tty__setchar():
 *	Set the tty characters
 */
el_private void
tty__setchar(struct termios* td, unsigned char* s) {
#ifdef VINTR
   td->c_cc[VINTR] = s[C_INTR];
#endif /* VINTR */
#ifdef VQUIT
   td->c_cc[VQUIT] = s[C_QUIT];
#endif /* VQUIT */
#ifdef VERASE
   td->c_cc[VERASE] = s[C_ERASE];
#endif /* VERASE */
#ifdef VKILL
   td->c_cc[VKILL] = s[C_KILL];
#endif /* VKILL */
#ifdef VEOF
   td->c_cc[VEOF] = s[C_EOF];
#endif /* VEOF */
#ifdef VEOL
   td->c_cc[VEOL] = s[C_EOL];
#endif /* VEOL */
#ifdef VEOL2
   td->c_cc[VEOL2] = s[C_EOL2];
#endif /* VEOL2 */
#ifdef VSWTCH
   td->c_cc[VSWTCH] = s[C_SWTCH];
#endif /* VSWTCH */
#ifdef VDSWTCH
   td->c_cc[VDSWTCH] = s[C_DSWTCH];
#endif /* VDSWTCH */
#ifdef VERASE2
   td->c_cc[VERASE2] = s[C_ERASE2];
#endif /* VERASE2 */
#ifdef VSTART
   td->c_cc[VSTART] = s[C_START];
#endif /* VSTART */
#ifdef VSTOP
   td->c_cc[VSTOP] = s[C_STOP];
#endif /* VSTOP */
#ifdef VWERASE
   td->c_cc[VWERASE] = s[C_WERASE];
#endif /* VWERASE */
#ifdef VSUSP
   td->c_cc[VSUSP] = s[C_SUSP];
#endif /* VSUSP */
#ifdef VDSUSP
   td->c_cc[VDSUSP] = s[C_DSUSP];
#endif /* VDSUSP */
#ifdef VREPRINT
   td->c_cc[VREPRINT] = s[C_REPRINT];
#endif /* VREPRINT */
#ifdef VDISCARD
   td->c_cc[VDISCARD] = s[C_DISCARD];
#endif /* VDISCARD */
#ifdef VLNEXT
   td->c_cc[VLNEXT] = s[C_LNEXT];
#endif /* VLNEXT */
#ifdef VSTATUS
   td->c_cc[VSTATUS] = s[C_STATUS];
#endif /* VSTATUS */
#ifdef VPAGE
   td->c_cc[VPAGE] = s[C_PAGE];
#endif /* VPAGE */
#ifdef VPGOFF
   td->c_cc[VPGOFF] = s[C_PGOFF];
#endif /* VPGOFF */
#ifdef VKILL2
   td->c_cc[VKILL2] = s[C_KILL2];
#endif /* VKILL2 */
#ifdef VMIN
   td->c_cc[VMIN] = s[C_MIN];
#endif /* VMIN */
#ifdef VTIME
   td->c_cc[VTIME] = s[C_TIME];
#endif /* VTIME */
}                               /* tty__setchar */


/* tty_bind_char():
 *	Rebind the SEditLine_t functions
 */
el_protected void
tty_bind_char(EditLine_t* el, int force) {
   unsigned char* t_n = el->fTTY.t_c[ED_IO];
   unsigned char* t_o = el->fTTY.t_ed.c_cc;
   unsigned char newp[2], old[2];
   const TTYMap_t* tp;
   ElAction_t* map, * alt;
   const ElAction_t* dmap, * dalt;
   newp[1] = old[1] = '\0';

   map = el->fMap.fKey;
   alt = el->fMap.fAlt;

   if (el->fMap.fType == MAP_VI) {
      dmap = el->fMap.fVii;
      dalt = el->fMap.fVic;
   } else {
      dmap = el->fMap.fEmacs;
      dalt = NULL;
   }

   for (tp = tty_map; tp->fNCh != -1; tp++) {
      newp[0] = t_n[tp->fNCh];
      old[0] = t_o[tp->fOCh];

      if (newp[0] == old[0] && !force) {
         continue;
      }
      /* Put the old default binding back, and set the new binding */
      key_clear(el, map, (char*) old);
      map[old[0]] = dmap[old[0]];
      key_clear(el, map, (char*) newp);
      /* MAP_VI == 1, MAP_EMACS == 0... */
      map[newp[0]] = tp->fBind[el->fMap.fType];

      if (dalt) {
         key_clear(el, alt, (char*) old);
         alt[old[0]] = dalt[old[0]];
         key_clear(el, alt, (char*) newp);
         alt[newp[0]] = tp->fBind[el->fMap.fType + 1];
      }
   }
} // tty_bind_char


/* tty_rawmode():
 *      Set terminal into 1 character at a time mode.
 */
el_protected int
tty_rawmode(EditLine_t* el) {
   if (tty_need_to_run_setup) {
      tty_setup(el);
      if (tty_need_to_run_setup)
         return 0;
   }

   if (el->fTTY.t_mode == ED_IO || el->fTTY.t_mode == QU_IO) {
      return 0;
   }

   if (el->fFlags & EDIT_DISABLED) {
      return 0;
   }

   if (tty_getty(el, &el->fTTY.t_ts) == -1) {
#ifdef DEBUG_TTY
         (void) fprintf(el->fErrFile, "tty_rawmode: tty_getty: %s\n",
                        strerror(errno));
#endif /* DEBUG_TTY */
      return -1;
   }

   /*
    * We always keep up with the eight bit setting and the speed of the
    * tty. But only we only believe changes that are made to cooked mode!
    */
   el->fTTY.t_eight = tty__geteightbit(&el->fTTY.t_ts);
   el->fTTY.t_speed = tty__getspeed(&el->fTTY.t_ts);

   if (tty__getspeed(&el->fTTY.t_ex) != el->fTTY.t_speed ||
       tty__getspeed(&el->fTTY.t_ed) != el->fTTY.t_speed) {
      (void) cfsetispeed(&el->fTTY.t_ex, el->fTTY.t_speed);
      (void) cfsetospeed(&el->fTTY.t_ex, el->fTTY.t_speed);
      (void) cfsetispeed(&el->fTTY.t_ed, el->fTTY.t_speed);
      (void) cfsetospeed(&el->fTTY.t_ed, el->fTTY.t_speed);
   }

   if (tty__cooked_mode(&el->fTTY.t_ts)) {
      if (el->fTTY.t_ts.c_cflag != el->fTTY.t_ex.c_cflag) {
         el->fTTY.t_ex.c_cflag =
            el->fTTY.t_ts.c_cflag;
         el->fTTY.t_ex.c_cflag &=
            ~el->fTTY.t_t[EX_IO][MD_CTL].t_clrmask;
         el->fTTY.t_ex.c_cflag |=
            el->fTTY.t_t[EX_IO][MD_CTL].t_setmask;

         el->fTTY.t_ed.c_cflag =
            el->fTTY.t_ts.c_cflag;
         el->fTTY.t_ed.c_cflag &=
            ~el->fTTY.t_t[ED_IO][MD_CTL].t_clrmask;
         el->fTTY.t_ed.c_cflag |=
            el->fTTY.t_t[ED_IO][MD_CTL].t_setmask;
      }

      if ((el->fTTY.t_ts.c_lflag != el->fTTY.t_ex.c_lflag) &&
          (el->fTTY.t_ts.c_lflag != el->fTTY.t_ed.c_lflag)) {
         el->fTTY.t_ex.c_lflag =
            el->fTTY.t_ts.c_lflag;
         el->fTTY.t_ex.c_lflag &=
            ~el->fTTY.t_t[EX_IO][MD_LIN].t_clrmask;
         el->fTTY.t_ex.c_lflag |=
            el->fTTY.t_t[EX_IO][MD_LIN].t_setmask;

         el->fTTY.t_ed.c_lflag =
            el->fTTY.t_ts.c_lflag;
         el->fTTY.t_ed.c_lflag &=
            ~el->fTTY.t_t[ED_IO][MD_LIN].t_clrmask;
         el->fTTY.t_ed.c_lflag |=
            el->fTTY.t_t[ED_IO][MD_LIN].t_setmask;
      }

      if ((el->fTTY.t_ts.c_iflag != el->fTTY.t_ex.c_iflag) &&
          (el->fTTY.t_ts.c_iflag != el->fTTY.t_ed.c_iflag)) {
         el->fTTY.t_ex.c_iflag =
            el->fTTY.t_ts.c_iflag;
         el->fTTY.t_ex.c_iflag &=
            ~el->fTTY.t_t[EX_IO][MD_INP].t_clrmask;
         el->fTTY.t_ex.c_iflag |=
            el->fTTY.t_t[EX_IO][MD_INP].t_setmask;

         el->fTTY.t_ed.c_iflag =
            el->fTTY.t_ts.c_iflag;
         el->fTTY.t_ed.c_iflag &=
            ~el->fTTY.t_t[ED_IO][MD_INP].t_clrmask;
         el->fTTY.t_ed.c_iflag |=
            el->fTTY.t_t[ED_IO][MD_INP].t_setmask;
      }

      if ((el->fTTY.t_ts.c_oflag != el->fTTY.t_ex.c_oflag) &&
          (el->fTTY.t_ts.c_oflag != el->fTTY.t_ed.c_oflag)) {
         el->fTTY.t_ex.c_oflag =
            el->fTTY.t_ts.c_oflag;
         el->fTTY.t_ex.c_oflag &=
            ~el->fTTY.t_t[EX_IO][MD_OUT].t_clrmask;
         el->fTTY.t_ex.c_oflag |=
            el->fTTY.t_t[EX_IO][MD_OUT].t_setmask;

         el->fTTY.t_ed.c_oflag =
            el->fTTY.t_ts.c_oflag;
         el->fTTY.t_ed.c_oflag &=
            ~el->fTTY.t_t[ED_IO][MD_OUT].t_clrmask;
         el->fTTY.t_ed.c_oflag |=
            el->fTTY.t_t[ED_IO][MD_OUT].t_setmask;
      }

      if (tty__gettabs(&el->fTTY.t_ex) == 0) {
         el->fTTY.t_tabs = 0;
      } else {
         el->fTTY.t_tabs = EL_CAN_TAB ? 1 : 0;
      }

      {
         int i;

         tty__getchar(&el->fTTY.t_ts, el->fTTY.t_c[TS_IO]);

         /*
          * Check if the user made any changes.
          * If he did, then propagate the changes to the
          * edit and execute data structures.
          */
         for (i = 0; i < C_NCC; i++) {
            if (el->fTTY.t_c[TS_IO][i] !=
                el->fTTY.t_c[EX_IO][i]) {
               break;
            }
         }

         if (i != C_NCC) {
            /*
             * Propagate changes only to the unprotected
             * chars that have been modified just now.
             */
            for (i = 0; i < C_NCC; i++) {
               if (!((el->fTTY.t_t[ED_IO][MD_CHAR].t_setmask & C_SH(i)))
                   && (el->fTTY.t_c[TS_IO][i] != el->fTTY.t_c[EX_IO][i])) {
                  el->fTTY.t_c[ED_IO][i] = el->fTTY.t_c[TS_IO][i];
               }

               if (el->fTTY.t_t[ED_IO][MD_CHAR].t_clrmask & C_SH(i)) {
                  el->fTTY.t_c[ED_IO][i] = el->fTTY.t_vdisable;
               }
            }
            tty_bind_char(el, 0);
            tty__setchar(&el->fTTY.t_ed, el->fTTY.t_c[ED_IO]);

            for (i = 0; i < C_NCC; i++) {
               if (!((el->fTTY.t_t[EX_IO][MD_CHAR].t_setmask & C_SH(i)))
                   && (el->fTTY.t_c[TS_IO][i] != el->fTTY.t_c[EX_IO][i])) {
                  el->fTTY.t_c[EX_IO][i] = el->fTTY.t_c[TS_IO][i];
               }

               if (el->fTTY.t_t[EX_IO][MD_CHAR].t_clrmask & C_SH(i)) {
                  el->fTTY.t_c[EX_IO][i] = el->fTTY.t_vdisable;
               }
            }
            tty__setchar(&el->fTTY.t_ex, el->fTTY.t_c[EX_IO]);
         }
      }
   }

   if (tty_setty(el, &el->fTTY.t_ed) == -1) {
#ifdef DEBUG_TTY
         (void) fprintf(el->fErrFile, "tty_rawmode: tty_setty: %s\n",
                        strerror(errno));
#endif /* DEBUG_TTY */
      return -1;
   }
   el->fTTY.t_mode = ED_IO;
   return 0;
} // tty_rawmode


/* tty_cookedmode():
 *	Set the tty back to normal mode
 */
el_protected int
tty_cookedmode(EditLine_t* el) {  /* set tty in normal setup */
   if (tty_need_to_run_setup) {
      tty_setup(el);
      if (tty_need_to_run_setup)
         return 0;
   }

   if (el->fTTY.t_mode == EX_IO) {
      return 0;
   }

   if (el->fFlags & EDIT_DISABLED) {
      return 0;
   }

   if (tty_setty(el, &el->fTTY.t_ex) == -1) {
#ifdef DEBUG_TTY
         (void) fprintf(el->fErrFile,
                        "tty_cookedmode: tty_setty: %s\n",
                        strerror(errno));
#endif /* DEBUG_TTY */
      return -1;
   }
   el->fTTY.t_mode = EX_IO;
   return 0;
} // tty_cookedmode


/* tty_quotemode():
 *	Turn on quote mode
 */
el_protected int
tty_quotemode(EditLine_t* el) {
   if (tty_need_to_run_setup) {
      tty_setup(el);
      if (tty_need_to_run_setup)
         return 0;
   }

   if (el->fTTY.t_mode == QU_IO) {
      return 0;
   }

   el->fTTY.t_qu = el->fTTY.t_ed;

   el->fTTY.t_qu.c_iflag &= ~el->fTTY.t_t[QU_IO][MD_INP].t_clrmask;
   el->fTTY.t_qu.c_iflag |= el->fTTY.t_t[QU_IO][MD_INP].t_setmask;

   el->fTTY.t_qu.c_oflag &= ~el->fTTY.t_t[QU_IO][MD_OUT].t_clrmask;
   el->fTTY.t_qu.c_oflag |= el->fTTY.t_t[QU_IO][MD_OUT].t_setmask;

   el->fTTY.t_qu.c_cflag &= ~el->fTTY.t_t[QU_IO][MD_CTL].t_clrmask;
   el->fTTY.t_qu.c_cflag |= el->fTTY.t_t[QU_IO][MD_CTL].t_setmask;

   el->fTTY.t_qu.c_lflag &= ~el->fTTY.t_t[QU_IO][MD_LIN].t_clrmask;
   el->fTTY.t_qu.c_lflag |= el->fTTY.t_t[QU_IO][MD_LIN].t_setmask;

   if (tty_setty(el, &el->fTTY.t_qu) == -1) {
#ifdef DEBUG_TTY
         (void) fprintf(el->fErrFile, "QuoteModeOn: tty_setty: %s\n",
                        strerror(errno));
#endif /* DEBUG_TTY */
      return -1;
   }
   el->fTTY.t_mode = QU_IO;
   return 0;
} // tty_quotemode


/* tty_noquotemode():
 *	Turn off quote mode
 */
el_protected int
tty_noquotemode(EditLine_t* el) {
   if (tty_need_to_run_setup) {
      tty_setup(el);
      if (tty_need_to_run_setup)
         return 0;
   }

   if (el->fTTY.t_mode != QU_IO) {
      return 0;
   }

   if (tty_setty(el, &el->fTTY.t_ed) == -1) {
#ifdef DEBUG_TTY
         (void) fprintf(el->fErrFile, "QuoteModeOff: tty_setty: %s\n",
                        strerror(errno));
#endif /* DEBUG_TTY */
      return -1;
   }
   el->fTTY.t_mode = ED_IO;
   return 0;
}


/* tty_stty():
 *	Stty builtin
 */
el_protected int
/*ARGSUSED*/
tty_stty(EditLine_t* el, int /*argc*/, const char** cargv) {
   char** argv = (char**) cargv;

   /** ^^^^ HUGE KLUDGE
       This func doesn't really modify argv, but does do
       pointer arithmatic on it.
       ----- stephan@s11n.net 28 Nov 2004
    */
   const TTYModes_t* m;
   char x, * d;
   int aflag = 0;
   char* s;
   char* name;
   int z = EX_IO;

   if (argv == NULL) {
      return -1;
   }
   name = *argv++;

   while (argv && *argv && argv[0][0] == '-' && argv[0][2] == '\0')
      switch (argv[0][1]) {
      case 'a':
         aflag++;
         argv++;
         break;
      case 'd':
         argv++;
         z = ED_IO;
         break;
      case 'x':
         argv++;
         z = EX_IO;
         break;
      case 'q':
         argv++;
         z = QU_IO;
         break;
      default:
         (void) fprintf(el->fErrFile,
                        "%s: Unknown switch `%c'.\n",
                        name, argv[0][1]);
         return -1;
      }

   if (!argv || !*argv) {
      int i = -1;
      int len = 0, st = 0, cu;

      for (m = ttymodes; m->fName; m++) {
         if (m->fType != i) {
            (void) fprintf(el->fOutFile, "%s%s",
                           i != -1 ? "\n" : "",
                           el->fTTY.t_t[z][m->fType].t_name);
            i = m->fType;
            st = len =
                    strlen(el->fTTY.t_t[z][m->fType].t_name);
         }
         x = (el->fTTY.t_t[z][i].t_setmask & m->fValue)
             ? '+' : '\0';
         x = (el->fTTY.t_t[z][i].t_clrmask & m->fValue)
             ? '-' : x;

         if (x != '\0' || aflag) {
            cu = strlen(m->fName) + (x != '\0') + 1;

            if (len + cu >= el->fTerm.fSize.fH) {
               (void) fprintf(el->fOutFile, "\n%*s",
                              st, "");
               len = st + cu;
            } else {
               len += cu;
            }

            if (x != '\0') {
               (void) fprintf(el->fOutFile, "%c%s ",
                              x, m->fName);
            } else {
               (void) fprintf(el->fOutFile, "%s ",
                              m->fName);
            }
         }
      }
      (void) fprintf(el->fOutFile, "\n");
      return 0;
   }

   while (argv && (s = *argv++)) {
      switch (*s) {
      case '+':
      case '-':
         x = *s++;
         break;
      default:
         x = '\0';
         break;
      }
      d = s;

      for (m = ttymodes; m->fName; m++) {
         if (strcmp(m->fName, d) == 0) {
            break;
         }
      }

      if (!m->fName) {
         (void) fprintf(el->fErrFile,
                        "%s: Invalid argument `%s'.\n", name, d);
         return -1;
      }

      switch (x) {
      case '+':
         el->fTTY.t_t[z][m->fType].t_setmask |= m->fValue;
         el->fTTY.t_t[z][m->fType].t_clrmask &= ~m->fValue;
         break;
      case '-':
         el->fTTY.t_t[z][m->fType].t_setmask &= ~m->fValue;
         el->fTTY.t_t[z][m->fType].t_clrmask |= m->fValue;
         break;
      default:
         el->fTTY.t_t[z][m->fType].t_setmask &= ~m->fValue;
         el->fTTY.t_t[z][m->fType].t_clrmask &= ~m->fValue;
         break;
      }
   }
   return 0;
} // tty_stty


#ifdef notyet

/* tty_printchar():
 *	DEbugging routine to print the tty characters
 */
el_private void
tty_printchar(EditLine_t* el, unsigned char* s) {
   TTYPerm_t* m;
   int i;

   for (i = 0; i < C_NCC; i++) {
      for (m = el->fTTY.t_t; m->fName; m++) {
         if (m->fType == MD_CHAR && C_SH(i) == m->fValue) {
            break;
         }
      }

      if (m->fName) {
         (void) fprintf(el->fErrFile, "%s ^%c ",
                        m->fName, s[i] + 'A' - 1);
      }

      if (i % 5 == 0) {
         (void) fprintf(el->fErrFile, "\n");
      }
   }
   (void) fprintf(el->fErrFile, "\n");
} // tty_printchar


#endif /* notyet */
