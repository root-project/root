// @(#)root/proof:$Id$
// Author: G. Ganis, Mar 2010

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef PQ2_actions
#define PQ2_actions

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// pq2actions                                                           //
//                                                                      //
// Prototypes for action functions used in PQ2 functions                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

void do_cache(bool clear = 1, const char *ds = nullptr);
void do_ls(const char *ds, const char *opt = "");
void do_ls_files_server(const char *ds, const char *server);
void do_info_server(const char *server);
void do_put(const char *ds, const char *opt);
void do_rm(const char *ds);
int do_verify(const char *ds, const char *opt = nullptr, const char *redir = nullptr);
void do_anadist(const char *ds, const char *newsrvs = nullptr, const char *ignsrvs = nullptr,
                const char *excsrvs = nullptr, const char *metrics = "F", const char *fout = nullptr,
                const char *plot = nullptr, const char *outfile = nullptr, const char *infile = nullptr);

#endif
