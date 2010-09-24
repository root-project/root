/*
 * $Header$
 * $Log$
 *
 * Various routines, and global variables
 */

#ifndef __XSVARIOUS_H
#define __XSVARIOUS_H

#include <TROOT.h>
#include <TList.h>
#include <TCanvas.h>
#include <TRootEmbeddedCanvas.h>

#include "NdbMTReacDesc.h"
#include "NdbMTReactionXS.h"

#include "XSGraph.h"
#include "XSElements.h"

#ifdef __XSVARIOUS_CXX
#	define	EXT
#else
#	define	EXT	extern
#endif

/* ---------------- Global DEFINES ---------------------- */
#define PATHSEP			"/"
#define	ICONDIR			"icons" PATHSEP
#define	DBDIR			"db" PATHSEP
#define	PTBL_ICON		ICONDIR "ptable_s.xpm"
#define	ISOTOPES_DESC_FILE	DBDIR "isotopes.dat"
#define	MT_DESC_FILE		DBDIR "mt.dat"

/* ---------------- Global Variables -------------------- */
EXT	XSElements	*XSelements;
EXT	NdbMTReacDesc	*XSReactionDesc;
EXT	TList		*graphList;

// --- GUI vars ---
EXT	GCValues_t       gval;
EXT	FontStruct_t     fixedFontStruct;
EXT	FontStruct_t     blueFontStruct;
EXT	GContext_t       fixedGC;
EXT	GContext_t       blueBoldGC;

EXT     TCanvas			*canvas;
EXT	TRootEmbeddedCanvas	*canvasWindow;

/* ----------------- function prototypes ---------------- */
void	XSinitialise();
void	XSfinalise();

void	Add2GraphList( XSGraph *gr);

#undef EXT
#endif
