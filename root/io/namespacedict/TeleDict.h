/********************************************************************
* TeleDict.h
********************************************************************/
#ifdef __CINT__
#error TeleDict.h/C is only for compilation. Abort cint.
#endif
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define G__ANSIHEADER
#define G__DICTIONARY
#include "G__ci.h"
extern "C" {
extern void G__cpp_setup_tagtableTeleDict();
extern void G__cpp_setup_inheritanceTeleDict();
extern void G__cpp_setup_typetableTeleDict();
extern void G__cpp_setup_memvarTeleDict();
extern void G__cpp_setup_globalTeleDict();
extern void G__cpp_setup_memfuncTeleDict();
extern void G__cpp_setup_funcTeleDict();
extern void G__set_cpp_environmentTeleDict();
}


#include "TROOT.h"
#include "TMemberInspector.h"
#include "Tele.h"

#ifndef G__MEMFUNCBODY
#endif

extern G__linked_taginfo G__TeleDictLN_SashcLcLEnvelopeEntrylESashcLcLTelescopecORecocLcLHillasParametersgR;
extern G__linked_taginfo G__TeleDictLN_SashcLcLEnvelopeEntrySimplelESashcLcLTelescopecORecocLcLHillasParametersgR;

/* STUB derived class for protected member access */
typedef Sash::EnvelopeEntry<Sash::Telescope,Reco::HillasParameters> G__SashcLcLEnvelopeEntrylESashcLcLTelescopecORecocLcLHillasParametersgR;
typedef Sash::EnvelopeEntrySimple<Sash::Telescope,Reco::HillasParameters> G__SashcLcLEnvelopeEntrySimplelESashcLcLTelescopecORecocLcLHillasParametersgR;
