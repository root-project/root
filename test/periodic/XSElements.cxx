/*
 * $Header$
 * $Log$
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "XSElements.h"

//ClassImp(XSElement)

/* =================== XSElement ===================== */
XSElement::XSElement()
{
   z = 0;
   name   = NULL;
   symbol   = NULL;
   isotope   = NULL;
   atomic_weight   = NULL;
   density   = NULL;
   melting_point   = NULL;
   boiling_point   = NULL;
   oxidation_states   = NULL;
   isotope   = NULL;
} // XSElement

/* ---------- ~XSElement ----------- */
XSElement::~XSElement()
{
   if (name)      free(name);
   if (symbol)      free(symbol);
   if (atomic_weight)   free(atomic_weight);
   if (density)      free(density);
   if (melting_point)   free(melting_point);
   if (boiling_point)   free(boiling_point);
   if (oxidation_states)   free(oxidation_states);
   for (int i=0; i<ni; i++) {
      free(isotope[i]);
      free(isotope_info[i]);
   }
   free(isotope);
   free(isotope_info);
   free(isotope_stable);
} // ~XSElement

/* ------- IsotopeInfo ------- */
// Search for information by name
const char*
XSElement::IsotopeInfo( const char *isot )
{
   for (int i=0; i<ni; i++)
      if (!strcmp(isotope[i],isot))
         return isotope_info[i];

   return "-";
} // IsotopeInfo

/* ------- ReadLine ------- */
/* Reads one line and allocates a string for it */
char *
XSElement::ReadLine(FILE *f)
{
   char   buf[256];
   char   *p=buf;
   char   ch;

   /* skip leading spaces */
   do {
      ch=fgetc(f);
   } while (isspace(ch));

   do {
      *p++ = ch;
      ch = fgetc(f);
   } while (ch != '\n');
   *p = 0;
   return strdup(buf);
} /* ReadLine */

/* ------- Read -------- */
void
XSElement::Read(FILE *f)
{
   char   tmpsym[5], tmpname[30];
   fscanf(f,"%d %s %s %d",&z,tmpsym,tmpname,&ni);

   symbol = strdup(tmpsym);
   name = strdup(tmpname);

   if (ni==0) return;

   atomic_weight = ReadLine(f);
   density = ReadLine(f);
   melting_point = ReadLine(f);
   boiling_point = ReadLine(f);
   oxidation_states = ReadLine(f);

   isotope = (char **)malloc(ni*sizeof(char*));
   isotope_info = (char **)malloc(ni*sizeof(char*));
   isotope_stable = (Bool_t *)malloc(ni*sizeof(Bool_t));

   for (int i=0; i<ni; i++) {
      char   ch;
      char   buf[30];

      /* get first character */
      ch = fgetc(f);
      if (ch != '*')
         ungetc(ch,f);   // Put it back

      fscanf(f,"%s",buf);
      isotope[i] = strdup(buf);
      isotope_info[i] = ReadLine(f);

      isotope_stable[i] = (ch=='*');
   }
} // Read

/* =================== XSElements ===================== */
//ClassImp(XSElements)

XSElements::XSElements(const char *filename)
{
   FILE   *f;

   if ((f=fopen(filename,"r"))==NULL) {
      fprintf(stderr,"XSElements::XSElements: Error opening file %s\n",filename);
      exit(0);
   }

   fscanf(f,"%d",&NElements);
   elements = new TObjArray(NElements);

   for (UInt_t i=0; i<NElements; i++) {
      elements->Add(new XSElement());
      ((XSElement*)(*elements)[i])->Read(f);
   }
   fclose(f);
} // XSElements

/* --------- ~XSElements ---------- */
XSElements::~XSElements()
{
   delete   elements;
} // ~XSElements

/* --------- Find ----------- */
UInt_t
XSElements::Find(const char *str)
{
   for (UInt_t z=1; z<=NElements; z++) {
      if (!strcmp(str,Name(z)))
         return z;
      if (!strcmp(str,Mnemonic(z)))
         return z;
   }
   return 0;
} // Find
