/* $XConsortium: pr.c /main/20 1996/12/04 10:11:41 swick $ */
/*

Copyright (c) 1993, 1994  X Consortium

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
X CONSORTIUM BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Except as contained in this notice, the name of the X Consortium shall not be
used in advertising or otherwise to promote the sale, use or other dealings
in this Software without prior written authorization from the X Consortium.

*/

#include "def.h"

extern struct inclist inclist[ MAXFILES ],
            *inclistp;
extern char *objprefix;
extern char *objsuffix;
extern int width;
extern boolean printed;
extern boolean verbose;
extern boolean show_where_not;

extern void included_by(struct inclist *ip, struct inclist *newfile);
extern int find_includes(struct filepointer *filep, struct inclist *file,
                            struct inclist *file_red, int recursion,
                            boolean failOK);
extern void freefile(struct filepointer *fp);

extern void ROOT_adddep(char* buf, size_t len);
extern void ROOT_newFile();

void add_include(struct filepointer *filep, struct inclist *file, struct inclist *file_red, char *include, boolean dot,
                 boolean failOK)
{
   register struct inclist *newfile;
   register struct filepointer *content;

   /*
    * First decide what the pathname of this include file really is.
    */
   newfile = inc_path(file->i_file, include, dot);
   if (newfile == NULL) {
      if (failOK)
         return;
      if (file != file_red)
         warning("%s (reading %s, line %d): ",
                 file_red->i_file, file->i_file, filep->f_line);
      else
         warning("%s, line %d: ", file->i_file, filep->f_line);
      warning1("cannot find include file \"%s\"\n", include);
      show_where_not = TRUE;
      newfile = inc_path(file->i_file, include, dot);
      show_where_not = FALSE;
   }

   if (newfile) {
      included_by(file, newfile);
      if (!(newfile->i_flags & SEARCHED)) {
         newfile->i_flags |= SEARCHED;
         if (strncmp(newfile->i_file, "/usr/include/", 13)) {
            content = getfile(newfile->i_file);
            find_includes(content, newfile, file_red, 0, failOK);
            freefile(content);
         }
      }
   }
}

void pr(register struct inclist *ip, char *file, char *base, char *dep)
{
   static char *lastfile;
   static int current_len;
   register int len, i;
   char buf[ BUFSIZ ];
   char    *ipifile;

   printed = TRUE;
   len = strlen(ip->i_file) + 1;
   ipifile = 0;
   if (len > 2 && ip->i_file[1] == ':') {
      if (getenv("OSTYPE") && !strcmp(getenv("OSTYPE"), "msys")) {
         /* windows path */
         ipifile = malloc(len);
         strcpy(ipifile, ip->i_file);
         ipifile[1] = ipifile[0];
         ipifile[0] = '/';
      } else {
#ifdef _MSC_VER
         /* native Windows */
         ipifile = malloc(len);
         strcpy(ipifile, ip->i_file);
#else
         /* generic cygwin */
         ipifile = malloc(len + 11);
         strcpy(ipifile, "/cygdrive/");
         ipifile[10] = ip->i_file[0];
         strcpy(ipifile + 11, ip->i_file + 2);
         len += 9;
#endif
      }
   } else ipifile = ip->i_file;

   if (current_len + len > width || file != lastfile) {
      lastfile = file;
      if (rootBuild)
         ROOT_newFile();
      if (dep == 0) {
         sprintf(buf, "\n%s%s%s: %s", objprefix, base, objsuffix,
                 ipifile);
      } else {
         sprintf(buf, "\n%s: %s", dep,
                 ipifile);
      }
      len = current_len = strlen(buf);
   } else {
      buf[0] = ' ';
      strcpy(buf + 1, ipifile);
      current_len += len;
   }
   if (len > 2 && ip->i_file[1] == ':')
      free(ipifile);

   if (rootBuild)
      ROOT_adddep(buf, len);
   else
      if (fwrite(buf, len, 1, stdout) != 1)
         fprintf(stderr, "pr: fwrite error\n");

   /*
    * If verbose is set, then print out what this file includes.
    */
   if (! verbose || ip->i_list == NULL || ip->i_flags & NOTIFIED)
      return;
   ip->i_flags |= NOTIFIED;
   lastfile = NULL;
   printf("\n# %s includes:", ip->i_file);
   for (i = 0; i < ip->i_listlen; i++)
      printf("\n#\t%s", ip->i_list[ i ]->i_incstring);
}

void recursive_pr_include(register struct inclist *head, register char *file, register char *base, register char *dep)
{
   register int i;

   if (head->i_flags & MARKED)
      return;
   head->i_flags |= MARKED;
   if (head->i_file != file)
      pr(head, file, base, dep);
   for (i = 0; i < head->i_listlen; i++)
      recursive_pr_include(head->i_list[ i ], file, base, dep);
}
