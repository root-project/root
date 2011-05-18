/*
   Copyright (c) 2002-3, Andrew McNab, University of Manchester
   All rights reserved.

   Redistribution and use in source and binary forms, with or
   without modification, are permitted provided that the following
   conditions are met:

     o Redistributions of source code must retain the above
       copyright notice, this list of conditions and the following
       disclaimer. 
     o Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials
       provided with the distribution. 

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
   CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS
   BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
   TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
   ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
   OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef VERSION
#define VERSION "x.x.x"
#endif

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>

#include <time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <dirent.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <ctype.h>

#include "gridsite.h"

void GRSThttpBodyInit(GRSThttpBody *thisbody)
{
  thisbody->size = 0; /* simple, but we don't expose internals to callers */
}

void GRSThttpPrintf(GRSThttpBody *thisbody, char *fmt, ...)
/* append printf() style format and arguments to *thisbody. */

{
  char    p[16384];
  size_t   size;
  va_list  args;

  va_start(args, fmt);
  size = vsprintf(p, fmt, args);  
  va_end(args);

  if (size >  0)
    {
      if (thisbody->size == 0) /* need to initialise */
        {
          thisbody->first = (GRSThttpCharsList *)malloc(sizeof(GRSThttpCharsList));
          thisbody->first->text = p;
          thisbody->first->next = NULL;
      
          thisbody->last = thisbody->first;          
          thisbody->size = size;
        }
      else
        {
          thisbody->last->next = (GRSThttpCharsList *)
                                               malloc(sizeof(GRSThttpCharsList));
          ((GRSThttpCharsList *) thisbody->last->next)->text = p;
          ((GRSThttpCharsList *) thisbody->last->next)->next = NULL;
      
          thisbody->last = thisbody->last->next;          
          thisbody->size = thisbody->size + size;
        }
    }
}

int GRSThttpCopy(GRSThttpBody *thisbody, char *file)
/* 
   copy a whole file, named file[], into the body output buffer, returning
   1 if file was found and copied ok, or 0 otherwise.
*/
{
  int         fd, len;
  char        *p;
  struct stat statbuf;

  fd = open(file, O_RDONLY);

  if (fd == -1) return 0;

  if (fstat(fd, &statbuf) != 0)
    {
      close(fd);
      return 0;
    }

  p = malloc(statbuf.st_size + 1);

  if (p == NULL)
    {
      close(fd);
      return 0;
    }

  len = read(fd, p, statbuf.st_size);
  p[len] = '\0';

  close(fd);
   
  if (thisbody->size == 0) /* need to initialise */
    {
      thisbody->first = (GRSThttpCharsList *) malloc(sizeof(GRSThttpCharsList));
      thisbody->first->text = p;
      thisbody->first->next = NULL;
      
      thisbody->last = thisbody->first;
      thisbody->size = len;
    }
  else
    { 
      thisbody->last->next=(GRSThttpCharsList *)malloc(sizeof(GRSThttpCharsList));
      ((GRSThttpCharsList *) thisbody->last->next)->text = p;
      ((GRSThttpCharsList *) thisbody->last->next)->next = NULL;
      
      thisbody->last = thisbody->last->next;
      thisbody->size = thisbody->size + len;
    }

  return 1;      
}

void GRSThttpWriteOut(GRSThttpBody *thisbody)
/* output Content-Length header, blank line then whole of the body to
   standard output */
{
  GRSThttpCharsList *p;
  
  printf("Content-Length: %d\n\n", (int)thisbody->size);

  p = thisbody->first;
  
  while (p != NULL)
    {
      fputs(p->text, stdout);
    
      p = p->next;      
    }
}

int GRSThttpPrintHeaderFooter(GRSThttpBody *bp, char *file, char *headfootname)
/* 
    try to print Header or Footer appropriate for absolute path file[],
    returning 1 rather than 0 if found.
*/
{
  int          found = 0;
  char        *pathfile, *p;
  struct stat  statbuf;

  pathfile = malloc(strlen(file) + strlen(headfootname) + 2);
  strcpy(pathfile, file);

  if ((pathfile[strlen(pathfile) - 1] != '/') &&
      (stat(pathfile, &statbuf) == 0) && 
       S_ISDIR(statbuf.st_mode)) strcat(pathfile, "/");
  
  for (;;)
     {
       p = rindex(pathfile, '/');
       if (p == NULL) break;
       p[1] = '\0';
       strcat(p, headfootname);

       if (stat(pathfile, &statbuf) == 0)
         {
           found = GRSThttpCopy(bp, pathfile);
           break;
         }

       p[0] = '\0';
     }

  free(pathfile);
  return found;
}

int GRSThttpPrintHeader(GRSThttpBody *bp, char *file)
{
  char *headname;
  
  headname = getenv("REDIRECT_GRST_HEAD_FILE");
  if (headname == NULL) headname = getenv("GRST_HEAD_FILE");
  if (headname == NULL) headname = GRST_HEADFILE;

  if (headname[0] == '/') /* absolute location */
    {
      return GRSThttpCopy(bp, headname);
    }
    
  return GRSThttpPrintHeaderFooter(bp, file, headname);
}

int GRSThttpPrintFooter(GRSThttpBody *bp, char *file)
{
  char *footname;
  
  footname = getenv("REDIRECT_GRST_FOOT_FILE");
  if (footname == NULL) footname = getenv("GRST_FOOT_FILE");
  if (footname == NULL) footname = GRST_FOOTFILE;

  if (footname[0] == '/') /* absolute location */
    {
      return GRSThttpCopy(bp, footname);
    }
    
  return GRSThttpPrintHeaderFooter(bp, file, footname);
}

char *GRSThttpGetCGI(char *name)
/* 
   Return a malloc()ed copy of CGI form parameter identified by name[],
   either received by QUERY_STRING (via GET) or on stdin (via POST).
   Caller must free() the returned string itself. If name[] is not found,
   an empty NUL-terminated malloc()ed string is returned. name[] has any
   URL-encoding reversed.
*/
{
  char   *p, *namepattern, *valuestart, *returnvalue, *querystring;
  int     c, i, j, n, contentlength = 0;
  static char *cgiposted = NULL;

  if (cgiposted == NULL) /* have to initialise cgiposted */
    {
      p = getenv("CONTENT_LENGTH");
      if (p != NULL) sscanf(p, "%d", &contentlength);

      querystring = getenv("REDIRECT_QUERY_STRING");
      if (querystring == NULL) querystring = getenv("QUERY_STRING");
      
      if (querystring == NULL) cgiposted = malloc(contentlength + 3);
      else cgiposted = malloc(contentlength + strlen(querystring) + 4);

      cgiposted[0] = '&';

      for (i = 1; i <= contentlength; ++i)
         {
           c = getchar();
           if (c == EOF) break;
           cgiposted[i] = c;           
         }

      cgiposted[i]   = '&';
      cgiposted[i+1] = '\0';

      if (querystring != NULL)
        {
          strcat(cgiposted, querystring);
          strcat(cgiposted, "&");
        }
    }
    
  namepattern = malloc(strlen(name) + 3);
  sprintf(namepattern, "&%s=", name);
  
  p = strstr(cgiposted, namepattern);
  free(namepattern);
  if (p == NULL) return strdup("");
     
  valuestart = &p[strlen(name) + 2];

  for (n=0; valuestart[n] != '&'; ++n) ;
  
  returnvalue = malloc(n + 1);
  
  j=0;
  
  for (i=0; i < n; ++i) 
     {
       if ((i < n - 2) && (valuestart[i] == '%')) /* url encoded as %HH */
         {
           returnvalue[j] = 0;
           
           if (isdigit(valuestart[i+1])) 
                 returnvalue[j] += 16 * (valuestart[i+1] - '0');
           else if (isalpha(valuestart[i+1])) 
                 returnvalue[j] += 16 * (10 + tolower(valuestart[i+1]) - 'a');
                         
           if (isdigit(valuestart[i+2])) 
                 returnvalue[j] += valuestart[i+2] - '0';
           else if (isalpha(valuestart[i+2])) 
                 returnvalue[j] += 10 + tolower(valuestart[i+2]) - 'a';

           i = i + 2;
         }
       else if (valuestart[i] == '+') returnvalue[j] = ' ';
       else                           returnvalue[j] = valuestart[i];
       
       if (returnvalue[j] == '\r') continue; /* CR/LF -> LF */
       ++j;
     }

  returnvalue[j] = '\0';

  return returnvalue;
}

/*                   *
 * Utility functions *
 *                   */

char *GRSThttpUrlDecode(char *in)
{
  int   i, j, n;
  char *out;
                                                                                
  n = strlen(in);
  out = malloc(n + 1);
                                                                                
  j=0;
                                                                                
  for (i=0; i < n; ++i)
     {
       if ((i < n - 2) && (in[i] == '%')) /* url encoded as %HH */
         {
           out[j] = 0;
                                                                                
           if (isdigit(in[i+1]))
                 out[j] += 16 * (in[i+1] - '0');
           else if (isalpha(in[i+1]))
                 out[j] += 16 * (10 + tolower(in[i+1]) - 'a');
                                                                                
           if (isdigit(in[i+2]))
                 out[j] += in[i+2] - '0';
           else if (isalpha(in[i+2]))
                 out[j] += 10 + tolower(in[i+2]) - 'a';
                                                                                
           i = i + 2;
         }
       else if (in[i] == '+') out[j] = ' ';
       else                   out[j] = in[i];
                                                                                
       ++j;
     }
                                                                                
  out[j] = '\0';
                                                                                
  return out;
}

char *GRSThttpUrlEncode(char *in)
/* Return a pointer to a malloc'd string holding a URL-encoded (RFC 1738)
   version of *in. Only A-Z a-z 0-9 . _ - are passed through unmodified.
   (DN's processed by GRSThttpUrlEncode can be used as valid Unix filenames,
   assuming they do not exceed restrictions on filename length.) */
{
  char *out, *p, *q;
  
  out = malloc(3*strlen(in) + 1);
  
  p = in;
  q = out;
  
  while (*p != '\0')
       {
         if (isalnum(*p) || (*p == '.') || (*p == '_') || (*p == '-'))
           {
             *q = *p;
             ++q;
           }
         else
           {
             sprintf(q, "%%%2X", *p);
             q = &q[3];
           }

         ++p;
       }
  
  *q = '\0';  
  return out;
}

char *GRSThttpUrlMildencode(char *in)
/* Return a pointer to a malloc'd string holding a partially URL-encoded
   version of *in. "Partially" means that A-Z a-z 0-9 . = - _ @ and / 
   are passed through unmodified. (DN's processed by GRSThttpUrlMildencode()
   can be used as valid Unix paths+filenames if you are prepared to
   create or simulate the resulting /X=xyz directories.) */
{
  char *out, *p, *q;
  
  out = malloc(3*strlen(in) + 1);
  
  p = in;
  q = out;
  
  while (*p != '\0')
       {
         if (isalnum(*p) || (*p == '.') || (*p == '=') || (*p == '-') 
                         || (*p == '/') || (*p == '@') || (*p == '_'))
           {
             *q = *p;
             ++q;
           }
         else if (*p == ' ')
           {
             *q = '+';
             ++q;
           }
         else
           {
             sprintf(q, "%%%2X", *p);
             q = &q[3];
           }

         ++p;
       }
  
  *q = '\0';  
  return out;
}

/// Return a one-time passcode string, for use with GridHTTP
/**
 *  Returns
 *
 *  String is timestamp+SHA1_HASH(timestamp+":"+method+":"+URL)
 *  Timestamps and hashes are in lowercase hexadecimal. Timestamps are
 *  seconds since 00:00:00 on January 1, 1970 UTC.
 */

/*
char *GRSThttpMakeOneTimePasscode(time_t timestamp, char *method, char *url)
{
  int    len, i;
  char  stringtohash[16384], hashedstring[EVP_MAX_MD_SIZE], *returnstring;
  const EVP_MD *m;
  EVP_MD_CTX ctx;

  m = EVP_sha1();
  if (m == NULL) return NULL;

  sprintf(stringtohash, "%08x:%s:%s", timestamp, method, url);
 
  EVP_DigestInit(&ctx, m);
  EVP_DigestUpdate(&ctx, stringtohash, strlen(stringtohash));
  EVP_DigestFinal(&ctx, hashedstring, &len);

  returnstring = malloc(9 + len * 2);

  sprintf(returnstring, "%08x", timestamp);

  for (i=0; 

  return returnstring;
}
*/
