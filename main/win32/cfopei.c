/* @(#)root/main:$Name$:$Id$ */
/* Author: Valery Fine(fine@vxcern.cern.ch)   02/02/98 */

/*
 * $Id: cfopei.c,v 1.4 1997/09/02 14:26:46 mclareni Exp $
 *
 * $Log: cfopei.c,v $
 * Revision 1.4  1997/09/02 14:26:46  mclareni
 * WINNT correction
 *
 * Revision 1.3  1997/02/04 17:35:11  mclareni
 * Merge Winnt and 97a versions
 *
 * Revision 1.2  1997/01/15 16:25:32  cernlib
 * fix from F.Hemmer to return rfio return code
 *
 * Revision 1.1.1.1.2.1  1997/01/21 11:30:10  mclareni
 * All mods for Winnt 96a on winnt branch
 *
 * Revision 1.1.1.1  1996/02/15 17:49:36  mclareni
 * Kernlib
 *
 */
/*>    ROUTINE CFOPEI
  CERN PROGLIB# Z310    CFOPEI          .VERSION KERNFOR  4.38  931108
  ORIG. 12/01/91, JZ
      CALL CFOPEN (LUNDES, MEDIUM, NWREC, MODE, NBUF, TEXT, ISTAT)
      open a file :
      *LUNDES  file descriptor
       MEDIUM  = 0,1,2,3 : primary disk/tape, secondary disk/tape
       NWREC   record length in number of words
       MODE    string selecting IO mode
               = 'r ', 'w ', 'a ', 'r+ ', ...
       NBUF    number of buffers to be allocated, (not used)
       TEXT    name of the file
      *ISTAT   status, =zero if success
*/
#include <fcntl.h>
#include <stdlib.h>
      int cfopen_perm = 0;

void __stdcall CFOPEI(lundes,medium,nwrec,mode,nbuf,ftext,len_ftext,stat,lgtx)
      int len_ftext;
      char *ftext;
      int  *lundes, *medium, *nwrec, *nbuf, *stat, *lgtx;
      int  *mode;
{
      char *pttext, *fchtak();
      int  flags;
      int  fildes;
      int  perm;

      *lundes = 0;
      *stat   = -1;

      perm = cfopen_perm;
      cfopen_perm = 0;

/*        construct flags :
            mode[0] =    0 r    1 w    2 a
            mode[1] =    1 +
*/
/*        flags for disk     */

      if (*medium == 1)            goto fltp;
      if (*medium == 3)            goto fltp;

      if (mode[0] == 0)
        {if (mode[1] == 0)
          flags = O_RDONLY;
        else
          flags = O_RDWR;}

      else if (mode[0] == 1)
        {if (mode[1] == 0)
          flags = O_WRONLY | O_CREAT | O_TRUNC;
        else
          flags = O_RDWR | O_CREAT | O_TRUNC;}

      else if (mode[0] == 2)
        {if (mode[1] == 0)
          flags = O_WRONLY | O_CREAT | O_APPEND;
        else
          flags = O_RDWR | O_CREAT | O_APPEND;}
      goto act;

/*        flags for tape     */

fltp: if (mode[0] == 0)
        {if (mode[1] == 0)
          flags = O_RDONLY;
        else
          flags = O_RDWR;}

      else if (mode[0] == 1)
        {if (mode[1] == 0)
          flags = O_WRONLY;
        else
          flags = O_RDWR;}

      else if (mode[0] == 2)       return;

/*        open the file      */

act:  pttext = fchtak(ftext,*lgtx);
      if (pttext == 0)             return;

      if (perm == 0)   perm = 0644;

      fildes = open (pttext, flags | O_BINARY, perm);
      if (fildes < 0)              goto errm;
      *lundes = fildes;
      *stat   = 0;
      goto done;

#if defined(CERNLIB_PROJSHIFT)
errm: *stat = (serrno ? serrno : (rfio_errno ? rfio_errno : errno));
#else
errm: *stat = errno;
#endif
      perror (" error in CFOPEN");

done: free(pttext);
      return;
}
/*> END <----------------------------------------------------------*/
