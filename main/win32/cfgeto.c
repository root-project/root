/* @(#)root/main:$Name$:$Id$ */
/* Author: Valery Fine(fine@vxcern.cern.ch)   02/02/98 */

/*
 * $Id: cfget.c,v 1.5 1997/10/23 16:33:18 mclareni Exp $
 *
 * $Log: cfget.c,v $
 * Revision 1.5  1997/10/23 16:33:18  mclareni
 * NT mods
 *
 * Revision 1.4  1997/09/02 14:26:44  mclareni
 * WINNT correction
 *
 * Revision 1.3  1997/02/04 17:35:10  mclareni
 * Merge Winnt and 97a versions
 *
 * Revision 1.2  1997/01/15 16:25:31  cernlib
 * fix from F.Hemmer to return rfio return code
 *
 * Revision 1.1.1.1.2.1  1997/01/21 11:30:09  mclareni
 * All mods for Winnt 96a on winnt branch
 *
 * Revision 1.1.1.1  1996/02/15 17:49:36  mclareni
 * Kernlib
 *
 */

/*>    ROUTINE CFGET
  CERN PROGLIB# Z310    CFGET           .VERSION KERNFOR  4.29  910718
  ORIG. 12/01/91, JZ
      CALL CFGET (LUNDES, MEDIUM, NWREC, NWTAK, MBUF, ISTAT)
      read from the file :
       LUNDES  file descriptor
       MEDIUM  = 0,1,2,3 : primary disk/tape, secondary disk/tape
       NWREC   number of words record size
      *NWTAK*  number of words to be read / actually read
      *MBUF    vector to be read into
      *ISTAT   status, =zero if success
*/
#include <fcntl.h>
#include <stdlib.h>
#define NBYTPW 4
void __stdcall CFGET(lundes, medium, nwrec, nwtak, mbuf, stat)
      char *mbuf;
      int  *lundes, *medium, *nwrec, *nwtak, *stat;
{
      int  fildes;
      int  nbdn, nbdo;

      *stat = 0;
      if (*nwtak <= 0)            return;

/*        read the file      */

      fildes = *lundes;
      nbdo   = *nwrec * NBYTPW;
      nbdn   = read (fildes, mbuf, nbdo);
      if (nbdn == 0)               goto heof;
      if (nbdn <  0)               goto herror;
retn: *nwtak = (nbdn - 1) / NBYTPW + 1;
      return;

/*        Handle exceptions        */

heof:     *stat = -1;
          return;

#if defined(CERNLIB_PROJSHIFT)
herror:   *stat = (serrno ? serrno : (rfio_errno ? rfio_errno : errno));
#else
herror:   *stat = errno;
#endif
          perror (" error in CFGET");
          return;
}
/*> END <----------------------------------------------------------*/
