#ifndef __XRDXROOTDREQID_HH_
#define __XRDXROOTDREQID_HH_
/******************************************************************************/
/*                                                                            */
/*                     X r d X r o o t d R e q I D . h h                      */
/*                                                                            */
/* (c) 2006 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <string.h>

//         $Id$

class XrdXrootdReqID
{
public:

inline unsigned long long getID() {return Req.ID;}

inline void               getID(unsigned char *sid, int &lid,unsigned int &linst)
                               {memcpy(sid, Req.ids.Sid, sizeof(Req.ids.Sid));
                                lid = static_cast<int>(Req.ids.Lid);
                                linst = Req.ids.Linst;
                               }

inline void               setID(unsigned long long id) {Req.ID = id;}

inline void               setID(const unsigned char *sid,int lid,unsigned int linst)
                               {memcpy(Req.ids.Sid, sid, sizeof(Req.ids.Sid));
                                Req.ids.Lid = static_cast<unsigned short>(lid);
                                Req.ids.Linst = linst;
                               }

inline unsigned long long setID(const unsigned char *sid)
                               {memcpy(Req.ids.Sid, sid, sizeof(Req.ids.Sid));
                                return Req.ID;
                               }

inline unsigned char     *Stream() {return Req.ids.Sid;}

        XrdXrootdReqID(unsigned long long id) {setID(id);}
        XrdXrootdReqID(const unsigned char *sid, int lid, unsigned int linst)
                      {setID(sid ? (unsigned char *)"\0\0" : sid, lid, linst);}
        XrdXrootdReqID() {}

private:

union {unsigned long long     ID;
       struct {unsigned int   Linst;
               unsigned short Lid;
               unsigned char  Sid[2];
              } ids;
      } Req;
};
#endif
