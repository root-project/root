#ifndef __SEC_ENTITY_H__
#define __SEC_ENTITY_H__
/******************************************************************************/
/*                                                                            */
/*                       X r d S e c E n t i t y . h h                        */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//       $Id$

// This structure is returned during authentication. This is most relevant
// for client authentication unless mutual authentication has been implemented
// in which case the client can also authenticate the server. It is embeded
// in each protocol object to facilitate mutual authentication. Note that the
// destructor does nothing and it is the responsibility of the protocol object
// to delete the XrdSecEntity data members, if need be. This is because
// there can only be one destructor instance for the class and it is ambiguous
// as to which shared library definition should be used. Since protocol objects
// have unique class names, each one can have a private destructor avoiding
// platform specific run-time loader address resolution ecentricities. The OO
// "fix" for this problem would require protocols to define a derived private
// destructor for this object which is more hassle than it's worth.
//
#define XrdSecPROTOIDSIZE 8

class  XrdSecEntity
{
public:
         char   prot[XrdSecPROTOIDSIZE];  // Protocol used
         char   *name;                    // Entity's name
         char   *host;                    // Entity's host name
         char   *vorg;                    // Entity's virtual organization
         char   *role;                    // Entity's role
         char   *grps;                    // Entity's group names
         char   *endorsements;            // Protocol specific endorsements
         char   *tident;                  // Trace identifier (do not touch)

         XrdSecEntity() {prot[0] = '\0';
                         name=host=vorg=role=grps=endorsements=tident = 0;
                        }
        ~XrdSecEntity() {}
};

#define XrdSecClientName XrdSecEntity
#define XrdSecServerName XrdSecEntity
#endif
