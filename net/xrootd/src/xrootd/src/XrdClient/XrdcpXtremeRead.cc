//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdXtremeRead                                                        //
//                                                                      //
// Author: Fabrizio Furano (CERN, 2009)                                 //
//                                                                      //
// Utility classes handling coordinated parallel reads from multiple    //
// XrdClient instances                                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//         $Id$

const char *XrdXtremeReadCVSID = "$Id$";

#include "XrdClient/XrdcpXtremeRead.hh"
#include "XrdClient/XrdClientAdmin.hh"

XrdXtRdFile::XrdXtRdFile(int blksize, long long filesize) {
   blocks = 0;
   clientidxcnt = 0;
   freeblks = 0;
   doneblks = 0;


   freeblks = nblks = (filesize + blksize - 1) / blksize;

   blocks = new XrdXtRdBlkInfo[nblks];

   // Init the list of blocks
   long long ofs = 0;
   for (int i = 0; i < nblks; i++) {
      blocks[i].offs = ofs;
      blocks[i].len = xrdmax(0, xrdmin(filesize, ofs+blksize) - ofs);
      ofs += blocks[i].len;
   }

}

XrdXtRdFile::~XrdXtRdFile() {
   delete []blocks;
}

int XrdXtRdFile::GimmeANewClientIdx() {
   XrdSysMutexHelper m(mtx);
   return ++clientidxcnt;
}

int XrdXtRdFile::GetBlkToPrefetch(int fromidx, int clientidx, XrdXtRdBlkInfo *&blkreadonly) {
   // Considering fromidx as a starting point in the blocks array,
   // finds a block which is worth prefetching
   // If there are free blocks it's trivial
   // Otherwise it will be stolen from other readers which are clearly late

   XrdSysMutexHelper m(mtx);


   // Find a non assigned blk
   for (int i = 0; i < nblks; i++) {
      int pos = (fromidx + i) % nblks;

      // Find a non assigned blk
      if (blocks[pos].requests.GetSize() == 0) {
         blocks[pos].requests.Push_back(clientidx);
         blocks[pos].lastrequested = time(0);
         blkreadonly = &blocks[pos];
         return pos;
      }   
   }

   // Steal an outstanding missing block, even if in progress
   // The outcome of this is that, at the end, all thethe fastest free clients will
   // ask for the missing blks
   // The only thing to avoid is that a client asks twice the same blk for itself

   for (int i = nblks; i > 0; i--) {
      int pos = (fromidx + i) % nblks;

      // Find a non finished blk to steal
      if (!blocks[pos].done && !blocks[pos].AlreadyRequested(clientidx) &&
          (blocks[pos].requests.GetSize() < 3) ) {

         blocks[pos].requests.Push_back(clientidx);
         blkreadonly = &blocks[pos];
         blocks[pos].lastrequested = time(0);
         return pos;
      }
   }

   // No blocks to request or steal... probably everything's finished
   return -1;

}

int XrdXtRdFile::GetBlkToRead(int fromidx, int clientidx, XrdXtRdBlkInfo *&blkreadonly) {
   // Get the next already prefetched block, now we want to get its content

   XrdSysMutexHelper m(mtx);

   for (int i = 0; i < nblks; i++) {
      int pos = (fromidx + i) % nblks;
      if (!blocks[pos].done &&
          blocks[pos].AlreadyRequested(clientidx)) {

         blocks[pos].lastrequested = time(0);
         blkreadonly = &blocks[pos];
         return pos;
      }
   }

   return -1;
}

int XrdXtRdFile::MarkBlkAsRead(int blkidx) {
   XrdSysMutexHelper m(mtx);

   int reward = 0;

   // If the block was stolen by somebody else then the reward is negative
   if (blocks[blkidx].done) reward = -1;
   if (!blocks[blkidx].done) {
      doneblks++;
      if (blocks[blkidx].requests.GetSize() > 1) reward = 1;
   }


   blocks[blkidx].done = true;
   return reward;
}


int XrdXtRdFile::GetListOfSources(XrdClient *ref, XrdOucString xtrememgr, XrdClientVector<XrdClient *> &clients) {
   // Exploit Locate in order to find as many sources as possible.
   // Make sure that ref appears once and only once
   // Instantiate and open the relative client instances

   XrdClientVector<XrdClientLocate_Info> hosts;
   if (xtrememgr == "") return 0;

   // In the simple case the xtrememgr is just the host of the original url.
   if (!xtrememgr.beginswith("root://") && !xtrememgr.beginswith("xroot://")) {
      
      // Create an acceptable xrootd url
      XrdOucString loc2;
      loc2 = "root://";
      loc2 += xtrememgr;
      loc2 += "/xyz";
      xtrememgr = loc2;
   }

   XrdClientAdmin adm(xtrememgr.c_str());
   if (!adm.Connect()) return 0;

   int locateok = adm.Locate((kXR_char *)ref->GetCurrentUrl().File.c_str(), hosts);
   if (!locateok || !hosts.GetSize()) return 0;

   // Here we have at least a result... hopefully
   bool found = false;
   for (int i = 0; i < hosts.GetSize(); i++)
      if (ref->GetCurrentUrl().HostWPort == (const char *)(hosts[i].Location)) {
         found = true;
         break;
      }

   // Now initialize the clients and start the parallel opens
   for (int i = 0; i < hosts.GetSize(); i++) {
      XrdOucString loc;

      loc = "root://";
      loc += (const char *)hosts[i].Location;
      loc += "/";
      loc += ref->GetCurrentUrl().File;
      cout << "Source #" << i+1 << " " << loc << endl;

      XrdClient *cli = new XrdClient(loc.c_str());
      if (cli) {
         if (cli->Open(0, 0, true)) {
            clients.Push_back(cli);
         }
         else {
            delete cli;
            cli = 0;
         }
      }

   }

   // Eventually add the ref client to the vector
   if (!found && ref) clients.Push_back(ref);

   return clients.GetSize();
}
