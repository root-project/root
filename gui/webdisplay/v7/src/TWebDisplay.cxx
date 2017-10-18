/// \file TWebDisplay.cxx
/// \ingroup WebGui ROOT7
/// \author Sergey Linev <s.linev@gsi.de>
/// \date 2017-10-16
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TWebDisplay.hxx"

#include "ROOT/TWebDisplayManager.hxx"

#include "THttpEngine.h"
#include "THttpCallArg.h"

#include <cassert>
#include <cstring>
#include <cstdlib>

namespace ROOT {
namespace Experimental {

/// just wrapper to deliver WS call-backs to the TWebDisplay class

class TDisplayWSHandler : public THttpWSHandler {
public:
   TWebDisplay *fDispl; ///<! back-pointer to display

   TDisplayWSHandler(TWebDisplay *displ) : THttpWSHandler("name","title"), fDispl(displ) {}
   ~TDisplayWSHandler() { fDispl = nullptr; }

   virtual Bool_t ProcessWS(THttpCallArg *arg) override
   {
      return fDispl->ProcessWS(arg);
   }

};


} // namespace Experimental
} // namespace ROOT

ROOT::Experimental::TWebDisplay::~TWebDisplay()
{
   if (fMgr)
      fMgr->CloseDisplay(this);

   if (fWSHandler) {
      delete fWSHandler;
      fWSHandler = nullptr;
   }
}

// TODO: add callback which executed when exactly this window is opened and connection is established

bool ROOT::Experimental::TWebDisplay::Show(const std::string &where)
{
   if (!fMgr) return false;

   bool first_time = false;

   if (!fWSHandler) {
      fWSHandler = new TDisplayWSHandler(this);
      fWSHandler->SetName(Form("win%u",GetId()));
      first_time = true;
   }

   return fMgr->Show(this, where, first_time);
}


bool ROOT::Experimental::TWebDisplay::ProcessWS(THttpCallArg *arg)
{
   if (!arg)
      return kTRUE;

   // try to identify connection for given WS request
   WebConn *conn = 0;
   auto iter = fConn.begin();
   while (iter != fConn.end()) {
      if (iter->fHandle && (iter->fHandle->GetId() == arg->GetWSId()) && arg->GetWSId()) {
         conn = &(*iter);
         break;
      }
      ++iter;
   }

   if (strcmp(arg->GetMethod(), "WS_CONNECT") == 0) {

      // accept all requests, in future one could limit number of connections
      // arg->Set404(); // refuse connection
      return true;
   }

   if (strcmp(arg->GetMethod(), "WS_READY") == 0) {
      THttpWSEngine *wshandle = dynamic_cast<THttpWSEngine *>(arg->TakeWSHandle());

      assert(conn == 0  && "WSHandle with given websocket id exists!!!");

      WebConn newconn;
      newconn.fHandle = wshandle;
      newconn.fConnId = ++fConnCnt; // unique connection id
      fConn.push_back(newconn);

      // CheckDataToSend();

      return true;
   }

   if (strcmp(arg->GetMethod(), "WS_CLOSE") == 0) {
      // connection is closed, one can remove handle

      // UInt_t connid = 0;

      if (conn && conn->fHandle) {
         // connid = conn->fHandle->GetId();
         conn->fHandle->ClearHandle();
         delete conn->fHandle;
         conn->fHandle = 0;
      }

      if (conn)
         fConn.erase(iter);

      // if there are no other connections - cancel all submitted commands
      // CancelCommands((fWebConn.size() == 0), connid);

      // CheckDataToSend(); // check if data should be send via other connections

      return true;
   }

   assert(strcmp(arg->GetMethod(), "WS_DATA") == 0 && "WS_DATA request expected!");

   assert(conn != 0 && "Get websocket data without valid connection - ignore!!!");

   if (conn->fHandle->PreviewData(arg))
      return true;

   if (arg->GetPostDataLength() <= 0)
      return true;

   // here processing of received data should be performed
   // this is task for the implemented windows

   const char *buf = (const char *)arg->GetPostData();
   char *str_end = 0;

   printf("Get portion of data %d %s\n", arg->GetPostDataLength(), buf);

   unsigned long ackn_oper = std::strtoul(buf, &str_end, 10);
   assert(str_end != 0 && *str_end != ':' && "missing number of acknowledged operations");

   unsigned long can_send = std::strtoul(str_end+1, &str_end, 10);
   assert(str_end != 0 && *str_end != ':' && "missing can_send counter");

   unsigned long nchannel = std::strtoul(str_end+1, &str_end, 10);
   assert(str_end != 0 && *str_end != ':' && "missing channel number");

   unsigned processed_len = (str_end + 1 - buf);

   assert(processed_len <= arg->GetPostDataLength() && "corrupted buffer");

   std::string cdata(str_end+1, arg->GetPostDataLength() - processed_len);

   conn->fSendCredits += ackn_oper;

   if (nchannel == 1) fDataCallback(conn->fConnId, cdata);
   else if (nchannel > 1) conn->fCallBack(conn->fConnId, cdata);

   CheckDataToSend();


/*

   std::string cdata((const char *)arg->GetPostData(), arg->GetPostDataLength());

   if (cdata.find("READY") == 0) {
      conn->fReady = kTRUE;
      CheckDataToSend();
   } else if (cdata.find("SNAPDONE:") == 0) {
      cdata.erase(0, 9);
      conn->fReady = kTRUE;
      conn->fDrawReady = kTRUE;                       // at least first drawing is performed
      conn->fDelivered = (uint64_t)std::stoll(cdata); // delivered version of the snapshot
      CheckDataToSend();
   } else if (cdata.find("RREADY:") == 0) {
      conn->fReady = kTRUE;
      conn->fDrawReady = kTRUE; // at least first drawing is performed
      CheckDataToSend();
   } else if (cdata.find("GETMENU:") == 0) {
      conn->fReady = kTRUE;
      cdata.erase(0, 8);
      conn->fGetMenu = cdata;
      CheckDataToSend();
   } else if (cdata == "QUIT") {
      if (gApplication)
         gApplication->Terminate(0);
   } else if (cdata == "RELOAD") {
      conn->fSend = 0; // reset send version, causes new data sending
      CheckDataToSend();
   } else if (cdata == "INTERRUPT") {
      gROOT->SetInterrupt();
   } else if (cdata.find("REPLY:") == 0) {
      cdata.erase(0, 6);
      const char *sid = cdata.c_str();
      const char *separ = strchr(sid, ':');
      std::string id;
      if (separ)
         id.append(sid, separ - sid);
      if (fCmds.size() == 0) {
         Error("ProcessWS", "Get REPLY without command");
      } else if (!fCmds.front().fRunning) {
         Error("ProcessWS", "Front command is not running when get reply");
      } else if (fCmds.front().fId != id) {
         Error("ProcessWS", "Mismatch with front command and ID in REPLY");
      } else {
         bool res = FrontCommandReplied(separ + 1);
         PopFrontCommand(res);
      }
      conn->fReady = kTRUE;
      CheckDataToSend();
   } else if (cdata.find("SAVE:") == 0) {
      cdata.erase(0,5);
      SaveCreatedFile(cdata);
   } else if (cdata.find("OBJEXEC:") == 0) {
      cdata.erase(0, 8);
      size_t pos = cdata.find(':');

      if ((pos != std::string::npos) && (pos > 0)) {
         std::string id(cdata, 0, pos);
         cdata.erase(0, pos + 1);
         ROOT::Experimental::TDrawable *drawable = FindDrawable(fCanvas, id);
         if (drawable && (cdata.length() > 0)) {
            printf("Execute %s for drawable %p\n", cdata.c_str(), drawable);
            drawable->Execute(cdata);
         } else if (id == ROOT::Experimental::TDisplayItem::MakeIDFromPtr((void *)&fCanvas)) {
            printf("Execute %s for canvas itself (ignore for the moment)\n", cdata.c_str());
         }
      }
   } else if (cdata == "KEEPALIVE") {
      // do nothing, it is just keep alive message for websocket
   }
*/
   return true;
}

void ROOT::Experimental::TWebDisplay::SendDataViaConnection(ROOT::Experimental::TWebDisplay::WebConn &conn, int chid, const std::string &data)
{
   assert(conn.fSendCredits>0 && "No credits to send data via connection");

   std::string buf;
   buf.reserve(data.length() + 100);

   buf.append(std::to_string(conn.fRecvCount));
   buf.append(":");
   conn.fRecvCount = 0; // we confirm how many packages was received
   conn.fSendCredits--;

   if (chid >= 0) {
      buf.append(std::to_string(chid));
      buf.append(":");
   }

   // TODO: should we add extra : just as placeholder for any kind of custom data??
   buf.append(data);

   conn.fHandle->SendCharStar(buf.c_str());
}

/// Check if data to any connection can be send
void ROOT::Experimental::TWebDisplay::CheckDataToSend(bool only_once)
{
   bool isany = false;

   do {
      isany = false;

      for (auto iter = fConn.begin(); iter != fConn.end(); ++iter) {
         if ((iter->fSendCredits > 0) && (iter->fQueue.size() > 0)) {
            SendDataViaConnection(*iter, 1, iter->fQueue.front());
            iter->fQueue.pop_front();
            isany = true;
         }
      }

   } while (isany && !only_once);
}


/// Sends data to specified connection
/// If connid==0, data will be send to all connections for channel 1
void ROOT::Experimental::TWebDisplay::Send(const std::string &data, unsigned connid)
{
   for (auto iter = fConn.begin(); iter != fConn.end(); ++iter) {
      if (connid && connid != iter->fConnId) connid++;

      if ((iter->fQueue.size()==0) && (iter->fSendCredits>0)) {
         SendDataViaConnection(*iter, 1, data);
      } else {
         assert(iter->fQueue.size() < fMaxQueueLength && "Maximum queue length achieved");
         iter->fQueue.push_back(data);
      }
   }

   CheckDataToSend();
}
