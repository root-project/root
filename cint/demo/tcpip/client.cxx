/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// client.cxx
//***************************************************************
// From the book "Win32 System Services: The Heart of Windows 95
// and Windows NT"
// by Marshall Brain
// Published by Prentice Hall
//
// Copyright 1995, by Prentice Hall.
//
// This code implements a TCP client.
//
// 1998  Modified by Masaharu Goto for CINT
//***************************************************************

#include <iostream.h>

#ifdef G__WIN32
#include <windows.h>
#include <winsock.h>
#else
#include <socket.h>
#endif

#define NO_FLAGS_SET 0

#define PORT (u_short) 44965
#define HOSTNAME
#ifdef HOSTNAME
const char *DEST_IP_HOST ="localhost";
#else
const char *DEST_IP_ADDR ="12.34.56.7";
#endif
#define MAXBUFLEN 256

////////////////////////////////////////////////////////////////////////
int main() {
  WSADATA Data;
  SOCKADDR_IN destSockAddr;
  SOCKET destSocket;
  unsigned long destAddr;
  int status;
  int numsnt;
  int numrcv;
  char sendText[MAXBUFLEN];
  char recvText[MAXBUFLEN];


  /* initialize the Windows Socket DLL */
  status=WSAStartup(MAKEWORD(1, 1), &Data);
  if (status != 0) cerr << "ERROR: WSAStartup unsuccessful" << endl;

#ifdef HOSTNAME
  struct hostent *serverHostent;

  cout << "Trying to connect to host: " << DEST_IP_HOST << endl;
  serverHostent = gethostbyname(DEST_IP_HOST);
  if(0==serverHostent) {
    cerr << "unknown host" << DEST_IP_HOST << endl;
    exit(1);
  }
  else {
    cout << "Trying to map hostname" << DEST_IP_HOST << endl;
    memcpy(&destSockAddr.sin_addr,serverHostent->h_addr_list[0]
           ,serverHostent->h_length);
  }
#else
  /* convert IP address into in_addr form*/
  cout << "Trying to connect to IP Address: " << DEST_IP_ADDR << endl;
  destAddr=inet_addr(DEST_IP_ADDR);
  /* copy destAddr into sockaddr_in structure */
  memcpy(&destSockAddr.sin_addr, &destAddr,sizeof(destAddr));
#endif

  /* specify the port portion of the address */
  destSockAddr.sin_port=htons(PORT);
  /* specify the address family as Internet */
  destSockAddr.sin_family=AF_INET;

  /* create a socket */
  destSocket=socket(AF_INET, SOCK_STREAM, 0);
  if (destSocket == INVALID_SOCKET) {
    cerr << "ERROR: socket unsuccessful" << endl;
    status=WSACleanup();
    if (status==SOCKET_ERROR) cerr<<"ERROR: WSACleanup unsuccessful"<<endl;
    return(1);
  }


  /* connect to the server */
  status=connect(destSocket,(LPSOCKADDR)(&destSockAddr),sizeof(destSockAddr));
  if (status == SOCKET_ERROR) {
    cerr << "ERROR: connect unsuccessful" << endl;
    status=closesocket(destSocket);
    if (status==SOCKET_ERROR) cerr<<"ERROR: closesocket unsuccessful"<<endl;
    status=WSACleanup();
    if (status==SOCKET_ERROR) cerr<<"ERROR: WSACleanup unsuccessful"<<endl;
    return(1);
  }

  cout << "Connected..." << endl;
  cout << "Type 'quit' to quit the session, 'kill' to kill the server" <<endl;

  while(1) {
    cout << "Type C/C++ expression to send: ";
    cin.getline(sendText, MAXBUFLEN);

    if(strcmp(sendText,"quit")==0) {
      cout << "Connection terminated." << endl;
      status=closesocket(destSocket);
      if (status==SOCKET_ERROR) cerr<<"ERROR: closesocket unsuccessful"<<endl;
      status=WSACleanup();
      if (status==SOCKET_ERROR) cerr<<"ERROR: WSACleanup unsuccessful"<<endl;
      return(1);
    }

    /* Send the message to the server */
    numsnt=send(destSocket, sendText, strlen(sendText) + 1, NO_FLAGS_SET);

    if (numsnt != (int)strlen(sendText) + 1 || strcmp(sendText,"kill")==0) {
      cout << "Connection terminated." << endl;
      status=closesocket(destSocket);
      if (status==SOCKET_ERROR) cerr<<"ERROR: closesocket unsuccessful"<<endl;
      status=WSACleanup();
      if (status==SOCKET_ERROR) cerr<<"ERROR: WSACleanup unsuccessful"<<endl;
      return(1);
    }

    /* Wait for a response from server */
    numrcv=recv(destSocket, recvText, MAXBUFLEN, NO_FLAGS_SET);

    if ((numrcv == 0) || (numrcv == SOCKET_ERROR)) {
      cout << "Connection terminated.";
      status=closesocket(destSocket);
      if (status==SOCKET_ERROR) cerr<<"ERROR: closesocket unsuccessful"<<endl;
      status=WSACleanup();
      if (status==SOCKET_ERROR) cerr<<"ERROR: WSACleanup unsuccessful"<<endl;
      return(1);
    }

    cout << "Returned result: '" <<  recvText << "'" << endl;

  } /* while */
}
