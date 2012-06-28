/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// server.cxx
//***************************************************************
// From the book "Win32 System Services: The Heart of Windows 95
// and Windows NT"
// by Marshall Brain
// Published by Prentice Hall
//
// Copyright 1995, by Prentice Hall.
//
// This code implements a TCP server.
//
// 1998  Modified by Masaharu Goto for CINT
//***************************************************************

#include <iostream.h>
#ifdef G__WIN32
#include <windows.h>
#endif
#include <winsock.h>
//#include <process.h>

#ifdef FORK
#include <unistd.h>
#endif

#define NO_FLAGS_SET 0

#define PORT (u_short) 44965
#define MAXBUFLEN 512

////////////////////////////////////////////////////////////////////////
void talkToClient(void *cs) {
  char buffer[MAXBUFLEN];
  char result[MAXBUFLEN];
  int status;
  int numsnt;
  int numrcv;
  SOCKET clientSocket=(SOCKET)cs;

  while(1) {
    numrcv=recv(clientSocket, buffer, MAXBUFLEN, NO_FLAGS_SET);
    if ((numrcv == 0) || (numrcv == SOCKET_ERROR)) {
      cout << "Connection terminated" << endl;
      break;
    }

    cout << "Received : '" << buffer << "'" << endl;

    if(strcmp(buffer,"kill")==0|| strcmp(buffer,"KILL")==0) {
      status=closesocket(clientSocket);
      if (status == SOCKET_ERROR)
	cerr << "ERROR: closesocket unsuccessful" << endl;
      exit(0);
    }

    /* CINT on the server side evaluates C/C++ statement */
    G__exec_text_str(buffer,result);
    cout << "Send back : '" << result << "'" << endl;

    numsnt=send(clientSocket, result, strlen(result) + 1, NO_FLAGS_SET);
    if (numsnt != (int)strlen(result) + 1) {
      cout << "Connection terminated." << endl;
      break;
    }

  } /* while */

  /* terminate the connection with the client (disable sending/receiving) */
  status=shutdown(clientSocket, 2);
  if (status==SOCKET_ERROR) cerr << "ERROR: shutdown unsuccessful" << endl;

  /* close the socket */
  status=closesocket(clientSocket);
  if (status==SOCKET_ERROR) cerr << "ERROR: closesocket unsuccessful" << endl;
}

////////////////////////////////////////////////////////////////////////
int main() {
  WSADATA Data;
  SOCKADDR_IN serverSockAddr;
  SOCKADDR_IN clientSockAddr;
  SOCKET serverSocket;
  SOCKET clientSocket;
  int addrLen=sizeof(SOCKADDR_IN);
  int status;

  /* initialize the Windows Socket DLL */
  status=WSAStartup(MAKEWORD(1, 1), &Data);
  if (status != 0) {
    cerr << "ERROR: WSAStartup unsuccessful" << endl;
    return(1);
  }

  /* zero the sockaddr_in structure */
  memset(&serverSockAddr,0,sizeof(serverSockAddr));
  /* specify the port portion of the address */
  serverSockAddr.sin_port=htons(PORT);
  /* specify the address family as Internet */
  serverSockAddr.sin_family=AF_INET;
  /* specify that the address does not matter */
  serverSockAddr.sin_addr.s_addr=htonl(INADDR_ANY);

  /* create a socket */
  serverSocket=socket(AF_INET, SOCK_STREAM, 0);
  if (serverSocket == INVALID_SOCKET) {
    cerr << "ERROR: socket unsuccessful" << endl;
    status=WSACleanup();
    if (status==SOCKET_ERROR) cerr<<"ERROR: WSACleanup unsuccessful"<<endl;
    return(1);
  }

  /* associate the socket with the address */
  status=bind(serverSocket,(LPSOCKADDR)(&serverSockAddr)
             ,sizeof(serverSockAddr));
  if (status == SOCKET_ERROR) {
    cerr << "ERROR: bind unsuccessful" << endl;
    status=closesocket(clientSocket);
    if (status==SOCKET_ERROR) cerr<<"ERROR: closesocket unsuccessful"<<endl;
    return(1);
  }

  /* allow the socket to take connections */
  status=listen(serverSocket, 1);
  if (status == SOCKET_ERROR) {
    cerr << "ERROR: listen unsuccessful" << endl;
    status=closesocket(clientSocket);
    if (status==SOCKET_ERROR) cerr<<"ERROR: closesocket unsuccessful"<<endl;
    return(1);
  }

  cout << "Start accepting TCP/IP connection" << endl;

  while(1) {
    /* accept the connection request when one is received */
    clientSocket=accept(serverSocket,(LPSOCKADDR)(&clientSockAddr),&addrLen);
    if (clientSocket == INVALID_SOCKET) {
      cerr << "ERROR: Unable to accept connection." << endl;
      return(1);
    }

#ifdef FORK
    pid_t pid = fork();
    if(0==pid) {
      talkToClient((void *)clientSocket);
      exit(0);
    }
#else
    talkToClient((void *)clientSocket);
#endif

  } /* while */
}
