
#include <XrdSys/XrdSysLogger.hh>
#include <XrdSec/XrdSecTLayer.hh>
#include <XrdNet/XrdNetSocket.hh>
#include <XrdNet/XrdNetOpts.hh>
#include <XrdSecssl/XrdSecProtocolssl.hh>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/time.h>

#define TESTLOOP 100

int main(int argc, char* argv[]) {
  if (argc < 2) {
    fprintf(stderr,"Error: you have to define if we are server or client!\n");
    fprintf(stderr,"usage: xrdsslprotocoltest server|client [args]\n");
    exit(-1);
  }

  // To silence a warning
  EPNAME("main");
  PRINT("dummy");

  setenv("XrdSecDEBUG","10",1);
  XrdSysLogger logger;
  XrdSysError eDest(&logger,"ssltest");
  
  if (!strcmp("server",argv[1])) {
    // server
    const char* args="-d:10 -cadir:/etc/grid-security/certificates/";
    if (argv[2]) {
      args = argv[2];
    }

    struct sockaddr  netaddr;
    XrdOucErrInfo error;
    
    XrdSecProtocolsslInit('s',args, &error);
    XrdSecProtocolssl* protocol = new XrdSecProtocolssl("localhost",(const struct sockaddr*)&netaddr);
    if (!protocol) {
      fprintf(stderr,"Error: cannot create protocol object\n");
      exit(-1);
    }

    XrdNetSocket* socket = new XrdNetSocket(&eDest);
    socket->Open(0,12345,XRDNET_SERVER);

    while(1) {
      // do an infinite handshake loop
      int theFd = socket->Accept();
      if (theFd<=0) {
	fprintf(stderr,"Accept failed on socket!\n");
	exit(-1);
      }
      protocol->secServer(theFd, &error);
      
      fprintf(stderr,"Authentication done: [%d : %s]\n", error.getErrInfo(),error.getErrText());
      close(theFd);
    }

    exit(0);
  }  else {
    if (!strcmp("client",argv[1])) {
      // client
      struct sockaddr  netaddr;
      XrdOucErrInfo error;

      XrdSecProtocolsslInit('c',"", &error);
      XrdSecProtocolssl* protocol = new XrdSecProtocolssl("localhost",(const struct sockaddr*)&netaddr);
      if (!protocol) {
	fprintf(stderr,"Error: cannot create protocol object\n");
	exit(-1);
      }
      XrdSecProtocolssl::allowSessions = false;
      struct timeval tv1, tv2, tv3;
      struct timezone tz;

      gettimeofday(&tv1,&tz);
      for (int i=0; i< TESTLOOP; i++) {
	XrdNetSocket* socket = new XrdNetSocket(&eDest);
	
	socket->Open(0,12345);
	
	int theFd = socket->Detach();
	if (theFd<=0) {
	  fprintf(stderr,"unable to connect to socket\n");
	  fprintf(stdout,"Client aborted: unable to connect to socket\n");
	  exit(-1);
	}
	protocol->secClient(theFd, &error);
	if (error.getErrInfo()) {
	  fprintf(stderr,"Authentication done: [%d : %s]\n", error.getErrInfo(),error.getErrText());
	  fprintf(stdout,"Client aborted: authentication failure: [%d : %s]\n", error.getErrInfo(),error.getErrText());
	  exit(-1);
	}

	delete socket;
      }
      gettimeofday(&tv2,&tz);
      XrdSecProtocolssl::allowSessions = true;
      for (int i=0; i< TESTLOOP; i++) {
	XrdNetSocket* socket = new XrdNetSocket(&eDest);
	
	socket->Open(0,12345);
	
	int theFd = socket->Detach();
	
	protocol->secClient(theFd, &error);
	if (error.getErrInfo()) {
	  fprintf(stderr,"Authentication done: [%d : %s]\n", error.getErrInfo(),error.getErrText());
	  exit(-1);
	}
	delete socket;
      }
      gettimeofday(&tv3,&tz);

      float inta = (((tv2.tv_sec-tv1.tv_sec) * 1000) + (tv2.tv_usec-tv1.tv_usec)/1000.0)/1000.0;
      float intb = (((tv3.tv_sec-tv2.tv_sec) * 1000) + (tv3.tv_usec-tv2.tv_usec)/1000.0)/1000.0;
      fprintf(stdout,"-----------------------------------------------------------------\n");
      fprintf(stdout,"Tested %d iterations without and with sessions...\n",TESTLOOP);
      fprintf(stdout,"-----------------------------------------------------------------\n");
      fprintf(stdout,"Performance without Sessions: %.02f authentications/s\n",TESTLOOP/inta);
      fprintf(stdout,"Performance with    Sessions: %.02f authentications/s\n",TESTLOOP/intb);
      fprintf(stdout,"-----------------------------------------------------------------\n");
      exit(0);
    }
  }
  fprintf(stderr,"Error: you have to define if we are server or client!\n");
  fprintf(stderr,"usage: xrdsslprotocoltest server|client\n");
  exit(-1);
}



