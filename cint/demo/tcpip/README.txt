demo/tcpip/README.txt

 This directory contains TCP/IP socket programming example using CINT.
Cint supports TCP/IP library in lib/socket directory. You need to build
include/cintsock.dll in order to run TCP/IP demo program.

FILES: 
  README.txt   : This file
  server.cxx   : TCP/IP Server program
  client.cxx   : TCP/IP Client program


RUNNING THE DEMO 1: (Single Client)

 Start server
  c:\cint\demo\tcpip>  cint server.cxx

 Start client in another terminal
  c:\cint\demo\tcpip>  cint client.cxx
  Type text to send:  abc
  Type text to send:  def
  Type text to send:  quit
  c:\cint\demo\tcpip>  cint client.cxx
  Type text to send:  hij
  Type text to send:  kill


RUNNING THE DEMO 2: (Multiple Client, UNIX only)

 One terminal
  c:\cint\demo\tcpip>  cint -DFORK server.cxx

 Another terminal
  c:\cint\demo\tcpip>  cint client.cxx
  Type text to send:  abc
  Type text to send:  def
  Type text to send:  quit
  c:\cint\demo\tcpip>  cint client.cxx
  Type text to send:  hij
  Type text to send:  klmn

 Note: If you start server using -DFORK option, sending 'kill' string does
       not kill the server. You need to hit CTL-C to stop the server.
