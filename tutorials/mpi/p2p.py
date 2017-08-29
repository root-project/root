## \file
## \ingroup tutorial_mpi
##
##  Peer to peer communication is the most basic communication operation, basically is send and receiv a message between two processes.
##
## ~~~{.cpp}
##  rootmpi -np 2 hello.py
## ~~~
##
##
## \macro_output
## \macro_code
##
## \author Omar Zapata

from ROOT import Mpi, TMatrixD
from ROOT.Mpi import TEnvironment, COMM_WORLD

def p2p():
   env=TEnvironment() # environment to start communication system

   if COMM_WORLD.GetSize() != 2 :   return # need 2 process


   # sending messages in process 0
   if COMM_WORLD.GetRank() == 0:
      # data to send
      mydict={"key":"hola"}                    # dict object
      mymat=TMatrixD (2, 2)                    # ROOT object
      a=0.0                                    # default datatype
      
      mymat[0][0] = 0.1
      mymat[0][1] = 0.2
      mymat[1][0] = 0.3
      mymat[1][1] = 0.4

      a = 123.0

      print("Sending scalar = %f"% a )
      COMM_WORLD.Send(a, 1, 0)
      print("Sending dict = %s"%mydict["key"] )
      COMM_WORLD.Send(mydict, 1, 0)
      print("Sending mat = ")
      mymat.Print()
      COMM_WORLD.Send(mymat, 1, 0)
      
   # Receiving messages in process 1
   if COMM_WORLD.GetRank() == 1 :
      scalar=COMM_WORLD.Recv( 0, 0)
      print("Recieved scalar = %f"%scalar)
      mydict=COMM_WORLD.Recv( 0, 0)
      print("Received map = %s"%mydict["key"])
      mymat=COMM_WORLD.Recv( 0, 0)
      print("Received mat = ")
      mymat.Print()

if __name__ == "__main__":
    p2p()
