# @(#)root/mpi:$Id$
# Author: Omar Zapata (Omar.Zapata@cern.ch) 2017


__version__ = '0.0.1'
__author__  = 'Omar Zapata (Omar.Zapata@cern.ch)'

from ROOT import Mpi, TString
from ROOT.Mpi import COMM_WORLD, TEnvironment, TCommunicator, TRequest, TMpiMessage
from ROOT.Mpi import PyPickleDumps as __PyPickleDumps
from ROOT.Mpi import PyPickleLoads as __PyPickleLoads

#########################
##TMpiMessage attributes#
#########################
def __WritePyObject(self,obj):
    msg=__PyPickleDumps(obj)
    self.WriteTString(msg)
    self.SetReadMode();
    self.Reset();
    
def __ReadPyObject(self):
    msg=TString()
    self.ReadTString(msg)
    return __PyPickleLoads(msg);

setattr(TMpiMessage,"WritePyObject",__WritePyObject)
setattr(TMpiMessage,"ReadPyObject",__ReadPyObject)

######################
##TRequest attributes#
######################

def __SetMsg(self,msg):
    self.fPyMsg=msg

def __GetMsg(self):
    return self.fPyMsg.ReadPyObject()

setattr(TRequest,"SetMsg",__SetMsg)
setattr(TRequest,"GetMsg",__GetMsg)

###########################
##TCommunicator attributes#
###########################

setattr(TCommunicator,"__PySend",TCommunicator.Send)
setattr(TCommunicator,"__PyRecv",TCommunicator.Recv)

def __Send(self,obj,dest,tag):
        msg=TMpiMessage()
        msg.WritePyObject(obj)
        self.__PySend("TMpiMessage")(msg,dest,tag)
        
def __Recv(self,dest,tag):
        msg=TMpiMessage()
        self.__PyRecv("TMpiMessage")(msg,dest,tag)
        return msg.ReadPyObject()

setattr(TCommunicator,"Send",__Send)
setattr(TCommunicator,"Recv",__Recv)

###########################
##TCommunicator attributes#
###########################
def __Scatter(self,in_vars,incount, outcount,root):
    if self.GetRank() == root:
        osize = self.GetSize() * outcount
        if incount % osize != 0 :
            self.Fatal("TCommunicator::Scatter","Number of elements sent and elements in receive are not divisible. Can't no split to scatter message")
            self.Abort(Mpi.ERR_COUNT)
        for i in range(0,self.GetSize(),1):
            if i == root : continue
            stride = outcount * i
            self.Send(in_vars[stride:stride+outcount], i, self.GetMaxTag()+1)

        stride = outcount * root
        return in_vars[stride:stride+outcount]
    else:
        return self.Recv(root, self.GetMaxTag()+1)

setattr(TCommunicator,"Scatter",__Scatter)


setattr(TCommunicator,"__PyBcast",TCommunicator.Bcast)

def __Bcast(self,obj, root):
    msg=TMpiMessage()
    if self.GetRank() == root:
       msg.WritePyObject(obj)
       
    self.__PyBcast("TMpiMessage")(msg, root)
    return msg.ReadPyObject()

setattr(TCommunicator,"Bcast",__Bcast)


setattr(TCommunicator,"__PyISend",TCommunicator.ISend)
setattr(TCommunicator,"__PyIRecv",TCommunicator.IRecv)


        
def __ISend(self,obj, dest, tag):
    msg=TMpiMessage()
    msg.WritePyObject(obj)
    return self.__PyISend("TMpiMessage")(msg, dest, tag)

def __IRecv(self,source, tag):
    msg = TMpiMessage();
    req = self.__PyIRecv("TMpiMessage")(msg, source, tag)
    req.SetMsg(msg)
    return req;

setattr(TCommunicator,"ISend",__ISend)
setattr(TCommunicator,"IRecv",__IRecv)
