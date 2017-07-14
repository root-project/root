# @(#)root/mpi:$Id$
# Author: Omar Zapata (Omar.Zapata@cern.ch) 2017


__version__ = '0.0.1'
__author__  = 'Omar Zapata (Omar.Zapata@cern.ch)'

from ROOT import Mpi, TString
from ROOT.Mpi import COMM_WORLD, TEnvironment, TPort, TMpiMessage, TStatus
from ROOT.Mpi import TCommunicator, TInterCommunicator, TIntraCommunicator, TNullCommunicator 
from ROOT.Mpi import TRequest, TGrequest, TPrequest, TInfo, TMpiFile, TOp
from ROOT.Mpi import TMpiTimer, TGroup, TErrorHandler
import pickle, codecs
#########################
##TMpiMessage attributes#
#########################
def __WritePyObject(self,obj):
    """Method to serialize python objects"""
    data=codecs.encode(pickle.dumps(obj), "base64").decode()
    msg=TString(data)
    self.WriteTString(msg)
    self.SetReadMode();
    self.Reset();
    
def __ReadPyObject(self):
    """Method to unserialize python objects"""
    msg=TString()
    self.ReadTString(msg)
    return pickle.loads(codecs.decode(msg.Data().encode(), "base64"))

setattr(TMpiMessage,"WritePyObject",__WritePyObject)
setattr(TMpiMessage,"ReadPyObject",__ReadPyObject)

######################
##TRequest attributes#
######################

def __SetMsg(self,msg):
    """Method to set message reference for non-blocking communication"""
    self.fPyMsg=msg

def __GetMsg(self):
    """Method to get message for non-bloking communication"""
    return self.fPyMsg.ReadPyObject()

setattr(TRequest,"SetMsg",__SetMsg)
setattr(TRequest,"GetMsg",__GetMsg)

###########################
##TCommunicator attributes#
###########################

setattr(TCommunicator,"__PySend",TCommunicator.Send)
setattr(TCommunicator,"__PyRecv",TCommunicator.Recv)

def __Send(self,obj,dest,tag):
    """ Method to send a message for p2p communication
    @param var any selializable object
    @param dest id with the destination(Rank/Process) of the message
    @param tag id of the message
    """
        msg=TMpiMessage()
        msg.WritePyObject(obj)
        self.__PySend("TMpiMessage")(msg,dest,tag)
        
def __Recv(self,dest,tag):
    """ Method to receive a message for p2p communication
    @param var any selializable object reference to receive the message
    @param source id with the origin(Rank/Process) of the message
    @param tag id of the message
    """
        msg=TMpiMessage()
        self.__PyRecv("TMpiMessage")(msg,dest,tag)
        return msg.ReadPyObject()

setattr(TCommunicator,"Send",__Send)
setattr(TCommunicator,"Recv",__Recv)

###########################
##TCommunicator attributes#
###########################
def __Scatter(self,in_vars,incount, outcount,root):
    """ """
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


#non-bloking p2p        
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

def __Gather(self,obj, incount, outcount, root):
    """ Each process (root process included) sends the contents of its send buffer to the root process.
    The root process receives the messages and stores them in rank order.
    The outcome is as if each of the n processes in the group (including the root process)
    @param in_vars any selializable object vector reference to send the message
    @param incount Number of elements in receive in \p in_vars
    @param outcount Number of elements in receive in \p out_vars
    @param root id of the main message where message was sent
    @return object type list with the message
    """
    if self.GetRank() == root :
        osize=self.GetSize() * incount
        if osize % outcount != 0 :
            self.Fatal("TCommunicator::Gather", "Number of elements sent can't be fitted in gather message")
            self.Abort(Mpi.ERR_COUNT)
        output=obj[0:incount]    
        for i in range(0,self.GetSize(),1):
            if i == root: continue
            msg=self.Recv( i, self.GetMaxTag()+1)
            output.extend(msg)
        return output    
    else:
      self.Send(obj[0:incount], root, self.GetMaxTag()+1)

setattr(TCommunicator,"Gather",__Gather)
