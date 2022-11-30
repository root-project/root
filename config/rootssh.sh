#!/bin/bash

# Invoke ssh and automatically configures port forwarding for remote ROOT session
# To start, just call:
#
#    [localhost] rootssh.sh  user@remotehost
#
# And then in the ssh shell do:
#
#    [user@remotehost] source path/to/bin/thisroot.sh
#    [user@remotehost] root --web=server  -e "new TBrowser"
#
# ROOT automatically recognizes that it runs in special session (because of ROOT_LISTENERSOCKET var)
# and provides to this socket information to configure port forwarding and to start web windows automatically

if [[ "$1" == "--as-listener--" ]] ; then

   listener_socket=$2

   sshfile=$3

   probe="probing"
   localport=7777
   while [[ "x$probe" != "x" ]] ; do
      localport=$((7000 + $RANDOM%1000))
      probe=$(netcat -zv localhost $localport 2>/dev/null)
   done

   echo "Start listening for socket $listener_socket with ssh file $sshfile, will use $localport"

   flag=1

   while [ $flag -ne 0 ] ; do

      line="$(netcat -l -U $listener_socket)"

      if [[ "${line:0:5}" == "http:" ]] ; then
         remoteport=${line:5}
         echo "Want to map remote port $localport:localhost:$remoteport"
         ssh -S $sshfile -O forward -R $localport:localhost:$remoteport _dummy_arg_ "exit"
      elif [[ "${line:0:4}" == "win:" ]] ; then
         winurl=${line:4}
         echo "Want to show window http://localhost:$localport/$winurl"
         xdg-open "http://localhost:$localport/$winurl"
      elif [[ "$line" == "stop" ]] ; then
         flag=0
      else
         echo "Command not recognized $line - stop"
         flag=0
      fi
   done

   echo "Stop listener"

   exit 0
fi


listener_file="$(mktemp /tmp/root.XXXXXXXXX)"
remote_file="$(mktemp /tmp/root.XXXXXXXXX)"
ssh_file="$(mktemp ~/.ssh/root.XXXXXXXXX)"

rm -f $ssh_file $listener_file $remote_file

# start listener process

$0 --as-listener-- $listener_file $ssh_file &

listener_id=$!

cmd="ROOT_LISTENERSOCKET=$remote_file"
cmd+=" $"
cmd+="SHELL; rm -f $remote_file"

echo "remote socket is $remote_file"

# start ssh

ssh -t -M -S $ssh_file -R $remote_file:$listener_file $1 "$cmd"

echo "stop" | netcat -U $listener_file -q 1

echo "Kill listener process $listener_id"
kill -9 $listener_id > /dev/null 2>&1

echo "Remove temporary files"
rm -f $ssh_file $listener_file $remote_file
