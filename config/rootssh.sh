#!/bin/bash

# Invoke ssh and automatically configures tunel for remote ROOT session
# To start, just call:
#
#    [localhost] rootssh.sh  user@remotehost
#
# And then in the ssh shell do:
#
#    [user@remotehost] source path/to/bin/thisroot.sh
#    [user@remotehost] root --web=server  -e "new TBrowser"
#
# ROOT automatically recognizes that it runs in special session (because of ROOT_LISTENER_SOCKET variable)
# and provides to this socket information to configure port forwarding and to start web windows automatically

if [[ "$1" == "--as-listener--" ]] ; then

   listener_socket=$2

   local_port=$3

   flag=1

   while [ $flag -ne 0 ] ; do

      line="$(netcat -l -U $listener_socket)"

      if [[ "${line:0:5}" == "http:" ]] ; then
         remoteport=${line:5}
#         echo "Want to map remote port $localport:localhost:$remoteport"
#         ssh -S $sshfile -O forward -R $localport:localhost:$remoteport _dummy_arg_ "exit"
      elif [[ "${line:0:7}" == "socket:" ]] ; then
         remotesocket=${line:7}
#         echo "Remote socket was created $remotesocket"
      elif [[ "${line:0:4}" == "win:" ]] ; then
         winurl=${line:4}
#         echo "Want to show window http://localhost:$local_port/$winurl"
         xdg-open "http://localhost:$local_port/$winurl"
      elif [[ "$line" == "stop" ]] ; then
         flag=0
      else
         echo "Command not recognized $line - stop"
         flag=0
      fi
   done

#  "Exit listener"

   exit 0
fi

listener_local="$(mktemp /tmp/root.XXXXXXXXX)"
listener_remote="$(mktemp /tmp/root.XXXXXXXXX)"
root_socket="$(mktemp /tmp/root.XXXXXXXXX)"

rm -f $listener_local $listener_remote

probe="probing"
localport=7777
while [[ "x$probe" != "x" ]] ; do
   localport=$((7000 + $RANDOM%1000))
   probe=$(netcat -zv localhost $localport 2>/dev/null)
done

echo "Use local port $localport for root_socket $root_socket redirection"

# start listener process

$0 --as-listener-- $listener_local $localport &

listener_id=$!

# start ssh

ssh -t -R $listener_remote:$listener_local -L $localport:$root_socket $1 $2 $3 $4 $5 \
"ROOT_LISTENER_SOCKET=$listener_remote ROOT_WEBGUI_SOCKET=$root_socket \$SHELL; rm -f $listener_remote $root_socket"


# try to stop listener with "stop" message

echo "stop" | netcat -U $listener_local -q 1

# Kill listener process $listener_id

kill -9 $listener_id > /dev/null 2>&1

# Remove temporary files

rm -f $listener_local $listener_remote
