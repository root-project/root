#!/bin/bash
#
# Circle files and commands over a cluster using scp and ssh.
#
# Author: Fons Rademakers

# On which platform do we run
ostype=`uname -s`

cwhich="type -path"
lsopt="--color=never"
ptimeout="-W"
if [ "$ostype" = "Darwin" ]; then
   lsopt=""
   ptimeout="-t"
fi

# Where to store values from a previous run
saverc="$HOME/.circlerc"

#--- stored in $saverc
nodes=
base=
domain=
lower=0
upper=0
width=1
ping=1
puser=
sshopt="-o StrictHostKeyChecking=no -o PasswordAuthentication=no"
#--- stored in $saverc

prog=`basename $0`
progdir=`dirname $0`
runningdir=`pwd`
if echo $progdir | grep -q -s ^/ || echo $progdir | grep -q -s ^~ ; then
   # absolute path
   fullpath=$progdir
else
   # relative path
   fullpath=$runningdir/$progdir
fi

# Set default arguments values
oper=
user=
nodelist=
ignore="no"
cluster=
ssho=
sshcmd=
scp1=
scp2=
pkey="$HOME/.ssh/id_dsa.pub"

# Process command line arguments
while [ $# -gt 0 ]; do
   case "$1" in
   -i|--ignore)
      ignore="yes"
      shift
      ;;
   -u|--user)
      shift
      user="$1"
      shift
      ;;
   -c|--cluster)
      shift
      cluster="$1"
      saverc="${saverc}-${cluster}"
      shift
      ;;
   -s|-sshopt)
      shift
      ssho="$1"
      shift
      ;;
   -n|--nodes)
      shift
      nodelist="$1"
      shift
      ;;
   -h|--help)
      echo "Usage: $prog ssh|scp|key [-u user] [-n \"nodes\"] [-c cluster] [-s \"sshopt\"] [-i] [-h] [ARGS...] "
      echo "Circle files and commands over a cluster using scp and ssh."
      echo ""
      echo "  -u, --user     specify the user on cluster"
      echo "  -n, --nodes    nodes on which to run or copy"
      echo "  -c, --cluster  cluster name, used to identify config file $saverc"
      echo "  -s, --sshopt   ssh options string"
      echo "  -i, --ignore   ignore an existing $saverc file"
      echo "  -h, --help     display this help and exit"
      echo ""
      echo "  ssh            use ssh to execute commands on all cluster nades,"
      echo "                 commands are specified as a single string in \"ARGS\"."
      echo "  scp            use scp to copy files on all cluster nodes,"
      echo "                 the \"what\" to \"where\" are specified as two ARGS."
      echo "  key            copy the users $HOME/.ssh/id_dsa.pub to all cluster"
      echo "                 nodes, no ARGS."
      echo "ARGS, depending on the command to execute (ssh, scp, key)."
      exit 0
      ;;
   -*)
      echo "$prog: invalid option -- $1"
      echo "Try \`$prog --help' for more information."
      exit 1
      ;;
   ssh)
      oper="$1"
      shift
      ;;
   scp)
      oper="$1"
      shift
      ;;
   key)
      oper="$1"
      if [ ! -r $pkey ]; then
         echo "$prog: no public key $pkey to copy around"
         exit 1
      fi
      shift
      ;;
   *)
      if [ "x$oper" = "xssh" ] && [ "x$sshcmd" = "x" ]; then
         sshcmd="$1"
         shift
      elif [ "x$oper" = "xscp" ] && [ "x$scp1" = "x"] && [ "x$scp2" = "x" ]; then
         scp1="$1"
         shift
         scp2="$1"
         shift
      else
         echo "$prog: unexpected argument -- $1"
         echo "Try \`$prog --help' for more information."
         exit 1
      fi
      ;;
   esac
done

if [ -r $saverc ] && [ "$ignore" = "no" ]; then
   source $saverc
fi

if [ "x$nodelist" != "x" ]; then
   nodes="$nodelist"
fi

if [ "x$user" != "x" ]; then
   puser="$user"
fi

if [ "x$ssho" != "x" ]; then
   sshopt="$ssho"
fi

# Store settings in $saverc
echo "nodes=\"$nodes\""    > $saverc
echo "base=\"$base\""     >> $saverc
echo "domain=\"$domain\"" >> $saverc
echo "lower=\"$lower\""   >> $saverc
echo "upper=\"$upper\""   >> $saverc
echo "width=\"$width\""   >> $saverc
echo "ping=\"$ping\""     >> $saverc
echo "puser=\"$puser\""   >> $saverc
echo "sshopt=\"$sshopt\"" >> $saverc


nodelist() {
   # This function will create a list of node names using a base
   # node name [$1] and a starting [$2] and ending [$3] sequence number.
   # The zero padded width is specified in [$4].
   # To exclude nodes that don't report to ping set [$5] to 1.
   # The result will be stored in nodes variable.

   if test $# -lt 6; then
      echo "$prog nodes: Too few arguments"
      return 1
   fi

   # Save arguments in local names
   nbase=$1;   shift
   ndomain=$1; shift
   nfirst=$1;  shift
   nlast=$1;   shift
   nwidth=$1;  shift
   nping=$1;   shift

   form="%0${nwidth}d"

   i=$nfirst
   nodes=
   while test $i -le $nlast; do
      ii=`printf $form $i`
      if [ "$ndomain" = "-" ]; then
         node=$nbase$ii
      else
         node=$nbase$ii.$ndomain
      fi
      if [ "x$nping" = "x1" ]; then
         p=`ping -c 1 $ptimeout 1 $node > /dev/null 2>&1`
         if [ $? -ne 0 ]; then
            node=
         fi
      fi
      if [ "x$node" != "x" ]; then
         if [ "x$nodes" != "x" ]; then
            nodes="$nodes "
         fi
         nodes=$nodes$node
      fi
      i=$((i+1))
   done
}

if [ "x$nodes" = "x" ]; then
   nodelist $base $domain $lower $upper $width $ping && exit 1
fi

case $oper in
ssh)
   for i in $nodes; do
      echo "ssh $sshopt $puser@$i $sshcmd"
      ssh $sshopt $puser@$i "$sshcmd"
   done
   ;;
scp)
   for i in $nodes; do
      echo "scp $sshopt $scp1 $puser@$i:$scp2"
      scp $sshopt $scp1 $puser@$i:$scp2
   done
   ;;
key)
   for i in $nodes; do
      echo "add $pkey to authorized_keys on $i"
      cat $pkey | ssh $sshopt $puser@$i "cat - >> ~/.ssh/authorized_keys"
   done
   ;;
esac
