#!/bin/sh
 
# chkconfig: 345 99 10
# chkconfig(sun): S3 99 K0 10 
# description: start and stop the CNS xrootd.
#

# set xrdDir where xrootd bin/lib/etc directories reside
#
xrdDir="/opt/xrootd"

# set cfg to the configuration file to use
#
cfg="$xrdDir/etc/cns.cfg"

# set user to be the username underwhich xrootd/cmsd is to run
#
user='daemon'

# set binDir where the bin directory resides (the default is probably ok)
#
binDir="$xrdDir/bin/"

##################################################################

if [ `uname` = "Linux" ]
then
    SU="su $user -s /bin/sh"
else
    SU="su $user"
fi

instanceName='cns'

status() {
    me=`basename $0`
    ps -ef | grep $user | egrep 'xrootd' | grep $instanceName
    return $?
}

start() {
    if [ `uname` = "Linux" ]
    then
        echo -n "Starting xrootd/CNS: "
    else
        echo "Starting xrootd/CNS: \c"
    fi
    status > /dev/null 2>&1
    if [ $? -eq 0 ]
    then
        echo " [Fail]"
    else
#        rm /tmp/xrd.log /tmp/olb.log /tmp/olbd.pid /tmp/olbd.mangr.pid > /dev/null 2>&1
#       rm -rf /tmp/${instanceName}/.xrootd /tmp/${instanceName}/cns.log > /dev/null 2>&1
        $SU -c "$binDir/xrootd -d -n cns -k 7 -l /tmp/cns.log -c $cfg &"
        echo " [OK]"
    fi
}

stop() {
    if [ `uname` = "Linux" ]
    then
        echo -n "Stopping xrootd/CNS: "
    else
        echo "Stopping xrootd/CNS: \c"
    fi
    status | awk '{print $2}' | while read pid 
    do
        kill $pid > /dev/null 2>&1
    done
    status
    if [ $? -eq 0 ] 
    then
        echo " [Fail]"
    else
        echo " [OK]"
    fi
}

case "$1" in
start)
    start
    ;;

stop)
    stop
    ;;

restart)
    stop
    start
    ;;

status)
    status
    ;;

*)
    echo "Usage: $0 {start|stop|restart|status}"
    exit 1
esac
