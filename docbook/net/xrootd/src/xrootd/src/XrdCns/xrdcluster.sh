#!/bin/sh
 
# chkconfig: 345 99 10
# chkconfig(sun): S3 99 K0 10 
# description: start and stop xrootd and oldb servers.
#

# set xrdDir where xrootd bin/lib/etc directories reside
#
xrdDir="/opt/xrootd"

# set cfg to the configuration file to use
#
cfg="$xrdDir/etc/xrdcluster.cfg"

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

status() {
    me=`basename $0`
    ps -U $user | egrep 'xrootd|cmsd' | grep -v grep 
    return $?
}

start() {
    if [ `uname` = "Linux" ]
    then
        echo -n "Starting xrootd and cmsd: "
    else
        echo "Starting xrootd and cmsd: \c"
    fi
    status > /dev/null 2>&1
    if [ $? -eq 0 ]
    then
        echo " [Fail]"
    else
#       rm /tmp/xrd.log /tmp/cms.log /tmp/cmsd.pid /tmp/cmsd.mangr.pid /tmp/cnsd.log > /dev/null 2>&1
#       rm -rf /tmp/.xrootd /tmp/.xrd /tmp/.olb > /dev/null 2>&1
        $SU -c "$binDir/xrootd -d -k 7 -l /tmp/xrd.log -c $cfg &"
        $SU -c "$binDir/cmsd   -d -k 7 -l /tmp/cms.log -c $cfg &"
        echo " [OK]"
    fi
}

stop() {
    if [ `uname` = "Linux" ]
    then
        echo -n "Stopping xrootd and cmsd: "
    else
        echo "Stopping xrootd and cmsd: \c"
    fi
    status | awk '{print $1}' | while read pid 
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
