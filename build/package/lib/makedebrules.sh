#!/bin/sh 
#
# $Id$
#
# Write lines for <pkg> to debian/rules
#
. build/package/lib/common.sh debian

if [ $# -lt 1 ] ; then 
    echo "$0: I need a package name - giving up"
    exit 2
fi

# save package name in logical variable 
pkg=$1

# first split the file at mark '@pkg@'
csplit -q -f ${cmndir}/tmp. -k ${cmndir}/rules.tmp "/@pkg@/" 

# Then output the first part 
cat ${cmndir}/tmp.00 > ${cmndir}/rules.tmp

# Then output the package line 
echo -e "\t${pkg} \\" >> ${cmndir}/rules.tmp

# and finally output the final part 
cat ${cmndir}/tmp.01 >> ${cmndir}/rules.tmp

# Clean up 
rm -f ${cmndir}/tmp.00 ${cmndir}/tmp.01

#
# $Log$
#
