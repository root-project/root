#!/bin/sh -e 
#
# $Id$
#
# Make the rpm spec file in ../root.spec
#
. build/package/lib/common.sh rpm 

### echo %%% Make the directory 
# mkdir -p ${tgtdir} 

### echo %%% Copy the README file to the directory 
# cp ${cmndir}/README ${tgtdir}/README.Redhat

### echo %%% Copy the header of the spec file to rpm/root.spec
if [ ! -f ${rpmdir}/head.spec.in ] ; then 
    echo "$0: Couldn't find ${rpmdir}/head.spec.in - very bad" 
    echo "Giving up. Something is very screwy"
    exit 10
fi
### echo %%% First thing to do, is to write version number. 
sed "s/@version@/${versi}/" < ${rpmdir}/head.spec.in > ${updir}/root.spec
echo "" >> ${updir}/root.spec

### echo %%% Make the sub-package stuff
for i in ${pkgs} ; do 
    echo "Processing for package $i ... "

    ### echo %%% first insert the missing peices
    ${libdir}/makerpmspecs.sh $i 
done 

### echo %%% finally cat the footer to the spec file 
echo "" >> ${updir}/root.spec

### echo %%% Copy the header of the spec file to rpm/root.spec
if [ ! -f ${rpmdir}/tail.spec.in ] ; then 
    echo "$0: Couldn't find ${rpmdir}/tail.spec.in - very bad" 
    echo "Giving up. Something is very screwy"
    exit 10
fi

### echo %%% Insert the configuration command 
### echo %%% first split file
csplit -q -f ${cmndir}/tmp. -k  ${rpmdir}/tail.spec.in "/@configure@/"

### echo %%% Cat the first part 
cat ${cmndir}/tmp.00 >> ${updir}/${base}.spec

### echo %%% now the configuration command 
${libdir}/makeconfigure.sh rpm >> ${updir}/${base}.spec

### echo %%% and finally the last part 
sed -e '/@configure@/d' \
    -e "s|@libdir@|${libdir}|" \
    < ${cmndir}/tmp.01 >> ${updir}/${base}.spec

### echo %%% clean up 
rm -f ${cmndir}/tmp.00 ${cmndir}/tmp.01

#
# $Log$
#
