#!/bin/sh -e 
#
# $Id$
#
# Make filelists files for all packages. 
#  
# We have to be a bit tricky here, since rpm will happily copy the
# same file into many sub-packages, so we have to do wildcard
# expansio, substract already installed files and so on. We employ a
# trick here that's similar to what is done in dh_movesfiles from the
# debhelper package on Debian GNU/Linux. First we pack the files into 
# tar archives, IN THE ORDER GIVEN by variable $pkgs from common file,
# taking care to remove the old files, then we list the contents of
# these tar files and make that our file list, and finally we unpack
# the tar archives again, so that rpm can find them. 
# 

if [ $# -lt 1 ] ; then 
    echo "I need the build root as argument" 
    exit 1
fi
    
. build/package/lib/common.sh rpm

### echo %%% save the build root in a variable 
blddir=$1

### echo %%% Make the directory 
mkdir -p ${tgtdir} 

### echo %%% The types of files we looking at. 
types="files conffiles docs examples shlibs" 

### echo %%% loop over the defined packages in specific order
for i in $pkgs ; do 

    ### echo %%% make sure we have a fresh file 
    rm -f ${cmndir}/tmp 

    for j in $types ; do 
	### echo %%% if ${cmndir}/${i}.${j} exists, use it
	if [ -f ${cmndir}/${i}.${j} ] ; then 
	    cat ${cmndir}/${i}.${j} >> ${cmndir}/tmp
	fi 
    done
    if [ ! -s ${cmndir}/tmp ] ; then 
	continue
    fi

    ### echo %%% do a primitive wildcard expansion on temp file list
    entries=`cat ${cmndir}/tmp`

    ### echo %%% save the original directory 
    savdir=`pwd`

    ### echo %%% make sure we have a fresh file 
    rm -f ${blddir}/tmp.${j}

    ### echo %%% cd into the subdirectory first 
    cd ${blddir}

    ### echo %%% let a for loop do wildcard expansion 
    for j in ${entries} ; do 
	echo "$j" >> tmp.${i}
    done

    ### echo %%% move the files to a temporary tar archive 
    echo -n "Creating tmp archive for $i ... "
    tar --create \
	--remove-files \
	--files-from=tmp.${i} \
	--file=${i}.tar 
    ### echo %%% clean up
    rm -f tmp.${i}

    ### echo %%% go back to top-level dir 
    cd ${savdir} 
    echo "done"

    ### echo %%% clean up
    rm -f ${cmndir}/tmp
done

### echo %%% Loop over the packages, and make the filelists
for i in $pkgs ; do 
    
    ### echo %%% if tar archive does not exit, continue to next 
    if [ ! -f ${blddir}/${i}.tar ] ; then 
	continue
    fi

    ### echo %%% create the file list 
    echo -n "Creating package list for $i ... "

    ### echo %%% first set the default attributes
    echo "%defattr(-,root,root)" > ${tgtdir}/${i}.files

    ### echo %%% and make /usr/share/doc/root a documentation dir
    echo "%docdir /usr/share/doc/root" >> ${tgtdir}/${i}.files

    ### echo %%% List the contents of the tar file, ignoring pure 
    ### echo %%% directories and marking files in /etc as
    ### echo %%% confuguration files and files in /usr/share/doc as
    ### echo %%% documentation files
    tar --list --file=${blddir}/${i}.tar | \
	sed -e 's|^\(.*\)|/\1|' \
	    -e 's|^/usr/share/doc|%doc /usr/share/doc|' \
	    -e 's|^/etc|%config /etc|' \
	    -e '\|.*/$|d' | \
	sort -u >> \
	${tgtdir}/${i}.files

    ### echo %%% unpack the tar archive so that rpm may find the files
    ### echo %%% again 	
    echo -n "extracting ... " 
    tar --extract --directory=${blddir} --file=${blddir}/${i}.tar 

    ### echo %%% clean up 
    rm -f ${blddir}/${i}.tar
    echo "done"
done 

#
# $Log$
#

