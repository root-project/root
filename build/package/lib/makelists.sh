#!/bin/sh

outdir=$1  	; shift
version=$1 	; shift
prefix=$1  	; shift
sysconfdir=$1	; shift
pkgdocdir=$1	; shift

rm -f $outdir/*.install

#
# Loop over the directories, and update the file lists based on the 
# information in Module.mk files in each subdirectory
#
for d in * ; do 
    # 
    # If there's no Module.mk file in the currently inspected
    # directory, continue  
    # 
    if test ! -d $d || test ! -f $d/Module.mk ; then continue ; fi 
    
    # 
    # Reset variables 
    # 
    pkg= 
    lib= 
    bin= 
    extra= 
    
    #
    # Deal with some special directories.  For each directory, check
    # if it's libraries and such should go into some special package. 
    # 
    case $d in 							
	base)       lib=libroot             ; dev=libroot-dev; bin=root-bin
	            extra="ALLLIBS=/usr/lib/root/libCore.so" ;;  	
	clib|cont|eg|g3d|ged*|geom*|gpad|graf|gui*|hist*|html|matrix)
	            lib=libroot             ; dev=libroot-dev; bin=root-bin ;;
	meta*|net|newdelete|physics|postscript|rint|table|thread|tree*) 
	            lib=libroot             ; dev=libroot-dev; bin=root-bin ;;
	unix|utils|vmc|x11*|x3d|zip|rpdutils|rootx|xml) 	
	            lib=libroot             ; dev=libroot-dev; bin=root-bin ;;
	globusauth) lib=root-plugin-globus  ; dev=$lib       ; bin=$lib ;;  
	qtroot)     lib=root-plugin-qt      ; dev=$lib       ; bin=$lib ;;
	pythia)     lib=root-plugin-pythia5 ; dev=$lib       ; bin=$lib ;;  
	pyroot)     lib=root-plugin-python  ; dev=$lib       ; bin=$lib ;;  
	rfio)       lib=root-plugin-castor  ; dev=$lib       ; bin=$lib ;;  
	cint)       lib=root-cint; dev=$lib ; bin=$lib 
	    extra="ALLLIBS=/usr/lib/root/libCint.so" ;;  	
	srputils)   lib=root-plugin-srp     ; dev=$lib       ; bin=$lib ;;  
	xmlparser)  lib=root-plugin-xml     ; dev=$lib       ; bin=$lib ;;
	krb5auth)   lib=root-plugin-krb5    ; dev=$lib       ; bin=$lib ;;
	rootd|proofd|xrootd) 
	    	    lib=root-$d             ; dev=$lib       ; bin=$lib ;;
	build|freetype|win*|main) continue ;; 			
	*)          lib=root-plugin-$d      ; dev=$lib       ; bin=$lib ;;  
    esac 

    # 
    # Update package list for based on the Module.mk in thie currenly
    # investiaged directory 
    #
    build/package/lib/makelist DIRS=$d DEV=$dev LIB=$lib BIN=$bin  \
	VERSION=$version PREFIX=$prefix OUT=$outdir $extra  \
	--no-print-directory all
done

#
# For each skeleton file, replace occurances of @prefix@,
# @sysconfdir@, and @pkgdocdir@ with the appropriate values 
#
for i in build/package/common/*.install.in ; do 
    if test ! -f $i ; then continue ; fi

    b=$outdir/`basename $i .install.in`
    sed -e "s|@prefix@|${prefix}|g" 		\
	-e "s|@sysconfdir@|${sysconfdir}|g"	\
	-e "s|@pkgdocdir@|${pkgdocdir}|g"	\
	< $i > $b.tmp
    if test -f $b.install ; then 
	cat $b.tmp $b.install > $b.tmp2
	mv  $b.tmp2 $b.install 
    else
	cp  $b.tmp $b.install
    fi
    rm -f $b.tmp $b.tmp2 
done 

for i in $outdir/*.install ; do 
    if test ! -f $i ; then continue ; fi
    sort -u $i > $i.tmp 
    mv $i.tmp $i
done
	

#
# EOF
#
