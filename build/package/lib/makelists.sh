#!/bin/sh

outdir=$1  	; shift
build=$1        ; shift
version=$1 	; shift
prefix=$1  	; shift
sysconfdir=$1	; shift
pkgdocdir=$1	; shift
sovers=`echo $version | sed 's/\([[:digit:]]*\.[[:digit:]]*\)\..*/\1/'`
rm -f $outdir/*.install

# install file lists that need no substitutions 
for i in build/package/common/*.install ; do 
    if test ! -f $i ; then continue ; fi 
    cp $i $outdir
done

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
	auth)       lib=libroot             ; dev=libroot-dev; 
	            bin=root-system-bin     ;;
	base)       lib=libroot             ; dev=libroot-dev; 
	            bin=root-system-bin
	            extra="ALLMAPS=${prefix}/lib/root/libCore.rootmap "
                    extra="$extra ALLLIBS=${prefix}/lib/root/libCore.so" ;; 
	cint)	    lib=libroot             ; dev=libroot-dev; 
	            bin=root-system-bin     ;
	    	    extra="ALLLIBS=${prefix}/lib/root/libCint.so" ;;   
	    	    # extra="NOMAP=1 ALLLIBS=${prefix}/lib/root/libCint.so" ;; 
        clib|cont|eg|foam|fitpanel|g3d|gdml|ged*|geom*|gpad|graf|gui*|hist*)
	            lib=libroot             ; dev=libroot-dev; 
		    bin=root-system-bin ;;
        html|io|math|mathcore|matrix|meta*|net|physics|postscript|rint)
 	            lib=libroot             ; dev=libroot-dev; 
		    bin=root-system-bin ;;
        spectrum*|table|thread|tree*|unix|utils|vmc|x11*|x3d|zip)
	            lib=libroot             ; dev=libroot-dev; 
		    bin=root-system-bin ;;
	rpdutils)   lib=libroot             ; dev=libroot-dev; 
		    bin=root-system-bin     ;; # extra="NOMAP=1"  ;;
        rootx|sessionviewer|smatrix|splot|unuran|xml)    
	            lib=libroot             ; dev=libroot-dev; 
		    bin=root-system-bin ;;
	newdelete)  lib=libroot		    ; dev=libroot-dev; 
	    	    bin=libroot-dev         ;; # extra="NOMAP=1" ;;
	reflex)     lib=libroot		    ; dev=libroot-dev; 
	    	    bin=libroot-dev         ;; # extra="NOMAP=1" ;;
	cintex)     lib=libroot		    ; dev=libroot-dev; 
	    	    bin=libroot-dev         ;;
	globusauth) lib=root-plugin-globus  ; dev=$lib       ; bin=$lib ;;  
	qtroot)     lib=root-plugin-qt      ; dev=$lib       ; bin=$lib ;;
	pythia)     lib=root-plugin-pythia5 ; dev=$lib       ; bin=$lib ;;  
	rfio)       lib=root-plugin-castor  ; dev=$lib       ; bin=$lib ;;  
	srputils)   lib=root-plugin-srp     ; dev=$lib       ; bin=$lib ;;  
	xmlparser)  lib=root-plugin-xml     ; dev=$lib       ; bin=$lib ;;
	krb5auth)   lib=root-plugin-krb5    ; dev=$lib       ; bin=$lib ;;
	proofd)	    lib=root-plugin-xproof  ; dev=$lib       ; 
	            bin=root-system-proofd  ;; 
	proofplayer)	    
	            lib=root-plugin-proof   ; dev=$lib       ; bin=$lib ;; 
	rootd)      lib=root-system-$d      ; dev=$lib       ; bin=$lib ;;
	xrootd)     lib=root-system-$d      ; dev=$lib       ; bin=$lib ;
	            xrdlibs=                ; extra="ALLLIBS= NOVERS=1" ;; 
                    # NOMAP=1" ;;     
	pyroot)     lib=libroot-python      ; dev=${lib}-dev ; bin=$lib ;;  
	clarens|ldap|mlp|quadp|roofit|ruby|mathmore|minuit|tmva)
	            lib=libroot-$d          ; dev=${lib}-dev ; bin=$lib ;;  
	build|freetype|win*|main) continue ;; 			
	proofx)     lib=root-plugin-xproof  ; dev=$lib       ; bin=$lib ;;  
	sapdb)      lib=root-plugin-maxdb   ; dev=$lib       ; bin=$lib ;;  
	qtgsi)      lib=root-plugin-qt      ; dev=$lib       ; bin=$lib ;;  
	fftw)       lib=root-plugin-${d}3   ; dev=$lib       ; bin=$lib ;;
	            # extra="NOMAP=1"         ;;  
	*)          lib=root-plugin-$d      ; dev=$lib       ; bin=$lib ;;  
    esac 

    # 
    # Update package list for based on the Module.mk in thie currenly
    # investiaged directory 
    #
    # echo "Making list for $d (dev=$dev lib=$lib bin=$bin extra=$extra)"
    # echo "Making list for $d"
    build/package/lib/makelist DIRS=$d DEV=$dev LIB=$lib BIN=$bin  \
	VERSION=$version PREFIX=$prefix OUT=$outdir BUILD=$build $extra  \
	--no-print-directory all
done

#
# For each skeleton file, replace occurances of @prefix@,
# @sysconfdir@, and @pkgdocdir@ with the appropriate values 
#
for i in build/package/common/*.install.in ; do 
    if test ! -f $i ; then continue ; fi
    b=`basename $i .install.in`
    case $b in 
	lib*-dev) b=$outdir/${b}          ;; 
	lib*)     b=$outdir/${b}${sovers} ;;
	*)        b=$outdir/${b}          ;; 
    esac
    grep -v "^#" $i | 					\
	sed -e "s|@prefix@|${prefix}|g" 		\
	    -e "s|@sysconfdir@|${sysconfdir}|g"		\
	    -e "s|@pkgdocdir@|${pkgdocdir}|g"		\
	    -e "s|@version@|${sovers}|g"		\
	> ${b}.tmp
    if test -f ${b}.install ; then 
	cat ${b}.tmp  ${b}.install > ${b}.tmp2
	mv  ${b}.tmp2 ${b}.install 
    else
	cp  ${b}.tmp ${b}.install
    fi
    rm -f ${b}.tmp ${b}.tmp2 
done 

for i in $outdir/*.install ; do 
    if test ! -f $i ; then continue ; fi
    sort -u ${i} > ${i}.tmp 
    mv ${i}.tmp ${i}
done
	

#
# EOF
#
