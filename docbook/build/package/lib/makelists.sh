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

set_lib_names()
{
    base=$1 ; shift
    sub=$1 ; shift
    
    lib=libroot-${base}
    if test "x$sub" != "x" ; then lib=${lib}-${sub} ; fi
    dev=${lib}-dev
    if test "x$1" != "x" ; then 
	bin=$1
    else
	bin=${lib}
    fi
}

set_plugin_names()
{
    base=$1 ; shift
    sub=$1 ; shift 
    lib=root-plugin-${base}-${sub}
    dev=${lib}
    if test "x$1" != "x" ; then 
	bin=$1
    else
	bin=${lib}
    fi
}
   
    
#
# Loop over the directories, and update the file lists based on the 
# information in Module.mk files in each subdirectory
#
l=`find . -name "Module.mk" -print0 | xargs -L 1 -0 dirname | sort -u | sed 's,./,,'`
for d in $l ; do 
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
    base=`dirname $d`
    sub=`basename $d`

    #
    # Deal with some special directories.  For each directory, check
    # if it's libraries and such should go into some special package. 
    # 
    case $d in 		
	bindings/pyroot)set_lib_names    $base python 	;;
	bindings/*)     set_lib_names    $base $sub 	;;
	build)          continue ;;
	core/winnt)     continue ;; 
	core/newdelete) set_lib_names    $base "" root-system-bin
	                extra="ALLMAPS=${prefix}/lib/root/libCore.rootmap ";;
	core/rint)      set_lib_names    $base "" root-system-bin ;;
	core/thread)    set_lib_names    $base "" root-system-bin ;;
	core/*)         set_lib_names    $base "" root-system-bin
	                extra="ALLMAPS=${prefix}/lib/root/libCore.rootmap "
                        extra="$extra ALLLIBS=${prefix}/lib/root/libCore.so" ;; 
	cint/cint)	set_lib_names    core  "" root-system-bin
	    	        extra="ALLLIBS=${prefix}/lib/root/libCint.so" ;;   
	    	        # extra="NOMAP=1 ALLLIBS=${prefix}/lib/root/libCint.so" 
	cint/reflex)	set_lib_names    core  "" libroot-core-dev
	    	        extra="REFLEXLIB=${prefix}/lib/root/libReflex.so" ;;   
	cint/*)	        set_lib_names    core  "" root-system-bin	;;
	geom/geom)	set_lib_names	 $base 		;;
	geom/*)	        set_plugin_names $base $sub	;;
	graf2d/gpad)    set_lib_names    $base $sub	;;
	graf2d/graf)    set_lib_names    $base $sub	;;
	graf2d/postscript) set_lib_names    $base $sub	;;
	graf2d/asimage) set_plugin_names $base $sub	;;
	graf2d/freetype)continue;;
	graf2d/win32gdk)continue;;
	graf2d/x11*)    set_plugin_names $base x11	;;
	graf2d/*)       set_plugin_names $base $sub	;;
	graf3d/ftgl)    set_lib_names    $base gl       ;;
	graf3d/gl)      set_lib_names    $base $sub	;;
	graf3d/g3d)     set_lib_names    $base $sub	;;
	graf3d/eve)     set_lib_names    $base $sub	;;
	graf3d/*)       set_plugin_names $base $sub	;;
	gui/gui)        set_lib_names    $base 		;;
	gui/guihtml)    set_lib_names    $base 		;;
	gui/ged)        set_lib_names    $base $sub	;;
	gui/qt*)        set_plugin_names $base qt	;;
	gui/*)          set_plugin_names $base $sub	;;
	hist/hist)      set_lib_names    $base		;;
	hist/spectrum)  set_lib_names    $base $sub	;;
	hist/*)         set_plugin_names $base $sub	;;
        html)           set_lib_names    $sub 		;;
	io/io)          set_lib_names    $base 		;;
	io/xmlparser)   set_lib_names    $base $sub	;;
	io/rfio)	continue;;
	io/*)           set_plugin_names $base $sub	;;
	main)           continue;;
        math/fftw)      set_plugin_names $base ${sub}3	;;
        math/fumili)    set_plugin_names $base $sub	;;
        math/minuit2)   set_plugin_names $base $sub	;;
	math/*)         set_lib_names    $base $sub	;;
        misc/*)         set_lib_names	 $base $sub	;; 
        montecarlo/pythia*)   
	                set_plugin_names $base $sub	;;
        montecarlo/*)   set_lib_names    $base $sub	;;
        net/auth)       set_lib_names    $base $sub	;;
	net/net)        set_lib_names    $base 		;;
	net/ldap)       set_lib_names    $base $sub	;;
	net/rootd)      lib=root-system-$sub    ; dev=$lib       ; bin=$lib ;;
	net/xrootd)     set_plugin_names $base $sub	;;
        net/globusauth) set_plugin_names $base globus	;;
        net/krb5auth)   set_plugin_names $base krb5	;;
        net/srputils)   set_plugin_names $base srp	;;
        net/rpdutils)   set_lib_names    core  ""	root-system-bin	;;
        net/*)          set_plugin_names $base $sub	;;
	proof/proofd)   set_plugin_names $base xproof   root-system-${base}d ;;
	proof/proofx)   set_plugin_names $base xproof   ;;
        proof/clarens)  set_lib_names    $base $sub	;;
        proof/proof)    set_lib_names    $base 		;;
        proof/*)        set_plugin_names $base $sub	;;
        roofit/*)       set_lib_names    $base  	;; 
	rootx)	        set_lib_names    core  ""	root-system-bin ;;
	sql/sapdb)      set_plugin_names $base maxdb	;;
	sql/*)          set_plugin_names $base $sub	;;
        tmva)           set_lib_names    $sub 		;;
        tree/tree)      set_lib_names    $base 		;;
        tree/treeplayer)set_lib_names    $base $sub	;;
	tree/*)         set_plugin_names $base $sub	;;
	*)              set_plugin_names $base $sub	;;  
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
	lib*static*)  b=$outdir/${b}          ;;
	lib*-dev)     b=$outdir/${b}          ;; 
	lib*)         b=$outdir/${b}${sovers} ;;
	*)            b=$outdir/${b}          ;; 
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
