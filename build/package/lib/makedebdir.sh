#!/bin/sh
#
# $Id: makedebdir.sh,v 1.15 2006/08/24 13:49:53 rdm Exp $
#
# Make the debian packaging directory 
#
purge=0
leave=0
clean=0
setup=1
upcl=1
root_sovers=`cat build/version_number | sed 's,/.*,,'` 

# ____________________________________________________________________
usage ()
{
    cat <<EOF
Usage: $0 [OPTIONS]
 
Options:
	-h,--help	This help 
	-p,--purge	Purge source directory
	-c,--clean	Clean the source directory 
	-n,--no-setup	Do not setup debian directory
EOF
    exit 0
}

# ____________________________________________________________________
purge ()
{
    test $purge -lt 1 && return 0 

    cat <<-EOF 
	=============================================================
	Warning: Purging sources of unwanted stuff
	
	I will expand tar-balls, and remove them.  I will also remove 
	non-free True Type Fonts.  To restore these files, you should 
	do a CVS update. 
	=============================================================
	EOF
    # Now, remove files we definitely don't want 
    # rm -f fonts/*.ttf 
    echo 1 -n "Removing unwanted files ... "
    rm -f \
	build/package/common/root-cint.control			\
	build/package/common/root-cint.copyright		\
	build/package/common/root-cint.install.in		\
	build/package/common/libroot-dev.control		\
	build/package/common/root-plugin-clarens.control	\
	build/package/common/root-plugin-ldap.control		\
	build/package/common/root-plugin-minuit.control		\
	build/package/common/root-plugin-mlp.control		\
	build/package/common/root-plugin-python.control		\
	build/package/common/root-plugin-python.install.in	\
	build/package/common/root-plugin-quadp.control		\
	build/package/common/root-plugin-roofit.control		\
	build/package/common/root-plugin-ruby.control		\
	build/package/common/root-plugin-sapdb.control		\
	build/package/common/root-rootd.install.in		\
	build/package/common/root-xrootd.install.old		\
	build/package/common/ttf-root.control			\
	build/package/common/ttf-root.install.in		\
	build/package/debian/libroot.postinst			\
	build/package/debian/libroot.postrm			\
	build/package/debian/pycompat				\
	build/package/debian/root-plugin-roofit.copyright	\
	build/package/debian/root-cint.copyright		\
	build/package/debian/root-cint.postinst.in		\
	build/package/debian/root-cint.postrm.in		\
	build/package/debian/root-cint.prerm.in			\
	build/package/debian/ttf-root.copyright			\
	build/package/lib/makerpmspecs.sh			\
	fonts/LICENSE						
    rm -rf asimage/src/libAfterImage				
    rm -rf xrootd/src/xrootd 
    rm -rf rootfit/src/
    rm -rf rootfit/inc/
    for i in fonts/*.ttf ; do 
	if test ! -f ${i} ; then continue ; fi 
	case $i in 
	    */symbol.ttf) ;; 
	    *) rm $i ;;
	esac
    done
    if test $leave -lt 1 ; then 
        # Remove old package files 
	for i in build/package/*/root-{bin,doc,common,xrootd,rootd,proofd}* 
	  do 
	  if test ! -f $i ; then continue ; fi 
	  rm $i 
	done
    fi
    echo "done"

    # Extract tar-balls, and remove the tar-balls. 
    echo -n "Extracting tar-balls ... "
    # Xrootd
    xtar=`find xrootd/src/ -name "*.tgz"` 
    echo -n "$xtar ... "
    tar -xzf $xtar -C xrootd/src/
    touch xrootd/src/headers.d
    rm -f unuran/src/unuran-*-root/config.status
    rm -f unuran/src/unuran-*-root/config.log
    rm -f $xtar
    # ASImage
    atar=`find asimage/src/ -name "*.tar.gz"` 
    echo -n "$atar ... "
    tar -xzf $atar -C asimage/src/
    touch asimage/src/headers.d
    rm -f $atar
    # Some extra files to delete from the unpacked sources of libAfterimage
    rm -rf asimage/src/libAfterImage/Makefile		\
	asimage/src/libAfterImage/afterbase.h		\
	asimage/src/libAfterImage/afterimage-config	\
	asimage/src/libAfterImage/afterimage-libs	\
	asimage/src/libAfterImage/config.h		\
	asimage/src/libAfterImage/config.log		\
	asimage/src/libAfterImage/config.status
    # Unuran
    utar=`find unuran/src/ -name "*.tar.gz"` 
    echo -n "$utar ... "
    tar -xzf $utar -C unuran/src/
    touch unuran/src/headers.d
    rm -f $utar
    # ROOFit
    ftar=`find roofit/ -name "*.tgz"` 
    echo -n "$ftar ... "
    tar -xzf $ftar -C roofit/
    touch roofit/headers.d
    rm -f $ftar
    echo "done"
}

# ____________________________________________________________________
clean()
{
    if test $clean -lt 1 ; then return 0 ; fi 

    echo -n "Cleaning ... " 
    touch unuran/src/.bogus.tar.gz
    make maintainer-clean \
	ASTEPVERS=.bogus ASTEPETAG=	\
	XROOTDDIRD=      XROOTDETAG=	\
	ROOFITDIRS=      ROOFITDIRI=    ROOFITETAG= \
	UNRVERS=.bogus   UNURANETAG=
    rm -f unuran/src/.bogus.tar.gz
    rm -rf debian
    rm -f fonts/s050000l.pfb
    rm -f fonts/s050000l.pe
    echo "done"
}


# ____________________________________________________________________
vers2num()
{
    echo $1 | 		\
	tr '/' '.' | 	\
	awk 'BEGIN {FS="."}{printf "%d", (($1*1000)+$2)*1000+$3}'
}    

# ____________________________________________________________________
update_cl()
{
    test $upcl -lt 1 && return 0 
    
    cl=build/package/debian/changelog
    echo -n "Update $cl ..."
    root_vers=`cat build/version_number` 
    last_vers=`head -n 1 $cl | sed 's/root-system (\(.*\)).*/\1/'`
    root_lvers=`vers2num $root_vers`
    last_lvers=`vers2num $last_vers`
    if test $root_lvers -gt $last_lvers ; then 
	dch -v ${root_vers}-1 -c $cl "New upstream version"
	echo "done"
    else 
	echo "same version"
    fi
}

# ____________________________________________________________________
setup()
{
    test $setup -lt 1 && return 0 

    ### echo %%% Make the directory 
    echo "Setting up debian directory ... "
    mkdir -p debian

    ### echo %%% Copy files to directory, making subsitutions if needed
    for i in build/package/debian/* ; do 
	if test -d $i ; then 
	    case $i in 
		*/CVS) continue ;;
	    esac
	    echo "Copying directory `basename $i` to debian/" 
	    cp -a $i debian/ 
	    continue
	fi

	case $i in 
	    */lib*-dev*)
		echo "Copying `basename $i` to debian/"
		cp -a $i debian/
		;;
	    */lib*.overrides.in)
		b=`basename $i .overrides.in `
		t="${b}${root_sovers}.overrides"
		echo "Copying ${b}.overrides to debian/${t}"
		sed "s/@libvers@/${root_sovers}/g" < $i > debian/${t}
		;;
	    */lib*.in)
		e=`basename $i .in | sed 's/.*\.//'`
		b=`basename $i .$e.in`
		t="${b}${root_sovers}.${e}.in"
		echo "Copying ${b}.${e}.in to debian/${t}"
		cp -a $i debian/${t}
		;;
	    */lib*)
		e=`basename $i | sed 's/.*\.//'`
		b=`basename $i .$e`
		t="${b}${root_sovers}.${e}"
		echo "Copying ${b}.${e}.in to debian/${t}n"
		cp -a $i debian/${t} 
		;; 
	    */s050000l.pfb|*/s050000l.pe)
                # Copying s050000l.pfb and s050000l.pe to font directory
		b=`basename $i` 
		echo "Copying $b to fonts/$b" 
		cp $i fonts/
		;;
	    *)
		b=`basename ${i}`
		echo "Copying $b to debian/$b"
		cp -a $i debian/
		;;
	esac
    done

    # cp -a build/package/debian/* debian/
    find debian -name "CVS" | xargs -r rm -frv 
    rm -fr debian/root-system-bin.png
    rm -fr debian/application-x-root.png
    chmod a+x debian/rules 
    chmod a+x build/package/lib/*

    # Make sure we rebuild debian/control
    touch debian/control.in
}

# ____________________________________________________________________
while test $# -gt 0 ; do 
    case $1 in 
	-h|--help) 	usage			;; 
	-p|--purge)    	purge=1 		;; 
	-c|--clean)   	clean=1 		;; 
	-n|--no-setup) 	setup=0 ; 	upcl=0 	;;
	-o|--leave-old) leave=1 		;;
	*) echo "Unknown option: $1, try $0 --help" > /dev/stderr ;;
    esac
    shift
done 

# ____________________________________________________________________
purge
clean
update_cl
setup

#
# EOF
#
