#!/bin/sh -e 
#
# $Id: makerpmspec.sh,v 1.13 2006/03/31 00:53:23 rdm Exp $
#
# Make the rpm spec file in ../root.spec
#
#
### echo %%% Some general variables 
chmod a+x build/package/lib/*
tgtdir=rpm

### echo %%% Packages ordered by preference
pkglist=`./configure --pkglist					\
		  --enable-table				\
		  --enable-ruby					\
		  --enable-qt					\
		  --enable-pythia				\
		  --enable-xrootd				\
		  --enable-shared				\
		  --enable-soversion				\
		  --enable-explicitlink				\
		  --disable-rpath				\
		  --disable-afs					\
		  --disable-gfal					\
		  --disable-srp					\
		  --disable-builtin-freetype			\
		  --disable-builtin-afterimage			\
		| sed -n -e 's/packages: //p' -e 's/ttf-root-installer//'` 
pkglist=`echo $pkglist | sed 's/libroot\([-a-zA-Z0-9]*\)/libroot\1 libroot\1-dev/g'`
builddepends=`build/package/lib/makebuilddepend.sh rpm $pkglist`
dpkglist="`echo $pkglist | sed -e 's/ *ttf-root[-a-z]* *//g' -e 's/ /, /g'`, root-ttf"

# ROOT version 
version=`cat build/version_number | tr '/' '.'`
major=`echo $version | cut -f1 -d.`
sovers=`echo $version | cut -f1,2 -d.`
### echo %%% make sure we've got a fresh file 
rm -f root.spec
csplit -f root.spec. build/package/rpm/spec.in '/@builddepends@/'
cat root.spec.00     >  root.spec.in
echo "$builddepends" >> root.spec.in
sed '/@builddepends/d' < root.spec.01 >> root.spec.in
rm -f root.spec.00 root.spec.01
### echo %%% Write header stuff 
sed -e "s/@version@/${version}/" \
    -e "s/@pkglist@/${dpkglist}/" \
    < root.spec.in > root.spec
rm -f root.spec.in
#    -e "s/@builddepends@/${builddepends}/" \

# Write out sub-package information 
for p in $pkglist ; do 
    if test "x$p" = "xttf-root-installer" ; then continue ; fi
    case $p in 
	root-common)  pp=$p       ; c=libroot ;;
	libroot*-dev) pp=$p 	  ; c=`echo $p | sed 's/-dev//'`;;  
	libroot*)     pp=$p$major ; c=$p ;; 
	*)            pp=$p       ; c=$p ;; 
    esac 
    echo "Adding package $p ($pp) to spec file" 
    cat >> root.spec <<-EOF
	# -----------------------------------------------
	# Package $pp
	EOF
    sed -n -e "/Package: $p/,/^ / { s/^Package: $p\(@libvers@\)*/%package -n $pp/p; s/^Description:/Summary:/p ; /^ /q}" < build/package/common/$c.control >>  root.spec
    # short=`sed -n 's/Description: //p' < build/package/common/$c.control` 
    # long=`sed -n -e 's/^ \./ /' -e 's/^ //p'<build/package/common/$c.control`
    #     cat >> root.spec <<-EOF
    # 	# -----------------------------------------------
    # 	# Package $pp
    # 	%package -n $pp
    # 	Summary: $short
    # 	Group: Applications/Physics
    # 	EOF
    echo "Group: Applications/Physics" >> root.spec
    case $pp in 
	ttf-root*) 
	    echo "Provides: root-ttf" 				>> root.spec 
	    ;;
	*xrootd|*rootd)
	    echo "Provides: root-file-server" 			>> root.spec 
	    ;;
	*minuit*|*fumili)
	    echo "Provides: root-fitter"			>> root.spec
            ;;
	libroot*)
	    echo "Provides: $p"					>> root.spec
	    ;;
    esac
    case $p in 
	root-bin) 
	    echo "Requires: root-fitter"   		        >> root.spec 
	    ;;
	libroot) 
	    echo "Requires: root-ttf"				>> root.spec 
	    ;; 
	*rootd) 
	    echo "Prefix: %_prefix" >> root.spec 
	    ;; 
    esac
    #     cat >> root.spec <<-EOF
    # 	%description -n $pp
    # 	$long
    sed -n "/Package: $p/,/^$/ { s/^Description:.*/%description -n $pp/p ; s/^ //p; /^$/q }"  < build/package/common/$c.control >>  root.spec
    case $p in 
	lib*-dev) files=rpm/${p}.install ;; 
	lib*)     files=rpm/${p}${sovers}.install ;;
	*) 	  files=rpm/${p}.install ;; 
    esac

    cat >> root.spec <<-EOF
	%files -n $pp -f ${files}
	%defattr(-,root,root)

	EOF

    for s in post postun pre preun ; do 
	if test -f build/package/rpm/$p.$s ; then 
	    echo "%$s -n $pp" 			>> root.spec
	    cat build/package/rpm/$p.$s		>> root.spec
echo "" >> root.spec
	fi
done 
echo "" >> root.spec
done 
cat >> root.spec <<EOF
#
# End of ROOT spec file
#
EOF

#
# EOF
#
