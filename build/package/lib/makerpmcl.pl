#!/usr/bin/perl

$since='';
$expect='first heading';
sub clerror { printf(STDERR "$_[0], at changelog line $.\n"); exit(1); }

sub printit { 
    $fulldate=$_[0];
    $fullauth=$_[1];
    $changes=$_[2];
    $changes =~ s/^\s+(\*)/   +/g;
    $_=$fulldate;
    if (m/([a-zA-Z]+), *([0-9]+) *([a-zA-Z]+) *([0-9]+).*/i) {
	$day=$1;
	$date=$2;
	$month=$3;
	$year=$4;
    }
    else {
	&clerror("Bad date format $_");
    }
    $_=$fullauth;
    if (m/([^<]+) *<([^>]+)>/i) {
	$name=$1;
	$email=$2;
    }
    else {
	&clerror("Bad maintainer format $_");
    }
    printf("* %s %s %02d %d %s %s\n%s\n", 
	   $day, $month, $date, $year, $name, $email, $changes);
}

while (<STDIN>) {
    # Eat white space 
    s/\s*\n$//;
    # printf(STDERR "%-39.39s %-39.39s\n",$expect,$_);
    # Match a line 
    if (m/^(\w[-+0-9a-z.]*) \(([^\(\) \t]+)\)((\s+[-0-9a-z]+)+)\;/i) {
        if ($expect eq 'first heading') {
            $f{'Source'}       =  $1;
            $f{'Version'}      =  $2;
            $f{'Distribution'} =  $3;
            $f{'Distribution'} =~ s/^\s+//;
	    $f{'Changes'}      =  "  \n   [" . $f{'Version'} . "]\n"; 
        } elsif ($expect eq 'next heading or eof') {
            last if $2 eq $since;
            $f{'Version'}      =  $2;
            $f{'Distribution'} =  $3;
            $f{'Distribution'} =~ s/^\s+//;
	    $f{'Changes'}      .= "  \n   [" . $f{'Version'} . "]\n"; 
        } else {
            &clerror("found start of entry where expected $expect");
        }
	$expect= 'start of change data'; $blanklines=0;
        # $f{'Changes'}.= " $_\n  \n";
    } elsif (m/^\S/) {
	&clerror("badly formatted heading line");
    } elsif (m/^ \-\- (.*) <(.*)>  ((\w+\,\s*)?\d{1,2}\s+\w+\s+\d{4}\s+\d{1,2}:\d\d:\d\d\s+[-+]\d{4}(\s+\([^\\\(\)]\))?)$/) {
        $expect eq 'more change data or trailer' ||
            &clerror("found trailer where expected $expect");
        $f{'Maintainer'}= "$1 <$2>" unless defined($f{'Maintainer'});
        $f{'Date'}= $3 unless defined($f{'Date'});
	&printit($f{'Date'},$f{'Maintainer'},$f{'Changes'});
	$f{'Changes'} = '';
        $expect= 'next heading or eof';
        # last if $since eq '';
    } elsif (m/^ \-\-/) {
        &clerror("badly formatted trailer line");
    } elsif (m/^\s{2,}\S/) {
        $expect eq 'start of change data' || 
	    $expect eq 'more change data or trailer' ||
            &clerror("found change data where expected $expect");
        $f{'Changes'} .= ("  \n"x$blanklines);
	s/^ *\*/  +/;
	$f{'Changes'} .= " $_\n"; 
	$blanklines=0;
        $expect= 'more change data or trailer';
    } elsif (!m/\S/) {
        next if $expect eq 'start of change data' || 
	    $expect eq 'next heading or eof';
        $expect eq 'more change data or trailer' ||
            &clerror("found blank line where expected $expect");
        $blanklines++;
    } else {
        &clerror("unrecognised line");
    }

    
}
