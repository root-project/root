{1}

sub monStage {my($usr, $mssfn, $sz, $etm) = @_; my($ip_addr);

# If we have no place to send this, return
#
  my($monhost, $monport) = $Config{mondest} =~ m/^(.*):(.*)$/;
  return if !$monhost || !$monport;

# Convert the mss directory to the logical filename
#
  my($lfn) = &mss2none($mssfn);

# Get a udp socket
#
  return &Emsg("Unable to get socket to $Config{mondest}; $!", 1)
         if !socket(MYSOCK, PF_INET, SOCK_DGRAM, getprotobyname("udp"));

# Convert destination to a sockaddr
#
  if (substr($monhost,0,1) =~ m/\d/) {$ip_addr = inet_aton($monhost);}
     else {$ip_addr = gethostbyname($monhost);}
  my($ip_dest) = pack_sockaddr_in($monport, $ip_addr);

# Construct message
#
  my($tod) = time();
  my($msg) = "$usr\n$lfn\n&tod=$tod&sz=$sz&tm=$etm";
  my($mln) = length($msg) + 8 + 4;
  my($pkt) = pack('CCnNNa*', 's', '\0', $mln, $tod, 0, $msg);

# Send the message
#
  return &Emsg("Unable to send rec to $Config{mondest}; $!", 1)
         if $mln != send(MYSOCK, $pkt, 0, $ip_dest);
  return;
}
