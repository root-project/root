C=== zbcdh.inc =================================================
      CHARACTER  CQALLC*96
      COMMON /ZBCDCH/ CQALLC
                      CHARACTER*1  CQLETT(96), CQNUM(10)
                      EQUIVALENCE (CQLETT(1),CQALLC(1:1))
                      EQUIVALENCE (CQNUM(1), CQALLC(27:27))
