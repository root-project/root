
/****************************************************************************
 * @(#) hpibdvr.c  Revision: 1.2  Date: 91/04/12
 *                                Copyright 1987-1991 Agilent Technologies
 *  cc file.c hpib.c -ldvio
 *  makecint -o cint -c hpib.c -l -ldvio
 ****************************************************************************/
static char *revid = "";

/***********/
/* include */
/***********/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <dvio.h>
#include <errno.h>
#include <time.h>


/**********/
/* define */
/**********/
						/* termination reasons */
#define	TR_RTC		0x02			/* read termination character*/
#define	TR_EOI		0x04			/* EOI line asserted */

#define	ENABLE		1			/* enable */
#define	DISABLE		0			/* disable */


/***************/
/* declaration */
/***************/

int hpib_open();
int hpib_output();
int hpib_enter();
int hpib_clear();
/* int hpib_spoll(); */


/*******************/
/* hpib_open() */
/*******************/

int hpib_open(sc)
int sc;
{
	char df[30];

	/* (void) sprintf(df, "/dev/pcs/hpib%d", sc); */
	(void) sprintf(df, "/dev/rmb/hpib%d", sc);
	return ( open(df, O_RDWR) );
}


/*********************/
/* hpib_output() */
/*********************/

int hpib_output(eid, addr, str, flag)
int eid, addr, flag;
char *str;
{
	char cmnd[3];
	int mta;

/* query for CPU bus address */
	if ( (mta = hpib_bus_status(eid, CURRENT_BUS_ADDRESS)) == -1 )
		return ( -1 );

/* check addr */
	if ( addr < 0 || 30 < addr ) {
		errno = EINVAL;
		return ( -1 );
	}

/* send meta commands (listener/talker configuration) */
	cmnd[0] = UNL;					     /* UNLISTEN */
	cmnd[1] = mta + TAG_BASE;		      /* MY TALK ADDRESS */
	cmnd[2] = addr + LAG_BASE;		      /* LISTNER ADDRESS */
	if ( hpib_send_cmnd(eid, cmnd, 3) == -1 )
		return ( -1 );

/* send bytes, considering termination mode specified by flag */
	switch ( flag ) {
	      case 0:				 /* Terminating by CR/LF */
		if ( hpib_eoi_ctl(eid, DISABLE) == -1 ) return ( -1 );
		if ( write(eid, str, strlen(str)) == -1 ) return ( -1 );
		if ( write(eid, "\r\n", 2) == -1 ) return ( -1 );
		return ( 0 );

	      case 1:				 /* Terminating with EOI */
		if ( hpib_eoi_ctl(eid, ENABLE) == -1 ) return ( -1 );
		if ( write(eid, str, strlen(str)) == -1 ) return ( -1 );
		return ( 0 );

	      case 2:					/* No terminator */
		if ( hpib_eoi_ctl(eid, DISABLE) == -1 ) return ( -1 );
		if ( write(eid, str, strlen(str)) == -1 ) return ( -1 );
		return ( 0 );
	}

	return ( 0 );
}


/********************/
/* hpib_enter() */
/********************/

int hpib_enter(eid, addr, str, n)
int eid, addr;
unsigned int n;
char *str;
{
	int mla;
	char cmnd[3];

/* query for CPU bus address */
	if ( (mla = hpib_bus_status(eid, CURRENT_BUS_ADDRESS)) == -1 )
		return ( -1 );

/* check addr */
	if ( addr < 0 || 30 < addr ) {
		errno = EINVAL;
		return ( -1 );
	}

/* send meta commands (listener/talker configuration) */
	cmnd[0] = UNL;					     /* UNLISTEN */
	cmnd[1] = addr + TAG_BASE;		       /* TALKER ADDRESS */
	cmnd[2] = mla + LAG_BASE;		    /* MY LISTEN ADDRESS */
	if ( hpib_send_cmnd(eid, cmnd, 3) == -1 ) return ( -1 );

/* read it */
	return ( read(eid, str, n-1) );
}


/********************/
/* hpib_clear() */
/********************/

int hpib_clear(eid, addr)
int eid, addr;
{
	char cmnd[3];

/* check addr */
	if ( addr > 30 ) {
		errno = EINVAL;
		return ( -1 );
	}

/* generic DEVICE CLEAR */
	if ( addr <= -1 ) {
		cmnd[0] = UNL;				     /* UNLISTEN */
		cmnd[1] = UNL;				       /* UNTALK */
		cmnd[2] = DCLR;				 /* DEVICE CLEAR */
		return ( hpib_send_cmnd(eid, cmnd, 3) );

/* selective DEVICE CLEAR */
	} else {
		cmnd[0] = UNL;				     /* UNLISTEN */
		cmnd[1] = addr + LAG_BASE;	     /* LISTENER ADDRESS */
		cmnd[2] = SDC;			/* SELECTED DEVICE CLEAR */
		return ( hpib_send_cmnd(eid, cmnd, 3) );
	}
}


/********************/
/* hpib_spoll() */
/********************/

int hpib_spoll2(eid, addr)
int eid, addr;
{
	int mla;
	char cmnd[4];
	unsigned char sbyte;

/* query for CPU bus address */
	if ( (mla = hpib_bus_status(eid, CURRENT_BUS_ADDRESS)) == -1 )
		return ( -1 );

/* check addr */
	if ( addr < 0 || 30 < addr ) {
		errno = EINVAL;
		return ( -1 );
	}

/* send SERIAL POLLing meta commands */
	cmnd[0] = UNL;					     /* UNLISTEN */
	cmnd[1] = addr + TAG_BASE;		       /* TALKER ADDRESS */
	cmnd[2] = mla + LAG_BASE;		    /* MY LISTEN ADDRESS */
	cmnd[3] = SPE;				   /* SERIAL POLL ENABLE */
	if ( hpib_send_cmnd(eid, cmnd, 4) == -1 ) return ( -1 );

/* read the status byte */
	if ( read(eid, (char *)(&sbyte), 1) == -1 ) return ( -1 );

/* end SERIAL POLLing */
	cmnd[0] = SPD;				  /* SERIAL POLL DISABLE */
	cmnd[1] = UNT;					       /* UNTALK */
	if ( hpib_send_cmnd(eid, cmnd, 3) == -1 ) return ( -1 );

	return ( sbyte );
}

/***************************************************
*  Waittime(usec)
*   wait time in micro second
*    resolution is 10usec
***************************************************/
void Waittime(usec)
int usec;
{
        struct timeval   tv;
        struct timezone  tz;

        (void)gettimeofday(&tv, &tz);
        usec += tv.tv_usec + tv.tv_sec*1000000;

        do
                (void)gettimeofday(&tv, &tz);
        while ((tv.tv_usec + tv.tv_sec*1000000) < usec);
}

