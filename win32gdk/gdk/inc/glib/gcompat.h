/* GLIB - Library of useful routines for C programming
 *
 * Copyright (C) 2000 Ali Abdin <aliabdin@aucegypt.edu>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307, USA.
 */

#ifndef __G_COMPAT_H__
#define __G_COMPAT_H__

#include <gmacros.h>

G_BEGIN_DECLS

#ifndef G_DISABLE_COMPAT_H

#define g_date_weekday 			g_date_get_weekday
#define g_date_month 			g_date_get_month
#define g_date_year 			g_date_get_year
#define g_date_day 			g_date_get_day
#define g_date_julian 			g_date_get_julian
#define g_date_day_of_year 		g_date_get_day_of_year
#define g_date_monday_week_of_year 	g_date_get_monday_week_of_year
#define g_date_sunday_week_of_year 	g_date_get_sunday_week_of_year
#define g_date_days_in_month 		g_date_get_days_in_month
#define g_date_monday_weeks_in_year 	g_date_get_monday_weeks_in_year
#define g_date_sunday_weeks_in_year	g_date_get_sunday_weeks_in_year

#endif /* G_DISABLE_COMPAT_H */

G_END_DECLS

#endif /* __G_COMPAT_H__ */
