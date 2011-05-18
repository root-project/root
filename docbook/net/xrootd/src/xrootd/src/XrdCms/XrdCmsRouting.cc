/******************************************************************************/
/*                                                                            */
/*                      X r d C m s R o u t i n g . c c                       */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//       $Id$

const char *XrdCmsRoutingCVSID = "$Id$";

#include "XrdCms/XrdCmsNode.hh"
#include "XrdCms/XrdCmsRouting.hh"

using namespace XrdCms;

/******************************************************************************/
/*                        G l o b a l   O b j e c t s                         */
/******************************************************************************/
/******************************************************************************/
/*                          R o u t e r   T a b l e                           */
/******************************************************************************/
  
namespace XrdCms
{
XrdCmsRouter::theRoute initRouter[] =
      {{kYR_login,   "login",  static_cast<XrdCmsRouter::NodeMethod_t>(0)},
       {kYR_chmod,   "chmod",  &XrdCmsNode::do_Chmod},
       {kYR_locate,  "locate", &XrdCmsNode::do_Locate},
       {kYR_mkdir,   "mkdir",  &XrdCmsNode::do_Mkdir},
       {kYR_mkpath,  "mkpath", &XrdCmsNode::do_Mkpath},
       {kYR_mv,      "mv",     &XrdCmsNode::do_Mv},
       {kYR_prepadd, "prepadd",&XrdCmsNode::do_PrepAdd},
       {kYR_prepdel, "prepdel",&XrdCmsNode::do_PrepDel},
       {kYR_rm,      "rm",     &XrdCmsNode::do_Rm},
       {kYR_rmdir,   "rmdir",  &XrdCmsNode::do_Rmdir},
       {kYR_select,  "select", &XrdCmsNode::do_Select},
       {kYR_statfs,  "statfs", &XrdCmsNode::do_StatFS},
       {kYR_stats,   "stats",  &XrdCmsNode::do_Stats},
       {kYR_trunc,   "trunc",  &XrdCmsNode::do_Trunc},
/* Server */
       {kYR_avail,   "avail",  &XrdCmsNode::do_Avail},
       {kYR_disc,    "disc",   &XrdCmsNode::do_Disc},
       {kYR_gone,    "gone",   &XrdCmsNode::do_Gone},
       {kYR_have,    "have",   &XrdCmsNode::do_Have},
       {kYR_load,    "load",   &XrdCmsNode::do_Load},
       {kYR_ping,    "ping",   &XrdCmsNode::do_Ping},
       {kYR_pong,    "pong",   &XrdCmsNode::do_Pong},
       {kYR_space,   "space",  &XrdCmsNode::do_Space},
       {kYR_state,   "state",  &XrdCmsNode::do_State},
       {kYR_status,  "status", &XrdCmsNode::do_Status},
       {kYR_try,     "try",    &XrdCmsNode::do_Try},
       {kYR_update,  "update", &XrdCmsNode::do_Update},
       {kYR_usage,   "usage",  &XrdCmsNode::do_Usage},
       {0,           0,        0}};
}

// Request Name and Method Routing Table
//
   XrdCmsRouter    XrdCms::Router(initRouter);

/******************************************************************************/
/*              R e d i r e c t o r   R o u t i n g   T a b l e               */
/******************************************************************************/
  
namespace XrdCms
{
XrdCmsRouting::theRouting initRDRrouting[] =
     {{kYR_chmod,   XrdCmsRouting::AsyncQ1   | XrdCmsRouting::Forward
                  | XrdCmsRouting::Repliable | XrdCmsRouting::Delayable},
      {kYR_locate,  XrdCmsRouting::AsyncQ0
                  | XrdCmsRouting::Repliable | XrdCmsRouting::Delayable},
      {kYR_mkdir,   XrdCmsRouting::AsyncQ1   | XrdCmsRouting::Forward
                  | XrdCmsRouting::Repliable | XrdCmsRouting::Delayable},
      {kYR_mkpath,  XrdCmsRouting::AsyncQ1   | XrdCmsRouting::Forward
                  | XrdCmsRouting::Repliable | XrdCmsRouting::Delayable},
      {kYR_mv,      XrdCmsRouting::AsyncQ1   | XrdCmsRouting::Forward
                  | XrdCmsRouting::Repliable | XrdCmsRouting::Delayable},
      {kYR_prepadd, XrdCmsRouting::isSync
                  | XrdCmsRouting::Repliable | XrdCmsRouting::Delayable},
      {kYR_prepdel, XrdCmsRouting::isSync    | XrdCmsRouting::Forward
                  | XrdCmsRouting::Repliable | XrdCmsRouting::Delayable},
      {kYR_rm,      XrdCmsRouting::AsyncQ1   | XrdCmsRouting::Forward
                  | XrdCmsRouting::Repliable | XrdCmsRouting::Delayable},
      {kYR_rmdir,   XrdCmsRouting::AsyncQ1   | XrdCmsRouting::Forward
                  | XrdCmsRouting::Repliable | XrdCmsRouting::Delayable},
      {kYR_select,  XrdCmsRouting::AsyncQ0
                  | XrdCmsRouting::Repliable | XrdCmsRouting::Delayable},
      {kYR_statfs,  XrdCmsRouting::AsyncQ0
                  | XrdCmsRouting::Repliable | XrdCmsRouting::Delayable},
      {kYR_stats,   XrdCmsRouting::AsyncQ0
                  | XrdCmsRouting::Repliable | XrdCmsRouting::Delayable},
      {kYR_trunc,   XrdCmsRouting::AsyncQ1   | XrdCmsRouting::Forward
                  | XrdCmsRouting::Repliable | XrdCmsRouting::Delayable},
      {kYR_update,  XrdCmsRouting::isSync    | XrdCmsRouting::noArgs
                  | XrdCmsRouting::Repliable},
      {0,           0}};
}

// Redirector routing by valid request
//
   XrdCmsRouting    XrdCms::rdrVOps(initRDRrouting);

/******************************************************************************/
/*                R e s p o n s e   R o u t i n g   T a b l e                 */
/******************************************************************************/
  
namespace XrdCms
{
XrdCmsRouting::theRouting initRSProuting[] =
     {{kYR_avail,   XrdCmsRouting::isSync},
      {kYR_disc,    XrdCmsRouting::isSync | XrdCmsRouting::noArgs},
      {kYR_gone,    XrdCmsRouting::isSync},
      {kYR_have,    XrdCmsRouting::AsyncQ0},
      {kYR_load,    XrdCmsRouting::isSync},
      {kYR_pong,    XrdCmsRouting::isSync | XrdCmsRouting::noArgs},
      {kYR_status,  XrdCmsRouting::isSync | XrdCmsRouting::noArgs},
      {0,           0}};
}

// Response routing by valid request
//
   XrdCmsRouting    XrdCms::rspVOps(initRSProuting);

/******************************************************************************/
/*                  S e r v e r   R o u t i n g   T a b l e                   */
/******************************************************************************/
  
namespace XrdCms
{
XrdCmsRouting::theRouting initSRVrouting[] =
     {{kYR_chmod,   XrdCmsRouting::AsyncQ1},
      {kYR_disc,    XrdCmsRouting::isSync  | XrdCmsRouting::noArgs},
      {kYR_mkdir,   XrdCmsRouting::AsyncQ1},
      {kYR_mkpath,  XrdCmsRouting::AsyncQ1},
      {kYR_mv,      XrdCmsRouting::AsyncQ1},
      {kYR_ping,    XrdCmsRouting::isSync  | XrdCmsRouting::noArgs},
      {kYR_prepadd, XrdCmsRouting::isSync},
      {kYR_prepdel, XrdCmsRouting::isSync},
      {kYR_rm,      XrdCmsRouting::AsyncQ1},
      {kYR_rmdir,   XrdCmsRouting::AsyncQ1},
      {kYR_space,   XrdCmsRouting::isSync  | XrdCmsRouting::noArgs},
      {kYR_state,   XrdCmsRouting::AsyncQ0},
      {kYR_stats,   XrdCmsRouting::AsyncQ0 | XrdCmsRouting::noArgs},
      {kYR_trunc,   XrdCmsRouting::AsyncQ1},
      {kYR_try,     XrdCmsRouting::isSync},
      {kYR_usage,   XrdCmsRouting::isSync  | XrdCmsRouting::noArgs},
      {0,           0}};
}
  
// Server routing by valid request
//
   XrdCmsRouting    XrdCms::srvVOps(initSRVrouting);

/******************************************************************************/
/*              S u p e r v i s o r   R o u t i n g   T a b l e               */
/******************************************************************************/
  
namespace XrdCms
{
XrdCmsRouting::theRouting initSUProuting[] =
     {{kYR_chmod,   XrdCmsRouting::AsyncQ1 | XrdCmsRouting::Forward},
      {kYR_disc,    XrdCmsRouting::isSync  | XrdCmsRouting::noArgs},
      {kYR_mkdir,   XrdCmsRouting::AsyncQ1 | XrdCmsRouting::Forward},
      {kYR_mkpath,  XrdCmsRouting::AsyncQ1 | XrdCmsRouting::Forward},
      {kYR_mv,      XrdCmsRouting::AsyncQ1 | XrdCmsRouting::Forward},
      {kYR_ping,    XrdCmsRouting::isSync  | XrdCmsRouting::noArgs},
      {kYR_prepadd, XrdCmsRouting::isSync},
      {kYR_prepdel, XrdCmsRouting::isSync  | XrdCmsRouting::Forward},
      {kYR_rm,      XrdCmsRouting::AsyncQ1 | XrdCmsRouting::Forward},
      {kYR_rmdir,   XrdCmsRouting::AsyncQ1 | XrdCmsRouting::Forward},
      {kYR_space,   XrdCmsRouting::isSync  | XrdCmsRouting::noArgs},
      {kYR_state,   XrdCmsRouting::isSync},
      {kYR_stats,   XrdCmsRouting::AsyncQ0 | XrdCmsRouting::noArgs},
      {kYR_trunc,   XrdCmsRouting::AsyncQ1 | XrdCmsRouting::Forward},
      {kYR_try,     XrdCmsRouting::isSync},
      {kYR_usage,   XrdCmsRouting::isSync  | XrdCmsRouting::noArgs},
      {0,           0}};
}
  
// Supervisor routing by valid request
//
   XrdCmsRouting    XrdCms::supVOps(initSUProuting);
