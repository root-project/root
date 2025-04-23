/* This macro has nothing to do with network tests. It tests the load and
 * autoparse of payload of libRLDAP.
 * See bug ROOT-6861
 *
 */

void execLDAPAttribute(){

   int loadRetCode=gInterpreter->AutoLoad("TLDAPAttribute"); // load libRLDAP if available

   if (loadRetCode==1)
      gInterpreter->ProcessLine("TLDAPAttribute myAttribute(\"myName\");");

}
