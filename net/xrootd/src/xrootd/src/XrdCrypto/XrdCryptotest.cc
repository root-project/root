// $Id$

const char *XrdCryptotestCVSID = "$Id$";
//
//  Test program for XrdCrypto
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <XrdOuc/XrdOucString.hh>

#include <XrdSut/XrdSutAux.hh>
#include <XrdSut/XrdSutBucket.hh>

#include <XrdCrypto/XrdCryptoAux.hh>
#include <XrdCrypto/XrdCryptoFactory.hh>
#include <XrdCrypto/XrdCryptoCipher.hh>
#include <XrdCrypto/XrdCryptoMsgDigest.hh>
#include <XrdCrypto/XrdCryptoRSA.hh>
#include <XrdCrypto/XrdCryptoX509.hh>

//
// Globals 

#define PRINT(x) {cerr <<x <<endl;}
XrdCryptoFactory *gCryptoFactory = 0;

int main( int argc, char **argv )
{
   // Test implemented functionality
   char cryptomod[64] = "ssl";
   char outname[256] = {0};

   //
   // Set debug flags
   XrdSutSetTrace(sutTRACE_Debug);
   XrdCryptoSetTrace(cryptoTRACE_Debug);

   //
   // Determine application name
   char *p = argv[0];
   int k = strlen(argv[0]);
   while (k--)
      if (p[k] == '/') break;
   strcpy(outname,p+k+1);

   //
   // Check/Use inputs
   if(!argv[1]) {
      printf("\n Usage: %s <crypto_module_name>\n",outname);
      printf("   e.g. %s ssl\n",outname);
      printf(" Assuming <crypto_module_name> = ssl\n\n");
   } else {
      strcpy(cryptomod,argv[1]);
   }
   bool local = !strcmp(cryptomod,"local");

   //
   // Load the crypto factory
   if (!(gCryptoFactory = XrdCryptoFactory::GetCryptoFactory(cryptomod))) {
      PRINT(outname<<": cannot instantiate factory "<<cryptomod);
      exit(1);
   }
   gCryptoFactory->SetTrace(cryptoTRACE_Debug);

   //
   // Message Digest of a simple message
   PRINT(outname<<": --------------------------------------------------- ");
   PRINT(outname<<": Testing MD ... ");
   XrdCryptoMsgDigest *MD_1 = gCryptoFactory->MsgDigest("md5");
   if (MD_1) {
      MD_1->Update("prova",strlen("prova"));
      MD_1->Final();
      // Check result
      char MD5prova[128] = "189bbbb00c5f1fb7fba9ad9285f193d1";
      if (strncmp(MD_1->AsHexString(),MD5prova,MD_1->Length())) {
         PRINT(outname<<": MD mismatch: ");
         PRINT(outname<<":        got: "<<MD_1->AsHexString());
         PRINT(outname<<": instead of: "<<MD5prova);
      } else {
         PRINT(outname<<": MD test OK ");
      }
      delete MD_1;
   } else
      PRINT(outname<<": MD object could not be instantiated: ");

   //
   // Instantiate a cipher
   PRINT(outname<<": --------------------------------------------------- ");
   PRINT(outname<<": Testing symmetric cipher ... ");
   XrdCryptoCipher *BF_1 = gCryptoFactory->Cipher("bf-cbc");
   if (BF_1) {
      PRINT(outname<<": cipher length: "<<BF_1->Length());
      PRINT(outname<<": cipher hex: "<<BF_1->AsHexString());
      char tm_1[64] = "Test message for cipher - 001";
      PRINT(outname<<": Test message:   "<<tm_1);
      int ltm_1 = strlen(tm_1);
      char *tmp_1 = new char[BF_1->EncOutLength(ltm_1)];
      if (tmp_1) {
         int ltmp = BF_1->Encrypt(tm_1,ltm_1,tmp_1);
         char tm_2[128] = {0};
         XrdSutToHex(tmp_1,ltmp,&tm_2[0]);
         PRINT(outname<<": cipher encrypted (hex):");
         PRINT(tm_2);
         char *tm_3 = new char[BF_1->DecOutLength(ltmp)];
         int lfin = BF_1->Decrypt(tmp_1,ltmp,tm_3);
         delete[] tmp_1;
         if (tm_3) {
            PRINT(outname<<": cipher decrypted:   "<<tm_3);
            if (strncmp(tm_1,tm_3,ltm_1)) {
               PRINT(outname<<": symmetric cipher test failed: ");
               PRINT(outname<<":        got: "<<tm_3<<" ("<<lfin<<" bytes)");
               PRINT(outname<<": instead of: "<<tm_1<<" ("<<ltm_1<<" bytes)");
            } else {
               PRINT(outname<<": symmetric cipher test OK ");
            }
            delete[] tm_3;
         } else
            PRINT(outname<<": cipher decryption failure");
      } else
         PRINT(outname<<": cipher encryption failure");

      // Bucket encryption
      PRINT(outname<<": testing bucket encryption");
      XrdOucString Astr("TestBucket");
      XrdSutBucket Bck0(Astr);
      PRINT(outname<<": length of string: "<<Bck0.size);
      XrdSutBucket Bck1(Astr);
      int lo1 = BF_1->Encrypt(Bck1);
      PRINT(outname<<": length of encryption: "<<lo1);
      int lo2 = BF_1->Decrypt(Bck1);
      PRINT(outname<<": length of decryption: "<<lo2);
      if (Bck1 != Bck0) {
         PRINT(outname<<": test bucket encryption failed: ");
         PRINT(outname<<":        got: "<<lo2<<" bytes)");
         PRINT(outname<<": instead of: "<<lo1<<" bytes)");
      } else {
         PRINT(outname<<": test bucket encryption OK");
      }

   } else
      PRINT(outname<<": cipher object could not be instantiated: ");

   //
   // Try KDFun ...
   PRINT(outname<<": --------------------------------------------------- ");
   PRINT(outname<<": Testing KDFun ... ");
   XrdCryptoKDFun_t KDFun = gCryptoFactory->KDFun();
   if (KDFun) {
      const char *pass = "pippo";
      int plen = strlen(pass);
      const char *salt = "$$10000$derek";
      int slen = strlen(salt);
      char key[128];
      char KDFunprova[128] = {0};
      bool matching = 0;
      if (local) {
         int klen = (*KDFun)(pass,plen,salt,slen,key,0);
         PRINT(outname<<": key is: "<< key<< " ("<<klen<<" bytes)");
         strcpy(KDFunprova,"igcdgcbcebkplgajngjkfjlbcbiponnkifmeafpdmglp"
                           "lnfkpkjgbmlgbnhehnec");
         matching = !strncmp(key,KDFunprova,klen);
      } else {
         int klen = (*KDFun)(pass,plen,salt,slen,key,0);
         char khex[2046] = {0};
         int i = 0;
         for(; i < klen; i++) sprintf(khex,"%s%02x",khex, 0xFF & key[i]);
         PRINT(outname<<": key is: "<< khex<< " ("<<klen<<" bytes)");
         strcpy(KDFunprova,"b8d309875d91b050eea1527d91559f6ffa023601da0976de");
         matching = !strncmp(khex,KDFunprova,strlen(khex));
      }
      // Check result
      if (!matching) {
         PRINT(outname<<": KDFun mismatch: ");
         PRINT(outname<<": key should have been: "<<KDFunprova);
      } else {
         PRINT(outname<<": KDFun test OK ");
      }
   } else
      PRINT(outname<<": KDFun object could not be instantiated: ");

   //
   // Instantiate a RSA pair
   PRINT(outname<<": --------------------------------------------------- ");
   PRINT(outname<<": Testing RSA ... ");
   XrdCryptoRSA *TestRSA_1 = gCryptoFactory->RSA(1024);
   if (TestRSA_1) {
      XrdCryptoRSA *CpyRSA = gCryptoFactory->RSA(*TestRSA_1);
      if (CpyRSA)
         CpyRSA->Dump();

      char RSApubexp[4096];
      TestRSA_1->ExportPublic(RSApubexp,4096);
      PRINT(outname<<": public export:"<<endl<<RSApubexp);
      PRINT(outname<<": The two printouts above should be equal");
      PRINT(outname<<": --------------------------------------------------- ");
      PRINT(outname<<": outlen : "<<TestRSA_1->GetPublen());
      PRINT(outname<<": --------------------------------------------------- ");
      char RSApriexp[4096];
      TestRSA_1->ExportPrivate(RSApriexp,4096);
      PRINT(outname<<": private export:"<<endl<<RSApriexp);
      PRINT(outname<<": --------------------------------------------------- ");
      PRINT(outname<<": outlen : "<<TestRSA_1->GetPrilen());
      PRINT(outname<<": --------------------------------------------------- ");
      PRINT(outname<<": --------------------------------------------------- ");
      PRINT(outname<<": testing import/export ");
      XrdCryptoRSA *TestRSA_2 = gCryptoFactory->RSA(1024);
      TestRSA_2->ImportPublic(RSApubexp,strlen(RSApubexp));
      TestRSA_2->ImportPrivate(RSApriexp,strlen(RSApriexp));

      PRINT(outname<<": --------------------------------------------------- ");
      char buf_1[128] = "Here I am ... in test";
      int lin = strlen(buf_1);
      char buf_2[4096];
      PRINT(outname<<": encrypting (public): "<<buf_1<<" ("<<strlen(buf_1)<<" bytes)");
      int lout1 = TestRSA_1->EncryptPublic(buf_1,strlen(buf_1),buf_2,512);
      char buf_2_hex[4096];
      XrdSutToHex(buf_2,lout1,buf_2_hex);
      PRINT(outname<<": output has "<<lout1<<" bytes: here is its hex:");
      PRINT(outname<<": "<<buf_2_hex);
      char buf_3[4096];
      PRINT(outname<<": decrypting (private): ("<<lout1<<" bytes)");
      int lout2 = TestRSA_2->DecryptPrivate(buf_2,lout1,buf_3,512);
      PRINT(outname<<": got: "<<buf_3<<" ("<<lout2<<" bytes)");
      if (memcmp(buf_1,buf_3,lin)) {
         PRINT(outname<<": RSA public enc / private dec mismatch: ");
         PRINT(outname<<":        got: "<<buf_3<<" ("<<lout2<<" bytes)");
         PRINT(outname<<": instead of: "<<buf_1<<" ("<<strlen(buf_1)<<" bytes)");
      } else if (lout2 > lin) {
         PRINT(outname<<": RSA public enc / private dec length mismatch: ");
         PRINT(outname<<":   got: "<<lout2<<" instead of "<<lin);
         int j = lin;
         for (; j<lout2; j++) printf("%s: %d: 0x%x\n",outname,j,(int)buf_3[j]);
      } else {
         PRINT(outname<<": RSA public enc / private dec test OK ");
      }
      PRINT(outname<<": --------------------------------------------------- ");
      PRINT(outname<<": encrypting (private): "<<buf_1<<" ("<<strlen(buf_1)<<" bytes)");
      lout1 = TestRSA_1->EncryptPrivate(buf_1,strlen(buf_1),buf_2,512);
      XrdSutToHex(buf_2,lout1,buf_2_hex);
      PRINT(outname<<": output has "<<lout1<<" bytes: here is its hex:");
      PRINT(outname<<": "<<buf_2_hex);
      PRINT(outname<<": decrypting (public): ("<<lout1<<" bytes)");
      lout2 = TestRSA_2->DecryptPublic(buf_2,lout1,buf_3,512);
      PRINT(outname<<": got: "<<buf_3<<" ("<<lout2<<" bytes)");
      if (memcmp(buf_1,buf_3,lin)) {
         PRINT(outname<<": RSA private enc / public dec mismatch: ");
         PRINT(outname<<":        got: "<<buf_3<<" ("<<lout2<<" bytes)");
         PRINT(outname<<": instead of: "<<buf_1<<" ("<<strlen(buf_1)<<" bytes)");
      } else if (lout2 > lin) {
         PRINT(outname<<": RSA private enc / public dec length mismatch: ");
         PRINT(outname<<":   got: "<<lout2<<" instead of "<<lin);
         int j = lin;
         for (; j<lout2; j++) printf("%s: %d: 0x%x\n",outname,j,(int)buf_3[j]);
      } else {
         PRINT(outname<<": RSA private enc / public dec test OK ");
      }

      // Bucket encryption
      PRINT(outname<<": testing bucket RSA encryption");
      XrdOucString Astr("TestBucket");
      XrdSutBucket Bck0(Astr);
      PRINT(outname<<": length of string: "<<Bck0.size);
      XrdSutBucket Bck1(Astr);
      int lo1 = TestRSA_1->EncryptPrivate(Bck1);
      PRINT(outname<<": length of private encryption: "<<lo1);
      int lo2 = TestRSA_1->DecryptPublic(Bck1);
      PRINT(outname<<": length of public decryption: "<<lo2);
      if (Bck1 != Bck0) {
         PRINT(outname<<": test bucket RSA priv enc / pub dec failed: ");
         PRINT(outname<<":        got: "<<lo2<<" bytes)");
         PRINT(outname<<": instead of: "<<lo1<<" bytes)");
      } else {
         PRINT(outname<<": test bucket RSA priv enc / pub dec  OK");
      }
      XrdSutBucket Bck2(Astr);
      lo1 = TestRSA_1->EncryptPublic(Bck2);
      PRINT(outname<<": length of public encryption: "<<lo1);
      lo2 = TestRSA_1->DecryptPrivate(Bck2);
      PRINT(outname<<": length of private decryption: "<<lo2);
      if (Bck2 != Bck0) {
         PRINT(outname<<": test bucket RSA pub enc / priv dec failed: ");
         PRINT(outname<<":        got: "<<lo2<<" bytes)");
         PRINT(outname<<": instead of: "<<lo1<<" bytes)");
      } else {
         PRINT(outname<<": test bucket RSA pub enc / priv dec  OK");
      }

      delete TestRSA_1;
#if 0
      PRINT(outname<<": --------------------------------------------------- ");
      // repeat 1000 times
      DebugON = 0;
      bool match = 1;
      int i = 0;      
      for (; i<1000; i++) {

         TestRSA_1 = gCryptoFactory->RSA(2048);

         lout1 = TestRSA_1->EncryptPrivate(buf_1,strlen(buf_1),buf_2,4096);
         lout2 = TestRSA_1->DecryptPublic(buf_2,lout1,buf_3,4096);
         if (memcmp(buf_1,buf_3,lin)) {
            PRINT(outname<<": RSA private enc / public dec mismatch: "<<i);
            PRINT(outname<<":        got: "<<buf_3<<" ("<<lout2<<" bytes)");
            PRINT(outname<<": instead of: "<<buf_1<<" ("<<strlen(buf_1)<<" bytes)");
         } else if (lout2 > lin) {
            PRINT(outname<<": RSA private enc / public dec length mismatch: "<<i);
            PRINT(outname<<":   got: "<<lout2<<" instead of "<<lin);
            int j = lin;
            for (; j<lout2; j++) printf("%s: %d: 0x%x\n",outname,j,(int)buf_3[j]);
         }
         delete TestRSA_1;

         if (i && !(i % 10)) PRINT(outname<<": done "<<i);
      }
#endif
   } else
      PRINT(outname<<": RSA object could not be instantiated: ");

   //
   // Test key agreement
   PRINT(outname<<": --------------------------------------------------- ");
   PRINT(outname<<": Testing key agreement for ciphers ... ");
   // Get first cipher
   char *bp1 = 0; 
   int lp1 = 0;
   PRINT(outname<<": CF_1: prepare ...");
   XrdCryptoCipher *CF_1 = gCryptoFactory->Cipher(0,0,0);
   if (CF_1 && CF_1->IsValid()) {
      // Get public part and save it to a buffer
      if (!(bp1 = CF_1->Public(lp1))) {
         PRINT(outname<<": CF_1 cipher: problems getting public part ");
         exit(1);
      }
   } else {
      PRINT(outname<<": CF_1 cipher object could not be instantiated: ");
   }
   // Get a third cipher directly from constructor
   char *bp3 = 0; 
   int lp3 = 0;
   PRINT(outname<<": CF_3: instantiate ... with pub");
   if (!local)
      PRINT(bp1);
   XrdCryptoCipher *CF_3 = gCryptoFactory->Cipher(0,bp1,lp1);
   if (CF_3 && CF_3->IsValid()) {
      // Get public part and save it to a buffer
      if (!(bp3 = CF_3->Public(lp3))) {
         PRINT(outname<<": CF_3 cipher: problems getting public part ");
         exit(1);
      }
   } else {
      PRINT(outname<<": CF_3 cipher object could not be instantiated: ");
   }
   // Complete initialization
   if (CF_1 && CF_1->IsValid() && bp3) {
      PRINT(outname<<": CF_1: finalize ... with pub");
      if (!local)
         PRINT(bp3);
      CF_1->Finalize(bp3,lp3,"default");
   } else {
      PRINT(outname<<": CF_1 cipher object could not be finalized ");
   }
   // Test matching now
   if (CF_1 && CF_1->IsValid() && CF_3 && CF_3->IsValid()) {
      char chex[128] = {0};
      XrdSutToHex(CF_1->Buffer(),CF_1->Length(),&chex[0]);
      PRINT(outname<<": cipher 1 encrypted (hex):");
      PRINT(chex);
      PRINT(outname<<": cipher 1 used length: "<<CF_1->Length());
      XrdSutToHex(CF_3->Buffer(),CF_3->Length(),&chex[0]);
      PRINT(outname<<": cipher 3 encrypted (hex):");
      PRINT(chex);
      PRINT(outname<<": cipher 3 used length: "<<CF_3->Length());
      if (CF_1->Length() == CF_3->Length()) {
         if (!memcmp(CF_1->Buffer(),CF_3->Buffer(),CF_3->Length())) {
            PRINT(outname<<": ciphers match !");
         } else {
            PRINT(outname<<": ciphers DO NOT match !");
         }
      }
   }

   // Encryption
   if (CF_1 && CF_1->IsValid() && CF_3 && CF_3->IsValid()) {
      char tm_1[64] = "Test message for cipher - 001";
      PRINT(outname<<": Test message:   "<<tm_1);
      int ltm_1 = strlen(tm_1);
      char *tmp_1 = new char[CF_1->EncOutLength(ltm_1)];
      if (tmp_1) {
         int ltmp = CF_1->Encrypt(tm_1,ltm_1,tmp_1);
         char tm_2[128] = {0};
         XrdSutToHex(tmp_1,ltmp,&tm_2[0]);
         PRINT(outname<<": cipher encrypted (hex):");
         PRINT(tm_2);
         char *tm_3 = new char[CF_3->DecOutLength(ltmp)+1];
         int lfin = CF_3->Decrypt(tmp_1,ltmp,tm_3);
         delete[] tmp_1;
         if (tm_3) {
            tm_3[lfin] = 0;
            PRINT(outname<<": cipher decrypted:   "<<tm_3);
            if (strncmp(tm_1,tm_3,ltm_1)) {
               PRINT(outname<<": symmetric cipher test failed: ");
               PRINT(outname<<":        got: "<<tm_3<<" ("<<lfin<<" bytes)");
               PRINT(outname<<": instead of: "<<tm_1<<" ("<<ltm_1<<" bytes)");
            } else {
               PRINT(outname<<": symmetric cipher test OK ");
            }
            delete[] tm_3;
         } else
            PRINT(outname<<": cipher decryption failure");
      } else
         PRINT(outname<<": cipher encryption failure");
   }

   if (CF_1) delete CF_1;
   if (CF_3) delete CF_3;

   if (bp1) delete bp1;
   if (bp3) delete bp3;


   //
   // Test X509 ...
   if (gCryptoFactory->ID() == 1) {
      PRINT(outname<<": --------------------------------------------------- ");
      PRINT(outname<<": Testing X509 functionality ... ");
      XrdCryptoX509 *x509 = gCryptoFactory->X509("/home/ganis/.globus/usercert.pem");
      if (x509) {
         x509->Dump();
      }
   }

   PRINT(outname<<": --------------------------------------------------- ");
   exit(0);
}
