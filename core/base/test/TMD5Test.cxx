#include "gtest/gtest.h"
#include "TMD5.h"
#include "TString.h"

TEST(TMD5Test, EmptyStringHash) {
    TMD5 md5;
    md5.Update((const UChar_t*)"", 0);
    md5.Final();
    
    // The MD5 hash of an empty string
    EXPECT_STREQ(md5.AsString(), "d41d8cd98f00b204e9800998ecf8427e");
}

TEST(TMD5Test, BasicStringHash) {
    TMD5 md5;
    const char* input = "abc";
    md5.Update((const UChar_t*)input, strlen(input));
    md5.Final();
    
    // The MD5 hash of "abc"
    EXPECT_STREQ(md5.AsString(), "900150983cd24fb0d6963f7d28e17f72");
}

TEST(TMD5Test, EqualityOperator) {
    TMD5 md5_1;
    const char* input1 = "hello world";
    md5_1.Update((const UChar_t*)input1, strlen(input1));
    md5_1.Final();

    TMD5 md5_2;
    const char* input2 = "hello world";
    md5_2.Update((const UChar_t*)input2, strlen(input2));
    md5_2.Final();
    
    EXPECT_TRUE(md5_1 == md5_2);
    
    TMD5 md5_3;
    const char* input3 = "different";
    md5_3.Update((const UChar_t*)input3, strlen(input3));
    md5_3.Final();
    
    EXPECT_TRUE(md5_1 != md5_3);
}

TEST(TMD5Test, SetDigest) {
    TMD5 md5;
    md5.SetDigest("900150983cd24fb0d6963f7d28e17f72");
    EXPECT_STREQ(md5.AsString(), "900150983cd24fb0d6963f7d28e17f72");
}
