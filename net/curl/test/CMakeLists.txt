ROOT_ADD_GTEST(curl_connection curl_connection.cxx LIBRARIES RCurlHttp Net)
ROOT_ADD_GTEST(rawfile_curl rawfile_curl.cxx LIBRARIES RCurlHttp)
ROOT_ADD_GTEST(tfile_curl tfile_curl.cxx LIBRARIES RIO RCurlHttp)
