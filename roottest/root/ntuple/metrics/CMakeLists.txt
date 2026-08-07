ROOTTEST_ADD_TEST(metrics_env_enabled
                  MACRO test_rntuple_metrics_env.C
                  ENVIRONMENT ROOT_EXPORT_RNTUPLE_METRICS=metrics_env_enabled.root
                  OUTREF metrics_env_enabled.ref)

ROOTTEST_ADD_TEST(metrics_env_disabled
                  MACRO test_rntuple_metrics_env.C
                  OUTREF metrics_env_disabled.ref)
