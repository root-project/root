CREATE TABLE runcatalog (
   dataset       VARCHAR(32) NOT NULL,
   run           INT         NOT NULL,
   firstevent    INT,
   events        INT,
   tag           INT,
   energy        FLOAT,
   runtype       ENUM('physics','cosmics','test'),
   target        VARCHAR(10),
   timef         TIMESTAMP NOT NULL,
   timel         TIMESTAMP NOT NULL,
   rawfilepath   VARCHAR(128),
   comments      VARCHAR(80)
)
