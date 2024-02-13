create user service with password 'servicePW';

GRANT USAGE ON SCHEMA football TO service;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA football TO service;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA football TO service;