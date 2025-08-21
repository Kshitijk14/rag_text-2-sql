$ uv run main.py

2025-08-20 22:20:40,012 [INFO]

2025-08-20 22:20:40,012 [INFO] ////--//--//----STARTING [PIPELINE 01] RAG PIPELINE----//--//--////

2025-08-20 22:20:40,012 [INFO]

2025-08-20 22:20:40,012 [INFO] ----------STARTING [STAGE 01] DATA PREPARATION----------

2025-08-20 22:20:40,013 [INFO]

2025-08-20 22:20:40,013 [INFO] ======== Starting Stage 01: Data Prep (schema -> table_summaries.db) ========

2025-08-20 22:20:40,014 [INFO] [utils.sql_schema_utils] Connecting to SQLite DB: C:\Users\Hp\AppData\Roaming\DBeaverData\workspace6\.metadata\sample-database-sqlite-1\Chinook.db

2025-08-20 22:20:40,016 [INFO] [utils.sql_schema_utils] Connecting to SQLite DB: db\Chinook_pipeline\sqlite\table_summaries.db

2025-08-20 22:20:40,016 [INFO] [Stage 01] Ensuring output DB schema exists...

2025-08-20 22:20:40,038 [INFO] [Stage 01] Output DB schema ensured.

2025-08-20 22:20:40,042 [INFO] [Stage 01] Found 11 user tables: ['Album', 'Artist', 'Customer', 'Employee', 'Genre', 'Invoice', 'InvoiceLine', 'MediaType', 'Playlist', 'PlaylistTrack', 'Track']

2025-08-20 22:20:40,042 [INFO] [Stage 01] Processing table: Album

2025-08-20 22:20:40,062 [INFO] [Stage 01] Processing table: Artist

2025-08-20 22:20:40,072 [INFO] [Stage 01] Processing table: Customer

2025-08-20 22:20:40,084 [INFO] [Stage 01] Processing table: Employee

2025-08-20 22:20:40,101 [INFO] [Stage 01] Processing table: Genre

2025-08-20 22:20:40,111 [INFO] [Stage 01] Processing table: Invoice

2025-08-20 22:20:40,133 [INFO] [Stage 01] Processing table: InvoiceLine

2025-08-20 22:20:40,153 [INFO] [Stage 01] Processing table: MediaType

2025-08-20 22:20:40,165 [INFO] [Stage 01] Processing table: Playlist

2025-08-20 22:20:40,174 [INFO] [Stage 01] Processing table: PlaylistTrack

2025-08-20 22:20:40,194 [INFO] [Stage 01] Processing table: Track

2025-08-20 22:20:40,211 [INFO] ======== Stage 01 completed successfully. ========

2025-08-20 22:20:40,212 [INFO]

2025-08-20 22:20:40,212 [INFO] ----------FINISHED [STAGE 01] DATA PREPARATION----------

2025-08-20 22:20:40,212 [INFO]

2025-08-20 22:20:40,213 [INFO] ////--//--//----FINISHED [PIPELINE 01] RAG PIPELINE----//--//--////

2025-08-20 22:20:40,213 [INFO]

it ran, but it didn't call the llm, & i wanna check if the summaries are there or not, this was doubtingly fast, and i'm not sure if the summaries are there or not
