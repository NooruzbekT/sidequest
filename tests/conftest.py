import asyncio
import sys

# psycopg async несовместим с ProactorEventLoop (дефолт Windows)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
