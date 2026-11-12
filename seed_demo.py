"""
Empty the app's database so the demo family is seeded fresh on the next start.

    MONGODB_URI=... python seed_demo.py --reset

This deletes every document in the LF database (users, chats, journals,
embeddings, relationships, sessions, caches). Other databases on the cluster
are not touched. After it finishes, restart the server with SEED_DEMO=true
and it will create the demo family from demo_data.py.
"""
import argparse
import os
import sys

import certifi
from dotenv import load_dotenv
from pymongo import MongoClient

DB_NAME = "LF"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="delete all documents in the LF database")
    args = parser.parse_args()
    if not args.reset:
        parser.print_help()
        return

    load_dotenv()
    uri = os.getenv("MONGODB_URI")
    if not uri:
        sys.exit("MONGODB_URI is not set")

    client = MongoClient(uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=15000)
    db = client[DB_NAME]
    counts = {name: db[name].count_documents({}) for name in sorted(db.list_collection_names())}
    print(f"Database '{DB_NAME}' currently has:")
    for name, n in counts.items():
        print(f"  {name:18} {n}")

    if input(f"\nType {DB_NAME} to delete all of this permanently: ").strip() != DB_NAME:
        sys.exit("Cancelled, nothing deleted.")

    for name in counts:
        db[name].delete_many({})
    print(f"Done. '{DB_NAME}' is empty. Restart the server with SEED_DEMO=true to seed the demo family.")
    client.close()


if __name__ == "__main__":
    main()
