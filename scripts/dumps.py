import json, sqlite3
from pathlib import Path

t = [json.loads(l) for l in open("data/test.jsonl", encoding="utf-8")]
ex = next(x for x in t if x["question"].startswith("How many flights depart from 'APG'"))

db = Path("spider_data/database/flight_2/flight_2.sqlite").resolve()
c = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)

print("gold  ", c.execute(ex["gold_sql"]).fetchall())
print("phase3", c.execute('SELECT count(*) FROM flights AS T1 JOIN airports AS T2 ON T1.SourceAirport = T2.AirportCode WHERE T2.City = "APG"').fetchall())
print("served", c.execute('SELECT count(*) FROM flights AS T1 JOIN airports AS T2 ON T1.SourceAirport = T2.AirportCode WHERE T2.City = "Apeldoorn"').fetchall())