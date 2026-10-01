"""Built-in demo examples.

`schema` is in the exact format used for training (see schema_to_text) and is the
only part that goes into the model prompt. `data` is INSERT statements that are
used only by the executor, so the prompt stays in-distribution.
"""

EXAMPLES = [
    {
        "name": "Singers and concerts: count",
        "schema": (
            "CREATE TABLE singer (singer_id number PRIMARY KEY, name text, country text, age number);\n"
            "CREATE TABLE concert (concert_id number PRIMARY KEY, concert_name text, year number);\n"
            "CREATE TABLE singer_in_concert (concert_id number, singer_id number);\n"
            "-- FK: singer_in_concert.concert_id -> concert.concert_id\n"
            "-- FK: singer_in_concert.singer_id -> singer.singer_id"
        ),
        "data": (
            "INSERT INTO singer VALUES (1,'Joe Sharp','Netherlands',52),(2,'Timbaland','United States',32),"
            "(3,'Justin Brown','France',29),(4,'Rose White','France',41);\n"
            "INSERT INTO concert VALUES (1,'Auditions',2014),(2,'Super bootcamp',2014),(3,'Home Visits',2015);\n"
            "INSERT INTO singer_in_concert VALUES (1,1),(1,2),(2,2),(3,3),(3,4);"
        ),
        "question": "How many singers are older than 30?",
    },
    {
        "name": "Singers and concerts: join",
        "schema": (
            "CREATE TABLE singer (singer_id number PRIMARY KEY, name text, country text, age number);\n"
            "CREATE TABLE concert (concert_id number PRIMARY KEY, concert_name text, year number);\n"
            "CREATE TABLE singer_in_concert (concert_id number, singer_id number);\n"
            "-- FK: singer_in_concert.concert_id -> concert.concert_id\n"
            "-- FK: singer_in_concert.singer_id -> singer.singer_id"
        ),
        "data": (
            "INSERT INTO singer VALUES (1,'Joe Sharp','Netherlands',52),(2,'Timbaland','United States',32),"
            "(3,'Justin Brown','France',29),(4,'Rose White','France',41);\n"
            "INSERT INTO concert VALUES (1,'Auditions',2014),(2,'Super bootcamp',2014),(3,'Home Visits',2015);\n"
            "INSERT INTO singer_in_concert VALUES (1,1),(1,2),(2,2),(3,3),(3,4);"
        ),
        "question": "Show the names of singers who performed in concerts held in 2015.",
    },
    {
        "name": "Students and pets",
        "schema": (
            "CREATE TABLE student (stuid number PRIMARY KEY, lname text, fname text, age number, major number);\n"
            "CREATE TABLE pets (petid number PRIMARY KEY, pettype text, pet_age number, weight number);\n"
            "CREATE TABLE has_pet (stuid number, petid number);\n"
            "-- FK: has_pet.stuid -> student.stuid\n"
            "-- FK: has_pet.petid -> pets.petid"
        ),
        "data": (
            "INSERT INTO student VALUES (1,'Smith','Linda',18,600),(2,'Kim','Tracy',19,600),"
            "(3,'Jones','Shiela',21,520),(4,'Lee','Dave',20,540);\n"
            "INSERT INTO pets VALUES (1,'cat',3,12.0),(2,'dog',2,13.4),(3,'dog',1,9.3);\n"
            "INSERT INTO has_pet VALUES (1,1),(2,2),(3,3);"
        ),
        "question": "Find the first names of students who have a cat.",
    },
]