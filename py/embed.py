import json
import time
from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, insert, select, text, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, mapped_column, Session
from scipy import spatial

# engine = create_engine('postgresql://postgres:postgres@localhost:5432/postgres')
engine = create_engine('postgresql://postgres:postgres@db/postgres')
with engine.connect() as conn:
    conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
    conn.commit()

Base = declarative_base()


class Document(Base):
    __tablename__ = 'document'

    id = mapped_column(Integer, primary_key=True)
    title = mapped_column(Text)
    data = mapped_column(JSONB)
    embedding = mapped_column(Vector(384))
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)
model = SentenceTransformer('all-MiniLM-L6-v2')

file = open("/app/data/products.ndjson", "r")
session = Session(engine)

i = 0
limit = 20000
batch = []
sentences = []
for line in file.readlines():
    item = json.loads(line)
    # batch embedding
    batch.append(item)
    sentences.append(f"title: {item['title']} description: {item['description']}")
    if len(batch) == limit:
        start = time.time()
        # print("Encoding batch #{}".format(i))
        embeddings = model.encode(sentences)
        documents = [dict(title=batch[i]["title"], data=batch[i], embedding=embedding) for i, embedding in enumerate(embeddings)]
        result = session.execute(insert(Document), documents)
        # print("Committing batch")
        session.commit()
        i += 1
        batch = []
        sentences = []
        end = time.time()
        print("Batch #{} ({} documents) took {} seconds".format(i, limit * i, end - start))

# model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings = model.encode(sentences)
# # print(embeddings)

# documents = [dict(content=sentences[i], embedding=embedding) for i, embedding in enumerate(embeddings)]

# session = Session(engine)
# result = session.execute(insert(Document), documents)

# session.commit()