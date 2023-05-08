from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, insert, select, text, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, mapped_column, Session
from scipy import spatial
import time

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

model = SentenceTransformer('all-MiniLM-L6-v2')
session = Session(engine)

def print_seconds_elapsed(action, start):
    mark = time.time()
    print('action: {} took {:.2f} seconds'.format(action, mark - start))
    return mark

# Find the most similar document to this query
query = ''
while query != 'quit':
    query = input('Enter a query: ')
    if query == 'q':
        break
    start = time.time()
    embed = model.encode([query], show_progress_bar=True)
    start = print_seconds_elapsed('encode', start)
    # sort by cosine distance
    result = session.scalars(select(Document).order_by(Document.embedding.cosine_distance(embed[0])).limit(10))
    start = print_seconds_elapsed('query', start)
    print('Most similar to "{}":'.format(query))
    for row in result:
        score = 1 - spatial.distance.cosine(embed[0], row.embedding)
        # show distance
        print(row.title, score)
    start = print_seconds_elapsed('print', start)