import requests
from bs4 import BeautifulSoup
import sqlite3
import pandas as pd

#1. Scrape the books from the website
def get_books():
    books = []

    BASE_URL = "https://books.toscrape.com"
    response = requests.get(BASE_URL)

    books_Category = ["travel_2","mystery_3","historical-fiction_4"]


    for page in books_Category:
        url = f"{BASE_URL}/catalogue/category/books/{page}/index.html"
        print(f"Scraping page {url}...")
        response = requests.get(url)
        html = response.text
        soup = BeautifulSoup(html, "html.parser")

        book_items = soup.find_all("article", class_="product_pod")

        for book in book_items:

            if len(books) >= 60:
                break

            title = book.h3.a["title"]
            price = book.find(class_="price_color").text
            star_rating = book.find(class_="star-rating")["class"][1]
            availability = book.find(class_="availability").text.strip()
            category = soup.find(class_="active").text

            if(len(books) < 60):
                books.append({
                    "title": title,
                    "price": price,
                    "star_rating": star_rating,
                    "availability": availability,
                    "category": category
                })

            
    return books;

#2 Clean the scraped fields into proper types:
def clean_books(books):
    rating_map = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5
    }
    for book in books:
        book["price_gbp"] = float(book["price"].replace("Â£", ""))
        book["rating"] = rating_map.get(book["star_rating"], 0)
        book["in_stock"] = "In stock" in book["availability"]
        book["price_inr"] = book["price_gbp"] * 105.50
    return books

#3. Design a normalized SQLite schema with at least two tables sharing a primary/foreign key relationship, for example:
def create_database():    
    #Create the SQLite database and tables
    conn = sqlite3.connect("books.db")
    cursor = conn.cursor()

    conn.execute("DROP TABLE IF EXISTS books")
    conn.execute("DROP TABLE IF EXISTS categories")

    #categories(category_id INTEGER PRIMARY KEY, category_name TEXT UNIQUE)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        )
    """)

    #books(book_id INTEGER PRIMARY KEY, title TEXT, price_gbp REAL, price_inr REAL, rating INTEGER, in_stock INTEGER, category_id INTEGER REFERENCES categories(category_id))
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL,
            price_inr REAL,
            rating INTEGER,
            in_stock  INTEGER,
            category_id INTEGER,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
    """)

    conn.commit()
    conn.close()
    
#4. Using Python's sqlite3 (or pandas.DataFrame.to_sql), insert your cleaned, converted data into this schema. 
def store_books(books):
    conn = sqlite3.connect("books.db")

    conn.execute("PRAGMA foreign_keys = ON")

    for book in books:
        # Insert category
        conn.execute(
            """
            INSERT OR IGNORE INTO categories (category_name)
            VALUES (?)
            """,
            (book["category"],)
        )

        # Get category_id
        category_id = conn.execute(
            """
            SELECT category_id
            FROM categories
            WHERE category_name = ?
            """,
            (book["category"],)
        ).fetchone()[0]

        # Insert book
        conn.execute(
            """
            INSERT INTO books
            (title, price_gbp, price_inr, rating, in_stock, category_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                book["title"],
                book["price_gbp"],
                book["price_inr"],
                book["rating"],
                int(book["in_stock"]),
                category_id
            )
        )

    conn.commit()
    conn.close()
#4.1 Then write and execute at least 5 SQL queries against the database that collectively 
    # demonstrate: SELECT/WHERE, 
                    #ORDER BY, 
                    # LIMIT, 
                    # DISTINCT, 
                    # and (IN or BETWEEN) — plus at least one JOIN between your two tables (e.g., "list the 10 highest-rated books per category"). 
                    # Save each query string and its output.
def fetch_queries():
    conn = sqlite3.connect("books.db")
    cursor = conn.cursor()

    # Example queries
    query1 = "SELECT * FROM books WHERE rating = 5"
    result1 = cursor.execute(query1).fetchall()

    query2 = "SELECT * FROM books ORDER BY price_gbp DESC LIMIT 10"
    result2 = cursor.execute(query2).fetchall()

    query3 = "SELECT DISTINCT title FROM books"
    result3 = cursor.execute(query3).fetchall()

    query4 = "SELECT * FROM books WHERE price_gbp BETWEEN 10 AND 20"
    result4 = cursor.execute(query4).fetchall()

    query5 = """
    SELECT b.title, b.rating, c.category_name
    FROM books b
    JOIN categories c ON b.category_id = c.category_id
    ORDER BY b.rating DESC
    LIMIT 10
    """
    result5 = cursor.execute(query5).fetchall()

    conn.close()
    return {
        "query1": result1,
        "query2": result2,
        "query3": result3,
        "query4": result4,
        "query5": result5
    }

#6 Read back at least two of the above query results into pandas DataFrames using pd.read_sql(...), and separately reproduce the join-query's result using pd.merge(...) directly on your in-memory DataFrames (no SQL) — show that both approaches produce equivalent output.
def fetch_queries_with_pandas():
    conn = sqlite3.connect("books.db")
    df_books = pd.read_sql("SELECT * FROM books", conn)
    df_categories = pd.read_sql("SELECT * FROM categories", conn)

    # Reproduce the join query using pd.merge
    df_merged = pd.merge(df_books, df_categories, on="category_id")
    df_merged = df_merged.sort_values(by="rating", ascending=False).head(10)

    conn.close()
    return df_books, df_categories, df_merged


if __name__ == "__main__":
#1. Scrape the books from the website
    books = get_books()
#2,3. Clean the scraped fields into proper types
    books = clean_books(books)
#4. Create the database and tables if they do not exist.
    create_database()
#5. Store data in to SQLite 3 DB
    store_books(books)
    #5.1 Execute and print the SQL queries
    query_results = fetch_queries()
    for query, result in query_results.items():
        print(f"{query}: {result}")
    
#6 Fetch and print the pandas DataFrames
    df_books, df_categories, df_merged = fetch_queries_with_pandas()
    print("df_books:")
    print(df_books)
    print("df_categories:")
    print(df_categories)
    print("df_merged:")
    print(df_merged)