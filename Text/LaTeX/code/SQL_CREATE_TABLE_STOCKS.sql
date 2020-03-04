CREATE table stocks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT,
    date DATE,
    source TEXT,
    high REAL,
    low REAL,
    open REAL,
    close REAL,
    adj_close REAL,
    volume INTEGER,
    UNIQUE(symbol, date) ON CONFLICT ROLLBACK
);