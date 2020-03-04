DELETE FROM tweets
    WHERE rowid NOT IN (
        SELECT MIN(rowid)
        FROM tweets
        GROUP BY text
        );