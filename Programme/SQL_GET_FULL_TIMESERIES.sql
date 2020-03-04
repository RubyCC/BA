SELECT T.date,
       AVG(T.vader_sentiment) AS AVG,
       COUNT(T.vader_sentiment) AS COUNT,
       S.adj_close AS STOCK
  FROM tweets AS T
       LEFT JOIN
       searches AS SE ON SE.search_id = T.search_id
       LEFT JOIN
       stocks AS S ON SE.symbol = S.symbol AND 
                      S.date = T.date
 WHERE (T.flag_faulty IS NULL OR 
        T.flag_faulty = 0) AND 
       T.date >= ? AND 
       T.date <= ? AND 
       SE.symbol = ?
 GROUP BY T.date,
          SE.symbol