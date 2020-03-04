UPDATE tweets
    SET flag_faulty = 1
    WHERE text_len > 280