# base model: Ranking Loss + 1024 + paired decode(e) + 5568(image) + 5568(sketch) + drop(0.3) (margin1 0)

Mon.
part 1: Ranking Loss + 1024 + paired decode(e) + 5568(image) + 5568(sketch) (drop 0.3) (margin1 0) result (32.9 21.4 0.6)
        Ranking Loss + 1024 + paired decode(e) + 5568(image) + 5568(sketch) (drop 0.2) (margin1 0) result (33.8 22.5 0.5)
        Ranking Loss + 1024 + paired decode(e) + 5568(image) + 5568(sketch) (drop 0.1) (margin1 0) result (33.7 22.5 0.5)
        Ranking Loss + 1024 + paired decode(e) + 5568(image) + 5568(sketch) (drop 0.0) (margin1 0) result (34.5 24.0 0.5)

part 2: Ranking Loss + 1024 + paired decode(s) + 5568(image) + 5568(sketch) (drop 0.3) (margin1 0) result (38.3 26.9 0.2)
        Ranking Loss + 1024 + paired decode(s) + 5568(image) + 5568(sketch) (drop 0.2) (margin1 0) result (38.2 26.4 0.4)
        Ranking Loss + 1024 + paired decode(s) + 5568(image) + 5568(sketch) (drop 0.1) (margin1 0) result (38.0 26.1 0.4)
        Ranking Loss + 1024 + paired decode(s) + 5568(image) + 5568(sketch) (drop 0.0) (margin1 0) result (38.2 26.4 0.5)

Tues.
part 1: Ranking Loss + 1024 + paired decode(e+s) + 5568(image) + 5568(sketch) (drop 0.3) (margin1 0) (38.1 26.8 0.1)
        Ranking Loss + 1024 + paired decode(e+s) + 5568(image) + 5568(sketch) (drop 0.2) (margin1 0) (38.3 26.9 0.3)
        Ranking Loss + 1024 + paired decode(e+s) + 5568(image) + 5568(sketch) (drop 0.1) (margin1 0) (38.4 26.8 0.4)
        Ranking Loss + 1024 + paired decode(e+s) + 5568(image) + 5568(sketch) (drop 0.0) (margin1 0) (38.3 27.4 0.4)

part 2: Ranking Loss + 1024 + paired decode(e) + 5568(image) + 5568(sketch) (drop 0.0) (margin1 0) (no orth) (34.4 22.9 0.6)
        Ranking Loss + 1024 + paired decode(s) + 5568(image) + 5568(sketch) (drop 0.0) (margin1 0) (no orth) (38.9 27.1 0.5)
        Ranking Loss + 1024 + paired decode(es) + 5568(image) + 5568(sketch) (drop 0.0) (margin1 0) (no orth) (37.9 26.4 0.5)

part 3: Ranking Loss + 1024 + paired decode(s) + 5568(image) + 5568(sketch) (drop 0.3) (margin1 0) (no orth) (38.0 26.6 0.3)
        Ranking Loss + 1024 + paired decode(s) + 5568(image) + 5568(sketch) (drop 0.2) (margin1 0) (no orth) (38.0 26.8 0.3)
        Ranking Loss + 1024 + paired decode(s) + 5568(image) + 5568(sketch) (drop 0.1) (margin1 0) (no orth) (37.9 26.4 0.3)
        Ranking Loss + 1024 + paired decode(s) + 5568(image) + 5568(sketch) (drop 0.0) (margin1 0) (no orth) (38.9 27.1 0.5)

Wes.

part 1: Ranking Loss + 1024 + paired decode(e) + 5568(image) + 5568(sketch) (drop 0.0) (margin1 1) (no orth) (34.0 22.7 0.6)
        Ranking Loss + 1024 + paired decode(e) + 5568(image) + 5568(sketch) (drop 0.0) (margin1 2) (no orth) (36.1 24.5 0.7)
        Ranking Loss + 1024 + paired decode(e) + 5568(image) + 5568(sketch) (drop 0.0) (margin1 3) (no orth) (35.8 24.1 0.6)
        Ranking Loss + 1024 + paired decode(e) + 5568(image) + 5568(sketch) (drop 0.0) (margin1 4) (no orth) (36.3 24.7 0.6)

Thus.

part 1: Ranking Loss + 1024 + paired decode(s) + 5568(image) + 5568(sketch) (drop 0.0) (margin1 1) (no orth) (38.4 26.8 0.5)
        Ranking Loss + 1024 + paired decode(s) + 5568(image) + 5568(sketch) (drop 0.0) (margin1 2) (no orth) (39.1 26.9 0.5)
        Ranking Loss + 1024 + paired decode(s) + 5568(image) + 5568(sketch) (drop 0.0) (margin1 3) (no orth) (39.8 27.9 0.4)
        Ranking Loss + 1024 + paired decode(s) + 5568(image) + 5568(sketch) (drop 0.0) (margin1 4) (no orth) (39.7 27.9 0.5)

part 2: Ranking Loss + 1024 + paired decode(e+s) + 5568(image) + 5568(sketch) (drop 0.0) (margin1 1) (no orth) (38.3 26.6 0.4)
        Ranking Loss + 1024 + paired decode(e+s) + 5568(image) + 5568(sketch) (drop 0.0) (margin1 2) (no orth) (38.8 27.5 0.5)
        Ranking Loss + 1024 + paired decode(e+s) + 5568(image) + 5568(sketch) (drop 0.0) (margin1 3) (no orth) (39.0 27.2 0.4)
        Ranking Loss + 1024 + paired decode(e+s) + 5568(image) + 5568(sketch) (drop 0.0) (margin1 4) (no orth) (39.5 27.9 0.4)

Fri.

part 1: Ranking Loss + 1024 + paired decode(s) + 4096(image) + 4096(sketch) (drop 0.0) (margin1 4) (no orth)
        Ranking Loss + 1024 + paired decode(s) + 3072(image) + 3072(sketch) (drop 0.0) (margin1 4) (no orth)
        Ranking Loss + 1024 + paired decode(s) + 2048(image) + 2048(sketch) (drop 0.0) (margin1 4) (no orth)
        Ranking Loss + 1024 + paired decode(s) + 1024(image) + 1024(sketch) (drop 0.0) (margin1 4) (no orth)

part 2: Ranking Loss + 512 + paired decode(s) + 512(image) + 512(sketch) (drop 0.0) (margin1 4) (no orth)
        Ranking Loss + 512 + paired decode(e) + 512(image) + 512(sketch) (drop 0.0) (margin1 4) (no orth)
        Ranking Loss + 512 + paired decode(es) + 512(image) + 512(sketch) (drop 0.0) (margin1 4) (no orth)
        Ranking Loss + 1024 + paired decode(s) + 4096(image) + 4096(sketch) (drop 0.0) (margin1 4) (no orth)