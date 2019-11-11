# Result of models

## CVAE

1. Paired + 4096D + Tanh in Varience Sample:

    Step: 27500(bs 64)

    Loss: 0.17 +- 0.05(KL); 0.24 +- 0.02(recon); 0.011 +- 0.001(srecon)

    P@200 - 31.1%

    mAP@200 - 20.4%

2. Unpaired + 4096D + Tanh in Varience Sample:

    Step: 27500(bs 64)

    Loss: 0.14 +- 0.05(KL); 0.31 +- 0.02(recon); 0.013 +- 0.001(srecon)

    P@200 - 28.5%

    mAP@200 - 18%

3. Paired + 5568D + Tanh in Varience Sample:

    Step: 14500(bs 64)

    Loss: 0.22 +- 0.05(KL); 0.22 +- 0.02(recon); 0.020 +- 0.001(srecon)

    P@200 - 27.8%

    mAP@200 - 16.6%

4. Unpaired + 5568D + Tanh in Varience Sample:

    Step: 11500(bs 64)

    Loss: 0.37 +- 0.05(KL); 0.27 +- 0.02(recon); 0.021 +- 0.001(srecon)

    P@200 - 22.6%

    mAP@200 - 12.1%

5. Paired + 4096D + no Tanh in Varience Sample(paper's data):

    Step: 19000

    Loss: 0.20 +- 0.05(KL); 1.1 +- 0.5(recon); 0.058 +- 0.001(srecon)

    P@200 - 31.8%

    mAP@200 - 20.7%

6. Unpaired + 4096D + Tanh in Varience Sample(paper's data):

    Step: 14000(bs 64)

    Loss: 0.35 +- 0.05(KL); 1.5 +- 0.5(recon); 0.051 +- 0.001(srecon)

    P@200 - 29.1%

    mAP@200 - 18.6%

7. Paired + 4096D + Tanh in Varience Sample(paper's data):

    Step: 14000(bs 64)

    Loss: 0.50 +- 0.05(KL); 1.2 +- 0.5(recon); 0.051 +- 0.001(srecon)

    P@200 - 32.3%

    mAP@200 - 21.2%

8. Paired + 4096D + no Tanh in Varience Sample(paper's data update image):

    Step: 22500(bs 64)

    Loss: 0.14 +- 0.05(KL); 1.1 +- 0.05(recon); 0.047 +- 0.001(srecon)

    P@200 - 33.6%

    mAP@200 - 22.5%

9. Paired + 4096D + no Tanh in Varience Sample + no sketch recon(paper's data update image): (converge faster)

    Step: 12000(bs 64)

    Loss: 0.216 +- 0.05(KL); 1.1 +- 0.05(recon)

    P@200 - 33.3%

    mAP@200 - 22.5%

10. Unpaired + 4096D + Tanh in Varience Sample(paper's data updated):

    Step: 25000(bs 64)

    Loss: 0.25 +- 0.05(KL); 1.5 +- 0.05(recon); 0.045 +- 0.001(srecon)

    P@200 - 29.3%

    mAP@200 - 19.1%

11. Paired + 4096D + Tanh in Varience Sample(paper's data updated):

    Step: 16500(bs 64)

    Loss: 0.32 +- 0.05(KL); 1.2 +- 0.05(recon); 0.050 +- 0.01(srecon)

    P@200 - 33.8%

    mAP@200 - 22.5%

12. Unpaired + 5568D + Tanh in Varience Sample(updated):

    Step: 14000(bs 64)

    Loss: 0.43 +- 0.05(KL); 0.93 +- 0.05(recon); 0.040 +- 0.001(srecon)

    P@200 - 35.7%

    mAP@200 - 24.5%

13. Paired + 5568D + Tanh in Varience Sample(updated):

    Step: 16500(bs 64)

    Loss: 0.36 +- 0.05(KL); 0.9 +- 0.05(recon); 0.041 +- 0.01(srecon)

    P@200 - 36.8%

    mAP@200 - 25.3%

## Siamese-1

1. Paired + 5568D(updated):

    Step: 2500(bs 64)

    Loss: 12.9 +- 0.1(Margin=10)

    P@200 - 16.2%

    mAP@200 - 7.38%

2. Unpaired + 5568(updated):

    Step: 10500(bs 64)

    Loss: 12.8 +- 0.1(Margin=10)

    P@200 - 28.1%

    mAP@200 - 16.1%

3. Paired + 5568D:

    Step: 2500(bs 64)

    Loss: 4.43 +- 0.01(Margin=10)

    P@200 - 10.9%

    mAP@200 - 3.58%

4. Unpaired + 5568D:

    Step: 15500(bs 64)

    Loss: 13.5 +- 0.1(Margin=10)

    P@200 - 26.8%

    mAP@200 - 13.1%

5. Unpaired + 4096D(paper's data updated):

    Step: 25500(bs 64)

    Loss: 12.5 +- 0.1(Margin=10)

    P@200 - 27.5%

    mAP@200 - 16.5%

6. Paired + 4096D(paper's data updated):

    Step: 3000(bs 64)

    Loss: 13.0 +- 0.1(Margin=10)

    P@200 - 20.4%

    mAP@200 - 9.74%

## Cross-Modal Domain Translation

* Ranking Loss + 1024 + paired decode(e) + 5568(image) + 5568(sketch)

    Step: 14000(bs 64)

    Loss: 13.6(Rank); 1.17(image)

    P@200: 30.4%

    mAP@200: 19.0%

    Lambda: 0.3

* Ranking Loss + 1024 + paired decode(e) + 5568(image) + 5568(sketch) (no drop)

    Step: 8500(bs 64)

    Loss: 12.2(Rank); 1.11(image)

    P@200: 33.7%

    mAP@200: 21.2%

    Lambda: Com

* Ranking Loss + 1024 + paired decode(e) + 5568(image) + 5568(sketch) (margin1=1)

    Step: 8500(bs 64)

    Loss: 15.2(Rank); 1.2(image)

    P@200: 31%

    mAP@200: 19.2%

    Lambda: 0.4

* Ranking Loss + 1024 + paired decode(e) + 5568(image) + 5568(sketch) (nodrop; margin1=1)

    Step: 20000(bs 64)

    Loss: 9.2(Rank); 1.0(image)

    P@200: 34.2%

    mAP@200: 23.4%

    Lambda: 0.5

* Siamese Loss + 1024 + paired decode(e) + 5568(image) + 5568(sketch)

* Ranking Loss + 1024 + paired decode(s) + 5568(image) + 5568(sketch)

    Step: 12500(bs 64)

    Loss: 13.9(Rank); 0.82(image)

    P@200: 38.3%

    mAP@200: 26.7%

    Lambda: 0.3

* Ranking Loss + 1024 + paired decode(s) + 5568(image) + 5568(sketch) (no drop)

    Step: 11500(bs 64)

    Loss: 11.3(Rank); 0.83(image)

    P@200: 39.2%

    mAP@200: 27.3%

    Lambda: 0.4

* Ranking Loss + 1024 + paired decode(s) + 5568(image) + 5568(sketch) (m1)

    Step: 11500(bs 64)

    Loss: 12.4(Rank); 0.71(image)

    P@200: 38.2%

    mAP@200: 26.8%

    Lambda: 0.3

* Ranking Loss + 1024 + paired decode(s) + 5568(image) + 5568(sketch) (no drop; m1)

    Step: 17000(bs 64)

    Loss: 9.7(Rank); 0.75(image)

    P@200: 39.1%

    mAP@200: 27.1%

    Lambda: 0.4

* Ranking Loss + 1024 + paired decode(s+e) + 5568(image) + 5568(sketch)

    Step: 8500(bs 64)

    Loss: 15.3(Rank); 0.87(image)

    P@200: 38.2%

    mAP@200: 26.5%

    Lambda: 0.3

* Ranking Loss + 1024 + paired decode(s+e) + 5568(image) + 5568(sketch) (no drop)

    Step: 14000(bs 64)

    Loss: 10.2(Rank); 0.78(image)

    P@200: 38.4%

    mAP@200: 27.2%

    Lambda: 0.3

* Ranking Loss + 1024 + paired decode(s+e) + 5568(image) + 5568(sketch) (m1)

    Step: 11000(bs 64)

    Loss: 14.1(Rank); 0.82(image)

    P@200: 38.6%

    mAP@200: 26.7%

    Lambda: 0.3

* Ranking Loss + 1024 + paired decode(s+e) + 5568(image) + 5568(sketch) (no drop; m1)

    Step: 17000(bs 64)

    Loss: 9.1(Rank); 0.76(image)

    P@200: 38.6%

    mAP@200: 27.2%

    Lambda: 0.3

* Ranking Loss + 1024 + paired decode(e) + 5568(image) + 5568(sketch) (deeper, dropout0.3 ***Since there are two dropout layer, the dropout number is equal to 0.51***)

    Step: 31500(bs 64)

    Loss: 8.7(Rank); 1.06(image)

    P@200: 30.4%

    mAP@200: 19.4%

    Lambda: 0.5

* Ranking Loss + 1024 + paired decode(e) + 5568(image) + 5568(sketch) (deeper, dropout0.16 ***Since there are two dropout layer, the dropout number is equal to 0.3***)

    Step: (bs 64)

    Loss: (Rank); (image)

    P@200: %

    mAP@200: %

    Lambda: 0.

* Ranking Loss + 2048 + paired decode(s) + 5568(image) + 5568(sketch)

    Step: 10500(bs 64)

    Loss: 14.6(Rank); 0.85(image)

    P@200: 37.6%

    mAP@200: 25.9%

    Lambda: 0.3

* Ranking Loss + 2048 + paired decode(s+e) + 5568(image) + 5568(sketch)

    Step: 24000(bs 64)

    Loss: 12.3(Rank); 0.75(image)

    P@200: 38%

    mAP@200: 26.8%

    Lambda: 0.3

* Ranking Loss + 2048 + paired decode(e) + 5568(image) + 5568(sketch)

    Step: 10500(bs 64)

    Loss: 15.2(Rank); 1.21(image)

    P@200: 29.8%

    mAP@200: 18.4%

    Lambda: 0.5

* Ranking Loss + 2048 + paired decode(e) + 5568(image) + 5568(sketch)

    Step: 12500(bs 64)

    Loss: 14.0(Rank); 1.20(image)

    P@200: 29.9%

    mAP@200: 18.3%

    Lambda: 0.4

* Ranking Loss + 2048 + paired decode(e) + 5568(image) + 5568(sketch) (no drop)

    Step: 11500(bs 64)

    Loss: 11.4(Rank); 1.1(image)

    P@200: 32.9%

    mAP@200: 22.0%

    Lambda: 0.4
