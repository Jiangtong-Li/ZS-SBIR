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

1. Ranking Loss + 1024 + paired decode(e) + 5568(image) + 5568(sketch) (**this night-1**)

2. Siamese Loss + 1024 + paired decode(e) + 5568(image) + 5568(sketch)

3. Ranking Loss + 2048 + paired decode(e) + 5568(image) + 5568(sketch)

    Step: 10500(bs 64)

    Loss: 15.2(Rank); 1.21(image)

    P@200: 29.8%

    mAP@200: 18.4%

    Lambda: 0.5

4. Ranking Loss + 2048 + paired decode(e) + 5568(image) + 5568(sketch)

    Step: 12500(bs 64)

    Loss: 14.0(Rank); 1.20(image)

    P@200: 29.9%

    mAP@200: 18.3%

    Lambda: 0.4

5. Ranking Loss + 1024 + paired decode(s) + 5568(image) + 5568(sketch)

    Step: 12500(bs 64)

    Loss: 13.9(Rank); 0.82(image)

    P@200: 38.3%

    mAP@200: 26.7%

    Lambda: 0.3

6. Ranking Loss + 1024 + paired decode(s+e) + 5568(image) + 5568(sketch)

    Step: 8500(bs 64)

    Loss: 15.3(Rank); 0.87(image)

    P@200: 38.2%

    mAP@200: 26.5%

    Lambda: 0.3

7. Ranking Loss + 2048 + paired decode(e) + 5568(image) + 5568(sketch) (no drop)

    Step: 11500(bs 64)

    Loss: 11.4(Rank); 1.1(image)

    P@200: 32.9%

    mAP@200: 22.0%

    Lambda: 0.4

8. Ranking Loss + 1024 + paired decode(e) + 5568(image) + 5568(sketch) (no drop) (**this night-2**)

9. Ranking Loss + 1024 + paired decode(e) + 5568(image) + 5568(sketch) (margin1=1)

    Step: 8500(bs 64)

    Loss: 15.2(Rank); 1.2(image)

    P@200: 31%

    mAP@200: 19.2%

    Lambda: 0.4

10. Ranking Loss + 2048 + paired decode(s) + 5568(image) + 5568(sketch) (**this night-1**)

11. Ranking Loss + 2048 + paired decode(s+e) + 5568(image) + 5568(sketch) (**this night-2**)
