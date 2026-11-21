| Setting | N | Tasks done | Of |
|---|---|---|---|
| mnist-fresh | 2 | 240 | 240 |
| mnist-trained | 2 | 240 | 240 |
| mnist-fresh | 4 | 240 | 240 |
| mnist-trained | 4 | 240 | 240 |

| Setting | N | E | Split | Attacker | Releases | Mean SSIM | Any image SSIM≥0.9 | Median drift | Mean s/release |
|---|---|---|---|---|---|---|---|---|---|
| mnist-fresh | 2 | 1 | 1x2 | cosine | 30 | 0.859 | 87% | 1.000 | 22 |
| mnist-fresh | 2 | 1 | 1x2 | lbfgs | 30 | 0.950 | 100% | 1.000 | 14 |
| mnist-fresh | 2 | 5 | 1x2 | cosine | 30 | 0.182 | 10% | 0.853 | 22 |
| mnist-fresh | 2 | 5 | 1x2 | lbfgs | 30 | 0.037 | 0% | 0.853 | 45 |
| mnist-fresh | 2 | 5 | 1x2 | unrolled | 30 | 0.410 | 10% | 0.853 | 99 |
| mnist-fresh | 2 | 5 | 2x1 | cosine | 30 | 0.856 | 87% | 0.853 | 22 |
| mnist-fresh | 2 | 5 | 2x1 | lbfgs | 30 | 0.049 | 0% | 0.853 | 43 |
| mnist-fresh | 2 | 5 | 2x1 | unrolled | 30 | 0.855 | 90% | 0.853 | 128 |
| mnist-fresh | 4 | 1 | 1x4 | cosine | 30 | 0.780 | 77% | 1.000 | 38 |
| mnist-fresh | 4 | 1 | 1x4 | lbfgs | 30 | 0.942 | 100% | 1.000 | 42 |
| mnist-fresh | 4 | 5 | 1x4 | cosine | 30 | 0.043 | 0% | 0.735 | 39 |
| mnist-fresh | 4 | 5 | 1x4 | lbfgs | 30 | 0.034 | 0% | 0.735 | 83 |
| mnist-fresh | 4 | 5 | 1x4 | unrolled | 30 | 0.148 | 0% | 0.735 | 178 |
| mnist-fresh | 4 | 5 | 4x1 | cosine | 30 | 0.775 | 80% | 0.735 | 38 |
| mnist-fresh | 4 | 5 | 4x1 | lbfgs | 30 | 0.049 | 0% | 0.735 | 83 |
| mnist-fresh | 4 | 5 | 4x1 | unrolled | 30 | 0.778 | 80% | 0.735 | 270 |
| mnist-trained | 2 | 1 | 1x2 | cosine | 30 | 0.630 | 50% | 1.000 | 22 |
| mnist-trained | 2 | 1 | 1x2 | lbfgs | 30 | 0.397 | 43% | 1.000 | 24 |
| mnist-trained | 2 | 5 | 1x2 | cosine | 30 | 0.531 | 30% | 0.995 | 21 |
| mnist-trained | 2 | 5 | 1x2 | lbfgs | 30 | 0.238 | 3% | 0.995 | 18 |
| mnist-trained | 2 | 5 | 1x2 | unrolled | 30 | 0.355 | 10% | 0.995 | 98 |
| mnist-trained | 2 | 5 | 2x1 | cosine | 30 | 0.648 | 60% | 0.995 | 21 |
| mnist-trained | 2 | 5 | 2x1 | lbfgs | 30 | 0.188 | 3% | 0.995 | 20 |
| mnist-trained | 2 | 5 | 2x1 | unrolled | 30 | 0.697 | 63% | 0.995 | 126 |
| mnist-trained | 4 | 1 | 1x4 | cosine | 30 | 0.439 | 27% | 1.000 | 40 |
| mnist-trained | 4 | 1 | 1x4 | lbfgs | 30 | 0.360 | 43% | 1.000 | 72 |
| mnist-trained | 4 | 5 | 1x4 | cosine | 30 | 0.351 | 10% | 0.944 | 40 |
| mnist-trained | 4 | 5 | 1x4 | lbfgs | 30 | 0.194 | 0% | 0.944 | 68 |
| mnist-trained | 4 | 5 | 1x4 | unrolled | 30 | 0.281 | 3% | 0.944 | 185 |
| mnist-trained | 4 | 5 | 4x1 | cosine | 30 | 0.478 | 37% | 0.944 | 40 |
| mnist-trained | 4 | 5 | 4x1 | lbfgs | 30 | 0.210 | 3% | 0.944 | 48 |
| mnist-trained | 4 | 5 | 4x1 | unrolled | 30 | 0.492 | 30% | 0.944 | 278 |
