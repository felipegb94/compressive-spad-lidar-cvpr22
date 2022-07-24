
### 08-11-2021

* Multi-step learning rate seems to help squeeze a bit extra performance..

* Increased validation set size significantly, which made training curves more stable. Could probably increase it more.
  * validation set 10*n_tbins
  * Although it does not seem
* Increased mini-batch size significantly too which makes us beat Fourier consistently
  * mini-batch size of 256

### 08-10-2021

* For high flux and very low SBR lowering beta helped get better MAE
  * Increaseing beta can help in 1tol, but hurts mae
* For high flux and low SBR
* MINI BATCH size seems to make a huge impact on the optimization
* 