# unoffical isofit fork

All code contained within here is a fully functional isofit build that can be rebuilt with pip install -e ./

This code was forked ISOFIT v3.4.1 (2 April 2025). There are several differences but the biggest is that it is built around SnowSurface() class.

There are example scripts under ./scripts/ that can be used to run the model.


## Major notes to remember on this isofit build:
- Sa is currently around Multicomponent Surface which does not make sense for this. We are also not using priors so its not an issue. But we are computing them every time. I'm going to turn them off for the time being just to save compute time.

- posterior uncert calc: I've gone back and forth on using , `S_hat = np.linalg.pinv(K.T @ K)` vs. `S_hat = np.linalg.pinv(K.T.dot(Seps_inv).dot(K) + Sa_inv)`. The former is what is currently active. In more testing of the latter, Seps interaction is making Shat way too certain.. This sort of linearization of just the forward model (first) is still an underestimation because it treats this uncertainty in the forward model as the uncertainty. But is still better than the alternative. To be revisited. 

- white/black/blue sky albedos are computed on the fly just using the rtm range of wl... It will be up to what the range of the RTM is? But todo soon need to confirm this does this automatically.
