### Speeding up

As far as we made an assumption of frames interindependence, video likelihood P(F_m | l_m) can be splitted into product of frame likelihoods given current label assignment. 

![alt text](https://github.com/Annusha/slim_mallow/blob/master/results/1.png)

Authors of the original paper modeled likelihood for each video F_m based on the entire video collection, but excluded the very same video.  Due to the definition of Gibbs sampling procedure, the probability of the video to have respective sub-activity frame-wise assignments is conditioned on the segmentation of the rest of the video collection. 
To obtain the exact values of likelihood for every frame, the Gaussian Mixture Model is fitted for each video separately.

![alt text](https://github.com/Annusha/slim_mallow/blob/master/results/2.png)

where Q is empirically defined number of components. 
Whereas we train the embedding to form distinct sub-activity clusters given the initial resp. updated segmentation at the beginning of each iteration, the exclusion of all frames belonging to the video provided that the accumulation of the rest frames does not affect the statistics of fitted gaussians noticeably. For the simplicity and quick computations we streamline the model and keep just one Gaussian Mixture Model per sub-action instead of N_m mixtures that give us

![alt text](https://github.com/Annusha/slim_mallow/blob/master/results/3.png)

