# detection-of-targets-with-unknown-strength-hyperspectral-images
simulation of article Veritas: an admissible detector for targets of unknown strength by James Theiler

https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11727/117270B/Veritas-an-admissible-detector-for-targets-of-unknown-strength/10.1117/12.2586125.short?SSO=1

# Summary
An approach to target detection is proposed based on a likelihood ratio test in which the unknown target strength is treated as known,
with strength chosen to correspond to a minimal level of detectability. For Gaussian distributions, this strength
typically corresponds to two or three sigmas. This detector is admissible, which means that there is no other
detector that is uniformly superior to it. The simplicity of the veritas detector permits closed-form solutions to
be derived for a variety of signal detection problems. In a series of numerical experiments, these simple detectors
are compared to traditional detectors, such as the locally most powerful detector and the generalized likelihood
ratio test detector.


# Evaluation method
![image](https://user-images.githubusercontent.com/72392859/119237032-6c5a6b00-bb43-11eb-82f7-009f83ed2381.png)

Since we are testing several different algorithms, we need a process to compare these algorithms. We seek a method that evaluates an algorithm in a generic way, i.e., being as independent as possible of the specific nature of the test data. Traditionally, researchers working on target detection test their algorithms on actual scenes with some number of known point targets in the image. This has the benefit of being actual realistic data but has the disadvantage of not characterizing the detection capabilities of the algorithms at all points in the image. However, the false alarm behavior is typically well captured by such a system. We used an alternative measure of merit for point target detection algorithms based on an exhaustive characterization of the performance of an algorithm over a given image. This empirical method focuses on classical receiver operating characteristic (ROC) curves. The first step is to run the algorithm under consideration over the complete scene without any targets implanted. This generates a target-absent histogram. Independently and iteratively, we then implant the same target at each pixel in the image (background signature + fraction*target signature) and reevaluate the algorithm. This generates a target-present histogram. we then apply the  algorithms on each pixel without the target and we receive a target-absent histogram. Each of the two histograms is then normalized by the number of pixels evaluated, and cumulative probability histograms are calculated. Subsequently, the complement of the cumulative probability histograms is computed to generate the inverse cumulative probability distributions. the x axis values of these distributions are the values of the different algorithms. For each x axis value, i.e. ,threshold, of the distributions, we obtain a pair of values corresponding to the probability of detection(target-present)and the probability of false alarm(target-absent)  these pairs are used to generate the ROC curve.

![image](https://user-images.githubusercontent.com/72392859/119237057-96139200-bb43-11eb-9be5-d2d6dccf8234.png)

# Simulation guidelines 
I simulate the detectors using python , with the RIT Cooke City, MT  dataset. where the HSI dimensions is 800x200 pixels with 126 bands.
I used several targets to test the detectors for different characteristic target strength(we can see that there is consistency in the value of a_0  )
	
![image](https://user-images.githubusercontent.com/72392859/119237384-30c0a080-bb45-11eb-8b20-dab402882402.png)

I choose the Area under curve (AUC) for a fixed False positive rate (PFR) metric, in the fallowing manner:

![image](https://user-images.githubusercontent.com/72392859/119237406-47ff8e00-bb45-11eb-8952-5d2545648f63.png)

![image](https://user-images.githubusercontent.com/72392859/119237411-5057c900-bb45-11eb-9dc6-8743af62058c.png)

# results summary
![image](https://user-images.githubusercontent.com/72392859/119236953-ef2ef600-bb42-11eb-9a60-6c7d3fc2aba5.png)

# some-roc curves
![image](https://user-images.githubusercontent.com/72392859/119237488-be9c8b80-bb45-11eb-95be-29a7379aff55.png)

![image](https://user-images.githubusercontent.com/72392859/119237496-cceaa780-bb45-11eb-84f9-841182dc6933.png)


