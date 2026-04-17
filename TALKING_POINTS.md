# EuCAP 2026 Presentation — Talking Points

**Paper:** Machine Learning-Based Anchor-Aware Conditional Flow Matching for RF Localization in Wireless Capsule Endoscopy

**Total time:** approximately 12 to 15 minutes across 15 slides

---

## Slide 1: Title Slide
**Duration: about 10 seconds**

"Good morning everyone. My name is Subramaniam Murugesan, and I am from Queen Mary University of London. Today I will present our work on using a generative machine learning method called conditional flow matching to localize a wireless capsule endoscope inside the human body using radio frequency signals."

---

## Slide 2: Motivation — Why Localize a Wireless Capsule?
**Duration: about 60 seconds**

"So why do we need to localize a capsule inside the body? Wireless capsule endoscopy is a procedure where the patient swallows a tiny camera — you can see the PillCam here on the right, it is only 11 millimeters wide and 27 millimeters long. This capsule travels through the entire gastrointestinal tract and captures thousands of images along the way.

Now, the clinical problem is this: when the capsule finds something important — say a polyp or a bleeding lesion — the doctor needs to know exactly WHERE inside the body that finding is located. Without knowing the position, the images are useful for diagnosis but the spatial context is completely lost. The doctor cannot plan a targeted intervention.

Radio frequency based localization is an attractive solution because it is non-ionizing, meaning it is safe for the patient, it does not require line-of-sight, and it can be built into a simple wearable garment.

However, the challenges are significant. Radio signals travel through different tissue layers causing multipath propagation. Every patient has a different body size and composition. The capsule rotates unpredictably inside the gut, which changes its radiation pattern. And the body-worn sensors can shift position.

Our contribution is a new framework called anchor-aware conditional flow matching that gives us not just a position estimate, but also tells us how confident we are in that estimate."

---

## Slide 3: Prior Work and Positioning
**Duration: about 45 seconds**

"Let me briefly survey the traditional approaches to this problem.

First, trilateration — this uses range circles from multiple anchors to find the intersection point. The problem is that it needs at least three good range estimates and it is very sensitive to measurement noise.

Second, fingerprinting methods like K-nearest neighbors — these match the observed signal pattern against a pre-recorded database. But the database is a discrete grid, so there is no smooth interpolation between stored points.

Third, tree-based methods like XGBoost and Random Forest — these are powerful but they make axis-aligned splits in the feature space. They cannot produce a smooth localization surface.

Fourth, deep neural network regression — a multilayer perceptron that directly outputs a position coordinate. This gives a single point prediction, but crucially, it tells you nothing about how confident that prediction is.

The common limitation across all four approaches: none of them provide uncertainty quantification. You get one answer, but you do not know whether to trust it."

---

## Slide 4: Our Approach — Conditional Flow Matching
**Duration: about 100 seconds**

"This slide shows how our approach is fundamentally different. Instead of predicting a single point, we learn what is called a velocity field — think of it as a set of arrows that tells particles which direction to move.

Watch the top animation on the left. At time zero, the green dots start scattered randomly inside the body — this is our prior, meaning we initially have no idea where the capsule is. The velocity field, trained using the radio frequency measurements, guides these particles along the gray arrows. As time progresses from zero to one, the particles converge toward the true capsule position — the red dot. The final spread of the particle cluster IS our uncertainty estimate — a tight cluster means high confidence; a diffuse cluster means ambiguous measurements.

Now the mathematics on the right. Rather than directly regressing position x, we learn a time-dependent velocity field f-theta conditioned on the radio frequency observations y. This defines an ordinary differential equation that transports particles from the prior at time zero toward the observation-conditioned posterior at time one.

During training we use a straight-line bridge. We take a random starting point sampled uniformly inside the cylinder and the known ground truth capsule position, and we linearly interpolate between them. The target velocity the network must learn is simply the vector from the random start to the ground truth — constant along the bridge. The network predicts both the mean velocity and a per-axis variance using a heteroscedastic loss, which lets the model learn position-dependent uncertainty — predictions near the center of the cylinder where all three antenna rings provide strong coverage will have low variance, while predictions near the boundary will have higher variance.

At inference we integrate the ODE on N particles using 50 to 100 Euler-Maruyama steps, injecting the learned noise sigma-theta to capture aleatoric uncertainty from measurement ambiguity. The bottom-left 3D visualization shows a representative tracking scenario — the gray dashed line is the ground truth trajectory through the phantom, and the blue line is our predicted trajectory through all 10 waypoints. The sample mean gives the position estimate, and the sample covariance gives calibrated per-axis uncertainty — something traditional regression simply cannot provide."

---

## Slide 5: System Overview
**Duration: about 45 seconds**

"Let me describe our physical setup. We use 9 body-worn anchor antennas arranged on 3 staggered rings positioned at 10, 20, and 30 centimeters height around a cylindrical body phantom. Each ring has 3 antennas spaced 120 degrees apart, and the middle ring is offset by 60 degrees to maximize spatial diversity.

On the top left you can see the radiation patterns of our antennas at 610 and 2400 megahertz. On the top right is a photograph of the actual wearable vest prototype built by our group. Below you can see the three-dimensional simulation setup and an exploded view of the capsule showing all its internal components.

From these 9 anchor measurements, we extract an 83-dimensional feature vector that includes: the received signal strength at each anchor, pairwise signal strength differences between anchors, ring-level statistics, time difference of arrival relative to a reference anchor, and angle of arrival in both azimuth and elevation."

---

## Slide 6: Antenna Designs and S-Parameters
**Duration: about 30 seconds**

"Since this is the European Conference on Antennas and Propagation, let me show you the antenna hardware in detail.

On the left in panel (a), you see the on-body anchor antenna — this is a compact printed slot antenna designed by Qamar and colleagues, optimized for wideband operation covering the Medical Implant Communication Service band and the Industrial, Scientific and Medical band. It is designed to sit comfortably on the body surface.

In the center, panel (b) shows the capsule antenna — an offset dipole encapsulated in a biocompatible PVC shell. This design fits within the 11 by 27 millimeter capsule form factor and provides good deep-implant propagation characteristics.

On the right in panel (c), the S-parameter measurements: S11 shows the anchor return loss is below minus 20 decibels at 403 megahertz, S22 confirms the capsule is well-matched at the MICS band, and S21 shows the in-body transmission coupling between the capsule and anchor. All of this was simulated using the CST full-wave electromagnetic solver."

---

## Slide 7: RSSI Distributions Across Anchors
**Duration: about 30 seconds**

"This figure shows what the model actually sees — the received signal strength distributions for each of the 9 anchors, color-coded by ring.

Ring 1 in blue, which is at the top of the cylinder, shows higher attenuation and wider spread. Ring 2 in orange, at the middle, provides a complementary view thanks to its azimuthal offset. Ring 3 in green, at the bottom, shows the strongest signals with the narrowest distributions.

The important point is that these systematic differences between rings encode three-dimensional position information. When the capsule moves closer to one ring, the signal strengths at those anchors change in a predictable way. Our model learns to decode these patterns into spatial coordinates."

---

## Slide 8: Anchor-Aware Encoder Architecture
**Duration: about 45 seconds**

"This diagram shows the complete encoder architecture.

Starting from the left: each of the 9 anchors is represented as a token — a vector containing 14 features including its signal strength, time difference of arrival, angle of arrival, known three-dimensional coordinates, ring index, and azimuthal encoding.

These 9 tokens are fed into a multi-head self-attention layer with 4 attention heads. This is the key design choice — self-attention allows the model to compare all pairs of anchors simultaneously. It learns which anchors are most informative for any given capsule position, and it naturally handles situations where some anchors have degraded signals.

The attention output goes through RSS-weighted pooling, which aggregates the 9 anchor vectors into a single context vector, down-weighting anchors with unreliable signal strength.

This is then concatenated with global features like pairwise signal strength differences and ring-level statistics, along with a spatial encoding of the current particle position and a Fourier time embedding. Everything feeds into a 6-layer residual multilayer perceptron with 512 hidden units, which outputs the velocity field prediction."

---

## Slide 9: Trajectory Error Analysis
**Duration: about 30 seconds**

"Looking at the prediction accuracy at each point along this trajectory, the mean error is 1.66 centimeters.

The error is lowest at mid-height positions — around points 3, 4, and 8 in the plot — where all three antenna rings provide strong vertical diversity. The error increases near the cylinder boundary and at the top and bottom extremes, which is expected because the geometric dilution of precision is worse when the capsule has fewer nearby anchors.

The capsule's orientation also plays a role — when the dipole radiation pattern has a null pointing toward certain anchors, those measurements become less informative, causing local increases in error."

---

## Slide 10: Results — Overall Performance
**Duration: about 60 seconds**

"Now the headline results.

The cumulative distribution function on the left shows the full error distribution. 17.5 percent of predictions are within 1 centimeter. 71.6 percent are within 2 centimeters. 97 percent are within 3 centimeters. And critically, 100 percent of all predictions fall within the 5 centimeter clinical usability threshold.

The bar chart on the right compares the per-axis accuracy of our method against a standard regression baseline. Our conditional flow matching achieves 0.79, 0.80, and 0.86 centimeters on the x, y, and z axes respectively. These are remarkably close to each other — within just 0.07 centimeters — which confirms the near-isotropic coverage provided by our 3-ring anchor geometry. The baseline regression is significantly worse on all three axes.

The overall mean localization error is 1.94 centimeters with a standard deviation of 0.77 centimeters.

To put this in context: ultra-wideband based methods in the literature report 1 to 4 millimeter accuracy, but they require specialized hardware and dedicated bandwidth. Standard received signal strength methods typically achieve only 5 to 10 centimeters. Our approach achieves sub-2 centimeter accuracy using standard radio frequency features, with the added benefit of built-in uncertainty quantification that no other method provides."

---

## Slide 11: Robustness — Varying Body Size
**Duration: about 30 seconds**

"An important practical question is: does this model work for different patients with different body sizes?

To test this, we swept the phantom diameter from 25 to 40 centimeters, which covers the range of human abdominal dimensions. The plot shows that error remains stable across the entire range, with only slight increases at the smallest and largest extremes. The gray envelope shows the full distribution.

The clinical implication is significant: a single trained model can handle diverse patient body sizes without any per-patient re-training or calibration."

---

## Slide 12: Robustness — Varying Tissue Properties
**Duration: about 30 seconds**

"Similarly, we tested what happens when the tissue permittivity changes. Permittivity directly affects how radio waves propagate inside the body — it changes their speed, attenuation, and wavelength.

We swept the relative permittivity from 10 to 80, which covers the range from fatty tissue at the low end to muscle and organ tissue at the high end. The model shows graceful degradation with no catastrophic failures at any setting.

This validates that a single trained model generalizes beyond its training conditions — which is essential because real patients have heterogeneous tissue composition, not uniform phantom material."

---

## Slide 13: Spatial Error Analysis
**Duration: about 30 seconds**

"This analysis reveals WHERE in the phantom the errors occur.

The top-down view on the left confirms lower errors near the center of the cylinder where the anchor geometry provides strong coverage from all directions. Errors increase near the cylinder walls where the geometric dilution of precision is worse.

The side view shows the same pattern along the height axis — best accuracy near the middle ring height, degrading toward the top and bottom.

The three main error sources are: geometric dilution of precision at the boundaries, capsule orientation effects where the dipole null aligns with specific anchors, and correlated shadowing that creates localized signal blind spots."

---

## Slide 14: Key Takeaways
**Duration: about 45 seconds**

"To summarize, we make five technical contributions in this work.

First, this is the first application of conditional flow matching — a generative modeling technique — to radio frequency based wireless capsule endoscopy localization.

Second, we introduce an anchor-aware architecture that uses per-anchor tokens with cross-anchor self-attention, allowing the model to learn which sensors are most informative at each position.

Third, all flow trajectories are geometrically constrained to remain within the cylindrical body domain, ensuring physically plausible predictions.

Fourth, the heteroscedastic velocity head provides calibrated, position-dependent uncertainty — the model knows where it is confident and where it is not.

And fifth, we validate robustness across a range of body sizes and tissue properties, demonstrating practical generalizability.

The bottom line: 1.94 centimeters mean error, 100 percent of predictions within the 5 centimeter clinical threshold, and near-isotropic per-axis accuracy of approximately 0.8 centimeters — all with built-in uncertainty quantification.

For future work, we plan to validate on real measured phantom data and eventually in-vivo data, incorporate patient-specific priors, extend the orientation and antenna modeling, and develop a real-time implementation with uncertainty-aware clinical decision logic."

---

## Slide 15: Thank You
**Duration: about 5 seconds**

"Thank you very much for your attention. I am happy to take any questions. My contact details are on the screen."

---

## Anticipated Questions and Prepared Answers

**Question: How does this compare to ultra-wideband localization?**

"Ultra-wideband achieves 1 to 4 millimeter accuracy, which is impressive. However, it requires specialized hardware, dedicated bandwidth allocation, and does not naturally provide uncertainty estimates. Our approach uses standard radio frequency features from body-worn antennas operating in the Medical Implant Communication Service band, and it provides calibrated uncertainty as a built-in feature of the generative modeling framework."

**Question: Have you validated this on real measurements, not just simulations?**

"This work uses a comprehensive synthetic dataset with realistic propagation effects — including correlated shadowing, Rician fading, capsule radiation patterns, and anchor calibration jitter — validated against CST full-wave electromagnetic simulations. Phantom measurement validation is our immediate next step, and we have the hardware setup already developed in our group."

**Question: Why use flow matching instead of a simpler regression approach?**

"The fundamental advantage is that flow matching gives us the full posterior distribution over possible capsule positions, not just a single point estimate. In a clinical setting, knowing that you are 90 percent confident the capsule is within a 1 centimeter region is far more useful than just getting a coordinate with no confidence measure. Standard regression cannot provide this."

**Question: What is the computational cost? Can this run in real time?**

"Inference takes approximately 600 milliseconds for a single position estimate using 50 integration steps on standard hardware. This is well within real-time requirements for capsule tracking, since the capsule moves slowly through the gastrointestinal tract — typically a few centimeters per minute."

**Question: How sensitive is the model to the number of anchors?**

"The self-attention mechanism naturally handles degraded or missing anchors by learning to down-weight them through the RSS-weighted pooling. We have tested with subsets of anchors and performance degrades gracefully rather than failing catastrophically. This is one of the practical advantages of the anchor-aware architecture."

**Question: Why a cylindrical phantom and not a realistic body model?**

"The cylindrical phantom allows controlled, reproducible experiments and is consistent with prior work in the wireless capsule endoscopy literature. Our robustness experiments across varying diameters and permittivities demonstrate that the approach generalizes. Extending to anatomically realistic body models is part of our future work plan."
