# datarobot_ts_helpers package
### A library of helper scripts to support complex time-series modeling using DataRobot AutoTS software

## Authors
#### Justin Swansburg, Jarred Bultema, Jess Lin

## Description
#### The modeling of large scale time-series problems is possible directly within DataRobot software via the GUI or via R or Python modeling APIs. While the software is capable of modeling up to 1 million series per project and applying state of the art modeling techniques, often there is motivation to model aspects of a data science problem across multiple DataRobot projects. Motivation for this may include a desire to externally cluster similar series, apply different data manipulations or corrections, utilize different data sources, apply different differencing strategies, utilize different Feature Derivation Windows, or investigate different Forecast Distance ranges. Regardless of the reasons, internally we have found that performance can often be improved on large or complex time-series use cases by breaking a large, challenging problem into smaller pieces and modeling each of those pieces separately.

#### This is feasible directly using the R or Python modeling APIs, but the challenge quickly becomes one of software engineering and logistics to manage, compare, and store outputs of numerous projects that are part of a single use-case. The purpose of the ts_helpers package is to automate this logistical challenge and allow the DataRobot user to focus on applying different approaches to solve their use case, rather than focusing on the less interesting aspects of the problem.


## Contents

#### This python package contains numerous functions to enable the user to easily scale from one to thousands of DataRobot projects starting with data preparation and continuing through modeling, model evaluation, iterative performance improvements, visualization of results, deployment of models, and serving ongoing predictions.

#### A detailed Table of Contents describes all functions present and the documentation string for each function. Detailed tutorials are also available to demonstrate the use of this ts_helpers package and all of the functions contained within.
