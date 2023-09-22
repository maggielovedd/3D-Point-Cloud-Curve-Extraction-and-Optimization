3D Point Cloud Curve Extraction and Optimization
===============================

`3D Point Cloud Curve Extraction and Optimization` is designed to derive a representative curve or center line from a noisy 3D point cloud, specifically for point clouds with discernible geometric patterns.

<img src="https://github.com/maggielovedd/3D-Point-Cloud-Curve-Extraction-and-Optimization/blob/main/figure/Figure_1.png" width="800" alt=""> 

Description
-----------

Given a noisy point cloud with an inherent structure or shape, this tool identifies and traces a continuous curve that encapsulates the essence of the shape. It synthesizes 3D point clouds based on predefined shapes, simulates noise to mimic real-world scenarios, thins out the point cloud to emphasize its core structure, ensures the curve's continuity, and uses Bayesian optimization to find the best parameters for curve extraction.

* blue: sample 3d point cloud
* orange: thinned points
* green: sorted points

<img src="https://github.com/maggielovedd/3D-Point-Cloud-Curve-Extraction-and-Optimization/blob/main/figure/curve.gif" width="400" alt="">  <img src="https://github.com/maggielovedd/3D-Point-Cloud-Curve-Extraction-and-Optimization/blob/main/figure/trefoil.gif" width="400" alt=""> 
<img src="https://github.com/maggielovedd/3D-Point-Cloud-Curve-Extraction-and-Optimization/blob/main/figure/sinwave.gif" width="400" alt="">  <img src="https://github.com/maggielovedd/3D-Point-Cloud-Curve-Extraction-and-Optimization/blob/main/figure/torus.gif" width="400" alt=""> 


Key Features
------------

*   **Synthetic Point Cloud Generation:** Generate 3D point clouds based on various predefined shapes.
*   **Noise Simulation:** Introduce synthetic noise to the generated points.
*   **Curve Thinning:** Thin out the point cloud to capture the central line.
*   **Curve Sorting:** Maintain a logical and continuous order in the extracted curve.
*   **Parameter Optimization:** Utilize Bayesian optimization to determine optimal parameters.
*   **Visualization:** Provide real-time and post-process visualization of the point clouds and extracted curves.

Key Parameters
--------------

The curve extraction process heavily relies on the optimization of three crucial parameters. Their appropriate setting can significantly influence the accuracy and quality of the extracted curve from the noisy point cloud.


| Parameter | Description | Effect when Value Increases | Effect when Value Decreases |
| --- | --- | --- | --- |
| `point_cloud_thickness` | Defines the radius around each point to identify its local neighborhood, determining the thickness of the cloud under consideration. | Considers more neighboring points. Might lead to capturing more noise and affecting the accuracy of the localized regression line. | Focuses on a narrower set of points. Could potentially miss crucial data if set too low, but will be more resistant to noise. |
| `sorted_point_distance` | Anticipated distance between subsequent points in the extracted curve. | Results in a curve with larger gaps between its points, potentially missing finer details. | Produces a denser curve with closely spaced points, capturing more details but can be influenced more by noise. |
| `search_ratio` | Modulates the search distance for neighboring points, specified as a ratio of the `sorted_point_distance`. | Expands the search area for neighbors, potentially incorporating unrelated or noisy points into the curve. | Restricts the search area, focusing more on immediate neighbors but might miss essential curve structures. |


These parameters are intricately fine-tuned through Bayesian optimization to generate an extracted curve that is both faithful to the intrinsic shape in the point cloud and resilient against the inherent noise and anomalies.

Optimization
------------

### Objective Function

The objective function evaluates the quality of the processed point cloud based on several criteria. It calculates the objective value based on:

1.  Total length of the sorted points.
2.  Penalty for discontinuities in the sorted points.
3.  Reward based on the number of points in the sorted set.


| Component | Description | Calculation |
| --- | --- | --- |
| Total Length | Sum of distances between consecutive sorted points. | `sum(distance.euclidean(sorted_points[i], sorted_points[i+1]) for i in range(len(sorted_points)-1))` |
| Penalty | Penalizes discontinuities between consecutive sorted points. | `penalty_factor * sum(abs(distance.euclidean(sorted_points[i], sorted_points[i+1]) - avg_distance) for i in range(len(sorted_points)-1))` |
| Reward | Rewards based on the number of sorted points. | `len(sorted_points) * reward_factor` |


The final objective value is a combination of the total length, penalty, and reward:  
`objective_value = -total_length + penalty - reward`

Limitation
------------
This code is primarily designed for open geometric patterns. It is not optimized for closed-loop or partially closed-loop geometries such as circles and mobius. Sorted point sometimes didn't order properly at where the sample points are thick and some part of the geometry close to others.

<img src="https://github.com/maggielovedd/3D-Point-Cloud-Curve-Extraction-and-Optimization/blob/main/figure/mobius.gif" width="600" alt="">  


Getting Started
---------------

### Prerequisites

To run the script, ensure you have the following libraries installed:

*   `numpy`
*   `matplotlib`
*   `mpl_toolkits`
*   `scipy`
*   `sklearn`
*   `vg`
*   `skopt`
*   `signal`

### Installation

```bash
git clone https://github.com/maggielovedd/3D-Point-Cloud-Curve-Extraction-and-Optimization.git
```

Contributing
------------

Contributions are welcome! 

License
-------

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.

Acknowledgements
----------------

  The method is developed based on [3D Point Cloud Curve Extraction](https://github.com/aliadnani/curves).





