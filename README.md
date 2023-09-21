3D Point Cloud Curve Extraction and Optimization
===============================

`3D Point Cloud Curve Extraction` is a tool designed to derive a representative curve or center line from a noisy 3D point cloud, specifically for point clouds with discernible geometric patterns.

Description
-----------

Given a noisy point cloud with an inherent structure or shape, this tool identifies and traces a continuous curve that encapsulates the essence of the shape. It synthesizes 3D point clouds based on predefined shapes, simulates noise to mimic real-world scenarios, thins out the point cloud to emphasize its core structure, ensures the curve's continuity, and uses Bayesian optimization to find the best parameters for curve extraction.


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
*   `signal` (Note: compatible with UNIX platforms. Windows users might need an alternative approach for timeout handling.)

### Installation

```bash
git clone https://github.com/your-username/3DPointCurveExtractor.git
```

Contributing
------------

Contributions are welcome! 

License
-------

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.

Acknowledgements
----------------

  Thanks to the open-source community for providing insights and tools that contributed to this project.





