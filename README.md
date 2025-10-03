CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Harris Kokkinakos
  * [LinkedIn](https://www.linkedin.com/in/haralambos-kokkinakos-5311a3210/), [personal website](https://harriskoko.github.io/Harris-Projects/)
* Tested on: Windows 24H2, i9-12900H @ 2.50GHz 16GB, RTX 3070TI Mobile

### Description

This project is a physically based Monte Carlo path tracer implemented in CUDA. It simulates global illumination by tracing rays through a scene, recursively scattering and absorbing light to approximate the rendering equation. The focus of this renderer is on leveraging GPU parallelism to achieve realistic images at interactive to near-interactive speeds. This enables efficient and physically realistic rendering of scenes. 

This path tracer supports:
*Diffuse, reflective, and refractive materials with multiple BSDFs.
*Intersection handling for spheres, boxes, and triangles
*GTLF Mesh loading for creation of complex scenes.
*BVH acceleration structure for efficient ray–scene intersection.
*Stream compaction to remove terminated rays and improve warp coherence.
*Sorting of material types in memory to reduce GPU memory lookups
*Cosine weighted hemisphere sampling.
*Stochastic sampled antialiasing 

### Results
Bathtub Scene
![Bathtub](img/bath1.png)
![Bathtub](img/bath2.png)

Geometry Scenes
![Bathtub](img/medium.png)
![Bathtub](img/medium2.png)

### Implementation
