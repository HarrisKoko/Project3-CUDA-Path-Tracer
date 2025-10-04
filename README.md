CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Harris Kokkinakos
  * [LinkedIn](https://www.linkedin.com/in/haralambos-kokkinakos-5311a3210/), [personal website](https://harriskoko.github.io/Harris-Projects/)
* Tested on: Windows 24H2, i9-12900H @ 2.50GHz 16GB, RTX 3070TI Mobile

### Description

This project is a physically based Monte Carlo path tracer implemented in CUDA. It simulates global illumination by tracing rays through a scene, recursively scattering and absorbing light to approximate the rendering equation. The focus of this renderer is on leveraging GPU parallelism to achieve realistic images at interactive to near-interactive speeds. This enables efficient and physically realistic rendering of scenes. 

This path tracer supports:
*Light Transport according Diffuse, reflective, and refractive materials.
*Intersection handling for spheres, boxes, and triangles.
*GTLF Mesh loading for creation of complex scenes.
*BVH acceleration structure for efficient ray–scene intersection.
*Cosine weighted hemisphere sampling.
*Stochastic sampled antialiasing.
*Stream compaction to remove terminated rays and improve warp coherence.
*Sorting of material types in memory to reduce GPU memory lookups.

RESULTS
================
### Bathtub Scene
![Bathtub](img/bath1.png)
![Bathtub2](img/bath2.png)

### Geometry Scenes
![med1](img/medium.png)
![med2](img/medium2.png)

IMPLEMENTATION
================

### Light Transport
![renderEQ](img/equation.png)
Path tracing is based on the rendering equation shown above. This equation describes how light realistically interacts with objects. At any point on a surface, the light leaving that point (Lo) is determined by the right-hand side of the equation. If the surface is a light source, we add the color of the light it emits (Le).

To handle incoming light, we consider the entire hemisphere around the surface normal. This represents all directions from which light can arrive at the point. We integrate over this hemisphere to accumulate the total light entering the point. Inside this integral, the incoming light (Li) is multiplied by two terms. The first is the Bidirectional Reflectance Distribution Function (BRDF, f), which is specific to the material of the surface. For example, a diffuse surface scatters light broadly in the hemisphere, while a specular surface reflects light more narrowly. The second is the geometry term, which accounts for the angle between the incoming light and the surface normal.

Since it is impossible to evaluate this integral exactly, we rely on Monte Carlo integration. Rather than summing over every direction, we take a limited number of random samples from the hemisphere. Each sample estimates the contribution of incoming light in its direction, weighted by the BRDF and geometry term. By averaging these random samples, we build an unbiased estimate of the true integral. Increasing the number of samples reduces noise and makes the estimate converge toward the correct solution.
### Bidirectional Scattering Distribution Function
This path tracer implements three different BSDFs:
1. Diffuse
2. Specular
3. Refraction

The implementations are influenced by Physically Based Rendering: From Theory to Implementation and Professor Adam Mally’s CIS561 course slides.

Diffuse:
The diffuse material samples a random direction in the hemisphere around the surface normal to scatter incoming light. This produces the characteristic rough, evenly colored appearance of matte surfaces, as shown in the results below.
![spec](img/diffuse.png)

Specular:
The specular case reflects light perfectly about the normal, producing mirror-like reflections.
![spec](img/specular.png)

Refraction:
Refraction allows light to pass through transparent objects, bending according to the object’s index of refraction. To simulate real glass, the implementation randomly chooses whether a ray is reflected (specular) or transmitted (refractive) based on Fresnel terms. This produces the combined reflective and transmissive behavior of glass. Attenuation within the medium is modeled using the Beer–Lambert Law. It also produces a caustic effect where the light exiting the other side of the sphere coalesces, creating a bright spot on the surface in the path of the outgoing rays. This is shown below.
![spec](img/glass.png)
![spec](img/caustics.png)

### Intersections

### GLTF Mesh loading

### Bounding Volume Heirarchy 

### Stochastic Sampled Antialiasing

### Stream Compaction

### Material Sorting

REFERENCES
================
University of Pennsylvania CIS 560 and CIS 561 Slides by Adam Mally

University of Pennsylvania CIS 565 Path Tracing Slides by Ruipeng Wang

Physically Based Rendering: From Theory To Implementation (PBRTv4) by Matt Pharr, Wenzel Jakob, and Greg Humphreys
