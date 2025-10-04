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

**Diffuse:**
The diffuse material samples a random direction in the hemisphere around the surface normal to scatter incoming light. This produces the characteristic rough, evenly colored appearance of matte surfaces, as shown in the results below.
![spec](img/diffuse.png)

**Specular:**
The specular case reflects light perfectly about the normal, producing mirror-like reflections.
![spec](img/specular.png)

**Refraction:**
Refraction allows light to pass through transparent objects, bending according to the object’s index of refraction. To simulate real glass, the implementation randomly chooses whether a ray is reflected (specular) or transmitted (refractive) based on Fresnel terms. This produces the combined reflective and transmissive behavior of glass. Attenuation within the medium is modeled using the Beer–Lambert Law. It also produces a caustic effect where the light exiting the other side of the sphere coalesces, creating a bright spot on the surface in the path of the outgoing rays. This is shown below.
![spec](img/glass.png)
![spec](img/caustics.png)

### Intersections

The path tracer supports intersection tests for spheres, boxes, and triangles. Each primitive uses a different geometric algorithm to determine ray hits.

**Spheres:**
Sphere intersection solves the ray-sphere equation analytically. The ray is transformed into object space, reducing the problem to a unit sphere at the origin. A quadratic equation determines if and where the ray intersects—the discriminant tells us if there's a hit, and the roots give us the intersection points. The implementation handles both exterior and interior hits by selecting the appropriate root, then transforms the result back to world space.

**Boxes:**
Box intersection uses the slab method, treating the box as three pairs of parallel planes. For each axis, we compute entry and exit points through the slabs. The intersection exists where all three axis intervals overlap. Like spheres, boxes are tested in object space to handle arbitrary rotations and scales.

**Triangles:**
Triangle intersection uses the Möller-Trumbore algorithm, which computes the hit point and barycentric coordinates in a single pass. The method performs early rejection tests and returns both the distance along the ray and the position within the triangle, useful for normal interpolation and texture mapping.

**Bounding Volume Hierarchy (BVH):**
Testing every triangle individually would be too expensive for complex meshes. The BVH organizes triangles into a tree of bounding boxes. If a ray misses a box, all triangles inside can be skipped. This reduces intersection tests from O(n) to O(log n), enabling real-time rendering of detailed geometry.

### GLTF Mesh Loading

![spec](img/bunny.png)
![spec](img/avocado.png)

The path tracer supports loading complex geometry from GLTF files, allowing for detailed scenes beyond simple primitives. GLTF (GL Transmission Format) is a standard 3D file format that stores mesh data, materials, and scene hierarchies in a compact, GPU-friendly structure.

The implementation uses the TinyGLTF library to parse GLTF files. For each mesh in the model, the loader extracts vertex positions, normals, and triangle indices from the file's buffer data. GLTF stores this data in a series of accessors and buffer views, which the loader traverses to reconstruct the geometry. Since GLTF supports multiple data types for indices (unsigned byte, short, or int), the code includes type detection to read indices correctly regardless of format.

Vertex positions are transformed and stored in a global vertex array, while triangle connectivity is preserved through index triplets. If a mesh includes explicit normals, those are loaded and normalized. Otherwise, the loader generates default upward-facing normals as a fallback. Additional information is stored in GLTF files like materials, however this implementation does not handle these. However, this information can be used to do texture mapping and physically based shading using roughness maps, metalness maps, and more. 

After loading all triangles, the implementation constructs a BVH acceleration structure to enable efficient ray-triangle intersection during rendering. This allows the path tracer to handle models with thousands of triangles at interactive frame rates. The rubber duck shown in the bathroom scene is an example of a GLTF mesh integrated into the renderer.

### Bounding Volume Hierarchy

For scenes with complex meshes containing thousands of triangles, testing every triangle against every ray would be computationally prohibitive. The BVH (Bounding Volume Hierarchy) acceleration structure solves this by organizing triangles into a tree of axis-aligned bounding boxes, allowing the renderer to quickly cull large portions of geometry that a ray cannot possibly hit.

**Construction:**
The BVH is built on the CPU during scene loading using a top-down approach. Starting with all triangles, the algorithm recursively splits them into two groups by choosing a split axis and position. The split axis is selected as the widest dimension of the triangles' centroid bounding box—this ensures splits happen along the direction where triangles are most spread out. Triangles are then partitioned based on whether their centroid falls to the left or right of the split position. Each node stores its bounding box and either points to two child nodes (interior nodes) or contains a list of triangles (leaf nodes). Leaf nodes are created when the triangle count drops below a threshold (8 triangles in this implementation).

**Traversal:**
During rendering, rays traverse the BVH using an explicit stack rather than recursion to avoid GPU stack limitations. Starting at the root, the algorithm tests if the ray intersects the node's bounding box using the same slab method as box intersection. If the ray misses, the entire subtree is skipped. If it hits and the node is a leaf, all triangles in that leaf are tested. If it's an interior node, both children are pushed onto the stack for further testing. This culling dramatically reduces the number of triangle intersection tests from O(n) to O(log n) on average, enabling real-time rendering of detailed geometry like the rubber duck mesh shown in the bathroom scene.

**Performance Impact:**
The BVH provides substantial performance improvements for mesh rendering. Without the acceleration structure, every ray must test against every triangle in the scene, resulting in millions of unnecessary intersection tests. With the BVH, rays only test triangles in the bounding boxes they actually intersect, reducing wasted work by orders of magnitude. The graph below shows the performance difference between naive triangle testing and BVH-accelerated rendering:

![spec](img/bvh.png)

### Stochastic Sampled Antialiasing

### Stream Compaction

### Material Sorting

REFERENCES
================
University of Pennsylvania CIS 560 and CIS 561 Slides by Adam Mally

University of Pennsylvania CIS 565 Path Tracing Slides by Ruipeng Wang

Physically Based Rendering: From Theory To Implementation (PBRTv4) by Matt Pharr, Wenzel Jakob, and Greg Humphreys
