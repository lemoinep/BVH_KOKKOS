
#include <Kokkos_Core.hpp>
#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>




    __host__ __device__
double dot(const double* a, const double* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__host__ __device__
void cross(const double* a, const double* b, double* result) {
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
}

struct Ray {
    double origin[3];
    double direction[3];
};


struct Triangle {
    double v0[3];
    double v1[3];
    double v2[3];


    void computeAABB(double min[3], double max[3]) const {
        min[0] = std::min({v0[0], v1[0], v2[0]});
        min[1] = std::min({v0[1], v1[1], v2[1]});
        min[2] = std::min({v0[2], v1[2], v2[2]});
        
        max[0] = std::max({v0[0], v1[0], v2[0]});
        max[1] = std::max({v0[1], v1[1], v2[1]});
        max[2] = std::max({v0[2], v1[2], v2[2]});
    }
};


struct AABB {
    double min[3];
    double max[3];

    __host__ __device__
    bool intersect(const Ray& ray) const {
        double tmin = (min[0] - ray.origin[0]) / ray.direction[0];
        double tmax = (max[0] - ray.origin[0]) / ray.direction[0];

        if (tmin > tmax) std::swap(tmin, tmax);

        for (int i = 1; i < 3; ++i) {
            double tymin = (min[i] - ray.origin[i]) / ray.direction[i];
            double tymax = (max[i] - ray.origin[i]) / ray.direction[i];

            if (tymin > tymax) std::swap(tymin, tymax);

            if ((tmin > tymax) || (tymin > tmax))
                return false;

            tmin = std::max(tmin, tymin);
            tmax = std::min(tmax, tymax);
        }
        return true;
    }
};


struct BVHNode {
    AABB bbox;
    int left;
    int right;
    int start;
    int count; 

    __host__ __device__ BVHNode() : left(-1), right(-1), start(-1), count(0) {}
};


void buildBVH(const std::vector<Triangle>& triangles, std::vector<BVHNode>& nodes, int& nodeCount) {
    if (triangles.empty()) return;


    BVHNode node;
    

    AABB bbox;
    bbox.min[0] = bbox.min[1] = bbox.min[2] = std::numeric_limits<double>::max();
    bbox.max[0] = bbox.max[1] = bbox.max[2] = -std::numeric_limits<double>::max();

    for (const auto& triangle : triangles) {
        double minAABB[3], maxAABB[3];
        triangle.computeAABB(minAABB, maxAABB);
        for (int i = 0; i < 3; ++i) {
            bbox.min[i] = std::min(bbox.min[i], minAABB[i]);
            bbox.max[i] = std::max(bbox.max[i], maxAABB[i]);
        }
    }

    node.bbox = bbox;
    node.start = 0;
    node.count = triangles.size();
    
    nodes[nodeCount++] = node;


    if (triangles.size() > 4) { 
        double centroidX = 0.0, centroidY = 0.0, centroidZ = 0.0;
        for (const auto& triangle : triangles) {
            centroidX += (triangle.v0[0] + triangle.v1[0] + triangle.v2[0]) / 3.0;
            centroidY += (triangle.v0[1] + triangle.v1[1] + triangle.v2[1]) / 3.0;
            centroidZ += (triangle.v0[2] + triangle.v1[2] + triangle.v2[2]) / 3.0;
        }
        centroidX /= triangles.size();
        centroidY /= triangles.size();
        centroidZ /= triangles.size();

        std::vector<Triangle> leftTriangles, rightTriangles;

        for (const auto& triangle : triangles) {
            double centerX = (triangle.v0[0] + triangle.v1[0] + triangle.v2[0]) / 3.0;
            if (centerX < centroidX) {
                leftTriangles.push_back(triangle);
            } else {
                rightTriangles.push_back(triangle);
            }
        }

        buildBVH(leftTriangles, nodes, nodeCount);
        buildBVH(rightTriangles, nodes, nodeCount);
        
        node.left = nodeCount - leftTriangles.size() - 1; 
        node.right = nodeCount - rightTriangles.size();

        nodes[nodeCount - 1] = node;
    }
}


__host__ __device__
bool intersect(const Ray& ray, const Triangle& triangle) {
    double edge1[3], edge2[3];
    
    for (int i = 0; i < 3; ++i) {
        edge1[i] = triangle.v1[i] - triangle.v0[i];
        edge2[i] = triangle.v2[i] - triangle.v0[i];
    }

    double h[3];
    cross(ray.direction, edge2, h);
    
    double a = dot(edge1, h);
    
    if (a > -std::numeric_limits<double>::epsilon() && a < std::numeric_limits<double>::epsilon())
        return false;

    double f = 1.0 / a;
    
    double s[3];
    for (int i = 0; i < 3; ++i)
        s[i] = ray.origin[i] - triangle.v0[i];

    double u = f * dot(s, h);
    
    if (u < 0.0 || u > 1.0)
        return false;

    double q[3];
    cross(s, edge1, q);
    
    double v = f * dot(ray.direction, q);
    
    if (v < 0.0 || u + v > 1.0)
        return false;


    double t = f * dot(edge2, q);
    
    return t > std::numeric_limits<double>::epsilon(); 
}


__global__ void traceRays(Ray* rays, Triangle* triangles, BVHNode* bvhNodes, int numRays, int numTriangles) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;

   if (idx < numRays) {
       Ray& ray = rays[idx];

       for (int i = 0; i < numTriangles; ++i) {
           const BVHNode& node = bvhNodes[i];

           if (!node.bbox.intersect(ray)) continue;

           for (int j = node.start; j < node.start + node.count; ++j) {
               if (intersect(ray, triangles[j])) {
                   printf("Ray %d intersects Triangle %d\n", idx, j);
               }
           }
       }
   }
}


int main(int argc, char* argv[]) {
    
    
    Kokkos::initialize(argc, argv);
    
    const int numTriangles = 10000; 
    std::vector<Triangle> triangles(numTriangles);
    
    for (int i = 0; i < numTriangles; ++i) {
        triangles[i].v0[0] = static_cast<double>(rand()) / RAND_MAX * 100.0;
        triangles[i].v0[1] = static_cast<double>(rand()) / RAND_MAX * 100.0;
        triangles[i].v0[2] = static_cast<double>(rand()) / RAND_MAX * 100.0;

        triangles[i].v1[0] = static_cast<double>(rand()) / RAND_MAX * 100.0;
        triangles[i].v1[1] = static_cast<double>(rand()) / RAND_MAX * 100.0;
        triangles[i].v1[2] = static_cast<double>(rand()) / RAND_MAX * 100.0;

        triangles[i].v2[0] = static_cast<double>(rand()) / RAND_MAX * 100.0;
        triangles[i].v2[1] = static_cast<double>(rand()) / RAND_MAX * 100.0;
        triangles[i].v2[2] = static_cast<double>(rand()) / RAND_MAX * 100.0;
    }


    Kokkos::View<Triangle*> d_triangles("triangles", numTriangles);
    Kokkos::View<Triangle*, Kokkos::HostSpace> h_triangles("h_triangles", numTriangles);
    for (int i = 0; i < numTriangles; ++i) {
        h_triangles(i) = triangles[i];
    }

    Kokkos::deep_copy(d_triangles, h_triangles);


    const int numRays = 1;

    Kokkos::View<Ray*> d_rays("rays", numRays);
    
    Kokkos::parallel_for("initialize_rays", numRays, KOKKOS_LAMBDA(int i) {
        d_rays(i).origin [0]= 50.5; 
        d_rays(i).origin [1]= 0.0;
        d_rays(i).origin [2]= -5.5; 

        d_rays(i).direction [0]= 0.0f;
        d_rays(i).direction [1]= 0.0f;
        d_rays(i).direction [2]= -1.0f;

        double length =
            sqrt(d_rays(i).direction [0]*d_rays(i).direction [0]
            + d_rays(i).direction [1]*d_rays(i).direction [1]
            + d_rays(i).direction [2]*d_rays(i).direction [2]);

        d_rays(i).direction [0]/=length;
        d_rays(i).direction [1]/=length;
        d_rays(i).direction [2]/=length;
    });


    std::vector<BVHNode> bvhNodes(numTriangles); 
    int nodeCount = 0;
    buildBVH(triangles, bvhNodes, nodeCount);
    Kokkos::View<BVHNode*, Kokkos::HostSpace> h_bvhNodes("h_bvhNodes", nodeCount);
    for (int i = 0; i < nodeCount; ++i) {
        h_bvhNodes(i) = bvhNodes[i];
    }
    Kokkos::View<BVHNode*> d_bvhNodes("d_bvhNodes", nodeCount);
    Kokkos::deep_copy(d_bvhNodes, h_bvhNodes);


    int threadsPerBlock = 512;
    int blocksPerGrid =(numRays + threadsPerBlock - 1) / threadsPerBlock;
    traceRays<<<blocksPerGrid, threadsPerBlock>>>(d_rays.data(), d_triangles.data(), d_bvhNodes.data(), numRays, numTriangles);
    hipDeviceSynchronize();
    Kokkos::finalize();

   return 0;
}


