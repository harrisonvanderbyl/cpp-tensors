
#include <SFML/Graphics.hpp>
#include <iostream>
#include <cstdint>
#include "tensor/tensor.hpp"
#include <chrono>
#include <cmath>
#include "ops/ops.h"
#include <cuda_runtime.h>

auto repulsion = 0.01;
auto gravity = 0.01;
auto warp = 0.0;
auto scale = 1;

#define SIZE 1
int size = SIZE;
auto numthreads = 1;
auto width = 256 * scale;
auto height = 256 * scale;
auto depth = 256;

int particlecount = 1000000;

void draw_circle(Tensor &screenbuffer, int x, int y, int radius, int r, int g, int b, int a)
{
    for (int i = -radius; i < radius; i++)
    {
        for (int j = -radius; j < radius; j++)
        {
            if (i * i + j * j < radius * radius && x + j > 0 && x + j < screenbuffer.shape[1] && y + i > 0 && y + i < screenbuffer.shape[0])
            {

                uint64_t *pixel = screenbuffer.get<uint64_t>(y + i, x + j);
                uint8_t *pixel8 = (uint8_t *)pixel;
                pixel8[0] += r;
                pixel8[1] += g;
                pixel8[2] += b;
                pixel8[3] = 255;
            }
        }
    }
};


#define FADEOFF(io, jo, ko) 1
// #define FADEOFF(io, jo) 1.0
// #define FADEOFF(io,jo) sqrt(io*io+jo*jo + 1.0)

__global__ void ProcessParticlesKernel(float4 *particles, float4 *tensorField, int numParticles,
                                       int width, int height, int depth, float gravity,
                                       float repulsion, int numblocks)
{

    for (int ioo = 0; ioo < numblocks; ioo++)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x + ioo * blockDim.x * gridDim.x;
        if (tid >= numParticles)
            return;

        float4 particle = particles[tid];
        if (particle.x == 0 || particle.y == 0)
            return;

        float friction = 0.0f;
        for (int i = -SIZE; i <= SIZE; i++)
        {
            for (int j = -SIZE; j <= SIZE; j++)
            {
                for (int k = -SIZE; k <= SIZE; k++)
                {
                    friction += 1.0f / FADEOFF(i, j, k);
                }
            }
        }
        friction = 1.0f / friction;

        float4 momentum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        for (int io = -SIZE; io <= SIZE; io++)
        {
            for (int jo = -SIZE; jo <= SIZE; jo++)
            {
                for (int k = -SIZE; k <= SIZE; k++)
                {
                    int4 offset = make_int4(io, jo, k, 0);
                    float4 *fieldPtr = tensorField + ((int(particle.y) + offset.y) * width + (int(particle.x) + offset.x)) * depth + (int(particle.z) + offset.z);
                    momentum.x += fieldPtr->x;
                    momentum.y += fieldPtr->y;
                    momentum.z += fieldPtr->z;
                    fieldPtr->x = 0;
                    fieldPtr->y = 0;
                    fieldPtr->z = 0;
                }
            }
        }
        float4* myloc = tensorField + ((int(particle.y)) * width + (int(particle.x))) * depth + (int(particle.z));
        myloc->w = 0;
        float4 oldloc = {particle.x, particle.y, particle.z, 0};
        // momentum.y += gravity;
        particle.x += momentum.x;
        particle.y += momentum.y;
        particle.z += momentum.z;

              
        
        // particle.x = fminf(fmaxf(particle.x, 1.0 + SIZE), width - (1.0 + SIZE));
        // particle.y = fminf(fmaxf(particle.y, 1.0 + SIZE), height - (1.0 + SIZE));
        // particle.z = fminf(fmaxf(particle.z, 1.0 + SIZE), depth - (1.0 + SIZE));
        if(particle.x < 1.0 + SIZE){
            particle.x = 1.0 + SIZE;
            momentum.x *= -0.5;
        }
        if(particle.y < 1.0 + SIZE){
            particle.y = 1.0 + SIZE;
            momentum.y *= -0.5;
            // momentum.x *= 0.9;
            // momentum.z *= 0.9;

        }
        if(particle.z < 1.0 + SIZE){
            particle.z = 1.0 + SIZE;
            momentum.z *= -0.5;
            // momentum.x *= 0.9;
            // momentum.y *= 0.9;
        }
        if(particle.x > width - (1.0 + SIZE)){
            particle.x = width - (1.0 + SIZE);
            momentum.x *= -0.5;
            // momentum.y *= 0.9;
            // momentum.z *= 0.9;
        }
        if(particle.y > height - (1.0 + SIZE)){
            particle.y = height - (1.0 + SIZE);
            momentum.y *= -0.5;
            // momentum.x *= 0.9;
            // momentum.z *= 0.9;
        }
        if(particle.z > depth - (1.0 + SIZE)){
            particle.z = depth - (1.0 + SIZE);
            momentum.z *= -0.5;
            // momentum.x *= 0.9;
            // momentum.y *= 0.9;
        }
        // add small force to keep particles in bounds

        // momentum.x += (width / 2 - particle.x)/100000.f;
        // momentum.y += (height / 2 - particle.y)/100000.f;
        // momentum.z += (depth / 2 - particle.z)/100000.f;
        // float4 tocenter = make_float4(width / 2 - particle.x, height / 2 - particle.y, depth / 2 - particle.z, 0);
        // float distance = sqrt(tocenter.x * tocenter.x + tocenter.y * tocenter.y + tocenter.z * tocenter.z);
        // tocenter.x /= distance;
        // tocenter.y /= distance;
        // tocenter.z /= distance;
        // float forcefelt = 0.001 / (distance * distance/1000 + 1);
        // momentum.x += tocenter.x * forcefelt;
        // momentum.y += tocenter.y * forcefelt;
        // momentum.z += tocenter.z * forcefelt;
        momentum.y += gravity;
       

        momentum.x *= friction;
        momentum.y *= friction;
        momentum.z *= friction;

        for (int io = -SIZE; io <= SIZE; io++)
        {
            for (int jo = -SIZE; jo <= SIZE; jo++)
            {
                for (int k = -SIZE; k <= SIZE; k++)
                {
                    int4 offset = make_int4(io, jo, k, 0);
                    float fade = FADEOFF(io, jo, k);
                    float4 offsetto;
                    offsetto.x = io / fade;
                    offsetto.y = jo / fade;
                    offsetto.z = k / fade;

                    float4 moffset;
                    moffset.x = (offsetto.x * repulsion + momentum.x) / fade;
                    moffset.y = (offsetto.y * repulsion + momentum.y) / fade;
                    moffset.z = (offsetto.z * repulsion + momentum.z) / fade;
                    float *fieldPtr = (&tensorField->x) + 4 * (((int(particle.y) + offset.y) * width + (int(particle.x) + offset.x)) * depth + (int(particle.z) + offset.z));
                    *(fieldPtr) += moffset.x;
                    *(fieldPtr + 1) += moffset.y;
                    *(fieldPtr + 2) += moffset.z;
                    
                }
            }
        }
        myloc = tensorField + ((int(particle.y)) * width + (int(particle.x))) * depth + (int(particle.z));
        if(myloc->w >= 0.75){
            particle.x = oldloc.x;
            particle.y = oldloc.y;
            particle.z = oldloc.z;
        }
        myloc->w = 1;
        

        particles[tid].x = particle.x;
        particles[tid].y = particle.y;
        particles[tid].z = particle.z;
        particles[tid].w = particle.w;
    }
}

void LaunchProcessParticlesKernel(float4 *d_particles, float4 *d_tensorField, int numParticles,
                                  int width, int height, int depth, float gravity, float repulsion)
{
    int blockSize = 1024; // Adjust this based on your GPU's capabilities
    int numBlocks = (numParticles + blockSize - 1) / blockSize;

    ProcessParticlesKernel<<<numBlocks, blockSize>>>(d_particles, d_tensorField, numParticles,
                                                     width, height, depth, gravity, repulsion, 1);

    cudaDeviceSynchronize(); // Ensure all threads have finished
}

struct thread
{
    Tensor *Particles;
    float4 *TensorField;
    int numParticles;
    float4 *gpuparticles;
    // std::thread t;

    thread(int numParticles)
    {
        this->Particles = new Tensor({numParticles, 4}, kFLOAT_32);

        cudaMalloc(&gpuparticles, numParticles * sizeof(float4));
        cudaMalloc(&this->TensorField, width * height * depth * 4 * sizeof(float));

        *Particles = 0;
        this->numParticles = numParticles;
    }

    void run()
    {
        // copy particles to device

        LaunchProcessParticlesKernel((float4 *)this->gpuparticles, this->TensorField, numParticles, width, height, depth, gravity, repulsion);
    }

    void update()
    {
        // copy particles to host
        cudaMemcpy(Particles->data, gpuparticles, numParticles * sizeof(float4), cudaMemcpyDeviceToHost);
    }

    void flushToGPU()
    {
        cudaMemcpy(gpuparticles, Particles->data, numParticles * sizeof(float4), cudaMemcpyHostToDevice);
    }
};

struct Camera
{
    float4 position;
    float4 rotation;
    float4 scaling;
    float4 offset;
    Tensor projectionMatrix = Tensor({4, 4}, kFLOAT_32);
    Tensor rotationMatrix = Tensor({4, 4}, kFLOAT_32);
    Tensor rotationTranslationMatrix = Tensor({4, 4}, kFLOAT_32);
    Tensor translationMatrix = Tensor({4, 4}, kFLOAT_32);
    Tensor worldViewProjectionMatrix = Tensor({4, 4}, kFLOAT_32);
    Tensor identityMatrix = Tensor({4, 4}, kFLOAT_32);
    float *worldViewProjectionMatrixGPU;
    float fov = 3.1415 / 4.0;
    float aspect = 1;
    float near = 0.00001;
    float far = 1000000;
    bool orthographic = false;

    Camera()
    {
        position = {40, 40, 2000, 0};
        rotation = {0, 0, 0, 0};
        scaling = {1, 1, 1, 1};
        offset = {0, 0, 0, 0};
        identityMatrix = 0;
        identityMatrix[0][0] = 1;
        identityMatrix[1][1] = 1;
        identityMatrix[2][2] = 1;
        identityMatrix[3][3] = 1;

        cudaMalloc(&worldViewProjectionMatrixGPU, 16 * sizeof(float));
    }

    void updateProjectionMatrix()
    {
        projectionMatrix = identityMatrix;
        if (!orthographic)
        {
            // Perspective Projection Matrix
            float tanHalfFov = tan(fov / 2.0f);
            projectionMatrix[0][0] = 1.0f / (aspect * tanHalfFov);
            projectionMatrix[1][1] = 1.0f / tanHalfFov;
            projectionMatrix[2][2] = -(far + near) / (far - near);
            projectionMatrix[2][3] = -1.0f;
            projectionMatrix[3][2] = -(2.0f * far * near) / (far - near);
        }
        else
        {
            // Orthographic Projection Matrix
            projectionMatrix[0][0] = 1.0f / (aspect * tan(fov / 2.0f));
            projectionMatrix[1][1] = 1.0f / tan(fov / 2.0f);
            projectionMatrix[2][2] = 2.0f / (far - near);
            projectionMatrix[2][3] = 0.0f;
            projectionMatrix[3][2] = -(far + near) / (far - near);
        }
    }

    void updateWorldViewProjectionMatrix()
    {

        // Rotation Matrix
        rotationMatrix = identityMatrix;
        rotationMatrix[0][0] = cos(rotation.y) * cos(rotation.z);
        rotationMatrix[0][1] = cos(rotation.z) * sin(rotation.x) * sin(rotation.y) - cos(rotation.x) * sin(rotation.z);
        rotationMatrix[0][2] = cos(rotation.x) * cos(rotation.z) * sin(rotation.y) + sin(rotation.x) * sin(rotation.z);
        rotationMatrix[1][0] = cos(rotation.y) * sin(rotation.z);

        rotationMatrix[1][1] = cos(rotation.x) * cos(rotation.z) + sin(rotation.x) * sin(rotation.y) * sin(rotation.z);
        rotationMatrix[1][2] = -cos(rotation.z) * sin(rotation.x) + cos(rotation.x) * sin(rotation.y) * sin(rotation.z);
        rotationMatrix[2][0] = -sin(rotation.y);

        rotationMatrix[2][1] = cos(rotation.y) * sin(rotation.x);
        rotationMatrix[2][2] = cos(rotation.x) * cos(rotation.y);

        // Translation Matrix
        translationMatrix = identityMatrix;
        translationMatrix[3][0] = -position.x;
        translationMatrix[3][1] = -position.y;
        translationMatrix[3][2] = -position.z;

        worldViewProjectionMatrix = 0;
        rotationTranslationMatrix = 0;

        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                for (int k = 0; k < 4; k++)
                {
                    *rotationTranslationMatrix[j][i].as<float>() += *rotationMatrix[k][i].as<float>() * *translationMatrix[j][k].as<float>();
                }
            }
        }

        // worldViewProjectionMatrix = worldViewProjectionMatrix * translationMatrix;
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                for (int k = 0; k < 4; k++)
                {
                    *worldViewProjectionMatrix[j][i].as<float>() += *projectionMatrix[k][i].as<float>() * *rotationTranslationMatrix[j][k].as<float>();
                }
            }
        }
    }

    void update()
    {
        updateProjectionMatrix();
        updateWorldViewProjectionMatrix();
        cudaMemcpy(worldViewProjectionMatrixGPU, worldViewProjectionMatrix.data, 16 * sizeof(float), cudaMemcpyHostToDevice);
    }

    float4 apply(float4 input)
    {
        float4 out = {0, 0, 0, 0};

        out.x += *worldViewProjectionMatrix[0][0].as<float>() * input.x;
        out.y += *worldViewProjectionMatrix[0][1].as<float>() * input.x;
        out.z += *worldViewProjectionMatrix[0][2].as<float>() * input.x;
        out.w += *worldViewProjectionMatrix[0][3].as<float>() * input.x;
        out.x += *worldViewProjectionMatrix[1][0].as<float>() * input.y;
        out.y += *worldViewProjectionMatrix[1][1].as<float>() * input.y;
        out.z += *worldViewProjectionMatrix[1][2].as<float>() * input.y;
        out.w += *worldViewProjectionMatrix[1][3].as<float>() * input.y;
        out.x += *worldViewProjectionMatrix[2][0].as<float>() * input.z;
        out.y += *worldViewProjectionMatrix[2][1].as<float>() * input.z;
        out.z += *worldViewProjectionMatrix[2][2].as<float>() * input.z;
        out.w += *worldViewProjectionMatrix[2][3].as<float>() * input.z;
        out.x += *worldViewProjectionMatrix[3][0].as<float>() * input.w;
        out.y += *worldViewProjectionMatrix[3][1].as<float>() * input.w;
        out.z += *worldViewProjectionMatrix[3][2].as<float>() * input.w;
        out.w += *worldViewProjectionMatrix[3][3].as<float>() * input.w;

        out.x = (out.x - 0.25) / (out.z + 0.0001) + 0.25;
        out.y = (out.y - 0.25) / (out.z + 0.0001) + 0.25;

        out.x = out.x * 2;
        out.y = out.y * 2;

        out.x *= 1024;
        out.y *= 1024;
        return out;
    }
};

#define MATMUL(matrix,vector) { matrix[0] * vector.x + matrix[4] * vector.y + matrix[8] * vector.z + matrix[12] * vector.w, matrix[1] * vector.x + matrix[5] * vector.y + matrix[9] * vector.z + matrix[13] * vector.w, matrix[2] * vector.x + matrix[6] * vector.y + matrix[10] * vector.z + matrix[14] * vector.w, matrix[3] * vector.x + matrix[7] * vector.y + matrix[11] * vector.z + matrix[15] * vector.w }
#define PERSPECTIVE(vector) vector.x = 2048 *  ((vector.x-0.25) / (vector.z + 0.0001) + 0.25); vector.y = 2048 * ((vector.y - 0.25) / (vector.z + 0.0001) + 0.25)
#define DRAWPIXEL(x,y,r,g,b) { uint64_t *pixel = ((uint64_t *)(screenbuffer)) + int(y) * 512 + int(x) / 2; uint8_t *pixel8 = (uint8_t *)pixel; pixel8[0] = (r); pixel8[1] = (g); pixel8[2] = (b); pixel8[3] = 255; }
#define DRAWPIXELMIX(x,y,r,g,b,mix) { uint64_t *pixel = ((uint64_t *)(screenbuffer)) + int(y) * 512 + int(x) / 2; uint8_t *pixel8 = (uint8_t *)pixel; pixel8[0] = (r) * (mix) + pixel8[0] * (1 - mix); pixel8[1] = (g) * (mix) + pixel8[1] * (1 - mix); pixel8[2] = (b) * (mix) + pixel8[2] * (1 - mix); pixel8[3] = 255; }
#define MIN8(a, b, c, d, e, f, g, h) min(min(min(min(min(min(min(a, b), c), d), e), f), g), h)
#define MAX8(a, b, c, d, e, f, g, h) max(max(max(max(max(max(max(a, b), c), d), e), f), g), h)

__global__ void DisplayParticlesKernel(float4 *tensor_field, float *camera, uint8_t *screenbuffer, int w, int h, int d, int width, int height)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int x = tid % w;
    int y = (tid / w) % h;
    for (int z = 0; z < d; z++)
    {
        float4 pz = tensor_field[(y * w + x) * d + z];
        float density = pz.w;//abs(pz.x) + abs(pz.y) + abs(pz.z);
        tensor_field[(y * w + x) * d + z].w *= 0.95;
        if(y > 200 && y < 200 + 5 && x < 200){
           tensor_field[(y * w + x) * d + z].w = 2;
        }
        // density = pz.w;
        if (density < 0.01)
            continue;
        float4 pos[8];
        pos[0] = {float(x), float(y), float(z), 1.0f};
        // pos[1] = {float(x), float(y), float(z) + 1.0f, 1.0f};
        // pos[2] = {float(x), float(y) + 1.0f, float(z), 1.0f};
        // pos[3] = {float(x), float(y) + 1.0f, float(z) + 1.0f, 1.0f};
        // pos[4] = {float(x) + 1.0f, float(y), float(z), 1.0f};
        // pos[5] = {float(x) + 1.0f, float(y), float(z) + 1.0f, 1.0f};
        // pos[6] = {float(x) + 1.0f, float(y) + 1.0f, float(z), 1.0f};
        // pos[7] = {float(x) + 1.0f, float(y) + 1.0f, float(z) + 1.0f, 1.0f};

        // for (int i = 0; i < 8; i++)
        // {
        //     pos[i] = MATMUL(camera, pos[i]);
        //     PERSPECTIVE(pos[i]);

        // }
        pos[0] = MATMUL(camera, pos[0]);
        PERSPECTIVE(pos[0]);
        if (pos[0].x < 0 || pos[0].x >= 1024 || pos[0].y < 0 || pos[0].y >= 1024 || pos[0].z < 0)
            continue;
        
        DRAWPIXELMIX(pos[0].x, pos[0].y, 0, (density-1)*255, 255, 0.20);
        DRAWPIXELMIX(pos[0].x+1, pos[0].y, 0, (density-1)*255, 255, 0.20);
        DRAWPIXELMIX(pos[0].x, pos[0].y, 0+1, (density-1)*255, 255, 0.20);
        DRAWPIXELMIX(pos[0].x-1, pos[0].y, 0, (density-1)*255, 255, 0.20);
        DRAWPIXELMIX(pos[0].x, pos[0].y-1, 0, (density-1)*255, 255, 0.20);
        // if (pos[0].x < 0 || pos[0].x >= 1024 || pos[0].y < 0 || pos[0].y >= 1024 || pos[0].z < 0)
        //     continue;
        
        // float4 boundingbox[2] = {pos[0], pos[0]};
        // for (int i = 1; i < 8; i++)
        // {
        //     boundingbox[0].x = min(boundingbox[0].x, pos[i].x);
        //     boundingbox[0].y = min(boundingbox[0].y, pos[i].y);
        //     boundingbox[1].x = max(boundingbox[1].x, pos[i].x);
        //     boundingbox[1].y = max(boundingbox[1].y, pos[i].y);
        //     boundingbox[0].z += pos[i].x/8;
        //     boundingbox[0].w += pos[i].y/8;
        // }
        
        // float4 center = {boundingbox[0].z, boundingbox[0].w, 0, 0};
        // for (int i = boundingbox[0].x; i < boundingbox[1].x; i++)
        // {
        //     for (int j = boundingbox[0].y; j < boundingbox[1].y; j++)
        //     {
        //         // DRAWPIXEL(i, j, 255, 255, 255);
        //         // determine if point is inside the box
        //         float4 tocenter = {center.x - i, center.y - j, 0, 0};
        //         // if the dot product of the tocentor vector and the vector from the point to all the pos points is positive, then the point is outside the box
        //         bool inside = 1;
        //         for (int k = 0; k < 8; k++)
        //         {
        //             float4 topos = {pos[k].x - i, pos[k].y - j, 0, 0};
        //             if (topos.x * tocenter.x + topos.y * tocenter.y > 0)
        //             {
        //                 inside = 1;
        //                 break;
        //             }
        //         }
        //         if (inside)
        //         {
        //             DRAWPIXELMIX(i, j, 0, 0, 255, 0.125);
        //         }
                

        //     }
        // }
        
    }

   
    
    
}

void LaunchDisplayParticlesKernel(float4 *tensorfield, float *d_camera, uint8_t *d_screenbuffer, int w, int h, int d, int width, int height)
{
    int blockSize = 1024; // Adjust this based on your GPU's capabilities
    int numBlocks = ((w*h) + blockSize - 1) / blockSize;

    DisplayParticlesKernel<<<numBlocks, blockSize>>>(tensorfield, d_camera, d_screenbuffer, w,h,d, width, height);

    cudaDeviceSynchronize(); // Ensure all threads have finished
}

int main()
{

    // auto friction = ((1.0-0.0)/ pow(size*2 + 1,2));

    // create the window
    sf::RenderWindow window(sf::VideoMode(1024, 1024), "Some Funky Title");

    // create a texture
    sf::Texture texture;
    texture.create(1024, 1024);

    // Create a pixel buffer to fill with RGBA data
    Tensor screenbuffer = Tensor({1024, 1024, 4}, kUINT_8);

    uint8_t *screenbufferGPU;
    cudaMalloc(&screenbufferGPU, 1024 * 1024 * 4 * sizeof(uint8_t));

    auto camera = Camera();

    std::vector<thread> threads;

    for (int i = 0; i < numthreads; i++)
    {
        threads.push_back(thread(particlecount / numthreads));
    }

    window.setMouseCursorGrabbed(true);
    window.setMouseCursorVisible(false);
    window.setMouseCursorGrabbed(true);

    auto count = 0;
    // The colour we will fill the window with
    unsigned char red = 0;
    unsigned char blue = 255;
    auto currenttime = std::chrono::system_clock::now();

    // run the program as long as the window is open
    auto lastFrame = std::chrono::system_clock::now();
    auto fps = 60;
    bool ispressed = false;
    int frame = 0;
    bool but = 0;
    uint32_t brushsize = 20;

    auto lastMousePos = sf::Mouse::getPosition(window);

    while (window.isOpen())
    {
        threads[0].run();
        // check all the window's events that were triggered since the last iteration of the loop
        sf::Event event;
        while (window.pollEvent(event))
        {
            // "close requested" event: we close the window
            if (event.type == sf::Event::Closed)
                window.close();

            // if esc is pressed close the window
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
                window.close();
            if (sf::Event(event).type == sf::Event::MouseWheelScrolled)
            {
                brushsize += int(sf::Event(event).mouseWheelScroll.delta);
            }
            if (sf::Event(event).type == sf::Event::MouseMoved)
            {
                // if(!window.hasFocus())continue;
                auto mpos = sf::Event(event).mouseMove;
                int2 delta = {mpos.x - lastMousePos.x, mpos.y - lastMousePos.y};
                // if(delta.x==0 && delta.y==0)continue;

                lastMousePos = {mpos.x, mpos.y};
                if (mpos.x == 512 && mpos.y == 512)
                    continue;

                camera.rotation.x += float(delta.y) / 300;
                camera.rotation.y -= float(delta.x) / 300;


                //
                // localPosition = sf::Mouse::getPosition(window);
            }
        }

        // clear the window with black color
        auto show = (std::chrono::system_clock::now() - lastFrame) > std::chrono::milliseconds(1000 / fps);
        if (show)
        {
            lastFrame = std::chrono::system_clock::now();

            window.clear(sf::Color::Black);

            // set screenbuffergpu to black
            cudaMemset(screenbufferGPU, 0, 1024 * 1024 * 4 * sizeof(uint8_t));
        }

        // Create RGBA value to fill screen with.
        // Increment red and decrement blue on each cycle. Leave green=0, and make opaque
        // uint32_t RGBA;
        // Stuff data into buffer
        // get mouse position in the window

        // TensorFieldLast = TensorFieldLast*0.5;
        // TensorField[] = a;
        bool lpressed = sf::Mouse::isButtonPressed(sf::Mouse::Left);
        bool rpressed = sf::Mouse::isButtonPressed(sf::Mouse::Right);
        bool mpressed = sf::Mouse::isButtonPressed(sf::Mouse::Middle);

        // scroll for brush size

        bool pressed = lpressed || rpressed;
        // if (pressed && !ispressed)
        // {
        //     startdragpos = {localPosition.x + rand() % scale, localPosition.y};
        //     but = lpressed;
        // }

        float4 mmpos = {width / 2, height / 2, depth / 2, 0};
        if (pressed && show)
        {

            float4 startdragpos = {mmpos.x - brushsize/2, mmpos.y - brushsize / 2, mmpos.z - brushsize / 2, 0.};
            but = lpressed;
            float4 enddragpos = {mmpos.x + brushsize/2, mmpos.y + brushsize / 2, mmpos.z + brushsize / 2, 0.};
            pressed = 0;
            ispressed = 1;

            // threads[0].update();
            float4 pos = {0, 0, 0, 1.0};

            for (int i = startdragpos.x; i < enddragpos.x; i += (size * 2) * but + 1)
            {
                for (int j = startdragpos.y; j < enddragpos.y; j += (size * 2) * but + 1)
                {
                    for (int k = startdragpos.z; k < enddragpos.z; k += (size * 2) * but + 1)
                    {
                        auto part = threads[0].gpuparticles + count;
                        // *part = double2{i * scale, j*scale };
                        pos.x = i;
                        pos.y = j;
                        pos.z = k;

                        cudaMemcpy(part, &pos, sizeof(float4), cudaMemcpyHostToDevice);
                        count += 1;
                    }
                }
            }
            // threads[count % numthreads].flushToGPU();
        }

        // capture mouse
        // // sf::Mouse::setPosition(sf::Vector2i(512, 512), window);
        // lastMousePos = {512, 512, 0, 0};
        // window.setMousePosition(sf::Vector2i(512, 512));

        ispressed = pressed;

        // Update screen
        if (show)
        {

            sf::Mouse::setPosition(sf::Vector2i(512, 512), window);
            // reset mouse position

            // lastMousePos = {float(localPosition.x), float(localPosition.y), 0., 0.};
            float speed = 10;
             if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space))
            {
                speed = 20;
            }
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::LShift))
            {
                speed = 1;
            }


            if (sf::Keyboard::isKeyPressed(sf::Keyboard::W))
            {
                camera.position.z -= cos(camera.rotation.y) * speed * cos(camera.rotation.x);
                camera.position.x -= sin(camera.rotation.y) * speed * cos(camera.rotation.x);
                camera.position.y += sin(camera.rotation.x) * speed;
            }
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
            {
                camera.position.z += cos(camera.rotation.y) * speed * cos(camera.rotation.x);
                camera.position.x += sin(camera.rotation.y) * speed * cos(camera.rotation.x);
                camera.position.y -= sin(camera.rotation.x) * speed;
            }
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
            {
                camera.position.x -= cos(camera.rotation.y) * speed;
                camera.position.z += sin(camera.rotation.y) * speed;
            }
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
            {
                camera.position.x += cos(camera.rotation.y) * speed;
                camera.position.z -= sin(camera.rotation.y) * speed;
            }
            // wasd

            // shift and ctrl
           
            camera.update();

            // for (int tt = 0; tt < numthreads; tt++)
            // {

            //     for (int i = 0; threads[tt].numParticles > i; i++)
            //     {
            //         auto part = (*threads[tt].Particles)[i].as<float4>();
            //         if (part->x == 0 || part->y == 0)
            //             continue;

            //         float4 pos = {part->x, part->y, part->z, 1.0};
            //         pos = camera.apply(pos);
            //         if (pos.z < 0)
            //             continue;
            //         draw_circle(screenbuffer, pos.x / scale, pos.y / scale, 1, 0, tt * 255 / numthreads, part->z * 20, 255);
            //     }
            // }

            // display particles
            for (int tt = 0; tt < numthreads; tt++)
            {
                LaunchDisplayParticlesKernel(threads[tt].TensorField, camera.worldViewProjectionMatrixGPU, screenbufferGPU, width,height,depth, 1024, 1024);
            }

            cudaMemcpy(screenbuffer.data, screenbufferGPU, 1024 * 1024 * 4 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
            // draw cube 10x10x10
            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    for (int k = 0; k < 10; k++)
                    {
                        float4 pos = {i * 10.f, j * 10.f, k * 10.f, 1.0f};
                        pos = camera.apply(pos);
                        if (pos.z < 0)
                            continue;
                        draw_circle(screenbuffer, pos.x / scale, pos.y / scale, 1, 255, 255, 255, 255);
                    }
                }
            }

            texture.update((uint8_t *)screenbuffer.data);
            sf::Sprite sprite(texture);
            window.draw(sprite);

            // end the current frame
            window.display();
            auto currtime = std::chrono::system_clock::now();
            auto camerpos = camera.position;
            auto zeropos = camera.apply({0, 0, 0, 1});
            // std::cout << "frames per second(0dp): " << int(1.0 / std::chrono::duration<double>(currtime - currenttime).count()) << " : Active Particles: " << std::min(count, particlecount) << " : Brush Size: " << brushsize << "\r";
            std::cout << "camera position: " << camerpos.x << " " << camerpos.y << " " << camerpos.z << " : zero position: " << zeropos.x << " " << zeropos.y << " " << zeropos.z << " : frames per second(0dp): " << int(1.0 / std::chrono::duration<double>(currtime - currenttime).count()) << " : Active Particles: " << std::min(count, particlecount) << " : Brush Size: " << brushsize << "\r";
            std::cout << std::flush;
            currenttime = currtime;
            frame++;
        }
        // if(frame==1000)break;
    }

    return 0;
}