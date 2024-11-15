
#include <SFML/Graphics.hpp>
#include <iostream>
#include <cstdint>
#include "tensor/tensor.hpp"
#include <chrono>
#include <cmath>
#include "ops/ops.h"
#include "vector/vectors.hpp"

#define repulsion 0.02
#define gravity 0.002
#define warp 0.0
#define scale 1
int size = 1;
auto numthreads = 8;
#define width 256 * scale
#define height 256 * scale
#define depth 256

int particlecount = height * width * height / size / size / 2;

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

#define FADEOFF(io, jo, ko) sqrt((io * io + jo * jo + ko * ko) + 1.0)
// #define FADEOFF(io, jo) 1.0
// #define FADEOFF(io,jo) sqrt(io*io+jo*jo + 1.0)
#include <thread>

void ProcessParticles(Tensor *Particles, Tensor *TensorField, int numParticles)
{

    while (1)
    {

        auto friction = 0.0;
        for (int i = -size; i < size + 1; i++)
        {
            for (int j = -size; j < size + 1; j++)
            {
                for (int k = -size; k < size + 1; k++)
                {
                    friction += 1.0 / FADEOFF(i, j, k);
                }
            }
        }
        friction = 1. / friction;
        for (int i = 0; i < numParticles; i++)
        {

            float4 &particle = *(*Particles)[i].as<float4>();
            if (particle.x == 0 || particle.y == 0)
            {
                continue;
            }
            auto mm = particle.w;

            if (mm > 0)
            {

                float4 momentum = {0, 0, 0, 0};

                // #pragma omp parallel for
                for (int io = -size; io < size + 1; io++)
                {
                    for (int jo = -size; jo < size + 1; jo++)
                    {
                        for (int k = -size; k < size + 1; k++)
                        {

                            int4 offset = {io, jo, k, 0};
                            momentum += *TensorField->get<float4>(particle.y + offset.y, particle.x + offset.x, particle.z + offset.z);
                            *TensorField->get<float4>(particle.y + offset.y, particle.x + offset.x, particle.z + offset.z) *= 0.0;
                        }
                    }
                }

                momentum.y += gravity;

                particle += momentum;

                if (particle.x + momentum.x < 1 || particle.x + momentum.x > width - 2 || particle.y + momentum.y < 1 || particle.y + momentum.y > height - 2 || particle.z + momentum.z < 1 || particle.z + momentum.z > depth - 2)
                {

                    if (particle.y + momentum.y < 1 || particle.y + momentum.y > height - 2)
                    {
                        momentum.y *= -1.0;
                        particle.y = std::max(1.0f + size, std::min(float(height - (1.0 + size)), particle.y + momentum.y));
                    }
                    if (particle.x + momentum.x < 1 || particle.x + momentum.x > width - 2)
                    {
                        momentum.x *= -1.0;
                        particle.x = std::max(1.0f + size, std::min(float(width - (1.0 + size)), particle.x + momentum.x));
                    }
                    if (particle.z + momentum.z < 1 || particle.z + momentum.z > depth - 2)
                    {
                        momentum.z *= -1.0;
                        particle.z = std::max(1.0f + size, std::min(float(depth - (1.0 + size)), particle.z + momentum.z));
                    }
                }

                // particle.x = std::max(1.0f + size+1, std::min(float(width - (1.0 + size)), particle.x));
                // particle.y = std::max(1.0f + size+1, std::min(float(height - (1.0 + size)), particle.y));

                auto aa = momentum * friction;
                // }
                // #pragma omp parallel for
                for (int io = -size; io < size + 1; io++)
                {
                    for (int jo = -size; jo < size + 1; jo++)
                    {
                        for (int k = -size; k < size + 1; k++)
                        {
                            int4 offset = {io, jo, k, 0};
                            auto fade = FADEOFF(io, jo, k);
                            float4 offsetto = float4({io, jo, k, 0}) / fade;
                            float4 moffset = (offsetto * repulsion + aa) / fade;
                            *TensorField->get<float4>(particle.y + offset.y, particle.x + offset.x, particle.z + offset.z) += moffset;
                        }
                    }
                }
            }
            else
            {
                // #pragma omp parallel for
                for (int io = -size - 2; io < size + 3; io++)
                {
                    for (int jo = -size - 2; jo < size + 3; jo++)
                    {
                        for (int k = -size - 2; k < size + 3; k++)
                        {
                            int4 offset = {io, jo, k, 0};
                            *TensorField->get<float4>(particle.y + offset.y, particle.x + offset.x, particle.z + offset.z) *= 0.0;
                        }
                    }
                }
            }

            //
        }
        // std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

struct thread
{
    Tensor *Particles;
    Tensor *TensorField;
    int numParticles;
    std::thread t;

    thread(Tensor &TensorField, int numParticles)
    {
        this->Particles = new Tensor({numParticles, 4}, kFLOAT_32);
        this->TensorField = &TensorField;

        *Particles = 0;
        this->numParticles = numParticles;
    }

    void start()
    {
        t = std::thread(
            [Particles = this->Particles, TensorField = this->TensorField, numParticles = this->numParticles]()
            {
                ProcessParticles(Particles, TensorField, numParticles);
            });
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
    float fov = 3.1415 / 4.0;
    float aspect = 1;
    float near = 0.00001;
    float far = 1000000;
    bool orthographic = false;

    Camera()
    {
        position = {0, 0, 0, 0};
        rotation = {0, 0, 0, 0};
        scaling = {1, 1, 1, 1};
        offset = {0, 0, 0, 0};
        identityMatrix = 0;
        identityMatrix[0][0] = 1;
        identityMatrix[1][1] = 1;
        identityMatrix[2][2] = 1;
        identityMatrix[3][3] = 1;
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
    }

    float4 apply(float4 input)
    {
        float4 out = {0, 0, 0, 0};
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                out[i] += *worldViewProjectionMatrix[j][i].as<float>() * input[j];
            }
        }

        // Perspective Division
        // auto aspect = width / height;
        // auto scale_point = width;
        // out.x = (out.x / out.w) * scale_point + width / 2;
        // out.y = (out.y / out.w) * scale_point + height / 2;
        // out.z = out.z / out.w;

        // out.x = out.x/width - 0.5;
        // out.y = out.y/height - 0.5;

        out.x = out.x / (out.z + 0.0001);
        out.y = out.y / (out.z + 0.0001);

        out.x = out.x * 2;
        out.y = out.y * 2;

        out.x *= 1024;
        out.y *= 1024;
        return out;
    }
};

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
    Tensor TensorField = Tensor({width, height, depth, 4}, kFLOAT_32);
    auto camera = Camera();

    std::vector<thread> threads;

    for (int i = 0; i < numthreads; i++)
    {
        threads.push_back(thread(TensorField, particlecount / numthreads));
        threads[i].start();
    }

    screenbuffer = 0;
    auto count = 0;
    TensorField = 0;

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

    sf::Vector2i localPosition = sf::Mouse::getPosition(window);
    float4 lastMousePos = {localPosition.x, localPosition.y, 0, 0};
    while (window.isOpen())
    {
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
            sf::Event(event).mouseMove
        }

        // clear the window with black color
        auto show = (std::chrono::system_clock::now() - lastFrame) > std::chrono::milliseconds(1000 / fps);
        if (show)
        {
            lastFrame = std::chrono::system_clock::now();

            window.clear(sf::Color::Black);

            screenbuffer = 0;
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

        localPosition = sf::Mouse::getPosition(window);

        auto deltamousepos = float4{localPosition.x - lastMousePos.x, localPosition.y - lastMousePos.y, 0., 0.};

        float4 mmpos = {width / 2, height / 2, depth / 2, 0};
        if (pressed && show)
        {
            float4 startdragpos = {mmpos.x - brushsize / 2, mmpos.y - brushsize / 2, mmpos.z - brushsize / 2, 0.};
            but = lpressed;
            float4 enddragpos = {mmpos.x + brushsize / 2, mmpos.y + brushsize / 2, mmpos.z + brushsize / 2, 0.};
            pressed = 0;
            ispressed = 1;

            for (int i = startdragpos.x; i < enddragpos.x; i += (size * 2) * but + 1)
            {
                for (int j = startdragpos.y; j < enddragpos.y; j += (size * 2) * but + 1)
                {
                    for (int k = startdragpos.z; k < enddragpos.z; k += (size * 2) * but + 1)
                    {
                        auto part = (*threads[count % numthreads].Particles)[(count++) / numthreads].as<float4>();
                        // *part = double2{i * scale, j*scale };
                        part->x = i * scale;
                        part->y = j * scale;
                        part->z = k * scale;
                        part->w = 1;
                    }
                }
            }
        }

        camera.rotation.x += deltamousepos.y / 300;
        camera.rotation.y -= deltamousepos.x / 300;

        // capture mouse
        // // sf::Mouse::setPosition(sf::Vector2i(512, 512), window);
        // lastMousePos = {512, 512, 0, 0};
        lastMousePos = {localPosition.x, localPosition.y, 0, 0};

        ispressed = pressed;

        // Update screen
        if (show)
        {
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::W))
            {
                camera.position.z -= cos(camera.rotation.y) * 20;
                camera.position.x -= sin(camera.rotation.y) * 20;
            }
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
            {
                camera.position.z += cos(camera.rotation.y) * 20;
                camera.position.x += sin(camera.rotation.y) * 20;
            }
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
            {
                camera.position.x -= cos(camera.rotation.y) * 20;
                camera.position.z += sin(camera.rotation.y) * 20;
            }
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
            {
                camera.position.x += cos(camera.rotation.y) * 20;
                camera.position.z -= sin(camera.rotation.y) * 20;
            }
            // wasd

            // shift and ctrl
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::LShift))
            {
                camera.position.y += 10;
            }
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space))
            {
                camera.position.y -= 10;
            }

            camera.update();

            for (int tt = 0; tt < numthreads; tt++)
            {

                for (int i = 0; threads[tt].numParticles > i; i++)
                {
                    auto part = (*threads[tt].Particles)[i].as<float4>();
                    if (part->x == 0 || part->y == 0)
                        continue;

                    float4 pos = {part->x, part->y, part->z, 1.0};
                    pos = camera.apply(pos);
                    if (pos.z < 0)
                        continue;
                    draw_circle(screenbuffer, pos.x / scale, pos.y / scale, 1, 0, tt * 255 / numthreads, part->z * 20, 255);
                }
            }

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