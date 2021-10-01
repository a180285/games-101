// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>

constexpr double MY_PI = 3.1415926;
using namespace std;
const bool debug = false;

rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}


static bool insideTriangle(float x, float y, const Vector3f* _v)
{   
    Vector3f p(x, y, 0);
    // cout << "p: \n" << p << endl;

    for (int i = 0; i < 3; i++) {
        Vector3f v = _v[(i + 1) % 3] - _v[i];
        auto tp = p - _v[i];
        v[2] = 0;
        // cout << "v : \n" << v << endl;
        if (v.cross(tp)[2] < 0) {
            return false;
        }
    }
    // Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
    return true;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind)
    {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        // rasterize_triangle(t);
        msaa_triangle(t);
    }
    final_msaa();
}

void rst::rasterizer::final_msaa() {
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            Vector3f color(0, 0, 0);
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    color += get_msaa_pixel(2*x+i, 2*y+j);
                }
            }
            set_pixel({x, y, 0}, color / 4);
        }
    }
}

void rst::rasterizer::msaa_triangle(const Triangle& t) {
    auto v = t.toVector4();

    double lx = min(min(t.v[0][0], t.v[1][0]), t.v[2][0]);
    double ly = min(min(t.v[0][1], t.v[1][1]), t.v[2][1]);
    double hx = max(max(t.v[0][0], t.v[1][0]), t.v[2][0]);
    double hy = max(max(t.v[0][1], t.v[1][1]), t.v[2][1]);

    for (int xi = 2 * max(0., lx); xi <= 2 * min(double(width), hx); xi++) {
        for (int yi = 2 * max(0., ly); yi <= 2 * min(double(height), hy); yi++) {
            if (!insideTriangle(xi / 2.0, yi / 2.0, t.v)) {
                continue;
            }
            // If so, use the following code to get the interpolated z value.
            auto [alpha, beta, gamma] = computeBarycentric2D(xi / 2.0, yi / 2.0, t.v);
            float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
            float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
            z_interpolated *= w_reciprocal;

            z_interpolated *= -1;
            auto& depth = msaa_depth_buf[yi * 2 * width + xi];
            if (z_interpolated < depth) {
                depth = z_interpolated;
                set_msaa_pixel({xi, yi, 0}, t.getColor());
            }

        }
    }

}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();

    if (debug) {
        cout << "t.v : \n";
        cout << t.v[0] << endl;
        cout << t.v[1] << endl;
        cout << t.v[2] << endl;
    }
    double lx = min(min(t.v[0][0], t.v[1][0]), t.v[2][0]);
    double ly = min(min(t.v[0][1], t.v[1][1]), t.v[2][1]);
    double hx = max(max(t.v[0][0], t.v[1][0]), t.v[2][0]);
    double hy = max(max(t.v[0][1], t.v[1][1]), t.v[2][1]);

    // TODO : Find out the bounding box of current triangle.
    // iterate through the pixel and find if the current pixel is inside the triangle

    for (int xi = max(0., lx); xi <= min(double(width), hx); xi++) {
        for (int yi = max(0., ly); yi <= min(double(height), hy); yi++) {
            if (!insideTriangle(xi, yi, t.v)) {
                continue;
            }
            // If so, use the following code to get the interpolated z value.
            auto [alpha, beta, gamma] = computeBarycentric2D(xi, yi, t.v);
            float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
            float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
            z_interpolated *= w_reciprocal;

            z_interpolated *= -1;
            auto& depth = depth_buf[yi * width + xi];
            if (z_interpolated < depth) {
                depth = z_interpolated;
                set_pixel({xi, yi, 0}, t.getColor());
            }

            // set the current pixel (use the set_pixel function) to 
            // the color of the triangle (use getColor function) if it should be painted.
        }
    }

}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
        std::fill(msaa_frame_buf.begin(), msaa_frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
        std::fill(msaa_depth_buf.begin(), msaa_depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
    msaa_frame_buf.resize(w * h * 4);
    msaa_depth_buf.resize(w * h * 4);
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height-1-point.y())*width + point.x();
    frame_buf[ind] = color;
}

void rst::rasterizer::set_msaa_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (2*height-1-point.y())*2 * width + point.x();
    msaa_frame_buf[ind] = color;
}

const Eigen::Vector3f& rst::rasterizer::get_msaa_pixel(int x, int y) {
    auto ind = (2*height-1-y)*2 * width + x;
    return msaa_frame_buf[ind];
}

// clang-format on