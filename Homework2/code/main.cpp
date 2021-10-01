// clang-format off
#include <iostream>
#include <opencv2/opencv.hpp>
#include "rasterizer.hpp"
#include "global.hpp"
#include "Triangle.hpp"

constexpr double MY_PI = 3.1415926;
using namespace std;
const bool debug = false;

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1,0,0,-eye_pos[0],
                 0,1,0,-eye_pos[1],
                 0,0,1,-eye_pos[2],
                 0,0,0,1;

    view = translate*view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();

    // Create the model matrix for rotating the triangle around the Z axis.
    // Then return it.
    double rotation = rotation_angle / 180.0 * M_PI;
    model <<  
        cos(rotation), -sin(rotation), 0, 0,
        sin(rotation), cos(rotation), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;
    if (debug) {
        cout << "rotate by Z: " << rotation_angle << endl;
        cout << "Model Matrix:" << endl;
        cout << model << endl;
    }

    return model;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio,
                                      float zNear, float zFar)
{
    // sample input 45, 1, 0.1, 50

    if (debug) {
        cout << "get_projection_matrix" << endl;
        cout << "eye_fov: " << eye_fov << endl;
        cout << "aspect_ratio: " << aspect_ratio << endl;
        cout << "zNear: " << zNear << endl;
        cout << "zFar: " << zFar << endl;
    }

    zNear *= -1;
    zFar *= -1;

    float height = 2 * zNear * tan(eye_fov / 2 / 180 * M_PI);
    float width = height * aspect_ratio;
    float zDepth = abs(zFar - zNear);

    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

    // Create the projection matrix for the given parameters.
    // Then return it.
    Eigen::Matrix4f culling;

    culling << 
        zNear, 0, 0, 0, 
        0, zNear, 0, 0, 
        0, 0, zNear + zFar, - zNear * zFar, 
        0, 0, 1, 0;
    if (debug) {
        cout << "culling Matrix: \n" << culling << endl;
        Vector4f p(0, 0, zNear, 1);
        cout << "test culling : " << culling * p << endl;
    }

    Eigen::Matrix4f translate;
    translate << 
        1, 0, 0, 0, 
        0, 1, 0, 0, 
        0, 0, 1, (zNear + zFar) / 2, 
        0, 0, 0, 1;

    Eigen::Matrix4f scale;
    scale << 
        1 / width, 0, 0, 0,
        0, 1 / height, 0, 0,
        0, 0, 1 / zDepth, 0,
        0, 0, 0, 1;

    projection = scale * translate * culling * projection;

    return projection;
}

int main(int argc, const char** argv)
{
    float angle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    if (argc == 2)
    {
        command_line = true;
        filename = std::string(argv[1]);
    }

    rst::rasterizer r(700, 700);

    Eigen::Vector3f eye_pos = {0,0,5};


    std::vector<Eigen::Vector3f> pos
            {
                    {2, 0, -2},
                    {0, 2, -2},
                    {-2, 0, -2},
                    {3.5, -1, -5},
                    {2.5, 1.5, -5},
                    {-1, 0.5, -5}
            };

    std::vector<Eigen::Vector3i> ind
            {
                    {0, 1, 2},
                    {3, 4, 5}
            };

    std::vector<Eigen::Vector3f> cols
            {
                    {217.0, 238.0, 185.0},
                    {217.0, 238.0, 185.0},
                    {217.0, 238.0, 185.0},
                    {185.0, 217.0, 238.0},
                    {185.0, 217.0, 238.0},
                    {185.0, 217.0, 238.0}
            };

    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);
    auto col_id = r.load_colors(cols);

    int key = 0;
    int frame_count = 0;

    if (command_line)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imwrite(filename, image);

        return 0;
    }

    while(key != 27)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';
    }

    return 0;
}
// clang-format on