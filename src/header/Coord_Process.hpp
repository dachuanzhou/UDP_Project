// -----------------------------------------------------------------------------
// Filename:    Coord_Process.hpp
// Revision:    None
// Date:        2018/09/24 - 10:10
// Author:      Haixiang HOU
// Email:       hexid26@outlook.com
// Website:     [NULL]
// Notes:       [NULL]
// -----------------------------------------------------------------------------
// Copyright:   2018 (c) Haixiang
// License:     GPL
// -----------------------------------------------------------------------------
// Version [1.0]
// Calculate the coordinates of senders/receivers and sample points.

#include <boost/asio.hpp>
#include <boost/atomic.hpp>
#include <boost/thread.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

#include "define.hpp"

class Coord_Process {
  private:
    boost::atomic_int saved_files;
    float dis_farest_point;                            // 成像点到圆心的距离
    unsigned long long pre_malloc_points_per_triangle; // 预先用来保存单个三角形点坐标的空间大小
    uint16_t point_sum_per_line;                       // 成像点每行的个数，图为正方形
    float coord_step;                                  // 当前成像分辨率下坐标标准方向上的步进
    void initialize(uint16_t num);
    void calc_ele_coords();
    void calc_all_triangles();
    inline float change_coord_to_int(float num, float step);
    void calc_sample_points_coords_by_ele_snd_id(uint16_t triangle_id);
    void save_sample_points_in_triangle(uint16_t ele_snd_id);

  public:
    float ele_coord_array_x[ELE_NO], ele_coord_array_y[ELE_NO]; //振元坐标
    float *triangle_vertex_x, *triangle_vertex_y;               // 所有三角形顶点坐标
    unsigned long long temp_sample_points_coords_cnt[ELE_NO];        //每个发射源扫描区的点数
    uint16_t **temp_sample_points_coords;              // 二维数组[ELE_NO][采样点数]
    void calc_all_sample_points_coords(int thread_sum);         // 多线程计算所有扫描区中，点的坐标
    void save_all_triangles(int thread_sum);                    // 多线程保存文件
    void print_ele_coords();
    void print_triangles_vertex();
    void print_sample_points_info();
    Coord_Process(uint16_t num);
    ~Coord_Process();
};

Coord_Process::Coord_Process(uint16_t num) { initialize(num); }

// num 是成像分辨率单维点数
void Coord_Process::initialize(uint16_t num) {
    dis_farest_point = RADIUS - 0.01; // 距离探头 10mm 的不成像
    point_sum_per_line = num;
    coord_step = IMAGE_WIDTH / (point_sum_per_line - 1);
    pre_malloc_points_per_triangle =
        (unsigned long long)(num * num * RADIUS * RADIUS * tan(PI / 18) / IMAGE_WIDTH / IMAGE_WIDTH);
    printf("Points per triangle: %llu\nPre malloc memory: %.2fGB\n", pre_malloc_points_per_triangle,
           pre_malloc_points_per_triangle * ELE_NO * 2 * 2 / 1024.0 / 1024.0 / 1024.0);
    calc_ele_coords();
    triangle_vertex_x = (float *)calloc(3 * sizeof(float), ELE_NO);
    triangle_vertex_y = (float *)calloc(3 * sizeof(float), ELE_NO);
    temp_sample_points_coords = new uint16_t *[ELE_NO];
    for (uint16_t ele_snd_id = 0; ele_snd_id < ELE_NO; ele_snd_id++) {
        temp_sample_points_coords[ele_snd_id] =
            (uint16_t *)calloc(2 * sizeof(uint16_t), pre_malloc_points_per_triangle);
    }
    calc_all_triangles();
}

Coord_Process::~Coord_Process() {
    free(triangle_vertex_x);
    free(triangle_vertex_y);
    for (uint16_t index = 0; index < point_sum_per_line; index++) {
        free(temp_sample_points_coords[index]);
    }
}

// 计算所有发射源的坐标
void Coord_Process::calc_ele_coords() {
    float raduis = RADIUS;
    float ele_angle_offset = (2 * PI * 43.4695 / (256 - 1)) / 360; //阵元间隔角度
    float start_ele_angle = 2 * PI * (45 - 43.4695) / 360;         //第一个阵元角度
    /* for (int start_ele_id = 0; start_ele_id < 256; start_ele_id++) {
        ele_coord_array_x[start_ele_id] =
            raduis * cos(start_ele_angle + start_ele_id * ele_angle_offset);
        ele_coord_array_y[start_ele_id] =
            -raduis * sin(start_ele_angle + start_ele_id * ele_angle_offset);
    } */
    // 把上面的 for 循环融入到下面会产生误差，但是上面的结果不符合可90度旋转原理
    for (int start_ele_id = 0; start_ele_id < ELE_NO; start_ele_id++) {
        ele_coord_array_x[start_ele_id] = raduis * cos(start_ele_angle + (start_ele_id % 256) * ele_angle_offset +
                                                       (int)(start_ele_id / 256) * PI / 4);
        ele_coord_array_y[start_ele_id] = -raduis * sin(start_ele_angle + (start_ele_id % 256) * ele_angle_offset +
                                                        (int)(start_ele_id / 256) * PI / 4);
    }
}

// 计算所有发射源扫描区(三角形)的三点坐标
void Coord_Process::calc_all_triangles() {
    float radius = RADIUS;
    float temp_point[3][2];   // 3个点，每个点2个坐标值
    uint16_t high_id, low_id; // 最高点和最低点的 id
    for (uint16_t start_ele_id = 0; start_ele_id < ELE_NO; start_ele_id++) {
        temp_point[0][0] = ele_coord_array_x[start_ele_id];
        temp_point[0][1] = ele_coord_array_y[start_ele_id];
        temp_point[1][0] = sin(PI / 18) * radius *
                           sqrtf(ele_coord_array_y[start_ele_id] * ele_coord_array_y[start_ele_id] / (radius * radius));
        temp_point[1][1] = -(temp_point[0][0] * temp_point[1][0]) / temp_point[0][1];
        temp_point[2][0] = -temp_point[1][0];
        temp_point[2][1] = -temp_point[1][1];
        // 排序

        for (uint16_t index = 0; index < 3; index++) {
            if (temp_point[index][1] == std::max(std::max(temp_point[0][1], temp_point[1][1]), temp_point[2][1])) {
                high_id = index;
            }
        }
        for (uint16_t index = 0; index < 3; index++) {
            if (temp_point[index][1] == std::min(std::min(temp_point[0][1], temp_point[1][1]), temp_point[2][1])) {
                low_id = index;
            }
        }

        triangle_vertex_x[3 * start_ele_id] = temp_point[high_id][0];
        triangle_vertex_y[3 * start_ele_id] = temp_point[high_id][1];
        triangle_vertex_x[3 * start_ele_id + 1] = temp_point[3 - high_id - low_id][0];
        triangle_vertex_y[3 * start_ele_id + 1] = temp_point[3 - high_id - low_id][1];
        triangle_vertex_x[3 * start_ele_id + 2] = temp_point[low_id][0];
        triangle_vertex_y[3 * start_ele_id + 2] = temp_point[low_id][1];
    }
    return;
}

// 把三角形三点坐标转移到整数空间
inline float Coord_Process::change_coord_to_int(float num, float step) {
    return (num + (point_sum_per_line - 1) / 2 * step) / step;
}

// 扫描单个三角形中的点
void Coord_Process::calc_sample_points_coords_by_ele_snd_id(uint16_t triangle_id) {
    float high_point_x = triangle_vertex_x[3 * triangle_id];
    float mid_point_x = triangle_vertex_x[3 * triangle_id + 1];
    float low_point_x = triangle_vertex_x[3 * triangle_id + 2];
    float high_point_y = triangle_vertex_y[3 * triangle_id];
    float mid_point_y = triangle_vertex_y[3 * triangle_id + 1];
    float low_point_y = triangle_vertex_y[3 * triangle_id + 2];
    high_point_x = change_coord_to_int(high_point_x, coord_step);
    mid_point_x = change_coord_to_int(mid_point_x, coord_step);
    low_point_x = change_coord_to_int(low_point_x, coord_step);
    high_point_y = change_coord_to_int(high_point_y, coord_step);
    mid_point_y = change_coord_to_int(mid_point_y, coord_step);
    low_point_y = change_coord_to_int(low_point_y, coord_step);

    unsigned long long sample_points_sum = 0;

    float temp1, temp2;
    int min_horizontal_id, max_horizontal_id;
    for (int row_id = std::max((int)(low_point_y + 1), 0); row_id <= std::min((int)mid_point_y, point_sum_per_line - 1);
         row_id++) {
        temp1 = ((float)row_id - low_point_y) * (mid_point_x - low_point_x) / (mid_point_y - low_point_y) + low_point_x;
        temp2 =
            ((float)row_id - low_point_y) * (high_point_x - low_point_x) / (high_point_y - low_point_y) + low_point_x;
        min_horizontal_id = (int)(std::min(temp1, temp2) + 1);
        max_horizontal_id = (int)(std::max(temp1, temp2));
        min_horizontal_id = std::max(min_horizontal_id, 0);
        min_horizontal_id = std::min(min_horizontal_id, point_sum_per_line - 1);
        max_horizontal_id = std::min(max_horizontal_id, point_sum_per_line - 1);
        max_horizontal_id = std::max(max_horizontal_id, 0);
        for (uint16_t index = min_horizontal_id; index <= max_horizontal_id; index++) {
            if (sqrtf(((float)index - point_sum_per_line / 2) * ((float)index - point_sum_per_line / 2) +
                      ((float)row_id - point_sum_per_line / 2) * ((float)row_id - point_sum_per_line / 2)) *
                    coord_step >
                dis_farest_point) {
                continue;
            }
            temp_sample_points_coords[triangle_id][2 * sample_points_sum] = (uint16_t)index;
            temp_sample_points_coords[triangle_id][2 * sample_points_sum + 1] = (uint16_t)row_id;
            sample_points_sum++;
        }
    }

    for (int row_id = std::max((int)(mid_point_y + 1), 0);
         row_id <= std::min((int)high_point_y, point_sum_per_line - 1); row_id++) {
        temp1 =
            ((float)row_id - high_point_y) * (high_point_x - mid_point_x) / (high_point_y - mid_point_y) + high_point_x;
        temp2 =
            ((float)row_id - low_point_y) * (high_point_x - low_point_x) / (high_point_y - low_point_y) + low_point_x;
        min_horizontal_id = (int)(std::min(temp1, temp2) + 1);
        max_horizontal_id = (int)(std::max(temp1, temp2));
        min_horizontal_id = std::max(min_horizontal_id, 0);
        min_horizontal_id = std::min(min_horizontal_id, point_sum_per_line - 1);
        max_horizontal_id = std::min(max_horizontal_id, point_sum_per_line - 1);
        max_horizontal_id = std::max(max_horizontal_id, 0);
        for (uint16_t index = min_horizontal_id; index <= max_horizontal_id; index++) {
            if (sqrtf(((float)index - point_sum_per_line / 2.0) * ((float)index - point_sum_per_line / 2.0) +
                      ((float)row_id - point_sum_per_line / 2.0) * ((float)row_id - point_sum_per_line / 2.0)) *
                    coord_step >
                dis_farest_point) {
                continue;
            }
            temp_sample_points_coords[triangle_id][2 * sample_points_sum] = (uint16_t)index;
            temp_sample_points_coords[triangle_id][2 * sample_points_sum + 1] = (uint16_t)row_id;
            sample_points_sum++;
        }
    }
    temp_sample_points_coords_cnt[triangle_id] = sample_points_sum;
}

// 多线程扫描所有的(2048)三角形，thread_sum 通过进程参数设置
void Coord_Process::calc_all_sample_points_coords(int thread_sum) {
    boost::asio::thread_pool thread_pool_calc(thread_sum);
    for (uint16_t ele_snd_id = 0; ele_snd_id < ELE_NO; ele_snd_id++) {
        boost::asio::post(thread_pool_calc,
                          boost::bind(&Coord_Process::calc_sample_points_coords_by_ele_snd_id, this, ele_snd_id));
    }
    thread_pool_calc.join();

    // for (uint16_t ele_snd_id = 0; ele_snd_id < ELE_NO; ele_snd_id++) {
    //     calc_sample_points_coords_by_ele_snd_id(ele_snd_id);
    // }
}

// 打印所有探头的坐标
void Coord_Process::print_ele_coords() {
    for (uint16_t index = 0; index < ELE_NO; index++) {
        printf("[%04d] = %10.8f, %10.8f; [%04d] = %10.8f, %10.8f; [%04d] = %10.8f, %10.8f; [%04d] = %10.8f, %10.8f;\n",
               index, ele_coord_array_x[index], ele_coord_array_y[index], index + ELE_NO / 4,
               ele_coord_array_x[index + ELE_NO / 4], ele_coord_array_y[index + ELE_NO / 4], index + ELE_NO * 2 / 4,
               ele_coord_array_x[index + ELE_NO * 2 / 4], ele_coord_array_y[index + ELE_NO * 2 / 4],
               index + ELE_NO * 3 / 4, ele_coord_array_x[index + ELE_NO * 3 / 4],
               ele_coord_array_y[index + ELE_NO * 3 / 4]);
    }
    return;
}

void Coord_Process::print_triangles_vertex() {
    for (uint16_t triangle_id = 0; triangle_id < ELE_NO; triangle_id++) {
        if (triangle_id % 64 == 0) {
            printf(
                "[[%10.8f, %10.8f, %10.8f, %10.8f],[%10.8f, %10.8f, %10.8f, "
                "%10.8f]]\n",
                triangle_vertex_x[3 * triangle_id], triangle_vertex_x[3 * triangle_id + 1],
                triangle_vertex_x[3 * triangle_id + 2], triangle_vertex_x[3 * triangle_id],
                triangle_vertex_y[3 * triangle_id], triangle_vertex_y[3 * triangle_id + 1],
                triangle_vertex_y[3 * triangle_id + 2], triangle_vertex_y[3 * triangle_id]);
        }
    }
}

void Coord_Process::print_sample_points_info() {
    unsigned long long point_sum = 0;
    for (uint16_t index = 0; index < ELE_NO; index++) {
        printf("Tri_%04u :: Sample point sum = %llu\n", index, temp_sample_points_coords_cnt[index]);
        point_sum += temp_sample_points_coords_cnt[index];
    }
}

void Coord_Process::save_sample_points_in_triangle(uint16_t ele_snd_id) {
    char *file_name = new char[17];
    sprintf(file_name, "Tri/Tri_%04d.txt", ele_snd_id);
    std::ofstream f_stream(file_name, std::fstream::out);
    std::string content = "";
    for (unsigned long long index = 0; index < temp_sample_points_coords_cnt[ele_snd_id]; index++) {
        content += std::to_string(temp_sample_points_coords[ele_snd_id][2 * index]) + "," +
                   std::to_string(temp_sample_points_coords[ele_snd_id][2 * index + 1]) + "\n";
    }
    f_stream << content;
    f_stream.close();
    saved_files++;
}

void Coord_Process::save_all_triangles(int thread_sum) {
    boost::asio::thread_pool thread_pool_save(40);
    saved_files = 0;
    for (uint16_t ele_snd_id = 0; ele_snd_id < ELE_NO; ele_snd_id++) {
        boost::asio::post(thread_pool_save,
                          boost::bind(&Coord_Process::save_sample_points_in_triangle, this, ele_snd_id));
    }
    while (saved_files < ELE_NO) {
        printf("\r%06.3f%%", (float)saved_files / ELE_NO * 100);
        fflush(stdout);
        sleep(1);
    }
    printf("\r%06.3f%%\n", (float)saved_files / ELE_NO * 100);
}
