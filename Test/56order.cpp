// -----------------------------------------------------------------------------
// Filename:    56head.cpp
// Revision:    None
// Date:        2018/09/02 - 08:14
// Author:      Haixiang HOU
// Email:       hexid26@outlook.com
// Website:     [NULL]
// Notes:       [NULL]
// -----------------------------------------------------------------------------
// Copyright:   2018 (c) Haixiang
// License:     GPL
// -----------------------------------------------------------------------------
// Version [1.0]
// 56 个数分别以4个一组，7个一组做高低位反序

#include <iostream>

void swap(uint8_t *a, uint8_t *b)
{
    uint8_t tmp = *a;
    *a = *b;
    *b = tmp;
}

void printa(uint8_t *ptr)
{

    for (size_t i = 0; i < 56; i++)
    {
        printf("%d, ", ptr[i]);
    }
    printf("\n");
}

void printhex(uint8_t *ptr)
{

    for (size_t i = 0; i < 56; i++)
    {
        printf("%02x ", ptr[i]);
    }
    printf("\n");
}

int main(int argc, char const *argv[])
{
    uint8_t a[56] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55};

    uint8_t value[56] = {0x10, 0x02, 0xff, 0xee, 0xf9, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x7f, 0x80, 0x03, 0x00, 0x1c, 0xe4, 0x00, 0xaf, 0xff, 0xfe, 0x00, 0x08, 0xff, 0x00, 0x0f, 0xfe, 0x0f, 0xf0, 0x00, 0x80, 0x0e, 0xf9, 0x00, 0x4f, 0xff, 0xfe, 0xcf, 0xfe, 0xbf, 0x3f, 0xf6, 0xff, 0xd7, 0x24, 0x00, 0xd0, 0x03, 0x02, 0xbf, 0xf8, 0x00, 0xff, 0xe3, 0xff, 0x90};

    for (size_t i = 0; i < 14; i++)
    {
        swap(&a[i * 4], &a[i * 4 + 3]);
        swap(&a[i * 4 + 1], &a[i * 4 + 2]);
    }

    for (size_t i = 0; i < 8; i++)
    {
        swap(&a[i * 7], &a[i * 7 + 6]);
        swap(&a[i * 7 + 1], &a[i * 7 + 5]);
        swap(&a[i * 7 + 2], &a[i * 7 + 4]);
    }
    printa(&a[0]);

    printhex(&value[0]);
    for (size_t i = 0; i < 14; i++)
    {
        swap(&value[i * 4], &value[i * 4 + 3]);
        swap(&value[i * 4 + 1], &value[i * 4 + 2]);
    }

    for (size_t i = 0; i < 8; i++)
    {
        swap(&value[i * 7], &value[i * 7 + 6]);
        swap(&value[i * 7 + 1], &value[i * 7 + 5]);
        swap(&value[i * 7 + 2], &value[i * 7 + 4]);
    }
    printhex(&value[0]);

    int16_t test = 8252;
    printf("%d\n", (int16_t)(test << 2));

    return 0;
}
