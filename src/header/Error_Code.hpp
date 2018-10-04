// -----------------------------------------------------------------------------
// Filename:    Error_Code.hpp
// Revision:    None
// Date:        2018/09/30 - 02:44
// Author:      Haixiang HOU
// Email:       hexid26@outlook.com
// Website:     [NULL]
// Notes:       [NULL]
// -----------------------------------------------------------------------------
// Copyright:   2018 (c) Haixiang
// License:     GPL
// -----------------------------------------------------------------------------
// Version [1.0]
// 错误码

class Error_Code {
  public:
    enum t_type { main_argc_err = 100, mem_malloc_error = 101 };

    static const char *Description(const t_type &pError) {
        switch (pError) {
        case 100:
            return "ERROR :: main 函数参数配置错误\n";
        case 101:
            return "ERROR :: 内存申请失败\n";
        default:
            break;
        }
    }
};