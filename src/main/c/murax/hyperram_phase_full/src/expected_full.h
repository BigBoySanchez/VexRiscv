#ifndef EXPECTED_FULL_H
#define EXPECTED_FULL_H

#include <stdint.h>

const int32_t EXPECTED_LOGITS[10] = {
    -10517, -52, -2758, -4096, 3954, 5469, -747, -103, 3491, 4913
};

const int32_t EXPECTED_CLASS = 5;

const uint32_t EXPECTED_HASHES[56] = {
    0x000b5a22, // conv1
    0x00118597, // layer1_0
    0x0014ec61, // layer1_1
    0x0016fa4d, // layer1_2
    0x00184d58, // layer1_3
    0x0016fbfd, // layer1_4
    0x00177787, // layer1_5
    0x0016cef7, // layer1_6
    0x0016025b, // layer1_7
    0x00160ad8, // layer1_8
    0x001609e4, // layer1_9
    0x00184fbf, // layer1_10
    0x00187c4f, // layer1_11
    0x0018e907, // layer1_12
    0x00181748, // layer1_13
    0x0017631d, // layer1_14
    0x0016c5eb, // layer1_15
    0x0017ba0f, // layer1_16
    0x00180a8f, // layer1_17
    0x00081c5e, // layer2_0
    0x000835b8, // layer2_1
    0x0007bcce, // layer2_2
    0x000739a5, // layer2_3
    0x0006b640, // layer2_4
    0x0006ad7d, // layer2_5
    0x0006939c, // layer2_6
    0x00061fba, // layer2_7
    0x0005a581, // layer2_8
    0x0005e459, // layer2_9
    0x0005d6cb, // layer2_10
    0x00063987, // layer2_11
    0x00066115, // layer2_12
    0x00067db7, // layer2_13
    0x0006aa3d, // layer2_14
    0x0006fcfe, // layer2_15
    0x00069aa2, // layer2_16
    0x0006c581, // layer2_17
    0x00026c25, // layer3_0
    0x0001d239, // layer3_1
    0x000178f0, // layer3_2
    0x000131a2, // layer3_3
    0x000161d3, // layer3_4
    0x000108f9, // layer3_5
    0x0000eb92, // layer3_6
    0x0000a0da, // layer3_7
    0x0000b111, // layer3_8
    0x0000975a, // layer3_9
    0x000071b2, // layer3_10
    0x0000a625, // layer3_11
    0x0000a842, // layer3_12
    0x0000893a, // layer3_13
    0x0000c387, // layer3_14
    0x0000aedf, // layer3_15
    0x0000b668, // layer3_16
    0x00024342, // layer3_17
    0x000008f5, // pool
};


#endif
