// SPDX-License-Identifier: GPL-2.0-only

#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_int.h>

typedef ap_axiu<16, 1, 0, 0> axis_in_t;
typedef ap_axiu<32, 1, 0, 0> axis_out_t;


extern "C" {

void raw10_to_y8_pack(
    hls::stream<axis_in_t>& s_axis,
    hls::stream<axis_out_t>& m_axis
) {
#pragma HLS INTERFACE axis port=s_axis
#pragma HLS INTERFACE axis port=m_axis
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS PIPELINE II=1

    // 8-bit画素を4個まとめて32-bitにするための保持領域
    static ap_uint<32> pack = 0;

    /*
     * 現在packに格納済みの位置
     *
     * 0, 1, 2, 3
     */
    static ap_uint<2> count = 0;

    /*
     * 入力TUSERを受け取ってから、
     * 最初の出力beatまで保持する。
     */
    static ap_uint<1> sof_pending = 0;

    axis_in_t in;
    axis_out_t out;

    /*
     * RAW10 1画素を受信
     */
    s_axis.read(in);

    /*
     * TUSERはフレーム先頭を表す。
     *
     * 新しいフレーム開始時は、
     * 途中のパック状態を破棄して初期化する。
     */
    if (in.user) {
        sof_pending = 1;
        count = 0;
        pack = 0;
    }

    /*
     * RAW10はdata[9:0]に格納されている前提。
     *
     * 上位8 bitを取り出してY8とする。
     */
    const ap_uint<10> raw10 =
        in.data.range(9, 0);

    const ap_uint<8> y8 =
        raw10.range(9, 2);

    /*
     * 現在のY8画素を32-bit packへ格納する。
     */
    pack.range(
        8 * count + 7,
        8 * count
    ) = y8;

    /*
     * 4画素分そろったら1 beat出力する。
     */
    if (count == 3) {
        out.data = pack;
        out.keep = 0xF;
        out.strb = 0xF;

        /*
         * 入力TUSERは1画素目で来るが、
         * 出力は4画素目で初めて生成されるため、
         * sof_pendingを出力する。
         */
        out.user = sof_pending;

        /*
         * 重要:
         *
         * 入力の最終画素に付いたTLASTを、
         * その4画素を含む出力beatへ引き継ぐ。
         *
         * 2464画素および3280画素はどちらも
         * 4の倍数なので、行末画素では必ずcount==3となる。
         */
        out.last = in.last;

        m_axis.write(out);

        /*
         * 次の4画素に備えて初期化
         */
        sof_pending = 0;
        pack = 0;
        count = 0;
    }
    else {
        count++;
    }
}

} // extern "C"